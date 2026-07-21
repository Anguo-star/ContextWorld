#!/usr/bin/env python3
"""Build independent frozen queries for the speed next-latent benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contextworld.evaluation.icl_catalog import validate_context_query_catalog
from contextworld.evaluation.icl_sensitive import (
    generate_same_room_geometries,
    sha256_file,
)
from contextworld.evaluation.speed_cube import build_speed_cube_catalog
from contextworld.paths import resolve_contextworld_path
from contextworld.synthesis.manifest import write_json
from contextworld.synthesis.stablewm import load_stable_worldmodel


PINNED_STABLEWM = "5864b74980f6ed328fd0045e777b3865962eff43"


def _assign_eval_seeds(
    catalog: dict[str, Any],
    *,
    eval_seeds: list[int],
    per_seed: int,
    assignment_seed: int,
) -> dict[str, Any]:
    """Partition each reference-speed row into disjoint eval-seed sets."""

    by_speed: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for bundle in catalog["bundles"]:
        by_speed[float(bundle["query_factors"]["agent.speed"])].append(bundle)
    expected = len(eval_seeds) * int(per_seed)
    assignments: dict[str, tuple[int, int]] = {}
    for speed, bundles in sorted(by_speed.items()):
        if len(bundles) != expected:
            raise RuntimeError(
                f"Reference speed {speed:g}: expected {expected} unique "
                f"queries, got {len(bundles)}"
            )
        # The same static query keeps the same eval seed at every reference
        # speed, which makes all 3x3 comparisons exactly paired.
        rng = np.random.default_rng(int(assignment_seed))
        ordered = sorted(bundles, key=lambda row: row["static_query_id"])
        permutation = rng.permutation(len(ordered))
        for seed_index, eval_seed in enumerate(eval_seeds):
            selected = permutation[
                seed_index * per_seed : (seed_index + 1) * per_seed
            ]
            for evaluation_index, query_index in enumerate(selected):
                bundle = ordered[int(query_index)]
                static_id = str(bundle["static_query_id"])
                previous = assignments.setdefault(
                    static_id, (int(eval_seed), int(evaluation_index))
                )
                if previous != (int(eval_seed), int(evaluation_index)):
                    raise RuntimeError(
                        f"Static query assignment changed by speed: {static_id}"
                    )
                bundle["eval_seed"] = int(eval_seed)
                bundle["evaluation_index"] = int(evaluation_index)
        observed = {
            (int(row["eval_seed"]), str(row["static_query_id"]))
            for row in bundles
        }
        if len(observed) != expected:
            raise RuntimeError(f"Non-unique assignment at speed {speed:g}")
    catalog["protocol"]["eval_seed_assignment_is_disjoint"] = True
    catalog["summary"]["eval_seeds"] = list(eval_seeds)
    catalog["summary"]["unique_queries_per_reference_speed_per_seed"] = int(
        per_seed
    )
    catalog["summary"]["unique_queries_per_reference_speed"] = expected
    catalog["summary"]["all_eval_seed_queries_are_disjoint"] = True
    return catalog


def run(args: argparse.Namespace) -> dict[str, Any]:
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    expected_status = (
        "preregistered_before_independent_catalog_generation_and_scoring"
    )
    if config.get("status") != expected_status:
        raise ValueError("Config is not frozen before catalog generation")
    swm = config["stable_worldmodel"]
    _, stable_repo, stable_commit = load_stable_worldmodel(
        ROOT, args.stablewm_repo, args.stablewm_ref
    )
    if stable_commit != swm["expected_ref"]:
        raise RuntimeError(f"StableWM commit mismatch: {stable_commit}")

    generation = config["data"]["generation"]
    evaluation = config["evaluation"]
    distances = [int(value) for value in generation["distance_bins_px"]]
    variants = int(generation["variants_per_distance"])
    geometries = generate_same_room_geometries(
        distances=distances,
        variants_per_distance=variants,
        geometry_seed=int(generation["geometry_seed"]),
    )
    expected_unique = int(evaluation["unique_queries_per_reference_speed"])
    if len(geometries) != expected_unique:
        raise RuntimeError(
            f"Expected {expected_unique} geometries, got {len(geometries)}"
        )

    payload_root = resolve_contextworld_path(
        config["artifacts"]["payload_root"], repo_root=ROOT
    )
    tracks: dict[str, Any] = {}
    cross_track_hashes: dict[str, dict[str, str]] = {}
    for track_name, track in config["data"]["tracks"].items():
        catalog_path = resolve_contextworld_path(
            track["catalog"], repo_root=ROOT
        )
        catalog = build_speed_cube_catalog(
            repo_root=ROOT,
            output_catalog=catalog_path,
            payload_root=payload_root / track_name,
            split=str(track["split"]),
            distances=distances,
            variants_per_distance=variants,
            geometry_seed=int(generation["geometry_seed"]),
            catalog_seed=int(generation["catalog_seed"]),
            stable_worldmodel_commit=stable_commit,
            speeds=tuple(float(value) for value in track["speeds"]),
            door_position=int(generation["door_position"]),
            benchmark_name=str(config["benchmark"]),
            track_name=str(track_name),
            geometries_override=geometries,
        )
        catalog = _assign_eval_seeds(
            catalog,
            eval_seeds=[int(value) for value in evaluation["eval_seeds"]],
            per_seed=int(
                evaluation["unique_queries_per_reference_speed_per_seed"]
            ),
            assignment_seed=int(generation["eval_seed_assignment_seed"]),
        )
        catalog_path.write_text(
            json.dumps(catalog, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        validation = validate_context_query_catalog(
            catalog_path,
            repo_root=ROOT,
            replay_simulator=not args.skip_simulator_replay,
            family="speed",
        )
        if not validation["passed"]:
            raise RuntimeError(
                f"{track_name} validation failed: {validation['failures'][:5]}"
            )
        hashes = {
            str(row["template"]["template_id"]): str(
                row["query_pixels_sha256"]
            )
            for row in catalog["bundles"]
        }
        cross_track_hashes[str(track_name)] = hashes
        tracks[str(track_name)] = {
            "catalog": str(catalog_path),
            "catalog_sha256": sha256_file(catalog_path),
            "summary": catalog["summary"],
            "validation": validation,
        }

    first_hashes = next(iter(cross_track_hashes.values()))
    cross_track_pass = all(
        hashes == first_hashes for hashes in cross_track_hashes.values()
    )
    if not cross_track_pass:
        raise RuntimeError("Static query pixels differ across tracks")
    report = {
        "schema_version": 1,
        "benchmark": config["benchmark"],
        "status": "passed",
        "config": {
            "path": str(config_path),
            "sha256": sha256_file(config_path),
        },
        "stable_worldmodel": {
            "repo": str(stable_repo),
            "commit": stable_commit,
        },
        "tracks": tracks,
        "count_audit": {
            "unique_static_queries_per_track": len(first_hashes),
            "unique_queries_per_reference_speed": expected_unique,
            "unique_queries_per_reference_speed_per_seed": int(
                evaluation["unique_queries_per_reference_speed_per_seed"]
            ),
            "eval_seeds": evaluation["eval_seeds"],
            "all_eval_seed_queries_are_disjoint": True,
            "passed": True,
        },
        "cross_track_static_query_pixel_audit": {
            "passed": cross_track_pass,
            "queries": len(first_hashes),
        },
        "online_environment_required_during_model_scoring": False,
    }
    output = resolve_contextworld_path(
        config["artifacts"]["build_report"], repo_root=ROOT
    )
    write_json(output, report)
    return {**report, "report": str(output)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "configs/benchmark/tworoom_speed_next_latent_v4.yaml",
    )
    parser.add_argument("--stablewm-repo", default="../stable-worldmodel")
    parser.add_argument("--stablewm-ref", default=PINNED_STABLEWM)
    parser.add_argument("--skip-simulator-replay", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(
        json.dumps(
            {
                "status": result["status"],
                "report": result["report"],
                "count_audit": result["count_audit"],
            },
            indent=2,
            sort_keys=True,
        )
    )
