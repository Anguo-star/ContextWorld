#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contextworld.evaluation.icl_catalog import (
    validate_context_query_catalog,
)
from contextworld.evaluation.icl_sensitive import (
    generate_same_room_geometries,
    sha256_file,
)
from contextworld.evaluation.speed_cube import build_speed_cube_catalog
from contextworld.paths import resolve_contextworld_path
from contextworld.synthesis.manifest import write_json
from contextworld.synthesis.stablewm import load_stable_worldmodel


PINNED_STABLEWM = "5864b74980f6ed328fd0045e777b3865962eff43"


def _load(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if (
        config["status"]
        != "preregistered_before_catalog_generation_and_scoring"
    ):
        raise ValueError("Speed cube config is not preregistered")
    return config


def _speed_support_audit(config: dict[str, Any]) -> dict[str, Any]:
    source = resolve_contextworld_path(
        config["frozen_scope"]["source_training_protocol"],
        repo_root=ROOT,
    )
    training = yaml.safe_load(source.read_text(encoding="utf-8"))
    support = training["speed_support"]
    original = set(map(float, support["original_train"]))
    train = set(map(float, support["multi_synthetic_train"]))
    monitor = set(map(float, support["training_monitor_only"]))
    calibration = set(map(float, support["planner_calibration"]))
    seen = set(map(float, support["validation_seen_for_multi"]))
    unseen = set(map(float, support["validation_unseen_interpolation"]))
    sealed_test = set(map(float, support["sealed_test_interpolation"]))
    checks = {
        "seen_is_multi_train_subset": seen <= train,
        "unseen_disjoint_original": not unseen & original,
        "unseen_disjoint_multi_train": not unseen & train,
        "unseen_disjoint_monitor": not unseen & monitor,
        "calibration_disjoint_validation_test": not calibration
        & (seen | unseen | sealed_test),
        "test_disjoint_all_development": not sealed_test
        & (original | train | monitor | calibration | seen | unseen),
        "unseen_inside_multi_train_range": (
            min(train) < min(unseen) < max(unseen) < max(train)
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Speed support isolation failed: {checks}")
    return {
        "passed": True,
        "source": str(source),
        "source_sha256": sha256_file(source),
        "sets": {
            "original_train": sorted(original),
            "multi_synthetic_train": sorted(train),
            "training_monitor_only": sorted(monitor),
            "planner_calibration": sorted(calibration),
            "validation_seen_for_multi": sorted(seen),
            "validation_unseen_interpolation": sorted(unseen),
            "sealed_test_interpolation": sorted(sealed_test),
        },
        "checks": checks,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    config_path = args.config.resolve()
    config = _load(config_path)
    frozen = config["frozen_scope"]
    artifacts = config["artifacts"]
    build_report = resolve_contextworld_path(
        artifacts["build_report"], repo_root=ROOT
    )
    score_roots = [
        resolve_contextworld_path(artifacts[name], repo_root=ROOT)
        for name in (
            "physical_transition_root",
            "fixed_candidate_root",
            "closed_loop_root",
        )
    ]
    existing_scores = [
        path
        for root in score_roots
        if root.exists()
        for path in root.rglob("*.json")
    ]
    if existing_scores and not args.allow_existing_scores:
        raise RuntimeError(
            "Refusing to rebuild catalogs after score files exist: "
            f"{existing_scores[:3]}"
        )

    _, stable_repo, stable_commit = load_stable_worldmodel(
        ROOT, args.stablewm_repo, args.stablewm_ref
    )
    if stable_commit != config["stable_worldmodel"]["expected_ref"]:
        raise RuntimeError(f"StableWM commit mismatch: {stable_commit}")
    support_audit = _speed_support_audit(config)

    geometries = generate_same_room_geometries(
        distances=[int(value) for value in frozen["distance_bins_px"]],
        variants_per_distance=int(frozen["variants_per_distance"]),
        geometry_seed=int(frozen["geometry_seed"]),
    )
    tracks: dict[str, Any] = {}
    static_hashes_by_template: dict[str, dict[str, str]] = {}
    for track in frozen["tracks"]:
        name = str(track["name"])
        catalog_path = resolve_contextworld_path(
            artifacts["catalogs"][name], repo_root=ROOT
        )
        payload_root = (
            resolve_contextworld_path(
                artifacts["payload_root"], repo_root=ROOT
            )
            / name
        )
        catalog = build_speed_cube_catalog(
            repo_root=ROOT,
            output_catalog=catalog_path,
            payload_root=payload_root,
            split=str(track["split"]),
            distances=[
                int(value) for value in frozen["distance_bins_px"]
            ],
            variants_per_distance=int(frozen["variants_per_distance"]),
            geometry_seed=int(frozen["geometry_seed"]),
            catalog_seed=int(frozen["catalog_seed"]),
            stable_worldmodel_commit=stable_commit,
            speeds=tuple(float(value) for value in track["speeds"]),
            door_position=int(frozen["door_position"]),
            benchmark_name=str(config["benchmark"]),
            track_name=name,
            geometries_override=geometries,
        )
        validation = validate_context_query_catalog(
            catalog_path,
            repo_root=ROOT,
            replay_simulator=not args.skip_simulator_replay,
            family="speed",
        )
        if not validation["passed"]:
            raise RuntimeError(
                f"{name} catalog validation failed: "
                f"{validation['failures'][:5]}"
            )
        by_template: dict[str, str] = {}
        for bundle in catalog["bundles"]:
            template = str(bundle["template"]["template_id"])
            observed = str(bundle["query_pixels_sha256"])
            previous = by_template.setdefault(template, observed)
            if previous != observed:
                raise RuntimeError(
                    f"{name}/{template}: query pixels differ by speed"
                )
        static_hashes_by_template[name] = by_template
        tracks[name] = {
            "role": track["role"],
            "speeds": [float(value) for value in track["speeds"]],
            "catalog": str(catalog_path),
            "catalog_sha256": sha256_file(catalog_path),
            "summary": catalog["summary"],
            "validation": validation,
        }

    reference_hashes = next(iter(static_hashes_by_template.values()))
    cross_track_static_query_pixels = all(
        hashes == reference_hashes
        for hashes in static_hashes_by_template.values()
    )
    if not cross_track_static_query_pixels:
        raise RuntimeError("Query pixels differ across speed tracks")
    expected_base = int(
        config["formal_eval"]["expected_base_queries_per_query_speed"]
    )
    count_checks = {
        name: (
            row["summary"]["base_geometries"] == expected_base
            and row["summary"]["bundles"] == expected_base * 3
            and row["summary"]["matrix_cells"] == 9
        )
        for name, row in tracks.items()
    }
    if not all(count_checks.values()):
        raise RuntimeError(f"Catalog count audit failed: {count_checks}")

    report = {
        "schema_version": 1,
        "benchmark": config["benchmark"],
        "status": "passed",
        "stage": "catalogs_built_before_model_scoring",
        "config": {
            "path": str(config_path),
            "sha256": sha256_file(config_path),
        },
        "stable_worldmodel": {
            "repo": str(stable_repo),
            "commit": stable_commit,
        },
        "speed_support_audit": support_audit,
        "tracks": tracks,
        "cross_track_audit": {
            "same_geometry_bank": True,
            "static_query_pixels_identical": (
                cross_track_static_query_pixels
            ),
            "templates": len(reference_hashes),
            "passed": True,
        },
        "count_audit": {
            "checks": count_checks,
            "eval_seeds": config["formal_eval"]["eval_seeds"],
            "evaluations_per_matrix_cell_per_seed": config[
                "formal_eval"
            ]["evaluations_per_matrix_cell_per_seed"],
            "evaluations_per_matrix_cell": config["formal_eval"][
                "evaluations_per_matrix_cell"
            ],
            "passed": all(count_checks.values()),
        },
    }
    write_json(build_report, report)
    return {**report, "report": str(build_report)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=(
            ROOT
            / "configs/benchmark/tworoom_speed_cube_eval_v2.yaml"
        ),
    )
    parser.add_argument("--stablewm-repo", default="../stable-worldmodel")
    parser.add_argument("--stablewm-ref", default=PINNED_STABLEWM)
    parser.add_argument("--skip-simulator-replay", action="store_true")
    parser.add_argument("--allow-existing-scores", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(
        json.dumps(
            {
                "status": result["status"],
                "report": result["report"],
                "tracks": {
                    name: row["summary"]
                    for name, row in result["tracks"].items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
