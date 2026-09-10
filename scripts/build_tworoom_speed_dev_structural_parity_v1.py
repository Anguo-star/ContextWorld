#!/usr/bin/env python3
"""Build the Public-Test-structured Development speed catalogs.

Replicates the frozen ``tworoom_history3_speed_multistep_extrap_v5`` structure
(four tracks, per-track reference-speed groups, three/four history
conditions per query, disjoint eval-seed partitions, identical audits) on a
fresh geometry bank, and adds the isolation step the Test build did not
need: candidate geometries whose ``reset_state`` appears in any frozen
Public Test catalog are removed before selection, and the finished catalogs
are audited so that no ``static_query_id``-level identity or rendered
``query_pixels_sha256`` is shared with the Public Test.

Reuses the frozen builder internals from
``scripts.build_tworoom_speed_multistep_catalogs`` (``_build_track``,
``_select_unique_reset_geometries``, ``_speed_support_audit``) without
changing their default behaviour; only this new entry point adds the
isolation filter, the retry-on-collision loop, and its own report.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contextworld.evaluation.icl_catalog import _array_sha256, _simulate_blocks
from contextworld.evaluation.icl_sensitive import (
    generate_same_room_geometries,
    sha256_file,
)
from contextworld.paths import resolve_contextworld_path
from contextworld.synthesis.manifest import write_json
from contextworld.synthesis.stablewm import load_stable_worldmodel

from scripts.build_tworoom_speed_multistep_catalogs import (
    PINNED_STABLEWM,
    _build_track,
    _select_unique_reset_geometries,
    _speed_support_audit,
)

TEST_CATALOG_DIR_DEFAULT = Path(
    "/opt/huawei/explorer-env/dataset/ag_data/data/world_model/"
    "ContextWorld-v1-full/artifacts/evaluation/history3/"
    "speed_multistep_extrap_v5/catalogs"
)
TEST_TRACK_CATALOGS = (
    "seen_for_multi.json",
    "unseen_interpolation.json",
    "extrapolation_low.json",
    "extrapolation_high.json",
)


def _load_test_isolation_sets(
    catalog_dir: Path,
) -> dict[str, Any]:
    resets: set[tuple[float, float]] = set()
    query_hashes: set[str] = set()
    static_banks: list[list[dict[str, Any]]] = []
    per_catalog = {}
    for name in TEST_TRACK_CATALOGS:
        path = catalog_dir / name
        catalog = json.loads(path.read_text(encoding="utf-8"))
        bank = catalog["geometry_bank"]
        static_banks.append(bank)
        resets.update(
            tuple(map(float, geometry["reset_state"])) for geometry in bank
        )
        hashes = {
            str(bundle["query_pixels_sha256"]) for bundle in catalog["bundles"]
        }
        query_hashes |= hashes
        per_catalog[name] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "geometries": len(bank),
            "bundles": len(catalog["bundles"]),
            "query_hashes": len(hashes),
        }
    banks_identical = all(
        bank == static_banks[0] for bank in static_banks[1:]
    )
    if not banks_identical:
        raise RuntimeError("Public Test geometry banks differ across tracks")
    return {
        "per_catalog": per_catalog,
        "test_reset_states": resets,
        "test_query_pixel_hashes": query_hashes,
        "banks_identical": banks_identical,
    }


def _render_query_hash(
    geometry: Any, door_position: int, seed: int
) -> str:
    """Render the static query frame of one geometry (door fixed)."""

    rollout = _simulate_blocks(
        {"agent.speed": 3.1, "door.position": int(door_position)},
        np.asarray(geometry.reset_state, dtype=np.float32),
        np.asarray(geometry.goal_state, dtype=np.float32),
        np.zeros((1, 5, 2), dtype=np.float32),
        seed=seed,
    )
    return _array_sha256(rollout["pixels"][0])


def _build_attempt(
    *,
    config: dict[str, Any],
    geometries: list[Any],
    payload_root: Path,
    stable_commit: str,
) -> dict[str, Any]:
    tracks = {}
    static_hashes_by_template: dict[str, dict[str, str]] = {}
    for track_name, track in config["data"]["tracks"].items():
        print(f"[speed-dev] build track {track_name}", flush=True)
        catalog = _build_track(
            config=config,
            track_name=str(track_name),
            track=dict(track),
            geometries=geometries,
            payload_root=payload_root / str(track_name),
            stable_commit=stable_commit,
        )
        catalog_path = resolve_contextworld_path(
            track["catalog"], repo_root=ROOT
        )
        by_template: dict[str, str] = {}
        for bundle in catalog["bundles"]:
            template = str(bundle["template"]["template_id"])
            observed = str(bundle["query_pixels_sha256"])
            previous = by_template.setdefault(template, observed)
            if previous != observed:
                raise RuntimeError(
                    f"{track_name}/{template}: query pixels differ by speed"
                )
        static_hashes_by_template[str(track_name)] = by_template
        tracks[str(track_name)] = {
            "role": str(track["role"]),
            "speeds": [float(value) for value in track["speeds"]],
            "catalog": str(catalog_path),
            "catalog_sha256": sha256_file(catalog_path),
            "summary": catalog["summary"],
        }
    reference = next(iter(static_hashes_by_template.values()))
    cross_track_pass = all(
        hashes == reference for hashes in static_hashes_by_template.values()
    )
    if not cross_track_pass:
        raise RuntimeError("Static query pixels differ across tracks")
    return {
        "tracks": tracks,
        "static_query_hashes": reference,
        "cross_track_static_query_pixels": cross_track_pass,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("status") != (
        "preregistered_before_catalog_generation_and_model_scoring"
    ):
        raise ValueError("Development speed config is not frozen before execution")
    _, stable_repo, stable_commit = load_stable_worldmodel(
        ROOT, args.stablewm_repo, PINNED_STABLEWM
    )
    if stable_commit != config["stable_worldmodel"]["expected_ref"]:
        raise RuntimeError(f"StableWM commit mismatch: {stable_commit}")

    generation = config["data"]["generation"]
    distances = [int(value) for value in generation["distance_bins_px"]]
    training_config = (
        ROOT / config["data"]["source_training_protocol"]
    ).resolve()
    support_audit = _speed_support_audit(config, training_config)

    test_isolation = _load_test_isolation_sets(Path(args.test_catalog_dir))
    artifacts = config["artifacts"]
    payload_root = resolve_contextworld_path(
        artifacts["payload_root"], repo_root=ROOT
    )
    artifacts_root = resolve_contextworld_path(artifacts["root"], repo_root=ROOT)

    expected = int(config["evaluation"]["unique_queries_per_reference_speed"])
    base_geometry_seed = int(generation["geometry_seed"])
    catalog_seed = int(generation["catalog_seed"])
    door = int(generation["door_position"])

    attempts: list[dict[str, Any]] = []
    built: dict[str, Any] | None = None
    for attempt_index in range(int(args.max_attempts)):
        geometry_seed = base_geometry_seed + 1000 * attempt_index
        candidates = generate_same_room_geometries(
            distances=distances,
            variants_per_distance=int(
                generation["candidate_variants_per_distance"]
            ),
            geometry_seed=geometry_seed,
        )
        test_resets = test_isolation["test_reset_states"]
        dropped = [
            geometry
            for geometry in candidates
            if tuple(map(float, geometry.reset_state)) in test_resets
        ]
        filtered = [
            geometry
            for geometry in candidates
            if tuple(map(float, geometry.reset_state)) not in test_resets
        ]
        print(
            f"[speed-dev] attempt {attempt_index}: {len(candidates)} "
            f"candidates, {len(dropped)} dropped by Test reset-state "
            f"isolation",
            flush=True,
        )
        geometries = _select_unique_reset_geometries(
            filtered,
            distances=distances,
            variants_per_distance=int(generation["variants_per_distance"]),
        )
        if len(geometries) != expected:
            raise RuntimeError(
                f"Expected {expected} geometries, got {len(geometries)}"
            )
        # Pre-build pixel-isolation precheck: render every candidate query
        # frame and drop the whole attempt on any Test pixel collision.
        pixel_collisions = []
        for geometry in geometries:
            seed = int(
                np.random.SeedSequence(
                    [
                        catalog_seed,
                        int(geometry.distance_bin),
                        int(geometry.geometry_variant),
                    ]
                ).generate_state(1)[0]
            )
            query_hash = _render_query_hash(geometry, door, seed)
            if query_hash in test_isolation["test_query_pixel_hashes"]:
                pixel_collisions.append(
                    {
                        "template_id": geometry.template_id,
                        "reset_state": list(map(float, geometry.reset_state)),
                        "query_pixels_sha256": query_hash,
                    }
                )
        attempt_record = {
            "attempt_index": attempt_index,
            "geometry_seed": geometry_seed,
            "candidate_geometries": len(candidates),
            "dropped_by_test_reset_state": len(dropped),
            "dropped_reset_states": [
                list(map(float, geometry.reset_state)) for geometry in dropped
            ],
            "selected_geometries": len(geometries),
            "precheck_pixel_collisions": pixel_collisions,
        }
        attempts.append(attempt_record)
        if pixel_collisions:
            print(
                f"[speed-dev] attempt {attempt_index}: "
                f"{len(pixel_collisions)} query-pixel collisions with the "
                "Public Test; retrying with a new geometry seed",
                flush=True,
            )
            continue

        if artifacts_root.exists():
            shutil.rmtree(artifacts_root)
        result = _build_attempt(
            config=config,
            geometries=geometries,
            payload_root=payload_root,
            stable_commit=stable_commit,
        )

        # Authoritative post-build isolation audit on the built catalogs.
        dev_static_hashes = set(result["static_query_hashes"].values())
        dev_resets = {
            tuple(map(float, geometry.reset_state)) for geometry in geometries
        }
        reset_intersection = sorted(
            map(list, dev_resets & test_isolation["test_reset_states"])
        )
        pixel_intersection = sorted(
            dev_static_hashes & test_isolation["test_query_pixel_hashes"]
        )
        attempt_record["post_build_isolation"] = {
            "reset_state_intersection": reset_intersection,
            "query_pixel_hash_intersection": pixel_intersection,
            "passed": not reset_intersection and not pixel_intersection,
        }
        if not attempt_record["post_build_isolation"]["passed"]:
            print(
                f"[speed-dev] attempt {attempt_index} failed the post-build "
                "isolation audit; retrying with a new geometry seed",
                flush=True,
            )
            continue
        built = result
        built["geometry_seed_used"] = geometry_seed
        built["isolation_attempt_record"] = attempt_record
        break

    if built is None:
        raise RuntimeError(
            "No geometry-seed attempt produced a Test-isolated Development "
            "catalog bank"
        )

    expected_trajectories = sum(
        int(row["condition_trajectories"])
        for row in config["evaluation"]["matrix_by_track"].values()
    )
    count_checks = {
        name: (
            row["summary"]["base_geometries"] == expected
            and row["summary"]["bundles"]
            == expected * len(row["speeds"])
            and row["summary"]["matrix_cells"] == len(row["speeds"]) ** 2
        )
        for name, row in built["tracks"].items()
    }
    if not all(count_checks.values()):
        raise RuntimeError(f"Catalog count audit failed: {count_checks}")

    report = {
        "schema_version": 1,
        "benchmark": config["benchmark"],
        "status": "passed",
        "purpose": (
            "Development replica of tworoom_history3_speed_multistep_extrap_v5 "
            "with identical structure and Test-isolated data rows"
        ),
        "config": {
            "path": str(config_path),
            "sha256": sha256_file(config_path),
        },
        "stable_worldmodel": {"repo": str(stable_repo), "commit": stable_commit},
        "speed_support_audit": support_audit,
        "test_isolation_inputs": test_isolation["per_catalog"],
        "geometry_isolation_attempts": attempts,
        "geometry_seed_used": built["geometry_seed_used"],
        "isolation_audit": {
            "test_reset_state_count": len(
                test_isolation["test_reset_states"]
            ),
            "test_query_pixel_hash_count": len(
                test_isolation["test_query_pixel_hashes"]
            ),
            "reset_state_intersection": built["isolation_attempt_record"][
                "post_build_isolation"
            ]["reset_state_intersection"],
            "query_pixel_hash_intersection": built["isolation_attempt_record"][
                "post_build_isolation"
            ]["query_pixel_hash_intersection"],
            "passed": built["isolation_attempt_record"]["post_build_isolation"][
                "passed"
            ],
        },
        "tracks": built["tracks"],
        "cross_track_audit": {
            "same_geometry_bank": True,
            "static_query_pixels_identical": built[
                "cross_track_static_query_pixels"
            ],
            "templates": len(built["static_query_hashes"]),
        },
        "count_audit": {
            "checks": count_checks,
            "condition_trajectories_per_checkpoint_all_tracks": (
                expected_trajectories
            ),
            "expected_unique_static_queries_per_track": expected,
            "passed": all(count_checks.values()),
        },
        "runtime_seconds": round(time.monotonic() - started, 1),
    }
    build_report_path = resolve_contextworld_path(
        artifacts["build_report"], repo_root=ROOT
    )
    write_json(build_report_path, report)
    return {**report, "report": str(build_report_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=(
            ROOT
            / "configs/benchmark/tworoom_speed_dev_structural_parity_v1.yaml"
        ),
    )
    parser.add_argument("--stablewm-repo", default="../stable-worldmodel")
    parser.add_argument(
        "--test-catalog-dir", type=Path, default=TEST_CATALOG_DIR_DEFAULT
    )
    parser.add_argument("--max-attempts", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(
        json.dumps(
            {
                "status": result["status"],
                "report": result["report"],
                "isolation_passed": result["isolation_audit"]["passed"],
                "geometry_seed_used": result["geometry_seed_used"],
                "tracks": {
                    name: row["summary"]
                    for name, row in result["tracks"].items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
