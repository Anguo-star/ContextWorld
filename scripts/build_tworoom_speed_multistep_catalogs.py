#!/usr/bin/env python3
"""Build frozen History-3 speed extrapolation and multi-step catalogs."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contextworld.evaluation.icl_catalog import (
    _array_sha256,
    _canonical_json,
    _jsonable_factors,
    _sha256_bytes,
    _sha256_file,
    _simulate_blocks,
)
from contextworld.evaluation.icl_sensitive import (
    SensitiveGeometry,
    generate_same_room_geometries,
    sha256_file,
)
from contextworld.paths import (
    portable_contextworld_path,
    resolve_contextworld_path,
)
from contextworld.synthesis.manifest import write_json
from contextworld.synthesis.stablewm import load_stable_worldmodel
from scripts.build_tworoom_speed_next_latent_catalogs import (
    _assign_eval_seeds,
)


PINNED_STABLEWM = "5864b74980f6ed328fd0045e777b3865962eff43"
ACTION_BLOCK = 5
HORIZONS = (1, 2, 3, 5)


def _condition_names(count: int) -> list[str]:
    if count == 3:
        return ["history_low", "history_mid", "history_high"]
    if count == 4:
        return [
            "history_low",
            "history_mid_low",
            "history_mid_high",
            "history_high",
        ]
    return [f"history_{index:02d}" for index in range(count)]


def _context_actions(
    geometry: SensitiveGeometry, magnitude: float = 0.5
) -> np.ndarray:
    direction = np.asarray(geometry.context_direction, dtype=np.float32)
    outward = np.repeat(
        (float(magnitude) * direction)[None], ACTION_BLOCK, axis=0
    )
    return np.stack([outward, -outward], axis=0)


def _query_actions(
    geometry: SensitiveGeometry,
    magnitude: float,
    family_index: int | None = None,
) -> tuple[str, np.ndarray]:
    reset = np.asarray(geometry.reset_state, dtype=np.float32)
    sx = 1.0 if reset[0] <= 168.0 else -1.0
    sy = 1.0 if reset[1] <= 112.0 else -1.0
    x = np.asarray([sx, 0.0], dtype=np.float32)
    y = np.asarray([0.0, sy], dtype=np.float32)
    diagonal = (x + y) / np.sqrt(np.float32(2.0))
    family = (
        int(geometry.geometry_variant)
        if family_index is None
        else int(family_index)
    ) % 3
    if family == 0:
        name = "bounded_axis_loop_x_first"
        directions = [x, y, -x, -y, x]
    elif family == 1:
        name = "bounded_axis_loop_y_first"
        directions = [y, x, -y, -x, y]
    else:
        name = "bounded_diagonal_loop"
        directions = [x, diagonal, -x, -diagonal, y]
    blocks = [
        np.repeat(
            (float(magnitude) * direction)[None], ACTION_BLOCK, axis=0
        )
        for direction in directions
    ]
    return name, np.asarray(blocks, dtype=np.float32)


def _free_motion_residual(
    reset_state: tuple[float, float],
    speed: float,
    actions: np.ndarray,
    next_states: np.ndarray,
) -> float:
    raw = np.asarray(actions, dtype=np.float32).reshape(-1, 2)
    cumulative = np.cumsum(np.clip(raw, -1.0, 1.0), axis=0)[
        ACTION_BLOCK - 1 :: ACTION_BLOCK
    ]
    expected = (
        np.asarray(reset_state, dtype=np.float32)[None]
        + np.float32(speed) * cumulative
    )
    return float(
        np.max(
            np.linalg.norm(
                np.asarray(next_states, dtype=np.float32) - expected,
                axis=-1,
            )
        )
    )


def _speed_support_audit(
    config: dict[str, Any], training_config_path: Path
) -> dict[str, Any]:
    training = yaml.safe_load(training_config_path.read_text(encoding="utf-8"))
    support = training["speed_support"]
    original = set(map(float, support["original_train"]))
    multi = set(map(float, support["multi_synthetic_train"]))
    monitor = set(map(float, support["training_monitor_only"]))
    calibration = set(map(float, support["planner_calibration"]))
    sealed = set(map(float, support["sealed_test_interpolation"]))
    tracks = {
        name: set(map(float, row["speeds"]))
        for name, row in config["data"]["tracks"].items()
    }
    low = tracks["extrapolation_low"]
    high = tracks["extrapolation_high"]
    used_development = original | multi | monitor | calibration | sealed
    checks = {
        "seen_is_multi_train_subset": tracks["seen_for_multi"] <= multi,
        "interpolation_disjoint_all_training": not tracks[
            "unseen_interpolation"
        ]
        & (original | multi | monitor),
        "interpolation_inside_multi_train_range": (
            min(multi)
            < min(tracks["unseen_interpolation"])
            < max(tracks["unseen_interpolation"])
            < max(multi)
        ),
        "low_strictly_below_multi_train": max(low) < min(multi),
        "high_strictly_above_multi_train": min(high) > max(multi),
        "extrapolation_disjoint_all_development_and_test": not (low | high)
        & used_development,
        "all_tracks_pairwise_disjoint_except_seen_train_role": all(
            not left & right
            for left_name, left in tracks.items()
            for right_name, right in tracks.items()
            if left_name < right_name
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Speed support isolation failed: {checks}")
    return {
        "passed": True,
        "source": str(training_config_path),
        "source_sha256": sha256_file(training_config_path),
        "multi_train_range": [min(multi), max(multi)],
        "tracks": {name: sorted(values) for name, values in tracks.items()},
        "checks": checks,
    }


def _build_track(
    *,
    config: dict[str, Any],
    track_name: str,
    track: dict[str, Any],
    geometries: list[SensitiveGeometry],
    payload_root: Path,
    stable_commit: str,
) -> dict[str, Any]:
    generation = config["data"]["generation"]
    evaluation = config["evaluation"]
    speeds = [float(value) for value in track["speeds"]]
    names = _condition_names(len(speeds))
    conditions = dict(zip(names, speeds))
    door = int(generation["door_position"])
    magnitude = float(generation["query_action_magnitude"])
    payload_root.mkdir(parents=True, exist_ok=True)
    bundles: list[dict[str, Any]] = []
    maximum_residual = 0.0
    context_pixel_failures = 0
    target_pixel_failures = {str(horizon): 0 for horizon in HORIZONS}
    family_counts: dict[str, int] = {}
    static_hashes: dict[str, str] = {}

    for geometry_index, geometry in enumerate(geometries):
        geometry_dict = asdict(geometry)
        static_identity = {
            "benchmark": config["benchmark"],
            "geometry": geometry_dict,
        }
        static_query_id = "twms-" + _sha256_bytes(
            _canonical_json(static_identity).encode("utf-8")
        )[:16]
        simulator_seed = int(
            np.random.SeedSequence(
                [
                    int(generation["catalog_seed"]),
                    int(geometry.distance_bin),
                    int(geometry.geometry_variant),
                ]
            ).generate_state(1)[0]
        )
        family, query_actions = _query_actions(
            geometry, magnitude, family_index=geometry_index
        )
        family_counts[family] = family_counts.get(family, 0) + 1
        context_actions = _context_actions(
            geometry, float(generation["context_action_magnitude"])
        )
        reset = np.asarray(geometry.reset_state, dtype=np.float32)
        goal = np.asarray(geometry.goal_state, dtype=np.float32)

        context_rollouts = {}
        for condition, speed in conditions.items():
            factors = {"agent.speed": speed, "door.position": door}
            rollout = _simulate_blocks(
                factors,
                reset,
                goal,
                context_actions,
                seed=simulator_seed,
            )
            residual = _free_motion_residual(
                geometry.reset_state,
                speed,
                context_actions,
                rollout["next_states"],
            )
            maximum_residual = max(maximum_residual, residual)
            if residual > 1e-3:
                raise RuntimeError(
                    f"Context collision/boundary residual {residual}: "
                    f"{track_name} {geometry.template_id} {speed:g}"
                )
            if not np.array_equal(rollout["next_pixels"][-1], rollout["pixels"][0]):
                raise RuntimeError("Context does not return to query pixels")
            context_rollouts[condition] = rollout
        context_intermediate_hashes = {
            _array_sha256(row["next_pixels"][0])
            for row in context_rollouts.values()
        }
        if len(context_intermediate_hashes) != len(speeds):
            context_pixel_failures += 1

        target_rollouts = {}
        for speed in speeds:
            factors = {"agent.speed": speed, "door.position": door}
            rollout = _simulate_blocks(
                factors,
                reset,
                goal,
                query_actions,
                seed=simulator_seed,
            )
            if not np.array_equal(
                rollout["pixels"][1:], rollout["next_pixels"][:-1]
            ):
                raise RuntimeError("Future rollout pixel continuity failed")
            residual = _free_motion_residual(
                geometry.reset_state,
                speed,
                query_actions,
                rollout["next_states"],
            )
            maximum_residual = max(maximum_residual, residual)
            if residual > 1e-3:
                raise RuntimeError(
                    f"Future collision/boundary residual {residual}: "
                    f"{track_name} {geometry.template_id} {speed:g}"
                )
            target_rollouts[speed] = rollout
        for horizon in HORIZONS:
            hashes = {
                _array_sha256(row["next_pixels"][horizon - 1])
                for row in target_rollouts.values()
            }
            if len(hashes) != len(speeds):
                target_pixel_failures[str(horizon)] += 1

        query_hashes = {
            _array_sha256(row["pixels"][0]) for row in target_rollouts.values()
        }
        if len(query_hashes) != 1:
            raise RuntimeError("Query pixels differ by reference speed")
        query_hash = next(iter(query_hashes))
        previous_hash = static_hashes.setdefault(static_query_id, query_hash)
        if previous_hash != query_hash:
            raise RuntimeError("Static query hash changed")

        for reference_speed in speeds:
            matching = next(
                name
                for name, speed in conditions.items()
                if np.isclose(speed, reference_speed, atol=1e-6, rtol=0.0)
            )
            identity = {
                **static_identity,
                "track": track_name,
                "reference_speed": reference_speed,
            }
            query_id = "twmsq-" + _sha256_bytes(
                _canonical_json(identity).encode("utf-8")
            )[:16]
            target = target_rollouts[reference_speed]
            arrays: dict[str, np.ndarray] = {
                "query_pixels": target["pixels"][0],
                "future_actions": query_actions,
                "future_pixels": target["pixels"],
                "future_next_pixels": target["next_pixels"],
                "future_states": target["states"],
                "future_next_states": target["next_states"],
            }
            condition_metadata = {}
            for condition, speed in conditions.items():
                rollout = context_rollouts[condition]
                prefix = f"context_b2_{condition}"
                arrays[f"{prefix}_pixels"] = rollout["pixels"]
                arrays[f"{prefix}_actions"] = rollout["actions"]
                arrays[f"{prefix}_next_pixels"] = rollout["next_pixels"]
                arrays[f"{prefix}_states"] = rollout["states"]
                arrays[f"{prefix}_next_states"] = rollout["next_states"]
                condition_metadata[condition] = {
                    "factors": _jsonable_factors(
                        {"agent.speed": speed, "door.position": door}
                    ),
                    "context_transitions": 2,
                }
            payload_path = payload_root / f"{query_id}.npz"
            np.savez_compressed(payload_path, **arrays)
            bundles.append(
                {
                    "query_id": query_id,
                    "static_query_id": static_query_id,
                    "track": track_name,
                    "split": str(track["split"]),
                    "role": str(track["role"]),
                    "simulator_seed": simulator_seed,
                    "template": {
                        "template_id": geometry.template_id,
                        "distance_bin": int(geometry.distance_bin),
                        "geometry_variant": int(geometry.geometry_variant),
                        "reset_state": list(geometry.reset_state),
                        "goal_state": list(geometry.goal_state),
                    },
                    "query_factors": _jsonable_factors(
                        {
                            "agent.speed": reference_speed,
                            "door.position": door,
                        }
                    ),
                    "conditions": condition_metadata,
                    "matching_condition": matching,
                    "query_action_family": family,
                    "payload": portable_contextworld_path(
                        payload_path, repo_root=ROOT
                    ),
                    "payload_sha256": _sha256_file(payload_path),
                    "query_pixels_sha256": query_hash,
                    "future_actions_sha256": _array_sha256(query_actions),
                    "target_pixels_sha256_by_horizon": {
                        str(horizon): _array_sha256(
                            target["next_pixels"][horizon - 1]
                        )
                        for horizon in HORIZONS
                    },
                    "model_visible_fields": ["pixels", "action"],
                    "privileged_fields": [
                        "query_factors",
                        "conditions.*.factors",
                        "state",
                        "target_state",
                    ],
                }
            )

    if context_pixel_failures or any(target_pixel_failures.values()):
        raise RuntimeError(
            "Pixel diagnosticity failed: "
            f"contexts={context_pixel_failures}, "
            f"targets={target_pixel_failures}"
        )
    catalog = {
        "schema_version": 1,
        "benchmark": config["benchmark"],
        "track": track_name,
        "split": str(track["split"]),
        "role": str(track["role"]),
        "stable_worldmodel_commit": stable_commit,
        "protocol": {
            "history_tokens": 3,
            "context_transitions": 2,
            "action_block_raw_steps": ACTION_BLOCK,
            "future_action_blocks": 5,
            "target_horizons_action_blocks": list(HORIZONS),
            "fully_autoregressive_model_scoring": True,
            "offline_true_future_targets": True,
            "query_actions_identical_across_reference_speeds": True,
            "query_pixels_identical_across_reference_speeds": True,
            "no_collision_boundary_clamp_or_termination": True,
        },
        "geometry_bank": [asdict(geometry) for geometry in geometries],
        "bundles": bundles,
        "summary": {
            "bundles": len(bundles),
            "base_geometries": len(geometries),
            "reference_speeds": speeds,
            "history_conditions": names,
            "matrix_cells": len(speeds) ** 2,
            "action_family_counts": family_counts,
            "maximum_free_motion_residual_px": maximum_residual,
            "context_pixel_diagnosticity_failures": context_pixel_failures,
            "target_pixel_diagnosticity_failures_by_horizon": (
                target_pixel_failures
            ),
            "passed": True,
        },
    }
    catalog = _assign_eval_seeds(
        catalog,
        eval_seeds=[int(value) for value in evaluation["eval_seeds"]],
        per_seed=int(
            evaluation["unique_queries_per_reference_speed_per_seed"]
        ),
        assignment_seed=int(generation["eval_seed_assignment_seed"]),
    )
    catalog_path = resolve_contextworld_path(track["catalog"], repo_root=ROOT)
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_path.write_text(
        json.dumps(catalog, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return catalog


def run(args: argparse.Namespace) -> dict[str, Any]:
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("status") != (
        "preregistered_before_catalog_generation_and_model_scoring"
    ):
        raise ValueError("v5 config is not frozen before execution")
    _, stable_repo, stable_commit = load_stable_worldmodel(
        ROOT, args.stablewm_repo, args.stablewm_ref
    )
    if stable_commit != config["stable_worldmodel"]["expected_ref"]:
        raise RuntimeError(f"StableWM commit mismatch: {stable_commit}")
    training_config = (ROOT / config["data"]["source_training_protocol"]).resolve()
    support_audit = _speed_support_audit(config, training_config)
    generation = config["data"]["generation"]
    geometries = generate_same_room_geometries(
        distances=[int(value) for value in generation["distance_bins_px"]],
        variants_per_distance=int(generation["variants_per_distance"]),
        geometry_seed=int(generation["geometry_seed"]),
    )
    expected = int(config["evaluation"]["unique_queries_per_reference_speed"])
    if len(geometries) != expected:
        raise RuntimeError(f"Expected {expected} geometries, got {len(geometries)}")
    payload_root = resolve_contextworld_path(
        config["artifacts"]["payload_root"], repo_root=ROOT
    )
    tracks = {}
    cross_track_query_hashes = {}
    for track_name, track in config["data"]["tracks"].items():
        print(f"[build] {track_name}", flush=True)
        catalog = _build_track(
            config=config,
            track_name=str(track_name),
            track=dict(track),
            geometries=geometries,
            payload_root=payload_root / str(track_name),
            stable_commit=stable_commit,
        )
        catalog_path = resolve_contextworld_path(track["catalog"], repo_root=ROOT)
        hashes = {
            str(row["static_query_id"]): str(row["query_pixels_sha256"])
            for row in catalog["bundles"]
        }
        cross_track_query_hashes[str(track_name)] = hashes
        tracks[str(track_name)] = {
            "catalog": str(catalog_path),
            "catalog_sha256": sha256_file(catalog_path),
            "summary": catalog["summary"],
        }
    reference_hashes = next(iter(cross_track_query_hashes.values()))
    cross_track_pass = all(
        hashes == reference_hashes
        for hashes in cross_track_query_hashes.values()
    )
    if not cross_track_pass:
        raise RuntimeError("Static query pixels differ across tracks")
    expected_trajectories = sum(
        int(row["condition_trajectories"])
        for row in config["evaluation"]["matrix_by_track"].values()
    )
    expected_static_queries = int(
        config["evaluation"]["unique_queries_per_reference_speed"]
    )
    if len(reference_hashes) != expected_static_queries:
        raise RuntimeError(
            "Static query count mismatch: "
            f"expected {expected_static_queries}, got {len(reference_hashes)}"
        )
    report = {
        "schema_version": 1,
        "benchmark": config["benchmark"],
        "status": "passed",
        "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
        "stable_worldmodel": {"repo": str(stable_repo), "commit": stable_commit},
        "speed_support_audit": support_audit,
        "tracks": tracks,
        "count_audit": {
            "unique_static_queries_per_track": len(reference_hashes),
            "expected_unique_static_queries_per_track": expected_static_queries,
            "unique_queries_per_reference_speed_per_seed": int(
                config["evaluation"]["unique_queries_per_reference_speed_per_seed"]
            ),
            "condition_trajectories_per_checkpoint_all_tracks": expected_trajectories,
            "horizon_losses_per_checkpoint_all_tracks": expected_trajectories
            * len(HORIZONS),
            "passed": len(reference_hashes) == expected_static_queries,
        },
        "cross_track_static_query_pixel_audit": {
            "queries": len(reference_hashes),
            "passed": cross_track_pass,
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
        / "configs/benchmark/tworoom_speed_multistep_extrap_v5.yaml",
    )
    parser.add_argument("--stablewm-repo", default="../stable-worldmodel")
    parser.add_argument("--stablewm-ref", default=PINNED_STABLEWM)
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
