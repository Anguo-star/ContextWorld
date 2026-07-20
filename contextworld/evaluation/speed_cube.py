from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from contextworld.paths import portable_contextworld_path

from .icl_catalog import (
    ACTION_BLOCK,
    _array_sha256,
    _canonical_json,
    _jsonable_factors,
    _sha256_bytes,
    _sha256_file,
    _simulate_blocks,
    diagnostic_action_blocks,
)
from .icl_sensitive import (
    DIRECT_POLICY_BUDGET,
    SensitiveGeometry,
    _direct_policy_feasible,
    generate_same_room_geometries,
)


def _history_labels(speeds: tuple[float, float, float]) -> dict[str, float]:
    if len(speeds) != 3:
        raise ValueError("A speed cube requires exactly three speeds")
    ordered = tuple(float(value) for value in speeds)
    if list(ordered) != sorted(ordered) or len(set(ordered)) != 3:
        raise ValueError(f"Speeds must be three unique increasing values: {speeds}")
    return {
        "history_low": ordered[0],
        "history_mid": ordered[1],
        "history_high": ordered[2],
    }


def _geometry_fingerprint(geometry: SensitiveGeometry) -> str:
    return hashlib.sha256(
        _canonical_json(asdict(geometry)).encode("utf-8")
    ).hexdigest()


def build_speed_cube_catalog(
    *,
    repo_root: Path,
    output_catalog: Path,
    payload_root: Path,
    split: str,
    distances: list[int],
    variants_per_distance: int,
    geometry_seed: int,
    catalog_seed: int,
    stable_worldmodel_commit: str,
    speeds: tuple[float, float, float],
    door_position: int = 49,
    benchmark_name: str = "tworoom_history3_speed_cube_v2",
    track_name: str = "unseen_interpolation",
    geometries_override: list[SensitiveGeometry] | None = None,
) -> dict[str, Any]:
    """Build a strict 3 query-speed × 3 history-speed paired catalog."""

    repo_root = repo_root.resolve()
    history = _history_labels(speeds)
    speed_values = tuple(history.values())
    geometries = (
        list(geometries_override)
        if geometries_override is not None
        else generate_same_room_geometries(
            distances=[int(value) for value in distances],
            variants_per_distance=int(variants_per_distance),
            geometry_seed=int(geometry_seed),
        )
    )
    expected = len(distances) * int(variants_per_distance)
    if len(geometries) != expected:
        raise ValueError(f"Expected {expected} geometries, got {len(geometries)}")
    if {geometry.distance_bin for geometry in geometries} != set(distances):
        raise ValueError("Geometry distance bins differ from the requested bins")

    payload_root.mkdir(parents=True, exist_ok=True)
    bundles: list[dict[str, Any]] = []
    feasibility: list[dict[str, Any]] = []
    static_pixel_hashes: dict[str, set[str]] = {}

    for query_speed in speed_values:
        same_speed_condition = next(
            name
            for name, history_speed in history.items()
            if np.isclose(
                history_speed, query_speed, rtol=0.0, atol=1e-6
            )
        )
        query_factors = {
            "agent.speed": float(query_speed),
            "door.position": int(door_position),
        }
        conditions = {
            name: {
                "agent.speed": float(history_speed),
                "door.position": int(door_position),
            }
            for name, history_speed in history.items()
        }
        for geometry in geometries:
            template = geometry.diagnostic_template()
            static_identity = {
                "benchmark": benchmark_name,
                "split": split,
                "track": track_name,
                "geometry": asdict(geometry),
            }
            static_query_id = "twsc-" + _sha256_bytes(
                _canonical_json(static_identity).encode("utf-8")
            )[:16]
            query_identity = {
                **static_identity,
                "query_speed": float(query_speed),
            }
            query_id = "twscq-" + _sha256_bytes(
                _canonical_json(query_identity).encode("utf-8")
            )[:16]
            # The simulator seed is geometry-only so all query-speed rows have
            # an exactly paired static query image.
            seed = int(
                np.random.SeedSequence(
                    [
                        int(catalog_seed),
                        int(geometry.distance_bin),
                        int(geometry.geometry_variant),
                    ]
                ).generate_state(1)[0]
            )
            reset_state = np.asarray(template.reset_state, dtype=np.float32)
            goal_state = np.asarray(template.goal_state, dtype=np.float32)
            query_actions = template.query_action_block()
            query_rollout = _simulate_blocks(
                query_factors,
                reset_state,
                goal_state,
                query_actions[None],
                seed=seed,
            )
            arrays: dict[str, np.ndarray] = {
                "query_pixels": query_rollout["pixels"][0],
                "query_action": query_actions,
                "query_state": query_rollout["states"][0],
                "target_pixels": query_rollout["next_pixels"][0],
                "target_state": query_rollout["next_states"][0],
            }
            context_metadata: dict[str, Any] = {}
            for condition_name, condition_factors in conditions.items():
                context_metadata[condition_name] = {
                    "factors": _jsonable_factors(condition_factors),
                    "budgets": [1, 2],
                }
                for budget in (1, 2):
                    action_blocks = diagnostic_action_blocks(
                        np.asarray(
                            template.context_direction, dtype=np.float32
                        ),
                        budget,
                    )
                    rollout = _simulate_blocks(
                        condition_factors,
                        reset_state,
                        goal_state,
                        action_blocks,
                        seed=seed,
                    )
                    prefix = f"context_b{budget}_{condition_name}"
                    for key, value in rollout.items():
                        arrays[f"{prefix}_{key}"] = value

            candidate_names = list(conditions)
            candidate_pixels: list[np.ndarray] = []
            candidate_states: list[np.ndarray] = []
            for candidate_name in candidate_names:
                rollout = _simulate_blocks(
                    conditions[candidate_name],
                    reset_state,
                    goal_state,
                    query_actions[None],
                    seed=seed,
                )
                candidate_pixels.append(rollout["next_pixels"][0])
                candidate_states.append(rollout["next_states"][0])
            arrays["candidate_pixels"] = np.stack(candidate_pixels)
            arrays["candidate_states"] = np.stack(candidate_states).astype(
                np.float32
            )

            payload_path = payload_root / f"{query_id}.npz"
            np.savez_compressed(payload_path, **arrays)
            query_pixels_hash = _array_sha256(arrays["query_pixels"])
            static_pixel_hashes.setdefault(static_query_id, set()).add(
                query_pixels_hash
            )
            feasible = _direct_policy_feasible(
                geometry,
                speed=float(query_speed),
                door_position=int(door_position),
                seed=seed,
                budget=DIRECT_POLICY_BUDGET,
            )
            feasibility.append(
                {
                    "query_id": query_id,
                    "query_speed": float(query_speed),
                    "template_id": template.template_id,
                    **feasible,
                }
            )
            bundle = {
                "query_id": query_id,
                "static_query_id": static_query_id,
                "paired_group_id": (
                    f"{split}:{track_name}:{template.template_id}"
                ),
                "track": track_name,
                "simulator_seed": seed,
                "family": "speed",
                "split": split,
                "regime": "same_room_speed_cube",
                "source_scenario_id": (
                    f"twsc-{split}-{track_name}-speed-{query_speed:g}-"
                    f"{template.template_id}"
                ),
                "source_manifest_fingerprint": _geometry_fingerprint(
                    geometry
                ),
                "template": {
                    "template_id": template.template_id,
                    "distance_bin": int(geometry.distance_bin),
                    "geometry_variant": int(geometry.geometry_variant),
                    "room_relation": "same_room",
                    "reset_state": reset_state.tolist(),
                    "goal_state": goal_state.tolist(),
                    "context_direction": list(template.context_direction),
                    "query_action_repeat": True,
                },
                "query_factors": _jsonable_factors(query_factors),
                "conditions": context_metadata,
                "same_speed_condition": same_speed_condition,
                "history_condition_speeds": {
                    name: float(value) for name, value in history.items()
                },
                "candidates": [
                    {
                        "name": name,
                        "factors": _jsonable_factors(conditions[name]),
                    }
                    for name in candidate_names
                ],
                # Kept for compatibility with the strict catalog validator;
                # this index means query-dynamics-matched, not that other
                # histories are invalid data.
                "correct_candidate_index": candidate_names.index(
                    same_speed_condition
                ),
                "payload": portable_contextworld_path(
                    payload_path, repo_root=repo_root
                ),
                "payload_sha256": _sha256_file(payload_path),
                "query_pixels_sha256": query_pixels_hash,
                "target_pixels_sha256": _array_sha256(
                    arrays["target_pixels"]
                ),
                "model_visible_fields": ["pixels", "action"],
                "privileged_catalog_fields": [
                    "query_factors",
                    "conditions.*.factors",
                    "candidate factors",
                    "state",
                    "distance_bin",
                ],
                "feasibility": feasible,
            }
            bundles.append(bundle)

    failed_feasibility = [row for row in feasibility if not row["passed"]]
    if failed_feasibility:
        raise RuntimeError(
            f"Direct-policy feasibility failed: {failed_feasibility[:5]}"
        )
    static_failures = {
        query_id: sorted(hashes)
        for query_id, hashes in static_pixel_hashes.items()
        if len(hashes) != 1
    }
    if static_failures:
        raise RuntimeError(
            "Static query pixels differ across query speeds: "
            f"{list(static_failures)[:5]}"
        )

    catalog = {
        "schema_version": 1,
        "benchmark": benchmark_name,
        "catalog_kind": "strict_speed_cube_context_query",
        "split": split,
        "track": track_name,
        "protocol": {
            "name": "tworoom_history3_speed_cube_v2",
            "action_block": ACTION_BLOCK,
            "model_history_tokens": 3,
            "supported_context_budgets": [1, 2],
            "maximum_prior_context_transitions": 2,
            "query_is_identical_across_history_conditions": True,
            "query_pixels_identical_across_query_speeds": True,
            "context_reset_goal_actions_are_paired": True,
            "factor_values_are_privileged_and_not_model_inputs": True,
            "task_relation": "same_room",
            "direct_policy_feasibility_budget": DIRECT_POLICY_BUDGET,
        },
        "stable_worldmodel_commit": stable_worldmodel_commit,
        "geometry_seed": int(geometry_seed),
        "generator_seed": int(catalog_seed),
        "source_manifests": {},
        "geometry_bank": [asdict(geometry) for geometry in geometries],
        "bundles": bundles,
        "summary": {
            "bundles": len(bundles),
            "by_family": {"speed": len(bundles)},
            "physical_payloads": len(bundles),
            "query_speeds": list(speed_values),
            "history_speeds": list(speed_values),
            "matrix_cells": 9,
            "distance_bins": [int(value) for value in distances],
            "variants_per_distance": int(variants_per_distance),
            "base_geometries": len(geometries),
            "static_query_groups": len(static_pixel_hashes),
            "static_query_pixel_audit_passed": True,
            "direct_policy_feasibility_passed": len(feasibility),
            "direct_policy_feasibility_failed": 0,
        },
    }
    output_catalog.parent.mkdir(parents=True, exist_ok=True)
    output_catalog.write_text(
        json.dumps(catalog, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return catalog


__all__ = ["build_speed_cube_catalog"]
