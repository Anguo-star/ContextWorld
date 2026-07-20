from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from contextworld.paths import portable_contextworld_path

from .icl_catalog import (
    ACTION_BLOCK,
    DiagnosticTemplate,
    _array_sha256,
    _candidate_factors,
    _canonical_json,
    _conditions_for_family,
    _jsonable_factors,
    _sha256_bytes,
    _sha256_file,
    _simulate_blocks,
    diagnostic_action_blocks,
)


EVAL_SPEEDS = (3.1, 3.3, 3.5, 4.1, 5.0, 5.1, 5.9, 7.0)
DEFAULT_DOOR_POSITION = 49
DIRECT_POLICY_BUDGET = 50
SUCCESS_RADIUS = 16.0


@dataclass(frozen=True)
class SensitiveGeometry:
    template_id: str
    distance_bin: int
    geometry_variant: int
    reset_state: tuple[float, float]
    goal_state: tuple[float, float]
    context_direction: tuple[float, float]
    query_action: tuple[float, float]

    def diagnostic_template(self) -> DiagnosticTemplate:
        return DiagnosticTemplate(
            template_id=self.template_id,
            reset_state=self.reset_state,
            goal_state=self.goal_state,
            context_direction=self.context_direction,
            query_action=self.query_action,
            query_action_repeat=True,
        )


def _geometry_fingerprint(geometry: SensitiveGeometry) -> str:
    return _sha256_bytes(
        _canonical_json(asdict(geometry)).encode("utf-8")
    )


def generate_same_room_geometries(
    *,
    distances: list[int],
    variants_per_distance: int,
    geometry_seed: int,
) -> list[SensitiveGeometry]:
    """Generate deterministic right-room geometries with exact distance bins."""

    if not distances or any(distance <= 32 for distance in distances):
        raise ValueError("Distances must be non-empty and greater than 32 px")
    if variants_per_distance <= 0:
        raise ValueError("variants_per_distance must be positive")

    directions = (
        np.asarray([0.0, 1.0], dtype=np.float64),
        np.asarray([0.0, -1.0], dtype=np.float64),
        np.asarray([0.6, 0.8], dtype=np.float64),
        np.asarray([-0.6, -0.8], dtype=np.float64),
        np.asarray([0.6, -0.8], dtype=np.float64),
        np.asarray([-0.6, 0.8], dtype=np.float64),
    )
    rng = np.random.default_rng(int(geometry_seed))
    geometries: list[SensitiveGeometry] = []
    seen_pairs: set[tuple[float, float, float, float]] = set()

    for distance in distances:
        for variant in range(variants_per_distance):
            direction = directions[
                (variant + int(geometry_seed) + distance // 8)
                % len(directions)
            ]
            goal_delta = float(distance) * direction
            geometry: SensitiveGeometry | None = None
            for _ in range(20_000):
                reset = np.asarray(
                    [
                        rng.integers(132, 203),
                        rng.integers(22, 203),
                    ],
                    dtype=np.float64,
                )
                goal = reset + goal_delta
                # Reset is kept far enough from the wall and outer border for
                # the diagnostic impulse. Goal may use the larger valid area.
                if not (
                    125.0 <= goal[0] <= 209.0
                    and 15.0 <= goal[1] <= 209.0
                ):
                    continue
                pair = tuple(
                    np.round(np.concatenate([reset, goal]), 6).tolist()
                )
                if pair in seen_pairs:
                    continue

                # Use the same exactly reversible binary-fraction action
                # components as the validated E4 templates. Arbitrary
                # normalized directions can leave a float32 endpoint residual
                # after the inverse impulse and break strict pixel pairing.
                context = np.asarray(
                    [
                        1.0 if reset[0] <= 167.0 else -1.0,
                        0.0,
                    ],
                    dtype=np.float64,
                )
                context_endpoint = reset + context * max(EVAL_SPEEDS)
                if not (
                    125.0 <= context_endpoint[0] <= 209.0
                    and 15.0 <= context_endpoint[1] <= 209.0
                ):
                    continue

                query_action = 0.35 * direction
                template_id = f"d{distance:03d}_g{variant:02d}"
                geometry = SensitiveGeometry(
                    template_id=template_id,
                    distance_bin=int(distance),
                    geometry_variant=int(variant),
                    reset_state=(float(reset[0]), float(reset[1])),
                    goal_state=(float(goal[0]), float(goal[1])),
                    context_direction=(
                        float(context[0]),
                        float(context[1]),
                    ),
                    query_action=(
                        float(query_action[0]),
                        float(query_action[1]),
                    ),
                )
                seen_pairs.add(pair)
                break
            if geometry is None:
                raise RuntimeError(
                    f"Could not generate geometry distance={distance}, "
                    f"variant={variant}"
                )
            observed_distance = float(
                np.linalg.norm(
                    np.asarray(geometry.goal_state)
                    - np.asarray(geometry.reset_state)
                )
            )
            if not math.isclose(
                observed_distance,
                float(distance),
                rel_tol=0.0,
                abs_tol=1e-5,
            ):
                raise RuntimeError(
                    f"Distance construction drift: {observed_distance}"
                )
            geometries.append(geometry)

    return geometries


def _direct_policy_feasible(
    geometry: SensitiveGeometry,
    *,
    speed: float,
    door_position: int,
    seed: int,
    budget: int = DIRECT_POLICY_BUDGET,
) -> dict[str, Any]:
    from stable_worldmodel.envs.two_room.env import TwoRoomEnv

    from .icl_catalog import _factor_options

    factors = {
        "agent.speed": float(speed),
        "door.position": int(door_position),
    }
    env = TwoRoomEnv(render_mode="rgb_array")
    try:
        env.reset(
            seed=int(seed),
            options={
                "variation": (),
                "variation_values": _factor_options(factors),
                "state": np.asarray(
                    geometry.reset_state, dtype=np.float32
                ),
                "target_state": np.asarray(
                    geometry.goal_state, dtype=np.float32
                ),
            },
        )
        terminated = False
        steps = 0
        for steps in range(1, int(budget) + 1):
            position = (
                env.agent_position.detach().cpu().numpy().astype(np.float32)
            )
            delta = (
                np.asarray(geometry.goal_state, dtype=np.float32) - position
            )
            action = np.clip(
                delta / float(speed), -1.0, 1.0
            ).astype(np.float32)
            _, _, terminated, _, _ = env.step(action)
            if terminated:
                break
        final_state = (
            env.agent_position.detach().cpu().numpy().astype(np.float32)
        )
        final_distance = float(
            np.linalg.norm(
                final_state
                - np.asarray(geometry.goal_state, dtype=np.float32)
            )
        )
        return {
            "passed": bool(terminated or final_distance < SUCCESS_RADIUS),
            "steps": int(steps),
            "final_distance": final_distance,
        }
    finally:
        env.close()


def build_speed_icl_sensitive_catalog(
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
    speeds: tuple[float, ...] = EVAL_SPEEDS,
    door_position: int = DEFAULT_DOOR_POSITION,
    wrong_speed_override: float | None = None,
    benchmark_name: str = "tworoom_speed_icl_sensitive_eval_v1",
    track_name: str = "T1_speed_icl_sensitive_planning",
    protocol_name: str = "tworoom_speed_icl_sensitive_v1",
    regime: str = "same_room_distance_calibration",
    geometries_override: list[SensitiveGeometry] | None = None,
) -> dict[str, Any]:
    """Build a strict paired speed catalog for distance calibration/heldout use."""

    repo_root = repo_root.resolve()
    geometries = (
        list(geometries_override)
        if geometries_override is not None
        else generate_same_room_geometries(
            distances=[int(value) for value in distances],
            variants_per_distance=int(variants_per_distance),
            geometry_seed=int(geometry_seed),
        )
    )
    expected_distances = {int(value) for value in distances}
    observed_distances = {
        int(geometry.distance_bin) for geometry in geometries
    }
    if observed_distances != expected_distances:
        raise ValueError(
            "Geometry override distance bins differ from the frozen bins: "
            f"{sorted(observed_distances)} != {sorted(expected_distances)}"
        )
    expected_geometries = (
        len(expected_distances) * int(variants_per_distance)
    )
    if len(geometries) != expected_geometries:
        raise ValueError(
            "Geometry count differs from distances × variants: "
            f"{len(geometries)} != {expected_geometries}"
        )
    by_distance = {
        distance: sum(
            int(geometry.distance_bin) == distance
            for geometry in geometries
        )
        for distance in expected_distances
    }
    if any(
        count != int(variants_per_distance)
        for count in by_distance.values()
    ):
        raise ValueError(
            "Geometry override variants per distance differ from the "
            f"frozen count: {by_distance}"
        )
    geometry_ids = {
        (geometry.template_id, int(geometry.geometry_variant))
        for geometry in geometries
    }
    if len(geometry_ids) != len(geometries):
        raise ValueError("Geometry override contains duplicate identities")
    payload_root.mkdir(parents=True, exist_ok=True)
    speed_values = [float(value) for value in speeds]
    door_values = [int(door_position), 85]
    bundles: list[dict[str, Any]] = []
    feasibility: list[dict[str, Any]] = []

    for speed_index, speed in enumerate(speed_values):
        query_factors = {
            "agent.speed": float(speed),
            "door.position": int(door_position),
        }
        conditions = _conditions_for_family(
            "speed",
            query_factors,
            speed_values=speed_values,
            door_values=door_values,
        )
        if wrong_speed_override is not None:
            if math.isclose(
                float(wrong_speed_override),
                float(speed),
                rel_tol=0.0,
                abs_tol=1e-6,
            ):
                raise ValueError(
                    "wrong_speed_override must differ from every query speed"
                )
            conditions["wrong"] = {
                "agent.speed": float(wrong_speed_override),
                "door.position": int(door_position),
            }
        candidates = _candidate_factors("speed", conditions)
        for geometry in geometries:
            template = geometry.diagnostic_template()
            identity = {
                "benchmark": str(benchmark_name),
                "split": str(split),
                "speed": float(speed),
                "template_id": template.template_id,
                "geometry": asdict(geometry),
            }
            query_id = "twsi-" + _sha256_bytes(
                _canonical_json(identity).encode("utf-8")
            )[:16]
            seed = int(
                np.random.SeedSequence(
                    [
                        int(catalog_seed),
                        int(speed_index),
                        int(geometry.distance_bin),
                        int(geometry.geometry_variant),
                    ]
                ).generate_state(1)[0]
            )
            reset_state = np.asarray(
                template.reset_state, dtype=np.float32
            )
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

            candidate_names = list(candidates)
            candidate_pixels: list[np.ndarray] = []
            candidate_states: list[np.ndarray] = []
            for candidate_name in candidate_names:
                rollout = _simulate_blocks(
                    candidates[candidate_name],
                    reset_state,
                    goal_state,
                    query_actions[None],
                    seed=seed,
                )
                candidate_pixels.append(rollout["next_pixels"][0])
                candidate_states.append(rollout["next_states"][0])
            arrays["candidate_pixels"] = np.stack(candidate_pixels)
            arrays["candidate_states"] = np.stack(
                candidate_states
            ).astype(np.float32)

            payload_path = payload_root / f"{query_id}.npz"
            np.savez_compressed(payload_path, **arrays)
            feasibility_result = _direct_policy_feasible(
                geometry,
                speed=float(speed),
                door_position=int(door_position),
                seed=seed,
            )
            feasibility.append(
                {
                    "query_id": query_id,
                    "speed": float(speed),
                    "template_id": template.template_id,
                    "distance_bin": int(geometry.distance_bin),
                    **feasibility_result,
                }
            )
            bundle = {
                "query_id": query_id,
                "paired_group_id": (
                    f"{split}:speed={speed:g}:{template.template_id}"
                ),
                "track": str(track_name),
                "simulator_seed": seed,
                "family": "speed",
                "split": str(split),
                "regime": str(regime),
                "source_scenario_id": (
                    f"twsi-{split}-speed-{speed:g}-"
                    f"{template.template_id}"
                ),
                "source_manifest_fingerprint": _geometry_fingerprint(
                    geometry
                ),
                "template": {
                    "template_id": template.template_id,
                    "distance_bin": int(geometry.distance_bin),
                    "geometry_variant": int(
                        geometry.geometry_variant
                    ),
                    "room_relation": "same_room",
                    "reset_state": reset_state.tolist(),
                    "goal_state": goal_state.tolist(),
                    "context_direction": list(
                        template.context_direction
                    ),
                    "query_action_repeat": True,
                },
                "query_factors": _jsonable_factors(query_factors),
                "conditions": context_metadata,
                "shuffled": {
                    "source": "correct",
                    "budget": 2,
                    "permutation": [1, 0],
                },
                "candidates": [
                    {
                        "name": name,
                        "factors": _jsonable_factors(candidates[name]),
                    }
                    for name in candidate_names
                ],
                "correct_candidate_index": candidate_names.index(
                    "correct"
                ),
                "payload": portable_contextworld_path(
                    payload_path, repo_root=repo_root
                ),
                "payload_sha256": _sha256_file(payload_path),
                "query_pixels_sha256": _array_sha256(
                    arrays["query_pixels"]
                ),
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
                "feasibility": feasibility_result,
            }
            bundles.append(bundle)

    failed_feasibility = [
        row for row in feasibility if not row["passed"]
    ]
    if failed_feasibility:
        raise RuntimeError(
            "Direct-policy feasibility failed: "
            f"{failed_feasibility[:5]}"
        )

    catalog = {
        "schema_version": 1,
        "benchmark": str(benchmark_name),
        "catalog_kind": "strict_paired_context_query",
        "split": str(split),
        "track": str(track_name),
        "protocol": {
            "name": str(protocol_name),
            "action_block": ACTION_BLOCK,
            "model_history_tokens": 3,
            "supported_context_budgets": [0, 1, 2],
            "maximum_prior_context_transitions": 2,
            "query_is_identical_across_context_conditions": True,
            "context_reset_goal_actions_are_paired": True,
            "factor_values_are_privileged_and_not_model_inputs": True,
            "task_relation": "same_room",
            "direct_policy_feasibility_budget": DIRECT_POLICY_BUDGET,
            "wrong_speed_override": (
                None
                if wrong_speed_override is None
                else float(wrong_speed_override)
            ),
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
            "speeds": speed_values,
            "distance_bins": [int(value) for value in distances],
            "variants_per_distance": int(variants_per_distance),
            "base_geometries": len(geometries),
            "direct_policy_feasibility_passed": len(feasibility),
            "direct_policy_feasibility_failed": 0,
            "wrong_speed_override": (
                None
                if wrong_speed_override is None
                else float(wrong_speed_override)
            ),
        },
    }
    output_catalog.parent.mkdir(parents=True, exist_ok=True)
    output_catalog.write_text(
        json.dumps(catalog, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return catalog


def geometry_pair_set(catalog: dict[str, Any]) -> set[tuple[float, ...]]:
    return {
        tuple(
            np.round(
                np.asarray(
                    [
                        *geometry["reset_state"],
                        *geometry["goal_state"],
                    ],
                    dtype=np.float64,
                ),
                6,
            ).tolist()
        )
        for geometry in catalog["geometry_bank"]
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "DEFAULT_DOOR_POSITION",
    "DIRECT_POLICY_BUDGET",
    "EVAL_SPEEDS",
    "SensitiveGeometry",
    "build_speed_icl_sensitive_catalog",
    "generate_same_room_geometries",
    "geometry_pair_set",
    "sha256_file",
]
