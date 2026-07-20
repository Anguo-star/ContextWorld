from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from contextworld.paths import portable_contextworld_path, resolve_contextworld_path


DEFAULT_SPEED = 5.0
DEFAULT_DOOR = 49
ACTION_BLOCK = 5
SUPPORTED_CONTEXT_BUDGETS = (0, 1, 2)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    header = f"{array.dtype.str}:{array.shape}".encode("utf-8")
    return _sha256_bytes(header + array.tobytes())


def _jsonable_factors(factors: dict[str, float | int]) -> dict[str, float | int]:
    return {
        "agent.speed": float(factors["agent.speed"]),
        "door.position": int(factors["door.position"]),
    }


def _factor_options(factors: dict[str, float | int]) -> dict[str, Any]:
    return {
        "agent.speed": np.asarray(
            [float(factors["agent.speed"])], dtype=np.float32
        ),
        "door.position": np.asarray(
            [int(factors["door.position"])] * 3, dtype=np.int64
        ),
        "door.size": np.asarray([14, 14, 14], dtype=np.int64),
        "door.number": 1,
        "wall.axis": 1,
        "wall.thickness": 10,
        "rendering.render_target": 0,
    }


def _farthest(value: float, candidates: Iterable[float]) -> float:
    choices = [float(candidate) for candidate in candidates if float(candidate) != float(value)]
    if not choices:
        raise ValueError(f"No counterfactual candidate differs from {value}")
    return max(choices, key=lambda candidate: (abs(candidate - float(value)), candidate))


def _impulse_block(direction: np.ndarray) -> np.ndarray:
    block = np.zeros((ACTION_BLOCK, 2), dtype=np.float32)
    block[0] = np.asarray(direction, dtype=np.float32)
    return block


def _zero_return_block(direction: np.ndarray) -> np.ndarray:
    block = np.zeros((ACTION_BLOCK, 2), dtype=np.float32)
    block[0] = np.asarray(direction, dtype=np.float32)
    block[1] = -np.asarray(direction, dtype=np.float32)
    return block


def diagnostic_action_blocks(direction: np.ndarray, budget: int) -> np.ndarray:
    """Return fixed open-loop blocks whose endpoint is exactly unchanged.

    K=1 is intentionally non-identifying for speed: its within-token impulse
    returns before the next observed frame. K=2 exposes a speed-dependent
    intermediate frame, then applies the exact inverse impulse so every factor
    reaches the same query state.
    """

    direction = np.asarray(direction, dtype=np.float32)
    if direction.shape != (2,) or not np.any(direction):
        raise ValueError(f"Expected a non-zero 2D direction, got {direction}")
    if np.any(np.abs(direction) > 1.0):
        raise ValueError(f"Direction exceeds action bounds: {direction}")
    if budget == 1:
        return _zero_return_block(direction)[None]
    if budget == 2:
        return np.stack(
            [_impulse_block(direction), _impulse_block(-direction)], axis=0
        )
    raise ValueError(f"Only non-zero budgets 1 and 2 are supported, got {budget}")


@dataclass(frozen=True)
class DiagnosticTemplate:
    template_id: str
    reset_state: tuple[float, float]
    goal_state: tuple[float, float]
    context_direction: tuple[float, float]
    query_action: tuple[float, float]
    query_action_repeat: bool

    def query_action_block(self) -> np.ndarray:
        action = np.asarray(self.query_action, dtype=np.float32)
        if self.query_action_repeat:
            return np.repeat(action[None], ACTION_BLOCK, axis=0)
        return _impulse_block(action)


def _templates(family: str, door_position: int) -> list[DiagnosticTemplate]:
    goal = (190.0, 190.0)
    if family == "speed":
        return [
            DiagnosticTemplate("s0", (55.0, 70.0), goal, (1.0, 0.5), (0.50, -0.20), True),
            DiagnosticTemplate("s1", (55.0, 150.0), goal, (1.0, -0.5), (0.40, 0.25), True),
            DiagnosticTemplate("s2", (169.0, 70.0), goal, (-1.0, 0.5), (-0.50, -0.20), True),
            DiagnosticTemplate("s3", (169.0, 150.0), goal, (-1.0, -0.5), (-0.40, 0.25), True),
        ]

    if family not in {"door", "speed_door_composition"}:
        raise ValueError(f"Unknown family {family!r}")
    # Motion stays away from the wall during context. The query crosses the
    # wall at the query door for composition; the door-only negative control
    # uses a stationary query so speed is truly irrelevant to its target.
    stationary = family == "door"
    left_query = (0.0, 0.0) if stationary else (1.0, 0.0)
    right_query = (0.0, 0.0) if stationary else (-1.0, 0.0)
    repeat = not stationary
    y = float(door_position)
    return [
        DiagnosticTemplate("g0", (90.0, y), goal, (-1.0, 0.5), left_query, repeat),
        DiagnosticTemplate("g1", (134.0, y), goal, (1.0, -0.5), right_query, repeat),
        DiagnosticTemplate("g2", (90.0, y), goal, (-0.75, -0.5), left_query, repeat),
        DiagnosticTemplate("g3", (134.0, y), goal, (0.75, 0.5), right_query, repeat),
    ]


def _simulate_blocks(
    factors: dict[str, float | int],
    reset_state: np.ndarray,
    goal_state: np.ndarray,
    action_blocks: np.ndarray,
    *,
    seed: int,
) -> dict[str, np.ndarray]:
    from stable_worldmodel.envs.two_room.env import TwoRoomEnv

    env = TwoRoomEnv(render_mode="rgb_array")
    observations: list[np.ndarray] = []
    next_observations: list[np.ndarray] = []
    states: list[np.ndarray] = []
    next_states: list[np.ndarray] = []
    try:
        env.reset(
            seed=int(seed),
            options={
                "variation": (),
                "variation_values": _factor_options(factors),
                "state": np.asarray(reset_state, dtype=np.float32).copy(),
                "target_state": np.asarray(goal_state, dtype=np.float32).copy(),
            },
        )
        expected_values = _factor_options(factors)
        speed_readback = np.asarray(
            env.variation_space["agent"]["speed"].value
        )
        door_readback = np.asarray(
            env.variation_space["door"]["position"].value
        )
        if not (
            np.array_equal(speed_readback, expected_values["agent.speed"])
            and np.array_equal(door_readback, expected_values["door.position"])
        ):
            raise RuntimeError(
                "Factor readback mismatch: "
                f"expected={factors}, observed_speed={speed_readback.tolist()}, "
                f"observed_door={door_readback.tolist()}"
            )

        for block_index, block in enumerate(np.asarray(action_blocks, dtype=np.float32)):
            observations.append(np.asarray(env.render(), dtype=np.uint8).copy())
            states.append(env.agent_position.detach().cpu().numpy().copy())
            for raw_index, action in enumerate(block):
                _, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    raise RuntimeError(
                        "Diagnostic sequence terminated at "
                        f"block={block_index}, raw_step={raw_index}, factors={factors}"
                    )
            next_observations.append(
                np.asarray(env.render(), dtype=np.uint8).copy()
            )
            next_states.append(env.agent_position.detach().cpu().numpy().copy())
    finally:
        env.close()

    return {
        "pixels": np.stack(observations),
        "actions": np.asarray(action_blocks, dtype=np.float32).copy(),
        "next_pixels": np.stack(next_observations),
        "states": np.stack(states).astype(np.float32),
        "next_states": np.stack(next_states).astype(np.float32),
    }


def _conditions_for_family(
    family: str,
    query: dict[str, float | int],
    *,
    speed_values: list[float],
    door_values: list[int],
) -> dict[str, dict[str, float | int]]:
    speed = float(query["agent.speed"])
    door = int(query["door.position"])
    wrong_speed = float(_farthest(speed, speed_values))
    wrong_door = int(_farthest(door, door_values))
    if family == "speed":
        return {
            "correct": {"agent.speed": speed, "door.position": door},
            "wrong": {"agent.speed": wrong_speed, "door.position": door},
            "irrelevant": {
                "agent.speed": speed,
                "door.position": wrong_door,
            },
        }
    if family == "door":
        return {
            "correct": {"agent.speed": speed, "door.position": door},
            "wrong": {"agent.speed": speed, "door.position": wrong_door},
            "irrelevant": {
                "agent.speed": wrong_speed,
                "door.position": door,
            },
        }
    if family == "speed_door_composition":
        return {
            "correct": {"agent.speed": speed, "door.position": door},
            "wrong_speed": {
                "agent.speed": wrong_speed,
                "door.position": door,
            },
            "irrelevant_door": {
                "agent.speed": speed,
                "door.position": wrong_door,
            },
            "wrong_both": {
                "agent.speed": wrong_speed,
                "door.position": wrong_door,
            },
        }
    raise ValueError(f"Unknown family {family!r}")


def _candidate_factors(
    family: str,
    conditions: dict[str, dict[str, float | int]],
) -> dict[str, dict[str, float | int]]:
    if family in {"speed", "door"}:
        return {
            "correct": conditions["correct"],
            "wrong": conditions["wrong"],
        }
    return {
        name: conditions[name]
        for name in ("correct", "wrong_speed", "irrelevant_door", "wrong_both")
    }


def _load_validation_manifests(
    manifest_paths: dict[str, Path],
) -> dict[str, list[dict[str, Any]]]:
    families: dict[str, list[dict[str, Any]]] = {}
    for family, path in manifest_paths.items():
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        selected = [row for row in rows if row.get("split") == "val"]
        if not selected:
            raise ValueError(f"No validation scenarios in {path}")
        families[family] = sorted(selected, key=lambda row: row["scenario_id"])
    return families


def build_tworoom_icl_validation_catalog(
    *,
    repo_root: Path,
    manifest_paths: dict[str, Path],
    output_catalog: Path,
    payload_root: Path,
    stable_worldmodel_commit: str,
    generator_seed: int = 20260714,
) -> dict[str, Any]:
    """Build physically paired TwoRoom validation prefixes and queries."""

    repo_root = repo_root.resolve()
    families = _load_validation_manifests(manifest_paths)
    speed_values = sorted(
        float(row["factors"]["agent.speed"])
        for row in families["speed"]
    )
    door_values = sorted(
        int(row["factors"]["door.position"])
        for row in families["door"]
    )
    payload_root.mkdir(parents=True, exist_ok=True)

    bundles: list[dict[str, Any]] = []
    for family in ("speed", "door", "speed_door_composition"):
        for scenario in families[family]:
            query_factors = {
                "agent.speed": float(
                    scenario["factors"].get("agent.speed", DEFAULT_SPEED)
                ),
                "door.position": int(
                    scenario["factors"].get("door.position", DEFAULT_DOOR)
                ),
            }
            conditions = _conditions_for_family(
                family,
                query_factors,
                speed_values=speed_values,
                door_values=door_values,
            )
            candidates = _candidate_factors(family, conditions)
            for replica, template in enumerate(
                _templates(family, int(query_factors["door.position"]))
            ):
                identity = {
                    "family": family,
                    "scenario_id": scenario["scenario_id"],
                    "template_id": template.template_id,
                    "replica": replica,
                    "split": "validation",
                    "protocol": "tworoom_paired_prefix_v1",
                }
                query_id = "twq-" + _sha256_bytes(
                    _canonical_json(identity).encode("utf-8")
                )[:16]
                seed = int(
                    np.random.SeedSequence(
                        [generator_seed, int(query_id.split("-")[1][:8], 16)]
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
                            np.asarray(template.context_direction, dtype=np.float32),
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
                arrays["candidate_states"] = np.stack(candidate_states).astype(
                    np.float32
                )

                payload_path = payload_root / f"{query_id}.npz"
                np.savez_compressed(payload_path, **arrays)
                correct_candidate = candidate_names.index("correct")
                bundle = {
                    "query_id": query_id,
                    "paired_group_id": f"{scenario['seed_group']}:{template.template_id}",
                    "track": "T1_trajectory_icl",
                    "simulator_seed": seed,
                    "family": family,
                    "split": "validation",
                    "regime": scenario["regime"],
                    "source_scenario_id": scenario["scenario_id"],
                    "source_manifest_fingerprint": scenario["fingerprint"],
                    "template": {
                        "template_id": template.template_id,
                        "reset_state": reset_state.tolist(),
                        "goal_state": goal_state.tolist(),
                        "context_direction": list(template.context_direction),
                        "query_action_repeat": template.query_action_repeat,
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
                    "correct_candidate_index": correct_candidate,
                    "payload": portable_contextworld_path(
                        payload_path, repo_root=repo_root
                    ),
                    "payload_sha256": _sha256_file(payload_path),
                    "query_pixels_sha256": _array_sha256(arrays["query_pixels"]),
                    "target_pixels_sha256": _array_sha256(arrays["target_pixels"]),
                    "model_visible_fields": ["pixels", "action"],
                    "privileged_catalog_fields": [
                        "query_factors",
                        "conditions.*.factors",
                        "candidate factors",
                        "state",
                    ],
                }
                bundles.append(bundle)

    catalog = {
        "schema_version": 1,
        "benchmark": "contextworld_tworoom_icl_v1",
        "catalog_kind": "strict_paired_context_query",
        "split": "validation",
        "track": "T0_zero_shot_ood_and_T1_trajectory_icl",
        "protocol": {
            "name": "tworoom_paired_prefix_v1",
            "action_block": ACTION_BLOCK,
            "model_history_tokens": 3,
            "supported_context_budgets": list(SUPPORTED_CONTEXT_BUDGETS),
            "maximum_prior_context_transitions": 2,
            "speed_identification_horizon": 2,
            "query_is_identical_across_context_conditions": True,
            "context_reset_goal_actions_are_paired": True,
            "factor_values_are_privileged_and_not_model_inputs": True,
        },
        "stable_worldmodel_commit": stable_worldmodel_commit,
        "generator_seed": int(generator_seed),
        "source_manifests": {
            family: portable_contextworld_path(path, repo_root=repo_root)
            for family, path in manifest_paths.items()
        },
        "bundles": bundles,
        "summary": {
            "bundles": len(bundles),
            "by_family": {
                family: sum(bundle["family"] == family for bundle in bundles)
                for family in ("speed", "door", "speed_door_composition")
            },
            "physical_payloads": len(bundles),
        },
    }
    output_catalog.parent.mkdir(parents=True, exist_ok=True)
    output_catalog.write_text(
        json.dumps(catalog, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return catalog


def _assert_array_equal(
    failures: list[dict[str, Any]],
    *,
    query_id: str,
    check: str,
    observed: np.ndarray,
    expected: np.ndarray,
) -> None:
    if not np.array_equal(observed, expected):
        failures.append(
            {
                "query_id": query_id,
                "check": check,
                "observed_shape": list(observed.shape),
                "expected_shape": list(expected.shape),
                "maximum_error": (
                    float(np.max(np.abs(observed.astype(np.float64) - expected.astype(np.float64))))
                    if observed.shape == expected.shape and observed.size
                    else None
                ),
            }
        )


def validate_context_query_catalog(
    catalog_path: Path,
    *,
    repo_root: Path,
    replay_simulator: bool = True,
    family: str | None = None,
) -> dict[str, Any]:
    """Fail closed on pairing, continuity, diagnosticity, and exact replay."""

    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    failures: list[dict[str, Any]] = []
    checked = 0
    query_ids: set[str] = set()
    payloads: set[str] = set()
    bundles = [
        bundle
        for bundle in catalog["bundles"]
        if family is None or bundle["family"] == family
    ]
    if not bundles:
        available = sorted({bundle["family"] for bundle in catalog["bundles"]})
        raise ValueError(f"No bundles for family={family!r}; available={available}")

    for bundle in bundles:
        query_id = bundle["query_id"]
        if query_id in query_ids:
            failures.append({"query_id": query_id, "check": "duplicate_query_id"})
        query_ids.add(query_id)
        payload_path = resolve_contextworld_path(
            bundle["payload"], repo_root=repo_root
        )
        if str(payload_path) in payloads:
            failures.append({"query_id": query_id, "check": "shared_payload"})
        payloads.add(str(payload_path))
        if not payload_path.is_file():
            failures.append({"query_id": query_id, "check": "missing_payload"})
            continue
        if _sha256_file(payload_path) != bundle["payload_sha256"]:
            failures.append({"query_id": query_id, "check": "payload_sha256"})
            continue

        with np.load(payload_path, allow_pickle=False) as payload:
            if _array_sha256(payload["query_pixels"]) != bundle["query_pixels_sha256"]:
                failures.append({"query_id": query_id, "check": "query_pixels_sha256"})
            if _array_sha256(payload["target_pixels"]) != bundle["target_pixels_sha256"]:
                failures.append({"query_id": query_id, "check": "target_pixels_sha256"})

            reference_actions: dict[int, np.ndarray] = {}
            reference_reset: dict[int, np.ndarray] = {}
            for condition_name, condition in bundle["conditions"].items():
                factors = condition["factors"]
                for budget in (1, 2):
                    prefix = f"context_b{budget}_{condition_name}"
                    pixels = payload[f"{prefix}_pixels"]
                    actions = payload[f"{prefix}_actions"]
                    next_pixels = payload[f"{prefix}_next_pixels"]
                    states = payload[f"{prefix}_states"]
                    next_states = payload[f"{prefix}_next_states"]
                    checked += 1
                    if budget not in reference_actions:
                        reference_actions[budget] = actions
                        reference_reset[budget] = states[0]
                    _assert_array_equal(
                        failures,
                        query_id=query_id,
                        check=f"paired_actions_b{budget}_{condition_name}",
                        observed=actions,
                        expected=reference_actions[budget],
                    )
                    _assert_array_equal(
                        failures,
                        query_id=query_id,
                        check=f"paired_reset_b{budget}_{condition_name}",
                        observed=states[0],
                        expected=reference_reset[budget],
                    )
                    if budget == 2:
                        _assert_array_equal(
                            failures,
                            query_id=query_id,
                            check=f"context_continuity_{condition_name}",
                            observed=next_pixels[0],
                            expected=pixels[1],
                        )
                    _assert_array_equal(
                        failures,
                        query_id=query_id,
                        check=f"endpoint_state_{condition_name}_b{budget}",
                        observed=next_states[-1],
                        expected=payload["query_state"],
                    )
                    if int(factors["door.position"]) == int(
                        bundle["query_factors"]["door.position"]
                    ):
                        _assert_array_equal(
                            failures,
                            query_id=query_id,
                            check=f"boundary_pixels_{condition_name}_b{budget}",
                            observed=next_pixels[-1],
                            expected=payload["query_pixels"],
                        )

                    if replay_simulator:
                        replay = _simulate_blocks(
                            factors,
                            np.asarray(bundle["template"]["reset_state"], dtype=np.float32),
                            np.asarray(bundle["template"]["goal_state"], dtype=np.float32),
                            actions,
                            seed=int(bundle["simulator_seed"]),
                        )
                        for key in ("pixels", "actions", "next_pixels", "states", "next_states"):
                            _assert_array_equal(
                                failures,
                                query_id=query_id,
                                check=f"simulator_replay_{condition_name}_b{budget}_{key}",
                                observed=payload[f"{prefix}_{key}"],
                                expected=replay[key],
                            )

            correct = "correct"
            wrong = "wrong_speed" if bundle["family"] == "speed_door_composition" else "wrong"
            correct_mid = payload[f"context_b2_{correct}_next_pixels"][0]
            wrong_mid = payload[f"context_b2_{wrong}_next_pixels"][0]
            if bundle["family"] in {"speed", "speed_door_composition"} and np.array_equal(
                correct_mid, wrong_mid
            ):
                failures.append({"query_id": query_id, "check": "speed_not_pixel_diagnostic_at_k2"})

            candidate_pixels = payload["candidate_pixels"]
            correct_index = int(bundle["correct_candidate_index"])
            _assert_array_equal(
                failures,
                query_id=query_id,
                check="correct_candidate_target_pixels",
                observed=candidate_pixels[correct_index],
                expected=payload["target_pixels"],
            )
            _assert_array_equal(
                failures,
                query_id=query_id,
                check="correct_candidate_target_state",
                observed=payload["candidate_states"][correct_index],
                expected=payload["target_state"],
            )

            if replay_simulator:
                query_actions = payload["query_action"]
                for candidate_index, candidate in enumerate(bundle["candidates"]):
                    replay = _simulate_blocks(
                        candidate["factors"],
                        payload["query_state"],
                        np.asarray(bundle["template"]["goal_state"], dtype=np.float32),
                        query_actions[None],
                        seed=int(bundle["simulator_seed"]),
                    )
                    _assert_array_equal(
                        failures,
                        query_id=query_id,
                        check=f"candidate_replay_pixels_{candidate['name']}",
                        observed=candidate_pixels[candidate_index],
                        expected=replay["next_pixels"][0],
                    )
                    _assert_array_equal(
                        failures,
                        query_id=query_id,
                        check=f"candidate_replay_state_{candidate['name']}",
                        observed=payload["candidate_states"][candidate_index],
                        expected=replay["next_states"][0],
                    )

    expected_bundles = (
        len(bundles) if family is not None else int(catalog["summary"]["bundles"])
    )
    if len(bundles) != expected_bundles:
        failures.append({"check": "summary_bundle_count"})
    return {
        "schema_version": 1,
        "catalog": str(catalog_path.resolve()),
        "passed": not failures,
        "bundles": len(bundles),
        "families": sorted({bundle["family"] for bundle in bundles}),
        "context_rollouts_checked": checked,
        "simulator_replay": bool(replay_simulator),
        "failures": failures,
    }


__all__ = [
    "ACTION_BLOCK",
    "SUPPORTED_CONTEXT_BUDGETS",
    "build_tworoom_icl_validation_catalog",
    "diagnostic_action_blocks",
    "validate_context_query_catalog",
]
