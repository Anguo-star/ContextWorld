from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from contextworld.paths import (
    portable_contextworld_path,
    resolve_contextworld_path,
)

from .hidden_passage_env import (
    PASSAGE_FACTOR,
    PASSAGE_RULES,
    make_hidden_passage_env,
)
from .speed_identifiability import agent_centroid_from_rgb


ACTION_BLOCK = 5
DIRECTIONS = ("left_to_right", "right_to_left")
RULE_NAMES = ("passable", "blocked")
MODEL_INPUT_KEYS = ("pixels", "action")
FROZEN_CONFIG_CANONICAL_SHA256 = (
    "9df7a4a1f178620426f26bd3935ca83f69511bcee8606293be81d2905c20667c"
)
REPLAY_ARRAY_KEYS = (
    "initial_observation",
    "history_pixels",
    "history_states",
    "history_raw_states",
    "history_actions",
    "query_pixels",
    "query_state",
    "query_action",
    "target_pixels",
    "target_state",
    "query_raw_states",
    "goal_pixels",
    "goal_state",
    "agent_speed",
    "passage_open",
    "door_number",
)


@dataclass(frozen=True)
class SpeedDoorRuleTemplate:
    template_id: str
    door_position: int
    direction: str
    doorway_offset_px: float
    reset_state: tuple[float, float]
    goal_state: tuple[float, float]
    simulator_seed: int


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(f"{array.dtype.str}:{array.shape}".encode("utf-8"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def frozen_config_sha256(config: dict[str, Any]) -> str:
    authored = {
        key: value for key, value in config.items() if key != "_config_path"
    }
    return _canonical_sha256(authored)


def validate_frozen_config(config: dict[str, Any]) -> str:
    observed = frozen_config_sha256(config)
    if observed != FROZEN_CONFIG_CANONICAL_SHA256:
        raise ValueError(
            "Speed-door-rule composition feasibility config differs from "
            "the frozen v1 contract: expected "
            f"{FROZEN_CONFIG_CANONICAL_SHA256}, observed {observed}"
        )
    return observed


def _payload_content_sha256(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(arrays.items()):
        digest.update(name.encode("utf-8"))
        digest.update(_array_sha256(value).encode("ascii"))
    return digest.hexdigest()


def _direction_sign(direction: str) -> float:
    if direction not in DIRECTIONS:
        raise ValueError(f"Unknown direction {direction!r}")
    return 1.0 if direction == "left_to_right" else -1.0


def _vertical_sign(door_position: int) -> float:
    return 1.0 if int(door_position) <= 112 else -1.0


def _factor_key(speed: float, rule: str) -> str:
    return f"speed_{float(speed):04.1f}_{rule}".replace(".", "p")


def _probe_block(
    template: SpeedDoorRuleTemplate,
    protocol: dict[str, Any],
) -> np.ndarray:
    block = np.zeros((ACTION_BLOCK, 2), dtype=np.float32)
    vertical = np.float32(
        _vertical_sign(template.door_position)
        * float(protocol["probe_action"]["vertical_speed_evidence"])
    )
    horizontal = np.float32(
        _direction_sign(template.direction)
        * float(protocol["probe_action"]["horizontal_rule_evidence"])
    )
    horizontal_steps = int(
        protocol["probe_action"]["horizontal_rule_steps"]
    )
    block[0, 1] = vertical
    block[1 : 1 + horizontal_steps, 0] = horizontal
    return block


def _recovery_block(
    template: SpeedDoorRuleTemplate,
    protocol: dict[str, Any],
) -> np.ndarray:
    if not protocol["recovery_action"]["cancel_vertical_speed_evidence"]:
        raise ValueError("The frozen v1 protocol requires vertical recovery")
    outward_steps = int(
        protocol["recovery_action"]["collision_projection_outward_steps"]
    )
    return_steps = int(
        protocol["recovery_action"]["collision_projection_return_steps"]
    )
    if (outward_steps, return_steps) != (1, 1):
        raise ValueError(
            "The frozen History=3 recovery currently supports one outward "
            "and one return collision-projection step"
        )
    block = np.zeros((ACTION_BLOCK, 2), dtype=np.float32)
    sign = np.float32(_vertical_sign(template.door_position))
    vertical_probe = np.float32(
        float(protocol["probe_action"]["vertical_speed_evidence"])
    )
    block[0, 1] = -sign * vertical_probe
    block[1, 1] = sign
    block[2, 1] = -sign
    return block


def _query_block(
    template: SpeedDoorRuleTemplate,
    protocol: dict[str, Any],
) -> np.ndarray:
    block = np.zeros((ACTION_BLOCK, 2), dtype=np.float32)
    horizontal_steps = int(protocol["query_action"]["horizontal_rule_steps"])
    if not 0 <= horizontal_steps <= ACTION_BLOCK:
        raise ValueError(
            f"Query horizontal steps must be in [0, {ACTION_BLOCK}]"
        )
    block[:horizontal_steps, 0] = np.float32(
        _direction_sign(template.direction)
        * float(protocol["query_action"]["horizontal_value"])
    )
    vertical_steps = int(
        protocol["query_action"].get("vertical_speed_steps", 1)
    )
    vertical_value = float(protocol["query_action"]["vertical_value"])
    if vertical_steps not in (0, 1):
        raise ValueError("Query vertical steps must be zero or one")
    if vertical_steps:
        if horizontal_steps >= ACTION_BLOCK:
            raise ValueError(
                "A vertical query step does not fit after the horizontal steps"
            )
        block[horizontal_steps, 1] = np.float32(
            _vertical_sign(template.door_position) * vertical_value
        )
    elif vertical_value != 0.0:
        raise ValueError(
            "A query with zero vertical steps must use vertical_value=0"
        )
    return block


def make_templates(
    *,
    door_positions: Iterable[int],
    directions: Iterable[str],
    doorway_offset_px: float,
    catalog_seed: int,
    left_to_right_reset_x: float,
    right_to_left_reset_x: float,
    left_to_right_goal_x: float,
    right_to_left_goal_x: float,
) -> list[SpeedDoorRuleTemplate]:
    templates: list[SpeedDoorRuleTemplate] = []
    for door in map(int, door_positions):
        sign_y = _vertical_sign(door)
        reset_y = float(door) + sign_y * float(doorway_offset_px)
        for direction in directions:
            if direction not in DIRECTIONS:
                raise ValueError(f"Unknown direction {direction!r}")
            left_to_right = direction == "left_to_right"
            reset_x = (
                float(left_to_right_reset_x)
                if left_to_right
                else float(right_to_left_reset_x)
            )
            goal_x = (
                float(left_to_right_goal_x)
                if left_to_right
                else float(right_to_left_goal_x)
            )
            seed = int(
                np.random.SeedSequence(
                    [
                        int(catalog_seed),
                        int(door),
                        DIRECTIONS.index(direction),
                    ]
                ).generate_state(1)[0]
            )
            templates.append(
                SpeedDoorRuleTemplate(
                    template_id=f"sdr-d{door:03d}-{direction}",
                    door_position=door,
                    direction=direction,
                    doorway_offset_px=float(doorway_offset_px),
                    reset_state=(reset_x, reset_y),
                    goal_state=(goal_x, reset_y),
                    simulator_seed=seed,
                )
            )
    return templates


def _variation_values(
    template: SpeedDoorRuleTemplate,
    *,
    speed: float,
    rule: str,
    door_number: int,
) -> dict[str, Any]:
    return {
        "agent.speed": np.asarray([speed], dtype=np.float32),
        "door.number": int(door_number),
        "door.position": np.asarray(
            [template.door_position] * 3, dtype=np.int64
        ),
        PASSAGE_FACTOR: int(PASSAGE_RULES[rule]),
    }


def _step_block(env: Any, block: np.ndarray, *, phase: str) -> np.ndarray:
    states: list[np.ndarray] = []
    for raw_step, action in enumerate(np.asarray(block, dtype=np.float32)):
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            raise RuntimeError(
                f"Composition feasibility trajectory terminated in {phase} "
                f"at raw step {raw_step}"
            )
        states.append(env.agent_position.detach().cpu().numpy().copy())
    return np.stack(states).astype(np.float32)


def simulate_template(
    template: SpeedDoorRuleTemplate,
    *,
    speed: float,
    rule: str,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    if rule not in RULE_NAMES:
        raise ValueError(f"Unknown hidden passage rule {rule!r}")
    door_number = int(protocol["door_number"])
    history_actions = np.stack(
        [
            _probe_block(template, protocol),
            _recovery_block(template, protocol),
        ]
    ).astype(np.float32)
    query_action = _query_block(template, protocol)
    env = make_hidden_passage_env(render_mode="rgb_array")
    history_pixels: list[np.ndarray] = []
    history_states: list[np.ndarray] = []
    try:
        observation, _ = env.reset(
            seed=int(template.simulator_seed),
            options={
                "variation": (),
                "variation_values": _variation_values(
                    template,
                    speed=float(speed),
                    rule=rule,
                    door_number=door_number,
                ),
                "state": np.asarray(
                    template.reset_state, dtype=np.float32
                ),
                "target_state": np.asarray(
                    template.goal_state, dtype=np.float32
                ),
            },
        )
        initial_observation = np.asarray(observation, dtype=np.float32).copy()
        history_pixels.append(np.asarray(env.render(), dtype=np.uint8).copy())
        history_states.append(
            env.agent_position.detach().cpu().numpy().copy()
        )

        probe_raw_states = _step_block(
            env, history_actions[0], phase="probe"
        )
        history_pixels.append(np.asarray(env.render(), dtype=np.uint8).copy())
        history_states.append(
            env.agent_position.detach().cpu().numpy().copy()
        )

        recovery_raw_states = _step_block(
            env, history_actions[1], phase="recovery"
        )
        history_pixels.append(np.asarray(env.render(), dtype=np.uint8).copy())
        history_states.append(
            env.agent_position.detach().cpu().numpy().copy()
        )

        query_pixels = history_pixels[-1].copy()
        query_state = history_states[-1].copy()
        goal_pixels = (
            env._target_img.detach().cpu().numpy().transpose(1, 2, 0).copy()
        )
        query_raw_states = _step_block(
            env, query_action, phase="query"
        )
        target_pixels = np.asarray(env.render(), dtype=np.uint8).copy()
        target_state = env.agent_position.detach().cpu().numpy().copy()
        speed_readback = float(
            np.asarray(
                env.variation_space["agent"]["speed"].value
            ).reshape(-1)[0]
        )
        passage_readback = int(env.passage_open)
        door_number_readback = int(env.num_doors)
    finally:
        env.close()

    return {
        "speed": float(speed),
        "rule": rule,
        "initial_observation": initial_observation,
        "history_pixels": np.stack(history_pixels).astype(np.uint8),
        "history_states": np.stack(history_states).astype(np.float32),
        "history_raw_states": np.concatenate(
            [probe_raw_states, recovery_raw_states],
            axis=0,
        ).astype(np.float32),
        "history_actions": history_actions,
        "query_pixels": query_pixels,
        "query_state": query_state.astype(np.float32),
        "query_action": query_action.astype(np.float32),
        "target_pixels": target_pixels,
        "target_state": target_state.astype(np.float32),
        "query_raw_states": query_raw_states.astype(np.float32),
        "goal_pixels": goal_pixels.astype(np.uint8),
        "goal_state": np.asarray(template.goal_state, dtype=np.float32),
        "agent_speed": np.asarray(speed_readback, dtype=np.float32),
        "passage_open": np.asarray(passage_readback, dtype=np.int64),
        "door_number": np.asarray(door_number_readback, dtype=np.int64),
    }


def model_input_projection(
    rollout: dict[str, Any],
) -> dict[str, np.ndarray]:
    actions = np.concatenate(
        [
            np.asarray(rollout["history_actions"], dtype=np.float32),
            np.asarray(rollout["query_action"], dtype=np.float32)[None],
        ],
        axis=0,
    )
    return {
        "pixels": np.asarray(rollout["history_pixels"], dtype=np.uint8),
        "action": actions,
    }


def replay_is_exact(
    left: dict[str, Any],
    right: dict[str, Any],
) -> bool:
    return all(
        np.array_equal(np.asarray(left[key]), np.asarray(right[key]))
        for key in REPLAY_ARRAY_KEYS
    )


def _all_arrays_equal(
    rollouts: dict[tuple[float, str], dict[str, Any]],
    key: str,
) -> bool:
    values = [np.asarray(rollout[key]) for rollout in rollouts.values()]
    return all(np.array_equal(values[0], value) for value in values[1:])


def _minimum_oriented_speed_gap(
    centroids: dict[tuple[float, str], np.ndarray],
    *,
    speeds: tuple[float, ...],
    rules: tuple[str, ...],
    vertical_sign: float,
) -> tuple[float, dict[str, list[float]]]:
    if len(speeds) == 1:
        return float("inf"), {rule: [] for rule in rules}
    gaps: dict[str, list[float]] = {}
    for rule in rules:
        gaps[rule] = [
            float(
                vertical_sign
                * (
                    centroids[(higher, rule)][1]
                    - centroids[(lower, rule)][1]
                )
            )
            for lower, higher in zip(speeds[:-1], speeds[1:])
        ]
    return min(gap for values in gaps.values() for gap in values), gaps


def _minimum_oriented_rule_gap(
    centroids_or_states: dict[tuple[float, str], np.ndarray],
    *,
    speeds: tuple[float, ...],
    direction_sign: float,
) -> tuple[float, dict[str, float]]:
    gaps = {
        f"{speed:g}": float(
            direction_sign
            * (
                centroids_or_states[(speed, "passable")][0]
                - centroids_or_states[(speed, "blocked")][0]
            )
        )
        for speed in speeds
    }
    return min(gaps.values()), gaps


def validate_factor_grid(
    template: SpeedDoorRuleTemplate,
    rollouts: dict[tuple[float, str], dict[str, Any]],
    *,
    speeds: tuple[float, ...],
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    expected_keys = {
        (speed, rule) for speed in speeds for rule in RULE_NAMES
    }
    if set(rollouts) != expected_keys:
        raise ValueError(
            f"Expected factor grid {sorted(expected_keys)}, "
            f"got {sorted(rollouts)}"
        )

    middle_centroids = {
        key: agent_centroid_from_rgb(rollout["history_pixels"][1])
        for key, rollout in rollouts.items()
    }
    future_centroids = {
        key: agent_centroid_from_rgb(rollout["target_pixels"])
        for key, rollout in rollouts.items()
    }
    future_states = {
        key: np.asarray(rollout["target_state"], dtype=np.float64)
        for key, rollout in rollouts.items()
    }
    minimum_middle_rule_gap, middle_rule_gaps = (
        _minimum_oriented_rule_gap(
            middle_centroids,
            speeds=speeds,
            direction_sign=_direction_sign(template.direction),
        )
    )
    minimum_middle_speed_gap, middle_speed_gaps = (
        _minimum_oriented_speed_gap(
            middle_centroids,
            speeds=speeds,
            rules=RULE_NAMES,
            vertical_sign=_vertical_sign(template.door_position),
        )
    )
    minimum_future_rule_gap, future_rule_gaps = (
        _minimum_oriented_rule_gap(
            future_states,
            speeds=speeds,
            direction_sign=_direction_sign(template.direction),
        )
    )
    minimum_future_speed_gap, future_speed_gaps = (
        _minimum_oriented_speed_gap(
            future_centroids,
            speeds=speeds,
            rules=RULE_NAMES,
            vertical_sign=_vertical_sign(template.door_position),
        )
    )
    middle_pixel_hashes = {
        _array_sha256(rollout["history_pixels"][1])
        for rollout in rollouts.values()
    }
    future_pixel_hashes = {
        _array_sha256(rollout["target_pixels"])
        for rollout in rollouts.values()
    }
    speed_readback = all(
        np.isclose(float(rollout["agent_speed"]), speed, atol=1e-6)
        for (speed, _), rollout in rollouts.items()
    )
    rule_readback = all(
        int(rollout["passage_open"]) == PASSAGE_RULES[rule]
        for (_, rule), rollout in rollouts.items()
    )
    history_third_frame_is_query = all(
        np.array_equal(
            rollout["history_pixels"][2], rollout["query_pixels"]
        )
        and np.array_equal(
            rollout["history_states"][2], rollout["query_state"]
        )
        for rollout in rollouts.values()
    )
    checks = {
        "factor_grid_complete": True,
        "speed_readback_exact": speed_readback,
        "rule_readback_exact": rule_readback,
        "door_number_readback_exact": all(
            int(rollout["door_number"]) == 1
            for rollout in rollouts.values()
        ),
        "initial_observation_identical": _all_arrays_equal(
            rollouts, "initial_observation"
        ),
        "initial_state_identical": all(
            np.array_equal(
                next(iter(rollouts.values()))["history_states"][0],
                rollout["history_states"][0],
            )
            for rollout in rollouts.values()
        ),
        "initial_pixels_identical": all(
            np.array_equal(
                next(iter(rollouts.values()))["history_pixels"][0],
                rollout["history_pixels"][0],
            )
            for rollout in rollouts.values()
        ),
        "history_actions_identical": _all_arrays_equal(
            rollouts, "history_actions"
        ),
        "query_state_identical": _all_arrays_equal(
            rollouts, "query_state"
        ),
        "query_pixels_identical": _all_arrays_equal(
            rollouts, "query_pixels"
        ),
        "history_third_frame_is_query": history_third_frame_is_query,
        "query_actions_identical": _all_arrays_equal(
            rollouts, "query_action"
        ),
        "goal_state_identical": _all_arrays_equal(
            rollouts, "goal_state"
        ),
        "goal_pixels_identical": _all_arrays_equal(
            rollouts, "goal_pixels"
        ),
        "six_unique_middle_pixels": (
            len(middle_pixel_hashes) == len(expected_keys)
        ),
        "middle_rule_gap_sufficient": (
            minimum_middle_rule_gap
            >= float(thresholds["minimum_middle_rule_centroid_gap_px"])
        ),
        "middle_speed_gap_sufficient": (
            minimum_middle_speed_gap
            >= float(
                thresholds[
                    "minimum_middle_adjacent_speed_centroid_gap_px"
                ]
            )
        ),
        "six_unique_future_pixels": (
            len(future_pixel_hashes) == len(expected_keys)
        ),
        "future_rule_gap_sufficient": (
            minimum_future_rule_gap
            >= float(thresholds["minimum_future_rule_state_gap_px"])
        ),
        "future_speed_gap_sufficient": (
            minimum_future_speed_gap
            >= float(
                thresholds[
                    "minimum_future_adjacent_speed_centroid_gap_px"
                ]
            )
        ),
    }
    representative = next(iter(rollouts.values()))
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "minimum_middle_rule_centroid_gap_px": (
            minimum_middle_rule_gap
        ),
        "middle_rule_centroid_gaps_px": middle_rule_gaps,
        "minimum_middle_adjacent_speed_centroid_gap_px": (
            minimum_middle_speed_gap
        ),
        "middle_adjacent_speed_centroid_gaps_px": middle_speed_gaps,
        "minimum_future_rule_state_gap_px": minimum_future_rule_gap,
        "future_rule_state_gaps_px": future_rule_gaps,
        "minimum_future_adjacent_speed_centroid_gap_px": (
            minimum_future_speed_gap
        ),
        "future_adjacent_speed_centroid_gaps_px": future_speed_gaps,
        "middle_agent_centroids_xy": {
            _factor_key(*key): value.tolist()
            for key, value in sorted(middle_centroids.items())
        },
        "future_agent_centroids_xy": {
            _factor_key(*key): value.tolist()
            for key, value in sorted(future_centroids.items())
        },
        "query_state": np.asarray(
            representative["query_state"]
        ).tolist(),
        "history_actions_sha256": _array_sha256(
            representative["history_actions"]
        ),
        "query_action_sha256": _array_sha256(
            representative["query_action"]
        ),
        "query_pixels_sha256": _array_sha256(
            representative["query_pixels"]
        ),
    }


def _prefixed_arrays(
    speed: float,
    rule: str,
    rollout: dict[str, Any],
) -> dict[str, np.ndarray]:
    prefix = _factor_key(speed, rule)
    return {
        f"{prefix}_{key}": np.asarray(rollout[key])
        for key in REPLAY_ARRAY_KEYS
    }


def _serialized_payload_audit(
    payload_path: Path,
    rollouts: dict[tuple[float, str], dict[str, Any]],
) -> dict[str, Any]:
    expected_arrays: dict[str, np.ndarray] = {}
    for (speed, rule), rollout in rollouts.items():
        expected_arrays.update(_prefixed_arrays(speed, rule, rollout))
    with np.load(payload_path, allow_pickle=False) as payload:
        keys_exact = set(payload.files) == set(expected_arrays)
        arrays_exact = keys_exact and all(
            np.array_equal(payload[name], expected)
            for name, expected in expected_arrays.items()
        )
        projections = {}
        for speed, rule in rollouts:
            prefix = _factor_key(speed, rule)
            serialized = {
                key: np.asarray(payload[f"{prefix}_{key}"]).copy()
                for key in REPLAY_ARRAY_KEYS
            }
            projections[(speed, rule)] = model_input_projection(serialized)

    expected_projections = {
        key: model_input_projection(rollout)
        for key, rollout in rollouts.items()
    }
    projection_keys_exact = all(
        tuple(projection) == MODEL_INPUT_KEYS
        for projection in projections.values()
    )
    projections_exact = all(
        np.array_equal(
            projections[factor][name],
            expected_projections[factor][name],
        )
        for factor in projections
        for name in MODEL_INPUT_KEYS
    )
    action_signatures = {
        _factor_key(*factor): _array_sha256(projection["action"])
        for factor, projection in projections.items()
    }
    query_signatures = {
        _factor_key(*factor): _array_sha256(
            rollouts[factor]["query_pixels"]
        )
        for factor in rollouts
    }
    projection_hashes = {
        _factor_key(*factor): _payload_content_sha256(projection)
        for factor, projection in projections.items()
    }
    checks = {
        "serialized_keys_exact": keys_exact,
        "serialized_arrays_roundtrip_exact": arrays_exact,
        "model_input_projection_keys_exact": projection_keys_exact,
        "model_input_projection_roundtrip_exact": projections_exact,
        "serialized_actions_identical_across_factor_grid": (
            len(set(action_signatures.values())) == 1
        ),
        "serialized_queries_identical_across_factor_grid": (
            len(set(query_signatures.values())) == 1
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "action_signatures": action_signatures,
        "query_signatures": query_signatures,
        "model_input_projection_sha256": projection_hashes,
    }


def _signature_leakage_audit(
    bundles: list[dict[str, Any]],
    *,
    signature_field: str,
    speeds: tuple[float, ...],
) -> dict[str, Any]:
    joint: dict[str, dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    speed_only: dict[str, dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    rule_only: dict[str, dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    for bundle in bundles:
        signatures = bundle[signature_field]
        for speed in speeds:
            for rule in RULE_NAMES:
                key = _factor_key(speed, rule)
                signature = signatures[key]
                joint[signature][key] += 1
                speed_only[signature][f"{speed:g}"] += 1
                rule_only[signature][rule] += 1

    def summarize(
        counts: dict[str, dict[str, int]],
        expected_accuracy: float,
    ) -> dict[str, Any]:
        total = sum(sum(row.values()) for row in counts.values())
        correct = sum(max(row.values()) for row in counts.values())
        accuracy = float(correct / total) if total else 1.0
        balanced = bool(total) and all(
            len(set(row.values())) == 1 for row in counts.values()
        )
        return {
            "passed": bool(
                balanced and np.isclose(accuracy, expected_accuracy)
            ),
            "best_signature_only_accuracy": accuracy,
            "expected_chance_accuracy": expected_accuracy,
            "balanced_within_signature": balanced,
        }

    joint_summary = summarize(joint, 1.0 / (len(speeds) * len(RULE_NAMES)))
    speed_summary = summarize(speed_only, 1.0 / len(speeds))
    rule_summary = summarize(rule_only, 1.0 / len(RULE_NAMES))
    return {
        "passed": all(
            item["passed"]
            for item in (joint_summary, speed_summary, rule_summary)
        ),
        "signatures": len(joint),
        "joint_factor": joint_summary,
        "speed": speed_summary,
        "door_rule": rule_summary,
    }


def build_feasibility_catalog(
    *,
    config: dict[str, Any],
    repo_root: Path,
    output_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    frozen_config_hash = validate_frozen_config(config)
    protocol = config["protocol"]
    geometry = config["geometry"]
    thresholds = config["gates"]
    speeds = tuple(float(speed) for speed in protocol["speeds"])
    if tuple(sorted(speeds)) != speeds or len(set(speeds)) != len(speeds):
        raise ValueError("Protocol speeds must be unique and increasing")
    templates = make_templates(
        door_positions=geometry["door_positions"],
        directions=geometry["directions"],
        doorway_offset_px=float(geometry["doorway_offset_px"]),
        catalog_seed=int(config["catalog_seed"]),
        left_to_right_reset_x=float(geometry["left_to_right_reset_x"]),
        right_to_left_reset_x=float(geometry["right_to_left_reset_x"]),
        left_to_right_goal_x=float(geometry["left_to_right_goal_x"]),
        right_to_left_goal_x=float(geometry["right_to_left_goal_x"]),
    )
    expected_templates = int(config["counts"]["templates"])
    expected_factor_combinations = int(
        config["counts"]["factor_combinations_per_template"]
    )
    expected_rollouts = int(config["counts"]["rollouts"])
    if len(templates) != expected_templates:
        raise ValueError(
            f"Expected {expected_templates} templates, built {len(templates)}"
        )
    if len(speeds) * len(RULE_NAMES) != expected_factor_combinations:
        raise ValueError("Factor-combination count does not match config")
    if len(templates) * expected_factor_combinations != expected_rollouts:
        raise ValueError("Rollout count does not match config")

    payload_root = output_root / "payloads"
    payload_root.mkdir(parents=True, exist_ok=True)
    bundles: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    replay_passed = 0
    query_hashes: set[str] = set()

    for template in templates:
        rollouts = {
            (speed, rule): simulate_template(
                template,
                speed=speed,
                rule=rule,
                protocol=protocol,
            )
            for speed in speeds
            for rule in RULE_NAMES
        }
        replays = {
            (speed, rule): simulate_template(
                template,
                speed=speed,
                rule=rule,
                protocol=protocol,
            )
            for speed in speeds
            for rule in RULE_NAMES
        }
        exact_replay = all(
            replay_is_exact(rollout, replays[factor])
            for factor, rollout in rollouts.items()
        )
        replay_passed += int(exact_replay)
        validation = validate_factor_grid(
            template,
            rollouts,
            speeds=speeds,
            thresholds=thresholds,
        )
        validation["checks"]["exact_replay"] = exact_replay

        arrays: dict[str, np.ndarray] = {}
        for (speed, rule), rollout in rollouts.items():
            arrays.update(_prefixed_arrays(speed, rule, rollout))
        payload_path = payload_root / f"{template.template_id}.npz"
        np.savez_compressed(payload_path, **arrays)
        serialized = _serialized_payload_audit(payload_path, rollouts)
        validation["checks"].update(serialized["checks"])
        validation["passed"] = all(validation["checks"].values())

        payload_reference = portable_contextworld_path(
            payload_path, repo_root=repo_root
        )
        resolved_payload = resolve_contextworld_path(
            payload_reference, repo_root=repo_root
        )
        if resolved_payload != payload_path.resolve():
            raise RuntimeError(
                "Serialized payload reference does not reopen its output: "
                f"{payload_reference!r} -> {resolved_payload}, "
                f"expected {payload_path.resolve()}"
            )
        if not resolved_payload.is_file():
            raise FileNotFoundError(
                f"Serialized payload is not replayable: {resolved_payload}"
            )
        query_hash = validation["query_pixels_sha256"]
        query_hashes.add(query_hash)
        bundle = {
            "template_id": template.template_id,
            "template": asdict(template),
            "speeds": list(speeds),
            "rules": list(RULE_NAMES),
            "factor_combinations": [
                {"speed": speed, "rule": rule}
                for speed in speeds
                for rule in RULE_NAMES
            ],
            "payload": payload_reference,
            "payload_sha256": _file_sha256(payload_path),
            "payload_content_sha256": _payload_content_sha256(arrays),
            "history_actions_sha256": validation[
                "history_actions_sha256"
            ],
            "query_action_sha256": validation["query_action_sha256"],
            "query_pixels_sha256": query_hash,
            "action_signatures": serialized["action_signatures"],
            "query_signatures": serialized["query_signatures"],
            "model_input_projection_sha256": serialized[
                "model_input_projection_sha256"
            ],
            "validation": validation,
        }
        bundles.append(bundle)
        if not validation["passed"]:
            failures.append(
                {
                    "template_id": template.template_id,
                    "failed_checks": sorted(
                        name
                        for name, passed in validation["checks"].items()
                        if not passed
                    ),
                }
            )

    action_leakage = _signature_leakage_audit(
        bundles,
        signature_field="action_signatures",
        speeds=speeds,
    )
    query_leakage = _signature_leakage_audit(
        bundles,
        signature_field="query_signatures",
        speeds=speeds,
    )
    by_direction = {
        direction: sum(
            bundle["template"]["direction"] == direction
            for bundle in bundles
        )
        for direction in DIRECTIONS
    }
    by_door = {
        str(door): sum(
            int(bundle["template"]["door_position"]) == int(door)
            for bundle in bundles
        )
        for door in geometry["door_positions"]
    }
    content_manifest = [
        {
            "template_id": bundle["template_id"],
            "payload_content_sha256": bundle["payload_content_sha256"],
            "query_pixels_sha256": bundle["query_pixels_sha256"],
            "history_actions_sha256": bundle["history_actions_sha256"],
            "query_action_sha256": bundle["query_action_sha256"],
            "action_signatures": bundle["action_signatures"],
            "query_signatures": bundle["query_signatures"],
            "model_input_projection_sha256": bundle[
                "model_input_projection_sha256"
            ],
        }
        for bundle in bundles
    ]
    content_manifest_sha = _canonical_sha256(content_manifest)
    minimums = {
        "middle_rule_centroid_gap_px": min(
            bundle["validation"][
                "minimum_middle_rule_centroid_gap_px"
            ]
            for bundle in bundles
        ),
        "middle_adjacent_speed_centroid_gap_px": min(
            bundle["validation"][
                "minimum_middle_adjacent_speed_centroid_gap_px"
            ]
            for bundle in bundles
        ),
        "future_rule_state_gap_px": min(
            bundle["validation"]["minimum_future_rule_state_gap_px"]
            for bundle in bundles
        ),
        "future_adjacent_speed_centroid_gap_px": min(
            bundle["validation"][
                "minimum_future_adjacent_speed_centroid_gap_px"
            ]
            for bundle in bundles
        ),
    }
    checks = {
        "all_factor_grids_pass": not failures,
        "exact_replay_all_templates": replay_passed == len(templates),
        "query_pixels_unique_across_templates": (
            len(query_hashes) == len(templates)
        ),
        "action_only_cannot_predict_factors": action_leakage["passed"],
        "query_only_cannot_predict_factors": query_leakage["passed"],
        "direction_balance": len(set(by_direction.values())) == 1,
        "door_balance": len(set(by_door.values())) == 1,
        "model_visible_fields_exclude_privileged_factors": (
            tuple(config["model_visible_fields"]) == MODEL_INPUT_KEYS
            and all(
                bundle["validation"]["checks"][
                    "model_input_projection_keys_exact"
                ]
                for bundle in bundles
            )
        ),
        "frozen_config_exact_match": (
            frozen_config_hash == FROZEN_CONFIG_CANONICAL_SHA256
        ),
        "serialized_payloads_roundtrip": all(
            bundle["validation"]["checks"][
                "serialized_arrays_roundtrip_exact"
            ]
            for bundle in bundles
        ),
    }
    counts = {
        "templates": len(templates),
        "factor_combinations_per_template": expected_factor_combinations,
        "rollouts": len(templates) * expected_factor_combinations,
        "by_direction": by_direction,
        "by_door_position": by_door,
        "by_speed": {
            f"{speed:g}": len(templates) * len(RULE_NAMES)
            for speed in speeds
        },
        "by_rule": {
            rule: len(templates) * len(speeds) for rule in RULE_NAMES
        },
    }
    catalog = {
        "schema_version": 1,
        "benchmark": str(config["benchmark"]),
        "status": str(config["status"]),
        "claim_limit": str(config["claim_limit"]),
        "protocol": protocol,
        "model_visible_fields": list(config["model_visible_fields"]),
        "privileged_audit_fields": list(config["privileged_audit_fields"]),
        "counts": counts,
        "content_manifest_sha256": content_manifest_sha,
        "bundles": bundles,
    }
    report = {
        "schema_version": 1,
        "benchmark": str(config["benchmark"]),
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "counts": counts,
        "thresholds": thresholds,
        "observed_minimums": minimums,
        "action_only_leakage_audit": action_leakage,
        "query_only_leakage_audit": query_leakage,
        "query_pixels": {
            "unique": len(query_hashes),
            "expected": len(templates),
        },
        "exact_replay_templates": replay_passed,
        "model_input_projection": {
            "keys": list(MODEL_INPUT_KEYS),
            "serialized_templates_passed": sum(
                bundle["validation"]["checks"][
                    "model_input_projection_roundtrip_exact"
                ]
                for bundle in bundles
            ),
            "expected_templates": len(bundles),
            "formal_stablewm_adapter_connected": False,
        },
        "failed_templates": failures,
        "content_manifest_sha256": content_manifest_sha,
        "frozen_config_canonical_sha256": frozen_config_hash,
        "known_limit": config["known_limit"],
        "interpretation": {
            "passed_means": (
                "Under the pinned TwoRoom collision engine, History=3 pixels "
                "contain separately measurable speed and door-rule evidence; "
                "the identical query and identical actions reveal neither "
                "factor, and both factors change the true future."
            ),
            "does_not_mean": (
                "No model composition ICL, formal training, held-out "
                "validation, planning, or cross-engine portability claim is "
                "established by this feasibility build."
            ),
        },
        "next_gate": config["next_gate"],
    }
    return catalog, report


__all__ = [
    "ACTION_BLOCK",
    "DIRECTIONS",
    "FROZEN_CONFIG_CANONICAL_SHA256",
    "MODEL_INPUT_KEYS",
    "RULE_NAMES",
    "SpeedDoorRuleTemplate",
    "build_feasibility_catalog",
    "frozen_config_sha256",
    "make_templates",
    "model_input_projection",
    "replay_is_exact",
    "simulate_template",
    "validate_factor_grid",
    "validate_frozen_config",
]
