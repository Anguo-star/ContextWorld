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


ACTION_BLOCK = 5
DIRECTIONS = ("left_to_right", "right_to_left")
RULE_NAMES = ("passable", "blocked")
MODEL_INPUT_KEYS = ("pixels", "action")
FROZEN_CONFIG_CANONICAL_SHA256 = (
    "d701ad20a895c324181406e6da9249a3babe911964705e328b44b0326e81b2fa"
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
    "passage_open",
    "door_number",
)


@dataclass(frozen=True)
class HiddenPassageTemplate:
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
    """Hash the authored YAML content, excluding loader-only metadata."""

    authored = {
        key: value for key, value in config.items() if key != "_config_path"
    }
    return _canonical_sha256(authored)


def validate_frozen_config(config: dict[str, Any]) -> str:
    """Fail closed if any field in the v1 feasibility protocol changes."""

    observed = frozen_config_sha256(config)
    if observed != FROZEN_CONFIG_CANONICAL_SHA256:
        raise ValueError(
            "Hidden-passage feasibility config differs from the frozen v1 "
            f"contract: expected {FROZEN_CONFIG_CANONICAL_SHA256}, "
            f"observed {observed}"
        )
    return observed


def _payload_content_sha256(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(arrays.items()):
        digest.update(name.encode("utf-8"))
        digest.update(_array_sha256(value).encode("ascii"))
    return digest.hexdigest()


def model_input_projection(
    rollout: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Return the only arrays allowed to reach the LeWM adapter."""

    actions = np.concatenate(
        [
            np.asarray(rollout["history_actions"], dtype=np.float32),
            np.asarray(
                rollout["query_action"], dtype=np.float32
            )[None, ...],
        ],
        axis=0,
    )
    return {
        "pixels": np.asarray(rollout["history_pixels"], dtype=np.uint8),
        "action": actions,
    }


def _direction_sign(direction: str) -> float:
    if direction not in DIRECTIONS:
        raise ValueError(f"Unknown direction {direction!r}")
    return 1.0 if direction == "left_to_right" else -1.0


def _doorway_vertical_sign(door_position: int) -> float:
    return 1.0 if int(door_position) <= 112 else -1.0


def _probe_block(direction: str) -> np.ndarray:
    block = np.zeros((ACTION_BLOCK, 2), dtype=np.float32)
    block[:2, 0] = np.float32(_direction_sign(direction))
    return block


def _recovery_block(door_position: int) -> np.ndarray:
    block = np.zeros((ACTION_BLOCK, 2), dtype=np.float32)
    sign = np.float32(_doorway_vertical_sign(door_position))
    block[0, 1] = sign
    block[1, 1] = -sign
    return block


def _query_block(direction: str) -> np.ndarray:
    action = np.asarray(
        [[_direction_sign(direction), 0.0]], dtype=np.float32
    )
    return np.repeat(action, ACTION_BLOCK, axis=0)


def make_templates(
    *,
    door_positions: Iterable[int],
    directions: Iterable[str],
    doorway_offsets_px: Iterable[float],
    catalog_seed: int,
    goal_state: tuple[float, float] = (190.0, 190.0),
) -> list[HiddenPassageTemplate]:
    templates: list[HiddenPassageTemplate] = []
    for door in map(int, door_positions):
        vertical_sign = _doorway_vertical_sign(door)
        for direction in directions:
            direction_sign = _direction_sign(direction)
            reset_x = 98.0 if direction_sign > 0 else 126.0
            for variant, offset in enumerate(doorway_offsets_px):
                offset = float(offset)
                reset_y = float(door) + vertical_sign * offset
                seed = int(
                    np.random.SeedSequence(
                        [
                            int(catalog_seed),
                            int(door),
                            DIRECTIONS.index(direction),
                            int(variant),
                        ]
                    ).generate_state(1)[0]
                )
                templates.append(
                    HiddenPassageTemplate(
                        template_id=(
                            f"hp-d{door:03d}-{direction}-v{variant:02d}"
                        ),
                        door_position=door,
                        direction=direction,
                        doorway_offset_px=offset,
                        reset_state=(reset_x, reset_y),
                        goal_state=tuple(map(float, goal_state)),
                        simulator_seed=seed,
                    )
                )
    return templates


def _variation_values(
    template: HiddenPassageTemplate,
    *,
    passage_open: int,
    agent_speed: float,
    door_number: int,
) -> dict[str, Any]:
    return {
        "agent.speed": np.asarray([agent_speed], dtype=np.float32),
        "door.number": int(door_number),
        "door.position": np.asarray(
            [template.door_position] * 3, dtype=np.int64
        ),
        PASSAGE_FACTOR: int(passage_open),
    }


def _step_block(env: Any, block: np.ndarray) -> np.ndarray:
    states: list[np.ndarray] = []
    for raw_step, action in enumerate(np.asarray(block, dtype=np.float32)):
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            raise RuntimeError(
                "Hidden-passage feasibility trajectory terminated at "
                f"raw step {raw_step}"
            )
        states.append(env.agent_position.detach().cpu().numpy().copy())
    return np.stack(states).astype(np.float32)


def simulate_template(
    template: HiddenPassageTemplate,
    *,
    rule: str,
    agent_speed: float = 5.0,
    door_number: int = 1,
) -> dict[str, Any]:
    if rule not in RULE_NAMES:
        raise ValueError(f"Unknown hidden passage rule {rule!r}")
    env = make_hidden_passage_env(render_mode="rgb_array")
    history_pixels: list[np.ndarray] = []
    history_states: list[np.ndarray] = []
    history_actions = np.stack(
        [
            _probe_block(template.direction),
            _recovery_block(template.door_position),
        ]
    ).astype(np.float32)
    query_action = _query_block(template.direction)
    try:
        observation, _ = env.reset(
            seed=int(template.simulator_seed),
            options={
                "variation": (),
                "variation_values": _variation_values(
                    template,
                    passage_open=PASSAGE_RULES[rule],
                    agent_speed=float(agent_speed),
                    door_number=int(door_number),
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

        probe_raw_states = _step_block(env, history_actions[0])
        history_pixels.append(np.asarray(env.render(), dtype=np.uint8).copy())
        history_states.append(
            env.agent_position.detach().cpu().numpy().copy()
        )

        recovery_raw_states = _step_block(env, history_actions[1])
        history_pixels.append(np.asarray(env.render(), dtype=np.uint8).copy())
        history_states.append(
            env.agent_position.detach().cpu().numpy().copy()
        )

        query_pixels = history_pixels[-1].copy()
        query_state = history_states[-1].copy()
        goal_pixels = (
            env._target_img.detach().cpu().numpy().transpose(1, 2, 0).copy()
        )
        query_raw_states = _step_block(env, query_action)
        target_pixels = np.asarray(env.render(), dtype=np.uint8).copy()
        target_state = env.agent_position.detach().cpu().numpy().copy()
        passage_readback = int(env.passage_open)
        door_number_readback = int(env.num_doors)
    finally:
        env.close()

    return {
        "rule": rule,
        "passage_open": passage_readback,
        "door_number": door_number_readback,
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
        "query_raw_states": query_raw_states,
        "goal_pixels": goal_pixels.astype(np.uint8),
        "goal_state": np.asarray(template.goal_state, dtype=np.float32),
    }


def replay_is_exact(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return all(
        np.array_equal(np.asarray(left[key]), np.asarray(right[key]))
        for key in REPLAY_ARRAY_KEYS
    )


def _maximum_zero_command_axis_displacement(
    rollout: dict[str, Any],
) -> float:
    history_actions = np.asarray(
        rollout["history_actions"], dtype=np.float32
    ).reshape(-1, 2)
    raw_states = np.asarray(
        rollout["history_raw_states"], dtype=np.float32
    )
    previous_states = np.concatenate(
        [
            np.asarray(rollout["history_states"][0], dtype=np.float32)[
                None, ...
            ],
            raw_states[:-1],
        ],
        axis=0,
    )
    displacement = np.abs(raw_states - previous_states)
    zero_command = np.isclose(history_actions, 0.0)
    if not zero_command.any():
        return 0.0
    return float(displacement[zero_command].max(initial=0.0))


def validate_pair(
    template: HiddenPassageTemplate,
    passable: dict[str, Any],
    blocked: dict[str, Any],
    *,
    minimum_middle_state_gap_px: float,
    minimum_future_state_gap_px: float,
) -> dict[str, Any]:
    middle_gap = float(
        np.linalg.norm(
            passable["history_states"][1] - blocked["history_states"][1]
        )
    )
    future_gap = float(
        np.linalg.norm(
            passable["target_state"] - blocked["target_state"]
        )
    )
    wall_center = 112.0
    sign = _direction_sign(template.direction)
    passable_crossed = bool(
        sign * (float(passable["target_state"][0]) - wall_center) > 0
    )
    blocked_crossed = bool(
        sign * (float(blocked["target_state"][0]) - wall_center) > 0
    )
    zero_command_displacement = {
        rule: _maximum_zero_command_axis_displacement(rollout)
        for rule, rollout in (
            ("passable", passable),
            ("blocked", blocked),
        )
    }
    checks = {
        "rule_readback": (
            int(passable["passage_open"]) == PASSAGE_RULES["passable"]
            and int(blocked["passage_open"]) == PASSAGE_RULES["blocked"]
        ),
        "door_number_readback": (
            int(passable["door_number"])
            == int(blocked["door_number"])
            == 1
        ),
        "initial_observation_identical": np.array_equal(
            passable["initial_observation"],
            blocked["initial_observation"],
        ),
        "initial_state_identical": np.array_equal(
            passable["history_states"][0],
            blocked["history_states"][0],
        ),
        "initial_pixels_identical": np.array_equal(
            passable["history_pixels"][0],
            blocked["history_pixels"][0],
        ),
        "history_actions_identical": np.array_equal(
            passable["history_actions"],
            blocked["history_actions"],
        ),
        "middle_pixels_different": not np.array_equal(
            passable["history_pixels"][1],
            blocked["history_pixels"][1],
        ),
        "middle_state_gap_sufficient": (
            middle_gap >= float(minimum_middle_state_gap_px)
        ),
        "query_state_identical": np.array_equal(
            passable["query_state"],
            blocked["query_state"],
        ),
        "query_pixels_identical": np.array_equal(
            passable["query_pixels"],
            blocked["query_pixels"],
        ),
        "history_third_frame_is_query": (
            np.array_equal(
                passable["history_pixels"][2], passable["query_pixels"]
            )
            and np.array_equal(
                blocked["history_pixels"][2], blocked["query_pixels"]
            )
            and np.array_equal(
                passable["history_states"][2], passable["query_state"]
            )
            and np.array_equal(
                blocked["history_states"][2], blocked["query_state"]
            )
        ),
        "query_actions_identical": np.array_equal(
            passable["query_action"],
            blocked["query_action"],
        ),
        "goal_state_identical": np.array_equal(
            passable["goal_state"], blocked["goal_state"]
        ),
        "goal_pixels_identical": np.array_equal(
            passable["goal_pixels"], blocked["goal_pixels"]
        ),
        "future_pixels_different": not np.array_equal(
            passable["target_pixels"], blocked["target_pixels"]
        ),
        "future_state_gap_sufficient": (
            future_gap >= float(minimum_future_state_gap_px)
        ),
        "passable_crosses_wall": passable_crossed,
        "blocked_does_not_cross_wall": not blocked_crossed,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "middle_state_gap_px": middle_gap,
        "future_state_gap_px": future_gap,
        "maximum_zero_command_axis_displacement_px": (
            zero_command_displacement
        ),
        "collision_projection_used_to_restore_query": bool(
            zero_command_displacement["passable"] > 0.0
        ),
        "passable_middle_state": passable["history_states"][1].tolist(),
        "blocked_middle_state": blocked["history_states"][1].tolist(),
        "query_state": passable["query_state"].tolist(),
        "passable_target_state": passable["target_state"].tolist(),
        "blocked_target_state": blocked["target_state"].tolist(),
        "history_actions_sha256": _array_sha256(
            passable["history_actions"]
        ),
        "query_action_sha256": _array_sha256(passable["query_action"]),
        "query_pixels_sha256": _array_sha256(passable["query_pixels"]),
    }


def _rule_arrays(
    rule: str, rollout: dict[str, Any]
) -> dict[str, np.ndarray]:
    return {
        f"{rule}_{key}": np.asarray(rollout[key])
        for key in REPLAY_ARRAY_KEYS
    }


def _serialized_payload_audit(
    payload_path: Path,
    rollouts: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    expected_arrays = {
        name: value
        for rule in RULE_NAMES
        for name, value in _rule_arrays(rule, rollouts[rule]).items()
    }
    with np.load(payload_path, allow_pickle=False) as payload:
        observed_names = set(payload.files)
        expected_names = set(expected_arrays)
        keys_exact = observed_names == expected_names
        arrays_exact = keys_exact and all(
            np.array_equal(payload[name], expected)
            for name, expected in expected_arrays.items()
        )
        serialized_rollouts = {
            rule: {
                key: np.asarray(payload[f"{rule}_{key}"]).copy()
                for key in REPLAY_ARRAY_KEYS
            }
            for rule in RULE_NAMES
        }

    projections = {
        rule: model_input_projection(serialized_rollouts[rule])
        for rule in RULE_NAMES
    }
    expected_projections = {
        rule: model_input_projection(rollouts[rule])
        for rule in RULE_NAMES
    }
    projection_keys_exact = all(
        tuple(projection) == MODEL_INPUT_KEYS
        for projection in projections.values()
    )
    projections_exact = all(
        np.array_equal(projections[rule][key], expected_projections[rule][key])
        for rule in RULE_NAMES
        for key in MODEL_INPUT_KEYS
    )
    action_signatures = {
        rule: _array_sha256(projections[rule]["action"])
        for rule in RULE_NAMES
    }
    projection_hashes = {
        rule: _payload_content_sha256(projections[rule])
        for rule in RULE_NAMES
    }
    checks = {
        "serialized_keys_exact": keys_exact,
        "serialized_arrays_roundtrip_exact": arrays_exact,
        "model_input_projection_keys_exact": projection_keys_exact,
        "model_input_projection_roundtrip_exact": projections_exact,
        "serialized_actions_identical_across_rules": (
            len(set(action_signatures.values())) == 1
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "expected_keys": sorted(expected_arrays),
        "action_signatures_by_rule": action_signatures,
        "model_input_projection_sha256_by_rule": projection_hashes,
    }


def _action_leakage_audit(
    bundles: list[dict[str, Any]],
) -> dict[str, Any]:
    signature_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    for bundle in bundles:
        for rule in RULE_NAMES:
            signature = str(
                bundle["action_signatures_by_rule"][rule]
            )
            signature_counts[signature][rule] += 1
    total = sum(sum(row.values()) for row in signature_counts.values())
    majority_correct = sum(max(row.values()) for row in signature_counts.values())
    balanced = all(
        set(row) == set(RULE_NAMES)
        and len(set(row.values())) == 1
        for row in signature_counts.values()
    )
    accuracy = float(majority_correct / total) if total else 1.0
    return {
        "passed": bool(total and balanced and accuracy == 0.5),
        "action_signatures": len(signature_counts),
        "rule_counts_by_signature": {
            key: dict(sorted(value.items()))
            for key, value in sorted(signature_counts.items())
        },
        "best_action_signature_only_accuracy": accuracy,
        "expected_chance_accuracy": 0.5,
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
    templates = make_templates(
        door_positions=geometry["door_positions"],
        directions=geometry["directions"],
        doorway_offsets_px=geometry["doorway_offsets_px"],
        catalog_seed=int(config["catalog_seed"]),
        goal_state=tuple(map(float, geometry["goal_state"])),
    )
    expected_pairs = int(config["counts"]["paired_templates"])
    if len(templates) != expected_pairs:
        raise ValueError(
            f"Expected {expected_pairs} templates, built {len(templates)}"
        )

    payload_root = output_root / "payloads"
    payload_root.mkdir(parents=True, exist_ok=True)
    bundles: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    query_hashes: set[str] = set()
    replay_passed = 0

    for template in templates:
        rollouts = {
            rule: simulate_template(
                template,
                rule=rule,
                agent_speed=float(protocol["agent_speed"]),
                door_number=int(protocol["door_number"]),
            )
            for rule in RULE_NAMES
        }
        replays = {
            rule: simulate_template(
                template,
                rule=rule,
                agent_speed=float(protocol["agent_speed"]),
                door_number=int(protocol["door_number"]),
            )
            for rule in RULE_NAMES
        }
        exact_replay = all(
            replay_is_exact(rollouts[rule], replays[rule])
            for rule in RULE_NAMES
        )
        replay_passed += int(exact_replay)
        validation = validate_pair(
            template,
            rollouts["passable"],
            rollouts["blocked"],
            minimum_middle_state_gap_px=float(
                thresholds["minimum_middle_state_gap_px"]
            ),
            minimum_future_state_gap_px=float(
                thresholds["minimum_future_state_gap_px"]
            ),
        )
        validation["checks"]["exact_replay"] = exact_replay
        validation["passed"] = all(validation["checks"].values())

        arrays: dict[str, np.ndarray] = {}
        for rule in RULE_NAMES:
            arrays.update(_rule_arrays(rule, rollouts[rule]))
        payload_path = payload_root / f"{template.template_id}.npz"
        np.savez_compressed(payload_path, **arrays)
        serialized_audit = _serialized_payload_audit(
            payload_path,
            rollouts,
        )
        validation["checks"].update(serialized_audit["checks"])
        validation["passed"] = all(validation["checks"].values())

        payload_reference = portable_contextworld_path(
            payload_path, repo_root=repo_root
        )
        resolved_payload = resolve_contextworld_path(
            payload_reference,
            repo_root=repo_root,
        )
        if resolved_payload != payload_path.resolve():
            raise RuntimeError(
                "Serialized payload does not resolve back to its output: "
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
            "rules": list(RULE_NAMES),
            "payload": payload_reference,
            "payload_sha256": _file_sha256(payload_path),
            "payload_content_sha256": _payload_content_sha256(arrays),
            "history_actions_sha256": validation[
                "history_actions_sha256"
            ],
            "query_action_sha256": validation["query_action_sha256"],
            "action_signatures_by_rule": serialized_audit[
                "action_signatures_by_rule"
            ],
            "model_input_projection_sha256_by_rule": serialized_audit[
                "model_input_projection_sha256_by_rule"
            ],
            "query_pixels_sha256": query_hash,
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

    action_leakage = _action_leakage_audit(bundles)
    query_uniqueness = len(query_hashes) == len(templates)
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
            "action_signatures_by_rule": bundle[
                "action_signatures_by_rule"
            ],
            "model_input_projection_sha256_by_rule": bundle[
                "model_input_projection_sha256_by_rule"
            ],
        }
        for bundle in bundles
    ]
    content_manifest_sha = _canonical_sha256(content_manifest)
    checks = {
        "all_template_pairs_pass": not failures,
        "exact_replay_all_templates": replay_passed == len(templates),
        "query_pixels_unique_across_templates": query_uniqueness,
        "action_signature_cannot_predict_rule": action_leakage["passed"],
        "direction_balance": len(set(by_direction.values())) == 1,
        "door_balance": len(set(by_door.values())) == 1,
        "model_visible_fields_exclude_rule": (
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
    catalog = {
        "schema_version": 1,
        "benchmark": str(config["benchmark"]),
        "status": str(config["status"]),
        "claim_limit": str(config["claim_limit"]),
        "protocol": protocol,
        "model_visible_fields": list(config["model_visible_fields"]),
        "privileged_audit_fields": list(config["privileged_audit_fields"]),
        "counts": {
            "paired_templates": len(templates),
            "rule_rollouts": len(templates) * len(RULE_NAMES),
            "by_direction": by_direction,
            "by_door_position": by_door,
        },
        "content_manifest_sha256": content_manifest_sha,
        "bundles": bundles,
    }
    report = {
        "schema_version": 1,
        "benchmark": str(config["benchmark"]),
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "counts": catalog["counts"],
        "thresholds": thresholds,
        "action_leakage_audit": action_leakage,
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
        "collision_projection": {
            "used_to_restore_identical_query": any(
                bundle["validation"][
                    "collision_projection_used_to_restore_query"
                ]
                for bundle in bundles
            ),
            "maximum_zero_command_axis_displacement_px": max(
                max(
                    bundle["validation"][
                        "maximum_zero_command_axis_displacement_px"
                    ].values()
                )
                for bundle in bundles
            ),
            "formal_training_approved": False,
            "next_gate": (
                "audit the pinned collision-specific History-3 transition "
                "in the World-to-Lance reload pilot, or move to a longer "
                "history with a more natural recovery"
            ),
        },
        "reuse_limits": {
            "formal_training_approved": False,
            "formal_planning_approved": False,
            "right_to_left_goal_direction_aligned": False,
            "reason": (
                "the feasibility goal is held constant for equality audits; "
                "fresh direction-aligned goals are required for planning"
            ),
        },
        "failed_templates": failures,
        "content_manifest_sha256": content_manifest_sha,
        "frozen_config_canonical_sha256": frozen_config_hash,
        "interpretation": {
            "passed_means": (
                "History-3 hidden-passage task is physically identifiable "
                "without query or action leakage"
            ),
            "does_not_mean": (
                "No model ICL claim is established before formal training "
                "and held-out evaluation"
            ),
        },
    }
    return catalog, report


__all__ = [
    "ACTION_BLOCK",
    "DIRECTIONS",
    "FROZEN_CONFIG_CANONICAL_SHA256",
    "HiddenPassageTemplate",
    "MODEL_INPUT_KEYS",
    "RULE_NAMES",
    "build_feasibility_catalog",
    "frozen_config_sha256",
    "make_templates",
    "model_input_projection",
    "replay_is_exact",
    "simulate_template",
    "validate_frozen_config",
    "validate_pair",
]
