from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from contextworld.paths import portable_contextworld_path

from .action_delay_env import (
    ACTION_DELAY_FACTOR,
    make_action_delay_env,
)


ACTION_BLOCK = 5
DELAY_VALUES = (0, 1, 2, 3, 4)
TRAIN_DELAY_VALUES = (0, 2, 4)
DIRECTIONS = ("up", "down")
MODEL_INPUT_KEYS = ("pixels", "action")
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
    "delay_steps",
)


@dataclass(frozen=True)
class ActionDelayTemplate:
    template_id: str
    direction: str
    reset_state: tuple[float, float]
    goal_state: tuple[float, float]
    simulator_seed: int


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(f"{array.dtype.str}:{array.shape}".encode("utf-8"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _direction_action(direction: str) -> np.ndarray:
    if direction == "up":
        return np.asarray([0.0, 1.0], dtype=np.float32)
    if direction == "down":
        return np.asarray([0.0, -1.0], dtype=np.float32)
    raise ValueError(f"Unknown action-delay direction {direction!r}")


def _action_block(direction: str) -> np.ndarray:
    return np.repeat(
        _direction_action(direction)[None, :],
        ACTION_BLOCK,
        axis=0,
    ).astype(np.float32)


def make_feasibility_templates(
    *,
    catalog_seed: int,
    x_positions: Iterable[float] = (40.0, 65.0, 160.0, 185.0),
    up_start_y: Iterable[float] = (25.0, 35.0, 45.0, 55.0),
    down_start_y: Iterable[float] = (200.0, 190.0, 180.0, 170.0),
) -> list[ActionDelayTemplate]:
    """Return 32 safe templates spanning both rooms and directions."""

    templates: list[ActionDelayTemplate] = []
    starts_by_direction = {
        "up": tuple(map(float, up_start_y)),
        "down": tuple(map(float, down_start_y)),
    }
    for direction_index, direction in enumerate(DIRECTIONS):
        for x_index, x_position in enumerate(map(float, x_positions)):
            goal = (
                (195.0, 205.0)
                if x_position < 112.0
                else (30.0, 20.0)
            )
            for y_index, y_position in enumerate(
                starts_by_direction[direction]
            ):
                seed = int(
                    np.random.SeedSequence(
                        [
                            int(catalog_seed),
                            direction_index,
                            x_index,
                            y_index,
                        ]
                    ).generate_state(1)[0]
                )
                templates.append(
                    ActionDelayTemplate(
                        template_id=(
                            f"ad-{direction}-x{x_index:02d}-y{y_index:02d}"
                        ),
                        direction=direction,
                        reset_state=(x_position, y_position),
                        goal_state=goal,
                        simulator_seed=seed,
                    )
                )
    if len(templates) != 32:
        raise RuntimeError(
            f"Expected 32 feasibility templates, got {len(templates)}"
        )
    if len({template.template_id for template in templates}) != len(
        templates
    ):
        raise RuntimeError("Action-delay feasibility template IDs repeat")
    return templates


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value).copy()


def _step_block(
    env: Any,
    block: np.ndarray,
) -> tuple[np.ndarray, bool]:
    states: list[np.ndarray] = []
    ended = False
    for action in np.asarray(block, dtype=np.float32):
        observation, _, terminated, truncated, _ = env.step(action)
        states.append(_as_numpy(observation)[:2])
        ended = ended or terminated or truncated
    return np.stack(states).astype(np.float32), ended


def simulate_template(
    template: ActionDelayTemplate,
    *,
    delay_steps: int,
    agent_speed: float = 7.0,
) -> dict[str, Any]:
    """Simulate the strict History=3 action-delay construction."""

    probe = _action_block(template.direction)
    flush = np.zeros_like(probe)
    env = make_action_delay_env(render_mode="rgb_array")
    history_pixels: list[np.ndarray] = []
    history_states: list[np.ndarray] = []
    try:
        initial_observation, _ = env.reset(
            seed=int(template.simulator_seed),
            options={
                "variation": (),
                "variation_values": {
                    "agent.speed": np.asarray(
                        [float(agent_speed)],
                        dtype=np.float32,
                    ),
                    ACTION_DELAY_FACTOR: int(delay_steps),
                },
                "state": np.asarray(
                    template.reset_state,
                    dtype=np.float32,
                ),
                "target_state": np.asarray(
                    template.goal_state,
                    dtype=np.float32,
                ),
            },
        )
        history_pixels.append(
            np.asarray(env.render(), dtype=np.uint8).copy()
        )
        history_states.append(_as_numpy(initial_observation)[:2])

        probe_states, probe_ended = _step_block(env, probe)
        history_pixels.append(
            np.asarray(env.render(), dtype=np.uint8).copy()
        )
        history_states.append(probe_states[-1].copy())

        flush_states, flush_ended = _step_block(env, flush)
        history_pixels.append(
            np.asarray(env.render(), dtype=np.uint8).copy()
        )
        history_states.append(flush_states[-1].copy())
        pending_at_query = env.pending_actions()

        query_states, query_ended = _step_block(env, probe)
        target_pixels = np.asarray(env.render(), dtype=np.uint8).copy()
        target_state = query_states[-1].copy()
        goal_pixels = (
            env._target_img.detach().cpu().numpy().transpose(1, 2, 0).copy()
        )
        delay_readback = int(env.action_delay_steps)
    finally:
        env.close()

    return {
        "delay_steps": np.asarray(delay_readback, dtype=np.int64),
        "initial_observation": _as_numpy(initial_observation).astype(
            np.float32
        ),
        "history_pixels": np.stack(history_pixels).astype(np.uint8),
        "history_states": np.stack(history_states).astype(np.float32),
        "history_raw_states": np.concatenate(
            [probe_states, flush_states],
            axis=0,
        ).astype(np.float32),
        "history_actions": np.stack([probe, flush]).astype(np.float32),
        "query_pixels": history_pixels[-1].copy(),
        "query_state": history_states[-1].copy(),
        "query_action": probe.copy(),
        "target_pixels": target_pixels,
        "target_state": target_state,
        "query_raw_states": query_states,
        "goal_pixels": goal_pixels.astype(np.uint8),
        "goal_state": np.asarray(template.goal_state, dtype=np.float32),
        "pending_actions_at_query": pending_at_query.astype(np.float32),
        "terminated_or_truncated": bool(
            probe_ended or flush_ended or query_ended
        ),
    }


def replay_is_exact(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return all(
        np.array_equal(np.asarray(left[key]), np.asarray(right[key]))
        for key in REPLAY_ARRAY_KEYS
    )


def model_input_projection(
    rollout: dict[str, Any],
) -> dict[str, np.ndarray]:
    return {
        "pixels": np.asarray(
            rollout["history_pixels"],
            dtype=np.uint8,
        ),
        "action": np.concatenate(
            [
                np.asarray(
                    rollout["history_actions"],
                    dtype=np.float32,
                ),
                np.asarray(
                    rollout["query_action"],
                    dtype=np.float32,
                )[None],
            ],
            axis=0,
        ),
    }


def validate_delay_family(
    template: ActionDelayTemplate,
    rollouts: dict[int, dict[str, Any]],
    *,
    agent_speed: float,
) -> dict[str, Any]:
    """Validate one five-delay family without assigning model scores."""

    if tuple(sorted(rollouts)) != DELAY_VALUES:
        raise ValueError(
            f"Expected delays {DELAY_VALUES}, got {tuple(sorted(rollouts))}"
        )
    ordered = [rollouts[delay] for delay in DELAY_VALUES]

    def all_equal(key: str) -> bool:
        values = [np.asarray(rollout[key]) for rollout in ordered]
        return all(
            np.array_equal(values[0], value) for value in values[1:]
        )

    def all_distinct(key: str) -> bool:
        values = [np.asarray(rollout[key]) for rollout in ordered]
        return all(
            not np.array_equal(left, right)
            for left_index, left in enumerate(values)
            for right in values[left_index + 1 :]
        )

    action = _direction_action(template.direction)
    initial = np.asarray(template.reset_state, dtype=np.float32)
    expected_query = initial + agent_speed * ACTION_BLOCK * action
    expected_middle = {
        delay: initial
        + agent_speed * (ACTION_BLOCK - delay) * action
        for delay in DELAY_VALUES
    }
    expected_future = {
        delay: expected_query
        + agent_speed * (ACTION_BLOCK - delay) * action
        for delay in DELAY_VALUES
    }
    exact_states = all(
        np.allclose(
            rollouts[delay]["history_states"][1],
            expected_middle[delay],
            atol=1e-6,
        )
        and np.allclose(
            rollouts[delay]["query_state"],
            expected_query,
            atol=1e-6,
        )
        and np.allclose(
            rollouts[delay]["target_state"],
            expected_future[delay],
            atol=1e-6,
        )
        for delay in DELAY_VALUES
    )
    queues_flushed = all(
        np.asarray(rollouts[delay]["pending_actions_at_query"]).shape
        == (delay, 2)
        and np.array_equal(
            rollouts[delay]["pending_actions_at_query"],
            np.zeros((delay, 2), dtype=np.float32),
        )
        for delay in DELAY_VALUES
    )
    replay_exact = all(
        replay_is_exact(
            rollouts[delay],
            simulate_template(
                template,
                delay_steps=delay,
                agent_speed=agent_speed,
            ),
        )
        for delay in DELAY_VALUES
    )
    checks = {
        "delay_readback_exact": all(
            int(rollouts[delay]["delay_steps"]) == delay
            for delay in DELAY_VALUES
        ),
        "initial_observation_identical": all_equal(
            "initial_observation"
        ),
        "initial_pixels_identical": all(
            np.array_equal(
                ordered[0]["history_pixels"][0],
                rollout["history_pixels"][0],
            )
            for rollout in ordered[1:]
        ),
        "history_actions_identical": all_equal("history_actions"),
        "query_action_identical": all_equal("query_action"),
        "query_state_identical": all_equal("query_state"),
        "query_pixels_identical": all_equal("query_pixels"),
        "history_midpoint_states_distinct": all(
            not np.array_equal(
                left["history_states"][1],
                right["history_states"][1],
            )
            for left_index, left in enumerate(ordered)
            for right in ordered[left_index + 1 :]
        ),
        "history_midpoint_pixels_distinct": all(
            not np.array_equal(
                left["history_pixels"][1],
                right["history_pixels"][1],
            )
            for left_index, left in enumerate(ordered)
            for right in ordered[left_index + 1 :]
        ),
        "target_states_distinct": all_distinct("target_state"),
        "target_pixels_distinct": all_distinct("target_pixels"),
        "analytical_states_exact": bool(exact_states),
        "pending_action_queues_flushed": bool(queues_flushed),
        "no_collision_or_early_termination": not any(
            rollout["terminated_or_truncated"] for rollout in ordered
        ),
        "deterministic_replay": bool(replay_exact),
        "model_projection_only_pixels_and_actions": all(
            tuple(model_input_projection(rollout)) == MODEL_INPUT_KEYS
            for rollout in ordered
        ),
    }
    middle_positions = {
        str(delay): rollouts[delay]["history_states"][1].tolist()
        for delay in DELAY_VALUES
    }
    future_positions = {
        str(delay): rollouts[delay]["target_state"].tolist()
        for delay in DELAY_VALUES
    }
    return {
        "template_id": template.template_id,
        "direction": template.direction,
        "passed": all(checks.values()),
        "checks": checks,
        "query_state": ordered[0]["query_state"].tolist(),
        "middle_states_by_delay": middle_positions,
        "future_states_by_delay": future_positions,
        "query_pixels_sha256": array_sha256(
            ordered[0]["query_pixels"]
        ),
        "history_pixels_sha256_by_delay": {
            str(delay): array_sha256(
                rollouts[delay]["history_pixels"]
            )
            for delay in DELAY_VALUES
        },
        "target_pixels_sha256_by_delay": {
            str(delay): array_sha256(
                rollouts[delay]["target_pixels"]
            )
            for delay in DELAY_VALUES
        },
    }


def build_feasibility_catalog(
    *,
    config: dict[str, Any],
    repo_root: Path,
    output_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = config["protocol"]
    if int(protocol["history_tokens"]) != 3:
        raise ValueError("Action-delay feasibility requires History=3")
    if int(protocol["raw_steps_per_action_block"]) != ACTION_BLOCK:
        raise ValueError("Action-delay action block must contain 5 raw steps")
    delays = tuple(map(int, protocol["delay_values"]))
    if delays != DELAY_VALUES:
        raise ValueError(
            f"Action-delay feasibility support must be {DELAY_VALUES}"
        )
    speed = float(protocol["agent_speed"])
    templates = make_feasibility_templates(
        catalog_seed=int(config["catalog_seed"])
    )
    family_reports = []
    catalog_rows = []
    for template in templates:
        rollouts = {
            delay: simulate_template(
                template,
                delay_steps=delay,
                agent_speed=speed,
            )
            for delay in DELAY_VALUES
        }
        family = validate_delay_family(
            template,
            rollouts,
            agent_speed=speed,
        )
        family_reports.append(family)
        catalog_rows.append(
            {
                "template": asdict(template),
                "query_state": family["query_state"],
                "query_pixels_sha256": family["query_pixels_sha256"],
                "history_pixels_sha256_by_delay": family[
                    "history_pixels_sha256_by_delay"
                ],
                "target_pixels_sha256_by_delay": family[
                    "target_pixels_sha256_by_delay"
                ],
            }
        )
    unique_query_pixels = len(
        {row["query_pixels_sha256"] for row in catalog_rows}
    )
    projection = {
        "benchmark": config["benchmark"],
        "protocol": {
            "history_tokens": 3,
            "raw_steps_per_action_block": ACTION_BLOCK,
            "agent_speed": speed,
            "delay_values": list(DELAY_VALUES),
        },
        "rows": catalog_rows,
    }
    content_sha256 = canonical_sha256(projection)
    catalog = {
        "schema_version": 1,
        **projection,
        "content_manifest_sha256": content_sha256,
    }
    checks = {
        "exact_template_count": len(templates)
        == int(config["counts"]["paired_templates"]),
        "exact_rollout_count": len(templates) * len(DELAY_VALUES)
        == int(config["counts"]["delay_rollouts"]),
        "all_five_delay_families_pass": all(
            family["passed"] for family in family_reports
        ),
        "query_pixels_unique_across_templates": unique_query_pixels
        == len(templates),
        "model_visible_fields_exact": tuple(
            config["model_visible_fields"]
        )
        == MODEL_INPUT_KEYS,
    }
    report = {
        "schema_version": 1,
        "benchmark": config["benchmark"],
        "status": "passed" if all(checks.values()) else "failed",
        "claim_limit": config["claim_limit"],
        "checks": checks,
        "counts": {
            "paired_templates": len(templates),
            "delay_rollouts": len(templates) * len(DELAY_VALUES),
            "unique_query_pixels": unique_query_pixels,
        },
        "content_manifest_sha256": content_sha256,
        "families": family_reports,
        "output_root": portable_contextworld_path(
            output_root,
            repo_root=repo_root,
        ),
    }
    return catalog, report


__all__ = [
    "ACTION_BLOCK",
    "DELAY_VALUES",
    "DIRECTIONS",
    "MODEL_INPUT_KEYS",
    "TRAIN_DELAY_VALUES",
    "ActionDelayTemplate",
    "array_sha256",
    "build_feasibility_catalog",
    "canonical_sha256",
    "make_feasibility_templates",
    "model_input_projection",
    "replay_is_exact",
    "simulate_template",
    "validate_delay_family",
]
