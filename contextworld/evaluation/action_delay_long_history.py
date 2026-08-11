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
    make_extended_action_delay_env,
)


ACTION_BLOCK = 5
CANDIDATE_HISTORY_TOKENS = (5, 6, 7, 9)
DELAY_VALUES = (0, 2, 4, 6, 8, 10)
FUTURE_HORIZONS = (1, 2, 3)
MODEL_VISIBLE_FIELDS = ("pixels", "action")


@dataclass(frozen=True)
class LongHistoryDelayTemplate:
    template_id: str
    direction: str
    reset_state: tuple[float, float]
    goal_state: tuple[float, float]
    simulator_seed: int


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(f"{array.dtype.str}:{array.shape}".encode("utf-8"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value).copy()


def _all_equal(values: Iterable[np.ndarray]) -> bool:
    arrays = [np.asarray(value) for value in values]
    return bool(
        arrays
        and all(np.array_equal(arrays[0], value) for value in arrays[1:])
    )


def _all_pairwise_distinct(values: Iterable[np.ndarray]) -> bool:
    arrays = [np.asarray(value) for value in values]
    return bool(
        arrays
        and all(
            not np.array_equal(left, right)
            for left_index, left in enumerate(arrays)
            for right in arrays[left_index + 1 :]
        )
    )


def make_templates(
    *,
    catalog_seed: int,
    x_positions: Iterable[float] = (45.0, 70.0, 154.0, 179.0),
    starts_per_direction: int = 3,
) -> list[LongHistoryDelayTemplate]:
    """Build collision-free templates in both rooms and both directions."""

    templates: list[LongHistoryDelayTemplate] = []
    for direction_index, direction in enumerate(("up", "down")):
        sign = 1.0 if direction == "up" else -1.0
        base_y = 45.0 if direction == "up" else 179.0
        for x_index, x_position in enumerate(map(float, x_positions)):
            for offset_index in range(int(starts_per_direction)):
                y_position = base_y + sign * 10.0 * offset_index
                goal_state = (
                    (190.0, 190.0)
                    if x_position < 112.0
                    else (30.0, 30.0)
                )
                simulator_seed = int(
                    np.random.SeedSequence(
                        [
                            int(catalog_seed),
                            direction_index,
                            x_index,
                            offset_index,
                        ]
                    ).generate_state(1)[0]
                )
                templates.append(
                    LongHistoryDelayTemplate(
                        template_id=(
                            f"ad-long-{direction}-x{x_index:02d}-"
                            f"y{offset_index:02d}"
                        ),
                        direction=direction,
                        reset_state=(x_position, y_position),
                        goal_state=goal_state,
                        simulator_seed=simulator_seed,
                    )
                )
    if len({value.template_id for value in templates}) != len(templates):
        raise RuntimeError("Long-history delay template IDs repeat")
    return templates


def _probe_block(
    direction: str,
    *,
    action_magnitude: float,
) -> np.ndarray:
    sign = 1.0 if direction == "up" else -1.0
    action = np.asarray(
        [0.0, sign * float(action_magnitude)],
        dtype=np.float32,
    )
    return np.repeat(action[None], ACTION_BLOCK, axis=0)


def history_action_blocks(
    *,
    history_tokens: int,
    direction: str,
    action_magnitude: float,
) -> np.ndarray:
    """Return probe, wait, inverse probe, then zero flush blocks."""

    history_tokens = int(history_tokens)
    if history_tokens < 4:
        raise ValueError("Long action-delay history needs at least 4 frames")
    probe = _probe_block(
        direction,
        action_magnitude=action_magnitude,
    )
    zero = np.zeros_like(probe)
    blocks = [probe, zero, -probe]
    blocks.extend(zero.copy() for _ in range(history_tokens - 4))
    result = np.stack(blocks).astype(np.float32)
    expected = (history_tokens - 1, ACTION_BLOCK, 2)
    if result.shape != expected:
        raise RuntimeError(
            f"Long-history action shape mismatch: {result.shape} != {expected}"
        )
    return result


def future_action_blocks(
    *,
    direction: str,
    action_magnitude: float,
) -> np.ndarray:
    probe = _probe_block(
        direction,
        action_magnitude=action_magnitude,
    )
    return np.repeat(
        probe[None],
        len(FUTURE_HORIZONS),
        axis=0,
    ).astype(np.float32)


def simulate_template(
    template: LongHistoryDelayTemplate,
    *,
    history_tokens: int,
    delay_steps: int,
    agent_speed: float,
    action_magnitude: float,
    maximum_delay_steps: int,
) -> dict[str, Any]:
    """Run one physically continuous history and three-step true future."""

    history_blocks = history_action_blocks(
        history_tokens=history_tokens,
        direction=template.direction,
        action_magnitude=action_magnitude,
    )
    future_blocks = future_action_blocks(
        direction=template.direction,
        action_magnitude=action_magnitude,
    )
    all_blocks = np.concatenate(
        [history_blocks, future_blocks],
        axis=0,
    )
    raw_commands = all_blocks.reshape(-1, 2)
    env = make_extended_action_delay_env(
        max_delay_steps=maximum_delay_steps,
        render_mode="rgb_array",
    )
    history_states: list[np.ndarray] = []
    history_pixels: list[np.ndarray] = []
    future_states: list[np.ndarray] = []
    future_pixels: list[np.ndarray] = []
    raw_states: list[np.ndarray] = []
    executed_actions: list[np.ndarray] = []
    ended = False
    pending_at_query: np.ndarray | None = None
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
        history_states.append(_as_numpy(initial_observation)[:2])
        history_pixels.append(
            np.asarray(env.render(), dtype=np.uint8).copy()
        )
        history_raw_steps = (history_tokens - 1) * ACTION_BLOCK
        for raw_index, command in enumerate(raw_commands):
            observation, _, terminated, truncated, info = env.step(command)
            raw_state = _as_numpy(observation)[:2]
            raw_states.append(raw_state)
            executed_actions.append(
                np.asarray(
                    info["contextworld.executed_action"],
                    dtype=np.float32,
                ).copy()
            )
            ended = ended or terminated or truncated
            if (raw_index + 1) % ACTION_BLOCK:
                continue
            if raw_index + 1 <= history_raw_steps:
                history_states.append(raw_state.copy())
                history_pixels.append(
                    np.asarray(env.render(), dtype=np.uint8).copy()
                )
                if raw_index + 1 == history_raw_steps:
                    pending_at_query = env.pending_actions()
            else:
                future_states.append(raw_state.copy())
                future_pixels.append(
                    np.asarray(env.render(), dtype=np.uint8).copy()
                )
        delay_readback = int(env.action_delay_steps)
    finally:
        env.close()

    executed = np.stack(executed_actions).astype(np.float32)
    zero = np.zeros(2, dtype=np.float32)
    expected_executed = np.stack(
        [
            (
                zero
                if raw_index < delay_steps
                else raw_commands[raw_index - delay_steps]
            )
            for raw_index in range(len(raw_commands))
        ]
    ).astype(np.float32)
    initial_state = np.asarray(template.reset_state, dtype=np.float32)
    expected_raw_states = (
        initial_state[None]
        + float(agent_speed) * np.cumsum(expected_executed, axis=0)
    ).astype(np.float32)
    raw_state_array = np.stack(raw_states).astype(np.float32)
    action_blocks = np.concatenate(
        [history_blocks, future_blocks],
        axis=0,
    ).astype(np.float32)
    if pending_at_query is None:
        raise RuntimeError("Long-history query boundary was not observed")
    return {
        "history_tokens": int(history_tokens),
        "delay_steps": int(delay_readback),
        "history_states": np.stack(history_states).astype(np.float32),
        "history_pixels": np.stack(history_pixels).astype(np.uint8),
        "action_blocks": action_blocks,
        "future_states": np.stack(future_states).astype(np.float32),
        "future_pixels": np.stack(future_pixels).astype(np.uint8),
        "pending_actions_at_query": pending_at_query.astype(np.float32),
        "executed_actions": executed,
        "expected_executed_actions": expected_executed,
        "raw_states": raw_state_array,
        "expected_raw_states": expected_raw_states,
        "terminated_or_truncated": bool(ended),
    }


def validate_history_candidate(
    template: LongHistoryDelayTemplate,
    rollouts: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    ordered_delays = tuple(sorted(rollouts))
    if ordered_delays != DELAY_VALUES:
        raise ValueError(
            f"Expected delays {DELAY_VALUES}, got {ordered_delays}"
        )
    ordered = [rollouts[value] for value in ordered_delays]
    history_tokens = int(ordered[0]["history_tokens"])
    histories = [value["history_states"] for value in ordered]
    history_pixels = [value["history_pixels"] for value in ordered]
    query_states = [value["history_states"][-1] for value in ordered]
    query_pixels = [value["history_pixels"][-1] for value in ordered]
    final_transition_stable = [
        np.array_equal(
            value["history_states"][-2],
            value["history_states"][-1],
        )
        for value in ordered
    ]
    future_distinct = {
        str(horizon): _all_pairwise_distinct(
            value["future_states"][horizon - 1] for value in ordered
        )
        for horizon in FUTURE_HORIZONS
    }
    queues_zero = all(
        value["pending_actions_at_query"].shape == (delay, 2)
        and np.array_equal(
            value["pending_actions_at_query"],
            np.zeros((delay, 2), dtype=np.float32),
        )
        for delay, value in zip(ordered_delays, ordered)
    )
    checks = {
        "delay_readback_exact": all(
            int(value["delay_steps"]) == delay
            for delay, value in zip(ordered_delays, ordered)
        ),
        "initial_state_identical": _all_equal(
            value[0] for value in histories
        ),
        "initial_pixels_identical": _all_equal(
            value[0] for value in history_pixels
        ),
        "history_actions_identical": _all_equal(
            value["action_blocks"] for value in ordered
        ),
        "history_trajectory_distinguishes_all_delays": (
            _all_pairwise_distinct(histories)
            and _all_pairwise_distinct(history_pixels)
        ),
        "query_state_identical": _all_equal(query_states),
        "query_pixels_identical": _all_equal(query_pixels),
        "pending_queue_empty": bool(queues_zero),
        "horizon2_true_states_distinguish_all_delays": future_distinct["2"],
        "horizon3_true_states_distinguish_all_delays": future_distinct["3"],
        "executed_action_trace_exact": all(
            np.array_equal(
                value["executed_actions"],
                value["expected_executed_actions"],
            )
            for value in ordered
        ),
        "analytical_state_trace_exact": all(
            np.allclose(
                value["raw_states"],
                value["expected_raw_states"],
                atol=1e-6,
            )
            for value in ordered
        ),
        "no_collision_or_early_termination": not any(
            value["terminated_or_truncated"] for value in ordered
        ),
        "model_projection_is_pixels_and_actions_only": (
            MODEL_VISIBLE_FIELDS == ("pixels", "action")
        ),
    }
    physical_alignment_passed = all(checks.values())
    robust_query_boundary_passed = bool(
        physical_alignment_passed and all(final_transition_stable)
    )
    return {
        "history_tokens": history_tokens,
        "physical_alignment_passed": bool(physical_alignment_passed),
        "robust_query_boundary_passed": robust_query_boundary_passed,
        "checks": checks,
        "final_transition_stable_by_delay": {
            str(delay): bool(value)
            for delay, value in zip(
                ordered_delays,
                final_transition_stable,
            )
        },
        "future_states_pairwise_distinct_by_horizon": future_distinct,
        "history_state_signatures": {
            str(delay): rollout["history_states"].tolist()
            for delay, rollout in zip(ordered_delays, ordered)
        },
        "future_states": {
            str(delay): rollout["future_states"].tolist()
            for delay, rollout in zip(ordered_delays, ordered)
        },
        "query_state": ordered[0]["history_states"][-1].tolist(),
        "query_pixels_sha256": (
            _array_sha256(ordered[0]["history_pixels"][-1])
            if _all_equal(query_pixels)
            else None
        ),
    }


def build_long_history_feasibility(
    *,
    config: dict[str, Any],
    repo_root: Path,
    output_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = config["protocol"]
    candidates = tuple(map(int, protocol["candidate_history_tokens"]))
    delays = tuple(map(int, protocol["delay_values"]))
    horizons = tuple(map(int, protocol["future_horizons_action_blocks"]))
    if candidates != CANDIDATE_HISTORY_TOKENS:
        raise ValueError(
            "Long-history candidates must be "
            f"{CANDIDATE_HISTORY_TOKENS}, got {candidates}"
        )
    if delays != DELAY_VALUES:
        raise ValueError(
            f"Long-history delays must be {DELAY_VALUES}, got {delays}"
        )
    if horizons != FUTURE_HORIZONS:
        raise ValueError(
            f"Long-history horizons must be {FUTURE_HORIZONS}, got {horizons}"
        )
    if int(protocol["raw_steps_per_action_block"]) != ACTION_BLOCK:
        raise ValueError("Long-history action block must contain 5 raw steps")
    maximum_delay = int(protocol["maximum_delay_steps"])
    if maximum_delay != max(DELAY_VALUES):
        raise ValueError("maximum_delay_steps must equal the largest delay")
    agent_speed = float(protocol["agent_speed"])
    action_magnitude = float(protocol["action_magnitude"])
    templates = make_templates(
        catalog_seed=int(config["catalog_seed"]),
        starts_per_direction=int(config["counts"]["starts_per_direction"]),
    )
    reports_by_candidate: dict[str, list[dict[str, Any]]] = {
        str(value): [] for value in candidates
    }
    catalog_rows: list[dict[str, Any]] = []
    for template in templates:
        row = {"template": asdict(template), "candidates": {}}
        for history_tokens in candidates:
            rollouts = {
                delay: simulate_template(
                    template,
                    history_tokens=history_tokens,
                    delay_steps=delay,
                    agent_speed=agent_speed,
                    action_magnitude=action_magnitude,
                    maximum_delay_steps=maximum_delay,
                )
                for delay in DELAY_VALUES
            }
            candidate_report = validate_history_candidate(
                template,
                rollouts,
            )
            reports_by_candidate[str(history_tokens)].append(
                candidate_report
            )
            row["candidates"][str(history_tokens)] = {
                "physical_alignment_passed": candidate_report[
                    "physical_alignment_passed"
                ],
                "robust_query_boundary_passed": candidate_report[
                    "robust_query_boundary_passed"
                ],
                "query_state": candidate_report["query_state"],
                "query_pixels_sha256": candidate_report[
                    "query_pixels_sha256"
                ],
            }
        catalog_rows.append(row)

    candidate_summary = {}
    for history_tokens in candidates:
        rows = reports_by_candidate[str(history_tokens)]
        candidate_summary[str(history_tokens)] = {
            "templates": len(rows),
            "physical_alignment_passed": bool(
                rows
                and all(
                    value["physical_alignment_passed"] for value in rows
                )
            ),
            "robust_query_boundary_passed": bool(
                rows
                and all(
                    value["robust_query_boundary_passed"] for value in rows
                )
            ),
            "all_delays_have_stable_final_transition": bool(
                rows
                and all(
                    all(
                        value[
                            "final_transition_stable_by_delay"
                        ].values()
                    )
                    for value in rows
                )
            ),
        }
    physical_candidates = [
        value
        for value in candidates
        if candidate_summary[str(value)]["physical_alignment_passed"]
    ]
    robust_candidates = [
        value
        for value in candidates
        if candidate_summary[str(value)]["robust_query_boundary_passed"]
    ]
    minimum_physical = min(physical_candidates, default=None)
    selected = min(robust_candidates, default=None)
    expected_minimum = int(config["selection"]["expected_physical_minimum"])
    expected_selected = int(config["selection"]["expected_formal_history"])
    selection_checks = {
        "physical_minimum_is_expected": minimum_physical
        == expected_minimum,
        "formal_history_is_expected": selected == expected_selected,
        "formal_history_adds_one_stable_transition": (
            selected is not None
            and minimum_physical is not None
            and selected == minimum_physical + 1
        ),
        "longer_candidates_add_no_new_identifiability_requirement": all(
            candidate_summary[str(value)][
                "robust_query_boundary_passed"
            ]
            for value in candidates
            if selected is not None and value >= selected
        ),
    }
    projection = {
        "schema_version": 1,
        "benchmark": config["benchmark"],
        "protocol": {
            "candidate_history_tokens": list(candidates),
            "delay_values": list(delays),
            "future_horizons_action_blocks": list(horizons),
            "raw_steps_per_action_block": ACTION_BLOCK,
            "agent_speed": agent_speed,
            "action_magnitude": action_magnitude,
        },
        "selection": {
            "physical_minimum_history_tokens": minimum_physical,
            "formal_history_tokens": selected,
            "reason": (
                "the formal history adds one all-zero transition after the "
                "latest delayed recovery command has executed"
            ),
        },
        "rows": catalog_rows,
    }
    content_sha256 = _canonical_sha256(projection)
    catalog = {
        **projection,
        "content_manifest_sha256": content_sha256,
    }
    global_checks = {
        "exact_template_count": len(templates)
        == int(config["counts"]["paired_templates"]),
        "all_candidate_families_have_expected_size": all(
            len(reports_by_candidate[str(value)]) == len(templates)
            for value in candidates
        ),
        "model_visible_fields_exact": tuple(config["model_visible_fields"])
        == MODEL_VISIBLE_FIELDS,
        **selection_checks,
    }
    report = {
        "schema_version": 1,
        "benchmark": config["benchmark"],
        "status": "passed" if all(global_checks.values()) else "failed",
        "claim_limit": config["claim_limit"],
        "checks": global_checks,
        "candidate_summary": candidate_summary,
        "selection": projection["selection"],
        "counts": {
            "paired_templates": len(templates),
            "candidate_histories": len(candidates),
            "delay_rollouts": (
                len(templates) * len(candidates) * len(DELAY_VALUES)
            ),
        },
        "families_by_candidate": reports_by_candidate,
        "content_manifest_sha256": content_sha256,
        "output_root": portable_contextworld_path(
            output_root,
            repo_root=repo_root,
        ),
    }
    return catalog, report


__all__ = [
    "ACTION_BLOCK",
    "CANDIDATE_HISTORY_TOKENS",
    "DELAY_VALUES",
    "FUTURE_HORIZONS",
    "LongHistoryDelayTemplate",
    "build_long_history_feasibility",
    "future_action_blocks",
    "history_action_blocks",
    "make_templates",
    "simulate_template",
    "validate_history_candidate",
]
