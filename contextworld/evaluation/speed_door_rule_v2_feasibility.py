from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

import numpy as np

from .hidden_passage_env import (
    PASSAGE_FACTOR,
    PASSAGE_RULES,
    make_hidden_passage_env,
)
from .speed_door_rule_validation import array_sha256
from .speed_identifiability import agent_centroid_from_rgb
from .speed_door_rule_v2_design import (
    require_valid_speed_door_rule_v2_design,
)


RULES = ("passable", "blocked")


def _direction_sign(direction: str) -> float:
    if direction == "left_to_right":
        return 1.0
    if direction == "right_to_left":
        return -1.0
    raise ValueError(f"Unknown direction {direction!r}")


def future_action_blocks(
    direction: str,
    config: dict[str, Any],
) -> np.ndarray:
    protocol = config["protocol"]
    future = protocol["future_action"]
    blocks = np.zeros(
        (
            int(future["action_blocks"]),
            int(protocol["raw_steps_per_action_block"]),
            2,
        ),
        dtype=np.float32,
    )
    blocks[:, :, 0] = np.float32(
        _direction_sign(direction)
        * float(future["horizontal_value"])
    )
    blocks[:, :, 1] = np.float32(future["vertical_value"])
    return blocks


def _variation_values(
    *,
    speed: float,
    rule: str,
    door_position: int,
) -> dict[str, Any]:
    return {
        "agent.speed": np.asarray([speed], dtype=np.float32),
        "door.number": 1,
        "door.position": np.asarray(
            [door_position] * 3, dtype=np.int64
        ),
        PASSAGE_FACTOR: int(PASSAGE_RULES[rule]),
    }


def simulate_v2_future(
    env: Any,
    *,
    bundle: dict[str, Any],
    speed: float,
    rule: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    template = bundle["template"]
    query_state = np.asarray(
        bundle["validation"]["query_state"], dtype=np.float32
    )
    goal_state = np.asarray(template["goal_state"], dtype=np.float32)
    observation, _ = env.reset(
        seed=int(template["simulator_seed"]),
        options={
            "variation": (),
            "variation_values": _variation_values(
                speed=speed,
                rule=rule,
                door_position=int(template["door_position"]),
            ),
            "state": query_state,
            "target_state": goal_state,
        },
    )
    query_pixels = np.asarray(env.render(), dtype=np.uint8).copy()
    action_blocks = future_action_blocks(
        str(template["direction"]), config
    )
    targets: dict[int, dict[str, Any]] = {}
    terminated_or_truncated = False
    for horizon_index, block in enumerate(action_blocks, start=1):
        raw_states = []
        for action in block:
            _, _, terminated, truncated, _ = env.step(action)
            terminated_or_truncated |= bool(terminated or truncated)
            raw_states.append(
                env.agent_position.detach().cpu().numpy().copy()
            )
        targets[horizon_index] = {
            "pixels": np.asarray(env.render(), dtype=np.uint8).copy(),
            "state": (
                env.agent_position.detach().cpu().numpy().copy().astype(
                    np.float32
                )
            ),
            "raw_states": np.stack(raw_states).astype(np.float32),
        }
    return {
        "query_observation": np.asarray(
            observation, dtype=np.float32
        ).copy(),
        "query_pixels": query_pixels,
        "query_state": query_state,
        "action_blocks": action_blocks,
        "targets": targets,
        "terminated_or_truncated": terminated_or_truncated,
    }


def _all_equal(values: list[np.ndarray]) -> bool:
    return all(np.array_equal(values[0], value) for value in values[1:])


def _minimum(values: list[float]) -> float:
    return min(values) if values else float("inf")


def validate_v2_training_factor_grid(
    template: Any,
    rollouts: dict[tuple[float, str], dict[str, Any]],
    *,
    speeds: tuple[float, ...],
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    """Validate the one-step training analogue of the v2 physical contract.

    History evidence retains the existing probe/recovery construction.  The
    query action is pure horizontal motion: passable futures remain
    speed-specific, while all blocked futures collapse to one physical state.
    """

    expected = {
        (float(speed), rule) for speed in speeds for rule in RULES
    }
    if set(rollouts) != expected:
        raise ValueError(
            f"Expected v2 factor grid {sorted(expected)}, "
            f"got {sorted(rollouts)}"
        )
    if not speeds or tuple(sorted(speeds)) != tuple(speeds):
        raise ValueError("v2 training speeds must be non-empty and increasing")
    direction_sign = _direction_sign(str(template.direction))
    vertical_sign = (
        1.0 if int(template.door_position) <= 112 else -1.0
    )
    first = next(iter(rollouts.values()))

    middle_centroids = {
        key: agent_centroid_from_rgb(
            np.asarray(value["history_pixels"])[1]
        )
        for key, value in rollouts.items()
    }
    target_centroids = {
        key: agent_centroid_from_rgb(value["target_pixels"])
        for key, value in rollouts.items()
    }
    target_states = {
        key: np.asarray(value["target_state"], dtype=np.float64)
        for key, value in rollouts.items()
    }
    middle_rule_gaps = [
        float(
            direction_sign
            * (
                middle_centroids[(speed, "passable")][0]
                - middle_centroids[(speed, "blocked")][0]
            )
        )
        for speed in speeds
    ]
    middle_speed_gaps = [
        float(
            vertical_sign
            * (
                middle_centroids[(higher, rule)][1]
                - middle_centroids[(lower, rule)][1]
            )
        )
        for rule in RULES
        for lower, higher in zip(speeds[:-1], speeds[1:])
    ]
    future_rule_gaps = [
        float(
            direction_sign
            * (
                target_states[(speed, "passable")][0]
                - target_states[(speed, "blocked")][0]
            )
        )
        for speed in speeds
    ]
    passable_speed_gaps = [
        float(
            direction_sign
            * (
                target_centroids[(higher, "passable")][0]
                - target_centroids[(lower, "passable")][0]
            )
        )
        for lower, higher in zip(speeds[:-1], speeds[1:])
    ]
    blocked_states = [
        target_states[(speed, "blocked")] for speed in speeds
    ]
    blocked_pixels = [
        np.asarray(rollouts[(speed, "blocked")]["target_pixels"])
        for speed in speeds
    ]
    passable_pixels = [
        np.asarray(rollouts[(speed, "passable")]["target_pixels"])
        for speed in speeds
    ]
    query_action = np.asarray(first["query_action"], dtype=np.float32)
    query_state = np.asarray(first["query_state"], dtype=np.float64)
    checks = {
        "factor_grid_complete": True,
        "speed_readback_exact": all(
            np.isclose(float(value["agent_speed"]), speed, atol=1.0e-6)
            for (speed, _), value in rollouts.items()
        ),
        "rule_readback_exact": all(
            int(value["passage_open"]) == PASSAGE_RULES[rule]
            for (_, rule), value in rollouts.items()
        ),
        "door_number_readback_exact": all(
            int(value["door_number"]) == 1
            for value in rollouts.values()
        ),
        "initial_observation_identical": _all_equal(
            [
                np.asarray(value["initial_observation"])
                for value in rollouts.values()
            ]
        ),
        "initial_pixels_identical": _all_equal(
            [
                np.asarray(value["history_pixels"])[0]
                for value in rollouts.values()
            ]
        ),
        "history_actions_identical": _all_equal(
            [
                np.asarray(value["history_actions"])
                for value in rollouts.values()
            ]
        ),
        "query_state_identical": _all_equal(
            [
                np.asarray(value["query_state"])
                for value in rollouts.values()
            ]
        ),
        "query_pixels_identical": _all_equal(
            [
                np.asarray(value["query_pixels"])
                for value in rollouts.values()
            ]
        ),
        "query_actions_identical": _all_equal(
            [
                np.asarray(value["query_action"])
                for value in rollouts.values()
            ]
        ),
        "goal_state_identical": _all_equal(
            [
                np.asarray(value["goal_state"])
                for value in rollouts.values()
            ]
        ),
        "goal_pixels_identical": _all_equal(
            [
                np.asarray(value["goal_pixels"])
                for value in rollouts.values()
            ]
        ),
        "history_third_frame_is_query": all(
            np.array_equal(
                value["history_pixels"][2], value["query_pixels"]
            )
            and np.array_equal(
                value["history_states"][2], value["query_state"]
            )
            for value in rollouts.values()
        ),
        "query_action_is_five_horizontal_steps": bool(
            query_action.shape == (5, 2)
            and np.all(np.abs(query_action[:, 0]) > 0)
            and np.all(query_action[:, 1] == 0)
        ),
        "middle_pixels_unique_by_factor": len(
            {
                array_sha256(value["history_pixels"][1])
                for value in rollouts.values()
            }
        )
        == len(expected),
        "middle_rule_gap_sufficient": _minimum(middle_rule_gaps)
        >= float(thresholds["minimum_middle_rule_centroid_gap_px"]),
        "middle_speed_gap_sufficient": _minimum(middle_speed_gaps)
        >= float(
            thresholds["minimum_middle_adjacent_speed_centroid_gap_px"]
        ),
        "blocked_future_states_shared": _all_equal(blocked_states),
        "blocked_future_pixels_shared": _all_equal(blocked_pixels),
        "blocked_future_is_query_contact": all(
            np.array_equal(state, query_state) for state in blocked_states
        ),
        "passable_future_speed_ordered": all(
            gap > 0 for gap in passable_speed_gaps
        ),
        "passable_future_speed_gap_sufficient": (
            _minimum(passable_speed_gaps)
            >= float(
                thresholds[
                    "minimum_future_adjacent_speed_centroid_gap_px"
                ]
            )
        ),
        "future_rule_gap_sufficient": _minimum(future_rule_gaps)
        >= float(thresholds["minimum_future_rule_state_gap_px"]),
        "physical_future_class_count_exact": len(
            {
                array_sha256(value)
                for value in [blocked_pixels[0], *passable_pixels]
            }
        )
        == len(speeds) + 1,
        "future_vertical_position_unchanged": all(
            np.isclose(
                state[1],
                np.asarray(
                    rollouts[key]["query_state"], dtype=np.float64
                )[1],
                atol=1.0e-6,
            )
            for key, state in target_states.items()
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "minimum_middle_rule_centroid_gap_px": _minimum(
            middle_rule_gaps
        ),
        "minimum_middle_adjacent_speed_centroid_gap_px": _minimum(
            middle_speed_gaps
        ),
        "minimum_future_rule_state_gap_px": _minimum(
            future_rule_gaps
        ),
        "minimum_future_adjacent_speed_centroid_gap_px": _minimum(
            passable_speed_gaps
        ),
        "blocked_future_state_spread_px": max(
            float(np.linalg.norm(left - right))
            for left in blocked_states
            for right in blocked_states
        ),
        "physical_future_classes": len(speeds) + 1,
        "query_state": query_state.tolist(),
    }


def audit_v2_query_bundles(
    *,
    config: dict[str, Any],
    bundles: Iterable[dict[str, Any]],
    require_full_catalog: bool = True,
) -> dict[str, Any]:
    """Replay proposed h1/h2 futures for frozen v1 query assignments."""

    bundles = tuple(bundles)
    if not bundles:
        raise ValueError("Cannot audit an empty v2 query set")
    design = require_valid_speed_door_rule_v2_design(config)
    speeds = tuple(map(float, config["protocol"]["eval_speeds"]))
    horizons = tuple(
        map(
            int,
            config["protocol"]["future_action"][
                "evaluated_horizons"
            ],
        )
    )
    registered = {
        horizon: {
            float(speed): float(displacement)
            for speed, displacement in config["registered_kinematics"][
                "horizontal_displacement_px"
            ][f"h{horizon}"].items()
        }
        for horizon in horizons
    }
    geometry = config["geometry"]
    expected_query_x = {
        "left_to_right": float(geometry["left_query_x"]),
        "right_to_left": float(geometry["right_query_x"]),
    }
    opposite_contact_x = {
        "left_to_right": float(geometry["right_query_x"]),
        "right_to_left": float(geometry["left_query_x"]),
    }

    rows = []
    env = make_hidden_passage_env(render_mode="rgb_array")
    try:
        for bundle in bundles:
            direction = str(bundle["direction"])
            direction_sign = _direction_sign(direction)
            grid = {
                (speed, rule): simulate_v2_future(
                    env,
                    bundle=bundle,
                    speed=speed,
                    rule=rule,
                    config=config,
                )
                for speed in speeds
                for rule in RULES
            }
            query_x = expected_query_x[direction]
            physical = {
                "minimum_h1_passable_rule_gap_px": float("inf"),
                "minimum_h2_far_side_margin_px": float("inf"),
                "minimum_h1_adjacent_speed_gap_px": float("inf"),
                "minimum_h2_adjacent_speed_gap_px": float("inf"),
                "maximum_blocked_state_spread_px": 0.0,
                "maximum_vertical_drift_px": 0.0,
            }
            checks: dict[str, bool] = {
                "query_state_x_is_registered_contact": all(
                    np.isclose(
                        rollout["query_state"][0],
                        query_x,
                        atol=1.0e-6,
                    )
                    for rollout in grid.values()
                ),
                "query_pixels_match_v1_catalog": all(
                    array_sha256(rollout["query_pixels"])
                    == bundle["query_pixels_sha256"]
                    for rollout in grid.values()
                ),
                "future_actions_identical": _all_equal(
                    [
                        rollout["action_blocks"]
                        for rollout in grid.values()
                    ]
                ),
                "no_termination_or_truncation": not any(
                    rollout["terminated_or_truncated"]
                    for rollout in grid.values()
                ),
            }
            for horizon in horizons:
                blocked_states = [
                    grid[(speed, "blocked")]["targets"][horizon][
                        "state"
                    ]
                    for speed in speeds
                ]
                blocked_pixels = [
                    grid[(speed, "blocked")]["targets"][horizon][
                        "pixels"
                    ]
                    for speed in speeds
                ]
                passable_states = [
                    grid[(speed, "passable")]["targets"][horizon][
                        "state"
                    ]
                    for speed in speeds
                ]
                passable_pixels = [
                    grid[(speed, "passable")]["targets"][horizon][
                        "pixels"
                    ]
                    for speed in speeds
                ]
                expected_passable_x = [
                    query_x
                    + direction_sign * registered[horizon][speed]
                    for speed in speeds
                ]
                passable_rule_gaps = [
                    direction_sign * (state[0] - blocked_states[0][0])
                    for state in passable_states
                ]
                adjacent_speed_gaps = [
                    direction_sign * (right[0] - left[0])
                    for left, right in zip(
                        passable_states[:-1],
                        passable_states[1:],
                    )
                ]
                blocked_spread = max(
                    float(np.linalg.norm(right - left))
                    for left in blocked_states
                    for right in blocked_states
                )
                vertical_drift = max(
                    abs(
                        float(rollout["targets"][horizon]["state"][1])
                        - float(rollout["query_state"][1])
                    )
                    for rollout in grid.values()
                )
                if horizon == 1:
                    physical["minimum_h1_passable_rule_gap_px"] = min(
                        passable_rule_gaps
                    )
                    physical["minimum_h1_adjacent_speed_gap_px"] = min(
                        adjacent_speed_gaps
                    )
                else:
                    physical["minimum_h2_adjacent_speed_gap_px"] = min(
                        adjacent_speed_gaps
                    )
                    physical["minimum_h2_far_side_margin_px"] = min(
                        direction_sign
                        * (state[0] - opposite_contact_x[direction])
                        for state in passable_states
                    )
                physical["maximum_blocked_state_spread_px"] = max(
                    physical["maximum_blocked_state_spread_px"],
                    blocked_spread,
                )
                physical["maximum_vertical_drift_px"] = max(
                    physical["maximum_vertical_drift_px"],
                    vertical_drift,
                )
                checks.update(
                    {
                        f"h{horizon}_blocked_states_shared": _all_equal(
                            blocked_states
                        ),
                        f"h{horizon}_blocked_pixels_shared": _all_equal(
                            blocked_pixels
                        ),
                        f"h{horizon}_blocked_at_contact": all(
                            np.isclose(
                                state[0], query_x, atol=1.0e-6
                            )
                            for state in blocked_states
                        ),
                        f"h{horizon}_passable_x_matches_nominal": all(
                            np.isclose(
                                state[0],
                                expected,
                                atol=1.0e-5,
                            )
                            for state, expected in zip(
                                passable_states,
                                expected_passable_x,
                            )
                        ),
                        f"h{horizon}_passable_speed_ordered": all(
                            direction_sign
                            * (right[0] - left[0])
                            > 0.0
                            for left, right in zip(
                                passable_states[:-1],
                                passable_states[1:],
                            )
                        ),
                        f"h{horizon}_four_physical_pixels_unique": (
                            len(
                                {
                                    array_sha256(value)
                                    for value in [
                                        blocked_pixels[0],
                                        *passable_pixels,
                                    ]
                                }
                            )
                            == 4
                        ),
                        f"h{horizon}_vertical_position_unchanged": all(
                            np.isclose(
                                rollout["targets"][horizon]["state"][1],
                                rollout["query_state"][1],
                                atol=1.0e-6,
                            )
                            for rollout in grid.values()
                        ),
                    }
                )
            checks["h2_all_passable_cross_far_contact"] = all(
                direction_sign
                * (
                    grid[(speed, "passable")]["targets"][2][
                        "state"
                    ][0]
                    - opposite_contact_x[direction]
                )
                > 0.0
                for speed in speeds
            )
            rows.append(
                {
                    "query_id": str(bundle["query_id"]),
                    "eval_seed": int(bundle["eval_seed"]),
                    "direction": direction,
                    "checks": checks,
                    "physical": physical,
                    "passed": all(checks.values()),
                }
            )
    finally:
        env.close()

    by_seed = Counter(row["eval_seed"] for row in rows)
    expected_by_seed = int(
        config["evaluation"]["unique_base_queries_per_seed"]
    )
    global_checks = {
        "design_contract_passed": bool(design["passed"]),
        "query_count_exact": (
            not require_full_catalog
            or len(rows)
            == int(config["evaluation"]["unique_base_queries"])
        ),
        "per_seed_counts_exact": (
            not require_full_catalog
            or all(
                by_seed[int(seed)] == expected_by_seed
                for seed in config["evaluation"]["eval_seeds"]
            )
        ),
        "every_query_passed": all(row["passed"] for row in rows),
    }
    failed = [
        {
            "query_id": row["query_id"],
            "failed_checks": sorted(
                name
                for name, passed in row["checks"].items()
                if not passed
            ),
        }
        for row in rows
        if not row["passed"]
    ]
    observed_physics = {
        "minimum_h1_passable_rule_gap_px": min(
            row["physical"]["minimum_h1_passable_rule_gap_px"]
            for row in rows
        ),
        "minimum_h2_far_side_margin_px": min(
            row["physical"]["minimum_h2_far_side_margin_px"]
            for row in rows
        ),
        "minimum_h1_adjacent_speed_gap_px": min(
            row["physical"]["minimum_h1_adjacent_speed_gap_px"]
            for row in rows
        ),
        "minimum_h2_adjacent_speed_gap_px": min(
            row["physical"]["minimum_h2_adjacent_speed_gap_px"]
            for row in rows
        ),
        "maximum_blocked_state_spread_px": max(
            row["physical"]["maximum_blocked_state_spread_px"]
            for row in rows
        ),
        "maximum_vertical_drift_px": max(
            row["physical"]["maximum_vertical_drift_px"]
            for row in rows
        ),
    }
    return {
        "benchmark": str(config["benchmark"]),
        "status": "passed" if all(global_checks.values()) else "failed",
        "passed": all(global_checks.values()),
        "checks": global_checks,
        "query_count": len(rows),
        "by_eval_seed": {
            str(seed): by_seed[int(seed)]
            for seed in config["evaluation"]["eval_seeds"]
        },
        "observed_physics": observed_physics,
        "failed_queries": failed,
        "claim_limit": (
            "Physical prototype audit only; no frozen v2 catalog, "
            "training result, or model composition claim."
        ),
    }


__all__ = [
    "audit_v2_query_bundles",
    "future_action_blocks",
    "simulate_v2_future",
    "validate_v2_training_factor_grid",
]
