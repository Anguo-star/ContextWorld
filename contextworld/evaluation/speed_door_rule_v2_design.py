from __future__ import annotations

from typing import Any


def _float_mapping(mapping: dict[str, Any]) -> dict[float, float]:
    return {
        float(speed): float(displacement)
        for speed, displacement in mapping.items()
    }


def validate_speed_door_rule_v2_design(
    config: dict[str, Any],
) -> dict[str, Any]:
    """Validate the arithmetic and semantic contract of the v2 design.

    This function does not run TwoRoom.  Passing it means that the authored
    protocol is internally consistent; the separate physical-feasibility
    build must still replay every registered query in the pinned simulator.
    """

    protocol = config["protocol"]
    geometry = config["geometry"]
    registered = config["registered_kinematics"]
    evaluation = config["evaluation"]
    metrics = config["metrics"]

    speeds = tuple(map(float, protocol["eval_speeds"]))
    raw_steps = int(protocol["raw_steps_per_action_block"])
    action_blocks = int(protocol["future_action"]["action_blocks"])
    horizons = tuple(map(int, protocol["future_action"]["evaluated_horizons"]))
    horizontal = float(protocol["future_action"]["horizontal_value"])
    vertical = float(protocol["future_action"]["vertical_value"])

    calculated = {
        horizon: {
            speed: speed * raw_steps * horizon * abs(horizontal)
            for speed in speeds
        }
        for horizon in horizons
    }
    registered_h1 = _float_mapping(
        registered["horizontal_displacement_px"]["h1"]
    )
    registered_h2 = _float_mapping(
        registered["horizontal_displacement_px"]["h2"]
    )
    registered_by_horizon = {1: registered_h1, 2: registered_h2}

    query_to_trigger = float(
        geometry["maximum_query_to_collision_trigger_px"]
    )
    wall_crossing_distance = float(
        geometry["distance_between_blocked_contact_states_px"]
    )
    slowest_h2_margin = calculated[2][min(speeds)] - wall_crossing_distance
    start_to_goal = float(geometry["left_to_right_goal_x"]) - float(
        geometry["left_query_x"]
    )
    fastest_h2_goal_distance = (
        start_to_goal - calculated[2][max(speeds)]
    )

    target_classes = registered["future_classes_per_horizon"]
    train_speeds = tuple(
        map(
            float,
            config["training_isolation"][
                "speed_factor_training_speeds"
            ],
        )
    )
    expected_queries = (
        len(evaluation["eval_seeds"])
        * int(evaluation["unique_base_queries_per_seed"])
    )

    checks = {
        "schema_version_is_two": int(config["schema_version"]) == 2,
        "design_is_not_claimed_as_completed": (
            str(config["status"])
            == "preregistered_design_before_feasibility_data_training_or_model_scoring"
        ),
        "history_is_three": int(protocol["history_tokens"]) == 3,
        "action_block_is_five_raw_steps": raw_steps == 5,
        "two_future_blocks_are_registered": action_blocks == 2,
        "h1_and_h2_are_separate": horizons == (1, 2),
        "future_action_is_horizontal_only": vertical == 0.0,
        "eval_speeds_are_unique_and_increasing": (
            speeds == tuple(sorted(set(speeds))) and len(speeds) == 3
        ),
        "registered_displacements_match_actions": all(
            abs(
                calculated[horizon][speed]
                - registered_by_horizon[horizon][speed]
            )
            <= 1.0e-9
            for horizon in horizons
            for speed in speeds
        ),
        "every_speed_contacts_door_in_h1": (
            min(calculated[1].values()) > query_to_trigger
        ),
        "slowest_speed_fully_crosses_in_h2": slowest_h2_margin > 0.0,
        "registered_slowest_crossing_margin_matches": (
            abs(
                slowest_h2_margin
                - float(registered["slowest_h2_far_side_margin_px"])
            )
            <= 1.0e-9
        ),
        "fastest_h2_does_not_terminate": (
            fastest_h2_goal_distance
            > float(geometry["termination_radius_px"])
        ),
        "registered_goal_distance_matches": (
            abs(
                fastest_h2_goal_distance
                - float(registered["fastest_h2_goal_distance_px"])
            )
            <= 1.0e-9
        ),
        "blocked_future_is_one_shared_class": (
            int(target_classes["blocked_shared_across_speeds"]) == 1
        ),
        "three_passable_speed_classes_are_kept": (
            int(target_classes["passable_speed_specific"]) == len(speeds)
        ),
        "physical_target_class_count_is_four": (
            int(target_classes["total_physical_classes"])
            == len(speeds) + 1
        ),
        "six_way_blocked_scoring_is_forbidden": (
            "six_way_accuracy_with_three_separate_blocked_targets"
            in metrics["forbidden"]
        ),
        "train_and_eval_speeds_do_not_overlap": not (
            set(train_speeds) & set(speeds)
        ),
        "eval_speeds_are_interpolation_only": all(
            min(train_speeds) < speed < max(train_speeds)
            for speed in speeds
        ),
        "base_query_count_is_50_times_6": (
            int(evaluation["unique_base_queries"]) == expected_queries == 300
        ),
        "six_histories_per_query": (
            int(evaluation["histories_per_query"]) == len(speeds) * 2
        ),
        "endpoint_count_matches_two_horizons": (
            int(evaluation["prediction_endpoints_per_checkpoint"])
            == expected_queries * len(speeds) * 2 * len(horizons)
        ),
        "single_factor_prerequisites_are_mandatory": all(
            bool(value)
            for value in config["decision_gates"]["prerequisites"].values()
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "calculated_horizontal_displacement_px": {
            f"h{horizon}": {
                f"{speed:g}": displacement
                for speed, displacement in values.items()
            }
            for horizon, values in calculated.items()
        },
        "calculated_slowest_h2_far_side_margin_px": slowest_h2_margin,
        "calculated_fastest_h2_goal_distance_px": fastest_h2_goal_distance,
        "claim_limit": (
            "Internal design consistency only; no physical feasibility, "
            "model result, or composition claim."
        ),
    }


def require_valid_speed_door_rule_v2_design(
    config: dict[str, Any],
) -> dict[str, Any]:
    report = validate_speed_door_rule_v2_design(config)
    failed = sorted(
        name for name, passed in report["checks"].items() if not passed
    )
    if failed:
        raise ValueError(
            "Invalid Speed × Door Rule v2 design: " + ", ".join(failed)
        )
    return report


__all__ = [
    "require_valid_speed_door_rule_v2_design",
    "validate_speed_door_rule_v2_design",
]
