from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import pytest

from contextworld.evaluation.speed_door_rule_composition import (
    make_templates,
    simulate_template,
)
from contextworld.evaluation.speed_door_rule_validation import array_sha256
from contextworld.evaluation.speed_door_rule_v2_design import (
    require_valid_speed_door_rule_v2_design,
    validate_speed_door_rule_v2_design,
)
from contextworld.evaluation.speed_door_rule_v2_feasibility import (
    audit_v2_query_bundles,
    validate_v2_training_factor_grid,
)
from contextworld.synthesis.config import load_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs/benchmark/tworoom_speed_door_rule_h3_v2.yaml"
)
V1_CONFIG = (
    ROOT
    / "configs/benchmark/"
    "tworoom_speed_door_rule_h3_feasibility_v1.yaml"
)


def _config() -> dict:
    return load_config(CONFIG)


def test_v2_design_has_two_clean_physical_stages() -> None:
    report = require_valid_speed_door_rule_v2_design(_config())

    assert report["passed"]
    assert report["calculated_horizontal_displacement_px"] == {
        "h1": {"3.1": 15.5, "5.1": 25.5, "7": 35.0},
        "h2": {"3.1": 31.0, "5.1": 51.0, "7": 70.0},
    }
    assert report["calculated_slowest_h2_far_side_margin_px"] == 6.0
    assert report["calculated_fastest_h2_goal_distance_px"] == 20.5


def test_v2_uses_one_shared_blocked_future_class() -> None:
    config = _config()
    classes = config["registered_kinematics"][
        "future_classes_per_horizon"
    ]

    assert classes == {
        "passable_speed_specific": 3,
        "blocked_shared_across_speeds": 1,
        "total_physical_classes": 4,
    }
    assert (
        "six_way_accuracy_with_three_separate_blocked_targets"
        in config["metrics"]["forbidden"]
    )


def test_v2_physical_prototype_suppresses_blocked_speed() -> None:
    v1 = load_config(V1_CONFIG)
    geometry = v1["geometry"]
    template = make_templates(
        door_positions=[geometry["door_positions"][0]],
        directions=["left_to_right"],
        doorway_offset_px=geometry["doorway_offset_px"],
        catalog_seed=23,
        left_to_right_reset_x=geometry["left_to_right_reset_x"],
        right_to_left_reset_x=geometry["right_to_left_reset_x"],
        left_to_right_goal_x=geometry["left_to_right_goal_x"],
        right_to_left_goal_x=geometry["right_to_left_goal_x"],
    )[0]
    rollout = simulate_template(
        template,
        speed=3.1,
        rule="passable",
        protocol=v1["protocol"],
    )
    report = audit_v2_query_bundles(
        config=_config(),
        bundles=[
            {
                "query_id": "prototype",
                "eval_seed": 42,
                "direction": template.direction,
                "template": asdict(template),
                "validation": {
                    "query_state": rollout["query_state"].tolist()
                },
                "query_pixels_sha256": array_sha256(
                    rollout["query_pixels"]
                ),
            }
        ],
        require_full_catalog=False,
    )

    assert report["passed"]
    assert report["observed_physics"][
        "maximum_blocked_state_spread_px"
    ] == 0.0
    assert report["observed_physics"][
        "maximum_vertical_drift_px"
    ] == 0.0
    assert report["observed_physics"][
        "minimum_h2_far_side_margin_px"
    ] == pytest.approx(6.0, abs=2.0e-5)


def test_v2_training_grid_has_four_speed_futures_and_one_blocked_future() -> None:
    training = load_config(
        ROOT
        / "configs/benchmark/"
        "tworoom_speed_door_rule_h3_training_data_v1.yaml"
    )
    protocol = deepcopy(training["protocol"])
    protocol["query_action"].update(
        horizontal_rule_steps=5,
        vertical_speed_steps=0,
        vertical_value=0.0,
    )
    geometry = training["geometry"]
    template = make_templates(
        door_positions=[37],
        directions=["left_to_right"],
        doorway_offset_px=13.5,
        catalog_seed=23,
        left_to_right_reset_x=99.0,
        right_to_left_reset_x=125.0,
        left_to_right_goal_x=geometry["wall_geometry"][
            "left_to_right_goal_x"
        ],
        right_to_left_goal_x=geometry["wall_geometry"][
            "right_to_left_goal_x"
        ],
    )[0]
    speeds = (2.7, 4.3, 6.1, 7.7)
    rollouts = {
        (speed, rule): simulate_template(
            template,
            speed=speed,
            rule=rule,
            protocol=protocol,
        )
        for speed in speeds
        for rule in ("passable", "blocked")
    }
    report = validate_v2_training_factor_grid(
        template,
        rollouts,
        speeds=speeds,
        thresholds=training["gates"],
    )

    assert report["passed"]
    assert report["physical_future_classes"] == 5
    assert report["blocked_future_state_spread_px"] == 0.0


@pytest.mark.parametrize(
    ("path", "value", "failed_check"),
    [
        (
            ("protocol", "future_action", "vertical_value"),
            0.5,
            "future_action_is_horizontal_only",
        ),
        (
            ("protocol", "future_action", "action_blocks"),
            1,
            "two_future_blocks_are_registered",
        ),
        (
            (
                "geometry",
                "distance_between_blocked_contact_states_px",
            ),
            32.0,
            "slowest_speed_fully_crosses_in_h2",
        ),
    ],
)
def test_v2_design_rejects_the_v1_confounds(
    path: tuple[str, ...],
    value: object,
    failed_check: str,
) -> None:
    config = deepcopy(_config())
    target = config
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    report = validate_speed_door_rule_v2_design(config)
    assert not report["passed"]
    assert not report["checks"][failed_check]
    with pytest.raises(ValueError, match=failed_check):
        require_valid_speed_door_rule_v2_design(config)
