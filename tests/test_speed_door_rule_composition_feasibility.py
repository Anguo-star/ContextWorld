from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from contextworld.evaluation.speed_door_rule_composition import (
    build_feasibility_catalog,
    make_templates,
    model_input_projection,
    simulate_template,
    validate_factor_grid,
    validate_frozen_config,
)
from contextworld.evaluation.speed_identifiability import (
    agent_centroid_from_rgb,
)
from contextworld.synthesis.config import load_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT
    / "configs/benchmark/"
    "tworoom_speed_door_rule_h3_feasibility_v1.yaml"
)


def _config() -> dict:
    return load_config(CONFIG_PATH)


def _first_template():
    geometry = _config()["geometry"]
    return make_templates(
        door_positions=[geometry["door_positions"][0]],
        directions=["left_to_right"],
        doorway_offset_px=geometry["doorway_offset_px"],
        catalog_seed=23,
        left_to_right_reset_x=geometry["left_to_right_reset_x"],
        right_to_left_reset_x=geometry["right_to_left_reset_x"],
        left_to_right_goal_x=geometry["left_to_right_goal_x"],
        right_to_left_goal_x=geometry["right_to_left_goal_x"],
    )[0]


def _factor_grid():
    config = _config()
    protocol = config["protocol"]
    template = _first_template()
    speeds = tuple(map(float, protocol["speeds"]))
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
    return config, template, speeds, rollouts


def test_six_factor_histories_return_to_one_identical_query() -> None:
    _, _, _, rollouts = _factor_grid()
    values = list(rollouts.values())

    assert all(
        np.array_equal(values[0]["query_state"], value["query_state"])
        for value in values[1:]
    )
    assert all(
        np.array_equal(values[0]["query_pixels"], value["query_pixels"])
        for value in values[1:]
    )
    assert all(
        np.array_equal(
            values[0]["history_actions"], value["history_actions"]
        )
        for value in values[1:]
    )
    assert all(
        np.array_equal(values[0]["query_action"], value["query_action"])
        for value in values[1:]
    )


def test_history_and_future_pixels_separate_both_hidden_factors() -> None:
    config, template, speeds, rollouts = _factor_grid()
    result = validate_factor_grid(
        template,
        rollouts,
        speeds=speeds,
        thresholds=config["gates"],
    )

    assert result["passed"]
    assert result["minimum_middle_rule_centroid_gap_px"] >= 2.0
    assert (
        result["minimum_middle_adjacent_speed_centroid_gap_px"] >= 0.4
    )
    assert result["minimum_future_rule_state_gap_px"] >= 10.0
    assert (
        result["minimum_future_adjacent_speed_centroid_gap_px"] >= 0.8
    )
    assert len(
        {
            tuple(agent_centroid_from_rgb(value["history_pixels"][1]))
            for value in rollouts.values()
        }
    ) == 6


def test_model_input_projection_excludes_speed_rule_and_state() -> None:
    _, _, _, rollouts = _factor_grid()
    projection = model_input_projection(next(iter(rollouts.values())))

    assert tuple(projection) == ("pixels", "action")
    assert projection["pixels"].shape == (3, 224, 224, 3)
    assert projection["action"].shape == (3, 5, 2)


def test_grid_validator_rejects_query_or_action_leakage() -> None:
    config, template, speeds, rollouts = _factor_grid()

    query_leak = deepcopy(rollouts)
    query_leak[(speeds[-1], "blocked")]["query_pixels"][0, 0, 0] ^= (
        np.uint8(1)
    )
    query_result = validate_factor_grid(
        template,
        query_leak,
        speeds=speeds,
        thresholds=config["gates"],
    )
    assert not query_result["passed"]
    assert not query_result["checks"]["query_pixels_identical"]

    action_leak = deepcopy(rollouts)
    action_leak[(speeds[-1], "blocked")]["history_actions"][0, 0, 0] = 0.5
    action_result = validate_factor_grid(
        template,
        action_leak,
        speeds=speeds,
        thresholds=config["gates"],
    )
    assert not action_result["passed"]
    assert not action_result["checks"]["history_actions_identical"]


def test_full_eight_template_catalog_passes(tmp_path: Path) -> None:
    catalog, report = build_feasibility_catalog(
        config=_config(),
        repo_root=tmp_path,
        output_root=tmp_path / "output",
    )

    assert report["status"] == "passed"
    assert report["counts"]["templates"] == 8
    assert report["counts"]["factor_combinations_per_template"] == 6
    assert report["counts"]["rollouts"] == 48
    assert report["failed_templates"] == []
    assert report["exact_replay_templates"] == 8
    assert report["action_only_leakage_audit"]["joint_factor"][
        "best_signature_only_accuracy"
    ] == pytest.approx(1.0 / 6.0)
    assert report["query_only_leakage_audit"]["joint_factor"][
        "best_signature_only_accuracy"
    ] == pytest.approx(1.0 / 6.0)
    assert report["action_only_leakage_audit"]["speed"][
        "best_signature_only_accuracy"
    ] == pytest.approx(1.0 / 3.0)
    assert report["action_only_leakage_audit"]["door_rule"][
        "best_signature_only_accuracy"
    ] == pytest.approx(1.0 / 2.0)
    assert report["query_pixels"] == {"unique": 8, "expected": 8}
    assert report["checks"]["serialized_payloads_roundtrip"]
    assert report["checks"]["frozen_config_exact_match"]
    assert len(catalog["bundles"]) == 8
    assert all(
        bundle["validation"]["passed"] for bundle in catalog["bundles"]
    )


def test_frozen_config_rejects_any_protocol_drift() -> None:
    mutations = (
        ("scope", "history_tokens", 5),
        ("protocol", "speeds", [3.1, 5.1]),
        ("protocol", "raw_steps_per_action_block", 4),
        ("counts", "rollouts", 47),
        ("gates", "query_pixels_bitwise_match", False),
    )
    for group, field, value in mutations:
        altered = deepcopy(_config())
        altered[group][field] = value
        with pytest.raises(ValueError, match="frozen v1 contract"):
            validate_frozen_config(altered)
