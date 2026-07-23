from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np

from contextworld.evaluation.hidden_passage import (
    build_feasibility_catalog,
    make_templates,
    model_input_projection,
    simulate_template,
    validate_frozen_config,
    validate_pair,
)
from contextworld.evaluation.hidden_passage_env import (
    HIDDEN_PASSAGE_ENV_ID,
    PASSAGE_RULES,
    make_hidden_passage_env,
    passage_open_value,
    register_hidden_passage_env,
)
from contextworld.synthesis.config import load_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT
    / "configs/benchmark/tworoom_hidden_passage_h3_feasibility_v1.yaml"
)


def _config() -> dict:
    return load_config(CONFIG_PATH)


def _first_template():
    return make_templates(
        door_positions=[49],
        directions=["left_to_right"],
        doorway_offsets_px=[14.0],
        catalog_seed=23,
    )[0]


def test_hidden_rule_changes_collision_but_not_equal_state_pixels() -> None:
    env = make_hidden_passage_env(render_mode="rgb_array")
    try:
        options = {
            "variation": (),
            "variation_values": {
                "agent.speed": np.asarray([5.0], dtype=np.float32),
                "door.position": np.asarray([49, 49, 49], dtype=np.int64),
                "passage.open": PASSAGE_RULES["passable"],
            },
            "state": np.asarray([98.0, 63.0], dtype=np.float32),
            "target_state": np.asarray([190.0, 190.0], dtype=np.float32),
        }
        open_observation, _ = env.reset(seed=7, options=options)
        open_pixels = env.render().copy()
        env.set_hidden_passage_rule(PASSAGE_RULES["blocked"])
        blocked_observation = env._get_obs().detach().cpu().numpy()
        blocked_pixels = env.render().copy()
    finally:
        env.close()

    assert np.array_equal(open_observation, blocked_observation)
    assert np.array_equal(open_pixels, blocked_pixels)


def test_hidden_passage_environment_has_an_isolated_gym_id() -> None:
    import gymnasium as gym

    assert register_hidden_passage_env() == HIDDEN_PASSAGE_ENV_ID
    env = gym.make(HIDDEN_PASSAGE_ENV_ID, render_mode="rgb_array")
    try:
        assert env.unwrapped._contextworld_hidden_passage_env
        assert env.unwrapped.passage_open == PASSAGE_RULES["passable"]
    finally:
        env.close()


def test_contiguous_history3_returns_to_identical_query() -> None:
    template = _first_template()
    passable = simulate_template(template, rule="passable")
    blocked = simulate_template(template, rule="blocked")
    result = validate_pair(
        template,
        passable,
        blocked,
        minimum_middle_state_gap_px=5.0,
        minimum_future_state_gap_px=20.0,
    )

    assert result["passed"]
    assert result["middle_state_gap_px"] == 8.5
    assert result["future_state_gap_px"] == 25.0
    assert result["maximum_zero_command_axis_displacement_px"] == {
        "passable": 8.5,
        "blocked": 0.0,
    }
    assert result["collision_projection_used_to_restore_query"]
    assert result["passable_middle_state"] == [108.0, 63.0]
    assert result["blocked_middle_state"] == [99.5, 63.0]
    assert result["query_state"] == [99.5, 63.0]
    assert result["passable_target_state"] == [124.5, 63.0]
    assert result["blocked_target_state"] == [99.5, 63.0]


def test_model_input_projection_contains_only_pixels_and_action() -> None:
    rollout = simulate_template(_first_template(), rule="passable")
    projection = model_input_projection(rollout)

    assert tuple(projection) == ("pixels", "action")
    assert projection["pixels"].shape == (3, 224, 224, 3)
    assert projection["action"].shape == (3, 5, 2)


def test_validator_rejects_action_or_query_leakage() -> None:
    template = _first_template()
    passable = simulate_template(template, rule="passable")
    blocked = simulate_template(template, rule="blocked")

    action_leak = deepcopy(blocked)
    action_leak["history_actions"][0, 0, 0] = 0.5
    action_result = validate_pair(
        template,
        passable,
        action_leak,
        minimum_middle_state_gap_px=5.0,
        minimum_future_state_gap_px=20.0,
    )
    assert not action_result["passed"]
    assert not action_result["checks"]["history_actions_identical"]

    query_leak = deepcopy(blocked)
    query_leak["query_pixels"][0, 0, 0] ^= np.uint8(1)
    query_result = validate_pair(
        template,
        passable,
        query_leak,
        minimum_middle_state_gap_px=5.0,
        minimum_future_state_gap_px=20.0,
    )
    assert not query_result["passed"]
    assert not query_result["checks"]["query_pixels_identical"]


def test_full_32_pair_feasibility_catalog_passes(tmp_path) -> None:
    catalog, report = build_feasibility_catalog(
        config=_config(),
        repo_root=tmp_path,
        output_root=tmp_path / "output",
    )

    assert report["status"] == "passed"
    assert report["counts"]["paired_templates"] == 32
    assert report["counts"]["rule_rollouts"] == 64
    assert report["failed_templates"] == []
    assert report["exact_replay_templates"] == 32
    assert report["action_leakage_audit"][
        "best_action_signature_only_accuracy"
    ] == 0.5
    assert report["query_pixels"] == {"unique": 32, "expected": 32}
    assert report["checks"]["frozen_config_exact_match"]
    assert report["checks"]["serialized_payloads_roundtrip"]
    assert report["model_input_projection"] == {
        "keys": ["pixels", "action"],
        "serialized_templates_passed": 32,
        "expected_templates": 32,
        "formal_stablewm_adapter_connected": False,
    }
    assert report["collision_projection"][
        "maximum_zero_command_axis_displacement_px"
    ] == 8.5
    assert not report["collision_projection"]["formal_training_approved"]
    assert not report["reuse_limits"]["formal_planning_approved"]
    assert not report["reuse_limits"][
        "right_to_left_goal_direction_aligned"
    ]
    assert len(catalog["bundles"]) == 32
    assert all(
        bundle["validation"]["passed"] for bundle in catalog["bundles"]
    )


def test_frozen_config_rejects_unconsumed_field_drift() -> None:
    mutations = (
        ("environment", "id", "wrong/Environment-v0"),
        ("protocol", "history_tokens", 99),
        ("protocol", "raw_steps_per_action_block", 1),
        ("counts", "rule_rollouts", 999),
        ("gates", "query_pixels_bitwise_match", False),
    )
    for group, field, value in mutations:
        altered = deepcopy(_config())
        altered[group][field] = value
        try:
            validate_frozen_config(altered)
        except ValueError:
            pass
        else:
            raise AssertionError(
                f"Accepted frozen config drift at {group}.{field}"
            )


def test_passage_open_value_is_strictly_binary() -> None:
    assert passage_open_value(0) == 0
    assert passage_open_value(np.asarray([1])) == 1
    for invalid in (-1, 2, 0.5, [0, 1]):
        try:
            passage_open_value(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Accepted invalid hidden rule {invalid!r}")
