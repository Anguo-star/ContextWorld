from __future__ import annotations

import numpy as np
import pytest

from contextworld.evaluation.action_delay_env import (
    ACTION_DELAY_FACTOR,
    action_delay_steps_value,
    make_action_delay_env,
)
from contextworld.synthesis.validator import (
    atom_oracle_runners,
    validate_action_delay_temporal_oracle,
)


@pytest.mark.parametrize("value", [0, 1, 2, 3, 4, 5])
def test_action_delay_value_accepts_frozen_support(value: int) -> None:
    assert action_delay_steps_value(value) == value


@pytest.mark.parametrize("value", [-1, 6, 1.5, True, [1, 2]])
def test_action_delay_value_rejects_invalid_values(value: object) -> None:
    with pytest.raises(ValueError):
        action_delay_steps_value(value)


def _reset(env, delay: int) -> None:
    env.reset(
        seed=7,
        options={
            "variation": (),
            "variation_values": {
                "agent.speed": np.asarray([7.0], dtype=np.float32),
                ACTION_DELAY_FACTOR: delay,
            },
            "state": np.asarray([50.0, 50.0], dtype=np.float32),
            "target_state": np.asarray(
                [205.0, 205.0],
                dtype=np.float32,
            ),
        },
    )


def test_delay_executes_command_after_exact_raw_step_count() -> None:
    env = make_action_delay_env(render_mode="rgb_array")
    try:
        _reset(env, 2)
        action = np.asarray([0.0, 1.0], dtype=np.float32)
        first, *_ = env.step(action)
        second, *_ = env.step(action)
        third, *_ = env.step(action)
    finally:
        env.close()

    np.testing.assert_array_equal(first[:2], [50.0, 50.0])
    np.testing.assert_array_equal(second[:2], [50.0, 50.0])
    np.testing.assert_array_equal(third[:2], [50.0, 57.0])


def test_delay_is_invisible_in_reset_pixels_and_observation() -> None:
    env = make_action_delay_env(render_mode="rgb_array")
    try:
        _reset(env, 0)
        observation_zero = env._get_obs().detach().cpu().numpy().copy()
        pixels_zero = env.render().copy()
        _reset(env, 4)
        observation_four = env._get_obs().detach().cpu().numpy().copy()
        pixels_four = env.render().copy()
    finally:
        env.close()

    np.testing.assert_array_equal(observation_zero, observation_four)
    np.testing.assert_array_equal(pixels_zero, pixels_four)


def test_pending_queue_round_trip_is_exact() -> None:
    env = make_action_delay_env(render_mode="rgb_array")
    pending = np.asarray(
        [[1.0, 0.0], [0.0, -1.0]],
        dtype=np.float32,
    )
    try:
        _reset(env, 2)
        env.restore_contextworld_action_delay(
            2,
            state=[60.0, 70.0],
            goal_state=[205.0, 205.0],
            pending_actions=pending,
        )
        np.testing.assert_array_equal(env.pending_actions(), pending)
        observation, *_ = env.step([0.0, 0.0])
    finally:
        env.close()

    np.testing.assert_array_equal(observation[:2], [67.0, 70.0])


def test_action_delay_temporal_oracle_proves_h3_pairing() -> None:
    result = validate_action_delay_temporal_oracle(
        {
            "delays": [0, 1, 2, 3, 4],
            "agent_speed": 7.0,
            "raw_steps_per_action_block": 5,
            "seed": 20260726,
        }
    )

    assert result["passed"] is True
    assert result["evidence"] == {
        "factor_readback": True,
        "state_transition": True,
        "pixel_transition": True,
        "temporal_alignment": True,
    }
    assert all(
        case["query_observation_and_pixels_exactly_equal"]
        and case["history_midpoints_distinguish_delays"]
        and case["true_futures_distinguish_delays"]
        for case in result["cases"]
    )


def test_action_delay_oracle_is_registered() -> None:
    assert (
        atom_oracle_runners()["action_delay_temporal_oracle"]
        is validate_action_delay_temporal_oracle
    )
