from __future__ import annotations

import numpy as np
import pytest

from contextworld.synthesis.reset_constraints import (
    apply_tworoom_reset_constraints,
    normalize_reset_constraints,
)


def _options(speed: float) -> dict:
    return {
        "variation": (
            "agent.position",
            "target.position",
            "agent.speed",
        ),
        "variation_values": {
            "agent.speed": np.asarray([speed], dtype=np.float32)
        },
    }


def test_reset_constraint_config_rejects_unknown_semantics() -> None:
    with pytest.raises(ValueError, match="Unsupported TwoRoom reset constraints"):
        normalize_reset_constraints({"unknown": True})
    with pytest.raises(ValueError, match="must be 'same' or 'opposite'"):
        normalize_reset_constraints({"target_room": "elsewhere"})
    with pytest.raises(ValueError, match="minimum_door_path_distance"):
        normalize_reset_constraints({"minimum_door_path_distance": -1})


def test_opposite_room_resets_are_speed_independent() -> None:
    from stable_worldmodel.envs.two_room.env import TwoRoomEnv

    constraints = {
        "target_room": "opposite",
        "exclude_wall_zone": True,
        "minimum_initial_distance": 40.0,
    }
    slow = TwoRoomEnv(render_mode="rgb_array")
    fast = TwoRoomEnv(render_mode="rgb_array")
    try:
        assert apply_tworoom_reset_constraints(slow, constraints) == 1
        assert apply_tworoom_reset_constraints(fast, constraints) == 1
        for seed in range(64):
            slow_observation, _ = slow.reset(seed=seed, options=_options(2.6))
            fast_observation, _ = fast.reset(seed=seed, options=_options(7.9))
            assert np.array_equal(slow_observation[:4], fast_observation[:4])
            start = slow_observation[:2]
            goal = slow_observation[2:4]
            assert (start[0] < slow.WALL_CENTER) != (goal[0] < slow.WALL_CENTER)
            assert np.linalg.norm(start - goal) >= 40.0
            wall_thickness = int(
                slow.variation_space["wall"]["thickness"].value
            )
            agent_radius = float(
                slow.variation_space["agent"]["radius"].value.item()
            )
            wall_half_extent = wall_thickness // 2 + agent_radius
            assert abs(float(goal[0]) - slow.WALL_CENTER) > wall_half_extent
    finally:
        slow.close()
        fast.close()


def test_same_room_template_bounds_and_constraint_replacement() -> None:
    from stable_worldmodel.envs.two_room.env import TwoRoomEnv

    env = TwoRoomEnv(render_mode="rgb_array")
    try:
        constraints = {
            "target_room": "same",
            "agent_position_bounds": [[161.0, 175.0], [63.0, 77.0]],
            "target_position_bounds": [[182.0, 196.0], [182.0, 196.0]],
            "minimum_initial_distance": 40.0,
        }
        assert apply_tworoom_reset_constraints(env, constraints) == 1
        for seed in range(16):
            observation, _ = env.reset(seed=seed, options=_options(5.0))
            state = np.asarray(observation)
            assert np.all(state[:2] >= [161.0, 63.0])
            assert np.all(state[:2] <= [175.0, 77.0])
            assert np.all(state[2:4] >= [182.0, 182.0])
            assert np.all(state[2:4] <= [196.0, 196.0])
            assert (state[0] < env.WALL_CENTER) == (
                state[2] < env.WALL_CENTER
            )

        assert apply_tworoom_reset_constraints(env, {}) == 1
        assert env.variation_space["agent"]["position"].constrain_fn([0, 0])
        assert env.variation_space["target"]["position"].constrain_fn([0, 0])
    finally:
        env.close()


def test_minimum_door_path_distance_restores_original_task_constraint() -> None:
    from stable_worldmodel.envs.two_room.env import TwoRoomEnv

    env = TwoRoomEnv(render_mode="rgb_array")
    try:
        constraints = {
            "target_room": "opposite",
            "exclude_wall_zone": True,
            # Historical original collection used min_steps=25 at speed=5.
            "minimum_door_path_distance": 125.0,
        }
        assert apply_tworoom_reset_constraints(env, constraints) == 1
        for seed in range(64):
            observation, _ = env.reset(seed=seed, options=_options(5.0))
            start = np.asarray(observation[:2], dtype=np.float64)
            goal = np.asarray(observation[2:4], dtype=np.float64)
            door = np.asarray(
                [
                    env.WALL_CENTER,
                    float(env.variation_space["door"]["position"].value[0]),
                ]
            )
            path_distance = np.linalg.norm(start - door) + np.linalg.norm(
                goal - door
            )
            assert path_distance >= 125.0
    finally:
        env.close()
