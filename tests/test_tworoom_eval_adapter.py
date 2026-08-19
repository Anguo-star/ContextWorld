from pathlib import Path
import sys

import numpy as np

from contextworld.synthesis.stablewm import load_stable_worldmodel


REPO_ROOT = Path(__file__).resolve().parents[1]
PINNED_STABLEWM = "5864b74980f6ed328fd0045e777b3865962eff43"


def _load_stablewm():
    pinned = Path("/tmp/stable-worldmodel-5864")
    configured = str(
        pinned if pinned.is_dir() else REPO_ROOT.parent / "stable-worldmodel"
    )
    for name in tuple(sys.modules):
        if name == "stable_worldmodel" or name.startswith("stable_worldmodel."):
            del sys.modules[name]
    swm, _, _ = load_stable_worldmodel(
        REPO_ROOT,
        configured,
        PINNED_STABLEWM,
    )
    return swm


def _restore_state_and_goal(env, state, goal) -> None:
    env._set_state(state)
    env._set_goal_state(goal)


def test_eval_setter_matches_collection_reset_for_speed_and_door() -> None:
    _load_stablewm()
    import gymnasium as gym
    from stable_worldmodel.envs.two_room.env import TwoRoomEnv

    from contextworld.evaluation.tworoom import (
        TWOROOM_EVAL_ENV_ID,
        register_tworoom_eval_env,
    )

    register_tworoom_eval_env()
    adapted = gym.make(TWOROOM_EVAL_ENV_ID).unwrapped
    reference = TwoRoomEnv(render_mode="rgb_array")
    state = np.asarray([79.25, 82.5], dtype=np.float32)
    goal = np.asarray([174.0, 151.0], dtype=np.float32)
    variation_values = {
        "agent.speed": np.asarray([8.25], dtype=np.float32),
        "door.position": np.asarray([85, 85, 85], dtype=np.int64),
    }

    try:
        adapted.reset(seed=17)
        adapted._set_contextworld_variation(
            agent_speed=np.asarray([8.25], dtype=np.float32),
            door_position=np.asarray([85.0, 85.0, 85.0], dtype=np.float32),
        )
        _restore_state_and_goal(adapted, state, goal)

        reference.reset(
            seed=17,
            options={
                "variation": (
                    "agent.position",
                    "target.position",
                    "agent.speed",
                    "door.position",
                ),
                "variation_values": variation_values,
            },
        )
        _restore_state_and_goal(reference, state, goal)

        assert adapted._contextworld_variation_readback.keys() == (
            variation_values.keys()
        )
        assert np.array_equal(adapted.render(), reference.render())

        action = np.asarray([0.7, -0.35], dtype=np.float32)
        adapted.step(action)
        reference.step(action)
        assert np.array_equal(
            adapted.agent_position.numpy(), reference.agent_position.numpy()
        )
        assert np.array_equal(adapted.render(), reference.render())
    finally:
        adapted.close()
        reference.close()


def test_eval_setter_matches_speed_only_temporal_dynamics() -> None:
    _load_stablewm()
    import gymnasium as gym
    from stable_worldmodel.envs.two_room.env import TwoRoomEnv

    from contextworld.evaluation.tworoom import (
        TWOROOM_EVAL_ENV_ID,
        register_tworoom_eval_env,
    )

    register_tworoom_eval_env()
    adapted = gym.make(TWOROOM_EVAL_ENV_ID).unwrapped
    reference = TwoRoomEnv(render_mode="rgb_array")
    state = np.asarray([40.0, 30.0], dtype=np.float32)
    goal = np.asarray([180.0, 190.0], dtype=np.float32)

    try:
        adapted.reset(seed=29)
        adapted._set_contextworld_variation(agent_speed=np.asarray([3.25]))
        _restore_state_and_goal(adapted, state, goal)

        reference.reset(
            seed=29,
            options={
                "variation": (
                    "agent.position",
                    "target.position",
                    "agent.speed",
                ),
                "variation_values": {
                    "agent.speed": np.asarray([3.25], dtype=np.float32)
                },
            },
        )
        _restore_state_and_goal(reference, state, goal)

        action = np.asarray([0.5, 0.25], dtype=np.float32)
        adapted.step(action)
        reference.step(action)
        assert np.array_equal(
            adapted.agent_position.numpy(), reference.agent_position.numpy()
        )
        assert np.array_equal(adapted.render(), reference.render())
    finally:
        adapted.close()
        reference.close()


def test_eval_setter_rejects_ambiguous_door_column() -> None:
    from contextworld.evaluation.tworoom import compile_tworoom_eval_variations

    try:
        compile_tworoom_eval_variations(door_position=[61, 85, 61])
    except ValueError as exc:
        assert "entries must agree" in str(exc)
    else:
        raise AssertionError("Expected inconsistent door positions to fail")


def test_eval_callables_restore_variation_before_state_and_goal() -> None:
    from contextworld.evaluation.tworoom import (
        VARIATION_SETTER,
        tworoom_eval_callables,
    )

    methods = [entry["method"] for entry in tworoom_eval_callables()]
    assert methods == [VARIATION_SETTER, "_set_state", "_set_goal_state"]
