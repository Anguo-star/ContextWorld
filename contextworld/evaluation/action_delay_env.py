from __future__ import annotations

from typing import Any

import numpy as np


ACTION_DELAY_ENV_ID = "contextworld/TwoRoomActionDelay-v1"
ACTION_DELAY_FACTOR = "action.delay_steps"
MIN_ACTION_DELAY_STEPS = 0
# History=3 with two five-step context transitions can identify at most one
# value above the original 0..4 training/Validation range while still ending
# with an empty queue and an identical query.  Delay 5 is reserved for that
# high-end Validation extension; formal training remains frozen at 0, 2, 4.
MAX_ACTION_DELAY_STEPS = 5

_ACTION_DELAY_ENV_CLASSES: dict[int, type] = {}


def action_delay_steps_value(
    value: Any,
    *,
    maximum: int = MAX_ACTION_DELAY_STEPS,
) -> int:
    """Return one strict raw-environment-step action delay.

    The public History=3 environment keeps its frozen ``0..5`` support.
    Long-history feasibility studies may request a larger explicit bound
    without silently widening the old environment or its release contract.
    """

    maximum = int(maximum)
    if maximum < MIN_ACTION_DELAY_STEPS:
        raise ValueError("maximum action delay must be non-negative")
    values = np.asarray(value).reshape(-1)
    if values.size != 1:
        raise ValueError(
            f"{ACTION_DELAY_FACTOR} must contain one value, "
            f"got {values.tolist()}"
        )
    scalar = values[0]
    if isinstance(scalar, (bool, np.bool_)):
        raise ValueError(
            f"{ACTION_DELAY_FACTOR} must be an integer in "
            f"[{MIN_ACTION_DELAY_STEPS}, {maximum}]"
        )
    try:
        integer = int(scalar)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{ACTION_DELAY_FACTOR} must be an integer"
        ) from exc
    if (
        float(scalar) != float(integer)
        or not MIN_ACTION_DELAY_STEPS
        <= integer
        <= maximum
    ):
        raise ValueError(
            f"{ACTION_DELAY_FACTOR} must be an integer in "
            f"[{MIN_ACTION_DELAY_STEPS}, {maximum}], "
            f"got {scalar!r}"
        )
    return integer


def action_delay_env_class(
    max_delay_steps: int = MAX_ACTION_DELAY_STEPS,
) -> type:
    """Build a TwoRoom environment with an invisible command delay.

    A delay of ``d`` means that each commanded action is executed ``d`` raw
    environment steps later.  The queue is initialized with zero actions on
    reset.  The delay and queue are absent from observations and rendered
    pixels.
    """

    maximum_delay_steps = int(max_delay_steps)
    if maximum_delay_steps < MIN_ACTION_DELAY_STEPS:
        raise ValueError("max_delay_steps must be non-negative")
    cached = _ACTION_DELAY_ENV_CLASSES.get(maximum_delay_steps)
    if cached is not None:
        return cached

    from stable_worldmodel import spaces as swm_spaces
    from stable_worldmodel.envs.two_room.env import TwoRoomEnv

    class ContextWorldTwoRoomActionDelayEnv(TwoRoomEnv):
        _contextworld_action_delay_env = True

        def _build_variation_space(self):
            space = super()._build_variation_space()
            space.spaces["action"] = swm_spaces.Dict(
                {
                    "delay_steps": swm_spaces.Discrete(
                        maximum_delay_steps
                        - MIN_ACTION_DELAY_STEPS
                        + 1,
                        start=MIN_ACTION_DELAY_STEPS,
                        init_value=MIN_ACTION_DELAY_STEPS,
                    )
                },
                sampling_order=["delay_steps"],
            )
            if "action" not in space._sampling_order:
                space._sampling_order.append("action")
            return space

        @property
        def action_delay_steps(self) -> int:
            return action_delay_steps_value(
                self.variation_space["action"]["delay_steps"].value,
                maximum=maximum_delay_steps,
            )

        def _reset_action_delay_queue(self) -> None:
            self._action_delay_queue = [
                np.zeros(2, dtype=np.float32)
                for _ in range(self.action_delay_steps)
            ]

        def reset(self, seed=None, options=None):
            observation, info = super().reset(seed=seed, options=options)
            self._reset_action_delay_queue()
            self._contextworld_action_delay_readback = (
                self.action_delay_steps
            )
            return observation, info

        def set_action_delay(self, value: Any) -> None:
            self.variation_space.set_value(
                {
                    ACTION_DELAY_FACTOR: action_delay_steps_value(
                        value,
                        maximum=maximum_delay_steps,
                    )
                }
            )
            self._reset_action_delay_queue()
            self._contextworld_action_delay_readback = (
                self.action_delay_steps
            )

        def restore_contextworld_action_delay(
            self,
            delay_steps: Any,
            state: Any | None = None,
            goal_state: Any | None = None,
            pending_actions: Any | None = None,
        ) -> None:
            self.set_action_delay(delay_steps)
            if state is not None:
                self._set_state(np.asarray(state, dtype=np.float32))
            if goal_state is not None:
                self._set_goal_state(
                    np.asarray(goal_state, dtype=np.float32)
                )
            if pending_actions is not None:
                pending = np.asarray(
                    pending_actions,
                    dtype=np.float32,
                )
                expected_shape = (self.action_delay_steps, 2)
                if pending.shape != expected_shape:
                    raise ValueError(
                        "pending_actions must have shape "
                        f"{expected_shape}, got {pending.shape}"
                    )
                self._action_delay_queue = [
                    row.copy() for row in pending
                ]

        def pending_actions(self) -> np.ndarray:
            if not hasattr(self, "_action_delay_queue"):
                self._reset_action_delay_queue()
            if not self._action_delay_queue:
                return np.empty((0, 2), dtype=np.float32)
            return np.stack(self._action_delay_queue).astype(
                np.float32,
                copy=True,
            )

        def step(self, action):
            commanded = np.clip(
                np.asarray(action, dtype=np.float32),
                -1.0,
                1.0,
            )
            if not hasattr(self, "_action_delay_queue"):
                self._reset_action_delay_queue()
            if self.action_delay_steps == 0:
                executed = commanded
            else:
                executed = self._action_delay_queue.pop(0)
                self._action_delay_queue.append(commanded.copy())
            observation, reward, terminated, truncated, info = super().step(
                executed
            )
            info["contextworld.commanded_action"] = commanded.copy()
            info["contextworld.executed_action"] = np.asarray(
                executed,
                dtype=np.float32,
            ).copy()
            return observation, reward, terminated, truncated, info

    class_name = (
        "ContextWorldTwoRoomActionDelayEnv"
        if maximum_delay_steps == MAX_ACTION_DELAY_STEPS
        else (
            "ContextWorldTwoRoomActionDelayEnv"
            f"Max{maximum_delay_steps}"
        )
    )
    ContextWorldTwoRoomActionDelayEnv.__name__ = class_name
    ContextWorldTwoRoomActionDelayEnv.__qualname__ = class_name
    _ACTION_DELAY_ENV_CLASSES[maximum_delay_steps] = (
        ContextWorldTwoRoomActionDelayEnv
    )
    return ContextWorldTwoRoomActionDelayEnv


def make_action_delay_env(**kwargs: Any):
    return action_delay_env_class()(**kwargs)


def make_extended_action_delay_env(
    *,
    max_delay_steps: int,
    **kwargs: Any,
):
    """Create an isolated wider-delay environment for long-history studies."""

    return action_delay_env_class(max_delay_steps)(**kwargs)


def register_action_delay_env(
    env_id: str = ACTION_DELAY_ENV_ID,
) -> str:
    import gymnasium as gym

    if env_id in gym.registry:
        entry_point = gym.spec(env_id).entry_point
        if entry_point is make_action_delay_env:
            return env_id
        if (
            isinstance(entry_point, str)
            and entry_point
            == "contextworld.evaluation.action_delay_env:"
            "make_action_delay_env"
        ):
            return env_id
        raise RuntimeError(
            f"Gym id {env_id!r} is already registered by another entry point"
        )
    gym.register(id=env_id, entry_point=make_action_delay_env)
    return env_id


__all__ = [
    "ACTION_DELAY_ENV_ID",
    "ACTION_DELAY_FACTOR",
    "MAX_ACTION_DELAY_STEPS",
    "MIN_ACTION_DELAY_STEPS",
    "action_delay_env_class",
    "action_delay_steps_value",
    "make_action_delay_env",
    "make_extended_action_delay_env",
    "register_action_delay_env",
]
