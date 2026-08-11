from __future__ import annotations

from typing import Any

import numpy as np
import torch


PORTAL_EXIT_ENV_ID = "contextworld/TwoRoomPortalExit-v1"
PORTAL_EXIT_FACTOR = "portal.exit_mode"
PORTAL_EXIT_MODES = {"near_border": 0, "farther_from_border": 1}

_PORTAL_EXIT_ENV_CLASS: type | None = None


def portal_exit_mode_value(value: Any) -> int:
    values = np.asarray(value).reshape(-1)
    if values.size != 1:
        raise ValueError(f"{PORTAL_EXIT_FACTOR} must contain one value")
    scalar = values[0]
    if isinstance(scalar, (bool, np.bool_)):
        return int(scalar)
    try:
        integer = int(scalar)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{PORTAL_EXIT_FACTOR} must be 0 or 1") from exc
    if float(scalar) != float(integer) or integer not in (0, 1):
        raise ValueError(f"{PORTAL_EXIT_FACTOR} must be 0 or 1")
    return integer


def portal_exit_env_class() -> type:
    """Return a TwoRoom environment with one visually hidden portal mapping.

    The visible doorway is unchanged.  Crossing it moves the agent to one of
    two exit coordinates on the destination side.  Which exit is active is
    absent from observations and pixels, so it can only be inferred from a
    previous interaction.
    """

    global _PORTAL_EXIT_ENV_CLASS
    if _PORTAL_EXIT_ENV_CLASS is not None:
        return _PORTAL_EXIT_ENV_CLASS

    from stable_worldmodel import spaces as swm_spaces
    from stable_worldmodel.envs.two_room.env import TwoRoomEnv

    class ContextWorldTwoRoomPortalExitEnv(TwoRoomEnv):
        _contextworld_portal_exit_env = True

        def _build_variation_space(self):
            space = super()._build_variation_space()
            space.spaces["portal"] = swm_spaces.Dict(
                {
                    "exit_mode": swm_spaces.Discrete(
                        2,
                        init_value=PORTAL_EXIT_MODES["near_border"],
                    )
                },
                sampling_order=["exit_mode"],
            )
            if "portal" not in space._sampling_order:
                space._sampling_order.append("portal")
            return space

        @property
        def portal_exit_mode(self) -> int:
            return portal_exit_mode_value(
                self.variation_space["portal"]["exit_mode"].value
            )

        def set_hidden_portal_exit(self, value: Any) -> None:
            self.variation_space.set_value(
                {PORTAL_EXIT_FACTOR: portal_exit_mode_value(value)}
            )

        def reset(self, seed=None, options=None):
            observation, info = super().reset(seed=seed, options=options)
            self._portal_armed = True
            self._portal_rearm_side = 0.0
            info["portal_transition"] = False
            return observation, info

        def restore_contextworld_portal_exit(
            self,
            exit_mode: Any,
            state: Any | None = None,
            goal_state: Any | None = None,
        ) -> None:
            self.set_hidden_portal_exit(exit_mode)
            if state is not None:
                self._set_state(np.asarray(state, dtype=np.float32))
            if goal_state is not None:
                self._set_goal_state(np.asarray(goal_state, dtype=np.float32))
            self._portal_armed = True
            self._portal_rearm_side = 0.0

        def _portal_secondary_coordinate(self) -> float:
            center = float(self.door_positions[0])
            low_side = center <= float(self.WALL_CENTER)
            border_center = float(self.BORDER_SIZE) + float(
                self.variation_space["agent"]["radius"].value.item()
            )
            distance = 10.0 if self.portal_exit_mode == 0 else 25.0
            if low_side:
                return border_center + distance
            return float(self.IMG_SIZE) - border_center - distance

        def _apply_collisions(self, pos1: torch.Tensor, pos2: torch.Tensor):
            result = super()._apply_collisions(pos1, pos2)
            primary_axis = 0 if self.wall_axis == 1 else 1
            secondary_axis = 1 - primary_axis
            start = float(pos1[primary_axis])
            end = float(result[primary_axis])
            center = float(self.WALL_CENTER)

            if not getattr(self, "_portal_armed", True):
                rearm_side = float(getattr(self, "_portal_rearm_side", 0.0))
                if rearm_side * (end - center) >= 13.0:
                    self._portal_armed = True
                return result

            crossed = (start < center <= end) or (start > center >= end)
            secondary = float(result[secondary_axis])
            if not crossed or not self._in_any_door_1d(secondary, 1.75):
                return result

            direction = 1.0 if end > start else -1.0
            result = result.clone()
            # Stay beyond the thickest configured wall plus agent radius, so
            # the following zero commands cannot be projected back to a wall
            # boundary in only one of the two hidden modes.
            result[primary_axis] = center + direction * 18.0
            result[secondary_axis] = self._portal_secondary_coordinate()
            self._portal_armed = False
            self._portal_rearm_side = -direction
            return result

    ContextWorldTwoRoomPortalExitEnv.__name__ = (
        "ContextWorldTwoRoomPortalExitEnv"
    )
    ContextWorldTwoRoomPortalExitEnv.__qualname__ = (
        "ContextWorldTwoRoomPortalExitEnv"
    )
    _PORTAL_EXIT_ENV_CLASS = ContextWorldTwoRoomPortalExitEnv
    return ContextWorldTwoRoomPortalExitEnv


def make_portal_exit_env(**kwargs: Any):
    return portal_exit_env_class()(**kwargs)


def register_portal_exit_env(
    env_id: str = PORTAL_EXIT_ENV_ID,
) -> str:
    import gymnasium as gym

    if env_id in gym.registry:
        entry_point = gym.spec(env_id).entry_point
        if entry_point is make_portal_exit_env:
            return env_id
        if entry_point == (
            "contextworld.evaluation.portal_exit_env:make_portal_exit_env"
        ):
            return env_id
        raise RuntimeError(f"Gym id {env_id!r} is already registered")
    gym.register(id=env_id, entry_point=make_portal_exit_env)
    return env_id


__all__ = [
    "PORTAL_EXIT_ENV_ID",
    "PORTAL_EXIT_FACTOR",
    "PORTAL_EXIT_MODES",
    "make_portal_exit_env",
    "portal_exit_env_class",
    "portal_exit_mode_value",
    "register_portal_exit_env",
]
