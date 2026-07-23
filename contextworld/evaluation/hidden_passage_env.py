from __future__ import annotations

from typing import Any

import numpy as np


HIDDEN_PASSAGE_ENV_ID = "contextworld/TwoRoomHiddenPassage-v1"
PASSAGE_FACTOR = "passage.open"
PASSAGE_RULES = {"blocked": 0, "passable": 1}

_HIDDEN_PASSAGE_ENV_CLASS: type | None = None


def passage_open_value(value: Any) -> int:
    """Return one strict binary hidden-passage value."""

    values = np.asarray(value).reshape(-1)
    if values.size != 1:
        raise ValueError(
            f"{PASSAGE_FACTOR} must contain one value, got {values.tolist()}"
        )
    scalar = values[0]
    if isinstance(scalar, (bool, np.bool_)):
        return int(scalar)
    try:
        integer = int(scalar)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{PASSAGE_FACTOR} must be 0 or 1") from exc
    if float(scalar) != float(integer) or integer not in (0, 1):
        raise ValueError(f"{PASSAGE_FACTOR} must be 0 or 1, got {scalar!r}")
    return integer


def hidden_passage_env_class() -> type:
    """Build a TwoRoom subclass whose passage rule changes no rendered pixel.

    Stable-WorldModel is imported lazily so callers can first pin and load the
    intended sibling checkout.  The hidden factor is deliberately absent from
    the observation returned by the stock TwoRoom environment.
    """

    global _HIDDEN_PASSAGE_ENV_CLASS
    if _HIDDEN_PASSAGE_ENV_CLASS is not None:
        return _HIDDEN_PASSAGE_ENV_CLASS

    from stable_worldmodel import spaces as swm_spaces
    from stable_worldmodel.envs.two_room.env import TwoRoomEnv

    class ContextWorldTwoRoomHiddenPassageEnv(TwoRoomEnv):
        _contextworld_hidden_passage_env = True

        def _build_variation_space(self):
            space = super()._build_variation_space()
            space.spaces["passage"] = swm_spaces.Dict(
                {
                    "open": swm_spaces.Discrete(
                        2,
                        init_value=PASSAGE_RULES["passable"],
                    )
                },
                sampling_order=["open"],
            )
            if "passage" not in space._sampling_order:
                space._sampling_order.append("passage")
            return space

        @property
        def passage_open(self) -> int:
            return passage_open_value(
                self.variation_space["passage"]["open"].value
            )

        def set_hidden_passage_rule(self, value: Any) -> None:
            self.variation_space.set_value(
                {PASSAGE_FACTOR: passage_open_value(value)}
            )

        def _in_any_door_1d(self, coord_1d: float, margin: float):
            if self.passage_open == PASSAGE_RULES["blocked"]:
                return False
            return super()._in_any_door_1d(coord_1d, margin)

    ContextWorldTwoRoomHiddenPassageEnv.__name__ = (
        "ContextWorldTwoRoomHiddenPassageEnv"
    )
    ContextWorldTwoRoomHiddenPassageEnv.__qualname__ = (
        "ContextWorldTwoRoomHiddenPassageEnv"
    )
    _HIDDEN_PASSAGE_ENV_CLASS = ContextWorldTwoRoomHiddenPassageEnv
    return ContextWorldTwoRoomHiddenPassageEnv


def make_hidden_passage_env(**kwargs: Any):
    return hidden_passage_env_class()(**kwargs)


def register_hidden_passage_env(
    env_id: str = HIDDEN_PASSAGE_ENV_ID,
) -> str:
    """Register the isolated ContextWorld environment without changing TwoRoom."""

    import gymnasium as gym

    if env_id in gym.registry:
        entry_point = gym.spec(env_id).entry_point
        if entry_point is make_hidden_passage_env:
            return env_id
        if (
            isinstance(entry_point, str)
            and entry_point
            == "contextworld.evaluation.hidden_passage_env:"
            "make_hidden_passage_env"
        ):
            return env_id
        raise RuntimeError(
            f"Gym id {env_id!r} is already registered by another entry point"
        )
    gym.register(id=env_id, entry_point=make_hidden_passage_env)
    return env_id


__all__ = [
    "HIDDEN_PASSAGE_ENV_ID",
    "PASSAGE_FACTOR",
    "PASSAGE_RULES",
    "hidden_passage_env_class",
    "make_hidden_passage_env",
    "passage_open_value",
    "register_hidden_passage_env",
]
