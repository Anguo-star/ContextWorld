from __future__ import annotations

from typing import Any

import numpy as np

from contextworld.synthesis.atoms import tworoom_atom_registry


TWOROOM_EVAL_ENV_ID = "contextworld/TwoRoomEval-v1"
VARIATION_SETTER = "_set_contextworld_variation"
SPEED_COLUMN = "variation_agent_speed"
DOOR_COLUMN = "variation_door_position"


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _speed_scalar(value: Any) -> float:
    values = _numpy(value).reshape(-1)
    if values.size != 1:
        raise ValueError(
            f"{SPEED_COLUMN} must contain exactly one value, got {values.tolist()}"
        )
    return float(values[0])


def _door_scalar(value: Any) -> int:
    values = _numpy(value).reshape(-1)
    if values.size not in (1, 3):
        raise ValueError(
            f"{DOOR_COLUMN} must contain one or three values, got {values.tolist()}"
        )
    if not np.all(values == values[0]):
        raise ValueError(
            f"{DOOR_COLUMN} entries must agree for the one-door task, "
            f"got {values.tolist()}"
        )
    return int(values[0])


def compile_tworoom_eval_variations(
    *,
    agent_speed: Any | None = None,
    door_position: Any | None = None,
) -> dict[str, np.ndarray]:
    """Compile stored factor columns back to StableWM variation values.

    The same atom adapters used for synthesis perform range, type, and shape
    validation here.  This keeps collection and evaluation from drifting into
    subtly different interpretations of a factor.
    """

    if agent_speed is None and door_position is None:
        raise ValueError(
            "Variation-aware TwoRoom eval requires at least one stored factor"
        )

    registry = tworoom_atom_registry()
    compiled: dict[str, np.ndarray] = {}
    if agent_speed is not None:
        atom = registry["agent_speed"].compile(_speed_scalar(agent_speed))
        compiled[atom.factor_key] = atom.variation_value
    if door_position is not None:
        atom = registry["door_position"].compile(_door_scalar(door_position))
        compiled[atom.factor_key] = atom.variation_value
    return compiled


def tworoom_eval_callables() -> list[dict[str, Any]]:
    """Return the ordered setup calls required by ``World.evaluate``.

    StableWM applies these calls after a default reset.  Variation restoration
    must precede state and goal restoration because door geometry participates
    in rendering and collision, while speed participates in the next
    transition.
    """

    return [
        {
            "method": VARIATION_SETTER,
            "args": {
                "agent_speed": {"value": SPEED_COLUMN, "in_dataset": True},
                "door_position": {"value": DOOR_COLUMN, "in_dataset": True},
            },
        },
        {
            "method": "_set_state",
            "args": {"state": {"value": "state", "in_dataset": True}},
        },
        {
            "method": "_set_goal_state",
            "args": {
                "goal_state": {"value": "goal_state", "in_dataset": True}
            },
        },
    ]


def validate_tworoom_factor_columns(column_names: list[str]) -> tuple[str, ...]:
    """Fail closed when an OOD table has no restorable benchmark factor."""

    present = tuple(
        column
        for column in (SPEED_COLUMN, DOOR_COLUMN)
        if column in set(column_names)
    )
    if not present:
        raise ValueError(
            "TwoRoom OOD eval dataset has neither "
            f"{SPEED_COLUMN!r} nor {DOOR_COLUMN!r}"
        )
    return present


def register_tworoom_eval_env(
    env_id: str = TWOROOM_EVAL_ENV_ID,
) -> str:
    """Register a TwoRoom subclass with a strict post-reset factor setter.

    StableWM's stock dataset-driven evaluator restores state and goal but not
    variation values.  The subclass changes no reset, render, collision, or
    step logic; it only exposes an atomic setter that the evaluator can invoke
    before restoring state and goal.
    """

    import gymnasium as gym
    from stable_worldmodel.envs.two_room.env import TwoRoomEnv

    if env_id in gym.registry:
        entry_point = gym.spec(env_id).entry_point
        if getattr(entry_point, "_contextworld_eval_env", False):
            return env_id
        raise RuntimeError(
            f"Gym id {env_id!r} is already registered by a different entry point"
        )

    class ContextWorldTwoRoomEvalEnv(TwoRoomEnv):
        _contextworld_eval_env = True

        def _set_contextworld_variation(
            self,
            agent_speed: Any | None = None,
            door_position: Any | None = None,
        ) -> None:
            values = compile_tworoom_eval_variations(
                agent_speed=agent_speed,
                door_position=door_position,
            )
            self.variation_space.set_value(values)
            self._cache_params()

            # Door geometry changes the target rendering as well as the live
            # frame.  The later _set_goal_state call refreshes this again with
            # the dataset goal; refreshing here keeps the setter independently
            # correct and makes call-order bugs observable in tests.
            self._target_img = self._render_frame(
                agent_pos=self.target_position
            )

            observed: dict[str, np.ndarray] = {}
            for factor_key, expected in values.items():
                group, name = factor_key.split(".", maxsplit=1)
                actual = np.asarray(
                    self.variation_space[group][name].value
                )
                if not np.array_equal(actual, np.asarray(expected)):
                    raise RuntimeError(
                        f"TwoRoom eval factor readback failed for {factor_key}: "
                        f"expected {np.asarray(expected).tolist()}, "
                        f"observed {actual.tolist()}"
                    )
                observed[factor_key] = actual.copy()
            self._contextworld_variation_readback = observed

    ContextWorldTwoRoomEvalEnv.__name__ = "ContextWorldTwoRoomEvalEnv"
    ContextWorldTwoRoomEvalEnv.__qualname__ = "ContextWorldTwoRoomEvalEnv"
    gym.register(id=env_id, entry_point=ContextWorldTwoRoomEvalEnv)
    return env_id
