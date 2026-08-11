from __future__ import annotations

from typing import Any


def register_contextworld_environment(env_id: str) -> str:
    """Register a ContextWorld-owned Gym environment when one is requested."""

    from contextworld.evaluation.action_delay_env import (
        ACTION_DELAY_ENV_ID,
        register_action_delay_env,
    )
    from contextworld.evaluation.hidden_passage_env import (
        HIDDEN_PASSAGE_ENV_ID,
        register_hidden_passage_env,
    )
    from contextworld.evaluation.portal_exit_env import (
        PORTAL_EXIT_ENV_ID,
        register_portal_exit_env,
    )

    if env_id == ACTION_DELAY_ENV_ID:
        return register_action_delay_env()
    if env_id == HIDDEN_PASSAGE_ENV_ID:
        return register_hidden_passage_env()
    if env_id == PORTAL_EXIT_ENV_ID:
        return register_portal_exit_env()
    return env_id


def make_raw_contextworld_environment(
    env_id: str,
    **kwargs: Any,
):
    """Create the raw environment used for exact replay."""

    from contextworld.evaluation.action_delay_env import (
        ACTION_DELAY_ENV_ID,
        make_action_delay_env,
    )
    from contextworld.evaluation.hidden_passage_env import (
        HIDDEN_PASSAGE_ENV_ID,
        make_hidden_passage_env,
    )
    from contextworld.evaluation.portal_exit_env import (
        PORTAL_EXIT_ENV_ID,
        make_portal_exit_env,
    )

    if env_id == ACTION_DELAY_ENV_ID:
        return make_action_delay_env(**kwargs)
    if env_id == HIDDEN_PASSAGE_ENV_ID:
        return make_hidden_passage_env(**kwargs)
    if env_id == PORTAL_EXIT_ENV_ID:
        return make_portal_exit_env(**kwargs)

    if env_id == "swm/TwoRoom-v1":
        from stable_worldmodel.envs.two_room.env import TwoRoomEnv

        return TwoRoomEnv(**kwargs)
    raise ValueError(f"Unsupported ContextWorld replay environment {env_id!r}")


__all__ = [
    "make_raw_contextworld_environment",
    "register_contextworld_environment",
]
