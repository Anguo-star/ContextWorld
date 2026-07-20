from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any

from .models import CompiledScenario
from .lance import build_lance_writer
from .reset_constraints import apply_tworoom_reset_constraints


def collect_scenario(
    swm: ModuleType,
    scenario: CompiledScenario,
    collection_config: dict[str, Any],
    *,
    resume: bool = False,
) -> str:
    """Collect one immutable scenario table using the original expert policy."""

    if scenario.output_path.exists():
        if resume:
            return "reused"
        raise FileExistsError(
            f"Refusing to append to existing scenario table {scenario.output_path}. "
            "Use --resume to validate/reuse it, or bump the experiment version."
        )

    # Import only after the configured Stable-WorldModel checkout is on sys.path.
    from stable_worldmodel.envs.two_room import ExpertPolicy

    scenario.output_path.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(collection_config.get("staging_root", "/tmp"))
    staging_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="contextworld-lance-", dir=staging_root
    ) as temporary:
        staged_path = Path(temporary) / scenario.output_path.name
        world = swm.World(
            scenario.env_id,
            num_envs=int(collection_config["num_envs"]),
            max_episode_steps=scenario.max_episode_steps,
            image_shape=scenario.image_shape,
            render_mode="rgb_array",
        )
        apply_tworoom_reset_constraints(world, scenario.reset_constraints)
        policy_config = collection_config["policy"]
        world.set_policy(
            ExpertPolicy(
                action_noise=float(policy_config["action_noise"]),
                action_repeat_prob=float(policy_config["action_repeat_prob"]),
                seed=scenario.policy_seed,
            )
        )
        try:
            writer = build_lance_writer(
                swm,
                staged_path,
                pixel_codec=scenario.pixel_codec,
            )
            world.collect(
                episodes=scenario.episodes,
                seed=scenario.env_seed,
                options={
                    "variation": scenario.variation,
                    "variation_values": scenario.variation_values,
                },
                writer=writer,
                progress=False,
            )
        finally:
            world.close()

        try:
            shutil.copytree(staged_path, scenario.output_path)
        except BaseException:
            if scenario.output_path.exists():
                shutil.rmtree(scenario.output_path)
            raise
    return "collected"
