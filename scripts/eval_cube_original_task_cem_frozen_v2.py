#!/usr/bin/env python3
"""OSMesa recovery wrapper for the frozen original-Cube CEM evaluator."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any, Sequence


if os.environ.get("MUJOCO_GL") not in {None, "osmesa"}:
    raise RuntimeError("Cube CEM v2 requires MUJOCO_GL=osmesa")
os.environ["MUJOCO_GL"] = "osmesa"

ROOT = Path(__file__).resolve().parents[1]
V1_EVALUATOR = ROOT / "scripts/eval_cube_original_task_cem_frozen.py"
SPEC = importlib.util.spec_from_file_location("cube_cem_v1_frozen", V1_EVALUATOR)
if SPEC is None or SPEC.loader is None:
    raise ImportError(V1_EVALUATOR)
v1: Any = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(v1)


def preflight_models(args: Any) -> None:
    runtime = v1.install_runtime(args.stable_worldmodel_root, args.expected_ref)
    models = v1.parse_models(args.model)
    rows = []
    for name, checkpoint in models.items():
        model = v1.load_checkpoint_model(checkpoint)
        rows.append(
            {
                "model": name,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": v1.file_sha256(checkpoint),
                "config_sha256": v1.file_sha256(
                    v1.checkpoint_config_path(checkpoint)
                ),
                "parameter_count": sum(
                    int(value.numel()) for value in model.parameters()
                ),
                "strict_load": True,
            }
        )
        del model

    world = v1.swm.World(
        env_name="swm/OGBCube-v0",
        num_envs=1,
        max_episode_steps=100,
        env_type="single",
        ob_type="states",
        multiview=False,
        width=224,
        height=224,
        visualize_info=False,
        terminate_at_goal=True,
        image_shape=(224, 224),
    )
    environment_preflight = {
        "mujoco_gl": os.environ["MUJOCO_GL"],
        "world_constructed": True,
        "num_envs": int(world.num_envs),
        "world_evaluate_called": False,
        "cem_episodes_consumed": 0,
    }
    del world
    print(
        json.dumps(
            {
                "runtime": runtime,
                "models": rows,
                "environment_preflight": environment_preflight,
            },
            sort_keys=True,
        )
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = v1.parse_args(argv)
    if args.command == "preflight-models":
        preflight_models(args)
    elif args.command == "prepare-queries":
        v1.prepare_queries(args)
    elif args.command == "eval":
        print("[contextworld] MUJOCO_GL=osmesa", flush=True)
        v1.evaluate(args)
    else:  # pragma: no cover - argparse enforces the command set.
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
