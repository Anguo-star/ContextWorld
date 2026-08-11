#!/usr/bin/env python3
"""Evaluate standard Push-T CEM ability without writing rollout videos.

This is a result-compatible, audit-oriented wrapper around
``stable-worldmodel/scripts/plan/eval_wm.py``.  It uses the same environment,
dataset query selection, preprocessing, PlanConfig, and CEM hyperparameters,
but serializes query identities and per-episode outcomes as JSON and disables
video generation.  Multiple checkpoints evaluated with the same seeds
therefore admit paired non-inferiority analysis.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn import preprocessing


os.environ.setdefault("MUJOCO_GL", "egl")

CONTEXTWORLD_ROOT = Path(__file__).resolve().parents[1]
STABLE_WORLD_MODEL_ROOT = CONTEXTWORLD_ROOT.parent / "stable-worldmodel"
STABLE_PLAN_ROOT = STABLE_WORLD_MODEL_ROOT / "scripts/plan"
for source_root in (
    CONTEXTWORLD_ROOT,
    STABLE_WORLD_MODEL_ROOT,
    STABLE_PLAN_ROOT,
):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from contextworld.paths import artifact_path  # noqa: E402
import eval_wm as stable_eval  # noqa: E402
import stable_worldmodel as swm  # noqa: E402
from stable_worldmodel.solver import CEMSolver  # noqa: E402


DEFAULT_DATASET = Path(
    "/opt/huawei/explorer-env/dataset/ag_data/data/world_model/quentinll/"
    "pusht_expert_train.h5"
)
DEFAULT_OUTPUT = artifact_path(
    "evaluation/history3/pusht_hidden_actuation_h3_v1/"
    "mixed_retention_seed3073_step2048/standard_pusht_cem"
)
CALLABLES = [
    {
        "method": "_set_state",
        "args": {"state": {"value": "state"}},
    },
    {
        "method": "_set_goal_state",
        "args": {"goal_state": {"value": "goal_state"}},
    },
]


def file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_dataset(path: Path):
    return swm.data.load_dataset(
        str(path),
        keys_to_cache=["action", "proprio", "state"],
    )


def build_processors(dataset) -> dict[str, Any]:
    processors = {}
    for column in ("action", "proprio", "state"):
        processor = preprocessing.StandardScaler()
        values = dataset.get_col_data(column)
        values = values[~np.isnan(values).any(axis=1)]
        processor.fit(values)
        processors[column] = processor
        if column != "action":
            processors[f"goal_{column}"] = processor
    return processors


def valid_query_rows(dataset) -> tuple[np.ndarray, str]:
    episode_column = (
        "episode_idx"
        if "episode_idx" in dataset.column_names
        else "ep_idx"
    )
    episode_indices = np.unique(dataset.get_col_data(episode_column))
    episode_lengths = stable_eval.get_episodes_length(
        dataset,
        episode_indices,
    )
    max_start = episode_lengths - 25 - 1
    max_start_by_episode = {
        episode_id: max_start[index]
        for index, episode_id in enumerate(episode_indices)
    }
    max_start_per_row = np.asarray(
        [
            max_start_by_episode[episode_id]
            for episode_id in dataset.get_col_data(episode_column)
        ]
    )
    valid = dataset.get_col_data("step_idx") <= max_start_per_row
    return np.nonzero(valid)[0], episode_column


def select_queries(
    dataset,
    *,
    valid_rows: np.ndarray,
    episode_column: str,
    seed: int,
    count: int,
) -> dict[str, np.ndarray]:
    generator = np.random.default_rng(seed)
    # Preserve eval_wm.py's registered selection expression exactly,
    # including its historical exclusion of the final valid-row index.
    positions = generator.choice(
        len(valid_rows) - 1,
        size=count,
        replace=False,
    )
    rows = np.sort(valid_rows[positions])
    payload = dataset.get_row_data(rows)
    return {
        "row_indices": rows,
        "episode_indices": np.asarray(payload[episode_column]),
        "start_steps": np.asarray(payload["step_idx"]),
    }


def build_policy(
    checkpoint: Path,
    *,
    device: torch.device,
    seed: int,
    processors: dict[str, Any],
):
    model = swm.wm.utils.load_pretrained(str(checkpoint))
    model = model.to(device).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    wrapped = stable_eval.ActionPaddedCostModel(model, action_block=5)
    solver = CEMSolver(
        model=wrapped,
        batch_size=1,
        num_samples=300,
        var_scale=1.0,
        n_steps=30,
        topk=30,
        device=str(device),
        seed=seed,
    )
    config = swm.PlanConfig(
        horizon=5,
        receding_horizon=5,
        history_len=3,
        action_block=5,
    )
    transform_cfg = OmegaConf.create({"eval": {"img_size": 224}})
    transform = {
        "pixels": stable_eval.img_transform(transform_cfg),
        "goal": stable_eval.img_transform(transform_cfg),
    }
    return swm.policy.WorldModelPolicy(
        solver=solver,
        config=config,
        process=processors,
        transform=transform,
    )


def parse_models(values: list[str]) -> dict[str, Path]:
    models = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--model must use NAME=CHECKPOINT")
        name, raw_path = value.split("=", 1)
        name = name.strip()
        path = Path(raw_path).expanduser().resolve()
        if not name or name in models:
            raise ValueError(f"Invalid or duplicate model name {name!r}")
        if not path.exists():
            raise FileNotFoundError(path)
        if not (path.parent / "config.json").exists():
            raise FileNotFoundError(path.parent / "config.json")
        models[name] = path
    if not models:
        raise ValueError("At least one --model is required")
    return models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="NAME=CHECKPOINT; may be repeated",
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--eval-seeds", default="42,43,44,45,46,47")
    parser.add_argument("--num-eval", type=int, default=50)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models = parse_models(args.model)
    seeds = tuple(
        int(value)
        for value in args.eval_seeds.split(",")
        if value.strip()
    )
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("--eval-seeds must be a unique non-empty list")
    if args.num_eval <= 0:
        raise ValueError("--num-eval must be positive")
    dataset_path = args.dataset.expanduser().resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)
    output = Path(os.path.abspath(args.output.expanduser()))
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite output: {output}")
    output.mkdir(parents=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    dataset = load_dataset(dataset_path)
    processors = build_processors(dataset)
    valid_rows, episode_column = valid_query_rows(dataset)
    queries = {
        seed: select_queries(
            dataset,
            valid_rows=valid_rows,
            episode_column=episode_column,
            seed=seed,
            count=args.num_eval,
        )
        for seed in seeds
    }
    query_identity = {
        str(seed): {
            key: [int(value) for value in values]
            for key, values in payload.items()
        }
        for seed, payload in queries.items()
    }
    query_path = output / "query_catalog.json"
    query_path.write_text(
        json.dumps(query_identity, indent=2, sort_keys=True) + "\n"
    )

    results = []
    for model_name, checkpoint in models.items():
        checkpoint_results = []
        for seed in seeds:
            print(
                f"[{model_name}] standard Push-T CEM seed={seed}",
                flush=True,
            )
            policy = build_policy(
                checkpoint,
                device=device,
                seed=seed,
                processors=processors,
            )
            world = swm.World(
                env_name="swm/PushT-v1",
                num_envs=args.num_eval,
                max_episode_steps=100,
                image_shape=(224, 224),
            )
            world.set_policy(policy)
            selected = queries[seed]
            started = time.monotonic()
            metrics = world.evaluate(
                dataset=dataset,
                start_steps=selected["start_steps"].tolist(),
                goal_offset=25,
                eval_budget=50,
                episodes_idx=selected["episode_indices"].tolist(),
                callables=CALLABLES,
                video=None,
            )
            elapsed = time.monotonic() - started
            successes = [
                bool(value) for value in metrics["episode_successes"]
            ]
            row = {
                "eval_seed": seed,
                "query_count": args.num_eval,
                "success_count": sum(successes),
                "success_rate": sum(successes) / len(successes),
                "episode_successes": successes,
                "elapsed_seconds": elapsed,
            }
            checkpoint_results.append(row)
            print(
                f"[{model_name}] seed={seed} "
                f"success={row['success_count']}/{args.num_eval}",
                flush=True,
            )
        aggregate_successes = [
            success
            for row in checkpoint_results
            for success in row["episode_successes"]
        ]
        model_result = {
            "model": model_name,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": file_sha256(checkpoint),
            "seeds": checkpoint_results,
            "aggregate": {
                "success_count": sum(aggregate_successes),
                "evaluation_count": len(aggregate_successes),
                "success_rate": (
                    sum(aggregate_successes) / len(aggregate_successes)
                ),
            },
        }
        results.append(model_result)
        partial = {
            "schema_version": 1,
            "status": "standard_pusht_real_environment_cem",
            "protocol": {
                "source": (
                    "stable-worldmodel/scripts/plan/config/pusht.yaml"
                ),
                "dataset": str(dataset_path),
                "dataset_size_bytes": dataset_path.stat().st_size,
                "num_eval_per_seed": args.num_eval,
                "eval_seeds": list(seeds),
                "goal_offset_steps": 25,
                "eval_budget": 50,
                "history_len": 3,
                "horizon": 5,
                "receding_horizon": 5,
                "action_block": 5,
                "cem_samples": 300,
                "cem_iterations": 30,
                "cem_topk": 30,
                "videos_written": False,
            },
            "query_catalog": {
                "path": str(query_path),
                "sha256": file_sha256(query_path),
            },
            "models": results,
        }
        (output / "aggregate.partial.json").write_text(
            json.dumps(partial, indent=2, sort_keys=True) + "\n"
        )
        del policy
        del world
        if device.type == "cuda":
            torch.cuda.empty_cache()

    report = {
        "schema_version": 1,
        "status": "standard_pusht_real_environment_cem",
        "protocol": {
            "source": "stable-worldmodel/scripts/plan/config/pusht.yaml",
            "dataset": str(dataset_path),
            "dataset_size_bytes": dataset_path.stat().st_size,
            "num_eval_per_seed": args.num_eval,
            "eval_seeds": list(seeds),
            "goal_offset_steps": 25,
            "eval_budget": 50,
            "history_len": 3,
            "horizon": 5,
            "receding_horizon": 5,
            "action_block": 5,
            "cem_samples": 300,
            "cem_iterations": 30,
            "cem_topk": 30,
            "videos_written": False,
        },
        "query_catalog": {
            "path": str(query_path),
            "sha256": file_sha256(query_path),
        },
        "models": results,
    }
    report_path = output / "aggregate.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "report": str(report_path),
                "sha256": file_sha256(report_path),
                "models": {
                    row["model"]: row["aggregate"] for row in results
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
