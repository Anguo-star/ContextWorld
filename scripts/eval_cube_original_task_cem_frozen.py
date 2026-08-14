#!/usr/bin/env python3
"""Evaluate frozen Cube checkpoints with the standard paired-query CEM protocol.

The evaluator deliberately takes an explicit Stable-WorldModel checkout and a
pre-materialized query catalog.  It therefore never imports the mutable sibling
checkout and every baseline/candidate sees the exact same original-Cube rows.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Sequence

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
import numpy as np
from omegaconf import OmegaConf
from sklearn import preprocessing
import torch


os.environ.setdefault("MUJOCO_GL", "egl")

stable_eval: Any = None
swm: Any = None
CEMSolver: Any = None
STABLE_WORLD_MODEL_ROOT: Path | None = None
STABLE_TRAIN_CONFIG_ROOT: Path | None = None


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(
    root: Path, *args: str, check: bool = True
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "git",
            f"--git-dir={root / '.git'}",
            f"--work-tree={root}",
            *args,
        ],
        check=check,
        capture_output=True,
        text=True,
    )


def verify_runtime(root: Path, expected_ref: str) -> dict[str, Any]:
    resolved = root.expanduser().resolve()
    if not (resolved / ".git").exists():
        raise FileNotFoundError(f"Stable-WorldModel checkout has no .git: {resolved}")
    observed_ref = _git(resolved, "rev-parse", "HEAD").stdout.strip()
    if observed_ref != expected_ref:
        raise RuntimeError(
            f"Stable-WorldModel ref drifted: expected {expected_ref}, got {observed_ref}"
        )
    status = _git(resolved, "status", "--porcelain", "--untracked-files=all").stdout
    if status:
        raise RuntimeError("Stable-WorldModel checkout is not clean")
    return {"root": str(resolved), "commit": observed_ref, "clean": True}


def install_runtime(root: Path, expected_ref: str) -> dict[str, Any]:
    global CEMSolver
    global STABLE_TRAIN_CONFIG_ROOT
    global STABLE_WORLD_MODEL_ROOT
    global stable_eval
    global swm

    identity = verify_runtime(root, expected_ref)
    resolved = Path(identity["root"])
    plan_root = resolved / "scripts/plan"
    train_config_root = resolved / "scripts/train/config"
    for required in (
        plan_root / "eval_wm.py",
        train_config_root / "lewm.yaml",
    ):
        if not required.is_file():
            raise FileNotFoundError(required)
    for source_root in (resolved, plan_root):
        value = str(source_root)
        if value not in sys.path:
            sys.path.insert(0, value)
    stable_eval = importlib.import_module("eval_wm")
    swm = importlib.import_module("stable_worldmodel")
    CEMSolver = importlib.import_module("stable_worldmodel.solver").CEMSolver
    STABLE_WORLD_MODEL_ROOT = resolved
    STABLE_TRAIN_CONFIG_ROOT = train_config_root
    return identity


def checkpoint_config_path(path: Path) -> Path:
    if path.suffix == ".pt":
        config = path.parent / "config.json"
    elif path.suffix == ".ckpt":
        config = path.parent / "config.yaml"
    else:
        raise ValueError(
            "Checkpoint must be a current .pt or legacy Lightning .ckpt: "
            f"{path}"
        )
    if not config.is_file() or config.is_symlink():
        raise FileNotFoundError(config)
    return config


def parse_models(values: Sequence[str]) -> dict[str, Path]:
    models: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--model must use NAME=CHECKPOINT")
        name, raw_path = value.split("=", 1)
        path = Path(raw_path).expanduser().resolve()
        if not name or name in models:
            raise ValueError(f"Invalid or duplicate model name {name!r}")
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(path)
        checkpoint_config_path(path)
        models[name] = path
    if not models:
        raise ValueError("At least one --model is required")
    return models


def valid_query_rows(dataset: Any) -> tuple[np.ndarray, str]:
    episode_column = (
        "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    )
    episode_values = np.asarray(dataset.get_col_data(episode_column))
    step_values = np.asarray(dataset.get_col_data("step_idx"))
    _, inverse = np.unique(episode_values, return_inverse=True)
    maximum_step = np.full(
        int(inverse.max()) + 1,
        np.iinfo(step_values.dtype).min,
        dtype=step_values.dtype,
    )
    np.maximum.at(maximum_step, inverse, step_values)
    max_start = maximum_step - 25
    valid = step_values <= max_start[inverse]
    return np.nonzero(valid)[0], episode_column


def select_queries(
    dataset: Any,
    *,
    valid_rows: np.ndarray,
    episode_column: str,
    seed: int,
    count: int,
) -> dict[str, np.ndarray]:
    generator = np.random.default_rng(seed)
    # Preserve the upstream evaluator's historical final-index exclusion.
    positions = generator.choice(len(valid_rows) - 1, size=count, replace=False)
    rows = np.sort(valid_rows[positions])
    payload = dataset.get_row_data(rows)
    return {
        "row_indices": rows,
        "episode_indices": np.asarray(payload[episode_column]),
        "start_steps": np.asarray(payload["step_idx"]),
    }


def _catalog_payload(
    dataset: Any, *, seeds: tuple[int, ...], count: int
) -> dict[str, Any]:
    valid_rows, episode_column = valid_query_rows(dataset)
    queries = {
        seed: select_queries(
            dataset,
            valid_rows=valid_rows,
            episode_column=episode_column,
            seed=seed,
            count=count,
        )
        for seed in seeds
    }
    return {
        "schema_version": 1,
        "task": "cube",
        "selection": {
            "algorithm": "numpy_default_rng_choice_sorted_valid_rows",
            "historical_final_index_exclusion": True,
            "goal_offset_steps": 25,
            "eval_seeds": list(seeds),
            "queries_per_seed": count,
        },
        "queries": {
            str(seed): {
                key: [int(value) for value in values]
                for key, values in payload.items()
            }
            for seed, payload in queries.items()
        },
    }


def _load_catalog(
    path: Path, *, dataset: Any, seeds: tuple[int, ...], count: int
) -> tuple[dict[str, Any], dict[int, dict[str, np.ndarray]]]:
    catalog_path = path.expanduser().resolve()
    if not catalog_path.is_file() or catalog_path.is_symlink():
        raise FileNotFoundError(catalog_path)
    payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    expected_selection = {
        "algorithm": "numpy_default_rng_choice_sorted_valid_rows",
        "historical_final_index_exclusion": True,
        "goal_offset_steps": 25,
        "eval_seeds": list(seeds),
        "queries_per_seed": count,
    }
    if (
        payload.get("schema_version") != 1
        or payload.get("task") != "cube"
        or payload.get("selection") != expected_selection
        or set(payload.get("queries", {})) != {str(seed) for seed in seeds}
    ):
        raise RuntimeError("Frozen Cube CEM query catalog contract drifted")
    _, episode_column = valid_query_rows(dataset)
    queries: dict[int, dict[str, np.ndarray]] = {}
    for seed in seeds:
        row = payload["queries"][str(seed)]
        rows = np.asarray(row.get("row_indices", ()), dtype=np.int64)
        episodes = np.asarray(row.get("episode_indices", ()), dtype=np.int64)
        starts = np.asarray(row.get("start_steps", ()), dtype=np.int64)
        if (
            len(rows) != count
            or len(np.unique(rows)) != count
            or not np.array_equal(rows, np.sort(rows))
            or len(episodes) != count
            or len(starts) != count
        ):
            raise RuntimeError(f"Frozen Cube CEM query count drifted for seed {seed}")
        observed = dataset.get_row_data(rows)
        if not np.array_equal(
            episodes, np.asarray(observed[episode_column], dtype=np.int64)
        ) or not np.array_equal(
            starts, np.asarray(observed["step_idx"], dtype=np.int64)
        ):
            raise RuntimeError(f"Frozen Cube CEM query identity drifted for seed {seed}")
        queries[seed] = {
            "row_indices": rows,
            "episode_indices": episodes,
            "start_steps": starts,
        }
    return payload, queries


def build_processors(dataset: Any) -> dict[str, Any]:
    processor = preprocessing.StandardScaler()
    values = dataset.get_col_data("action")
    values = values[~np.isnan(values).any(axis=1)]
    processor.fit(values)
    return {"action": processor}


def load_legacy_lightning_model(checkpoint: Path) -> Any:
    if STABLE_TRAIN_CONFIG_ROOT is None:
        raise RuntimeError("Stable-WorldModel runtime is not installed")
    config_path = checkpoint_config_path(checkpoint)
    legacy = OmegaConf.load(config_path)
    output_model_name = str(legacy.get("output_model_name", "")).lower()
    if "pldm" in output_model_name:
        family = "pldm"
    elif "lewm" in output_model_name:
        family = "lewm"
    else:
        raise ValueError(
            "Could not infer legacy model family from output_model_name in "
            f"{config_path}"
        )
    action_dim = int(legacy.wm.action_dim)
    frameskip = int(legacy.data.dataset.frameskip)
    action_input_dim = action_dim * frameskip
    with initialize_config_dir(
        config_dir=str(STABLE_TRAIN_CONFIG_ROOT), version_base=None
    ):
        current = compose(
            config_name=family,
            overrides=[f"model.action_encoder.input_dim={action_input_dim}"],
        )
    model = instantiate(current.model)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(
        payload.get("state_dict"), dict
    ):
        raise ValueError(f"Legacy checkpoint has no Lightning state_dict: {checkpoint}")
    model_state = {
        key.removeprefix("model."): value
        for key, value in payload["state_dict"].items()
        if key.startswith("model.")
    }
    if not model_state:
        raise ValueError(f"Legacy checkpoint has no model.* namespace: {checkpoint}")
    model.load_state_dict(model_state, strict=True)
    return model


def load_checkpoint_model(checkpoint: Path) -> Any:
    if checkpoint.suffix == ".ckpt":
        return load_legacy_lightning_model(checkpoint)
    if swm is None:
        raise RuntimeError("Stable-WorldModel runtime is not installed")
    return swm.wm.utils.load_pretrained(str(checkpoint))


def build_policy(
    checkpoint: Path,
    *,
    device: torch.device,
    seed: int,
    processors: dict[str, Any],
) -> Any:
    if stable_eval is None or swm is None or CEMSolver is None:
        raise RuntimeError("Stable-WorldModel runtime is not installed")
    model = load_checkpoint_model(checkpoint)
    model = model.to(device).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    solver = CEMSolver(
        model=stable_eval.ActionPaddedCostModel(model, action_block=5),
        batch_size=1,
        num_samples=300,
        var_scale=1.0,
        n_steps=30,
        topk=30,
        device=str(device),
        seed=seed,
    )
    transform_cfg = OmegaConf.create({"eval": {"img_size": 224}})
    transform = {
        "pixels": stable_eval.img_transform(transform_cfg),
        "goal": stable_eval.img_transform(transform_cfg),
    }
    return swm.policy.WorldModelPolicy(
        solver=solver,
        config=swm.PlanConfig(
            horizon=5,
            receding_horizon=5,
            history_len=3,
            action_block=5,
        ),
        process=processors,
        transform=transform,
    )


def _parse_seeds(raw: str) -> tuple[int, ...]:
    seeds = tuple(int(value) for value in raw.split(",") if value)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("--eval-seeds must be unique and non-empty")
    return seeds


def _load_cube_dataset(path: Path) -> Any:
    if swm is None:
        raise RuntimeError("Stable-WorldModel runtime is not installed")
    dataset = path.expanduser().resolve()
    if not dataset.is_file() or dataset.is_symlink():
        raise FileNotFoundError(dataset)
    return swm.data.load_dataset(str(dataset), keys_to_cache=["action"])


def prepare_queries(args: argparse.Namespace) -> None:
    runtime = install_runtime(args.stable_worldmodel_root, args.expected_ref)
    seeds = _parse_seeds(args.eval_seeds)
    if args.num_eval <= 0:
        raise ValueError("--num-eval must be positive")
    dataset_path = args.dataset.expanduser().resolve()
    dataset = _load_cube_dataset(dataset_path)
    payload = _catalog_payload(dataset, seeds=seeds, count=args.num_eval)
    payload["runtime"] = runtime
    payload["dataset"] = {
        "path": str(dataset_path),
        "size_bytes": dataset_path.stat().st_size,
        "sha256": args.expected_dataset_sha256,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": file_sha256(output),
                "query_count": len(seeds) * args.num_eval,
            },
            sort_keys=True,
        )
    )


def preflight_models(args: argparse.Namespace) -> None:
    runtime = install_runtime(args.stable_worldmodel_root, args.expected_ref)
    models = parse_models(args.model)
    rows = []
    for name, checkpoint in models.items():
        model = load_checkpoint_model(checkpoint)
        parameter_count = sum(int(value.numel()) for value in model.parameters())
        rows.append(
            {
                "model": name,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": file_sha256(checkpoint),
                "config_sha256": file_sha256(checkpoint_config_path(checkpoint)),
                "parameter_count": parameter_count,
                "strict_load": True,
            }
        )
        del model
    print(json.dumps({"runtime": runtime, "models": rows}, sort_keys=True))


def evaluate(args: argparse.Namespace) -> None:
    runtime = install_runtime(args.stable_worldmodel_root, args.expected_ref)
    models = parse_models(args.model)
    if len(models) != 1:
        raise ValueError("Frozen Cube CEM evaluator requires exactly one model per job")
    seeds = _parse_seeds(args.eval_seeds)
    if args.num_eval <= 0:
        raise ValueError("--num-eval must be positive")
    dataset_path = args.dataset.expanduser().resolve()
    if dataset_path.stat().st_size != args.expected_dataset_size:
        raise RuntimeError("Original Cube H5 size drifted")
    dataset = _load_cube_dataset(dataset_path)
    _, queries = _load_catalog(
        args.query_catalog,
        dataset=dataset,
        seeds=seeds,
        count=args.num_eval,
    )
    processors = build_processors(dataset)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite output: {output}")
    output.mkdir(parents=True)
    catalog_copy = output / "query_catalog.json"
    catalog_copy.write_bytes(args.query_catalog.expanduser().resolve().read_bytes())

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    specification = {
        "world": {
            "env_name": "swm/OGBCube-v0",
            "env_type": "single",
            "ob_type": "states",
            "multiview": False,
            "width": 224,
            "height": 224,
            "visualize_info": False,
            "terminate_at_goal": True,
        },
        "callables": [
            {
                "method": "set_state",
                "args": {
                    "qpos": {"value": "qpos"},
                    "qvel": {"value": "qvel"},
                },
            },
            {
                "method": "set_target_pos",
                "args": {
                    "cube_id": {"value": 0, "in_dataset": False},
                    "target_pos": {"value": "goal_privileged_block_0_pos"},
                    "target_quat": {"value": "goal_privileged_block_0_quat"},
                },
            },
        ],
    }
    model_results = []
    for model_name, checkpoint in models.items():
        config_path = checkpoint_config_path(checkpoint)
        seed_results = []
        for seed in seeds:
            print(f"[cube/{model_name}] CEM seed={seed}", flush=True)
            policy = build_policy(
                checkpoint,
                device=device,
                seed=seed,
                processors=processors,
            )
            world = swm.World(
                **specification["world"],
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
                callables=specification["callables"],
                video=None,
            )
            elapsed = time.monotonic() - started
            successes = [bool(value) for value in metrics["episode_successes"]]
            if len(successes) != args.num_eval:
                raise RuntimeError("Cube CEM evaluator returned the wrong episode count")
            row = {
                "eval_seed": seed,
                "query_count": len(successes),
                "success_count": sum(successes),
                "success_rate": sum(successes) / len(successes),
                "episode_successes": successes,
                "elapsed_seconds": elapsed,
            }
            seed_results.append(row)
            print(
                f"[cube/{model_name}] seed={seed} "
                f"success={row['success_count']}/{len(successes)}",
                flush=True,
            )
            del policy
            del world
            if device.type == "cuda":
                torch.cuda.empty_cache()
        all_successes = [
            value for row in seed_results for value in row["episode_successes"]
        ]
        model_results.append(
            {
                "model": model_name,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": file_sha256(checkpoint),
                "checkpoint_format": (
                    "legacy_lightning_ckpt"
                    if checkpoint.suffix == ".ckpt"
                    else "save_pretrained_pt"
                ),
                "config": str(config_path),
                "config_sha256": file_sha256(config_path),
                "seeds": seed_results,
                "aggregate": {
                    "success_count": sum(all_successes),
                    "evaluation_count": len(all_successes),
                    "success_rate": sum(all_successes) / len(all_successes),
                },
            }
        )
    plan_config = Path(runtime["root"]) / "scripts/plan/config/cube.yaml"
    report = {
        "schema_version": 1,
        "status": "standard_original_task_real_environment_cem",
        "task": "cube",
        "runtime": runtime,
        "protocol": {
            "source": str(plan_config),
            "source_sha256": file_sha256(plan_config),
            "dataset": str(dataset_path),
            "dataset_size_bytes": dataset_path.stat().st_size,
            "dataset_sha256": args.expected_dataset_sha256,
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
            "frozen_source": str(args.query_catalog.expanduser().resolve()),
            "path": str(catalog_copy),
            "sha256": file_sha256(catalog_copy),
        },
        "public_test": {
            "access_status": "closed_not_read_not_scored",
            "opened": False,
            "read": False,
            "hashed": False,
            "scored": False,
        },
        "models": model_results,
    }
    report_path = output / "aggregate.json"
    with report_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(
        json.dumps(
            {
                "report": str(report_path),
                "sha256": file_sha256(report_path),
                "models": {
                    row["model"]: row["aggregate"] for row in model_results
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_runtime(target: argparse.ArgumentParser) -> None:
        target.add_argument("--stable-worldmodel-root", type=Path, required=True)
        target.add_argument("--expected-ref", required=True)

    prepare = subparsers.add_parser("prepare-queries")
    add_runtime(prepare)
    prepare.add_argument("--dataset", type=Path, required=True)
    prepare.add_argument("--expected-dataset-sha256", required=True)
    prepare.add_argument("--eval-seeds", default="42,43,44")
    prepare.add_argument("--num-eval", type=int, default=100)
    prepare.add_argument("--output", type=Path, required=True)

    preflight = subparsers.add_parser("preflight-models")
    add_runtime(preflight)
    preflight.add_argument("--model", action="append", default=[])

    evaluate_parser = subparsers.add_parser("eval")
    add_runtime(evaluate_parser)
    evaluate_parser.add_argument("--model", action="append", default=[])
    evaluate_parser.add_argument("--dataset", type=Path, required=True)
    evaluate_parser.add_argument("--expected-dataset-size", type=int, required=True)
    evaluate_parser.add_argument("--expected-dataset-sha256", required=True)
    evaluate_parser.add_argument("--query-catalog", type=Path, required=True)
    evaluate_parser.add_argument("--output", type=Path, required=True)
    evaluate_parser.add_argument("--eval-seeds", default="42,43,44")
    evaluate_parser.add_argument("--num-eval", type=int, default=100)
    evaluate_parser.add_argument("--device", default="cuda:0")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "prepare-queries":
        prepare_queries(args)
    elif args.command == "preflight-models":
        preflight_models(args)
    elif args.command == "eval":
        evaluate(args)
    else:  # pragma: no cover - argparse enforces the command set.
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
