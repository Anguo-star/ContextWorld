#!/usr/bin/env python3
"""Evaluate a frozen PushT or Reacher original-task CEM catalog.

This additive evaluator closes the interfaces that the historical wrappers
left implicit: it accepts the canonical legacy Lightning checkpoints, a
pre-materialized query catalog, an explicit clean Stable-WorldModel checkout,
and exact dataset/catalog identities.  It never samples replacement queries.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Sequence

import numpy as np
import torch
from sklearn import preprocessing


ROOT = Path(__file__).resolve().parents[1]
CUBE_EVALUATOR = ROOT / "scripts/eval_cube_original_task_cem_frozen.py"
SPEC = importlib.util.spec_from_file_location(
    "contextworld_frozen_cem_runtime_v1", CUBE_EVALUATOR
)
if SPEC is None or SPEC.loader is None:  # pragma: no cover
    raise ImportError(CUBE_EVALUATOR)
base: Any = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(base)

from contextworld.evaluation.icl_model import state_dict_sha256  # noqa: E402


TASKS: dict[str, dict[str, Any]] = {
    "pusht": {
        "mujoco_gl": "egl",
        "world": {"env_name": "swm/PushT-v1"},
        "callables": [
            {
                "method": "_set_state",
                "args": {"state": {"value": "state"}},
            },
            {
                "method": "_set_goal_state",
                "args": {"goal_state": {"value": "goal_state"}},
            },
        ],
        "cache_keys": ["action", "proprio", "state"],
        "processor_columns": ["action", "proprio", "state"],
        "default_seeds": "42,43,44,45,46,47",
        "default_num_eval": 50,
    },
    "reacher": {
        "mujoco_gl": "osmesa",
        "world": {
            "env_name": "swm/ReacherDMControl-v0",
            "task": "qpos_match",
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
                "method": "set_target_qpos",
                "args": {"target_qpos": {"value": "goal_qpos"}},
            },
        ],
        "cache_keys": ["action"],
        "processor_columns": ["action"],
        "default_seeds": "42,43,44",
        "default_num_eval": 100,
    },
}


def _seeds(raw: str) -> tuple[int, ...]:
    values = tuple(int(value) for value in raw.split(",") if value.strip())
    if not values or len(values) != len(set(values)):
        raise ValueError("--eval-seeds must be unique and non-empty")
    return values


def _assert_file_identity(
    path: Path,
    *,
    expected_sha256: str,
    expected_size: int | None,
    label: str,
    hash_content: bool = True,
) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise FileNotFoundError(resolved)
    if expected_size is not None and resolved.stat().st_size != expected_size:
        raise RuntimeError(f"{label} size drifted")
    observed = base.file_sha256(resolved) if hash_content else expected_sha256
    if observed != expected_sha256:
        raise RuntimeError(
            f"{label} SHA-256 drifted: expected {expected_sha256}, got {observed}"
        )
    return {
        "path": str(resolved),
        "sha256": observed,
        "size_bytes": resolved.stat().st_size,
        "content_hash_checked_in_job": hash_content,
    }


def _load_dataset(path: Path, *, cache_keys: list[str]) -> Any:
    if base.swm is None:
        raise RuntimeError("Stable-WorldModel runtime is not installed")
    return base.swm.data.load_dataset(str(path), keys_to_cache=cache_keys)


def _processors(dataset: Any, columns: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for column in columns:
        processor = preprocessing.StandardScaler()
        values = np.asarray(dataset.get_col_data(column))
        values = values[~np.isnan(values).any(axis=1)]
        processor.fit(values)
        result[column] = processor
        if column != "action":
            result[f"goal_{column}"] = processor
    return result


def _load_frozen_queries(
    path: Path,
    *,
    dataset: Any,
    seeds: tuple[int, ...],
    count: int,
) -> tuple[dict[str, Any], dict[int, dict[str, np.ndarray]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows_by_seed = payload.get("queries", payload)
    if not isinstance(rows_by_seed, dict) or set(rows_by_seed) != {
        str(seed) for seed in seeds
    }:
        raise RuntimeError("Frozen CEM catalog seed set drifted")
    valid_rows, episode_column = base.valid_query_rows(dataset)
    queries: dict[int, dict[str, np.ndarray]] = {}
    for seed in seeds:
        row = rows_by_seed[str(seed)]
        if not isinstance(row, dict):
            raise RuntimeError(f"Frozen CEM catalog row is invalid for seed {seed}")
        indices = np.asarray(row.get("row_indices", ()), dtype=np.int64)
        episodes = np.asarray(row.get("episode_indices", ()), dtype=np.int64)
        starts = np.asarray(row.get("start_steps", ()), dtype=np.int64)
        if (
            len(indices) != count
            or len(np.unique(indices)) != count
            or not np.array_equal(indices, np.sort(indices))
            or len(episodes) != count
            or len(starts) != count
        ):
            raise RuntimeError(f"Frozen CEM query count drifted for seed {seed}")
        if np.any(indices < 0):
            raise RuntimeError(f"Frozen CEM catalog has a negative row for seed {seed}")
        positions = np.searchsorted(valid_rows, indices)
        if (
            np.any(positions >= len(valid_rows))
            or not np.array_equal(valid_rows[positions], indices)
        ):
            raise RuntimeError(
                f"Frozen CEM catalog has non-eligible starts for seed {seed}"
            )
        observed = dataset.get_row_data(indices)
        if not np.array_equal(
            episodes, np.asarray(observed[episode_column], dtype=np.int64)
        ) or not np.array_equal(
            starts, np.asarray(observed["step_idx"], dtype=np.int64)
        ):
            raise RuntimeError(f"Frozen CEM query identity drifted for seed {seed}")
        queries[seed] = {
            "row_indices": indices,
            "episode_indices": episodes,
            "start_steps": starts,
        }
    return payload, queries


def _load_and_validate_evaluation_inputs(
    args: argparse.Namespace, *, task: dict[str, Any]
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    Any,
    dict[int, dict[str, np.ndarray]],
    dict[str, Any],
]:
    """Close the frozen dataset/catalog contract without evaluating CEM."""

    if args.num_eval <= 0:
        raise ValueError("--num-eval must be positive")
    dataset_identity = _assert_file_identity(
        args.dataset,
        expected_sha256=args.expected_dataset_sha256,
        expected_size=args.expected_dataset_size,
        label="original-task dataset",
        hash_content=False,
    )
    audit_identity = _assert_file_identity(
        args.input_identity_audit,
        expected_sha256=args.expected_input_identity_audit_sha256,
        expected_size=args.expected_input_identity_audit_size,
        label="frozen full-file input identity audit",
    )
    audit_payload = json.loads(
        Path(audit_identity["path"]).read_text(encoding="utf-8")
    )
    audit_row = audit_payload.get("datasets", {}).get(args.task)
    if (
        audit_payload.get("schema_version") != 1
        or audit_payload.get("audit_id")
        != "contextworld_original_baseline_cem_input_identity_audit_v1"
        or not isinstance(audit_row, dict)
        or audit_row.get("content_hash_checked") is not True
        or Path(str(audit_row.get("path", ""))).expanduser().resolve()
        != Path(dataset_identity["path"])
        or str(audit_row.get("sha256", "")) != dataset_identity["sha256"]
        or int(audit_row.get("size_bytes", -1)) != dataset_identity["size_bytes"]
    ):
        raise RuntimeError("Dataset identity is not closed by the frozen input audit")
    dataset_identity["content_hash_authority"] = audit_identity
    catalog_identity = _assert_file_identity(
        args.query_catalog,
        expected_sha256=args.expected_catalog_sha256,
        expected_size=args.expected_catalog_size,
        label="frozen query catalog",
    )
    dataset = _load_dataset(
        Path(dataset_identity["path"]), cache_keys=task["cache_keys"]
    )
    _, queries = _load_frozen_queries(
        Path(catalog_identity["path"]),
        dataset=dataset,
        seeds=_seeds(args.eval_seeds),
        count=args.num_eval,
    )
    processors = _processors(dataset, task["processor_columns"])
    return dataset_identity, catalog_identity, dataset, queries, processors


def _validate_task_callables(
    world: Any, *, task: dict[str, Any], dataset: Any
) -> list[dict[str, Any]]:
    """Verify that StableWM will not silently skip a frozen setup callable.

    ``World._apply_callables`` deliberately ignores unavailable methods.  That
    behavior is convenient for generic evaluation but unacceptable for this
    frozen protocol, so preflight verifies both method and argument bindings
    on the actual unwrapped environment without invoking the methods.
    """

    environments = getattr(getattr(world, "envs", None), "envs", None)
    if not isinstance(environments, (list, tuple)) or len(environments) != 1:
        raise RuntimeError("CEM preflight did not construct exactly one environment")
    environment = getattr(environments[0], "unwrapped", None)
    if environment is None:
        raise RuntimeError("CEM preflight environment has no unwrapped target")
    columns = {str(value) for value in dataset.column_names}
    available_values = columns | {f"goal_{column}" for column in columns}
    verified: list[dict[str, Any]] = []
    for specification in task["callables"]:
        method_name = str(specification.get("method", ""))
        method = getattr(environment, method_name, None)
        if not callable(method):
            raise RuntimeError(
                f"Frozen task callable is unavailable on the environment: {method_name}"
            )
        arguments = specification.get("args", {})
        if not isinstance(arguments, dict):
            raise RuntimeError(f"Frozen task callable args are invalid: {method_name}")
        bound_arguments: dict[str, object] = {}
        for name, source in arguments.items():
            if not isinstance(source, dict):
                raise RuntimeError(
                    f"Frozen task callable argument is invalid: {method_name}.{name}"
                )
            if source.get("in_dataset", True):
                value = source.get("value")
                if not isinstance(value, str) or value not in available_values:
                    raise RuntimeError(
                        "Frozen task callable source is unavailable in the dataset: "
                        f"{method_name}.{name}={value!r}"
                    )
            bound_arguments[str(name)] = object()
        try:
            inspect.signature(method).bind(**bound_arguments)
        except (TypeError, ValueError) as error:
            raise RuntimeError(
                f"Frozen task callable signature drifted: {method_name}"
            ) from error
        verified.append(
            {
                "method": method_name,
                "arguments": sorted(bound_arguments),
            }
        )
    return verified


def _preflight_world(
    *, task: dict[str, Any], dataset: Any
) -> dict[str, Any]:
    """Construct and validate one environment without calling ``evaluate``."""

    if base.swm is None:
        raise RuntimeError("Stable-WorldModel runtime is not installed")
    world = base.swm.World(
        **task["world"],
        num_envs=1,
        max_episode_steps=100,
        image_shape=(224, 224),
    )
    try:
        callables = _validate_task_callables(world, task=task, dataset=dataset)
        return {
            "world_constructed": True,
            "num_envs": 1,
            "task_callables": callables,
            "world_evaluate_called": False,
            "cem_episodes_consumed": 0,
            "mujoco_gl": os.environ.get("MUJOCO_GL"),
        }
    finally:
        world.close()


def _policy_model_identity(policy: Any) -> dict[str, Any]:
    """Hash the exact model instance used by one CEM policy."""

    solver = getattr(policy, "solver", None)
    cost_model = getattr(solver, "model", None)
    model = getattr(cost_model, "model", cost_model)
    if model is None or not callable(getattr(model, "state_dict", None)):
        raise RuntimeError("Could not locate the model used by the CEM policy")
    return {
        "state_dict_sha256": state_dict_sha256(model),
        "parameter_count": sum(
            int(parameter.numel()) for parameter in model.parameters()
        ),
    }


def _model_identity(checkpoint: Path) -> dict[str, Any]:
    model = base.load_checkpoint_model(checkpoint)
    try:
        return {
            "state_dict_sha256": state_dict_sha256(model),
            "parameter_count": sum(
                int(parameter.numel()) for parameter in model.parameters()
            ),
        }
    finally:
        del model
        gc.collect()


def _checkpoint_format(checkpoint: Path) -> str:
    """Report the on-disk serialization format used by the evaluator."""

    return (
        "legacy_lightning_ckpt"
        if checkpoint.suffix == ".ckpt"
        else "save_pretrained_pt"
    )


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    task = TASKS[args.task]
    if os.environ.get("MUJOCO_GL") != task["mujoco_gl"]:
        raise RuntimeError(
            f"{args.task} CEM requires MUJOCO_GL={task['mujoco_gl']}"
        )
    runtime = base.install_runtime(args.stable_worldmodel_root, args.expected_ref)
    plan_config = Path(runtime["root"]) / f"scripts/plan/config/{args.task}.yaml"
    plan_config_identity = _assert_file_identity(
        plan_config,
        expected_sha256=args.expected_plan_config_sha256,
        expected_size=args.expected_plan_config_size,
        label="frozen planning config",
    )
    models = base.parse_models(args.model)
    if len(models) != 1:
        raise ValueError("Frozen CEM preflight requires exactly one model")
    checkpoint = next(iter(models.values()))
    checkpoint_identity = _assert_file_identity(
        checkpoint,
        expected_sha256=args.expected_checkpoint_sha256,
        expected_size=args.expected_checkpoint_size,
        label="canonical checkpoint",
    )
    config_identity = _assert_file_identity(
        base.checkpoint_config_path(checkpoint),
        expected_sha256=args.expected_config_sha256,
        expected_size=args.expected_config_size,
        label="canonical checkpoint config",
    )
    row = _model_identity(checkpoint)
    row.update(
        {
            "checkpoint": checkpoint_identity,
            "config": config_identity,
            "strict_load": True,
        }
    )
    (
        dataset_identity,
        catalog_identity,
        dataset,
        queries,
        processors,
    ) = _load_and_validate_evaluation_inputs(args, task=task)
    environment_preflight = _preflight_world(task=task, dataset=dataset)
    return {
        "runtime": runtime,
        "plan_config": plan_config_identity,
        "model": row,
        "inputs": {
            "dataset": dataset_identity,
            "query_catalog": catalog_identity,
            "query_count": sum(len(value["row_indices"]) for value in queries.values()),
            "processor_columns": sorted(processors),
        },
        "environment_preflight": environment_preflight,
        "cem_episodes_consumed": 0,
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    task = TASKS[args.task]
    if os.environ.get("MUJOCO_GL") != task["mujoco_gl"]:
        raise RuntimeError(
            f"{args.task} CEM requires MUJOCO_GL={task['mujoco_gl']}"
        )
    runtime = base.install_runtime(args.stable_worldmodel_root, args.expected_ref)
    plan_config = Path(runtime["root"]) / f"scripts/plan/config/{args.task}.yaml"
    plan_config_identity = _assert_file_identity(
        plan_config,
        expected_sha256=args.expected_plan_config_sha256,
        expected_size=args.expected_plan_config_size,
        label="frozen planning config",
    )
    models = base.parse_models(args.model)
    if len(models) != 1:
        raise ValueError("Frozen CEM evaluator requires exactly one model per job")
    model_name, checkpoint = next(iter(models.items()))
    checkpoint_identity = _assert_file_identity(
        checkpoint,
        expected_sha256=args.expected_checkpoint_sha256,
        expected_size=args.expected_checkpoint_size,
        label="canonical checkpoint",
    )
    config_path = base.checkpoint_config_path(checkpoint)
    config_identity = _assert_file_identity(
        config_path,
        expected_sha256=args.expected_config_sha256,
        expected_size=args.expected_config_size,
        label="canonical checkpoint config",
    )
    seeds = _seeds(args.eval_seeds)
    if args.num_eval <= 0:
        raise ValueError("--num-eval must be positive")
    (
        dataset_identity,
        catalog_identity,
        dataset,
        queries,
        processors,
    ) = _load_and_validate_evaluation_inputs(args, task=task)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite output: {output}")
    output.mkdir(parents=True)
    catalog_copy = output / "query_catalog.json"
    catalog_copy.write_bytes(Path(catalog_identity["path"]).read_bytes())

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    seed_results: list[dict[str, Any]] = []
    for seed in seeds:
        print(f"[{args.task}/{model_name}] CEM seed={seed}", flush=True)
        policy = base.build_policy(
            checkpoint,
            device=device,
            seed=seed,
            processors=processors,
        )
        before = _policy_model_identity(policy)
        world: Any | None = None
        try:
            world = base.swm.World(
                **task["world"],
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
                callables=task["callables"],
                video=None,
            )
            elapsed = time.monotonic() - started
            successes = [bool(value) for value in metrics["episode_successes"]]
            if len(successes) != args.num_eval:
                raise RuntimeError("CEM evaluator returned the wrong episode count")
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
                f"[{args.task}/{model_name}] seed={seed} "
                f"success={row['success_count']}/{len(successes)}",
                flush=True,
            )
        finally:
            after = _policy_model_identity(policy)
            if world is not None:
                world.close()
            del policy
            if world is not None:
                del world
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        if before != after:
            raise RuntimeError(
                f"Frozen model identity changed during CEM evaluation for seed {seed}"
            )
        seed_results[-1]["frozen_state_audit"] = {
            "before": before,
            "after": after,
            "passed": True,
        }
    all_successes = [
        value for row in seed_results for value in row["episode_successes"]
    ]
    report = {
        "schema_version": 1,
        "status": "standard_original_task_real_environment_cem",
        "task": args.task,
        "runtime": runtime,
        "protocol": {
            "source": str(plan_config),
            "source_sha256": plan_config_identity["sha256"],
            "source_size_bytes": plan_config_identity["size_bytes"],
            "dataset": dataset_identity,
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
            "frozen_source": catalog_identity,
            "path": str(catalog_copy),
            "sha256": base.file_sha256(catalog_copy),
        },
        "public_test": {
            "access_status": "not_applicable_original_environment_dataset",
            "contextworld_public_test_read": False,
            "contextworld_public_test_scored": False,
        },
        "model": {
            "model": model_name,
            "checkpoint": checkpoint_identity["path"],
            "checkpoint_sha256": checkpoint_identity["sha256"],
            "checkpoint_size_bytes": checkpoint_identity["size_bytes"],
            "checkpoint_format": _checkpoint_format(checkpoint),
            "config": config_identity["path"],
            "config_sha256": config_identity["sha256"],
            "config_size_bytes": config_identity["size_bytes"],
            "frozen_state_audit": {
                "scope": "actual_policy_model_per_seed",
                "passed": all(
                    bool(row["frozen_state_audit"]["passed"])
                    for row in seed_results
                ),
                "seeds": [
                    {
                        "eval_seed": row["eval_seed"],
                        **row["frozen_state_audit"],
                    }
                    for row in seed_results
                ],
            },
            "seeds": seed_results,
            "aggregate": {
                "success_count": sum(all_successes),
                "evaluation_count": len(all_successes),
                "success_rate": sum(all_successes) / len(all_successes),
            },
        },
    }
    report_path = output / "aggregate.json"
    with report_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return {
        "report": str(report_path),
        "sha256": base.file_sha256(report_path),
        "aggregate": report["model"]["aggregate"],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight-model", "eval"))
    parser.add_argument("--task", choices=tuple(TASKS), required=True)
    parser.add_argument("--stable-worldmodel-root", type=Path, required=True)
    parser.add_argument("--expected-ref", required=True)
    parser.add_argument("--expected-plan-config-sha256", required=True)
    parser.add_argument("--expected-plan-config-size", type=int, required=True)
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--expected-checkpoint-size", type=int, required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--expected-config-size", type=int, required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--expected-dataset-size", type=int)
    parser.add_argument("--expected-dataset-sha256")
    parser.add_argument("--input-identity-audit", type=Path)
    parser.add_argument("--expected-input-identity-audit-sha256")
    parser.add_argument("--expected-input-identity-audit-size", type=int)
    parser.add_argument("--query-catalog", type=Path)
    parser.add_argument("--expected-catalog-sha256")
    parser.add_argument("--expected-catalog-size", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--eval-seeds")
    parser.add_argument("--num-eval", type=int)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    task = TASKS[args.task]
    args.eval_seeds = args.eval_seeds or task["default_seeds"]
    args.num_eval = args.num_eval or task["default_num_eval"]
    if args.command in {"preflight-model", "eval"}:
        required = (
            "dataset",
            "expected_dataset_size",
            "expected_dataset_sha256",
            "input_identity_audit",
            "expected_input_identity_audit_sha256",
            "expected_input_identity_audit_size",
            "query_catalog",
            "expected_catalog_sha256",
            "expected_catalog_size",
        )
        if args.command == "eval":
            required = (*required, "output")
        missing = [name for name in required if getattr(args, name) is None]
        if missing:
            parser.error(f"{args.command} requires: " + ", ".join(missing))
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = preflight(args) if args.command == "preflight-model" else evaluate(args)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
