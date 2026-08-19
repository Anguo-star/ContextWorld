#!/usr/bin/env python3
"""Run the one-shot Cube PLDM original-baseline CEM protocol.

This is intentionally additive to ``eval_cube_original_task_cem_frozen_v2.py``.
It fixes one *descriptive* original-environment cell: the canonical legacy
Cube PLDM checkpoint on one pre-frozen three-seed / 300-episode catalog.  The
wrapper is fail-closed: it verifies every supplied input identity, requires a
clean pinned Stable-WorldModel checkout, never samples replacement queries,
and refuses to reuse an output namespace.

``--preflight`` performs the same identity, strict-load, catalog, and renderer
checks as evaluation, but never calls ``World.evaluate`` and therefore
consumes zero CEM episodes.
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


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

V2_EVALUATOR = ROOT / "scripts/eval_cube_original_task_cem_frozen.py"
SPEC = importlib.util.spec_from_file_location(
    "contextworld_cube_original_baseline_cem_v1_v2", V2_EVALUATOR
)
if SPEC is None or SPEC.loader is None:  # pragma: no cover - repository error.
    raise ImportError(V2_EVALUATOR)
v2: Any = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(v2)
base: Any = v2

from contextworld.evaluation.icl_model import state_dict_sha256  # noqa: E402


MODEL_ID = "cube_pldm_original"
EVAL_SEEDS = (42, 43, 44)
QUERIES_PER_SEED = 100
TOTAL_EVALUATIONS = len(EVAL_SEEDS) * QUERIES_PER_SEED

CUBE_WORLD = {
    "env_name": "swm/OGBCube-v0",
    "env_type": "single",
    "ob_type": "states",
    "multiview": False,
    "width": 224,
    "height": 224,
    "visualize_info": False,
    "terminate_at_goal": True,
}
CUBE_CALLABLES = [
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
]


def _require_osmesa() -> None:
    if os.environ.get("MUJOCO_GL") != "osmesa":
        raise RuntimeError("Cube PLDM CEM requires MUJOCO_GL=osmesa")


def _require_sha256(value: str, *, label: str) -> str:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{label} must be a lowercase SHA-256 hex digest")
    return value


def _assert_file_identity(
    path: Path,
    *,
    expected_sha256: str,
    expected_size: int,
    label: str,
    hash_content: bool = True,
) -> dict[str, Any]:
    """Hash and size-check a regular non-symlink file before it is read."""

    expected = _require_sha256(expected_sha256, label=f"{label} SHA-256")
    if expected_size < 0:
        raise ValueError(f"{label} expected size must be non-negative")
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise FileNotFoundError(resolved)
    observed_size = resolved.stat().st_size
    if observed_size != expected_size:
        raise RuntimeError(
            f"{label} size drifted: expected {expected_size}, got {observed_size}"
        )
    observed_sha256 = base.file_sha256(resolved) if hash_content else expected
    if observed_sha256 != expected:
        raise RuntimeError(
            f"{label} SHA-256 drifted: expected {expected}, got {observed_sha256}"
        )
    return {
        "path": str(resolved),
        "sha256": observed_sha256,
        "size_bytes": observed_size,
        "content_hash_checked_in_job": hash_content,
    }


def _assert_pldm_config(config_path: Path) -> None:
    """Prevent a correctly-hashed but wrong-family legacy checkpoint wiring."""

    legacy = base.OmegaConf.load(config_path)
    output_model_name = str(legacy.get("output_model_name", "")).lower()
    if "pldm" not in output_model_name or "cube" not in output_model_name:
        raise RuntimeError(
            "Cube original-baseline wrapper requires a Cube PLDM legacy config"
        )


def _model_state(model: Any) -> dict[str, Any]:
    return {
        "state_dict_sha256": state_dict_sha256(model),
        "parameter_count": sum(int(parameter.numel()) for parameter in model.parameters()),
        "training": bool(model.training),
    }


def _strict_load_state(checkpoint: Path) -> dict[str, Any]:
    """Strict-load a separate copy for preflight without starting CEM."""

    model = base.load_checkpoint_model(checkpoint)
    try:
        return {**_model_state(model), "strict_load": True}
    finally:
        del model
        gc.collect()


def _build_policy_with_model(
    checkpoint: Path,
    *,
    device: torch.device,
    seed: int,
    processors: dict[str, Any],
) -> tuple[Any, Any]:
    """Reuse the frozen Cube core while retaining the actual raw model handle."""

    if base.stable_eval is None or base.swm is None or base.CEMSolver is None:
        raise RuntimeError("Stable-WorldModel runtime is not installed")
    model = base.load_checkpoint_model(checkpoint)
    model = model.to(device).eval()
    model.requires_grad_(False)
    # This is the weight-preserving setting used by the frozen Cube evaluator.
    model.interpolate_pos_encoding = True
    solver = base.CEMSolver(
        model=base.stable_eval.ActionPaddedCostModel(model, action_block=5),
        batch_size=1,
        num_samples=300,
        var_scale=1.0,
        n_steps=30,
        topk=30,
        device=str(device),
        seed=seed,
    )
    transform_cfg = base.OmegaConf.create({"eval": {"img_size": 224}})
    transform = {
        "pixels": base.stable_eval.img_transform(transform_cfg),
        "goal": base.stable_eval.img_transform(transform_cfg),
    }
    policy = base.swm.policy.WorldModelPolicy(
        solver=solver,
        config=base.swm.PlanConfig(
            horizon=5,
            receding_horizon=5,
            history_len=3,
            action_block=5,
        ),
        process=processors,
        transform=transform,
    )
    return policy, model


def _reserve_output(path: Path) -> Path:
    output = path.expanduser().resolve()
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise FileExistsError(f"Refusing to overwrite output: {output}") from error
    return output


def _copy_catalog_exclusively(source: Path, output: Path) -> Path:
    target = output / "query_catalog.json"
    with target.open("xb") as stream:
        stream.write(source.read_bytes())
    return target


def _assert_query_contract(
    payload: dict[str, Any],
    queries: dict[int, dict[str, Any]],
) -> None:
    expected_selection = {
        "algorithm": "numpy_default_rng_choice_sorted_valid_rows",
        "historical_final_index_exclusion": True,
        "goal_offset_steps": 25,
        "eval_seeds": list(EVAL_SEEDS),
        "queries_per_seed": QUERIES_PER_SEED,
    }
    if payload.get("selection") != expected_selection:
        raise RuntimeError("Frozen Cube PLDM query selection contract drifted")
    if set(queries) != set(EVAL_SEEDS) or any(
        len(queries[seed]["row_indices"]) != QUERIES_PER_SEED
        for seed in EVAL_SEEDS
    ):
        raise RuntimeError("Frozen Cube PLDM query count contract drifted")


def _verify_direct_rows(
    dataset: Any,
    *,
    rows: np.ndarray,
    episodes: np.ndarray,
    starts: np.ndarray,
) -> None:
    """Verify only the frozen H5 columns needed for start/goal identity.

    ``HDF5Dataset.get_row_data`` reads every key, including 224px images and
    object/string metadata.  That made 300 frozen random-row checks take many
    minutes.  Direct column reads preserve the identity and +25 eligibility
    contract without touching unrelated observations.
    """

    episode_column = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    opener = getattr(dataset, "_open", None)
    if callable(opener):
        opener()
    handle = getattr(dataset, "h5_file", None)
    if handle is None:
        raise RuntimeError("Cube dataset does not expose its frozen HDF5 handle")
    if np.any(rows < 0) or np.any(rows + 25 >= len(handle[episode_column])):
        raise RuntimeError("Frozen Cube catalog has an out-of-range start")
    observed_episodes = np.asarray(handle[episode_column][rows], dtype=np.int64)
    observed_starts = np.asarray(handle["step_idx"][rows], dtype=np.int64)
    goal_episodes = np.asarray(handle[episode_column][rows + 25], dtype=np.int64)
    goal_steps = np.asarray(handle["step_idx"][rows + 25], dtype=np.int64)
    if (
        not np.array_equal(observed_episodes, episodes)
        or not np.array_equal(observed_starts, starts)
        or not np.array_equal(goal_episodes, episodes)
        or not np.array_equal(goal_steps, starts + 25)
    ):
        raise RuntimeError("Frozen Cube catalog has non-eligible starts")


def _load_frozen_catalog(
    path: Path, *, dataset: Any
) -> tuple[dict[str, Any], dict[int, dict[str, np.ndarray]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema_version") != 1
        or payload.get("task") != "cube"
        or set(payload.get("queries", {})) != {str(seed) for seed in EVAL_SEEDS}
    ):
        raise RuntimeError("Frozen Cube PLDM catalog envelope drifted")
    queries: dict[int, dict[str, np.ndarray]] = {}
    for seed in EVAL_SEEDS:
        source = payload["queries"][str(seed)]
        rows = np.asarray(source.get("row_indices", ()), dtype=np.int64)
        episodes = np.asarray(source.get("episode_indices", ()), dtype=np.int64)
        starts = np.asarray(source.get("start_steps", ()), dtype=np.int64)
        if (
            len(rows) != QUERIES_PER_SEED
            or len(np.unique(rows)) != QUERIES_PER_SEED
            or not np.array_equal(rows, np.sort(rows))
            or len(episodes) != QUERIES_PER_SEED
            or len(starts) != QUERIES_PER_SEED
        ):
            raise RuntimeError(f"Frozen Cube query count drifted for seed {seed}")
        _verify_direct_rows(
            dataset, rows=rows, episodes=episodes, starts=starts
        )
        queries[seed] = {
            "row_indices": rows,
            "episode_indices": episodes,
            "start_steps": starts,
        }
    _assert_query_contract(payload, queries)
    return payload, queries


def _prepare_inputs(args: argparse.Namespace) -> dict[str, Any]:
    """Close every identity and validate the exact Cube 3x100 query contract."""

    runtime = base.install_runtime(args.stable_worldmodel_root, args.expected_ref)
    plan_config = Path(runtime["root"]) / "scripts/plan/config/cube.yaml"
    plan_config_identity = _assert_file_identity(
        plan_config,
        expected_sha256=args.expected_plan_config_sha256,
        expected_size=args.expected_plan_config_size,
        label="frozen Cube planning config",
    )
    checkpoint = _assert_file_identity(
        args.checkpoint,
        expected_sha256=args.expected_checkpoint_sha256,
        expected_size=args.expected_checkpoint_size,
        label="canonical Cube PLDM checkpoint",
    )
    checkpoint_path = Path(checkpoint["path"])
    if checkpoint_path.suffix != ".ckpt":
        raise RuntimeError("Cube original PLDM checkpoint must use legacy .ckpt format")
    config_path = base.checkpoint_config_path(checkpoint_path)
    config = _assert_file_identity(
        config_path,
        expected_sha256=args.expected_config_sha256,
        expected_size=args.expected_config_size,
        label="canonical Cube PLDM config",
    )
    _assert_pldm_config(Path(config["path"]))
    dataset = _assert_file_identity(
        args.dataset,
        expected_sha256=args.expected_dataset_sha256,
        expected_size=args.expected_dataset_size,
        label="original Cube dataset",
        hash_content=False,
    )
    audit = _assert_file_identity(
        args.input_identity_audit,
        expected_sha256=args.expected_input_identity_audit_sha256,
        expected_size=args.expected_input_identity_audit_size,
        label="frozen full-file input identity audit",
    )
    audit_payload = json.loads(Path(audit["path"]).read_text(encoding="utf-8"))
    audit_row = audit_payload.get("datasets", {}).get("cube")
    if (
        audit_payload.get("schema_version") != 1
        or audit_payload.get("audit_id")
        != "contextworld_original_baseline_cem_input_identity_audit_v1"
        or not isinstance(audit_row, dict)
        or audit_row.get("content_hash_checked") is not True
        or Path(str(audit_row.get("path", ""))).expanduser().resolve()
        != Path(dataset["path"])
        or str(audit_row.get("sha256", "")) != dataset["sha256"]
        or int(audit_row.get("size_bytes", -1)) != dataset["size_bytes"]
    ):
        raise RuntimeError("Cube dataset identity is not closed by the input audit")
    dataset["content_hash_authority"] = audit
    catalog = _assert_file_identity(
        args.query_catalog,
        expected_sha256=args.expected_catalog_sha256,
        expected_size=args.expected_catalog_size,
        label="frozen Cube query catalog",
    )
    loaded_dataset = base._load_cube_dataset(Path(dataset["path"]))
    catalog_payload, queries = _load_frozen_catalog(
        Path(catalog["path"]), dataset=loaded_dataset
    )
    return {
        "runtime": runtime,
        "plan_config": plan_config_identity,
        "checkpoint": checkpoint,
        "config": config,
        "dataset": dataset,
        "catalog": catalog,
        "catalog_payload": catalog_payload,
        "loaded_dataset": loaded_dataset,
        "queries": queries,
    }


def _renderer_preflight(dataset: Any) -> dict[str, Any]:
    if base.swm is None:
        raise RuntimeError("Stable-WorldModel runtime is not installed")
    world = base.swm.World(
        **CUBE_WORLD,
        num_envs=1,
        max_episode_steps=100,
        image_shape=(224, 224),
    )
    try:
        environments = getattr(getattr(world, "envs", None), "envs", None)
        if not isinstance(environments, (list, tuple)) or len(environments) != 1:
            raise RuntimeError("Cube preflight did not construct exactly one env")
        target = getattr(environments[0], "unwrapped", None)
        if target is None:
            raise RuntimeError("Cube preflight environment has no unwrapped target")
        columns = {str(value) for value in dataset.column_names}
        available = columns | {f"goal_{value}" for value in columns}
        verified: list[dict[str, Any]] = []
        for row in CUBE_CALLABLES:
            method_name = str(row["method"])
            method = getattr(target, method_name, None)
            if not callable(method):
                raise RuntimeError(f"Cube task callable is missing: {method_name}")
            bound: dict[str, object] = {}
            for name, source in row["args"].items():
                if source.get("in_dataset", True):
                    value = source.get("value")
                    if not isinstance(value, str) or value not in available:
                        raise RuntimeError(
                            f"Cube task callable source is missing: {method_name}.{name}"
                        )
                bound[str(name)] = object()
            try:
                inspect.signature(method).bind(**bound)
            except (TypeError, ValueError) as error:
                raise RuntimeError(
                    f"Cube task callable signature drifted: {method_name}"
                ) from error
            verified.append({"method": method_name, "arguments": sorted(bound)})
        return {
            "mujoco_gl": os.environ["MUJOCO_GL"],
            "world_constructed": True,
            "num_envs": int(world.num_envs),
            "world_evaluate_called": False,
            "task_callables": verified,
            "cem_episodes_consumed": 0,
        }
    finally:
        close = getattr(world, "close", None)
        if callable(close):
            close()
        del world
        gc.collect()


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    """Validate the complete execution chain without evaluating any episode."""

    _require_osmesa()
    prepared = _prepare_inputs(args)
    model_state = _strict_load_state(Path(prepared["checkpoint"]["path"]))
    renderer = _renderer_preflight(prepared["loaded_dataset"])
    return {
        "schema_version": 1,
        "status": "cube_pldm_original_baseline_cem_preflight_passed",
        "model_id": MODEL_ID,
        "runtime": prepared["runtime"],
        "plan_config": prepared["plan_config"],
        "inputs": {
            key: prepared[key] for key in ("checkpoint", "config", "dataset", "catalog")
        },
        "query_contract": {
            "eval_seeds": list(EVAL_SEEDS),
            "queries_per_seed": QUERIES_PER_SEED,
            "evaluation_count": TOTAL_EVALUATIONS,
            "catalog_selection": prepared["catalog_payload"]["selection"],
        },
        "strict_loaded_model": model_state,
        "renderer_preflight": renderer,
        "cem_episodes_consumed": 0,
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    _require_osmesa()
    prepared = _prepare_inputs(args)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    output = _reserve_output(args.output)
    catalog_copy = _copy_catalog_exclusively(Path(prepared["catalog"]["path"]), output)
    processors = base.build_processors(prepared["loaded_dataset"])
    seed_results: list[dict[str, Any]] = []
    loaded_states: list[str] = []

    for seed in EVAL_SEEDS:
        print(f"[cube/{MODEL_ID}] CEM seed={seed}", flush=True)
        policy, model = _build_policy_with_model(
            Path(prepared["checkpoint"]["path"]),
            device=device,
            seed=seed,
            processors=processors,
        )
        before = _model_state(model)
        loaded_states.append(before["state_dict_sha256"])
        world = base.swm.World(
            **CUBE_WORLD,
            num_envs=QUERIES_PER_SEED,
            max_episode_steps=100,
            image_shape=(224, 224),
        )
        try:
            world.set_policy(policy)
            selected = prepared["queries"][seed]
            started = time.monotonic()
            metrics = world.evaluate(
                dataset=prepared["loaded_dataset"],
                start_steps=selected["start_steps"].tolist(),
                goal_offset=25,
                eval_budget=50,
                episodes_idx=selected["episode_indices"].tolist(),
                callables=CUBE_CALLABLES,
                video=None,
            )
            elapsed = time.monotonic() - started
            successes = [bool(value) for value in metrics["episode_successes"]]
            if len(successes) != QUERIES_PER_SEED:
                raise RuntimeError("Cube CEM evaluator returned the wrong episode count")
            after = _model_state(model)
            if before != after:
                raise RuntimeError(
                    "Actual evaluated Cube PLDM model state changed during CEM"
                )
            row = {
                "eval_seed": seed,
                "query_count": len(successes),
                "success_count": sum(successes),
                "success_rate": sum(successes) / len(successes),
                "episode_successes": successes,
                "elapsed_seconds": elapsed,
                "actual_evaluated_model_state": {
                    "before": before,
                    "after": after,
                    "passed": True,
                },
            }
            seed_results.append(row)
            print(
                f"[cube/{MODEL_ID}] seed={seed} "
                f"success={row['success_count']}/{len(successes)}",
                flush=True,
            )
        finally:
            close = getattr(world, "close", None)
            if callable(close):
                close()
            del policy, world, model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    if len(set(loaded_states)) != 1:
        raise RuntimeError("Canonical checkpoint produced inconsistent loaded states")
    all_successes = [
        success for row in seed_results for success in row["episode_successes"]
    ]
    if len(all_successes) != TOTAL_EVALUATIONS:
        raise RuntimeError("Frozen Cube PLDM CEM total evaluation count drifted")
    plan_config = Path(prepared["plan_config"]["path"])
    report = {
        "schema_version": 1,
        "status": "standard_original_task_real_environment_cem",
        "task": "cube",
        "model_id": MODEL_ID,
        "runtime": prepared["runtime"],
        "protocol": {
            "source": str(plan_config),
            "source_sha256": prepared["plan_config"]["sha256"],
            "source_size_bytes": prepared["plan_config"]["size_bytes"],
            "dataset": prepared["dataset"],
            "eval_seeds": list(EVAL_SEEDS),
            "queries_per_seed": QUERIES_PER_SEED,
            "evaluation_count": TOTAL_EVALUATIONS,
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
            "frozen_source": prepared["catalog"],
            "path": str(catalog_copy),
            "sha256": base.file_sha256(catalog_copy),
            "size_bytes": catalog_copy.stat().st_size,
            "contract": {
                "eval_seeds": list(EVAL_SEEDS),
                "queries_per_seed": QUERIES_PER_SEED,
                "evaluation_count": TOTAL_EVALUATIONS,
            },
        },
        "public_test": {
            "access_status": "closed_not_read_not_scored",
            "opened": False,
            "read": False,
            "hashed": False,
            "scored": False,
        },
        "model": {
            "model": MODEL_ID,
            "family": "pldm",
            "checkpoint": prepared["checkpoint"],
            "config": prepared["config"],
            "strict_load": True,
            "loaded_state_consistent_across_seeds": True,
            "loaded_state_dict_sha256": loaded_states[0],
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
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="validate all identities and loadability without evaluating episodes",
    )
    parser.add_argument("--stable-worldmodel-root", type=Path, required=True)
    parser.add_argument("--expected-ref", required=True)
    parser.add_argument("--expected-plan-config-sha256", required=True)
    parser.add_argument("--expected-plan-config-size", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--expected-checkpoint-size", type=int, required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--expected-config-size", type=int, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--expected-dataset-sha256", required=True)
    parser.add_argument("--expected-dataset-size", type=int, required=True)
    parser.add_argument("--input-identity-audit", type=Path, required=True)
    parser.add_argument("--expected-input-identity-audit-sha256", required=True)
    parser.add_argument(
        "--expected-input-identity-audit-size", type=int, required=True
    )
    parser.add_argument("--query-catalog", type=Path, required=True)
    parser.add_argument("--expected-catalog-sha256", required=True)
    parser.add_argument("--expected-catalog-size", type=int, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    if args.preflight:
        if args.output is not None:
            parser.error("--preflight must not reserve or write an output directory")
    elif args.output is None:
        parser.error("evaluation requires --output")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = preflight(args) if args.preflight else evaluate(args)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
