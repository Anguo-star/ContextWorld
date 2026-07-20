#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# Keep the standalone script runnable from a fresh checkout without requiring
# an editable install first.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from contextworld.evaluation.protocol import (
    allocate_scenario_evaluations,
    factor_readback_audit,
    infer_model_protocol,
    load_catalog_regime,
    load_legacy_cost_model,
    load_pretrained_cost_model,
    frozen_normalizer_process,
    original_h5_process,
    scenario_seed,
    select_episode_balanced_starts,
)
from contextworld.evaluation.tworoom import (
    DOOR_COLUMN,
    SPEED_COLUMN,
    TWOROOM_EVAL_ENV_ID,
    register_tworoom_eval_env,
    tworoom_eval_callables,
    validate_tworoom_factor_columns,
)
from contextworld.synthesis.manifest import write_json
from contextworld.paths import artifact_path, resolve_contextworld_path
from contextworld.synthesis.stablewm import load_stable_worldmodel


DEFAULT_STABLEWM_REF = "5864b74980f6ed328fd0045e777b3865962eff43"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_transform(img_size: int):
    import stable_pretraining as spt
    import torch
    from torchvision.transforms import v2 as transforms

    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=img_size),
        ]
    )


def _make_policy(args, swm, world, process, model):
    if model is None:
        return swm.policy.RandomPolicy(seed=args.seed), None

    protocol = infer_model_protocol(model, action_dim=2)
    if args.action_block is not None and args.action_block != protocol["action_block"]:
        raise ValueError(
            f"Configured action_block={args.action_block} differs from model "
            f"action_block={protocol['action_block']}"
        )
    action_block = protocol["action_block"]
    if args.horizon * action_block > args.eval_budget:
        raise ValueError("horizon * action_block exceeds eval_budget")

    setattr(model, "history_size", protocol["history_size"])
    setattr(model, "interpolate_pos_encoding", True)
    solver = swm.solver.CEMSolver(
        model=model,
        batch_size=args.cem_batch_size,
        num_samples=args.cem_num_samples,
        var_scale=args.cem_var_scale,
        n_steps=args.cem_steps,
        topk=args.cem_topk,
        device=args.device,
        seed=args.seed,
    )
    config = swm.PlanConfig(
        horizon=args.horizon,
        receding_horizon=args.receding_horizon,
        history_len=protocol["history_size"],
        action_block=action_block,
        warm_start=True,
    )
    policy = swm.policy.WorldModelPolicy(
        solver=solver,
        config=config,
        process=process,
        transform={
            "pixels": image_transform(args.img_size),
            "goal": image_transform(args.img_size),
        },
    )
    return policy, protocol


def _scenario_dataset(swm, path: Path):
    probe = swm.data.LanceDataset(path=path, frameskip=1, num_steps=1)
    factors = validate_tworoom_factor_columns(probe.column_names)
    keys = ["pixels", "action", "proprio", "state", *factors]
    missing = [key for key in keys if key not in probe.column_names]
    if missing:
        raise KeyError(f"{path} is missing eval columns {missing}")
    return swm.data.LanceDataset(
        path=path,
        frameskip=1,
        num_steps=1,
        keys_to_load=keys,
    )


def evaluate_scenario(args, swm, path, process, model, evaluation_count):
    dataset = _scenario_dataset(swm, path)
    starts = select_episode_balanced_starts(
        dataset.lengths,
        goal_offset=args.goal_offset,
        count=evaluation_count,
        seed=scenario_seed(args.seed, path),
    )
    world = swm.World(
        TWOROOM_EVAL_ENV_ID,
        num_envs=len(starts.episodes),
        max_episode_steps=2 * args.eval_budget,
        image_shape=(args.img_size, args.img_size),
        render_mode="rgb_array",
    )
    try:
        policy, model_protocol = _make_policy(args, swm, world, process, model)
        world.set_policy(policy)
        started = time.time()
        metrics = world.evaluate(
            dataset=dataset,
            episodes_idx=starts.episodes,
            start_steps=starts.steps,
            goal_offset=args.goal_offset,
            eval_budget=args.eval_budget,
            callables=tworoom_eval_callables(),
            video=None,
        )
        elapsed = time.time() - started
        readback = factor_readback_audit(dataset, world, starts)
        if not readback["passed"]:
            raise RuntimeError(f"Factor readback failed for {path}: {readback}")
        chunks = dataset.load_chunk(
            np.asarray(starts.episodes, dtype=np.int64),
            np.asarray(starts.steps, dtype=np.int64),
            np.asarray(starts.steps, dtype=np.int64) + args.goal_offset + 1,
        )
        goal_states = np.stack(
            [
                np.asarray(
                    chunk["state"][-1]
                    if "state" in chunk
                    else chunk["proprio"][-1],
                    dtype=np.float32,
                )
                for chunk in chunks
            ]
        )
        final_states = np.stack(
            [
                np.asarray(
                    env.unwrapped.agent_position.detach().cpu().numpy(),
                    dtype=np.float32,
                )
                for env in world.envs.envs
            ]
        )
        final_distances = np.linalg.norm(final_states - goal_states, axis=1)
        return {
            "scenario": path.name,
            "dataset": str(path),
            "evaluations": len(starts.episodes),
            "episodes": [int(value) for value in starts.episodes],
            "start_steps": [int(value) for value in starts.steps],
            "successes": np.asarray(metrics["episode_successes"]).astype(int).tolist(),
            "goal_states": goal_states.tolist(),
            "final_states": final_states.tolist(),
            "final_distances": final_distances.tolist(),
            "success_rate": float(metrics["success_rate"]),
            "factor_readback": readback,
            "model_protocol": model_protocol,
            "elapsed_seconds": elapsed,
        }
    finally:
        world.close()


def run(args) -> dict:
    args.catalog = resolve_contextworld_path(args.catalog, repo_root=REPO_ROOT)
    args.output = resolve_contextworld_path(args.output, repo_root=REPO_ROOT)
    os.environ.setdefault("MUJOCO_GL", "egl")
    swm, stable_repo, stable_commit = load_stable_worldmodel(
        REPO_ROOT,
        args.stablewm_repo,
        args.stablewm_ref,
    )
    register_tworoom_eval_env()

    paths = load_catalog_regime(
        args.catalog.resolve(), args.regime, repo_root=REPO_ROOT
    )
    if args.scenario_indices is not None:
        if len(set(args.scenario_indices)) != len(args.scenario_indices):
            raise ValueError("--scenario-indices must not contain duplicates")
        invalid = [
            index
            for index in args.scenario_indices
            if index < 0 or index >= len(paths)
        ]
        if invalid:
            raise ValueError(
                f"Scenario indices out of range for {len(paths)} catalog entries: "
                f"{invalid}"
            )
        paths = [paths[index] for index in args.scenario_indices]
    if args.max_scenarios is not None:
        paths = paths[: args.max_scenarios]
    if not paths:
        raise ValueError("No scenarios selected")

    if args.num_eval is not None:
        scenario_counts = allocate_scenario_evaluations(
            scenario_count=len(paths),
            total_evaluations=args.num_eval,
            seed=args.seed,
        )
        budget_mode = "fixed_total_scenario_balanced"
    else:
        if args.evals_per_scenario is None or args.evals_per_scenario <= 0:
            raise ValueError("--evals-per-scenario must be positive")
        scenario_counts = [args.evals_per_scenario] * len(paths)
        budget_mode = "fixed_per_scenario"

    model = None
    process = {}
    policy_identity: dict[str, object] = {"kind": "random"}
    if args.policy_checkpoint is not None:
        if args.original_h5 is None:
            raise ValueError("Model eval requires --original-h5")
        process = (
            frozen_normalizer_process(args.normalizer.resolve())
            if args.normalizer is not None
            else original_h5_process(args.original_h5.resolve())
        )
        checkpoint = args.policy_checkpoint.resolve()
        if checkpoint.suffix.lower() == ".pt":
            model = load_pretrained_cost_model(
                checkpoint,
                swm,
                cache_dir=artifact_path(
                    "evaluation/model_cache", repo_root=REPO_ROOT
                ),
            )
            serialization = "stablewm_pretrained"
        else:
            if args.legacy_code_root is None:
                raise ValueError(
                    "Legacy object checkpoint eval requires --legacy-code-root"
                )
            model = load_legacy_cost_model(
                checkpoint, args.legacy_code_root.resolve()
            )
            serialization = "legacy_object"
        import torch

        model = model.to(args.device).eval()
        model.requires_grad_(False)
        policy_identity = {
            "kind": "world_model",
            "serialization": serialization,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "class": f"{type(model).__module__}.{type(model).__name__}",
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
        }

    results = []
    for index, (path, evaluation_count) in enumerate(
        zip(paths, scenario_counts), start=1
    ):
        print(
            f"[{index}/{len(paths)}] {path.name} "
            f"({evaluation_count} evaluations)",
            flush=True,
        )
        results.append(
            evaluate_scenario(
                args,
                swm,
                path,
                process,
                model,
                evaluation_count,
            )
        )

    successes = [value for result in results for value in result["successes"]]
    raw_records = []
    for result in results:
        for episode, start, success, final_state, goal_state, final_distance in zip(
            result["episodes"],
            result["start_steps"],
            result["successes"],
            result["final_states"],
            result["goal_states"],
            result["final_distances"],
            strict=True,
        ):
            raw_records.append(
                {
                    "evaluation_id": (
                        f"s{args.seed}-{result['scenario']}-"
                        f"ep{episode:04d}-t{start:04d}"
                    ),
                    "eval_seed": int(args.seed),
                    "scenario": result["scenario"],
                    "episode": int(episode),
                    "start_step": int(start),
                    "success": bool(success),
                    "final_state": final_state,
                    "goal_state": goal_state,
                    "final_distance": float(final_distance),
                }
            )
    scenario_success_rates = [result["success_rate"] for result in results]
    pooled_success_rate = 100.0 * float(sum(successes)) / len(successes)
    scenario_balanced_success_rate = float(np.mean(scenario_success_rates))
    output = {
        "schema_version": 1,
        "benchmark": "tworoom_benchmark_step1_v1",
        "run_kind": args.run_kind,
        "status": "passed",
        "stable_worldmodel": {
            "repo": str(stable_repo),
            "commit": stable_commit,
        },
        "policy": policy_identity,
        "dataset_selection": {
            "catalog": str(args.catalog.resolve()),
            "regime": args.regime,
            "scenarios": len(results),
            "num_eval": len(successes),
            "budget_mode": budget_mode,
            "evaluations_per_scenario": {
                path.name: count
                for path, count in zip(paths, scenario_counts)
            },
            "sampling": (
                "fixed_total_scenario_balanced_then_episode_round_robin"
                if budget_mode == "fixed_total_scenario_balanced"
                else "fixed_per_scenario_then_episode_round_robin"
            ),
        },
        "protocol": {
            "eval_seed": args.seed,
            "goal_offset": args.goal_offset,
            "eval_budget": args.eval_budget,
            "horizon": args.horizon,
            "receding_horizon": args.receding_horizon,
            "cem_num_samples": args.cem_num_samples,
            "cem_steps": args.cem_steps,
            "cem_topk": args.cem_topk,
            "normalization_source": (
                str(args.normalizer.resolve())
                if args.normalizer is not None
                else (
                    None
                    if args.original_h5 is None
                    else str(args.original_h5.resolve())
                )
            ),
            "variation_callables": tworoom_eval_callables(),
        },
        "aggregate": {
            "evaluations": len(successes),
            "successes": int(sum(successes)),
            "success_rate": scenario_balanced_success_rate,
            "scenario_balanced_success_rate": scenario_balanced_success_rate,
            "pooled_success_rate": pooled_success_rate,
            "factor_readback_passed": all(
                result["factor_readback"]["passed"] for result in results
            ),
            "mean_final_distance": float(
                np.mean([record["final_distance"] for record in raw_records])
            ),
        },
        "scenarios": results,
        "raw_records": raw_records,
    }
    write_json(args.output.resolve(), output)
    return output


def parse_args():
    parser = argparse.ArgumentParser(
        description="Variation-aware TwoRoom Step-1 dataset evaluation"
    )
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--regime", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--run-kind",
        choices=("protocol_smoke", "qualitative_probe", "benchmark"),
        default="protocol_smoke",
    )
    parser.add_argument("--stablewm-repo", default="../stable-worldmodel")
    parser.add_argument("--stablewm-ref", default=DEFAULT_STABLEWM_REF)
    parser.add_argument("--policy-checkpoint", type=Path)
    parser.add_argument("--legacy-code-root", type=Path)
    parser.add_argument("--original-h5", type=Path)
    parser.add_argument("--normalizer", type=Path)
    parser.add_argument("--max-scenarios", type=int)
    parser.add_argument("--scenario-indices", type=int, nargs="+")
    budget = parser.add_mutually_exclusive_group()
    budget.add_argument(
        "--num-eval",
        type=int,
        help=(
            "Total evaluations for this eval seed, distributed as evenly as "
            "possible over the selected scenarios."
        ),
    )
    budget.add_argument(
        "--evals-per-scenario",
        type=int,
        help="Compatibility mode: use this many evaluations in every scenario.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--goal-offset", type=int, default=10)
    parser.add_argument("--eval-budget", type=int, default=10)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--horizon", type=int, default=2)
    parser.add_argument("--receding-horizon", type=int, default=2)
    parser.add_argument("--action-block", type=int)
    parser.add_argument("--cem-batch-size", type=int, default=1)
    parser.add_argument("--cem-num-samples", type=int, default=8)
    parser.add_argument("--cem-var-scale", type=float, default=1.0)
    parser.add_argument("--cem-steps", type=int, default=2)
    parser.add_argument("--cem-topk", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.num_eval is None and parsed.evals_per_scenario is None:
        parsed.evals_per_scenario = 1
    result = run(parsed)
    print(json.dumps(result["aggregate"], sort_keys=True))
