#!/usr/bin/env python3
"""Evaluate the frozen E4 query schedule without a preceding context prompt."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from contextworld.evaluation.icl_catalog import validate_context_query_catalog
from contextworld.evaluation.icl_model import file_sha256, state_dict_sha256
from contextworld.evaluation.icl_planning import PairedQueryDataset
from contextworld.evaluation.protocol import (
    EvaluationStarts,
    factor_readback_audit,
    frozen_normalizer_process,
    infer_model_protocol,
    load_pretrained_cost_model,
)
from contextworld.evaluation.tworoom import (
    TWOROOM_EVAL_ENV_ID,
    register_tworoom_eval_env,
    tworoom_eval_callables,
)
from contextworld.paths import artifact_path, resolve_contextworld_path
from contextworld.synthesis.manifest import write_json
from contextworld.synthesis.stablewm import load_stable_worldmodel
from scripts.eval_tworoom_icl_planning import (
    _balanced_evaluation_schedule,
    _generator_sha256,
    _load_query_assets,
    _load_selected_queries,
    image_transform,
)


PINNED_STABLEWM = "5864b74980f6ed328fd0045e777b3865962eff43"


def _run_one(
    *,
    args: argparse.Namespace,
    swm: Any,
    model: Any,
    process: dict[str, Any],
    protocol: dict[str, int],
    asset: dict[str, Any],
    evaluation_id: str,
    evaluation_index: int,
    repeat_index: int,
    cem_seed: int,
) -> dict[str, Any]:
    import torch

    episode = asset["episode"]
    dataset = PairedQueryDataset([episode])
    world = swm.World(
        TWOROOM_EVAL_ENV_ID,
        num_envs=1,
        max_episode_steps=2 * args.eval_budget,
        image_shape=(args.img_size, args.img_size),
        render_mode="rgb_array",
    )
    solver = swm.solver.CEMSolver(
        model=model,
        batch_size=1,
        num_samples=args.cem_num_samples,
        var_scale=1.0,
        n_steps=args.cem_steps,
        topk=args.cem_topk,
        device=args.device,
        seed=cem_seed,
    )
    rng_before = _generator_sha256(solver.torch_gen)
    policy = swm.policy.WorldModelPolicy(
        solver=solver,
        config=swm.PlanConfig(
            horizon=args.horizon,
            receding_horizon=args.receding_horizon,
            history_len=protocol["history_size"],
            action_block=protocol["action_block"],
            warm_start=True,
        ),
        process=process,
        transform={
            "pixels": image_transform(args.img_size),
            "goal": image_transform(args.img_size),
        },
    )
    starts = EvaluationStarts(episodes=[0], steps=[0])
    try:
        world.set_policy(policy)
        started = time.time()
        metrics = world.evaluate(
            dataset=dataset,
            episodes_idx=[0],
            start_steps=[0],
            goal_offset=1,
            eval_budget=args.eval_budget,
            callables=tworoom_eval_callables(),
            video=None,
        )
        elapsed = time.time() - started
        factor_audit = factor_readback_audit(dataset, world, starts)
        if not factor_audit["passed"]:
            raise RuntimeError(
                f"Factor readback failed for {episode.query_id}: {factor_audit}"
            )
        final_state = (
            world.envs.envs[0]
            .unwrapped.agent_position.detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        goal = np.asarray(episode.goal_state, dtype=np.float32)
        return {
            "evaluation_id": evaluation_id,
            "evaluation_index": int(evaluation_index),
            "repeat_index": int(repeat_index),
            "eval_seed": int(args.seed),
            "query_id": episode.query_id,
            "source_scenario_id": episode.scenario_id,
            "template_id": episode.template_id,
            "speed": float(episode.speed),
            "door_position": int(episode.door_position),
            "condition": "none",
            "success": bool(np.asarray(metrics["episode_successes"])[0]),
            "final_state": final_state.tolist(),
            "goal_state": goal.tolist(),
            "final_distance": float(np.linalg.norm(final_state - goal)),
            "elapsed_seconds": elapsed,
            "cem_seed": int(cem_seed),
            "cem_rng_state_sha256_before": rng_before,
            "cem_rng_state_sha256_after": _generator_sha256(solver.torch_gen),
            "factor_readback": factor_audit,
        }
    finally:
        world.close()
        if torch.cuda.is_available() and str(args.device).startswith("cuda"):
            torch.cuda.empty_cache()


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[f"speed={record['speed']:g}"].append(record)
        grouped[f"template={record['template_id']}"].append(record)

    def summarize(selected: list[dict[str, Any]]) -> dict[str, Any]:
        successes = sum(bool(record["success"]) for record in selected)
        return {
            "evaluations": len(selected),
            "successes": int(successes),
            "success_rate": float(successes / len(selected)),
            "mean_final_distance": float(
                np.mean([record["final_distance"] for record in selected])
            ),
        }

    return {
        **summarize(records),
        "strata": {
            key: summarize(selected)
            for key, selected in sorted(grouped.items())
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    os.environ.setdefault("MUJOCO_GL", "egl")
    catalog = resolve_contextworld_path(args.catalog, repo_root=REPO_ROOT)
    checkpoint = resolve_contextworld_path(
        args.checkpoint, repo_root=REPO_ROOT
    )
    normalizer = resolve_contextworld_path(
        args.normalizer, repo_root=REPO_ROOT
    )
    output = resolve_contextworld_path(args.output, repo_root=REPO_ROOT)
    if args.horizon * 5 > args.eval_budget:
        raise ValueError("horizon * action_block exceeds eval budget")

    swm, stable_repo, stable_commit = load_stable_worldmodel(
        REPO_ROOT, args.stablewm_repo, args.stablewm_ref
    )
    register_tworoom_eval_env()
    validation = validate_context_query_catalog(
        catalog,
        repo_root=REPO_ROOT,
        replay_simulator=not args.skip_catalog_replay,
        family="speed",
    )
    if not validation["passed"]:
        raise RuntimeError(f"Catalog validation failed: {validation['failures'][:5]}")
    process = frozen_normalizer_process(normalizer)
    model = load_pretrained_cost_model(
        checkpoint,
        swm,
        cache_dir=artifact_path("evaluation/model_cache", repo_root=REPO_ROOT),
    )
    protocol = infer_model_protocol(model, action_dim=2)
    if protocol != {"action_block": 5, "history_size": 3}:
        raise RuntimeError(f"Unexpected model protocol: {protocol}")
    model = model.to(args.device).eval()
    model.requires_grad_(False)
    setattr(model, "history_size", protocol["history_size"])
    setattr(model, "interpolate_pos_encoding", True)
    before = state_dict_sha256(model)

    selected = _load_selected_queries(
        catalog, speeds=args.speeds, templates=args.templates
    )
    assets = [
        _load_query_assets(bundle, process=process) for bundle in selected
    ]
    schedule = _balanced_evaluation_schedule(
        assets, num_eval=args.num_eval, eval_seed=args.seed
    )
    records = []
    for index, scheduled in enumerate(schedule, start=1):
        episode = scheduled["asset"]["episode"]
        print(
            f"[{index}/{len(schedule)}] speed={episode.speed:g} "
            f"template={episode.template_id} "
            f"repeat={scheduled['repeat_index']} condition=none",
            flush=True,
        )
        records.append(
            _run_one(
                args=args,
                swm=swm,
                model=model,
                process=process,
                protocol=protocol,
                asset=scheduled["asset"],
                evaluation_id=scheduled["evaluation_id"],
                evaluation_index=scheduled["evaluation_index"],
                repeat_index=scheduled["repeat_index"],
                cem_seed=scheduled["cem_seed"],
            )
        )
    after = state_dict_sha256(model)
    if before != after:
        raise RuntimeError("Model state changed during frozen no-context E4")
    payload = {
        "schema_version": 1,
        "benchmark": "contextworld_tworoom_history3_e4_nocontext_v1",
        "experiment_id": "E4-no-context",
        "status": "passed",
        "stable_worldmodel": {
            "repo": str(stable_repo),
            "commit": stable_commit,
        },
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": file_sha256(checkpoint),
        },
        "normalizer": {
            "path": str(normalizer),
            "sha256": file_sha256(normalizer),
        },
        "catalog_validation": validation,
        "selection": {
            "family": "speed",
            "speeds": [float(value) for value in args.speeds],
            "templates": list(args.templates),
            "evaluations": len(schedule),
            "schedule": [
                {
                    "evaluation_id": scheduled["evaluation_id"],
                    "evaluation_index": scheduled["evaluation_index"],
                    "repeat_index": scheduled["repeat_index"],
                    "cem_seed": scheduled["cem_seed"],
                    "query_id": scheduled["asset"]["episode"].query_id,
                }
                for scheduled in schedule
            ],
        },
        "protocol": {
            **protocol,
            "eval_seed": args.seed,
            "eval_budget": args.eval_budget,
            "horizon": args.horizon,
            "receding_horizon": args.receding_horizon,
            "cem_samples": args.cem_num_samples,
            "cem_steps": args.cem_steps,
            "cem_topk": args.cem_topk,
            "condition": "none",
            "model_input_layout": "live history only; no fixed prompt",
        },
        "frozen_weight_audit": {
            "requires_grad_false": not any(
                parameter.requires_grad for parameter in model.parameters()
            ),
            "state_dict_sha256_before": before,
            "state_dict_sha256_after": after,
            "passed": before == after,
        },
        "aggregate": _aggregate(records),
        "records": records,
    }
    write_json(output, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path(
            "artifacts/evaluation/icl/"
            "tworoom_icl_v1_validation_context_query_catalog.json"
        ),
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--normalizer",
        type=Path,
        default=Path(
            "artifacts/splits/tworoom_original_train_s3072_normalizer.json"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stablewm-repo", default="../stable-worldmodel")
    parser.add_argument("--stablewm-ref", default=PINNED_STABLEWM)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num-eval", type=int, default=50)
    parser.add_argument(
        "--speeds",
        type=float,
        nargs="+",
        default=[3.1, 3.3, 3.5, 4.1, 5.0, 5.1, 5.9, 7.0],
    )
    parser.add_argument(
        "--templates", nargs="+", default=["s0", "s1", "s2", "s3"]
    )
    parser.add_argument("--eval-budget", type=int, default=50)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--receding-horizon", type=int, default=5)
    parser.add_argument("--cem-num-samples", type=int, default=300)
    parser.add_argument("--cem-steps", type=int, default=30)
    parser.add_argument("--cem-topk", type=int, default=30)
    parser.add_argument("--skip-catalog-replay", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps(result["aggregate"], sort_keys=True))
