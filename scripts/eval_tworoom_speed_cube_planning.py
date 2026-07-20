#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contextworld.evaluation.icl_catalog import (
    validate_context_query_catalog,
)
from contextworld.evaluation.icl_model import file_sha256, state_dict_sha256
from contextworld.evaluation.protocol import (
    frozen_normalizer_process,
    infer_model_protocol,
    load_pretrained_cost_model,
)
from contextworld.evaluation.tworoom import register_tworoom_eval_env
from contextworld.paths import artifact_path, resolve_contextworld_path
from contextworld.synthesis.manifest import write_json
from contextworld.synthesis.stablewm import load_stable_worldmodel
from scripts.eval_tworoom_icl_planning import (
    PINNED_STABLEWM,
    _balanced_evaluation_schedule,
    _load_query_assets,
    _load_selected_queries,
    _run_one,
)


def _aggregate(
    records: list[dict[str, Any]],
    *,
    deadline_budgets: list[int],
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["condition"])].append(record)
    result = {}
    for condition, rows in sorted(grouped.items()):
        successes = np.asarray(
            [bool(row["success"]) for row in rows], dtype=bool
        )
        distances = np.asarray(
            [float(row["final_distance"]) for row in rows],
            dtype=np.float64,
        )
        auc = np.asarray(
            [
                float(row["trajectory"]["normalized_distance_auc"])
                for row in rows
            ],
            dtype=np.float64,
        )
        progress = np.asarray(
            [
                float(row["trajectory"]["progress_per_path_length"])
                for row in rows
            ],
            dtype=np.float64,
        )
        result[condition] = {
            "evaluations": len(rows),
            "successes": int(successes.sum()),
            "success_rate": float(successes.mean()),
            "mean_final_distance_px": float(distances.mean()),
            "mean_normalized_distance_auc": float(auc.mean()),
            "mean_progress_per_path_length": float(progress.mean()),
            "deadline_success_curve": {
                str(budget): {
                    "successes": int(
                        sum(
                            bool(
                                row["trajectory"][
                                    "success_by_budget_raw_steps"
                                ][str(budget)]
                            )
                            for row in rows
                        )
                    ),
                    "success_rate": float(
                        np.mean(
                            [
                                bool(
                                    row["trajectory"][
                                        "success_by_budget_raw_steps"
                                    ][str(budget)]
                                )
                                for row in rows
                            ]
                        )
                    ),
                }
                for budget in deadline_budgets
            },
        }
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    os.environ.setdefault("MUJOCO_GL", "egl")
    args.catalog = resolve_contextworld_path(args.catalog, repo_root=ROOT)
    args.output = resolve_contextworld_path(args.output, repo_root=ROOT)
    args.checkpoint = resolve_contextworld_path(
        args.checkpoint, repo_root=ROOT
    )
    args.normalizer = resolve_contextworld_path(
        args.normalizer, repo_root=ROOT
    )
    if args.horizon * 5 > args.eval_budget:
        raise ValueError("horizon * action_block exceeds eval budget")
    validation = validate_context_query_catalog(
        args.catalog,
        repo_root=ROOT,
        replay_simulator=not args.skip_catalog_replay,
        family="speed",
    )
    if not validation["passed"]:
        raise RuntimeError(
            f"Catalog validation failed: {validation['failures'][:5]}"
        )
    swm, stable_repo, stable_commit = load_stable_worldmodel(
        ROOT, args.stablewm_repo, args.stablewm_ref
    )
    register_tworoom_eval_env()
    process = frozen_normalizer_process(args.normalizer.resolve())
    model = load_pretrained_cost_model(
        args.checkpoint.resolve(),
        swm,
        cache_dir=artifact_path(
            "evaluation/model_cache", repo_root=ROOT
        ),
    )
    protocol = infer_model_protocol(model, action_dim=2)
    if protocol != {"action_block": 5, "history_size": 3}:
        raise RuntimeError(f"Unexpected model protocol: {protocol}")
    model = model.to(args.device).eval()
    model.requires_grad_(False)
    setattr(model, "history_size", 3)
    setattr(model, "interpolate_pos_encoding", True)
    weight_before = state_dict_sha256(model)

    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    templates = sorted(
        {
            str(bundle["template"]["template_id"])
            for bundle in catalog["bundles"]
            if np.isclose(
                float(bundle["query_factors"]["agent.speed"]),
                args.query_speed,
                rtol=0.0,
                atol=1e-6,
            )
        }
    )
    selected = _load_selected_queries(
        args.catalog,
        speeds=[args.query_speed],
        templates=templates,
    )
    assets = [
        _load_query_assets(bundle, process=process) for bundle in selected
    ]
    schedule = _balanced_evaluation_schedule(
        assets, num_eval=args.num_eval, eval_seed=args.seed
    )
    conditions = list(selected[0]["conditions"])
    if conditions != ["history_low", "history_mid", "history_high"]:
        raise RuntimeError(f"Unexpected speed cube conditions: {conditions}")

    records = []
    deadline_budgets = sorted(
        {
            int(value)
            for value in (
                args.deadline_budgets
                if args.deadline_budgets is not None
                else [args.eval_budget]
            )
        }
    )
    if (
        not deadline_budgets
        or deadline_budgets[-1] != int(args.eval_budget)
        or deadline_budgets[0] <= 0
    ):
        raise ValueError(
            "Deadline budgets must be positive and end at eval_budget"
        )
    total = len(schedule) * len(conditions)
    run_index = 0
    for scheduled in schedule:
        asset = scheduled["asset"]
        for condition in conditions:
            run_index += 1
            print(
                f"[{run_index}/{total}] query_speed="
                f"{asset['episode'].speed:g} "
                f"query={asset['episode'].query_id} "
                f"condition={condition}",
                flush=True,
            )
            record = _run_one(
                args=args,
                swm=swm,
                model=model,
                process=process,
                protocol=protocol,
                asset=asset,
                condition=condition,
                evaluation_id=scheduled["evaluation_id"],
                evaluation_index=scheduled["evaluation_index"],
                repeat_index=scheduled["repeat_index"],
                cem_seed=scheduled["cem_seed"],
            )
            history_speed = float(
                asset["bundle"]["conditions"][condition]["factors"][
                    "agent.speed"
                ]
            )
            record["query_speed"] = float(asset["episode"].speed)
            record["history_speed"] = history_speed
            record["history_relation"] = (
                "same"
                if np.isclose(
                    history_speed,
                    asset["episode"].speed,
                    rtol=0.0,
                    atol=1e-6,
                )
                else (
                    "slower"
                    if history_speed < asset["episode"].speed
                    else "faster"
                )
            )
            record["static_query_id"] = asset["bundle"][
                "static_query_id"
            ]
            steps_to_success = record["trajectory"]["steps_to_success"]
            record["trajectory"]["success_by_budget_raw_steps"] = {
                str(budget): bool(
                    steps_to_success is not None
                    and int(steps_to_success) <= budget
                )
                for budget in deadline_budgets
            }
            records.append(record)

    weight_after = state_dict_sha256(model)
    if weight_before != weight_after:
        raise RuntimeError("Model weights changed during evaluation")
    output = {
        "schema_version": 1,
        "benchmark": "tworoom_history3_speed_closed_loop_v2",
        "status": "passed",
        "track": catalog["track"],
        "query_speed": float(args.query_speed),
        "eval_seed": int(args.seed),
        "model": {
            "checkpoint": str(args.checkpoint.resolve()),
            "sha256": file_sha256(args.checkpoint.resolve()),
        },
        "normalizer": {
            "path": str(args.normalizer.resolve()),
            "sha256": file_sha256(args.normalizer.resolve()),
        },
        "stable_worldmodel": {
            "repo": str(stable_repo),
            "commit": stable_commit,
        },
        "catalog_validation": validation,
        "protocol": {
            **protocol,
            "evaluations_per_condition": len(schedule),
            "conditions": conditions,
            "eval_budget_raw_steps": int(args.eval_budget),
            "deadline_budgets_raw_steps": deadline_budgets,
            "horizon_action_blocks": int(args.horizon),
            "receding_horizon_action_blocks": int(
                args.receding_horizon
            ),
            "cem_samples": int(args.cem_num_samples),
            "cem_iterations": int(args.cem_steps),
            "cem_topk": int(args.cem_topk),
            "same_query_and_cem_seed_across_conditions": True,
        },
        "frozen_weight_audit": {
            "state_dict_sha256_before": weight_before,
            "state_dict_sha256_after": weight_after,
            "passed": weight_before == weight_after,
        },
        "count_audit": {
            "records": len(records),
            "conditions_per_evaluation": len(conditions),
            "evaluations_per_condition": len(schedule),
            "expected_records": int(args.num_eval) * len(conditions),
            "passed": (
                len(records) == int(args.num_eval) * len(conditions)
            ),
        },
        "aggregate": _aggregate(
            records, deadline_budgets=deadline_budgets
        ),
        "records": records,
    }
    write_json(args.output.resolve(), output)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--normalizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--query-speed", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num-eval", type=int, default=50)
    parser.add_argument(
        "--run-kind",
        choices=("confirmation", "qualitative_probe"),
        default="confirmation",
    )
    parser.add_argument("--eval-budget", type=int, default=100)
    parser.add_argument("--deadline-budgets", type=int, nargs="+")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--receding-horizon", type=int, default=5)
    parser.add_argument("--cem-batch-size", type=int, default=1)
    parser.add_argument("--cem-num-samples", type=int, default=300)
    parser.add_argument("--cem-var-scale", type=float, default=1.0)
    parser.add_argument("--cem-steps", type=int, default=30)
    parser.add_argument("--cem-topk", type=int, default=30)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--stablewm-repo", default="../stable-worldmodel")
    parser.add_argument("--stablewm-ref", default=PINNED_STABLEWM)
    parser.add_argument("--skip-catalog-replay", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(
        json.dumps(
            {
                "status": result["status"],
                "track": result["track"],
                "query_speed": result["query_speed"],
                "condition_records": len(result["records"]),
            },
            sort_keys=True,
        )
    )
