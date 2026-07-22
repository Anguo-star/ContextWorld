#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contextworld.evaluation.door_planning import (
    ACTION_BLOCK,
    aggregate_door_records,
    doorway_crossing,
    load_door_planning_cell,
    validate_door_planning_catalog_header,
)
from contextworld.evaluation.icl_model import file_sha256, state_dict_sha256
from contextworld.evaluation.sealed_test_gate import (
    canonical_door_planning_catalog,
    require_canonical_split_path,
    require_sealed_test_gate,
)
from contextworld.evaluation.planner_mechanism import array_sha256
from contextworld.evaluation.protocol import (
    frozen_normalizer_process,
    infer_model_protocol,
    load_pretrained_cost_model,
)
from contextworld.evaluation.tworoom import register_tworoom_eval_env
from contextworld.paths import artifact_path, resolve_contextworld_path
from contextworld.synthesis.manifest import write_json
from contextworld.synthesis.stablewm import load_stable_worldmodel
from scripts.eval_tworoom_icl_planning import PINNED_STABLEWM, _run_one


DEFAULT_CONFIG = (
    ROOT / "configs/benchmark/tworoom_door_visual_generalization_v1.yaml"
)


def _frozen_protocol(config: dict[str, Any]) -> dict[str, int]:
    row = config["closed_loop_planning"]
    return {
        "per_seed": int(row["evaluations_per_door_per_seed"]),
        "eval_budget": int(row["execution_budget_raw_steps"]),
        "horizon": int(row["horizon_action_blocks"]),
        "receding_horizon": int(row["receding_horizon_action_blocks"]),
        "samples": int(row["candidates"]),
        "iterations": int(row["iterations"]),
        "topk": int(row["topk"]),
    }


def _assert_args_match_config(
    args: argparse.Namespace, protocol: dict[str, int]
) -> None:
    pairs = (
        ("eval_budget", "eval_budget"),
        ("horizon", "horizon"),
        ("receding_horizon", "receding_horizon"),
        ("cem_num_samples", "samples"),
        ("cem_steps", "iterations"),
        ("cem_topk", "topk"),
    )
    for argument, key in pairs:
        actual = int(getattr(args, argument))
        expected = int(protocol[key])
        if actual != expected:
            raise ValueError(
                f"--{argument.replace('_', '-')}={actual} does not match "
                f"frozen config value {expected}"
            )
    if args.horizon * ACTION_BLOCK > args.eval_budget:
        raise ValueError("rollout horizon exceeds the execution budget")
    if args.run_kind == "confirmation" and int(args.num_eval) != int(
        protocol["per_seed"]
    ):
        raise ValueError(
            "A confirmation run must use all 50 queries for one door/seed cell"
        )


def _assert_rolling_history_protocol(config: dict[str, Any]) -> None:
    row = config["closed_loop_planning"].get("history3_replan_context", {})
    expected = {
        "first_replan": "frozen_contiguous_catalog_history",
        "later_replans": "two_most_recent_complete_live_action_blocks",
        "stale_initial_history_reuse": "forbidden",
        (
            "hash_every_replan_pixels_raw_actions_normalized_actions_"
            "and_current_state"
        ): True,
    }
    if row != expected:
        raise ValueError(
            "Door CEM config does not match the frozen rolling History-3 "
            f"protocol: expected {expected}, got {row}"
        )


def run(args: argparse.Namespace) -> dict[str, Any]:
    os.environ.setdefault("MUJOCO_GL", "egl")
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    track_split = str(
        config["evaluation_data"]["tracks"].get(args.track, {}).get("split", "")
    )
    gate_split = (
        "sealed_test"
        if "sealed_test" in (args.split, track_split)
        else "validation"
    )
    gate_audit = require_sealed_test_gate(
        split=gate_split,
        config_path=config_path,
        config=config,
        manifest_path=getattr(args, "sealed_test_gate", None),
        repo_root=ROOT,
    )
    if track_split != args.split:
        raise RuntimeError("Configured track/evaluator split mismatch")
    args.catalog = require_canonical_split_path(
        args.catalog,
        canonical=canonical_door_planning_catalog(
            config, split=args.split, repo_root=ROOT
        ),
        split=args.split,
        label="Door planning catalog input",
    )
    frozen = _frozen_protocol(config)
    _assert_args_match_config(args, frozen)
    _assert_rolling_history_protocol(config)
    args.checkpoint = resolve_contextworld_path(args.checkpoint, repo_root=ROOT)
    args.normalizer = resolve_contextworld_path(args.normalizer, repo_root=ROOT)
    args.output = resolve_contextworld_path(args.output, repo_root=ROOT)

    catalog_header = json.loads(args.catalog.read_text(encoding="utf-8"))
    catalog_header_audit = validate_door_planning_catalog_header(
        catalog_header,
        config=config,
        config_sha256=file_sha256(config_path),
        split=args.split,
        run_kind=args.run_kind,
    )

    cell = load_door_planning_cell(
        args.catalog,
        repo_root=ROOT,
        track=args.track,
        door_position=args.door_position,
        eval_seed=args.seed,
        candidates=frozen["samples"],
        horizon=frozen["horizon"],
        expected_queries=(
            frozen["per_seed"] if args.run_kind == "confirmation" else None
        ),
    )
    if str(cell.catalog.get("split_role")) != args.split:
        raise RuntimeError("Planning catalog/evaluator split mismatch")
    if args.num_eval > len(cell.assets):
        raise ValueError(
            f"Requested {args.num_eval} queries, cell has {len(cell.assets)}"
        )
    assets = list(cell.assets[: args.num_eval])

    swm, stable_repo, stable_commit = load_stable_worldmodel(
        ROOT, args.stablewm_repo, args.stablewm_ref
    )
    if str(catalog_header.get("stable_worldmodel", {}).get("commit")) != str(
        stable_commit
    ):
        raise RuntimeError("Planning catalog/StableWorldModel commit mismatch")
    register_tworoom_eval_env()
    process = frozen_normalizer_process(args.normalizer.resolve())
    model = load_pretrained_cost_model(
        args.checkpoint.resolve(),
        swm,
        cache_dir=artifact_path("evaluation/model_cache", repo_root=ROOT),
    )
    model_protocol = infer_model_protocol(model, action_dim=2)
    if model_protocol != {"action_block": ACTION_BLOCK, "history_size": 3}:
        raise RuntimeError(f"Unexpected model protocol: {model_protocol}")
    model = model.to(args.device).eval()
    model.requires_grad_(False)
    setattr(model, "history_size", 3)
    setattr(model, "interpolate_pos_encoding", True)
    weight_before = state_dict_sha256(model)

    records: list[dict[str, Any]] = []
    for index, asset in enumerate(assets):
        normalized_history = (
            process["action"]
            .transform(asset["history_raw_actions"].reshape(-1, 2))
            .astype(np.float32)
            .reshape(2, ACTION_BLOCK * 2)
        )
        run_asset = dict(asset)
        run_asset["contexts"] = {
            "natural_history3": {
                "pixels": asset["history_pixels"],
                "raw_actions": asset["history_raw_actions"],
                "normalized_actions": normalized_history,
                "pixels_sha256": asset["array_hashes"]["history_pixels"],
                "raw_actions_sha256": asset["array_hashes"][
                    "history_actions"
                ],
                "normalized_actions_sha256": (
                    # Normalized actions are checkpoint-independent because
                    # every model uses the frozen original-data normalizer.
                    array_sha256(normalized_history)
                ),
                "source_factors": {
                    "agent.speed": float(asset["episode"].speed),
                    "door.position": int(asset["episode"].door_position),
                },
            }
        }
        print(
            f"[{index + 1}/{len(assets)}] door="
            f"{asset['episode'].door_position} query="
            f"{asset['episode'].query_id}",
            flush=True,
        )
        result = _run_one(
            args=args,
            swm=swm,
            model=model,
            process=process,
            protocol=model_protocol,
            asset=run_asset,
            condition="natural_history3",
            evaluation_id=(
                f"s{args.seed}-e{asset['evaluation_index']:03d}-"
                f"{asset['episode'].query_id}"
            ),
            evaluation_index=asset["evaluation_index"],
            repeat_index=0,
            cem_seed=asset["cem_seed"],
            rolling_context=True,
        )
        crossing = doorway_crossing(
            np.asarray(result["trajectory"]["states"], dtype=np.float32),
            door_position=asset["episode"].door_position,
            goal_state=asset["episode"].goal_state,
        )
        result.update(
            {
                "track": args.track,
                "task": asset["task"],
                "direction": asset["direction"],
                "door_relative_vertical_offset_px": int(
                    asset["door_relative_vertical_offset_px"]
                ),
                "history_pixels_sha256": asset["array_hashes"][
                    "history_pixels"
                ],
                "history_actions_sha256": asset["array_hashes"][
                    "history_actions"
                ],
                "final_distance_px": float(result["final_distance"]),
                "steps_to_success": result["trajectory"]["steps_to_success"],
                "doorway_crossing": bool(crossing["crossed"]),
                "first_doorway_crossing_raw_step": crossing[
                    "first_crossing_raw_step"
                ],
            }
        )
        records.append(result)

    weight_after = state_dict_sha256(model)
    if weight_before != weight_after:
        raise RuntimeError("Model weights changed during door CEM planning")
    rolling_audits = [row["rolling_history3_audit"] for row in records]
    if not all(audit is not None and audit["passed"] for audit in rolling_audits):
        raise RuntimeError("One or more rolling History-3 audits failed")
    output = {
        "schema_version": 1,
        "benchmark": "tworoom_door_closed_loop_planning_v1",
        "status": "passed",
        "evidence_role": "closed_loop_planning_not_latent_accuracy",
        "run_kind": args.run_kind,
        "evaluation_split": args.split,
        "sealed_test_gate": gate_audit,
        "track": args.track,
        "door_position": int(args.door_position),
        "eval_seed": int(args.seed),
        "config": {
            "path": str(config_path),
            "sha256": file_sha256(config_path),
        },
        "catalog": {
            "path": str(cell.catalog_path),
            "sha256": cell.catalog_sha256,
            "header_audit": catalog_header_audit,
            "cell_audit": cell.audit,
        },
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
        "protocol": {
            **model_protocol,
            "queries": len(records),
            "eval_budget_raw_steps": int(args.eval_budget),
            "horizon_action_blocks": int(args.horizon),
            "receding_horizon_action_blocks": int(args.receding_horizon),
            "cem_samples": int(args.cem_num_samples),
            "cem_iterations": int(args.cem_steps),
            "cem_topk": int(args.cem_topk),
            "same_query_and_initial_cem_seed_across_models": True,
            "rolling_causally_aligned_natural_history3": True,
            "initial_catalog_history_used_only_for_first_replan": True,
            "agent_speed": 5.0,
        },
        "rolling_history3_audit": {
            "passed": True,
            "records": len(rolling_audits),
            "total_replans": int(
                sum(audit["replans"] for audit in rolling_audits)
            ),
            "rolling_replans": int(
                sum(audit["rolling_replans"] for audit in rolling_audits)
            ),
            "all_replans_causally_aligned": True,
        },
        "frozen_weight_audit": {
            "state_dict_sha256_before": weight_before,
            "state_dict_sha256_after": weight_after,
            "passed": weight_before == weight_after,
        },
        "count_audit": {
            "records": len(records),
            "expected_records": int(args.num_eval),
            "passed": len(records) == int(args.num_eval),
        },
        "aggregate": aggregate_door_records(records),
        "records": records,
    }
    write_json(args.output.resolve(), output)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--split", choices=("validation", "sealed_test"), default="validation"
    )
    parser.add_argument("--sealed-test-gate", type=Path)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--normalizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--track", required=True)
    parser.add_argument("--door-position", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num-eval", type=int, default=50)
    parser.add_argument(
        "--run-kind", choices=("confirmation", "smoke"), default="confirmation"
    )
    parser.add_argument("--eval-budget", type=int, default=100)
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
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(
        json.dumps(
            {
                "status": result["status"],
                "door_position": result["door_position"],
                "eval_seed": result["eval_seed"],
                "records": result["count_audit"]["records"],
            },
            sort_keys=True,
        )
    )
