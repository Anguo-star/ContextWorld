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
    load_door_planning_cell,
    simulate_door_candidates,
    summarize_fixed_candidate_selection,
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
from contextworld.paths import artifact_path, resolve_contextworld_path
from contextworld.synthesis.manifest import write_json
from contextworld.synthesis.stablewm import load_stable_worldmodel
from scripts.eval_tworoom_fixed_candidate_mechanism import _images
from scripts.eval_tworoom_icl_planning import PINNED_STABLEWM, image_transform


DEFAULT_CONFIG = (
    ROOT / "configs/benchmark/tworoom_door_visual_generalization_v1.yaml"
)


def _frozen_protocol(config: dict[str, Any]) -> dict[str, int]:
    row = config["fixed_candidate_evaluation"]
    return {
        "candidates": int(row["candidates_per_query"]),
        "horizon": int(row["horizon_action_blocks"]),
        "per_seed": int(row["evaluations_per_door_per_seed"]),
    }


def _assert_args_match_config(
    args: argparse.Namespace, protocol: dict[str, int]
) -> None:
    for argument, key in (("candidates", "candidates"), ("horizon", "horizon")):
        if int(getattr(args, argument)) != int(protocol[key]):
            raise ValueError(
                f"--{argument.replace('_', '-')}={getattr(args, argument)} "
                f"does not match frozen config value {protocol[key]}"
            )
    if args.run_kind == "confirmation" and int(args.num_eval) != int(
        protocol["per_seed"]
    ):
        raise ValueError(
            "A confirmation run must use all 50 queries for one door/seed cell"
        )


def _candidate_costs(
    *,
    model: Any,
    asset: dict[str, Any],
    normalized_candidates: np.ndarray,
    normalized_history_actions: np.ndarray,
    transform: Any,
    device: str,
) -> np.ndarray:
    import torch

    count, horizon, action_width = normalized_candidates.shape
    history_pixels = _images(asset["history_pixels"], transform, device)
    query_pixels = _images(
        asset["episode"].query_pixels[None], transform, device
    )
    goal_pixels = _images(
        asset["episode"].goal_pixels[None], transform, device
    )
    dtype = next(model.parameters()).dtype
    candidate_tensor = torch.from_numpy(normalized_candidates[None]).to(
        device=device, dtype=dtype
    )
    history_action_tensor = torch.from_numpy(
        normalized_history_actions
    ).to(device=device, dtype=dtype)
    model_info = {
        "pixels": torch.cat(
            [
                history_pixels[None, None].expand(
                    1, count, 2, *history_pixels.shape[1:]
                ),
                query_pixels[None, None].expand(
                    1, count, 1, *query_pixels.shape[1:]
                ),
            ],
            dim=2,
        ),
        "goal": goal_pixels[None, None].expand(
            1, count, 1, *goal_pixels.shape[1:]
        ),
        "action": torch.zeros(
            1,
            count,
            1,
            action_width,
            device=device,
            dtype=dtype,
        ),
    }
    actions = torch.cat(
        [
            history_action_tensor[None, None].expand(
                1, count, 2, action_width
            ),
            candidate_tensor,
        ],
        dim=2,
    )
    with torch.inference_mode():
        costs = model.get_cost(model_info, actions)
    predicted = model_info.get("predicted_emb")
    if predicted is None or predicted.shape[2] < horizon:
        raise RuntimeError(
            f"Model did not expose {horizon} candidate rollout states"
        )
    values = costs[0].detach().cpu().float().numpy()
    if values.shape != (count,) or not np.isfinite(values).all():
        raise RuntimeError(f"Unexpected model costs: {values.shape}")
    return values


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
        candidates=args.candidates,
        horizon=args.horizon,
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
    transform = image_transform(args.img_size)

    records: list[dict[str, Any]] = []
    for index, asset in enumerate(assets):
        raw_candidates = asset["fixed_candidate_raw_actions"]
        normalized_candidates = (
            process["action"]
            .transform(raw_candidates.reshape(-1, 2))
            .astype(np.float32)
            .reshape(args.candidates, args.horizon, ACTION_BLOCK * 2)
        )
        normalized_history = (
            process["action"]
            .transform(asset["history_raw_actions"].reshape(-1, 2))
            .astype(np.float32)
            .reshape(2, ACTION_BLOCK * 2)
        )
        costs = _candidate_costs(
            model=model,
            asset=asset,
            normalized_candidates=normalized_candidates,
            normalized_history_actions=normalized_history,
            transform=transform,
            device=args.device,
        )
        episode = asset["episode"]
        dynamics = simulate_door_candidates(
            query_state=episode.query_state,
            goal_state=episode.goal_state,
            raw_actions=raw_candidates,
            speed=episode.speed,
            door_position=episode.door_position,
        )
        result = summarize_fixed_candidate_selection(costs, dynamics)
        records.append(
            {
                "evaluation_id": (
                    f"s{args.seed}-e{asset['evaluation_index']:03d}-"
                    f"{episode.query_id}"
                ),
                "evaluation_index": int(asset["evaluation_index"]),
                "eval_seed": int(args.seed),
                "query_id": episode.query_id,
                "track": args.track,
                "task": asset["task"],
                "door_position": int(episode.door_position),
                "agent_speed": float(episode.speed),
                "direction": asset["direction"],
                "door_relative_vertical_offset_px": int(
                    asset["door_relative_vertical_offset_px"]
                ),
                "candidate_bank_raw_sha256": asset["array_hashes"][
                    "fixed_candidate_raw_actions"
                ],
                "candidate_bank_normalized_sha256": array_sha256(
                    normalized_candidates
                ),
                "history_pixels_sha256": asset["array_hashes"][
                    "history_pixels"
                ],
                "history_actions_sha256": asset["array_hashes"][
                    "history_actions"
                ],
                **result,
                "success": bool(result["selected_true_success"]),
                "final_distance_px": float(
                    result["selected_true_final_distance_px"]
                ),
                "steps_to_success": result[
                    "selected_true_steps_to_success"
                ],
                "doorway_crossing": bool(
                    result["selected_true_doorway_crossing"]
                ),
            }
        )
        print(
            f"[{index + 1}/{len(assets)}] door={episode.door_position} "
            f"query={episode.query_id}",
            flush=True,
        )

    weight_after = state_dict_sha256(model)
    if weight_before != weight_after:
        raise RuntimeError("Model weights changed during fixed-candidate scoring")
    aggregate = aggregate_door_records(records)
    aggregate.update(
        {
            "mean_endpoint_regret_px": float(
                np.mean(
                    [
                        row["exact_environment_endpoint_regret_px"]
                        for row in records
                    ]
                )
            ),
            "mean_cost_distance_spearman": float(
                np.mean(
                    [
                        row[
                            "predicted_cost_vs_true_endpoint_distance_spearman"
                        ]
                        for row in records
                    ]
                )
            ),
        }
    )
    output = {
        "schema_version": 1,
        "benchmark": "tworoom_door_fixed_candidates_v1",
        "status": "passed",
        "evidence_role": "planning_action_ranking_not_latent_accuracy",
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
            "candidates_per_query": int(args.candidates),
            "horizon_action_blocks": int(args.horizon),
            "horizon_raw_steps": int(args.horizon * ACTION_BLOCK),
            "same_frozen_candidate_bank_across_models": True,
            "agent_speed": 5.0,
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
        "aggregate": aggregate,
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
    parser.add_argument("--candidates", type=int, default=300)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--img-size", type=int, default=224)
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
