#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
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
from contextworld.evaluation.planner_mechanism import (
    array_sha256,
    fixed_candidate_bank,
    simulate_tworoom_candidates,
    spearman,
)
from contextworld.evaluation.protocol import (
    frozen_normalizer_process,
    infer_model_protocol,
    load_pretrained_cost_model,
)
from contextworld.paths import artifact_path, resolve_contextworld_path
from contextworld.synthesis.manifest import write_json
from contextworld.synthesis.stablewm import load_stable_worldmodel
from scripts.eval_tworoom_fixed_candidate_mechanism import _images
from scripts.eval_tworoom_icl_planning import (
    PINNED_STABLEWM,
    _balanced_evaluation_schedule,
    _load_query_assets,
    _load_selected_queries,
    image_transform,
)


ACTION_BLOCK = 5


def _goal_directed_candidates(
    bank: np.ndarray,
    *,
    query_state: np.ndarray,
    goal_state: np.ndarray,
    process: dict[str, Any],
) -> np.ndarray:
    """Replace the first candidates with a deterministic goal-directed fan."""

    horizon = int(bank.shape[1])
    raw_steps = horizon * ACTION_BLOCK
    result = np.asarray(bank, dtype=np.float32).copy()
    delta = np.asarray(goal_state) - np.asarray(query_state)
    direction = delta / max(float(np.linalg.norm(delta)), 1e-8)
    rows = []
    for angle in np.linspace(-0.45, 0.45, 7):
        rotation = np.asarray(
            [
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)],
            ],
            dtype=np.float32,
        )
        rotated = rotation @ direction
        for magnitude in np.linspace(0.10, 1.0, 14):
            raw = np.repeat((rotated * magnitude)[None], raw_steps, axis=0)
            normalized = process["action"].transform(raw).astype(np.float32)
            rows.append(normalized.reshape(horizon, ACTION_BLOCK * 2))
    count = min(len(rows), len(result) - 1)
    result[1 : count + 1] = np.stack(rows[:count])
    result[0] = 0.0
    return result


def _condition_costs(
    *,
    model: Any,
    asset: dict[str, Any],
    condition: str,
    candidates: np.ndarray,
    transform: Any,
    device: str,
) -> np.ndarray:
    import torch

    sample_count, horizon, action_width = candidates.shape
    context = asset["contexts"][condition]
    context_pixels = _images(context["pixels"], transform, device)
    query_pixels = _images(
        asset["episode"].query_pixels[None], transform, device
    )
    goal_pixels = _images(
        asset["episode"].goal_pixels[None], transform, device
    )
    context_actions = torch.from_numpy(
        context["normalized_actions"]
    ).to(device)
    action_tensor = torch.from_numpy(candidates[None]).to(
        device=device, dtype=next(model.parameters()).dtype
    )
    model_info = {
        "pixels": torch.cat(
            [
                context_pixels[None, None].expand(
                    1, sample_count, 2, *context_pixels.shape[1:]
                ),
                query_pixels[None, None].expand(
                    1, sample_count, 1, *query_pixels.shape[1:]
                ),
            ],
            dim=2,
        ),
        "goal": goal_pixels[None, None].expand(
            1, sample_count, 1, *goal_pixels.shape[1:]
        ),
        "action": torch.zeros(
            1,
            sample_count,
            1,
            action_width,
            device=device,
            dtype=action_tensor.dtype,
        ),
    }
    prompted_actions = torch.cat(
        [
            context_actions[None, None].expand(
                1, sample_count, 2, action_width
            ),
            action_tensor,
        ],
        dim=2,
    )
    with torch.inference_mode():
        costs = model.get_cost(model_info, prompted_actions)
    predicted = model_info["predicted_emb"][:, :, -horizon:]
    if predicted.shape[2] != horizon:
        raise RuntimeError(
            f"Expected {horizon} predicted states, got {predicted.shape}"
        )
    return costs[0].detach().cpu().float().numpy()


def _condition_summary(
    costs: np.ndarray,
    true_dynamics: dict[str, np.ndarray],
) -> dict[str, Any]:
    selected = int(np.argmin(costs))
    oracle = int(np.argmin(true_dynamics["final_distances"]))
    selected_distance = float(true_dynamics["final_distances"][selected])
    oracle_distance = float(true_dynamics["final_distances"][oracle])
    return {
        "selected_candidate": selected,
        "selected_true_final_distance_px": selected_distance,
        "oracle_candidate": oracle,
        "oracle_true_final_distance_px": oracle_distance,
        "exact_query_dynamics_regret_px": float(
            selected_distance - oracle_distance
        ),
        "selected_true_success": bool(
            true_dynamics["success"][selected]
        ),
        "selected_true_steps_to_success": (
            int(true_dynamics["steps_to_success"][selected])
            if true_dynamics["success"][selected]
            else None
        ),
        "cost_vs_true_distance_spearman": spearman(
            costs, true_dynamics["final_distances"]
        ),
        "cost_sha256": array_sha256(
            np.asarray(costs, dtype=np.float32)
        ),
    }


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
    transform = image_transform(224)

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
    asset_index = {
        asset["episode"].query_id: index
        for index, asset in enumerate(assets)
    }

    records = []
    for record_index, scheduled in enumerate(schedule):
        asset = scheduled["asset"]
        episode = asset["episode"]
        bank = fixed_candidate_bank(
            eval_seed=args.seed,
            evaluation_index=scheduled["evaluation_index"],
            query_index=asset_index[episode.query_id],
            candidates=args.candidates,
            horizon=args.horizon,
            action_block=ACTION_BLOCK,
        )
        bank = _goal_directed_candidates(
            bank,
            query_state=episode.query_state,
            goal_state=episode.goal_state,
            process=process,
        )
        raw_actions = (
            process["action"]
            .inverse_transform(bank.reshape(-1, 2))
            .astype(np.float32)
            .reshape(
                args.candidates,
                args.horizon * ACTION_BLOCK,
                2,
            )
        )
        dynamics = simulate_tworoom_candidates(
            query_state=episode.query_state,
            goal_state=episode.goal_state,
            raw_actions=raw_actions,
            speed=float(episode.speed),
            door_position=float(episode.door_position),
        )
        condition_rows = {}
        for condition in conditions:
            history_speed = float(
                asset["bundle"]["conditions"][condition]["factors"][
                    "agent.speed"
                ]
            )
            costs = _condition_costs(
                model=model,
                asset=asset,
                condition=condition,
                candidates=bank,
                transform=transform,
                device=args.device,
            )
            condition_rows[condition] = {
                "history_speed": history_speed,
                "history_relation": (
                    "same"
                    if np.isclose(
                        history_speed,
                        episode.speed,
                        rtol=0.0,
                        atol=1e-6,
                    )
                    else (
                        "slower"
                        if history_speed < episode.speed
                        else "faster"
                    )
                ),
                **_condition_summary(costs, dynamics),
            }
        records.append(
            {
                "evaluation_id": scheduled["evaluation_id"],
                "evaluation_index": int(
                    scheduled["evaluation_index"]
                ),
                "repeat_index": int(scheduled["repeat_index"]),
                "eval_seed": int(args.seed),
                "query_id": episode.query_id,
                "static_query_id": asset["bundle"]["static_query_id"],
                "query_speed": float(episode.speed),
                "template_id": episode.template_id,
                "candidate_bank_sha256": array_sha256(bank),
                "raw_candidate_actions_sha256": array_sha256(
                    raw_actions
                ),
                "candidate_count": int(args.candidates),
                "horizon_action_blocks": int(args.horizon),
                "oracle": {
                    "best_candidate": int(
                        np.argmin(dynamics["final_distances"])
                    ),
                    "best_final_distance_px": float(
                        np.min(dynamics["final_distances"])
                    ),
                    "successful_candidates": int(
                        np.sum(dynamics["success"])
                    ),
                },
                "conditions": condition_rows,
            }
        )
        print(
            f"[{record_index + 1}/{len(schedule)}] "
            f"query_speed={episode.speed:g} query={episode.query_id}",
            flush=True,
        )

    weight_after = state_dict_sha256(model)
    if weight_before != weight_after:
        raise RuntimeError("Model weights changed during evaluation")
    output = {
        "schema_version": 1,
        "benchmark": "tworoom_history3_speed_fixed_candidate_v2",
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
            "candidates": int(args.candidates),
            "horizon_action_blocks": int(args.horizon),
            "horizon_raw_steps": int(args.horizon * ACTION_BLOCK),
            "same_candidate_bank_across_conditions": True,
            "candidate_scoring_uses_model_cost": True,
            "regret_uses_exact_query_dynamics": True,
        },
        "frozen_weight_audit": {
            "state_dict_sha256_before": weight_before,
            "state_dict_sha256_after": weight_after,
            "passed": weight_before == weight_after,
        },
        "count_audit": {
            "records": len(records),
            "conditions_per_record": 3,
            "condition_records": len(records) * 3,
            "expected_records": int(args.num_eval),
            "passed": len(records) == int(args.num_eval),
        },
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
    parser.add_argument("--candidates", type=int, default=300)
    parser.add_argument("--horizon", type=int, default=10)
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
                "records": len(result["records"]),
            },
            sort_keys=True,
        )
    )
