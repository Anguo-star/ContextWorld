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

from contextworld.evaluation.icl_model import file_sha256, state_dict_sha256
from contextworld.evaluation.icl_planning import (
    CONTEXT_ACTIONS_KEY,
    CONTEXT_PIXELS_KEY,
)
from contextworld.evaluation.planner_mechanism import (
    array_sha256,
    fixed_candidate_bank,
    simulate_tworoom_candidates,
    spearman,
    summarize_costs,
    topk_overlap,
)
from contextworld.evaluation.protocol import (
    frozen_normalizer_process,
    infer_model_protocol,
    load_pretrained_cost_model,
)
from contextworld.paths import artifact_path, resolve_contextworld_path
from contextworld.synthesis.manifest import write_json
from contextworld.synthesis.stablewm import load_stable_worldmodel

from scripts.eval_tworoom_icl_planning import (
    PINNED_STABLEWM,
    _balanced_evaluation_schedule,
    _fixed_variation_values,
    _load_query_assets,
    _load_selected_queries,
    image_transform,
)


def _images(value: np.ndarray, transform: Any, device: str):
    import torch
    from torchvision import tv_tensors

    array = np.asarray(value)
    return torch.stack(
        [
            transform(
                tv_tensors.Image(
                    torch.from_numpy(frame.transpose(2, 0, 1).copy())
                )
            )
            for frame in array
        ]
    ).to(device)


def _condition_costs(
    *,
    model: Any,
    asset: dict[str, Any],
    condition: str,
    candidates: np.ndarray,
    transform: Any,
    device: str,
) -> tuple[np.ndarray, np.ndarray, Any]:
    import torch

    sample_count = candidates.shape[0]
    context = asset["contexts"][condition]
    context_pixels = _images(context["pixels"], transform, device)
    query_pixels = _images(asset["episode"].query_pixels[None], transform, device)
    goal_pixels = _images(asset["episode"].goal_pixels[None], transform, device)
    context_actions = torch.from_numpy(context["normalized_actions"]).to(device)
    action_tensor = torch.from_numpy(candidates[None]).to(
        device=device, dtype=next(model.parameters()).dtype
    )
    model_info = {
        "pixels": query_pixels[None, None].expand(
            1, sample_count, 1, *query_pixels.shape[1:]
        ),
        "goal": goal_pixels[None, None].expand(
            1, sample_count, 1, *goal_pixels.shape[1:]
        ),
        "action": torch.zeros(
            1,
            sample_count,
            1,
            candidates.shape[-1],
            device=device,
            dtype=action_tensor.dtype,
        ),
    }
    model_info["pixels"] = torch.cat(
        [
            context_pixels[None, None].expand(
                1, sample_count, 2, *context_pixels.shape[1:]
            ),
            model_info["pixels"],
        ],
        dim=2,
    )
    prompted_actions = torch.cat(
        [
            context_actions[None, None].expand(1, sample_count, 2, 10),
            action_tensor,
        ],
        dim=2,
    )
    with torch.inference_mode():
        costs = model.get_cost(model_info, prompted_actions)
    predicted = model_info["predicted_emb"][:, :, -5:]
    goal = model_info["goal_emb"][:, None, -1:, :]
    step_costs = ((predicted - goal) ** 2).sum(dim=-1)[0]
    return (
        costs[0].detach().cpu().float().numpy(),
        step_costs.detach().cpu().float().numpy(),
        predicted[0].detach(),
    )


def _add_goal_directed_candidates(
    bank: np.ndarray, episode: Any, process: dict[str, Any]
) -> np.ndarray:
    result = bank.copy()
    delta = np.asarray(episode.goal_state) - np.asarray(episode.query_state)
    direction = delta / max(float(np.linalg.norm(delta)), 1e-8)
    rows = []
    for angle in np.linspace(-0.35, 0.35, 5):
        rotation = np.asarray(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
            dtype=np.float32,
        )
        rotated = rotation @ direction
        for magnitude in np.linspace(0.15, 1.0, 12):
            raw = np.repeat((rotated * magnitude)[None], 25, axis=0)
            normalized = process["action"].transform(raw).astype(np.float32)
            rows.append(normalized.reshape(5, 10))
    result[1:61] = np.stack(rows)
    result[0] = 0.0
    return result


def _true_probe_embeddings(
    *,
    model: Any,
    episode: Any,
    raw_actions: np.ndarray,
    transform: Any,
    device: str,
):
    import torch
    from stable_worldmodel.envs.two_room.env import TwoRoomEnv

    env = TwoRoomEnv(render_mode="rgb_array")
    frames = []
    try:
        env.reset(
            seed=episode.simulator_seed,
            options={
                "variation": (),
                "variation_values": _fixed_variation_values(
                    episode.speed, episode.door_position
                ),
                "state": episode.query_state,
                "target_state": episode.goal_state,
            },
        )
        for index, action in enumerate(raw_actions, start=1):
            env.step(action)
            if index in (5, 10, 15, 20, 25):
                frames.append(np.asarray(env.render(), dtype=np.uint8))
    finally:
        env.close()
    pixels = _images(np.stack(frames), transform, device)[None]
    with torch.inference_mode():
        return model.encode({"pixels": pixels})["emb"][0].detach()


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    os.environ.setdefault("MUJOCO_GL", "egl")
    swm, stable_repo, stable_commit = load_stable_worldmodel(
        ROOT, args.stablewm_repo, args.stablewm_ref
    )
    process = frozen_normalizer_process(args.normalizer.resolve())
    model = load_pretrained_cost_model(
        args.checkpoint.resolve(),
        swm,
        cache_dir=artifact_path("evaluation/model_cache", repo_root=ROOT),
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

    slow_bundles = _load_selected_queries(
        args.slow_catalog, speeds=[5.0, 5.1], templates=args.templates
    )
    fast_bundles = _load_selected_queries(
        args.fast_catalog, speeds=[5.0, 5.1], templates=args.templates
    )
    slow_assets = [_load_query_assets(row, process=process) for row in slow_bundles]
    fast_assets = [_load_query_assets(row, process=process) for row in fast_bundles]
    if [x["episode"].query_id for x in slow_assets] != [
        x["episode"].query_id for x in fast_assets
    ]:
        raise RuntimeError("Directional catalogs do not share queries")
    schedule = _balanced_evaluation_schedule(
        slow_assets, num_eval=50, eval_seed=args.seed
    )
    fast_by_query = {x["episode"].query_id: x for x in fast_assets}
    records = []
    for index, scheduled in enumerate(schedule):
        slow_asset = scheduled["asset"]
        episode = slow_asset["episode"]
        fast_asset = fast_by_query[episode.query_id]
        query_index = next(
            i for i, x in enumerate(slow_assets) if x["episode"].query_id == episode.query_id
        )
        bank = fixed_candidate_bank(
            eval_seed=args.seed,
            evaluation_index=scheduled["evaluation_index"],
            query_index=query_index,
        )
        bank = _add_goal_directed_candidates(bank, episode, process)
        raw = process["action"].inverse_transform(
            bank.reshape(-1, 2)
        ).astype(np.float32).reshape(300, 25, 2)
        dynamics = {
            "slow": simulate_tworoom_candidates(
                query_state=episode.query_state, goal_state=episode.goal_state,
                raw_actions=raw, speed=3.1, door_position=episode.door_position,
            ),
            "correct": simulate_tworoom_candidates(
                query_state=episode.query_state, goal_state=episode.goal_state,
                raw_actions=raw, speed=episode.speed, door_position=episode.door_position,
            ),
            "fast": simulate_tworoom_candidates(
                query_state=episode.query_state, goal_state=episode.goal_state,
                raw_actions=raw, speed=7.0, door_position=episode.door_position,
            ),
        }
        condition_assets = {
            "slow": (slow_asset, "wrong"),
            "correct": (slow_asset, "correct"),
            "fast": (fast_asset, "wrong"),
        }
        condition_rows = {}
        cost_vectors = {}
        predicted_rows = {}
        for label, (asset, condition) in condition_assets.items():
            costs, step_costs, predicted = _condition_costs(
                model=model, asset=asset, condition=condition, candidates=bank,
                transform=transform, device=args.device,
            )
            condition_rows[label] = summarize_costs(
                costs, step_costs, dynamics["correct"]
            )
            cost_vectors[label] = costs
            predicted_rows[label] = predicted
        oracle = {}
        true = dynamics["correct"]
        for label, result in dynamics.items():
            selected = int(np.argmin(result["final_distances"]))
            oracle[label] = {
                "selected_candidate": selected,
                "selected_true_final_distance": float(true["final_distances"][selected]),
                "selected_true_success": bool(true["success"][selected]),
                "selected_true_steps_to_success": (
                    int(true["steps_to_success"][selected])
                    if true["success"][selected] else None
                ),
            }
        probe_index = oracle["correct"]["selected_candidate"]
        true_emb = _true_probe_embeddings(
            model=model, episode=episode, raw_actions=raw[probe_index],
            transform=transform, device=args.device,
        )
        for label in condition_rows:
            pred = predicted_rows[label][probe_index]
            condition_rows[label]["prediction_mse_to_true_by_block"] = (
                ((pred - true_emb) ** 2).mean(dim=-1).detach().cpu().tolist()
            )
        records.append(
            {
                "evaluation_id": scheduled["evaluation_id"],
                "evaluation_index": scheduled["evaluation_index"],
                "query_id": episode.query_id,
                "query_speed": episode.speed,
                "candidate_bank_sha256": array_sha256(bank),
                "conditions": condition_rows,
                "oracle": oracle,
                "context_cost_rank_spearman": {
                    "slow_correct": spearman(cost_vectors["slow"], cost_vectors["correct"]),
                    "correct_fast": spearman(cost_vectors["correct"], cost_vectors["fast"]),
                    "slow_fast": spearman(cost_vectors["slow"], cost_vectors["fast"]),
                },
                "topk_overlap": {
                    "slow_correct": len(
                        set(condition_rows["slow"]["topk_indices"])
                        & set(condition_rows["correct"]["topk_indices"])
                    ) / 30.0,
                    "correct_fast": len(
                        set(condition_rows["correct"]["topk_indices"])
                        & set(condition_rows["fast"]["topk_indices"])
                    ) / 30.0,
                },
            }
        )
        print(f"[{index + 1}/50] {episode.query_id}", flush=True)
    if state_dict_sha256(model) != weight_before:
        raise RuntimeError("Model weights changed")
    output = {
        "schema_version": 1,
        "candidate_generation_version": "goal_directed_mixture_v1",
        "status": "passed",
        "eval_seed": args.seed,
        "model": {
            "checkpoint": str(args.checkpoint.resolve()),
            "sha256": file_sha256(args.checkpoint.resolve()),
        },
        "stable_worldmodel": {"repo": str(stable_repo), "commit": stable_commit},
        "records": records,
    }
    write_json(args.output.resolve(), output)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slow-catalog", type=Path, required=True)
    parser.add_argument("--fast-catalog", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--normalizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--templates", nargs="+", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--stablewm-repo", default="../stable-worldmodel")
    parser.add_argument("--stablewm-ref", default=PINNED_STABLEWM)
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps({"status": result["status"], "records": len(result["records"])}))
