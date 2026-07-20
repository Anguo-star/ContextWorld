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
from contextworld.evaluation.planner_mechanism import (
    array_sha256,
    fixed_candidate_bank,
    simulate_tworoom_candidates,
)
from contextworld.evaluation.protocol import (
    frozen_normalizer_process,
    infer_model_protocol,
    load_pretrained_cost_model,
)
from contextworld.paths import artifact_path
from contextworld.synthesis.manifest import write_json
from contextworld.synthesis.stablewm import load_stable_worldmodel
from scripts.eval_tworoom_fixed_candidate_mechanism import (
    _add_goal_directed_candidates,
    _images,
)
from scripts.eval_tworoom_icl_planning import (
    PINNED_STABLEWM,
    _balanced_evaluation_schedule,
    _fixed_variation_values,
    _load_query_assets,
    _load_selected_queries,
    image_transform,
)


RAW_STEPS = tuple(range(5, 51, 5))


def _true_rollout(
    *,
    model: Any,
    episode: Any,
    raw_actions: np.ndarray,
    transform: Any,
    device: str,
) -> tuple[Any, list[float], list[list[float]]]:
    import torch
    from stable_worldmodel.envs.two_room.env import TwoRoomEnv

    actions = np.asarray(raw_actions, dtype=np.float32)
    if actions.shape != (50, 2):
        raise ValueError(f"Expected 50 raw actions, got {actions.shape}")
    env = TwoRoomEnv(render_mode="rgb_array")
    frames = []
    distances = []
    states = []
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
        for index, action in enumerate(actions, start=1):
            env.step(action)
            if index % 5 == 0:
                state = (
                    env.unwrapped.agent_position.detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
                states.append(state.tolist())
                distances.append(
                    float(
                        np.linalg.norm(
                            state - np.asarray(episode.goal_state)
                        )
                    )
                )
                frames.append(np.asarray(env.render(), dtype=np.uint8))
    finally:
        env.close()
    pixels = _images(np.stack(frames), transform, device)[None]
    with torch.inference_mode():
        embeddings = model.encode({"pixels": pixels})["emb"][0].detach()
    return embeddings, distances, states


def _condition_prediction(
    *,
    model: Any,
    asset: dict[str, Any],
    condition: str,
    normalized_probe: np.ndarray,
    true_embeddings: Any,
    transform: Any,
    device: str,
) -> dict[str, Any]:
    import torch

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
    probe = torch.from_numpy(
        np.asarray(normalized_probe, dtype=np.float32)
    ).to(device=device, dtype=next(model.parameters()).dtype)
    model_info = {
        "pixels": query_pixels[None, None],
        "goal": goal_pixels[None, None],
        "action": torch.zeros(
            1,
            1,
            1,
            probe.shape[-1],
            device=device,
            dtype=probe.dtype,
        ),
    }
    model_info["pixels"] = torch.cat(
        [
            context_pixels[None, None],
            model_info["pixels"],
        ],
        dim=2,
    )
    prompted_actions = torch.cat(
        [
            context_actions[None, None],
            probe[None, None],
        ],
        dim=2,
    )
    with torch.inference_mode():
        terminal_cost = model.get_cost(model_info, prompted_actions)
    predicted = model_info["predicted_emb"][0, 0, -10:]
    if predicted.shape != true_embeddings.shape:
        raise RuntimeError(
            "Prediction/target shape mismatch: "
            f"{predicted.shape} vs {true_embeddings.shape}"
        )
    mse = (
        ((predicted - true_embeddings) ** 2)
        .mean(dim=-1)
        .detach()
        .cpu()
        .float()
        .tolist()
    )
    return {
        "prediction_mse_to_true_by_block": mse,
        "terminal_latent_goal_cost": float(
            terminal_cost[0, 0].detach().cpu()
        ),
        "context_pixels_sha256": context["pixels_sha256"],
        "context_raw_actions_sha256": context["raw_actions_sha256"],
        "context_normalized_actions_sha256": context[
            "normalized_actions_sha256"
        ],
    }


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
        args.slow_catalog,
        speeds=[5.0, 5.1],
        templates=args.templates,
    )
    fast_bundles = _load_selected_queries(
        args.fast_catalog,
        speeds=[5.0, 5.1],
        templates=args.templates,
    )
    slow_assets = [
        _load_query_assets(row, process=process) for row in slow_bundles
    ]
    fast_assets = [
        _load_query_assets(row, process=process) for row in fast_bundles
    ]
    slow_ids = [asset["episode"].query_id for asset in slow_assets]
    if slow_ids != [asset["episode"].query_id for asset in fast_assets]:
        raise RuntimeError("Directional catalogs do not share query order")
    fast_by_query = {
        asset["episode"].query_id: asset for asset in fast_assets
    }
    query_index = {
        asset["episode"].query_id: index
        for index, asset in enumerate(slow_assets)
    }
    schedule = _balanced_evaluation_schedule(
        slow_assets,
        num_eval=args.num_eval,
        eval_seed=args.seed,
    )

    records = []
    for index, scheduled in enumerate(schedule):
        slow_asset = scheduled["asset"]
        episode = slow_asset["episode"]
        fast_asset = fast_by_query[episode.query_id]
        bank = fixed_candidate_bank(
            eval_seed=args.seed,
            evaluation_index=scheduled["evaluation_index"],
            query_index=query_index[episode.query_id],
        )
        bank = _add_goal_directed_candidates(bank, episode, process)
        raw_bank = (
            process["action"]
            .inverse_transform(bank.reshape(-1, 2))
            .astype(np.float32)
            .reshape(300, 25, 2)
        )
        exact = simulate_tworoom_candidates(
            query_state=episode.query_state,
            goal_state=episode.goal_state,
            raw_actions=raw_bank,
            speed=episode.speed,
            door_position=episode.door_position,
        )
        selected = int(np.argmin(exact["final_distances"]))
        first_half = raw_bank[selected]
        raw_probe = np.concatenate(
            [first_half, np.zeros((25, 2), dtype=np.float32)],
            axis=0,
        )
        normalized_probe = (
            process["action"]
            .transform(raw_probe)
            .astype(np.float32)
            .reshape(10, 10)
        )
        true_embeddings, true_distances, true_states = _true_rollout(
            model=model,
            episode=episode,
            raw_actions=raw_probe,
            transform=transform,
            device=args.device,
        )
        condition_assets = {
            "slow": (slow_asset, "wrong"),
            "correct": (slow_asset, "correct"),
            "fast": (fast_asset, "wrong"),
        }
        conditions = {
            label: _condition_prediction(
                model=model,
                asset=asset,
                condition=condition,
                normalized_probe=normalized_probe,
                true_embeddings=true_embeddings,
                transform=transform,
                device=args.device,
            )
            for label, (asset, condition) in condition_assets.items()
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
                "query_speed": float(episode.speed),
                "template_id": episode.template_id,
                "door_position": int(episode.door_position),
                "probe": {
                    "candidate_generation_version": (
                        "goal_directed_mixture_v1"
                    ),
                    "selected_candidate": selected,
                    "candidate_bank_sha256": array_sha256(bank),
                    "first_25_raw_actions_sha256": array_sha256(
                        first_half
                    ),
                    "full_50_raw_actions_sha256": array_sha256(raw_probe),
                    "last_25_raw_actions_are_zero": bool(
                        np.all(raw_probe[25:] == 0.0)
                    ),
                    "exact_correct_speed_final_distance_at_25": float(
                        exact["final_distances"][selected]
                    ),
                    "true_distances_by_block": true_distances,
                    "true_states_by_block": true_states,
                },
                "conditions": conditions,
            }
        )
        print(
            f"[{index + 1}/{len(schedule)}] {episode.query_id}",
            flush=True,
        )

    weight_after = state_dict_sha256(model)
    if weight_after != weight_before:
        raise RuntimeError("Model weights changed")
    output = {
        "schema_version": 1,
        "benchmark": "tworoom_long_horizon_prediction_v1",
        "status": "passed",
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
        "protocol": {
            **protocol,
            "evaluations": len(schedule),
            "prediction_horizon_action_blocks": 10,
            "prediction_horizon_raw_steps": 50,
            "observed_raw_steps": list(RAW_STEPS),
            "probe_first_25": (
                "exact-correct-speed terminal argmin from "
                "goal_directed_mixture_v1"
            ),
            "probe_last_25": "zero raw actions",
            "contexts": ["slow", "correct", "fast"],
        },
        "frozen_weight_audit": {
            "state_dict_sha256_before": weight_before,
            "state_dict_sha256_after": weight_after,
            "passed": weight_before == weight_after,
        },
        "records": records,
    }
    write_json(args.output.resolve(), output)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate SpeedFull prediction drift on a fixed 50-step "
            "probe with a 25-step zero-action tail."
        )
    )
    parser.add_argument("--slow-catalog", type=Path, required=True)
    parser.add_argument("--fast-catalog", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--normalizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--templates", nargs="+", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num-eval", type=int, default=50)
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
                "records": len(result["records"]),
            }
        )
    )
