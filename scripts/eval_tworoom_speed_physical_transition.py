#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contextworld.evaluation.icl_model import file_sha256, state_dict_sha256
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
    _array_sha256,
    _balanced_evaluation_schedule,
    _fixed_variation_values,
    _load_query_assets,
    _load_selected_queries,
    image_transform,
)


HORIZONS = (1, 2, 3, 5, 10)
ACTION_BLOCK = 5


def _unit(value: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    if norm <= 1e-8:
        return np.asarray([1.0, 0.0], dtype=np.float32)
    return np.asarray(value / norm, dtype=np.float32)


def physical_action_probe(
    *,
    query_state: np.ndarray,
    evaluation_index: int,
    raw_steps: int = 50,
) -> tuple[str, np.ndarray]:
    """Return a bounded action probe directed into the room interior."""

    if raw_steps != 50:
        raise ValueError("The frozen physical probe uses 50 raw steps")
    center = np.asarray([167.0, 112.0], dtype=np.float32)
    direction = _unit(center - np.asarray(query_state, dtype=np.float32))
    perpendicular = np.asarray(
        [-direction[1], direction[0]], dtype=np.float32
    )
    family_index = int(evaluation_index) % 3
    if family_index == 0:
        family = "constant_direction"
        actions = np.repeat((0.12 * direction)[None], raw_steps, axis=0)
    elif family_index == 1:
        family = "varying_magnitude"
        magnitudes = np.asarray(
            [
                0.05,
                0.08,
                0.11,
                0.14,
                0.17,
                0.17,
                0.14,
                0.11,
                0.08,
                0.05,
            ],
            dtype=np.float32,
        )
        actions = np.concatenate(
            [
                np.repeat((magnitude * direction)[None], ACTION_BLOCK, axis=0)
                for magnitude in magnitudes
            ],
            axis=0,
        )
    else:
        family = "turning"
        angles = np.linspace(-0.75, 0.75, 10, dtype=np.float32)
        blocks = []
        for angle in angles:
            rotated = (
                math.cos(float(angle)) * direction
                + math.sin(float(angle)) * perpendicular
            )
            blocks.append(
                np.repeat((0.11 * rotated)[None], ACTION_BLOCK, axis=0)
            )
        actions = np.concatenate(blocks, axis=0)
    return family, np.asarray(actions, dtype=np.float32)


def _speed_grid(low: float, high: float, step: float) -> np.ndarray:
    if not low < high or step <= 0:
        raise ValueError("Invalid oracle speed grid")
    count = int(round((high - low) / step))
    grid = low + step * np.arange(count + 1, dtype=np.float64)
    if not np.isclose(grid[-1], high, rtol=0.0, atol=1e-9):
        raise ValueError("Oracle speed grid does not end at the upper bound")
    return np.round(grid, 8)


def _simulate_oracle_grid(
    *,
    episode: Any,
    raw_actions: np.ndarray,
    speed_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from stable_worldmodel.envs.two_room.env import TwoRoomEnv

    frames_by_speed = []
    states_by_speed = []
    terminated_by_speed = []
    env = TwoRoomEnv(render_mode="rgb_array")
    try:
        for speed in speed_grid:
            frames = []
            states = []
            ever_terminated = False
            env.reset(
                seed=episode.simulator_seed,
                options={
                    "variation": (),
                    "variation_values": _fixed_variation_values(
                        float(speed), episode.door_position
                    ),
                    "state": episode.query_state,
                    "target_state": episode.goal_state,
                },
            )
            for raw_index, action in enumerate(raw_actions, start=1):
                _, _, terminated, truncated, _ = env.step(action)
                ever_terminated = bool(
                    ever_terminated or terminated or truncated
                )
                if raw_index % ACTION_BLOCK == 0:
                    states.append(
                        env.unwrapped.agent_position.detach()
                        .cpu()
                        .numpy()
                        .astype(np.float32)
                    )
                    frames.append(np.asarray(env.render(), dtype=np.uint8))
            frames_by_speed.append(np.stack(frames))
            states_by_speed.append(np.stack(states))
            terminated_by_speed.append(ever_terminated)
    finally:
        env.close()
    return (
        np.stack(frames_by_speed),
        np.stack(states_by_speed).astype(np.float32),
        np.asarray(terminated_by_speed, dtype=bool),
    )


def _encode_oracle_grid(
    *,
    model: Any,
    frames: np.ndarray,
    transform: Any,
    device: str,
    chunk_speeds: int,
):
    import torch

    rows = []
    for start in range(0, len(frames), int(chunk_speeds)):
        stop = min(len(frames), start + int(chunk_speeds))
        chunk = frames[start:stop]
        flat = _images(
            chunk.reshape(-1, *chunk.shape[2:]), transform, device
        )
        tensor = flat.reshape(
            stop - start,
            chunk.shape[1],
            *flat.shape[1:],
        )
        with torch.inference_mode():
            rows.append(model.encode({"pixels": tensor})["emb"].detach())
    return torch.cat(rows, dim=0)


def _predict_rollout(
    *,
    model: Any,
    asset: dict[str, Any],
    condition: str,
    normalized_actions: np.ndarray,
    transform: Any,
    device: str,
):
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
    future_actions = torch.from_numpy(
        np.asarray(normalized_actions, dtype=np.float32)
    ).to(device=device, dtype=next(model.parameters()).dtype)
    model_info = {
        "pixels": torch.cat(
            [context_pixels[None, None], query_pixels[None, None]],
            dim=2,
        ),
        "goal": goal_pixels[None, None],
        "action": torch.zeros(
            1,
            1,
            1,
            future_actions.shape[-1],
            device=device,
            dtype=future_actions.dtype,
        ),
    }
    prompted_actions = torch.cat(
        [
            context_actions[None, None],
            future_actions[None, None],
        ],
        dim=2,
    )
    with torch.inference_mode():
        model.get_cost(model_info, prompted_actions)
    predicted = model_info["predicted_emb"][0, 0, -10:].detach()
    if predicted.shape[0] != 10:
        raise RuntimeError(f"Expected 10 predicted blocks: {predicted.shape}")
    return predicted


def _angle_error_degrees(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm <= 1e-8 or right_norm <= 1e-8:
        return 0.0 if left_norm <= 1e-8 and right_norm <= 1e-8 else 180.0
    cosine = float(
        np.clip(np.dot(left, right) / (left_norm * right_norm), -1.0, 1.0)
    )
    return float(np.degrees(np.arccos(cosine)))


def physical_metrics(
    *,
    predicted_embeddings: Any,
    oracle_embeddings: Any,
    oracle_states: np.ndarray,
    speed_grid: np.ndarray,
    query_speed: float,
    history_speed: float,
    query_state: np.ndarray,
) -> dict[str, Any]:
    import torch

    if predicted_embeddings.shape[0] != oracle_embeddings.shape[1]:
        raise ValueError("Prediction/oracle horizon mismatch")
    query_index = int(
        np.argmin(np.abs(speed_grid - float(query_speed)))
    )
    if not np.isclose(
        speed_grid[query_index], query_speed, rtol=0.0, atol=1e-6
    ):
        raise ValueError(f"Query speed {query_speed} is absent from grid")
    mse = (
        (
            oracle_embeddings
            - predicted_embeddings[None].to(oracle_embeddings.dtype)
        )
        ** 2
    ).mean(dim=-1)
    rows: dict[str, Any] = {}
    for horizon in HORIZONS:
        block_index = horizon - 1
        errors = mse[:, block_index]
        inferred_index = int(torch.argmin(errors).detach().cpu())
        inferred_speed = float(speed_grid[inferred_index])
        predicted_position = np.asarray(
            oracle_states[inferred_index, block_index], dtype=np.float32
        )
        true_position = np.asarray(
            oracle_states[query_index, block_index], dtype=np.float32
        )
        predicted_displacement = (
            predicted_position - np.asarray(query_state, dtype=np.float32)
        )
        true_displacement = (
            true_position - np.asarray(query_state, dtype=np.float32)
        )
        rows[str(horizon)] = {
            "inferred_speed": inferred_speed,
            "inferred_minus_history_speed": float(
                inferred_speed - history_speed
            ),
            "inferred_minus_query_speed": float(
                inferred_speed - query_speed
            ),
            "predicted_position": predicted_position.tolist(),
            "true_position": true_position.tolist(),
            "position_error_px": float(
                np.linalg.norm(predicted_position - true_position)
            ),
            "predicted_displacement": predicted_displacement.tolist(),
            "true_displacement": true_displacement.tolist(),
            "displacement_magnitude_error_px": float(
                abs(
                    np.linalg.norm(predicted_displacement)
                    - np.linalg.norm(true_displacement)
                )
            ),
            "displacement_direction_error_deg": _angle_error_degrees(
                predicted_displacement, true_displacement
            ),
            "latent_mse_to_true_query_future": float(
                errors[query_index].detach().cpu()
            ),
            "latent_mse_to_nearest_oracle": float(
                errors[inferred_index].detach().cpu()
            ),
        }
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    os.environ.setdefault("MUJOCO_GL", "egl")
    args.catalog = resolve_contextworld_path(args.catalog, repo_root=ROOT)
    args.output = resolve_contextworld_path(args.output, repo_root=ROOT)
    args.checkpoint = resolve_contextworld_path(
        args.checkpoint, repo_root=ROOT
    )
    args.normalizer = resolve_contextworld_path(
        args.normalizer, repo_root=ROOT
    )
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
    speed_grid = _speed_grid(
        args.oracle_speed_min,
        args.oracle_speed_max,
        args.oracle_speed_step,
    )

    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    all_templates = sorted(
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
        templates=all_templates,
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
    for record_index, scheduled in enumerate(schedule):
        asset = scheduled["asset"]
        episode = asset["episode"]
        family, raw_actions = physical_action_probe(
            query_state=episode.query_state,
            evaluation_index=scheduled["evaluation_index"],
        )
        normalized_actions = (
            process["action"]
            .transform(raw_actions)
            .astype(np.float32)
            .reshape(10, 10)
        )
        frames, states, terminated = _simulate_oracle_grid(
            episode=episode,
            raw_actions=raw_actions,
            speed_grid=speed_grid,
        )
        oracle_embeddings = _encode_oracle_grid(
            model=model,
            frames=frames,
            transform=transform,
            device=args.device,
            chunk_speeds=args.oracle_encode_chunk_speeds,
        )
        condition_rows = {}
        for condition in conditions:
            history_speed = float(
                asset["bundle"]["conditions"][condition]["factors"][
                    "agent.speed"
                ]
            )
            predicted = _predict_rollout(
                model=model,
                asset=asset,
                condition=condition,
                normalized_actions=normalized_actions,
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
                "by_horizon": physical_metrics(
                    predicted_embeddings=predicted,
                    oracle_embeddings=oracle_embeddings,
                    oracle_states=states,
                    speed_grid=speed_grid,
                    query_speed=float(episode.speed),
                    history_speed=history_speed,
                    query_state=episode.query_state,
                ),
                "context_pixels_sha256": asset["contexts"][condition][
                    "pixels_sha256"
                ],
                "context_actions_sha256": asset["contexts"][condition][
                    "raw_actions_sha256"
                ],
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
                "action_probe": {
                    "family": family,
                    "raw_actions_sha256": _array_sha256(raw_actions),
                    "normalized_actions_sha256": _array_sha256(
                        normalized_actions
                    ),
                    "raw_steps": len(raw_actions),
                    "action_blocks": 10,
                },
                "oracle": {
                    "speed_min": float(speed_grid[0]),
                    "speed_max": float(speed_grid[-1]),
                    "speed_step": float(args.oracle_speed_step),
                    "speed_count": len(speed_grid),
                    "any_termination_or_truncation": bool(
                        np.any(terminated)
                    ),
                },
                "conditions": condition_rows,
            }
        )
        print(
            f"[{record_index + 1}/{len(schedule)}] "
            f"query_speed={episode.speed:g} query={episode.query_id} "
            f"probe={family}",
            flush=True,
        )

    weight_after = state_dict_sha256(model)
    if weight_after != weight_before:
        raise RuntimeError("Model weights changed during evaluation")
    output = {
        "schema_version": 1,
        "benchmark": "tworoom_history3_speed_physical_transition_v2",
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
        "protocol": {
            **protocol,
            "evaluations_per_condition": len(schedule),
            "conditions": conditions,
            "horizons_action_blocks": list(HORIZONS),
            "horizons_raw_steps": [
                ACTION_BLOCK * value for value in HORIZONS
            ],
            "target_used_for_primary_metric": False,
            "physical_state_estimator": (
                "nearest exact-simulator oracle trajectory in the frozen "
                "model encoder latent space"
            ),
            "oracle_speed_grid": {
                "min": float(speed_grid[0]),
                "max": float(speed_grid[-1]),
                "step": float(args.oracle_speed_step),
                "count": len(speed_grid),
            },
            "action_probe_families": [
                "constant_direction",
                "varying_magnitude",
                "turning",
            ],
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
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--oracle-speed-min", type=float, default=2.5)
    parser.add_argument("--oracle-speed-max", type=float, default=8.0)
    parser.add_argument("--oracle-speed-step", type=float, default=0.05)
    parser.add_argument(
        "--oracle-encode-chunk-speeds", type=int, default=16
    )
    parser.add_argument("--stablewm-repo", default="../stable-worldmodel")
    parser.add_argument("--stablewm-ref", default=PINNED_STABLEWM)
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
