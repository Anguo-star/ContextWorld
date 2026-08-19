#!/usr/bin/env python3
"""Separate physical, real-latent, and predicted Push-T cost minima."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


CONTEXTWORLD_ROOT = Path(__file__).resolve().parents[1]
STABLE_WORLD_MODEL_ROOT = CONTEXTWORLD_ROOT.parent / "stable-worldmodel"
for source_root in (
    CONTEXTWORLD_ROOT,
    STABLE_WORLD_MODEL_ROOT,
    Path(__file__).resolve().parent,
):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from contextworld.evaluation.pusht_hidden_actuation import (  # noqa: E402
    MODE_SCALES,
    _variation_values,
)
from contextworld.paths import artifact_path  # noqa: E402
import eval_pusht_hidden_actuation_cem as cem  # noqa: E402
from stable_worldmodel.envs.pusht.env import PushT  # noqa: E402


DEFAULT_CHECKPOINT = artifact_path(
    "evaluation/history3/pusht_hidden_actuation_h3_v1/"
    "dynamics_response_sigreg_v2/training_seed13313_step2048/"
    "mixed_dynamics_response_sigreg_0p02_step2048.pt"
)
DEFAULT_OUTPUT = artifact_path(
    "evaluation/history3/pusht_hidden_actuation_h3_v1/"
    "dynamics_response_sigreg_v2/training_seed13313_step2048/"
    "hidden_cost_surface_diagnostic/report.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data-root", type=Path, default=cem.DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--original-dataset",
        type=Path,
        default=cem.DEFAULT_ORIGINAL_DATASET,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--grid-size", type=int, default=101)
    parser.add_argument("--encode-batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def execute_grid(
    condition: cem.Condition,
    amplitudes: np.ndarray,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    pixels = []
    outcomes = []
    for amplitude in amplitudes:
        env = PushT(
            resolution=224,
            with_target=True,
            render_mode="rgb_array",
        )
        env.action_scale = float(MODE_SCALES[condition.mode])
        direction = np.asarray(
            condition.contact_direction,
            dtype=np.float32,
        )
        actions = np.zeros((5, 2), dtype=np.float32)
        actions[:2] = np.float32(amplitude) * direction
        try:
            env.reset(
                seed=int(condition.template.simulator_seed),
                options={
                    "variation": (),
                    "variation_values": _variation_values(
                        condition.template
                    ),
                    "state": condition.template.reset_state,
                    "goal_state": condition.template.goal_state,
                },
            )
            for action in actions:
                env.step(action)
            final_state = np.asarray(env._get_obs(), dtype=np.float64)
            success, state_distance = env.eval_state(
                condition.template.goal_state,
                final_state,
            )
            image = np.asarray(env.render(), dtype=np.uint8).copy()
        finally:
            env.close()
        pixels.append(torch.from_numpy(image).permute(2, 0, 1))
        outcomes.append(
            {
                "amplitude": float(amplitude),
                "success": bool(success),
                "state_distance": float(state_distance),
            }
        )
    return torch.stack(pixels), outcomes


def selected_outcome(
    outcomes: list[dict[str, Any]],
    index: int,
) -> dict[str, Any]:
    return dict(outcomes[int(index)])


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for mode in MODE_SCALES:
        selected = [row for row in rows if row["mode"] == mode]
        result[mode] = {
            "count": len(selected),
            "mean_physical_minimum_amplitude": float(
                np.mean(
                    [
                        row["physical_minimum"]["amplitude"]
                        for row in selected
                    ]
                )
            ),
            "mean_real_latent_minimum_amplitude": float(
                np.mean(
                    [
                        row["real_latent_minimum"]["amplitude"]
                        for row in selected
                    ]
                )
            ),
            "mean_predicted_minimum_amplitude": float(
                np.mean(
                    [
                        row["predicted_minimum"]["amplitude"]
                        for row in selected
                    ]
                )
            ),
            "real_latent_minimum_success_rate": float(
                np.mean(
                    [
                        row["real_latent_minimum"]["success"]
                        for row in selected
                    ]
                )
            ),
            "predicted_minimum_success_rate": float(
                np.mean(
                    [
                        row["predicted_minimum"]["success"]
                        for row in selected
                    ]
                )
            ),
            "mean_absolute_predicted_to_real_latent_gap": float(
                np.mean(
                    [
                        abs(
                            row["predicted_minimum"]["amplitude"]
                            - row["real_latent_minimum"]["amplitude"]
                        )
                        for row in selected
                    ]
                )
            ),
            "mean_absolute_real_latent_to_physical_gap": float(
                np.mean(
                    [
                        abs(
                            row["real_latent_minimum"]["amplitude"]
                            - row["physical_minimum"]["amplitude"]
                        )
                        for row in selected
                    ]
                )
            ),
        }
    return result


def main() -> None:
    args = parse_args()
    if args.grid_size != 101:
        raise ValueError("The diagnostic grid size is fixed at 101")
    checkpoint = args.checkpoint.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    original_dataset = args.original_dataset.expanduser().resolve()
    output = Path(os.path.abspath(args.output.expanduser()))
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    for path in (
        checkpoint,
        data_root / "manifest.json",
        data_root / "eval_payloads",
        original_dataset,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    action_mean, action_std = cem.action_stats(original_dataset)
    conditions = cem.load_conditions(
        data_root,
        mean=action_mean,
        std=action_std,
    )
    model = cem.load_model(checkpoint, device)
    cache = cem.cache_model_inputs(
        model,
        conditions,
        device=device,
        batch_size=args.encode_batch_size,
    )
    grid = np.linspace(0.0, 1.0, args.grid_size, dtype=np.float32)
    rows = []
    for index, condition in enumerate(conditions):
        final_pixels, outcomes = execute_grid(condition, grid)
        real_embedding = cem.encode_observations(
            model,
            final_pixels[:, None],
            device=device,
            batch_size=args.encode_batch_size,
        )[:, 0]
        goal = cache["goal_embedding"][index]
        real_cost = torch.square(real_embedding - goal).sum(dim=-1).cpu()
        amplitudes = torch.from_numpy(grid).to(device)[None]
        with torch.no_grad(), torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            predicted_cost = cem.candidate_cost(
                model,
                cache,
                torch.tensor([index], device=device),
                amplitudes,
                action_mean=action_mean,
                action_std=action_std,
            )[0].float().cpu()
        physical_index = int(
            np.argmin([row["state_distance"] for row in outcomes])
        )
        real_index = int(torch.argmin(real_cost))
        predicted_index = int(torch.argmin(predicted_cost))
        row = {
            "condition_id": condition.condition_id,
            "pair_index": condition.pair_index,
            "mode": condition.mode,
            "physical_minimum": selected_outcome(
                outcomes,
                physical_index,
            ),
            "real_latent_minimum": {
                **selected_outcome(outcomes, real_index),
                "latent_cost": float(real_cost[real_index]),
            },
            "predicted_minimum": {
                **selected_outcome(outcomes, predicted_index),
                "predicted_cost": float(predicted_cost[predicted_index]),
                "real_latent_cost": float(real_cost[predicted_index]),
            },
        }
        rows.append(row)
        if (index + 1) % 10 == 0:
            print(
                f"[cost-surface] {index + 1}/{len(conditions)}",
                flush=True,
            )

    report = {
        "schema_version": 1,
        "status": "physical_real_latent_predicted_cost_surface_diagnostic",
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": cem.file_sha256(checkpoint),
        },
        "data": {
            "root": str(data_root),
            "manifest_sha256": cem.file_sha256(
                data_root / "manifest.json"
            ),
            "condition_count": len(conditions),
        },
        "grid": {
            "size": args.grid_size,
            "minimum": 0.0,
            "maximum": 1.0,
        },
        "summary_by_mode": summarize(rows),
        "records": rows,
        "interpretation": {
            "physical_to_real_latent_gap": (
                "goal-representation metric calibration"
            ),
            "real_latent_to_predicted_gap": (
                "predictor/action-response calibration"
            ),
            "candidate_ranking_or_spearman_computed": False,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": cem.file_sha256(output),
                "summary_by_mode": report["summary_by_mode"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
