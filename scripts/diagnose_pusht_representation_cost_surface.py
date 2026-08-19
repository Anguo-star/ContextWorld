#!/usr/bin/env python3
"""Localize PushT goal-metric distortion to Encoder or Projector space.

For every hidden-actuation condition, the script executes a fixed 101-point
real-environment action grid.  It then finds the physical optimum and the
nearest real future to the goal in raw Encoder space and Projector space.
No predicted futures, candidate ranking proxy, or CEM approximation is used.
"""

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

from contextworld.paths import artifact_path  # noqa: E402
import diagnose_pusht_hidden_cost_surface as existing  # noqa: E402
import eval_pusht_hidden_actuation_cem as cem  # noqa: E402


DEFAULT_OUTPUT = artifact_path(
    "evaluation/history3/pusht_hidden_actuation_h3_v1/"
    "representation_cost_surface_comparison/report.json"
)


def parse_models(values: list[str]) -> dict[str, Path]:
    models: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--model must use NAME=CHECKPOINT")
        name, raw_path = value.split("=", 1)
        path = Path(raw_path).expanduser().resolve()
        if not name or name in models:
            raise ValueError(f"Invalid or duplicate model name {name!r}")
        if not path.exists():
            raise FileNotFoundError(path)
        models[name] = path
    if not models:
        raise ValueError("At least one --model is required")
    return models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", default=[])
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


@torch.no_grad()
def encode_layers(
    model: torch.nn.Module,
    pixels: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    raw_rows = []
    projected_rows = []
    for start in range(0, pixels.size(0), batch_size):
        values = cem.preprocess_pixels(
            pixels[start : start + batch_size],
            device=device,
        )
        batch, frames = values.shape[:2]
        raw = model.encoder(
            values.flatten(0, 1),
            interpolate_pos_encoding=True,
        ).last_hidden_state[:, 0]
        projected = model.projector(raw)
        raw_rows.append(raw.reshape(batch, frames, -1).float())
        projected_rows.append(
            projected.reshape(batch, frames, -1).float()
        )
    return torch.cat(raw_rows), torch.cat(projected_rows)


def minimum_record(
    outcomes: list[dict[str, Any]],
    costs: torch.Tensor,
) -> dict[str, Any]:
    index = int(torch.argmin(costs))
    return {
        **outcomes[index],
        "grid_index": index,
        "latent_cost": float(costs[index]),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output = {}
    for mode in ("low_gain", "high_gain"):
        selected = [row for row in rows if row["mode"] == mode]
        mode_summary: dict[str, Any] = {"count": len(selected)}
        for space in ("raw_encoder", "projector"):
            mode_summary[space] = {
                "mean_minimum_amplitude": float(
                    np.mean(
                        [
                            row[f"{space}_minimum"]["amplitude"]
                            for row in selected
                        ]
                    )
                ),
                "minimum_success_rate": float(
                    np.mean(
                        [
                            row[f"{space}_minimum"]["success"]
                            for row in selected
                        ]
                    )
                ),
                "mean_absolute_amplitude_gap_to_physical": float(
                    np.mean(
                        [
                            abs(
                                row[f"{space}_minimum"]["amplitude"]
                                - row["physical_minimum"]["amplitude"]
                            )
                            for row in selected
                        ]
                    )
                ),
            }
        mode_summary["mean_physical_minimum_amplitude"] = float(
            np.mean(
                [
                    row["physical_minimum"]["amplitude"]
                    for row in selected
                ]
            )
        )
        output[mode] = mode_summary
    return output


def summarize_pixel(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output = {}
    for mode in ("low_gain", "high_gain"):
        selected = [row for row in rows if row["mode"] == mode]
        output[mode] = {
            "count": len(selected),
            "mean_physical_minimum_amplitude": float(
                np.mean(
                    [
                        row["physical_minimum"]["amplitude"]
                        for row in selected
                    ]
                )
            ),
            "mean_pixel_minimum_amplitude": float(
                np.mean(
                    [
                        row["pixel_minimum"]["amplitude"]
                        for row in selected
                    ]
                )
            ),
            "pixel_minimum_success_rate": float(
                np.mean(
                    [
                        row["pixel_minimum"]["success"]
                        for row in selected
                    ]
                )
            ),
            "mean_absolute_amplitude_gap_to_physical": float(
                np.mean(
                    [
                        abs(
                            row["pixel_minimum"]["amplitude"]
                            - row["physical_minimum"]["amplitude"]
                        )
                        for row in selected
                    ]
                )
            ),
        }
    return output


def main() -> None:
    args = parse_args()
    if args.grid_size != 101:
        raise ValueError("The registered grid size is fixed at 101")
    models = parse_models(args.model)
    output = Path(os.path.abspath(args.output.expanduser()))
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    action_mean, action_std = cem.action_stats(args.original_dataset)
    conditions = cem.load_conditions(
        args.data_root,
        mean=action_mean,
        std=action_std,
    )
    loaded = {
        name: cem.load_model(path, device)
        for name, path in models.items()
    }
    goals = torch.stack([row.goal_pixels for row in conditions])[:, None]
    goal_embeddings = {}
    for name, model in loaded.items():
        raw, projected = encode_layers(
            model,
            goals,
            device=device,
            batch_size=args.encode_batch_size,
        )
        goal_embeddings[name] = {
            "raw_encoder": raw[:, 0],
            "projector": projected[:, 0],
        }

    grid = np.linspace(0.0, 1.0, args.grid_size, dtype=np.float32)
    model_rows: dict[str, list[dict[str, Any]]] = {
        name: [] for name in models
    }
    pixel_rows: list[dict[str, Any]] = []
    for condition_index, condition in enumerate(conditions):
        final_pixels, outcomes = existing.execute_grid(condition, grid)
        physical_index = int(
            np.argmin([row["state_distance"] for row in outcomes])
        )
        physical_minimum = dict(outcomes[physical_index])
        pixel_cost = torch.square(
            final_pixels.float().div(255.0)
            - condition.goal_pixels.float().div(255.0)
        ).mean(dim=(1, 2, 3))
        pixel_rows.append(
            {
                "condition_id": condition.condition_id,
                "pair_index": condition.pair_index,
                "mode": condition.mode,
                "physical_minimum": physical_minimum,
                "pixel_minimum": minimum_record(outcomes, pixel_cost),
            }
        )
        for name, model in loaded.items():
            raw, projected = encode_layers(
                model,
                final_pixels[:, None],
                device=device,
                batch_size=args.encode_batch_size,
            )
            raw_cost = torch.square(
                raw[:, 0] - goal_embeddings[name]["raw_encoder"][condition_index]
            ).sum(dim=-1)
            projected_cost = torch.square(
                projected[:, 0]
                - goal_embeddings[name]["projector"][condition_index]
            ).sum(dim=-1)
            model_rows[name].append(
                {
                    "condition_id": condition.condition_id,
                    "pair_index": condition.pair_index,
                    "mode": condition.mode,
                    "physical_minimum": physical_minimum,
                    "raw_encoder_minimum": minimum_record(
                        outcomes,
                        raw_cost,
                    ),
                    "projector_minimum": minimum_record(
                        outcomes,
                        projected_cost,
                    ),
                }
            )
        if (condition_index + 1) % 10 == 0:
            print(
                f"[representation-surface] "
                f"{condition_index + 1}/{len(conditions)}",
                flush=True,
            )

    report = {
        "schema_version": 1,
        "status": "real_environment_representation_cost_surface_diagnostic",
        "benchmark": "pusht_hidden_actuation_history3_action_coverage_v2",
        "grid": {
            "minimum": 0.0,
            "maximum": 1.0,
            "points": args.grid_size,
        },
        "models": {
            name: {
                "checkpoint": str(models[name]),
                "checkpoint_sha256": cem.file_sha256(models[name]),
                "summary_by_mode": summarize(rows),
                "rows": rows,
            }
            for name, rows in model_rows.items()
        },
        "pixel_space": {
            "summary_by_mode": summarize_pixel(pixel_rows),
            "rows": pixel_rows,
        },
        "interpretation_scope": [
            "Every latent minimum is computed from a real simulator future.",
            "The raw Encoder and Projector see identical rendered images.",
            "Pixel-space MSE uses the same rendered images without a model.",
            "This diagnostic does not evaluate Predictor or CEM search.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": cem.file_sha256(output),
                "summaries": {
                    name: value["summary_by_mode"]
                    for name, value in report["models"].items()
                },
                "pixel_space": report["pixel_space"]["summary_by_mode"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
