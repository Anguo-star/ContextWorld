#!/usr/bin/env python3
"""Measure standard Push-T prediction calibration and BatchNorm drift.

This is a read-only mechanism diagnostic.  It evaluates deterministic
History-3 next-latent MSE on one fixed standard Push-T replay sample, then
repeats the same forward pass after restoring only the source checkpoint's
Projector and/or prediction-Projector BatchNorm buffers.  It does not train,
rank CEM candidates, or alter any checkpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

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
import run_pusht_hidden_actuation_mixed as mixed  # noqa: E402
import run_pusht_hidden_actuation_pilot as pilot  # noqa: E402


DEFAULT_OUTPUT = artifact_path(
    "evaluation/history3/pusht_hidden_actuation_h3_v1/"
    "prediction_calibration_diagnostic_seed12289/report.json"
)
BN_BUFFER_SUFFIXES = (
    "running_mean",
    "running_var",
    "num_batches_tracked",
)


def parse_named_paths(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--model must use NAME=CHECKPOINT")
        name, raw_path = value.split("=", 1)
        path = Path(raw_path).expanduser().resolve()
        if not name or name in result:
            raise ValueError(f"Invalid or duplicate model name {name!r}")
        if not path.exists():
            raise FileNotFoundError(path)
        result[name] = path
    if not result:
        raise ValueError("At least one --model is required")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument(
        "--source-checkpoint",
        type=Path,
        default=pilot.DEFAULT_CHECKPOINT,
    )
    parser.add_argument(
        "--original-lance",
        type=Path,
        default=mixed.DEFAULT_ORIGINAL_LANCE,
    )
    parser.add_argument(
        "--action-normalizer-source",
        type=Path,
        default=pilot.DEFAULT_ORIGINAL_DATASET,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=12289)
    parser.add_argument("--sample-count", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def fixed_replay_batches(
    *,
    path: Path,
    stats: dict[str, Any],
    seed: int,
    sample_count: int,
    batch_size: int,
) -> list[dict[str, torch.Tensor]]:
    if sample_count % batch_size:
        raise ValueError("sample-count must be divisible by batch-size")
    dataset, loader = mixed.original_loader(
        path,
        batch_size=batch_size,
        seed=seed,
        num_workers=0,
    )
    iterator = iter(loader)
    batches = []
    for _ in range(sample_count // batch_size):
        raw = next(iterator)
        batches.append(
            {
                "pixels": raw["pixels"].clone(),
                "action": pilot.normalize_action_blocks(
                    torch.nan_to_num(raw["action"].float(), 0.0),
                    stats,
                ).clone(),
            }
        )
    del iterator
    del loader
    del dataset
    return batches


def restore_bn_buffers(
    state: dict[str, torch.Tensor],
    source: dict[str, torch.Tensor],
    prefixes: tuple[str, ...],
) -> dict[str, torch.Tensor]:
    result = {name: value.clone() for name, value in state.items()}
    for name in result:
        if (
            name.split(".", 1)[0] in prefixes
            and name.endswith(BN_BUFFER_SUFFIXES)
        ):
            result[name] = source[name].clone()
    return result


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    batches: list[dict[str, torch.Tensor]],
    *,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    squared_errors = []
    targets = []
    predictions = []
    for raw in batches:
        pixels = pilot.preprocess_pixels(raw["pixels"], device)
        action = raw["action"].to(device=device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            output = model.encode({"pixels": pixels, "action": action})
            prediction = model.predict(
                output["emb"][:, :3],
                output["act_emb"][:, :3],
            )
            target = output["emb"][:, 1:]
        squared_errors.append(
            torch.square(prediction.float() - target.float()).mean(dim=-1).cpu()
        )
        targets.append(target.float().cpu())
        predictions.append(prediction.float().cpu())

    error = torch.cat(squared_errors)
    target = torch.cat(targets)
    prediction = torch.cat(predictions)
    target_centered = target - target.mean(dim=(0, 1), keepdim=True)
    prediction_centered = prediction - prediction.mean(
        dim=(0, 1),
        keepdim=True,
    )
    return {
        "sample_count": int(error.size(0)),
        "prediction_mse": float(error.mean()),
        "prediction_mse_by_transition": [
            float(value) for value in error.mean(dim=0)
        ],
        "target_per_dimension_variance": float(
            target_centered.square().mean()
        ),
        "prediction_per_dimension_variance": float(
            prediction_centered.square().mean()
        ),
        "prediction_to_target_variance_ratio": float(
            prediction_centered.square().mean()
            / target_centered.square().mean().clamp_min(1e-12)
        ),
        "target_mean_l2": float(
            torch.linalg.vector_norm(target.mean(dim=(0, 1)))
        ),
        "prediction_mean_l2": float(
            torch.linalg.vector_norm(prediction.mean(dim=(0, 1)))
        ),
        "prediction_target_mean_gap_l2": float(
            torch.linalg.vector_norm(
                prediction.mean(dim=(0, 1))
                - target.mean(dim=(0, 1))
            )
        ),
    }


def main() -> None:
    args = parse_args()
    if args.seed != 12289:
        raise ValueError("The registered diagnostic seed is 12289")
    models = parse_named_paths(args.model)
    output = Path(os.path.abspath(args.output.expanduser()))
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    pilot.set_reproducible_seed(args.seed)
    stats = pilot.original_action_stats(args.action_normalizer_source)
    batches = fixed_replay_batches(
        path=args.original_lance,
        stats=stats,
        seed=args.seed,
        sample_count=args.sample_count,
        batch_size=args.batch_size,
    )
    source_state = pilot.checkpoint_model_state(args.source_checkpoint)
    results: dict[str, Any] = {}
    for name, path in models.items():
        state = pilot.checkpoint_model_state(path)
        conditions = {
            "trained_state": state,
            "source_projector_bn": restore_bn_buffers(
                state,
                source_state,
                ("projector",),
            ),
            "source_prediction_projector_bn": restore_bn_buffers(
                state,
                source_state,
                ("pred_proj",),
            ),
            "source_both_projector_bn": restore_bn_buffers(
                state,
                source_state,
                ("projector", "pred_proj"),
            ),
        }
        rows = {}
        for condition, condition_state in conditions.items():
            model, _ = pilot.load_model(path, device=device)
            model.load_state_dict(condition_state, strict=True)
            rows[condition] = evaluate(model, batches, device=device)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        results[name] = {
            "checkpoint": str(path),
            "checkpoint_sha256": pilot.file_sha256(path),
            "conditions": rows,
        }
        print(
            name,
            {
                condition: row["prediction_mse"]
                for condition, row in rows.items()
            },
            flush=True,
        )

    report = {
        "schema_version": 1,
        "status": "standard_replay_prediction_calibration_diagnostic",
        "seed": args.seed,
        "sample_count": args.sample_count,
        "batch_size": args.batch_size,
        "source_checkpoint": {
            "path": str(args.source_checkpoint),
            "sha256": pilot.file_sha256(args.source_checkpoint),
        },
        "original_lance": str(args.original_lance),
        "model_mode": "eval",
        "ranking_metrics_computed": False,
        "results": results,
        "interpretation_scope": [
            "This diagnostic measures deterministic replay prediction MSE.",
            "It does not replace real-environment CEM evaluation.",
            "Buffer restoration is a read-only module-swap diagnostic.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": pilot.file_sha256(output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
