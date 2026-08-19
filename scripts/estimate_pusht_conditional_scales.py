#!/usr/bin/env python3
"""Estimate frozen source contrast scales for calibrated conditional SIGReg."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

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
import run_pusht_hidden_actuation_pilot as pilot  # noqa: E402


DEFAULT_PROTOCOL = CONTEXTWORLD_ROOT / (
    "configs/benchmark/"
    "pusht_hidden_actuation_scale_calibrated_sigreg_v1.yaml"
)
DEFAULT_OUTPUT = artifact_path(
    "evaluation/history3/pusht_hidden_actuation_h3_v1/"
    "scale_calibrated_sigreg_v1/source_scales.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hidden-data-root",
        type=Path,
        default=pilot.DEFAULT_DATA_ROOT,
    )
    parser.add_argument(
        "--action-normalizer-source",
        type=Path,
        default=pilot.DEFAULT_ORIGINAL_DATASET,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=pilot.DEFAULT_CHECKPOINT,
    )
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--benchmark",
        default="pusht_hidden_actuation_history3_action_coverage_v2",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size != 128:
        raise ValueError("The registered image batch size is 128")
    output = Path(os.path.abspath(args.output.expanduser()))
    protocol = args.protocol.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if not protocol.exists():
        raise FileNotFoundError(f"Missing protocol: {protocol}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    action_stats = pilot.original_action_stats(
        args.action_normalizer_source
    )
    hidden = pilot.materialize_lance_split(
        args.hidden_data_root / "train.lance",
        action_stats=action_stats,
    )
    model, checkpoint_receipt = pilot.load_model(
        args.checkpoint,
        device=device,
    )
    model.eval()
    _, projected = pilot.encode_pixels(
        model,
        hidden.pixels,
        device=device,
        batch_size=args.batch_size,
    )
    pairs = torch.arange(projected.size(0)).reshape(-1, 2)
    contrasts = (
        projected[pairs[:, 0]] - projected[pairs[:, 1]]
    ).double() / (2.0**0.5)
    scales = contrasts.square().mean(dim=(0, 2)).sqrt()
    pair_mse = (
        projected[pairs[:, 0]].double()
        - projected[pairs[:, 1]].double()
    ).square().mean(dim=(0, 2))
    active = [False, True, False, True]
    active_scales = {
        str(index): float(scales[index])
        for index, enabled in enumerate(active)
        if enabled
    }
    if not all(
        torch.isfinite(scales[index]) and scales[index] > 0
        for index, enabled in enumerate(active)
        if enabled
    ):
        raise RuntimeError(f"Invalid active source scales: {active_scales}")

    report = {
        "schema_version": 1,
        "status": "frozen_source_conditional_scales_estimated",
        "benchmark": str(args.benchmark),
        "method": "scale_calibrated_conditional_sigreg",
        "protocol": {
            "path": str(protocol),
            "sha256": pilot.file_sha256(protocol),
        },
        "source_checkpoint": checkpoint_receipt,
        "training_split": {
            "path": str(args.hidden_data_root / "train.lance"),
            "pair_count": hidden.pair_count,
            "sample_count": int(hidden.pixels.size(0)),
            "manifest_sha256": pilot.file_sha256(
                args.hidden_data_root / "manifest.json"
            ),
            "evaluation_split_used": False,
        },
        "estimator": {
            "formula": "RMS((z_low-z_high)/sqrt(2))",
            "model_mode": "eval",
            "encoder_output_dtype": str(projected.dtype),
            "accumulation_dtype": "torch.float64",
            "image_batch_size": args.batch_size,
            "embedding_dimension": int(projected.size(-1)),
            "active_observation_indices": [1, 3],
        },
        "contrast_scales_by_observation_index": {
            str(index): float(value)
            for index, value in enumerate(scales)
        },
        "active_contrast_scales": active_scales,
        "pair_mse_by_observation_index": {
            str(index): float(value)
            for index, value in enumerate(pair_mse)
        },
        "frozen_during_training": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": pilot.file_sha256(output),
                "active_contrast_scales": active_scales,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
