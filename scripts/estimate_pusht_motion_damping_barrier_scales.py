#!/usr/bin/env python3
"""Freeze paired-geometry loss divisors on Motion Damping Training only."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = Path(__file__).resolve().parent
STABLE_WORLD_MODEL_ROOT = ROOT.parent / "stable-worldmodel"
for source in (ROOT, SCRIPT_ROOT, STABLE_WORLD_MODEL_ROOT):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

from contextworld.benchmarks.motion_damping_icl_data import (  # noqa: E402
    _read_lance_pairs as read_motion_damping_pairs,
    file_sha256,
)
from contextworld.training.paired_prediction_geometry import (  # noqa: E402
    paired_prediction_geometry_terms,
)
import run_pusht_contact_friction_h3_train as base  # noqa: E402
import run_pusht_hidden_actuation_pilot as pilot  # noqa: E402


DEFAULT_DATA_ROOT = (
    ROOT / "artifacts/synthesis/pusht_motion_damping_h3_release_v4"
)
DEFAULT_OUTPUT = (
    ROOT
    / "artifacts/evaluation/history3/pusht_motion_damping_release_v1/"
    "release_audit/motion_damping_center_barrier_training_scales.json"
)
EXPECTED_TRAINING_PAIRS = 8192
EXPECTED_MANIFEST_SHA256 = (
    "48246aa4ae4a13d5b1c9677ba37a92fe114129027745f8e258137a016899563b"
)
LOSS_MODULE_PATH = (
    ROOT / "contextworld/training/paired_prediction_geometry.py"
)
EXPECTED_LOSS_MODULE_SHA256 = (
    "3d657ccc3d24fff4a2228974f261273a3a2e4959cb73f67e2255a01866ac6ba5"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=base.DEFAULT_CHECKPOINT,
    )
    parser.add_argument(
        "--original-h5",
        type=Path,
        default=base.DEFAULT_ORIGINAL_H5,
    )
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if file_sha256(LOSS_MODULE_PATH) != EXPECTED_LOSS_MODULE_SHA256:
        raise RuntimeError("Paired-geometry loss changed after preregistration")
    data_root = args.data_root.expanduser().resolve()
    manifest_path = data_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_pairs = int(manifest["pair_counts"]["train"])
    if (
        file_sha256(manifest_path) != EXPECTED_MANIFEST_SHA256
        or manifest.get("passed") is not True
        or expected_pairs != EXPECTED_TRAINING_PAIRS
        or manifest.get("protocol")
        != "pusht_motion_damping_history3_strict_causal_release_v3"
    ):
        raise RuntimeError(
            "Expected the audited 8,192-pair motion-damping v4 Training set"
        )
    if manifest["splits"]["train"].get("passed") is not True:
        raise RuntimeError("Motion-damping v4 Training audit did not pass")

    output = Path(os.path.abspath(args.output.expanduser()))
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite output: {output}")
    checkpoint = args.checkpoint.expanduser().resolve()
    original_h5 = args.original_h5.expanduser().resolve()
    device = torch.device(args.device)
    action_stats = pilot.original_action_stats(original_h5)
    original_reader = base._read_lance_pairs
    base._read_lance_pairs = read_motion_damping_pairs
    try:
        training = base._training_split(
            data_root / manifest["splits"]["train"]["table_path"],
            expected_pairs=expected_pairs,
            action_stats=action_stats,
        )
    finally:
        base._read_lance_pairs = original_reader

    model, checkpoint_receipt = pilot.load_model(
        checkpoint,
        device=device,
    )
    model.eval()
    with torch.inference_mode():
        predictions = pilot.predict_histories(
            model,
            training.pixels[:, :3],
            training.action[:, :3],
            device=device,
            batch_size=args.batch_size,
        )
        _, targets = pilot.encode_pixels(
            model,
            training.pixels[:, 3:4],
            device=device,
            batch_size=args.batch_size,
        )
    targets = targets[:, 0]
    terms = paired_prediction_geometry_terms(
        predicted_left=predictions[0::2],
        predicted_right=predictions[1::2],
        target_left=targets[0::2],
        target_right=targets[1::2],
        history_margin=0.20,
        response_reference_ratio=1.50,
    )
    raw = {name: float(value.detach()) for name, value in terms.items()}
    response_reference_loss = math.log(1.50) ** 2
    divisors = {
        "center_barrier_loss": raw["center_barrier_loss"],
        "response_calibration_loss": min(
            raw["response_calibration_loss"],
            response_reference_loss,
        ),
    }
    passed = bool(
        all(value > 1.0e-8 for value in divisors.values())
        and training.pair_count == expected_pairs
    )
    report = {
        "schema_version": 1,
        "status": "passed" if passed else "failed",
        "role": "training_only_loss_scale_freeze_not_model_selection",
        "public_test_opened": False,
        "development_opened": False,
        "data": {
            "root": str(data_root),
            "manifest_sha256": file_sha256(manifest_path),
            "split": "train",
            "pair_count": training.pair_count,
            "complete_conditions": 2 * training.pair_count,
            "table_sha256": manifest["splits"]["train"]["table_sha256"],
        },
        "checkpoint": checkpoint_receipt,
        "definition": {
            "history_margin": 0.20,
            "response_unique_zero_ratio": 1.0,
            "response_reference_ratio": 1.50,
            "response_reference_loss": response_reference_loss,
        },
        "raw_training_means": raw,
        "frozen_loss_divisors": divisors,
        "normalization": (
            "The center loss uses its full-Training mean at the common "
            "original initialization. The response divisor is the smaller "
            "of its Training mean and log(1.5)^2, ensuring a ratio of 1.5 "
            "has at least unit normalized penalty."
        ),
        "passed": passed,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
