#!/usr/bin/env python3
"""Diagnose strict contact-friction response sign and scale by family.

The diagnostic uses Training or Development only.  It never opens Public
Test and never changes a checkpoint.  The two strict construction families
are read from the data manifest and are used only for analysis.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
for source_root in (
    ROOT,
    ROOT.parent / "stable-worldmodel",
    Path(__file__).resolve().parent,
):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from contextworld.benchmarks.contact_friction_icl_data import (  # noqa: E402
    DEFAULT_CONTACT_FRICTION_RELEASE_CONFIG,
    _read_lance_pairs,
    file_sha256,
    load_contact_friction_icl_release,
)
from contextworld.paths import resolve_contextworld_path  # noqa: E402
import run_pusht_contact_friction_h3_train as friction  # noqa: E402
import run_pusht_hidden_actuation_pilot as pilot  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--release-config",
        type=Path,
        default=DEFAULT_CONTACT_FRICTION_RELEASE_CONFIG,
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--response-scales", type=Path, required=True)
    parser.add_argument(
        "--split",
        choices=("train", "loader_validation"),
        default="loader_validation",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--original-h5",
        type=Path,
        default=friction.DEFAULT_ORIGINAL_H5,
    )
    return parser.parse_args()


def _mean(values: torch.Tensor) -> float:
    return float(values.double().mean())


def _summary(
    indices: np.ndarray,
    *,
    target: torch.Tensor,
    predicted: torch.Tensor,
    history: torch.Tensor,
    low_to_low: torch.Tensor,
    low_to_high: torch.Tensor,
    high_to_low: torch.Tensor,
    high_to_high: torch.Tensor,
    history_scale: float,
    future_scale: float,
) -> dict[str, Any]:
    take = torch.from_numpy(indices)
    target = target[take]
    predicted = predicted[take]
    history = history[take]
    cosine_target_prediction = torch.nn.functional.cosine_similarity(
        target,
        predicted,
        dim=-1,
        eps=1e-12,
    )
    cosine_history_target = torch.nn.functional.cosine_similarity(
        history,
        target,
        dim=-1,
        eps=1e-12,
    )
    cosine_history_prediction = torch.nn.functional.cosine_similarity(
        history,
        predicted,
        dim=-1,
        eps=1e-12,
    )

    def normalized_rms(values: torch.Tensor, scale: float) -> torch.Tensor:
        return (
            values.double().square().mean(dim=-1).sqrt()
            / (2.0**0.5)
            / scale
        )

    low_correct = low_to_low[take] < low_to_high[take]
    high_correct = high_to_high[take] < high_to_low[take]
    return {
        "pair_count": int(indices.size),
        "correct_future_rate": _mean(
            torch.cat([low_correct, high_correct]).float()
        ),
        "low_correct_future_rate": _mean(low_correct.float()),
        "high_correct_future_rate": _mean(high_correct.float()),
        "target_prediction_dot_positive_rate": _mean(
            ((target * predicted).sum(dim=-1) > 0).float()
        ),
        "target_prediction_cosine": {
            "mean": _mean(cosine_target_prediction),
            "minimum": float(cosine_target_prediction.min()),
            "maximum": float(cosine_target_prediction.max()),
        },
        "history_target_cosine_mean": _mean(cosine_history_target),
        "history_prediction_cosine_mean": _mean(
            cosine_history_prediction
        ),
        "source_scale_normalized_rms": {
            "history_response": _mean(
                normalized_rms(history, history_scale)
            ),
            "true_future_response": _mean(
                normalized_rms(target, future_scale)
            ),
            "predicted_response": _mean(
                normalized_rms(predicted, future_scale)
            ),
        },
    }


def main() -> None:
    args = parse_args()
    output = Path(os.path.abspath(args.output.expanduser()))
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite output: {output}")
    release_path = args.release_config.expanduser().resolve()
    release = load_contact_friction_icl_release(release_path)
    data_root = resolve_contextworld_path(
        release["data"]["artifact_tree"]["root"],
        repo_root=ROOT,
    )
    expected_pairs = int(release["data"]["pair_counts"][args.split])
    arrays = _read_lance_pairs(
        data_root / release["data"]["lance_tables"][args.split],
        expected_pairs=expected_pairs,
        expected_split=args.split,
    )
    manifest = json.loads(
        (data_root / "manifest.json").read_text(encoding="utf-8")
    )
    family_by_pair = {
        str(row["template"]["template_id"]): int(
            row["template"]["strict_family_id"]
        )
        for row in manifest["splits"][args.split]["pairs"]
    }
    family = np.asarray(
        [family_by_pair[pair_id] for pair_id in arrays.pair_ids],
        dtype=np.int64,
    )
    if set(family.tolist()) != {0, 1}:
        raise RuntimeError("Both strict families are required")

    def pixels(values: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(values.copy()).permute(0, 1, 4, 2, 3)

    low_pixels = pixels(arrays.low_pixels)
    high_pixels = pixels(arrays.high_pixels)
    raw_actions = torch.from_numpy(
        arrays.raw_action_blocks.copy()
    ).reshape(expected_pairs, 4, friction.ACTION_INPUT_DIM)
    action_stats = pilot.original_action_stats(
        args.original_h5.expanduser().resolve()
    )
    actions = pilot.normalize_action_blocks(
        raw_actions.float(),
        action_stats,
    )

    device = torch.device(args.device)
    model, checkpoint_receipt = pilot.load_model(
        args.checkpoint.expanduser().resolve(),
        device=device,
    )
    model.eval()
    histories = torch.cat([low_pixels[:, :3], high_pixels[:, :3]])
    history_actions = torch.cat([actions[:, :3], actions[:, :3]])
    predicted = pilot.predict_histories(
        model,
        histories,
        history_actions,
        device=device,
        batch_size=args.batch_size,
    )
    predicted_low = predicted[:expected_pairs]
    predicted_high = predicted[expected_pairs:]
    predicted_response = predicted_high - predicted_low

    future_pixels = torch.cat(
        [low_pixels[:, 3:4], high_pixels[:, 3:4]]
    )
    _, future = pilot.encode_pixels(
        model,
        future_pixels,
        device=device,
        batch_size=args.batch_size,
    )
    target_low = future[:expected_pairs, 0]
    target_high = future[expected_pairs:, 0]
    target_response = target_high - target_low

    history_pixels = torch.cat(
        [low_pixels[:, 1:2], high_pixels[:, 1:2]]
    )
    _, history_embedding = pilot.encode_pixels(
        model,
        history_pixels,
        device=device,
        batch_size=args.batch_size,
    )
    history_response = (
        history_embedding[expected_pairs:, 0]
        - history_embedding[:expected_pairs, 0]
    )

    def mse(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return (left - right).square().mean(dim=-1)

    low_to_low = mse(predicted_low, target_low)
    low_to_high = mse(predicted_low, target_high)
    high_to_low = mse(predicted_high, target_low)
    high_to_high = mse(predicted_high, target_high)

    scales_path = args.response_scales.expanduser().resolve()
    scales = json.loads(scales_path.read_text(encoding="utf-8"))[
        "active_contrast_scales"
    ]
    history_scale = float(scales["1"])
    future_scale = float(scales["3"])
    summaries = {
        "all": _summary(
            np.arange(expected_pairs),
            target=target_response,
            predicted=predicted_response,
            history=history_response,
            low_to_low=low_to_low,
            low_to_high=low_to_high,
            high_to_low=high_to_low,
            high_to_high=high_to_high,
            history_scale=history_scale,
            future_scale=future_scale,
        ),
        **{
            f"strict_family_{value}": _summary(
                np.flatnonzero(family == value),
                target=target_response,
                predicted=predicted_response,
                history=history_response,
                low_to_low=low_to_low,
                low_to_high=low_to_high,
                high_to_low=high_to_low,
                high_to_high=high_to_high,
                history_scale=history_scale,
                future_scale=future_scale,
            )
            for value in (0, 1)
        },
    }
    family_target_means = [
        target_response[torch.from_numpy(np.flatnonzero(family == value))]
        .double()
        .mean(dim=0)
        for value in (0, 1)
    ]
    family_prediction_means = [
        predicted_response[
            torch.from_numpy(np.flatnonzero(family == value))
        ]
        .double()
        .mean(dim=0)
        for value in (0, 1)
    ]
    payload = {
        "schema_version": 1,
        "status": "completed",
        "role": "training_and_development_root_cause_only",
        "public_test_opened": False,
        "release": {
            "release_id": release["release_id"],
            "manifest_sha256": release["data"]["manifest_sha256"],
        },
        "split": args.split,
        "checkpoint": checkpoint_receipt,
        "response_scales": {
            "path": str(scales_path),
            "sha256": file_sha256(scales_path),
            "history_observation_1": history_scale,
            "future_observation_3": future_scale,
        },
        "response_definition": {
            "orientation": "high_friction_minus_low_friction",
            "history": "projected_z_high_x1 - projected_z_low_x1",
            "target": "projected_z_high_x3 - projected_z_low_x3",
            "prediction": "predicted_z_high_x3 - predicted_z_low_x3",
        },
        "summaries": summaries,
        "family_mean_response_cosine": {
            "target_family_0_vs_1": float(
                torch.nn.functional.cosine_similarity(
                    family_target_means[0],
                    family_target_means[1],
                    dim=0,
                    eps=1e-12,
                )
            ),
            "prediction_family_0_vs_1": float(
                torch.nn.functional.cosine_similarity(
                    family_prediction_means[0],
                    family_prediction_means[1],
                    dim=0,
                    eps=1e-12,
                )
            ),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
