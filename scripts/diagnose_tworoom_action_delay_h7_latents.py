#!/usr/bin/env python3
"""Diagnose History-7 prediction-direction alignment after formal scoring."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contextworld.benchmarks.adapters import (
    StableWorldModelLeWMHistory7Adapter,
)
from contextworld.evaluation.action_delay_h7_score import (
    load_h7_validation_assets,
    physical_future_group,
)
from contextworld.evaluation.action_delay_h7_validation import (
    DELAYS,
    FUTURE_HORIZONS,
    QUERY_COUNT,
    file_sha256,
)
from contextworld.paths import resolve_contextworld_path
from contextworld.synthesis.manifest import write_json


DEFAULT_CONFIG = (
    ROOT / "configs/benchmark/tworoom_action_delay_h7_scoring_v1.yaml"
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _models(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(row["slug"]): {**row, "role": role}
        for role, rows in config["models"].items()
        for row in rows
    }


def _artifact(
    config: dict[str, Any],
    field: str,
    slug: str,
) -> Path:
    return resolve_contextworld_path(
        str(config["model_artifact_pattern"][field]).format(slug=slug),
        repo_root=ROOT,
    )


def _alignment_metrics(
    predicted: np.ndarray,
    targets: np.ndarray,
    *,
    horizon: int | str,
) -> dict[str, Any]:
    _require(
        predicted.shape == targets.shape
        and predicted.shape[:2] == (QUERY_COUNT, len(DELAYS)),
        f"Unexpected diagnostic arrays: {predicted.shape}/{targets.shape}",
    )
    pairs = [
        (left, right)
        for left in DELAYS
        for right in DELAYS
        if left < right
        and physical_future_group(left, horizon)
        != physical_future_group(right, horizon)
    ]
    left = np.asarray([pair[0] for pair in pairs], dtype=np.int64)
    right = np.asarray([pair[1] for pair in pairs], dtype=np.int64)
    prediction_delta = predicted[:, left] - predicted[:, right]
    target_delta = targets[:, left] - targets[:, right]
    prediction_norm_sq = np.sum(prediction_delta**2, axis=-1)
    target_norm_sq = np.sum(target_delta**2, axis=-1)
    dot = np.sum(prediction_delta * target_delta, axis=-1)
    valid = (prediction_norm_sq > 1e-18) & (target_norm_sq > 1e-18)
    cosine = dot[valid] / np.sqrt(
        prediction_norm_sq[valid] * target_norm_sq[valid]
    )
    gain = dot[valid] / target_norm_sq[valid]

    prediction_centered = predicted - predicted.mean(axis=1, keepdims=True)
    target_centered = targets - targets.mean(axis=1, keepdims=True)
    centered_dot = np.sum(
        prediction_centered * target_centered, axis=(1, 2)
    )
    centered_prediction_norm = np.sum(
        prediction_centered**2, axis=(1, 2)
    )
    centered_target_norm = np.sum(target_centered**2, axis=(1, 2))
    centered_valid = (
        centered_prediction_norm > 1e-18
    ) & (centered_target_norm > 1e-18)
    centered_cosine = centered_dot[centered_valid] / np.sqrt(
        centered_prediction_norm[centered_valid]
        * centered_target_norm[centered_valid]
    )

    target_pair_mse = float(np.mean(target_delta**2))
    prediction_pair_mse = float(np.mean(prediction_delta**2))
    aligned_prediction_mse = float(np.mean((predicted - targets) ** 2))
    return {
        "queries": int(predicted.shape[0]),
        "physical_pair_comparisons_per_query": len(pairs),
        "target_pair_mse": target_pair_mse,
        "prediction_pair_mse": prediction_pair_mse,
        "prediction_to_target_pair_magnitude_ratio": float(
            np.sqrt(
                prediction_pair_mse / max(target_pair_mse, 1e-18)
            )
        ),
        "pair_delta_alignment_mse": float(
            np.mean((prediction_delta - target_delta) ** 2)
        ),
        "pair_delta_alignment_mse_over_target_pair_mse": float(
            np.mean((prediction_delta - target_delta) ** 2)
            / max(target_pair_mse, 1e-18)
        ),
        "pair_direction_cosine_mean": float(np.mean(cosine)),
        "pair_direction_cosine_median": float(np.median(cosine)),
        "pair_direction_positive_fraction": float(np.mean(dot[valid] > 0)),
        "pair_direction_gain_mean": float(np.mean(gain)),
        "centered_delay_pattern_cosine_mean": float(
            np.mean(centered_cosine)
        ),
        "centered_delay_pattern_cosine_median": float(
            np.median(centered_cosine)
        ),
        "aligned_prediction_mse": aligned_prediction_mse,
        "target_centered_variance": float(
            np.mean(target_centered**2)
        ),
        "prediction_centered_variance": float(
            np.mean(prediction_centered**2)
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--stablewm-repo")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    models = _models(config)
    _require(args.model in models, f"未知模型：{args.model}")
    model = models[args.model]
    checkpoint = _artifact(config, "checkpoint", args.model)
    report_path = _artifact(config, "training_report", args.model)
    formal_result = resolve_contextworld_path(
        Path(config["artifacts"]["results_root"]) / f"{args.model}.json",
        repo_root=ROOT,
    )
    _require(checkpoint.is_file(), f"checkpoint 不存在：{checkpoint}")
    _require(report_path.is_file(), f"训练报告不存在：{report_path}")
    _require(formal_result.is_file(), f"正式评分不存在：{formal_result}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    result = json.loads(formal_result.read_text(encoding="utf-8"))
    _require(
        report.get("passed") is True
        and result.get("status") == "completed"
        and result["identity"]["checkpoint_sha256"]
        == file_sha256(checkpoint),
        "训练或正式评分身份不一致",
    )

    catalog_path = resolve_contextworld_path(
        config["source_identity"]["validation_catalog"]["path"],
        repo_root=ROOT,
    )
    _, assets = load_h7_validation_assets(
        catalog_path, repo_root=ROOT
    )
    histories = np.concatenate(
        [asset["history_pixels"] for asset in assets], axis=0
    )
    actions = np.concatenate(
        [
            np.repeat(
                asset["action_blocks"][None], len(DELAYS), axis=0
            )
            for asset in assets
        ],
        axis=0,
    )
    target_pixels = np.concatenate(
        [asset["true_future_pixels"] for asset in assets], axis=0
    )
    normalizer = resolve_contextworld_path(
        config["source_identity"]["normalizer"]["path"],
        repo_root=ROOT,
    )
    adapter = StableWorldModelLeWMHistory7Adapter.from_checkpoint(
        checkpoint,
        normalizer=normalizer,
        repo_root=ROOT,
        stablewm_repo=str(
            args.stablewm_repo or config["stable_worldmodel"]["repo"]
        ),
        stablewm_ref=str(config["stable_worldmodel"]["commit"]),
        device=args.device,
    )
    before = adapter.frozen_state_hash()
    predicted = np.asarray(
        adapter.rollout_latents(
            histories,
            actions,
            batch_size=int(args.batch_size),
        ),
        dtype=np.float32,
    ).reshape(QUERY_COUNT, len(DELAYS), 3, -1)
    encoded = np.asarray(
        adapter.encode_pixels(
            target_pixels.reshape(-1, *target_pixels.shape[-3:]),
            batch_size=int(args.batch_size),
        ),
        dtype=np.float32,
    ).reshape(QUERY_COUNT, len(DELAYS), 3, -1)
    after = adapter.frozen_state_hash()
    _require(before == after, "latent 诊断期间模型状态发生变化")

    metrics = {
        f"h{horizon}": _alignment_metrics(
            predicted[:, :, index],
            encoded[:, :, index],
            horizon=horizon,
        )
        for index, horizon in enumerate(FUTURE_HORIZONS)
    }
    metrics["trajectory"] = _alignment_metrics(
        predicted.reshape(QUERY_COUNT, len(DELAYS), -1),
        encoded.reshape(QUERY_COUNT, len(DELAYS), -1),
        horizon="trajectory",
    )
    output = resolve_contextworld_path(
        (
            args.output
            if args.output is not None
            else Path(config["artifacts"]["results_root"])
            / "latent_diagnostics"
            / f"{args.model}.json"
        ),
        repo_root=ROOT,
    )
    payload = {
        "schema_version": 1,
        "benchmark": config["benchmark"],
        "status": "diagnostic_completed_not_part_of_primary_gate",
        "model_slug": args.model,
        "model_id": model["model_id"],
        "training_role": model["role"],
        "training_seed": int(model["training_seed"]),
        "identity": {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": file_sha256(checkpoint),
            "formal_result": str(formal_result),
            "formal_result_sha256": file_sha256(formal_result),
            "entrypoint": str(Path(__file__).resolve()),
            "entrypoint_sha256": file_sha256(Path(__file__).resolve()),
        },
        "model_state_sha256_before": before,
        "model_state_sha256_after": after,
        "metrics": metrics,
    }
    write_json(output, payload)
    print(
        json.dumps(
            {
                "model": args.model,
                "output": str(output),
                "metrics": metrics,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
