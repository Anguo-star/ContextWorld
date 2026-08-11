from __future__ import annotations

import json
from pathlib import Path
import statistics
from typing import Any, Iterable

import numpy as np

from contextworld.benchmarks.adapters import ReacherArmMassICLModelAdapter
from contextworld.benchmarks.reacher_arm_mass_icl_data import (
    DEFAULT_REACHER_ARM_MASS_RELEASE_CONFIG,
    ReacherArmMassICLEvalDataset,
    file_sha256,
    load_reacher_arm_mass_icl_release,
)
from contextworld.benchmarks.paired_latent_response import (
    paired_latent_response_gate_checks,
    paired_latent_response_metrics,
)
from contextworld.paths import repository_root


def _mse(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.square(left - right).mean(axis=-1)


def _paired_bootstrap_lower_bound(
    values: np.ndarray,
    *,
    resamples: int = 10_000,
    seed: int = 2026080204,
    confidence: float = 0.95,
) -> float:
    """Bootstrap whole query pairs so the two mass decisions stay paired."""

    rows = np.asarray(values, dtype=np.float64)
    if rows.ndim == 1:
        rows = rows[:, None]
    if rows.ndim != 2 or not len(rows):
        raise ValueError("Bootstrap values must have shape (pairs, decisions)")
    rng = np.random.default_rng(seed)
    estimates = []
    remaining = int(resamples)
    while remaining:
        count = min(1_000, remaining)
        indices = rng.integers(0, len(rows), size=(count, len(rows)))
        estimates.append(rows[indices].mean(axis=(1, 2)))
        remaining -= count
    lower_tail = 0.5 * (1.0 - float(confidence))
    return float(np.quantile(np.concatenate(estimates), lower_tail))


def reacher_arm_mass_prediction_metrics(
    *,
    pair_ids: tuple[str, ...],
    predicted_lighter: np.ndarray,
    predicted_heavier: np.ndarray,
    target_lighter: np.ndarray,
    target_heavier: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    lighter_lighter = _mse(predicted_lighter, target_lighter)
    lighter_heavier = _mse(predicted_lighter, target_heavier)
    heavier_lighter = _mse(predicted_heavier, target_lighter)
    heavier_heavier = _mse(predicted_heavier, target_heavier)
    lighter_future = lighter_lighter < lighter_heavier
    heavier_future = heavier_heavier < heavier_lighter
    lighter_history = lighter_lighter < heavier_lighter
    heavier_history = heavier_heavier < lighter_heavier
    switch = np.sum(
        (predicted_heavier - predicted_lighter)
        * (target_heavier - target_lighter),
        axis=-1,
    ) > 0
    correct_future = np.concatenate([lighter_future, heavier_future])
    correct_history = np.concatenate([lighter_history, heavier_history])
    correct_losses = np.concatenate([lighter_lighter, heavier_heavier])
    other_losses = np.concatenate([lighter_heavier, heavier_lighter])
    metrics = {
        "pair_count": len(pair_ids),
        "decision_count": 2 * len(pair_ids),
        "correct_future_rate": float(correct_future.mean()),
        "correct_history_rate": float(correct_history.mean()),
        "context_switch_rate": float(switch.mean()),
        "lighter_correct_future_rate": float(lighter_future.mean()),
        "heavier_correct_future_rate": float(heavier_future.mean()),
        "worst_mass_correct_future_rate": float(
            min(lighter_future.mean(), heavier_future.mean())
        ),
        "correct_future_mse_mean": float(correct_losses.mean()),
        "other_future_mse_mean": float(other_losses.mean()),
        "other_minus_correct_mse_margin_mean": float(
            (other_losses - correct_losses).mean()
        ),
        "current_frame_only_accuracy_bound": 0.5,
        "paired_bootstrap_95_lower_bound": {
            "correct_future_rate": _paired_bootstrap_lower_bound(
                np.stack([lighter_future, heavier_future], axis=1)
            ),
            "correct_history_rate": _paired_bootstrap_lower_bound(
                np.stack([lighter_history, heavier_history], axis=1)
            ),
            "context_switch_rate": _paired_bootstrap_lower_bound(switch),
        },
    }
    latent_response, latent_response_records = (
        paired_latent_response_metrics(
            pair_ids=pair_ids,
            predicted_first=predicted_lighter,
            predicted_second=predicted_heavier,
            target_first=target_lighter,
            target_second=target_heavier,
        )
    )
    metrics["latent_response"] = latent_response
    calibrated_response = np.asarray(
        [
            row["calibrated_response_success"]
            for row in latent_response_records
        ],
        dtype=bool,
    )
    joint_icl_pair_success = (
        lighter_future
        & heavier_future
        & lighter_history
        & heavier_history
        & calibrated_response
    )
    metrics["joint_icl_pair_success_rate"] = float(
        joint_icl_pair_success.mean()
    )
    records = [
        {
            "pair_id": pair_id,
            "lighter": {
                "correct_future_mse": float(lighter_lighter[index]),
                "other_future_mse": float(lighter_heavier[index]),
                "correct_future": bool(lighter_future[index]),
                "correct_history": bool(lighter_history[index]),
            },
            "heavier": {
                "correct_future_mse": float(heavier_heavier[index]),
                "other_future_mse": float(heavier_lighter[index]),
                "correct_future": bool(heavier_future[index]),
                "correct_history": bool(heavier_history[index]),
            },
            "context_switch_correct": bool(switch[index]),
            "joint_icl_pair_success": bool(
                joint_icl_pair_success[index]
            ),
            "latent_response": {
                name: value
                for name, value in latent_response_records[index].items()
                if name != "pair_id"
            },
        }
        for index, pair_id in enumerate(pair_ids)
    ]
    return metrics, records


def reacher_arm_mass_prediction_gate(
    metrics: dict[str, Any], *, release: dict[str, Any]
) -> dict[str, Any]:
    thresholds = release["scoring"]["hidden_future_prediction"]["gates"]
    checks = {
        name: metrics[name] >= float(thresholds[f"{name}_minimum"])
        for name in (
            "correct_future_rate",
            "correct_history_rate",
            "context_switch_rate",
            "worst_mass_correct_future_rate",
        )
    }
    checks.update(
        paired_latent_response_gate_checks(
            metrics, thresholds=thresholds
        )
    )
    uncertainty = release["scoring"]["hidden_future_prediction"].get(
        "uncertainty", {}
    )
    lower_minimum = uncertainty.get("lower_bound_minimum", {})
    observed_lower = metrics.get("paired_bootstrap_95_lower_bound", {})
    checks.update(
        {
            f"{name}_bootstrap_lower_bound": (
                float(observed_lower[name]) >= float(minimum)
            )
            for name, minimum in lower_minimum.items()
        }
    )
    return {"checks": checks, "passed": all(checks.values())}


def evaluate_reacher_arm_mass_icl_model(
    *,
    adapter: ReacherArmMassICLModelAdapter,
    model_name: str,
    training_recipe: str,
    training_seed: int | None,
    release_config: Path | str = DEFAULT_REACHER_ARM_MASS_RELEASE_CONFIG,
    repo_root: Path | None = None,
    batch_size: int = 64,
    include_records: bool = True,
) -> dict[str, Any]:
    root = (repo_root or repository_root()).resolve()
    release = load_reacher_arm_mass_icl_release(release_config)
    dataset = ReacherArmMassICLEvalDataset(release=release, repo_root=root)
    arrays = dataset.arrays
    histories = np.concatenate(
        [arrays.lighter_pixels[:, :3], arrays.heavier_pixels[:, :3]]
    )
    actions = np.concatenate(
        [arrays.raw_action_blocks[:, :3], arrays.raw_action_blocks[:, :3]]
    )
    before = adapter.frozen_state_hash()
    predicted = adapter.rollout_latents(histories, actions, batch_size=batch_size)
    if predicted.ndim != 3 or predicted.shape[1] != 1:
        raise RuntimeError("Reacher Arm Mass adapter must return one future")
    true_futures = np.concatenate(
        [arrays.lighter_pixels[:, 3], arrays.heavier_pixels[:, 3]]
    )
    encoded = adapter.encode_pixels(true_futures, batch_size=batch_size)
    after = adapter.frozen_state_hash()
    if before != after:
        raise RuntimeError("Model state changed during Reacher Arm Mass scoring")
    count = arrays.pair_count
    metrics, records = reacher_arm_mass_prediction_metrics(
        pair_ids=arrays.pair_ids,
        predicted_lighter=predicted[:count, 0],
        predicted_heavier=predicted[count:, 0],
        target_lighter=encoded[:count],
        target_heavier=encoded[count:],
    )
    release_path = Path(release["_config_path"])
    result = {
        "schema_version": 1,
        "benchmark": "reacher_history3_arm_mass_icl_v1",
        "submission_kind": "single_checkpoint",
        "status": "completed",
        "release": {
            "release_id": release["release_id"],
            "release_config_sha256": file_sha256(release_path),
            "data_manifest_sha256": release["data"]["manifest_sha256"],
        },
        "model": {
            "name": str(model_name),
            "training_recipe": str(training_recipe),
            "training_seed": training_seed,
            "adapter": adapter.metadata,
            "state_sha256_before": before,
            "state_sha256_after": after,
        },
        "data": dataset.describe(),
        "metrics": metrics,
        "gate": reacher_arm_mass_prediction_gate(metrics, release=release),
    }
    if include_records:
        result["records"] = records
    return result


def score_reacher_arm_mass_icl_results(
    *,
    result_paths: Iterable[Path | str],
    method_name: str,
    release_config: Path | str = DEFAULT_REACHER_ARM_MASS_RELEASE_CONFIG,
) -> dict[str, Any]:
    release = load_reacher_arm_mass_icl_release(release_config)
    results = [json.loads(Path(path).read_text()) for path in result_paths]
    if any(
        "latent_response" not in row.get("metrics", {}) for row in results
    ):
        raise ValueError(
            "Legacy Reacher Arm Mass results must be rescored from their "
            "checkpoints before a formal method claim"
        )
    required = int(release["scoring"]["method_level"]["training_seeds_required"])
    seeds = [row["model"]["training_seed"] for row in results]
    if len(results) != required or len(set(seeds)) != required:
        raise ValueError(f"Method scoring requires {required} distinct seeds")
    names = (
        "correct_future_rate",
        "correct_history_rate",
        "context_switch_rate",
        "worst_mass_correct_future_rate",
        "other_minus_correct_mse_margin_mean",
    )
    if all(
        "joint_icl_pair_success_rate" in row["metrics"]
        for row in results
    ):
        names += ("joint_icl_pair_success_rate",)
    return {
        "schema_version": 1,
        "benchmark": "reacher_history3_arm_mass_icl_v1",
        "submission_kind": "three_seed_method",
        "status": "completed",
        "method_name": str(method_name),
        "training_seeds": sorted(seeds),
        "checkpoint_results": results,
        "aggregate": {
            name: {
                "mean": float(
                    statistics.mean(row["metrics"][name] for row in results)
                ),
                "minimum": float(min(row["metrics"][name] for row in results)),
                "maximum": float(max(row["metrics"][name] for row in results)),
            }
            for name in names
        },
        "passed": all(row["gate"]["passed"] for row in results),
    }


__all__ = [
    "evaluate_reacher_arm_mass_icl_model",
    "reacher_arm_mass_prediction_gate",
    "reacher_arm_mass_prediction_metrics",
    "score_reacher_arm_mass_icl_results",
]
