from __future__ import annotations

import json
from pathlib import Path
import statistics
from typing import Any, Iterable

import numpy as np

from contextworld.benchmarks.adapters import (
    PortalExitICLModelAdapter,
    validate_adapter_protocol,
)
from contextworld.benchmarks.portal_exit_icl_data import (
    DEFAULT_PORTAL_EXIT_RELEASE_CONFIG,
    PortalExitICLEvalDataset,
    file_sha256,
    load_portal_exit_icl_release,
)
from contextworld.benchmarks.paired_latent_response import (
    paired_latent_response_gate_checks,
    paired_latent_response_metrics,
)
from contextworld.paths import repository_root


def _mse(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.square(left - right).mean(axis=-1)


def portal_exit_prediction_metrics(
    *,
    pair_ids: tuple[str, ...],
    predicted_near: np.ndarray,
    predicted_farther: np.ndarray,
    target_near: np.ndarray,
    target_farther: np.ndarray,
    bootstrap_resamples: int = 10_000,
    bootstrap_seed: int = 2026080205,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    near_near = _mse(predicted_near, target_near)
    near_farther = _mse(predicted_near, target_farther)
    farther_near = _mse(predicted_farther, target_near)
    farther_farther = _mse(predicted_farther, target_farther)
    near_future = near_near < near_farther
    farther_future = farther_farther < farther_near
    near_history = near_near < farther_near
    farther_history = farther_farther < near_farther
    switch = np.sum(
        (predicted_farther - predicted_near) * (target_farther - target_near),
        axis=-1,
    ) > 0
    correct_future = np.concatenate([near_future, farther_future])
    correct_history = np.concatenate([near_history, farther_history])
    correct_losses = np.concatenate([near_near, farther_farther])
    other_losses = np.concatenate([near_farther, farther_near])
    metrics = {
        "pair_count": len(pair_ids),
        "decision_count": 2 * len(pair_ids),
        "correct_future_rate": float(correct_future.mean()),
        "correct_history_rate": float(correct_history.mean()),
        "context_switch_rate": float(switch.mean()),
        "near_border_correct_future_rate": float(near_future.mean()),
        "farther_from_border_correct_future_rate": float(farther_future.mean()),
        "worst_exit_correct_future_rate": float(
            min(near_future.mean(), farther_future.mean())
        ),
        "correct_future_mse_mean": float(correct_losses.mean()),
        "other_future_mse_mean": float(other_losses.mean()),
        "other_minus_correct_mse_margin_mean": float(
            (other_losses - correct_losses).mean()
        ),
        "current_frame_only_accuracy_bound": 0.5,
    }
    if bootstrap_resamples <= 0:
        raise ValueError("bootstrap_resamples must be positive")
    rng = np.random.default_rng(bootstrap_seed)
    draws = rng.integers(
        0, len(pair_ids), size=(bootstrap_resamples, len(pair_ids))
    )
    near_future_draws = near_future[draws].mean(axis=1)
    farther_future_draws = farther_future[draws].mean(axis=1)
    near_history_draws = near_history[draws].mean(axis=1)
    farther_history_draws = farther_history[draws].mean(axis=1)
    bootstrap = {
        "correct_future_rate": 0.5
        * (near_future_draws + farther_future_draws),
        "correct_history_rate": 0.5
        * (near_history_draws + farther_history_draws),
        "context_switch_rate": switch[draws].mean(axis=1),
        "worst_exit_correct_future_rate": np.minimum(
            near_future_draws, farther_future_draws
        ),
    }
    metrics["uncertainty"] = {
        "method": "paired_query_bootstrap",
        "unit": "portal_exit_query_pair",
        "resamples": int(bootstrap_resamples),
        "confidence_level": 0.95,
        "random_seed": int(bootstrap_seed),
        "lower_bounds": {
            name: float(np.quantile(values, 0.025))
            for name, values in bootstrap.items()
        },
    }
    latent_response, latent_response_records = (
        paired_latent_response_metrics(
            pair_ids=pair_ids,
            predicted_first=predicted_near,
            predicted_second=predicted_farther,
            target_first=target_near,
            target_second=target_farther,
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
        near_future
        & farther_future
        & near_history
        & farther_history
        & calibrated_response
    )
    metrics["joint_icl_pair_success_rate"] = float(
        joint_icl_pair_success.mean()
    )
    records = [
        {
            "pair_id": pair_id,
            "near_border": {
                "correct_future_mse": float(near_near[index]),
                "other_future_mse": float(near_farther[index]),
                "correct_future": bool(near_future[index]),
                "correct_history": bool(near_history[index]),
            },
            "farther_from_border": {
                "correct_future_mse": float(farther_farther[index]),
                "other_future_mse": float(farther_near[index]),
                "correct_future": bool(farther_future[index]),
                "correct_history": bool(farther_history[index]),
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


def portal_exit_prediction_gate(
    metrics: dict[str, Any], *, release: dict[str, Any]
) -> dict[str, Any]:
    thresholds = release["scoring"]["hidden_future_prediction"]["gates"]
    checks = {
        name: metrics[name] >= float(thresholds[f"{name}_minimum"])
        for name in (
            "correct_future_rate",
            "correct_history_rate",
            "context_switch_rate",
            "worst_exit_correct_future_rate",
        )
    }
    checks.update(
        paired_latent_response_gate_checks(
            metrics, thresholds=thresholds
        )
    )
    lower_minimum = thresholds.get("bootstrap_lower_bound_minimum", {})
    lower_bounds = metrics.get("uncertainty", {}).get("lower_bounds", {})
    uncertainty_checks = {
        name: float(lower_bounds.get(name, float("-inf"))) >= float(minimum)
        for name, minimum in lower_minimum.items()
    }
    return {
        "checks": checks,
        "uncertainty_checks": uncertainty_checks,
        "passed": all(checks.values()) and all(uncertainty_checks.values()),
    }


def evaluate_portal_exit_icl_model(
    *,
    adapter: PortalExitICLModelAdapter,
    model_name: str,
    training_recipe: str,
    training_seed: int | None,
    release_config: Path | str = DEFAULT_PORTAL_EXIT_RELEASE_CONFIG,
    repo_root: Path | None = None,
    batch_size: int = 64,
    include_records: bool = True,
) -> dict[str, Any]:
    root = (repo_root or repository_root()).resolve()
    release = load_portal_exit_icl_release(release_config)
    validate_adapter_protocol(
        adapter,
        history_tokens=3,
        action_block_raw_steps=5,
        action_dim=2,
        minimum_future_action_blocks=1,
        task_name="Portal Exit v1",
    )
    dataset = PortalExitICLEvalDataset(release=release, repo_root=root)
    arrays = dataset.arrays
    histories = np.concatenate(
        [arrays.near_border_pixels[:, :3], arrays.farther_from_border_pixels[:, :3]]
    )
    actions = np.concatenate(
        [arrays.raw_action_blocks[:, :3], arrays.raw_action_blocks[:, :3]]
    )
    before = adapter.frozen_state_hash()
    predicted = adapter.rollout_latents(histories, actions, batch_size=batch_size)
    if predicted.ndim != 3 or predicted.shape[1] != 1:
        raise RuntimeError("Portal Exit adapter must return one future")
    true_futures = np.concatenate(
        [arrays.near_border_pixels[:, 3], arrays.farther_from_border_pixels[:, 3]]
    )
    encoded = adapter.encode_pixels(true_futures, batch_size=batch_size)
    after = adapter.frozen_state_hash()
    if before != after:
        raise RuntimeError("Model state changed during Portal Exit scoring")
    count = arrays.pair_count
    metrics, records = portal_exit_prediction_metrics(
        pair_ids=arrays.pair_ids,
        predicted_near=predicted[:count, 0],
        predicted_farther=predicted[count:, 0],
        target_near=encoded[:count],
        target_farther=encoded[count:],
    )
    release_path = Path(release["_config_path"])
    result = {
        "schema_version": 1,
        "benchmark": "tworoom_history3_portal_exit_icl_v1",
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
        "gate": portal_exit_prediction_gate(metrics, release=release),
    }
    if include_records:
        result["records"] = records
    return result


def score_portal_exit_icl_results(
    *,
    result_paths: Iterable[Path | str],
    method_name: str,
    release_config: Path | str = DEFAULT_PORTAL_EXIT_RELEASE_CONFIG,
) -> dict[str, Any]:
    release = load_portal_exit_icl_release(release_config)
    results = [json.loads(Path(path).read_text()) for path in result_paths]
    if any(
        "latent_response" not in row.get("metrics", {}) for row in results
    ):
        raise ValueError(
            "Legacy Portal Exit results must be rescored from their "
            "checkpoints before a formal method claim"
        )
    required = int(release["scoring"]["method_level"]["training_seeds_required"])
    seeds = [row["model"]["training_seed"] for row in results]
    if len(results) != required or len(set(seeds)) != required:
        raise ValueError(f"Method scoring requires {required} distinct seeds")
    names = (
        "correct_future_rate", "correct_history_rate", "context_switch_rate",
        "worst_exit_correct_future_rate", "other_minus_correct_mse_margin_mean",
    )
    if all(
        "joint_icl_pair_success_rate" in row["metrics"]
        for row in results
    ):
        names += ("joint_icl_pair_success_rate",)
    return {
        "schema_version": 1,
        "benchmark": "tworoom_history3_portal_exit_icl_v1",
        "submission_kind": "three_seed_method",
        "status": "completed",
        "method_name": str(method_name),
        "training_seeds": sorted(seeds),
        "checkpoint_results": results,
        "aggregate": {
            name: {
                "mean": float(statistics.mean(row["metrics"][name] for row in results)),
                "minimum": float(min(row["metrics"][name] for row in results)),
                "maximum": float(max(row["metrics"][name] for row in results)),
            }
            for name in names
        },
        "passed": all(row["gate"]["passed"] for row in results),
    }


__all__ = [
    "evaluate_portal_exit_icl_model",
    "portal_exit_prediction_gate",
    "portal_exit_prediction_metrics",
    "score_portal_exit_icl_results",
]
