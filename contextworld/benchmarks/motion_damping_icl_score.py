from __future__ import annotations

import json
from pathlib import Path
import statistics
from typing import Any, Iterable

import numpy as np

from contextworld.benchmarks.adapters import MotionDampingICLModelAdapter
from contextworld.benchmarks.motion_damping_icl_data import (
    DEFAULT_MOTION_DAMPING_RELEASE_CONFIG,
    MotionDampingICLEvalDataset,
    audit_motion_damping_icl_release,
    file_sha256,
    load_motion_damping_icl_release,
)
from contextworld.benchmarks.paired_latent_response import (
    paired_latent_response_gate_checks,
    paired_latent_response_metrics,
)
from contextworld.paths import repository_root


def _mse(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.square(left - right).mean(axis=-1)


def motion_damping_prediction_metrics(
    *,
    pair_ids: tuple[str, ...],
    predicted_faster_decay: np.ndarray,
    predicted_no_extra_decay: np.ndarray,
    target_faster_decay: np.ndarray,
    target_no_extra_decay: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Score predictions only against real checkpoint-native future latents."""

    fast_fast = _mse(predicted_faster_decay, target_faster_decay)
    fast_no_extra = _mse(predicted_faster_decay, target_no_extra_decay)
    no_extra_fast = _mse(predicted_no_extra_decay, target_faster_decay)
    no_extra_no_extra = _mse(
        predicted_no_extra_decay, target_no_extra_decay
    )
    fast_future = fast_fast < fast_no_extra
    no_extra_future = no_extra_no_extra < no_extra_fast
    fast_history = fast_fast < no_extra_fast
    no_extra_history = no_extra_no_extra < fast_no_extra
    switch = np.sum(
        (predicted_no_extra_decay - predicted_faster_decay)
        * (target_no_extra_decay - target_faster_decay),
        axis=-1,
    ) > 0
    correct_future = np.concatenate([fast_future, no_extra_future])
    correct_history = np.concatenate([fast_history, no_extra_history])
    correct_losses = np.concatenate([fast_fast, no_extra_no_extra])
    other_losses = np.concatenate([fast_no_extra, no_extra_fast])
    metrics = {
        "pair_count": len(pair_ids),
        "decision_count": 2 * len(pair_ids),
        "correct_future_rate": float(correct_future.mean()),
        "correct_history_rate": float(correct_history.mean()),
        "context_switch_rate": float(switch.mean()),
        "faster_decay_correct_future_rate": float(fast_future.mean()),
        "no_extra_decay_correct_future_rate": float(no_extra_future.mean()),
        "worst_damping_correct_future_rate": float(
            min(fast_future.mean(), no_extra_future.mean())
        ),
        "correct_future_mse_mean": float(correct_losses.mean()),
        "other_future_mse_mean": float(other_losses.mean()),
        "other_minus_correct_mse_margin_mean": float(
            (other_losses - correct_losses).mean()
        ),
        "current_frame_only_accuracy_bound": 0.5,
    }
    latent_response, latent_response_records = (
        paired_latent_response_metrics(
            pair_ids=pair_ids,
            predicted_first=predicted_faster_decay,
            predicted_second=predicted_no_extra_decay,
            target_first=target_faster_decay,
            target_second=target_no_extra_decay,
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
        fast_future
        & no_extra_future
        & fast_history
        & no_extra_history
        & calibrated_response
    )
    metrics["joint_icl_pair_success_rate"] = float(
        joint_icl_pair_success.mean()
    )
    records = [
        {
            "pair_id": pair_id,
            "faster_decay": {
                "correct_future_mse": float(fast_fast[index]),
                "other_future_mse": float(fast_no_extra[index]),
                "correct_future": bool(fast_future[index]),
                "correct_history": bool(fast_history[index]),
            },
            "no_extra_decay": {
                "correct_future_mse": float(no_extra_no_extra[index]),
                "other_future_mse": float(no_extra_fast[index]),
                "correct_future": bool(no_extra_future[index]),
                "correct_history": bool(no_extra_history[index]),
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


def motion_damping_prediction_gate(
    metrics: dict[str, Any], *, release: dict[str, Any]
) -> dict[str, Any]:
    thresholds = release["scoring"]["hidden_future_prediction"]["gates"]
    checks = {
        "correct_future_rate": metrics["correct_future_rate"]
        >= float(thresholds["correct_future_rate_minimum"]),
        "correct_history_rate": metrics["correct_history_rate"]
        >= float(thresholds["correct_history_rate_minimum"]),
        "context_switch_rate": metrics["context_switch_rate"]
        >= float(thresholds["context_switch_rate_minimum"]),
        "worst_damping_correct_future_rate": metrics[
            "worst_damping_correct_future_rate"
        ]
        >= float(thresholds["worst_damping_correct_future_rate_minimum"]),
    }
    checks.update(
        paired_latent_response_gate_checks(
            metrics, thresholds=thresholds
        )
    )
    return {"checks": checks, "passed": all(checks.values())}


def evaluate_motion_damping_icl_model(
    *,
    adapter: MotionDampingICLModelAdapter,
    model_name: str,
    training_recipe: str,
    training_seed: int | None,
    release_config: Path | str = DEFAULT_MOTION_DAMPING_RELEASE_CONFIG,
    repo_root: Path | None = None,
    batch_size: int = 64,
    include_records: bool = True,
) -> dict[str, Any]:
    root = (repo_root or repository_root()).resolve()
    audit = audit_motion_damping_icl_release(
        release_config=release_config,
        repo_root=root,
        full=False,
    )
    if not audit["passed"]:
        raise RuntimeError(
            "Motion-damping strict causal release audit failed before scoring"
        )
    release = load_motion_damping_icl_release(release_config)
    dataset = MotionDampingICLEvalDataset(release=release, repo_root=root)
    arrays = dataset.arrays
    if not dataset.is_full_protocol:
        raise RuntimeError("Formal Motion Damping scoring requires 256 pairs")
    if adapter.protocol.history_tokens != 3:
        raise ValueError("Motion Damping requires History=3")
    histories = np.concatenate(
        [
            arrays.faster_decay_pixels[:, :3],
            arrays.no_extra_decay_pixels[:, :3],
        ],
        axis=0,
    )
    actions = np.concatenate(
        [arrays.raw_action_blocks[:, :3], arrays.raw_action_blocks[:, :3]],
        axis=0,
    )
    before = adapter.frozen_state_hash()
    predicted = adapter.rollout_latents(histories, actions, batch_size=batch_size)
    if predicted.ndim != 3 or predicted.shape[1] != 1:
        raise RuntimeError("Motion Damping adapter must return one future")
    true_futures = np.concatenate(
        [
            arrays.faster_decay_pixels[:, 3],
            arrays.no_extra_decay_pixels[:, 3],
        ],
        axis=0,
    )
    encoded = adapter.encode_pixels(true_futures, batch_size=batch_size)
    after = adapter.frozen_state_hash()
    if before != after:
        raise RuntimeError("Model state changed during Motion Damping scoring")
    count = arrays.pair_count
    metrics, records = motion_damping_prediction_metrics(
        pair_ids=arrays.pair_ids,
        predicted_faster_decay=predicted[:count, 0],
        predicted_no_extra_decay=predicted[count:, 0],
        target_faster_decay=encoded[:count],
        target_no_extra_decay=encoded[count:],
    )
    release_path = Path(release["_config_path"])
    result = {
        "schema_version": 1,
        "benchmark": "pusht_history3_motion_damping_icl",
        "submission_kind": "single_checkpoint",
        "status": "completed",
        "release": {
            "release_id": release["release_id"],
            "release_config_sha256": file_sha256(release_path),
            "data_manifest_sha256": release["data"]["manifest_sha256"],
            "sealed_test_included": False,
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
        "gate": motion_damping_prediction_gate(metrics, release=release),
    }
    if include_records:
        result["records"] = records
    return result


def score_motion_damping_icl_results(
    *,
    result_paths: Iterable[Path | str],
    method_name: str,
    release_config: Path | str = DEFAULT_MOTION_DAMPING_RELEASE_CONFIG,
) -> dict[str, Any]:
    """Aggregate exactly three distinct training seeds for one method."""

    release = load_motion_damping_icl_release(release_config)
    results = [json.loads(Path(path).read_text()) for path in result_paths]
    if any(
        "latent_response" not in row.get("metrics", {}) for row in results
    ):
        raise ValueError(
            "Legacy Motion Damping results must be rescored from their "
            "checkpoints before a formal method claim"
        )
    required = int(release["scoring"]["method_level"]["training_seeds_required"])
    seeds = [row["model"]["training_seed"] for row in results]
    if len(results) != required or len(set(seeds)) != required:
        raise ValueError(f"Method scoring requires {required} distinct seeds")
    if any(row["release"]["release_id"] != release["release_id"] for row in results):
        raise RuntimeError("Motion Damping result release mismatch")
    metric_names = (
        "correct_future_rate",
        "correct_history_rate",
        "context_switch_rate",
        "worst_damping_correct_future_rate",
        "other_minus_correct_mse_margin_mean",
    )
    if all(
        "joint_icl_pair_success_rate" in row["metrics"]
        for row in results
    ):
        metric_names += ("joint_icl_pair_success_rate",)
    aggregate = {
        name: {
            "mean": float(statistics.mean(row["metrics"][name] for row in results)),
            "minimum": float(min(row["metrics"][name] for row in results)),
            "maximum": float(max(row["metrics"][name] for row in results)),
        }
        for name in metric_names
    }
    passed = all(row["gate"]["passed"] for row in results)
    return {
        "schema_version": 1,
        "benchmark": "pusht_history3_motion_damping_icl",
        "submission_kind": "three_seed_method",
        "status": "completed",
        "method_name": str(method_name),
        "training_seeds": sorted(seeds),
        "checkpoint_results": results,
        "aggregate": aggregate,
        "passed": passed,
    }


__all__ = [
    "evaluate_motion_damping_icl_model",
    "motion_damping_prediction_gate",
    "motion_damping_prediction_metrics",
    "score_motion_damping_icl_results",
]
