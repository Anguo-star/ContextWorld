from __future__ import annotations

import json
from pathlib import Path
import statistics
from typing import Any, Iterable

import numpy as np

from contextworld.benchmarks.adapters import (
    CubeGraspRuleICLModelAdapter,
    validate_adapter_protocol,
)
from contextworld.benchmarks.cube_grasp_rule_icl_data import (
    DEFAULT_CUBE_GRASP_RULE_RELEASE_CONFIG,
    CubeGraspRuleICLEvalDataset,
    file_sha256,
    load_cube_grasp_rule_icl_release,
)
from contextworld.benchmarks.paired_latent_response import (
    paired_latent_response_gate_checks,
    paired_latent_response_metrics,
)
from contextworld.paths import repository_root


def _mse(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.square(left - right).mean(axis=-1)


def _validated_metric_inputs(
    *,
    pair_ids: tuple[str, ...],
    predicted_cannot_hold: np.ndarray,
    predicted_can_hold: np.ndarray,
    target_cannot_hold: np.ndarray,
    target_can_hold: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not pair_ids or len(set(pair_ids)) != len(pair_ids):
        raise ValueError("Cube pair_ids must be non-empty and unique")
    arrays = tuple(
        np.asarray(value)
        for value in (
            predicted_cannot_hold,
            predicted_can_hold,
            target_cannot_hold,
            target_can_hold,
        )
    )
    expected_shape = arrays[0].shape
    if len(expected_shape) != 2 or expected_shape[0] != len(pair_ids):
        raise ValueError(
            "Cube latent arrays must have shape (pair_count, latent_dim)"
        )
    if any(value.shape != expected_shape for value in arrays[1:]):
        raise ValueError("Cube predicted and target latent shapes must match")
    if not all(np.isfinite(value).all() for value in arrays):
        raise ValueError("Cube predicted and target latents must be finite")
    return arrays


def _paired_bootstrap_lower_bound(
    values: np.ndarray,
    *,
    resamples: int = 10_000,
    seed: int = 2026080314,
    confidence: float = 0.95,
) -> float:
    rows = np.asarray(values, dtype=np.float64)
    if rows.ndim == 1:
        rows = rows[:, None]
    rng = np.random.default_rng(seed)
    estimates = []
    remaining = int(resamples)
    while remaining:
        count = min(1_000, remaining)
        indices = rng.integers(0, len(rows), size=(count, len(rows)))
        estimates.append(rows[indices].mean(axis=(1, 2)))
        remaining -= count
    return float(
        np.quantile(np.concatenate(estimates), 0.5 * (1.0 - confidence))
    )


def cube_grasp_rule_prediction_metrics(
    *,
    pair_ids: tuple[str, ...],
    predicted_cannot_hold: np.ndarray,
    predicted_can_hold: np.ndarray,
    target_cannot_hold: np.ndarray,
    target_can_hold: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    (
        predicted_cannot_hold,
        predicted_can_hold,
        target_cannot_hold,
        target_can_hold,
    ) = _validated_metric_inputs(
        pair_ids=pair_ids,
        predicted_cannot_hold=predicted_cannot_hold,
        predicted_can_hold=predicted_can_hold,
        target_cannot_hold=target_cannot_hold,
        target_can_hold=target_can_hold,
    )
    off_off = _mse(predicted_cannot_hold, target_cannot_hold)
    off_on = _mse(predicted_cannot_hold, target_can_hold)
    on_off = _mse(predicted_can_hold, target_cannot_hold)
    on_on = _mse(predicted_can_hold, target_can_hold)
    off_future = off_off < off_on
    on_future = on_on < on_off
    off_history = off_off < on_off
    on_history = on_on < off_on
    switch = np.sum(
        (predicted_can_hold - predicted_cannot_hold)
        * (target_can_hold - target_cannot_hold),
        axis=-1,
    ) > 0
    correct_future = np.concatenate([off_future, on_future])
    correct_history = np.concatenate([off_history, on_history])
    correct_losses = np.concatenate([off_off, on_on])
    other_losses = np.concatenate([off_on, on_off])
    metrics = {
        "pair_count": len(pair_ids),
        "decision_count": 2 * len(pair_ids),
        "correct_future_rate": float(correct_future.mean()),
        "correct_history_rate": float(correct_history.mean()),
        "context_switch_rate": float(switch.mean()),
        "cannot_hold_correct_future_rate": float(off_future.mean()),
        "can_hold_correct_future_rate": float(on_future.mean()),
        "worst_rule_correct_future_rate": float(
            min(off_future.mean(), on_future.mean())
        ),
        "correct_future_mse_mean": float(correct_losses.mean()),
        "other_future_mse_mean": float(other_losses.mean()),
        "other_minus_correct_mse_margin_mean": float(
            (other_losses - correct_losses).mean()
        ),
        "current_frame_only_accuracy_bound": 0.5,
        "paired_bootstrap_95_lower_bound": {
            "correct_future_rate": _paired_bootstrap_lower_bound(
                np.stack([off_future, on_future], axis=1)
            ),
            "correct_history_rate": _paired_bootstrap_lower_bound(
                np.stack([off_history, on_history], axis=1)
            ),
            "context_switch_rate": _paired_bootstrap_lower_bound(switch),
        },
    }
    latent_response, latent_response_records = paired_latent_response_metrics(
        pair_ids=pair_ids,
        predicted_first=predicted_cannot_hold,
        predicted_second=predicted_can_hold,
        target_first=target_cannot_hold,
        target_second=target_can_hold,
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
        off_future
        & on_future
        & off_history
        & on_history
        & calibrated_response
    )
    metrics["joint_icl_pair_success_rate"] = float(
        joint_icl_pair_success.mean()
    )
    records = [
        {
            "pair_id": pair_id,
            "cannot_hold": {
                "correct_future_mse": float(off_off[index]),
                "other_future_mse": float(off_on[index]),
                "correct_future": bool(off_future[index]),
                "correct_history": bool(off_history[index]),
            },
            "can_hold": {
                "correct_future_mse": float(on_on[index]),
                "other_future_mse": float(on_off[index]),
                "correct_future": bool(on_future[index]),
                "correct_history": bool(on_history[index]),
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


def cube_grasp_rule_prediction_gate(
    metrics: dict[str, Any], *, release: dict[str, Any]
) -> dict[str, Any]:
    thresholds = release["scoring"]["hidden_future_prediction"]["gates"]
    names = (
        "correct_future_rate",
        "correct_history_rate",
        "context_switch_rate",
        "worst_rule_correct_future_rate",
    )
    checks = {
        name: metrics[name] >= float(thresholds[f"{name}_minimum"])
        for name in names
    }
    checks.update(
        paired_latent_response_gate_checks(
            metrics, thresholds=thresholds
        )
    )
    lower_minimum = release["scoring"]["hidden_future_prediction"].get(
        "uncertainty", {}
    ).get("lower_bound_minimum", {})
    observed = metrics["paired_bootstrap_95_lower_bound"]
    checks.update(
        {
            f"{name}_bootstrap_lower_bound": float(observed[name])
            >= float(minimum)
            for name, minimum in lower_minimum.items()
        }
    )
    return {"checks": checks, "passed": all(checks.values())}


def _validate_cube_adapter_protocol(
    adapter: CubeGraspRuleICLModelAdapter,
) -> None:
    """Reject incompatible adapters before any Public Test data is opened."""

    message = (
        "Cube Gripper-Carry v1 requires a History=3 latent adapter "
        "with 5x5 raw-action blocks and at least one future block; "
        f"got {getattr(adapter, 'protocol', None)}"
    )
    try:
        validate_adapter_protocol(
            adapter,
            history_tokens=3,
            action_block_raw_steps=5,
            action_dim=5,
            minimum_future_action_blocks=1,
            task_name="Cube Gripper-Carry v1",
        )
    except ValueError as exc:
        raise ValueError(
            message
        ) from exc


def evaluate_cube_grasp_rule_icl_model(
    *,
    adapter: CubeGraspRuleICLModelAdapter,
    model_name: str,
    training_recipe: str,
    training_seed: int | None,
    release_config: Path | str = DEFAULT_CUBE_GRASP_RULE_RELEASE_CONFIG,
    repo_root: Path | None = None,
    batch_size: int = 64,
    include_records: bool = True,
) -> dict[str, Any]:
    root = (repo_root or repository_root()).resolve()
    release = load_cube_grasp_rule_icl_release(release_config)
    _validate_cube_adapter_protocol(adapter)
    dataset = CubeGraspRuleICLEvalDataset(release=release, repo_root=root)
    arrays = dataset.arrays
    histories = np.concatenate(
        [arrays.cannot_hold_pixels[:, :3], arrays.can_hold_pixels[:, :3]]
    )
    actions = np.concatenate(
        [arrays.raw_action_blocks[:, :3], arrays.raw_action_blocks[:, :3]]
    )
    before = adapter.frozen_state_hash()
    predicted = adapter.rollout_latents(histories, actions, batch_size=batch_size)
    count = arrays.pair_count
    if (
        predicted.ndim != 3
        or predicted.shape[0] != 2 * count
        or predicted.shape[1] != 1
        or not np.isfinite(predicted).all()
    ):
        raise RuntimeError(
            "Cube Gripper-Carry adapter must return finite latents with "
            "shape (2 * pair_count, 1, latent_dim)"
        )
    true_futures = np.concatenate(
        [arrays.cannot_hold_pixels[:, 3], arrays.can_hold_pixels[:, 3]]
    )
    encoded = adapter.encode_pixels(true_futures, batch_size=batch_size)
    if (
        encoded.shape != (2 * count, predicted.shape[2])
        or not np.isfinite(encoded).all()
    ):
        raise RuntimeError(
            "Cube encoded targets must match the predicted latent shape"
        )
    after = adapter.frozen_state_hash()
    if before != after:
        raise RuntimeError("Model state changed during Cube scoring")
    metrics, records = cube_grasp_rule_prediction_metrics(
        pair_ids=arrays.pair_ids,
        predicted_cannot_hold=predicted[:count, 0],
        predicted_can_hold=predicted[count:, 0],
        target_cannot_hold=encoded[:count],
        target_can_hold=encoded[count:],
    )
    release_path = Path(release["_config_path"])
    result = {
        "schema_version": 1,
        "benchmark": "cube_history3_gripper_carry_icl_v1",
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
        "gate": cube_grasp_rule_prediction_gate(metrics, release=release),
    }
    if include_records:
        result["records"] = records
    return result


def score_cube_grasp_rule_icl_results(
    *,
    result_paths: Iterable[Path | str],
    method_name: str,
    release_config: Path | str = DEFAULT_CUBE_GRASP_RULE_RELEASE_CONFIG,
) -> dict[str, Any]:
    release = load_cube_grasp_rule_icl_release(release_config)
    release_path = Path(release["_config_path"])
    expected_release = {
        "release_id": release["release_id"],
        "release_config_sha256": file_sha256(release_path),
        "data_manifest_sha256": release["data"]["manifest_sha256"],
    }
    results = []
    for value in result_paths:
        path = Path(value)
        row = json.loads(path.read_text(encoding="utf-8"))
        if (
            row.get("schema_version") != 1
            or row.get("benchmark")
            != "cube_history3_gripper_carry_icl_v1"
            or row.get("submission_kind") != "single_checkpoint"
            or row.get("status") != "completed"
        ):
            raise ValueError(f"Unsupported Cube result: {path}")
        if row.get("release") != expected_release:
            raise RuntimeError(f"Cube release identity mismatch: {path}")
        seed = row.get("model", {}).get("training_seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError(f"Cube result has no integer training seed: {path}")
        if not isinstance(row.get("gate", {}).get("passed"), bool):
            raise ValueError(f"Cube result has no completed gate: {path}")
        results.append(row)
    if any(
        "latent_response" not in row.get("metrics", {}) for row in results
    ):
        raise ValueError(
            "Legacy Cube results must be rescored from their checkpoints "
            "before a formal method claim"
        )
    required = int(release["scoring"]["method_level"]["training_seeds_required"])
    seeds = [row["model"]["training_seed"] for row in results]
    if len(results) != required or len(set(seeds)) != required:
        raise ValueError(f"Method scoring requires {required} distinct seeds")
    names = (
        "correct_future_rate",
        "correct_history_rate",
        "context_switch_rate",
        "worst_rule_correct_future_rate",
        "other_minus_correct_mse_margin_mean",
        "joint_icl_pair_success_rate",
    )
    return {
        "schema_version": 1,
        "benchmark": "cube_history3_gripper_carry_icl_v1",
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
    "cube_grasp_rule_prediction_gate",
    "cube_grasp_rule_prediction_metrics",
    "evaluate_cube_grasp_rule_icl_model",
    "score_cube_grasp_rule_icl_results",
]
