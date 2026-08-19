from __future__ import annotations

import json
from pathlib import Path
import statistics
from typing import Any, Iterable

import numpy as np

from contextworld.benchmarks.adapters import (
    ContactFrictionICLModelAdapter,
    validate_adapter_protocol,
)
from contextworld.benchmarks.contact_friction_icl_data import (
    DEFAULT_CONTACT_FRICTION_RELEASE_CONFIG,
    ContactFrictionICLDevelopmentDataset,
    ContactFrictionICLEvalDataset,
    contact_friction_development_data_contract,
    file_sha256,
    load_contact_friction_icl_release,
)
from contextworld.benchmarks.paired_latent_response import (
    paired_latent_response_gate_checks,
    paired_latent_response_metrics,
    paired_latent_response_summaries_close,
    summarize_paired_latent_response_records,
)
from contextworld.paths import repository_root


CONTACT_FRICTION_DEVELOPMENT_BENCHMARK = (
    "pusht_history3_contact_friction_icl_development_v1"
)


def _mse(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.square(left - right).mean(axis=-1)


def contact_friction_prediction_metrics(
    *,
    pair_ids: tuple[str, ...],
    predicted_low: np.ndarray,
    predicted_high: np.ndarray,
    target_low: np.ndarray,
    target_high: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Compare predictions only with real, checkpoint-native future latents."""

    low_to_low = _mse(predicted_low, target_low)
    low_to_high = _mse(predicted_low, target_high)
    high_to_low = _mse(predicted_high, target_low)
    high_to_high = _mse(predicted_high, target_high)

    low_future = low_to_low < low_to_high
    high_future = high_to_high < high_to_low
    low_history = low_to_low < high_to_low
    high_history = high_to_high < low_to_high
    switch = np.sum(
        (predicted_high - predicted_low) * (target_high - target_low),
        axis=-1,
    ) > 0
    correct_future = np.concatenate([low_future, high_future])
    correct_history = np.concatenate([low_history, high_history])
    correct_losses = np.concatenate([low_to_low, high_to_high])
    other_losses = np.concatenate([low_to_high, high_to_low])
    metrics = {
        "pair_count": len(pair_ids),
        "decision_count": 2 * len(pair_ids),
        "correct_future_rate": float(correct_future.mean()),
        "correct_history_rate": float(correct_history.mean()),
        "context_switch_rate": float(switch.mean()),
        "low_friction_correct_future_rate": float(low_future.mean()),
        "high_friction_correct_future_rate": float(high_future.mean()),
        "worst_friction_correct_future_rate": float(
            min(low_future.mean(), high_future.mean())
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
            predicted_first=predicted_low,
            predicted_second=predicted_high,
            target_first=target_low,
            target_second=target_high,
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
        low_future
        & high_future
        & low_history
        & high_history
        & calibrated_response
    )
    metrics["joint_icl_pair_success_rate"] = float(
        joint_icl_pair_success.mean()
    )
    records = [
        {
            "pair_id": pair_id,
            "low_friction": {
                "correct_future_mse": float(low_to_low[index]),
                "other_future_mse": float(low_to_high[index]),
                "correct_future": bool(low_future[index]),
                "correct_history": bool(low_history[index]),
            },
            "high_friction": {
                "correct_future_mse": float(high_to_high[index]),
                "other_future_mse": float(high_to_low[index]),
                "correct_future": bool(high_future[index]),
                "correct_history": bool(high_history[index]),
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


def contact_friction_prediction_gate(
    metrics: dict[str, Any],
    *,
    release: dict[str, Any],
) -> dict[str, Any]:
    thresholds = release["scoring"]["hidden_future_prediction"]["gates"]
    checks = {
        "correct_future_rate": (
            metrics["correct_future_rate"]
            >= float(thresholds["correct_future_rate_minimum"])
        ),
        "correct_history_rate": (
            metrics["correct_history_rate"]
            >= float(thresholds["correct_history_rate_minimum"])
        ),
        "context_switch_rate": (
            metrics["context_switch_rate"]
            >= float(thresholds["context_switch_rate_minimum"])
        ),
        "worst_friction_correct_future_rate": (
            metrics["worst_friction_correct_future_rate"]
            >= float(
                thresholds[
                    "worst_friction_correct_future_rate_minimum"
                ]
            )
        ),
    }
    checks.update(
        paired_latent_response_gate_checks(
            metrics, thresholds=thresholds
        )
    )
    return {"checks": checks, "passed": all(checks.values())}


def _contact_friction_development_contract(
    release: dict[str, Any],
) -> dict[str, Any]:
    development = contact_friction_development_data_contract(release)
    return {
        "release_id": release["release_id"],
        "release_config_sha256": file_sha256(
            Path(release["_config_path"])
        ),
        "development_data_manifest_sha256": development[
            "data_manifest_sha256"
        ],
        "development_split": development["split"],
        "development_lance_table": development["lance_table"],
        "development_lance_table_sha256": development[
            "lance_table_sha256"
        ],
    }


def _development_checkpoint_identity(
    adapter: ContactFrictionICLModelAdapter,
) -> tuple[dict[str, Any], dict[str, str]]:
    metadata = adapter.metadata
    checkpoint_sha256 = metadata.get("checkpoint_sha256")
    checkpoint_path = metadata.get("checkpoint")
    if (
        not isinstance(checkpoint_sha256, str)
        or len(checkpoint_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in checkpoint_sha256.lower()
        )
        or not isinstance(checkpoint_path, str)
        or not checkpoint_path
    ):
        raise RuntimeError(
            "Development-only scoring requires adapter checkpoint path and "
            "SHA-256 metadata"
        )
    return dict(metadata), {
        "path": checkpoint_path,
        "sha256": checkpoint_sha256,
    }


def evaluate_contact_friction_icl_development_model(
    *,
    adapter: ContactFrictionICLModelAdapter,
    model_name: str,
    training_recipe: str,
    training_seed: int | None,
    release_config: (
        Path | str
    ) = DEFAULT_CONTACT_FRICTION_RELEASE_CONFIG,
    repo_root: Path | None = None,
    batch_size: int = 64,
) -> dict[str, Any]:
    """Score a checkpoint on Loader Validation without opening Public Test.

    This is intentionally a separate entry point from
    :func:`evaluate_contact_friction_icl_model`: it only instantiates the
    Development dataset and pins its table digest before model inference.
    Its result cannot be passed to the Public-Test method aggregator.
    """

    root = (repo_root or repository_root()).resolve()
    release = load_contact_friction_icl_release(release_config)
    development_contract = _contact_friction_development_contract(release)
    dataset = ContactFrictionICLDevelopmentDataset(
        release=release,
        repo_root=root,
    )
    data_identity = dataset.identity
    if not data_identity["passed"]:
        raise RuntimeError(
            "Contact Friction Development data identity does not match the "
            "release contract"
        )
    arrays = dataset.arrays
    if not dataset.is_full_protocol:
        raise RuntimeError(
            "Development Contact Friction scoring requires all "
            f"{development_contract['development_split']} pairs"
        )
    validate_adapter_protocol(
        adapter,
        history_tokens=3,
        action_block_raw_steps=5,
        action_dim=2,
        minimum_future_action_blocks=1,
        task_name="Contact Friction Development",
    )
    adapter_metadata, checkpoint = _development_checkpoint_identity(adapter)

    histories = np.concatenate(
        [arrays.low_pixels[:, :3], arrays.high_pixels[:, :3]],
        axis=0,
    )
    actions = np.concatenate(
        [arrays.raw_action_blocks[:, :3], arrays.raw_action_blocks[:, :3]],
        axis=0,
    )
    before = adapter.frozen_state_hash()
    predicted = adapter.rollout_latents(
        histories,
        actions,
        batch_size=int(batch_size),
    )
    count = arrays.pair_count
    if (
        predicted.ndim != 3
        or predicted.shape != (2 * count, 1, predicted.shape[-1])
        or not np.isfinite(predicted).all()
    ):
        raise RuntimeError(
            "Contact Friction Development adapter must return finite "
            "(2 * pairs, 1, latent_dim) predictions"
        )
    true_futures = np.concatenate(
        [arrays.low_pixels[:, 3], arrays.high_pixels[:, 3]],
        axis=0,
    )
    encoded = adapter.encode_pixels(
        true_futures,
        batch_size=int(batch_size),
    )
    if (
        encoded.ndim != 2
        or encoded.shape != (2 * count, predicted.shape[-1])
        or not np.isfinite(encoded).all()
    ):
        raise RuntimeError(
            "Contact Friction Development target latents do not match "
            "predictions"
        )
    after = adapter.frozen_state_hash()
    if before != after:
        raise RuntimeError(
            "Model state changed during Contact Friction Development scoring"
        )
    metrics, records = contact_friction_prediction_metrics(
        pair_ids=arrays.pair_ids,
        predicted_low=predicted[:count, 0],
        predicted_high=predicted[count:, 0],
        target_low=encoded[:count],
        target_high=encoded[count:],
    )
    return {
        "schema_version": 1,
        "benchmark": CONTACT_FRICTION_DEVELOPMENT_BENCHMARK,
        "submission_kind": "single_checkpoint",
        "status": "completed",
        "contract": development_contract,
        "model": {
            "name": str(model_name),
            "training_recipe": str(training_recipe),
            "training_seed": (
                None if training_seed is None else int(training_seed)
            ),
            "checkpoint": checkpoint,
            "adapter": adapter_metadata,
            "state_sha256_before": before,
            "state_sha256_after": after,
        },
        "data": dataset.describe(),
        "metrics": metrics,
        "gate": contact_friction_prediction_gate(metrics, release=release),
        "claim_scope": "Development_only_not_Public_or_release",
        "public_test": dataset.development["public_test"],
        "records": records,
    }


def evaluate_contact_friction_icl_model(
    *,
    adapter: ContactFrictionICLModelAdapter,
    model_name: str,
    training_recipe: str,
    training_seed: int | None,
    release_config: (
        Path | str
    ) = DEFAULT_CONTACT_FRICTION_RELEASE_CONFIG,
    repo_root: Path | None = None,
    batch_size: int = 64,
    include_records: bool = True,
) -> dict[str, Any]:
    """Score one frozen checkpoint on all independent friction pairs."""

    root = (repo_root or repository_root()).resolve()
    release = load_contact_friction_icl_release(release_config)
    dataset = ContactFrictionICLEvalDataset(
        release=release,
        repo_root=root,
    )
    arrays = dataset.arrays
    if not dataset.is_full_protocol:
        raise RuntimeError(
            "Formal Contact Friction scoring requires all 256 pairs"
        )
    validate_adapter_protocol(
        adapter,
        history_tokens=3,
        action_block_raw_steps=5,
        action_dim=2,
        minimum_future_action_blocks=1,
        task_name="Contact Friction v1",
    )

    histories = np.concatenate(
        [arrays.low_pixels[:, :3], arrays.high_pixels[:, :3]],
        axis=0,
    )
    actions = np.concatenate(
        [
            arrays.raw_action_blocks[:, :3],
            arrays.raw_action_blocks[:, :3],
        ],
        axis=0,
    )
    before = adapter.frozen_state_hash()
    predicted = adapter.rollout_latents(
        histories,
        actions,
        batch_size=int(batch_size),
    )
    if predicted.ndim != 3 or predicted.shape[1] != 1:
        raise RuntimeError(
            "Contact Friction adapter must return one predicted future"
        )
    predicted = predicted[:, 0]
    true_futures = np.concatenate(
        [arrays.low_pixels[:, 3], arrays.high_pixels[:, 3]],
        axis=0,
    )
    encoded = adapter.encode_pixels(
        true_futures,
        batch_size=int(batch_size),
    )
    after = adapter.frozen_state_hash()
    if before != after:
        raise RuntimeError(
            "Model state changed during Contact Friction scoring"
        )

    count = arrays.pair_count
    metrics, records = contact_friction_prediction_metrics(
        pair_ids=arrays.pair_ids,
        predicted_low=predicted[:count],
        predicted_high=predicted[count:],
        target_low=encoded[:count],
        target_high=encoded[count:],
    )
    gate = contact_friction_prediction_gate(metrics, release=release)
    release_path = Path(release["_config_path"])
    payload = {
        "schema_version": 1,
        "benchmark": "pusht_history3_contact_friction_icl_v1",
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
            "training_seed": (
                None if training_seed is None else int(training_seed)
            ),
            "adapter": adapter.metadata,
            "state_sha256_before": before,
            "state_sha256_after": after,
        },
        "data": dataset.describe(),
        "metrics": metrics,
        "gate": gate,
    }
    if include_records:
        payload["records"] = records
    return payload


def _rescore_result(
    path: Path,
    *,
    release: dict[str, Any],
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema_version") != 1
        or payload.get("benchmark")
        != "pusht_history3_contact_friction_icl_v1"
        or payload.get("submission_kind") != "single_checkpoint"
        or payload.get("status") != "completed"
    ):
        raise ValueError(f"Unsupported Contact Friction result: {path}")
    expected_release = {
        "release_id": release["release_id"],
        "release_config_sha256": file_sha256(
            Path(release["_config_path"])
        ),
        "data_manifest_sha256": release["data"]["manifest_sha256"],
        "sealed_test_included": False,
    }
    if payload.get("release") != expected_release:
        raise RuntimeError(f"Release identity mismatch: {path}")
    records = payload.get("records")
    expected_pairs = int(release["evaluation"]["pair_count"])
    if not isinstance(records, list) or len(records) != expected_pairs:
        raise ValueError(
            f"Independent rescoring requires all {expected_pairs} records"
        )

    def booleans(mode: str, field: str) -> np.ndarray:
        return np.asarray(
            [row[mode][field] for row in records],
            dtype=bool,
        )

    low_future = booleans("low_friction", "correct_future")
    high_future = booleans("high_friction", "correct_future")
    low_history = booleans("low_friction", "correct_history")
    high_history = booleans("high_friction", "correct_history")
    switch = np.asarray(
        [row["context_switch_correct"] for row in records],
        dtype=bool,
    )
    correct_losses = np.asarray(
        [
            row["low_friction"]["correct_future_mse"]
            for row in records
        ]
        + [
            row["high_friction"]["correct_future_mse"]
            for row in records
        ],
        dtype=np.float64,
    )
    other_losses = np.asarray(
        [
            row["low_friction"]["other_future_mse"]
            for row in records
        ]
        + [
            row["high_friction"]["other_future_mse"]
            for row in records
        ],
        dtype=np.float64,
    )
    metrics = {
        "pair_count": len(records),
        "decision_count": 2 * len(records),
        "correct_future_rate": float(
            np.concatenate([low_future, high_future]).mean()
        ),
        "correct_history_rate": float(
            np.concatenate([low_history, high_history]).mean()
        ),
        "context_switch_rate": float(switch.mean()),
        "low_friction_correct_future_rate": float(low_future.mean()),
        "high_friction_correct_future_rate": float(high_future.mean()),
        "worst_friction_correct_future_rate": float(
            min(low_future.mean(), high_future.mean())
        ),
        "correct_future_mse_mean": float(correct_losses.mean()),
        "other_future_mse_mean": float(other_losses.mean()),
        "other_minus_correct_mse_margin_mean": float(
            (other_losses - correct_losses).mean()
        ),
        "current_frame_only_accuracy_bound": 0.5,
    }
    response_rows_present = [
        isinstance(row.get("latent_response"), dict) for row in records
    ]
    if not any(response_rows_present):
        raise ValueError(
            "Legacy Contact Friction result lacks mandatory latent response "
            f"metrics and must be rescored from its checkpoint: {path}"
        )
    if any(response_rows_present):
        if not all(response_rows_present):
            raise RuntimeError(
                f"Incomplete Contact Friction latent response records: {path}"
            )
        metrics["latent_response"] = (
            summarize_paired_latent_response_records(
                [
                    {
                        "pair_id": row["pair_id"],
                        **row["latent_response"],
                    }
                    for row in records
                ]
            )
        )
        calibrated_response = np.asarray(
            [
                row["latent_response"][
                    "calibrated_response_success"
                ]
                for row in records
            ],
            dtype=bool,
        )
        joint = (
            low_future
            & high_future
            & low_history
            & high_history
            & calibrated_response
        )
        if not all(
            row.get("joint_icl_pair_success") == bool(joint[index])
            for index, row in enumerate(records)
        ):
            raise RuntimeError(
                f"Invalid Contact Friction joint ICL records: {path}"
            )
        metrics["joint_icl_pair_success_rate"] = float(joint.mean())
    gate = contact_friction_prediction_gate(metrics, release=release)
    stored = payload.get("metrics")
    scalar_metric_names = set(metrics) - {"latent_response"}
    metrics_match = (
        isinstance(stored, dict)
        and set(stored) == set(metrics)
        and all(
            (
                value == stored[name]
                if isinstance(value, int)
                else np.isclose(
                    value,
                    stored[name],
                    rtol=1e-7,
                    atol=1e-9,
                )
            )
            for name, value in metrics.items()
            if name in scalar_metric_names
        )
        and (
            "latent_response" not in metrics
            or paired_latent_response_summaries_close(
                stored["latent_response"], metrics["latent_response"]
            )
        )
    )
    if not metrics_match or gate != payload.get("gate"):
        raise RuntimeError(f"Stored Contact Friction score changed: {path}")
    return payload


def rescore_contact_friction_icl_development_result(
    path: Path | str,
    *,
    release_config: (
        Path | str
    ) = DEFAULT_CONTACT_FRICTION_RELEASE_CONFIG,
) -> dict[str, Any]:
    """Independently recompute a Development-only result from its records."""

    release = load_contact_friction_icl_release(release_config)
    result_path = Path(path).expanduser().resolve()
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    if (
        payload.get("schema_version") != 1
        or payload.get("benchmark")
        != CONTACT_FRICTION_DEVELOPMENT_BENCHMARK
        or payload.get("submission_kind") != "single_checkpoint"
        or payload.get("status") != "completed"
        or payload.get("claim_scope")
        != "Development_only_not_Public_or_release"
    ):
        raise ValueError(
            f"Unsupported Contact Friction Development result: {result_path}"
        )
    if payload.get("contract") != _contact_friction_development_contract(
        release
    ):
        raise RuntimeError(
            f"Contact Friction Development contract mismatch: {result_path}"
        )
    development = contact_friction_development_data_contract(release)
    if payload.get("public_test") != development["public_test"]:
        raise RuntimeError(
            f"Contact Friction Development result opened Public Test: {result_path}"
        )
    model = payload.get("model", {})
    checkpoint = model.get("checkpoint")
    adapter = model.get("adapter", {})
    if (
        not isinstance(checkpoint, dict)
        or not isinstance(adapter, dict)
        or checkpoint.get("sha256") != adapter.get("checkpoint_sha256")
        or not isinstance(checkpoint.get("path"), str)
        or not isinstance(checkpoint.get("sha256"), str)
        or len(checkpoint["sha256"]) != 64
    ):
        raise RuntimeError(
            f"Contact Friction Development checkpoint identity mismatch: {result_path}"
        )
    records = payload.get("records")
    expected_pairs = int(development["pair_count"])
    if not isinstance(records, list) or len(records) != expected_pairs:
        raise ValueError(
            "Independent Development rescoring requires all "
            f"{expected_pairs} records"
        )

    def booleans(mode: str, field: str) -> np.ndarray:
        return np.asarray(
            [row[mode][field] for row in records],
            dtype=bool,
        )

    low_future = booleans("low_friction", "correct_future")
    high_future = booleans("high_friction", "correct_future")
    low_history = booleans("low_friction", "correct_history")
    high_history = booleans("high_friction", "correct_history")
    switch = np.asarray(
        [row["context_switch_correct"] for row in records],
        dtype=bool,
    )
    correct_losses = np.asarray(
        [row["low_friction"]["correct_future_mse"] for row in records]
        + [row["high_friction"]["correct_future_mse"] for row in records],
        dtype=np.float64,
    )
    other_losses = np.asarray(
        [row["low_friction"]["other_future_mse"] for row in records]
        + [row["high_friction"]["other_future_mse"] for row in records],
        dtype=np.float64,
    )
    metrics = {
        "pair_count": len(records),
        "decision_count": 2 * len(records),
        "correct_future_rate": float(
            np.concatenate([low_future, high_future]).mean()
        ),
        "correct_history_rate": float(
            np.concatenate([low_history, high_history]).mean()
        ),
        "context_switch_rate": float(switch.mean()),
        "low_friction_correct_future_rate": float(low_future.mean()),
        "high_friction_correct_future_rate": float(high_future.mean()),
        "worst_friction_correct_future_rate": float(
            min(low_future.mean(), high_future.mean())
        ),
        "correct_future_mse_mean": float(correct_losses.mean()),
        "other_future_mse_mean": float(other_losses.mean()),
        "other_minus_correct_mse_margin_mean": float(
            (other_losses - correct_losses).mean()
        ),
        "current_frame_only_accuracy_bound": 0.5,
    }
    response_rows_present = [
        isinstance(row.get("latent_response"), dict) for row in records
    ]
    if not all(response_rows_present):
        raise ValueError(
            "Contact Friction Development records must include latent "
            f"response evidence: {result_path}"
        )
    metrics["latent_response"] = summarize_paired_latent_response_records(
        [
            {"pair_id": row["pair_id"], **row["latent_response"]}
            for row in records
        ]
    )
    calibrated_response = np.asarray(
        [
            row["latent_response"]["calibrated_response_success"]
            for row in records
        ],
        dtype=bool,
    )
    joint = (
        low_future
        & high_future
        & low_history
        & high_history
        & calibrated_response
    )
    if not all(
        row.get("joint_icl_pair_success") == bool(joint[index])
        for index, row in enumerate(records)
    ):
        raise RuntimeError(
            "Invalid Contact Friction Development joint ICL records: "
            f"{result_path}"
        )
    metrics["joint_icl_pair_success_rate"] = float(joint.mean())
    gate = contact_friction_prediction_gate(metrics, release=release)
    stored = payload.get("metrics")
    scalar_metric_names = set(metrics) - {"latent_response"}
    metrics_match = (
        isinstance(stored, dict)
        and set(stored) == set(metrics)
        and all(
            (
                value == stored[name]
                if isinstance(value, int)
                else np.isclose(value, stored[name], rtol=1e-7, atol=1e-9)
            )
            for name, value in metrics.items()
            if name in scalar_metric_names
        )
        and paired_latent_response_summaries_close(
            stored["latent_response"], metrics["latent_response"]
        )
    )
    if not metrics_match or gate != payload.get("gate"):
        raise RuntimeError(
            f"Stored Contact Friction Development score changed: {result_path}"
        )
    return payload


def _stats(values: Iterable[float]) -> dict[str, float]:
    rows = [float(value) for value in values]
    return {
        "mean": float(statistics.fmean(rows)),
        "sample_std": (
            float(statistics.stdev(rows)) if len(rows) > 1 else 0.0
        ),
        "minimum": float(min(rows)),
        "maximum": float(max(rows)),
    }


def score_contact_friction_icl_results(
    *,
    result_paths: Iterable[Path | str],
    method_name: str,
    release_config: (
        Path | str
    ) = DEFAULT_CONTACT_FRICTION_RELEASE_CONFIG,
) -> dict[str, Any]:
    """Rescore one descriptive checkpoint or three independent seeds."""

    release = load_contact_friction_icl_release(release_config)
    paths = [Path(value).expanduser().resolve() for value in result_paths]
    if len(paths) not in {1, 3}:
        raise ValueError(
            "Provide one result for a descriptive checkpoint or three "
            "results for a method-level claim"
        )
    results = [
        _rescore_result(path, release=release) for path in paths
    ]
    hashes = [
        str(result["model"]["adapter"].get("checkpoint_sha256", ""))
        for result in results
    ]
    if (
        any(len(value) != 64 for value in hashes)
        or len(set(hashes)) != len(hashes)
    ):
        raise ValueError("Every result must bind a distinct checkpoint hash")
    seeds = [result["model"]["training_seed"] for result in results]
    if len(paths) == 3:
        if any(seed is None for seed in seeds) or len(set(seeds)) != 3:
            raise ValueError("A method claim requires three training seeds")
        recipes = {
            str(result["model"]["training_recipe"]) for result in results
        }
        if len(recipes) != 1:
            raise ValueError("A method score cannot mix training recipes")
    metric_names = (
        "correct_future_rate",
        "correct_history_rate",
        "context_switch_rate",
        "worst_friction_correct_future_rate",
    )
    if all(
        "joint_icl_pair_success_rate" in result["metrics"]
        for result in results
    ):
        metric_names += ("joint_icl_pair_success_rate",)
    checkpoints = [
        {
            "path": str(path),
            "checkpoint_sha256": result["model"]["adapter"][
                "checkpoint_sha256"
            ],
            "training_seed": result["model"]["training_seed"],
            **{
                name: result["metrics"][name] for name in metric_names
            },
            "passed": bool(result["gate"]["passed"]),
        }
        for path, result in zip(paths, results, strict=True)
    ]
    formal = len(paths) == 3
    passed = formal and all(row["passed"] for row in checkpoints)
    return {
        "schema_version": 1,
        "benchmark": "pusht_history3_contact_friction_icl_v1",
        "submission_kind": (
            "three_seed_method" if formal else "descriptive_checkpoint"
        ),
        "status": "completed",
        "method_name": str(method_name),
        "release_id": release["release_id"],
        "checkpoints": checkpoints,
        "aggregate": {
            metric: _stats(row[metric] for row in checkpoints)
            for metric in metric_names
        },
        "decision": {
            "passed": passed,
            "formal_method_claim": formal,
            "reason": (
                "all_three_training_seeds_passed"
                if passed
                else (
                    "one_or_more_training_seeds_failed"
                    if formal
                    else "single_checkpoint_is_descriptive_only"
                )
            ),
        },
    }


__all__ = [
    "CONTACT_FRICTION_DEVELOPMENT_BENCHMARK",
    "contact_friction_prediction_gate",
    "contact_friction_prediction_metrics",
    "evaluate_contact_friction_icl_development_model",
    "evaluate_contact_friction_icl_model",
    "rescore_contact_friction_icl_development_result",
    "score_contact_friction_icl_results",
]
