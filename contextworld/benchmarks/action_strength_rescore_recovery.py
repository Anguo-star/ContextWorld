"""Immutable float32 recovery for one frozen Action Strength baseline.

The original Action Strength evaluator computes its three MSE aggregates from
``float32`` arrays.  JSON records preserve the individual values exactly, but
the generic rescorer historically rebuilt those arrays with NumPy's implicit
``float64`` dtype.  That changes the mean margin by about 1.7e-9 for the
frozen original PushT LeWM receipt and is enough to trip its deliberately
tight numerical comparison.

This module is intentionally narrow.  It can recover only that frozen raw
receipt, only against the frozen original-baseline preregistration, and only
by writing an additive, x-exclusive receipt below
``original_baseline_matrix_v1/rescore_recovery``.  It never loads a model or
rewrites an existing result.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from contextworld.benchmarks.action_strength_icl_data import (
    DEFAULT_ACTION_STRENGTH_RELEASE_CONFIG,
    load_action_strength_icl_release,
)
from contextworld.benchmarks.action_strength_icl_score import (
    _prediction_contract_sha256,
    _prediction_gate,
)
from contextworld.benchmarks.paired_latent_response import (
    paired_latent_response_summaries_close,
    summarize_paired_latent_response_records,
)
from contextworld.paths import repository_root


RECOVERY_ID = (
    "contextworld_original_baseline_matrix_v1_"
    "action_strength_lewm_float32_rescore_recovery_v1"
)
RECOVERY_NAMESPACE = Path(
    "artifacts/evaluation/original_baseline_matrix_v1/rescore_recovery"
)
DEFAULT_RAW_RECEIPT = Path(
    "artifacts/evaluation/original_baseline_matrix_v1/"
    "contextworld-action-strength/lewm.json"
)
DEFAULT_PREREGISTRATION = Path(
    "configs/benchmark/contextworld_original_baseline_completion_prereg_v1.yaml"
)
DEFAULT_FREEZE_RECEIPT = Path(
    "configs/benchmark/contextworld_original_baseline_completion_freeze_v1.json"
)
ACTION_STRENGTH_RELEASE_LOGICAL_PATH = Path(
    "configs/benchmark/pusht_action_strength_icl_release_v1.yaml"
)
DEFAULT_RECOVERY_RECEIPT = RECOVERY_NAMESPACE / (
    "contextworld-action-strength/lewm_float32_rescore_recovery_v1.json"
)

_EXPECTED_BENCHMARK = "pusht_history3_action_strength_icl_v1"
_EXPECTED_CHECKPOINT_ID = "pusht_lewm_original"
_EXPECTED_COMPONENT_ID = "contextworld-action-strength"
_SCALAR_METRIC_NAMES = (
    "pair_count",
    "decision_count",
    "correct_future_rate",
    "correct_history_rate",
    "rule_switch_rate",
    "low_strength_correct_future_rate",
    "high_strength_correct_future_rate",
    "worst_strength_correct_future_rate",
    "correct_future_mse_mean",
    "other_future_mse_mean",
    "other_minus_correct_mse_margin_mean",
    "current_frame_only_accuracy_bound",
    "joint_icl_pair_success_rate",
)
_FLOAT32_LOSS_METRIC_NAMES = {
    "correct_future_mse_mean",
    "other_future_mse_mean",
    "other_minus_correct_mse_margin_mean",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(path: Path, *, logical_path: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required recovery input is missing: {path}")
    return {
        "path": logical_path,
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _canonical_input(
    value: Path | str,
    *,
    repo_root: Path,
    logical_path: Path,
    label: str,
) -> Path:
    expected = (repo_root / logical_path).resolve()
    observed = Path(value).expanduser().resolve()
    if observed != expected:
        raise ValueError(
            f"{label} must be the frozen canonical input {expected}, got "
            f"{observed}"
        )
    if not expected.is_file():
        raise FileNotFoundError(f"{label} is missing: {expected}")
    return expected


def validate_recovery_output_path(
    output_path: Path | str,
    *,
    repo_root: Path | None = None,
) -> tuple[Path, str]:
    """Require an additive receipt below the dedicated recovery namespace."""

    root = (repo_root or repository_root()).resolve()
    output = Path(output_path).expanduser()
    if not output.is_absolute():
        output = root / output
    output = output.resolve()
    try:
        logical = output.relative_to(root)
    except ValueError as error:
        raise ValueError("Recovery output must remain inside the repository") from error
    try:
        logical.relative_to(RECOVERY_NAMESPACE)
    except ValueError as error:
        raise ValueError(
            "Recovery output must be under "
            f"{RECOVERY_NAMESPACE.as_posix()}"
        ) from error
    if output.suffix != ".json":
        raise ValueError("Recovery output must be a JSON receipt")
    return output, logical.as_posix()


def write_recovery_receipt_exclusive(
    output_path: Path | str,
    payload: Mapping[str, Any],
    *,
    repo_root: Path | None = None,
) -> Path:
    """Create one recovery receipt without replacing an existing file."""

    output, _ = validate_recovery_output_path(output_path, repo_root=repo_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(dict(payload), indent=2, sort_keys=True) + "\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(output, flags, 0o644)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        # A partial receipt is never a valid recovery output.  We only unlink
        # the file descriptor that this invocation created with O_EXCL.
        output.unlink(missing_ok=True)
        raise
    return output


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _records(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        raise ValueError("Raw Action Strength receipt records must be a list")
    rows = [_mapping(row, label=f"record {index}") for index, row in enumerate(value)]
    if len(rows) != 256:
        raise ValueError("Frozen Action Strength recovery requires 256 records")
    identifiers = [str(row.get("pair_id", "")) for row in rows]
    if not all(identifiers) or len(set(identifiers)) != len(identifiers):
        raise ValueError("Raw Action Strength record IDs must be nonempty and unique")
    return rows


def _boolean_values(
    records: list[Mapping[str, Any]], *, section: str, field: str
) -> np.ndarray:
    values: list[bool] = []
    for index, row in enumerate(records):
        value = _mapping(row.get(section), label=f"record {index}.{section}").get(field)
        if type(value) is not bool:
            raise ValueError(f"record {index}.{section}.{field} must be boolean")
        values.append(value)
    return np.asarray(values, dtype=bool)


def _top_level_boolean_values(
    records: list[Mapping[str, Any]], *, field: str
) -> np.ndarray:
    values: list[bool] = []
    for index, row in enumerate(records):
        value = row.get(field)
        if type(value) is not bool:
            raise ValueError(f"record {index}.{field} must be boolean")
        values.append(value)
    return np.asarray(values, dtype=bool)


def _float32_values(
    records: list[Mapping[str, Any]], *, section: str, field: str
) -> np.ndarray:
    values: list[float] = []
    for index, row in enumerate(records):
        value = _mapping(row.get(section), label=f"record {index}.{section}").get(field)
        if isinstance(value, bool):
            raise ValueError(f"record {index}.{section}.{field} must be numeric")
        try:
            number = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"record {index}.{section}.{field} must be numeric"
            ) from error
        if not np.isfinite(number):
            raise ValueError(f"record {index}.{section}.{field} must be finite")
        values.append(number)
    result = np.asarray(values, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError(f"{section}.{field} cannot be represented as finite float32")
    return result


def _reconstruct_metrics(
    records: list[Mapping[str, Any]], *, release: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Rebuild every metric with the original evaluator's dtype semantics."""

    low_target = _boolean_values(
        records, section="low_strength", field="correct_future"
    )
    high_target = _boolean_values(
        records, section="high_strength", field="correct_future"
    )
    low_history = _boolean_values(
        records, section="low_strength", field="correct_history"
    )
    high_history = _boolean_values(
        records, section="high_strength", field="correct_history"
    )
    switch = _top_level_boolean_values(records, field="rule_switch_correct")

    # This is the recovery's material distinction from the legacy generic
    # rescorer: the source evaluator's low/high MSE arrays are float32.
    correct_losses = np.concatenate(
        [
            _float32_values(
                records, section="low_strength", field="correct_future_mse"
            ),
            _float32_values(
                records, section="high_strength", field="correct_future_mse"
            ),
        ]
    )
    other_losses = np.concatenate(
        [
            _float32_values(
                records, section="low_strength", field="other_future_mse"
            ),
            _float32_values(
                records, section="high_strength", field="other_future_mse"
            ),
        ]
    )
    if correct_losses.dtype != np.float32 or other_losses.dtype != np.float32:
        raise AssertionError("Float32 Action Strength aggregation was lost")

    latent_records: list[dict[str, Any]] = []
    for index, row in enumerate(records):
        response = _mapping(
            row.get("latent_response"), label=f"record {index}.latent_response"
        )
        if type(response.get("calibrated_response_success")) is not bool:
            raise ValueError(
                "Action Strength latent calibrated_response_success must be boolean"
            )
        latent_records.append({"pair_id": row["pair_id"], **response})
    latent_response = summarize_paired_latent_response_records(latent_records)
    calibrated_response = np.asarray(
        [row["calibrated_response_success"] for row in latent_records], dtype=bool
    )
    joint = (
        low_target
        & high_target
        & low_history
        & high_history
        & calibrated_response
    )
    for index, row in enumerate(records):
        if row.get("joint_icl_pair_success") != bool(joint[index]):
            raise RuntimeError(
                "Raw Action Strength joint ICL record disagrees with its "
                f"reconstructed inputs at pair {row['pair_id']}"
            )

    metrics: dict[str, Any] = {
        "pair_count": len(records),
        "decision_count": 2 * len(records),
        "correct_future_rate": float(np.concatenate([low_target, high_target]).mean()),
        "correct_history_rate": float(
            np.concatenate([low_history, high_history]).mean()
        ),
        "rule_switch_rate": float(switch.mean()),
        "low_strength_correct_future_rate": float(low_target.mean()),
        "high_strength_correct_future_rate": float(high_target.mean()),
        "worst_strength_correct_future_rate": float(
            min(low_target.mean(), high_target.mean())
        ),
        "correct_future_mse_mean": float(correct_losses.mean()),
        "other_future_mse_mean": float(other_losses.mean()),
        "other_minus_correct_mse_margin_mean": float(
            (other_losses - correct_losses).mean()
        ),
        "current_frame_only_accuracy_bound": 0.5,
        "latent_response": latent_response,
        "joint_icl_pair_success_rate": float(joint.mean()),
    }
    return metrics, _prediction_gate(metrics, release=dict(release))


def _float64_bits(value: float) -> str:
    return f"0x{struct.unpack('>Q', struct.pack('>d', float(value)))[0]:016x}"


def _float32_bits(value: float) -> str:
    bits = np.asarray(value, dtype=np.float32).view(np.uint32).item()
    return f"0x{bits:08x}"


def _scalar_comparison(
    stored: Mapping[str, Any], recomputed: Mapping[str, Any]
) -> dict[str, Any]:
    if set(stored) != set(recomputed):
        raise RuntimeError("Stored and reconstructed Action Strength metric keys differ")
    rows: dict[str, Any] = {}
    for name in _SCALAR_METRIC_NAMES:
        expected = stored.get(name)
        actual = recomputed.get(name)
        if isinstance(expected, bool) or isinstance(actual, bool):
            raise RuntimeError(f"Action Strength scalar {name} cannot be boolean")
        if isinstance(expected, int) and isinstance(actual, int):
            exact = expected == actual
            rows[name] = {
                "comparison": "integer_exact",
                "stored": expected,
                "recomputed": actual,
                "exact": exact,
            }
            continue
        try:
            expected_float = float(expected)
            actual_float = float(actual)
        except (TypeError, ValueError) as error:
            raise RuntimeError(f"Action Strength scalar {name} is not numeric") from error
        exact = _float64_bits(expected_float) == _float64_bits(actual_float)
        row: dict[str, Any] = {
            "comparison": "float64_json_bitwise",
            "stored": expected_float,
            "recomputed": actual_float,
            "stored_float64_bits": _float64_bits(expected_float),
            "recomputed_float64_bits": _float64_bits(actual_float),
            "exact": exact,
            "within_legacy_tolerance": bool(
                np.isclose(actual_float, expected_float, rtol=1.0e-7, atol=1.0e-9)
            ),
        }
        if name in _FLOAT32_LOSS_METRIC_NAMES:
            row.update(
                {
                    "source_aggregation_dtype": "float32",
                    "stored_float32_bits": _float32_bits(expected_float),
                    "recomputed_float32_bits": _float32_bits(actual_float),
                    "float32_bitwise_equal": (
                        _float32_bits(expected_float) == _float32_bits(actual_float)
                    ),
                }
            )
        rows[name] = row
    exact = all(bool(row["exact"]) for row in rows.values())
    float32_exact = all(
        bool(rows[name]["float32_bitwise_equal"])
        for name in _FLOAT32_LOSS_METRIC_NAMES
    )
    return {
        "comparison_scope": "all top-level non-latent scalar metrics",
        "all_float64_json_bitwise_equal": exact,
        "all_float32_loss_aggregates_bitwise_equal": float32_exact,
        "all_within_legacy_tolerance": all(
            bool(row.get("within_legacy_tolerance", True))
            for row in rows.values()
        ),
        "metrics": rows,
    }


def _latent_leaf_comparison(
    stored: Any, recomputed: Any, *, path: str = "latent_response"
) -> list[dict[str, Any]]:
    if isinstance(stored, Mapping) or isinstance(recomputed, Mapping):
        if not isinstance(stored, Mapping) or not isinstance(recomputed, Mapping):
            raise RuntimeError(f"Latent response mapping mismatch at {path}")
        if set(stored) != set(recomputed):
            raise RuntimeError(f"Latent response keys differ at {path}")
        leaves: list[dict[str, Any]] = []
        for name in sorted(stored):
            leaves.extend(
                _latent_leaf_comparison(
                    stored[name], recomputed[name], path=f"{path}.{name}"
                )
            )
        return leaves
    if isinstance(stored, bool) or isinstance(recomputed, bool):
        return [
            {
                "path": path,
                "comparison": "exact",
                "stored": stored,
                "recomputed": recomputed,
                "exact": stored == recomputed,
            }
        ]
    if isinstance(stored, str) or isinstance(recomputed, str):
        return [
            {
                "path": path,
                "comparison": "exact",
                "stored": stored,
                "recomputed": recomputed,
                "exact": stored == recomputed,
            }
        ]
    try:
        expected = float(stored)
        actual = float(recomputed)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"Unsupported latent response value at {path}") from error
    return [
        {
            "path": path,
            "comparison": "float64_json_bitwise_and_tolerance",
            "stored": expected,
            "recomputed": actual,
            "stored_float64_bits": _float64_bits(expected),
            "recomputed_float64_bits": _float64_bits(actual),
            "exact": _float64_bits(expected) == _float64_bits(actual),
            "within_paired_response_tolerance": bool(
                np.isclose(actual, expected, rtol=1.0e-7, atol=1.0e-10)
            ),
        }
    ]


def _latent_comparison(
    stored: Mapping[str, Any], recomputed: Mapping[str, Any]
) -> dict[str, Any]:
    leaves = _latent_leaf_comparison(stored, recomputed)
    return {
        "aggregation_dtype": "float64_in_frozen_paired_latent_response_metric",
        "all_float64_json_bitwise_equal": all(
            bool(row["exact"]) for row in leaves
        ),
        "all_within_paired_response_tolerance": all(
            bool(row.get("within_paired_response_tolerance", True))
            for row in leaves
        ),
        "paired_latent_response_summaries_close": (
            paired_latent_response_summaries_close(stored, recomputed)
        ),
        "leaves": leaves,
    }


def _load_frozen_bindings(
    *,
    root: Path,
    preregistration: Path,
    freeze_receipt: Path,
    release_config: Path,
    raw_receipt: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    prereg_identity = _identity(
        preregistration, logical_path=DEFAULT_PREREGISTRATION.as_posix()
    )
    freeze_identity = _identity(
        freeze_receipt, logical_path=DEFAULT_FREEZE_RECEIPT.as_posix()
    )
    release_identity = _identity(
        release_config,
        logical_path=ACTION_STRENGTH_RELEASE_LOGICAL_PATH.as_posix(),
    )
    raw_identity = _identity(raw_receipt, logical_path=DEFAULT_RAW_RECEIPT.as_posix())

    prereg = yaml.safe_load(preregistration.read_text(encoding="utf-8"))
    prereg = dict(_mapping(prereg, label="frozen original-baseline preregistration"))
    if (
        prereg.get("preregistration_id")
        != "contextworld_original_baseline_completion_v1"
        or prereg.get("status") != "frozen_before_new_baseline_scoring"
    ):
        raise RuntimeError("Unexpected original-baseline preregistration")
    freeze = json.loads(freeze_receipt.read_text(encoding="utf-8"))
    freeze = dict(_mapping(freeze, label="original-baseline freeze receipt"))
    if (
        freeze.get("freeze_id") != "contextworld_original_baseline_completion_freeze_v1"
        or freeze.get("status") != "frozen_before_new_baseline_scoring"
        or freeze.get("preregistration") != prereg_identity
    ):
        raise RuntimeError("Frozen original-baseline preregistration binding failed")

    components = _mapping({row["capability_id"]: row for row in prereg["components"]}, label="components")
    component = _mapping(
        components.get(_EXPECTED_COMPONENT_ID), label="Action Strength component"
    )
    if component.get("release_config") != release_identity:
        raise RuntimeError("Frozen Action Strength release identity binding failed")
    checkpoints = _mapping(
        {row["checkpoint_id"]: row for row in prereg["checkpoints"]},
        label="checkpoints",
    )
    checkpoint = _mapping(
        checkpoints.get(_EXPECTED_CHECKPOINT_ID), label="frozen PushT LeWM checkpoint"
    )
    if checkpoint.get("contextworld_capability_training_used") is not False:
        raise RuntimeError("Recovery checkpoint is not frozen as original-only")
    weights = _mapping(checkpoint.get("weights"), label="frozen checkpoint weights")
    if not isinstance(weights.get("sha256"), str) or len(weights["sha256"]) != 64:
        raise RuntimeError("Frozen checkpoint SHA-256 is invalid")

    raw = json.loads(raw_receipt.read_text(encoding="utf-8"))
    raw = dict(_mapping(raw, label="raw Action Strength receipt"))
    return prereg, freeze, raw, {
        "preregistration": prereg_identity,
        "freeze_receipt": freeze_identity,
        "release_config": release_identity,
        "raw_receipt": raw_identity,
        "checkpoint": dict(checkpoint),
    }


def _validate_raw_receipt(
    raw: Mapping[str, Any], *, release: Mapping[str, Any], bindings: Mapping[str, Any]
) -> None:
    if (
        raw.get("schema_version") != 1
        or raw.get("benchmark") != _EXPECTED_BENCHMARK
        or raw.get("submission_kind") != "single_checkpoint"
        or raw.get("status") != "completed"
    ):
        raise RuntimeError("Raw receipt is not the frozen Action Strength result")
    raw_release = _mapping(raw.get("release"), label="raw receipt release")
    release_identity = _mapping(bindings["release_config"], label="release identity")
    expected_release = {
        "release_id": release["release_id"],
        "release_config_sha256_at_evaluation": release_identity["sha256"],
        "prediction_contract_sha256": _prediction_contract_sha256(dict(release)),
        "training_manifest_sha256": release["training"]["manifest_sha256"],
        "confirmation_manifest_sha256": release["evaluation"]["manifest_sha256"],
        "sealed_test_included": False,
    }
    if raw_release != expected_release:
        raise RuntimeError("Raw receipt does not bind the frozen release exactly")
    raw_model = _mapping(raw.get("model"), label="raw receipt model")
    raw_adapter = _mapping(raw_model.get("adapter"), label="raw receipt adapter")
    frozen_checkpoint = _mapping(bindings["checkpoint"], label="frozen checkpoint")
    frozen_weights = _mapping(frozen_checkpoint["weights"], label="frozen weights")
    if raw_adapter.get("checkpoint_sha256") != frozen_weights["sha256"]:
        raise RuntimeError("Raw receipt checkpoint SHA differs from frozen preregistration")
    if raw_model.get("training_recipe") != "original_environment_only":
        raise RuntimeError("Raw receipt does not disclose the original-only training recipe")
    raw_data = _mapping(raw.get("data"), label="raw receipt data")
    if raw_data.get("pair_count") != 256 or raw_data.get("condition_count") != 512:
        raise RuntimeError("Raw receipt does not cover the complete frozen query set")


def _descriptive_score(
    *, raw: Mapping[str, Any], raw_logical_path: str, metrics: Mapping[str, Any]
) -> dict[str, Any]:
    model = _mapping(raw["model"], label="raw receipt model")
    adapter = _mapping(model["adapter"], label="raw receipt adapter")
    metric_names = (
        "correct_future_rate",
        "correct_history_rate",
        "rule_switch_rate",
        "worst_strength_correct_future_rate",
        "joint_icl_pair_success_rate",
    )
    checkpoint = {
        "path": raw_logical_path,
        "checkpoint_sha256": adapter["checkpoint_sha256"],
        "training_seed": model["training_seed"],
        **{name: metrics[name] for name in metric_names},
        "passed": bool(_mapping(raw["gate"], label="raw receipt gate")["passed"]),
    }
    return {
        "schema_version": 1,
        "benchmark": _EXPECTED_BENCHMARK,
        "submission_kind": "descriptive_checkpoint_recovery",
        "status": "completed",
        "method_name": model["name"],
        "release_id": _mapping(raw["release"], label="raw receipt release")["release_id"],
        "checkpoints": [checkpoint],
        "aggregate": {
            name: {
                "mean": metrics[name],
                "sample_std": 0.0,
                "minimum": metrics[name],
                "maximum": metrics[name],
            }
            for name in metric_names
        },
        "decision": {
            "passed": False,
            "formal_method_claim": False,
            "reason": "single_checkpoint_is_descriptive_only",
        },
    }


def build_action_strength_lewm_recovery_receipt(
    *,
    repo_root: Path | None = None,
    raw_receipt: Path | str = DEFAULT_RAW_RECEIPT,
    preregistration: Path | str = DEFAULT_PREREGISTRATION,
    freeze_receipt: Path | str = DEFAULT_FREEZE_RECEIPT,
    release_config: Path | str = DEFAULT_ACTION_STRENGTH_RELEASE_CONFIG,
    output_path: Path | str = DEFAULT_RECOVERY_RECEIPT,
) -> dict[str, Any]:
    """Build, but do not write, the frozen float32 recovery receipt."""

    root = (repo_root or repository_root()).resolve()
    raw_path = _canonical_input(
        raw_receipt,
        repo_root=root,
        logical_path=DEFAULT_RAW_RECEIPT,
        label="raw receipt",
    )
    prereg_path = _canonical_input(
        preregistration,
        repo_root=root,
        logical_path=DEFAULT_PREREGISTRATION,
        label="preregistration",
    )
    freeze_path = _canonical_input(
        freeze_receipt,
        repo_root=root,
        logical_path=DEFAULT_FREEZE_RECEIPT,
        label="freeze receipt",
    )
    release_path = _canonical_input(
        release_config,
        repo_root=root,
        logical_path=ACTION_STRENGTH_RELEASE_LOGICAL_PATH,
        label="release config",
    )
    _, output_logical_path = validate_recovery_output_path(output_path, repo_root=root)

    _, _, raw, bindings = _load_frozen_bindings(
        root=root,
        preregistration=prereg_path,
        freeze_receipt=freeze_path,
        release_config=release_path,
        raw_receipt=raw_path,
    )
    release = load_action_strength_icl_release(release_path)
    _validate_raw_receipt(raw, release=release, bindings=bindings)
    records = _records(raw.get("records"))
    metrics, gate = _reconstruct_metrics(records, release=release)
    stored_metrics = _mapping(raw.get("metrics"), label="raw receipt metrics")
    scalar_comparison = _scalar_comparison(stored_metrics, metrics)
    stored_latent = _mapping(
        stored_metrics.get("latent_response"), label="stored latent response"
    )
    latent_comparison = _latent_comparison(
        stored_latent, _mapping(metrics["latent_response"], label="recomputed latent response")
    )
    gate_equal = gate == raw.get("gate")
    if not (
        scalar_comparison["all_float64_json_bitwise_equal"]
        and scalar_comparison["all_float32_loss_aggregates_bitwise_equal"]
        and latent_comparison["paired_latent_response_summaries_close"]
        and latent_comparison["all_float64_json_bitwise_equal"]
        and gate_equal
    ):
        raise RuntimeError("Float32 recovery did not reproduce the frozen result")

    # Rehash after all reads.  This makes the receipt evidence that the frozen
    # inputs remained unchanged throughout recovery, without writing to them.
    after = {
        "preregistration": _identity(
            prereg_path, logical_path=DEFAULT_PREREGISTRATION.as_posix()
        ),
        "freeze_receipt": _identity(
            freeze_path, logical_path=DEFAULT_FREEZE_RECEIPT.as_posix()
        ),
        "release_config": _identity(
            release_path,
            logical_path=ACTION_STRENGTH_RELEASE_LOGICAL_PATH.as_posix(),
        ),
        "raw_receipt": _identity(
            raw_path, logical_path=DEFAULT_RAW_RECEIPT.as_posix()
        ),
    }
    input_unchanged = all(after[name] == bindings[name] for name in after)
    if not input_unchanged:
        raise RuntimeError("A frozen recovery input changed while it was rescored")
    checkpoint = _mapping(bindings["checkpoint"], label="frozen checkpoint")
    checkpoint_weights = _mapping(checkpoint["weights"], label="frozen weights")
    adapter = _mapping(_mapping(raw["model"], label="raw model")["adapter"], label="raw adapter")
    source_score = root / "contextworld/benchmarks/action_strength_icl_score.py"
    paired_metric = root / "contextworld/benchmarks/paired_latent_response.py"
    recovery_module = root / "contextworld/benchmarks/action_strength_rescore_recovery.py"
    implementation = {
        "source_action_strength_metric": _identity(
            source_score,
            logical_path="contextworld/benchmarks/action_strength_icl_score.py",
        ),
        "paired_latent_response_metric": _identity(
            paired_metric,
            logical_path="contextworld/benchmarks/paired_latent_response.py",
        ),
        "recovery_module": _identity(
            recovery_module,
            logical_path="contextworld/benchmarks/action_strength_rescore_recovery.py",
        ),
    }
    return {
        "schema_version": 1,
        "recovery_id": RECOVERY_ID,
        "status": "completed",
        "scope": {
            "kind": "additive_independent_rescore_recovery",
            "capability_id": _EXPECTED_COMPONENT_ID,
            "family": "lewm",
            "model_evaluation_rerun_performed": False,
            "raw_receipt_rewritten": False,
            "frozen_preregistration_rewritten": False,
            "release_config_rewritten": False,
            "formal_scoreboard_mutated": False,
        },
        "output_policy": {
            "logical_path": output_logical_path,
            "namespace": RECOVERY_NAMESPACE.as_posix(),
            "exclusive_create_required": True,
            "overwrite_permitted": False,
        },
        "output": {
            "path": output_logical_path,
            "content_sha256_not_embedded_to_avoid_self_reference": True,
        },
        "bindings": {
            "frozen_preregistration": bindings["preregistration"],
            "freeze_receipt": bindings["freeze_receipt"],
            "release_config": bindings["release_config"],
            "raw_receipt": bindings["raw_receipt"],
            "checkpoint": {
                "checkpoint_id": _EXPECTED_CHECKPOINT_ID,
                "frozen_weights": checkpoint_weights,
                "raw_adapter_checkpoint_path": adapter.get("checkpoint"),
                "raw_adapter_checkpoint_sha256": adapter.get("checkpoint_sha256"),
                "raw_receipt_matches_frozen_checkpoint_sha256": (
                    adapter.get("checkpoint_sha256") == checkpoint_weights["sha256"]
                ),
            },
            "implementation": implementation,
        },
        "input_integrity": {
            "all_frozen_inputs_unchanged_during_recovery": input_unchanged,
            "identities_after_recovery_read": after,
        },
        "reconstruction": {
            "scalar_record_aggregation": {
                "mse_record_dtype": "float32",
                "mse_aggregate_operations": [
                    "correct_losses.mean()",
                    "other_losses.mean()",
                    "(other_losses - correct_losses).mean()",
                ],
                "boolean_decision_aggregation": "original NumPy boolean means",
            },
            "latent_record_aggregation": {
                "implementation": "summarize_paired_latent_response_records",
                "dtype": "float64_in_original_paired_latent_response_metric",
            },
            "metrics": metrics,
            "gate": gate,
        },
        "verification": {
            "passed": True,
            "scalar_metrics": scalar_comparison,
            "latent_metrics": latent_comparison,
            "gate_exact_equal": gate_equal,
            "stored_model_gate_passed": _mapping(raw["gate"], label="raw gate")[
                "passed"
            ],
            "recomputed_model_gate_passed": gate["passed"],
        },
        "descriptive_checkpoint_score": _descriptive_score(
            raw=raw,
            raw_logical_path=DEFAULT_RAW_RECEIPT.as_posix(),
            metrics=metrics,
        ),
    }


def recover_action_strength_lewm_baseline(
    *,
    repo_root: Path | None = None,
    raw_receipt: Path | str = DEFAULT_RAW_RECEIPT,
    preregistration: Path | str = DEFAULT_PREREGISTRATION,
    freeze_receipt: Path | str = DEFAULT_FREEZE_RECEIPT,
    release_config: Path | str = DEFAULT_ACTION_STRENGTH_RELEASE_CONFIG,
    output_path: Path | str = DEFAULT_RECOVERY_RECEIPT,
) -> dict[str, Any]:
    """Build and x-exclusively persist the narrow recovery receipt."""

    root = (repo_root or repository_root()).resolve()
    receipt = build_action_strength_lewm_recovery_receipt(
        repo_root=root,
        raw_receipt=raw_receipt,
        preregistration=preregistration,
        freeze_receipt=freeze_receipt,
        release_config=release_config,
        output_path=output_path,
    )
    output = write_recovery_receipt_exclusive(
        output_path, receipt, repo_root=root
    )
    if output.relative_to(root).as_posix() != receipt["output"]["path"]:
        raise AssertionError("Recovery receipt was written to the wrong path")
    return receipt


__all__ = [
    "DEFAULT_FREEZE_RECEIPT",
    "DEFAULT_PREREGISTRATION",
    "DEFAULT_RAW_RECEIPT",
    "DEFAULT_RECOVERY_RECEIPT",
    "RECOVERY_ID",
    "RECOVERY_NAMESPACE",
    "build_action_strength_lewm_recovery_receipt",
    "recover_action_strength_lewm_baseline",
    "validate_recovery_output_path",
    "write_recovery_receipt_exclusive",
]
