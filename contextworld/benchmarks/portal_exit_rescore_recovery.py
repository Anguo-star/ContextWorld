"""Immutable float32 recovery for the frozen Portal Exit baseline receipts.

The original Portal Exit evaluator obtains per-pair MSE values from float32
latent arrays and then keeps those values in float32 through each loss
aggregate.  The generic record rescorer instead constructed its loss arrays
from JSON with NumPy's default float64 dtype.  That makes only the three loss
aggregates differ; the boolean rates, paired-bootstrap bounds, latent-response
summary, and prediction gate are otherwise already reproducible from the
frozen records.

This is deliberately a narrow, read-only recovery.  It accepts only the two
frozen original-baseline Portal Exit receipts, binds each to the frozen
preregistration/freeze/release/checkpoint identities, and writes only a new
receipt under the dedicated additive recovery namespace with O_EXCL.  It does
not load a checkpoint, initialise an adapter, invoke a GPU, or rewrite an
existing raw or rescore receipt.
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

from contextworld.benchmarks.paired_latent_response import (
    paired_latent_response_summaries_close,
    summarize_paired_latent_response_records,
)
from contextworld.benchmarks.portal_exit_icl_data import (
    load_portal_exit_icl_release,
)
from contextworld.paths import repository_root


RECOVERY_NAMESPACE = Path(
    "artifacts/evaluation/original_baseline_matrix_v1/rescore_recovery/"
    "contextworld-portal-exit"
)
DEFAULT_PREREGISTRATION = Path(
    "configs/benchmark/contextworld_original_baseline_completion_prereg_v1.yaml"
)
DEFAULT_FREEZE_RECEIPT = Path(
    "configs/benchmark/contextworld_original_baseline_completion_freeze_v1.json"
)
PORTAL_EXIT_RELEASE_LOGICAL_PATH = Path(
    "configs/benchmark/tworoom_portal_exit_icl_release_v1.yaml"
)
DEFAULT_RAW_RECEIPTS = {
    "lewm": Path(
        "artifacts/evaluation/original_baseline_matrix_v1/"
        "contextworld-portal-exit/lewm.json"
    ),
    "pldm": Path(
        "artifacts/evaluation/original_baseline_matrix_v1/"
        "contextworld-portal-exit/pldm.json"
    ),
}
DEFAULT_LEGACY_FAILED_RESCORES = {
    "lewm": Path(
        "artifacts/evaluation/original_baseline_matrix_v1/"
        "contextworld-portal-exit/lewm.rescore.json"
    ),
    "pldm": Path(
        "artifacts/evaluation/original_baseline_matrix_v1/"
        "contextworld-portal-exit/pldm.rescore.json"
    ),
}
DEFAULT_RECOVERY_RECEIPTS = {
    "lewm": RECOVERY_NAMESPACE / "lewm_float32_rescore_recovery_v1.json",
    "pldm": RECOVERY_NAMESPACE / "pldm_float32_rescore_recovery_v1.json",
}
RECOVERY_IDS = {
    family: (
        "contextworld_original_baseline_matrix_v1_"
        f"portal_exit_{family}_float32_rescore_recovery_v1"
    )
    for family in DEFAULT_RAW_RECEIPTS
}

_EXPECTED_BENCHMARK = "tworoom_history3_portal_exit_icl_v1"
_EXPECTED_COMPONENT_ID = "contextworld-portal-exit"
_EXPECTED_PAIR_COUNT = 256
_BOOTSTRAP_RESAMPLES = 10_000
_BOOTSTRAP_SEED = 2026080205
_SCALAR_METRIC_NAMES = (
    "pair_count",
    "decision_count",
    "correct_future_rate",
    "correct_history_rate",
    "context_switch_rate",
    "near_border_correct_future_rate",
    "farther_from_border_correct_future_rate",
    "worst_exit_correct_future_rate",
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
_SCALAR_SEMANTICS = {
    "pair_count": "integer_exact",
    "decision_count": "integer_exact",
    "correct_future_rate": "np.bool_.mean() -> float64",
    "correct_history_rate": "np.bool_.mean() -> float64",
    "context_switch_rate": "np.bool_.mean() -> float64",
    "near_border_correct_future_rate": "np.bool_.mean() -> float64",
    "farther_from_border_correct_future_rate": "np.bool_.mean() -> float64",
    "worst_exit_correct_future_rate": "min(float64_boolean_means)",
    "correct_future_mse_mean": "float32_loss_array.mean() -> float32",
    "other_future_mse_mean": "float32_loss_array.mean() -> float32",
    "other_minus_correct_mse_margin_mean": (
        "(float32_other_loss_array - float32_correct_loss_array).mean() -> float32"
    ),
    "current_frame_only_accuracy_bound": "literal_float64",
    "joint_icl_pair_success_rate": "np.bool_.mean() -> float64",
}
_FAMILY_SPECS = {
    "lewm": {
        "checkpoint_id": "tworoom_lewm_original",
        "model_name": "tworoom_lewm_original",
        "adapter_id": "stable_worldmodel_lewm_portal_exit_v1",
        "adapter_class": (
            "contextworld.benchmarks.adapters."
            "StableWorldModelLeWMPortalExitAdapter"
        ),
    },
    "pldm": {
        "checkpoint_id": "tworoom_pldm_original",
        "model_name": "tworoom_pldm_original",
        "adapter_id": "stable_worldmodel_pldm_portal_exit_v1",
        "adapter_class": (
            "contextworld.benchmarks.adapters."
            "StableWorldModelPLDMPortalExitAdapter"
        ),
    },
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


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _family_spec(family: str) -> Mapping[str, str]:
    try:
        return _FAMILY_SPECS[family]
    except KeyError as error:
        allowed = ", ".join(sorted(_FAMILY_SPECS))
        raise ValueError(f"family must be one of {allowed}, got {family!r}") from error


def _canonical_input(
    value: Path | str,
    *,
    repo_root: Path,
    logical_path: Path,
    label: str,
) -> Path:
    expected = (repo_root / logical_path).resolve()
    observed = Path(value).expanduser()
    if not observed.is_absolute():
        observed = repo_root / observed
    observed = observed.resolve()
    if observed != expected:
        raise ValueError(
            f"{label} must be the frozen canonical input {expected}, got {observed}"
        )
    if not expected.is_file():
        raise FileNotFoundError(f"{label} is missing: {expected}")
    return expected


def validate_recovery_output_path(
    output_path: Path | str,
    *,
    repo_root: Path | None = None,
) -> tuple[Path, str]:
    """Require an additive JSON receipt below the Portal-only namespace."""

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
    """Persist one recovery receipt without replacing a prior result."""

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
        # This invocation created the path using O_EXCL; never leave a partial
        # receipt behind if a write fails.
        output.unlink(missing_ok=True)
        raise
    return output


def _records(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        raise ValueError("Raw Portal Exit receipt records must be a list")
    records = [_mapping(row, label=f"record {index}") for index, row in enumerate(value)]
    if len(records) != _EXPECTED_PAIR_COUNT:
        raise ValueError(
            f"Frozen Portal Exit recovery requires {_EXPECTED_PAIR_COUNT} records"
        )
    identifiers = [str(row.get("pair_id", "")) for row in records]
    if not all(identifiers) or len(set(identifiers)) != len(identifiers):
        raise ValueError("Raw Portal Exit record IDs must be nonempty and unique")
    return records


def _section(record: Mapping[str, Any], *, index: int, name: str) -> Mapping[str, Any]:
    return _mapping(record.get(name), label=f"record {index}.{name}")


def _boolean_values(
    records: list[Mapping[str, Any]], *, section: str | None, field: str
) -> np.ndarray:
    values: list[bool] = []
    for index, row in enumerate(records):
        source = _section(row, index=index, name=section) if section else row
        value = source.get(field)
        label = f"record {index}.{section + '.' if section else ''}{field}"
        if type(value) is not bool:
            raise ValueError(f"{label} must be boolean")
        values.append(value)
    return np.asarray(values, dtype=bool)


def _numeric_values(
    records: list[Mapping[str, Any]],
    *,
    section: str,
    field: str,
    dtype: np.dtype[Any],
) -> np.ndarray:
    values: list[float] = []
    for index, row in enumerate(records):
        value = _section(row, index=index, name=section).get(field)
        label = f"record {index}.{section}.{field}"
        if isinstance(value, bool):
            raise ValueError(f"{label} must be numeric")
        try:
            number = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{label} must be numeric") from error
        if not np.isfinite(number):
            raise ValueError(f"{label} must be finite")
        values.append(number)
    result = np.asarray(values, dtype=dtype)
    if not np.isfinite(result).all():
        raise ValueError(f"{section}.{field} cannot be represented as finite {dtype}")
    return result


def _loss_arrays(
    records: list[Mapping[str, Any]], *, dtype: np.dtype[Any]
) -> tuple[np.ndarray, np.ndarray]:
    """Recreate the evaluator's correct/other loss-array ordering."""

    correct_losses = np.concatenate(
        [
            _numeric_values(
                records,
                section="near_border",
                field="correct_future_mse",
                dtype=dtype,
            ),
            _numeric_values(
                records,
                section="farther_from_border",
                field="correct_future_mse",
                dtype=dtype,
            ),
        ]
    )
    other_losses = np.concatenate(
        [
            _numeric_values(
                records,
                section="near_border",
                field="other_future_mse",
                dtype=dtype,
            ),
            _numeric_values(
                records,
                section="farther_from_border",
                field="other_future_mse",
                dtype=dtype,
            ),
        ]
    )
    if correct_losses.dtype != dtype or other_losses.dtype != dtype:
        raise AssertionError("Portal Exit loss aggregation dtype was lost")
    return correct_losses, other_losses


def _bootstrap_uncertainty(
    *,
    near_future: np.ndarray,
    farther_future: np.ndarray,
    near_history: np.ndarray,
    farther_history: np.ndarray,
    switch: np.ndarray,
) -> dict[str, Any]:
    """Copy the evaluator's seeded paired-bootstrap operation exactly."""

    pair_count = len(near_future)
    rng = np.random.default_rng(_BOOTSTRAP_SEED)
    draws = rng.integers(0, pair_count, size=(_BOOTSTRAP_RESAMPLES, pair_count))
    near_future_draws = near_future[draws].mean(axis=1)
    farther_future_draws = farther_future[draws].mean(axis=1)
    near_history_draws = near_history[draws].mean(axis=1)
    farther_history_draws = farther_history[draws].mean(axis=1)
    bootstrap = {
        "correct_future_rate": 0.5 * (near_future_draws + farther_future_draws),
        "correct_history_rate": 0.5 * (near_history_draws + farther_history_draws),
        "context_switch_rate": switch[draws].mean(axis=1),
        "worst_exit_correct_future_rate": np.minimum(
            near_future_draws, farther_future_draws
        ),
    }
    return {
        "method": "paired_query_bootstrap",
        "unit": "portal_exit_query_pair",
        "resamples": _BOOTSTRAP_RESAMPLES,
        "confidence_level": 0.95,
        "random_seed": _BOOTSTRAP_SEED,
        "lower_bounds": {
            name: float(np.quantile(values, 0.025))
            for name, values in bootstrap.items()
        },
    }


def _latent_records(records: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    names = (
        "response_gain",
        "response_alignment",
        "normalized_response_error",
        "target_response_mse",
        "prediction_response_mse",
    )
    result: list[dict[str, Any]] = []
    for index, row in enumerate(records):
        response = _section(row, index=index, name="latent_response")
        reconstructed: dict[str, Any] = {"pair_id": row["pair_id"]}
        for name in names:
            value = response.get(name)
            if isinstance(value, bool):
                raise ValueError(f"record {index}.latent_response.{name} must be numeric")
            try:
                number = float(value)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"record {index}.latent_response.{name} must be numeric"
                ) from error
            if not np.isfinite(number):
                raise ValueError(f"record {index}.latent_response.{name} must be finite")
            reconstructed[name] = number
        success = response.get("calibrated_response_success")
        if type(success) is not bool:
            raise ValueError(
                "record "
                f"{index}.latent_response.calibrated_response_success must be boolean"
            )
        reconstructed["calibrated_response_success"] = success
        result.append(reconstructed)
    return result


def _reconstruct_gate(metrics: Mapping[str, Any], *, release: Mapping[str, Any]) -> dict[str, Any]:
    """Independently apply the frozen Portal Exit gate semantics."""

    thresholds = _mapping(
        _mapping(release.get("scoring"), label="release scoring").get(
            "hidden_future_prediction"
        ),
        label="release hidden_future_prediction scoring",
    ).get("gates")
    thresholds = _mapping(thresholds, label="Portal Exit gate thresholds")
    checks = {
        name: metrics[name] >= float(thresholds[f"{name}_minimum"])
        for name in (
            "correct_future_rate",
            "correct_history_rate",
            "context_switch_rate",
            "worst_exit_correct_future_rate",
        )
    }
    response = _mapping(metrics.get("latent_response"), label="latent response")
    separation = _mapping(
        response.get("target_latent_separation"), label="target latent separation"
    )
    response_gain = float(response.get("response_gain", float("-inf")))
    normalized_error = float(response.get("normalized_response_error", float("inf")))
    checks.update(
        {
            "target_latent_separation": bool(
                thresholds.get("target_latent_separation_required") is True
                and separation.get("passed") is True
            ),
            "response_gain": bool(
                np.isfinite(response_gain)
                and response_gain >= float(thresholds["response_gain_minimum"])
            ),
            "normalized_response_error": bool(
                np.isfinite(normalized_error)
                and normalized_error
                < float(thresholds["normalized_response_error_strict_maximum"])
            ),
        }
    )
    lower_minimum = _mapping(
        thresholds.get("bootstrap_lower_bound_minimum", {}),
        label="bootstrap lower-bound minimums",
    )
    uncertainty = _mapping(metrics.get("uncertainty"), label="uncertainty")
    lower_bounds = _mapping(uncertainty.get("lower_bounds"), label="lower bounds")
    uncertainty_checks = {
        name: float(lower_bounds.get(name, float("-inf"))) >= float(minimum)
        for name, minimum in lower_minimum.items()
    }
    return {
        "checks": checks,
        "uncertainty_checks": uncertainty_checks,
        "passed": all(checks.values()) and all(uncertainty_checks.values()),
    }


def _reconstruct_metrics(
    records: list[Mapping[str, Any]], *, release: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Rebuild every Portal Exit metric using the original dtype semantics."""

    near_future = _boolean_values(
        records, section="near_border", field="correct_future"
    )
    farther_future = _boolean_values(
        records, section="farther_from_border", field="correct_future"
    )
    near_history = _boolean_values(
        records, section="near_border", field="correct_history"
    )
    farther_history = _boolean_values(
        records, section="farther_from_border", field="correct_history"
    )
    switch = _boolean_values(records, section=None, field="context_switch_correct")

    # `_mse` in portal_exit_prediction_metrics returns float32 when its latent
    # inputs are float32.  The source evaluator then concatenates and means
    # those arrays without a dtype override, so all three loss aggregates stay
    # float32 until the final JSON-safe `float(...)` conversion.
    correct_losses, other_losses = _loss_arrays(records, dtype=np.dtype(np.float32))
    latent_records = _latent_records(records)
    latent_response = summarize_paired_latent_response_records(latent_records)
    calibrated_response = np.asarray(
        [row["calibrated_response_success"] for row in latent_records], dtype=bool
    )
    joint = (
        near_future
        & farther_future
        & near_history
        & farther_history
        & calibrated_response
    )
    for index, row in enumerate(records):
        if row.get("joint_icl_pair_success") != bool(joint[index]):
            raise RuntimeError(
                "Raw Portal Exit joint ICL record disagrees with reconstructed "
                f"inputs at pair {row['pair_id']}"
            )

    metrics: dict[str, Any] = {
        "pair_count": len(records),
        "decision_count": 2 * len(records),
        "correct_future_rate": float(
            np.concatenate([near_future, farther_future]).mean()
        ),
        "correct_history_rate": float(
            np.concatenate([near_history, farther_history]).mean()
        ),
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
        "uncertainty": _bootstrap_uncertainty(
            near_future=near_future,
            farther_future=farther_future,
            near_history=near_history,
            farther_history=farther_history,
            switch=switch,
        ),
        "latent_response": latent_response,
        "joint_icl_pair_success_rate": float(joint.mean()),
    }
    return metrics, _reconstruct_gate(metrics, release=release)


def _legacy_float64_loss_metrics(records: list[Mapping[str, Any]]) -> dict[str, float]:
    """Recreate the exact coercion used by the failed generic rescorer."""

    correct_losses, other_losses = _loss_arrays(records, dtype=np.dtype(np.float64))
    return {
        "correct_future_mse_mean": float(correct_losses.mean()),
        "other_future_mse_mean": float(other_losses.mean()),
        "other_minus_correct_mse_margin_mean": float(
            (other_losses - correct_losses).mean()
        ),
    }


def _float64_bits(value: float) -> str:
    return f"0x{struct.unpack('>Q', struct.pack('>d', float(value)))[0]:016x}"


def _float32_bits(value: float) -> str:
    bits = np.asarray(value, dtype=np.float32).view(np.uint32).item()
    return f"0x{bits:08x}"


def _numeric_comparison(
    *,
    path: str,
    stored: Any,
    recomputed: Any,
    legacy: Any,
    aggregation_semantics: str | None = None,
) -> dict[str, Any]:
    if isinstance(stored, bool) or isinstance(recomputed, bool) or isinstance(legacy, bool):
        return {
            "path": path,
            "comparison": "boolean_exact",
            "stored": stored,
            "recomputed": recomputed,
            "legacy_reconstructed": legacy,
            "recovery_exact": stored == recomputed,
            "legacy_exact": stored == legacy,
        }
    if isinstance(stored, str) or isinstance(recomputed, str) or isinstance(legacy, str):
        return {
            "path": path,
            "comparison": "exact",
            "stored": stored,
            "recomputed": recomputed,
            "legacy_reconstructed": legacy,
            "recovery_exact": stored == recomputed,
            "legacy_exact": stored == legacy,
        }
    if isinstance(stored, int) and isinstance(recomputed, int) and isinstance(legacy, int):
        return {
            "path": path,
            "comparison": "integer_exact",
            "stored": stored,
            "recomputed": recomputed,
            "legacy_reconstructed": legacy,
            "recovery_exact": stored == recomputed,
            "legacy_exact": stored == legacy,
            "recomputed_minus_stored": recomputed - stored,
            "legacy_minus_stored": legacy - stored,
            **(
                {"source_aggregation_semantics": aggregation_semantics}
                if aggregation_semantics
                else {}
            ),
        }
    try:
        stored_float = float(stored)
        recomputed_float = float(recomputed)
        legacy_float = float(legacy)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"Unsupported numeric comparison at {path}") from error
    row: dict[str, Any] = {
        "path": path,
        "comparison": "float64_json_bitwise",
        "stored": stored_float,
        "recomputed": recomputed_float,
        "legacy_reconstructed": legacy_float,
        "stored_float64_bits": _float64_bits(stored_float),
        "recomputed_float64_bits": _float64_bits(recomputed_float),
        "legacy_float64_bits": _float64_bits(legacy_float),
        "recovery_exact": _float64_bits(stored_float) == _float64_bits(recomputed_float),
        "legacy_exact": _float64_bits(stored_float) == _float64_bits(legacy_float),
        "recomputed_minus_stored": recomputed_float - stored_float,
        "legacy_minus_stored": legacy_float - stored_float,
    }
    if aggregation_semantics:
        row["source_aggregation_semantics"] = aggregation_semantics
    return row


def _nested_comparison_leaves(
    stored: Any,
    recomputed: Any,
    legacy: Any,
    *,
    path: str,
    aggregation_semantics: str | None = None,
) -> list[dict[str, Any]]:
    if isinstance(stored, Mapping) or isinstance(recomputed, Mapping) or isinstance(legacy, Mapping):
        if not (
            isinstance(stored, Mapping)
            and isinstance(recomputed, Mapping)
            and isinstance(legacy, Mapping)
        ):
            raise RuntimeError(f"Mapping mismatch at {path}")
        if set(stored) != set(recomputed) or set(stored) != set(legacy):
            raise RuntimeError(f"Nested keys differ at {path}")
        leaves: list[dict[str, Any]] = []
        for name in sorted(stored):
            leaves.extend(
                _nested_comparison_leaves(
                    stored[name],
                    recomputed[name],
                    legacy[name],
                    path=f"{path}.{name}",
                    aggregation_semantics=aggregation_semantics,
                )
            )
        return leaves
    return [
        _numeric_comparison(
            path=path,
            stored=stored,
            recomputed=recomputed,
            legacy=legacy,
            aggregation_semantics=aggregation_semantics,
        )
    ]


def _all_exact(rows: list[Mapping[str, Any]], name: str) -> bool:
    return all(bool(row[name]) for row in rows)


def _scalar_comparison(
    stored: Mapping[str, Any],
    recomputed: Mapping[str, Any],
    legacy: Mapping[str, Any],
) -> dict[str, Any]:
    expected_keys = set(_SCALAR_METRIC_NAMES) | {"uncertainty", "latent_response"}
    if set(stored) != expected_keys or set(recomputed) != expected_keys or set(legacy) != expected_keys:
        raise RuntimeError("Portal Exit metric keys differ from the frozen schema")
    rows = {
        name: _numeric_comparison(
            path=name,
            stored=stored[name],
            recomputed=recomputed[name],
            legacy=legacy[name],
            aggregation_semantics=_SCALAR_SEMANTICS[name],
        )
        for name in _SCALAR_METRIC_NAMES
    }
    float32_exact = all(
        _float32_bits(float(stored[name])) == _float32_bits(float(recomputed[name]))
        for name in _FLOAT32_LOSS_METRIC_NAMES
    )
    legacy_mismatches = sorted(
        name for name, row in rows.items() if not bool(row["legacy_exact"])
    )
    return {
        "comparison_scope": "all top-level non-nested Portal Exit scalar metrics",
        "all_recovery_float64_json_bitwise_equal": all(
            bool(row["recovery_exact"]) for row in rows.values()
        ),
        "all_float32_loss_aggregates_bitwise_equal": float32_exact,
        "legacy_mismatch_metric_names": legacy_mismatches,
        "legacy_only_float32_loss_aggregation_mismatches": (
            legacy_mismatches == sorted(_FLOAT32_LOSS_METRIC_NAMES)
        ),
        "metrics": rows,
    }


def _nested_comparison(
    stored: Mapping[str, Any],
    recomputed: Mapping[str, Any],
    legacy: Mapping[str, Any],
    *,
    path: str,
    aggregation_semantics: str,
) -> dict[str, Any]:
    leaves = _nested_comparison_leaves(
        stored,
        recomputed,
        legacy,
        path=path,
        aggregation_semantics=aggregation_semantics,
    )
    return {
        "aggregation_semantics": aggregation_semantics,
        "all_recovery_exact": _all_exact(leaves, "recovery_exact"),
        "all_legacy_exact": _all_exact(leaves, "legacy_exact"),
        "leaves": leaves,
    }


def _load_frozen_bindings(
    *,
    family: str,
    preregistration: Path,
    freeze_receipt: Path,
    release_config: Path,
    raw_receipt: Path,
    legacy_failed_rescore: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    spec = _family_spec(family)
    prereg_identity = _identity(
        preregistration, logical_path=DEFAULT_PREREGISTRATION.as_posix()
    )
    freeze_identity = _identity(
        freeze_receipt, logical_path=DEFAULT_FREEZE_RECEIPT.as_posix()
    )
    release_identity = _identity(
        release_config, logical_path=PORTAL_EXIT_RELEASE_LOGICAL_PATH.as_posix()
    )
    raw_identity = _identity(
        raw_receipt, logical_path=DEFAULT_RAW_RECEIPTS[family].as_posix()
    )
    legacy_identity = _identity(
        legacy_failed_rescore,
        logical_path=DEFAULT_LEGACY_FAILED_RESCORES[family].as_posix(),
    )

    prereg = dict(
        _mapping(
            yaml.safe_load(preregistration.read_text(encoding="utf-8")),
            label="frozen original-baseline preregistration",
        )
    )
    if (
        prereg.get("preregistration_id")
        != "contextworld_original_baseline_completion_v1"
        or prereg.get("status") != "frozen_before_new_baseline_scoring"
    ):
        raise RuntimeError("Unexpected original-baseline preregistration")
    freeze = dict(
        _mapping(
            json.loads(freeze_receipt.read_text(encoding="utf-8")),
            label="original-baseline freeze receipt",
        )
    )
    if (
        freeze.get("freeze_id")
        != "contextworld_original_baseline_completion_freeze_v1"
        or freeze.get("status") != "frozen_before_new_baseline_scoring"
        or freeze.get("preregistration") != prereg_identity
    ):
        raise RuntimeError("Frozen original-baseline preregistration binding failed")

    components = {
        str(_mapping(row, label="prereg component").get("capability_id")): row
        for row in prereg.get("components", [])
    }
    component = _mapping(
        components.get(_EXPECTED_COMPONENT_ID), label="Portal Exit component"
    )
    if component.get("release_config") != release_identity:
        raise RuntimeError("Frozen Portal Exit release identity binding failed")
    cells = [
        _mapping(row, label="prereg ICL cell")
        for row in prereg.get("icl_cells", [])
        if _mapping(row, label="prereg ICL cell").get("capability_id")
        == _EXPECTED_COMPONENT_ID
        and _mapping(row, label="prereg ICL cell").get("family") == family
    ]
    if len(cells) != 1:
        raise RuntimeError("Frozen Portal Exit ICL cell binding is ambiguous")
    cell = cells[0]
    if (
        cell.get("checkpoint_id") != spec["checkpoint_id"]
        or cell.get("output") != DEFAULT_RAW_RECEIPTS[family].as_posix()
        or cell.get("formal_scoreboard_eligible") is not False
    ):
        raise RuntimeError("Frozen Portal Exit ICL cell does not match recovery scope")
    checkpoints = {
        str(_mapping(row, label="prereg checkpoint").get("checkpoint_id")): row
        for row in prereg.get("checkpoints", [])
    }
    checkpoint = _mapping(
        checkpoints.get(spec["checkpoint_id"]), label="frozen Portal Exit checkpoint"
    )
    if (
        checkpoint.get("environment") != "tworoom"
        or checkpoint.get("family") != family
        or checkpoint.get("contextworld_capability_training_used") is not False
    ):
        raise RuntimeError("Recovery checkpoint is not the frozen original-only one")
    weights = _mapping(checkpoint.get("weights"), label="frozen checkpoint weights")
    if not isinstance(weights.get("sha256"), str) or len(weights["sha256"]) != 64:
        raise RuntimeError("Frozen checkpoint SHA-256 is invalid")

    raw = dict(
        _mapping(
            json.loads(raw_receipt.read_text(encoding="utf-8")),
            label="raw Portal Exit receipt",
        )
    )
    legacy = dict(
        _mapping(
            json.loads(legacy_failed_rescore.read_text(encoding="utf-8")),
            label="legacy failed Portal Exit rescore receipt",
        )
    )
    return prereg, raw, legacy, {
        "preregistration": prereg_identity,
        "freeze_receipt": freeze_identity,
        "release_config": release_identity,
        "raw_receipt": raw_identity,
        "legacy_failed_rescore": legacy_identity,
        "checkpoint": dict(checkpoint),
    }


def _validate_raw_and_legacy(
    *,
    family: str,
    raw: Mapping[str, Any],
    legacy: Mapping[str, Any],
    release: Mapping[str, Any],
    bindings: Mapping[str, Any],
) -> None:
    spec = _family_spec(family)
    if (
        raw.get("schema_version") != 1
        or raw.get("benchmark") != _EXPECTED_BENCHMARK
        or raw.get("submission_kind") != "single_checkpoint"
        or raw.get("status") != "completed"
    ):
        raise RuntimeError("Raw receipt is not the frozen Portal Exit result")
    expected_release = {
        "release_id": release["release_id"],
        "release_config_sha256": bindings["release_config"]["sha256"],
        "data_manifest_sha256": release["data"]["manifest_sha256"],
    }
    if _mapping(raw.get("release"), label="raw receipt release") != expected_release:
        raise RuntimeError("Raw receipt does not bind the frozen Portal Exit release")
    raw_model = _mapping(raw.get("model"), label="raw receipt model")
    raw_adapter = _mapping(raw_model.get("adapter"), label="raw receipt adapter")
    frozen_weights = _mapping(
        _mapping(bindings["checkpoint"], label="frozen checkpoint").get("weights"),
        label="frozen checkpoint weights",
    )
    if (
        raw_model.get("name") != spec["model_name"]
        or raw_model.get("training_recipe") != "original_environment_only"
        or raw_adapter.get("adapter_id") != spec["adapter_id"]
        or raw_adapter.get("adapter_class") != spec["adapter_class"]
        or raw_adapter.get("checkpoint_sha256") != frozen_weights["sha256"]
    ):
        raise RuntimeError("Raw receipt model does not bind the frozen checkpoint")
    raw_data = _mapping(raw.get("data"), label="raw receipt data")
    if (
        raw_data.get("pair_count") != _EXPECTED_PAIR_COUNT
        or raw_data.get("condition_count") != 2 * _EXPECTED_PAIR_COUNT
        or raw_data.get("history_tokens") != 3
        or raw_data.get("online_environment_calls") != 0
    ):
        raise RuntimeError("Raw receipt does not cover the complete offline query set")

    expected_legacy_checks = {
        "release_identity_matches": True,
        "metrics_reconstructed_from_records": False,
        "latent_response_reconstructed_from_records": True,
        "gate_recomputed": True,
    }
    if (
        legacy.get("schema_version") != 1
        or legacy.get("benchmark") != _EXPECTED_BENCHMARK
        or legacy.get("submission_kind") != "single_checkpoint_record_rescore"
        or legacy.get("status") != "failed"
        or legacy.get("record_count") != _EXPECTED_PAIR_COUNT
        or legacy.get("release_config_sha256") != bindings["release_config"]["sha256"]
        or _mapping(legacy.get("raw_receipt"), label="legacy raw receipt")
        != {
            "path": bindings["raw_receipt"]["path"],
            "sha256": bindings["raw_receipt"]["sha256"],
        }
        or _mapping(legacy.get("checks"), label="legacy checks") != expected_legacy_checks
    ):
        raise RuntimeError("Legacy Portal Exit failure receipt no longer has its known shape")


def _descriptive_checkpoint_score(
    *, raw: Mapping[str, Any], raw_logical_path: str, metrics: Mapping[str, Any]
) -> dict[str, Any]:
    model = _mapping(raw["model"], label="raw receipt model")
    adapter = _mapping(model["adapter"], label="raw receipt adapter")
    metric_names = (
        "correct_future_rate",
        "correct_history_rate",
        "context_switch_rate",
        "worst_exit_correct_future_rate",
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
        "release_id": _mapping(raw["release"], label="raw receipt release")[
            "release_id"
        ],
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


def build_portal_exit_rescore_recovery_receipt(
    *,
    family: str,
    repo_root: Path | None = None,
    raw_receipt: Path | str | None = None,
    legacy_failed_rescore: Path | str | None = None,
    preregistration: Path | str = DEFAULT_PREREGISTRATION,
    freeze_receipt: Path | str = DEFAULT_FREEZE_RECEIPT,
    release_config: Path | str = PORTAL_EXIT_RELEASE_LOGICAL_PATH,
    output_path: Path | str | None = None,
) -> dict[str, Any]:
    """Build, but do not write, one frozen Portal Exit recovery receipt."""

    _family_spec(family)
    root = (repo_root or repository_root()).resolve()
    raw_logical_path = DEFAULT_RAW_RECEIPTS[family]
    legacy_logical_path = DEFAULT_LEGACY_FAILED_RESCORES[family]
    raw_path = _canonical_input(
        raw_receipt or raw_logical_path,
        repo_root=root,
        logical_path=raw_logical_path,
        label="raw receipt",
    )
    legacy_path = _canonical_input(
        legacy_failed_rescore or legacy_logical_path,
        repo_root=root,
        logical_path=legacy_logical_path,
        label="legacy failed rescore receipt",
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
        logical_path=PORTAL_EXIT_RELEASE_LOGICAL_PATH,
        label="Portal Exit release config",
    )
    requested_output = output_path or DEFAULT_RECOVERY_RECEIPTS[family]
    _, output_logical_path = validate_recovery_output_path(
        requested_output, repo_root=root
    )

    _, raw, legacy, bindings = _load_frozen_bindings(
        family=family,
        preregistration=prereg_path,
        freeze_receipt=freeze_path,
        release_config=release_path,
        raw_receipt=raw_path,
        legacy_failed_rescore=legacy_path,
    )
    release = load_portal_exit_icl_release(release_path)
    _validate_raw_and_legacy(
        family=family,
        raw=raw,
        legacy=legacy,
        release=release,
        bindings=bindings,
    )
    records = _records(raw.get("records"))
    metrics, gate = _reconstruct_metrics(records, release=release)
    stored_metrics = _mapping(raw.get("metrics"), label="raw receipt metrics")
    legacy_metrics = _mapping(
        legacy.get("reconstructed_metrics"), label="legacy reconstructed metrics"
    )
    scalar_comparison = _scalar_comparison(stored_metrics, metrics, legacy_metrics)
    uncertainty_comparison = _nested_comparison(
        _mapping(stored_metrics["uncertainty"], label="stored uncertainty"),
        _mapping(metrics["uncertainty"], label="recomputed uncertainty"),
        _mapping(legacy_metrics["uncertainty"], label="legacy uncertainty"),
        path="uncertainty",
        aggregation_semantics=(
            "seeded NumPy Generator paired bootstrap; bool means and quantiles are float64"
        ),
    )
    latent_comparison = _nested_comparison(
        _mapping(stored_metrics["latent_response"], label="stored latent response"),
        _mapping(metrics["latent_response"], label="recomputed latent response"),
        _mapping(legacy_metrics["latent_response"], label="legacy latent response"),
        path="latent_response",
        aggregation_semantics=(
            "summarize_paired_latent_response_records uses float64 response geometry"
        ),
    )
    legacy_float64_losses = _legacy_float64_loss_metrics(records)
    if any(
        _float64_bits(legacy_metrics[name]) != _float64_bits(value)
        for name, value in legacy_float64_losses.items()
    ):
        raise RuntimeError(
            "Legacy Portal Exit rescore differs from the documented float64 "
            "loss coercion"
        )
    if not (
        scalar_comparison["all_recovery_float64_json_bitwise_equal"]
        and scalar_comparison["all_float32_loss_aggregates_bitwise_equal"]
        and scalar_comparison["legacy_only_float32_loss_aggregation_mismatches"]
        and uncertainty_comparison["all_recovery_exact"]
        and uncertainty_comparison["all_legacy_exact"]
        and latent_comparison["all_recovery_exact"]
        and latent_comparison["all_legacy_exact"]
        and paired_latent_response_summaries_close(
            _mapping(stored_metrics["latent_response"], label="stored latent response"),
            _mapping(metrics["latent_response"], label="recomputed latent response"),
        )
        and gate == raw.get("gate")
        and gate == legacy.get("reconstructed_gate")
    ):
        raise RuntimeError("Float32 recovery did not reproduce the frozen Portal Exit result")

    # Rehash after all reads to make the receipt evidence that every frozen
    # input, including the retained failure receipt, remained unchanged.
    after = {
        "preregistration": _identity(
            prereg_path, logical_path=DEFAULT_PREREGISTRATION.as_posix()
        ),
        "freeze_receipt": _identity(
            freeze_path, logical_path=DEFAULT_FREEZE_RECEIPT.as_posix()
        ),
        "release_config": _identity(
            release_path, logical_path=PORTAL_EXIT_RELEASE_LOGICAL_PATH.as_posix()
        ),
        "raw_receipt": _identity(raw_path, logical_path=raw_logical_path.as_posix()),
        "legacy_failed_rescore": _identity(
            legacy_path, logical_path=legacy_logical_path.as_posix()
        ),
    }
    input_unchanged = all(after[name] == bindings[name] for name in after)
    if not input_unchanged:
        raise RuntimeError("A frozen Portal Exit recovery input changed while rescoring")

    checkpoint = _mapping(bindings["checkpoint"], label="frozen checkpoint")
    checkpoint_weights = _mapping(checkpoint["weights"], label="frozen weights")
    raw_adapter = _mapping(
        _mapping(raw["model"], label="raw model")["adapter"], label="raw adapter"
    )
    implementation = {
        "source_portal_exit_evaluator": _identity(
            root / "contextworld/benchmarks/portal_exit_icl_score.py",
            logical_path="contextworld/benchmarks/portal_exit_icl_score.py",
        ),
        "paired_latent_response_metric": _identity(
            root / "contextworld/benchmarks/paired_latent_response.py",
            logical_path="contextworld/benchmarks/paired_latent_response.py",
        ),
        "recovery_module": _identity(
            root / "contextworld/benchmarks/portal_exit_rescore_recovery.py",
            logical_path="contextworld/benchmarks/portal_exit_rescore_recovery.py",
        ),
    }
    raw_gate = _mapping(raw["gate"], label="raw gate")
    return {
        "schema_version": 1,
        "recovery_id": RECOVERY_IDS[family],
        "status": "completed",
        "scope": {
            "kind": "additive_independent_rescore_recovery",
            "capability_id": _EXPECTED_COMPONENT_ID,
            "family": family,
            "model_evaluation_rerun_performed": False,
            "gpu_used": False,
            "raw_receipt_rewritten": False,
            "legacy_failed_rescore_rewritten": False,
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
            "portal_release_config": bindings["release_config"],
            "raw_receipt": bindings["raw_receipt"],
            "retained_legacy_failed_rescore": bindings["legacy_failed_rescore"],
            "checkpoint": {
                "checkpoint_id": checkpoint["checkpoint_id"],
                "frozen_weights": checkpoint_weights,
                "raw_adapter_checkpoint_path": raw_adapter.get("checkpoint"),
                "raw_adapter_checkpoint_sha256": raw_adapter.get("checkpoint_sha256"),
                "raw_receipt_matches_frozen_checkpoint_sha256": (
                    raw_adapter.get("checkpoint_sha256") == checkpoint_weights["sha256"]
                ),
            },
            "implementation": implementation,
        },
        "input_integrity": {
            "all_frozen_inputs_unchanged_during_recovery": input_unchanged,
            "identities_after_recovery_read": after,
        },
        "legacy_failure_diagnosis": {
            "legacy_status": legacy["status"],
            "legacy_checks": legacy["checks"],
            "only_failed_check": "metrics_reconstructed_from_records",
            "root_cause": (
                "legacy record rescore used implicit float64 loss arrays; the "
                "source Portal Exit evaluator retains float32 loss arrays through "
                "mean and subtraction"
            ),
            "legacy_float64_loss_reconstruction": legacy_float64_losses,
            "per_scalar_metric": scalar_comparison["metrics"],
            "uncertainty": uncertainty_comparison,
            "latent_response": latent_comparison,
            "gate_exact_equal_despite_loss_deltas": gate == raw_gate,
        },
        "reconstruction": {
            "scalar_record_aggregation": {
                "mse_record_dtype": "float32",
                "mse_aggregate_operations": [
                    "correct_losses.mean()",
                    "other_losses.mean()",
                    "(other_losses - correct_losses).mean()",
                ],
                "boolean_decision_aggregation": "original NumPy boolean means -> float64",
                "paired_bootstrap": {
                    "dtype": "float64",
                    "resamples": _BOOTSTRAP_RESAMPLES,
                    "seed": _BOOTSTRAP_SEED,
                    "quantile": 0.025,
                },
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
            "uncertainty_metrics": uncertainty_comparison,
            "latent_metrics": latent_comparison,
            "gate_exact_equal": gate == raw_gate,
            "legacy_gate_exact_equal": gate == legacy.get("reconstructed_gate"),
            "stored_model_gate_passed": raw_gate["passed"],
            "recomputed_model_gate_passed": gate["passed"],
        },
        "descriptive_checkpoint_score": _descriptive_checkpoint_score(
            raw=raw,
            raw_logical_path=raw_logical_path.as_posix(),
            metrics=metrics,
        ),
    }


def recover_portal_exit_rescore_recovery(
    *,
    family: str,
    repo_root: Path | None = None,
    raw_receipt: Path | str | None = None,
    legacy_failed_rescore: Path | str | None = None,
    preregistration: Path | str = DEFAULT_PREREGISTRATION,
    freeze_receipt: Path | str = DEFAULT_FREEZE_RECEIPT,
    release_config: Path | str = PORTAL_EXIT_RELEASE_LOGICAL_PATH,
    output_path: Path | str | None = None,
) -> dict[str, Any]:
    """Build and x-exclusively persist one Portal Exit recovery receipt."""

    root = (repo_root or repository_root()).resolve()
    receipt = build_portal_exit_rescore_recovery_receipt(
        family=family,
        repo_root=root,
        raw_receipt=raw_receipt,
        legacy_failed_rescore=legacy_failed_rescore,
        preregistration=preregistration,
        freeze_receipt=freeze_receipt,
        release_config=release_config,
        output_path=output_path,
    )
    output = write_recovery_receipt_exclusive(
        output_path or DEFAULT_RECOVERY_RECEIPTS[family], receipt, repo_root=root
    )
    if output.relative_to(root).as_posix() != receipt["output"]["path"]:
        raise AssertionError("Recovery receipt was written to the wrong path")
    return receipt


__all__ = [
    "DEFAULT_FREEZE_RECEIPT",
    "DEFAULT_LEGACY_FAILED_RESCORES",
    "DEFAULT_PREREGISTRATION",
    "DEFAULT_RAW_RECEIPTS",
    "DEFAULT_RECOVERY_RECEIPTS",
    "PORTAL_EXIT_RELEASE_LOGICAL_PATH",
    "RECOVERY_IDS",
    "RECOVERY_NAMESPACE",
    "build_portal_exit_rescore_recovery_receipt",
    "recover_portal_exit_rescore_recovery",
    "validate_recovery_output_path",
    "write_recovery_receipt_exclusive",
]
