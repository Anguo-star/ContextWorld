from __future__ import annotations

import json

import numpy as np
import pytest

from contextworld.benchmarks.action_strength_rescore_recovery import (
    DEFAULT_RAW_RECEIPT,
    RECOVERY_NAMESPACE,
    validate_recovery_output_path,
    write_recovery_receipt_exclusive,
)
from contextworld.benchmarks.original_baseline_archive import (
    audit_archived_original_baseline_matrix,
)
from contextworld.paths import repository_root


ARCHIVED_RECOVERY_RECEIPT = (
    RECOVERY_NAMESPACE
    / "contextworld-action-strength/lewm_float32_rescore_recovery_v1.json"
)


def _archived_recovery_receipt() -> dict:
    audit = audit_archived_original_baseline_matrix()
    assert audit["status"] == "passed"
    path = repository_root() / ARCHIVED_RECOVERY_RECEIPT
    return json.loads(path.read_text(encoding="utf-8"))


def test_frozen_action_strength_lewm_float32_recovery_is_exact() -> None:
    receipt = _archived_recovery_receipt()
    verification = receipt["verification"]
    scalar = verification["scalar_metrics"]
    latent = verification["latent_metrics"]
    assert verification["passed"] is True
    assert verification["gate_exact_equal"] is True
    assert verification["stored_model_gate_passed"] is False
    assert verification["recomputed_model_gate_passed"] is False
    assert scalar["all_float64_json_bitwise_equal"] is True
    assert scalar["all_float32_loss_aggregates_bitwise_equal"] is True
    assert latent["paired_latent_response_summaries_close"] is True
    assert latent["all_float64_json_bitwise_equal"] is True
    margin = scalar["metrics"]["other_minus_correct_mse_margin_mean"]
    assert margin["source_aggregation_dtype"] == "float32"
    assert margin["float32_bitwise_equal"] is True
    assert receipt["bindings"]["raw_receipt"]["path"] == DEFAULT_RAW_RECEIPT.as_posix()
    assert receipt["bindings"]["checkpoint"][
        "raw_receipt_matches_frozen_checkpoint_sha256"
    ] is True
    assert receipt["input_integrity"][
        "all_frozen_inputs_unchanged_during_recovery"
    ] is True


def test_float32_semantics_eliminate_the_known_float64_margin_delta() -> None:
    raw = json.loads((repository_root() / DEFAULT_RAW_RECEIPT).read_text())
    records = raw["records"]
    correct = np.asarray(
        [row["low_strength"]["correct_future_mse"] for row in records]
        + [row["high_strength"]["correct_future_mse"] for row in records],
        dtype=np.float64,
    )
    other = np.asarray(
        [row["low_strength"]["other_future_mse"] for row in records]
        + [row["high_strength"]["other_future_mse"] for row in records],
        dtype=np.float64,
    )
    legacy_float64_margin = float((other - correct).mean())
    stored_margin = raw["metrics"]["other_minus_correct_mse_margin_mean"]
    assert abs(legacy_float64_margin - stored_margin) > 1.0e-9

    receipt = _archived_recovery_receipt()
    recovered_margin = receipt["reconstruction"]["metrics"][
        "other_minus_correct_mse_margin_mean"
    ]
    assert recovered_margin == stored_margin


def test_recovery_output_is_namespaced_and_exclusive(tmp_path) -> None:
    inside, logical = validate_recovery_output_path(
        tmp_path / RECOVERY_NAMESPACE / "receipt.json", repo_root=tmp_path
    )
    assert logical == (RECOVERY_NAMESPACE / "receipt.json").as_posix()
    write_recovery_receipt_exclusive(
        inside, {"schema_version": 1, "status": "completed"}, repo_root=tmp_path
    )
    with pytest.raises(FileExistsError):
        write_recovery_receipt_exclusive(
            inside,
            {"schema_version": 1, "status": "completed"},
            repo_root=tmp_path,
        )
    with pytest.raises(ValueError, match="rescore_recovery"):
        validate_recovery_output_path(
            tmp_path / "artifacts/evaluation/original_baseline_matrix_v1/not-recovery.json",
            repo_root=tmp_path,
        )
