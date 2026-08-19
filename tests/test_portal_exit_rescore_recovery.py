from __future__ import annotations

import json

import pytest

from contextworld.benchmarks.portal_exit_rescore_recovery import (
    DEFAULT_RAW_RECEIPTS,
    RECOVERY_NAMESPACE,
    build_portal_exit_rescore_recovery_receipt,
    validate_recovery_output_path,
    write_recovery_receipt_exclusive,
)
from contextworld.benchmarks.original_baseline_archive import (
    audit_archived_original_baseline_matrix,
)
from contextworld.paths import repository_root


def _archived_recovery_receipt(family: str) -> dict:
    audit = audit_archived_original_baseline_matrix()
    assert audit["status"] == "passed"
    path = (
        repository_root()
        / RECOVERY_NAMESPACE
        / f"{family}_float32_rescore_recovery_v1.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    ("family", "expected_deltas"),
    (
        (
            "lewm",
            {
                "correct_future_mse_mean": -1.1851079761981964e-07,
                "other_future_mse_mean": 2.4703331291675568e-07,
                "other_minus_correct_mse_margin_mean": -1.3969838619232178e-09,
            },
        ),
        (
            "pldm",
            {
                "correct_future_mse_mean": 8.065253496170044e-07,
                "other_future_mse_mean": -1.2693926692008972e-06,
                "other_minus_correct_mse_margin_mean": 2.7939677238464355e-09,
            },
        ),
    ),
)
def test_frozen_portal_exit_float32_recovery_is_exact(
    family: str, expected_deltas: dict[str, float]
) -> None:
    receipt = _archived_recovery_receipt(family)

    verification = receipt["verification"]
    scalar = verification["scalar_metrics"]
    assert receipt["status"] == "completed"
    assert verification["passed"] is True
    assert scalar["all_recovery_float64_json_bitwise_equal"] is True
    assert scalar["all_float32_loss_aggregates_bitwise_equal"] is True
    assert scalar["legacy_only_float32_loss_aggregation_mismatches"] is True
    assert verification["uncertainty_metrics"]["all_recovery_exact"] is True
    assert verification["uncertainty_metrics"]["all_legacy_exact"] is True
    assert verification["latent_metrics"]["all_recovery_exact"] is True
    assert verification["latent_metrics"]["all_legacy_exact"] is True
    assert verification["gate_exact_equal"] is True
    assert verification["legacy_gate_exact_equal"] is True
    assert verification["stored_model_gate_passed"] is False
    assert verification["recomputed_model_gate_passed"] is False
    assert receipt["bindings"]["raw_receipt"]["path"] == DEFAULT_RAW_RECEIPTS[
        family
    ].as_posix()
    assert receipt["bindings"]["checkpoint"][
        "raw_receipt_matches_frozen_checkpoint_sha256"
    ] is True
    assert receipt["input_integrity"][
        "all_frozen_inputs_unchanged_during_recovery"
    ] is True

    for metric, expected_delta in expected_deltas.items():
        row = scalar["metrics"][metric]
        assert row["source_aggregation_semantics"].startswith("float32") or row[
            "source_aggregation_semantics"
        ].startswith("(float32")
        assert row["recovery_exact"] is True
        assert row["legacy_exact"] is False
        assert row["recomputed_minus_stored"] == 0.0
        assert row["legacy_minus_stored"] == expected_delta


@pytest.mark.parametrize("family", ("lewm", "pldm"))
def test_legacy_portal_rescore_is_preserved_and_explained(family: str) -> None:
    root = repository_root()
    raw = json.loads((root / DEFAULT_RAW_RECEIPTS[family]).read_text(encoding="utf-8"))
    receipt = _archived_recovery_receipt(family)
    diagnosis = receipt["legacy_failure_diagnosis"]
    assert diagnosis["legacy_status"] == "failed"
    assert diagnosis["only_failed_check"] == "metrics_reconstructed_from_records"
    assert diagnosis["legacy_checks"] == {
        "release_identity_matches": True,
        "metrics_reconstructed_from_records": False,
        "latent_response_reconstructed_from_records": True,
        "gate_recomputed": True,
    }
    assert diagnosis["gate_exact_equal_despite_loss_deltas"] is True
    assert diagnosis["latent_response"]["all_legacy_exact"] is True
    assert diagnosis["uncertainty"]["all_legacy_exact"] is True
    for metric in (
        "correct_future_mse_mean",
        "other_future_mse_mean",
        "other_minus_correct_mse_margin_mean",
    ):
        assert diagnosis["legacy_float64_loss_reconstruction"][metric] == diagnosis[
            "per_scalar_metric"
        ][metric]["legacy_reconstructed"]
        assert raw["metrics"][metric] == diagnosis["per_scalar_metric"][metric][
            "stored"
        ]


def test_portal_recovery_output_is_namespaced_and_exclusive(tmp_path) -> None:
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
    with pytest.raises(ValueError, match="contextworld-portal-exit"):
        validate_recovery_output_path(
            tmp_path / "artifacts/evaluation/original_baseline_matrix_v1/not-recovery.json",
            repo_root=tmp_path,
        )


def test_portal_recovery_rejects_cross_family_raw_receipt() -> None:
    with pytest.raises(ValueError, match="raw receipt must be the frozen canonical input"):
        build_portal_exit_rescore_recovery_receipt(
            family="lewm",
            raw_receipt=DEFAULT_RAW_RECEIPTS["pldm"],
            output_path=RECOVERY_NAMESPACE / "test-invalid-raw.json",
        )
