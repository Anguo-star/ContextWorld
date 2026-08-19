from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

import contextworld.benchmarks.suite_v2_integrity_reseal as integrity_reseal
from contextworld.benchmarks.suite_data import SUITE_V2_COMPONENT_IDS
from contextworld.benchmarks.suite_v2_integrity_reseal import (
    RESEAL_ID,
    audit_current_identity_drift,
    build_integrity_reseal_decision,
    validate_integrity_reseal_decision,
)


ROOT = Path(__file__).resolve().parents[1]


def _install_fake_cem_result_freeze(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    logical_path = integrity_reseal.DESCRIPTIVE_RESULT_FREEZE_SPECS[
        "original_baseline_cem"
    ]["path"]
    fake = tmp_path / "contextworld_original_baseline_cem_results_freeze_v1.json"
    fake.write_text(
        json.dumps(
            {
                "freeze_id": integrity_reseal.DESCRIPTIVE_RESULT_FREEZE_SPECS[
                    "original_baseline_cem"
                ]["freeze_id"],
                "status": "frozen_test_fixture",
            }
        ),
        encoding="utf-8",
    )
    original_resolve = integrity_reseal.resolve_contextworld_path

    def resolve_with_fake_cem(value: str | Path, **kwargs: object) -> Path:
        if str(value) == logical_path:
            return fake
        return original_resolve(value, **kwargs)

    monkeypatch.setattr(
        integrity_reseal,
        "resolve_contextworld_path",
        resolve_with_fake_cem,
    )
    return fake


def test_reseal_audit_reports_the_current_historical_identity_drift() -> None:
    audit = audit_current_identity_drift(repo_root=ROOT)

    assert audit["reseal_id"] == RESEAL_ID
    assert set(audit["component_release_drift"]) == set(SUITE_V2_COMPONENT_IDS)
    assert "contextworld/benchmarks/adapters.py" in audit["source_drift"]
    assert audit["public_document_drift"]["drifted"] is True
    assert audit["missing_required_descriptive_result_freezes"] == []
    assert set(audit["descriptive_result_freezes"]) == {
        "original_baseline_icl",
        "original_baseline_cem",
    }
    assert audit["requires_additive_reseal"] is True


def test_reseal_decision_binds_every_current_material_exactly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_cem = _install_fake_cem_result_freeze(tmp_path, monkeypatch)
    decision = build_integrity_reseal_decision(repo_root=ROOT)

    assert decision["reseal_config"]["path"] == (
        "configs/benchmark/contextworld_icl_suite_v2_integrity_reseal_v1.yaml"
    )
    assert decision["release_materials"]["sources"][
        "contextworld/benchmarks/suite_data.py"
    ]["sha256"] == hashlib.sha256(
        (ROOT / "contextworld/benchmarks/suite_data.py").read_bytes()
    ).hexdigest()
    assert set(decision["release_materials"]["authority_implementation"]) == {
        "contextworld/benchmarks/suite_v2_integrity_reseal.py",
        "scripts/freeze_contextworld_icl_suite_v2_integrity_reseal_v1.py",
    }
    assert set(decision["release_materials"]["components"]) == set(
        SUITE_V2_COMPONENT_IDS
    )
    assert decision["release_materials"]["descriptive_result_freezes"] == {
        "original_baseline_icl": {
            "path": (
                "configs/benchmark/"
                "contextworld_original_baseline_matrix_results_freeze_v1.json"
            ),
            "sha256": hashlib.sha256(
                (
                    ROOT
                    / "configs/benchmark/"
                    "contextworld_original_baseline_matrix_results_freeze_v1.json"
                ).read_bytes()
            ).hexdigest(),
            "size_bytes": (
                ROOT
                / "configs/benchmark/"
                "contextworld_original_baseline_matrix_results_freeze_v1.json"
            ).stat().st_size,
        },
        "original_baseline_cem": {
            "path": integrity_reseal.DESCRIPTIVE_RESULT_FREEZE_SPECS[
                "original_baseline_cem"
            ]["path"],
            "sha256": hashlib.sha256(fake_cem.read_bytes()).hexdigest(),
            "size_bytes": fake_cem.stat().st_size,
        },
    }
    assert validate_integrity_reseal_decision(decision, repo_root=ROOT) == {
        "passed": True,
        "reseal_id": RESEAL_ID,
    }

    stale = copy.deepcopy(decision)
    stale["release_materials"]["components"]["speed"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="does not match current bytes"):
        validate_integrity_reseal_decision(stale, repo_root=ROOT)
