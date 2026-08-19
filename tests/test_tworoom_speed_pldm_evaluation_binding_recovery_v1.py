from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
PREREGISTRATION = (
    ROOT
    / "configs/benchmark/tworoom_speed_pldm_evaluation_binding_recovery_v1.yaml"
)
FAILED_RECEIPT = (
    ROOT
    / "artifacts/evaluation/history3/tworoom_speed_pldm_reference_completion_v1"
    / "evaluation_binding_v1/evaluation_binding_receipt.json"
)
RECOVERED_RECEIPT = FAILED_RECEIPT.parent / "recovery_v1/evaluation_binding_receipt.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_binding_recovery_preserves_original_failure_and_repairs_only_registered_checks() -> None:
    preregistration = yaml.safe_load(PREREGISTRATION.read_text(encoding="utf-8"))
    failed = _load_json(FAILED_RECEIPT)
    recovered = _load_json(RECOVERED_RECEIPT)
    frozen = preregistration["frozen_inputs"]
    registered_failures = set(
        preregistration["bounded_repair"]["original_failed_checks"]
    )
    observed_failures = {
        name for name, row in failed["checks"].items() if row.get("passed") is not True
    }

    assert _sha256(FAILED_RECEIPT) == frozen["failed_binding_receipt"]["sha256"]
    assert failed["status"] == "failed_evaluation_binding_freeze"
    assert failed["passed"] is False
    assert observed_failures == registered_failures
    assert recovered["status"] == "passed_evaluation_binding_freeze"
    assert recovered["passed"] is True
    assert all(row.get("passed") is True for row in recovered["checks"].values())
    assert recovered["binding_freeze_recovery"]["failed_binding_receipt"] == {
        **frozen["failed_binding_receipt"]
    }


def test_binding_recovery_uses_exact_catalog_copy_and_authoritative_completion_evidence() -> None:
    preregistration = yaml.safe_load(PREREGISTRATION.read_text(encoding="utf-8"))
    recovered = _load_json(RECOVERED_RECEIPT)
    catalog = preregistration["bounded_repair"]["catalog_restore"]
    source = _resolve(catalog["source"]["path"])
    destination = _resolve(catalog["destination"]["path"])

    assert _sha256(source) == catalog["source"]["sha256"]
    assert _sha256(destination) == catalog["destination"]["sha256"]
    assert source.stat().st_size == destination.stat().st_size == 119993
    assert recovered["binding_freeze_recovery"]["catalog_restore"][
        "content_regenerated"
    ] is False
    for seed in (3072, 4096, 5120):
        check = recovered["checks"][f"training_preflight_{seed}"]
        evidence = check["completion_evidence"]
        assert check["passed"] is True
        assert (
            check["preflight_receipt_role"]
            == "preflight_only_not_training_completion_evidence"
        )
        assert check["preflight"]["training_started"] is False
        assert "training_completed" not in check["preflight"]
        assert evidence["fixed_optimizer_steps"] == 12840
        assert evidence["terminal_report_recovery_optimizer_steps"] == 0


def test_binding_recovery_keeps_public_and_training_closed() -> None:
    preregistration = yaml.safe_load(PREREGISTRATION.read_text(encoding="utf-8"))
    recovered = _load_json(RECOVERED_RECEIPT)
    recovery = recovered["binding_freeze_recovery"]
    launcher = preregistration["implementation"]["recovery_launcher"]

    assert _sha256(_resolve(launcher["path"])) == launcher["sha256"]
    assert recovery["optimizer_steps_executed_by_binding_recovery"] == 0
    assert recovery["training_executed_by_binding_recovery"] is False
    assert recovery["model_or_checkpoint_selection_performed"] is False
    assert recovery["public_test_accessed"] is False
    assert recovery["formal_icl_executed"] is False
    assert recovery["cem_executed"] is False
    assert recovered["public_test"]["accessed_by_binding"] is False
    assert recovered["public_test"]["scored_by_binding"] is False
