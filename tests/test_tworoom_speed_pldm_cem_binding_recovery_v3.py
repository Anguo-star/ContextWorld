from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from contextworld.paths import resolve_contextworld_path


ROOT = Path(__file__).resolve().parents[1]
COMPLETION_ROOT = (
    ROOT
    / "artifacts/evaluation/history3/tworoom_speed_pldm_reference_completion_v1"
)
BINDING = COMPLETION_ROOT / "formal_icl_v1/cem_binding_v1.json"
PREREGISTRATION = (
    ROOT / "configs/benchmark/tworoom_speed_pldm_cem_binding_recovery_v3.yaml"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_cem_binding_recovery_preserves_failed_attempts_and_positive_chain() -> None:
    binding = _load_json(BINDING)
    recovery = binding["binding_recovery"]
    failure_v1 = _load_json(COMPLETION_ROOT / "formal_icl_v1/cem_binding_recovery_v1_failure.json")
    failure_v2 = _load_json(COMPLETION_ROOT / "formal_icl_v1/cem_binding_recovery_v2_failure.json")

    assert failure_v1["status"] == "failed_before_cem_binding_or_execution"
    assert failure_v1["boundary"]["cem_binding_written"] is False
    assert failure_v2["status"] == "failed_before_cem_binding_or_execution"
    assert failure_v2["boundary"]["cem_binding_written"] is False
    assert binding["status"] == "frozen_after_passed_three_seed_public_icl_before_cem"
    assert binding["passed"] is True
    assert binding["cem"] == {"authorized": True, "executed": False}
    assert recovery["recovery_id"] == "tworoom_speed_pldm_cem_binding_recovery_v3"
    assert recovery["prepublic_cem_protocol_changed"] is False
    assert recovery["model_or_environment_execution_performed"] is False
    assert recovery["public_test_reopened"] is False
    assert recovery["cem_executed"] is False
    assert binding["frozen_chain"]["evaluation_binding_receipt"]["path"].endswith(
        "evaluation_binding_v1/recovery_v1/evaluation_binding_receipt.json"
    )


def test_cem_binding_recovery_keeps_frozen_files_and_portable_receipt_identities() -> None:
    preregistration = yaml.safe_load(PREREGISTRATION.read_text(encoding="utf-8"))
    binding = _load_json(BINDING)
    results_freeze = _load_json(
        ROOT / "configs/benchmark/contextworld_original_baseline_cem_results_freeze_v1.json"
    )
    paired = binding["tracks"]["original_task_retention_cem"]["paired_baseline"]

    launcher = preregistration["implementation"]["recovery_launcher"]
    assert _sha256(ROOT / launcher["path"]) == launcher["sha256"]
    assert results_freeze["matrix_summary"]["strictly_reused_episodes"] == 300
    assert paired["expected"] == {"successes": 278, "evaluations": 300}
    assert [row["eval_seed"] for row in paired["raw_receipts"]] == [
        42,
        43,
        44,
        45,
        46,
        47,
    ]
    for row in paired["raw_receipts"]:
        identity = row["receipt"]
        assert identity["path"].startswith("artifacts/")
        path = resolve_contextworld_path(identity["path"], repo_root=ROOT)
        assert path.stat().st_size == identity["size_bytes"]
        assert _sha256(path) == identity["sha256"]


def test_cem_binding_recovery_retains_identical_before_after_snapshots() -> None:
    binding = _load_json(BINDING)
    integrity = binding["input_integrity"]

    assert integrity["all_frozen_inputs_unchanged_during_binding"] is True
    assert integrity["identities_before_binding"] == integrity["identities_after_binding"]
    assert set(binding["tracks"]) == {
        "shared",
        "action_planning_cem",
        "original_task_retention_cem",
    }
