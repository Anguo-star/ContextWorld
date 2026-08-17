from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from contextworld.benchmarks import suite_data
from contextworld.benchmarks.cube_grasp_rule_suite_registration import (
    file_identity,
    read_yaml,
    tree_identity,
)
from contextworld.benchmarks.cube_grasp_rule_suite_registration_recovery import (
    RECOVERY_PREREGISTRATION_LOGICAL_PATH,
    RECOVERY_REGISTRATION_ID,
    validate_registration_recovery_preregistration_contract,
)
from contextworld.paths import repository_root
import scripts.finalize_cube_grasp_rule_h3_v4r1_suite_registration_recovery_v2 as finalizer
import scripts.freeze_cube_grasp_rule_h3_v4r1_suite_registration_recovery_v2 as freezer


ROOT = repository_root()
PREREGISTRATION = ROOT / RECOVERY_PREREGISTRATION_LOGICAL_PATH
PRE_RECOVERY_SUITE = (
    ROOT / "configs/benchmark/contextworld_icl_suite_v2.yaml"
)


def test_recovery_v2_preregistration_is_exact_and_fail_closed() -> None:
    prereg = read_yaml(
        PREREGISTRATION, label="registration recovery preregistration"
    )
    validate_registration_recovery_preregistration_contract(
        prereg, preregistration_path=PREREGISTRATION
    )
    assert prereg["prior_failed_registration"]["failed_staging"] == {
        "path": (
            "artifacts/evaluation/history3/"
            "cube_gripper_carry_h3_v4r1_suite_registration_v1/"
            ".suite_v2_copy_export_v1.staging"
        ),
        "files": 11609,
        "bytes": 25861414526,
        "sha256": (
            "4dc46df3f9188391069eeebb2339b1197"
            "238269f4113e332e6d43bf84eae35e9"
        ),
    }
    assert (
        prereg["prior_failed_registration"]["operational_observation"][
            "authoritative_for_recovery"
        ]
        is False
    )


@pytest.mark.parametrize(
    ("section", "key", "replacement"),
    [
        ("scope", "public_test_rerun_authorized", True),
        ("recovery_contract", "directory_rename_authorized", True),
        (
            "recovery_contract",
            "prior_failed_staging_as_export_source_authorized",
            True,
        ),
        (
            "recovery_contract",
            "registration_decision_written_last",
            False,
        ),
        (
            "registration_gates",
            "exported_bundle_full_reaudit_required",
            False,
        ),
    ],
)
def test_recovery_v2_contract_rejects_authority_drift(
    section: str, key: str, replacement: object
) -> None:
    prereg = read_yaml(
        PREREGISTRATION, label="registration recovery preregistration"
    )
    changed = deepcopy(prereg)
    changed[section][key] = replacement
    with pytest.raises(RuntimeError):
        validate_registration_recovery_preregistration_contract(
            changed, preregistration_path=PREREGISTRATION
        )


def test_recovery_v2_uses_a_new_candidate_and_decision_namespace() -> None:
    historical = suite_data.load_icl_suite_release(PRE_RECOVERY_SUITE)
    current = suite_data.load_icl_suite_release(
        suite_data.SUITE_V2_RECOVERY_CONFIG
    )
    assert historical["membership_authority"]["decision_path"].endswith(
        "suite_registration_v1/registration_decision_v1.json"
    )
    assert current["membership_authority"] == {
        "config_alone_grants_membership": False,
        "activation_condition": "passed_registration_decision_v2",
        "registration_id": RECOVERY_REGISTRATION_ID,
        "decision_path": (
            "artifacts/evaluation/history3/"
            "cube_gripper_carry_h3_v4r1_suite_registration_recovery_v2/"
            "registration_decision_v2.json"
        ),
        "decision_is_commit_marker": True,
        "partial_outputs_grant_membership": False,
        "failed_finalization_requires_new_preregistration": True,
        "recovery_protocol": (
            "direct_one_use_export_reservation_no_directory_rename"
        ),
        "prior_failed_registration_id": (
            "contextworld_cube_gripper_carry_h3_v4r1_"
            "suite_registration_v1"
        ),
        "directory_rename_authorized": False,
        "prior_failed_staging_reuse_authorized": False,
        "note": (
            "benchmark_component_status denotes technical readiness; Cube "
            "Suite v2 membership is effective only when the recovery-v2 "
            "canonical registration decision has status "
            "suite_registration_passed and passed true. The pre-recovery v1 "
            "candidate remains permanently uncommitted; its staging is "
            "preserved and is neither moved nor reused. Any failed recovery "
            "leaves non-authoritative partial outputs and requires a new "
            "preregistration."
        ),
    }
    with pytest.raises(RuntimeError, match="permanently uncommitted"):
        suite_data.require_suite_membership_activation(historical)

    recovery_sources = set(current["repository"]["source_sha256"])
    assert {
        "contextworld/benchmarks/"
        "cube_grasp_rule_suite_registration_recovery.py",
        "scripts/freeze_cube_grasp_rule_h3_v4r1_"
        "suite_registration_recovery_v2.py",
        "scripts/finalize_cube_grasp_rule_h3_v4r1_"
        "suite_registration_recovery_v2.py",
    }.issubset(recovery_sources)


def test_recovery_finalizer_reserves_before_audits_and_decides_last() -> None:
    source = Path(finalizer.__file__).read_text(encoding="utf-8")
    reservation = source.index("export_destination.mkdir()")
    reservation_receipt = source.index(
        '_write_json(outputs["export_reservation"], reservation)'
    )
    component_audit = source.index("component_audit = audit_cube")
    suite_audit = source.index(
        "suite_audit = _audit_icl_suite_release_for_registration"
    )
    direct_copy = source.index(
        "export = _export_icl_suite_artifacts_for_registration"
    )
    copy_identity = source.index("export_tree = tree_identity")
    copy_complete = source.index(
        '_write_json(outputs["copy_complete"], copy_complete)'
    )
    bundle_audit = source.index(
        "bundle_audit = _audit_icl_suite_release_for_registration"
    )
    first_formal_audit = source.index(
        '_write_json(outputs["component_audit"], component_audit)'
    )
    decision = source.index(
        '_write_json(outputs["registration_decision"], decision)'
    )
    assert (
        reservation
        < reservation_receipt
        < component_audit
        < suite_audit
        < direct_copy
        < copy_identity
        < copy_complete
        < bundle_audit
        < first_formal_audit
        < decision
    )
    for forbidden in (
        "os.replace",
        "os.rename",
        "shutil.move",
        "shutil.rmtree",
        "from_checkpoint",
    ):
        assert forbidden not in source


def test_registration_copy_is_exclusive_and_rejects_symlinks(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"cube-recovery")
    target = tmp_path / "target.bin"
    suite_data._exclusive_copy_file(source, target)
    assert target.read_bytes() == b"cube-recovery"
    with pytest.raises(FileExistsError):
        suite_data._exclusive_copy_file(source, target)

    source_tree = tmp_path / "source-tree"
    source_tree.mkdir()
    (source_tree / "payload.bin").write_bytes(b"payload")
    (source_tree / "link.bin").symlink_to(source_tree / "payload.bin")
    with pytest.raises(RuntimeError, match="symlink or special node"):
        suite_data._exclusive_copy_tree(
            source_tree, tmp_path / "target-tree"
        )


def test_activation_revalidates_receipts_and_the_complete_export_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    export = tmp_path / "suite_v2_copy_export_v2"
    benchmark = export / "benchmark"
    benchmark.mkdir(parents=True)
    inventory_path = benchmark / "inventory.json"
    inventory = {
        "schema_version": 1,
        "release_id": suite_data.SUITE_V2_RELEASE_ID,
        "status": "passed",
        "mode": "copy",
        "components": list(suite_data.SUITE_V2_COMPONENT_IDS),
        "membership_activation": {
            "active": False,
            "status": "pending_registration_internal_audit",
            "decision_path": suite_data.SUITE_V2_REGISTRATION_DECISION,
            "partial_outputs_grant_membership": False,
        },
    }
    inventory_path.write_text(
        json.dumps(inventory, sort_keys=True) + "\n", encoding="utf-8"
    )
    preregistration_identity = {
        "path": "configs/benchmark/recovery.yaml",
        "sha256": "1" * 64,
        "size_bytes": 1,
    }
    freeze_identity = {
        "path": "artifacts/recovery/freeze.json",
        "sha256": "2" * 64,
        "size_bytes": 1,
    }
    reservation = {
        "schema_version": 1,
        "registration_id": RECOVERY_REGISTRATION_ID,
        "status": "direct_export_exclusively_reserved",
        "export_path": suite_data.SUITE_V2_RECOVERY_EXPORT,
        "reservation_operation": "mkdir_exist_ok_false",
        "directory_rename_authorized": False,
        "prior_failed_staging_reuse_authorized": False,
        "registration_decision_is_only_membership_commit_marker": True,
        "partial_outputs_grant_membership": False,
        "preregistration": preregistration_identity,
        "freeze_receipt": freeze_identity,
        "passed": True,
    }
    reservation_path = tmp_path / "export_reservation_v2.json"
    reservation_path.write_text(
        json.dumps(reservation, sort_keys=True) + "\n", encoding="utf-8"
    )
    copy_complete = {
        "schema_version": 1,
        "registration_id": RECOVERY_REGISTRATION_ID,
        "status": "fresh_direct_copy_complete",
        "export_path": suite_data.SUITE_V2_RECOVERY_EXPORT,
        "export_tree": tree_identity(export),
        "copy_manifest": file_identity(
            inventory_path,
            logical_path=(
                f"{suite_data.SUITE_V2_RECOVERY_EXPORT}/"
                "benchmark/inventory.json"
            ),
        ),
        "fresh_copy_from_canonical_sources": True,
        "prior_failed_staging_reused": False,
        "prior_failed_namespace_mutated": False,
        "directory_rename_used": False,
        "exclusive_file_creation_required": True,
        "passed": True,
    }
    copy_complete_path = tmp_path / "suite_v2_copy_complete_v2.json"
    copy_complete_path.write_text(
        json.dumps(copy_complete, sort_keys=True) + "\n", encoding="utf-8"
    )
    export_audit = {
        "status": "passed",
        "passed": True,
        "copy_export": {
            "status": "passed",
            "mode": "copy",
            "destination": str(export),
            "components": list(suite_data.SUITE_V2_COMPONENT_IDS),
        },
        "copy_completion": copy_complete,
        "bundle_reaudit": {"passed": True},
        "direct_export_commit": {
            "direct_target_exclusively_reserved": True,
            "fresh_copy_tree_identity_verified": True,
            "bundle_reaudit_completed_before_formal_audit_writes": True,
            "directory_rename_used": False,
            "committed_destination": str(export),
            "registration_decision_is_only_membership_commit_marker": True,
        },
        "prior_failed_registration": {
            "registration_id": (
                "contextworld_cube_gripper_carry_h3_v4r1_"
                "suite_registration_v1"
            ),
            "staging_reused": False,
            "namespace_mutated": False,
        },
        "cube_public_test_rerun": False,
        "cube_formal_checkpoint_opened": False,
    }
    export_audit_path = tmp_path / "suite_v2_export_audit_v2.json"
    export_audit_path.write_text(
        json.dumps(export_audit, sort_keys=True) + "\n", encoding="utf-8"
    )
    evidence = {
        "preregistration": preregistration_identity,
        "freeze_receipt": freeze_identity,
    }
    evidence_audits = {
        "export_reservation": {"path": str(reservation_path)},
        "copy_complete": {"path": str(copy_complete_path)},
        "export_audit": {"path": str(export_audit_path)},
    }

    def resolve_export(
        value: str | Path,
        *,
        repo_root: Path | None = None,
        label: str,
        allow_missing: bool = False,
    ) -> Path:
        assert str(value) == suite_data.SUITE_V2_RECOVERY_EXPORT
        return export

    monkeypatch.setattr(
        suite_data, "resolve_no_symlink_contextworld_path", resolve_export
    )
    assert suite_data._validate_recovery_v2_export_commit(
        evidence, evidence_audits, repo_root=tmp_path
    )["passed"] is True

    changed_reservation = {**reservation, "directory_rename_authorized": True}
    reservation_path.write_text(
        json.dumps(changed_reservation, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="export reservation drifted"):
        suite_data._validate_recovery_v2_export_commit(
            evidence, evidence_audits, repo_root=tmp_path
        )

    reservation_path.write_text(
        json.dumps(reservation, sort_keys=True) + "\n", encoding="utf-8"
    )
    (export / "post-decision-tamper.bin").write_bytes(b"tamper")
    with pytest.raises(RuntimeError, match="committed export tree drifted"):
        suite_data._validate_recovery_v2_export_commit(
            evidence, evidence_audits, repo_root=tmp_path
        )


def test_recovery_freeze_binds_every_execution_implementation() -> None:
    assert set(freezer.IMPLEMENTATION_PATHS) == {
        "historical_registration_contract",
        "recovery_contract",
        "suite_data_api",
        "recovery_freezer",
        "recovery_finalizer",
    }


def test_recovery_finalizer_rejects_noncanonical_preregistration(
    tmp_path: Path,
) -> None:
    with pytest.raises(RuntimeError, match="canonical path"):
        finalizer.finalize_registration_recovery(
            preregistration=tmp_path / "copy.yaml"
        )
