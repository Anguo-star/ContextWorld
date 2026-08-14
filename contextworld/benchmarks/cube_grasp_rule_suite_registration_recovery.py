from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

from contextworld.benchmarks.cube_grasp_rule_suite_registration import (
    COMPONENT_ID,
    EXPECTED_PREREGISTRATION_LOGICAL_PATH as PRIOR_PREREGISTRATION_LOGICAL_PATH,
    FREEZE_RECEIPT_ID as PRIOR_FREEZE_RECEIPT_ID,
    REGISTRATION_ID as PRIOR_REGISTRATION_ID,
    RELEASE_ID,
    SUITE_RELEASE_ID,
    exact_value_equal,
    file_identity,
    identity_equal,
    lexical_absolute,
    read_json,
    read_yaml,
    require_no_symlink_components,
    resolve_no_symlink_contextworld_path,
    tree_identity,
)
from contextworld.paths import repository_root


RECOVERY_REGISTRATION_ID = (
    "contextworld_cube_gripper_carry_h3_v4r1_"
    "suite_registration_recovery_v2"
)
RECOVERY_FREEZE_RECEIPT_ID = f"{RECOVERY_REGISTRATION_ID}_freeze_v2"
RECOVERY_PREREGISTRATION_LOGICAL_PATH = (
    "configs/benchmark/"
    "cube_gripper_carry_h3_v4r1_suite_registration_recovery_v2_prereg.yaml"
)
PRIOR_REGISTRATION_ROOT = (
    "artifacts/evaluation/history3/"
    "cube_gripper_carry_h3_v4r1_suite_registration_v1"
)
PRIOR_FREEZE_RECEIPT_LOGICAL_PATH = (
    f"{PRIOR_REGISTRATION_ROOT}/registration_freeze_receipt_v1.json"
)
PRIOR_FAILED_STAGING_LOGICAL_PATH = (
    f"{PRIOR_REGISTRATION_ROOT}/.suite_v2_copy_export_v1.staging"
)
RECOVERY_REGISTRATION_ROOT = (
    "artifacts/evaluation/history3/"
    "cube_gripper_carry_h3_v4r1_suite_registration_recovery_v2"
)

EXPECTED_SCOPE = {
    "purpose": "recover_suite_registration_after_uncommitted_v1_staging",
    "environment": "Cube",
    "capability": "infer_hidden_gripper_carry_rule_from_recent_interaction",
    "history_tokens": 3,
    "public_test_already_completed": True,
    "public_test_rerun_authorized": False,
    "training_or_checkpoint_selection_authorized": False,
    "threshold_or_recipe_change_authorized": False,
    "prior_registration_namespace_mutation_authorized": False,
}
EXPECTED_RECOVERY_CONTRACT = {
    "strategy": "direct_one_use_export_reservation_no_directory_rename",
    "export_destination_reserved_before_component_or_suite_audit": True,
    "reservation_operation": "mkdir_exist_ok_false",
    "directory_rename_authorized": False,
    "prior_failed_staging_as_export_source_authorized": False,
    "prior_failed_namespace_delete_or_repair_authorized": False,
    "source_full_audit_required": True,
    "bundle_full_reaudit_required": True,
    "formal_audit_json_written_after_bundle_reaudit": True,
    "registration_decision_written_last": True,
    "registration_decision_is_only_membership_commit_marker": True,
    "partial_outputs_grant_membership": False,
    "failed_recovery_requires_new_preregistration": True,
    "cube_public_test_rerun_authorized": False,
    "cube_formal_checkpoint_open_authorized": False,
    "machine_specific_paths_allowed_in_export": False,
    "symlinks_allowed": False,
}
EXPECTED_REGISTRATION_GATES = {
    "component_full_audit_required": True,
    "suite_v2_full_audit_required": True,
    "direct_copy_export_required": True,
    "exported_bundle_full_reaudit_required": True,
    "suite_v2_components": 9,
    "formal_scoreboard_rows": 11,
    "formal_scoreboard_components": 7,
    "suite_v1_components": 8,
    "suite_v1_formal_scoreboard_rows": 10,
    "suite_v1_cube_absent": True,
}
EXPECTED_PLANNED_REPOSITORY_OUTPUTS = {
    "release_config": (
        "configs/benchmark/cube_gripper_carry_h3_v4r1_icl_release_v1.yaml"
    ),
    "suite_config": (
        "configs/benchmark/contextworld_icl_suite_v2_recovery_v2.yaml"
    ),
    "suite_data_api": "contextworld/benchmarks/suite_data.py",
    "historical_registration_contract": (
        "contextworld/benchmarks/cube_grasp_rule_suite_registration.py"
    ),
    "recovery_contract": (
        "contextworld/benchmarks/"
        "cube_grasp_rule_suite_registration_recovery.py"
    ),
    "recovery_freezer": (
        "scripts/freeze_cube_grasp_rule_h3_v4r1_"
        "suite_registration_recovery_v2.py"
    ),
    "recovery_finalizer": (
        "scripts/finalize_cube_grasp_rule_h3_v4r1_"
        "suite_registration_recovery_v2.py"
    ),
    "public_document": "docs/ContextWorld_ICL_Benchmark.md",
}
EXPECTED_PLANNED_ARTIFACTS = {
    "registration_freeze_receipt": (
        f"{RECOVERY_REGISTRATION_ROOT}/registration_freeze_receipt_v2.json"
    ),
    "suite_export": f"{RECOVERY_REGISTRATION_ROOT}/suite_v2_copy_export_v2",
    "export_reservation": (
        f"{RECOVERY_REGISTRATION_ROOT}/export_reservation_v2.json"
    ),
    "copy_complete": (
        f"{RECOVERY_REGISTRATION_ROOT}/suite_v2_copy_complete_v2.json"
    ),
    "component_audit": (
        f"{RECOVERY_REGISTRATION_ROOT}/component_release_audit_v2.json"
    ),
    "suite_audit": f"{RECOVERY_REGISTRATION_ROOT}/suite_v2_audit_v2.json",
    "export_audit": (
        f"{RECOVERY_REGISTRATION_ROOT}/suite_v2_export_audit_v2.json"
    ),
    "registration_decision": (
        f"{RECOVERY_REGISTRATION_ROOT}/registration_decision_v2.json"
    ),
}
EXPECTED_ALLOWED_CLAIMS = {
    "benchmark_component_status": "ready",
    "reference_result_status": "passed_public_test_3_of_3",
    "suite_membership": SUITE_RELEASE_ID,
    "distribution": "local_technical_release_candidate",
    "registration_recovery": "direct_reservation_v2",
}
EXPECTED_PROHIBITED_CLAIMS = [
    "cube_public_test_was_rerun_during_registration_recovery",
    "cube_formal_checkpoint_was_opened_during_registration_recovery",
    "prior_failed_registration_namespace_was_deleted_renamed_or_repaired",
    "prior_failed_staging_was_promoted_or_reused",
    "directory_rename_was_used_by_recovery_v2",
    "suite_v1_was_rewritten_as_a_nine_component_release",
    "public_distribution_ready_without_license_and_download_configuration",
]
EXPECTED_FORMAL_ABSENCES = [
    f"{PRIOR_REGISTRATION_ROOT}/component_release_audit_v1.json",
    f"{PRIOR_REGISTRATION_ROOT}/suite_v2_audit_v1.json",
    f"{PRIOR_REGISTRATION_ROOT}/suite_v2_export_audit_v1.json",
    f"{PRIOR_REGISTRATION_ROOT}/registration_decision_v1.json",
    f"{PRIOR_REGISTRATION_ROOT}/suite_v2_copy_export_v1",
]
EXPECTED_OPERATIONAL_OBSERVATION = {
    "authoritative_for_recovery": False,
    "operation": "os.replace(export_staging, export_destination)",
    "errno": 1,
    "exception": "PermissionError",
    "message": "Operation not permitted",
    "traceback_artifact_registered": False,
    "claim_boundary": (
        "The interactive stderr established the failing operation, but no "
        "standalone traceback artifact was preregistered. This note is not an "
        "authorization basis: only the preserved staging tree and absence of "
        "all formal outputs are frozen machine-auditable evidence."
    ),
}


def _assert_file_identity_specification(
    value: Any, *, expected_path: str, label: str
) -> None:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"path", "sha256", "size_bytes"}
        or value.get("path") != expected_path
        or len(str(value.get("sha256", ""))) != 64
        or type(value.get("size_bytes")) is not int
        or value["size_bytes"] <= 0
    ):
        raise RuntimeError(f"Invalid file identity specification: {label}")


def _assert_tree_identity_specification(
    value: Any, *, expected_path: str, label: str
) -> None:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"path", "files", "bytes", "sha256"}
        or value.get("path") != expected_path
        or type(value.get("files")) is not int
        or value["files"] <= 0
        or type(value.get("bytes")) is not int
        or value["bytes"] <= 0
        or len(str(value.get("sha256", ""))) != 64
    ):
        raise RuntimeError(f"Invalid tree identity specification: {label}")


def validate_registration_recovery_preregistration_contract(
    preregistration: Mapping[str, Any],
    *,
    preregistration_path: Path,
) -> None:
    expected_path = lexical_absolute(
        repository_root() / RECOVERY_PREREGISTRATION_LOGICAL_PATH
    )
    if lexical_absolute(preregistration_path) != expected_path:
        raise RuntimeError(
            "Recovery freeze/finalizer must use the canonical preregistration"
        )
    expected_keys = {
        "schema_version",
        "registration_id",
        "component_id",
        "release_id",
        "suite_release_id",
        "status",
        "registered_date",
        "scope",
        "prior_failed_registration",
        "recovery_contract",
        "registration_gates",
        "planned_repository_outputs",
        "planned_artifacts",
        "allowed_claim_after_all_gates_pass",
        "prohibited_claims",
    }
    if set(preregistration) != expected_keys:
        raise RuntimeError("Recovery preregistration keys drifted")
    scalar_expected = {
        "schema_version": 1,
        "registration_id": RECOVERY_REGISTRATION_ID,
        "component_id": COMPONENT_ID,
        "release_id": RELEASE_ID,
        "suite_release_id": SUITE_RELEASE_ID,
        "status": "suite_registration_infrastructure_recovery_preregistered",
        "registered_date": "2026-08-14",
    }
    if any(
        preregistration.get(key) != expected
        for key, expected in scalar_expected.items()
    ):
        raise RuntimeError("Recovery preregistration scalar identity drifted")
    for key, expected in (
        ("scope", EXPECTED_SCOPE),
        ("recovery_contract", EXPECTED_RECOVERY_CONTRACT),
        ("registration_gates", EXPECTED_REGISTRATION_GATES),
        (
            "planned_repository_outputs",
            EXPECTED_PLANNED_REPOSITORY_OUTPUTS,
        ),
        ("planned_artifacts", EXPECTED_PLANNED_ARTIFACTS),
        ("allowed_claim_after_all_gates_pass", EXPECTED_ALLOWED_CLAIMS),
        ("prohibited_claims", EXPECTED_PROHIBITED_CLAIMS),
    ):
        if not exact_value_equal(preregistration.get(key), expected):
            raise RuntimeError(f"Recovery preregistration {key} drifted")

    prior = preregistration.get("prior_failed_registration")
    if not isinstance(prior, Mapping) or set(prior) != {
        "registration_id",
        "registration_root",
        "preregistration",
        "freeze_receipt",
        "failed_staging",
        "anchors",
        "required_semantics",
        "formal_outputs_required_absent",
        "operational_observation",
        "preservation",
    }:
        raise RuntimeError("Prior failed-registration evidence drifted")
    if (
        prior.get("registration_id") != PRIOR_REGISTRATION_ID
        or prior.get("registration_root") != PRIOR_REGISTRATION_ROOT
        or prior.get("formal_outputs_required_absent")
        != EXPECTED_FORMAL_ABSENCES
        or not exact_value_equal(
            prior.get("operational_observation"),
            EXPECTED_OPERATIONAL_OBSERVATION,
        )
        or prior.get("preservation")
        != {
            "delete_authorized": False,
            "rename_authorized": False,
            "repair_or_retry_in_place_authorized": False,
            "reuse_as_recovery_export_source_authorized": False,
        }
    ):
        raise RuntimeError("Prior failed-registration boundary drifted")
    _assert_file_identity_specification(
        prior.get("preregistration"),
        expected_path=PRIOR_PREREGISTRATION_LOGICAL_PATH,
        label="prior preregistration",
    )
    _assert_file_identity_specification(
        prior.get("freeze_receipt"),
        expected_path=PRIOR_FREEZE_RECEIPT_LOGICAL_PATH,
        label="prior freeze receipt",
    )
    _assert_tree_identity_specification(
        prior.get("failed_staging"),
        expected_path=PRIOR_FAILED_STAGING_LOGICAL_PATH,
        label="prior failed staging",
    )
    anchor_paths = {
        "readme": f"{PRIOR_FAILED_STAGING_LOGICAL_PATH}/README.md",
        "inventory": (
            f"{PRIOR_FAILED_STAGING_LOGICAL_PATH}/benchmark/inventory.json"
        ),
        "suite_config": (
            f"{PRIOR_FAILED_STAGING_LOGICAL_PATH}/benchmark/suite.yaml"
        ),
        "cube_release_config": (
            f"{PRIOR_FAILED_STAGING_LOGICAL_PATH}/benchmark/releases/"
            "cube_gripper_carry.yaml"
        ),
    }
    anchors = prior.get("anchors")
    if not isinstance(anchors, Mapping) or set(anchors) != set(anchor_paths):
        raise RuntimeError("Prior failed-staging anchors drifted")
    for name, path in anchor_paths.items():
        _assert_file_identity_specification(
            anchors[name], expected_path=path, label=f"prior staging {name}"
        )
    if prior.get("required_semantics") != {
        "membership_active": False,
        "membership_status": "pending_registration_internal_audit",
        "suite_decision_path": (
            f"{PRIOR_REGISTRATION_ROOT}/registration_decision_v1.json"
        ),
    }:
        raise RuntimeError("Prior failed-staging semantics drifted")


def validate_prior_failed_registration_evidence(
    preregistration: Mapping[str, Any],
    *,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    root = (repo_root or repository_root()).resolve()
    prior = preregistration["prior_failed_registration"]
    prior_preregistration = require_no_symlink_components(
        root / PRIOR_PREREGISTRATION_LOGICAL_PATH,
        anchor=root,
        label="prior Suite-registration preregistration",
    )
    prior_freeze = resolve_no_symlink_contextworld_path(
        PRIOR_FREEZE_RECEIPT_LOGICAL_PATH,
        repo_root=root,
        label="prior Suite-registration freeze receipt",
    )
    prior_staging = resolve_no_symlink_contextworld_path(
        PRIOR_FAILED_STAGING_LOGICAL_PATH,
        repo_root=root,
        label="prior failed Suite export staging",
    )
    observed_preregistration = file_identity(
        prior_preregistration,
        logical_path=PRIOR_PREREGISTRATION_LOGICAL_PATH,
    )
    observed_freeze = file_identity(
        prior_freeze,
        logical_path=PRIOR_FREEZE_RECEIPT_LOGICAL_PATH,
    )
    observed_staging = {
        "path": PRIOR_FAILED_STAGING_LOGICAL_PATH,
        **tree_identity(prior_staging),
    }
    anchor_relatives = {
        "readme": "README.md",
        "inventory": "benchmark/inventory.json",
        "suite_config": "benchmark/suite.yaml",
        "cube_release_config": "benchmark/releases/cube_gripper_carry.yaml",
    }
    observed_anchors = {
        name: file_identity(
            prior_staging / relative,
            logical_path=(
                f"{PRIOR_FAILED_STAGING_LOGICAL_PATH}/{relative}"
            ),
        )
        for name, relative in anchor_relatives.items()
    }
    if not identity_equal(prior["preregistration"], observed_preregistration):
        raise RuntimeError("Prior preregistration identity drifted")
    if not identity_equal(prior["freeze_receipt"], observed_freeze):
        raise RuntimeError("Prior freeze-receipt identity drifted")
    if prior["failed_staging"] != observed_staging:
        raise RuntimeError("Prior failed staging tree identity drifted")
    if prior["anchors"] != observed_anchors:
        raise RuntimeError("Prior failed staging anchor identity drifted")
    freeze_payload = read_json(
        prior_freeze, label="prior Suite-registration freeze receipt"
    )
    if (
        freeze_payload.get("receipt_id") != PRIOR_FREEZE_RECEIPT_ID
        or freeze_payload.get("registration_id") != PRIOR_REGISTRATION_ID
        or freeze_payload.get("checks_passed") is not True
    ):
        raise RuntimeError("Prior freeze receipt is not the frozen v1 receipt")
    required_bundle_files = (
        prior_staging / "README.md",
        prior_staging / "benchmark/suite.yaml",
        prior_staging / "benchmark/inventory.json",
    )
    if not all(path.is_file() and not path.is_symlink() for path in required_bundle_files):
        raise RuntimeError("Prior failed staging is not a complete Suite bundle")
    inventory = read_json(
        prior_staging / "benchmark/inventory.json",
        label="prior failed Suite export inventory",
    )
    suite_payload = read_yaml(
        prior_staging / "benchmark/suite.yaml",
        label="prior failed Suite export config",
    )
    observed_semantics = {
        "membership_active": inventory.get("membership_activation", {}).get(
            "active"
        ),
        "membership_status": inventory.get("membership_activation", {}).get(
            "status"
        ),
        "suite_decision_path": suite_payload.get(
            "membership_authority", {}
        ).get("decision_path"),
    }
    if observed_semantics != prior["required_semantics"]:
        raise RuntimeError("Prior failed staging membership semantics drifted")
    observed_absences = {}
    for logical in EXPECTED_FORMAL_ABSENCES:
        path = resolve_no_symlink_contextworld_path(
            logical,
            repo_root=root,
            label=f"prior formal-output absence {logical}",
            allow_missing=True,
        )
        observed_absences[logical] = not os.path.lexists(path)
    if not all(observed_absences.values()):
        raise RuntimeError(
            "Prior failed namespace contains a formal output that should be absent"
        )
    return {
        "preregistration": observed_preregistration,
        "freeze_receipt": observed_freeze,
        "failed_staging": observed_staging,
        "anchors": observed_anchors,
        "semantics": observed_semantics,
        "formal_outputs_absent": observed_absences,
        "prior_namespace_preserved": True,
        "passed": True,
    }


__all__ = [
    "RECOVERY_FREEZE_RECEIPT_ID",
    "RECOVERY_PREREGISTRATION_LOGICAL_PATH",
    "RECOVERY_REGISTRATION_ID",
    "RECOVERY_REGISTRATION_ROOT",
    "validate_prior_failed_registration_evidence",
    "validate_registration_recovery_preregistration_contract",
]
