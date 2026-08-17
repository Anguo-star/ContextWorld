from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
import shutil
import stat
from pathlib import Path
from typing import Any, Iterable

import yaml

from contextworld.benchmarks.action_delay_icl_data import (
    audit_action_delay_icl_release,
    load_action_delay_icl_release,
)
from contextworld.benchmarks.action_strength_icl_data import (
    audit_action_strength_icl_release,
    load_action_strength_icl_release,
    resolve_action_strength_initial_checkpoint,
    resolve_action_strength_original_h5,
    resolve_action_strength_original_lance,
)
from contextworld.benchmarks.contact_friction_icl_data import (
    audit_contact_friction_icl_release,
    load_contact_friction_icl_release,
)
from contextworld.benchmarks.cube_grasp_rule_v4r1_icl_data import (
    audit_cube_grasp_rule_v4r1_icl_release,
    load_cube_grasp_rule_v4r1_icl_release,
)
from contextworld.benchmarks.cube_grasp_rule_suite_registration import (
    file_identity as _registered_file_identity,
    resolve_no_symlink_contextworld_path,
    tree_identity as _registered_tree_identity,
)
from contextworld.benchmarks.cube_grasp_rule_suite_registration_recovery import (
    RECOVERY_REGISTRATION_ID as CUBE_SUITE_REGISTRATION_ID,
    RECOVERY_REGISTRATION_ROOT,
)
from contextworld.benchmarks.motion_damping_icl_data import (
    audit_motion_damping_icl_release,
    load_motion_damping_icl_release,
)
from contextworld.benchmarks.portal_exit_icl_data import (
    audit_portal_exit_icl_release,
    load_portal_exit_icl_release,
    resolve_portal_original_lance,
)
from contextworld.benchmarks.public_score import (
    make_public_scoreboard_from_spec,
)
from contextworld.benchmarks.reacher_arm_mass_icl_data import (
    audit_reacher_arm_mass_icl_release,
    load_reacher_arm_mass_icl_release,
    resolve_reacher_initial_checkpoint,
    resolve_reacher_initial_checkpoint_config,
    resolve_reacher_original_h5,
    resolve_reacher_original_lance,
)
from contextworld.benchmarks.door_icl_data import (
    audit_door_icl_release,
    door_icl_export_entries,
    load_door_icl_release,
)
from contextworld.benchmarks.speed_icl_data import (
    audit_speed_icl_release,
    load_speed_icl_release,
    resolve_original_h5,
)
from contextworld.paths import repository_root, resolve_contextworld_path


SUITE_RELEASE_ID = "contextworld_icl_benchmark_suite_v1"
SUITE_V2_RELEASE_ID = "contextworld_icl_benchmark_suite_v2"
SUPPORTED_SUITE_RELEASE_IDS = {SUITE_RELEASE_ID, SUITE_V2_RELEASE_ID}
DEFAULT_SUITE_RELEASE_CONFIG = (
    repository_root()
    / "configs/benchmark/contextworld_icl_suite_v1.yaml"
)
SUITE_V2_RECOVERY_CONFIG = (
    repository_root()
    / "configs/benchmark/contextworld_icl_suite_v2_recovery_v2.yaml"
)
DEFAULT_SUITE_V2_RELEASE_CONFIG = (
    repository_root()
    / "configs/benchmark/"
    "contextworld_icl_suite_v2_public_document_amendment_v1.yaml"
)
COMPONENT_IDS = (
    "speed",
    "door",
    "action_delay",
    "action_strength",
    "contact_friction",
    "motion_damping",
    "robot_arm_mass",
    "portal_exit",
)
SUITE_V2_COMPONENT_IDS = (*COMPONENT_IDS, "cube_gripper_carry")
REFERENCE_RESULT_STATUSES = {
    "speed": "passed_public_test_3_of_3",
    "door": "passed_public_test_3_of_3",
    "action_delay": "passed_public_test_3_of_3",
    "action_strength": "passed_public_test_3_of_3",
    "contact_friction": "failed_development",
    "motion_damping": "failed_development",
    "robot_arm_mass": "passed_public_test_3_of_3",
    "portal_exit": "failed_public_test_0_of_3",
}
SUITE_V2_REFERENCE_RESULT_STATUSES = {
    **REFERENCE_RESULT_STATUSES,
    "cube_gripper_carry": "passed_public_test_3_of_3",
}
SUITE_V2_REGISTRATION_DECISION = (
    f"{RECOVERY_REGISTRATION_ROOT}/registration_decision_v2.json"
)
SUITE_V2_RECOVERY_EXPORT = (
    f"{RECOVERY_REGISTRATION_ROOT}/suite_v2_copy_export_v2"
)
SUITE_V2_CONFIG_LOGICAL_PATH = (
    "configs/benchmark/contextworld_icl_suite_v2_recovery_v2.yaml"
)
SUITE_V2_DOCUMENT_AMENDMENT_ID = (
    "contextworld_icl_suite_v2_public_document_amendment_v1"
)
SUITE_V2_DOCUMENT_AMENDMENT_CONFIG_LOGICAL_PATH = (
    "configs/benchmark/"
    "contextworld_icl_suite_v2_public_document_amendment_v1.yaml"
)
SUITE_V2_DOCUMENT_AMENDMENT_DECISION = (
    "configs/benchmark/"
    "contextworld_icl_suite_v2_public_document_amendment_decision_v1.json"
)
SUITE_V2_DOCUMENT_AMENDMENT_ACTIVATION = (
    "passed_public_document_amendment_decision_v1"
)
SUITE_V2_BASE_PUBLIC_DOCUMENT_SHA256 = (
    "72031232d008b77f809d387348f8bc320532f80517e387837571a2995932cccc"
)
SUITE_V2_BASE_CONFIG_SHA256 = (
    "3b03f759dfdb934dcfd4f08a59d1385d2ed536fc62ed6b597c83ff18540f6d4e"
)
SUITE_V2_BASE_DECISION_SHA256 = (
    "0c6d38ec4304d0ffa078fe8680a7b6d90b851eb69780d22936a456bf0bb7859d"
)
SUITE_V2_BASE_EXPORT_PUBLIC_DOCUMENT = (
    f"{SUITE_V2_RECOVERY_EXPORT}/README.md"
)
SUITE_V2_DOCUMENT_AMENDMENT_SOURCE_OVERRIDES = {
    "contextworld/benchmarks/suite_data.py",
    "scripts/finalize_cube_grasp_rule_h3_v4r1_"
    "suite_registration_recovery_v2.py",
}
SUITE_V2_PRE_RECOVERY_DECISION = (
    "artifacts/evaluation/history3/"
    "cube_gripper_carry_h3_v4r1_suite_registration_v1/"
    "registration_decision_v1.json"
)
SUITE_V2_REQUIRED_EVIDENCE_PATHS = {
    "preregistration": (
        "configs/benchmark/"
        "cube_gripper_carry_h3_v4r1_"
        "suite_registration_recovery_v2_prereg.yaml"
    ),
    "freeze_receipt": (
        "artifacts/evaluation/history3/"
        "cube_gripper_carry_h3_v4r1_suite_registration_recovery_v2/"
        "registration_freeze_receipt_v2.json"
    ),
    "cube_release_config": (
        "configs/benchmark/"
        "cube_gripper_carry_h3_v4r1_icl_release_v1.yaml"
    ),
    "suite_v2_config": SUITE_V2_CONFIG_LOGICAL_PATH,
    "suite_v1_historical_config": (
        "configs/benchmark/contextworld_icl_suite_v1.yaml"
    ),
    "component_audit": (
        "artifacts/evaluation/history3/"
        "cube_gripper_carry_h3_v4r1_suite_registration_recovery_v2/"
        "component_release_audit_v2.json"
    ),
    "suite_audit": (
        "artifacts/evaluation/history3/"
        "cube_gripper_carry_h3_v4r1_suite_registration_recovery_v2/"
        "suite_v2_audit_v2.json"
    ),
    "export_audit": (
        "artifacts/evaluation/history3/"
        "cube_gripper_carry_h3_v4r1_suite_registration_recovery_v2/"
        "suite_v2_export_audit_v2.json"
    ),
    "export_reservation": (
        "artifacts/evaluation/history3/"
        "cube_gripper_carry_h3_v4r1_suite_registration_recovery_v2/"
        "export_reservation_v2.json"
    ),
    "copy_complete": (
        "artifacts/evaluation/history3/"
        "cube_gripper_carry_h3_v4r1_suite_registration_recovery_v2/"
        "suite_v2_copy_complete_v2.json"
    ),
}
_SUITE_V2_REGISTRATION_AUDIT_CAPABILITY = object()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_suite_v2_membership_authority(
    suite: dict[str, Any],
) -> dict[str, Any]:
    authority = suite.get("membership_authority")
    expected = {
        "config_alone_grants_membership": False,
        "activation_condition": "passed_registration_decision_v2",
        "registration_id": CUBE_SUITE_REGISTRATION_ID,
        "decision_path": SUITE_V2_REGISTRATION_DECISION,
        "decision_is_commit_marker": True,
        "partial_outputs_grant_membership": False,
        "failed_finalization_requires_new_preregistration": True,
        "recovery_protocol": "direct_one_use_export_reservation_no_directory_rename",
        "prior_failed_registration_id": (
            "contextworld_cube_gripper_carry_h3_v4r1_suite_registration_v1"
        ),
        "directory_rename_authorized": False,
        "prior_failed_staging_reuse_authorized": False,
    }
    historical = {
        "config_alone_grants_membership": False,
        "activation_condition": "passed_registration_decision_v1",
        "decision_path": SUITE_V2_PRE_RECOVERY_DECISION,
        "decision_is_commit_marker": True,
        "partial_outputs_grant_membership": False,
        "failed_finalization_requires_new_preregistration": True,
    }
    documentation_amendment = {
        "config_alone_grants_membership": False,
        "activation_condition": SUITE_V2_DOCUMENT_AMENDMENT_ACTIVATION,
        "amendment_id": SUITE_V2_DOCUMENT_AMENDMENT_ID,
        "decision_path": SUITE_V2_DOCUMENT_AMENDMENT_DECISION,
        "decision_is_commit_marker": True,
        "base_registration_id": CUBE_SUITE_REGISTRATION_ID,
        "base_decision_path": SUITE_V2_REGISTRATION_DECISION,
        "base_release_config": SUITE_V2_CONFIG_LOGICAL_PATH,
        "base_membership_must_remain_active": True,
        "partial_outputs_grant_membership": False,
        "public_test_rerun_authorized": False,
        "training_or_checkpoint_selection_authorized": False,
        "formal_scoreboard_mutation_authorized": False,
        "component_release_mutation_authorized": False,
    }
    if not isinstance(authority, dict) or not (
        all(authority.get(key) == value for key, value in expected.items())
        or all(
            authority.get(key) == value for key, value in historical.items()
        )
        or all(
            authority.get(key) == value
            for key, value in documentation_amendment.items()
        )
    ):
        raise ValueError("Suite v2 membership authority is not fail-closed")
    return authority


def _audit_registered_identity(
    identity: Any,
    *,
    expected_path: str,
    repo_root: Path,
) -> dict[str, Any]:
    if not isinstance(identity, dict) or identity.get("path") != expected_path:
        raise RuntimeError(
            f"Suite v2 decision evidence path drifted: {expected_path}"
        )
    expected_sha256 = identity.get("sha256")
    expected_size = identity.get("size_bytes")
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise RuntimeError(
            f"Suite v2 decision evidence hash is invalid: {expected_path}"
        )
    if type(expected_size) is not int or expected_size < 0:
        raise RuntimeError(
            f"Suite v2 decision evidence size is invalid: {expected_path}"
        )
    path = resolve_no_symlink_contextworld_path(
        expected_path,
        repo_root=repo_root,
        label=f"Suite v2 decision evidence {expected_path}",
    )
    if not path.is_file():
        raise RuntimeError(
            f"Suite v2 decision evidence is missing: {expected_path}"
        )
    observed_sha256 = _sha256(path)
    observed_size = path.stat().st_size
    if observed_sha256 != expected_sha256 or observed_size != expected_size:
        raise RuntimeError(
            f"Suite v2 decision evidence identity drifted: {expected_path}"
        )
    return {
        "path": str(path),
        "logical_path": expected_path,
        "sha256": observed_sha256,
        "size_bytes": observed_size,
        "passed": True,
    }


def _validate_recovery_v2_export_commit(
    evidence: dict[str, Any],
    evidence_audits: dict[str, dict[str, Any]],
    *,
    repo_root: Path,
) -> dict[str, Any]:
    """Revalidate the recovery receipts and the complete committed export."""

    reservation = _read_json_object(
        Path(evidence_audits["export_reservation"]["path"])
    )
    copy_complete = _read_json_object(
        Path(evidence_audits["copy_complete"]["path"])
    )
    expected_reservation_keys = {
        "schema_version",
        "registration_id",
        "status",
        "export_path",
        "reservation_operation",
        "directory_rename_authorized",
        "prior_failed_staging_reuse_authorized",
        "registration_decision_is_only_membership_commit_marker",
        "partial_outputs_grant_membership",
        "preregistration",
        "freeze_receipt",
        "passed",
    }
    if (
        set(reservation) != expected_reservation_keys
        or reservation.get("schema_version") != 1
        or reservation.get("registration_id")
        != CUBE_SUITE_REGISTRATION_ID
        or reservation.get("status")
        != "direct_export_exclusively_reserved"
        or reservation.get("export_path") != SUITE_V2_RECOVERY_EXPORT
        or reservation.get("reservation_operation")
        != "mkdir_exist_ok_false"
        or reservation.get("directory_rename_authorized") is not False
        or reservation.get("prior_failed_staging_reuse_authorized")
        is not False
        or reservation.get(
            "registration_decision_is_only_membership_commit_marker"
        )
        is not True
        or reservation.get("partial_outputs_grant_membership") is not False
        or reservation.get("preregistration")
        != evidence.get("preregistration")
        or reservation.get("freeze_receipt")
        != evidence.get("freeze_receipt")
        or reservation.get("passed") is not True
    ):
        raise RuntimeError(
            "Suite v2 membership is not active: export reservation drifted"
        )

    expected_copy_complete_keys = {
        "schema_version",
        "registration_id",
        "status",
        "export_path",
        "export_tree",
        "copy_manifest",
        "fresh_copy_from_canonical_sources",
        "prior_failed_staging_reused",
        "prior_failed_namespace_mutated",
        "directory_rename_used",
        "exclusive_file_creation_required",
        "passed",
    }
    if (
        set(copy_complete) != expected_copy_complete_keys
        or copy_complete.get("schema_version") != 1
        or copy_complete.get("registration_id")
        != CUBE_SUITE_REGISTRATION_ID
        or copy_complete.get("status") != "fresh_direct_copy_complete"
        or copy_complete.get("export_path") != SUITE_V2_RECOVERY_EXPORT
        or copy_complete.get("fresh_copy_from_canonical_sources") is not True
        or copy_complete.get("prior_failed_staging_reused") is not False
        or copy_complete.get("prior_failed_namespace_mutated") is not False
        or copy_complete.get("directory_rename_used") is not False
        or copy_complete.get("exclusive_file_creation_required") is not True
        or copy_complete.get("passed") is not True
    ):
        raise RuntimeError(
            "Suite v2 membership is not active: copy-complete receipt drifted"
        )

    export_path = resolve_no_symlink_contextworld_path(
        SUITE_V2_RECOVERY_EXPORT,
        repo_root=repo_root,
        label="Suite v2 recovery committed export",
    )
    observed_tree = _registered_tree_identity(export_path)
    if copy_complete.get("export_tree") != observed_tree:
        raise RuntimeError(
            "Suite v2 membership is not active: committed export tree drifted"
        )
    inventory_logical = (
        f"{SUITE_V2_RECOVERY_EXPORT}/benchmark/inventory.json"
    )
    inventory_path = export_path / "benchmark/inventory.json"
    observed_manifest = _registered_file_identity(
        inventory_path,
        logical_path=inventory_logical,
    )
    if copy_complete.get("copy_manifest") != observed_manifest:
        raise RuntimeError(
            "Suite v2 membership is not active: export manifest drifted"
        )
    inventory = _read_json_object(inventory_path)
    activation = inventory.get("membership_activation")
    if (
        inventory.get("schema_version") != 1
        or inventory.get("release_id") != SUITE_V2_RELEASE_ID
        or inventory.get("status") != "passed"
        or inventory.get("mode") != "copy"
        or inventory.get("components") != list(SUITE_V2_COMPONENT_IDS)
        or not isinstance(activation, dict)
        or activation.get("active") is not False
        or activation.get("status")
        != "pending_registration_internal_audit"
        or activation.get("decision_path") != SUITE_V2_REGISTRATION_DECISION
        or activation.get("partial_outputs_grant_membership") is not False
    ):
        raise RuntimeError(
            "Suite v2 membership is not active: export inventory drifted"
        )
    export_audit = _read_json_object(
        Path(evidence_audits["export_audit"]["path"])
    )
    copy_export = export_audit.get("copy_export")
    bundle_reaudit = export_audit.get("bundle_reaudit")
    direct_commit = export_audit.get("direct_export_commit")
    prior_registration = export_audit.get("prior_failed_registration")
    if (
        export_audit.get("status") != "passed"
        or export_audit.get("passed") is not True
        or export_audit.get("copy_completion") != copy_complete
        or not isinstance(copy_export, dict)
        or copy_export.get("status") != "passed"
        or copy_export.get("mode") != "copy"
        or copy_export.get("destination") != str(export_path)
        or copy_export.get("components") != list(SUITE_V2_COMPONENT_IDS)
        or not isinstance(direct_commit, dict)
        or direct_commit.get("direct_target_exclusively_reserved") is not True
        or direct_commit.get("fresh_copy_tree_identity_verified") is not True
        or direct_commit.get(
            "bundle_reaudit_completed_before_formal_audit_writes"
        )
        is not True
        or direct_commit.get("directory_rename_used") is not False
        or direct_commit.get("committed_destination") != str(export_path)
        or direct_commit.get(
            "registration_decision_is_only_membership_commit_marker"
        )
        is not True
        or not isinstance(bundle_reaudit, dict)
        or bundle_reaudit.get("passed") is not True
        or not isinstance(prior_registration, dict)
        or prior_registration.get("registration_id")
        != "contextworld_cube_gripper_carry_h3_v4r1_suite_registration_v1"
        or prior_registration.get("staging_reused") is not False
        or prior_registration.get("namespace_mutated") is not False
        or export_audit.get("cube_public_test_rerun") is not False
        or export_audit.get("cube_formal_checkpoint_opened") is not False
    ):
        raise RuntimeError(
            "Suite v2 membership is not active: export audit drifted"
        )
    return {
        "export_path": str(export_path),
        "tree_identity": observed_tree,
        "manifest": observed_manifest,
        "reservation_passed": True,
        "copy_complete_passed": True,
        "passed": True,
    }


def _require_suite_documentation_amendment_activation(
    suite: dict[str, Any], *, repo_root: Path
) -> dict[str, Any]:
    """Validate the documentation-only layer over the frozen recovery-v2 release."""

    authority = suite["membership_authority"]
    decision_path = resolve_no_symlink_contextworld_path(
        authority["decision_path"],
        repo_root=repo_root,
        label="Suite v2 documentation amendment decision",
    )
    if not decision_path.is_file():
        raise RuntimeError(
            "Suite v2 membership is not active: documentation amendment "
            "decision is missing"
        )
    decision = _read_json_object(decision_path)
    base_suite = load_icl_suite_release(SUITE_V2_RECOVERY_CONFIG)
    base_activation = _require_suite_membership_activation(
        base_suite, repo_root=repo_root
    )
    if base_activation.get("active") is not True:
        raise RuntimeError(
            "Suite v2 documentation amendment requires active recovery-v2 membership"
        )
    expected_claims = {
        "amendment_scope": "documentation_only_reference_table_expansion",
        "base_membership_preserved": True,
        "component_release_mutated": False,
        "formal_scoreboard_mutated": False,
        "formal_scoreboard_rows_added": 0,
        "public_test_rerun": False,
        "training_or_checkpoint_selection": False,
        "threshold_or_recipe_changed": False,
        "original_lewm_icl_score_invented": False,
        "pldm_result_scope": "development_only_not_public",
        "external_model_slots": "empty_not_run_not_authorized",
    }
    expected_reference_table = {
        "cube_comparison_rows": [
            {
                "comparison_id": "original_lewm",
                "model_family": "lewm",
                "icl_scope": "not_evaluated_under_v4r1",
                "icl_score": "not_evaluated",
                "cem_successes": 198,
                "cem_trials": 300,
            },
            {
                "comparison_id": "trained_lewm",
                "model_family": "lewm",
                "icl_scope": "public",
                "public_correct_future_rates": [
                    0.77734375,
                    0.791015625,
                    0.78515625,
                ],
                "public_correct_future_rate_mean": 0.7845052083333334,
                "public_decision": "passed_3_of_3",
                "cem_successes_by_training_seed": [186, 183, 185],
                "cem_trials_per_training_seed": 300,
                "planning_decision": "passed_retention",
            },
            {
                "comparison_id": "trained_pldm",
                "model_family": "pldm",
                "icl_scope": "development_only",
                "development_correct_future_rates": [
                    0.501953125,
                    0.501953125,
                    0.5,
                ],
                "development_correct_future_rate_mean": 0.5013020833333334,
                "development_decision": "failed_0_of_3",
                "public_score": "not_authorized_not_run",
                "cem_score": "not_authorized_not_run",
            },
        ],
        "external_model_slots": [
            "External-01",
            "External-02",
            "External-03",
        ],
        "formal_scoreboard_rows_added": 0,
    }
    expected_decision_keys = {
        "schema_version",
        "amendment_id",
        "suite_release_id",
        "status",
        "passed",
        "amendment_config",
        "base_release_config",
        "base_registration_decision",
        "base_export_public_document",
        "amended_public_document",
        "amended_suite_data",
        "amended_recovery_finalizer",
        "unchanged_public_results",
        "reference_table",
        "claims",
    }
    unchanged = decision.get("unchanged_public_results")
    if (
        set(decision) != expected_decision_keys
        or decision.get("schema_version") != 1
        or decision.get("amendment_id") != SUITE_V2_DOCUMENT_AMENDMENT_ID
        or decision.get("suite_release_id") != SUITE_V2_RELEASE_ID
        or decision.get("status") != "documentation_amendment_passed"
        or decision.get("passed") is not True
        or decision.get("claims") != expected_claims
        or decision.get("reference_table") != expected_reference_table
        or not isinstance(unchanged, dict)
        or set(unchanged)
        != {
            "formal_reference_rows",
            "formal_reference_components",
            "specification",
            "scoreboard",
            "cube_formal_rows",
            "cube_formal_family",
            "external_results_included",
        }
        or unchanged.get("formal_reference_rows") != 11
        or unchanged.get("formal_reference_components") != 7
        or unchanged.get("cube_formal_rows") != 1
        or unchanged.get("cube_formal_family") != "lewm"
        or unchanged.get("external_results_included") is not False
    ):
        raise RuntimeError(
            "Suite v2 membership is not active: documentation amendment "
            "decision failed its exact contract"
        )

    identities = {
        "amendment_config": (
            SUITE_V2_DOCUMENT_AMENDMENT_CONFIG_LOGICAL_PATH,
            decision.get("amendment_config"),
        ),
        "base_release_config": (
            SUITE_V2_CONFIG_LOGICAL_PATH,
            decision.get("base_release_config"),
        ),
        "base_registration_decision": (
            SUITE_V2_REGISTRATION_DECISION,
            decision.get("base_registration_decision"),
        ),
        "base_export_public_document": (
            SUITE_V2_BASE_EXPORT_PUBLIC_DOCUMENT,
            decision.get("base_export_public_document"),
        ),
        "amended_public_document": (
            suite["repository"]["public_document"]["path"],
            decision.get("amended_public_document"),
        ),
        "amended_suite_data": (
            "contextworld/benchmarks/suite_data.py",
            decision.get("amended_suite_data"),
        ),
        "amended_recovery_finalizer": (
            "scripts/finalize_cube_grasp_rule_h3_v4r1_"
            "suite_registration_recovery_v2.py",
            decision.get("amended_recovery_finalizer"),
        ),
        "specification": (
            suite["public_results"]["specification"]["path"],
            unchanged.get("specification"),
        ),
        "scoreboard": (
            suite["public_results"]["scoreboard"]["path"],
            unchanged.get("scoreboard"),
        ),
    }
    identity_audits = {
        name: _audit_registered_identity(
            identity, expected_path=logical, repo_root=repo_root
        )
        for name, (logical, identity) in identities.items()
    }
    loaded_config = Path(suite["_config_path"])
    if (
        _sha256(loaded_config) != decision["amendment_config"]["sha256"]
        or loaded_config.stat().st_size
        != decision["amendment_config"]["size_bytes"]
        or decision["base_release_config"]["sha256"]
        != SUITE_V2_BASE_CONFIG_SHA256
        or decision["base_registration_decision"]["sha256"]
        != SUITE_V2_BASE_DECISION_SHA256
        or decision["base_export_public_document"]["sha256"]
        != SUITE_V2_BASE_PUBLIC_DOCUMENT_SHA256
        or decision["amended_public_document"]["sha256"]
        != suite["repository"]["public_document"]["sha256"]
        or decision["amended_suite_data"]["sha256"]
        != suite["repository"]["source_sha256"][
            "contextworld/benchmarks/suite_data.py"
        ]
        or decision["amended_recovery_finalizer"]["sha256"]
        != suite["repository"]["source_sha256"][
            "scripts/finalize_cube_grasp_rule_h3_v4r1_"
            "suite_registration_recovery_v2.py"
        ]
        or unchanged["specification"]["sha256"]
        != suite["public_results"]["specification"]["sha256"]
        or unchanged["scoreboard"]["sha256"]
        != suite["public_results"]["scoreboard"]["sha256"]
    ):
        raise RuntimeError(
            "Suite v2 documentation amendment identities do not match the release"
        )

    invariant_sections = (
        "scope",
        "model_interface",
        "public_results",
        "bundle",
        "components",
        "extension",
        "distribution",
    )
    if any(suite[key] != base_suite[key] for key in invariant_sections):
        raise RuntimeError(
            "Suite v2 documentation amendment changed a frozen release section"
        )
    base_sources = base_suite["repository"]["source_sha256"]
    amended_sources = suite["repository"]["source_sha256"]
    changed_sources = {
        logical
        for logical in set(base_sources) | set(amended_sources)
        if base_sources.get(logical) != amended_sources.get(logical)
    }
    if changed_sources != SUITE_V2_DOCUMENT_AMENDMENT_SOURCE_OVERRIDES:
        raise RuntimeError(
            "Suite v2 documentation amendment source boundary drifted"
        )

    scoreboard_path = Path(identity_audits["scoreboard"]["path"])
    scoreboard = _read_json_object(scoreboard_path)
    rows = scoreboard.get("component_results")
    cube_rows = (
        [row for row in rows if row.get("component_id") == "cube_gripper_carry"]
        if isinstance(rows, list)
        else []
    )
    if (
        len(rows or []) != 11
        or len(cube_rows) != 1
        or not str(cube_rows[0].get("method_name", "")).startswith("LeWM")
        or "pldm" in str(cube_rows[0].get("method_name", "")).lower()
    ):
        raise RuntimeError(
            "Suite v2 documentation amendment changed the formal Cube scoreboard"
        )

    document_path = Path(identity_audits["amended_public_document"]["path"])
    document = document_path.read_text(encoding="utf-8")
    required_document_fragments = (
        "| Cube 夹爪携带规则 | LeWM | 原始 checkpoint |",
        "| Cube 夹爪携带规则 | LeWM | 固定图像编码器，拟合配对真实未来 |",
        "| Cube 夹爪携带规则 | PLDM | 使用相同合成数据",
        "### 5.1 Cube 多开源模型对比（Public v1 待补齐）",
        "### 5.2 Cube Public recovery 边界",
        "机器可读正式 scoreboard 仍保持 11 行",
    )
    if (
        any(fragment not in document for fragment in required_document_fragments)
        or document.count("| External-0") != 3
    ):
        raise RuntimeError(
            "Suite v2 documentation amendment reference table is incomplete"
        )

    return {
        "required": True,
        "active": True,
        "status": "suite_registration_passed_with_documentation_amendment_v1",
        "decision_path": str(decision_path),
        "decision_is_commit_marker": True,
        "base_membership": base_activation,
        "amendment_id": SUITE_V2_DOCUMENT_AMENDMENT_ID,
        "amendment_scope": "documentation_only_reference_table_expansion",
        "formal_scoreboard_mutated": False,
        "public_test_rerun": False,
        "partial_outputs_grant_membership": False,
        "evidence": identity_audits,
        "passed": True,
    }


def _require_suite_membership_activation(
    suite: dict[str, Any],
    *,
    repo_root: Path | None = None,
    registration_capability: object | None = None,
) -> dict[str, Any]:
    """Implement the Suite v2 gate, including a private finalizer capability."""

    root = (repo_root or repository_root()).resolve()
    if suite.get("release_id") != SUITE_V2_RELEASE_ID:
        return {
            "required": False,
            "active": True,
            "status": "not_required_for_suite_v1",
            "passed": True,
        }

    authority = _validate_suite_v2_membership_authority(suite)
    if authority["decision_path"] == SUITE_V2_PRE_RECOVERY_DECISION:
        raise RuntimeError(
            "Suite v2 membership is not active: the pre-recovery candidate "
            "is permanently uncommitted"
        )
    if (
        authority.get("activation_condition")
        == SUITE_V2_DOCUMENT_AMENDMENT_ACTIVATION
    ):
        if registration_capability is not None:
            raise RuntimeError(
                "Documentation amendment does not expose registration-audit bypass"
            )
        return _require_suite_documentation_amendment_activation(
            suite, repo_root=root
        )
    if registration_capability is not None and (
        registration_capability is not _SUITE_V2_REGISTRATION_AUDIT_CAPABILITY
    ):
        raise RuntimeError("Invalid Suite v2 registration-audit capability")
    if registration_capability is _SUITE_V2_REGISTRATION_AUDIT_CAPABILITY:
        return {
            "required": True,
            "active": False,
            "status": "pending_registration_internal_audit",
            "decision_path": authority["decision_path"],
            "partial_outputs_grant_membership": False,
            "passed": True,
        }

    decision_path = resolve_no_symlink_contextworld_path(
        authority["decision_path"],
        repo_root=root,
        label="Suite v2 canonical registration decision",
    )
    if not decision_path.is_file():
        raise RuntimeError(
            "Suite v2 membership is not active: canonical registration "
            "decision is missing"
        )
    decision = _read_json_object(decision_path)
    claims = decision.get("claims")
    gates = decision.get("gate_summary")
    evidence = decision.get("evidence")
    if (
        decision.get("schema_version") != 1
        or decision.get("registration_id") != CUBE_SUITE_REGISTRATION_ID
        or decision.get("suite_release_id") != SUITE_V2_RELEASE_ID
        or decision.get("release_id")
        != suite["components"]["cube_gripper_carry"]["release_id"]
        or decision.get("status") != "suite_registration_passed"
        or decision.get("passed") is not True
        or not isinstance(claims, dict)
        or claims.get("suite_membership") != SUITE_V2_RELEASE_ID
        or claims.get("suite_membership_granted") is not True
        or claims.get("registration_decision_is_commit_marker") is not True
        or claims.get("partial_outputs_grant_membership") is not False
        or claims.get("registration_recovery") != "direct_reservation_v2"
        or claims.get("recovery_of_registration")
        != "contextworld_cube_gripper_carry_h3_v4r1_suite_registration_v1"
        or claims.get("prior_failed_staging_reused") is not False
        or claims.get("prior_failed_namespace_mutated") is not False
        or claims.get("directory_rename_used") is not False
        or claims.get("direct_target_exclusively_reserved") is not True
        or claims.get("fresh_copy_manifest_verified") is not True
        or not isinstance(gates, dict)
        or gates.get("component_full_audit") is not True
        or gates.get("suite_v2_full_audit") is not True
        or gates.get("portable_copy_export") is not True
        or gates.get("exported_bundle_full_reaudit") is not True
        or gates.get("components") != 9
        or gates.get("formal_scoreboard_rows") != 11
        or gates.get("formal_scoreboard_components") != 7
        or gates.get("suite_v1_components") != 8
        or gates.get("suite_v1_formal_scoreboard_rows") != 10
        or gates.get("suite_v1_cube_absent") is not True
        or gates.get("direct_target_exclusively_reserved") is not True
        or gates.get("fresh_copy_tree_identity_verified") is not True
        or gates.get("directory_rename_used") is not False
        or not isinstance(evidence, dict)
        or evidence.get("formal_reference_rows") != 11
        or evidence.get("formal_reference_components") != 7
        or evidence.get("cube_rows") != 1
        or evidence.get("cube_family") != "lewm"
        or evidence.get("external_results_included") is not False
        or evidence.get("passed") is not True
    ):
        raise RuntimeError(
            "Suite v2 membership is not active: registration decision failed "
            "its exact contract"
        )

    expected_evidence_paths = {
        **SUITE_V2_REQUIRED_EVIDENCE_PATHS,
        "specification": suite["public_results"]["specification"]["path"],
        "scoreboard": suite["public_results"]["scoreboard"]["path"],
    }
    evidence_audits = {
        name: _audit_registered_identity(
            evidence.get(name),
            expected_path=logical_path,
            repo_root=root,
        )
        for name, logical_path in expected_evidence_paths.items()
    }
    suite_path = Path(suite["_config_path"])
    suite_identity = evidence["suite_v2_config"]
    if (
        _sha256(suite_path) != suite_identity["sha256"]
        or suite_path.stat().st_size != suite_identity["size_bytes"]
    ):
        raise RuntimeError(
            "Suite v2 membership is not active: loaded suite config is not "
            "the registered config"
        )
    for name in ("component_audit", "suite_audit", "export_audit"):
        payload = _read_json_object(Path(evidence_audits[name]["path"]))
        if payload.get("passed") is not True:
            raise RuntimeError(
                f"Suite v2 decision references a failed audit: {name}"
            )
    export_commit = _validate_recovery_v2_export_commit(
        evidence,
        evidence_audits,
        repo_root=root,
    )
    return {
        "required": True,
        "active": True,
        "status": "suite_registration_passed",
        "decision_path": str(decision_path),
        "decision_is_commit_marker": True,
        "partial_outputs_grant_membership": False,
        "evidence": evidence_audits,
        "export_commit": export_commit,
        "passed": True,
    }


def require_suite_membership_activation(
    suite: dict[str, Any],
    *,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Require the canonical Suite v2 decision and all registered evidence."""

    return _require_suite_membership_activation(suite, repo_root=repo_root)


def load_icl_suite_release(
    path: Path | str = DEFAULT_SUITE_RELEASE_CONFIG,
) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "documentation_amendment" in payload:
        raw = payload
        expected_overlay_keys = {
            "schema_version",
            "release_id",
            "release_status",
            "candidate_date",
            "documentation_amendment",
            "membership_authority",
            "repository_overrides",
        }
        amendment = raw.get("documentation_amendment")
        base_identity = (
            amendment.get("base_release_config")
            if isinstance(amendment, dict)
            else None
        )
        expected_amendment_keys = {
            "amendment_id",
            "status",
            "scope",
            "base_release_config",
            "base_registration_decision",
            "base_export_public_document",
            "prior_public_document",
            "allowed_changes",
            "prohibited_changes",
        }
        if (
            set(raw) != expected_overlay_keys
            or raw.get("schema_version") != 1
            or raw.get("release_id") != SUITE_V2_RELEASE_ID
            or not isinstance(amendment, dict)
            or set(amendment) != expected_amendment_keys
            or amendment.get("amendment_id")
            != SUITE_V2_DOCUMENT_AMENDMENT_ID
            or amendment.get("status") != "frozen_documentation_only"
            or amendment.get("scope")
            != "reference_table_model_comparison_and_empty_external_slots"
            or amendment.get("allowed_changes")
            != [
                "public_document_reference_results_section",
                "suite_activation_and_audit_plumbing",
                "historical_recovery_finalizer_default_pin",
            ]
            or amendment.get("prohibited_changes")
            != [
                "component_release_or_data",
                "formal_scoreboard_or_specification",
                "training_or_checkpoint_selection",
                "public_test_access_or_rerun",
                "threshold_or_recipe",
            ]
            or not isinstance(base_identity, dict)
            or base_identity.get("path") != SUITE_V2_CONFIG_LOGICAL_PATH
            or base_identity.get("sha256") != SUITE_V2_BASE_CONFIG_SHA256
            or base_identity.get("size_bytes") != 14191
            or amendment.get("base_registration_decision")
            != {
                "path": SUITE_V2_REGISTRATION_DECISION,
                "sha256": SUITE_V2_BASE_DECISION_SHA256,
                "size_bytes": 8701,
            }
            or amendment.get("base_export_public_document")
            != {
                "path": SUITE_V2_BASE_EXPORT_PUBLIC_DOCUMENT,
                "sha256": SUITE_V2_BASE_PUBLIC_DOCUMENT_SHA256,
                "size_bytes": 31181,
            }
            or amendment.get("prior_public_document")
            != {
                "path": "docs/ContextWorld_ICL_Benchmark.md",
                "sha256": SUITE_V2_BASE_PUBLIC_DOCUMENT_SHA256,
                "size_bytes": 31181,
            }
        ):
            raise ValueError(
                "Suite v2 documentation amendment config is not exact"
            )
        root = repository_root()
        base_path = resolve_no_symlink_contextworld_path(
            SUITE_V2_CONFIG_LOGICAL_PATH,
            repo_root=root,
            label="Suite v2 documentation amendment base config",
        )
        if (
            _sha256(base_path) != base_identity["sha256"]
            or base_path.stat().st_size != base_identity["size_bytes"]
        ):
            raise ValueError(
                "Suite v2 documentation amendment base config drifted"
            )
        base_payload = yaml.safe_load(base_path.read_text(encoding="utf-8"))
        if not isinstance(base_payload, dict):
            raise ValueError("Suite v2 documentation amendment base is invalid")
        overrides = raw.get("repository_overrides")
        if (
            not isinstance(overrides, dict)
            or set(overrides) != {"source_sha256", "public_document"}
            or not isinstance(overrides.get("source_sha256"), dict)
            or set(overrides["source_sha256"])
            != SUITE_V2_DOCUMENT_AMENDMENT_SOURCE_OVERRIDES
            or any(
                len(str(value)) != 64
                for value in overrides["source_sha256"].values()
            )
            or overrides.get("public_document", {}).get("path")
            != "docs/ContextWorld_ICL_Benchmark.md"
            or len(
                str(overrides.get("public_document", {}).get("sha256", ""))
            )
            != 64
        ):
            raise ValueError(
                "Suite v2 documentation amendment repository overrides are invalid"
            )
        payload = deepcopy(base_payload)
        payload["release_status"] = raw["release_status"]
        payload["candidate_date"] = raw["candidate_date"]
        payload["membership_authority"] = deepcopy(
            raw["membership_authority"]
        )
        payload["repository"]["source_sha256"].update(
            overrides["source_sha256"]
        )
        payload["repository"]["public_document"] = deepcopy(
            overrides["public_document"]
        )
        payload["documentation_amendment"] = deepcopy(amendment)
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError(f"Unsupported ICL suite config: {config_path}")
    if payload.get("release_id") not in SUPPORTED_SUITE_RELEASE_IDS:
        raise ValueError(f"Unexpected ICL suite release id: {config_path}")
    if payload.get("release_status") not in {
        "validation_release_candidate",
        "validation_release",
        "public_test_release_candidate",
        "public_test_release",
    }:
        raise ValueError("Unsupported ICL suite release status")
    scope = payload.get("scope", {})
    if scope.get("public_test_included") is not True:
        raise ValueError("The public ICL suite must include Public Test")
    if scope.get("sealed_test_included") is not False:
        raise ValueError("The public ICL suite must not contain sealed Test")
    model_interface = payload.get("model_interface", {})
    if (
        model_interface.get("primary_model_type") != "latent_world_model"
        or model_interface.get("decoder_required") is not False
        or model_interface.get(
            "raw_latent_loss_cross_model_comparison_allowed"
        )
        is not False
    ):
        raise ValueError(
            "The Suite must use the decoder-free latent-world-model contract"
        )
    components = payload.get("components")
    if not isinstance(components, dict):
        raise ValueError("ICL suite components must be a mapping")
    expected_components = (
        SUITE_V2_COMPONENT_IDS
        if payload["release_id"] == SUITE_V2_RELEASE_ID
        else COMPONENT_IDS
    )
    expected_result_statuses = (
        SUITE_V2_REFERENCE_RESULT_STATUSES
        if payload["release_id"] == SUITE_V2_RELEASE_ID
        else REFERENCE_RESULT_STATUSES
    )
    if tuple(components) != expected_components:
        raise ValueError(
            f"ICL suite components must be ordered as {expected_components}"
        )
    for component_id, component in components.items():
        if component.get("release_id") is None:
            raise ValueError(f"{component_id} is missing release_id")
        if component.get("benchmark_component_status") != "ready":
            raise ValueError(
                f"{component_id} benchmark component is not ready"
            )
        if (
            component.get("reference_result_status")
            != expected_result_statuses[component_id]
        ):
            raise ValueError(
                f"{component_id} reference result status is not frozen"
            )
        expected = str(component.get("release_config_sha256", ""))
        if len(expected) != 64:
            raise ValueError(
                f"{component_id} release hash has not been frozen"
            )
    for logical_path, expected in payload["repository"][
        "source_sha256"
    ].items():
        if len(str(expected)) != 64:
            raise ValueError(f"Source hash is not frozen: {logical_path}")
    document_hash = str(
        payload["repository"]["public_document"]["sha256"]
    )
    if len(document_hash) != 64:
        raise ValueError("Public document hash has not been frozen")
    public_results = payload.get("public_results")
    if not isinstance(public_results, dict):
        raise ValueError("Suite public_results must be registered")
    for key in ("specification", "scoreboard"):
        specification = public_results.get(key)
        if (
            not isinstance(specification, dict)
            or not str(specification.get("path", "")).startswith(
                "artifacts/"
            )
            or len(str(specification.get("sha256", ""))) != 64
        ):
            raise ValueError(f"Public result {key!r} is not frozen")
    if int(public_results.get("formal_reference_rows", 0)) <= 0:
        raise ValueError("Suite public result row count must be positive")
    formal_components = public_results.get(
        "components_with_formal_results"
    )
    if (
        not isinstance(formal_components, list)
        or not formal_components
        or not set(formal_components).issubset(expected_components)
    ):
        raise ValueError("Invalid components_with_formal_results")
    if payload["release_id"] == SUITE_V2_RELEASE_ID:
        _validate_suite_v2_membership_authority(payload)
    return {**payload, "_config_path": str(config_path)}


def _audit_file(
    logical_path: str,
    expected_sha256: str,
    *,
    repo_root: Path,
) -> dict[str, Any]:
    path = resolve_contextworld_path(logical_path, repo_root=repo_root)
    exists = path.is_file()
    observed = _sha256(path) if exists else None
    return {
        "logical_path": logical_path,
        "path": str(path),
        "exists": exists,
        "expected_sha256": expected_sha256,
        "observed_sha256": observed,
        "passed": bool(exists and observed == expected_sha256),
    }


def _bundled_component_release(
    suite: dict[str, Any],
    component_id: str,
    *,
    repo_root: Path,
) -> Path:
    suite_path = Path(suite["_config_path"])
    bundled = suite_path.parent / "releases" / f"{component_id}.yaml"
    if bundled.is_file():
        return bundled
    logical = suite["components"][component_id]["release_config"]
    return resolve_contextworld_path(logical, repo_root=repo_root)


def _bundled_readme(suite: dict[str, Any], *, repo_root: Path) -> Path:
    suite_path = Path(suite["_config_path"])
    candidate = suite_path.parent.parent / "README.md"
    if candidate.is_file():
        return candidate
    logical = suite["repository"]["public_document"]["path"]
    return resolve_contextworld_path(logical, repo_root=repo_root)


def _bundled_public_result(
    suite: dict[str, Any],
    key: str,
    *,
    repo_root: Path,
) -> Path:
    """Resolve a Suite-level result in both source and exported layouts."""

    specification = suite["public_results"][key]
    logical = Path(str(specification["path"]))
    suite_path = Path(suite["_config_path"])
    if logical.parts and logical.parts[0] == "artifacts":
        bundled = suite_path.parent.joinpath(*logical.parts[1:])
        if bundled.is_file():
            return bundled
    return resolve_contextworld_path(logical, repo_root=repo_root)


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _audit_public_results(
    suite: dict[str, Any],
    *,
    repo_root: Path,
) -> dict[str, Any]:
    """Verify the compact public table against its formal seed-level spec."""

    rows: dict[str, Any] = {}
    for key in ("specification", "scoreboard"):
        specification = suite["public_results"][key]
        path = _bundled_public_result(suite, key, repo_root=repo_root)
        exists = path.is_file()
        observed = _sha256(path) if exists else None
        rows[key] = {
            "path": str(path),
            "exists": exists,
            "expected_sha256": specification["sha256"],
            "observed_sha256": observed,
            "passed": bool(
                exists and observed == specification["sha256"]
            ),
        }

    reproduction: dict[str, Any]
    if all(row["passed"] for row in rows.values()):
        try:
            source = _read_json_object(
                Path(rows["specification"]["path"])
            )
            observed_scoreboard = _read_json_object(
                Path(rows["scoreboard"]["path"])
            )
            expected_scoreboard = make_public_scoreboard_from_spec(source)
            result_rows = observed_scoreboard.get("component_results", [])
            expected_components = set(
                suite["public_results"]["components_with_formal_results"]
            )
            observed_components = {
                row.get("component_id")
                for row in result_rows
                if isinstance(row, dict)
            }
            reproduction = {
                "scoreboard_exactly_reproduced": (
                    observed_scoreboard == expected_scoreboard
                ),
                "expected_reference_rows": suite["public_results"][
                    "formal_reference_rows"
                ],
                "observed_reference_rows": len(result_rows),
                "expected_components": sorted(expected_components),
                "observed_components": sorted(observed_components),
            }
            reproduction["passed"] = bool(
                reproduction["scoreboard_exactly_reproduced"]
                and reproduction["observed_reference_rows"]
                == reproduction["expected_reference_rows"]
                and observed_components == expected_components
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            reproduction = {"passed": False, "error": str(error)}
    else:
        reproduction = {
            "passed": False,
            "error": "public result file hash audit failed",
        }
    return {
        "files": rows,
        "reproduction": reproduction,
        "passed": bool(
            all(row["passed"] for row in rows.values())
            and reproduction["passed"]
        ),
    }


def load_public_scoreboard(
    release_config: Path | str = DEFAULT_SUITE_RELEASE_CONFIG,
    *,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Load the frozen compact result table and reject stale metadata."""

    root = (repo_root or repository_root()).resolve()
    suite = load_icl_suite_release(release_config)
    require_suite_membership_activation(
        suite,
        repo_root=root,
    )
    audit = _audit_public_results(suite, repo_root=root)
    if not audit["passed"]:
        raise RuntimeError("Public scoreboard audit failed")
    return _read_json_object(Path(audit["files"]["scoreboard"]["path"]))


def _audit_public_document_template(
    document_path: Path,
    suite: dict[str, Any],
) -> dict[str, Any]:
    template = suite["extension"]["public_document_template"]
    expected_subsections = list(template["subsections"])
    section_titles = template["component_sections"]
    lines = document_path.read_text(encoding="utf-8").splitlines()
    observed: dict[str, list[str]] = {}
    component_ids = tuple(suite["components"])
    for component_id in component_ids:
        section_heading = f"### {section_titles[component_id]}"
        try:
            start = lines.index(section_heading) + 1
        except ValueError:
            observed[component_id] = []
            continue
        end = next(
            (
                index
                for index in range(start, len(lines))
                if lines[index].startswith("### ")
            ),
            len(lines),
        )
        observed[component_id] = [
            line.removeprefix("#### ")
            for line in lines[start:end]
            if line.startswith("#### ")
        ]
    return {
        "expected_subsections": expected_subsections,
        "observed_subsections": observed,
        "passed": all(
            observed[component_id] == expected_subsections
            for component_id in component_ids
        ),
    }


def _assert_frozen_export_inputs(
    suite: dict[str, Any],
    *,
    repo_root: Path,
) -> None:
    """Reject a bundle when the frozen public entry points are stale.

    Component content audits remain available through ``audit``.  Export has
    a smaller fail-closed gate of its own so it cannot silently copy a changed
    document, source file, or component release YAML under an old Suite hash.
    """

    failures: list[str] = []
    repository = suite.get("repository", {})
    for logical, expected in repository.get("source_sha256", {}).items():
        result = _audit_file(logical, str(expected), repo_root=repo_root)
        if not result["passed"]:
            failures.append(
                f"source {logical}: {result['observed_sha256']} != {expected}"
            )

    document = repository.get("public_document")
    if isinstance(document, dict) and document.get("sha256"):
        path = _bundled_readme(suite, repo_root=repo_root)
        observed = _sha256(path) if path.is_file() else None
        if observed != document["sha256"]:
            failures.append(
                f"public document {path}: {observed} != {document['sha256']}"
            )
        elif not _audit_public_document_template(path, suite)["passed"]:
            failures.append("public document component template is invalid")

    for component_id, component in suite.get("components", {}).items():
        expected = component.get("release_config_sha256")
        if not expected:
            continue
        path = _bundled_component_release(
            suite,
            component_id,
            repo_root=repo_root,
        )
        observed = _sha256(path) if path.is_file() else None
        if observed != expected:
            failures.append(
                f"component {component_id}: {observed} != {expected}"
            )

    if "public_results" in suite:
        public_results = _audit_public_results(suite, repo_root=repo_root)
        if not public_results["passed"]:
            failures.append("public scoreboard is stale or not reproducible")

    if failures:
        raise RuntimeError(
            "Suite export inputs are not frozen:\n- " + "\n- ".join(failures)
        )


_PORTABLE_TEXT_SUFFIXES = {
    ".csv",
    ".json",
    ".jsonl",
    ".log",
    ".md",
    ".txt",
    ".yaml",
    ".yml",
}
_NON_PORTABLE_MARKERS = (
    "/opt/",
    "/tmp/",
    "/home/",
    "/root/",
    "../../data/",
    "\\Users\\",
)


def _artifact_text_files(path: Path, *, kind: str) -> Iterable[Path]:
    candidates = (path,) if kind == "file" else path.rglob("*")
    for candidate in candidates:
        if (
            candidate.is_file()
            and candidate.suffix.lower() in _PORTABLE_TEXT_SUFFIXES
        ):
            yield candidate


def _assert_portable_export_entries(
    entries: Iterable[tuple[str, str]],
    *,
    repo_root: Path,
) -> None:
    """Reject machine-specific paths in files copied into the public bundle."""

    violations: list[str] = []
    seen: set[str] = set()
    for logical_path, kind in entries:
        if logical_path in seen:
            continue
        seen.add(logical_path)
        source = resolve_contextworld_path(logical_path, repo_root=repo_root)
        if kind == "file" and not source.is_file():
            raise FileNotFoundError(source)
        if kind == "directory" and not source.is_dir():
            raise FileNotFoundError(source)
        for path in _artifact_text_files(source, kind=kind):
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            markers = [
                marker for marker in _NON_PORTABLE_MARKERS if marker in text
            ]
            if markers:
                violations.append(
                    f"{logical_path}:{path.relative_to(source) if kind == 'directory' else path.name} "
                    f"contains {', '.join(markers)}"
                )
    if violations:
        raise RuntimeError(
            "Suite export contains machine-specific paths:\n- "
            + "\n- ".join(violations)
        )


def _assert_portable_source_files(
    files: Iterable[tuple[str, Path]],
) -> None:
    """Apply the public-path policy to files copied outside artifact entries."""

    violations: list[str] = []
    for label, path in files:
        if not path.is_file():
            raise FileNotFoundError(path)
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        markers = [
            marker for marker in _NON_PORTABLE_MARKERS if marker in content
        ]
        if markers:
            violations.append(f"{label} contains {', '.join(markers)}")
    if violations:
        raise RuntimeError(
            "Suite export contains machine-specific paths:\n- "
            + "\n- ".join(violations)
        )


def _enforce_component_causal_gate(
    component_audit: dict[str, Any],
) -> dict[str, bool]:
    """Require every Suite component to publish the same causal-data proof."""

    causal_contract = component_audit.get("causal_data_contract")
    causal_gate = {
        "present": isinstance(causal_contract, dict),
        "passed": bool(
            isinstance(causal_contract, dict)
            and causal_contract.get("passed") is True
        ),
    }
    component_audit["suite_causal_data_gate"] = causal_gate
    if not causal_gate["passed"]:
        component_audit["passed"] = False
        component_audit["status"] = "failed"
        component_audit["reason"] = (
            "component is missing a passed causal data contract"
        )
    return causal_gate


def _audit_icl_suite_release_impl(
    *,
    release_config: Path | str = DEFAULT_SUITE_RELEASE_CONFIG,
    repo_root: Path | None = None,
    components: Iterable[str] | None = None,
    full: bool = False,
    original_h5: Path | str | None = None,
    registration_capability: object | None = None,
) -> dict[str, Any]:
    root = (repo_root or repository_root()).resolve()
    suite = load_icl_suite_release(release_config)
    membership_activation = _require_suite_membership_activation(
        suite,
        repo_root=root,
        registration_capability=registration_capability,
    )
    suite_component_ids = tuple(suite["components"])
    selected = tuple(suite_component_ids if components is None else components)
    if not selected or not set(selected).issubset(suite_component_ids):
        raise ValueError(
            f"components must be a non-empty subset of {suite_component_ids}"
        )
    if len(selected) != len(set(selected)):
        raise ValueError("components must be unique")

    code_audits = [
        _audit_file(logical, expected, repo_root=root)
        for logical, expected in suite["repository"][
            "source_sha256"
        ].items()
    ]
    document_path = _bundled_readme(suite, repo_root=root)
    expected_document_hash = suite["repository"]["public_document"]["sha256"]
    document_audit = {
        "path": str(document_path),
        "exists": document_path.is_file(),
        "expected_sha256": expected_document_hash,
        "observed_sha256": (
            _sha256(document_path) if document_path.is_file() else None
        ),
    }
    document_audit["component_template"] = (
        _audit_public_document_template(document_path, suite)
        if document_path.is_file()
        else {"passed": False}
    )
    document_audit["passed"] = bool(
        document_audit["exists"]
        and document_audit["observed_sha256"] == expected_document_hash
        and document_audit["component_template"]["passed"]
    )
    public_results_audit = _audit_public_results(suite, repo_root=root)

    component_audits: dict[str, Any] = {}
    component_release_audits: dict[str, Any] = {}
    for component_id in selected:
        component = suite["components"][component_id]
        component_path = _bundled_component_release(
            suite,
            component_id,
            repo_root=root,
        )
        release_exists = component_path.is_file()
        observed_hash = _sha256(component_path) if release_exists else None
        release_audit = {
            "path": str(component_path),
            "exists": release_exists,
            "expected_sha256": component["release_config_sha256"],
            "observed_sha256": observed_hash,
            "passed": bool(
                release_exists
                and observed_hash == component["release_config_sha256"]
            ),
        }
        component_release_audits[component_id] = release_audit
        if not release_audit["passed"]:
            component_audits[component_id] = {
                "status": "failed",
                "passed": False,
                "reason": "component release config audit failed",
            }
            continue
        if component_id == "speed":
            speed_release = load_speed_icl_release(component_path)
            resolved_original = original_h5
            if resolved_original is None:
                suite_path = Path(suite["_config_path"])
                bundled_original = (
                    suite_path.parent
                    / "upstream/lewm-tworooms/tworoom.h5"
                )
                if bundled_original.is_file():
                    resolved_original = bundled_original
            component_audits[component_id] = audit_speed_icl_release(
                release_config=component_path,
                repo_root=root,
                original_h5=resolved_original,
                verify_all_eval_payloads=full,
            )
            if (
                component_audits[component_id]["release_id"]
                != speed_release["release_id"]
            ):
                component_audits[component_id]["passed"] = False
        elif component_id == "door":
            door_release = load_door_icl_release(component_path)
            component_audits[component_id] = audit_door_icl_release(
                release_config=component_path,
                repo_root=root,
                full=full,
            )
            if (
                component_audits[component_id]["release_id"]
                != door_release["release_id"]
            ):
                component_audits[component_id]["passed"] = False
        elif component_id == "action_delay":
            action_delay_release = load_action_delay_icl_release(
                component_path
            )
            component_audits[
                component_id
            ] = audit_action_delay_icl_release(
                release_config=component_path,
                repo_root=root,
                full=full,
            )
            if (
                component_audits[component_id]["release_id"]
                != action_delay_release["release_id"]
            ):
                component_audits[component_id]["passed"] = False
        elif component_id == "action_strength":
            action_strength_release = load_action_strength_icl_release(
                component_path
            )
            component_audits[
                component_id
            ] = audit_action_strength_icl_release(
                release_config=component_path,
                repo_root=root,
                full=full,
            )
            if (
                component_audits[component_id]["release_id"]
                != action_strength_release["release_id"]
            ):
                component_audits[component_id]["passed"] = False
        elif component_id == "contact_friction":
            contact_friction_release = load_contact_friction_icl_release(
                component_path
            )
            component_audits[
                component_id
            ] = audit_contact_friction_icl_release(
                release_config=component_path,
                repo_root=root,
                full=full,
            )
            if (
                component_audits[component_id]["release_id"]
                != contact_friction_release["release_id"]
            ):
                component_audits[component_id]["passed"] = False
        elif component_id == "motion_damping":
            motion_damping_release = load_motion_damping_icl_release(
                component_path
            )
            component_audits[
                component_id
            ] = audit_motion_damping_icl_release(
                release_config=component_path,
                repo_root=root,
                full=full,
            )
            if (
                component_audits[component_id]["release_id"]
                != motion_damping_release["release_id"]
            ):
                component_audits[component_id]["passed"] = False
        elif component_id == "robot_arm_mass":
            robot_arm_mass_release = load_reacher_arm_mass_icl_release(
                component_path
            )
            component_audits[
                component_id
            ] = audit_reacher_arm_mass_icl_release(
                release_config=component_path,
                repo_root=root,
                full=full,
            )
            if (
                component_audits[component_id]["release_id"]
                != robot_arm_mass_release["release_id"]
            ):
                component_audits[component_id]["passed"] = False
        elif component_id == "portal_exit":
            portal_exit_release = load_portal_exit_icl_release(
                component_path
            )
            component_audits[component_id] = audit_portal_exit_icl_release(
                release_config=component_path,
                repo_root=root,
                full=full,
            )
            if (
                component_audits[component_id]["release_id"]
                != portal_exit_release["release_id"]
            ):
                component_audits[component_id]["passed"] = False
        elif component_id == "cube_gripper_carry":
            cube_release = load_cube_grasp_rule_v4r1_icl_release(component_path)
            component_audits[component_id] = (
                audit_cube_grasp_rule_v4r1_icl_release(
                    release_config=component_path,
                    repo_root=root,
                    full=full,
                    layout=(
                        "bundle"
                        if component_path.parent.name == "releases"
                        else "source"
                    ),
                )
            )
            if (
                component_audits[component_id]["release_id"]
                != cube_release["release_id"]
            ):
                component_audits[component_id]["passed"] = False
        else:  # pragma: no cover - selected ids are validated above
            raise AssertionError(f"Unhandled benchmark component: {component_id}")

        _enforce_component_causal_gate(component_audits[component_id])

    technical_passed = bool(
        all(row["passed"] for row in code_audits)
        and document_audit["passed"]
        and public_results_audit["passed"]
        and all(
            row["passed"] for row in component_release_audits.values()
        )
        and all(row.get("passed") is True for row in component_audits.values())
    )
    distribution = suite["distribution"]
    public_ready = bool(
        technical_passed
        and distribution["code_license_status"] == "declared"
        and distribution["generated_data_license_status"] == "declared"
        and distribution["public_download_status"] == "configured"
    )
    return {
        "schema_version": 1,
        "release_id": suite["release_id"],
        "status": "passed" if technical_passed else "failed",
        "release_config": suite["_config_path"],
        "membership_activation": membership_activation,
        "artifact_root_override": os.environ.get(
            "CONTEXTWORLD_ARTIFACT_ROOT"
        ),
        "selected_components": list(selected),
        "full_content_hash_audit": full,
        "public_test_included": True,
        "sealed_test_included": False,
        "repository_code": code_audits,
        "public_document": document_audit,
        "public_results": public_results_audit,
        "component_release_configs": component_release_audits,
        "components": component_audits,
        "causal_data_contract_required_for_every_component": True,
        "technical_release_candidate_passed": technical_passed,
        "public_distribution_ready": public_ready,
        "distribution_blockers": [
            label
            for label, passed in (
                (
                    "ContextWorld source license declaration",
                    distribution["code_license_status"] == "declared",
                ),
                (
                    "ContextWorld-generated data license declaration",
                    distribution["generated_data_license_status"]
                    == "declared",
                ),
                (
                    "public artifact download URL",
                    distribution["public_download_status"] == "configured",
                ),
            )
            if not passed
        ],
        "passed": technical_passed,
    }


def audit_icl_suite_release(
    *,
    release_config: Path | str = DEFAULT_SUITE_RELEASE_CONFIG,
    repo_root: Path | None = None,
    components: Iterable[str] | None = None,
    full: bool = False,
    original_h5: Path | str | None = None,
) -> dict[str, Any]:
    return _audit_icl_suite_release_impl(
        release_config=release_config,
        repo_root=repo_root,
        components=components,
        full=full,
        original_h5=original_h5,
    )


def _audit_icl_suite_release_for_registration(
    *,
    release_config: Path | str = DEFAULT_SUITE_V2_RELEASE_CONFIG,
    repo_root: Path | None = None,
    components: Iterable[str] | None = None,
    full: bool = False,
    original_h5: Path | str | None = None,
) -> dict[str, Any]:
    """Run only the preregistered pre-decision finalizer audit."""

    return _audit_icl_suite_release_impl(
        release_config=release_config,
        repo_root=repo_root,
        components=components,
        full=full,
        original_h5=original_h5,
        registration_capability=_SUITE_V2_REGISTRATION_AUDIT_CAPABILITY,
    )


def _speed_export_entries(release: dict[str, Any]) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    for row in release["training"]["synthetic"].values():
        entries.append((row["data_root"], "directory"))
        for key in ("catalog", "manifest", "report"):
            entries.append((row[key], "file"))
    entries.append((release["evaluation"]["normalizer"], "file"))
    causal_audit = release["evaluation"].get("causal_data_audit")
    if causal_audit:
        entries.append((str(causal_audit), "file"))
    entries.extend(
        [
            (
                "artifacts/evaluation/history3/"
                "speed_multistep_extrap_v5/catalogs",
                "directory",
            ),
            (
                "artifacts/evaluation/history3/"
                "speed_multistep_extrap_v5/payloads",
                "directory",
            ),
        ]
    )
    if release.get("planning"):
        entries.extend(
            [
                (
                    "artifacts/evaluation/history3/"
                    "speed_isolated_v2/catalogs",
                    "directory",
                ),
                (
                    "artifacts/evaluation/history3/"
                    "speed_isolated_v2/payloads",
                    "directory",
                ),
            ]
        )
    entries.extend(
        (specification["path"], "file")
        for specification in release.get("reference_results", {}).values()
        if isinstance(specification, dict)
        and "path" in specification
        and "sha256" in specification
        and str(specification["path"]).startswith("artifacts/")
    )
    return entries


def _door_export_entries(release: dict[str, Any]) -> list[tuple[str, str]]:
    return door_icl_export_entries(release)


def _action_delay_export_entries(
    release: dict[str, Any],
) -> list[tuple[str, str]]:
    stages = release["training"].get("stages")
    if isinstance(stages, dict) and stages:
        training_roots = [
            (stage["artifact_tree"]["root"], "directory")
            for stage in stages.values()
        ]
    else:
        training_roots = [
            (release["training"]["artifact_tree"]["root"], "directory")
        ]
    evaluation_root = str(
        release["evaluation"]["artifact_tree"]["root"]
    ).rstrip("/")
    entries = training_roots + [
        (evaluation_root, "directory"),
        (release["evaluation"]["normalizer"], "file"),
    ]
    initialization = release["training"].get("initialization")
    if isinstance(initialization, dict):
        entries.extend(
            [
                (initialization["checkpoint"], "file"),
                (initialization["checkpoint_config"], "file"),
            ]
        )
    entries.extend(
        _external_artifact_files(
            {
                "evaluation_artifacts": release["evaluation"].get(
                    "artifacts", {}
                ),
                "reference_results": release.get("reference_results", {}),
            },
            bundled_roots=(
                *(str(path).rstrip("/") for path, _ in training_roots),
                evaluation_root,
            ),
        )
    )
    return entries


def _action_strength_export_entries(
    release: dict[str, Any],
) -> list[tuple[str, str]]:
    training_root = str(
        release["training"]["artifact_tree"]["root"]
    ).rstrip("/")
    evaluation_root = str(
        release["evaluation"]["artifact_tree"]["root"]
    ).rstrip("/")
    entries = [
        (training_root, "directory"),
        (evaluation_root, "directory"),
    ]
    bundled_roots = [training_root, evaluation_root]
    reference_method = release.get("reference_method", {})
    reference_tree = reference_method.get("artifact_tree", {})
    reference_root = reference_tree.get("root")
    if isinstance(reference_root, str):
        reference_root = reference_root.rstrip("/")
        entries.append((reference_root, "directory"))
        bundled_roots.append(reference_root)
    for section, key in (
        (release.get("training", {}), "contrast_scales"),
        (release.get("evaluation", {}), "planning_oracle"),
    ):
        specification = section.get(key)
        if not isinstance(specification, dict):
            continue
        path = specification.get("path")
        if isinstance(path, str) and not any(
            path == root or path.startswith(root + "/")
            for root in bundled_roots
        ):
            entries.append((path, "file"))
    for _root_path, artifacts in (
        (training_root, release["training"].get("artifacts", {})),
        (evaluation_root, release["evaluation"].get("artifacts", {})),
    ):
        entries.extend(
            (specification["path"], "file")
            for specification in artifacts.values()
            if isinstance(specification, dict)
            and str(specification.get("path", "")).startswith("artifacts/")
            and not any(
                str(specification["path"]) == bundled_root
                or str(specification["path"]).startswith(
                    bundled_root + "/"
                )
                for bundled_root in bundled_roots
            )
        )
    entries.extend(
        _external_artifact_files(
            release.get("reference_results", {}),
            bundled_roots=bundled_roots,
        )
    )
    return entries


def _external_artifact_files(
    value: Any,
    *,
    bundled_roots: Iterable[str] = (),
) -> list[tuple[str, str]]:
    """Collect hashed artifact files that sit outside bundled data trees.

    A component may organize receipts differently as its reference method
    evolves.  The Suite exporter therefore follows the stable public contract
    (an ``artifacts/...`` path plus its SHA-256) instead of depending on a
    method-specific diagnostic hierarchy.
    """

    normalized_roots = tuple(str(root).rstrip("/") for root in bundled_roots)
    entries: list[tuple[str, str]] = []

    def visit(item: Any) -> None:
        if isinstance(item, dict):
            path = item.get("path")
            digest = item.get("sha256")
            if (
                isinstance(path, str)
                and path.startswith("artifacts/")
                and isinstance(digest, str)
                and len(digest) == 64
                and not any(
                    path == root or path.startswith(root + "/")
                    for root in normalized_roots
                )
            ):
                entries.append((path, "file"))
            for child in item.values():
                visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)

    visit(value)
    return entries


def _contact_friction_export_entries(
    release: dict[str, Any],
) -> list[tuple[str, str]]:
    data_root = str(release["data"]["artifact_tree"]["root"]).rstrip("/")
    return [(data_root, "directory")] + _external_artifact_files(
        {
            "data_artifacts": release["data"].get("artifacts", {}),
            "reference_results": release.get("reference_results", {}),
        },
        bundled_roots=(data_root,),
    )


def _motion_damping_export_entries(
    release: dict[str, Any],
) -> list[tuple[str, str]]:
    data_root = str(release["data"]["artifact_tree"]["root"]).rstrip("/")
    return [(data_root, "directory")] + _external_artifact_files(
        {
            "data_artifacts": release["data"].get("artifacts", {}),
            "reference_results": release.get("reference_results", {}),
        },
        bundled_roots=(data_root,),
    )


def _robot_arm_mass_export_entries(
    release: dict[str, Any],
) -> list[tuple[str, str]]:
    data_root = str(release["data"]["artifact_tree"]["root"]).rstrip("/")
    entries = [(data_root, "directory")]
    entries.extend(
        (specification["path"], "file")
        for specification in release["data"].get("artifacts", {}).values()
        if isinstance(specification, dict)
        and "path" in specification
        and str(specification["path"]).startswith("artifacts/")
        and not str(specification["path"]).startswith(data_root + "/")
    )
    entries.extend(
        (specification["path"], "file")
        for specification in release.get("reference_results", {}).values()
        if isinstance(specification, dict)
        and "path" in specification
        and "sha256" in specification
        and str(specification["path"]).startswith("artifacts/")
    )
    return entries


def _portal_exit_export_entries(
    release: dict[str, Any],
) -> list[tuple[str, str]]:
    data_root = str(release["data"]["artifact_tree"]["root"]).rstrip("/")
    entries = [(data_root, "directory")]
    entries.extend(
        (specification["path"], "file")
        for specification in release["data"].get("artifacts", {}).values()
        if isinstance(specification, dict)
        and "path" in specification
        and str(specification["path"]).startswith("artifacts/")
        and not str(specification["path"]).startswith(data_root + "/")
    )
    entries.extend(
        (specification["path"], "file")
        for specification in release.get("reference_results", {}).values()
        if isinstance(specification, dict)
        and "path" in specification
        and "sha256" in specification
        and str(specification["path"]).startswith("artifacts/")
    )
    entries.extend(
        [
            (release["training"]["initialization"]["checkpoint"], "file"),
            (
                release["training"]["initialization"]["frozen_normalizer"],
                "file",
            ),
        ]
    )
    entries.extend(
        _external_artifact_files(
            release.get("scoring", {}).get(
                "original_task_retention", {}
            ),
            bundled_roots=(data_root,),
        )
    )
    return entries


def _cube_gripper_carry_export_entries(
    release: dict[str, Any],
) -> list[tuple[str, str]]:
    """Export only the audited portable projection, never raw recovery receipts."""

    return [(str(release["data"]["artifact_tree"]["root"]), "directory")]


def _artifact_target(benchmark_root: Path, logical_path: str) -> Path:
    relative = Path(logical_path)
    if not relative.parts or relative.parts[0] != "artifacts":
        raise ValueError(f"Expected artifacts/... path, got {logical_path}")
    return benchmark_root.joinpath(*relative.parts[1:])


def _deduplicate_export_entries(
    entries: Iterable[tuple[str, str]],
    *,
    repo_root: Path,
) -> list[tuple[str, str]]:
    """Collapse exact duplicates and files already covered by a data tree.

    A directory entry exports its complete subtree.  Listing one of its files
    again would either create the same target twice or, more seriously, hide
    a conflict between two different sources.  Descendants are skipped only
    when their resolved source is exactly the corresponding member of the
    exported directory; every other overlap fails closed.
    """

    unique: list[tuple[str, str]] = []
    kinds_by_path: dict[str, str] = {}
    for logical_path, kind in entries:
        if kind not in {"file", "directory"}:
            raise ValueError(f"Unsupported export entry kind: {kind!r}")
        path = Path(logical_path)
        if (
            path.is_absolute()
            or not path.parts
            or path.parts[0] != "artifacts"
            or ".." in path.parts
        ):
            raise ValueError(
                f"Export entry must be a normalized artifacts/... path: "
                f"{logical_path!r}"
            )
        normalized = path.as_posix()
        previous_kind = kinds_by_path.get(normalized)
        if previous_kind is not None:
            if previous_kind != kind:
                raise RuntimeError(
                    "Export target is registered as both a file and a "
                    f"directory: {normalized}"
                )
            continue
        kinds_by_path[normalized] = kind
        unique.append((normalized, kind))

    directory_paths = tuple(
        Path(logical_path)
        for logical_path, kind in unique
        if kind == "directory"
    )
    result: list[tuple[str, str]] = []
    for logical_path, kind in unique:
        path = Path(logical_path)
        ancestors = [
            directory
            for directory in directory_paths
            if directory != path and directory in path.parents
        ]
        if not ancestors:
            result.append((logical_path, kind))
            continue
        covering = min(ancestors, key=lambda value: len(value.parts))
        source = resolve_contextworld_path(
            logical_path,
            repo_root=repo_root,
        )
        covering_source = resolve_contextworld_path(
            covering.as_posix(),
            repo_root=repo_root,
        )
        expected_source = covering_source.joinpath(
            *path.relative_to(covering).parts
        )
        if source.resolve() != expected_source.resolve():
            raise RuntimeError(
                "Overlapping export entries resolve to different sources: "
                f"{covering.as_posix()} covers {logical_path}, but "
                f"{expected_source} != {source}"
            )
        if kind == "file" and not expected_source.is_file():
            raise RuntimeError(
                f"Covered export file is missing: {expected_source}"
            )
        if kind == "directory" and not expected_source.is_dir():
            raise RuntimeError(
                f"Covered export directory is missing: {expected_source}"
            )
    return result


def _exclusive_copy_file(source: Path, target: Path) -> None:
    metadata = os.lstat(source)
    if not stat.S_ISREG(metadata.st_mode) or source.is_symlink():
        raise RuntimeError(f"Exclusive export source is not a regular file: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.parent.is_symlink() or not target.parent.is_dir():
        raise RuntimeError(f"Exclusive export parent is unsafe: {target.parent}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(target, flags, metadata.st_mode & 0o777)
    try:
        with os.fdopen(descriptor, "wb") as destination:
            descriptor = -1
            with source.open("rb") as stream:
                shutil.copyfileobj(stream, destination, 8 * 1024 * 1024)
            destination.flush()
            os.fsync(destination.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if target.stat().st_size != metadata.st_size:
        raise RuntimeError(f"Exclusive export copy size mismatch: {target}")


def _exclusive_copy_tree(source: Path, target: Path) -> None:
    metadata = os.lstat(source)
    if not stat.S_ISDIR(metadata.st_mode) or source.is_symlink():
        raise RuntimeError(f"Exclusive export source is not a directory: {source}")
    target.mkdir()
    for child in sorted(source.rglob("*")):
        destination = target / child.relative_to(source)
        child_metadata = os.lstat(child)
        if stat.S_ISDIR(child_metadata.st_mode) and not child.is_symlink():
            destination.mkdir()
        elif stat.S_ISREG(child_metadata.st_mode) and not child.is_symlink():
            _exclusive_copy_file(child, destination)
        else:
            raise RuntimeError(
                f"Exclusive export source contains a symlink or special node: {child}"
            )


def _copy_export_file(
    source: Path, target: Path, *, exclusive: bool
) -> None:
    if exclusive:
        _exclusive_copy_file(source, target)
    else:
        shutil.copy2(source, target)


def _copy_export_tree(
    source: Path, target: Path, *, exclusive: bool
) -> None:
    if exclusive:
        _exclusive_copy_tree(source, target)
    else:
        shutil.copytree(source, target)


def _write_export_inventory(
    path: Path, payload: dict[str, Any], *, exclusive: bool
) -> None:
    content = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    if not exclusive:
        path.write_bytes(content)
        return
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _export_icl_suite_artifacts_impl(
    destination: Path | str,
    *,
    release_config: Path | str = DEFAULT_SUITE_RELEASE_CONFIG,
    repo_root: Path | None = None,
    mode: str = "copy",
    include_upstream_original: bool = True,
    registration_capability: object | None = None,
) -> dict[str, Any]:
    """Export one README plus one integrated benchmark data directory."""

    if mode not in {"copy", "symlink"}:
        raise ValueError("Export mode must be 'copy' or 'symlink'")
    exclusive_copy = (
        registration_capability
        is _SUITE_V2_REGISTRATION_AUDIT_CAPABILITY
    )
    if exclusive_copy and mode != "copy":
        raise ValueError("Registration recovery export must use copy mode")
    root = (repo_root or repository_root()).resolve()
    suite = load_icl_suite_release(release_config)
    membership_activation = _require_suite_membership_activation(
        suite,
        repo_root=root,
        registration_capability=registration_capability,
    )
    _assert_frozen_export_inputs(suite, repo_root=root)
    destination = Path(destination).expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Export destination is not empty: {destination}")
    if exclusive_copy and (
        not destination.is_dir() or destination.is_symlink()
    ):
        raise RuntimeError(
            "Registration recovery must exclusively reserve the export "
            "directory before copying"
        )

    speed_config = resolve_contextworld_path(
        suite["components"]["speed"]["release_config"],
        repo_root=root,
    )
    door_config = resolve_contextworld_path(
        suite["components"]["door"]["release_config"],
        repo_root=root,
    )
    action_delay_config = resolve_contextworld_path(
        suite["components"]["action_delay"]["release_config"],
        repo_root=root,
    )
    action_strength_config = resolve_contextworld_path(
        suite["components"]["action_strength"]["release_config"],
        repo_root=root,
    )
    contact_friction_config = resolve_contextworld_path(
        suite["components"]["contact_friction"]["release_config"],
        repo_root=root,
    )
    motion_damping_config = resolve_contextworld_path(
        suite["components"]["motion_damping"]["release_config"],
        repo_root=root,
    )
    robot_arm_mass_config = resolve_contextworld_path(
        suite["components"]["robot_arm_mass"]["release_config"],
        repo_root=root,
    )
    portal_exit_config = resolve_contextworld_path(
        suite["components"]["portal_exit"]["release_config"],
        repo_root=root,
    )
    cube_config = (
        resolve_contextworld_path(
            suite["components"]["cube_gripper_carry"]["release_config"],
            repo_root=root,
        )
        if "cube_gripper_carry" in suite["components"]
        else None
    )
    speed_release = load_speed_icl_release(speed_config)
    door_release = load_door_icl_release(door_config)
    action_delay_release = load_action_delay_icl_release(
        action_delay_config
    )
    action_strength_release = load_action_strength_icl_release(
        action_strength_config
    )
    contact_friction_release = load_contact_friction_icl_release(
        contact_friction_config
    )
    motion_damping_release = load_motion_damping_icl_release(
        motion_damping_config
    )
    robot_arm_mass_release = load_reacher_arm_mass_icl_release(
        robot_arm_mass_config
    )
    portal_exit_release = load_portal_exit_icl_release(portal_exit_config)
    cube_release = (
        load_cube_grasp_rule_v4r1_icl_release(cube_config)
        if cube_config is not None
        else None
    )

    reacher_checkpoint_configs = tuple(
        (
            f"robot_arm_mass.{family}_checkpoint_config",
            resolve_reacher_initial_checkpoint_config(
                robot_arm_mass_release,
                family,
                repo_root=root,
            ),
        )
        for family in ("lewm", "pldm")
    )
    _assert_portable_source_files(reacher_checkpoint_configs)

    entries = (
        _speed_export_entries(speed_release)
        + _door_export_entries(door_release)
        + _action_delay_export_entries(action_delay_release)
        + _action_strength_export_entries(action_strength_release)
        + _contact_friction_export_entries(contact_friction_release)
        + _motion_damping_export_entries(motion_damping_release)
        + _robot_arm_mass_export_entries(robot_arm_mass_release)
        + _portal_exit_export_entries(portal_exit_release)
        + (
            _cube_gripper_carry_export_entries(cube_release)
            if cube_release is not None
            else []
        )
        + [
            (suite["public_results"][key]["path"], "file")
            for key in ("specification", "scoreboard")
            if "public_results" in suite
        ]
    )
    entries = _deduplicate_export_entries(entries, repo_root=root)
    _assert_portable_export_entries(entries, repo_root=root)

    if not exclusive_copy:
        destination.mkdir(parents=True, exist_ok=True)
    benchmark_root = destination / "benchmark"
    benchmark_root.mkdir()
    seen: set[str] = set()
    inventory_entries: list[dict[str, Any]] = []
    for logical_path, kind in entries:
        if logical_path in seen:
            continue
        seen.add(logical_path)
        source = resolve_contextworld_path(logical_path, repo_root=root)
        target = _artifact_target(benchmark_root, logical_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if kind == "file":
            if not source.is_file():
                raise FileNotFoundError(source)
            if mode == "symlink":
                target.symlink_to(source)
            else:
                _copy_export_file(source, target, exclusive=exclusive_copy)
            inventory_entries.append(
                {
                    "logical_path": logical_path,
                    "kind": kind,
                    "bytes": source.stat().st_size,
                    "sha256": _sha256(source),
                }
            )
        else:
            if not source.is_dir():
                raise FileNotFoundError(source)
            if mode == "symlink":
                target.symlink_to(source, target_is_directory=True)
            else:
                _copy_export_tree(source, target, exclusive=exclusive_copy)
            inventory_entries.append(
                {
                    "logical_path": logical_path,
                    "kind": kind,
                    "export_mode": mode,
                }
            )

    original_entry: dict[str, Any] | None = None
    tworoom_lance_entry: dict[str, Any] | None = None
    pusht_upstream_entries: list[dict[str, Any]] = []
    reacher_upstream_entries: list[dict[str, Any]] = []
    if include_upstream_original:
        original_source = resolve_original_h5(speed_release, repo_root=root)
        if not original_source.is_file():
            raise FileNotFoundError(original_source)
        original_target = (
            benchmark_root / "upstream/lewm-tworooms/tworoom.h5"
        )
        original_target.parent.mkdir(parents=True, exist_ok=True)
        if mode == "symlink":
            original_target.symlink_to(original_source)
        else:
            _copy_export_file(
                original_source,
                original_target,
                exclusive=exclusive_copy,
            )
        original_entry = {
            "logical_path": "upstream/lewm-tworooms/tworoom.h5",
            "kind": "file",
            "bytes": original_source.stat().st_size,
            "sha256": _sha256(original_source),
            "source": speed_release["training"]["original"]["source"],
            "license_reported_upstream": speed_release["training"][
                "original"
            ]["license"],
        }
        inventory_entries.append(original_entry)

        portal_upstream = portal_exit_release.get("training", {}).get(
            "upstream", {}
        )
        portal_lance_specification = portal_upstream.get("original_lance")
        if portal_lance_specification is not None:
            portal_lance_source = resolve_portal_original_lance(
                portal_exit_release,
                repo_root=root,
            )
            if not portal_lance_source.is_dir():
                raise FileNotFoundError(portal_lance_source)
            portal_lance_target = (
                benchmark_root
                / "upstream/stable-worldmodel/lewm_tworoom.lance"
            )
            portal_lance_target.parent.mkdir(parents=True, exist_ok=True)
            if mode == "symlink":
                portal_lance_target.symlink_to(
                    portal_lance_source,
                    target_is_directory=True,
                )
            else:
                _copy_export_tree(
                    portal_lance_source,
                    portal_lance_target,
                    exclusive=exclusive_copy,
                )
            tworoom_lance_entry = {
                "logical_path": (
                    "upstream/stable-worldmodel/lewm_tworoom.lance"
                ),
                "kind": "directory",
                "bytes": int(portal_lance_specification["bytes"]),
                "source_role": portal_lance_specification["role"],
            }
            inventory_entries.append(tworoom_lance_entry)

        pusht_sources = (
            (
                "original_h5",
                resolve_action_strength_original_h5(
                    action_strength_release,
                    repo_root=root,
                ),
                benchmark_root
                / "upstream/stable-worldmodel/pusht_expert_train.h5",
                "file",
            ),
            (
                "original_lance",
                resolve_action_strength_original_lance(
                    action_strength_release,
                    repo_root=root,
                ),
                benchmark_root
                / "upstream/stable-worldmodel/lewm_pusht.lance",
                "directory",
            ),
            (
                "initial_checkpoint",
                resolve_action_strength_initial_checkpoint(
                    action_strength_release,
                    repo_root=root,
                ),
                benchmark_root
                / "upstream/stable-worldmodel/"
                "pusht_lewm_baseline_seed3073_weights.ckpt",
                "file",
            ),
        )
        for name, source, target, kind in pusht_sources:
            if not source.exists():
                raise FileNotFoundError(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            if mode == "symlink":
                target.symlink_to(
                    source,
                    target_is_directory=(kind == "directory"),
                )
            elif kind == "directory":
                _copy_export_tree(
                    source, target, exclusive=exclusive_copy
                )
            else:
                _copy_export_file(
                    source, target, exclusive=exclusive_copy
                )
            specification = (
                action_strength_release["training"]["initialization"]
                if name == "initial_checkpoint"
                else action_strength_release["training"]["upstream"][name]
            )
            entry = {
                "logical_path": (
                    "upstream/stable-worldmodel/" + target.name
                ),
                "kind": kind,
                "bytes": int(specification["bytes"]),
                "source_role": specification["role"],
            }
            if specification.get("sha256"):
                entry["sha256"] = specification["sha256"]
            pusht_upstream_entries.append(entry)
            inventory_entries.append(entry)

        reacher_sources = (
            (
                "original_h5",
                resolve_reacher_original_h5(
                    robot_arm_mass_release,
                    repo_root=root,
                ),
                benchmark_root / "upstream/stable-worldmodel/reacher.h5",
                "file",
                robot_arm_mass_release["training"]["upstream"][
                    "original_h5"
                ],
            ),
            (
                "original_lance",
                resolve_reacher_original_lance(
                    robot_arm_mass_release,
                    repo_root=root,
                ),
                benchmark_root
                / "upstream/stable-worldmodel/lewm_reacher.lance",
                "directory",
                robot_arm_mass_release["training"]["upstream"][
                    "original_lance"
                ],
            ),
            (
                "lewm_checkpoint",
                resolve_reacher_initial_checkpoint(
                    robot_arm_mass_release,
                    "lewm",
                    repo_root=root,
                ),
                benchmark_root
                / "upstream/stable-worldmodel/reacher_lewm/"
                "reacher_lewm_weights.ckpt",
                "file",
                robot_arm_mass_release["training"]["reference_matrix"][
                    "initial_checkpoints"
                ]["lewm"],
            ),
            (
                "pldm_checkpoint",
                resolve_reacher_initial_checkpoint(
                    robot_arm_mass_release,
                    "pldm",
                    repo_root=root,
                ),
                benchmark_root
                / "upstream/stable-worldmodel/reacher_pldm_baseline/"
                "reacher_pldm_baseline_weights.ckpt",
                "file",
                robot_arm_mass_release["training"]["reference_matrix"][
                    "initial_checkpoints"
                ]["pldm"],
            ),
        )
        for name, source, target, kind, specification in reacher_sources:
            if not source.exists():
                raise FileNotFoundError(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            if mode == "symlink":
                target.symlink_to(
                    source,
                    target_is_directory=(kind == "directory"),
                )
            elif kind == "directory":
                _copy_export_tree(
                    source, target, exclusive=exclusive_copy
                )
            else:
                _copy_export_file(
                    source, target, exclusive=exclusive_copy
                )
            entry = {
                "logical_path": target.relative_to(benchmark_root).as_posix(),
                "kind": kind,
                "bytes": int(specification["bytes"]),
                "source_role": specification["role"],
            }
            if specification.get("sha256"):
                entry["sha256"] = specification["sha256"]
            reacher_upstream_entries.append(entry)
            inventory_entries.append(entry)

        for family in ("lewm", "pldm"):
            specification = robot_arm_mass_release["training"][
                "reference_matrix"
            ]["initial_checkpoints"][family]
            source = resolve_reacher_initial_checkpoint_config(
                robot_arm_mass_release,
                family,
                repo_root=root,
            )
            target = benchmark_root / specification[
                "config_bundled_artifact_path"
            ]
            target.parent.mkdir(parents=True, exist_ok=True)
            if mode == "symlink":
                target.symlink_to(source)
            else:
                _copy_export_file(
                    source, target, exclusive=exclusive_copy
                )
            entry = {
                "logical_path": target.relative_to(benchmark_root).as_posix(),
                "kind": "file",
                "bytes": int(specification["config_bytes"]),
                "sha256": specification["config_sha256"],
                "source_role": f"{family}_checkpoint_configuration",
            }
            reacher_upstream_entries.append(entry)
            inventory_entries.append(entry)

    releases_dir = benchmark_root / "releases"
    releases_dir.mkdir()
    _copy_export_file(
        speed_config, releases_dir / "speed.yaml", exclusive=exclusive_copy
    )
    _copy_export_file(
        door_config, releases_dir / "door.yaml", exclusive=exclusive_copy
    )
    _copy_export_file(
        action_delay_config,
        releases_dir / "action_delay.yaml",
        exclusive=exclusive_copy,
    )
    _copy_export_file(
        action_strength_config,
        releases_dir / "action_strength.yaml",
        exclusive=exclusive_copy,
    )
    _copy_export_file(
        contact_friction_config,
        releases_dir / "contact_friction.yaml",
        exclusive=exclusive_copy,
    )
    _copy_export_file(
        motion_damping_config,
        releases_dir / "motion_damping.yaml",
        exclusive=exclusive_copy,
    )
    _copy_export_file(
        robot_arm_mass_config,
        releases_dir / "robot_arm_mass.yaml",
        exclusive=exclusive_copy,
    )
    _copy_export_file(
        portal_exit_config,
        releases_dir / "portal_exit.yaml",
        exclusive=exclusive_copy,
    )
    if cube_config is not None:
        _copy_export_file(
            cube_config,
            releases_dir / "cube_gripper_carry.yaml",
            exclusive=exclusive_copy,
        )
    _copy_export_file(
        Path(suite["_config_path"]),
        benchmark_root / "suite.yaml",
        exclusive=exclusive_copy,
    )
    document = resolve_contextworld_path(
        suite["repository"]["public_document"]["path"],
        repo_root=root,
    )
    _copy_export_file(
        document, destination / "README.md", exclusive=exclusive_copy
    )

    payload = {
        "schema_version": 1,
        "release_id": suite["release_id"],
        "status": "passed",
        "release_kind": "local_technical_release_candidate",
        "membership_activation": membership_activation,
        "mode": mode,
        "top_level_entries": ["README.md", "benchmark"],
        "benchmark_root": "benchmark",
        "suite_config": "benchmark/suite.yaml",
        "component_release_configs": {
            "speed": "benchmark/releases/speed.yaml",
            "door": "benchmark/releases/door.yaml",
            "action_delay": "benchmark/releases/action_delay.yaml",
            "action_strength": (
                "benchmark/releases/action_strength.yaml"
            ),
            "contact_friction": (
                "benchmark/releases/contact_friction.yaml"
            ),
            "motion_damping": (
                "benchmark/releases/motion_damping.yaml"
            ),
            "robot_arm_mass": (
                "benchmark/releases/robot_arm_mass.yaml"
            ),
            "portal_exit": "benchmark/releases/portal_exit.yaml",
            **(
                {
                    "cube_gripper_carry": (
                        "benchmark/releases/cube_gripper_carry.yaml"
                    )
                }
                if cube_config is not None
                else {}
            ),
        },
        "public_results": (
            {
                key: str(
                    Path("benchmark").joinpath(
                        *Path(
                            suite["public_results"][key]["path"]
                        ).parts[1:]
                    )
                )
                for key in ("specification", "scoreboard")
            }
            if "public_results" in suite
            else {}
        ),
        "components": list(suite["components"]),
        "includes_upstream_original_h5": include_upstream_original,
        "public_test_included": True,
        "sealed_test_included": False,
        "redistribution_granted_by_export": False,
        "distribution": suite["distribution"],
        "entries": inventory_entries,
    }
    inventory_path = benchmark_root / "inventory.json"
    _write_export_inventory(
        inventory_path,
        payload,
        exclusive=exclusive_copy,
    )
    return {
        **payload,
        "destination": str(destination),
        "readme": str(destination / "README.md"),
        "benchmark_root_path": str(benchmark_root),
        "suite_config_path": str(benchmark_root / "suite.yaml"),
        "inventory": str(inventory_path),
        "upstream_original": (
            str(benchmark_root / "upstream/lewm-tworooms/tworoom.h5")
            if original_entry is not None
            else None
        ),
        "tworoom_upstream": (
            {
                "original_h5": str(
                    benchmark_root / "upstream/lewm-tworooms/tworoom.h5"
                ),
                "original_lance": str(
                    benchmark_root
                    / "upstream/stable-worldmodel/lewm_tworoom.lance"
                ),
            }
            if original_entry is not None and tworoom_lance_entry is not None
            else None
        ),
        "pusht_upstream": (
            {
                "original_h5": str(
                    benchmark_root
                    / "upstream/stable-worldmodel/pusht_expert_train.h5"
                ),
                "original_lance": str(
                    benchmark_root
                    / "upstream/stable-worldmodel/lewm_pusht.lance"
                ),
                "initial_checkpoint": str(
                    benchmark_root
                    / "upstream/stable-worldmodel/"
                    "pusht_lewm_baseline_seed3073_weights.ckpt"
                ),
            }
            if pusht_upstream_entries
            else None
        ),
        "reacher_upstream": (
            {
                "original_h5": str(
                    benchmark_root
                    / "upstream/stable-worldmodel/reacher.h5"
                ),
                "original_lance": str(
                    benchmark_root
                    / "upstream/stable-worldmodel/lewm_reacher.lance"
                ),
                "lewm_checkpoint": str(
                    benchmark_root
                    / "upstream/stable-worldmodel/reacher_lewm/"
                    "reacher_lewm_weights.ckpt"
                ),
                "pldm_checkpoint": str(
                    benchmark_root
                    / "upstream/stable-worldmodel/reacher_pldm_baseline/"
                    "reacher_pldm_baseline_weights.ckpt"
                ),
            }
            if reacher_upstream_entries
            else None
        ),
    }


def export_icl_suite_artifacts(
    destination: Path | str,
    *,
    release_config: Path | str = DEFAULT_SUITE_RELEASE_CONFIG,
    repo_root: Path | None = None,
    mode: str = "copy",
    include_upstream_original: bool = True,
) -> dict[str, Any]:
    return _export_icl_suite_artifacts_impl(
        destination,
        release_config=release_config,
        repo_root=repo_root,
        mode=mode,
        include_upstream_original=include_upstream_original,
    )


def _export_icl_suite_artifacts_for_registration(
    destination: Path | str,
    *,
    release_config: Path | str = DEFAULT_SUITE_V2_RELEASE_CONFIG,
    repo_root: Path | None = None,
    mode: str = "copy",
    include_upstream_original: bool = True,
) -> dict[str, Any]:
    """Export only inside the preregistered pre-decision finalizer."""

    return _export_icl_suite_artifacts_impl(
        destination,
        release_config=release_config,
        repo_root=repo_root,
        mode=mode,
        include_upstream_original=include_upstream_original,
        registration_capability=_SUITE_V2_REGISTRATION_AUDIT_CAPABILITY,
    )


__all__ = [
    "COMPONENT_IDS",
    "DEFAULT_SUITE_RELEASE_CONFIG",
    "DEFAULT_SUITE_V2_RELEASE_CONFIG",
    "SUITE_RELEASE_ID",
    "SUITE_V2_COMPONENT_IDS",
    "SUITE_V2_RELEASE_ID",
    "audit_icl_suite_release",
    "export_icl_suite_artifacts",
    "load_icl_suite_release",
    "load_public_scoreboard",
    "require_suite_membership_activation",
]
