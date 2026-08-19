"""Inactive, additive integrity reseal for the final Suite v2 handoff.

The v1 recovery, documentation amendment, and integrity-reseal decision are
historical commit markers.  This module never rewrites or activates them.  It
only provides a fail-closed, one-use decision builder once the four PLDM
completion outcomes, independent scoreboard resolution, and public documents
have all settled.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping

import yaml

from contextworld.benchmarks.action_strength_icl_data import (
    audit_action_strength_icl_release,
    load_action_strength_icl_release,
)
from contextworld.benchmarks.cube_grasp_rule_suite_registration import (
    resolve_no_symlink_contextworld_path,
)
from contextworld.benchmarks.original_baseline_archive import (
    audit_archived_original_baseline_matrix,
)
from contextworld.benchmarks.pldm_reference_completion_aggregate import (
    AGGREGATE_ID as FINAL_PLDMS_AGGREGATE_ID,
    validate_written_completion_aggregate_and_scoreboard,
)
from contextworld.benchmarks.public_score import make_public_scoreboard_from_spec
from contextworld.paths import repository_root, resolve_contextworld_path


RESEAL_ID = "contextworld_icl_suite_v2_integrity_reseal_v2"
RESEAL_CONFIG = (
    repository_root()
    / "configs/benchmark/contextworld_icl_suite_v2_integrity_reseal_v2.yaml"
)
RESEAL_DECISION_PATH = (
    "configs/benchmark/contextworld_icl_suite_v2_integrity_reseal_decision_v2.json"
)
CURRENT_RESULTS_OVERLAY_ID = "contextworld_icl_suite_v2_current_results_overlay_v1"
CURRENT_RESULTS_OVERLAY_PATH = (
    "configs/benchmark/contextworld_icl_suite_v2_current_results_overlay_v1.yaml"
)
HISTORICAL_V1_RESEAL_CONFIG_PATH = (
    "configs/benchmark/contextworld_icl_suite_v2_integrity_reseal_v1.yaml"
)
ACTION_STRENGTH_FLOAT32_AMENDMENT_ID = (
    "pusht_action_strength_score_float32_consistency_amendment_v1"
)
ACTION_STRENGTH_FLOAT32_AMENDMENT_PATH = (
    "configs/benchmark/"
    "pusht_action_strength_score_float32_consistency_amendment_v1.yaml"
)
ACTION_STRENGTH_RELEASE_CONFIG_PATH = (
    "configs/benchmark/pusht_action_strength_icl_release_v1.yaml"
)
ACTION_STRENGTH_SCORER_PATH = (
    "contextworld/benchmarks/action_strength_icl_score.py"
)
ACTION_STRENGTH_RELEASE_AUDITOR_PATH = (
    "contextworld/benchmarks/action_strength_icl_data.py"
)
ORIGINAL_BASELINE_ARCHIVE_AUDITOR_MODULE_PATH = (
    "contextworld/benchmarks/original_baseline_archive.py"
)
ORIGINAL_BASELINE_ARCHIVE_AUDITOR_MODULE_SHA256 = (
    "0b076e05b7ee6d065dd0dba62556d1ff559367320dcf00669fc462acf9d403cc"
)
ORIGINAL_BASELINE_ARCHIVE_AUDITOR_MODULE_SIZE_BYTES = 11462
ORIGINAL_BASELINE_ARCHIVE_AUDITOR_CLI_PATH = (
    "scripts/audit_contextworld_original_baseline_matrix_freeze_v1.py"
)
ORIGINAL_BASELINE_ARCHIVE_AUDITOR_CLI_SHA256 = (
    "0c971e75d76e29382df50ae5e3197e49c12b5ec444810353fd54ed69a4e753f3"
)
ORIGINAL_BASELINE_ARCHIVE_AUDITOR_CLI_SIZE_BYTES = 559
ORIGINAL_BASELINE_ARCHIVE_AUDIT_ID = (
    "contextworld_original_baseline_archive_audit_v1"
)
FINAL_PLDMS_AGGREGATE_FREEZE_ID = (
    "contextworld_pldm_reference_completion_aggregate_results_freeze_v1"
)
FINAL_PLDMS_AGGREGATE_FREEZE_PATH = (
    "configs/benchmark/contextworld_pldm_reference_completion_"
    "aggregate_results_freeze_v1.json"
)
FINAL_PLDMS_AGGREGATE_PREREGISTRATION_PATH = (
    "configs/benchmark/contextworld_pldm_reference_completion_"
    "aggregate_prereg_v1.yaml"
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
    "cube_gripper_carry",
)
PLDM_COMPLETION_COMPONENT_IDS = (
    "speed",
    "action_strength",
    "contact_friction",
    "motion_damping",
)
RESEAL_AUTHORITY_IMPLEMENTATION_PATHS = (
    "contextworld/benchmarks/suite_v2_integrity_reseal_v2.py",
    "scripts/freeze_contextworld_icl_suite_v2_integrity_reseal_v2.py",
)


class ResealBlocked(ValueError):
    """Raised when evidence required for the one-use decision is incomplete."""

    def __init__(self, blockers: list[str]) -> None:
        self.blockers = blockers
        super().__init__("; ".join(blockers))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(logical_path: str, *, repo_root: Path) -> dict[str, Any]:
    path = resolve_contextworld_path(logical_path, repo_root=repo_root)
    if not path.is_file():
        raise FileNotFoundError(logical_path)
    return {
        "path": logical_path,
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _config_identity(path: Path, *, repo_root: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    try:
        logical_path = str(resolved.relative_to(repo_root))
    except ValueError as error:
        raise ValueError("reseal config must live under the repository root") from error
    return _identity(logical_path, repo_root=repo_root)


def _read_yaml(logical_path: str, *, repo_root: Path) -> dict[str, Any]:
    path = resolve_contextworld_path(logical_path, repo_root=repo_root)
    if not path.is_file():
        raise FileNotFoundError(logical_path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{logical_path} must contain a mapping")
    return payload


def _read_json(logical_path: str, *, repo_root: Path) -> dict[str, Any]:
    path = resolve_contextworld_path(logical_path, repo_root=repo_root)
    if not path.is_file():
        raise FileNotFoundError(logical_path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{logical_path} is not valid JSON") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{logical_path} must contain a JSON object")
    return payload


def _require_identity(
    observed: Mapping[str, Any], expected: Any, *, label: str
) -> None:
    if not isinstance(expected, Mapping) or dict(observed) != dict(expected):
        raise ValueError(f"{label} identity drifted")


def _expected_mapping(
    value: Any, *, label: str, keys: set[str] | None = None
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    if keys is not None and set(value) != keys:
        raise ValueError(f"{label} keys are invalid")
    return value


def load_current_results_overlay_preregistration(
    path: Path | str = CURRENT_RESULTS_OVERLAY_PATH,
    *,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Load the pointer-only current-results view without activating it.

    The overlay is deliberately static.  Its decision and final-result hashes
    live only in the later v2 decision, which prevents a configuration hash
    from depending on the decision that must bind that configuration.
    """

    root = (repo_root or repository_root()).resolve()
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        try:
            logical_path = str(candidate.resolve().relative_to(root))
        except ValueError as error:
            raise ValueError(
                "current-results overlay must live under the repository root"
            ) from error
    else:
        logical_path = str(candidate)
    if logical_path != CURRENT_RESULTS_OVERLAY_PATH:
        raise ValueError("current-results overlay path is not canonical")
    config_path = resolve_contextworld_path(logical_path, repo_root=root)
    if not config_path.is_file():
        raise FileNotFoundError(logical_path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    expected_keys = {
        "schema_version",
        "release_id",
        "release_status",
        "candidate_date",
        "current_results_overlay",
        "membership_authority",
    }
    if (
        not isinstance(payload, dict)
        or set(payload) != expected_keys
        or payload.get("schema_version") != 1
        or payload.get("release_id") != "contextworld_icl_benchmark_suite_v2"
        or payload.get("release_status") != "public_test_release_candidate"
    ):
        raise ValueError("current-results overlay preregistration is invalid")

    overlay = _expected_mapping(
        payload.get("current_results_overlay"),
        label="current-results overlay",
        keys={
            "overlay_id",
            "status",
            "scope",
            "base_release_config",
            "final_reseal",
            "result_source",
            "allowed_changes",
            "prohibited_changes",
            "decision_contract",
        },
    )
    base = _expected_mapping(
        overlay.get("base_release_config"),
        label="current-results overlay base release config",
        keys={"path", "sha256", "size_bytes"},
    )
    final_reseal = _expected_mapping(
        overlay.get("final_reseal"),
        label="current-results overlay final reseal",
        keys={"reseal_id", "reseal_config_path", "decision_path"},
    )
    contract = _expected_mapping(
        overlay.get("decision_contract"),
        label="current-results overlay decision contract",
        keys={
            "overlay_identity_must_be_bound_by_v2_decision",
            "decision_hash_is_not_declared_in_overlay",
            "activation_requires_final_v2_decision",
            "formal_rows_and_components_must_be_derived_from_bound_scoreboard",
            "partial_outputs_grant_current_results",
        },
    )
    if (
        overlay.get("overlay_id") != CURRENT_RESULTS_OVERLAY_ID
        or overlay.get("status")
        != "preregistered_pending_integrity_reseal_v2"
        or overlay.get("scope") != "decision_gated_current_public_results_view"
        or base.get("path") != HISTORICAL_V1_RESEAL_CONFIG_PATH
        or not isinstance(base.get("sha256"), str)
        or len(base["sha256"]) != 64
        or type(base.get("size_bytes")) is not int
        or final_reseal
        != {
            "reseal_id": RESEAL_ID,
            "reseal_config_path": (
                "configs/benchmark/"
                "contextworld_icl_suite_v2_integrity_reseal_v2.yaml"
            ),
            "decision_path": RESEAL_DECISION_PATH,
        }
        or overlay.get("result_source")
        != "release_materials.public_scoreboard"
        or overlay.get("allowed_changes")
        != [
            "in_memory_current_identity_overlay",
            "runtime_default_selection_after_valid_final_decision",
        ]
        or overlay.get("prohibited_changes")
        != [
            "rewrite_historical_v1_config_or_scoreboard",
            "declare_final_decision_hash_in_overlay",
            "declare_final_scoreboard_hash_or_row_count_in_overlay",
            "raw_dataset_or_checkpoint_mutation",
            "training_or_checkpoint_selection",
            "public_test_access_or_rerun_authorization",
        ]
        or contract
        != {
            "overlay_identity_must_be_bound_by_v2_decision": True,
            "decision_hash_is_not_declared_in_overlay": True,
            "activation_requires_final_v2_decision": True,
            "formal_rows_and_components_must_be_derived_from_bound_scoreboard": True,
            "partial_outputs_grant_current_results": False,
        }
    ):
        raise ValueError("current-results overlay contract is invalid")

    authority = _expected_mapping(
        payload.get("membership_authority"),
        label="current-results overlay membership authority",
    )
    expected_authority = {
        "config_alone_grants_membership": False,
        "activation_condition": "passed_integrity_reseal_decision_v2",
        "overlay_id": CURRENT_RESULTS_OVERLAY_ID,
        "reseal_id": RESEAL_ID,
        "decision_path": RESEAL_DECISION_PATH,
        "decision_is_commit_marker": True,
        "base_release_config": HISTORICAL_V1_RESEAL_CONFIG_PATH,
        "historical_v1_results_remain_explicitly_addressable": True,
        "partial_outputs_grant_membership": False,
        "public_test_rerun_authorized": False,
        "training_or_checkpoint_selection_authorized": False,
        "formal_scoreboard_mutation_authorized": False,
        "raw_dataset_or_checkpoint_mutation_authorized": False,
    }
    if authority != expected_authority:
        raise ValueError("current-results overlay membership authority is invalid")
    return {
        "overlay": overlay,
        "membership_authority": authority,
        "base_release_config": base,
        "_config_path": str(config_path),
        "_config_identity": _identity(CURRENT_RESULTS_OVERLAY_PATH, repo_root=root),
    }


def load_integrity_reseal_v2_preregistration(
    path: Path | str = RESEAL_CONFIG,
    *,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Load the inactive plan and reject drift in every v1 predecessor file."""

    root = (repo_root or repository_root()).resolve()
    config_path = Path(path).expanduser().resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    expected_keys = {
        "schema_version",
        "release_id",
        "release_status",
        "candidate_date",
        "integrity_reseal",
        "membership_authority",
    }
    if (
        not isinstance(payload, dict)
        or set(payload) != expected_keys
        or payload.get("schema_version") != 1
        or payload.get("release_id") != "contextworld_icl_benchmark_suite_v2"
        or payload.get("release_status") != "public_test_release_candidate"
    ):
        raise ValueError("Suite v2 integrity-reseal-v2 preregistration is invalid")

    reseal = _expected_mapping(
        payload.get("integrity_reseal"),
        label="integrity_reseal",
        keys={
            "reseal_id",
            "status",
            "scope",
            "historical_predecessor",
            "required_evidence",
            "decision_contract",
        },
    )
    if (
        reseal.get("reseal_id") != RESEAL_ID
        or reseal.get("status")
        != "preregistered_pending_final_identity_freeze"
    ):
        raise ValueError("Suite v2 integrity-reseal-v2 identity is invalid")

    predecessor = _expected_mapping(
        reseal.get("historical_predecessor"),
        label="historical_predecessor",
        keys={
            "v1_preregistration",
            "v1_decision",
            "v1_authority_module",
            "v1_freeze_cli",
        },
    )
    expected_predecessor_paths = {
        "v1_preregistration": (
            "configs/benchmark/contextworld_icl_suite_v2_integrity_reseal_v1.yaml"
        ),
        "v1_decision": (
            "configs/benchmark/"
            "contextworld_icl_suite_v2_integrity_reseal_decision_v1.json"
        ),
        "v1_authority_module": (
            "contextworld/benchmarks/suite_v2_integrity_reseal.py"
        ),
        "v1_freeze_cli": (
            "scripts/freeze_contextworld_icl_suite_v2_integrity_reseal_v1.py"
        ),
    }
    for name, logical_path in expected_predecessor_paths.items():
        _require_identity(
            _identity(logical_path, repo_root=root),
            predecessor[name],
            label=f"historical predecessor {name}",
        )

    # A byte-identical v1 preregistration is necessary but not sufficient if a
    # file named inside its own historical chain has later been replaced.  Read
    # the immutable v1 chain directly and recheck each recorded identity.
    v1_payload = _read_yaml(
        expected_predecessor_paths["v1_preregistration"], repo_root=root
    )
    v1_reseal = _expected_mapping(
        v1_payload.get("integrity_reseal"), label="v1 integrity_reseal"
    )
    v1_chain = _expected_mapping(
        v1_reseal.get("historical_chain"), label="v1 historical_chain"
    )
    expected_v1_chain = {
        "documentation_amendment_config",
        "documentation_amendment_decision",
        "recovery_v2_config",
        "recovery_v2_decision",
    }
    if set(v1_chain) != expected_v1_chain:
        raise ValueError("v1 historical chain structure drifted")
    for name, identity in v1_chain.items():
        if not isinstance(identity, Mapping) or not isinstance(
            identity.get("path"), str
        ):
            raise ValueError(f"v1 historical chain {name} is invalid")
        _require_identity(
            _identity(identity["path"], repo_root=root),
            identity,
            label=f"v1 historical chain {name}",
        )

    evidence = _expected_mapping(
        reseal.get("required_evidence"),
        label="required_evidence",
        keys={
            "engineering_identity_amendment",
            "capability_taxonomy_documentation_amendment",
            "action_strength_float32_consistency_amendment",
            "original_baseline_archive_auditor",
            "current_results_overlay",
            "final_public_documents",
            "current_component_release_configs",
            "registered_suite_sources",
            "original_baseline_result_freezes",
            "pldm_completion_preregistrations",
            "final_pldm_completion_aggregate_results_freeze",
            "public_scoreboard",
        },
    )
    component_paths = _expected_mapping(
        evidence.get("current_component_release_configs"),
        label="current_component_release_configs",
    )
    if tuple(component_paths) != COMPONENT_IDS:
        raise ValueError("current component release configs are not complete and ordered")
    pldm_configs = _expected_mapping(
        evidence.get("pldm_completion_preregistrations"),
        label="pldm_completion_preregistrations",
    )
    if tuple(pldm_configs) != PLDM_COMPLETION_COMPONENT_IDS:
        raise ValueError("PLDM completion preregistrations are incomplete")
    final_freeze = _expected_mapping(
        evidence.get("final_pldm_completion_aggregate_results_freeze"),
        label="final_pldm_completion_aggregate_results_freeze",
    )
    if (
        final_freeze.get("path") != FINAL_PLDMS_AGGREGATE_FREEZE_PATH
        or final_freeze.get("freeze_id") != FINAL_PLDMS_AGGREGATE_FREEZE_ID
        or final_freeze.get("required_status")
        != "frozen_after_all_four_pldm_reference_completion_outcomes"
    ):
        raise ValueError("final PLDM aggregate-freeze contract is invalid")

    overlay_specification = _expected_mapping(
        evidence.get("current_results_overlay"),
        label="current-results overlay evidence",
        keys={"path", "overlay_id"},
    )
    if (
        overlay_specification.get("path") != CURRENT_RESULTS_OVERLAY_PATH
        or overlay_specification.get("overlay_id")
        != CURRENT_RESULTS_OVERLAY_ID
    ):
        raise ValueError("current-results overlay evidence is invalid")
    overlay = load_current_results_overlay_preregistration(
        CURRENT_RESULTS_OVERLAY_PATH,
        repo_root=root,
    )
    if overlay["overlay"].get("overlay_id") != CURRENT_RESULTS_OVERLAY_ID:
        raise ValueError("current-results overlay identity is invalid")

    _action_strength_float32_consistency_material(
        _expected_mapping(
            evidence.get("action_strength_float32_consistency_amendment"),
            label="action_strength_float32_consistency_amendment",
        ),
        repo_root=root,
        require_current_release_audit=False,
    )
    _original_baseline_archive_auditor_material(
        _expected_mapping(
            evidence.get("original_baseline_archive_auditor"),
            label="original_baseline_archive_auditor",
        ),
        repo_root=root,
    )

    contract = _expected_mapping(reseal.get("decision_contract"), label="decision_contract")
    if (
        contract.get("decision_path") != RESEAL_DECISION_PATH
        or contract.get("exclusive_creation_required") is not True
        or contract.get("current_materials_must_be_exactly_bound") is not True
        or contract.get("historical_predecessor_must_remain_byte_identical")
        is not True
        or contract.get("missing_evidence_blocks_check_only_and_decision_creation")
        is not True
        or contract.get("scoreboard_resolution_must_be_independent") is not True
        or contract.get("historical_base_scoreboard_must_remain_byte_identical")
        is not True
        or contract.get("addendum_output_must_not_overwrite_historical_base")
        is not True
        or contract.get("no_extension_requires_exactly_eleven_formal_rows")
        is not True
        or contract.get("additive_extension_row_count_must_be_derived_from_final_files")
        is not True
        or contract.get("activation_requires_new_decision") is not True
    ):
        raise ValueError("integrity-reseal-v2 decision contract is invalid")

    authority = _expected_mapping(payload.get("membership_authority"), label="membership_authority")
    required_authority = {
        "config_alone_grants_membership": False,
        "activation_condition": "passed_integrity_reseal_decision_v2",
        "reseal_id": RESEAL_ID,
        "decision_path": RESEAL_DECISION_PATH,
        "decision_is_commit_marker": True,
        "historical_predecessor_must_remain_byte_identical": True,
        "partial_outputs_grant_membership": False,
        "public_test_rerun_authorized": False,
        "training_or_checkpoint_selection_authorized": False,
        "raw_dataset_or_checkpoint_mutation_authorized": False,
        "config_is_not_a_suite_default_or_activation_switch": True,
    }
    if authority != required_authority:
        raise ValueError("integrity-reseal-v2 membership authority is invalid")
    return {
        **reseal,
        "_config_path": str(config_path),
        "_validated_v1_historical_chain": v1_chain,
    }


def _engineering_material(
    specification: Mapping[str, Any], *, repo_root: Path
) -> dict[str, Any]:
    logical_path = str(specification.get("path", ""))
    payload = _read_yaml(logical_path, repo_root=repo_root)
    amendment = _expected_mapping(
        payload.get("engineering_identity_amendment"),
        label="engineering identity amendment",
    )
    if (
        amendment.get("amendment_id") != specification.get("amendment_id")
        or amendment.get("status") != "preregistered_pending_integrity_reseal_v2"
    ):
        raise ValueError("engineering identity amendment identity is invalid")
    updates = _expected_mapping(
        amendment.get("approved_component_identity_updates"),
        label="engineering component updates",
    )
    if tuple(updates) != COMPONENT_IDS:
        raise ValueError("engineering amendment does not cover all nine components")
    sources: dict[str, Any] = {}
    for component_id, update in updates.items():
        update = _expected_mapping(update, label=f"engineering update {component_id}")
        source_paths = update.get("source_paths")
        if not isinstance(source_paths, list) or not source_paths:
            raise ValueError(f"engineering update {component_id} has no source paths")
        for source_path in source_paths:
            if not isinstance(source_path, str) or not source_path:
                raise ValueError(f"engineering update {component_id} source path is invalid")
            sources[source_path] = _identity(source_path, repo_root=repo_root)
    return {
        "record": _identity(logical_path, repo_root=repo_root),
        "declared_current_sources": {
            path: sources[path] for path in sorted(sources)
        },
    }


def _action_strength_float32_consistency_material(
    specification: Mapping[str, Any],
    *,
    repo_root: Path,
    require_current_release_audit: bool = True,
) -> dict[str, Any]:
    """Bind the numerical-reconstruction amendment to a passed live audit.

    The historical recovery release hash intentionally differs from the
    current Action Strength release.  This successor verifies the named
    maintenance amendment and audits the current release instead of treating
    that historical hash as a live-byte expectation.
    """

    expected = _expected_mapping(
        specification,
        label="Action Strength float32 consistency amendment evidence",
        keys={
            "path",
            "sha256",
            "size_bytes",
            "amendment_id",
            "release_config",
            "scorer_path",
            "release_auditor_path",
        },
    )
    if (
        expected.get("path") != ACTION_STRENGTH_FLOAT32_AMENDMENT_PATH
        or expected.get("sha256")
        != "07ea18d4ebf9df5f798e5b3d0761b19145c5caab162ef513ffdb4150bc368d0a"
        or expected.get("size_bytes") != 3676
        or expected.get("amendment_id")
        != ACTION_STRENGTH_FLOAT32_AMENDMENT_ID
        or expected.get("release_config") != ACTION_STRENGTH_RELEASE_CONFIG_PATH
        or expected.get("scorer_path") != ACTION_STRENGTH_SCORER_PATH
        or expected.get("release_auditor_path")
        != ACTION_STRENGTH_RELEASE_AUDITOR_PATH
    ):
        raise ValueError("Action Strength float32 amendment evidence is invalid")
    amendment_identity = _identity(
        ACTION_STRENGTH_FLOAT32_AMENDMENT_PATH, repo_root=repo_root
    )
    _require_identity(
        amendment_identity,
        {
            key: expected[key] for key in ("path", "sha256", "size_bytes")
        },
        label="Action Strength float32 amendment",
    )
    amendment = _read_yaml(
        ACTION_STRENGTH_FLOAT32_AMENDMENT_PATH, repo_root=repo_root
    )
    purpose = _expected_mapping(
        amendment.get("purpose"), label="Action Strength float32 purpose"
    )
    required_behavior = _expected_mapping(
        amendment.get("required_post_change_behavior"),
        label="Action Strength float32 required behavior",
    )
    if (
        amendment.get("schema_version") != 1
        or amendment.get("amendment_id") != ACTION_STRENGTH_FLOAT32_AMENDMENT_ID
        or amendment.get("release_id")
        != "contextworld_pusht_action_strength_icl_history3_v1"
        or purpose.get("scientific_effect") != "none"
        or purpose.get("gate_or_threshold_change_authorized") is not False
        or purpose.get("raw_result_rewrite_authorized") is not False
        or purpose.get("model_evaluation_rerun_authorized") is not False
        or purpose.get("checkpoint_selection_authorized") is not False
        or purpose.get("public_test_reopen_authorized") is not False
        or required_behavior.get(
            "public_score_command_accepts_all_three_frozen_raw_receipts"
        )
        is not True
        or required_behavior.get(
            "reconstructed_gate_matches_independent_recovery_bit_for_bit"
        )
        is not True
        or required_behavior.get("cem_authorization_must_remain_false")
        is not True
        or required_behavior.get("historical_receipts_must_remain_byte_identical")
        is not True
    ):
        raise ValueError("Action Strength float32 amendment contract is invalid")

    release = load_action_strength_icl_release(
        resolve_contextworld_path(
            ACTION_STRENGTH_RELEASE_CONFIG_PATH, repo_root=repo_root
        )
    )
    amendments = release.get("maintenance_amendments")
    expected_release_amendment = {
        "amendment_id": ACTION_STRENGTH_FLOAT32_AMENDMENT_ID,
        "path": ACTION_STRENGTH_FLOAT32_AMENDMENT_PATH,
        "sha256": amendment_identity["sha256"],
        "scope": "public_score_command_numeric_reconstruction_only",
    }
    if (
        release.get("release_id")
        != "contextworld_pusht_action_strength_icl_history3_v1"
        or not isinstance(amendments, list)
        or expected_release_amendment not in amendments
    ):
        raise ValueError(
            "current Action Strength release does not bind the float32 amendment"
        )
    material: dict[str, Any] = {
        "amendment": amendment_identity,
        "current_release_config": _identity(
            ACTION_STRENGTH_RELEASE_CONFIG_PATH, repo_root=repo_root
        ),
        "current_scorer": _identity(
            ACTION_STRENGTH_SCORER_PATH, repo_root=repo_root
        ),
        "current_release_auditor": _identity(
            ACTION_STRENGTH_RELEASE_AUDITOR_PATH, repo_root=repo_root
        ),
    }
    if not require_current_release_audit:
        return material
    audit = audit_action_strength_icl_release(
        release_config=resolve_contextworld_path(
            ACTION_STRENGTH_RELEASE_CONFIG_PATH, repo_root=repo_root
        ),
        repo_root=repo_root,
        full=False,
    )
    if (
        audit.get("release_id") != release["release_id"]
        or audit.get("status") != "passed"
        or audit.get("passed") is not True
    ):
        raise ValueError("current Action Strength release audit did not pass")
    material["current_release_audit"] = {
        "release_id": audit["release_id"],
        "status": audit["status"],
        "passed": True,
        "full_content_hash_audit": audit.get("full_content_hash_audit"),
    }
    return material


def _original_baseline_archive_auditor_material(
    specification: Mapping[str, Any], *, repo_root: Path
) -> dict[str, Any]:
    """Bind the immutable-baseline auditor and a successful archive check.

    The audit verifies frozen original-baseline receipts only.  It does not
    reconstruct historical results from live release files, because later
    maintenance can legitimately change those files without changing the
    archived result evidence.
    """

    expected = _expected_mapping(
        specification,
        label="original baseline archive auditor evidence",
        keys={
            "module",
            "cli",
            "audit_id",
            "expected_status",
            "archive_scope",
            "live_release_rederivation_performed",
        },
    )
    expected_module = {
        "path": ORIGINAL_BASELINE_ARCHIVE_AUDITOR_MODULE_PATH,
        "sha256": ORIGINAL_BASELINE_ARCHIVE_AUDITOR_MODULE_SHA256,
        "size_bytes": ORIGINAL_BASELINE_ARCHIVE_AUDITOR_MODULE_SIZE_BYTES,
    }
    expected_cli = {
        "path": ORIGINAL_BASELINE_ARCHIVE_AUDITOR_CLI_PATH,
        "sha256": ORIGINAL_BASELINE_ARCHIVE_AUDITOR_CLI_SHA256,
        "size_bytes": ORIGINAL_BASELINE_ARCHIVE_AUDITOR_CLI_SIZE_BYTES,
    }
    if (
        expected.get("module") != expected_module
        or expected.get("cli") != expected_cli
        or expected.get("audit_id") != ORIGINAL_BASELINE_ARCHIVE_AUDIT_ID
        or expected.get("expected_status") != "passed"
        or expected.get("archive_scope") != "immutable_frozen_results_only"
        or expected.get("live_release_rederivation_performed") is not False
    ):
        raise ValueError("original baseline archive auditor evidence is invalid")
    module_identity = _identity(
        ORIGINAL_BASELINE_ARCHIVE_AUDITOR_MODULE_PATH, repo_root=repo_root
    )
    cli_identity = _identity(
        ORIGINAL_BASELINE_ARCHIVE_AUDITOR_CLI_PATH, repo_root=repo_root
    )
    _require_identity(
        module_identity,
        expected_module,
        label="original baseline archive auditor module",
    )
    _require_identity(
        cli_identity,
        expected_cli,
        label="original baseline archive auditor CLI",
    )
    audit = audit_archived_original_baseline_matrix(repo_root=repo_root)
    if (
        audit.get("audit_id") != ORIGINAL_BASELINE_ARCHIVE_AUDIT_ID
        or audit.get("status") != "passed"
        or audit.get("archive_scope") != "immutable_frozen_results_only"
        or audit.get("live_release_rederivation_performed") is not False
    ):
        raise ValueError("original baseline archive audit did not pass")
    return {
        "module": module_identity,
        "cli": cli_identity,
        "audit": {
            "audit_id": audit["audit_id"],
            "status": audit["status"],
            "archive_scope": audit["archive_scope"],
            "live_release_rederivation_performed": False,
        },
    }


def _taxonomy_documentation_material(
    specification: Mapping[str, Any],
    document_paths: Mapping[str, Any],
    *,
    repo_root: Path,
) -> dict[str, Any]:
    logical_path = str(specification.get("path", ""))
    payload = _read_yaml(logical_path, repo_root=repo_root)
    amendment = _expected_mapping(
        payload.get("capability_taxonomy_documentation_amendment"),
        label="capability taxonomy documentation amendment",
    )
    chronology = _expected_mapping(amendment.get("chronology"), label="taxonomy chronology")
    targets = _expected_mapping(
        amendment.get("documentation_targets"), label="taxonomy documentation targets")
    if (
        amendment.get("record_id") != specification.get("record_id")
        or amendment.get("status")
        != "current_documentation_changes_pending_v2_reseal"
        or chronology.get("preregistration_claimed") is not False
        or targets != dict(document_paths)
    ):
        raise ValueError("capability taxonomy documentation amendment is invalid")
    allowed = amendment.get("allowed_changes")
    prohibited = amendment.get("prohibited_changes")
    if (
        not isinstance(allowed, list)
        or not isinstance(prohibited, list)
        or "external_narrative_organization_by_capability_type" not in allowed
        or "public_heading_and_navigation_labels" not in allowed
        or "public_task_card_template_audit_structure" not in allowed
        or "raw_dataset_or_checkpoint_mutation" not in prohibited
        or "score_or_threshold_mutation" not in prohibited
        or "public_test_access_or_rerun_authorization" not in prohibited
        or "training_or_checkpoint_selection" not in prohibited
    ):
        raise ValueError("capability taxonomy amendment scope is not fail-closed")
    return {
        "record": _identity(logical_path, repo_root=repo_root),
        "final_public_documents": {
            name: _identity(str(path), repo_root=repo_root)
            for name, path in document_paths.items()
        },
    }


def _registered_suite_source_material(
    specification: Mapping[str, Any], *, repo_root: Path
) -> dict[str, Any]:
    manifest_path = str(specification.get("source_manifest", ""))
    expected_paths = specification.get("paths")
    if not isinstance(expected_paths, list) or not all(
        isinstance(path, str) and path for path in expected_paths
    ):
        raise ValueError("registered suite source paths are invalid")
    if len(set(expected_paths)) != len(expected_paths):
        raise ValueError("registered suite source paths are duplicated")
    manifest = _read_yaml(manifest_path, repo_root=repo_root)
    observed_paths = set(
        _expected_mapping(
            _expected_mapping(manifest.get("repository"), label="registered source manifest repository").get(
                "source_sha256"
            ),
            label="registered source manifest source_sha256",
        )
    )
    if observed_paths != set(expected_paths):
        raise ValueError("registered suite source manifest no longer matches v2")
    return {
        "source_manifest": _identity(manifest_path, repo_root=repo_root),
        "sources": {
            path: _identity(path, repo_root=repo_root)
            for path in sorted(expected_paths)
        },
    }


def _baseline_freeze_material(
    specifications: Mapping[str, Any], *, repo_root: Path
) -> dict[str, Any]:
    expected_names = {"original_icl", "original_cem"}
    if set(specifications) != expected_names:
        raise ValueError("original baseline freeze contract is incomplete")
    material: dict[str, Any] = {}
    for name, specification in specifications.items():
        specification = _expected_mapping(specification, label=f"{name} freeze")
        logical_path = str(specification.get("path", ""))
        payload = _read_json(logical_path, repo_root=repo_root)
        if payload.get("freeze_id") != specification.get("freeze_id"):
            raise ValueError(f"{name} freeze_id is invalid")
        material[name] = _identity(logical_path, repo_root=repo_root)
    return material


def _pldm_completion_material(
    specifications: Mapping[str, Any], *, repo_root: Path
) -> dict[str, Any]:
    if tuple(specifications) != PLDM_COMPLETION_COMPONENT_IDS:
        raise ValueError("PLDM completion preregistration contract is incomplete")
    material: dict[str, Any] = {}
    for component_id, specification in specifications.items():
        specification = _expected_mapping(
            specification, label=f"PLDM completion {component_id}"
        )
        logical_path = str(specification.get("path", ""))
        payload = _read_yaml(logical_path, repo_root=repo_root)
        scope = _expected_mapping(payload.get("scope"), label=f"PLDM {component_id} scope")
        training = _expected_mapping(
            payload.get("training"), label=f"PLDM {component_id} training"
        )
        model_family = scope.get("model_family", training.get("model_family"))
        if (
            payload.get("completion_id") != specification.get("completion_id")
            or str(model_family or "").lower() != "pldm"
        ):
            raise ValueError(f"PLDM completion preregistration {component_id} is invalid")
        material[component_id] = _identity(logical_path, repo_root=repo_root)
    return material


def _aggregate_completion_material(
    specification: Mapping[str, Any], *, repo_root: Path
) -> dict[str, Any]:
    logical_path = str(specification.get("path", ""))
    if (
        logical_path != FINAL_PLDMS_AGGREGATE_FREEZE_PATH
        or specification.get("freeze_id") != FINAL_PLDMS_AGGREGATE_FREEZE_ID
        or FINAL_PLDMS_AGGREGATE_ID != FINAL_PLDMS_AGGREGATE_FREEZE_ID
    ):
        raise ValueError("final PLDM completion aggregate identity is invalid")

    # A hash link between an aggregate and a scoreboard is not enough: a
    # self-consistent hand-written pair could otherwise change the claim
    # scope, omit a failed formal result, or promote Development-only evidence.
    # The production aggregate validator rebuilds every frozen receipt and
    # compares all four local commit files byte-for-byte.  The aggregate module
    # does not import this reseal module, so this direct production import does
    # not create an import cycle.
    validation = validate_written_completion_aggregate_and_scoreboard(
        aggregate_config=(
            repo_root / FINAL_PLDMS_AGGREGATE_PREREGISTRATION_PATH
        ),
        repo_root=repo_root,
    )
    expected_validation = {
        "passed": True,
        "aggregate_id": FINAL_PLDMS_AGGREGATE_FREEZE_ID,
        "formal_reference_rows": 13,
        "formal_reference_rows_added": 2,
        "components_added": ["speed", "action_strength"],
        "development_only_components_not_added": [
            "contact_friction",
            "motion_damping",
        ],
        "speed_evidence_scope": "behavioral",
        "speed_training_attribution_claim": False,
        "action_strength_formal_row_included": True,
        "action_strength_ability_passed": False,
    }
    if (
        not isinstance(validation, Mapping)
        or any(validation.get(key) != value for key, value in expected_validation.items())
        or not isinstance(validation.get("local_outputs"), Mapping)
    ):
        raise ValueError("final PLDM aggregate semantic validation is incomplete")
    expected_local_outputs = {
        "aggregate_freeze": FINAL_PLDMS_AGGREGATE_FREEZE_PATH,
        "addendum_specification": (
            "artifacts/evaluation/contextworld_icl_suite_v2_release_addendum_v1/"
            "public_scoreboard_spec.json"
        ),
        "addendum_scoreboard": (
            "artifacts/evaluation/contextworld_icl_suite_v2_release_addendum_v1/"
            "public_scoreboard.json"
        ),
        "scoreboard_resolution_decision": (
            "configs/benchmark/contextworld_icl_suite_v2_scoreboard_addendum_"
            "decision_v1.json"
        ),
    }
    for name, expected_path in expected_local_outputs.items():
        identity = validation["local_outputs"].get(name)
        if (
            not isinstance(identity, Mapping)
            or identity.get("path") != expected_path
            or not isinstance(identity.get("sha256"), str)
            or len(identity["sha256"]) != 64
            or type(identity.get("size_bytes")) is not int
        ):
            raise ValueError(
                "final PLDM aggregate local-output validation is incomplete"
            )

    payload = _read_json(logical_path, repo_root=repo_root)
    results = _expected_mapping(payload.get("completion_results"), label="aggregate completion_results")
    if (
        payload.get("schema_version") != 1
        or payload.get("freeze_id") != specification.get("freeze_id")
        or payload.get("status") != specification.get("required_status")
        or payload.get("all_four_completion_outcomes_finalized") is not True
        or set(results) != set(PLDM_COMPLETION_COMPONENT_IDS)
    ):
        raise ValueError("final PLDM completion aggregate freeze is invalid")
    for component_id, result in results.items():
        result = _expected_mapping(result, label=f"aggregate completion {component_id}")
        if (
            result.get("completion_id") is None
            or result.get("finalized") is not True
            or not isinstance(result.get("outcome"), str)
            or not result["outcome"].strip()
        ):
            raise ValueError(f"aggregate PLDM completion {component_id} is incomplete")
    return _identity(logical_path, repo_root=repo_root)


def _validate_scoreboard_addendum_preregistration(
    specification: Mapping[str, Any],
    *,
    base_specification: Mapping[str, Any],
    base_scoreboard: Mapping[str, Any],
    addendum_specification_path: str,
    addendum_scoreboard_path: str,
    repo_root: Path,
) -> dict[str, Any]:
    logical_path = str(specification.get("path", ""))
    payload = _read_yaml(logical_path, repo_root=repo_root)
    addendum = _expected_mapping(payload.get("scoreboard_addendum"), label="scoreboard addendum")
    resolution = _expected_mapping(
        addendum.get("final_resolution_decision"), label="scoreboard resolution contract"
    )
    historical_base = _expected_mapping(
        addendum.get("historical_base_evidence"), label="scoreboard historical base"
    )
    output_namespace = _expected_mapping(
        addendum.get("additive_output_namespace"), label="scoreboard output namespace"
    )
    if (
        addendum.get("record_id") != specification.get("record_id")
        or addendum.get("status") != "pending_final_pldm_completion_aggregate"
        or addendum.get("base_formal_reference_rows") != 11
        or resolution.get("path")
        != "configs/benchmark/contextworld_icl_suite_v2_scoreboard_addendum_decision_v1.json"
        or resolution.get("decision_id")
        != "contextworld_icl_suite_v2_scoreboard_addendum_decision_v1"
        or historical_base.get("specification")
        != {
            key: base_specification.get(key)
            for key in ("path", "sha256", "size_bytes")
        }
        or historical_base.get("scoreboard")
        != {
            key: base_scoreboard.get(key)
            for key in ("path", "sha256", "size_bytes")
        }
        or output_namespace.get("specification") != addendum_specification_path
        or output_namespace.get("scoreboard") != addendum_scoreboard_path
    ):
        raise ValueError("scoreboard addendum preregistration is invalid")
    outcomes = _expected_mapping(addendum.get("allowed_outcomes"), label="scoreboard outcomes")
    if set(outcomes) != {
        "no_additive_scoreboard_extension",
        "additive_scoreboard_extension_authorized",
    }:
        raise ValueError("scoreboard addendum outcomes are incomplete")
    return _identity(logical_path, repo_root=repo_root)


def _scoreboard_resolution_material(
    specification: Mapping[str, Any],
    *,
    aggregate_identity: Mapping[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    base_specification = _expected_mapping(
        specification.get("historical_base_specification"),
        label="historical base specification",
    )
    base_scoreboard = _expected_mapping(
        specification.get("historical_base_scoreboard"),
        label="historical base scoreboard",
    )
    base_spec_path = str(base_specification.get("path", ""))
    base_scoreboard_path = str(base_scoreboard.get("path", ""))
    base_spec_identity = {
        key: base_specification.get(key) for key in ("path", "sha256", "size_bytes")
    }
    base_scoreboard_identity = {
        key: base_scoreboard.get(key) for key in ("path", "sha256", "size_bytes")
    }
    if (
        base_specification.get("formal_reference_rows") != 11
        or base_scoreboard.get("formal_reference_rows") != 11
    ):
        raise ValueError("historical base scoreboard row contract is invalid")
    _require_identity(
        _identity(base_spec_path, repo_root=repo_root),
        base_spec_identity,
        label="historical base specification",
    )
    _require_identity(
        _identity(base_scoreboard_path, repo_root=repo_root),
        base_scoreboard_identity,
        label="historical base scoreboard",
    )
    addendum_spec_path = str(specification.get("addendum_specification", ""))
    addendum_scoreboard_path = str(specification.get("addendum_scoreboard", ""))
    if (
        not addendum_spec_path.startswith(
            "artifacts/evaluation/contextworld_icl_suite_v2_release_addendum_v1/"
        )
        or not addendum_scoreboard_path.startswith(
            "artifacts/evaluation/contextworld_icl_suite_v2_release_addendum_v1/"
        )
    ):
        raise ValueError("scoreboard addendum output namespace is invalid")
    addendum_spec = _expected_mapping(
        specification.get("additive_resolution_preregistration"),
        label="scoreboard addendum preregistration specification",
    )
    decision_spec = _expected_mapping(
        specification.get("additive_resolution_decision"),
        label="scoreboard resolution decision specification",
    )
    addendum_identity = _validate_scoreboard_addendum_preregistration(
        addendum_spec,
        base_specification=base_specification,
        base_scoreboard=base_scoreboard,
        addendum_specification_path=addendum_spec_path,
        addendum_scoreboard_path=addendum_scoreboard_path,
        repo_root=repo_root,
    )
    base_spec_payload = _read_json(base_spec_path, repo_root=repo_root)
    base_scoreboard_payload = _read_json(base_scoreboard_path, repo_root=repo_root)
    base_spec_rows = base_spec_payload.get("components")
    base_scoreboard_rows = base_scoreboard_payload.get("component_results")
    if (
        not isinstance(base_spec_rows, list)
        or not isinstance(base_scoreboard_rows, list)
        or len(base_spec_rows) != 11
        or len(base_scoreboard_rows) != 11
    ):
        raise ValueError("historical base scoreboard does not contain eleven rows")
    spec_payload = _read_json(addendum_spec_path, repo_root=repo_root)
    scoreboard_payload = _read_json(addendum_scoreboard_path, repo_root=repo_root)
    spec_rows = spec_payload.get("components")
    scoreboard_rows = scoreboard_payload.get("component_results")
    if not isinstance(spec_rows, list) or not isinstance(scoreboard_rows, list):
        raise ValueError("final scoreboard/specification row lists are missing")
    try:
        expected_base_scoreboard = make_public_scoreboard_from_spec(
            base_spec_payload
        )
        expected_addendum_scoreboard = make_public_scoreboard_from_spec(
            spec_payload
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("scoreboard specification cannot be reproduced") from error
    if base_scoreboard_payload != expected_base_scoreboard:
        raise ValueError("historical base scoreboard cannot be reproduced")
    if scoreboard_payload != expected_addendum_scoreboard:
        raise ValueError("final addendum scoreboard cannot be reproduced")

    def scoreboard_key(row: Any, *, label: str) -> tuple[str, str]:
        if not isinstance(row, Mapping):
            raise ValueError(f"{label} row is invalid")
        component_id = row.get("component_id")
        method_name = row.get("method_name")
        if (
            not isinstance(component_id, str)
            or not component_id
            or not isinstance(method_name, str)
            or not method_name
        ):
            raise ValueError(f"{label} row identity is invalid")
        return component_id, method_name

    base_keys = [
        scoreboard_key(row, label="historical base scoreboard")
        for row in base_scoreboard_rows
    ]
    final_keys = [
        scoreboard_key(row, label="final addendum scoreboard")
        for row in scoreboard_rows
    ]
    if len(set(base_keys)) != len(base_keys) or len(set(final_keys)) != len(
        final_keys
    ):
        raise ValueError("scoreboard rows must have unique component/method keys")
    base_key_set = set(base_keys)
    historical_projection = [
        row
        for row in scoreboard_rows
        if scoreboard_key(row, label="final addendum scoreboard")
        in base_key_set
    ]

    resolution_path = str(decision_spec.get("path", ""))
    resolution = _read_json(resolution_path, repo_root=repo_root)
    expected_resolution_id = decision_spec.get("decision_id")
    if (
        resolution.get("schema_version") != 1
        or resolution.get("decision_id") != expected_resolution_id
        or resolution.get("passed") is not True
    ):
        raise ValueError("scoreboard resolution decision is invalid")
    formal_rows = resolution.get("formal_reference_rows")
    added_rows = resolution.get("formal_reference_rows_added")
    extension_authorized = resolution.get("scoreboard_extension_authorized")
    if (
        type(formal_rows) is not int
        or type(added_rows) is not int
        or type(extension_authorized) is not bool
        or len(spec_rows) != formal_rows
        or len(scoreboard_rows) != formal_rows
    ):
        raise ValueError("scoreboard resolution row count does not match final files")
    status = resolution.get("status")
    if status == "no_additive_scoreboard_extension":
        if (
            formal_rows != 11
            or added_rows != 0
            or extension_authorized
            or spec_rows != base_spec_rows
            or scoreboard_rows != base_scoreboard_rows
        ):
            raise ValueError("no-extension resolution must prove exactly eleven rows")
    elif status == "additive_scoreboard_extension_authorized":
        if (
            formal_rows <= 11
            or added_rows != formal_rows - 11
            or not extension_authorized
            or spec_rows[:11] != base_spec_rows
            or historical_projection != base_scoreboard_rows
        ):
            raise ValueError("additive scoreboard extension is not result-derived")
    else:
        raise ValueError("scoreboard resolution status is invalid")

    expected_specification = _identity(addendum_spec_path, repo_root=repo_root)
    expected_scoreboard = _identity(addendum_scoreboard_path, repo_root=repo_root)
    if (
        resolution.get("historical_base_specification") != base_spec_identity
        or resolution.get("historical_base_scoreboard") != base_scoreboard_identity
        or resolution.get("addendum_specification") != expected_specification
        or resolution.get("addendum_scoreboard") != expected_scoreboard
        or resolution.get("final_pldm_completion_aggregate_results_freeze")
        != dict(aggregate_identity)
    ):
        raise ValueError("scoreboard resolution does not bind current evidence")
    return {
        "historical_base_specification": base_spec_identity,
        "historical_base_scoreboard": base_scoreboard_identity,
        "addendum_specification": expected_specification,
        "addendum_scoreboard": expected_scoreboard,
        "additive_resolution_preregistration": addendum_identity,
        "additive_resolution_decision": _identity(resolution_path, repo_root=repo_root),
    }


def _current_results_overlay_material(
    specification: Mapping[str, Any], *, repo_root: Path
) -> dict[str, Any]:
    overlay_specification = _expected_mapping(
        specification,
        label="current-results overlay evidence",
        keys={"path", "overlay_id"},
    )
    if (
        overlay_specification.get("path") != CURRENT_RESULTS_OVERLAY_PATH
        or overlay_specification.get("overlay_id")
        != CURRENT_RESULTS_OVERLAY_ID
    ):
        raise ValueError("current-results overlay evidence is invalid")
    overlay = load_current_results_overlay_preregistration(
        CURRENT_RESULTS_OVERLAY_PATH,
        repo_root=repo_root,
    )
    return dict(overlay["_config_identity"])


def _current_materials(
    preregistration: Mapping[str, Any], *, repo_root: Path
) -> dict[str, Any]:
    evidence = _expected_mapping(
        preregistration.get("required_evidence"), label="required_evidence"
    )
    document_paths = _expected_mapping(
        evidence.get("final_public_documents"), label="final_public_documents"
    )
    component_paths = _expected_mapping(
        evidence.get("current_component_release_configs"),
        label="current_component_release_configs",
    )
    aggregate = _aggregate_completion_material(
        _expected_mapping(
            evidence.get("final_pldm_completion_aggregate_results_freeze"),
            label="final_pldm_completion_aggregate_results_freeze",
        ),
        repo_root=repo_root,
    )
    return {
        "engineering_identity_amendment": _engineering_material(
            _expected_mapping(
                evidence.get("engineering_identity_amendment"),
                label="engineering_identity_amendment",
            ),
            repo_root=repo_root,
        ),
        "capability_taxonomy_documentation_amendment": (
            _taxonomy_documentation_material(
                _expected_mapping(
                    evidence.get("capability_taxonomy_documentation_amendment"),
                    label="capability_taxonomy_documentation_amendment",
                ),
                document_paths,
                repo_root=repo_root,
            )
        ),
        "action_strength_float32_consistency_amendment": (
            _action_strength_float32_consistency_material(
                _expected_mapping(
                    evidence.get("action_strength_float32_consistency_amendment"),
                    label="action_strength_float32_consistency_amendment",
                ),
                repo_root=repo_root,
            )
        ),
        "original_baseline_archive_auditor": (
            _original_baseline_archive_auditor_material(
                _expected_mapping(
                    evidence.get("original_baseline_archive_auditor"),
                    label="original_baseline_archive_auditor",
                ),
                repo_root=repo_root,
            )
        ),
        "current_results_overlay": _current_results_overlay_material(
            _expected_mapping(
                evidence.get("current_results_overlay"),
                label="current_results_overlay",
            ),
            repo_root=repo_root,
        ),
        "current_component_release_configs": {
            component_id: _identity(str(path), repo_root=repo_root)
            for component_id, path in component_paths.items()
        },
        "registered_suite_sources": _registered_suite_source_material(
            _expected_mapping(
                evidence.get("registered_suite_sources"),
                label="registered_suite_sources",
            ),
            repo_root=repo_root,
        ),
        "original_baseline_result_freezes": _baseline_freeze_material(
            _expected_mapping(
                evidence.get("original_baseline_result_freezes"),
                label="original_baseline_result_freezes",
            ),
            repo_root=repo_root,
        ),
        "pldm_completion_preregistrations": _pldm_completion_material(
            _expected_mapping(
                evidence.get("pldm_completion_preregistrations"),
                label="pldm_completion_preregistrations",
            ),
            repo_root=repo_root,
        ),
        "final_pldm_completion_aggregate_results_freeze": aggregate,
        "public_scoreboard": _scoreboard_resolution_material(
            _expected_mapping(
                evidence.get("public_scoreboard"), label="public_scoreboard"
            ),
            aggregate_identity=aggregate,
            repo_root=repo_root,
        ),
        "reseal_authority_implementation": {
            logical_path: _identity(logical_path, repo_root=repo_root)
            for logical_path in RESEAL_AUTHORITY_IMPLEMENTATION_PATHS
        },
    }


def _declared_required_paths(preregistration: Mapping[str, Any]) -> set[str]:
    """Return every file path that check-only can report without freezing it."""

    evidence = _expected_mapping(
        preregistration.get("required_evidence"), label="required_evidence"
    )
    paths = set(RESEAL_AUTHORITY_IMPLEMENTATION_PATHS)
    paths.add(str(evidence["engineering_identity_amendment"]["path"]))
    paths.add(str(evidence["capability_taxonomy_documentation_amendment"]["path"]))
    action_strength_amendment = _expected_mapping(
        evidence["action_strength_float32_consistency_amendment"],
        label="action_strength_float32_consistency_amendment",
    )
    paths.update(
        str(action_strength_amendment[key])
        for key in ("path", "release_config", "scorer_path", "release_auditor_path")
    )
    archive_auditor = _expected_mapping(
        evidence["original_baseline_archive_auditor"],
        label="original_baseline_archive_auditor",
    )
    for key in ("module", "cli"):
        identity = _expected_mapping(
            archive_auditor[key], label=f"original baseline archive auditor {key}"
        )
        paths.add(str(identity["path"]))
    paths.add(str(evidence["current_results_overlay"]["path"]))
    paths.update(str(path) for path in evidence["final_public_documents"].values())
    paths.update(
        str(path) for path in evidence["current_component_release_configs"].values()
    )
    sources = _expected_mapping(
        evidence["registered_suite_sources"], label="registered_suite_sources"
    )
    paths.add(str(sources["source_manifest"]))
    paths.update(str(path) for path in sources["paths"])
    paths.update(
        str(specification["path"])
        for specification in evidence["original_baseline_result_freezes"].values()
    )
    paths.update(
        str(specification["path"])
        for specification in evidence["pldm_completion_preregistrations"].values()
    )
    paths.add(
        str(evidence["final_pldm_completion_aggregate_results_freeze"]["path"])
    )
    public_scoreboard = _expected_mapping(
        evidence["public_scoreboard"], label="public_scoreboard"
    )
    paths.add(str(public_scoreboard["historical_base_specification"]["path"]))
    paths.add(str(public_scoreboard["historical_base_scoreboard"]["path"]))
    paths.add(str(public_scoreboard["addendum_specification"]))
    paths.add(str(public_scoreboard["addendum_scoreboard"]))
    paths.add(str(public_scoreboard["additive_resolution_preregistration"]["path"]))
    paths.add(str(public_scoreboard["additive_resolution_decision"]["path"]))
    return paths


def audit_integrity_reseal_v2_readiness(
    *,
    reseal_config: Path | str = RESEAL_CONFIG,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Report blockers without persisting any current identity as a decision."""

    root = (repo_root or repository_root()).resolve()
    try:
        preregistration = load_integrity_reseal_v2_preregistration(
            reseal_config, repo_root=root
        )
    except (FileNotFoundError, ValueError) as error:
        blockers = (
            error.blockers if isinstance(error, ResealBlocked) else [str(error)]
        )
        return {
            "schema_version": 1,
            "reseal_id": RESEAL_ID,
            "status": "blocked_pending_final_identity_freeze",
            "ready": False,
            "blockers": blockers,
            "decision_created": False,
        }
    blockers = [
        f"missing required evidence: {logical_path}"
        for logical_path in sorted(_declared_required_paths(preregistration))
        if not resolve_contextworld_path(logical_path, repo_root=root).is_file()
    ]
    if not blockers:
        try:
            _current_materials(preregistration, repo_root=root)
        except (FileNotFoundError, ValueError) as error:
            message = str(error)
            if message and not any(message in blocker for blocker in blockers):
                blockers.append(message)
    if blockers:
        return {
            "schema_version": 1,
            "reseal_id": RESEAL_ID,
            "status": "blocked_pending_final_identity_freeze",
            "ready": False,
            "blockers": blockers,
            "decision_created": False,
        }
    return {
        "schema_version": 1,
        "reseal_id": RESEAL_ID,
        "status": "ready_for_explicit_one_use_decision_creation",
        "ready": True,
        "blockers": [],
        "decision_created": False,
    }


def build_integrity_reseal_v2_decision(
    *,
    reseal_config: Path | str = RESEAL_CONFIG,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Build a non-persisted decision only when every final material is exact."""

    root = (repo_root or repository_root()).resolve()
    try:
        preregistration = load_integrity_reseal_v2_preregistration(
            reseal_config, repo_root=root
        )
    except (FileNotFoundError, ValueError) as error:
        if isinstance(error, ResealBlocked):
            raise error
        raise ResealBlocked([str(error)]) from error
    try:
        materials = _current_materials(preregistration, repo_root=root)
    except (FileNotFoundError, ValueError) as error:
        if isinstance(error, ResealBlocked):
            raise error
        raise ResealBlocked([str(error)]) from error
    return {
        "schema_version": 1,
        "reseal_id": RESEAL_ID,
        "suite_release_id": "contextworld_icl_benchmark_suite_v2",
        "status": "integrity_reseal_passed",
        "passed": True,
        "reseal_config": _config_identity(
            Path(preregistration["_config_path"]), repo_root=root
        ),
        "historical_predecessor": preregistration["historical_predecessor"],
        "historical_v1_chain": preregistration["_validated_v1_historical_chain"],
        "release_materials": materials,
        "claims": {
            "reseal_is_additive": True,
            "historical_v1_predecessor_byte_identical": True,
            "historical_files_rewritten": False,
            "suite_default_or_activation_switched_by_this_record": False,
            "current_materials_exactly_bound": True,
            "scoreboard_resolution_is_bound_by_separate_record": True,
            "taxonomy_record_does_not_claim_prior_preregistration": True,
            "raw_dataset_or_checkpoint_mutation_authorized": False,
            "training_or_checkpoint_selection_authorized": False,
            "public_test_rerun_authorized": False,
        },
    }


def validate_integrity_reseal_v2_decision(
    decision: Mapping[str, Any],
    *,
    reseal_config: Path | str = RESEAL_CONFIG,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Reject a decision if a predecessor or any current material drifted."""

    expected = build_integrity_reseal_v2_decision(
        reseal_config=reseal_config, repo_root=repo_root
    )
    if not isinstance(decision, Mapping) or dict(decision) != expected:
        raise ValueError("Suite v2 integrity-reseal-v2 decision does not match current bytes")
    return {"passed": True, "reseal_id": RESEAL_ID}


def _canonical_decision_target(
    output: Path | str, *, repo_root: Path
) -> Path:
    """Resolve only the preregistered, local one-use commit-marker path."""

    candidate = Path(output).expanduser()
    if candidate.is_absolute():
        try:
            logical_path = candidate.relative_to(repo_root).as_posix()
        except ValueError as error:
            raise ValueError(
                "integrity reseal v2 decision must use its canonical "
                "repo-local output path"
            ) from error
    else:
        logical_path = candidate.as_posix()
    if logical_path != RESEAL_DECISION_PATH:
        raise ValueError(
            "integrity reseal v2 decision must use its preregistered "
            "canonical output path"
        )
    return resolve_no_symlink_contextworld_path(
        RESEAL_DECISION_PATH,
        repo_root=repo_root,
        label="Suite v2 integrity-reseal-v2 decision output",
        allow_missing=True,
    )


def write_integrity_reseal_v2_decision(
    output: Path | str,
    *,
    reseal_config: Path | str = RESEAL_CONFIG,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Write a one-use decision with exclusive creation; never overwrite it."""

    root = (repo_root or repository_root()).resolve()
    target = _canonical_decision_target(output, repo_root=root)
    if target.exists():
        raise FileExistsError(
            "Refusing to overwrite the integrity-reseal-v2 decision: "
            f"{target}"
        )
    decision = build_integrity_reseal_v2_decision(
        reseal_config=reseal_config, repo_root=root
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as stream:
        json.dump(decision, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    return decision
