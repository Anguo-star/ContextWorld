from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
from typing import Iterator

from contextworld.benchmarks.cube_grasp_rule_suite_registration import (
    EXPECTED_PREREGISTRATION_LOGICAL_PATH,
    FREEZE_RECEIPT_ID,
    REGISTRATION_ID,
    file_identity,
    identity_equal,
    lexical_absolute,
    read_json,
    read_yaml,
    require_no_symlink_components,
    resolve_no_symlink_contextworld_path,
    validate_registration_preregistration_contract,
)
from contextworld.benchmarks.cube_grasp_rule_v4r1_icl_data import (
    DEFAULT_CUBE_GRASP_RULE_V4R1_RELEASE_CONFIG,
    audit_cube_grasp_rule_v4r1_icl_release,
    load_cube_grasp_rule_v4r1_icl_release,
    recompute_cube_grasp_rule_v4r1_public_reference,
)
from contextworld.benchmarks.public_score import make_public_scoreboard_from_spec
from contextworld.benchmarks.suite_data import (
    COMPONENT_IDS,
    DEFAULT_SUITE_RELEASE_CONFIG,
    DEFAULT_SUITE_V2_RELEASE_CONFIG,
    SUITE_RELEASE_ID,
    SUITE_V2_COMPONENT_IDS,
    _audit_icl_suite_release_for_registration,
    _export_icl_suite_artifacts_for_registration,
    load_icl_suite_release,
)
from contextworld.paths import repository_root, resolve_contextworld_path
from scripts.build_contextworld_icl_suite_v2_scoreboard import (
    canonical_cube_scoreboard_spec_row,
)


ROOT = repository_root()
DEFAULT_PREREGISTRATION = ROOT / (
    EXPECTED_PREREGISTRATION_LOGICAL_PATH
)
DEFAULT_FREEZE_RECEIPT = resolve_contextworld_path(
    "artifacts/evaluation/history3/"
    "cube_gripper_carry_h3_v4r1_suite_registration_v1/"
    "registration_freeze_receipt_v1.json",
    repo_root=ROOT,
)
DEFAULT_EXPORT = resolve_contextworld_path(
    "artifacts/evaluation/history3/"
    "cube_gripper_carry_h3_v4r1_suite_registration_v1/"
    "suite_v2_copy_export_v1",
    repo_root=ROOT,
)
DEFAULT_EXPORT_STAGING = DEFAULT_EXPORT.with_name(
    f".{DEFAULT_EXPORT.name}.staging"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")


def _canonical_repository_file(
    path: Path, *, logical_path: str, label: str
) -> Path:
    expected = require_no_symlink_components(
        ROOT / logical_path,
        anchor=ROOT,
        label=label,
    )
    if lexical_absolute(path) != expected:
        raise RuntimeError(f"{label} must use the canonical path: {expected}")
    return expected


def _canonical_artifact_path(
    path: Path,
    *,
    logical_path: str,
    label: str,
    allow_missing: bool,
) -> Path:
    expected = resolve_no_symlink_contextworld_path(
        logical_path,
        repo_root=ROOT,
        label=label,
        allow_missing=allow_missing,
    )
    if lexical_absolute(path) != expected:
        raise RuntimeError(f"{label} must use the canonical path: {expected}")
    return expected


def _rewrite_path_prefix(value: object, *, source: Path, target: Path) -> object:
    """Rewrite reports after a same-filesystem atomic directory commit."""

    if isinstance(value, dict):
        return {
            key: _rewrite_path_prefix(item, source=source, target=target)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _rewrite_path_prefix(item, source=source, target=target)
            for item in value
        ]
    if isinstance(value, str):
        source_text = str(source)
        if value == source_text:
            return str(target)
        prefix = source_text + os.sep
        if value.startswith(prefix):
            return str(target) + value[len(source_text) :]
    return value


@contextmanager
def _artifact_root(path: Path) -> Iterator[None]:
    name = "CONTEXTWORLD_ARTIFACT_ROOT"
    previous = os.environ.get(name)
    os.environ[name] = str(path)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


def _validate_scoreboard(
    *, suite: dict, reference: dict, repo_root: Path
) -> dict:
    spec_info = suite["public_results"]["specification"]
    scoreboard_info = suite["public_results"]["scoreboard"]
    spec_path = resolve_contextworld_path(spec_info["path"], repo_root=repo_root)
    scoreboard_path = resolve_contextworld_path(
        scoreboard_info["path"], repo_root=repo_root
    )
    spec = read_json(spec_path, label="Suite v2 scoreboard specification")
    scoreboard = read_json(scoreboard_path, label="Suite v2 scoreboard")
    if (
        _sha256(spec_path) != spec_info["sha256"]
        or _sha256(scoreboard_path) != scoreboard_info["sha256"]
        or make_public_scoreboard_from_spec(spec) != scoreboard
        or len(spec.get("components", [])) != 11
        or len(scoreboard.get("component_results", [])) != 11
    ):
        raise RuntimeError("Suite v2 scoreboard identity/reproduction drifted")
    cube_spec_rows = [
        row
        for row in spec["components"]
        if row.get("component_id") == "cube_gripper_carry"
    ]
    cube_rows = [
        row
        for row in scoreboard["component_results"]
        if row.get("component_id") == "cube_gripper_carry"
    ]
    expected_cube = canonical_cube_scoreboard_spec_row(reference)
    if (
        cube_spec_rows != [expected_cube]
        or len(cube_rows) != 1
        or cube_rows[0].get("icl_ability", {}).get("result") != "PASS"
        or cube_rows[0].get("icl_ability", {}).get("evidence_scope")
        != "behavioral"
        or "pldm" in cube_rows[0].get("method_name", "").lower()
        or any(
            key in cube_rows[0]
            for key in (
                "submission_kind",
                "claim_boundary",
                "external_result",
                "formal_scoreboard_eligible",
            )
        )
    ):
        raise RuntimeError(
            "Suite v2 Cube row is not the canonical LeWM-only frozen reference"
        )
    return {
        "specification": file_identity(
            spec_path, logical_path=spec_info["path"]
        ),
        "scoreboard": file_identity(
            scoreboard_path, logical_path=scoreboard_info["path"]
        ),
        "formal_reference_rows": 11,
        "formal_reference_components": 7,
        "cube_rows": 1,
        "cube_family": "lewm",
        "external_results_included": False,
        "passed": True,
    }


def finalize_registration(
    *,
    preregistration: Path = DEFAULT_PREREGISTRATION,
    freeze_receipt: Path = DEFAULT_FREEZE_RECEIPT,
    cube_release_config: Path = DEFAULT_CUBE_GRASP_RULE_V4R1_RELEASE_CONFIG,
    suite_config: Path = DEFAULT_SUITE_V2_RELEASE_CONFIG,
    export_destination: Path = DEFAULT_EXPORT,
) -> dict:
    preregistration = _canonical_repository_file(
        preregistration,
        logical_path=EXPECTED_PREREGISTRATION_LOGICAL_PATH,
        label="Cube Suite-registration preregistration",
    )
    freeze_logical = (
        "artifacts/evaluation/history3/"
        "cube_gripper_carry_h3_v4r1_suite_registration_v1/"
        "registration_freeze_receipt_v1.json"
    )
    freeze_receipt = _canonical_artifact_path(
        freeze_receipt,
        logical_path=freeze_logical,
        label="Cube Suite-registration freeze receipt",
        allow_missing=False,
    )
    cube_release_logical = (
        "configs/benchmark/cube_gripper_carry_h3_v4r1_icl_release_v1.yaml"
    )
    cube_release_config = _canonical_repository_file(
        cube_release_config,
        logical_path=cube_release_logical,
        label="Cube v4r1 release config",
    )
    suite_v2_logical = "configs/benchmark/contextworld_icl_suite_v2.yaml"
    suite_config = _canonical_repository_file(
        suite_config,
        logical_path=suite_v2_logical,
        label="ContextWorld Suite v2 config",
    )
    export_logical = (
        "artifacts/evaluation/history3/"
        "cube_gripper_carry_h3_v4r1_suite_registration_v1/"
        "suite_v2_copy_export_v1"
    )
    export_destination = _canonical_artifact_path(
        export_destination,
        logical_path=export_logical,
        label="Suite v2 copy export",
        allow_missing=True,
    )
    export_staging = require_no_symlink_components(
        DEFAULT_EXPORT_STAGING,
        anchor=DEFAULT_EXPORT_STAGING.parent,
        label="Suite v2 copy-export staging namespace",
        allow_missing=True,
    )
    prereg = read_yaml(preregistration, label="Cube Suite-registration preregistration")
    validate_registration_preregistration_contract(
        prereg, preregistration_path=preregistration
    )
    freeze = read_json(freeze_receipt, label="Cube registration freeze receipt")
    prereg_identity = file_identity(
        preregistration,
        logical_path="configs/benchmark/"
        "cube_gripper_carry_h3_v4r1_suite_registration_prereg_v1.yaml",
    )
    expected_authorization = {
        "projection_packaging_allowed": True,
        "public_test_rerun_allowed": False,
        "suite_membership_before_final_audit_allowed": False,
        "training_or_checkpoint_selection_allowed": False,
    }
    implementation_paths = {
        "registration_contract": prereg["planned_repository_outputs"][
            "registration_contract"
        ],
        "packaging_script": prereg["planned_repository_outputs"][
            "packaging_script"
        ],
        "registration_freezer": prereg["planned_repository_outputs"][
            "registration_freezer"
        ],
    }
    implementation_identities = {
        name: file_identity(
            require_no_symlink_components(
                ROOT / logical,
                anchor=ROOT,
                label=f"frozen registration implementation {name}",
            ),
            logical_path=logical,
        )
        for name, logical in implementation_paths.items()
    }
    if (
        freeze.get("schema_version") != 1
        or freeze.get("receipt_id") != FREEZE_RECEIPT_ID
        or freeze.get("receipt_path") != freeze_logical
        or freeze.get("registration_id") != REGISTRATION_ID
        or freeze.get("status") != "suite_registration_packaging_frozen"
        or freeze.get("checks_passed") is not True
        or not identity_equal(freeze.get("preregistration", {}), prereg_identity)
        or freeze.get("implementation_identities") != implementation_identities
        or freeze.get("authorization") != expected_authorization
        or freeze.get("planned_artifacts") != prereg["planned_artifacts"]
    ):
        raise RuntimeError("Cube Suite-registration freeze receipt drifted")

    outputs = {
        name: resolve_no_symlink_contextworld_path(
            logical,
            repo_root=ROOT,
            label=f"Cube registration output {name}",
            allow_missing=True,
        )
        for name, logical in prereg["planned_artifacts"].items()
        if name in {"component_audit", "suite_audit", "export_audit", "registration_decision"}
    }
    consumed = [
        path
        for path in [*outputs.values(), export_destination, export_staging]
        if path.exists()
    ]
    if consumed:
        raise FileExistsError(
            "Cube final registration namespace is already consumed; partial "
            "outputs do not grant membership and must not be deleted or "
            "retried in place. Create a new preregistration to recover: "
            + ", ".join(str(path) for path in consumed)
        )

    release = load_cube_grasp_rule_v4r1_icl_release(cube_release_config)
    reference = recompute_cube_grasp_rule_v4r1_public_reference(
        release, repo_root=ROOT, layout="source"
    )
    suite = load_icl_suite_release(suite_config)
    membership_authority = suite.get("membership_authority", {})
    if (
        suite["release_id"] != "contextworld_icl_benchmark_suite_v2"
        or tuple(suite["components"]) != SUITE_V2_COMPONENT_IDS
        or suite["components"]["cube_gripper_carry"]["release_id"]
        != release["release_id"]
        or suite["components"]["cube_gripper_carry"]["benchmark_component_status"]
        != "ready"
        or suite["components"]["cube_gripper_carry"]["reference_result_status"]
        != "passed_public_test_3_of_3"
        or membership_authority.get("config_alone_grants_membership") is not False
        or membership_authority.get("activation_condition")
        != "passed_registration_decision_v1"
        or membership_authority.get("decision_path")
        != prereg["planned_artifacts"]["registration_decision"]
        or membership_authority.get("decision_is_commit_marker") is not True
        or membership_authority.get("partial_outputs_grant_membership") is not False
        or membership_authority.get(
            "failed_finalization_requires_new_preregistration"
        )
        is not True
    ):
        raise RuntimeError("Suite v2 proposed Cube membership drifted")
    suite_v1_path = _canonical_repository_file(
        DEFAULT_SUITE_RELEASE_CONFIG,
        logical_path="configs/benchmark/contextworld_icl_suite_v1.yaml",
        label="ContextWorld Suite v1 historical config",
    )
    suite_v1 = load_icl_suite_release(suite_v1_path)
    if (
        suite_v1["release_id"] != SUITE_RELEASE_ID
        or tuple(suite_v1["components"]) != COMPONENT_IDS
        or suite_v1["public_results"]["formal_reference_rows"] != 10
        or "cube_gripper_carry" in suite_v1["components"]
    ):
        raise RuntimeError("Suite v1 historical 8-component/10-row semantics drifted")
    scoreboard = _validate_scoreboard(
        suite=suite, reference=reference, repo_root=ROOT
    )

    component_audit = audit_cube_grasp_rule_v4r1_icl_release(
        release_config=cube_release_config,
        repo_root=ROOT,
        full=True,
        layout="source",
    )
    if component_audit.get("passed") is not True:
        raise RuntimeError("Cube full source component audit failed")

    suite_audit = _audit_icl_suite_release_for_registration(
        release_config=suite_config,
        repo_root=ROOT,
        full=True,
    )
    if (
        suite_audit.get("passed") is not True
        or len(suite_audit.get("components", {})) != 9
        or suite_audit.get("public_results", {})
        .get("reproduction", {})
        .get("observed_reference_rows")
        != 11
    ):
        raise RuntimeError("Suite v2 full source audit failed")

    export = _export_icl_suite_artifacts_for_registration(
        export_staging,
        release_config=suite_config,
        repo_root=ROOT,
        mode="copy",
        include_upstream_original=False,
    )
    if (
        export.get("status") != "passed"
        or export.get("mode") != "copy"
        or export.get("components") != list(SUITE_V2_COMPONENT_IDS)
    ):
        raise RuntimeError("Suite v2 portable copy export failed")
    bundle_artifact_root = export_staging / "benchmark"
    with _artifact_root(bundle_artifact_root):
        bundle_audit = _audit_icl_suite_release_for_registration(
            release_config=export_staging / "benchmark/suite.yaml",
            repo_root=ROOT,
            full=True,
        )
    if bundle_audit.get("passed") is not True:
        raise RuntimeError("Exported Suite v2 bundle re-audit failed")
    os.replace(export_staging, export_destination)
    export = _rewrite_path_prefix(
        export, source=export_staging, target=export_destination
    )
    bundle_audit = _rewrite_path_prefix(
        bundle_audit, source=export_staging, target=export_destination
    )
    export_audit = {
        "schema_version": 1,
        "status": "passed",
        "copy_export": export,
        "bundle_reaudit": bundle_audit,
        "atomic_export_commit": {
            "same_filesystem_rename": True,
            "bundle_reaudit_completed_before_commit": True,
            "committed_destination": str(export_destination),
        },
        "portability_boundary": {
            "self_contained": False,
            "requires_contextworld_checkout": True,
            "external_upstream_dependencies_required": True,
            "upstream_not_redistributed": True,
            "public_distribution_ready": False,
            "registration_decision_is_commit_marker": True,
            "partial_outputs_grant_membership": False,
            "validation_scope": "declared_component_release_contracts",
            "non_cryptographic_upstream_checks": [
                "tworoom_lance_bytes",
                "reacher_h5_bytes",
                "reacher_lance_bytes",
            ],
            "cryptographic_upstream_checks": [
                "tworoom_h5_sha256_and_bytes",
                "reacher_lewm_checkpoint_sha256_bytes_and_config_sha256",
                "reacher_pldm_checkpoint_sha256_bytes_and_config_sha256",
            ],
            "resolved_dependency_checks_recorded_in": (
                "bundle_reaudit.components"
            ),
        },
        "cube_public_test_rerun": False,
        "cube_formal_checkpoint_opened": False,
        "non_cube_checkpoint_identity_audit": True,
        "external_results_included": False,
        "passed": True,
    }
    for payload in (component_audit, suite_audit, export_audit):
        json.dumps(payload, indent=2, sort_keys=True)
    _write_json(outputs["component_audit"], component_audit)
    _write_json(outputs["suite_audit"], suite_audit)
    _write_json(outputs["export_audit"], export_audit)

    evidence = {
        "preregistration": prereg_identity,
        "freeze_receipt": file_identity(
            freeze_receipt,
            logical_path=prereg["planned_artifacts"]["registration_freeze_receipt"],
        ),
        "cube_release_config": file_identity(
            cube_release_config,
            logical_path=prereg["planned_repository_outputs"]["release_config"],
        ),
        "suite_v2_config": file_identity(
            suite_config,
            logical_path=prereg["planned_repository_outputs"]["suite_config"],
        ),
        "suite_v1_historical_config": file_identity(
            suite_v1_path,
            logical_path="configs/benchmark/contextworld_icl_suite_v1.yaml",
        ),
        "component_audit": file_identity(
            outputs["component_audit"],
            logical_path=prereg["planned_artifacts"]["component_audit"],
        ),
        "suite_audit": file_identity(
            outputs["suite_audit"],
            logical_path=prereg["planned_artifacts"]["suite_audit"],
        ),
        "export_audit": file_identity(
            outputs["export_audit"],
            logical_path=prereg["planned_artifacts"]["export_audit"],
        ),
        **scoreboard,
    }
    decision = {
        "schema_version": 1,
        "registration_id": REGISTRATION_ID,
        "release_id": release["release_id"],
        "suite_release_id": suite["release_id"],
        "status": "suite_registration_passed",
        "evidence": evidence,
        "claims": {
            **prereg["allowed_claim_after_all_gates_pass"],
            "suite_membership_granted": True,
            "formal_reference_family": "lewm",
            "formal_reference_rows_for_cube": 1,
            "external_results_formal_scoreboard_eligible": False,
            "pldm_public_result_included": False,
            "cube_public_test_rerun": False,
            "cube_formal_checkpoint_opened": False,
            "failed_public_v1_namespace_written_by_finalizer": False,
            "self_contained_bundle": False,
            "requires_contextworld_checkout": True,
            "external_upstream_dependencies_required": True,
            "upstream_not_redistributed": True,
            "public_distribution_ready": False,
            "registration_decision_is_commit_marker": True,
            "partial_outputs_grant_membership": False,
        },
        "gate_summary": {
            "component_full_audit": True,
            "suite_v2_full_audit": True,
            "portable_copy_export": True,
            "exported_bundle_full_reaudit": True,
            "components": 9,
            "formal_scoreboard_rows": 11,
            "formal_scoreboard_components": 7,
            "cube_public_test_rerun": False,
            "cube_formal_checkpoint_opened": False,
            "non_cube_checkpoint_identity_audit": True,
            "suite_v1_components": 8,
            "suite_v1_formal_scoreboard_rows": 10,
            "suite_v1_cube_absent": True,
            "registration_decision_is_commit_marker": True,
            "partial_outputs_grant_membership": False,
        },
        "passed": True,
    }
    _write_json(outputs["registration_decision"], decision)
    return decision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration", type=Path, default=DEFAULT_PREREGISTRATION)
    parser.add_argument("--freeze-receipt", type=Path, default=DEFAULT_FREEZE_RECEIPT)
    parser.add_argument(
        "--cube-release-config",
        type=Path,
        default=DEFAULT_CUBE_GRASP_RULE_V4R1_RELEASE_CONFIG,
    )
    parser.add_argument("--suite-config", type=Path, default=DEFAULT_SUITE_V2_RELEASE_CONFIG)
    parser.add_argument("--export-destination", type=Path, default=DEFAULT_EXPORT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    decision = finalize_registration(
        preregistration=args.preregistration,
        freeze_receipt=args.freeze_receipt,
        cube_release_config=args.cube_release_config,
        suite_config=args.suite_config,
        export_destination=args.export_destination,
    )
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
