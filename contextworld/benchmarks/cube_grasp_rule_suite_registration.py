from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import statistics
from typing import Any, Mapping

import yaml

from contextworld.benchmarks.cube_grasp_rule_icl_score import (
    cube_grasp_rule_prediction_gate,
)
from contextworld.paths import artifact_root, repository_root


REGISTRATION_ID = "contextworld_cube_gripper_carry_h3_v4r1_suite_registration_v1"
FREEZE_RECEIPT_ID = f"{REGISTRATION_ID}_freeze_v1"
COMPONENT_ID = "cube_gripper_carry"
RELEASE_ID = "contextworld_cube_gripper_carry_icl_history3_v4r1"
SUITE_RELEASE_ID = "contextworld_icl_benchmark_suite_v2"
EXPECTED_SEEDS = (17321, 17322, 17323)
EXPECTED_BASIS_KEYS = frozenset(
    {
        "recovery_preregistration",
        "recovery_freeze_receipt",
        "public_data_success",
        "development_build_report",
        "public_generation_started",
        "public_request",
        "public_build_report",
        "public_manifest",
        "public_matrix_request",
        "public_access_started",
        "public_seed17321",
        "public_seed17322",
        "public_seed17323",
        "public_score_success",
        "public_matrix_score",
        "public_release_decision",
        "original_task_retention_decision",
        "reference_development_decision",
    }
)
EXPECTED_SOURCE_TABLES = {
    "train": (2048, 16384),
    "loader_validation": (256, 2048),
    "validation": (256, 2048),
}
EXPECTED_PREREGISTRATION_KEYS = frozenset(
    {
        "schema_version",
        "registration_id",
        "component_id",
        "release_id",
        "suite_release_id",
        "status",
        "registered_date",
        "scope",
        "authorization_basis",
        "source_tables",
        "packaging_contract",
        "registration_gates",
        "planned_repository_outputs",
        "planned_artifacts",
        "allowed_claim_after_all_gates_pass",
        "prohibited_claims",
    }
)
EXPECTED_SCOPE = {
    "purpose": "package_the_completed_cube_v4r1_campaign_and_audit_suite_registration",
    "environment": "Cube",
    "capability": "infer_hidden_gripper_carry_rule_from_recent_interaction",
    "history_tokens": 3,
    "public_test_already_completed": True,
    "public_test_rerun_authorized": False,
    "training_or_checkpoint_selection_authorized": False,
    "threshold_or_recipe_change_authorized": False,
    "sealed_test_included": False,
}
EXPECTED_AUTHORIZATION_BASIS = {
    "recovery_preregistration": {
        "path": "configs/benchmark/cube_gripper_carry_h3_v4r1_public_recovery_prereg_v1.yaml",
        "sha256": "9432bda4aafccd5531c3bce0e0551ec611a653c6655102fa6091bc61ca88142c",
        "size_bytes": 20369,
    },
    "recovery_freeze_receipt": {
        "path": (
            "artifacts/evaluation/history3/"
            "cube_gripper_carry_h3_public_recovery_v1/"
            "public_recovery_freeze_receipt_v1.json"
        ),
        "sha256": "3bad917618995fc572e333a3ee51ee796eb5b6e368f31432d4ebdd947259dcea",
        "size_bytes": 2186262,
    },
    "public_data_success": {
        "path": (
            "artifacts/synthesis/"
            "cube_gripper_carry_rule_h3_public_v4r1_recovery_v1/_SUCCESS.json"
        ),
        "sha256": "0ad35a2a2e7337828d118de30d5a4ec140832aaa77c5b46e5882070bc00bbd31",
        "size_bytes": 4643,
    },
    "development_build_report": {
        "path": (
            "artifacts/synthesis/"
            "cube_gripper_carry_rule_h3_development_v4r1/build_report.json"
        ),
        "sha256": "98ddf562ec91a2e449ddceb288bceb7f3b765b47dd6f7ebe0b141cde51bd84bf",
        "size_bytes": 52453701,
    },
    "public_generation_started": {
        "path": (
            "artifacts/synthesis/"
            "cube_gripper_carry_rule_h3_public_v4r1_recovery_v1/"
            "_GENERATION_STARTED.json"
        ),
        "sha256": "9dba5df836f653e62d9f8ba8d4711a92a499e54f388a67b5f47cfc4ae1802da6",
        "size_bytes": 923,
    },
    "public_request": {
        "path": (
            "artifacts/synthesis/"
            "cube_gripper_carry_rule_h3_public_v4r1_recovery_v1/request.json"
        ),
        "sha256": "4289c6a906ac058407501743ca2584dcc47e4419d6a6ef0e3f9ec6730bf5f6ee",
        "size_bytes": 2087376,
    },
    "public_build_report": {
        "path": (
            "artifacts/synthesis/"
            "cube_gripper_carry_rule_h3_public_v4r1_recovery_v1/build_report.json"
        ),
        "sha256": "4e617418de641811e5e47713d49af589f07e0c704c423a41d81d635d6f8d5c30",
        "size_bytes": 7977056,
    },
    "public_manifest": {
        "path": (
            "artifacts/synthesis/"
            "cube_gripper_carry_rule_h3_public_v4r1_recovery_v1/manifest.json"
        ),
        "sha256": "79d3e4f915f723770d7ef487064608b446f26873f96c0ce46748ad89010c6313",
        "size_bytes": 1333,
    },
    "public_matrix_request": {
        "path": (
            "artifacts/evaluation/history3/"
            "cube_gripper_carry_h3_public_recovery_v1/public_score_v1/"
            "matrix_request.json"
        ),
        "sha256": "cd2380d038df699aa7980a637b910c99355fbf3e3f24478377096cb09dbe179a",
        "size_bytes": 4124,
    },
    "public_access_started": {
        "path": (
            "artifacts/evaluation/history3/"
            "cube_gripper_carry_h3_public_recovery_v1/public_score_v1/"
            "public_access_started.json"
        ),
        "sha256": "6619dcf5ecd0e903cb053b09a989a6822ec20ebf3914fd429ac4245637e30c08",
        "size_bytes": 537,
    },
    "public_seed17321": {
        "path": (
            "artifacts/evaluation/history3/"
            "cube_gripper_carry_h3_public_recovery_v1/public_score_v1/"
            "lewm_seed17321.json"
        ),
        "sha256": "bf4bd0dad8cfb899a44b838b2d6e25c4efc59012ac7921801c5cdfb8c046b000",
        "size_bytes": 229091,
    },
    "public_seed17322": {
        "path": (
            "artifacts/evaluation/history3/"
            "cube_gripper_carry_h3_public_recovery_v1/public_score_v1/"
            "lewm_seed17322.json"
        ),
        "sha256": "ffaa52be0045ad3d8e701b8904b623c511b4296f788e094f65c67106d561c7e8",
        "size_bytes": 229071,
    },
    "public_seed17323": {
        "path": (
            "artifacts/evaluation/history3/"
            "cube_gripper_carry_h3_public_recovery_v1/public_score_v1/"
            "lewm_seed17323.json"
        ),
        "sha256": "b536e4d8213edae2008868867e8315fa630c9eedbd85da7079a6329568882ac0",
        "size_bytes": 229049,
    },
    "public_score_success": {
        "path": (
            "artifacts/evaluation/history3/"
            "cube_gripper_carry_h3_public_recovery_v1/public_score_v1/"
            "_SUCCESS.json"
        ),
        "sha256": "ec43aae4b10f5134200901f8301e4b7cef0f85201c0a4f9eebeb46ba4c1587fb",
        "size_bytes": 1356,
    },
    "public_matrix_score": {
        "path": (
            "artifacts/evaluation/history3/"
            "cube_gripper_carry_h3_public_recovery_v1/public_score_v1/"
            "matrix_score.json"
        ),
        "sha256": "df7916d8ca943eb80d5fcef8c03b90e3587e35a3fa49fb98dc67cd7adc261bdc",
        "size_bytes": 767345,
    },
    "public_release_decision": {
        "path": (
            "artifacts/evaluation/history3/"
            "cube_gripper_carry_h3_public_recovery_v1/"
            "public_recovery_decision_v1.json"
        ),
        "sha256": "d868a25638a62a47601d19ace689e766580d67495e78908eefb3990491f2234a",
        "size_bytes": 4518,
    },
    "original_task_retention_decision": {
        "path": (
            "artifacts/evaluation/history3/"
            "cube_gripper_carry_h3_development_v4r1/"
            "original_task_retention_decision_v2.json"
        ),
        "sha256": "12dbe11eb4cf025359987962dfd869e73e0deb0ecb0eca007fad727889a07ef0",
        "size_bytes": 10564,
    },
    "reference_development_decision": {
        "path": (
            "artifacts/evaluation/history3/"
            "cube_gripper_carry_h3_development_v4r1/"
            "reference_development_decision_v3.json"
        ),
        "sha256": "797e5a9722435257fae55e1f9d97424cc77d2d3779576833322b84160375954f",
        "size_bytes": 4211,
    },
}
EXPECTED_SOURCE_TABLE_SPECIFICATIONS = {
    "train": {
        "source": "artifacts/synthesis/cube_gripper_carry_rule_h3_development_v4r1/train.lance",
        "pair_count": 2048,
        "files": 3,
        "bytes": 162330955,
        "sha256": "d1afff921ef7580ecb8a832514b59c6d2b000351ede7bc6e22517fd19fae0a45",
    },
    "loader_validation": {
        "source": (
            "artifacts/synthesis/"
            "cube_gripper_carry_rule_h3_development_v4r1/"
            "loader_validation.lance"
        ),
        "pair_count": 256,
        "files": 3,
        "bytes": 20354755,
        "sha256": "c51a2c74b5aa4163c5338fcf15fbf38dc2d6cda07800385ae487a60d9c2ce0d8",
    },
    "validation": {
        "source": (
            "artifacts/synthesis/"
            "cube_gripper_carry_rule_h3_public_v4r1_recovery_v1/"
            "validation.lance"
        ),
        "pair_count": 256,
        "files": 3,
        "bytes": 20544003,
        "sha256": "ba72017e5f47408b3cd351398a323e626f052ee1a9e252698f3c78efd550fb6f",
    },
}
EXPECTED_PACKAGING_CONTRACT = {
    "projection_root": "artifacts/synthesis/cube_gripper_carry_rule_h3_v4r1_release_projection_v1",
    "source_roots_are_immutable": True,
    "projection_contains": [
        "_PACKAGING_STARTED.json",
        "train.lance",
        "loader_validation.lance",
        "validation.lance",
        "portable_provenance.json",
        "_SUCCESS.json",
    ],
    "machine_specific_paths_allowed": False,
    "symlinks_allowed": False,
    "source_table_bytes_must_be_identical": True,
    "rerun_in_same_namespace_authorized": False,
}
EXPECTED_REGISTRATION_GATES = {
    "recovery_decision_status": "public_test_release_candidate_reference_passed",
    "public_checkpoints_passed": 3,
    "public_checkpoints_required": 3,
    "positive_reference_public_claim_required": True,
    "retention_status": "passed_retention",
    "retention_checkpoints_passed": 3,
    "causal_data_contract_required": True,
    "public_pair_count": 256,
    "action_template_counts": {
        "endpoint4": 64,
        "front_hold": 64,
        "plateau": 64,
        "ramp4": 64,
    },
    "all_cross_split_overlaps_required": 0,
    "suite_component_count": 9,
    "formal_scoreboard_rows": 11,
    "formal_scoreboard_components": 7,
    "full_component_audit_required": True,
    "full_suite_audit_required": True,
    "portable_copy_export_required": True,
    "exported_bundle_reaudit_required": True,
}
EXPECTED_PLANNED_REPOSITORY_OUTPUTS = {
    "release_config": "configs/benchmark/cube_gripper_carry_h3_v4r1_icl_release_v1.yaml",
    "release_data_api": "contextworld/benchmarks/cube_grasp_rule_v4r1_icl_data.py",
    "release_score_api": "contextworld/benchmarks/cube_grasp_rule_v4r1_icl_score.py",
    "release_cli": "contextworld/benchmarks/cube_grasp_rule_v4r1_icl_cli.py",
    "registration_contract": "contextworld/benchmarks/cube_grasp_rule_suite_registration.py",
    "suite_config": "configs/benchmark/contextworld_icl_suite_v2.yaml",
    "public_document": "docs/ContextWorld_ICL_Benchmark.md",
    "packaging_script": "scripts/package_cube_grasp_rule_h3_v4r1_icl_release.py",
    "registration_freezer": "scripts/freeze_cube_grasp_rule_h3_v4r1_suite_registration.py",
    "registration_finalizer": "scripts/finalize_cube_grasp_rule_h3_v4r1_suite_registration.py",
}
EXPECTED_PLANNED_ARTIFACTS = {
    "registration_freeze_receipt": (
        "artifacts/evaluation/history3/"
        "cube_gripper_carry_h3_v4r1_suite_registration_v1/"
        "registration_freeze_receipt_v1.json"
    ),
    "projection_root": EXPECTED_PACKAGING_CONTRACT["projection_root"],
    "component_audit": (
        "artifacts/evaluation/history3/"
        "cube_gripper_carry_h3_v4r1_suite_registration_v1/"
        "component_release_audit_v1.json"
    ),
    "suite_audit": (
        "artifacts/evaluation/history3/"
        "cube_gripper_carry_h3_v4r1_suite_registration_v1/"
        "suite_v2_audit_v1.json"
    ),
    "export_audit": (
        "artifacts/evaluation/history3/"
        "cube_gripper_carry_h3_v4r1_suite_registration_v1/"
        "suite_v2_export_audit_v1.json"
    ),
    "registration_decision": (
        "artifacts/evaluation/history3/"
        "cube_gripper_carry_h3_v4r1_suite_registration_v1/"
        "registration_decision_v1.json"
    ),
}
EXPECTED_ALLOWED_CLAIM = {
    "benchmark_component_status": "ready",
    "reference_result_status": "passed_public_test_3_of_3",
    "suite_membership": SUITE_RELEASE_ID,
    "distribution": "local_technical_release_candidate",
}
EXPECTED_PROHIBITED_CLAIMS = [
    "public_test_was_rerun_during_packaging",
    "pldm_was_publicly_evaluated_or_passed",
    "failed_public_v1_namespace_was_repaired",
    "suite_v1_was_rewritten_as_a_nine_component_release",
    "public_distribution_ready_without_license_and_download_configuration",
]
EXPECTED_PREREGISTRATION_LOGICAL_PATH = (
    "configs/benchmark/"
    "cube_gripper_carry_h3_v4r1_suite_registration_prereg_v1.yaml"
)
PORTABLE_TEXT_SUFFIXES = {
    ".csv",
    ".json",
    ".jsonl",
    ".log",
    ".md",
    ".txt",
    ".yaml",
    ".yml",
}
NON_PORTABLE_MARKERS = (
    "/opt/",
    "/tmp/",
    "/home/",
    "/root/",
    "../../data/",
    "\\Users\\",
)


def lexical_absolute(path: Path) -> Path:
    """Return an absolute path without dereferencing symlinks."""

    return Path(os.path.abspath(os.fspath(path.expanduser())))


def require_no_symlink_components(
    path: Path,
    *,
    anchor: Path,
    label: str,
    allow_missing: bool = False,
) -> Path:
    """Reject symlinks between a trusted anchor and a lexical target path."""

    target = lexical_absolute(path)
    trusted = lexical_absolute(anchor)
    try:
        relative = target.relative_to(trusted)
    except ValueError as error:
        raise RuntimeError(f"{label} escapes its trusted root: {target}") from error
    current = trusted
    missing = False
    for index, part in enumerate(relative.parts):
        current = current / part
        if missing or not os.path.lexists(current):
            if not allow_missing:
                raise FileNotFoundError(f"{label} is missing: {current}")
            missing = True
            continue
        metadata = os.lstat(current)
        if stat.S_ISLNK(metadata.st_mode):
            raise RuntimeError(f"{label} traverses a symlink: {current}")
        if index < len(relative.parts) - 1 and not stat.S_ISDIR(metadata.st_mode):
            raise RuntimeError(f"{label} traverses a non-directory: {current}")
    return target


def resolve_no_symlink_contextworld_path(
    value: str | Path,
    *,
    repo_root: Path | None = None,
    label: str,
    allow_missing: bool = False,
) -> Path:
    """Resolve stable repo/artifact references while preserving symlink evidence."""

    logical = Path(value).expanduser()
    if logical.is_absolute():
        raise RuntimeError(f"{label} must use a portable logical path")
    root = lexical_absolute(repo_root or repository_root())
    if logical.parts and logical.parts[0] == "artifacts":
        bundled = root / logical
        require_no_symlink_components(
            bundled,
            anchor=root,
            label=f"{label} bundled path",
            allow_missing=True,
        )
        if bundled.exists():
            return require_no_symlink_components(
                bundled,
                anchor=root,
                label=label,
                allow_missing=allow_missing,
            )
        external_root = lexical_absolute(artifact_root(root))
        return require_no_symlink_components(
            external_root.joinpath(*logical.parts[1:]),
            anchor=external_root,
            label=label,
            allow_missing=allow_missing,
        )
    return require_no_symlink_components(
        root / logical,
        anchor=root,
        label=label,
        allow_missing=allow_missing,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_regular_file(path: Path, *, label: str) -> None:
    metadata = os.lstat(path)
    if not stat.S_ISREG(metadata.st_mode) or path.is_symlink():
        raise RuntimeError(f"{label} is not a regular no-symlink file: {path}")


def require_regular_tree(path: Path, *, label: str) -> list[Path]:
    metadata = os.lstat(path)
    if not stat.S_ISDIR(metadata.st_mode) or path.is_symlink():
        raise RuntimeError(f"{label} is not a regular no-symlink directory: {path}")
    files: list[Path] = []
    for child in sorted(path.rglob("*")):
        child_metadata = os.lstat(child)
        if stat.S_ISLNK(child_metadata.st_mode):
            raise RuntimeError(f"{label} contains a symlink: {child}")
        if stat.S_ISREG(child_metadata.st_mode):
            files.append(child)
        elif not stat.S_ISDIR(child_metadata.st_mode):
            raise RuntimeError(f"{label} contains a special node: {child}")
    return files


def file_identity(path: Path, *, logical_path: str) -> dict[str, Any]:
    require_regular_file(path, label=logical_path)
    return {
        "path": logical_path,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def tree_identity(path: Path) -> dict[str, Any]:
    files = require_regular_tree(path, label="artifact tree")
    digest = hashlib.sha256()
    for child in files:
        digest.update(child.relative_to(path).as_posix().encode())
        digest.update(b"\0")
        digest.update(sha256_file(child).encode())
        digest.update(b"\0")
    return {
        "files": len(files),
        "bytes": sum(child.stat().st_size for child in files),
        "sha256": digest.hexdigest(),
    }


def identity_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return all(
        left.get(name) == right.get(name)
        for name in ("path", "sha256", "size_bytes")
    )


def exact_value_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(right, dict):
        return set(left) == set(right) and all(
            exact_value_equal(left[key], value) for key, value in right.items()
        )
    if isinstance(right, list):
        return len(left) == len(right) and all(
            exact_value_equal(observed, expected)
            for observed, expected in zip(left, right)
        )
    return left == right


def read_json(path: Path, *, label: str) -> dict[str, Any]:
    require_regular_file(path, label=label)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a JSON object")
    return value


def read_yaml(path: Path, *, label: str) -> dict[str, Any]:
    require_regular_file(path, label=label)
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a YAML object")
    return value


def _assert_identity_specification(value: Any, *, label: str) -> None:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"path", "sha256", "size_bytes"}
        or not str(value.get("path", ""))
        or len(str(value.get("sha256", ""))) != 64
        or int(value.get("size_bytes", -1)) <= 0
    ):
        raise RuntimeError(f"Invalid identity specification: {label}")


def validate_registration_preregistration_contract(
    prereg: Mapping[str, Any], *, preregistration_path: Path
) -> None:
    repo = lexical_absolute(repository_root())
    expected_preregistration = require_no_symlink_components(
        repo / EXPECTED_PREREGISTRATION_LOGICAL_PATH,
        anchor=repo,
        label="Cube Suite-registration preregistration",
    )
    if lexical_absolute(preregistration_path) != expected_preregistration:
        raise RuntimeError("Cube Suite-registration preregistration path drifted")
    if set(prereg) != EXPECTED_PREREGISTRATION_KEYS:
        raise RuntimeError("Cube Suite-registration top-level shape drifted")
    if not exact_value_equal(
        {
            key: prereg.get(key)
            for key in (
                "schema_version",
                "registration_id",
                "component_id",
                "release_id",
                "suite_release_id",
                "status",
                "registered_date",
            )
        },
        {
            "schema_version": 1,
            "registration_id": REGISTRATION_ID,
            "component_id": COMPONENT_ID,
            "release_id": RELEASE_ID,
            "suite_release_id": SUITE_RELEASE_ID,
            "status": "registered_not_frozen",
            "registered_date": "2026-08-14",
        },
    ):
        raise RuntimeError("Cube Suite-registration preregistration identity drifted")
    if not exact_value_equal(prereg.get("scope"), EXPECTED_SCOPE):
        raise RuntimeError("Cube Suite-registration scope drifted")
    basis = prereg.get("authorization_basis")
    if not exact_value_equal(basis, EXPECTED_AUTHORIZATION_BASIS):
        raise RuntimeError("Cube Suite-registration authorization basis drifted")
    tables = prereg.get("source_tables")
    if not exact_value_equal(tables, EXPECTED_SOURCE_TABLE_SPECIFICATIONS):
        raise RuntimeError("Cube Suite-registration source tables drifted")
    if not exact_value_equal(
        prereg.get("packaging_contract"), EXPECTED_PACKAGING_CONTRACT
    ):
        raise RuntimeError("Cube release projection contract drifted")
    if not exact_value_equal(
        prereg.get("registration_gates"), EXPECTED_REGISTRATION_GATES
    ):
        raise RuntimeError("Cube Suite-registration gates drifted")
    if not exact_value_equal(
        prereg.get("planned_repository_outputs"),
        EXPECTED_PLANNED_REPOSITORY_OUTPUTS,
    ):
        raise RuntimeError("Cube Suite-registration repository outputs drifted")
    if not exact_value_equal(
        prereg.get("planned_artifacts"), EXPECTED_PLANNED_ARTIFACTS
    ):
        raise RuntimeError("Cube Suite-registration planned artifacts drifted")
    if not exact_value_equal(
        prereg.get("allowed_claim_after_all_gates_pass"), EXPECTED_ALLOWED_CLAIM
    ):
        raise RuntimeError("Cube Suite-registration allowed claim drifted")
    if not exact_value_equal(
        prereg.get("prohibited_claims"), EXPECTED_PROHIBITED_CLAIMS
    ):
        raise RuntimeError("Cube Suite-registration prohibited claims drifted")


def _observed_basis(
    prereg: Mapping[str, Any], *, repo_root: Path
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    identities: dict[str, dict[str, Any]] = {}
    payloads: dict[str, dict[str, Any]] = {}
    for name, expected in prereg["authorization_basis"].items():
        logical = str(expected["path"])
        path = resolve_no_symlink_contextworld_path(
            logical,
            repo_root=repo_root,
            label=f"historical evidence {name}",
        )
        observed = file_identity(path, logical_path=logical)
        if not identity_equal(observed, expected):
            raise RuntimeError(f"Historical evidence drifted: {name}")
        identities[name] = observed
        payloads[name] = (
            read_yaml(path, label=name)
            if path.suffix in {".yaml", ".yml"}
            else read_json(path, label=name)
        )
    return identities, payloads


def _require_bound(
    observed: Mapping[str, Any], expected: Mapping[str, Any], *, label: str
) -> None:
    if not identity_equal(observed, expected):
        raise RuntimeError(f"Historical identity chain drifted: {label}")


def _validate_public_chain(
    identities: Mapping[str, Mapping[str, Any]],
    payloads: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    prereg = payloads["recovery_preregistration"]
    freeze = payloads["recovery_freeze_receipt"]
    data_success = payloads["public_data_success"]
    score_success = payloads["public_score_success"]
    matrix = payloads["public_matrix_score"]
    decision = payloads["public_release_decision"]
    retention = payloads["original_task_retention_decision"]

    _require_bound(
        identities["recovery_preregistration"],
        freeze.get("preregistration", {}),
        label="freeze.preregistration",
    )
    if (
        freeze.get("status") != "frozen_before_public_generation_or_access"
        or freeze.get("checks_passed") is not True
        or freeze.get("authorization", {}).get("suite_registration") is not False
        or freeze.get("authorization", {}).get("public_test_rerun_after_access")
        is not False
    ):
        raise RuntimeError("Recovery freeze authorization drifted")
    for key, basis_name in (
        ("preregistration", "recovery_preregistration"),
        ("freeze_receipt", "recovery_freeze_receipt"),
        ("generation_started", "public_generation_started"),
        ("request", "public_request"),
        ("build_report", "public_build_report"),
        ("manifest", "public_manifest"),
    ):
        _require_bound(
            identities[basis_name], data_success.get(key, {}), label=f"data_success.{key}"
        )
    if (
        data_success.get("status")
        != "public_data_generated_and_integrity_validated_not_model_read_or_scored"
        or data_success.get("public_test", {}).get("read_by_model") is not False
        or data_success.get("public_test", {}).get("scored") is not False
    ):
        raise RuntimeError("Public data success state drifted")
    expected_seed_identities = [
        identities[f"public_seed{seed}"] for seed in EXPECTED_SEEDS
    ]
    if len(score_success.get("checkpoint_results", [])) != 3:
        raise RuntimeError("Public score success lacks three results")
    for observed, expected in zip(
        expected_seed_identities, score_success["checkpoint_results"]
    ):
        _require_bound(observed, expected, label="score_success.checkpoint_result")
    _require_bound(
        identities["public_matrix_score"],
        score_success.get("matrix_score", {}),
        label="score_success.matrix_score",
    )
    if (
        score_success.get("status") != "completed_one_use_public_scoring"
        or score_success.get("rerun_authorized") is not False
        or score_success.get("public_test", {}).get(
            "used_for_training_or_selection"
        )
        is not False
    ):
        raise RuntimeError("Public score success state drifted")
    _require_bound(
        identities["public_matrix_request"],
        matrix.get("authorization", {}).get("matrix_request", {}),
        label="matrix.authorization.matrix_request",
    )
    _require_bound(
        identities["public_access_started"],
        matrix.get("authorization", {}).get("public_access_started", {}),
        label="matrix.authorization.public_access_started",
    )
    matrix_rows = matrix.get("checkpoint_results")
    if not isinstance(matrix_rows, list) or len(matrix_rows) != 3:
        raise RuntimeError("Public matrix must contain three checkpoint results")
    expected_checkpoints = {
        int(row["training_seed"]): row for row in prereg["public_evaluation"]["checkpoints"]
    }
    aggregates: dict[str, list[float]] = {
        name: []
        for name in (
            "correct_future_rate",
            "correct_history_rate",
            "context_switch_rate",
            "worst_rule_correct_future_rate",
            "other_minus_correct_mse_margin_mean",
            "joint_icl_pair_success_rate",
        )
    }
    per_seed: dict[str, Any] = {}
    for index, (row, seed) in enumerate(zip(matrix_rows, EXPECTED_SEEDS)):
        raw = payloads[f"public_seed{seed}"]
        if row != raw:
            raise RuntimeError(f"Matrix result does not equal seed receipt: {seed}")
        model = row.get("model", {})
        expected = expected_checkpoints.get(seed)
        if expected is None or (
            int(model.get("training_seed", -1)) != seed
            or model.get("family") != "lewm"
            or model.get("training_recipe") != expected.get("training_recipe")
            or model.get("checkpoint_sha256") != expected.get("sha256")
            or int(model.get("checkpoint_size_bytes", -1))
            != int(expected.get("size_bytes", -2))
            or model.get("state_sha256_before")
            != expected.get("model_state_sha256")
            or model.get("state_sha256_after")
            != expected.get("model_state_sha256")
        ):
            raise RuntimeError(f"Public checkpoint identity drifted: {seed}")
        metrics = row.get("metrics")
        if not isinstance(metrics, dict):
            raise RuntimeError(f"Public metrics are missing: {seed}")
        gate = cube_grasp_rule_prediction_gate(metrics, release=dict(prereg))
        if row.get("gate") != gate or gate.get("passed") is not True:
            raise RuntimeError(f"Public gate failed to reproduce: {seed}")
        for name in aggregates:
            aggregates[name].append(float(metrics[name]))
        per_seed[str(seed)] = {
            "checkpoint_sha256": expected["sha256"],
            "checkpoint_size_bytes": int(expected["size_bytes"]),
            "model_state_sha256": expected["model_state_sha256"],
            "metrics": metrics,
            "gate": gate,
        }
    recomputed_aggregate = {
        name: {
            "mean": float(statistics.mean(values)),
            "minimum": float(min(values)),
            "maximum": float(max(values)),
        }
        for name, values in aggregates.items()
    }
    if matrix.get("aggregate") != recomputed_aggregate or (
        matrix.get("status") != "completed_one_use_public_scoring"
        or matrix.get("model_family") != "lewm"
        or matrix.get("training_seeds") != list(EXPECTED_SEEDS)
        or matrix.get("checkpoints_passed") != 3
        or matrix.get("checkpoints_required") != 3
        or matrix.get("passed") is not True
        or matrix.get("public_test", {}).get("used_for_training_or_selection")
        is not False
    ):
        raise RuntimeError("Public matrix aggregate/state drifted")

    chain = decision.get("authorization_chain", {})
    for key, basis_name in (
        ("development_decision", "reference_development_decision"),
        ("freeze_receipt", "recovery_freeze_receipt"),
        ("matrix_score", "public_matrix_score"),
        ("preregistration", "recovery_preregistration"),
        ("public_data_success", "public_data_success"),
        ("public_score_success", "public_score_success"),
        ("retention_decision", "original_task_retention_decision"),
    ):
        _require_bound(identities[basis_name], chain.get(key, {}), label=f"decision.{key}")
    claims = decision.get("claims", {})
    public_decision = decision.get("public_evaluation", {})
    if (
        decision.get("status")
        != "public_test_release_candidate_reference_passed"
        or public_decision.get("passed") is not True
        or public_decision.get("checkpoints_passed") != 3
        or public_decision.get("checkpoints_required") != 3
        or claims.get("positive_reference_public_claim_allowed") is not True
        or claims.get("local_data_and_scoring_release_packaging_allowed")
        is not True
        or claims.get("suite_registration_allowed") is not False
        or claims.get("public_test_rerun_allowed") is not False
        or decision.get("public_test", {}).get("used_for_training_or_selection")
        is not False
    ):
        raise RuntimeError("Public release decision drifted")

    comparisons = retention.get("comparisons")
    if (
        retention.get("status") != "passed_retention"
        or not isinstance(comparisons, list)
        or len(comparisons) != 3
    ):
        raise RuntimeError("Cube retention decision drifted")
    retention_summary = []
    for row in comparisons:
        seed = int(row.get("training_seed", -1))
        expected = expected_checkpoints.get(seed)
        if expected is None or (
            row.get("checkpoint_sha256") != expected["sha256"]
            or row.get("model_family") != "lewm"
            or row.get("passed") is not True
            or int(row.get("baseline_successes", -1)) != 198
            or int(row.get("evaluation_count", -1)) != 300
            or int(row.get("noninferiority_margin_successes", -1)) != 15
            or int(row.get("candidate_successes", -1))
            not in {183, 185, 186}
        ):
            raise RuntimeError(f"Cube retention comparison drifted: {seed}")
        retention_summary.append(
            {
                "training_seed": seed,
                "checkpoint_sha256": row["checkpoint_sha256"],
                "baseline_successes": int(row["baseline_successes"]),
                "candidate_successes": int(row["candidate_successes"]),
                "evaluation_count": int(row["evaluation_count"]),
                "noninferiority_margin_successes": int(
                    row["noninferiority_margin_successes"]
                ),
                "success_delta": int(row["success_delta"]),
                "passed": True,
            }
        )
    retention_summary.sort(key=lambda row: row["training_seed"])
    if [row["training_seed"] for row in retention_summary] != list(EXPECTED_SEEDS):
        raise RuntimeError("Cube retention seed set drifted")
    return {
        "public_reference": {
            "status": matrix["status"],
            "model_family": "lewm",
            "training_recipe": matrix["training_recipe"],
            "training_seeds": list(EXPECTED_SEEDS),
            "checkpoints_passed": 3,
            "checkpoints_required": 3,
            "passed": True,
            "aggregate": recomputed_aggregate,
            "checkpoint_results": per_seed,
            "public_test": matrix["public_test"],
            "recovery_decision": {
                "status": decision["status"],
                "claims": decision["claims"],
                "public_evaluation": decision["public_evaluation"],
            },
        },
        "original_task_retention": {
            "status": retention["status"],
            "baseline_success_count": 198,
            "comparisons": retention_summary,
            "passed": True,
        },
    }


def _validate_data_contract(
    payloads: Mapping[str, Mapping[str, Any]],
    source_table_specifications: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    development = payloads["development_build_report"]
    public = payloads["public_build_report"]
    development_splits = development.get("splits", {})
    public_split = public.get("splits", {}).get("validation", {})
    public_causal = public.get("causal_data_contract", {})
    if (
        development.get("passed") is not True
        or set(development_splits) != {"train", "loader_validation"}
        or public.get("passed") is not True
        or public.get("pair_count") != 256
        or public_split.get("passed") is not True
        or development.get("causal_data_contract", {}).get("passed") is not True
        or development.get("fresh_simulator_replay", {}).get("passed") is not True
        or public_causal.get("passed") is not True
        or public_causal.get("all_pairs_passed") is not True
        or public_causal.get("fresh_simulator_replay_passed") is not True
        or public_split.get("fresh_simulator_replay", {}).get("passed") is not True
    ):
        raise RuntimeError("Cube causal data reports did not both pass")
    report_splits = {
        "train": development_splits["train"],
        "loader_validation": development_splits["loader_validation"],
        "validation": public_split,
    }
    table_report_bindings: dict[str, dict[str, Any]] = {}
    for split_name, report_split in report_splits.items():
        specification = source_table_specifications[split_name]
        expected_report_identity = {
            "split": split_name,
            "pair_count": specification["pair_count"],
            "table_path": f"{split_name}.lance",
            "table_files": specification["files"],
            "table_bytes": specification["bytes"],
            "table_sha256": specification["sha256"],
        }
        observed_report_identity = {
            key: report_split.get(key) for key in expected_report_identity
        }
        if not exact_value_equal(
            observed_report_identity, expected_report_identity
        ):
            raise RuntimeError(
                f"Cube source table is not bound to its build report: {split_name}"
            )
        table_report_bindings[split_name] = expected_report_identity
    for split, count in (
        (development_splits["train"], 512),
        (development_splits["loader_validation"], 64),
        (public_split, 64),
    ):
        if split.get("action_anchor_counts") != {
            "endpoint4": count,
            "front_hold": count,
            "plateau": count,
            "ramp4": count,
        }:
            raise RuntimeError("Cube four-template pair balance drifted")
        extrema = split.get("profile_constraint_extrema", {})
        if any(
            extrema.get(name, {}).get(bound) != 0.0
            for name in ("probe_sum", "probe_final_z")
            for bound in ("minimum", "maximum")
        ):
            raise RuntimeError("Cube action profile constraints drifted")
        if (
            int(split.get("maximum_state_installations_after_x0", -1)) != 0
            or float(split.get("maximum_query_simulator_state_gap", 1.0)) > 1e-12
            or float(split.get("maximum_prequery_object_state_residual", 1.0))
            > 1e-12
            or int(split.get("minimum_history_changed_rgb_values", 0)) <= 0
            or int(split.get("minimum_future_changed_rgb_values", 0)) <= 0
        ):
            raise RuntimeError("Cube shared-query causal invariants drifted")
    development_isolation = development.get("cross_split_audit", {})
    if development_isolation.get("passed") is not True or any(
        int(development_isolation.get(name, {}).get("count", -1)) != 0
        for name in (
            "exact_action_profile_id_overlap",
            "pair_content_hash_overlap",
            "query_pixel_hash_overlap",
            "scene_template_content_hash_overlap",
            "source_episode_overlap",
        )
    ):
        raise RuntimeError("Cube Development split isolation drifted")
    public_isolation = public.get("cross_split_isolation", {})
    if set(public_isolation) != {
        "source_episode_overlap_with_all_prior_content",
        "action_profile_overlap_with_all_prior_content",
        "scene_template_overlap_with_all_prior_content",
        "pair_content_overlap_with_all_prior_content",
        "query_pixel_overlap_with_all_prior_content",
    } or any(int(value) != 0 for value in public_isolation.values()):
        raise RuntimeError("Cube Public split isolation drifted")

    def split_summary(row: Mapping[str, Any]) -> dict[str, Any]:
        keys = (
            "pair_count",
            "passed",
            "action_anchor_counts",
            "maximum_query_physical_gap",
            "maximum_query_simulator_state_gap",
            "maximum_prequery_object_state_residual",
            "maximum_state_installations_after_x0",
            "minimum_history_changed_rgb_values",
            "minimum_future_changed_rgb_values",
            "minimum_history_cube_height_gap_m",
            "minimum_future_cube_height_gap_m",
            "profile_constraint_extrema",
        )
        return {key: row[key] for key in keys if key in row}

    return {
        "development_passed": True,
        "public_passed": True,
        "development_splits": {
            name: split_summary(development_splits[name])
            for name in ("train", "loader_validation")
        },
        "public_split": split_summary(public_split),
        "development_isolation": development_isolation,
        "public_isolation": public_isolation,
        "source_table_report_bindings": table_report_bindings,
        "causal_data_contract": {
            "development_passed": True,
            "public_passed": True,
            "passed": True,
        },
    }


def validate_historical_evidence(
    prereg: Mapping[str, Any], *, repo_root: Path | None = None
) -> dict[str, Any]:
    root = lexical_absolute(repo_root or repository_root())
    identities, payloads = _observed_basis(prereg, repo_root=root)
    chain = _validate_public_chain(identities, payloads)
    data_contract = _validate_data_contract(payloads, prereg["source_tables"])
    source_tables: dict[str, dict[str, Any]] = {}
    for split, (_, expected_rows) in EXPECTED_SOURCE_TABLES.items():
        specification = prereg["source_tables"][split]
        logical = str(specification["source"])
        path = resolve_no_symlink_contextworld_path(
            logical,
            repo_root=root,
            label=f"source table {split}",
        )
        observed = tree_identity(path)
        expected = {
            key: specification[key] for key in ("files", "bytes", "sha256")
        }
        if observed != expected:
            raise RuntimeError(f"Cube source table identity drifted: {split}")
        import lance

        rows = int(lance.dataset(path).count_rows())
        if rows != expected_rows:
            raise RuntimeError(f"Cube source table row count drifted: {split}")
        source_tables[split] = {
            "path": logical,
            **observed,
            "rows": rows,
            "pair_count": int(specification["pair_count"]),
        }
    return {
        "authorization_basis": identities,
        "source_tables": source_tables,
        "data_contract": data_contract,
        **chain,
    }


def assert_portable_tree(path: Path) -> None:
    files = require_regular_tree(path, label="portable projection")
    violations: list[str] = []
    for child in files:
        if child.suffix.lower() not in PORTABLE_TEXT_SUFFIXES:
            continue
        try:
            text = child.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        markers = [marker for marker in NON_PORTABLE_MARKERS if marker in text]
        if markers:
            violations.append(
                f"{child.relative_to(path)} contains {', '.join(markers)}"
            )
    if violations:
        raise RuntimeError(
            "Cube portable projection contains machine paths:\n- "
            + "\n- ".join(violations)
        )


__all__ = [
    "COMPONENT_ID",
    "EXPECTED_BASIS_KEYS",
    "EXPECTED_AUTHORIZATION_BASIS",
    "EXPECTED_PACKAGING_CONTRACT",
    "EXPECTED_PLANNED_ARTIFACTS",
    "EXPECTED_PLANNED_REPOSITORY_OUTPUTS",
    "EXPECTED_PROHIBITED_CLAIMS",
    "EXPECTED_REGISTRATION_GATES",
    "EXPECTED_SCOPE",
    "EXPECTED_SEEDS",
    "EXPECTED_SOURCE_TABLES",
    "EXPECTED_SOURCE_TABLE_SPECIFICATIONS",
    "FREEZE_RECEIPT_ID",
    "NON_PORTABLE_MARKERS",
    "REGISTRATION_ID",
    "RELEASE_ID",
    "SUITE_RELEASE_ID",
    "assert_portable_tree",
    "file_identity",
    "identity_equal",
    "lexical_absolute",
    "read_json",
    "read_yaml",
    "require_regular_tree",
    "require_no_symlink_components",
    "resolve_no_symlink_contextworld_path",
    "tree_identity",
    "validate_historical_evidence",
    "validate_registration_preregistration_contract",
]
