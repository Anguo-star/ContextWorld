from __future__ import annotations

import copy
import hashlib
import os
from pathlib import Path
import stat
from typing import Any, Mapping

from contextworld.benchmarks.cube_grasp_rule_public_contract import (
    EXPECTED_AUTHORIZATION_BASIS_KEYS,
    EXPECTED_IMPLEMENTATION_KEYS,
    FREEZE_STATUS,
    PREREGISTRATION_ID as FAILED_PREREGISTRATION_ID,
    PREREGISTRATION_STATUS,
    PROTOCOL_ID,
    PublicAuthorization,
    _identity_matches,
    _resolve_identity_path,
    _validate_frozen_identity_group,
    file_identity,
    read_json_nofollow,
    read_yaml_nofollow,
    validate_public_freeze_receipt_contract,
    validate_public_preregistration_contract,
)
from contextworld.paths import repository_root, resolve_contextworld_path


PREREGISTRATION_ID = "contextworld_cube_gripper_carry_h3_v4r1_public_recovery_v1"
RECOVERY_AUTHORIZATION_ID = (
    "cube_gripper_carry_rule_history3_v4r1_public_recovery_v1"
)
FREEZE_RECEIPT_ID = (
    "contextworld_cube_gripper_carry_h3_v4r1_public_recovery_freeze_v1"
)
DECISION_ID = (
    "contextworld_cube_gripper_carry_h3_v4r1_public_recovery_decision_v1"
)
FAILED_FREEZE_RECEIPT_ID = (
    "contextworld_cube_gripper_carry_h3_v4r1_public_release_freeze_v1"
)

DEFAULT_PREREGISTRATION = repository_root() / (
    "configs/benchmark/"
    "cube_gripper_carry_h3_v4r1_public_recovery_prereg_v1.yaml"
)
DEFAULT_FREEZE_RECEIPT = resolve_contextworld_path(
    "artifacts/evaluation/history3/"
    "cube_gripper_carry_h3_public_recovery_v1/"
    "public_recovery_freeze_receipt_v1.json"
)

EXPECTED_PLANNED_ARTIFACTS = {
    "freeze_receipt": (
        "artifacts/evaluation/history3/"
        "cube_gripper_carry_h3_public_recovery_v1/"
        "public_recovery_freeze_receipt_v1.json"
    ),
    "public_data_root": (
        "artifacts/synthesis/"
        "cube_gripper_carry_rule_h3_public_v4r1_recovery_v1"
    ),
    "public_score_root": (
        "artifacts/evaluation/history3/"
        "cube_gripper_carry_h3_public_recovery_v1/public_score_v1"
    ),
    "public_release_decision": (
        "artifacts/evaluation/history3/"
        "cube_gripper_carry_h3_public_recovery_v1/"
        "public_recovery_decision_v1.json"
    ),
}

EXPECTED_RECOVERY_IMPLEMENTATION_KEYS = frozenset(
    {
        "recovery_contract",
        "recovery_builder",
        "recovery_matrix_runner",
        "recovery_freezer",
        "recovery_finalizer",
        "recovery_protocol",
        "recovery_contract_tests",
        "generation_failure_record",
    }
)
EXPECTED_RECOVERY_LINEAGE_KEYS = frozenset(
    {
        "failed_public_preregistration",
        "failed_public_freeze_receipt",
        "failed_generation_started",
        "failed_generation_failure",
    }
)
EXPECTED_RECOVERY_LINEAGE_PATHS = {
    "failed_public_preregistration": (
        "configs/benchmark/"
        "cube_gripper_carry_h3_v4r1_public_release_prereg_v1.yaml"
    ),
    "failed_public_freeze_receipt": (
        "artifacts/evaluation/history3/"
        "cube_gripper_carry_h3_public_release_v1/"
        "public_release_freeze_receipt_v1.json"
    ),
    "failed_generation_started": (
        "artifacts/synthesis/cube_gripper_carry_rule_h3_public_v4r1/"
        "_GENERATION_STARTED.json"
    ),
    "failed_generation_failure": (
        "artifacts/synthesis/cube_gripper_carry_rule_h3_public_v4r1/"
        "_GENERATION_FAILURE.json"
    ),
}


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _require_exact_keys(
    value: Mapping[str, Any], expected: set[str] | frozenset[str], *, label: str
) -> None:
    if set(value) != set(expected):
        raise RuntimeError(
            f"{label} key set mismatch: "
            f"missing={sorted(set(expected) - set(value))}, "
            f"extra={sorted(set(value) - set(expected))}"
        )


def _identity_core(value: Mapping[str, Any]) -> dict[str, Any]:
    return {name: value.get(name) for name in ("path", "sha256", "size_bytes")}


def _identity_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return _identity_core(left) == _identity_core(right)


def _assert_absent(path: Path, *, label: str) -> None:
    try:
        os.lstat(path)
    except FileNotFoundError:
        return
    raise FileExistsError(f"{label} already exists: {path}")


def validate_recovery_preregistration_contract(prereg: Mapping[str, Any]) -> None:
    if (
        prereg.get("schema_version") != 1
        or prereg.get("preregistration_id") != PREREGISTRATION_ID
        or prereg.get("recovery_authorization_id")
        != RECOVERY_AUTHORIZATION_ID
        or prereg.get("protocol_id") != PROTOCOL_ID
        or prereg.get("status") != PREREGISTRATION_STATUS
        or prereg.get("phase") != "public_generation_and_evaluation_only"
    ):
        raise RuntimeError("Cube Public recovery preregistration identity drifted")
    validate_public_preregistration_contract(prereg)

    recovery = _mapping(prereg.get("recovery"), label="recovery")
    expected_recovery = {
        "failed_preregistration_id": FAILED_PREREGISTRATION_ID,
        "failed_freeze_receipt_id": FAILED_FREEZE_RECEIPT_ID,
        "failed_generation_status": (
            "public_generation_failed_namespace_consumed_no_rerun"
        ),
        "failed_error_type": "KeyError",
        "failed_error_message": "'preregistration'",
        "failed_namespace_consumed": True,
        "failed_public_model_read": False,
        "failed_public_scored": False,
        "scientific_protocol_unchanged": True,
    }
    if dict(recovery) != expected_recovery:
        raise RuntimeError("Cube Public recovery lineage declaration drifted")

    identity = _mapping(prereg.get("identity"), label="identity")
    if identity.get("preregistration_path") != str(
        DEFAULT_PREREGISTRATION.relative_to(repository_root())
    ):
        raise RuntimeError("Cube Public recovery preregistration path drifted")
    implementations = _mapping(
        identity.get("implementation"), label="identity.implementation"
    )
    recovery_implementations = _mapping(
        identity.get("recovery_implementation"),
        label="identity.recovery_implementation",
    )
    _require_exact_keys(
        implementations,
        EXPECTED_IMPLEMENTATION_KEYS,
        label="identity.implementation",
    )
    _require_exact_keys(
        recovery_implementations,
        EXPECTED_RECOVERY_IMPLEMENTATION_KEYS,
        label="identity.recovery_implementation",
    )

    basis = _mapping(prereg.get("authorization_basis"), label="authorization_basis")
    _require_exact_keys(
        basis,
        EXPECTED_AUTHORIZATION_BASIS_KEYS,
        label="authorization_basis",
    )
    lineage = _mapping(prereg.get("recovery_lineage"), label="recovery_lineage")
    _require_exact_keys(
        lineage,
        EXPECTED_RECOVERY_LINEAGE_KEYS,
        label="recovery_lineage",
    )
    for name, expected_path in EXPECTED_RECOVERY_LINEAGE_PATHS.items():
        entry = _mapping(lineage[name], label=f"recovery_lineage.{name}")
        if (
            set(entry) != {"path", "sha256", "size_bytes"}
            or entry.get("path") != expected_path
            or not isinstance(entry.get("sha256"), str)
            or len(str(entry.get("sha256"))) != 64
            or int(entry.get("size_bytes", -1)) <= 0
        ):
            raise RuntimeError(f"Cube Public recovery lineage identity drifted: {name}")

    planned = _mapping(prereg.get("planned_artifacts"), label="planned_artifacts")
    if dict(planned) != EXPECTED_PLANNED_ARTIFACTS:
        raise RuntimeError("Cube Public recovery artifact paths drifted")


def validate_recovery_lineage(
    prereg: Mapping[str, Any], *, root: Path | None = None
) -> dict[str, dict[str, Any]]:
    root = root or repository_root()
    lineage = _mapping(prereg.get("recovery_lineage"), label="recovery_lineage")
    observed: dict[str, dict[str, Any]] = {}
    payloads: dict[str, dict[str, Any]] = {}
    for name, expected_path in EXPECTED_RECOVERY_LINEAGE_PATHS.items():
        entry = _mapping(lineage[name], label=f"recovery_lineage.{name}")
        path = _resolve_identity_path(expected_path, root=root)
        identity = file_identity(path, logical_path=expected_path)
        if not _identity_equal(identity, entry):
            raise RuntimeError(f"Cube Public recovery lineage changed: {name}")
        observed[name] = identity
        if path.suffix == ".yaml":
            _, payloads[name] = read_yaml_nofollow(path, label=name)
        else:
            _, payloads[name] = read_json_nofollow(path, label=name)

    failed_prereg = payloads["failed_public_preregistration"]
    failed_freeze = payloads["failed_public_freeze_receipt"]
    started = payloads["failed_generation_started"]
    failure = payloads["failed_generation_failure"]
    if (
        failed_prereg.get("preregistration_id") != FAILED_PREREGISTRATION_ID
        or failed_prereg.get("protocol_id") != PROTOCOL_ID
        or failed_freeze.get("receipt_id") != FAILED_FREEZE_RECEIPT_ID
        or failed_freeze.get("preregistration_id") != FAILED_PREREGISTRATION_ID
        or failed_freeze.get("protocol_id") != PROTOCOL_ID
        or failed_freeze.get("status") != FREEZE_STATUS
        or failed_freeze.get("checks_passed") is not True
        or not _identity_equal(
            failed_freeze.get("preregistration", {}),
            observed["failed_public_preregistration"],
        )
    ):
        raise RuntimeError("Cube failed Public preregistration/freeze chain drifted")
    if (
        started.get("schema_version") != 1
        or started.get("protocol_id") != PROTOCOL_ID
        or started.get("status")
        != "public_generation_attempt_started_one_use_namespace_reserved"
        or int(started.get("generation_attempt", -1)) != 1
        or started.get("output")
        != failed_prereg["planned_artifacts"]["public_data_root"]
        or started.get("public_table_opened") is not False
        or started.get("public_model_read") is not False
        or started.get("rerun_authorized") is not False
        or not _identity_equal(
            started.get("preregistration", {}),
            observed["failed_public_preregistration"],
        )
        or not _identity_equal(
            started.get("freeze_receipt", {}),
            observed["failed_public_freeze_receipt"],
        )
    ):
        raise RuntimeError("Cube failed Public generation-start chain drifted")
    if (
        failure.get("schema_version") != 1
        or failure.get("protocol_id") != PROTOCOL_ID
        or failure.get("status")
        != "public_generation_failed_namespace_consumed_no_rerun"
        or failure.get("error_type") != "KeyError"
        or failure.get("error_message") != "'preregistration'"
        or failure.get("public_model_read") is not False
        or failure.get("public_scored") is not False
        or failure.get("rerun_authorized") is not False
        or failure.get("next_step")
        != "archive and freeze a distinct recovery authorization"
        or not _identity_equal(
            failure.get("generation_started", {}),
            observed["failed_generation_started"],
        )
    ):
        raise RuntimeError("Cube failed Public generation-failure chain drifted")

    failed_root = _resolve_identity_path(
        failed_prereg["planned_artifacts"]["public_data_root"], root=root
    )
    metadata = os.lstat(failed_root)
    if not stat.S_ISDIR(metadata.st_mode) or failed_root.is_symlink():
        raise RuntimeError("Cube failed Public root is missing or aliased")
    children = sorted(path.name for path in failed_root.iterdir())
    if children != ["_GENERATION_FAILURE.json", "_GENERATION_STARTED.json"]:
        raise RuntimeError("Cube failed Public root contains unexpected content")
    _assert_absent(
        _resolve_identity_path(
            failed_prereg["planned_artifacts"]["public_score_root"], root=root
        ),
        label="failed Public score root",
    )
    _assert_absent(
        _resolve_identity_path(
            failed_prereg["planned_artifacts"]["public_release_decision"],
            root=root,
        ),
        label="failed Public decision",
    )
    return observed


def validate_recovery_freeze_receipt_contract(
    *, prereg: Mapping[str, Any], freeze: Mapping[str, Any], root: Path | None = None
) -> None:
    root = root or repository_root()
    validate_recovery_preregistration_contract(prereg)
    expected_top_level = {
        "schema_version",
        "receipt_id",
        "receipt_path",
        "preregistration_id",
        "recovery_authorization_id",
        "protocol_id",
        "status",
        "frozen_at_utc",
        "checks_passed",
        "preregistration",
        "implementation_identities",
        "recovery_implementation_identities",
        "frozen_inputs",
        "runtime",
        "public_exclusions",
        "recovery_lineage",
        "authorization",
        "public_test",
        "planned_artifacts",
    }
    _require_exact_keys(freeze, expected_top_level, label="recovery freeze receipt")
    if (
        freeze.get("schema_version") != 1
        or freeze.get("receipt_id") != FREEZE_RECEIPT_ID
        or freeze.get("preregistration_id") != PREREGISTRATION_ID
        or freeze.get("recovery_authorization_id")
        != RECOVERY_AUTHORIZATION_ID
        or freeze.get("protocol_id") != PROTOCOL_ID
        or freeze.get("status") != FREEZE_STATUS
        or freeze.get("checks_passed") is not True
        or freeze.get("receipt_path")
        != EXPECTED_PLANNED_ARTIFACTS["freeze_receipt"]
        or freeze.get("planned_artifacts") != EXPECTED_PLANNED_ARTIFACTS
    ):
        raise RuntimeError("Cube Public recovery freeze identity/status drifted")

    base_prereg = copy.deepcopy(dict(prereg))
    base_prereg["preregistration_id"] = FAILED_PREREGISTRATION_ID
    base_freeze = copy.deepcopy(dict(freeze))
    base_freeze.pop("recovery_authorization_id")
    base_freeze.pop("recovery_implementation_identities")
    base_freeze.pop("recovery_lineage")
    base_freeze["receipt_id"] = FAILED_FREEZE_RECEIPT_ID
    base_freeze["preregistration_id"] = FAILED_PREREGISTRATION_ID
    validate_public_freeze_receipt_contract(
        prereg=base_prereg,
        freeze=base_freeze,
        root=root,
    )

    expected_recovery_implementations = _mapping(
        prereg["identity"].get("recovery_implementation"),
        label="identity.recovery_implementation",
    )
    observed_recovery_implementations = _mapping(
        freeze.get("recovery_implementation_identities"),
        label="recovery_implementation_identities",
    )
    _require_exact_keys(
        observed_recovery_implementations,
        EXPECTED_RECOVERY_IMPLEMENTATION_KEYS,
        label="recovery_implementation_identities",
    )
    for name in EXPECTED_RECOVERY_IMPLEMENTATION_KEYS:
        if not _identity_equal(
            _mapping(observed_recovery_implementations[name], label=name),
            _mapping(expected_recovery_implementations[name], label=name),
        ):
            raise RuntimeError(f"recovery implementation binding drifted: {name}")
    if freeze.get("recovery_lineage") != prereg.get("recovery_lineage"):
        raise RuntimeError("Cube Public recovery lineage receipt binding drifted")


def load_public_recovery_authorization(
    *,
    preregistration_path: Path = DEFAULT_PREREGISTRATION,
    freeze_receipt_path: Path = DEFAULT_FREEZE_RECEIPT,
    require_public_absent: bool = False,
    validate_implementation_identities: bool = True,
    validate_frozen_inputs: bool = True,
) -> PublicAuthorization:
    root = repository_root()
    preregistration_path = preregistration_path.expanduser().absolute()
    freeze_receipt_path = freeze_receipt_path.expanduser().absolute()
    prereg_raw, prereg = read_yaml_nofollow(
        preregistration_path, label="Cube Public recovery preregistration"
    )
    freeze_raw, freeze = read_json_nofollow(
        freeze_receipt_path, label="Cube Public recovery freeze receipt"
    )
    validate_recovery_preregistration_contract(prereg)
    validate_recovery_freeze_receipt_contract(
        prereg=prereg, freeze=freeze, root=root
    )

    declared_prereg_path = str(prereg["identity"]["preregistration_path"])
    if _resolve_identity_path(declared_prereg_path, root=root) != preregistration_path:
        raise RuntimeError("Cube Public recovery preregistration resolved path drifted")
    observed_prereg = {
        "path": declared_prereg_path,
        "sha256": hashlib.sha256(prereg_raw).hexdigest(),
        "size_bytes": len(prereg_raw),
    }
    _identity_matches(
        observed_prereg,
        freeze.get("preregistration", {}),
        label="Cube Public recovery preregistration",
    )
    freeze_logical_path = str(freeze.get("receipt_path", ""))
    if _resolve_identity_path(freeze_logical_path, root=root) != freeze_receipt_path:
        raise RuntimeError("Cube Public recovery freeze resolved path drifted")
    freeze_identity = {
        "path": freeze_logical_path,
        "sha256": hashlib.sha256(freeze_raw).hexdigest(),
        "size_bytes": len(freeze_raw),
    }

    if validate_implementation_identities:
        _validate_frozen_identity_group(
            freeze.get("implementation_identities"),
            label="implementation_identities",
            root=root,
            rehash=True,
        )
        _validate_frozen_identity_group(
            freeze.get("recovery_implementation_identities"),
            label="recovery_implementation_identities",
            root=root,
            rehash=True,
        )
    if validate_frozen_inputs:
        _validate_frozen_identity_group(
            freeze.get("frozen_inputs"),
            label="frozen_inputs",
            root=root,
            rehash=False,
        )
        validate_recovery_lineage(prereg, root=root)

    public = _mapping(freeze.get("public_test"), label="public_test")
    if public != {
        "access_status": "authorized_not_generated_not_opened_not_read_not_scored",
        "generated": False,
        "opened": False,
        "read": False,
        "hashed": False,
        "scored": False,
    }:
        raise RuntimeError("Cube Public recovery freeze has invalid pre-access state")

    result = PublicAuthorization(
        preregistration_path=preregistration_path,
        freeze_receipt_path=freeze_receipt_path,
        preregistration=dict(prereg),
        freeze_receipt=dict(freeze),
        freeze_receipt_identity=freeze_identity,
    )
    if require_public_absent:
        for label, path in (
            ("public_data_root", result.public_root),
            ("public_score_root", result.score_root),
            ("public_release_decision", result.decision_path),
        ):
            _assert_absent(path, label=f"one-use recovery {label}")
    return result


__all__ = [
    "DECISION_ID",
    "DEFAULT_FREEZE_RECEIPT",
    "DEFAULT_PREREGISTRATION",
    "EXPECTED_PLANNED_ARTIFACTS",
    "EXPECTED_RECOVERY_IMPLEMENTATION_KEYS",
    "EXPECTED_RECOVERY_LINEAGE_KEYS",
    "FREEZE_RECEIPT_ID",
    "PREREGISTRATION_ID",
    "RECOVERY_AUTHORIZATION_ID",
    "load_public_recovery_authorization",
    "validate_recovery_freeze_receipt_contract",
    "validate_recovery_lineage",
    "validate_recovery_preregistration_contract",
]
