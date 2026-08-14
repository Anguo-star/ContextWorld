#!/usr/bin/env python3
"""Finalize the original Cube v4 attempt as an infrastructure failure.

This finalizer is deliberately archival.  It verifies the immutable original
v4 preregistration and freeze chain, the orphan Lance fragment, the failed
attempt receipt, and the separately reconstructed raw-query identities.  It
does not open Lance as a dataset, replay MuJoCo, run the RGB probe, invoke a
model, or inspect Public Test.  The only write is one x-exclusive Development
decision after every input and cross-receipt invariant has passed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import yaml


PROTOCOL = "cube_gripper_carry_rule_history3_development_v4"
PREREGISTRATION_ID = "contextworld_cube_gripper_carry_h3_development_v4"
DECISION_ID = "cube_gripper_carry_h3_v4_infrastructure_failure_v1"
DECISION_STATUS = "failed_development"
FAILURE_STAGE = "formal_build_lance_train_commit_atomic_rename"
CLASSIFICATION = "infrastructure_failure_not_scientific_gate_failure"
OUTPUT_NAME = "development_decision_infrastructure_failure_v1.json"

PREREG_STATUS = "preregistered_before_first_v4_build"
FREEZE_STATUS = "frozen_before_first_v4_build"
FAILED_RECEIPT_STATUS = "infrastructure_failed_immutable_attempt"
QUERY_RECEIPT_STATUS = "failed_attempt_content_frozen_for_future_prior_exclusion"
FAILED_RECEIPT_ID = "cube_gripper_carry_h3_v4_failed_formal_attempt_v1"
QUERY_RECEIPT_ID = "cube_gripper_carry_h3_v4_failed_attempt_query_reconstruction_v1"

EXPECTED_PREREG_SHA256 = "f8f940bd01c0dfbc7c822e8c5885e517ba6ec2ccffda64801655f45aa847761f"
EXPECTED_PREREG_SIZE_BYTES = 28_730
EXPECTED_FREEZE_SHA256 = "a58549ec9d5856345d4fea72ca7a7690a74204e54062ec909080c336b77af837"
EXPECTED_FREEZE_SIZE_BYTES = 9_765
EXPECTED_PRIOR_SHA256 = "8c181529c3012cf89ecf8390d595093d256449d909c5e911297f78ed997161b4"
EXPECTED_PRIOR_SIZE_BYTES = 736_689
EXPECTED_FAILURE_RECEIPT_SHA256 = "5f20da08a538f2fd0c72c5c172e64cb2359a2e5bdad1746cf2c4249bbf739936"
EXPECTED_FAILURE_RECEIPT_SIZE_BYTES = 1_956_930
EXPECTED_QUERY_RECEIPT_SHA256 = "a85c3343464cbbea5c13ac167d419c87bbd5b8ce942767900af171db8474e5e0"
EXPECTED_QUERY_RECEIPT_SIZE_BYTES = 2_215_188

EXPECTED_BUILDER_SHA256 = "b1ac55103f66754149466c75ef51dd6f5676497e9c92afb04137e5dc3df433df"
EXPECTED_BUILDER_SIZE_BYTES = 95_698
EXPECTED_PHYSICS_SHA256 = "886a9a3147b7f6b29db70c6cb85b017befb5209fccdc12aeb6147d2aca9b829b"
EXPECTED_PHYSICS_SIZE_BYTES = 22_268
EXPECTED_PROBE_SHA256 = "66408cfbee0314fbdac62f807b19e8e9a2b5215f8bd47d4cc9d0332e341d96a5"
EXPECTED_PROBE_SIZE_BYTES = 46_673
EXPECTED_PROBE_TESTS_SHA256 = "d5179b34ae3853ef8298348d7788694b00e4bd2bc195694d7ca836419fe3d0dc"
EXPECTED_PROBE_TESTS_SIZE_BYTES = 22_269
EXPECTED_ACTION_SUPPORT_SHA256 = "b5ef1658036589f32a256194c1bac1247a2d86d230d5888a68ae910dc913ecb8"
EXPECTED_ACTION_SUPPORT_SIZE_BYTES = 38_302
EXPECTED_ACTION_SUPPORT_TESTS_SHA256 = "d7d5362b3bf2558e23ec67f1d797334f8f34ad4de586b479de88573877dc3395"
EXPECTED_ACTION_SUPPORT_TESTS_SIZE_BYTES = 21_195

EXPECTED_REQUEST_SHA256 = "711cdf5ecf52d9f93366c65d7f3f276eafe9e88570f0de8c6f5a2cacff05b328"
EXPECTED_REQUEST_SIZE_BYTES = 10_562
EXPECTED_FRAGMENT_NAME = "1101011000111110110110016388234b8b874d495c9eaf2528.lance"
EXPECTED_FRAGMENT_SHA256 = "15f4a5c423ba13d803a1b44f684b9f1916f6b899352f5b2ed623906cad59a920"
EXPECTED_FRAGMENT_SIZE_BYTES = 162_695_360

EXPECTED_PAIR_COUNT = 2_048
EXPECTED_EPISODE_COUNT = 4_096
EXPECTED_ROW_COUNT = 16_384
EXPECTED_CATALOG_START = 1_000_000
EXPECTED_CATALOG_STOP = 1_002_048
EXPECTED_ANCHOR_COUNTS = {
    "endpoint4": 512,
    "front_hold": 512,
    "plateau": 512,
    "ramp4": 512,
}
EXPECTED_LOGICAL_FAILED_ROOT = "artifacts/synthesis/cube_gripper_carry_rule_h3_development_v4"

SOURCE_SYMBOL = "upstream_cube_single_expert_h5"
SOURCE_SHA256 = "0664d507c4ff12009010644c9ae950836f954e700c172ccf22e7423af1a55625"
SOURCE_SIZE_BYTES = 101_942_558_720
SOURCE_ROW_COUNT = 2_010_000
SOURCE_EPISODE_COUNT = 10_000

CONTENT_FIELDS = (
    "action_profile_ids",
    "scene_template_content_hashes",
    "pair_content_hashes",
    "query_pixel_hashes",
)
ALL_EXCLUSION_FIELDS = ("source_episodes", *CONTENT_FIELDS)
SHA256_RE = re.compile(r"[0-9a-f]{64}")
FORENSIC_JPEG_NAMESPACE = "contextworld-cube-failed-attempt-forensic-query-jpeg-v1"
FORBIDDEN_PUBLIC_COMPONENTS = {
    "validation",
    "validation.lance",
    "public",
    "public_test",
    "public-test",
    "publictest",
}

FREEZE_TOP_KEYS = {
    "authorized_splits", "checks_passed", "frozen_at_utc", "frozen_evidence",
    "identity", "preregistration", "prior_episode_exclusion_basis", "protocol_id",
    "public_test", "reference_model_training_or_scoring_authorized",
    "rgb_history_probe", "schema_version", "scientific_change", "scope", "source_h5",
    "status",
}
PRIOR_TOP_KEYS = {
    "basis_receipt", "checks_passed", "coverage", "excluded_source_episode_count",
    "excluded_source_episodes", "excluded_source_episodes_sha256",
    "formal_build_requirement", "freeze_receipt", "input_artifacts", "preregistration",
    "prior_content_exclusions", "protocol_id", "public_test", "receipt_id",
    "reference_model_training_or_scoring", "schema_version", "source_h5", "status",
    "v4_preformal_build_report_count", "v4_preformal_content_receipt",
}
FAILURE_TOP_KEYS = {
    "build_passed", "checks_passed", "failed_attempt_content", "failed_output", "failure",
    "formal_build_attempt_consumed", "frozen_runtime_dependencies_from_original_freeze",
    "input_identities", "protocol_id", "raw_query_reconstruction_requirement", "receipt_id",
    "recovery_policy", "retry_authorized_under_original_preregistration", "schema_version",
    "scope", "stage_completion", "status",
}
QUERY_TOP_KEYS = {
    "checks_passed", "failed_attempt_content", "failed_attempt_receipt", "input_identities",
    "prior_overlap", "protocol_id", "public_test", "receipt_id", "reconstruction_contract",
    "reference_model_optimizer_steps", "reference_model_training_or_scoring", "rgb_probe",
    "schema_version", "status",
}
FAILURE_PAIR_KEYS = {
    "pair_id", "catalog_index", "source_row", "source_episode", "source_step",
    "action_anchor_id", "action_profile_id", "scene_template_content_hash",
    "pair_content_hash", "query_jpeg_sha256",
}
QUERY_PAIR_KEYS = FAILURE_PAIR_KEYS | {"split", "raw_query_pixel_hash"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_episode_digest(values: Sequence[int]) -> str:
    normalized = [int(value) for value in values]
    if normalized != sorted(set(normalized)) or any(value < 0 for value in normalized):
        raise RuntimeError("source episode values must be nonnegative, sorted, and unique")
    payload = b"".join(value.to_bytes(8, "little", signed=True) for value in normalized)
    return hashlib.sha256(b"contextworld-cube-prior-source-episodes-v1\0" + payload).hexdigest()


def content_digest(values: Sequence[str], *, field_name: str) -> str:
    normalized = list(values)
    if normalized != sorted(set(normalized)):
        raise RuntimeError(f"{field_name} values must be sorted and unique")
    decoded: list[bytes] = []
    for value in normalized:
        if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
            raise RuntimeError(f"{field_name} contains a malformed SHA256")
        decoded.append(bytes.fromhex(value))
    return hashlib.sha256(
        b"contextworld-cube-prior-content-exclusions-v1\0"
        + field_name.encode("ascii") + b"\0" + b"".join(decoded)
    ).hexdigest()


def forensic_jpeg_digest(values: Sequence[str]) -> str:
    normalized = list(values)
    if normalized != sorted(set(normalized)):
        raise RuntimeError("forensic JPEG values must be sorted and unique")
    decoded = []
    for value in normalized:
        if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
            raise RuntimeError("forensic JPEG set contains a malformed SHA256")
        decoded.append(bytes.fromhex(value))
    return hashlib.sha256(
        FORENSIC_JPEG_NAMESPACE.encode("ascii") + b"\0" + b"".join(decoded)
    ).hexdigest()


def _absolute_without_resolve(path: Path) -> Path:
    return Path(os.path.abspath(path.expanduser()))


def _reject_public(value: Path | str, *, label: str) -> None:
    for part in Path(value).parts:
        if part.lower() in FORBIDDEN_PUBLIC_COMPONENTS:
            raise RuntimeError(f"{label} contains forbidden Public component {part!r}")


def _regular_file(path: Path, *, label: str) -> Path:
    _reject_public(path, label=label)
    if path.is_symlink():
        raise FileNotFoundError(f"{label} must not be a symlink: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"{label} must be a regular file: {path}")
    return path


def _regular_directory(path: Path, *, label: str) -> Path:
    _reject_public(path, label=label)
    if path.is_symlink():
        raise FileNotFoundError(f"{label} must not be a symlink: {path}")
    if not path.is_dir():
        raise FileNotFoundError(f"{label} must be a directory: {path}")
    return path


def _read_bytes(path: Path, *, label: str) -> bytes:
    _regular_file(path, label=label)
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(descriptor, "rb") as stream:
        return stream.read()


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(_read_bytes(path, label=label).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{label} is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} root must be an object")
    return value


def _assert_identity(path: Path, expected_sha: str, expected_size: int, *, label: str) -> dict[str, Any]:
    _regular_file(path, label=label)
    size = path.stat().st_size
    if size != expected_size:
        raise RuntimeError(f"{label} size mismatch: {size} != {expected_size}")
    digest = file_sha256(path)
    if digest != expected_sha:
        raise RuntimeError(f"{label} SHA256 mismatch: {digest} != {expected_sha}")
    return {"sha256": digest, "size_bytes": size}


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], *, label: str) -> None:
    observed = set(value)
    if observed != expected:
        raise RuntimeError(
            f"{label} schema keys mismatch: extra={sorted(observed - expected)}, "
            f"missing={sorted(expected - observed)}"
        )


def _closed_public(value: Any, *, label: str, generated_key: bool = False) -> None:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{label} lacks Public closure object")
    if value.get("access_status") != "closed_not_read_not_scored":
        raise RuntimeError(f"{label} Public access status is not closed")
    flags = ["opened", "read", "hashed", "scored"]
    if generated_key:
        flags.append("generated")
    if any(value.get(flag) is not False for flag in flags):
        raise RuntimeError(f"{label} reports Public access or generation")


def _identity_matches(value: Any, expected_sha: str, expected_size: int, *, label: str) -> None:
    if not isinstance(value, Mapping) or value.get("sha256") != expected_sha or int(
        value.get("size_bytes", -1)
    ) != expected_size:
        raise RuntimeError(f"{label} identity mismatch")


def _source_matches(value: Any, *, label: str) -> None:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{label} source identity is missing")
    expected = {
        "symbol": SOURCE_SYMBOL,
        "sha256": SOURCE_SHA256,
        "size_bytes": SOURCE_SIZE_BYTES,
        "row_count": SOURCE_ROW_COUNT,
        "episode_count": SOURCE_EPISODE_COUNT,
    }
    if {key: value.get(key) for key in expected} != expected:
        raise RuntimeError(f"{label} source identity mismatch")


def _source_entry(value: Any, *, label: str) -> list[int]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{label} must be an object")
    raw = value.get("values")
    if not isinstance(raw, list) or any(isinstance(item, bool) or not isinstance(item, int) for item in raw):
        raise RuntimeError(f"{label}.values must be an integer array")
    values = [int(item) for item in raw]
    if int(value.get("count", -1)) != len(values) or value.get("sha256") != source_episode_digest(values):
        raise RuntimeError(f"{label} count or digest mismatch")
    return values


def _content_entry(value: Any, *, field: str, label: str) -> list[str]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{label} must be an object")
    raw = value.get("values")
    if not isinstance(raw, list) or any(not isinstance(item, str) for item in raw):
        raise RuntimeError(f"{label}.values must be a string array")
    values = list(raw)
    if int(value.get("count", -1)) != len(values) or value.get("sha256") != content_digest(
        values, field_name=field
    ):
        raise RuntimeError(f"{label} count or digest mismatch")
    return values


def _entry(values: Sequence[int] | Sequence[str], *, field: str) -> dict[str, Any]:
    normalized = sorted(values)
    digest = source_episode_digest(normalized) if field == "source_episodes" else content_digest(
        normalized, field_name=field
    )
    return {"values": normalized, "count": len(normalized), "sha256": digest}


def _validate_prereg(current_path: Path, snapshot_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    current_identity = _assert_identity(
        current_path, EXPECTED_PREREG_SHA256, EXPECTED_PREREG_SIZE_BYTES,
        label="current original-v4 preregistration",
    )
    snapshot_identity = _assert_identity(
        snapshot_path, EXPECTED_PREREG_SHA256, EXPECTED_PREREG_SIZE_BYTES,
        label="immutable original-v4 preregistration snapshot",
    )
    current_bytes = _read_bytes(current_path, label="current original-v4 preregistration")
    snapshot_bytes = _read_bytes(snapshot_path, label="original-v4 preregistration snapshot")
    if current_bytes != snapshot_bytes:
        raise RuntimeError("current original-v4 preregistration is not byte-equal to its snapshot")
    try:
        document = yaml.safe_load(current_bytes.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise RuntimeError("original-v4 preregistration is not valid YAML") from error
    if not isinstance(document, Mapping):
        raise RuntimeError("original-v4 preregistration root must be a mapping")
    if (
        document.get("schema_version") != 1
        or document.get("preregistration_id") != PREREGISTRATION_ID
        or document.get("protocol_id") != PROTOCOL
        or document.get("status") != PREREG_STATUS
        or document.get("phase") != "development_only"
    ):
        raise RuntimeError("original-v4 preregistration identity/schema mismatch")
    attempts = document.get("attempt_budget_and_stop_rules")
    if not isinstance(attempts, Mapping) or {
        key: attempts.get(key)
        for key in (
            "v4_builder_lance_smoke_attempts_authorized",
            "formal_build_attempts_authorized",
            "model_training_or_scoring_attempts_authorized",
            "public_test_attempts_authorized",
        )
    } != {
        "v4_builder_lance_smoke_attempts_authorized": 0,
        "formal_build_attempts_authorized": 1,
        "model_training_or_scoring_attempts_authorized": 0,
        "public_test_attempts_authorized": 0,
    }:
        raise RuntimeError("original-v4 attempt budget mismatch")
    required_failure_actions = {
        "write_failed_development_with_exact_failure_stage",
        "do_not_run_rgb_probe_when_its_inputs_are_invalid",
        "do_not_rebuild_under_this_preregistration",
        "keep_public_test_closed",
    }
    if set(attempts.get("on_formal_build_failure", [])) != required_failure_actions:
        raise RuntimeError("original-v4 formal-build failure policy mismatch")
    reference = document.get("reference_model_phase")
    if not isinstance(reference, Mapping) or reference.get("training_and_scoring_authorized") is not False:
        raise RuntimeError("original-v4 preregistration authorizes reference models")
    _closed_public(document.get("public_test"), label="original-v4 preregistration")
    return current_identity, snapshot_identity


def _validate_snapshots(paths: Mapping[str, Path]) -> dict[str, dict[str, Any]]:
    specs = {
        "builder": (EXPECTED_BUILDER_SHA256, EXPECTED_BUILDER_SIZE_BYTES,
                    "scripts/build_cube_grasp_rule_h3_v4_data.py",
                    "artifacts/evaluation/history3/cube_gripper_carry_h3_development_v4/v4_failed_attempt_builder_snapshot.py"),
        "physics": (EXPECTED_PHYSICS_SHA256, EXPECTED_PHYSICS_SIZE_BYTES,
                    "contextworld/evaluation/cube_grasp_rule_h3_v4.py",
                    "artifacts/evaluation/history3/cube_gripper_carry_h3_development_v4/failed_attempt_v1_snapshots/physics_v4.py"),
        "rgb_probe": (EXPECTED_PROBE_SHA256, EXPECTED_PROBE_SIZE_BYTES,
                      "scripts/probe_cube_grasp_rule_h3_v4_rgb_history.py",
                      "artifacts/evaluation/history3/cube_gripper_carry_h3_development_v4/failed_attempt_v1_snapshots/probe_v4.py"),
        "rgb_probe_tests": (EXPECTED_PROBE_TESTS_SHA256, EXPECTED_PROBE_TESTS_SIZE_BYTES,
                            "tests/test_cube_grasp_rule_h3_v4_rgb_history.py",
                            "artifacts/evaluation/history3/cube_gripper_carry_h3_development_v4/failed_attempt_v1_snapshots/probe_v4_tests.py"),
        "action_support": (EXPECTED_ACTION_SUPPORT_SHA256, EXPECTED_ACTION_SUPPORT_SIZE_BYTES,
                           "scripts/audit_cube_grasp_rule_h3_v4_action_support.py",
                           "artifacts/evaluation/history3/cube_gripper_carry_h3_development_v4/failed_attempt_v1_snapshots/action_support_v4.py"),
        "action_support_tests": (EXPECTED_ACTION_SUPPORT_TESTS_SHA256, EXPECTED_ACTION_SUPPORT_TESTS_SIZE_BYTES,
                                 "tests/test_cube_grasp_rule_h3_v4_action_support.py",
                                 "artifacts/evaluation/history3/cube_gripper_carry_h3_development_v4/failed_attempt_v1_snapshots/action_support_v4_tests.py"),
    }
    receipts: dict[str, dict[str, Any]] = {}
    for name, (digest, size, original_path, snapshot_path) in specs.items():
        identity = _assert_identity(paths[name], digest, size, label=f"original-v4 {name} snapshot")
        receipts[name] = {
            "original_path": original_path,
            "snapshot_path": snapshot_path,
            **identity,
        }
    return receipts


def _validate_freeze(path: Path, snapshot_identities: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    identity = _assert_identity(path, EXPECTED_FREEZE_SHA256, EXPECTED_FREEZE_SIZE_BYTES, label="original-v4 freeze receipt")
    value = _read_json(path, label="original-v4 freeze receipt")
    _require_exact_keys(value, FREEZE_TOP_KEYS, label="original-v4 freeze receipt")
    if value.get("schema_version") != 1 or value.get("protocol_id") != PROTOCOL or value.get(
        "status"
    ) != FREEZE_STATUS or value.get("checks_passed") is not True:
        raise RuntimeError("original-v4 freeze status/schema mismatch")
    _identity_matches(value.get("preregistration"), EXPECTED_PREREG_SHA256, EXPECTED_PREREG_SIZE_BYTES, label="freeze preregistration")
    _source_matches(value.get("source_h5"), label="freeze")
    _closed_public(value.get("public_test"), label="original-v4 freeze")
    if value.get("reference_model_training_or_scoring_authorized") is not False:
        raise RuntimeError("original-v4 freeze authorizes reference models")
    frozen = value.get("identity")
    if not isinstance(frozen, Mapping):
        raise RuntimeError("original-v4 freeze lacks identity map")
    mapping = {
        "builder": "v4_builder",
        "physics": "v4_physics",
        "rgb_probe": "v4_probe",
        "rgb_probe_tests": "v4_probe_tests",
        "action_support": "v4_action_support_audit",
        "action_support_tests": "v4_action_support_audit_tests",
    }
    for snapshot_name, freeze_name in mapping.items():
        frozen_identity = frozen.get(freeze_name)
        expected = snapshot_identities[snapshot_name]
        _identity_matches(frozen_identity, str(expected["sha256"]), int(expected["size_bytes"]), label=f"freeze {freeze_name}")
        if not isinstance(frozen_identity, Mapping) or frozen_identity.get("path") != expected["original_path"]:
            raise RuntimeError(f"freeze {freeze_name} original path mismatch")
    return {"document": value, "identity": identity}


def _validate_prior(path: Path) -> dict[str, Any]:
    identity = _assert_identity(path, EXPECTED_PRIOR_SHA256, EXPECTED_PRIOR_SIZE_BYTES, label="original-v4 final prior exclusion receipt")
    value = _read_json(path, label="original-v4 final prior exclusion receipt")
    _require_exact_keys(value, PRIOR_TOP_KEYS, label="original-v4 final prior exclusion receipt")
    if value.get("schema_version") != 1 or value.get("protocol_id") != PROTOCOL or value.get(
        "status"
    ) != FREEZE_STATUS or value.get("checks_passed") is not True:
        raise RuntimeError("original-v4 prior receipt status/schema mismatch")
    _identity_matches(value.get("preregistration"), EXPECTED_PREREG_SHA256, EXPECTED_PREREG_SIZE_BYTES, label="prior preregistration")
    _identity_matches(value.get("freeze_receipt"), EXPECTED_FREEZE_SHA256, EXPECTED_FREEZE_SIZE_BYTES, label="prior freeze")
    _source_matches(value.get("source_h5"), label="prior receipt")
    _closed_public(value.get("public_test"), label="original-v4 prior receipt")
    if value.get("reference_model_training_or_scoring") is not False:
        raise RuntimeError("original-v4 prior receipt reports model use")
    source_values = value.get("excluded_source_episodes")
    if not isinstance(source_values, list):
        raise RuntimeError("prior excluded_source_episodes must be an array")
    normalized_source = [int(item) for item in source_values]
    if normalized_source != sorted(set(normalized_source)) or int(value.get("excluded_source_episode_count", -1)) != len(
        normalized_source
    ) or value.get("excluded_source_episodes_sha256") != source_episode_digest(normalized_source):
        raise RuntimeError("prior source exclusion set/count/digest mismatch")
    raw_sets = value.get("prior_content_exclusions")
    if not isinstance(raw_sets, Mapping) or set(raw_sets) != set(CONTENT_FIELDS):
        raise RuntimeError("prior content exclusion schema mismatch")
    sets: dict[str, list[int] | list[str]] = {"source_episodes": normalized_source}
    for field in CONTENT_FIELDS:
        sets[field] = _content_entry(raw_sets[field], field=field, label=f"prior.{field}")
    return {"document": value, "identity": identity, "sets": sets}


def _validate_request_and_inventory(root: Path, request_path: Path, fragment_path: Path) -> dict[str, Any]:
    _regular_directory(root, label="failed formal output root")
    request_identity = _assert_identity(request_path, EXPECTED_REQUEST_SHA256, EXPECTED_REQUEST_SIZE_BYTES, label="failed formal request")
    fragment_identity = _assert_identity(fragment_path, EXPECTED_FRAGMENT_SHA256, EXPECTED_FRAGMENT_SIZE_BYTES, label="failed train fragment")
    if request_path != root / "request.json":
        raise RuntimeError("request path is not failed_output_root/request.json")
    if fragment_path != root / "train.lance" / "data" / EXPECTED_FRAGMENT_NAME:
        raise RuntimeError("partial fragment path/name mismatch")
    expected_paths = {
        Path("request.json"), Path("train.lance"), Path("train.lance/data"),
        Path("train.lance/_versions"), Path("train.lance/_transactions"),
        Path("train.lance/data") / EXPECTED_FRAGMENT_NAME,
    }
    observed: set[Path] = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if path.is_symlink():
            raise RuntimeError(f"failed output contains symlink {relative}")
        observed.add(relative)
    if observed != expected_paths:
        raise RuntimeError(
            f"failed output inventory mismatch: extra={sorted(map(str, observed - expected_paths))}, "
            f"missing={sorted(map(str, expected_paths - observed))}"
        )
    for directory_name in ("_versions", "_transactions"):
        if any((root / "train.lance" / directory_name).iterdir()):
            raise RuntimeError(f"train.lance/{directory_name} is not empty")
    request = _read_json(request_path, label="failed formal request")
    if (
        request.get("protocol") != PROTOCOL
        or request.get("resolved_output") != EXPECTED_LOGICAL_FAILED_ROOT
        or request.get("logical_default_output") != EXPECTED_LOGICAL_FAILED_ROOT
        or request.get("pair_counts") != {"train": EXPECTED_PAIR_COUNT, "loader_validation": 256}
        or request.get("active_splits") != ["train", "loader_validation"]
        or request.get("public_test_opened") is not False
        or request.get("public_test_generated") is not False
    ):
        raise RuntimeError("failed formal request contract mismatch")
    _identity_matches(request.get("freeze_receipt"), EXPECTED_FREEZE_SHA256, EXPECTED_FREEZE_SIZE_BYTES, label="request freeze")
    _identity_matches(request.get("prior_episode_exclusion_receipt"), EXPECTED_PRIOR_SHA256, EXPECTED_PRIOR_SIZE_BYTES, label="request prior")
    inventory = [
        {"path": f"{EXPECTED_LOGICAL_FAILED_ROOT}/request.json", "type": "regular_file", **request_identity},
        {"path": f"{EXPECTED_LOGICAL_FAILED_ROOT}/train.lance/data/{EXPECTED_FRAGMENT_NAME}", "type": "regular_file", **fragment_identity},
        {"path": f"{EXPECTED_LOGICAL_FAILED_ROOT}/train.lance/_versions", "type": "empty_directory", "entry_count": 0},
        {"path": f"{EXPECTED_LOGICAL_FAILED_ROOT}/train.lance/_transactions", "type": "empty_directory", "entry_count": 0},
    ]
    return {"request": request, "request_identity": request_identity, "fragment_identity": fragment_identity, "inventory": inventory}


def _validate_failure_receipt(path: Path, inventory: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    identity = _assert_identity(path, EXPECTED_FAILURE_RECEIPT_SHA256, EXPECTED_FAILURE_RECEIPT_SIZE_BYTES, label="failed formal attempt receipt")
    value = _read_json(path, label="failed formal attempt receipt")
    _require_exact_keys(value, FAILURE_TOP_KEYS, label="failed formal attempt receipt")
    if (
        value.get("schema_version") != 1 or value.get("protocol_id") != PROTOCOL
        or value.get("receipt_id") != FAILED_RECEIPT_ID or value.get("status") != FAILED_RECEIPT_STATUS
        or value.get("checks_passed") is not True or value.get("build_passed") is not False
        or value.get("formal_build_attempt_consumed") is not True
        or value.get("retry_authorized_under_original_preregistration") is not False
    ):
        raise RuntimeError("failed formal attempt receipt status/schema mismatch")
    inputs = value.get("input_identities")
    if not isinstance(inputs, Mapping):
        raise RuntimeError("failed formal attempt receipt lacks input identities")
    for field, digest, size in (
        ("preregistration", EXPECTED_PREREG_SHA256, EXPECTED_PREREG_SIZE_BYTES),
        ("freeze_receipt", EXPECTED_FREEZE_SHA256, EXPECTED_FREEZE_SIZE_BYTES),
        ("prior_exclusion_receipt", EXPECTED_PRIOR_SHA256, EXPECTED_PRIOR_SIZE_BYTES),
        ("builder_snapshot", EXPECTED_BUILDER_SHA256, EXPECTED_BUILDER_SIZE_BYTES),
        ("request_json", EXPECTED_REQUEST_SHA256, EXPECTED_REQUEST_SIZE_BYTES),
        ("partial_train_fragment", EXPECTED_FRAGMENT_SHA256, EXPECTED_FRAGMENT_SIZE_BYTES),
    ):
        _identity_matches(inputs.get(field), digest, size, label=f"failed receipt {field}")
    _source_matches(inputs.get("source_h5"), label="failed receipt")
    runtime = value.get("frozen_runtime_dependencies_from_original_freeze")
    if not isinstance(runtime, Mapping):
        raise RuntimeError("failed receipt lacks frozen runtime dependencies")
    _identity_matches(runtime.get("v4_physics"), EXPECTED_PHYSICS_SHA256, EXPECTED_PHYSICS_SIZE_BYTES, label="failed receipt v4 physics")
    failure = value.get("failure")
    if not isinstance(failure, Mapping) or {
        "exit_code": failure.get("exit_code"), "stage": failure.get("stage"),
        "errno_name": failure.get("errno_name"), "errno_number": failure.get("errno_number"),
        "exception_type": failure.get("exception_type"), "persistent_log_present": failure.get("persistent_log_present"),
    } != {
        "exit_code": 1, "stage": "lance_train_commit_atomic_rename", "errno_name": "EPERM",
        "errno_number": 1, "exception_type": "OSError", "persistent_log_present": False,
    }:
        raise RuntimeError("failed receipt exception classification mismatch")
    stage = value.get("stage_completion")
    expected_stage = {
        "train_generation_accepted_pairs": EXPECTED_PAIR_COUNT,
        "train_generation_attempted_candidates": EXPECTED_PAIR_COUNT,
        "train_lance_data_fragment_written": True,
        "train_lance_commit_completed": False,
        "loader_validation_started": False,
        "build_report_written": False,
        "manifest_written": False,
        "scientifically_inspectable_partial_output": True,
    }
    if not isinstance(stage, Mapping) or {key: stage.get(key) for key in expected_stage} != expected_stage:
        raise RuntimeError("failed receipt stage completion mismatch")
    failed_output = value.get("failed_output")
    if not isinstance(failed_output, Mapping) or failed_output.get("logical_root") != EXPECTED_LOGICAL_FAILED_ROOT or failed_output.get(
        "allowed_inventory_only"
    ) is not True or failed_output.get("inventory") != list(inventory):
        raise RuntimeError("failed receipt inventory binding mismatch")
    scope = value.get("scope")
    if not isinstance(scope, Mapping):
        raise RuntimeError("failed receipt lacks scope")
    _closed_public(scope.get("public_test"), label="failed receipt")
    if scope.get("rgb_probe_run") is not False or scope.get("reference_model_training_or_scoring") is not False or scope.get(
        "optimizer_steps"
    ) != 0:
        raise RuntimeError("failed receipt reports probe/model activity")
    policy = value.get("recovery_policy")
    if not isinstance(policy, Mapping) or any(
        policy.get(field) is not True for field in (
            "original_v4_preregistration_attempt_budget_exhausted",
            "original_failed_tree_must_remain_immutable",
            "silent_retry_or_overwrite_forbidden",
            "newly_frozen_recovery_preregistration_required",
            "failed_source_action_scene_pair_and_reconstructed_raw_query_must_be_excluded",
        )
    ):
        raise RuntimeError("failed receipt recovery policy mismatch")
    content = value.get("failed_attempt_content")
    if not isinstance(content, Mapping):
        raise RuntimeError("failed receipt lacks failed content")
    if {
        "split": content.get("split"), "row_count": content.get("row_count"),
        "episode_count": content.get("episode_count"), "pair_count": content.get("pair_count"),
        "catalog_start": content.get("catalog_index_start_inclusive"),
        "catalog_stop": content.get("catalog_index_stop_exclusive"),
        "anchors": content.get("action_anchor_counts"),
    } != {
        "split": "train", "row_count": EXPECTED_ROW_COUNT, "episode_count": EXPECTED_EPISODE_COUNT,
        "pair_count": EXPECTED_PAIR_COUNT, "catalog_start": EXPECTED_CATALOG_START,
        "catalog_stop": EXPECTED_CATALOG_STOP, "anchors": EXPECTED_ANCHOR_COUNTS,
    }:
        raise RuntimeError("failed receipt content counts/catalog/anchors mismatch")
    constraints = content.get("profile_constraints")
    if not isinstance(constraints, Mapping) or constraints.get("passed") is not True or any(
        constraints.get(field) != 0 for field in (
            "maximum_abs_sum_p", "maximum_abs_final_p", "maximum_abs_moment_error",
            "terminal_nonzero_value_count",
        )
    ):
        raise RuntimeError("failed receipt action constraints mismatch")
    source_values = _source_entry(content.get("source_episodes"), label="failed.source_episodes")
    raw_sets = content.get("prior_content_exclusions")
    if not isinstance(raw_sets, Mapping) or set(raw_sets) != set(CONTENT_FIELDS[:3]):
        raise RuntimeError("failed receipt inspectable content set schema mismatch")
    sets: dict[str, list[int] | list[str]] = {"source_episodes": source_values}
    for field in CONTENT_FIELDS[:3]:
        sets[field] = _content_entry(raw_sets[field], field=field, label=f"failed.{field}")
    jpeg_entry = content.get("query_jpeg_sha256")
    if not isinstance(jpeg_entry, Mapping) or jpeg_entry.get("digest_namespace") != FORENSIC_JPEG_NAMESPACE or jpeg_entry.get(
        "role"
    ) != "forensic_binding_only_not_raw_query_pixel_hash":
        raise RuntimeError("failed receipt forensic JPEG contract mismatch")
    jpeg_values = jpeg_entry.get("values")
    if not isinstance(jpeg_values, list) or int(jpeg_entry.get("count", -1)) != len(jpeg_values) or jpeg_entry.get(
        "sha256"
    ) != forensic_jpeg_digest(jpeg_values):
        raise RuntimeError("failed receipt forensic JPEG set mismatch")
    pairs = content.get("pairs")
    if not isinstance(pairs, list) or len(pairs) != EXPECTED_PAIR_COUNT:
        raise RuntimeError("failed receipt pair list count mismatch")
    pair_map: dict[str, dict[str, Any]] = {}
    extracted = {field: set() for field in ("source_episodes", *CONTENT_FIELDS[:3])}
    extracted_jpegs: set[str] = set()
    anchors: dict[str, int] = {name: 0 for name in EXPECTED_ANCHOR_COUNTS}
    for local_index, pair in enumerate(pairs):
        if not isinstance(pair, Mapping):
            raise RuntimeError("failed receipt pair is not an object")
        _require_exact_keys(pair, FAILURE_PAIR_KEYS, label="failed receipt pair")
        pair_id = f"cube-carry-v4-train-{local_index:06d}"
        if pair.get("pair_id") != pair_id or pair.get("catalog_index") != EXPECTED_CATALOG_START + local_index:
            raise RuntimeError("failed receipt pair ordering/catalog mismatch")
        anchor = pair.get("action_anchor_id")
        if anchor not in anchors:
            raise RuntimeError("failed receipt pair has unknown anchor")
        anchors[str(anchor)] += 1
        for field in ("action_profile_id", "scene_template_content_hash", "pair_content_hash", "query_jpeg_sha256"):
            if not isinstance(pair.get(field), str) or SHA256_RE.fullmatch(str(pair[field])) is None:
                raise RuntimeError(f"failed receipt pair has malformed {field}")
        pair_map[pair_id] = dict(pair)
        extracted["source_episodes"].add(int(pair["source_episode"]))
        extracted["action_profile_ids"].add(str(pair["action_profile_id"]))
        extracted["scene_template_content_hashes"].add(str(pair["scene_template_content_hash"]))
        extracted["pair_content_hashes"].add(str(pair["pair_content_hash"]))
        extracted_jpegs.add(str(pair["query_jpeg_sha256"]))
    if anchors != EXPECTED_ANCHOR_COUNTS or extracted_jpegs != set(jpeg_values):
        raise RuntimeError("failed receipt pair anchor/JPEG set mismatch")
    for field in sets:
        if extracted[field] != set(sets[field]):
            raise RuntimeError(f"failed receipt pair-derived {field} set mismatch")
    return {"document": value, "identity": identity, "sets": sets, "pairs": pair_map, "jpeg_values": jpeg_values}


def _validate_query_receipt(path: Path, failure: Mapping[str, Any]) -> dict[str, Any]:
    identity = _assert_identity(path, EXPECTED_QUERY_RECEIPT_SHA256, EXPECTED_QUERY_RECEIPT_SIZE_BYTES, label="failed query reconstruction receipt")
    value = _read_json(path, label="failed query reconstruction receipt")
    _require_exact_keys(value, QUERY_TOP_KEYS, label="failed query reconstruction receipt")
    if (
        value.get("schema_version") != 1 or value.get("protocol_id") != PROTOCOL
        or value.get("receipt_id") != QUERY_RECEIPT_ID or value.get("status") != QUERY_RECEIPT_STATUS
        or value.get("checks_passed") is not True
    ):
        raise RuntimeError("query reconstruction receipt status/schema mismatch")
    _identity_matches(value.get("failed_attempt_receipt"), EXPECTED_FAILURE_RECEIPT_SHA256, EXPECTED_FAILURE_RECEIPT_SIZE_BYTES, label="query failed receipt")
    inputs = value.get("input_identities")
    if not isinstance(inputs, Mapping):
        raise RuntimeError("query receipt lacks input identities")
    for field, digest, size in (
        ("preregistration", EXPECTED_PREREG_SHA256, EXPECTED_PREREG_SIZE_BYTES),
        ("freeze_receipt", EXPECTED_FREEZE_SHA256, EXPECTED_FREEZE_SIZE_BYTES),
        ("prior_exclusion_receipt", EXPECTED_PRIOR_SHA256, EXPECTED_PRIOR_SIZE_BYTES),
        ("failed_attempt_receipt", EXPECTED_FAILURE_RECEIPT_SHA256, EXPECTED_FAILURE_RECEIPT_SIZE_BYTES),
        ("builder_snapshot", EXPECTED_BUILDER_SHA256, EXPECTED_BUILDER_SIZE_BYTES),
        ("physics_snapshot", EXPECTED_PHYSICS_SHA256, EXPECTED_PHYSICS_SIZE_BYTES),
        ("request_json", EXPECTED_REQUEST_SHA256, EXPECTED_REQUEST_SIZE_BYTES),
        ("partial_train_fragment", EXPECTED_FRAGMENT_SHA256, EXPECTED_FRAGMENT_SIZE_BYTES),
    ):
        _identity_matches(inputs.get(field), digest, size, label=f"query receipt {field}")
    _source_matches(inputs.get("source_h5"), label="query receipt")
    contract = value.get("reconstruction_contract")
    expected_contract = {
        "all_inputs_reverified_unchanged_after_replay": True,
        "builder_snapshot_loaded_by_explicit_path": True,
        "dataset_manifest_opened": False,
        "fragment_read_api": "lance.file.LanceFileReader_single_file",
        "jpeg_quality": 95,
        "jpeg_reencoding_bitwise_equal_to_fragment": True,
        "lance_written": False,
        "physics_snapshot_loaded_by_explicit_path": True,
        "query_model_step_idx": 2,
        "raw_query_frame": "pixels[2]_before_JPEG",
        "raw_query_hash": "Cube_array_sha256_dtype_shape_bytes",
        "raw_query_hashes_unique": True,
        "raw_query_prior_overlap_zero": True,
        "reencoded_query_jpegs_match_fragment": True,
        "replayed_mode": "cannot_hold_only",
        "stored_paired_query_jpegs_equal": True,
        "workers": 16,
    }
    if not isinstance(contract, Mapping) or dict(contract) != expected_contract:
        raise RuntimeError("query reconstruction contract mismatch")
    _closed_public(value.get("public_test"), label="query reconstruction receipt")
    probe = value.get("rgb_probe")
    if not isinstance(probe, Mapping) or {key: probe.get(key) for key in ("opened", "run", "scored")} != {
        "opened": False, "run": False, "scored": False,
    }:
        raise RuntimeError("query receipt reports RGB probe activity")
    if value.get("reference_model_training_or_scoring") is not False or value.get("reference_model_optimizer_steps") != 0:
        raise RuntimeError("query receipt reports reference-model activity")
    content = value.get("failed_attempt_content")
    if not isinstance(content, Mapping) or {
        "split": content.get("split"), "row_count": content.get("row_count"),
        "episode_count": content.get("episode_count"), "pair_count": content.get("pair_count"),
    } != {"split": "train", "row_count": EXPECTED_ROW_COUNT, "episode_count": EXPECTED_EPISODE_COUNT, "pair_count": EXPECTED_PAIR_COUNT}:
        raise RuntimeError("query receipt content counts mismatch")
    source_values = _source_entry(content.get("source_episodes"), label="query.source_episodes")
    raw_sets = content.get("prior_content_exclusions")
    if not isinstance(raw_sets, Mapping) or set(raw_sets) != set(CONTENT_FIELDS):
        raise RuntimeError("query receipt content set schema mismatch")
    sets: dict[str, list[int] | list[str]] = {"source_episodes": source_values}
    for field in CONTENT_FIELDS:
        sets[field] = _content_entry(raw_sets[field], field=field, label=f"query.{field}")
    for field in ("source_episodes", *CONTENT_FIELDS[:3]):
        if sets[field] != failure["sets"][field]:
            raise RuntimeError(f"query receipt {field} differs from failed receipt")
    pairs = content.get("pairs")
    if not isinstance(pairs, list) or len(pairs) != EXPECTED_PAIR_COUNT:
        raise RuntimeError("query receipt pair count mismatch")
    extracted_raw: set[str] = set()
    for local_index, pair in enumerate(pairs):
        if not isinstance(pair, Mapping):
            raise RuntimeError("query receipt pair is not an object")
        _require_exact_keys(pair, QUERY_PAIR_KEYS, label="query receipt pair")
        pair_id = f"cube-carry-v4-train-{local_index:06d}"
        raw_hash = pair.get("raw_query_pixel_hash")
        if pair.get("pair_id") != pair_id or pair.get("split") != "train" or not isinstance(raw_hash, str) or SHA256_RE.fullmatch(raw_hash) is None:
            raise RuntimeError("query receipt pair identity/raw hash mismatch")
        failed_pair = failure["pairs"].get(pair_id)
        comparable = {key: pair[key] for key in FAILURE_PAIR_KEYS}
        if comparable != failed_pair:
            raise RuntimeError("query receipt pair differs from failed receipt")
        extracted_raw.add(raw_hash)
    if extracted_raw != set(sets["query_pixel_hashes"]):
        raise RuntimeError("query receipt pair-derived raw query set mismatch")
    overlap = value.get("prior_overlap")
    expected_overlap_keys = {"source_episode", *CONTENT_FIELDS, "passed"}
    if not isinstance(overlap, Mapping) or set(overlap) != expected_overlap_keys or overlap.get("passed") is not True:
        raise RuntimeError("query receipt prior-overlap schema/status mismatch")
    for field in ("source_episode", *CONTENT_FIELDS):
        item = overlap.get(field)
        if not isinstance(item, Mapping) or set(item) != {"count", "values"} or item.get("count") != 0 or item.get("values") != []:
            raise RuntimeError(f"query receipt prior overlap is nonzero for {field}")
    return {"document": value, "identity": identity, "sets": sets}


def finalize_infrastructure_failure(
    *, current_old_prereg: Path, original_prereg_snapshot: Path, freeze_receipt: Path,
    prior_exclusion_receipt: Path, failed_attempt_receipt: Path,
    query_reconstruction_receipt: Path, builder_snapshot: Path, physics_snapshot: Path,
    probe_snapshot: Path, probe_tests_snapshot: Path, action_support_snapshot: Path,
    action_support_tests_snapshot: Path, failed_output_root: Path, request_json: Path,
    partial_train_fragment: Path, output: Path,
) -> dict[str, Any]:
    all_paths = {
        "current_old_prereg": current_old_prereg,
        "original_prereg_snapshot": original_prereg_snapshot,
        "freeze_receipt": freeze_receipt,
        "prior_exclusion_receipt": prior_exclusion_receipt,
        "failed_attempt_receipt": failed_attempt_receipt,
        "query_reconstruction_receipt": query_reconstruction_receipt,
        "builder_snapshot": builder_snapshot,
        "physics_snapshot": physics_snapshot,
        "probe_snapshot": probe_snapshot,
        "probe_tests_snapshot": probe_tests_snapshot,
        "action_support_snapshot": action_support_snapshot,
        "action_support_tests_snapshot": action_support_tests_snapshot,
        "failed_output_root": failed_output_root,
        "request_json": request_json,
        "partial_train_fragment": partial_train_fragment,
        "output": output,
    }
    for label, path in all_paths.items():
        _reject_public(path, label=label)
    if output.name != OUTPUT_NAME:
        raise RuntimeError(f"output filename must be {OUTPUT_NAME}")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite Development decision {output}")
    if not output.parent.is_dir() or output.parent.is_symlink():
        raise FileNotFoundError("Development decision parent must be an existing non-symlink directory")

    current_prereg_identity, prereg_snapshot_identity = _validate_prereg(current_old_prereg, original_prereg_snapshot)
    snapshot_paths = {
        "builder": builder_snapshot, "physics": physics_snapshot, "rgb_probe": probe_snapshot,
        "rgb_probe_tests": probe_tests_snapshot, "action_support": action_support_snapshot,
        "action_support_tests": action_support_tests_snapshot,
    }
    snapshots = _validate_snapshots(snapshot_paths)
    freeze = _validate_freeze(freeze_receipt, snapshots)
    prior = _validate_prior(prior_exclusion_receipt)
    physical = _validate_request_and_inventory(failed_output_root, request_json, partial_train_fragment)
    failure = _validate_failure_receipt(failed_attempt_receipt, physical["inventory"])
    query = _validate_query_receipt(query_reconstruction_receipt, failure)

    # Recompute overlap independently; receipt assertions alone are insufficient.
    overlap_receipt: dict[str, dict[str, Any] | bool] = {}
    recovery_union: dict[str, dict[str, Any]] = {}
    failed_entries: dict[str, dict[str, Any]] = {}
    for field in ALL_EXCLUSION_FIELDS:
        old_values = set(prior["sets"][field])
        failed_values = set(query["sets"][field])
        intersection = sorted(old_values & failed_values)
        overlap_receipt[field] = {"values": intersection, "count": len(intersection)}
        if intersection:
            raise RuntimeError(f"failed formal content overlaps prior evidence for {field}")
        failed_entries[field] = _entry(sorted(failed_values), field=field)
        recovery_union[field] = _entry(sorted(old_values | failed_values), field=field)
    overlap_receipt["passed"] = True

    identities = {
        "current_old_preregistration": {
            "path": "configs/benchmark/cube_gripper_carry_h3_development_prereg_v4.yaml",
            **current_prereg_identity,
        },
        "original_preregistration_snapshot": {
            "path": "artifacts/evaluation/history3/cube_gripper_carry_h3_development_v4/failed_attempt_v1_snapshots/prereg_v4.yaml",
            "byte_equal_to_current_old_preregistration": True,
            **prereg_snapshot_identity,
        },
        "freeze_receipt": {
            "path": "artifacts/evaluation/history3/cube_gripper_carry_h3_development_v4/development_prereg_freeze_receipt_v1.json",
            **freeze["identity"],
        },
        "final_prior_exclusion_receipt": {
            "path": "artifacts/evaluation/history3/cube_gripper_carry_h3_development_v4/prior_episode_exclusions_final_v1.json",
            **prior["identity"],
        },
        "failed_formal_attempt_receipt": {
            "path": "artifacts/evaluation/history3/cube_gripper_carry_h3_development_v4/failed_formal_attempt_receipt_v1.json",
            **failure["identity"],
        },
        "query_reconstruction_receipt": {
            "path": "artifacts/evaluation/history3/cube_gripper_carry_h3_development_v4/failed_formal_attempt_query_reconstruction_receipt_v1.json",
            **query["identity"],
        },
        "request_json": {
            "path": f"{EXPECTED_LOGICAL_FAILED_ROOT}/request.json", **physical["request_identity"],
        },
        "partial_train_fragment": {
            "path": f"{EXPECTED_LOGICAL_FAILED_ROOT}/train.lance/data/{EXPECTED_FRAGMENT_NAME}",
            **physical["fragment_identity"],
        },
    }
    decision = {
        "schema_version": 1,
        "protocol_id": PROTOCOL,
        "preregistration_id": PREREGISTRATION_ID,
        "decision_id": DECISION_ID,
        "status": DECISION_STATUS,
        "failure_stage": FAILURE_STAGE,
        "classification": CLASSIFICATION,
        "checks_passed": True,
        "summary": {
            "formal_build_completed": False,
            "train_generation_completed": True,
            "train_lance_commit_completed": False,
            "development_split_started": False,
            "scientific_data_gates_reached": False,
            "rgb_history_probe_reached": False,
            "development_ready": False,
            "reason": "The sole original-v4 formal attempt generated all 2048 Training pairs but failed during the Lance atomic commit rename with EPERM before Development generation or any scientific gate.",
        },
        "input_identities": identities,
        "original_frozen_code": snapshots,
        "source_identity": {
            "symbol": SOURCE_SYMBOL, "path_recorded": False, "sha256": SOURCE_SHA256,
            "size_bytes": SOURCE_SIZE_BYTES, "row_count": SOURCE_ROW_COUNT,
            "episode_count": SOURCE_EPISODE_COUNT,
        },
        "formal_build": {
            "attempt_number": 1,
            "attempt_budget": 1,
            "attempt_consumed": True,
            "exit_code": 1,
            "exception_type": "OSError",
            "errno_name": "EPERM",
            "errno_number": 1,
            "failure_stage_from_attempt_receipt": "lance_train_commit_atomic_rename",
            "train_generation": {
                "accepted_pairs": EXPECTED_PAIR_COUNT,
                "attempted_candidates": EXPECTED_PAIR_COUNT,
                "rejected_candidates": 0,
                "acceptance_rate": 1.0,
                "row_count": EXPECTED_ROW_COUNT,
                "condition_episode_count": EXPECTED_EPISODE_COUNT,
                "action_anchor_counts": dict(EXPECTED_ANCHOR_COUNTS),
                "profile_constraints_passed": True,
            },
            "train_lance_data_fragment_written": True,
            "train_lance_commit_completed": False,
            "loader_validation_started": False,
            "development_started": False,
            "build_report_written": False,
            "manifest_written": False,
            "canonical_dataset_openable": False,
        },
        "failed_output": {
            "logical_root": EXPECTED_LOGICAL_FAILED_ROOT,
            "immutable_partial_not_canonical_dataset": True,
            "allowed_inventory_only": True,
            "inventory": physical["inventory"],
            "lance_versions_directory_empty": True,
            "lance_transactions_directory_empty": True,
        },
        "failed_content_exclusions": failed_entries,
        "prior_overlap": overlap_receipt,
        "recovery_exclusion_union": recovery_union,
        "required_gate_summary": {
            "original_preregistration_and_freeze_identity": True,
            "formal_train_generation": True,
            "formal_train_commit": False,
            "formal_development_generation": False,
            "formal_data_contract": False,
            "actual_formal_action_support_binding": False,
            "formal_causal_and_fresh_replay_gates": False,
            "rgb_history_probe": False,
            "all_required": False,
            "scientific_gate_failure_observed": False,
        },
        "rgb_history_probe": {
            "authorized_attempts_under_original_preregistration": 1,
            "inputs_valid_after_build_failure": False,
            "opened": False,
            "run": False,
            "scored": False,
            "attempts_consumed": 0,
            "not_run_reason": "formal_build_failed_before_valid_train_and_loader_validation_inputs",
        },
        "reference_model_phase": {
            "training_or_scoring_authorized": False,
            "trainer_invoked": False,
            "optimizer_steps_run": 0,
            "checkpoints_created": False,
            "lewm_or_pldm_scoring_run": False,
        },
        "public_test": {
            "access_status": "closed_not_read_not_scored",
            "generated": False, "opened": False, "read": False, "hashed": False, "scored": False,
        },
        "claims": {
            "formal_development_data_constructed": False,
            "formal_development_data_contract_passed": False,
            "data_readiness_passed": False,
            "development_ready": False,
            "scientific_gate_failure_claimed": False,
            "positive_rgb_history_recoverability_claim_allowed": False,
            "positive_reference_model_claim_allowed": False,
            "release_claim_allowed": False,
            "suite_registration_allowed": False,
            "public_test_claim_allowed": False,
        },
        "recovery_policy": {
            "original_v4_formal_attempt_consumed": True,
            "retry_authorized_under_original_preregistration": False,
            "silent_retry_or_overwrite_forbidden": True,
            "original_failed_tree_must_remain_immutable": True,
            "partial_output_promotable": False,
            "new_frozen_recovery_preregistration_required": True,
            "five_failed_content_sets_must_be_excluded": True,
            "recovery_must_use_recovery_exclusion_union": True,
        },
    }
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(output, flags, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(decision, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return decision


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "current-old-prereg", "original-prereg-snapshot", "freeze-receipt",
        "prior-exclusion-receipt", "failed-attempt-receipt", "query-reconstruction-receipt",
        "builder-snapshot", "physics-snapshot", "probe-snapshot", "probe-tests-snapshot",
        "action-support-snapshot", "action-support-tests-snapshot", "failed-output-root",
        "request-json", "partial-train-fragment", "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args(argv)
    for name, value in vars(args).items():
        _reject_public(value, label=name)
        setattr(args, name, _absolute_without_resolve(value))
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    decision = finalize_infrastructure_failure(
        current_old_prereg=args.current_old_prereg,
        original_prereg_snapshot=args.original_prereg_snapshot,
        freeze_receipt=args.freeze_receipt,
        prior_exclusion_receipt=args.prior_exclusion_receipt,
        failed_attempt_receipt=args.failed_attempt_receipt,
        query_reconstruction_receipt=args.query_reconstruction_receipt,
        builder_snapshot=args.builder_snapshot,
        physics_snapshot=args.physics_snapshot,
        probe_snapshot=args.probe_snapshot,
        probe_tests_snapshot=args.probe_tests_snapshot,
        action_support_snapshot=args.action_support_snapshot,
        action_support_tests_snapshot=args.action_support_tests_snapshot,
        failed_output_root=args.failed_output_root,
        request_json=args.request_json,
        partial_train_fragment=args.partial_train_fragment,
        output=args.output,
    )
    print(json.dumps({
        "output": str(args.output), "status": decision["status"],
        "failure_stage": decision["failure_stage"], "classification": decision["classification"],
        "train_accepted_pairs": decision["formal_build"]["train_generation"]["accepted_pairs"],
        "public_test_read": decision["public_test"]["read"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
