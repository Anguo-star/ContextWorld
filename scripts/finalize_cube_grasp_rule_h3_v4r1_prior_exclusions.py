#!/usr/bin/env python3
"""Finalize Cube v4r1 recovery exclusions before its only recovery build.

The original scientific protocol remains Cube v4.  This receipt creates a
new, non-scientific recovery authorization namespace after the first formal
v4 build failed while committing a Lance manifest.  It unions the complete
old-v4 prior with every scientifically inspectable identity from that failed
attempt, including raw query pixels recovered by deterministic replay.

Only the five explicitly supplied authorization/evidence files are read.  No
Lance dataset, source H5 path, Public Test, RGB probe, or model path is opened.
The output is written once with exclusive creation after every input has been
reverified unchanged.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import yaml


SCIENTIFIC_PROTOCOL_ID = "cube_gripper_carry_rule_history3_development_v4"
RECOVERY_AUTHORIZATION_ID = "cube_gripper_carry_h3_development_v4r1"
PREREG_STATUS = "preregistered_before_v4r1_recovery_build"
FREEZE_STATUS = "frozen_before_v4r1_recovery_build"
OUTPUT_STATUS = FREEZE_STATUS

OLD_PRIOR_RECEIPT_ID = "cube_gripper_carry_h3_v4_prior_exclusions_final_v1"
OLD_PRIOR_STATUS = "frozen_before_first_v4_build"
FAILED_RECEIPT_ID = "cube_gripper_carry_h3_v4_failed_formal_attempt_v1"
FAILED_RECEIPT_STATUS = "infrastructure_failed_immutable_attempt"
QUERY_RECEIPT_ID = (
    "cube_gripper_carry_h3_v4_failed_attempt_query_reconstruction_v1"
)
QUERY_RECEIPT_STATUS = "failed_attempt_content_frozen_for_future_prior_exclusion"
OUTPUT_RECEIPT_ID = "cube_gripper_carry_h3_v4r1_prior_exclusions_final_v1"

EXPECTED_OLD_PRIOR_SHA256 = (
    "8c181529c3012cf89ecf8390d595093d256449d909c5e911297f78ed997161b4"
)
EXPECTED_FAILED_ATTEMPT_SHA256 = (
    "5f20da08a538f2fd0c72c5c172e64cb2359a2e5bdad1746cf2c4249bbf739936"
)
EXPECTED_QUERY_RECONSTRUCTION_SHA256 = (
    "a85c3343464cbbea5c13ac167d419c87bbd5b8ce942767900af171db8474e5e0"
)

EXPECTED_OLD_SOURCE_COUNT = 2321
EXPECTED_OLD_CONTENT_COUNTS = {
    "action_profile_ids": 2322,
    "scene_template_content_hashes": 2330,
    "pair_content_hashes": 2330,
    "query_pixel_hashes": 2330,
}
EXPECTED_FAILED_COUNT = 2048
EXPECTED_FINAL_SOURCE_COUNT = 4369
EXPECTED_FINAL_CONTENT_COUNTS = {
    "action_profile_ids": 4370,
    "scene_template_content_hashes": 4378,
    "pair_content_hashes": 4378,
    "query_pixel_hashes": 4378,
}
EXPECTED_FINAL_SOURCE_SHA256 = (
    "a2167602269492d464e7f07b2a4c1c8ba3e8c46fc1df4791ba69cd0e6027a021"
)
EXPECTED_FINAL_CONTENT_SHA256 = {
    "action_profile_ids": (
        "a65e5534e0db40617126e5c916c650b273e7554247e145bdd3b5bf28a36c3b16"
    ),
    "scene_template_content_hashes": (
        "a5437c01f480e3ad6a22b90f2d31f8cda9bec2a029889fc0ffc8794ba7d89dbc"
    ),
    "pair_content_hashes": (
        "58404c522605e0129d4c3a59680e4a8143a9eb2d651a05d34c4dc5ebd37826f7"
    ),
    "query_pixel_hashes": (
        "7a54a31c301b780af492153122eaaa095dfc9af384d95bb5a4875c2795f05b4e"
    ),
}

CONTENT_FIELDS = (
    "action_profile_ids",
    "scene_template_content_hashes",
    "pair_content_hashes",
    "query_pixel_hashes",
)
DIRECT_FAILED_CONTENT_FIELDS = CONTENT_FIELDS[:3]
OLD_COVERAGE_FIELDS = (
    "v3_formal",
    "v3_smokes",
    "v3_pilots",
    "v4_preformal_smokes_and_pilots",
)
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
FORENSIC_QUERY_JPEG_DIGEST_NAMESPACE = (
    "contextworld-cube-failed-attempt-forensic-query-jpeg-v1"
)
FORBIDDEN_PUBLIC_COMPONENTS = {
    "validation",
    "validation.lance",
    "public",
    "public_test",
    "public-test",
    "publictest",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def excluded_source_episodes_sha256(values: Sequence[int]) -> str:
    normalized = [int(value) for value in values]
    if normalized != sorted(set(normalized)) or any(value < 0 for value in normalized):
        raise ValueError("source episodes must be nonnegative, sorted, and unique")
    payload = b"".join(value.to_bytes(8, "little", signed=True) for value in normalized)
    return hashlib.sha256(
        b"contextworld-cube-prior-source-episodes-v1\0" + payload
    ).hexdigest()


def canonical_content_digest(values: Sequence[str], *, field_name: str) -> str:
    normalized = list(values)
    if normalized != sorted(set(normalized)):
        raise ValueError(f"{field_name} must be sorted and unique")
    decoded: list[bytes] = []
    for value in normalized:
        if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
            raise ValueError(f"{field_name} contains a non-SHA256 value")
        decoded.append(bytes.fromhex(value))
    return hashlib.sha256(
        b"contextworld-cube-prior-content-exclusions-v1\0"
        + field_name.encode("ascii")
        + b"\0"
        + b"".join(decoded)
    ).hexdigest()


def forensic_query_jpeg_digest(values: Sequence[str]) -> str:
    normalized = list(values)
    if normalized != sorted(set(normalized)):
        raise ValueError("query JPEG hashes must be sorted and unique")
    decoded: list[bytes] = []
    for value in normalized:
        if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
            raise ValueError("query JPEG identity is not a SHA256")
        decoded.append(bytes.fromhex(value))
    return hashlib.sha256(
        FORENSIC_QUERY_JPEG_DIGEST_NAMESPACE.encode("ascii")
        + b"\0"
        + b"".join(decoded)
    ).hexdigest()


def _reject_public_path(value: Path | str, *, label: str) -> None:
    component = next(
        (
            part
            for part in Path(value).parts
            if part.lower() in FORBIDDEN_PUBLIC_COMPONENTS
        ),
        None,
    )
    if component is not None:
        raise RuntimeError(f"{label} contains forbidden Public component {component!r}")


def _read_bytes_nofollow(path: Path, *, label: str) -> bytes:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"{label} must be a regular non-symlink file: {path}")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(descriptor, "rb") as stream:
        return stream.read()


def _read_json_nofollow(path: Path, *, label: str) -> tuple[bytes, dict[str, Any]]:
    raw = _read_bytes_nofollow(path, label=label)
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{label} is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} root must be an object")
    return raw, value


def _read_yaml_nofollow(path: Path, *, label: str) -> tuple[bytes, dict[str, Any]]:
    raw = _read_bytes_nofollow(path, label=label)
    try:
        value = yaml.safe_load(raw.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise RuntimeError(f"{label} is not valid UTF-8 YAML") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} root must be an object")
    return raw, value


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{label} must be an object")
    return value


def _closed_public(value: Any, *, label: str) -> None:
    gate = _mapping(value, label=label)
    if gate.get("access_status") != "closed_not_read_not_scored":
        raise RuntimeError(f"{label} access status is not closed")
    for key in ("opened", "read", "hashed", "scored"):
        if gate.get(key) is not False:
            raise RuntimeError(f"{label}.{key} is not false")


def _identity_from_raw(path: Path, raw: bytes) -> dict[str, Any]:
    return {
        "path": path.as_posix(),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
    }


def _identity_core(value: Mapping[str, Any], *, label: str) -> tuple[str, int]:
    digest = value.get("sha256")
    size = value.get("size_bytes")
    if not isinstance(digest, str) or SHA256_PATTERN.fullmatch(digest) is None:
        raise RuntimeError(f"{label}.sha256 is invalid")
    try:
        normalized_size = int(size)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"{label}.size_bytes is invalid") from error
    if normalized_size <= 0:
        raise RuntimeError(f"{label}.size_bytes is invalid")
    return digest, normalized_size


def _require_binding(
    document: Mapping[str, Any],
    *,
    aliases: Sequence[str],
    expected: Mapping[str, Any],
    label: str,
) -> Mapping[str, Any]:
    containers: list[Mapping[str, Any]] = [document]
    for key in (
        "recovery_inputs",
        "authorization_inputs",
        "input_identities",
        "frozen_evidence",
    ):
        value = document.get(key)
        if isinstance(value, Mapping):
            containers.append(value)
    matches: list[Mapping[str, Any]] = []
    for container in containers:
        for alias in aliases:
            value = container.get(alias)
            if isinstance(value, Mapping):
                matches.append(value)
    if not matches:
        raise RuntimeError(f"{label} binding is missing")
    expected_core = _identity_core(expected, label=f"expected {label}")
    for match in matches:
        if _identity_core(match, label=label) != expected_core:
            raise RuntimeError(f"{label} binding mismatch")
    return matches[0]


def _require_source_binding(
    document: Mapping[str, Any], *, expected: Mapping[str, Any], label: str
) -> Mapping[str, Any]:
    binding = _require_binding(
        document,
        aliases=("source_h5", "frozen_source_h5"),
        expected=expected,
        label=label,
    )
    for key in ("symbol", "row_count", "episode_count"):
        if binding.get(key) != expected.get(key):
            raise RuntimeError(f"{label}.{key} mismatch")
    if binding.get("path_recorded") not in (None, False) or binding.get("path") not in (
        None,
        "",
    ):
        raise RuntimeError(f"{label} records a source path")
    return binding


def _validate_source_entry(
    entry: Any, *, label: str, expected_count: int
) -> list[int]:
    value = _mapping(entry, label=label)
    values = [int(item) for item in value.get("values", [])]
    if values != sorted(set(values)) or any(item < 0 for item in values):
        raise RuntimeError(f"{label} values are invalid")
    if len(values) != expected_count or int(value.get("count", -1)) != len(values):
        raise RuntimeError(f"{label} count mismatch")
    if value.get("sha256") != excluded_source_episodes_sha256(values):
        raise RuntimeError(f"{label} digest mismatch")
    return values


def _validate_content_entry(
    entry: Any, *, field_name: str, expected_count: int
) -> list[str]:
    value = _mapping(entry, label=field_name)
    values = [str(item) for item in value.get("values", [])]
    if values != sorted(set(values)) or len(values) != expected_count:
        raise RuntimeError(f"{field_name} values/count mismatch")
    if int(value.get("count", -1)) != len(values):
        raise RuntimeError(f"{field_name} declared count mismatch")
    if value.get("sha256") != canonical_content_digest(
        values, field_name=field_name
    ):
        raise RuntimeError(f"{field_name} digest mismatch")
    return values


def _validate_source_identity(source: Any, *, label: str) -> dict[str, Any]:
    value = _mapping(source, label=label)
    core = _identity_core(value, label=label)
    result = {
        "symbol": value.get("symbol"),
        "sha256": core[0],
        "size_bytes": core[1],
        "row_count": int(value.get("row_count", -1)),
        "episode_count": int(value.get("episode_count", -1)),
    }
    if (
        result["symbol"] != "upstream_cube_single_expert_h5"
        or result["row_count"] <= 0
        or result["episode_count"] <= 0
        or value.get("path_recorded") not in (None, False)
        or value.get("path") not in (None, "")
    ):
        raise RuntimeError(f"{label} source identity contract mismatch")
    return result


def _load_old_prior(
    path: Path,
) -> tuple[bytes, dict[str, Any], list[int], dict[str, list[str]], dict[str, Any]]:
    raw, receipt = _read_json_nofollow(path, label="old final prior")
    if hashlib.sha256(raw).hexdigest() != EXPECTED_OLD_PRIOR_SHA256:
        raise RuntimeError("old final prior canonical SHA256 mismatch")
    if (
        receipt.get("schema_version") != 1
        or receipt.get("protocol_id") != SCIENTIFIC_PROTOCOL_ID
        or receipt.get("receipt_id") != OLD_PRIOR_RECEIPT_ID
        or receipt.get("status") != OLD_PRIOR_STATUS
        or receipt.get("checks_passed") is not True
    ):
        raise RuntimeError("old final prior identity/status mismatch")
    _closed_public(receipt.get("public_test"), label="old prior Public Test")
    if receipt.get("reference_model_training_or_scoring") is not False:
        raise RuntimeError("old final prior reports reference-model work")
    coverage = _mapping(receipt.get("coverage"), label="old prior coverage")
    if set(coverage) != set(OLD_COVERAGE_FIELDS) or any(
        coverage.get(field) is not True for field in OLD_COVERAGE_FIELDS
    ):
        raise RuntimeError("old final prior coverage mismatch")
    source = _validate_source_identity(receipt.get("source_h5"), label="old source H5")
    episodes = [int(value) for value in receipt.get("excluded_source_episodes", [])]
    if (
        episodes != sorted(set(episodes))
        or len(episodes) != EXPECTED_OLD_SOURCE_COUNT
        or int(receipt.get("excluded_source_episode_count", -1)) != len(episodes)
        or receipt.get("excluded_source_episodes_sha256")
        != excluded_source_episodes_sha256(episodes)
    ):
        raise RuntimeError("old final prior source set mismatch")
    if not episodes or episodes[-1] >= source["episode_count"]:
        raise RuntimeError("old final prior source set is outside source H5")
    raw_content = _mapping(
        receipt.get("prior_content_exclusions"), label="old prior content"
    )
    if set(raw_content) != set(CONTENT_FIELDS):
        raise RuntimeError("old final prior content fields mismatch")
    content = {
        field: _validate_content_entry(
            raw_content[field],
            field_name=field,
            expected_count=EXPECTED_OLD_CONTENT_COUNTS[field],
        )
        for field in CONTENT_FIELDS
    }
    artifacts = receipt.get("input_artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise RuntimeError("old final prior input_artifacts are missing")
    for index, artifact in enumerate(artifacts):
        item = _mapping(artifact, label=f"old input_artifacts[{index}]")
        _identity_core(item, label=f"old input_artifacts[{index}]")
        if not isinstance(item.get("role"), str) or not isinstance(item.get("path"), str):
            raise RuntimeError("old final prior has malformed input artifact")
        _reject_public_path(str(item["path"]), label="old input artifact")
    return raw, receipt, episodes, content, source


def _validate_failed_pairs(
    pairs: Any,
    *,
    source_values: Sequence[int],
    content: Mapping[str, Sequence[str]],
) -> tuple[list[dict[str, Any]], list[str]]:
    if not isinstance(pairs, list) or len(pairs) != EXPECTED_FAILED_COUNT:
        raise RuntimeError("failed-attempt pair records count mismatch")
    normalized: list[dict[str, Any]] = []
    jpeg_values: list[str] = []
    pair_ids: list[str] = []
    collected_source: set[int] = set()
    collected = {field: set() for field in DIRECT_FAILED_CONTENT_FIELDS}
    for index, raw_pair in enumerate(pairs):
        pair = dict(_mapping(raw_pair, label=f"failed pair {index}"))
        pair_id = pair.get("pair_id")
        if not isinstance(pair_id, str):
            raise RuntimeError("failed-attempt pair ID is invalid")
        pair_ids.append(pair_id)
        collected_source.add(int(pair.get("source_episode", -1)))
        for field in DIRECT_FAILED_CONTENT_FIELDS:
            value = pair.get(
                {
                    "action_profile_ids": "action_profile_id",
                    "scene_template_content_hashes": "scene_template_content_hash",
                    "pair_content_hashes": "pair_content_hash",
                }[field]
            )
            if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
                raise RuntimeError(f"failed pair {pair_id} has invalid {field}")
            collected[field].add(value)
        jpeg = pair.get("query_jpeg_sha256")
        if not isinstance(jpeg, str) or SHA256_PATTERN.fullmatch(jpeg) is None:
            raise RuntimeError(f"failed pair {pair_id} has invalid query JPEG")
        jpeg_values.append(jpeg)
        normalized.append(pair)
    if pair_ids != sorted(set(pair_ids)):
        raise RuntimeError("failed-attempt pair IDs are not sorted/unique")
    if collected_source != set(source_values):
        raise RuntimeError("failed-attempt pair/source set mismatch")
    for field in DIRECT_FAILED_CONTENT_FIELDS:
        if collected[field] != set(content[field]):
            raise RuntimeError(f"failed-attempt pair/{field} set mismatch")
    if len(set(jpeg_values)) != EXPECTED_FAILED_COUNT:
        raise RuntimeError("failed-attempt query JPEG identities are not unique")
    return normalized, sorted(jpeg_values)


def _load_failed_attempt(
    path: Path,
    *,
    old_prior_identity: Mapping[str, Any],
    old_prior: Mapping[str, Any],
    source: Mapping[str, Any],
) -> tuple[bytes, dict[str, Any], list[int], dict[str, list[str]], list[dict[str, Any]]]:
    raw, receipt = _read_json_nofollow(path, label="failed formal attempt")
    if hashlib.sha256(raw).hexdigest() != EXPECTED_FAILED_ATTEMPT_SHA256:
        raise RuntimeError("failed formal attempt canonical SHA256 mismatch")
    if (
        receipt.get("schema_version") != 1
        or receipt.get("protocol_id") != SCIENTIFIC_PROTOCOL_ID
        or receipt.get("receipt_id") != FAILED_RECEIPT_ID
        or receipt.get("status") != FAILED_RECEIPT_STATUS
        or receipt.get("checks_passed") is not True
        or receipt.get("build_passed") is not False
        or receipt.get("formal_build_attempt_consumed") is not True
        or receipt.get("retry_authorized_under_original_preregistration") is not False
    ):
        raise RuntimeError("failed formal attempt identity/budget mismatch")
    scope = _mapping(receipt.get("scope"), label="failed-attempt scope")
    _closed_public(scope.get("public_test"), label="failed-attempt Public Test")
    if (
        scope.get("rgb_probe_run") is not False
        or scope.get("reference_model_training_or_scoring") is not False
        or int(scope.get("optimizer_steps", -1)) != 0
    ):
        raise RuntimeError("failed attempt reports probe/model work")
    policy = _mapping(receipt.get("recovery_policy"), label="failed recovery policy")
    for key in (
        "original_v4_preregistration_attempt_budget_exhausted",
        "original_failed_tree_must_remain_immutable",
        "silent_retry_or_overwrite_forbidden",
        "newly_frozen_recovery_preregistration_required",
        "failed_source_action_scene_pair_and_reconstructed_raw_query_must_be_excluded",
    ):
        if policy.get(key) is not True:
            raise RuntimeError(f"failed recovery policy {key} is not true")
    inputs = _mapping(receipt.get("input_identities"), label="failed inputs")
    _require_binding(
        inputs,
        aliases=("prior_exclusion_receipt", "old_final_prior_receipt"),
        expected=old_prior_identity,
        label="failed-attempt old prior",
    )
    _require_source_binding(inputs, expected=source, label="failed-attempt source H5")
    for old_name, failed_name in (
        ("preregistration", "preregistration"),
        ("freeze_receipt", "freeze_receipt"),
    ):
        _require_binding(
            inputs,
            aliases=(failed_name,),
            expected=_mapping(old_prior.get(old_name), label=f"old prior {old_name}"),
            label=f"failed-attempt original {old_name}",
        )
    content_doc = _mapping(
        receipt.get("failed_attempt_content"), label="failed-attempt content"
    )
    if (
        content_doc.get("split") != "train"
        or int(content_doc.get("row_count", -1)) != 8 * EXPECTED_FAILED_COUNT
        or int(content_doc.get("episode_count", -1)) != 2 * EXPECTED_FAILED_COUNT
        or int(content_doc.get("pair_count", -1)) != EXPECTED_FAILED_COUNT
        or content_doc.get("query_pixel_hash_status")
        != "pending_deterministic_raw_reconstruction_not_present_in_fragment"
    ):
        raise RuntimeError("failed-attempt content cardinality/status mismatch")
    episodes = _validate_source_entry(
        content_doc.get("source_episodes"),
        label="failed source episodes",
        expected_count=EXPECTED_FAILED_COUNT,
    )
    raw_content = _mapping(
        content_doc.get("prior_content_exclusions"),
        label="failed direct content",
    )
    if set(raw_content) != set(DIRECT_FAILED_CONTENT_FIELDS):
        raise RuntimeError("failed direct content fields mismatch")
    content = {
        field: _validate_content_entry(
            raw_content[field], field_name=field, expected_count=EXPECTED_FAILED_COUNT
        )
        for field in DIRECT_FAILED_CONTENT_FIELDS
    }
    pairs, jpeg_values = _validate_failed_pairs(
        content_doc.get("pairs"), source_values=episodes, content=content
    )
    constraints = _mapping(
        content_doc.get("profile_constraints"),
        label="failed-attempt profile constraints",
    )
    if constraints.get("passed") is not True:
        raise RuntimeError("failed-attempt profile constraints did not pass")
    jpeg_entry = _mapping(
        content_doc.get("query_jpeg_sha256"), label="failed query JPEG set"
    )
    if (
        [str(value) for value in jpeg_entry.get("values", [])] != jpeg_values
        or int(jpeg_entry.get("count", -1)) != EXPECTED_FAILED_COUNT
        or jpeg_entry.get("sha256") != forensic_query_jpeg_digest(jpeg_values)
        or jpeg_entry.get("digest_namespace")
        != FORENSIC_QUERY_JPEG_DIGEST_NAMESPACE
        or jpeg_entry.get("role")
        != "forensic_binding_only_not_raw_query_pixel_hash"
    ):
        raise RuntimeError("failed query JPEG forensic binding mismatch")
    overlap = _mapping(content_doc.get("prior_overlap"), label="failed prior overlap")
    for key in (
        "source_episode_count",
        "action_profile_id_count",
        "scene_template_content_hash_count",
        "pair_content_hash_count",
    ):
        if int(overlap.get(key, -1)) != 0:
            raise RuntimeError(f"failed receipt reports old-prior overlap: {key}")
    if (
        overlap.get("query_pixel_hash_count", "missing") is not None
        or overlap.get("query_pixel_hash_overlap_status")
        != "not_computable_until_raw_query_reconstruction"
        or overlap.get("passed_for_directly_inspectable_identities") is not True
    ):
        raise RuntimeError("failed receipt raw-query overlap state mismatch")
    return raw, receipt, episodes, content, pairs


def _load_query_reconstruction(
    path: Path,
    *,
    failed_identity: Mapping[str, Any],
    failed_receipt: Mapping[str, Any],
    failed_episodes: Sequence[int],
    failed_content: Mapping[str, Sequence[str]],
    failed_pairs: Sequence[Mapping[str, Any]],
    old_prior_identity: Mapping[str, Any],
    source: Mapping[str, Any],
) -> tuple[bytes, dict[str, Any], list[int], dict[str, list[str]]]:
    raw, receipt = _read_json_nofollow(path, label="query reconstruction")
    if hashlib.sha256(raw).hexdigest() != EXPECTED_QUERY_RECONSTRUCTION_SHA256:
        raise RuntimeError("query reconstruction canonical SHA256 mismatch")
    if (
        receipt.get("schema_version") != 1
        or receipt.get("protocol_id") != SCIENTIFIC_PROTOCOL_ID
        or receipt.get("receipt_id") != QUERY_RECEIPT_ID
        or receipt.get("status") != QUERY_RECEIPT_STATUS
        or receipt.get("checks_passed") is not True
    ):
        raise RuntimeError("query reconstruction identity/status mismatch")
    _require_binding(
        receipt,
        aliases=("failed_attempt_receipt",),
        expected=failed_identity,
        label="query reconstruction failed attempt",
    )
    inputs = _mapping(receipt.get("input_identities"), label="query inputs")
    _require_binding(
        inputs,
        aliases=("failed_attempt_receipt",),
        expected=failed_identity,
        label="query input failed attempt",
    )
    _require_binding(
        inputs,
        aliases=("prior_exclusion_receipt", "old_final_prior_receipt"),
        expected=old_prior_identity,
        label="query input old prior",
    )
    _require_source_binding(inputs, expected=source, label="query source H5")
    failed_inputs = _mapping(
        failed_receipt.get("input_identities"), label="failed inputs"
    )
    for name in (
        "preregistration",
        "freeze_receipt",
        "prior_exclusion_receipt",
        "builder_snapshot",
        "request_json",
        "partial_train_fragment",
        "source_h5",
    ):
        if name not in inputs or name not in failed_inputs:
            raise RuntimeError(f"query/failed input {name} is missing")
        if _identity_core(
            _mapping(inputs[name], label=f"query input {name}"),
            label=f"query input {name}",
        ) != _identity_core(
            _mapping(failed_inputs[name], label=f"failed input {name}"),
            label=f"failed input {name}",
        ):
            raise RuntimeError(f"query/failed input binding mismatch: {name}")
    _closed_public(receipt.get("public_test"), label="query Public Test")
    rgb = _mapping(receipt.get("rgb_probe"), label="query RGB probe")
    if any(rgb.get(key) is not False for key in ("opened", "run", "scored")):
        raise RuntimeError("query reconstruction reports RGB probe work")
    if (
        receipt.get("reference_model_training_or_scoring") is not False
        or int(receipt.get("reference_model_optimizer_steps", -1)) != 0
    ):
        raise RuntimeError("query reconstruction reports model work")
    contract = _mapping(
        receipt.get("reconstruction_contract"), label="query reconstruction contract"
    )
    for key in (
        "jpeg_reencoding_bitwise_equal_to_fragment",
        "reencoded_query_jpegs_match_fragment",
        "stored_paired_query_jpegs_equal",
        "raw_query_hashes_unique",
        "raw_query_prior_overlap_zero",
        "builder_snapshot_loaded_by_explicit_path",
        "physics_snapshot_loaded_by_explicit_path",
        "all_inputs_reverified_unchanged_after_replay",
    ):
        if contract.get(key) is not True:
            raise RuntimeError(f"query reconstruction contract {key} is not true")
    if (
        contract.get("dataset_manifest_opened") is not False
        or contract.get("lance_written") is not False
        or contract.get("replayed_mode") != "cannot_hold_only"
        or int(contract.get("query_model_step_idx", -1)) != 2
        or int(contract.get("jpeg_quality", -1)) != 95
    ):
        raise RuntimeError("query reconstruction scope/recipe mismatch")
    content_doc = _mapping(
        receipt.get("failed_attempt_content"), label="query failed content"
    )
    if (
        content_doc.get("split") != "train"
        or int(content_doc.get("row_count", -1)) != 8 * EXPECTED_FAILED_COUNT
        or int(content_doc.get("episode_count", -1)) != 2 * EXPECTED_FAILED_COUNT
        or int(content_doc.get("pair_count", -1)) != EXPECTED_FAILED_COUNT
    ):
        raise RuntimeError("query failed-content cardinality mismatch")
    episodes = _validate_source_entry(
        content_doc.get("source_episodes"),
        label="query source episodes",
        expected_count=EXPECTED_FAILED_COUNT,
    )
    if episodes != list(failed_episodes):
        raise RuntimeError("query/failed source episodes mismatch")
    raw_content = _mapping(
        content_doc.get("prior_content_exclusions"), label="query content sets"
    )
    if set(raw_content) != set(CONTENT_FIELDS):
        raise RuntimeError("query content fields mismatch")
    content = {
        field: _validate_content_entry(
            raw_content[field], field_name=field, expected_count=EXPECTED_FAILED_COUNT
        )
        for field in CONTENT_FIELDS
    }
    for field in DIRECT_FAILED_CONTENT_FIELDS:
        if content[field] != list(failed_content[field]):
            raise RuntimeError(f"query/failed {field} mismatch")
    query_pairs = content_doc.get("pairs")
    if not isinstance(query_pairs, list) or len(query_pairs) != EXPECTED_FAILED_COUNT:
        raise RuntimeError("query pair records count mismatch")
    raw_queries: list[str] = []
    for failed_pair, raw_query_pair in zip(failed_pairs, query_pairs):
        pair = dict(_mapping(raw_query_pair, label="query pair"))
        raw_hash = pair.pop("raw_query_pixel_hash", None)
        split = pair.pop("split", None)
        if split != "train" or pair != dict(failed_pair):
            raise RuntimeError("query pair does not bind failed-attempt pair")
        if not isinstance(raw_hash, str) or SHA256_PATTERN.fullmatch(raw_hash) is None:
            raise RuntimeError("query pair raw hash is invalid")
        raw_queries.append(raw_hash)
    if sorted(raw_queries) != content["query_pixel_hashes"] or len(
        set(raw_queries)
    ) != EXPECTED_FAILED_COUNT:
        raise RuntimeError("query pair/raw-query set mismatch")
    overlap = _mapping(receipt.get("prior_overlap"), label="query prior overlap")
    if overlap.get("passed") is not True:
        raise RuntimeError("query prior overlap gate failed")
    for field in ("source_episode", *CONTENT_FIELDS):
        entry = _mapping(overlap.get(field), label=f"query overlap {field}")
        if int(entry.get("count", -1)) != 0 or entry.get("values") != []:
            raise RuntimeError(f"query receipt reports prior overlap: {field}")
    return raw, receipt, episodes, content


def _validate_authorization_document(
    document: Mapping[str, Any],
    *,
    label: str,
    expected_status: str,
    old_prior_identity: Mapping[str, Any],
    failed_identity: Mapping[str, Any],
    query_identity: Mapping[str, Any],
    source: Mapping[str, Any],
    prereg_identity: Mapping[str, Any] | None,
) -> None:
    if document.get("schema_version") != 1:
        raise RuntimeError(f"{label} schema mismatch")
    protocol_values = [
        document[key]
        for key in ("scientific_protocol_id", "protocol_id")
        if key in document
    ]
    if not protocol_values or any(
        value != SCIENTIFIC_PROTOCOL_ID for value in protocol_values
    ):
        raise RuntimeError(f"{label} scientific protocol mismatch")
    if document.get("recovery_authorization_id") != RECOVERY_AUTHORIZATION_ID:
        raise RuntimeError(f"{label} recovery authorization mismatch")
    if document.get("status") != expected_status:
        raise RuntimeError(f"{label} status mismatch")
    if label == "v4r1 freeze receipt" and document.get("checks_passed") is not True:
        raise RuntimeError("v4r1 freeze receipt did not pass")
    _closed_public(document.get("public_test"), label=f"{label} Public Test")
    model_keys = (
        "reference_model_training_or_scoring",
        "reference_model_training_or_scoring_authorized",
    )
    present_model_keys = [key for key in model_keys if key in document]
    if not present_model_keys or any(document.get(key) is not False for key in present_model_keys):
        raise RuntimeError(f"{label} reference-model authorization mismatch")
    _require_binding(
        document,
        aliases=("old_final_prior_receipt", "old_final_prior", "prior_exclusion_receipt"),
        expected=old_prior_identity,
        label=f"{label} old final prior",
    )
    _require_binding(
        document,
        aliases=("failed_formal_attempt_receipt", "failed_attempt_receipt"),
        expected=failed_identity,
        label=f"{label} failed attempt",
    )
    _require_binding(
        document,
        aliases=(
            "query_reconstruction_receipt",
            "failed_attempt_query_reconstruction_receipt",
        ),
        expected=query_identity,
        label=f"{label} query reconstruction",
    )
    _require_source_binding(document, expected=source, label=f"{label} source H5")
    if prereg_identity is not None:
        _require_binding(
            document,
            aliases=("preregistration", "v4r1_preregistration"),
            expected=prereg_identity,
            label=f"{label} preregistration",
        )


def _content_receipt(values: Sequence[str], *, field_name: str) -> dict[str, Any]:
    normalized = list(values)
    return {
        "values": normalized,
        "count": len(normalized),
        "sha256": canonical_content_digest(normalized, field_name=field_name),
    }


def _source_receipt(values: Sequence[int]) -> dict[str, Any]:
    normalized = [int(value) for value in values]
    return {
        "values": normalized,
        "count": len(normalized),
        "sha256": excluded_source_episodes_sha256(normalized),
    }


def _verify_unchanged(path: Path, raw: bytes, *, label: str) -> None:
    current = _read_bytes_nofollow(path, label=f"postflight {label}")
    if current != raw:
        raise RuntimeError(f"{label} mutated during finalization")


def write_receipt_exclusive(path: Path, receipt: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite {path}")
    payload = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(payload)


def finalize(
    *,
    old_final_prior: Path,
    failed_attempt_receipt: Path,
    query_reconstruction_receipt: Path,
    prereg_path: Path,
    freeze_receipt_path: Path,
    output: Path,
) -> dict[str, Any]:
    paths = (
        ("old final prior", old_final_prior),
        ("failed attempt receipt", failed_attempt_receipt),
        ("query reconstruction receipt", query_reconstruction_receipt),
        ("v4r1 preregistration", prereg_path),
        ("v4r1 freeze receipt", freeze_receipt_path),
        ("output", output),
    )
    for label, path in paths:
        _reject_public_path(path, label=label)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite {output}")

    old_raw, old, old_episodes, old_content, source = _load_old_prior(
        old_final_prior
    )
    old_identity = _identity_from_raw(old_final_prior, old_raw)
    failed_raw, failed, failed_episodes, failed_content, failed_pairs = (
        _load_failed_attempt(
            failed_attempt_receipt,
            old_prior_identity=old_identity,
            old_prior=old,
            source=source,
        )
    )
    failed_identity = _identity_from_raw(failed_attempt_receipt, failed_raw)
    query_raw, query, query_episodes, query_content = _load_query_reconstruction(
        query_reconstruction_receipt,
        failed_identity=failed_identity,
        failed_receipt=failed,
        failed_episodes=failed_episodes,
        failed_content=failed_content,
        failed_pairs=failed_pairs,
        old_prior_identity=old_identity,
        source=source,
    )
    query_identity = _identity_from_raw(query_reconstruction_receipt, query_raw)
    if query_episodes != failed_episodes:
        raise RuntimeError("failed/query source sets mismatch")

    overlaps: dict[str, list[Any]] = {
        "source_episodes": sorted(set(old_episodes) & set(query_episodes)),
        **{
            field: sorted(set(old_content[field]) & set(query_content[field]))
            for field in CONTENT_FIELDS
        },
    }
    if any(overlaps.values()):
        raise RuntimeError(f"failed formal attempt overlaps old prior: {overlaps}")

    merged_episodes = sorted(set(old_episodes) | set(query_episodes))
    merged_content = {
        field: sorted(set(old_content[field]) | set(query_content[field]))
        for field in CONTENT_FIELDS
    }
    if len(merged_episodes) != EXPECTED_FINAL_SOURCE_COUNT:
        raise RuntimeError("v4r1 final source count mismatch")
    for field in CONTENT_FIELDS:
        if len(merged_content[field]) != EXPECTED_FINAL_CONTENT_COUNTS[field]:
            raise RuntimeError(f"v4r1 final {field} count mismatch")
    if merged_episodes[-1] >= source["episode_count"]:
        raise RuntimeError("v4r1 source exclusion is outside source H5")
    calculated_source_digest = excluded_source_episodes_sha256(merged_episodes)
    if calculated_source_digest != EXPECTED_FINAL_SOURCE_SHA256:
        raise RuntimeError("v4r1 final source digest mismatch")
    calculated_content_digests = {
        field: canonical_content_digest(merged_content[field], field_name=field)
        for field in CONTENT_FIELDS
    }
    if calculated_content_digests != EXPECTED_FINAL_CONTENT_SHA256:
        raise RuntimeError("v4r1 final content digest mismatch")

    prereg_raw, prereg = _read_yaml_nofollow(
        prereg_path, label="v4r1 preregistration"
    )
    prereg_identity = _identity_from_raw(prereg_path, prereg_raw)
    _validate_authorization_document(
        prereg,
        label="v4r1 preregistration",
        expected_status=PREREG_STATUS,
        old_prior_identity=old_identity,
        failed_identity=failed_identity,
        query_identity=query_identity,
        source=source,
        prereg_identity=None,
    )
    freeze_raw, freeze = _read_json_nofollow(
        freeze_receipt_path, label="v4r1 freeze receipt"
    )
    freeze_identity = _identity_from_raw(freeze_receipt_path, freeze_raw)
    _validate_authorization_document(
        freeze,
        label="v4r1 freeze receipt",
        expected_status=FREEZE_STATUS,
        old_prior_identity=old_identity,
        failed_identity=failed_identity,
        query_identity=query_identity,
        source=source,
        prereg_identity=prereg_identity,
    )

    # The old evidence list remains byte-for-byte equivalent as JSON values;
    # only the two immutable failed-attempt receipts are appended.
    input_artifacts = copy.deepcopy(old["input_artifacts"])
    input_artifacts.extend(
        (
            {
                "role": "v4_failed_formal_attempts",
                "artifact_kind": "failed_formal_attempt_receipt",
                **failed_identity,
                "source_episode_count": EXPECTED_FAILED_COUNT,
                "content_counts": {
                    field: EXPECTED_FAILED_COUNT
                    for field in DIRECT_FAILED_CONTENT_FIELDS
                },
                "raw_query_pixel_hash_status": "reconstructed_in_companion_receipt",
                "public_test_opened_read_hashed_or_scored": False,
                "reference_model_training_or_scoring": False,
            },
            {
                "role": "v4_failed_formal_attempts",
                "artifact_kind": "failed_attempt_query_reconstruction_receipt",
                **query_identity,
                "source_episode_count": EXPECTED_FAILED_COUNT,
                "content_counts": {
                    field: EXPECTED_FAILED_COUNT for field in CONTENT_FIELDS
                },
                "raw_query_pixel_hashes_complete": True,
                "public_test_opened_read_hashed_or_scored": False,
                "reference_model_training_or_scoring": False,
            },
        )
    )
    content_receipt = {
        field: _content_receipt(merged_content[field], field_name=field)
        for field in CONTENT_FIELDS
    }
    source_set_receipt = _source_receipt(merged_episodes)
    receipt = {
        "schema_version": 1,
        "protocol_id": SCIENTIFIC_PROTOCOL_ID,
        "scientific_protocol_id": SCIENTIFIC_PROTOCOL_ID,
        "recovery_authorization_id": RECOVERY_AUTHORIZATION_ID,
        "receipt_id": OUTPUT_RECEIPT_ID,
        "status": OUTPUT_STATUS,
        "checks_passed": True,
        "preregistration": prereg_identity,
        "freeze_receipt": freeze_identity,
        "old_final_prior_receipt": old_identity,
        "failed_formal_attempt_receipt": failed_identity,
        "query_reconstruction_receipt": query_identity,
        "source_h5": source,
        "coverage": {
            **{field: True for field in OLD_COVERAGE_FIELDS},
            "v4_failed_formal_attempts": True,
        },
        "input_artifacts": input_artifacts,
        "excluded_source_episodes": source_set_receipt["values"],
        "excluded_source_episode_count": source_set_receipt["count"],
        "excluded_source_episodes_sha256": source_set_receipt["sha256"],
        "prior_content_exclusions": content_receipt,
        "failed_attempt_old_prior_overlap": {
            "source_episodes": {"count": 0, "values": []},
            **{
                field: {"count": 0, "values": []} for field in CONTENT_FIELDS
            },
            "passed": True,
        },
        "recovery_contract": {
            "scientific_protocol_unchanged_from_v4": True,
            "old_prior_preserved_and_extended": True,
            "failed_attempt_source_action_scene_pair_query_all_excluded": True,
            "failed_attempt_raw_queries_deterministically_reconstructed": True,
            "all_inputs_reverified_unchanged_before_output": True,
            "original_v4_attempt_not_retried_or_overwritten": True,
            "lance_opened_or_written": False,
            "public_test_opened_read_hashed_or_scored": False,
            "rgb_probe_run": False,
            "reference_model_training_or_scoring": False,
        },
        "formal_build_requirement": (
            "the sole v4r1 recovery Training/Development build must use this "
            "newest complete prior-exclusion receipt"
        ),
        "public_test": {
            "access_status": "closed_not_read_not_scored",
            "opened": False,
            "read": False,
            "hashed": False,
            "scored": False,
        },
        "rgb_probe": {"opened": False, "run": False, "scored": False},
        "reference_model_training_or_scoring": False,
        "reference_model_optimizer_steps": 0,
    }

    for label, path, raw in (
        ("old final prior", old_final_prior, old_raw),
        ("failed attempt receipt", failed_attempt_receipt, failed_raw),
        ("query reconstruction receipt", query_reconstruction_receipt, query_raw),
        ("v4r1 preregistration", prereg_path, prereg_raw),
        ("v4r1 freeze receipt", freeze_receipt_path, freeze_raw),
    ):
        _verify_unchanged(path, raw, label=label)
    write_receipt_exclusive(output, receipt)
    return receipt


def parse_args(values: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-final-prior", type=Path, required=True)
    parser.add_argument("--failed-attempt-receipt", type=Path, required=True)
    parser.add_argument("--query-reconstruction-receipt", type=Path, required=True)
    parser.add_argument("--prereg", type=Path, required=True)
    parser.add_argument("--freeze-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(values)


def main(values: Sequence[str] | None = None) -> None:
    args = parse_args(values)
    receipt = finalize(
        old_final_prior=args.old_final_prior.expanduser().resolve(),
        failed_attempt_receipt=args.failed_attempt_receipt.expanduser().resolve(),
        query_reconstruction_receipt=(
            args.query_reconstruction_receipt.expanduser().resolve()
        ),
        prereg_path=args.prereg.expanduser().resolve(),
        freeze_receipt_path=args.freeze_receipt.expanduser().resolve(),
        output=args.output.expanduser().resolve(),
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "checks_passed": receipt["checks_passed"],
                "status": receipt["status"],
                "source_episode_count": receipt["excluded_source_episode_count"],
                "content_counts": {
                    field: receipt["prior_content_exclusions"][field]["count"]
                    for field in CONTENT_FIELDS
                },
                "public_test_read": False,
                "reference_model_training_or_scoring": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
