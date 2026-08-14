#!/usr/bin/env python3
"""Seal the Cube v4r1 Development data-readiness decision.

This is a result-packaging command.  It revalidates the frozen recovery
authorization, the complete formal publication, and the one-shot RGB-history
probe before writing one decision with exclusive creation.  It never opens a
Lance table, a Public Test path, a model, or a checkpoint.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
from typing import Any, Mapping, Sequence

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contextworld.paths import artifact_path, portable_contextworld_path  # noqa: E402


PROTOCOL = "cube_gripper_carry_rule_history3_development_v4"
RECOVERY_ID = "cube_gripper_carry_h3_development_v4r1"
DECISION_ID = "cube_gripper_carry_h3_v4r1_data_readiness_20260812"
DECISION_STATUS = "passed_development"
DECISION_SCOPE = "data_readiness_only_not_reference_model_or_public"

DEFAULT_PREREG = ROOT / (
    "configs/benchmark/"
    "cube_gripper_carry_h3_development_recovery_prereg_v4r1.yaml"
)
DEFAULT_FREEZE = artifact_path(
    "evaluation/history3/cube_gripper_carry_h3_development_v4r1/"
    "development_recovery_freeze_receipt_v1.json"
)
DEFAULT_PRIOR = artifact_path(
    "evaluation/history3/cube_gripper_carry_h3_development_v4r1/"
    "prior_episode_exclusions_final_v1.json"
)
DEFAULT_ARTIFACT_ROOT = artifact_path(
    "synthesis/cube_gripper_carry_rule_h3_development_v4r1"
)
DEFAULT_PROBE = artifact_path(
    "evaluation/history3/cube_gripper_carry_h3_development_v4r1/"
    "rgb_history_probe_v1.json"
)
DEFAULT_OUTPUT = artifact_path(
    "evaluation/history3/cube_gripper_carry_h3_development_v4r1/"
    "development_decision.json"
)

EXPECTED_INPUTS: dict[str, tuple[str, int]] = {
    "preregistration": (
        "365c3c94ee6e54617cdcdc529c89e170759e451846083ee9d8b49a00fc31d44e",
        14_474,
    ),
    "freeze_receipt": (
        "21cc822f78eb66e6f73806ade4e067d930aaebb8804e5ec87fa5ddc2e0077e98",
        16_273,
    ),
    "prior_exclusion_receipt": (
        "8bfb8a6a6d87a61c439f7b1d8c2945ec33ff2f9e861050e7d2b6f68d8af0c366",
        1_382_566,
    ),
    "rgb_history_probe": (
        "0739f3d18b0eef1c952bcac5706061c806e4aab00cf69e48559b2206796461e6",
        22_765,
    ),
}
EXPECTED_RELEASE_FILES: dict[str, tuple[str, int]] = {
    "request.json": (
        "9fbb85e6185d153c605ee1f1260529705af815749df7c2c88ccae6c184320e0c",
        10_914,
    ),
    "build_report.json": (
        "98ddf562ec91a2e449ddceb288bceb7f3b765b47dd6f7ebe0b141cde51bd84bf",
        52_453_701,
    ),
    "manifest.json": (
        "2e0e451565da209291e69d7bb388cf0af9bc781c97fca0bd7d135936815ac4bb",
        1_851,
    ),
    "_SUCCESS.json": (
        "95aedc40cdf347846db3c2332aebefb967759ec91e2b18c0f89183acd796516b",
        4_113,
    ),
}
EXPECTED_FAILURE_DECISION = {
    "sha256": "5f159f58eac81894fda36013a309d356a9d80d0b01a7224e43d5880813b2ea75",
    "size_bytes": 2_054_086,
}
EXPECTED_GATES = {
    "action_only_accuracy_at_most_0_51",
    "bootstrap_2_5_percent_lower_bound_at_least_0_70",
    "cross_split_content_isolation_passed",
    "frozen_prior_exclusion_overlap_zero",
    "overall_accuracy_at_least_0_75",
    "paired_x0_query_actions_identical",
    "permuted_label_mean_accuracy_at_most_0_60",
    "query_x2_only_accuracy_at_most_0_51",
    "worst_anchor_accuracy_at_least_0_70",
    "worst_mode_accuracy_at_least_0_70",
    "x0_only_accuracy_at_most_0_51",
}
EXPECTED_RELEASE_DIRECTORIES = {
    "",
    "loader_validation.lance",
    "loader_validation.lance/_transactions",
    "loader_validation.lance/_versions",
    "loader_validation.lance/data",
    "train.lance",
    "train.lance/_transactions",
    "train.lance/_versions",
    "train.lance/data",
}
FORBIDDEN_PUBLIC_COMPONENTS = {
    "validation",
    "validation.lance",
    "public",
    "public_test",
    "public-test",
    "publictest",
}


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{label} must be an object")
    return value


def _reject_public(value: str | Path, *, label: str) -> None:
    component = next(
        (
            part
            for part in Path(value).parts
            if part.lower() in FORBIDDEN_PUBLIC_COMPONENTS
        ),
        None,
    )
    if component is not None:
        raise RuntimeError(
            f"{label} contains forbidden Public component {component!r}"
        )


def _absolute_without_resolve(path: Path) -> Path:
    return Path(os.path.abspath(path.expanduser()))


def _canonical(path: Path, expected: Path, *, label: str) -> Path:
    actual = _absolute_without_resolve(path)
    wanted = _absolute_without_resolve(expected)
    _reject_public(actual, label=label)
    if actual != wanted:
        raise RuntimeError(f"{label} must be the canonical path {wanted}")
    return actual


def _read_bytes(path: Path, *, label: str) -> bytes:
    _reject_public(path, label=label)
    metadata = os.lstat(path)
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError(f"{label} must be a regular non-symlink file")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(descriptor, "rb") as stream:
        return stream.read()


def _json(raw: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{label} is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} root must be an object")
    return value


def _yaml(raw: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = yaml.safe_load(raw.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise RuntimeError(f"{label} is not valid UTF-8 YAML") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} root must be an object")
    return value


def _identity(raw: bytes, *, path: str | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
    }
    if path is not None:
        result["path"] = path
    return result


def _stat_identity(value: os.stat_result) -> tuple[int, int, int]:
    return (value.st_dev, value.st_ino, stat.S_IFMT(value.st_mode))


def _read_fd(descriptor: int) -> bytes:
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    while True:
        chunk = os.read(descriptor, 8 * 1024 * 1024)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)


def _hash_fd(descriptor: int) -> tuple[str, int]:
    os.lseek(descriptor, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    size = 0
    while True:
        chunk = os.read(descriptor, 8 * 1024 * 1024)
        if not chunk:
            return digest.hexdigest(), size
        digest.update(chunk)
        size += len(chunk)


def _require_identity(
    raw: bytes,
    expected: tuple[str, int],
    *,
    label: str,
) -> None:
    observed = (hashlib.sha256(raw).hexdigest(), len(raw))
    if observed != expected:
        raise RuntimeError(
            f"{label} identity mismatch: expected {expected}, got {observed}"
        )


def _same_identity(value: Any, expected: Mapping[str, Any], *, label: str) -> None:
    entry = _mapping(value, label=label)
    if (
        entry.get("sha256") != expected.get("sha256")
        or entry.get("size_bytes") != expected.get("size_bytes")
    ):
        raise RuntimeError(f"{label} identity mismatch")


def _closed_public(value: Any, *, label: str) -> dict[str, Any]:
    public = dict(_mapping(value, label=label))
    if public.get("access_status") != "closed_not_read_not_scored":
        raise RuntimeError(f"{label} is not closed")
    for key in ("generated", "opened", "read", "hashed", "scored"):
        if key in public and public[key] is not False:
            raise RuntimeError(f"{label}.{key} must be false")
    return public


def _zero_overlap(value: Any, *, label: str) -> None:
    entry = _mapping(value, label=label)
    if int(entry.get("count", -1)) != 0 or entry.get("values") != []:
        raise RuntimeError(f"{label} must be exactly empty")


def _validate_prereg(
    prereg: Mapping[str, Any], prereg_identity: Mapping[str, Any]
) -> None:
    if (
        prereg.get("protocol_id") != PROTOCOL
        or prereg.get("scientific_protocol_id") != PROTOCOL
        or prereg.get("recovery_authorization_id") != RECOVERY_ID
        or prereg.get("status")
        != "preregistered_before_v4r1_recovery_build"
    ):
        raise RuntimeError("preregistration protocol/status mismatch")
    if prereg.get("reference_model_training_or_scoring_authorized") is not False:
        raise RuntimeError("preregistration authorizes a reference model")
    phase = _mapping(
        prereg.get("reference_model_phase"), label="preregistration model phase"
    )
    if (
        phase.get("training_and_scoring_authorized") is not False
        or phase.get("trainer_invoked") is not False
        or int(phase.get("optimizer_steps_run", -1)) != 0
        or int(phase.get("optimizer_steps_authorized", -1)) != 0
        or phase.get("checkpoint_creation_authorized") is not False
    ):
        raise RuntimeError("preregistration model phase is not closed")
    _closed_public(prereg.get("public_test"), label="preregistration Public Test")
    planned = _mapping(prereg.get("planned_artifacts"), label="planned artifacts")
    if planned.get("development_decision") != (
        "artifacts/evaluation/history3/"
        "cube_gripper_carry_h3_development_v4r1/development_decision.json"
    ):
        raise RuntimeError("planned decision path mismatch")
    failure = _mapping(
        _mapping(prereg.get("recovery_inputs"), label="recovery inputs").get(
            "infrastructure_failure_decision"
        ),
        label="frozen infrastructure failure decision",
    )
    if {
        "sha256": failure.get("sha256"),
        "size_bytes": failure.get("size_bytes"),
    } != EXPECTED_FAILURE_DECISION:
        raise RuntimeError("original v4 failure evidence identity changed")
    if prereg_identity != {
        "sha256": EXPECTED_INPUTS["preregistration"][0],
        "size_bytes": EXPECTED_INPUTS["preregistration"][1],
    }:
        raise RuntimeError("unexpected preregistration identity")


def _validate_freeze(
    freeze: Mapping[str, Any], prereg_identity: Mapping[str, Any]
) -> None:
    if (
        freeze.get("protocol_id") != PROTOCOL
        or freeze.get("scientific_protocol_id") != PROTOCOL
        or freeze.get("recovery_authorization_id") != RECOVERY_ID
        or freeze.get("status") != "frozen_before_v4r1_recovery_build"
        or freeze.get("checks_passed") is not True
    ):
        raise RuntimeError("freeze receipt protocol/status mismatch")
    _same_identity(
        freeze.get("preregistration"), prereg_identity, label="freeze preregistration"
    )
    if (
        int(freeze.get("recovery_build_attempts_authorized", -1)) != 1
        or int(freeze.get("rgb_history_probe_attempts_authorized", -1)) != 1
        or freeze.get("reference_model_training_or_scoring_authorized") is not False
        or int(freeze.get("reference_model_optimizer_steps_authorized", -1)) != 0
    ):
        raise RuntimeError("freeze attempt/model authorization mismatch")
    _closed_public(freeze.get("public_test"), label="freeze Public Test")
    failure = _mapping(
        _mapping(freeze.get("authorization_inputs"), label="authorization inputs").get(
            "infrastructure_failure_decision"
        ),
        label="freeze failure decision",
    )
    if {
        "sha256": failure.get("sha256"),
        "size_bytes": failure.get("size_bytes"),
    } != EXPECTED_FAILURE_DECISION:
        raise RuntimeError("freeze no longer binds original failure evidence")


def _validate_prior(
    prior: Mapping[str, Any],
    prereg_identity: Mapping[str, Any],
    freeze_identity: Mapping[str, Any],
) -> None:
    if (
        prior.get("protocol_id") != PROTOCOL
        or prior.get("scientific_protocol_id") != PROTOCOL
        or prior.get("recovery_authorization_id") != RECOVERY_ID
        or prior.get("receipt_id")
        != "cube_gripper_carry_h3_v4r1_prior_exclusions_final_v1"
        or prior.get("status") != "frozen_before_v4r1_recovery_build"
        or prior.get("checks_passed") is not True
    ):
        raise RuntimeError("prior exclusion receipt protocol/status mismatch")
    _same_identity(prior.get("preregistration"), prereg_identity, label="prior prereg")
    _same_identity(prior.get("freeze_receipt"), freeze_identity, label="prior freeze")
    if int(prior.get("excluded_source_episode_count", -1)) != 4_369:
        raise RuntimeError("prior source exclusion count mismatch")
    expected = {
        "action_profile_ids": (4_370, "a65e5534e0db40617126e5c916c650b273e7554247e145bdd3b5bf28a36c3b16"),
        "scene_template_content_hashes": (4_378, "a5437c01f480e3ad6a22b90f2d31f8cda9bec2a029889fc0ffc8794ba7d89dbc"),
        "pair_content_hashes": (4_378, "58404c522605e0129d4c3a59680e4a8143a9eb2d651a05d34c4dc5ebd37826f7"),
        "query_pixel_hashes": (4_378, "7a54a31c301b780af492153122eaaa095dfc9af384d95bb5a4875c2795f05b4e"),
    }
    content = _mapping(prior.get("prior_content_exclusions"), label="prior content")
    for key, (count, digest) in expected.items():
        entry = _mapping(content.get(key), label=f"prior content {key}")
        if entry.get("count") != count or entry.get("sha256") != digest:
            raise RuntimeError(f"prior content identity mismatch: {key}")
    overlap = _mapping(
        prior.get("failed_attempt_old_prior_overlap"), label="prior overlap"
    )
    if overlap.get("passed") is not True:
        raise RuntimeError("old/failed prior overlap gate failed")
    for key in (
        "source_episodes",
        "action_profile_ids",
        "scene_template_content_hashes",
        "pair_content_hashes",
        "query_pixel_hashes",
    ):
        _zero_overlap(overlap.get(key), label=f"prior overlap {key}")
    _closed_public(prior.get("public_test"), label="prior Public Test")
    if (
        prior.get("reference_model_training_or_scoring") is not False
        or int(prior.get("reference_model_optimizer_steps", -1)) != 0
    ):
        raise RuntimeError("prior receipt records model activity")


def _validate_request(
    request: Mapping[str, Any],
    freeze_identity: Mapping[str, Any],
    prior_identity: Mapping[str, Any],
) -> None:
    if (
        request.get("protocol") != PROTOCOL
        or request.get("recovery_authorization_id") != RECOVERY_ID
        or request.get("active_splits") != ["train", "loader_validation"]
        or request.get("pair_counts") != {"train": 2_048, "loader_validation": 256}
        or request.get("jpeg_quality") != 95
        or request.get("workers") != 16
        or request.get("public_test_generated") is not False
        or request.get("public_test_opened") is not False
    ):
        raise RuntimeError("formal request contract mismatch")
    _same_identity(request.get("freeze_receipt"), freeze_identity, label="request freeze")
    _same_identity(
        request.get("prior_episode_exclusion_receipt"),
        prior_identity,
        label="request prior",
    )


def _validate_build(report: Mapping[str, Any], request: Mapping[str, Any]) -> None:
    if (
        report.get("protocol") != PROTOCOL
        or report.get("recovery_authorization_id") != RECOVERY_ID
        or report.get("active_splits") != ["train", "loader_validation"]
        or report.get("passed") is not True
        or report.get("public_test_generated") is not False
        or report.get("public_test_opened") is not False
        or report.get("request") != request
    ):
        raise RuntimeError("build report top-level contract mismatch")
    splits = _mapping(report.get("splits"), label="build splits")
    expected = {
        "train": (2_048, 16_384, 512),
        "loader_validation": (256, 2_048, 64),
    }
    for name, (pairs, rows, per_anchor) in expected.items():
        split = _mapping(splits.get(name), label=f"build split {name}")
        if (
            split.get("pair_count") != pairs
            or split.get("model_rows") != rows
            or split.get("attempted_candidates") != pairs
            or split.get("acceptance_rate") != 1.0
            or split.get("action_anchor_counts")
            != {
                "endpoint4": per_anchor,
                "front_hold": per_anchor,
                "plateau": per_anchor,
                "ramp4": per_anchor,
            }
            or split.get("passed") is not True
            or split.get("all_causal_checks_passed") is not True
            or _mapping(
                split.get("fresh_simulator_replay"), label=f"{name} replay"
            ).get("passed")
            is not True
            or _mapping(
                split.get("prior_episode_and_content_exclusion"),
                label=f"{name} prior exclusion",
            ).get("passed")
            is not True
        ):
            raise RuntimeError(f"formal split contract failed: {name}")
    cross = _mapping(report.get("cross_split_audit"), label="cross-split audit")
    if cross.get("passed") is not True or not all(
        value is True
        for value in _mapping(cross.get("checks"), label="cross checks").values()
    ):
        raise RuntimeError("cross-split gate failed")
    for key in (
        "source_episode_overlap",
        "exact_action_profile_id_overlap",
        "scene_template_content_hash_overlap",
        "pair_content_hash_overlap",
        "query_pixel_hash_overlap",
    ):
        _zero_overlap(cross.get(key), label=f"cross-split {key}")
    causal = _mapping(report.get("causal_data_contract"), label="causal contract")
    if causal.get("passed") is not True or not all(
        value is True
        for value in _mapping(causal.get("checks"), label="causal checks").values()
    ):
        raise RuntimeError("causal data contract failed")
    replay = _mapping(report.get("fresh_simulator_replay"), label="fresh replay")
    if (
        replay.get("passed") is not True
        or replay.get("pair_count") != 2_304
        or replay.get("mode_replay_count") != 4_608
        or replay.get("maximum_physical_state_gap") != 0
        or replay.get("maximum_simulator_state_gap") != 0
        or replay.get("query_gap_used_as_replay_substitute") is not False
    ):
        raise RuntimeError("fresh simulator replay gate failed")
    source = _mapping(
        report.get("source_h5_post_build_integrity"), label="source integrity"
    )
    if (
        source.get("passed") is not True
        or source.get("full_content_rehashed_after_local_build_before_publish")
        is not True
        or source.get("expected_sha256") != source.get("observed_sha256")
    ):
        raise RuntimeError("source post-build integrity failed")
    storage = _mapping(
        report.get("storage_publication_contract"), label="storage contract"
    )
    if (
        storage.get("lance_committed_and_reopened_on_local_staging") is not True
        or storage.get("final_publish_method")
        != "verified_x_exclusive_copytree"
        or storage.get("final_success_marker") != "_SUCCESS.json"
        or storage.get("failed_publish_is_never_marked_complete") is not True
    ):
        raise RuntimeError("storage publication contract failed")


def _validate_manifest(
    manifest: Mapping[str, Any],
    prior_identity: Mapping[str, Any],
    release_files: Mapping[str, Mapping[str, Any]],
) -> None:
    if (
        manifest.get("protocol") != PROTOCOL
        or manifest.get("recovery_authorization_id") != RECOVERY_ID
        or manifest.get("active_splits") != ["train", "loader_validation"]
        or manifest.get("build_passed") is not True
        or manifest.get("public_test_generated") is not False
        or manifest.get("public_test_opened") is not False
    ):
        raise RuntimeError("manifest contract mismatch")
    prior = _mapping(
        manifest.get("prior_episode_exclusion_receipt"), label="manifest prior"
    )
    if prior.get("sha256") != prior_identity.get("sha256"):
        raise RuntimeError("manifest prior identity mismatch")
    files = _mapping(manifest.get("files"), label="manifest files")
    if set(files) != set(release_files) - {"manifest.json", "_SUCCESS.json"}:
        raise RuntimeError("manifest file set mismatch")
    for path, digest in files.items():
        if digest != release_files[path]["sha256"]:
            raise RuntimeError(f"manifest file digest mismatch: {path}")


def _validate_success(
    success: Mapping[str, Any],
    release_files: Mapping[str, Mapping[str, Any]],
) -> None:
    if (
        success.get("protocol") != PROTOCOL
        or success.get("recovery_authorization_id") != RECOVERY_ID
        or success.get("status") != "complete"
        or success.get("checks_passed") is not True
        or success.get("public_test_generated") is not False
        or success.get("public_test_opened") is not False
    ):
        raise RuntimeError("success marker contract mismatch")
    bound = _mapping(success.get("bound_files"), label="success bound files")
    for name in ("request.json", "build_report.json", "manifest.json"):
        _same_identity(bound.get(name), release_files[name], label=f"success {name}")
    receipts = success.get("file_receipts_without_success_marker")
    if not isinstance(receipts, list):
        raise RuntimeError("success file receipts must be a list")
    expected = {
        path: (entry["sha256"], entry["size_bytes"])
        for path, entry in release_files.items()
        if path != "_SUCCESS.json"
    }
    observed: dict[str, tuple[Any, Any]] = {}
    for value in receipts:
        entry = _mapping(value, label="success file receipt")
        path = entry.get("path")
        if not isinstance(path, str) or path in observed:
            raise RuntimeError("success file receipt path invalid or duplicated")
        observed[path] = (entry.get("sha256"), entry.get("size_bytes"))
    if observed != expected:
        raise RuntimeError("success marker does not bind the exact release tree")
    tables = _mapping(success.get("lance_tables"), label="success Lance tables")
    for name, rows in (("train", 16_384), ("loader_validation", 2_048)):
        table = _mapping(tables.get(name), label=f"success table {name}")
        if (
            table.get("row_count") != rows
            or table.get("passed") is not True
            or table.get("schema_equals_frozen_v4") is not True
        ):
            raise RuntimeError(f"success Lance table failed: {name}")
    publication = _mapping(success.get("publication"), label="publication")
    if (
        publication.get("source_and_destination_file_receipts_equal") is not True
        or publication.get("success_marker_written_last") is not True
        or publication.get("method") != "verified_x_exclusive_copytree"
        or publication.get("nonempty_directory_rename_used") is not False
    ):
        raise RuntimeError("publication receipt failed")


def _validate_probe(
    probe: Mapping[str, Any],
    prereg_identity: Mapping[str, Any],
    freeze_identity: Mapping[str, Any],
    prior_identity: Mapping[str, Any],
    release_files: Mapping[str, Mapping[str, Any]],
) -> None:
    if (
        probe.get("protocol") != PROTOCOL
        or probe.get("recovery_authorization_id") != RECOVERY_ID
        or probe.get("probe_id")
        != "cube_gripper_carry_h3_v4r1_rgb_history_probe_v1"
        or probe.get("role")
        != "frozen_rgb_history_data_probe_not_reference_model_evaluation"
        or probe.get("status") != "passed"
        or probe.get("passed") is not True
        or probe.get("active_splits") != ["train", "loader_validation"]
    ):
        raise RuntimeError("RGB-history probe top-level contract mismatch")
    gates = _mapping(probe.get("gates"), label="probe gates")
    if set(gates) != EXPECTED_GATES or not all(value is True for value in gates.values()):
        raise RuntimeError("not every frozen RGB-history gate passed")
    inputs = _mapping(probe.get("inputs"), label="probe inputs")
    auth = _mapping(inputs.get("authorization_identities"), label="probe auth")
    for key, identity in (
        ("preregistration", prereg_identity),
        ("freeze_receipt", freeze_identity),
        ("prior_exclusion_receipt", prior_identity),
    ):
        _same_identity(auth.get(key), identity, label=f"probe {key}")
    metadata = _mapping(inputs.get("metadata_identities"), label="probe metadata")
    for name in ("request.json", "build_report.json", "manifest.json"):
        _same_identity(metadata.get(name), release_files[name], label=f"probe {name}")
    success = _mapping(inputs.get("success_marker"), label="probe success marker")
    _same_identity(success, release_files["_SUCCESS.json"], label="probe success marker")
    if (
        inputs.get("authorization_chain_verified_before_artifact_root") is not True
        or inputs.get("authorization_inputs_reverified_after_lance_reads") is not True
        or inputs.get("release_identity_unchanged_during_reads") is not True
        or inputs.get("validation_or_public_table_read") is not False
        or inputs.get("only_authorized_lance_tables_opened")
        != ["train.lance", "loader_validation.lance"]
    ):
        raise RuntimeError("probe trusted-input contract failed")
    metrics = _mapping(
        _mapping(probe.get("primary_probe"), label="primary probe").get("metrics"),
        label="primary metrics",
    )
    if (
        metrics.get("overall_accuracy") != 0.791015625
        or _mapping(metrics.get("worst_mode"), label="worst mode")
        != {"accuracy": 0.7578125, "hidden_mode": "cannot_hold"}
        or _mapping(metrics.get("worst_anchor_family"), label="worst anchor")
        != {"accuracy": 0.765625, "action_anchor_id": "front_hold"}
    ):
        raise RuntimeError("frozen primary RGB-history metrics mismatch")
    bootstrap = _mapping(
        probe.get("pair_cluster_anchor_stratified_bootstrap"), label="bootstrap"
    )
    if (
        bootstrap.get("passed") is not True
        or bootstrap.get("lower_bound_2_5_percent") != 0.76171875
        or bootstrap.get("resamples") != 10_000
        or bootstrap.get("unit") != "pair_cluster"
    ):
        raise RuntimeError("frozen bootstrap result mismatch")
    controls = _mapping(probe.get("negative_controls"), label="negative controls")
    expected_controls = {
        "label_permutation": ("mean_accuracy", 0.501708984375),
        "x0_only": ("accuracy", 0.5),
        "query_x2_only": ("accuracy", 0.5),
        "action_only": ("accuracy", 0.5),
    }
    for name, (field, value) in expected_controls.items():
        control = _mapping(controls.get(name), label=f"control {name}")
        if control.get("passed") is not True or control.get(field) != value:
            raise RuntimeError(f"negative control mismatch: {name}")
    cross = _mapping(
        probe.get("cross_split_content_isolation"), label="probe isolation"
    )
    if cross.get("passed") is not True:
        raise RuntimeError("probe split isolation failed")
    for key in (
        "source_episode_overlap",
        "exact_action_profile_id_overlap",
        "scene_template_content_hash_overlap",
        "pair_content_hash_overlap",
        "query_pixel_hash_overlap",
    ):
        _zero_overlap(cross.get(key), label=f"probe {key}")
    integrity = _mapping(probe.get("data_integrity"), label="probe data integrity")
    splits = _mapping(integrity.get("splits"), label="probe integrity splits")
    if (
        _mapping(splits.get("train"), label="probe train").get("pair_count") != 2_048
        or _mapping(splits.get("loader_validation"), label="probe dev").get(
            "pair_count"
        )
        != 256
        or _mapping(
            integrity.get("frozen_prior_exclusion_audit"),
            label="probe prior audit",
        ).get("passed")
        is not True
    ):
        raise RuntimeError("probe data-integrity receipt failed")
    fit = _mapping(probe.get("fit_contract"), label="probe fit contract")
    if (
        fit.get("development_evaluated_once_without_tuning") is not True
        or fit.get("reference_model_or_checkpoint_loaded") is not False
        or fit.get("standard_scaler_fit_split_only") != "train"
    ):
        raise RuntimeError("probe fit contract mismatch")
    _closed_public(probe.get("public_test"), label="probe Public Test")


@dataclass
class _HeldRelease:
    """A nofollow release snapshot whose directories and files remain held."""

    root: Path
    root_identity: tuple[int, int, int]
    directory_fds: dict[str, int]
    directory_identities: dict[str, tuple[int, int, int]]
    directory_entries: dict[str, tuple[str, ...]]
    file_fds: dict[str, int]
    file_identities: dict[str, tuple[int, int, int]]
    receipts: dict[str, dict[str, Any]]
    json_bytes: dict[str, bytes]

    @classmethod
    def open(cls, root: Path) -> "_HeldRelease":
        _reject_public(root, label="artifact root")
        path_metadata = os.lstat(root)
        if stat.S_ISLNK(path_metadata.st_mode) or not stat.S_ISDIR(
            path_metadata.st_mode
        ):
            raise RuntimeError("artifact root must be a non-symlink directory")
        root_fd = os.open(
            root,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        held = cls(
            root=root,
            root_identity=_stat_identity(os.fstat(root_fd)),
            directory_fds={"": root_fd},
            directory_identities={"": _stat_identity(os.fstat(root_fd))},
            directory_entries={},
            file_fds={},
            file_identities={},
            receipts={},
            json_bytes={},
        )
        try:
            if held.root_identity != _stat_identity(path_metadata):
                raise RuntimeError("artifact root identity changed while opening")
            held._scan_directory("")
            if set(held.directory_fds) != EXPECTED_RELEASE_DIRECTORIES:
                raise RuntimeError(
                    "release directory inventory mismatch: "
                    f"{sorted(held.directory_fds)}"
                )
            if len(held.receipts) != 10:
                raise RuntimeError(
                    f"release must contain exactly 10 files, got {len(held.receipts)}"
                )
            held.receipts = dict(sorted(held.receipts.items()))
            held.reverify()
            return held
        except BaseException:
            held.close()
            raise

    def _scan_directory(self, relative_directory: str) -> None:
        descriptor = self.directory_fds[relative_directory]
        names = tuple(sorted(os.listdir(descriptor)))
        self.directory_entries[relative_directory] = names
        for name in names:
            if name in {"", ".", ".."} or "/" in name:
                raise RuntimeError(f"invalid release entry name: {name!r}")
            relative = (
                f"{relative_directory}/{name}" if relative_directory else name
            )
            _reject_public(relative, label="release entry")
            metadata = os.stat(
                name, dir_fd=descriptor, follow_symlinks=False
            )
            if stat.S_ISLNK(metadata.st_mode):
                raise RuntimeError(f"release contains symlink: {relative}")
            if stat.S_ISDIR(metadata.st_mode):
                child = os.open(
                    name,
                    os.O_RDONLY
                    | getattr(os, "O_DIRECTORY", 0)
                    | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=descriptor,
                )
                if _stat_identity(os.fstat(child)) != _stat_identity(metadata):
                    os.close(child)
                    raise RuntimeError(
                        f"release directory changed while opening: {relative}"
                    )
                self.directory_fds[relative] = child
                self.directory_identities[relative] = _stat_identity(metadata)
                self._scan_directory(relative)
                continue
            if not stat.S_ISREG(metadata.st_mode):
                raise RuntimeError(f"release entry is not regular: {relative}")
            child = os.open(
                name,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=descriptor,
            )
            if _stat_identity(os.fstat(child)) != _stat_identity(metadata):
                os.close(child)
                raise RuntimeError(f"release file changed while opening: {relative}")
            digest, size = _hash_fd(child)
            self.file_fds[relative] = child
            self.file_identities[relative] = _stat_identity(metadata)
            self.receipts[relative] = {
                "path": relative,
                "sha256": digest,
                "size_bytes": size,
            }
            if relative in EXPECTED_RELEASE_FILES:
                self.json_bytes[relative] = _read_fd(child)

    def reverify(self) -> None:
        path_metadata = os.lstat(self.root)
        if (
            stat.S_ISLNK(path_metadata.st_mode)
            or _stat_identity(path_metadata) != self.root_identity
            or _stat_identity(os.fstat(self.directory_fds[""]))
            != self.root_identity
        ):
            raise RuntimeError("artifact root identity changed during finalization")
        for relative, descriptor in self.directory_fds.items():
            observed_entries = tuple(sorted(os.listdir(descriptor)))
            if observed_entries != self.directory_entries[relative]:
                raise RuntimeError(
                    f"release directory entries changed: {relative or '.'}"
                )
            if relative == "":
                continue
            parent, name = relative.rsplit("/", 1) if "/" in relative else ("", relative)
            path_metadata = os.stat(
                name,
                dir_fd=self.directory_fds[parent],
                follow_symlinks=False,
            )
            if (
                _stat_identity(path_metadata)
                != self.directory_identities[relative]
                or _stat_identity(os.fstat(descriptor))
                != self.directory_identities[relative]
            ):
                raise RuntimeError(
                    f"release directory identity changed: {relative}"
                )
        for relative, descriptor in self.file_fds.items():
            parent, name = relative.rsplit("/", 1) if "/" in relative else ("", relative)
            path_metadata = os.stat(
                name,
                dir_fd=self.directory_fds[parent],
                follow_symlinks=False,
            )
            if (
                _stat_identity(path_metadata) != self.file_identities[relative]
                or _stat_identity(os.fstat(descriptor))
                != self.file_identities[relative]
            ):
                raise RuntimeError(f"release file identity changed: {relative}")
            digest, size = _hash_fd(descriptor)
            receipt = self.receipts[relative]
            if (digest, size) != (receipt["sha256"], receipt["size_bytes"]):
                raise RuntimeError(f"release file bytes changed: {relative}")
            if relative in self.json_bytes and _read_fd(descriptor) != self.json_bytes[relative]:
                raise RuntimeError(f"release JSON bytes changed: {relative}")

    def close(self) -> None:
        for descriptor in self.file_fds.values():
            try:
                os.close(descriptor)
            except OSError:
                pass
        for relative in sorted(
            self.directory_fds, key=lambda value: value.count("/"), reverse=True
        ):
            try:
                os.close(self.directory_fds[relative])
            except OSError:
                pass
        self.file_fds.clear()
        self.directory_fds.clear()

    def __enter__(self) -> "_HeldRelease":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()


def _release_receipts(root: Path) -> dict[str, dict[str, Any]]:
    with _HeldRelease.open(root) as held:
        return dict(held.receipts)


def _write_exclusive(
    path: Path,
    payload: Mapping[str, Any],
    *,
    post_write_check: Any | None = None,
) -> None:
    _reject_public(path, label="decision output")
    path.parent.mkdir(parents=True, exist_ok=True)
    parent_metadata = os.lstat(path.parent)
    if stat.S_ISLNK(parent_metadata.st_mode) or not stat.S_ISDIR(
        parent_metadata.st_mode
    ):
        raise RuntimeError("decision output parent must be a non-symlink directory")
    parent_fd = os.open(
        path.parent,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    parent_identity = _stat_identity(os.fstat(parent_fd))
    if parent_identity != _stat_identity(parent_metadata):
        os.close(parent_fd)
        raise RuntimeError("decision output parent changed while opening")
    raw = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor: int | None = None
    created_identity: tuple[int, int, int] | None = None
    try:
        descriptor = os.open(
            path.name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0),
            0o644,
            dir_fd=parent_fd,
        )
        created_identity = _stat_identity(os.fstat(descriptor))
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            descriptor = None
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        published = os.stat(
            path.name, dir_fd=parent_fd, follow_symlinks=False
        )
        if _stat_identity(published) != created_identity:
            raise RuntimeError("decision output identity changed after write")
        parent_post = os.lstat(path.parent)
        if (
            stat.S_ISLNK(parent_post.st_mode)
            or _stat_identity(parent_post) != parent_identity
            or _stat_identity(os.fstat(parent_fd)) != parent_identity
        ):
            raise RuntimeError("decision output parent changed during write")
        verification_fd = os.open(
            path.name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
        try:
            if _stat_identity(os.fstat(verification_fd)) != created_identity:
                raise RuntimeError("decision output changed before verification")
            if _read_fd(verification_fd) != raw:
                raise RuntimeError("decision output byte verification failed")
        finally:
            os.close(verification_fd)
        os.fsync(parent_fd)
        final_parent = os.lstat(path.parent)
        final_output = os.lstat(path)
        if (
            stat.S_ISLNK(final_parent.st_mode)
            or _stat_identity(final_parent) != parent_identity
            or stat.S_ISLNK(final_output.st_mode)
            or _stat_identity(final_output) != created_identity
        ):
            raise RuntimeError(
                "decision output path identity changed after directory fsync"
            )
        if post_write_check is not None:
            post_write_check()
    except BaseException:
        if descriptor is not None:
            os.close(descriptor)
        if created_identity is not None:
            try:
                current = os.stat(
                    path.name, dir_fd=parent_fd, follow_symlinks=False
                )
                if _stat_identity(current) == created_identity:
                    os.unlink(path.name, dir_fd=parent_fd)
                    os.fsync(parent_fd)
            except (FileNotFoundError, OSError):
                pass
        raise
    finally:
        os.close(parent_fd)


def finalize(
    *,
    prereg_path: Path,
    freeze_path: Path,
    prior_path: Path,
    artifact_root: Path,
    probe_path: Path,
    output: Path,
    enforce_canonical_paths: bool = True,
) -> dict[str, Any]:
    paths = {
        "preregistration": prereg_path,
        "freeze_receipt": freeze_path,
        "prior_exclusion_receipt": prior_path,
        "artifact_root": artifact_root,
        "rgb_history_probe": probe_path,
        "output": output,
    }
    defaults = {
        "preregistration": DEFAULT_PREREG,
        "freeze_receipt": DEFAULT_FREEZE,
        "prior_exclusion_receipt": DEFAULT_PRIOR,
        "artifact_root": DEFAULT_ARTIFACT_ROOT,
        "rgb_history_probe": DEFAULT_PROBE,
        "output": DEFAULT_OUTPUT,
    }
    for key, value in list(paths.items()):
        paths[key] = (
            _canonical(value, defaults[key], label=key)
            if enforce_canonical_paths
            else _absolute_without_resolve(value)
        )
        _reject_public(paths[key], label=key)
    if paths["output"].exists() or paths["output"].is_symlink():
        raise FileExistsError(f"refusing to overwrite {paths['output']}")

    raw_inputs: dict[str, bytes] = {}
    documents: dict[str, dict[str, Any]] = {}
    for key in (
        "preregistration",
        "freeze_receipt",
        "prior_exclusion_receipt",
        "rgb_history_probe",
    ):
        raw = _read_bytes(paths[key], label=key)
        _require_identity(raw, EXPECTED_INPUTS[key], label=key)
        raw_inputs[key] = raw
        documents[key] = (
            _yaml(raw, label=key)
            if key == "preregistration"
            else _json(raw, label=key)
        )
    identities = {key: _identity(raw) for key, raw in raw_inputs.items()}

    _validate_prereg(documents["preregistration"], identities["preregistration"])
    _validate_freeze(documents["freeze_receipt"], identities["preregistration"])
    _validate_prior(
        documents["prior_exclusion_receipt"],
        identities["preregistration"],
        identities["freeze_receipt"],
    )

    held_release = _HeldRelease.open(paths["artifact_root"])
    release_receipts = held_release.receipts
    try:
        for name, expected in EXPECTED_RELEASE_FILES.items():
            entry = release_receipts.get(name)
            if entry is None or (entry["sha256"], entry["size_bytes"]) != expected:
                raise RuntimeError(f"canonical release identity mismatch: {name}")
        release_documents = {
            name: _json(held_release.json_bytes[name], label=f"release {name}")
            for name in (
                "request.json",
                "build_report.json",
                "manifest.json",
                "_SUCCESS.json",
            )
        }
        _validate_request(
            release_documents["request.json"],
            identities["freeze_receipt"],
            identities["prior_exclusion_receipt"],
        )
        _validate_build(
            release_documents["build_report.json"],
            release_documents["request.json"],
        )
        _validate_manifest(
            release_documents["manifest.json"],
            identities["prior_exclusion_receipt"],
            release_receipts,
        )
        _validate_success(release_documents["_SUCCESS.json"], release_receipts)
        _validate_probe(
            documents["rgb_history_probe"],
            identities["preregistration"],
            identities["freeze_receipt"],
            identities["prior_exclusion_receipt"],
            release_receipts,
        )
    except BaseException:
        held_release.close()
        raise

    payload: dict[str, Any] = {
        "schema_version": 1,
        "decision_id": DECISION_ID,
        "protocol_id": PROTOCOL,
        "recovery_authorization_id": RECOVERY_ID,
        "decided_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "status": DECISION_STATUS,
        "scope": DECISION_SCOPE,
        "summary": {
            "formal_data_build_completed": True,
            "formal_data_contract_passed": True,
            "action_support_passed": True,
            "causal_and_fresh_replay_gates_passed": True,
            "five_class_split_isolation_passed": True,
            "rgb_history_probe_passed": True,
            "development_ready": True,
            "reason": (
                "The sole v4r1 recovery build and frozen RGB History-3 probe "
                "passed every preregistered data-readiness gate."
            ),
        },
        "authorization_chain": {
            "preregistration": {
                "path": portable_contextworld_path(paths["preregistration"]),
                **identities["preregistration"],
            },
            "freeze_receipt": {
                "path": portable_contextworld_path(paths["freeze_receipt"]),
                **identities["freeze_receipt"],
            },
            "prior_exclusion_receipt": {
                "path": portable_contextworld_path(paths["prior_exclusion_receipt"]),
                **identities["prior_exclusion_receipt"],
            },
            "rgb_history_probe": {
                "path": portable_contextworld_path(paths["rgb_history_probe"]),
                **identities["rgb_history_probe"],
            },
            "original_v4_infrastructure_failure_decision": dict(
                EXPECTED_FAILURE_DECISION
            ),
        },
        "formal_data": {
            "logical_root": portable_contextworld_path(paths["artifact_root"]),
            "success_marker": release_receipts["_SUCCESS.json"],
            "request": release_receipts["request.json"],
            "build_report": release_receipts["build_report.json"],
            "manifest": release_receipts["manifest.json"],
            "pair_counts": {"train": 2_048, "loader_validation": 256},
            "model_row_counts": {"train": 16_384, "loader_validation": 2_048},
            "pairs_per_anchor": {"train": 512, "loader_validation": 64},
            "anchor_families": ["endpoint4", "front_hold", "plateau", "ramp4"],
            "cross_split_overlap_counts": {
                "source_episode": 0,
                "action_profile": 0,
                "scene_template": 0,
                "pair_content": 0,
                "query_pixel": 0,
            },
            "publication_complete": True,
        },
        "rgb_history_probe": {
            "overall_accuracy": 0.791015625,
            "worst_mode": {"name": "cannot_hold", "accuracy": 0.7578125},
            "worst_anchor": {"name": "front_hold", "accuracy": 0.765625},
            "pair_cluster_bootstrap_lower_bound_2_5_percent": 0.76171875,
            "label_permutation_mean_accuracy": 0.501708984375,
            "x0_only_accuracy": 0.5,
            "query_x2_only_accuracy": 0.5,
            "action_only_accuracy": 0.5,
            "all_frozen_gates_passed": True,
            "reference_model_or_checkpoint_loaded": False,
        },
        "required_gate_summary": {
            "freeze_identity": True,
            "formal_data_contract": True,
            "exact_pair_and_anchor_counts": True,
            "five_class_split_disjoint_content": True,
            "action_support": True,
            "causal_continuous_trajectory": True,
            "fresh_independent_replay": True,
            "rgb_history_probe": True,
            "all_required": True,
        },
        "public_test": {
            "access_status": "closed_not_read_not_scored",
            "generated": False,
            "opened": False,
            "read": False,
            "hashed": False,
            "scored": False,
        },
        "reference_model_phase": {
            "training_or_scoring_authorized": False,
            "trainer_invoked": False,
            "optimizer_steps_run": 0,
            "checkpoints_created": False,
            "lewm_or_pldm_development_scoring_run": False,
            "original_task_retention_run": False,
            "public_test_model_scoring_opened": False,
        },
        "claims": {
            "formal_development_data_constructed": True,
            "formal_development_data_contract_passed": True,
            "data_readiness_passed": True,
            "positive_rgb_history_recoverability_claim_allowed": True,
            "positive_reference_model_claim_allowed": False,
            "release_claim_allowed": False,
            "suite_registration_allowed": False,
            "public_test_claim_allowed": False,
        },
        "decision": {
            "rgb_probe_will_not_be_rerun_under_this_preregistration": True,
            "reference_model_training_requires_new_frozen_preregistration": True,
            "public_test_will_remain_closed": True,
            "current_v4r1_artifacts_retained_as_passed_development_evidence": True,
            "next_step": (
                "freeze a separate LeWM/PLDM reference-training preregistration "
                "before any optimizer step or model scoring"
            ),
        },
    }

    for key, raw in raw_inputs.items():
        if _read_bytes(paths[key], label=f"postflight {key}") != raw:
            raise RuntimeError(f"{key} changed during finalization")
    try:
        held_release.reverify()
        _write_exclusive(
            paths["output"],
            payload,
            post_write_check=held_release.reverify,
        )
    finally:
        held_release.close()
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prereg", type=Path, required=True)
    parser.add_argument("--freeze-receipt", type=Path, required=True)
    parser.add_argument("--prior-exclusion-receipt", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = finalize(
        prereg_path=args.prereg,
        freeze_path=args.freeze_receipt,
        prior_path=args.prior_exclusion_receipt,
        artifact_root=args.artifact_root,
        probe_path=args.probe,
        output=args.output,
    )
    print(
        json.dumps(
            {
                "output": str(_absolute_without_resolve(args.output)),
                "status": payload["status"],
                "data_readiness_passed": payload["claims"]["data_readiness_passed"],
                "public_test_read": payload["public_test"]["read"],
                "reference_model_optimizer_steps": payload["reference_model_phase"]["optimizer_steps_run"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
