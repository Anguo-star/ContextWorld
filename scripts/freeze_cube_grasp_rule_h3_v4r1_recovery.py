#!/usr/bin/env python3
"""Freeze the sole infrastructure recovery authorization for scientific v4.

The original Cube v4 scientific protocol and its consumed formal attempt are
immutable.  This command authorizes one recovery build in the v4r1 namespace
only after verifying the complete old authorization/failure chain, the raw
query reconstruction, the upstream H5 identity, the current recovery
implementation, and the local-staging publication contract.  It never opens
Lance, runs the builder/probe, starts a model, or accesses Public Test.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any, Mapping, Sequence

import h5py
import yaml


ROOT = Path(__file__).absolute().parents[1]
DEFAULT_PREREG = ROOT / (
    "configs/benchmark/"
    "cube_gripper_carry_h3_development_recovery_prereg_v4r1.yaml"
)

SCIENTIFIC_PROTOCOL_ID = "cube_gripper_carry_rule_history3_development_v4"
RECOVERY_AUTHORIZATION_ID = "cube_gripper_carry_h3_development_v4r1"
PREREG_STATUS = "preregistered_before_v4r1_recovery_build"
FREEZE_STATUS = "frozen_before_v4r1_recovery_build"
OLD_PREREG_STATUS = "preregistered_before_first_v4_build"
OLD_FREEZE_STATUS = "frozen_before_first_v4_build"
FAILED_STATUS = "infrastructure_failed_immutable_attempt"
QUERY_STATUS = "failed_attempt_content_frozen_for_future_prior_exclusion"

SOURCE_SYMBOL = "upstream_cube_single_expert_h5"
SOURCE_SHA256 = "0664d507c4ff12009010644c9ae950836f954e700c172ccf22e7423af1a55625"
SOURCE_SIZE_BYTES = 101_942_558_720
SOURCE_ROW_COUNT = 2_010_000
SOURCE_EPISODE_COUNT = 10_000
SOURCE_ACTION_SHAPE = (2_010_000, 5)
SOURCE_ACTION_DTYPE = "float32"

EXPECTED_OLD_PREREG_SHA256 = (
    "f8f940bd01c0dfbc7c822e8c5885e517ba6ec2ccffda64801655f45aa847761f"
)
EXPECTED_OLD_PREREG_SIZE_BYTES = 28_730
EXPECTED_OLD_FREEZE_SHA256 = (
    "a58549ec9d5856345d4fea72ca7a7690a74204e54062ec909080c336b77af837"
)
EXPECTED_OLD_FREEZE_SIZE_BYTES = 9_765
EXPECTED_OLD_PRIOR_SHA256 = (
    "8c181529c3012cf89ecf8390d595093d256449d909c5e911297f78ed997161b4"
)
EXPECTED_OLD_PRIOR_SIZE_BYTES = 736_689
EXPECTED_FAILURE_DECISION_SHA256 = (
    "5f159f58eac81894fda36013a309d356a9d80d0b01a7224e43d5880813b2ea75"
)
EXPECTED_FAILURE_DECISION_SIZE_BYTES = 2_054_086
EXPECTED_FAILED_SHA256 = (
    "5f20da08a538f2fd0c72c5c172e64cb2359a2e5bdad1746cf2c4249bbf739936"
)
EXPECTED_FAILED_SIZE_BYTES = 1_956_930
EXPECTED_QUERY_SHA256 = (
    "a85c3343464cbbea5c13ac167d419c87bbd5b8ce942767900af171db8474e5e0"
)
EXPECTED_QUERY_SIZE_BYTES = 2_215_188
EXPECTED_ACTION_SUPPORT_SHA256 = (
    "d35c06992c99680cceb6b0c29a5018772ace6a9af013928ee9894f29c6397001"
)
EXPECTED_ACTION_SUPPORT_SIZE_BYTES = 23_847

OLD_PRIOR_RECEIPT_ID = "cube_gripper_carry_h3_v4_prior_exclusions_final_v1"
FAILED_RECEIPT_ID = "cube_gripper_carry_h3_v4_failed_formal_attempt_v1"
QUERY_RECEIPT_ID = (
    "cube_gripper_carry_h3_v4_failed_attempt_query_reconstruction_v1"
)

ACTIVE_SPLITS = ("train", "loader_validation")
PAIR_COUNTS = {"train": 2048, "loader_validation": 256}
ACTION_SUPPORT_PROFILE_COUNTS = {"train": 4096, "loader_validation": 512}
ANCHORS = ("endpoint4", "plateau", "ramp4", "front_hold")
V4_COUPLING_N = 0.40
V4R1_CATALOG_OFFSET = 2_000_000
OUTPUT_LOGICAL_ROOT = (
    "artifacts/synthesis/cube_gripper_carry_rule_h3_development_v4r1"
)
SUCCESS_MARKER = "_SUCCESS.json"

CONTENT_FIELDS = (
    "action_profile_ids",
    "scene_template_content_hashes",
    "pair_content_hashes",
    "query_pixel_hashes",
)
FAILED_SET_IDENTITIES = {
    "source_episodes": {
        "count": 2048,
        "sha256": "21e842365ad64b5c7282ae868c28140ec5c487ab1b7e4cdaf28ab9f053b02bbf",
    },
    "action_profile_ids": {
        "count": 2048,
        "sha256": "54056db3cac5d1a58b66ead41f8a339892ff48217e05c4b85cd2c9243a078552",
    },
    "scene_template_content_hashes": {
        "count": 2048,
        "sha256": "ae0be9df0c4789a3419bca6a30f42ec3b4f085dc38485e6aa67c75362f5a2153",
    },
    "pair_content_hashes": {
        "count": 2048,
        "sha256": "5e13c6b92d264281b3fee073bd112d73f61962c04f6c57bf5c47493e66842458",
    },
    "query_pixel_hashes": {
        "count": 2048,
        "sha256": "7965e7748c44f1a304b1003b6c30f7894a60318318a7b8bc6bc553aa192d6353",
    },
}
OLD_PRIOR_IDENTITIES = {
    "source_episodes": {
        "count": 2321,
        "sha256": "9722ad14b5f1852e53e8bf176480fa6d2a18ca26c15e2056f124991a0f6ace63",
    },
    "action_profile_ids": {
        "count": 2322,
        "sha256": "637dc3d084524e02cee2284654e829e33613bd55d737dd3d24d7e24591dceee1",
    },
    "scene_template_content_hashes": {
        "count": 2330,
        "sha256": "a78c47bdba0534630e19febab31789d278f470260e2397f7910ec1f6fcb73912",
    },
    "pair_content_hashes": {
        "count": 2330,
        "sha256": "ec4033f984ff6eae1e7b20b26f1a460ea1662f192be335c51f5df4293941d006",
    },
    "query_pixel_hashes": {
        "count": 2330,
        "sha256": "7e3701dac3e229d902318088f4c9a1a38e018bf28a68a643c2a2f59d41365c1e",
    },
}
RECOVERY_UNION_IDENTITIES = {
    "source_episodes": {
        "count": 4369,
        "sha256": "a2167602269492d464e7f07b2a4c1c8ba3e8c46fc1df4791ba69cd0e6027a021",
    },
    "action_profile_ids": {
        "count": 4370,
        "sha256": "a65e5534e0db40617126e5c916c650b273e7554247e145bdd3b5bf28a36c3b16",
    },
    "scene_template_content_hashes": {
        "count": 4378,
        "sha256": "a5437c01f480e3ad6a22b90f2d31f8cda9bec2a029889fc0ffc8794ba7d89dbc",
    },
    "pair_content_hashes": {
        "count": 4378,
        "sha256": "58404c522605e0129d4c3a59680e4a8143a9eb2d651a05d34c4dc5ebd37826f7",
    },
    "query_pixel_hashes": {
        "count": 4378,
        "sha256": "7a54a31c301b780af492153122eaaa095dfc9af384d95bb5a4875c2795f05b4e",
    },
}

PROBE_RECIPE = {
    "input": "decoded_x0_x1_x2_rgb_only",
    "resize_shape": [16, 16],
    "resize_interpolation": "Pillow_Resampling_BILINEAR",
    "arithmetic_dtype": "float64",
    "fixed_feature": "flatten(2*x1-x0-x2)_C_order",
    "standard_scaler_fit_split_only": "train",
    "estimator": "StandardScaler_then_RidgeClassifier_alpha_1",
    "label_encoding": {"cannot_hold": 0, "can_hold": 1},
}
PROBE_THRESHOLDS = {
    "overall_accuracy_minimum": 0.75,
    "worst_mode_accuracy_minimum": 0.70,
    "worst_anchor_family_accuracy_minimum": 0.70,
    "pair_cluster_bootstrap_lower_bound_minimum": 0.70,
    "label_permutation_mean_accuracy_maximum": 0.60,
    "x0_only_accuracy_maximum": 0.51,
    "query_only_accuracy_maximum": 0.51,
    "action_only_accuracy_maximum": 0.51,
}
PROBE_TRUSTED_INPUT_CONTRACT = {
    "required_explicit_cli_options": [
        "--artifact-root",
        "--prereg",
        "--freeze-receipt",
        "--prior-exclusion-receipt",
        "--output",
    ],
    "authorization_documents_read_before_artifact_root": True,
    "preregistration_status": PREREG_STATUS,
    "freeze_receipt_status": FREEZE_STATUS,
    "prior_exclusion_receipt_id": (
        "cube_gripper_carry_h3_v4r1_prior_exclusions_final_v1"
    ),
    "prior_exclusion_receipt_status": FREEZE_STATUS,
    "freeze_receipt_must_bind_exact_preregistration": True,
    "prior_exclusion_receipt_must_bind_exact_preregistration_and_freeze": True,
    "metadata_files_parsed_before_lance_open": [
        "request.json",
        "build_report.json",
        "manifest.json",
    ],
    "success_marker_must_exactly_bind_metadata_files": True,
    "request_must_bind_exact_freeze_and_prior_receipts": True,
    "build_report_request_must_equal_request_json": True,
    "manifest_must_bind_exact_prior_and_files": True,
    "metadata_protocol_recovery_splits_public_closure_revalidated": True,
    "input_bytes_reverified_after_lance_reads": True,
    "public_shaped_cli_paths_rejected": True,
    "symlinks_rejected": True,
    "artifact_root_path_recorded": False,
}

REQUIRED_IDENTITY_KEYS = (
    "base_v2_physics",
    "v3_physics_dependency",
    "common_causal_contract",
    "v4_physics",
    "v4_builder",
    "v4_physics_tests",
    "v4_builder_tests",
    "v4_action_support_audit",
    "v4_action_support_audit_tests",
    "v4_probe",
    "v4_probe_tests",
    "v4r1_prior_finalizer",
    "v4r1_prior_finalizer_tests",
    "recovery_freezer",
    "recovery_freezer_tests",
    "recovery_protocol_document",
)
REQUIRED_INPUT_KEYS = (
    "original_v4_preregistration",
    "original_v4_freeze_receipt",
    "old_final_prior_receipt",
    "infrastructure_failure_decision",
    "failed_formal_attempt_receipt",
    "query_reconstruction_receipt",
    "v4r1_action_support_audit",
    "source_h5",
)

EXPECTED_INPUT_LOGICAL_PATHS = {
    "original_v4_preregistration": (
        "artifacts/evaluation/history3/cube_gripper_carry_h3_development_v4/"
        "failed_attempt_v1_snapshots/prereg_v4.yaml"
    ),
    "original_v4_freeze_receipt": (
        "artifacts/evaluation/history3/cube_gripper_carry_h3_development_v4/"
        "development_prereg_freeze_receipt_v1.json"
    ),
    "old_final_prior_receipt": (
        "artifacts/evaluation/history3/cube_gripper_carry_h3_development_v4/"
        "prior_episode_exclusions_final_v1.json"
    ),
    "infrastructure_failure_decision": (
        "artifacts/evaluation/history3/cube_gripper_carry_h3_development_v4/"
        "development_decision_infrastructure_failure_v1.json"
    ),
    "failed_formal_attempt_receipt": (
        "artifacts/evaluation/history3/cube_gripper_carry_h3_development_v4/"
        "failed_formal_attempt_receipt_v1.json"
    ),
    "query_reconstruction_receipt": (
        "artifacts/evaluation/history3/cube_gripper_carry_h3_development_v4/"
        "failed_formal_attempt_query_reconstruction_receipt_v1.json"
    ),
    "v4r1_action_support_audit": (
        "artifacts/evaluation/history3/cube_gripper_carry_h3_development_v4r1/"
        "action_support_audit_v2.json"
    ),
}
EXPECTED_IDENTITY_PATHS = {
    "base_v2_physics": "contextworld/evaluation/cube_grasp_rule_h3.py",
    "v3_physics_dependency": "contextworld/evaluation/cube_grasp_rule_h3_v3.py",
    "common_causal_contract": "contextworld/benchmarks/causal_data_contract.py",
    "v4_physics": "contextworld/evaluation/cube_grasp_rule_h3_v4.py",
    "v4_builder": "scripts/build_cube_grasp_rule_h3_v4_data.py",
    "v4_physics_tests": "tests/test_cube_grasp_rule_h3_v4.py",
    "v4_builder_tests": "tests/test_cube_grasp_rule_h3_v4_builder.py",
    "v4_action_support_audit": "scripts/audit_cube_grasp_rule_h3_v4_action_support.py",
    "v4_action_support_audit_tests": "tests/test_cube_grasp_rule_h3_v4_action_support.py",
    "v4_probe": "scripts/probe_cube_grasp_rule_h3_v4_rgb_history.py",
    "v4_probe_tests": "tests/test_cube_grasp_rule_h3_v4_rgb_history.py",
    "v4r1_prior_finalizer": "scripts/finalize_cube_grasp_rule_h3_v4r1_prior_exclusions.py",
    "v4r1_prior_finalizer_tests": "tests/test_finalize_cube_grasp_rule_h3_v4r1_prior_exclusions.py",
    "recovery_freezer": "scripts/freeze_cube_grasp_rule_h3_v4r1_recovery.py",
    "recovery_freezer_tests": "tests/test_cube_grasp_rule_h3_v4r1_recovery_freeze.py",
    "recovery_protocol_document": "docs/protocols/Cube_Gripper_Carry_History3_Development_v4r1_Recovery_Protocol.md",
}

SHA256_RE = re.compile(r"[0-9a-f]{64}")
PLACEHOLDER_TOKENS = (
    "TO_BE_FROZEN",
    "PLACEHOLDER",
    "REPLACE_ME",
    "PENDING_SHA256",
    "TBD",
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


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{label} must be an object")
    return value


def _contains_placeholder(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(_contains_placeholder(child) for child in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_placeholder(child) for child in value)
    return isinstance(value, str) and any(
        token in value.upper() for token in PLACEHOLDER_TOKENS
    )


def _reject_public(value: Path | str, *, label: str) -> None:
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
    _reject_public(path, label=label)
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"{label} must be a regular non-symlink file: {path}")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(descriptor, "rb") as stream:
        return stream.read()


def _read_json(path: Path, *, label: str) -> tuple[bytes, dict[str, Any]]:
    raw = _read_bytes_nofollow(path, label=label)
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{label} is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} root must be an object")
    return raw, value


def _read_yaml(path: Path, *, label: str) -> tuple[bytes, dict[str, Any]]:
    raw = _read_bytes_nofollow(path, label=label)
    try:
        value = yaml.safe_load(raw.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise RuntimeError(f"{label} is not valid UTF-8 YAML") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} root must be an object")
    return raw, value


def _identity(path: Path, raw: bytes | None = None, *, recorded_path: str | None = None) -> dict[str, Any]:
    payload = raw if raw is not None else _read_bytes_nofollow(path, label="identity file")
    return {
        "path": recorded_path if recorded_path is not None else path.as_posix(),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _identity_core(value: Any, *, label: str) -> tuple[str, int]:
    entry = _mapping(value, label=label)
    digest = entry.get("sha256")
    size = entry.get("size_bytes")
    if not isinstance(digest, str) or SHA256_RE.fullmatch(digest) is None:
        raise RuntimeError(f"{label}.sha256 must be a lowercase SHA256")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise RuntimeError(f"{label}.size_bytes must be a positive integer")
    return digest, size


def _same_identity(left: Any, right: Any, *, label: str) -> None:
    if _identity_core(left, label=f"{label} left") != _identity_core(
        right, label=f"{label} right"
    ):
        raise RuntimeError(f"{label} identity mismatch")


def _closed_public(value: Any, *, label: str, require_validation_gate: bool = False) -> dict[str, Any]:
    gate = _mapping(value, label=label)
    if gate.get("access_status") != "closed_not_read_not_scored":
        raise RuntimeError(f"{label} access_status is not closed")
    for key in ("opened", "read", "hashed", "scored"):
        if gate.get(key) is not False:
            raise RuntimeError(f"{label}.{key} must be false")
    if require_validation_gate and gate.get("validation_lance_access_allowed") is not False:
        raise RuntimeError(f"{label}.validation_lance_access_allowed must be false")
    return {
        "access_status": "closed_not_read_not_scored",
        "opened": False,
        "read": False,
        "hashed": False,
        "scored": False,
    }


def _resolve_declared_path(path: str, *, artifact_root: Path) -> Path:
    _reject_public(path, label="declared path")
    value = Path(path)
    if value.is_absolute():
        return value
    if value.parts and value.parts[0] == "artifacts":
        return artifact_root.joinpath(*value.parts[1:])
    return ROOT / value


def _verify_declared_file(
    entry: Any,
    *,
    artifact_root: Path,
    label: str,
    explicit_path: Path | None = None,
    expected_logical_path: str | None = None,
) -> dict[str, Any]:
    declared = _mapping(entry, label=label)
    logical = declared.get("path")
    if not isinstance(logical, str) or not logical:
        raise RuntimeError(f"{label}.path is missing")
    if expected_logical_path is not None and logical != expected_logical_path:
        raise RuntimeError(f"{label}.path is not the canonical logical path")
    path = explicit_path if explicit_path is not None else _resolve_declared_path(
        logical, artifact_root=artifact_root
    )
    raw = _read_bytes_nofollow(path, label=label)
    observed = _identity(path, raw, recorded_path=logical)
    if _identity_core(declared, label=label) != _identity_core(
        observed, label=f"observed {label}"
    ):
        raise RuntimeError(f"{label} file identity mismatch")
    return observed


def _require_exact_file(
    observed: Mapping[str, Any], *, digest: str, size: int, label: str
) -> None:
    if (observed.get("sha256"), observed.get("size_bytes")) != (digest, size):
        raise RuntimeError(f"{label} is not the canonical frozen input")


def _model_disabled(document: Mapping[str, Any], *, label: str) -> None:
    direct = [
        document.get("reference_model_training_or_scoring"),
        document.get("reference_model_training_or_scoring_authorized"),
    ]
    if any(value is True for value in direct):
        raise RuntimeError(f"{label} reports or authorizes reference-model work")
    phase = document.get("reference_model_phase")
    if isinstance(phase, Mapping):
        for key in (
            "training_and_scoring_authorized",
            "training_or_scoring_authorized",
            "trainer_invoked",
            "checkpoints_created",
            "lewm_or_pldm_development_scoring_run",
        ):
            if key in phase and phase.get(key) is not False:
                raise RuntimeError(f"{label}.{key} must be false")
        if "optimizer_steps_run" in phase and int(phase["optimizer_steps_run"]) != 0:
            raise RuntimeError(f"{label} optimizer steps must be zero")


def _scientific_contract(document: Mapping[str, Any]) -> dict[str, Any]:
    scientific = _mapping(
        document.get("scientific_protocol_contract"),
        label="scientific_protocol_contract",
    )
    expected = {
        "unchanged_from_original_v4": True,
        "history_tokens": 3,
        "context_transitions": 2,
        "prediction_horizon_action_blocks": 1,
        "raw_steps_per_action_block": 5,
        "can_hold_vertical_force_coupling_n": V4_COUPLING_N,
        "hidden_modes": ["cannot_hold", "can_hold"],
        "action_temporal_pattern": ["p", "negative_p", "p", "terminal_zero"],
        "action_anchor_ids": list(ANCHORS),
        "sum_p_target": 0.0,
        "final_p_target": 0.0,
        "displacement_moment_weights": [4.0, 3.0, 2.0, 1.0, 0.0],
        "displacement_moment_target": 1.0,
        "constraint_absolute_tolerance": 1.0e-6,
        "jpeg_quality": 95,
        "query_state_and_pixels_equal_across_modes": True,
        "paired_actions_bitwise_equal": True,
        "no_state_installation_after_x0": True,
    }
    if {key: scientific.get(key) for key in expected} != expected:
        raise RuntimeError("v4r1 changes the frozen scientific v4 contract")
    return dict(expected)


def _validate_recovery_contract(document: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    recovery = _mapping(document.get("recovery_contract"), label="recovery_contract")
    expected = {
        "failure_class": "infrastructure_lance_atomic_rename_eperm",
        "original_v4_formal_attempt_consumed": True,
        "retry_under_original_v4_preregistration_authorized": False,
        "original_failed_tree_immutable": True,
        "scientific_protocol_changed": False,
        "recovery_build_attempts_authorized": 1,
        "builder_or_lance_smoke_attempts_authorized": 0,
        "rgb_history_probe_attempts_authorized": 1,
        "formal_catalog_index_offset": V4R1_CATALOG_OFFSET,
        "formal_catalog_offset_four_aligned": True,
        "failed_batch_identities_must_be_excluded": True,
    }
    if {key: recovery.get(key) for key in expected} != expected:
        raise RuntimeError("v4r1 recovery authorization contract mismatch")

    storage = _mapping(
        document.get("storage_publication_contract"),
        label="storage_publication_contract",
    )
    storage_expected = {
        "staging_root_class": "local_tmp_filesystem",
        "default_staging_root": "/tmp",
        "lance_commit_completed_on_local_staging": True,
        "lance_reopened_and_verified_before_publish": True,
        "local_staging_contains_success_marker": False,
        "destination_creation": "x_exclusive_copytree",
        "copy_function": "shutil.copy2",
        "dirs_exist_ok": False,
        "source_destination_file_receipts_must_match": True,
        "source_destination_lance_identities_must_match": True,
        "nonempty_directory_rename_used": False,
        "success_marker_name": SUCCESS_MARKER,
        "success_marker_written_last": True,
        "failed_copy_marked_complete": False,
    }
    if {key: storage.get(key) for key in storage_expected} != storage_expected:
        raise RuntimeError("v4r1 local-staging/publication contract mismatch")
    return dict(expected), dict(storage_expected)


def _validate_data_and_probe(document: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    data = _mapping(document.get("data_contract"), label="data_contract")
    expected = {
        "logical_output_root": OUTPUT_LOGICAL_ROOT,
        "authorized_splits": list(ACTIVE_SPLITS),
        "pair_counts": dict(PAIR_COUNTS),
        "workers": 16,
        "episodes_per_pair": 2,
        "rows_per_pair": 8,
        "pairs_per_anchor": {"train": 512, "loader_validation": 64},
        "formal_catalog_index_offset": V4R1_CATALOG_OFFSET,
        "catalog_index_offset_modulo_anchor_count": 0,
        "source_episode_overlap_between_splits_required": 0,
        "action_profile_overlap_between_splits_required": 0,
        "scene_template_overlap_between_splits_required": 0,
        "pair_content_overlap_between_splits_required": 0,
        "query_pixel_overlap_between_splits_required": 0,
    }
    if {key: data.get(key) for key in expected} != expected:
        raise RuntimeError("v4r1 data contract mismatch")

    probe = _mapping(document.get("rgb_history_probe"), label="rgb_history_probe")
    probe_expected = {
        "attempts_authorized": 1,
        "run_only_after_complete_success_marker": True,
        "recipe_unchanged_from_v4": True,
        "thresholds_unchanged_from_v4": True,
        "recipe": PROBE_RECIPE,
        "thresholds": PROBE_THRESHOLDS,
        "trusted_input_contract": PROBE_TRUSTED_INPUT_CONTRACT,
    }
    if {key: probe.get(key) for key in probe_expected} != probe_expected:
        raise RuntimeError("v4r1 RGB-history probe contract changed")
    return dict(expected), dict(probe_expected)


def _validate_union_declaration(document: Mapping[str, Any]) -> dict[str, Any]:
    prior = _mapping(
        document.get("recovery_prior_exclusion_contract"),
        label="recovery_prior_exclusion_contract",
    )
    expected = {
        "old_prior": OLD_PRIOR_IDENTITIES,
        "failed_attempt": FAILED_SET_IDENTITIES,
        "required_union": RECOVERY_UNION_IDENTITIES,
        "old_prior_failed_attempt_overlap_required_zero": True,
        "all_five_identity_classes_required": True,
        "finalizer_required_after_recovery_freeze": True,
    }
    if {key: prior.get(key) for key in expected} != expected:
        raise RuntimeError("v4r1 prior-exclusion union declaration mismatch")
    return dict(expected)


def _validate_recovery_capacity(document: Mapping[str, Any]) -> dict[str, Any]:
    value = _mapping(
        document.get("recovery_capacity_check"), label="recovery_capacity_check"
    )
    expected = {
        "role": "non_scientific_deterministic_population_capacity",
        "eligible_before_old_prior": 9998,
        "removed_by_old_prior": 2321,
        "eligible_after_old_prior": 7677,
        "failed_train_source_episode_count": 2048,
        "failed_train_old_prior_overlap": 0,
        "eligible_after_recovery_union": 5629,
        "candidate_pool_required": 4608,
        "candidate_pool_margin": 1021,
        "runtime_recheck_required": True,
        "used_to_select_scientific_parameters": False,
    }
    if {key: value.get(key) for key in expected} != expected:
        raise RuntimeError("v4r1 deterministic recovery-capacity check mismatch")
    return dict(expected)


def _validate_action_support_authorization(
    document: Mapping[str, Any],
) -> dict[str, Any]:
    value = _mapping(
        document.get("action_support_authorization"),
        label="action_support_authorization",
    )
    expected = {
        "authorizing_audit_id": (
            "cube_gripper_carry_h3_v4r1_action_support_v2"
        ),
        "authorizing_artifact": (
            "artifacts/evaluation/history3/"
            "cube_gripper_carry_h3_development_v4r1/"
            "action_support_audit_v2.json"
        ),
        "candidate_profile_counts": dict(ACTION_SUPPORT_PROFILE_COUNTS),
        "total_candidate_profiles": 4608,
        "v1_audit_id": "cube_gripper_carry_h3_v4r1_action_support_v1",
        "v1_status": (
            "superseded_non_authorizing_incomplete_candidate_pool_coverage"
        ),
        "v1_total_profiles": 2304,
        "v1_authorizes_recovery_freeze": False,
        "v2_is_only_authorizing_action_support_input": True,
    }
    if {key: value.get(key) for key in expected} != expected:
        raise RuntimeError("v4r1 action-support authorization declaration mismatch")
    return dict(expected)


def _validate_prereg_scope(document: Mapping[str, Any]) -> None:
    if document.get("schema_version") != 1:
        raise RuntimeError("v4r1 prereg schema_version must be 1")
    if document.get("protocol_id") != SCIENTIFIC_PROTOCOL_ID or document.get(
        "scientific_protocol_id"
    ) != SCIENTIFIC_PROTOCOL_ID:
        raise RuntimeError("v4r1 scientific protocol identity mismatch")
    if document.get("recovery_authorization_id") != RECOVERY_AUTHORIZATION_ID:
        raise RuntimeError("v4r1 recovery authorization identity mismatch")
    if document.get("status") != PREREG_STATUS or document.get("phase") != "development_only":
        raise RuntimeError("v4r1 prereg status/phase mismatch")
    if document.get("reference_model_training_or_scoring_authorized") is not False:
        raise RuntimeError(
            "v4r1 top-level reference-model authorization must be false"
        )
    _closed_public(
        document.get("public_test"),
        label="prereg Public Test",
        require_validation_gate=True,
    )
    public_declaration = _mapping(
        document.get("public_test"), label="prereg Public Test"
    )
    if public_declaration.get("generated") is not False:
        raise RuntimeError("v4r1 prereg Public Test generated must be false")
    model = _mapping(document.get("reference_model_phase"), label="reference_model_phase")
    expected_model = {
        "training_and_scoring_authorized": False,
        "trainer_invoked": False,
        "optimizer_steps_authorized": 0,
        "optimizer_steps_run": 0,
        "checkpoint_creation_authorized": False,
    }
    if {key: model.get(key) for key in expected_model} != expected_model:
        raise RuntimeError("v4r1 must keep all reference-model work disabled")


def _validate_original_prereg(document: Mapping[str, Any]) -> None:
    if document.get("protocol_id") != SCIENTIFIC_PROTOCOL_ID or document.get(
        "status"
    ) != OLD_PREREG_STATUS:
        raise RuntimeError("original v4 preregistration identity/status mismatch")
    change = _mapping(document.get("scientific_change"), label="old scientific_change")
    expected = {
        "sole_change": "can_hold_vertical_force_coupling_n",
        "v3_baseline_vertical_force_coupling_n": 0.30,
        "v4_vertical_force_coupling_n": V4_COUPLING_N,
        "capability_semantics_unchanged": True,
        "history3_causal_sequence_unchanged": True,
        "action_profiles_and_constraints_unchanged_except_new_seeds": True,
    }
    if {key: change.get(key) for key in expected} != expected:
        raise RuntimeError("original v4 scientific contract mismatch")
    _closed_public(document.get("public_test"), label="old prereg Public Test", require_validation_gate=True)
    _model_disabled(document, label="old prereg")


def _validate_original_freeze(
    receipt: Mapping[str, Any], *, old_prereg: Mapping[str, Any]
) -> None:
    if receipt.get("protocol_id") != SCIENTIFIC_PROTOCOL_ID or receipt.get(
        "status"
    ) != OLD_FREEZE_STATUS or receipt.get("checks_passed") is not True:
        raise RuntimeError("original v4 freeze receipt identity/status mismatch")
    _same_identity(receipt.get("preregistration"), old_prereg, label="old freeze/prereg")
    _closed_public(receipt.get("public_test"), label="old freeze Public Test")
    if receipt.get("reference_model_training_or_scoring_authorized") is not False:
        raise RuntimeError("original v4 freeze authorizes model work")
    change = _mapping(receipt.get("scientific_change"), label="old freeze scientific_change")
    if float(change.get("v4_vertical_force_coupling_n", -1)) != V4_COUPLING_N:
        raise RuntimeError("original v4 freeze coupling mismatch")


def _validate_source_binding(value: Any, *, label: str) -> None:
    source = _mapping(value, label=label)
    expected = {
        "symbol": SOURCE_SYMBOL,
        "sha256": SOURCE_SHA256,
        "size_bytes": SOURCE_SIZE_BYTES,
        "row_count": SOURCE_ROW_COUNT,
        "episode_count": SOURCE_EPISODE_COUNT,
    }
    if {key: source.get(key) for key in expected} != expected:
        raise RuntimeError(f"{label} source identity mismatch")
    if source.get("path_recorded") not in (None, False) or source.get("path") not in (None, ""):
        raise RuntimeError(f"{label} must not record the source H5 path")


def _entry_summary(entry: Any, *, label: str) -> dict[str, Any]:
    value = _mapping(entry, label=label)
    result = {"count": int(value.get("count", -1)), "sha256": value.get("sha256")}
    if result["count"] < 0 or not isinstance(result["sha256"], str):
        raise RuntimeError(f"{label} count/digest is malformed")
    return result


def _validate_old_prior(
    receipt: Mapping[str, Any], *, old_prereg: Mapping[str, Any], old_freeze: Mapping[str, Any]
) -> None:
    if (
        receipt.get("protocol_id") != SCIENTIFIC_PROTOCOL_ID
        or receipt.get("receipt_id") != OLD_PRIOR_RECEIPT_ID
        or receipt.get("status") != OLD_FREEZE_STATUS
        or receipt.get("checks_passed") is not True
    ):
        raise RuntimeError("old final prior identity/status mismatch")
    _same_identity(receipt.get("preregistration"), old_prereg, label="old prior/prereg")
    _same_identity(receipt.get("freeze_receipt"), old_freeze, label="old prior/freeze")
    _validate_source_binding(receipt.get("source_h5"), label="old prior")
    _closed_public(receipt.get("public_test"), label="old prior Public Test")
    if receipt.get("reference_model_training_or_scoring") is not False:
        raise RuntimeError("old prior reports reference-model work")
    observed = {
        "source_episodes": {
            "count": int(receipt.get("excluded_source_episode_count", -1)),
            "sha256": receipt.get("excluded_source_episodes_sha256"),
        }
    }
    content = _mapping(receipt.get("prior_content_exclusions"), label="old prior content")
    observed.update(
        {field: _entry_summary(content.get(field), label=f"old prior {field}") for field in CONTENT_FIELDS}
    )
    if observed != OLD_PRIOR_IDENTITIES:
        raise RuntimeError("old final prior set identities mismatch")


def _validate_failed_receipt(
    receipt: Mapping[str, Any], *, old_prereg: Mapping[str, Any], old_freeze: Mapping[str, Any], old_prior: Mapping[str, Any]
) -> None:
    if (
        receipt.get("protocol_id") != SCIENTIFIC_PROTOCOL_ID
        or receipt.get("receipt_id") != FAILED_RECEIPT_ID
        or receipt.get("status") != FAILED_STATUS
        or receipt.get("checks_passed") is not True
        or receipt.get("build_passed") is not False
        or receipt.get("formal_build_attempt_consumed") is not True
        or receipt.get("retry_authorized_under_original_preregistration") is not False
    ):
        raise RuntimeError("failed formal-attempt receipt identity/status mismatch")
    identities = _mapping(receipt.get("input_identities"), label="failed input identities")
    for key, expected in (
        ("preregistration", old_prereg),
        ("freeze_receipt", old_freeze),
        ("prior_exclusion_receipt", old_prior),
    ):
        _same_identity(identities.get(key), expected, label=f"failed receipt {key}")
    _validate_source_binding(identities.get("source_h5"), label="failed receipt")
    failure = _mapping(receipt.get("failure"), label="failed receipt failure")
    if (
        failure.get("stage") != "lance_train_commit_atomic_rename"
        or int(failure.get("exit_code", -1)) != 1
        or failure.get("errno_name") != "EPERM"
        or int(failure.get("errno_number", -1)) != 1
    ):
        raise RuntimeError("failed formal-attempt infrastructure stage mismatch")
    stage = _mapping(receipt.get("stage_completion"), label="failed stage completion")
    expected_stage = {
        "train_generation_accepted_pairs": 2048,
        "train_generation_attempted_candidates": 2048,
        "train_lance_data_fragment_written": True,
        "train_lance_commit_completed": False,
        "loader_validation_started": False,
        "build_report_written": False,
        "manifest_written": False,
    }
    if {key: stage.get(key) for key in expected_stage} != expected_stage:
        raise RuntimeError("failed formal-attempt completion stage mismatch")
    scope = _mapping(receipt.get("scope"), label="failed receipt scope")
    _closed_public(scope.get("public_test"), label="failed receipt Public Test")
    if scope.get("reference_model_training_or_scoring") is not False or int(
        scope.get("optimizer_steps", -1)
    ) != 0 or scope.get("rgb_probe_run") is not False:
        raise RuntimeError("failed formal attempt scope is contaminated")
    content = _mapping(receipt.get("failed_attempt_content"), label="failed content")
    observed = {
        "source_episodes": _entry_summary(content.get("source_episodes"), label="failed source")
    }
    direct = _mapping(content.get("prior_content_exclusions"), label="failed direct content")
    observed.update(
        {field: _entry_summary(direct.get(field), label=f"failed {field}") for field in CONTENT_FIELDS[:3]}
    )
    if observed != {key: FAILED_SET_IDENTITIES[key] for key in observed}:
        raise RuntimeError("failed formal-attempt inspectable set identities mismatch")
    overlap = _mapping(content.get("prior_overlap"), label="failed prior overlap")
    for key in (
        "source_episode_count",
        "action_profile_id_count",
        "scene_template_content_hash_count",
        "pair_content_hash_count",
    ):
        if int(overlap.get(key, -1)) != 0:
            raise RuntimeError("failed formal attempt overlaps old prior")


def _validate_query_receipt(
    receipt: Mapping[str, Any], *, failed: Mapping[str, Any], old_prior: Mapping[str, Any]
) -> None:
    if (
        receipt.get("protocol_id") != SCIENTIFIC_PROTOCOL_ID
        or receipt.get("receipt_id") != QUERY_RECEIPT_ID
        or receipt.get("status") != QUERY_STATUS
        or receipt.get("checks_passed") is not True
    ):
        raise RuntimeError("query reconstruction receipt identity/status mismatch")
    _same_identity(receipt.get("failed_attempt_receipt"), failed, label="query/failed")
    identities = _mapping(receipt.get("input_identities"), label="query input identities")
    _same_identity(identities.get("failed_attempt_receipt"), failed, label="query input failed")
    _same_identity(identities.get("prior_exclusion_receipt"), old_prior, label="query input prior")
    _validate_source_binding(identities.get("source_h5"), label="query receipt")
    reconstruction = _mapping(receipt.get("reconstruction_contract"), label="query reconstruction")
    required = {
        "fragment_read_api": "lance.file.LanceFileReader_single_file",
        "dataset_manifest_opened": False,
        "lance_written": False,
        "replayed_mode": "cannot_hold_only",
        "query_model_step_idx": 2,
        "jpeg_quality": 95,
        "jpeg_reencoding_bitwise_equal_to_fragment": True,
        "builder_snapshot_loaded_by_explicit_path": True,
        "physics_snapshot_loaded_by_explicit_path": True,
        "all_inputs_reverified_unchanged_after_replay": True,
        "raw_query_hashes_unique": True,
        "raw_query_prior_overlap_zero": True,
    }
    if {key: reconstruction.get(key) for key in required} != required:
        raise RuntimeError("query reconstruction contract mismatch")
    content = _mapping(receipt.get("failed_attempt_content"), label="query content")
    observed = {
        "source_episodes": _entry_summary(content.get("source_episodes"), label="query source")
    }
    sets = _mapping(content.get("prior_content_exclusions"), label="query sets")
    observed.update(
        {field: _entry_summary(sets.get(field), label=f"query {field}") for field in CONTENT_FIELDS}
    )
    if observed != FAILED_SET_IDENTITIES:
        raise RuntimeError("query reconstruction five-set identities mismatch")
    overlap = _mapping(receipt.get("prior_overlap"), label="query overlap")
    if overlap.get("passed") is not True:
        raise RuntimeError("query reconstruction prior-overlap gate failed")
    for field in ("source_episode", *CONTENT_FIELDS):
        entry = _mapping(overlap.get(field), label=f"query overlap {field}")
        if int(entry.get("count", -1)) != 0 or entry.get("values") != []:
            raise RuntimeError(f"query reconstruction overlaps old prior: {field}")
    _closed_public(receipt.get("public_test"), label="query receipt Public Test")
    if receipt.get("reference_model_training_or_scoring") is not False or int(
        receipt.get("reference_model_optimizer_steps", -1)
    ) != 0:
        raise RuntimeError("query reconstruction reports model work")
    probe = _mapping(receipt.get("rgb_probe"), label="query receipt probe")
    if any(probe.get(key) is not False for key in ("opened", "run", "scored")):
        raise RuntimeError("query reconstruction reports RGB-probe work")


def _validate_action_support_audit(receipt: Mapping[str, Any]) -> None:
    if (
        receipt.get("schema_version") != 1
        or receipt.get("protocol") != SCIENTIFIC_PROTOCOL_ID
        or receipt.get("recovery_authorization_id") != RECOVERY_AUTHORIZATION_ID
        or receipt.get("audit_id")
        != "cube_gripper_carry_h3_v4r1_action_support_v2"
        or receipt.get("status") != "passed"
        or receipt.get("passed") is not True
    ):
        raise RuntimeError("v4r1 action-support audit identity/status mismatch")
    scope = _mapping(receipt.get("scope"), label="v4r1 action-support scope")
    if (
        scope.get("phase") != "development_only"
        or scope.get("active_splits") != list(ACTIVE_SPLITS)
        or scope.get("frozen_profile_counts") != ACTION_SUPPORT_PROFILE_COUNTS
        or int(scope.get("total_concrete_profiles", -1)) != 4608
        or scope.get("public_test_opened") is not False
        or scope.get("public_test_generated") is not False
        or scope.get("public_test_inputs") != []
        or scope.get("lance_tables_opened") != []
    ):
        raise RuntimeError("v4r1 action-support audit scope mismatch")
    namespace = _mapping(
        scope.get("formal_catalog_namespace"),
        label="v4r1 action-support catalog namespace",
    )
    if (
        int(namespace.get("catalog_index_offset", -1)) != V4R1_CATALOG_OFFSET
        or namespace.get("local_index_policy")
        != "zero_based_contiguous_within_each_split"
        or namespace.get("catalog_index_formula")
        != "FORMAL_CATALOG_INDEX_OFFSET + local_index"
        or int(namespace.get("offset_modulo_anchor_count", -1)) != 0
        or namespace.get("offset_positive") is not True
        or namespace.get("prior_catalog_namespaces_excluded")
        != [
            {"start_inclusive": 0, "stop_exclusive": 2},
            {"start_inclusive": 1_000_000, "stop_exclusive": 1_002_048},
        ]
        or namespace.get("per_split_ranges")
        != {
            "train": {
                "local_index_start_inclusive": 0,
                "local_index_stop_exclusive": 4096,
                "catalog_index_start_inclusive": 2_000_000,
                "catalog_index_stop_exclusive": 2_004_096,
            },
            "loader_validation": {
                "local_index_start_inclusive": 0,
                "local_index_stop_exclusive": 512,
                "catalog_index_start_inclusive": 2_000_000,
                "catalog_index_stop_exclusive": 2_000_512,
            },
        }
    ):
        raise RuntimeError("v4r1 action-support catalog namespace mismatch")
    overall = _mapping(receipt.get("overall"), label="v4r1 action-support overall")
    if (
        int(overall.get("profile_count", -1)) != 4608
        or int(overall.get("unique_profile_count", -1)) != 4608
        or int(overall.get("passed_profile_count", -1)) != 4608
        or int(overall.get("conservatively_supported_profile_count", -1))
        != 4608
        or overall.get("failed_profile_ids") != []
        or overall.get("passed") is not True
    ):
        raise RuntimeError("v4r1 action-support profile gate mismatch")
    failed = _mapping(
        receipt.get("failed_v4_attempt_exclusion"),
        label="v4r1 action-support failed-attempt exclusion",
    )
    if failed != {
        "failed_action_profile_count": 2048,
        "overlap_count": 0,
        "overlap_values": [],
        "passed": True,
        "recovery_action_profile_count": 4608,
    }:
        raise RuntimeError("v4r1 action-support failed-batch exclusion mismatch")
    splits = _mapping(receipt.get("splits"), label="v4r1 action-support splits")
    split_expected = {
        "train": {
            "profile_count": 4096,
            "unique_profile_count": 4096,
            "passed_profile_count": 4096,
            "conservatively_supported_profile_count": 4096,
            "action_anchor_counts": {anchor: 1024 for anchor in ANCHORS},
        },
        "loader_validation": {
            "profile_count": 512,
            "unique_profile_count": 512,
            "passed_profile_count": 512,
            "conservatively_supported_profile_count": 512,
            "action_anchor_counts": {anchor: 128 for anchor in ANCHORS},
        },
    }
    for split, expected in split_expected.items():
        observed = _mapping(splits.get(split), label=f"action-support {split}")
        if {key: observed.get(key) for key in expected} != expected:
            raise RuntimeError(f"v4r1 action-support {split} coverage mismatch")
        if observed.get("failed_profile_ids") != [] or observed.get("passed") is not True:
            raise RuntimeError(f"v4r1 action-support {split} gate failed")
    cross = _mapping(receipt.get("cross_split"), label="v4r1 action-support cross split")
    profile_overlap = _mapping(
        cross.get("profile_content_overlap"), label="action overlap"
    )
    anchor_overlap = _mapping(
        cross.get("anchor_family_overlap"), label="anchor overlap"
    )
    if (
        cross.get("passed") is not True
        or profile_overlap.get("count") != 0
        or profile_overlap.get("values") != []
        or anchor_overlap.get("count") != 4
        or anchor_overlap.get("values") != sorted(ANCHORS)
    ):
        raise RuntimeError("v4r1 action-support cross-split overlap gate failed")


def _find_binding(document: Mapping[str, Any], aliases: Sequence[str]) -> Mapping[str, Any] | None:
    containers = [document]
    for key in (
        "input_identities",
        "frozen_inputs",
        "recovery_inputs",
        "authorization_inputs",
        "evidence",
        "infrastructure_failure",
        "original_v4_failure",
    ):
        value = document.get(key)
        if isinstance(value, Mapping):
            containers.append(value)
    matches = [container[alias] for container in containers for alias in aliases if isinstance(container.get(alias), Mapping)]
    if not matches:
        return None
    first = matches[0]
    for value in matches[1:]:
        _same_identity(value, first, label="duplicate decision binding")
    return first


def _validate_failure_decision(
    decision: Mapping[str, Any], *, old_prereg: Mapping[str, Any], old_freeze: Mapping[str, Any], old_freeze_document: Mapping[str, Any], old_prior: Mapping[str, Any], failed: Mapping[str, Any], failed_document: Mapping[str, Any], query: Mapping[str, Any]
) -> None:
    if (
        decision.get("schema_version") != 1
        or decision.get("protocol_id") != SCIENTIFIC_PROTOCOL_ID
        or decision.get("decision_id")
        != "cube_gripper_carry_h3_v4_infrastructure_failure_v1"
        or decision.get("classification")
        != "infrastructure_failure_not_scientific_gate_failure"
        or decision.get("checks_passed") is not True
    ):
        raise RuntimeError("infrastructure failure decision protocol/schema mismatch")
    if decision.get("status") != "failed_development":
        raise RuntimeError("infrastructure failure decision is not failed_development")
    stage = decision.get("failure_stage")
    if stage != "formal_build_lance_train_commit_atomic_rename":
        raise RuntimeError("infrastructure failure decision stage mismatch")
    _closed_public(decision.get("public_test"), label="failure decision Public Test")
    public = _mapping(decision.get("public_test"), label="failure decision Public Test")
    if public.get("generated") is not False:
        raise RuntimeError("failure decision reports generated Public Test data")
    reference = _mapping(
        decision.get("reference_model_phase"), label="failure decision reference model"
    )
    expected_reference = {
        "training_or_scoring_authorized": False,
        "trainer_invoked": False,
        "optimizer_steps_run": 0,
        "checkpoints_created": False,
        "lewm_or_pldm_scoring_run": False,
    }
    if {key: reference.get(key) for key in expected_reference} != expected_reference:
        raise RuntimeError("failure decision reports reference-model work")
    probe = _mapping(decision.get("rgb_history_probe"), label="failure decision probe")
    if any(probe.get(key) is not False for key in ("run", "opened", "scored")):
        raise RuntimeError("failure decision reports RGB-probe work")
    summary = _mapping(decision.get("summary"), label="failure decision summary")
    expected_summary = {
        "formal_build_completed": False,
        "train_generation_completed": True,
        "train_lance_commit_completed": False,
        "development_split_started": False,
        "scientific_data_gates_reached": False,
        "rgb_history_probe_reached": False,
        "development_ready": False,
    }
    if {key: summary.get(key) for key in expected_summary} != expected_summary:
        raise RuntimeError("failure decision summary mismatch")
    formal = _mapping(decision.get("formal_build"), label="failure decision formal build")
    train = _mapping(formal.get("train_generation"), label="failure train generation")
    if (
        formal.get("attempt_consumed") is not True
        or int(train.get("accepted_pairs", -1)) != 2048
        or int(train.get("attempted_candidates", -1)) != 2048
        or int(train.get("rejected_candidates", -1)) != 0
        or train.get("action_anchor_counts")
        != {"endpoint4": 512, "front_hold": 512, "plateau": 512, "ramp4": 512}
        or train.get("profile_constraints_passed") is not True
        or formal.get("train_lance_commit_completed") is not False
        or formal.get("loader_validation_started") is not False
        or formal.get("development_started") is not False
        or formal.get("build_report_written") is not False
        or formal.get("manifest_written") is not False
    ):
        raise RuntimeError("failure decision formal-build stage mismatch")
    failed_sets = _mapping(
        decision.get("failed_content_exclusions"),
        label="failure decision failed content exclusions",
    )
    if {
        key: _entry_summary(failed_sets.get(key), label=f"decision failed {key}")
        for key in FAILED_SET_IDENTITIES
    } != FAILED_SET_IDENTITIES:
        raise RuntimeError("failure decision failed-set identities mismatch")
    union = _mapping(
        decision.get("recovery_exclusion_union"),
        label="failure decision recovery exclusion union",
    )
    if {
        key: _entry_summary(union.get(key), label=f"decision union {key}")
        for key in RECOVERY_UNION_IDENTITIES
    } != RECOVERY_UNION_IDENTITIES:
        raise RuntimeError("failure decision recovery-union identities mismatch")
    overlap = _mapping(decision.get("prior_overlap"), label="failure decision overlap")
    if overlap.get("passed") is not True:
        raise RuntimeError("failure decision prior-overlap gate failed")
    for key in FAILED_SET_IDENTITIES:
        entry = _mapping(overlap.get(key), label=f"failure decision overlap {key}")
        if int(entry.get("count", -1)) != 0 or entry.get("values") != []:
            raise RuntimeError(f"failure decision reports prior overlap: {key}")
    claims = _mapping(decision.get("claims"), label="failure decision claims")
    for key in (
        "development_ready",
        "data_readiness_passed",
        "release_claim_allowed",
        "suite_registration_allowed",
        "public_test_claim_allowed",
        "scientific_gate_failure_claimed",
    ):
        if claims.get(key) is not False:
            raise RuntimeError(f"failure decision claim {key} must be false")
    policy = _mapping(decision.get("recovery_policy"), label="failure recovery policy")
    expected_policy = {
        "original_v4_formal_attempt_consumed": True,
        "retry_authorized_under_original_preregistration": False,
        "new_frozen_recovery_preregistration_required": True,
        "original_failed_tree_must_remain_immutable": True,
        "partial_output_promotable": False,
    }
    if {key: policy.get(key) for key in expected_policy} != expected_policy:
        raise RuntimeError("failure decision recovery policy mismatch")
    decision_failed_output = _mapping(
        decision.get("failed_output"), label="failure decision failed output"
    )
    receipt_failed_output = _mapping(
        failed_document.get("failed_output"), label="failed receipt failed output"
    )
    if (
        decision_failed_output.get("allowed_inventory_only") is not True
        or decision_failed_output.get("immutable_partial_not_canonical_dataset") is not True
        or decision_failed_output.get("logical_root")
        != receipt_failed_output.get("logical_root")
        or decision_failed_output.get("inventory")
        != receipt_failed_output.get("inventory")
        or decision_failed_output.get("lance_versions_directory_empty") is not True
        or decision_failed_output.get("lance_transactions_directory_empty") is not True
    ):
        raise RuntimeError("failure decision does not freeze an exact failed inventory")
    for aliases, expected, label in (
        (("original_v4_freeze_receipt", "freeze_receipt", "freeze"), old_freeze, "old freeze"),
        (("old_final_prior_receipt", "prior_exclusion_receipt", "final_prior", "final_prior_exclusion_receipt"), old_prior, "old prior"),
        (("failed_formal_attempt_receipt", "failed_attempt_receipt", "failed_receipt"), failed, "failed receipt"),
        (("query_reconstruction_receipt", "failed_attempt_query_reconstruction_receipt", "query_receipt"), query, "query receipt"),
    ):
        binding = _find_binding(decision, aliases)
        if binding is None:
            raise RuntimeError(f"failure decision lacks {label} binding")
        _same_identity(binding, expected, label=f"failure decision {label}")
    for alias in (
        "current_old_preregistration",
        "original_preregistration_snapshot",
    ):
        binding = _find_binding(decision, (alias,))
        if binding is None:
            raise RuntimeError(f"failure decision lacks {alias} binding")
        _same_identity(binding, old_prereg, label=f"failure decision {alias}")

    code = _mapping(decision.get("original_frozen_code"), label="original_frozen_code")
    freeze_identity = _mapping(
        old_freeze_document.get("identity"), label="old freeze identity"
    )
    required_code_aliases = {
        "builder": ("builder", "builder_snapshot", "v4_builder"),
        "physics": ("physics", "physics_snapshot", "v4_physics"),
        "probe": ("probe", "probe_snapshot", "v4_probe", "rgb_probe"),
        "probe_tests": (
            "probe_tests",
            "probe_tests_snapshot",
            "v4_probe_tests",
            "rgb_probe_tests",
        ),
        "action_support": (
            "action_support",
            "action_support_snapshot",
            "v4_action_support_audit",
        ),
        "action_support_tests": (
            "action_support_tests",
            "action_support_tests_snapshot",
            "v4_action_support_audit_tests",
        ),
    }
    old_freeze_names = {
        "builder": "v4_builder",
        "physics": "v4_physics",
        "probe": "v4_probe",
        "probe_tests": "v4_probe_tests",
        "action_support": "v4_action_support_audit",
        "action_support_tests": "v4_action_support_audit_tests",
    }
    for label, aliases in required_code_aliases.items():
        matches = [code[name] for name in aliases if isinstance(code.get(name), Mapping)]
        if len(matches) != 1:
            raise RuntimeError(f"failure decision lacks exactly one old {label} snapshot")
        frozen = freeze_identity.get(old_freeze_names[label])
        _same_identity(matches[0], frozen, label=f"old {label} snapshot/freeze")


def _validate_source(path: Path, declared: Any) -> dict[str, Any]:
    _reject_public(path, label="source H5")
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError("source H5 must be a regular non-symlink file")
    _validate_source_binding(declared, label="prereg")
    if path.stat().st_size != SOURCE_SIZE_BYTES or file_sha256(path) != SOURCE_SHA256:
        raise RuntimeError("source H5 full-file identity mismatch")
    with h5py.File(path, "r", swmr=True) as handle:
        if "action" not in handle or "ep_len" not in handle:
            raise RuntimeError("source H5 lacks action or ep_len")
        action = handle["action"]
        if tuple(action.shape) != SOURCE_ACTION_SHAPE or str(action.dtype) != SOURCE_ACTION_DTYPE:
            raise RuntimeError("source H5 action shape/dtype mismatch")
        if int(handle["ep_len"].shape[0]) != SOURCE_EPISODE_COUNT:
            raise RuntimeError("source H5 episode count mismatch")
    return {
        "symbol": SOURCE_SYMBOL,
        "path_recorded": False,
        "sha256": SOURCE_SHA256,
        "size_bytes": SOURCE_SIZE_BYTES,
        "row_count": SOURCE_ROW_COUNT,
        "episode_count": SOURCE_EPISODE_COUNT,
    }


def _verify_postflight(path: Path, raw: bytes, *, label: str) -> None:
    if _read_bytes_nofollow(path, label=f"postflight {label}") != raw:
        raise RuntimeError(f"{label} mutated during freeze")


def freeze(
    *,
    prereg_path: Path,
    artifact_root: Path,
    source_h5: Path,
    original_v4_prereg: Path,
    original_v4_freeze_receipt: Path,
    old_final_prior: Path,
    infrastructure_failure_decision: Path,
    failed_attempt_receipt: Path,
    query_reconstruction_receipt: Path,
    action_support_audit: Path,
    output: Path,
) -> dict[str, Any]:
    for label, path in (
        ("prereg", prereg_path),
        ("artifact root", artifact_root),
        ("source H5", source_h5),
        ("original v4 prereg", original_v4_prereg),
        ("original v4 freeze", original_v4_freeze_receipt),
        ("old final prior", old_final_prior),
        ("infrastructure failure decision", infrastructure_failure_decision),
        ("failed attempt receipt", failed_attempt_receipt),
        ("query reconstruction receipt", query_reconstruction_receipt),
        ("v4r1 action-support audit", action_support_audit),
        ("output", output),
    ):
        _reject_public(path, label=label)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite {output}")

    prereg_raw, prereg = _read_yaml(prereg_path, label="v4r1 preregistration")
    if _contains_placeholder(prereg):
        raise RuntimeError("v4r1 preregistration contains an unresolved placeholder")
    _validate_prereg_scope(prereg)
    science = _scientific_contract(prereg)
    recovery, storage = _validate_recovery_contract(prereg)
    data, probe = _validate_data_and_probe(prereg)
    prior_contract = _validate_union_declaration(prereg)
    capacity = _validate_recovery_capacity(prereg)
    action_support_authorization = _validate_action_support_authorization(prereg)

    try:
        artifact_metadata = os.lstat(artifact_root)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"artifact root is missing: {artifact_root}") from error
    if not stat.S_ISDIR(artifact_metadata.st_mode):
        raise RuntimeError("artifact root must be a real non-symlink directory")

    raw_inputs: dict[str, bytes] = {}
    documents: dict[str, dict[str, Any]] = {}
    for key, path, kind in (
        ("original_v4_preregistration", original_v4_prereg, "yaml"),
        ("original_v4_freeze_receipt", original_v4_freeze_receipt, "json"),
        ("old_final_prior_receipt", old_final_prior, "json"),
        ("infrastructure_failure_decision", infrastructure_failure_decision, "json"),
        ("failed_formal_attempt_receipt", failed_attempt_receipt, "json"),
        ("query_reconstruction_receipt", query_reconstruction_receipt, "json"),
        ("v4r1_action_support_audit", action_support_audit, "json"),
    ):
        raw, document = _read_yaml(path, label=key) if kind == "yaml" else _read_json(path, label=key)
        raw_inputs[key] = raw
        documents[key] = document

    declared_inputs = _mapping(prereg.get("recovery_inputs"), label="recovery_inputs")
    if set(declared_inputs) != set(REQUIRED_INPUT_KEYS):
        raise RuntimeError("recovery_inputs must contain exactly the required bindings")
    explicit_paths = {
        "original_v4_preregistration": original_v4_prereg,
        "original_v4_freeze_receipt": original_v4_freeze_receipt,
        "old_final_prior_receipt": old_final_prior,
        "infrastructure_failure_decision": infrastructure_failure_decision,
        "failed_formal_attempt_receipt": failed_attempt_receipt,
        "query_reconstruction_receipt": query_reconstruction_receipt,
        "v4r1_action_support_audit": action_support_audit,
    }
    verified_inputs = {
        key: _verify_declared_file(
            declared_inputs[key], artifact_root=artifact_root, label=f"recovery_inputs.{key}", explicit_path=path,
            expected_logical_path=EXPECTED_INPUT_LOGICAL_PATHS[key],
        )
        for key, path in explicit_paths.items()
    }
    verified_inputs["source_h5"] = _validate_source(source_h5, declared_inputs["source_h5"])

    _require_exact_file(
        verified_inputs["original_v4_preregistration"],
        digest=EXPECTED_OLD_PREREG_SHA256,
        size=EXPECTED_OLD_PREREG_SIZE_BYTES,
        label="original v4 preregistration",
    )
    _require_exact_file(
        verified_inputs["original_v4_freeze_receipt"],
        digest=EXPECTED_OLD_FREEZE_SHA256,
        size=EXPECTED_OLD_FREEZE_SIZE_BYTES,
        label="original v4 freeze receipt",
    )
    _require_exact_file(
        verified_inputs["old_final_prior_receipt"],
        digest=EXPECTED_OLD_PRIOR_SHA256,
        size=EXPECTED_OLD_PRIOR_SIZE_BYTES,
        label="old final prior",
    )
    _require_exact_file(
        verified_inputs["infrastructure_failure_decision"],
        digest=EXPECTED_FAILURE_DECISION_SHA256,
        size=EXPECTED_FAILURE_DECISION_SIZE_BYTES,
        label="infrastructure failure decision",
    )
    _require_exact_file(
        verified_inputs["failed_formal_attempt_receipt"],
        digest=EXPECTED_FAILED_SHA256,
        size=EXPECTED_FAILED_SIZE_BYTES,
        label="failed formal attempt receipt",
    )
    _require_exact_file(
        verified_inputs["query_reconstruction_receipt"],
        digest=EXPECTED_QUERY_SHA256,
        size=EXPECTED_QUERY_SIZE_BYTES,
        label="query reconstruction receipt",
    )
    _require_exact_file(
        verified_inputs["v4r1_action_support_audit"],
        digest=EXPECTED_ACTION_SUPPORT_SHA256,
        size=EXPECTED_ACTION_SUPPORT_SIZE_BYTES,
        label="v4r1 action-support audit",
    )

    _validate_original_prereg(documents["original_v4_preregistration"])
    _validate_original_freeze(
        documents["original_v4_freeze_receipt"],
        old_prereg=verified_inputs["original_v4_preregistration"],
    )
    _validate_old_prior(
        documents["old_final_prior_receipt"],
        old_prereg=verified_inputs["original_v4_preregistration"],
        old_freeze=verified_inputs["original_v4_freeze_receipt"],
    )
    _validate_failed_receipt(
        documents["failed_formal_attempt_receipt"],
        old_prereg=verified_inputs["original_v4_preregistration"],
        old_freeze=verified_inputs["original_v4_freeze_receipt"],
        old_prior=verified_inputs["old_final_prior_receipt"],
    )
    _validate_query_receipt(
        documents["query_reconstruction_receipt"],
        failed=verified_inputs["failed_formal_attempt_receipt"],
        old_prior=verified_inputs["old_final_prior_receipt"],
    )
    _validate_action_support_audit(documents["v4r1_action_support_audit"])
    _validate_failure_decision(
        documents["infrastructure_failure_decision"],
        old_prereg=verified_inputs["original_v4_preregistration"],
        old_freeze=verified_inputs["original_v4_freeze_receipt"],
        old_freeze_document=documents["original_v4_freeze_receipt"],
        old_prior=verified_inputs["old_final_prior_receipt"],
        failed=verified_inputs["failed_formal_attempt_receipt"],
        failed_document=documents["failed_formal_attempt_receipt"],
        query=verified_inputs["query_reconstruction_receipt"],
    )

    declared_identity = _mapping(prereg.get("identity"), label="identity")
    if set(declared_identity) != set(REQUIRED_IDENTITY_KEYS):
        raise RuntimeError("identity must contain exactly the required implementation files")
    identity = {
        key: _verify_declared_file(
            declared_identity[key], artifact_root=artifact_root, label=f"identity.{key}",
            expected_logical_path=EXPECTED_IDENTITY_PATHS[key],
        )
        for key in REQUIRED_IDENTITY_KEYS
    }
    identity_paths = {
        key: _resolve_declared_path(
            str(identity[key]["path"]), artifact_root=artifact_root
        )
        for key in REQUIRED_IDENTITY_KEYS
    }

    receipt = {
        "schema_version": 1,
        "protocol_id": SCIENTIFIC_PROTOCOL_ID,
        "scientific_protocol_id": SCIENTIFIC_PROTOCOL_ID,
        "recovery_authorization_id": RECOVERY_AUTHORIZATION_ID,
        "status": FREEZE_STATUS,
        "checks_passed": True,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "one_v4r1_Training_Development_recovery_build_and_one_frozen_rgb_probe",
        "preregistration": _identity(
            prereg_path,
            prereg_raw,
            recorded_path=(
                prereg_path.relative_to(ROOT).as_posix()
                if prereg_path.is_relative_to(ROOT)
                else prereg_path.as_posix()
            ),
        ),
        "authorization_inputs": verified_inputs,
        "identity": identity,
        "source_h5": verified_inputs["source_h5"],
        "scientific_protocol_contract": science,
        "recovery_contract": recovery,
        "storage_publication_contract": storage,
        "data_contract": data,
        "recovery_prior_exclusion_contract": prior_contract,
        "recovery_capacity_check": capacity,
        "action_support_authorization": action_support_authorization,
        "rgb_history_probe": probe,
        "authorized_splits": list(ACTIVE_SPLITS),
        "recovery_build_attempts_authorized": 1,
        "rgb_history_probe_attempts_authorized": 1,
        "public_test": {
            "access_status": "closed_not_read_not_scored",
            "generated": False,
            "opened": False,
            "read": False,
            "hashed": False,
            "scored": False,
        },
        "reference_model_training_or_scoring_authorized": False,
        "reference_model_optimizer_steps_authorized": 0,
    }

    _verify_postflight(prereg_path, prereg_raw, label="v4r1 preregistration")
    for key, path in explicit_paths.items():
        _verify_postflight(path, raw_inputs[key], label=key)
    for key in REQUIRED_IDENTITY_KEYS:
        postflight_identity = _verify_declared_file(
            declared_identity[key],
            artifact_root=artifact_root,
            label=f"postflight identity.{key}",
            explicit_path=identity_paths[key],
            expected_logical_path=EXPECTED_IDENTITY_PATHS[key],
        )
        _same_identity(
            postflight_identity,
            identity[key],
            label=f"postflight identity.{key}",
        )
    postflight_source = _validate_source(source_h5, declared_inputs["source_h5"])
    if postflight_source != verified_inputs["source_h5"]:
        raise RuntimeError("source H5 identity changed during recovery freeze")
    output.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(output, flags, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return receipt


def parse_args(values: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prereg", type=Path, default=DEFAULT_PREREG)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--source-h5", type=Path, required=True)
    parser.add_argument("--original-v4-prereg", type=Path, required=True)
    parser.add_argument("--original-v4-freeze-receipt", type=Path, required=True)
    parser.add_argument("--old-final-prior", type=Path, required=True)
    parser.add_argument("--infrastructure-failure-decision", type=Path, required=True)
    parser.add_argument("--failed-attempt-receipt", type=Path, required=True)
    parser.add_argument("--query-reconstruction-receipt", type=Path, required=True)
    parser.add_argument("--action-support-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(values)
    for name, value in vars(args).items():
        _reject_public(value, label=name)
    return args


def _absolute_without_resolve(path: Path) -> Path:
    """Make a CLI path absolute without dereferencing its final symlink."""

    return Path(os.path.abspath(path.expanduser()))


def main(values: Sequence[str] | None = None) -> None:
    args = parse_args(values)
    receipt = freeze(
        prereg_path=_absolute_without_resolve(args.prereg),
        artifact_root=_absolute_without_resolve(args.artifact_root),
        source_h5=_absolute_without_resolve(args.source_h5),
        original_v4_prereg=_absolute_without_resolve(args.original_v4_prereg),
        original_v4_freeze_receipt=_absolute_without_resolve(
            args.original_v4_freeze_receipt
        ),
        old_final_prior=_absolute_without_resolve(args.old_final_prior),
        infrastructure_failure_decision=_absolute_without_resolve(
            args.infrastructure_failure_decision
        ),
        failed_attempt_receipt=_absolute_without_resolve(
            args.failed_attempt_receipt
        ),
        query_reconstruction_receipt=_absolute_without_resolve(
            args.query_reconstruction_receipt
        ),
        action_support_audit=_absolute_without_resolve(
            args.action_support_audit
        ),
        output=_absolute_without_resolve(args.output),
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "status": receipt["status"],
                "checks_passed": receipt["checks_passed"],
                "public_test_read": False,
                "model_optimizer_steps_authorized": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
