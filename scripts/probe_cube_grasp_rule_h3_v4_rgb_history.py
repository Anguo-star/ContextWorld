#!/usr/bin/env python3
"""Run the frozen Development-only Cube v4 RGB-history probe.

Only a completed v4r1 recovery publication may be probed.  Its success marker
and every published file identity are verified before ``train.lance`` and
``loader_validation.lance`` are opened.  Public Test (``validation.lance``)
is closed by protocol and is rejected before any Lance dataset is read.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import platform
import re
import stat
import sys
from typing import Any, Iterator, Mapping, Sequence

import lance
import numpy as np
import PIL
from PIL import Image, UnidentifiedImageError
import pyarrow as pa
import scipy
import sklearn
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler
import yaml

from contextworld.evaluation.cube_grasp_rule_h3_v4 import (
    V4R1_FORMAL_CATALOG_INDEX_OFFSET,
    action_blocks as frozen_v4_action_blocks,
    make_v4_action_profile,
)
from contextworld.paths import artifact_path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


PROTOCOL = "cube_gripper_carry_rule_history3_development_v4"
RECOVERY_AUTHORIZATION_ID = "cube_gripper_carry_h3_development_v4r1"
PROBE_ID = "cube_gripper_carry_h3_v4r1_rgb_history_probe_v1"
SUCCESS_MARKER_NAME = "_SUCCESS.json"
PREREG_STATUS = "preregistered_before_v4r1_recovery_build"
FREEZE_STATUS = "frozen_before_v4r1_recovery_build"
PRIOR_RECEIPT_ID = "cube_gripper_carry_h3_v4r1_prior_exclusions_final_v1"
ACTIVE_SPLITS = ("train", "loader_validation")
EXPECTED_PAIR_COUNTS = {"train": 2048, "loader_validation": 256}
EXPECTED_MODEL_ROWS = {
    split: 8 * pair_count
    for split, pair_count in EXPECTED_PAIR_COUNTS.items()
}
TABLE_NAMES = {
    "train": "train.lance",
    "loader_validation": "loader_validation.lance",
}
HIDDEN_MODES = ("cannot_hold", "can_hold")
LABEL_ENCODING = {"cannot_hold": 0, "can_hold": 1}
ACTION_ANCHORS = ("endpoint4", "plateau", "ramp4", "front_hold")
MODEL_STEPS = (0, 1, 2, 3)
DECODED_HISTORY_STEPS = (0, 1, 2)
RESIZE_SHAPE = (16, 16)
ACTION_PROFILE_SHAPE = (4, 5, 5)
METADATA_FILE_NAMES = ("request.json", "build_report.json", "manifest.json")
FORMAL_CATALOG_POOL_COUNTS = {"train": 4096, "loader_validation": 512}

FROZEN_ARROW_SCHEMA = pa.schema(
    [
        pa.field("episode_idx", pa.int32()),
        pa.field("model_step_idx", pa.int32()),
        pa.field("pixels", pa.binary()),
        pa.field("action_block", pa.list_(pa.float32(), 25)),
        pa.field("physical_state", pa.list_(pa.float32(), 7)),
        pa.field("hidden_grasp_enabled", pa.list_(pa.float32(), 1)),
        pa.field("pair_id", pa.string()),
        pa.field("hidden_mode", pa.string()),
        pa.field("split", pa.string()),
        pa.field("catalog_index", pa.int32()),
        pa.field("source_row", pa.int64()),
        pa.field("source_episode", pa.int32()),
        pa.field("source_step", pa.int32()),
        pa.field("action_anchor_id", pa.string()),
        pa.field("action_profile_id", pa.string()),
        pa.field("scene_template_content_hash", pa.string()),
        pa.field("pair_content_hash", pa.string()),
    ]
)

TRUSTED_INPUT_CONTRACT = {
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
    "prior_exclusion_receipt_id": PRIOR_RECEIPT_ID,
    "prior_exclusion_receipt_status": FREEZE_STATUS,
    "freeze_receipt_must_bind_exact_preregistration": True,
    "prior_exclusion_receipt_must_bind_exact_preregistration_and_freeze": True,
    "metadata_files_parsed_before_lance_open": list(METADATA_FILE_NAMES),
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

FROZEN_SCIENTIFIC_CONTRACT = {
    "unchanged_from_original_v4": True,
    "history_tokens": 3,
    "context_transitions": 2,
    "prediction_horizon_action_blocks": 1,
    "raw_steps_per_action_block": 5,
    "can_hold_vertical_force_coupling_n": 0.40,
    "hidden_modes": list(HIDDEN_MODES),
    "action_temporal_pattern": ["p", "negative_p", "p", "terminal_zero"],
    "action_anchor_ids": list(ACTION_ANCHORS),
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

FROZEN_STORAGE_CONTRACT = {
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
    "success_marker_name": SUCCESS_MARKER_NAME,
    "success_marker_written_last": True,
    "failed_copy_marked_complete": False,
}

FROZEN_PROBE_RECIPE = {
    "input": "decoded_x0_x1_x2_rgb_only",
    "resize_shape": [16, 16],
    "resize_interpolation": "Pillow_Resampling_BILINEAR",
    "arithmetic_dtype": "float64",
    "fixed_feature": "flatten(2*x1-x0-x2)_C_order",
    "standard_scaler_fit_split_only": "train",
    "estimator": "StandardScaler_then_RidgeClassifier_alpha_1",
    "label_encoding": dict(LABEL_ENCODING),
}

FROZEN_PROBE_THRESHOLDS = {
    "overall_accuracy_minimum": 0.75,
    "worst_mode_accuracy_minimum": 0.70,
    "worst_anchor_family_accuracy_minimum": 0.70,
    "pair_cluster_bootstrap_lower_bound_minimum": 0.70,
    "label_permutation_mean_accuracy_maximum": 0.60,
    "x0_only_accuracy_maximum": 0.51,
    "query_only_accuracy_maximum": 0.51,
    "action_only_accuracy_maximum": 0.51,
}

FROZEN_RECOVERY_CONTRACT = {
    "failure_class": "infrastructure_lance_atomic_rename_eperm",
    "original_v4_formal_attempt_consumed": True,
    "retry_under_original_v4_preregistration_authorized": False,
    "original_failed_tree_immutable": True,
    "scientific_protocol_changed": False,
    "recovery_build_attempts_authorized": 1,
    "builder_or_lance_smoke_attempts_authorized": 0,
    "rgb_history_probe_attempts_authorized": 1,
    "formal_catalog_index_offset": V4R1_FORMAL_CATALOG_INDEX_OFFSET,
    "formal_catalog_offset_four_aligned": True,
    "failed_batch_identities_must_be_excluded": True,
}

FROZEN_DATA_CONTRACT = {
    "logical_output_root": (
        "artifacts/synthesis/cube_gripper_carry_rule_h3_development_v4r1"
    ),
    "authorized_splits": list(ACTIVE_SPLITS),
    "pair_counts": dict(EXPECTED_PAIR_COUNTS),
    "workers": 16,
    "episodes_per_pair": 2,
    "rows_per_pair": 8,
    "pairs_per_anchor": {"train": 512, "loader_validation": 64},
    "formal_catalog_index_offset": V4R1_FORMAL_CATALOG_INDEX_OFFSET,
    "catalog_index_offset_modulo_anchor_count": 0,
    "source_episode_overlap_between_splits_required": 0,
    "action_profile_overlap_between_splits_required": 0,
    "scene_template_overlap_between_splits_required": 0,
    "pair_content_overlap_between_splits_required": 0,
    "query_pixel_overlap_between_splits_required": 0,
}

CANONICAL_PREREG_PATH = ROOT / (
    "configs/benchmark/cube_gripper_carry_h3_development_recovery_prereg_v4r1.yaml"
)
CANONICAL_FREEZE_RECEIPT_PATH = artifact_path(
    "evaluation/history3/cube_gripper_carry_h3_development_v4r1/"
    "development_recovery_freeze_receipt_v1.json"
)
CANONICAL_PRIOR_EXCLUSION_PATH = artifact_path(
    "evaluation/history3/cube_gripper_carry_h3_development_v4r1/"
    "prior_episode_exclusions_final_v1.json"
)
CANONICAL_ARTIFACT_ROOT = artifact_path(
    "synthesis/cube_gripper_carry_rule_h3_development_v4r1"
)
CANONICAL_OUTPUT_PATH = artifact_path(
    "evaluation/history3/cube_gripper_carry_h3_development_v4r1/"
    "rgb_history_probe_v1.json"
)

FROZEN_IDENTITY_PATHS = {
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
    "recovery_protocol_document": (
        "docs/protocols/Cube_Gripper_Carry_History3_Development_v4r1_Recovery_Protocol.md"
    ),
}

EXPECTED_PRIOR_IDENTITIES = {
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

BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 2026081203
BOOTSTRAP_LOWER_QUANTILE = 0.025
PERMUTATION_REPETITIONS = 16
PERMUTATION_SEED = 2026081204

OVERALL_ACCURACY_MINIMUM = 0.75
WORST_MODE_ACCURACY_MINIMUM = 0.70
WORST_ANCHOR_ACCURACY_MINIMUM = 0.70
BOOTSTRAP_LOWER_BOUND_MINIMUM = 0.70
PERMUTATION_MEAN_ACCURACY_MAXIMUM = 0.60
SHORTCUT_ACCURACY_MAXIMUM = 0.51

TABLE_COLUMNS = (
    "model_step_idx",
    "pixels",
    "action_block",
    "hidden_grasp_enabled",
    "pair_id",
    "hidden_mode",
    "split",
    "catalog_index",
    "source_episode",
    "action_anchor_id",
    "action_profile_id",
    "scene_template_content_hash",
    "pair_content_hash",
)
METADATA_ACTION_COLUMNS = tuple(
    name for name in TABLE_COLUMNS if name != "pixels"
)
PIXEL_JOIN_COLUMNS = (
    "pair_id",
    "hidden_mode",
    "split",
    "model_step_idx",
    "pixels",
)
PIXEL_FILTER = "model_step_idx <= 2"
MAIN_FEATURE_COLUMNS = ("pixels",)
NEGATIVE_CONTROL_ONLY_COLUMNS = ("action_block",)
AUDIT_ONLY_COLUMNS = tuple(
    name
    for name in TABLE_COLUMNS
    if name not in MAIN_FEATURE_COLUMNS + NEGATIVE_CONTROL_ONLY_COLUMNS
)
PRIVILEGED_COLUMNS_EXCLUDED_FROM_MAIN_FEATURE = (
    "physical_state",
    "hidden_grasp_enabled",
    "episode_idx",
    "model_step_idx",
    "pair_id",
    "hidden_mode",
    "split",
    "catalog_index",
    "source_row",
    "source_episode",
    "source_step",
    "action_anchor_id",
    "action_profile_id",
    "scene_template_content_hash",
    "pair_content_hash",
)


@dataclass(frozen=True)
class _Condition:
    pair_id: str
    hidden_mode: str
    action_anchor_id: str
    action_profile_id: str
    catalog_index: int
    source_episode: int
    scene_template_content_hash: str
    pair_content_hash: str
    x0_jpeg: bytes
    query_jpeg: bytes
    x0_rgb: np.ndarray
    query_rgb: np.ndarray
    main_feature: np.ndarray
    action_blocks: np.ndarray


@dataclass(frozen=True)
class PreparedSplit:
    split: str
    main_features: np.ndarray
    x0_features: np.ndarray
    query_features: np.ndarray
    action_features: np.ndarray
    labels: np.ndarray
    pair_ids: np.ndarray
    hidden_modes: np.ndarray
    action_anchors: np.ndarray
    action_profile_ids: frozenset[str]
    source_episodes: frozenset[int]
    query_pixel_hashes: frozenset[str]
    scene_template_content_hashes: frozenset[str]
    pair_content_hashes: frozenset[str]
    pair_count: int
    condition_count: int
    row_count: int
    anchor_pair_counts: Mapping[str, int]


@dataclass(frozen=True)
class _FitResult:
    predictions: np.ndarray
    transformed_train: np.ndarray
    transformed_development: np.ndarray
    receipt: Mapping[str, Any]


def _array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def _require_sha256(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{field_name} must be a canonical lowercase SHA256")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _read_bytes_nofollow(path: Path, *, label: str) -> bytes:
    """Read one explicitly supplied regular file without following its leaf."""

    value = Path(os.path.abspath(Path(path).expanduser()))
    forbidden = _forbidden_closed_component(value)
    if forbidden is not None:
        raise ValueError(f"{label} has forbidden validation/Public component {forbidden!r}")
    current = Path(value.anchor)
    for part in value.parts[1:]:
        current /= part
        try:
            component = os.lstat(current)
        except FileNotFoundError as error:
            raise FileNotFoundError(f"{label} is missing: {value}") from error
        if stat.S_ISLNK(component.st_mode):
            raise ValueError(f"{label} path cannot contain symlinks: {current}")
    try:
        metadata = os.lstat(value)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"{label} is missing: {value}") from error
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{label} must be a regular non-symlink file")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(value, flags)
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or (
            opened.st_dev,
            opened.st_ino,
        ) != (metadata.st_dev, metadata.st_ino):
            raise RuntimeError(f"{label} changed while it was opened")
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = -1
            return stream.read()
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _decode_mapping(raw: bytes, *, label: str, yaml_document: bool) -> dict[str, Any]:
    try:
        text = raw.decode("utf-8")
        value = yaml.safe_load(text) if yaml_document else json.loads(text)
    except (UnicodeDecodeError, json.JSONDecodeError, yaml.YAMLError) as error:
        kind = "YAML" if yaml_document else "JSON"
        raise ValueError(f"{label} is not valid UTF-8 {kind}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} root must be an object")
    return value


def _raw_identity(raw: bytes) -> dict[str, Any]:
    return {"sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw)}


def _identity_core(value: Any, *, label: str) -> tuple[str, int]:
    entry = _mapping(value, label=label)
    digest = _require_sha256(entry.get("sha256"), field_name=f"{label}.sha256")
    size = entry.get("size_bytes")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ValueError(f"{label}.size_bytes must be a positive integer")
    return digest, size


def _require_identity_binding(
    value: Any, *, expected: Mapping[str, Any], label: str
) -> None:
    if _identity_core(value, label=label) != _identity_core(
        expected, label=f"expected {label}"
    ):
        raise ValueError(f"{label} identity mismatch")


def _require_public_closed(
    value: Any, *, label: str, require_validation_gate: bool = False
) -> None:
    gate = _mapping(value, label=label)
    if gate.get("access_status") != "closed_not_read_not_scored" or any(
        gate.get(name) is not False for name in ("opened", "read", "hashed", "scored")
    ):
        raise ValueError(f"{label} does not keep Public Test fully closed")
    if require_validation_gate and gate.get("validation_lance_access_allowed") is not False:
        raise ValueError(f"{label} must forbid validation.lance access")


def _require_model_closed(document: Mapping[str, Any], *, label: str) -> None:
    for name in (
        "reference_model_training_or_scoring",
        "reference_model_training_or_scoring_authorized",
    ):
        if name in document and document.get(name) is not False:
            raise ValueError(f"{label}.{name} must be false")
    for name in (
        "reference_model_optimizer_steps",
        "reference_model_optimizer_steps_authorized",
    ):
        if name in document and document.get(name) != 0:
            raise ValueError(f"{label}.{name} must be zero")
    phase = document.get("reference_model_phase")
    if isinstance(phase, Mapping):
        expected = {
            "training_and_scoring_authorized": False,
            "trainer_invoked": False,
            "optimizer_steps_authorized": 0,
            "optimizer_steps_run": 0,
            "checkpoint_creation_authorized": False,
        }
        if {name: phase.get(name) for name in expected} != expected:
            raise ValueError(f"{label}.reference_model_phase is not fully closed")


def _require_protocol_scope(document: Mapping[str, Any], *, label: str) -> None:
    protocol = document.get("protocol", document.get("protocol_id"))
    if protocol != PROTOCOL:
        raise ValueError(f"{label} protocol mismatch")
    if "scientific_protocol_id" in document and document.get(
        "scientific_protocol_id"
    ) != PROTOCOL:
        raise ValueError(f"{label} scientific protocol mismatch")
    if document.get("recovery_authorization_id") != RECOVERY_AUTHORIZATION_ID:
        raise ValueError(f"{label} recovery authorization mismatch")


def _require_exact_subset(
    value: Any, expected: Mapping[str, Any], *, label: str
) -> Mapping[str, Any]:
    entry = _mapping(value, label=label)
    if {name: entry.get(name) for name in expected} != dict(expected):
        raise ValueError(f"{label} differs from the frozen contract")
    return entry


def _canonical_source_episode_digest(values: Sequence[int]) -> str:
    normalized = [int(value) for value in values]
    payload = np.asarray(normalized, dtype="<i8").tobytes()
    return hashlib.sha256(
        b"contextworld-cube-prior-source-episodes-v1\0" + payload
    ).hexdigest()


def _canonical_prior_content_digest(
    values: Sequence[str], *, field_name: str
) -> str:
    decoded = b"".join(
        bytes.fromhex(_require_sha256(value, field_name=field_name))
        for value in values
    )
    return hashlib.sha256(
        b"contextworld-cube-prior-content-exclusions-v1\0"
        + field_name.encode("ascii")
        + b"\0"
        + decoded
    ).hexdigest()


def _validate_frozen_implementation_identity(freeze: Mapping[str, Any]) -> None:
    identities = _mapping(freeze.get("identity"), label="freeze identity")
    if set(identities) != set(FROZEN_IDENTITY_PATHS):
        raise ValueError("freeze identity set is not the complete frozen implementation")
    for name, relative_path in FROZEN_IDENTITY_PATHS.items():
        entry = _mapping(identities[name], label=f"freeze identity {name}")
        if entry.get("path") != relative_path:
            raise ValueError(f"freeze identity {name} path mismatch")
        raw = _read_bytes_nofollow(ROOT / relative_path, label=f"current {name}")
        _require_identity_binding(
            entry,
            expected=_raw_identity(raw),
            label=f"freeze/current {name}",
        )


def _validate_prior_identity_sets(prior: Mapping[str, Any]) -> dict[str, frozenset[Any]]:
    episodes_raw = prior.get("excluded_source_episodes")
    if not isinstance(episodes_raw, list) or any(
        isinstance(value, bool) or not isinstance(value, int) for value in episodes_raw
    ):
        raise ValueError("prior excluded_source_episodes must be an integer list")
    episodes = [int(value) for value in episodes_raw]
    if episodes != sorted(set(episodes)) or any(value < 0 for value in episodes):
        raise ValueError("prior source episode exclusions are not sorted/unique")
    expected_source = EXPECTED_PRIOR_IDENTITIES["source_episodes"]
    if (
        prior.get("excluded_source_episode_count") != expected_source["count"]
        or len(episodes) != expected_source["count"]
        or prior.get("excluded_source_episodes_sha256") != expected_source["sha256"]
        or _canonical_source_episode_digest(episodes) != expected_source["sha256"]
    ):
        raise ValueError("prior source episode exclusions are not the canonical union")

    content = _mapping(
        prior.get("prior_content_exclusions"), label="prior content exclusions"
    )
    expected_fields = set(EXPECTED_PRIOR_IDENTITIES) - {"source_episodes"}
    if set(content) != expected_fields:
        raise ValueError("prior content exclusion field set mismatch")
    result: dict[str, frozenset[Any]] = {"source_episodes": frozenset(episodes)}
    for field_name in sorted(expected_fields):
        entry = _mapping(content[field_name], label=f"prior {field_name}")
        values = entry.get("values")
        if not isinstance(values, list) or values != sorted(set(values)):
            raise ValueError(f"prior {field_name} is not sorted/unique")
        expected = EXPECTED_PRIOR_IDENTITIES[field_name]
        digest = _canonical_prior_content_digest(values, field_name=field_name)
        if (
            len(values) != expected["count"]
            or entry.get("count") != expected["count"]
            or entry.get("sha256") != expected["sha256"]
            or digest != expected["sha256"]
        ):
            raise ValueError(f"prior {field_name} is not the canonical union")
        result[field_name] = frozenset(values)
    return result


def validate_authorization_chain(
    *, prereg_path: Path, freeze_receipt_path: Path, prior_exclusion_path: Path
) -> dict[str, Any]:
    """Read and cross-bind the three explicit authorization documents."""

    paths = {
        "preregistration": Path(os.path.abspath(Path(prereg_path).expanduser())),
        "freeze_receipt": Path(
            os.path.abspath(Path(freeze_receipt_path).expanduser())
        ),
        "prior_exclusion_receipt": Path(
            os.path.abspath(Path(prior_exclusion_path).expanduser())
        ),
    }
    canonical = {
        "preregistration": CANONICAL_PREREG_PATH,
        "freeze_receipt": CANONICAL_FREEZE_RECEIPT_PATH,
        "prior_exclusion_receipt": CANONICAL_PRIOR_EXCLUSION_PATH,
    }
    for name, path in paths.items():
        if path != Path(os.path.abspath(canonical[name])):
            raise ValueError(f"{name} must use its canonical frozen path")
    raw = {
        name: _read_bytes_nofollow(path, label=name) for name, path in paths.items()
    }
    if any(
        token in payload.decode("utf-8", errors="ignore")
        for payload in raw.values()
        for token in ("PENDING_SHA256", "TO_BE_FROZEN", "PLACEHOLDER", "REPLACE_ME", "TBD")
    ):
        raise ValueError("authorization chain contains an unresolved placeholder")
    documents = {
        "preregistration": _decode_mapping(
            raw["preregistration"], label="preregistration", yaml_document=True
        ),
        "freeze_receipt": _decode_mapping(
            raw["freeze_receipt"], label="freeze receipt", yaml_document=False
        ),
        "prior_exclusion_receipt": _decode_mapping(
            raw["prior_exclusion_receipt"],
            label="prior exclusion receipt",
            yaml_document=False,
        ),
    }
    identities = {name: _raw_identity(payload) for name, payload in raw.items()}
    prereg = documents["preregistration"]
    freeze = documents["freeze_receipt"]
    prior = documents["prior_exclusion_receipt"]

    for label, document in documents.items():
        if document.get("schema_version") != 1:
            raise ValueError(f"{label} schema_version must be 1")
        _require_protocol_scope(document, label=label)
        _require_public_closed(
            document.get("public_test"),
            label=f"{label} Public Test",
            require_validation_gate=label == "preregistration",
        )
        _require_model_closed(document, label=label)
    if prereg.get("status") != PREREG_STATUS or prereg.get("phase") != "development_only":
        raise ValueError("preregistration status/phase mismatch")
    if prereg.get("reference_model_training_or_scoring_authorized") is not False:
        raise ValueError("preregistration must explicitly disable reference-model work")
    phase = _mapping(prereg.get("reference_model_phase"), label="prereg model phase")
    expected_phase = {
        "training_and_scoring_authorized": False,
        "trainer_invoked": False,
        "optimizer_steps_authorized": 0,
        "optimizer_steps_run": 0,
        "checkpoint_creation_authorized": False,
    }
    if {name: phase.get(name) for name in expected_phase} != expected_phase:
        raise ValueError("preregistration reference-model phase is not closed")
    if freeze.get("status") != FREEZE_STATUS or freeze.get("checks_passed") is not True:
        raise ValueError("freeze receipt is not a passing frozen authorization")
    if freeze.get("reference_model_training_or_scoring_authorized") is not False or freeze.get(
        "reference_model_optimizer_steps_authorized"
    ) != 0:
        raise ValueError("freeze receipt must explicitly disable reference-model work")
    if freeze.get("authorized_splits") != list(ACTIVE_SPLITS):
        raise ValueError("freeze receipt authorized_splits mismatch")
    if freeze.get("recovery_build_attempts_authorized") != 1 or freeze.get(
        "rgb_history_probe_attempts_authorized"
    ) != 1:
        raise ValueError("freeze receipt attempt budget mismatch")
    if (
        prior.get("receipt_id") != PRIOR_RECEIPT_ID
        or prior.get("status") != FREEZE_STATUS
        or prior.get("checks_passed") is not True
    ):
        raise ValueError("prior exclusion identity/status mismatch")
    if prior.get("reference_model_training_or_scoring") is not False or prior.get(
        "reference_model_optimizer_steps"
    ) != 0:
        raise ValueError("prior receipt must explicitly disable reference-model work")
    rgb_probe = _mapping(prior.get("rgb_probe"), label="prior rgb_probe")
    if any(rgb_probe.get(name) is not False for name in ("opened", "run", "scored")):
        raise ValueError("prior receipt must keep the RGB probe unused")
    _require_identity_binding(
        freeze.get("preregistration"),
        expected=identities["preregistration"],
        label="freeze/preregistration",
    )
    _require_identity_binding(
        prior.get("preregistration"),
        expected=identities["preregistration"],
        label="prior/preregistration",
    )
    _require_identity_binding(
        prior.get("freeze_receipt"),
        expected=identities["freeze_receipt"],
        label="prior/freeze receipt",
    )
    if prereg.get("identity") != freeze.get("identity"):
        raise ValueError("preregistration and freeze implementation identities differ")
    _validate_frozen_implementation_identity(freeze)

    for document, label in ((prereg, "preregistration"), (freeze, "freeze receipt")):
        _require_exact_subset(
            document.get("scientific_protocol_contract"),
            FROZEN_SCIENTIFIC_CONTRACT,
            label=f"{label} scientific_protocol_contract",
        )
        _require_exact_subset(
            document.get("recovery_contract"),
            FROZEN_RECOVERY_CONTRACT,
            label=f"{label} recovery_contract",
        )
        _require_exact_subset(
            document.get("storage_publication_contract"),
            FROZEN_STORAGE_CONTRACT,
            label=f"{label} storage_publication_contract",
        )
        _require_exact_subset(
            document.get("data_contract"),
            FROZEN_DATA_CONTRACT,
            label=f"{label} data_contract",
        )
        probe = _mapping(document.get("rgb_history_probe"), label=f"{label} rgb probe")
        if (
            probe.get("attempts_authorized") != 1
            or probe.get("run_only_after_complete_success_marker") is not True
            or probe.get("recipe_unchanged_from_v4") is not True
            or probe.get("thresholds_unchanged_from_v4") is not True
            or probe.get("recipe") != FROZEN_PROBE_RECIPE
            or probe.get("thresholds") != FROZEN_PROBE_THRESHOLDS
            or probe.get("trusted_input_contract") != TRUSTED_INPUT_CONTRACT
        ):
            raise ValueError(f"{label} RGB probe authorization mismatch")

    recovery = _mapping(prior.get("recovery_contract"), label="prior recovery_contract")
    required_true = (
        "scientific_protocol_unchanged_from_v4",
        "old_prior_preserved_and_extended",
        "failed_attempt_source_action_scene_pair_query_all_excluded",
        "failed_attempt_raw_queries_deterministically_reconstructed",
        "all_inputs_reverified_unchanged_before_output",
        "original_v4_attempt_not_retried_or_overwritten",
    )
    required_false = (
        "lance_opened_or_written",
        "public_test_opened_read_hashed_or_scored",
        "rgb_probe_run",
        "reference_model_training_or_scoring",
    )
    if any(recovery.get(name) is not True for name in required_true) or any(
        recovery.get(name) is not False for name in required_false
    ):
        raise ValueError("prior exclusion recovery contract is incomplete")
    coverage = _mapping(prior.get("coverage"), label="prior coverage")
    if coverage.get("v4_failed_formal_attempts") is not True:
        raise ValueError("prior exclusion does not cover the failed v4 attempt")
    artifacts = prior.get("input_artifacts")
    failed_kinds = {
        str(row.get("artifact_kind"))
        for row in artifacts
        if isinstance(row, Mapping) and row.get("role") == "v4_failed_formal_attempts"
    } if isinstance(artifacts, list) else set()
    if failed_kinds != {
        "failed_formal_attempt_receipt",
        "failed_attempt_query_reconstruction_receipt",
    }:
        raise ValueError("prior exclusion lacks both failed-attempt receipts")
    prior_sets = _validate_prior_identity_sets(prior)
    return {
        "paths": paths,
        "raw": raw,
        "documents": documents,
        "identities": identities,
        "prior_sets": prior_sets,
        "checks_passed": True,
    }


def _canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def _tree_sha256_from_receipts(receipts: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in receipts:
        digest.update(str(row["path"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(row["sha256"]).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _regular_release_file_receipts(root: Path) -> list[dict[str, Any]]:
    """Hash a real-directory tree without following aliases or special nodes."""

    release = Path(root)
    try:
        root_metadata = os.lstat(release)
    except FileNotFoundError as error:
        raise FileNotFoundError(release) from error
    if not stat.S_ISDIR(root_metadata.st_mode):
        raise ValueError("artifact root must be a real directory")

    files: list[Path] = []

    def visit(directory: Path) -> None:
        with os.scandir(directory) as iterator:
            entries = sorted(iterator, key=lambda value: value.name)
        for entry in entries:
            child = directory / entry.name
            metadata = entry.stat(follow_symlinks=False)
            relative = child.relative_to(release)
            forbidden = _forbidden_closed_component(relative)
            if forbidden is not None:
                raise ValueError(
                    "artifact release contains a validation/Public-shaped "
                    f"component before content hashing: {relative.as_posix()}"
                )
            if stat.S_ISDIR(metadata.st_mode):
                visit(child)
            elif stat.S_ISREG(metadata.st_mode):
                files.append(child)
            else:
                raise ValueError(
                    "artifact release contains a symlink or special node: "
                    f"{child.relative_to(release).as_posix()}"
                )

    visit(release)
    receipts: list[dict[str, Any]] = []
    for child in sorted(
        files, key=lambda value: value.relative_to(release).as_posix()
    ):
        relative = child.relative_to(release).as_posix()
        raw = _read_bytes_nofollow(child, label=f"release file {relative}")
        receipts.append(
            {
                "path": relative,
                "size_bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    return receipts


def _read_release_file_nofollow(root: Path, relative_path: str) -> bytes:
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe release path: {relative_path!r}")
    return _read_bytes_nofollow(root / relative, label=relative_path)


def _normalized_declared_receipts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError("success marker file receipts must be a non-empty list")
    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(value):
        if not isinstance(row, Mapping):
            raise ValueError(f"success marker receipt {index} is not an object")
        path = str(row.get("path", ""))
        pure = Path(path)
        if (
            not path
            or pure.is_absolute()
            or path != pure.as_posix()
            or ".." in pure.parts
            or path == SUCCESS_MARKER_NAME
        ):
            raise ValueError(f"success marker receipt {index} has unsafe path")
        size = row.get("size_bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError(f"success marker receipt {index} has invalid size")
        normalized.append(
            {
                "path": path,
                "size_bytes": size,
                "sha256": _require_sha256(
                    row.get("sha256"),
                    field_name=f"file_receipts_without_success_marker[{index}]",
                ),
            }
        )
    paths = [row["path"] for row in normalized]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise ValueError("success marker file receipts must be unique and sorted")
    return normalized


def validate_success_marker(root: Path) -> dict[str, Any]:
    """Verify the publisher's completion marker before any Lance open."""

    release = Path(root)
    all_receipts = _regular_release_file_receipts(release)
    marker_rows = [
        row for row in all_receipts if row["path"] == SUCCESS_MARKER_NAME
    ]
    if len(marker_rows) != 1:
        raise ValueError("completed v4r1 release requires exactly one _SUCCESS.json")
    try:
        marker_raw = _read_release_file_nofollow(release, SUCCESS_MARKER_NAME)
        payload = json.loads(marker_raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("success marker is not valid UTF-8 JSON") from error
    if not isinstance(payload, Mapping):
        raise ValueError("success marker root must be an object")
    if payload.get("schema_version") != 1:
        raise ValueError("success marker schema_version mismatch")
    if payload.get("protocol") != PROTOCOL:
        raise ValueError("success marker scientific protocol mismatch")
    if payload.get("recovery_authorization_id") != RECOVERY_AUTHORIZATION_ID:
        raise ValueError("success marker recovery authorization mismatch")
    if payload.get("status") != "complete" or payload.get("checks_passed") is not True:
        raise ValueError("success marker does not declare a complete passing release")
    if payload.get("public_test_opened") is not False or payload.get(
        "public_test_generated"
    ) is not False:
        raise ValueError("success marker does not keep Public Test closed")

    declared = _normalized_declared_receipts(
        payload.get("file_receipts_without_success_marker")
    )
    observed = [
        row for row in all_receipts if row["path"] != SUCCESS_MARKER_NAME
    ]
    if declared != observed:
        raise ValueError("published file identities differ from _SUCCESS.json")

    publication = payload.get("publication")
    if not isinstance(publication, Mapping):
        raise ValueError("success marker publication section is missing")
    receipt_digest = _canonical_json_sha256(observed)
    tree_digest = _tree_sha256_from_receipts(observed)
    expected_publication = {
        "method": "verified_x_exclusive_copytree",
        "nonempty_directory_rename_used": False,
        "success_marker_written_last": True,
        "failed_copy_is_never_marked_complete": True,
        "source_and_destination_file_receipts_equal": True,
        "file_count_without_success_marker": len(observed),
        "bytes_without_success_marker": sum(
            int(row["size_bytes"]) for row in observed
        ),
        "tree_sha256_without_success_marker": tree_digest,
        "file_receipts_sha256": receipt_digest,
    }
    for field_name, expected in expected_publication.items():
        if publication.get(field_name) != expected:
            raise ValueError(
                f"success marker publication field {field_name!r} mismatch"
            )

    receipts_by_path = {row["path"]: row for row in observed}
    bound_files = payload.get("bound_files")
    if not isinstance(bound_files, Mapping) or set(bound_files) != set(
        METADATA_FILE_NAMES
    ):
        raise ValueError("success marker must bind exactly the three metadata files")
    for name in METADATA_FILE_NAMES:
        if name not in receipts_by_path or bound_files[name] != receipts_by_path[name]:
            raise ValueError(f"success marker does not exactly bind {name}")

    lance_tables = payload.get("lance_tables")
    if not isinstance(lance_tables, Mapping) or set(lance_tables) != set(
        ACTIVE_SPLITS
    ):
        raise ValueError("success marker must bind exactly two authorized tables")
    table_receipts: dict[str, dict[str, Any]] = {}
    for split, table_name in TABLE_NAMES.items():
        entry = lance_tables.get(split)
        if not isinstance(entry, Mapping) or entry.get("table") != table_name:
            raise ValueError(f"success marker has invalid {split} Lance identity")
        prefix = f"{table_name}/"
        rows = [
            {
                "path": row["path"][len(prefix) :],
                "size_bytes": row["size_bytes"],
                "sha256": row["sha256"],
            }
            for row in observed
            if row["path"].startswith(prefix)
        ]
        checks = {
            "passed": entry.get("passed") is True,
            "schema_equals_frozen_v4": entry.get("schema_equals_frozen_v4") is True,
            "row_count": entry.get("row_count") == EXPECTED_MODEL_ROWS[split],
            "file_count": entry.get("file_count") == len(rows),
            "size_bytes": entry.get("size_bytes")
            == sum(int(row["size_bytes"]) for row in rows),
            "tree_sha256": entry.get("tree_sha256")
            == _tree_sha256_from_receipts(rows),
            "file_receipts_sha256": entry.get("file_receipts_sha256")
            == _canonical_json_sha256(rows),
        }
        if not rows or not all(checks.values()):
            raise ValueError(f"success marker {split} Lance file identity mismatch")
        table_receipts[split] = {
            "table": table_name,
            "row_count": int(entry.get("row_count", -1)),
            "file_count": len(rows),
            "size_bytes": sum(int(row["size_bytes"]) for row in rows),
            "tree_sha256": _tree_sha256_from_receipts(rows),
            "file_receipts_sha256": _canonical_json_sha256(rows),
        }

    return {
        "relative_path": SUCCESS_MARKER_NAME,
        "sha256": marker_rows[0]["sha256"],
        "size_bytes": marker_rows[0]["size_bytes"],
        "status": "complete",
        "recovery_authorization_id": RECOVERY_AUTHORIZATION_ID,
        "file_count_without_success_marker": len(observed),
        "bytes_without_success_marker": sum(
            int(row["size_bytes"]) for row in observed
        ),
        "tree_sha256_without_success_marker": tree_digest,
        "file_receipts_sha256": receipt_digest,
        "lance_tables": table_receipts,
        "checks_passed": True,
        "payload": dict(payload),
    }


def validate_release_metadata(
    root: Path,
    *,
    marker: Mapping[str, Any],
    authorization: Mapping[str, Any],
) -> dict[str, Any]:
    """Parse and cross-bind all publisher metadata before any Lance open."""

    release = Path(root)
    raw = {
        name: _read_release_file_nofollow(release, name) for name in METADATA_FILE_NAMES
    }
    documents = {
        name: _decode_mapping(payload, label=name, yaml_document=False)
        for name, payload in raw.items()
    }
    identities = {name: _raw_identity(payload) for name, payload in raw.items()}
    marker_payload = _mapping(marker.get("payload"), label="success marker payload")
    bound = _mapping(marker_payload.get("bound_files"), label="success marker bound_files")
    if set(bound) != set(METADATA_FILE_NAMES):
        raise ValueError("success marker metadata binding set mismatch")
    for name in METADATA_FILE_NAMES:
        _require_identity_binding(bound[name], expected=identities[name], label=f"marker/{name}")

    request = documents["request.json"]
    build = documents["build_report.json"]
    manifest = documents["manifest.json"]
    for name, document in documents.items():
        _require_protocol_scope(document, label=name)
        if document.get("active_splits") != list(ACTIVE_SPLITS):
            raise ValueError(f"{name} active_splits mismatch")
        if document.get("public_test_opened") is not False or document.get(
            "public_test_generated"
        ) is not False:
            raise ValueError(f"{name} does not keep Public closed")
    if request.get("pair_counts") != EXPECTED_PAIR_COUNTS:
        raise ValueError("request pair_counts mismatch")
    if request.get("workers") != 16 or request.get("jpeg_quality") != 95:
        raise ValueError("request frozen worker/JPEG contract mismatch")

    auth_identities = _mapping(authorization.get("identities"), label="authorization identities")
    freeze_binding = _mapping(request.get("freeze_receipt"), label="request freeze receipt")
    prior_binding = _mapping(
        request.get("prior_episode_exclusion_receipt"), label="request prior receipt"
    )
    _require_identity_binding(
        freeze_binding,
        expected=_mapping(auth_identities.get("freeze_receipt"), label="freeze identity"),
        label="request/freeze receipt",
    )
    _require_identity_binding(
        prior_binding,
        expected=_mapping(
            auth_identities.get("prior_exclusion_receipt"), label="prior identity"
        ),
        label="request/prior exclusion receipt",
    )
    if freeze_binding.get("status") != FREEZE_STATUS or freeze_binding.get(
        "checks_passed"
    ) is not True:
        raise ValueError("request freeze receipt status mismatch")
    if prior_binding.get("status") != FREEZE_STATUS or prior_binding.get(
        "checks_passed"
    ) is not True:
        raise ValueError("request prior receipt status mismatch")
    if build.get("request") != request:
        raise ValueError("build_report.request differs from request.json")
    if build.get("passed") is not True:
        raise ValueError("build_report does not declare a passing build")
    source_integrity = _mapping(
        build.get("source_h5_post_build_integrity"),
        label="build_report source H5 integrity",
    )
    if (
        source_integrity.get("passed") is not True
        or source_integrity.get("expected_sha256")
        != source_integrity.get("observed_sha256")
    ):
        raise ValueError("build_report source H5 post-build integrity failed")
    _require_sha256(
        source_integrity.get("expected_sha256"),
        field_name="source_h5_post_build_integrity.expected_sha256",
    )

    cross = _mapping(build.get("cross_split_audit"), label="build cross-split audit")
    cross_fields = (
        "query_pixel_hash_overlap",
        "source_episode_overlap",
        "exact_action_profile_id_overlap",
        "scene_template_content_hash_overlap",
        "pair_content_hash_overlap",
    )
    if cross.get("passed") is not True or any(
        _mapping(cross.get(name), label=f"cross_split_audit.{name}").get("count")
        != 0
        for name in cross_fields
    ):
        raise ValueError("build_report cross-split isolation failed")

    fresh = _mapping(build.get("fresh_simulator_replay"), label="build fresh replay")
    if (
        fresh.get("passed") is not True
        or fresh.get("pair_count") != sum(EXPECTED_PAIR_COUNTS.values())
        or fresh.get("mode_replay_count") != 2 * sum(EXPECTED_PAIR_COUNTS.values())
        or fresh.get("query_gap_used_as_replay_substitute") is not False
    ):
        raise ValueError("build_report aggregate fresh replay failed")
    causal = _mapping(build.get("causal_data_contract"), label="build causal contract")
    if causal.get("passed") is not True:
        raise ValueError("build_report causal data contract failed")
    splits = _mapping(build.get("splits"), label="build_report.splits")
    if set(splits) != set(ACTIVE_SPLITS):
        raise ValueError("build_report split set mismatch")
    for split in ACTIVE_SPLITS:
        split_report = _mapping(splits[split], label=f"build_report split {split}")
        split_fresh = _mapping(
            split_report.get("fresh_simulator_replay"),
            label=f"build_report {split} fresh replay",
        )
        split_prior = _mapping(
            split_report.get("prior_episode_and_content_exclusion"),
            label=f"build_report {split} prior exclusion",
        )
        accepted_overlap = _mapping(
            split_prior.get("accepted_overlap"),
            label=f"build_report {split} accepted prior overlap",
        )
        expected_anchor_count = EXPECTED_PAIR_COUNTS[split] // len(ACTION_ANCHORS)
        action_anchor_counts = _mapping(
            split_report.get("action_anchor_counts"),
            label=f"build_report {split} anchor counts",
        )
        if (
            split_report.get("passed") is not True
            or split_report.get("pair_count") != EXPECTED_PAIR_COUNTS[split]
            or split_report.get("episode_count") != 2 * EXPECTED_PAIR_COUNTS[split]
            or split_report.get("model_rows") != EXPECTED_MODEL_ROWS[split]
            or split_report.get("table_path") != TABLE_NAMES[split]
            or split_report.get("all_causal_checks_passed") is not True
            or split_report.get("unique_action_profile_count")
            != EXPECTED_PAIR_COUNTS[split]
            or split_report.get("unique_scene_template_content_hash_count")
            != EXPECTED_PAIR_COUNTS[split]
            or split_report.get("unique_pair_content_hash_count")
            != EXPECTED_PAIR_COUNTS[split]
            or action_anchor_counts
            != {anchor: expected_anchor_count for anchor in ACTION_ANCHORS}
            or split_report.get("action_anchor_expected_count_each")
            != expected_anchor_count
            or split_fresh.get("passed") is not True
            or split_fresh.get("pair_count") != EXPECTED_PAIR_COUNTS[split]
            or split_fresh.get("mode_replay_count")
            != 2 * EXPECTED_PAIR_COUNTS[split]
            or split_fresh.get("query_gap_used_as_replay_substitute") is not False
            or split_prior.get("passed") is not True
            or split_prior.get("candidate_catalog_source_episode_overlap_count") != 0
            or set(accepted_overlap) != {
                "source_episode_count",
                "action_profile_id_count",
                "scene_template_content_hash_count",
                "pair_content_hash_count",
                "query_pixel_hash_count",
            }
            or any(value != 0 for value in accepted_overlap.values())
        ):
            raise ValueError(f"build_report {split} identity mismatch")

    prior_document = _mapping(
        _mapping(authorization.get("documents"), label="authorization documents").get(
            "prior_exclusion_receipt"
        ),
        label="prior exclusion document",
    )
    expected_prior_manifest = {
        "sha256": auth_identities["prior_exclusion_receipt"]["sha256"],
        "excluded_source_episode_count": prior_document.get(
            "excluded_source_episode_count"
        ),
        "excluded_source_episodes_sha256": prior_document.get(
            "excluded_source_episodes_sha256"
        ),
    }
    if manifest.get("prior_episode_exclusion_receipt") != expected_prior_manifest:
        raise ValueError("manifest prior-exclusion binding mismatch")
    if manifest.get("build_passed") is not True:
        raise ValueError("manifest does not bind a passing build")
    manifest_files = _mapping(manifest.get("files"), label="manifest.files")
    observed_before_manifest = _regular_release_file_receipts(release)
    expected_files = {
        str(row["path"]): str(row["sha256"])
        for row in observed_before_manifest
        if row["path"] not in {SUCCESS_MARKER_NAME, "manifest.json"}
    }
    if dict(manifest_files) != expected_files:
        raise ValueError("manifest files do not exactly bind pre-manifest release files")
    if manifest.get("file_count_without_manifest") != len(expected_files):
        raise ValueError("manifest file_count_without_manifest mismatch")
    expected_bytes = sum(
        int(row["size_bytes"])
        for row in observed_before_manifest
        if row["path"] not in {SUCCESS_MARKER_NAME, "manifest.json"}
    )
    if manifest.get("bytes_without_manifest") != expected_bytes:
        raise ValueError("manifest bytes_without_manifest mismatch")
    return {
        "raw": raw,
        "documents": documents,
        "identities": identities,
        "checks_passed": True,
    }


def action_profile_content_sha256(action_blocks: np.ndarray) -> str:
    blocks = np.asarray(action_blocks, dtype=np.float32)
    if blocks.shape != ACTION_PROFILE_SHAPE:
        raise ValueError(
            f"action profile must have shape {ACTION_PROFILE_SHAPE}, got "
            f"{blocks.shape}"
        )
    if not np.isfinite(blocks).all():
        raise ValueError("action profile contains a non-finite value")
    if np.count_nonzero(blocks[3]):
        raise ValueError("terminal fourth action block must be exactly zero")
    return _array_sha256(blocks)


def _validate_actual_action_contract(
    blocks: np.ndarray,
    *,
    split: str,
    catalog_index: int,
    anchor: str,
    profile_id: str,
    context: str,
) -> None:
    expected_range = range(
        V4R1_FORMAL_CATALOG_INDEX_OFFSET,
        V4R1_FORMAL_CATALOG_INDEX_OFFSET + FORMAL_CATALOG_POOL_COUNTS[split],
    )
    if catalog_index not in expected_range:
        raise ValueError(f"{context}: catalog_index is outside the frozen v4r1 pool")
    profile = make_v4_action_profile(split=split, catalog_index=catalog_index)
    expected = np.asarray(frozen_v4_action_blocks(profile), dtype=np.float32)
    if profile.action_anchor_id != anchor:
        raise ValueError(f"{context}: anchor does not match frozen catalog index")
    if profile.action_profile_id != profile_id:
        raise ValueError(f"{context}: profile ID does not match frozen catalog index")
    if not np.array_equal(blocks, expected):
        raise ValueError(f"{context}: action blocks differ from frozen catalog profile")


def pair_content_sha256(scene_hash: str, profile_hash: str) -> str:
    scene = bytes.fromhex(
        _require_sha256(scene_hash, field_name="scene_template_content_hash")
    )
    profile = bytes.fromhex(
        _require_sha256(profile_hash, field_name="action_profile_id")
    )
    return hashlib.sha256(scene + profile).hexdigest()


def _decode_rgb_frame(payload: Any) -> np.ndarray:
    if isinstance(payload, memoryview):
        payload = payload.tobytes()
    if isinstance(payload, bytearray):
        payload = bytes(payload)
    if not isinstance(payload, bytes) or not payload:
        raise ValueError("pixels must contain non-empty encoded JPEG bytes")
    try:
        with Image.open(BytesIO(payload)) as image:
            if image.format != "JPEG":
                raise ValueError(
                    f"pixels must use the frozen JPEG container, got {image.format!r}"
                )
            rgb = image.convert("RGB")
            try:
                resized = rgb.resize(
                    RESIZE_SHAPE,
                    resample=Image.Resampling.BILINEAR,
                )
                try:
                    values = np.asarray(resized, dtype=np.float64).copy()
                finally:
                    resized.close()
            finally:
                rgb.close()
    except (OSError, UnidentifiedImageError) as error:
        raise ValueError("pixels could not be decoded as JPEG by Pillow") from error
    if values.shape != (16, 16, 3) or values.dtype != np.float64:
        raise RuntimeError("Pillow RGB decoder violated the frozen output contract")
    return values


def rgb_history_feature(x0: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    frames = [np.asarray(value) for value in (x0, x1, x2)]
    if any(value.shape != (16, 16, 3) for value in frames):
        raise ValueError("x0/x1/x2 must each have shape [16,16,3]")
    if any(value.dtype != np.float64 for value in frames):
        raise TypeError("x0/x1/x2 arithmetic must use float64")
    feature = 2.0 * frames[1] - frames[0] - frames[2]
    return np.ascontiguousarray(feature.reshape(-1, order="C"), dtype=np.float64)


def _constant_text(
    rows: Sequence[Mapping[str, Any]], field_name: str, *, context: str
) -> str:
    values = {str(row[field_name]) for row in rows}
    if len(values) != 1:
        raise ValueError(f"{context}: {field_name} changed across four rows")
    value = values.pop()
    if not value:
        raise ValueError(f"{context}: {field_name} must be non-empty")
    return value


def _action_block(row: Mapping[str, Any], *, context: str) -> np.ndarray:
    values = np.asarray(row["action_block"], dtype=np.float32)
    if values.size != 25:
        raise ValueError(f"{context}: action_block must contain 25 float32 values")
    values = values.reshape(5, 5)
    if not np.isfinite(values).all():
        raise ValueError(f"{context}: action_block contains a non-finite value")
    return values


def _constant_integer(
    rows: Sequence[Mapping[str, Any]], field_name: str, *, context: str
) -> int:
    values = {row[field_name] for row in rows}
    if len(values) != 1:
        raise ValueError(f"{context}: {field_name} is not constant")
    raw = next(iter(values))
    if isinstance(raw, (bool, np.bool_)) or not isinstance(raw, (int, np.integer)):
        raise TypeError(f"{context}: {field_name} must be an integer")
    return int(raw)


def _condition_from_rows(
    rows: Sequence[Mapping[str, Any]], *, expected_split: str
) -> _Condition:
    pair_id = _constant_text(rows, "pair_id", context="condition")
    hidden_mode = _constant_text(rows, "hidden_mode", context=pair_id)
    context = f"{pair_id}/{hidden_mode}"
    if hidden_mode not in HIDDEN_MODES:
        raise ValueError(f"{context}: unexpected hidden mode")
    if _constant_text(rows, "split", context=context) != expected_split:
        raise ValueError(f"{context}: split metadata mismatch")
    anchor = _constant_text(rows, "action_anchor_id", context=context)
    if anchor not in ACTION_ANCHORS:
        raise ValueError(f"{context}: unexpected action anchor {anchor!r}")
    profile_id = _require_sha256(
        _constant_text(rows, "action_profile_id", context=context),
        field_name="action_profile_id",
    )
    scene_hash = _require_sha256(
        _constant_text(rows, "scene_template_content_hash", context=context),
        field_name="scene_template_content_hash",
    )
    pair_hash = _require_sha256(
        _constant_text(rows, "pair_content_hash", context=context),
        field_name="pair_content_hash",
    )
    catalog_index = _constant_integer(rows, "catalog_index", context=context)
    source_episode = _constant_integer(rows, "source_episode", context=context)
    if source_episode < 0:
        raise ValueError(f"{context}: source_episode must be non-negative")
    local_index = catalog_index - V4R1_FORMAL_CATALOG_INDEX_OFFSET
    expected_pair_id = f"cube-carry-v4r1-{expected_split}-{local_index:06d}"
    if pair_id != expected_pair_id:
        raise ValueError(f"{context}: pair_id does not match frozen catalog index")

    by_step: dict[int, Mapping[str, Any]] = {}
    for row in rows:
        raw_step = row["model_step_idx"]
        if isinstance(raw_step, (bool, np.bool_)) or not isinstance(
            raw_step, (int, np.integer)
        ):
            raise TypeError(f"{context}: model_step_idx must be an integer")
        step = int(raw_step)
        if step in by_step:
            raise ValueError(f"{context}: duplicate model_step_idx={step}")
        by_step[step] = row
    if set(by_step) != set(MODEL_STEPS):
        raise ValueError(
            f"{context}: expected exactly model_step_idx 0..3, got "
            f"{sorted(by_step)}"
        )

    expected_hidden = np.float32(LABEL_ENCODING[hidden_mode])
    for step, row in by_step.items():
        hidden = np.asarray(row["hidden_grasp_enabled"], dtype=np.float32)
        if hidden.shape != (1,) or not np.array_equal(
            hidden, np.asarray([expected_hidden], dtype=np.float32)
        ):
            raise ValueError(
                f"{context}: hidden_grasp_enabled at step {step} does not "
                "match hidden_mode"
            )

    blocks = np.stack(
        [_action_block(by_step[step], context=context) for step in MODEL_STEPS]
    ).astype(np.float32, copy=False)
    calculated_profile = action_profile_content_sha256(blocks)
    if calculated_profile != profile_id:
        raise ValueError(
            f"{context}: action_profile_id does not match actual float32 actions"
        )
    _validate_actual_action_contract(
        blocks,
        split=expected_split,
        catalog_index=catalog_index,
        anchor=anchor,
        profile_id=profile_id,
        context=context,
    )
    calculated_pair = pair_content_sha256(scene_hash, profile_id)
    if calculated_pair != pair_hash:
        raise ValueError(
            f"{context}: pair_content_hash does not bind scene/profile content"
        )

    encoded = []
    decoded = []
    for step in DECODED_HISTORY_STEPS:
        if "pixels" not in by_step[step]:
            raise ValueError(f"{context}: missing pixels at model_step_idx={step}")
        payload = by_step[step]["pixels"]
        if isinstance(payload, memoryview):
            payload = payload.tobytes()
        if isinstance(payload, bytearray):
            payload = bytes(payload)
        if not isinstance(payload, bytes):
            raise ValueError(f"{context}: pixels at step {step} are not bytes")
        encoded.append(payload)
        decoded.append(_decode_rgb_frame(payload))
    x0, x1, x2 = decoded
    return _Condition(
        pair_id=pair_id,
        hidden_mode=hidden_mode,
        action_anchor_id=anchor,
        action_profile_id=profile_id,
        catalog_index=catalog_index,
        source_episode=source_episode,
        scene_template_content_hash=scene_hash,
        pair_content_hash=pair_hash,
        x0_jpeg=encoded[0],
        query_jpeg=encoded[2],
        x0_rgb=x0,
        query_rgb=x2,
        main_feature=rgb_history_feature(x0, x1, x2),
        action_blocks=np.ascontiguousarray(blocks),
    )


def prepare_split(
    rows: Sequence[Mapping[str, Any]], *, expected_split: str
) -> PreparedSplit:
    if expected_split not in ACTIVE_SPLITS:
        raise ValueError(f"inactive or Public split refused: {expected_split!r}")
    if not rows:
        raise ValueError(f"{expected_split}: empty table")
    required = set(METADATA_ACTION_COLUMNS)
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row_index, row in enumerate(rows):
        missing = sorted(required - set(row))
        if missing:
            raise ValueError(f"row {row_index}: missing required columns {missing}")
        split = str(row["split"])
        if split != expected_split:
            raise ValueError(
                f"row {row_index}: expected split={expected_split!r}, got {split!r}"
            )
        key = (str(row["pair_id"]), str(row["hidden_mode"]))
        groups.setdefault(key, []).append(row)

    conditions = {
        key: _condition_from_rows(group, expected_split=expected_split)
        for key, group in groups.items()
    }
    pair_ids = sorted({key[0] for key in conditions})
    ordered: list[_Condition] = []
    anchor_pair_counts = {anchor: 0 for anchor in ACTION_ANCHORS}
    profile_ids: set[str] = set()
    source_episodes: set[int] = set()
    query_pixel_hashes: set[str] = set()
    scene_hashes: set[str] = set()
    pair_hashes: set[str] = set()
    for pair_id in pair_ids:
        modes = {
            mode for candidate_pair, mode in conditions if candidate_pair == pair_id
        }
        if modes != set(HIDDEN_MODES):
            raise ValueError(f"{pair_id}: incomplete hidden-mode pair: {sorted(modes)}")
        cannot = conditions[(pair_id, "cannot_hold")]
        can = conditions[(pair_id, "can_hold")]
        metadata = (
            "action_anchor_id",
            "action_profile_id",
            "catalog_index",
            "source_episode",
            "scene_template_content_hash",
            "pair_content_hash",
        )
        for field_name in metadata:
            if getattr(cannot, field_name) != getattr(can, field_name):
                raise ValueError(f"{pair_id}: paired {field_name} values differ")
        if cannot.x0_jpeg != can.x0_jpeg or not np.array_equal(
            cannot.x0_rgb, can.x0_rgb
        ):
            raise ValueError(f"{pair_id}: paired x0 pixels are not bitwise identical")
        if cannot.query_jpeg != can.query_jpeg or not np.array_equal(
            cannot.query_rgb, can.query_rgb
        ):
            raise ValueError(
                f"{pair_id}: paired query/x2 pixels are not bitwise identical"
            )
        if not np.array_equal(cannot.action_blocks, can.action_blocks):
            raise ValueError(f"{pair_id}: paired actions are not bitwise identical")

        anchor_pair_counts[cannot.action_anchor_id] += 1
        query_hash = hashlib.sha256(cannot.query_jpeg).hexdigest()
        if cannot.source_episode in source_episodes:
            raise ValueError(f"{pair_id}: duplicate source_episode inside split")
        if query_hash in query_pixel_hashes:
            raise ValueError(f"{pair_id}: duplicate query JPEG identity inside split")
        source_episodes.add(cannot.source_episode)
        query_pixel_hashes.add(query_hash)
        for value, target, field_name in (
            (cannot.action_profile_id, profile_ids, "action_profile_id"),
            (
                cannot.scene_template_content_hash,
                scene_hashes,
                "scene_template_content_hash",
            ),
            (cannot.pair_content_hash, pair_hashes, "pair_content_hash"),
        ):
            if value in target:
                raise ValueError(f"{pair_id}: duplicate {field_name} inside split")
            target.add(value)
        ordered.extend((cannot, can))

    if not pair_ids or len(pair_ids) % len(ACTION_ANCHORS):
        raise ValueError("pair count must be positive and divisible by four anchors")
    expected_anchor_count = len(pair_ids) // len(ACTION_ANCHORS)
    if set(anchor_pair_counts.values()) != {expected_anchor_count}:
        raise ValueError(
            f"{expected_split}: action anchors are not exactly balanced: "
            f"{anchor_pair_counts}"
        )

    return PreparedSplit(
        split=expected_split,
        main_features=np.ascontiguousarray(
            np.stack([condition.main_feature for condition in ordered]),
            dtype=np.float64,
        ),
        x0_features=np.ascontiguousarray(
            np.stack(
                [condition.x0_rgb.reshape(-1, order="C") for condition in ordered]
            ),
            dtype=np.float64,
        ),
        query_features=np.ascontiguousarray(
            np.stack(
                [
                    condition.query_rgb.reshape(-1, order="C")
                    for condition in ordered
                ]
            ),
            dtype=np.float64,
        ),
        action_features=np.ascontiguousarray(
            np.stack(
                [
                    condition.action_blocks.astype(np.float64).reshape(
                        -1, order="C"
                    )
                    for condition in ordered
                ]
            ),
            dtype=np.float64,
        ),
        labels=np.asarray(
            [LABEL_ENCODING[condition.hidden_mode] for condition in ordered],
            dtype=np.int64,
        ),
        pair_ids=np.asarray([condition.pair_id for condition in ordered]),
        hidden_modes=np.asarray(
            [condition.hidden_mode for condition in ordered]
        ),
        action_anchors=np.asarray(
            [condition.action_anchor_id for condition in ordered]
        ),
        action_profile_ids=frozenset(profile_ids),
        source_episodes=frozenset(source_episodes),
        query_pixel_hashes=frozenset(query_pixel_hashes),
        scene_template_content_hashes=frozenset(scene_hashes),
        pair_content_hashes=frozenset(pair_hashes),
        pair_count=len(pair_ids),
        condition_count=len(ordered),
        row_count=len(rows),
        anchor_pair_counts=dict(anchor_pair_counts),
    )


def cross_split_content_audit(
    train: PreparedSplit, development: PreparedSplit
) -> dict[str, Any]:
    if train.split != "train" or development.split != "loader_validation":
        raise ValueError("cross-split audit requires Training then Development")

    def overlap(left: frozenset[Any], right: frozenset[Any]) -> list[Any]:
        return sorted(left & right)

    profiles = overlap(train.action_profile_ids, development.action_profile_ids)
    source_episodes = overlap(train.source_episodes, development.source_episodes)
    query_pixels = overlap(train.query_pixel_hashes, development.query_pixel_hashes)
    scenes = overlap(
        train.scene_template_content_hashes,
        development.scene_template_content_hashes,
    )
    pairs = overlap(train.pair_content_hashes, development.pair_content_hashes)
    train_anchors = sorted(set(train.action_anchors.tolist()))
    development_anchors = sorted(set(development.action_anchors.tolist()))
    expected_anchors = sorted(ACTION_ANCHORS)
    checks = {
        "exact_action_profile_id_overlap_zero": not profiles,
        "source_episode_overlap_zero": not source_episodes,
        "query_pixel_hash_overlap_zero": not query_pixels,
        "scene_template_content_hash_overlap_zero": not scenes,
        "pair_content_hash_overlap_zero": not pairs,
        "four_anchor_families_present_in_both_splits": (
            train_anchors == expected_anchors
            and development_anchors == expected_anchors
        ),
    }
    return {
        "evidence_source": {
            "action_profile_id": "recomputed_from_table_float32_action_blocks",
            "source_episode": "frozen_table_column",
            "query_pixel_hash": "sha256_of_raw_x2_JPEG_bytes",
            "scene_template_content_hash": "frozen_table_column",
            "pair_content_hash": "recomputed_from_table_scene_and_profile_digests",
            "manifest_read": False,
        },
        "exact_action_profile_id_overlap": {
            "count": len(profiles),
            "values": profiles,
        },
        "source_episode_overlap": {
            "count": len(source_episodes),
            "values": source_episodes,
        },
        "query_pixel_hash_overlap": {
            "count": len(query_pixels),
            "values": query_pixels,
        },
        "scene_template_content_hash_overlap": {
            "count": len(scenes),
            "values": scenes,
        },
        "pair_content_hash_overlap": {
            "count": len(pairs),
            "values": pairs,
        },
        "anchor_families": {
            "expected": expected_anchors,
            "train": train_anchors,
            "loader_validation": development_anchors,
            "shared_families_are_expected_not_content_leakage": True,
        },
        "pair_id_is_content_isolation_evidence": False,
        "checks": checks,
        "passed": bool(all(checks.values())),
    }


def prior_exclusion_audit(
    train: PreparedSplit,
    development: PreparedSplit,
    *,
    prior_sets: Mapping[str, frozenset[Any]],
) -> dict[str, Any]:
    actual = {
        "source_episodes": train.source_episodes | development.source_episodes,
        "action_profile_ids": train.action_profile_ids | development.action_profile_ids,
        "scene_template_content_hashes": (
            train.scene_template_content_hashes
            | development.scene_template_content_hashes
        ),
        "pair_content_hashes": train.pair_content_hashes | development.pair_content_hashes,
        "query_pixel_hashes": train.query_pixel_hashes | development.query_pixel_hashes,
    }
    if set(prior_sets) != set(actual):
        raise ValueError("prior exclusion set universe mismatch")
    overlaps = {
        name: sorted(values & prior_sets[name]) for name, values in actual.items()
    }
    checks = {f"{name}_overlap_zero": not values for name, values in overlaps.items()}
    if not all(checks.values()):
        counts = {name: len(values) for name, values in overlaps.items()}
        raise ValueError(f"published tables overlap frozen prior exclusions: {counts}")
    return {
        "actual_counts": {name: len(values) for name, values in actual.items()},
        "overlaps": {
            name: {"count": len(values), "values": values}
            for name, values in overlaps.items()
        },
        "checks": checks,
        "passed": True,
    }
def _fit_ridge(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    development_features: np.ndarray,
    *,
    feature_name: str,
) -> _FitResult:
    train_values = np.asarray(train_features, dtype=np.float64)
    development_values = np.asarray(development_features, dtype=np.float64)
    labels = np.asarray(train_labels, dtype=np.int64)
    if train_values.ndim != 2 or development_values.ndim != 2:
        raise ValueError(f"{feature_name}: features must be rank-2")
    if train_values.shape[1] != development_values.shape[1]:
        raise ValueError(f"{feature_name}: split feature dimensions differ")
    if train_values.shape[0] != labels.size or set(labels.tolist()) != {0, 1}:
        raise ValueError(f"{feature_name}: Training labels must contain classes 0/1")
    if not np.isfinite(train_values).all() or not np.isfinite(
        development_values
    ).all():
        raise ValueError(f"{feature_name}: features contain non-finite values")

    scaler = StandardScaler()
    transformed_train = scaler.fit_transform(train_values)
    transformed_development = scaler.transform(development_values)
    classifier = RidgeClassifier(alpha=1.0)
    classifier.fit(transformed_train, labels)
    predictions = np.asarray(
        classifier.predict(transformed_development), dtype=np.int64
    )
    receipt = {
        "feature_name": feature_name,
        "feature_dimension": int(train_values.shape[1]),
        "standard_scaler": {
            "fit_split": "train",
            "development_used_for_fit": False,
            "with_mean": bool(scaler.with_mean),
            "with_std": bool(scaler.with_std),
            "n_samples_seen": int(scaler.n_samples_seen_),
            "mean_float64_sha256": _array_sha256(
                np.asarray(scaler.mean_, dtype=np.float64)
            ),
            "scale_float64_sha256": _array_sha256(
                np.asarray(scaler.scale_, dtype=np.float64)
            ),
        },
        "ridge_classifier": {
            "alpha": 1.0,
            "decision_rule": "sklearn.linear_model.RidgeClassifier.predict",
            "classes": [int(value) for value in classifier.classes_],
            "coefficient_sha256": _array_sha256(
                np.asarray(classifier.coef_, dtype=np.float64)
            ),
            "intercept_sha256": _array_sha256(
                np.asarray(classifier.intercept_, dtype=np.float64)
            ),
        },
    }
    return _FitResult(
        predictions=predictions,
        transformed_train=np.ascontiguousarray(transformed_train),
        transformed_development=np.ascontiguousarray(transformed_development),
        receipt=receipt,
    )


def _accuracy(labels: np.ndarray, predictions: np.ndarray) -> float:
    actual = np.asarray(labels, dtype=np.int64)
    predicted = np.asarray(predictions, dtype=np.int64)
    if actual.shape != predicted.shape or actual.ndim != 1 or not actual.size:
        raise ValueError("accuracy requires equal non-empty one-dimensional arrays")
    return float(np.mean(actual == predicted))


def stratified_pair_cluster_bootstrap(
    labels: np.ndarray,
    predictions: np.ndarray,
    pair_ids: np.ndarray,
    action_anchors: np.ndarray,
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    actual = np.asarray(labels, dtype=np.int64)
    predicted = np.asarray(predictions, dtype=np.int64)
    pairs = np.asarray(pair_ids).astype(str)
    anchors = np.asarray(action_anchors).astype(str)
    if not (
        actual.shape == predicted.shape == pairs.shape == anchors.shape
        and actual.ndim == 1
    ):
        raise ValueError("bootstrap inputs must be aligned one-dimensional arrays")
    if isinstance(resamples, bool) or int(resamples) <= 0:
        raise ValueError("bootstrap resamples must be positive")

    correct = actual == predicted
    clusters_by_anchor: dict[str, np.ndarray] = {}
    stratum_counts: dict[str, int] = {}
    for anchor in ACTION_ANCHORS:
        anchor_pairs = sorted(set(pairs[anchors == anchor].tolist()))
        if not anchor_pairs:
            raise ValueError(f"bootstrap anchor stratum {anchor!r} is empty")
        cluster_accuracy: list[float] = []
        for pair_id in anchor_pairs:
            indices = np.flatnonzero(pairs == pair_id)
            if indices.size != 2:
                raise ValueError(f"bootstrap pair {pair_id!r} must contain two modes")
            if set(anchors[indices].tolist()) != {anchor}:
                raise ValueError(f"bootstrap pair {pair_id!r} crosses anchors")
            cluster_accuracy.append(float(np.mean(correct[indices])))
        clusters_by_anchor[anchor] = np.asarray(
            cluster_accuracy, dtype=np.float64
        )
        stratum_counts[anchor] = len(anchor_pairs)

    rng = np.random.default_rng(seed)
    bootstrap_sums = np.zeros(int(resamples), dtype=np.float64)
    sampled_pairs = 0
    for anchor in ACTION_ANCHORS:
        values = clusters_by_anchor[anchor]
        draws = rng.integers(
            0,
            values.size,
            size=(int(resamples), values.size),
        )
        bootstrap_sums += values[draws].sum(axis=1)
        sampled_pairs += int(values.size)
    samples = bootstrap_sums / sampled_pairs
    lower = float(
        np.quantile(samples, BOOTSTRAP_LOWER_QUANTILE, method="linear")
    )
    upper = float(np.quantile(samples, 0.975, method="linear"))
    return {
        "unit": "pair_cluster",
        "stratification": "action_anchor_id",
        "resamples": int(resamples),
        "seed": int(seed),
        "lower_quantile": BOOTSTRAP_LOWER_QUANTILE,
        "quantile_method": "numpy_linear",
        "stratum_pair_counts": stratum_counts,
        "overall_accuracy": _accuracy(actual, predicted),
        "lower_bound_2_5_percent": lower,
        "upper_bound_97_5_percent": upper,
        "bootstrap_mean": float(np.mean(samples)),
        "bootstrap_minimum": float(np.min(samples)),
        "bootstrap_maximum": float(np.max(samples)),
        "gate_minimum": BOOTSTRAP_LOWER_BOUND_MINIMUM,
        "passed": bool(lower >= BOOTSTRAP_LOWER_BOUND_MINIMUM),
    }


def _permuted_label_control(
    transformed_train: np.ndarray,
    train_labels: np.ndarray,
    transformed_development: np.ndarray,
    development_labels: np.ndarray,
    *,
    repetitions: int = PERMUTATION_REPETITIONS,
    seed: int = PERMUTATION_SEED,
) -> dict[str, Any]:
    if isinstance(repetitions, bool) or int(repetitions) <= 0:
        raise ValueError("permutation repetitions must be positive")
    rng = np.random.default_rng(seed)
    scores: list[float] = []
    for _ in range(int(repetitions)):
        permuted = rng.permutation(np.asarray(train_labels, dtype=np.int64))
        classifier = RidgeClassifier(alpha=1.0)
        classifier.fit(transformed_train, permuted)
        predictions = classifier.predict(transformed_development)
        scores.append(_accuracy(development_labels, predictions))
    mean = float(np.mean(scores))
    return {
        "permutation_target": "Training condition labels only",
        "development_labels_remain_true": True,
        "feature_scaler_reused_from_primary_train_fit": True,
        "repetitions": int(repetitions),
        "seed": int(seed),
        "scores": scores,
        "mean_accuracy": mean,
        "maximum_mean_accuracy": PERMUTATION_MEAN_ACCURACY_MAXIMUM,
        "passed": bool(mean <= PERMUTATION_MEAN_ACCURACY_MAXIMUM),
    }


def _group_metrics(
    development: PreparedSplit, predictions: np.ndarray
) -> dict[str, Any]:
    labels = development.labels
    overall = _accuracy(labels, predictions)
    per_mode = {
        mode: _accuracy(
            labels[development.hidden_modes == mode],
            predictions[development.hidden_modes == mode],
        )
        for mode in HIDDEN_MODES
    }
    per_anchor = {
        anchor: _accuracy(
            labels[development.action_anchors == anchor],
            predictions[development.action_anchors == anchor],
        )
        for anchor in ACTION_ANCHORS
    }
    worst_mode = min(HIDDEN_MODES, key=lambda mode: per_mode[mode])
    worst_anchor = min(ACTION_ANCHORS, key=lambda anchor: per_anchor[anchor])
    return {
        "overall_accuracy": overall,
        "per_mode_accuracy": per_mode,
        "worst_mode": {
            "hidden_mode": worst_mode,
            "accuracy": per_mode[worst_mode],
        },
        "per_anchor_family_accuracy": per_anchor,
        "worst_anchor_family": {
            "action_anchor_id": worst_anchor,
            "accuracy": per_anchor[worst_anchor],
        },
    }


def _shortcut_control(
    name: str,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    development_features: np.ndarray,
    development_labels: np.ndarray,
) -> dict[str, Any]:
    fit = _fit_ridge(
        train_features,
        train_labels,
        development_features,
        feature_name=name,
    )
    accuracy = _accuracy(development_labels, fit.predictions)
    return {
        "accuracy": accuracy,
        "maximum_accuracy": SHORTCUT_ACCURACY_MAXIMUM,
        "passed": bool(accuracy <= SHORTCUT_ACCURACY_MAXIMUM),
        "fit_receipt": fit.receipt,
    }


def _package_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "Pillow": PIL.__version__,
        "Pillow_jpeglib": str(getattr(Image.core, "jpeglib_version", "unknown")),
        "scikit-learn": sklearn.__version__,
        "scipy": scipy.__version__,
        "lance": lance.__version__,
        "pyarrow": pa.__version__,
    }


def evaluate_prepared_splits(
    train: PreparedSplit,
    development: PreparedSplit,
    *,
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    permutation_repetitions: int = PERMUTATION_REPETITIONS,
    permutation_seed: int = PERMUTATION_SEED,
) -> dict[str, Any]:
    content_audit = cross_split_content_audit(train, development)
    primary = _fit_ridge(
        train.main_features,
        train.labels,
        development.main_features,
        feature_name="flatten(2*x1-x0-x2)_C_order",
    )
    metrics = _group_metrics(development, primary.predictions)
    bootstrap = stratified_pair_cluster_bootstrap(
        development.labels,
        primary.predictions,
        development.pair_ids,
        development.action_anchors,
        resamples=bootstrap_resamples,
        seed=bootstrap_seed,
    )
    controls = {
        "label_permutation": _permuted_label_control(
            primary.transformed_train,
            train.labels,
            primary.transformed_development,
            development.labels,
            repetitions=permutation_repetitions,
            seed=permutation_seed,
        ),
        "x0_only": _shortcut_control(
            "x0_only",
            train.x0_features,
            train.labels,
            development.x0_features,
            development.labels,
        ),
        "query_x2_only": _shortcut_control(
            "query_x2_only",
            train.query_features,
            train.labels,
            development.query_features,
            development.labels,
        ),
        "action_only": _shortcut_control(
            "action_only",
            train.action_features,
            train.labels,
            development.action_features,
            development.labels,
        ),
    }
    gates = {
        "cross_split_content_isolation_passed": bool(content_audit["passed"]),
        "paired_x0_query_actions_identical": True,
        "overall_accuracy_at_least_0_75": bool(
            metrics["overall_accuracy"] >= OVERALL_ACCURACY_MINIMUM
        ),
        "worst_mode_accuracy_at_least_0_70": bool(
            metrics["worst_mode"]["accuracy"]
            >= WORST_MODE_ACCURACY_MINIMUM
        ),
        "worst_anchor_accuracy_at_least_0_70": bool(
            metrics["worst_anchor_family"]["accuracy"]
            >= WORST_ANCHOR_ACCURACY_MINIMUM
        ),
        "bootstrap_2_5_percent_lower_bound_at_least_0_70": bool(
            bootstrap["passed"]
        ),
        "permuted_label_mean_accuracy_at_most_0_60": bool(
            controls["label_permutation"]["passed"]
        ),
        "x0_only_accuracy_at_most_0_51": bool(controls["x0_only"]["passed"]),
        "query_x2_only_accuracy_at_most_0_51": bool(
            controls["query_x2_only"]["passed"]
        ),
        "action_only_accuracy_at_most_0_51": bool(
            controls["action_only"]["passed"]
        ),
    }
    passed = bool(all(gates.values()))
    return {
        "schema_version": 1,
        "probe_id": PROBE_ID,
        "protocol": PROTOCOL,
        "recovery_authorization_id": RECOVERY_AUTHORIZATION_ID,
        "status": "passed" if passed else "failed",
        "role": "frozen_rgb_history_data_probe_not_reference_model_evaluation",
        "active_splits": list(ACTIVE_SPLITS),
        "public_test": {
            "canonical_split": "validation",
            "access_status": "closed_not_read_not_scored",
            "opened": False,
            "read": False,
            "hashed": False,
            "scored": False,
        },
        "decoder_and_feature_contract": {
            "source_column": "pixels",
            "decoded_steps": [0, 1, 2],
            "x3_decoded_or_used": False,
            "x3_pixel_bytes_not_read_from_lance": True,
            "container": "JPEG_only",
            "decoder": "PIL.Image.open(BytesIO(payload)).convert('RGB')",
            "channel_order": "RGB",
            "resize_shape": [16, 16],
            "resize_interpolation": "PIL.Image.Resampling.BILINEAR",
            "arithmetic_dtype": "float64",
            "fixed_main_feature": "flatten(2*x1-x0-x2)_C_order",
            "main_feature_dimension": 16 * 16 * 3,
            "main_feature_columns": list(MAIN_FEATURE_COLUMNS),
            "negative_control_only_columns": list(
                NEGATIVE_CONTROL_ONLY_COLUMNS
            ),
            "audit_only_columns": list(AUDIT_ONLY_COLUMNS),
            "privileged_columns_excluded_from_main_feature": list(
                PRIVILEGED_COLUMNS_EXCLUDED_FROM_MAIN_FEATURE
            ),
            "action_is_negative_control_only": True,
            "ids_labels_metadata_and_row_order_used_as_main_feature": False,
        },
        "label_contract": {
            "encoding": dict(LABEL_ENCODING),
            "label_source_used_only_after_feature_construction": "hidden_mode",
        },
        "data_integrity": {
            "grouping_key": ["pair_id", "hidden_mode"],
            "required_rows_per_condition": 4,
            "required_model_step_indices": list(MODEL_STEPS),
            "paired_x0_jpeg_and_decoded_rgb_bitwise_equal": True,
            "paired_query_x2_jpeg_and_decoded_rgb_bitwise_equal": True,
            "paired_float32_action_blocks_bitwise_equal": True,
            "action_profile_id_recomputed_from_actual_float32_blocks": True,
            "pair_content_hash_recomputed_from_scene_and_profile_digests": True,
            "splits": {
                split.split: {
                    "row_count": split.row_count,
                    "pair_count": split.pair_count,
                    "condition_count": split.condition_count,
                    "anchor_pair_counts": dict(split.anchor_pair_counts),
                }
                for split in (train, development)
            },
        },
        "cross_split_content_isolation": content_audit,
        "fit_contract": {
            "standard_scaler_fit_split_only": "train",
            "ridge_classifier_alpha": 1.0,
            "development_evaluated_once_without_tuning": True,
            "reference_model_or_checkpoint_loaded": False,
            "primary_fit_receipt": primary.receipt,
        },
        "primary_probe": {
            "metrics": metrics,
            "thresholds": {
                "overall_accuracy_minimum": OVERALL_ACCURACY_MINIMUM,
                "worst_mode_accuracy_minimum": WORST_MODE_ACCURACY_MINIMUM,
                "worst_anchor_family_accuracy_minimum": (
                    WORST_ANCHOR_ACCURACY_MINIMUM
                ),
            },
        },
        "pair_cluster_anchor_stratified_bootstrap": bootstrap,
        "negative_controls": controls,
        "gates": gates,
        "package_versions": _package_versions(),
        "passed": passed,
    }


def evaluate_fixture_rows(
    train_rows: Sequence[Mapping[str, Any]],
    development_rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    permutation_repetitions: int = PERMUTATION_REPETITIONS,
    permutation_seed: int = PERMUTATION_SEED,
) -> dict[str, Any]:
    """Pure-row entry point used by unit fixtures; it never opens a table."""

    train = prepare_split(train_rows, expected_split="train")
    development = prepare_split(
        development_rows,
        expected_split="loader_validation",
    )
    return evaluate_prepared_splits(
        train,
        development,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
        permutation_repetitions=permutation_repetitions,
        permutation_seed=permutation_seed,
    )


def _forbidden_closed_component(path: Path) -> str | None:
    forbidden = {
        "validation",
        "validation.lance",
        "public",
        "public_test",
        "public-test",
        "public test",
        "publictest",
    }
    for part in path.parts:
        if part.lower() in forbidden:
            return part
    return None


def resolve_allowed_tables(artifact_root: Path) -> tuple[Path, dict[str, Path]]:
    root_input = artifact_root.expanduser()
    forbidden = _forbidden_closed_component(root_input)
    if forbidden is not None:
        raise ValueError(
            f"Cube v4 probe explicitly refuses validation/Public path component "
            f"{forbidden!r}"
        )
    if root_input.name.lower().endswith(".lance"):
        raise ValueError("--artifact-root must be a root, not a Lance table path")
    root_descriptor = _open_directory_nofollow(
        Path(os.path.abspath(root_input)), label="artifact root"
    )
    os.close(root_descriptor)
    root = root_input.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)

    # Inspect the complete namespace before any regular-file content is read.
    # A nested Public-shaped directory is just as closed as a root child.
    for directory, names, files in os.walk(root, followlinks=False):
        current = Path(directory)
        for name in sorted([*names, *files]):
            relative = (current / name).relative_to(root)
            forbidden = _forbidden_closed_component(relative)
            if forbidden is not None:
                raise ValueError(
                    "Cube v4 probe refuses validation/Public component before "
                    f"content read: {relative.as_posix()}"
                )
            metadata = os.lstat(current / name)
            if stat.S_ISLNK(metadata.st_mode):
                raise ValueError(
                    f"artifact release contains a symlink: {relative.as_posix()}"
                )
            if not (stat.S_ISDIR(metadata.st_mode) or stat.S_ISREG(metadata.st_mode)):
                raise ValueError(
                    f"artifact release contains a special node: {relative.as_posix()}"
                )

    lance_children = sorted(
        value for value in root.iterdir() if value.name.lower().endswith(".lance")
    )
    allowed_names = set(TABLE_NAMES.values())
    extras = [value.name for value in lance_children if value.name not in allowed_names]
    if extras:
        raise ValueError(
            "Cube v4 probe refuses validation/Public or any non-authorized Lance "
            f"table under the artifact root: {extras}"
        )
    tables: dict[str, Path] = {}
    for split, name in TABLE_NAMES.items():
        path = root / name
        if path.is_symlink():
            raise ValueError(f"authorized Lance table cannot be a symlink: {name}")
        if not path.is_dir():
            raise FileNotFoundError(path)
        resolved = path.resolve()
        if resolved.parent != root or resolved.name != name:
            raise ValueError(f"authorized Lance table escapes artifact root: {name}")
        tables[split] = resolved
    return root, tables


def _validate_exact_release_root_namespace(root: Path) -> None:
    expected = {
        SUCCESS_MARKER_NAME,
        *METADATA_FILE_NAMES,
        *TABLE_NAMES.values(),
    }
    with os.scandir(root) as iterator:
        observed = {entry.name for entry in iterator}
    if observed != expected:
        raise ValueError(
            "artifact root must contain exactly three metadata files, two "
            f"authorized Lance tables, and the success marker; observed={sorted(observed)}"
        )


def _regular_fd_tree_receipts(descriptor: int) -> list[dict[str, Any]]:
    anchor = Path(f"/proc/self/fd/{descriptor}")
    files: list[Path] = []

    def visit(directory: Path) -> None:
        with os.scandir(directory) as iterator:
            entries = sorted(iterator, key=lambda entry: entry.name)
        for entry in entries:
            child = directory / entry.name
            relative = child.relative_to(anchor)
            if _forbidden_closed_component(relative) is not None:
                raise ValueError(
                    f"Lance table contains a validation/Public component: {relative}"
                )
            metadata = entry.stat(follow_symlinks=False)
            if stat.S_ISDIR(metadata.st_mode):
                visit(child)
            elif stat.S_ISREG(metadata.st_mode):
                files.append(child)
            else:
                raise ValueError(f"Lance table contains a symlink or special node: {relative}")

    visit(anchor)
    receipts = []
    for child in sorted(files, key=lambda value: value.relative_to(anchor).as_posix()):
        relative = child.relative_to(anchor).as_posix()
        data = _read_bytes_at(descriptor, Path(relative), label=f"Lance file {relative}")
        receipts.append(
            {
                "path": relative,
                "size_bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    return receipts


def _read_bytes_at(root_descriptor: int, relative: Path, *, label: str) -> bytes:
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError(f"{label} has an unsafe relative path")
    current = os.dup(root_descriptor)
    try:
        for component in relative.parts[:-1]:
            child = os.open(
                component,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=current,
            )
            os.close(current)
            current = child
        leaf = os.open(
            relative.parts[-1], os.O_RDONLY | os.O_NOFOLLOW, dir_fd=current
        )
        with os.fdopen(leaf, "rb") as stream:
            return stream.read()
    finally:
        os.close(current)


def _require_anchored_table_identity(
    descriptor: int, expected: Mapping[str, Any], *, split: str
) -> None:
    receipts = _regular_fd_tree_receipts(descriptor)
    checks = {
        "file_count": expected.get("file_count") == len(receipts),
        "size_bytes": expected.get("size_bytes")
        == sum(int(row["size_bytes"]) for row in receipts),
        "tree_sha256": expected.get("tree_sha256")
        == _tree_sha256_from_receipts(receipts),
        "file_receipts_sha256": expected.get("file_receipts_sha256")
        == _canonical_json_sha256(receipts),
    }
    if not receipts or not all(checks.values()):
        raise RuntimeError(f"{split} fd-anchored Lance identity mismatch")


def _projection_key(
    row: Mapping[str, Any], *, source: str
) -> tuple[str, str, str, int]:
    required = {"pair_id", "hidden_mode", "split", "model_step_idx"}
    missing = sorted(required - set(row))
    if missing:
        raise ValueError(f"{source}: projection row is missing join keys {missing}")
    raw_step = row["model_step_idx"]
    if isinstance(raw_step, (bool, np.bool_)) or not isinstance(
        raw_step, (int, np.integer)
    ):
        raise TypeError(f"{source}: model_step_idx join key must be an integer")
    return (
        str(row["pair_id"]),
        str(row["hidden_mode"]),
        str(row["split"]),
        int(raw_step),
    )


def _merge_lance_projections(
    metadata_rows: Sequence[Mapping[str, Any]],
    pixel_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Join x0/x1/x2 pixels onto four-row metadata without reading x3 bytes."""

    metadata_keys: set[tuple[str, str, str, int]] = set()
    ordered_metadata: list[tuple[tuple[str, str, str, int], Mapping[str, Any]]] = []
    for row in metadata_rows:
        if "pixels" in row:
            raise ValueError("metadata/action projection unexpectedly contains pixels")
        missing = sorted(set(METADATA_ACTION_COLUMNS) - set(row))
        if missing:
            raise ValueError(f"metadata/action projection is missing {missing}")
        key = _projection_key(row, source="metadata/action")
        if key in metadata_keys:
            raise ValueError(f"duplicate metadata/action projection key: {key}")
        metadata_keys.add(key)
        ordered_metadata.append((key, row))

    pixels_by_key: dict[tuple[str, str, str, int], bytes] = {}
    for row in pixel_rows:
        missing = sorted(set(PIXEL_JOIN_COLUMNS) - set(row))
        if missing:
            raise ValueError(f"pixel projection is missing {missing}")
        key = _projection_key(row, source="pixels")
        if key[3] not in DECODED_HISTORY_STEPS:
            raise ValueError(
                "pixel projection returned x3 or an out-of-contract step despite "
                f"the frozen filter: {key}"
            )
        if key in pixels_by_key:
            raise ValueError(f"duplicate pixel projection key: {key}")
        payload = row["pixels"]
        if isinstance(payload, memoryview):
            payload = payload.tobytes()
        if isinstance(payload, bytearray):
            payload = bytes(payload)
        if not isinstance(payload, bytes):
            raise ValueError(f"pixel projection payload is not bytes: {key}")
        pixels_by_key[key] = payload

    expected_pixel_keys = {
        key for key in metadata_keys if key[3] in DECODED_HISTORY_STEPS
    }
    if set(pixels_by_key) != expected_pixel_keys:
        missing = sorted(expected_pixel_keys - set(pixels_by_key))
        extra = sorted(set(pixels_by_key) - expected_pixel_keys)
        raise ValueError(
            "filtered pixel projection does not exactly cover x0/x1/x2 keys: "
            f"missing={missing[:3]}, extra={extra[:3]}"
        )

    merged: list[dict[str, Any]] = []
    for key, row in ordered_metadata:
        combined = dict(row)
        if key[3] in DECODED_HISTORY_STEPS:
            combined["pixels"] = pixels_by_key[key]
        merged.append(combined)
    return merged


def _read_lance_rows(
    path: Path, *, expected_split: str, fd_anchored: bool = False
) -> list[dict[str, Any]]:
    forbidden = _forbidden_closed_component(path)
    if forbidden is not None:
        raise ValueError(
            "refusing validation/Public path component before Lance open: "
            f"{forbidden!r}"
        )
    if expected_split not in ACTIVE_SPLITS:
        raise ValueError(f"inactive or Public split refused: {expected_split!r}")
    expected_name = TABLE_NAMES[expected_split]
    if not fd_anchored and path.name != expected_name:
        raise ValueError(
            f"refusing non-authorized table {path.name!r}; expected {expected_name!r}"
        )
    if fd_anchored and (
        path.parent != Path("/proc/self/fd") or not path.name.isdigit()
    ):
        raise ValueError("fd-anchored Lance path must be under /proc/self/fd")
    dataset = lance.dataset(str(path))
    if dataset.schema != FROZEN_ARROW_SCHEMA:
        raise ValueError(f"{expected_name}: Arrow schema differs from frozen v4")
    row_count = int(dataset.count_rows())
    if row_count != EXPECTED_MODEL_ROWS[expected_split]:
        raise ValueError(
            f"{expected_name}: expected {EXPECTED_MODEL_ROWS[expected_split]} rows, "
            f"got {row_count}"
        )
    metadata_table = dataset.to_table(columns=list(METADATA_ACTION_COLUMNS))
    pixel_table = dataset.to_table(
        columns=list(PIXEL_JOIN_COLUMNS),
        filter=PIXEL_FILTER,
    )
    return _merge_lance_projections(
        metadata_table.to_pylist(),
        pixel_table.to_pylist(),
    )


@contextmanager
def _anchored_lance_table(
    path: Path,
    *,
    expected_split: str,
    expected_identity: Mapping[str, Any],
) -> Iterator[Path]:
    """Keep a no-follow directory fd alive while Lance reads through procfs."""

    expected_name = TABLE_NAMES[expected_split]
    if Path(path).name != expected_name:
        raise ValueError(f"unexpected Lance table path for {expected_split}")
    if not hasattr(os, "O_DIRECTORY") or not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError("safe Lance fd anchoring requires O_DIRECTORY and O_NOFOLLOW")
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISDIR(metadata.st_mode):
            raise ValueError(f"{expected_name} is not a real directory")
        anchored = Path(f"/proc/self/fd/{descriptor}")
        if not Path("/proc/self/fd").is_dir() or not anchored.is_dir():
            raise RuntimeError("safe Lance fd anchoring requires readable /proc/self/fd")
        resolved = anchored.resolve()
        expected = Path(path).resolve()
        if resolved != expected:
            raise RuntimeError(f"{expected_name} fd anchor resolves to a different table")
        _require_anchored_table_identity(
            descriptor, expected_identity, split=expected_split
        )
        yield anchored
        post = os.fstat(descriptor)
        if (post.st_dev, post.st_ino) != (metadata.st_dev, metadata.st_ino):
            raise RuntimeError(f"{expected_name} fd identity changed during read")
        _require_anchored_table_identity(
            descriptor, expected_identity, split=expected_split
        )
    finally:
        os.close(descriptor)


def _reverify_authorization_inputs(authorization: Mapping[str, Any]) -> None:
    paths = _mapping(authorization.get("paths"), label="authorization paths")
    raw = _mapping(authorization.get("raw"), label="authorization raw bytes")
    if set(paths) != set(raw):
        raise RuntimeError("authorization path/byte sets changed")
    for name in sorted(paths):
        observed = _read_bytes_nofollow(Path(paths[name]), label=f"postflight {name}")
        if observed != raw[name]:
            raise RuntimeError(f"authorization input changed during probe: {name}")
    documents = _mapping(
        authorization.get("documents"), label="authorization documents"
    )
    _validate_frozen_implementation_identity(
        _mapping(documents.get("freeze_receipt"), label="freeze receipt")
    )


def _marker_summary(marker: Mapping[str, Any]) -> dict[str, Any]:
    return {name: value for name, value in marker.items() if name != "payload"}


def run_probe(
    artifact_root: Path, *, authorization: Mapping[str, Any]
) -> dict[str, Any]:
    root, tables = resolve_allowed_tables(artifact_root)
    _validate_exact_release_root_namespace(root)
    success_marker_preflight = validate_success_marker(root)
    metadata_preflight = validate_release_metadata(
        root,
        marker=success_marker_preflight,
        authorization=authorization,
    )
    rows: dict[str, list[dict[str, Any]]] = {}
    for split, path in tables.items():
        with _anchored_lance_table(
            path,
            expected_split=split,
            expected_identity=_mapping(
                _mapping(
                    success_marker_preflight.get("lance_tables"),
                    label="marker Lance tables",
                ).get(split),
                label=f"marker {split} Lance identity",
            ),
        ) as anchored:
            rows[split] = _read_lance_rows(
                anchored, expected_split=split, fd_anchored=True
            )
    for split in ACTIVE_SPLITS:
        if len(rows[split]) != EXPECTED_MODEL_ROWS[split]:
            raise ValueError(
                f"{split}: frozen v4r1 row count mismatch: "
                f"expected={EXPECTED_MODEL_ROWS[split]}, actual={len(rows[split])}"
            )
    success_marker_postflight = validate_success_marker(root)
    if success_marker_postflight != success_marker_preflight:
        raise RuntimeError("v4r1 release identity changed during probe input reads")
    metadata_postflight = validate_release_metadata(
        root,
        marker=success_marker_postflight,
        authorization=authorization,
    )
    if metadata_postflight["raw"] != metadata_preflight["raw"]:
        raise RuntimeError("release metadata bytes changed during probe input reads")
    _reverify_authorization_inputs(authorization)

    train = prepare_split(rows["train"], expected_split="train")
    development = prepare_split(
        rows["loader_validation"], expected_split="loader_validation"
    )
    prior_audit = prior_exclusion_audit(
        train,
        development,
        prior_sets=_mapping(
            authorization.get("prior_sets"), label="authorization prior sets"
        ),
    )
    report = evaluate_prepared_splits(
        train,
        development,
        bootstrap_resamples=BOOTSTRAP_RESAMPLES,
        bootstrap_seed=BOOTSTRAP_SEED,
        permutation_repetitions=PERMUTATION_REPETITIONS,
        permutation_seed=PERMUTATION_SEED,
    )
    report["data_integrity"]["frozen_prior_exclusion_audit"] = prior_audit
    report["gates"]["frozen_prior_exclusion_overlap_zero"] = True
    report["passed"] = bool(all(report["gates"].values()))
    report["status"] = "passed" if report["passed"] else "failed"
    marker_preflight_summary = _marker_summary(success_marker_preflight)
    marker_postflight_summary = _marker_summary(success_marker_postflight)
    report["inputs"] = {
        "artifact_root_path_recorded": False,
        "trusted_input_contract": dict(TRUSTED_INPUT_CONTRACT),
        "authorization_chain_verified_before_artifact_root": True,
        "authorization_inputs_reverified_after_lance_reads": True,
        "authorization_identities": {
            name: dict(value)
            for name, value in _mapping(
                authorization.get("identities"), label="authorization identities"
            ).items()
        },
        "completed_publication_verified_before_lance_open": True,
        "success_marker": marker_preflight_summary,
        "success_marker_preflight": marker_preflight_summary,
        "success_marker_postflight": marker_postflight_summary,
        "release_identity_unchanged_during_reads": True,
        "only_authorized_lance_tables_opened": list(TABLE_NAMES.values()),
        "metadata_files_parsed_before_lance_open": list(METADATA_FILE_NAMES),
        "manifest_or_build_report_parsed": True,
        "manifest_and_build_report_bytes_hashed": True,
        "metadata_identities": {
            name: dict(value)
            for name, value in metadata_preflight["identities"].items()
        },
        "metadata_bytes_unchanged_during_reads": True,
        "validation_or_public_table_read": False,
        "tables": {
            split: {
                "relative_path": path.name,
                "table_directory_hashed": True,
                "directory_fd_held_during_lance_read": True,
                "lance_open_path": "/proc/self/fd/<held-directory-fd>",
                "identity_reverified_through_held_fd_after_read": True,
                "projections": [
                    {
                        "columns": list(METADATA_ACTION_COLUMNS),
                        "filter": None,
                        "row_scope": "all_four_model_steps",
                    },
                    {
                        "columns": list(PIXEL_JOIN_COLUMNS),
                        "filter": PIXEL_FILTER,
                        "row_scope": "x0_x1_x2_only",
                    },
                ],
                "x3_pixel_bytes_read": False,
                "rows_read": len(rows[split]),
            }
            for split, path in tables.items()
        },
    }
    return report


def _reject_forbidden_cli(values: Sequence[str]) -> None:
    forbidden_prefixes = (
        "--validation",
        "--public",
        "--test",
    )
    for value in values:
        option = value.split("=", 1)[0].lower()
        if option.startswith(forbidden_prefixes):
            raise ValueError(
                "Cube v4 RGB-history probe explicitly refuses validation/Public "
                "Test options"
            )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    values = list(sys.argv[1:] if argv is None else argv)
    _reject_forbidden_cli(values)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--prereg", type=Path, required=True)
    parser.add_argument("--freeze-receipt", type=Path, required=True)
    parser.add_argument("--prior-exclusion-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(values)


def _open_directory_nofollow(path: Path, *, label: str) -> int:
    value = Path(os.path.abspath(path))
    if not value.is_absolute():
        raise ValueError(f"{label} must be absolute")
    descriptor = os.open(value.anchor, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for component in value.parts[1:]:
            child = os.open(
                component,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _write_json_exclusive(
    path: Path, payload: Mapping[str, Any], *, parent_descriptor: int | None = None
) -> None:
    if path.suffix.lower() != ".json":
        raise ValueError("probe output must use a .json filename")
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    expected_sha256 = hashlib.sha256(encoded).hexdigest()
    owned = parent_descriptor is None
    descriptor = (
        _open_directory_nofollow(path.parent, label="probe output parent")
        if owned
        else os.dup(parent_descriptor)
    )
    output_fd: int | None = None
    created = False
    try:
        _assert_directory_fd_path(
            descriptor, path.parent, label="probe output parent"
        )
        flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
        output_fd = os.open(path.name, flags, 0o644, dir_fd=descriptor)
        created = True
        offset = 0
        while offset < len(encoded):
            written = os.write(output_fd, encoded[offset:])
            if written <= 0:
                raise OSError("probe output write made no progress")
            offset += written
        os.fsync(output_fd)
        metadata = os.fstat(output_fd)
        os.lseek(output_fd, 0, os.SEEK_SET)
        observed = bytearray()
        while True:
            chunk = os.read(output_fd, 1024 * 1024)
            if not chunk:
                break
            observed.extend(chunk)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_size != len(encoded)
            or hashlib.sha256(observed).hexdigest() != expected_sha256
            or bytes(observed) != encoded
        ):
            raise RuntimeError("exclusive probe output identity verification failed")
        os.close(output_fd)
        output_fd = None
        _assert_directory_fd_path(
            descriptor, path.parent, label="probe output parent"
        )
        os.fsync(descriptor)
    except BaseException:
        if output_fd is not None:
            os.close(output_fd)
        if created:
            try:
                os.unlink(path.name, dir_fd=descriptor)
                os.fsync(descriptor)
            except FileNotFoundError:
                pass
        raise
    finally:
        os.close(descriptor)


def _assert_directory_fd_path(descriptor: int, path: Path, *, label: str) -> None:
    held = os.fstat(descriptor)
    try:
        current = os.stat(path, follow_symlinks=False)
    except FileNotFoundError as error:
        raise RuntimeError(f"{label} disappeared") from error
    if not stat.S_ISDIR(current.st_mode) or (held.st_dev, held.st_ino) != (
        current.st_dev,
        current.st_ino,
    ):
        raise RuntimeError(f"{label} identity changed")


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output = Path(os.path.abspath(args.output.expanduser()))
    artifact_root = Path(os.path.abspath(args.artifact_root.expanduser()))
    for label, path in (
        ("artifact root", artifact_root),
        ("preregistration", args.prereg),
        ("freeze receipt", args.freeze_receipt),
        ("prior exclusion receipt", args.prior_exclusion_receipt),
        ("probe output", output),
    ):
        forbidden = _forbidden_closed_component(Path(path))
        if forbidden is not None:
            raise ValueError(
                f"{label} has forbidden validation/Public component {forbidden!r}"
            )
    if artifact_root != CANONICAL_ARTIFACT_ROOT:
        raise ValueError("--artifact-root must be the canonical frozen publication")
    if output != CANONICAL_OUTPUT_PATH:
        raise ValueError("--output must be the canonical one-shot probe receipt")
    if output.suffix.lower() != ".json":
        raise ValueError("probe output must use a .json filename")
    try:
        os.lstat(output)
    except FileNotFoundError:
        pass
    else:
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    if _is_relative_to(output, artifact_root):
        raise ValueError("probe output must remain outside the immutable artifact root")
    output_parent_fd = _open_directory_nofollow(
        output.parent, label="probe output parent"
    )
    try:
        authorization = validate_authorization_chain(
            prereg_path=args.prereg,
            freeze_receipt_path=args.freeze_receipt,
            prior_exclusion_path=args.prior_exclusion_receipt,
        )
        report = run_probe(artifact_root, authorization=authorization)
        _write_json_exclusive(
            output, report, parent_descriptor=output_parent_fd
        )
    finally:
        os.close(output_parent_fd)
    print(
        json.dumps(
            {
                "output": str(output),
                "status": report["status"],
                "passed": report["passed"],
                "public_test_read": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
