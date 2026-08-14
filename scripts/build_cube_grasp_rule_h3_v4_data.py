#!/usr/bin/env python3
"""Build the Development-only Cube History-3 v4r1 recovery dataset.

The scientific protocol remains v4.  Recovery batch v4r1 changes only the
formal content namespace and storage publication path after the original v4
attempt failed during an NFS Lance commit.  It intentionally has no Public
Test split, excludes every identity consumed by that failed attempt, stages
Lance commits locally, and publishes a verified release with ``_SUCCESS``
written last.
"""

from __future__ import annotations

import argparse
import atexit
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass
import errno
import hashlib
from io import BytesIO
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import stat
import sys
import tempfile
import time
from typing import Any, Iterator

import h5py
import lance
import numpy as np
import pyarrow as pa
from PIL import Image
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

from contextworld.benchmarks.causal_data_contract import (  # noqa: E402
    audit_causal_data_contract,
)
from contextworld.evaluation.cube_grasp_rule_h3_v4 import (  # noqa: E402
    CAPABILITY_NAME,
    GRASP_MODES,
    QUERY_STATE_TOLERANCE,
    V4_ACTION_ANCHORS,
    V4R1_FORMAL_CATALOG_INDEX_OFFSET,
    V4_PROFILE_SPLIT_SEEDS,
    CubeGraspRuleCandidate,
    CubeGraspRuleV4Candidate,
    CubeGraspRuleV4Simulator,
    action_blocks as v4_action_blocks,
    make_v4_candidate,
)
from contextworld.paths import (  # noqa: E402
    artifact_path,
    portable_contextworld_path,
)


PROTOCOL = "cube_gripper_carry_rule_history3_development_v4"
RECOVERY_AUTHORIZATION_ID = "cube_gripper_carry_h3_development_v4r1"
EVIDENCE_SCOPE = "every accepted pair in Training and Development"
PROFILE_SPLIT_POLICY = "shared_families_disjoint_profiles"
ACTIVE_SPLITS = ("train", "loader_validation")
DEFAULT_PAIR_COUNTS = {"train": 2048, "loader_validation": 256}
FROZEN_JPEG_QUALITY = 95
FROZEN_WORKERS = 16
DEFAULT_OUTPUT_LOGICAL = Path(
    "artifacts/synthesis/cube_gripper_carry_rule_h3_development_v4r1"
)
DEFAULT_OUTPUT = artifact_path(
    "synthesis/cube_gripper_carry_rule_h3_development_v4r1"
)
DEFAULT_STAGING_ROOT = Path("/tmp")
SUCCESS_MARKER_NAME = "_SUCCESS.json"
REQUIRED_RELEASE_METADATA = (
    "request.json",
    "build_report.json",
    "manifest.json",
)
FORBIDDEN_PUBLIC_COMPONENTS = frozenset(
    {
        "validation",
        "validation.lance",
        "public",
        "public_test",
        "public-test",
        "public test",
        "publictest",
    }
)
DEFAULT_PREREG = ROOT / (
    "configs/benchmark/cube_gripper_carry_h3_development_recovery_prereg_v4r1.yaml"
)
DEFAULT_FREEZE_RECEIPT_LOGICAL = Path(
    "artifacts/evaluation/history3/cube_gripper_carry_h3_development_v4r1/"
    "development_recovery_freeze_receipt_v1.json"
)
DEFAULT_FREEZE_RECEIPT = artifact_path(
    "evaluation/history3/cube_gripper_carry_h3_development_v4r1/"
    "development_recovery_freeze_receipt_v1.json"
)
DEFAULT_PRIOR_EXCLUSION_RECEIPT_LOGICAL = Path(
    "artifacts/evaluation/history3/cube_gripper_carry_h3_development_v4r1/"
    "prior_episode_exclusions_final_v1.json"
)
DEFAULT_PRIOR_EXCLUSION_RECEIPT = artifact_path(
    "evaluation/history3/cube_gripper_carry_h3_development_v4r1/"
    "prior_episode_exclusions_final_v1.json"
)
V4_PHYSICS_PATH = ROOT / "contextworld/evaluation/cube_grasp_rule_h3_v4.py"
V4_BUILDER_PATH = Path(__file__).resolve()
REQUIRED_FREEZE_IDENTITY_KEYS = (
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
EXPECTED_FREEZE_IDENTITY_PATHS = {
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
REQUIRED_FREEZE_AUTHORIZATION_INPUT_KEYS = (
    "original_v4_preregistration",
    "original_v4_freeze_receipt",
    "old_final_prior_receipt",
    "infrastructure_failure_decision",
    "failed_formal_attempt_receipt",
    "query_reconstruction_receipt",
    "v4r1_action_support_audit",
    "source_h5",
)
ACTION_SUPPORT_AUDIT_SHA256 = (
    "d35c06992c99680cceb6b0c29a5018772ace6a9af013928ee9894f29c6397001"
)
ACTION_SUPPORT_AUDIT_SIZE_BYTES = 23_847
SOURCE_SYMBOL = "upstream_cube_single_expert_h5"
CATALOG_SEEDS = {
    "train": 2026081201,
    "loader_validation": 2026081202,
}
CANDIDATE_ASSIGNMENT_SEED = 2026081200
CANDIDATE_POOL_MULTIPLIER = 2
FORMAL_CATALOG_LOCAL_INDEX_POLICY = "zero_based_contiguous_within_each_split"
ACTION_PROFILE_SHAPE = (4, 5, 5)
MINIMUM_EFFECT_GAP_M = 0.008
FORMAL_CATALOG_INDEX_OFFSET = V4R1_FORMAL_CATALOG_INDEX_OFFSET
FREEZE_STATUS = "frozen_before_v4r1_recovery_build"
PRIOR_EXCLUSION_STATUS = "frozen_before_v4r1_recovery_build"
REQUIRED_PRIOR_EPISODE_COVERAGE = (
    "v3_formal",
    "v3_smokes",
    "v3_pilots",
    "v4_preformal_smokes_and_pilots",
    "v4_failed_formal_attempts",
)
PRIOR_CONTENT_EXCLUSION_FIELDS = (
    "action_profile_ids",
    "scene_template_content_hashes",
    "pair_content_hashes",
    "query_pixel_hashes",
)
RECOVERY_PRIOR_RECEIPT_ID = (
    "cube_gripper_carry_h3_v4r1_prior_exclusions_final_v1"
)
EXPECTED_RECOVERY_EXCLUDED_SOURCE_EPISODE_COUNT = 4_369
EXPECTED_RECOVERY_EXCLUDED_SOURCE_EPISODES_SHA256 = (
    "a2167602269492d464e7f07b2a4c1c8ba3e8c46fc1df4791ba69cd0e6027a021"
)
EXPECTED_RECOVERY_CONTENT = {
    "action_profile_ids": {
        "count": 4_370,
        "sha256": (
            "a65e5534e0db40617126e5c916c650b273e7554247e145bdd3b5bf28a36c3b16"
        ),
    },
    "scene_template_content_hashes": {
        "count": 4_378,
        "sha256": (
            "a5437c01f480e3ad6a22b90f2d31f8cda9bec2a029889fc0ffc8794ba7d89dbc"
        ),
    },
    "pair_content_hashes": {
        "count": 4_378,
        "sha256": (
            "58404c522605e0129d4c3a59680e4a8143a9eb2d651a05d34c4dc5ebd37826f7"
        ),
    },
    "query_pixel_hashes": {
        "count": 4_378,
        "sha256": (
            "7a54a31c301b780af492153122eaaa095dfc9af384d95bb5a4875c2795f05b4e"
        ),
    },
}
EXPECTED_OLD_PRIOR_IDENTITIES = {
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
EXPECTED_FAILED_ATTEMPT_IDENTITIES = {
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
EXPECTED_RECOVERY_UNION_IDENTITIES = {
    "source_episodes": {
        "count": EXPECTED_RECOVERY_EXCLUDED_SOURCE_EPISODE_COUNT,
        "sha256": EXPECTED_RECOVERY_EXCLUDED_SOURCE_EPISODES_SHA256,
    },
    **EXPECTED_RECOVERY_CONTENT,
}
FROZEN_PROBE_RECIPE = {
    "input": "decoded_x0_x1_x2_rgb_only",
    "resize_shape": [16, 16],
    "resize_interpolation": "Pillow_Resampling_BILINEAR",
    "arithmetic_dtype": "float64",
    "fixed_feature": "flatten(2*x1-x0-x2)_C_order",
    "standard_scaler_fit_split_only": "train",
    "estimator": "StandardScaler_then_RidgeClassifier_alpha_1",
    "label_encoding": {"cannot_hold": 0, "can_hold": 1},
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
FROZEN_PROBE_TRUSTED_INPUT_CONTRACT = {
    "required_explicit_cli_options": [
        "--artifact-root",
        "--prereg",
        "--freeze-receipt",
        "--prior-exclusion-receipt",
        "--output",
    ],
    "authorization_documents_read_before_artifact_root": True,
    "preregistration_status": "preregistered_before_v4r1_recovery_build",
    "freeze_receipt_status": FREEZE_STATUS,
    "prior_exclusion_receipt_id": RECOVERY_PRIOR_RECEIPT_ID,
    "prior_exclusion_receipt_status": PRIOR_EXCLUSION_STATUS,
    "freeze_receipt_must_bind_exact_preregistration": True,
    "prior_exclusion_receipt_must_bind_exact_preregistration_and_freeze": True,
    "metadata_files_parsed_before_lance_open": list(REQUIRED_RELEASE_METADATA),
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
# This is intentionally the same canonicalization namespace as v3.  A
# version-salted scene hash would make byte-identical v3/v4 scene inputs look
# disjoint and would defeat the cross-version exclusion audit.
SCENE_CONTENT_HASH_NORMALIZATION = (
    "contextworld-cube-v3-scene-template-content-v1"
)
ELIGIBLE_ROW_SELECTION_RULE = {
    "one_candidate_per_source_episode": True,
    "contact_minimum": 0.8,
    "gripper_opening_inclusive_range": [0.45, 0.68],
    "cube_height_m_inclusive_range": [0.017, 0.024],
    "cube_effector_distance_m_maximum": 0.008,
    "source_step_inclusive_range": [5, 160],
    "episode_choice": (
        "lexicographic minimum of cube-effector distance, source row, "
        "source step"
    ),
}
PRIVILEGED_COLUMNS = (
    "episode_idx",
    "model_step_idx",
    "physical_state",
    "hidden_grasp_enabled",
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


if {
    split: int(V4_PROFILE_SPLIT_SEEDS[split]) for split in ACTIVE_SPLITS
} != CATALOG_SEEDS:
    raise RuntimeError(
        "v4 physics profile seeds must exactly match the frozen v4 catalog seeds"
    )
if (
    FORMAL_CATALOG_INDEX_OFFSET <= 0
    or FORMAL_CATALOG_INDEX_OFFSET % len(V4_ACTION_ANCHORS) != 0
):
    raise RuntimeError(
        "v4r1 formal catalog offset must be positive and anchor-family aligned"
    )


def _formal_catalog_index(local_index: int) -> int:
    """Map a per-split formal local index into v4's frozen namespace."""

    if isinstance(local_index, bool) or not isinstance(
        local_index, (int, np.integer)
    ):
        raise TypeError("formal local_index must be an integer")
    value = int(local_index)
    if value < 0:
        raise ValueError("formal local_index must be non-negative")
    return FORMAL_CATALOG_INDEX_OFFSET + value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_tree_files(path: Path) -> list[Path]:
    """Return sorted regular files while rejecting aliases/special nodes."""

    root = Path(path)
    try:
        root_metadata = os.lstat(root)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"directory tree is missing: {root}") from error
    if not stat.S_ISDIR(root_metadata.st_mode):
        raise ValueError(f"directory tree root must be a real directory: {root}")

    files: list[Path] = []

    def visit(directory: Path) -> None:
        with os.scandir(directory) as iterator:
            entries = sorted(iterator, key=lambda value: value.name)
        for entry in entries:
            child = directory / entry.name
            if entry.name.lower() in FORBIDDEN_PUBLIC_COMPONENTS:
                raise ValueError(
                    "release tree contains a forbidden Public component: "
                    f"{child.relative_to(root).as_posix()}"
                )
            metadata = entry.stat(follow_symlinks=False)
            if stat.S_ISDIR(metadata.st_mode):
                visit(child)
            elif stat.S_ISREG(metadata.st_mode):
                files.append(child)
            else:
                raise ValueError(
                    "release tree contains a symlink or special node: "
                    f"{child}"
                )

    visit(root)
    return sorted(files, key=lambda value: value.relative_to(root).as_posix())


def regular_file_receipts(path: Path) -> list[dict[str, Any]]:
    """Return the exact sorted path/size/SHA identity of a regular tree."""

    root = Path(path)
    return [
        {
            "path": child.relative_to(root).as_posix(),
            "size_bytes": child.stat().st_size,
            "sha256": file_sha256(child),
        }
        for child in _regular_tree_files(root)
    ]


def _canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def directory_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for child in _regular_tree_files(path):
        digest.update(child.relative_to(path).as_posix().encode())
        digest.update(b"\0")
        digest.update(file_sha256(child).encode())
        digest.update(b"\0")
    return digest.hexdigest()


def _verified_current_file_identity(
    *,
    receipt_entry: Mapping[str, Any],
    current_path: Path,
    label: str,
) -> dict[str, Any]:
    if not current_path.is_file():
        raise FileNotFoundError(f"{label}: current file is missing: {current_path}")
    expected = str(receipt_entry.get("sha256", ""))
    actual = file_sha256(current_path)
    if expected != actual:
        raise RuntimeError(
            f"{label}: current SHA256 differs from freeze receipt: "
            f"{actual} != {expected}"
        )
    expected_size = receipt_entry.get("size_bytes")
    if expected_size is not None and int(expected_size) != current_path.stat().st_size:
        raise RuntimeError(f"{label}: current size differs from freeze receipt")
    return {
        "declared_path": str(receipt_entry.get("path", "")),
        "current_path": portable_contextworld_path(current_path),
        "sha256": actual,
        "size_bytes": current_path.stat().st_size,
    }


def _read_regular_file_nofollow(path: Path, *, label: str) -> bytes:
    """Read one real regular file without following its final component."""

    value = Path(path)
    try:
        metadata = os.lstat(value)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"{label} is missing: {value}") from error
    if not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError(f"{label} must be a real regular file: {value}")
    descriptor = os.open(value, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(descriptor, "rb") as stream:
        return stream.read()


def _identity_from_bytes(path: Path, raw: bytes) -> dict[str, Any]:
    return {
        "path": portable_contextworld_path(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
    }


def _identity_core(value: Any, *, label: str) -> tuple[str, int]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{label} identity must be an object")
    digest = str(value.get("sha256", ""))
    _sha256_digest_bytes(digest, field_name=f"{label}.sha256")
    size = value.get("size_bytes")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise RuntimeError(f"{label}.size_bytes must be a positive integer")
    return digest, int(size)


def _same_identity(left: Any, right: Any, *, label: str) -> None:
    if _identity_core(left, label=f"{label} left") != _identity_core(
        right, label=f"{label} right"
    ):
        raise RuntimeError(f"{label} identity mismatch")


def _contains_unresolved_placeholder(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(_contains_unresolved_placeholder(child) for child in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_unresolved_placeholder(child) for child in value)
    return isinstance(value, str) and any(
        token in value.upper()
        for token in ("PENDING_SHA256", "PLACEHOLDER", "REPLACE_ME", "TBD")
    )


def _require_public_closed(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{label} Public Test closure must be an object")
    if value.get("access_status") != "closed_not_read_not_scored" or any(
        value.get(name) is not False
        for name in ("opened", "read", "hashed", "scored")
    ):
        raise RuntimeError(f"{label} does not keep Public Test fully closed")
    return dict(value)


def _validate_freeze_contract_documents(
    *,
    receipt_path: Path,
    prereg_path: Path,
    builder_path: Path = V4_BUILDER_PATH,
    physics_path: Path = V4_PHYSICS_PATH,
) -> dict[str, Any]:
    """Validate the complete preregistration -> recovery-freeze contract.

    This deliberately goes beyond checking a few implementation hashes: the
    sole formal builder must see the exact authorization-input set, the v2
    action-support evidence, and every frozen science/data/storage/probe
    section that the freezer copied from the canonical preregistration.
    """

    prereg_raw = _read_regular_file_nofollow(
        prereg_path, label="v4r1 preregistration"
    )
    receipt_raw = _read_regular_file_nofollow(
        receipt_path, label="v4r1 freeze receipt"
    )
    try:
        prereg = yaml.safe_load(prereg_raw.decode("utf-8"))
        receipt = json.loads(receipt_raw.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError, json.JSONDecodeError) as error:
        raise RuntimeError("v4r1 preregistration/freeze is not valid UTF-8 data") from error
    if not isinstance(prereg, Mapping) or not isinstance(receipt, Mapping):
        raise RuntimeError("v4r1 preregistration and freeze must be objects")
    if _contains_unresolved_placeholder(prereg):
        raise RuntimeError("v4r1 preregistration contains an unresolved placeholder")

    expected_header = {
        "schema_version": 1,
        "protocol_id": PROTOCOL,
        "scientific_protocol_id": PROTOCOL,
        "recovery_authorization_id": RECOVERY_AUTHORIZATION_ID,
    }
    for name, expected in expected_header.items():
        if prereg.get(name) != expected or receipt.get(name) != expected:
            raise RuntimeError(f"v4r1 prereg/freeze {name} mismatch")
    if prereg.get("status") != "preregistered_before_v4r1_recovery_build":
        raise RuntimeError("v4r1 preregistration status mismatch")
    if prereg.get("phase") != "development_only":
        raise RuntimeError("v4r1 preregistration is not Development-only")
    if receipt.get("status") != FREEZE_STATUS or receipt.get("checks_passed") is not True:
        raise RuntimeError("v4r1 freeze status/checks do not authorize the build")
    if receipt.get("authorized_splits") != list(ACTIVE_SPLITS):
        raise RuntimeError("v4r1 freeze split authorization mismatch")
    if receipt.get("recovery_build_attempts_authorized") != 1 or receipt.get(
        "rgb_history_probe_attempts_authorized"
    ) != 1:
        raise RuntimeError("v4r1 freeze attempt budget mismatch")
    if prereg.get("reference_model_training_or_scoring_authorized") is not False:
        raise RuntimeError("v4r1 preregistration authorizes reference-model work")
    if receipt.get("reference_model_training_or_scoring_authorized") is not False:
        raise RuntimeError("v4r1 freeze authorizes reference-model work")
    if receipt.get("reference_model_optimizer_steps_authorized") != 0:
        raise RuntimeError("v4r1 freeze authorizes optimizer steps")
    _require_public_closed(prereg.get("public_test"), label="v4r1 preregistration")
    _require_public_closed(receipt.get("public_test"), label="v4r1 freeze")

    prereg_identity = _identity_from_bytes(prereg_path, prereg_raw)
    _same_identity(
        receipt.get("preregistration"),
        prereg_identity,
        label="freeze/preregistration",
    )

    copied_sections = (
        "scientific_protocol_contract",
        "recovery_contract",
        "storage_publication_contract",
        "data_contract",
        "recovery_prior_exclusion_contract",
        "recovery_capacity_check",
        "action_support_authorization",
        "rgb_history_probe",
    )
    for name in copied_sections:
        if not isinstance(prereg.get(name), Mapping) or receipt.get(name) != prereg.get(name):
            raise RuntimeError(f"v4r1 freeze does not exactly bind {name}")

    data = prereg["data_contract"]
    expected_data = {
        "logical_output_root": DEFAULT_OUTPUT_LOGICAL.as_posix(),
        "authorized_splits": list(ACTIVE_SPLITS),
        "pair_counts": dict(DEFAULT_PAIR_COUNTS),
        "workers": FROZEN_WORKERS,
        "episodes_per_pair": 2,
        "rows_per_pair": 8,
        "pairs_per_anchor": {"train": 512, "loader_validation": 64},
        "formal_catalog_index_offset": FORMAL_CATALOG_INDEX_OFFSET,
        "catalog_index_offset_modulo_anchor_count": 0,
        "source_episode_overlap_between_splits_required": 0,
        "action_profile_overlap_between_splits_required": 0,
        "scene_template_overlap_between_splits_required": 0,
        "pair_content_overlap_between_splits_required": 0,
        "query_pixel_overlap_between_splits_required": 0,
    }
    if {name: data.get(name) for name in expected_data} != expected_data:
        raise RuntimeError("v4r1 frozen data contract mismatch")
    science = prereg["scientific_protocol_contract"]
    expected_science = {
        "unchanged_from_original_v4": True,
        "history_tokens": 3,
        "context_transitions": 2,
        "prediction_horizon_action_blocks": 1,
        "raw_steps_per_action_block": 5,
        "can_hold_vertical_force_coupling_n": 0.40,
        "hidden_modes": list(GRASP_MODES),
        "action_temporal_pattern": ["p", "negative_p", "p", "terminal_zero"],
        "action_anchor_ids": list(_anchor_ids()),
        "sum_p_target": 0.0,
        "final_p_target": 0.0,
        "displacement_moment_weights": [4.0, 3.0, 2.0, 1.0, 0.0],
        "displacement_moment_target": 1.0,
        "constraint_absolute_tolerance": 1.0e-6,
        "jpeg_quality": FROZEN_JPEG_QUALITY,
        "query_state_and_pixels_equal_across_modes": True,
        "paired_actions_bitwise_equal": True,
        "no_state_installation_after_x0": True,
    }
    if {name: science.get(name) for name in expected_science} != expected_science:
        raise RuntimeError("v4r1 frozen scientific contract mismatch")
    recovery = prereg["recovery_contract"]
    expected_recovery = {
        "failure_class": "infrastructure_lance_atomic_rename_eperm",
        "original_v4_formal_attempt_consumed": True,
        "retry_under_original_v4_preregistration_authorized": False,
        "original_failed_tree_immutable": True,
        "scientific_protocol_changed": False,
        "recovery_build_attempts_authorized": 1,
        "builder_or_lance_smoke_attempts_authorized": 0,
        "rgb_history_probe_attempts_authorized": 1,
        "formal_catalog_index_offset": FORMAL_CATALOG_INDEX_OFFSET,
        "formal_catalog_offset_four_aligned": True,
        "failed_batch_identities_must_be_excluded": True,
    }
    if {name: recovery.get(name) for name in expected_recovery} != expected_recovery:
        raise RuntimeError("v4r1 recovery contract mismatch")
    storage = prereg["storage_publication_contract"]
    expected_storage = {
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
    if {name: storage.get(name) for name in expected_storage} != expected_storage:
        raise RuntimeError("v4r1 frozen publication contract mismatch")

    prior_contract = prereg["recovery_prior_exclusion_contract"]
    expected_prior_contract = {
        "old_prior": EXPECTED_OLD_PRIOR_IDENTITIES,
        "failed_attempt": EXPECTED_FAILED_ATTEMPT_IDENTITIES,
        "required_union": EXPECTED_RECOVERY_UNION_IDENTITIES,
        "old_prior_failed_attempt_overlap_required_zero": True,
        "all_five_identity_classes_required": True,
        "finalizer_required_after_recovery_freeze": True,
    }
    if {
        name: prior_contract.get(name) for name in expected_prior_contract
    } != expected_prior_contract:
        raise RuntimeError("v4r1 frozen prior-exclusion union contract mismatch")
    capacity = prereg["recovery_capacity_check"]
    expected_capacity = {
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
    if {name: capacity.get(name) for name in expected_capacity} != expected_capacity:
        raise RuntimeError("v4r1 frozen recovery-capacity contract mismatch")

    probe = prereg["rgb_history_probe"]
    expected_probe = {
        "attempts_authorized": 1,
        "run_only_after_complete_success_marker": True,
        "recipe_unchanged_from_v4": True,
        "thresholds_unchanged_from_v4": True,
        "recipe": FROZEN_PROBE_RECIPE,
        "thresholds": FROZEN_PROBE_THRESHOLDS,
        "trusted_input_contract": FROZEN_PROBE_TRUSTED_INPUT_CONTRACT,
    }
    if {name: probe.get(name) for name in expected_probe} != expected_probe:
        raise RuntimeError("v4r1 frozen RGB-history probe contract mismatch")
    action_support = prereg["action_support_authorization"]
    if action_support.get("authorizing_audit_id") != (
        "cube_gripper_carry_h3_v4r1_action_support_v2"
    ) or action_support.get("candidate_profile_counts") != {
        "train": 4096,
        "loader_validation": 512,
    } or action_support.get("total_candidate_profiles") != 4608 or action_support.get(
        "v2_is_only_authorizing_action_support_input"
    ) is not True:
        raise RuntimeError("v4r1 action-support authorization mismatch")

    declared_inputs = prereg.get("recovery_inputs")
    authorized_inputs = receipt.get("authorization_inputs")
    if not isinstance(declared_inputs, Mapping) or set(declared_inputs) != set(
        REQUIRED_FREEZE_AUTHORIZATION_INPUT_KEYS
    ):
        raise RuntimeError("v4r1 preregistration recovery-input set mismatch")
    if not isinstance(authorized_inputs, Mapping) or set(authorized_inputs) != set(
        REQUIRED_FREEZE_AUTHORIZATION_INPUT_KEYS
    ):
        raise RuntimeError("v4r1 freeze authorization-input set mismatch")
    for name in REQUIRED_FREEZE_AUTHORIZATION_INPUT_KEYS:
        _same_identity(
            authorized_inputs[name], declared_inputs[name], label=f"authorization input {name}"
        )
    action_identity = authorized_inputs["v4r1_action_support_audit"]
    if _identity_core(action_identity, label="v4r1 action-support audit") != (
        ACTION_SUPPORT_AUDIT_SHA256,
        ACTION_SUPPORT_AUDIT_SIZE_BYTES,
    ):
        raise RuntimeError("freeze does not bind canonical action-support audit v2")

    declared_implementation = prereg.get("identity")
    frozen_implementation = receipt.get("identity")
    if not isinstance(declared_implementation, Mapping) or set(
        declared_implementation
    ) != set(REQUIRED_FREEZE_IDENTITY_KEYS):
        raise RuntimeError("v4r1 preregistration implementation identity set mismatch")
    if not isinstance(frozen_implementation, Mapping) or set(
        frozen_implementation
    ) != set(REQUIRED_FREEZE_IDENTITY_KEYS):
        raise RuntimeError("v4r1 freeze implementation identity set mismatch")
    verified_implementation: dict[str, Any] = {}
    for name in REQUIRED_FREEZE_IDENTITY_KEYS:
        declared = declared_implementation[name]
        frozen = frozen_implementation[name]
        _same_identity(frozen, declared, label=f"implementation {name}")
        if not isinstance(declared, Mapping):
            raise RuntimeError(f"implementation {name} identity is malformed")
        logical_path = declared.get("path")
        expected_logical_path = EXPECTED_FREEZE_IDENTITY_PATHS[name]
        if logical_path != expected_logical_path:
            raise RuntimeError(f"implementation {name} path is not canonical")
        if name == "v4_builder":
            current_path = builder_path
        elif name == "v4_physics":
            current_path = physics_path
        else:
            current_path = ROOT / logical_path
        raw = _read_regular_file_nofollow(current_path, label=f"implementation {name}")
        current = _identity_from_bytes(current_path, raw)
        _same_identity(current, frozen, label=f"current implementation {name}")
        verified_implementation[name] = current

    return {
        "preregistration": prereg_identity,
        "freeze_receipt": _identity_from_bytes(receipt_path, receipt_raw),
        "authorization_inputs": dict(authorized_inputs),
        "identity": verified_implementation,
        "checks_passed": True,
        "_preregistration_document": prereg,
        "_freeze_receipt_document": receipt,
        "_preregistration_raw": prereg_raw,
        "_freeze_receipt_raw": receipt_raw,
    }


def validate_freeze_receipt(
    *,
    receipt_path: Path,
    prereg_path: Path,
    source_h5: Path,
    builder_path: Path = V4_BUILDER_PATH,
    physics_path: Path = V4_PHYSICS_PATH,
) -> dict[str, Any]:
    """Validate the immutable authorization before creating build output.

    The source H5 content digest is trusted only after verifying the receipt's
    own identity bindings and rehashing the complete current file.  A second
    complete hash is required after local generation and before publication.
    """

    contract_audit = _validate_freeze_contract_documents(
        receipt_path=receipt_path,
        prereg_path=prereg_path,
        builder_path=builder_path,
        physics_path=physics_path,
    )
    receipt = contract_audit["_freeze_receipt_document"]
    if not isinstance(receipt, Mapping):  # defensive invariant
        raise RuntimeError("Freeze receipt root must be an object")
    if not source_h5.is_file():
        raise FileNotFoundError(f"source H5 is missing: {source_h5}")
    if receipt.get("schema_version") != 1:
        raise RuntimeError("Freeze receipt schema_version must be 1")
    if receipt.get("protocol_id") != PROTOCOL:
        raise RuntimeError("Freeze receipt protocol_id mismatch")
    if receipt.get("recovery_authorization_id") != RECOVERY_AUTHORIZATION_ID:
        raise RuntimeError("Freeze receipt recovery_authorization_id mismatch")
    if receipt.get("status") != FREEZE_STATUS:
        raise RuntimeError(
            "Freeze receipt status does not authorize the v4r1 recovery build"
        )
    if receipt.get("checks_passed") is not True:
        raise RuntimeError("Freeze receipt checks_passed is not true")
    if receipt.get("authorized_splits") != list(ACTIVE_SPLITS):
        raise RuntimeError(
            "Freeze receipt authorized_splits must be exactly train and "
            "loader_validation"
        )
    public = receipt.get("public_test")
    if not isinstance(public, Mapping):
        raise RuntimeError("Freeze receipt is missing Public Test closure")
    if public.get("access_status") != "closed_not_read_not_scored" or any(
        public.get(name) is not False
        for name in ("opened", "read", "scored", "hashed")
    ):
        raise RuntimeError("Freeze receipt does not keep Public Test fully closed")
    if receipt.get("reference_model_training_or_scoring_authorized") is not False:
        raise RuntimeError("Freeze receipt unexpectedly authorizes model work")

    preregistration = receipt.get("preregistration")
    identity = receipt.get("identity")
    if not isinstance(preregistration, Mapping) or not isinstance(identity, Mapping):
        raise RuntimeError("Freeze receipt identity section is incomplete")
    verified_prereg = _verified_current_file_identity(
        receipt_entry=preregistration,
        current_path=prereg_path,
        label="preregistration",
    )
    try:
        builder_entry = identity["v4_builder"]
        physics_entry = identity["v4_physics"]
    except KeyError as error:
        raise RuntimeError(
            "Freeze receipt lacks v4_builder or v4_physics identity"
        ) from error
    if not isinstance(builder_entry, Mapping) or not isinstance(
        physics_entry, Mapping
    ):
        raise RuntimeError("Freeze receipt builder/physics identity is malformed")
    verified_builder = _verified_current_file_identity(
        receipt_entry=builder_entry,
        current_path=builder_path,
        label="v4_builder",
    )
    verified_physics = _verified_current_file_identity(
        receipt_entry=physics_entry,
        current_path=physics_path,
        label="v4_physics",
    )

    source = receipt.get("source_h5")
    if not isinstance(source, Mapping):
        raise RuntimeError("Freeze receipt source_h5 identity is missing")
    if source.get("symbol") != SOURCE_SYMBOL:
        raise RuntimeError("Freeze receipt source_h5 symbol mismatch")
    source_sha256 = str(source.get("sha256", ""))
    _sha256_digest_bytes(source_sha256, field_name="source_h5.sha256")
    source_size = source_h5.stat().st_size
    if source_size != int(source.get("size_bytes", -1)):
        raise RuntimeError("Current source H5 size differs from freeze receipt")
    with h5py.File(source_h5, "r", swmr=True) as handle:
        source_rows = int(handle["action"].shape[0])
        source_episodes = int(handle["ep_len"].shape[0])
    if source_rows != int(source.get("row_count", -1)):
        raise RuntimeError("Current source H5 row count differs from freeze receipt")
    if source_episodes != int(source.get("episode_count", -1)):
        raise RuntimeError(
            "Current source H5 episode count differs from freeze receipt"
        )
    observed_source_sha256 = file_sha256(source_h5)
    if observed_source_sha256 != source_sha256:
        raise RuntimeError("Current source H5 SHA256 differs from freeze receipt")

    postflight = _validate_freeze_contract_documents(
        receipt_path=receipt_path,
        prereg_path=prereg_path,
        builder_path=builder_path,
        physics_path=physics_path,
    )
    if postflight["preregistration"] != contract_audit["preregistration"] or postflight[
        "freeze_receipt"
    ] != contract_audit["freeze_receipt"] or postflight["identity"] != contract_audit[
        "identity"
    ]:
        raise RuntimeError("v4r1 authorization inputs changed during source validation")

    return {
        "path": portable_contextworld_path(receipt_path),
        "sha256": contract_audit["freeze_receipt"]["sha256"],
        "size_bytes": contract_audit["freeze_receipt"]["size_bytes"],
        "protocol_id": PROTOCOL,
        "recovery_authorization_id": RECOVERY_AUTHORIZATION_ID,
        "status": receipt["status"],
        "checks_passed": True,
        "authorized_splits": list(ACTIVE_SPLITS),
        "public_test": dict(public),
        "preregistration": verified_prereg,
        "identity": dict(contract_audit["identity"]),
        "authorization_inputs": dict(contract_audit["authorization_inputs"]),
        "source_h5": {
            "symbol": SOURCE_SYMBOL,
            "sha256": source_sha256,
            "size_bytes": source_size,
            "row_count": source_rows,
            "episode_count": source_episodes,
            "content_hash_reused_from_validated_freeze_receipt": True,
            "content_rehashed_by_builder_before_candidate_selection": True,
            "observed_sha256": observed_source_sha256,
        },
    }


def verify_source_h5_unchanged_after_build(
    source_h5: Path,
    *,
    frozen_source_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Rehash the full source after generation and before release publish."""

    expected = str(frozen_source_identity.get("sha256", ""))
    _sha256_digest_bytes(expected, field_name="source_h5.sha256")
    expected_size = int(frozen_source_identity.get("size_bytes", -1))
    observed_size = source_h5.stat().st_size
    if observed_size != expected_size:
        raise RuntimeError("Source H5 size changed during the v4r1 build")
    observed = file_sha256(source_h5)
    if observed != expected:
        raise RuntimeError("Source H5 SHA256 changed during the v4r1 build")
    return {
        "source_symbol": SOURCE_SYMBOL,
        "path_recorded": False,
        "expected_sha256": expected,
        "observed_sha256": observed,
        "size_bytes": observed_size,
        "full_content_rehashed_after_local_build_before_publish": True,
        "passed": True,
    }


def excluded_source_episodes_sha256(episode_ids: Sequence[int]) -> str:
    """Hash the canonical sorted episode exclusion set without JSON ambiguity."""

    values = np.asarray(list(episode_ids))
    if values.ndim != 1:
        raise ValueError("excluded source episodes must be one-dimensional")
    normalized: list[int] = []
    for raw in values.tolist():
        if isinstance(raw, (bool, np.bool_)) or not isinstance(
            raw, (int, np.integer)
        ):
            raise TypeError("excluded source episode IDs must be integers")
        value = int(raw)
        if value < 0:
            raise ValueError("excluded source episode IDs must be non-negative")
        normalized.append(value)
    if normalized != sorted(set(normalized)):
        raise ValueError(
            "excluded source episode IDs must be strictly sorted and unique"
        )
    payload = np.asarray(normalized, dtype="<i8").tobytes()
    return hashlib.sha256(
        b"contextworld-cube-prior-source-episodes-v1\0" + payload
    ).hexdigest()


def _canonical_sha256_values_digest(
    values: Sequence[str], *, field_name: str
) -> str:
    normalized = list(values)
    if normalized != sorted(set(normalized)):
        raise ValueError(f"{field_name} must be strictly sorted and unique")
    decoded = b"".join(
        _sha256_digest_bytes(value, field_name=field_name)
        for value in normalized
    )
    return hashlib.sha256(
        b"contextworld-cube-prior-content-exclusions-v1\0"
        + field_name.encode("ascii")
        + b"\0"
        + decoded
    ).hexdigest()


def _read_frozen_json_once(path: Path, *, label: str) -> tuple[bytes, Mapping[str, Any]]:
    """Read exactly the explicitly supplied frozen JSON, without discovery.

    ``O_NOFOLLOW`` prevents a symlink from changing the authorized target.
    The returned bytes are also used for the receipt digest, so the file is
    not reopened for hashing after validation.
    """

    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise RuntimeError(f"cannot exclusively read frozen {label}: {path}") from error
    try:
        with os.fdopen(descriptor, "rb") as stream:
            raw = stream.read()
    except Exception:
        # fdopen owns the descriptor after it succeeds; this branch is only
        # reached for failures before ownership is transferred.
        raise
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{label} is not valid UTF-8 JSON") from error
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"{label} root must be an object")
    return raw, payload


def _verified_receipt_binding(
    entry: Any, *, current_path: Path, label: str
) -> dict[str, Any]:
    if not isinstance(entry, Mapping):
        raise RuntimeError(f"prior exclusion receipt lacks {label} binding")
    return _verified_current_file_identity(
        receipt_entry=entry,
        current_path=current_path,
        label=label,
    )


def validate_prior_episode_exclusion_receipt(
    *,
    receipt_path: Path,
    prereg_path: Path,
    freeze_receipt_path: Path,
    freeze_receipt_audit: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the immutable union of all episodes/content seen before v4.

    The receipt is authoritative only when it binds the current preregistration,
    the current v4 freeze receipt, and the same upstream H5 identity already
    validated by that freeze receipt.  Prior artifacts themselves are not
    opened during a formal build; their frozen hashes and coverage roles are
    recorded in this receipt.
    """

    raw, receipt = _read_frozen_json_once(
        receipt_path, label="prior episode exclusion receipt"
    )
    if receipt.get("schema_version") != 1:
        raise RuntimeError("prior exclusion receipt schema_version must be 1")
    if receipt.get("protocol_id") != PROTOCOL:
        raise RuntimeError("prior exclusion receipt protocol_id mismatch")
    if receipt.get("recovery_authorization_id") != RECOVERY_AUTHORIZATION_ID:
        raise RuntimeError(
            "prior exclusion receipt recovery_authorization_id mismatch"
        )
    if receipt.get("status") != PRIOR_EXCLUSION_STATUS:
        raise RuntimeError(
            "prior exclusion receipt status does not authorize v4r1 recovery"
        )
    if receipt.get("receipt_id") != RECOVERY_PRIOR_RECEIPT_ID:
        raise RuntimeError("prior exclusion receipt_id mismatch")
    if receipt.get("checks_passed") is not True:
        raise RuntimeError("prior exclusion receipt checks_passed is not true")
    if receipt.get("reference_model_training_or_scoring") is not False:
        raise RuntimeError("prior exclusion receipt does not keep model work closed")
    if receipt.get("reference_model_optimizer_steps") != 0:
        raise RuntimeError("prior exclusion receipt optimizer steps must be zero")
    rgb_probe = receipt.get("rgb_probe")
    if not isinstance(rgb_probe, Mapping) or any(
        rgb_probe.get(name) is not False for name in ("opened", "run", "scored")
    ):
        raise RuntimeError("prior exclusion receipt does not keep RGB probe closed")
    recovery_contract = receipt.get("recovery_contract")
    required_recovery_checks = (
        "scientific_protocol_unchanged_from_v4",
        "old_prior_preserved_and_extended",
        "failed_attempt_source_action_scene_pair_query_all_excluded",
        "failed_attempt_raw_queries_deterministically_reconstructed",
        "all_inputs_reverified_unchanged_before_output",
        "original_v4_attempt_not_retried_or_overwritten",
    )
    if not isinstance(recovery_contract, Mapping) or any(
        recovery_contract.get(name) is not True
        for name in required_recovery_checks
    ):
        raise RuntimeError("prior exclusion recovery contract is incomplete")
    if any(
        recovery_contract.get(name) is not False
        for name in (
            "lance_opened_or_written",
            "public_test_opened_read_hashed_or_scored",
            "rgb_probe_run",
            "reference_model_training_or_scoring",
        )
    ):
        raise RuntimeError("prior exclusion recovery contract opened closed scope")

    public = receipt.get("public_test")
    if not isinstance(public, Mapping) or (
        public.get("access_status") != "closed_not_read_not_scored"
        or any(
            public.get(name) is not False
            for name in ("opened", "read", "scored", "hashed")
        )
    ):
        raise RuntimeError("prior exclusion receipt does not keep Public closed")

    prereg_binding = _verified_receipt_binding(
        receipt.get("preregistration"),
        current_path=prereg_path,
        label="preregistration",
    )
    freeze_binding = _verified_receipt_binding(
        receipt.get("freeze_receipt"),
        current_path=freeze_receipt_path,
        label="freeze_receipt",
    )
    if prereg_binding["sha256"] != str(
        freeze_receipt_audit["preregistration"]["sha256"]
    ):
        raise RuntimeError("prior exclusion preregistration binding disagrees with freeze")
    if freeze_binding["sha256"] != str(freeze_receipt_audit["sha256"]):
        raise RuntimeError("prior exclusion freeze-receipt binding mismatch")

    source = receipt.get("source_h5")
    frozen_source = freeze_receipt_audit.get("source_h5")
    if not isinstance(source, Mapping) or not isinstance(frozen_source, Mapping):
        raise RuntimeError("prior exclusion source_h5 identity is missing")
    required_source = {
        "symbol": SOURCE_SYMBOL,
        "sha256": str(frozen_source.get("sha256", "")),
        "size_bytes": int(frozen_source.get("size_bytes", -1)),
        "row_count": int(frozen_source.get("row_count", -1)),
        "episode_count": int(frozen_source.get("episode_count", -1)),
    }
    observed_source = {
        "symbol": source.get("symbol"),
        "sha256": str(source.get("sha256", "")),
        "size_bytes": int(source.get("size_bytes", -1)),
        "row_count": int(source.get("row_count", -1)),
        "episode_count": int(source.get("episode_count", -1)),
    }
    _sha256_digest_bytes(observed_source["sha256"], field_name="source_h5.sha256")
    if observed_source != required_source:
        raise RuntimeError("prior exclusion source H5 identity disagrees with freeze")

    episodes_raw = receipt.get("excluded_source_episodes")
    if not isinstance(episodes_raw, list) or not episodes_raw:
        raise RuntimeError("prior exclusion must contain a non-empty episode list")
    episodes = [int(value) for value in episodes_raw]
    episode_digest = excluded_source_episodes_sha256(episodes_raw)
    if int(receipt.get("excluded_source_episode_count", -1)) != len(episodes):
        raise RuntimeError("prior exclusion episode count mismatch")
    if receipt.get("excluded_source_episodes_sha256") != episode_digest:
        raise RuntimeError("prior exclusion episode digest mismatch")
    if (
        len(episodes) != EXPECTED_RECOVERY_EXCLUDED_SOURCE_EPISODE_COUNT
        or episode_digest
        != EXPECTED_RECOVERY_EXCLUDED_SOURCE_EPISODES_SHA256
    ):
        raise RuntimeError("prior exclusion source union is not the frozen v4r1 union")
    if episodes[-1] >= observed_source["episode_count"]:
        raise RuntimeError("prior exclusion episode is outside the source H5")

    coverage = receipt.get("coverage")
    expected_coverage = {name: True for name in REQUIRED_PRIOR_EPISODE_COVERAGE}
    if coverage != expected_coverage:
        raise RuntimeError(
            "prior exclusion coverage must explicitly include v3 formal, "
            "v3 smokes, v3 pilots, v4 preformal exploration, and the failed "
            "formal v4 attempt"
        )

    artifacts = receipt.get("input_artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise RuntimeError("prior exclusion input_artifacts must be non-empty")
    artifact_roles: set[str] = set()
    normalized_artifacts: list[dict[str, Any]] = []
    failed_artifact_kinds: set[str] = set()
    for index, artifact in enumerate(artifacts):
        if not isinstance(artifact, Mapping):
            raise RuntimeError(f"input_artifacts[{index}] must be an object")
        role = str(artifact.get("role", ""))
        path = str(artifact.get("path", ""))
        digest = str(artifact.get("sha256", ""))
        size = artifact.get("size_bytes")
        if role not in REQUIRED_PRIOR_EPISODE_COVERAGE:
            raise RuntimeError(f"input_artifacts[{index}] has unknown role {role!r}")
        if not path or any(
            part.lower() in {"validation", "validation.lance", "public", "public_test"}
            for part in Path(path).parts
        ):
            raise RuntimeError(f"input_artifacts[{index}] has forbidden path")
        _sha256_digest_bytes(digest, field_name=f"input_artifacts[{index}].sha256")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise RuntimeError(f"input_artifacts[{index}] has invalid size_bytes")
        artifact_roles.add(role)
        if role == "v4_failed_formal_attempts":
            failed_artifact_kinds.add(str(artifact.get("artifact_kind", "")))
        normalized_artifacts.append(
            {"role": role, "path": path, "sha256": digest, "size_bytes": size}
        )
    if artifact_roles != set(REQUIRED_PRIOR_EPISODE_COVERAGE):
        raise RuntimeError("prior exclusion input artifacts do not cover every role")
    if failed_artifact_kinds != {
        "failed_formal_attempt_receipt",
        "failed_attempt_query_reconstruction_receipt",
    }:
        raise RuntimeError(
            "prior exclusion must bind both failed-attempt content receipts"
        )

    content = receipt.get("prior_content_exclusions")
    if not isinstance(content, Mapping):
        raise RuntimeError("prior exclusion lacks prior_content_exclusions")
    normalized_content: dict[str, dict[str, Any]] = {}
    for field_name in PRIOR_CONTENT_EXCLUSION_FIELDS:
        entry = content.get(field_name)
        if not isinstance(entry, Mapping):
            raise RuntimeError(f"prior content exclusion {field_name} is missing")
        values = entry.get("values")
        if not isinstance(values, list) or not values:
            raise RuntimeError(f"prior content exclusion {field_name} must be non-empty")
        digest = _canonical_sha256_values_digest(values, field_name=field_name)
        if int(entry.get("count", -1)) != len(values):
            raise RuntimeError(f"prior content exclusion {field_name} count mismatch")
        if entry.get("sha256") != digest:
            raise RuntimeError(f"prior content exclusion {field_name} digest mismatch")
        expected_content = EXPECTED_RECOVERY_CONTENT[field_name]
        if (
            len(values) != int(expected_content["count"])
            or digest != str(expected_content["sha256"])
        ):
            raise RuntimeError(
                f"prior content exclusion {field_name} is not the frozen v4r1 union"
            )
        normalized_content[field_name] = {
            "values": list(values),
            "count": len(values),
            "sha256": digest,
        }

    return {
        "path": portable_contextworld_path(receipt_path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
        "protocol_id": PROTOCOL,
        "recovery_authorization_id": RECOVERY_AUTHORIZATION_ID,
        "status": PRIOR_EXCLUSION_STATUS,
        "checks_passed": True,
        "public_test": dict(public),
        "preregistration": prereg_binding,
        "freeze_receipt": freeze_binding,
        "source_h5": observed_source,
        "coverage": expected_coverage,
        "input_artifacts": normalized_artifacts,
        "excluded_source_episodes": episodes,
        "excluded_source_episode_count": len(episodes),
        "excluded_source_episodes_sha256": episode_digest,
        "prior_content_exclusions": normalized_content,
        "read_contract": {
            "explicit_cli_path_only": True,
            "symlink_followed": False,
            "receipt_bytes_read_once": True,
            "prior_artifacts_opened_by_builder": False,
        },
    }


def _fixed(values: np.ndarray, size: int) -> pa.FixedSizeListArray:
    flat = np.asarray(values, dtype=np.float32).reshape(-1, size)
    return pa.FixedSizeListArray.from_arrays(
        pa.array(flat.reshape(-1), type=pa.float32()), size
    )


SCHEMA = pa.schema(
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


def _lance_table_identity(
    table_path: Path,
    *,
    expected_row_count: int,
    expected_tree_sha256: str | None = None,
) -> dict[str, Any]:
    """Reopen one committed Lance table and bind its schema/rows/bytes."""

    if table_path.is_symlink() or not table_path.is_dir():
        raise RuntimeError(f"Lance table is not a real directory: {table_path}")
    receipts = regular_file_receipts(table_path)
    if not receipts:
        raise RuntimeError(f"Lance table contains no regular files: {table_path}")
    dataset = lance.dataset(str(table_path))
    if dataset.schema != SCHEMA:
        raise RuntimeError(
            f"Lance schema differs from the frozen v4 schema: {table_path}"
        )
    row_count = int(dataset.count_rows())
    if row_count != int(expected_row_count):
        raise RuntimeError(
            f"Lance row count mismatch for {table_path.name}: "
            f"expected={expected_row_count}, actual={row_count}"
        )
    tree_sha256 = directory_sha256(table_path)
    if (
        expected_tree_sha256 is not None
        and tree_sha256 != expected_tree_sha256
    ):
        raise RuntimeError(
            f"Lance tree hash mismatch for {table_path.name}: "
            f"expected={expected_tree_sha256}, actual={tree_sha256}"
        )
    return {
        "table": table_path.name,
        "schema_equals_frozen_v4": True,
        "row_count": row_count,
        "file_count": len(receipts),
        "size_bytes": sum(int(row["size_bytes"]) for row in receipts),
        "tree_sha256": tree_sha256,
        "file_receipts_sha256": _canonical_json_sha256(receipts),
        "passed": True,
    }


def _validate_release_lance_tables(
    root: Path,
    reports: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Validate exactly the two authorized Lance tables under one root."""

    if set(reports) != set(ACTIVE_SPLITS):
        raise RuntimeError("Lance validation requires exactly the active splits")
    identities: dict[str, dict[str, Any]] = {}
    for split in ACTIVE_SPLITS:
        report = reports[split]
        expected_name = f"{split}.lance"
        if report.get("table_path") != expected_name:
            raise RuntimeError(
                f"{split}: unexpected table path {report.get('table_path')!r}"
            )
        identity = _lance_table_identity(
            root / expected_name,
            expected_row_count=int(report["model_rows"]),
            expected_tree_sha256=str(report["table_sha256"]),
        )
        if int(report.get("table_files", -1)) != identity["file_count"]:
            raise RuntimeError(f"{split}: table file count changed")
        if int(report.get("table_bytes", -1)) != identity["size_bytes"]:
            raise RuntimeError(f"{split}: table byte count changed")
        identities[split] = identity
    return identities


def _validate_pair_counts(pair_counts: Mapping[str, int]) -> dict[str, int]:
    """Validate the closed Development-only split universe."""

    observed = set(pair_counts)
    expected = set(ACTIVE_SPLITS)
    if observed != expected:
        extra = sorted(observed - expected)
        missing = sorted(expected - observed)
        raise ValueError(
            "v4 is Development-only and accepts exactly the active splits; "
            f"extra={extra}, missing={missing}"
        )
    result: dict[str, int] = {}
    for split in ACTIVE_SPLITS:
        value = pair_counts[split]
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{split} pair count must be an integer")
        count = int(value)
        if count <= 0 or count % 4:
            raise ValueError(
                f"{split} pair count must be positive and divisible by 4"
            )
        result[split] = count
    return result


def _validate_frozen_pair_counts(
    pair_counts: Mapping[str, int],
) -> dict[str, int]:
    """Require the exact preregistered v4r1 Training/Development sizes."""

    counts = _validate_pair_counts(pair_counts)
    if counts != DEFAULT_PAIR_COUNTS:
        raise ValueError(
            "v4r1 recovery pair counts are frozen at "
            f"{DEFAULT_PAIR_COUNTS}, got {counts}"
        )
    return counts


def _validate_frozen_jpeg_quality(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError("jpeg quality must be an integer")
    quality = int(value)
    if quality != FROZEN_JPEG_QUALITY:
        raise ValueError(
            f"v4r1 JPEG quality is frozen at {FROZEN_JPEG_QUALITY}, got {quality}"
        )
    return quality


def _validate_frozen_workers(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError("workers must be an integer")
    workers = int(value)
    if workers != FROZEN_WORKERS:
        raise ValueError(
            f"v4r1 worker count is frozen at {FROZEN_WORKERS}, got {workers}"
        )
    return workers


def _open_absolute_directory_nofollow(path: Path, *, create: bool = False) -> int:
    """Open an absolute directory one component at a time without aliases."""

    value = Path(os.path.abspath(path.expanduser()))
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(
        os, "O_NOFOLLOW", 0
    )
    descriptor = os.open("/", flags)
    try:
        for component in value.parts[1:]:
            if component.lower() in FORBIDDEN_PUBLIC_COMPONENTS:
                raise ValueError(
                    f"path contains forbidden Public component {component!r}"
                )
            try:
                child = os.open(component, flags, dir_fd=descriptor)
            except FileNotFoundError:
                if not create:
                    raise
                os.mkdir(component, dir_fd=descriptor)
                child = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        metadata = os.fstat(descriptor)
        if not stat.S_ISDIR(metadata.st_mode):
            raise ValueError(f"path is not a real directory: {value}")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _directory_identity_from_fd(descriptor: int) -> tuple[int, int]:
    metadata = os.fstat(descriptor)
    return int(metadata.st_dev), int(metadata.st_ino)


def _assert_parent_binding(output: Path, expected: tuple[int, int]) -> None:
    descriptor = _open_absolute_directory_nofollow(output.parent)
    try:
        if _directory_identity_from_fd(descriptor) != expected:
            raise RuntimeError("formal output parent identity changed during publication")
    finally:
        os.close(descriptor)


def _validate_formal_output(path: Path) -> Path:
    output = Path(os.path.abspath(path.expanduser()))
    expected = Path(os.path.abspath(DEFAULT_OUTPUT.expanduser()))
    if output != expected:
        raise ValueError(
            f"v4r1 formal output is frozen at {expected}, got {output}"
        )
    parent_fd = _open_absolute_directory_nofollow(output.parent, create=True)
    try:
        try:
            os.stat(output.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError(f"Refusing to overwrite {output}")
    finally:
        os.close(parent_fd)
    return output


def _validate_formal_input_file(
    path: Path, *, label: str, expected: Path | None = None
) -> Path:
    """Validate one formal CLI file without dereferencing an alias."""

    value = Path(os.path.abspath(path.expanduser()))
    forbidden = next(
        (
            component
            for component in value.parts
            if component.lower() in FORBIDDEN_PUBLIC_COMPONENTS
        ),
        None,
    )
    if forbidden is not None:
        raise ValueError(
            f"{label} contains forbidden Public component {forbidden!r}"
        )
    if expected is not None:
        frozen = Path(os.path.abspath(expected.expanduser()))
        if value != frozen:
            raise ValueError(f"{label} is frozen at {frozen}, got {value}")
    parent_fd = _open_absolute_directory_nofollow(value.parent)
    try:
        try:
            metadata = os.stat(value.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError as error:
            raise FileNotFoundError(f"{label} is missing: {value}") from error
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"{label} must be a real non-symlink regular file")
    finally:
        os.close(parent_fd)
    return value


def _is_path_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _validate_local_staging_root(path: Path, *, output: Path) -> Path:
    raw = Path(os.path.abspath(path.expanduser()))
    descriptor = _open_absolute_directory_nofollow(raw)
    try:
        staging_device = int(os.fstat(descriptor).st_dev)
    finally:
        os.close(descriptor)
    resolved = raw.resolve(strict=True)
    if resolved != raw:
        raise ValueError("staging root path must not contain a symlink")
    local_tmp = Path("/tmp").resolve()
    if resolved != local_tmp and not _is_path_within(resolved, local_tmp):
        raise ValueError("v4r1 staging root must be /tmp or its real descendant")
    if staging_device != int(os.stat(local_tmp).st_dev):
        raise ValueError("v4r1 staging root must use the same local filesystem as /tmp")
    if output == resolved or _is_path_within(output, resolved):
        raise ValueError("formal output must remain outside the local staging root")
    return resolved


def _anchor_ids() -> tuple[str, str, str, str]:
    values: Sequence[Any]
    if isinstance(V4_ACTION_ANCHORS, Mapping):
        values = tuple(V4_ACTION_ANCHORS)
    else:
        values = tuple(V4_ACTION_ANCHORS)
    ids: list[str] = []
    for value in values:
        if isinstance(value, str):
            identifier = value
        elif hasattr(value, "action_anchor_id"):
            identifier = str(value.action_anchor_id)
        elif isinstance(value, Sequence) and value:
            identifier = str(value[0])
        else:
            raise TypeError(f"Cannot resolve v4 action anchor ID from {value!r}")
        if not identifier:
            raise ValueError("v4 action anchor IDs must be non-empty")
        ids.append(identifier)
    if len(ids) != 4 or len(set(ids)) != 4:
        raise ValueError(f"v4 requires exactly four distinct anchors, got {ids}")
    return tuple(ids)  # type: ignore[return-value]


def action_profile_content_sha256(action_blocks: np.ndarray) -> str:
    """Hash only the actual canonical float32 action-block bytes.

    Split, candidate, anchor, shape strings, and other metadata are excluded.
    The required shape is checked separately so two IDs cannot differ because
    of metadata serialization choices.
    """

    blocks = np.asarray(action_blocks, dtype=np.float32)
    if blocks.shape != ACTION_PROFILE_SHAPE:
        raise ValueError(
            f"action profile must have shape {ACTION_PROFILE_SHAPE}, "
            f"got {blocks.shape}"
        )
    if not np.isfinite(blocks).all():
        raise ValueError("action profile must contain only finite float32 values")
    if np.count_nonzero(blocks[3]):
        raise ValueError(
            "action profile terminal fourth [5,5] format block must be exactly zero"
        )
    return hashlib.sha256(np.ascontiguousarray(blocks).tobytes()).hexdigest()


def _json_mapping(value: Any) -> dict[str, Any]:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(f"Expected dataclass or mapping receipt, got {type(value)!r}")


def _normalized_scene_vector(
    candidate: Mapping[str, Any],
    field_name: str,
    *,
    exact_size: int | None = None,
) -> np.ndarray:
    values = np.asarray(candidate[field_name], dtype=np.float64)
    if values.ndim != 1 or not values.size:
        raise ValueError(f"scene field {field_name} must be a non-empty vector")
    if exact_size is not None and values.size != exact_size:
        raise ValueError(
            f"scene field {field_name} must contain {exact_size} values"
        )
    if not np.isfinite(values).all():
        raise ValueError(f"scene field {field_name} must be finite")
    # Normalize byte order, contiguity, and signed zero before hashing.
    normalized = np.ascontiguousarray(values.astype("<f8", copy=True))
    normalized[normalized == 0.0] = 0.0
    return normalized


def scene_template_content_sha256(candidate: Any) -> str:
    """Hash only normalized inputs that generate the visible Cube scene.

    Split names, candidate IDs, action anchors, and action profiles are
    intentionally never inspected.  Integer fields are canonical little-
    endian int64; continuous fields are finite, one-dimensional little-endian
    float64 with signed zero normalized to positive zero.
    """

    values = _json_mapping(candidate)
    digest = hashlib.sha256()
    digest.update(SCENE_CONTENT_HASH_NORMALIZATION.encode("ascii") + b"\0")
    for field_name in (
        "source_row",
        "source_episode",
        "source_step",
        "simulator_seed",
        "task_id",
    ):
        value = values[field_name]
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise TypeError(f"scene field {field_name} must be an integer")
        digest.update(field_name.encode("ascii") + b"\0")
        digest.update(np.asarray([int(value)], dtype="<i8").tobytes())
    for field_name, exact_size in (
        ("qpos", 21),
        ("control", 7),
        ("cube_color", 3),
        ("target_position", 3),
    ):
        vector = _normalized_scene_vector(
            values,
            field_name,
            exact_size=exact_size,
        )
        digest.update(field_name.encode("ascii") + b"\0")
        digest.update(np.asarray([vector.size], dtype="<i8").tobytes())
        digest.update(vector.tobytes())
    return digest.hexdigest()


def _sha256_digest_bytes(value: str, *, field_name: str) -> bytes:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{field_name} must be a 64-character SHA256 hex digest")
    try:
        decoded = bytes.fromhex(value)
    except ValueError as error:
        raise ValueError(f"{field_name} must be hexadecimal") from error
    if len(decoded) != 32:
        raise ValueError(f"{field_name} must decode to 32 bytes")
    return decoded


def pair_content_sha256(
    scene_template_content_hash: str,
    action_profile_id: str,
) -> str:
    """Bind one normalized scene digest to one exact action-content digest."""

    scene_bytes = _sha256_digest_bytes(
        scene_template_content_hash,
        field_name="scene_template_content_hash",
    )
    profile_bytes = _sha256_digest_bytes(
        action_profile_id,
        field_name="action_profile_id",
    )
    return hashlib.sha256(scene_bytes + profile_bytes).hexdigest()


def _profile_from_candidate(
    candidate: CubeGraspRuleV4Candidate,
) -> tuple[str, str, np.ndarray, dict[str, Any]]:
    profile = candidate.action_profile
    receipt = _json_mapping(profile)
    blocks = np.asarray(v4_action_blocks(profile), dtype=np.float32)
    calculated = action_profile_content_sha256(blocks)
    claimed = str(receipt.get("action_profile_id", ""))
    anchor = str(receipt.get("action_anchor_id", ""))
    if claimed != calculated:
        raise RuntimeError(
            f"{candidate.candidate_id}: action profile ID is not its float32 "
            f"content hash: claimed={claimed}, calculated={calculated}"
        )
    if anchor not in _anchor_ids():
        raise RuntimeError(
            f"{candidate.candidate_id}: unknown action anchor {anchor!r}"
        )
    if receipt.get("split") != candidate.split:
        raise RuntimeError(f"{candidate.candidate_id}: profile split mismatch")
    if int(receipt.get("catalog_index", -1)) != candidate.catalog_index:
        raise RuntimeError(
            f"{candidate.candidate_id}: profile catalog index mismatch"
        )
    return anchor, claimed, blocks, receipt


def _eligible_source_rows(source: Path) -> list[tuple[int, int, int]]:
    """Return one high-quality table-level grasp state per source episode."""

    with h5py.File(source, "r", swmr=True) as handle:
        contact = np.asarray(handle["proprio_gripper_contact"][:, 0])
        opening = np.asarray(handle["proprio_gripper_opening"][:, 0])
        cube = np.asarray(handle["privileged_block_0_pos"])
        effector = np.asarray(handle["proprio_effector_pos"])
        episodes = np.asarray(handle["ep_idx"], dtype=np.int32)
        steps = np.asarray(handle["step_idx"], dtype=np.int32)
    distance = np.linalg.norm(cube - effector, axis=1)
    mask = (
        (contact >= 0.8)
        & (opening >= 0.45)
        & (opening <= 0.68)
        & (cube[:, 2] >= 0.017)
        & (cube[:, 2] <= 0.024)
        & (distance <= 0.008)
        & (steps >= 5)
        & (steps <= 160)
    )
    best: dict[int, tuple[float, int, int]] = {}
    for row in np.flatnonzero(mask):
        episode = int(episodes[row])
        candidate = (float(distance[row]), int(row), int(steps[row]))
        if episode not in best or candidate < best[episode]:
            best[episode] = candidate
    return [
        (row, episode, step)
        for episode, (_, row, step) in sorted(best.items())
    ]


def _source_h5_receipt(
    source: Path,
    *,
    eligible_episode_count: int,
    frozen_source_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the immutable Training-source identity used by this build."""

    with h5py.File(source, "r", swmr=True) as handle:
        row_count = int(handle["qpos"].shape[0])
        episode_count = int(handle["ep_len"].shape[0])
        if tuple(handle["qpos"].shape[1:]) != (21,):
            raise RuntimeError("Cube source qpos rows must have exactly 21 values")
        if tuple(handle["control"].shape[1:]) != (7,):
            raise RuntimeError("Cube source control rows must have exactly 7 values")
        audited_columns = (
            "qpos",
            "control",
            "action",
            "ep_idx",
            "step_idx",
            "proprio_gripper_contact",
            "proprio_gripper_opening",
            "privileged_block_0_pos",
            "proprio_effector_pos",
        )
        mismatched = {
            name: int(handle[name].shape[0])
            for name in audited_columns
            if int(handle[name].shape[0]) != row_count
        }
    if mismatched:
        raise RuntimeError(
            f"Cube source H5 row-count mismatch: expected={row_count}, "
            f"actual={mismatched}"
        )
    source_size = source.stat().st_size
    if source_size != int(frozen_source_identity.get("size_bytes", -1)):
        raise RuntimeError("Source H5 size changed after freeze-receipt validation")
    if row_count != int(frozen_source_identity.get("row_count", -1)):
        raise RuntimeError(
            "Source H5 row count changed after freeze-receipt validation"
        )
    if episode_count != int(frozen_source_identity.get("episode_count", -1)):
        raise RuntimeError(
            "Source H5 episode count changed after freeze-receipt validation"
        )
    source_sha256 = str(frozen_source_identity.get("sha256", ""))
    _sha256_digest_bytes(source_sha256, field_name="source_h5.sha256")
    return {
        "source_symbol": SOURCE_SYMBOL,
        "environment_variable": "CONTEXTWORLD_CUBE_H5",
        "local_source_path_recorded": False,
        "source_size_bytes": source_size,
        "source_row_count": row_count,
        "source_episode_count": episode_count,
        "source_file_sha256": source_sha256,
        "source_content_hash_reused_from_validated_freeze_receipt": True,
        "source_content_rehashed_by_builder_before_candidate_selection": bool(
            frozen_source_identity.get(
                "content_rehashed_by_builder_before_candidate_selection"
            )
        ),
        "eligible_source_episode_count": int(eligible_episode_count),
        "eligible_row_selection_rule": ELIGIBLE_ROW_SELECTION_RULE,
    }


def _prior_exclusion_sets(
    prior_exclusion_audit: Mapping[str, Any],
) -> tuple[set[int], dict[str, set[str]]]:
    if prior_exclusion_audit.get("checks_passed") is not True:
        raise RuntimeError("prior episode exclusion receipt was not validated")
    episodes = {
        int(value)
        for value in prior_exclusion_audit.get("excluded_source_episodes", [])
    }
    if not episodes:
        raise RuntimeError("validated prior exclusion episode set is empty")
    content_receipt = prior_exclusion_audit.get("prior_content_exclusions")
    if not isinstance(content_receipt, Mapping):
        raise RuntimeError("validated prior content exclusion sets are missing")
    content: dict[str, set[str]] = {}
    for field_name in PRIOR_CONTENT_EXCLUSION_FIELDS:
        entry = content_receipt.get(field_name)
        if not isinstance(entry, Mapping):
            raise RuntimeError(f"validated prior exclusion lacks {field_name}")
        values = {str(value) for value in entry.get("values", [])}
        if not values:
            raise RuntimeError(f"validated prior exclusion {field_name} is empty")
        content[field_name] = values
    return episodes, content


def build_candidate_catalogs(
    source: Path,
    *,
    pair_counts: Mapping[str, int],
    frozen_source_identity: Mapping[str, Any],
    prior_exclusion_audit: Mapping[str, Any],
) -> tuple[dict[str, list[CubeGraspRuleV4Candidate]], dict[str, Any]]:
    counts = _validate_pair_counts(pair_counts)
    excluded_episodes, prior_content = _prior_exclusion_sets(
        prior_exclusion_audit
    )
    eligible_before_exclusion = _eligible_source_rows(source)
    eligible_excluded = [
        row for row in eligible_before_exclusion if row[1] in excluded_episodes
    ]
    # The historical exclusion happens before the v4 assignment permutation;
    # it cannot be emulated by rejecting already-selected candidates.
    eligible = [
        row for row in eligible_before_exclusion if row[1] not in excluded_episodes
    ]
    required_pool = sum(
        CANDIDATE_POOL_MULTIPLIER * counts[split]
        for split in ACTIVE_SPLITS
    )
    if len(eligible) < required_pool:
        raise RuntimeError(
            f"Only {len(eligible)} eligible source episodes for "
            f"{required_pool} requested candidates"
        )
    order = np.random.default_rng(CANDIDATE_ASSIGNMENT_SEED).permutation(
        len(eligible)
    )
    cursor = 0
    assignments: dict[str, list[tuple[int, int, int]]] = {}
    for split in ACTIVE_SPLITS:
        count = CANDIDATE_POOL_MULTIPLIER * counts[split]
        assignments[split] = [
            eligible[int(index)] for index in order[cursor : cursor + count]
        ]
        cursor += count

    requested_rows = sorted(
        {row for rows in assignments.values() for row, _, _ in rows}
    )
    with h5py.File(source, "r", swmr=True) as handle:
        qpos = np.asarray(handle["qpos"][requested_rows], dtype=np.float64)
        control = np.asarray(handle["control"][requested_rows], dtype=np.float64)
    source_values = {
        row: (qpos[index], control[index])
        for index, row in enumerate(requested_rows)
    }

    catalogs: dict[str, list[CubeGraspRuleV4Candidate]] = {}
    catalog_profile_ids: dict[str, set[str]] = {}
    catalog_scene_hashes: dict[str, set[str]] = {}
    catalog_pair_hashes: dict[str, set[str]] = {}
    catalog_anchor_counts: dict[str, dict[str, int]] = {}
    anchors = _anchor_ids()
    for split in ACTIVE_SPLITS:
        catalog: list[CubeGraspRuleV4Candidate] = []
        profile_ids: set[str] = set()
        scene_hashes: set[str] = set()
        pair_hashes: set[str] = set()
        anchor_counts = {anchor: 0 for anchor in anchors}
        for local_index, (source_row, source_episode, source_step) in enumerate(
            assignments[split]
        ):
            formal_catalog_index = _formal_catalog_index(local_index)
            rng = np.random.default_rng(
                np.random.SeedSequence([CATALOG_SEEDS[split], local_index])
            )
            source_qpos, source_control = source_values[source_row]
            base_candidate = CubeGraspRuleCandidate(
                candidate_id=f"cube-carry-v4r1-{split}-{local_index:06d}",
                split=split,
                catalog_index=formal_catalog_index,
                source_row=source_row,
                source_episode=source_episode,
                source_step=source_step,
                simulator_seed=int(rng.integers(0, 2**31 - 1)),
                task_id=1 + local_index % 5,
                qpos=tuple(float(value) for value in source_qpos),
                control=tuple(float(value) for value in source_control),
                cube_color=tuple(float(value) for value in rng.uniform(0.18, 0.92, 3)),
                target_position=(
                    float(rng.uniform(0.32, 0.53)),
                    float(rng.uniform(-0.24, 0.24)),
                    0.02,
                ),
            )
            candidate = make_v4_candidate(base_candidate)
            anchor, profile_id, _, _ = _profile_from_candidate(candidate)
            expected_anchor = anchors[formal_catalog_index % len(anchors)]
            if anchor != expected_anchor:
                raise RuntimeError(
                    f"{candidate.candidate_id}: expected formal index%4 anchor "
                    f"{expected_anchor!r}, got {anchor!r}"
                )
            if profile_id in profile_ids:
                raise RuntimeError(
                    f"{candidate.candidate_id}: duplicate exact action profile "
                    f"inside {split}: {profile_id}"
                )
            scene_hash = scene_template_content_sha256(candidate)
            pair_hash = pair_content_sha256(scene_hash, profile_id)
            historical_overlap = {
                "action_profile_ids": profile_id
                in prior_content["action_profile_ids"],
                "scene_template_content_hashes": scene_hash
                in prior_content["scene_template_content_hashes"],
                "pair_content_hashes": pair_hash
                in prior_content["pair_content_hashes"],
            }
            if any(historical_overlap.values()):
                raise RuntimeError(
                    f"{candidate.candidate_id}: v4 catalog overlaps prior v3/"
                    f"exploration content: {historical_overlap}"
                )
            if source_episode in excluded_episodes:
                raise RuntimeError(
                    f"{candidate.candidate_id}: excluded source episode selected"
                )
            if scene_hash in scene_hashes or pair_hash in pair_hashes:
                raise RuntimeError(
                    f"{candidate.candidate_id}: duplicate normalized scene or "
                    f"scene/action pair content inside {split}"
                )
            profile_ids.add(profile_id)
            scene_hashes.add(scene_hash)
            pair_hashes.add(pair_hash)
            anchor_counts[anchor] += 1
            catalog.append(candidate)
        expected_catalog_per_anchor = len(catalog) // 4
        if set(anchor_counts.values()) != {expected_catalog_per_anchor}:
            raise RuntimeError(
                f"{split}: candidate catalog anchors are not balanced: "
                f"{anchor_counts}"
            )
        catalogs[split] = catalog
        catalog_profile_ids[split] = profile_ids
        catalog_scene_hashes[split] = scene_hashes
        catalog_pair_hashes[split] = pair_hashes
        catalog_anchor_counts[split] = anchor_counts

    left, right = ACTIVE_SPLITS
    source_overlap = len(
        {value.source_episode for value in catalogs[left]}
        & {value.source_episode for value in catalogs[right]}
    )
    profile_overlap = len(
        catalog_profile_ids[left] & catalog_profile_ids[right]
    )
    scene_overlap = len(catalog_scene_hashes[left] & catalog_scene_hashes[right])
    pair_overlap = len(catalog_pair_hashes[left] & catalog_pair_hashes[right])
    if source_overlap or profile_overlap or scene_overlap or pair_overlap:
        raise RuntimeError(
            "v4 catalog split-disjointness failed before simulation: "
            f"source_episode_overlap={source_overlap}, "
            f"exact_action_profile_id_overlap={profile_overlap}, "
            f"scene_template_content_hash_overlap={scene_overlap}, "
            f"pair_content_hash_overlap={pair_overlap}"
        )
    all_catalog_episodes = {
        candidate.source_episode
        for split in ACTIVE_SPLITS
        for candidate in catalogs[split]
    }
    all_catalog_profiles = set().union(*catalog_profile_ids.values())
    all_catalog_scenes = set().union(*catalog_scene_hashes.values())
    all_catalog_pairs = set().union(*catalog_pair_hashes.values())
    prior_overlap = {
        "source_episode_count": len(all_catalog_episodes & excluded_episodes),
        "action_profile_id_count": len(
            all_catalog_profiles & prior_content["action_profile_ids"]
        ),
        "scene_template_content_hash_count": len(
            all_catalog_scenes & prior_content["scene_template_content_hashes"]
        ),
        "pair_content_hash_count": len(
            all_catalog_pairs & prior_content["pair_content_hashes"]
        ),
    }
    if any(prior_overlap.values()):
        raise RuntimeError(f"v4 catalog historical exclusion failed: {prior_overlap}")
    receipt = {
        **_source_h5_receipt(
            source,
            eligible_episode_count=len(eligible_before_exclusion),
            frozen_source_identity=frozen_source_identity,
        ),
        "eligible_source_episode_count_before_prior_exclusion": len(
            eligible_before_exclusion
        ),
        "eligible_source_episode_count_removed_by_prior_exclusion": len(
            eligible_excluded
        ),
        "eligible_source_episode_count_after_prior_exclusion": len(eligible),
        "prior_episode_exclusion": {
            "receipt_sha256": str(prior_exclusion_audit["sha256"]),
            "excluded_source_episode_count": int(
                prior_exclusion_audit["excluded_source_episode_count"]
            ),
            "excluded_source_episodes_sha256": str(
                prior_exclusion_audit["excluded_source_episodes_sha256"]
            ),
            "applied_before_candidate_assignment": True,
            "catalog_overlap": prior_overlap,
            "passed": not any(prior_overlap.values()),
        },
        "candidate_pool_per_split": {
            split: len(catalogs[split]) for split in ACTIVE_SPLITS
        },
        "candidate_pool_multiplier": CANDIDATE_POOL_MULTIPLIER,
        "formal_catalog_namespace": {
            "catalog_index_offset": FORMAL_CATALOG_INDEX_OFFSET,
            "local_index_policy": FORMAL_CATALOG_LOCAL_INDEX_POLICY,
            "catalog_index_formula": (
                "FORMAL_CATALOG_INDEX_OFFSET + local_index"
            ),
            "scene_rng_task_and_candidate_id_use_local_index": True,
            "offset_positive": FORMAL_CATALOG_INDEX_OFFSET > 0,
            "offset_modulo_anchor_count": (
                FORMAL_CATALOG_INDEX_OFFSET % len(anchors)
            ),
            "per_split_ranges": {
                split: {
                    "local_index_start_inclusive": 0,
                    "local_index_stop_exclusive": len(catalogs[split]),
                    "catalog_index_start_inclusive": (
                        FORMAL_CATALOG_INDEX_OFFSET
                    ),
                    "catalog_index_stop_exclusive": (
                        FORMAL_CATALOG_INDEX_OFFSET
                        + len(catalogs[split])
                    ),
                }
                for split in ACTIVE_SPLITS
            },
        },
        "candidate_assignment_seed": CANDIDATE_ASSIGNMENT_SEED,
        "catalog_seeds": dict(CATALOG_SEEDS),
        "profile_split_seeds": {
            split: int(V4_PROFILE_SPLIT_SEEDS[split])
            for split in ACTIVE_SPLITS
        },
        "source_episode_overlap": source_overlap,
        "exact_action_profile_id_overlap": profile_overlap,
        "scene_template_content_hash_overlap": scene_overlap,
        "pair_content_hash_overlap": pair_overlap,
        "action_anchor_counts": catalog_anchor_counts,
        "action_anchor_family_overlap_expected": len(anchors),
    }
    return catalogs, receipt


_WORKER_SIMULATOR: CubeGraspRuleV4Simulator | None = None
_WORKER_REPLAY_SIMULATOR: CubeGraspRuleV4Simulator | None = None
_WORKER_QUALITY = 95


def _worker_initialize(quality: int) -> None:
    global _WORKER_SIMULATOR, _WORKER_REPLAY_SIMULATOR, _WORKER_QUALITY
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
    _WORKER_QUALITY = int(quality)
    _WORKER_SIMULATOR = CubeGraspRuleV4Simulator()
    _WORKER_REPLAY_SIMULATOR = CubeGraspRuleV4Simulator()
    if _WORKER_REPLAY_SIMULATOR is _WORKER_SIMULATOR:
        raise RuntimeError("primary and fresh-replay simulators must be distinct")
    atexit.register(_WORKER_SIMULATOR.close)
    atexit.register(_WORKER_REPLAY_SIMULATOR.close)


def _encode(value: np.ndarray) -> bytes:
    buffer = BytesIO()
    Image.fromarray(np.asarray(value, dtype=np.uint8)).save(
        buffer, format="JPEG", quality=_WORKER_QUALITY
    )
    return buffer.getvalue()


def _build_candidate(candidate: CubeGraspRuleV4Candidate) -> dict[str, Any] | None:
    assert _WORKER_SIMULATOR is not None
    assert _WORKER_REPLAY_SIMULATOR is not None
    if _WORKER_REPLAY_SIMULATOR is _WORKER_SIMULATOR:
        raise RuntimeError("fresh replay cannot share the primary simulator")
    result = _WORKER_SIMULATOR.build_pair(
        candidate,
        replay_simulator=_WORKER_REPLAY_SIMULATOR,
    )
    if result is None:
        return None
    scene_hash = scene_template_content_sha256(candidate)
    profile_id = str(candidate.action_profile.action_profile_id)
    pair_hash = pair_content_sha256(scene_hash, profile_id)
    return {
        "candidate": result["candidate"],
        "action_profile": result["action_profile"],
        "content_hashes": {
            "scene_template_content_hash": scene_hash,
            "action_profile_id": profile_id,
            "pair_content_hash": pair_hash,
        },
        "audit": result["audit"],
        "episodes": {
            mode: {
                "pixels": [_encode(value) for value in result[mode]["pixels"]],
                "action_blocks": np.asarray(
                    result[mode]["action_blocks"], dtype=np.float32
                ),
                "physical_state": np.asarray(
                    result[mode]["physical_state"], dtype=np.float32
                ),
                "hidden_value": float(result[mode]["hidden_value"]),
                "action_anchor_id": str(result[mode]["action_anchor_id"]),
                "action_profile_id": str(result[mode]["action_profile_id"]),
            }
            for mode in GRASP_MODES
        },
    }


def _validate_built_result(result: Mapping[str, Any], split: str) -> dict[str, Any]:
    candidate = _json_mapping(result["candidate"])
    profile_receipt = _json_mapping(result["action_profile"])
    content_hashes = _json_mapping(result["content_hashes"])
    audit = _json_mapping(result["audit"])
    episodes = result["episodes"]
    low = episodes[GRASP_MODES[0]]
    high = episodes[GRASP_MODES[1]]
    low_blocks = np.asarray(low["action_blocks"], dtype=np.float32)
    high_blocks = np.asarray(high["action_blocks"], dtype=np.float32)
    if not np.array_equal(low_blocks, high_blocks):
        raise RuntimeError("paired v4 action blocks differ between hidden modes")
    calculated_profile_id = action_profile_content_sha256(low_blocks)
    v4_audit = _json_mapping(audit.get("v4", {}))
    candidate_profile = _json_mapping(candidate.get("action_profile", {}))
    claimed_profile_ids = {
        str(profile_receipt.get("action_profile_id", "")),
        str(candidate_profile.get("action_profile_id", "")),
        str(v4_audit.get("action_profile_id", "")),
        str(low.get("action_profile_id", "")),
        str(high.get("action_profile_id", "")),
    }
    if claimed_profile_ids != {calculated_profile_id}:
        raise RuntimeError(
            "v4 action_profile_id does not consistently equal the actual "
            f"float32 action-block hash: {claimed_profile_ids}, "
            f"calculated={calculated_profile_id}"
        )
    claimed_anchors = {
        str(profile_receipt.get("action_anchor_id", "")),
        str(candidate_profile.get("action_anchor_id", "")),
        str(v4_audit.get("action_anchor_id", "")),
        str(low.get("action_anchor_id", "")),
        str(high.get("action_anchor_id", "")),
    }
    if len(claimed_anchors) != 1:
        raise RuntimeError(f"inconsistent v4 action anchor IDs: {claimed_anchors}")
    anchor = next(iter(claimed_anchors))
    if anchor not in _anchor_ids():
        raise RuntimeError(f"unknown v4 action anchor: {anchor!r}")
    if candidate.get("split") != split or profile_receipt.get("split") != split:
        raise RuntimeError("v4 built result split mismatch")
    expected_anchor = _anchor_ids()[int(candidate["catalog_index"]) % 4]
    if anchor != expected_anchor:
        raise RuntimeError(
            f"catalog_index%4 anchor mismatch: expected={expected_anchor}, "
            f"actual={anchor}"
        )
    calculated_scene_hash = scene_template_content_sha256(candidate)
    calculated_pair_hash = pair_content_sha256(
        calculated_scene_hash,
        calculated_profile_id,
    )
    if content_hashes != {
        "scene_template_content_hash": calculated_scene_hash,
        "action_profile_id": calculated_profile_id,
        "pair_content_hash": calculated_pair_hash,
    }:
        raise RuntimeError(
            "v4 content-hash receipt does not match normalized scene and "
            "actual action content"
        )
    constraints = _json_mapping(v4_audit.get("profile_constraints", {}))
    fresh_replay = _json_mapping(v4_audit.get("fresh_simulator_replay", {}))
    if fresh_replay.get("passed") is not True:
        raise RuntimeError("v4 pair lacks a passed fresh-simulator replay audit")
    if fresh_replay.get("independent_simulator_instance") is not True:
        raise RuntimeError("v4 replay audit did not use an independent simulator")
    if fresh_replay.get("provided_reusable_instance") is not True:
        raise RuntimeError("v4 builder replay audit did not use its worker instance")
    return {
        "candidate": candidate,
        "audit": audit,
        "query_hash": str(audit["hashes"]["query_pixels"]),
        "action_anchor_id": anchor,
        "action_profile_id": calculated_profile_id,
        "scene_template_content_hash": calculated_scene_hash,
        "pair_content_hash": calculated_pair_hash,
        "content_hash_receipt": content_hashes,
        "action_profile_receipt": profile_receipt,
        "profile_constraints": constraints,
        "fresh_simulator_replay": fresh_replay,
        "action_axis_minimum": low_blocks.min(axis=(0, 1)).tolist(),
        "action_axis_maximum": low_blocks.max(axis=(0, 1)).tolist(),
    }


def _record_batch(
    episode: Mapping[str, Any],
    *,
    episode_index: int,
    split: str,
    candidate: Mapping[str, Any],
    mode: str,
    action_anchor_id: str,
    action_profile_id: str,
    scene_template_content_hash: str,
    pair_content_hash: str,
) -> pa.RecordBatch:
    count = 4
    arrays: list[pa.Array] = [
        pa.array(np.full(count, episode_index, dtype=np.int32)),
        pa.array(np.arange(count, dtype=np.int32)),
        pa.array(episode["pixels"], type=pa.binary()),
        _fixed(np.asarray(episode["action_blocks"]), 25),
        _fixed(np.asarray(episode["physical_state"]), 7),
        _fixed(
            np.full((count, 1), episode["hidden_value"], dtype=np.float32),
            1,
        ),
        pa.array([candidate["candidate_id"]] * count),
        pa.array([mode] * count),
        pa.array([split] * count),
        pa.array(np.full(count, candidate["catalog_index"], dtype=np.int32)),
        pa.array(np.full(count, candidate["source_row"], dtype=np.int64)),
        pa.array(np.full(count, candidate["source_episode"], dtype=np.int32)),
        pa.array(np.full(count, candidate["source_step"], dtype=np.int32)),
        pa.array([action_anchor_id] * count),
        pa.array([action_profile_id] * count),
        pa.array([scene_template_content_hash] * count),
        pa.array([pair_content_hash] * count),
    ]
    return pa.record_batch(arrays, schema=SCHEMA)


@dataclass
class _BalancedAcceptance:
    pair_count: int
    anchors: tuple[str, ...] = field(default_factory=_anchor_ids)
    counts: dict[str, int] = field(init=False)
    profile_ids: set[str] = field(default_factory=set)
    duplicate_profile_candidates: int = 0
    quota_full_candidates: int = 0

    def __post_init__(self) -> None:
        _validate_pair_counts(
            {"train": self.pair_count, "loader_validation": self.pair_count}
        )
        self.counts = {anchor: 0 for anchor in self.anchors}

    @property
    def quota(self) -> int:
        return self.pair_count // len(self.anchors)

    @property
    def accepted_count(self) -> int:
        return sum(self.counts.values())

    @property
    def complete(self) -> bool:
        return self.accepted_count == self.pair_count and set(
            self.counts.values()
        ) == {self.quota}

    def consider(self, *, anchor: str, profile_id: str) -> bool:
        if anchor not in self.counts:
            raise ValueError(f"unknown action anchor {anchor!r}")
        if profile_id in self.profile_ids:
            self.duplicate_profile_candidates += 1
            return False
        if self.counts[anchor] >= self.quota:
            self.quota_full_candidates += 1
            return False
        self.counts[anchor] += 1
        self.profile_ids.add(profile_id)
        return True


def _flatten_numeric(
    prefix: str,
    value: Any,
    rows: dict[str, list[float]],
) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            _flatten_numeric(name, child, rows)
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            name = f"{prefix}[{index}]"
            _flatten_numeric(name, child, rows)
    elif isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
        value, (bool, np.bool_)
    ):
        scalar = float(value)
        if not math.isfinite(scalar):
            raise RuntimeError(f"non-finite profile constraint {prefix}={scalar}")
        rows.setdefault(prefix, []).append(scalar)


def _constraint_extrema(accepted: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values: dict[str, list[float]] = {}
    for row in accepted:
        _flatten_numeric("", row["profile_constraints"], values)
        profile = row["action_profile_receipt"]
        if "perturbation_coefficients" in profile:
            _flatten_numeric(
                "perturbation_coefficients",
                profile["perturbation_coefficients"],
                values,
            )
    return {
        name: {"minimum": min(rows), "maximum": max(rows)}
        for name, rows in sorted(values.items())
    }


def _fresh_replay_summary(
    accepted: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not accepted:
        raise ValueError("fresh replay summary requires accepted pairs")
    audits = [
        _json_mapping(row["fresh_simulator_replay"])
        for row in accepted
    ]
    mode_summaries: dict[str, Any] = {}
    for mode in GRASP_MODES:
        rows = [_json_mapping(audit["modes"][mode]) for audit in audits]
        mode_summaries[mode] = {
            "pair_count": len(rows),
            "all_checks_passed": all(row.get("passed") is True for row in rows),
            "maximum_physical_state_gap": max(
                float(row["maximum_physical_state_gap"]) for row in rows
            ),
            "maximum_simulator_state_gap": max(
                float(row["maximum_simulator_state_gap"]) for row in rows
            ),
            "total_changed_rgb_values": sum(
                int(row["changed_rgb_values"]) for row in rows
            ),
            "total_changed_pixels": sum(
                int(row["changed_pixels"]) for row in rows
            ),
        }
    summary = {
        "pair_count": len(audits),
        "mode_replay_count": len(audits) * len(GRASP_MODES),
        "all_pair_replays_passed": all(
            audit.get("passed") is True for audit in audits
        ),
        "all_independent_simulator_instances": all(
            audit.get("independent_simulator_instance") is True
            for audit in audits
        ),
        "all_used_worker_replay_instance": all(
            audit.get("provided_reusable_instance") is True
            for audit in audits
        ),
        "maximum_physical_state_gap": max(
            float(audit["maximum_physical_state_gap"]) for audit in audits
        ),
        "maximum_simulator_state_gap": max(
            float(audit["maximum_simulator_state_gap"]) for audit in audits
        ),
        "total_changed_rgb_values": sum(
            int(audit["total_changed_rgb_values"]) for audit in audits
        ),
        "total_changed_pixels": sum(
            int(audit["total_changed_pixels"]) for audit in audits
        ),
        "modes": mode_summaries,
        "source": "audit.v4.fresh_simulator_replay for every accepted pair",
        "query_gap_used_as_replay_substitute": False,
    }
    summary["passed"] = bool(
        summary["all_pair_replays_passed"]
        and summary["all_independent_simulator_instances"]
        and summary["all_used_worker_replay_instance"]
        and summary["maximum_physical_state_gap"] <= QUERY_STATE_TOLERANCE
        and summary["maximum_simulator_state_gap"] <= QUERY_STATE_TOLERANCE
        and summary["total_changed_rgb_values"] == 0
        and summary["total_changed_pixels"] == 0
        and all(row["all_checks_passed"] for row in mode_summaries.values())
    )
    return summary


def build_split(
    root: Path,
    *,
    split: str,
    pair_count: int,
    candidates: list[CubeGraspRuleV4Candidate],
    quality: int,
    workers: int,
    prior_exclusion_audit: Mapping[str, Any],
) -> dict[str, Any]:
    if split not in ACTIVE_SPLITS:
        raise ValueError(f"inactive split refused by v4 builder: {split!r}")
    _validate_pair_counts({split: pair_count, **{
        active: pair_count for active in ACTIVE_SPLITS if active != split
    }})
    excluded_episodes, prior_content = _prior_exclusion_sets(
        prior_exclusion_audit
    )
    candidate_episode_overlap = sorted(
        {
            int(candidate.source_episode)
            for candidate in candidates
            if int(candidate.source_episode) in excluded_episodes
        }
    )
    if candidate_episode_overlap:
        raise RuntimeError(
            f"{split}: candidate catalog contains historically excluded episodes"
        )
    table_path = root / f"{split}.lance"
    accepted: list[dict[str, Any]] = []
    tracker = _BalancedAcceptance(pair_count)
    started = time.monotonic()
    attempted = 0

    def batches() -> Iterator[pa.RecordBatch]:
        nonlocal attempted
        context = mp.get_context("spawn")
        with context.Pool(
            processes=workers,
            initializer=_worker_initialize,
            initargs=(quality,),
        ) as pool:
            episode_index = 0
            for index, result in enumerate(
                pool.imap(_build_candidate, candidates, chunksize=1)
            ):
                attempted = index + 1
                if result is None:
                    continue
                row = _validate_built_result(result, split)
                historical_overlap = {
                    "source_episode": int(row["candidate"]["source_episode"])
                    in excluded_episodes,
                    "action_profile_id": row["action_profile_id"]
                    in prior_content["action_profile_ids"],
                    "scene_template_content_hash": row[
                        "scene_template_content_hash"
                    ]
                    in prior_content["scene_template_content_hashes"],
                    "pair_content_hash": row["pair_content_hash"]
                    in prior_content["pair_content_hashes"],
                    "query_pixel_hash": row["query_hash"]
                    in prior_content["query_pixel_hashes"],
                }
                if any(historical_overlap.values()):
                    raise RuntimeError(
                        f"{split}: simulated pair overlaps prior v3/exploration "
                        f"evidence: {historical_overlap}"
                    )
                if not tracker.consider(
                    anchor=row["action_anchor_id"],
                    profile_id=row["action_profile_id"],
                ):
                    continue
                accepted.append(row)
                for mode in GRASP_MODES:
                    yield _record_batch(
                        result["episodes"][mode],
                        episode_index=episode_index,
                        split=split,
                        candidate=row["candidate"],
                        mode=mode,
                        action_anchor_id=row["action_anchor_id"],
                        action_profile_id=row["action_profile_id"],
                        scene_template_content_hash=row[
                            "scene_template_content_hash"
                        ],
                        pair_content_hash=row["pair_content_hash"],
                    )
                    episode_index += 1
                count = len(accepted)
                if count <= 3 or count % 128 == 0:
                    print(
                        f"{split}: accepted {count}/{pair_count}, "
                        f"anchors={tracker.counts}, attempted={attempted}, "
                        f"elapsed={time.monotonic() - started:.1f}s",
                        flush=True,
                    )
                if tracker.complete:
                    break
            if not tracker.complete:
                raise RuntimeError(
                    f"Only {tracker.accepted_count}/{pair_count} balanced "
                    f"valid {split} pairs after {attempted} candidates; "
                    f"anchor_counts={tracker.counts}"
                )

    lance.write_dataset(
        pa.RecordBatchReader.from_batches(SCHEMA, batches()),
        str(table_path),
        mode="create",
    )
    local_lance_identity = _lance_table_identity(
        table_path,
        expected_row_count=8 * pair_count,
    )
    action_minimum = np.min(
        np.asarray([row["action_axis_minimum"] for row in accepted]), axis=0
    )
    action_maximum = np.max(
        np.asarray([row["action_axis_maximum"] for row in accepted]), axis=0
    )
    fresh_replay_summary = _fresh_replay_summary(accepted)
    accepted_prior_overlap = {
        "source_episode_count": len(
            {int(row["candidate"]["source_episode"]) for row in accepted}
            & excluded_episodes
        ),
        "action_profile_id_count": len(
            {str(row["action_profile_id"]) for row in accepted}
            & prior_content["action_profile_ids"]
        ),
        "scene_template_content_hash_count": len(
            {str(row["scene_template_content_hash"]) for row in accepted}
            & prior_content["scene_template_content_hashes"]
        ),
        "pair_content_hash_count": len(
            {str(row["pair_content_hash"]) for row in accepted}
            & prior_content["pair_content_hashes"]
        ),
        "query_pixel_hash_count": len(
            {str(row["query_hash"]) for row in accepted}
            & prior_content["query_pixel_hashes"]
        ),
    }
    report = {
        "split": split,
        "pair_count": pair_count,
        "episode_count": 2 * pair_count,
        "model_rows": 8 * pair_count,
        "attempted_candidates": attempted,
        "acceptance_rate": pair_count / attempted,
        "duplicate_profile_candidates_skipped": tracker.duplicate_profile_candidates,
        "anchor_quota_full_candidates_skipped": tracker.quota_full_candidates,
        "table_path": table_path.name,
        "table_files": local_lance_identity["file_count"],
        "table_bytes": local_lance_identity["size_bytes"],
        "table_sha256": local_lance_identity["tree_sha256"],
        "local_lance_commit_validation": local_lance_identity,
        "catalog_seed": CATALOG_SEEDS[split],
        "query_hashes": [row["query_hash"] for row in accepted],
        "pair_ids": [row["candidate"]["candidate_id"] for row in accepted],
        "source_episodes": [row["candidate"]["source_episode"] for row in accepted],
        "action_profile_ids": [row["action_profile_id"] for row in accepted],
        "scene_template_content_hashes": [
            row["scene_template_content_hash"] for row in accepted
        ],
        "pair_content_hashes": [row["pair_content_hash"] for row in accepted],
        "action_anchor_ids": [row["action_anchor_id"] for row in accepted],
        "action_anchor_counts": dict(tracker.counts),
        "action_anchor_expected_count_each": pair_count // 4,
        "unique_action_profile_count": len(tracker.profile_ids),
        "unique_scene_template_content_hash_count": len(
            {row["scene_template_content_hash"] for row in accepted}
        ),
        "unique_pair_content_hash_count": len(
            {row["pair_content_hash"] for row in accepted}
        ),
        "profile_constraint_extrema": _constraint_extrema(accepted),
        "fresh_simulator_replay": fresh_replay_summary,
        "prior_episode_and_content_exclusion": {
            "receipt_sha256": str(prior_exclusion_audit["sha256"]),
            "excluded_source_episode_count": int(
                prior_exclusion_audit["excluded_source_episode_count"]
            ),
            "excluded_source_episodes_sha256": str(
                prior_exclusion_audit["excluded_source_episodes_sha256"]
            ),
            "candidate_catalog_source_episode_overlap_count": len(
                candidate_episode_overlap
            ),
            "accepted_overlap": accepted_prior_overlap,
            "passed": not candidate_episode_overlap
            and not any(accepted_prior_overlap.values()),
        },
        "action_axis_extrema": {
            f"axis_{index}": {
                "minimum": float(action_minimum[index]),
                "maximum": float(action_maximum[index]),
            }
            for index in range(5)
        },
        "minimum_history_cube_height_gap_m": min(
            row["audit"]["history_cube_height_gap_m"] for row in accepted
        ),
        "minimum_future_cube_height_gap_m": min(
            row["audit"]["future_cube_height_gap_m"] for row in accepted
        ),
        "maximum_query_physical_gap": max(
            row["audit"]["maximum_query_physical_gap"] for row in accepted
        ),
        "maximum_query_simulator_state_gap": max(
            row["audit"]["maximum_query_simulator_state_gap"] for row in accepted
        ),
        "maximum_prequery_object_state_residual": max(
            row["audit"]["maximum_prequery_object_state_residual"]
            for row in accepted
        ),
        "maximum_state_installations_after_x0": max(
            row["audit"]["state_installations_after_x0"] for row in accepted
        ),
        "all_causal_checks_passed": all(
            row["audit"]["passed"] for row in accepted
        ),
        "minimum_history_changed_rgb_values": min(
            row["audit"]["history_changed_rgb_values"] for row in accepted
        ),
        "minimum_future_changed_rgb_values": min(
            row["audit"]["future_changed_rgb_values"] for row in accepted
        ),
        "pairs": accepted,
    }
    report["passed"] = bool(
        report["all_causal_checks_passed"]
        and report["fresh_simulator_replay"]["passed"]
        and report["prior_episode_and_content_exclusion"]["passed"]
        and report["unique_action_profile_count"] == pair_count
        and report["unique_scene_template_content_hash_count"] == pair_count
        and report["unique_pair_content_hash_count"] == pair_count
        and set(report["action_anchor_counts"].values()) == {pair_count // 4}
    )
    return report


def _cross_split_audit(reports: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    if set(reports) != set(ACTIVE_SPLITS):
        raise ValueError("cross-split audit requires exactly the active splits")
    left_name, right_name = ACTIVE_SPLITS
    left, right = reports[left_name], reports[right_name]

    def overlap(field_name: str) -> list[Any]:
        return sorted(set(left[field_name]) & set(right[field_name]))

    query_overlap = overlap("query_hashes")
    source_overlap = overlap("source_episodes")
    profile_overlap = overlap("action_profile_ids")
    scene_content_overlap = overlap("scene_template_content_hashes")
    pair_content_overlap = overlap("pair_content_hashes")
    anchor_overlap = overlap("action_anchor_ids")
    expected_anchors = sorted(_anchor_ids())
    checks = {
        "query_pixel_hash_overlap_zero": not query_overlap,
        "source_episode_overlap_zero": not source_overlap,
        "exact_action_profile_id_overlap_zero": not profile_overlap,
        "scene_template_content_hash_overlap_zero": not scene_content_overlap,
        "pair_content_hash_overlap_zero": not pair_content_overlap,
        "four_common_action_anchor_families_expected": (
            anchor_overlap == expected_anchors
        ),
    }
    return {
        "split_pair": [left_name, right_name],
        "query_pixel_hash_overlap": {
            "count": len(query_overlap),
            "values": query_overlap,
        },
        "source_episode_overlap": {
            "count": len(source_overlap),
            "values": source_overlap,
        },
        "exact_action_profile_id_overlap": {
            "count": len(profile_overlap),
            "values": profile_overlap,
        },
        "scene_template_content_hash_overlap": {
            "count": len(scene_content_overlap),
            "values": scene_content_overlap,
        },
        "pair_content_hash_overlap": {
            "count": len(pair_content_overlap),
            "values": pair_content_overlap,
        },
        "action_anchor_family_overlap": {
            "count": len(anchor_overlap),
            "expected_count": 4,
            "values": anchor_overlap,
            "expected_values": expected_anchors,
            "interpretation": "expected shared anchor families; not exact profiles",
        },
        "pair_id_is_content_isolation_evidence": False,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _fresh_replay_build_summary(
    reports: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if set(reports) != set(ACTIVE_SPLITS):
        raise ValueError("fresh replay build summary requires active splits")
    split_summaries = {
        split: _json_mapping(reports[split]["fresh_simulator_replay"])
        for split in ACTIVE_SPLITS
    }
    return {
        "source": (
            "per-pair audit.v4.fresh_simulator_replay aggregated from "
            "both active splits"
        ),
        "query_gap_used_as_replay_substitute": False,
        "pair_count": sum(
            int(summary["pair_count"])
            for summary in split_summaries.values()
        ),
        "mode_replay_count": sum(
            int(summary["mode_replay_count"])
            for summary in split_summaries.values()
        ),
        "maximum_physical_state_gap": max(
            float(summary["maximum_physical_state_gap"])
            for summary in split_summaries.values()
        ),
        "maximum_simulator_state_gap": max(
            float(summary["maximum_simulator_state_gap"])
            for summary in split_summaries.values()
        ),
        "total_changed_rgb_values": sum(
            int(summary["total_changed_rgb_values"])
            for summary in split_summaries.values()
        ),
        "total_changed_pixels": sum(
            int(summary["total_changed_pixels"])
            for summary in split_summaries.values()
        ),
        "splits": split_summaries,
        "passed": all(
            summary.get("passed") is True
            for summary in split_summaries.values()
        ),
    }


def _request_payload(
    *,
    pair_counts: Mapping[str, int],
    source_receipt: Mapping[str, Any],
    jpeg_quality: int,
    workers: int,
    output: Path,
    freeze_receipt_audit: Mapping[str, Any],
    prior_exclusion_audit: Mapping[str, Any],
) -> dict[str, Any]:
    counts = _validate_frozen_pair_counts(pair_counts)
    jpeg_quality = _validate_frozen_jpeg_quality(jpeg_quality)
    workers = _validate_frozen_workers(workers)
    required_source_identity = (
        "source_symbol",
        "source_size_bytes",
        "source_row_count",
        "source_episode_count",
        "source_file_sha256",
        "eligible_row_selection_rule",
    )
    missing_source_identity = [
        name for name in required_source_identity if name not in source_receipt
    ]
    if missing_source_identity:
        raise ValueError(
            "source receipt is missing fixed identity fields: "
            f"{missing_source_identity}"
        )
    if source_receipt["source_symbol"] != SOURCE_SYMBOL:
        raise ValueError("source receipt uses an unexpected source_symbol")
    if prior_exclusion_audit.get("checks_passed") is not True:
        raise ValueError("request requires a validated prior exclusion receipt")
    source_prior = source_receipt.get("prior_episode_exclusion")
    if not isinstance(source_prior, Mapping) or source_prior.get("passed") is not True:
        raise ValueError("source catalog did not pass prior episode exclusion")
    if source_prior.get("receipt_sha256") != prior_exclusion_audit.get("sha256"):
        raise ValueError("source catalog and prior exclusion receipt disagree")
    active_profile_seeds = {
        split: int(V4_PROFILE_SPLIT_SEEDS[split])
        for split in ACTIVE_SPLITS
    }
    return {
        "protocol": PROTOCOL,
        "recovery_authorization_id": RECOVERY_AUTHORIZATION_ID,
        "capability": CAPABILITY_NAME,
        "display_name_zh": "Cube 夹爪升降是否带动方块（多动作支持）",
        "transition_rule": (
            "MuJoCo generalized-force coupling; no state installation after x0"
        ),
        "evidence_scope": EVIDENCE_SCOPE,
        "profile_split_policy": PROFILE_SPLIT_POLICY,
        "active_splits": list(ACTIVE_SPLITS),
        "public_test_opened": False,
        "public_test_generated": False,
        "freeze_receipt": {
            "path": str(freeze_receipt_audit["path"]),
            "sha256": str(freeze_receipt_audit["sha256"]),
            "size_bytes": int(freeze_receipt_audit["size_bytes"]),
            "status": str(freeze_receipt_audit["status"]),
            "checks_passed": bool(freeze_receipt_audit["checks_passed"]),
        },
        "prior_episode_exclusion_receipt": {
            "path": str(prior_exclusion_audit["path"]),
            "sha256": str(prior_exclusion_audit["sha256"]),
            "size_bytes": int(prior_exclusion_audit["size_bytes"]),
            "status": str(prior_exclusion_audit["status"]),
            "checks_passed": bool(prior_exclusion_audit["checks_passed"]),
            "excluded_source_episode_count": int(
                prior_exclusion_audit["excluded_source_episode_count"]
            ),
            "excluded_source_episodes_sha256": str(
                prior_exclusion_audit["excluded_source_episodes_sha256"]
            ),
            "coverage": dict(prior_exclusion_audit["coverage"]),
            "prior_content_exclusions": {
                field_name: {
                    "count": int(
                        prior_exclusion_audit["prior_content_exclusions"]
                        [field_name]["count"]
                    ),
                    "sha256": str(
                        prior_exclusion_audit["prior_content_exclusions"]
                        [field_name]["sha256"]
                    ),
                }
                for field_name in PRIOR_CONTENT_EXCLUSION_FIELDS
            },
            "applied_before_candidate_assignment": True,
            "catalog_overlap": dict(source_prior["catalog_overlap"]),
            "public_test_read": False,
        },
        "source_content_sha256": str(
            freeze_receipt_audit["source_h5"]["sha256"]
        ),
        "hidden_modes": list(GRASP_MODES),
        "pair_counts": counts,
        "reproducibility_contract": {
            "candidate_assignment_seed": CANDIDATE_ASSIGNMENT_SEED,
            "catalog_seeds": dict(CATALOG_SEEDS),
            "profile_split_seeds": active_profile_seeds,
            "candidate_pool_multiplier": CANDIDATE_POOL_MULTIPLIER,
            "formal_catalog_index_offset": FORMAL_CATALOG_INDEX_OFFSET,
            "formal_catalog_local_index_policy": (
                FORMAL_CATALOG_LOCAL_INDEX_POLICY
            ),
            "formal_catalog_index_formula": (
                "FORMAL_CATALOG_INDEX_OFFSET + local_index"
            ),
            "scene_rng_task_and_candidate_id_use_local_index": True,
            "eligible_row_selection_rule": ELIGIBLE_ROW_SELECTION_RULE,
            "source_h5_identity": {
                "size_bytes": int(source_receipt["source_size_bytes"]),
                "row_count": int(source_receipt["source_row_count"]),
                "episode_count": int(source_receipt["source_episode_count"]),
                "file_sha256": str(source_receipt["source_file_sha256"]),
            },
        },
        "action_profile_contract": {
            "action_anchor_ids": list(_anchor_ids()),
            "anchor_count": 4,
            "anchor_assignment": "catalog_index modulo 4",
            "formal_catalog_index_offset": FORMAL_CATALOG_INDEX_OFFSET,
            "formal_catalog_offset_positive_and_four_aligned": True,
            "preformal_catalog_indices_excluded": [0, 1],
            "failed_v4_formal_catalog_range_excluded": {
                "start_inclusive": 1_000_000,
                "stop_exclusive": 1_002_048,
            },
            "each_split_anchor_balance": "exact",
            "action_profile_id": (
                "sha256 of only contiguous actual float32 [4,5,5] "
                "action-block bytes"
            ),
            "exact_profile_ids_split_disjoint": True,
            "anchor_families_shared_across_active_splits": True,
            "terminal_fourth_block": {
                "block_index": 3,
                "shape": [5, 5],
                "dtype": "float32",
                "all_values_exactly_zero": True,
                "role": "format-only terminal block; no transition target",
            },
        },
        "content_identity_contract": {
            "scene_template_content_hash": {
                "algorithm": "sha256",
                "normalization_version": (
                    SCENE_CONTENT_HASH_NORMALIZATION
                ),
                "cross_version_comparable_with_v3": True,
                "included_fields": [
                    "source_row",
                    "source_episode",
                    "source_step",
                    "simulator_seed",
                    "task_id",
                    "qpos",
                    "control",
                    "cube_color",
                    "target_position",
                ],
                "excluded_fields": [
                    "split",
                    "candidate_id",
                    "action_anchor_id",
                    "action_profile_id",
                    "action_profile",
                ],
                "integer_encoding": "little-endian int64",
                "continuous_encoding": (
                    "finite 1-D little-endian float64; signed zero canonicalized"
                ),
            },
            "pair_content_hash": (
                "sha256(raw 32-byte scene_template_content_hash digest + "
                "raw 32-byte action_profile_id digest)"
            ),
            "pair_id_is_content_isolation_evidence": False,
            "scene_and_pair_hashes_split_disjoint": True,
            "scene_pair_profile_and_query_disjoint_from_prior_v3_and_exploration": (
                True
            ),
            "scene_pair_profile_and_query_disjoint_from_failed_v4_attempt": True,
        },
        "fresh_simulator_replay_contract": {
            "required_for_every_accepted_pair": True,
            "primary_and_replay_simulators_distinct": True,
            "environments_not_shared": True,
            "one_reusable_primary_and_one_reusable_replay_instance_per_worker": True,
            "maximum_physical_state_gap": QUERY_STATE_TOLERANCE,
            "maximum_complete_simulator_state_gap": QUERY_STATE_TOLERANCE,
            "pixels_bitwise_equal": True,
            "actions_bitwise_equal": True,
            "query_gap_may_substitute_for_replay": False,
        },
        "privileged_columns": list(PRIVILEGED_COLUMNS),
        "model_visible_columns": ["pixels", "action_block"],
        "jpeg_quality": int(jpeg_quality),
        "workers": int(workers),
        "logical_default_output": DEFAULT_OUTPUT_LOGICAL.as_posix(),
        "resolved_output": portable_contextworld_path(output),
        "source": dict(source_receipt),
    }


def _manifest_payload(
    output: Path,
    *,
    build_report: Mapping[str, Any],
) -> dict[str, Any]:
    receipts = regular_file_receipts(output)
    request = build_report.get("request")
    if not isinstance(request, Mapping):
        raise ValueError("manifest requires the frozen build request")
    prior = request.get("prior_episode_exclusion_receipt")
    if not isinstance(prior, Mapping):
        raise ValueError("manifest requires the prior exclusion receipt summary")
    return {
        "protocol": PROTOCOL,
        "recovery_authorization_id": RECOVERY_AUTHORIZATION_ID,
        "evidence_scope": EVIDENCE_SCOPE,
        "profile_split_policy": PROFILE_SPLIT_POLICY,
        "active_splits": list(ACTIVE_SPLITS),
        "public_test_opened": False,
        "public_test_generated": False,
        "prior_episode_exclusion_receipt": {
            "sha256": str(prior["sha256"]),
            "excluded_source_episode_count": int(
                prior["excluded_source_episode_count"]
            ),
            "excluded_source_episodes_sha256": str(
                prior["excluded_source_episodes_sha256"]
            ),
        },
        "files": {
            str(row["path"]): str(row["sha256"])
            for row in receipts
        },
        "file_count_without_manifest": len(receipts),
        "bytes_without_manifest": sum(
            int(row["size_bytes"]) for row in receipts
        ),
        "build_passed": bool(build_report["passed"]),
    }


def _path_exists_or_is_symlink(path: Path) -> bool:
    try:
        os.lstat(path)
    except FileNotFoundError:
        return False
    return True


def _bound_receipt(
    receipts: Sequence[Mapping[str, Any]], *, relative_path: str
) -> dict[str, Any]:
    matches = [row for row in receipts if row.get("path") == relative_path]
    if len(matches) != 1:
        raise RuntimeError(
            f"release must contain exactly one {relative_path}, got {len(matches)}"
        )
    row = matches[0]
    return {
        "path": relative_path,
        "size_bytes": int(row["size_bytes"]),
        "sha256": str(row["sha256"]),
    }


def _validate_release_root_namespace(
    root: Path, *, success_marker_expected: bool
) -> None:
    required = set(REQUIRED_RELEASE_METADATA) | set(
        f"{split}.lance" for split in ACTIVE_SPLITS
    )
    if success_marker_expected:
        required.add(SUCCESS_MARKER_NAME)
    try:
        with os.scandir(root) as iterator:
            entries = {entry.name: entry for entry in iterator}
    except FileNotFoundError as error:
        raise FileNotFoundError(f"release root is missing: {root}") from error
    if set(entries) != required:
        raise RuntimeError(
            "release root namespace must contain exactly the two authorized "
            f"Lance tables and frozen metadata; got {sorted(entries)}"
        )
    for name in REQUIRED_RELEASE_METADATA:
        if not entries[name].is_file(follow_symlinks=False):
            raise RuntimeError(f"release metadata must be a real file: {name}")
    for split in ACTIVE_SPLITS:
        name = f"{split}.lance"
        if not entries[name].is_dir(follow_symlinks=False):
            raise RuntimeError(f"authorized Lance table must be a real directory: {name}")
    if success_marker_expected and not entries[SUCCESS_MARKER_NAME].is_file(
        follow_symlinks=False
    ):
        raise RuntimeError("success marker must be a real regular file")


def _publish_staged_release(
    *,
    staged_root: Path,
    output: Path,
    reports: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Copy a verified local release to NFS and publish success last.

    A failed or interrupted copy is intentionally left in place without the
    success marker.  This function never removes a destination and therefore
    cannot delete or overwrite an existing release after a race.
    """

    staged_root = Path(staged_root)
    output = Path(os.path.abspath(Path(output).expanduser()))
    parent_fd = _open_absolute_directory_nofollow(output.parent, create=True)
    try:
        try:
            os.stat(output.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError(f"Refusing to overwrite {output}")
    finally:
        os.close(parent_fd)
    _validate_release_root_namespace(staged_root, success_marker_expected=False)
    local_receipts = regular_file_receipts(staged_root)
    if any(row["path"] == SUCCESS_MARKER_NAME for row in local_receipts):
        raise RuntimeError("local staging must not contain a success marker")
    local_lance = _validate_release_lance_tables(staged_root, reports)
    local_tree_sha256 = directory_sha256(staged_root)
    local_receipts_sha256 = _canonical_json_sha256(local_receipts)

    parent_fd = _open_absolute_directory_nofollow(output.parent, create=True)
    parent_identity = _directory_identity_from_fd(parent_fd)
    anchored_output = Path(f"/proc/self/fd/{parent_fd}") / output.name
    try:
        try:
            os.stat(output.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError(f"Refusing to overwrite {output}")
        shutil.copytree(
            staged_root,
            anchored_output,
            copy_function=shutil.copy2,
            dirs_exist_ok=False,
        )
    except BaseException:
        os.close(parent_fd)
        raise

    _assert_parent_binding(output, parent_identity)
    _validate_release_root_namespace(
        anchored_output, success_marker_expected=False
    )
    destination_receipts = regular_file_receipts(anchored_output)
    if destination_receipts != local_receipts:
        os.close(parent_fd)
        raise RuntimeError(
            "published release files differ in path, size, or SHA256"
        )
    destination_tree_sha256 = directory_sha256(anchored_output)
    if destination_tree_sha256 != local_tree_sha256:
        os.close(parent_fd)
        raise RuntimeError("published release tree SHA256 differs from staging")
    destination_lance = _validate_release_lance_tables(anchored_output, reports)
    if destination_lance != local_lance:
        os.close(parent_fd)
        raise RuntimeError("published Lance identities differ from local commits")

    bound_files = {
        name: _bound_receipt(destination_receipts, relative_path=name)
        for name in REQUIRED_RELEASE_METADATA
    }
    success_payload = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "recovery_authorization_id": RECOVERY_AUTHORIZATION_ID,
        "status": "complete",
        "checks_passed": True,
        "public_test_opened": False,
        "public_test_generated": False,
        "staging": {
            "local_before_publish": True,
            "path_recorded": False,
        },
        "publication": {
            "method": "verified_x_exclusive_copytree",
            "copy_function": "shutil.copy2",
            "dirs_exist_ok": False,
            "nonempty_directory_rename_used": False,
            "success_marker_written_last": True,
            "failed_copy_is_never_marked_complete": True,
            "file_count_without_success_marker": len(destination_receipts),
            "bytes_without_success_marker": sum(
                int(row["size_bytes"]) for row in destination_receipts
            ),
            "tree_sha256_without_success_marker": destination_tree_sha256,
            "file_receipts_sha256": local_receipts_sha256,
            "source_and_destination_file_receipts_equal": True,
        },
        "bound_files": bound_files,
        "lance_tables": destination_lance,
        "file_receipts_without_success_marker": destination_receipts,
    }
    marker_bytes = (
        json.dumps(success_payload, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    output_fd = os.open(
        output.name,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=parent_fd,
    )
    marker_created = False
    marker_fd: int | None = None
    try:
        marker_fd = os.open(
            SUCCESS_MARKER_NAME,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0),
            0o644,
            dir_fd=output_fd,
        )
        marker_created = True
        with os.fdopen(marker_fd, "wb") as stream:
            marker_fd = None
            stream.write(marker_bytes)
            stream.flush()
            try:
                os.fsync(stream.fileno())
            except OSError as error:
                if error.errno not in {
                    errno.EINVAL,
                    errno.ENOTSUP,
                    errno.EOPNOTSUPP,
                }:
                    raise
        _assert_parent_binding(output, parent_identity)
        _validate_release_root_namespace(
            anchored_output, success_marker_expected=True
        )
        expected_with_marker = destination_receipts + [
            {
                "path": SUCCESS_MARKER_NAME,
                "size_bytes": len(marker_bytes),
                "sha256": hashlib.sha256(marker_bytes).hexdigest(),
            }
        ]
        observed_with_marker = regular_file_receipts(anchored_output)
        if observed_with_marker != sorted(
            expected_with_marker, key=lambda row: str(row["path"])
        ):
            raise RuntimeError("release changed during success-marker commit")
        tree_with_success = directory_sha256(anchored_output)
        _assert_parent_binding(output, parent_identity)
        result = {
            "path": SUCCESS_MARKER_NAME,
            "sha256": hashlib.sha256(marker_bytes).hexdigest(),
            "size_bytes": len(marker_bytes),
            "tree_sha256_without_success_marker": destination_tree_sha256,
            "tree_sha256_with_success_marker": tree_with_success,
            "file_receipts_sha256": local_receipts_sha256,
            "lance_tables": destination_lance,
            "checks_passed": True,
        }
    except BaseException:
        if marker_fd is not None:
            os.close(marker_fd)
        if marker_created:
            try:
                os.unlink(SUCCESS_MARKER_NAME, dir_fd=output_fd)
            except FileNotFoundError:
                pass
        raise
    finally:
        os.close(output_fd)
        os.close(parent_fd)
    return result


def _audit_causal_contract_from_real_replay(
    reports: Mapping[str, Mapping[str, Any]],
    fresh_replay_build_summary: Mapping[str, Any],
) -> dict[str, Any]:
    if set(reports) != set(ACTIVE_SPLITS):
        raise ValueError("causal audit requires exactly the active splits")
    all_reports = [reports[split] for split in ACTIVE_SPLITS]
    maximum_query_state_gap = max(
        float(row["maximum_query_simulator_state_gap"])
        for row in all_reports
    )
    return audit_causal_data_contract(
        component_id="cube_gripper_carry_rule_v4r1_recovery_development",
        evidence_scope=EVIDENCE_SCOPE,
        continuous_environment_trajectory=True,
        state_installations_after_x0=max(
            int(row["maximum_state_installations_after_x0"])
            for row in all_reports
        ),
        query_simulator_recreated=False,
        maximum_query_state_gap=maximum_query_state_gap,
        query_state_tolerance=QUERY_STATE_TOLERANCE,
        query_pixels_exact=True,
        query_actions_exact=True,
        history_effect_present=min(
            float(row["minimum_history_cube_height_gap_m"])
            for row in all_reports
        )
        >= MINIMUM_EFFECT_GAP_M,
        true_future_effect_present=min(
            float(row["minimum_future_cube_height_gap_m"])
            for row in all_reports
        )
        >= MINIMUM_EFFECT_GAP_M,
        x0_policy="shared_visible_start",
        x0_static_leakage_check_passed=True,
        solver_cache_check_required=True,
        solver_cache_check_passed=(
            fresh_replay_build_summary.get("passed") is True
        ),
        evidence=(
            "Each condition resets only before x0 and then uses env.step.",
            "The hidden rule changes qfrc_applied, not qpos or qvel.",
            "The complete query audit includes solver warm-start state.",
            "Every pair was rerun in a distinct simulator and matched full "
            "simulator state and RGB exactly.",
            "The query-state gap is only the paired-query gate and is not "
            "used as a deterministic-replay substitute.",
            "Only Training and Development evidence was generated.",
        ),
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    values = list(sys.argv[1:] if argv is None else argv)
    forbidden_pair_options = (
        "--public-test-pairs",
        "--validation-pairs",
        "--test-pairs",
    )
    for value in values:
        option = value.split("=", 1)[0].lower()
        if option.startswith(("--public", "--validation", "--test")):
            raise ValueError(
                "v4 builder explicitly refuses every validation/Public Test option"
            )
        if any(
            value == option or value.startswith(f"{option}=")
            for option in forbidden_pair_options
        ):
            raise ValueError(
                "v4 builder explicitly refuses validation/Public Test pairs; "
                "only --train-pairs and --development-pairs are active"
            )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--prereg", type=Path, required=True)
    parser.add_argument(
        "--freeze-receipt",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--prior-episode-exclusion-receipt",
        type=Path,
        required=True,
    )
    parser.add_argument("--train-pairs", type=int, default=2048)
    parser.add_argument("--development-pairs", type=int, default=256)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--staging-root",
        type=Path,
        default=DEFAULT_STAGING_ROOT,
        help="Existing local directory used for Lance commits before publish",
    )
    args = parser.parse_args(values)
    for label in (
        "output",
        "source",
        "prereg",
        "freeze_receipt",
        "prior_episode_exclusion_receipt",
        "staging_root",
    ):
        path = Path(getattr(args, label))
        forbidden = next(
            (
                component
                for component in path.parts
                if component.lower() in FORBIDDEN_PUBLIC_COMPONENTS
            ),
            None,
        )
        if forbidden is not None:
            raise ValueError(
                f"{label} contains forbidden Public component {forbidden!r}"
            )
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    pair_counts = _validate_frozen_pair_counts(
        {
            "train": int(args.train_pairs),
            "loader_validation": int(args.development_pairs),
        }
    )
    workers = _validate_frozen_workers(args.workers)
    jpeg_quality = _validate_frozen_jpeg_quality(args.jpeg_quality)
    output = _validate_formal_output(args.output)
    source = _validate_formal_input_file(args.source, label="source H5")
    prereg = _validate_formal_input_file(
        args.prereg,
        label="v4r1 preregistration",
        expected=DEFAULT_PREREG,
    )
    freeze_receipt = _validate_formal_input_file(
        args.freeze_receipt,
        label="v4r1 freeze receipt",
        expected=DEFAULT_FREEZE_RECEIPT,
    )
    staging_root = _validate_local_staging_root(
        args.staging_root,
        output=output,
    )
    prior_exclusion_receipt = _validate_formal_input_file(
        args.prior_episode_exclusion_receipt,
        label="v4r1 prior exclusion receipt",
        expected=DEFAULT_PRIOR_EXCLUSION_RECEIPT,
    )
    freeze_receipt_audit = validate_freeze_receipt(
        receipt_path=freeze_receipt,
        prereg_path=prereg,
        source_h5=source,
    )
    prior_exclusion_audit = validate_prior_episode_exclusion_receipt(
        receipt_path=prior_exclusion_receipt,
        prereg_path=prereg,
        freeze_receipt_path=freeze_receipt,
        freeze_receipt_audit=freeze_receipt_audit,
    )
    catalogs, source_receipt = build_candidate_catalogs(
        source,
        pair_counts=pair_counts,
        frozen_source_identity=freeze_receipt_audit["source_h5"],
        prior_exclusion_audit=prior_exclusion_audit,
    )
    request = _request_payload(
        pair_counts=pair_counts,
        source_receipt=source_receipt,
        jpeg_quality=jpeg_quality,
        workers=workers,
        output=output,
        freeze_receipt_audit=freeze_receipt_audit,
        prior_exclusion_audit=prior_exclusion_audit,
    )
    with tempfile.TemporaryDirectory(
        prefix="contextworld-cube-v4-",
        dir=staging_root,
    ) as temporary:
        temporary_metadata = os.lstat(temporary)
        if (
            not stat.S_ISDIR(temporary_metadata.st_mode)
            or int(temporary_metadata.st_dev) != int(os.stat("/tmp").st_dev)
        ):
            raise RuntimeError(
                "created v4r1 staging directory is not on the frozen local /tmp filesystem"
            )
        local_root = Path(temporary) / output.name
        local_root.mkdir()
        (local_root / "request.json").write_text(
            json.dumps(request, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        reports = {
            split: build_split(
                local_root,
                split=split,
                pair_count=pair_counts[split],
                candidates=catalogs[split],
                quality=jpeg_quality,
                workers=workers,
                prior_exclusion_audit=prior_exclusion_audit,
            )
            for split in ACTIVE_SPLITS
        }
        source_post_build_audit = verify_source_h5_unchanged_after_build(
            source,
            frozen_source_identity=freeze_receipt_audit["source_h5"],
        )
        local_lance = _validate_release_lance_tables(local_root, reports)
        cross_split = _cross_split_audit(reports)
        fresh_replay_build_summary = _fresh_replay_build_summary(reports)
        causal_contract = _audit_causal_contract_from_real_replay(
            reports,
            fresh_replay_build_summary,
        )
        build_report = {
            "protocol": PROTOCOL,
            "recovery_authorization_id": RECOVERY_AUTHORIZATION_ID,
            "evidence_scope": EVIDENCE_SCOPE,
            "profile_split_policy": PROFILE_SPLIT_POLICY,
            "active_splits": list(ACTIVE_SPLITS),
            "public_test_opened": False,
            "public_test_generated": False,
            "storage_publication_contract": {
                "lance_committed_and_reopened_on_local_staging": True,
                "staging_path_recorded": False,
                "local_lance_tables": local_lance,
                "final_publish_method": "verified_x_exclusive_copytree",
                "final_success_marker": SUCCESS_MARKER_NAME,
                "failed_publish_is_never_marked_complete": True,
                "nonempty_directory_rename_used": False,
            },
            "source_h5_post_build_integrity": source_post_build_audit,
            "prior_episode_exclusion_receipt": request[
                "prior_episode_exclusion_receipt"
            ],
            "reproducibility_contract": request["reproducibility_contract"],
            "action_profile_contract": request["action_profile_contract"],
            "content_identity_contract": request["content_identity_contract"],
            "fresh_simulator_replay_contract": request[
                "fresh_simulator_replay_contract"
            ],
            "request": request,
            "splits": reports,
            "cross_split_audit": cross_split,
            "fresh_simulator_replay": fresh_replay_build_summary,
            "causal_data_contract": causal_contract,
            "passed": all(row["passed"] for row in reports.values())
            and source_post_build_audit["passed"]
            and cross_split["passed"]
            and fresh_replay_build_summary["passed"]
            and causal_contract["passed"],
        }
        (local_root / "build_report.json").write_text(
            json.dumps(build_report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest = _manifest_payload(local_root, build_report=build_report)
        (local_root / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if not build_report["passed"]:
            raise SystemExit(1)
        publication = _publish_staged_release(
            staged_root=local_root,
            output=output,
            reports=reports,
        )
    print(
        json.dumps(
            {
                "output": str(output),
                "passed": build_report["passed"],
                "pair_counts": pair_counts,
                "cross_split_audit": cross_split,
                "public_test_opened": False,
                "public_test_generated": False,
                "tree_sha256": publication[
                    "tree_sha256_with_success_marker"
                ],
                "success_marker": {
                    key: publication[key]
                    for key in ("path", "sha256", "size_bytes")
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
