#!/usr/bin/env python3
"""Build the Development-only Cube History-3 v3 paired dataset.

The v3 builder intentionally has no Public Test split.  It materializes only
Training and loader Development data, balances four action-anchor families in
each split, and requires every exact float32 action profile to be disjoint
across the two active splits.  Split-neutral scene and scene/action content
hashes provide the isolation evidence; split-prefixed pair IDs do not.
"""

from __future__ import annotations

import argparse
import atexit
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass
import hashlib
from io import BytesIO
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
from typing import Any, Iterator

import h5py
import lance
import numpy as np
import pyarrow as pa
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

from contextworld.benchmarks.causal_data_contract import (  # noqa: E402
    audit_causal_data_contract,
)
from contextworld.evaluation.cube_grasp_rule_h3_v3 import (  # noqa: E402
    CAPABILITY_NAME,
    GRASP_MODES,
    QUERY_STATE_TOLERANCE,
    V3_ACTION_ANCHORS,
    V3_PROFILE_SPLIT_SEEDS,
    CubeGraspRuleCandidate,
    CubeGraspRuleV3Candidate,
    CubeGraspRuleV3Simulator,
    action_blocks as v3_action_blocks,
    make_v3_candidate,
)
from contextworld.paths import (  # noqa: E402
    artifact_path,
    portable_contextworld_path,
)


PROTOCOL = "cube_gripper_carry_rule_history3_development_v3"
EVIDENCE_SCOPE = "every accepted pair in Training and Development"
PROFILE_SPLIT_POLICY = "shared_families_disjoint_profiles"
ACTIVE_SPLITS = ("train", "loader_validation")
DEFAULT_PAIR_COUNTS = {"train": 2048, "loader_validation": 256}
DEFAULT_OUTPUT_LOGICAL = Path(
    "artifacts/synthesis/cube_gripper_carry_rule_h3_development_v3"
)
DEFAULT_OUTPUT = artifact_path(
    "synthesis/cube_gripper_carry_rule_h3_development_v3"
)
DEFAULT_PREREG = ROOT / (
    "configs/benchmark/cube_gripper_carry_h3_development_prereg_v3.yaml"
)
DEFAULT_FREEZE_RECEIPT_LOGICAL = Path(
    "artifacts/evaluation/history3/cube_gripper_carry_h3_development_v3/"
    "development_prereg_freeze_receipt_v2.json"
)
DEFAULT_FREEZE_RECEIPT = artifact_path(
    "evaluation/history3/cube_gripper_carry_h3_development_v3/"
    "development_prereg_freeze_receipt_v2.json"
)
V3_PHYSICS_PATH = ROOT / "contextworld/evaluation/cube_grasp_rule_h3_v3.py"
V3_BUILDER_PATH = Path(__file__).resolve()
SOURCE_SYMBOL = "upstream_cube_single_expert_h5"
CATALOG_SEEDS = {
    "train": 2026081101,
    "loader_validation": 2026081102,
}
CANDIDATE_ASSIGNMENT_SEED = 2026081100
CANDIDATE_POOL_MULTIPLIER = 2
ACTION_PROFILE_SHAPE = (4, 5, 5)
MINIMUM_EFFECT_GAP_M = 0.008
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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for child in sorted(value for value in path.rglob("*") if value.is_file()):
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


def validate_freeze_receipt(
    *,
    receipt_path: Path,
    prereg_path: Path,
    source_h5: Path,
    builder_path: Path = V3_BUILDER_PATH,
    physics_path: Path = V3_PHYSICS_PATH,
) -> dict[str, Any]:
    """Validate the immutable authorization before creating build output.

    The source H5 content digest is trusted only after verifying the receipt's
    own identity bindings.  The large H5 is not rehashed here; its current
    byte size and row count are checked locally against the frozen receipt.
    """

    for label, path in (
        ("freeze receipt", receipt_path),
        ("preregistration", prereg_path),
        ("source H5", source_h5),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} is missing: {path}")
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("Freeze receipt is not valid UTF-8 JSON") from error
    if not isinstance(receipt, Mapping):
        raise RuntimeError("Freeze receipt root must be an object")
    if receipt.get("schema_version") != 1:
        raise RuntimeError("Freeze receipt schema_version must be 1")
    if receipt.get("protocol_id") != PROTOCOL:
        raise RuntimeError("Freeze receipt protocol_id mismatch")
    if receipt.get("status") != "frozen_before_first_v3_data_build":
        raise RuntimeError("Freeze receipt status does not authorize a v3 build")
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
        builder_entry = identity["v3_builder"]
        physics_entry = identity["v3_physics"]
    except KeyError as error:
        raise RuntimeError(
            "Freeze receipt lacks v3_builder or v3_physics identity"
        ) from error
    if not isinstance(builder_entry, Mapping) or not isinstance(
        physics_entry, Mapping
    ):
        raise RuntimeError("Freeze receipt builder/physics identity is malformed")
    verified_builder = _verified_current_file_identity(
        receipt_entry=builder_entry,
        current_path=builder_path,
        label="v3_builder",
    )
    verified_physics = _verified_current_file_identity(
        receipt_entry=physics_entry,
        current_path=physics_path,
        label="v3_physics",
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

    return {
        "path": portable_contextworld_path(receipt_path),
        "sha256": file_sha256(receipt_path),
        "size_bytes": receipt_path.stat().st_size,
        "protocol_id": PROTOCOL,
        "status": receipt["status"],
        "checks_passed": True,
        "authorized_splits": list(ACTIVE_SPLITS),
        "public_test": dict(public),
        "preregistration": verified_prereg,
        "identity": {
            "v3_builder": verified_builder,
            "v3_physics": verified_physics,
        },
        "source_h5": {
            "symbol": SOURCE_SYMBOL,
            "sha256": source_sha256,
            "size_bytes": source_size,
            "row_count": source_rows,
            "episode_count": source_episodes,
            "content_hash_reused_from_validated_freeze_receipt": True,
            "content_rehashed_by_builder": False,
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


def _validate_pair_counts(pair_counts: Mapping[str, int]) -> dict[str, int]:
    """Validate the closed Development-only split universe."""

    observed = set(pair_counts)
    expected = set(ACTIVE_SPLITS)
    if observed != expected:
        extra = sorted(observed - expected)
        missing = sorted(expected - observed)
        raise ValueError(
            "v3 is Development-only and accepts exactly the active splits; "
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


def _anchor_ids() -> tuple[str, str, str, str]:
    values: Sequence[Any]
    if isinstance(V3_ACTION_ANCHORS, Mapping):
        values = tuple(V3_ACTION_ANCHORS)
    else:
        values = tuple(V3_ACTION_ANCHORS)
    ids: list[str] = []
    for value in values:
        if isinstance(value, str):
            identifier = value
        elif hasattr(value, "action_anchor_id"):
            identifier = str(value.action_anchor_id)
        elif isinstance(value, Sequence) and value:
            identifier = str(value[0])
        else:
            raise TypeError(f"Cannot resolve v3 action anchor ID from {value!r}")
        if not identifier:
            raise ValueError("v3 action anchor IDs must be non-empty")
        ids.append(identifier)
    if len(ids) != 4 or len(set(ids)) != 4:
        raise ValueError(f"v3 requires exactly four distinct anchors, got {ids}")
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
    digest.update(b"contextworld-cube-v3-scene-template-content-v1\0")
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
    candidate: CubeGraspRuleV3Candidate,
) -> tuple[str, str, np.ndarray, dict[str, Any]]:
    profile = candidate.action_profile
    receipt = _json_mapping(profile)
    blocks = np.asarray(v3_action_blocks(profile), dtype=np.float32)
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
        "source_content_rehashed_by_builder": False,
        "eligible_source_episode_count": int(eligible_episode_count),
        "eligible_row_selection_rule": ELIGIBLE_ROW_SELECTION_RULE,
    }


def build_candidate_catalogs(
    source: Path,
    *,
    pair_counts: Mapping[str, int],
    frozen_source_identity: Mapping[str, Any],
) -> tuple[dict[str, list[CubeGraspRuleV3Candidate]], dict[str, Any]]:
    counts = _validate_pair_counts(pair_counts)
    eligible = _eligible_source_rows(source)
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

    catalogs: dict[str, list[CubeGraspRuleV3Candidate]] = {}
    catalog_profile_ids: dict[str, set[str]] = {}
    catalog_scene_hashes: dict[str, set[str]] = {}
    catalog_pair_hashes: dict[str, set[str]] = {}
    catalog_anchor_counts: dict[str, dict[str, int]] = {}
    anchors = _anchor_ids()
    for split in ACTIVE_SPLITS:
        catalog: list[CubeGraspRuleV3Candidate] = []
        profile_ids: set[str] = set()
        scene_hashes: set[str] = set()
        pair_hashes: set[str] = set()
        anchor_counts = {anchor: 0 for anchor in anchors}
        for index, (source_row, source_episode, source_step) in enumerate(
            assignments[split]
        ):
            rng = np.random.default_rng(
                np.random.SeedSequence([CATALOG_SEEDS[split], index])
            )
            source_qpos, source_control = source_values[source_row]
            base_candidate = CubeGraspRuleCandidate(
                candidate_id=f"cube-carry-v3-{split}-{index:06d}",
                split=split,
                catalog_index=index,
                source_row=source_row,
                source_episode=source_episode,
                source_step=source_step,
                simulator_seed=int(rng.integers(0, 2**31 - 1)),
                task_id=1 + index % 5,
                qpos=tuple(float(value) for value in source_qpos),
                control=tuple(float(value) for value in source_control),
                cube_color=tuple(float(value) for value in rng.uniform(0.18, 0.92, 3)),
                target_position=(
                    float(rng.uniform(0.32, 0.53)),
                    float(rng.uniform(-0.24, 0.24)),
                    0.02,
                ),
            )
            candidate = make_v3_candidate(base_candidate)
            anchor, profile_id, _, _ = _profile_from_candidate(candidate)
            expected_anchor = anchors[index % len(anchors)]
            if anchor != expected_anchor:
                raise RuntimeError(
                    f"{candidate.candidate_id}: expected index%4 anchor "
                    f"{expected_anchor!r}, got {anchor!r}"
                )
            if profile_id in profile_ids:
                raise RuntimeError(
                    f"{candidate.candidate_id}: duplicate exact action profile "
                    f"inside {split}: {profile_id}"
                )
            scene_hash = scene_template_content_sha256(candidate)
            pair_hash = pair_content_sha256(scene_hash, profile_id)
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
            "v3 catalog split-disjointness failed before simulation: "
            f"source_episode_overlap={source_overlap}, "
            f"exact_action_profile_id_overlap={profile_overlap}, "
            f"scene_template_content_hash_overlap={scene_overlap}, "
            f"pair_content_hash_overlap={pair_overlap}"
        )
    receipt = {
        **_source_h5_receipt(
            source,
            eligible_episode_count=len(eligible),
            frozen_source_identity=frozen_source_identity,
        ),
        "candidate_pool_per_split": {
            split: len(catalogs[split]) for split in ACTIVE_SPLITS
        },
        "candidate_pool_multiplier": CANDIDATE_POOL_MULTIPLIER,
        "candidate_assignment_seed": CANDIDATE_ASSIGNMENT_SEED,
        "catalog_seeds": dict(CATALOG_SEEDS),
        "profile_split_seeds": {
            split: int(V3_PROFILE_SPLIT_SEEDS[split])
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


_WORKER_SIMULATOR: CubeGraspRuleV3Simulator | None = None
_WORKER_REPLAY_SIMULATOR: CubeGraspRuleV3Simulator | None = None
_WORKER_QUALITY = 95


def _worker_initialize(quality: int) -> None:
    global _WORKER_SIMULATOR, _WORKER_REPLAY_SIMULATOR, _WORKER_QUALITY
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
    _WORKER_QUALITY = int(quality)
    _WORKER_SIMULATOR = CubeGraspRuleV3Simulator()
    _WORKER_REPLAY_SIMULATOR = CubeGraspRuleV3Simulator()
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


def _build_candidate(candidate: CubeGraspRuleV3Candidate) -> dict[str, Any] | None:
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
        raise RuntimeError("paired v3 action blocks differ between hidden modes")
    calculated_profile_id = action_profile_content_sha256(low_blocks)
    v3_audit = _json_mapping(audit.get("v3", {}))
    candidate_profile = _json_mapping(candidate.get("action_profile", {}))
    claimed_profile_ids = {
        str(profile_receipt.get("action_profile_id", "")),
        str(candidate_profile.get("action_profile_id", "")),
        str(v3_audit.get("action_profile_id", "")),
        str(low.get("action_profile_id", "")),
        str(high.get("action_profile_id", "")),
    }
    if claimed_profile_ids != {calculated_profile_id}:
        raise RuntimeError(
            "v3 action_profile_id does not consistently equal the actual "
            f"float32 action-block hash: {claimed_profile_ids}, "
            f"calculated={calculated_profile_id}"
        )
    claimed_anchors = {
        str(profile_receipt.get("action_anchor_id", "")),
        str(candidate_profile.get("action_anchor_id", "")),
        str(v3_audit.get("action_anchor_id", "")),
        str(low.get("action_anchor_id", "")),
        str(high.get("action_anchor_id", "")),
    }
    if len(claimed_anchors) != 1:
        raise RuntimeError(f"inconsistent v3 action anchor IDs: {claimed_anchors}")
    anchor = next(iter(claimed_anchors))
    if anchor not in _anchor_ids():
        raise RuntimeError(f"unknown v3 action anchor: {anchor!r}")
    if candidate.get("split") != split or profile_receipt.get("split") != split:
        raise RuntimeError("v3 built result split mismatch")
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
            "v3 content-hash receipt does not match normalized scene and "
            "actual action content"
        )
    constraints = _json_mapping(v3_audit.get("profile_constraints", {}))
    fresh_replay = _json_mapping(v3_audit.get("fresh_simulator_replay", {}))
    if fresh_replay.get("passed") is not True:
        raise RuntimeError("v3 pair lacks a passed fresh-simulator replay audit")
    if fresh_replay.get("independent_simulator_instance") is not True:
        raise RuntimeError("v3 replay audit did not use an independent simulator")
    if fresh_replay.get("provided_reusable_instance") is not True:
        raise RuntimeError("v3 builder replay audit did not use its worker instance")
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
        "source": "audit.v3.fresh_simulator_replay for every accepted pair",
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
    candidates: list[CubeGraspRuleV3Candidate],
    quality: int,
    workers: int,
) -> dict[str, Any]:
    if split not in ACTIVE_SPLITS:
        raise ValueError(f"inactive split refused by v3 builder: {split!r}")
    _validate_pair_counts({split: pair_count, **{
        active: pair_count for active in ACTIVE_SPLITS if active != split
    }})
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
    files = [path for path in table_path.rglob("*") if path.is_file()]
    action_minimum = np.min(
        np.asarray([row["action_axis_minimum"] for row in accepted]), axis=0
    )
    action_maximum = np.max(
        np.asarray([row["action_axis_maximum"] for row in accepted]), axis=0
    )
    fresh_replay_summary = _fresh_replay_summary(accepted)
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
        "table_files": len(files),
        "table_bytes": sum(path.stat().st_size for path in files),
        "table_sha256": directory_sha256(table_path),
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
            "per-pair audit.v3.fresh_simulator_replay aggregated from "
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
) -> dict[str, Any]:
    counts = _validate_pair_counts(pair_counts)
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
    active_profile_seeds = {
        split: int(V3_PROFILE_SPLIT_SEEDS[split])
        for split in ACTIVE_SPLITS
    }
    return {
        "protocol": PROTOCOL,
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
                    "contextworld-cube-v3-scene-template-content-v1"
                ),
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
    files = [path for path in output.rglob("*") if path.is_file()]
    return {
        "protocol": PROTOCOL,
        "evidence_scope": EVIDENCE_SCOPE,
        "profile_split_policy": PROFILE_SPLIT_POLICY,
        "active_splits": list(ACTIVE_SPLITS),
        "public_test_opened": False,
        "public_test_generated": False,
        "files": {
            path.relative_to(output).as_posix(): file_sha256(path)
            for path in sorted(files)
        },
        "file_count_without_manifest": len(files),
        "bytes_without_manifest": sum(path.stat().st_size for path in files),
        "build_passed": bool(build_report["passed"]),
    }


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
        component_id="cube_gripper_carry_rule_v3_development",
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
        if any(
            value == option or value.startswith(f"{option}=")
            for option in forbidden_pair_options
        ):
            raise ValueError(
                "v3 builder explicitly refuses validation/Public Test pairs; "
                "only --train-pairs and --development-pairs are active"
            )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--prereg", type=Path, default=DEFAULT_PREREG)
    parser.add_argument(
        "--freeze-receipt",
        type=Path,
        default=DEFAULT_FREEZE_RECEIPT,
    )
    parser.add_argument("--train-pairs", type=int, default=2048)
    parser.add_argument("--development-pairs", type=int, default=256)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--workers", type=int, default=16)
    return parser.parse_args(values)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    pair_counts = _validate_pair_counts(
        {
            "train": int(args.train_pairs),
            "loader_validation": int(args.development_pairs),
        }
    )
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg-quality must be in [1, 100]")
    output = args.output.expanduser().resolve()
    source = args.source.expanduser().resolve()
    prereg = args.prereg.expanduser().resolve()
    freeze_receipt = args.freeze_receipt.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if not source.is_file():
        raise FileNotFoundError(source)
    freeze_receipt_audit = validate_freeze_receipt(
        receipt_path=freeze_receipt,
        prereg_path=prereg,
        source_h5=source,
    )
    output.mkdir(parents=True)
    catalogs, source_receipt = build_candidate_catalogs(
        source,
        pair_counts=pair_counts,
        frozen_source_identity=freeze_receipt_audit["source_h5"],
    )
    request = _request_payload(
        pair_counts=pair_counts,
        source_receipt=source_receipt,
        jpeg_quality=args.jpeg_quality,
        workers=args.workers,
        output=output,
        freeze_receipt_audit=freeze_receipt_audit,
    )
    (output / "request.json").write_text(
        json.dumps(request, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    reports = {
        split: build_split(
            output,
            split=split,
            pair_count=pair_counts[split],
            candidates=catalogs[split],
            quality=args.jpeg_quality,
            workers=args.workers,
        )
        for split in ACTIVE_SPLITS
    }
    cross_split = _cross_split_audit(reports)
    fresh_replay_build_summary = _fresh_replay_build_summary(reports)
    causal_contract = _audit_causal_contract_from_real_replay(
        reports,
        fresh_replay_build_summary,
    )
    build_report = {
        "protocol": PROTOCOL,
        "evidence_scope": EVIDENCE_SCOPE,
        "profile_split_policy": PROFILE_SPLIT_POLICY,
        "active_splits": list(ACTIVE_SPLITS),
        "public_test_opened": False,
        "public_test_generated": False,
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
        and cross_split["passed"]
        and fresh_replay_build_summary["passed"]
        and causal_contract["passed"],
    }
    (output / "build_report.json").write_text(
        json.dumps(build_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = _manifest_payload(output, build_report=build_report)
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
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
                "tree_sha256": directory_sha256(output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    if not build_report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
