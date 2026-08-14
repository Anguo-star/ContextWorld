#!/usr/bin/env python3
"""Recover raw query identities from the failed Cube v4 formal fragment.

The first Cube v4 formal build wrote a complete Training data fragment before
the NFS-backed Lance commit failed.  Lance's low-level file reader can recover
the rows without a dataset manifest, but the fragment stores JPEG bytes while
the benchmark's query identity is a hash of the pre-JPEG uint8 frame.  This
tool therefore:

* verifies the immutable failed-attempt receipt and its complete frozen chain;
* reads the orphan data file directly, without opening or writing a dataset;
* reconstructs each frozen candidate with snapshotted builder/physics code;
* runs only the ``cannot_hold`` trajectory and hashes its raw x2 frame;
* re-encodes that frame with the frozen JPEG recipe and requires byte equality
  with both stored modes; and
* writes one x-exclusive receipt suitable for a future prior-exclusion freeze.

It never calls ``lance.write_dataset`` and has no Public-Test, probe, model
training, or scoring path.
"""

from __future__ import annotations

import argparse
import atexit
from collections import defaultdict
from dataclasses import dataclass
import hashlib
from io import BytesIO
import importlib.util
import json
import multiprocessing as mp
import os
from pathlib import Path
import re
import sys
from typing import Any, Callable, Iterable, Mapping, Sequence

import h5py
import numpy as np
import pyarrow as pa
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROTOCOL = "cube_gripper_carry_rule_history3_development_v4"
FAILED_RECEIPT_ID = "cube_gripper_carry_h3_v4_failed_formal_attempt_v1"
FAILED_STATUS = "infrastructure_failed_immutable_attempt"
RECEIPT_ID = (
    "cube_gripper_carry_h3_v4_failed_attempt_query_reconstruction_v1"
)
RECEIPT_STATUS = "failed_attempt_content_frozen_for_future_prior_exclusion"
SOURCE_SYMBOL = "upstream_cube_single_expert_h5"
EXPECTED_SPLIT = "train"
EXPECTED_MODES = ("cannot_hold", "can_hold")
EXPECTED_MODEL_STEPS = (0, 1, 2, 3)
QUERY_MODEL_STEP = 2
DEFAULT_JPEG_QUALITY = 95
CONTENT_FIELDS = (
    "action_profile_ids",
    "scene_template_content_hashes",
    "pair_content_hashes",
    "query_pixel_hashes",
)
FRAGMENT_CONTENT_FIELDS = CONTENT_FIELDS[:3]
SHA256_RE = re.compile(r"[0-9a-f]{64}")
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
REQUIRED_FRAGMENT_COLUMNS = (
    "model_step_idx",
    "pixels",
    "action_block",
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
class FragmentPair:
    pair_id: str
    split: str
    catalog_index: int
    source_row: int
    source_episode: int
    source_step: int
    action_anchor_id: str
    action_profile_id: str
    scene_template_content_hash: str
    pair_content_hash: str
    query_jpeg_sha256: str
    query_jpeg: bytes

    def public_record(self) -> dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "split": self.split,
            "catalog_index": self.catalog_index,
            "source_row": self.source_row,
            "source_episode": self.source_episode,
            "source_step": self.source_step,
            "action_anchor_id": self.action_anchor_id,
            "action_profile_id": self.action_profile_id,
            "scene_template_content_hash": self.scene_template_content_hash,
            "pair_content_hash": self.pair_content_hash,
            "query_jpeg_sha256": self.query_jpeg_sha256,
        }


@dataclass(frozen=True)
class ReplayResult:
    pair_id: str
    raw_query_pixel_hash: str
    query_jpeg_sha256: str


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    """The exact raw-array namespace frozen by Cube History-3."""

    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def excluded_source_episodes_sha256(values: Sequence[int]) -> str:
    normalized = [int(value) for value in values]
    if normalized != sorted(set(normalized)) or any(value < 0 for value in normalized):
        raise ValueError("source episode IDs must be sorted, unique, and nonnegative")
    payload = b"".join(value.to_bytes(8, "little", signed=True) for value in normalized)
    return hashlib.sha256(
        b"contextworld-cube-prior-source-episodes-v1\0" + payload
    ).hexdigest()


def canonical_content_digest(values: Sequence[str], *, field_name: str) -> str:
    normalized = list(values)
    if normalized != sorted(set(normalized)):
        raise ValueError(f"{field_name} values must be sorted and unique")
    decoded: list[bytes] = []
    for value in normalized:
        if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
            raise ValueError(f"{field_name} contains a non-SHA256 value")
        decoded.append(bytes.fromhex(value))
    return hashlib.sha256(
        b"contextworld-cube-prior-content-exclusions-v1\0"
        + field_name.encode("ascii")
        + b"\0"
        + b"".join(decoded)
    ).hexdigest()


def forensic_query_jpeg_digest(values: Sequence[str]) -> str:
    """Hash JPEG bindings in a namespace that cannot be mistaken for raw RGB."""

    normalized = list(values)
    if normalized != sorted(set(normalized)):
        raise ValueError("query JPEG identities must be sorted and unique")
    decoded: list[bytes] = []
    for value in normalized:
        if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
            raise ValueError("query JPEG identity is not a SHA256")
        decoded.append(bytes.fromhex(value))
    return hashlib.sha256(
        FORENSIC_QUERY_JPEG_DIGEST_NAMESPACE.encode("ascii")
        + b"\0"
        + b"".join(decoded)
    ).hexdigest()


def _identity(path: Path, *, recorded_path: str | None = None) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"identity input is not a regular non-symlink file: {path}")
    return {
        "path": recorded_path if recorded_path is not None else path.as_posix(),
        "sha256": file_sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _read_json_nofollow(path: Path, *, label: str) -> tuple[bytes, dict[str, Any]]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"{label} must be a regular non-symlink file: {path}")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(descriptor, "rb") as stream:
        raw = stream.read()
    try:
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{label} is not valid UTF-8 JSON") from error
    if not isinstance(document, dict):
        raise RuntimeError(f"{label} root must be an object")
    return raw, document


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{label} must be an object")
    return value


def _false_public_gate(value: Any, *, label: str) -> None:
    gate = _mapping(value, label=label)
    if gate.get("access_status") != "closed_not_read_not_scored":
        raise RuntimeError(f"{label} access status is not closed")
    for key in ("opened", "read", "hashed", "scored"):
        if gate.get(key) is not False:
            raise RuntimeError(f"{label}.{key} is not false")


def reject_public_path(path: Path | str, *, label: str) -> None:
    component = next(
        (
            part
            for part in Path(path).parts
            if part.lower() in FORBIDDEN_PUBLIC_COMPONENTS
        ),
        None,
    )
    if component is not None:
        raise RuntimeError(
            f"{label} contains forbidden Public-Test component {component!r}"
        )


def _verify_file_identity(
    path: Path, declared: Mapping[str, Any], *, label: str
) -> dict[str, Any]:
    observed = _identity(path)
    if observed["sha256"] != declared.get("sha256") or observed[
        "size_bytes"
    ] != declared.get("size_bytes"):
        raise RuntimeError(f"{label} file identity mismatch")
    result = {
        "sha256": observed["sha256"],
        "size_bytes": observed["size_bytes"],
    }
    if isinstance(declared.get("path"), str):
        result["path"] = str(declared["path"])
    for key in (
        "symbol",
        "path_recorded",
        "row_count",
        "episode_count",
        "schema",
        "lance_file_major_version",
        "lance_file_minor_version",
    ):
        if key in declared:
            result[key] = declared[key]
    return result


def _validate_set_entry(
    entry: Any,
    *,
    field_name: str,
    expected_values: Sequence[str] | None = None,
) -> list[str]:
    value = _mapping(entry, label=field_name)
    values = [str(item) for item in value.get("values", [])]
    if not values or values != sorted(set(values)):
        raise RuntimeError(f"{field_name} values are not non-empty/sorted/unique")
    if int(value.get("count", -1)) != len(values):
        raise RuntimeError(f"{field_name} count mismatch")
    expected_digest = canonical_content_digest(values, field_name=field_name)
    if value.get("sha256") != expected_digest:
        raise RuntimeError(f"{field_name} digest mismatch")
    if expected_values is not None and values != list(expected_values):
        raise RuntimeError(f"{field_name} does not match the fragment")
    return values


def _validate_query_jpeg_entry(
    entry: Any, *, expected_values: Sequence[str]
) -> list[str]:
    value = _mapping(entry, label="query_jpeg_sha256")
    values = [str(item) for item in value.get("values", [])]
    if not values or values != sorted(set(values)):
        raise RuntimeError("query_jpeg_sha256 values are not sorted/unique")
    if int(value.get("count", -1)) != len(values):
        raise RuntimeError("query_jpeg_sha256 count mismatch")
    if value.get("digest_namespace") != FORENSIC_QUERY_JPEG_DIGEST_NAMESPACE:
        raise RuntimeError("query_jpeg_sha256 digest namespace mismatch")
    if value.get("role") != "forensic_binding_only_not_raw_query_pixel_hash":
        raise RuntimeError("query_jpeg_sha256 role could be confused with raw RGB")
    if value.get("sha256") != forensic_query_jpeg_digest(values):
        raise RuntimeError("query_jpeg_sha256 digest mismatch")
    if values != list(expected_values):
        raise RuntimeError("query_jpeg_sha256 values do not match the fragment")
    return values


def _validate_source_entry(
    entry: Any, *, expected_values: Sequence[int] | None = None
) -> list[int]:
    value = _mapping(entry, label="source_episodes")
    values = [int(item) for item in value.get("values", [])]
    if not values or values != sorted(set(values)) or any(item < 0 for item in values):
        raise RuntimeError("source_episodes values are invalid")
    if int(value.get("count", -1)) != len(values):
        raise RuntimeError("source_episodes count mismatch")
    if value.get("sha256") != excluded_source_episodes_sha256(values):
        raise RuntimeError("source_episodes digest mismatch")
    if expected_values is not None and values != list(expected_values):
        raise RuntimeError("source_episodes do not match the fragment")
    return values


def _content_entry(values: Iterable[str], *, field_name: str) -> dict[str, Any]:
    normalized = sorted(set(values))
    if not normalized:
        raise RuntimeError(f"cannot emit empty {field_name}")
    return {
        "values": normalized,
        "count": len(normalized),
        "sha256": canonical_content_digest(normalized, field_name=field_name),
    }


def _source_entry(values: Iterable[int]) -> dict[str, Any]:
    normalized = sorted(set(int(value) for value in values))
    if not normalized:
        raise RuntimeError("cannot emit an empty source-episode set")
    return {
        "values": normalized,
        "count": len(normalized),
        "sha256": excluded_source_episodes_sha256(normalized),
    }


def _table_from_fragment(fragment: Path) -> pa.Table:
    # Deliberately use the single-file API.  lance.dataset(fragment.parent)
    # would require the manifest that the failed NFS commit never installed.
    from lance.file import LanceFileReader

    return LanceFileReader(
        str(fragment), columns=list(REQUIRED_FRAGMENT_COLUMNS)
    ).read_all(batch_size=2048, batch_readahead=4).to_table()


def extract_fragment_pairs(table: pa.Table) -> tuple[list[FragmentPair], int]:
    missing = sorted(set(REQUIRED_FRAGMENT_COLUMNS) - set(table.column_names))
    if missing:
        raise RuntimeError(f"partial fragment lacks required columns: {missing}")
    rows = table.select(list(REQUIRED_FRAGMENT_COLUMNS)).to_pylist()
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["pair_id"])].append(row)
    if not grouped:
        raise RuntimeError("partial fragment contains no pairs")

    pairs: list[FragmentPair] = []
    for pair_id in sorted(grouped):
        group = grouped[pair_id]
        if len(group) != 8:
            raise RuntimeError(f"{pair_id}: expected 8 rows, found {len(group)}")
        common_fields = (
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
        common: dict[str, Any] = {}
        for field in common_fields:
            values = {row[field] for row in group}
            if len(values) != 1:
                raise RuntimeError(f"{pair_id}: inconsistent {field}")
            common[field] = next(iter(values))
        if common["split"] != EXPECTED_SPLIT:
            raise RuntimeError(f"{pair_id}: unexpected split {common['split']!r}")
        for field in (
            "action_profile_id",
            "scene_template_content_hash",
            "pair_content_hash",
        ):
            if SHA256_RE.fullmatch(str(common[field])) is None:
                raise RuntimeError(f"{pair_id}: invalid {field}")

        by_mode: dict[str, list[dict[str, Any]]] = {}
        for mode in EXPECTED_MODES:
            mode_rows = sorted(
                (row for row in group if row["hidden_mode"] == mode),
                key=lambda row: int(row["model_step_idx"]),
            )
            if [int(row["model_step_idx"]) for row in mode_rows] != list(
                EXPECTED_MODEL_STEPS
            ):
                raise RuntimeError(f"{pair_id}: malformed {mode} step sequence")
            by_mode[mode] = mode_rows
            blocks = np.asarray(
                [row["action_block"] for row in mode_rows], dtype=np.float32
            ).reshape(4, 5, 5)
            calculated = hashlib.sha256(
                np.ascontiguousarray(blocks).tobytes()
            ).hexdigest()
            if calculated != common["action_profile_id"]:
                raise RuntimeError(f"{pair_id}: action-profile bytes mismatch")
        low_blocks = np.asarray(
            [row["action_block"] for row in by_mode[EXPECTED_MODES[0]]],
            dtype=np.float32,
        )
        high_blocks = np.asarray(
            [row["action_block"] for row in by_mode[EXPECTED_MODES[1]]],
            dtype=np.float32,
        )
        if not np.array_equal(low_blocks, high_blocks):
            raise RuntimeError(f"{pair_id}: paired action blocks differ")
        calculated_pair = hashlib.sha256(
            bytes.fromhex(str(common["scene_template_content_hash"]))
            + bytes.fromhex(str(common["action_profile_id"]))
        ).hexdigest()
        if calculated_pair != common["pair_content_hash"]:
            raise RuntimeError(f"{pair_id}: pair-content hash mismatch")

        query_bytes = [
            bytes(by_mode[mode][QUERY_MODEL_STEP]["pixels"])
            for mode in EXPECTED_MODES
        ]
        if query_bytes[0] != query_bytes[1]:
            raise RuntimeError(f"{pair_id}: stored paired query JPEGs differ")
        jpeg_sha = hashlib.sha256(query_bytes[0]).hexdigest()
        pairs.append(
            FragmentPair(
                pair_id=pair_id,
                split=str(common["split"]),
                catalog_index=int(common["catalog_index"]),
                source_row=int(common["source_row"]),
                source_episode=int(common["source_episode"]),
                source_step=int(common["source_step"]),
                action_anchor_id=str(common["action_anchor_id"]),
                action_profile_id=str(common["action_profile_id"]),
                scene_template_content_hash=str(
                    common["scene_template_content_hash"]
                ),
                pair_content_hash=str(common["pair_content_hash"]),
                query_jpeg_sha256=jpeg_sha,
                query_jpeg=query_bytes[0],
            )
        )

    pair_ids = [pair.pair_id for pair in pairs]
    if pair_ids != sorted(set(pair_ids)):
        raise RuntimeError("fragment pair IDs are not sorted and unique")
    for field in (
        "source_episode",
        "action_profile_id",
        "scene_template_content_hash",
        "pair_content_hash",
        "query_jpeg_sha256",
    ):
        if len({getattr(pair, field) for pair in pairs}) != len(pairs):
            raise RuntimeError(f"fragment has duplicate {field}")
    return pairs, table.num_rows


def _fragment_sets(pairs: Sequence[FragmentPair]) -> dict[str, list[Any]]:
    return {
        "source_episodes": sorted({pair.source_episode for pair in pairs}),
        "action_profile_ids": sorted({pair.action_profile_id for pair in pairs}),
        "scene_template_content_hashes": sorted(
            {pair.scene_template_content_hash for pair in pairs}
        ),
        "pair_content_hashes": sorted({pair.pair_content_hash for pair in pairs}),
        "query_jpeg_sha256": sorted({pair.query_jpeg_sha256 for pair in pairs}),
    }


def validate_failed_attempt_content(
    receipt: Mapping[str, Any],
    *,
    pairs: Sequence[FragmentPair],
    row_count: int,
) -> None:
    if receipt.get("schema_version") != 1:
        raise RuntimeError("failed-attempt receipt schema mismatch")
    if receipt.get("protocol_id") != PROTOCOL:
        raise RuntimeError("failed-attempt receipt protocol mismatch")
    if receipt.get("receipt_id") != FAILED_RECEIPT_ID:
        raise RuntimeError("failed-attempt receipt ID mismatch")
    if receipt.get("status") != FAILED_STATUS or receipt.get("checks_passed") is not True:
        raise RuntimeError("failed-attempt receipt does not record a passed freeze")
    if receipt.get("build_passed") is not False:
        raise RuntimeError("failed-attempt receipt does not record build failure")
    if receipt.get("formal_build_attempt_consumed") is not True or receipt.get(
        "retry_authorized_under_original_preregistration"
    ) is not False:
        raise RuntimeError("failed-attempt retry/attempt-budget contract mismatch")
    scope = _mapping(receipt.get("scope"), label="failed-attempt scope")
    _false_public_gate(scope.get("public_test"), label="failed-attempt public_test")
    if scope.get("rgb_probe_run") is not False:
        raise RuntimeError("failed-attempt receipt reports an RGB probe")
    if scope.get("reference_model_training_or_scoring") is not False or int(
        scope.get("optimizer_steps", -1)
    ) != 0:
        raise RuntimeError("failed-attempt receipt reports reference-model work")
    content = _mapping(
        receipt.get("failed_attempt_content"), label="failed_attempt_content"
    )
    if content.get("split") != EXPECTED_SPLIT:
        raise RuntimeError("failed-attempt split mismatch")
    if int(content.get("row_count", -1)) != row_count:
        raise RuntimeError("failed-attempt row count mismatch")
    if int(content.get("pair_count", -1)) != len(pairs):
        raise RuntimeError("failed-attempt pair count mismatch")
    if int(content.get("episode_count", -1)) != 2 * len(pairs):
        raise RuntimeError("failed-attempt episode count mismatch")

    observed = _fragment_sets(pairs)
    _validate_source_entry(
        content.get("source_episodes"), expected_values=observed["source_episodes"]
    )
    prior_content = _mapping(
        content.get("prior_content_exclusions"),
        label="failed_attempt_content.prior_content_exclusions",
    )
    if set(prior_content) != set(FRAGMENT_CONTENT_FIELDS):
        raise RuntimeError("failed-attempt receipt has unexpected fragment content sets")
    for field in FRAGMENT_CONTENT_FIELDS:
        _validate_set_entry(
            prior_content.get(field),
            field_name=field,
            expected_values=observed[field],
        )
    if content.get("query_pixel_hash_status") != (
        "pending_deterministic_raw_reconstruction_not_present_in_fragment"
    ):
        raise RuntimeError("failed-attempt raw-query status mismatch")
    _validate_query_jpeg_entry(
        content.get("query_jpeg_sha256"),
        expected_values=observed["query_jpeg_sha256"],
    )
    declared_pairs = content.get("pairs")
    observed_pairs = [
        {
            key: value
            for key, value in pair.public_record().items()
            if key != "split"
        }
        for pair in pairs
    ]
    if not isinstance(declared_pairs, list) or declared_pairs != observed_pairs:
        raise RuntimeError("failed-attempt pair records do not match the fragment")
    constraints = _mapping(
        content.get("profile_constraints"), label="failed profile constraints"
    )
    if constraints.get("passed") is not True:
        raise RuntimeError("failed-attempt action-profile constraints did not pass")
    overlap = _mapping(content.get("prior_overlap"), label="failed prior overlap")
    for field in (
        "source_episode_count",
        "action_profile_id_count",
        "scene_template_content_hash_count",
        "pair_content_hash_count",
    ):
        if int(overlap.get(field, -1)) != 0:
            raise RuntimeError(f"failed-attempt receipt reports prior {field}")
    if overlap.get("query_pixel_hash_count", "missing") is not None or overlap.get(
        "query_pixel_hash_overlap_status"
    ) != "not_computable_until_raw_query_reconstruction":
        raise RuntimeError("failed-attempt receipt prematurely claims raw-query overlap")
    if overlap.get("passed_for_directly_inspectable_identities") is not True:
        raise RuntimeError("failed-attempt directly inspectable overlap gate failed")


def _load_prior_sets(
    prior: Mapping[str, Any],
) -> tuple[set[int], dict[str, set[str]]]:
    if prior.get("schema_version") != 1 or prior.get("protocol_id") != PROTOCOL:
        raise RuntimeError("prior-exclusion receipt identity mismatch")
    if prior.get("status") != "frozen_before_first_v4_build":
        raise RuntimeError("prior-exclusion receipt status mismatch")
    if prior.get("checks_passed") is not True:
        raise RuntimeError("prior-exclusion receipt did not pass")
    _false_public_gate(prior.get("public_test"), label="prior public_test")
    if prior.get("reference_model_training_or_scoring") is not False:
        raise RuntimeError("prior-exclusion receipt reports reference-model work")
    content = _mapping(
        prior.get("prior_content_exclusions"), label="prior content exclusions"
    )
    if set(content) != set(CONTENT_FIELDS):
        raise RuntimeError("prior-exclusion receipt content fields mismatch")
    content_sets = {
        field: set(_validate_set_entry(content.get(field), field_name=field))
        for field in CONTENT_FIELDS
    }
    episode_values = [
        int(value) for value in prior.get("excluded_source_episodes", [])
    ]
    if not episode_values or episode_values != sorted(set(episode_values)):
        raise RuntimeError("prior-exclusion source episodes are invalid")
    if int(prior.get("excluded_source_episode_count", -1)) != len(
        episode_values
    ) or prior.get("excluded_source_episodes_sha256") != (
        excluded_source_episodes_sha256(episode_values)
    ):
        raise RuntimeError("prior-exclusion source episode count/digest mismatch")
    return set(episode_values), content_sets


def _load_prior_queries(prior: Mapping[str, Any]) -> set[str]:
    return _load_prior_sets(prior)[1]["query_pixel_hashes"]


def _validate_frozen_chain(
    *,
    failed_receipt: Mapping[str, Any],
    prereg_path: Path,
    freeze_receipt_path: Path,
    prior_path: Path,
    builder_snapshot: Path,
    physics_snapshot: Path,
    request_path: Path,
    fragment_path: Path,
    source_h5: Path,
) -> dict[str, dict[str, Any]]:
    identities = _mapping(
        failed_receipt.get("input_identities"), label="failed input identities"
    )
    required = {
        "preregistration": prereg_path,
        "freeze_receipt": freeze_receipt_path,
        "prior_exclusion_receipt": prior_path,
        "builder_snapshot": builder_snapshot,
        "request_json": request_path,
        "partial_train_fragment": fragment_path,
        "source_h5": source_h5,
    }
    missing = sorted(set(required) - set(identities))
    if missing:
        raise RuntimeError(f"failed-attempt receipt lacks input identities: {missing}")
    observed = {
        name: _verify_file_identity(
            path,
            _mapping(identities[name], label=f"input_identities.{name}"),
            label=name,
        )
        for name, path in required.items()
    }

    freeze_raw, freeze = _read_json_nofollow(
        freeze_receipt_path, label="v4 prereg freeze receipt"
    )
    prereg_identity = _identity(prereg_path)
    if freeze.get("status") != "frozen_before_first_v4_build" or freeze.get(
        "checks_passed"
    ) is not True:
        raise RuntimeError("v4 freeze receipt is not valid")
    _false_public_gate(freeze.get("public_test"), label="freeze public_test")
    if freeze.get("reference_model_training_or_scoring_authorized") is not False:
        raise RuntimeError("v4 freeze receipt authorizes reference models")
    if {
        key: _mapping(freeze.get("preregistration"), label="freeze preregistration").get(key)
        for key in ("sha256", "size_bytes")
    } != {
        "sha256": prereg_identity["sha256"],
        "size_bytes": prereg_identity["size_bytes"],
    }:
        raise RuntimeError("freeze/preregistration binding mismatch")

    freeze_identity = _mapping(freeze.get("identity"), label="freeze identity")
    builder_declared = _mapping(
        freeze_identity.get("v4_builder"), label="freeze identity.v4_builder"
    )
    physics_declared = _mapping(
        freeze_identity.get("v4_physics"), label="freeze identity.v4_physics"
    )
    observed["physics_snapshot"] = _verify_file_identity(
        physics_snapshot, physics_declared, label="physics snapshot"
    )
    observed["physics_snapshot"]["path"] = physics_snapshot.as_posix()
    if observed["builder_snapshot"]["sha256"] != builder_declared.get(
        "sha256"
    ) or observed["builder_snapshot"]["size_bytes"] != builder_declared.get(
        "size_bytes"
    ):
        raise RuntimeError("builder snapshot differs from frozen builder")
    for name in ("base_v2_physics", "v3_physics_dependency"):
        declared = _mapping(freeze_identity.get(name), label=f"freeze identity.{name}")
        dependency = ROOT / str(declared.get("path", ""))
        _verify_file_identity(dependency, declared, label=name)

    _, prior = _read_json_nofollow(prior_path, label="prior-exclusion receipt")
    _load_prior_sets(prior)
    prior_prereg = _mapping(prior.get("preregistration"), label="prior preregistration")
    prior_freeze = _mapping(prior.get("freeze_receipt"), label="prior freeze receipt")
    if prior_prereg.get("sha256") != prereg_identity["sha256"]:
        raise RuntimeError("prior/preregistration binding mismatch")
    freeze_digest = hashlib.sha256(freeze_raw).hexdigest()
    if prior_freeze.get("sha256") != freeze_digest:
        raise RuntimeError("prior/freeze binding mismatch")

    _, request = _read_json_nofollow(request_path, label="failed build request")
    if request.get("protocol") != PROTOCOL:
        raise RuntimeError("failed build request protocol mismatch")
    if request.get("public_test_opened") is not False or request.get(
        "public_test_generated"
    ) is not False:
        raise RuntimeError("failed build request did not keep Public closed")
    request_freeze = _mapping(request.get("freeze_receipt"), label="request freeze")
    request_prior = _mapping(
        request.get("prior_episode_exclusion_receipt"), label="request prior"
    )
    if request_freeze.get("sha256") != freeze_digest:
        raise RuntimeError("request/freeze binding mismatch")
    if request_prior.get("sha256") != observed["prior_exclusion_receipt"]["sha256"]:
        raise RuntimeError("request/prior binding mismatch")
    source = _mapping(freeze.get("source_h5"), label="freeze source_h5")
    if source.get("symbol") != SOURCE_SYMBOL or source.get("path_recorded") is not False:
        raise RuntimeError("frozen source symbol/path contract mismatch")
    if source.get("sha256") != observed["source_h5"]["sha256"] or source.get(
        "size_bytes"
    ) != observed["source_h5"]["size_bytes"]:
        raise RuntimeError("source H5 differs from frozen source")
    return observed


_WORKER_BUILDER: Any = None
_WORKER_PHYSICS: Any = None
_WORKER_SOURCE: h5py.File | None = None
_WORKER_SIMULATOR: Any = None
_WORKER_JPEG_QUALITY = DEFAULT_JPEG_QUALITY


def _load_module(path: Path, *, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load frozen module {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_frozen_modules(builder_snapshot: Path, physics_snapshot: Path) -> tuple[Any, Any]:
    canonical_physics_name = "contextworld.evaluation.cube_grasp_rule_h3_v4"
    physics = _load_module(physics_snapshot, name=canonical_physics_name)
    builder = _load_module(
        builder_snapshot,
        name="_contextworld_cube_v4_failed_attempt_builder_snapshot",
    )
    if builder.CubeGraspRuleV4Simulator is not physics.CubeGraspRuleV4Simulator:
        raise RuntimeError("builder snapshot did not bind the supplied physics snapshot")
    return builder, physics


def _worker_close() -> None:
    global _WORKER_SIMULATOR, _WORKER_SOURCE
    if _WORKER_SIMULATOR is not None:
        _WORKER_SIMULATOR.close()
        _WORKER_SIMULATOR = None
    if _WORKER_SOURCE is not None:
        _WORKER_SOURCE.close()
        _WORKER_SOURCE = None


def _worker_initialize(
    builder_snapshot: str,
    physics_snapshot: str,
    source_h5: str,
    jpeg_quality: int,
) -> None:
    global _WORKER_BUILDER, _WORKER_PHYSICS, _WORKER_SOURCE
    global _WORKER_SIMULATOR, _WORKER_JPEG_QUALITY
    _WORKER_BUILDER, _WORKER_PHYSICS = _load_frozen_modules(
        Path(builder_snapshot), Path(physics_snapshot)
    )
    _WORKER_SOURCE = h5py.File(source_h5, "r", swmr=True)
    if "qpos" not in _WORKER_SOURCE or "control" not in _WORKER_SOURCE:
        raise RuntimeError("source H5 lacks qpos/control")
    _WORKER_SIMULATOR = _WORKER_PHYSICS.CubeGraspRuleV4Simulator(resolution=224)
    _WORKER_JPEG_QUALITY = int(jpeg_quality)
    atexit.register(_worker_close)


def _jpeg_bytes(value: np.ndarray, *, quality: int) -> bytes:
    stream = BytesIO()
    Image.fromarray(np.asarray(value, dtype=np.uint8)).save(
        stream, format="JPEG", quality=quality
    )
    return stream.getvalue()


def _replay_one_worker(pair: FragmentPair) -> ReplayResult:
    if any(
        value is None
        for value in (
            _WORKER_BUILDER,
            _WORKER_PHYSICS,
            _WORKER_SOURCE,
            _WORKER_SIMULATOR,
        )
    ):
        raise RuntimeError("reconstruction worker is not initialized")
    builder = _WORKER_BUILDER
    physics = _WORKER_PHYSICS
    local_index = int(pair.catalog_index) - int(
        physics.V4_FORMAL_CATALOG_INDEX_OFFSET
    )
    if local_index < 0:
        raise RuntimeError(f"{pair.pair_id}: catalog index is outside formal namespace")
    expected_pair_id = f"cube-carry-v4-{pair.split}-{local_index:06d}"
    if pair.pair_id != expected_pair_id:
        raise RuntimeError(f"{pair.pair_id}: candidate ID/catalog index mismatch")
    rng = np.random.default_rng(
        np.random.SeedSequence([builder.CATALOG_SEEDS[pair.split], local_index])
    )
    source_qpos = np.asarray(_WORKER_SOURCE["qpos"][pair.source_row], dtype=np.float64)
    source_control = np.asarray(
        _WORKER_SOURCE["control"][pair.source_row], dtype=np.float64
    )
    base_candidate = physics.CubeGraspRuleCandidate(
        candidate_id=pair.pair_id,
        split=pair.split,
        catalog_index=pair.catalog_index,
        source_row=pair.source_row,
        source_episode=pair.source_episode,
        source_step=pair.source_step,
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
    candidate = physics.make_v4_candidate(base_candidate)
    profile = candidate.action_profile
    scene_hash = builder.scene_template_content_sha256(candidate)
    pair_hash = builder.pair_content_sha256(
        scene_hash, profile.action_profile_id
    )
    if (
        profile.action_anchor_id != pair.action_anchor_id
        or profile.action_profile_id != pair.action_profile_id
        or scene_hash != pair.scene_template_content_hash
        or pair_hash != pair.pair_content_hash
    ):
        raise RuntimeError(f"{pair.pair_id}: frozen candidate identity mismatch")
    blocks = physics.action_blocks(profile)
    payload = _WORKER_SIMULATOR._run_mode(
        candidate, mode="cannot_hold", blocks=blocks
    )
    raw_query = np.asarray(payload["pixels"])[QUERY_MODEL_STEP]
    raw_hash = array_sha256(raw_query)
    encoded = _jpeg_bytes(raw_query, quality=_WORKER_JPEG_QUALITY)
    encoded_hash = hashlib.sha256(encoded).hexdigest()
    if encoded != pair.query_jpeg or encoded_hash != pair.query_jpeg_sha256:
        raise RuntimeError(f"{pair.pair_id}: deterministic query JPEG mismatch")
    return ReplayResult(
        pair_id=pair.pair_id,
        raw_query_pixel_hash=raw_hash,
        query_jpeg_sha256=encoded_hash,
    )


def reconstruct_query_hashes(
    pairs: Sequence[FragmentPair],
    *,
    prior_query_hashes: set[str],
    replay: Callable[[FragmentPair], ReplayResult],
) -> tuple[list[ReplayResult], dict[str, Any]]:
    results = [replay(pair) for pair in pairs]
    if [result.pair_id for result in results] != [pair.pair_id for pair in pairs]:
        raise RuntimeError("reconstruction result order/pair identity mismatch")
    for pair, result in zip(pairs, results):
        if result.query_jpeg_sha256 != pair.query_jpeg_sha256:
            raise RuntimeError(f"{pair.pair_id}: replay JPEG identity mismatch")
        if SHA256_RE.fullmatch(result.raw_query_pixel_hash) is None:
            raise RuntimeError(f"{pair.pair_id}: invalid raw query hash")
    raw_hashes = [result.raw_query_pixel_hash for result in results]
    if len(set(raw_hashes)) != len(raw_hashes):
        raise RuntimeError("raw query-pixel hash collision/duplicate detected")
    overlap = sorted(set(raw_hashes) & prior_query_hashes)
    if overlap:
        raise RuntimeError(
            f"failed-attempt raw query hashes overlap prior exclusions: {overlap}"
        )
    return results, {
        "raw_query_hashes_unique": True,
        "raw_query_prior_overlap_zero": True,
        "stored_paired_query_jpegs_equal": True,
        "reencoded_query_jpegs_match_fragment": True,
        "replayed_mode": "cannot_hold_only",
        "query_model_step_idx": QUERY_MODEL_STEP,
    }


def _parallel_replay(
    pairs: Sequence[FragmentPair],
    *,
    builder_snapshot: Path,
    physics_snapshot: Path,
    source_h5: Path,
    jpeg_quality: int,
    workers: int,
) -> list[ReplayResult]:
    context = mp.get_context("spawn")
    with context.Pool(
        processes=workers,
        initializer=_worker_initialize,
        initargs=(
            str(builder_snapshot),
            str(physics_snapshot),
            str(source_h5),
            jpeg_quality,
        ),
    ) as pool:
        return list(pool.imap(_replay_one_worker, pairs, chunksize=1))


def _identity_cores(
    values: Mapping[str, Mapping[str, Any]]
) -> dict[str, tuple[str, int]]:
    return {
        name: (str(value.get("sha256", "")), int(value.get("size_bytes", -1)))
        for name, value in values.items()
    }


def write_receipt_exclusive(path: Path, receipt: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write("\n")


def reconstruct(
    *,
    failed_attempt_receipt: Path,
    fragment: Path,
    builder_snapshot: Path,
    physics_snapshot: Path,
    source_h5: Path,
    prereg: Path,
    freeze_receipt: Path,
    prior_exclusion_receipt: Path,
    request_json: Path,
    output: Path,
    workers: int,
    jpeg_quality: int,
) -> dict[str, Any]:
    if workers <= 0:
        raise ValueError("workers must be positive")
    if jpeg_quality != DEFAULT_JPEG_QUALITY:
        raise ValueError("failed attempt used the frozen JPEG quality 95")
    inputs = (
        failed_attempt_receipt,
        fragment,
        builder_snapshot,
        physics_snapshot,
        source_h5,
        prereg,
        freeze_receipt,
        prior_exclusion_receipt,
        request_json,
        output,
    )
    for path in inputs:
        reject_public_path(path, label="reconstruction path")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite {output}")

    failed_raw, failed = _read_json_nofollow(
        failed_attempt_receipt, label="failed-attempt receipt"
    )
    observed_inputs = _validate_frozen_chain(
        failed_receipt=failed,
        prereg_path=prereg,
        freeze_receipt_path=freeze_receipt,
        prior_path=prior_exclusion_receipt,
        builder_snapshot=builder_snapshot,
        physics_snapshot=physics_snapshot,
        request_path=request_json,
        fragment_path=fragment,
        source_h5=source_h5,
    )
    table = _table_from_fragment(fragment)
    pairs, row_count = extract_fragment_pairs(table)
    validate_failed_attempt_content(failed, pairs=pairs, row_count=row_count)
    _, prior = _read_json_nofollow(
        prior_exclusion_receipt, label="prior-exclusion receipt"
    )
    prior_episodes, prior_content = _load_prior_sets(prior)
    fragment_sets = _fragment_sets(pairs)
    directly_inspectable_overlap = {
        "source_episode": sorted(
            set(fragment_sets["source_episodes"]) & prior_episodes
        ),
        **{
            field: sorted(set(fragment_sets[field]) & prior_content[field])
            for field in FRAGMENT_CONTENT_FIELDS
        },
    }
    if any(directly_inspectable_overlap.values()):
        raise RuntimeError(
            "failed fragment overlaps frozen prior exclusions: "
            f"{directly_inspectable_overlap}"
        )
    replayed = _parallel_replay(
        pairs,
        builder_snapshot=builder_snapshot,
        physics_snapshot=physics_snapshot,
        source_h5=source_h5,
        jpeg_quality=jpeg_quality,
        workers=workers,
    )
    # Apply all collision/overlap/JPEG result gates in the main process too.
    by_pair = {value.pair_id: value for value in replayed}
    replayed, reconstruction_checks = reconstruct_query_hashes(
        pairs,
        prior_query_hashes=prior_content["query_pixel_hashes"],
        replay=lambda pair: by_pair[pair.pair_id],
    )
    # Re-verify every input after the worker pool has closed.  This catches a
    # source, snapshot, fragment, or authorization file changing between its
    # preflight hash and the end of the deterministic replay.
    postflight_inputs = _validate_frozen_chain(
        failed_receipt=failed,
        prereg_path=prereg,
        freeze_receipt_path=freeze_receipt,
        prior_path=prior_exclusion_receipt,
        builder_snapshot=builder_snapshot,
        physics_snapshot=physics_snapshot,
        request_path=request_json,
        fragment_path=fragment,
        source_h5=source_h5,
    )
    post_failed_raw, post_failed = _read_json_nofollow(
        failed_attempt_receipt, label="postflight failed-attempt receipt"
    )
    if post_failed != failed or post_failed_raw != failed_raw:
        raise RuntimeError("failed-attempt receipt mutated during reconstruction")
    if _identity_cores(postflight_inputs) != _identity_cores(observed_inputs):
        raise RuntimeError("a frozen input mutated during reconstruction")
    raw_query_by_pair = {
        value.pair_id: value.raw_query_pixel_hash for value in replayed
    }
    content = {
        field: _content_entry(fragment_sets[field], field_name=field)
        for field in FRAGMENT_CONTENT_FIELDS
    }
    content["query_pixel_hashes"] = _content_entry(
        raw_query_by_pair.values(), field_name="query_pixel_hashes"
    )
    source_entry = _source_entry(pair.source_episode for pair in pairs)
    failed_identity = {
        "path": failed_attempt_receipt.as_posix(),
        "sha256": hashlib.sha256(failed_raw).hexdigest(),
        "size_bytes": len(failed_raw),
    }
    receipt = {
        "schema_version": 1,
        "protocol_id": PROTOCOL,
        "receipt_id": RECEIPT_ID,
        "status": RECEIPT_STATUS,
        "checks_passed": True,
        "failed_attempt_receipt": failed_identity,
        "input_identities": {
            **observed_inputs,
            "failed_attempt_receipt": failed_identity,
        },
        "reconstruction_contract": {
            "fragment_read_api": "lance.file.LanceFileReader_single_file",
            "dataset_manifest_opened": False,
            "lance_written": False,
            "replayed_mode": "cannot_hold_only",
            "raw_query_frame": "pixels[2]_before_JPEG",
            "raw_query_hash": "Cube_array_sha256_dtype_shape_bytes",
            "jpeg_quality": jpeg_quality,
            "jpeg_reencoding_bitwise_equal_to_fragment": True,
            "builder_snapshot_loaded_by_explicit_path": True,
            "physics_snapshot_loaded_by_explicit_path": True,
            "all_inputs_reverified_unchanged_after_replay": True,
            "workers": workers,
            **reconstruction_checks,
        },
        "failed_attempt_content": {
            "split": EXPECTED_SPLIT,
            "row_count": row_count,
            "episode_count": 2 * len(pairs),
            "pair_count": len(pairs),
            "source_episodes": source_entry,
            "prior_content_exclusions": content,
            "pairs": [
                {
                    **pair.public_record(),
                    "raw_query_pixel_hash": raw_query_by_pair[pair.pair_id],
                }
                for pair in pairs
            ],
        },
        "prior_overlap": {
            "source_episode": {"count": 0, "values": []},
            "action_profile_ids": {"count": 0, "values": []},
            "scene_template_content_hashes": {"count": 0, "values": []},
            "pair_content_hashes": {"count": 0, "values": []},
            "query_pixel_hashes": {"count": 0, "values": []},
            "passed": True,
        },
        "public_test": {
            "access_status": "closed_not_read_not_scored",
            "opened": False,
            "read": False,
            "hashed": False,
            "scored": False,
        },
        "rgb_probe": {
            "opened": False,
            "run": False,
            "scored": False,
        },
        "reference_model_training_or_scoring": False,
        "reference_model_optimizer_steps": 0,
    }
    write_receipt_exclusive(output, receipt)
    return receipt


def parse_args(values: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--failed-attempt-receipt", type=Path, required=True)
    parser.add_argument("--fragment", type=Path, required=True)
    parser.add_argument("--builder-snapshot", type=Path, required=True)
    parser.add_argument("--physics-snapshot", type=Path, required=True)
    parser.add_argument("--source-h5", type=Path, required=True)
    parser.add_argument("--prereg", type=Path, required=True)
    parser.add_argument("--freeze-receipt", type=Path, required=True)
    parser.add_argument("--prior-exclusion-receipt", type=Path, required=True)
    parser.add_argument("--request-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--jpeg-quality", type=int, default=DEFAULT_JPEG_QUALITY)
    return parser.parse_args(values)


def main(values: Sequence[str] | None = None) -> None:
    args = parse_args(values)
    receipt = reconstruct(
        failed_attempt_receipt=args.failed_attempt_receipt.expanduser().resolve(),
        fragment=args.fragment.expanduser().resolve(),
        builder_snapshot=args.builder_snapshot.expanduser().resolve(),
        physics_snapshot=args.physics_snapshot.expanduser().resolve(),
        source_h5=args.source_h5.expanduser().resolve(),
        prereg=args.prereg.expanduser().resolve(),
        freeze_receipt=args.freeze_receipt.expanduser().resolve(),
        prior_exclusion_receipt=args.prior_exclusion_receipt.expanduser().resolve(),
        request_json=args.request_json.expanduser().resolve(),
        output=args.output.expanduser().resolve(),
        workers=args.workers,
        jpeg_quality=args.jpeg_quality,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "checks_passed": receipt["checks_passed"],
                "pair_count": receipt["failed_attempt_content"]["pair_count"],
                "public_test_read": receipt["public_test"]["read"],
                "lance_written": receipt["reconstruction_contract"]["lance_written"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
