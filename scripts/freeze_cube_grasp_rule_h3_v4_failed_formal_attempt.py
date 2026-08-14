#!/usr/bin/env python3
"""Freeze the immutable Cube v4 formal-build infrastructure failure.

The first and only build authorized by the original v4 preregistration wrote
one complete train data fragment and then failed while Lance attempted its
atomic commit rename on the artifact NFS mount.  This tool does not repair,
commit, move, delete, or replay that output.  It verifies the exact failed
tree and reads the standalone Lance data file directly, then emits an
exclusive receipt containing every identity needed to exclude the inspected
partial train population from a separately preregistered recovery build.

Raw query-pixel hashes are deliberately *not* inferred from lossy JPEG bytes.
The receipt records per-pair JPEG identities only as forensic bindings and
leaves the canonical raw query hashes pending deterministic reconstruction.
No Public path is accepted or discovered.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from io import BytesIO
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import h5py
from lance.file import LanceFileReader
import numpy as np
import pyarrow as pa
from PIL import Image
import yaml


ROOT = Path(__file__).absolute().parents[1]
PROTOCOL = "cube_gripper_carry_rule_history3_development_v4"
PREREG_STATUS = "preregistered_before_first_v4_build"
FREEZE_STATUS = "frozen_before_first_v4_build"
RECEIPT_STATUS = "infrastructure_failed_immutable_attempt"
RECEIPT_ID = "cube_gripper_carry_h3_v4_failed_formal_attempt_v1"
SOURCE_SYMBOL = "upstream_cube_single_expert_h5"

EXPECTED_PREREG_SHA256 = (
    "f8f940bd01c0dfbc7c822e8c5885e517ba6ec2ccffda64801655f45aa847761f"
)
EXPECTED_FREEZE_SHA256 = (
    "a58549ec9d5856345d4fea72ca7a7690a74204e54062ec909080c336b77af837"
)
EXPECTED_PRIOR_EXCLUSION_SHA256 = (
    "8c181529c3012cf89ecf8390d595093d256449d909c5e911297f78ed997161b4"
)
EXPECTED_BUILDER_SHA256 = (
    "b1ac55103f66754149466c75ef51dd6f5676497e9c92afb04137e5dc3df433df"
)
EXPECTED_REQUEST_SHA256 = (
    "711cdf5ecf52d9f93366c65d7f3f276eafe9e88570f0de8c6f5a2cacff05b328"
)
EXPECTED_FRAGMENT_SHA256 = (
    "15f4a5c423ba13d803a1b44f684b9f1916f6b899352f5b2ed623906cad59a920"
)
EXPECTED_FRAGMENT_SIZE_BYTES = 162_695_360
EXPECTED_SOURCE_SHA256 = (
    "0664d507c4ff12009010644c9ae950836f954e700c172ccf22e7423af1a55625"
)
EXPECTED_SOURCE_SIZE_BYTES = 101_942_558_720
EXPECTED_SOURCE_ROW_COUNT = 2_010_000
EXPECTED_SOURCE_EPISODE_COUNT = 10_000

EXPECTED_PAIR_COUNT = 2_048
EXPECTED_ROW_COUNT = 16_384
EXPECTED_EPISODE_COUNT = 4_096
EXPECTED_CATALOG_INDEX_OFFSET = 1_000_000
EXPECTED_CATALOG_INDEX_STOP_EXCLUSIVE = 1_002_048
EXPECTED_IMAGE_SIZE = (224, 224)
EXPECTED_ANCHORS = ("endpoint4", "plateau", "ramp4", "front_hold")
EXPECTED_LOGICAL_OUTPUT = (
    "artifacts/synthesis/cube_gripper_carry_rule_h3_development_v4"
)
EXPECTED_BUILDER_SNAPSHOT_LOGICAL_PATH = (
    "artifacts/evaluation/history3/cube_gripper_carry_h3_development_v4/"
    "v4_failed_attempt_builder_snapshot.py"
)
EXPECTED_FREEZE_LOGICAL_PATH = (
    "artifacts/evaluation/history3/cube_gripper_carry_h3_development_v4/"
    "development_prereg_freeze_receipt_v1.json"
)
EXPECTED_PRIOR_EXCLUSION_LOGICAL_PATH = (
    "artifacts/evaluation/history3/cube_gripper_carry_h3_development_v4/"
    "prior_episode_exclusions_final_v1.json"
)

CONTENT_FIELDS = (
    "action_profile_ids",
    "scene_template_content_hashes",
    "pair_content_hashes",
)
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
FORBIDDEN_PUBLIC_COMPONENTS = {
    "validation",
    "validation.lance",
    "public",
    "public_test",
    "public-test",
    "publictest",
}

EXPECTED_SCHEMA = pa.schema(
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

FAILURE_TRACEBACK = (
    "train: accepted 2048/2048, anchors={'endpoint4': 512, 'plateau': 512, "
    "'ramp4': 512, 'front_hold': 512}, attempted=2048, elapsed=855.1s\n"
    "Traceback (most recent call last):\n"
    "  File \"/opt/huawei/explorer-env/dataset/ag_data/code/ContextWorld/"
    "scripts/build_cube_grasp_rule_h3_v4_data.py\", line 2371, in <module>\n"
    "    main()\n"
    "  File \"/opt/huawei/explorer-env/dataset/ag_data/code/ContextWorld/"
    "scripts/build_cube_grasp_rule_h3_v4_data.py\", line 2298, in main\n"
    "    reports = {\n"
    "  File \"/opt/huawei/explorer-env/dataset/ag_data/code/ContextWorld/"
    "scripts/build_cube_grasp_rule_h3_v4_data.py\", line 2299, in <dictcomp>\n"
    "    split: build_split(\n"
    "  File \"/opt/huawei/explorer-env/dataset/ag_data/code/ContextWorld/"
    "scripts/build_cube_grasp_rule_h3_v4_data.py\", line 1672, in build_split\n"
    "    lance.write_dataset(\n"
    "  File \"/usr/local/lib/python3.10/dist-packages/lance/dataset.py\", "
    "line 6698, in write_dataset\n"
    "    inner_ds = _write_dataset(reader, uri, params)\n"
    "OSError: LanceError(IO): Generic LocalFileSystem error: Unable to rename "
    "file: Operation not permitted (os error 1), /home/runner/work/lance/"
    "lance/rust/lance-table/src/io/commit.rs:1101:50"
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def excluded_source_episodes_sha256(values: Sequence[int]) -> str:
    normalized = [int(value) for value in values]
    if normalized != sorted(set(normalized)) or any(value < 0 for value in normalized):
        raise ValueError("source episode IDs must be nonnegative, sorted, and unique")
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
        raise ValueError("query JPEG identities must be sorted and unique")
    decoded: list[bytes] = []
    for value in normalized:
        if SHA256_PATTERN.fullmatch(value) is None:
            raise ValueError("query JPEG identity is not a SHA256")
        decoded.append(bytes.fromhex(value))
    return hashlib.sha256(
        b"contextworld-cube-failed-attempt-forensic-query-jpeg-v1\0"
        + b"".join(decoded)
    ).hexdigest()


def action_profile_content_sha256(action_blocks: np.ndarray) -> str:
    blocks = np.asarray(action_blocks, dtype=np.float32)
    if blocks.shape != (4, 5, 5):
        raise RuntimeError(f"action profile shape mismatch: {blocks.shape}")
    if not np.isfinite(blocks).all():
        raise RuntimeError("action profile contains nonfinite values")
    if np.count_nonzero(blocks[3]):
        raise RuntimeError("terminal action block is not exactly zero")
    return hashlib.sha256(np.ascontiguousarray(blocks).tobytes()).hexdigest()


def pair_content_sha256(scene_hash: str, action_profile_id: str) -> str:
    for label, value in (
        ("scene_template_content_hash", scene_hash),
        ("action_profile_id", action_profile_id),
    ):
        if SHA256_PATTERN.fullmatch(value) is None:
            raise RuntimeError(f"{label} is not a lowercase SHA256")
    return hashlib.sha256(bytes.fromhex(scene_hash) + bytes.fromhex(action_profile_id)).hexdigest()


def _absolute_without_resolve(path: Path) -> Path:
    return Path(os.path.abspath(path.expanduser()))


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


def _reject_symlink(path: Path, *, label: str) -> None:
    if path.is_symlink():
        raise FileNotFoundError(f"{label} must not be a symlink: {path}")


def _regular_file(path: Path, *, label: str) -> Path:
    _reject_public(path, label=label)
    _reject_symlink(path, label=label)
    if not path.is_file():
        raise FileNotFoundError(f"{label} must be a regular file: {path}")
    return path


def _regular_directory(path: Path, *, label: str) -> Path:
    _reject_public(path, label=label)
    _reject_symlink(path, label=label)
    if not path.is_dir():
        raise FileNotFoundError(f"{label} must be a directory: {path}")
    return path


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    _regular_file(path, label=label)
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(descriptor, "rb") as stream:
        raw = stream.read()
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{label} is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} root must be an object")
    return value


def _identity(path: Path, *, logical_path: str) -> dict[str, Any]:
    _reject_public(logical_path, label="logical identity path")
    return {
        "path": logical_path,
        "sha256": file_sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _assert_sha(path: Path, expected: str, *, label: str) -> dict[str, Any]:
    _regular_file(path, label=label)
    actual = file_sha256(path)
    if actual != expected:
        raise RuntimeError(f"{label} SHA256 mismatch: {actual} != {expected}")
    return {"sha256": actual, "size_bytes": path.stat().st_size}


def _closed_public(value: Any, *, label: str) -> None:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{label} lacks Public closure")
    if value.get("access_status") != "closed_not_read_not_scored" or any(
        value.get(name) is not False for name in ("opened", "read", "hashed", "scored")
    ):
        raise RuntimeError(f"{label} did not keep Public fully closed")


def _validated_prior_sets(
    receipt: Mapping[str, Any],
) -> tuple[set[int], dict[str, set[str]]]:
    episodes = [int(value) for value in receipt.get("excluded_source_episodes", [])]
    if not episodes or episodes != sorted(set(episodes)):
        raise RuntimeError("prior exclusion source episode set is not canonical")
    if int(receipt.get("excluded_source_episode_count", -1)) != len(episodes):
        raise RuntimeError("prior exclusion source episode count mismatch")
    if receipt.get("excluded_source_episodes_sha256") != excluded_source_episodes_sha256(
        episodes
    ):
        raise RuntimeError("prior exclusion source episode digest mismatch")
    raw_content = receipt.get("prior_content_exclusions")
    if not isinstance(raw_content, Mapping):
        raise RuntimeError("prior exclusion lacks content sets")
    content: dict[str, set[str]] = {}
    for field in (*CONTENT_FIELDS, "query_pixel_hashes"):
        entry = raw_content.get(field)
        if not isinstance(entry, Mapping):
            raise RuntimeError(f"prior exclusion lacks {field}")
        values = [str(value) for value in entry.get("values", [])]
        if not values or values != sorted(set(values)):
            raise RuntimeError(f"prior exclusion {field} is not canonical")
        if int(entry.get("count", -1)) != len(values) or entry.get(
            "sha256"
        ) != canonical_content_digest(values, field_name=field):
            raise RuntimeError(f"prior exclusion {field} count/digest mismatch")
        content[field] = set(values)
    return set(episodes), content


def _verify_fixed_inputs(
    *,
    prereg_path: Path,
    freeze_path: Path,
    prior_path: Path,
    builder_snapshot: Path,
    builder_snapshot_logical_path: str,
    source_h5: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    prereg_identity = _assert_sha(
        prereg_path, EXPECTED_PREREG_SHA256, label="original v4 preregistration"
    )
    try:
        prereg = yaml.safe_load(prereg_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise RuntimeError("original preregistration is not valid YAML") from error
    if not isinstance(prereg, Mapping) or prereg.get("protocol_id") != PROTOCOL:
        raise RuntimeError("original preregistration protocol mismatch")
    if prereg.get("status") != PREREG_STATUS:
        raise RuntimeError("original preregistration status mismatch")
    _closed_public(prereg.get("public_test"), label="original preregistration")
    reference = prereg.get("reference_model_phase")
    if not isinstance(reference, Mapping) or reference.get(
        "training_and_scoring_authorized"
    ) is not False:
        raise RuntimeError("original preregistration unexpectedly authorizes training")

    freeze_identity = _assert_sha(
        freeze_path, EXPECTED_FREEZE_SHA256, label="original v4 freeze receipt"
    )
    freeze = _read_json(freeze_path, label="original v4 freeze receipt")
    if (
        freeze.get("schema_version") != 1
        or freeze.get("protocol_id") != PROTOCOL
        or freeze.get("status") != FREEZE_STATUS
        or freeze.get("checks_passed") is not True
    ):
        raise RuntimeError("original v4 freeze receipt identity mismatch")
    if freeze.get("preregistration", {}).get("sha256") != EXPECTED_PREREG_SHA256:
        raise RuntimeError("freeze receipt does not bind original preregistration")
    if freeze.get("identity", {}).get("v4_builder", {}).get(
        "sha256"
    ) != EXPECTED_BUILDER_SHA256:
        raise RuntimeError("freeze receipt does not bind original builder")
    _closed_public(freeze.get("public_test"), label="original freeze receipt")
    if freeze.get("reference_model_training_or_scoring_authorized") is not False:
        raise RuntimeError("original freeze receipt unexpectedly authorizes training")

    prior_identity = _assert_sha(
        prior_path,
        EXPECTED_PRIOR_EXCLUSION_SHA256,
        label="original v4 prior-exclusion receipt",
    )
    prior = _read_json(prior_path, label="original v4 prior-exclusion receipt")
    if (
        prior.get("schema_version") != 1
        or prior.get("protocol_id") != PROTOCOL
        or prior.get("status") != FREEZE_STATUS
        or prior.get("checks_passed") is not True
    ):
        raise RuntimeError("original prior-exclusion receipt identity mismatch")
    if prior.get("preregistration", {}).get("sha256") != EXPECTED_PREREG_SHA256:
        raise RuntimeError("prior exclusion does not bind original preregistration")
    if prior.get("freeze_receipt", {}).get("sha256") != EXPECTED_FREEZE_SHA256:
        raise RuntimeError("prior exclusion does not bind original freeze receipt")
    _closed_public(prior.get("public_test"), label="original prior exclusion")
    if prior.get("reference_model_training_or_scoring") is not False:
        raise RuntimeError("prior exclusion reports model training/scoring")

    if builder_snapshot_logical_path != EXPECTED_BUILDER_SNAPSHOT_LOGICAL_PATH:
        raise RuntimeError("builder snapshot logical path mismatch")
    builder_identity = _assert_sha(
        builder_snapshot,
        EXPECTED_BUILDER_SHA256,
        label="immutable failed-attempt builder snapshot",
    )

    source_identity = _assert_sha(
        source_h5, EXPECTED_SOURCE_SHA256, label="frozen source H5"
    )
    if source_identity["size_bytes"] != EXPECTED_SOURCE_SIZE_BYTES:
        raise RuntimeError("frozen source H5 size mismatch")
    with h5py.File(source_h5, "r", swmr=True) as handle:
        if "action" not in handle or "ep_len" not in handle:
            raise RuntimeError("frozen source H5 lacks action or ep_len")
        row_count = int(handle["action"].shape[0])
        episode_count = int(handle["ep_len"].shape[0])
    if row_count != EXPECTED_SOURCE_ROW_COUNT or episode_count != EXPECTED_SOURCE_EPISODE_COUNT:
        raise RuntimeError("frozen source H5 row/episode count mismatch")
    source_from_freeze = freeze.get("source_h5")
    source_from_prior = prior.get("source_h5")
    expected_source = {
        "symbol": SOURCE_SYMBOL,
        "sha256": EXPECTED_SOURCE_SHA256,
        "size_bytes": EXPECTED_SOURCE_SIZE_BYTES,
        "row_count": EXPECTED_SOURCE_ROW_COUNT,
        "episode_count": EXPECTED_SOURCE_EPISODE_COUNT,
    }
    for label, value in (
        ("freeze source", source_from_freeze),
        ("prior-exclusion source", source_from_prior),
    ):
        if not isinstance(value, Mapping) or {
            key: value.get(key) for key in expected_source
        } != expected_source:
            raise RuntimeError(f"{label} identity mismatch")

    identities = {
        "preregistration": {
            "path": "configs/benchmark/cube_gripper_carry_h3_development_prereg_v4.yaml",
            **prereg_identity,
        },
        "freeze_receipt": {"path": EXPECTED_FREEZE_LOGICAL_PATH, **freeze_identity},
        "prior_exclusion_receipt": {
            "path": EXPECTED_PRIOR_EXCLUSION_LOGICAL_PATH,
            **prior_identity,
        },
        "builder_snapshot": {
            "path": builder_snapshot_logical_path,
            **builder_identity,
        },
        "source_h5": {
            **expected_source,
            "path_recorded": False,
            "content_rehashed_for_failure_receipt": True,
        },
    }
    runtime = {
        key: dict(freeze["identity"][key])
        for key in ("v4_physics", "v3_physics_dependency", "common_causal_contract")
    }
    return identities, runtime, prior, freeze


def _verify_inventory(
    *, failed_output_root: Path, request_json: Path, fragment: Path
) -> list[dict[str, Any]]:
    _regular_directory(failed_output_root, label="failed formal output root")
    _regular_file(request_json, label="failed build request")
    _regular_file(fragment, label="partial train fragment")
    if request_json != failed_output_root / "request.json":
        raise RuntimeError("request must be the failed root request.json")
    expected_fragment_parent = failed_output_root / "train.lance" / "data"
    if fragment.parent != expected_fragment_parent or fragment.suffix != ".lance":
        raise RuntimeError("partial fragment must be the sole train.lance/data/*.lance")
    expected_paths = {
        Path("request.json"),
        Path("train.lance"),
        Path("train.lance/data"),
        Path("train.lance/_versions"),
        Path("train.lance/_transactions"),
        Path("train.lance/data") / fragment.name,
    }
    observed: set[Path] = set()
    for path in failed_output_root.rglob("*"):
        relative = path.relative_to(failed_output_root)
        if path.is_symlink():
            raise RuntimeError(f"failed output contains symlink: {relative}")
        observed.add(relative)
    if observed != expected_paths:
        raise RuntimeError(
            "failed output inventory mismatch: "
            f"extra={sorted(map(str, observed - expected_paths))}, "
            f"missing={sorted(map(str, expected_paths - observed))}"
        )
    for name in ("_versions", "_transactions"):
        directory = failed_output_root / "train.lance" / name
        if any(directory.iterdir()):
            raise RuntimeError(f"train.lance/{name} must be empty after failed commit")
    return [
        {
            "path": f"{EXPECTED_LOGICAL_OUTPUT}/request.json",
            "type": "regular_file",
            "sha256": file_sha256(request_json),
            "size_bytes": request_json.stat().st_size,
        },
        {
            "path": f"{EXPECTED_LOGICAL_OUTPUT}/train.lance/data/{fragment.name}",
            "type": "regular_file",
            "sha256": file_sha256(fragment),
            "size_bytes": fragment.stat().st_size,
        },
        *[
            {
                "path": f"{EXPECTED_LOGICAL_OUTPUT}/train.lance/{name}",
                "type": "empty_directory",
                "entry_count": 0,
            }
            for name in ("_versions", "_transactions")
        ],
    ]


def _verify_request(
    request_path: Path,
    *,
    freeze_identity: Mapping[str, Any],
    prior_identity: Mapping[str, Any],
) -> dict[str, Any]:
    identity = _assert_sha(request_path, EXPECTED_REQUEST_SHA256, label="failed request")
    request = _read_json(request_path, label="failed request")
    expected_pairs = {"train": EXPECTED_PAIR_COUNT, "loader_validation": 256}
    if request.get("protocol") != PROTOCOL:
        raise RuntimeError("failed request protocol mismatch")
    if request.get("resolved_output") != EXPECTED_LOGICAL_OUTPUT or request.get(
        "logical_default_output"
    ) != EXPECTED_LOGICAL_OUTPUT:
        raise RuntimeError("failed request output identity mismatch")
    if request.get("pair_counts") != expected_pairs:
        raise RuntimeError("failed request pair counts mismatch")
    if request.get("active_splits") != ["train", "loader_validation"]:
        raise RuntimeError("failed request active splits mismatch")
    if request.get("jpeg_quality") != 95 or request.get("workers") != 16:
        raise RuntimeError("failed request execution parameters mismatch")
    if request.get("public_test_opened") is not False or request.get(
        "public_test_generated"
    ) is not False:
        raise RuntimeError("failed request reports Public access")
    frozen = request.get("freeze_receipt")
    prior = request.get("prior_episode_exclusion_receipt")
    if not isinstance(frozen, Mapping) or frozen.get("sha256") != freeze_identity[
        "sha256"
    ]:
        raise RuntimeError("failed request freeze binding mismatch")
    if not isinstance(prior, Mapping) or prior.get("sha256") != prior_identity[
        "sha256"
    ]:
        raise RuntimeError("failed request prior-exclusion binding mismatch")
    source = request.get("source")
    if not isinstance(source, Mapping) or {
        key: source.get(key)
        for key in (
            "source_symbol",
            "source_file_sha256",
            "source_size_bytes",
            "source_row_count",
            "source_episode_count",
        )
    } != {
        "source_symbol": SOURCE_SYMBOL,
        "source_file_sha256": EXPECTED_SOURCE_SHA256,
        "source_size_bytes": EXPECTED_SOURCE_SIZE_BYTES,
        "source_row_count": EXPECTED_SOURCE_ROW_COUNT,
        "source_episode_count": EXPECTED_SOURCE_EPISODE_COUNT,
    }:
        raise RuntimeError("failed request source identity mismatch")
    namespace = source.get("formal_catalog_namespace")
    train_range = namespace.get("per_split_ranges", {}).get("train", {}) if isinstance(namespace, Mapping) else {}
    if train_range.get("catalog_index_start_inclusive") != EXPECTED_CATALOG_INDEX_OFFSET:
        raise RuntimeError("failed request train catalog start mismatch")
    return {
        "path": f"{EXPECTED_LOGICAL_OUTPUT}/request.json",
        **identity,
        "protocol": PROTOCOL,
        "pair_counts": dict(expected_pairs),
        "jpeg_quality": 95,
        "workers": 16,
        "public_test_opened_or_generated": False,
    }


def _jpeg_identity(value: bytes, *, label: str) -> str:
    if not value:
        raise RuntimeError(f"{label} JPEG is empty")
    try:
        with Image.open(BytesIO(value)) as image:
            image.verify()
        with Image.open(BytesIO(value)) as image:
            if image.format != "JPEG" or image.mode != "RGB" or image.size != EXPECTED_IMAGE_SIZE:
                raise RuntimeError(
                    f"{label} JPEG identity mismatch: "
                    f"format={image.format}, mode={image.mode}, size={image.size}"
                )
    except (OSError, SyntaxError) as error:
        raise RuntimeError(f"{label} is not a valid JPEG") from error
    return hashlib.sha256(value).hexdigest()


def _single(values: Sequence[Any], *, label: str) -> Any:
    unique = set(values)
    if len(unique) != 1:
        raise RuntimeError(f"{label} is not constant within a pair: {unique}")
    return next(iter(unique))


def _validate_fragment(
    fragment: Path,
    *,
    prior_episodes: set[int],
    prior_content: Mapping[str, set[str]],
) -> dict[str, Any]:
    fragment_identity = _assert_sha(
        fragment, EXPECTED_FRAGMENT_SHA256, label="partial train Lance fragment"
    )
    if fragment_identity["size_bytes"] != EXPECTED_FRAGMENT_SIZE_BYTES:
        raise RuntimeError("partial train fragment size mismatch")
    reader = LanceFileReader(str(fragment))
    metadata = reader.metadata()
    if int(metadata.num_rows) != EXPECTED_ROW_COUNT:
        raise RuntimeError("partial train fragment metadata row count mismatch")
    if not metadata.schema.equals(EXPECTED_SCHEMA, check_metadata=False):
        raise RuntimeError(
            "partial train fragment schema mismatch: "
            f"observed={metadata.schema}, expected={EXPECTED_SCHEMA}"
        )
    table = reader.read_all(batch_size=2_048).to_table()
    if table.num_rows != EXPECTED_ROW_COUNT or not table.schema.equals(
        EXPECTED_SCHEMA, check_metadata=False
    ):
        raise RuntimeError("partial train fragment decoded row/schema mismatch")

    scalar_columns = {
        name: table[name].to_pylist()
        for name in (
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
    }
    if set(scalar_columns["split"]) != {"train"}:
        raise RuntimeError("partial fragment contains a non-train split")
    pairs: dict[str, list[int]] = defaultdict(list)
    for index, pair_id in enumerate(scalar_columns["pair_id"]):
        pairs[str(pair_id)].append(index)
    if len(pairs) != EXPECTED_PAIR_COUNT:
        raise RuntimeError("partial fragment pair count mismatch")

    actions = np.asarray(table["action_block"].to_pylist(), dtype=np.float32).reshape(
        EXPECTED_ROW_COUNT, 5, 5
    )
    physical = np.asarray(table["physical_state"].to_pylist(), dtype=np.float32).reshape(
        EXPECTED_ROW_COUNT, 7
    )
    hidden = np.asarray(
        table["hidden_grasp_enabled"].to_pylist(), dtype=np.float32
    ).reshape(EXPECTED_ROW_COUNT)
    if not np.isfinite(actions).all() or not np.isfinite(physical).all() or not np.isfinite(hidden).all():
        raise RuntimeError("partial fragment contains nonfinite numeric content")

    episode_ids: set[int] = set()
    source_episodes: set[int] = set()
    profile_ids: set[str] = set()
    scene_hashes: set[str] = set()
    pair_hashes: set[str] = set()
    jpeg_hashes: set[str] = set()
    pair_receipts: list[dict[str, Any]] = []
    anchor_counts: Counter[str] = Counter()
    extrema = {
        "maximum_abs_sum_p": 0.0,
        "maximum_abs_final_p": 0.0,
        "maximum_abs_moment_error": 0.0,
        "terminal_nonzero_value_count": 0,
    }
    expected_catalogs = list(
        range(EXPECTED_CATALOG_INDEX_OFFSET, EXPECTED_CATALOG_INDEX_STOP_EXCLUSIVE)
    )
    observed_catalogs: list[int] = []
    pixels = table["pixels"]
    for pair_id in sorted(pairs):
        indices = pairs[pair_id]
        if len(indices) != 8:
            raise RuntimeError(f"{pair_id} does not contain exactly eight rows")
        catalog_index = int(
            _single(
                [scalar_columns["catalog_index"][index] for index in indices],
                label=f"{pair_id}.catalog_index",
            )
        )
        local_index = catalog_index - EXPECTED_CATALOG_INDEX_OFFSET
        expected_pair_id = f"cube-carry-v4-train-{local_index:06d}"
        if pair_id != expected_pair_id:
            raise RuntimeError(f"pair ID/catalog mismatch: {pair_id} != {expected_pair_id}")
        observed_catalogs.append(catalog_index)
        metadata_fields = {}
        for name in (
            "source_row",
            "source_episode",
            "source_step",
            "action_anchor_id",
            "action_profile_id",
            "scene_template_content_hash",
            "pair_content_hash",
        ):
            metadata_fields[name] = _single(
                [scalar_columns[name][index] for index in indices],
                label=f"{pair_id}.{name}",
            )
        anchor = str(metadata_fields["action_anchor_id"])
        expected_anchor = EXPECTED_ANCHORS[catalog_index % len(EXPECTED_ANCHORS)]
        if anchor != expected_anchor:
            raise RuntimeError(f"{pair_id} anchor/catalog mismatch")
        anchor_counts[anchor] += 1

        mode_rows: dict[str, list[int]] = {}
        for mode, hidden_value in (("cannot_hold", 0.0), ("can_hold", 1.0)):
            selected = [
                index
                for index in indices
                if scalar_columns["hidden_mode"][index] == mode
            ]
            selected.sort(key=lambda index: scalar_columns["model_step_idx"][index])
            if [scalar_columns["model_step_idx"][index] for index in selected] != [0, 1, 2, 3]:
                raise RuntimeError(f"{pair_id}/{mode} does not contain model steps 0..3")
            if any(float(hidden[index]) != hidden_value for index in selected):
                raise RuntimeError(f"{pair_id}/{mode} hidden value mismatch")
            one_episode = int(
                _single(
                    [scalar_columns["episode_idx"][index] for index in selected],
                    label=f"{pair_id}/{mode}.episode_idx",
                )
            )
            if one_episode in episode_ids:
                raise RuntimeError("episode_idx is reused across pair/mode groups")
            episode_ids.add(one_episode)
            mode_rows[mode] = selected
        low = mode_rows["cannot_hold"]
        high = mode_rows["can_hold"]
        if not np.array_equal(actions[low], actions[high]):
            raise RuntimeError(f"{pair_id} paired action blocks differ")
        blocks = actions[low]
        calculated_profile = action_profile_content_sha256(blocks)
        stored_profile = str(metadata_fields["action_profile_id"])
        if stored_profile != calculated_profile:
            raise RuntimeError(f"{pair_id} action profile hash mismatch")
        p = blocks[0, :, 2]
        sum_p = float(np.sum(p, dtype=np.float64))
        final_p = float(p[-1])
        moment_error = float(
            np.dot(np.asarray([4, 3, 2, 1, 0], dtype=np.float64), p) - 1.0
        )
        if (
            sum_p != 0.0
            or final_p != 0.0
            or moment_error != 0.0
            or not np.array_equal(blocks[1, :, 2], -p)
            or not np.array_equal(blocks[2, :, 2], p)
        ):
            raise RuntimeError(f"{pair_id} action recovery constraints failed")
        nonzero_terminal = int(np.count_nonzero(blocks[3]))
        if nonzero_terminal:
            raise RuntimeError(f"{pair_id} terminal block is nonzero")
        extrema["maximum_abs_sum_p"] = max(extrema["maximum_abs_sum_p"], abs(sum_p))
        extrema["maximum_abs_final_p"] = max(extrema["maximum_abs_final_p"], abs(final_p))
        extrema["maximum_abs_moment_error"] = max(
            extrema["maximum_abs_moment_error"], abs(moment_error)
        )
        extrema["terminal_nonzero_value_count"] += nonzero_terminal

        scene_hash = str(metadata_fields["scene_template_content_hash"])
        stored_pair_hash = str(metadata_fields["pair_content_hash"])
        if SHA256_PATTERN.fullmatch(scene_hash) is None:
            raise RuntimeError(f"{pair_id} scene hash is malformed")
        if pair_content_sha256(scene_hash, stored_profile) != stored_pair_hash:
            raise RuntimeError(f"{pair_id} pair content hash mismatch")
        query_rows = [low[2], high[2]]
        query_bytes = [pixels[index].as_py() for index in query_rows]
        if query_bytes[0] != query_bytes[1]:
            raise RuntimeError(f"{pair_id} paired query JPEG bytes differ")
        query_jpeg = _jpeg_identity(query_bytes[0], label=f"{pair_id}.query")

        source_episode = int(metadata_fields["source_episode"])
        source_episodes.add(source_episode)
        profile_ids.add(stored_profile)
        scene_hashes.add(scene_hash)
        pair_hashes.add(stored_pair_hash)
        if query_jpeg in jpeg_hashes:
            raise RuntimeError("query JPEG identity is reused across failed train pairs")
        jpeg_hashes.add(query_jpeg)
        pair_receipts.append(
            {
                "pair_id": pair_id,
                "catalog_index": catalog_index,
                "source_row": int(metadata_fields["source_row"]),
                "source_episode": source_episode,
                "source_step": int(metadata_fields["source_step"]),
                "action_anchor_id": anchor,
                "action_profile_id": stored_profile,
                "scene_template_content_hash": scene_hash,
                "pair_content_hash": stored_pair_hash,
                "query_jpeg_sha256": query_jpeg,
            }
        )

    if observed_catalogs != expected_catalogs:
        raise RuntimeError("failed train catalog indices are not the exact frozen range")
    if episode_ids != set(range(EXPECTED_EPISODE_COUNT)):
        raise RuntimeError("episode_idx values are not contiguous 0..4095")
    expected_unique = EXPECTED_PAIR_COUNT
    for label, values in (
        ("source episodes", source_episodes),
        ("action profiles", profile_ids),
        ("scene hashes", scene_hashes),
        ("pair hashes", pair_hashes),
        ("query JPEG hashes", jpeg_hashes),
    ):
        if len(values) != expected_unique:
            raise RuntimeError(f"failed fragment {label} are not pair-unique")
    expected_anchor_count = EXPECTED_PAIR_COUNT // len(EXPECTED_ANCHORS)
    if anchor_counts != Counter({name: expected_anchor_count for name in EXPECTED_ANCHORS}):
        raise RuntimeError(f"failed train anchors are not balanced: {anchor_counts}")

    prior_overlap = {
        "source_episode_count": len(source_episodes & prior_episodes),
        "action_profile_id_count": len(profile_ids & prior_content["action_profile_ids"]),
        "scene_template_content_hash_count": len(
            scene_hashes & prior_content["scene_template_content_hashes"]
        ),
        "pair_content_hash_count": len(pair_hashes & prior_content["pair_content_hashes"]),
    }
    if any(prior_overlap.values()):
        raise RuntimeError(f"failed train fragment overlaps prior evidence: {prior_overlap}")

    source_values = sorted(source_episodes)
    set_values = {
        "action_profile_ids": sorted(profile_ids),
        "scene_template_content_hashes": sorted(scene_hashes),
        "pair_content_hashes": sorted(pair_hashes),
    }
    jpeg_values = sorted(jpeg_hashes)
    return {
        "fragment_identity": {
            **fragment_identity,
            "row_count": EXPECTED_ROW_COUNT,
            "schema": str(EXPECTED_SCHEMA),
            "lance_file_major_version": int(metadata.major_version),
            "lance_file_minor_version": int(metadata.minor_version),
        },
        "split": "train",
        "row_count": EXPECTED_ROW_COUNT,
        "episode_count": EXPECTED_EPISODE_COUNT,
        "pair_count": EXPECTED_PAIR_COUNT,
        "catalog_index_start_inclusive": EXPECTED_CATALOG_INDEX_OFFSET,
        "catalog_index_stop_exclusive": EXPECTED_CATALOG_INDEX_STOP_EXCLUSIVE,
        "action_anchor_counts": dict(sorted(anchor_counts.items())),
        "source_episodes": {
            "values": source_values,
            "count": len(source_values),
            "sha256": excluded_source_episodes_sha256(source_values),
        },
        "prior_content_exclusions": {
            field: {
                "values": values,
                "count": len(values),
                "sha256": canonical_content_digest(values, field_name=field),
            }
            for field, values in set_values.items()
        },
        "query_pixel_hash_status": (
            "pending_deterministic_raw_reconstruction_not_present_in_fragment"
        ),
        "query_jpeg_sha256": {
            "values": jpeg_values,
            "count": len(jpeg_values),
            "sha256": forensic_query_jpeg_digest(jpeg_values),
            "digest_namespace": (
                "contextworld-cube-failed-attempt-forensic-query-jpeg-v1"
            ),
            "role": "forensic_binding_only_not_raw_query_pixel_hash",
        },
        "pairs": pair_receipts,
        "profile_constraints": {**extrema, "passed": True},
        "prior_overlap": {
            **prior_overlap,
            "query_pixel_hash_count": None,
            "query_pixel_hash_overlap_status": (
                "not_computable_until_raw_query_reconstruction"
            ),
            "passed_for_directly_inspectable_identities": True,
        },
    }


def freeze_failed_attempt(
    *,
    failed_output_root: Path,
    prereg_path: Path,
    freeze_receipt_path: Path,
    prior_exclusion_receipt_path: Path,
    builder_snapshot: Path,
    builder_snapshot_logical_path: str,
    source_h5: Path,
    request_json: Path,
    partial_train_fragment: Path,
    output: Path,
) -> dict[str, Any]:
    paths = {
        "failed_output_root": failed_output_root,
        "prereg_path": prereg_path,
        "freeze_receipt_path": freeze_receipt_path,
        "prior_exclusion_receipt_path": prior_exclusion_receipt_path,
        "builder_snapshot": builder_snapshot,
        "source_h5": source_h5,
        "request_json": request_json,
        "partial_train_fragment": partial_train_fragment,
        "output": output,
    }
    for label, path in paths.items():
        _reject_public(path, label=label)
    _reject_public(builder_snapshot_logical_path, label="builder snapshot logical path")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite failed-attempt receipt {output}")
    if not output.parent.is_dir() or output.parent.is_symlink():
        raise FileNotFoundError("failed-attempt receipt parent must be an existing directory")

    identities, runtime, prior, _ = _verify_fixed_inputs(
        prereg_path=prereg_path,
        freeze_path=freeze_receipt_path,
        prior_path=prior_exclusion_receipt_path,
        builder_snapshot=builder_snapshot,
        builder_snapshot_logical_path=builder_snapshot_logical_path,
        source_h5=source_h5,
    )
    inventory = _verify_inventory(
        failed_output_root=failed_output_root,
        request_json=request_json,
        fragment=partial_train_fragment,
    )
    request_identity = _verify_request(
        request_json,
        freeze_identity=identities["freeze_receipt"],
        prior_identity=identities["prior_exclusion_receipt"],
    )
    prior_episodes, prior_content = _validated_prior_sets(prior)
    content = _validate_fragment(
        partial_train_fragment,
        prior_episodes=prior_episodes,
        prior_content=prior_content,
    )
    identities["request_json"] = request_identity
    identities["partial_train_fragment"] = {
        "path": (
            f"{EXPECTED_LOGICAL_OUTPUT}/train.lance/data/"
            f"{partial_train_fragment.name}"
        ),
        **content["fragment_identity"],
    }

    receipt = {
        "schema_version": 1,
        "protocol_id": PROTOCOL,
        "receipt_id": RECEIPT_ID,
        "status": RECEIPT_STATUS,
        "checks_passed": True,
        "build_passed": False,
        "formal_build_attempt_consumed": True,
        "retry_authorized_under_original_preregistration": False,
        "failure": {
            "exit_code": 1,
            "stage": "lance_train_commit_atomic_rename",
            "errno_name": "EPERM",
            "errno_number": 1,
            "exception_type": "OSError",
            "exception_message": (
                "LanceError(IO): Generic LocalFileSystem error: Unable to rename "
                "file: Operation not permitted (os error 1)"
            ),
            "rename_source_or_destination_reported": False,
            "traceback_provenance": "operator_transcript_not_persistent_log",
            "persistent_log_present": False,
            "operator_transcript": FAILURE_TRACEBACK,
            "operator_transcript_is_independently_file_bound": False,
        },
        "stage_completion": {
            "train_generation_accepted_pairs": EXPECTED_PAIR_COUNT,
            "train_generation_attempted_candidates": EXPECTED_PAIR_COUNT,
            "train_lance_data_fragment_written": True,
            "train_lance_commit_completed": False,
            "loader_validation_started": False,
            "build_report_written": False,
            "manifest_written": False,
            "scientifically_inspectable_partial_output": True,
        },
        "input_identities": identities,
        "frozen_runtime_dependencies_from_original_freeze": runtime,
        "failed_output": {
            "logical_root": EXPECTED_LOGICAL_OUTPUT,
            "inventory": inventory,
            "allowed_inventory_only": True,
            "lance_versions_directory_empty": True,
            "lance_transactions_directory_empty": True,
        },
        "failed_attempt_content": {
            key: value for key, value in content.items() if key != "fragment_identity"
        },
        "raw_query_reconstruction_requirement": {
            "status": "required_before_any_recovery_prior_exclusion_receipt",
            "must_use_frozen_source_builder_and_physics": True,
            "must_reencode_jpeg95_and_match_per_pair_forensic_identity": True,
            "jpeg_identity_must_not_be_used_as_raw_query_pixel_hash": True,
            "recovery_build_forbidden_until_complete": True,
        },
        "scope": {
            "public_test": {
                "access_status": "closed_not_read_not_scored",
                "opened": False,
                "read": False,
                "hashed": False,
                "scored": False,
            },
            "rgb_probe_run": False,
            "reference_model_training_or_scoring": False,
            "optimizer_steps": 0,
        },
        "recovery_policy": {
            "original_v4_preregistration_attempt_budget_exhausted": True,
            "original_failed_tree_must_remain_immutable": True,
            "silent_retry_or_overwrite_forbidden": True,
            "newly_frozen_recovery_preregistration_required": True,
            "failed_source_action_scene_pair_and_reconstructed_raw_query_must_be_excluded": True,
        },
    }
    with output.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return receipt


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--failed-output-root", type=Path, required=True)
    parser.add_argument("--prereg", type=Path, required=True)
    parser.add_argument("--freeze-receipt", type=Path, required=True)
    parser.add_argument("--prior-exclusion-receipt", type=Path, required=True)
    parser.add_argument("--builder-snapshot", type=Path, required=True)
    parser.add_argument("--builder-snapshot-logical-path", required=True)
    parser.add_argument("--source-h5", type=Path, required=True)
    parser.add_argument("--request-json", type=Path, required=True)
    parser.add_argument("--partial-train-fragment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    for name in (
        "failed_output_root",
        "prereg",
        "freeze_receipt",
        "prior_exclusion_receipt",
        "builder_snapshot",
        "source_h5",
        "request_json",
        "partial_train_fragment",
        "output",
    ):
        value = getattr(args, name)
        _reject_public(value, label=name)
        setattr(args, name, _absolute_without_resolve(value))
    _reject_public(
        args.builder_snapshot_logical_path,
        label="builder_snapshot_logical_path",
    )
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    receipt = freeze_failed_attempt(
        failed_output_root=args.failed_output_root,
        prereg_path=args.prereg,
        freeze_receipt_path=args.freeze_receipt,
        prior_exclusion_receipt_path=args.prior_exclusion_receipt,
        builder_snapshot=args.builder_snapshot,
        builder_snapshot_logical_path=args.builder_snapshot_logical_path,
        source_h5=args.source_h5,
        request_json=args.request_json,
        partial_train_fragment=args.partial_train_fragment,
        output=args.output,
    )
    content = receipt["failed_attempt_content"]
    print(
        json.dumps(
            {
                "output": str(args.output),
                "status": receipt["status"],
                "checks_passed": receipt["checks_passed"],
                "pair_count": content["pair_count"],
                "source_episode_count": content["source_episodes"]["count"],
                "raw_query_pixel_hash_status": content["query_pixel_hash_status"],
                "public_test_read": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
