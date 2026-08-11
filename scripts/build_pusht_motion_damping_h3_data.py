#!/usr/bin/env python3
"""Build formal PushT History-3 motion-damping benchmark data."""

from __future__ import annotations

import argparse
from collections import Counter
import copy
from dataclasses import asdict
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any, Iterator

import lance
import numpy as np
import pyarrow as pa
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
STABLE_WORLD_MODEL_ROOT = ROOT.parent / "stable-worldmodel"
for source_root in (ROOT, STABLE_WORLD_MODEL_ROOT):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from contextworld.evaluation.pusht_motion_damping_h3 import (  # noqa: E402
    DAMPING_VALUES,
    ENDPOINT_MODES,
    QUERY_STATE_TOLERANCE,
    make_catalog_template,
    simulate_motion_damping_clip,
    validate_motion_damping_pair,
)
from contextworld.release_metadata import (  # noqa: E402
    frozen_predecessor_reference,
    portable_release_metadata,
    write_portable_release_json,
)


PROTOCOL = "pusht_motion_damping_history3_strict_causal_release_v3"
SPLITS = ("train", "loader_validation", "validation")
DEFAULT_OUTPUT = (
    ROOT / "artifacts/synthesis/pusht_motion_damping_h3_release_v3"
)
DEFAULT_LEGACY_ROOT = (
    ROOT / "artifacts/synthesis/pusht_motion_damping_h3_release_v1"
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
        digest.update(child.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_sha256(child).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    data = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(data.dtype).encode("ascii"))
    digest.update(np.asarray(data.shape, dtype=np.int64).tobytes())
    digest.update(data.tobytes())
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def model_visible_sha256(
    table_path: Path,
    *,
    episode_count: int | None = None,
) -> dict[str, Any]:
    """Hash the exact stored pixels/actions consumed by the reference loader."""

    dataset = lance.dataset(str(table_path))
    filter_expression = (
        None if episode_count is None else f"episode_idx < {int(episode_count)}"
    )
    scanner = dataset.scanner(
        columns=["episode_idx", "step_idx", "pixels", "action"],
        filter=filter_expression,
    )
    digest = hashlib.sha256()
    row_count = 0
    for batch in scanner.to_batches():
        episodes = np.asarray(batch.column("episode_idx"), dtype=np.int32)
        steps = np.asarray(batch.column("step_idx"), dtype=np.int32)
        pixels = batch.column("pixels").to_pylist()
        actions = np.asarray(
            batch.column("action").to_pylist(), dtype=np.float32
        ).reshape(-1, 2)
        for episode, step, pixel, action in zip(
            episodes, steps, pixels, actions, strict=True
        ):
            digest.update(np.asarray([episode, step], dtype=np.int32).tobytes())
            digest.update(np.asarray([len(pixel)], dtype=np.int64).tobytes())
            digest.update(pixel)
            digest.update(np.ascontiguousarray(action).tobytes())
            row_count += 1
    return {"row_count": row_count, "sha256": digest.hexdigest()}


def safe_output_path(path: Path) -> Path:
    result = Path(os.path.abspath(path.expanduser()))
    if result.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {result}")
    result.parent.mkdir(parents=True, exist_ok=True)
    return result


LANCE_SCHEMA = pa.schema(
    [
        pa.field("episode_idx", pa.int32()),
        pa.field("step_idx", pa.int32()),
        pa.field("pixels", pa.binary()),
        pa.field("action", pa.list_(pa.float32(), 2)),
        pa.field("proprio", pa.list_(pa.float32(), 4)),
        pa.field("state", pa.list_(pa.float32(), 7)),
        pa.field("goal_state", pa.list_(pa.float32(), 7)),
        pa.field("physics_state", pa.list_(pa.float32(), 12)),
        pa.field("n_contacts", pa.list_(pa.float32(), 1)),
        pa.field("hidden_motion_damping", pa.list_(pa.float32(), 1)),
        pa.field("pair_id", pa.string()),
        pa.field("hidden_mode", pa.string()),
        pa.field("split", pa.string()),
        pa.field("synthesis_version", pa.string()),
        pa.field("catalog_index", pa.list_(pa.float32(), 1)),
    ]
)


def _fixed(values: list[Any], size: int) -> pa.FixedSizeListArray:
    flat = np.asarray(values, dtype=np.float32).reshape(-1, size)
    return pa.FixedSizeListArray.from_arrays(
        pa.array(flat.reshape(-1), type=pa.float32()), size
    )


def _encode_rgb(value: np.ndarray, quality: int) -> bytes:
    buffer = BytesIO()
    Image.fromarray(np.asarray(value, dtype=np.uint8)).save(
        buffer, format="JPEG", quality=quality
    )
    return buffer.getvalue()


def _episode_rows(
    rollout: dict[str, Any], *, split: str, catalog_index: int
) -> dict[str, list[Any]]:
    rows = {key: list(value) for key, value in rollout["rows"].items()}
    count = len(rows["pixels"])
    rows["split"] = [split] * count
    rows["synthesis_version"] = ["motion_damping_h3_strict_causal_v3"] * count
    rows["catalog_index"] = [
        np.asarray([catalog_index], dtype=np.float32) for _ in range(count)
    ]
    return rows


def _batch(
    rows: dict[str, list[Any]], *, episode_index: int, jpeg_quality: int
) -> pa.RecordBatch:
    count = len(rows["pixels"])
    arrays: list[pa.Array] = [
        pa.array(np.full(count, episode_index, dtype=np.int32)),
        pa.array(np.arange(count, dtype=np.int32)),
        pa.array(
            [_encode_rgb(value, jpeg_quality) for value in rows["pixels"]],
            type=pa.binary(),
        ),
    ]
    for name, size in (
        ("action", 2),
        ("proprio", 4),
        ("state", 7),
        ("goal_state", 7),
        ("physics_state", 12),
        ("n_contacts", 1),
        ("hidden_motion_damping", 1),
    ):
        arrays.append(_fixed(rows[name], size))
    for name in ("pair_id", "hidden_mode", "split", "synthesis_version"):
        arrays.append(pa.array(rows[name], type=pa.string()))
    arrays.append(_fixed(rows["catalog_index"], 1))
    return pa.record_batch(arrays, schema=LANCE_SCHEMA)


def _write_lance(
    path: Path,
    episodes: Iterator[dict[str, list[Any]]],
    *,
    jpeg_quality: int,
) -> None:
    def batches() -> Iterator[pa.RecordBatch]:
        for episode_index, rows in enumerate(episodes):
            yield _batch(
                rows,
                episode_index=episode_index,
                jpeg_quality=jpeg_quality,
            )

    lance.write_dataset(
        pa.RecordBatchReader.from_batches(LANCE_SCHEMA, batches()),
        str(path),
        mode="create",
    )


def build_split(
    *,
    root: Path,
    split: str,
    pair_count: int,
    catalog_seed: int,
    resolution: int,
    jpeg_quality: int,
) -> dict[str, Any]:
    table_path = root / f"{split}.lance"
    reports: list[dict[str, Any]] = []
    query_hashes: set[str] = set()
    template_ids: set[str] = set()
    accepted_indices: list[int] = []
    started = time.monotonic()

    def episodes() -> Iterator[dict[str, list[Any]]]:
        catalog_index = 0
        while len(reports) < pair_count:
            template = make_catalog_template(
                split=split,
                catalog_index=catalog_index,
                catalog_seed=catalog_seed,
            )
            faster = simulate_motion_damping_clip(
                template, mode=ENDPOINT_MODES[0], resolution=resolution
            )
            no_extra = simulate_motion_damping_clip(
                template, mode=ENDPOINT_MODES[1], resolution=resolution
            )
            audit = validate_motion_damping_pair(faster, no_extra)
            if not audit["passed"]:
                failed = [name for name, passed in audit["checks"].items() if not passed]
                raise RuntimeError(
                    f"Frozen catalog candidate failed {template.template_id}: {failed}"
                )
            query_hash = audit["hashes"]["query_pixels"]
            if query_hash in query_hashes:
                raise RuntimeError(f"Duplicate query pixels: {template.template_id}")
            query_hashes.add(query_hash)
            template_ids.add(template.template_id)
            accepted_indices.append(catalog_index)
            angle = float(template.expected_natural_query_snapshot[10])
            reports.append(
                {
                    "pair_index": len(reports),
                    "catalog_index": catalog_index,
                    "template": asdict(template),
                    "orientation_bin": int(round(angle / (np.pi / 2))) % 4,
                    "audit": audit,
                }
            )
            yield _episode_rows(
                faster, split=split, catalog_index=catalog_index
            )
            yield _episode_rows(
                no_extra, split=split, catalog_index=catalog_index
            )
            catalog_index += 1
            if len(reports) <= 4 or len(reports) % 64 == 0:
                print(
                    f"  {split}: {len(reports)}/{pair_count} pairs, "
                    f"elapsed={time.monotonic() - started:.1f}s",
                    flush=True,
                )

    _write_lance(table_path, episodes(), jpeg_quality=jpeg_quality)
    orientation_counts = {
        str(index): sum(row["orientation_bin"] == index for row in reports)
        for index in range(4)
    }
    history_gaps = [
        row["audit"]["history_visible_response_gap"]["px_equivalent"]
        for row in reports
    ]
    reference_deviations = [
        value
        for row in reports
        for value in row["audit"]["query_reference_deviation"].values()
    ]
    query_state_gaps = [
        row["audit"]["max_pair_full_state_gap"] for row in reports
    ]
    query_pixel_differences = [
        row["audit"]["max_pair_query_pixel_difference"] for row in reports
    ]
    query_action_differences = [
        row["audit"]["max_pair_query_action_difference"] for row in reports
    ]
    future_gaps = [
        row["audit"]["future_gap"]["block_position_px"] for row in reports
    ]
    initial_hash_counts = {
        mode: Counter(
            row["audit"]["hashes"][f"{mode}_initial_pixels"]
            for row in reports
        )
        for mode in ENDPOINT_MODES
    }
    x0_rgb_bayes_correct = sum(
        max(
            initial_hash_counts[ENDPOINT_MODES[0]][value],
            initial_hash_counts[ENDPOINT_MODES[1]][value],
        )
        for value in (
            set(initial_hash_counts[ENDPOINT_MODES[0]])
            | set(initial_hash_counts[ENDPOINT_MODES[1]])
        )
    )
    x0_rgb_bayes_accuracy = x0_rgb_bayes_correct / (2 * pair_count)
    maximum_arbiter_count = max(
        row["audit"]["maximum_arbiter_count_from_x0_through_x3"]
        for row in reports
    )
    files = [path for path in table_path.rglob("*") if path.is_file()]
    return {
        "split": split,
        "pair_count": pair_count,
        "episode_count": 2 * pair_count,
        "raw_rows": 40 * pair_count,
        "table_path": table_path.name,
        "table_files": len(files),
        "table_bytes": sum(path.stat().st_size for path in files),
        "table_sha256": directory_sha256(table_path),
        "catalog_seed": int(catalog_seed),
        "accepted_catalog_indices_sha256": array_sha256(
            np.asarray(accepted_indices, dtype=np.int64)
        ),
        "query_hash_count": len(query_hashes),
        "query_hashes": sorted(query_hashes),
        "template_ids": sorted(template_ids),
        "orientation_bin_counts": orientation_counts,
        "pair_count": pair_count,
        "state_installations_after_x0": 0,
        "query_simulator_recreated": False,
        "query_full_state_tolerance": QUERY_STATE_TOLERANCE,
        "query_full_state_dimensions": (
            "agent_position_velocity_angle_angular_velocity_and_"
            "block_position_velocity_angle_angular_velocity"
        ),
        "max_pair_full_state_gap": float(max(query_state_gaps)),
        "max_pair_query_pixel_difference": int(max(query_pixel_differences)),
        "max_pair_query_action_difference": float(max(query_action_differences)),
        "min_history_effect": float(min(history_gaps)),
        "max_query_reference_deviation": float(max(reference_deviations)),
        "min_true_future_effect": float(min(future_gaps)),
        "maximum_arbiter_count_from_x0_through_x3": int(
            maximum_arbiter_count
        ),
        "x0_rgb_hash_multisets_identical_across_modes": (
            initial_hash_counts[ENDPOINT_MODES[0]]
            == initial_hash_counts[ENDPOINT_MODES[1]]
        ),
        "x0_rgb_static_bayes_accuracy_upper_bound": float(
            x0_rgb_bayes_accuracy
        ),
        "pairs": reports,
        "passed": bool(
            len(reports) == pair_count
            and all(row["audit"]["passed"] for row in reports)
            and all(value > 0 for value in orientation_counts.values())
            and maximum_arbiter_count == 0
            and initial_hash_counts[ENDPOINT_MODES[0]]
            == initial_hash_counts[ENDPOINT_MODES[1]]
            and x0_rgb_bayes_accuracy == 0.5
        ),
    }


def _cross_split(reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    overlaps = {}
    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1 :]:
            overlaps[f"{left}__{right}"] = {
                "query_pixel_hashes": len(
                    set(reports[left]["query_hashes"])
                    & set(reports[right]["query_hashes"])
                ),
                "template_ids": len(
                    set(reports[left]["template_ids"])
                    & set(reports[right]["template_ids"])
                ),
            }
    return {
        "overlap_counts": overlaps,
        "passed": all(
            value == 0 for row in overlaps.values() for value in row.values()
        ),
    }


def _visible_x0_geometry(report: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Return RGB-visible x0 geometry and labels; velocities are excluded."""

    features: list[np.ndarray] = []
    labels: list[float] = []
    for pair in report["pairs"]:
        template = pair["template"]
        goal = np.asarray(template["goal_state"], dtype=np.float64)
        for label, field in enumerate(
            (
                "faster_decay_reset_snapshot",
                "no_extra_decay_reset_snapshot",
            )
        ):
            snapshot = np.asarray(template[field], dtype=np.float64)
            features.append(
                np.asarray(
                    [
                        snapshot[0],
                        snapshot[1],
                        snapshot[6],
                        snapshot[7],
                        np.sin(snapshot[10]),
                        np.cos(snapshot[10]),
                        goal[2],
                        goal[3],
                        np.sin(goal[4]),
                        np.cos(goal[4]),
                    ],
                    dtype=np.float64,
                )
            )
            labels.append(float(label))
    return np.stack(features), np.asarray(labels, dtype=np.float64)


def _x0_only_geometry_classifier(
    train_report: dict[str, Any], public_test_report: dict[str, Any]
) -> dict[str, Any]:
    """Fit a deterministic ridge-linear x0-only label classifier."""

    train_x, train_y = _visible_x0_geometry(train_report)
    test_x, test_y = _visible_x0_geometry(public_test_report)
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale < 1e-12] = 1.0
    train_design = np.column_stack(
        [np.ones(len(train_x)), (train_x - mean) / scale]
    )
    test_design = np.column_stack(
        [np.ones(len(test_x)), (test_x - mean) / scale]
    )
    targets = 2.0 * train_y - 1.0
    ridge = 1e-6 * np.eye(train_design.shape[1], dtype=np.float64)
    ridge[0, 0] = 0.0
    weights = np.linalg.solve(
        train_design.T @ train_design + ridge,
        train_design.T @ targets,
    )
    scores = test_design @ weights
    predictions = (scores >= 0.0).astype(np.float64)
    accuracy = float(np.mean(predictions == test_y))
    return {
        "training_split": "train",
        "evaluation_split": "validation",
        "feature_source": "x0_rgb_visible_geometry_only",
        "features": [
            "agent_xy",
            "block_xy",
            "block_angle_sin_cos",
            "goal_xy",
            "goal_angle_sin_cos",
        ],
        "excluded_invisible_fields": [
            "linear_velocity",
            "angular_velocity",
            "hidden_motion_damping",
        ],
        "classifier": "ridge_linear",
        "train_examples": int(len(train_y)),
        "public_test_examples": int(len(test_y)),
        "public_test_accuracy": accuracy,
        "maximum_allowed_accuracy": 0.55,
        "maximum_absolute_score": float(np.max(np.abs(scores), initial=0.0)),
        "passed": accuracy <= 0.55,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-pairs", type=int, default=2048)
    parser.add_argument("--loader-validation-pairs", type=int, default=256)
    parser.add_argument("--validation-pairs", type=int, default=256)
    parser.add_argument("--catalog-seed", type=int, default=20260803)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument(
        "--legacy-root",
        type=Path,
        default=DEFAULT_LEGACY_ROOT,
        help=(
            "Optional v1 artifact root used only to decide whether old "
            "pixels/actions are byte-identical to the strict-causal build."
        ),
    )
    parser.add_argument(
        "--reuse-evaluation-root",
        type=Path,
        default=None,
        help=(
            "Copy loader_validation.lance and validation.lance byte for "
            "byte from an audited release while rebuilding Training."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = safe_output_path(args.output)
    counts = {
        "train": int(args.train_pairs),
        "loader_validation": int(args.loader_validation_pairs),
        "validation": int(args.validation_pairs),
    }
    if any(value <= 0 for value in counts.values()):
        raise ValueError("Every split must contain at least one pair")
    seeds = {
        split: int(
            np.random.SeedSequence([args.catalog_seed, offset]).generate_state(1)[0]
        )
        for split, offset in zip(SPLITS, (19, 23, 29), strict=True)
    }
    reuse_root = (
        None
        if args.reuse_evaluation_root is None
        else args.reuse_evaluation_root.expanduser().resolve()
    )
    reuse_manifest = None
    predecessor_reference = None
    if reuse_root is not None:
        reuse_manifest_path = reuse_root / "manifest.json"
        if not reuse_manifest_path.exists():
            raise FileNotFoundError(
                f"Missing reuse manifest: {reuse_manifest_path}"
            )
        reuse_manifest = json.loads(reuse_manifest_path.read_text())
        if not reuse_manifest.get("passed"):
            raise RuntimeError("Evaluation reuse source did not pass its audit")
        for split in ("loader_validation", "validation"):
            source_count = int(
                reuse_manifest["splits"][split]["pair_count"]
            )
            if source_count != counts[split]:
                raise ValueError(
                    f"Reused {split} has {source_count} pairs, expected "
                    f"{counts[split]}"
                )
            if int(reuse_manifest["split_catalog_seeds"][split]) != seeds[split]:
                raise ValueError(
                    f"Reused {split} catalog seed does not match the new "
                    "release request"
                )
        predecessor_reference = frozen_predecessor_reference(
            manifest_sha256=file_sha256(reuse_manifest_path),
            table_sha256={
                split: reuse_manifest["splits"][split]["table_sha256"]
                for split in ("loader_validation", "validation")
            },
        )
    request = {
        "protocol": PROTOCOL,
        "history_tokens": 3,
        "raw_steps_per_action_block": 5,
        "pair_counts": counts,
        "damping_values": DAMPING_VALUES,
        "catalog_seed": int(args.catalog_seed),
        "split_catalog_seeds": seeds,
        "resolution": int(args.resolution),
        "jpeg_quality": int(args.jpeg_quality),
        "public_test_split_internal_name": "validation",
        "sealed_test_included": False,
        "evaluation_tables_reused_byte_for_byte": (
            None
            if reuse_root is None
            else {
                **predecessor_reference,
                "splits": ["loader_validation", "validation"],
            }
        ),
        "causal_chain": {
            "state_installations_after_x0": 0,
            "query_simulator_recreated": False,
            "query_full_state_tolerance": QUERY_STATE_TOLERANCE,
            "query_full_state_dimensions": (
                "agent_position_velocity_angle_angular_velocity_and_"
                "block_position_velocity_angle_angular_velocity"
            ),
        },
    }
    with tempfile.TemporaryDirectory(prefix="pusht-motion-damping-h3-", dir="/tmp") as temporary:
        root = Path(temporary) / output.name
        root.mkdir()
        request = write_portable_release_json(
            root / "request.json",
            request,
            release_root=output,
            frozen_predecessor_root=reuse_root,
        )
        reports = {}
        for split in SPLITS:
            if reuse_root is not None and split != "train":
                print(
                    f"Copying frozen {split}: {counts[split]} pairs",
                    flush=True,
                )
                source_table = reuse_root / f"{split}.lance"
                destination_table = root / f"{split}.lance"
                shutil.copytree(source_table, destination_table)
                reports[split] = copy.deepcopy(
                    reuse_manifest["splits"][split]
                )
                source_hash = directory_sha256(source_table)
                destination_hash = directory_sha256(destination_table)
                file_receipts = []
                for source_file in sorted(
                    path for path in source_table.rglob("*") if path.is_file()
                ):
                    relative = source_file.relative_to(source_table)
                    destination_file = destination_table / relative
                    source_file_hash = file_sha256(source_file)
                    destination_file_hash = file_sha256(destination_file)
                    if (
                        source_file.stat().st_size
                        != destination_file.stat().st_size
                        or source_file_hash != destination_file_hash
                    ):
                        raise RuntimeError(
                            f"Byte-for-byte file reuse failed: {relative}"
                        )
                    file_receipts.append(
                        {
                            "path": destination_file.relative_to(root).as_posix(),
                            "bytes": destination_file.stat().st_size,
                            "sha256": destination_file_hash,
                        }
                    )
                reports[split]["frozen_split_reuse"] = {
                    "source": predecessor_reference["source"],
                    "source_manifest_sha256": predecessor_reference[
                        "manifest_sha256"
                    ],
                    "source_table_sha256": source_hash,
                    "destination_table_sha256": destination_hash,
                    "pair_identity_preserved": True,
                    "model_visible_bytes_preserved": True,
                    "file_receipts": file_receipts,
                    "passed": source_hash == destination_hash,
                }
                if source_hash != destination_hash:
                    raise RuntimeError(
                        f"Byte-for-byte {split} table reuse failed"
                    )
            else:
                print(f"Building {split}: {counts[split]} pairs", flush=True)
                reports[split] = build_split(
                    root=root,
                    split=split,
                    pair_count=counts[split],
                    catalog_seed=seeds[split],
                    resolution=int(args.resolution),
                    jpeg_quality=int(args.jpeg_quality),
                )
            new_visible = model_visible_sha256(
                root / f"{split}.lance",
            )
            legacy_table = args.legacy_root.expanduser().resolve() / f"{split}.lance"
            if legacy_table.exists():
                old_visible = model_visible_sha256(
                    legacy_table,
                    episode_count=2 * counts[split],
                )
                identical = (
                    old_visible["row_count"] == new_visible["row_count"]
                    and old_visible["sha256"] == new_visible["sha256"]
                )
                reports[split]["legacy_model_visible_comparison"] = {
                    "source": "external_legacy_comparison",
                    "legacy_prefix": old_visible,
                    "strict_causal": new_visible,
                    "model_visible_hash_completely_identical": identical,
                    "checkpoint_retraining_required": not identical,
                }
            else:
                reports[split]["legacy_model_visible_comparison"] = {
                    "source": "external_legacy_comparison",
                    "status": "legacy_table_not_found",
                    "model_visible_hash_completely_identical": None,
                    "checkpoint_retraining_required": None,
                }
        cross_split = _cross_split(reports)
        x0_only_classifier = _x0_only_geometry_classifier(
            reports["train"], reports["validation"]
        )
        visible_comparisons = {
            split: reports[split]["legacy_model_visible_comparison"]
            for split in SPLITS
        }
        comparison_flags = [
            value["model_visible_hash_completely_identical"]
            for value in visible_comparisons.values()
        ]
        causal_audit = {
            "pair_count": int(sum(counts.values())),
            "state_installations_after_x0": 0,
            "query_simulator_recreated": False,
            "query_full_state_tolerance": QUERY_STATE_TOLERANCE,
            "query_full_state_dimensions": (
                "agent_position_velocity_angle_angular_velocity_and_"
                "block_position_velocity_angle_angular_velocity"
            ),
            "max_pair_full_state_gap": float(
                max(row["max_pair_full_state_gap"] for row in reports.values())
            ),
            "max_pair_query_pixel_difference": int(
                max(
                    row["max_pair_query_pixel_difference"]
                    for row in reports.values()
                )
            ),
            "max_pair_query_action_difference": float(
                max(
                    row["max_pair_query_action_difference"]
                    for row in reports.values()
                )
            ),
            "min_history_effect": float(
                min(row["min_history_effect"] for row in reports.values())
            ),
            "min_true_future_effect": float(
                min(row["min_true_future_effect"] for row in reports.values())
            ),
            "maximum_arbiter_count_from_x0_through_x3": int(
                max(
                    row["maximum_arbiter_count_from_x0_through_x3"]
                    for row in reports.values()
                )
            ),
            "all_split_x0_rgb_hash_multisets_identical_across_modes": all(
                row["x0_rgb_hash_multisets_identical_across_modes"]
                for row in reports.values()
            ),
            "maximum_x0_rgb_static_bayes_accuracy_upper_bound": float(
                max(
                    row["x0_rgb_static_bayes_accuracy_upper_bound"]
                    for row in reports.values()
                )
            ),
            "training_to_public_test_x0_only_geometry_classifier": (
                x0_only_classifier
            ),
            "legacy_model_visible_comparison": visible_comparisons,
            "model_visible_hash_completely_identical": (
                all(comparison_flags)
                if all(value is not None for value in comparison_flags)
                else None
            ),
            "checkpoint_retraining_required": (
                not all(comparison_flags)
                if all(value is not None for value in comparison_flags)
                else None
            ),
        }
        manifest = portable_release_metadata(
            {
                **request,
                "request_sha256": canonical_json_sha256(request),
                "splits": reports,
                "cross_split_audit": cross_split,
                "causal_audit": causal_audit,
                "passed": bool(
                    cross_split["passed"]
                    and x0_only_classifier["passed"]
                    and all(
                        report["passed"] for report in reports.values()
                    )
                ),
            },
            release_root=output,
            frozen_predecessor_root=reuse_root,
        )
        manifest_path = root / "manifest.json"
        manifest = write_portable_release_json(
            manifest_path,
            manifest,
            release_root=output,
            frozen_predecessor_root=reuse_root,
        )
        summary = {
            "protocol": PROTOCOL,
            "status": "passed" if manifest["passed"] else "failed",
            "root": str(output),
            "manifest": manifest_path.name,
            "manifest_sha256": file_sha256(manifest_path),
            "pair_counts": counts,
            "cross_split_audit": cross_split,
            "causal_audit": causal_audit,
            "split_metrics": {
                split: {
                    key: reports[split][key]
                    for key in (
                        "pair_count",
                        "state_installations_after_x0",
                        "query_simulator_recreated",
                        "query_full_state_tolerance",
                        "max_pair_full_state_gap",
                        "max_pair_query_pixel_difference",
                        "max_pair_query_action_difference",
                        "min_history_effect",
                        "min_true_future_effect",
                        "maximum_arbiter_count_from_x0_through_x3",
                        "x0_rgb_hash_multisets_identical_across_modes",
                        "x0_rgb_static_bayes_accuracy_upper_bound",
                    )
                }
                for split in SPLITS
            },
            "passed": manifest["passed"],
        }
        summary = write_portable_release_json(
            root / "build_report.json",
            summary,
            release_root=output,
            frozen_predecessor_root=reuse_root,
        )
        shutil.copytree(root, output)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
