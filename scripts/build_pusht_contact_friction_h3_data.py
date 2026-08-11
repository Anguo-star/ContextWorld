#!/usr/bin/env python3
"""Build formal PushT History-3 contact-friction benchmark data."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
from io import BytesIO
import json
import multiprocessing
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

from contextworld.evaluation.pusht_contact_friction_h3 import (  # noqa: E402
    ENDPOINT_MODES,
    FRICTION_VALUES,
    array_sha256,
    make_contact_friction_catalog_template,
    make_stratified_contact_friction_training_template,
    simulate_contact_friction_clip,
    stratified_contact_friction_training_coordinates,
    validate_contact_friction_pair,
)
from contextworld.release_metadata import (  # noqa: E402
    frozen_predecessor_reference,
    portable_release_metadata,
    write_portable_release_json,
)


PROTOCOL = "pusht_contact_friction_history3_strict_continuous_v2"
SPLITS = ("train", "loader_validation", "validation")
DEFAULT_OUTPUT = (
    ROOT
    / "artifacts/synthesis/pusht_contact_friction_h3_strict_v2"
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
        relative = child.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_sha256(child).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def directory_file_receipts(path: Path) -> list[dict[str, Any]]:
    """Return a stable per-file receipt for one copied Lance directory."""

    return [
        {
            "path": child.relative_to(path).as_posix(),
            "bytes": child.stat().st_size,
            "sha256": file_sha256(child),
        }
        for child in sorted(value for value in path.rglob("*") if value.is_file())
    ]


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def safe_output_path(path: Path) -> Path:
    result = Path(os.path.abspath(path.expanduser()))
    if result.exists():
        raise FileExistsError(
            f"Output already exists; refusing to overwrite: {result}"
        )
    result.parent.mkdir(parents=True, exist_ok=True)
    return result


def _failure_key(audit: dict[str, Any]) -> str:
    failed = [
        name for name, passed in audit["checks"].items() if not passed
    ]
    return ",".join(failed) or "unknown"


def _model_visible_pair_sha256(
    low: dict[str, Any],
    high: dict[str, Any],
) -> str:
    """Hash exactly the History/query pixels and actions seen by a model."""

    digest = hashlib.sha256()
    for mode, rollout in (("low", low), ("high", high)):
        digest.update(mode.encode("ascii"))
        for value in (
            np.asarray(rollout["model_pixels"][:3], dtype=np.uint8),
            np.asarray(rollout["action_blocks"][:3], dtype=np.float32),
        ):
            digest.update(array_sha256(value).encode("ascii"))
    return digest.hexdigest()


def _simulate_stratified_training_pair(
    *,
    pair_index: int,
    catalog_seed: int,
    resolution: int,
    maximum_attempts: int,
) -> dict[str, Any]:
    """Worker-safe strict simulation for one fixed Training stratum."""

    failures: dict[str, int] = {}
    for attempt_index in range(maximum_attempts):
        template = make_stratified_contact_friction_training_template(
            pair_index=pair_index,
            attempt_index=attempt_index,
            catalog_seed=catalog_seed,
        )
        try:
            low = simulate_contact_friction_clip(
                template,
                mode=ENDPOINT_MODES[0],
                resolution=resolution,
            )
            high = simulate_contact_friction_clip(
                template,
                mode=ENDPOINT_MODES[1],
                resolution=resolution,
            )
            audit = validate_contact_friction_pair(low, high)
        except (AssertionError, RuntimeError, ValueError) as error:
            key = f"exception:{type(error).__name__}"
            failures[key] = failures.get(key, 0) + 1
            continue
        if not audit["passed"]:
            key = _failure_key(audit)
            failures[key] = failures.get(key, 0) + 1
            continue
        return {
            "pair_index": int(pair_index),
            "accepted_attempt_index": int(attempt_index),
            "attempts": int(attempt_index + 1),
            "failure_counts": failures,
            "template": template,
            "low": low,
            "high": high,
            "audit": audit,
            "stratum": stratified_contact_friction_training_coordinates(
                pair_index
            ),
        }
    return {
        "pair_index": int(pair_index),
        "attempts": int(maximum_attempts),
        "failure_counts": failures,
        "error": "maximum_attempts_exhausted",
    }


def _episode_rows(
    rollout: dict[str, Any],
    *,
    split: str,
    catalog_index: int,
) -> dict[str, list[Any]]:
    rows = {key: list(value) for key, value in rollout["rows"].items()}
    row_count = len(rows["pixels"])
    rows["split"] = [split] * row_count
    rows["synthesis_version"] = [
        "contact_friction_h3_strict_continuous_v2"
    ] * row_count
    rows["catalog_index"] = [
        np.asarray([catalog_index], dtype=np.float32)
        for _ in range(row_count)
    ]
    return rows


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
        pa.field(
            "hidden_contact_friction",
            pa.list_(pa.float32(), 1),
        ),
        pa.field("pair_id", pa.string()),
        pa.field("hidden_mode", pa.string()),
        pa.field("split", pa.string()),
        pa.field("synthesis_version", pa.string()),
        pa.field("catalog_index", pa.list_(pa.float32(), 1)),
    ]
)


def _fixed_size_list_array(
    values: list[Any],
    *,
    size: int,
) -> pa.FixedSizeListArray:
    flat = np.asarray(values, dtype=np.float32).reshape(-1, size)
    return pa.FixedSizeListArray.from_arrays(
        pa.array(flat.reshape(-1), type=pa.float32()),
        size,
    )


def _encode_rgb(value: np.ndarray, *, jpeg_quality: int) -> bytes:
    buffer = BytesIO()
    Image.fromarray(np.asarray(value, dtype=np.uint8)).save(
        buffer,
        format="JPEG",
        quality=jpeg_quality,
    )
    return buffer.getvalue()


def _episode_batch(
    rows: dict[str, list[Any]],
    *,
    episode_index: int,
    jpeg_quality: int,
) -> pa.RecordBatch:
    row_count = len(rows["pixels"])
    arrays: list[pa.Array] = [
        pa.array(
            np.full(row_count, episode_index, dtype=np.int32),
            type=pa.int32(),
        ),
        pa.array(np.arange(row_count, dtype=np.int32), type=pa.int32()),
        pa.array(
            [
                _encode_rgb(value, jpeg_quality=jpeg_quality)
                for value in rows["pixels"]
            ],
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
        ("hidden_contact_friction", 1),
    ):
        arrays.append(_fixed_size_list_array(rows[name], size=size))
    for name in (
        "pair_id",
        "hidden_mode",
        "split",
        "synthesis_version",
    ):
        arrays.append(pa.array(rows[name], type=pa.string()))
    arrays.append(_fixed_size_list_array(rows["catalog_index"], size=1))
    return pa.record_batch(arrays, schema=LANCE_SCHEMA)


def _write_lance_episodes(
    path: Path,
    episodes: Iterator[dict[str, list[Any]]],
    *,
    jpeg_quality: int,
) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite Lance table: {path}")

    def batches() -> Iterator[pa.RecordBatch]:
        for episode_index, rows in enumerate(episodes):
            yield _episode_batch(
                rows,
                episode_index=episode_index,
                jpeg_quality=jpeg_quality,
            )

    reader = pa.RecordBatchReader.from_batches(LANCE_SCHEMA, batches())
    lance.write_dataset(reader, str(path), mode="create")


def _orientation_bin(template: dict[str, Any]) -> int:
    angle = float(template["reset_state"][4]) % (2 * np.pi)
    return int(np.floor(angle / (2 * np.pi) * 8)) % 8


def _position_bin(template: dict[str, Any]) -> str:
    snapshot = np.asarray(
        template["canonical_query_snapshot"],
        dtype=np.float64,
    )
    x_bin = int(np.clip(np.floor((snapshot[6] - 220.0) / 40.0), 0, 4))
    y_bin = int(np.clip(np.floor((snapshot[7] - 220.0) / 40.0), 0, 4))
    return f"x{x_bin}-y{y_bin}"


def build_split(
    *,
    root: Path,
    split: str,
    pair_count: int,
    catalog_seed: int,
    resolution: int,
    jpeg_quality: int,
    maximum_attempts_per_pair: int,
    stratified_training: bool = False,
    synthesis_workers: int = 1,
) -> dict[str, Any]:
    if split not in SPLITS:
        raise ValueError(f"Unknown split {split!r}")
    table_path = root / f"{split}.lance"
    pair_reports: list[dict[str, Any]] = []
    query_hashes: set[str] = set()
    action_hashes: set[str] = set()
    model_visible_pair_hashes: list[str] = []
    template_ids: set[str] = set()
    accepted_catalog_indices: list[int] = []
    failure_counts: dict[str, int] = {}
    stratum_counts: dict[str, int] = {}
    catalog_index = 0
    attempts_total = 0
    started = time.monotonic()

    def stratified_episodes() -> Iterator[dict[str, list[Any]]]:
        nonlocal attempts_total
        worker_count = max(1, int(synthesis_workers))
        pending_limit = max(worker_count, 2 * worker_count)
        # Lance is imported by this module but is intentionally confined to
        # the parent writer.  Spawn clean simulator workers so no inherited
        # Lance runtime state can cross the process boundary.
        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=multiprocessing.get_context("spawn"),
        ) as executor:
            futures = {
                pair_index: executor.submit(
                    _simulate_stratified_training_pair,
                    pair_index=pair_index,
                    catalog_seed=catalog_seed,
                    resolution=resolution,
                    maximum_attempts=maximum_attempts_per_pair,
                )
                for pair_index in range(min(pair_count, pending_limit))
            }
            next_submit = len(futures)
            for pair_index in range(pair_count):
                result = futures.pop(pair_index).result()
                if next_submit < pair_count:
                    futures[next_submit] = executor.submit(
                        _simulate_stratified_training_pair,
                        pair_index=next_submit,
                        catalog_seed=catalog_seed,
                        resolution=resolution,
                        maximum_attempts=maximum_attempts_per_pair,
                    )
                    next_submit += 1
                attempts_total += int(result["attempts"])
                for key, count in result["failure_counts"].items():
                    failure_counts[key] = failure_counts.get(key, 0) + int(
                        count
                    )
                if "error" in result:
                    raise RuntimeError(
                        f"Could not build train pair {pair_index}; "
                        f"failures={result['failure_counts']}"
                    )
                low = result["low"]
                high = result["high"]
                audit = result["audit"]
                template = result["template"]
                stratum = result["stratum"]
                candidate_index = (
                    pair_index * maximum_attempts_per_pair
                    + int(result["accepted_attempt_index"])
                )
                query_hash = audit["hashes"]["query_pixels"]
                if query_hash in query_hashes:
                    raise RuntimeError(
                        "Stratified Training produced duplicate query pixels"
                    )
                if template.template_id in template_ids:
                    raise RuntimeError(
                        f"Duplicate template id {template.template_id}"
                    )
                query_hashes.add(query_hash)
                action_hashes.add(audit["hashes"]["raw_actions"])
                template_ids.add(template.template_id)
                accepted_catalog_indices.append(candidate_index)
                stratum_key = (
                    f"f{stratum['family_id']}-"
                    f"a{stratum['angle_bin']:02d}-"
                    f"x{stratum['translation_x_bin']}-"
                    f"y{stratum['translation_y_bin']}"
                )
                stratum_counts[stratum_key] = (
                    stratum_counts.get(stratum_key, 0) + 1
                )
                pair_reports.append(
                    {
                        "pair_index": pair_index,
                        "catalog_index": candidate_index,
                        "template": asdict(template),
                        "orientation_bin": _orientation_bin(
                            asdict(template)
                        ),
                        "position_bin": _position_bin(asdict(template)),
                        "audit": audit,
                        "model_visible_pair_sha256": (
                            _model_visible_pair_sha256(low, high)
                        ),
                        "training_stratum": stratum,
                    }
                )
                model_visible_pair_hashes.append(
                    pair_reports[-1]["model_visible_pair_sha256"]
                )
                if (
                    len(pair_reports) <= 4
                    or len(pair_reports) % 64 == 0
                    or len(pair_reports) == pair_count
                ):
                    elapsed = time.monotonic() - started
                    print(
                        f"  {split}: {len(pair_reports)}/{pair_count} "
                        f"pairs, attempts={attempts_total}, "
                        f"elapsed={elapsed:.1f}s",
                        flush=True,
                    )
                yield _episode_rows(
                    low,
                    split=split,
                    catalog_index=candidate_index,
                )
                yield _episode_rows(
                    high,
                    split=split,
                    catalog_index=candidate_index,
                )

    def episodes() -> Iterator[dict[str, list[Any]]]:
        nonlocal catalog_index, attempts_total
        if stratified_training:
            yield from stratified_episodes()
            return
        for pair_index in range(pair_count):
            accepted = False
            for attempt_index in range(maximum_attempts_per_pair):
                candidate_index = catalog_index
                catalog_index += 1
                attempts_total += 1
                if stratified_training:
                    if split != "train":
                        raise ValueError(
                            "Stratified construction is Training-only"
                        )
                    template = (
                        make_stratified_contact_friction_training_template(
                            pair_index=pair_index,
                            attempt_index=attempt_index,
                            catalog_seed=catalog_seed,
                        )
                    )
                    stratum = (
                        stratified_contact_friction_training_coordinates(
                            pair_index
                        )
                    )
                else:
                    template = make_contact_friction_catalog_template(
                        split=split,
                        catalog_index=candidate_index,
                        catalog_seed=catalog_seed,
                    )
                    stratum = None
                if attempts_total <= 4:
                    print(
                        f"  {split}: simulating candidate "
                        f"{candidate_index}",
                        flush=True,
                    )
                try:
                    low = simulate_contact_friction_clip(
                        template,
                        mode=ENDPOINT_MODES[0],
                        resolution=resolution,
                    )
                    if attempts_total <= 4:
                        print(
                            f"  {split}: low-friction rollout complete",
                            flush=True,
                        )
                    high = simulate_contact_friction_clip(
                        template,
                        mode=ENDPOINT_MODES[1],
                        resolution=resolution,
                    )
                    if attempts_total <= 4:
                        print(
                            f"  {split}: high-friction rollout complete",
                            flush=True,
                        )
                    audit = validate_contact_friction_pair(low, high)
                except (AssertionError, RuntimeError, ValueError) as error:
                    key = f"exception:{type(error).__name__}"
                    failure_counts[key] = failure_counts.get(key, 0) + 1
                    continue
                if not audit["passed"]:
                    key = _failure_key(audit)
                    failure_counts[key] = failure_counts.get(key, 0) + 1
                    continue
                query_hash = audit["hashes"]["query_pixels"]
                if query_hash in query_hashes:
                    failure_counts["duplicate_query_pixels"] = (
                        failure_counts.get("duplicate_query_pixels", 0) + 1
                    )
                    continue
                template_id = template.template_id
                if template_id in template_ids:
                    raise RuntimeError(f"Duplicate template id {template_id}")

                query_hashes.add(query_hash)
                action_hashes.add(audit["hashes"]["raw_actions"])
                template_ids.add(template_id)
                accepted_catalog_indices.append(candidate_index)
                pair_reports.append(
                    {
                        "pair_index": pair_index,
                        "catalog_index": candidate_index,
                        "template": asdict(template),
                        "orientation_bin": _orientation_bin(
                            asdict(template)
                        ),
                        "position_bin": _position_bin(asdict(template)),
                        "audit": audit,
                        "model_visible_pair_sha256": (
                            _model_visible_pair_sha256(low, high)
                        ),
                        "training_stratum": stratum,
                    }
                )
                if stratum is not None:
                    stratum_key = (
                        f"f{stratum['family_id']}-"
                        f"a{stratum['angle_bin']:02d}-"
                        f"x{stratum['translation_x_bin']}-"
                        f"y{stratum['translation_y_bin']}"
                    )
                    stratum_counts[stratum_key] = (
                        stratum_counts.get(stratum_key, 0) + 1
                    )
                model_visible_pair_hashes.append(
                    pair_reports[-1]["model_visible_pair_sha256"]
                )
                yield _episode_rows(
                    low,
                    split=split,
                    catalog_index=candidate_index,
                )
                yield _episode_rows(
                    high,
                    split=split,
                    catalog_index=candidate_index,
                )
                accepted = True
                if (
                    len(pair_reports) <= 4
                    or len(pair_reports) % 64 == 0
                    or len(pair_reports) == pair_count
                ):
                    elapsed = time.monotonic() - started
                    print(
                        f"  {split}: {len(pair_reports)}/{pair_count} "
                        f"pairs, attempts={attempts_total}, "
                        f"elapsed={elapsed:.1f}s",
                        flush=True,
                    )
                break
            if not accepted:
                raise RuntimeError(
                    f"Could not build {split} pair {pair_index}; "
                    f"failures={failure_counts}"
                )

    _write_lance_episodes(
        table_path,
        episodes(),
        jpeg_quality=jpeg_quality,
    )

    if len(pair_reports) != pair_count:
        raise RuntimeError(
            f"Expected {pair_count} {split} pairs, got "
            f"{len(pair_reports)}"
        )
    natural_query_residuals = [
        value
        for pair in pair_reports
        for value in pair["audit"][
            "query_precanonical_correction"
        ].values()
    ]
    history_gaps = [
        pair["audit"]["history_visible_response_gap"]["px_equivalent"]
        for pair in pair_reports
    ]
    future_position_gaps = [
        pair["audit"]["future_gap"]["block_position_px"]
        for pair in pair_reports
    ]
    query_state_gaps = [
        pair["audit"]["query_physics_max_abs_gap"]
        for pair in pair_reports
    ]
    query_pixel_differences = [
        pair["audit"]["query_pixel_max_abs_difference"]
        for pair in pair_reports
    ]
    query_action_differences = [
        pair["audit"]["query_action_max_abs_difference"]
        for pair in pair_reports
    ]
    clean_replay_gaps = [
        endpoint["future_full_state_max_abs_gap"]
        for pair in pair_reports
        for endpoint in pair["audit"]["clean_simulator_replay"].values()
    ]
    cache_clear_steps = [
        value
        for pair in pair_reports
        for value in pair["audit"][
            "trailing_no_contact_steps_before_query"
        ].values()
    ]
    orientation_counts = {
        str(index): sum(
            pair["orientation_bin"] == index for pair in pair_reports
        )
        for index in range(8)
    }
    position_counts: dict[str, int] = {}
    for pair in pair_reports:
        key = pair["position_bin"]
        position_counts[key] = position_counts.get(key, 0) + 1
    family_counts: dict[str, int] = {}
    offset_order_counts = {
        "low_offset_larger": 0,
        "high_offset_larger": 0,
        "equal": 0,
    }
    offset_direction_bin_counts = {str(index): 0 for index in range(8)}
    for pair in pair_reports:
        template = pair["template"]
        family_key = str(template["strict_family_id"])
        family_counts[family_key] = family_counts.get(family_key, 0) + 1
        nominal = np.asarray(template["reset_state"], dtype=np.float64)
        low_reset = np.asarray(
            template["low_friction_reset_state"], dtype=np.float64
        )
        high_reset = np.asarray(
            template["high_friction_reset_state"], dtype=np.float64
        )
        low_magnitude = float(np.linalg.norm(low_reset[2:4] - nominal[2:4]))
        high_magnitude = float(
            np.linalg.norm(high_reset[2:4] - nominal[2:4])
        )
        if np.isclose(low_magnitude, high_magnitude, atol=1.0e-12):
            offset_order_counts["equal"] += 1
        elif low_magnitude > high_magnitude:
            offset_order_counts["low_offset_larger"] += 1
        else:
            offset_order_counts["high_offset_larger"] += 1
        direction = low_reset[2:4] - high_reset[2:4]
        direction_angle = float(np.arctan2(direction[1], direction[0]))
        direction_bin = int(
            np.floor((direction_angle % (2 * np.pi)) / (2 * np.pi) * 8)
        ) % 8
        offset_direction_bin_counts[str(direction_bin)] += 1
    table_files = [
        path for path in table_path.rglob("*") if path.is_file()
    ]
    orientation_coverage_required = pair_count >= 64
    orientation_coverage_passed = all(
        count > 0 for count in orientation_counts.values()
    )
    family_coverage_required = pair_count >= 64
    family_coverage_passed = all(
        family_counts.get(str(index), 0) > 0 for index in range(2)
    )
    return {
        "split": split,
        "pair_count": pair_count,
        "episode_count": 2 * pair_count,
        "raw_rows": 2 * pair_count * 20,
        "table_path": table_path.name,
        "table_files": len(table_files),
        "table_bytes": sum(path.stat().st_size for path in table_files),
        "table_sha256": directory_sha256(table_path),
        "catalog_seed": int(catalog_seed),
        "catalog_attempts": attempts_total,
        "accepted_catalog_indices_sha256": array_sha256(
            np.asarray(accepted_catalog_indices, dtype=np.int64)
        ),
        "failure_counts": failure_counts,
        "query_hash_count": len(query_hashes),
        "query_hashes": sorted(query_hashes),
        "action_hash_count": len(action_hashes),
        "action_hashes": sorted(action_hashes),
        "model_visible_pair_hashes_sha256": array_sha256(
            np.asarray(model_visible_pair_hashes, dtype="S64")
        ),
        "template_ids": sorted(template_ids),
        "orientation_bin_counts": orientation_counts,
        "orientation_coverage_required": orientation_coverage_required,
        "orientation_coverage_passed": orientation_coverage_passed,
        "position_bin_counts": dict(sorted(position_counts.items())),
        "strict_family_counts": dict(sorted(family_counts.items())),
        "strict_family_coverage_required": family_coverage_required,
        "strict_family_coverage_passed": family_coverage_passed,
        "stratified_training": bool(stratified_training),
        "training_stratum_counts": dict(sorted(stratum_counts.items())),
        "training_stratum_count": len(stratum_counts),
        "training_stratum_minimum_count": (
            min(stratum_counts.values()) if stratum_counts else None
        ),
        "training_stratum_maximum_count": (
            max(stratum_counts.values()) if stratum_counts else None
        ),
        "training_strata_balanced": (
            len(set(stratum_counts.values())) == 1
            if stratum_counts
            else None
        ),
        "initial_offset_order_counts": offset_order_counts,
        "initial_offset_direction_bin_counts": (
            offset_direction_bin_counts
        ),
        "minimum_history_gap_px_equivalent": float(min(history_gaps)),
        "maximum_natural_query_target_residual": float(
            max(natural_query_residuals)
        ),
        "state_installations_after_x0": int(
            max(
                pair["audit"]["state_installations_after_x0"]
                for pair in pair_reports
            )
        ),
        "query_simulator_recreated": bool(
            any(
                pair["audit"]["query_simulator_recreated"]
                for pair in pair_reports
            )
        ),
        "max_pair_full_state_gap": float(max(query_state_gaps)),
        "max_pair_query_pixel_difference": int(
            max(query_pixel_differences)
        ),
        "max_pair_query_action_difference": float(
            max(query_action_differences)
        ),
        "min_history_effect": float(min(history_gaps)),
        "min_true_future_effect": float(min(future_position_gaps)),
        "minimum_cache_clear_steps_before_query": int(
            min(cache_clear_steps)
        ),
        "maximum_clean_simulator_replay_full_state_gap": float(
            max(clean_replay_gaps)
        ),
        "all_paired_x0_pixels_bitwise_identical": bool(
            all(
                pair["audit"]["checks"]["initial_pixels_identical"]
                for pair in pair_reports
            )
        ),
        "mode_label_static_x0_accuracy": 0.5,
        "swapped_history_scoring_required": True,
        "minimum_future_block_position_gap_px": float(
            min(future_position_gaps)
        ),
        "pairs": pair_reports,
        "passed": bool(
            all(pair["audit"]["passed"] for pair in pair_reports)
            and (
                not orientation_coverage_required
                or orientation_coverage_passed
            )
            and (
                not family_coverage_required or family_coverage_passed
            )
            and max(query_state_gaps) <= 1.0e-5
            and max(query_pixel_differences) == 0
            and max(query_action_differences) == 0.0
            and min(cache_clear_steps) >= 3
            and max(clean_replay_gaps) <= 1.0e-5
            and (
                not stratified_training
                or (
                    len(stratum_counts) == 2048
                    and len(set(stratum_counts.values())) == 1
                )
            )
        ),
    }


def reuse_evaluation_split(
    *,
    source_root: Path,
    destination_root: Path,
    split: str,
    expected_pairs: int,
) -> dict[str, Any]:
    """Copy one frozen Development/Public split byte for byte."""

    if split not in {"loader_validation", "validation"}:
        raise ValueError("Only evaluation splits may be reused")
    source_manifest_path = source_root / "manifest.json"
    source_manifest = json.loads(
        source_manifest_path.read_text(encoding="utf-8")
    )
    source_report = source_manifest["splits"][split]
    if int(source_report["pair_count"]) != int(expected_pairs):
        raise RuntimeError(
            f"Frozen {split} pair count changed: "
            f"expected={expected_pairs}, "
            f"source={source_report['pair_count']}"
        )
    source_table = source_root / source_report["table_path"]
    destination_table = destination_root / source_report["table_path"]
    before = directory_file_receipts(source_table)
    shutil.copytree(source_table, destination_table, copy_function=shutil.copy2)
    after = directory_file_receipts(destination_table)
    if before != after:
        raise RuntimeError(f"Byte-for-byte reuse failed for {split}")
    if directory_sha256(destination_table) != source_report["table_sha256"]:
        raise RuntimeError(f"Directory hash changed while copying {split}")
    report = json.loads(json.dumps(source_report))
    predecessor = frozen_predecessor_reference(
        manifest_sha256=file_sha256(source_manifest_path),
        table_sha256={split: source_report["table_sha256"]},
    )
    report["frozen_split_reuse"] = {
        "source": predecessor["source"],
        "source_manifest_sha256": predecessor["manifest_sha256"],
        "source_table_sha256": predecessor["table_sha256"][split],
        "destination_table_sha256": directory_sha256(destination_table),
        "file_receipts": after,
        "pair_identity_preserved": True,
        "model_visible_bytes_preserved": True,
        "passed": True,
    }
    return report


def _cross_split_audit(
    reports: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    query_sets = {
        split: set(report["query_hashes"])
        for split, report in reports.items()
    }
    template_sets = {
        split: set(report["template_ids"])
        for split, report in reports.items()
    }
    overlaps: dict[str, dict[str, int]] = {}
    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1 :]:
            key = f"{left}__{right}"
            overlaps[key] = {
                "query_pixel_hashes": len(
                    query_sets[left] & query_sets[right]
                ),
                "template_ids": len(
                    template_sets[left] & template_sets[right]
                ),
            }
    passed = all(
        value == 0
        for overlap in overlaps.values()
        for value in overlap.values()
    )
    return {
        "overlap_counts": overlaps,
        "split_specific_catalog_seeds": {
            split: reports[split]["catalog_seed"] for split in SPLITS
        },
        "passed": passed,
    }


def training_coverage_comparison(
    *,
    expanded: dict[str, Any],
    source: dict[str, Any],
) -> dict[str, Any]:
    """Compare expanded stratified Training with its prior 2,048-pair set."""

    def count_summary(
        values: dict[str, int],
    ) -> dict[str, Any]:
        counts = [int(value) for value in values.values()]
        return {
            "occupied_bins": len(counts),
            "minimum_count": min(counts) if counts else 0,
            "maximum_count": max(counts) if counts else 0,
            "total_count": sum(counts),
        }

    source_pairs = int(source["pair_count"])
    expanded_pairs = int(expanded["pair_count"])
    dimensions = {
        "strict_family": {
            "source": count_summary(source["strict_family_counts"]),
            "expanded": count_summary(expanded["strict_family_counts"]),
        },
        "orientation_8_bins": {
            "source": count_summary(source["orientation_bin_counts"]),
            "expanded": count_summary(expanded["orientation_bin_counts"]),
        },
        "rendered_position_bins": {
            "source": count_summary(source["position_bin_counts"]),
            "expanded": count_summary(expanded["position_bin_counts"]),
        },
    }
    source_query = set(source["query_hashes"])
    expanded_query = set(expanded["query_hashes"])
    source_templates = set(source["template_ids"])
    expanded_templates = set(expanded["template_ids"])
    complete_strata = 2 * 16 * 8 * 8
    passed = bool(
        expanded_pairs == 4 * source_pairs
        and expanded.get("stratified_training") is True
        and int(expanded.get("training_stratum_count", 0))
        == complete_strata
        and expanded.get("training_strata_balanced") is True
        and int(expanded.get("training_stratum_minimum_count", 0)) == 4
        and int(expanded.get("training_stratum_maximum_count", 0)) == 4
        and not (source_query & expanded_query)
        and not (source_templates & expanded_templates)
        and all(
            row["expanded"]["occupied_bins"]
            >= row["source"]["occupied_bins"]
            for row in dimensions.values()
        )
    )
    return {
        "source_pair_count": source_pairs,
        "expanded_pair_count": expanded_pairs,
        "pair_count_multiplier": expanded_pairs / source_pairs,
        "stratification": {
            "physics_families": 2,
            "angle_bins": 16,
            "translation_x_bins": 8,
            "translation_y_bins": 8,
            "complete_strata": complete_strata,
            "observed_strata": int(
                expanded.get("training_stratum_count", 0)
            ),
            "pairs_per_stratum_minimum": expanded.get(
                "training_stratum_minimum_count"
            ),
            "pairs_per_stratum_maximum": expanded.get(
                "training_stratum_maximum_count"
            ),
            "balanced": expanded.get("training_strata_balanced"),
        },
        "distribution_dimensions": dimensions,
        "source_expanded_query_hash_overlap": len(
            source_query & expanded_query
        ),
        "source_expanded_template_id_overlap": len(
            source_templates & expanded_templates
        ),
        "passed": passed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-pairs", type=int, default=2048)
    parser.add_argument(
        "--loader-validation-pairs",
        type=int,
        default=256,
    )
    parser.add_argument("--validation-pairs", type=int, default=256)
    parser.add_argument("--catalog-seed", type=int, default=20260801)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument(
        "--maximum-attempts-per-pair",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--stratified-training",
        action="store_true",
        help=(
            "Balance Training over both physics families, 16 angle bins, "
            "and an 8x8 translation grid."
        ),
    )
    parser.add_argument(
        "--reuse-evaluation-from",
        type=Path,
        default=None,
        help=(
            "Copy loader_validation and validation Lance tables and pair "
            "identities byte for byte from an audited prior release."
        ),
    )
    parser.add_argument(
        "--synthesis-workers",
        type=int,
        default=1,
        help="Parallel strict simulators used only for stratified Training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = safe_output_path(args.output)
    pair_counts = {
        "train": int(args.train_pairs),
        "loader_validation": int(args.loader_validation_pairs),
        "validation": int(args.validation_pairs),
    }
    if any(value <= 0 for value in pair_counts.values()):
        raise ValueError("Every split must contain at least one pair")
    if args.stratified_training and pair_counts["train"] % 2048:
        raise ValueError(
            "Stratified Training pair count must be a multiple of 2,048"
        )
    if int(args.resolution) <= 0:
        raise ValueError("resolution must be positive")
    if int(args.synthesis_workers) <= 0:
        raise ValueError("synthesis_workers must be positive")
    reuse_evaluation_from = (
        None
        if args.reuse_evaluation_from is None
        else args.reuse_evaluation_from.expanduser().resolve()
    )
    reuse_manifest = None
    if reuse_evaluation_from is not None:
        required_reuse_inputs = [
            reuse_evaluation_from / "manifest.json",
            reuse_evaluation_from / "loader_validation.lance",
            reuse_evaluation_from / "validation.lance",
        ]
        missing = [path for path in required_reuse_inputs if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Missing frozen evaluation reuse input(s):\n"
                + "\n".join(map(str, missing))
            )
        reuse_manifest = json.loads(
            (reuse_evaluation_from / "manifest.json").read_text(
                encoding="utf-8"
            )
        )
    catalog_seeds = {
        split: int(
            np.random.SeedSequence(
                [int(args.catalog_seed), offset]
            ).generate_state(1)[0]
        )
        for split, offset in zip(SPLITS, (11, 13, 17), strict=True)
    }
    request = {
        "protocol": PROTOCOL,
        "history_tokens": 3,
        "raw_steps_per_action_block": 5,
        "pair_counts": pair_counts,
        "friction_values": {
            mode: FRICTION_VALUES[mode] for mode in ENDPOINT_MODES
        },
        "catalog_seed": int(args.catalog_seed),
        "split_catalog_seeds": catalog_seeds,
        "resolution": int(args.resolution),
        "jpeg_quality": int(args.jpeg_quality),
        "maximum_attempts_per_pair": int(
            args.maximum_attempts_per_pair
        ),
        "stratified_training": bool(args.stratified_training),
        "synthesis_workers": int(args.synthesis_workers),
        "evaluation_split_policy": (
            "byte_for_byte_reuse"
            if reuse_evaluation_from is not None
            else "fresh_generation"
        ),
        "evaluation_reuse_source": (
            None
            if reuse_evaluation_from is None
            else frozen_predecessor_reference(
                manifest_sha256=file_sha256(
                    reuse_evaluation_from / "manifest.json"
                ),
                table_sha256={
                    split: reuse_manifest["splits"][split][
                        "table_sha256"
                    ]
                    for split in ("loader_validation", "validation")
                },
            )
        ),
        "sealed_test_included": False,
        "causal_chain": {
            "state_installations_after_x0": 0,
            "query_simulator_recreated": False,
            "query_full_state_tolerance": 1.0e-5,
            "minimum_contact_free_steps_before_query": 3,
            "paired_x0_pixels_must_be_bitwise_identical": True,
            "clean_simulator_replay_is_diagnostic_only": True,
        },
    }

    with tempfile.TemporaryDirectory(
        prefix="pusht-contact-friction-h3-",
        dir="/tmp",
    ) as temporary:
        root = Path(temporary) / output.name
        root.mkdir()
        request = write_portable_release_json(
            root / "request.json",
            request,
            release_root=output,
            frozen_predecessor_root=reuse_evaluation_from,
        )
        reports: dict[str, dict[str, Any]] = {}
        for split in SPLITS:
            print(
                f"Building {split}: {pair_counts[split]} pairs",
                flush=True,
            )
            if reuse_evaluation_from is not None and split != "train":
                reports[split] = reuse_evaluation_split(
                    source_root=reuse_evaluation_from,
                    destination_root=root,
                    split=split,
                    expected_pairs=pair_counts[split],
                )
            else:
                reports[split] = build_split(
                    root=root,
                    split=split,
                    pair_count=pair_counts[split],
                    catalog_seed=catalog_seeds[split],
                    resolution=int(args.resolution),
                    jpeg_quality=int(args.jpeg_quality),
                    maximum_attempts_per_pair=int(
                        args.maximum_attempts_per_pair
                    ),
                    stratified_training=(
                        bool(args.stratified_training)
                        and split == "train"
                    ),
                    synthesis_workers=int(args.synthesis_workers),
                )
        cross_split = _cross_split_audit(reports)
        if not cross_split["passed"]:
            raise RuntimeError(
                f"Cross-split isolation failed: {cross_split}"
            )
        training_coverage = None
        if reuse_evaluation_from is not None:
            training_coverage = training_coverage_comparison(
                expanded=reports["train"],
                source=reuse_manifest["splits"]["train"],
            )
            if not training_coverage["passed"]:
                raise RuntimeError(
                    "Expanded Training coverage audit failed: "
                    f"{training_coverage}"
                )
        manifest = portable_release_metadata(
            {
                **request,
                "request_sha256": canonical_json_sha256(request),
                "splits": reports,
                "cross_split_audit": cross_split,
                "training_coverage_vs_reused_release": training_coverage,
                "passed": bool(
                    cross_split["passed"]
                    and (
                        training_coverage is None
                        or training_coverage["passed"]
                    )
                    and all(
                        report["passed"] for report in reports.values()
                    )
                ),
            },
            release_root=output,
            frozen_predecessor_root=reuse_evaluation_from,
        )
        manifest_path = root / "manifest.json"
        manifest = write_portable_release_json(
            manifest_path,
            manifest,
            release_root=output,
            frozen_predecessor_root=reuse_evaluation_from,
        )
        summary = {
            "protocol": PROTOCOL,
            "status": "passed" if manifest["passed"] else "failed",
            "root": str(output),
            "manifest": manifest_path.name,
            "manifest_sha256": file_sha256(manifest_path),
            "pair_counts": pair_counts,
            "cross_split_audit": cross_split,
            "training_coverage_vs_reused_release": training_coverage,
            "split_metrics": {
                split: {
                    key: reports[split][key]
                    for key in (
                        "minimum_history_gap_px_equivalent",
                        "maximum_natural_query_target_residual",
                        "max_pair_full_state_gap",
                        "max_pair_query_pixel_difference",
                        "max_pair_query_action_difference",
                        "state_installations_after_x0",
                        "query_simulator_recreated",
                        "minimum_cache_clear_steps_before_query",
                        "maximum_clean_simulator_replay_full_state_gap",
                        "minimum_future_block_position_gap_px",
                        "model_visible_pair_hashes_sha256",
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
            frozen_predecessor_root=reuse_evaluation_from,
        )
        shutil.copytree(root, output)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
