#!/usr/bin/env python3
"""Build the frozen History-3 TwoRoom portal-exit data release."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
from io import BytesIO
import json
from pathlib import Path
import sys
import time
from typing import Any, Iterator

import lance
import numpy as np
import pyarrow as pa
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contextworld.evaluation.portal_exit_h3 import (  # noqa: E402
    EXIT_MODES,
    make_template,
    simulate_portal_exit_episode,
    validate_portal_exit_episode_pair,
)


PROTOCOL = "tworoom_portal_exit_history3_release_v1"
SPLITS = ("train", "loader_validation", "validation")
DEFAULT_OUTPUT = ROOT / "artifacts/synthesis/tworoom_portal_exit_h3_release_v1"


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


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


SCHEMA = pa.schema(
    [
        pa.field("episode_idx", pa.int32()),
        pa.field("step_idx", pa.int32()),
        pa.field("pixels", pa.binary()),
        pa.field("action", pa.list_(pa.float32(), 2)),
        pa.field("proprio", pa.list_(pa.float32(), 2)),
        pa.field("state", pa.list_(pa.float32(), 10)),
        pa.field("goal_state", pa.list_(pa.float32(), 2)),
        pa.field("hidden_portal_exit", pa.list_(pa.float32(), 1)),
        pa.field("pair_id", pa.string()),
        pa.field("hidden_mode", pa.string()),
        pa.field("split", pa.string()),
        pa.field("catalog_index", pa.list_(pa.float32(), 1)),
    ]
)


def _fixed(values: list[Any], size: int) -> pa.FixedSizeListArray:
    flat = np.asarray(values, dtype=np.float32).reshape(-1, size)
    return pa.FixedSizeListArray.from_arrays(
        pa.array(flat.reshape(-1), type=pa.float32()), size
    )


def _encode(value: np.ndarray, quality: int) -> bytes:
    buffer = BytesIO()
    Image.fromarray(np.asarray(value, dtype=np.uint8)).save(
        buffer, format="JPEG", quality=quality
    )
    return buffer.getvalue()


def _batch(
    rows: dict[str, list[Any]],
    *,
    episode_index: int,
    split: str,
    catalog_index: int,
    quality: int,
) -> pa.RecordBatch:
    count = len(rows["pixels"])
    arrays: list[pa.Array] = [
        pa.array(np.full(count, episode_index, dtype=np.int32)),
        pa.array(np.arange(count, dtype=np.int32)),
        pa.array([_encode(value, quality) for value in rows["pixels"]]),
        _fixed(rows["action"], 2),
        _fixed(rows["proprio"], 2),
        _fixed(rows["state"], 10),
        _fixed(rows["goal_state"], 2),
        _fixed(rows["hidden_portal_exit"], 1),
        pa.array(rows["pair_id"], type=pa.string()),
        pa.array(rows["hidden_mode"], type=pa.string()),
        pa.array([split] * count, type=pa.string()),
        _fixed(
            [np.asarray([catalog_index], dtype=np.float32)] * count,
            1,
        ),
    ]
    return pa.record_batch(arrays, schema=SCHEMA)


def build_split(
    root: Path,
    *,
    split: str,
    pair_count: int,
    catalog_seed: int,
    quality: int,
) -> dict[str, Any]:
    reports: list[dict[str, Any]] = []
    query_hashes: set[str] = set()
    table_path = root / f"{split}.lance"
    started = time.monotonic()

    def batches() -> Iterator[pa.RecordBatch]:
        episode_index = 0
        for index in range(pair_count):
            template = make_template(
                split=split, index=index, catalog_seed=catalog_seed
            )
            near = simulate_portal_exit_episode(template, mode=EXIT_MODES[0])
            farther = simulate_portal_exit_episode(template, mode=EXIT_MODES[1])
            audit = validate_portal_exit_episode_pair(near, farther)
            if not audit["passed"]:
                raise RuntimeError(f"Pair failed: {template.template_id}: {audit}")
            query_hash = audit["hashes"]["query_pixels"]
            if query_hash in query_hashes:
                raise RuntimeError(f"Duplicate query pixels in {split}: {template.template_id}")
            query_hashes.add(query_hash)
            reports.append(
                {
                    "pair_index": index,
                    "template": asdict(template),
                    "audit": audit,
                }
            )
            for rollout in (near, farther):
                yield _batch(
                    rollout["rows"],
                    episode_index=episode_index,
                    split=split,
                    catalog_index=index,
                    quality=quality,
                )
                episode_index += 1
            if index < 3 or (index + 1) % 128 == 0:
                print(
                    f"{split}: {index + 1}/{pair_count} pairs, "
                    f"elapsed={time.monotonic() - started:.1f}s",
                    flush=True,
                )

    lance.write_dataset(
        pa.RecordBatchReader.from_batches(SCHEMA, batches()),
        str(table_path),
        mode="create",
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
        "catalog_seed": catalog_seed,
        "query_hashes": sorted(query_hashes),
        "template_ids": [row["template"]["template_id"] for row in reports],
        "minimum_history_exit_gap_px": min(
            row["audit"]["middle_state_gap_px"] for row in reports
        ),
        "minimum_true_future_gap_px": min(
            row["audit"]["future_state_gap_px"] for row in reports
        ),
        "maximum_query_state_gap": max(
            row["audit"]["maximum_query_state_gap"] for row in reports
        ),
        "pairs": reports,
        "passed": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-pairs", type=int, default=2048)
    parser.add_argument("--development-pairs", type=int, default=256)
    parser.add_argument("--public-test-pairs", type=int, default=256)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.mkdir(parents=True)
    request = {
        "protocol": PROTOCOL,
        "pair_counts": {
            "train": args.train_pairs,
            "loader_validation": args.development_pairs,
            "validation": args.public_test_pairs,
        },
        "catalog_seeds": {
            "train": 202608011,
            "loader_validation": 202608012,
            "validation": 202608013,
        },
        "jpeg_quality": args.jpeg_quality,
    }
    (output / "request.json").write_text(
        json.dumps(request, indent=2, sort_keys=True), encoding="utf-8"
    )
    reports = {}
    for split in SPLITS:
        reports[split] = build_split(
            output,
            split=split,
            pair_count=int(request["pair_counts"][split]),
            catalog_seed=int(request["catalog_seeds"][split]),
            quality=args.jpeg_quality,
        )
    overlaps = {}
    for left, right in (
        ("train", "loader_validation"),
        ("train", "validation"),
        ("loader_validation", "validation"),
    ):
        overlaps[f"{left}_vs_{right}_query_hashes"] = len(
            set(reports[left]["query_hashes"]) & set(reports[right]["query_hashes"])
        )
        overlaps[f"{left}_vs_{right}_template_ids"] = len(
            set(reports[left]["template_ids"]) & set(reports[right]["template_ids"])
        )
    build_report = {
        "protocol": PROTOCOL,
        "request": request,
        "splits": reports,
        "cross_split_overlap": overlaps,
        "passed": all(row["passed"] for row in reports.values())
        and not any(overlaps.values()),
    }
    (output / "build_report.json").write_text(
        json.dumps(build_report, indent=2, sort_keys=True), encoding="utf-8"
    )
    files_before_manifest = [
        path for path in output.rglob("*") if path.is_file()
    ]
    manifest = {
        "protocol": PROTOCOL,
        "files": {
            path.relative_to(output).as_posix(): file_sha256(path)
            for path in sorted(files_before_manifest)
        },
        "file_count_without_manifest": len(files_before_manifest),
        "bytes_without_manifest": sum(path.stat().st_size for path in files_before_manifest),
        "build_passed": build_report["passed"],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({
        "output": str(output),
        "passed": build_report["passed"],
        "pair_counts": request["pair_counts"],
        "cross_split_overlap": overlaps,
        "tree_sha256": directory_sha256(output),
    }, indent=2))
    if not build_report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
