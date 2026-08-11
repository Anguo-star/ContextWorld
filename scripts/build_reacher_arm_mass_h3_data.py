#!/usr/bin/env python3
"""Build the frozen History-3 Reacher arm-mass data release."""

from __future__ import annotations

import argparse
import atexit
import hashlib
from io import BytesIO
import json
import multiprocessing as mp
import os
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
os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

from contextworld.evaluation.reacher_arm_mass_h3 import (  # noqa: E402
    ARM_DENSITIES,
    MASS_MODES,
    ReacherArmMassSimulator,
    make_candidate,
)
from contextworld.paths import artifact_path  # noqa: E402


PROTOCOL = "reacher_arm_mass_history3_release_v1"
SPLITS = ("train", "loader_validation", "validation")
DEFAULT_OUTPUT = artifact_path("synthesis/reacher_arm_mass_h3_release_v1")


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
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _fixed(values: np.ndarray, size: int) -> pa.FixedSizeListArray:
    flat = np.asarray(values, dtype=np.float32).reshape(-1, size)
    return pa.FixedSizeListArray.from_arrays(
        pa.array(flat.reshape(-1), type=pa.float32()), size
    )


SCHEMA = pa.schema(
    [
        pa.field("episode_idx", pa.int32()),
        pa.field("step_idx", pa.int32()),
        pa.field("pixels", pa.binary()),
        pa.field("action", pa.list_(pa.float32(), 2)),
        pa.field("proprio", pa.list_(pa.float32(), 4)),
        pa.field("observation", pa.list_(pa.float32(), 6)),
        pa.field("finger_pos", pa.list_(pa.float32(), 2)),
        pa.field("hidden_arm_density", pa.list_(pa.float32(), 1)),
        pa.field("pair_id", pa.string()),
        pa.field("hidden_mode", pa.string()),
        pa.field("split", pa.string()),
        pa.field("catalog_index", pa.list_(pa.float32(), 1)),
    ]
)


_WORKER_SIMULATOR: ReacherArmMassSimulator | None = None
_WORKER_QUALITY = 95


def _worker_initialize(quality: int) -> None:
    global _WORKER_SIMULATOR, _WORKER_QUALITY
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
    _WORKER_QUALITY = int(quality)
    _WORKER_SIMULATOR = ReacherArmMassSimulator()
    atexit.register(_WORKER_SIMULATOR.close)


def _encode(value: np.ndarray) -> bytes:
    buffer = BytesIO()
    Image.fromarray(np.asarray(value, dtype=np.uint8)).save(
        buffer,
        format="JPEG",
        quality=_WORKER_QUALITY,
    )
    return buffer.getvalue()


def _compact_episode(episode: dict[str, Any]) -> dict[str, Any]:
    rows = episode["rows"]
    return {
        "pixels": [_encode(value) for value in rows["pixels"]],
        "action": np.asarray(rows["action"], dtype=np.float32),
        "state": np.asarray(rows["state"], dtype=np.float32),
        "observation": np.asarray(rows["observation"], dtype=np.float32),
        "finger_pos": np.asarray(rows["finger_pos"], dtype=np.float32),
        "density": float(episode["density"]),
    }


def _build_candidate(
    request: tuple[str, int, int]
) -> dict[str, Any] | None:
    split, index, catalog_seed = request
    assert _WORKER_SIMULATOR is not None
    candidate = make_candidate(
        split=split,
        index=index,
        catalog_seed=catalog_seed,
    )
    result = _WORKER_SIMULATOR.build_pair(candidate)
    if result is None:
        return None
    lighter = result["lighter"]
    return {
        "candidate": result["candidate"],
        "audit": result["audit"],
        "query_hash": array_sha256(lighter["model_pixels"][2]),
        "episodes": {
            mode: _compact_episode(result[mode]) for mode in MASS_MODES
        },
    }


def _record_batch(
    episode: dict[str, Any],
    *,
    episode_index: int,
    split: str,
    pair_id: str,
    mode: str,
    catalog_index: int,
) -> pa.RecordBatch:
    count = len(episode["pixels"])
    arrays: list[pa.Array] = [
        pa.array(np.full(count, episode_index, dtype=np.int32)),
        pa.array(np.arange(count, dtype=np.int32)),
        pa.array(episode["pixels"], type=pa.binary()),
        _fixed(episode["action"], 2),
        _fixed(episode["state"], 4),
        _fixed(episode["observation"], 6),
        _fixed(episode["finger_pos"], 2),
        _fixed(
            np.full((count, 1), episode["density"], dtype=np.float32),
            1,
        ),
        pa.array([pair_id] * count, type=pa.string()),
        pa.array([mode] * count, type=pa.string()),
        pa.array([split] * count, type=pa.string()),
        _fixed(
            np.full((count, 1), catalog_index, dtype=np.float32),
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
    workers: int,
) -> dict[str, Any]:
    table_path = root / f"{split}.lance"
    accepted: list[dict[str, Any]] = []
    started = time.monotonic()
    attempted = 0
    maximum_candidates = 20 * pair_count

    def batches() -> Iterator[pa.RecordBatch]:
        nonlocal attempted
        context = mp.get_context("spawn")
        with context.Pool(
            processes=workers,
            initializer=_worker_initialize,
            initargs=(quality,),
        ) as pool:
            requests = (
                (split, index, catalog_seed)
                for index in range(maximum_candidates)
            )
            episode_index = 0
            for index, result in enumerate(
                pool.imap(_build_candidate, requests, chunksize=1)
            ):
                attempted = index + 1
                if result is None:
                    continue
                accepted.append(
                    {
                        "candidate": result["candidate"],
                        "audit": result["audit"],
                        "query_hash": result["query_hash"],
                    }
                )
                pair_id = result["candidate"]["candidate_id"]
                catalog_index = int(
                    result["candidate"]["catalog_index"]
                )
                for mode in MASS_MODES:
                    yield _record_batch(
                        result["episodes"][mode],
                        episode_index=episode_index,
                        split=split,
                        pair_id=pair_id,
                        mode=mode,
                        catalog_index=catalog_index,
                    )
                    episode_index += 1
                count = len(accepted)
                if count <= 3 or count % 128 == 0:
                    print(
                        f"{split}: accepted {count}/{pair_count}, "
                        f"attempted={attempted}, "
                        f"elapsed={time.monotonic() - started:.1f}s",
                        flush=True,
                    )
                if count == pair_count:
                    break
            if len(accepted) != pair_count:
                raise RuntimeError(
                    f"Only {len(accepted)}/{pair_count} valid {split} pairs "
                    f"after {attempted} candidates"
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
        "attempted_candidates": attempted,
        "acceptance_rate": pair_count / attempted,
        "table_path": table_path.name,
        "table_files": len(files),
        "table_bytes": sum(path.stat().st_size for path in files),
        "table_sha256": directory_sha256(table_path),
        "catalog_seed": catalog_seed,
        "query_hashes": [row["query_hash"] for row in accepted],
        "template_ids": [
            row["candidate"]["candidate_id"] for row in accepted
        ],
        "minimum_history_qpos_gap": min(
            row["audit"]["history_qpos_gap"] for row in accepted
        ),
        "minimum_true_future_qpos_gap": min(
            row["audit"]["future_qpos_gap"] for row in accepted
        ),
        "maximum_query_state_gap": max(
            row["audit"]["query_state_gap"] for row in accepted
        ),
        "minimum_history_changed_rgb_values": min(
            row["audit"]["history_changed_rgb_values"]
            for row in accepted
        ),
        "minimum_future_changed_rgb_values": min(
            row["audit"]["future_changed_rgb_values"]
            for row in accepted
        ),
        "pairs": accepted,
        "passed": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-pairs", type=int, default=2048)
    parser.add_argument("--development-pairs", type=int, default=256)
    parser.add_argument("--public-test-pairs", type=int, default=256)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.mkdir(parents=True)
    request = {
        "protocol": PROTOCOL,
        "hidden_factor": (
            "shared agent.arm_density and agent.finger_density"
        ),
        "hidden_modes": ARM_DENSITIES,
        "pair_counts": {
            "train": args.train_pairs,
            "loader_validation": args.development_pairs,
            "validation": args.public_test_pairs,
        },
        "catalog_seeds": {
            "train": 2026080211,
            "loader_validation": 2026080212,
            "validation": 2026080213,
        },
        "jpeg_quality": args.jpeg_quality,
        "workers": args.workers,
    }
    (output / "request.json").write_text(
        json.dumps(request, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    reports = {}
    for split in SPLITS:
        reports[split] = build_split(
            output,
            split=split,
            pair_count=int(request["pair_counts"][split]),
            catalog_seed=int(request["catalog_seeds"][split]),
            quality=args.jpeg_quality,
            workers=args.workers,
        )
    overlaps = {}
    for left, right in (
        ("train", "loader_validation"),
        ("train", "validation"),
        ("loader_validation", "validation"),
    ):
        overlaps[f"{left}_vs_{right}_query_hashes"] = len(
            set(reports[left]["query_hashes"])
            & set(reports[right]["query_hashes"])
        )
        overlaps[f"{left}_vs_{right}_template_ids"] = len(
            set(reports[left]["template_ids"])
            & set(reports[right]["template_ids"])
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
        json.dumps(build_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
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
        "bytes_without_manifest": sum(
            path.stat().st_size for path in files_before_manifest
        ),
        "build_passed": build_report["passed"],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "passed": build_report["passed"],
                "pair_counts": request["pair_counts"],
                "cross_split_overlap": overlaps,
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
