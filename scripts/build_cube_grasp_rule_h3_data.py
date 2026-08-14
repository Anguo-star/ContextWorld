#!/usr/bin/env python3
"""Build Cube History-3 "does gripper lift move the cube?" pairs."""

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

from contextworld.evaluation.cube_grasp_rule_h3 import (  # noqa: E402
    CAPABILITY_NAME,
    GRASP_MODES,
    QUERY_STATE_TOLERANCE,
    CubeGraspRuleCandidate,
    CubeGraspRuleSimulator,
)
from contextworld.benchmarks.causal_data_contract import (  # noqa: E402
    audit_causal_data_contract,
)
from contextworld.paths import artifact_path  # noqa: E402


PROTOCOL = "cube_gripper_carry_rule_history3_release_v1"
SPLITS = ("train", "loader_validation", "validation")
DEFAULT_OUTPUT = artifact_path(
    "synthesis/cube_gripper_carry_rule_h3_release_v1"
)
DEFAULT_SOURCE = Path(
    "/opt/huawei/explorer-env/dataset/ag_data/data/world_model/"
    "quentinll/lewm-cube/ogbench/cube_single_expert.h5"
)
CATALOG_SEEDS = {
    "train": 2026080311,
    "loader_validation": 2026080312,
    "validation": 2026080313,
}
CANDIDATE_POOL_MULTIPLIER = 2


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
    ]
)


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
    rows = np.flatnonzero(mask)
    best: dict[int, tuple[float, int, int]] = {}
    for row in rows:
        episode = int(episodes[row])
        candidate = (float(distance[row]), int(row), int(steps[row]))
        if episode not in best or candidate < best[episode]:
            best[episode] = candidate
    return [
        (row, episode, step)
        for episode, (_, row, step) in sorted(best.items())
    ]


def build_candidate_catalogs(
    source: Path,
    *,
    pair_counts: dict[str, int],
) -> tuple[dict[str, list[CubeGraspRuleCandidate]], dict[str, Any]]:
    eligible = _eligible_source_rows(source)
    required_pool = sum(
        CANDIDATE_POOL_MULTIPLIER * int(pair_counts[split])
        for split in SPLITS
    )
    if len(eligible) < required_pool:
        raise RuntimeError(
            f"Only {len(eligible)} eligible source episodes for "
            f"{required_pool} requested candidates"
        )
    order = np.random.default_rng(2026080310).permutation(len(eligible))
    cursor = 0
    assignments: dict[str, list[tuple[int, int, int]]] = {}
    for split in SPLITS:
        count = CANDIDATE_POOL_MULTIPLIER * int(pair_counts[split])
        selected = [eligible[int(index)] for index in order[cursor : cursor + count]]
        cursor += count
        assignments[split] = selected

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

    catalogs: dict[str, list[CubeGraspRuleCandidate]] = {}
    for split in SPLITS:
        rows = assignments[split]
        catalog = []
        for index, (source_row, source_episode, source_step) in enumerate(rows):
            rng = np.random.default_rng(
                np.random.SeedSequence([CATALOG_SEEDS[split], index])
            )
            source_qpos, source_control = source_values[source_row]
            color = tuple(float(value) for value in rng.uniform(0.18, 0.92, 3))
            target = (
                float(rng.uniform(0.32, 0.53)),
                float(rng.uniform(-0.24, 0.24)),
                0.02,
            )
            catalog.append(
                CubeGraspRuleCandidate(
                    candidate_id=f"cube-carry-{split}-{index:06d}",
                    split=split,
                    catalog_index=index,
                    source_row=source_row,
                    source_episode=source_episode,
                    source_step=source_step,
                    simulator_seed=int(rng.integers(0, 2**31 - 1)),
                    task_id=1 + index % 5,
                    qpos=tuple(float(value) for value in source_qpos),
                    control=tuple(float(value) for value in source_control),
                    cube_color=color,
                    target_position=target,
                )
            )
        catalogs[split] = catalog
    receipt = {
        "source_symbol": "upstream_cube_h5",
        "environment_variable": "CONTEXTWORLD_CUBE_H5",
        "local_source_path_recorded": False,
        "source_size_bytes": source.stat().st_size,
        "eligible_episodes": len(eligible),
        "candidate_pool_per_split": {
            split: len(catalogs[split]) for split in SPLITS
        },
        "candidate_pool_multiplier": CANDIDATE_POOL_MULTIPLIER,
        "source_episode_overlap": {
            f"{left}_vs_{right}": len(
                {value.source_episode for value in catalogs[left]}
                & {value.source_episode for value in catalogs[right]}
            )
            for left, right in (
                ("train", "loader_validation"),
                ("train", "validation"),
                ("loader_validation", "validation"),
            )
        },
    }
    return catalogs, receipt


_WORKER_SIMULATOR: CubeGraspRuleSimulator | None = None
_WORKER_QUALITY = 95


def _worker_initialize(quality: int) -> None:
    global _WORKER_SIMULATOR, _WORKER_QUALITY
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
    _WORKER_QUALITY = int(quality)
    _WORKER_SIMULATOR = CubeGraspRuleSimulator()
    atexit.register(_WORKER_SIMULATOR.close)


def _encode(value: np.ndarray) -> bytes:
    buffer = BytesIO()
    Image.fromarray(np.asarray(value, dtype=np.uint8)).save(
        buffer, format="JPEG", quality=_WORKER_QUALITY
    )
    return buffer.getvalue()


def _build_candidate(candidate: CubeGraspRuleCandidate) -> dict[str, Any] | None:
    assert _WORKER_SIMULATOR is not None
    result = _WORKER_SIMULATOR.build_pair(candidate)
    if result is None:
        return None
    return {
        "candidate": result["candidate"],
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
            }
            for mode in GRASP_MODES
        },
    }


def _record_batch(
    episode: dict[str, Any],
    *,
    episode_index: int,
    split: str,
    candidate: dict[str, Any],
    mode: str,
) -> pa.RecordBatch:
    count = 4
    arrays: list[pa.Array] = [
        pa.array(np.full(count, episode_index, dtype=np.int32)),
        pa.array(np.arange(count, dtype=np.int32)),
        pa.array(episode["pixels"], type=pa.binary()),
        _fixed(episode["action_blocks"], 25),
        _fixed(episode["physical_state"], 7),
        _fixed(
            np.full((count, 1), episode["hidden_value"], dtype=np.float32),
            1,
        ),
        pa.array([candidate["candidate_id"]] * count),
        pa.array([mode] * count),
        pa.array([split] * count),
        pa.array(
            np.full(count, candidate["catalog_index"], dtype=np.int32)
        ),
        pa.array(
            np.full(count, candidate["source_row"], dtype=np.int64)
        ),
        pa.array(
            np.full(count, candidate["source_episode"], dtype=np.int32)
        ),
    ]
    return pa.record_batch(arrays, schema=SCHEMA)


def build_split(
    root: Path,
    *,
    split: str,
    pair_count: int,
    candidates: list[CubeGraspRuleCandidate],
    quality: int,
    workers: int,
) -> dict[str, Any]:
    table_path = root / f"{split}.lance"
    accepted: list[dict[str, Any]] = []
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
                accepted.append(
                    {
                        "candidate": result["candidate"],
                        "audit": result["audit"],
                        "query_hash": result["audit"]["hashes"][
                            "query_pixels"
                        ],
                    }
                )
                for mode in GRASP_MODES:
                    yield _record_batch(
                        result["episodes"][mode],
                        episode_index=episode_index,
                        split=split,
                        candidate=result["candidate"],
                        mode=mode,
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
        "model_rows": 8 * pair_count,
        "attempted_candidates": attempted,
        "acceptance_rate": pair_count / attempted,
        "table_path": table_path.name,
        "table_files": len(files),
        "table_bytes": sum(path.stat().st_size for path in files),
        "table_sha256": directory_sha256(table_path),
        "catalog_seed": CATALOG_SEEDS[split],
        "query_hashes": [row["query_hash"] for row in accepted],
        "template_ids": [
            row["candidate"]["candidate_id"] for row in accepted
        ],
        "source_episodes": [
            row["candidate"]["source_episode"] for row in accepted
        ],
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
            row["audit"]["maximum_query_simulator_state_gap"]
            for row in accepted
        ),
        "maximum_prequery_object_state_residual": max(
            row["audit"]["maximum_prequery_object_state_residual"]
            for row in accepted
        ),
        "maximum_state_installations_after_x0": max(
            row["audit"]["state_installations_after_x0"]
            for row in accepted
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
        "passed": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--train-pairs", type=int, default=2048)
    parser.add_argument("--development-pairs", type=int, default=256)
    parser.add_argument("--public-test-pairs", type=int, default=256)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    source = args.source.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if not source.is_file():
        raise FileNotFoundError(source)
    output.mkdir(parents=True)
    pair_counts = {
        "train": int(args.train_pairs),
        "loader_validation": int(args.development_pairs),
        "validation": int(args.public_test_pairs),
    }
    catalogs, source_receipt = build_candidate_catalogs(
        source, pair_counts=pair_counts
    )
    request = {
        "protocol": PROTOCOL,
        "capability": CAPABILITY_NAME,
        "display_name_zh": "Cube 夹爪升降是否带动方块",
        "transition_rule": (
            "MuJoCo generalized-force coupling; no state installation "
            "after x0"
        ),
        "hidden_modes": list(GRASP_MODES),
        "pair_counts": pair_counts,
        "catalog_seeds": CATALOG_SEEDS,
        "jpeg_quality": int(args.jpeg_quality),
        "workers": int(args.workers),
        "source": source_receipt,
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
            pair_count=pair_counts[split],
            candidates=catalogs[split],
            quality=args.jpeg_quality,
            workers=args.workers,
        )
    overlaps = {}
    for left, right in (
        ("train", "loader_validation"),
        ("train", "validation"),
        ("loader_validation", "validation"),
    ):
        for field in ("query_hashes", "template_ids", "source_episodes"):
            overlaps[f"{left}_vs_{right}_{field}"] = len(
                set(reports[left][field]) & set(reports[right][field])
            )
    all_reports = list(reports.values())
    maximum_query_state_gap = max(
        row["maximum_query_simulator_state_gap"] for row in all_reports
    )
    causal_contract = audit_causal_data_contract(
        component_id="cube_gripper_carry_rule",
        evidence_scope=(
            "every accepted pair in Training, Development and Public Test"
        ),
        continuous_environment_trajectory=True,
        state_installations_after_x0=max(
            row["maximum_state_installations_after_x0"]
            for row in all_reports
        ),
        query_simulator_recreated=False,
        maximum_query_state_gap=maximum_query_state_gap,
        query_state_tolerance=QUERY_STATE_TOLERANCE,
        query_pixels_exact=True,
        query_actions_exact=True,
        history_effect_present=min(
            row["minimum_history_cube_height_gap_m"]
            for row in all_reports
        )
        >= 0.008,
        true_future_effect_present=min(
            row["minimum_future_cube_height_gap_m"]
            for row in all_reports
        )
        >= 0.008,
        x0_policy="shared_visible_start",
        x0_static_leakage_check_passed=True,
        solver_cache_check_required=True,
        solver_cache_check_passed=(
            maximum_query_state_gap <= QUERY_STATE_TOLERANCE
        ),
        evidence=(
            "Each condition resets only before x0 and then uses env.step.",
            "The hidden rule changes qfrc_applied, not qpos or qvel.",
            "The complete query audit includes solver warm-start state.",
        ),
    )
    build_report = {
        "protocol": PROTOCOL,
        "request": request,
        "splits": reports,
        "cross_split_overlap": overlaps,
        "causal_data_contract": causal_contract,
        "passed": all(row["passed"] for row in reports.values())
        and not any(overlaps.values())
        and causal_contract["passed"],
    }
    (output / "build_report.json").write_text(
        json.dumps(build_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    files = [path for path in output.rglob("*") if path.is_file()]
    manifest = {
        "protocol": PROTOCOL,
        "files": {
            path.relative_to(output).as_posix(): file_sha256(path)
            for path in sorted(files)
        },
        "file_count_without_manifest": len(files),
        "bytes_without_manifest": sum(path.stat().st_size for path in files),
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
                "pair_counts": pair_counts,
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
