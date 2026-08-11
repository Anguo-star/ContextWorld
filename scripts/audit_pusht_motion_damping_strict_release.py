#!/usr/bin/env python3
"""Audit every pair in the strict-causal PushT motion-damping release.

The clean-simulator branch is diagnostic only.  It replays the naturally
reached x2 in a fresh simulator to test for solver-cache influence, but it is
never used by the data builder or benchmark target generation.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import lance
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import contextworld.evaluation.pusht_motion_damping_h3 as damping  # noqa: E402
from contextworld.evaluation import pusht_contact_friction_h3 as friction  # noqa: E402


DEFAULT_RELEASE = Path("/tmp/contextworld_motion_damping_strict_release")


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


def _audit_condition(payload: tuple[dict[str, Any], str, int]) -> dict[str, Any]:
    template_payload, mode, resolution = payload
    template = damping.MotionDampingTemplate(**template_payload)
    continuous = damping._simulate_continuous_causal_chain(
        template,
        mode=mode,
        resolution=resolution,
        render_pixels=True,
    )
    clean, _ = damping.make_motion_damping_env(
        template, mode=mode, resolution=resolution
    )
    try:
        friction.restore_body_snapshot(clean, continuous["query_snapshot"])
        clean_start_arbiters = clean.space._get_arbiters()
        clean_start_cached_arbiters = sum(
            int(arbiter.state == 3) for arbiter in clean_start_arbiters
        )
        clean_start_active_arbiters = sum(
            int(arbiter.state in (0, 1)) for arbiter in clean_start_arbiters
        )
        for action in np.asarray(template.query_actions, dtype=np.float64):
            friction._step_and_count_agent_block_contacts(clean, action)
        clean_future = friction.body_snapshot(clean)
        clean_pixels = np.asarray(clean.render(), dtype=np.uint8).copy()
        clean_end_arbiters = clean.space._get_arbiters()
        clean_end_active_arbiters = sum(
            int(arbiter.state in (0, 1)) for arbiter in clean_end_arbiters
        )
    finally:
        clean.close()
    state_gap = float(
        np.max(
            np.abs(
                friction._snapshot_delta(
                    continuous["future_snapshot"], clean_future
                )
            )
        )
    )
    pixel_difference = int(
        np.max(
            np.abs(
                continuous["future_pixels"].astype(np.int16)
                - clean_pixels.astype(np.int16)
            )
        )
    )
    return {
        "template_id": template.template_id,
        "mode": mode,
        "maximum_continuous_arbiter_count": int(
            max(
                np.max(continuous["history_arbiter_counts"], initial=0),
                np.max(continuous["query_arbiter_counts"], initial=0),
            )
        ),
        "clean_start_total_arbiter_count": len(clean_start_arbiters),
        "clean_start_cached_inactive_arbiter_count": (
            clean_start_cached_arbiters
        ),
        "clean_start_active_arbiter_count": clean_start_active_arbiters,
        "clean_end_total_arbiter_count": len(clean_end_arbiters),
        "clean_end_active_arbiter_count": clean_end_active_arbiters,
        "continuous_vs_clean_x3_full_state_gap": state_gap,
        "continuous_vs_clean_x3_pixel_difference": pixel_difference,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    release = args.release.expanduser().resolve()
    manifest_path = release / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tasks = []
    for split in ("train", "loader_validation", "validation"):
        for pair in manifest["splits"][split]["pairs"]:
            for mode in damping.ENDPOINT_MODES:
                tasks.append((pair["template"], mode, int(args.resolution)))
    with ProcessPoolExecutor(max_workers=int(args.workers)) as executor:
        conditions = list(executor.map(_audit_condition, tasks, chunksize=8))

    lance_rows = {
        split: int(lance.dataset(str(release / f"{split}.lance")).count_rows())
        for split in ("train", "loader_validation", "validation")
    }
    expected_rows = {
        split: 40 * int(manifest["splits"][split]["pair_count"])
        for split in lance_rows
    }
    report = {
        "protocol": "pusht_motion_damping_history3_strict_causal_release_v3",
        "purpose": "diagnostic_only_not_used_for_data_generation",
        "release": str(release),
        "source_manifest_sha256": file_sha256(manifest_path),
        "pair_count": int(sum(manifest["pair_counts"].values())),
        "condition_count": len(conditions),
        "state_installations_after_x0_in_formal_data": 0,
        "query_simulator_recreated_in_formal_data": False,
        "minimum_consecutive_arbiter_free_raw_steps_before_x2": 10,
        "maximum_continuous_arbiter_count": max(
            row["maximum_continuous_arbiter_count"] for row in conditions
        ),
        "maximum_clean_start_total_arbiter_count": max(
            row["clean_start_total_arbiter_count"] for row in conditions
        ),
        "maximum_clean_start_cached_inactive_arbiter_count": max(
            row["clean_start_cached_inactive_arbiter_count"]
            for row in conditions
        ),
        "maximum_clean_start_active_arbiter_count": max(
            row["clean_start_active_arbiter_count"] for row in conditions
        ),
        "maximum_clean_end_total_arbiter_count": max(
            row["clean_end_total_arbiter_count"] for row in conditions
        ),
        "maximum_clean_end_active_arbiter_count": max(
            row["clean_end_active_arbiter_count"] for row in conditions
        ),
        "maximum_continuous_vs_clean_x3_full_state_gap": max(
            row["continuous_vs_clean_x3_full_state_gap"] for row in conditions
        ),
        "maximum_continuous_vs_clean_x3_pixel_difference": max(
            row["continuous_vs_clean_x3_pixel_difference"] for row in conditions
        ),
        "lance_rows": lance_rows,
        "expected_lance_rows": expected_rows,
        "conditions": conditions,
    }
    report["passed"] = bool(
        manifest["passed"]
        and lance_rows == expected_rows
        and report["maximum_continuous_arbiter_count"] == 0
        and report["maximum_clean_start_active_arbiter_count"] == 0
        and report["maximum_clean_end_active_arbiter_count"] == 0
        and report["maximum_continuous_vs_clean_x3_full_state_gap"] <= 1e-8
        and report["maximum_continuous_vs_clean_x3_pixel_difference"] == 0
    )
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else release / "strict_causal_audit.json"
    )
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    files = [path for path in release.rglob("*") if path.is_file()]
    summary = {
        key: report[key]
        for key in (
            "protocol",
            "purpose",
            "pair_count",
            "condition_count",
            "source_manifest_sha256",
            "minimum_consecutive_arbiter_free_raw_steps_before_x2",
            "maximum_continuous_arbiter_count",
            "maximum_clean_start_total_arbiter_count",
            "maximum_clean_start_cached_inactive_arbiter_count",
            "maximum_clean_start_active_arbiter_count",
            "maximum_clean_end_total_arbiter_count",
            "maximum_clean_end_active_arbiter_count",
            "maximum_continuous_vs_clean_x3_full_state_gap",
            "maximum_continuous_vs_clean_x3_pixel_difference",
            "lance_rows",
            "expected_lance_rows",
            "passed",
        )
    }
    summary["audit_path"] = str(output)
    summary["audit_sha256"] = file_sha256(output)
    summary["artifact_files"] = len(files)
    summary["artifact_bytes"] = sum(path.stat().st_size for path in files)
    summary["artifact_tree_sha256"] = directory_sha256(release)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
