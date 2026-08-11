#!/usr/bin/env python3
"""Aggregate paired standard PushT CEM retention for strict causal checkpoints."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contextworld.benchmarks.motion_damping_icl_data import (  # noqa: E402
    DEFAULT_MOTION_DAMPING_RELEASE_CONFIG,
    file_sha256,
    load_motion_damping_icl_release,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--release-config",
        type=Path,
        default=DEFAULT_MOTION_DAMPING_RELEASE_CONFIG,
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        required=True,
        help="aggregate.json from the original LeWM checkpoint",
    )
    parser.add_argument(
        "--candidate",
        type=Path,
        action="append",
        required=True,
        help="aggregate.json from one retrained checkpoint; repeat six times",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load_single(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("status") != "standard_pusht_real_environment_cem":
        raise ValueError(f"Unexpected CEM report: {path}")
    if len(report.get("models", [])) != 1:
        raise ValueError(f"Expected one model in {path}")
    model = report["models"][0]
    successes = [
        bool(value)
        for seed in model["seeds"]
        for value in seed["episode_successes"]
    ]
    if len(successes) != 300:
        raise ValueError(f"Expected 300 episodes in {path}")
    return report, {**model, "flat_successes": successes}


def _catalog(path: Path, report: dict[str, Any]) -> dict[str, Any]:
    catalog_path = Path(report["query_catalog"]["path"])
    if not catalog_path.is_file():
        candidate = path.parent / "query_catalog.json"
        if not candidate.is_file():
            raise FileNotFoundError(catalog_path)
        catalog_path = candidate
    if file_sha256(catalog_path) != report["query_catalog"]["sha256"]:
        raise RuntimeError(f"Query catalog hash mismatch: {catalog_path}")
    return json.loads(catalog_path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    release_path = args.release_config.expanduser().resolve()
    release = load_motion_damping_icl_release(release_path)
    baseline_path = args.baseline.expanduser().resolve()
    candidate_paths = [path.expanduser().resolve() for path in args.candidate]
    if len(candidate_paths) != 6 or len(set(candidate_paths)) != 6:
        raise ValueError("Exactly six distinct candidate CEM reports are required")
    baseline_report, baseline = _load_single(baseline_path)
    expected_protocol = baseline_report["protocol"]
    reference_catalog = _catalog(baseline_path, baseline_report)
    baseline_success = baseline["flat_successes"]
    retention = release["scoring"]["original_task_retention"]
    fixed_minimum = int(retention["noninferiority_success_minimum"])
    margin = int(retention["baseline_margin_successes"])
    rows = []
    names = set()
    for path in candidate_paths:
        report, model = _load_single(path)
        if report["protocol"] != expected_protocol:
            raise RuntimeError(f"CEM protocol differs: {path}")
        if _catalog(path, report) != reference_catalog:
            raise RuntimeError(f"CEM query catalog differs: {path}")
        name = str(model["model"])
        if name in names:
            raise ValueError(f"Duplicate candidate model {name}")
        names.add(name)
        candidate_success = model["flat_successes"]
        paired = {
            "both_success": sum(
                left and right
                for left, right in zip(
                    candidate_success, baseline_success, strict=True
                )
            ),
            "candidate_only_success": sum(
                left and not right
                for left, right in zip(
                    candidate_success, baseline_success, strict=True
                )
            ),
            "baseline_only_success": sum(
                not left and right
                for left, right in zip(
                    candidate_success, baseline_success, strict=True
                )
            ),
            "both_failure": sum(
                not left and not right
                for left, right in zip(
                    candidate_success, baseline_success, strict=True
                )
            ),
        }
        success_count = int(model["aggregate"]["success_count"])
        baseline_count = int(baseline["aggregate"]["success_count"])
        checks = {
            "fixed_minimum": success_count >= fixed_minimum,
            "within_baseline_margin": success_count >= baseline_count - margin,
        }
        rows.append(
            {
                "model": name,
                "checkpoint": model["checkpoint"],
                "checkpoint_sha256": model["checkpoint_sha256"],
                "source_report": {
                    "path": str(path),
                    "sha256": file_sha256(path),
                },
                "success_count": success_count,
                "evaluation_count": 300,
                "success_rate": success_count / 300.0,
                "success_difference_from_baseline": (
                    success_count - baseline_count
                ),
                "paired_outcomes": paired,
                "checks": checks,
                "passed": all(checks.values()),
            }
        )
    model_groups = {
        name: [row for row in rows if row["model"].startswith(name + "_")]
        for name in ("lewm", "pldm")
    }
    if any(len(group) != 3 for group in model_groups.values()):
        raise ValueError("Candidates must contain three LeWM and three PLDM models")
    methods = {
        name: {
            "checkpoint_count": len(group),
            "success_counts": [row["success_count"] for row in group],
            "all_three_checkpoints_passed": all(row["passed"] for row in group),
        }
        for name, group in model_groups.items()
    }
    payload = {
        "schema_version": 1,
        "status": "completed",
        "release": {
            "release_id": release["release_id"],
            "path": str(release_path),
            "sha256": file_sha256(release_path),
        },
        "protocol": expected_protocol,
        "query_catalog_identical_for_all_checkpoints": True,
        "baseline": {
            "model": baseline["model"],
            "checkpoint": baseline["checkpoint"],
            "checkpoint_sha256": baseline["checkpoint_sha256"],
            "source_report": {
                "path": str(baseline_path),
                "sha256": file_sha256(baseline_path),
            },
            "success_count": baseline["aggregate"]["success_count"],
            "evaluation_count": 300,
            "success_rate": baseline["aggregate"]["success_rate"],
        },
        "noninferiority_contract": {
            "fixed_success_minimum": fixed_minimum,
            "baseline_margin_successes": margin,
            "both_checks_required": True,
        },
        "checkpoints": rows,
        "methods": methods,
    }
    output = Path(os.path.abspath(args.output.expanduser()))
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
