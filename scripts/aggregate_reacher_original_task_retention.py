#!/usr/bin/env python3
"""Compare Reacher CEM results on one shared set of deterministic queries."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_result(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    root = path.expanduser().resolve()
    aggregate_path = root / "aggregate.json" if root.is_dir() else root
    payload = json.loads(aggregate_path.read_text(encoding="utf-8"))
    if payload.get("task") != "reacher" or len(payload.get("models", [])) != 1:
        raise ValueError(f"Expected one Reacher model result: {aggregate_path}")
    catalog_path = aggregate_path.parent / "query_catalog.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    return (
        {
            "aggregate_path": str(aggregate_path),
            "aggregate_sha256": _sha256(aggregate_path),
            "query_catalog_path": str(catalog_path),
            "query_catalog_sha256": _sha256(catalog_path),
            "protocol": payload["protocol"],
            "model": payload["models"][0],
        },
        catalog,
    )


def _paired_comparison(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    margin: int,
) -> dict[str, Any]:
    baseline_seeds = {
        int(row["eval_seed"]): row for row in baseline["model"]["seeds"]
    }
    candidate_seeds = {
        int(row["eval_seed"]): row for row in candidate["model"]["seeds"]
    }
    if set(baseline_seeds) != set(candidate_seeds):
        raise ValueError("Baseline and candidate Eval seeds differ")
    seed_rows = []
    for seed in sorted(baseline_seeds):
        base = baseline_seeds[seed]
        new = candidate_seeds[seed]
        base_outcomes = [bool(value) for value in base["episode_successes"]]
        new_outcomes = [bool(value) for value in new["episode_successes"]]
        if len(base_outcomes) != len(new_outcomes):
            raise ValueError(f"Query count differs for Eval seed {seed}")
        seed_rows.append(
            {
                "eval_seed": seed,
                "query_count": len(base_outcomes),
                "baseline_successes": sum(base_outcomes),
                "candidate_successes": sum(new_outcomes),
                "success_delta": sum(new_outcomes) - sum(base_outcomes),
                "regressions": sum(
                    before and not after
                    for before, after in zip(base_outcomes, new_outcomes)
                ),
                "improvements": sum(
                    not before and after
                    for before, after in zip(base_outcomes, new_outcomes)
                ),
            }
        )
    baseline_total = sum(row["baseline_successes"] for row in seed_rows)
    candidate_total = sum(row["candidate_successes"] for row in seed_rows)
    return {
        "model": candidate["model"]["model"],
        "checkpoint": candidate["model"]["checkpoint"],
        "checkpoint_sha256": candidate["model"]["checkpoint_sha256"],
        "evaluation_count": sum(row["query_count"] for row in seed_rows),
        "baseline_successes": baseline_total,
        "candidate_successes": candidate_total,
        "success_delta": candidate_total - baseline_total,
        "noninferiority_margin_successes": margin,
        "passed": candidate_total >= baseline_total - margin,
        "by_eval_seed": seed_rows,
        "source_result": {
            "path": candidate["aggregate_path"],
            "sha256": candidate["aggregate_sha256"],
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, action="append", required=True)
    parser.add_argument("--margin", type=int, default=15)
    parser.add_argument("--mujoco-gl", default="osmesa")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.margin < 0:
        raise ValueError("--margin must be non-negative")
    if len(args.candidate) != 3:
        raise ValueError("Reacher retention requires exactly three candidates")
    baseline, catalog = _load_result(args.baseline)
    if int(baseline["model"]["aggregate"]["evaluation_count"]) != 300:
        raise ValueError("Reacher baseline retention must contain 300 episodes")
    candidates = []
    for path in args.candidate:
        candidate, candidate_catalog = _load_result(path)
        if candidate_catalog != catalog:
            raise ValueError(f"Query catalog differs: {path}")
        if candidate["protocol"] != baseline["protocol"]:
            raise ValueError(f"CEM protocol differs: {path}")
        if int(candidate["model"]["aggregate"]["evaluation_count"]) != 300:
            raise ValueError(f"Reacher candidate must contain 300 episodes: {path}")
        candidates.append(candidate)
    model_names = [row["model"]["model"] for row in candidates]
    if len(set(model_names)) != len(model_names):
        raise ValueError("Candidate model names must be distinct")
    comparisons = [
        _paired_comparison(baseline, candidate, margin=args.margin)
        for candidate in candidates
    ]
    payload = {
        "schema_version": 1,
        "status": "completed",
        "task": "reacher_original_task_retention",
        "query_catalog": {
            "path": baseline["query_catalog_path"],
            "sha256": baseline["query_catalog_sha256"],
            "identical_across_all_results": True,
        },
        "protocol": baseline["protocol"],
        "runtime": {
            "mujoco_gl": str(args.mujoco_gl),
            "note": (
                "渲染后端不改变冻结的 query、物理状态、CEM 参数或执行预算。"
            ),
        },
        "baseline": {
            "model": baseline["model"]["model"],
            "checkpoint": baseline["model"]["checkpoint"],
            "checkpoint_sha256": baseline["model"]["checkpoint_sha256"],
            "success_count": baseline["model"]["aggregate"]["success_count"],
            "evaluation_count": baseline["model"]["aggregate"][
                "evaluation_count"
            ],
            "source_result": {
                "path": baseline["aggregate_path"],
                "sha256": baseline["aggregate_sha256"],
            },
        },
        "comparisons": comparisons,
        "passed": bool(comparisons and all(row["passed"] for row in comparisons)),
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), "passed": payload["passed"]}))


if __name__ == "__main__":
    main()
