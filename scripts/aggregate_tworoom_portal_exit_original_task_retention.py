#!/usr/bin/env python3
"""Audit and aggregate paired TwoRoom CEM retention results for Portal Exit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contextworld.benchmarks.portal_exit_icl_data import (  # noqa: E402
    DEFAULT_PORTAL_EXIT_RELEASE_CONFIG,
    load_portal_exit_icl_release,
)
from contextworld.paths import artifact_path  # noqa: E402


DEFAULT_INPUT = artifact_path(
    "evaluation/history3/tworoom_portal_exit_h3_release_v1/"
    "original_task_retention"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _query_identity(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        record["evaluation_id"],
        int(record["evaluation_index"]),
        int(record["eval_seed"]),
        record["source_kind"],
        record["source_path"],
        int(record["episode"]),
        int(record["start_step"]),
        int(record["goal_offset"]),
        int(record["cem_group_seed"]),
        tuple(record["initial_state"]),
        tuple(record["goal_state"]),
    )


def _load_model(root: Path, name: str, seeds: tuple[int, ...]) -> dict[str, Any]:
    cells = []
    for seed in seeds:
        path = root / name / f"seed{seed}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "passed":
            raise ValueError(f"Incomplete CEM cell: {path}")
        if int(payload["protocol"]["eval_seed"]) != seed:
            raise ValueError(f"Eval seed mismatch: {path}")
        if int(payload["aggregate"]["evaluations"]) != 50:
            raise ValueError(f"Each CEM cell must contain 50 queries: {path}")
        if payload["frozen_weight_audit"]["passed"] is not True:
            raise ValueError(f"Checkpoint changed during evaluation: {path}")
        cells.append((path, payload))
    records = [record for _, payload in cells for record in payload["raw_records"]]
    successes = sum(bool(record["success"]) for record in records)
    return {
        "name": name,
        "cells": cells,
        "records": records,
        "successes": int(successes),
        "evaluations": len(records),
        "success_rate": float(successes / len(records)),
        "mean_final_distance": float(
            statistics.mean(float(record["final_distance"]) for record in records)
        ),
        "seed_successes": {
            str(seed): int(payload["aggregate"]["successes"])
            for seed, (_, payload) in zip(seeds, cells, strict=True)
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--release-config", type=Path, default=DEFAULT_PORTAL_EXIT_RELEASE_CONFIG
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    release = load_portal_exit_icl_release(args.release_config)
    retention = release["scoring"]["original_task_retention"]
    seeds = tuple(int(value) for value in retention["eval_seeds"])
    training_seeds = tuple(
        int(value)
        for value in release["training"]["reference_matrix"]["training_seeds"]
    )
    root = args.input.expanduser().resolve()
    output = (args.output or root / "summary.json").expanduser().resolve()
    baseline = _load_model(root, "baseline_lewm", seeds)
    candidates = [
        _load_model(root, f"icl_lewm_seed{seed}", seeds)
        for seed in training_seeds
    ]
    expected_protocol = {
        "action_block": 5,
        "cem_samples": 300,
        "cem_steps": 30,
        "cem_topk": 30,
        "eval_budget": 50,
        "evaluations": 50,
        "history_size": 3,
        "horizon": 5,
        "receding_horizon": 5,
    }
    baseline_queries = [_query_identity(record) for record in baseline["records"]]
    catalog_hashes = set()
    runtime_commits = set()
    for model in (baseline, *candidates):
        if model["evaluations"] != 300:
            raise ValueError(f"{model['name']} does not contain 6 x 50 queries")
        if [_query_identity(record) for record in model["records"]] != baseline_queries:
            raise ValueError(f"Query identity differs for {model['name']}")
        for _, cell in model["cells"]:
            protocol = dict(cell["protocol"])
            protocol.pop("eval_seed")
            if protocol != expected_protocol:
                raise ValueError(f"CEM protocol differs for {model['name']}")
            catalog_hashes.add(cell["catalog"]["sha256"])
            runtime_commits.add(cell["stable_worldmodel"]["commit"])
    expected_catalog = retention["query_catalog"]["sha256"]
    if catalog_hashes != {expected_catalog}:
        raise ValueError(f"Unexpected query catalog hashes: {catalog_hashes}")
    expected_commit = release["runtime"]["stable_worldmodel"]["expected_ref"]
    if runtime_commits != {expected_commit}:
        raise ValueError(f"Unexpected Stable-WorldModel commits: {runtime_commits}")
    margin = int(retention["noninferiority_margin_successes"])
    rows = []
    for model in candidates:
        delta = model["successes"] - baseline["successes"]
        rows.append(
            {
                "name": model["name"],
                "successes": model["successes"],
                "evaluations": model["evaluations"],
                "success_rate": model["success_rate"],
                "mean_final_distance": model["mean_final_distance"],
                "seed_successes": model["seed_successes"],
                "success_delta_from_baseline": delta,
                "minimum_allowed_delta": -margin,
                "passed": delta >= -margin,
            }
        )
    payload = {
        "schema_version": 1,
        "benchmark": "tworoom_portal_exit_original_task_retention_v1",
        "status": "completed",
        "protocol": {
            "eval_seeds": list(seeds),
            "queries_per_seed": 50,
            "queries_per_checkpoint": 300,
            **expected_protocol,
            "query_catalog_sha256": expected_catalog,
            "stable_worldmodel_commit": expected_commit,
            "same_query_identity_across_models": True,
        },
        "baseline": {
            "name": baseline["name"],
            "successes": baseline["successes"],
            "evaluations": baseline["evaluations"],
            "success_rate": baseline["success_rate"],
            "mean_final_distance": baseline["mean_final_distance"],
            "seed_successes": baseline["seed_successes"],
        },
        "candidates": rows,
        "gate": {
            "noninferiority_margin_successes": margin,
            "checkpoints_required": 3,
            "checkpoints_passed": sum(row["passed"] for row in rows),
            "passed": len(rows) == 3 and all(row["passed"] for row in rows),
        },
        "source_files": {
            str(path.relative_to(root)): _sha256(path)
            for model in (baseline, *candidates)
            for path, _ in model["cells"]
        },
    }
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    if not payload["gate"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
