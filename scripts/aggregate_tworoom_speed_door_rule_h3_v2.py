#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contextworld.evaluation.speed_door_rule_v2_score import (
    PRIMARY_METRICS,
)
from contextworld.evaluation.speed_door_rule_v2_validation import (
    file_sha256,
)
from contextworld.paths import resolve_contextworld_path
from contextworld.synthesis.config import load_config
from contextworld.synthesis.manifest import write_json


DEFAULT_CONFIG = (
    ROOT / "configs/benchmark/tworoom_speed_door_rule_h3_v2.yaml"
)
SEEDS = (3072, 4096, 5120)


def _stats(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(statistics.fmean(values)),
        "std": float(
            statistics.stdev(values) if len(values) > 1 else 0.0
        ),
        "minimum": float(min(values)),
        "maximum": float(max(values)),
    }


def _role_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = (
        *PRIMARY_METRICS,
        "door_anchor_future_accuracy",
        "passable_speed_history_guidance",
        "door_history_guidance",
    )
    by_horizon = {}
    for horizon in ("h1", "h2"):
        payloads = [
            row["summary"]["by_horizon"][horizon]["overall"]
            for row in rows
        ]
        by_horizon[horizon] = {
            metric: _stats(
                [float(payload[metric]) for payload in payloads]
            )
            for metric in metrics
        }
    return {
        "training_seeds": sorted(
            int(row["training_seed"]) for row in rows
        ),
        "checkpoints": len(rows),
        "all_checkpoint_gates_passed": bool(rows)
        and all(row["checkpoint_gate"]["passed"] for row in rows),
        "by_horizon": by_horizon,
    }


def _descriptive_summary(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    summary = _role_summary(rows)
    summary.pop("all_checkpoint_gates_passed")
    summary["claim_limit"] = (
        "Read-only diagnostic; no pass/fail gate applies."
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--results-root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = load_config(args.config.expanduser().resolve())
    results_root = (
        args.results_root.expanduser().resolve()
        if args.results_root is not None
        else resolve_contextworld_path(
            config["artifacts"]["validation_results_root"],
            repo_root=ROOT,
        )
    )
    result_paths = sorted(results_root.glob("*.json"))
    if not result_paths:
        raise FileNotFoundError(results_root)
    rows = []
    for path in result_paths:
        with path.open("r", encoding="utf-8") as handle:
            row = json.load(handle)
        if row.get("benchmark") == config["benchmark"]:
            rows.append(row)
    if not rows:
        raise RuntimeError("No v2 results found")
    catalog_hashes = {
        row["identity"]["catalog_sha256"] for row in rows
    }
    asset_hashes = {
        row["asset_audit"]["content_manifest_sha256"]
        for row in rows
    }
    if len(catalog_hashes) != 1 or len(asset_hashes) != 1:
        raise RuntimeError("v2 results do not share one frozen dataset")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    descriptive_grouped: dict[str, list[dict[str, Any]]] = defaultdict(
        list
    )
    for row in rows:
        role = str(row["role"])
        if role == "descriptive":
            descriptive_grouped[str(row["model_id"])].append(row)
        else:
            grouped[role].append(row)
    role_summaries = {
        role: _role_summary(role_rows)
        for role, role_rows in sorted(grouped.items())
    }
    descriptive_model_summaries = {
        model_id: _descriptive_summary(model_rows)
        for model_id, model_rows in sorted(
            descriptive_grouped.items()
        )
    }

    prerequisite_checks = {}
    for role in ("speed_only", "door_only"):
        role_rows = grouped.get(role, [])
        observed = {
            int(row["training_seed"]) for row in role_rows
        }
        prerequisite_checks[role] = {
            "expected_training_seeds": list(SEEDS),
            "observed_training_seeds": sorted(observed),
            "all_three_present": observed == set(SEEDS),
            "all_three_checkpoint_gates_passed": (
                observed == set(SEEDS)
                and all(
                    row["checkpoint_gate"]["passed"]
                    for row in role_rows
                )
            ),
        }
    prerequisites_passed = all(
        row["all_three_checkpoint_gates_passed"]
        for row in prerequisite_checks.values()
    )
    joint_rows = grouped.get("joint", [])
    joint_seeds = {
        int(row["training_seed"]) for row in joint_rows
    }
    joint_gate = {
        "required": prerequisites_passed,
        "expected_training_seeds": list(SEEDS),
        "observed_training_seeds": sorted(joint_seeds),
        "all_three_present": joint_seeds == set(SEEDS),
        "all_three_checkpoint_gates_passed": (
            joint_seeds == set(SEEDS)
            and all(
                row["checkpoint_gate"]["passed"]
                for row in joint_rows
            )
        ),
    }
    formal_composition_passed = bool(
        prerequisites_passed
        and joint_gate["all_three_checkpoint_gates_passed"]
    )
    result = {
        "schema_version": 2,
        "benchmark": str(config["benchmark"]),
        "status": "completed",
        "result_files": [
            {
                "path": str(path),
                "sha256": file_sha256(path),
            }
            for path in result_paths
            if any(
                row.get("model_id") in path.name for row in rows
            )
        ],
        "frozen_dataset": {
            "catalog_sha256": next(iter(catalog_hashes)),
            "content_manifest_sha256": next(iter(asset_hashes)),
        },
        "role_summaries": role_summaries,
        "descriptive_model_summaries": (
            descriptive_model_summaries
        ),
        "prerequisites": {
            "checks": prerequisite_checks,
            "passed": prerequisites_passed,
        },
        "joint_method_gate": joint_gate,
        "formal_composition_claim_passed": formal_composition_passed,
        "interpretation": (
            "History=3 speed and door-rule composition passed."
            if formal_composition_passed
            else (
                "Single-factor prerequisites failed; no joint model was "
                "required and no composition claim is made."
                if not prerequisites_passed
                else "Single-factor prerequisites passed, but the joint "
                "three-seed method gate failed."
            )
        ),
    }
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else resolve_contextworld_path(
            config["artifacts"]["validation_aggregate"],
            repo_root=ROOT,
        )
    )
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
