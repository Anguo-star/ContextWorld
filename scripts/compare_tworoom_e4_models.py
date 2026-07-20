#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from contextworld.paths import resolve_contextworld_path
from contextworld.synthesis.manifest import write_json


def _sign_test(left_only: int, right_only: int) -> dict[str, Any]:
    discordant = left_only + right_only
    if discordant == 0:
        return {"discordant_pairs": 0, "two_sided_p_value": 1.0}
    tail = sum(
        math.comb(discordant, value)
        for value in range(min(left_only, right_only) + 1)
    ) / (2**discordant)
    return {
        "discordant_pairs": discordant,
        "two_sided_p_value": min(1.0, 2.0 * tail),
    }


def _load_records(summary_path: Path) -> tuple[dict[str, Any], dict[tuple[str, str], dict[str, Any]]]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    records: dict[tuple[str, str], dict[str, Any]] = {}
    for raw_path in summary["protocol"]["raw_results"]:
        resolved_raw = Path(raw_path)
        if not resolved_raw.is_file():
            try:
                legacy_relative = resolved_raw.relative_to(REPO_ROOT)
            except ValueError:
                legacy_relative = resolved_raw
            resolved_raw = resolve_contextworld_path(
                legacy_relative, repo_root=REPO_ROOT
            )
        raw = json.loads(resolved_raw.read_text(encoding="utf-8"))
        for record in raw["records"]:
            key = (record["evaluation_id"], record["condition"])
            if key in records:
                raise ValueError(f"Duplicate E4 record {key} in {summary_path}")
            records[key] = record
    return summary, records


def _condition_comparison(
    reference: dict[tuple[str, str], dict[str, Any]],
    candidate: dict[tuple[str, str], dict[str, Any]],
    condition: str,
) -> dict[str, Any]:
    keys = sorted(key for key in reference if key[1] == condition)
    rows = []
    by_speed: dict[float, list[dict[str, Any]]] = defaultdict(list)
    by_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for key in keys:
        left = reference[key]
        right = candidate[key]
        paired_fields = (
            "eval_seed",
            "query_id",
            "speed",
            "template_id",
            "cem_seed",
            "goal_state",
        )
        mismatches = {
            field: {"reference": left[field], "candidate": right[field]}
            for field in paired_fields
            if left[field] != right[field]
        }
        if mismatches:
            raise ValueError(f"Cross-model E4 pairing mismatch for {key}: {mismatches}")
        row = {
            "evaluation_id": key[0],
            "eval_seed": int(left["eval_seed"]),
            "speed": float(left["speed"]),
            "reference_success": bool(left["success"]),
            "candidate_success": bool(right["success"]),
            "reference_final_distance": float(left["final_distance"]),
            "candidate_final_distance": float(right["final_distance"]),
            "candidate_minus_reference_final_distance": float(
                right["final_distance"] - left["final_distance"]
            ),
        }
        rows.append(row)
        by_speed[row["speed"]].append(row)
        by_seed[row["eval_seed"]].append(row)

    def aggregate(values: list[dict[str, Any]]) -> dict[str, Any]:
        both = sum(row["reference_success"] and row["candidate_success"] for row in values)
        reference_only = sum(row["reference_success"] and not row["candidate_success"] for row in values)
        candidate_only = sum(not row["reference_success"] and row["candidate_success"] for row in values)
        neither = len(values) - both - reference_only - candidate_only
        distance_deltas = [row["candidate_minus_reference_final_distance"] for row in values]
        tolerance = 1e-12
        return {
            "evaluations": len(values),
            "both_successes": both,
            "reference_only_successes": reference_only,
            "candidate_only_successes": candidate_only,
            "neither_successes": neither,
            "reference_successes": both + reference_only,
            "candidate_successes": both + candidate_only,
            "candidate_minus_reference_success_rate_points": 100.0
            * (candidate_only - reference_only)
            / len(values),
            "paired_sign_test": _sign_test(reference_only, candidate_only),
            "reference_mean_final_distance": statistics.fmean(
                row["reference_final_distance"] for row in values
            ),
            "candidate_mean_final_distance": statistics.fmean(
                row["candidate_final_distance"] for row in values
            ),
            "candidate_minus_reference_mean_final_distance": statistics.fmean(
                distance_deltas
            ),
            "candidate_minus_reference_median_final_distance": statistics.median(
                distance_deltas
            ),
            "mean_absolute_paired_final_distance_difference": statistics.fmean(
                abs(value) for value in distance_deltas
            ),
            "candidate_lower_final_distance_pairs": sum(
                value < -tolerance for value in distance_deltas
            ),
            "reference_lower_final_distance_pairs": sum(
                value > tolerance for value in distance_deltas
            ),
            "equal_final_distance_pairs": sum(
                abs(value) <= tolerance for value in distance_deltas
            ),
        }

    result = aggregate(rows)
    result["by_speed"] = {
        str(speed): aggregate(values) for speed, values in sorted(by_speed.items())
    }
    result["by_seed"] = {
        str(seed): aggregate(values) for seed, values in sorted(by_seed.items())
    }
    outcome_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["reference_success"] and row["candidate_success"]:
            label = "both_success"
        elif row["reference_success"]:
            label = "reference_only_success"
        elif row["candidate_success"]:
            label = "candidate_only_success"
        else:
            label = "both_failure"
        outcome_groups[label].append(row)
    result["by_paired_success_outcome"] = {
        label: aggregate(values) for label, values in sorted(outcome_groups.items())
    }
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    reference_path = resolve_contextworld_path(args.reference, repo_root=REPO_ROOT)
    candidate_path = resolve_contextworld_path(args.candidate, repo_root=REPO_ROOT)
    output_path = resolve_contextworld_path(args.output, repo_root=REPO_ROOT)
    reference_summary, reference = _load_records(reference_path)
    candidate_summary, candidate = _load_records(candidate_path)
    if set(reference) != set(candidate):
        raise ValueError(
            "E4 model summaries do not contain identical evaluation IDs/conditions: "
            f"reference_only={len(set(reference)-set(candidate))}, "
            f"candidate_only={len(set(candidate)-set(reference))}"
        )
    conditions = {
        condition: _condition_comparison(reference, candidate, condition)
        for condition in ("correct", "wrong")
    }
    reference_effect = float(
        reference_summary["aggregate"]["correct_minus_wrong_success_rate_points"]
    )
    candidate_effect = float(
        candidate_summary["aggregate"]["correct_minus_wrong_success_rate_points"]
    )
    payload = {
        "schema_version": 1,
        "benchmark": "contextworld_tworoom_e4_cross_model_paired_v1",
        "status": "passed",
        "reference": str(reference_path),
        "candidate": str(candidate_path),
        "paired_records": len(reference),
        "conditions": conditions,
        "context_effect": {
            "reference_correct_minus_wrong_points": reference_effect,
            "candidate_correct_minus_wrong_points": candidate_effect,
            "difference_in_differences_points": candidate_effect - reference_effect,
        },
    }
    write_json(output_path, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pair two E4 model results by evaluation ID")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps({"conditions": result["conditions"], "context_effect": result["context_effect"]}, sort_keys=True))
