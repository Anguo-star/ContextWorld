#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from contextworld.evaluation.icl_catalog import _simulate_blocks
from contextworld.evaluation.speed_identifiability import (
    agent_centroid_from_rgb,
    alternating_impulse_blocks,
    estimate_speed_from_transitions,
    nearest_speed,
)
from contextworld.paths import artifact_path, resolve_contextworld_path
from contextworld.synthesis.manifest import write_json
from contextworld.synthesis.stablewm import load_stable_worldmodel


PINNED_STABLEWM = "5864b74980f6ed328fd0045e777b3865962eff43"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _aggregate(rows: list[dict[str, Any]], estimator: str) -> dict[str, Any]:
    errors = np.asarray(
        [abs(row[f"{estimator}_estimate"] - row["true_speed"]) for row in rows],
        dtype=np.float64,
    )
    correct = np.asarray(
        [row[f"{estimator}_prediction"] == row["true_speed"] for row in rows],
        dtype=bool,
    )
    by_speed: dict[str, Any] = {}
    grouped: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[float(row["true_speed"])].append(row)
    for speed, speed_rows in sorted(grouped.items()):
        speed_errors = np.asarray(
            [abs(row[f"{estimator}_estimate"] - speed) for row in speed_rows]
        )
        speed_correct = sum(
            row[f"{estimator}_prediction"] == speed for row in speed_rows
        )
        by_speed[str(speed)] = {
            "examples": len(speed_rows),
            "correct": int(speed_correct),
            "accuracy": float(speed_correct / len(speed_rows)),
            "mean_absolute_error": float(speed_errors.mean()),
            "maximum_absolute_error": float(speed_errors.max()),
        }
    return {
        "examples": len(rows),
        "correct": int(correct.sum()),
        "accuracy": float(correct.mean()),
        "mean_absolute_error": float(errors.mean()),
        "maximum_absolute_error": float(errors.max()),
        "by_speed": by_speed,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    catalog_path = resolve_contextworld_path(args.catalog, repo_root=REPO_ROOT)
    output_path = resolve_contextworld_path(args.output, repo_root=REPO_ROOT)
    budgets = sorted({int(value) for value in args.context_transitions})
    if not budgets or any(value <= 0 or value % 2 for value in budgets):
        raise ValueError("All context transition budgets must be positive and even")

    _, stable_repo, stable_commit = load_stable_worldmodel(
        REPO_ROOT, args.stablewm_repo, args.stablewm_ref
    )
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    bundles = [
        bundle
        for bundle in catalog["bundles"]
        if bundle["family"] == "speed" and bundle["split"] == "validation"
    ]
    if not bundles:
        raise ValueError(f"No speed validation bundles in {catalog_path}")
    candidate_speeds = sorted(
        {float(bundle["query_factors"]["agent.speed"]) for bundle in bundles}
    )

    results: dict[str, Any] = {}
    all_endpoint_errors: list[float] = []
    all_collision_residuals: list[float] = []
    for budget in budgets:
        rows: list[dict[str, Any]] = []
        for bundle in bundles:
            factors = dict(bundle["conditions"]["correct"]["factors"])
            true_speed = float(factors["agent.speed"])
            direction = np.asarray(
                bundle["template"]["context_direction"], dtype=np.float32
            )
            actions = alternating_impulse_blocks(direction, budget)
            rollout = _simulate_blocks(
                factors,
                np.asarray(bundle["template"]["reset_state"], dtype=np.float32),
                np.asarray(bundle["template"]["goal_state"], dtype=np.float32),
                actions,
                seed=int(bundle["simulator_seed"]),
            )
            state_estimate, state_per_transition = estimate_speed_from_transitions(
                rollout["states"], rollout["next_states"], actions
            )
            pixel_states = agent_centroid_from_rgb(rollout["pixels"])
            pixel_next_states = agent_centroid_from_rgb(rollout["next_pixels"])
            pixel_estimate, pixel_per_transition = estimate_speed_from_transitions(
                pixel_states, pixel_next_states, actions
            )
            effective_actions = actions.sum(axis=1).astype(np.float64)
            state_displacements = (
                rollout["next_states"] - rollout["states"]
            ).astype(np.float64)
            collision_residuals = np.linalg.norm(
                state_displacements - effective_actions * true_speed, axis=1
            )
            endpoint_error = float(
                np.linalg.norm(
                    rollout["next_states"][-1]
                    - np.asarray(bundle["template"]["reset_state"], dtype=np.float32)
                )
            )
            all_endpoint_errors.append(endpoint_error)
            all_collision_residuals.extend(collision_residuals.tolist())
            rows.append(
                {
                    "query_id": bundle["query_id"],
                    "scenario_id": bundle["source_scenario_id"],
                    "template_id": bundle["template"]["template_id"],
                    "true_speed": true_speed,
                    "state_estimate": state_estimate,
                    "state_prediction": nearest_speed(state_estimate, candidate_speeds),
                    "pixel_estimate": pixel_estimate,
                    "pixel_prediction": nearest_speed(pixel_estimate, candidate_speeds),
                    "state_per_transition": state_per_transition.tolist(),
                    "pixel_per_transition": pixel_per_transition.tolist(),
                    "endpoint_error": endpoint_error,
                    "maximum_collision_residual": float(collision_residuals.max()),
                }
            )
        results[f"k{budget}"] = {
            "context_transitions": budget,
            "model_history_tokens_if_matched": budget + 1,
            "state_oracle": _aggregate(rows, "state"),
            "pixel_oracle": _aggregate(rows, "pixel"),
            "maximum_endpoint_error": float(max(row["endpoint_error"] for row in rows)),
            "maximum_collision_residual": float(
                max(row["maximum_collision_residual"] for row in rows)
            ),
            "rows": rows,
        }

    k2 = results.get("k2")
    if k2 is None:
        decision = "k2_not_evaluated"
    else:
        longer_pixel_accuracy = max(
            result["pixel_oracle"]["accuracy"]
            for key, result in results.items()
            if key != "k2"
        ) if len(results) > 1 else k2["pixel_oracle"]["accuracy"]
        if (
            k2["state_oracle"]["accuracy"] == 1.0
            and k2["pixel_oracle"]["accuracy"] >= 0.95
            and longer_pixel_accuracy - k2["pixel_oracle"]["accuracy"] <= 0.02
        ):
            decision = "k2_is_information_sufficient_history_length_not_primary_bottleneck"
        elif longer_pixel_accuracy > k2["pixel_oracle"]["accuracy"] + 0.02:
            decision = "longer_context_improves_observable_speed_identification"
        else:
            decision = "inconclusive"

    output = {
        "schema_version": 1,
        "benchmark": "tworoom_speed_context_identifiability_v1",
        "run_kind": "deterministic_oracle_diagnostic",
        "catalog": str(catalog_path),
        "catalog_sha256": _sha256_file(catalog_path),
        "stable_worldmodel": {"repo": str(stable_repo), "commit": stable_commit},
        "split": "validation",
        "family": "speed",
        "candidate_speeds": candidate_speeds,
        "bundles": len(bundles),
        "context_transitions": budgets,
        "protocol": {
            "prefix": "alternating impulse/inverse pairs",
            "query_state_unchanged_for_all_even_k": True,
            "state_oracle_inputs": ["privileged state", "model-visible action"],
            "pixel_oracle_inputs": ["model-visible RGB", "model-visible action"],
            "pixel_decoder": "deterministic red-excess weighted centroid",
            "classifier": "nearest validation speed",
            "factor_values_used_as_model_inputs": False,
        },
        "audit": {
            "maximum_endpoint_error": float(max(all_endpoint_errors)),
            "maximum_collision_residual": float(max(all_collision_residuals)),
        },
        "results": results,
        "decision": decision,
    }
    write_json(output_path, output)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate K=2/4/8 speed identifiability from paired context"
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=artifact_path(
            "evaluation/icl/tworoom_icl_v1_validation_context_query_catalog.json",
            repo_root=REPO_ROOT,
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=artifact_path(
            "evaluation/history3/oracle_speed_identifiability_k2_k4_k8.json",
            repo_root=REPO_ROOT,
        ),
    )
    parser.add_argument(
        "--context-transitions", type=int, nargs="+", default=[2, 4, 8]
    )
    parser.add_argument("--stablewm-repo", default="../stable-worldmodel")
    parser.add_argument("--stablewm-ref", default=PINNED_STABLEWM)
    return parser.parse_args()


if __name__ == "__main__":
    payload = run(parse_args())
    summary = {
        key: {
            "state_accuracy": value["state_oracle"]["accuracy"],
            "pixel_accuracy": value["pixel_oracle"]["accuracy"],
            "pixel_mae": value["pixel_oracle"]["mean_absolute_error"],
        }
        for key, value in payload["results"].items()
    }
    print(json.dumps({"decision": payload["decision"], "results": summary}, sort_keys=True))
