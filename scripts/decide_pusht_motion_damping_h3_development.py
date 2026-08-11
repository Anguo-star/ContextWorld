#!/usr/bin/env python3
"""Seal the fixed Motion Damping Development endpoint decision.

The command reads one fixed-step Training report whose embedded evaluation is
the frozen Development split. It never reads or scores Public Test.
"""

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


METRIC_MAP = {
    "correct_future_rate": "two_real_future_target_selection_rate",
    "correct_history_rate": "correct_history_preference_rate",
    "context_switch_rate": "correct_rule_switch_rate",
    "worst_damping_correct_future_rate": (
        "worst_mode_target_selection_rate"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--release-config",
        type=Path,
        default=DEFAULT_MOTION_DAMPING_RELEASE_CONFIG,
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="The fixed 8,192-step Development training_report.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _checkpoint_receipt(report: dict[str, Any]) -> str:
    value = report["result"]["final_checkpoint"]
    checkpoint = Path(value["path"] if isinstance(value, dict) else value)
    observed = file_sha256(checkpoint)
    if isinstance(value, dict) and value.get("sha256") != observed:
        raise RuntimeError("The fixed endpoint checkpoint receipt changed")
    return observed


def main() -> None:
    args = parse_args()
    release = load_motion_damping_icl_release(
        args.release_config.expanduser().resolve()
    )
    report_path = args.input.expanduser().resolve()
    if not report_path.is_file():
        raise FileNotFoundError(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    provenance = report["provenance"]
    snapshot = report["result"]["snapshots"][-1]
    metrics = snapshot["hidden_evaluation"]
    endpoint = release["training"]["reference_matrix"][
        "reported_endpoint"
    ]
    common = release["training"]["reference_matrix"]["common"]
    thresholds = release["scoring"]["hidden_future_prediction"]["gates"]
    threshold_keys = {
        "correct_future_rate": "correct_future_rate_minimum",
        "correct_history_rate": "correct_history_rate_minimum",
        "context_switch_rate": "context_switch_rate_minimum",
        "worst_damping_correct_future_rate": (
            "worst_damping_correct_future_rate_minimum"
        ),
    }
    values = {
        name: float(metrics[source]) for name, source in METRIC_MAP.items()
    }
    gates = {
        name: {
            "minimum": float(thresholds[threshold_keys[name]]),
            "passed": values[name]
            >= float(thresholds[threshold_keys[name]]),
        }
        for name in METRIC_MAP
    }
    failed_metrics = [
        name for name, result in gates.items() if not result["passed"]
    ]
    batching = report["result"].get("batch", {}).get(
        "motion_damping_twin_grouping", {}
    )
    data = provenance["data"]
    contract = {
        "recipe_matches": provenance["variant"] == endpoint["recipe"],
        "model_family_matches": provenance["model"]
        == endpoint["model_family"].lower(),
        "release_id_matches": provenance["release"]["release_id"]
        == release["release_id"],
        "data_manifest_matches": data["manifest_sha256"]
        == release["data"]["manifest_sha256"],
        "formal_data_root_used": data.get("data_root_override") is False,
        "public_test_not_opened": data.get(
            "independent_validation_opened"
        )
        is False,
        "training_seed_matches": int(provenance["seed"])
        == int(endpoint["training_seed"]),
        "fixed_optimizer_step_matches": (
            int(report["fixed_checkpoint_step"])
            == int(snapshot["optimizer_step"])
            == int(endpoint["optimizer_step"])
            == int(common["fixed_checkpoint_step"])
        ),
        "complete_twin_batching": (
            batching.get("enabled") is True
            and batching.get("condition_rows_per_group") == 4
            and batching.get(
                "x0_rgb_label_exchange_complete_in_every_group"
            )
            is True
        ),
    }
    if not all(contract.values()):
        failed = [name for name, value in contract.items() if not value]
        raise RuntimeError(f"Endpoint contract failed: {failed}")
    passed = not failed_metrics
    payload = {
        "schema_version": 1,
        "component": "pusht_motion_damping_icl",
        "status": "passed_development" if passed else "failed_development",
        "data_release": {
            "protocol": release["data"]["protocol"],
            "manifest_sha256": release["data"]["manifest_sha256"],
            "pair_counts": {
                "training": int(release["data"]["pair_counts"]["train"]),
                "development": int(
                    release["data"]["pair_counts"]["loader_validation"]
                ),
                "public_test": int(
                    release["data"]["pair_counts"]["validation"]
                ),
            },
        },
        "reported_endpoint": {
            "model_family": endpoint["model_family"],
            "training_receipt": {
                "runner": release["identity"]["reference_trainer"]["path"],
                "runner_sha256": release["identity"]["reference_trainer"][
                    "sha256"
                ],
                "recipe": endpoint["recipe"],
                "training_seed": int(endpoint["training_seed"]),
                "optimizer_step": int(endpoint["optimizer_step"]),
                "training_report_sha256": file_sha256(report_path),
                "checkpoint_sha256": _checkpoint_receipt(report),
            },
            "selection_contract": {
                "development_used_for_recipe_selection": True,
                "development_used_for_checkpoint_selection": False,
                "checkpoint_step_was_fixed_before_scoring": True,
                "public_test_used_for_selection": False,
            },
            "metrics": values,
            "gates": gates,
            "failed_metrics": failed_metrics,
            "passed": passed,
        },
        "public_model_scoring_opened": False,
        "additional_training_seeds_run": False,
        "original_task_cem_run": False,
        "positive_reference_claim": False,
    }
    output = Path(os.path.abspath(args.output.expanduser()))
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "output": str(output),
                "failed_metrics": failed_metrics,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
