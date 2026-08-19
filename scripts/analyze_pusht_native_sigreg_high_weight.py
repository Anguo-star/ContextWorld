#!/usr/bin/env python3
"""Seal the Push-T mixed-training sweep of unchanged native SIGReg weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


DEFAULT_MIXED_ROOT = Path(
    "/opt/huawei/explorer-env/dataset/ag_data/data/world_model/context_world/"
    "evaluation/history3/pusht_hidden_actuation_h3_v1/"
    "mixed_retention_seed3073_step2048"
)
DEFAULT_SWEEP_ROOT = Path(
    "/opt/huawei/explorer-env/dataset/ag_data/data/world_model/context_world/"
    "evaluation/history3/pusht_hidden_actuation_h3_v1/"
    "native_sigreg_high_weight_seed3073_step2048"
)

REPORTS = {
    "0.09": (DEFAULT_MIXED_ROOT, "native/mixed_report.json"),
    "0.20": (DEFAULT_SWEEP_ROOT, "native_0p20/mixed_report.json"),
    "0.30": (DEFAULT_SWEEP_ROOT, "native_0p30/mixed_report.json"),
    "0.50": (DEFAULT_SWEEP_ROOT, "native_0p50/mixed_report.json"),
    "0.90": (DEFAULT_SWEEP_ROOT, "native_0p90/mixed_report.json"),
    "2.05": (DEFAULT_SWEEP_ROOT, "native_2p05/mixed_report.json"),
}
PREDICTION_GATES = {
    "target_selection": 0.95,
    "correct_history": 0.95,
    "rule_switch": 0.95,
    "worst_mode": 0.90,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mixed-root", type=Path, default=DEFAULT_MIXED_ROOT)
    parser.add_argument("--sweep-root", type=Path, default=DEFAULT_SWEEP_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def one(items: list[dict[str, Any]], context: str) -> dict[str, Any]:
    if len(items) != 1:
        raise ValueError(f"{context}: expected one item, found {len(items)}")
    return items[0]


def summarize_report(
    report: dict[str, Any],
    *,
    expected_weight: float,
) -> dict[str, Any]:
    result = one(report["results"], "training result")
    contract = report["training_contract"]
    final = result["snapshots"][-1]
    if final["optimizer_step"] != contract["max_steps"]:
        raise ValueError("Report does not end at the registered optimizer step")
    if abs(float(result["regularizer_weight"]) - expected_weight) > 1e-12:
        raise ValueError(
            f"Expected weight {expected_weight}, "
            f"found {result['regularizer_weight']}"
        )
    if result["regularizer"] != "native":
        raise ValueError(
            f"Expected unchanged native SIGReg, found {result['regularizer']}"
        )
    if contract != {
        "batch_size": 128,
        "hidden_fraction": 0.5,
        "max_steps": 2048,
        "original_fraction": 0.5,
        "same_hidden_pair_order_seed": True,
        "same_original_sampler_seed": True,
        "same_source_checkpoint": True,
        "seed": 3073,
    }:
        raise ValueError(f"Unexpected training contract: {contract}")

    metrics = final["hidden_evaluation"]
    values = {
        "target_selection": metrics[
            "two_real_future_target_selection_rate"
        ],
        "correct_history": metrics["correct_history_preference_rate"],
        "rule_switch": metrics["correct_rule_switch_rate"],
        "worst_mode": metrics["worst_mode_target_selection_rate"],
        "prediction_mse_margin": metrics["prediction_mse"][
            "incorrect_minus_correct_margin"
        ],
        "paired_to_unrelated_ratio": metrics["representation_geometry"][
            "prediction_space"
        ]["paired_to_unrelated_ratio"],
        "future_effective_rank": metrics["representation_geometry"][
            "future_effective_rank"
        ],
        "final_training_pred_loss": result["loss_trace"][-1]["pred_loss"],
        "final_training_regularizer_loss": result["loss_trace"][-1][
            "regularizer_loss"
        ],
        "checkpoint_sha256": result["final_checkpoint"]["sha256"],
    }
    values["prediction_gate_passed"] = all(
        values[name] >= threshold
        for name, threshold in PREDICTION_GATES.items()
    )
    return values


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    mixed_root = Path(os.path.abspath(args.mixed_root.expanduser()))
    sweep_root = Path(os.path.abspath(args.sweep_root.expanduser()))
    output = (
        Path(os.path.abspath(args.output.expanduser()))
        if args.output
        else sweep_root / "summary.json"
    )
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {output}")

    roots = {"0.09": mixed_root}
    roots.update({weight: sweep_root for weight in REPORTS if weight != "0.09"})
    weights: dict[str, Any] = {}
    sources: dict[str, Any] = {}
    for weight, (_, relative) in REPORTS.items():
        path = roots[weight] / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        report = load_json(path)
        weights[weight] = summarize_report(
            report,
            expected_weight=float(weight),
        )
        sources[weight] = {
            "path": str(path),
            "sha256": sha256(path),
        }

    passing = [
        weight
        for weight, values in weights.items()
        if values["prediction_gate_passed"]
    ]
    payload = {
        "schema_version": 1,
        "benchmark": "pusht_hidden_actuation_history3_action_coverage_v2",
        "experiment": "unchanged_native_sigreg_high_weight_mixed_v1",
        "owner": "ContextWorld",
        "status": (
            "complete_hidden_prediction_pass"
            if passing
            else "complete_no_hidden_prediction_pass_through_2p05"
        ),
        "method_name_zh": "原始 SIGReg 高权重版",
        "training_contract": {
            "seed": 3073,
            "optimizer_steps": 2048,
            "batch_size": 128,
            "standard_pusht_fraction": 0.5,
            "hidden_actuation_fraction": 0.5,
            "same_source_checkpoint_and_batch_order": True,
            "regularizer": "unchanged_full_batch_native_sigreg",
        },
        "prediction_gates": PREDICTION_GATES,
        "weights": weights,
        "passing_weights": passing,
        "standard_pusht_cem": {
            "protocol": "exact_standard_history3_cem_300x30_top30",
            "scheduled_for_every_prediction_passing_weight": True,
            "executed_weights": [],
            "skip_reason": (
                None
                if passing
                else "No high-weight checkpoint passed all hidden-prediction gates."
            ),
        },
        "decision": {
            "two_room_range_closed_through": 2.05,
            "hidden_rule_learned_by_unchanged_native_sigreg": bool(passing),
            "high_weight_hidden_rule_vs_cem_tradeoff_observed": False,
            "interpretation": (
                "Across the pre-existing TwoRoom coefficient range, larger "
                "unchanged native SIGReg improved some conditional geometry "
                "metrics but did not make PushT hidden-actuation prediction "
                "reliable. The planned standard PushT CEM tradeoff test was "
                "therefore not reached."
            ),
            "claim_boundary": (
                "This single-seed adaptive screen does not prove that an "
                "unbounded coefficient can never pass."
            ),
        },
        "sources": sources,
    }
    atomic_write_json(output, payload)
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": sha256(output),
                "passing_weights": passing,
                "status": payload["status"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
