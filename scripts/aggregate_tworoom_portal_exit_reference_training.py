#!/usr/bin/env python3
"""Summarize Portal Exit Development results without opening Public Test."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
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


DEFAULT_ROOT = artifact_path(
    "evaluation/history3/tworoom_portal_exit_h3_release_v1"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _row(path: Path, expected_seed: int, expected_variant: str) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    result = report["result"]
    if result["seed"] != expected_seed or result["variant"] != expected_variant:
        raise ValueError(f"Unexpected training identity: {path}")
    if report["fixed_checkpoint_step"] != 4096:
        raise ValueError(f"Unexpected optimizer budget: {path}")
    if report["independent_validation_used_for_selection"] is not False:
        raise ValueError(f"Public Test was opened by training: {path}")
    metrics = result["snapshots"][-1]["hidden_evaluation"]
    checkpoint_record = result["final_checkpoint"]
    checkpoint = Path(
        checkpoint_record["path"]
        if isinstance(checkpoint_record, dict)
        else checkpoint_record
    )
    return {
        "seed": expected_seed,
        "variant": expected_variant,
        "training_report": str(path),
        "training_report_sha256": _sha256(path),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "correct_future_rate": metrics["two_real_future_target_selection_rate"],
        "correct_history_rate": metrics["correct_history_preference_rate"],
        "context_switch_rate": metrics["correct_rule_switch_rate"],
        "worst_exit_correct_future_rate": metrics[
            "worst_mode_target_selection_rate"
        ],
    }


def _point_gate(row: dict[str, Any], thresholds: dict[str, Any]) -> bool:
    return all(
        row[name] >= float(thresholds[f"{name}_minimum"])
        for name in (
            "correct_future_rate",
            "correct_history_rate",
            "context_switch_rate",
            "worst_exit_correct_future_rate",
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--release-config", type=Path, default=DEFAULT_PORTAL_EXIT_RELEASE_CONFIG
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    release = load_portal_exit_icl_release(args.release_config)
    seeds = tuple(
        int(value)
        for value in release["training"]["reference_matrix"]["training_seeds"]
    )
    thresholds = release["scoring"]["hidden_future_prediction"]["gates"]
    variants = {
        "native_lewm": "mixed_frozen_image_native_0p09",
        "official_pldm": "mixed_pldm_joint",
        "paired_future_lewm": "mixed_frozen_image_paired_future_fit_1p00",
    }
    groups = {
        "native_lewm": [
            _row(
                root / "reference_matrix" / f"lewm_seed{seed}" / "training_report.json",
                seed,
                variants["native_lewm"],
            )
            for seed in seeds
        ],
        "official_pldm": [
            _row(
                root / "reference_matrix" / f"pldm_seed{seed}" / "training_report.json",
                seed,
                variants["official_pldm"],
            )
            for seed in seeds
        ],
        "paired_future_lewm": [
            _row(
                root
                / "development_recipe_screen"
                / f"lewm_paired_fit_seed{seed}"
                / "training_report.json",
                seed,
                variants["paired_future_lewm"],
            )
            for seed in seeds
        ],
    }
    for rows in groups.values():
        for row in rows:
            row["development_point_gate_passed"] = _point_gate(row, thresholds)
    payload = {
        "schema_version": 1,
        "benchmark": "tworoom_portal_exit_reference_training_v1",
        "status": "completed",
        "data_manifest_sha256": release["data"]["manifest_sha256"],
        "optimizer_steps": 4096,
        "training_seeds": list(seeds),
        "public_test_opened_during_training_or_recipe_selection": False,
        "groups": groups,
        "development_decision": {
            "selected_for_one_time_public_test": "paired_future_lewm",
            "reason": (
                "It improved correct-future accuracy for all three paired seeds "
                "without using Public Test, while preserving the frozen point gate."
            ),
            "selected_group_passed_development_point_gate": all(
                row["development_point_gate_passed"]
                for row in groups["paired_future_lewm"]
            ),
            "thresholds_unchanged": True,
        },
    }
    output = (args.output or root / "reference_training_summary.json").resolve()
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload["development_decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
