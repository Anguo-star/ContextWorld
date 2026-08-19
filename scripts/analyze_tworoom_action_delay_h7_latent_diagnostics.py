#!/usr/bin/env python3
"""Aggregate post-gate History-7 latent-alignment diagnostics."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contextworld.evaluation.action_delay_h7_validation import file_sha256
from contextworld.paths import resolve_contextworld_path
from contextworld.synthesis.manifest import write_json


DEFAULT_INPUT = Path(
    "artifacts/evaluation/history7/action_delay_validation_v1/"
    "model_results/latent_diagnostics"
)

ROLES = (
    "original_only",
    "single_delay_control",
    "multi_delay_target",
)

METRICS = (
    "target_pair_mse",
    "prediction_pair_mse",
    "prediction_to_target_pair_magnitude_ratio",
    "pair_delta_alignment_mse_over_target_pair_mse",
    "pair_direction_cosine_mean",
    "pair_direction_positive_fraction",
    "pair_direction_gain_mean",
    "centered_delay_pattern_cosine_mean",
    "target_centered_variance",
    "prediction_centered_variance",
)


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values),
    }


def _aggregate(
    rows: list[dict[str, Any]],
    horizon: str,
) -> dict[str, dict[str, float]]:
    return {
        metric: _summary(
            [float(row["metrics"][horizon][metric]) for row in rows]
        )
        for metric in METRICS
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_root = resolve_contextworld_path(args.input_root, repo_root=ROOT)
    paths = sorted(input_root.glob("h7_*.json"))
    if len(paths) != 9:
        raise ValueError(f"Expected 9 diagnostic files, found {len(paths)}")
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if any(
        row.get("status")
        != "diagnostic_completed_not_part_of_primary_gate"
        for row in rows
    ):
        raise ValueError("One or more diagnostic files are incomplete")

    grouped: dict[str, list[dict[str, Any]]] = {
        role: [
            row for row in rows if row.get("training_role") == role
        ]
        for role in ROLES
    }
    if any(len(role_rows) != 3 for role_rows in grouped.values()):
        raise ValueError(
            "Expected exactly three training seeds for every role"
        )

    group_summary = {
        role: {
            "models": 3,
            "training_seeds": sorted(
                int(row["training_seed"]) for row in role_rows
            ),
            "trajectory": _aggregate(role_rows, "trajectory"),
            "by_horizon": {
                horizon: _aggregate(role_rows, horizon)
                for horizon in ("h1", "h2", "h3")
            },
        }
        for role, role_rows in grouped.items()
    }
    multi = group_summary["multi_delay_target"]["trajectory"]
    result = {
        "schema_version": 1,
        "benchmark": "tworoom_action_delay_history7_latent_diagnostic_v1",
        "status": "diagnostic_completed_not_part_of_primary_gate",
        "interpretation": {
            "target_latents_separate_delay_futures": (
                multi["target_pair_mse"]["mean"] > 0
            ),
            "prediction_change_fraction_of_target_magnitude": (
                multi[
                    "prediction_to_target_pair_magnitude_ratio"
                ]["mean"]
            ),
            "prediction_change_direction_cosine": (
                multi["pair_direction_cosine_mean"]["mean"]
            ),
            "prediction_change_direction_positive_fraction": (
                multi["pair_direction_positive_fraction"]["mean"]
            ),
            "conclusion": (
                "Multi-delay predictions change with history, but the "
                "change is too small and is not aligned with the true "
                "delay-conditioned future."
            ),
        },
        "group_summary": group_summary,
        "identity": {
            "entrypoint": str(Path(__file__).resolve()),
            "entrypoint_sha256": file_sha256(Path(__file__).resolve()),
            "input_files": {
                path.stem: file_sha256(path) for path in paths
            },
        },
    }
    output = resolve_contextworld_path(
        args.output
        or input_root / "latent_diagnostics_summary.json",
        repo_root=ROOT,
    )
    write_json(output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
