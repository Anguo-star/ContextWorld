#!/usr/bin/env python3
"""Aggregate the frozen History-3 speed next-frame latent benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contextworld.evaluation.icl_model import file_sha256
from contextworld.paths import resolve_contextworld_path
from contextworld.synthesis.manifest import write_json


def _models(config: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    return [
        (str(group), dict(model))
        for group, models in config["models"].items()
        for model in models
    ]


def _formal_track_pass(track: dict[str, Any]) -> bool:
    return all(
        row["matching_history_advantage"] > 0
        and row["all_eval_seed_directions_positive"]
        for row in track["summary"]["by_reference_speed"].values()
    )


def _strict_track_pass(track: dict[str, Any]) -> bool:
    return bool(
        track["summary"]["decision"][
            "matching_below_each_other_history_all_speeds"
        ]
    )


def _mean(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _speed_support_audit(
    config: dict[str, Any], training_config_path: Path
) -> dict[str, Any]:
    training = yaml.safe_load(training_config_path.read_text(encoding="utf-8"))
    support = training["speed_support"]
    original = set(map(float, support["original_train"]))
    multi = set(map(float, support["multi_synthetic_train"]))
    monitor = set(map(float, support["training_monitor_only"]))
    calibration = set(map(float, support["planner_calibration"]))
    sealed_test = set(map(float, support["sealed_test_interpolation"]))
    seen = set(map(float, config["data"]["tracks"]["seen_for_multi"]["speeds"]))
    unseen = set(
        map(
            float,
            config["data"]["tracks"]["unseen_interpolation"]["speeds"],
        )
    )
    checks = {
        "seen_is_multi_train_subset": seen <= multi,
        "unseen_disjoint_original_train": not unseen & original,
        "unseen_disjoint_multi_train": not unseen & multi,
        "unseen_disjoint_training_monitor": not unseen & monitor,
        "unseen_disjoint_planner_calibration": not unseen & calibration,
        "sealed_test_not_opened_by_validation": not sealed_test & (seen | unseen),
        "unseen_inside_multi_train_range": (
            min(multi) < min(unseen) < max(unseen) < max(multi)
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Speed support isolation failed: {checks}")
    return {
        "passed": True,
        "source": str(training_config_path),
        "source_sha256": file_sha256(training_config_path),
        "checks": checks,
        "seen_for_multi": sorted(seen),
        "unseen_interpolation": sorted(unseen),
        "multi_train_count": len(multi),
        "multi_train_range": [min(multi), max(multi)],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    root = resolve_contextworld_path(config["artifacts"]["root"], repo_root=ROOT)
    config_hash = file_sha256(config_path)
    speed_support_audit = _speed_support_audit(
        config, args.training_config.resolve()
    )
    models = {}
    input_files = []
    for group, model in _models(config):
        path = root / f"{model['slug']}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "passed":
            raise RuntimeError(f"Failed input: {path}")
        if payload["config"]["sha256"] != config_hash:
            raise RuntimeError(f"Config hash mismatch: {path}")
        if payload["model"]["group"] != group:
            raise RuntimeError(f"Model group mismatch: {path}")
        if payload["online_environment_calls"] != 0:
            raise RuntimeError(f"Online environment was used: {path}")
        if not payload["frozen_weight_audit"]["passed"]:
            raise RuntimeError(f"Weight audit failed: {path}")
        track_rows = {}
        for track_name, track in payload["tracks"].items():
            count_audit = track["summary"]["count_audit"]
            if not count_audit["passed"]:
                raise RuntimeError(f"Count audit failed: {path} {track_name}")
            if not count_audit[
                "all_queries_unique_within_eval_seed_cells"
            ]:
                raise RuntimeError(
                    f"Repeated deterministic query: {path} {track_name}"
                )
            overall = track["summary"]["overall"]
            speed_reductions = [
                float(row["relative_loss_reduction"])
                for row in track["summary"]["by_reference_speed"].values()
            ]
            track_rows[track_name] = {
                "pooled_relative_loss_reduction": float(
                    overall["relative_loss_reduction"]
                ),
                "pooled_relative_loss_reduction_ci": overall[
                    "relative_loss_reduction_ci"
                ],
                "reference_speed_balanced_relative_loss_reduction": _mean(
                    speed_reductions
                ),
                "by_reference_speed": {
                    speed: {
                        "matching_loss": float(row["matching_loss"]),
                        "other_history_mean_loss": float(
                            row["other_history_mean_loss"]
                        ),
                        "relative_loss_reduction": float(
                            row["relative_loss_reduction"]
                        ),
                        "relative_loss_reduction_ci": row[
                            "relative_loss_reduction_ci"
                        ],
                        "matching_below_each_other_history": bool(
                            row["matching_below_each_other_history"]
                        ),
                        "all_eval_seed_directions_positive": bool(
                            row["all_eval_seed_directions_positive"]
                        ),
                    }
                    for speed, row in track["summary"][
                        "by_reference_speed"
                    ].items()
                },
                "formal_primary_pass": _formal_track_pass(track),
                "strict_both_alternatives_pass": _strict_track_pass(track),
                "records": int(count_audit["records"]),
            }
        models[model["slug"]] = {
            "group": group,
            "training_seed": int(model["training_seed"]),
            "result": str(path),
            "tracks": track_rows,
        }
        input_files.append({"path": str(path), "sha256": file_sha256(path)})

    track_names = list(config["data"]["tracks"])
    group_summaries = {}
    for group in config["models"]:
        selected = [row for row in models.values() if row["group"] == group]
        group_summaries[group] = {
            "models": len(selected),
            "tracks": {
                track: {
                    "mean_reference_speed_balanced_relative_loss_reduction": _mean(
                        [
                            row["tracks"][track][
                                "reference_speed_balanced_relative_loss_reduction"
                            ]
                            for row in selected
                        ]
                    ),
                    "by_model": {
                        slug: row["tracks"][track][
                            "reference_speed_balanced_relative_loss_reduction"
                        ]
                        for slug, row in models.items()
                        if row["group"] == group
                    },
                    "formal_primary_passed_models": sum(
                        row["tracks"][track]["formal_primary_pass"]
                        for row in selected
                    ),
                    "strict_both_alternatives_passed_models": sum(
                        row["tracks"][track][
                            "strict_both_alternatives_pass"
                        ]
                        for row in selected
                    ),
                }
                for track in track_names
            },
        }

    paired_effects = {}
    for track in track_names:
        rows = []
        for seed in (3072, 4096, 5120):
            single = models[f"h3_speed_single_v2_s{seed}"]["tracks"][track][
                "reference_speed_balanced_relative_loss_reduction"
            ]
            multi = models[f"h3_speed_multi_v2_s{seed}"]["tracks"][track][
                "reference_speed_balanced_relative_loss_reduction"
            ]
            rows.append(
                {
                    "training_seed": seed,
                    "single_speed_relative_loss_reduction": single,
                    "multi_speed_relative_loss_reduction": multi,
                    "multi_minus_single": multi - single,
                }
            )
        paired_effects[track] = {
            "by_training_seed": rows,
            "mean_multi_minus_single": _mean(
                [row["multi_minus_single"] for row in rows]
            ),
            "all_three_directions_positive": all(
                row["multi_minus_single"] > 0 for row in rows
            ),
        }

    multi_models = [
        row for row in models.values() if row["group"] == "multi_speed_target"
    ]
    formal_pass = all(
        row["tracks"][track]["formal_primary_pass"]
        for row in multi_models
        for track in track_names
    ) and all(
        paired_effects[track]["all_three_directions_positive"]
        for track in track_names
    )
    strict_pass = all(
        row["tracks"][track]["strict_both_alternatives_pass"]
        for row in multi_models
        for track in track_names
    )
    expected_records = int(
        config["evaluation"]["expected_records_per_checkpoint_per_track"]
    )
    payload = {
        "schema_version": 1,
        "benchmark": config["benchmark"],
        "status": "passed",
        "config": {"path": str(config_path), "sha256": config_hash},
        "input_files": input_files,
        "speed_support_audit": speed_support_audit,
        "count_audit": {
            "models": len(models),
            "tracks_per_model": len(track_names),
            "records_per_model_per_track": expected_records,
            "total_records": sum(
                track["records"]
                for row in models.values()
                for track in row["tracks"].values()
            ),
            "expected_total_records": (
                len(models) * len(track_names) * expected_records
            ),
            "unique_queries_per_reference_speed_per_seed": int(
                config["evaluation"][
                    "unique_queries_per_reference_speed_per_seed"
                ]
            ),
            "all_deterministic_queries_disjoint_within_cells": True,
            "online_environment_calls": 0,
        },
        "models": models,
        "group_summaries": group_summaries,
        "paired_training_seed_effects": paired_effects,
        "decision": {
            "formal_speed_icl_passed": formal_pass,
            "strict_both_alternatives_passed": strict_pass,
            "formal_definition": (
                "At every reference speed, matching-history loss is below "
                "the mean of the two other-history losses with positive "
                "direction in all six eval seeds; multi-speed exceeds its "
                "paired single-speed control for all three training seeds."
            ),
            "strict_diagnostic_definition": (
                "Matching-history mean loss is separately below each of the "
                "two other-history mean losses for every multi-speed model, "
                "track, and reference speed."
            ),
        },
    }
    if payload["count_audit"]["total_records"] != payload["count_audit"][
        "expected_total_records"
    ]:
        raise RuntimeError("Total record count mismatch")
    write_json(args.output.resolve(), payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "configs/benchmark/tworoom_speed_next_latent_v4.yaml",
    )
    parser.add_argument(
        "--training-config",
        type=Path,
        default=ROOT / "configs/benchmark/tworoom_speed_isolated_v2.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=resolve_contextworld_path(
            "artifacts/evaluation/history3/speed_next_latent_v4/final_summary.json",
            repo_root=ROOT,
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps(result["decision"], indent=2, sort_keys=True))
