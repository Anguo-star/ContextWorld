#!/usr/bin/env python3
"""Aggregate the frozen v5 speed extrapolation/multi-step benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
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


HORIZONS = (1, 2, 3, 5)
TRAINING_SEEDS = (3072, 4096, 5120)


def _mean(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _models(config: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    return [
        (str(group), dict(model))
        for group, rows in config["models"].items()
        for model in rows
    ]


def _longest_contiguous(passes: dict[str, bool]) -> int:
    longest = 0
    for horizon in HORIZONS:
        if not passes[str(horizon)]:
            break
        longest = horizon
    return longest


def _query_level_metrics(
    records: list[dict[str, Any]], horizon: int
) -> dict[str, Any]:
    """Return paired metrics that are easier to interpret than percentages."""

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[
            (
                float(row["reference_speed"]),
                int(row["eval_seed"]),
                str(row["query_id"]),
            )
        ].append(row)

    by_speed: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for (speed, eval_seed, query_id), rows in grouped.items():
        matching_condition = str(rows[0]["matching_condition"])
        matching_rows = [
            row for row in rows if row["condition"] == matching_condition
        ]
        if len(matching_rows) != 1 or len(rows) < 2:
            raise RuntimeError(
                f"Incomplete query matrix: {speed} {eval_seed} {query_id}"
            )
        matching_loss = float(
            matching_rows[0]["latent_mse_by_horizon"][str(horizon)]
        )
        other_losses = [
            float(row["latent_mse_by_horizon"][str(horizon)])
            for row in rows
            if row["condition"] != matching_condition
        ]
        other_mean = float(np.mean(other_losses))
        by_speed[speed].append(
            {
                "matching_loss": matching_loss,
                "other_mean_loss": other_mean,
                "matching_beats_other_mean": matching_loss < other_mean,
                "matching_beats_every_other": all(
                    matching_loss < loss for loss in other_losses
                ),
            }
        )

    speed_rows = {}
    for speed, rows in sorted(by_speed.items()):
        matching = _mean([row["matching_loss"] for row in rows])
        other = _mean([row["other_mean_loss"] for row in rows])
        speed_rows[str(speed)] = {
            "queries": len(rows),
            "matching_loss": matching,
            "other_history_mean_loss": other,
            "matching_to_other_loss_ratio": float(
                matching / max(other, 1e-12)
            ),
            "query_win_rate_vs_other_mean": _mean(
                [float(row["matching_beats_other_mean"]) for row in rows]
            ),
            "strict_query_win_rate_vs_every_other": _mean(
                [float(row["matching_beats_every_other"]) for row in rows]
            ),
        }

    return {
        "reference_speed_balanced_matching_loss": _mean(
            [row["matching_loss"] for row in speed_rows.values()]
        ),
        "reference_speed_balanced_other_history_mean_loss": _mean(
            [row["other_history_mean_loss"] for row in speed_rows.values()]
        ),
        "reference_speed_balanced_matching_to_other_loss_ratio": _mean(
            [row["matching_to_other_loss_ratio"] for row in speed_rows.values()]
        ),
        "reference_speed_balanced_query_win_rate_vs_other_mean": _mean(
            [row["query_win_rate_vs_other_mean"] for row in speed_rows.values()]
        ),
        "reference_speed_balanced_strict_query_win_rate_vs_every_other": _mean(
            [
                row["strict_query_win_rate_vs_every_other"]
                for row in speed_rows.values()
            ]
        ),
        "by_reference_speed": speed_rows,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config_hash = file_sha256(config_path)
    root = resolve_contextworld_path(config["artifacts"]["root"], repo_root=ROOT)
    build_report_path = resolve_contextworld_path(
        config["artifacts"]["build_report"], repo_root=ROOT
    )
    build_report = json.loads(build_report_path.read_text(encoding="utf-8"))
    if build_report.get("status") != "passed":
        raise RuntimeError("Catalog build report did not pass")
    if build_report["config"]["sha256"] != config_hash:
        raise RuntimeError("Catalog/config hash mismatch")
    expected_stable_commit = str(config["stable_worldmodel"]["expected_ref"])
    if build_report["stable_worldmodel"]["commit"] != expected_stable_commit:
        raise RuntimeError("Build report/StableWM commit mismatch")
    build_report_hash = file_sha256(build_report_path)
    expected_normalizer = resolve_contextworld_path(
        config["evaluation"]["normalizer"], repo_root=ROOT
    )
    expected_normalizer_hash = file_sha256(expected_normalizer)

    models = {}
    input_files = []
    for group, model in _models(config):
        path = root / f"{model['slug']}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "passed":
            raise RuntimeError(f"Failed result: {path}")
        if payload["config"]["sha256"] != config_hash:
            raise RuntimeError(f"Config hash mismatch: {path}")
        if payload["model"]["group"] != group:
            raise RuntimeError(f"Group mismatch: {path}")
        if payload["model"]["slug"] != model["slug"]:
            raise RuntimeError(f"Model slug mismatch: {path}")
        if int(payload["model"]["training_seed"]) != int(
            model["training_seed"]
        ):
            raise RuntimeError(f"Training seed mismatch: {path}")
        expected_checkpoint = resolve_contextworld_path(
            model["checkpoint"], repo_root=ROOT
        )
        if Path(payload["model"]["checkpoint"]).resolve() != expected_checkpoint:
            raise RuntimeError(f"Checkpoint path mismatch: {path}")
        if payload["model"]["checkpoint_sha256"] != file_sha256(
            expected_checkpoint
        ):
            raise RuntimeError(f"Checkpoint hash mismatch: {path}")
        if payload["stable_worldmodel"]["commit"] != expected_stable_commit:
            raise RuntimeError(f"StableWM commit mismatch: {path}")
        if Path(payload["normalizer"]["path"]).resolve() != expected_normalizer:
            raise RuntimeError(f"Normalizer path mismatch: {path}")
        if payload["normalizer"]["sha256"] != expected_normalizer_hash:
            raise RuntimeError(f"Normalizer hash mismatch: {path}")
        if payload["build_report"]["sha256"] != build_report_hash:
            raise RuntimeError(f"Build report hash mismatch: {path}")
        if payload["online_environment_calls"] != 0:
            raise RuntimeError(f"Environment used during scoring: {path}")
        if not payload["frozen_weight_audit"]["passed"]:
            raise RuntimeError(f"Weight audit failed: {path}")
        track_rows = {}
        for track_name, track in payload["tracks"].items():
            expected_catalog_hash = build_report["tracks"][track_name][
                "catalog_sha256"
            ]
            if track["data_audit"]["catalog_sha256"] != expected_catalog_hash:
                raise RuntimeError(
                    f"Catalog hash mismatch: {path} {track_name}"
                )
            if not track["data_audit"][
                "all_eval_seed_queries_are_disjoint"
            ]:
                raise RuntimeError(
                    f"Eval-seed partitions overlap: {path} {track_name}"
                )
            if int(track["data_audit"]["unique_payload_hashes"]) != int(
                track["data_audit"]["bundles"]
            ):
                raise RuntimeError(
                    f"Repeated payload contents: {path} {track_name}"
                )
            if not track["summary"]["count_audit"]["passed"]:
                raise RuntimeError(f"Count audit failed: {path} {track_name}")
            if not track["autoregressive_prefix_audit"]["passed"]:
                raise RuntimeError(f"Prefix audit failed: {path} {track_name}")
            horizons = {}
            for horizon, row in track["summary"]["by_horizon"].items():
                query_metrics = _query_level_metrics(
                    track["records"], int(horizon)
                )
                expected_ratio = 1.0 - float(
                    row[
                        "reference_speed_balanced_relative_loss_reduction"
                    ]
                )
                observed_ratio = query_metrics[
                    "reference_speed_balanced_matching_to_other_loss_ratio"
                ]
                if not np.isclose(
                    observed_ratio, expected_ratio, atol=1e-10, rtol=1e-8
                ):
                    raise RuntimeError(
                        f"Loss-ratio audit failed: {path} {track_name} "
                        f"h={horizon}"
                    )
                horizons[horizon] = {
                    "reference_speed_balanced_relative_loss_reduction": float(
                        row[
                            "reference_speed_balanced_relative_loss_reduction"
                        ]
                    ),
                    "formal_within_checkpoint_pass": bool(
                        row["formal_within_checkpoint_pass"]
                    ),
                    "strict_each_alternative_pass": bool(
                        row["strict_each_alternative_pass"]
                    ),
                    "reader_facing_metrics": query_metrics,
                    "by_reference_speed": {
                        speed: {
                            "matching_loss": float(values["matching_loss"]),
                            "other_history_mean_loss": float(
                                values["other_history_mean_loss"]
                            ),
                            "matching_history_advantage": float(
                                values["matching_history_advantage"]
                            ),
                            "relative_loss_reduction": float(
                                values["relative_loss_reduction"]
                            ),
                            "relative_loss_reduction_ci": values[
                                "relative_loss_reduction_ci"
                            ],
                            "all_eval_seed_directions_positive": bool(
                                values[
                                    "all_eval_seed_directions_positive"
                                ]
                            ),
                            "matching_below_each_other_history": bool(
                                values[
                                    "matching_below_each_other_history"
                                ]
                            ),
                        }
                        for speed, values in row[
                            "by_reference_speed"
                        ].items()
                    },
                }
            track_rows[track_name] = {
                "horizons": horizons,
                "unique_payload_hashes": int(
                    track["data_audit"]["unique_payload_hashes"]
                ),
                "condition_trajectories": int(
                    track["summary"]["count_audit"][
                        "condition_trajectories"
                    ]
                ),
                "horizon_loss_records": int(
                    track["summary"]["count_audit"][
                        "horizon_loss_records"
                    ]
                ),
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
        tracks = {}
        for track in track_names:
            h_rows = {}
            h1 = _mean(
                [
                    row["tracks"][track]["horizons"]["1"][
                        "reference_speed_balanced_relative_loss_reduction"
                    ]
                    for row in selected
                ]
            )
            for horizon in HORIZONS:
                values = [
                    row["tracks"][track]["horizons"][str(horizon)][
                        "reference_speed_balanced_relative_loss_reduction"
                    ]
                    for row in selected
                ]
                mean_value = _mean(values)
                reader_rows = [
                    row["tracks"][track]["horizons"][str(horizon)][
                        "reader_facing_metrics"
                    ]
                    for row in selected
                ]
                h_rows[str(horizon)] = {
                    "mean_reference_speed_balanced_relative_loss_reduction": (
                        mean_value
                    ),
                    "retention_ratio_to_h1": (
                        float(mean_value / h1) if abs(h1) > 1e-12 else None
                    ),
                    "by_model": {
                        slug: row["tracks"][track]["horizons"][str(horizon)][
                            "reference_speed_balanced_relative_loss_reduction"
                        ]
                        for slug, row in models.items()
                        if row["group"] == group
                    },
                    "mean_matching_to_other_loss_ratio": _mean(
                        [
                            row[
                                "reference_speed_balanced_matching_to_other_loss_ratio"
                            ]
                            for row in reader_rows
                        ]
                    ),
                    "mean_query_win_rate_vs_other_mean": _mean(
                        [
                            row[
                                "reference_speed_balanced_query_win_rate_vs_other_mean"
                            ]
                            for row in reader_rows
                        ]
                    ),
                    "mean_strict_query_win_rate_vs_every_other": _mean(
                        [
                            row[
                                "reference_speed_balanced_strict_query_win_rate_vs_every_other"
                            ]
                            for row in reader_rows
                        ]
                    ),
                    "formal_within_checkpoint_passed_models": sum(
                        row["tracks"][track]["horizons"][str(horizon)][
                            "formal_within_checkpoint_pass"
                        ]
                        for row in selected
                    ),
                    "strict_each_alternative_passed_models": sum(
                        row["tracks"][track]["horizons"][str(horizon)][
                            "strict_each_alternative_pass"
                        ]
                        for row in selected
                    ),
                }
            tracks[track] = {"horizons": h_rows}
        group_summaries[group] = {
            "models": len(selected),
            "tracks": tracks,
        }

    paired_effects = {}
    for track in track_names:
        paired_effects[track] = {}
        for horizon in HORIZONS:
            rows = []
            for seed in TRAINING_SEEDS:
                single = models[f"h3_speed_single_v2_s{seed}"]["tracks"][track][
                    "horizons"
                ][str(horizon)][
                    "reference_speed_balanced_relative_loss_reduction"
                ]
                multi = models[f"h3_speed_multi_v2_s{seed}"]["tracks"][track][
                    "horizons"
                ][str(horizon)][
                    "reference_speed_balanced_relative_loss_reduction"
                ]
                single_reader = models[f"h3_speed_single_v2_s{seed}"][
                    "tracks"
                ][track]["horizons"][str(horizon)]["reader_facing_metrics"]
                multi_reader = models[f"h3_speed_multi_v2_s{seed}"][
                    "tracks"
                ][track]["horizons"][str(horizon)]["reader_facing_metrics"]
                rows.append(
                    {
                        "training_seed": seed,
                        "single_speed_relative_loss_reduction": single,
                        "multi_speed_relative_loss_reduction": multi,
                        "multi_minus_single": multi - single,
                        "single_speed_query_win_rate": single_reader[
                            "reference_speed_balanced_query_win_rate_vs_other_mean"
                        ],
                        "multi_speed_query_win_rate": multi_reader[
                            "reference_speed_balanced_query_win_rate_vs_other_mean"
                        ],
                        "query_win_rate_multi_minus_single": multi_reader[
                            "reference_speed_balanced_query_win_rate_vs_other_mean"
                        ]
                        - single_reader[
                            "reference_speed_balanced_query_win_rate_vs_other_mean"
                        ],
                        "single_speed_strict_query_win_rate": single_reader[
                            "reference_speed_balanced_strict_query_win_rate_vs_every_other"
                        ],
                        "multi_speed_strict_query_win_rate": multi_reader[
                            "reference_speed_balanced_strict_query_win_rate_vs_every_other"
                        ],
                        "strict_query_win_rate_multi_minus_single": multi_reader[
                            "reference_speed_balanced_strict_query_win_rate_vs_every_other"
                        ]
                        - single_reader[
                            "reference_speed_balanced_strict_query_win_rate_vs_every_other"
                        ],
                    }
                )
            paired_effects[track][str(horizon)] = {
                "by_training_seed": rows,
                "mean_multi_minus_single": _mean(
                    [row["multi_minus_single"] for row in rows]
                ),
                "all_three_directions_positive": all(
                    row["multi_minus_single"] > 0 for row in rows
                ),
                "mean_query_win_rate_multi_minus_single": _mean(
                    [row["query_win_rate_multi_minus_single"] for row in rows]
                ),
                "mean_strict_query_win_rate_multi_minus_single": _mean(
                    [
                        row["strict_query_win_rate_multi_minus_single"]
                        for row in rows
                    ]
                ),
            }

    multi_models = [
        row for row in models.values() if row["group"] == "multi_speed_target"
    ]
    formal_track_horizon = {}
    strict_track_horizon = {}
    for track in track_names:
        formal_track_horizon[track] = {}
        strict_track_horizon[track] = {}
        for horizon in HORIZONS:
            formal_track_horizon[track][str(horizon)] = bool(
                all(
                    row["tracks"][track]["horizons"][str(horizon)][
                        "formal_within_checkpoint_pass"
                    ]
                    for row in multi_models
                )
                and paired_effects[track][str(horizon)][
                    "all_three_directions_positive"
                ]
            )
            strict_track_horizon[track][str(horizon)] = bool(
                all(
                    row["tracks"][track]["horizons"][str(horizon)][
                        "strict_each_alternative_pass"
                    ]
                    for row in multi_models
                )
            )
    longest = {
        track: _longest_contiguous(formal_track_horizon[track])
        for track in track_names
    }
    extrap_low_h1 = formal_track_horizon["extrapolation_low"]["1"]
    extrap_high_h1 = formal_track_horizon["extrapolation_high"]["1"]
    total_trajectories = sum(
        track["condition_trajectories"]
        for row in models.values()
        for track in row["tracks"].values()
    )
    total_horizon_losses = sum(
        track["horizon_loss_records"]
        for row in models.values()
        for track in row["tracks"].values()
    )
    expected_per_model = int(
        config["evaluation"][
            "condition_trajectories_per_checkpoint_all_tracks"
        ]
    )
    expected_horizon_per_model = int(
        config["evaluation"]["horizon_losses_per_checkpoint_all_tracks"]
    )
    all_payloads_unique = all(
        track["unique_payload_hashes"]
        == track["condition_trajectories"]
        // len(config["data"]["tracks"][track_name]["speeds"])
        for row in models.values()
        for track_name, track in row["tracks"].items()
    )
    all_query_pixels_unique = all(
        int(track["summary"]["unique_query_pixel_hashes"])
        == int(
            config["evaluation"]["unique_queries_per_reference_speed"]
        )
        for track in build_report["tracks"].values()
    )
    if not all_payloads_unique or not all_query_pixels_unique:
        raise RuntimeError("Deterministic query content uniqueness audit failed")
    payload = {
        "schema_version": 1,
        "benchmark": config["benchmark"],
        "status": "passed",
        "config": {"path": str(config_path), "sha256": config_hash},
        "build_report": {
            "path": str(build_report_path),
            "sha256": file_sha256(build_report_path),
        },
        "speed_support_audit": build_report["speed_support_audit"],
        "input_files": input_files,
        "count_audit": {
            "models": len(models),
            "tracks_per_model": len(track_names),
            "condition_trajectories": total_trajectories,
            "expected_condition_trajectories": len(models) * expected_per_model,
            "horizon_loss_records": total_horizon_losses,
            "expected_horizon_loss_records": (
                len(models) * expected_horizon_per_model
            ),
            "online_environment_calls": 0,
            "all_deterministic_queries_disjoint_across_eval_seeds": True,
            "all_catalog_payload_hashes_unique": all_payloads_unique,
            "all_tracks_have_300_unique_query_pixels": (
                all_query_pixels_unique
            ),
        },
        "models": models,
        "group_summaries": group_summaries,
        "paired_training_seed_effects": paired_effects,
        "decision": {
            "formal_pass_by_track_and_horizon": formal_track_horizon,
            "strict_diagnostic_by_track_and_horizon": strict_track_horizon,
            "longest_contiguous_passing_horizon_by_track": longest,
            "one_step_low_extrapolation_passed": extrap_low_h1,
            "one_step_high_extrapolation_passed": extrap_high_h1,
            "bilateral_one_step_extrapolation_passed": bool(
                extrap_low_h1 and extrap_high_h1
            ),
            "bilateral_extrapolation_common_longest_horizon": min(
                longest["extrapolation_low"],
                longest["extrapolation_high"],
            ),
            "formal_definition": (
                "Per track and horizon: every multi-speed checkpoint has "
                "positive matching-history advantage at every reference "
                "speed and in every one of six eval-seed partitions; the "
                "reference-speed-equal multi-minus-single effect is positive "
                "for all three paired training seeds."
            ),
        },
    }
    if total_trajectories != len(models) * expected_per_model:
        raise RuntimeError("Condition trajectory count mismatch")
    if total_horizon_losses != len(models) * expected_horizon_per_model:
        raise RuntimeError("Horizon loss count mismatch")
    write_json(args.output.resolve(), payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "configs/benchmark/tworoom_speed_multistep_extrap_v5.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=resolve_contextworld_path(
            "artifacts/evaluation/history3/speed_multistep_extrap_v5/final_summary.json",
            repo_root=ROOT,
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps(result["decision"], indent=2, sort_keys=True))
