#!/usr/bin/env python3
"""汇总动作延迟多步与高端延迟扩展的正式结果。"""

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

from contextworld.evaluation.action_delay_multistep import (
    DELAYS,
    EVAL_SEEDS,
    HIGH_ENDPOINT_DELAYS,
    HORIZONS,
    INTERPOLATION_DELAYS,
    LOSS_RECORDS_PER_CHECKPOINT,
    PREDICTIONS_PER_CHECKPOINT,
    QUERY_COUNT,
    TARGET_ENCODINGS_PER_CHECKPOINT,
    TRAINING_SEEN_DELAYS,
)
from contextworld.evaluation.action_delay_validation import file_sha256
from contextworld.paths import resolve_contextworld_path
from contextworld.synthesis.manifest import write_json


DEFAULT_CONFIG = (
    ROOT
    / "configs/benchmark/tworoom_action_delay_h3_multistep_extrap_v1.yaml"
)
FORMAL_SEEDS = (3072, 4096, 5120)
METRICS = (
    "mean_matching_history_loss",
    "mean_other_history_loss",
    "mean_history_margin",
    "mean_history_loss_ratio",
    "matching_history_strict_win_rate",
    "history_selection_accuracy",
    "target_selection_accuracy",
)
DISPLAY_NAMES = {
    "original_reference": "原始 TwoRoom 数据训练模型",
    "single_delay_control": "原始数据 + 单一动作延迟",
    "multi_delay_target": "原始数据 + 多种动作延迟",
}
TRACK_DELAYS = {
    "training_seen": TRAINING_SEEN_DELAYS,
    "interpolation": INTERPOLATION_DELAYS,
    "high_endpoint_extrapolation": HIGH_ENDPOINT_DELAYS,
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _mean_std(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
    }


def _project(summary: dict[str, Any]) -> dict[str, float]:
    return {metric: float(summary[metric]) for metric in METRICS}


def _load_result(
    path: Path,
    *,
    benchmark: str,
    slug: str,
    seed: int,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(payload.get("status") == "completed", f"结果未完成：{path}")
    _require(payload.get("benchmark") == benchmark, f"benchmark 不一致：{path}")
    _require(payload.get("model_slug") == slug, f"模型身份不一致：{path}")
    _require(int(payload.get("training_seed")) == seed, f"训练种子不一致：{path}")
    audit = payload["score_audit"]
    _require(
        audit["queries"] == QUERY_COUNT
        and audit["model_rollouts"] == PREDICTIONS_PER_CHECKPOINT
        and audit["target_encodings"] == TARGET_ENCODINGS_PER_CHECKPOINT
        and audit["horizon_loss_records"] == LOSS_RECORDS_PER_CHECKPOINT
        and audit["online_environment_calls"] == 0,
        f"结果计数不完整：{path}",
    )
    return payload


def _aggregate_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    by_horizon = {}
    for horizon in HORIZONS:
        horizon_key = str(horizon)
        overall_rows = [
            run["summary"]["by_horizon"][horizon_key]["overall"]
            for run in runs
        ]
        by_horizon[horizon_key] = {
            "overall": {
                metric: _mean_std(
                    [float(row[metric]) for row in overall_rows]
                )
                for metric in METRICS
            },
            "by_track": {
                track: {
                    metric: _mean_std(
                        [
                            float(
                                run["summary"]["by_horizon"][horizon_key][
                                    "by_track"
                                ][track][metric]
                            )
                            for run in runs
                        ]
                    )
                    for metric in METRICS
                }
                for track in TRACK_DELAYS
            },
            "by_target_delay": {
                str(delay): {
                    metric: _mean_std(
                        [
                            float(
                                run["summary"]["by_horizon"][horizon_key][
                                    "by_target_delay"
                                ][str(delay)][metric]
                            )
                            for run in runs
                        ]
                    )
                    for metric in METRICS
                }
                for delay in DELAYS
            },
        }
    return {
        "training_seeds": [int(run["training_seed"]) for run in runs],
        "by_horizon": by_horizon,
    }


def _multi_delay_gate(
    run: dict[str, Any],
    *,
    delay: int,
    horizon: int,
) -> dict[str, Any]:
    horizon_summary = run["summary"]["by_horizon"][str(horizon)]
    summary = horizon_summary["by_target_delay"][str(delay)]
    by_eval_seed = horizon_summary["by_target_delay_and_eval_seed"][
        str(delay)
    ]
    checks = {
        "mean_history_margin_positive": (
            float(summary["mean_history_margin"]) > 0.0
        ),
        "all_six_eval_seed_margins_positive": all(
            float(by_eval_seed[str(seed)]["mean_history_margin"]) > 0.0
            for seed in EVAL_SEEDS
        ),
        "history_selection_accuracy_at_least_0_60": (
            float(summary["history_selection_accuracy"]) >= 0.60
        ),
        "target_selection_accuracy_at_least_0_60": (
            float(summary["target_selection_accuracy"]) >= 0.60
        ),
        "strict_win_rate_at_least_0_50": (
            float(summary["matching_history_strict_win_rate"]) >= 0.50
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": _project(summary),
    }


def _paired_track_gate(
    multi: dict[str, Any],
    single: dict[str, Any],
    *,
    track: str,
    horizon: int,
) -> dict[str, Any]:
    multi_summary = multi["summary"]["by_horizon"][str(horizon)][
        "by_track"
    ][track]
    single_summary = single["summary"]["by_horizon"][str(horizon)][
        "by_track"
    ][track]
    deltas = {
        "history_selection_accuracy": float(
            multi_summary["history_selection_accuracy"]
            - single_summary["history_selection_accuracy"]
        ),
        "target_selection_accuracy": float(
            multi_summary["target_selection_accuracy"]
            - single_summary["target_selection_accuracy"]
        ),
        "strict_win_rate": float(
            multi_summary["matching_history_strict_win_rate"]
            - single_summary["matching_history_strict_win_rate"]
        ),
    }
    checks = {
        "multi_minus_single_history_selection_accuracy_positive": (
            deltas["history_selection_accuracy"] > 0.0
        ),
        "multi_minus_single_target_selection_accuracy_positive": (
            deltas["target_selection_accuracy"] > 0.0
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "deltas": deltas,
    }


def _longest_contiguous_passing_horizon(
    track_horizons: dict[str, dict[str, Any]],
) -> int:
    longest = 0
    for horizon in HORIZONS:
        if not track_horizons[str(horizon)]["passed"]:
            break
        longest = int(horizon)
    return longest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--results-root", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    benchmark = str(config["benchmark"])
    _require(
        benchmark
        in {
            "tworoom_action_delay_history3_multistep_extrap_v1",
            "tworoom_action_delay_history3_multistep_extrap_v2",
        },
        "不是冻结的动作延迟多步扩展配置",
    )
    results_root = resolve_contextworld_path(
        (
            args.results_root
            if args.results_root is not None
            else config["artifacts"]["results"]
        ),
        repo_root=ROOT,
    )
    output = resolve_contextworld_path(
        (
            args.output
            if args.output is not None
            else config["artifacts"]["final_summary"]
        ),
        repo_root=ROOT,
    )

    group_runs: dict[str, list[dict[str, Any]]] = {}
    result_files = {}
    slug_to_run = {}
    for group_name, rows in config["models"].items():
        group_runs[group_name] = []
        for row in rows:
            slug = str(row["slug"])
            seed = int(row["training_seed"])
            path = results_root / f"{slug}.json"
            _require(path.is_file(), f"缺少正式结果：{path}")
            run = _load_result(
                path,
                benchmark=benchmark,
                slug=slug,
                seed=seed,
            )
            group_runs[group_name].append(run)
            slug_to_run[slug] = run
            result_files[slug] = {
                "path": str(path),
                "sha256": file_sha256(path),
            }

    multi_by_seed = {
        int(run["training_seed"]): run
        for run in group_runs["multi_delay_target"]
    }
    single_by_seed = {
        int(run["training_seed"]): run
        for run in group_runs["single_delay_control"]
    }
    _require(
        tuple(sorted(multi_by_seed)) == FORMAL_SEEDS
        and tuple(sorted(single_by_seed)) == FORMAL_SEEDS,
        "单延迟和多延迟模型必须使用相同的三个正式训练种子",
    )

    multi_gates = {
        str(seed): {
            str(horizon): {
                str(delay): _multi_delay_gate(
                    multi_by_seed[seed],
                    delay=delay,
                    horizon=horizon,
                )
                for delay in DELAYS
            }
            for horizon in HORIZONS
        }
        for seed in FORMAL_SEEDS
    }
    paired_gates = {
        str(seed): {
            str(horizon): {
                track: _paired_track_gate(
                    multi_by_seed[seed],
                    single_by_seed[seed],
                    track=track,
                    horizon=horizon,
                )
                for track in TRACK_DELAYS
            }
            for horizon in HORIZONS
        }
        for seed in FORMAL_SEEDS
    }

    track_horizons = {}
    for track, delays in TRACK_DELAYS.items():
        track_horizons[track] = {}
        for horizon in HORIZONS:
            seed_delay_pass = all(
                multi_gates[str(seed)][str(horizon)][str(delay)][
                    "passed"
                ]
                for seed in FORMAL_SEEDS
                for delay in delays
            )
            paired_pass = all(
                paired_gates[str(seed)][str(horizon)][track]["passed"]
                for seed in FORMAL_SEEDS
            )
            track_horizons[track][str(horizon)] = {
                "passed": bool(seed_delay_pass and paired_pass),
                "all_multi_seed_delay_gates_passed": bool(
                    seed_delay_pass
                ),
                "all_paired_attribution_gates_passed": bool(paired_pass),
            }

    persistence = {
        track: {
            "longest_contiguous_passing_horizon": (
                _longest_contiguous_passing_horizon(horizons)
            ),
            "passes_through_h5": (
                _longest_contiguous_passing_horizon(horizons) == 5
            ),
            "by_horizon": horizons,
        }
        for track, horizons in track_horizons.items()
    }
    conclusions = {
        "new_queries_confirm_one_step_action_delay_icl": all(
            persistence[track]["longest_contiguous_passing_horizon"] >= 1
            for track in TRACK_DELAYS
        ),
        "training_seen_one_step_passed": (
            persistence["training_seen"][
                "longest_contiguous_passing_horizon"
            ]
            >= 1
        ),
        "interpolation_one_step_passed": (
            persistence["interpolation"][
                "longest_contiguous_passing_horizon"
            ]
            >= 1
        ),
        "delay5_high_endpoint_one_step_passed": (
            persistence["high_endpoint_extrapolation"][
                "longest_contiguous_passing_horizon"
            ]
            >= 1
        ),
        "training_seen_response_persists_through_h5": persistence[
            "training_seen"
        ]["passes_through_h5"],
        "interpolation_response_persists_through_h5": persistence[
            "interpolation"
        ]["passes_through_h5"],
        "delay5_high_endpoint_response_persists_through_h5": persistence[
            "high_endpoint_extrapolation"
        ]["passes_through_h5"],
        "delay5_is_only_one_endpoint_not_a_broad_range": True,
    }
    conclusions["ready_for_one_step_speed_delay_combination_design"] = (
        conclusions["new_queries_confirm_one_step_action_delay_icl"]
    )
    conclusions[
        "ready_for_multistep_or_planning_speed_delay_combination"
    ] = all(
        (
            conclusions["training_seen_response_persists_through_h5"],
            conclusions["interpolation_response_persists_through_h5"],
            conclusions[
                "delay5_high_endpoint_response_persists_through_h5"
            ],
        )
    )

    result = {
        "schema_version": 1,
        "benchmark": benchmark,
        "status": "completed",
        "identity": {
            "config": str(config_path),
            "config_sha256": file_sha256(config_path),
        },
        "result_files": result_files,
        "model_display_names": DISPLAY_NAMES,
        "tracks": {
            track: list(delays) for track, delays in TRACK_DELAYS.items()
        },
        "models": {
            group: _aggregate_runs(runs)
            for group, runs in group_runs.items()
        },
        "frozen_gates": {
            "multi_delay_by_training_seed_horizon_and_delay": (
                multi_gates
            ),
            "paired_multi_minus_single_by_seed_horizon_and_track": (
                paired_gates
            ),
            "track_persistence": persistence,
        },
        "conclusions": conclusions,
    }
    write_json(output, result)
    print(
        json.dumps(
            {
                "output": str(output),
                "conclusions": conclusions,
                "track_persistence": persistence,
                "multi_delay_overall": {
                    horizon: {
                        metric: values["overall"][metric]["mean"]
                        for metric in (
                            "history_selection_accuracy",
                            "target_selection_accuracy",
                            "matching_history_strict_win_rate",
                            "mean_history_loss_ratio",
                        )
                    }
                    for horizon, values in result["models"][
                        "multi_delay_target"
                    ]["by_horizon"].items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
