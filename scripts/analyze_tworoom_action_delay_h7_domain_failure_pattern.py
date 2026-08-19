#!/usr/bin/env python3
"""Summarize the post-gate History-7 Action Delay failure pattern.

This analysis is intentionally separate from the frozen formal scorer.  It
explains a failed Validation result; it cannot change that result or authorize
Test, CEM, or ability-retention evaluation.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contextworld.evaluation.action_delay_h7_validation import file_sha256
from contextworld.paths import (
    portable_contextworld_path,
    resolve_contextworld_path,
)
from contextworld.synthesis.manifest import write_json


DEFAULT_CONFIG = (
    ROOT
    / "configs/benchmark/"
    "tworoom_action_delay_h7_domain_diagnostic_scoring_v1.yaml"
)

LOSS_KEYS = ("loss", "pred_loss", "sigreg_loss")
SOURCE_METRICS = (
    "exact_target_selection_rate",
    "exact_history_selection_rate",
    "matching_history_strict_win_rate",
    "mean_history_loss_ratio",
)
ALIGNMENT_METRICS = (
    "target_pair_mse",
    "prediction_pair_mse",
    "prediction_to_target_pair_magnitude_ratio",
    "pair_direction_cosine_mean",
    "pair_direction_positive_fraction",
)
HORIZONS = ("h1", "h2", "h3", "trajectory")
DELAYS = (0, 4, 8)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _stats(values: list[float]) -> dict[str, float]:
    _require(bool(values), "不能汇总空数列")
    return {
        "mean": float(mean(values)),
        "sample_std": float(stdev(values)) if len(values) > 1 else 0.0,
        "count": len(values),
    }


def _track_parts(track: str) -> tuple[str, int]:
    if track.startswith("training_replay_delay_"):
        split = "training_replay"
    elif track.startswith("loader_validation_delay_"):
        split = "loader_validation"
    else:
        raise ValueError(f"未知诊断轨道：{track}")
    return split, int(track.rsplit("_", 1)[1])


def _models(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for role, models in config["models"].items():
        for model in models:
            rows.append({**model, "role": role})
    return rows


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_trace(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    _require(bool(rows), f"loss trace 为空：{path}")
    return rows


def _source_query_rows(
    result: dict[str, Any],
    track: str,
) -> list[dict[str, Any]]:
    rows = result["tracks"][track]["by_horizon"]["1"]["query_metrics"]
    return [
        row
        for row in rows
        if bool(row["is_source_supervised_target"])
    ]


def _plain(value: Any) -> Any:
    if isinstance(value, defaultdict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    return value


def _aggregate_confusion(
    confusion: dict[str, Any],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for role, splits in confusion.items():
        output[role] = {}
        for split, rows in splits.items():
            split_counts = Counter()
            source_rows: dict[str, Any] = {}
            for source_delay, counts in rows.items():
                counts = Counter(counts)
                split_counts.update(counts)
                units = sum(counts.values())
                source_rows[str(source_delay)] = {
                    "units": units,
                    "selected_target_counts": {
                        str(delay): int(counts[delay]) for delay in DELAYS
                    },
                    "selected_target_rates": {
                        str(delay): float(counts[delay] / units)
                        for delay in DELAYS
                    },
                    "correct_rate": float(
                        counts[int(source_delay)] / units
                    ),
                }
            total = sum(split_counts.values())
            dominant = max(
                DELAYS,
                key=lambda delay: (split_counts[delay], -delay),
            )
            output[role][split] = {
                "rows_by_true_source_delay": source_rows,
                "all_source_delays": {
                    "units": total,
                    "selected_target_counts": {
                        str(delay): int(split_counts[delay])
                        for delay in DELAYS
                    },
                    "selected_target_rates": {
                        str(delay): float(split_counts[delay] / total)
                        for delay in DELAYS
                    },
                    "dominant_selected_target": dominant,
                    "dominant_selected_target_rate": float(
                        split_counts[dominant] / total
                    ),
                },
            }
    return output


def _aggregate_nested_metrics(
    values: dict[str, Any],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for role, splits in values.items():
        output[role] = {}
        for split, source_delays in splits.items():
            output[role][split] = {}
            for source_delay, sections in source_delays.items():
                output[role][split][str(source_delay)] = {
                    section: {
                        metric: _stats(metric_values)
                        for metric, metric_values in metrics.items()
                    }
                    for section, metrics in sections.items()
                }
    return output


def _aggregate_alignment(
    values: dict[str, Any],
) -> dict[str, Any]:
    return {
        role: {
            split: {
                horizon: {
                    metric: _stats(metric_values)
                    for metric, metric_values in metrics.items()
                }
                for horizon, metrics in horizons.items()
            }
            for split, horizons in splits.items()
        }
        for role, splits in values.items()
    }


def _aggregate_loss_traces(
    traces: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for role, rows in traces.items():
        output[role] = {
            "models": len(rows),
            "training_seeds": sorted(
                int(row["training_seed"]) for row in rows
            ),
            "all_traces_complete": all(
                row["first_optimizer_step"] == 1
                and row["last_optimizer_step"] == 1024
                and row["records"] == 53
                for row in rows
            ),
            "every_final_pred_loss_below_first": all(
                row["final"]["pred_loss"] < row["first"]["pred_loss"]
                for row in rows
            ),
            "losses": {
                key: {
                    "first": _stats(
                        [float(row["first"][key]) for row in rows]
                    ),
                    "final": _stats(
                        [float(row["final"][key]) for row in rows]
                    ),
                    "last_five_mean": _stats(
                        [
                            float(row["last_five_mean"][key])
                            for row in rows
                        ]
                    ),
                    "first_to_final_reduction_fraction": _stats(
                        [
                            float(
                                (
                                    row["first"][key]
                                    - row["final"][key]
                                )
                                / row["first"][key]
                            )
                            for row in rows
                        ]
                    ),
                }
                for key in LOSS_KEYS
            },
            "per_training_seed": {
                str(row["training_seed"]): row for row in rows
            },
        }
    return output


def _multi_data_balance(
    reports: list[tuple[dict[str, Any], dict[str, Any]]],
) -> dict[str, Any]:
    rows = []
    for model, report in reports:
        if model["role"] != "multi_delay_target":
            continue
        group = report["data"]["groups"]["action_delay_multi"]
        exposure = report["training"]["plan"]["group_exposure"][
            "action_delay_multi"
        ]
        rows.append(
            {
                "training_seed": int(model["training_seed"]),
                "training_report_passed": report["passed"] is True,
                "group_weight": float(
                    report["data"]["group_weights"][
                        "action_delay_multi"
                    ]
                ),
                "raw_clips_per_delay": group["train_balancing"][
                    "raw_clips_per_factor"
                ],
                "scenarios_per_delay": group["train_balancing"][
                    "scenarios_per_factor"
                ],
                "total_synthetic_draws": int(exposure["total_draws"]),
                "unique_raw_clips_exposed": int(
                    exposure["unique_raw_clips_exposed"]
                ),
                "raw_clips_never_drawn": int(
                    exposure["raw_clips_never_drawn"]
                ),
                "run_unique_raw_fraction": float(
                    exposure["run_unique_raw_fraction"]
                ),
            }
        )
    _require(len(rows) == 3, "多延迟训练报告必须恰好有三个种子")
    expected_clips = {"0": 5120, "4": 5120, "8": 5120}
    expected_scenarios = {"0": 32, "4": 32, "8": 32}
    checks = {
        "all_training_reports_passed": all(
            row["training_report_passed"] for row in rows
        ),
        "delay_pool_balanced": all(
            row["raw_clips_per_delay"] == expected_clips
            and row["scenarios_per_delay"] == expected_scenarios
            for row in rows
        ),
        "every_synthetic_clip_exposed": all(
            row["raw_clips_never_drawn"] == 0
            and row["run_unique_raw_fraction"] == 1.0
            for row in rows
        ),
        "synthetic_group_weight_half": all(
            row["group_weight"] == 0.5 for row in rows
        ),
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "per_training_seed": {
            str(row["training_seed"]): row for row in rows
        },
    }


def analyze(config_path: Path, output_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    result_root = resolve_contextworld_path(
        config["artifacts"]["results_root"],
        repo_root=ROOT,
    )
    aggregate_path = resolve_contextworld_path(
        config["artifacts"]["aggregate"],
        repo_root=ROOT,
    )
    aggregate = _read_json(aggregate_path)
    _require(
        aggregate["status"] == "completed"
        and aggregate["diagnosis"] == "source_supervised_h1_not_fitted",
        "训练域诊断汇总状态或诊断不符合预期",
    )

    confusion: dict[str, Any] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(Counter))
    )
    nested_metrics: dict[str, Any] = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(lambda: defaultdict(list))
            )
        )
    )
    alignment_values: dict[str, Any] = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )
    )
    trace_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    report_rows: list[tuple[dict[str, Any], dict[str, Any]]] = []
    identities: dict[str, Any] = {
        "model_results": {},
        "training_reports": {},
        "loss_traces": {},
    }

    expected_queries = int(
        config["evaluation"]["queries_per_track"]
    )
    for model in _models(config):
        role = str(model["role"])
        slug = str(model["slug"])
        seed = int(model["training_seed"])
        result_path = result_root / f"{slug}.json"
        _require(result_path.is_file(), f"缺少模型结果：{result_path}")
        result = _read_json(result_path)
        _require(
            result["status"] == "completed"
            and result["model_slug"] == slug
            and int(result["training_seed"]) == seed,
            f"{slug}: 模型结果身份不一致",
        )
        identities["model_results"][slug] = {
            "path": portable_contextworld_path(
                result_path,
                repo_root=ROOT,
            ),
            "sha256": file_sha256(result_path),
        }

        report_path = resolve_contextworld_path(
            model["training_report"],
            repo_root=ROOT,
        )
        _require(
            file_sha256(report_path)
            == model["training_report_sha256"],
            f"{slug}: 训练报告哈希不一致",
        )
        report = _read_json(report_path)
        _require(
            report["passed"] is True
            and report["run_name"] == slug,
            f"{slug}: 训练报告审计失败",
        )
        report_rows.append((model, report))
        identities["training_reports"][slug] = {
            "path": portable_contextworld_path(
                report_path,
                repo_root=ROOT,
            ),
            "sha256": file_sha256(report_path),
        }

        trace_info = report["artifacts"]["loss_trace"]
        trace_path = resolve_contextworld_path(
            trace_info["path"],
            repo_root=ROOT,
        )
        _require(
            file_sha256(trace_path) == trace_info["sha256"],
            f"{slug}: loss trace 哈希不一致",
        )
        trace = _read_trace(trace_path)
        _require(
            len(trace) == int(trace_info["records"])
            and int(trace[0]["optimizer_step"]) == 1
            and int(trace[-1]["optimizer_step"]) == 1024,
            f"{slug}: loss trace 步数不完整",
        )
        first = {
            key: float(trace[0]["losses"][key]) for key in LOSS_KEYS
        }
        final = {
            key: float(trace[-1]["losses"][key]) for key in LOSS_KEYS
        }
        last_five = trace[-5:]
        trace_rows[role].append(
            {
                "training_seed": seed,
                "records": len(trace),
                "first_optimizer_step": int(
                    trace[0]["optimizer_step"]
                ),
                "last_optimizer_step": int(
                    trace[-1]["optimizer_step"]
                ),
                "first": first,
                "final": final,
                "last_five_mean": {
                    key: float(
                        mean(
                            float(row["losses"][key])
                            for row in last_five
                        )
                    )
                    for key in LOSS_KEYS
                },
            }
        )
        identities["loss_traces"][slug] = {
            "path": portable_contextworld_path(
                trace_path,
                repo_root=ROOT,
            ),
            "sha256": file_sha256(trace_path),
        }

        for track in config["evaluation"]["tracks"]:
            split, source_delay = _track_parts(str(track))
            _require(source_delay in DELAYS, f"未支持延迟：{track}")
            source_rows = _source_query_rows(result, str(track))
            _require(
                len(source_rows) == expected_queries,
                f"{slug}/{track}: source h1 数量错误",
            )
            _require(
                all(
                    int(row["source_delay"]) == source_delay
                    and int(row["target_delay"]) == source_delay
                    for row in source_rows
                ),
                f"{slug}/{track}: source h1 标签错误",
            )
            for row in source_rows:
                selected = int(row["selected_target"])
                _require(
                    selected in DELAYS,
                    f"{slug}/{track}: 未知 selected_target={selected}",
                )
                confusion[role][split][source_delay][selected] += 1

            source = result["tracks"][track]["by_horizon"]["1"][
                "source_supervised_target"
            ]["overall"]
            for metric in SOURCE_METRICS:
                nested_metrics[role][split][source_delay][
                    "source_supervised_h1"
                ][metric].append(float(source[metric]))
            trajectory = result["tracks"][track]["trajectory"]["overall"]
            for metric in SOURCE_METRICS:
                nested_metrics[role][split][source_delay][
                    "three_step_trajectory"
                ][metric].append(float(trajectory[metric]))
            for horizon in HORIZONS:
                alignment = result["tracks"][track][
                    "latent_alignment"
                ][horizon]
                for metric in ALIGNMENT_METRICS:
                    alignment_values[role][split][horizon][
                        metric
                    ].append(float(alignment[metric]))

    confusion_summary = _aggregate_confusion(_plain(confusion))
    metric_summary = _aggregate_nested_metrics(
        _plain(nested_metrics)
    )
    alignment_summary = _aggregate_alignment(
        _plain(alignment_values)
    )
    loss_summary = _aggregate_loss_traces(trace_rows)
    balance = _multi_data_balance(report_rows)

    multi_train = confusion_summary["multi_delay_target"][
        "training_replay"
    ]["all_source_delays"]
    multi_loader = confusion_summary["multi_delay_target"][
        "loader_validation"
    ]["all_source_delays"]
    train_loader_target_gaps = []
    for source_delay in DELAYS:
        train_rate = metric_summary["multi_delay_target"][
            "training_replay"
        ][str(source_delay)]["source_supervised_h1"][
            "exact_target_selection_rate"
        ]["mean"]
        loader_rate = metric_summary["multi_delay_target"][
            "loader_validation"
        ][str(source_delay)]["source_supervised_h1"][
            "exact_target_selection_rate"
        ]["mean"]
        train_loader_target_gaps.append(abs(train_rate - loader_rate))

    observations = {
        "multi_training_replay_selected_delay_4_rate": multi_train[
            "selected_target_rates"
        ]["4"],
        "multi_loader_validation_selected_delay_4_rate": multi_loader[
            "selected_target_rates"
        ]["4"],
        "maximum_train_loader_source_h1_target_rate_gap": max(
            train_loader_target_gaps
        ),
        "multi_training_h1_target_pair_mse": alignment_summary[
            "multi_delay_target"
        ]["training_replay"]["h1"]["target_pair_mse"]["mean"],
        "multi_training_h1_prediction_pair_mse": alignment_summary[
            "multi_delay_target"
        ]["training_replay"]["h1"]["prediction_pair_mse"]["mean"],
        "multi_training_h1_prediction_to_target_magnitude_ratio": (
            alignment_summary["multi_delay_target"][
                "training_replay"
            ]["h1"]["prediction_to_target_pair_magnitude_ratio"][
                "mean"
            ]
        ),
        "multi_training_h1_direction_cosine": alignment_summary[
            "multi_delay_target"
        ]["training_replay"]["h1"][
            "pair_direction_cosine_mean"
        ]["mean"],
        "multi_pred_loss_first_mean": loss_summary[
            "multi_delay_target"
        ]["losses"]["pred_loss"]["first"]["mean"],
        "multi_pred_loss_final_mean": loss_summary[
            "multi_delay_target"
        ]["losses"]["pred_loss"]["final"]["mean"],
    }
    conclusion_checks = {
        "same_delay_4_mode_on_train_and_loader": (
            multi_train["dominant_selected_target"] == 4
            and multi_loader["dominant_selected_target"] == 4
            and observations[
                "multi_training_replay_selected_delay_4_rate"
            ]
            > 0.95
            and observations[
                "multi_loader_validation_selected_delay_4_rate"
            ]
            > 0.95
        ),
        "train_loader_gap_is_small": observations[
            "maximum_train_loader_source_h1_target_rate_gap"
        ]
        < 0.02,
        "true_targets_are_separated": observations[
            "multi_training_h1_target_pair_mse"
        ]
        > 0.0,
        "history_induced_prediction_change_is_small": observations[
            "multi_training_h1_prediction_to_target_magnitude_ratio"
        ]
        < 0.10,
        "training_pred_loss_decreased": loss_summary[
            "multi_delay_target"
        ]["every_final_pred_loss_below_first"],
        "training_data_balance_passed": balance["passed"],
    }
    _require(
        all(conclusion_checks.values()),
        f"失败模式证据不完整：{conclusion_checks}",
    )

    payload = {
        "schema_version": 1,
        "benchmark": (
            "tworoom_action_delay_history7_post_gate_"
            "failure_pattern_v1"
        ),
        "status": "completed",
        "claim_boundary": {
            "post_gate_diagnostic_only": True,
            "changes_formal_validation_gate": False,
            "hidden_test_remains_sealed": True,
            "cem_authorized": False,
            "ability_retention_authorized": False,
        },
        "diagnosis": (
            "middle_delay_4_default_instead_of_"
            "history_conditioned_delay_adaptation"
        ),
        "plain_language_conclusion": (
            "多延迟模型在训练时见过和同分布未见的场景上，"
            "都几乎总把下一帧预测成延迟 4 的结果；"
            "它没有根据七帧历史在延迟 0、4、8 之间切换。"
        ),
        "formal_validation_gate_passed": False,
        "source_h1_selected_target_confusion": confusion_summary,
        "metrics_by_source_delay": metric_summary,
        "latent_alignment": alignment_summary,
        "training_loss_traces": loss_summary,
        "multi_delay_training_data_balance": balance,
        "root_cause_observations": observations,
        "root_cause_checks": conclusion_checks,
        "root_cause_checks_passed": all(
            conclusion_checks.values()
        ),
        "identity": {
            "analysis_entrypoint": portable_contextworld_path(
                Path(__file__).resolve(),
                repo_root=ROOT,
            ),
            "analysis_entrypoint_sha256": file_sha256(
                Path(__file__).resolve()
            ),
            "scoring_config": portable_contextworld_path(
                config_path,
                repo_root=ROOT,
            ),
            "scoring_config_sha256": file_sha256(config_path),
            "domain_diagnostic_summary": {
                "path": portable_contextworld_path(
                    aggregate_path,
                    repo_root=ROOT,
                ),
                "sha256": file_sha256(aggregate_path),
            },
            **identities,
        },
    }
    write_json(output_path, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config_path = args.config.expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    aggregate_path = resolve_contextworld_path(
        config["artifacts"]["aggregate"],
        repo_root=ROOT,
    )
    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else aggregate_path.with_name("failure_pattern_summary.json")
    )
    payload = analyze(config_path, output_path)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "diagnosis": payload["diagnosis"],
                "formal_validation_gate_passed": payload[
                    "formal_validation_gate_passed"
                ],
                "output": str(output_path),
                "output_sha256": file_sha256(output_path),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
