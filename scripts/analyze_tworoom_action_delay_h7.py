#!/usr/bin/env python3
"""Aggregate all nine formal History-7 Action Delay results."""

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

from contextworld.evaluation.action_delay_h7_validation import file_sha256
from contextworld.paths import resolve_contextworld_path
from contextworld.synthesis.manifest import write_json


DEFAULT_CONFIG = (
    ROOT / "configs/benchmark/tworoom_action_delay_h7_scoring_v1.yaml"
)
RATE_METRICS = (
    "exact_history_selection_rate",
    "exact_target_selection_rate",
    "matching_history_strict_win_rate",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _mean_std(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
    }


def _models(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result = {}
    for role, rows in config["models"].items():
        for row in rows:
            slug = str(row["slug"])
            _require(slug not in result, f"模型 slug 重复：{slug}")
            result[slug] = {**row, "role": str(role)}
    return result


def _load_results(
    config: dict[str, Any],
    *,
    config_path: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    root = resolve_contextworld_path(
        config["artifacts"]["results_root"], repo_root=ROOT
    )
    expected_config_hash = file_sha256(config_path)
    loaded = {}
    files = {}
    checkpoint_hashes = set()
    for slug, model in _models(config).items():
        path = root / f"{slug}.json"
        _require(path.is_file(), f"缺少正式评分结果：{path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        checks = {
            "status_completed": payload.get("status") == "completed",
            "slug_exact": payload.get("model_slug") == slug,
            "model_id_exact": payload.get("model_id")
            == model["model_id"],
            "role_exact": payload.get("training_role") == model["role"],
            "seed_exact": int(payload.get("training_seed", -1))
            == int(model["training_seed"]),
            "config_hash_exact": payload.get("identity", {}).get(
                "scoring_config_sha256"
            )
            == expected_config_hash,
            "training_receipt_passed": payload.get(
                "training_receipt", {}
            ).get("passed")
            is True,
            "score_audit_passed": payload.get("score_audit", {}).get(
                "passed"
            )
            is True,
            "model_state_frozen": payload.get(
                "model_state_sha256_before"
            )
            == payload.get("model_state_sha256_after"),
        }
        _require(
            all(checks.values()),
            f"正式结果审计失败 {slug}: {checks}",
        )
        checkpoint_hash = payload["identity"]["checkpoint_sha256"]
        _require(
            checkpoint_hash not in checkpoint_hashes,
            f"checkpoint 重复：{slug}",
        )
        checkpoint_hashes.add(checkpoint_hash)
        loaded[slug] = payload
        files[slug] = file_sha256(path)
    return loaded, files


def _model_gate(
    config: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any]:
    gate = config["method_gate"]
    trajectory_gate = gate["three_step_trajectory_per_delay"]
    trajectory = result["summary"]["trajectory"]
    delay_checks = {}
    for delay, values in trajectory["by_target_delay"].items():
        eval_seed_values = trajectory[
            "by_target_delay_and_eval_seed"
        ][delay]
        checks = {
            "history_selection_rate": values[
                "exact_history_selection_rate"
            ]
            >= float(
                trajectory_gate[
                    "exact_history_selection_rate_minimum"
                ]
            ),
            "target_selection_rate": values[
                "exact_target_selection_rate"
            ]
            >= float(
                trajectory_gate[
                    "exact_target_selection_rate_minimum"
                ]
            ),
            "strict_history_win_rate": values[
                "matching_history_strict_win_rate"
            ]
            >= float(
                trajectory_gate[
                    "matching_history_strict_win_rate_minimum"
                ]
            ),
            "mean_history_margin_positive": values[
                "mean_history_margin"
            ]
            > 0.0,
            "every_eval_seed_history_margin_positive": all(
                row["mean_history_margin"] > 0.0
                for row in eval_seed_values.values()
            ),
        }
        delay_checks[delay] = {
            "checks": checks,
            "passed": all(checks.values()),
        }

    horizon_gate = gate["autoregressive_horizon_checks"]
    horizon_checks = {}
    for horizon in ("1", "2", "3"):
        values = result["summary"]["by_horizon"][horizon][
            "by_target_delay"
        ]
        if horizon == "1":
            metric = "physical_target_group_selection_rate"
            minimum = float(horizon_gate["h1"]["per_delay_minimum"])
        else:
            metric = "exact_target_selection_rate"
            minimum = float(
                horizon_gate[f"h{horizon}"]["per_delay_minimum"]
            )
        per_delay = {
            delay: row[metric] >= minimum
            for delay, row in values.items()
        }
        horizon_checks[horizon] = {
            "metric": metric,
            "minimum": minimum,
            "per_delay": per_delay,
            "passed": all(per_delay.values()),
        }
    checks = {
        "all_trajectory_delays_pass": all(
            row["passed"] for row in delay_checks.values()
        ),
        "all_autoregressive_horizons_pass": all(
            row["passed"] for row in horizon_checks.values()
        ),
    }
    return {
        "trajectory_by_delay": delay_checks,
        "autoregressive_horizons": horizon_checks,
        "checks": checks,
        "passed_before_paired_control": all(checks.values()),
    }


def _paired_comparisons(
    config: dict[str, Any],
    results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    by_role_seed = {
        (str(model["role"]), int(model["training_seed"])): slug
        for slug, model in _models(config).items()
    }
    comparisons = {}
    for seed in config["method_gate"]["required_training_seeds"]:
        seed = int(seed)
        single_slug = by_role_seed[("single_delay_control", seed)]
        multi_slug = by_role_seed[("multi_delay_target", seed)]
        single = results[single_slug]["summary"]["trajectory"]["overall"]
        multi = results[multi_slug]["summary"]["trajectory"]["overall"]
        metrics = {
            metric: {
                "single": float(single[metric]),
                "multi": float(multi[metric]),
                "delta": float(multi[metric] - single[metric]),
                "passed": float(multi[metric]) > float(single[metric]),
            }
            for metric in RATE_METRICS
        }
        comparisons[str(seed)] = {
            "single_slug": single_slug,
            "multi_slug": multi_slug,
            "metrics": metrics,
            "passed": all(row["passed"] for row in metrics.values()),
        }
    return comparisons


def _group_summary(
    config: dict[str, Any],
    results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for slug, model in _models(config).items():
        grouped[str(model["role"])].append(results[slug])
    summary = {}
    for role, rows in grouped.items():
        summary[role] = {
            "models": len(rows),
            "training_seeds": sorted(
                int(row["training_seed"]) for row in rows
            ),
            "trajectory_overall": {
                metric: _mean_std(
                    [
                        float(
                            row["summary"]["trajectory"]["overall"][
                                metric
                            ]
                        )
                        for row in rows
                    ]
                )
                for metric in (
                    *RATE_METRICS,
                    "physical_target_group_selection_rate",
                    "mean_history_loss_ratio",
                )
            },
            "by_horizon": {
                horizon: {
                    metric: _mean_std(
                        [
                            float(
                                row["summary"]["by_horizon"][horizon][
                                    "overall"
                                ][metric]
                            )
                            for row in rows
                        ]
                    )
                    for metric in (
                        "exact_history_selection_rate",
                        "exact_target_selection_rate",
                        "physical_target_group_selection_rate",
                        "matching_history_strict_win_rate",
                    )
                }
                for horizon in ("1", "2", "3")
            },
            "trajectory_by_track": {
                track: {
                    metric: _mean_std(
                        [
                            float(
                                row["summary"]["trajectory"]["by_track"][
                                    track
                                ][metric]
                            )
                            for row in rows
                        ]
                    )
                    for metric in RATE_METRICS
                }
                for track in (
                    "training_seen",
                    "within_range_unseen",
                    "above_range_unseen",
                )
            },
        }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    _require(
        config.get("benchmark")
        == "tworoom_action_delay_history7_scoring_v1",
        "不是冻结的 History=7 Action Delay 评分配置",
    )
    results, result_files = _load_results(
        config, config_path=config_path
    )
    model_gates = {
        slug: _model_gate(config, result)
        for slug, result in results.items()
    }
    paired = _paired_comparisons(config, results)

    multi_by_seed = {
        int(model["training_seed"]): slug
        for slug, model in _models(config).items()
        if model["role"] == "multi_delay_target"
    }
    seed_decisions = {}
    for seed in config["method_gate"]["required_training_seeds"]:
        seed = int(seed)
        slug = multi_by_seed[seed]
        checks = {
            "multi_model_gate": model_gates[slug][
                "passed_before_paired_control"
            ],
            "paired_multi_outperforms_single": paired[str(seed)][
                "passed"
            ],
        }
        seed_decisions[str(seed)] = {
            "model_slug": slug,
            "checks": checks,
            "passed": all(checks.values()),
        }
    primary_passed = all(
        row["passed"] for row in seed_decisions.values()
    )

    output = resolve_contextworld_path(
        (
            args.output
            if args.output is not None
            else config["artifacts"]["aggregate"]
        ),
        repo_root=ROOT,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "benchmark": config["benchmark"],
        "status": "completed",
        "identity": {
            "config": str(config_path),
            "config_sha256": file_sha256(config_path),
            "aggregation_entrypoint": str(Path(__file__).resolve()),
            "aggregation_entrypoint_sha256": file_sha256(
                Path(__file__).resolve()
            ),
            "result_files": result_files,
        },
        "group_summary": _group_summary(config, results),
        "model_gates": model_gates,
        "paired_multi_vs_single": paired,
        "seed_decisions": seed_decisions,
        "primary_prediction_gate": {
            "passed": primary_passed,
            "required_seed_count": len(seed_decisions),
            "passed_seed_count": sum(
                row["passed"] for row in seed_decisions.values()
            ),
            "ability_retention_required_next": primary_passed,
            "hidden_test_may_run": False,
        },
        "claim": (
            "History=7 multi-delay training passed the frozen Validation "
            "prediction gate."
            if primary_passed
            else "History=7 multi-delay training did not pass the frozen "
            "Validation prediction gate."
        ),
    }
    write_json(output, payload)
    print(
        json.dumps(
            {
                "output": str(output),
                "primary_prediction_gate": payload[
                    "primary_prediction_gate"
                ],
                "group_summary": payload["group_summary"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
