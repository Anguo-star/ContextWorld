#!/usr/bin/env python3
"""Aggregate the preregistered action-delay History-3 comparison."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contextworld.evaluation.action_delay import DELAY_VALUES
from contextworld.evaluation.action_delay_validation import (
    EVAL_SEEDS,
    INTERPOLATION_DELAYS,
    QUERY_COUNT,
    SEEN_DELAYS,
    file_sha256,
)
from contextworld.paths import artifact_path
from contextworld.synthesis.manifest import write_json


MODEL_NAMES = {
    "H3_Original_LEWM": "原始 TwoRoom 数据训练模型",
    "H3_ActionDelay_SingleControl": "原始数据 + 单一延迟合成数据",
    "H3_ActionDelay_Multi": "原始数据 + 多延迟合成数据",
}
FORMAL_SEEDS = (3072, 4096, 5120)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(payload.get("status") == "completed", f"结果未完成：{path}")
    _require(
        payload.get("benchmark")
        == "tworoom_action_delay_history3_validation_v1",
        f"结果 benchmark 不一致：{path}",
    )
    _require(
        payload["score_audit"]["queries"] == QUERY_COUNT
        and payload["score_audit"]["model_predictions"] == 1500
        and payload["score_audit"]["loss_records"] == 7500,
        f"结果计数不完整：{path}",
    )
    return payload


def _mean_std(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
    }


def _metric_projection(summary: dict[str, Any]) -> dict[str, float]:
    return {
        key: float(summary[key])
        for key in (
            "mean_matching_history_loss",
            "mean_other_history_loss",
            "mean_history_margin",
            "mean_history_loss_ratio",
            "matching_history_strict_win_rate",
            "history_selection_accuracy",
            "target_selection_accuracy",
        )
    }


def _aggregate_model(runs: list[dict[str, Any]]) -> dict[str, Any]:
    by_delay: dict[str, Any] = {}
    for delay in DELAY_VALUES:
        summaries = [
            run["summary"]["by_target_delay"][str(delay)] for run in runs
        ]
        by_delay[str(delay)] = {
            metric: _mean_std(
                [float(summary[metric]) for summary in summaries]
            )
            for metric in _metric_projection(summaries[0])
        }
    by_track = {}
    for track in ("training_seen", "interpolation"):
        summaries = [run["summary"]["by_track"][track] for run in runs]
        by_track[track] = {
            metric: _mean_std(
                [float(summary[metric]) for summary in summaries]
            )
            for metric in _metric_projection(summaries[0])
        }
    overall = {
        metric: _mean_std(
            [float(run["summary"]["overall"][metric]) for run in runs]
        )
        for metric in _metric_projection(runs[0]["summary"]["overall"])
    }
    return {
        "training_seeds": [int(run["training_seed"]) for run in runs],
        "overall": overall,
        "by_track": by_track,
        "by_target_delay": by_delay,
    }


def _multi_seed_gate(run: dict[str, Any]) -> dict[str, Any]:
    delay_results = {}
    for delay in DELAY_VALUES:
        summary = run["summary"]["by_target_delay"][str(delay)]
        seed_summaries = run["summary"][
            "by_target_delay_and_eval_seed"
        ][str(delay)]
        checks = {
            "mean_history_margin_positive": (
                float(summary["mean_history_margin"]) > 0.0
            ),
            "all_six_eval_seed_margins_positive": all(
                float(seed_summaries[str(seed)]["mean_history_margin"])
                > 0.0
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
        delay_results[str(delay)] = {
            "passed": all(checks.values()),
            "checks": checks,
            "metrics": _metric_projection(summary),
        }
    return {
        "passed": all(item["passed"] for item in delay_results.values()),
        "by_target_delay": delay_results,
    }


def _paired_gate(
    multi: dict[str, Any],
    single: dict[str, Any],
) -> dict[str, Any]:
    multi_overall = multi["summary"]["overall"]
    single_overall = single["summary"]["overall"]
    deltas = {
        "mean_history_margin": float(
            multi_overall["mean_history_margin"]
            - single_overall["mean_history_margin"]
        ),
        "history_selection_accuracy": float(
            multi_overall["history_selection_accuracy"]
            - single_overall["history_selection_accuracy"]
        ),
        "target_selection_accuracy": float(
            multi_overall["target_selection_accuracy"]
            - single_overall["target_selection_accuracy"]
        ),
        "strict_win_rate": float(
            multi_overall["matching_history_strict_win_rate"]
            - single_overall["matching_history_strict_win_rate"]
        ),
    }
    checks = {
        "multi_minus_single_history_margin_positive": (
            deltas["mean_history_margin"] > 0.0
        ),
        "multi_minus_single_history_selection_accuracy_positive": (
            deltas["history_selection_accuracy"] > 0.0
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "deltas": deltas,
    }


def parse_args() -> argparse.Namespace:
    root = artifact_path(
        "evaluation/history3/action_delay_validation_v1/results",
        repo_root=ROOT,
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=root)
    parser.add_argument(
        "--output",
        type=Path,
        default=root.parent / "comparison_summary.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.results_root.expanduser().resolve()
    expected = {
        ("H3_Original_LEWM", 3072): root / "original_s3072.json",
        **{
            ("H3_ActionDelay_SingleControl", seed): (
                root / f"single_s{seed}.json"
            )
            for seed in FORMAL_SEEDS
        },
        **{
            ("H3_ActionDelay_Multi", seed): (
                root / f"multi_s{seed}.json"
            )
            for seed in FORMAL_SEEDS
        },
    }
    runs = {}
    for identity, path in expected.items():
        _require(path.is_file(), f"缺少正式结果：{path}")
        payload = _load(path)
        _require(
            (payload["model_id"], int(payload["training_seed"]))
            == identity,
            f"结果身份不匹配：{path}",
        )
        runs[identity] = payload

    original = [runs[("H3_Original_LEWM", 3072)]]
    single = [
        runs[("H3_ActionDelay_SingleControl", seed)]
        for seed in FORMAL_SEEDS
    ]
    multi = [
        runs[("H3_ActionDelay_Multi", seed)]
        for seed in FORMAL_SEEDS
    ]
    seed_gates = {
        str(seed): _multi_seed_gate(
            runs[("H3_ActionDelay_Multi", seed)]
        )
        for seed in FORMAL_SEEDS
    }
    paired_gates = {
        str(seed): _paired_gate(
            runs[("H3_ActionDelay_Multi", seed)],
            runs[("H3_ActionDelay_SingleControl", seed)],
        )
        for seed in FORMAL_SEEDS
    }
    claim_passed = all(
        gate["passed"] for gate in seed_gates.values()
    ) and all(gate["passed"] for gate in paired_gates.values())
    result = {
        "schema_version": 1,
        "benchmark": "tworoom_action_delay_history3_validation_v1",
        "status": "completed",
        "result_files": {
            f"{model_id}:{seed}": {
                "path": str(path),
                "sha256": file_sha256(path),
            }
            for (model_id, seed), path in expected.items()
        },
        "model_display_names": MODEL_NAMES,
        "delay_tracks": {
            "training_seen": list(SEEN_DELAYS),
            "interpolation": list(INTERPOLATION_DELAYS),
        },
        "models": {
            "original": _aggregate_model(original),
            "single_delay_control": _aggregate_model(single),
            "multi_delay_target": _aggregate_model(multi),
        },
        "preregistered_gates": {
            "multi_delay_each_training_seed": seed_gates,
            "paired_training_attribution": paired_gates,
        },
        "decision": {
            "passed": claim_passed,
            "claim": (
                "多延迟训练带来了动作响应延迟 ICL"
                if claim_passed
                else "未达到声明动作响应延迟 ICL 的预注册门槛"
            ),
        },
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    print(
        json.dumps(
            {
                "output": str(output),
                "decision": result["decision"],
                "models": {
                    key: value["overall"]
                    for key, value in result["models"].items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
