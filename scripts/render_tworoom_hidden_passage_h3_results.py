#!/usr/bin/env python3
"""Render the strictly verified H3 hidden-passage results for readers."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contextworld.evaluation.hidden_passage_validation import (
    PAIRED_BOOTSTRAP_METRICS,
    canonical_sha256,
    file_sha256,
)
from contextworld.paths import resolve_contextworld_path
from contextworld.synthesis.manifest import write_json
from scripts.analyze_tworoom_hidden_passage_h3 import (
    aggregate_validation_results,
)


DEFAULT_CONFIG = (
    ROOT
    / "configs/benchmark/tworoom_hidden_passage_h3_validation_v2.yaml"
)

MODEL_DISPLAY_NAMES = {
    "H3_Original_LEWM": "原始 TwoRoom 数据训练（基线）",
    "H3_Passage_PassableOnly": (
        "原始 H3 续训：只用“门可通过”合成数据"
    ),
    "H3_Passage_BlockedOnly": (
        "原始 H3 续训：只用“门不可通过”合成数据"
    ),
    "H3_Passage_MixedRules": (
        "原始 H3 续训：同时用两种规则合成数据"
    ),
}

TRAINED_FAMILY_IDS = (
    "H3_Passage_PassableOnly",
    "H3_Passage_BlockedOnly",
    "H3_Passage_MixedRules",
)

RULE_DISPLAY_NAMES = {
    "passable": "真实下一帧：门可通过",
    "blocked": "真实下一帧：门不可通过",
}

CI_DISPLAY_NAMES = {
    "passable/same_vs_other_rule_history": (
        "门可通过：对应规则历史相对另一种历史的优势"
    ),
    "passable/same_vs_no_crossing_attempt": (
        "门可通过：对应规则历史相对未尝试穿门历史的优势"
    ),
    "blocked/same_vs_other_rule_history": (
        "门不可通过：对应规则历史相对另一种历史的优势"
    ),
    "blocked/same_vs_no_crossing_attempt": (
        "门不可通过：对应规则历史相对未尝试穿门历史的优势"
    ),
    "passable/matching_history_two_target_margin": (
        "门可通过：对应规则历史的两目标判断余量"
    ),
    "blocked/matching_history_two_target_margin": (
        "门不可通过：对应规则历史的两目标判断余量"
    ),
}

ATTRIBUTION_CHECKS = (
    "mixed_rules_passes_all_three_training_seeds",
    "original_baseline_fails",
    "passable_only_family_does_not_pass_all_three_seeds",
    "blocked_only_family_does_not_pass_all_three_seeds",
)


def _finite_number(value: Any, *, field: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def _rate(value: Any, *, field: str) -> float:
    number = _finite_number(value, field=field)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{field} must be in [0, 1]")
    return number


def _required_identities(
    config: dict[str, Any],
) -> list[tuple[str, int]]:
    required_results = config["comparison"]["required_results"]
    if set(required_results) != set(MODEL_DISPLAY_NAMES):
        raise ValueError(
            "Reader renderer supports only the frozen four-family V2 matrix"
        )
    identities = [
        (str(model_id), int(seed))
        for model_id in MODEL_DISPLAY_NAMES
        for seed in required_results[model_id]
    ]
    if len(identities) != 10 or len(set(identities)) != 10:
        raise ValueError("Reader renderer requires exactly ten unique results")
    for model_id in TRAINED_FAMILY_IDS:
        if len(tuple(required_results[model_id])) != 3:
            raise ValueError(
                f"{model_id} must contain exactly three training seeds"
            )
    return identities


def _strictly_revalidate(
    *,
    stored_aggregate: dict[str, Any],
    results: list[dict[str, Any]],
    result_paths: list[Path],
    config: dict[str, Any],
    config_path: Path,
    expected_catalog_sha256: str,
) -> dict[str, Any]:
    """Run the formal analyzer again and reject any aggregate difference."""

    recomputed = aggregate_validation_results(
        results=results,
        paths=result_paths,
        config=config,
        config_path=config_path,
        expected_catalog_sha256=expected_catalog_sha256,
    )
    if canonical_sha256(recomputed) != canonical_sha256(stored_aggregate):
        raise ValueError(
            "Stored aggregate differs from strict identity revalidation"
        )
    if recomputed.get("status") != "completed":
        raise ValueError("Only a completed aggregate can be rendered")
    if recomputed.get("benchmark") != config["benchmark"]:
        raise ValueError("Aggregate benchmark differs from the V2 config")
    contract = recomputed.get("comparison_contract", {})
    if (
        contract.get(
            "native_latent_mse_cross_checkpoint_comparison_allowed"
        )
        is not False
    ):
        raise ValueError(
            "Aggregate does not forbid cross-checkpoint raw latent MSE "
            "comparison"
        )
    return recomputed


def _checkpoint_result(
    *,
    result: dict[str, Any],
    expected_identity: tuple[str, int],
) -> dict[str, Any]:
    model_id, seed = expected_identity
    observed_identity = (
        str(result.get("model_id")),
        int(result.get("training_seed", -1)),
    )
    if observed_identity != expected_identity:
        raise ValueError(
            f"Result identity mismatch: {observed_identity} != "
            f"{expected_identity}"
        )

    summary = result["summary"]
    if (
        summary["metric_contract"].get(
            "raw_latent_mse_cross_checkpoint_comparison_allowed"
        )
        is not False
    ):
        raise ValueError(
            f"{model_id}/s{seed} permits an unsafe raw-loss comparison"
        )

    rules: dict[str, Any] = {}
    for rule in ("passable", "blocked"):
        overall = summary["by_true_rule"][rule]["overall"]
        effects = overall["paired_advantage"]
        rules[rule] = {
            "display_name_zh": RULE_DISPLAY_NAMES[rule],
            "same_vs_other_history_advantage": _finite_number(
                effects["same_vs_other_rule_history"],
                field=f"{model_id}/s{seed}/{rule}/same-vs-other",
            ),
            "same_vs_no_attempt_advantage": _finite_number(
                effects["same_vs_no_crossing_attempt"],
                field=f"{model_id}/s{seed}/{rule}/same-vs-no-attempt",
            ),
            "matching_history_target_accuracy": _rate(
                overall["same_history_two_target_accuracy"],
                field=f"{model_id}/s{seed}/{rule}/target-accuracy",
            ),
            "strict_win_rate": _rate(
                overall["strict_win_rate"],
                field=f"{model_id}/s{seed}/{rule}/strict-win",
            ),
        }

    bootstrap = summary["paired_static_query_bootstrap"]
    if set(bootstrap["metrics"]) != set(PAIRED_BOOTSTRAP_METRICS):
        raise ValueError(
            f"{model_id}/s{seed} does not contain the frozen six CI metrics"
        )
    lower_threshold = _finite_number(
        summary["decision"]["thresholds"][
            "minimum_bootstrap_lower_bound_exclusive"
        ],
        field=f"{model_id}/s{seed}/CI-threshold",
    )
    intervals = []
    for metric_id in PAIRED_BOOTSTRAP_METRICS:
        interval = bootstrap["metrics"][metric_id]
        lower = _finite_number(
            interval["lower"],
            field=f"{model_id}/s{seed}/{metric_id}/lower",
        )
        upper = _finite_number(
            interval["upper"],
            field=f"{model_id}/s{seed}/{metric_id}/upper",
        )
        mean = _finite_number(
            interval["mean"],
            field=f"{model_id}/s{seed}/{metric_id}/mean",
        )
        if lower > upper or not lower <= mean <= upper:
            raise ValueError(
                f"{model_id}/s{seed}/{metric_id} has an invalid interval"
            )
        intervals.append(
            {
                "metric_id": metric_id,
                "display_name_zh": CI_DISPLAY_NAMES[metric_id],
                "mean": mean,
                "lower": lower,
                "upper": upper,
                "passed": bool(lower > lower_threshold),
            }
        )
    ci_passed_count = sum(row["passed"] for row in intervals)
    all_six_ci_passed = ci_passed_count == len(PAIRED_BOOTSTRAP_METRICS)
    stored_ci_gate = bool(
        summary["decision"]["checks"][
            "paired_static_query_bootstrap_lower_bounds_above_threshold"
        ]
    )
    if stored_ci_gate != all_six_ci_passed:
        raise ValueError(
            f"{model_id}/s{seed} CI decision differs from its six intervals"
        )

    return {
        "model_id": model_id,
        "training_recipe_zh": MODEL_DISPLAY_NAMES[model_id],
        "training_seed": seed,
        "evaluation_scope_zh": (
            "该 checkpoint 在 6 个 Eval seed、每个 seed 50 个 query 上的"
            "汇总；每行不是多个模型之间的绝对 loss 对比"
        ),
        "by_true_rule": rules,
        "confidence_intervals_95": intervals,
        "ci_passed_count": ci_passed_count,
        "ci_required_count": len(PAIRED_BOOTSTRAP_METRICS),
        "all_six_ci_passed": all_six_ci_passed,
        "checkpoint_validation_passed": bool(
            summary["decision"]["passed"]
        ),
        "failed_checkpoint_gates": list(
            summary["decision"]["failed_checks"]
        ),
    }


def _attribution_summary(
    *,
    checkpoints: list[dict[str, Any]],
    verified_aggregate: dict[str, Any],
) -> dict[str, Any]:
    passed = {
        (row["model_id"], int(row["training_seed"])): bool(
            row["checkpoint_validation_passed"]
        )
        for row in checkpoints
    }
    original_passed = passed[("H3_Original_LEWM", 3072)]
    families = []
    all_three_by_family = {}
    for model_id in TRAINED_FAMILY_IDS:
        seed_rows = [
            row for row in checkpoints if row["model_id"] == model_id
        ]
        seed_rows.sort(key=lambda row: int(row["training_seed"]))
        if len(seed_rows) != 3:
            raise ValueError(f"{model_id} does not have three result rows")
        passed_count = sum(
            bool(row["checkpoint_validation_passed"]) for row in seed_rows
        )
        all_three = passed_count == 3
        all_three_by_family[model_id] = all_three
        if model_id == "H3_Passage_MixedRules":
            role = "目标训练配方：三个训练种子都必须通过"
            attribution_condition = all_three
        else:
            role = "单规则对照：不能三个训练种子都通过"
            attribution_condition = not all_three
        families.append(
            {
                "model_id": model_id,
                "training_recipe_zh": MODEL_DISPLAY_NAMES[model_id],
                "seed_results": [
                    {
                        "training_seed": int(row["training_seed"]),
                        "checkpoint_validation_passed": bool(
                            row["checkpoint_validation_passed"]
                        ),
                    }
                    for row in seed_rows
                ],
                "passed_seed_count": passed_count,
                "required_seed_count": 3,
                "all_three_seeds_passed": all_three,
                "attribution_role_zh": role,
                "attribution_condition_satisfied": attribution_condition,
            }
        )

    derived_checks = {
        "mixed_rules_passes_all_three_training_seeds": all_three_by_family[
            "H3_Passage_MixedRules"
        ],
        "original_baseline_fails": not original_passed,
        "passable_only_family_does_not_pass_all_three_seeds": (
            not all_three_by_family["H3_Passage_PassableOnly"]
        ),
        "blocked_only_family_does_not_pass_all_three_seeds": (
            not all_three_by_family["H3_Passage_BlockedOnly"]
        ),
    }
    stored = verified_aggregate["attribution"]
    stored_checks = stored["checks"]
    if set(stored_checks) != set(ATTRIBUTION_CHECKS):
        raise ValueError("Aggregate attribution check set changed")
    if any(
        bool(stored_checks[name]) != bool(derived_checks[name])
        for name in ATTRIBUTION_CHECKS
    ):
        raise ValueError(
            "Aggregate attribution checks differ from checkpoint decisions"
        )
    attribution_passed = bool(all(derived_checks.values()))
    if bool(stored["passed"]) != attribution_passed:
        raise ValueError("Aggregate attribution decision is inconsistent")

    if attribution_passed:
        conclusion = (
            "通过预注册归因条件：双规则合成训练在三个训练种子上"
            "全部通过；原始基线未通过；两个单规则训练对照都没有"
            "三个种子"
            "全部通过。结果支持“双规则合成训练带来了 History=3 下"
            "的局部门通行规则适应”。"
        )
    else:
        failed = [
            name for name, value in derived_checks.items() if not value
        ]
        conclusion = (
            "未通过预注册归因条件，因此当前结果不能归因为"
            "“双规则"
            "合成训练带来了 History=3 下的局部门通行规则适应”。"
            f"未满足的条件：{', '.join(failed)}。"
        )
    return {
        "passed": attribution_passed,
        "original_baseline": {
            "training_recipe_zh": MODEL_DISPLAY_NAMES["H3_Original_LEWM"],
            "training_seed": 3072,
            "checkpoint_validation_passed": original_passed,
            "required_for_attribution_zh": "原始基线必须不通过",
            "condition_satisfied": not original_passed,
        },
        "three_training_families": families,
        "checks": derived_checks,
        "conclusion_zh": conclusion,
    }


def build_reader_outputs(
    *,
    stored_aggregate: dict[str, Any],
    aggregate_path: Path,
    results: list[dict[str, Any]],
    result_paths: list[Path],
    config: dict[str, Any],
    config_path: Path,
    expected_catalog_sha256: str,
) -> tuple[dict[str, Any], str]:
    """Strictly verify all inputs, then build JSON and Chinese Markdown."""

    identities = _required_identities(config)
    if len(results) != 10 or len(result_paths) != 10:
        raise ValueError("Exactly ten result files are required")
    by_identity: dict[tuple[str, int], dict[str, Any]] = {}
    for result in results:
        identity = (
            str(result.get("model_id")),
            int(result.get("training_seed", -1)),
        )
        if identity in by_identity:
            raise ValueError(f"Duplicate result identity: {identity}")
        by_identity[identity] = result
    if set(by_identity) != set(identities):
        raise ValueError("The ten result identities differ from the V2 matrix")

    verified_aggregate = _strictly_revalidate(
        stored_aggregate=stored_aggregate,
        results=results,
        result_paths=result_paths,
        config=config,
        config_path=config_path,
        expected_catalog_sha256=expected_catalog_sha256,
    )
    checkpoints = [
        _checkpoint_result(
            result=by_identity[identity],
            expected_identity=identity,
        )
        for identity in identities
    ]
    attribution = _attribution_summary(
        checkpoints=checkpoints,
        verified_aggregate=verified_aggregate,
    )
    payload = {
        "schema_version": 1,
        "benchmark": config["benchmark"],
        "status": "completed_after_strict_identity_revalidation",
        "language": "zh-CN",
        "source": {
            "aggregate": str(aggregate_path),
            "aggregate_file_sha256": file_sha256(aggregate_path),
            "aggregate_content_sha256": canonical_sha256(
                verified_aggregate
            ),
            "config": str(config_path),
            "config_sha256": file_sha256(config_path),
            "catalog_sha256": expected_catalog_sha256,
            "result_count": len(results),
            "strict_identity_revalidation_passed": True,
        },
        "metric_contract": {
            "raw_latent_mse_cross_checkpoint_comparison_allowed": False,
            "absolute_native_latent_mse_included": False,
            "paired_advantage_definition_zh": (
                "同一 checkpoint、同一 query 内，用另一种历史或"
                "未尝试穿门历史的真实下一帧 loss，减去对应规则历史"
                "的 loss；"
                "正数表示对应规则历史更好。"
            ),
            "comparison_warning_zh": (
                "优势值只用于同一 checkpoint 内的成对判断。不得按"
                "数值"
                "大小比较不同 checkpoint 的绝对 latent loss。"
            ),
        },
        "checkpoint_results": checkpoints,
        "attribution": attribution,
    }
    return payload, render_markdown(payload)


def _fmt_effect(value: float) -> str:
    return f"{float(value):.6g}"


def _fmt_rate(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def _fmt_status(value: bool) -> str:
    return "通过" if value else "未通过"


def _fmt_ci(interval: dict[str, Any]) -> str:
    marker = "✓" if interval["passed"] else "✗"
    return (
        f"{_fmt_effect(interval['mean'])} "
        f"[{_fmt_effect(interval['lower'])}, "
        f"{_fmt_effect(interval['upper'])}] {marker}"
    )


def render_markdown(payload: dict[str, Any]) -> str:
    """Render a compact public-facing Chinese report."""

    checkpoints = payload["checkpoint_results"]
    lines = [
        "# History=3 局部门通行规则评测结果",
        "",
        "本报告只展示通过 Validation v2 严格身份重验的 10 个结果。"
        "每个训练 checkpoint 都在同一组 50×6 个 query 上评测。",
        "",
        "“相对另一种历史的优势”是：另一种历史的真实下一帧 "
        "loss − "
        "对应规则历史的 loss；“相对未尝试历史的优势”同理。"
        "正数表示对应规则历史更好。表中的优势只解释同一 "
        "checkpoint "
        "内部的成对差异，不能用来比较不同 checkpoint 的绝对 latent "
        "loss 大小。",
        "",
    ]
    for rule in ("passable", "blocked"):
        lines.extend(
            [
                f"## {RULE_DISPLAY_NAMES[rule]}",
                "",
                "| 训练数据 | 训练种子 | 相对另一种历史的优势 | "
                "相对未尝试历史的优势 | 目标判对率 | 严格胜率 |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in checkpoints:
            metric = row["by_true_rule"][rule]
            lines.append(
                f"| {row['training_recipe_zh']} | "
                f"{row['training_seed']} | "
                f"{_fmt_effect(metric['same_vs_other_history_advantage'])} | "
                f"{_fmt_effect(metric['same_vs_no_attempt_advantage'])} | "
                f"{_fmt_rate(metric['matching_history_target_accuracy'])} | "
                f"{_fmt_rate(metric['strict_win_rate'])} |"
            )
        lines.append("")

    lines.extend(
        [
            "## 六项 95% 置信区间",
            "",
            "每个单元格为“均值 [95% 下界, 95% 上界]”。只有下界"
            "严格大于 "
            "0 才记为 ✓；六项全部为 ✓ 才算这一项门槛通过。",
            "",
            "| 训练数据 | 种子 | 可通过/另一历史 | 可通过/未尝试 | "
            "不可通过/另一历史 | 不可通过/未尝试 | "
            "可通过/目标余量 | 不可通过/目标余量 | 六项最终结果 | "
            "checkpoint 最终结果 |",
            "|---|---:|---|---|---|---|---|---|---|---|",
        ]
    )
    for row in checkpoints:
        intervals = {
            interval["metric_id"]: interval
            for interval in row["confidence_intervals_95"]
        }
        ci_cells = [
            _fmt_ci(intervals[metric_id])
            for metric_id in PAIRED_BOOTSTRAP_METRICS
        ]
        lines.append(
            f"| {row['training_recipe_zh']} | {row['training_seed']} | "
            + " | ".join(ci_cells)
            + f" | {row['ci_passed_count']}/6（"
            f"{_fmt_status(row['all_six_ci_passed'])}） | "
            f"{_fmt_status(row['checkpoint_validation_passed'])} |"
        )
    lines.extend(
        [
            "",
            "## 三种训练数据配方的 3/3 归因检查",
            "",
            "| 训练数据 | 通过的训练种子 | 通过数 | 是否 3/3 | "
            "归因条件是否满足 |",
            "|---|---|---:|---|---|",
        ]
    )
    for family in payload["attribution"]["three_training_families"]:
        passed_seeds = [
            str(row["training_seed"])
            for row in family["seed_results"]
            if row["checkpoint_validation_passed"]
        ]
        lines.append(
            f"| {family['training_recipe_zh']} | "
            f"{', '.join(passed_seeds) if passed_seeds else '无'} | "
            f"{family['passed_seed_count']}/3 | "
            f"{_fmt_status(family['all_three_seeds_passed'])} | "
            f"{_fmt_status(family['attribution_condition_satisfied'])} |"
        )
    baseline = payload["attribution"]["original_baseline"]
    lines.extend(
        [
            "",
            f"原始基线 checkpoint："
            f"{_fmt_status(baseline['checkpoint_validation_passed'])}；"
            f"“原始基线必须不通过”这一归因条件："
            f"{_fmt_status(baseline['condition_satisfied'])}。",
            "",
            "## 阶段结论",
            "",
            payload["attribution"]["conclusion_zh"],
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Strictly revalidate and render the ten-result hidden-passage "
            "History-3 comparison"
        )
    )
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--results", type=Path, nargs="+", required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    args = parser.parse_args()

    aggregate_path = resolve_contextworld_path(
        args.aggregate,
        repo_root=ROOT,
    )
    config_path = args.config.resolve()
    result_paths = [
        resolve_contextworld_path(path, repo_root=ROOT)
        for path in args.results
    ]
    json_output = resolve_contextworld_path(
        args.json_output,
        repo_root=ROOT,
    )
    markdown_output = resolve_contextworld_path(
        args.markdown_output,
        repo_root=ROOT,
    )
    if json_output == markdown_output:
        raise ValueError("JSON and Markdown outputs must be different files")
    existing = [
        path for path in (json_output, markdown_output) if path.exists()
    ]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite reader output: "
            + ", ".join(map(str, existing))
        )

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    stored_aggregate = json.loads(
        aggregate_path.read_text(encoding="utf-8")
    )
    results = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in result_paths
    ]
    catalog_path = resolve_contextworld_path(
        config["artifacts"]["catalog"],
        repo_root=ROOT,
    )
    if not catalog_path.is_file():
        raise FileNotFoundError(catalog_path)
    payload, markdown = build_reader_outputs(
        stored_aggregate=stored_aggregate,
        aggregate_path=aggregate_path,
        results=results,
        result_paths=result_paths,
        config=config,
        config_path=config_path,
        expected_catalog_sha256=file_sha256(catalog_path),
    )

    write_json(json_output, payload)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.write_text(markdown, encoding="utf-8")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "attribution_passed": payload["attribution"]["passed"],
                "json_output": str(json_output),
                "markdown_output": str(markdown_output),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
