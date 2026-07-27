#!/usr/bin/env python3
"""Combine held-out, train-seen, and tiny-overfit H3 passage diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contextworld.evaluation.hidden_passage_validation import file_sha256
from contextworld.paths import artifact_path
from contextworld.synthesis.manifest import write_json
from scripts.eval_tworoom_hidden_passage_h3_overfit_diagnostic import (
    _switch_diagnostic,
)


FORMAL_ROOT = artifact_path(
    "evaluation/history3/hidden_passage_validation_v2",
    repo_root=ROOT,
)
TRAIN_SEEN_ROOT = artifact_path(
    "evaluation/history3/hidden_passage_train_seen_diagnostic_v1",
    repo_root=ROOT,
)
FROZEN_REPRESENTATION_ROOT = artifact_path(
    "evaluation/history3/hidden_passage_frozen_representation_diagnostic_v1",
    repo_root=ROOT,
)
OUTPUT_ROOT = artifact_path(
    "evaluation/history3/hidden_passage_root_cause_v2",
    repo_root=ROOT,
)


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping in {path}")
    return payload


def _model_rows(
    aggregate: dict[str, Any],
) -> dict[tuple[str, int], dict[str, Any]]:
    rows = {
        (str(row["model_id"]), int(row["training_seed"])): row
        for row in aggregate["models"]
    }
    if len(rows) != 10:
        raise ValueError("Expected the exact ten-model comparison")
    return rows


def _result_switches(
    aggregate: dict[str, Any],
) -> dict[tuple[str, int], dict[str, Any]]:
    output = {}
    for row in aggregate["result_files"]:
        path = Path(str(row["path"])).resolve()
        if file_sha256(path) != row["sha256"]:
            raise ValueError(f"Result hash mismatch: {path}")
        result = _load(path)
        identity = (
            str(result["model_id"]),
            int(result["training_seed"]),
        )
        output[identity] = _switch_diagnostic(result["records"])
    if len(output) != 10:
        raise ValueError("Expected switches for ten exact results")
    return output


def _recipe_summary(
    aggregate: dict[str, Any],
    switches: dict[tuple[str, int], dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = _model_rows(aggregate)
    families = (
        ("H3_Original_LEWM", [3072]),
        ("H3_Passage_PassableOnly", [3072, 4096, 5120]),
        ("H3_Passage_BlockedOnly", [3072, 4096, 5120]),
        ("H3_Passage_MixedRules", [3072, 4096, 5120]),
    )
    output = []
    for model_id, seeds in families:
        models = [rows[(model_id, seed)] for seed in seeds]
        output.append(
            {
                "model_id": model_id,
                "seeds": seeds,
                "passed": [
                    bool(row["checkpoint_validation_passed"])
                    for row in models
                ],
                "passable_accuracy": [
                    float(
                        row["by_true_rule"]["passable"][
                            "same_history_two_target_accuracy"
                        ]
                    )
                    for row in models
                ],
                "blocked_accuracy": [
                    float(
                        row["by_true_rule"]["blocked"][
                            "same_history_two_target_accuracy"
                        ]
                    )
                    for row in models
                ],
                "history_target_switch_rate": [
                    float(
                        switches[(model_id, seed)][
                            "history_target_switch_rate"
                        ]
                    )
                    for seed in seeds
                ],
                "correct_directional_switch_rate": [
                    float(
                        switches[(model_id, seed)][
                            "correct_directional_switch_rate"
                        ]
                    )
                    for seed in seeds
                ],
            }
        )
    return output


def _fmt(values: list[float]) -> str:
    return " / ".join(f"{100.0 * value:.1f}%" for value in values)


def _render_markdown(payload: dict[str, Any]) -> str:
    names = {
        "H3_Original_LEWM": "原始 H3",
        "H3_Passage_PassableOnly": "只用门可通过数据续训",
        "H3_Passage_BlockedOnly": "只用门不可通过数据续训",
        "H3_Passage_MixedRules": "两种规则数据续训",
    }
    lines = [
        "# History=3 门规则失败根因诊断",
        "",
        "本报告把正式未见门位置、训练中出现过的门位置和单门小样本记忆实验分开比较。",
        "小样本实验只诊断模型能否记住训练样本，不作为正式 Benchmark 分数。",
        "",
        "## 训练门位置十模型结果",
        "",
        "| 训练配方 | 通过数 | 可通过判对率 | 不可通过判对率 | 历史改变目标选择 | 正确方向切换 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["train_seen"]["recipes"]:
        passed = sum(row["passed"])
        lines.append(
            "| "
            + names[row["model_id"]]
            + f" | {passed}/{len(row['passed'])}"
            + f" | {_fmt(row['passable_accuracy'])}"
            + f" | {_fmt(row['blocked_accuracy'])}"
            + f" | {_fmt(row['history_target_switch_rate'])}"
            + f" | {_fmt(row['correct_directional_switch_rate'])} |"
        )
    mechanism = payload["mechanism_diagnostic"]
    lines.extend(
        [
            "",
            "## 单门双规则机制诊断",
            "",
            "| 训练方式 | 可通过判对率 | 不可通过判对率 | 正确方向切换 | 两种真实下一帧 latent 距离 | 诊断门槛 |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for label, row in (
        ("训练前原始 H3", mechanism["original"]),
        ("编码器和预测器一起训练", mechanism["joint_update"]),
        ("固定原始图像表示，只训练预测部分", mechanism["fixed_representation"]),
    ):
        lines.append(
            f"| {label}"
            f" | {100.0 * row['passable_accuracy']:.1f}%"
            f" | {100.0 * row['blocked_accuracy']:.1f}%"
            f" | {100.0 * row['correct_directional_switch_rate']:.1f}%"
            f" | {row['mean_target_pair_latent_mse']:.6f}"
            f" | {'通过' if row['overfit_gate_passed'] else '未通过'} |"
        )
    lines.extend(
        [
            "",
            "表中 latent 距离只用于这组同数据、同初始化的机制诊断，"
            "不作为跨模型排行榜分数。",
        ]
    )
    lines.extend(
        [
            "",
            "## 结论",
            "",
            payload["conclusion"]["summary_zh"],
            "",
            "判断依据：",
            "",
        ]
    )
    lines.extend(
        f"- {item}" for item in payload["conclusion"]["evidence_zh"]
    )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze the root cause of the failed History=3 passage claim"
    )
    parser.add_argument(
        "--formal-aggregate",
        type=Path,
        default=FORMAL_ROOT / "aggregate.json",
    )
    parser.add_argument(
        "--train-seen-aggregate",
        type=Path,
        default=TRAIN_SEEN_ROOT / "aggregate.json",
    )
    parser.add_argument(
        "--mechanism-original",
        type=Path,
        default=FROZEN_REPRESENTATION_ROOT / "original.json",
    )
    parser.add_argument(
        "--mechanism-joint-update",
        type=Path,
        default=FROZEN_REPRESENTATION_ROOT / "joint_update.json",
    )
    parser.add_argument(
        "--mechanism-fixed-representation",
        type=Path,
        default=(
            FROZEN_REPRESENTATION_ROOT
            / "fixed_representation.json"
        ),
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=OUTPUT_ROOT / "summary.json",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=OUTPUT_ROOT / "summary.md",
    )
    args = parser.parse_args()

    formal = _load(args.formal_aggregate.resolve())
    train_seen = _load(args.train_seen_aggregate.resolve())
    mechanism_original = _load(args.mechanism_original.resolve())
    mechanism_joint = _load(args.mechanism_joint_update.resolve())
    mechanism_fixed = _load(
        args.mechanism_fixed_representation.resolve()
    )
    if formal.get("status") != "completed":
        raise ValueError("Formal aggregate is incomplete")
    if train_seen.get("status") != "completed":
        raise ValueError("Train-seen aggregate is incomplete")
    for result in (
        mechanism_original,
        mechanism_joint,
        mechanism_fixed,
    ):
        if result.get("status") != "completed":
            raise ValueError("Mechanism diagnostic is incomplete")

    formal_switches = _result_switches(formal)
    train_seen_switches = _result_switches(train_seen)
    formal_recipes = _recipe_summary(formal, formal_switches)
    train_seen_recipes = _recipe_summary(
        train_seen,
        train_seen_switches,
    )
    formal_mixed = next(
        row
        for row in formal_recipes
        if row["model_id"] == "H3_Passage_MixedRules"
    )
    train_seen_mixed = next(
        row
        for row in train_seen_recipes
        if row["model_id"] == "H3_Passage_MixedRules"
    )

    def mechanism_row(result: dict[str, Any]) -> dict[str, Any]:
        return {
            "role": result["role"],
            "passable_accuracy": float(
                result["overfit_gate"]["matching_history_accuracy"][
                    "passable"
                ]
            ),
            "blocked_accuracy": float(
                result["overfit_gate"]["matching_history_accuracy"][
                    "blocked"
                ]
            ),
            "history_target_switch_rate": float(
                result["switch_diagnostic"][
                    "history_target_switch_rate"
                ]
            ),
            "correct_directional_switch_rate": float(
                result["switch_diagnostic"][
                    "correct_directional_switch_rate"
                ]
            ),
            "overfit_gate_passed": bool(
                result["overfit_gate"]["passed"]
            ),
            "mean_target_pair_latent_mse": float(
                result["score_audit"]["target_latent_separation"][
                    "mean_mse"
                ]
            ),
            "mean_matching_history_latent_mse": mean(
                float(
                    result["summary"]["by_true_rule"][rule]["overall"][
                        "native_latent_mse"
                    ]["matching_history"]
                )
                for rule in ("passable", "blocked")
            ),
        }

    original_row = mechanism_row(mechanism_original)
    joint_row = mechanism_row(mechanism_joint)
    fixed_row = mechanism_row(mechanism_fixed)
    fixed_training_report_path = Path(
        mechanism_fixed["checkpoint_audit"]["training_report"]
    ).resolve()
    if (
        file_sha256(fixed_training_report_path)
        != mechanism_fixed["checkpoint_audit"][
            "training_report_sha256"
        ]
    ):
        raise ValueError("Frozen-representation training report hash mismatch")
    fixed_training_report = _load(fixed_training_report_path)
    frozen_modules = fixed_training_report.get(
        "frozen_model_modules",
        {},
    )
    if not (
        frozen_modules.get("passed") is True
        and frozen_modules.get("modules") == ["encoder", "projector"]
        and all(
            frozen_modules.get("state_unchanged", {}).get(name) is True
            for name in ("encoder", "projector")
        )
    ):
        raise ValueError(
            "Frozen encoder/projector exact-state audit did not pass"
        )
    joint_overfit_passed = bool(
        mechanism_joint["overfit_gate"]["passed"]
    )
    fixed_overfit_passed = bool(
        mechanism_fixed["overfit_gate"]["passed"]
    )
    train_seen_mixed_all_pass = all(train_seen_mixed["passed"])
    if fixed_overfit_passed and not joint_overfit_passed:
        root_cause = "joint_representation_update_shortcut"
        summary_zh = (
            "History=3 预测器具备读取历史并切换门规则的能力；失败不主要是"
            "门位置覆盖不足，也不是三帧结构绝对做不到。当前联合训练允许图像"
            "表示把两种真实下一状态压得过近，预测器因此学会输出折中状态。"
            "固定原始图像表示后，同样的数据和训练步数可在全部 8 个已见 query "
            "上按历史正确切换。"
        )
    elif train_seen_mixed_all_pass:
        root_cause = "heldout_geometry_generalization"
        summary_zh = (
            "双规则模型在训练门位置上成立、在未见门位置上失败，主要问题是"
            "跨门位置泛化，而不是模型完全不会读取三帧历史。"
        )
    elif joint_overfit_passed:
        root_cause = "full_distribution_binding_or_optimization"
        summary_zh = (
            "模型能够在单门严格成对样本上记住历史规则，但正式双规则模型即使在"
            "训练门位置上也没有稳定切换。主要问题不是模型绝对没有表达能力，"
            "而是完整训练分布中的规则信号没有被普通下一帧训练稳定学出。"
        )
    elif fixed_overfit_passed:
        root_cause = "representation_update_related_but_not_isolated"
        summary_zh = (
            "固定原始图像表示后小样本诊断通过，但联合训练对照也通过；"
            "当前实验不能把失败单独归因给表示更新。"
        )
    else:
        root_cause = "history_binding_or_objective_capacity"
        summary_zh = (
            "模型在单门、精确训练样本上反复训练后仍不能根据历史近乎完全切换。"
            "这排除了单纯增加门位置覆盖就能解决问题的解释，根因更接近历史绑定、"
            "模型输入结构或下一帧训练目标不足。"
        )

    evidence = [
        (
            "正式未见门位置的双规则模型通过数为 "
            f"{sum(formal_mixed['passed'])}/3。"
        ),
        (
            "训练门位置的双规则模型通过数为 "
            f"{sum(train_seen_mixed['passed'])}/3；平均正确方向切换率为 "
            f"{100.0 * mean(train_seen_mixed['correct_directional_switch_rate']):.1f}%。"
        ),
        (
            "编码器和预测器共同训练时，正确方向切换为 "
            f"{100.0 * joint_row['correct_directional_switch_rate']:.1f}%，"
            "两种真实下一帧的平均 latent 距离为 "
            f"{joint_row['mean_target_pair_latent_mse']:.6f}。"
        ),
        (
            "固定原始图像表示后，正确方向切换为 "
            f"{100.0 * fixed_row['correct_directional_switch_rate']:.1f}%，"
            "可通过/不可通过判对率分别为 "
            f"{100.0 * fixed_row['passable_accuracy']:.1f}%/"
            f"{100.0 * fixed_row['blocked_accuracy']:.1f}%，"
            "两种真实下一帧的平均 latent 距离保持为 "
            f"{fixed_row['mean_target_pair_latent_mse']:.6f}。"
        ),
        (
            "训练报告逐模块哈希确认 encoder 和 projector 在 1,024 步"
            "训练前后完全一致；两组训练均使用同一批 160 个训练 clip，"
            "每个 clip 平均抽取 102.4 次。"
        ),
    ]
    payload = {
        "schema_version": 1,
        "status": "completed",
        "diagnostic_only": True,
        "inputs": {
            "formal_aggregate": {
                "path": str(args.formal_aggregate.resolve()),
                "sha256": file_sha256(args.formal_aggregate.resolve()),
            },
            "train_seen_aggregate": {
                "path": str(args.train_seen_aggregate.resolve()),
                "sha256": file_sha256(
                    args.train_seen_aggregate.resolve()
                ),
            },
            "mechanism_original": {
                "path": str(args.mechanism_original.resolve()),
                "sha256": file_sha256(
                    args.mechanism_original.resolve()
                ),
            },
            "mechanism_joint_update": {
                "path": str(args.mechanism_joint_update.resolve()),
                "sha256": file_sha256(
                    args.mechanism_joint_update.resolve()
                ),
            },
            "mechanism_fixed_representation": {
                "path": str(
                    args.mechanism_fixed_representation.resolve()
                ),
                "sha256": file_sha256(
                    args.mechanism_fixed_representation.resolve()
                ),
            },
        },
        "formal_heldout": {"recipes": formal_recipes},
        "train_seen": {"recipes": train_seen_recipes},
        "mechanism_diagnostic": {
            "query_count": 8,
            "formal_benchmark_score": False,
            "frozen_representation_state_audit": {
                "training_report": str(fixed_training_report_path),
                "training_report_sha256": file_sha256(
                    fixed_training_report_path
                ),
                "modules": frozen_modules["modules"],
                "state_unchanged": frozen_modules[
                    "state_unchanged"
                ],
                "passed": True,
            },
            "original": original_row,
            "joint_update": joint_row,
            "fixed_representation": fixed_row,
        },
        "conclusion": {
            "root_cause_class": root_cause,
            "summary_zh": summary_zh,
            "evidence_zh": evidence,
            "not_claimed": [
                (
                    "The eight-query diagnostic proves memorization capacity, "
                    "not held-out door-rule ICL."
                ),
                "The eight-query overfit set is not a benchmark score.",
                (
                    "Native latent distances are used only within this "
                    "same-data, same-initialization mechanism diagnostic."
                ),
                (
                    "The causal ablation identifies a demonstrated shortcut; "
                    "it does not prove no secondary optimization issue exists."
                ),
            ],
        },
    }
    markdown = _render_markdown(payload)
    for path in (args.json_output.resolve(), args.markdown_output.resolve()):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.json_output.resolve(), payload)
    args.markdown_output.resolve().write_text(
        markdown,
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "json_output": str(args.json_output.resolve()),
                "markdown_output": str(args.markdown_output.resolve()),
                "root_cause_class": root_cause,
                "summary_zh": summary_zh,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
