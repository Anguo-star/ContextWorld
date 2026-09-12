"""Render reference tables from v3 and the declared original-DINO supplement."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections.abc import Callable
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TASKS = {
    "速度": "speed", "推手移动幅度": "action_strength", "机械臂质量": "robot_arm_mass",
    "动作延迟": "action_delay", "接触摩擦": "contact_friction", "运动阻尼": "motion_damping",
    "Cube 夹爪携带规则": "cube_gripper_carry", "门通行规则": "door", "传送门出口位置": "portal_exit",
}
DEV_ONLY = {"contact_friction", "motion_damping"}
FAMILIES = ("LeWM", "PLDM", "DINO-WM")
TRAINING_STAGES = ("original_environment_only", "post_component_training")
TRAINING_SEEDS = (3072, 3073, 3074)
ENVIRONMENTS = {
    "speed": "tworoom", "action_delay": "tworoom", "door": "tworoom", "portal_exit": "tworoom",
    "action_strength": "pusht", "contact_friction": "pusht", "motion_damping": "pusht",
    "robot_arm_mass": "reacher", "cube_gripper_carry": "cube",
}
ENVIRONMENT_NAMES = {"tworoom": "TwoRoom", "pusht": "PushT", "reacher": "Reacher", "cube": "Cube"}

ICL_MATRIX_BEGIN = "<!-- BEGIN CURRENT_REFERENCE_ICL_MATRIX -->"
ICL_MATRIX_END = "<!-- END CURRENT_REFERENCE_ICL_MATRIX -->"
CEM_MATRIX_BEGIN = "<!-- BEGIN CURRENT_REFERENCE_CEM_MATRIX -->"
CEM_MATRIX_END = "<!-- END CURRENT_REFERENCE_CEM_MATRIX -->"
DETAIL_BEGIN = "<!-- BEGIN CURRENT_REFERENCE_DETAIL -->"
DETAIL_END = "<!-- END CURRENT_REFERENCE_DETAIL -->"
ORIGINAL_DINO = ROOT / "configs/benchmark/contextworld_dinowm_original_fixed_context_results_v1.json"


def render(freeze: dict, document: str, appendix: str) -> tuple[str, str]:
    supplement = json.loads(ORIGINAL_DINO.read_text())
    if supplement["claim_boundary"]["data_only_ablation"] is not False:
        raise ValueError("Original DINO inference boundary must remain explicit")
    base = supplement["inputs"]["base_freeze"]
    if hashlib.sha256((ROOT / base["path"]).read_bytes()).hexdigest() != base["sha256"]:
        raise ValueError("Supplement base freeze changed")
    if json.loads((ROOT / base["path"]).read_text()) != freeze:
        raise ValueError("Supplement belongs to a different base freeze")
    rows = freeze["checkpoint_results"] + supplement["checkpoint_results"]
    if len(DEV_ONLY) != 2 or len(TASKS) - len(DEV_ONLY) != 7:
        raise ValueError("Reference report scope must remain 7 Test tasks + 2 Development tasks")
    dino_source = freeze["inputs"]["dino_original_diagnostic"]
    dino_path = ROOT / dino_source["path"]
    if hashlib.sha256(dino_path.read_bytes()).hexdigest() != dino_source["sha256"]:
        raise ValueError("DINO-WM original CEM source differs from the freeze")
    dino_cem = json.loads(dino_path.read_text())["original_environment_cem"]["environments"]

    def cells(task: str, family: str, stage: str) -> list[dict]:
        found = sorted(
            (r for r in rows if (r["component_id"], r["family"], r["stage"]) == (task, family, stage)),
            key=lambda r: r["training_seed"],
        )
        if [r["training_seed"] for r in found] != list(TRAINING_SEEDS):
            raise ValueError(f"Missing three-seed record: {task}/{family}/{stage}")
        return found

    def scores(values: list[dict], split: str, *, detail: bool = False) -> str:
        if not values:
            return "—"
        values = [r[split]["main_score"] * 100 for r in values]
        mean, sd = statistics.mean(values), statistics.stdev(values)
        if detail:
            return f"{mean:.2f} ± {sd:.2f}（{' / '.join(f'{v:.2f}' for v in values)}）"
        return f"{mean:.2f}% ± {sd:.2f}pp"

    def compact_scores(values: list[dict], split: str) -> str:
        """Return a matrix cell in percent units without repeating the unit."""
        if not values:
            return "—"
        numbers = [r[split]["main_score"] * 100 for r in values]
        return f"{statistics.mean(numbers):.2f} ± {statistics.stdev(numbers):.2f}"

    def cem_values(task: str, family: str, stage: str) -> list[float]:
        selected = cells(task, family, stage)
        if stage == "original_environment_only" and family == "DINO-WM":
            by_seed = dino_cem[ENVIRONMENTS[task]]["success_rate_by_training_seed"]
            expected = {str(seed) for seed in TRAINING_SEEDS}
            if set(by_seed) != expected:
                raise ValueError(f"Missing DINO CEM three-seed record: {task}")
            return [float(by_seed[str(seed)]) * 100 for seed in TRAINING_SEEDS]
        if not selected:
            raise ValueError(f"Missing CEM record: {task}/{family}/{stage}")
        if stage == "original_environment_only":
            return [
                float(row["original_cem_evidence"]["member"]["success_rate"])
                * 100
                for row in selected
            ]
        return [
            statistics.mean(
                row["original_task_cem"]["success_rate_percent_by_eval_seed"].values()
            )
            for row in selected
        ]

    def cem_cell(values: list[float], family: str) -> str:
        """One CEM retention cell: mean ± sample sd in percent units."""
        cell = f"{statistics.mean(values):.2f}% ± {statistics.stdev(values):.2f}pp"
        return cell + ("†" if family == "DINO-WM" else "")

    def matrix(header_labels: list[str], value_for: Callable[[str, str, str], str]) -> str:
        lines = [
            "| 模型 | 训练数据 | " + " | ".join(header_labels) + " |",
            "|---|---|" + "---:|" * len(header_labels),
        ]
        for family in FAMILIES:
            for stage, recipe in zip(TRAINING_STAGES, ("原环境数据", "原环境 + 对应 ICL 数据")):
                cells_for_row = [
                    value_for(task, family, stage)
                    for task in TASKS.values()
                ]
                lines.append("| " + " | ".join((family, recipe, *cells_for_row)) + " |")
        if len(lines) - 2 != 6:
            raise ValueError("Expected six rows in reference matrix")
        return "\n".join(lines)

    icl_labels = [
        label + "（Dev）" if task in DEV_ONLY else label
        for label, task in TASKS.items()
    ]
    def icl_cell(task: str, family: str, stage: str) -> str:
        value = compact_scores(
            cells(task, family, stage),
            "development" if task in DEV_ONLY else "test",
        )
        return value + ("‡" if family == "DINO-WM" and stage == "original_environment_only" else "")

    icl_matrix = matrix(icl_labels, icl_cell)

    def replace_marker(text: str, begin: str, end: str, replacement: str) -> str:
        if text.count(begin) != 1 or text.count(end) != 1:
            raise ValueError(f"Expected exactly one marker pair: {begin}")
        begin_at = text.index(begin) + len(begin)
        end_at = text.index(end, begin_at)
        return text[:begin_at] + "\n" + replacement + "\n" + text[end_at:]

    if CEM_MATRIX_BEGIN in document or CEM_MATRIX_END in document:
        raise ValueError(
            "CURRENT_REFERENCE_CEM_MATRIX moved to the appendix section "
            "5.3 原任务规划能力保持（CEM）; it must not appear in the main document"
        )
    document = replace_marker(document, ICL_MATRIX_BEGIN, ICL_MATRIX_END, icl_matrix)

    # §5.1 detail is ICL-only now; the CEM retention table owns its columns
    # in appendix section 5.3.
    detail_lines = [
        "| 能力类型 | 任务 | 模型 | 随机基线 | 原始 ICL 起点 | 组件训练后 ICL 主分数 | ICL 门槛结果 |",
        "|---|---|---|---:|---:|---:|:--|",
    ]
    ability = {
        "speed": "即时连续响应", "action_strength": "即时连续响应", "robot_arm_mass": "即时连续响应",
        "action_delay": "时间延迟动力学", "contact_friction": "接触或附着条件动力学",
        "motion_damping": "接触或附着条件动力学", "cube_gripper_carry": "接触或附着条件动力学",
        "door": "隐藏结构转移", "portal_exit": "隐藏结构转移",
    }
    for label, task in TASKS.items():
        for family in FAMILIES:
            split = "development" if task in DEV_ONLY else "test"
            original = cells(task, family, "original_environment_only")
            trained = cells(task, family, "post_component_training")
            suffix = "（Development）" if split == "development" else ""
            verdicts = [row[split]["all_gates_passed"] for row in trained]
            if not all(isinstance(value, bool) for value in verdicts):
                raise ValueError(f"Missing decision: {task}/{family}/{split}")
            count_passed = sum(verdicts)
            verdict = ("Development " if split == "development" else "") + (
                "3/3 通过" if count_passed == 3 else f"未通过（{count_passed}/3）"
            )
            chance = "33.33%" if task == "speed" else "16.67%" if task == "action_delay" else "50%"
            detail_lines.append(
                "| " + " | ".join(
                    (
                        ability[task], label, family, chance,
                        scores(original, split) + ("‡" if family == "DINO-WM" else "") + suffix,
                        scores(trained, split) + suffix,
                        verdict,
                    )
                ) + " |"
            )
    if len(detail_lines) - 2 != 27:
        raise ValueError(f"Expected 27 reference detail rows, found {len(detail_lines) - 2}")
    appendix = replace_marker(appendix, DETAIL_BEGIN, DETAIL_END, "\n".join(detail_lines))

    # §5.3 原任务规划能力保持（CEM）: one long row per model/component; the
    # delta column is plain arithmetic (post minus original) on the already
    # frozen per-seed scores, never a new metric or a pass claim.
    cem_lines = [
        "| 模型 | ICL 训练组件 | 原环境任务 | 原环境数据 CEM | ICL 配比 CEM | 变化（pp） |",
        "|---|---|---|---:|---:|---:|",
    ]
    for family in FAMILIES:
        for label, task in TASKS.items():
            original_values = cem_values(task, family, "original_environment_only")
            trained_values = cem_values(task, family, "post_component_training")
            delta = statistics.mean(trained_values) - statistics.mean(original_values)
            cem_lines.append(
                "| " + " | ".join(
                    (
                        family, label, ENVIRONMENT_NAMES[ENVIRONMENTS[task]],
                        cem_cell(original_values, family),
                        cem_cell(trained_values, family),
                        f"{delta:+.2f}",
                    )
                ) + " |"
            )
    if len(cem_lines) - 2 != 27:
        raise ValueError(f"Expected 27 CEM retention rows, found {len(cem_lines) - 2}")
    appendix = replace_marker(appendix, CEM_MATRIX_BEGIN, CEM_MATRIX_END, "\n".join(cem_lines))

    dev_lines = ["| 任务 | 模型 | 随机基线 | 训练前（逐种子） | 训练后（逐种子） |", "|---|---|---:|---:|---:|"]
    for label, task in TASKS.items():
        for family in FAMILIES:
            chance = "33.33%" if task == "speed" else "16.67%" if task == "action_delay" else "50%"
            original = scores(cells(task, family, "original_environment_only"), "development", detail=True)
            if family == "DINO-WM":
                original += "‡"
            trained = scores(cells(task, family, "post_component_training"), "development", detail=True)
            dev_lines.append(f"| {label} | {family} | {chance} | {original} | {trained} |")
    if len(dev_lines) - 2 != 27:
        raise ValueError(f"Expected 27 Development rows, found {len(dev_lines) - 2}")
    start = appendix.index("| 任务 | 模型 | 随机基线 |", appendix.index("## 5.2 Development"))
    end = appendix.index("\n\n", start)
    appendix = appendix[:start] + "\n".join(dev_lines) + appendix[end:]
    return document, appendix


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, default=ROOT / "configs/benchmark/contextworld_joint_scratch_v1_reference_results_freeze_v3.json")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    paths = [ROOT / "docs/ContextWorld_ICL_Benchmark.md", ROOT / "docs/reference/Benchmark_Result_Provenance.md"]
    current = [p.read_text() for p in paths]
    rendered = render(json.loads(args.freeze.read_text()), *current)
    if args.check:
        changed = [str(p) for p, before, after in zip(paths, current, rendered) if before != after]
        print("tables match freeze" if not changed else "tables drifted: " + ", ".join(changed))
        return int(bool(changed))
    for path, text in zip(paths, rendered):
        path.write_text(text)
    print("rendered 1 six-row ICL matrix, 27 ICL detail rows, 27 CEM retention rows, and 27 Development rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
