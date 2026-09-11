"""Render the current reference tables from one frozen numerical source."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TASKS = {
    "速度": "speed", "推手移动幅度": "action_strength", "机械臂质量": "robot_arm_mass",
    "动作延迟": "action_delay", "接触摩擦": "contact_friction", "运动阻尼": "motion_damping",
    "Cube 夹爪携带规则": "cube_gripper_carry", "门通行规则": "door", "传送门出口位置": "portal_exit",
}
DEV_ONLY = {"contact_friction", "motion_damping"}
ENVIRONMENTS = {
    "speed": "tworoom", "action_delay": "tworoom", "door": "tworoom", "portal_exit": "tworoom",
    "action_strength": "pusht", "contact_friction": "pusht", "motion_damping": "pusht",
    "robot_arm_mass": "reacher", "cube_gripper_carry": "cube",
}


def render(freeze: dict, document: str, appendix: str) -> tuple[str, str]:
    rows = freeze["checkpoint_results"]
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
        if not found and family == "DINO-WM" and stage == "original_environment_only":
            return []
        if [r["training_seed"] for r in found] != [3072, 3073, 3074]:
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

    lines = document.splitlines()
    table_start = next(i for i, line in enumerate(lines) if line.startswith("| 能力类型 | 任务 | 模型 | 随机基线 |"))
    count = 0
    for i in range(table_start + 2, len(lines)):
        if not lines[i].startswith("|"):
            break
        cols = [c.strip() for c in lines[i].strip("|").split("|")]
        task, family = TASKS[cols[1]], cols[2]
        split = "development" if task in DEV_ONLY else "test"
        original = cells(task, family, "original_environment_only")
        trained = cells(task, family, "post_component_training")
        suffix = "（Development）" if split == "development" else ""
        cols[4] = scores(original, split) + (suffix if original else "")
        cols[5] = scores(trained, split) + suffix
        verdicts = [r[split]["all_gates_passed"] for r in trained]
        if not all(isinstance(v, bool) for v in verdicts):
            raise ValueError(f"Missing decision: {task}/{family}/{split}")
        count_passed = sum(verdicts)
        cols[6] = ("Development " if split == "development" else "") + (
            "3/3 通过" if count_passed == 3 else f"未通过（{count_passed}/3）"
        )
        if original:
            original_cem = [r["original_cem_evidence"]["member"]["success_rate"] * 100 for r in original]
        else:
            original_cem = [v * 100 for v in dino_cem[ENVIRONMENTS[task]]["success_rate_by_training_seed"].values()]
        cols[7] = f"{statistics.mean(original_cem):.2f}% ± {statistics.stdev(original_cem):.2f}pp" + ("†" if family == "DINO-WM" else "")
        cem = [statistics.mean(r["original_task_cem"]["success_rate_percent_by_eval_seed"].values()) for r in trained]
        cols[8] = f"{statistics.mean(cem):.2f}% ± {statistics.stdev(cem):.2f}pp" + ("†" if family == "DINO-WM" else "")
        lines[i] = "| " + " | ".join(cols) + " |"
        count += 1
    if count != 27:
        raise ValueError(f"Expected 27 reference rows, found {count}")

    dev_lines = ["| 任务 | 模型 | 随机基线 | 训练前（逐种子） | 训练后（逐种子） |", "|---|---|---:|---:|---:|"]
    for label, task in TASKS.items():
        for family in ("LeWM", "PLDM", "DINO-WM"):
            chance = "33.33%" if task == "speed" else "16.67%" if task == "action_delay" else "50%"
            original = scores(cells(task, family, "original_environment_only"), "development", detail=True)
            trained = scores(cells(task, family, "post_component_training"), "development", detail=True)
            dev_lines.append(f"| {label} | {family} | {chance} | {original} | {trained} |")
    start = appendix.index("| 任务 | 模型 | 随机基线 |", appendix.index("## 5.2 Development"))
    end = appendix.index("\n\n", start)
    appendix = appendix[:start] + "\n".join(dev_lines) + appendix[end:]
    return "\n".join(lines) + "\n", appendix


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
    print("rendered 27 current reference rows and 27 Development rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
