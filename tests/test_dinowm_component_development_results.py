"""Claim boundaries for the partial DINO-WM component result record."""

from __future__ import annotations

import json
from pathlib import Path
import re
from statistics import mean, stdev

import pytest


ROOT = Path(__file__).resolve().parents[1]
RESULTS = (
    ROOT
    / "configs/benchmark/contextworld_dinowm_component_development_results_v1.json"
)
BENCHMARK = ROOT / "docs/ContextWorld_ICL_Benchmark.md"

ROUNDING = 0.005 + 1e-9
PERCENT = re.compile(r"(\d+(?:\.\d+)?)%")
SPREAD = re.compile(r"±\s*(\d+(?:\.\d+)?)\s*pp")


def _comparison_rows(document: str) -> dict[tuple[str, str], tuple[str, ...]]:
    header = (
        "| 能力类型 | 任务 | 模型 | 原始 ICL 起点 | 原始 CEM 起点 "
        "| 组件训练后 ICL 主分数 | ICL 结果 | 训练后原任务 CEM "
        "| 规划结果 | 补充证据（非正式） |"
    )
    lines = document.splitlines()
    start = lines.index(header)
    rows: dict[tuple[str, str], tuple[str, ...]] = {}
    for line in lines[start + 2 :]:
        if not line.startswith("|"):
            break
        cells = tuple(cell.strip() for cell in line.strip("|").split("|"))
        rows[(cells[1], cells[2])] = cells
    return rows


def test_partial_results_never_claim_public_test_or_scoreboard_status() -> None:
    record = json.loads(RESULTS.read_text(encoding="utf-8"))

    assert record["status"] == "partial_non_frozen_supplementary_evidence"
    assert record["claim_boundary"] == {
        "evaluation_split": "development",
        "public_test_accessed": False,
        "formal_pass_available": False,
        "official_scoreboard_row": False,
        "note": (
            "These values describe public ContextWorld-v1 Development "
            "evaluations and same-seed original-environment CEM retention. "
            "They are not frozen Public Test results."
        ),
    }
    assert "/opt/" not in RESULTS.read_text(encoding="utf-8")


def test_only_complete_three_seed_components_have_aggregate_values() -> None:
    record = json.loads(RESULTS.read_text(encoding="utf-8"))
    components = record["components"]
    complete = {
        name
        for name, result in components.items()
        if result["status"] == "complete_three_seed_development"
    }

    assert complete == {
        "speed",
        "door",
        "portal_exit",
        "contact_friction",
        "motion_damping",
        "robot_arm_mass",
        "cube_gripper_carry",
    }
    for name in complete:
        result = components[name]
        assert len(result["icl_primary_by_training_seed"]) == 3
        assert len(result["cem_successes_by_training_seed"]) == 3
        assert len(result["cem_delta_vs_original_by_training_seed"]) == 3
        manifests = result["eval_manifest_sha256_by_training_seed"]
        assert len(manifests) == 3
        assert all(len(digest) == 64 for digest in manifests)

    for name in {"action_delay", "action_strength"}:
        assert components[name]["icl_primary_by_training_seed"] is None
        assert components[name]["cem_successes_by_training_seed"] is None

    cube = components["cube_gripper_carry"]
    assert cube["icl_recovery_manifest_sha256_by_training_seed"][0] is None
    assert all(
        len(digest) == 64
        for digest in cube["icl_recovery_manifest_sha256_by_training_seed"][1:]
    )


def test_public_document_reports_all_nine_component_states() -> None:
    document = BENCHMARK.read_text(encoding="utf-8")

    assert "public_test_accessed=false" not in document
    assert "Public Test 没有打开" in document
    for label in (
        "速度",
        "门通行规则",
        "动作延迟",
        "传送门出口位置",
        "推手移动幅度",
        "接触摩擦",
        "运动阻尼",
        "机械臂质量",
        "Cube 夹爪携带规则",
    ):
        assert f"| {label} |" in document


def test_public_document_uses_one_split_aware_comparison_table() -> None:
    document = BENCHMARK.read_text(encoding="utf-8")
    start = document.index("## 5. 参考结果")
    end = document.index("\n## 6. 任务说明", start)
    section = document[start:end]

    assert section.count(
        "| 能力类型 | 任务 | 模型 | 原始 ICL 起点 | 原始 CEM 起点 "
        "| 组件训练后 ICL 主分数 | ICL 结果 | 训练后原任务 CEM "
        "| 规划结果 | 补充证据（非正式） |"
    ) == 1
    assert "### 5.3 DINO-WM / PreJEPA" not in section
    assert "Development 与 Public Test 可以出现在同一张表中" in section
    assert "尚未训练" in section
    assert "无可评分的 epoch-10 检查点" in section
    assert "complete_comparison_v2.json" in section
    assert "contextworld_dinowm_component_development_results_v1.json" in section


def test_documented_dinowm_component_values_match_the_three_seed_record() -> None:
    record = json.loads(RESULTS.read_text(encoding="utf-8"))
    rows = _comparison_rows(BENCHMARK.read_text(encoding="utf-8"))
    labels = {
        "speed": "速度",
        "door": "门通行规则",
        "action_delay": "动作延迟",
        "portal_exit": "传送门出口位置",
        "action_strength": "推手移动幅度",
        "contact_friction": "接触摩擦",
        "motion_damping": "运动阻尼",
        "robot_arm_mass": "机械臂质量",
        "cube_gripper_carry": "Cube 夹爪携带规则",
    }

    for component_id, result in record["components"].items():
        row = rows[(labels[component_id], "DINO-WM")]
        icl_cell = row[5]
        cem_cell = row[7]
        if result["status"] != "complete_three_seed_development":
            assert not PERCENT.findall(icl_cell)
            assert not PERCENT.findall(cem_cell)
            continue

        icl_values = result["icl_primary_by_training_seed"]
        documented_icl = [float(value) for value in PERCENT.findall(icl_cell)]
        assert documented_icl == pytest.approx(
            [mean(icl_values) * 100], abs=ROUNDING
        )
        icl_spread = SPREAD.search(icl_cell)
        assert icl_spread is not None
        assert float(icl_spread.group(1)) == pytest.approx(
            stdev(icl_values) * 100, abs=ROUNDING
        )

        cem_rates = [value / 300 for value in result["cem_successes_by_training_seed"]]
        documented_cem = [float(value) for value in PERCENT.findall(cem_cell)]
        assert documented_cem == pytest.approx(
            [mean(cem_rates) * 100], abs=ROUNDING
        )
        cem_spread = SPREAD.search(cem_cell)
        assert cem_spread is not None
        assert float(cem_spread.group(1)) == pytest.approx(
            stdev(cem_rates) * 100, abs=ROUNDING
        )
