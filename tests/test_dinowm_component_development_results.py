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
        "| 能力类型 | 任务 | 模型 | 随机基线 | 原始 ICL 起点 | 组件训练后 ICL 主分数 | ICL 门槛结果 | 原始 CEM 起点 | 训练后原任务 CEM |"
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
        "| 能力类型 | 任务 | 模型 | 随机基线 | 原始 ICL 起点 | 组件训练后 ICL 主分数 | ICL 门槛结果 | 原始 CEM 起点 | 训练后原任务 CEM |"
    ) == 1
    assert "### 5.3 DINO-WM / PreJEPA" not in section
    assert "Development 与 Public Test 可以出现在同一张表中" in section
    # The mid-training phrasing ("尚未训练" / "无可评分的 epoch-10 检查点") is
    # deliberately gone: DINO-WM now has all nine components at three seeds, so
    # a document still claiming otherwise would be stale.  What §5 must keep is
    # the pointer to its machine-readable source, plus the superseded records
    # kept for provenance.
    assert (
        "contextworld_joint_scratch_v1_reference_results_freeze_v1.json" in section
    )
    assert "complete_comparison_v2.json" in section
    assert "contextworld_dinowm_component_development_results_v1.json" in section
    assert "尚未训练" not in section
    assert "无可评分的 epoch-10 检查点" not in section


def test_dinowm_development_snapshot_is_marked_superseded() -> None:
    """This record is a mid-training Development snapshot, not a table source.

    It was written while DINO-WM component training was still running --
    ``action_delay`` had no checkpoint and ``action_strength`` had not started --
    and its speed entry reports ``history_better_rate``, which the record itself
    labels ``history_utility_diagnostic_not_matched_counterfactual``: a
    different metric from the speed main score used for LeWM and PLDM.  The
    comparison table must therefore NOT be pinned to it, or the benchmark would
    carry two speed standards at once.  The complete nine-component matrix in
    the joint_scratch_v1 freeze supersedes it, and
    ``test_public_document_numbers_match_frozen_results`` binds the documented
    DINO-WM cells to that freeze instead.  What stays authoritative here is the
    per-seed eval manifest identity and the CEM successes recorded at the time.
    """
    record = json.loads(RESULTS.read_text(encoding="utf-8"))
    superseded = record["superseded_by"]
    assert superseded["record"] == (
        "configs/benchmark/"
        "contextworld_joint_scratch_v1_reference_results_freeze_v1.json"
    )
    assert (ROOT / superseded["record"]).is_file(), (
        "the superseding freeze is named but absent"
    )
    assert record["evaluation"]["icl_split"] == "development"

    incomplete = {
        component_id
        for component_id, result in record["components"].items()
        if result["status"] != "complete_three_seed_development"
    }
    assert incomplete == {"action_delay", "action_strength"}, (
        "the snapshot's incomplete set changed; re-check whether it is still "
        f"the mid-training record this test describes: {sorted(incomplete)}"
    )
    assert record["components"]["speed"]["metric_kind"] == (
        "history_utility_diagnostic_not_matched_counterfactual"
    )
    for component_id, result in record["components"].items():
        if result["status"] != "complete_three_seed_development":
            continue
        assert len(result["icl_primary_by_training_seed"]) == 3, component_id
        assert len(result["eval_manifest_sha256_by_training_seed"]) == 3, component_id
        assert len(result["cem_successes_by_training_seed"]) == 3, component_id
