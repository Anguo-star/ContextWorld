"""Claim boundaries for the partial DINO-WM component result record."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import re

import pytest


ROOT = Path(__file__).resolve().parents[1]
RESULTS = (
    ROOT
    / "configs/benchmark/contextworld_dinowm_component_development_results_v1.json"
)
BENCHMARK = ROOT / "docs/ContextWorld_ICL_Benchmark.md"
APPENDIX = ROOT / "docs/reference/Benchmark_Result_Provenance.md"
SOURCE = ROOT / "docs/research/data/icl_training_study_v2.json"
RENDERER = ROOT / "scripts/render_training_comparison.py"

_SEPARATOR_LINE = re.compile(r"^\|(\s*:?-{3,}:?\s*\|)+$")

ROUNDING = 0.005 + 1e-9
PERCENT = re.compile(r"(\d+(?:\.\d+)?)%")
SPREAD = re.compile(r"±\s*(\d+(?:\.\d+)?)\s*pp")


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
    assert "Public Test 没有打开" not in document
    assert "Development" in document and "冻结参考附录" in document
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


def _load_renderer():
    spec = importlib.util.spec_from_file_location("render_training_comparison_layout", RENDERER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _table_lines(text: str, marker: str) -> list[str]:
    begin, end = f"<!-- BEGIN TRAINING_COMPARISON_{marker} -->", f"<!-- END TRAINING_COMPARISON_{marker} -->"
    body = text.split(begin)[1].split(end)[0]
    return [line for line in body.splitlines() if line.startswith("|")]


def _body_rows(lines: list[str]) -> list[str]:
    """Table lines without the header/separator rows of each rendered table."""
    return [
        line
        for i, line in enumerate(lines)
        if not _SEPARATOR_LINE.match(line)
        and not (i + 1 < len(lines) and _SEPARATOR_LINE.match(lines[i + 1]))
    ]


def test_public_document_uses_overview_and_task_detail_tables() -> None:
    """The comparison section is an 11-row overview plus per-task detail blocks.

    The old layout rendered one 158-line unified table (one line per Speed
    distribution) straight from the representative rows.  The published study
    now carries per-training-run ``replicates``, so the renderer aggregates
    instead: a horizontal overview (model x scheme, one ``ICL / CEM`` cell per
    task, without a composite score, and
    comparable), one detail block per task with n(ICL)/n(CEM) and mean ± SD,
    a scaling block, and an appendix-only historical block.  This test pins
    that structure to the source JSON and the renderer itself.
    """
    document = BENCHMARK.read_text(encoding="utf-8")
    appendix = APPENDIX.read_text(encoding="utf-8")
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    rows = source["rows"]
    renderer = _load_renderer()

    # The single unified table is gone; each block appears exactly once in its owning doc.
    assert "TRAINING_COMPARISON_FULL" not in document
    for marker in renderer.MAIN_MARKERS:
        assert document.count(f"<!-- BEGIN TRAINING_COMPARISON_{marker} -->") == 1, marker
    assert document.count("<!-- BEGIN TRAINING_COMPARISON_HISTORICAL -->") == 0
    assert appendix.count("<!-- BEGIN TRAINING_COMPARISON_HISTORICAL -->") == 1
    assert appendix.count("<!-- BEGIN CURRENT_REFERENCE_ICL_MATRIX -->") == 1
    assert appendix.count("<!-- BEGIN CURRENT_REFERENCE_CEM_MATRIX -->") == 1

    start = document.index("## 5. 模型与训练方案比较")
    end = document.index("\n## 6. 任务说明", start)
    section = document[start:end]
    tasks = document[end:document.index("\n## 7. 接入与复现", end)]
    assert section.count("<!-- BEGIN TRAINING_COMPARISON_OVERVIEW -->") == 1
    assert section.count("<!-- BEGIN TRAINING_COMPARISON_SCALING -->") == 1
    for marker in renderer.DETAIL_MARKERS.values():
        assert tasks.count(f"<!-- BEGIN TRAINING_COMPARISON_{marker} -->") == 1, marker
    assert "JSON" in section and "CSV" in section
    assert "排队" not in section

    # Overview: eleven model x scheme rows, nine task cells, no composite score or seed column.
    overview = _table_lines(document, "OVERVIEW")
    body = _body_rows(overview)
    assert len(body) == 11
    assert all(len(line.split("|")) == 13 for line in body)  # 11 columns
    assert "ICL Avg" not in overview[0]
    for task in renderer.TASK_ORDER:
        assert renderer.TASK_ZH[task] in overview[0]
    pairs = [(line.split("|")[1].strip(), line.split("|")[2].strip()) for line in body]
    assert pairs == [
        (renderer.MODEL_ZH[model], renderer.REGIME_ZH[regime])
        for model, regime in renderer.OVERVIEW_GROUPS
    ]
    groups = dict(renderer.overview_groups(rows))
    for line, (model, regime) in zip(body, renderer.OVERVIEW_GROUPS):
        cells = [cell.strip() for cell in line.split("|")[3:-1]]
        for cell, task in zip(cells, renderer.TASK_ORDER):
            metrics, agg = renderer.display_stats(groups[(model, regime)][task])
            icl = renderer.f2(metrics["main"]["mean"])
            if task == "action_delay" and regime == "original" and icl != "—":
                icl += "†"
            cem = renderer.f2(agg["scores"]["cem"]["mean"])
            assert cell == ("—" if icl == cem == "—" else f"{icl} / {cem}")
    assert "†" in "\n".join(body)  # H3 Delay original reference stays flagged
    assert "历史转换初始化" not in "\n".join(body)

    # Detail blocks: every current ordinary row of the task, nothing else.
    for task, marker in renderer.DETAIL_MARKERS.items():
        lines = _table_lines(document, marker)
        detail_body = _body_rows(lines)
        current = [r for r in renderer.ordered_current(rows, task)
                   if any(st["n"] for st in renderer.display_stats(r)[0].values())]
        expected_labels = [
            (renderer.MODEL_ZH[r["model"]], renderer.scheme_label(r))
            for r in current for _ in range(4 if task == "speed" else 1)
        ]
        summary = [line for line in detail_body if len(line.split("|")) == len(lines[0].split("|"))]
        rendered = [(line.split("|")[1].strip(), line.split("|")[2].strip()) for line in summary]
        assert rendered == expected_labels, task
        assert "NRE↓" not in lines[0] and "响应分↑" in lines[0]
        assert "n(ICL)" in lines[0] and "n(CEM)" in lines[0] and "CEM↑" in lines[0]
        assert "历史转换初始化" not in "\n".join(detail_body)
        assert "2k" not in "\n".join(detail_body) and "10k 独立来源" not in "\n".join(detail_body)
        assert sum(line.startswith("| 模型") for line in lines) == 1
        if task == "speed":
            assert len(detail_body) == 4 * len(current)
            assert "评测条件" in lines[0]
            for label in ("低端外推", "高端外推", "未见速度插值", "训练中已见速度"):
                assert label in "\n".join(detail_body)
            assert "同组" in tasks
        if task in ("speed", "action_delay", "door"):
            assert "Joint↑" not in lines[0]  # Undefined metrics do not need empty columns
        if task == "action_delay":
            assert "原始模型†" in "\n".join(detail_body)
    assert "（未报告）" not in tasks and "（仅 CEM）" in tasks

    # Scaling: six small-vs-large Scratch comparisons plus the LeWM strength Joint ladder.
    scaling = _body_rows(_table_lines(document, "SCALING"))
    assert len(scaling) == len(renderer.scaling_rows(rows))
    assert len(scaling) == 39
    scaling_text = "\n".join(scaling)
    assert "10k 独立来源" in scaling_text and "32k 覆盖扩展" in scaling_text
    assert "二阶段冻结" not in scaling_text and "历史转换初始化" not in scaling_text

    # Historical projected rows live only in the provenance appendix.
    historical = _body_rows(_table_lines(appendix, "HISTORICAL"))
    assert len(historical) == sum(1 for r in rows if r["regime"] == "projected")
    assert len(historical) == 6
    assert all("DINO-WM" in line and "历史转换初始化" in line for line in historical)
    assert "ICL 从头" not in "\n".join(historical)


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
