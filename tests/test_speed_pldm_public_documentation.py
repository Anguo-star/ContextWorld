from __future__ import annotations

from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_DOCUMENT = ROOT / "docs/ContextWorld_ICL_Benchmark.md"


def _trained_reference_section(document: str) -> str:
    start = document.index("### 5.4 组件训练后的参考结果")
    end = document.index("\n### 5.5 外部模型验证状态", start)
    return document[start:end]


def _trained_reference_rows(section: str) -> list[list[str]]:
    """Rows of the 任务—模型 reference table only.

    The section also carries the per-task ICL gate table, whose rows must not
    be counted as reference entries, so rows are collected only while inside
    the table introduced by the 任务 | 模型 header.
    """
    rows: list[list[str]] = []
    inside = False
    for line in section.splitlines():
        if not line.startswith("|"):
            inside = False
            continue
        if line.startswith("|---"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if cells[0] == "任务":
            inside = cells[1:2] == ["模型"]
            continue
        if inside:
            rows.append(cells)
    return rows


def _speed_task_card(document: str) -> str:
    start = document.index("#### 6.1.1 速度")
    end = document.index("\n#### 6.1.2 推手移动幅度", start)
    return document[start:end]


def test_trained_reference_table_covers_all_nine_tasks_and_both_models() -> None:
    document = PUBLIC_DOCUMENT.read_text(encoding="utf-8")
    section = _trained_reference_section(document)
    rows = _trained_reference_rows(section)

    assert len(rows) == 18
    assert Counter(row[0] for row in rows) == {
        "速度": 2,
        "推手移动幅度": 2,
        "机械臂质量": 2,
        "动作延迟": 2,
        "接触摩擦": 2,
        "运动阻尼": 2,
        "Cube 夹爪携带规则": 2,
        "门通行规则": 2,
        "传送门出口位置": 2,
    }
    assert all({row[1] for row in rows if row[0] == task} == {"LeWM", "PLDM"}
               for task in Counter(row[0] for row in rows))


def test_speed_pldm_public_writing_separates_evidence_scopes() -> None:
    document = PUBLIC_DOCUMENT.read_text(encoding="utf-8")
    section = _trained_reference_section(document)
    card = _speed_task_card(document)

    for stale in (
        "PLDM 结果待冻结",
        "速度 PLDM 当前没有已冻结的训练后结果",
        "尚无已冻结的训练后 reference",
        "当前没有已冻结的三个训练后 PLDM 结果",
    ):
        assert stale not in document

    assert "96.70%" in section
    assert "94.33%、94.33%、95.67%" in section
    assert "未运行 CEM" in section
    assert "原任务 CEM" in section
    assert "共 13 行" not in section

    assert "97.22%、96.44% 和" in card
    assert "平均 69.56%" in card
    assert "原任务 CEM 平均 94.78%" in card
    assert "不能把观察到的变化归因于多速度数据本身" in card


def test_formal_reference_is_not_presented_as_public_v1_distribution() -> None:
    document = PUBLIC_DOCUMENT.read_text(encoding="utf-8")
    section = _trained_reference_section(document)
    compact = "".join(document.split())

    assert "测试数据本身仍保持封存" in compact
    assert "结果表同时保留正结果和负结果" in section
