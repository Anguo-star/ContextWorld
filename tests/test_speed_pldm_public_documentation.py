from __future__ import annotations

from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_DOCUMENT = ROOT / "docs/ContextWorld_ICL_Benchmark.md"


def _trained_reference_section(document: str) -> str:
    start = document.index("### 5.3 训练后参考结果")
    end = document.index("\n### 5.4 Cube 参考对照", start)
    return document[start:end]


def _speed_task_card(document: str) -> str:
    start = document.index("#### 6.1.1 速度")
    end = document.index("\n#### 6.1.2 推手移动幅度", start)
    return document[start:end]


def test_trained_reference_table_covers_all_nine_tasks_and_both_models() -> None:
    document = PUBLIC_DOCUMENT.read_text(encoding="utf-8")
    section = _trained_reference_section(document)
    rows = []
    for line in section.splitlines():
        if not line.startswith("|") or line.startswith("|---"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if cells[0] == "任务":
            continue
        rows.append(cells)

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
    assert "平均 94.78%" in section
    assert "该诊断没有性能通过门" in section
    assert "不能把表现变化归因于多速度训练" in section
    assert "共 13 行" in section

    assert "97.22%、96.44% 和" in card
    assert "平均 69.56%" in card
    assert "配对 retention CEM" in card
    assert "这些观察结果不能归因于多速度训练" in card


def test_formal_reference_is_not_presented_as_public_v1_distribution() -> None:
    section = _trained_reference_section(
        PUBLIC_DOCUMENT.read_text(encoding="utf-8")
    )
    normalized = " ".join(section.split())

    assert "不表示 Public v1 已经对外发布或可下载" in normalized
    assert "进入 scoreboard 不表示通过能力门" in section
