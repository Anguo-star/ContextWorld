"""The numbers printed in the public document must match the frozen results.

The integrity reseal binds the public document's exact bytes, which proves the
document has not drifted since it was sealed.  It does not prove the document
was ever right: a score could be transcribed incorrectly, sealed, and pass
every check thereafter.  The existing documentation tests assert on literal
strings (``assert "96.70%" in section``), so editing a score and its assertion
together also passes.

This module closes that gap from the other side.  Every ICL primary score in
the section 5.3 reference table is parsed out of the document and compared
against the frozen scoreboard that produced it.  Prose may be rewritten
freely; a number may not move without the frozen result moving with it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_DOCUMENT = ROOT / "docs/ContextWorld_ICL_Benchmark.md"
SCOREBOARD = (
    ROOT
    / "artifacts/evaluation/contextworld_icl_suite_v2_release_addendum_v1"
    / "public_scoreboard.json"
)

# Document row label -> (component_id, method substring).  The scoreboard keys
# components by id and distinguishes the two methods by ``method_name``, which
# carries a longer human label than the document's 模型 column.
ROW_TO_COMPONENT = {
    ("速度", "LeWM"): ("speed", "LeWM"),
    ("速度", "PLDM"): ("speed", "PLDM"),
    ("推手移动幅度", "LeWM"): ("action_strength", "LeWM"),
    ("推手移动幅度", "PLDM"): ("action_strength", "PLDM"),
    ("机械臂质量", "LeWM"): ("robot_arm_mass", "LeWM"),
    ("机械臂质量", "PLDM"): ("robot_arm_mass", "PLDM"),
    ("动作延迟", "LeWM"): ("action_delay", "LeWM"),
    ("动作延迟", "PLDM"): ("action_delay", "PLDM"),
    ("门通行规则", "LeWM"): ("door", "LeWM"),
    ("门通行规则", "PLDM"): ("door", "PLDM"),
    ("传送门出口位置", "LeWM"): ("portal_exit", "LeWM"),
    ("传送门出口位置", "PLDM"): ("portal_exit", "PLDM"),
}


# ``artifacts/`` is gitignored, so the frozen scoreboard is present on a
# machine that holds the evaluation tree and absent in a clean checkout.  These
# tests skip when it is absent rather than fail, because a contributor without
# the artifacts has not done anything wrong.  The skip is deliberately loud
# about what went unverified -- see ``test_scoreboard_presence_is_reported``.
SCOREBOARD_AVAILABLE = SCOREBOARD.is_file()
requires_scoreboard = pytest.mark.skipif(
    not SCOREBOARD_AVAILABLE,
    reason=(
        "frozen scoreboard is not in this checkout (artifacts/ is gitignored); "
        "document-number verification did NOT run"
    ),
)


def _document() -> str:
    return PUBLIC_DOCUMENT.read_text(encoding="utf-8")


def _reference_rows() -> list[list[str]]:
    """The task-by-model reference table in the public result section."""
    document = _document()
    start = document.index("### 5.4 组件训练后的参考结果")
    end = document.index("\n### 5.5 外部模型验证状态", start)
    rows: list[list[str]] = []
    inside = False
    for line in document[start:end].splitlines():
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


def _scoreboard_entries() -> dict[tuple[str, str], dict]:
    payload = json.loads(SCOREBOARD.read_text(encoding="utf-8"))
    entries: dict[tuple[str, str], dict] = {}
    for entry in payload["component_results"]:
        method = entry["method_name"]
        family = "LeWM" if "LeWM" in method else "PLDM" if "PLDM" in method else None
        if family is None:
            continue
        entries[(entry["component_id"], family)] = entry
    return entries


def _percent(cell: str) -> float | None:
    """First percentage in a table cell, as a fraction."""
    import re

    match = re.search(r"(\d+\.\d+)%", cell)
    return float(match.group(1)) / 100.0 if match else None


def test_scoreboard_presence_is_reported() -> None:
    """Always runs, so a checkout without artifacts still says so out loud."""
    if not SCOREBOARD_AVAILABLE:
        pytest.skip(
            f"no frozen scoreboard at {SCOREBOARD.relative_to(ROOT)}; the "
            "documented scores were NOT checked against frozen results in "
            "this run"
        )
    assert SCOREBOARD.stat().st_size > 0


@requires_scoreboard
def test_scoreboard_is_readable_and_nonempty() -> None:
    entries = _scoreboard_entries()
    assert entries, "frozen scoreboard produced no LeWM/PLDM entries"


@requires_scoreboard
def test_every_mapped_component_exists_in_the_frozen_scoreboard() -> None:
    """A wrong component_id would otherwise skip, disguising a mapping typo.

    The per-row tests skip when a component is absent, because some documented
    rows are Development-only and never entered the formal scoreboard.  That
    escape hatch must not also absorb a misspelled id, so the mapping is
    checked against the scoreboard directly here.
    """
    entries = _scoreboard_entries()
    available = {component_id for component_id, _ in entries}
    mapped = {component_id for component_id, _ in ROW_TO_COMPONENT.values()}
    unknown = sorted(mapped - available)
    assert not unknown, (
        f"these component ids are not in the frozen scoreboard: {unknown}; "
        f"available ids are {sorted(available)}"
    )


@requires_scoreboard
@pytest.mark.parametrize("row_key", sorted(ROW_TO_COMPONENT))
def test_documented_icl_score_matches_frozen_scoreboard(
    row_key: tuple[str, str],
) -> None:
    """A documented ICL score must equal the frozen mean it came from."""
    task, family = row_key
    component_id, method_family = ROW_TO_COMPONENT[row_key]

    entries = _scoreboard_entries()
    entry = entries.get((component_id, method_family))
    if entry is None:
        pytest.skip(f"{component_id}/{method_family} is not in the frozen scoreboard")

    rows = [r for r in _reference_rows() if r[0] == task and r[1] == family]
    assert len(rows) == 1, f"expected exactly one {task}/{family} row, got {len(rows)}"

    documented = _percent(rows[0][3])
    assert documented is not None, f"no percentage in the {task}/{family} ICL cell"

    frozen = entry["icl_ability"]["primary_metric"]["mean"]
    assert documented == pytest.approx(frozen, abs=5e-5), (
        f"{task}/{family}: document says {documented:.4%} but the frozen "
        f"scoreboard mean is {frozen:.4%}"
    )


@requires_scoreboard
@pytest.mark.parametrize("row_key", sorted(ROW_TO_COMPONENT))
def test_documented_icl_verdict_matches_frozen_seed_stability(
    row_key: tuple[str, str],
) -> None:
    """通过（n/3）/未通过（0/3）must match the frozen per-checkpoint counts."""
    import re

    task, family = row_key
    component_id, method_family = ROW_TO_COMPONENT[row_key]

    entries = _scoreboard_entries()
    entry = entries.get((component_id, method_family))
    if entry is None:
        pytest.skip(f"{component_id}/{method_family} is not in the frozen scoreboard")

    rows = [r for r in _reference_rows() if r[0] == task and r[1] == family]
    assert len(rows) == 1
    verdict_cell = rows[0][4]

    stability = entry["icl_ability"]["training_seed_stability"]
    passed = stability["passed_checkpoints"]
    required = stability["required_checkpoints"]

    counts = re.search(r"(\d+)\s*/\s*(\d+)", verdict_cell)
    if counts is not None:
        assert (int(counts.group(1)), int(counts.group(2))) == (passed, required), (
            f"{task}/{family}: document verdict cell says {counts.group(0)} but "
            f"the frozen scoreboard recorded {passed}/{required}"
        )

    documented_pass = "未通过" not in verdict_cell
    frozen_pass = entry["icl_ability"]["result"] == "PASS"
    assert documented_pass == frozen_pass, (
        f"{task}/{family}: document reads "
        f"{'通过' if documented_pass else '未通过'} but the frozen result is "
        f"{entry['icl_ability']['result']}"
    )
