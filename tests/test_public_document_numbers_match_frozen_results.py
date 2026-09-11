"""The numbers printed in the public document must match their sources.

The integrity reseal binds the public document's exact bytes, which proves the
document has not drifted since it was sealed.  It does not prove the document
was ever right: a score could be transcribed incorrectly, sealed, and pass
every check thereafter.  A test that asserts on a literal string
(``assert "96.70%" in section``) does not close that gap either, because
editing a score and its assertion together still passes.

This module closes it from the other side.  Every documented reference number
is parsed out of the document and compared against the machine-readable result
it came from:

* LeWM/PLDM ICL scores, seed verdicts and post-training CEM retention against
  the frozen public scoreboard;
* DINO-WM / PreJEPA ICL and original-environment CEM numbers against
  ``artifacts/evaluation/dinowm_original_diagnostic_v1/summary.json``, which is
  the only place those numbers exist.

Prose may be rewritten freely and headings may be renumbered; a number may not
move without its source moving with it.  The DINO diagnostic also records that
it is *not* part of the frozen formal matrix, so the document is required to
label it that way.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import re
import statistics

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_DOCUMENT = ROOT / "docs/ContextWorld_ICL_Benchmark.md"
SUITE_REGISTRY = ROOT / "configs/benchmark/contextworld_icl_suite_v2.yaml"
SCOREBOARD = (
    ROOT
    / "artifacts/evaluation/contextworld_icl_suite_v2_release_addendum_v1"
    / "public_scoreboard.json"
)
DINO_DIAGNOSTIC = (
    ROOT / "artifacts/evaluation/dinowm_original_diagnostic_v1/summary.json"
)
COMPLETE_COMPARISON = (
    ROOT
    / "artifacts/evaluation/complete_reference_comparison_v1"
    / "complete_comparison_v2.json"
)

# Rounding to two decimals moves a documented percentage by at most half of the
# last digit; anything beyond that is a transcription error, not formatting.
ROUNDING = 0.005 + 1e-9
# A mean recomputed from three already-rounded numbers can drift by the same
# half-digit again.
MEAN_ROUNDING = 0.005 + ROUNDING

_HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*$", re.MULTILINE)
_NUMBERING = re.compile(r"^\d+(?:\.\d+)*[.、]?\s+")
# CJK punctuation, unified ideographs and fullwidth forms (U+3000-U+303F,
# U+4E00-U+9FFF, U+FF01-U+FF65), written as literals to keep the class short.
_CJK = r"[　-〿一-鿿！-･]"
_PERCENT = re.compile(r"(\d+(?:\.\d+)?)%")
_SPREAD = re.compile(r"±\s*(\d+(?:\.\d+)?)\s*pp")

DINO_LABEL = re.compile(r"(DINO[\s-]?WM|DINO|PreJEPA)", re.IGNORECASE)
NON_FROZEN = re.compile(
    r"(非冻结|未冻结|不(属于|进入|改写|计入)[^，。；]{0,16}冻结|non[\s-]?frozen)"
)
SUPPLEMENTAL = re.compile(r"(补充|辅助|附加|supplement)", re.IGNORECASE)

FAMILIES = ("LeWM", "PLDM")


def _strip_numbering(title: str) -> str:
    return _NUMBERING.sub("", title).strip()


def _flatten(text: str) -> str:
    """Whitespace-normalized text with hard-wrap breaks inside CJK removed."""
    collapsed = re.sub(r"\s+", " ", text)
    return re.sub(rf"(?<={_CJK}) (?={_CJK})", "", collapsed)


def _statements(text: str) -> list[str]:
    return [part for part in re.split(r"[。；！？;]", _flatten(text)) if part.strip()]


@dataclass(frozen=True)
class Table:
    """A markdown table together with the heading path and prose around it."""

    path: tuple[str, ...]
    context: str
    header: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]

    def column(self, *keywords: str) -> int:
        """The one column whose header holds every keyword; exact hits win."""
        exact = [index for index, cell in enumerate(self.header) if cell == keywords[0]]
        if len(keywords) == 1 and len(exact) == 1:
            return exact[0]
        matches = [
            index
            for index, cell in enumerate(self.header)
            if all(keyword in cell for keyword in keywords)
        ]
        assert len(matches) == 1, (
            f"expected exactly one column matching {keywords}, "
            f"header is {list(self.header)}"
        )
        return matches[0]


def _markdown_tables(
    text: str,
) -> list[tuple[tuple[str, ...], tuple[tuple[str, ...], ...]]]:
    tables: list[tuple[tuple[str, ...], tuple[tuple[str, ...], ...]]] = []
    header: tuple[str, ...] | None = None
    rows: list[tuple[str, ...]] = []

    def flush() -> None:
        nonlocal header, rows
        if header is not None and rows:
            tables.append((header, tuple(rows)))
        header, rows = None, []

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            flush()
            continue
        cells = tuple(cell.strip() for cell in stripped.strip("|").split("|"))
        if header is None:
            header = cells
        elif set("".join(cells)) <= set("-:"):
            continue
        else:
            rows.append(cells)
    flush()
    return tables


def _sections(document: str) -> list[tuple[tuple[str, ...], str]]:
    """(heading path, own body) for every heading, nested sections excluded."""
    marks = list(_HEADING.finditer(document))
    stack: list[tuple[int, str]] = []
    sections: list[tuple[tuple[str, ...], str]] = []
    for index, mark in enumerate(marks):
        level = len(mark.group(1))
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, _strip_numbering(mark.group(2))))
        end = marks[index + 1].start() if index + 1 < len(marks) else len(document)
        sections.append((tuple(title for _, title in stack), document[mark.end():end]))
    return sections


def _tables(document: str) -> list[Table]:
    return [
        Table(path, body, header, rows)
        for path, body in _sections(document)
        for header, rows in _markdown_tables(body)
    ]


def _subtree(document: str, title: str) -> str:
    """Text under the heading with this numbering-free title, subsections included."""
    marks = list(_HEADING.finditer(document))
    matches = [mark for mark in marks if _strip_numbering(mark.group(2)) == title]
    assert len(matches) == 1, (
        f"expected exactly one heading titled {title!r}, found {len(matches)}"
    )
    mark = matches[0]
    level = len(mark.group(1))
    end = len(document)
    for later in marks[marks.index(mark) + 1:]:
        if len(later.group(1)) <= level:
            end = later.start()
            break
    return document[mark.end():end]


def _display_names() -> dict[str, str]:
    """Component id -> the task name the public document uses for it."""
    suite = yaml.safe_load(SUITE_REGISTRY.read_text(encoding="utf-8"))
    sections = suite["extension"]["public_document_template"]["component_sections"]
    return {
        component_id: _strip_numbering(section)
        for component_id, section in sections.items()
    }


DISPLAY_NAMES = _display_names()
ROW_KEYS = sorted(
    (component_id, family) for component_id in DISPLAY_NAMES for family in FAMILIES
)


def _document() -> str:
    return PUBLIC_DOCUMENT.read_text(encoding="utf-8")


def _reference_table(document: str) -> Table:
    """Read the paper matrices and join their cells to the appendix decisions.

    Numerical checks consume the displayed wide-table cells. The appendix is
    also checked for equality, so moving the detailed table cannot hide drift
    in either representation.
    """
    matrices = []
    for metric in ("ICL", "CEM"):
        begin = f"<!-- BEGIN CURRENT_REFERENCE_{metric}_MATRIX -->"
        end = f"<!-- END CURRENT_REFERENCE_{metric}_MATRIX -->"
        assert document.count(begin) == document.count(end) == 1
        block = document.split(begin)[1].split(end)[0]
        tables = _markdown_tables(block)
        assert len(tables) == 1
        matrices.append(tables[0])

    appendix = (ROOT / "docs/reference/Benchmark_Result_Provenance.md").read_text()
    details = [table for table in _tables(appendix)
               if "任务" in table.header and "ICL 门槛结果" in table.header]
    assert len(details) == 1
    detail = details[0]
    assert len(detail.rows) == 27
    labels = list(dict.fromkeys(row[1] for row in detail.rows))
    assert len(labels) == 9
    families = ("LeWM", "PLDM", "DINO-WM")
    recipes = ("原环境数据", "原环境 + 对应 ICL 数据")
    expected_keys = {(family, recipe) for family in families for recipe in recipes}
    indexed = []
    for index, (header, rows) in enumerate(matrices):
        expected_labels = [
            label + ("（Dev）" if index == 0 and label in ("接触摩擦", "运动阻尼") else "")
            for label in labels
        ]
        assert list(header) == ["模型", "训练数据", *expected_labels]
        assert len(rows) == 6 and all(len(row) == 11 for row in rows)
        assert {(row[0], row[1]) for row in rows} == expected_keys
        indexed.append({(row[0], row[1]): row for row in rows})

    def cell(metric: int, family: str, recipe: str, task: str) -> str:
        column = labels.index(task) + 2
        value = indexed[metric][family, recipe][column]
        if value == "—":
            return value
        match = re.fullmatch(r"(\d+\.\d+) ± (\d+\.\d+)([†‡]?)", value)
        assert match, f"malformed matrix value: {value!r}"
        mean, spread, suffix = match.groups()
        value = f"{mean}% ± {spread}pp{suffix}"
        if metric == 0 and "（Dev）" in matrices[0][0][column]:
            value += "（Development）"
        return value

    joined = []
    for row in detail.rows:
        values = list(row)
        task, family = row[1:3]
        for metric, recipe, column in ((0, recipes[0], 4), (0, recipes[1], 5),
                                       (1, recipes[0], 7), (1, recipes[1], 8)):
            values[column] = cell(metric, family, recipe, task)
            assert values[column] == row[column], (
                f"matrix/appendix mismatch for {task}/{family}/{recipe}: "
                f"{values[column]} != {row[column]}"
            )
        joined.append(tuple(values))
    displayed = [table for table in _tables(document) if "训练数据" in table.header]
    assert len(displayed) == 2
    return Table(displayed[0].path, displayed[0].context, detail.header, tuple(joined))


def _reference_row(component_id: str, family: str) -> tuple[Table, tuple[str, ...]]:
    table = _reference_table(_document())
    task = DISPLAY_NAMES[component_id]
    rows = [
        row
        for row in table.rows
        if row[table.column("任务")] == task and row[table.column("模型")] == family
    ]
    assert len(rows) == 1, (
        f"expected exactly one {task}/{family} row in the reference table, "
        f"got {len(rows)}"
    )
    return table, rows[0]


def _percentages(cell: str) -> list[float]:
    """Every percentage in a cell, in percent units."""
    return [float(value) for value in _PERCENT.findall(cell)]


def _spread(cell: str) -> float | None:
    """The ``± x pp`` sample standard deviation of a cell, in percent units."""
    match = _SPREAD.search(cell)
    return float(match.group(1)) if match else None


# ``artifacts/`` is gitignored apart from the DINO diagnostic summary, so the
# frozen scoreboard is present on a machine that holds the evaluation tree and
# absent in a clean checkout.  These tests skip when it is absent rather than
# fail, because a contributor without the artifacts has not done anything
# wrong.  The skip is deliberately loud about what went unverified -- see
# ``test_scoreboard_presence_is_reported``.
SCOREBOARD_AVAILABLE = SCOREBOARD.is_file()
requires_scoreboard = pytest.mark.skipif(
    not SCOREBOARD_AVAILABLE,
    reason=(
        "frozen scoreboard is not in this checkout (artifacts/ is gitignored); "
        "document-number verification did NOT run"
    ),
)


def _scoreboard_entries() -> dict[tuple[str, str], dict]:
    payload = json.loads(SCOREBOARD.read_text(encoding="utf-8"))
    entries: dict[tuple[str, str], dict] = {}
    for entry in payload["component_results"]:
        method = entry["method_name"]
        family = next((name for name in FAMILIES if name in method), None)
        if family is None:
            continue
        entries[(entry["component_id"], family)] = entry
    return entries


def _scoreboard_entry(row_key: tuple[str, str]) -> dict:
    entry = _scoreboard_entries().get(row_key)
    if entry is None:
        component_id, family = row_key
        pytest.skip(f"{component_id}/{family} is not in the frozen scoreboard")
    return entry


def _dino_summary() -> dict:
    return json.loads(DINO_DIAGNOSTIC.read_text(encoding="utf-8"))


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
def test_frozen_scoreboard_components_are_registered() -> None:
    """Row lookup keys come from the registry, so drift can only be one way.

    A component in the scoreboard that no longer exists in the suite registry
    would silently stop being compared against the document, because the
    per-row tests iterate over registered components.
    """
    scoreboard_ids = {component_id for component_id, _ in _scoreboard_entries()}
    unknown = sorted(scoreboard_ids - set(DISPLAY_NAMES))
    assert not unknown, (
        f"these frozen scoreboard components are not in the suite registry: "
        f"{unknown}; registered ids are {sorted(DISPLAY_NAMES)}"
    )


def test_dino_diagnostic_summary_is_available() -> None:
    """Unlike the scoreboard, this summary is kept in the repository."""
    assert DINO_DIAGNOSTIC.is_file(), (
        f"{DINO_DIAGNOSTIC.relative_to(ROOT)} is the only source for the "
        "documented DINO-WM / PreJEPA numbers and must stay in the checkout"
    )
    summary = _dino_summary()
    assert summary["icl"]["components"]
    assert summary["original_environment_cem"]["environments"]


def test_documented_cem_budget_matches_the_recorded_budget() -> None:
    """The standard planning budget is recorded per checkpoint, not asserted."""
    cem = _dino_summary()["original_environment_cem"]
    first, last = cem["eval_seeds"][0], cem["eval_seeds"][-1]
    per_seed = cem["episodes_per_eval_seed"]
    total = cem["episodes_per_checkpoint"]
    assert len(cem["eval_seeds"]) * per_seed == total

    quoted = [
        statement
        for statement in _statements(_document())
        if "CEM" in statement and "种子" in statement
    ]
    assert any(
        all(str(number) in statement for number in (first, last, per_seed, total))
        for statement in quoted
    ), (
        f"the document must state the standard CEM budget as seeds {first}-{last}, "
        f"{per_seed} episodes each, {total} in total"
    )


# ---------------------------------------------------------------------------
# §5.1 is the CURRENT standard reference: the joint_scratch_v1 recipe, scored
# under the completed anti-shortcut gates.  It is a different training
# generation from ``public_scoreboard.json``, whose method names still carry
# the superseded per-task recipes, so the two are deliberately NOT compared
# here -- the historical numbers keep their own static record under
# ``docs/archive/``.  What follows binds every documented §5.1 cell to
# ``contextworld_joint_scratch_v1_reference_results_freeze_v3.json``, which
# holds the per-checkpoint main score and gate verdict for both splits.
# ---------------------------------------------------------------------------

CURRENT_FREEZE = (
    ROOT
    / "configs/benchmark"
    / "contextworld_joint_scratch_v1_reference_results_freeze_v3.json"
)
# The two components marked ``failed_development`` have historical Test
# results, but those are excluded from final reporting. Their current cells
# carry the Development number and must say so.
DEVELOPMENT_ONLY_SPLIT = "development"


def _current_freeze() -> dict:
    return json.loads(CURRENT_FREEZE.read_text(encoding="utf-8"))


def _registry_status(component_id: str) -> str:
    suite = yaml.safe_load(SUITE_REGISTRY.read_text(encoding="utf-8"))
    return suite["components"][component_id]["reference_result_status"]


def _documented_split(component_id: str) -> str:
    if "development" in _registry_status(component_id):
        return DEVELOPMENT_ONLY_SPLIT
    return "test"


def _frozen_cells(
    component_id: str, family: str, stage: str
) -> list[dict]:
    return [
        row
        for row in _current_freeze()["checkpoint_results"]
        if row["component_id"] == component_id
        and row["family"] == family
        and row["stage"] == stage
    ]


CURRENT_ROW_KEYS = sorted(
    (component_id, family)
    for component_id in DISPLAY_NAMES
    for family in ("LeWM", "PLDM", "DINO-WM")
)


def test_current_freeze_covers_every_documented_row() -> None:
    """Every §5.1 row must have per-checkpoint evidence behind it."""
    table = _reference_table(_document())
    task = table.column("任务")
    model = table.column("模型")
    documented = {(row[task], row[model]) for row in table.rows}
    names = {name: cid for cid, name in DISPLAY_NAMES.items()}
    missing = []
    for task_name, family in sorted(documented):
        component_id = names[task_name]
        cells = _frozen_cells(component_id, family, "post_component_training")
        if len(cells) != 3:
            missing.append((task_name, family, len(cells)))
    assert not missing, (
        "these documented rows do not have three frozen checkpoints: "
        f"{missing}"
    )


@pytest.mark.parametrize("row_key", CURRENT_ROW_KEYS)
def test_documented_post_training_score_matches_the_current_freeze(
    row_key: tuple[str, str],
) -> None:
    """The 组件训练后 cell is the mean ± sample sd of the three checkpoints."""
    component_id, family = row_key
    split = _documented_split(component_id)
    cells = _frozen_cells(component_id, family, "post_component_training")
    if len(cells) != 3:
        pytest.skip(f"{component_id}/{family} has no three-checkpoint record")
    scores = [row[split]["main_score"] * 100 for row in cells]

    table, row = _reference_row(component_id, family)
    cell = row[table.column("ICL", "主分数")]
    if split == DEVELOPMENT_ONLY_SPLIT:
        assert "Development" in cell, (
            f"{row_key}: {_registry_status(component_id)} in the registry, so "
            "the documented score must say Development"
        )
    documented = _percentages(cell)
    assert documented, f"{row_key}: no percentage in {cell!r}"
    assert abs(documented[0] - statistics.mean(scores)) <= MEAN_ROUNDING, (
        f"{row_key}: document prints {documented[0]} but the three frozen "
        f"{split} checkpoints average {statistics.mean(scores)}"
    )
    spread = _spread(cell)
    assert spread is not None, f"{row_key}: cell carries no ± spread"
    assert abs(spread - statistics.stdev(scores)) <= MEAN_ROUNDING, (
        f"{row_key}: document prints ± {spread} but the frozen checkpoints "
        f"spread is {statistics.stdev(scores)}"
    )


@pytest.mark.parametrize("row_key", CURRENT_ROW_KEYS)
def test_documented_pre_training_score_matches_the_current_freeze(
    row_key: tuple[str, str],
) -> None:
    """Original scores bind to v3 or the marked DINO inference supplement."""
    component_id, family = row_key
    split = _documented_split(component_id)
    cells = _frozen_cells(component_id, family, "original_environment_only")
    table, row = _reference_row(component_id, family)
    cell = row[table.column("原始", "ICL")]

    if not cells:
        assert family == "DINO-WM"
        supplement = json.loads((ROOT / "configs/benchmark/contextworld_dinowm_original_fixed_context_results_v1.json").read_text())
        cells = [r for r in supplement["checkpoint_results"] if r["component_id"] == component_id]
        assert len(cells) == 3
        assert "‡" in cell
        assert supplement["claim_boundary"]["diagnostic"] is True
        assert supplement["claim_boundary"]["data_only_ablation"] is False
    scores = [row_[split]["main_score"] * 100 for row_ in cells]
    documented = _percentages(cell)
    assert documented, f"{row_key}: no percentage in {cell!r}"
    assert abs(documented[0] - statistics.mean(scores)) <= MEAN_ROUNDING, (
        f"{row_key}: document prints {documented[0]} as the starting point but "
        f"the frozen {split} baselines average {statistics.mean(scores)}"
    )
    spread = _spread(cell)
    assert spread is not None
    assert abs(spread - statistics.stdev(scores)) <= MEAN_ROUNDING


@pytest.mark.parametrize("row_key", CURRENT_ROW_KEYS)
def test_documented_gate_verdict_matches_the_current_freeze(
    row_key: tuple[str, str],
) -> None:
    """通过 only when all three checkpoints clear every gate of that component.

    This is the column that carries the S1 judgment, so a high main score with
    a failed gate must read 未通过 -- otherwise the table would present an
    unexcluded shortcut as a demonstrated ability.
    """
    component_id, family = row_key
    split = _documented_split(component_id)
    cells = _frozen_cells(component_id, family, "post_component_training")
    if len(cells) != 3:
        pytest.skip(f"{component_id}/{family} has no three-checkpoint record")
    verdicts = [row[split]["all_gates_passed"] for row in cells]
    assert all(isinstance(value, bool) for value in verdicts), (
        f"{component_id}/{family} lacks a versioned decision for the displayed split"
    )
    passed = sum(1 for value in verdicts if value)

    table, row = _reference_row(component_id, family)
    cell = row[table.column("ICL", "结果")]
    counts = re.search(r"(\d+)\s*/\s*(\d+)", cell)
    assert counts is not None, f"{row_key}: verdict cell {cell!r} has no n/3"
    assert (int(counts.group(1)), int(counts.group(2))) == (passed, 3), (
        f"{row_key}: document verdict says {counts.group(0)} but the frozen "
        f"checkpoints pass {passed}/3"
    )
    documented_pass = "未通过" not in cell
    assert documented_pass == (passed == 3), (
        f"{row_key}: document reads "
        f"{'通过' if documented_pass else '未通过'} but {passed}/3 checkpoints "
        "cleared every gate"
    )


@pytest.mark.parametrize("row_key", CURRENT_ROW_KEYS)
def test_documented_post_training_cem_matches_the_current_freeze(
    row_key: tuple[str, str],
) -> None:
    """训练后原任务 CEM is the mean ± sd over the three training seeds.

    Each seed's own value is the mean over its CEM evaluation seeds, so the
    documented spread is across training seeds, not across evaluation seeds.
    """
    component_id, family = row_key
    cells = _frozen_cells(component_id, family, "post_component_training")
    per_seed = [
        statistics.mean(
            row["original_task_cem"]["success_rate_percent_by_eval_seed"].values()
        )
        for row in cells
        if row.get("original_task_cem")
    ]
    if len(per_seed) != 3:
        pytest.skip(f"{component_id}/{family} has no three-seed CEM record")

    table, row = _reference_row(component_id, family)
    cell = row[table.column("训练后", "CEM")]
    documented = _percentages(cell)
    assert documented, f"{row_key}: no percentage in CEM cell {cell!r}"
    assert abs(documented[0] - statistics.mean(per_seed)) <= MEAN_ROUNDING, (
        f"{row_key}: document prints {documented[0]} but the frozen CEM runs "
        f"average {statistics.mean(per_seed)}"
    )
    spread = _spread(cell)
    assert spread is not None
    assert abs(spread - statistics.stdev(per_seed)) <= MEAN_ROUNDING


@pytest.mark.parametrize("row_key", CURRENT_ROW_KEYS)
def test_documented_original_cem_matches_the_current_freeze(
    row_key: tuple[str, str],
) -> None:
    component_id, family = row_key
    cells = _frozen_cells(component_id, family, "original_environment_only")
    if cells:
        values = [
            row["original_cem_evidence"]["member"]["success_rate"] * 100
            for row in cells
        ]
    else:
        assert family == "DINO-WM"
        environments = {
            "speed": "tworoom", "door": "tworoom", "portal_exit": "tworoom",
            "action_delay": "tworoom", "action_strength": "pusht",
            "contact_friction": "pusht", "motion_damping": "pusht",
            "robot_arm_mass": "reacher", "cube_gripper_carry": "cube",
        }
        original = _dino_summary()["original_environment_cem"]["environments"]
        values = [
            value * 100
            for value in original[environments[component_id]][
                "success_rate_by_training_seed"
            ].values()
        ]
    assert len(values) == 3
    table, row = _reference_row(component_id, family)
    cell = row[table.column("原始", "CEM")]
    assert abs(_percentages(cell)[0] - statistics.mean(values)) <= MEAN_ROUNDING
    spread = _spread(cell)
    assert spread is not None
    assert abs(spread - statistics.stdev(values)) <= MEAN_ROUNDING


def test_dino_cem_cells_are_marked_non_frozen() -> None:
    """DINO-WM CEM is supplemental evidence and must stay marked as such."""
    table = _reference_table(_document())
    model = table.column("模型")
    for column in (table.column("原始", "CEM"), table.column("训练后", "CEM")):
        for row in table.rows:
            if row[model] != "DINO-WM":
                continue
            if not _percentages(row[column]):
                continue
            assert "†" in row[column], (
                "DINO-WM CEM cells carry non-frozen supplemental evidence and "
                f"must be marked with †, got {row[column]!r}"
            )


def test_reference_table_still_reports_both_outcomes() -> None:
    """A table that only ever says 通过 has stopped being a check."""
    table = _reference_table(_document())
    verdicts = [row[table.column("ICL", "结果")] for row in table.rows]
    assert any("未通过" in verdict for verdict in verdicts)
    assert any(
        "未通过" not in verdict and "通过" in verdict for verdict in verdicts
    )
