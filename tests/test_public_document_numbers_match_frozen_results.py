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
    """The task-by-model matrix of post-training reference results."""
    candidates = [
        table
        for table in _tables(document)
        if "任务" in table.header
        and "模型" in table.header
        and any("ICL" in cell for cell in table.header)
    ]
    assert len(candidates) == 1, (
        "expected exactly one 任务/模型 reference table carrying ICL columns, "
        f"found {len(candidates)}"
    )
    return candidates[0]


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


@requires_scoreboard
@pytest.mark.parametrize("row_key", ROW_KEYS)
def test_documented_icl_score_matches_frozen_scoreboard(
    row_key: tuple[str, str],
) -> None:
    """A documented ICL score must equal the frozen mean it came from."""
    entry = _scoreboard_entry(row_key)
    table, row = _reference_row(*row_key)

    documented = _percentages(row[table.column("ICL", "主分数")])
    assert documented, f"no percentage in the {row_key} ICL cell"

    frozen = entry["icl_ability"]["primary_metric"]["mean"] * 100
    assert documented[0] == pytest.approx(frozen, abs=ROUNDING), (
        f"{row_key}: document says {documented[0]:.2f}% but the frozen "
        f"scoreboard mean is {frozen:.4f}%"
    )


@requires_scoreboard
@pytest.mark.parametrize("row_key", ROW_KEYS)
def test_documented_icl_verdict_matches_frozen_seed_stability(
    row_key: tuple[str, str],
) -> None:
    """通过（n/3）/未通过（0/3）must match the frozen per-checkpoint counts."""
    entry = _scoreboard_entry(row_key)
    table, row = _reference_row(*row_key)
    verdict = row[table.column("ICL", "结果")]

    stability = entry["icl_ability"]["training_seed_stability"]
    passed = stability["passed_checkpoints"]
    required = stability["required_checkpoints"]

    counts = re.search(r"(\d+)\s*/\s*(\d+)", verdict)
    if counts is not None:
        assert (int(counts.group(1)), int(counts.group(2))) == (passed, required), (
            f"{row_key}: document verdict cell says {counts.group(0)} but the "
            f"frozen scoreboard recorded {passed}/{required}"
        )

    documented_pass = "未通过" not in verdict
    frozen_pass = entry["icl_ability"]["result"] == "PASS"
    assert documented_pass == frozen_pass, (
        f"{row_key}: document reads "
        f"{'通过' if documented_pass else '未通过'} but the frozen result is "
        f"{entry['icl_ability']['result']}"
    )


@requires_scoreboard
@pytest.mark.parametrize("row_key", ROW_KEYS)
def test_documented_retention_matches_frozen_retention(
    row_key: tuple[str, str],
) -> None:
    """Post-training CEM cells carry per-checkpoint values, not a summary.

    The frozen scoreboard keeps the minimum, maximum and mean of the retention
    runs, so the documented per-checkpoint list must span exactly that range
    and average to that mean.  A method whose retention was never authorized
    must show that instead of a number.
    """
    entry = _scoreboard_entry(row_key)
    table, row = _reference_row(*row_key)
    cell = row[table.column("训练后", "CEM")]
    values = _percentages(cell)
    retention = entry["original_task_retention"]

    if retention["result"] == "NOT_EVALUATED":
        assert not values, (
            f"{row_key}: the frozen scoreboard has no retention run "
            f"({retention.get('reason', '')}) but the document prints {cell!r}"
        )
        assert re.search(r"(未运行|未评测|未执行)", cell), (
            f"{row_key}: retention was not evaluated; the cell must say so, "
            f"not {cell!r}"
        )
        return

    metric = retention["primary_metric"]
    assert len(values) == retention["evaluated_checkpoints"], (
        f"{row_key}: the frozen scoreboard evaluated "
        f"{retention['evaluated_checkpoints']} checkpoints but the document "
        f"prints {len(values)} values"
    )
    assert min(values) == pytest.approx(metric["minimum"] * 100, abs=ROUNDING)
    assert max(values) == pytest.approx(metric["maximum"] * 100, abs=ROUNDING)
    assert sum(values) / len(values) == pytest.approx(
        metric["mean"] * 100, abs=MEAN_ROUNDING
    )

    documented_kept = "未保持" not in row[table.column("规划")]
    assert documented_kept == (retention["result"] == "PASS"), (
        f"{row_key}: document reads "
        f"{'保持' if documented_kept else '未保持'} but the frozen retention "
        f"result is {retention['result']}"
    )


@requires_scoreboard
def test_speed_card_per_checkpoint_values_match_the_frozen_range() -> None:
    """The Speed card quotes individual PLDM checkpoints; they are frozen too."""
    metric = _scoreboard_entry(("speed", "PLDM"))["icl_ability"]["primary_metric"]
    card = _subtree(_document(), DISPLAY_NAMES["speed"])
    quoted = [
        statement
        for statement in _statements(card)
        if "PLDM" in statement and "检查点" in statement
    ]
    assert quoted, "the Speed card no longer reports PLDM per-checkpoint ICL values"

    values = _percentages(" ".join(quoted))
    assert values
    assert min(values) == pytest.approx(metric["minimum"] * 100, abs=ROUNDING)
    assert max(values) == pytest.approx(metric["maximum"] * 100, abs=ROUNDING)


def test_dino_diagnostic_summary_is_available() -> None:
    """Unlike the scoreboard, this summary is kept in the repository."""
    assert DINO_DIAGNOSTIC.is_file(), (
        f"{DINO_DIAGNOSTIC.relative_to(ROOT)} is the only source for the "
        "documented DINO-WM / PreJEPA numbers and must stay in the checkout"
    )
    summary = _dino_summary()
    assert summary["icl"]["components"]
    assert summary["original_environment_cem"]["environments"]


def _dino_rows(table: Table) -> dict[str, tuple[str, ...]]:
    """Component id -> DINO-WM row in the unified task-by-model table."""
    task = table.column("任务")
    model = table.column("模型")
    rows = {
        component_id: next(
            (
                row
                for row in table.rows
                if row[task] == display_name and row[model] == "DINO-WM"
            ),
            None,
        )
        for component_id, display_name in DISPLAY_NAMES.items()
    }
    missing = [component_id for component_id, row in rows.items() if row is None]
    assert not missing, f"unified comparison table lacks DINO-WM rows for {missing}"
    return {component_id: row for component_id, row in rows.items() if row is not None}


def test_documented_dino_numbers_come_from_the_diagnostic_summary() -> None:
    """Every DINO original-data ICL/CEM value is bound to its diagnostic."""
    summary = _dino_summary()
    icl = summary["icl"]["components"]
    environments = summary["original_environment_cem"]["environments"]
    table = _reference_table(_document())
    rows = _dino_rows(table)
    original_icl = table.column("原始", "ICL")
    original_cem = table.column("原始", "CEM")

    assert set(rows) == set(icl)
    for component_id, row in rows.items():
        icl_source = icl[component_id]
        icl_values = _percentages(row[original_icl])
        assert len(icl_values) == 1
        assert icl_values[0] == pytest.approx(
            icl_source["mean"] * 100, abs=ROUNDING
        )
        assert _spread(row[original_icl]) == pytest.approx(
            icl_source["sample_standard_deviation"] * 100, abs=ROUNDING
        )

        cem_source = environments[icl_source["environment"]]
        cem_values = _percentages(row[original_cem])
        assert len(cem_values) == 1
        assert cem_values[0] == pytest.approx(
            cem_source["mean"] * 100, abs=ROUNDING
        )
        assert _spread(row[original_cem]) == pytest.approx(
            cem_source["sample_standard_deviation"] * 100, abs=ROUNDING
        )


def test_dino_cem_is_labelled_non_frozen_supplemental_evidence() -> None:
    """The diagnostic itself denies being part of the frozen formal matrix."""
    summary = _dino_summary()
    cem = summary["original_environment_cem"]
    assert cem["official_frozen_matrix"] is False
    assert summary["claim_boundary"]["cem_official_frozen_matrix"] is False
    assert summary["claim_boundary"]["public_reference_result"] is False

    table = _reference_table(_document())
    original_cem = table.column("原始", "CEM")
    for component_id, row in _dino_rows(table).items():
        cell = row[original_cem]
        assert NON_FROZEN.search(cell), (
            f"{component_id}: DINO CEM must say it is non-frozen"
        )
        assert SUPPLEMENTAL.search(cell), (
            f"{component_id}: DINO CEM must say it is supplementary evidence "
            f"({summary['status']})"
        )


def test_unified_matrix_keeps_frozen_and_diagnostic_evidence_distinct() -> None:
    """One display table must not erase the formal/non-frozen boundary."""
    table = _reference_table(_document())
    model = table.column("模型")
    original_cem = table.column("原始", "CEM")

    dino = [row for row in table.rows if row[model] == "DINO-WM"]
    frozen = [row for row in table.rows if row[model] in FAMILIES]
    assert len(dino) == len(DISPLAY_NAMES)
    assert len(frozen) == len(DISPLAY_NAMES) * len(FAMILIES)
    assert all(
        NON_FROZEN.search(row[original_cem])
        and SUPPLEMENTAL.search(row[original_cem])
        for row in dino
    )
    assert all(not NON_FROZEN.search(row[original_cem]) for row in frozen)


def test_supplemental_column_matches_complete_comparison_record() -> None:
    """Non-scoreboard ICL/CEM values remain bound to the complete record."""
    if not COMPLETE_COMPARISON.is_file():
        pytest.skip(
            f"no complete comparison at {COMPLETE_COMPARISON.relative_to(ROOT)}"
        )

    payload = json.loads(COMPLETE_COMPARISON.read_text(encoding="utf-8"))
    entries = {
        (entry["component_id"], entry["family"]): entry
        for entry in payload["rows"]
    }
    checked: set[tuple[str, str]] = set()

    for row_key, entry in entries.items():
        table, row = _reference_row(*row_key)
        cell = row[table.column("补充证据")]
        if cell == "—":
            continue

        values = _percentages(cell)
        if "ICL" in cell:
            assert values[0] == pytest.approx(
                entry["icl"]["mean"] * 100, abs=ROUNDING
            )
            assert _spread(cell) == pytest.approx(
                entry["icl"]["sample_std"] * 100, abs=ROUNDING
            )
            values = values[1:]

        expected_cem = [
            seed["success_rate"] * 100
            for seed in entry["original_task_cem"]["per_seed"]
        ]
        assert sorted(values) == pytest.approx(sorted(expected_cem), abs=ROUNDING)
        documented_kept = "未保持" not in cell
        assert documented_kept == (entry["original_task_cem"]["result"] == "PASS")
        checked.add(row_key)

    assert checked == {
        ("action_strength", "PLDM"),
        ("robot_arm_mass", "PLDM"),
        ("portal_exit", "PLDM"),
        ("contact_friction", "LeWM"),
        ("contact_friction", "PLDM"),
        ("motion_damping", "LeWM"),
        ("motion_damping", "PLDM"),
        ("cube_gripper_carry", "PLDM"),
    }


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
