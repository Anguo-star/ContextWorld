"""Evidence-scope guards for the public ContextWorld benchmark document.

The document is rewritten for readability from time to time, so nothing here
depends on heading numbers or on verbatim sentences.  Sections and tables are
located by their role -- the one table with 任务 and 模型 columns, the task card
whose heading is the component's registry name -- task names are read from
``configs/benchmark/contextworld_icl_suite_v2.yaml``, and the claims that must
survive a rewrite are matched as patterns over normalized text.

Two of those claims decide how a reader reads the result tables:

* appearing in the formal reference matrix or in the public scoreboard is not
  the same as passing a component;
* the reference evidence collected here is not the Public v1 test
  distribution -- the test data itself stays withheld.

Both were dropped in an earlier documentation compression and are asserted
again below.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import re

import yaml


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_DOCUMENT = ROOT / "docs/ContextWorld_ICL_Benchmark.md"
SUITE_REGISTRY = ROOT / "configs/benchmark/contextworld_icl_suite_v2.yaml"

_HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*$", re.MULTILINE)
_NUMBERING = re.compile(r"^\d+(?:\.\d+)*[.、]?\s+")
# CJK punctuation, unified ideographs and fullwidth forms (U+3000-U+303F,
# U+4E00-U+9FFF, U+FF01-U+FF65), written as literals to keep the class short.
_CJK = r"[　-〿一-鿿！-･]"

# Phrasings a rewrite must not bring back: each described a state of the Speed
# PLDM evidence that stopped being true once those results were frozen.
STALE_CLAIMS = (
    "PLDM 结果待冻结",
    "速度 PLDM 当前没有已冻结的训练后结果",
    "尚无已冻结的训练后 reference",
    "当前没有已冻结的三个训练后 PLDM 结果",
)

# "being in the table" and "not the same as passing" have to meet in one
# sentence; either half on its own says nothing.
MEMBERSHIP = re.compile(
    r"(进入|列入|收录|纳入|写入|计入|出现在)[^，。；]{0,24}"
    r"(scoreboard|记分板|排行榜|结果表|参考(结果|矩阵|表)|正式(矩阵|结果|参考))"
)
NOT_A_PASS = re.compile(
    r"(不(表示|代表|等于|等同|意味着)|并不(表示|代表|等于|意味着)|≠)"
    r"[^，。；]{0,24}(通过|达标|PASS)"
)
TEST_SPLIT = re.compile(r"(Public\s*Test|Public\s*v1|测试(数据|集|划分))")
WITHHELD = re.compile(
    r"(封存|withheld"
    r"|不(随[^，。；]{0,12})?(分发|发布|公开|下载|提供)"
    r"|未(对外)?(分发|发布|公开)|尚未(公布|发布|开放)|不可下载)"
)
UNEVALUATED = re.compile(r"(未运行|未评测|未执行)")
EXPLAINS_UNEVALUATED = re.compile(r"(未满足|未通过|预先规定|预注册|未获授权|按预定规则)")


def _strip_numbering(title: str) -> str:
    return _NUMBERING.sub("", title).strip()


def _flatten(text: str) -> str:
    """Whitespace-normalized text with hard-wrap breaks inside CJK removed."""
    collapsed = re.sub(r"\s+", " ", text)
    return re.sub(rf"(?<={_CJK}) (?={_CJK})", "", collapsed)


def _statements(text: str) -> list[str]:
    """Sentence-sized fragments of normalized text."""
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


def _markdown_tables(text: str) -> list[tuple[tuple[str, ...], tuple[tuple[str, ...], ...]]]:
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
        f"expected exactly one heading titled {title!r} in "
        f"{PUBLIC_DOCUMENT.name}, found {len(matches)}"
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
    names = {
        component_id: _strip_numbering(section)
        for component_id, section in sections.items()
    }
    assert set(names) == set(suite["components"]), (
        "the registry's documentation sections and its components disagree: "
        f"{sorted(set(names) ^ set(suite['components']))}"
    )
    return names


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


def test_reference_table_covers_every_registered_component_and_both_models() -> None:
    table = _reference_table(_document())
    names = _display_names()
    task = table.column("任务")
    model = table.column("模型")

    tasks = [row[task] for row in table.rows]
    assert Counter(tasks) == {name: 2 for name in names.values()}, (
        "the reference table must carry one row per registered component per "
        f"reference model; it has {sorted(Counter(tasks).items())}"
    )
    for name in names.values():
        assert {row[model] for row in table.rows if row[task] == name} == {
            "LeWM",
            "PLDM",
        }


def test_speed_pldm_writing_separates_frozen_and_diagnostic_evidence() -> None:
    document = _document()
    flat = _flatten(document)
    card = _flatten(_subtree(document, _display_names()["speed"]))

    for stale in STALE_CLAIMS:
        assert stale not in flat, f"stale Speed PLDM claim is back: {stale}"

    # The frozen retention evidence and the ungated action-planning analysis
    # are different measurements and must stay distinguishable in the card.
    assert re.search(r"原任务\s*CEM", card), (
        "the Speed card must name the frozen original-task CEM retention result"
    )
    assert re.search(r"(action-planning|动作规划|规划)[^。；]{0,8}(分析|诊断)", card), (
        "the Speed card must name the separate action-planning analysis"
    )
    assert re.search(r"(没有|不设|未设|无)[^。；]{0,12}门槛", card), (
        "the Speed card must say the action-planning analysis has no "
        "pre-registered performance gate, so its number is not a pass"
    )

    # Without a single-speed control at the same training seed, nothing in the
    # card may be attributed to the multi-speed data itself.
    assert re.search(r"(单速度|单一速度)[^。；]{0,40}对照", card)
    assert re.search(r"不能[^。；]{0,24}归因", card)


def test_development_only_components_are_labelled_as_development() -> None:
    """A Development number must never be readable as a Public Test number."""
    suite = yaml.safe_load(SUITE_REGISTRY.read_text(encoding="utf-8"))
    names = _display_names()
    table = _reference_table(_document())
    task = table.column("任务")
    score = table.column("ICL", "主分数")

    development_only = [
        component_id
        for component_id, component in suite["components"].items()
        if "development" in component["reference_result_status"]
    ]
    assert development_only, "the registry no longer marks any component Development-only"

    for component_id in development_only:
        name = names[component_id]
        rows = [row for row in table.rows if row[task] == name]
        assert rows, f"no reference rows for {name}"
        for row in rows:
            assert "Development" in row[score], (
                f"{name} is {suite['components'][component_id]['reference_result_status']} "
                "in the registry, so its documented score must say Development"
            )


def test_scoreboard_membership_is_not_presented_as_a_capability_pass() -> None:
    document = _document()
    table = _reference_table(document)
    verdicts = [row[table.column("ICL", "结果")] for row in table.rows]

    assert any("未通过" in verdict for verdict in verdicts), (
        "the reference table has stopped reporting negative results"
    )
    assert any(
        "未通过" not in verdict and "通过" in verdict for verdict in verdicts
    ), "the reference table has stopped reporting positive results"

    assert any(
        MEMBERSHIP.search(statement) and NOT_A_PASS.search(statement)
        for statement in _statements(document)
    ), (
        "the document must state that being listed in the formal reference "
        "matrix or in the scoreboard is not the same as passing a component; "
        "no single sentence carries both halves of that claim"
    )


def test_reference_results_are_not_the_public_test_distribution() -> None:
    document = _document()
    table = _reference_table(document)
    assert len(table.path) >= 2, "the reference table is not inside a result section"
    results = _subtree(document, table.path[1])

    assert any(
        TEST_SPLIT.search(statement) and WITHHELD.search(statement)
        for statement in _statements(results)
    ), (
        "next to the numbers, the result section must say that the Public Test "
        "distribution itself stays withheld: recorded reference evidence is "
        "not a released test set"
    )


def test_unevaluated_retention_cells_are_explained() -> None:
    table = _reference_table(_document())
    cem = table.column("CEM")
    unevaluated = [row for row in table.rows if UNEVALUATED.search(row[cem])]
    if not unevaluated:
        return

    prose = _flatten(table.context)
    assert UNEVALUATED.search(prose) and EXPLAINS_UNEVALUATED.search(prose), (
        "the reference table leaves post-training CEM cells empty without "
        "explaining, in the same section, why those runs were not authorized"
    )
