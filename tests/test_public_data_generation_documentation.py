"""Regression tests for ``docs/Data_Generation.md``.

The data-generation guide is the public description of how the nine
ContextWorld components are built, so an external reader has to be able to
learn from it what the data actually is, and every repository entry point it
points at has to be real.

Nothing here pins prose or heading numbers.  Component names are read from
``configs/benchmark/contextworld_icl_suite_v2.yaml``, paths are checked against
the working tree, and the claims that make the data interpretable are matched
semantically, paragraph by paragraph:

* the future is a continuous simulator rollout, with no reset or state write
  between the context and the future;
* matched construction and split isolation;
* Speed Development is a history-utility diagnostic rather than the matched
  formal scoring construction used by most components;
* the Cube action-template constraints ``sum(p)=0`` and ``p[-1]=0``;
* which fields a model sees and which stay audit-only;
* the guide covers Training/Development/Test generation, states that Test is
  public for final reporting, and makes clear that packaging does not generate
  a new Test split.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import yaml


ROOT = Path(__file__).resolve().parents[1]
DOCUMENT = ROOT / "docs/Data_Generation.md"
SUITE_REGISTRY = ROOT / "configs/benchmark/contextworld_icl_suite_v2.yaml"

REPOSITORY_ROOTS = ("configs/", "scripts/", "contextworld/", "docs/", "tests/", "research/")
ENTRY_POINT_ROOTS = ("configs/", "scripts/", "contextworld/")

_HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*$", re.MULTILINE)
_NUMBERING = re.compile(r"^\d+(?:\.\d+)*[.、]?\s+")
# CJK punctuation, unified ideographs and fullwidth forms (U+3000-U+303F,
# U+4E00-U+9FFF, U+FF01-U+FF65), written as literals to keep the class short.
_CJK = r"[　-〿一-鿿！-･]"
_PATH = re.compile(r"(?:configs|scripts|contextworld|docs|tests|research)/[A-Za-z0-9_./-]+")
_LINK = re.compile(r"\]\(([^)\s]+)\)")

# The causal chain, however it is drawn: "x0 --u0--> x1 --u1--> x2 --u2--> x3".
ROLLOUT = re.compile(r"x_?0[^x]{0,24}x_?1[^x]{0,24}x_?2[^x]{0,24}x_?3", re.IGNORECASE)
SIMULATOR = re.compile(r"(模拟器|仿真器|simulator)")
CONTINUOUS = re.compile(r"(连续|向前执行|同一条|rollout|continuous)", re.IGNORECASE)
NO_RESET = re.compile(r"(没有|不会|不|无)[^，。；]{0,16}(reset|重置|重新初始化|重建)")
NO_STATE_WRITE = re.compile(r"不[^，。；]{0,12}(写入|写回|覆盖|改写|修改)[^，。；]{0,8}状态")
CONTEXT_AND_FUTURE = re.compile(r"(上下文|历史|context)[^。；]{0,24}(未来|future)")

MATCHED = re.compile(r"(配对|匹配|反事实|matched|counterfactual)", re.IGNORECASE)
SPLIT_TERMS = re.compile(r"(Training|Development|训练集|开发集)")
DISJOINT = re.compile(r"(互不重叠|不重叠|不相交|互斥|隔离|disjoint)", re.IGNORECASE)

SPEED = re.compile(r"(速度|speed)", re.IGNORECASE)
DIAGNOSTIC = re.compile(r"(诊断|diagnostic|history[\s-]?utility)", re.IGNORECASE)
NOT_FORMAL_SCORING = re.compile(
    r"(不是|并非|不同于|区别于|不产生|不构成|不复用|不属于|not)"
    r"[^，。；]{0,20}(匹配|配对|正式|通过判定|formal|matched|scoring)",
    re.IGNORECASE,
)

SUM_ZERO = re.compile(r"sum\s*\(\s*p\s*\)\s*=\s*0")
LAST_ZERO = re.compile(r"p\s*\[\s*-\s*1\s*\]\s*=\s*0")

MODEL_VISIBLE = re.compile(
    r"(模型可见|可见字段|模型输入|模型只(能)?(接收|看到|读取)|model[\s-]?visible)",
    re.IGNORECASE,
)
AUDIT_ONLY = re.compile(
    r"(仅用于审计|只用于审计|只保留在审计|审计通道|审计专用|audit[\s-]?only"
    r"|不进入模型输入|不对模型可见|不提供给模型)",
    re.IGNORECASE,
)
HIDDEN_STATE = re.compile(r"(隐藏|hidden|模拟器状态|完整状态|privileged)", re.IGNORECASE)

PUBLIC_TEST = re.compile(r"((?:Public\s*)?Test|测试(集|数据|划分))", re.IGNORECASE)
PUBLIC_DISTRIBUTION = re.compile(r"(公开|随数据包|public|distribut)", re.IGNORECASE)
FINAL_REPORTING = re.compile(r"(最终报告|final[ -]?report)", re.IGNORECASE)
NO_NEW_TEST_GENERATION = re.compile(
    r"(不会|不)[^。；]{0,16}(生成|重新生成)[^。；]{0,12}(测试|Test)",
    re.IGNORECASE,
)


def _strip_numbering(title: str) -> str:
    return _NUMBERING.sub("", title).strip()


def _flatten(text: str) -> str:
    """Whitespace-normalized text with hard-wrap breaks inside CJK removed."""
    collapsed = re.sub(r"\s+", " ", text)
    return re.sub(rf"(?<={_CJK}) (?={_CJK})", "", collapsed)


def _document() -> str:
    assert DOCUMENT.is_file(), (
        f"{DOCUMENT.relative_to(ROOT)} is missing; it is the public "
        "data-generation methodology for ContextWorld-v1"
    )
    return DOCUMENT.read_text(encoding="utf-8")


def _paragraphs(document: str) -> list[str]:
    """Blank-line separated blocks, normalized for matching."""
    return [
        _flatten(block)
        for block in re.split(r"\n\s*\n", document)
        if block.strip()
    ]


@dataclass(frozen=True)
class Table:
    path: tuple[str, ...]
    header: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]


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
        Table(path, header, rows)
        for path, body in _sections(document)
        for header, rows in _markdown_tables(body)
    ]


def _display_names() -> dict[str, str]:
    """Component id -> the task name the public documents use for it."""
    suite = yaml.safe_load(SUITE_REGISTRY.read_text(encoding="utf-8"))
    sections = suite["extension"]["public_document_template"]["component_sections"]
    names = {
        component_id: _strip_numbering(section)
        for component_id, section in sections.items()
    }
    assert set(names) == set(suite["components"])
    return names


DISPLAY_NAMES = _display_names()


def _component_of(label: str) -> str | None:
    """The one component a row label or heading names, if it names exactly one."""
    matches = [
        component_id
        for component_id, name in DISPLAY_NAMES.items()
        if name in label or component_id in label
    ]
    return matches[0] if len(matches) == 1 else None


def _referenced_paths(document: str) -> set[str]:
    """Repository paths the document points at, as repo-relative strings."""
    paths: set[str] = set()

    for match in _PATH.finditer(document):
        before = document[match.start() - 1: match.start()]
        if before and (before.isalnum() or before in {"/", ".", "-", "_", ":"}):
            continue  # part of a longer token, for example a URL
        if document[match.end(): match.end() + 1] in {"<", "{", "*"}:
            continue  # a templated path such as scripts/build_<component>_data.py
        paths.add(match.group(0).rstrip("./,;:)`"))

    for match in _LINK.finditer(document):
        target = match.group(1).split("#")[0]
        if not target or target.startswith(("http", "mailto:")):
            continue
        if target.startswith(REPOSITORY_ROOTS):
            paths.add(target)
            continue
        resolved = (DOCUMENT.parent / target).resolve()
        try:
            paths.add(str(resolved.relative_to(ROOT)))
        except ValueError:
            continue

    return paths


def _component_scopes(document: str) -> dict[str, list[str]]:
    """Text that belongs to a single component: its matrix rows and sections."""
    scopes: dict[str, list[str]] = {component_id: [] for component_id in DISPLAY_NAMES}
    for path, body in _sections(document):
        component = _component_of(path[-1])
        if component is not None:
            scopes[component].append(body)
    for table in _tables(document):
        for row in table.rows:
            component = _component_of(row[0])
            if component is not None:
                scopes[component].append(" | ".join(row))
    return scopes


def test_every_registered_component_is_documented() -> None:
    document = _flatten(_document())
    missing = [
        component_id
        for component_id, name in DISPLAY_NAMES.items()
        if name not in document and component_id not in document
    ]
    assert not missing, (
        f"the data-generation guide does not mention {missing}; all "
        f"{len(DISPLAY_NAMES)} registered components must be covered"
    )


def test_every_component_has_a_real_builder_or_config_entry_point() -> None:
    """The component matrix has to be usable, not decorative."""
    document = _document()
    scopes = _component_scopes(document)

    unscoped = sorted(
        component_id for component_id, texts in scopes.items() if not texts
    )
    assert not unscoped, (
        f"no matrix row or section names {unscoped}; the guide needs one entry "
        "per component so its generation path can be looked up"
    )

    missing: dict[str, list[str]] = {}
    for component_id, texts in scopes.items():
        candidates = {
            path
            for text in texts
            for path in _referenced_paths(text)
            if path.startswith(ENTRY_POINT_ROOTS)
        }
        existing = sorted(path for path in candidates if (ROOT / path).exists())
        if not existing:
            missing[component_id] = sorted(candidates)
    assert not missing, (
        "these components cite no builder or config entry point that exists in "
        f"the repository: {missing}"
    )


def test_referenced_repository_paths_exist() -> None:
    document = _document()
    missing = sorted(
        path for path in _referenced_paths(document) if not (ROOT / path).exists()
    )
    assert not missing, (
        f"the data-generation guide points at paths that do not exist: {missing}"
    )


def test_explains_the_continuous_simulator_rollout() -> None:
    document = _document()
    flat = _flatten(document)
    paragraphs = _paragraphs(document)

    assert ROLLOUT.search(flat), (
        "the guide must show the continuous causal chain x0 -> x1 -> x2 -> x3 "
        "that produces the query state and its real future"
    )
    assert any(
        SIMULATOR.search(paragraph) and CONTINUOUS.search(paragraph)
        for paragraph in paragraphs
    ), "the guide must say the rollout is executed continuously by the simulator"
    assert any(
        NO_RESET.search(paragraph) or NO_STATE_WRITE.search(paragraph)
        for paragraph in paragraphs
    ), (
        "the guide must say there is no reset and no state write between the "
        "context and the future"
    )
    assert any(
        CONTEXT_AND_FUTURE.search(paragraph)
        and (NO_RESET.search(paragraph) or NO_STATE_WRITE.search(paragraph))
        for paragraph in paragraphs
    ), (
        "the continuity claim must be tied to the context/future boundary, "
        "which is the boundary a reader cannot verify from the data alone"
    )


def test_explains_matched_construction_and_split_isolation() -> None:
    paragraphs = _paragraphs(_document())

    assert any(MATCHED.search(paragraph) for paragraph in paragraphs), (
        "the guide must explain matched / counterfactual construction"
    )
    assert any(
        SPLIT_TERMS.search(paragraph) and DISJOINT.search(paragraph)
        for paragraph in paragraphs
    ), (
        "the guide must explain that Training and Development are constructed "
        "split-disjoint"
    )


def test_marks_speed_development_as_a_history_utility_diagnostic() -> None:
    paragraphs = _paragraphs(_document())

    assert any(
        SPEED.search(paragraph)
        and "Development" in paragraph
        and DIAGNOSTIC.search(paragraph)
        for paragraph in paragraphs
    ), "the guide must describe Speed Development as a diagnostic"
    assert any(
        SPEED.search(paragraph)
        and DIAGNOSTIC.search(paragraph)
        and NOT_FORMAL_SCORING.search(paragraph)
        for paragraph in paragraphs
    ), (
        "the guide must say that Speed Development is not the matched formal "
        "scoring construction used by most components, so its numbers are not "
        "read as a component score"
    )


def test_states_the_cube_action_template_constraints() -> None:
    flat = _flatten(_document())

    assert SUM_ZERO.search(flat), (
        "the Cube construction must state the sum(p)=0 action-template constraint"
    )
    assert LAST_ZERO.search(flat), (
        "the Cube construction must state the p[-1]=0 action-template constraint"
    )


def test_separates_model_visible_fields_from_audit_only_metadata() -> None:
    paragraphs = _paragraphs(_document())

    assert any(
        MODEL_VISIBLE.search(paragraph) and AUDIT_ONLY.search(paragraph)
        for paragraph in paragraphs
    ), (
        "the guide must say, in one place, which fields a model sees and which "
        "hidden state or metadata stays audit-only"
    )
    assert any(
        AUDIT_ONLY.search(paragraph) and HIDDEN_STATE.search(paragraph)
        for paragraph in paragraphs
    ), "the audit-only channel must be described in terms of the hidden state it holds"


def test_public_test_is_final_reporting_and_export_does_not_regenerate_it() -> None:
    paragraphs = _paragraphs(_document())

    assert any(
        PUBLIC_TEST.search(paragraph)
        and PUBLIC_DISTRIBUTION.search(paragraph)
        and FINAL_REPORTING.search(paragraph)
        for paragraph in paragraphs
    ), (
        "the guide must say that Test is public and reserved for final reporting"
    )
    assert NO_NEW_TEST_GENERATION.search(_flatten(_document())), (
        "the guide must say that clean export packages frozen Test bytes and "
        "does not generate a new Test split"
    )
