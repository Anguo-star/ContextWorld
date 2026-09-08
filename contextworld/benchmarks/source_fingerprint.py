"""Semantic fingerprints for the sources that release contracts hash-pin.

A release configuration pins the byte-exact ``sha256`` of every evaluation
source it was produced with.  That is the right contract for reproducing a
published number, but the wrong contract for auditing live drift: any edit
-- a clarified comment, a docstring fix, a reordered import, a blank line --
trips it exactly like a behavioural change would.  This module adds the
missing notion of *semantic* identity: the ``sha256`` of a normalized AST
dump with docstrings stripped and source positions excluded.  Sources with
equal semantic fingerprints parse to the same program modulo documentation
and layout, so byte drift between them cannot change evaluation behaviour.

Known limitations, kept on purpose:

* Renaming a local variable changes the fingerprint, because ``ast.dump``
  keeps identifier names.  A safe rename therefore still shows up as
  semantic drift and needs a correction record; erring on that side is the
  intended bias of an audit rule.
* The fingerprint says nothing about *runtime reachability*.  A branch that
  the pinned runtime can never activate still counts as semantic drift;
  accepting such a change is a recorded human decision, not something this
  module decides.
"""

from __future__ import annotations

import ast
import hashlib
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import yaml


_DOCSTRING_BEARING = (
    ast.Module,
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
)


@dataclass(frozen=True)
class SourceFingerprint:
    """Both identities of one source file.

    ``semantic_sha256`` is ``None`` for non-Python sources, for which no
    AST normalization exists and only the byte identity is meaningful.
    """

    byte_sha256: str
    semantic_sha256: str | None


def _strip_docstrings(tree: ast.AST) -> None:
    """Remove docstring statements in place.

    A statement is a docstring when it is the first statement of a module,
    function, or class and is a string constant expression.  Removing it can
    leave a body empty, which is not valid Python, so such bodies fall back
    to a single ``pass`` -- a canonical placeholder that keeps the tree
    comparable.
    """

    for node in ast.walk(tree):
        if isinstance(node, _DOCSTRING_BEARING) and node.body:
            first = node.body[0]
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                node.body = node.body[1:] or [ast.Pass()]


def semantic_fingerprint(source: str) -> str:
    """Return the semantic fingerprint of Python ``source``.

    The fingerprint is the sha256 of ``ast.dump`` with fields annotated and
    attributes excluded.  Excluding attributes drops line and column
    numbers, so blank lines, comment lines, indentation width and other
    layout-only edits leave the fingerprint unchanged.  Comments never enter
    the AST.  Docstrings are stripped first because they are string values
    in the tree rather than layout.

    As documented in the module docstring, identifier renames do change
    this fingerprint.
    """

    tree = ast.parse(source)
    _strip_docstrings(tree)
    dumped = ast.dump(tree, annotate_fields=True, include_attributes=False)
    return hashlib.sha256(dumped.encode("utf-8")).hexdigest()


def fingerprint_file(path: Path) -> SourceFingerprint:
    """Fingerprint one file, computing the semantic tier for ``.py`` only."""

    data = path.read_bytes()
    semantic = (
        semantic_fingerprint(data.decode("utf-8"))
        if path.suffix == ".py"
        else None
    )
    return SourceFingerprint(
        byte_sha256=hashlib.sha256(data).hexdigest(),
        semantic_sha256=semantic,
    )


@lru_cache(maxsize=None)
def _history_byte_sha_index(root: Path, relative: str) -> dict[str, str]:
    """Map every reachable historical byte sha256 of ``relative`` to content.

    Only ``git log --all`` reachable commits are visited, so blobs that
    predate a shallow clone boundary are absent from the index.  Any git
    failure yields an empty index rather than an error: the caller treats
    "cannot recover the pinned blob" as "cannot prove semantic equality".
    """

    log = subprocess.run(
        ["git", "-C", str(root), "log", "--all", "--format=%H", "--", relative],
        capture_output=True,
    )
    if log.returncode != 0:
        return {}
    index: dict[str, str] = {}
    for commit in log.stdout.decode("utf-8").split():
        blob = subprocess.run(
            ["git", "-C", str(root), "show", f"{commit}:{relative}"],
            capture_output=True,
        )
        if blob.returncode != 0:
            continue
        content = blob.stdout
        index.setdefault(hashlib.sha256(content).hexdigest(), content.decode("utf-8"))
    return index


def pinned_source_from_history(
    root: Path, relative: str, pinned_sha256: str
) -> str | None:
    """Recover the exact pinned content of ``relative`` from git history.

    Returns ``None`` when no reachable commit holds a blob whose byte
    ``sha256`` equals ``pinned_sha256`` (history rewritten or truncated by a
    shallow clone), in which case semantic equality with that pin cannot be
    established from the repository alone.
    """

    return _history_byte_sha_index(root, relative).get(pinned_sha256)


# ---------------------------------------------------------------------------
# Three-level grading of a pinned runtime source, shared by the pin audits.
#
# A pin either matches byte-for-byte (the reproduction contract), or the byte
# drift is excused by a registered accepted transition whose recorded current
# state is re-derived from the live file (a human decision with teeth), or the
# drift is proven documentation-only because the recovered pinned blob parses
# to the same program (``semantic_unchanged``), or the pin has genuinely
# drifted and the caller must fail.
# ---------------------------------------------------------------------------

BYTE_MATCH = "byte_match"
ACCEPTED_TRANSITION = "accepted_transition"
SEMANTIC_UNCHANGED = "semantic_unchanged"
DRIFTED = "drifted"


@dataclass(frozen=True)
class PinGrading:
    """The verdict for one pinned source, plus the reason for the tier."""

    status: str
    message: str | None = None

    @property
    def passed(self) -> bool:
        return self.status != DRIFTED


def load_accepted_pin_transitions(
    correction_yaml: Path,
) -> dict[tuple[str, str, str], dict[str, Any]]:
    """Load ``(config, path, pinned sha) -> acceptance row`` from the registry.

    Accepts the same ``accepted_metadata_correction`` record shape the release
    pin audit consumes: ``accepted_transitions`` plus the superseded-lineage
    and historical-receipt records.  The scope and classification claims are
    asserted, not trusted, so a record edited into a blanket exemption fails
    loudly here instead of silently downgrading every pin.
    """

    payload = yaml.safe_load(correction_yaml.read_text(encoding="utf-8"))
    if payload.get("status") != "accepted_metadata_correction":
        raise ValueError(f"Unexpected correction status: {correction_yaml}")
    if payload["scope"]["model_results_changed"]:
        raise ValueError("A pin correction must not change model results")
    if payload["finding"]["classification"] != (
        "runtime_source_fingerprint_updated_behavior_unchanged"
    ):
        raise ValueError("A pin correction must classify a behaviour-neutral drift")
    accepted: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in payload["accepted_transitions"]:
        accepted[(row["config"], row["path"], row["pinned_sha256"])] = row
    superseded = payload["superseded_release_lineage"]
    for row in superseded["affected_records"]:
        accepted[(superseded["release_config"], row["path"], row["pinned_sha256"])] = row
    for row in payload["historical_execution_receipts"]:
        accepted[(row["config"], row["path"], row["pinned_sha256"])] = row
    return accepted


def grade_source_pin(
    *,
    repo_root: Path,
    config_relative: str,
    path_relative: str,
    pinned_sha256: str,
    accepted: Mapping[tuple[str, str, str], dict[str, Any]],
) -> PinGrading:
    """Grade one pinned repository source on the three-level scale.

    ``DRIFTED`` messages are written for an audit failure, so they name the
    observed state and the reason no lower tier applies.
    """

    source = (repo_root / path_relative).resolve()
    try:
        source.relative_to(repo_root.resolve())
    except ValueError:
        return PinGrading(
            DRIFTED,
            f"{path_relative}: pinned path escapes the repository; byte pins on "
            "the third-party runtime snapshot are governed by its pinned ref, "
            "not by the local checkout state",
        )
    if not source.is_file():
        return PinGrading(DRIFTED, f"{path_relative}: missing from the checkout")
    observed = fingerprint_file(source)
    if observed.byte_sha256 == pinned_sha256:
        return PinGrading(BYTE_MATCH)
    acceptance = accepted.get((config_relative, path_relative, pinned_sha256))
    if acceptance is not None:
        if (
            acceptance.get("accepted_current_sha256") == observed.byte_sha256
            and acceptance.get("accepted_current_semantic_sha256")
            == observed.semantic_sha256
        ):
            return PinGrading(
                ACCEPTED_TRANSITION,
                f"{path_relative}: byte sha256 drifted "
                f"({pinned_sha256[:12]}… -> {observed.byte_sha256[:12]}…) but the "
                f"transition is a registered accepted correction in "
                f"{Path(config_relative).name}",
            )
        return PinGrading(
            DRIFTED,
            f"{path_relative}: pinned {pinned_sha256[:12]}… but file is "
            f"{observed.byte_sha256[:12]}…, and the registered state in the "
            "correction record no longer matches either — the source moved "
            "after the correction was accepted; re-assess before trusting "
            "results",
        )
    if observed.semantic_sha256 is not None:
        pinned_source = pinned_source_from_history(
            repo_root, path_relative, pinned_sha256
        )
        if pinned_source is not None and semantic_fingerprint(
            pinned_source
        ) == observed.semantic_sha256:
            return PinGrading(
                SEMANTIC_UNCHANGED,
                f"{path_relative}: byte sha256 drifted "
                f"({pinned_sha256[:12]}… -> {observed.byte_sha256[:12]}…) but "
                "the semantic fingerprint is unchanged — documentation or "
                "layout-only drift; update the pin record when convenient",
            )
        semantic = (
            observed.semantic_sha256[:12] + "…"
            if observed.semantic_sha256
            else "n/a for non-Python source"
        )
        return PinGrading(
            DRIFTED,
            f"{path_relative}: pinned {pinned_sha256[:12]}… but file is "
            f"{observed.byte_sha256[:12]}… (semantic {semantic}); evaluation "
            "behaviour may have changed — assess whether the results sealed "
            "against this pin need re-running",
        )
    return PinGrading(
        DRIFTED,
        f"{path_relative}: pinned {pinned_sha256[:12]}… but file is "
        f"{observed.byte_sha256[:12]}… and no semantic tier exists for a "
        "non-Python source",
    )
