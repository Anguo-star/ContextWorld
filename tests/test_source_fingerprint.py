"""The semantic fingerprint must ignore layout and documentation, nothing else.

These properties are the entire value of the semantic tier in
``test_external_model_cli``: the fingerprint has to stay stable across every
edit that cannot change evaluation behaviour, and it has to move for every
edit that can.  If an assertion here starts failing, the audit rule's
pass/warn/fail split is wrong, not this file.
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import yaml

from contextworld.benchmarks.source_fingerprint import (
    fingerprint_file,
    pinned_source_from_history,
    semantic_fingerprint,
)


ROOT = Path(__file__).resolve().parents[1]

BASE = '''\
"""Module docstring for the frozen scorer."""

import hashlib


def score(value, limit=3):
    """Compute the benchmark score."""
    total = 0
    for item in range(limit):
        total += value * item  # weighted accumulation
    if total > 10:
        return total - 10
    return total
'''


class TestInvariance:
    def test_adding_only_comments_keeps_the_fingerprint(self) -> None:
        commented = "# header comment\n" + BASE.replace(
            "    return total\n", "    # trailing note\n    return total\n"
        )

        assert semantic_fingerprint(commented) == semantic_fingerprint(BASE)
        # ... while the byte identity of course moves.
        assert hashlib.sha256(commented.encode()).hexdigest() != (
            hashlib.sha256(BASE.encode()).hexdigest()
        )

    def test_editing_docstrings_keeps_the_fingerprint(self) -> None:
        redocumented = BASE.replace(
            "Compute the benchmark score.", "Recompute the benchmark score.",
        ).replace(
            "Module docstring for the frozen scorer.",
            "A completely different module summary.",
        )

        assert semantic_fingerprint(redocumented) == semantic_fingerprint(BASE)

    def test_blank_lines_keep_the_fingerprint(self) -> None:
        spaced = BASE.replace("    total = 0\n", "\n\n    total = 0\n")

        assert semantic_fingerprint(spaced) == semantic_fingerprint(BASE)

    def test_indentation_width_keeps_the_fingerprint(self) -> None:
        # Re-indent every statement from four spaces per level to two, by
        # rewriting only the leading run of spaces.
        reindented = "\n".join(
            " " * ((len(line) - len(line.lstrip(" "))) // 2) + line.lstrip(" ")
            for line in BASE.splitlines()
        ) + "\n"

        assert semantic_fingerprint(reindented) == semantic_fingerprint(BASE)

    def test_arbitrary_reserialization_keeps_the_fingerprint(self) -> None:
        """``ast.unparse`` changes quotes, spacing and parentheses everywhere."""

        reserialized = ast.unparse(ast.parse(BASE))

        assert semantic_fingerprint(reserialized) == semantic_fingerprint(BASE)


class TestSensitivity:
    def test_changing_a_constant_value_moves_the_fingerprint(self) -> None:
        assert semantic_fingerprint(
            BASE.replace("limit=3", "limit=4")
        ) != semantic_fingerprint(BASE)

    def test_changing_a_comparison_operator_moves_the_fingerprint(self) -> None:
        assert semantic_fingerprint(
            BASE.replace("total > 10", "total >= 10")
        ) != semantic_fingerprint(BASE)

    def test_adding_a_branch_moves_the_fingerprint(self) -> None:
        assert semantic_fingerprint(
            BASE.replace(
                "    return total\n",
                "    if total == 10:\n        return 0\n    return total\n",
            )
        ) != semantic_fingerprint(BASE)

    def test_removing_a_branch_moves_the_fingerprint(self) -> None:
        assert semantic_fingerprint(
            BASE.replace("    if total > 10:\n        return total - 10\n", "")
        ) != semantic_fingerprint(BASE)

    def test_renaming_a_local_variable_moves_the_fingerprint(self) -> None:
        """Documented limitation, not a defect: the AST keeps identifier names.

        A safe rename therefore surfaces as semantic drift and needs a
        correction record; see the module docstring.
        """

        assert semantic_fingerprint(
            BASE.replace("total", "accumulator")
        ) != semantic_fingerprint(BASE)


class TestFileLevelApi:
    def test_python_files_carry_both_identities(self, tmp_path: Path) -> None:
        source = tmp_path / "scorer.py"
        source.write_text(BASE, encoding="utf-8")

        fingerprint = fingerprint_file(source)

        assert fingerprint.byte_sha256 == (
            hashlib.sha256(BASE.encode("utf-8")).hexdigest()
        )
        assert fingerprint.semantic_sha256 == semantic_fingerprint(BASE)

    def test_non_python_sources_have_no_semantic_tier(
        self, tmp_path: Path
    ) -> None:
        launcher = tmp_path / "train.sh"
        launcher.write_text("#!/bin/sh\nexec python -m train\n", encoding="utf-8")

        fingerprint = fingerprint_file(launcher)

        assert fingerprint.semantic_sha256 is None
        assert fingerprint.byte_sha256 == hashlib.sha256(
            b"#!/bin/sh\nexec python -m train\n"
        ).hexdigest()


class TestHistoryRecovery:
    def test_a_live_release_pin_is_recoverable_byte_exact(self) -> None:
        door = yaml.safe_load(
            (ROOT / "configs/benchmark/tworoom_door_icl_release_v1.yaml")
            .read_text(encoding="utf-8")
        )
        pinned = door["runtime"]["contextworld"]["source_sha256"][
            "contextworld/benchmarks/adapters.py"
        ]

        recovered = pinned_source_from_history(
            ROOT, "contextworld/benchmarks/adapters.py", pinned
        )

        assert recovered is not None
        assert hashlib.sha256(recovered.encode("utf-8")).hexdigest() == pinned

    def test_an_unknown_sha_recovers_nothing(self) -> None:
        assert pinned_source_from_history(
            ROOT,
            "contextworld/benchmarks/adapters.py",
            "0" * 64,
        ) is None
