from __future__ import annotations

import copy

import pytest

from contextworld.benchmarks.original_baseline_archive import (
    audit_archived_original_baseline_matrix,
    validate_archived_original_baseline_summary,
)
from contextworld.paths import resolve_contextworld_path


def _summary() -> dict:
    import json

    path = resolve_contextworld_path(
        "artifacts/evaluation/original_baseline_matrix_v1/matrix_summary.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def test_archived_original_baseline_matrix_passes_without_live_rederivation() -> None:
    result = audit_archived_original_baseline_matrix()
    assert result["status"] == "passed"
    assert result["archive_scope"] == "immutable_frozen_results_only"
    assert result["live_release_rederivation_performed"] is False
    assert result["counts"]["icl_cells"] == 18


def test_archive_rejects_duplicate_or_missing_model_family_cell() -> None:
    summary = _summary()
    summary["cells"][1] = copy.deepcopy(summary["cells"][0])
    with pytest.raises(ValueError, match="Duplicate matrix cell"):
        validate_archived_original_baseline_summary(summary)


def test_archive_rejects_scoreboard_or_training_claim_expansion() -> None:
    summary = _summary()
    summary["formal_scoreboard_mutated"] = True
    with pytest.raises(ValueError, match="descriptive scope"):
        validate_archived_original_baseline_summary(summary)


def test_archive_rejects_changed_frozen_counts() -> None:
    summary = _summary()
    summary["counts"]["passing_single_checkpoint_gates"] = 2
    with pytest.raises(ValueError, match="count summary changed"):
        validate_archived_original_baseline_summary(summary)
