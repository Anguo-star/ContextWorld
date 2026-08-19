from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from contextworld.benchmarks.original_baseline_cem_matrix import (
    EXPECTED_CELLS,
    EXPECTED_EXECUTION_CELLS,
    load_preregistration,
)


ROOT = Path(__file__).resolve().parents[1]


def test_original_baseline_cem_matrix_is_exactly_four_by_two() -> None:
    assert len(EXPECTED_CELLS) == 8
    assert len(EXPECTED_EXECUTION_CELLS) == 7
    assert ("tworoom", "lewm") in EXPECTED_EXECUTION_CELLS
    assert ("cube", "lewm") not in EXPECTED_EXECUTION_CELLS


def test_original_baseline_cem_preregistration_is_closed() -> None:
    # Output absence was a pre-execution condition.  The completed matrix now
    # keeps that preregistration immutable and binds the final eight-cell result
    # in a separate freeze.
    prereg = yaml.safe_load(
        (
            ROOT
            / "configs/benchmark/contextworld_original_baseline_cem_prereg_v1.yaml"
        ).read_text(encoding="utf-8")
    )
    assert prereg["authority"]["authorized_new_cells"] == 7
    assert prereg["scientific_scope"]["newly_executed_episodes"] == 2100
    assert prereg["scientific_scope"]["pass_fail_threshold"] is None
    assert prereg["status"] == "frozen_before_cem_execution"
    assert len(prereg["implementation"]) >= 5
    freeze = json.loads(
        (
            ROOT
            / "configs/benchmark/contextworld_original_baseline_cem_results_freeze_v1.json"
        ).read_text(encoding="utf-8")
    )
    assert freeze["status"] == "frozen_after_completed_descriptive_matrix"
    assert freeze["matrix_summary"]["matrix_cells"] == 8
    assert freeze["matrix_summary"]["total_matrix_episodes"] == 2400


def test_pending_identity_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "pending.yaml"
    path.write_text("status: PENDING\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="pending identities"):
        load_preregistration(path, repo_root=ROOT)
