from __future__ import annotations

from contextworld.benchmarks.original_baseline_matrix import (
    audit_original_baseline_prereg,
    load_original_baseline_prereg,
)
from contextworld.benchmarks.original_baseline_archive import (
    audit_archived_original_baseline_matrix,
)
import pytest


def test_original_baseline_matrix_freezes_eight_checkpoints_and_18_cells() -> None:
    prereg = load_original_baseline_prereg()
    assert len(prereg["checkpoints"]) == 8
    assert len(prereg["components"]) == 9
    assert len(prereg["icl_cells"]) == 18
    assert {
        (row["environment"], row["family"])
        for row in prereg["checkpoints"]
    } == {
        (environment, family)
        for environment in ("tworoom", "pusht", "reacher", "cube")
        for family in ("lewm", "pldm")
    }
    assert all(
        row["contextworld_capability_training_used"] is False
        for row in prereg["checkpoints"]
    )
    assert all(
        row["formal_scoreboard_eligible"] is False
        for row in prereg["icl_cells"]
    )


def test_original_baseline_matrix_keeps_closed_public_splits_closed() -> None:
    prereg = load_original_baseline_prereg()
    components = {
        row["capability_id"]: row for row in prereg["components"]
    }
    for capability in (
        "contextworld-contact-friction",
        "contextworld-motion-damping",
    ):
        assert components[capability]["eval_subcommand"] == "eval-development"
        assert components[capability]["evaluation_phase"] == (
            "development_only_public_closed"
        )
        assert {
            row["phase"]
            for row in prereg["icl_cells"]
            if row["capability_id"] == capability
        } == {"development_only_public_closed"}


def test_original_action_delay_baselines_disclose_h3_to_h7_adaptation() -> None:
    prereg = load_original_baseline_prereg()
    rows = [
        row
        for row in prereg["icl_cells"]
        if row["capability_id"] == "contextworld-action-delay"
    ]
    assert len(rows) == 2
    assert all(
        row["history_adapter"]
        == "frozen_history7_inference_from_h3_checkpoint"
        and row["native_history7_checkpoint"] is False
        for row in rows
    )
    assert prereg["action_delay_disclosure"]["prohibited_claim"] == (
        "native_H7_original_checkpoint"
    )


def test_historical_matrix_is_audited_as_an_immutable_archive() -> None:
    # The preregistration correctly refuses to re-derive historical scores
    # against release files that evolved later.  The result archive is the
    # authority for those already-frozen receipts.
    with pytest.raises(RuntimeError, match="identity mismatch"):
        audit_original_baseline_prereg()
    audit = audit_archived_original_baseline_matrix()
    assert audit["status"] == "passed"
    assert audit["archive_scope"] == "immutable_frozen_results_only"
    assert audit["live_release_rederivation_performed"] is False
    assert audit["counts"]["canonical_checkpoints"] == 8
    assert audit["counts"]["icl_cells"] == 18
    assert audit["counts"]["authorized_cem_jobs"] == 0
