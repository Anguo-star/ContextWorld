from __future__ import annotations

import pytest

import scripts.finalize_contextworld_original_baseline_cem_recovery_v1 as recovery


def _row(**overrides):
    row = {"success_count": 2, "query_count": 3, "success_rate": 2 / 3}
    row.update(overrides)
    return row


def test_query_count_is_the_only_added_evaluation_alias() -> None:
    assert recovery._validate_aggregate_with_query_count(
        _row(), successes=2, evaluations=3, label="seed"
    ) == {
        "success_count": 2,
        "evaluation_count": 3,
        "success_rate": 2 / 3,
    }


def test_query_count_conflict_fails_closed() -> None:
    with pytest.raises(
        recovery.original.FinalizationError,
        match="evaluation-count aliases conflict",
    ):
        recovery._validate_aggregate_with_query_count(
            _row(evaluation_count=4),
            successes=2,
            evaluations=3,
            label="seed",
        )


def test_all_original_aggregate_checks_still_apply() -> None:
    with pytest.raises(
        recovery.original.FinalizationError, match="success count drifted"
    ):
        recovery._validate_aggregate_with_query_count(
            _row(success_count=1),
            successes=2,
            evaluations=3,
            label="seed",
        )
    with pytest.raises(
        recovery.original.FinalizationError, match="evaluation count drifted"
    ):
        recovery._validate_aggregate_with_query_count(
            _row(query_count=4),
            successes=2,
            evaluations=3,
            label="seed",
        )
