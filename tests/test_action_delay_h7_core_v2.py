from __future__ import annotations

from scripts.analyze_tworoom_action_delay_h7_core_v2 import (
    summarize_h1_physical,
)
from contextworld.benchmarks.adapters import (
    StableWorldModelLeWMHistory7Adapter,
    StableWorldModelPLDMHistory7Adapter,
)


def _rows(*, fail_group_zero: bool = False) -> list[dict]:
    rows = []
    for query_index in range(4):
        for delay in range(11):
            selected = delay
            if fail_group_zero and delay == 0:
                selected = 1
            rows.append(
                {
                    "query_id": f"q{query_index}",
                    "eval_seed": 42 + query_index,
                    "horizon": 1,
                    "target_delay": delay,
                    "selected_target": selected,
                }
            )
    return rows


def test_core_metric_is_perfect_for_correct_physical_futures() -> None:
    result = summarize_h1_physical(
        _rows(),
        bootstrap_resamples=200,
        bootstrap_seed=7,
    )

    assert result["queries"] == 4
    assert result["history_conditions"] == 44
    assert result["physical_group_macro_accuracy"] == 1.0
    assert result["minimum_physical_group_accuracy"] == 1.0
    assert (
        result["paired_query_bootstrap_95_percent_interval"]["lower"]
        == 1.0
    )


def test_core_metric_weights_six_physical_groups_equally() -> None:
    result = summarize_h1_physical(
        _rows(fail_group_zero=True),
        bootstrap_resamples=200,
        bootstrap_seed=7,
    )

    assert result["by_physical_group"]["0"]["accuracy"] == 0.0
    assert result["by_physical_group"]["5"]["accuracy"] == 1.0
    assert result["physical_group_macro_accuracy"] == 5 / 6
    assert result["minimum_physical_group_accuracy"] == 0.0


def test_public_stable_worldmodel_adapters_use_history_seven() -> None:
    for adapter in (
        StableWorldModelLeWMHistory7Adapter,
        StableWorldModelPLDMHistory7Adapter,
    ):
        assert adapter.required_history_tokens == 7
        assert adapter.maximum_future_action_blocks == 3
