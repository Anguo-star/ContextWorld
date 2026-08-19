from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from contextworld.evaluation.action_delay_h7_validation import (
    ARRAY_KEYS,
    DELAYS,
    EVAL_SEEDS,
    QUERY_COUNT,
    QUERIES_PER_SEED,
    build_validation_asset,
    select_validation_assignments,
)
from contextworld.synthesis.config import load_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/benchmark/tworoom_action_delay_h7_v1.yaml"


def test_h7_validation_selects_300_distinct_balanced_queries() -> None:
    assignments = select_validation_assignments(load_config(CONFIG))
    assert len(assignments) == QUERY_COUNT == 300
    assert len({value.query_id for value in assignments}) == QUERY_COUNT
    assert (
        len({value.template.reset_state for value in assignments})
        == QUERY_COUNT
    )
    assert (
        len({value.template.simulator_seed for value in assignments})
        == QUERY_COUNT
    )
    counts = Counter(value.eval_seed for value in assignments)
    assert counts == Counter(
        {seed: QUERIES_PER_SEED for seed in EVAL_SEEDS}
    )
    for seed in EVAL_SEEDS:
        subset = [value for value in assignments if value.eval_seed == seed]
        assert Counter(value.room for value in subset) == Counter(
            {"left": 25, "right": 25}
        )
        assert Counter(value.template.direction for value in subset) == Counter(
            {"up": 25, "down": 25}
        )


def test_h7_validation_asset_has_exact_physical_groups() -> None:
    config = load_config(CONFIG)
    assignment = select_validation_assignments(config)[0]
    arrays, audit = build_validation_asset(
        assignment,
        agent_speed=float(config["environment"]["agent_speed"]),
        action_magnitude=float(
            config["history_protocol"]["action_magnitude"]
        ),
    )
    assert audit["physical"]["passed"]
    assert audit["physical"]["future_state_group_counts"] == {
        "1": 6,
        "2": 11,
        "3": 11,
    }
    assert audit["physical"]["future_pixel_group_counts"] == {
        "1": 6,
        "2": 11,
        "3": 11,
    }
    assert tuple(sorted(arrays)) == tuple(sorted(ARRAY_KEYS))
    assert arrays["history_pixels"].shape == (11, 7, 224, 224, 3)
    assert arrays["action_blocks"].shape == (9, 5, 2)
    assert arrays["true_future_pixels"].shape == (11, 3, 224, 224, 3)
    np.testing.assert_array_equal(arrays["history_delays"], DELAYS)
    np.testing.assert_array_equal(arrays["target_delays"], DELAYS)
    np.testing.assert_array_equal(
        arrays["audit_pending_action_lengths"],
        DELAYS,
    )
    assert np.count_nonzero(
        arrays["audit_pending_actions_at_query"]
    ) == 0
    np.testing.assert_array_equal(
        arrays["history_pixels"][:, -1],
        np.repeat(arrays["query_pixels"][None], len(DELAYS), axis=0),
    )


def test_horizon1_equivalence_is_not_mislabeled_exact_delay() -> None:
    config = load_config(CONFIG)
    arrays, _ = build_validation_asset(
        select_validation_assignments(config)[1],
        agent_speed=float(config["environment"]["agent_speed"]),
        action_magnitude=float(
            config["history_protocol"]["action_magnitude"]
        ),
    )
    h1 = arrays["true_future_pixels"][:, 0]
    assert len({value.tobytes() for value in h1}) == 6
    assert len({value.tobytes() for value in h1[5:]}) == 1
    h2 = arrays["true_future_pixels"][:, 1]
    h3 = arrays["true_future_pixels"][:, 2]
    assert len({value.tobytes() for value in h2}) == 11
    assert len({value.tobytes() for value in h3}) == 11
