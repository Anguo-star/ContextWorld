from __future__ import annotations

import numpy as np

from contextworld.evaluation.portal_exit_h3 import (
    make_template,
    simulate_portal_exit_clip,
    simulate_portal_exit_episode,
    validate_portal_exit_pair,
    validate_portal_exit_episode_pair,
)


def test_portal_exit_history3_pair_is_physically_identifiable() -> None:
    for index in range(32):
        template = make_template(split="test", index=index, catalog_seed=20260801)
        near = simulate_portal_exit_clip(template, mode="near_border")
        farther = simulate_portal_exit_clip(template, mode="farther_from_border")
        audit = validate_portal_exit_pair(near, farther)
        assert audit["passed"], (index, audit)


def test_portal_exit_replay_is_deterministic() -> None:
    template = make_template(split="test", index=7, catalog_seed=20260801)
    first = simulate_portal_exit_clip(template, mode="farther_from_border")
    second = simulate_portal_exit_clip(template, mode="farther_from_border")
    for key in (
        "history_pixels",
        "history_states",
        "history_actions",
        "query_pixels",
        "query_state",
        "query_action",
        "future_pixels",
        "future_state",
    ):
        assert np.array_equal(first[key], second[key]), key


def test_formal_twenty_row_episode_preserves_the_pair_contract() -> None:
    template = make_template(split="test", index=19, catalog_seed=20260801)
    near = simulate_portal_exit_episode(template, mode="near_border")
    farther = simulate_portal_exit_episode(template, mode="farther_from_border")
    assert len(near["rows"]["pixels"]) == 20
    assert validate_portal_exit_episode_pair(near, farther)["passed"]
