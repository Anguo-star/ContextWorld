from __future__ import annotations

from collections import Counter

from contextworld.evaluation.pusht_motion_damping_h3 import (
    ENDPOINT_MODES,
    evaluate_template,
    make_confirmation_templates,
    simulate_motion_damping_clip,
    validate_motion_damping_pair,
)


def test_eight_history3_motion_damping_templates_pass() -> None:
    templates = make_confirmation_templates()
    assert len(templates) == 8
    results = [evaluate_template(template, resolution=96) for template in templates]
    assert all(result["passed"] for result in results)
    assert min(
        result["history_visible_response_gap"]["px_equivalent"]
        for result in results
    ) >= 3.0
    assert max(result["max_pair_full_state_gap"] for result in results) <= 1e-8
    assert max(
        value
        for result in results
        for value in result["query_reference_deviation"].values()
    ) <= 1e-8
    assert all(result["state_installations_after_x0"] == 0 for result in results)
    assert all(not result["query_simulator_recreated"] for result in results)
    assert all(
        result["maximum_arbiter_count_from_x0_through_x3"] == 0
        for result in results
    )
    assert Counter(
        result["hashes"]["faster_decay_initial_pixels"] for result in results
    ) == Counter(
        result["hashes"]["no_extra_decay_initial_pixels"] for result in results
    )
    assert min(
        result["future_gap"]["block_position_px"] for result in results
    ) >= 2.0


def test_formal_clip_keeps_query_exactly_paired() -> None:
    template = make_confirmation_templates()[0]
    faster = simulate_motion_damping_clip(
        template, mode=ENDPOINT_MODES[0], resolution=96
    )
    no_extra = simulate_motion_damping_clip(
        template, mode=ENDPOINT_MODES[1], resolution=96
    )
    audit = validate_motion_damping_pair(faster, no_extra)
    assert audit["passed"]
    assert audit["max_pair_full_state_gap"] <= 1e-8
    assert audit["max_pair_query_pixel_difference"] == 0
    assert audit["max_pair_query_action_difference"] == 0.0
    assert audit["state_installations_after_x0"] == 0
    assert not audit["query_simulator_recreated"]
    assert audit["maximum_arbiter_count_from_x0_through_x3"] == 0
    assert faster["query_boundary"] == (
        "single_continuous_simulator_from_x0_through_x3"
    )
    assert faster["model_pixels"].shape == (4, 96, 96, 3)
    assert len(faster["rows"]["pixels"]) == 20
