"""Tests for the PushT contact-friction History-3 physical audit."""

from __future__ import annotations

import numpy as np
import pytest

from contextworld.evaluation.pusht_contact_friction_h3 import (
    AGENT_COLLISION_TYPE,
    BLOCK_COLLISION_TYPE,
    FRICTION_VALUES,
    evaluate_contact_friction_candidate,
    make_contact_friction_catalog_template,
    make_contact_friction_env,
    make_frozen_confirmation_templates,
    make_frozen_search_best_template,
    midpoint_snapshot,
    model_input_projection,
    simulate_contact_friction_clip,
    simulate_history,
    simulate_query_future,
    validate_contact_friction_pair,
)


@pytest.fixture(scope="module")
def frozen_audit():
    return evaluate_contact_friction_candidate(
        make_frozen_search_best_template(),
        resolution=64,
        require_primary_shape=True,
    )


def test_contact_friction_is_assigned_to_shapes_not_motion_damping():
    template = make_frozen_search_best_template()
    env, _ = make_contact_friction_env(
        template,
        mode="low_friction",
        resolution=32,
    )
    try:
        requested = FRICTION_VALUES["low_friction"]
        products = [
            agent_shape.friction * block_shape.friction
            for agent_shape in env.agent.shapes
            for block_shape in env.block.shapes
        ]
        assert products
        assert all(
            np.isclose(value, requested, atol=1e-12, rtol=0.0)
            for value in products
        )
        assert env.space.damping == 0.0
        assert {
            shape.collision_type for shape in env.agent.shapes
        } == {AGENT_COLLISION_TYPE}
        assert {
            shape.collision_type for shape in env.block.shapes
        } == {BLOCK_COLLISION_TYPE}
    finally:
        env.close()


def test_frozen_h3_search_result_fails_only_the_common_query_gate(
    frozen_audit,
):
    assert not frozen_audit["passed"]
    failed = {
        name
        for name, passed in frozen_audit["checks"].items()
        if not passed
    }
    assert failed == {"precanonical_pair_state_within_tolerance"}
    assert (
        frozen_audit["history_visible_response_gap"]["px_equivalent"]
        >= 3.0
    )
    assert (
        frozen_audit["precanonical_query_state"]["pair_max_abs_gap"]
        > 0.002
    )
    assert frozen_audit["future_gap"]["block_position_px"] >= 2.0
    assert frozen_audit["contact_steps"]["low_history"] > 0
    assert frozen_audit["contact_steps"]["high_history"] > 0
    assert frozen_audit["contact_steps"]["low_query"] > 0
    assert frozen_audit["contact_steps"]["high_query"] > 0


def test_history3_confirmations_pass_without_increasing_history_length():
    templates = make_frozen_confirmation_templates()
    audits = [
        evaluate_contact_friction_candidate(
            template,
            resolution=64,
            require_primary_shape=True,
            query_state_gate="per_endpoint_correction",
        )
        for template in templates
    ]

    assert len(templates) == 8
    assert len({template.template_id for template in templates}) == 8
    assert all(audit["passed"] for audit in audits)
    assert max(
        max(
            audit["precanonical_query_state"][
                "low_to_canonical_max_abs"
            ],
            audit["precanonical_query_state"][
                "high_to_canonical_max_abs"
            ],
        )
        for audit in audits
    ) <= 0.002
    assert min(
        audit["history_visible_response_gap"]["px_equivalent"]
        for audit in audits
    ) >= 3.0
    assert min(
        audit["future_gap"]["block_position_px"] for audit in audits
    ) >= 2.0


def test_formal_catalog_is_deterministic_and_split_specific():
    first = make_contact_friction_catalog_template(
        split="train",
        catalog_index=7,
        catalog_seed=20260801,
    )
    replay = make_contact_friction_catalog_template(
        split="train",
        catalog_index=7,
        catalog_seed=20260801,
    )
    validation = make_contact_friction_catalog_template(
        split="validation",
        catalog_index=7,
        catalog_seed=20260801,
    )

    assert first == replay
    assert first.template_id != validation.template_id
    assert first != validation
    assert first.canonical_query_snapshot is not None


def test_formal_clip_is_a_valid_history3_causal_pair():
    for catalog_index in range(32):
        template = make_contact_friction_catalog_template(
            split="loader_validation",
            catalog_index=catalog_index,
            catalog_seed=20260801,
        )
        low = simulate_contact_friction_clip(
            template,
            mode="low_friction",
            resolution=224,
        )
        high = simulate_contact_friction_clip(
            template,
            mode="high_friction",
            resolution=224,
        )
        audit = validate_contact_friction_pair(low, high)
        if audit["passed"]:
            break
    else:
        pytest.fail("No strict continuous catalog pair passed")

    assert audit["passed"]
    assert all(len(values) == 20 for values in low["rows"].values())
    assert low["model_pixels"].shape == (4, 224, 224, 3)
    assert low["action_blocks"].shape == (4, 5, 2)
    assert audit["state_installations_after_x0"] == 0
    assert not audit["query_simulator_recreated"]
    assert audit["query_physics_max_abs_gap"] <= 1.0e-5
    assert audit["query_pixel_max_abs_difference"] == 0
    assert audit["query_action_max_abs_difference"] == 0.0
    assert min(
        audit["trailing_no_contact_steps_before_query"].values()
    ) >= 3
    assert all(
        value["passed"]
        for value in audit["clean_simulator_replay"].values()
    )
    assert np.array_equal(low["model_pixels"][0], high["model_pixels"][0])
    assert np.array_equal(
        low["model_pixels"][2],
        high["model_pixels"][2],
    )
    assert not np.array_equal(
        low["model_pixels"][1],
        high["model_pixels"][1],
    )
    assert not np.array_equal(
        low["model_pixels"][3],
        high["model_pixels"][3],
    )


def test_model_projection_contains_only_pixels_and_actions():
    template = make_frozen_search_best_template()
    low = simulate_history(
        template,
        mode="low_friction",
        resolution=32,
    )
    high = simulate_history(
        template,
        mode="high_friction",
        resolution=32,
        render_pixels=False,
    )
    canonical = midpoint_snapshot(
        low["snapshots"][-1],
        high["snapshots"][-1],
    )
    future = simulate_query_future(
        template,
        mode="low_friction",
        canonical_query_snapshot=canonical,
        resolution=32,
    )
    projection = model_input_projection(low, future)

    assert set(projection) == {"pixels", "action"}
    assert projection["pixels"].shape == (3, 32, 32, 3)
    assert projection["pixels"].dtype == np.uint8
    assert projection["action"].shape == (3, 5, 2)
    assert projection["action"].dtype == np.float32
