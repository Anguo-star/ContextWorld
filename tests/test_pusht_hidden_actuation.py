"""Tests for the condition-matched Push-T hidden-actuation construction."""

import inspect
from dataclasses import replace

import numpy as np

import contextworld.evaluation.pusht_hidden_actuation as hidden_actuation

from contextworld.evaluation.pusht_hidden_actuation import (
    HiddenActuationTemplate,
    model_input_projection,
    simulate_hidden_actuation,
    validate_hidden_actuation_pair,
)


def _template() -> HiddenActuationTemplate:
    return HiddenActuationTemplate(
        template_id='unit-right-v0',
        agent_position=(120.0, 350.0),
        block_position=(200.0, 350.0),
        block_angle=0.0,
        contact_direction=(1.0, 0.0),
        probe_sign=1,
        goal_agent_position=(180.0, 350.0),
        goal_block_position=(240.0, 350.0),
        goal_block_angle=0.0,
        simulator_seed=42,
    )


def test_hidden_actuation_pair_has_same_query_and_different_future():
    low = simulate_hidden_actuation(_template(), mode='low_gain', resolution=64)
    high = simulate_hidden_actuation(
        _template(),
        mode='high_gain',
        resolution=64,
    )

    audit = validate_hidden_actuation_pair(low, high)

    assert audit['passed'], audit
    assert (
        audit['query_physics_max_abs_gap']
        <= audit['query_physics_tolerance']
    )
    assert audit['state_installations_after_x0'] == 0
    assert audit['query_simulator_recreated'] is False
    assert audit['full_state_dimensions'] == 12
    assert audit['pair_query_pixel_difference'] == 0
    assert audit['pair_query_action_difference'] == 0.0
    assert low['state_installations_after_x0'] == 0
    assert high['state_installations_after_x0'] == 0
    np.testing.assert_allclose(
        low['query_natural_snapshot'],
        low['query_reference_snapshot'],
        atol=low['query_state_tolerance'],
        rtol=0.0,
    )
    assert audit['middle_agent_gap_px'] > 20.0
    assert audit['future_gap']['block_position_px'] > 20.0
    assert audit['query_contact_steps']['low_gain'] > 0
    assert audit['query_contact_steps']['high_gain'] > 0


def test_hidden_actuation_has_no_query_boundary_state_installation():
    source = inspect.getsource(hidden_actuation.simulate_hidden_actuation)

    assert '_restore_body_snapshot' not in source
    assert source.count('env.reset(') == 1


def test_model_projection_excludes_hidden_mode_and_physics_metadata():
    rollout = simulate_hidden_actuation(
        _template(),
        mode='low_gain',
        resolution=64,
    )

    projection = model_input_projection(rollout)

    assert set(projection) == {'pixels', 'action'}
    assert projection['pixels'].shape == (4, 64, 64, 3)
    assert projection['action'].shape == (4, 5, 2)
    assert projection['pixels'].dtype == np.uint8
    assert projection['action'].dtype == np.float32


def test_query_amplitude_changes_only_the_final_action_block_contract():
    base = _template()
    stronger = replace(base, query_amplitude=0.8)

    baseline = simulate_hidden_actuation(
        base,
        mode='low_gain',
        resolution=64,
    )
    changed = simulate_hidden_actuation(
        stronger,
        mode='low_gain',
        resolution=64,
    )

    np.testing.assert_array_equal(
        baseline['action_blocks'][:2],
        changed['action_blocks'][:2],
    )
    np.testing.assert_array_equal(
        baseline['model_pixels'][:3],
        changed['model_pixels'][:3],
    )
    np.testing.assert_allclose(
        changed['action_blocks'][2, :2],
        np.asarray([[0.8, 0.0], [0.8, 0.0]], dtype=np.float32),
    )
    assert not np.array_equal(
        baseline['model_pixels'][3],
        changed['model_pixels'][3],
    )
