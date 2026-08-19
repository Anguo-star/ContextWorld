"""Tests for replay-matched Push-T hidden-actuation pairs."""

import inspect

import numpy as np

import contextworld.evaluation.pusht_replay_matched_hidden_actuation as replay

from contextworld.evaluation.pusht_replay_matched_hidden_actuation import (
    ReplayMatchedHiddenActuationTemplate,
    fast_replay_matched_pair_audit,
    project_recovery_to_nullspace,
    replay_candidate_rows,
    rotate_action_block_to_direction,
    simulate_replay_matched_hidden_actuation,
    validate_replay_matched_pair,
)
from contextworld.evaluation.pusht_hidden_actuation import (
    PROBE_PROFILE,
    RECOVERY_PROFILE,
)


def _block(values):
    return tuple(tuple(map(float, row)) for row in values)


def _template():
    source_query = np.zeros((5, 2), dtype=np.float64)
    source_query[:2, 0] = 0.4
    probe = rotate_action_block_to_direction(
        source_query,
        (-1.0, 0.0),
    )
    recovery = project_recovery_to_nullspace(
        probe,
        np.zeros((5, 2), dtype=np.float64),
    )
    return ReplayMatchedHiddenActuationTemplate(
        template_id="replay-unit-v0",
        source_row_index=123,
        source_episode_index=7,
        source_step_index=11,
        agent_position=(120.0, 350.0),
        block_position=(200.0, 350.0),
        block_angle=0.0,
        goal_agent_position=(180.0, 350.0),
        goal_block_position=(240.0, 350.0),
        goal_block_angle=0.0,
        probe_actions=_block(probe),
        recovery_actions=_block(recovery),
        query_actions=_block(source_query),
        filler_actions=_block(np.zeros((5, 2))),
        simulator_seed=42,
    )


def test_nullspace_projection_reproduces_v1_minimum_norm_recovery():
    probe = np.zeros((5, 2), dtype=np.float64)
    probe[:, 0] = PROBE_PROFILE

    recovery = project_recovery_to_nullspace(
        probe,
        np.zeros((5, 2), dtype=np.float64),
    )

    np.testing.assert_allclose(
        recovery[:, 0],
        RECOVERY_PROFILE,
        atol=2e-7,
        rtol=0.0,
    )
    np.testing.assert_array_equal(recovery[:, 1], np.zeros(5))


def test_rotation_preserves_per_step_magnitudes_and_orients_mean():
    source = np.asarray(
        [
            [0.2, 0.0],
            [0.3, 0.1],
            [0.1, -0.1],
            [0.0, 0.0],
            [0.1, 0.0],
        ]
    )

    rotated = rotate_action_block_to_direction(source, (0.0, -1.0))

    np.testing.assert_allclose(
        np.linalg.norm(rotated, axis=1),
        np.linalg.norm(source, axis=1),
        atol=1e-12,
    )
    mean = rotated.mean(axis=0)
    np.testing.assert_allclose(
        mean / np.linalg.norm(mean),
        np.asarray([0.0, -1.0]),
        atol=1e-12,
    )


def test_replay_matched_pair_preserves_causal_query_contract():
    fast = fast_replay_matched_pair_audit(_template())
    assert fast["passed"], fast

    low = simulate_replay_matched_hidden_actuation(
        _template(),
        mode="low_gain",
        resolution=64,
    )
    high = simulate_replay_matched_hidden_actuation(
        _template(),
        mode="high_gain",
        resolution=64,
    )
    audit = validate_replay_matched_pair(low, high)

    assert audit["passed"], audit
    assert audit["history_contact_steps"] == {
        "low_gain": 0,
        "high_gain": 0,
    }
    assert audit["source"]["row_index"] == 123
    assert audit["state_installations_after_x0"] == 0
    assert audit["query_simulator_recreated"] is False
    assert audit["pair_query_pixel_difference"] == 0
    assert audit["pair_query_action_difference"] == 0.0
    assert (
        audit["query_physics_max_abs_gap"]
        <= audit["query_physics_tolerance"]
    )
    np.testing.assert_array_equal(
        low["action_blocks"][2],
        np.asarray(_template().query_actions, dtype=np.float32),
    )


def test_replay_simulator_has_no_query_boundary_state_installation():
    source = inspect.getsource(replay._simulate)

    assert '_restore_body_snapshot' not in source
    assert source.count('env.reset(') == 1


def test_replay_candidate_rows_keep_natural_contact_queries():
    states = np.zeros((40, 7), dtype=np.float32)
    states[:, :2] = (120.0, 350.0)
    states[:, 2:4] = (200.0, 350.0)
    states[:, 4] = 0.3
    states[-1, 2:4] = (240.0, 350.0)
    states[5, 2] = 202.0
    actions = np.zeros((40, 2), dtype=np.float32)
    actions[:5, 0] = 0.2

    rows = replay_candidate_rows(
        states,
        actions,
        np.asarray([0]),
        np.asarray([40]),
        [0],
    )

    assert rows[0] == 0


def test_replay_candidate_rows_reject_static_query_blocks():
    states = np.zeros((40, 7), dtype=np.float32)
    states[:, :2] = (120.0, 350.0)
    states[:, 2:4] = (200.0, 350.0)
    states[-1, 2:4] = (240.0, 350.0)
    actions = np.zeros((40, 2), dtype=np.float32)
    actions[:5, 0] = 0.2

    rows = replay_candidate_rows(
        states,
        actions,
        np.asarray([0]),
        np.asarray([40]),
        [0],
    )

    assert 0 not in rows
