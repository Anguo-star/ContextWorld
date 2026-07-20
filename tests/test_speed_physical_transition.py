from __future__ import annotations

import numpy as np
import torch

from scripts.eval_tworoom_speed_physical_transition import (
    HORIZONS,
    _free_motion_residual,
    _speed_grid,
    physical_action_probe,
    physical_metrics,
)


def test_physical_action_probes_are_bounded_and_cover_three_families() -> None:
    families = []
    for index in range(3):
        family, actions = physical_action_probe(
            query_state=np.asarray([195.0, 180.0], dtype=np.float32),
            evaluation_index=index,
        )
        families.append(family)
        assert actions.shape == (50, 2)
        assert np.max(np.abs(actions)) <= 1.0
        assert np.linalg.norm(actions.sum(axis=0)) > 0.0
    assert families == [
        "constant_direction",
        "varying_magnitude",
        "turning",
    ]


def test_physical_probe_points_into_the_query_room_interior() -> None:
    _, left = physical_action_probe(
        query_state=np.asarray([95.0, 112.0], dtype=np.float32),
        evaluation_index=0,
    )
    _, right = physical_action_probe(
        query_state=np.asarray([129.0, 112.0], dtype=np.float32),
        evaluation_index=0,
    )

    assert left[0, 0] < 0.0
    assert right[0, 0] > 0.0


def test_physical_metrics_recovers_exact_oracle_speed_and_position() -> None:
    speed_grid = np.asarray([3.4, 4.8, 6.9], dtype=np.float64)
    horizons = np.arange(1, 11, dtype=np.float32)
    oracle_embeddings = torch.stack(
        [
            torch.stack(
                [
                    torch.tensor([speed, speed * float(horizon)])
                    for horizon in horizons
                ]
            )
            for speed in speed_grid
        ]
    )
    oracle_states = np.stack(
        [
            np.stack(
                [
                    np.asarray(
                        [10.0 + speed * horizon, 20.0],
                        dtype=np.float32,
                    )
                    for horizon in horizons
                ]
            )
            for speed in speed_grid
        ]
    )

    exact = physical_metrics(
        predicted_embeddings=oracle_embeddings[1],
        oracle_embeddings=oracle_embeddings,
        oracle_states=oracle_states,
        speed_grid=speed_grid,
        query_speed=4.8,
        history_speed=4.8,
        query_state=np.asarray([10.0, 20.0], dtype=np.float32),
    )

    assert set(map(int, exact)) == set(HORIZONS)
    for row in exact.values():
        assert row["inferred_speed"] == 4.8
        assert row["position_error_px"] == 0.0
        assert row["latent_mse_to_true_query_future"] == 0.0

    mismatched = physical_metrics(
        predicted_embeddings=oracle_embeddings[0],
        oracle_embeddings=oracle_embeddings,
        oracle_states=oracle_states,
        speed_grid=speed_grid,
        query_speed=4.8,
        history_speed=3.4,
        query_state=np.asarray([10.0, 20.0], dtype=np.float32),
    )
    assert all(
        row["inferred_speed"] == 3.4 for row in mismatched.values()
    )
    assert all(
        row["position_error_px"] > 0.0 for row in mismatched.values()
    )


def test_oracle_speed_grid_is_exact_and_includes_decimal_speeds() -> None:
    grid = _speed_grid(2.5, 8.0, 0.05)
    assert len(grid) == 111
    assert 3.4 in grid
    assert 4.8 in grid
    assert 6.9 in grid


def test_free_motion_residual_is_zero_for_exact_speed_scaled_states() -> None:
    actions = np.repeat(
        np.asarray([[0.1, -0.05]], dtype=np.float32), 50, axis=0
    )
    speeds = np.asarray([3.0, 5.0, 7.0], dtype=np.float64)
    query = np.asarray([100.0, 120.0], dtype=np.float32)
    cumulative = np.cumsum(actions, axis=0)[4::5]
    states = (
        query[None, None]
        + speeds.astype(np.float32)[:, None, None]
        * cumulative[None]
    )

    assert (
        _free_motion_residual(
            query_state=query,
            raw_actions=actions,
            speed_grid=speeds,
            oracle_states=states,
        )
        == 0.0
    )
