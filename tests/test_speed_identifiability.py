from __future__ import annotations

import numpy as np
import pytest

from contextworld.evaluation.speed_identifiability import (
    agent_centroid_from_rgb,
    alternating_impulse_blocks,
    estimate_speed_from_transitions,
    nearest_speed,
)


def test_alternating_impulse_blocks_are_paired() -> None:
    blocks = alternating_impulse_blocks(np.asarray([1.0, 0.5]), 8)
    assert blocks.shape == (8, 5, 2)
    assert np.array_equal(blocks[:, 1:], np.zeros((8, 4, 2)))
    assert np.allclose(blocks.sum(axis=(0, 1)), 0.0)
    with pytest.raises(ValueError, match="positive even"):
        alternating_impulse_blocks(np.asarray([1.0, 0.5]), 3)


def test_state_speed_estimator_recovers_scalar_dynamics() -> None:
    blocks = alternating_impulse_blocks(np.asarray([1.0, -0.5]), 4)
    actions = blocks.sum(axis=1)
    starts = np.asarray([[50.0, 60.0], [57.0, 56.5], [50.0, 60.0], [57.0, 56.5]])
    ends = starts + actions * 7.0
    estimate, per_transition = estimate_speed_from_transitions(starts, ends, blocks)
    assert estimate == pytest.approx(7.0)
    assert np.allclose(per_transition, 7.0)
    assert nearest_speed(estimate, [3.1, 5.0, 7.0]) == 7.0


def test_red_excess_centroid_uses_xy_coordinate_order() -> None:
    image = np.full((11, 13, 3), 255, dtype=np.uint8)
    image[7, 4] = np.asarray([255, 0, 0], dtype=np.uint8)
    centroid = agent_centroid_from_rgb(image)
    assert centroid.shape == (2,)
    assert np.allclose(centroid, [4.0, 7.0])
