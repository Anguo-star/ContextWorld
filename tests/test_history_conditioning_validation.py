from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import validate_history_conditioning as validation


def test_cross_target_distances_keep_all_ordered_mismatches_equal_weight() -> None:
    # Two conditions, one candidate, one horizon.  The diagonal is zero and
    # both ordered mismatches have energy four.
    target = np.asarray([[[[0.0]]], [[[2.0]]]], dtype=np.float32)
    distance = np.asarray([[[[0.0]], [[4.0]]], [[[4.0]], [[0.0]]]], dtype=np.float64)
    summary = validation.summarize_cross_target_distances(
        distance,
        target,
        horizons=[5],
        task="synthetic",
        data={"conditions": np.asarray([0, 1])},
    )
    assert summary["matched"]["energy_sum"] == [0.0]
    assert summary["matched"]["energy_mean"] == [0.0]
    assert summary["wrong_allother"]["energy_sum"] == [8.0]
    assert summary["wrong_allother"]["energy_mean"] == [4.0]
    assert summary["history_conditioning_gain"]["wrong_minus_matched_energy_mean"] == [4.0]


def test_oracle_condition_mean_counterfactual_is_target_separation() -> None:
    target = np.asarray(
        [
            [[[0.0], [0.0]]],
            [[[2.0], [4.0]]],
        ],
        dtype=np.float32,
    )
    distance = np.zeros((2, 2, 1, 2), dtype=np.float64)
    summary = validation.summarize_cross_target_distances(
        distance,
        target,
        horizons=[5, 10],
        task="synthetic",
        data={"conditions": np.asarray([0, 1])},
    )
    # The condition mean is [1,0] at h1 and [1,2] at h2.
    assert summary["oracle_condition_mean_counterfactual"]["energy_sum"] == [2.0, 8.0]
    assert summary["target_separation_energy"] == [2.0, 8.0]


def test_action_delay_physical_mask_keeps_h1_stationary_group_secondary() -> None:
    conditions = np.arange(11)
    target = np.zeros((11, 1, 2, 1), dtype=np.float32)
    distance = np.ones((11, 11, 1, 2), dtype=np.float64)
    np.einsum("iict->ict", distance)[:] = 0.0
    summary = validation.summarize_cross_target_distances(
        distance,
        target,
        horizons=[5, 10],
        task="action_delay",
        data={"conditions": conditions},
    )
    mask = summary["physical_group_mask"]
    assert mask["available"] is True
    assert mask["distinguishable_pair_count_by_horizon"] == [80, 110]
    assert summary["wrong_allother"]["pair_count"] == 110
    assert mask["wrong_physically_distinguishable"]["pair_count_by_horizon"] == [80, 110]


def test_physical_mask_mean_weights_pairs_and_candidates_once() -> None:
    # Six ordered mismatches and two candidates contain twelve scalar entries,
    # but the reported pair count stays six and the per-entry mean stays one.
    distance = np.ones((3, 3, 2, 1), dtype=np.float64)
    mask = ~np.eye(3, dtype=bool)[:, :, None]
    sums, means, counts = validation._energy_by_mask(distance, mask)
    assert sums == [12.0]
    assert means == [1.0]
    assert counts == [6]


def test_dinowm_compression_uses_row_major_16x16_patch_grid() -> None:
    values = np.zeros((1, 1, 1, 16 * 16 * 384), dtype=np.float32)
    grid = values.reshape(1, 1, 1, 16, 16, 384)
    for row in range(16):
        for col in range(16):
            grid[0, 0, 0, row, col, :] = row * 100.0 + col
    compressed = validation.compress_features(values, "dinowm")
    assert compressed.shape == (1, 1, 1, 6144)
    pooled = compressed.reshape(1, 1, 1, 4, 4, 384)
    assert pooled[0, 0, 0, 0, 0, 0] == pytest.approx(151.5)
    assert pooled[0, 0, 0, 3, 3, 0] == pytest.approx(1363.5)


def test_conditioning_audit_exposes_action_delay_queue_length_confound() -> None:
    history = np.zeros((11, 3, 1, 1, 3), dtype=np.uint8)
    history[:, 0, 0, 0, 0] = np.arange(11)
    history[:, 1:, 0, 0, 0] = 9
    pending = np.zeros((11, 10, 2), dtype=np.float32)
    lengths = np.arange(11, dtype=np.int64)
    data = {
        "history_pixels": history,
        "context_actions": np.zeros((11, 2, 5, 2), dtype=np.float32),
        "conditions": np.arange(11),
        "query_state": np.asarray([1.0, 2.0]),
        "aux_pending_actions_at_query": pending,
        "aux_pending_action_lengths": lengths,
    }
    audit = validation.conditioning_audit("action_delay", data)
    assert audit["current_image_bitwise_equal"] is True
    assert audit["context_actions_equal"] is True
    assert audit["pending_queue_values_equal"] is True
    assert audit["pending_queue_lengths_equal"] is False
    assert audit["same_current_image_different_pending_queue"] is True


def test_pairwise_distance_uses_float64_and_direct_matching_diagonal() -> None:
    predicted = np.asarray([[[[1e6, 1.0]]], [[[1e6, -1.0]]]], dtype=np.float32)
    target = np.asarray([[[[1e6, 0.0]]], [[[1e6, 0.0]]]], dtype=np.float32)
    distance = validation._pairwise_squared_distance(predicted, target)
    assert distance.dtype == np.float64
    assert distance.shape == (2, 2, 1, 1)
    assert distance[0, 0, 0, 0] == pytest.approx(1.0)
    assert distance[1, 1, 0, 0] == pytest.approx(1.0)
