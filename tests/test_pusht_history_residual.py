from __future__ import annotations

import torch

from contextworld.training.pusht_history_residual import (
    FrozenHistoryResidualHead,
    complete_pair_batch_stream,
    complete_twin_group_batch_stream,
    paired_center_response_loss,
    paired_prediction_metrics,
)


def test_residual_head_is_zero_initialized_and_uses_history_shape() -> None:
    head = FrozenHistoryResidualHead(
        latent_dim=8,
        action_dim=4,
        hidden_dim=16,
    )
    latents = torch.randn(5, 3, 8)
    actions = torch.randn(5, 3, 4)
    residual = head(latents, actions)
    assert residual.shape == (5, 8)
    assert torch.count_nonzero(residual) == 0
    assert head.trainable_parameter_count == sum(
        parameter.numel() for parameter in head.parameters()
    )


def test_pair_stream_never_splits_adjacent_conditions() -> None:
    rows = next(
        complete_pair_batch_stream(
            pair_count=8,
            rows_per_batch=8,
            seed=1,
        )
    ).reshape(-1, 2)
    assert torch.equal(rows[:, 1], rows[:, 0] + 1)
    assert torch.equal(rows[:, 0] % 2, torch.zeros(4, dtype=torch.long))


def test_twin_stream_never_splits_four_row_groups() -> None:
    rows = next(
        complete_twin_group_batch_stream(
            pair_count=8,
            rows_per_batch=8,
            seed=2,
        )
    ).reshape(-1, 4)
    assert torch.equal(rows[:, 1:], rows[:, :1] + torch.arange(1, 4))
    assert torch.equal(rows[:, 0] % 4, torch.zeros(2, dtype=torch.long))


def test_paired_loss_and_metrics_are_exact_for_real_futures() -> None:
    target = torch.tensor(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [1.0, 3.0],
            [1.0, 5.0],
        ]
    )
    pairs = torch.arange(4).reshape(-1, 2)
    loss, components = paired_center_response_loss(
        prediction=target,
        target=target,
        pair_indices=pairs,
    )
    assert loss == 0
    assert components["center_ratio_mean"] == 0
    assert components["response_ratio_mean"] == 0
    metrics = paired_prediction_metrics(prediction=target, target=target)
    assert metrics["correct_future_rate"] == 1.0
    assert metrics["correct_history_rate"] == 1.0
    assert metrics["context_switch_rate"] == 1.0
    assert metrics["worst_mode_correct_future_rate"] == 1.0

