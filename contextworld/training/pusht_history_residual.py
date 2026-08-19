"""Low-capacity History-3 residual head for frozen PushT world models."""

from __future__ import annotations

from collections.abc import Iterator

import torch
from torch import nn

from stable_worldmodel.wm.lewm.lewm import LeWM


class FrozenHistoryResidualHead(nn.Module):
    """Correct a frozen next-latent prediction using visible history only.

    The feature contains the current latent, two temporal latent differences,
    and the three corresponding action embeddings.  The final layer is zero
    initialized, so attaching the head initially preserves the base model
    exactly.
    """

    def __init__(
        self,
        *,
        latent_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        if latent_dim <= 0 or action_dim <= 0 or hidden_dim <= 0:
            raise ValueError("All dimensions must be positive")
        self.latent_dim = int(latent_dim)
        self.action_dim = int(action_dim)
        self.history_tokens = 3
        feature_dim = 3 * self.latent_dim + 3 * self.action_dim
        self.network = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.latent_dim),
        )
        final = self.network[-1]
        if not isinstance(final, nn.Linear):
            raise RuntimeError("Unexpected residual-head final module")
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def features(
        self, latents: torch.Tensor, action_embeddings: torch.Tensor
    ) -> torch.Tensor:
        if latents.ndim != 3 or action_embeddings.ndim != 3:
            raise ValueError("Latents and actions must have shape (B,3,D)")
        if latents.shape[:2] != (latents.shape[0], self.history_tokens):
            raise ValueError("Exactly three latent history tokens are required")
        if action_embeddings.shape[:2] != (
            latents.shape[0],
            self.history_tokens,
        ):
            raise ValueError("Action history must match latent batch and time")
        if latents.shape[-1] != self.latent_dim:
            raise ValueError("Unexpected latent dimension")
        if action_embeddings.shape[-1] != self.action_dim:
            raise ValueError("Unexpected action-embedding dimension")
        z0, z1, z2 = latents.unbind(dim=1)
        return torch.cat(
            [
                z2,
                z2 - z1,
                z1 - z0,
                action_embeddings.flatten(start_dim=1),
            ],
            dim=-1,
        )

    def forward(
        self, latents: torch.Tensor, action_embeddings: torch.Tensor
    ) -> torch.Tensor:
        return self.network(self.features(latents, action_embeddings))

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class HistoryResidualLeWM(LeWM):
    """LeWM with a visible-history residual on each full History-3 window.

    The class keeps the native Stable-WorldModel module names and checkpoint
    format.  Consequently the same offline adapter and CEM planner can load
    and execute it; the only extra state is ``history_residual``.
    """

    def __init__(self, *, history_residual: nn.Module, **kwargs) -> None:
        super().__init__(**kwargs)
        self.history_residual = history_residual
        expected = getattr(self.predictor, "num_frames", None)
        if expected is not None and int(expected) != 3:
            raise ValueError("HistoryResidualLeWM requires History=3")

    def predict(
        self,
        emb: torch.Tensor,
        act_emb: torch.Tensor,
    ) -> torch.Tensor:
        prediction = super().predict(emb, act_emb)
        if emb.size(1) != 3:
            return prediction
        correction = self.history_residual(emb, act_emb)
        corrected_final = prediction[:, -1] + correction
        if prediction.size(1) == 1:
            return corrected_final[:, None]
        return torch.cat(
            [prediction[:, :-1], corrected_final[:, None]],
            dim=1,
        )


def paired_center_response_loss(
    *,
    prediction: torch.Tensor,
    target: torch.Tensor,
    pair_indices: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Fit pair centers and response vectors in target-pair units."""

    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError("Prediction and target must have shape (B,D)")
    if pair_indices.ndim != 2 or pair_indices.shape[1] != 2:
        raise ValueError("pair_indices must have shape (P,2)")
    if pair_indices.dtype != torch.long:
        raise TypeError("pair_indices must use torch.long")
    left, right = pair_indices.unbind(dim=1)
    predicted_left, predicted_right = prediction[left], prediction[right]
    target_left, target_right = target[left], target[right]
    target_delta = target_right - target_left
    prediction_delta = predicted_right - predicted_left
    target_center = 0.5 * (target_left + target_right)
    prediction_center = 0.5 * (predicted_left + predicted_right)
    scale = target_delta.square().mean(dim=-1).detach().clamp_min(1e-8)
    center_ratio = (
        (prediction_center - target_center).square().mean(dim=-1) / scale
    )
    response_ratio = (
        (prediction_delta - target_delta).square().mean(dim=-1) / scale
    )
    center_loss = torch.log1p(center_ratio).mean()
    response_loss = torch.log1p(response_ratio).mean()
    loss = center_loss + response_loss
    return loss, {
        "center_loss": center_loss,
        "response_loss": response_loss,
        "center_ratio_mean": center_ratio.mean(),
        "response_ratio_mean": response_ratio.mean(),
    }


def complete_pair_batch_stream(
    *,
    pair_count: int,
    rows_per_batch: int,
    seed: int,
) -> Iterator[torch.Tensor]:
    """Yield adjacent two-condition pairs with no pair split across batches."""

    if pair_count <= 0 or rows_per_batch <= 0 or rows_per_batch % 2:
        raise ValueError("Invalid pair stream dimensions")
    pairs_per_batch = rows_per_batch // 2
    if pair_count % pairs_per_batch:
        raise ValueError("pair_count must divide evenly by pairs_per_batch")
    generator = torch.Generator().manual_seed(seed)
    while True:
        order = torch.randperm(pair_count, generator=generator)
        for start in range(0, pair_count, pairs_per_batch):
            selected = order[start : start + pairs_per_batch]
            yield torch.stack([2 * selected, 2 * selected + 1], dim=1).flatten()


def complete_twin_group_batch_stream(
    *,
    pair_count: int,
    rows_per_batch: int,
    seed: int,
) -> Iterator[torch.Tensor]:
    """Yield four-row forward/reverse groups with both conditions present."""

    if pair_count <= 0 or pair_count % 2:
        raise ValueError("Twin-group data require an even pair count")
    if rows_per_batch <= 0 or rows_per_batch % 4:
        raise ValueError("Twin-group batches require a multiple of four rows")
    group_count = pair_count // 2
    groups_per_batch = rows_per_batch // 4
    if group_count % groups_per_batch:
        raise ValueError("group_count must divide evenly by groups_per_batch")
    generator = torch.Generator().manual_seed(seed)
    offsets = torch.arange(4)
    while True:
        order = torch.randperm(group_count, generator=generator)
        for start in range(0, group_count, groups_per_batch):
            selected = order[start : start + groups_per_batch]
            yield (4 * selected[:, None] + offsets[None]).flatten()


def paired_prediction_metrics(
    *, prediction: torch.Tensor, target: torch.Tensor
) -> dict[str, float]:
    """Return the four public paired-prediction metrics for adjacent rows."""

    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError("Prediction and target must have shape (2P,D)")
    if prediction.shape[0] % 2:
        raise ValueError("An even number of adjacent condition rows is required")
    predicted_low, predicted_high = prediction[0::2], prediction[1::2]
    target_low, target_high = target[0::2], target[1::2]

    def mse(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        return (first - second).square().mean(dim=-1)

    low_low = mse(predicted_low, target_low)
    low_high = mse(predicted_low, target_high)
    high_low = mse(predicted_high, target_low)
    high_high = mse(predicted_high, target_high)
    low_future = low_low < low_high
    high_future = high_high < high_low
    low_history = low_low < high_low
    high_history = high_high < low_high
    switch = (
        (predicted_high - predicted_low) * (target_high - target_low)
    ).sum(dim=-1) > 0
    return {
        "pair_count": float(predicted_low.shape[0]),
        "correct_future_rate": float(
            torch.cat([low_future, high_future]).float().mean()
        ),
        "correct_history_rate": float(
            torch.cat([low_history, high_history]).float().mean()
        ),
        "context_switch_rate": float(switch.float().mean()),
        "low_correct_future_rate": float(low_future.float().mean()),
        "high_correct_future_rate": float(high_future.float().mean()),
        "worst_mode_correct_future_rate": float(
            min(low_future.float().mean(), high_future.float().mean())
        ),
        "correct_future_mse_mean": float(
            torch.cat([low_low, high_high]).mean()
        ),
        "other_future_mse_mean": float(
            torch.cat([low_high, high_low]).mean()
        ),
        "target_pair_mse_mean": float(mse(target_low, target_high).mean()),
        "prediction_pair_mse_mean": float(
            mse(predicted_low, predicted_high).mean()
        ),
    }


__all__ = [
    "FrozenHistoryResidualHead",
    "complete_pair_batch_stream",
    "complete_twin_group_batch_stream",
    "paired_center_response_loss",
    "paired_prediction_metrics",
]
