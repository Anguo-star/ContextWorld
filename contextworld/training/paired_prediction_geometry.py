"""Task-independent losses for paired hidden-dynamics predictions."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch


@dataclass(frozen=True)
class PairedGeometryBarrierSpec:
    """Frozen definition of the boundary-aware paired-geometry loss."""

    history_margin: float = 0.20
    response_reference_ratio: float = 1.50
    center_weight: float = 1.0
    response_weight: float = 1.0
    center_loss_divisor: float = 1.0
    response_loss_divisor: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 < self.history_margin < 1.0:
            raise ValueError("history_margin must be in (0, 1)")
        if self.response_reference_ratio <= 1.0:
            raise ValueError("response_reference_ratio must exceed one")
        for name in (
            "center_weight",
            "response_weight",
            "center_loss_divisor",
            "response_loss_divisor",
        ):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be positive")

    def describe(self) -> dict[str, float]:
        return {
            "history_margin": self.history_margin,
            "response_reference_ratio": self.response_reference_ratio,
            "center_weight": self.center_weight,
            "response_weight": self.response_weight,
            "center_loss_divisor": self.center_loss_divisor,
            "response_loss_divisor": self.response_loss_divisor,
        }


def paired_prediction_geometry_terms(
    *,
    predicted_left: torch.Tensor,
    predicted_right: torch.Tensor,
    target_left: torch.Tensor,
    target_right: torch.Tensor,
    history_margin: float = 0.20,
    response_reference_ratio: float = 1.50,
) -> dict[str, torch.Tensor]:
    """Return exact paired-history margins and response-scale calibration.

    The two history margins are the same squared-distance comparisons used
    by the benchmark metric.  Requiring both to exceed ``history_margin``
    is therefore a direct barrier around the pair-center decision boundary.
    The response term is the squared log response ratio.  Its unique zero is
    the real response scale (ratio one), so increasing the predicted pair
    separation can never enter an unpenalized interval.
    """

    shapes = {
        tuple(value.shape)
        for value in (
            predicted_left,
            predicted_right,
            target_left,
            target_right,
        )
    }
    if len(shapes) != 1 or predicted_left.ndim != 2:
        raise ValueError(
            "All paired prediction tensors must share shape [pairs, latent]"
        )
    if not 0.0 < float(history_margin) < 1.0:
        raise ValueError("history_margin must be in (0, 1)")
    if float(response_reference_ratio) <= 1.0:
        raise ValueError("response_reference_ratio must exceed one")

    def row_mse(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return (left.float() - right.float()).square().mean(dim=-1)

    left_left = row_mse(predicted_left, target_left)
    left_right = row_mse(predicted_left, target_right)
    right_left = row_mse(predicted_right, target_left)
    right_right = row_mse(predicted_right, target_right)
    target_pair_scale = row_mse(target_left, target_right).detach().clamp_min(
        1.0e-8
    )
    history_left_margin = (right_left - left_left) / target_pair_scale
    history_right_margin = (left_right - right_right) / target_pair_scale
    history_margins = torch.cat(
        [history_left_margin, history_right_margin]
    )
    center_barrier = torch.relu(
        float(history_margin) - history_margins
    ).square().mean()

    predicted_pair_scale = row_mse(
        predicted_left,
        predicted_right,
    ).clamp_min(1.0e-12)
    response_ratio = (predicted_pair_scale / target_pair_scale).sqrt()
    response_calibration = response_ratio.log().square().mean()
    reference_log_distance = math.log(float(response_reference_ratio))

    return {
        "center_barrier_loss": center_barrier,
        "response_calibration_loss": response_calibration,
        "history_left_margin_mean": history_left_margin.detach().mean(),
        "history_right_margin_mean": history_right_margin.detach().mean(),
        "history_margin_pass_rate": (
            history_margins.detach() >= float(history_margin)
        )
        .float()
        .mean(),
        "response_ratio_mean": response_ratio.detach().mean(),
        "response_ratio_within_reference_range_rate": (
            (
                response_ratio.detach()
                >= 1.0 / float(response_reference_ratio)
            )
            & (
                response_ratio.detach()
                <= float(response_reference_ratio)
            )
        )
        .float()
        .mean(),
        "response_reference_log_distance": torch.as_tensor(
            reference_log_distance,
            dtype=response_ratio.dtype,
            device=response_ratio.device,
        ),
    }


def normalized_paired_prediction_geometry_loss(
    *,
    predicted_left: torch.Tensor,
    predicted_right: torch.Tensor,
    target_left: torch.Tensor,
    target_right: torch.Tensor,
    spec: PairedGeometryBarrierSpec,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return the frozen normalized barrier loss and auditable components."""

    terms = paired_prediction_geometry_terms(
        predicted_left=predicted_left,
        predicted_right=predicted_right,
        target_left=target_left,
        target_right=target_right,
        history_margin=spec.history_margin,
        response_reference_ratio=spec.response_reference_ratio,
    )
    normalized_center = (
        terms["center_barrier_loss"] / spec.center_loss_divisor
    )
    normalized_response = (
        terms["response_calibration_loss"] / spec.response_loss_divisor
    )
    loss = (
        spec.center_weight * normalized_center
        + spec.response_weight * normalized_response
    )
    components: dict[str, torch.Tensor] = {
        **terms,
        "center_barrier_normalized_loss": normalized_center,
        "response_calibration_normalized_loss": normalized_response,
        "paired_geometry_barrier_loss": loss,
    }
    return loss, components


def paired_batch_prediction_geometry_loss(
    *,
    embeddings: torch.Tensor,
    deterministic_prediction: torch.Tensor,
    pair_indices: torch.Tensor,
    spec: PairedGeometryBarrierSpec,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Apply the task-independent loss to paired final latent predictions."""

    if embeddings.ndim != 3 or deterministic_prediction.ndim != 3:
        raise ValueError("Expected embeddings and predictions [batch,time,latent]")
    if pair_indices.ndim != 2 or pair_indices.shape[1] != 2:
        raise ValueError("pair_indices must have shape [pairs,2]")
    left = pair_indices[:, 0]
    right = pair_indices[:, 1]
    return normalized_paired_prediction_geometry_loss(
        predicted_left=deterministic_prediction[left, 2],
        predicted_right=deterministic_prediction[right, 2],
        target_left=embeddings[left, 3],
        target_right=embeddings[right, 3],
        spec=spec,
    )


def scale_payload_to_spec(
    payload: dict[str, Any],
    *,
    center_weight: float,
) -> PairedGeometryBarrierSpec:
    """Construct a frozen barrier specification from a Training-only audit."""

    definition = payload["definition"]
    scales = payload["frozen_loss_divisors"]
    if payload.get("passed") is not True:
        raise ValueError("Training loss-scale audit did not pass")
    return PairedGeometryBarrierSpec(
        history_margin=float(definition["history_margin"]),
        response_reference_ratio=float(
            definition["response_reference_ratio"]
        ),
        center_weight=float(center_weight),
        response_weight=1.0,
        center_loss_divisor=float(scales["center_barrier_loss"]),
        response_loss_divisor=float(scales["response_calibration_loss"]),
    )
