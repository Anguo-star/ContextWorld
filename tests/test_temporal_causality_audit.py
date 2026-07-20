from __future__ import annotations

import torch

from scripts.audit_tworoom_temporal_causality import (
    temporal_causality_probe,
)


class CausalToyPredictor(torch.nn.Module):
    input_dim = 4

    def forward(
        self, observations: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        return torch.cumsum(observations + actions, dim=1)


class FullToyPredictor(torch.nn.Module):
    input_dim = 4

    def forward(
        self, observations: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        total = (observations + actions).sum(dim=1, keepdim=True)
        return total.expand_as(observations)


class ConstantToyPredictor(torch.nn.Module):
    input_dim = 4

    def forward(
        self, observations: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        return torch.zeros_like(observations)


def test_temporal_causality_probe_accepts_causal_predictor() -> None:
    result = temporal_causality_probe(
        CausalToyPredictor(),
        seed=3,
        trials=2,
        sequence_length=3,
        tolerance=1.0e-6,
    )

    assert result["passed"]
    assert result["maximum_change_at_or_before_boundary"] == 0.0
    assert result["future_perturbation_changed_a_future_output"]


def test_temporal_causality_probe_rejects_full_predictor() -> None:
    result = temporal_causality_probe(
        FullToyPredictor(),
        seed=3,
        trials=2,
        sequence_length=3,
        tolerance=1.0e-6,
    )

    assert not result["passed"]
    assert result["maximum_change_at_or_before_boundary"] > 0.0


def test_temporal_causality_probe_rejects_constant_predictor() -> None:
    result = temporal_causality_probe(
        ConstantToyPredictor(),
        seed=3,
        trials=2,
        sequence_length=3,
        tolerance=1.0e-6,
    )

    assert not result["passed"]
    assert result["maximum_change_at_or_before_boundary"] == 0.0
    assert not result["future_perturbation_changed_a_future_output"]
