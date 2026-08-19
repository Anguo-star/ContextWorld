from __future__ import annotations

import pytest
import torch

from scripts import run_pusht_hidden_actuation_mixed as mixed
from contextworld.training.pusht_history_residual import (
    FrozenHistoryResidualHead,
    HistoryResidualLeWM,
)


def _inputs(
    *, center_offset: float = 0.0, response_scale: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    embeddings = torch.zeros(2, 4, 2)
    embeddings[0, 3] = torch.tensor([-1.0, 0.0])
    embeddings[1, 3] = torch.tensor([1.0, 0.0])
    prediction = torch.zeros(2, 3, 2, requires_grad=True)
    prediction.data[0, 2] = torch.tensor(
        [center_offset - response_scale, 0.0]
    )
    prediction.data[1, 2] = torch.tensor(
        [center_offset + response_scale, 0.0]
    )
    pairs = torch.tensor([[0, 1]], dtype=torch.long)
    return embeddings, prediction, pairs


def test_projected_geometry_is_calibrated_for_matching_pair() -> None:
    embeddings, prediction, pairs = _inputs()
    loss, components = mixed.paired_future_projected_geometry_loss(
        embeddings=embeddings,
        deterministic_prediction=prediction,
        pair_indices=pairs,
        include_projected_center=True,
        include_response_log_norm=True,
    )

    assert float(
        components["projected_center_loss"].detach()
    ) == pytest.approx(0.0)
    assert float(
        components["response_log_norm_loss"].detach()
    ) == pytest.approx(0.0)
    assert float(
        components["response_rms_ratio_mean"].detach()
    ) == pytest.approx(1.0)
    loss.backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


def test_projected_center_penalizes_common_pair_translation_only() -> None:
    embeddings, prediction, pairs = _inputs(center_offset=2.0)
    _, components = mixed.paired_future_projected_geometry_loss(
        embeddings=embeddings,
        deterministic_prediction=prediction,
        pair_indices=pairs,
        include_projected_center=True,
        include_response_log_norm=True,
    )

    assert float(components["projected_center_loss"].detach()) > 0.0
    assert float(
        components["projected_center_abs_mean"].detach()
    ) == pytest.approx(2.0)
    assert float(
        components["response_log_norm_loss"].detach()
    ) == pytest.approx(0.0)


def test_response_log_norm_penalizes_scale_without_center_fit() -> None:
    embeddings, prediction, pairs = _inputs(response_scale=3.0)
    _, components = mixed.paired_future_projected_geometry_loss(
        embeddings=embeddings,
        deterministic_prediction=prediction,
        pair_indices=pairs,
        include_projected_center=True,
        include_response_log_norm=True,
    )

    assert float(
        components["projected_center_loss"].detach()
    ) == pytest.approx(0.0)
    assert float(
        components["response_rms_ratio_mean"].detach()
    ) == pytest.approx(3.0)
    assert float(components["response_log_norm_loss"].detach()) > 0.0


def test_projected_geometry_requires_a_scalar_calibration_term() -> None:
    embeddings, prediction, pairs = _inputs()
    with pytest.raises(ValueError, match="At least one"):
        mixed.paired_future_projected_geometry_loss(
            embeddings=embeddings,
            deterministic_prediction=prediction,
            pair_indices=pairs,
            include_projected_center=False,
            include_response_log_norm=False,
        )


@pytest.mark.parametrize(
    "population",
    [
        "paired_future_projected_center",
        "paired_future_response_log_norm",
        "paired_future_projected_geometry",
    ],
)
def test_projected_geometry_supervises_only_identifiable_hidden_future(
    population: str,
) -> None:
    embeddings = torch.zeros(4, 4, 1)
    prediction = torch.ones(4, 3, 1, requires_grad=True)
    loss = mixed.mixed_prediction_loss(
        prediction=prediction,
        embeddings=embeddings,
        original_batch_size=2,
        conditional_population=population,
    )
    loss.backward()

    assert torch.all(prediction.grad[:2] != 0)
    assert torch.all(prediction.grad[2:, :2] == 0)
    assert torch.all(prediction.grad[2:, 2] != 0)


class _ZeroPredictor(torch.nn.Module):
    num_frames = 3

    def forward(
        self, embedding: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        return torch.zeros_like(embedding)


def test_history_residual_lewm_changes_only_full_history_final_output() -> None:
    head = FrozenHistoryResidualHead(
        latent_dim=2,
        action_dim=2,
        hidden_dim=4,
    )
    final = head.network[-1]
    assert isinstance(final, torch.nn.Linear)
    final.bias.data.fill_(1.0)
    model = HistoryResidualLeWM(
        encoder=torch.nn.Linear(2, 2),
        predictor=_ZeroPredictor(),
        action_encoder=torch.nn.Identity(),
        projector=torch.nn.Identity(),
        pred_proj=torch.nn.Identity(),
        history_residual=head,
    )
    history = torch.zeros(2, 3, 2)
    actions = torch.zeros(2, 3, 2)
    short_history = torch.zeros(2, 2, 2)
    short_actions = torch.zeros(2, 2, 2)

    full = model.predict(history, actions)
    short = model.predict(short_history, short_actions)

    torch.testing.assert_close(full[:, :2], torch.zeros(2, 2, 2))
    torch.testing.assert_close(full[:, 2], torch.ones(2, 2))
    torch.testing.assert_close(short, torch.zeros(2, 2, 2))
    assert "history_residual.network.1.weight" in model.state_dict()
