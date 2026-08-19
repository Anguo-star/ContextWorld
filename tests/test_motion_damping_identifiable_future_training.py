from __future__ import annotations

import torch

from scripts import run_pusht_hidden_actuation_mixed as mixed
from scripts import run_pusht_motion_damping_h3_train as damping_train


def test_motion_damping_variants_are_component_local_and_history_identifiable() -> None:
    # Registration occurs in the component entry point rather than changing
    # the defaults of any other PushT capability.
    assert damping_train.LEWM_REFERENCE_VARIANT not in mixed.VARIANT_WEIGHTS
    assert damping_train.PLDM_REFERENCE_VARIANT not in mixed.VARIANT_WEIGHTS


def test_identifiable_future_mask_keeps_all_standard_transitions() -> None:
    embeddings = torch.zeros(4, 4, 1)
    prediction = torch.ones(4, 3, 1, requires_grad=True)
    loss = mixed.mixed_prediction_loss(
        prediction=prediction,
        embeddings=embeddings,
        original_batch_size=2,
        conditional_population="identifiable_future_only",
    )
    loss.backward()
    assert torch.all(prediction.grad[:2] != 0)


def test_identifiable_future_mask_uses_only_final_hidden_transition() -> None:
    embeddings = torch.zeros(4, 4, 1)
    prediction = torch.ones(4, 3, 1, requires_grad=True)
    loss = mixed.mixed_prediction_loss(
        prediction=prediction,
        embeddings=embeddings,
        original_batch_size=2,
        conditional_population="identifiable_future_only",
    )
    loss.backward()
    assert torch.all(prediction.grad[2:, :2] == 0)
    assert torch.all(prediction.grad[2:, 2] != 0)


def test_default_prediction_loss_still_uses_every_transition() -> None:
    embeddings = torch.zeros(4, 4, 1)
    prediction = torch.ones(4, 3, 1, requires_grad=True)
    loss = mixed.mixed_prediction_loss(
        prediction=prediction,
        embeddings=embeddings,
        original_batch_size=2,
        conditional_population="marginal",
    )
    loss.backward()
    assert torch.all(prediction.grad != 0)
