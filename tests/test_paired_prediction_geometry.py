from __future__ import annotations

import json
import math

import pytest
import torch

from contextworld.benchmarks.contact_friction_icl_data import file_sha256
from contextworld.training.paired_prediction_geometry import (
    PairedGeometryBarrierSpec,
    normalized_paired_prediction_geometry_loss,
    paired_prediction_geometry_terms,
    scale_payload_to_spec,
)
from scripts import run_pusht_contact_friction_center_barrier as runner


def _pair(
    *,
    center: float = 0.0,
    response_ratio: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    target_left = torch.tensor([[-1.0]])
    target_right = torch.tensor([[1.0]])
    predicted_left = torch.tensor(
        [[center - response_ratio]],
        requires_grad=True,
    )
    predicted_right = torch.tensor(
        [[center + response_ratio]],
        requires_grad=True,
    )
    return predicted_left, predicted_right, target_left, target_right


def test_exact_pair_has_zero_barrier_and_zero_response_penalty() -> None:
    predicted_left, predicted_right, target_left, target_right = _pair()
    terms = paired_prediction_geometry_terms(
        predicted_left=predicted_left,
        predicted_right=predicted_right,
        target_left=target_left,
        target_right=target_right,
    )
    assert float(terms["center_barrier_loss"].detach()) == 0.0
    assert float(terms["response_calibration_loss"].detach()) == 0.0
    assert float(terms["history_margin_pass_rate"]) == 1.0
    assert float(terms["response_ratio_mean"]) == pytest.approx(1.0)


def test_center_barrier_is_the_exact_two_sided_history_boundary() -> None:
    values = _pair(center=1.2)
    terms = paired_prediction_geometry_terms(
        predicted_left=values[0],
        predicted_right=values[1],
        target_left=values[2],
        target_right=values[3],
        history_margin=0.20,
    )
    assert float(terms["history_left_margin_mean"]) == pytest.approx(2.2)
    assert float(terms["history_right_margin_mean"]) == pytest.approx(-0.2)
    assert float(terms["history_margin_pass_rate"]) == pytest.approx(0.5)
    assert float(terms["center_barrier_loss"].detach()) > 0.0


@pytest.mark.parametrize("response_ratio", [1.5, 6.8])
def test_response_calibration_penalizes_every_amplification(
    response_ratio: float,
) -> None:
    values = _pair(response_ratio=response_ratio)
    terms = paired_prediction_geometry_terms(
        predicted_left=values[0],
        predicted_right=values[1],
        target_left=values[2],
        target_right=values[3],
        response_reference_ratio=1.5,
    )
    assert float(terms["center_barrier_loss"].detach()) == 0.0
    assert float(terms["response_ratio_mean"]) == pytest.approx(
        response_ratio
    )
    assert float(terms["response_calibration_loss"].detach()) > 0.0


def _matching_plus_barrier_loss(response_ratio: float) -> float:
    embeddings = torch.zeros((2, 4, 1))
    embeddings[0, 3, 0] = -1.0
    embeddings[1, 3, 0] = 1.0
    prediction = torch.zeros((2, 3, 1))
    prediction[0, 2, 0] = -response_ratio
    prediction[1, 2, 0] = response_ratio
    pairs = torch.tensor([[0, 1]], dtype=torch.long)
    matching, _ = runner.mixed.paired_future_matching_loss(
        embeddings=embeddings,
        deterministic_prediction=prediction,
        pair_indices=pairs,
    )
    barrier, _ = normalized_paired_prediction_geometry_loss(
        predicted_left=prediction[0:1, 2],
        predicted_right=prediction[1:2, 2],
        target_left=embeddings[0:1, 3],
        target_right=embeddings[1:2, 3],
        spec=PairedGeometryBarrierSpec(
            response_loss_divisor=math.log(1.5) ** 2,
        ),
    )
    return float((matching + barrier).detach())


def test_total_loss_prefers_true_response_over_amplification() -> None:
    exact = _matching_plus_barrier_loss(1.0)
    moderate = _matching_plus_barrier_loss(1.5)
    extreme = _matching_plus_barrier_loss(6.8)
    assert exact < moderate < extreme


def test_degenerate_prediction_has_finite_loss_and_gradient() -> None:
    target_left = torch.tensor([[-1.0, 0.0]])
    target_right = torch.tensor([[1.0, 0.0]])
    predicted_left = torch.zeros((1, 2), requires_grad=True)
    predicted_right = torch.zeros((1, 2), requires_grad=True)
    spec = PairedGeometryBarrierSpec()
    loss, _ = normalized_paired_prediction_geometry_loss(
        predicted_left=predicted_left,
        predicted_right=predicted_right,
        target_left=target_left,
        target_right=target_right,
        spec=spec,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert predicted_left.grad is not None
    assert predicted_right.grad is not None
    assert torch.isfinite(predicted_left.grad).all()
    assert torch.isfinite(predicted_right.grad).all()


def test_scale_payload_freezes_exactly_two_center_strengths() -> None:
    payload = {
        "passed": True,
        "definition": {
            "history_margin": 0.20,
            "response_reference_ratio": 1.50,
        },
        "frozen_loss_divisors": {
            "center_barrier_loss": 0.25,
            "response_calibration_loss": 4.0,
        },
    }
    b1 = scale_payload_to_spec(payload, center_weight=1.0)
    b2 = scale_payload_to_spec(payload, center_weight=2.0)
    assert b1.center_loss_divisor == b2.center_loss_divisor == 0.25
    assert b1.response_loss_divisor == b2.response_loss_divisor == 4.0
    assert b2.center_weight == 2.0 * b1.center_weight
    assert set(runner.CANDIDATES) == {
        "mixed_frozen_image_history_center_barrier_b1",
        "mixed_frozen_image_history_center_barrier_b2",
    }


def test_runner_is_bound_to_unchanged_shared_training_code() -> None:
    shared = runner.SCRIPT_ROOT / "run_pusht_hidden_actuation_mixed.py"
    assert file_sha256(shared) == runner.SHARED_RUNNER_SHA256
    source = (
        runner.ROOT
        / "contextworld/training/paired_prediction_geometry.py"
    )
    assert source.is_file()
    assert json.loads(
        json.dumps({"candidate_count": len(runner.CANDIDATES)})
    ) == {"candidate_count": 2}
