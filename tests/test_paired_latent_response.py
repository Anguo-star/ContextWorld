from __future__ import annotations

import numpy as np
import pytest

from contextworld.benchmarks.contact_friction_icl_score import (
    contact_friction_prediction_gate,
    contact_friction_prediction_metrics,
)
from contextworld.benchmarks.contact_friction_icl_data import (
    load_contact_friction_icl_release,
)
from contextworld.benchmarks.paired_latent_response import (
    TargetLatentSeparationError,
    paired_latent_response_gate_checks,
    paired_latent_response_metrics,
    summarize_paired_latent_response_records,
)


PAIR_IDS = ("pair-0", "pair-1")
TARGET_FIRST = np.asarray([[0.0, 0.0], [1.0, -1.0]])
TARGET_SECOND = np.asarray([[1.0, 0.0], [1.0, 1.0]])


def response_metrics(
    predicted_first: np.ndarray,
    predicted_second: np.ndarray,
) -> dict[str, object]:
    metrics, records = paired_latent_response_metrics(
        pair_ids=PAIR_IDS,
        predicted_first=predicted_first,
        predicted_second=predicted_second,
        target_first=TARGET_FIRST,
        target_second=TARGET_SECOND,
    )
    assert summarize_paired_latent_response_records(records) == metrics
    return metrics


def test_perfect_response_has_ideal_dimensionless_metrics() -> None:
    metrics = response_metrics(TARGET_FIRST, TARGET_SECOND)
    assert metrics["response_gain"] == pytest.approx(1.0)
    assert metrics["response_alignment"] == pytest.approx(1.0)
    assert metrics["normalized_response_error"] == pytest.approx(0.0)
    assert metrics["calibrated_response_success_rate"] == 1.0
    assert metrics["target_latent_separation"]["passed"] is True


def test_no_history_constant_does_not_look_like_a_learned_response() -> None:
    common = 0.5 * (TARGET_FIRST + TARGET_SECOND)
    metrics = response_metrics(common, common)
    assert metrics["response_gain"] == pytest.approx(0.0)
    assert metrics["response_alignment"] == pytest.approx(0.0)
    assert metrics["normalized_response_error"] == pytest.approx(1.0)
    assert metrics["calibrated_response_success_rate"] == 0.0


def test_wrong_direction_response_is_explicitly_negative() -> None:
    metrics = response_metrics(TARGET_SECOND, TARGET_FIRST)
    assert metrics["response_gain"] == pytest.approx(-1.0)
    assert metrics["response_alignment"] == pytest.approx(-1.0)
    assert metrics["normalized_response_error"] == pytest.approx(4.0)
    assert metrics["calibrated_response_success_rate"] == 0.0


def test_far_but_correctly_ordered_predictions_fail_amplitude_metric() -> None:
    target_first = np.asarray([[0.0], [0.0]])
    target_second = np.asarray([[1.0], [1.0]])
    predicted_first = np.asarray([[-100.0], [-100.0]])
    predicted_second = np.asarray([[101.0], [101.0]])

    ordering, _ = contact_friction_prediction_metrics(
        pair_ids=PAIR_IDS,
        predicted_low=predicted_first,
        predicted_high=predicted_second,
        target_low=target_first,
        target_high=target_second,
    )
    response = ordering["latent_response"]
    assert ordering["correct_future_rate"] == 1.0
    assert ordering["correct_history_rate"] == 1.0
    assert ordering["context_switch_rate"] == 1.0
    assert ordering["joint_icl_pair_success_rate"] == 0.0
    assert response["response_gain"] == pytest.approx(201.0)
    assert response["response_alignment"] == pytest.approx(1.0)
    assert response["normalized_response_error"] == pytest.approx(40_000.0)
    assert response["calibrated_response_success_rate"] == 0.0
    gate = contact_friction_prediction_gate(
        ordering, release=load_contact_friction_icl_release()
    )
    assert gate["passed"] is False
    assert gate["checks"]["normalized_response_error"] is False


def test_internal_response_gate_fails_closed_and_uses_theory_anchors() -> None:
    thresholds = {
        "target_latent_separation_required": True,
        "response_gain_minimum": 0.5,
        "normalized_response_error_strict_maximum": 1.0,
    }
    assert paired_latent_response_gate_checks(
        {}, thresholds=thresholds
    ) == {
        "target_latent_separation": False,
        "response_gain": False,
        "normalized_response_error": False,
    }
    checks = paired_latent_response_gate_checks(
        {
            "latent_response": {
                "target_latent_separation": {"passed": True},
                "response_gain": 0.5,
                "normalized_response_error": np.nextafter(1.0, 0.0),
            }
        },
        thresholds=thresholds,
    )
    assert all(checks.values())
    assert paired_latent_response_gate_checks(
        {
            "latent_response": {
                "target_latent_separation": {"passed": True},
                "response_gain": np.nextafter(0.5, 0.0),
                "normalized_response_error": 0.0,
            }
        },
        thresholds=thresholds,
    )["response_gain"] is False
    assert paired_latent_response_gate_checks(
        {
            "latent_response": {
                "target_latent_separation": {"passed": True},
                "response_gain": 1.0,
                "normalized_response_error": 1.0,
            }
        },
        thresholds=thresholds,
    )["normalized_response_error"] is False


def test_identical_real_future_latents_are_rejected() -> None:
    target = np.zeros((2, 3), dtype=np.float32)
    with pytest.raises(
        TargetLatentSeparationError,
        match="identical for 2 pair",
    ):
        paired_latent_response_metrics(
            pair_ids=PAIR_IDS,
            predicted_first=target,
            predicted_second=target,
            target_first=target,
            target_second=target,
        )
