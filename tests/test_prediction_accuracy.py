import numpy as np
import pytest

from scripts.prediction_accuracy import (
    PositionGroup,
    clustered_paired_bootstrap_ci,
    geometry_position_errors,
    position_errors,
    score_errors,
)
from scripts.validate_physical_readout import physical_targets_from_panel


def test_score_perfect_predictions_are_one_hundred_and_failure_is_zero():
    thresholds = [0.5, 1.0, 2.0]
    assert score_errors(np.zeros((2, 3, 4)), thresholds).score == 100.0
    assert score_errors(np.full((2, 3, 4), 2.0), thresholds).score == 0.0


def test_score_uses_strict_tolerance_boundary_and_increases_with_error_tolerance():
    errors = np.array([[[0.99, 1.0, 1.01]]])
    result = score_errors(errors, [1.0, 1.01])

    assert result.tolerance_curve.tolist() == pytest.approx([100 / 3, 200 / 3])
    assert result.score == pytest.approx(50.0)
    assert result.tolerance_curve[1] > result.tolerance_curve[0]


def test_score_exposes_horizon_curve_max_horizon_and_raw_rmse():
    result = score_errors(np.array([[[0.0, 1.0], [1.0, 2.0]]]), [1.0, 2.0])

    assert result.horizon_curve.shape == (2,)
    assert result.tolerance_horizon_curve.shape == (2, 2)
    assert result.max_horizon_score == result.horizon_curve[-1]
    assert result.max_horizon_tolerance_curve.tolist() == pytest.approx(
        result.tolerance_horizon_curve[-1]
    )
    assert result.raw_rmse == pytest.approx(np.sqrt(1.5))


def test_score_macro_averages_scenes_before_aggregating_candidates():
    # Scene 0 has one accurate candidate among three, while scene 1 has three
    # inaccurate candidates. Each scene gets half the final weight despite
    # having a different number of candidates.
    errors = [np.array([[0.0], [2.0], [2.0]]), np.array([[2.0]])]
    result = score_errors(errors, [1.0])

    assert result.score == pytest.approx(100.0 / 6.0)


def test_matched_and_wrong_contexts_use_the_same_metric_for_same_predictions():
    target = np.zeros((2, 2, 3, 2))
    prediction = np.ones_like(target) * 0.25
    matched = position_errors(target, prediction, {"body": PositionGroup((0, 1))})
    wrong_context = position_errors(target, prediction, {"body": PositionGroup((0, 1))})
    matched_result = score_errors(matched["body"], [0.5, 1.0])
    wrong_result = score_errors(wrong_context["body"], [0.5, 1.0])

    assert matched_result.score == wrong_result.score
    assert matched_result.score - wrong_result.score == 0.0


def test_pusht_block_uses_sine_cosine_geometry_so_angle_wrap_is_small():
    theta_target = np.pi - 0.01
    theta_prediction = -np.pi + 0.01
    target = np.zeros((1, 1, 1, 6))
    prediction = np.zeros_like(target)
    target[..., 4] = 40.0 * np.sin(theta_target)
    target[..., 5] = 40.0 * np.cos(theta_target)
    prediction[..., 4] = 40.0 * np.sin(theta_prediction)
    prediction[..., 5] = 40.0 * np.cos(theta_prediction)

    errors = geometry_position_errors(target, prediction, "pusht")

    assert errors["agent"].item() == 0.0
    assert errors["block"].item() == pytest.approx(80 * np.sin(0.01))
    assert errors["block"].item() < 1.0


def test_geometry_groups_read_mass_and_cube_positions_as_canonical_millimetres():
    target = np.zeros((1, 1, 1, 6))
    prediction = np.zeros_like(target)
    prediction[..., 0] = 3.0
    prediction[..., 3] = 4.0

    mass = geometry_position_errors(target[..., :2], prediction[..., :2], "mass")
    cube = geometry_position_errors(target, prediction, "cube")

    assert mass["object"].item() == pytest.approx(3.0)
    assert cube["effector"].item() == pytest.approx(3.0)
    assert cube["cube"].item() == pytest.approx(4.0)


@pytest.mark.parametrize("state_width", [7, 12])
def test_pusht_geometry_integrates_with_canonical_physical_panel_mapper(state_width):
    target_state = np.zeros((1, 1, 2, state_width))
    prediction_state = np.zeros_like(target_state)
    prediction_state[..., 0] = 1.0
    if state_width == 7:
        prediction_state[..., 2] = 3.0
        prediction_state[..., 4] = 0.1
        task = "action_strength"
    else:
        prediction_state[..., 6] = 3.0
        prediction_state[..., 10] = 0.1
        task = "contact_friction"

    target = physical_targets_from_panel({"future_states": target_state}, task)
    prediction = physical_targets_from_panel({"future_states": prediction_state}, task)
    errors = geometry_position_errors(target, prediction, "pusht")

    assert errors["agent"].ravel().tolist() == pytest.approx([1.0, 1.0])
    expected_block_error = np.sqrt(
        3.0**2 + (40.0 * np.sin(0.1)) ** 2 + (40.0 * (np.cos(0.1) - 1.0)) ** 2
    )
    assert errors["block"].ravel().tolist() == pytest.approx(
        [expected_block_error, expected_block_error]
    )


def test_custom_position_groups_keep_object_errors_separate():
    target = np.zeros((1, 1, 1, 4))
    prediction = np.array([[[[3.0, 4.0, 5.0, 12.0]]]])

    errors = position_errors(
        target,
        prediction,
        {"effector": [0, 1], "object": slice(2, 4)},
    )

    assert set(errors) == {"effector", "object"}
    assert errors["effector"].item() == pytest.approx(5.0)
    assert errors["object"].item() == pytest.approx(13.0)


@pytest.mark.parametrize(
    "errors, thresholds",
    [
        (np.array([[[np.nan]]]), [1.0]),
        (np.array([[[-1.0]]]), [1.0]),
        (np.zeros((1, 1)), [1.0]),
        (np.zeros((1, 1, 1)), [1.0, 1.0]),
        (np.zeros((1, 1, 1)), [0.0]),
    ],
)
def test_score_rejects_nonfinite_negative_or_malformed_inputs(errors, thresholds):
    with pytest.raises(ValueError):
        score_errors(errors, thresholds)


def test_position_errors_reject_mismatched_counts_and_nonfinite_values():
    groups = {"object": PositionGroup((0, 1))}
    with pytest.raises(ValueError, match="same scene"):
        position_errors(np.zeros((1, 1, 1, 2)), np.zeros((2, 1, 1, 2)), groups)
    with pytest.raises(ValueError, match="finite"):
        position_errors(
            np.zeros((1, 1, 1, 2)),
            np.array([[[[np.inf, 0.0]]]]),
            groups,
        )


def test_paired_cluster_bootstrap_resamples_sources_but_weights_scenes_equally():
    # Source A has three scenes and source B has one. The estimate is the
    # equal-scene mean (5), while a resampled A+B replicate is also scene
    # weighted (5), not the equal-cluster mean (0).
    scores_a = np.array([10.0, 10.0, 10.0, -10.0])
    scores_b = np.zeros_like(scores_a)
    result = clustered_paired_bootstrap_ci(
        scores_a,
        scores_b,
        ["A", "A", "A", "B"],
        n_bootstrap=1000,
        seed=4,
    )

    assert result.estimate == pytest.approx(5.0)
    assert result.mean_a == pytest.approx(5.0)
    assert result.mean_b == pytest.approx(0.0)
    assert result.delta == pytest.approx(5.0)
    assert result.to_dict()["delta"] == pytest.approx(5.0)
    assert result.bootstrap_estimates.shape == (1000,)
    assert set(np.unique(result.bootstrap_estimates)) == {-10.0, 5.0, 10.0}
    assert result.lower <= result.estimate <= result.upper


def test_bootstrap_rejects_unpaired_scene_counts():
    with pytest.raises(ValueError, match="same non-empty scene axis"):
        clustered_paired_bootstrap_ci([1.0], [1.0, 2.0], ["a"])
