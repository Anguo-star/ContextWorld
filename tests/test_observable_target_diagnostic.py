from __future__ import annotations

import numpy as np

from scripts import diagnose_observable_targets as diagnostic
from scripts import validate_physical_readout as readout


def _push_scene(index: int) -> readout.FeatureScene:
    conditions, candidates, horizons, dimensions = 2, 2, 2, 6
    physical = np.zeros((conditions, candidates, horizons, dimensions), dtype=np.float32)
    # Inclusive canvas boundaries are inside. The second time cell is
    # pointwise off-canvas for the pusher under only one hidden condition.
    physical[0, 0, 0, 0:2] = [0.0, 5.0]
    physical[1, 0, 0, 0:2] = [512.0, 5.0]
    physical[0, 0, 1, 0:2] = [-1.0, 5.0]
    physical[1, 0, 1, 0:2] = [5.0, 5.0]
    physical[0, 1, 0, 0:2] = [10.0, 10.0]
    physical[1, 1, 0, 0:2] = [11.0, 10.0]
    physical[0, 1, 1, 0:2] = [20.0, 20.0]
    physical[1, 1, 1, 0:2] = [21.0, 20.0]

    # A block center outside in one condition makes that whole C,T cell
    # ineligible for the paired in-canvas mask.
    physical[..., 2:4] = [30.0 + index, 40.0]
    physical[1, 1, 0, 2] = 512.01
    physical[..., 4] = np.asarray([0.1, 0.2], dtype=np.float32)[:, None, None]
    physical[..., 5] = np.asarray([0.9, 0.8], dtype=np.float32)[:, None, None]

    rng = np.random.default_rng(1000 + index)
    target = np.empty((conditions, candidates, horizons, 8), dtype=np.float32)
    target[..., :6] = physical
    target[..., 6:] = rng.normal(size=(conditions, candidates, horizons, 2))
    pred = target + np.float32(0.05) + rng.normal(
        scale=0.01, size=target.shape
    ).astype(np.float32)
    return readout.FeatureScene(
        scene_id=f"query-{index}",
        source_group=f"source-{index}",
        pred=pred,
        target=target,
        physical=physical,
    )


def _push_records() -> list[readout.FeatureScene]:
    return [_push_scene(index) for index in range(6)]


def test_push_canvas_masks_keep_complete_condition_pairs_and_report_coverage() -> None:
    physical = _push_scene(0).physical
    masks, receipt = diagnostic._paired_canvas_masks(physical)

    assert masks["paired_all_centers_in_canvas"][:, 0, 0].tolist() == [True, True]
    assert masks["paired_all_centers_in_canvas"][:, 0, 1].tolist() == [False, False]
    assert masks["paired_all_centers_in_canvas"][:, 1, 0].tolist() == [False, False]
    assert masks["paired_all_centers_in_canvas"][:, 1, 1].tolist() == [True, True]
    assert np.array_equal(
        masks["paired_all_centers_in_canvas"]
        | masks["paired_not_all_centers_in_canvas"],
        masks["all_rows"],
    )
    assert receipt["pointwise_centers"]["pusher_center_in_canvas_rows"] == 7
    assert receipt["pointwise_centers"]["block_center_in_canvas_rows"] == 7
    assert receipt["condition_paired_canvas"][
        "all_conditions_both_centers_in_canvas_cells"
    ] == 2
    assert receipt["condition_paired_canvas"][
        "all_conditions_not_both_centers_in_canvas_cells"
    ] == 2


def test_source_groups_do_not_leak_and_full_metrics_match_legacy_readout() -> None:
    records = _push_records()
    args = {
        "task": "action_strength",
        "n_folds": 3,
        "maximum_samples_per_source": 32,
        "ridge_alpha": readout.DEFAULT_RIDGE_ALPHA,
        "projection_dim": 8,
        "projection_seed": readout.DEFAULT_PROJECTION_SEED,
        "bootstrap_reps": 5,
        "seed": 20261007,
    }
    reference = readout.crossfit_physical_readout(records, **args)
    result = diagnostic.diagnose_records(
        records,
        reference_result=reference,
        **args,
    )

    for fold in result["folds"]:
        assert set(fold["train_groups"]).isdisjoint(fold["heldout_groups"])
        assert fold["readout_fit_uses"] == "true_future_target_latent_only"
    assert result["reference_check"]["passed_atol_1e-8"] is True
    assert result["reference_check"]["queries_compared"] == len(records)
    assert set(result["reference_check"]["aggregate_parity"]) == set(
        diagnostic.METRIC_BRANCHES
    )
    for parity in result["reference_check"]["aggregate_parity"].values():
        assert parity["reference_values_present"] is True
        assert parity["absolute_mse_delta"] <= 1e-8
        assert parity["absolute_rmse_delta"] <= 1e-8


def test_object_error_decomposition_conserves_full_and_masked_mse() -> None:
    result = diagnostic.diagnose_records(
        _push_records(),
        task="action_strength",
        n_folds=3,
        projection_dim=8,
        bootstrap_reps=5,
        seed=20261007,
    )
    object_dims = result["target"]["objects"]
    for row in result["query_metrics"]:
        for category in (
            "all_rows",
            "paired_all_centers_in_canvas",
            "paired_not_all_centers_in_canvas",
        ):
            for branch in diagnostic.METRIC_BRANCHES:
                metric = row["categories"][category][branch]
                if metric["query_mse"] is None:
                    continue
                dimension_count = sum(len(indices) for indices in object_dims.values())
                recomposed = sum(
                    len(indices) * metric["object_metrics"][name]["query_mse"]
                    for name, indices in object_dims.items()
                    if metric["object_metrics"][name]["query_mse"] is not None
                ) / dimension_count
                np.testing.assert_allclose(recomposed, metric["query_mse"], rtol=1e-12, atol=1e-12)

    contribution = result["mse_contribution_decomposition"]
    assert contribution["all_branches_conserve"] is True
    for branch in contribution["branches"].values():
        assert branch["conservation_passed_atol_1e-10"] is True
        np.testing.assert_allclose(
            branch["all_mse"],
            branch["in_canvas_contribution_mse"]
            + branch["not_all_in_canvas_contribution_mse"],
            rtol=0.0,
            atol=1e-10,
        )
        for obj in branch["objects"].values():
            if "residual_mse" in obj:
                assert abs(obj["residual_mse"]) <= 1e-10


def test_non_pusht_cached_decomposition_does_not_claim_visible_coverage() -> None:
    records = [
        readout.FeatureScene(
            scene_id=f"speed-{index}",
            source_group=f"episode-{index}",
            pred=scene.pred[..., :2],
            target=scene.target[..., :2],
            physical=scene.physical[..., :2],
        )
        for index, scene in enumerate(_push_records())
    ]
    reference = readout.crossfit_physical_readout(
        records,
        task="speed",
        projection_dim=2,
        bootstrap_reps=5,
        seed=20261007,
    )
    result = diagnostic.decompose_cached_reference(reference, task="speed")
    assert result["coverage"]["scored_full_rows"] is True
    assert result["coverage"]["visibility_mask_applied"] is False
    assert "not_assessed" in result["coverage"]["visibility_status"]
    assert result["normalization"]["per_coordinate_normalized_rmse"].startswith("unavailable")
    assert set(result["summaries"]["predicted_latent_matched"]["objects"]) == {"agent"}


def test_cached_parity_derives_legacy_mse_when_summary_stores_only_rmse() -> None:
    records = [
        readout.FeatureScene(
            scene_id=f"speed-{index}",
            source_group=f"episode-{index}",
            pred=scene.pred[..., :2],
            target=scene.target[..., :2],
            physical=scene.physical[..., :2],
        )
        for index, scene in enumerate(_push_records())
    ]
    reference = readout.crossfit_physical_readout(
        records,
        task="speed",
        projection_dim=2,
        bootstrap_reps=5,
        seed=20261007,
    )
    for summary in reference["summaries"].values():
        summary.pop("point_estimate_mse", None)

    result = diagnostic.decompose_cached_reference(reference, task="speed")

    parity = result["full_aggregate_parity"]
    assert parity["passed_atol_1e-8"] is True
    for branch in diagnostic.METRIC_BRANCHES:
        metrics = parity["branches"][branch]
        assert metrics["reference_values_present"] is True
        assert metrics["reference_mse_source"] == "derived_from_point_estimate_squared"
        np.testing.assert_allclose(
            metrics["reference_mse"], metrics["reference_rmse"] ** 2, rtol=0.0, atol=0.0
        )
        assert metrics["absolute_rmse_delta"] <= 1e-8
        assert metrics["max_abs_query_mse_delta"] <= 1e-8


def test_zero_variance_queries_keep_their_error_in_normalized_numerator() -> None:
    rows = [
        {
            "source_group": "zero-signal",
            "metric": {"query_mse": 100.0, "normalization_denominator": 0.0},
        },
        {
            "source_group": "positive-signal",
            "metric": {"query_mse": 0.0, "normalization_denominator": 1.0},
        },
    ]
    summary = diagnostic._summarize_metric_rows(rows, np.empty((0, 0), dtype=np.int64))

    assert summary["normalized_rmse"] == 10.0
    assert summary["n_queries_with_zero_condition_variance"] == 1
    assert summary["n_queries_missing_condition_variance"] == 0
