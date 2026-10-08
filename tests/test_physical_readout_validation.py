from __future__ import annotations

import json
import shutil

import numpy as np

from scripts.validate_physical_readout import (
    FeatureScene,
    FixedRandomProjector,
    align_physical_targets,
    crossfit_physical_readout,
    fit_ridge_readout,
    load_feature_scenes,
    physical_targets_from_panel,
    upgrade_physical_readout_result,
)


def _synthetic_scene(index: int, *, conditions: int = 3, candidates: int = 2) -> FeatureScene:
    physical = np.zeros((conditions, candidates, 2, 2), dtype=np.float32)
    physical[..., 0] = float(index) + np.arange(conditions, dtype=np.float32)[:, None, None]
    physical[..., 1] = np.arange(candidates, dtype=np.float32)[None, :, None]
    target = np.zeros((conditions, candidates, 2, 4), dtype=np.float32)
    target[..., 0] = physical[..., 0]
    target[..., 1] = physical[..., 1]
    target[..., 2] = 1.0
    target[..., 3] = float(index)
    pred = target.copy()
    pred[..., 0] += 0.05
    return FeatureScene(
        scene_id=f"scene-{index}",
        source_group=f"episode:{index}",
        pred=pred,
        target=target,
        physical=physical,
    )


def test_pusht_and_condition_first_axes_are_explicit() -> None:
    states = np.zeros((3, 2, 1, 7), dtype=np.float64)
    states[..., :4] = np.arange(24, dtype=np.float64).reshape(3, 2, 1, 4)
    states[..., 4] = np.pi / 2.0
    states[..., 5:] = 99.0  # excluded public state columns
    physical = physical_targets_from_panel({"future_states": states}, "action_strength")

    assert physical.shape == (3, 2, 1, 6)
    np.testing.assert_allclose(physical[..., :4], states[..., :4])
    np.testing.assert_allclose(physical[..., 4], 40.0)
    np.testing.assert_allclose(physical[..., 5], 0.0, atol=1e-12)

    aligned = align_physical_targets(physical, (3, 2, 1, 8))
    # K=3 and C=2 deliberately catch an accidental candidate/condition swap.
    np.testing.assert_array_equal(aligned, physical)


def test_mass_and_cube_targets_exclude_hidden_velocity_like_columns() -> None:
    finger = np.arange(3 * 2 * 1 * 2, dtype=np.float64).reshape(3, 2, 1, 2)
    mass = physical_targets_from_panel({"finger_positions": finger}, "robot_arm_mass")
    np.testing.assert_allclose(mass, finger * 1000.0)

    cube_state = np.zeros((3, 2, 1, 7), dtype=np.float64)
    cube_state[..., 0:2] = [1.0, 2.0]
    cube_state[..., 2:5] = [3.0, 4.0, 5.0]
    cube_state[..., 5] = 123.0  # gripper/opening-like value; must be excluded
    cube_state[..., 6] = 6.0
    cube = physical_targets_from_panel({"future_states": cube_state}, "cube")
    expected = np.broadcast_to(
        np.asarray([1.0, 2.0, 6.0, 3.0, 4.0, 5.0]) * 1000.0,
        cube.shape,
    )
    np.testing.assert_allclose(cube, expected)


def test_crossfit_keeps_source_groups_out_of_readout_fit() -> None:
    records = [_synthetic_scene(index) for index in range(6)]
    result = crossfit_physical_readout(
        records,
        task="speed",
        bootstrap_reps=5,
        seed=7,
    )

    for fold in result["folds"]:
        assert set(fold["train_groups"]).isdisjoint(fold["heldout_groups"])
        assert fold["readout_fit_uses"] == "true_future_target_latent_only"
        assert all(count <= 32 for count in fold["sampled_rows_by_train_group"].values())
    assert result["source_group_crossfit"]["readout_fit_excludes_heldout_source"] is True
    assert result["summaries"]["oracle_true_latent_floor_rmse"]["n_source_groups"] == 6

    first = result["query_metrics"][0]
    assert first["oracle_true_latent"]["physical_variance_across_conditions"] > 0.0
    # Condition-axis mismatch is explicitly different from the matching pair.
    assert first["predicted_latent_mismatched"]["rmse"] > first["predicted_latent_matched"]["rmse"]

    calibration = result["paired_calibration_gap"]
    history = result["paired_history_gain"]
    assert "paired_gain" not in result
    assert calibration["raw_rmse"]["left"] == "predicted_latent_matched"
    assert calibration["raw_rmse"]["right"] == "oracle_true_latent"
    assert history["raw_rmse"]["left"] == "predicted_latent_mismatched"
    assert history["raw_rmse"]["right"] == "predicted_latent_matched"
    assert history["normalized_rmse"]["left"] == "predicted_latent_mismatched"
    assert history["normalized_rmse"]["right"] == "predicted_latent_matched"
    assert calibration["raw_rmse"]["same_source_bootstrap_draws_for_left_and_right"] is True
    assert history["raw_rmse"]["same_source_bootstrap_draws_for_left_and_right"] is True
    assert history["raw_rmse"]["bootstrap_replicates"] == 5
    assert history["normalized_rmse"]["bootstrap_replicates"] == 5
    assert result["metric_units"] == {
        "physical_mse": "px^2",
        "physical_rmse": "px",
        "normalized_mse": "dimensionless",
        "normalized_rmse": "dimensionless",
    }


def test_upgrade_old_query_metrics_adds_semantic_pair_fields_without_refit() -> None:
    # The varying normalization denominators make sure normalized MSE is
    # summed before the square root (and is not a mean of query differences).
    old = {
        "schema_version": 1,
        "target": {"units": "px"},
        "paired_gain": {"raw_rmse": {"bootstrap_replicates": 7}},
        "query_metrics": [
            {
                "scene_id": "scene-a",
                "source_group": "source-a",
                "oracle_true_latent": {"query_mse": 1.0, "normalization_denominator": 1.0},
                "predicted_latent_matched": {"query_mse": 4.0, "normalization_denominator": 1.0},
                "predicted_latent_mismatched": {"query_mse": 9.0, "normalization_denominator": 1.0},
            },
            {
                "scene_id": "scene-b",
                "source_group": "source-b",
                "oracle_true_latent": {"query_mse": 9.0, "normalization_denominator": 9.0},
                "predicted_latent_matched": {"query_mse": 16.0, "normalization_denominator": 9.0},
                "predicted_latent_mismatched": {"query_mse": 25.0, "normalization_denominator": 9.0},
            },
        ],
    }
    old_query_metrics = old["query_metrics"]
    upgraded = upgrade_physical_readout_result(old, bootstrap_reps=0)

    assert upgraded["schema_version"] == 2
    assert "paired_gain" not in upgraded
    assert upgraded["compatibility_upgrade"]["source"] == "query_metrics_only"
    assert upgraded["compatibility_upgrade"]["readout_refit"] is False
    assert upgraded["query_metrics"] == old_query_metrics

    calibration = upgraded["paired_calibration_gap"]
    history = upgraded["paired_history_gain"]
    assert calibration["raw_rmse"]["left_minus_right_mse"] == 5.0
    assert history["raw_rmse"]["left_minus_right_mse"] == 7.0
    # (4+16)/(1+9) - (1+9)/(1+9) = 1, while the old per-query raw-MSE
    # difference would have been 5.  History is (9+25)/10 - (4+16)/10 = 1.4.
    assert calibration["normalized_rmse"]["left_minus_right_mse"] == 1.0
    assert history["normalized_rmse"]["left_minus_right_mse"] == 1.4
    assert calibration["raw_rmse"]["bootstrap_replicates"] == 0
    assert history["normalized_rmse"]["bootstrap_replicates"] == 0
    assert upgraded["metric_units"]["normalized_mse"] == "dimensionless"


def test_zero_condition_variance_is_not_hidden_by_normalization() -> None:
    record = _synthetic_scene(0)
    physical = record.physical.copy()
    physical[:] = physical[0:1]  # no across-condition physical variation
    records = [
        FeatureScene(
            scene_id=f"scene-{i}",
            source_group=f"episode:{i}",
            pred=record.pred.copy(),
            target=record.target.copy(),
            physical=physical.copy(),
        )
        for i in range(6)
    ]
    result = crossfit_physical_readout(records, task="speed", bootstrap_reps=0)
    row = result["query_metrics"][0]["oracle_true_latent"]
    assert row["normalized_denominator_zero"] is True
    assert row["normalized_rmse"] is None
    summary = result["summaries"]["oracle_true_latent_floor_normalized_rmse"]
    assert summary["point_estimate"] is None
    assert summary["numerator_sum_query_mse"] > 0.0
    assert summary["denominator_sum"] == 0.0


def test_wide_features_use_fixed_sparse_projection_and_float64_fit() -> None:
    projector = FixedRandomProjector(6144, 512, seed=3)
    features = np.zeros((2, 1, 1, 6144), dtype=np.float32)
    features[..., 17] = 1.0
    projected = projector.transform(features)
    assert projected.shape == (2, 512)
    assert projected.dtype == np.float64
    assert projector.projected is True
    model = fit_ridge_readout(
        np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
        np.asarray([[0.0], [1.0]], dtype=np.float32),
    )
    assert model["coefficient"].dtype == np.float64


def test_single_condition_has_no_cross_scene_mismatch_fallback() -> None:
    base = _synthetic_scene(0, conditions=1)
    records = [
        FeatureScene(
            scene_id=f"scene-{i}",
            source_group=f"episode:{i}",
            pred=base.pred,
            target=base.target,
            physical=base.physical,
        )
        for i in range(6)
    ]
    result = crossfit_physical_readout(records, task="speed", bootstrap_reps=0)
    mismatch = result["query_metrics"][0]["predicted_latent_mismatched"]
    assert mismatch["undefined_condition_mismatch"] is True
    assert mismatch["query_mse"] is None
    assert result["summaries"]["predicted_latent_mismatched_rmse"]["n_queries"] == 0


def test_loader_keeps_condition_first_and_preprojects_per_file(tmp_path) -> None:
    panels = tmp_path / "panels" / "speed"
    features = tmp_path / "features"
    panels.mkdir(parents=True)
    features.mkdir()
    scene_id = "query-000"
    panel_path = panels / "panel-000.npz"
    states = np.arange(3 * 2 * 1 * 2, dtype=np.float32).reshape(3, 2, 1, 2)
    np.savez(panel_path, future_states=states)
    (panels / "manifest.json").write_text(
        json.dumps(
            {
                "scenes": [
                    {
                        "scene_id": scene_id,
                        "path": panel_path.name,
                        "bootstrap_cluster": "episode:heldout",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    latent = np.arange(3 * 2 * 1 * 4, dtype=np.float32).reshape(3, 2, 1, 4)
    np.savez(features / f"features_{scene_id}.npz", pred=latent, target=latent)
    shards = features / "shards" / "0"
    shards.mkdir(parents=True)
    shutil.copy2(features / f"features_{scene_id}.npz", shards / f"features_{scene_id}.npz")

    records = load_feature_scenes(features, tmp_path / "panels", "speed", projection_dim=2)
    assert len(records) == 1
    assert records[0].pred.dtype == np.float32
    assert records[0].pred.shape == (3, 2, 1, 2)
    assert records[0].original_latent_dim == 4
    assert records[0].source_group == "bootstrap_cluster:episode:heldout"
    np.testing.assert_array_equal(records[0].physical, states)


def test_loader_prefers_merged_top_level_features_over_shard_copies(tmp_path) -> None:
    panels = tmp_path / "panels"
    features = tmp_path / "features"
    panels.mkdir()
    features.mkdir()
    scene_id = "query-merged"
    states = np.zeros((2, 2, 1, 2), dtype=np.float32)
    np.savez(panels / f"{scene_id}.npz", future_states=states)
    latent = np.zeros((2, 2, 1, 2), dtype=np.float32)
    top_level = features / f"features_{scene_id}.npz"
    np.savez(top_level, pred=latent, target=latent)
    shard = features / "shards" / "7"
    shard.mkdir(parents=True)
    shutil.copy2(top_level, shard / top_level.name)

    records = load_feature_scenes(features, panels, "speed")
    assert [record.scene_id for record in records] == [scene_id]


def test_loader_maps_feature_pair_alias_to_manifest_scene_id(tmp_path) -> None:
    panels = tmp_path / "panels" / "speed"
    features = tmp_path / "features"
    panels.mkdir(parents=True)
    features.mkdir()
    panel_alias = "pair_0000"
    formal_scene_id = "phrm-validation-00000"
    states = np.arange(2 * 1 * 1 * 2, dtype=np.float32).reshape(2, 1, 1, 2)
    np.savez(panels / f"{panel_alias}.npz", future_states=states)
    (panels / "manifest.json").write_text(
        json.dumps(
            {
                "scenes": [
                    {
                        "scene_id": formal_scene_id,
                        "path": f"{panel_alias}.npz",
                        "bootstrap_cluster": "episode:formal-source",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    latent = states.copy()
    # There is no scene_id archive field; the filename is only the panel-path
    # alias and must be replaced by the manifest's formal query ID.
    np.savez(features / f"features_{panel_alias}.npz", pred=latent, target=latent)

    records = load_feature_scenes(features, tmp_path / "panels", "speed")
    assert [record.scene_id for record in records] == [formal_scene_id]
    assert records[0].panel_path.endswith(f"{panel_alias}.npz")
    assert records[0].source_group == "bootstrap_cluster:episode:formal-source"
