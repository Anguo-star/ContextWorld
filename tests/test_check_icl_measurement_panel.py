from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import check_icl_measurement_panel as validation


def test_physical_coordinate_mappings_match_panel_contract() -> None:
    pusht7 = np.zeros((1, 1, 1, 7), dtype=np.float64)
    pusht7[..., :5] = [1.0, 2.0, 3.0, 4.0, np.pi / 2.0]
    mapped7 = validation.physical_targets_from_panel(
        {"future_states": pusht7}, "action_strength"
    )
    np.testing.assert_allclose(
        mapped7[0, 0, 0], [1.0, 2.0, 3.0, 4.0, 40.0, 0.0], atol=1e-12
    )

    pusht12 = np.zeros((1, 1, 1, 12), dtype=np.float64)
    pusht12[..., 0:2] = [10.0, 11.0]
    pusht12[..., 6:8] = [20.0, 21.0]
    pusht12[..., 10] = np.pi
    mapped12 = validation.physical_targets_from_panel(
        {"future_states": pusht12}, "contact_friction"
    )
    np.testing.assert_allclose(
        mapped12[0, 0, 0], [10.0, 11.0, 20.0, 21.0, 0.0, -40.0], atol=1e-12
    )

    mass = np.asarray([[[[0.1, 0.2]]]], dtype=np.float64)
    np.testing.assert_allclose(
        validation.physical_targets_from_panel(
            {"finger_positions": mass}, "robot_arm_mass"
        ),
        [[[[100.0, 200.0]]]],
    )

    cube = np.asarray([[[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]]]], dtype=np.float64)
    np.testing.assert_allclose(
        validation.physical_targets_from_panel(
            {"future_states": cube}, "cube_gripper_carry"
        ),
        [[[[1000.0, 2000.0, 7000.0, 3000.0, 4000.0, 5000.0]]]],
    )


def test_physical_separability_retains_zero_rows() -> None:
    targets = np.zeros((2, 2, 1, 2), dtype=np.float64)
    targets[1, 1, 0] = [1.0, 0.0]
    summary = validation.physical_separability(targets)
    assert summary["pair_count_per_horizon"] == 2
    assert summary["separable_pair_count_by_horizon"] == [1]
    assert summary["zero_or_below_tolerance_pair_count_by_horizon"] == [1]
    assert summary["separable_pair_fraction_by_horizon"] == [0.5]
    assert summary["zero_rows_retained"] is True


def _write_door_fixture(root: Path, *, split: str = "development") -> None:
    panel = root / "door"
    panel.mkdir()
    k, candidates, times = 2, 2, 5
    history = np.zeros((k, 3, 224, 224, 3), dtype=np.uint8)
    history[1, 0, 0, 0, 0] = 1
    history[1, 1, 0, 0, 0] = 2
    context_actions = np.zeros((k, 2, 5, 2), dtype=np.float32)
    candidate_actions = np.zeros((candidates, times, 2, 2), dtype=np.float32)
    candidate_actions[1, 0, 0, 0] = 1.0
    future_states = np.zeros((k, candidates, times, 2), dtype=np.float32)
    future_states[0, 1, :, 0] = 1.0
    future_states[1, 1, :, 0] = 2.0
    path = panel / "scene.npz"
    np.savez(
        path,
        history_pixels=history,
        context_actions=context_actions,
        candidate_actions=candidate_actions,
        future_states=future_states,
        conditions=np.asarray(["passable", "blocked"]),
        physical_steps=np.asarray([5, 10, 15, 20, 25]),
    )
    manifest = {
        "task": "door",
        "evaluation_split": split,
        "scenes": [
            {
                "scene_id": "fixture",
                "path": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bootstrap_cluster": "fixture-group",
            }
        ],
    }
    (panel / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_check_panel_reports_shared_query_history_and_source_group(tmp_path: Path) -> None:
    _write_door_fixture(tmp_path)
    result = validation.check_panel(tmp_path / "door")
    assert result["scene_count"] == 1
    assert result["source_group_count"] == 1
    assert result["current_image"]["all_scenes_equal"] is True
    assert result["context_actions"]["all_scenes_equal"] is True
    assert result["historical_pixels"]["all_scenes_have_a_historical_difference"] is True
    assert result["physical_separability"]["separable_pair_fraction_by_horizon"] == [0.5] * 5
    assert result["zero_action"]["retained_in_separability_denominator"] is True


def test_development_only_guard_rejects_test_panel(tmp_path: Path) -> None:
    _write_door_fixture(tmp_path, split="test")
    with pytest.raises(ValueError, match="Development-only"):
        validation.check_panel(tmp_path / "door")


def test_algebraic_oracle_and_blind_controls() -> None:
    result = validation.algebraic_self_test()
    assert result["passed"] is True
    assert result["ideal_oracle"]["matched_error"] == [0.0, 0.0]
    assert result["ideal_oracle"]["score"] == [100.0, 100.0]
    assert result["condition_mean_blind"]["score"] == [0.0, 0.0]
    assert result["condition_mean_blind"]["history_advantage"] == [0.0, 0.0]


def test_history_sufficiency_audit_separates_prior_evidence_from_panel_claim() -> None:
    result = validation.history_sufficiency_audit(
        Path(__file__).resolve().parents[1]
    )
    assert result["current_panel_construction_only_tasks"] == list(
        validation.EXPECTED_TASKS
    )
    assert set(result["prior_direct_history_decoder_tasks"]) == {
        "speed",
        "contact_friction",
        "motion_damping",
        "cube_gripper_carry",
    }
    assert set(result["prior_development_response_tasks"]) == set(
        validation.EXPECTED_TASKS
    )
    assert result["checks"]["all_nine_current_panels_construction_only"] is True
    assert result["checks"]["history_pixel_difference_not_called_sufficient"] is True
    assert result["checks"]["source_ids_and_condition_labels_not_history_evidence"] is True
    assert result["checks"]["no_new_decoder_or_model_run"] is True
    assert result["tasks"]["speed"]["prior_evidence_applies_to_current_panel"] is False


def test_documentation_audit_recognizes_history_response_contract() -> None:
    result = validation.documentation_audit(
        Path(__file__).resolve().parents[1]
    )
    assert result["checks"]["history_must_have_measurable_physical_response"] is True
    assert result["passed"] is True
