from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import validate_prediction_accuracy_controls as controls  # noqa: E402


def _write_fixture_panels(root: Path) -> tuple[Path, Path]:
    pusht_root = root / "pusht"
    tworoom_root = root / "tworoom"
    for task in controls.PUSHT_TASKS:
        panel_dir = pusht_root / task
        panel_dir.mkdir(parents=True)
        width = 7 if task == "action_strength" else 12
        future = np.zeros((2, 1, 5, width), dtype=np.float32)
        scene_id = f"{task}-scene"
        scene = {"scene_id": scene_id, "path": f"{scene_id}.npz"}
        fields = {"future_states": future}
        if task != "action_strength":
            fields["query_state"] = np.zeros((2, width), dtype=np.float32)
        np.savez_compressed(panel_dir / scene["path"], **fields)
        (panel_dir / "manifest.json").write_text(
            json.dumps({"scenes": [scene]}), encoding="utf-8"
        )

    for task in controls.TWOROOM_TASKS:
        panel_dir = tworoom_root / task
        panel_dir.mkdir(parents=True)
        k = 3 if task == "speed" else 2
        future = np.zeros((k, 1, 5, 2), dtype=np.float32)
        scene_id = f"{task}-scene"
        scene = {"scene_id": scene_id, "path": f"{scene_id}.npz"}
        if task == "speed":
            scene["query_state"] = [0.0, 0.0]
            fields = {"future_states": future}
        else:
            fields = {
                "future_states": future,
                "query_state": np.zeros(2, dtype=np.float32),
            }
        np.savez_compressed(panel_dir / scene["path"], **fields)
        (panel_dir / "manifest.json").write_text(
            json.dumps({"scenes": [scene]}), encoding="utf-8"
        )
    return pusht_root, tworoom_root


def test_full_controls_have_perfect_truth_offsets_and_ceiling_and_horizon_summary(tmp_path):
    pusht_root, tworoom_root = _write_fixture_panels(tmp_path)
    strength_states = {"action_strength-scene": np.zeros((2, 7), dtype=np.float32)}

    report = controls.validate(
        pusht_root,
        tworoom_root,
        strength_query_states=strength_states,
        strength_state_provenance={"fixture": True},
    )

    assert all(report["control_invariants"].values())
    assert report["current_state_availability"]["action_strength"]["available"]
    assert all(row["accuracy_percent"] == 100.0 for row in report["threshold_metrics"] if row["control"] == "exact_truth")

    alpha_one = [
        row
        for row in report["threshold_metrics"]
        if row["task"] == "action_strength"
        and row["task_relevant_object"]
        and row["control"] == "current_state_displacement_scale"
        and row["parameter"] == "1"
    ]
    assert len(alpha_one) == 25
    assert all(row["accuracy_percent"] == 100.0 for row in alpha_one)

    summary_alpha_one = [
        row
        for row in report["all_horizon_aggregate_summary"]
        if row["task"] == "action_strength"
        and row["object"] == "agent"
        and row["control"] == "current_state_displacement_scale"
        and row["parameter"] == "1"
        and row["threshold_subset"] == "all5"
    ]
    assert len(summary_alpha_one) == 1
    assert summary_alpha_one[0]["accuracy_percent"] == 100.0
    assert summary_alpha_one[0]["horizons_included"] == [5, 10, 15, 20, 25]

    for task in ("action_strength", "contact_friction", "motion_damping", "door"):
        assert all(
            row["history_blind_ceiling_percent"] == 100.0
            for row in report["threshold_metrics"]
            if row["task"] == task and row["history_blind_ceiling_percent"] is not None
        )


def test_fixed_offset_control_is_monotonic_and_binary_ceiling_uses_strict_two_tau_rule():
    truth = np.zeros((2, 1, 3, 2), dtype=np.float64)
    truth[1, 0, :, 0] = [0.5, 2.0, 3.0]
    ceiling = controls.binary_history_blind_ceiling(truth, tau=1.0)
    assert ceiling[0, 0].tolist() == [1.0, 0.5, 0.5]
    assert ceiling[1, 0].tolist() == [1.0, 0.5, 0.5]

    predictions, _ = controls._prediction_controls(truth, np.zeros((2, 1, 1, 2)))
    offsets = [x for x in predictions if x[0] == "fixed_norm_offset"]
    tau = 8.0
    scores = [float(np.mean(np.linalg.norm(pred - truth, axis=-1) < tau)) for _, _, pred in offsets]
    assert all(left >= right for left, right in zip(scores, scores[1:]))


def test_missing_current_state_is_explicit_and_partial_sidecar_fails(tmp_path):
    truth = np.zeros((2, 1, 5, 2), dtype=np.float64)
    _, unavailable = controls._prediction_controls(truth, current=None)
    assert {row["control"] for row in unavailable} == {
        "current_state_displacement_scale",
        "temporal_lag_prepend_current",
    }

    pusht_root, tworoom_root = _write_fixture_panels(tmp_path)
    with pytest.raises(ValueError, match="map every Action Strength scene ID exactly"):
        controls.validate(pusht_root, tworoom_root, strength_query_states={})
