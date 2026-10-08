from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import summarize_icl_measurement_validation as summary


def _history_run(run_id: str, seed: int, offset: float = 0.0) -> dict:
    rows = []
    for index, group in enumerate(("g0", "g1")):
        matched = np.asarray([1, 2, 3, 4, 5], dtype=np.float64) + offset + index
        wrong = matched + 2.0
        separation = np.full(5, 10.0, dtype=np.float64)
        rows.append(
            {
                "scene_id": f"q{index}",
                "source_group": group,
                "matched_by_horizon": matched,
                "wrong_by_horizon": wrong,
                "B_by_horizon": separation,
                "matched": float(matched.mean()),
                "wrong": float(wrong.mean()),
                "Bq": float(separation.mean()),
            }
        )
    return {
        "id": run_id,
        "training_seed": seed,
        "queries": rows,
    }


def test_history_ratio_uses_shared_all_horizon_denominator_and_equal_seeds() -> None:
    runs = [_history_run("r0", 1), _history_run("r1", 2, offset=2.0)]
    indices = np.asarray([[0, 1], [1, 1], [0, 0]], dtype=np.int64)
    result = summary.aggregate_history(runs, indices)

    # Each run has B=10 per group, matched means 3 and 4 (then 5 and 6),
    # while each wrong value is exactly two larger.
    assert result["matched_error_ratio"] == pytest.approx(0.45)
    assert result["wrong_error_ratio"] == pytest.approx(0.65)
    assert result["gain"] == pytest.approx(0.20)
    assert result["score"] == pytest.approx(55.0)
    assert result["matched_error_ratio_by_horizon"] == pytest.approx(
        [0.25, 0.35, 0.45, 0.55, 0.65]
    )
    assert result["wrong_error_ratio_by_horizon"] == pytest.approx(
        [0.45, 0.55, 0.65, 0.75, 0.85]
    )
    assert result["gain_by_horizon"] == pytest.approx([0.2] * 5)
    assert result["bootstrap"]["gain_ci95"] is not None


def _physical_metric(query_mse: float, denominator: float) -> dict:
    return {
        "query_mse": query_mse,
        "mse": query_mse,
        "normalization_denominator": denominator,
        "rmse": float(np.sqrt(query_mse)),
        "normalized_rmse": float(np.sqrt(query_mse / denominator)),
    }


def _physical_payload(scale: float = 1.0, scene_ids=("q0", "q1", "q2")) -> dict:
    # Unequal B values and unequal source-group sizes make pooled MSE/B differ
    # from an average of per-query normalized RMSE values.
    mse = {
        "oracle": [1.0, 9.0, 25.0],
        "pred": [4.0, 36.0, 100.0],
        "wrong": [9.0, 81.0, 400.0],
    }
    denominators = [1.0, 9.0, 100.0]
    rows = []
    for index, scene_id in enumerate(scene_ids):
        rows.append(
            {
                "scene_id": scene_id,
                "oracle_true_latent": _physical_metric(
                    mse["oracle"][index] * scale, denominators[index]
                ),
                "predicted_latent_matched": _physical_metric(
                    mse["pred"][index] * scale, denominators[index]
                ),
                "predicted_latent_mismatched": _physical_metric(
                    mse["wrong"][index] * scale, denominators[index]
                ),
            }
        )
    return {
        "protocol": "development_source_group_3fold_crossfit_physical_readout_v1",
        "target": {"units": "px"},
        "query_metrics": rows,
    }


def test_physical_summary_keeps_oracle_floor_and_reports_paired_gain() -> None:
    groups = {"q0": "g0", "q1": "g1", "q2": "g1"}
    result = summary.aggregate_physical(
        [_physical_payload(), _physical_payload(4.0)],
        groups,
        np.asarray([[0, 1], [1, 0]], dtype=np.int64),
    )
    assert result["available"] is True
    pooled_oracle_norm = np.sqrt(35.0 / 110.0)
    pooled_pred_norm = np.sqrt(140.0 / 110.0)
    pooled_wrong_norm = np.sqrt(490.0 / 110.0)
    assert result["oracle_readout_normalized_rmse"] == pytest.approx(1.5 * pooled_oracle_norm)
    assert result["predicted_normalized_rmse"] == pytest.approx(1.5 * pooled_pred_norm)
    assert result["wrong_normalized_rmse"] == pytest.approx(1.5 * pooled_wrong_norm)
    assert result["gain_normalized_rmse"] == pytest.approx(
        1.5 * (pooled_wrong_norm - pooled_pred_norm)
    )
    assert result["gain_normalized_mse"] == pytest.approx(2.5 * (490.0 - 140.0) / 110.0)
    assert result["oracle_readout_rmse"] == pytest.approx(1.5 * np.sqrt(35.0 / 3.0))
    assert result["gain_rmse"] == pytest.approx(1.5 * (np.sqrt(490.0 / 3.0) - np.sqrt(140.0 / 3.0)))
    assert "not a common physical benchmark" in result["calibration_claim"]
    assert result["bootstrap"]["gain_normalized_rmse_ci95"] is not None


def _write_minimal_run(root: Path, spec: dict, panel: Path, physical_root: Path) -> None:
    result_dir = root / "results" / spec["id"]
    result_dir.mkdir(parents=True)
    panel_manifest = panel / "manifest.json"
    panel_sha = hashlib.sha256(panel_manifest.read_bytes()).hexdigest()
    entries = json.loads(panel_manifest.read_text())["scenes"]
    entry_hashes = {}
    for entry in entries:
        row_path = result_dir / f"{entry['scene_id']}.json"
        row = {
            "scene_id": entry["scene_id"],
            "checkpoint_sha256": spec["checkpoint_sha256"],
            "source_sha256": entry["sha256"],
            "horizons": [5, 10, 15, 20, 25],
            "conditions": 2,
            "candidates": 1,
            "matched": {"energy_mean": [1, 1, 1, 1, 1], "energy_sum": [2, 2, 2, 2, 2]},
            "wrong_allother": {"energy_mean": [2, 2, 2, 2, 2], "energy_sum": [4, 4, 4, 4, 4]},
            "target_separation_energy": [10, 10, 10, 10, 10],
        }
        row_path.write_text(json.dumps(row))
        entry_hashes[row_path.name] = hashlib.sha256(row_path.read_bytes()).hexdigest()
    receipt = {
        "checkpoint_sha256": spec["checkpoint_sha256"],
        "panel_sha256": panel_sha,
        "state_hash_before": "same",
        "state_hash_after": "same",
        "no_training": True,
        "scenes": len(entries),
        "entry_hashes": entry_hashes,
    }
    (result_dir / "receipt.json").write_text(json.dumps(receipt))
    physical_dir = physical_root / spec["id"].rsplit("/", 1)[0]
    physical_dir.mkdir(parents=True, exist_ok=True)
    (physical_root / f"{spec['id']}.json").parent.mkdir(parents=True, exist_ok=True)
    payload = _physical_payload(scene_ids=tuple(entry["scene_id"] for entry in entries))
    (physical_root / f"{spec['id']}.json").write_text(json.dumps(payload))


def test_allow_partial_is_required_until_formal_run_and_row_counts_exist(tmp_path: Path) -> None:
    root = tmp_path / "root"
    panel = tmp_path / "panels" / "task"
    panel.mkdir(parents=True)
    scenes = [
        {"scene_id": "q0", "path": "q0.npz", "sha256": "source0"},
        {"scene_id": "q1", "path": "q1.npz", "sha256": "source1"},
    ]
    (panel / "manifest.json").write_text(json.dumps({"scenes": scenes}))
    (tmp_path / "bootstrap_clusters.json").write_text(
        json.dumps({"task": {"q0": "g0", "q1": "g1"}})
    )
    spec = {
        "id": "task/lewm/original/s1",
        "task": "task",
        "family": "lewm",
        "regime": "original",
        "training_seed": 1,
        "checkpoint_sha256": "checkpoint",
    }
    (tmp_path / "models.json").write_text(json.dumps([spec]))
    _write_minimal_run(root, spec, panel, root / "physical")
    with pytest.raises(RuntimeError, match="Formal summary"):
        summary.summarize(
            root=root,
            models_path=tmp_path / "models.json",
            panels_root=tmp_path / "panels",
            physical_root=root / "physical",
            bootstrap_map_path=tmp_path / "bootstrap_clusters.json",
            bootstrap_reps=3,
        )
    partial = summary.summarize(
        root=root,
        models_path=tmp_path / "models.json",
        panels_root=tmp_path / "panels",
        physical_root=root / "physical",
        bootstrap_map_path=tmp_path / "bootstrap_clusters.json",
        bootstrap_reps=3,
        allow_partial=True,
    )
    assert partial["formal_gate"]["complete"] is False
    assert partial["formal_gate"]["observed_runs"] == 1
    assert len(partial["rows"]) == 1


def test_markdown_and_csv_emit_one_row_per_validation_combination(tmp_path: Path) -> None:
    physical = {
        "oracle_readout_normalized_rmse": 0.1,
        "predicted_normalized_rmse": 0.2,
        "wrong_normalized_rmse": 0.3,
        "gain_normalized_rmse": 0.1,
        "gain_normalized_mse": 0.1,
        "oracle_readout_rmse": 1.0,
        "predicted_rmse": 2.0,
        "wrong_rmse": 3.0,
        "gain_rmse": 1.0,
        "rmse_units": "px",
    }
    rows = [
        {
            "task": f"task{i}", "family": "lewm", "regime": "original",
            "training_repetitions": 1, "n_queries": 2, "n_source_groups": 2,
            "matched_error_ratio": 0.1, "wrong_error_ratio": 0.2, "gain": 0.1,
            "score": 90.0, "physical": physical,
        }
        for i in range(86)
    ]
    paths = summary.write_outputs({"rows": rows}, tmp_path / "validation")
    assert all(path.exists() for path in paths)
    markdown = paths[2].read_text()
    assert sum(line.startswith("| LeWM |") for line in markdown.splitlines()) == 86
    with paths[1].open(newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 86
