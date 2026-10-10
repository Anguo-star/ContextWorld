from __future__ import annotations

import json
import hashlib
import sys
from pathlib import Path

import numpy as np

from scripts import export_prediction_accuracy_readouts as exporter
from scripts.export_prediction_accuracy_readouts import compare_export_unit


def _metric(prediction: np.ndarray, truth: np.ndarray) -> dict[str, object]:
    difference = prediction - truth
    rmse_by_dimension = np.sqrt(np.mean(np.square(difference), axis=(0, 1, 2)))
    return {"rmse_by_dimension": rmse_by_dimension.tolist()}


def test_compare_export_unit_checks_coverage_groups_folds_and_dimensional_rmse(tmp_path) -> None:
    output_root = tmp_path / "output"
    unit_id = "speed/lewm/original/s3073"
    suite = "cached-suite"
    source_path = tmp_path / "source.json"
    prediction_dir = output_root / unit_id
    prediction_dir.mkdir(parents=True)

    truth = np.arange(2 * 2 * 1 * 2, dtype=np.float64).reshape(2, 2, 1, 2)
    matched = truth + np.asarray([0.25, -0.5])
    calibration = truth - np.asarray([1.0, 0.75])
    matched_metric = _metric(matched, truth)
    calibration_metric = _metric(calibration, truth)
    query = {
        "scene_id": "scene-0",
        "source_group": "source-0",
        "fold": 0,
        "predicted_latent_matched": matched_metric,
        "oracle_true_latent": calibration_metric,
    }
    fold_receipt = {
        "fold": 0,
        "train_groups": ["source-1"],
        "heldout_groups": ["source-0"],
        "train_rows": 32,
        "sampled_rows_by_train_group": {"source-1": 32},
        "readout_fit_uses": "true_future_target_latent_only",
        "readout_solver": "primal_gram",
    }
    readout_receipt = {
        "kind": "intercept_ridge",
        "alpha": 0.001,
        "alpha_definition": "fixed test alpha",
        "feature_scale_eps": 1.0e-12,
        "input_dim": 4,
        "output_dim": 2,
        "projection": "identity",
        "projection_seed": 20261007,
        "features_standardized_float64": True,
        "hyperparameter_selection": "none; fixed before heldout scoring",
    }
    source_payload = {
        "query_metrics": [query],
        "folds": [fold_receipt],
        "readout": readout_receipt,
        "source_group_crossfit": {
            "enabled": True,
            "n_folds": 3,
            "readout_fit_excludes_heldout_source": True,
            "readout_fit_latent": "true_future_target_only",
            "maximum_samples_per_source": 32,
            "fold_assignment": {"source-0": 0, "source-1": 1},
        },
        "summaries": {},
    }
    source_path.write_text(json.dumps(source_payload), encoding="utf-8")
    aggregate_path = output_root / "aggregates" / suite / unit_id
    aggregate_path = aggregate_path.with_name(f"{aggregate_path.name}.json")
    aggregate_path.parent.mkdir(parents=True)
    aggregate_path.write_text(json.dumps(source_payload), encoding="utf-8")
    np.savez_compressed(
        prediction_dir / "scene-0.npz",
        truth=truth,
        matched=matched,
        calibration=calibration,
        scene_id=np.asarray("scene-0"),
        source_group=np.asarray("source-0"),
        fold=np.asarray(0, dtype=np.int64),
    )

    result = compare_export_unit(
        {
            "suite": suite,
            "id": unit_id,
            "task": "speed",
            "panel_identity": {"suite": suite, "task": "speed"},
            "source_physical_json": str(source_path),
        },
        output_root=output_root,
    )

    assert result["passed"] is True
    assert result["n_source_query_rows"] == result["n_prediction_archives"] == 1
    assert result["n_source_groups"] == 1
    assert result["checked_rmse_dimension_values"] == 2
    assert result["checks"]["scene_coverage_exact"] is True
    assert result["checks"]["source_groups_match_source"] is True
    assert result["checks"]["fold_assignment_matches_source"] is True
    assert result["max_abs_delta_matched_rmse_by_dimension"] == 0.0
    assert result["max_abs_delta_calibration_rmse_by_dimension"] == 0.0


def test_batch_runner_writes_summarizer_manifest_and_preserves_model_identity(
    tmp_path, monkeypatch
) -> None:
    experiment = tmp_path / "experiment"
    cache = experiment / "cache"
    features = cache / "features"
    panel_root = cache / "panels" / "speed"
    features.mkdir(parents=True)
    panel_root.mkdir(parents=True)
    unit_id = "speed/lewm/original/s3073"
    model_manifest = cache / "models.json"
    model_manifest.write_text(
        json.dumps([{"id": unit_id, "checkpoint_sha256": "checkpoint-sha"}]),
        encoding="utf-8",
    )
    source_physical = cache / "physical.json"
    source_physical.write_text(
        json.dumps(
            {
                "query_metrics": [
                    {"scene_id": f"query-{i}", "source_group": f"group-{i}"}
                    for i in range(3)
                ]
            }
        ),
        encoding="utf-8",
    )
    settings = {
        "folds": 3,
        "maximum_samples_per_source": 32,
        "ridge_alpha": 0.001,
        "projection_dim": 512,
        "projection_seed": 20261007,
        "bootstrap_reps": 1000,
        "seed": 20261007,
    }
    unit = {
        "suite": "suite-a",
        "id": unit_id,
        "task": "speed",
        "features_dir": str(features.relative_to(experiment)),
        "panel_root": str(panel_root.relative_to(experiment)),
        "source_physical_json": str(source_physical.relative_to(experiment)),
        "models_manifest": str(model_manifest.relative_to(experiment)),
        "models_manifest_sha256": hashlib.sha256(model_manifest.read_bytes()).hexdigest(),
        "checkpoint_sha256": "checkpoint-sha",
        "panel_identity": {"suite": "suite-a", "task": "speed", "manifest_sha256": "panel-sha"},
    }
    input_manifest = experiment / "input_manifest.json"
    input_manifest.write_text(json.dumps({"settings": settings, "units": [unit]}), encoding="utf-8")

    output_root = experiment / "readout"
    (output_root / unit_id).mkdir(parents=True)
    for i in range(3):
        truth = np.zeros((2, 2, 1, 2), dtype=np.float32)
        matched = np.full_like(truth, 0.1 + i * 0.01)
        calibration = np.full_like(truth, 0.05)
        np.savez_compressed(
            output_root / unit_id / f"query-{i}.npz",
            truth=truth,
            matched=matched,
            calibration=calibration,
            scene_id=np.asarray(f"query-{i}"),
            source_group=np.asarray(f"group-{i}"),
            fold=np.asarray(i % 3, dtype=np.int64),
        )

    def fake_run_unit(unit_arg, *, settings, output_root, python_executable):
        return {"suite": unit_arg["suite"], "id": unit_arg["id"], "returncode": 0}

    monkeypatch.setattr(exporter, "_run_unit", fake_run_unit)
    monkeypatch.setattr(
        exporter,
        "compare_export_unit",
        lambda unit_arg, *, output_root: {"suite": unit_arg["suite"], "id": unit_arg["id"], "passed": True},
    )

    assert exporter.main(["--manifest", str(input_manifest), "--output-root", str(output_root)]) == 0
    run_manifest = json.loads((output_root / "run_manifest.json").read_text())
    execution_manifest = json.loads((output_root / "execution_manifest.json").read_text())
    assert run_manifest == execution_manifest
    assert run_manifest["settings"] == settings
    emitted_unit = run_manifest["units"][0]
    assert emitted_unit["models_manifest"] == str(model_manifest.resolve())
    assert emitted_unit["checkpoint_sha256"] == "checkpoint-sha"
    assert emitted_unit["panel_identity"] == unit["panel_identity"]
    assert emitted_unit["source_physical_json"] == str(source_physical.resolve())
    command = exporter.build_readout_command(
        emitted_unit,
        settings=settings,
        output_root=output_root,
    )
    assert command[command.index("--panels-dir") + 1] == str(panel_root.resolve())

    scripts_dir = str(Path(exporter.__file__).resolve().parent)
    monkeypatch.syspath_prepend(scripts_dir)
    import summarize_prediction_accuracy as summarizer

    (experiment / "protocol.json").write_text(
        json.dumps(
            {
                "thresholds": {"tworoom": [0.2, 0.5, 1.0, 2.0, 5.0]},
                "primary_object": {"speed": "object"},
            }
        ),
        encoding="utf-8",
    )
    data, _ = summarizer.build(experiment)
    assert data["rows"][0]["id"] == unit_id
    assert data["rows"][0]["checkpoint_sha256"] == "checkpoint-sha"
    assert data["rows"][0]["panel_manifest_sha256"] == "panel-sha"
