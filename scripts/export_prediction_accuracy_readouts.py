#!/usr/bin/env python3
"""Export and verify per-scene predictions for cached physical readout units.

The input manifest is the JSON receipt produced by a cached readout run.  This
script reruns the same fixed source-group readout from cached features, asks
``validate_physical_readout.py`` to save each held-out prediction, then checks
coverage, source groups, folds, and per-dimension RMSE against the saved
physical JSON.  It never trains or runs a world model.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np


DEFAULT_RMSE_RTOL = 1.0e-12
DEFAULT_RMSE_ATOL = 1.0e-12
FOLD_RECEIPT_FIELDS = (
    "fold",
    "train_groups",
    "heldout_groups",
    "train_rows",
    "sampled_rows_by_train_group",
    "readout_fit_uses",
    "readout_solver",
)
READOUT_RECEIPT_FIELDS = (
    "kind",
    "alpha",
    "alpha_definition",
    "feature_scale_eps",
    "input_dim",
    "output_dim",
    "projection",
    "projection_seed",
    "features_standardized_float64",
    "hyperparameter_selection",
)
CROSSFIT_RECEIPT_FIELDS = (
    "enabled",
    "n_folds",
    "readout_fit_excludes_heldout_source",
    "readout_fit_latent",
    "maximum_samples_per_source",
)
PREDICTION_ARCHIVE_FIELDS = {
    "truth",
    "matched",
    "calibration",
    "scene_id",
    "source_group",
    "fold",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return payload


def _safe_relative_path(value: str, *, name: str) -> Path:
    path = PurePosixPath(str(value))
    if path.is_absolute() or not path.parts or any(part in {".", ".."} for part in path.parts):
        raise ValueError(f"{name} must be a safe relative path, got {value!r}")
    return Path(*path.parts)


def _validate_unit(unit: Mapping[str, Any]) -> None:
    required = ("id", "task", "features_dir", "panel_root", "source_physical_json")
    missing = [key for key in required if not unit.get(key)]
    if missing:
        raise ValueError(f"manifest unit is missing required fields: {', '.join(missing)}")
    _safe_relative_path(str(unit["id"]), name="unit id")
    suite = unit.get("suite", "default")
    _safe_relative_path(str(suite), name="suite")
    for key in ("features_dir", "panel_root", "source_physical_json"):
        if not Path(str(unit[key])).exists():
            raise FileNotFoundError(f"unit {unit['id']}: {key} does not exist: {unit[key]}")


def _readout_cli_settings(settings: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    """Map required manifest settings to CLI options without applying defaults."""

    mapping = (
        ("folds", "--folds"),
        ("maximum_samples_per_source", "--max-samples-per-source"),
        ("ridge_alpha", "--ridge-alpha"),
        ("projection_dim", "--projection-dim"),
        ("projection_seed", "--projection-seed"),
        ("bootstrap_reps", "--bootstrap-reps"),
        ("seed", "--seed"),
    )
    missing = [key for key, _ in mapping if key not in settings]
    if missing:
        raise ValueError(
            "manifest settings must include the original readout values for: "
            + ", ".join(missing)
        )
    return tuple((option, str(settings[key])) for key, option in mapping)


def _unit_paths(unit: Mapping[str, Any], output_root: Path) -> dict[str, Path]:
    suite = _safe_relative_path(str(unit.get("suite", "default")), name="suite")
    unit_id = _safe_relative_path(str(unit["id"]), name="unit id")
    unit_filename = Path(f"{unit_id.name}.json")
    log_filename = Path(f"{unit_id.name}.log")
    return {
        "prediction_dir": output_root / unit_id,
        "aggregate": output_root / "aggregates" / suite / unit_id.parent / unit_filename,
        "log": output_root / "logs" / suite / unit_id.parent / log_filename,
    }


def build_readout_command(
    unit: Mapping[str, Any],
    *,
    settings: Mapping[str, Any],
    output_root: str | Path,
    python_executable: str = sys.executable,
) -> list[str]:
    """Build the exact CLI invocation described by the manifest settings."""

    _validate_unit(unit)
    paths = _unit_paths(unit, Path(output_root))
    panel_root = Path(str(unit["panel_root"]))
    command = [
        python_executable,
        str(Path(__file__).with_name("validate_physical_readout.py")),
        "--features-dir",
        str(unit["features_dir"]),
        "--panels-dir",
        str(panel_root),
        "--task",
        str(unit["task"]),
        "--output",
        str(paths["aggregate"]),
        "--prediction-output-dir",
        str(paths["prediction_dir"]),
    ]
    for option, value in _readout_cli_settings(settings):
        command.extend((option, value))
    return command


def _run_unit(
    unit: Mapping[str, Any],
    *,
    settings: Mapping[str, Any],
    output_root: Path,
    python_executable: str,
) -> dict[str, Any]:
    paths = _unit_paths(unit, output_root)
    paths["aggregate"].parent.mkdir(parents=True, exist_ok=True)
    paths["log"].parent.mkdir(parents=True, exist_ok=True)
    paths["prediction_dir"].mkdir(parents=True, exist_ok=True)
    command = build_readout_command(
        unit,
        settings=settings,
        output_root=output_root,
        python_executable=python_executable,
    )
    env = os.environ.copy()
    for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[variable] = str(settings.get(variable, "1"))
    try:
        with paths["log"].open("w", encoding="utf-8") as log:
            process = subprocess.run(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                check=False,
                text=True,
            )
        return {
            "suite": str(unit.get("suite", "default")),
            "id": str(unit["id"]),
            "returncode": int(process.returncode),
            "command": command,
            "aggregate": str(paths["aggregate"]),
            "prediction_dir": str(paths["prediction_dir"]),
            "log": str(paths["log"]),
        }
    except OSError as exc:
        return {
            "suite": str(unit.get("suite", "default")),
            "id": str(unit["id"]),
            "returncode": -1,
            "command": command,
            "aggregate": str(paths["aggregate"]),
            "prediction_dir": str(paths["prediction_dir"]),
            "log": str(paths["log"]),
            "error": str(exc),
        }


def _allclose(actual: Any, expected: Any, *, rtol: float, atol: float) -> tuple[bool, float, float]:
    actual_array = np.asarray(actual, dtype=np.float64)
    expected_array = np.asarray(expected, dtype=np.float64)
    if actual_array.shape != expected_array.shape:
        return False, math.inf, math.inf
    delta = np.abs(actual_array - expected_array)
    maximum_absolute = float(delta.max(initial=0.0))
    maximum_relative = float((delta / np.maximum(np.abs(expected_array), atol)).max(initial=0.0))
    return bool(np.allclose(actual_array, expected_array, rtol=rtol, atol=atol)), maximum_absolute, maximum_relative


def compare_export_unit(
    unit: Mapping[str, Any],
    *,
    output_root: str | Path,
    rtol: float = DEFAULT_RMSE_RTOL,
    atol: float = DEFAULT_RMSE_ATOL,
) -> dict[str, Any]:
    """Check one unit's exported predictions against its saved physical JSON."""

    output_root = Path(output_root)
    paths = _unit_paths(unit, output_root)
    source_path = Path(str(unit["source_physical_json"]))
    source = _load_json(source_path)
    recomputed = _load_json(paths["aggregate"])
    source_rows = {str(row["scene_id"]): row for row in source.get("query_metrics", [])}
    recomputed_rows = {str(row["scene_id"]): row for row in recomputed.get("query_metrics", [])}
    if len(source_rows) != len(source.get("query_metrics", [])):
        raise ValueError(f"{unit['id']}: source physical JSON contains duplicate scene IDs")
    if len(recomputed_rows) != len(recomputed.get("query_metrics", [])):
        raise ValueError(f"{unit['id']}: recomputed JSON contains duplicate scene IDs")

    archive_rows: dict[str, dict[str, Any]] = {}
    for archive_path in sorted(paths["prediction_dir"].glob("*.npz")):
        with np.load(archive_path, allow_pickle=False) as archive:
            scene_id = str(archive["scene_id"].item())
            if scene_id in archive_rows:
                raise ValueError(f"{unit['id']}: duplicate prediction archive for {scene_id}")
            archive_rows[scene_id] = {
                "truth": archive["truth"].copy(),
                "matched": archive["matched"].copy(),
                "calibration": archive["calibration"].copy(),
                "source_group": str(archive["source_group"].item()),
                "fold": int(archive["fold"].item()),
                "fields": set(archive.files),
            }

    scene_ids_equal = set(source_rows) == set(recomputed_rows) == set(archive_rows)
    matched_ok = True
    calibration_ok = True
    source_groups_ok = True
    folds_ok = True
    archive_fields_ok = True
    max_matched_abs = max_calibration_abs = 0.0
    max_matched_rel = max_calibration_rel = 0.0
    checked_dimensions = 0
    for scene_id in sorted(set(source_rows) & set(recomputed_rows) & set(archive_rows)):
        source_row = source_rows[scene_id]
        recomputed_row = recomputed_rows[scene_id]
        archive_row = archive_rows[scene_id]
        source_groups_ok &= (
            archive_row["source_group"]
            == str(source_row["source_group"])
            == str(recomputed_row["source_group"])
        )
        folds_ok &= (
            archive_row["fold"]
            == int(source_row["fold"])
            == int(recomputed_row["fold"])
        )
        archive_fields_ok &= archive_row["fields"] == PREDICTION_ARCHIVE_FIELDS
        truth = archive_row["truth"]
        matched = archive_row["matched"]
        calibration = archive_row["calibration"]
        if (
            truth.ndim != 4
            or truth.shape != matched.shape
            or truth.shape != calibration.shape
            or not np.all(np.isfinite(truth))
            or not np.all(np.isfinite(matched))
            or not np.all(np.isfinite(calibration))
        ):
            archive_fields_ok = False
            continue
        checked_dimensions += int(truth.shape[-1])
        matched_rmse = np.sqrt(np.mean(np.square(matched - truth), axis=(0, 1, 2)))
        calibration_rmse = np.sqrt(np.mean(np.square(calibration - truth), axis=(0, 1, 2)))
        source_matched = source_row["predicted_latent_matched"]["rmse_by_dimension"]
        source_calibration = source_row["oracle_true_latent"]["rmse_by_dimension"]
        matched_pass, matched_abs, matched_rel = _allclose(
            matched_rmse, source_matched, rtol=rtol, atol=atol
        )
        calibration_pass, calibration_abs, calibration_rel = _allclose(
            calibration_rmse, source_calibration, rtol=rtol, atol=atol
        )
        matched_ok &= matched_pass
        calibration_ok &= calibration_pass
        max_matched_abs = max(max_matched_abs, matched_abs)
        max_matched_rel = max(max_matched_rel, matched_rel)
        max_calibration_abs = max(max_calibration_abs, calibration_abs)
        max_calibration_rel = max(max_calibration_rel, calibration_rel)
        matched_ok &= _allclose(
            recomputed_row["predicted_latent_matched"]["rmse_by_dimension"],
            source_matched,
            rtol=rtol,
            atol=atol,
        )[0]
        calibration_ok &= _allclose(
            recomputed_row["oracle_true_latent"]["rmse_by_dimension"],
            source_calibration,
            rtol=rtol,
            atol=atol,
        )[0]

    old_groups: dict[str, list[str]] = collections.defaultdict(list)
    new_groups: dict[str, list[str]] = collections.defaultdict(list)
    for row in source.get("query_metrics", []):
        old_groups[str(row["source_group"])].append(str(row["scene_id"]))
    for row in recomputed.get("query_metrics", []):
        new_groups[str(row["source_group"])].append(str(row["scene_id"]))
    old_groups_sorted = {key: sorted(value) for key, value in sorted(old_groups.items())}
    new_groups_sorted = {key: sorted(value) for key, value in sorted(new_groups.items())}
    source_groups_ok &= old_groups_sorted == new_groups_sorted
    source_fold_map = source.get("source_group_crossfit", {}).get("fold_assignment", {})
    recomputed_fold_map = recomputed.get("source_group_crossfit", {}).get("fold_assignment", {})
    fold_assignment_equal = source_fold_map == recomputed_fold_map
    old_folds = source.get("folds", [])
    new_folds = recomputed.get("folds", [])
    fold_receipts_equal = len(old_folds) == len(new_folds) and all(
        all(old.get(key) == new.get(key) for key in FOLD_RECEIPT_FIELDS)
        for old, new in zip(old_folds, new_folds)
    )
    readout_configuration_equal = all(
        source.get("readout", {}).get(key) == recomputed.get("readout", {}).get(key)
        for key in READOUT_RECEIPT_FIELDS
    )
    source_crossfit = source.get("source_group_crossfit", {})
    recomputed_crossfit = recomputed.get("source_group_crossfit", {})
    crossfit_configuration_equal = all(
        source_crossfit.get(key) == recomputed_crossfit.get(key)
        for key in CROSSFIT_RECEIPT_FIELDS
    )

    metadata_hashes: dict[str, bool] = {}
    source_hash_after = _sha256(source_path)
    expected_source_hash = unit.get("source_physical_sha256_before")
    if expected_source_hash:
        metadata_hashes["source_physical_json"] = source_hash_after == expected_source_hash
    panel_manifest = unit.get("source_panel_manifest")
    panel_identity = unit.get("panel_identity", {})
    expected_panel_hash = panel_identity.get("manifest_sha256") if isinstance(panel_identity, Mapping) else None
    panel_hash_after = None
    if panel_manifest and expected_panel_hash:
        panel_hash_after = _sha256(Path(str(panel_manifest)))
        metadata_hashes["panel_manifest"] = panel_hash_after == expected_panel_hash
    models_manifest = unit.get("models_manifest")
    expected_models_hash = unit.get("models_manifest_sha256")
    models_hash_after = None
    if models_manifest and expected_models_hash:
        models_hash_after = _sha256(Path(str(models_manifest)))
        metadata_hashes["models_manifest"] = models_hash_after == expected_models_hash

    checks = {
        "scene_coverage_exact": scene_ids_equal,
        "archive_fields_and_shapes_valid": archive_fields_ok,
        "matched_rmse_by_dimension_matches_source": matched_ok,
        "calibration_rmse_by_dimension_matches_source": calibration_ok,
        "source_groups_match_source": source_groups_ok,
        "folds_match_source": folds_ok,
        "fold_assignment_matches_source": fold_assignment_equal,
        "fold_fit_receipts_match_source": fold_receipts_equal,
        "readout_configuration_matches_source": readout_configuration_equal,
        "crossfit_configuration_matches_source": crossfit_configuration_equal,
        "source_metadata_hashes_unchanged": all(metadata_hashes.values()) if metadata_hashes else None,
    }
    passed = all(value is True for key, value in checks.items() if value is not None)
    group_counts = {key: len(value) for key, value in old_groups_sorted.items()}
    result = {
        "suite": str(unit.get("suite", "default")),
        "id": str(unit["id"]),
        "task": str(unit["task"]),
        "panel_identity": unit.get("panel_identity"),
        "source_physical_json": str(source_path),
        "source_physical_sha256_after": source_hash_after,
        "source_panel_manifest_sha256_after": panel_hash_after,
        "models_manifest_sha256_after": models_hash_after,
        "n_source_query_rows": len(source_rows),
        "n_recomputed_query_rows": len(recomputed_rows),
        "n_prediction_archives": len(archive_rows),
        "n_source_groups": len(old_groups_sorted),
        "source_group_scene_counts": group_counts,
        "source_group_to_scenes": old_groups_sorted,
        "checked_rmse_dimension_values": checked_dimensions,
        "tolerance": {"rtol": rtol, "atol": atol},
        "max_abs_delta_matched_rmse_by_dimension": max_matched_abs,
        "max_relative_delta_matched_rmse_by_dimension": max_matched_rel,
        "max_abs_delta_calibration_rmse_by_dimension": max_calibration_abs,
        "max_relative_delta_calibration_rmse_by_dimension": max_calibration_rel,
        "source_summary_rmse": {
            metric: source.get("summaries", {}).get(metric, {}).get("point_estimate")
            for metric in (
                "oracle_true_latent_floor_rmse",
                "predicted_latent_matched_rmse",
                "oracle_true_latent_floor_normalized_rmse",
                "predicted_latent_matched_normalized_rmse",
            )
        },
        "checks": checks,
        "passed": passed,
    }
    if str(unit["task"]) == "motion_damping":
        result["motion_128_group_inventory"] = {
            "query_scenes": len(source_rows),
            "source_group_count": len(old_groups_sorted),
            "scenes_per_group": dict(collections.Counter(group_counts.values())),
            "group_to_scenes": old_groups_sorted,
        }
    return result


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--python", dest="python_executable", default=sys.executable)
    return parser


def _resolve_manifest_paths(
    units: Sequence[Mapping[str, Any]], manifest_dir: Path
) -> list[dict[str, Any]]:
    path_fields = (
        "features_dir",
        "panel_root",
        "panels_dir",
        "source_physical_json",
        "source_panel_manifest",
        "models_manifest",
    )
    resolved: list[dict[str, Any]] = []
    for unit in units:
        item = dict(unit)
        for key in path_fields:
            value = item.get(key)
            if value:
                path = Path(str(value))
                item[key] = str(path if path.is_absolute() else (manifest_dir / path).resolve())
        resolved.append(item)
    return resolved


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")
    manifest_path = args.manifest.resolve()
    manifest = _load_json(manifest_path)
    settings = manifest.get("settings")
    raw_units = manifest.get("units")
    if not isinstance(settings, Mapping) or not isinstance(raw_units, list) or not raw_units:
        raise SystemExit("manifest must contain a settings object and a non-empty units list")
    for unit in raw_units:
        if not isinstance(unit, Mapping):
            raise SystemExit("every manifest unit must be a JSON object")
    units = _resolve_manifest_paths(raw_units, manifest_path.parent)
    for unit in units:
        _validate_unit(unit)
    unit_ids = [str(unit["id"]) for unit in units]
    if len(set(unit_ids)) != len(unit_ids):
        raise SystemExit("manifest contains duplicate unit IDs")
    _readout_cli_settings(settings)

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    execution_manifest = {
        "schema_version": 1,
        "source_manifest": str(manifest_path),
        "output_root": str(output_root),
        "python_executable": args.python_executable,
        "workers": args.workers,
        "settings": dict(settings),
        "units": units,
    }
    manifest_text = json.dumps(execution_manifest, indent=2, sort_keys=True) + "\n"
    (output_root / "execution_manifest.json").write_text(manifest_text, encoding="utf-8")
    # The summarizer consumes readout/run_manifest.json, so publish the same
    # resolved unit records and settings under that canonical name as well.
    (output_root / "run_manifest.json").write_text(manifest_text, encoding="utf-8")

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        run_status = list(
            pool.map(
                lambda unit: _run_unit(
                    unit,
                    settings=settings,
                    output_root=output_root,
                    python_executable=args.python_executable,
                ),
                units,
            )
        )
    (output_root / "run_status.json").write_text(
        json.dumps(run_status, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    status_by_id = {row["id"]: row for row in run_status}
    unit_receipts: list[dict[str, Any]] = []
    for unit in units:
        status = status_by_id[str(unit["id"])]
        if status["returncode"] != 0:
            unit_receipts.append(
                {
                    "suite": str(unit.get("suite", "default")),
                    "id": str(unit["id"]),
                    "panel_identity": unit.get("panel_identity"),
                    "passed": False,
                    "error": status.get("error", f"readout CLI exited {status['returncode']}"),
                }
            )
            continue
        try:
            unit_receipts.append(compare_export_unit(unit, output_root=output_root))
        except Exception as exc:  # Preserve a machine-readable receipt for every failed unit.
            unit_receipts.append(
                {
                    "suite": str(unit.get("suite", "default")),
                    "id": str(unit["id"]),
                    "panel_identity": unit.get("panel_identity"),
                    "passed": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    passed_count = sum(bool(row.get("passed")) for row in unit_receipts)
    receipt = {
        "schema_version": 1,
        "purpose": (
            "Compare exported held-out predictions with source physical JSON dimensional RMSE, "
            "scene coverage, source groups, and fold receipts."
        ),
        "settings": dict(settings),
        "tolerance": {"rtol": DEFAULT_RMSE_RTOL, "atol": DEFAULT_RMSE_ATOL},
        "n_units": len(unit_receipts),
        "n_units_passed": passed_count,
        "all_checks_passed": passed_count == len(unit_receipts),
        "units": unit_receipts,
    }
    (output_root / "comparison_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for row in unit_receipts:
        print(f"{'PASS' if row.get('passed') else 'FAIL'} {row.get('suite')} {row.get('id')}")
    return 0 if receipt["all_checks_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
