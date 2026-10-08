#!/usr/bin/env python3
"""Decompose frozen physical readout diagnostics by target object.

This is an additive diagnostic around ``validate_physical_readout``.  It does
not change the existing score or its readout.  For non-PushT tasks it can
reaggregate the cached per-coordinate RMSEs from an existing physical result
JSON.  PushT requires the existing frozen features so the same cross-fitted
readout can be replayed and evaluated on complete-condition canvas subsets.

The PushT center-in-canvas subset is a geometric necessary condition only; it
does not establish pixel-level visibility or identifiability.  A paired cell
is called in-canvas only when every hidden condition has both pusher and block
centers in the inclusive 0..512 canvas.  The complement is retained as a
separate paired diagnostic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from scripts import validate_physical_readout as readout


PUSHT_TASKS = {"action_strength", "contact_friction", "motion_damping"}
CANVAS_LOW = 0.0
CANVAS_HIGH = 512.0
METRIC_BRANCHES = (
    "oracle_true_latent",
    "predicted_latent_matched",
    "predicted_latent_mismatched",
)


def target_objects(task: str) -> dict[str, tuple[int, ...]]:
    """Return object-to-target-dimension indices for the fixed target schema."""

    family = readout.canonical_task(task)
    if family == "tworoom":
        return {"agent": (0, 1)}
    if family == "pusht":
        return {"pusher": (0, 1), "block": (2, 3, 4, 5)}
    if family == "mass":
        return {"finger": (0, 1)}
    return {"effector": (0, 1, 2), "cube": (3, 4, 5)}


def target_dimensions(task: str) -> tuple[str, ...]:
    return readout.target_spec(task).names


def _finite_metric(values: np.ndarray) -> float | None:
    if values.size == 0:
        return None
    result = float(np.mean(np.square(values)))
    return result if math.isfinite(result) else None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_receipt(
    *,
    reference_path: Path,
    task: str,
    features_dir: Path | None = None,
    panels_dir: Path | None = None,
    records: Sequence[readout.FeatureScene] | None = None,
    models_receipt_path: Path | None = None,
    model_id: str | None = None,
) -> dict[str, Any]:
    """Record checksums and cheap inventories without rescanning large inputs."""

    here = Path(__file__).resolve()
    legacy = Path(readout.__file__).resolve()
    receipt: dict[str, Any] = {
        "reference_result": {
            "path": str(reference_path.resolve()),
            "sha256": sha256_file(reference_path),
        },
        "diagnostic_script": {"path": str(here), "sha256": sha256_file(here)},
        "legacy_readout_script": {"path": str(legacy), "sha256": sha256_file(legacy)},
        "feature_cache_inventory": None,
        "panel_manifest": None,
        "panel_archive_inventory": None,
        "models_receipt": None,
        "model_checkpoint_sha256": None,
    }
    if features_dir is not None:
        root = features_dir.resolve()
        if root.is_file():
            feature_paths = [root]
        else:
            feature_paths = sorted(root.glob("features_*.npz"))
            if not feature_paths:
                feature_paths = sorted(root.rglob("features_*.npz"))
        inventory = [
            {"path": str(path.resolve()), "size_bytes": int(path.stat().st_size)}
            for path in feature_paths
        ]
        inventory_bytes = json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()
        receipt["feature_cache_inventory"] = {
            "root": str(root),
            "archive_count": len(inventory),
            "total_size_bytes": int(sum(row["size_bytes"] for row in inventory)),
            "path_size_inventory_sha256": hashlib.sha256(inventory_bytes).hexdigest(),
            "content_hashing": False,
        }
    if panels_dir is not None and records is not None:
        family = readout.canonical_task(task)
        candidates = [panels_dir / task, panels_dir / family]
        panel_root = next((path for path in candidates if path.is_dir()), panels_dir)
        manifest_path = panel_root / "manifest.json"
        if manifest_path.is_file():
            receipt["panel_manifest"] = {
                "path": str(manifest_path.resolve()),
                "sha256": sha256_file(manifest_path),
            }
        panel_paths = sorted({Path(record.panel_path).resolve() for record in records})
        panel_inventory = [
            {"path": str(path), "size_bytes": int(path.stat().st_size)}
            for path in panel_paths
        ]
        panel_inventory_bytes = json.dumps(
            panel_inventory, sort_keys=True, separators=(",", ":")
        ).encode()
        receipt["panel_archive_inventory"] = {
            "archive_count": len(panel_inventory),
            "total_size_bytes": int(sum(row["size_bytes"] for row in panel_inventory)),
            "path_size_inventory_sha256": hashlib.sha256(panel_inventory_bytes).hexdigest(),
            "content_hashing": False,
        }
    if models_receipt_path is not None:
        model_payload = json.loads(models_receipt_path.read_text(encoding="utf-8"))
        selected = [
            row
            for row in model_payload
            if model_id is not None and str(row.get("id")) == model_id
        ]
        if model_id is not None and len(selected) != 1:
            raise ValueError(f"expected one model-receipt row for {model_id!r}, found {len(selected)}")
        receipt["models_receipt"] = {
            "path": str(models_receipt_path.resolve()),
            "sha256": sha256_file(models_receipt_path),
            "model_id": model_id,
        }
        if selected:
            receipt["model_checkpoint_sha256"] = selected[0].get("checkpoint_sha256")
    return receipt


def _condition_variance(
    truth: np.ndarray,
    cell_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return per-dimension mean condition variance and summed energy.

    ``truth`` is [K,C,T,P]; ``cell_mask`` is [C,T].  The same complete K
    condition set is used in every selected cell, so masking cannot create an
    accidental mismatch between condition labels.
    """

    dimension_count = truth.shape[-1]
    if not np.any(cell_mask):
        return np.full(dimension_count, np.nan), np.zeros(dimension_count), 0
    selected = truth[:, cell_mask, :]
    centered = selected - selected.mean(axis=0, keepdims=True)
    per_cell_variance = np.mean(np.square(centered), axis=0)
    return (
        np.mean(per_cell_variance, axis=0),
        np.sum(np.square(centered), axis=(0, 1)),
        int(selected.shape[1]),
    )


def _metric_components(
    prediction: np.ndarray,
    truth: np.ndarray,
    *,
    row_mask: np.ndarray,
    cell_mask: np.ndarray,
    objects: Mapping[str, Sequence[int]],
    mismatched: bool,
) -> dict[str, Any]:
    """Calculate raw and condition-normalized errors on a fixed row subset."""

    if prediction.shape != truth.shape or prediction.ndim != 4:
        raise ValueError("prediction and truth must share [K,C,T,P] shape")
    if row_mask.shape != prediction.shape[:3]:
        raise ValueError("row mask must have shape [K,C,T]")
    if cell_mask.shape != prediction.shape[1:3]:
        raise ValueError("cell mask must have shape [C,T]")
    if not np.array_equal(row_mask, np.broadcast_to(cell_mask[None], row_mask.shape)):
        raise ValueError("masked error metrics require complete K-condition cells")

    k = prediction.shape[0]
    off_diagonal = ~np.eye(k, dtype=bool)
    if mismatched:
        pair_mask = (
            off_diagonal[:, :, None, None]
            & row_mask[:, None, :, :]
            & row_mask[None, :, :, :]
        )
        difference = prediction[:, None, :, :, :] - truth[None, :, :, :, :]
        if np.any(pair_mask):
            selected_difference = difference[pair_mask]
        else:
            selected_difference = np.empty((0, prediction.shape[-1]), dtype=np.float64)
        pair_count = int(np.count_nonzero(pair_mask))
    else:
        difference = prediction - truth
        selected_difference = difference[row_mask]
        pair_count = int(np.count_nonzero(row_mask))

    dimensions = prediction.shape[-1]
    per_dimension_mse: list[float | None] = []
    per_dimension_variance: list[float | None] = []
    per_dimension_rmse: list[float | None] = []
    per_dimension_normalized_rmse: list[float | None] = []
    variance_by_dimension, energy_by_dimension, selected_cells = _condition_variance(
        truth, cell_mask
    )
    for dim in range(dimensions):
        mse = _finite_metric(selected_difference[:, dim])
        variance = float(variance_by_dimension[dim])
        valid_variance = math.isfinite(variance) and variance > readout.PHYSICAL_VARIANCE_EPS
        per_dimension_mse.append(mse)
        per_dimension_variance.append(variance if math.isfinite(variance) else None)
        per_dimension_rmse.append(math.sqrt(mse) if mse is not None else None)
        per_dimension_normalized_rmse.append(
            math.sqrt(mse / variance) if mse is not None and valid_variance else None
        )

    object_metrics: dict[str, Any] = {}
    for name, indices_value in objects.items():
        indices = tuple(int(index) for index in indices_value)
        object_mses = [per_dimension_mse[index] for index in indices]
        object_variances = [per_dimension_variance[index] for index in indices]
        object_valid_mses = [value for value in object_mses if value is not None]
        object_valid_variances = [value for value in object_variances if value is not None]
        object_mse = (
            float(np.mean(object_valid_mses))
            if len(object_valid_mses) == len(indices)
            else None
        )
        object_variance = (
            float(np.mean(object_valid_variances))
            if len(object_valid_variances) == len(indices)
            else None
        )
        normalized_rmse = (
            math.sqrt(object_mse / object_variance)
            if object_mse is not None
            and object_variance is not None
            and object_variance > readout.PHYSICAL_VARIANCE_EPS
            else None
        )
        object_metrics[name] = {
            "dimension_indices": list(indices),
            "query_mse": object_mse,
            "rmse": math.sqrt(object_mse) if object_mse is not None else None,
            "normalization_denominator": object_variance,
            "normalized_rmse": normalized_rmse,
            "rmse_by_dimension": [per_dimension_rmse[index] for index in indices],
            "normalized_rmse_by_dimension": [
                per_dimension_normalized_rmse[index] for index in indices
            ],
        }

    return {
        "query_mse": _finite_metric(selected_difference.reshape(-1)),
        "rmse": (
            math.sqrt(_finite_metric(selected_difference.reshape(-1)))
            if selected_difference.size
            else None
        ),
        "rmse_by_dimension": per_dimension_rmse,
        "normalized_rmse_by_dimension": per_dimension_normalized_rmse,
        "variance_by_dimension": per_dimension_variance,
        "object_metrics": object_metrics,
        "n_selected_rows": pair_count,
        "n_selected_cells": selected_cells,
        "n_comparisons": pair_count,
        "undefined_condition_mismatch": bool(mismatched and k <= 1),
    }


def _paired_canvas_masks(physical: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Build complete-condition PushT masks and pointwise coverage receipts."""

    if physical.ndim != 4 or physical.shape[-1] != 6:
        raise ValueError(f"PushT targets must have shape [K,C,T,6], got {physical.shape}")
    pusher_xy = physical[..., 0:2]
    block_xy = physical[..., 2:4]
    pusher_in = np.all((pusher_xy >= CANVAS_LOW) & (pusher_xy <= CANVAS_HIGH), axis=-1)
    block_in = np.all((block_xy >= CANVAS_LOW) & (block_xy <= CANVAS_HIGH), axis=-1)
    both_in = pusher_in & block_in
    paired_in_cells = np.all(both_in, axis=0)
    paired_in = np.broadcast_to(paired_in_cells[None, :, :], physical.shape[:3]).copy()
    paired_out = ~paired_in
    total_rows = int(np.prod(physical.shape[:3]))
    pusher_in_count = int(pusher_in.sum())
    block_in_count = int(block_in.sum())
    point_both_in_count = int(both_in.sum())
    cell_count = int(np.prod(physical.shape[1:3]))
    paired_in_cell_count = int(paired_in_cells.sum())
    paired_total_rows = int(np.prod(physical.shape[:3]))
    pointwise = {
        "total_rows": total_rows,
        "pusher_center_in_canvas_rows": pusher_in_count,
        "pusher_center_in_canvas_fraction": pusher_in_count / total_rows if total_rows else None,
        "pusher_center_out_of_canvas_rows": total_rows - pusher_in_count,
        "pusher_center_out_of_canvas_fraction": (total_rows - pusher_in_count) / total_rows if total_rows else None,
        "block_center_in_canvas_rows": block_in_count,
        "block_center_in_canvas_fraction": block_in_count / total_rows if total_rows else None,
        "block_center_out_of_canvas_rows": total_rows - block_in_count,
        "block_center_out_of_canvas_fraction": (total_rows - block_in_count) / total_rows if total_rows else None,
        "both_centers_in_canvas_rows": point_both_in_count,
        "both_centers_in_canvas_fraction": point_both_in_count / total_rows if total_rows else None,
    }
    paired = {
        "condition_complete_cells": cell_count,
        "all_conditions_both_centers_in_canvas_cells": paired_in_cell_count,
        "all_conditions_both_centers_in_canvas_fraction": (
            paired_in_cell_count / cell_count if cell_count else None
        ),
        "all_conditions_not_both_centers_in_canvas_cells": cell_count - paired_in_cell_count,
        "all_conditions_not_both_centers_in_canvas_fraction": (
            (cell_count - paired_in_cell_count) / cell_count if cell_count else None
        ),
        "paired_rows": paired_total_rows,
        "in_canvas_paired_rows": int(paired_in.sum()),
        "not_all_in_canvas_paired_rows": int(paired_out.sum()),
    }
    return {
        "all_rows": np.ones(physical.shape[:3], dtype=bool),
        "paired_all_centers_in_canvas": paired_in,
        "paired_not_all_centers_in_canvas": paired_out,
    }, {"pointwise_centers": pointwise, "condition_paired_canvas": paired}


def _condition_signal_energy(
    records: Sequence[readout.FeatureScene],
    masks_by_scene: Mapping[str, Mapping[str, np.ndarray]],
    objects: Mapping[str, Sequence[int]],
    dimensions: Sequence[str],
) -> dict[str, Any]:
    """Report within-condition physical signal energy kept by each cell mask."""

    category_names = tuple(next(iter(masks_by_scene.values())).keys())
    totals: dict[str, np.ndarray] = {
        category: np.zeros(len(dimensions), dtype=np.float64)
        for category in category_names
    }
    full_total = np.zeros(len(dimensions), dtype=np.float64)
    cell_counts = {category: 0 for category in category_names}
    for record in records:
        truth = np.asarray(record.physical, dtype=np.float64)
        _, full_energy, full_cells = _condition_variance(
            truth, np.ones(truth.shape[1:3], dtype=bool)
        )
        full_total += full_energy
        for category, mask in masks_by_scene[record.scene_id].items():
            cell_mask = np.all(mask, axis=0)
            _, energy, count = _condition_variance(truth, cell_mask)
            totals[category] += energy
            cell_counts[category] += count
    def named(values: np.ndarray) -> dict[str, float]:
        return {name: float(values[index]) for index, name in enumerate(dimensions)}
    result: dict[str, Any] = {
        "definition": "sum over K,C,T of (physical_target - mean_over_K_for_same_C_T)^2",
        "all_rows_energy_by_dimension": named(full_total),
        "all_rows_energy_by_object": {
            name: float(np.sum(full_total[list(indices)])) for name, indices in objects.items()
        },
        "categories": {},
    }
    for category in category_names:
        energy = totals[category]
        result["categories"][category] = {
            "selected_cells_across_queries": int(cell_counts[category]),
            "energy_by_dimension": named(energy),
            "fraction_of_all_rows_energy_by_dimension": {
                name: (float(energy[index] / full_total[index]) if full_total[index] > 0 else None)
                for index, name in enumerate(dimensions)
            },
            "energy_by_object": {
                name: float(np.sum(energy[list(indices)])) for name, indices in objects.items()
            },
            "fraction_of_all_rows_energy_by_object": {
                name: (
                    float(np.sum(energy[list(indices)]) / np.sum(full_total[list(indices)]))
                    if np.sum(full_total[list(indices)]) > 0
                    else None
                )
                for name, indices in objects.items()
            },
        }
    return result


def _summary_for_components(
    per_query: Sequence[dict[str, Any]],
    *,
    category: str,
    branch: str,
    objects: Mapping[str, Sequence[int]],
    dimensions: Sequence[str],
    bootstrap_reps: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    groups = sorted({str(row["source_group"]) for row in per_query})
    indices = readout._bootstrap_group_indices(
        groups, bootstrap_reps=bootstrap_reps, seed=bootstrap_seed
    )
    result: dict[str, Any] = {"objects": {}, "dimensions": {}}
    for object_name, dims in objects.items():
        object_index = list(int(x) for x in dims)
        rows = []
        for row in per_query:
            metric = row["categories"][category][branch]
            obj = metric["object_metrics"][object_name]
            rows.append(
                {
                    "source_group": row["source_group"],
                    "metric": {
                        "query_mse": obj["query_mse"],
                        "normalization_denominator": obj["normalization_denominator"],
                    },
                }
            )
        result["objects"][object_name] = _summarize_metric_rows(rows, indices)
        for local_index, dim in enumerate(object_index):
            dim_rows = []
            for row in per_query:
                metric = row["categories"][category][branch]
                mse_by_dim = metric.get("mse_by_dimension", [])
                variance_by_dim = metric.get("variance_by_dimension", [])
                dim_rows.append(
                    {
                        "source_group": row["source_group"],
                        "metric": {
                            "query_mse": mse_by_dim[dim] if dim < len(mse_by_dim) else None,
                            "normalization_denominator": (
                                variance_by_dim[dim] if dim < len(variance_by_dim) else None
                            ),
                        },
                    }
                )
            result["dimensions"][dimensions[dim]] = _summarize_metric_rows(dim_rows, indices)
    return result


def _mse_contribution_decomposition(
    per_query: Sequence[dict[str, Any]],
    *,
    objects: Mapping[str, Sequence[int]],
) -> dict[str, Any]:
    """Verify full MSE equals in/out numerators over one full-data denominator."""

    categories = ("paired_all_centers_in_canvas", "paired_not_all_centers_in_canvas")
    if not per_query or any(
        category not in per_query[0]["categories"] for category in categories
    ):
        return {"status": "not_applicable"}
    output: dict[str, Any] = {
        "definition": "each subset MSE is multiplied by its selected comparison fraction, then averaged over every query; empty subsets contribute zero",
        "same_full_row_denominator": True,
        "branches": {},
    }
    for branch in METRIC_BRANCHES:
        all_values: list[float] = []
        in_contributions: list[float] = []
        out_contributions: list[float] = []
        branch_objects: dict[str, Any] = {}
        object_values: dict[str, dict[str, list[float]]] = {
            name: {"all": [], categories[0]: [], categories[1]: []}
            for name in objects
        }
        for row in per_query:
            all_metric = row["categories"]["all_rows"][branch]
            in_metric = row["categories"][categories[0]][branch]
            out_metric = row["categories"][categories[1]][branch]
            full_count = int(all_metric["n_comparisons"])
            if full_count <= 0 or all_metric["query_mse"] is None:
                continue
            all_values.append(float(all_metric["query_mse"]))
            in_fraction = int(in_metric["n_comparisons"]) / full_count
            out_fraction = int(out_metric["n_comparisons"]) / full_count
            in_contributions.append(
                0.0 if in_metric["query_mse"] is None else float(in_metric["query_mse"]) * in_fraction
            )
            out_contributions.append(
                0.0 if out_metric["query_mse"] is None else float(out_metric["query_mse"]) * out_fraction
            )
            for name in objects:
                all_obj = all_metric["object_metrics"][name]["query_mse"]
                in_obj = in_metric["object_metrics"][name]["query_mse"]
                out_obj = out_metric["object_metrics"][name]["query_mse"]
                if all_obj is not None:
                    object_values[name]["all"].append(float(all_obj))
                    object_values[name][categories[0]].append(
                        0.0 if in_obj is None else float(in_obj) * in_fraction
                    )
                    object_values[name][categories[1]].append(
                        0.0 if out_obj is None else float(out_obj) * out_fraction
                    )
        if not all_values:
            output["branches"][branch] = {"status": "no_valid_queries"}
            continue
        all_mse = float(np.mean(all_values))
        in_mse = float(np.mean(in_contributions))
        out_mse = float(np.mean(out_contributions))
        residual = all_mse - in_mse - out_mse
        branch_objects = {}
        for name, values in object_values.items():
            if not values["all"]:
                branch_objects[name] = {"status": "no_valid_queries"}
                continue
            object_all = float(np.mean(values["all"]))
            object_in = float(np.mean(values[categories[0]]))
            object_out = float(np.mean(values[categories[1]]))
            branch_objects[name] = {
                "all_mse": object_all,
                "in_canvas_contribution_mse": object_in,
                "not_all_in_canvas_contribution_mse": object_out,
                "sum_of_contributions_mse": object_in + object_out,
                "residual_mse": object_all - object_in - object_out,
            }
        output["branches"][branch] = {
            "n_queries": len(all_values),
            "all_mse": all_mse,
            "in_canvas_contribution_mse": in_mse,
            "not_all_in_canvas_contribution_mse": out_mse,
            "sum_of_contributions_mse": in_mse + out_mse,
            "residual_mse": residual,
            "conservation_passed_atol_1e-10": bool(abs(residual) <= 1.0e-10),
            "objects": branch_objects,
        }
    output["all_branches_conserve"] = all(
        item.get("conservation_passed_atol_1e-10", False)
        for item in output["branches"].values()
        if item.get("status") != "no_valid_queries"
    )
    return output


def _summarize_metric_rows(
    rows: Sequence[dict[str, Any]], bootstrap_indices: np.ndarray
) -> dict[str, Any]:
    valid = [row for row in rows if row["metric"]["query_mse"] is not None]
    normalized_valid = [
        row
        for row in valid
        if row["metric"]["normalization_denominator"] is not None
        and math.isfinite(float(row["metric"]["normalization_denominator"]))
        and float(row["metric"]["normalization_denominator"]) >= 0.0
    ]
    total_groups = sorted({str(row["source_group"]) for row in rows})
    zero_variance_queries = sum(
        1
        for row in normalized_valid
        if float(row["metric"]["normalization_denominator"])
        <= readout.PHYSICAL_VARIANCE_EPS
    )
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in valid:
        grouped[str(row["source_group"])].append(row["metric"])

    def aggregate(selected: Sequence[dict[str, Any]], normalized: bool) -> float | None:
        use = [item for item in selected if item.get("query_mse") is not None]
        if not use:
            return None
        numerator = float(sum(float(item["query_mse"]) for item in use))
        if normalized:
            denom_items = [
                item
                for item in use
                if item.get("normalization_denominator") is not None
                and math.isfinite(float(item["normalization_denominator"]))
                and float(item["normalization_denominator"]) >= 0.0
            ]
            if len(denom_items) != len(use):
                return None
            denominator = float(
                sum(float(item["normalization_denominator"]) for item in denom_items)
            )
            if denominator <= readout.PHYSICAL_VARIANCE_EPS:
                return None
        else:
            denominator = float(len(use))
        return math.sqrt(numerator / denominator)

    result = {
        "raw_rmse": aggregate([row["metric"] for row in valid], normalized=False),
        "normalized_rmse": aggregate([row["metric"] for row in normalized_valid], normalized=True),
        "n_queries_total": len(rows),
        "n_queries_valid": len(valid),
        "n_queries_no_selected_rows": len(rows) - len(valid),
        "n_queries_with_zero_condition_variance": zero_variance_queries,
        "n_queries_missing_condition_variance": len(valid) - len(normalized_valid),
        "n_source_groups_total": len(total_groups),
        "bootstrap_replicates": int(len(bootstrap_indices)),
        "bootstrap_ci95_raw_rmse": None,
        "bootstrap_ci95_normalized_rmse": None,
        "aggregation": "equal_query raw MSE then sqrt; sum query MSE / sum query condition variance then sqrt for normalized",
    }
    if len(bootstrap_indices) and total_groups:
        samples_raw: list[float] = []
        samples_norm: list[float] = []
        for draw in bootstrap_indices:
            selected: list[dict[str, Any]] = []
            for group_index in draw:
                selected.extend(grouped.get(total_groups[int(group_index)], []))
            raw = aggregate(selected, normalized=False)
            norm = aggregate(selected, normalized=True)
            if raw is not None:
                samples_raw.append(raw)
            if norm is not None:
                samples_norm.append(norm)
        if samples_raw:
            result["bootstrap_ci95_raw_rmse"] = [
                float(np.percentile(samples_raw, 2.5)),
                float(np.percentile(samples_raw, 97.5)),
            ]
        if samples_norm:
            result["bootstrap_ci95_normalized_rmse"] = [
                float(np.percentile(samples_norm, 2.5)),
                float(np.percentile(samples_norm, 97.5)),
            ]
    return result


def diagnose_records(
    records: Sequence[readout.FeatureScene],
    *,
    task: str,
    reference_result: Mapping[str, Any] | None = None,
    n_folds: int = readout.DEFAULT_FOLDS,
    maximum_samples_per_source: int = readout.DEFAULT_MAX_SAMPLES_PER_SOURCE,
    ridge_alpha: float = readout.DEFAULT_RIDGE_ALPHA,
    projection_dim: int = readout.DEFAULT_PROJECTION_DIM,
    projection_seed: int = readout.DEFAULT_PROJECTION_SEED,
    bootstrap_reps: int = readout.DEFAULT_BOOTSTRAP_REPS,
    seed: int = 20261007,
) -> dict[str, Any]:
    """Replay fixed cross-fit predictions and score full/paired target groups."""

    if not records:
        raise ValueError("at least one feature scene is required")
    family = readout.canonical_task(task)
    target = readout.target_spec(family)
    objects = target_objects(family)
    dimensions = target.names
    if any(record.physical.shape[-1] != len(dimensions) for record in records):
        raise ValueError("physical target width does not match task target schema")
    fold_for_group = readout.assign_source_folds(
        (record.source_group for record in records), n_folds=n_folds, seed=seed
    )
    projector = readout.FixedRandomProjector(
        records[0].target.shape[-1], projection_dim, projection_seed
    )
    masks_by_scene: dict[str, dict[str, np.ndarray]] = {}
    total_target_rows = sum(int(np.prod(record.physical.shape[:3])) for record in records)
    coverage_receipt: dict[str, Any] = {
        "all_rows": {
            "rows": total_target_rows,
            "fraction": 1.0,
            "visibility_claim": "none; full evaluation coverage is not a visible-row claim",
        }
    }
    for record in records:
        if family == "pusht":
            masks, receipt = _paired_canvas_masks(record.physical)
            masks_by_scene[record.scene_id] = masks
            coverage_receipt.setdefault("pointwise_center_counts", {})[record.scene_id] = receipt[
                "pointwise_centers"
            ]
            coverage_receipt.setdefault("paired_canvas_counts", {})[record.scene_id] = receipt[
                "condition_paired_canvas"
            ]
        else:
            masks_by_scene[record.scene_id] = {
                "all_rows": np.ones(record.physical.shape[:3], dtype=bool)
            }
    if family == "pusht":
        point_rows = list(coverage_receipt["pointwise_center_counts"].values())
        paired_cells = list(coverage_receipt["paired_canvas_counts"].values())
        total_point_rows = sum(int(row["total_rows"]) for row in point_rows)
        total_cells = sum(int(row["condition_complete_cells"]) for row in paired_cells)
        pusher_in = sum(int(row["pusher_center_in_canvas_rows"]) for row in point_rows)
        block_in = sum(int(row["block_center_in_canvas_rows"]) for row in point_rows)
        both_in = sum(int(row["both_centers_in_canvas_rows"]) for row in point_rows)
        paired_in_cells = sum(
            int(row["all_conditions_both_centers_in_canvas_cells"]) for row in paired_cells
        )
        coverage_receipt["summary"] = {
            "pointwise_total_rows": total_point_rows,
            "pusher_center_in_canvas_rows": pusher_in,
            "pusher_center_in_canvas_fraction": pusher_in / total_point_rows if total_point_rows else None,
            "block_center_in_canvas_rows": block_in,
            "block_center_in_canvas_fraction": block_in / total_point_rows if total_point_rows else None,
            "both_centers_in_canvas_rows": both_in,
            "both_centers_in_canvas_fraction": both_in / total_point_rows if total_point_rows else None,
            "complete_condition_cells": total_cells,
            "all_conditions_both_centers_in_canvas_cells": paired_in_cells,
            "paired_in_canvas_fraction": paired_in_cells / total_cells if total_cells else None,
            "paired_not_all_in_canvas_cells": total_cells - paired_in_cells,
            "paired_not_all_in_canvas_fraction": (
                (total_cells - paired_in_cells) / total_cells if total_cells else None
            ),
            "in_canvas_is_not_a_full_visibility_claim": True,
        }

    per_query: list[dict[str, Any]] = []
    folds: list[dict[str, Any]] = []
    for fold in range(int(n_folds)):
        train_records = [
            record for record in records if fold_for_group[record.source_group] != fold
        ]
        test_records = [
            record for record in records if fold_for_group[record.source_group] == fold
        ]
        if not train_records or not test_records:
            raise ValueError(f"fold {fold} has empty training or held-out source groups")
        x_train, y_train, sampled = readout._sample_training_rows(
            train_records,
            projector,
            maximum_per_source=maximum_samples_per_source,
        )
        model = readout.fit_ridge_readout(x_train, y_train, alpha=ridge_alpha)
        train_groups = sorted({record.source_group for record in train_records})
        heldout_groups = sorted({record.source_group for record in test_records})
        if not set(train_groups).isdisjoint(heldout_groups):
            raise AssertionError("source-group leakage across cross-fit fold")
        folds.append(
            {
                "fold": fold,
                "train_groups": train_groups,
                "heldout_groups": heldout_groups,
                "train_rows": int(len(x_train)),
                "sampled_rows_by_train_group": sampled,
                "readout_solver": str(model["solver"]),
                "readout_fit_uses": "true_future_target_latent_only",
            }
        )
        for record in test_records:
            pred_latent = projector.transform(record.pred)
            true_latent = projector.transform(record.target)
            prediction = readout.apply_ridge_readout(model, pred_latent).reshape(
                record.physical.shape
            )
            oracle = readout.apply_ridge_readout(model, true_latent).reshape(
                record.physical.shape
            )
            categories: dict[str, Any] = {}
            for category, row_mask in masks_by_scene[record.scene_id].items():
                cell_mask = np.all(row_mask, axis=0)
                branches = {
                    "oracle_true_latent": _metric_components(
                        oracle,
                        record.physical,
                        row_mask=row_mask,
                        cell_mask=cell_mask,
                        objects=objects,
                        mismatched=False,
                    ),
                    "predicted_latent_matched": _metric_components(
                        prediction,
                        record.physical,
                        row_mask=row_mask,
                        cell_mask=cell_mask,
                        objects=objects,
                        mismatched=False,
                    ),
                    "predicted_latent_mismatched": _metric_components(
                        prediction,
                        record.physical,
                        row_mask=row_mask,
                        cell_mask=cell_mask,
                        objects=objects,
                        mismatched=True,
                    ),
                }
                for branch, metric in branches.items():
                    metric["mse_by_dimension"] = [
                        value * value if value is not None else None
                        for value in metric["rmse_by_dimension"]
                    ]
                categories[category] = branches
            per_query.append(
                {
                    "scene_id": record.scene_id,
                    "source_group": record.source_group,
                    "fold": fold,
                    "categories": categories,
                }
            )

    category_names = list(per_query[0]["categories"])
    summaries: dict[str, Any] = {}
    for category in category_names:
        summaries[category] = {}
        for branch in METRIC_BRANCHES:
            summaries[category][branch] = _summary_for_components(
                per_query,
                category=category,
                branch=branch,
                objects=objects,
                dimensions=dimensions,
                bootstrap_reps=bootstrap_reps,
                bootstrap_seed=seed + 101,
            )

    if family == "pusht":
        signal_energy = _condition_signal_energy(records, masks_by_scene, objects, dimensions)
    else:
        signal_energy = {
            "status": "not_computed_from_reference_json",
            "visibility_claim": "none; all rows are scored without an object-visibility mask",
        }

    output: dict[str, Any] = {
        "schema_version": 1,
        "protocol": "observable_target_decomposition_replay_v1",
        "task": task,
        "task_family": family,
        "evaluation_split": "development",
        "claim_scope": (
            "Object and coordinate diagnostics for the existing fixed readout. "
            "No change to the frozen model score and no cross-task unified score."
        ),
        "world_model_training": "none; existing features and frozen encoder predictions are reused",
        "world_model_inference": "none; no encoder or world-model inference was run",
        "target": {
            "names": list(dimensions),
            "objects": {name: list(indices) for name, indices in objects.items()},
            "units": target.units,
            "description": target.description,
        },
        "readout": {
            "kind": "same_as_development_source_group_3fold_crossfit_physical_readout_v2",
            "n_folds": int(n_folds),
            "maximum_samples_per_source": int(maximum_samples_per_source),
            "ridge_alpha": float(ridge_alpha),
            "projection_dim": int(projection_dim),
            "projection_seed": int(projection_seed),
            "fold_assignment_seed": int(seed),
            "bootstrap_replicates": int(bootstrap_reps),
            "bootstrap_seed": int(seed) + 101,
        },
        "n_scenes": len(records),
        "coverage": coverage_receipt,
        "condition_signal_energy": signal_energy,
        "mse_contribution_decomposition": (
            _mse_contribution_decomposition(per_query, objects=objects)
            if family == "pusht"
            else {"status": "not_applicable_without_visibility_masks"}
        ),
        "folds": folds,
        "summaries": summaries,
        "query_metrics": per_query,
        "interpretation": [
            "Raw RMSEs across masks describe different row distributions and are not before/after model improvements or a ranking.",
            "The paired PushT canvas mask is necessary but not sufficient for pixel visibility or target identifiability.",
            "Oracle, predicted-matched, and predicted-mismatched errors are reported side by side; no difference is attributed to a pure model error.",
            "A high oracle error can reflect target information absent from the latent, random projection loss, or linear-readout limits; it does not by itself establish encoder failure.",
        ],
    }
    if reference_result is not None:
        output["reference_check"] = compare_full_metrics_to_reference(
            per_query, reference_result
        )
    return output


def _reference_summary_point_metrics(
    summary: Mapping[str, Any],
) -> tuple[float | None, float | None, str]:
    """Read aggregate metrics, deriving MSE for older RMSE-only summaries."""
    raw_rmse = summary.get("point_estimate")
    rmse = float(raw_rmse) if raw_rmse is not None else None
    raw_mse = summary.get("point_estimate_mse")
    if raw_mse is not None:
        return float(raw_mse), rmse, "stored_point_estimate_mse"
    if rmse is not None:
        return rmse * rmse, rmse, "derived_from_point_estimate_squared"
    return None, None, "unavailable"


def compare_full_metrics_to_reference(
    per_query: Sequence[dict[str, Any]], reference: Mapping[str, Any]
) -> dict[str, Any]:
    """Compare full-row per-query errors with the cached legacy result."""

    if "all_rows" not in per_query[0]["categories"]:
        raise ValueError("diagnostic output lacks all_rows category")
    old_by_scene = {
        str(row["scene_id"]): row for row in reference.get("query_metrics", [])
    }
    if set(old_by_scene) != {str(row["scene_id"]) for row in per_query}:
        raise ValueError("cached reference and replay contain different scene IDs")
    deltas: dict[str, Any] = {}
    max_delta = 0.0
    compared = 0
    for row in per_query:
        scene_id = str(row["scene_id"])
        if scene_id not in old_by_scene:
            raise ValueError(f"cached reference lacks scene {scene_id!r}")
        for branch in METRIC_BRANCHES:
            new_metric = row["categories"]["all_rows"][branch]
            old_metric = old_by_scene[scene_id][branch]
            old_mse = old_metric.get("query_mse")
            new_mse = new_metric.get("query_mse")
            if old_mse is None or new_mse is None:
                if old_mse != new_mse:
                    raise ValueError(f"legacy query metric null mismatch for {scene_id} {branch}")
                continue
            delta = abs(float(old_mse) - float(new_mse))
            deltas[branch] = max(float(deltas.get(branch, 0.0)), delta)
            max_delta = max(max_delta, delta)
            compared += 1
            old_rmse_dims = old_metric.get("rmse_by_dimension")
            new_rmse_dims = new_metric.get("rmse_by_dimension")
            if old_rmse_dims is None or new_rmse_dims is None:
                continue
            dim_delta = np.max(
                np.abs(np.asarray(old_rmse_dims, dtype=np.float64) - np.asarray(new_rmse_dims, dtype=np.float64))
            )
            deltas[branch] = max(float(deltas[branch]), float(dim_delta))
            max_delta = max(max_delta, float(dim_delta))
    aggregate_parity: dict[str, Any] = {}
    old_summaries = reference.get("summaries", {})
    summary_names = {
        "oracle_true_latent": "oracle_true_latent_floor_rmse",
        "predicted_latent_matched": "predicted_latent_matched_rmse",
        "predicted_latent_mismatched": "predicted_latent_mismatched_rmse",
    }
    for branch, summary_name in summary_names.items():
        query_metrics = [
            {
                "query_mse": row["categories"]["all_rows"][branch]["query_mse"],
            }
            for row in per_query
        ]
        new_mse = readout._aggregate_mse(query_metrics, normalized=False)
        new_rmse = math.sqrt(new_mse) if new_mse is not None else None
        old_summary = old_summaries.get(summary_name, {})
        old_mse, old_rmse, old_mse_source = _reference_summary_point_metrics(old_summary)
        mse_delta = (
            abs(float(old_mse) - float(new_mse))
            if old_mse is not None and new_mse is not None
            else None
        )
        rmse_delta = (
            abs(float(old_rmse) - float(new_rmse))
            if old_rmse is not None and new_rmse is not None
            else None
        )
        if mse_delta is not None:
            max_delta = max(max_delta, mse_delta)
        if rmse_delta is not None:
            max_delta = max(max_delta, rmse_delta)
        aggregate_parity[branch] = {
            "reference_values_present": old_mse is not None and old_rmse is not None,
            "reference_summary_key": summary_name,
            "reference_mse": old_mse,
            "reference_mse_source": old_mse_source,
            "replayed_mse": new_mse,
            "absolute_mse_delta": mse_delta,
            "reference_rmse": old_rmse,
            "replayed_rmse": new_rmse,
            "absolute_rmse_delta": rmse_delta,
        }
    parity_values_present = all(
        value["reference_values_present"] for value in aggregate_parity.values()
    )
    return {
        "reference_protocol": reference.get("protocol"),
        "queries_compared": len(per_query),
        "branch_query_metrics_compared": compared,
        "aggregate_parity": aggregate_parity,
        "max_abs_error_or_rmse_by_dimension_delta": max_delta,
        "max_abs_delta_by_branch": deltas,
        "parity_values_present": parity_values_present,
        "passed_atol_1e-8": bool(parity_values_present and max_delta <= 1.0e-8),
        "calibration_or_prediction_difference_used_for_attribution": False,
    }


def decompose_cached_reference(
    reference: Mapping[str, Any], *, task: str, bootstrap_reps: int | None = None
) -> dict[str, Any]:
    """Decompose per-coordinate raw errors from an existing readout JSON.

    This path intentionally makes no visibility claim.  The old JSON contains
    raw ``rmse_by_dimension`` values but no per-coordinate normalization
    denominator or condition-level predictions.
    """

    family = readout.canonical_task(task)
    dimensions = target_dimensions(family)
    objects = target_objects(family)
    rows = reference.get("query_metrics")
    if not isinstance(rows, list) or not rows:
        raise ValueError("reference result must contain non-empty query_metrics")
    groups = sorted({str(row.get("source_group", row["scene_id"])) for row in rows})
    crossfit = reference.get("source_group_crossfit", {})
    reps = int(
        crossfit.get("bootstrap_replicates", 0)
        if bootstrap_reps is None
        else bootstrap_reps
    )
    bootstrap_seed = int(crossfit.get("bootstrap_seed", 101))
    indices = readout._bootstrap_group_indices(groups, bootstrap_reps=reps, seed=bootstrap_seed)
    summaries: dict[str, Any] = {}
    per_query: list[dict[str, Any]] = []
    full_aggregate_parity: dict[str, Any] = {}
    old_summaries = reference.get("summaries", {})
    summary_names = {
        "oracle_true_latent": "oracle_true_latent_floor_rmse",
        "predicted_latent_matched": "predicted_latent_matched_rmse",
        "predicted_latent_mismatched": "predicted_latent_mismatched_rmse",
    }
    for branch in METRIC_BRANCHES:
        branch_rows = []
        for row in rows:
            metric = row[branch]
            rmse_by_dim = metric.get("rmse_by_dimension")
            if rmse_by_dim is None:
                mse_by_dim = [None] * len(dimensions)
            else:
                if len(rmse_by_dim) != len(dimensions):
                    raise ValueError(
                        f"{row['scene_id']} {branch}: expected {len(dimensions)} coordinate errors, got {len(rmse_by_dim)}"
                    )
                mse_by_dim = [
                    float(value) ** 2 if value is not None else None for value in rmse_by_dim
                ]
            object_values = {}
            for name, dims in objects.items():
                values = [mse_by_dim[int(index)] for index in dims]
                object_values[name] = (
                    float(np.mean(values)) if all(value is not None for value in values) else None
                )
            branch_rows.append(
                {
                    "scene_id": str(row["scene_id"]),
                    "source_group": str(row.get("source_group", row["scene_id"])),
                    "fold": row.get("fold"),
                    "mse_by_dimension": mse_by_dim,
                    "mse_by_object": object_values,
                }
            )
        summaries[branch] = {"objects": {}, "dimensions": {}}
        recomposed_query_mse: list[float] = []
        query_delta = 0.0
        query_nulls_match = True
        original_by_scene = {str(row["scene_id"]): row[branch] for row in rows}
        for row in branch_rows:
            mse_by_dim = row["mse_by_dimension"]
            old_query_mse = original_by_scene[row["scene_id"]].get("query_mse")
            if all(value is not None for value in mse_by_dim):
                full_query_mse = float(np.mean(mse_by_dim))
            else:
                full_query_mse = None
            if old_query_mse is None or full_query_mse is None:
                if old_query_mse != full_query_mse:
                    query_nulls_match = False
                continue
            recomposed_query_mse.append(full_query_mse)
            query_delta = max(query_delta, abs(float(old_query_mse) - full_query_mse))
        recomposed_mse = (
            float(np.mean(recomposed_query_mse)) if recomposed_query_mse else None
        )
        recomposed_rmse = math.sqrt(recomposed_mse) if recomposed_mse is not None else None
        old_summary = old_summaries.get(summary_names[branch], {})
        old_mse, old_rmse, old_mse_source = _reference_summary_point_metrics(old_summary)
        mse_delta = (
            abs(float(old_mse) - recomposed_mse)
            if old_mse is not None and recomposed_mse is not None
            else None
        )
        rmse_delta = (
            abs(float(old_rmse) - recomposed_rmse)
            if old_rmse is not None and recomposed_rmse is not None
            else None
        )
        full_aggregate_parity[branch] = {
            "reference_summary_key": summary_names[branch],
            "reference_values_present": old_mse is not None and old_rmse is not None,
            "reference_mse": old_mse,
            "reference_mse_source": old_mse_source,
            "recomposed_mse": recomposed_mse,
            "absolute_mse_delta": mse_delta,
            "reference_rmse": old_rmse,
            "recomposed_rmse": recomposed_rmse,
            "absolute_rmse_delta": rmse_delta,
            "max_abs_query_mse_delta": query_delta,
            "query_nulls_match": query_nulls_match,
        }
        for name in objects:
            metric_rows = [
                {
                    "source_group": row["source_group"],
                    "metric": {"query_mse": row["mse_by_object"][name], "normalization_denominator": None},
                }
                for row in branch_rows
            ]
            summaries[branch]["objects"][name] = _summarize_metric_rows(metric_rows, indices)
            summaries[branch]["objects"][name]["normalized_rmse"] = None
        for dim, name in enumerate(dimensions):
            metric_rows = [
                {
                    "source_group": row["source_group"],
                    "metric": {"query_mse": row["mse_by_dimension"][dim], "normalization_denominator": None},
                }
                for row in branch_rows
            ]
            summaries[branch]["dimensions"][name] = _summarize_metric_rows(metric_rows, indices)
            summaries[branch]["dimensions"][name]["normalized_rmse"] = None
        per_query.append({"branch": branch, "rows": branch_rows})
    return {
        "schema_version": 1,
        "protocol": "observable_target_cached_per_dimension_decomposition_v1",
        "task": task,
        "task_family": family,
        "source_protocol": reference.get("protocol"),
        "world_model_training": "none; no model weights were changed",
        "world_model_inference": "none; cached readout JSON was decomposed without encoder inference",
        "target": {
            "names": list(dimensions),
            "objects": {name: list(indices) for name, indices in objects.items()},
            "units": readout.target_spec(family).units,
        },
        "coverage": {
            "full_readout_query_count": len(rows),
            "scored_full_rows": True,
            "visibility_mask_applied": False,
            "visibility_status": "not_assessed; full-row coverage is not a visible-row claim",
        },
        "normalization": {
            "per_coordinate_normalized_rmse": "unavailable_from_cached_json; per-coordinate variance is not stored",
            "condition_signal_energy": "unavailable_from_cached_json",
        },
        "full_aggregate_parity": {
            "branches": full_aggregate_parity,
            "passed_atol_1e-8": bool(
                all(row["reference_values_present"] for row in full_aggregate_parity.values())
                and all(
                    row["absolute_mse_delta"] is not None
                    and row["absolute_mse_delta"] <= 1.0e-8
                    and row["absolute_rmse_delta"] is not None
                    and row["absolute_rmse_delta"] <= 1.0e-8
                    and row["max_abs_query_mse_delta"] <= 1.0e-8
                    and row["query_nulls_match"]
                    for row in full_aggregate_parity.values()
                )
            ),
        },
        "summaries": summaries,
        "query_metrics_by_branch": per_query,
        "interpretation": [
            "All cached rows are decomposed; this does not mean all targets are visible or identifiable from a frame.",
            "The reference JSON lacks per-coordinate normalization denominators and condition-level prediction errors.",
            "Raw errors are a decomposition of the fixed old readout and do not replace its task score.",
        ],
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--reference-result", required=True, type=Path)
    parser.add_argument("--features-dir", type=Path)
    parser.add_argument("--panels-dir", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--folds", type=int)
    parser.add_argument(
        "--max-samples-per-source",
        type=int,
    )
    parser.add_argument("--ridge-alpha", type=float)
    parser.add_argument("--projection-dim", type=int)
    parser.add_argument("--projection-seed", type=int)
    parser.add_argument("--bootstrap-reps", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--models-receipt", type=Path)
    parser.add_argument("--model-id")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    reference = json.loads(args.reference_result.read_text(encoding="utf-8"))
    family = readout.canonical_task(args.task)
    crossfit_config = reference.get("source_group_crossfit", {})
    readout_config = reference.get("readout", {})
    folds = args.folds if args.folds is not None else int(crossfit_config.get("n_folds", readout.DEFAULT_FOLDS))
    maximum_samples = (
        args.max_samples_per_source
        if args.max_samples_per_source is not None
        else int(crossfit_config.get("maximum_samples_per_source", readout.DEFAULT_MAX_SAMPLES_PER_SOURCE))
    )
    ridge_alpha = (
        args.ridge_alpha
        if args.ridge_alpha is not None
        else float(readout_config.get("alpha", readout.DEFAULT_RIDGE_ALPHA))
    )
    projection_dim = (
        args.projection_dim
        if args.projection_dim is not None
        else int(readout_config.get("output_dim", readout.DEFAULT_PROJECTION_DIM))
    )
    projection_seed = (
        args.projection_seed
        if args.projection_seed is not None
        else int(readout_config.get("projection_seed", readout.DEFAULT_PROJECTION_SEED))
    )
    bootstrap_reps = (
        args.bootstrap_reps
        if args.bootstrap_reps is not None
        else int(crossfit_config.get("bootstrap_replicates", 0))
    )
    seed = (
        args.seed
        if args.seed is not None
        else int(crossfit_config.get("bootstrap_seed", 20261108)) - 101
    )
    if family == "pusht":
        if args.features_dir is None or args.panels_dir is None:
            raise SystemExit("PushT requires --features-dir and --panels-dir for paired-mask replay")
        records = readout.load_feature_scenes(
            args.features_dir,
            args.panels_dir,
            args.task,
            projection_dim=projection_dim,
            projection_seed=projection_seed,
        )
        result = diagnose_records(
            records,
            task=args.task,
            reference_result=reference,
            n_folds=folds,
            maximum_samples_per_source=maximum_samples,
            ridge_alpha=ridge_alpha,
            projection_dim=projection_dim,
            projection_seed=projection_seed,
            bootstrap_reps=bootstrap_reps,
            seed=seed,
        )
        if not result["reference_check"]["passed_atol_1e-8"]:
            raise SystemExit(
                "replayed full-row errors did not match cached physical readout result: "
                + json.dumps(result["reference_check"], sort_keys=True)
            )
        result["source_files"] = source_receipt(
            reference_path=args.reference_result,
            task=args.task,
            features_dir=args.features_dir,
            panels_dir=args.panels_dir,
            records=records,
            models_receipt_path=args.models_receipt,
            model_id=args.model_id,
        )
    else:
        result = decompose_cached_reference(
            reference, task=args.task, bootstrap_reps=args.bootstrap_reps
        )
        if not result["full_aggregate_parity"]["passed_atol_1e-8"]:
            raise SystemExit(
                "coordinate decomposition did not conserve the cached full aggregate: "
                + json.dumps(result["full_aggregate_parity"], sort_keys=True)
            )
        result["source_files"] = source_receipt(
            reference_path=args.reference_result,
            task=args.task,
            models_receipt_path=args.models_receipt,
            model_id=args.model_id,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(_json_safe(result), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
