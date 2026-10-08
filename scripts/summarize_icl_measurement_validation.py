"""Aggregate the history and auxiliary physical-readout validation outputs.

This is a reporting step only.  It reads frozen per-query JSON rows and the
cross-fitted physical-readout JSON files; it never loads a checkpoint.  Run
ratios are formed first and then averaged equally over training seeds.  Source
groups, rather than conditions/candidates/horizons, are the bootstrap unit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


HORIZONS = (5, 10, 15, 20, 25)
DEFAULT_BOOTSTRAP_REPS = 4000
DEFAULT_BOOTSTRAP_SEED = 20260930
EXPECTED_RUNS = 136
EXPECTED_ROWS = 86


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _manifest(panel_root: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    path = Path(panel_root) / "manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("scenes") or payload.get("pairs") or payload.get("queries")
    if not isinstance(entries, list):
        raise ValueError(f"Panel manifest has no scene list: {path}")
    indexed: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        scene = entry.get("scene_id", entry.get("pair_id", entry.get("query_id")))
        if scene is not None:
            indexed[str(scene)] = dict(entry)
    if len(indexed) != len(entries):
        raise ValueError(f"Panel has missing or duplicate scene ids: {path}")
    return payload, indexed


def _load_bootstrap_groups(path: Path, task: str) -> dict[str, str]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, Mapping) and isinstance(payload.get(task), Mapping):
        payload = payload[task]
    if not isinstance(payload, Mapping):
        raise ValueError(f"Bootstrap map for {task} is not an object: {path}")
    return {str(key): str(value) for key, value in payload.items()}


def _source_group(
    scene_id: str,
    entry: Mapping[str, Any],
    explicit: Mapping[str, str],
) -> str:
    if scene_id in explicit:
        return explicit[scene_id]
    for key in ("bootstrap_cluster", "source_group", "source_sha256", "source_id", "pair_id"):
        if key in entry and str(entry[key]) not in {"", "None", "nan"}:
            return f"{key}:{entry[key]}"
    return f"scene:{scene_id}"


def _find_physical_file(physical_root: Path, model_id: str) -> Path | None:
    candidates = (
        physical_root / f"{model_id}.json",
        physical_root / model_id / "result.json",
        physical_root / f"{model_id.replace('/', '__')}.json",
        physical_root / model_id / "physical.json",
    )
    return next((path for path in candidates if path.is_file()), None)


def _receipt_path(result_dir: Path) -> Path | None:
    direct = result_dir / "receipt.json"
    if direct.is_file():
        return direct
    receipts = sorted(result_dir.glob("shard*_receipt.json"))
    return receipts[0] if len(receipts) == 1 else None


def _history_query(row: Mapping[str, Any], group: str) -> dict[str, Any]:
    if tuple(row.get("horizons", ())) != HORIZONS:
        raise ValueError(f"{row.get('scene_id')} has unexpected horizons")
    conditions = int(row["conditions"])
    candidates = int(row["candidates"])
    size = conditions * candidates
    if size <= 0:
        raise ValueError(f"{row.get('scene_id')} has invalid K*C={size}")
    matched_object = row.get("matched") or {}
    wrong_object = row.get("wrong_allother") or row.get("wrong") or {}
    matched_value = matched_object.get("energy_mean")
    if matched_value is None:
        matched_value = np.asarray(matched_object["energy_sum"], dtype=np.float64) / size
    wrong_value = wrong_object.get("energy_mean")
    if wrong_value is None:
        wrong_value = np.asarray(wrong_object["energy_sum"], dtype=np.float64) / size
    matched = np.asarray(matched_value, dtype=np.float64)
    wrong = np.asarray(wrong_value, dtype=np.float64)
    separation = np.asarray(row["target_separation_energy"], dtype=np.float64) / size
    if matched.shape != (5,) or wrong.shape != (5,) or separation.shape != (5,):
        raise ValueError(f"{row.get('scene_id')} has malformed history energy arrays")
    if not np.isfinite(np.concatenate([matched, wrong, separation])).all():
        raise ValueError(f"{row.get('scene_id')} has non-finite history energies")
    if np.any(matched < 0) or np.any(wrong < 0) or np.any(separation < 0):
        raise ValueError(f"{row.get('scene_id')} has negative history energies")
    return {
        "scene_id": str(row["scene_id"]),
        "source_group": group,
        "matched_by_horizon": matched,
        "wrong_by_horizon": wrong,
        "B_by_horizon": separation,
        # Bq is the all-horizon target-separation energy.  Every horizon uses
        # this same denominator, matching the frozen coverage-v2 contract.
        "matched": float(matched.mean()),
        "wrong": float(wrong.mean()),
        "Bq": float(separation.mean()),
    }


def read_history_run(
    spec: Mapping[str, Any],
    result_dir: Path,
    panel_root: Path,
    bootstrap_map_path: Path,
) -> dict[str, Any]:
    manifest, entries = _manifest(panel_root)
    receipt_path = _receipt_path(result_dir)
    if receipt_path is None:
        raise FileNotFoundError(f"No unambiguous receipt in {result_dir}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("state_hash_before") != receipt.get("state_hash_after"):
        raise ValueError(f"Frozen state changed in {receipt_path}")
    if receipt.get("no_training") is not True:
        raise ValueError(f"Receipt is not marked no_training: {receipt_path}")
    if receipt.get("checkpoint_sha256") != spec["checkpoint_sha256"]:
        raise ValueError(f"Checkpoint identity mismatch: {receipt_path}")
    panel_manifest = panel_root / "manifest.json"
    if receipt.get("panel_sha256") != file_sha256(panel_manifest):
        raise ValueError(f"Panel identity mismatch: {receipt_path}")
    groups = _load_bootstrap_groups(bootstrap_map_path, str(spec["task"]))
    rows: list[dict[str, Any]] = []
    for filename, expected_hash in (receipt.get("entry_hashes") or {}).items():
        row_path = result_dir / str(filename)
        if not row_path.is_file() or file_sha256(row_path) != expected_hash:
            raise ValueError(f"Query row hash mismatch: {row_path}")
        row = json.loads(row_path.read_text(encoding="utf-8"))
        scene_id = str(row.get("scene_id"))
        if scene_id not in entries:
            raise ValueError(f"Unknown query scene {scene_id} in {row_path}")
        entry = entries[scene_id]
        source_hash = entry.get("sha256")
        if row.get("checkpoint_sha256") != spec["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint identity mismatch: {row_path}")
        if source_hash is not None and row.get("source_sha256") != source_hash:
            raise ValueError(f"Source identity mismatch: {row_path}")
        group = _source_group(scene_id, entry, groups)
        rows.append(_history_query(row, group))
    rows.sort(key=lambda row: row["scene_id"])
    if len({row["scene_id"] for row in rows}) != len(rows):
        raise ValueError(f"Duplicate query rows in {result_dir}")
    return {
        "id": str(spec["id"]),
        "task": str(spec["task"]),
        "family": str(spec["family"]),
        "regime": str(spec["regime"]),
        "training_seed": int(spec["training_seed"]),
        "checkpoint_sha256": str(spec["checkpoint_sha256"]),
        "panel_sha256": file_sha256(panel_manifest),
        "receipt_sha256": file_sha256(receipt_path),
        "expected_queries": len(entries),
        "queries": rows,
        "receipt_scenes": int(receipt.get("scenes", len(rows))),
    }


def _group_rows(rows: Sequence[Mapping[str, Any]], key: str) -> tuple[list[str], np.ndarray]:
    groups = sorted({str(row["source_group"]) for row in rows})
    values = np.asarray(
        [
            np.sum(
                [float(row[key]) for row in rows if row["source_group"] == group],
                dtype=np.float64,
            )
            for group in groups
        ],
        dtype=np.float64,
    )
    return groups, values


def _group_horizon_rows(
    rows: Sequence[Mapping[str, Any]], key: str, groups: Sequence[str]
) -> np.ndarray:
    return np.asarray(
        [
            np.sum(
                [np.asarray(row[key], dtype=np.float64) for row in rows if row["source_group"] == group],
                axis=0,
                dtype=np.float64,
            )
            for group in groups
        ],
        dtype=np.float64,
    )


def _ci(values: np.ndarray) -> list[float] | None:
    if values.size == 0:
        return None
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def _bootstrap_indices(group_count: int, reps: int, seed: int) -> np.ndarray:
    if group_count <= 0:
        raise ValueError("No source groups available")
    if reps <= 0:
        return np.empty((0, group_count), dtype=np.int64)
    return np.random.default_rng(int(seed)).integers(
        0, group_count, size=(int(reps), group_count), dtype=np.int64
    )


def _run_history_values(run: Mapping[str, Any], groups: Sequence[str]) -> dict[str, Any]:
    rows = run["queries"]
    by_id = {row["scene_id"]: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("Duplicate history query ids")
    b = np.asarray(
        [
            np.sum(
                [row["Bq"] for row in rows if row["source_group"] == group],
                dtype=np.float64,
            )
            for group in groups
        ]
    )
    matched = np.asarray(
        [
            np.sum(
                [row["matched"] for row in rows if row["source_group"] == group],
                dtype=np.float64,
            )
            for group in groups
        ]
    )
    wrong = np.asarray(
        [
            np.sum(
                [row["wrong"] for row in rows if row["source_group"] == group],
                dtype=np.float64,
            )
            for group in groups
        ]
    )
    matched_h = _group_horizon_rows(rows, "matched_by_horizon", groups)
    wrong_h = _group_horizon_rows(rows, "wrong_by_horizon", groups)
    return {"B": b, "matched": matched, "wrong": wrong, "matched_h": matched_h, "wrong_h": wrong_h}


def _ratio(numerator: np.ndarray, denominator: np.ndarray) -> float:
    n = float(np.sum(numerator, dtype=np.float64))
    d = float(np.sum(denominator, dtype=np.float64))
    if not np.isfinite([n, d]).all() or n < 0 or d <= 0:
        raise ValueError(f"Invalid normalized error numerator/denominator: {n}, {d}")
    return n / d


def aggregate_history(
    runs: Sequence[Mapping[str, Any]],
    indices: np.ndarray,
) -> dict[str, Any]:
    if not runs:
        raise ValueError("No runs for history aggregation")
    query_ids = [row["scene_id"] for row in runs[0]["queries"]]
    groups = sorted({row["source_group"] for row in runs[0]["queries"]})
    for run in runs:
        if [row["scene_id"] for row in run["queries"]] != query_ids:
            raise ValueError("Training repetitions do not share query ids")
        if sorted({row["source_group"] for row in run["queries"]}) != groups:
            raise ValueError("Training repetitions do not share source groups")
    per_run = [_run_history_values(run, groups) for run in runs]
    point_m = np.asarray([_ratio(item["matched"], item["B"]) for item in per_run])
    point_w = np.asarray([_ratio(item["wrong"], item["B"]) for item in per_run])
    point_g = point_w - point_m
    point_score = 100.0 * (1.0 - point_m)
    denominator = np.asarray([np.sum(item["B"]) for item in per_run])
    point_m_h = np.asarray(
        [item["matched_h"].sum(axis=0) / item["B"].sum() for item in per_run]
    )
    point_w_h = np.asarray(
        [item["wrong_h"].sum(axis=0) / item["B"].sum() for item in per_run]
    )
    point_g_h = point_w_h - point_m_h
    if indices.size:
        boot_m = []
        boot_w = []
        boot_m_h = []
        boot_w_h = []
        for item in per_run:
            denominator_draw = item["B"][indices].sum(axis=1)
            boot_m.append(item["matched"][indices].sum(axis=1) / denominator_draw)
            boot_w.append(item["wrong"][indices].sum(axis=1) / denominator_draw)
            boot_m_h.append(item["matched_h"][indices].sum(axis=1) / denominator_draw[:, None])
            boot_w_h.append(item["wrong_h"][indices].sum(axis=1) / denominator_draw[:, None])
        boot_m_array = np.mean(np.stack(boot_m), axis=0)
        boot_w_array = np.mean(np.stack(boot_w), axis=0)
        boot_m_h_array = np.mean(np.stack(boot_m_h), axis=0)
        boot_w_h_array = np.mean(np.stack(boot_w_h), axis=0)
    else:
        boot_m_array = boot_w_array = np.empty(0, dtype=np.float64)
        boot_m_h_array = boot_w_h_array = np.empty((0, 5), dtype=np.float64)
    gain_boot = boot_w_array - boot_m_array
    gain_h_boot = boot_w_h_array - boot_m_h_array
    return {
        "training_repetitions": len(runs),
        "training_seeds": [int(run["training_seed"]) for run in runs],
        "n_queries": len(query_ids),
        "n_source_groups": len(groups),
        "source_groups": groups,
        "denominator_B_total_mean_over_runs": float(denominator.mean()),
        "matched_error_ratio": float(point_m.mean()),
        "wrong_error_ratio": float(point_w.mean()),
        "gain": float(point_g.mean()),
        "score": float(point_score.mean()),
        "matched_error_ratio_by_horizon": point_m_h.mean(axis=0).tolist(),
        "wrong_error_ratio_by_horizon": point_w_h.mean(axis=0).tolist(),
        "gain_by_horizon": point_g_h.mean(axis=0).tolist(),
        "bootstrap": {
            "matched_error_ratio_ci95": _ci(boot_m_array),
            "wrong_error_ratio_ci95": _ci(boot_w_array),
            "gain_ci95": _ci(gain_boot),
            "score_ci95": _ci(100.0 * (1.0 - boot_m_array)),
            "matched_error_ratio_by_horizon_ci95": (
                [_ci(boot_m_h_array[:, index]) for index in range(5)]
                if boot_m_h_array.size
                else None
            ),
            "wrong_error_ratio_by_horizon_ci95": (
                [_ci(boot_w_h_array[:, index]) for index in range(5)]
                if boot_w_h_array.size
                else None
            ),
            "gain_by_horizon_ci95": (
                [_ci(gain_h_boot[:, index]) for index in range(5)]
                if gain_h_boot.size
                else None
            ),
            "unit": "source_group paired bootstrap; same draws for matched, wrong and gain",
        },
    }


def _physical_values(payload: Mapping[str, Any], groups_by_scene: Mapping[str, str]) -> dict[str, Any]:
    by_scene = {str(row["scene_id"]): row for row in payload.get("query_metrics", [])}
    if len(by_scene) != len(payload.get("query_metrics", [])):
        raise ValueError("Duplicate physical query metrics")
    metric_names = ("oracle", "pred", "wrong")
    values: dict[str, dict[str, dict[str, float | int]]] = {}
    for scene_id, row in by_scene.items():
        if scene_id not in groups_by_scene:
            raise ValueError(f"Physical query is absent from history rows: {scene_id}")
        group = groups_by_scene[scene_id]
        values.setdefault(
            group,
            {
                name: {"mse": 0.0, "denominator": 0.0, "count": 0}
                for name in metric_names
            },
        )
        names = {
            "oracle": "oracle_true_latent",
            "pred": "predicted_latent_matched",
            "wrong": "predicted_latent_mismatched",
        }
        for short, long_name in names.items():
            metric = row[long_name]
            # The producer's normalized_rmse is sqrt(query_mse / Bq).  Do not
            # average that derived quantity: retain its query_mse and Bq and
            # pool those sums at the same level as validate_physical_readout.
            query_mse = metric.get("query_mse", metric.get("mse"))
            if query_mse is None:
                continue
            denominator = metric.get(
                "normalization_denominator",
                metric.get("physical_variance_across_conditions"),
            )
            if denominator is None:
                raise ValueError(f"{scene_id}/{long_name} lacks normalization denominator")
            stats = values[group][short]
            stats["mse"] += float(query_mse)
            stats["denominator"] += float(denominator)
            stats["count"] += 1

    groups = sorted(values)
    totals: dict[str, dict[str, float | int]] = {}
    for name in metric_names:
        totals[name] = {
            key: sum(float(values[group][name][key]) for group in groups)
            for key in ("mse", "denominator", "count")
        }
    return {"groups": groups, "group_stats": values, "totals": totals}


def _physical_metric(stats: Mapping[str, Any], *, normalized: bool) -> float | None:
    count = int(stats["count"])
    mse = float(stats["mse"])
    if count <= 0:
        return None
    if normalized:
        denominator = float(stats["denominator"])
        if denominator <= 1.0e-12:
            return None
        return float(np.sqrt(mse / denominator))
    return float(np.sqrt(mse / float(count)))


def _physical_normalized_mse(stats: Mapping[str, Any]) -> float | None:
    if int(stats["count"]) <= 0 or float(stats["denominator"]) <= 1.0e-12:
        return None
    return float(float(stats["mse"]) / float(stats["denominator"]))


def aggregate_physical(
    physical_payloads: Sequence[Mapping[str, Any]],
    groups_by_scene: Mapping[str, str],
    indices: np.ndarray,
) -> dict[str, Any]:
    if not physical_payloads:
        return {
            "available": False,
            "reason": "physical readout files unavailable",
            "training_repetitions": 0,
        }
    per_run = [_physical_values(payload, groups_by_scene) for payload in physical_payloads]
    groups = per_run[0]["groups"]
    if any(item["groups"] != groups for item in per_run):
        raise ValueError("Physical readout runs do not share source groups")
    metric_names = ("oracle", "pred", "wrong")
    # Keep source-group bootstrap aggregation in dense arrays.  A direct
    # Python loop over 4,000 draws, ~300 groups, three metrics, and all seeds
    # needlessly dominates this reporting-only step.  The last axis is
    # [query_mse, normalization_denominator, query_count].
    stats_by_metric: dict[str, np.ndarray] = {}
    for name in metric_names:
        stats_by_metric[name] = np.asarray(
            [
                [
                    [
                        float(run["group_stats"][group][name]["mse"]),
                        float(run["group_stats"][group][name]["denominator"]),
                        float(run["group_stats"][group][name]["count"]),
                    ]
                    for group in groups
                ]
                for run in per_run
            ],
            dtype=np.float64,
        )
    points: dict[str, float | None] = {}
    normalized_mse_points: dict[str, float | None] = {}
    boot: dict[str, np.ndarray] = {}
    boot_normalized_mse: dict[str, np.ndarray] = {}
    for name in metric_names:
        raw_values: list[float] = []
        normalized_values: list[float] = []
        normalized_mse_values: list[float] = []
        raw_boot: list[np.ndarray] = []
        normalized_boot: list[np.ndarray] = []
        normalized_mse_boot: list[np.ndarray] = []
        for run in per_run:
            total = run["totals"][name]
            raw = _physical_metric(total, normalized=False)
            normalized = _physical_metric(total, normalized=True)
            normalized_mse = _physical_normalized_mse(total)
            if raw is not None:
                raw_values.append(raw)
            if normalized is not None:
                normalized_values.append(normalized)
            if normalized_mse is not None:
                normalized_mse_values.append(normalized_mse)
        if indices.size:
            # Chunk the [seed, bootstrap, source_group, statistic] gather so
            # the vectorization does not materialize the full 4-D tensor.
            metric_stats = stats_by_metric[name]
            chunk_size = 256
            raw_chunks: list[np.ndarray] = []
            normalized_chunks: list[np.ndarray] = []
            normalized_mse_chunks: list[np.ndarray] = []
            for start in range(0, len(indices), chunk_size):
                sampled = metric_stats[:, indices[start : start + chunk_size], :]
                totals = np.sum(sampled, axis=2, dtype=np.float64)
                mse = totals[..., 0]
                denominator = totals[..., 1]
                count = totals[..., 2]
                raw_values_chunk = np.full(mse.shape, np.nan, dtype=np.float64)
                np.divide(mse, count, out=raw_values_chunk, where=count > 0.0)
                raw_values_chunk = np.sqrt(raw_values_chunk)
                normalized_values_chunk = np.full(mse.shape, np.nan, dtype=np.float64)
                np.divide(
                    mse,
                    denominator,
                    out=normalized_values_chunk,
                    where=denominator > 1.0e-12,
                )
                normalized_values_chunk = np.sqrt(normalized_values_chunk)
                normalized_mse_chunk = np.full(mse.shape, np.nan, dtype=np.float64)
                np.divide(
                    mse,
                    denominator,
                    out=normalized_mse_chunk,
                    where=denominator > 1.0e-12,
                )
                raw_chunks.append(np.mean(raw_values_chunk, axis=0))
                normalized_chunks.append(np.mean(normalized_values_chunk, axis=0))
                normalized_mse_chunks.append(np.mean(normalized_mse_chunk, axis=0))
            raw_boot = [np.concatenate(raw_chunks)]
            normalized_boot = [np.concatenate(normalized_chunks)]
            normalized_mse_boot = [np.concatenate(normalized_mse_chunks)]
        # Each checkpoint/seed contributes equally to the point estimate.
        points[f"{name}_rmse"] = None if len(raw_values) != len(per_run) else float(np.mean(raw_values))
        points[f"{name}_normalized_rmse"] = (
            None if len(normalized_values) != len(per_run) else float(np.mean(normalized_values))
        )
        normalized_mse_points[name] = (
            None
            if len(normalized_mse_values) != len(per_run)
            else float(np.mean(normalized_mse_values))
        )
        boot[f"{name}_rmse"] = (
            np.mean(np.stack(raw_boot), axis=0)
            if raw_boot and all(np.isfinite(value).all() for value in raw_boot)
            else np.empty(0, dtype=np.float64)
        )
        boot[f"{name}_normalized_rmse"] = (
            np.mean(np.stack(normalized_boot), axis=0)
            if normalized_boot and all(np.isfinite(value).all() for value in normalized_boot)
            else np.full(len(indices), np.nan, dtype=np.float64)
        )
        boot_normalized_mse[name] = (
            np.mean(np.stack(normalized_mse_boot), axis=0)
            if normalized_mse_boot and all(np.isfinite(value).all() for value in normalized_mse_boot)
            else np.full(len(indices), np.nan, dtype=np.float64)
        )

    def difference(
        wrong_key: str, pred_key: str, values: Mapping[str, float | None], arrays: Mapping[str, np.ndarray]
    ) -> tuple[float | None, np.ndarray]:
        if values[wrong_key] is None or values[pred_key] is None:
            return None, np.empty(0, dtype=np.float64)
        if not arrays[wrong_key].size or not arrays[pred_key].size:
            return values[wrong_key] - values[pred_key], np.empty(0, dtype=np.float64)
        pair = arrays[wrong_key] - arrays[pred_key]
        return values[wrong_key] - values[pred_key], pair[np.isfinite(pair)]

    gain_norm, gain_norm_boot = difference(
        "wrong_normalized_rmse", "pred_normalized_rmse", points, boot
    )
    gain_rmse, gain_rmse_boot = difference("wrong_rmse", "pred_rmse", points, boot)
    gain_normalized_mse, gain_normalized_mse_boot = difference(
        "wrong", "pred", normalized_mse_points, boot_normalized_mse
    )

    def ci(array: np.ndarray) -> list[float] | None:
        finite = array[np.isfinite(array)]
        return (
            None
            if not finite.size
            else [float(np.percentile(finite, 2.5)), float(np.percentile(finite, 97.5))]
        )

    return {
        "available": True,
        "training_repetitions": len(physical_payloads),
        "n_source_groups": len(groups),
        "source_groups": groups,
        "calibration_scheme": str(physical_payloads[0].get("protocol", "unknown")),
        "calibration_readout": physical_payloads[0].get("readout", {}),
        "calibration_claim": (
            "Auxiliary Development source-group cross-fitted readout; its oracle error is "
            "retained. This is not a common physical benchmark; it does not by itself establish comparable physical accuracy across encoders."
        ),
        "oracle_readout_normalized_rmse": points["oracle_normalized_rmse"],
        "predicted_normalized_rmse": points["pred_normalized_rmse"],
        "wrong_normalized_rmse": points["wrong_normalized_rmse"],
        "gain_normalized_rmse": gain_norm,
        "oracle_readout_normalized_mse": normalized_mse_points["oracle"],
        "predicted_normalized_mse": normalized_mse_points["pred"],
        "wrong_normalized_mse": normalized_mse_points["wrong"],
        "gain_normalized_mse": gain_normalized_mse,
        "oracle_readout_rmse": points["oracle_rmse"],
        "predicted_rmse": points["pred_rmse"],
        "wrong_rmse": points["wrong_rmse"],
        "gain_rmse": gain_rmse,
        "rmse_units": str(
            physical_payloads[0].get("target", {}).get("units", "readout target units")
        ),
        "bootstrap": {
            "oracle_readout_normalized_rmse_ci95": ci(boot["oracle_normalized_rmse"]),
            "predicted_normalized_rmse_ci95": ci(boot["pred_normalized_rmse"]),
            "wrong_normalized_rmse_ci95": ci(boot["wrong_normalized_rmse"]),
            "gain_normalized_rmse_ci95": ci(gain_norm_boot),
            "gain_normalized_mse_ci95": ci(gain_normalized_mse_boot),
            "oracle_readout_rmse_ci95": _ci(boot["oracle_rmse"]),
            "predicted_rmse_ci95": _ci(boot["pred_rmse"]),
            "wrong_rmse_ci95": _ci(boot["wrong_rmse"]),
            "gain_rmse_ci95": _ci(gain_rmse_boot),
            "unit": "source_group paired bootstrap; pooled query_mse and normalization_denominator; same draws for oracle, pred, wrong and gain",
        },
    }


def _result_dir(root: Path, spec: Mapping[str, Any]) -> Path:
    preferred = root / "results" / str(spec["id"])
    if preferred.exists() or not spec.get("result_dir"):
        return preferred
    return Path(str(spec["result_dir"]))


def _physical_payload(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload.get("query_metrics"), list):
        raise ValueError(f"Physical output has no query_metrics: {path}")
    return payload


def _paired_indices(groups: Sequence[str], reps: int, seed: int) -> np.ndarray:
    return _bootstrap_indices(len(groups), reps, seed)


def _reference_scores(path: Path | None) -> dict[tuple[str, str, str], float]:
    if path is None or not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        (str(row["task"]), str(row["family"]), str(row["regime"])): float(row["score"])
        for row in payload.get("rows", [])
    }


def _flatten_rows(
    combo: tuple[str, str, str],
    history: Mapping[str, Any],
    physical: Mapping[str, Any],
    reference: Mapping[tuple[str, str, str], float],
) -> dict[str, Any]:
    task, family, regime = combo
    row = {
        "task": task,
        "family": family,
        "regime": regime,
        "training_repetitions": history["training_repetitions"],
        "training_seeds": history["training_seeds"],
        "n_queries": history["n_queries"],
        "n_source_groups": history["n_source_groups"],
        "matched_error_ratio": history["matched_error_ratio"],
        "wrong_error_ratio": history["wrong_error_ratio"],
        "gain": history["gain"],
        "score": history["score"],
        "matched_error_ratio_by_horizon": history["matched_error_ratio_by_horizon"],
        "wrong_error_ratio_by_horizon": history["wrong_error_ratio_by_horizon"],
        "gain_by_horizon": history["gain_by_horizon"],
        "history_bootstrap": history["bootstrap"],
        "history": history,
        "physical": physical,
    }
    key = combo
    if key in reference:
        delta = float(row["score"] - reference[key])
        row["coverage_v2_score"] = reference[key]
        row["coverage_v2_score_delta"] = delta
        row["coverage_v2_score_match_rtol_1e-6_atol_1e-4"] = bool(
            np.isclose(row["score"], reference[key], rtol=1e-6, atol=1e-4)
        )
    else:
        row["coverage_v2_score"] = None
        row["coverage_v2_score_delta"] = None
        row["coverage_v2_score_match_rtol_1e-6_atol_1e-4"] = None
    return row


def summarize(
    *,
    root: Path,
    models_path: Path,
    panels_root: Path,
    physical_root: Path,
    bootstrap_map_path: Path,
    bootstrap_reps: int = DEFAULT_BOOTSTRAP_REPS,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    reference_path: Path | None = None,
    allow_partial: bool = False,
) -> dict[str, Any]:
    specs_payload = json.loads(models_path.read_text(encoding="utf-8"))
    specs = specs_payload.get("specs", specs_payload) if isinstance(specs_payload, Mapping) else specs_payload
    if not isinstance(specs, list):
        raise ValueError("models manifest must be a list or {specs: list}")
    expected_combos = sorted({(str(s["task"]), str(s["family"]), str(s["regime"])) for s in specs})
    runs: list[dict[str, Any]] = []
    missing_runs: list[dict[str, Any]] = []
    for spec in specs:
        result_dir = _result_dir(root, spec)
        panel_root = panels_root / str(spec["task"])
        try:
            run = read_history_run(spec, result_dir, panel_root, bootstrap_map_path)
            if run["receipt_scenes"] != run["expected_queries"] or len(run["queries"]) != run["expected_queries"]:
                raise ValueError(
                    f"incomplete query coverage ({len(run['queries'])}/{run['expected_queries']})"
                )
            runs.append(run)
        except (FileNotFoundError, OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            missing_runs.append({"id": str(spec["id"]), "reason": str(exc)})

    if missing_runs and not allow_partial:
        raise RuntimeError(
            f"Formal summary requires all {len(specs)} runs; missing/invalid {len(missing_runs)}. "
            "Use --allow-partial only for temporary inspection."
        )
    combos_present = sorted({(run["task"], run["family"], run["regime"]) for run in runs})
    rows: list[dict[str, Any]] = []
    physical_missing: list[dict[str, Any]] = []
    reference = _reference_scores(reference_path)
    for combo in combos_present:
        combo_runs = [run for run in runs if (run["task"], run["family"], run["regime"]) == combo]
        groups = sorted({row["source_group"] for run in combo_runs for row in run["queries"]})
        indices = _paired_indices(groups, bootstrap_reps, bootstrap_seed)
        # The run reader already checks that repetitions share their query set;
        # this map gives the physical output the same source grouping exactly.
        first_query_groups = {row["scene_id"]: row["source_group"] for row in combo_runs[0]["queries"]}
        history = aggregate_history(combo_runs, indices)
        payloads: list[Mapping[str, Any]] = []
        for run in combo_runs:
            physical_path = _find_physical_file(physical_root, run["id"])
            if physical_path is None:
                physical_missing.append({"id": run["id"], "reason": "physical readout unavailable"})
                continue
            try:
                payload = _physical_payload(physical_path)
                metric_ids = {str(item.get("scene_id")) for item in payload["query_metrics"]}
                expected_ids = set(first_query_groups)
                if metric_ids != expected_ids:
                    raise ValueError(
                        f"physical query coverage {len(metric_ids)}/{len(expected_ids)}"
                    )
                payloads.append(payload)
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                physical_missing.append({"id": run["id"], "reason": str(exc)})
        physical = aggregate_physical(payloads, first_query_groups, indices)
        rows.append(_flatten_rows(combo, history, physical, reference))

    score_checks = [row["coverage_v2_score_delta"] for row in rows if row["coverage_v2_score_delta"] is not None]
    mismatched_reference = [
        row for row in rows if row["coverage_v2_score_match_rtol_1e-6_atol_1e-4"] is False
    ]
    complete = (
        len(runs) == len(specs) == EXPECTED_RUNS
        and len(rows) == len(expected_combos) == EXPECTED_ROWS
        and not physical_missing
        and not missing_runs
    )
    if not allow_partial and not complete:
        raise RuntimeError(
            f"Formal summary validation gate requires {EXPECTED_RUNS} runs/{EXPECTED_ROWS} rows and physical outputs; "
            f"observed {len(runs)} runs/{len(rows)} rows, missing physical={len(physical_missing)}"
        )
    if not allow_partial and mismatched_reference:
        raise RuntimeError("History score disagrees with coverage_v2 beyond rtol=1e-6, atol=1e-4")
    return _json_safe(
        {
            "schema": "contextworld.icl_measurement_validation_summary.v1",
            "evaluation_split": "development",
            "scope": "nine-task Development panels; frozen history outputs and auxiliary readout diagnostics",
            "primary_score_unchanged": True,
            "aggregation": {
                "seed_weight": "equal over all training seeds within task/family/regime",
                "history_denominator": "Bq = mean over five target-separation horizons after K*C normalization; same sum(Bq) denominator for every horizon",
                "source_bootstrap": "source groups shared across conditions, candidates, horizons, seeds, history and physical contrasts",
                "paired_contrasts": {
                    "history": "wrong_error_ratio - matched_error_ratio",
                    "physical": "wrong_normalized_rmse/rmse - predicted_normalized_rmse/rmse from query_metrics query_mse",
                    "same_bootstrap_draws": True,
                },
                "bootstrap_replicates": int(bootstrap_reps),
                "bootstrap_seed": int(bootstrap_seed),
            },
            "calibration_claim": (
                "Physical readout is a source-group cross-fitted Development calibration diagnostic. "
                "It does not by itself establish comparable physical accuracy across encoders."
            ),
            "formal_gate": {
                "allow_partial": bool(allow_partial),
                "complete": bool(complete),
                "expected_runs": EXPECTED_RUNS,
                "observed_runs": len(runs),
                "expected_rows": EXPECTED_ROWS,
                "observed_rows": len(rows),
                "expected_physical_files": EXPECTED_RUNS,
                "observed_physical_files": len(runs) - len(physical_missing),
            },
            "reference_score_check": {
                "path": None if reference_path is None else str(reference_path),
                "available_rows": len(reference),
                "max_abs_delta": None if not score_checks else float(np.max(np.abs(score_checks))),
                "tolerance": "rtol=1e-6, atol=1e-4",
                "all_within_rtol_1e-6_atol_1e-4": None if not score_checks else not mismatched_reference,
            },
            "missing_runs": missing_runs,
            "missing_physical": physical_missing,
            "runs": [
                {
                    "id": run["id"],
                    "task": run["task"],
                    "family": run["family"],
                    "regime": run["regime"],
                    "training_seed": run["training_seed"],
                    "n_queries": len(run["queries"]),
                    "receipt_sha256": run["receipt_sha256"],
                }
                for run in sorted(runs, key=lambda item: item["id"])
            ],
            "rows": rows,
        }
    )


def _base_output(path: Path) -> Path:
    path = Path(path)
    return path.with_suffix("") if path.suffix in {".json", ".csv", ".md"} else path


def write_outputs(payload: Mapping[str, Any], output: Path) -> tuple[Path, Path, Path]:
    base = _base_output(output)
    base.parent.mkdir(parents=True, exist_ok=True)
    json_path, csv_path, md_path = base.with_suffix(".json"), base.with_suffix(".csv"), base.with_suffix(".md")
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    fields = [
        "task", "family", "regime", "training_repetitions", "n_queries", "n_source_groups",
        "matched_error_ratio", "wrong_error_ratio", "gain", "score",
        "oracle_readout_normalized_rmse", "predicted_normalized_rmse",
        "wrong_normalized_rmse", "gain_normalized_rmse", "gain_normalized_mse",
        "oracle_readout_rmse", "predicted_rmse", "wrong_rmse", "gain_rmse", "rmse_units",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in payload.get("rows", []):
            physical = row.get("physical", {})
            writer.writerow(
                {
                    key: row.get(key, physical.get(key))
                    for key in fields
                }
            )
    task_labels = {
        "speed": ("速度", "px"),
        "action_strength": ("推手移动幅度", "px-equivalent"),
        "robot_arm_mass": ("机械臂质量", "mm"),
        "action_delay": ("动作延迟", "px"),
        "contact_friction": ("接触摩擦", "px-equivalent"),
        "motion_damping": ("运动阻尼", "px-equivalent"),
        "cube_gripper_carry": ("Cube 夹爪携带", "mm"),
        "door": ("门通行规则", "px"),
        "portal_exit": ("传送门出口", "px"),
    }
    task_order = tuple(task_labels)
    model_labels = {"lewm": "LeWM", "pldm": "PLDM", "dinowm": "DINO-WM"}
    model_order = {"lewm": 0, "pldm": 1, "dinowm": 2}
    scheme_labels = {
        "original": "T0",
        "scratch": "T1",
        "joint": "T2",
        "frozen": "T3",
    }
    scheme_order = {"original": 0, "scratch": 1, "joint": 2, "frozen": 3}

    def number(value: Any) -> str:
        if value is None:
            return "—"
        if isinstance(value, (float, int)):
            return f"{float(value):.3g}"
        return str(value)

    def estimate_with_ci(value: Any, interval: Any) -> str:
        if value is None:
            return "—"
        if isinstance(interval, (list, tuple)) and len(interval) == 2:
            return f"{number(value)} [{number(interval[0])}, {number(interval[1])}]"
        return number(value)

    rows_by_task: dict[str, list[Mapping[str, Any]]] = {}
    for row in payload.get("rows", []):
        rows_by_task.setdefault(str(row.get("task", "unknown")), []).append(row)
    unknown_tasks = sorted(set(rows_by_task) - set(task_labels))
    ordered_tasks = [task for task in task_order if task in rows_by_task]
    ordered_tasks.extend(unknown_tasks)

    lines = [
        "# ICL 历史条件验证表",
        "",
        "正值表示错误历史的误差高于正确历史，说明正确历史带来更低误差；负值表示相反。",
        "latent 历史收益为 wrong−matched 的误差比，物理历史收益为 wrong−matched 的 RMSE。",
        "物理读出是按来源组交叉拟合的 Development 辅助诊断；真实 latent 校准 RMSE 用于说明读出误差，不能单独建立统一物理基准。",
        "表中不含主分或过门判定；所有可用方案均列出。",
        "",
    ]
    for task in ordered_tasks:
        task_rows = rows_by_task[task]
        task_rows = sorted(
            task_rows,
            key=lambda row: (
                model_order.get(str(row.get("family", "")), 99),
                scheme_order.get(str(row.get("regime", "")), 99),
                str(row.get("family", "")),
                str(row.get("regime", "")),
            ),
        )
        label, default_units = task_labels.get(task, (task, "未提供"))
        observed_units = sorted(
            {
                str((row.get("physical") or {}).get("rmse_units"))
                for row in task_rows
                if (row.get("physical") or {}).get("rmse_units")
            }
        )
        units = observed_units[0] if len(observed_units) == 1 else default_units
        lines.extend(
            [
                f"## {label}",
                "",
                f"单位：latent 历史收益为无量纲误差比；真实 latent 校准 RMSE、正确历史预测读出 RMSE 和物理历史收益为 {units}。",
                "",
                "| 模型 | 方案 | latent 历史收益 [95% CI] | 真实 latent 校准 RMSE | 正确历史预测读出 RMSE | 物理历史收益 [95% CI] |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for row in task_rows:
            history = row.get("history") or {}
            history_bootstrap = history.get("bootstrap") or row.get("history_bootstrap") or {}
            physical = row.get("physical") or {}
            physical_bootstrap = physical.get("bootstrap") or {}
            family = str(row.get("family", ""))
            regime = str(row.get("regime", ""))
            lines.append(
                "| "
                + " | ".join(
                    [
                        model_labels.get(family, family),
                        scheme_labels.get(regime, regime),
                        estimate_with_ci(row.get("gain", history.get("gain")), history_bootstrap.get("gain_ci95")),
                        number(physical.get("oracle_readout_rmse")),
                        number(physical.get("predicted_rmse")),
                        estimate_with_ci(physical.get("gain_rmse"), physical_bootstrap.get("gain_rmse_ci95")),
                    ]
                )
                + " |"
            )
        lines.append("")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, csv_path, md_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--models", type=Path)
    parser.add_argument("--panels-root", type=Path, required=True)
    parser.add_argument("--physical-root", type=Path)
    parser.add_argument("--bootstrap-map", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bootstrap-reps", type=int, default=DEFAULT_BOOTSTRAP_REPS)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--allow-partial", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.root
    models = args.models or root / "models.json"
    physical = args.physical_root or root / "physical"
    bootstrap_map = args.bootstrap_map or args.panels_root.parent / "bootstrap_clusters.json"
    reference = args.reference
    if reference is None:
        candidate = Path(__file__).resolve().parents[1] / "docs/research/data/multistep_prediction_coverage_v2.json"
        reference = candidate if candidate.exists() else None
    output = args.output or root / "measurement_validation"
    payload = summarize(
        root=root,
        models_path=models,
        panels_root=args.panels_root,
        physical_root=physical,
        bootstrap_map_path=bootstrap_map,
        bootstrap_reps=args.bootstrap_reps,
        bootstrap_seed=args.bootstrap_seed,
        reference_path=reference,
        allow_partial=args.allow_partial,
    )
    paths = write_outputs(payload, output)
    print(
        f"Wrote {paths[0]}, {paths[1]}, {paths[2]} "
        f"({payload['formal_gate']['observed_runs']} runs, {payload['formal_gate']['observed_rows']} rows)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
