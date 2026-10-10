#!/usr/bin/env python3
"""Calibrate native latent geometry against true-trajectory selected geometry.

This script only reads cached target embeddings and original future-state
panels. It never reads the cached ``pred`` array, loads a model, fits a
readout, or generates simulator data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np


TASKS = ("action_strength", "motion_damping", "robot_arm_mass")
FAMILIES = ("lewm", "pldm")
REGIMES = ("original", "scratch", "joint", "frozen")
HORIZONS = (5, 10, 15, 20, 25)
EXPECTED_SCENES = 256
EXPECTED_K = 2
EXPECTED_C = 11
EXPECTED_T = 5
N_BOOTSTRAP = 4000
BOOTSTRAP_SEED = 20261008
ABS_TOL = 1.0e-8
REL_TOL = 1.0e-7

DATA_DIR = Path(__file__).resolve().parents[1] / "docs" / "research" / "data"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object at {path}")
    return value


def physical_targets(panel: Any, task: str) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    """Extract selected, explicitly task-specific physical trajectory geometry."""
    if task == "action_strength":
        states = np.asarray(panel["future_states"], dtype=np.float64)
        if states.shape[-1] != 7:
            raise ValueError(f"Strength expected final state width 7, got {states.shape}")
        selected = np.concatenate(
            [states[..., 0:4],
             (40.0 * np.sin(states[..., 4]))[..., None],
             (40.0 * np.cos(states[..., 4]))[..., None]],
            axis=-1,
        )
        description = {
            "source_array": "future_states",
            "selected_raw_components": ["0:4", "angle_index_4_as_40_sin_cos"],
            "coordinate_description": "selected agent xy, block xy, and orientation embedding",
            "dimensions": 6,
            "units": "pixels-equivalent",
            "scope": "selected pose geometry, not the full state",
            "task_relevant_subgeometry": {
                "selected_raw_components": ["agent_xy:0:2"],
                "coordinate_description": "agent xy only",
                "dimensions": 2,
            },
        }
        relevant = states[..., 0:2]
    elif task == "motion_damping":
        states = np.asarray(panel["future_states"], dtype=np.float64)
        if states.shape[-1] < 12:
            raise ValueError(f"Damping expected state width >=12, got {states.shape}")
        selected = np.concatenate(
            [states[..., 0:2], states[..., 6:8],
             (40.0 * np.sin(states[..., 10]))[..., None],
             (40.0 * np.cos(states[..., 10]))[..., None]],
            axis=-1,
        )
        description = {
            "source_array": "future_states",
            "selected_raw_components": ["0:2", "6:8", "angle_index_10_as_40_sin_cos"],
            "coordinate_description": "selected tracked xy coordinates and orientation embedding",
            "dimensions": 6,
            "units": "pixels-equivalent",
            "scope": "selected pose geometry, not the full state",
            "task_relevant_subgeometry": {
                "selected_raw_components": ["block_xy:6:8", "angle_index_10_as_40_sin_cos"],
                "coordinate_description": "block xy and orientation embedding",
                "dimensions": 4,
            },
        }
        relevant = np.concatenate(
            [states[..., 6:8],
             (40.0 * np.sin(states[..., 10]))[..., None],
             (40.0 * np.cos(states[..., 10]))[..., None]],
            axis=-1,
        )
    elif task == "robot_arm_mass":
        selected = np.asarray(panel["finger_positions"], dtype=np.float64) * 1000.0
        description = {
            "source_array": "finger_positions",
            "selected_raw_components": ["x", "y"],
            "coordinate_description": "fingertip xy",
            "dimensions": 2,
            "units": "millimetres",
            "scope": "selected fingertip geometry, not all joint states",
            "task_relevant_subgeometry": "same as selected geometry; no duplicate ranking emitted",
        }
        relevant = None
    else:
        raise ValueError(f"Unsupported task: {task}")
    if selected.ndim != 4 or selected.shape[:3] != (EXPECTED_K, EXPECTED_C, EXPECTED_T):
        raise ValueError(f"Unexpected physical target shape for {task}: {selected.shape}")
    if not np.isfinite(selected).all():
        raise ValueError(f"Non-finite physical target for {task}")
    if relevant is not None and (not np.isfinite(relevant).all() or relevant.shape[:3] != selected.shape[:3]):
        raise ValueError(f"Invalid task-relevant geometry for {task}: {relevant.shape}")
    return selected, relevant, description


def pairwise_trajectory_mse(trajectories: np.ndarray) -> np.ndarray:
    """Mean over T of squared Euclidean distance; retain exact zero diagonal."""
    x = np.asarray(trajectories, dtype=np.float64)
    if x.ndim != 3:
        raise ValueError(f"Expected [trajectory,time,dimension], got {x.shape}")
    n, t_count, _ = x.shape
    delta = x[:, None, :, :] - x[None, :, :, :]
    distances = np.sum(delta * delta, axis=(2, 3), dtype=np.float64) / float(t_count)
    np.fill_diagonal(distances, 0.0)
    return distances


def _empty_counts() -> dict[str, int]:
    return {
        "comparison_count": 0,
        "physical_tie_count": 0,
        "physical_strict_count": 0,
        "latent_tie_among_physical_strict_count": 0,
        "both_strict_count": 0,
        "reversed_count": 0,
        "concordant_count": 0,
    }


def _add_comparisons(counts: dict[str, int], p_a: np.ndarray, p_b: np.ndarray,
                     z_a: np.ndarray, z_b: np.ndarray) -> None:
    if p_a.size == 0:
        return
    p_diff = p_a - p_b
    z_diff = z_a - z_b
    p_tol = ABS_TOL + REL_TOL * np.maximum(np.abs(p_a), np.abs(p_b))
    z_tol = ABS_TOL + REL_TOL * np.maximum(np.abs(z_a), np.abs(z_b))
    p_tie = np.abs(p_diff) <= p_tol
    p_strict = ~p_tie
    z_tie = np.abs(z_diff) <= z_tol
    both = p_strict & ~z_tie
    reversed_mask = both & (np.sign(p_diff) != np.sign(z_diff))
    counts["comparison_count"] += int(p_a.size)
    counts["physical_tie_count"] += int(np.count_nonzero(p_tie))
    counts["physical_strict_count"] += int(np.count_nonzero(p_strict))
    counts["latent_tie_among_physical_strict_count"] += int(np.count_nonzero(p_strict & z_tie))
    counts["both_strict_count"] += int(np.count_nonzero(both))
    counts["reversed_count"] += int(np.count_nonzero(reversed_mask))
    counts["concordant_count"] += int(np.count_nonzero(both & ~reversed_mask))


def rates_from_counts(counts: dict[str, int]) -> dict[str, float | None]:
    p_strict = counts["physical_strict_count"]
    both = counts["both_strict_count"]
    return {
        "reversal_over_physical_strict": (
            counts["reversed_count"] / p_strict if p_strict else None
        ),
        "latent_tie_fraction_over_physical_strict": (
            counts["latent_tie_among_physical_strict_count"] / p_strict if p_strict else None
        ),
        "reversal_over_both_strict": counts["reversed_count"] / both if both else None,
    }


def scene_rank_counts(physical: np.ndarray, latent: np.ndarray,
                      strong_rms_floor: float,
                      task_relevant: np.ndarray | None = None) -> dict[str, Any]:
    """Rank all alternative-error pairs for every true trajectory anchor."""
    k_count, c_count, t_count, p_dim = physical.shape
    z_dim = latent.shape[-1]
    phys = pairwise_trajectory_mse(physical.reshape(k_count * c_count, t_count, p_dim))
    zlat = pairwise_trajectory_mse(latent.reshape(k_count * c_count, t_count, z_dim))
    schemes: dict[str, dict[str, Any]] = {}
    for scheme in ("all_other_trajectories", "same_condition_other_actions"):
        counts = _empty_counts()
        strong = _empty_counts()
        for anchor in range(k_count * c_count):
            condition = anchor // c_count
            if scheme == "all_other_trajectories":
                alternatives = np.asarray([i for i in range(k_count * c_count) if i != anchor], dtype=np.int64)
            else:
                alternatives = np.asarray(
                    [condition * c_count + c for c in range(c_count) if c != anchor % c_count],
                    dtype=np.int64,
                )
            left, right = np.triu_indices(alternatives.size, 1)
            a = alternatives[left]
            b = alternatives[right]
            p_a, p_b = phys[anchor, a], phys[anchor, b]
            z_a, z_b = zlat[anchor, a], zlat[anchor, b]
            _add_comparisons(counts, p_a, p_b, z_a, z_b)
            max_p = np.maximum(p_a, p_b)
            min_p = np.minimum(p_a, p_b)
            strong_mask = (max_p >= 4.0 * min_p) & (np.sqrt(max_p) >= strong_rms_floor)
            _add_comparisons(strong, p_a[strong_mask], p_b[strong_mask],
                             z_a[strong_mask], z_b[strong_mask])
        schemes[scheme] = {
            "counts": counts,
            "rates": rates_from_counts(counts),
            "strong_subset": {"counts": strong, "rates": rates_from_counts(strong)},
        }
    if task_relevant is not None:
        # This adds one pre-specified physical subgeometry diagnostic for
        # Strength and Damping; Mass's task-relevant geometry is identical to
        # its selected fingertip geometry and is therefore not duplicated.
        relevant_result = scene_rank_counts(task_relevant, latent, strong_rms_floor)
        schemes["task_relevant_geometry_all_other"] = relevant_result["all_other_trajectories"]
    return schemes


def _merge_counts(rows: list[dict[str, Any]], key_path: tuple[str, ...]) -> dict[str, int]:
    out = _empty_counts()
    for row in rows:
        value: Any = row
        for key in key_path:
            value = value[key]
        for name in out:
            out[name] += int(value.get(name, 0))
    return out


def _percentile_ci(samples: list[float]) -> dict[str, Any]:
    if not samples:
        return {"lower_2_5": None, "upper_97_5": None, "valid_bootstrap_draws": 0}
    lo, hi = np.percentile(np.asarray(samples, dtype=np.float64), [2.5, 97.5])
    return {
        "lower_2_5": float(lo),
        "upper_97_5": float(hi),
        "valid_bootstrap_draws": len(samples),
    }


def bootstrap_macro_ci(scene_values: list[float | None], scene_clusters: list[str],
                       cluster_draws: np.ndarray) -> dict[str, Any]:
    """Cluster-bootstrap equal-scene means, keeping mirrored scenes together."""
    clusters = sorted(set(scene_clusters))
    cluster_index = {cluster: i for i, cluster in enumerate(clusters)}
    scene_cluster_indices = np.asarray([cluster_index[c] for c in scene_clusters], dtype=np.int64)
    values = np.asarray([np.nan if v is None else float(v) for v in scene_values], dtype=np.float64)
    valid = np.isfinite(values)
    samples: list[float] = []
    for draw in cluster_draws:
        multiplicity = np.bincount(draw, minlength=len(clusters))
        weights = multiplicity[scene_cluster_indices].astype(np.float64)
        included = valid & (weights > 0)
        denominator = float(np.sum(weights[included]))
        if denominator > 0:
            samples.append(float(np.sum(weights[included] * values[included]) / denominator))
    return _percentile_ci(samples)


def _build_bootstrap_draws(cluster_names: list[str], rng: np.random.Generator) -> np.ndarray:
    n_clusters = len(set(cluster_names))
    if n_clusters == 0:
        raise ValueError("No source clusters")
    return rng.integers(0, n_clusters, size=(N_BOOTSTRAP, n_clusters), endpoint=False)


def _synthetic_sanity() -> dict[str, Any]:
    # Identity gives exact zero trajectory distances on the diagonal.
    x = np.asarray([[[0.0, 0.0]], [[2.0, 1.0]], [[-1.0, 4.0]]], dtype=np.float64)
    d = pairwise_trajectory_mse(x)
    if not np.array_equal(np.diag(d), np.zeros(3)):
        raise AssertionError("Identity diagonal was not exactly zero")

    def all_anchor_counts(p: np.ndarray, z: np.ndarray) -> dict[str, Any]:
        return scene_rank_counts(p[None, ...], z[None, ...], strong_rms_floor=0.0)["all_other_trajectories"]

    points = np.asarray([[[0.0, 0.0]], [[1.0, 2.0]], [[3.0, -1.0]], [[7.0, 4.0]]])
    q = np.asarray([[0.0, -1.0], [1.0, 0.0]])  # orthogonal rotation
    rotated = points @ q
    isometry = all_anchor_counts(points, rotated)
    if isometry["counts"]["reversed_count"] != 0:
        raise AssertionError("Known isometry changed the ordering")

    reversal_p = np.asarray([[[0.0]], [[1.0]], [[2.0]], [[3.0]]])
    reversal_z = np.asarray([[[0.0]], [[3.0]], [[2.0]], [[1.0]]])
    constructed = all_anchor_counts(reversal_p, reversal_z)
    if constructed["counts"]["reversed_count"] <= 0:
        raise AssertionError("Constructed reversal failed to produce a reversal")

    scaled = all_anchor_counts(reversal_p * 3.7, reversal_z * 0.4)
    comparable = ("comparison_count", "physical_tie_count", "physical_strict_count",
                  "latent_tie_among_physical_strict_count", "both_strict_count", "reversed_count")
    if any(constructed["counts"][key] != scaled["counts"][key] for key in comparable):
        raise AssertionError("Unit-scale change altered non-near-tie ranking")
    return {
        "identity_diagonal_exact_zero": True,
        "known_isometry_reversals": isometry["counts"]["reversed_count"],
        "constructed_reversal_count": constructed["counts"]["reversed_count"],
        "unit_scale_invariance_non_near_tie": True,
        "tolerance": {"absolute": ABS_TOL, "relative": REL_TOL},
    }


def _read_measurement_clusters(measurement_queries: Path) -> dict[str, dict[str, str]]:
    import gzip

    with gzip.open(measurement_queries, "rt", encoding="utf-8") as stream:
        document = json.load(stream)
    cluster_maps: dict[str, dict[str, str]] = {}
    for task in TASKS:
        key = f"{task}/lewm/original/s3072"
        run = document["runs"][key]
        mapping = {str(row["scene_id"]): str(row["source_group"]) for row in run["history"]}
        if len(mapping) != len(run["history"]):
            raise ValueError(f"Duplicate measurement scene IDs in {key}")
        for family in FAMILIES:
            for regime in REGIMES:
                model_key = f"{task}/{family}/{regime}/s3072"
                other = document["runs"].get(model_key)
                if other is None:
                    raise FileNotFoundError(f"Missing source-cluster measurement run {model_key}")
                other_map = {str(row["scene_id"]): str(row["source_group"]) for row in other["history"]}
                if other_map != mapping:
                    raise ValueError(f"Source cluster map differs in {model_key}")
        cluster_maps[task] = mapping
    return cluster_maps


def _load_panel_index(task: str, panels_root: Path) -> tuple[Path, dict[str, Any], dict[str, dict[str, Any]]]:
    panel_dir = panels_root / task
    manifest_path = panel_dir / "manifest.json"
    manifest = _json(manifest_path)
    entries = {str(row["scene_id"]): row for row in manifest["scenes"]}
    if len(entries) != len(manifest["scenes"]):
        raise ValueError(f"Duplicate source scenes in {manifest_path}")
    if len(entries) != EXPECTED_SCENES:
        raise ValueError(f"Expected {EXPECTED_SCENES} scenes for {task}, found {len(entries)}")
    return panel_dir, manifest, entries


def _load_physical(panel_dir: Path, scene_entry: dict[str, Any], task: str) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    panel_path = panel_dir / str(scene_entry["path"])
    if sha256_file(panel_path) != scene_entry["sha256"]:
        raise ValueError(f"Panel source SHA mismatch: {panel_path}")
    with np.load(panel_path, allow_pickle=False) as panel:
        target, relevant, desc = physical_targets(panel, task)
    return target, relevant, desc


def _model_rows(model_dir: Path) -> list[tuple[Path, dict[str, Any]]]:
    rows: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(model_dir.glob("*.json")):
        row = _json(path)
        if row.get("features_file"):
            rows.append((path, row))
    if len(rows) != EXPECTED_SCENES:
        raise ValueError(f"Expected {EXPECTED_SCENES} feature-backed rows in {model_dir}, found {len(rows)}")
    scene_ids = [str(row["scene_id"]) for _, row in rows]
    if len(set(scene_ids)) != EXPECTED_SCENES:
        raise ValueError(f"Duplicate scene IDs in {model_dir}")
    return sorted(rows, key=lambda item: str(item[1]["scene_id"]))


def _scene_metrics(task: str, row_path: Path, row: dict[str, Any],
                   panel_dir: Path, panel_entry: dict[str, Any],
                   expected_cluster: str, strong_floor: float,
                   physical_cache: dict[str, tuple[np.ndarray, np.ndarray | None, dict[str, Any]]]) -> tuple[dict[str, Any], str, str, dict[str, Any]]:
    scene_id = str(row["scene_id"])
    if str(row.get("task")) != task:
        raise ValueError(f"Task mismatch in {row_path}")
    if str(row.get("source_sha256")) != str(panel_entry.get("sha256")):
        raise ValueError(f"Panel source SHA metadata mismatch for {task}/{scene_id}")
    if str(row.get("manifest_sha256")) != sha256_file(panel_dir / "manifest.json"):
        raise ValueError(f"Panel manifest SHA mismatch for {task}/{scene_id}")
    manifest_cluster = panel_entry.get("bootstrap_cluster")
    if manifest_cluster is not None and str(manifest_cluster) != expected_cluster:
        raise ValueError(f"Cluster mismatch against panel manifest for {task}/{scene_id}")
    if expected_cluster == "" or not expected_cluster:
        raise ValueError(f"Missing source cluster for {task}/{scene_id}")

    feature_path = row_path.parent / str(row["features_file"])
    if not feature_path.is_file():
        raise FileNotFoundError(feature_path)
    feature_sha = sha256_file(feature_path)
    if feature_sha != str(row.get("features_file_sha256")):
        raise ValueError(f"Feature archive SHA mismatch for {task}/{scene_id}")
    with np.load(feature_path, allow_pickle=False) as archive:
        # Deliberately retrieve only `target`; `pred` is never read.
        latent = np.asarray(archive["target"], dtype=np.float64)
    if latent.shape != (EXPECTED_K, EXPECTED_C, EXPECTED_T, 192):
        raise ValueError(f"Expected native [2,11,5,192] target layout, got {latent.shape} for {task}/{scene_id}")
    if not np.isfinite(latent).all():
        raise ValueError(f"Non-finite native target for {task}/{scene_id}")
    if row.get("feature_layout", {}).get("kind") != "native_latent" or int(row.get("latent_dimension", -1)) != 192:
        raise ValueError(f"Target is not unprojected native 192-D latent in {task}/{scene_id}")
    if list(row.get("horizons", [])) != list(HORIZONS):
        raise ValueError(f"Unexpected horizons in {task}/{scene_id}: {row.get('horizons')}")
    if int(row.get("conditions", -1)) != EXPECTED_K or int(row.get("candidates", -1)) != EXPECTED_C:
        raise ValueError(f"Unexpected K,C metadata in {task}/{scene_id}")

    if scene_id not in physical_cache:
        physical_cache[scene_id] = _load_physical(panel_dir, panel_entry, task)
    physical, relevant, physical_desc = physical_cache[scene_id]
    if physical.shape[:3] != latent.shape[:3]:
        raise ValueError(f"Physical/native target KCT mismatch in {task}/{scene_id}")
    metrics = scene_rank_counts(physical, latent, strong_floor, task_relevant=relevant)
    target_digest = hashlib.sha256()
    target_digest.update(np.asarray(latent, dtype="<f8").tobytes(order="C"))
    return metrics, feature_sha, target_digest.hexdigest(), physical_desc


def _summary_for_scheme(scene_rows: list[dict[str, Any]], scene_clusters: list[str],
                        scheme: str, bootstrap_draws: np.ndarray) -> dict[str, Any]:
    main_counts = _merge_counts(scene_rows, ("schemes", scheme, "counts"))
    strong_counts = _merge_counts(scene_rows, ("schemes", scheme, "strong_subset", "counts"))
    rates = rates_from_counts(main_counts)
    strong_rates = rates_from_counts(strong_counts)
    fields = (
        "reversal_over_both_strict",
        "reversal_over_physical_strict",
        "latent_tie_fraction_over_physical_strict",
    )
    macro: dict[str, Any] = {}
    strong_macro: dict[str, Any] = {}
    for label in fields:
        per_scene = []
        per_strong_scene = []
        for row in scene_rows:
            per_scene.append(row["schemes"][scheme]["rates"][label])
            per_strong_scene.append(row["schemes"][scheme]["strong_subset"]["rates"][label])
        valid_values = [float(v) for v in per_scene if v is not None]
        strong_valid_values = [float(v) for v in per_strong_scene if v is not None]
        macro[label] = {
            "equal_scene_mean": float(np.mean(valid_values)) if valid_values else None,
            "eligible_scenes": len(valid_values),
            "undefined_scenes": len(per_scene) - len(valid_values),
            "source_cluster_bootstrap_ci95": bootstrap_macro_ci(per_scene, scene_clusters, bootstrap_draws),
        }
        strong_macro[label] = {
            "equal_scene_mean": float(np.mean(strong_valid_values)) if strong_valid_values else None,
            "eligible_scenes": len(strong_valid_values),
            "undefined_scenes": len(per_strong_scene) - len(strong_valid_values),
            "source_cluster_bootstrap_ci95": bootstrap_macro_ci(per_strong_scene, scene_clusters, bootstrap_draws),
        }
    return {
        "aggregate_pair_counts": main_counts,
        "pooled_pair_rates": rates,
        "macro_by_scene": macro,
        "strong_subset": {
            "definition": "larger physical RMS >= 2x smaller physical RMS and >= 1 task unit",
            "aggregate_pair_counts": strong_counts,
            "pooled_pair_rates": strong_rates,
            "macro_by_scene": strong_macro,
        },
    }


def run(output_path: Path, jsonl_path: Path, only_cell: str | None = None,
        max_scenes: int | None = None, *, cache_root: Path, panels_root: Path,
        measurement_queries: Path, protocol_path: Path) -> dict[str, Any]:
    protocol = _json(protocol_path)
    cluster_maps = _read_measurement_clusters(measurement_queries)
    panel_indices = {task: _load_panel_index(task, panels_root) for task in TASKS}
    physical_caches: dict[str, dict[str, tuple[np.ndarray, np.ndarray | None, dict[str, Any]]]] = {task: {} for task in TASKS}
    cells: list[tuple[str, str, str]] = [
        (task, family, regime)
        for task in TASKS for family in FAMILIES for regime in REGIMES
    ]
    if only_cell:
        wanted = tuple(only_cell.split("/"))
        cells = [cell for cell in cells if cell == wanted]
        if len(cells) != 1:
            raise ValueError(f"--only-cell must be task/family/regime; got {only_cell!r}")

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    task_draws: dict[str, np.ndarray] = {}
    for task in TASKS:
        task_draws[task] = _build_bootstrap_draws(list(cluster_maps[task].values()), rng)

    all_results: list[dict[str, Any]] = []
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("w", encoding="utf-8") as query_stream:
        for task, family, regime in cells:
            model_id = f"{task}/{family}/{regime}/s3072"
            run_dir = cache_root / task / family / regime / "s3072"
            panel_dir, panel_manifest, panel_entries = panel_indices[task]
            model_rows = _model_rows(run_dir)
            if max_scenes is not None:
                model_rows = model_rows[:max_scenes]
            scene_ids = [str(row["scene_id"]) for _, row in model_rows]
            expected_ids = set(panel_entries)
            if max_scenes is None and set(scene_ids) != expected_ids:
                raise ValueError(f"Model/panel scene IDs differ for {model_id}")
            checkpoint_shas = {str(row.get("checkpoint_sha256")) for _, row in model_rows}
            if len(checkpoint_shas) != 1 or "None" in checkpoint_shas:
                raise ValueError(f"Inconsistent/missing checkpoint SHA for {model_id}")
            checkpoint_sha = next(iter(checkpoint_shas))

            per_scene_rows: list[dict[str, Any]] = []
            target_digest = hashlib.sha256()
            archive_digest = hashlib.sha256()
            panel_source_digest = hashlib.sha256()
            physical_description: dict[str, Any] | None = None
            source_clusters: list[str] = []
            source_hashes: set[str] = set()
            for row_path, row in model_rows:
                scene_id = str(row["scene_id"])
                if scene_id not in panel_entries:
                    raise ValueError(f"Missing source scene for {model_id}/{scene_id}")
                cluster = cluster_maps[task].get(scene_id)
                if cluster is None:
                    raise ValueError(f"No measurement source cluster for {task}/{scene_id}")
                source_clusters.append(cluster)
                metrics, feature_sha, target_sha, desc = _scene_metrics(
                    task, row_path, row, panel_dir, panel_entries[scene_id], cluster,
                    strong_floor=1.0, physical_cache=physical_caches[task],
                )
                if physical_description is None:
                    physical_description = desc
                elif desc != physical_description:
                    raise ValueError(f"Physical description changed within {task}")
                feature_sha_source = str(row["source_sha256"])
                source_hashes.add(feature_sha_source)
                target_digest.update(scene_id.encode("utf-8") + b"\0")
                target_digest.update(bytes.fromhex(target_sha))
                archive_digest.update(scene_id.encode("utf-8") + b"\0")
                archive_digest.update(bytes.fromhex(feature_sha))
                panel_source_digest.update(scene_id.encode("utf-8") + b"\0")
                panel_source_digest.update(bytes.fromhex(feature_sha_source))

                record: dict[str, Any] = {
                    "model_id": model_id,
                    "task": task,
                    "family": family,
                    "regime": regime,
                    "scene_id": scene_id,
                    "source_cluster": cluster,
                    "panel_source_sha256": feature_sha_source,
                    "features_archive_sha256": feature_sha,
                    "native_target_content_sha256": target_sha,
                    "schemes": {},
                }
                for scheme, result in metrics.items():
                    record["schemes"][scheme] = {
                        "counts": result["counts"],
                        "rates": result["rates"],
                        "strong_subset": result["strong_subset"],
                    }
                per_scene_rows.append(record)
                query_stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")

            if not model_rows:
                raise ValueError(f"No scenes in {model_id}")
            if len(source_hashes) != len(model_rows):
                raise ValueError(f"Repeated panel content SHA across distinct scene IDs for {model_id}")
            # The full run must match all 256 source scenes and use all task clusters.
            unique_clusters = sorted(set(source_clusters))
            draws = task_draws[task]
            cell_result: dict[str, Any] = {
                "model_id": model_id,
                "task": task,
                "family": family,
                "regime": regime,
                "checkpoint_sha256": checkpoint_sha,
                "scene_count": len(model_rows),
                "source_cluster_count": len(unique_clusters),
                "source_cluster_membership": "icl_measurement_validation_v1_queries.json.gz history.source_group joined by exact scene_id; verified against multistep manifest bootstrap_cluster where present; Damping forward/reverse mirror pairs share cluster",
                "source_manifest_sha256": sha256_file(panel_dir / "manifest.json"),
                "panel_source_content_sha256": panel_source_digest.hexdigest(),
                "features_archive_content_sha256": archive_digest.hexdigest(),
                "native_target_content_sha256": target_digest.hexdigest(),
                "native_target_layout": {
                    "field": "features_file.npz:target only; `pred` was not read",
                    "shape_per_scene": [EXPECTED_K, EXPECTED_C, EXPECTED_T, 192],
                    "axes": ["condition", "candidate_action", "physical_horizon", "native_latent_dimension"],
                    "latent_dimension": 192,
                    "layout_kind": "native_latent; unprojected",
                    "conditions_K": EXPECTED_K,
                    "candidate_actions_C": EXPECTED_C,
                    "horizons": list(HORIZONS),
                },
                "selected_physical_geometry": physical_description,
                "distance_definition": "mean over 5 horizons of squared Euclidean distance across selected physical coordinates or all 192 native latent dimensions",
                "comparison_schemes": {},
                "per_scene_rows_in_jsonl": len(per_scene_rows),
            }
            summary_schemes = ["all_other_trajectories", "same_condition_other_actions"]
            if task in {"action_strength", "motion_damping"}:
                summary_schemes.append("task_relevant_geometry_all_other")
            for scheme in summary_schemes:
                cell_result["comparison_schemes"][scheme] = _summary_for_scheme(
                    per_scene_rows, source_clusters, scheme, draws,
                )
            all_results.append(cell_result)

    query_sha = sha256_file(jsonl_path)
    output = {
        "schema": "contextworld.native_true_target_geometry_calibration.v1",
        "as_of": "2026-10-08",
        "purpose": "Calibrate ranking of true native latent trajectory distances against selected physical trajectory geometry.",
        "protocol": protocol,
        "protocol_sha256": sha256_file(protocol_path),
        "execution": {
            "model_inference": False,
            "training": False,
            "simulator_generation": False,
            "readout_fitting": False,
            "native_prediction_array_accessed": False,
            "native_target_array_accessed": True,
            "scene_level_query_output": str(jsonl_path),
            "scene_level_query_sha256": query_sha,
            "bootstrap_samples": N_BOOTSTRAP,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "ordering_tolerance": {"absolute": ABS_TOL, "relative": REL_TOL,
                                   "interpretation": "numeric tie tolerance, not a scientific threshold"},
            "strong_subset": {
                "definition": "larger physical RMS >= 2x smaller physical RMS and >=1 task unit",
                "task_units": {"action_strength": "pixels-equivalent", "motion_damping": "pixels-equivalent", "robot_arm_mass": "millimetres"},
            },
        },
        "inputs": {
            "native_cache_root": str(cache_root),
            "physical_panel_root": str(panels_root),
            "measurement_cluster_source": str(measurement_queries),
            "tasks": list(TASKS),
            "families": list(FAMILIES),
            "regimes": list(REGIMES),
            "cache_seed": 3072,
            "expected_full_cells": len(TASKS) * len(FAMILIES) * len(REGIMES),
            "DINO_excluded": "DINO cached features are pooled 4x4, not the native 16x16 patch representation used for this native-192-D report; they are not commensurate with the native target layout.",
        },
        "sanity_checks": _synthetic_sanity(),
        "interpretation_limits": [
            "This is a true-target representation-geometry ordering diagnostic: each anchor and every alternative are real trajectories from one source scene.",
            "It does not measure model-predicted physical error; cached model predictions were not read.",
            "It does not establish that a predictor learned or failed to learn dynamics, and it does not measure goal cost or planning quality.",
            "Selected physical coordinates are not complete states; visibility, rendering, and selected-coordinate omissions can contribute to disagreement, so reversals are not attributed solely to the encoder.",
            "A good true-target distance ordering is not sufficient to validate off-manifold predicted latent physical error.",
            "No scientific acceptance threshold is introduced; ordering tolerances only handle numerical ties.",
        ],
        "cells": all_results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True, help="Feature results root containing task/family/regime/s3072")
    parser.add_argument("--panels-root", type=Path, required=True, help="Root containing task panel manifests")
    parser.add_argument("--measurement-queries", type=Path, default=DATA_DIR / "icl_measurement_validation_v1_queries.json.gz")
    parser.add_argument("--protocol", type=Path, default=DATA_DIR / "target_geometry_protocol_v1.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jsonl", type=Path, help="Defaults to an adjacent <output>_queries.jsonl")
    parser.add_argument("--only-cell", help="smoke only one task/family/regime cell")
    parser.add_argument("--max-scenes", type=int, help="smoke with the first N scenes of the selected cell")
    args = parser.parse_args()
    if args.jsonl is None:
        args.jsonl = args.output.with_name(args.output.stem + "_queries.jsonl")
    result = run(args.output, args.jsonl, only_cell=args.only_cell, max_scenes=args.max_scenes,
                 cache_root=args.cache_root, panels_root=args.panels_root,
                 measurement_queries=args.measurement_queries, protocol_path=args.protocol)
    print(json.dumps({
        "output": str(args.output),
        "jsonl": str(args.jsonl),
        "cells": len(result["cells"]),
        "scenes": sum(row["scene_count"] for row in result["cells"]),
        "first_cell": result["cells"][0]["model_id"] if result["cells"] else None,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
