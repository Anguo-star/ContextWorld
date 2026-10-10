#!/usr/bin/env python3
"""Compare cached old and visible-canvas true-target geometry, without inference."""
from __future__ import annotations

import hashlib
import importlib.util
import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


OLD_RESULTS: Path
OLD_PANELS: Path
NEW_RESULTS: Path
NEW_PANELS: Path
OUT: Path
EXPECTED_HORIZONS = [5, 10, 15, 20, 25]
N_BOOT = 4000
SEED = 20261008
ABS_TOL = 1e-8
REL_TOL = 1e-7
REGIMES = {
    "original": {"old_seed": "s3073", "new_seed": "s3073", "checkpoint_sha256": "9f13b2c28bb909338047e08d62bf6eb16ea3616cb53e57cfa9b5072b43db1c59"},
    "frozen": {"old_seed": "s3072", "new_seed": "s3072", "checkpoint_sha256": "0097aa2e5c352e957cb17382c7518a49c8a266f42a8dc1384d41934a5515d225"},
}


def load_utility(utility_path: Path):
    spec = importlib.util.spec_from_file_location("native_geometry_utility", utility_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import pure geometry utilities from {utility_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


geom: Any = None


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha256_array(a: np.ndarray) -> str:
    x = np.ascontiguousarray(a)
    return hashlib.sha256(x.dtype.str.encode() + b"\0" + str(x.shape).encode() + b"\0" + x.tobytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    d = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(d, dict):
        raise TypeError(f"Expected object in {path}")
    return d


def load_rows(root: Path, regime: str, seed: str) -> dict[str, tuple[Path, dict[str, Any]]]:
    model_dir = root / regime / seed
    rows: dict[str, tuple[Path, dict[str, Any]]] = {}
    for p in sorted(model_dir.glob("*.json")):
        row = read_json(p)
        if not row.get("features_file"):
            continue
        sid = str(row["scene_id"])
        if sid in rows:
            raise ValueError(f"Duplicate row for {regime}/{seed}/{sid}")
        rows[sid] = (p, row)
    if len(rows) != 256:
        raise ValueError(f"Expected 256 feature rows at {model_dir}, got {len(rows)}")
    return rows


def target_from_row(row_path: Path, row: dict[str, Any], expected_c: int | None = None) -> tuple[np.ndarray, str]:
    fp = row_path.parent / str(row["features_file"])
    actual_sha = sha256_file(fp)
    if actual_sha != row.get("features_file_sha256"):
        raise ValueError(f"Feature archive SHA mismatch: {fp}")
    with np.load(fp, allow_pickle=False) as z:
        # Only native true targets are read; never access `pred`.
        target = np.asarray(z["target"], dtype=np.float64)
    if target.ndim != 4 or target.shape[0] != 2 or target.shape[2:] != (5, 192):
        raise ValueError(f"Unexpected native target shape {target.shape} in {fp}")
    if expected_c is not None and target.shape[1] != expected_c:
        raise ValueError(f"Target candidate axis mismatch in {fp}: {target.shape[1]} vs {expected_c}")
    if not np.isfinite(target).all():
        raise ValueError(f"Non-finite true target in {fp}")
    if row.get("feature_layout", {}).get("kind") != "native_latent" or int(row.get("latent_dimension", -1)) != 192:
        raise ValueError(f"Not native 192-D features in {fp}")
    if list(row.get("horizons", [])) != EXPECTED_HORIZONS:
        raise ValueError(f"Unexpected horizons in {fp}")
    return target, actual_sha


def selected_geometry(states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    states = np.asarray(states, dtype=np.float64)
    if states.ndim != 4 or states.shape[0] != 2 or states.shape[2] != 5 or states.shape[-1] != 7:
        raise ValueError(f"Unexpected Strength future_states shape {states.shape}")
    all6 = np.concatenate(
        [states[..., 0:4], (40.0 * np.sin(states[..., 4]))[..., None],
         (40.0 * np.cos(states[..., 4]))[..., None]], axis=-1,
    )
    agent_xy = states[..., 0:2]
    return all6, agent_xy


def rate_bundle(metric: dict[str, Any]) -> dict[str, Any]:
    c = metric["counts"]
    return {
        "counts": c,
        "reversal_over_both_strict": metric["rates"]["reversal_over_both_strict"],
        "reversal_over_physical_strict": metric["rates"]["reversal_over_physical_strict"],
        "latent_tie_fraction_over_physical_strict": metric["rates"]["latent_tie_fraction_over_physical_strict"],
        "strong_subset": {
            "counts": metric["strong_subset"]["counts"],
            "reversal_over_both_strict": metric["strong_subset"]["rates"]["reversal_over_both_strict"],
            "reversal_over_physical_strict": metric["strong_subset"]["rates"]["reversal_over_physical_strict"],
            "latent_tie_fraction_over_physical_strict": metric["strong_subset"]["rates"]["latent_tie_fraction_over_physical_strict"],
        },
    }


def bootstrap_summary(values: list[float | None], clusters: list[str], draws: np.ndarray) -> dict[str, Any]:
    valid = [float(v) for v in values if v is not None]
    ci = geom.bootstrap_macro_ci(values, clusters, draws)
    return {
        "equal_scene_mean": float(np.mean(valid)) if valid else None,
        "defined_scenes": len(valid),
        "undefined_scenes": len(values) - len(valid),
        "source_cluster_bootstrap_ci95": ci,
    }


def paired_summary(old_values: list[float | None], new_values: list[float | None],
                    clusters: list[str], draws: np.ndarray) -> dict[str, Any]:
    deltas: list[float | None] = []
    for a, b in zip(old_values, new_values):
        deltas.append(float(b - a) if a is not None and b is not None else None)
    out = bootstrap_summary(deltas, clusters, draws)
    out["definition"] = "visible minus original; paired by exact scene_id, cluster-bootstrap on shared source clusters"
    out["paired_scenes"] = out["defined_scenes"]
    return out


def per_scene_geometry(physical: np.ndarray, agent_xy: np.ndarray, latent: np.ndarray) -> dict[str, Any]:
    if physical.shape[:3] != latent.shape[:3] or agent_xy.shape[:3] != latent.shape[:3]:
        raise ValueError("Physical and native target KCT axes do not align")
    result = geom.scene_rank_counts(physical, latent, strong_rms_floor=1.0, task_relevant=agent_xy)
    return {
        "all6": rate_bundle(result["all_other_trajectories"]),
        "task_agent_xy": rate_bundle(result["task_relevant_geometry_all_other"]),
    }


def scene_mse(trajectories: np.ndarray) -> np.ndarray:
    k, c, t, d = trajectories.shape
    return geom.pairwise_trajectory_mse(trajectories.reshape(k * c, t, d))


def unchanged_mapping(old_panel: Any, new_panel: Any) -> tuple[list[int], list[int], list[dict[str, Any]]]:
    old_actions = np.asarray(old_panel["candidate_actions"])
    slot_actions = np.asarray(new_panel["candidate_slot_actions"])
    new_actions = np.asarray(new_panel["candidate_actions"])
    factor = np.asarray(new_panel["candidate_factor"])
    slot_to_unique = np.asarray(new_panel["candidate_slot_to_unique"], dtype=np.int64)
    if old_actions.shape != (11, 5, 5, 2) or slot_actions.shape != (11, 5, 5, 2):
        raise ValueError(f"Unexpected old/slot action shapes {old_actions.shape}, {slot_actions.shape}")
    if factor.shape != (11,) or slot_to_unique.shape != (11,):
        raise ValueError("Visible factor/slot map must have 11 slots")
    if np.any(slot_to_unique < 0) or np.any(slot_to_unique >= new_actions.shape[0]):
        raise ValueError("Visible candidate_slot_to_unique points outside candidate_actions")
    for slot, unique in enumerate(slot_to_unique):
        if not np.array_equal(slot_actions[slot], new_actions[unique]):
            raise ValueError(f"Visible slot-to-unique action mismatch at slot {slot}")
    representatives: dict[int, int] = {}
    matches: list[dict[str, Any]] = []
    for slot in range(11):
        if float(factor[slot]) != 1.0:
            continue
        if not np.array_equal(slot_actions[slot], old_actions[slot]):
            continue
        unique = int(slot_to_unique[slot])
        if not np.array_equal(new_actions[unique], old_actions[slot]):
            raise ValueError(f"factor==1 mapped action differs from old slot {slot}")
        matches.append({"old_slot": slot, "visible_unique_candidate": unique})
        representatives.setdefault(unique, slot)
    visible_ids = sorted(representatives)
    old_slots = [representatives[i] for i in visible_ids]
    return old_slots, visible_ids, matches


def unchanged_diagnostics(old_physical: np.ndarray, new_physical: np.ndarray,
                          old_agent: np.ndarray, new_agent: np.ndarray,
                          old_target: np.ndarray, new_target: np.ndarray,
                          old_slots: list[int], visible_ids: list[int]) -> dict[str, Any]:
    if len(old_slots) < 3:
        return {"included": False, "reason": "fewer_than_3_unique_unchanged_candidates", "unique_candidate_count": len(old_slots)}
    op = old_physical[:, old_slots]
    np_ = new_physical[:, visible_ids]
    oa = old_agent[:, old_slots]
    na = new_agent[:, visible_ids]
    oz = old_target[:, old_slots]
    nz = new_target[:, visible_ids]
    old_stats = per_scene_geometry(op, oa, oz)
    new_stats = per_scene_geometry(np_, na, nz)
    op_dist, np_dist = scene_mse(op), scene_mse(np_)
    oz_dist, nz_dist = scene_mse(oz), scene_mse(nz)
    return {
        "included": True,
        "unique_candidate_count": len(old_slots),
        "old_slots": old_slots,
        "visible_unique_candidates": visible_ids,
        "physical_target_arrays_exactly_equal": bool(np.array_equal(op, np_)),
        "native_target_arrays_exactly_equal": bool(np.array_equal(oz, nz)),
        "physical_pairwise_distance_matrix_exactly_equal": bool(np.array_equal(op_dist, np_dist)),
        "physical_pairwise_distance_matrix_max_abs_delta": float(np.max(np.abs(op_dist - np_dist))),
        "native_pairwise_distance_matrix_exactly_equal": bool(np.array_equal(oz_dist, nz_dist)),
        "native_pairwise_distance_matrix_max_abs_delta": float(np.max(np.abs(oz_dist - nz_dist))),
        "old_ordering": old_stats,
        "visible_ordering": new_stats,
    }


def first_agent_xy_strong_reversal(regimes_data: dict[str, list[dict[str, Any]]]) -> dict[str, Any] | None:
    for regime in ("original", "frozen"):
        for item in regimes_data[regime]:
            k, c, t, _ = item["agent_xy"].shape
            p = scene_mse(item["agent_xy"])
            z = scene_mse(item["target"])
            condition_values = item["conditions"]
            for anchor in range(k * c):
                alternatives = [i for i in range(k * c) if i != anchor]
                for x in range(len(alternatives)):
                    for y in range(x + 1, len(alternatives)):
                        a, b = alternatives[x], alternatives[y]
                        pa, pb = float(p[anchor, a]), float(p[anchor, b])
                        za, zb = float(z[anchor, a]), float(z[anchor, b])
                        larger, smaller = max(pa, pb), min(pa, pb)
                        if larger < 4.0 * smaller or math.sqrt(larger) < 1.0:
                            continue
                        ptol = ABS_TOL + REL_TOL * max(abs(pa), abs(pb))
                        ztol = ABS_TOL + REL_TOL * max(abs(za), abs(zb))
                        if abs(pa - pb) <= ptol or abs(za - zb) <= ztol:
                            continue
                        if (pa > pb) == (za > zb):
                            continue
                        def decode(i: int) -> dict[str, Any]:
                            ci, ai = divmod(i, c)
                            return {"condition_index": ci, "condition_value": float(condition_values[ci]),
                                    "visible_unique_candidate": ai}
                        return {
                            "regime": regime,
                            "scene_id": item["scene_id"],
                            "source_sha256": item["source_sha256"],
                            "features_archive_sha256": item["features_archive_sha256"],
                            "native_target_content_sha256": item["target_content_sha256"],
                            "checkpoint_sha256": item["checkpoint_sha256"],
                            "geometry": "agent xy only; true future trajectories; no model prediction",
                            "anchor": decode(anchor),
                            "alternative_a": decode(a),
                            "alternative_b": decode(b),
                            "physical_rms": {"alternative_a": math.sqrt(pa), "alternative_b": math.sqrt(pb),
                                             "larger_over_smaller": math.sqrt(larger / smaller) if smaller > 0 else None,
                                             "larger_task_units_at_least_1": math.sqrt(larger) >= 1.0},
                            "native_mean_squared_distance": {"alternative_a": za, "alternative_b": zb},
                            "ordering_reversed": True,
                        }
    return None


def main() -> None:
    global OLD_RESULTS, OLD_PANELS, NEW_RESULTS, NEW_PANELS, OUT, geom
    parser = argparse.ArgumentParser(
        description="Compare cached old and visible Strength true-target geometry without model inference."
    )
    parser.add_argument(
        "--old-cache-root", type=Path, required=True,
        help="Old results root at the action_strength/lewm level (contains regime/seed directories).",
    )
    parser.add_argument(
        "--visible-cache-root", type=Path, required=True,
        help="Visible results root at the action_strength/lewm level (contains regime/seed directories).",
    )
    parser.add_argument(
        "--old-panels-root", type=Path, required=True,
        help="Old panel root at the action_strength level (contains manifest.json and pair_*.npz).",
    )
    parser.add_argument(
        "--visible-panels-root", type=Path, required=True,
        help="Visible panel root at the action_strength level (contains manifest.json and pair_*.npz).",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for visible_geometry.json and its JSONL.")
    parser.add_argument(
        "--utility-path", type=Path,
        default=Path(__file__).resolve().with_name("calibrate_native_geometry.py"),
        help="Optional path to calibrate_native_geometry.py; defaults to the copy beside this script.",
    )
    args = parser.parse_args()
    OLD_RESULTS, NEW_RESULTS = args.old_cache_root, args.visible_cache_root
    OLD_PANELS, NEW_PANELS = args.old_panels_root, args.visible_panels_root
    OUT = args.output_dir
    geom = load_utility(args.utility_path)
    manifest_old_path = OLD_PANELS / "manifest.json"
    manifest_new_path = NEW_PANELS / "manifest.json"
    old_manifest = read_json(manifest_old_path)
    new_manifest = read_json(manifest_new_path)
    old_entries = {str(e["scene_id"]): e for e in old_manifest["scenes"]}
    new_entries = {str(e["scene_id"]): e for e in new_manifest["scenes"]}
    if len(old_entries) != 256 or len(new_entries) != 256 or set(old_entries) != set(new_entries):
        raise ValueError("Old/new panel manifests do not contain the same 256 scene IDs")
    old_manifest_sha, new_manifest_sha = sha256_file(manifest_old_path), sha256_file(manifest_new_path)
    output_rows: list[dict[str, Any]] = []
    results: dict[str, Any] = {}
    strong_reversal_scan: dict[str, list[dict[str, Any]]] = {r: [] for r in REGIMES}
    hist_digest: dict[str, hashlib._Hash] = {r: hashlib.sha256() for r in REGIMES}

    cluster_by_scene: dict[str, str] = {}
    for sid in sorted(old_entries):
        oe, ne = old_entries[sid], new_entries[sid]
        old_cluster = str(oe.get("bootstrap_cluster", ""))
        new_cluster = str(ne.get("bootstrap_cluster", ne.get("source_group", "")))
        if not old_cluster or old_cluster != new_cluster:
            raise ValueError(f"Source cluster mismatch for {sid}: {old_cluster!r} vs {new_cluster!r}")
        if int(oe.get("source_index", -1)) != int(ne.get("source_index", -2)):
            raise ValueError(f"Source-index mismatch for {sid}")
        cluster_by_scene[sid] = old_cluster
    clusters = [cluster_by_scene[sid] for sid in sorted(cluster_by_scene)]
    if len(set(clusters)) != 243:
        raise ValueError(f"Expected 243 source clusters, found {len(set(clusters))}")
    rng = np.random.default_rng(SEED)
    draws = geom._build_bootstrap_draws(clusters, rng)

    for regime, cfg in REGIMES.items():
        old_rows = load_rows(OLD_RESULTS, regime, cfg["old_seed"])
        new_rows = load_rows(NEW_RESULTS, regime, cfg["new_seed"])
        if set(old_rows) != set(new_rows) or set(old_rows) != set(old_entries):
            raise ValueError(f"Cache/panel source scene IDs differ in {regime}; refusing to drop scenes")
        old_ckpt = {str(row.get("checkpoint_sha256")) for _, row in old_rows.values()}
        new_ckpt = {str(row.get("checkpoint_sha256")) for _, row in new_rows.values()}
        if old_ckpt != {cfg["checkpoint_sha256"]} or new_ckpt != {cfg["checkpoint_sha256"]}:
            raise ValueError(f"Checkpoint mismatch in {regime}: {old_ckpt}, {new_ckpt}")
        per_scene: list[dict[str, Any]] = []
        visible_counts: list[int] = []
        manifest_count_mismatches = 0
        regime_strong_data: list[dict[str, Any]] = []
        for sid in sorted(old_entries):
            old_path, old_row = old_rows[sid]
            new_path, new_row = new_rows[sid]
            oe, ne = old_entries[sid], new_entries[sid]
            old_panel_path, new_panel_path = OLD_PANELS / str(oe["path"]), NEW_PANELS / str(ne["path"])
            old_panel_sha, new_panel_sha = sha256_file(old_panel_path), sha256_file(new_panel_path)
            if old_panel_sha != oe.get("sha256") or new_panel_sha != ne.get("sha256"):
                raise ValueError(f"Panel content SHA mismatch for {sid}")
            if old_row.get("source_sha256") != old_panel_sha or new_row.get("source_sha256") != new_panel_sha:
                raise ValueError(f"Result-row source SHA mismatch for {sid}/{regime}")
            if old_row.get("manifest_sha256") != old_manifest_sha or new_row.get("manifest_sha256") != new_manifest_sha:
                raise ValueError(f"Manifest SHA mismatch for {sid}/{regime}")
            if old_row.get("model_id") != f"action_strength/lewm/original/s3073" and regime == "original":
                raise ValueError(f"Unexpected old original model ID for {sid}")
            expected_model_id = f"action_strength/lewm/{regime}/{cfg['old_seed']}"
            if old_row.get("model_id") != expected_model_id:
                raise ValueError(f"Old model_id mismatch for {sid}: {old_row.get('model_id')}")
            new_model_id = f"action_strength/lewm/{regime}/{cfg['new_seed']}"
            if new_row.get("model_id") != new_model_id:
                raise ValueError(f"Visible model_id mismatch for {sid}: {new_row.get('model_id')}")

            with np.load(old_panel_path, allow_pickle=False) as old_panel, np.load(new_panel_path, allow_pickle=False) as new_panel:
                # Require the same source history, hidden conditions, context actions, and horizon labels.
                for field in ("history_pixels", "context_actions", "conditions", "physical_steps"):
                    if not np.array_equal(old_panel[field], new_panel[field]):
                        raise ValueError(f"Source/history/condition/horizon mismatch for {sid}: {field}")
                if list(np.asarray(old_panel["physical_steps"], dtype=int)) != EXPECTED_HORIZONS:
                    raise ValueError(f"Unexpected old horizons for {sid}")
                old_physical, old_agent = selected_geometry(np.asarray(old_panel["future_states"]))
                new_physical, new_agent = selected_geometry(np.asarray(new_panel["future_states"]))
                old_actions = np.asarray(old_panel["candidate_actions"])
                new_actions = np.asarray(new_panel["candidate_actions"])
                if old_actions.shape != (11, 5, 5, 2):
                    raise ValueError(f"Old action axis should be exactly 11 for {sid}")
                if new_actions.ndim != 4 or new_actions.shape[1:] != (5, 5, 2):
                    raise ValueError(f"Invalid visible candidate action shape for {sid}: {new_actions.shape}")
                if new_physical.shape[1] != new_actions.shape[0]:
                    raise ValueError(f"Visible panel candidate axis mismatch for {sid}")
                old_slots, visible_ids, unchanged_matches = unchanged_mapping(old_panel, new_panel)
                conditions = np.asarray(new_panel["conditions"], dtype=np.float64)
                slot_to_unique_values = np.asarray(new_panel["candidate_slot_to_unique"], dtype=int).tolist()
                candidate_factor_values = np.asarray(new_panel["candidate_factor"], dtype=float).tolist()
                old_history_sha = sha256_array(np.asarray(old_panel["history_pixels"]))
                new_history_sha = sha256_array(np.asarray(new_panel["history_pixels"]))
                old_context_sha = sha256_array(np.asarray(old_panel["context_actions"]))
                new_context_sha = sha256_array(np.asarray(new_panel["context_actions"]))

            old_target, old_feature_sha = target_from_row(old_path, old_row, expected_c=11)
            actual_visible_c = int(new_actions.shape[0])
            if int(new_row.get("candidates", -1)) != actual_visible_c:
                raise ValueError(f"Visible result candidate metadata disagrees with panel for {sid}")
            new_target, new_feature_sha = target_from_row(new_path, new_row, expected_c=actual_visible_c)
            visible_counts.append(actual_visible_c)
            manifest_count = int(ne.get("candidate_unique_count", -1))
            manifest_count_matches = manifest_count == actual_visible_c
            if not manifest_count_matches:
                manifest_count_mismatches += 1

            old_metrics = per_scene_geometry(old_physical, old_agent, old_target)
            new_metrics = per_scene_geometry(new_physical, new_agent, new_target)
            unchanged = unchanged_diagnostics(old_physical, new_physical, old_agent, new_agent,
                                              old_target, new_target, old_slots, visible_ids)
            source_cluster = cluster_by_scene[sid]
            hist_digest[regime].update(sid.encode() + b"\0" + bytes.fromhex(new_history_sha))

            scene_row = {
                "regime": regime,
                "scene_id": sid,
                "source_index": int(oe["source_index"]),
                "source_cluster": source_cluster,
                "source_cluster_matches_old_and_visible": True,
                "checkpoint_sha256": cfg["checkpoint_sha256"],
                "old_seed": cfg["old_seed"],
                "visible_seed": cfg["new_seed"],
                "old_panel_source_sha256": old_panel_sha,
                "visible_panel_source_sha256": new_panel_sha,
                "old_features_archive_sha256": old_feature_sha,
                "visible_features_archive_sha256": new_feature_sha,
                "history_pixels_exactly_equal": old_history_sha == new_history_sha,
                "history_pixels_sha256": new_history_sha,
                "context_actions_exactly_equal": old_context_sha == new_context_sha,
                "conditions": conditions.tolist(),
                "physical_steps": EXPECTED_HORIZONS,
                "old_unique_candidate_count": int(old_target.shape[1]),
                "visible_unique_candidate_count_from_arrays": actual_visible_c,
                "visible_manifest_candidate_unique_count": manifest_count,
                "visible_manifest_count_matches_arrays": manifest_count_matches,
                "candidate_slot_to_unique": slot_to_unique_values,
                "candidate_factor": candidate_factor_values,
                "factor_one_exact_unchanged_slot_matches": unchanged_matches,
                "unchanged_unique_candidate_count": len(old_slots),
                "old": old_metrics,
                "visible": new_metrics,
                "unchanged_candidate_subset": unchanged,
            }
            per_scene.append(scene_row)
            output_rows.append(scene_row)
            regime_strong_data.append({
                "scene_id": sid,
                "agent_xy": new_agent,
                "target": new_target,
                "conditions": conditions,
                "source_sha256": new_panel_sha,
                "features_archive_sha256": new_feature_sha,
                "target_content_sha256": sha256_array(np.asarray(new_target, dtype="<f8")),
                "checkpoint_sha256": cfg["checkpoint_sha256"],
            })

        if len(per_scene) != 256:
            raise ValueError(f"Internal coverage failure for {regime}: {len(per_scene)} scenes")
        metric_paths = {
            "all6": ("all6", False),
            "task_agent_xy": ("task_agent_xy", False),
            "all6_strong_subset": ("all6", True),
            "task_agent_xy_strong_subset": ("task_agent_xy", True),
        }
        summary_metrics: dict[str, Any] = {}
        source_clusters = [cluster_by_scene[row["scene_id"]] for row in per_scene]
        for name, (key, strong) in metric_paths.items():
            old_values: dict[str, list[float | None]] = {}
            new_values: dict[str, list[float | None]] = {}
            for rate_name in ("reversal_over_both_strict", "reversal_over_physical_strict", "latent_tie_fraction_over_physical_strict"):
                old_values[rate_name] = []
                new_values[rate_name] = []
                for row in per_scene:
                    o = row["old"][key]["strong_subset"] if strong else row["old"][key]
                    n = row["visible"][key]["strong_subset"] if strong else row["visible"][key]
                    old_values[rate_name].append(o[rate_name])
                    new_values[rate_name].append(n[rate_name])
            summary_metrics[name] = {
                "old": {k: bootstrap_summary(v, source_clusters, draws) for k, v in old_values.items()},
                "visible": {k: bootstrap_summary(v, source_clusters, draws) for k, v in new_values.items()},
                "paired_difference_visible_minus_old": {
                    k: paired_summary(old_values[k], new_values[k], source_clusters, draws)
                    for k in old_values
                },
            }

        unchanged_rows = [r for r in per_scene if r["unchanged_candidate_subset"]["included"]]
        unchanged_summary: dict[str, Any] = {
            "included_scenes": len(unchanged_rows),
            "excluded_scenes_fewer_than_3_unchanged_unique_candidates": 256 - len(unchanged_rows),
            "candidate_count_distribution_included": {},
        }
        for row in unchanged_rows:
            n = str(row["unchanged_candidate_subset"]["unique_candidate_count"])
            unchanged_summary["candidate_count_distribution_included"][n] = unchanged_summary["candidate_count_distribution_included"].get(n, 0) + 1
        for label, side in (("old", "old_ordering"), ("visible", "visible_ordering")):
            for metric_key in ("all6", "task_agent_xy"):
                vals = [r["unchanged_candidate_subset"][side][metric_key]["reversal_over_both_strict"] for r in unchanged_rows]
                cls = [cluster_by_scene[r["scene_id"]] for r in unchanged_rows]
                unchanged_summary.setdefault(label, {})[metric_key] = bootstrap_summary(vals, cls, draws)
        for metric_key in ("all6", "task_agent_xy"):
            ov = [r["unchanged_candidate_subset"]["old_ordering"][metric_key]["reversal_over_both_strict"] for r in unchanged_rows]
            nv = [r["unchanged_candidate_subset"]["visible_ordering"][metric_key]["reversal_over_both_strict"] for r in unchanged_rows]
            cls = [cluster_by_scene[r["scene_id"]] for r in unchanged_rows]
            unchanged_summary.setdefault("paired_difference_visible_minus_old", {})[metric_key] = paired_summary(ov, nv, cls, draws)

        results[regime] = {
            "model_id_old": f"action_strength/lewm/{regime}/{cfg['old_seed']}",
            "model_id_visible": f"action_strength/lewm/{regime}/{cfg['new_seed']}",
            "checkpoint_sha256": cfg["checkpoint_sha256"],
            "scene_count": len(per_scene),
            "source_cluster_count": len(set(source_clusters)),
            "old_seed": cfg["old_seed"],
            "visible_seed": cfg["new_seed"],
            "visible_unique_candidate_count_histogram_from_arrays": {str(c): visible_counts.count(c) for c in sorted(set(visible_counts))},
            "visible_manifest_candidate_count_mismatches": manifest_count_mismatches,
            "source_histories_sha256_old_and_visible_equal": all(r["history_pixels_exactly_equal"] for r in per_scene),
            "source_context_actions_equal": all(r["context_actions_exactly_equal"] for r in per_scene),
            "same_conditions_and_horizons_verified": True,
            "geometry_metrics": summary_metrics,
            "factor_one_exact_unchanged_candidate_subset": unchanged_summary,
            "per_scene_count": len(per_scene),
        }
        strong_reversal_scan[regime] = regime_strong_data

    example = first_agent_xy_strong_reversal(strong_reversal_scan)
    OUT.mkdir(parents=True, exist_ok=True)
    jsonl_path = OUT / "visible_geometry_queries.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in output_rows:
            f.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    jsonl_sha = sha256_file(jsonl_path)
    output = {
        "schema": "contextworld.visible_geometry_comparison.v1",
        "task": "action_strength",
        "as_of": "2026-10-08",
        "comparison": "original vs visible-canvas candidate set, using each version's full unique candidate set",
        "inputs": {
            "old_cache_root": str(OLD_RESULTS),
            "visible_cache_root": str(NEW_RESULTS),
            "old_panel_root": str(OLD_PANELS),
            "visible_panel_root": str(NEW_PANELS),
            "old_panel_manifest_sha256": old_manifest_sha,
            "visible_panel_manifest_sha256": new_manifest_sha,
            "native_target_layout": "target only; [condition=2, version-specific unique candidate count, horizon=5, native dimension=192]; pred not read",
            "checkpoint_sha256_by_regime": {k: v["checkpoint_sha256"] for k, v in REGIMES.items()},
            "seeds": {k: {"old": v["old_seed"], "visible": v["new_seed"]} for k, v in REGIMES.items()},
            "exact_scene_matches": 256,
            "source_cluster_count": 243,
            "source_history_conditions_horizons": "all 256 matched by exact scene_id and source_index; history_pixels, context_actions, conditions, and physical_steps [5,10,15,20,25] checked equal",
            "visible_candidate_count_authority": "actual candidate_actions axis, cross-checked against target shape and result-row candidates; manifest candidate_unique_count mismatches are explicitly counted and retained",
        },
        "metric_definition": {
            "physical_all6": "selected state components 0:4 plus 40*sin/cos(angle index 4), pixels-equivalent",
            "task_agent_xy": "future_states components 0:2, pixels-equivalent",
            "distance": "mean across five horizons of squared Euclidean distance; all latent dimensions included",
            "primary": "for each true trajectory anchor, compare all pairs of other true trajectories; count reversed order only when physical and native ties both excluded; equal-scene mean",
            "strong_subset": "larger physical RMS >=2x smaller and larger RMS >=1 px-equivalent",
            "tolerance": {"absolute": ABS_TOL, "relative": REL_TOL, "meaning": "numerical tie tolerance only"},
            "bootstrap": {"samples": N_BOOT, "seed": SEED, "cluster_source": "matching old/new panel bootstrap_cluster by scene_id; equal-scene source-cluster bootstrap"},
        },
        "interpretation_limits": [
            "This measures ordering among true trajectory targets, not model-predicted physical error.",
            "Visible-minus-old deltas reflect the changed candidate action distribution and cannot be attributed to a pure off-screen effect.",
            "No model prediction, physical readout, training, or inference is used.",
            "The factor-one exact-action subset is a coverage-limited comparison; it does not replace the full-candidate primary comparison.",
            "No reversal is written up as a model error or root cause.",
        ],
        "unchanged_candidate_subset_semantics": "Keep unique visible candidates whose originating slot factor is exactly 1 and whose full slot action tensor is exactly array-equal to the same old candidate slot; deduplicate by candidate_slot_to_unique; include a scene only when at least 3 unique matched actions exist, while retaining every scene and exclusion reason in JSONL.",
        "first_visible_task_agent_xy_strong_reversal": example,
        "scene_level_jsonl": str(jsonl_path),
        "scene_level_jsonl_sha256": jsonl_sha,
        "cells": results,
        "all_scene_jsonl_rows": len(output_rows),
    }
    out_path = OUT / "visible_geometry.json"
    out_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(out_path), "jsonl": str(jsonl_path), "scene_rows": len(output_rows), "strong_reversal_found": example is not None}, sort_keys=True))


if __name__ == "__main__":
    main()
