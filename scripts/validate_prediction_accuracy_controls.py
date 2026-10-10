#!/usr/bin/env python3
"""Score tolerance-based controls directly against materialized true futures.

This data-only diagnostic never loads model predictions or decodes panel
pixels for scoring. Its optional Action Strength sidecar path reads original
source images only to verify the frozen history prefix exactly. It uses the
repository's ``physical_targets_from_panel`` mapping and reports macro PCK-style
accuracy (strict error < tolerance), first averaging conditions and candidates
within each scene, then averaging scenes. PushT visible-future panels and legacy
TwoRoom Speed/Door panels stay separate.

The selected tolerances are exploratory, borrowed from TAP-style pixel
thresholds. They are not declared task-adequacy criteria.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from check_icl_measurement_panel import physical_targets_from_panel  # noqa: E402


PUSHT_TASKS = ("action_strength", "contact_friction", "motion_damping")
TWOROOM_TASKS = ("speed", "door")
PUSHT_TAUS_WORLD_512 = np.asarray([2.0, 4.0, 8.0, 16.0, 32.0])
TWOROOM_TAUS_EQ256 = np.asarray([1.0, 2.0, 4.0, 8.0, 16.0])
OFFSET_NORMS = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)
CURRENT_SCALES = (0.0, 0.5, 1.0, 1.5, 2.0)
HORIZONS = (5, 10, 15, 20, 25)
THRESHOLD_SUBSETS = {
    "finest3": (0, 1, 2),
    "all5": (0, 1, 2, 3, 4),
    "coarsest3": (2, 3, 4),
}


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot encode {type(value).__name__} as JSON")


def _scene_records(panel_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = panel_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = manifest.get("scenes")
    if not isinstance(rows, list):
        raise ValueError(f"{manifest_path} does not contain a scenes list")
    records = []
    for ordinal, raw in enumerate(rows):
        row = dict(raw)
        scene_id = str(row.get("scene_id") or row.get("pair_id") or row.get("id") or ordinal)
        relpath = row.get("path") or row.get("file") or f"{scene_id}.npz"
        path = panel_dir / str(relpath)
        if not path.is_file():
            raise FileNotFoundError(f"Missing scene panel for {scene_id}: {path}")
        records.append({"scene_id": scene_id, "path": path, "row": row})
    return manifest, records


def _sha256_path(path: Path) -> str:
    """Hash one file or a directory tree using sorted relative file names."""
    path = Path(path)
    if path.is_file():
        return hashlib.sha256(path.read_bytes()).hexdigest()
    if not path.is_dir():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    for item in sorted(p for p in path.rglob("*") if p.is_file()):
        relative = item.relative_to(path).as_posix().encode("utf-8")
        file_digest = hashlib.sha256(item.read_bytes()).digest()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(file_digest)
    return digest.hexdigest()


def build_strength_query_state_sidecar(
    output_path: Path,
    *,
    pusht_root: Path,
    payload_root: Path | None = None,
) -> dict[str, Any]:
    """Derive query states from the source Lance payload and verify panel prefixes.

    This imports the registered Development payload reader and the Action
    Strength Lance reader used by the panel builder. It performs no simulation.
    Every low/high historical image and both context action blocks are compared
    bit-for-bit against the corresponding frozen panel before states are saved.
    """
    panel_dir = Path(pusht_root) / "action_strength"
    panel_manifest, records = _scene_records(panel_dir)
    builder_inputs = panel_manifest.get("builder_inputs")
    if not isinstance(builder_inputs, dict):
        raise ValueError("Action Strength manifest is missing builder_inputs")
    contextworld_root = str(builder_inputs["contextworld_root"])
    stable_worldmodel_root = str(builder_inputs["stable_worldmodel_snapshot"])
    resolved_payload_root = Path(
        payload_root or builder_inputs["development_payload_root"]
    )

    # Match builder_visible_strength.install_import_paths and its verified
    # Development-payload path; no environment/simulator module is imported.
    for import_path in (contextworld_root, stable_worldmodel_root):
        if import_path not in sys.path:
            sys.path.insert(0, import_path)
    sys.modules.setdefault("flash_attn", None)
    from contextworld.benchmarks import bundle_development as bd  # noqa: PLC0415
    from contextworld.benchmarks.action_strength_icl_data import (  # noqa: PLC0415
        _read_lance_pairs,
    )

    payload = bd.resolve_development_payload(
        resolved_payload_root, task="action_strength"
    )
    if len(payload.members) != 1:
        raise ValueError(
            f"Expected one Action Strength Development member, got {payload.members}"
        )
    member_path = payload.members[0]
    arrays = _read_lance_pairs(member_path, expected_pairs=256)
    source_ids = list(arrays.pair_ids)
    panel_source_ids = panel_manifest.get("source_pair_ids")
    if not isinstance(panel_source_ids, list) or source_ids != [str(x) for x in panel_source_ids]:
        raise ValueError("Lance pair IDs/order do not exactly match the panel source_pair_ids")
    panel_ids = [record["scene_id"] for record in records]
    if source_ids != panel_ids:
        raise ValueError("Lance pair IDs/order do not exactly match panel scene IDs")
    if len(records) != 256:
        raise ValueError(f"Expected all 256 Action Strength panels, got {len(records)}")

    states_by_scene: dict[str, list[list[float]]] = {}
    for source_index, record in enumerate(records):
        with np.load(record["path"], allow_pickle=False) as archive:
            panel_history = np.asarray(archive["history_pixels"])
            panel_context = np.asarray(archive["context_actions"])
        expected_history = np.stack(
            [
                arrays.low_pixels[source_index, :3],
                arrays.high_pixels[source_index, :3],
            ],
            axis=0,
        )
        expected_context = np.broadcast_to(
            arrays.raw_action_blocks[source_index, :2][None, ...],
            (2, 2, 5, 2),
        )
        if not np.array_equal(panel_history, expected_history):
            raise ValueError(
                f"{record['scene_id']}: frozen history_pixels differ from the source Lance frames"
            )
        if not np.array_equal(panel_context, expected_context):
            raise ValueError(
                f"{record['scene_id']}: frozen context_actions differ from source Lance blocks"
            )
        # The stored query is source frame index 2: after two context blocks,
        # at the final history frame and immediately before the future action.
        query = np.stack(
            [
                arrays.low_states[source_index, 2],
                arrays.high_states[source_index, 2],
            ],
            axis=0,
        ).astype(np.float32, copy=False)
        states_by_scene[record["scene_id"]] = query.astype(np.float64).tolist()

    source_ids_sha256 = hashlib.sha256(
        ("\n".join(source_ids) + "\n").encode("utf-8")
    ).hexdigest()
    all_query_states = np.stack(
        [np.asarray(states_by_scene[scene_id], dtype=np.float32) for scene_id in source_ids]
    )
    query_states_sha256 = hashlib.sha256(
        np.ascontiguousarray(all_query_states).tobytes()
    ).hexdigest()
    source_manifest_path = Path(str(panel_manifest.get("source", "")))
    source_manifest_sha256 = (
        _sha256_path(source_manifest_path) if source_manifest_path.is_file() else None
    )
    source_manifest_expected = panel_manifest.get("source_sha256")
    if (
        source_manifest_sha256 is not None
        and source_manifest_expected is not None
        and source_manifest_sha256 != source_manifest_expected
    ):
        raise ValueError(
            "Action Strength source release manifest hash differs from the frozen panel receipt"
        )

    sidecar = {
        "schema": "contextworld.action_strength_query_states.v1",
        "states_by_scene_id": states_by_scene,
        "provenance": {
            "derived_from": "registered original Action Strength Development Lance data; no simulator",
            "source_uri": panel_manifest.get("source"),
            "source_release_manifest_sha256": source_manifest_sha256 or source_manifest_expected,
            "source_payload_root": str(payload.root),
            "source_payload_member": str(member_path),
            "source_payload_member_tree_sha256": _sha256_path(member_path),
            "source_payload_manifest_sha256": payload.manifest_sha256,
            "source_task_registry_sha256": payload.task_registry_sha256,
            "source_pair_count": len(source_ids),
            "source_pair_ids_sha256": source_ids_sha256,
            "query_state_array_sha256_float32": query_states_sha256,
            "source_query_state_frame_index": 2,
            "source_state_frame_semantics": "four recorded frames are [initial, after context block 1, after context block 2/current query, after the source query action block]",
            "source_history_pixels_bitwise_verified": True,
            "source_context_actions_bitwise_verified": True,
            "verified_scene_count": len(records),
            "verified_history_frame_count": len(records) * 2 * 3,
            "verified_context_action_block_count": len(records) * 2 * 2,
            "panel_manifest_sha256": _sha256_path(panel_dir / "manifest.json"),
        },
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(sidecar, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return sidecar


def _load_strength_query_states(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("--strength-query-states must contain a JSON object")
    if isinstance(document.get("states_by_scene_id"), dict):
        mapping = document["states_by_scene_id"]
    elif isinstance(document.get("states"), dict):
        mapping = document["states"]
    else:
        # Also accept a plain portable {scene_id: [K,D]} JSON mapping.
        mapping = document
    provenance = document.get("provenance", {})
    if not isinstance(provenance, dict):
        provenance = {"provided_provenance": provenance}
    return dict(mapping), {"path": str(Path(path).resolve()), **provenance}


def _physical_current_state(
    panel: dict[str, np.ndarray],
    task: str,
    manifest_row: dict[str, Any],
    k: int,
    query_state_override: Any | None = None,
) -> np.ndarray | None:
    """Return [K,1,1,P] current query state using stored state values only."""
    if query_state_override is not None:
        raw = np.asarray(query_state_override, dtype=np.float64)
    elif "query_state" in panel:
        raw = np.asarray(panel["query_state"], dtype=np.float64)
    elif "query_state" in manifest_row:
        raw = np.asarray(manifest_row["query_state"], dtype=np.float64)
    else:
        return None

    if raw.ndim == 1:
        raw = np.broadcast_to(raw[None, :], (k, raw.shape[0])).copy()
    if raw.ndim != 2 or raw.shape[0] != k:
        raise ValueError(
            f"{task} query_state must be [D] or [K,D] with K={k}; got {raw.shape}"
        )
    mapped = physical_targets_from_panel(
        {"future_states": raw[:, None, None, :]}, task
    )
    current = mapped[:, :, :, :]
    if current.shape[1:3] != (1, 1):
        raise AssertionError(f"Unexpected current target shape {current.shape}")
    return current


def _object_specs(task: str) -> list[dict[str, Any]]:
    if task in PUSHT_TASKS:
        relevant = "agent" if task == "action_strength" else "block"
        return [
            {"name": "agent", "start": 0, "stop": 2, "task_relevant": relevant == "agent"},
            {
                "name": "block",
                "start": 2,
                "stop": 6,
                "task_relevant": relevant == "block",
            },
        ]
    return [{"name": "agent", "start": 0, "stop": 2, "task_relevant": True}]


def _prediction_controls(
    truth: np.ndarray, current: np.ndarray | None
) -> tuple[list[tuple[str, str, np.ndarray]], list[dict[str, str]]]:
    """Construct controls for one object's [K,C,T,D] true future."""
    k, c, t, d = truth.shape
    controls: list[tuple[str, str, np.ndarray]] = [("exact_truth", "", truth)]
    for offset in OFFSET_NORMS:
        prediction = truth.copy()
        prediction[..., 0] += float(offset)
        controls.append(("fixed_norm_offset", f"{offset:g}", prediction))

    unavailable: list[dict[str, str]] = []
    if current is None:
        unavailable.extend(
            [
                {
                    "control": "current_state_displacement_scale",
                    "reason": "No physical query_state is stored in this panel or supplied by the portable sidecar; no state is inferred from pixels or future samples.",
                },
                {
                    "control": "temporal_lag_prepend_current",
                    "reason": "No physical query_state is stored in this panel or supplied by the portable sidecar; no state is inferred from pixels or future samples.",
                },
            ]
        )
    else:
        if current.shape[0] != k or current.shape[-1] != d:
            raise ValueError(
                f"current state {current.shape} incompatible with object future {truth.shape}"
            )
        for scale in CURRENT_SCALES:
            prediction = current + float(scale) * (truth - current)
            controls.append(("current_state_displacement_scale", f"{scale:g}", prediction))
        current_first = np.broadcast_to(current, (k, c, 1, d))
        lagged = np.concatenate([current_first, truth[:, :, :-1, :]], axis=2)
        controls.append(("temporal_lag_prepend_current", "one_horizon", lagged))

    # For K conditions, each prediction is the next condition in a fixed cycle.
    # This is guaranteed to select a different condition when K > 1.
    controls.append(("wrong_condition_true_future", "cyclic_shift_by_1", np.roll(truth, 1, axis=0)))

    common_mean = np.mean(truth, axis=0, keepdims=True, dtype=np.float64)
    common_mean = np.broadcast_to(common_mean, truth.shape)
    controls.append(
        ("condition_mean_common_privileged", "mean_across_hidden_conditions", common_mean)
    )
    return controls, unavailable


def binary_history_blind_ceiling(
    true_future: np.ndarray, tau: float
) -> np.ndarray:
    """Return the binary-condition common-point upper bound per [K,C,T] unit."""
    truth = np.asarray(true_future, dtype=np.float64)
    if truth.ndim != 4 or truth.shape[0] != 2:
        raise ValueError(f"binary ceiling requires [2,C,T,D], got {truth.shape}")
    if not np.isfinite(truth).all() or not np.isfinite(tau) or tau <= 0.0:
        raise ValueError("binary ceiling requires finite targets and positive tau")
    gap = np.linalg.norm(truth[0] - truth[1], axis=-1)
    bound = np.where(gap < 2.0 * float(tau), 1.0, 0.5)
    return np.broadcast_to(bound[None, ...], truth.shape[:-1]).copy()


def _tau_configuration(task: str) -> tuple[np.ndarray, np.ndarray, str]:
    if task in PUSHT_TASKS:
        # The PushT state coordinate spans 512 world units; 1 source-256 pixel
        # therefore corresponds to 2 world units.
        return PUSHT_TAUS_WORLD_512.copy(), np.asarray([1, 2, 4, 8, 16]), "512-world-pixels"
    # TwoRoom uses direct 224×224 canvas coordinates (confirmed from the
    # checked-out TwoRoomEnv, whose state and rendering both use IMG_SIZE=224).
    return TWOROOM_TAUS_EQ256 * (224.0 / 256.0), TWOROOM_TAUS_EQ256.copy(), "224-canvas-pixels"


def _load_truth(panel_path: Path, task: str) -> tuple[dict[str, np.ndarray], np.ndarray]:
    with np.load(panel_path, allow_pickle=False) as archive:
        if "future_states" not in archive.files:
            raise KeyError(f"{panel_path} is missing future_states")
        panel = {"future_states": np.asarray(archive["future_states"])}
        if "query_state" in archive.files:
            panel["query_state"] = np.asarray(archive["query_state"])
    if "future_states" not in panel:
        raise KeyError(f"{panel_path} is missing future_states")
    truth = physical_targets_from_panel(panel, task)
    if truth.ndim != 4:
        raise ValueError(f"{panel_path}: mapped future targets must be [K,C,T,P], got {truth.shape}")
    return panel, truth


def _macro(value_by_scene: dict[tuple[Any, ...], list[float]]) -> dict[tuple[Any, ...], float]:
    return {key: float(np.mean(values, dtype=np.float64)) for key, values in value_by_scene.items()}


def validate(
    pusht_root: Path,
    tworoom_root: Path,
    *,
    strength_query_states: dict[str, Any] | None = None,
    strength_state_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    # Every listed panel is consumed; no scene is dropped based on its score.
    score_by_scene: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    ceiling_by_scene: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    scene_counts: dict[str, int] = {}
    current_gaps: dict[str, list[float]] = defaultdict(list)
    unavailable_controls: dict[str, list[dict[str, str]]] = defaultdict(list)
    condition_counts: dict[str, set[int]] = defaultdict(set)
    candidate_counts: dict[str, list[int]] = defaultdict(list)
    exact_all_100 = True
    offset_monotone = True
    common_mean_under_ceiling = True

    for task in (*PUSHT_TASKS, *TWOROOM_TASKS):
        source_root = pusht_root if task in PUSHT_TASKS else tworoom_root
        panel_dir = source_root / task
        manifest, records = _scene_records(panel_dir)
        source_family = "visible_pusht_futures_2026-10-08" if task in PUSHT_TASKS else "legacy_tworoom_multistep_panels_2026-09-30"
        scene_counts[task] = len(records)
        tau_values, equivalent_256, tau_units = _tau_configuration(task)
        expected_k = 3 if task == "speed" else 2
        if task == "action_strength" and strength_query_states is not None:
            expected_ids = {record["scene_id"] for record in records}
            supplied_ids = set(strength_query_states)
            missing = sorted(expected_ids - supplied_ids)
            extra = sorted(supplied_ids - expected_ids)
            if missing or extra:
                raise ValueError(
                    "--strength-query-states must map every Action Strength scene ID exactly; "
                    f"missing={missing[:5]}, extra={extra[:5]}"
                )

        for record in records:
            panel, full_truth = _load_truth(record["path"], task)
            k, c, t, physical_dim = full_truth.shape
            if k != expected_k or t != len(HORIZONS):
                raise ValueError(
                    f"{task}/{record['scene_id']}: expected K={expected_k}, T=5; got {full_truth.shape}"
                )
            if not np.isfinite(full_truth).all():
                raise ValueError(f"{task}/{record['scene_id']}: nonfinite target coordinates")
            condition_counts[task].add(k)
            candidate_counts[task].append(c)
            query_state_override = None
            if task == "action_strength" and strength_query_states is not None:
                query_state_override = strength_query_states[record["scene_id"]]
            current_full = _physical_current_state(
                panel, task, record["row"], k, query_state_override
            )
            if current_full is not None:
                gap = float(np.max(np.abs(current_full - current_full[:1])))
                current_gaps[task].append(gap)
                if gap > 1e-4:
                    raise ValueError(
                        f"{task}/{record['scene_id']}: query state differs across conditions by {gap:g}"
                    )

            for object_spec in _object_specs(task):
                object_name = object_spec["name"]
                obj_truth = full_truth[..., object_spec["start"] : object_spec["stop"]]
                obj_current = (
                    current_full[..., object_spec["start"] : object_spec["stop"]]
                    if current_full is not None
                    else None
                )
                controls, unavailable = _prediction_controls(obj_truth, obj_current)
                for item in unavailable:
                    entry = dict(item)
                    entry.update({"object": object_name, "status": "unavailable"})
                    if entry not in unavailable_controls[task]:
                        unavailable_controls[task].append(entry)

                score_by_tau: dict[tuple[str, str, int], list[float]] = {}
                ceiling_for_tau: dict[tuple[str, int], float | None] = {}
                for tau_index, (tau, equiv) in enumerate(zip(tau_values, equivalent_256)):
                    # The history-blind ceiling is an exact binary-condition bound.
                    if k == 2:
                        ceiling_samples = binary_history_blind_ceiling(obj_truth, tau)
                        ceiling_by_horizon = np.mean(
                            ceiling_samples, axis=(0, 1), dtype=np.float64
                        )
                        for h, ceiling_score in enumerate(ceiling_by_horizon):
                            ceiling_by_scene[(task, object_name, tau_index, h)].append(
                                float(ceiling_score)
                            )
                        ceiling_for_tau[("ceiling", tau_index)] = ceiling_by_horizon
                    else:
                        ceiling_for_tau[("ceiling", tau_index)] = None

                    for control, parameter, prediction in controls:
                        error = np.linalg.norm(prediction - obj_truth, axis=-1)
                        within = (error < float(tau)).astype(np.float64)
                        # Equal-weight conditions and candidates within a scene;
                        # scene means are averaged only after all scenes are processed.
                        scene_by_horizon = np.mean(within, axis=(0, 1), dtype=np.float64)
                        for h, score in enumerate(scene_by_horizon):
                            key = (task, object_name, control, parameter, tau_index, h)
                            score_by_scene[key].append(float(score))
                        score_by_tau[(control, parameter, tau_index)] = [
                            float(x) for x in scene_by_horizon
                        ]

                # Check exact oracle and monotonicity on the fixed-offset ladder.
                for tau_index in range(len(tau_values)):
                    exact = score_by_tau[("exact_truth", "", tau_index)]
                    exact_all_100 = exact_all_100 and all(x == 1.0 for x in exact)
                    for h in range(len(HORIZONS)):
                        offset_scores = [
                            score_by_tau[("fixed_norm_offset", f"{v:g}", tau_index)][h]
                            for v in OFFSET_NORMS
                        ]
                        offset_monotone = offset_monotone and all(
                            left >= right for left, right in zip(offset_scores, offset_scores[1:])
                        )
                        if k == 2:
                            common_score = score_by_tau[(
                                "condition_mean_common_privileged",
                                "mean_across_hidden_conditions",
                                tau_index,
                            )][h]
                            bound = ceiling_for_tau[("ceiling", tau_index)]
                            if bound is None or common_score > bound[h] + 1e-12:
                                common_mean_under_ceiling = False

    macro_scores = _macro(score_by_scene)
    macro_ceiling = _macro(ceiling_by_scene)
    threshold_rows: list[dict[str, Any]] = []
    for key, score in sorted(macro_scores.items()):
        task, object_name, control, parameter, tau_index, horizon_index = key
        tau_values, equiv_values, units = _tau_configuration(task)
        tau = float(tau_values[tau_index])
        equiv = float(equiv_values[tau_index])
        ceiling = macro_ceiling.get((task, object_name, tau_index, horizon_index))
        relevant = next(x["task_relevant"] for x in _object_specs(task) if x["name"] == object_name)
        threshold_rows.append(
            {
                "task": task,
                "source_family": "visible_pusht_futures_2026-10-08" if task in PUSHT_TASKS else "legacy_tworoom_multistep_panels_2026-09-30",
                "object": object_name,
                "task_relevant_object": relevant,
                "control": control,
                "parameter": parameter,
                "tau": tau,
                "tau_units": units,
                "equivalent_256px": equiv,
                "horizon_steps": HORIZONS[horizon_index],
                "accuracy": score,
                "accuracy_percent": 100.0 * score,
                "calibration_error_percent": 100.0 * (1.0 - score),
                "history_blind_ceiling": ceiling,
                "history_blind_ceiling_percent": None if ceiling is None else 100.0 * ceiling,
                "accuracy_minus_ceiling_pp": None if ceiling is None else 100.0 * (score - ceiling),
                "scene_count": scene_counts[task],
                "macro_weighting": "equal scenes; within each scene equal conditions and equal candidates; horizons reported separately",
            }
        )

    # Fixed tolerance subsets are averaged as registered; no threshold is selected
    # from scores. Per-scene weighting is equal because every scene contributes a
    # score at every threshold before this equal-threshold average is taken.
    index_by_tau = {"finest3": (0, 1, 2), "all5": (0, 1, 2, 3, 4), "coarsest3": (2, 3, 4)}
    sensitivity_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in threshold_rows:
        base = (
            row["task"], row["source_family"], row["object"], row["task_relevant_object"],
            row["control"], row["parameter"], row["horizon_steps"],
        )
        tau_index = int(round(np.log2(row["equivalent_256px"])))
        for subset, indices in index_by_tau.items():
            if tau_index in indices:
                sensitivity_groups[base + (subset,)].append(row)

    sensitivity_rows = []
    for key, values in sorted(sensitivity_groups.items()):
        task, source_family, object_name, relevant, control, parameter, horizon, subset = key
        score = float(np.mean([x["accuracy"] for x in values]))
        ceilings = [x["history_blind_ceiling"] for x in values]
        ceiling = None if any(x is None for x in ceilings) else float(np.mean(ceilings))
        sensitivity_rows.append(
            {
                "task": task,
                "source_family": source_family,
                "object": object_name,
                "task_relevant_object": relevant,
                "control": control,
                "parameter": parameter,
                "horizon_steps": horizon,
                "threshold_subset": subset,
                "threshold_count": len(values),
                "accuracy_mean_over_fixed_tolerances": score,
                "accuracy_percent": 100.0 * score,
                "calibration_error_percent": 100.0 * (1.0 - score),
                "history_blind_ceiling_mean": ceiling,
                "history_blind_ceiling_percent": None if ceiling is None else 100.0 * ceiling,
                "accuracy_minus_ceiling_pp": None if ceiling is None else 100.0 * (score - ceiling),
                "threshold_selection": "fixed a priori subset; no score-based optimization",
            }
        )

    # Macro-average the five horizon-specific scene scores equally. This makes
    # early-horizon tolerance saturation visible alongside the full-trajectory
    # calibration result.
    all_horizon_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    relevant_controls = {
        "exact_truth",
        "current_state_displacement_scale",
        "condition_mean_common_privileged",
        "wrong_condition_true_future",
        "temporal_lag_prepend_current",
        "fixed_norm_offset",
    }
    for row in sensitivity_rows:
        if not row["task_relevant_object"] or row["control"] not in relevant_controls:
            continue
        key = (
            row["task"], row["source_family"], row["object"], row["control"],
            row["parameter"], row["threshold_subset"],
        )
        all_horizon_groups[key].append(row)

    all_horizon_summary = []
    for key, horizon_rows in sorted(all_horizon_groups.items()):
        task, source_family, object_name, control, parameter, subset = key
        horizon_rows = sorted(horizon_rows, key=lambda row: int(row["horizon_steps"]))
        if [int(row["horizon_steps"]) for row in horizon_rows] != list(HORIZONS):
            raise ValueError(f"Incomplete five-horizon summary for {key}")
        score = float(np.mean(
            [row["accuracy_mean_over_fixed_tolerances"] for row in horizon_rows],
            dtype=np.float64,
        ))
        ceilings = [row["history_blind_ceiling_mean"] for row in horizon_rows]
        ceiling = None if any(value is None for value in ceilings) else float(
            np.mean(ceilings, dtype=np.float64)
        )
        all_horizon_summary.append(
            {
                "task": task,
                "source_family": source_family,
                "object": object_name,
                "control": control,
                "parameter": parameter,
                "threshold_subset": subset,
                "horizon_count": len(HORIZONS),
                "horizons_included": list(HORIZONS),
                "accuracy_mean_over_all_five_horizons": score,
                "accuracy_percent": 100.0 * score,
                "calibration_error_percent": 100.0 * (1.0 - score),
                "history_blind_ceiling_mean_over_all_five_horizons": ceiling,
                "history_blind_ceiling_percent": None if ceiling is None else 100.0 * ceiling,
                "accuracy_minus_ceiling_pp": None if ceiling is None else 100.0 * (score - ceiling),
                "horizon_weighting": "equal mean over horizons 5,10,15,20,25 after scene/condition/candidate macro averaging",
            }
        )
        if ceiling is not None and control == "condition_mean_common_privileged":
            all_horizon_summary.append(
                {
                    "task": task,
                    "source_family": source_family,
                    "object": object_name,
                    "control": "history_blind_ceiling",
                    "parameter": "binary_condition_upper_bound",
                    "threshold_subset": subset,
                    "horizon_count": len(HORIZONS),
                    "horizons_included": list(HORIZONS),
                    "accuracy_mean_over_all_five_horizons": ceiling,
                    "accuracy_percent": 100.0 * ceiling,
                    "calibration_error_percent": None,
                    "history_blind_ceiling_mean_over_all_five_horizons": ceiling,
                    "history_blind_ceiling_percent": 100.0 * ceiling,
                    "accuracy_minus_ceiling_pp": 0.0,
                    "horizon_weighting": "equal mean over horizons 5,10,15,20,25 after scene/candidate macro averaging",
                }
            )

    tau_metadata = {}
    for task in (*PUSHT_TASKS, *TWOROOM_TASKS):
        tau_values, equiv_values, units = _tau_configuration(task)
        tau_metadata[task] = {
            "thresholds": tau_values.tolist(),
            "threshold_units": units,
            "equivalent_256px": equiv_values.tolist(),
        }

    status = {
        "exact_truth_100_percent_every_horizon_threshold": exact_all_100,
        "fixed_offset_accuracy_monotonic_in_offset": offset_monotone,
        "binary_condition_mean_below_or_at_blind_ceiling": common_mean_under_ceiling,
    }
    if not all(status.values()):
        raise AssertionError(f"Control invariants failed: {status}")

    return {
        "schema": "prediction_accuracy_controls.v1",
        "created_by": str(Path(__file__).resolve()),
        "created_from": {
            "pusht_panels": str(pusht_root),
            "tworoom_panels": str(tworoom_root),
            "tasks": {
                task: {
                    "scene_count": scene_counts[task],
                    "candidate_count_min": min(candidate_counts[task]),
                    "candidate_count_max": max(candidate_counts[task]),
                    "condition_counts": sorted(condition_counts[task]),
                    "distribution": "visible PushT panels (2026-10-08)" if task in PUSHT_TASKS else "older TwoRoom multistep panels (2026-09-30), kept separate",
                }
                for task in (*PUSHT_TASKS, *TWOROOM_TASKS)
            },
        },
        "scope_and_interpretation": {
            "data_only": True,
            "model_predictions_loaded": False,
            "images_decoded_for_metric": False,
            "source_images_decoded_for_prefix_verification": bool(
                strength_state_provenance
                and strength_state_provenance.get("source_history_pixels_bitwise_verified")
            ),
            "future_labels_are_scoring_targets": True,
            "score": "PCK-style fraction with Euclidean object error strictly less than tau",
            "not_a_task_adequacy_claim": "Tolerances are exploratory values borrowed from TAP-style pixel thresholds, not declared task-adequacy criteria.",
            "pushT_coordinates": "512-world-pixel coordinates; candidate thresholds [2,4,8,16,32], equivalent to [1,2,4,8,16] source-256 px.",
            "twoRoom_coordinates": "Direct 224x224 canvas px from TwoRoomEnv.IMG_SIZE and state coordinates; thresholds are [1,2,4,8,16] equivalent-256 px scaled by 224/256.",
            "pushT_object_readout": "agent = physical target dims [0:2]; block = dims [2:6] with 40*sin(theta),40*cos(theta); block norm is px-equivalent, not real point distance.",
            "macro_averaging": "For each scene, mean equally across conditions and candidate futures at each horizon; then mean scene scores equally. Horizons remain separate.",
            "condition_mean_common": "Privileged history-blind reference computed from both conditions' future labels; not an attainable prediction baseline.",
            "wrong_condition_control": "Cyclic wrong-condition truth is a label-swapping control, not an empirical model baseline. For three-condition Speed the cyclic shift covers each unordered pair once across conditions.",
            "calibration_error": "These controls calibrate tolerance stringency; their scores are not rigorous floors for learned predictors because prediction errors can interact with or cancel target bias.",
            "history_blind_ceiling": "For binary-condition tasks, per scene/candidate/horizon/object unit ceiling is 1 when true-target norm gap < 2*tau, otherwise 0.5. Speed has three conditions, so this binary formula is not applied and its ceiling is null.",
            "source_groups_and_intervals": "No confidence intervals are reported, so no bootstrap source-group resampling or MD128 de-duplication is used.",
            "nine_task_coverage": "This validates the three current PushT visible-future tasks plus Speed and Door positive controls only; it does not validate the full nine-task release.",
        },
        "tolerances": tau_metadata,
        "control_definitions": {
            "exact_truth": "prediction equals that unit's true future",
            "fixed_norm_offset": "add a deterministic displacement of the named norm along the first coordinate",
            "current_state_displacement_scale": "query_state + alpha * (true_future - query_state); Action Strength query_state comes from a source-verified sidecar of original frame index 2",
            "temporal_lag_prepend_current": "prediction horizons are [query_state, true_horizon_1, ..., true_horizon_4]",
            "wrong_condition_true_future": "assign each condition the next condition's true future in a deterministic cyclic shift",
            "condition_mean_common_privileged": "give every condition the mean true future across hidden conditions for that scene/candidate/horizon",
            "history_blind_ceiling": "binary-condition upper bound on the fraction of the two futures a single common point can cover at the specified tolerance",
        },
        "current_state_availability": {
            task: {
                "available": bool(current_gaps[task]),
                "max_cross_condition_coordinate_gap": max(current_gaps[task]) if current_gaps[task] else None,
                "unavailable_controls": unavailable_controls.get(task, []),
                "provenance": strength_state_provenance if task == "action_strength" else None,
            }
            for task in (*PUSHT_TASKS, *TWOROOM_TASKS)
        },
        "control_invariants": status,
        "threshold_metrics": threshold_rows,
        "threshold_sensitivity": sensitivity_rows,
        "all_horizon_aggregate_summary": all_horizon_summary,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pusht-panels", type=Path, required=True)
    parser.add_argument("--tworoom-panels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--strength-query-states",
        type=Path,
        help="Portable JSON mapping Action Strength scene IDs to [2,7] physical query states.",
    )
    parser.add_argument(
        "--write-strength-query-states-from-source",
        type=Path,
        help="Build a sidecar from the registered original Lance payload after exact prefix verification, then use it.",
    )
    args = parser.parse_args()

    if args.strength_query_states and args.write_strength_query_states_from_source:
        parser.error("Choose either --strength-query-states or --write-strength-query-states-from-source")
    sidecar_path = args.strength_query_states
    if args.write_strength_query_states_from_source:
        build_strength_query_state_sidecar(
            args.write_strength_query_states_from_source,
            pusht_root=args.pusht_panels,
        )
        sidecar_path = args.write_strength_query_states_from_source

    strength_states = None
    strength_provenance = None
    if sidecar_path is not None:
        strength_states, strength_provenance = _load_strength_query_states(sidecar_path)

    report = validate(
        args.pusht_panels,
        args.tworoom_panels,
        strength_query_states=strength_states,
        strength_state_provenance=strength_provenance,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "prediction_accuracy_controls.json"
    csv_path = args.output_dir / "prediction_accuracy_controls.csv"
    sensitivity_path = args.output_dir / "threshold_sensitivity.csv"
    horizon_summary_path = args.output_dir / "horizon_aggregate_summary.csv"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    _write_csv(csv_path, report["threshold_metrics"])
    _write_csv(sensitivity_path, report["threshold_sensitivity"])
    _write_csv(horizon_summary_path, report["all_horizon_aggregate_summary"])

    compact_horizon_summary = []
    for task, object_spec in (
        (task, spec)
        for task in (*PUSHT_TASKS, *TWOROOM_TASKS)
        for spec in _object_specs(task)
        if spec["task_relevant"]
    ):
        selected = [
            row for row in report["all_horizon_aggregate_summary"]
            if row["task"] == task
            and row["object"] == object_spec["name"]
            and row["threshold_subset"] == "all5"
            and row["control"] in {
                "condition_mean_common_privileged",
                "wrong_condition_true_future",
                "history_blind_ceiling",
                "temporal_lag_prepend_current",
                "fixed_norm_offset",
            }
        ]
        compact_horizon_summary.extend(
            {
                "task": row["task"],
                "object": row["object"],
                "control": row["control"],
                "parameter": row["parameter"],
                "all5_horizon_accuracy_percent": row["accuracy_percent"],
                "calibration_error_percent": row["calibration_error_percent"],
            }
            for row in selected
        )
    print(json.dumps({
        "json": str(json_path),
        "csv": str(csv_path),
        "threshold_sensitivity_csv": str(sensitivity_path),
        "all_five_horizons_csv": str(horizon_summary_path),
        "strength_query_states": None if sidecar_path is None else str(sidecar_path),
        "tasks": {task: values["scene_count"] for task, values in report["created_from"]["tasks"].items()},
        "control_invariants": report["control_invariants"],
        "all_five_horizons_all5_thresholds_relevant_object": compact_horizon_summary,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
