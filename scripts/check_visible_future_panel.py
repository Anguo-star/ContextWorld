#!/usr/bin/env python3
"""Model-free checks and visibility diagnostics for PushT future panels.

This validator reads one scene at a time. Builder receipts are the evidence
that the simulator replay was continuous; this script checks the receipt,
replayed-prefix identity, saved raw-step states, and full-body bounds without
rerunning physics or consulting a learned model.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

TASKS = ("action_strength", "contact_friction", "motion_damping")
LEGACY_ROOT = Path("/tmp/cw-multistep-coverage-20260930/panels")
DEFAULT_PANEL_ROOT = Path("/tmp/cw-visible-futures-20261008/panels")
DEFAULT_REPORT = Path("/tmp/cw-visible-futures-20261008/visible_validation.json")
EXPECTED_SOURCES = 256
HORIZON_STEPS = np.asarray([5, 10, 15, 20, 25], dtype=np.int64)
RAW_FUTURE_STEPS = 25
CANDIDATE_FACTORS = np.asarray([1.0, 0.75, 0.5, 0.25, 0.125, 0.0625, 0.0], dtype=np.float32)
CANVAS_MIN = 2.0
CANVAS_MAX = 510.0
RENDER_SIZE = 224
WORLD_PER_RENDER_PIXEL = 512.0 / RENDER_SIZE
PHYSICAL_DIFF_THRESHOLD_WORLD = 2.0 * WORLD_PER_RENDER_PIXEL
ANGLE_RADIUS_WORLD = 40.0
OBJECT_RGB = {"agent": np.asarray([65, 105, 225], dtype=np.int16), "block": np.asarray([119, 136, 153], dtype=np.int16)}
OBJECT_COLOR_CHANNEL_TOLERANCE = 24  # diagnostic segmentation only; never a pass/fail gate
STATE_SAMPLE_RAW_INDICES = np.asarray([4, 9, 14, 19, 24], dtype=np.int64)
PREFIX_FIELDS = (
    "history_pixels",
    "context_actions",
    "conditions",
    "condition_values",
    "physical_steps",
    "goal_pixels",
    "goal_state",
    "initial_state",
    "query_state",
)
LEAK_KEY = re.compile(r"^(?:hidden(?:_condition|_parameter(?:_value)?)?|true_condition(?:_value)?|actual_condition(?:_value)?|friction(?:_coefficient|_value)?|damping(?:_coefficient|_value)?|pymunk_damping|action_scale|condition_parameter)$", re.I)
ALLOWED_LABEL_KEYS = {"conditions", "condition_values", "candidate_factor", "candidate_parent_index"}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return sha256_bytes(array.tobytes())


def canonical_action_bytes(action: np.ndarray) -> bytes:
    """Canonical float32 bytes, with -0.0 normalized to +0.0."""
    value = np.asarray(action, dtype=np.float32).copy()
    value[value == 0.0] = np.float32(0.0)
    return np.ascontiguousarray(value).tobytes()


def scalar_text(value: Any) -> str:
    if isinstance(value, np.ndarray):
        if value.shape == ():
            value = value.item()
        elif value.size == 1:
            value = value.reshape(-1)[0].item()
    return str(value)


def _row_scene_id(row: dict[str, Any], fallback: str | None = None) -> str:
    for key in ("scene_id", "pair_id", "query_id", "id"):
        if key in row:
            return str(row[key])
    if fallback is not None:
        return fallback
    raise ValueError(f"scene row has no identifier: {sorted(row)}")


def _scene_records(manifest: dict[str, Any], directory: Path) -> list[dict[str, Any]]:
    """Normalize legacy and new scene manifests to ordered {id,path,index,row} records."""
    rows = manifest.get("scenes")
    if isinstance(rows, list):
        records = []
        for ordinal, raw in enumerate(rows):
            row = dict(raw)
            scene_id = _row_scene_id(row)
            relative = row.get("path") or row.get("file") or f"{scene_id}.npz"
            source_index = row.get("source_index")
            if source_index is None:
                indices = manifest.get("source_indices")
                source_index = indices[ordinal] if isinstance(indices, list) and ordinal < len(indices) else ordinal
            records.append({"scene_id": scene_id, "path": directory / str(relative), "source_index": int(source_index), "row": row})
        return records

    ids = manifest.get("scene_ids")
    if not isinstance(ids, list):
        raise ValueError("manifest must contain an ordered scenes or scene_ids list")
    indices = manifest.get("source_indices")
    files = sorted(directory.glob("*.npz"))
    by_pair_id: dict[str, Path] = {}
    for path in files:
        try:
            with np.load(path, allow_pickle=False) as archive:
                if "pair_id" in archive.files:
                    by_pair_id[scalar_text(archive["pair_id"])] = path
        except Exception:
            continue
    records = []
    for ordinal, scene_id_value in enumerate(ids):
        scene_id = str(scene_id_value)
        path = by_pair_id.get(scene_id, directory / f"{scene_id}.npz")
        source_index = indices[ordinal] if isinstance(indices, list) and ordinal < len(indices) else ordinal
        records.append({"scene_id": scene_id, "path": path, "source_index": int(source_index), "row": {}})
    return records


def _load_manifest(directory: Path) -> dict[str, Any]:
    path = directory / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _receipt_for(manifest: dict[str, Any], row: dict[str, Any], scene_id: str) -> dict[str, Any]:
    receipt = row.get("trajectory_receipt") or row.get("builder_receipt") or row.get("receipt")
    result = dict(receipt) if isinstance(receipt, dict) else {}
    # Builders may store full factor-search evidence beside the trajectory
    # receipt. Normalize the two equivalent locations without weakening any
    # required check on completeness or the fixed factor order.
    factor_receipt = row.get("factor_search_receipt")
    if isinstance(factor_receipt, dict):
        result.setdefault("factor_search_complete", factor_receipt.get("factor_search_complete"))
        result.setdefault("factor_search_factors", factor_receipt.get("factor_search_factors", factor_receipt.get("factors")))
    if isinstance(receipt, dict):
        return result
    receipts = manifest.get("scene_receipts")
    if isinstance(receipts, dict):
        value = receipts.get(scene_id)
        if isinstance(value, dict):
            result = dict(value.get("trajectory_receipt", value))
            factor_receipt = value.get("factor_search_receipt")
            if isinstance(factor_receipt, dict):
                result.setdefault("factor_search_complete", factor_receipt.get("factor_search_complete"))
                result.setdefault("factor_search_factors", factor_receipt.get("factor_search_factors", factor_receipt.get("factors")))
            return result
    if isinstance(receipts, list):
        for value in receipts:
            if isinstance(value, dict) and _row_scene_id(value, "") == scene_id:
                nested = value.get("trajectory_receipt") or value.get("builder_receipt")
                result = dict(nested if isinstance(nested, dict) else value)
                factor_receipt = value.get("factor_search_receipt")
                if isinstance(factor_receipt, dict):
                    result.setdefault("factor_search_complete", factor_receipt.get("factor_search_complete"))
                    result.setdefault("factor_search_factors", factor_receipt.get("factor_search_factors", factor_receipt.get("factors")))
                return result
    return {}


def _receipt_value(receipt: dict[str, Any], key: str, default: Any = None) -> Any:
    return receipt.get(key, default)


def _add_failure(failures: list[dict[str, str]], task: str, scene_id: str, check: str, detail: str) -> None:
    failures.append({"task": task, "scene_id": scene_id, "check": check, "detail": detail})


def _manifest_metadata_mismatches(
    old_records: list[dict[str, Any]], new_records: list[dict[str, Any]],
) -> dict[str, list[str]]:
    """Find source-group fields present in the frozen manifest but changed or dropped."""
    old_by_id = {item["scene_id"]: item for item in old_records}
    result: dict[str, list[str]] = {}
    for field in ("bootstrap_cluster", "source_group"):
        if not any(field in item["row"] for item in old_records):
            continue
        mismatches = []
        for new_item in new_records:
            old_item = old_by_id.get(new_item["scene_id"])
            if old_item is None or field not in old_item["row"]:
                continue
            if field not in new_item["row"] or new_item["row"].get(field) != old_item["row"].get(field):
                mismatches.append(new_item["scene_id"])
        result[field] = mismatches
    return result


def _load_npz_fields(path: Path, keys: tuple[str, ...] | list[str] | None = None) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        selected = archive.files if keys is None else [key for key in keys if key in archive.files]
        return {key: np.array(archive[key]) for key in selected}


def _infer_npz_id(path: Path) -> str:
    with np.load(path, allow_pickle=False) as archive:
        if "pair_id" in archive.files:
            return scalar_text(archive["pair_id"])
    return path.stem


def _check_coverage(
    task: str,
    old_manifest: dict[str, Any],
    new_manifest: dict[str, Any],
    old_records: list[dict[str, Any]],
    new_records: list[dict[str, Any]],
    failures: list[dict[str, str]],
    expected_sources: int = EXPECTED_SOURCES,
) -> dict[str, Any]:
    old_ids = [item["scene_id"] for item in old_records]
    new_ids = [item["scene_id"] for item in new_records]
    new_indices = [item["source_index"] for item in new_records]
    selected_legacy = [item for item in old_records if item["scene_id"] in set(new_ids)]
    checks = {
        "legacy_scene_count_256": len(old_records) == EXPECTED_SOURCES,
        "new_scene_count_expected": len(new_records) == expected_sources,
        "legacy_ids_unique": len(set(old_ids)) == len(old_ids),
        "new_ids_unique": len(set(new_ids)) == len(new_ids),
        "source_ids_order_matches_legacy": new_ids == [item["scene_id"] for item in selected_legacy],
        "source_indices_order_matches_legacy": new_indices == [item["source_index"] for item in selected_legacy] and len(set(new_indices)) == len(new_indices),
        "all_legacy_npz_present": all(item["path"].is_file() for item in old_records),
        "all_new_npz_present": all(item["path"].is_file() for item in new_records),
        "new_npz_file_count_expected": len(list(new_records[0]["path"].parent.glob("*.npz"))) == expected_sources if new_records else False,
    }
    metadata_mismatches = _manifest_metadata_mismatches(old_records, new_records)
    metadata_counts = {}
    for field, scene_ids in metadata_mismatches.items():
        old_total = sum(field in item["row"] for item in old_records)
        old_by_id = {item["scene_id"]: item for item in old_records}
        compared_total = sum(field in old_by_id[item["scene_id"]]["row"] for item in new_records if item["scene_id"] in old_by_id)
        mismatch_count = len(scene_ids)
        checks[f"legacy_{field}_preserved"] = mismatch_count == 0
        metadata_counts[field] = {"legacy_rows_with_value": old_total, "new_rows_compared": compared_total, "matched_rows": compared_total - mismatch_count, "mismatched_scene_count": mismatch_count, "mismatch_examples": scene_ids[:12]}
        if mismatch_count:
            _add_failure(failures, task, "<panel>", f"legacy_{field}_preserved", f"mismatched or missing new values in {scene_ids[:12]} (matched {compared_total - mismatch_count}/{compared_total} compared rows)")
    for name, ok in checks.items():
        if not ok:
            if name not in {f"legacy_{field}_preserved" for field in metadata_mismatches}:
                _add_failure(failures, task, "<panel>", name, "coverage/source order check failed")
    for label, manifest, expected in (("legacy", old_manifest, EXPECTED_SOURCES), ("new", new_manifest, expected_sources)):
        coverage = manifest.get("source_coverage")
        if isinstance(coverage, dict):
            produced = coverage.get("produced")
            requested = coverage.get("requested", coverage.get("requested_source_count"))
            if produced is not None and int(produced) != expected:
                _add_failure(failures, task, "<panel>", f"{label}_manifest_source_coverage", f"produced={produced}")
            if requested is not None and int(requested) != expected:
                _add_failure(failures, task, "<panel>", f"{label}_manifest_source_coverage", f"requested={requested}")
    if new_manifest.get("attempted_failures_retained") is False:
        _add_failure(failures, task, "<panel>", "attempted_failures_retained", "manifest explicitly says failed attempts were not retained")
    return {"checks": checks, "manifest_metadata_preservation": metadata_counts, "legacy_count": len(old_records), "new_count": len(new_records), "expected_new_count": expected_sources, "ordered_ids_match": checks["source_ids_order_matches_legacy"], "source_indices": new_indices}


def _same_array(left: np.ndarray, right: np.ndarray) -> bool:
    return left.shape == right.shape and np.array_equal(left, right)


def _check_prefix(
    task: str,
    scene_id: str,
    old: dict[str, np.ndarray],
    new: dict[str, np.ndarray],
    failures: list[dict[str, str]],
) -> dict[str, Any]:
    checked: dict[str, bool] = {}
    required = ("history_pixels", "context_actions", "conditions", "physical_steps")
    if "condition_values" in old:
        required += ("condition_values",)
    for key in required:
        ok = key in old and key in new and _same_array(old[key], new[key])
        checked[key] = ok
        if not ok:
            _add_failure(failures, task, scene_id, f"frozen_prefix_{key}", "missing or differs byte/value-wise from legacy panel")
    for key in PREFIX_FIELDS:
        if key in old and key not in required:
            if key in new:
                ok = _same_array(old[key], new[key])
                checked[key] = ok
                if not ok:
                    _add_failure(failures, task, scene_id, f"frozen_prefix_{key}", "differs from legacy panel")
            else:
                checked[key] = False
                _add_failure(failures, task, scene_id, f"frozen_prefix_{key}", "legacy prefix field missing from new panel")
    return checked


def _find_candidate_actions(actions: np.ndarray) -> tuple[np.ndarray, bool]:
    value = np.asarray(actions)
    if value.ndim == 4:
        return value, True
    if value.ndim == 5 and value.shape[0] == 2:
        shared = np.array_equal(value[0], value[1])
        return value[0], shared
    raise ValueError(f"candidate_actions must be [C,5,5,2] or [2,C,5,5,2], got {value.shape}")


def _find_slot_actions(actions: np.ndarray) -> tuple[np.ndarray, bool]:
    value = np.asarray(actions)
    if value.ndim == 3 and value.shape[1:] == (25, 2):
        return value.reshape(len(value), 5, 5, 2), True
    return _find_candidate_actions(value)


def _candidate_fingerprints(actions: np.ndarray) -> tuple[list[str], list[int], dict[str, list[int]]]:
    fingerprints: list[str] = []
    unique_slots: list[int] = []
    groups: dict[str, list[int]] = {}
    for index, action in enumerate(np.asarray(actions, dtype=np.float32)):
        raw = canonical_action_bytes(action)
        key = sha256_bytes(raw)
        if key not in groups:
            groups[key] = []
            unique_slots.append(index)
        groups[key].append(index)
        fingerprints.append(key)
    return fingerprints, unique_slots, groups


def _select_maximum_safe_factor(safe: np.ndarray) -> tuple[float, bool]:
    value = np.asarray(safe, dtype=bool)
    if value.shape != CANDIDATE_FACTORS.shape:
        raise ValueError(f"factor safety row must have shape {CANDIDATE_FACTORS.shape}, got {value.shape}")
    if not value.any():
        return 0.0, False
    return float(np.max(CANDIDATE_FACTORS[value])), True


def _check_bank(
    task: str,
    scene_id: str,
    old: dict[str, np.ndarray],
    new: dict[str, np.ndarray],
    receipt: dict[str, Any],
    failures: list[dict[str, str]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    try:
        old_actions, old_shared = _find_candidate_actions(old["candidate_actions"])
        unique_actions, new_shared = _find_candidate_actions(new["candidate_actions"])
    except (KeyError, ValueError) as exc:
        _add_failure(failures, task, scene_id, "candidate_action_shape", str(exc))
        return {"error": str(exc)}
    unique_count = len(unique_actions)
    slot_actions_raw = new.get("candidate_slot_actions")
    if slot_actions_raw is None:
        _add_failure(failures, task, scene_id, "candidate_slot_actions", "missing 11-slot scaled action mapping")
        slot_actions = np.zeros((0, 5, 5, 2), dtype=np.float32)
    else:
        try:
            slot_actions, slot_shared = _find_slot_actions(slot_actions_raw)
            if not slot_shared:
                _add_failure(failures, task, scene_id, "candidate_slot_actions_shared_across_conditions", "slot actions differ across conditions")
        except ValueError as exc:
            _add_failure(failures, task, scene_id, "candidate_slot_actions_shape", str(exc))
            slot_actions = np.zeros((0, 5, 5, 2), dtype=np.float32)
    slot_count = len(slot_actions)
    result["candidate_count"] = unique_count
    result["legacy_slot_count"] = slot_count
    result["legacy_candidate_count"] = len(old_actions)
    result["actions_shared_across_conditions"] = bool(new_shared)
    if not old_shared or not new_shared:
        _add_failure(failures, task, scene_id, "candidate_actions_shared_across_conditions", "condition-specific candidate action arrays differ")
    if unique_actions.shape[1:] != (5, 5, 2):
        _add_failure(failures, task, scene_id, "candidate_action_shape", f"expected unique [K,5,5,2], got {unique_actions.shape}")
    if not np.isfinite(unique_actions).all() or (np.abs(unique_actions) > 1.0 + 1e-7).any():
        _add_failure(failures, task, scene_id, "candidate_actions_legal", "actions are nonfinite or outside [-1,1]")
    if slot_actions.shape[1:] != (5, 5, 2):
        _add_failure(failures, task, scene_id, "candidate_slot_action_shape", f"expected [11,5,5,2], got {slot_actions.shape}")
    if not np.isfinite(slot_actions).all() or (np.abs(slot_actions) > 1.0 + 1e-7).any():
        _add_failure(failures, task, scene_id, "candidate_slot_actions_legal", "slot actions are nonfinite or outside [-1,1]")
    parents = new.get("candidate_parent_index")
    factors = new.get("candidate_factor")
    if parents is None or parents.shape != (slot_count,):
        _add_failure(failures, task, scene_id, "candidate_parent_mapping", "candidate_parent_index must have shape [11]")
        parents = np.full(slot_count, -1, dtype=np.int64)
    else:
        parents = np.asarray(parents, dtype=np.int64)
    if factors is None or factors.shape != (slot_count,):
        _add_failure(failures, task, scene_id, "candidate_factor_mapping", "candidate_factor must have shape [C]")
        factors = np.full(slot_count, np.nan, dtype=np.float32)
    else:
        factors = np.asarray(factors, dtype=np.float32)
    if slot_count != len(old_actions):
        _add_failure(failures, task, scene_id, "candidate_slot_count", f"expected one slot per legacy candidate ({len(old_actions)}), got {slot_count}")
    expected_parents = set(range(len(old_actions)))
    seen_parents = set(int(value) for value in parents if 0 <= value < len(old_actions))
    if seen_parents != expected_parents or len(set(int(value) for value in parents)) != len(parents):
        _add_failure(failures, task, scene_id, "candidate_parent_coverage", f"mapped old slots={sorted(seen_parents)} expected={sorted(expected_parents)}")
    if ((parents < 0) | (parents >= len(old_actions))).any():
        _add_failure(failures, task, scene_id, "candidate_parent_mapping", "parent index outside legacy bank")
    slot_to_unique = new.get("candidate_slot_to_unique")
    if slot_to_unique is None or np.asarray(slot_to_unique).shape != (slot_count,):
        _add_failure(failures, task, scene_id, "candidate_slot_to_unique", "candidate_slot_to_unique must have shape [11]")
        slot_to_unique = np.full(slot_count, -1, dtype=np.int64)
    else:
        slot_to_unique = np.asarray(slot_to_unique, dtype=np.int64)
    if ((slot_to_unique < 0) | (slot_to_unique >= unique_count)).any():
        _add_failure(failures, task, scene_id, "candidate_slot_to_unique", "slot maps outside unique candidate axis")
    elif set(int(value) for value in slot_to_unique) != set(range(unique_count)):
        _add_failure(failures, task, scene_id, "candidate_slot_to_unique", "unique candidate axis contains an unreferenced action")
    fingerprints, unique_slots, duplicate_groups = _candidate_fingerprints(unique_actions)
    if len(unique_slots) != unique_count:
        _add_failure(failures, task, scene_id, "candidate_unique_axis", "candidate_actions contains duplicate float32 action bytes")
    slot_fingerprints, _, _ = _candidate_fingerprints(slot_actions)
    for slot, (parent, factor) in enumerate(zip(parents, factors)):
        if parent < 0 or parent >= len(old_actions):
            continue
        if not np.any(CANDIDATE_FACTORS == factor):
            _add_failure(failures, task, scene_id, "candidate_factor_legal", f"slot {slot}: factor {factor} is outside the frozen ladder")
            continue
        expected = np.asarray(old_actions[parent], dtype=np.float32) * np.float32(factor)
        if not np.array_equal(np.asarray(slot_actions[slot], dtype=np.float32), expected):
            _add_failure(failures, task, scene_id, "candidate_action_mapping", f"slot {slot} is not legacy parent {parent} multiplied by factor {factor}")
        unique_index = int(slot_to_unique[slot]) if slot < len(slot_to_unique) else -1
        if 0 <= unique_index < unique_count and not np.array_equal(slot_actions[slot], unique_actions[unique_index]):
            _add_failure(failures, task, scene_id, "candidate_slot_action_mapping", f"slot {slot} does not match unique action {unique_index}")
    if "native_candidate_index" in old and "native_candidate_index" in new:
        native = int(np.asarray(old["native_candidate_index"]).item())
        refreshed_native = int(np.asarray(new["native_candidate_index"]).item())
        mapped_native = int(slot_to_unique[native]) if 0 <= native < len(slot_to_unique) else -1
        if refreshed_native not in (native, mapped_native):
            _add_failure(failures, task, scene_id, "native_candidate_mapping", f"new native index {refreshed_native} is neither old slot {native} nor mapped unique index {mapped_native}")
    result.update({
        "unique_action_count": unique_count,
        "duplicate_action_groups": [slots for slots in duplicate_groups.values() if len(slots) > 1],
        "unique_action_sha256": fingerprints,
        "candidate_slot_sha256": slot_fingerprints,
        "candidate_slot_to_unique": slot_to_unique.astype(int).tolist(),
        "factor_choices": [{"parent": int(parent), "factor": float(factor)} for parent, factor in zip(parents, factors)],
        "equal_weighting_rule": "one representative per distinct float32 candidate_actions bytes within each source scene",
    })
    factor_search = new.get("factor_search_safe")
    if factor_search is None:
        candidate_receipt = receipt.get("factor_search_safe")
        if candidate_receipt is not None:
            factor_search = np.asarray(candidate_receipt)
    if factor_search is None:
        _add_failure(failures, task, scene_id, "factor_search_receipt", "missing factor_search_safe [legacy C,7]")
        factor_search = np.zeros((len(old_actions), len(CANDIDATE_FACTORS)), dtype=bool)
    factor_search = np.asarray(factor_search, dtype=bool)
    if factor_search.shape != (len(old_actions), len(CANDIDATE_FACTORS)):
        _add_failure(failures, task, scene_id, "factor_search_receipt", f"expected {(len(old_actions), len(CANDIDATE_FACTORS))}, got {factor_search.shape}")
    else:
        result["factor_search_safe_counts"] = factor_search.sum(axis=1).astype(int).tolist()
        for slot, (parent, factor) in enumerate(zip(parents, factors)):
            if not 0 <= parent < len(old_actions):
                continue
            safe = factor_search[parent]
            selected, has_safe = _select_maximum_safe_factor(safe)
            if not has_safe:
                _add_failure(failures, task, scene_id, "no_safe_candidate_factor", f"old candidate {parent} has no factor safe in both conditions for all 25 raw steps; source retained")
            elif float(factor) != selected:
                _add_failure(failures, task, scene_id, "maximum_safe_factor", f"old candidate {parent}: selected {factor}, maximum safe factor is {selected}")
    candidate_safe = new.get("candidate_safe")
    if candidate_safe is None:
        candidate_safe = receipt.get("candidate_safe")
    if candidate_safe is None or np.asarray(candidate_safe).shape != (slot_count,):
        _add_failure(failures, task, scene_id, "candidate_safety_receipt", "candidate_safe must have shape [11 legacy slots]")
    else:
        result["unsafe_candidate_count"] = int((~np.asarray(candidate_safe, dtype=bool)).sum())
        for slot, (parent, safe) in enumerate(zip(parents, np.asarray(candidate_safe, dtype=bool))):
            if 0 <= parent < len(old_actions) and factor_search.shape == (len(old_actions), len(CANDIDATE_FACTORS)):
                expected_safe = bool(factor_search[parent].any())
                if bool(safe) != expected_safe:
                    _add_failure(failures, task, scene_id, "candidate_safety_receipt", f"slot {slot}: candidate_safe does not match factor search")
                if not safe:
                    _add_failure(failures, task, scene_id, "no_safe_candidate_factor", f"slot {slot}: unsafe factor-0 fallback retained in output")
    return result


def _geometry_safety(bounds: np.ndarray) -> tuple[bool, np.ndarray, dict[str, Any]]:
    value = np.asarray(bounds, dtype=np.float64)
    if value.ndim != 5 or value.shape[0] != 2 or value.shape[2:] != (25, 2, 4):
        return False, np.zeros(0, dtype=bool), {"shape": list(value.shape), "expected": "[2,C,25,2,4]"}
    finite = np.isfinite(value).all()
    ordered = bool((value[..., 0] <= value[..., 2]).all() and (value[..., 1] <= value[..., 3]).all())
    inside = bool(((value[..., 0] >= CANVAS_MIN) & (value[..., 1] >= CANVAS_MIN) & (value[..., 2] <= CANVAS_MAX) & (value[..., 3] <= CANVAS_MAX)).all()) if finite else False
    candidate_safe = np.isfinite(value).all(axis=(0, 2, 3, 4))
    candidate_safe &= ((value[..., 0] >= CANVAS_MIN) & (value[..., 1] >= CANVAS_MIN) & (value[..., 2] <= CANVAS_MAX) & (value[..., 3] <= CANVAS_MAX)).all(axis=(0, 2, 3))
    return bool(finite and ordered and inside), candidate_safe, {"finite": bool(finite), "ordered": ordered, "all_full_shapes_inside_2_510": inside, "unsafe_candidate_count": int((~candidate_safe).sum()), "bounds_shape": list(value.shape), "coordinate_space": "512x512 PushT canvas pixels", "object_axis": ["agent", "block"], "bound_order": ["xmin", "ymin", "xmax", "ymax"]}


def _state_objects(task: str, state: np.ndarray) -> dict[str, tuple[np.ndarray, float | None]]:
    value = np.asarray(state, dtype=np.float64)
    if task == "action_strength":
        if value.shape != (7,):
            raise ValueError(f"{task} state must be 7-D, got {value.shape}")
        return {"agent": (value[0:2], None), "block": (value[2:4], float(value[4]))}
    if value.shape != (12,):
        raise ValueError(f"{task} state must be 12-D, got {value.shape}")
    return {"agent": (value[0:2], None), "block": (value[6:8], float(value[10]))}


def _state_gaps(task: str, left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    a = _state_objects(task, left)
    b = _state_objects(task, right)
    agent_gap = float(np.linalg.norm(a["agent"][0] - b["agent"][0]))
    block_xy = float(np.linalg.norm(a["block"][0] - b["block"][0]))
    theta_a = float(a["block"][1])
    theta_b = float(b["block"][1])
    angle_equivalent = float(ANGLE_RADIUS_WORLD * np.linalg.norm(np.asarray([np.cos(theta_a), np.sin(theta_a)]) - np.asarray([np.cos(theta_b), np.sin(theta_b)])))
    return {"agent_world_gap": agent_gap, "block_xy_world_gap": block_xy, "block_angle_equivalent_world_gap": angle_equivalent, "block_combined_world_gap": float(np.hypot(block_xy, angle_equivalent))}


def _color_mask(frame: np.ndarray, color: np.ndarray) -> np.ndarray:
    rgb = np.asarray(frame, dtype=np.int16)
    return np.max(np.abs(rgb - color.reshape(1, 1, 3)), axis=-1) <= OBJECT_COLOR_CHANNEL_TOLERANCE


def _mask_diagnostics(left: np.ndarray, right: np.ndarray, color: np.ndarray) -> dict[str, Any]:
    mask_a = _color_mask(left, color)
    mask_b = _color_mask(right, color)
    count_a = int(mask_a.sum())
    count_b = int(mask_b.sum())
    union = int(np.logical_or(mask_a, mask_b).sum())
    intersection = int(np.logical_and(mask_a, mask_b).sum())
    def centroid(mask: np.ndarray) -> list[float] | None:
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            return None
        return [float(xs.mean()), float(ys.mean())]
    ca, cb = centroid(mask_a), centroid(mask_b)
    centroid_gap = None if ca is None or cb is None else float(np.linalg.norm(np.asarray(ca) - np.asarray(cb)))
    return {"left_pixels": count_a, "right_pixels": count_b, "mask_iou": float(intersection / union) if union else 1.0, "mask_exact_equal": bool(np.array_equal(mask_a, mask_b)), "centroid_gap_render_px": centroid_gap}


def _unique_slots(actions: np.ndarray) -> list[int]:
    return _candidate_fingerprints(actions)[1]


def _signal_scene(task: str, pixels: np.ndarray, states: np.ndarray, actions: np.ndarray) -> dict[str, Any]:
    """Per-scene metrics; duplicates in candidate_actions have equal single weight."""
    representatives = _unique_slots(actions)
    fingerprints = _candidate_fingerprints(actions)[0]
    horizons: list[list[dict[str, Any]]] = [[] for _ in range(5)]
    for candidate in representatives:
        for horizon, raw_index in enumerate(STATE_SAMPLE_RAW_INDICES):
            image_a = np.asarray(pixels[0, candidate, horizon])
            image_b = np.asarray(pixels[1, candidate, horizon])
            delta = np.abs(image_a.astype(np.int16) - image_b.astype(np.int16))
            state_gap = _state_gaps(task, states[0, candidate, horizon], states[1, candidate, horizon])
            horizons[horizon].append({
                "candidate_action_sha256": fingerprints[candidate],
                "pixel_exact_equal": bool(np.array_equal(image_a, image_b)),
                "pixel_changed_fraction": float(np.any(delta != 0, axis=-1).mean()),
                "pixel_mean_abs_rgb_delta": float(delta.mean()),
                "agent": {"world_gap": state_gap["agent_world_gap"], "visible_over_2_render_px": state_gap["agent_world_gap"] > PHYSICAL_DIFF_THRESHOLD_WORLD, "mask": _mask_diagnostics(image_a, image_b, OBJECT_RGB["agent"])},
                "block": {"world_gap": state_gap["block_combined_world_gap"], "xy_gap": state_gap["block_xy_world_gap"], "angle_equivalent_gap": state_gap["block_angle_equivalent_world_gap"], "visible_over_2_render_px": state_gap["block_combined_world_gap"] > PHYSICAL_DIFF_THRESHOLD_WORLD, "mask": _mask_diagnostics(image_a, image_b, OBJECT_RGB["block"])},
            })
    summary: dict[str, Any] = {"unique_candidate_action_count": len(representatives), "raw_candidate_slot_count": len(actions), "per_horizon": []}
    for horizon, rows in enumerate(horizons):
        objects = {}
        for obj in ("agent", "block"):
            gaps = np.asarray([row[obj]["world_gap"] for row in rows], dtype=np.float64)
            visible = np.asarray([row[obj]["visible_over_2_render_px"] for row in rows], dtype=bool)
            mask_iou = np.asarray([row[obj]["mask"]["mask_iou"] for row in rows], dtype=np.float64)
            mask_equal = np.asarray([row[obj]["mask"]["mask_exact_equal"] for row in rows], dtype=bool)
            centroid_gaps = [row[obj]["mask"]["centroid_gap_render_px"] for row in rows if row[obj]["mask"]["centroid_gap_render_px"] is not None]
            objects[obj] = {
                "mean_world_gap": float(gaps.mean()) if len(gaps) else None,
                "zero_within_2_render_px_ratio": float((~visible).mean()) if len(visible) else None,
                "over_2_render_px_ratio": float(visible.mean()) if len(visible) else None,
                "object_color_mask_exact_equal_ratio_diagnostic": float(mask_equal.mean()) if len(mask_equal) else None,
                "mean_object_color_mask_iou_diagnostic": float(mask_iou.mean()) if len(mask_iou) else None,
                "mean_object_color_mask_centroid_gap_render_px_diagnostic": float(np.mean(centroid_gaps)) if centroid_gaps else None,
            }
        changed = np.asarray([row["pixel_changed_fraction"] for row in rows], dtype=np.float64)
        exact = np.asarray([row["pixel_exact_equal"] for row in rows], dtype=bool)
        summary["per_horizon"].append({
            "physical_step": int(HORIZON_STEPS[horizon]),
            "unique_candidate_observations": len(rows),
            "exact_same_full_frame_ratio": float(exact.mean()) if len(exact) else None,
            "mean_changed_full_frame_pixel_fraction": float(changed.mean()) if len(changed) else None,
            "mean_abs_rgb_delta_levels": float(np.mean([row["pixel_mean_abs_rgb_delta"] for row in rows])) if rows else None,
            "objects": objects,
        })
    return summary


def _merge_macro_summaries(scene_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """Average each scene's C×horizon means equally, then report horizon/object diagnostics."""
    if not scene_summaries:
        return {"scene_count": 0, "per_horizon": []}
    merged = []
    for h in range(5):
        row = {"physical_step": int(HORIZON_STEPS[h]), "objects": {}}
        exact = [s["per_horizon"][h]["exact_same_full_frame_ratio"] for s in scene_summaries if s["per_horizon"][h]["exact_same_full_frame_ratio"] is not None]
        changed = [s["per_horizon"][h]["mean_changed_full_frame_pixel_fraction"] for s in scene_summaries if s["per_horizon"][h]["mean_changed_full_frame_pixel_fraction"] is not None]
        row["scene_equal_mean_exact_same_full_frame_ratio"] = float(np.mean(exact)) if exact else None
        row["scene_equal_mean_changed_full_frame_pixel_fraction"] = float(np.mean(changed)) if changed else None
        for obj in ("agent", "block"):
            row["objects"][obj] = {}
            for key in ("mean_world_gap", "zero_within_2_render_px_ratio", "over_2_render_px_ratio", "object_color_mask_exact_equal_ratio_diagnostic", "mean_object_color_mask_iou_diagnostic", "mean_object_color_mask_centroid_gap_render_px_diagnostic"):
                vals = [s["per_horizon"][h]["objects"][obj][key] for s in scene_summaries if s["per_horizon"][h]["objects"][obj][key] is not None]
                row["objects"][obj][key] = float(np.mean(vals)) if vals else None
        merged.append(row)
    return {"scene_count": len(scene_summaries), "candidate_weighting": "within each source scene, unique float32 action bytes; scenes then equally weighted", "per_horizon": merged}


def _find_exact_pixel_aliases(
    pixels: np.ndarray,
    states: np.ndarray,
    actions: np.ndarray,
    task: str,
    max_examples: int = 24,
) -> dict[str, Any]:
    """Report only byte-identical decoded frames with >2-render-pixel state gaps."""
    fingerprints, representatives, _ = _candidate_fingerprints(actions)
    seen: dict[str, list[tuple[int, int, int]]] = {}
    exact_frame_collisions = 0
    examples: list[dict[str, Any]] = []
    counts = {"same_candidate_across_horizons": 0, "same_horizon_across_conditions": 0, "other": 0}
    for condition in range(2):
        for candidate in representatives:
            for horizon in range(5):
                frame = np.ascontiguousarray(pixels[condition, candidate, horizon])
                key = sha256_bytes(frame.tobytes())
                location = (condition, candidate, horizon)
                prior = seen.setdefault(key, [])
                for other in prior:
                    oc, ok, oh = other
                    if not np.array_equal(pixels[oc, ok, oh], frame):
                        continue
                    exact_frame_collisions += 1
                    raw_a = int(STATE_SAMPLE_RAW_INDICES[oh])
                    raw_b = int(STATE_SAMPLE_RAW_INDICES[horizon])
                    gaps = _state_gaps(task, states[oc, ok, raw_a], states[condition, candidate, raw_b])
                    qualifying = []
                    if gaps["agent_world_gap"] > PHYSICAL_DIFF_THRESHOLD_WORLD:
                        qualifying.append("agent")
                    if gaps["block_combined_world_gap"] > PHYSICAL_DIFF_THRESHOLD_WORLD:
                        qualifying.append("block")
                    if not qualifying:
                        continue
                    if ok == candidate and fingerprints[ok] == fingerprints[candidate] and oc == condition and oh != horizon:
                        category = "same_candidate_across_horizons"
                    elif ok == candidate and oc != condition and oh == horizon:
                        category = "same_horizon_across_conditions"
                    else:
                        category = "other"
                    counts[category] += 1
                    if len(examples) < max_examples:
                        examples.append({
                            "category": category,
                            "condition_a": oc,
                            "candidate_slot_a": ok,
                            "candidate_action_sha256_a": fingerprints[ok],
                            "physical_step_a": int(HORIZON_STEPS[oh]),
                            "condition_b": condition,
                            "candidate_slot_b": candidate,
                            "candidate_action_sha256_b": fingerprints[candidate],
                            "physical_step_b": int(HORIZON_STEPS[horizon]),
                            "objects_over_2_render_px": qualifying,
                            "state_gaps_world": gaps,
                            "pixel_sha256": key,
                        })
                prior.append(location)
    total = int(sum(counts.values()))
    return {
        "threshold": {"full_frame_pixel_equality": "exact uint8 array equality; SHA-256 collision is confirmed with np.array_equal", "physical_state_difference_world_pixels": PHYSICAL_DIFF_THRESHOLD_WORLD, "render_pixels": 2.0, "angle_equivalent_radius_world_pixels": ANGLE_RADIUS_WORLD},
        "exact_same_frame_pairs_examined": exact_frame_collisions,
        "qualifying_alias_pair_counts": counts,
        "qualifying_alias_pair_count": total,
        "examples_capped_at": max_examples,
        "examples": examples,
        "interpretation": "A found pair is a local exact-pixel alias with materially different object state. Zero found pairs do not establish sufficient observability.",
    }


def _receipt_checks(task: str, scene_id: str, receipt: dict[str, Any], failures: list[dict[str, str]]) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    expected = {
        "replayed_original_history": True,
        "query_state_injected": False,
        "all_candidates_continuous": True,
        "future_raw_steps": RAW_FUTURE_STEPS,
    }
    for key, expected_value in expected.items():
        actual = receipt.get(key)
        ok = actual == expected_value
        checks[key] = bool(ok)
        if not ok:
            _add_failure(failures, task, scene_id, f"trajectory_receipt_{key}", f"expected {expected_value!r}, got {actual!r}")
    pixels = receipt.get("prefix_pixels_exact")
    ok_pixels = isinstance(pixels, (list, tuple)) and len(pixels) == 2 and all(bool(value) for value in pixels)
    checks["prefix_pixels_exact"] = bool(ok_pixels)
    if not ok_pixels:
        _add_failure(failures, task, scene_id, "trajectory_receipt_prefix_pixels_exact", "expected [true,true]")
    gaps = receipt.get("prefix_state_gap")
    ok_gaps = isinstance(gaps, (list, tuple)) and len(gaps) == 2 and np.isfinite(np.asarray(gaps, dtype=np.float64)).all()
    checks["prefix_state_gap_finite"] = bool(ok_gaps)
    if not ok_gaps:
        _add_failure(failures, task, scene_id, "trajectory_receipt_prefix_state_gap", "expected two finite state gaps")
    factor_complete = receipt.get("factor_search_complete")
    checks["factor_search_complete"] = factor_complete is True
    if factor_complete is not True:
        _add_failure(failures, task, scene_id, "factor_search_complete", "builder did not receipt a complete seven-factor search")
    factor_values = receipt.get("factor_search_factors")
    expected_factors = CANDIDATE_FACTORS.astype(np.float32)
    factors_ok = factor_values is not None and np.array_equal(np.asarray(factor_values, dtype=np.float32), expected_factors)
    checks["factor_search_factors"] = bool(factors_ok)
    if not factors_ok:
        _add_failure(failures, task, scene_id, "factor_search_factors", f"expected ladder in order {expected_factors.astype(float).tolist()}")
    return checks


def _check_state_and_geometry(
    task: str,
    scene_id: str,
    new: dict[str, np.ndarray],
    failures: list[dict[str, str]],
) -> tuple[dict[str, Any], np.ndarray | None]:
    result: dict[str, Any] = {}
    step_states = new.get("future_step_states")
    future_states = new.get("future_states")
    if step_states is None or step_states.ndim != 4 or step_states.shape[0] != 2 or step_states.shape[2] != RAW_FUTURE_STEPS:
        _add_failure(failures, task, scene_id, "future_step_states_shape", "expected [2,C,25,D]")
        return result, None
    if future_states is None or future_states.ndim != 4 or future_states.shape[:2] != step_states.shape[:2] or future_states.shape[2] != 5 or future_states.shape[3] != step_states.shape[3]:
        _add_failure(failures, task, scene_id, "future_states_shape", "expected [2,C,5,D] matching future_step_states")
    else:
        sampled = np.take(step_states, STATE_SAMPLE_RAW_INDICES, axis=2)
        ok = np.allclose(sampled, future_states, rtol=0.0, atol=1e-5, equal_nan=False)
        result["sampled_state_matches_raw_steps"] = bool(ok)
        if not ok:
            _add_failure(failures, task, scene_id, "sampled_state_matches_raw_steps", "future_states do not match raw state indices 4,9,14,19,24 within 1e-5")
    if not np.isfinite(step_states).all():
        _add_failure(failures, task, scene_id, "future_step_states_finite", "nonfinite state values")
    if step_states.shape[-1] != (7 if task == "action_strength" else 12):
        _add_failure(failures, task, scene_id, "future_step_state_dimension", f"unexpected dimension {step_states.shape[-1]}")
    bounds = new.get("future_geometry_bounds")
    if bounds is None:
        _add_failure(failures, task, scene_id, "future_geometry_bounds", "missing per-raw-step full-shape bounds")
        return result, step_states
    geometry_ok, geometry_candidate_safe, geometry = _geometry_safety(bounds)
    result["geometry"] = geometry
    result["all_geometry_safe"] = geometry_ok
    if bounds.shape[1] != step_states.shape[1]:
        _add_failure(failures, task, scene_id, "future_geometry_bounds_shape", "candidate axis differs from state/action candidate axis")
    if not geometry_ok:
        _add_failure(failures, task, scene_id, "full_shape_canvas_containment", "at least one full body shape extends outside [2,510] or bounds are invalid; samples remain included")
    candidate_safe = new.get("candidate_safe")
    slot_to_unique = np.asarray(new.get("candidate_slot_to_unique", []), dtype=np.int64)
    if candidate_safe is not None and np.asarray(candidate_safe).shape == slot_to_unique.shape:
        if ((slot_to_unique < 0) | (slot_to_unique >= len(geometry_candidate_safe))).any():
            _add_failure(failures, task, scene_id, "candidate_safe_matches_full_geometry", "candidate_slot_to_unique is invalid")
        else:
            slot_geometry_safe = geometry_candidate_safe[slot_to_unique]
            mismatch = np.flatnonzero(np.asarray(candidate_safe, dtype=bool) != slot_geometry_safe)
            if len(mismatch):
                _add_failure(failures, task, scene_id, "candidate_safe_matches_full_geometry", f"candidate slots disagree with full saved bounds: {mismatch.tolist()}")
    elif candidate_safe is not None:
        _add_failure(failures, task, scene_id, "candidate_safe_matches_full_geometry", "candidate_safe or candidate_slot_to_unique has wrong shape")
    result["geometry_safe_by_candidate"] = geometry_candidate_safe.astype(bool).tolist()
    return result, step_states


def _check_leakage(task: str, scene_id: str, new_keys: set[str], old_keys: set[str], failures: list[dict[str, str]]) -> list[str]:
    leaked = sorted(key for key in new_keys if key not in old_keys and key not in ALLOWED_LABEL_KEYS and LEAK_KEY.search(key))
    if leaked:
        _add_failure(failures, task, scene_id, "non_hidden_parameter_leakage", f"new NPZ keys disclose parameter-like values: {leaked}")
    return leaked


def _scene_paths_are_unique(records: list[dict[str, Any]]) -> bool:
    return len({str(item["path"].resolve()) for item in records}) == len(records)


def validate_task(task: str, panel_root: Path, legacy_root: Path, expected_sources: int = EXPECTED_SOURCES) -> dict[str, Any]:
    failures: list[dict[str, str]] = []
    old_dir = legacy_root / task
    new_dir = panel_root / task
    if not (new_dir / "manifest.json").is_file() and (panel_root / "manifest.json").is_file():
        new_dir = panel_root
    try:
        old_manifest = _load_manifest(old_dir)
        new_manifest = _load_manifest(new_dir)
        old_records = _scene_records(old_manifest, old_dir)
        new_records = _scene_records(new_manifest, new_dir)
    except Exception as exc:
        return {"task": task, "passed": False, "failures": [{"task": task, "scene_id": "<panel>", "check": "load_manifests", "detail": str(exc)}]}
    coverage = _check_coverage(task, old_manifest, new_manifest, old_records, new_records, failures, expected_sources)
    if not _scene_paths_are_unique(old_records) or not _scene_paths_are_unique(new_records):
        _add_failure(failures, task, "<panel>", "scene_path_uniqueness", "manifest maps multiple source scenes to one file")
    old_by_id = {item["scene_id"]: item for item in old_records}
    records = [(old_by_id[item["scene_id"]], item) for item in new_records if item["scene_id"] in old_by_id]
    if len(records) != len(new_records):
        _add_failure(failures, task, "<panel>", "new_source_id_in_legacy", "one or more new source IDs are absent from the frozen legacy panel")
    legacy_scene_signals: list[dict[str, Any]] = []
    new_scene_signals: list[dict[str, Any]] = []
    alias_count = 0
    alias_examples: list[dict[str, Any]] = []
    alias_categories = {"same_candidate_across_horizons": 0, "same_horizon_across_conditions": 0, "other": 0}
    geometry_unsafe_slots = 0
    geometry_candidate_slots = 0
    candidate_count_histogram: dict[str, int] = {}
    unique_count_histogram: dict[str, int] = {}
    factor_distribution_by_parent: dict[str, dict[str, int]] = {}
    receipt_pass_scene_count = 0
    prefix_equal_counts = {key: 0 for key in PREFIX_FIELDS}
    for index, (old_record, new_record) in enumerate(records):
        scene_id = old_record["scene_id"]
        if scene_id != new_record["scene_id"] or old_record["source_index"] != new_record["source_index"]:
            _add_failure(failures, task, scene_id, "scene_source_identity", f"new scene maps to {new_record['scene_id']} at index {new_record['source_index']}")
        if not old_record["path"].is_file() or not new_record["path"].is_file():
            _add_failure(failures, task, scene_id, "scene_file_present", "legacy or new source scene file missing; source retained in coverage report")
            continue
        try:
            old = _load_npz_fields(old_record["path"])
            new = _load_npz_fields(new_record["path"])
        except Exception as exc:
            _add_failure(failures, task, scene_id, "npz_readable", str(exc))
            continue
        prefix = _check_prefix(task, scene_id, old, new, failures)
        for key, value in prefix.items():
            if value:
                prefix_equal_counts[key] = prefix_equal_counts.get(key, 0) + 1
        receipt = _receipt_for(new_manifest, new_record["row"], scene_id)
        if not receipt and "trajectory_receipt" in new:
            try:
                receipt = json.loads(scalar_text(new["trajectory_receipt"]))
            except Exception:
                receipt = {}
        receipt_checks = _receipt_checks(task, scene_id, receipt, failures)
        if receipt_checks and all(receipt_checks.values()):
            receipt_pass_scene_count += 1
        bank = _check_bank(task, scene_id, old, new, receipt, failures)
        bank_count = int(bank.get("candidate_count", 0))
        candidate_count_histogram[str(bank_count)] = candidate_count_histogram.get(str(bank_count), 0) + 1
        unique_count = int(bank.get("unique_action_count", 0))
        unique_count_histogram[str(unique_count)] = unique_count_histogram.get(str(unique_count), 0) + 1
        for choice in bank.get("factor_choices", []):
            parent_key = str(choice["parent"])
            factor_key = str(choice["factor"])
            parent_factors = factor_distribution_by_parent.setdefault(parent_key, {})
            parent_factors[factor_key] = parent_factors.get(factor_key, 0) + 1
        state_geometry, step_states = _check_state_and_geometry(task, scene_id, new, failures)
        geometry = state_geometry.get("geometry", {})
        geometry_candidate_slots += bank_count
        geometry_unsafe_slots += int(geometry.get("unsafe_candidate_count", 0))
        leaks = _check_leakage(task, scene_id, set(new), set(old), failures)
        try:
            old_actions, _ = _find_candidate_actions(old["candidate_actions"])
            new_actions, _ = _find_candidate_actions(new["candidate_actions"])
            old_pixels = np.asarray(old["future_pixels"])
            old_states = np.asarray(old["future_states"])
            new_pixels = np.asarray(new["future_pixels"])
            new_states = np.asarray(new["future_states"])
            if old_pixels.shape[:3] != (2, len(old_actions), 5) or new_pixels.shape[:3] != (2, len(new_actions), 5):
                raise ValueError(f"future_pixels shape mismatch: old={old_pixels.shape}, new={new_pixels.shape}")
            if old_pixels.shape[-3:] != (224, 224, 3) or new_pixels.shape[-3:] != (224, 224, 3):
                raise ValueError("future pixel frames must be decoded RGB uint8 224x224x3")
            legacy_scene_signals.append(_signal_scene(task, old_pixels, old_states, old_actions))
            new_scene_signals.append(_signal_scene(task, new_pixels, new_states, new_actions))
            aliases = _find_exact_pixel_aliases(new_pixels, step_states, new_actions, task)
            alias_count += int(aliases["qualifying_alias_pair_count"])
            for key, value in aliases["qualifying_alias_pair_counts"].items():
                alias_categories[key] += int(value)
            for example in aliases["examples"]:
                if len(alias_examples) < 80:
                    alias_examples.append({"scene_id": scene_id, **example})
        except Exception as exc:
            _add_failure(failures, task, scene_id, "visible_diagnostic", str(exc))
        # Receipt and source/prefix checks are included in report by count; zero-signal scenes are never removed.
        if index % 32 == 31:
            print(f"{task}: checked {index + 1}/{len(records)} source scenes", file=sys.stderr, flush=True)
    all_checks = {
        "coverage": coverage,
        "prefix_equal_scene_counts": prefix_equal_counts,
        "trajectory_receipts_required": True,
        "trajectory_receipt_pass_scene_count": receipt_pass_scene_count,
        "candidate_count_histogram": candidate_count_histogram,
        "unique_candidate_count_histogram": unique_count_histogram,
        "candidate_count_range": [min((int(key) for key in candidate_count_histogram), default=0), max((int(key) for key in candidate_count_histogram), default=0)],
        "unique_candidate_count_range": [min((int(key) for key in unique_count_histogram), default=0), max((int(key) for key in unique_count_histogram), default=0)],
        "factor_distribution_by_legacy_slot": factor_distribution_by_parent,
        "full_shape_unsafe_candidate_slots": geometry_unsafe_slots,
        "candidate_slots_examined": geometry_candidate_slots,
        "non_hidden_parameter_leakage_check": "parameter-like new NPZ keys rejected; historical condition label fields allowed",
        "factor_ladder": CANDIDATE_FACTORS.astype(float).tolist(),
        "geometry_rule": {"bounds": "all Pymunk body shapes, Circle radius and Poly vertices+radius", "safe_range": [CANVAS_MIN, CANVAS_MAX], "sampled_raw_steps": RAW_FUTURE_STEPS, "condition_coverage": 2},
        "state_diagnostics": {"physical_gap_metric": "mean per-object Euclidean distance between the two condition states at matched horizons; agent uses xy L2, block combines xy L2 with 40px-scaled sin/cos angular distance", "physical_gap_threshold_world_px": PHYSICAL_DIFF_THRESHOLD_WORLD, "equivalent_render_pixels": 2.0, "angle_radius_world_px": ANGLE_RADIUS_WORLD, "threshold_is_diagnostic_not_selection_gate": True, "not_energy_or_model_readout": True},
        "pixel_diagnostics": {"same_image_rule": "exact decoded uint8 equality; confirm hash matches with array equality", "color_mask_tolerance_channel_levels": OBJECT_COLOR_CHANNEL_TOLERANCE, "color_masks_are_diagnostic_only": True},
        "candidate_weighting": "deduplicate by exact float32 candidate action bytes inside each source scene, then weight source scenes equally",
        "zero_signal_handling": "all source/candidate rows retained; zero-signal ratios are reported and never filtered",
    }
    legacy_signal = _merge_macro_summaries(legacy_scene_signals)
    new_signal = _merge_macro_summaries(new_scene_signals)
    retention = []
    for horizon_index in range(5):
        for obj in ("agent", "block"):
            old_obj = legacy_signal["per_horizon"][horizon_index]["objects"][obj]
            new_obj = new_signal["per_horizon"][horizon_index]["objects"][obj]
            old_gap = old_obj.get("mean_world_gap")
            new_gap = new_obj.get("mean_world_gap")
            old_rate = old_obj.get("over_2_render_px_ratio")
            new_rate = new_obj.get("over_2_render_px_ratio")
            retention.append({
                "physical_step": int(HORIZON_STEPS[horizon_index]),
                "object": obj,
                "legacy_mean_world_gap": old_gap,
                "new_mean_world_gap": new_gap,
                "new_to_legacy_mean_world_gap_ratio": (float(new_gap / old_gap) if old_gap not in (None, 0.0) and new_gap is not None else None),
                "legacy_over_2_render_px_ratio": old_rate,
                "new_over_2_render_px_ratio": new_rate,
                "new_to_legacy_over_2_render_px_ratio": (float(new_rate / old_rate) if old_rate not in (None, 0.0) and new_rate is not None else None),
            })
    return {
        "task": task,
        "passed": not failures,
        "failures": failures,
        "checks": all_checks,
        "legacy_conditional_signal": legacy_signal,
        "new_conditional_signal": new_signal,
        "objectwise_conditional_signal_retention": {
            "averaging": "mean over unique candidates and horizons within each source, then equal weight over source scenes",
            "metric": "Euclidean object-state distance between conditions (agent xy; block xy plus angle-equivalent distance); threshold event ratio uses the fixed 2-render-pixel-equivalent gap",
            "interpretation": "descriptive physical panel signal, not energy or learned-model readout; candidate sets differ and this is not a quality score",
            "by_horizon_and_object": retention,
        },
        "local_exact_pixel_aliases": {
            "qualifying_alias_pair_count": alias_count,
            "counts_by_relation": alias_categories,
            "examples_capped_at": 80,
            "examples": alias_examples,
            "interpretation": "A found pair is a local exact-pixel alias with >2-render-pixel-equivalent object-state difference. No found pair does not establish sufficient observability.",
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-root", type=Path, default=DEFAULT_PANEL_ROOT)
    parser.add_argument("--legacy-root", type=Path, default=LEGACY_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--expected-sources", type=int, default=EXPECTED_SOURCES, help="expected new panel source count; use 4 for a frozen pilot subset")
    args = parser.parse_args(argv)
    results = {task: validate_task(task, args.panel_root, args.legacy_root, args.expected_sources) for task in args.tasks}
    report = {
        "schema": "contextworld.pusht_visible_future_validation.v1",
        "panel_root": str(args.panel_root),
        "legacy_root": str(args.legacy_root),
        "expected_source_count_per_task": args.expected_sources,
        "learned_model_readout": "not evaluated by this model-free validator",
        "passed": all(result["passed"] for result in results.values()),
        "tasks": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "output": str(args.output), "failures": {task: len(result["failures"]) for task, result in results.items()}}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
