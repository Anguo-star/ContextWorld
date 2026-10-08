"""Validate history conditioning against every counterfactual target condition.

The existing cross-task runner stores only the diagonal (matching-history)
error.  This runner uses the same adapter, encoder, and free autoregressive
prediction path, then compares every predicted history with every true target
condition.  A scene therefore needs one model pass; the cross-target table is
formed from those predictions and encoded targets in memory.

The full model latents are deliberately not written.  They are used to write
the float64 distance table and a compact float32 feature file.  DINO-WM's
16x16x384 patch grid is average pooled to 4x4x384 (6144 features) before the
feature file is written; LeWM and PLDM retain their native latent shape.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import io
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# These are intentionally imported from the frozen diagnostic runner.  In
# particular, do not reimplement adapter loading, pixel encoding, or the free
# autoregressive prediction loop here: the validity run must use the same
# model path as the saved baseline.
import diagnose_cross_task_decisions as _diagnostic


sha = _diagnostic.sha
load_adapter = _diagnostic.load_adapter
encode_unique = _diagnostic.encode_unique
predict_all = _diagnostic.predict_all


DINOWM_CHANNELS = 384
DINOWM_GRID = 16
DINOWM_FEATURE_GRID = 4
DINOWM_FEATURE_DIM = (
    DINOWM_FEATURE_GRID * DINOWM_FEATURE_GRID * DINOWM_CHANNELS
)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _json_value(value: Any) -> Any:
    """Convert numpy scalars/arrays to JSON-safe values without NaN."""

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    return value


def _same_condition_axis(value: Any, conditions: int, *, atol: float = 0.0) -> bool | None:
    """Return whether an array has equal condition rows, or ``None`` if scalar."""

    array = np.asarray(value)
    if array.ndim == 0 or array.shape[0] != conditions:
        return None
    if array.dtype.kind in "OUS":
        return bool(np.array_equal(array, np.repeat(array[:1], conditions, axis=0)))
    return bool(
        np.allclose(
            array,
            np.repeat(array[:1], conditions, axis=0),
            rtol=0.0,
            atol=atol,
        )
    )


def conditioning_audit(task: str, data: Mapping[str, Any]) -> dict[str, Any]:
    """Describe visual/state matching and queue confounds without changing data."""

    pixels = np.asarray(data["history_pixels"])
    if pixels.ndim < 3:
        raise ValueError(f"history_pixels must have a condition axis: {pixels.shape}")
    conditions = int(pixels.shape[0])
    current_equal = bool(
        np.array_equal(pixels[:, -1], np.repeat(pixels[:1, -1], conditions, axis=0))
    )
    full_history_equal = bool(np.array_equal(pixels, np.repeat(pixels[:1], conditions, axis=0)))

    actions_equal = _same_condition_axis(data["context_actions"], conditions, atol=1e-7)
    audit: dict[str, Any] = {
        "task": str(task),
        "conditions": _json_value(np.asarray(data.get("conditions", []))),
        "condition_count": conditions,
        "current_image_bitwise_equal": current_equal,
        "history_pixels_bitwise_equal": full_history_equal,
        "history_contains_condition_signal": not full_history_equal,
        "context_actions_equal": actions_equal,
        "state_matching": {},
    }

    for name in ("query_state", "initial_state"):
        if name not in data:
            continue
        equal = _same_condition_axis(data[name], conditions, atol=1e-6)
        # Some TwoRoom payloads store one shared query-state vector because
        # the visual query is shared.  That is still an equality claim, but it
        # is useful to disclose that the payload has no condition axis.
        if equal is None:
            equal = True
            audit[f"{name}_condition_axis"] = False
            audit[f"{name}_shared_payload"] = True
        else:
            audit[f"{name}_condition_axis"] = True
            audit[f"{name}_shared_payload"] = False
        audit[f"{name}_matches"] = bool(equal)
        audit["state_matching"][name] = bool(equal)

    pending_name = "aux_pending_actions_at_query"
    length_name = "aux_pending_action_lengths"
    if pending_name in data:
        pending = np.asarray(data[pending_name])
        values_equal = _same_condition_axis(pending, conditions, atol=1e-7)
        lengths = np.asarray(data.get(length_name, np.zeros(conditions, dtype=np.int64)))
        lengths_equal = _same_condition_axis(lengths, conditions, atol=0.0)
        if values_equal is None:
            values_equal = True
        if lengths_equal is None:
            lengths_equal = True
        audit.update(
            {
                "pending_queue_present": True,
                "pending_queue_values_equal": bool(values_equal),
                "pending_queue_lengths_equal": bool(lengths_equal),
                "pending_queue_semantically_different": bool(
                    (not values_equal) or (not lengths_equal)
                ),
                "pending_queue_lengths": _json_value(lengths),
                "pending_queue_max_abs_value": float(
                    np.max(np.abs(pending)) if pending.size else 0.0
                ),
            }
        )
    else:
        audit.update(
            {
                "pending_queue_present": False,
                "pending_queue_values_equal": None,
                "pending_queue_lengths_equal": None,
                "pending_queue_semantically_different": False,
            }
        )

    # This explicit flag is useful when reviewing Action Delay: the rendered
    # query frame is shared, while the queue length remains a hidden simulator
    # state.  It is deliberately reported rather than folded into the main
    # all-other-history metric.
    audit["same_current_image_different_pending_queue"] = bool(
        current_equal and audit["pending_queue_semantically_different"]
    )
    return audit


def compress_features(features: np.ndarray, family: str) -> np.ndarray:
    """Return the feature representation used by the physical-readout probe."""

    values = np.asarray(features)
    if family != "dinowm":
        return np.asarray(values, dtype=np.float32)
    if values.shape[-1] != DINOWM_GRID * DINOWM_GRID * DINOWM_CHANNELS:
        raise ValueError(
            "DINO-WM latent does not have the frozen 16x16x384 patch layout: "
            f"{values.shape}"
        )
    prefix = values.shape[:-1]
    # Patch order is row-major 16x16x384.  The reshape groups each 4x4
    # spatial tile, then the two inner tile axes are averaged.
    grid = values.reshape(*prefix, DINOWM_GRID, DINOWM_GRID, DINOWM_CHANNELS)
    grouped = grid.reshape(
        *prefix,
        DINOWM_FEATURE_GRID,
        DINOWM_GRID // DINOWM_FEATURE_GRID,
        DINOWM_FEATURE_GRID,
        DINOWM_GRID // DINOWM_FEATURE_GRID,
        DINOWM_CHANNELS,
    )
    pooled = grouped.mean(axis=(-4, -2), dtype=np.float64)
    result = pooled.reshape(*prefix, DINOWM_FEATURE_DIM)
    return np.asarray(result, dtype=np.float32)


def _pairwise_squared_distance(
    predicted: np.ndarray,
    target: np.ndarray,
    *,
    chunk_features: int = 4096,
) -> np.ndarray:
    """Compute [history, truth, candidate, time] squared latent distances.

    The feature axis is chunked so DINO-WM's native 98,304-dimensional
    latents never form a KxKxC time broadcast.  Accumulation and products are
    float64 to avoid cancellation in the cross-target comparison.
    """

    predicted = np.asarray(predicted)
    target = np.asarray(target)
    if predicted.shape != target.shape or predicted.ndim != 4:
        raise ValueError(
            "predicted and target must share [K,C,T,D] shape: "
            f"{predicted.shape} vs {target.shape}"
        )
    conditions, candidates, times, dimensions = predicted.shape
    distance = np.zeros(
        (conditions, conditions, candidates, times), dtype=np.float64
    )
    chunk_features = max(1, int(chunk_features))
    for start in range(0, dimensions, chunk_features):
        stop = min(dimensions, start + chunk_features)
        p = np.asarray(predicted[..., start:stop], dtype=np.float64)
        y = np.asarray(target[..., start:stop], dtype=np.float64)
        p_norm = np.sum(p * p, axis=-1, dtype=np.float64)
        y_norm = np.sum(y * y, axis=-1, dtype=np.float64)
        dot = np.einsum("hctd,kctd->hkct", p, y, optimize=True)
        distance += p_norm[:, None] + y_norm[None, :] - 2.0 * dot

    # The direct diagonal agrees with the historical per-history error
    # calculation and avoids any residual norm subtraction for the matching
    # condition.  It is still accumulated in float64.
    diagonal = np.sum(
        (np.asarray(predicted, dtype=np.float64) - np.asarray(target, dtype=np.float64)) ** 2,
        axis=-1,
        dtype=np.float64,
    )
    indices = np.arange(conditions)
    distance[indices, indices] = diagonal
    # Floating point norm identities can produce tiny negative values for
    # identical vectors.  Distances are energies, so clamp only that artifact.
    return np.maximum(distance, 0.0)


def _centered_stats(
    predicted: np.ndarray,
    target: np.ndarray,
    *,
    chunk_features: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    """Return response-error and target-separation sums by time."""

    predicted = np.asarray(predicted)
    target = np.asarray(target)
    if predicted.shape != target.shape:
        raise ValueError("Predicted and target shapes differ")
    dimensions = predicted.shape[-1]
    p_mean = np.mean(np.asarray(predicted, dtype=np.float64), axis=0)
    y_mean = np.mean(np.asarray(target, dtype=np.float64), axis=0)
    response = np.zeros(predicted.shape[2], dtype=np.float64)
    separation = np.zeros(predicted.shape[2], dtype=np.float64)
    for start in range(0, dimensions, max(1, int(chunk_features))):
        stop = min(dimensions, start + max(1, int(chunk_features)))
        p = np.asarray(predicted[..., start:stop], dtype=np.float64)
        y = np.asarray(target[..., start:stop], dtype=np.float64)
        pm = p_mean[..., start:stop]
        ym = y_mean[..., start:stop]
        response += np.sum(
            ((p - pm[None]) - (y - ym[None])) ** 2,
            axis=(0, 1, 3),
            dtype=np.float64,
        )
        separation += np.sum(
            (y - ym[None]) ** 2,
            axis=(0, 1, 3),
            dtype=np.float64,
        )
    return response, separation


def _condition_mean_energy(target: np.ndarray, *, chunk_features: int = 4096) -> np.ndarray:
    """Energy of the oracle condition-mean target, by time."""

    target = np.asarray(target)
    mean = np.mean(np.asarray(target, dtype=np.float64), axis=0)
    result = np.zeros(target.shape[2], dtype=np.float64)
    for start in range(0, target.shape[-1], max(1, int(chunk_features))):
        stop = min(target.shape[-1], start + max(1, int(chunk_features)))
        values = np.asarray(target[..., start:stop], dtype=np.float64)
        result += np.sum(
            (values - mean[None, ..., start:stop]) ** 2,
            axis=(0, 1, 3),
            dtype=np.float64,
        )
    return result


def _physical_group_info(
    task: str,
    data: Mapping[str, Any],
    conditions: int,
    times: int,
) -> tuple[np.ndarray | None, np.ndarray | None, str | None]:
    """Return action-delay distinguishability groups and ordered-pair mask."""

    if task != "action_delay" or conditions != 11:
        return None, None, None
    values = np.asarray(data.get("conditions", np.arange(conditions)))
    try:
        delays = np.rint(values.astype(np.float64)).astype(np.int64)
    except (TypeError, ValueError) as exc:
        raise ValueError("Action Delay conditions must be numeric delays") from exc
    if delays.shape != (conditions,):
        raise ValueError(f"Unexpected Action Delay conditions: {delays.shape}")
    groups = np.repeat(delays[:, None], times, axis=1)
    if times:
        # At the first 5-raw-step endpoint, delays 5..10 are stationary and
        # therefore physically indistinguishable.  Longer trajectories split
        # every delay, as in the frozen Action Delay scorer.
        groups[:, 0] = np.minimum(delays, 5)
    mask = groups[:, None, :] != groups[None, :, :]
    return (
        groups,
        mask,
        "Action Delay physical_future_group: min(delay,5) at h1; delay at h2-h5",
    )


def _energy_by_mask(
    distance: np.ndarray,
    mask: np.ndarray,
) -> tuple[list[float], list[float], list[int]]:
    """Sum/mean distances over an ordered pair mask with equal pair weights."""

    values = np.asarray(distance)
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != values.shape[:2] + (values.shape[-1],):
        raise ValueError(f"Mask shape {mask.shape} incompatible with {values.shape}")
    sums: list[float] = []
    means: list[float] = []
    counts: list[int] = []
    for time_index in range(values.shape[-1]):
        selected = values[:, :, :, time_index][mask[:, :, time_index]]
        # The mask selects ordered history/target pairs; each selected row
        # still contains every candidate.  Keep pair_count separate from the
        # number of scalar entries so the mean remains candidate-weighted once
        # when C > 1.
        pair_count = int(selected.shape[0])
        counts.append(pair_count)
        total = float(np.sum(selected, dtype=np.float64))
        sums.append(total)
        means.append(
            total / float(pair_count * values.shape[2])
            if pair_count
            else float("nan")
        )
    return sums, means, counts


def summarize_cross_target_distances(
    distance: np.ndarray,
    target: np.ndarray,
    *,
    horizons: Sequence[int],
    task: str,
    data: Mapping[str, Any],
) -> dict[str, Any]:
    """Summarize matching and all-other history errors without filtering."""

    distance = np.asarray(distance, dtype=np.float64)
    target = np.asarray(target)
    if distance.ndim != 4 or distance.shape[0] != distance.shape[1]:
        raise ValueError(f"Expected [K,K,C,T] distance table, got {distance.shape}")
    conditions, _, candidates, times = distance.shape
    if target.shape[:3] != (conditions, candidates, times):
        raise ValueError(f"Target shape {target.shape} disagrees with {distance.shape}")
    if len(horizons) != times:
        raise ValueError(f"Horizon count {len(horizons)} disagrees with T={times}")

    diagonal = distance[np.arange(conditions), np.arange(conditions)]
    off_diagonal_mask = ~np.eye(conditions, dtype=bool)
    off_diagonal = distance[off_diagonal_mask]
    matched_sum = np.sum(diagonal, axis=(0, 1), dtype=np.float64)
    matched_mean = np.mean(diagonal, axis=(0, 1), dtype=np.float64)
    wrong_sum = np.sum(off_diagonal, axis=(0, 1), dtype=np.float64)
    wrong_mean = np.mean(off_diagonal, axis=(0, 1), dtype=np.float64)
    oracle_sum = _condition_mean_energy(target)

    result: dict[str, Any] = {
        "conditions": int(conditions),
        "candidates": int(candidates),
        "horizons": [int(value) for value in horizons],
        "matched": {
            "energy_sum": matched_sum.tolist(),
            "energy_mean": matched_mean.tolist(),
            "pair_count": int(conditions * candidates),
            "definition": "history condition equals target condition; equal weight per condition/candidate",
        },
        "wrong_allother": {
            "energy_sum": wrong_sum.tolist(),
            "energy_mean": wrong_mean.tolist(),
            "pair_count": int(conditions * (conditions - 1) * candidates),
            "definition": "all ordered history/target condition mismatches; equal weight per ordered pair/candidate",
        },
        "history_conditioning_gain": {
            "wrong_minus_matched_energy_mean": (wrong_mean - matched_mean).tolist(),
            "wrong_minus_matched_energy_sum_per_matched_pair": (
                wrong_mean - matched_mean
            ).tolist(),
        },
        "oracle_condition_mean_counterfactual": {
            "energy_sum": oracle_sum.tolist(),
            "energy_mean": (oracle_sum / float(conditions * candidates)).tolist(),
            "definition": "oracle prediction is the mean true target over all condition rows for each candidate/time",
        },
        # The old result schema calls this target separation energy.  Keep it
        # as a sum so the free-error compatibility check is unit-consistent.
        "target_separation_energy": oracle_sum.tolist(),
    }
    # Keep a short alias for consumers that call the comparison simply
    # "matched versus wrong"; the explicit wrong_allother object remains the
    # canonical name because it records that no physical-pair filtering was
    # applied.
    result["wrong"] = result["wrong_allother"]

    groups, distinguishable, group_definition = _physical_group_info(
        task, data, conditions, times
    )
    if groups is not None and distinguishable is not None:
        physical_sum, physical_mean, physical_counts = _energy_by_mask(
            distance, distinguishable
        )
        result["physical_group_mask"] = {
            "available": True,
            "definition": group_definition,
            "group_ids_by_condition_and_horizon": groups.tolist(),
            "distinguishable_pair_count_by_horizon": physical_counts,
            "wrong_physically_distinguishable": {
                "energy_sum": physical_sum,
                "energy_mean": physical_mean,
                "pair_count_by_horizon": physical_counts,
                "note": "Secondary diagnostic; wrong_allother remains the primary unfiltered comparison.",
            },
        }
    else:
        result["physical_group_mask"] = {
            "available": False,
            "definition": "No task-specific physical-group mask registered for this panel.",
        }
    return _json_value(result)


def evaluate_conditioning(
    adapter: Any,
    data: Mapping[str, Any],
    family: str,
    *,
    task: str,
    batch_size: int = 8,
) -> dict[str, Any]:
    """Run one free prediction pass and form all history/target comparisons."""

    pixels = np.asarray(data["history_pixels"])
    conditions, history_tokens = pixels.shape[:2]
    future_pixels = np.asarray(data["future_pixels"])
    candidate_actions = np.asarray(data["candidate_actions"])
    candidates, times = candidate_actions.shape[:2]
    if future_pixels.shape[:3] != (conditions, candidates, times):
        raise ValueError(
            "future_pixels must have [K,C,T] axes: "
            f"{future_pixels.shape} vs {(conditions, candidates, times)}"
        )
    audit = conditioning_audit(task, data)
    if not audit["current_image_bitwise_equal"]:
        raise ValueError("Current query image is not bitwise equal across conditions")
    if audit["context_actions_equal"] is not True:
        raise ValueError("Past context actions are not equal across conditions")

    # Encode all history and all true future rows once.  Include the frozen
    # goal frame in this call when present so the encoder batching/order is the
    # same as diagnose_cross_task_decisions.evaluate; discard its latent below.
    # This keeps the saved diagonal free error within the old-result tolerance.
    encode_inputs: list[np.ndarray] = [pixels, future_pixels]
    if "goal_pixels" in data:
        encode_inputs.append(np.asarray(data["goal_pixels"])[None])
    encoded = encode_unique(adapter, *encode_inputs)
    history_latent, target_latent = encoded[:2]
    history_latent = np.asarray(history_latent)
    target_latent = np.asarray(target_latent)
    if target_latent.ndim != 4:
        raise ValueError(f"Unexpected target latent shape: {target_latent.shape}")
    if history_latent.shape[:2] != (conditions, history_tokens):
        raise ValueError(f"Unexpected history latent shape: {history_latent.shape}")
    dimensions = int(target_latent.shape[-1])
    if history_latent.shape[-1] != dimensions:
        raise ValueError("History and target latent dimensions differ")

    repeated_history = np.repeat(history_latent, candidates, axis=0)
    raw_actions = np.concatenate(
        [
            np.repeat(np.asarray(data["context_actions"]), candidates, axis=0),
            np.tile(candidate_actions, (conditions, 1, 1, 1)),
        ],
        axis=1,
    )
    # One call covers all K history rows and all C candidate sequences.  The
    # target argument is used only for shape/rollout length by this mode.
    predicted = predict_all(
        adapter,
        repeated_history,
        target_latent.reshape(conditions * candidates, times, dimensions),
        raw_actions,
        family,
        batch_size=batch_size,
        modes=("free",),
    )["free"].reshape(conditions, candidates, times, dimensions)

    distance = _pairwise_squared_distance(predicted, target_latent)
    horizons = np.asarray(
        data.get("physical_steps", np.arange(1, times + 1) * 5)
    ).reshape(-1)
    if len(horizons) != times:
        raise ValueError(f"Unexpected physical_steps: {horizons.shape}")
    summary = summarize_cross_target_distances(
        distance,
        target_latent,
        horizons=[int(value) for value in horizons],
        task=task,
        data=data,
    )
    response, separation = _centered_stats(predicted, target_latent)
    summary["modes"] = {
        "free": {
            "prediction_error_sum": summary["matched"]["energy_sum"],
            "response_error_sum": response.tolist(),
            "ranking_is_not_used": True,
        }
    }
    summary["target_separation_energy"] = separation.tolist()
    summary["conditioning_audit"] = audit
    summary["latent_dimension"] = dimensions
    summary["feature_dimension"] = (
        DINOWM_FEATURE_DIM if family == "dinowm" else dimensions
    )
    summary["feature_layout"] = (
        {
            "kind": "dino_patch_average_pool",
            "source_patch_grid": [DINOWM_GRID, DINOWM_GRID, DINOWM_CHANNELS],
            "pool_grid": [DINOWM_FEATURE_GRID, DINOWM_FEATURE_GRID, DINOWM_CHANNELS],
            "flatten_order": "row_major_spatial_then_channel",
        }
        if family == "dinowm"
        else {
            "kind": "native_latent",
            "source_last_dimension": dimensions,
            "flatten_order": "native",
        }
    )
    summary["feature_not_used_in_native_score"] = True
    summary["free_prediction_shape"] = [int(v) for v in predicted.shape]
    return {
        "summary": _json_value(summary),
        "distance": distance,
        "predicted": predicted,
        "target": target_latent,
        "features_pred": compress_features(predicted, family),
        "features_target": compress_features(target_latent, family),
    }


def _cached_row(
    path: Path,
    *,
    checkpoint_sha256: str,
    source_sha256: str,
) -> dict[str, Any] | None:
    """Validate a completed scene cache before reusing it."""

    if not path.exists():
        return None
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("checkpoint_sha256") != checkpoint_sha256:
            return None
        if row.get("source_sha256") != source_sha256:
            return None
        if row.get("old_result_check", {}).get("passed") is False:
            raise RuntimeError(f"Cached row has a failed old-result check: {path}")
        for key, array_names in (
            ("distance_file", ("distance",)),
            ("features_file", ("pred", "target")),
        ):
            array_path = path.parent / str(row[key])
            if not array_path.exists() or row.get(f"{key}_sha256") != sha(array_path):
                return None
            with np.load(array_path, allow_pickle=False) as archive:
                if not set(array_names).issubset(archive.files):
                    return None
        return row
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _old_result_check(
    spec: Mapping[str, Any],
    scene_path: Path,
    source_sha256: str,
    matched_energy_sum: Sequence[float],
) -> dict[str, Any]:
    """Compare the new free diagonal energy to the saved baseline row."""

    previous_root = spec.get("previous_result_dir")
    if not previous_root:
        return {"status": "not_configured", "rtol": 1e-5, "atol": 1e-8}
    old_path = Path(str(previous_root)) / f"{scene_path.stem}.json"
    base: dict[str, Any] = {
        "status": "missing",
        "path": str(old_path),
        "rtol": 1e-5,
        "atol": 1e-8,
        "checkpoint_match": None,
        "source_match": None,
        "free_error_match": None,
    }
    if not old_path.exists():
        return base
    try:
        old = json.loads(old_path.read_text(encoding="utf-8"))
        old_error = np.asarray(
            old["modes"]["free"]["prediction_error_sum"], dtype=np.float64
        )
        new_error = np.asarray(matched_energy_sum, dtype=np.float64)
        checkpoint_match = old.get("checkpoint_sha256") == spec.get(
            "checkpoint_sha256"
        )
        source_match = old.get("source_sha256") == source_sha256
        free_match = bool(
            old_error.shape == new_error.shape
            and np.allclose(new_error, old_error, rtol=1e-5, atol=1e-8)
        )
        base.update(
            {
                "status": "checked",
                "checkpoint_match": bool(checkpoint_match),
                "source_match": bool(source_match),
                "free_error_match": free_match,
                "old_free_prediction_error_sum": old_error.tolist(),
                "new_free_prediction_error_sum": new_error.tolist(),
                "passed": bool(checkpoint_match and source_match and free_match),
            }
        )
        return base
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        base.update({"status": "invalid", "error": str(exc), "passed": False})
        return base


def _manifest_entries(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    for key in ("scenes", "pairs", "queries"):
        entries = manifest.get(key)
        if entries is not None:
            return [dict(entry) for entry in entries]
    raise ValueError("Panel manifest has no scenes/pairs/queries list")


def _normalization(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    normalization = manifest.get("normalization")
    if normalization is None:
        protocol = manifest.get("protocol", {})
        normalization = protocol.get("normalization")
    if not normalization:
        raise ValueError("Panel manifest has no frozen normalization")
    return normalization


def run_model_panel(
    *,
    spec: Mapping[str, Any],
    panel: Path,
    output: Path,
    device: str = "cpu",
    threads: int = 2,
    batch_size: int = 8,
    shard: int = 0,
    shards: int = 1,
) -> dict[str, Any]:
    """Run/resume one model specification over a panel shard."""

    if shards < 1 or shard < 0 or shard >= shards:
        raise ValueError(f"Invalid shard {shard}/{shards}")
    panel = Path(panel)
    output = Path(output)
    manifest_path = panel / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = _manifest_entries(manifest)
    output.mkdir(parents=True, exist_ok=True)

    # Import torch only for the model runner.  No CUDA call is made when the
    # caller selects --device cpu.
    import torch

    torch.set_num_threads(int(threads))
    torch.set_num_interop_threads(1)
    torch.manual_seed(20260930)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    adapter = load_adapter(
        dict(spec), _normalization(manifest), output, device=str(device)
    )
    before = adapter.frozen_state_hash() if hasattr(adapter, "frozen_state_hash") else None

    rows: list[dict[str, Any]] = []
    selected_total = max(0, (len(entries) - 1 - shard) // shards + 1)
    for index, entry in enumerate(entries):
        if index % shards != shard:
            continue
        scene_path = panel / str(entry["path"])
        source_bytes = scene_path.read_bytes()
        source_sha256 = hashlib.sha256(source_bytes).hexdigest()
        expected_sha = entry.get("sha256")
        if expected_sha is not None and source_sha256 != expected_sha:
            raise ValueError(f"Source hash changed: {scene_path}")
        row_path = output / f"{scene_path.stem}.json"
        cached = _cached_row(
            row_path,
            checkpoint_sha256=str(spec["checkpoint_sha256"]),
            source_sha256=source_sha256,
        )
        if cached is not None:
            rows.append(cached)
            if len(rows) % 16 == 0 or len(rows) == selected_total:
                print(f"{spec['id']} {len(rows)}/{selected_total} scenes", flush=True)
            continue

        with np.load(io.BytesIO(source_bytes), allow_pickle=False) as archive:
            data = {name: archive[name] for name in archive.files}
        evaluated = evaluate_conditioning(
            adapter,
            data,
            str(spec["family"]),
            task=str(spec["task"]),
            batch_size=int(batch_size),
        )
        summary = evaluated["summary"]
        old_result_check = _old_result_check(
            spec,
            scene_path,
            source_sha256,
            summary["matched"]["energy_sum"],
        )
        if old_result_check.get("passed") is False:
            raise RuntimeError(
                "Saved free-error compatibility check failed for "
                f"{scene_path}: {json.dumps(old_result_check, ensure_ascii=False)}"
            )
        distance_path = output / f"distance_{scene_path.stem}.npz"
        feature_path = output / f"features_{scene_path.stem}.npz"
        np.savez_compressed(distance_path, distance=evaluated["distance"])
        # Keep features uncompressed: the files are already float32 and the
        # DINO representation can be tens of GB across all scenes; avoiding a
        # second compression pass materially reduces batch CPU time.
        np.savez(
            feature_path,
            pred=np.asarray(evaluated["features_pred"], dtype=np.float32),
            target=np.asarray(evaluated["features_target"], dtype=np.float32),
        )
        row = {
            "schema_version": 1,
            "scene_id": entry.get(
                "scene_id", entry.get("pair_id", entry.get("query_id", scene_path.stem))
            ),
            "model_id": str(spec["id"]),
            "task": str(spec["task"]),
            "family": str(spec["family"]),
            "checkpoint_sha256": str(spec["checkpoint_sha256"]),
            "source_sha256": source_sha256,
            "manifest_sha256": sha(manifest_path),
            "distance_file": distance_path.name,
            "distance_file_sha256": sha(distance_path),
            "features_file": feature_path.name,
            "features_file_sha256": sha(feature_path),
            "old_result_check": old_result_check,
            **summary,
        }
        _write_json(row_path, _json_value(row))
        rows.append(row)
        if len(rows) % 16 == 0 or len(rows) == selected_total:
            print(f"{spec['id']} {len(rows)}/{selected_total} scenes", flush=True)

    after = adapter.frozen_state_hash() if hasattr(adapter, "frozen_state_hash") else None
    if before is not None and after is not None and before != after:
        raise RuntimeError("Adapter state changed during validity evaluation")
    receipt_name = "receipt.json" if shards == 1 else f"shard{shard}_receipt.json"
    receipt = {
        "schema_version": 1,
        "model_id": str(spec["id"]),
        "task": str(spec["task"]),
        "family": str(spec["family"]),
        "checkpoint_sha256": str(spec["checkpoint_sha256"]),
        "panel_sha256": sha(manifest_path),
        "scenes": len(rows),
        "shard": int(shard),
        "shards": int(shards),
        "device": str(device),
        "inference_batch_size": int(batch_size),
        "no_training": True,
        "free_only_single_prediction_pass": True,
        "state_hash_before": before,
        "state_hash_after": after,
        "entry_hashes": {
            path.name: sha(path)
            for path in output.glob("*.json")
            if path.name != receipt_name
        },
    }
    _write_json(output / receipt_name, receipt)
    return receipt


def _load_spec(models_path: Path, model_id: str) -> dict[str, Any]:
    specs = json.loads(Path(models_path).read_text(encoding="utf-8"))
    matches = [dict(spec) for spec in specs if str(spec.get("id")) == str(model_id)]
    if len(matches) != 1:
        raise ValueError(f"Expected one model id {model_id!r}, found {len(matches)}")
    return matches[0]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--id", required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output", type=Path)

    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args(argv)
    spec = _load_spec(args.models, args.id)
    panel = args.panel
    if args.output is None and not spec.get("result_dir"):
        parser.error("Specify --output or result_dir in the model manifest")
    output = args.output or Path(str(spec["result_dir"]))
    run_model_panel(
        spec=spec,
        panel=panel,
        output=output,
        device=args.device,
        threads=args.threads,
        batch_size=args.batch_size,
        shard=args.shard,
        shards=args.shards,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
