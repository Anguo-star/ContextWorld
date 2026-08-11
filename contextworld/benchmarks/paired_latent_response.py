"""Shared response-amplitude metrics for paired hidden-dynamics tasks.

Each benchmark pair holds the current observation and query action fixed while
changing only the history.  The simulator supplies two corresponding future
latents, ``target_first`` and ``target_second``.  This module measures whether
the model reproduces the *difference* between those futures, rather than only
putting its two predictions on the correct side of a nearest-target decision.

All headline metrics are dimensionless and are computed from

``prediction_response = predicted_second - predicted_first`` and
``target_response = target_second - target_first``.

Their ideal values are: response gain 1, response alignment 1, and normalized
response error 0.  A pair's response succeeds when its normalized error is
strictly below 1: this is exactly better than the no-history-response baseline
``prediction_response = 0``.  No method-level pass threshold is defined here;
that threshold must be calibrated on Development and frozen before Public Test.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Sequence

import numpy as np


class TargetLatentSeparationError(RuntimeError):
    """Raised when a paired target is indistinguishable in latent space."""


def _as_latent_rows(name: str, value: np.ndarray) -> np.ndarray:
    rows = np.asarray(value)
    if rows.ndim < 2:
        raise ValueError(f"{name} must have shape (pairs, latent_dims...)")
    if not np.issubdtype(rows.dtype, np.number):
        raise TypeError(f"{name} must be numeric")
    rows = rows.astype(np.float64, copy=False).reshape(len(rows), -1)
    if not np.isfinite(rows).all():
        raise ValueError(f"{name} contains a non-finite latent value")
    return rows


def paired_latent_response_metrics(
    *,
    pair_ids: Sequence[str],
    predicted_first: np.ndarray,
    predicted_second: np.ndarray,
    target_first: np.ndarray,
    target_second: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Measure the magnitude and direction of a paired latent response.

    The aggregate response gain is the least-squares scalar projection of the
    predicted response onto the target response.  Response alignment is their
    cosine similarity after concatenating all pairs.  Normalized response
    error is squared response error divided by squared target response.

    The function deliberately rejects even one exactly unseparated target
    pair.  Such a pair cannot identify history-conditioned prediction and
    therefore must not silently count toward an ICL score.  Near-zero but
    nonzero separations remain visible through the reported minimum target
    response MSE; a release-specific practical threshold must be calibrated
    from Development data rather than invented in this shared implementation.
    """

    identifiers = tuple(str(pair_id) for pair_id in pair_ids)
    if not identifiers:
        raise ValueError("At least one paired query is required")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("pair_ids must be unique")

    arrays = {
        "predicted_first": _as_latent_rows(
            "predicted_first", predicted_first
        ),
        "predicted_second": _as_latent_rows(
            "predicted_second", predicted_second
        ),
        "target_first": _as_latent_rows("target_first", target_first),
        "target_second": _as_latent_rows("target_second", target_second),
    }
    shapes = {rows.shape for rows in arrays.values()}
    if len(shapes) != 1:
        observed = {name: rows.shape for name, rows in arrays.items()}
        raise ValueError(f"Paired latent arrays must share a shape: {observed}")
    pair_count = len(arrays["target_first"])
    if pair_count != len(identifiers):
        raise ValueError(
            "pair_ids and paired latent arrays must have the same length"
        )

    target_response = arrays["target_second"] - arrays["target_first"]
    prediction_response = (
        arrays["predicted_second"] - arrays["predicted_first"]
    )
    separated = np.any(target_response != 0.0, axis=1)
    if not separated.all():
        collapsed = [
            identifiers[index]
            for index in np.flatnonzero(~separated).tolist()
        ]
        preview = ", ".join(collapsed[:5])
        suffix = "" if len(collapsed) <= 5 else ", ..."
        raise TargetLatentSeparationError(
            "Real future target latents are identical for "
            f"{len(collapsed)} pair(s): {preview}{suffix}"
        )

    target_energy = np.square(target_response).mean(axis=1)
    prediction_energy = np.square(prediction_response).mean(axis=1)
    response_error_energy = np.square(
        prediction_response - target_response
    ).mean(axis=1)
    cross_energy = (prediction_response * target_response).mean(axis=1)

    # Target energy is strictly positive after the exact separation check.
    pair_gain = cross_energy / target_energy
    pair_normalized_error = response_error_energy / target_energy
    pair_alignment_denominator = np.sqrt(
        prediction_energy * target_energy
    )
    pair_alignment = np.divide(
        cross_energy,
        pair_alignment_denominator,
        out=np.zeros_like(cross_energy),
        where=pair_alignment_denominator > 0.0,
    )

    records = [
        {
            "pair_id": pair_id,
            "response_gain": float(pair_gain[index]),
            "response_alignment": float(pair_alignment[index]),
            "normalized_response_error": float(
                pair_normalized_error[index]
            ),
            "calibrated_response_success": bool(
                pair_normalized_error[index] < 1.0
            ),
            "target_response_mse": float(target_energy[index]),
            "prediction_response_mse": float(prediction_energy[index]),
        }
        for index, pair_id in enumerate(identifiers)
    ]
    metrics = summarize_paired_latent_response_records(records)
    return metrics, records


def summarize_paired_latent_response_records(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Reconstruct aggregate response metrics from auditable pair records."""

    if not records:
        raise ValueError("At least one paired response record is required")
    identifiers = tuple(str(row["pair_id"]) for row in records)
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Paired response record IDs must be unique")

    def values(name: str) -> np.ndarray:
        result = np.asarray(
            [float(row[name]) for row in records], dtype=np.float64
        )
        if not np.isfinite(result).all():
            raise ValueError(f"Paired response field {name} is not finite")
        return result

    gain = values("response_gain")
    alignment = values("response_alignment")
    normalized_error = values("normalized_response_error")
    calibrated_success = np.asarray(
        [bool(row["calibrated_response_success"]) for row in records],
        dtype=bool,
    )
    target_energy = values("target_response_mse")
    prediction_energy = values("prediction_response_mse")
    if np.any(target_energy <= 0.0):
        collapsed = [
            identifiers[index]
            for index in np.flatnonzero(target_energy <= 0.0).tolist()
        ]
        raise TargetLatentSeparationError(
            "Target response MSE must be strictly positive for every pair: "
            + ", ".join(collapsed[:5])
        )
    if np.any(prediction_energy < 0.0) or np.any(normalized_error < 0.0):
        raise ValueError(
            "Response energies and normalized error cannot be negative"
        )

    cross_energy = gain * target_energy
    expected_normalized_error = (
        prediction_energy / target_energy + 1.0 - 2.0 * gain
    )
    if not np.allclose(
        normalized_error,
        expected_normalized_error,
        rtol=1e-7,
        atol=1e-10,
    ):
        raise ValueError("Paired response record geometry is inconsistent")
    expected_success = normalized_error < 1.0
    if not np.array_equal(calibrated_success, expected_success):
        raise ValueError(
            "Calibrated response success must be normalized error < 1"
        )
    pair_alignment_denominator = np.sqrt(
        prediction_energy * target_energy
    )
    expected_alignment = np.divide(
        cross_energy,
        pair_alignment_denominator,
        out=np.zeros_like(cross_energy),
        where=pair_alignment_denominator > 0.0,
    )
    if not np.allclose(
        alignment,
        expected_alignment,
        rtol=1e-7,
        atol=1e-10,
    ):
        raise ValueError("Paired response alignment record is inconsistent")

    target_total = float(target_energy.sum())
    prediction_total = float(prediction_energy.sum())
    cross_total = float(cross_energy.sum())
    error_total = float((normalized_error * target_energy).sum())
    aggregate_alignment_denominator = np.sqrt(
        prediction_total * target_total
    )
    response_alignment = (
        cross_total / aggregate_alignment_denominator
        if aggregate_alignment_denominator > 0.0
        else 0.0
    )
    return {
        "response_gain": float(cross_total / target_total),
        "response_alignment": float(response_alignment),
        "normalized_response_error": float(error_total / target_total),
        "calibrated_response_success_rate": float(
            calibrated_success.mean()
        ),
        "calibrated_response_success_baseline": {
            "name": "no_history_response",
            "prediction_response": "zero",
            "normalized_response_error": 1.0,
            "pair_success_criterion": "normalized_response_error < 1.0",
        },
        "target_latent_separation": {
            "passed": True,
            "criterion": "finite_and_nonidentical_for_every_pair",
            "zero_separation_pair_count": 0,
            "minimum_target_response_mse": float(target_energy.min()),
            "median_target_response_mse": float(
                np.median(target_energy)
            ),
            "mean_target_response_mse": float(target_energy.mean()),
        },
        "ideal_values": {
            "response_gain": 1.0,
            "response_alignment": 1.0,
            "normalized_response_error": 0.0,
            "calibrated_response_success_rate": 1.0,
        },
    }


def paired_latent_response_summaries_close(
    observed: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    rtol: float = 1e-7,
    atol: float = 1e-10,
) -> bool:
    """Compare independently reconstructed nested response summaries."""

    def close(left: Any, right: Any) -> bool:
        if isinstance(left, Mapping) or isinstance(right, Mapping):
            return bool(
                isinstance(left, Mapping)
                and isinstance(right, Mapping)
                and set(left) == set(right)
                and all(close(left[name], right[name]) for name in left)
            )
        if isinstance(left, (str, bool)) or isinstance(right, (str, bool)):
            return left == right
        try:
            return bool(
                np.isclose(
                    float(left), float(right), rtol=rtol, atol=atol
                )
            )
        except (TypeError, ValueError):
            return left == right

    return close(observed, expected)


def paired_latent_response_gate_checks(
    metrics: Mapping[str, Any],
    *,
    thresholds: Mapping[str, Any],
) -> dict[str, bool]:
    """Return the three theory-anchored anti-spoofing gate checks.

    Missing response fields fail closed.  The thresholds operate only on
    within-checkpoint, target-normalized response geometry; they never compare
    raw latent losses across checkpoints.
    """

    response = metrics.get("latent_response")
    if not isinstance(response, Mapping):
        response = {}
    separation = response.get("target_latent_separation")
    if not isinstance(separation, Mapping):
        separation = {}
    try:
        response_gain = float(response.get("response_gain", float("-inf")))
    except (TypeError, ValueError):
        response_gain = float("-inf")
    try:
        normalized_error = float(
            response.get("normalized_response_error", float("inf"))
        )
    except (TypeError, ValueError):
        normalized_error = float("inf")
    return {
        "target_latent_separation": bool(
            thresholds.get("target_latent_separation_required") is True
            and separation.get("passed") is True
        ),
        "response_gain": bool(
            np.isfinite(response_gain)
            and response_gain
            >= float(thresholds["response_gain_minimum"])
        ),
        "normalized_response_error": bool(
            np.isfinite(normalized_error)
            and normalized_error
            < float(
                thresholds[
                    "normalized_response_error_strict_maximum"
                ]
            )
        ),
    }


__all__ = [
    "TargetLatentSeparationError",
    "paired_latent_response_metrics",
    "paired_latent_response_gate_checks",
    "paired_latent_response_summaries_close",
    "summarize_paired_latent_response_records",
]
