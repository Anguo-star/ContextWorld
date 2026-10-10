"""Physical multistep prediction accuracy from fixed position tolerances.

Errors have shape ``[scene, candidate, horizon]`` and targets/predictions have
shape ``[scene, candidate, horizon, component]``.  The scalar accuracy is the
equal-scene mean percentage of error samples strictly inside each supplied
tolerance, averaged over tolerances.  Thresholds are supplied by the caller;
this module does not select or tune them from model outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class PositionGroup:
    """Components and optional per-component unit conversion for one object."""

    indices: tuple[int, ...]
    scale: float | tuple[float, ...] = 1.0


# Inputs use the canonical physical units exported by
# validate_physical_readout.physical_targets_from_panel: pixels for TwoRoom,
# pixel-equivalents for PushT, and millimetres for mass/cube. PushT's two angle
# channels are already scaled to (40*sin(theta), 40*cos(theta)).
POSITION_ERROR_GROUPS: dict[str, dict[str, PositionGroup]] = {
    "pusht": {
        "agent": PositionGroup((0, 1)),
        "block": PositionGroup((2, 3, 4, 5)),
    },
    "tworoom": {"object": PositionGroup((0, 1))},
    "mass": {"object": PositionGroup((0, 1))},
    "cube": {
        "effector": PositionGroup((0, 1, 2)),
        "cube": PositionGroup((3, 4, 5)),
    },
}


@dataclass(frozen=True)
class AccuracyResult:
    """One scalar score plus curves and raw-unit RMSE for interpretation."""

    score: float
    tolerance_curve: FloatArray
    horizon_curve: FloatArray
    tolerance_horizon_curve: FloatArray
    max_horizon_score: float
    max_horizon_tolerance_curve: FloatArray
    raw_rmse: float

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready summary without NumPy scalar/array values."""

        return {
            "score": self.score,
            "tolerance_curve": self.tolerance_curve.tolist(),
            "horizon_curve": self.horizon_curve.tolist(),
            "tolerance_horizon_curve": self.tolerance_horizon_curve.tolist(),
            "max_horizon_score": self.max_horizon_score,
            "max_horizon_tolerance_curve": self.max_horizon_tolerance_curve.tolist(),
            "raw_rmse": self.raw_rmse,
        }


@dataclass(frozen=True)
class PairedBootstrapResult:
    """Paired cluster-bootstrap estimate and percentile confidence interval."""

    estimate: float
    lower: float
    upper: float
    bootstrap_estimates: FloatArray
    mean_a: float
    mean_b: float
    confidence: float

    @property
    def delta(self) -> float:
        """Paired mean difference ``A - B``."""

        return self.estimate

    def to_dict(self) -> dict[str, object]:
        """Return a compact JSON-ready bootstrap summary."""

        return {
            "mean_a": self.mean_a,
            "mean_b": self.mean_b,
            "delta": self.delta,
            "lower": self.lower,
            "upper": self.upper,
            "confidence": self.confidence,
            "n_bootstrap": int(self.bootstrap_estimates.size),
        }


def _as_finite_array(value: ArrayLike, name: str) -> FloatArray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric array") from exc
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _validate_targets_predictions(
    targets: ArrayLike, predictions: ArrayLike
) -> tuple[FloatArray, FloatArray]:
    target_array = _as_finite_array(targets, "targets")
    prediction_array = _as_finite_array(predictions, "predictions")
    if target_array.ndim != 4:
        raise ValueError("targets must have shape [scene, candidate, horizon, component]")
    if prediction_array.ndim != 4:
        raise ValueError(
            "predictions must have shape [scene, candidate, horizon, component]"
        )
    if target_array.shape != prediction_array.shape:
        raise ValueError(
            "targets and predictions must have the same scene, candidate, "
            "horizon, and component counts"
        )
    if any(size == 0 for size in target_array.shape):
        raise ValueError("targets and predictions must have non-empty dimensions")
    return target_array, prediction_array


def position_errors(
    targets: ArrayLike,
    predictions: ArrayLike,
    groups: Mapping[str, PositionGroup | Sequence[int] | slice],
) -> dict[str, FloatArray]:
    """Return a separate Euclidean error series for every named object group.

    A group value may be a :class:`PositionGroup`, a sequence of component
    indices, or a slice.  A PositionGroup can scale components before taking
    the norm, for example to turn metre coordinates into millimetres.
    """

    target_array, prediction_array = _validate_targets_predictions(targets, predictions)
    if not isinstance(groups, Mapping) or not groups:
        raise ValueError("groups must be a non-empty mapping of object names to components")
    component_count = target_array.shape[-1]
    errors: dict[str, FloatArray] = {}
    for name, spec in groups.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("group names must be non-empty strings")
        if isinstance(spec, PositionGroup):
            indices = spec.indices
            raw_scale = spec.scale
        elif isinstance(spec, slice):
            start, stop, step = spec.indices(component_count)
            indices = tuple(range(start, stop, step))
            raw_scale = 1.0
        else:
            try:
                indices = tuple(spec)
            except TypeError as exc:
                raise ValueError(f"group {name!r} must specify component indices") from exc
            raw_scale = 1.0

        if not indices or any(
            isinstance(index, bool) or not isinstance(index, (int, np.integer))
            for index in indices
        ):
            raise ValueError(f"group {name!r} must contain integer component indices")
        indices = tuple(int(index) for index in indices)
        if len(set(indices)) != len(indices):
            raise ValueError(f"group {name!r} contains duplicate component indices")
        if any(index < 0 or index >= component_count for index in indices):
            raise ValueError(f"group {name!r} references a component outside the input shape")

        try:
            scales = np.asarray(raw_scale, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"group {name!r} has invalid component scales") from exc
        if scales.ndim == 0:
            scales = np.full(len(indices), float(scales), dtype=np.float64)
        if scales.shape != (len(indices),):
            raise ValueError(f"group {name!r} needs one scale per component")
        if not np.all(np.isfinite(scales)) or np.any(scales <= 0):
            raise ValueError(f"group {name!r} scales must be finite and positive")

        delta = prediction_array[..., indices] - target_array[..., indices]
        group_error = np.linalg.norm(delta * scales, axis=-1)
        if not np.all(np.isfinite(group_error)):
            raise ValueError(f"group {name!r} produced a non-finite position error")
        errors[name] = group_error
    return errors


def geometry_position_errors(
    targets: ArrayLike, predictions: ArrayLike, geometry: str
) -> dict[str, FloatArray]:
    """Convenience wrapper for the repository's fixed physical geometries."""

    try:
        groups = POSITION_ERROR_GROUPS[geometry.lower()]
    except (AttributeError, KeyError) as exc:
        known = ", ".join(sorted(POSITION_ERROR_GROUPS))
        raise ValueError(f"unknown geometry {geometry!r}; expected one of: {known}") from exc
    return position_errors(targets, predictions, groups)


def _validate_thresholds(thresholds: ArrayLike) -> FloatArray:
    values = _as_finite_array(thresholds, "thresholds")
    if values.ndim != 1 or values.size == 0:
        raise ValueError("thresholds must be a non-empty one-dimensional sequence")
    if np.any(values <= 0):
        raise ValueError("thresholds must be positive")
    if np.any(np.diff(values) <= 0):
        raise ValueError("thresholds must be strictly increasing")
    return values


def _scene_error_arrays(errors: ArrayLike) -> list[FloatArray]:
    """Parse dense [K,C,T] arrays or ragged per-scene [C,T] arrays."""

    if isinstance(errors, (list, tuple)):
        try:
            scenes = [_as_finite_array(scene, "errors") for scene in errors]
        except ValueError:
            raise
        if scenes and all(scene.ndim == 2 for scene in scenes):
            if any(size == 0 for scene in scenes for size in scene.shape):
                raise ValueError("each scene needs non-empty candidate and horizon axes")
            if any(scene.shape[1] != scenes[0].shape[1] for scene in scenes):
                raise ValueError("all scenes must have the same horizon count")
            if any(np.any(scene < 0) for scene in scenes):
                raise ValueError("errors must be non-negative Euclidean distances")
            return scenes

    error_array = _as_finite_array(errors, "errors")
    if error_array.ndim != 3:
        raise ValueError(
            "errors must have shape [scene, candidate, horizon] or be a list "
            "of per-scene [candidate, horizon] arrays"
        )
    if any(size == 0 for size in error_array.shape):
        raise ValueError("errors must have non-empty scene, candidate, and horizon axes")
    if np.any(error_array < 0):
        raise ValueError("errors must be non-negative Euclidean distances")
    return [error_array[index] for index in range(error_array.shape[0])]


def score_errors(errors: ArrayLike, thresholds: ArrayLike) -> AccuracyResult:
    """Score non-negative ``[scene, candidate, horizon]`` position errors.

    Each scene contributes its own mean over candidates and horizons before
    the scene means are macro-averaged.  Every threshold uses a strict
    ``error < threshold`` comparison.  Errors can also be a list of per-scene
    ``[candidate, horizon]`` arrays when candidate counts vary.  The scalar
    ranges from 0 to 100.
    """

    threshold_array = _validate_thresholds(thresholds)
    scenes = _scene_error_arrays(errors)
    if not scenes:
        raise ValueError("errors must contain at least one scene")

    # Reduce candidates within each scene before aggregating. This also keeps
    # equal scene weight when scenes have differing candidate counts.
    scene_curves = [
        np.mean(scene[..., None] < threshold_array, axis=0) * 100.0
        for scene in scenes
    ]  # each [horizon, tolerance]
    tolerance_horizon_macro = np.mean(np.stack(scene_curves, axis=0), axis=0)
    tolerance_curve = np.mean(tolerance_horizon_macro, axis=0)
    horizon_curve = np.mean(tolerance_horizon_macro, axis=-1)
    final_tolerance_curve = tolerance_horizon_macro[-1].copy()
    scene_rmse = []
    for scene in scenes:
        scale = float(np.max(scene))
        scene_rmse.append(
            0.0 if scale == 0.0 else scale * float(np.sqrt(np.mean(np.square(scene / scale))))
        )
    scene_rmse_array = np.asarray(scene_rmse)
    rmse_scale = float(np.max(scene_rmse_array))
    macro_rmse = (
        0.0
        if rmse_scale == 0.0
        else rmse_scale * float(np.sqrt(np.mean(np.square(scene_rmse_array / rmse_scale))))
    )
    return AccuracyResult(
        score=float(np.mean(tolerance_curve)),
        tolerance_curve=tolerance_curve,
        horizon_curve=horizon_curve,
        tolerance_horizon_curve=tolerance_horizon_macro,
        max_horizon_score=float(horizon_curve[-1]),
        max_horizon_tolerance_curve=final_tolerance_curve,
        raw_rmse=macro_rmse,
    )


def clustered_paired_bootstrap_ci(
    scores_a: ArrayLike,
    scores_b: ArrayLike,
    source_clusters: Sequence[object],
    *,
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> PairedBootstrapResult:
    """Bootstrap a paired mean difference by resampling source clusters.

    Inputs begin with the scene axis; any trailing candidate/metric axes are
    averaged within each scene.  Each selected cluster contributes all its
    scenes, and the replicate statistic is the mean over those scenes, so
    individual scenes retain equal weight even when source clusters differ in
    size.  The reported difference is ``scores_a - scores_b``.
    """

    a = _as_finite_array(scores_a, "scores_a")
    b = _as_finite_array(scores_b, "scores_b")
    if a.shape != b.shape or a.ndim < 1 or a.shape[0] == 0:
        raise ValueError("paired score arrays must have the same non-empty scene axis")
    if any(size == 0 for size in a.shape):
        raise ValueError("paired score arrays must have non-empty dimensions")
    try:
        clusters = np.asarray(source_clusters, dtype=object)
    except (TypeError, ValueError) as exc:
        raise ValueError("source_clusters must provide one cluster ID per scene") from exc
    if clusters.ndim != 1 or clusters.shape[0] != a.shape[0]:
        raise ValueError("source_clusters must provide one cluster ID per scene")
    if any(cluster is None for cluster in clusters):
        raise ValueError("source cluster IDs cannot be None")
    try:
        unique_clusters, inverse = np.unique(clusters, return_inverse=True)
    except TypeError as exc:
        raise ValueError("source cluster IDs must have a consistent sortable type") from exc
    if unique_clusters.size == 0:
        raise ValueError("at least one source cluster is required")
    if isinstance(n_bootstrap, bool) or not isinstance(n_bootstrap, (int, np.integer)):
        raise ValueError("n_bootstrap must be a positive integer")
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be a positive integer")
    if not np.isfinite(confidence) or not (0.0 < confidence < 1.0):
        raise ValueError("confidence must be strictly between 0 and 1")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")

    scene_mean_a = np.mean(a, axis=tuple(range(1, a.ndim))) if a.ndim > 1 else a
    scene_mean_b = np.mean(b, axis=tuple(range(1, b.ndim))) if b.ndim > 1 else b
    scene_difference = scene_mean_a - scene_mean_b
    mean_a = float(np.mean(scene_mean_a))
    mean_b = float(np.mean(scene_mean_b))
    estimate = mean_a - mean_b
    rng = np.random.default_rng(int(seed))
    n_clusters = unique_clusters.size
    replicates = np.empty(int(n_bootstrap), dtype=np.float64)
    member_indices = [np.flatnonzero(inverse == cluster_index) for cluster_index in range(n_clusters)]
    for replicate in range(int(n_bootstrap)):
        chosen = rng.integers(0, n_clusters, size=n_clusters)
        sampled_scenes = np.concatenate([member_indices[index] for index in chosen])
        replicates[replicate] = float(np.mean(scene_difference[sampled_scenes]))

    tail = (1.0 - float(confidence)) / 2.0
    lower, upper = np.quantile(replicates, (tail, 1.0 - tail))
    return PairedBootstrapResult(
        estimate=estimate,
        lower=float(lower),
        upper=float(upper),
        bootstrap_estimates=replicates,
        mean_a=mean_a,
        mean_b=mean_b,
        confidence=float(confidence),
    )
