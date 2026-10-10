#!/usr/bin/env python3
"""Cross-fitted physical readout diagnostics for frozen latent predictions.

The input feature files are produced by the history evaluator and contain
``pred[K, C, T, D]`` and ``target[K, C, T, D]`` where K is the condition axis
and C is the candidate axis.  ``target`` is the frozen
encoder representation of the real future.  Physical labels are read from
the already materialized Development panels; this module never simulates an
environment and never updates a world model.

The readout is deliberately small and fixed: source groups are assigned to
three deterministic folds, a standardized float64 ridge readout is fitted on
true target latents from the other folds, and the held-out fold is scored on
both true and predicted latents.  A fixed random projection is used only when
the latent dimension is larger than the configured cap (the DINO 4x4
patch-pool representation is the intended case).  No hyperparameter is
chosen from held-out errors.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


DEFAULT_FOLDS = 3
DEFAULT_MAX_SAMPLES_PER_SOURCE = 32
DEFAULT_RIDGE_ALPHA = 1.0e-3
DEFAULT_PROJECTION_DIM = 512
DEFAULT_PROJECTION_SEED = 20261007
DEFAULT_BOOTSTRAP_REPS = 1000
FEATURE_SCALE_EPS = 1.0e-12
PHYSICAL_VARIANCE_EPS = 1.0e-12
MAX_PRIMAL_FEATURES = 2048


TASK_ALIASES = {
    "speed": "tworoom",
    "door": "tworoom",
    "portal": "tworoom",
    "portal_exit": "tworoom",
    "action_delay": "tworoom",
    "two_room": "tworoom",
    "tworoom": "tworoom",
    "action_strength": "pusht",
    "contact_friction": "pusht",
    "motion_damping": "pusht",
    "pusht": "pusht",
    "reacher": "mass",
    "reacher_arm_mass": "mass",
    "robot_arm_mass": "mass",
    "mass": "mass",
    "cube": "cube",
    "cube_carry": "cube",
    "cube_gripper_carry": "cube",
}


@dataclass(frozen=True)
class TargetSpec:
    """Physical target definition used by the diagnostic."""

    task: str
    names: tuple[str, ...]
    units: str
    description: str


@dataclass(frozen=True)
class FeatureScene:
    """One query scene joined to its frozen panel physical labels."""

    scene_id: str
    source_group: str
    pred: np.ndarray
    target: np.ndarray
    physical: np.ndarray
    panel_path: str = ""
    # When the loader preprojects a wide representation, retain its original
    # width so the receipt still records what was projected.
    original_latent_dim: int | None = None
    projection_seed: int | None = None


def canonical_task(task: str) -> str:
    """Return the task family used for physical-state extraction."""

    key = str(task).strip().lower().replace("-", "_")
    try:
        return TASK_ALIASES[key]
    except KeyError as exc:
        supported = ", ".join(sorted(TASK_ALIASES))
        raise ValueError(f"Unsupported task {task!r}; expected one of {supported}") from exc


def target_spec(task: str) -> TargetSpec:
    family = canonical_task(task)
    if family == "tworoom":
        return TargetSpec(
            family,
            ("x_px", "y_px"),
            "px",
            "TwoRoom agent position; no velocity or other hidden state.",
        )
    if family == "pusht":
        return TargetSpec(
            family,
            (
                "agent_x_px",
                "agent_y_px",
                "block_x_px",
                "block_y_px",
                "block_angle_sin_px_equivalent",
                "block_angle_cos_px_equivalent",
            ),
            "px-equivalent",
            "PushT simulator agent/block positions and 40*sin/cos(block angle); positions may leave the rendered canvas.",
        )
    if family == "mass":
        return TargetSpec(
            family,
            ("finger_x_mm", "finger_y_mm"),
            "mm",
            "Reacher finger position only; qpos/qvel and terminal speed are excluded.",
        )
    return TargetSpec(
        family,
        (
            "effector_x_mm",
            "effector_y_mm",
            "effector_z_mm",
            "cube_x_mm",
            "cube_y_mm",
            "cube_z_mm",
        ),
        "mm",
        "Cube effector state[0,1,6] and cube state[2:5], scaled to millimetres.",
    )


def _as_float_array(value: Any, *, name: str, ndim: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def _as_feature_array(value: Any, *, name: str) -> np.ndarray:
    """Keep stored features float32; cast only the active fit batch to float64."""

    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 4:
        raise ValueError(f"{name} must have shape [K,C,T,D], got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return np.ascontiguousarray(array)


def physical_targets_from_panel(panel: Mapping[str, Any], task: str) -> np.ndarray:
    """Extract selected physical future targets in shared ``[K,C,T,P]`` order."""

    family = canonical_task(task)
    if family == "mass":
        if "finger_positions" not in panel:
            raise KeyError("Mass panel is missing finger_positions")
        return _as_float_array(panel["finger_positions"], name="finger_positions", ndim=4) * 1000.0

    if "future_states" not in panel:
        raise KeyError("Panel is missing future_states")
    states = _as_float_array(panel["future_states"], name="future_states", ndim=4)

    if family == "tworoom":
        if states.shape[-1] < 2:
            raise ValueError(f"TwoRoom future_states has invalid width {states.shape[-1]}")
        return states[..., :2]

    if family == "cube":
        if states.shape[-1] < 7:
            raise ValueError(f"Cube future_states must have width >=7, got {states.shape[-1]}")
        # The panel's seven visible values are [effector x,y, cube x,y,z,
        # gripper opening, effector z].  Keep positions only.
        effector = states[..., [0, 1, 6]]
        cube = states[..., 2:5]
        return np.concatenate([effector, cube], axis=-1) * 1000.0

    # The action-strength panel stores the public 7-vector with the block
    # angle at index 4.  Contact-friction and motion-damping store the full
    # 12-vector, where the block position is [6:8] and angle is index 10.
    if states.shape[-1] == 7:
        positions = states[..., :4]
        angle = states[..., 4]
    elif states.shape[-1] >= 12:
        positions = np.concatenate([states[..., 0:2], states[..., 6:8]], axis=-1)
        angle = states[..., 10]
    else:
        raise ValueError(
            "PushT future_states must have width 7 or at least 12, "
            f"got {states.shape[-1]}"
        )
    return np.concatenate(
        [
            positions,
            (40.0 * np.sin(angle))[..., None],
            (40.0 * np.cos(angle))[..., None],
        ],
        axis=-1,
    )


def align_physical_targets(
    physical: np.ndarray,
    feature_shape: Sequence[int],
) -> np.ndarray:
    """Validate physical labels in the shared ``[K_condition,C_candidate,T,P]`` order."""

    value = _as_float_array(physical, name="physical targets", ndim=4)
    if len(feature_shape) != 4:
        raise ValueError(f"feature shape must be [K,C,T,D], got {tuple(feature_shape)}")
    k, c, t, _ = map(int, feature_shape)
    if value.shape[:3] != (k, c, t):
        raise ValueError(
            "physical and feature axes disagree: "
            f"physical={value.shape[:3]}, expected condition-first feature={(k, c, t)}"
        )
    return np.ascontiguousarray(value)


def _decode_scalar(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return _decode_scalar(value.item())
        if value.size == 1:
            return _decode_scalar(value.reshape(-1)[0])
    return str(value)


def _manifest_index(panel_root: Path) -> tuple[dict[str, Path], dict[str, dict[str, Any]]]:
    """Index manifest scene IDs and panel paths, when a manifest is present."""

    manifest_path = panel_root / "manifest.json"
    if not manifest_path.exists():
        return {}, {}
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    paths: dict[str, Path] = {}
    entries: dict[str, dict[str, Any]] = {}
    for row in payload.get("scenes", []):
        if not isinstance(row, Mapping):
            continue
        scene = str(row.get("scene_id", ""))
        rel = row.get("path")
        if not scene or not isinstance(rel, str):
            continue
        path = panel_root / rel
        paths[scene] = path
        paths.setdefault(Path(rel).stem, path)
        entries[scene] = dict(row)
        entries.setdefault(Path(rel).stem, dict(row))
    # A few older manifests expose only a files map.  It still gives a useful
    # path index; absent source metadata is intentionally handled conservatively
    # as one group per scene below.
    for rel in payload.get("files", {}):
        if isinstance(rel, str):
            paths.setdefault(Path(rel).stem, panel_root / rel)
    return paths, entries


def _resolve_panel_root(panel_dir: Path, task: str) -> Path:
    family = canonical_task(task)
    aliases = [task, family]
    for alias in aliases:
        candidate = panel_dir / alias
        if candidate.is_dir():
            return candidate
    return panel_dir


def _feature_scene_id(path: Path, archive: Mapping[str, Any]) -> str:
    for key in ("scene_id", "query_id", "id"):
        if key in archive:
            return _decode_scalar(archive[key])
    prefix = "features_"
    return path.stem[len(prefix) :] if path.stem.startswith(prefix) else path.stem


def _metadata_group(
    *,
    scene_id: str,
    archive: Mapping[str, Any],
    panel: Mapping[str, Any],
    manifest_entry: Mapping[str, Any] | None,
) -> str:
    """Return the most specific audited source group available."""

    candidates: list[tuple[str, Any]] = []
    if manifest_entry:
        candidates.extend(
            (key, manifest_entry[key])
            for key in (
                "bootstrap_cluster",
                "source_group",
                "source_sha256",
                "source_index",
                "source_id",
                "pair_id",
            )
            if key in manifest_entry
        )
    candidates.extend(
        (key, archive[key])
        for key in ("source_group", "bootstrap_cluster", "source_sha256", "source_id")
        if key in archive
    )
    candidates.extend(
        (key, panel[key])
        for key in ("source_group", "bootstrap_cluster", "source_sha256", "pair_id")
        if key in panel
    )
    for key, value in candidates:
        text = _decode_scalar(value)
        if text and text.lower() not in {"none", "nan"}:
            return f"{key}:{text}"
    # Legacy fallback only: scene identifiers do not establish independence.
    # Callers must provide source groups when several scenes share a source.
    return f"scene:{scene_id}"


def _find_panel_path(
    *,
    scene_id: str,
    archive: Mapping[str, Any],
    panel_root: Path,
    manifest_paths: Mapping[str, Path],
) -> Path:
    for key in ("panel_path", "panel", "source_panel"):
        if key in archive:
            value = Path(_decode_scalar(archive[key]))
            candidate = value if value.is_absolute() else panel_root / value
            if candidate.exists():
                return candidate
    if scene_id in manifest_paths and manifest_paths[scene_id].exists():
        return manifest_paths[scene_id]
    direct = panel_root / f"{scene_id}.npz"
    if direct.exists():
        return direct
    matches = sorted(panel_root.rglob(f"{scene_id}.npz"))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(
        f"Cannot resolve panel for feature scene {scene_id!r} under {panel_root}; "
        "provide a manifest scene_id/path or panel_path in the feature archive"
    )


def load_feature_scenes(
    features_dir: str | Path,
    panel_dir: str | Path,
    task: str,
    *,
    projection_dim: int | None = None,
    projection_seed: int = DEFAULT_PROJECTION_SEED,
) -> list[FeatureScene]:
    """Load and join feature files to panels, optionally preprojecting wide inputs.

    Preprojection is done file-by-file in float32 output so a full DINO run
    does not retain the original 6144-dimensional arrays in memory.  The
    actual ridge fit still converts only its active batch to float64.
    """

    features_path = Path(features_dir)
    panel_root = _resolve_panel_root(Path(panel_dir), task)
    if features_path.is_file():
        feature_files = [features_path]
    else:
        # Merged history results expose top-level hard links while retaining
        # the original shard copies.  Prefer the merge view; recurse only for
        # an unmerged result directory so one query cannot be counted twice.
        feature_files = sorted(features_path.glob("features_*.npz"))
        if not feature_files:
            feature_files = sorted(features_path.rglob("features_*.npz"))
    if not feature_files:
        raise FileNotFoundError(f"No features_*.npz files found under {features_path}")

    manifest_paths, manifest_entries = _manifest_index(panel_root)
    records: list[FeatureScene] = []
    seen: set[str] = set()
    preprojector: FixedRandomProjector | None = None
    for feature_file in feature_files:
        with np.load(feature_file, allow_pickle=False) as archive:
            if "pred" not in archive or "target" not in archive:
                raise KeyError(f"{feature_file} must contain pred and target arrays")
            pred = _as_feature_array(archive["pred"], name=f"{feature_file}:pred")
            target = _as_feature_array(archive["target"], name=f"{feature_file}:target")
            if pred.shape != target.shape:
                raise ValueError(f"{feature_file}: pred and target shapes differ: {pred.shape} vs {target.shape}")
            original_latent_dim: int | None = None
            stored_projection_seed: int | None = None
            if projection_dim is not None and int(projection_dim) > 0 and pred.shape[-1] > int(projection_dim):
                if preprojector is None:
                    preprojector = FixedRandomProjector(
                        int(pred.shape[-1]), int(projection_dim), int(projection_seed)
                    )
                if pred.shape[-1] != preprojector.input_dim:
                    raise ValueError("feature files in one run have inconsistent latent widths")
                original_latent_dim = int(pred.shape[-1])
                pred_shape = pred.shape
                target_shape = target.shape
                pred = preprojector.transform(pred).reshape(
                    *pred_shape[:3], preprojector.output_dim
                ).astype(np.float32)
                target = preprojector.transform(target).reshape(
                    *target_shape[:3], preprojector.output_dim
                ).astype(np.float32)
                stored_projection_seed = int(projection_seed)
            # The feature filename can be a compact panel-path alias (for
            # example ``pair_0000``) while the manifest carries the formal
            # query ID.  Keep the alias for panel lookup, then expose the
            # manifest ID in the joined record so downstream coverage joins
            # use the same namespace as the panels.
            feature_scene_id = _feature_scene_id(feature_file, archive)
            panel_file = _find_panel_path(
                scene_id=feature_scene_id,
                archive=archive,
                panel_root=panel_root,
                manifest_paths=manifest_paths,
            )
            manifest_entry = manifest_entries.get(feature_scene_id)
            if manifest_entry is None:
                manifest_entry = manifest_entries.get(panel_file.stem)
            scene_id = feature_scene_id
            if manifest_entry is not None and manifest_entry.get("scene_id"):
                scene_id = _decode_scalar(manifest_entry["scene_id"])
            if scene_id in seen:
                raise ValueError(f"Duplicate feature scene id {scene_id!r}; point features_dir at one model")
            seen.add(scene_id)
            with np.load(panel_file, allow_pickle=False) as panel:
                physical_raw = physical_targets_from_panel(panel, task)
                physical = align_physical_targets(physical_raw, pred.shape)
                group = _metadata_group(
                    scene_id=scene_id,
                    archive=archive,
                    panel=panel,
                    manifest_entry=manifest_entry,
                )
            records.append(
                FeatureScene(
                    scene_id=scene_id,
                    source_group=group,
                    pred=np.ascontiguousarray(pred),
                    target=np.ascontiguousarray(target),
                    physical=np.ascontiguousarray(physical, dtype=np.float32),
                    panel_path=str(panel_file),
                    original_latent_dim=original_latent_dim,
                    projection_seed=stored_projection_seed,
                )
            )
    return sorted(records, key=lambda row: row.scene_id)


def assign_source_folds(
    source_groups: Iterable[str], *, n_folds: int = DEFAULT_FOLDS, seed: int = 0
) -> dict[str, int]:
    """Assign whole source groups to deterministic, balanced folds."""

    if int(n_folds) < 2:
        raise ValueError("source-group crossfit requires at least two folds")
    groups = sorted({str(group) for group in source_groups})
    if len(groups) < int(n_folds):
        raise ValueError(f"need at least {n_folds} source groups, got {len(groups)}")
    ordered = sorted(
        groups,
        key=lambda group: hashlib.sha256(f"{int(seed)}\0{group}".encode()).hexdigest(),
    )
    return {group: index % int(n_folds) for index, group in enumerate(ordered)}


def uniform_row_indices(count: int, maximum: int) -> np.ndarray:
    """Select at most ``maximum`` rows at fixed, result-independent positions."""

    count = int(count)
    maximum = int(maximum)
    if count < 0 or maximum <= 0:
        raise ValueError("count must be non-negative and maximum must be positive")
    if count <= maximum:
        return np.arange(count, dtype=np.int64)
    # Rounding a linspace is deterministic and covers both endpoints.  A
    # strictly increasing choice avoids duplicate samples for normal inputs.
    values = np.rint(np.linspace(0, count - 1, maximum)).astype(np.int64)
    return np.unique(values)


@dataclass
class FixedRandomProjector:
    """A deterministic sparse random projection for very wide features.

    Each input coordinate contributes to three fixed output buckets with a
    random sign.  Keeping the sparse structure as bucket lists avoids ever
    constructing a 6144 by 6144 Gram matrix (and avoids materializing a dense
    6144 by 512 projection matrix).
    """

    input_dim: int
    requested_dim: int
    seed: int = DEFAULT_PROJECTION_SEED

    def __post_init__(self) -> None:
        self.input_dim = int(self.input_dim)
        self.requested_dim = int(self.requested_dim)
        if self.input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if self.requested_dim < 0:
            raise ValueError("requested projection dimension must be non-negative")
        self.output_dim = self.input_dim if self.requested_dim == 0 else min(self.input_dim, self.requested_dim)
        self.projected = self.output_dim < self.input_dim
        self.bucket_columns: tuple[np.ndarray, ...] = ()
        self.bucket_signs: tuple[np.ndarray, ...] = ()
        self.nonzeros_per_input = 0
        if self.projected:
            rng = np.random.default_rng(int(self.seed))
            self.nonzeros_per_input = min(3, self.output_dim)
            buckets = rng.integers(
                0,
                self.output_dim,
                size=(self.input_dim, self.nonzeros_per_input),
                dtype=np.int64,
            )
            signs = rng.choice(
                np.asarray([-1.0, 1.0], dtype=np.float64),
                size=(self.input_dim, self.nonzeros_per_input),
            )
            columns: list[list[int]] = [[] for _ in range(self.output_dim)]
            column_signs: list[list[float]] = [[] for _ in range(self.output_dim)]
            for input_index in range(self.input_dim):
                for nonzero_index in range(self.nonzeros_per_input):
                    bucket = int(buckets[input_index, nonzero_index])
                    columns[bucket].append(input_index)
                    column_signs[bucket].append(float(signs[input_index, nonzero_index]))
            self.bucket_columns = tuple(np.asarray(value, dtype=np.int64) for value in columns)
            self.bucket_signs = tuple(np.asarray(value, dtype=np.float64) for value in column_signs)

    def transform(self, values: np.ndarray) -> np.ndarray:
        value = _as_float_array(values, name="latent features")
        if value.shape[-1] != self.input_dim:
            raise ValueError(f"latent width {value.shape[-1]} does not match projector width {self.input_dim}")
        flat = value.reshape(-1, self.input_dim)
        if not self.projected:
            result = flat
        else:
            result = np.zeros((len(flat), self.output_dim), dtype=np.float64)
            scale = math.sqrt(float(self.nonzeros_per_input))
            for bucket, (columns, signs) in enumerate(zip(self.bucket_columns, self.bucket_signs)):
                if len(columns):
                    result[:, bucket] = (flat[:, columns] @ signs) / scale
        return np.ascontiguousarray(result)


def fit_ridge_readout(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    alpha: float = DEFAULT_RIDGE_ALPHA,
    feature_scale_eps: float = FEATURE_SCALE_EPS,
) -> dict[str, np.ndarray | float | int]:
    """Fit an intercept ridge readout on standardized float64 features.

    ``alpha`` is dimensionless here: it is added to the standardized-feature
    Gram matrix after that matrix is divided by the number of sampled rows;
    the intercept is not regularized.
    """

    x = _as_float_array(features, name="readout features", ndim=2)
    y = _as_float_array(targets, name="readout targets", ndim=2)
    if x.shape[0] != y.shape[0] or x.shape[0] == 0:
        raise ValueError(f"readout row mismatch or empty data: {x.shape} vs {y.shape}")
    if float(alpha) < 0.0 or not np.isfinite(alpha):
        raise ValueError("ridge alpha must be finite and non-negative")
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale = np.where(scale < float(feature_scale_eps), 1.0, scale)
    standardized = (x - mean) / scale
    target_mean = y.mean(axis=0)
    centered_y = y - target_mean
    count = float(x.shape[0])
    if standardized.shape[1] > MAX_PRIMAL_FEATURES:
        # Equivalent dual form for the same objective.  The normal path uses
        # the fixed sparse projection above, but this guard prevents an
        # accidental 6144 by 6144 primal Gram matrix for direct callers.
        dual = standardized @ standardized.T
        dual.flat[:: dual.shape[0] + 1] += float(alpha) * count
        try:
            dual_solution = np.linalg.solve(dual, centered_y)
        except np.linalg.LinAlgError:
            dual_solution = np.linalg.lstsq(dual, centered_y, rcond=None)[0]
        coefficient = standardized.T @ dual_solution
        solver = "dual_gram"
    else:
        gram = (standardized.T @ standardized) / count
        rhs = (standardized.T @ centered_y) / count
        gram.flat[:: gram.shape[0] + 1] += float(alpha)
        try:
            coefficient = np.linalg.solve(gram, rhs)
        except np.linalg.LinAlgError:
            coefficient = np.linalg.lstsq(gram, rhs, rcond=None)[0]
        solver = "primal_gram"
    return {
        "feature_mean": mean,
        "feature_scale": scale,
        "target_mean": target_mean,
        "coefficient": np.asarray(coefficient, dtype=np.float64),
        "alpha": float(alpha),
        "feature_scale_eps": float(feature_scale_eps),
        "n_rows": int(x.shape[0]),
        "solver": solver,
    }


def apply_ridge_readout(model: Mapping[str, Any], features: np.ndarray) -> np.ndarray:
    x = _as_float_array(features, name="readout features", ndim=2)
    mean = np.asarray(model["feature_mean"], dtype=np.float64)
    scale = np.asarray(model["feature_scale"], dtype=np.float64)
    target_mean = np.asarray(model["target_mean"], dtype=np.float64)
    coefficient = np.asarray(model["coefficient"], dtype=np.float64)
    if x.shape[1] != mean.shape[0] or coefficient.shape[0] != x.shape[1]:
        raise ValueError("readout feature width does not match fitted model")
    return ((x - mean) / scale) @ coefficient + target_mean


# Short aliases are convenient for small callers and preserve discoverability.
fit_ridge = fit_ridge_readout
predict_ridge = apply_ridge_readout


def _record_rows(record: FeatureScene, projector: FixedRandomProjector) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pred = projector.transform(record.pred)
    target = projector.transform(record.target)
    physical = record.physical.reshape(-1, record.physical.shape[-1])
    return pred, target, physical


def _sample_training_rows(
    records: Sequence[FeatureScene],
    projector: FixedRandomProjector,
    *,
    maximum_per_source: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    by_group: dict[str, list[FeatureScene]] = {}
    for record in records:
        by_group.setdefault(record.source_group, []).append(record)
    feature_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    counts: dict[str, int] = {}
    for group in sorted(by_group):
        rows_x: list[np.ndarray] = []
        rows_y: list[np.ndarray] = []
        for record in sorted(by_group[group], key=lambda row: row.scene_id):
            _, true_latent, physical = _record_rows(record, projector)
            rows_x.append(true_latent)
            rows_y.append(physical)
        group_x = np.concatenate(rows_x, axis=0)
        group_y = np.concatenate(rows_y, axis=0)
        chosen = uniform_row_indices(len(group_x), maximum_per_source)
        feature_rows.append(group_x[chosen])
        target_rows.append(group_y[chosen])
        counts[group] = int(len(chosen))
    if not feature_rows:
        raise ValueError("crossfit fold has no training source groups")
    return np.concatenate(feature_rows), np.concatenate(target_rows), counts


def _physical_error(
    prediction: np.ndarray,
    truth: np.ndarray,
    *,
    normalization_truth: np.ndarray,
) -> dict[str, Any]:
    prediction = _as_float_array(prediction, name="physical prediction", ndim=4)
    truth = _as_float_array(truth, name="physical truth", ndim=4)
    normalization_truth = _as_float_array(normalization_truth, name="normalization truth", ndim=4)
    if prediction.shape != truth.shape:
        raise ValueError(f"physical prediction/truth shape mismatch: {prediction.shape} vs {truth.shape}")
    difference = prediction - truth
    mse = float(np.mean(np.square(difference)))
    # K is the condition axis.  The denominator is the true within-query
    # condition variance, averaged over candidates, horizons, and dimensions.
    variance = float(np.mean(np.var(normalization_truth, axis=0)))
    zero_variance = bool(variance <= PHYSICAL_VARIANCE_EPS)
    normalized = None if zero_variance else float(math.sqrt(mse / variance))
    return {
        "query_mse": mse,
        "numerator": mse,
        "normalization_denominator": variance,
        "rmse": float(math.sqrt(mse)),
        "mse": mse,
        "rmse_by_dimension": np.sqrt(np.mean(np.square(difference), axis=(0, 1, 2))).tolist(),
        "physical_variance_across_conditions": variance,
        "normalized_rmse": normalized,
        "normalized_denominator_zero": zero_variance,
        "comparison_count": int(np.prod(prediction.shape[:-1])),
    }


def _mismatched_physical_error(
    prediction: np.ndarray,
    truth: np.ndarray,
    *,
    normalization_truth: np.ndarray,
) -> dict[str, Any]:
    """Compare each condition prediction with the mean error to all others.

    The candidate and horizon axes stay paired.  There is no cross-scene
    fallback: with one condition there is no valid condition mismatch and the
    query is explicitly excluded from the mismatch aggregate.
    """

    prediction = _as_float_array(prediction, name="mismatched physical prediction", ndim=4)
    truth = _as_float_array(truth, name="mismatched physical truth", ndim=4)
    normalization_truth = _as_float_array(
        normalization_truth, name="mismatched normalization truth", ndim=4
    )
    if prediction.shape != truth.shape:
        raise ValueError(f"mismatched prediction/truth shape mismatch: {prediction.shape} vs {truth.shape}")
    condition_count = int(prediction.shape[0])
    variance = float(np.mean(np.var(normalization_truth, axis=0)))
    zero_variance = bool(variance <= PHYSICAL_VARIANCE_EPS)
    if condition_count <= 1:
        return {
            "query_mse": None,
            "numerator": None,
            "normalization_denominator": variance,
            "rmse": None,
            "mse": None,
            "rmse_by_dimension": None,
            "physical_variance_across_conditions": variance,
            "normalized_rmse": None,
            "normalized_denominator_zero": zero_variance,
            "comparison_count": 0,
            "undefined_condition_mismatch": True,
        }
    pair_difference = prediction[:, None] - truth[None, :]
    off_diagonal = ~np.eye(condition_count, dtype=bool)
    difference = pair_difference[off_diagonal]
    mse = float(np.mean(np.square(difference)))
    normalized = None if zero_variance else float(math.sqrt(mse / variance))
    return {
        "query_mse": mse,
        "numerator": mse,
        "normalization_denominator": variance,
        "rmse": float(math.sqrt(mse)),
        "mse": mse,
        "rmse_by_dimension": np.sqrt(np.mean(np.square(difference), axis=(0, 1, 2))).tolist(),
        "physical_variance_across_conditions": variance,
        "normalized_rmse": normalized,
        "normalized_denominator_zero": zero_variance,
        "comparison_count": int(condition_count * (condition_count - 1) * np.prod(prediction.shape[1:-1])),
        "undefined_condition_mismatch": False,
    }


def _bootstrap_group_indices(
    groups: Sequence[str], *, bootstrap_reps: int, seed: int
) -> np.ndarray:
    if not groups or int(bootstrap_reps) <= 0:
        return np.empty((0, 0), dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    return rng.integers(0, len(groups), size=(int(bootstrap_reps), len(groups)), dtype=np.int64)


def _aggregate_metric(
    metric_rows: Sequence[dict[str, Any]], *, normalized: bool
) -> float | None:
    aggregate_mse = _aggregate_mse(metric_rows, normalized=normalized)
    if aggregate_mse is None:
        return None
    return float(math.sqrt(aggregate_mse))


def _aggregate_mse(
    metric_rows: Sequence[dict[str, Any]], *, normalized: bool
) -> float | None:
    """Aggregate query MSE before taking a square root.

    Raw aggregation is ``sum(query_mse) / n_queries``.  Normalized
    aggregation is ``sum(query_mse) / sum(physical_variance)``.  Keeping this
    operation separate makes paired differences use the same point estimate
    as the standalone RMSE summaries.
    """

    valid = [row for row in metric_rows if row.get("query_mse") is not None]
    if not valid:
        return None
    numerator = float(sum(float(row["query_mse"]) for row in valid))
    if normalized:
        denominator = float(sum(float(row["normalization_denominator"]) for row in valid))
        if denominator <= PHYSICAL_VARIANCE_EPS:
            return None
    else:
        denominator = float(len(valid))
    return float(numerator / denominator)


def _summary(
    rows: Sequence[dict[str, Any]],
    key: str,
    *,
    normalized: bool,
    bootstrap_indices: np.ndarray,
    bootstrap_groups: Sequence[str],
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    metric_rows: list[dict[str, Any]] = []
    zero_variance_rows = 0
    undefined_rows = 0
    for row in rows:
        metric = row[key]
        if bool(metric.get("normalized_denominator_zero", False)):
            zero_variance_rows += 1
        if bool(metric.get("undefined_condition_mismatch", False)):
            undefined_rows += 1
        if metric.get("query_mse") is not None:
            metric_rows.append(metric)
            grouped.setdefault(str(row["source_group"]), []).append(metric)
    group_values = {
        group: _aggregate_metric(values, normalized=normalized)
        for group, values in sorted(grouped.items())
    }
    group_values = {group: value for group, value in group_values.items() if value is not None}
    result: dict[str, Any] = {
        "point_estimate": _aggregate_metric(metric_rows, normalized=normalized),
        "point_estimate_mse": _aggregate_mse(metric_rows, normalized=normalized),
        "source_group_means": group_values,
        "n_source_groups": int(len(group_values)),
        "n_queries": int(len(metric_rows)),
        "n_zero_physical_variance_queries": int(zero_variance_rows),
        "n_undefined_condition_mismatch_queries": int(undefined_rows),
        "query_weighting": "equal_query_for_raw_rmse; summed_query_numerator_and_denominator_for_normalized_rmse",
        "numerator_sum_query_mse": float(sum(float(row["query_mse"]) for row in metric_rows)) if metric_rows else 0.0,
        "denominator_sum": (
            float(sum(float(row["normalization_denominator"]) for row in metric_rows))
            if normalized
            else float(len(metric_rows))
        ),
        "bootstrap_replicates": int(len(bootstrap_indices)),
        "bootstrap_ci95": None,
    }
    if len(bootstrap_indices) and bootstrap_groups:
        group_rows = {group: grouped.get(group, []) for group in bootstrap_groups}
        samples: list[float] = []
        for sample in bootstrap_indices:
            selected: list[dict[str, Any]] = []
            for index in sample:
                selected.extend(group_rows[bootstrap_groups[int(index)]])
            value = _aggregate_metric(selected, normalized=normalized)
            if value is not None:
                samples.append(value)
        if samples:
            result["bootstrap_ci95"] = [
                float(np.percentile(samples, 2.5)),
                float(np.percentile(samples, 97.5)),
            ]
    return result


def _paired_gain_summary(
    rows: Sequence[dict[str, Any]],
    left_key: str,
    right_key: str,
    *,
    normalized: bool,
    bootstrap_indices: np.ndarray,
    bootstrap_groups: Sequence[str],
) -> dict[str, Any]:
    grouped: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for row in rows:
        left, right = row[left_key], row[right_key]
        if left.get("query_mse") is None or right.get("query_mse") is None:
            continue
        pair = (left, right)
        pairs.append(pair)
        grouped.setdefault(str(row["source_group"]), []).append(pair)

    def aggregate(values: Sequence[tuple[dict[str, Any], dict[str, Any]]]) -> tuple[float | None, float | None]:
        left_value = _aggregate_metric([pair[0] for pair in values], normalized=normalized)
        right_value = _aggregate_metric([pair[1] for pair in values], normalized=normalized)
        if left_value is None or right_value is None:
            return None, None
        left_mse = _aggregate_mse([pair[0] for pair in values], normalized=normalized)
        right_mse = _aggregate_mse([pair[1] for pair in values], normalized=normalized)
        if left_mse is None or right_mse is None:
            return None, None
        # This is an aggregate MSE difference, rather than a mean of
        # per-query differences.  It therefore has the same denominator as
        # the corresponding standalone RMSE summary.
        mse_gain = float(left_mse - right_mse)
        return float(left_value - right_value), mse_gain

    point_rmse, point_mse = aggregate(pairs)
    result: dict[str, Any] = {
        "left_minus_right_rmse": point_rmse,
        "left_minus_right_mse": point_mse,
        # Keep the original key as a compatibility alias.  Its value is now
        # the aggregate MSE difference, including the normalized denominator
        # when ``normalized=True``.
        "left_minus_right_query_mse": point_mse,
        "left": left_key,
        "right": right_key,
        "mse_aggregation": (
            "sum_query_mse_over_n_queries"
            if not normalized
            else "sum_query_mse_over_sum_condition_variance"
        ),
        "mse_units": "dimensionless" if normalized else "target_units^2",
        "rmse_units": "dimensionless" if normalized else "target_units",
        "same_source_bootstrap_draws_for_left_and_right": True,
        "bootstrap_replicates": int(len(bootstrap_indices)),
        "bootstrap_ci95_rmse": None,
        "bootstrap_ci95_mse": None,
        "bootstrap_ci95_query_mse": None,
    }
    if len(bootstrap_indices) and bootstrap_groups:
        group_rows = {group: grouped.get(group, []) for group in bootstrap_groups}
        rmse_samples: list[float] = []
        mse_samples: list[float] = []
        for sample in bootstrap_indices:
            selected: list[tuple[dict[str, Any], dict[str, Any]]] = []
            for index in sample:
                selected.extend(group_rows[bootstrap_groups[int(index)]])
            rmse_value, mse_value = aggregate(selected)
            if rmse_value is not None:
                rmse_samples.append(rmse_value)
            if mse_value is not None:
                mse_samples.append(mse_value)
        if rmse_samples:
            result["bootstrap_ci95_rmse"] = [
                float(np.percentile(rmse_samples, 2.5)),
                float(np.percentile(rmse_samples, 97.5)),
            ]
        if mse_samples:
            ci = [
                float(np.percentile(mse_samples, 2.5)),
                float(np.percentile(mse_samples, 97.5)),
            ]
            result["bootstrap_ci95_mse"] = ci
            result["bootstrap_ci95_query_mse"] = ci
    return result


def _metric_for_upgrade(metric: Mapping[str, Any], *, mismatch_default: bool) -> dict[str, Any]:
    """Normalize one old query metric without touching latent/readout data."""

    value = dict(metric)
    query_mse = value.get("query_mse", value.get("mse"))
    if query_mse is None and value.get("rmse") is not None:
        query_mse = float(value["rmse"]) ** 2
    if query_mse is not None:
        query_mse = float(query_mse)
        if not np.isfinite(query_mse) or query_mse < 0.0:
            raise ValueError("old query metric has an invalid query_mse")
    value["query_mse"] = query_mse
    value["mse"] = query_mse
    value["numerator"] = query_mse
    value["rmse"] = None if query_mse is None else float(math.sqrt(query_mse))

    denominator = value.get(
        "normalization_denominator",
        value.get("physical_variance_across_conditions", 0.0),
    )
    if denominator is None:
        denominator = 0.0
    denominator = float(denominator)
    if not np.isfinite(denominator) or denominator < 0.0:
        raise ValueError("old query metric has an invalid normalization denominator")
    value["normalization_denominator"] = denominator
    value["physical_variance_across_conditions"] = denominator
    value["normalized_denominator_zero"] = bool(denominator <= PHYSICAL_VARIANCE_EPS)
    if query_mse is None or denominator <= PHYSICAL_VARIANCE_EPS:
        value["normalized_rmse"] = None
    else:
        value["normalized_rmse"] = float(math.sqrt(query_mse / denominator))
    value["undefined_condition_mismatch"] = bool(
        value.get("undefined_condition_mismatch", mismatch_default and query_mse is None)
    )
    return value


def _metric_units_for_target(units: str) -> dict[str, str]:
    target_units = str(units or "target_units")
    return {
        "physical_mse": f"{target_units}^2",
        "physical_rmse": target_units,
        "normalized_mse": "dimensionless",
        "normalized_rmse": "dimensionless",
    }


def _infer_bootstrap_reps(result: Mapping[str, Any]) -> int:
    """Reuse the old receipt's replicate count when one was recorded."""

    for key in ("paired_gain", "paired_calibration_gap", "paired_history_gain"):
        section = result.get(key)
        if not isinstance(section, Mapping):
            continue
        for value in section.values():
            if isinstance(value, Mapping) and "bootstrap_replicates" in value:
                try:
                    reps = int(value["bootstrap_replicates"])
                except (TypeError, ValueError):
                    continue
                if reps >= 0:
                    return reps
    return DEFAULT_BOOTSTRAP_REPS


def upgrade_physical_readout_result(
    result: Mapping[str, Any] | str | Path,
    *,
    bootstrap_reps: int | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Upgrade an old JSON result using only its stored ``query_metrics``.

    The operation is deliberately a receipt/statistics migration: it does
    not load features, fit a readout, or rerun a world model.  The old
    ``paired_gain`` is interpreted according to the historical implementation
    as predicted-matched minus oracle (the calibration gap).  A second field
    is computed with the same source-group bootstrap draws in the direction
    mismatched minus matched (the history gain).

    ``result`` may be an already loaded mapping or a JSON path.  The returned
    mapping is a deep copy and retains the original query metrics.
    """

    if isinstance(result, (str, Path)):
        payload: Any = json.loads(Path(result).read_text(encoding="utf-8"))
    else:
        payload = result
    if not isinstance(payload, Mapping):
        raise TypeError("physical readout result must be a mapping or JSON path")
    upgraded = copy.deepcopy(dict(payload))
    raw_rows = upgraded.get("query_metrics")
    if not isinstance(raw_rows, Sequence) or isinstance(raw_rows, (str, bytes)) or not raw_rows:
        raise ValueError("old physical readout result must contain non-empty query_metrics")

    rows: list[dict[str, Any]] = []
    for index, raw_row in enumerate(raw_rows):
        if not isinstance(raw_row, Mapping):
            raise ValueError(f"query_metrics[{index}] must be a mapping")
        row = dict(raw_row)
        row["source_group"] = str(
            raw_row.get("source_group", raw_row.get("scene_id", f"query:{index}"))
        )
        for key in (
            "oracle_true_latent",
            "predicted_latent_matched",
            "predicted_latent_mismatched",
        ):
            metric = raw_row.get(key)
            if not isinstance(metric, Mapping):
                raise ValueError(f"query_metrics[{index}] is missing metric {key!r}")
            row[key] = _metric_for_upgrade(
                metric,
                mismatch_default=key == "predicted_latent_mismatched",
            )
        rows.append(row)

    reps = _infer_bootstrap_reps(upgraded) if bootstrap_reps is None else int(bootstrap_reps)
    if reps < 0:
        raise ValueError("bootstrap_reps must be non-negative")
    groups = sorted({str(row["source_group"]) for row in rows})
    bootstrap_seed = int(seed) + 101
    bootstrap_indices = _bootstrap_group_indices(
        groups,
        bootstrap_reps=reps,
        seed=bootstrap_seed,
    )
    calibration = {
        "raw_rmse": _paired_gain_summary(
            rows,
            "predicted_latent_matched",
            "oracle_true_latent",
            normalized=False,
            bootstrap_indices=bootstrap_indices,
            bootstrap_groups=groups,
        ),
        "normalized_rmse": _paired_gain_summary(
            rows,
            "predicted_latent_matched",
            "oracle_true_latent",
            normalized=True,
            bootstrap_indices=bootstrap_indices,
            bootstrap_groups=groups,
        ),
    }
    history = {
        "raw_rmse": _paired_gain_summary(
            rows,
            "predicted_latent_mismatched",
            "predicted_latent_matched",
            normalized=False,
            bootstrap_indices=bootstrap_indices,
            bootstrap_groups=groups,
        ),
        "normalized_rmse": _paired_gain_summary(
            rows,
            "predicted_latent_mismatched",
            "predicted_latent_matched",
            normalized=True,
            bootstrap_indices=bootstrap_indices,
            bootstrap_groups=groups,
        ),
    }

    old_schema = upgraded.get("schema_version", 1)
    try:
        old_schema_int = int(old_schema)
    except (TypeError, ValueError):
        old_schema_int = 1
    target = upgraded.get("target")
    target_units = target.get("units", "target_units") if isinstance(target, Mapping) else "target_units"
    upgraded.pop("paired_gain", None)
    upgraded["schema_version"] = 2
    upgraded["protocol"] = "development_source_group_3fold_crossfit_physical_readout_v2"
    upgraded["metric_units"] = _metric_units_for_target(str(target_units))
    upgraded["paired_calibration_gap"] = calibration
    upgraded["paired_history_gain"] = history
    crossfit = upgraded.get("source_group_crossfit")
    if isinstance(crossfit, Mapping):
        crossfit_receipt = dict(crossfit)
    else:
        crossfit_receipt = {}
    crossfit_receipt["bootstrap_replicates"] = int(reps)
    crossfit_receipt["bootstrap_seed"] = bootstrap_seed
    upgraded["source_group_crossfit"] = crossfit_receipt
    upgraded["compatibility_upgrade"] = {
        "from_schema_version": old_schema_int,
        "to_schema_version": 2,
        "source": "query_metrics_only",
        "readout_refit": False,
        "world_model_rerun": False,
        "bootstrap_seed": bootstrap_seed,
    }
    return upgraded


def _prediction_export_path(output_dir: Path, scene_id: str) -> Path:
    """Build a stable, filesystem-safe filename for one scene's predictions."""

    safe_id = re.sub(r"[^A-Za-z0-9._-]+", "_", str(scene_id)).strip("._-") or "scene"
    digest = hashlib.sha256(str(scene_id).encode("utf-8")).hexdigest()[:12]
    return output_dir / f"{safe_id[:96]}-{digest}.npz"


def _export_physical_predictions(
    output_dir: Path,
    record: FeatureScene,
    *,
    fold: int,
    matched: np.ndarray,
    calibration: np.ndarray,
) -> None:
    """Save the exact held-out predictions used by the aggregate metrics."""

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        _prediction_export_path(output_dir, record.scene_id),
        truth=record.physical,
        matched=matched,
        calibration=calibration,
        scene_id=np.asarray(record.scene_id),
        source_group=np.asarray(record.source_group),
        fold=np.asarray(int(fold), dtype=np.int64),
    )


def crossfit_physical_readout(
    records: Sequence[FeatureScene],
    *,
    task: str,
    n_folds: int = DEFAULT_FOLDS,
    maximum_samples_per_source: int = DEFAULT_MAX_SAMPLES_PER_SOURCE,
    ridge_alpha: float = DEFAULT_RIDGE_ALPHA,
    projection_dim: int = DEFAULT_PROJECTION_DIM,
    projection_seed: int = DEFAULT_PROJECTION_SEED,
    bootstrap_reps: int = DEFAULT_BOOTSTRAP_REPS,
    seed: int = 0,
    prediction_output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run source-group crossfit and return a JSON-ready diagnostic payload."""

    if not records:
        raise ValueError("at least one feature scene is required")
    family = canonical_task(task)
    if int(maximum_samples_per_source) <= 0:
        raise ValueError("maximum_samples_per_source must be positive")
    if int(bootstrap_reps) < 0:
        raise ValueError("bootstrap_reps must be non-negative")
    feature_width = int(records[0].target.shape[-1])
    stored_projection = any(record.original_latent_dim is not None for record in records)
    if stored_projection and not all(record.original_latent_dim is not None for record in records):
        raise ValueError("either all feature scenes must be preprojected or none may be preprojected")
    for record in records:
        if record.pred.shape != record.target.shape:
            raise ValueError(f"{record.scene_id}: pred/target shapes differ")
        if record.pred.ndim != 4 or record.pred.shape[-1] != feature_width:
            raise ValueError(f"{record.scene_id}: invalid latent shape {record.pred.shape}")
        if record.physical.shape[:3] != record.pred.shape[:3]:
            raise ValueError(f"{record.scene_id}: physical and latent axes differ")
    fold_for_group = assign_source_folds(
        (record.source_group for record in records), n_folds=n_folds, seed=seed
    )
    if stored_projection:
        original_widths = {int(record.original_latent_dim) for record in records}
        projection_seeds = {int(record.projection_seed) for record in records if record.projection_seed is not None}
        if len(original_widths) != 1 or len(projection_seeds) != 1:
            raise ValueError("preprojected feature scenes have inconsistent projection metadata")
        projector = FixedRandomProjector(feature_width, 0, projection_seed)
        projection_input_dim = int(next(iter(original_widths)))
        projection_seed_receipt = int(next(iter(projection_seeds)))
        projection_description = "fixed_sparse_random_preloaded"
    else:
        projector = FixedRandomProjector(feature_width, projection_dim, projection_seed)
        projection_input_dim = int(projector.input_dim)
        projection_seed_receipt = int(projection_seed)
        projection_description = "fixed_sparse_random" if projector.projected else "identity"
    bootstrap_groups = sorted(fold_for_group)
    bootstrap_indices = _bootstrap_group_indices(
        bootstrap_groups,
        bootstrap_reps=bootstrap_reps,
        seed=seed + 101,
    )
    prediction_dir = (
        None if prediction_output_dir is None else Path(prediction_output_dir)
    )
    row_metrics: list[dict[str, Any]] = []
    fold_receipts: list[dict[str, Any]] = []
    for fold in range(int(n_folds)):
        train_records = [record for record in records if fold_for_group[record.source_group] != fold]
        test_records = [record for record in records if fold_for_group[record.source_group] == fold]
        if not train_records or not test_records:
            raise ValueError(f"fold {fold} has empty training or held-out source groups")
        x_train, y_train, sampled = _sample_training_rows(
            train_records,
            projector,
            maximum_per_source=maximum_samples_per_source,
        )
        readout = fit_ridge_readout(x_train, y_train, alpha=ridge_alpha)
        fold_receipts.append(
            {
                "fold": fold,
                "train_groups": sorted({record.source_group for record in train_records}),
                "heldout_groups": sorted({record.source_group for record in test_records}),
                "train_rows": int(len(x_train)),
                "sampled_rows_by_train_group": sampled,
                "readout_fit_uses": "true_future_target_latent_only",
                "readout_solver": str(readout["solver"]),
            }
        )
        for record in test_records:
            pred_latent, true_latent, _ = _record_rows(record, projector)
            pred_physical = apply_ridge_readout(readout, pred_latent).reshape(record.physical.shape)
            oracle_physical = apply_ridge_readout(readout, true_latent).reshape(record.physical.shape)
            if prediction_dir is not None:
                _export_physical_predictions(
                    prediction_dir,
                    record,
                    fold=fold,
                    matched=pred_physical,
                    calibration=oracle_physical,
                )
            row_metrics.append(
                {
                    "scene_id": record.scene_id,
                    "source_group": record.source_group,
                    "fold": fold,
                    "oracle_true_latent": _physical_error(
                        oracle_physical,
                        record.physical,
                        normalization_truth=record.physical,
                    ),
                    "predicted_latent_matched": _physical_error(
                        pred_physical,
                        record.physical,
                        normalization_truth=record.physical,
                    ),
                    "predicted_latent_mismatched": _mismatched_physical_error(
                        pred_physical,
                        record.physical,
                        normalization_truth=record.physical,
                    ),
                }
            )

    summaries = {
        "oracle_true_latent_floor_rmse": _summary(
            row_metrics,
            "oracle_true_latent",
            normalized=False,
            bootstrap_indices=bootstrap_indices,
            bootstrap_groups=bootstrap_groups,
        ),
        "predicted_latent_matched_rmse": _summary(
            row_metrics,
            "predicted_latent_matched",
            normalized=False,
            bootstrap_indices=bootstrap_indices,
            bootstrap_groups=bootstrap_groups,
        ),
        "predicted_latent_mismatched_rmse": _summary(
            row_metrics,
            "predicted_latent_mismatched",
            normalized=False,
            bootstrap_indices=bootstrap_indices,
            bootstrap_groups=bootstrap_groups,
        ),
        "oracle_true_latent_floor_normalized_rmse": _summary(
            row_metrics,
            "oracle_true_latent",
            normalized=True,
            bootstrap_indices=bootstrap_indices,
            bootstrap_groups=bootstrap_groups,
        ),
        "predicted_latent_matched_normalized_rmse": _summary(
            row_metrics,
            "predicted_latent_matched",
            normalized=True,
            bootstrap_indices=bootstrap_indices,
            bootstrap_groups=bootstrap_groups,
        ),
        "predicted_latent_mismatched_normalized_rmse": _summary(
            row_metrics,
            "predicted_latent_mismatched",
            normalized=True,
            bootstrap_indices=bootstrap_indices,
            bootstrap_groups=bootstrap_groups,
        ),
    }
    paired_calibration_gap = {
        "raw_rmse": _paired_gain_summary(
            row_metrics,
            "predicted_latent_matched",
            "oracle_true_latent",
            normalized=False,
            bootstrap_indices=bootstrap_indices,
            bootstrap_groups=bootstrap_groups,
        ),
        "normalized_rmse": _paired_gain_summary(
            row_metrics,
            "predicted_latent_matched",
            "oracle_true_latent",
            normalized=True,
            bootstrap_indices=bootstrap_indices,
            bootstrap_groups=bootstrap_groups,
        ),
    }
    paired_history_gain = {
        "raw_rmse": _paired_gain_summary(
            row_metrics,
            "predicted_latent_mismatched",
            "predicted_latent_matched",
            normalized=False,
            bootstrap_indices=bootstrap_indices,
            bootstrap_groups=bootstrap_groups,
        ),
        "normalized_rmse": _paired_gain_summary(
            row_metrics,
            "predicted_latent_mismatched",
            "predicted_latent_matched",
            normalized=True,
            bootstrap_indices=bootstrap_indices,
            bootstrap_groups=bootstrap_groups,
        ),
    }
    target = target_spec(family)
    return {
        "schema_version": 2,
        "protocol": "development_source_group_3fold_crossfit_physical_readout_v2",
        "task": family,
        "evaluation_split": "development",
        "claim_scope": (
            "Cross-fitted Development diagnostic only; this is not a unified physical benchmark. "
            "Predicted-latent physical error is interpretable only after the held-out true-latent "
            "readout floor is shown to be sufficiently small."
        ),
        "limitations": [
            "The readout is fitted separately for each frozen checkpoint and task; raw latent errors remain encoder-specific.",
            "A nearest simulator trajectory or candidate retrieval score would be discrete identification, not continuous physical prediction accuracy.",
            "Zero across-condition physical variance is reported with a null normalized metric rather than hidden by an epsilon.",
        ],
        "world_model_training": "none; feature encoder and predictions are treated as frozen inputs",
        "source_group_crossfit": {
            "enabled": True,
            "n_folds": int(n_folds),
            "n_source_groups": int(len(fold_for_group)),
            "fold_assignment": fold_for_group,
            "readout_fit_excludes_heldout_source": True,
            "readout_fit_latent": "true_future_target_only",
            "maximum_samples_per_source": int(maximum_samples_per_source),
            "bootstrap_replicates": int(len(bootstrap_indices)),
            "bootstrap_seed": int(seed) + 101,
        },
        "readout": {
            "kind": "intercept_ridge",
            "alpha": float(ridge_alpha),
            "alpha_definition": (
                "dimensionless penalty added to the standardized-feature Gram matrix "
                "after division by sampled-row count; intercept unpenalized"
            ),
            "features_standardized_float64": True,
            "feature_scale_eps": FEATURE_SCALE_EPS,
            "projection": projection_description,
            "input_dim": projection_input_dim,
            "output_dim": int(projector.output_dim),
            "projection_seed": projection_seed_receipt,
            "hyperparameter_selection": "none; fixed before heldout scoring",
        },
        "target": {
            "task_family": target.task,
            "names": list(target.names),
            "units": target.units,
            "description": target.description,
            "zero_physical_variance_is_reported": True,
            "normalization": "per-query mean variance across conditions of true physical target",
        },
        "metric_units": _metric_units_for_target(target.units),
        "n_scenes": int(len(records)),
        "feature_layout": "[K_condition,C_candidate,T,D]",
        "folds": fold_receipts,
        "summaries": summaries,
        "paired_calibration_gap": paired_calibration_gap,
        "paired_history_gain": paired_history_gain,
        "query_metrics": row_metrics,
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
    parser.add_argument("--features-dir", "--features-root", dest="features_dir", type=Path, required=True)
    parser.add_argument("--panels-dir", "--panel-dir", dest="panels_dir", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    parser.add_argument("--max-samples-per-source", type=int, default=DEFAULT_MAX_SAMPLES_PER_SOURCE)
    parser.add_argument("--ridge-alpha", type=float, default=DEFAULT_RIDGE_ALPHA)
    parser.add_argument("--projection-dim", type=int, default=DEFAULT_PROJECTION_DIM)
    parser.add_argument("--projection-seed", type=int, default=DEFAULT_PROJECTION_SEED)
    parser.add_argument("--bootstrap-reps", type=int, default=DEFAULT_BOOTSTRAP_REPS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prediction-output-dir", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    records = load_feature_scenes(
        args.features_dir,
        args.panels_dir,
        args.task,
        projection_dim=args.projection_dim,
        projection_seed=args.projection_seed,
    )
    result = crossfit_physical_readout(
        records,
        task=args.task,
        n_folds=args.folds,
        maximum_samples_per_source=args.max_samples_per_source,
        ridge_alpha=args.ridge_alpha,
        projection_dim=args.projection_dim,
        projection_seed=args.projection_seed,
        bootstrap_reps=args.bootstrap_reps,
        seed=args.seed,
        prediction_output_dir=args.prediction_output_dir,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(_json_safe(result), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
