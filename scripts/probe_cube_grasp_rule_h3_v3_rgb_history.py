#!/usr/bin/env python3
"""Run the frozen Development-only Cube v3 RGB-history probe.

Only ``train.lance`` and ``loader_validation.lance`` beneath the supplied
artifact root may be opened.  Public Test (``validation.lance``) is closed by
protocol and is rejected before any Lance dataset is read.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import platform
import re
import sys
from typing import Any, Mapping, Sequence

import lance
import numpy as np
import PIL
from PIL import Image, UnidentifiedImageError
import pyarrow as pa
import scipy
import sklearn
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


PROTOCOL = "cube_gripper_carry_rule_history3_development_v3"
PROBE_ID = "cube_gripper_carry_h3_v3_rgb_history_probe_v1"
ACTIVE_SPLITS = ("train", "loader_validation")
TABLE_NAMES = {
    "train": "train.lance",
    "loader_validation": "loader_validation.lance",
}
HIDDEN_MODES = ("cannot_hold", "can_hold")
LABEL_ENCODING = {"cannot_hold": 0, "can_hold": 1}
ACTION_ANCHORS = ("endpoint4", "plateau", "ramp4", "front_hold")
MODEL_STEPS = (0, 1, 2, 3)
DECODED_HISTORY_STEPS = (0, 1, 2)
RESIZE_SHAPE = (16, 16)
ACTION_PROFILE_SHAPE = (4, 5, 5)

BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 2026081103
BOOTSTRAP_LOWER_QUANTILE = 0.025
PERMUTATION_REPETITIONS = 16
PERMUTATION_SEED = 2026081104

OVERALL_ACCURACY_MINIMUM = 0.75
WORST_MODE_ACCURACY_MINIMUM = 0.70
WORST_ANCHOR_ACCURACY_MINIMUM = 0.70
BOOTSTRAP_LOWER_BOUND_MINIMUM = 0.70
PERMUTATION_MEAN_ACCURACY_MAXIMUM = 0.60
SHORTCUT_ACCURACY_MAXIMUM = 0.51

TABLE_COLUMNS = (
    "model_step_idx",
    "pixels",
    "action_block",
    "pair_id",
    "hidden_mode",
    "split",
    "action_anchor_id",
    "action_profile_id",
    "scene_template_content_hash",
    "pair_content_hash",
)
METADATA_ACTION_COLUMNS = tuple(
    name for name in TABLE_COLUMNS if name != "pixels"
)
PIXEL_JOIN_COLUMNS = (
    "pair_id",
    "hidden_mode",
    "split",
    "model_step_idx",
    "pixels",
)
PIXEL_FILTER = "model_step_idx <= 2"
MAIN_FEATURE_COLUMNS = ("pixels",)
NEGATIVE_CONTROL_ONLY_COLUMNS = ("action_block",)
AUDIT_ONLY_COLUMNS = tuple(
    name
    for name in TABLE_COLUMNS
    if name not in MAIN_FEATURE_COLUMNS + NEGATIVE_CONTROL_ONLY_COLUMNS
)
PRIVILEGED_COLUMNS_EXCLUDED_FROM_MAIN_FEATURE = (
    "physical_state",
    "hidden_grasp_enabled",
    "episode_idx",
    "model_step_idx",
    "pair_id",
    "hidden_mode",
    "split",
    "catalog_index",
    "source_row",
    "source_episode",
    "source_step",
    "action_anchor_id",
    "action_profile_id",
    "scene_template_content_hash",
    "pair_content_hash",
)


@dataclass(frozen=True)
class _Condition:
    pair_id: str
    hidden_mode: str
    action_anchor_id: str
    action_profile_id: str
    scene_template_content_hash: str
    pair_content_hash: str
    x0_jpeg: bytes
    query_jpeg: bytes
    x0_rgb: np.ndarray
    query_rgb: np.ndarray
    main_feature: np.ndarray
    action_blocks: np.ndarray


@dataclass(frozen=True)
class PreparedSplit:
    split: str
    main_features: np.ndarray
    x0_features: np.ndarray
    query_features: np.ndarray
    action_features: np.ndarray
    labels: np.ndarray
    pair_ids: np.ndarray
    hidden_modes: np.ndarray
    action_anchors: np.ndarray
    action_profile_ids: frozenset[str]
    scene_template_content_hashes: frozenset[str]
    pair_content_hashes: frozenset[str]
    pair_count: int
    condition_count: int
    row_count: int
    anchor_pair_counts: Mapping[str, int]


@dataclass(frozen=True)
class _FitResult:
    predictions: np.ndarray
    transformed_train: np.ndarray
    transformed_development: np.ndarray
    receipt: Mapping[str, Any]


def _array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def _require_sha256(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{field_name} must be a canonical lowercase SHA256")
    return value


def action_profile_content_sha256(action_blocks: np.ndarray) -> str:
    blocks = np.asarray(action_blocks, dtype=np.float32)
    if blocks.shape != ACTION_PROFILE_SHAPE:
        raise ValueError(
            f"action profile must have shape {ACTION_PROFILE_SHAPE}, got "
            f"{blocks.shape}"
        )
    if not np.isfinite(blocks).all():
        raise ValueError("action profile contains a non-finite value")
    if np.count_nonzero(blocks[3]):
        raise ValueError("terminal fourth action block must be exactly zero")
    return _array_sha256(blocks)


def pair_content_sha256(scene_hash: str, profile_hash: str) -> str:
    scene = bytes.fromhex(
        _require_sha256(scene_hash, field_name="scene_template_content_hash")
    )
    profile = bytes.fromhex(
        _require_sha256(profile_hash, field_name="action_profile_id")
    )
    return hashlib.sha256(scene + profile).hexdigest()


def _decode_rgb_frame(payload: Any) -> np.ndarray:
    if isinstance(payload, memoryview):
        payload = payload.tobytes()
    if isinstance(payload, bytearray):
        payload = bytes(payload)
    if not isinstance(payload, bytes) or not payload:
        raise ValueError("pixels must contain non-empty encoded JPEG bytes")
    try:
        with Image.open(BytesIO(payload)) as image:
            if image.format != "JPEG":
                raise ValueError(
                    f"pixels must use the frozen JPEG container, got {image.format!r}"
                )
            rgb = image.convert("RGB")
            try:
                resized = rgb.resize(
                    RESIZE_SHAPE,
                    resample=Image.Resampling.BILINEAR,
                )
                try:
                    values = np.asarray(resized, dtype=np.float64).copy()
                finally:
                    resized.close()
            finally:
                rgb.close()
    except (OSError, UnidentifiedImageError) as error:
        raise ValueError("pixels could not be decoded as JPEG by Pillow") from error
    if values.shape != (16, 16, 3) or values.dtype != np.float64:
        raise RuntimeError("Pillow RGB decoder violated the frozen output contract")
    return values


def rgb_history_feature(x0: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    frames = [np.asarray(value) for value in (x0, x1, x2)]
    if any(value.shape != (16, 16, 3) for value in frames):
        raise ValueError("x0/x1/x2 must each have shape [16,16,3]")
    if any(value.dtype != np.float64 for value in frames):
        raise TypeError("x0/x1/x2 arithmetic must use float64")
    feature = 2.0 * frames[1] - frames[0] - frames[2]
    return np.ascontiguousarray(feature.reshape(-1, order="C"), dtype=np.float64)


def _constant_text(
    rows: Sequence[Mapping[str, Any]], field_name: str, *, context: str
) -> str:
    values = {str(row[field_name]) for row in rows}
    if len(values) != 1:
        raise ValueError(f"{context}: {field_name} changed across four rows")
    value = values.pop()
    if not value:
        raise ValueError(f"{context}: {field_name} must be non-empty")
    return value


def _action_block(row: Mapping[str, Any], *, context: str) -> np.ndarray:
    values = np.asarray(row["action_block"], dtype=np.float32)
    if values.size != 25:
        raise ValueError(f"{context}: action_block must contain 25 float32 values")
    values = values.reshape(5, 5)
    if not np.isfinite(values).all():
        raise ValueError(f"{context}: action_block contains a non-finite value")
    return values


def _condition_from_rows(
    rows: Sequence[Mapping[str, Any]], *, expected_split: str
) -> _Condition:
    pair_id = _constant_text(rows, "pair_id", context="condition")
    hidden_mode = _constant_text(rows, "hidden_mode", context=pair_id)
    context = f"{pair_id}/{hidden_mode}"
    if hidden_mode not in HIDDEN_MODES:
        raise ValueError(f"{context}: unexpected hidden mode")
    if _constant_text(rows, "split", context=context) != expected_split:
        raise ValueError(f"{context}: split metadata mismatch")
    anchor = _constant_text(rows, "action_anchor_id", context=context)
    if anchor not in ACTION_ANCHORS:
        raise ValueError(f"{context}: unexpected action anchor {anchor!r}")
    profile_id = _require_sha256(
        _constant_text(rows, "action_profile_id", context=context),
        field_name="action_profile_id",
    )
    scene_hash = _require_sha256(
        _constant_text(rows, "scene_template_content_hash", context=context),
        field_name="scene_template_content_hash",
    )
    pair_hash = _require_sha256(
        _constant_text(rows, "pair_content_hash", context=context),
        field_name="pair_content_hash",
    )

    by_step: dict[int, Mapping[str, Any]] = {}
    for row in rows:
        raw_step = row["model_step_idx"]
        if isinstance(raw_step, (bool, np.bool_)) or not isinstance(
            raw_step, (int, np.integer)
        ):
            raise TypeError(f"{context}: model_step_idx must be an integer")
        step = int(raw_step)
        if step in by_step:
            raise ValueError(f"{context}: duplicate model_step_idx={step}")
        by_step[step] = row
    if set(by_step) != set(MODEL_STEPS):
        raise ValueError(
            f"{context}: expected exactly model_step_idx 0..3, got "
            f"{sorted(by_step)}"
        )

    blocks = np.stack(
        [_action_block(by_step[step], context=context) for step in MODEL_STEPS]
    ).astype(np.float32, copy=False)
    calculated_profile = action_profile_content_sha256(blocks)
    if calculated_profile != profile_id:
        raise ValueError(
            f"{context}: action_profile_id does not match actual float32 actions"
        )
    calculated_pair = pair_content_sha256(scene_hash, profile_id)
    if calculated_pair != pair_hash:
        raise ValueError(
            f"{context}: pair_content_hash does not bind scene/profile content"
        )

    encoded = []
    decoded = []
    for step in DECODED_HISTORY_STEPS:
        if "pixels" not in by_step[step]:
            raise ValueError(f"{context}: missing pixels at model_step_idx={step}")
        payload = by_step[step]["pixels"]
        if isinstance(payload, memoryview):
            payload = payload.tobytes()
        if isinstance(payload, bytearray):
            payload = bytes(payload)
        if not isinstance(payload, bytes):
            raise ValueError(f"{context}: pixels at step {step} are not bytes")
        encoded.append(payload)
        decoded.append(_decode_rgb_frame(payload))
    x0, x1, x2 = decoded
    return _Condition(
        pair_id=pair_id,
        hidden_mode=hidden_mode,
        action_anchor_id=anchor,
        action_profile_id=profile_id,
        scene_template_content_hash=scene_hash,
        pair_content_hash=pair_hash,
        x0_jpeg=encoded[0],
        query_jpeg=encoded[2],
        x0_rgb=x0,
        query_rgb=x2,
        main_feature=rgb_history_feature(x0, x1, x2),
        action_blocks=np.ascontiguousarray(blocks),
    )


def prepare_split(
    rows: Sequence[Mapping[str, Any]], *, expected_split: str
) -> PreparedSplit:
    if expected_split not in ACTIVE_SPLITS:
        raise ValueError(f"inactive or Public split refused: {expected_split!r}")
    if not rows:
        raise ValueError(f"{expected_split}: empty table")
    required = set(METADATA_ACTION_COLUMNS)
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row_index, row in enumerate(rows):
        missing = sorted(required - set(row))
        if missing:
            raise ValueError(f"row {row_index}: missing required columns {missing}")
        split = str(row["split"])
        if split != expected_split:
            raise ValueError(
                f"row {row_index}: expected split={expected_split!r}, got {split!r}"
            )
        key = (str(row["pair_id"]), str(row["hidden_mode"]))
        groups.setdefault(key, []).append(row)

    conditions = {
        key: _condition_from_rows(group, expected_split=expected_split)
        for key, group in groups.items()
    }
    pair_ids = sorted({key[0] for key in conditions})
    ordered: list[_Condition] = []
    anchor_pair_counts = {anchor: 0 for anchor in ACTION_ANCHORS}
    profile_ids: set[str] = set()
    scene_hashes: set[str] = set()
    pair_hashes: set[str] = set()
    for pair_id in pair_ids:
        modes = {
            mode for candidate_pair, mode in conditions if candidate_pair == pair_id
        }
        if modes != set(HIDDEN_MODES):
            raise ValueError(f"{pair_id}: incomplete hidden-mode pair: {sorted(modes)}")
        cannot = conditions[(pair_id, "cannot_hold")]
        can = conditions[(pair_id, "can_hold")]
        metadata = (
            "action_anchor_id",
            "action_profile_id",
            "scene_template_content_hash",
            "pair_content_hash",
        )
        for field_name in metadata:
            if getattr(cannot, field_name) != getattr(can, field_name):
                raise ValueError(f"{pair_id}: paired {field_name} values differ")
        if cannot.x0_jpeg != can.x0_jpeg or not np.array_equal(
            cannot.x0_rgb, can.x0_rgb
        ):
            raise ValueError(f"{pair_id}: paired x0 pixels are not bitwise identical")
        if cannot.query_jpeg != can.query_jpeg or not np.array_equal(
            cannot.query_rgb, can.query_rgb
        ):
            raise ValueError(
                f"{pair_id}: paired query/x2 pixels are not bitwise identical"
            )
        if not np.array_equal(cannot.action_blocks, can.action_blocks):
            raise ValueError(f"{pair_id}: paired actions are not bitwise identical")

        anchor_pair_counts[cannot.action_anchor_id] += 1
        for value, target, field_name in (
            (cannot.action_profile_id, profile_ids, "action_profile_id"),
            (
                cannot.scene_template_content_hash,
                scene_hashes,
                "scene_template_content_hash",
            ),
            (cannot.pair_content_hash, pair_hashes, "pair_content_hash"),
        ):
            if value in target:
                raise ValueError(f"{pair_id}: duplicate {field_name} inside split")
            target.add(value)
        ordered.extend((cannot, can))

    if not pair_ids or len(pair_ids) % len(ACTION_ANCHORS):
        raise ValueError("pair count must be positive and divisible by four anchors")
    expected_anchor_count = len(pair_ids) // len(ACTION_ANCHORS)
    if set(anchor_pair_counts.values()) != {expected_anchor_count}:
        raise ValueError(
            f"{expected_split}: action anchors are not exactly balanced: "
            f"{anchor_pair_counts}"
        )

    return PreparedSplit(
        split=expected_split,
        main_features=np.ascontiguousarray(
            np.stack([condition.main_feature for condition in ordered]),
            dtype=np.float64,
        ),
        x0_features=np.ascontiguousarray(
            np.stack(
                [condition.x0_rgb.reshape(-1, order="C") for condition in ordered]
            ),
            dtype=np.float64,
        ),
        query_features=np.ascontiguousarray(
            np.stack(
                [
                    condition.query_rgb.reshape(-1, order="C")
                    for condition in ordered
                ]
            ),
            dtype=np.float64,
        ),
        action_features=np.ascontiguousarray(
            np.stack(
                [
                    condition.action_blocks.astype(np.float64).reshape(
                        -1, order="C"
                    )
                    for condition in ordered
                ]
            ),
            dtype=np.float64,
        ),
        labels=np.asarray(
            [LABEL_ENCODING[condition.hidden_mode] for condition in ordered],
            dtype=np.int64,
        ),
        pair_ids=np.asarray([condition.pair_id for condition in ordered]),
        hidden_modes=np.asarray(
            [condition.hidden_mode for condition in ordered]
        ),
        action_anchors=np.asarray(
            [condition.action_anchor_id for condition in ordered]
        ),
        action_profile_ids=frozenset(profile_ids),
        scene_template_content_hashes=frozenset(scene_hashes),
        pair_content_hashes=frozenset(pair_hashes),
        pair_count=len(pair_ids),
        condition_count=len(ordered),
        row_count=len(rows),
        anchor_pair_counts=dict(anchor_pair_counts),
    )


def cross_split_content_audit(
    train: PreparedSplit, development: PreparedSplit
) -> dict[str, Any]:
    if train.split != "train" or development.split != "loader_validation":
        raise ValueError("cross-split audit requires Training then Development")

    def overlap(left: frozenset[str], right: frozenset[str]) -> list[str]:
        return sorted(left & right)

    profiles = overlap(train.action_profile_ids, development.action_profile_ids)
    scenes = overlap(
        train.scene_template_content_hashes,
        development.scene_template_content_hashes,
    )
    pairs = overlap(train.pair_content_hashes, development.pair_content_hashes)
    train_anchors = sorted(set(train.action_anchors.tolist()))
    development_anchors = sorted(set(development.action_anchors.tolist()))
    expected_anchors = sorted(ACTION_ANCHORS)
    checks = {
        "exact_action_profile_id_overlap_zero": not profiles,
        "scene_template_content_hash_overlap_zero": not scenes,
        "pair_content_hash_overlap_zero": not pairs,
        "four_anchor_families_present_in_both_splits": (
            train_anchors == expected_anchors
            and development_anchors == expected_anchors
        ),
    }
    return {
        "evidence_source": {
            "action_profile_id": "recomputed_from_table_float32_action_blocks",
            "scene_template_content_hash": "frozen_table_column",
            "pair_content_hash": "recomputed_from_table_scene_and_profile_digests",
            "manifest_read": False,
        },
        "exact_action_profile_id_overlap": {
            "count": len(profiles),
            "values": profiles,
        },
        "scene_template_content_hash_overlap": {
            "count": len(scenes),
            "values": scenes,
        },
        "pair_content_hash_overlap": {
            "count": len(pairs),
            "values": pairs,
        },
        "anchor_families": {
            "expected": expected_anchors,
            "train": train_anchors,
            "loader_validation": development_anchors,
            "shared_families_are_expected_not_content_leakage": True,
        },
        "pair_id_is_content_isolation_evidence": False,
        "checks": checks,
        "passed": bool(all(checks.values())),
    }


def _fit_ridge(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    development_features: np.ndarray,
    *,
    feature_name: str,
) -> _FitResult:
    train_values = np.asarray(train_features, dtype=np.float64)
    development_values = np.asarray(development_features, dtype=np.float64)
    labels = np.asarray(train_labels, dtype=np.int64)
    if train_values.ndim != 2 or development_values.ndim != 2:
        raise ValueError(f"{feature_name}: features must be rank-2")
    if train_values.shape[1] != development_values.shape[1]:
        raise ValueError(f"{feature_name}: split feature dimensions differ")
    if train_values.shape[0] != labels.size or set(labels.tolist()) != {0, 1}:
        raise ValueError(f"{feature_name}: Training labels must contain classes 0/1")
    if not np.isfinite(train_values).all() or not np.isfinite(
        development_values
    ).all():
        raise ValueError(f"{feature_name}: features contain non-finite values")

    scaler = StandardScaler()
    transformed_train = scaler.fit_transform(train_values)
    transformed_development = scaler.transform(development_values)
    classifier = RidgeClassifier(alpha=1.0)
    classifier.fit(transformed_train, labels)
    predictions = np.asarray(
        classifier.predict(transformed_development), dtype=np.int64
    )
    receipt = {
        "feature_name": feature_name,
        "feature_dimension": int(train_values.shape[1]),
        "standard_scaler": {
            "fit_split": "train",
            "development_used_for_fit": False,
            "with_mean": bool(scaler.with_mean),
            "with_std": bool(scaler.with_std),
            "n_samples_seen": int(scaler.n_samples_seen_),
            "mean_float64_sha256": _array_sha256(
                np.asarray(scaler.mean_, dtype=np.float64)
            ),
            "scale_float64_sha256": _array_sha256(
                np.asarray(scaler.scale_, dtype=np.float64)
            ),
        },
        "ridge_classifier": {
            "alpha": 1.0,
            "decision_rule": "sklearn.linear_model.RidgeClassifier.predict",
            "classes": [int(value) for value in classifier.classes_],
            "coefficient_sha256": _array_sha256(
                np.asarray(classifier.coef_, dtype=np.float64)
            ),
            "intercept_sha256": _array_sha256(
                np.asarray(classifier.intercept_, dtype=np.float64)
            ),
        },
    }
    return _FitResult(
        predictions=predictions,
        transformed_train=np.ascontiguousarray(transformed_train),
        transformed_development=np.ascontiguousarray(transformed_development),
        receipt=receipt,
    )


def _accuracy(labels: np.ndarray, predictions: np.ndarray) -> float:
    actual = np.asarray(labels, dtype=np.int64)
    predicted = np.asarray(predictions, dtype=np.int64)
    if actual.shape != predicted.shape or actual.ndim != 1 or not actual.size:
        raise ValueError("accuracy requires equal non-empty one-dimensional arrays")
    return float(np.mean(actual == predicted))


def stratified_pair_cluster_bootstrap(
    labels: np.ndarray,
    predictions: np.ndarray,
    pair_ids: np.ndarray,
    action_anchors: np.ndarray,
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    actual = np.asarray(labels, dtype=np.int64)
    predicted = np.asarray(predictions, dtype=np.int64)
    pairs = np.asarray(pair_ids).astype(str)
    anchors = np.asarray(action_anchors).astype(str)
    if not (
        actual.shape == predicted.shape == pairs.shape == anchors.shape
        and actual.ndim == 1
    ):
        raise ValueError("bootstrap inputs must be aligned one-dimensional arrays")
    if isinstance(resamples, bool) or int(resamples) <= 0:
        raise ValueError("bootstrap resamples must be positive")

    correct = actual == predicted
    clusters_by_anchor: dict[str, np.ndarray] = {}
    stratum_counts: dict[str, int] = {}
    for anchor in ACTION_ANCHORS:
        anchor_pairs = sorted(set(pairs[anchors == anchor].tolist()))
        if not anchor_pairs:
            raise ValueError(f"bootstrap anchor stratum {anchor!r} is empty")
        cluster_accuracy: list[float] = []
        for pair_id in anchor_pairs:
            indices = np.flatnonzero(pairs == pair_id)
            if indices.size != 2:
                raise ValueError(f"bootstrap pair {pair_id!r} must contain two modes")
            if set(anchors[indices].tolist()) != {anchor}:
                raise ValueError(f"bootstrap pair {pair_id!r} crosses anchors")
            cluster_accuracy.append(float(np.mean(correct[indices])))
        clusters_by_anchor[anchor] = np.asarray(
            cluster_accuracy, dtype=np.float64
        )
        stratum_counts[anchor] = len(anchor_pairs)

    rng = np.random.default_rng(seed)
    bootstrap_sums = np.zeros(int(resamples), dtype=np.float64)
    sampled_pairs = 0
    for anchor in ACTION_ANCHORS:
        values = clusters_by_anchor[anchor]
        draws = rng.integers(
            0,
            values.size,
            size=(int(resamples), values.size),
        )
        bootstrap_sums += values[draws].sum(axis=1)
        sampled_pairs += int(values.size)
    samples = bootstrap_sums / sampled_pairs
    lower = float(
        np.quantile(samples, BOOTSTRAP_LOWER_QUANTILE, method="linear")
    )
    upper = float(np.quantile(samples, 0.975, method="linear"))
    return {
        "unit": "pair_cluster",
        "stratification": "action_anchor_id",
        "resamples": int(resamples),
        "seed": int(seed),
        "lower_quantile": BOOTSTRAP_LOWER_QUANTILE,
        "quantile_method": "numpy_linear",
        "stratum_pair_counts": stratum_counts,
        "overall_accuracy": _accuracy(actual, predicted),
        "lower_bound_2_5_percent": lower,
        "upper_bound_97_5_percent": upper,
        "bootstrap_mean": float(np.mean(samples)),
        "bootstrap_minimum": float(np.min(samples)),
        "bootstrap_maximum": float(np.max(samples)),
        "gate_minimum": BOOTSTRAP_LOWER_BOUND_MINIMUM,
        "passed": bool(lower >= BOOTSTRAP_LOWER_BOUND_MINIMUM),
    }


def _permuted_label_control(
    transformed_train: np.ndarray,
    train_labels: np.ndarray,
    transformed_development: np.ndarray,
    development_labels: np.ndarray,
    *,
    repetitions: int = PERMUTATION_REPETITIONS,
    seed: int = PERMUTATION_SEED,
) -> dict[str, Any]:
    if isinstance(repetitions, bool) or int(repetitions) <= 0:
        raise ValueError("permutation repetitions must be positive")
    rng = np.random.default_rng(seed)
    scores: list[float] = []
    for _ in range(int(repetitions)):
        permuted = rng.permutation(np.asarray(train_labels, dtype=np.int64))
        classifier = RidgeClassifier(alpha=1.0)
        classifier.fit(transformed_train, permuted)
        predictions = classifier.predict(transformed_development)
        scores.append(_accuracy(development_labels, predictions))
    mean = float(np.mean(scores))
    return {
        "permutation_target": "Training condition labels only",
        "development_labels_remain_true": True,
        "feature_scaler_reused_from_primary_train_fit": True,
        "repetitions": int(repetitions),
        "seed": int(seed),
        "scores": scores,
        "mean_accuracy": mean,
        "maximum_mean_accuracy": PERMUTATION_MEAN_ACCURACY_MAXIMUM,
        "passed": bool(mean <= PERMUTATION_MEAN_ACCURACY_MAXIMUM),
    }


def _group_metrics(
    development: PreparedSplit, predictions: np.ndarray
) -> dict[str, Any]:
    labels = development.labels
    overall = _accuracy(labels, predictions)
    per_mode = {
        mode: _accuracy(
            labels[development.hidden_modes == mode],
            predictions[development.hidden_modes == mode],
        )
        for mode in HIDDEN_MODES
    }
    per_anchor = {
        anchor: _accuracy(
            labels[development.action_anchors == anchor],
            predictions[development.action_anchors == anchor],
        )
        for anchor in ACTION_ANCHORS
    }
    worst_mode = min(HIDDEN_MODES, key=lambda mode: per_mode[mode])
    worst_anchor = min(ACTION_ANCHORS, key=lambda anchor: per_anchor[anchor])
    return {
        "overall_accuracy": overall,
        "per_mode_accuracy": per_mode,
        "worst_mode": {
            "hidden_mode": worst_mode,
            "accuracy": per_mode[worst_mode],
        },
        "per_anchor_family_accuracy": per_anchor,
        "worst_anchor_family": {
            "action_anchor_id": worst_anchor,
            "accuracy": per_anchor[worst_anchor],
        },
    }


def _shortcut_control(
    name: str,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    development_features: np.ndarray,
    development_labels: np.ndarray,
) -> dict[str, Any]:
    fit = _fit_ridge(
        train_features,
        train_labels,
        development_features,
        feature_name=name,
    )
    accuracy = _accuracy(development_labels, fit.predictions)
    return {
        "accuracy": accuracy,
        "maximum_accuracy": SHORTCUT_ACCURACY_MAXIMUM,
        "passed": bool(accuracy <= SHORTCUT_ACCURACY_MAXIMUM),
        "fit_receipt": fit.receipt,
    }


def _package_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "Pillow": PIL.__version__,
        "Pillow_jpeglib": str(getattr(Image.core, "jpeglib_version", "unknown")),
        "scikit-learn": sklearn.__version__,
        "scipy": scipy.__version__,
        "lance": lance.__version__,
        "pyarrow": pa.__version__,
    }


def evaluate_prepared_splits(
    train: PreparedSplit,
    development: PreparedSplit,
    *,
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    permutation_repetitions: int = PERMUTATION_REPETITIONS,
    permutation_seed: int = PERMUTATION_SEED,
) -> dict[str, Any]:
    content_audit = cross_split_content_audit(train, development)
    primary = _fit_ridge(
        train.main_features,
        train.labels,
        development.main_features,
        feature_name="flatten(2*x1-x0-x2)_C_order",
    )
    metrics = _group_metrics(development, primary.predictions)
    bootstrap = stratified_pair_cluster_bootstrap(
        development.labels,
        primary.predictions,
        development.pair_ids,
        development.action_anchors,
        resamples=bootstrap_resamples,
        seed=bootstrap_seed,
    )
    controls = {
        "label_permutation": _permuted_label_control(
            primary.transformed_train,
            train.labels,
            primary.transformed_development,
            development.labels,
            repetitions=permutation_repetitions,
            seed=permutation_seed,
        ),
        "x0_only": _shortcut_control(
            "x0_only",
            train.x0_features,
            train.labels,
            development.x0_features,
            development.labels,
        ),
        "query_x2_only": _shortcut_control(
            "query_x2_only",
            train.query_features,
            train.labels,
            development.query_features,
            development.labels,
        ),
        "action_only": _shortcut_control(
            "action_only",
            train.action_features,
            train.labels,
            development.action_features,
            development.labels,
        ),
    }
    gates = {
        "cross_split_content_isolation_passed": bool(content_audit["passed"]),
        "paired_x0_query_actions_identical": True,
        "overall_accuracy_at_least_0_75": bool(
            metrics["overall_accuracy"] >= OVERALL_ACCURACY_MINIMUM
        ),
        "worst_mode_accuracy_at_least_0_70": bool(
            metrics["worst_mode"]["accuracy"]
            >= WORST_MODE_ACCURACY_MINIMUM
        ),
        "worst_anchor_accuracy_at_least_0_70": bool(
            metrics["worst_anchor_family"]["accuracy"]
            >= WORST_ANCHOR_ACCURACY_MINIMUM
        ),
        "bootstrap_2_5_percent_lower_bound_at_least_0_70": bool(
            bootstrap["passed"]
        ),
        "permuted_label_mean_accuracy_at_most_0_60": bool(
            controls["label_permutation"]["passed"]
        ),
        "x0_only_accuracy_at_most_0_51": bool(controls["x0_only"]["passed"]),
        "query_x2_only_accuracy_at_most_0_51": bool(
            controls["query_x2_only"]["passed"]
        ),
        "action_only_accuracy_at_most_0_51": bool(
            controls["action_only"]["passed"]
        ),
    }
    passed = bool(all(gates.values()))
    return {
        "schema_version": 1,
        "probe_id": PROBE_ID,
        "protocol": PROTOCOL,
        "status": "passed" if passed else "failed",
        "role": "frozen_rgb_history_data_probe_not_reference_model_evaluation",
        "active_splits": list(ACTIVE_SPLITS),
        "public_test": {
            "canonical_split": "validation",
            "access_status": "closed_not_read_not_scored",
            "opened": False,
            "read": False,
            "hashed": False,
            "scored": False,
        },
        "decoder_and_feature_contract": {
            "source_column": "pixels",
            "decoded_steps": [0, 1, 2],
            "x3_decoded_or_used": False,
            "x3_pixel_bytes_not_read_from_lance": True,
            "container": "JPEG_only",
            "decoder": "PIL.Image.open(BytesIO(payload)).convert('RGB')",
            "channel_order": "RGB",
            "resize_shape": [16, 16],
            "resize_interpolation": "PIL.Image.Resampling.BILINEAR",
            "arithmetic_dtype": "float64",
            "fixed_main_feature": "flatten(2*x1-x0-x2)_C_order",
            "main_feature_dimension": 16 * 16 * 3,
            "main_feature_columns": list(MAIN_FEATURE_COLUMNS),
            "negative_control_only_columns": list(
                NEGATIVE_CONTROL_ONLY_COLUMNS
            ),
            "audit_only_columns": list(AUDIT_ONLY_COLUMNS),
            "privileged_columns_excluded_from_main_feature": list(
                PRIVILEGED_COLUMNS_EXCLUDED_FROM_MAIN_FEATURE
            ),
            "action_is_negative_control_only": True,
            "ids_labels_metadata_and_row_order_used_as_main_feature": False,
        },
        "label_contract": {
            "encoding": dict(LABEL_ENCODING),
            "label_source_used_only_after_feature_construction": "hidden_mode",
        },
        "data_integrity": {
            "grouping_key": ["pair_id", "hidden_mode"],
            "required_rows_per_condition": 4,
            "required_model_step_indices": list(MODEL_STEPS),
            "paired_x0_jpeg_and_decoded_rgb_bitwise_equal": True,
            "paired_query_x2_jpeg_and_decoded_rgb_bitwise_equal": True,
            "paired_float32_action_blocks_bitwise_equal": True,
            "action_profile_id_recomputed_from_actual_float32_blocks": True,
            "pair_content_hash_recomputed_from_scene_and_profile_digests": True,
            "splits": {
                split.split: {
                    "row_count": split.row_count,
                    "pair_count": split.pair_count,
                    "condition_count": split.condition_count,
                    "anchor_pair_counts": dict(split.anchor_pair_counts),
                }
                for split in (train, development)
            },
        },
        "cross_split_content_isolation": content_audit,
        "fit_contract": {
            "standard_scaler_fit_split_only": "train",
            "ridge_classifier_alpha": 1.0,
            "development_evaluated_once_without_tuning": True,
            "reference_model_or_checkpoint_loaded": False,
            "primary_fit_receipt": primary.receipt,
        },
        "primary_probe": {
            "metrics": metrics,
            "thresholds": {
                "overall_accuracy_minimum": OVERALL_ACCURACY_MINIMUM,
                "worst_mode_accuracy_minimum": WORST_MODE_ACCURACY_MINIMUM,
                "worst_anchor_family_accuracy_minimum": (
                    WORST_ANCHOR_ACCURACY_MINIMUM
                ),
            },
        },
        "pair_cluster_anchor_stratified_bootstrap": bootstrap,
        "negative_controls": controls,
        "gates": gates,
        "package_versions": _package_versions(),
        "passed": passed,
    }


def evaluate_fixture_rows(
    train_rows: Sequence[Mapping[str, Any]],
    development_rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    permutation_repetitions: int = PERMUTATION_REPETITIONS,
    permutation_seed: int = PERMUTATION_SEED,
) -> dict[str, Any]:
    """Pure-row entry point used by unit fixtures; it never opens a table."""

    train = prepare_split(train_rows, expected_split="train")
    development = prepare_split(
        development_rows,
        expected_split="loader_validation",
    )
    return evaluate_prepared_splits(
        train,
        development,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
        permutation_repetitions=permutation_repetitions,
        permutation_seed=permutation_seed,
    )


def _forbidden_closed_component(path: Path) -> str | None:
    forbidden = {
        "validation",
        "validation.lance",
        "public",
        "public_test",
        "public-test",
        "public test",
        "publictest",
    }
    for part in path.parts:
        if part.lower() in forbidden:
            return part
    return None


def resolve_allowed_tables(artifact_root: Path) -> tuple[Path, dict[str, Path]]:
    root_input = artifact_root.expanduser()
    forbidden = _forbidden_closed_component(root_input)
    if forbidden is not None:
        raise ValueError(
            f"Cube v3 probe explicitly refuses validation/Public path component "
            f"{forbidden!r}"
        )
    if root_input.name.lower().endswith(".lance"):
        raise ValueError("--artifact-root must be a root, not a Lance table path")
    root = root_input.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)

    lance_children = sorted(
        value for value in root.iterdir() if value.name.lower().endswith(".lance")
    )
    allowed_names = set(TABLE_NAMES.values())
    extras = [value.name for value in lance_children if value.name not in allowed_names]
    if extras:
        raise ValueError(
            "Cube v3 probe refuses validation/Public or any non-authorized Lance "
            f"table under the artifact root: {extras}"
        )
    tables: dict[str, Path] = {}
    for split, name in TABLE_NAMES.items():
        path = root / name
        if path.is_symlink():
            raise ValueError(f"authorized Lance table cannot be a symlink: {name}")
        if not path.is_dir():
            raise FileNotFoundError(path)
        resolved = path.resolve()
        if resolved.parent != root or resolved.name != name:
            raise ValueError(f"authorized Lance table escapes artifact root: {name}")
        tables[split] = resolved
    return root, tables


def _projection_key(
    row: Mapping[str, Any], *, source: str
) -> tuple[str, str, str, int]:
    required = {"pair_id", "hidden_mode", "split", "model_step_idx"}
    missing = sorted(required - set(row))
    if missing:
        raise ValueError(f"{source}: projection row is missing join keys {missing}")
    raw_step = row["model_step_idx"]
    if isinstance(raw_step, (bool, np.bool_)) or not isinstance(
        raw_step, (int, np.integer)
    ):
        raise TypeError(f"{source}: model_step_idx join key must be an integer")
    return (
        str(row["pair_id"]),
        str(row["hidden_mode"]),
        str(row["split"]),
        int(raw_step),
    )


def _merge_lance_projections(
    metadata_rows: Sequence[Mapping[str, Any]],
    pixel_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Join x0/x1/x2 pixels onto four-row metadata without reading x3 bytes."""

    metadata_keys: set[tuple[str, str, str, int]] = set()
    ordered_metadata: list[tuple[tuple[str, str, str, int], Mapping[str, Any]]] = []
    for row in metadata_rows:
        if "pixels" in row:
            raise ValueError("metadata/action projection unexpectedly contains pixels")
        missing = sorted(set(METADATA_ACTION_COLUMNS) - set(row))
        if missing:
            raise ValueError(f"metadata/action projection is missing {missing}")
        key = _projection_key(row, source="metadata/action")
        if key in metadata_keys:
            raise ValueError(f"duplicate metadata/action projection key: {key}")
        metadata_keys.add(key)
        ordered_metadata.append((key, row))

    pixels_by_key: dict[tuple[str, str, str, int], bytes] = {}
    for row in pixel_rows:
        missing = sorted(set(PIXEL_JOIN_COLUMNS) - set(row))
        if missing:
            raise ValueError(f"pixel projection is missing {missing}")
        key = _projection_key(row, source="pixels")
        if key[3] not in DECODED_HISTORY_STEPS:
            raise ValueError(
                "pixel projection returned x3 or an out-of-contract step despite "
                f"the frozen filter: {key}"
            )
        if key in pixels_by_key:
            raise ValueError(f"duplicate pixel projection key: {key}")
        payload = row["pixels"]
        if isinstance(payload, memoryview):
            payload = payload.tobytes()
        if isinstance(payload, bytearray):
            payload = bytes(payload)
        if not isinstance(payload, bytes):
            raise ValueError(f"pixel projection payload is not bytes: {key}")
        pixels_by_key[key] = payload

    expected_pixel_keys = {
        key for key in metadata_keys if key[3] in DECODED_HISTORY_STEPS
    }
    if set(pixels_by_key) != expected_pixel_keys:
        missing = sorted(expected_pixel_keys - set(pixels_by_key))
        extra = sorted(set(pixels_by_key) - expected_pixel_keys)
        raise ValueError(
            "filtered pixel projection does not exactly cover x0/x1/x2 keys: "
            f"missing={missing[:3]}, extra={extra[:3]}"
        )

    merged: list[dict[str, Any]] = []
    for key, row in ordered_metadata:
        combined = dict(row)
        if key[3] in DECODED_HISTORY_STEPS:
            combined["pixels"] = pixels_by_key[key]
        merged.append(combined)
    return merged


def _read_lance_rows(path: Path, *, expected_split: str) -> list[dict[str, Any]]:
    forbidden = _forbidden_closed_component(path)
    if forbidden is not None:
        raise ValueError(
            "refusing validation/Public path component before Lance open: "
            f"{forbidden!r}"
        )
    if expected_split not in ACTIVE_SPLITS:
        raise ValueError(f"inactive or Public split refused: {expected_split!r}")
    expected_name = TABLE_NAMES[expected_split]
    if path.name != expected_name:
        raise ValueError(
            f"refusing non-authorized table {path.name!r}; expected {expected_name!r}"
        )
    dataset = lance.dataset(str(path))
    missing = sorted(set(TABLE_COLUMNS) - set(dataset.schema.names))
    if missing:
        raise ValueError(f"{expected_name}: missing required columns {missing}")
    metadata_table = dataset.to_table(columns=list(METADATA_ACTION_COLUMNS))
    pixel_table = dataset.to_table(
        columns=list(PIXEL_JOIN_COLUMNS),
        filter=PIXEL_FILTER,
    )
    return _merge_lance_projections(
        metadata_table.to_pylist(),
        pixel_table.to_pylist(),
    )


def run_probe(artifact_root: Path) -> dict[str, Any]:
    _, tables = resolve_allowed_tables(artifact_root)
    rows = {
        split: _read_lance_rows(path, expected_split=split)
        for split, path in tables.items()
    }
    report = evaluate_fixture_rows(
        rows["train"],
        rows["loader_validation"],
        bootstrap_resamples=BOOTSTRAP_RESAMPLES,
        bootstrap_seed=BOOTSTRAP_SEED,
        permutation_repetitions=PERMUTATION_REPETITIONS,
        permutation_seed=PERMUTATION_SEED,
    )
    report["inputs"] = {
        "artifact_root_path_recorded": False,
        "only_authorized_lance_tables_opened": list(TABLE_NAMES.values()),
        "manifest_or_build_report_read": False,
        "validation_or_public_table_read": False,
        "tables": {
            split: {
                "relative_path": path.name,
                "table_directory_hashed": False,
                "projections": [
                    {
                        "columns": list(METADATA_ACTION_COLUMNS),
                        "filter": None,
                        "row_scope": "all_four_model_steps",
                    },
                    {
                        "columns": list(PIXEL_JOIN_COLUMNS),
                        "filter": PIXEL_FILTER,
                        "row_scope": "x0_x1_x2_only",
                    },
                ],
                "x3_pixel_bytes_read": False,
                "rows_read": len(rows[split]),
            }
            for split, path in tables.items()
        },
    }
    return report


def _reject_forbidden_cli(values: Sequence[str]) -> None:
    forbidden_prefixes = (
        "--validation",
        "--public",
        "--test",
    )
    for value in values:
        option = value.split("=", 1)[0].lower()
        if option.startswith(forbidden_prefixes):
            raise ValueError(
                "Cube v3 RGB-history probe explicitly refuses validation/Public "
                "Test options"
            )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    values = list(sys.argv[1:] if argv is None else argv)
    _reject_forbidden_cli(values)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(values)


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    if path.suffix.lower() != ".json":
        raise ValueError("probe output must use a .json filename")
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output = Path(os.path.abspath(args.output.expanduser()))
    if output.suffix.lower() != ".json":
        raise ValueError("probe output must use a .json filename")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    artifact_root = args.artifact_root.expanduser().resolve()
    if _is_relative_to(output, artifact_root):
        raise ValueError("probe output must remain outside the immutable artifact root")
    report = run_probe(args.artifact_root)
    _write_json_exclusive(output, report)
    print(
        json.dumps(
            {
                "output": str(output),
                "status": report["status"],
                "passed": report["passed"],
                "public_test_read": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
