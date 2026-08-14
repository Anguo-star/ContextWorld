#!/usr/bin/env python3
"""Audit frozen Cube v3 action support without opening Development/Public data.

The audit enumerates the deterministic action-profile factory for exactly the
frozen Training and loader-Development counts.  It combines two quantities in
the same population-standardized 15-step action metric:

* the frozen nearest-original-H5 distance for each of the four anchors; and
* the exact concrete-profile-to-anchor distance computed here.

Their sum is a conservative upper bound on the concrete profile's distance to
the anchor's nearest H5 window.  The standardization vector and original-H5
gripper maximum are recomputed from the upstream H5 ``action`` dataset, whose
canonical content hash is recorded.  No Lance table, simulator, or Public Test
artifact is accepted as an input.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contextworld.evaluation.cube_grasp_rule_h3_v3 import (  # noqa: E402
    V3_ACTION_ANCHORS,
    V3_ANCHOR_GRIPPER_ACTIONS,
    V3_ANCHOR_PROFILES,
    action_blocks,
    action_profile_content_sha256,
    make_v3_action_profile,
    validate_v3_action_profile,
)


AUDIT_ID = "cube_gripper_carry_h3_v3_action_support_v1"
FEASIBILITY_EVIDENCE_SOURCE_SYMBOL = "frozen_action_template_feasibility_json"
ACTIVE_SPLITS = ("train", "loader_validation")
FROZEN_PROFILE_COUNTS = {"train": 2048, "loader_validation": 256}
ACTION_PROFILE_SHAPE = (4, 5, 5)
EFFECTIVE_ACTION_SHAPE = (15, 5)
MOMENT_WEIGHTS = np.asarray([4.0, 3.0, 2.0, 1.0, 0.0])

ORIGINAL_H5_ACTION_DATASET = "action"
ORIGINAL_H5_SOURCE_SYMBOL = "upstream_cube_single_expert_h5"
FROZEN_ORIGINAL_H5_ROW_COUNT = 2_010_000
FROZEN_ORIGINAL_H5_FINITE_ACTION_ROWS = 2_000_000
FROZEN_ORIGINAL_H5_EXCLUDED_NONFINITE_ROWS = 10_000
FROZEN_ORIGINAL_H5_ACTION_DATASET_SHA256 = (
    "e94371078958ee8ad62edba91435841714c293ea9684ae6523a03574793faa40"
)
FROZEN_ORIGINAL_H5_FINITE_ACTION_CONTENT_SHA256 = (
    "ea2cf67d4b7500981298499d76de0129bdc46e50da05e160f81d12e24edfc80b"
)
FROZEN_ORIGINAL_H5_METRIC_STD_POPULATION_FLOAT32_SHA256 = (
    "3929f3ba78d594bb372f48f14c588e0f42c8f33e83b6a2312b3f872d1e222954"
)
H5_STATS_CHUNK_ROWS = 100_000
ENVIRONMENT_ACTION_ABSOLUTE_MAXIMUM = 1.0
SUPPORT_NRMSE_MAXIMUM = 0.5

_SHA256_PATTERN = re.compile(r"[0-9a-fA-F]{64}")
_FORBIDDEN_PUBLIC_OPTIONS = {
    "--validation",
    "--validation-lance",
    "--validation-pairs",
    "--public",
    "--public-test",
    "--public-test-lance",
    "--public-test-pairs",
    "--test",
    "--test-pairs",
}
_PROFILE_INVARIANT_CHECKS = (
    "factory_profile_audit_passed",
    "factory_split_matches_requested_split",
    "factory_catalog_index_matches",
    "finite_float32_actions",
    "probe_sum_exact_zero",
    "probe_last_exact_zero",
    "probe_moment_exact_one",
    "environment_action_bounds_passed",
    "gripper_at_or_below_original_h5_maximum",
    "terminal_format_block_zero",
    "profile_content_id_matches_float32_blocks",
)


@dataclass(frozen=True)
class AnchorEvidence:
    """Frozen original-H5 support evidence for one v3 action anchor."""

    anchor_id: str
    source_key: str
    nearest_original_h5_nrmse: float
    h5_window_support_count_at_or_below_0p5: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_key": self.source_key,
            "nearest_original_h5_nrmse": self.nearest_original_h5_nrmse,
            "h5_window_support_count_at_or_below_nrmse_0p5": (
                self.h5_window_support_count_at_or_below_0p5
            ),
        }


@dataclass(frozen=True)
class OriginalH5ActionStats:
    """Recomputed normalization and provenance for the H5 action dataset."""

    file_size_bytes: int
    dataset: str
    dataset_shape: tuple[int, int]
    dataset_dtype: str
    dataset_chunks: tuple[int, int] | None
    total_rows: int
    finite_action_rows: int
    excluded_nonfinite_rows: int
    population_mean_float64: tuple[float, ...]
    population_std_float64: tuple[float, ...]
    metric_mean_float32: tuple[float, ...]
    metric_std_population_float32: tuple[float, ...]
    gripper_maximum: float
    action_dataset_sha256: str
    finite_action_content_sha256: str
    population_std_float64_sha256: str
    metric_std_population_float32_sha256: str

    @property
    def metric_std(self) -> np.ndarray:
        return np.asarray(self.metric_std_population_float32, dtype=np.float64)

    @property
    def checks(self) -> dict[str, bool]:
        return {
            "action_dataset_is_float32_width_five": (
                self.dataset_dtype == "float32"
                and self.dataset_shape[1] == EFFECTIVE_ACTION_SHAPE[1]
            ),
            "frozen_total_row_count_exact": (
                self.total_rows == FROZEN_ORIGINAL_H5_ROW_COUNT
            ),
            "frozen_finite_action_row_count_exact": (
                self.finite_action_rows
                == FROZEN_ORIGINAL_H5_FINITE_ACTION_ROWS
            ),
            "frozen_excluded_nonfinite_row_count_exact": (
                self.excluded_nonfinite_rows
                == FROZEN_ORIGINAL_H5_EXCLUDED_NONFINITE_ROWS
            ),
            "action_dataset_sha256_matches_frozen": (
                self.action_dataset_sha256
                == FROZEN_ORIGINAL_H5_ACTION_DATASET_SHA256
            ),
            "finite_action_content_sha256_matches_frozen": (
                self.finite_action_content_sha256
                == FROZEN_ORIGINAL_H5_FINITE_ACTION_CONTENT_SHA256
            ),
            "metric_std_population_float32_sha256_matches_frozen": (
                self.metric_std_population_float32_sha256
                == FROZEN_ORIGINAL_H5_METRIC_STD_POPULATION_FLOAT32_SHA256
            ),
            "population_statistics_finite": bool(
                np.isfinite(self.population_mean_float64).all()
                and np.isfinite(self.population_std_float64).all()
            ),
            "population_standard_deviations_positive": bool(
                np.asarray(self.population_std_float64, dtype=np.float64).min()
                > 0.0
            ),
            "gripper_maximum_finite": math.isfinite(self.gripper_maximum),
        }

    @property
    def passed(self) -> bool:
        return all(self.checks.values())

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_symbol": ORIGINAL_H5_SOURCE_SYMBOL,
            "source_path_is_not_identity": True,
            "path_recorded": False,
            "file_size_bytes": self.file_size_bytes,
            "dataset": self.dataset,
            "dataset_shape": list(self.dataset_shape),
            "dataset_dtype": self.dataset_dtype,
            "dataset_chunks": (
                list(self.dataset_chunks) if self.dataset_chunks is not None else None
            ),
            "total_rows": self.total_rows,
            "finite_action_rows": self.finite_action_rows,
            "excluded_nonfinite_rows": self.excluded_nonfinite_rows,
            "population_mean_float64": list(self.population_mean_float64),
            "population_std_float64": list(self.population_std_float64),
            "metric_mean_float32": list(self.metric_mean_float32),
            "metric_std_population_float32": list(
                self.metric_std_population_float32
            ),
            "gripper_maximum": self.gripper_maximum,
            "hashes": {
                "action_dataset_sha256": self.action_dataset_sha256,
                "finite_action_content_sha256": (
                    self.finite_action_content_sha256
                ),
                "population_std_float64_sha256": (
                    self.population_std_float64_sha256
                ),
                "metric_std_population_float32_sha256": (
                    self.metric_std_population_float32_sha256
                ),
                "canonicalization": (
                    "SHA256 of row-major little-endian numeric bytes; action "
                    "hashes use float32 and std hashes use their named dtype"
                ),
            },
            "expected_hashes": {
                "action_dataset_sha256": (
                    FROZEN_ORIGINAL_H5_ACTION_DATASET_SHA256
                ),
                "finite_action_content_sha256": (
                    FROZEN_ORIGINAL_H5_FINITE_ACTION_CONTENT_SHA256
                ),
                "metric_std_population_float32_sha256": (
                    FROZEN_ORIGINAL_H5_METRIC_STD_POPULATION_FLOAT32_SHA256
                ),
            },
            "checks": self.checks,
            "passed": self.passed,
        }


def _canonical_numeric_bytes(values: np.ndarray, *, dtype: str) -> bytes:
    return np.ascontiguousarray(values, dtype=np.dtype(dtype)).tobytes(order="C")


def compute_original_h5_action_population_stats(
    path: Path | str,
    *,
    chunk_rows: int = H5_STATS_CHUNK_ROWS,
) -> OriginalH5ActionStats:
    """Stream the original H5 action data and recompute its ddof=0 stats.

    Only the ``action`` dataset is opened.  Statistics use float64 accumulation;
    the metric vector is the canonical float32 serialization of the recomputed
    population standard deviation, matching the frozen action-normalization
    representation used by the anchor evidence.
    """

    if isinstance(chunk_rows, bool) or not isinstance(chunk_rows, int):
        raise ValueError("chunk_rows must be a positive integer")
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be a positive integer")
    source_path = Path(path).expanduser().resolve()
    file_size_bytes = source_path.stat().st_size
    all_action_digest = hashlib.sha256()
    finite_action_digest = hashlib.sha256()
    running_count = 0
    running_mean = np.zeros(EFFECTIVE_ACTION_SHAPE[1], dtype=np.float64)
    running_m2 = np.zeros(EFFECTIVE_ACTION_SHAPE[1], dtype=np.float64)
    gripper_maximum = -math.inf

    with h5py.File(source_path, "r", swmr=True) as handle:
        if ORIGINAL_H5_ACTION_DATASET not in handle:
            raise ValueError(
                f"Original H5 has no {ORIGINAL_H5_ACTION_DATASET!r} dataset"
            )
        dataset = handle[ORIGINAL_H5_ACTION_DATASET]
        if not isinstance(dataset, h5py.Dataset):
            raise ValueError("Original H5 action entry is not a dataset")
        if len(dataset.shape) != 2 or dataset.shape[1] != EFFECTIVE_ACTION_SHAPE[1]:
            raise ValueError("Original H5 action dataset must have shape [N,5]")
        dataset_shape = (int(dataset.shape[0]), int(dataset.shape[1]))
        dataset_dtype = str(dataset.dtype)
        dataset_chunks = (
            tuple(int(value) for value in dataset.chunks)
            if dataset.chunks is not None
            else None
        )
        for start in range(0, dataset_shape[0], chunk_rows):
            stop = min(start + chunk_rows, dataset_shape[0])
            chunk = np.asarray(dataset[start:stop])
            if chunk.shape != (stop - start, EFFECTIVE_ACTION_SHAPE[1]):
                raise ValueError("Original H5 action dataset changed while reading")
            canonical_chunk = np.ascontiguousarray(chunk, dtype=np.dtype("<f4"))
            all_action_digest.update(canonical_chunk.tobytes(order="C"))
            finite_mask = np.isfinite(chunk).all(axis=1)
            if not finite_mask.any():
                continue
            finite_f32 = np.ascontiguousarray(
                chunk[finite_mask], dtype=np.dtype("<f4")
            )
            finite_action_digest.update(finite_f32.tobytes(order="C"))
            finite = finite_f32.astype(np.float64)
            batch_count = int(finite.shape[0])
            batch_mean = np.mean(finite, axis=0, dtype=np.float64)
            centered = finite - batch_mean[None, :]
            batch_m2 = np.sum(centered * centered, axis=0, dtype=np.float64)
            combined_count = running_count + batch_count
            delta = batch_mean - running_mean
            running_m2 += (
                batch_m2
                + delta
                * delta
                * (float(running_count) * float(batch_count) / combined_count)
            )
            running_mean += delta * (float(batch_count) / combined_count)
            running_count = combined_count
            gripper_maximum = max(
                gripper_maximum, float(np.max(finite[:, 4]))
            )

    if running_count == 0:
        raise ValueError("Original H5 action dataset has no fully finite rows")
    population_variance = running_m2 / float(running_count)
    if not np.isfinite(population_variance).all() or np.any(
        population_variance <= 0.0
    ):
        raise ValueError(
            "Original H5 finite actions must have positive finite variance "
            "on every axis"
        )
    population_std = np.sqrt(population_variance)
    metric_mean = running_mean.astype(np.float32)
    metric_std = population_std.astype(np.float32)
    total_rows = dataset_shape[0]
    excluded_nonfinite_rows = total_rows - running_count
    return OriginalH5ActionStats(
        file_size_bytes=int(file_size_bytes),
        dataset=ORIGINAL_H5_ACTION_DATASET,
        dataset_shape=dataset_shape,
        dataset_dtype=dataset_dtype,
        dataset_chunks=dataset_chunks,
        total_rows=total_rows,
        finite_action_rows=running_count,
        excluded_nonfinite_rows=excluded_nonfinite_rows,
        population_mean_float64=tuple(float(value) for value in running_mean),
        population_std_float64=tuple(float(value) for value in population_std),
        metric_mean_float32=tuple(float(value) for value in metric_mean),
        metric_std_population_float32=tuple(
            float(value) for value in metric_std
        ),
        gripper_maximum=float(gripper_maximum),
        action_dataset_sha256=all_action_digest.hexdigest(),
        finite_action_content_sha256=finite_action_digest.hexdigest(),
        population_std_float64_sha256=hashlib.sha256(
            _canonical_numeric_bytes(population_std, dtype="<f8")
        ).hexdigest(),
        metric_std_population_float32_sha256=hashlib.sha256(
            _canonical_numeric_bytes(metric_std, dtype="<f4")
        ).hexdigest(),
    )


def _mapping(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Evidence field {field} must be a mapping")
    return value


def _string_list(value: Any, *, field: str) -> list[str]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise ValueError(f"Evidence field {field} must be a list of strings")
    return list(value)


def _validate_expected_sha256(value: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError("Expected feasibility-evidence SHA256 must be 64 hex digits")
    return value.lower()


def _evidence_scope(payload: Mapping[str, Any]) -> dict[str, Any]:
    scope = _mapping(payload.get("scope"), field="scope")
    data_read = _string_list(scope.get("data_read"), field="scope.data_read")
    explicitly_not_read = _string_list(
        scope.get("explicitly_not_read"), field="scope.explicitly_not_read"
    )
    read_text = "\n".join(data_read).lower()
    forbidden_reads = (
        "loader_validation",
        "validation.lance",
        "public test",
        "public_test",
    )
    contaminated = [value for value in forbidden_reads if value in read_text]
    if contaminated:
        raise ValueError(
            "Feasibility evidence declares Development/Public data reads: "
            + ", ".join(contaminated)
        )
    not_read_text = "\n".join(explicitly_not_read).lower()
    if "loader_validation.lance" not in not_read_text or not (
        "validation.lance" in not_read_text
        and ("public test" in not_read_text or "public_test" in not_read_text)
    ):
        raise ValueError(
            "Feasibility evidence must explicitly state that loader_validation "
            "and validation.lance/Public Test were not read"
        )
    return {
        "data_read": data_read,
        "explicitly_not_read": explicitly_not_read,
        "development_or_public_data_read": False,
    }


def _match_anchor_evidence(
    payload: Mapping[str, Any],
) -> dict[str, AnchorEvidence]:
    templates = _mapping(payload.get("templates"), field="templates")
    matched: dict[str, AnchorEvidence] = {}
    used_source_keys: set[str] = set()
    for anchor_id in V3_ACTION_ANCHORS:
        expected_profile = np.asarray(
            V3_ANCHOR_PROFILES[anchor_id], dtype=np.float32
        )
        expected_gripper = np.float32(V3_ANCHOR_GRIPPER_ACTIONS[anchor_id])
        candidates: list[tuple[str, Mapping[str, Any]]] = []
        for key, untyped_row in templates.items():
            if not isinstance(key, str) or not isinstance(untyped_row, dict):
                continue
            row = untyped_row
            try:
                profile = np.asarray(row.get("profile"), dtype=np.float32)
                gripper = np.float32(row.get("gripper"))
            except (TypeError, ValueError):
                continue
            if (
                profile.shape == (5,)
                and np.array_equal(profile, expected_profile)
                and gripper == expected_gripper
            ):
                candidates.append((key, row))
        if len(candidates) != 1:
            raise ValueError(
                f"Expected exactly one frozen evidence row for {anchor_id}, "
                f"found {len(candidates)}"
            )
        source_key, row = candidates[0]
        if source_key in used_source_keys:
            raise ValueError(f"Evidence row {source_key!r} matched multiple anchors")
        used_source_keys.add(source_key)
        nearest = row.get("support_nearest_nrmse")
        support_count = row.get("support_count_nrmse_le_0p5")
        if (
            isinstance(nearest, bool)
            or not isinstance(nearest, (int, float))
            or not math.isfinite(float(nearest))
            or float(nearest) < 0.0
        ):
            raise ValueError(f"{source_key}: invalid support_nearest_nrmse")
        if (
            isinstance(support_count, bool)
            or not isinstance(support_count, int)
            or support_count < 0
        ):
            raise ValueError(f"{source_key}: invalid support_count_nrmse_le_0p5")
        matched[anchor_id] = AnchorEvidence(
            anchor_id=anchor_id,
            source_key=source_key,
            nearest_original_h5_nrmse=float(nearest),
            h5_window_support_count_at_or_below_0p5=int(support_count),
        )
    return matched


def _load_frozen_evidence(
    path: Path, *, expected_sha256: str
) -> tuple[dict[str, Any], dict[str, AnchorEvidence], dict[str, Any]]:
    evidence_path = path.expanduser().resolve()
    expected = _validate_expected_sha256(expected_sha256)
    evidence_bytes = evidence_path.read_bytes()
    observed = hashlib.sha256(evidence_bytes).hexdigest()
    if observed != expected:
        raise ValueError(
            "Feasibility-evidence SHA256 mismatch: "
            f"expected={expected}, observed={observed}"
        )
    payload = json.loads(evidence_bytes)
    if not isinstance(payload, dict):
        raise ValueError("Feasibility evidence must contain a JSON object")
    scope = _evidence_scope(payload)
    support_definition = _mapping(
        payload.get("support_definition"), field="support_definition"
    )
    joint_definition = support_definition.get("joint")
    same_episode_windows = support_definition.get("same_episode_windows")
    if not isinstance(joint_definition, str) or not all(
        token in joint_definition.lower()
        for token in ("standardized rmse", "75", "15-step", "same-episode")
    ):
        raise ValueError("Evidence does not declare the frozen joint-15-step metric")
    if (
        isinstance(same_episode_windows, bool)
        or not isinstance(same_episode_windows, int)
        or same_episode_windows <= 0
    ):
        raise ValueError("Evidence has no positive same_episode_windows count")
    anchors = _match_anchor_evidence(payload)
    receipt = {
        "source_symbol": FEASIBILITY_EVIDENCE_SOURCE_SYMBOL,
        "path_recorded": False,
        "size_bytes": len(evidence_bytes),
        "expected_sha256": expected,
        "observed_sha256": observed,
        "sha256_verified": True,
        "scope": scope,
        "support_definition": support_definition,
        "joint_15_step_standardized_metric_contract_validated": True,
        "all_causal_audits_passed": bool(
            payload.get("all_causal_audits_passed") is True
        ),
        "anchors": {
            anchor_id: anchors[anchor_id].as_dict()
            for anchor_id in V3_ACTION_ANCHORS
        },
    }
    return payload, anchors, receipt


def _effective_anchor_actions(anchor_id: str) -> np.ndarray:
    if anchor_id not in V3_ACTION_ANCHORS:
        raise ValueError(f"Unknown v3 action anchor {anchor_id!r}")
    probe = np.asarray(V3_ANCHOR_PROFILES[anchor_id], dtype=np.float64)
    actions = np.zeros(EFFECTIVE_ACTION_SHAPE, dtype=np.float64)
    actions[:, 2] = np.concatenate((probe, -probe, probe))
    actions[:, 4] = float(V3_ANCHOR_GRIPPER_ACTIONS[anchor_id])
    return actions


def standardized_joint_15_step_nrmse(
    left: np.ndarray,
    right: np.ndarray,
    *,
    action_std_population: Sequence[float],
) -> float:
    """Return RMSE over all 75 population-standardized action values."""

    left_value = np.asarray(left, dtype=np.float64)
    right_value = np.asarray(right, dtype=np.float64)
    std = np.asarray(action_std_population, dtype=np.float64)
    if left_value.shape != EFFECTIVE_ACTION_SHAPE or right_value.shape != (
        EFFECTIVE_ACTION_SHAPE
    ):
        raise ValueError(
            "The frozen support metric requires two finite [15,5] sequences"
        )
    if not np.isfinite(left_value).all() or not np.isfinite(right_value).all():
        raise ValueError("The frozen support metric requires finite sequences")
    if (
        std.shape != (EFFECTIVE_ACTION_SHAPE[1],)
        or not np.isfinite(std).all()
        or np.any(std <= 0.0)
    ):
        raise ValueError(
            "The frozen support metric requires five positive finite scales"
        )
    standardized = (left_value - right_value) / std[None, :]
    return float(np.sqrt(np.mean(np.square(standardized))))


def _profile_record(
    *,
    split: str,
    catalog_index: int,
    anchor_evidence: Mapping[str, AnchorEvidence],
    original_h5_stats: OriginalH5ActionStats,
) -> dict[str, Any]:
    profile = make_v3_action_profile(split=split, catalog_index=catalog_index)
    blocks = np.asarray(action_blocks(profile))
    profile_audit = validate_v3_action_profile(profile)
    if blocks.shape != ACTION_PROFILE_SHAPE or blocks.dtype != np.float32:
        raise RuntimeError("v3 profile factory returned noncanonical action blocks")
    probe = np.asarray(profile.probe_profile, dtype=np.float32)
    anchor_id = str(profile.action_anchor_id)
    if anchor_id not in anchor_evidence:
        raise RuntimeError(f"Profile references unknown anchor {anchor_id!r}")
    evidence = anchor_evidence[anchor_id]
    effective = blocks[:3].reshape(EFFECTIVE_ACTION_SHAPE).astype(np.float64)
    concrete_to_anchor = standardized_joint_15_step_nrmse(
        effective,
        _effective_anchor_actions(anchor_id),
        action_std_population=original_h5_stats.metric_std,
    )
    conservative_upper_bound = (
        evidence.nearest_original_h5_nrmse + concrete_to_anchor
    )
    profile_sum = float(np.sum(probe.astype(np.float64)))
    profile_last = float(probe[-1])
    profile_moment = float(
        np.dot(MOMENT_WEIGHTS, probe.astype(np.float64))
    )
    content_id = action_profile_content_sha256(blocks)
    finite = bool(np.isfinite(blocks).all())
    action_absolute_maximum = (
        float(np.max(np.abs(blocks))) if finite else math.inf
    )
    checks = {
        "factory_profile_audit_passed": bool(profile_audit.get("passed")),
        "factory_split_matches_requested_split": profile.split == split,
        "factory_catalog_index_matches": int(profile.catalog_index)
        == int(catalog_index),
        "finite_float32_actions": finite,
        "probe_sum_exact_zero": profile_sum == 0.0,
        "probe_last_exact_zero": profile_last == 0.0,
        "probe_moment_exact_one": profile_moment == 1.0,
        "environment_action_bounds_passed": action_absolute_maximum
        <= ENVIRONMENT_ACTION_ABSOLUTE_MAXIMUM,
        "gripper_at_or_below_original_h5_maximum": float(
            profile.gripper_action
        )
        <= original_h5_stats.gripper_maximum,
        "terminal_format_block_zero": bool(np.count_nonzero(blocks[3]) == 0),
        "profile_content_id_matches_float32_blocks": (
            profile.action_profile_id == content_id
        ),
        "anchor_has_original_h5_support_windows": (
            evidence.h5_window_support_count_at_or_below_0p5 > 0
        ),
        "conservative_original_h5_nrmse_upper_bound_passed": (
            conservative_upper_bound <= SUPPORT_NRMSE_MAXIMUM
        ),
    }
    return {
        "split": split,
        "catalog_index": int(catalog_index),
        "action_anchor_id": anchor_id,
        "action_profile_id": content_id,
        "measurements": {
            "absolute_probe_sum_residual": abs(profile_sum),
            "absolute_probe_last_residual": abs(profile_last),
            "absolute_probe_moment_residual": abs(profile_moment - 1.0),
            "action_absolute_maximum": action_absolute_maximum,
            "gripper_action": float(profile.gripper_action),
            "concrete_to_anchor_standardized_nrmse": concrete_to_anchor,
            "anchor_nearest_original_h5_nrmse": (
                evidence.nearest_original_h5_nrmse
            ),
            "conservative_original_h5_nrmse_upper_bound": (
                conservative_upper_bound
            ),
            "anchor_h5_window_support_count_at_or_below_nrmse_0p5": (
                evidence.h5_window_support_count_at_or_below_0p5
            ),
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def _profile_ids_sha256(profile_ids: Sequence[str]) -> str:
    canonical = "\n".join(sorted(profile_ids)) + "\n"
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


def _group_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("Cannot summarize an empty profile group")
    profile_ids = [str(row["action_profile_id"]) for row in records]
    anchor_counts = {
        anchor_id: sum(
            row["action_anchor_id"] == anchor_id for row in records
        )
        for anchor_id in V3_ACTION_ANCHORS
    }

    def maximum(name: str) -> float:
        return max(float(row["measurements"][name]) for row in records)

    support_counts = {
        anchor_id: {
            int(
                row["measurements"][
                    "anchor_h5_window_support_count_at_or_below_nrmse_0p5"
                ]
            )
            for row in records
            if row["action_anchor_id"] == anchor_id
        }
        for anchor_id in V3_ACTION_ANCHORS
    }
    normalized_support_counts = {
        anchor_id: next(iter(values)) if len(values) == 1 else None
        for anchor_id, values in support_counts.items()
        if values
    }
    failed = [str(row["action_profile_id"]) for row in records if not row["passed"]]
    return {
        "profile_count": len(records),
        "unique_profile_count": len(set(profile_ids)),
        "profile_ids_sha256": _profile_ids_sha256(profile_ids),
        "passed_profile_count": sum(bool(row["passed"]) for row in records),
        "conservatively_supported_profile_count": sum(
            bool(
                row["checks"][
                    "conservative_original_h5_nrmse_upper_bound_passed"
                ]
            )
            for row in records
        ),
        "action_anchor_counts": anchor_counts,
        "anchor_h5_window_support_count_at_or_below_nrmse_0p5": (
            normalized_support_counts
        ),
        "maxima": {
            "absolute_probe_sum_residual": maximum(
                "absolute_probe_sum_residual"
            ),
            "absolute_probe_last_residual": maximum(
                "absolute_probe_last_residual"
            ),
            "absolute_probe_moment_residual": maximum(
                "absolute_probe_moment_residual"
            ),
            "action_absolute_maximum": maximum("action_absolute_maximum"),
            "gripper_action": maximum("gripper_action"),
            "concrete_to_anchor_standardized_nrmse": maximum(
                "concrete_to_anchor_standardized_nrmse"
            ),
            "anchor_nearest_original_h5_nrmse": maximum(
                "anchor_nearest_original_h5_nrmse"
            ),
            "conservative_original_h5_nrmse_upper_bound": maximum(
                "conservative_original_h5_nrmse_upper_bound"
            ),
        },
        "failed_profile_ids": failed,
        "passed": not failed,
    }


def audit_action_support(
    feasibility_evidence: Path | str,
    *,
    feasibility_evidence_sha256: str,
    original_h5: Path | str,
) -> dict[str, Any]:
    """Run the fixed 2,304-profile Development-only action-support audit."""

    evidence_path = Path(feasibility_evidence)
    _, anchor_evidence, evidence_receipt = _load_frozen_evidence(
        evidence_path,
        expected_sha256=feasibility_evidence_sha256,
    )
    original_h5_stats = compute_original_h5_action_population_stats(original_h5)
    records_by_split = {
        split: [
            _profile_record(
                split=split,
                catalog_index=index,
                anchor_evidence=anchor_evidence,
                original_h5_stats=original_h5_stats,
            )
            for index in range(FROZEN_PROFILE_COUNTS[split])
        ]
        for split in ACTIVE_SPLITS
    }
    split_summaries = {
        split: _group_summary(records)
        for split, records in records_by_split.items()
    }
    all_records = [
        row for split in ACTIVE_SPLITS for row in records_by_split[split]
    ]
    overall = _group_summary(all_records)
    per_anchor = {
        anchor_id: _group_summary(
            [row for row in all_records if row["action_anchor_id"] == anchor_id]
        )
        for anchor_id in V3_ACTION_ANCHORS
    }

    train_profile_ids = {
        str(row["action_profile_id"]) for row in records_by_split["train"]
    }
    development_profile_ids = {
        str(row["action_profile_id"])
        for row in records_by_split["loader_validation"]
    }
    profile_overlap = sorted(train_profile_ids & development_profile_ids)
    train_anchors = {
        str(row["action_anchor_id"]) for row in records_by_split["train"]
    }
    development_anchors = {
        str(row["action_anchor_id"])
        for row in records_by_split["loader_validation"]
    }
    anchor_overlap = sorted(train_anchors & development_anchors)
    expected_anchors = sorted(V3_ACTION_ANCHORS)
    cross_split_checks = {
        "profile_content_overlap_zero": not profile_overlap,
        "four_shared_anchor_families": anchor_overlap == expected_anchors,
    }
    cross_split = {
        "policy": "shared_families_disjoint_profiles",
        "profile_content_overlap": {
            "count": len(profile_overlap),
            "expected_count": 0,
            "values": profile_overlap,
        },
        "anchor_family_overlap": {
            "count": len(anchor_overlap),
            "expected_count": 4,
            "values": anchor_overlap,
            "expected_values": expected_anchors,
        },
        "checks": cross_split_checks,
        "passed": all(cross_split_checks.values()),
    }

    expected_per_anchor = {
        split: FROZEN_PROFILE_COUNTS[split] // len(V3_ACTION_ANCHORS)
        for split in ACTIVE_SPLITS
    }
    checks = {
        "feasibility_evidence_sha256_verified": bool(
            evidence_receipt["sha256_verified"]
        ),
        "feasibility_causal_evidence_passed": bool(
            evidence_receipt["all_causal_audits_passed"]
        ),
        "original_h5_action_source_and_population_stats_passed": (
            original_h5_stats.passed
        ),
        "anchor_and_concrete_distances_share_standardized_metric_contract": bool(
            evidence_receipt[
                "joint_15_step_standardized_metric_contract_validated"
            ]
            and original_h5_stats.passed
        ),
        "only_training_and_development_profiles_enumerated": (
            tuple(records_by_split) == ACTIVE_SPLITS
        ),
        "frozen_profile_counts_exact": all(
            len(records_by_split[split]) == FROZEN_PROFILE_COUNTS[split]
            for split in ACTIVE_SPLITS
        ),
        "four_anchor_balance_exact_per_split": all(
            split_summaries[split]["action_anchor_counts"]
            == {
                anchor_id: expected_per_anchor[split]
                for anchor_id in V3_ACTION_ANCHORS
            }
            for split in ACTIVE_SPLITS
        ),
        "all_profile_invariants_and_action_bounds_passed": all(
            all(bool(row["checks"][name]) for name in _PROFILE_INVARIANT_CHECKS)
            for row in all_records
        ),
        "all_anchors_have_original_h5_support_windows": all(
            bool(row["checks"]["anchor_has_original_h5_support_windows"])
            for row in all_records
        ),
        "all_profiles_conservatively_within_original_h5_support_gate": (
            overall["conservatively_supported_profile_count"]
            == overall["profile_count"]
        ),
        "profile_content_overlap_zero": not profile_overlap,
        "anchor_family_overlap_four": anchor_overlap == expected_anchors,
        "no_public_test_input_or_generation": True,
    }
    passed = all(checks.values())
    support_counts = {
        anchor_id: anchor_evidence[
            anchor_id
        ].h5_window_support_count_at_or_below_0p5
        for anchor_id in V3_ACTION_ANCHORS
    }
    return {
        "schema_version": 1,
        "audit_id": AUDIT_ID,
        "status": "passed" if passed else "failed",
        "scope": {
            "phase": "development_only",
            "active_splits": list(ACTIVE_SPLITS),
            "frozen_profile_counts": dict(FROZEN_PROFILE_COUNTS),
            "total_concrete_profiles": sum(FROZEN_PROFILE_COUNTS.values()),
            "profile_source": (
                "contextworld.evaluation.cube_grasp_rule_h3_v3."
                "make_v3_action_profile"
            ),
            "external_data_inputs": [
                "SHA256-verified feasibility evidence JSON",
                "upstream original H5 action dataset only",
            ],
            "original_h5_datasets_opened": [ORIGINAL_H5_ACTION_DATASET],
            "lance_tables_opened": [],
            "public_test_inputs": [],
            "public_test_opened": False,
            "public_test_generated": False,
        },
        "metric": {
            "name": "joint_15_step_population_standardized_nrmse",
            "effective_action_shape": list(EFFECTIVE_ACTION_SHAPE),
            "value_count": 75,
            "normalization_source": (
                "recomputed finite rows of upstream original H5 action "
                "dataset, population ddof=0"
            ),
            "population_accumulation_dtype": "float64",
            "metric_scale_storage_dtype": "float32",
            "action_std_population": list(
                original_h5_stats.metric_std_population_float32
            ),
            "action_std_population_sha256": (
                original_h5_stats.metric_std_population_float32_sha256
            ),
            "formula": "sqrt(mean(((concrete-anchor)/std_population)^2))",
            "triangle_upper_bound": (
                "anchor_nearest_original_h5_nrmse + "
                "concrete_to_anchor_standardized_nrmse"
            ),
            "conservative_upper_bound_maximum": SUPPORT_NRMSE_MAXIMUM,
            "compatibility_basis": (
                "The SHA256-bound evidence declares the original-H5 joint "
                "15-step standardized RMSE, and concrete-to-anchor distances "
                "use the population scale recomputed from that H5 action "
                "dataset; no unstandardized L2 distance is added."
            ),
        },
        "gates": {
            "probe_sum": 0.0,
            "probe_last": 0.0,
            "probe_moment": 1.0,
            "environment_action_absolute_maximum": (
                ENVIRONMENT_ACTION_ABSOLUTE_MAXIMUM
            ),
            "original_h5_gripper_maximum": (
                original_h5_stats.gripper_maximum
            ),
            "conservative_original_h5_nrmse_upper_bound_maximum": (
                SUPPORT_NRMSE_MAXIMUM
            ),
            "profile_content_overlap_between_splits": 0,
            "anchor_family_overlap_between_splits": 4,
        },
        "feasibility_evidence": evidence_receipt,
        "original_h5_action": original_h5_stats.as_dict(),
        "support_counts": {
            "concrete_profiles_evaluated": overall["profile_count"],
            "concrete_profiles_conservatively_supported": overall[
                "conservatively_supported_profile_count"
            ],
            "anchor_h5_windows_at_or_below_nrmse_0p5": support_counts,
            "minimum_anchor_h5_window_count": min(support_counts.values()),
            "note": (
                "Anchor H5-window counts are frozen evidence counts, not a "
                "deduplicated union across anchors.  The triangle gate proves "
                "at least one supporting H5 window for each passing concrete "
                "profile."
            ),
        },
        "overall": overall,
        "splits": split_summaries,
        "anchors": per_anchor,
        "cross_split": cross_split,
        "checks": checks,
        "passed": passed,
    }


def _reject_public_arguments(argv: Sequence[str]) -> None:
    for argument in argv:
        option = argument.split("=", 1)[0]
        if option in _FORBIDDEN_PUBLIC_OPTIONS:
            raise ValueError(
                f"{option} is forbidden: this audit accepts no Public Test input"
            )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    values = list(sys.argv[1:] if argv is None else argv)
    _reject_public_arguments(values)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feasibility-evidence", type=Path, required=True)
    parser.add_argument(
        "--feasibility-evidence-sha256",
        required=True,
        help="Expected SHA256 of the frozen feasibility evidence JSON",
    )
    parser.add_argument(
        "--original-h5",
        type=Path,
        required=True,
        help="Upstream Cube H5; only its action dataset is read",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(values)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = audit_action_support(
        args.feasibility_evidence,
        feasibility_evidence_sha256=args.feasibility_evidence_sha256,
        original_h5=args.original_h5,
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
