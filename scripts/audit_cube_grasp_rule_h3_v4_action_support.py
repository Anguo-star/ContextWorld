#!/usr/bin/env python3
"""Audit frozen Cube v4r1 action support without Development/Public access.

The audit is semantically isomorphic to the v3 action-support audit and
enumerates the complete 2x candidate pools: exactly 4,096 Training plus 512
loader-Development profiles.  V4
changes only the hidden ``can_hold`` force coupling, so evidence has two
strictly separated roles:

* the canonical v4 coupling-feasibility JSON justifies selecting 0.40 N and
  contributes no action-support distance or H5-window count; and
* the canonical v3 ``action_template_feasibility_input.json`` remains the
  anchor-to-original-H5 support evidence because every anchor profile and
  gripper value is unchanged in v4.

Concrete v4r1 profile-to-anchor distance is recomputed in the same
population-standardized 15-step metric.  Adding it to the frozen nearest-H5
anchor distance gives a conservative upper bound.  The original-H5 action
hashes, population scale, and gripper gate are identical to the frozen v3
audit.  No Lance table, simulator, validation input, or Public Test artifact
is accepted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contextworld.evaluation.cube_grasp_rule_h3_v4 import (  # noqa: E402
    V4_ACTION_ANCHORS,
    V4_ANCHOR_GRIPPER_ACTIONS,
    V4_ANCHOR_PROFILES,
    V4_FORMAL_CATALOG_INDEX_OFFSET,
    V4R1_FORMAL_CATALOG_INDEX_OFFSET,
    V4_PROFILE_SPLIT_SEEDS,
    V4_PROTOCOL,
    V4_VERTICAL_FORCE_COUPLING_N,
    action_blocks,
    action_profile_content_sha256,
    make_v4_action_profile,
    validate_v4_action_profile,
)
from scripts import audit_cube_grasp_rule_h3_v3_action_support as _v3  # noqa: E402


AUDIT_ID = "cube_gripper_carry_h3_v4r1_action_support_v2"
RECOVERY_AUTHORIZATION_ID = "cube_gripper_carry_h3_development_v4r1"
FORMAL_CATALOG_INDEX_OFFSET = V4R1_FORMAL_CATALOG_INDEX_OFFSET
ACTIVE_SPLITS = ("train", "loader_validation")
FROZEN_PROFILE_COUNTS = {"train": 4096, "loader_validation": 512}
FORMAL_CATALOG_LOCAL_INDEX_POLICY = "zero_based_contiguous_within_each_split"
ACTION_PROFILE_SHAPE = (4, 5, 5)
EFFECTIVE_ACTION_SHAPE = (15, 5)
MOMENT_WEIGHTS = np.asarray([4.0, 3.0, 2.0, 1.0, 0.0])

COUPLING_FEASIBILITY_SOURCE_SYMBOL = "canonical_v4_coupling_feasibility_json"
ANCHOR_SUPPORT_EVIDENCE_SOURCE_SYMBOL = (
    "canonical_v3_action_template_feasibility_input_json"
)
FROZEN_V4_COUPLING_FEASIBILITY_SHA256 = (
    "b9050fb203904bbc0dc8aec2c32e5b950567b1014cb91fb923338f1979cacad7"
)
FROZEN_V3_ANCHOR_SUPPORT_EVIDENCE_SHA256 = (
    "20dd1f7f629f569719886360c6ffca004a44df17e8632e55f7d37a1c400ed055"
)
FROZEN_V4_FAILED_FORMAL_ATTEMPT_RECEIPT_SHA256 = (
    "5f20da08a538f2fd0c72c5c172e64cb2359a2e5bdad1746cf2c4249bbf739936"
)
FROZEN_V3_ANCHOR_SUPPORT_LOGICAL_NAME = (
    "artifacts/evaluation/history3/"
    "cube_gripper_carry_h3_development_v3/"
    "action_template_feasibility_input.json"
)

# These identities are deliberately the same frozen source/action identities
# as v3.  The whole 100-GB H5 hash is recorded as provenance; this audit
# streams and verifies the action dataset hashes rather than rehashing all
# unrelated observation datasets.
FROZEN_ORIGINAL_H5_FILE_SIZE_BYTES = 101_942_558_720
FROZEN_ORIGINAL_H5_FILE_SHA256 = (
    "0664d507c4ff12009010644c9ae950836f954e700c172ccf22e7423af1a55625"
)
ORIGINAL_H5_ACTION_DATASET = _v3.ORIGINAL_H5_ACTION_DATASET
ORIGINAL_H5_SOURCE_SYMBOL = _v3.ORIGINAL_H5_SOURCE_SYMBOL
FROZEN_ORIGINAL_H5_ROW_COUNT = _v3.FROZEN_ORIGINAL_H5_ROW_COUNT
FROZEN_ORIGINAL_H5_FINITE_ACTION_ROWS = (
    _v3.FROZEN_ORIGINAL_H5_FINITE_ACTION_ROWS
)
FROZEN_ORIGINAL_H5_EXCLUDED_NONFINITE_ROWS = (
    _v3.FROZEN_ORIGINAL_H5_EXCLUDED_NONFINITE_ROWS
)
FROZEN_ORIGINAL_H5_ACTION_DATASET_SHA256 = (
    _v3.FROZEN_ORIGINAL_H5_ACTION_DATASET_SHA256
)
FROZEN_ORIGINAL_H5_FINITE_ACTION_CONTENT_SHA256 = (
    _v3.FROZEN_ORIGINAL_H5_FINITE_ACTION_CONTENT_SHA256
)
FROZEN_ORIGINAL_H5_METRIC_STD_POPULATION_FLOAT32_SHA256 = (
    _v3.FROZEN_ORIGINAL_H5_METRIC_STD_POPULATION_FLOAT32_SHA256
)
H5_STATS_CHUNK_ROWS = _v3.H5_STATS_CHUNK_ROWS
ENVIRONMENT_ACTION_ABSOLUTE_MAXIMUM = (
    _v3.ENVIRONMENT_ACTION_ABSOLUTE_MAXIMUM
)
SUPPORT_NRMSE_MAXIMUM = _v3.SUPPORT_NRMSE_MAXIMUM

AnchorEvidence = _v3.AnchorEvidence
OriginalH5ActionStats = _v3.OriginalH5ActionStats
compute_original_h5_action_population_stats = (
    _v3.compute_original_h5_action_population_stats
)
standardized_joint_15_step_nrmse = _v3.standardized_joint_15_step_nrmse

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
_FORBIDDEN_PUBLIC_PATH_COMPONENTS = {
    "validation",
    "validation.lance",
    "public",
    "public_test",
    "public-test",
    "publictest",
}
_PROFILE_INVARIANT_CHECKS = (
    "factory_profile_audit_passed",
    "factory_split_matches_requested_split",
    "factory_catalog_index_matches",
    "factory_split_seed_matches_v4",
    "finite_float32_actions",
    "probe_sum_exact_zero",
    "probe_last_exact_zero",
    "probe_moment_exact_one",
    "environment_action_bounds_passed",
    "gripper_at_or_below_original_h5_maximum",
    "terminal_format_block_zero",
    "profile_content_id_matches_float32_blocks",
)


def _validate_sha256(value: str, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field} must be 64 hexadecimal digits")
    return value.lower()


def _validated_nonpublic_input_path(
    path: Path | str, *, field: str
) -> Path:
    raw = Path(path).expanduser()
    for candidate in (raw, raw.resolve(strict=False)):
        forbidden = next(
            (
                part
                for part in candidate.parts
                if part.lower() in _FORBIDDEN_PUBLIC_PATH_COMPONENTS
            ),
            None,
        )
        if forbidden is not None:
            raise ValueError(
                f"{field} contains forbidden Public path component "
                f"{forbidden!r}"
            )
    try:
        metadata = os.lstat(raw)
    except FileNotFoundError:
        return raw.resolve(strict=False)
    if not os.path.isfile(raw) or os.path.islink(raw):
        raise ValueError(f"{field} must be a real non-symlink file")
    if metadata.st_nlink < 1:
        raise ValueError(f"{field} has invalid filesystem link metadata")
    return raw.resolve(strict=False)


def _read_sha256_json(
    path: Path | str,
    *,
    expected_sha256: str,
    field: str,
) -> tuple[dict[str, Any], bytes, str]:
    expected = _validate_sha256(expected_sha256, field=f"{field} SHA256")
    input_path = _validated_nonpublic_input_path(path, field=field)
    payload_bytes = input_path.read_bytes()
    observed = hashlib.sha256(payload_bytes).hexdigest()
    if observed != expected:
        raise ValueError(
            f"{field} SHA256 mismatch: expected={expected}, observed={observed}"
        )
    payload = json.loads(payload_bytes)
    if not isinstance(payload, dict):
        raise ValueError(f"{field} must contain a JSON object")
    return payload, payload_bytes, observed


def _resolve_canonical_sha256(
    supplied: str | None,
    *,
    frozen: str,
    field: str,
) -> str:
    if supplied is None:
        return frozen
    value = _validate_sha256(supplied, field=field)
    if value != frozen:
        raise ValueError(
            f"{field} must equal the frozen canonical SHA256 {frozen}"
        )
    return value


def _canonical_content_digest(values: Sequence[str], *, field_name: str) -> str:
    normalized = list(values)
    if normalized != sorted(set(normalized)):
        raise ValueError(f"{field_name} must be sorted and unique")
    decoded: list[bytes] = []
    for value in normalized:
        digest = _validate_sha256(value, field=field_name)
        decoded.append(bytes.fromhex(digest))
    return hashlib.sha256(
        b"contextworld-cube-prior-content-exclusions-v1\0"
        + field_name.encode("ascii")
        + b"\0"
        + b"".join(decoded)
    ).hexdigest()


def _load_failed_formal_attempt_action_profiles(
    path: Path | str, *, expected_sha256: str
) -> tuple[frozenset[str], dict[str, Any]]:
    payload, raw, observed = _read_sha256_json(
        path,
        expected_sha256=expected_sha256,
        field="failed formal v4 attempt receipt",
    )
    scope = payload.get("scope")
    failed = payload.get("failed_attempt_content")
    if (
        payload.get("schema_version") != 1
        or payload.get("protocol_id") != V4_PROTOCOL
        or payload.get("status") != "infrastructure_failed_immutable_attempt"
        or payload.get("formal_build_attempt_consumed") is not True
        or payload.get("checks_passed") is not True
        or payload.get("build_passed") is not False
        or not isinstance(scope, Mapping)
        or not isinstance(failed, Mapping)
    ):
        raise ValueError("failed formal v4 attempt receipt contract mismatch")
    public = scope.get("public_test")
    if (
        not isinstance(public, Mapping)
        or public.get("access_status") != "closed_not_read_not_scored"
        or any(
            public.get(name) is not False
            for name in ("opened", "read", "hashed", "scored")
        )
        or scope.get("reference_model_training_or_scoring") is not False
        or scope.get("optimizer_steps") != 0
    ):
        raise ValueError("failed formal v4 attempt did not keep Public/model closed")
    content = failed.get("prior_content_exclusions")
    if not isinstance(content, Mapping):
        raise ValueError("failed formal v4 attempt lacks content identities")
    entry = content.get("action_profile_ids")
    if not isinstance(entry, Mapping):
        raise ValueError("failed formal v4 attempt lacks action profile identities")
    values = [str(value) for value in entry.get("values", [])]
    if (
        len(values) != 2048
        or int(entry.get("count", -1)) != len(values)
        or entry.get("sha256")
        != _canonical_content_digest(values, field_name="action_profile_ids")
        or failed.get("split") != "train"
        or int(failed.get("pair_count", -1)) != 2048
        or int(failed.get("catalog_index_start_inclusive", -1))
        != V4_FORMAL_CATALOG_INDEX_OFFSET
        or int(failed.get("catalog_index_stop_exclusive", -1))
        != V4_FORMAL_CATALOG_INDEX_OFFSET + 2048
    ):
        raise ValueError("failed formal v4 action profile set is not canonical")
    return frozenset(values), {
        "path_recorded": False,
        "size_bytes": len(raw),
        "expected_sha256": expected_sha256,
        "observed_sha256": observed,
        "sha256_verified": True,
        "status": payload["status"],
        "formal_build_attempt_consumed": True,
        "failed_catalog_index_range": {
            "start_inclusive": V4_FORMAL_CATALOG_INDEX_OFFSET,
            "stop_exclusive": V4_FORMAL_CATALOG_INDEX_OFFSET + 2048,
        },
        "failed_action_profile_count": len(values),
        "failed_action_profile_ids_sha256": entry["sha256"],
        "public_test_read": False,
        "model_training_or_scoring": False,
        "checks_passed": True,
    }


def _load_anchor_support_evidence(
    path: Path | str,
    *,
    expected_sha256: str,
) -> tuple[dict[str, AnchorEvidence], dict[str, Any]]:
    """Load the canonical v3 anchor evidence for unchanged v4 actions."""

    expected = _resolve_canonical_sha256(
        expected_sha256,
        frozen=FROZEN_V3_ANCHOR_SUPPORT_EVIDENCE_SHA256,
        field="anchor-support evidence SHA256",
    )
    _, anchors, inherited_receipt = _v3._load_frozen_evidence(
        Path(path), expected_sha256=expected
    )
    receipt = dict(inherited_receipt)
    receipt.update(
        {
            "source_symbol": ANCHOR_SUPPORT_EVIDENCE_SOURCE_SYMBOL,
            "canonical_logical_name": (
                FROZEN_V3_ANCHOR_SUPPORT_LOGICAL_NAME
            ),
            "scientific_role": (
                "anchor_and_original_h5_action_support_only"
            ),
            "reused_from_v3": True,
            "reuse_is_valid_because": (
                "v4 action anchors, gripper values, [p,-p,p] construction, "
                "and support metric are unchanged from v3"
            ),
            "coupling_selection_evidence": False,
        }
    )
    return anchors, receipt


def _mapping(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a mapping")
    return value


def _variant_causal_checks(variant: Mapping[str, Any]) -> dict[str, bool]:
    return {
        "positive_pair_count": bool(
            isinstance(variant.get("pair_count"), int)
            and not isinstance(variant.get("pair_count"), bool)
            and int(variant["pair_count"]) > 0
        ),
        "paired_initial_pixels_equal": (
            variant.get("all_initial_pixels_equal") is True
        ),
        "paired_query_pixels_equal": (
            variant.get("all_query_pixels_equal") is True
        ),
        "paired_actions_equal": variant.get("all_actions_equal") is True,
        "continuous_trajectory": variant.get("all_continuous") is True,
        "exact_profile_constraints": (
            variant.get("all_exact_profile_constraints") is True
        ),
        "all_pair_checks_passed": (
            variant.get("all_pair_checks_passed") is True
        ),
        "query_state_exact": bool(
            isinstance(
                variant.get("maximum_query_simulator_state_gap"),
                (int, float),
            )
            and not isinstance(
                variant.get("maximum_query_simulator_state_gap"), bool
            )
            and math.isfinite(
                float(variant["maximum_query_simulator_state_gap"])
            )
            and float(variant["maximum_query_simulator_state_gap"]) <= 1e-12
        ),
    }


def _minimum_measurement(
    variant: Mapping[str, Any], *, field: str
) -> float:
    summary = _mapping(variant.get(field), field=f"variant.{field}")
    value = summary.get("minimum")
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"variant.{field}.minimum must be finite")
    return float(value)


def _median_measurement(
    variant: Mapping[str, Any], *, field: str
) -> float:
    summary = _mapping(variant.get(field), field=f"variant.{field}")
    value = summary.get("median")
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"variant.{field}.median must be finite")
    return float(value)


def _load_coupling_feasibility(
    path: Path | str,
    *,
    expected_sha256: str,
) -> dict[str, Any]:
    """Validate evidence used only to select v4's 0.40 N coupling."""

    expected = _resolve_canonical_sha256(
        expected_sha256,
        frozen=FROZEN_V4_COUPLING_FEASIBILITY_SHA256,
        field="coupling-feasibility SHA256",
    )
    payload, payload_bytes, observed = _read_sha256_json(
        path,
        expected_sha256=expected,
        field="Coupling-feasibility evidence",
    )
    scope = _mapping(payload.get("scope"), field="scope")
    design = _mapping(payload.get("design"), field="design")
    variants = _mapping(payload.get("variants"), field="variants")
    baseline = _mapping(
        variants.get("coupling_0.30_n"),
        field="variants.coupling_0.30_n",
    )
    selected = _mapping(
        variants.get("coupling_0.40_n"),
        field="variants.coupling_0.40_n",
    )
    baseline_causal = _variant_causal_checks(baseline)
    selected_causal = _variant_causal_checks(selected)
    baseline_height = _minimum_measurement(
        baseline, field="history_height_gap_m"
    )
    selected_height = _minimum_measurement(
        selected, field="history_height_gap_m"
    )
    baseline_rgb = _median_measurement(
        baseline, field="history_changed_rgb_values"
    )
    selected_rgb = _median_measurement(
        selected, field="history_changed_rgb_values"
    )
    baseline_feature = _median_measurement(
        baseline, field="feature_effect_rms"
    )
    selected_feature = _median_measurement(
        selected, field="feature_effect_rms"
    )
    unchanged = design.get("unchanged")
    unchanged_text = (
        "\n".join(str(value) for value in unchanged).lower()
        if isinstance(unchanged, list)
        else ""
    )
    couplings = np.asarray(design.get("couplings_n"), dtype=np.float64)
    checks = {
        "canonical_sha256_verified": observed == expected,
        "nonformal_parameter_selection_role": payload.get("role")
        == "nonformal_v4_design_feasibility_not_a_frozen_gate",
        "history_three": scope.get("history") == 3,
        "training_source_only": "training" in str(scope.get("source", "")).lower(),
        "public_test_not_opened_read_hashed_or_scored": (
            scope.get("public_test_opened_read_hashed_or_scored") is False
        ),
        "no_reference_model_training_or_scoring": (
            scope.get("reference_model_training_or_scoring") is False
        ),
        "repository_not_modified": scope.get("repository_modified") is False,
        "only_scientific_variable_is_vertical_coupling": (
            "vertical force coupling"
            in str(design.get("only_variable", "")).lower()
        ),
        "selected_coupling_present": bool(
            couplings.ndim == 1
            and np.isfinite(couplings).all()
            and np.any(couplings == V4_VERTICAL_FORCE_COUPLING_N)
        ),
        "paired_x0_declared_unchanged": "paired x0" in unchanged_text,
        "paired_actions_declared_unchanged": (
            "paired actions" in unchanged_text
        ),
        "continuous_p_negative_p_p_declared_unchanged": (
            "continuous [p,-p,p]" in unchanged_text
        ),
        "exact_constraints_declared_unchanged": (
            "sum(p)=0" in unchanged_text
            and "p[-1]=0" in unchanged_text
            and "moment(p)=1" in unchanged_text
        ),
        "baseline_causal_contract_passed": all(baseline_causal.values()),
        "selected_causal_contract_passed": all(selected_causal.values()),
        "selected_height_is_about_12p60_mm": (
            0.0125 <= selected_height <= 0.0127
        ),
        "selected_height_exceeds_v3_baseline": selected_height > baseline_height,
        "selected_raw_rgb_median_exceeds_baseline": selected_rgb > baseline_rgb,
        "selected_16x16_effect_median_exceeds_baseline": (
            selected_feature > baseline_feature
        ),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError(
            "Coupling-feasibility evidence violates the v4 selection contract: "
            + ", ".join(failed)
        )
    return {
        "source_symbol": COUPLING_FEASIBILITY_SOURCE_SYMBOL,
        "path_recorded": False,
        "size_bytes": len(payload_bytes),
        "expected_sha256": expected,
        "observed_sha256": observed,
        "sha256_verified": True,
        "scientific_role": "select_v4_vertical_force_coupling_only",
        "selected_vertical_force_coupling_n": V4_VERTICAL_FORCE_COUPLING_N,
        "action_support_distance_or_window_count_contribution": False,
        "baseline_history_height_gap_m_minimum": baseline_height,
        "selected_history_height_gap_m_minimum": selected_height,
        "baseline_history_changed_rgb_values_median": baseline_rgb,
        "selected_history_changed_rgb_values_median": selected_rgb,
        "baseline_16x16_feature_effect_rms_median": baseline_feature,
        "selected_16x16_feature_effect_rms_median": selected_feature,
        "baseline_causal_checks": baseline_causal,
        "selected_causal_checks": selected_causal,
        "checks": checks,
        "passed": True,
    }


def _effective_anchor_actions(anchor_id: str) -> np.ndarray:
    if anchor_id not in V4_ACTION_ANCHORS:
        raise ValueError(f"Unknown v4 action anchor {anchor_id!r}")
    probe = np.asarray(V4_ANCHOR_PROFILES[anchor_id], dtype=np.float64)
    actions = np.zeros(EFFECTIVE_ACTION_SHAPE, dtype=np.float64)
    actions[:, 2] = np.concatenate((probe, -probe, probe))
    actions[:, 4] = float(V4_ANCHOR_GRIPPER_ACTIONS[anchor_id])
    return actions


def _profile_record(
    *,
    split: str,
    local_index: int,
    anchor_evidence: Mapping[str, AnchorEvidence],
    original_h5_stats: OriginalH5ActionStats,
) -> dict[str, Any]:
    catalog_index = FORMAL_CATALOG_INDEX_OFFSET + int(local_index)
    profile = make_v4_action_profile(split=split, catalog_index=catalog_index)
    blocks = np.asarray(action_blocks(profile))
    profile_audit = validate_v4_action_profile(profile)
    if blocks.shape != ACTION_PROFILE_SHAPE or blocks.dtype != np.float32:
        raise RuntimeError("v4 profile factory returned noncanonical action blocks")
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
        "factory_split_seed_matches_v4": int(profile.split_seed)
        == int(V4_PROFILE_SPLIT_SEEDS[split]),
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
        "local_index": int(local_index),
        "catalog_index": int(catalog_index),
        "split_seed": int(profile.split_seed),
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
        for anchor_id in V4_ACTION_ANCHORS
    }

    def maximum(name: str) -> float:
        return max(float(row["measurements"][name]) for row in records)

    support_counts = {
        anchor_id: {
            int(
                row["measurements"]
                ["anchor_h5_window_support_count_at_or_below_nrmse_0p5"]
            )
            for row in records
            if row["action_anchor_id"] == anchor_id
        }
        for anchor_id in V4_ACTION_ANCHORS
    }
    normalized_support_counts = {
        anchor_id: next(iter(values)) if len(values) == 1 else None
        for anchor_id, values in support_counts.items()
        if values
    }
    failed = [str(row["action_profile_id"]) for row in records if not row["passed"]]
    return {
        "profile_count": len(records),
        "local_index_range": {
            "minimum": min(int(row["local_index"]) for row in records),
            "maximum": max(int(row["local_index"]) for row in records),
        },
        "catalog_index_range": {
            "minimum": min(int(row["catalog_index"]) for row in records),
            "maximum": max(int(row["catalog_index"]) for row in records),
        },
        "unique_profile_count": len(set(profile_ids)),
        "profile_ids_sha256": _profile_ids_sha256(profile_ids),
        "passed_profile_count": sum(bool(row["passed"]) for row in records),
        "conservatively_supported_profile_count": sum(
            bool(
                row["checks"]
                ["conservative_original_h5_nrmse_upper_bound_passed"]
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
    coupling_feasibility: Path | str,
    failed_formal_attempt_receipt: Path | str,
    original_h5: Path | str,
    feasibility_evidence_sha256: str | None = None,
    coupling_feasibility_sha256: str | None = None,
    failed_formal_attempt_receipt_sha256: str | None = None,
) -> dict[str, Any]:
    """Audit all 4,608 profiles the fixed v4r1 candidate pools may use."""

    feasibility_evidence = _validated_nonpublic_input_path(
        feasibility_evidence, field="anchor-support evidence"
    )
    coupling_feasibility = _validated_nonpublic_input_path(
        coupling_feasibility, field="coupling-feasibility evidence"
    )
    failed_formal_attempt_receipt = _validated_nonpublic_input_path(
        failed_formal_attempt_receipt,
        field="failed formal v4 attempt receipt",
    )
    original_h5 = _validated_nonpublic_input_path(
        original_h5, field="upstream original H5"
    )

    anchor_sha = _resolve_canonical_sha256(
        feasibility_evidence_sha256,
        frozen=FROZEN_V3_ANCHOR_SUPPORT_EVIDENCE_SHA256,
        field="anchor-support evidence SHA256",
    )
    coupling_sha = _resolve_canonical_sha256(
        coupling_feasibility_sha256,
        frozen=FROZEN_V4_COUPLING_FEASIBILITY_SHA256,
        field="coupling-feasibility SHA256",
    )
    failed_sha = _resolve_canonical_sha256(
        failed_formal_attempt_receipt_sha256,
        frozen=FROZEN_V4_FAILED_FORMAL_ATTEMPT_RECEIPT_SHA256,
        field="failed formal v4 attempt receipt SHA256",
    )
    anchor_evidence, anchor_receipt = _load_anchor_support_evidence(
        feasibility_evidence,
        expected_sha256=anchor_sha,
    )
    coupling_receipt = _load_coupling_feasibility(
        coupling_feasibility,
        expected_sha256=coupling_sha,
    )
    failed_profile_ids, failed_attempt_receipt = (
        _load_failed_formal_attempt_action_profiles(
            failed_formal_attempt_receipt,
            expected_sha256=failed_sha,
        )
    )
    original_h5_stats = compute_original_h5_action_population_stats(original_h5)
    records_by_split = {
        split: [
            _profile_record(
                split=split,
                local_index=local_index,
                anchor_evidence=anchor_evidence,
                original_h5_stats=original_h5_stats,
            )
            for local_index in range(FROZEN_PROFILE_COUNTS[split])
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
        for anchor_id in V4_ACTION_ANCHORS
    }

    train_profile_ids = {
        str(row["action_profile_id"]) for row in records_by_split["train"]
    }
    development_profile_ids = {
        str(row["action_profile_id"])
        for row in records_by_split["loader_validation"]
    }
    profile_overlap = sorted(train_profile_ids & development_profile_ids)
    all_profile_ids = train_profile_ids | development_profile_ids
    failed_attempt_profile_overlap = sorted(
        all_profile_ids & failed_profile_ids
    )
    train_anchors = {
        str(row["action_anchor_id"]) for row in records_by_split["train"]
    }
    development_anchors = {
        str(row["action_anchor_id"])
        for row in records_by_split["loader_validation"]
    }
    anchor_overlap = sorted(train_anchors & development_anchors)
    expected_anchors = sorted(V4_ACTION_ANCHORS)
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
        split: FROZEN_PROFILE_COUNTS[split] // len(V4_ACTION_ANCHORS)
        for split in ACTIVE_SPLITS
    }
    checks = {
        "canonical_v4_coupling_selection_evidence_passed": bool(
            coupling_receipt["passed"]
        ),
        "coupling_evidence_not_used_as_action_support": (
            coupling_receipt[
                "action_support_distance_or_window_count_contribution"
            ]
            is False
        ),
        "canonical_v3_anchor_support_evidence_sha256_verified": bool(
            anchor_receipt["sha256_verified"]
        ),
        "anchor_support_causal_evidence_passed": bool(
            anchor_receipt["all_causal_audits_passed"]
        ),
        "original_h5_action_source_and_population_stats_passed": (
            original_h5_stats.passed
        ),
        "v3_frozen_original_h5_action_hashes_reused_unchanged": (
            FROZEN_ORIGINAL_H5_ACTION_DATASET_SHA256
            == _v3.FROZEN_ORIGINAL_H5_ACTION_DATASET_SHA256
            and FROZEN_ORIGINAL_H5_FINITE_ACTION_CONTENT_SHA256
            == _v3.FROZEN_ORIGINAL_H5_FINITE_ACTION_CONTENT_SHA256
            and FROZEN_ORIGINAL_H5_METRIC_STD_POPULATION_FLOAT32_SHA256
            == _v3.FROZEN_ORIGINAL_H5_METRIC_STD_POPULATION_FLOAT32_SHA256
        ),
        "anchor_and_concrete_distances_share_standardized_metric_contract": bool(
            anchor_receipt[
                "joint_15_step_standardized_metric_contract_validated"
            ]
            and original_h5_stats.passed
        ),
        "v4_protocol_and_profile_seeds_frozen": (
            V4_PROTOCOL
            == "cube_gripper_carry_rule_history3_development_v4"
            and V4_PROFILE_SPLIT_SEEDS
            == {"train": 2026081201, "loader_validation": 2026081202}
        ),
        "only_training_and_development_profiles_enumerated": (
            tuple(records_by_split) == ACTIVE_SPLITS
        ),
        "frozen_profile_counts_exact": all(
            len(records_by_split[split]) == FROZEN_PROFILE_COUNTS[split]
            for split in ACTIVE_SPLITS
        ),
        "formal_catalog_offset_positive_and_four_aligned": (
            FORMAL_CATALOG_INDEX_OFFSET > V4_FORMAL_CATALOG_INDEX_OFFSET
            and FORMAL_CATALOG_INDEX_OFFSET % len(V4_ACTION_ANCHORS) == 0
        ),
        "formal_catalog_indices_equal_offset_plus_local_index": all(
            int(row["catalog_index"])
            == FORMAL_CATALOG_INDEX_OFFSET + int(row["local_index"])
            for row in all_records
        ),
        "formal_catalog_indices_disjoint_from_preformal_zero_and_one": all(
            int(row["catalog_index"]) not in {0, 1} for row in all_records
        ),
        "four_anchor_balance_exact_per_split": all(
            split_summaries[split]["action_anchor_counts"]
            == {
                anchor_id: expected_per_anchor[split]
                for anchor_id in V4_ACTION_ANCHORS
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
        "failed_v4_attempt_action_profile_overlap_zero": (
            not failed_attempt_profile_overlap
        ),
        "anchor_family_overlap_four": anchor_overlap == expected_anchors,
        "no_public_test_input_or_generation": True,
    }
    passed = all(checks.values())
    support_counts = {
        anchor_id: anchor_evidence[
            anchor_id
        ].h5_window_support_count_at_or_below_0p5
        for anchor_id in V4_ACTION_ANCHORS
    }
    return {
        "schema_version": 1,
        "audit_id": AUDIT_ID,
        "protocol": V4_PROTOCOL,
        "recovery_authorization_id": RECOVERY_AUTHORIZATION_ID,
        "status": "passed" if passed else "failed",
        "scope": {
            "phase": "development_only",
            "active_splits": list(ACTIVE_SPLITS),
            "frozen_profile_counts": dict(FROZEN_PROFILE_COUNTS),
            "profile_split_seeds": dict(V4_PROFILE_SPLIT_SEEDS),
            "formal_catalog_namespace": {
                "catalog_index_offset": FORMAL_CATALOG_INDEX_OFFSET,
                "local_index_policy": FORMAL_CATALOG_LOCAL_INDEX_POLICY,
                "catalog_index_formula": (
                    "FORMAL_CATALOG_INDEX_OFFSET + local_index"
                ),
                "offset_positive": FORMAL_CATALOG_INDEX_OFFSET > 0,
                "offset_modulo_anchor_count": (
                    FORMAL_CATALOG_INDEX_OFFSET
                    % len(V4_ACTION_ANCHORS)
                ),
                "prior_catalog_namespaces_excluded": [
                    {"start_inclusive": 0, "stop_exclusive": 2},
                    {
                        "start_inclusive": V4_FORMAL_CATALOG_INDEX_OFFSET,
                        "stop_exclusive": (
                            V4_FORMAL_CATALOG_INDEX_OFFSET + 2048
                        ),
                    },
                ],
                "per_split_ranges": {
                    split: {
                        "local_index_start_inclusive": 0,
                        "local_index_stop_exclusive": (
                            FROZEN_PROFILE_COUNTS[split]
                        ),
                        "catalog_index_start_inclusive": (
                            FORMAL_CATALOG_INDEX_OFFSET
                        ),
                        "catalog_index_stop_exclusive": (
                            FORMAL_CATALOG_INDEX_OFFSET
                            + FROZEN_PROFILE_COUNTS[split]
                        ),
                    }
                    for split in ACTIVE_SPLITS
                },
            },
            "total_concrete_profiles": sum(FROZEN_PROFILE_COUNTS.values()),
            "profile_source": (
                "contextworld.evaluation.cube_grasp_rule_h3_v4."
                "make_v4_action_profile"
            ),
            "external_data_inputs": [
                "canonical SHA256-bound v4 coupling-feasibility JSON",
                "canonical SHA256-bound v3 anchor-support evidence JSON",
                "canonical SHA256-bound failed formal v4 attempt receipt",
                "upstream original H5 action dataset only",
            ],
            "original_h5_datasets_opened": [ORIGINAL_H5_ACTION_DATASET],
            "lance_tables_opened": [],
            "public_test_inputs": [],
            "public_test_opened": False,
            "public_test_generated": False,
        },
        "evidence_roles": {
            "coupling_selection": coupling_receipt,
            "anchor_original_h5_support": anchor_receipt,
            "failed_v4_formal_attempt_action_exclusion": (
                failed_attempt_receipt
            ),
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
                "V4 actions and anchors are unchanged, so the canonical v3 "
                "anchor evidence and the same original-H5 population metric "
                "remain applicable. Coupling feasibility contributes no "
                "action distance."
            ),
        },
        "gates": {
            "probe_sum": 0.0,
            "probe_last": 0.0,
            "probe_moment": 1.0,
            "environment_action_absolute_maximum": (
                ENVIRONMENT_ACTION_ABSOLUTE_MAXIMUM
            ),
            "original_h5_gripper_maximum": original_h5_stats.gripper_maximum,
            "conservative_original_h5_nrmse_upper_bound_maximum": (
                SUPPORT_NRMSE_MAXIMUM
            ),
            "profile_content_overlap_between_splits": 0,
            "anchor_family_overlap_between_splits": 4,
        },
        "frozen_original_h5_source_identity": {
            "source_symbol": ORIGINAL_H5_SOURCE_SYMBOL,
            "file_size_bytes": FROZEN_ORIGINAL_H5_FILE_SIZE_BYTES,
            "whole_file_sha256": FROZEN_ORIGINAL_H5_FILE_SHA256,
            "whole_file_hash_recomputed_by_this_action_only_audit": False,
            "action_dataset_sha256": FROZEN_ORIGINAL_H5_ACTION_DATASET_SHA256,
            "finite_action_content_sha256": (
                FROZEN_ORIGINAL_H5_FINITE_ACTION_CONTENT_SHA256
            ),
        },
        "original_h5_action": original_h5_stats.as_dict(),
        "support_counts": {
            "concrete_profiles_evaluated": overall["profile_count"],
            "concrete_profiles_conservatively_supported": overall[
                "conservatively_supported_profile_count"
            ],
            "anchor_h5_windows_at_or_below_nrmse_0p5": support_counts,
            "minimum_anchor_h5_window_count": min(support_counts.values()),
            "note": (
                "Counts come only from the canonical v3 anchor evidence; "
                "they remain valid because v4 actions/anchors are unchanged."
            ),
        },
        "overall": overall,
        "splits": split_summaries,
        "anchors": per_anchor,
        "cross_split": cross_split,
        "failed_v4_attempt_exclusion": {
            "failed_action_profile_count": len(failed_profile_ids),
            "recovery_action_profile_count": len(all_profile_ids),
            "overlap_count": len(failed_attempt_profile_overlap),
            "overlap_values": failed_attempt_profile_overlap,
            "passed": not failed_attempt_profile_overlap,
        },
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
    parser.add_argument(
        "--feasibility-evidence",
        type=Path,
        required=True,
        help="Canonical v3 action_template_feasibility_input.json",
    )
    parser.add_argument(
        "--feasibility-evidence-sha256",
        default=None,
        help="Must equal the frozen canonical v3 evidence SHA256",
    )
    parser.add_argument(
        "--coupling-feasibility",
        type=Path,
        required=True,
        help="Canonical v4 coupling_feasibility JSON",
    )
    parser.add_argument(
        "--coupling-feasibility-sha256",
        default=None,
        help="Must equal the frozen canonical v4 evidence SHA256",
    )
    parser.add_argument(
        "--failed-formal-attempt-receipt",
        type=Path,
        required=True,
        help="Canonical immutable failed formal v4 attempt receipt",
    )
    parser.add_argument(
        "--failed-formal-attempt-receipt-sha256",
        default=None,
        help="Must equal the frozen canonical failed-attempt receipt SHA256",
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
        coupling_feasibility=args.coupling_feasibility,
        coupling_feasibility_sha256=args.coupling_feasibility_sha256,
        failed_formal_attempt_receipt=args.failed_formal_attempt_receipt,
        failed_formal_attempt_receipt_sha256=(
            args.failed_formal_attempt_receipt_sha256
        ),
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
