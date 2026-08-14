#!/usr/bin/env python3
"""Freeze the Development-only Cube History-3 v4 preregistration.

This command verifies only explicitly supplied frozen inputs.  It does not
discover or open Lance data and it refuses every Public-Test-shaped path.  The
receipt is written exclusively and is the authorization consumed by the v4
builder; creating it does not build data or run the RGB probe.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import h5py
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREREG = ROOT / (
    "configs/benchmark/cube_gripper_carry_h3_development_prereg_v4.yaml"
)
PROTOCOL = "cube_gripper_carry_rule_history3_development_v4"
PREREG_STATUS = "preregistered_before_first_v4_build"
RECEIPT_STATUS = "frozen_before_first_v4_build"
ACTIVE_SPLITS = ("train", "loader_validation")
PLACEHOLDER_TOKENS = (
    "TO_BE_FROZEN",
    "PLACEHOLDER",
    "TBD",
    "REPLACE_ME",
)
SOURCE_SYMBOL = "upstream_cube_single_expert_h5"
V3_COUPLING_N = 0.30
V4_COUPLING_N = 0.40

SOURCE_FILE_SHA256 = (
    "0664d507c4ff12009010644c9ae950836f954e700c172ccf22e7423af1a55625"
)
SOURCE_SIZE_BYTES = 101_942_558_720
SOURCE_ROW_COUNT = 2_010_000
SOURCE_EPISODE_COUNT = 10_000
ACTION_SHAPE = (2_010_000, 5)
ACTION_DTYPE = "float32"
ACTION_FINITE_ROW_COUNT = 2_000_000
ACTION_EXCLUDED_NONFINITE_ROW_COUNT = 10_000
ACTION_DATA_SHA256 = (
    "e94371078958ee8ad62edba91435841714c293ea9684ae6523a03574793faa40"
)
ACTION_FINITE_CONTENT_SHA256 = (
    "ea2cf67d4b7500981298499d76de0129bdc46e50da05e160f81d12e24edfc80b"
)
ACTION_STD_FLOAT32 = (
    0.28941983,
    0.39371696,
    0.64313650,
    0.39280161,
    0.25030735,
)
ACTION_STD_FLOAT32_SHA256 = (
    "3929f3ba78d594bb372f48f14c588e0f42c8f33e83b6a2312b3f872d1e222954"
)

BASIS_CANONICAL_PATH = (
    "artifacts/evaluation/history3/cube_gripper_carry_h3_development_v4/"
    "prior_episode_exclusions_basis_v1.json"
)
BASIS_RECEIPT_SHA256 = (
    "fd02914c9e3157df2a7ea9766ca9c712130def2deb371f932777535f6f0ce59f"
)
BASIS_EPISODE_COUNT = 2320
BASIS_EPISODE_IDS_SHA256 = (
    "6a61cc77e5f2c769ce006a9dbbd3e7a16187ed59fbf5beec2f025713b68ac152"
)

PROBE_RECIPE = {
    "input": "decoded_x0_x1_x2_rgb_only",
    "resize_shape": [16, 16],
    "resize_interpolation": "Pillow_Resampling_BILINEAR",
    "arithmetic_dtype": "float64",
    "fixed_feature": "flatten(2*x1-x0-x2)_C_order",
    "standard_scaler_fit_split_only": "train",
    "estimator": "StandardScaler_then_RidgeClassifier_alpha_1",
    "label_encoding": {"cannot_hold": 0, "can_hold": 1},
}
PROBE_THRESHOLDS = {
    "overall_accuracy_minimum": 0.75,
    "worst_mode_accuracy_minimum": 0.70,
    "worst_anchor_family_accuracy_minimum": 0.70,
    "pair_cluster_bootstrap_lower_bound_minimum": 0.70,
    "label_permutation_mean_accuracy_maximum": 0.60,
    "x0_only_accuracy_maximum": 0.51,
    "query_only_accuracy_maximum": 0.51,
    "action_only_accuracy_maximum": 0.51,
}

REQUIRED_IDENTITY_KEYS = (
    "base_v2_physics",
    "common_causal_contract",
    "v3_physics_dependency",
    "v4_physics",
    "v4_builder",
    "v4_physics_tests",
    "v4_builder_tests",
    "v4_action_support_audit",
    "v4_action_support_audit_tests",
    "v4_probe",
    "v4_probe_tests",
    "prereg_freezer",
    "prereg_freezer_tests",
    "prior_basis_freezer",
    "prior_basis_freezer_tests",
    "preformal_content_freezer",
    "preformal_content_freezer_tests",
    "prior_finalizer",
    "prior_finalizer_tests",
    "protocol_document",
)
REQUIRED_EVIDENCE_KEYS = (
    "coupling_pilot",
    "exploratory_diagnostic",
    "v3_failed_development_decision",
    "action_feasibility_report",
    "action_feasibility_narrative",
    "v4_action_support_audit",
    "v4_preformal_content_receipt",
)
_SHA256 = re.compile(r"[0-9a-f]{64}")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _contains_placeholder(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(_contains_placeholder(child) for child in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_placeholder(child) for child in value)
    return isinstance(value, str) and any(
        token in value.upper() for token in PLACEHOLDER_TOKENS
    )


def _forbidden_public_component(path: Path) -> str | None:
    forbidden = {
        "validation",
        "validation.lance",
        "public",
        "public_test",
        "public-test",
        "publictest",
    }
    return next((part for part in path.parts if part.lower() in forbidden), None)


def _resolve_declared_path(value: str, *, artifact_root: Path) -> Path:
    path = Path(value)
    forbidden = _forbidden_public_component(path)
    if forbidden is not None:
        raise RuntimeError(f"refusing Public path component {forbidden!r}")
    if path.is_absolute():
        return path.resolve()
    if path.parts and path.parts[0] == "artifacts":
        bundled = (ROOT / path).resolve()
        if bundled.is_file():
            return bundled
        return artifact_root.joinpath(*path.parts[1:]).resolve()
    return (ROOT / path).resolve()


def _required_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{label} must be a mapping")
    return value


def _canonical_float32_vector(
    value: Any, *, length: int, label: str
) -> list[float]:
    """Return the exact Python-float expansion of a finite float32 vector.

    YAML/JSON commonly renders float32 values using a shorter decimal spelling
    than ``float(np.float32(value))``.  Compare the frozen float32 values, not
    those two equivalent spellings.
    """

    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise RuntimeError(f"{label} must be a length-{length} sequence")
    normalized: list[float] = []
    for index, raw in enumerate(value):
        if isinstance(raw, (bool, np.bool_)) or not isinstance(
            raw, (int, float, np.integer, np.floating)
        ):
            raise RuntimeError(f"{label}[{index}] must be numeric")
        numeric = float(raw)
        if not math.isfinite(numeric):
            raise RuntimeError(f"{label}[{index}] must be finite")
        canonical = np.float32(numeric)
        if not np.isfinite(canonical):
            raise RuntimeError(f"{label}[{index}] is outside finite float32")
        normalized.append(float(canonical))
    return normalized


def _validate_sha256(value: Any, *, label: str) -> str:
    digest = str(value)
    if _SHA256.fullmatch(digest) is None:
        raise RuntimeError(f"{label} must be a lowercase SHA256")
    return digest


def _verified_file_entry(
    entry: Any, *, artifact_root: Path, label: str
) -> dict[str, Any]:
    mapping = _required_mapping(entry, label=label)
    if set(("path", "sha256")) - set(mapping):
        raise RuntimeError(f"{label} must declare path and sha256")
    path = _resolve_declared_path(str(mapping["path"]), artifact_root=artifact_root)
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"{label}: missing regular declared file {path}")
    expected = _validate_sha256(mapping["sha256"], label=f"{label}.sha256")
    observed = file_sha256(path)
    if observed != expected:
        raise RuntimeError(f"{label}: sha256 mismatch: {observed} != {expected}")
    if "size_bytes" in mapping and int(mapping["size_bytes"]) != path.stat().st_size:
        raise RuntimeError(f"{label}: size_bytes mismatch")
    return {
        "path": str(mapping["path"]),
        "sha256": observed,
        "size_bytes": path.stat().st_size,
    }


def _verified_named_entries(
    entries: Any,
    *,
    required_names: Sequence[str],
    artifact_root: Path,
    label: str,
) -> dict[str, dict[str, Any]]:
    mapping = _required_mapping(entries, label=label)
    missing = sorted(set(required_names) - set(mapping))
    if missing:
        raise RuntimeError(f"{label} lacks required entries: {missing}")
    return {
        name: _verified_file_entry(
            mapping[name], artifact_root=artifact_root, label=f"{label}.{name}"
        )
        for name in required_names
    }


def _validate_public_and_scope(document: Mapping[str, Any]) -> None:
    if document.get("status") != PREREG_STATUS:
        raise RuntimeError("Cube v4 preregistration status mismatch")
    if document.get("phase") != "development_only":
        raise RuntimeError("Cube v4 freeze requires phase=development_only")
    if document.get("protocol_id") != PROTOCOL:
        raise RuntimeError("Unexpected Cube v4 protocol_id")
    public = _required_mapping(document.get("public_test"), label="public_test")
    if public.get("access_status") != "closed_not_read_not_scored" or any(
        public.get(name) is not False
        for name in (
            "validation_lance_access_allowed",
            "opened",
            "read",
            "hashed",
            "scored",
        )
    ):
        raise RuntimeError("Public Test must be fully closed")
    reference = _required_mapping(
        document.get("reference_model_phase"), label="reference_model_phase"
    )
    if reference.get("training_and_scoring_authorized") is not False:
        raise RuntimeError("v4 freeze must not authorize reference training")


def _validate_scientific_change(document: Mapping[str, Any]) -> dict[str, Any]:
    change = _required_mapping(
        document.get("scientific_change"), label="scientific_change"
    )
    expected = {
        "sole_change": "can_hold_vertical_force_coupling_n",
        "v3_baseline_vertical_force_coupling_n": V3_COUPLING_N,
        "v4_vertical_force_coupling_n": V4_COUPLING_N,
        "capability_semantics_unchanged": True,
        "history3_causal_sequence_unchanged": True,
        "action_profiles_and_constraints_unchanged_except_new_seeds": True,
    }
    if {name: change.get(name) for name in expected} != expected:
        raise RuntimeError("v4 must declare the unique 0.30-to-0.40 coupling change")
    return dict(expected)


def _validate_probe_contract(document: Mapping[str, Any]) -> dict[str, Any]:
    learnability = _required_mapping(
        document.get("learnability_gates"), label="learnability_gates"
    )
    probe = _required_mapping(
        learnability.get("rgb_history_probe"), label="rgb_history_probe"
    )
    recipe_unchanged = probe.get(
        "recipe_unchanged_from_v3", learnability.get("recipe_unchanged_from_v3")
    )
    thresholds_unchanged = probe.get(
        "thresholds_unchanged_from_v3",
        learnability.get("thresholds_unchanged_from_v3"),
    )
    if recipe_unchanged is not True:
        raise RuntimeError("v4 probe recipe must be unchanged from v3")
    if thresholds_unchanged is not True:
        raise RuntimeError("v4 probe thresholds must be unchanged from v3")

    recipe = probe.get("recipe")
    if recipe is None:
        resize = _required_mapping(probe.get("resize"), label="rgb_history_probe.resize")
        recipe = {
            "input": probe.get("input"),
            "resize_shape": resize.get("shape"),
            "resize_interpolation": resize.get("interpolation"),
            "arithmetic_dtype": probe.get("arithmetic_dtype"),
            "fixed_feature": probe.get("fixed_feature"),
            "standard_scaler_fit_split_only": probe.get(
                "standard_scaler_fit_split_only"
            ),
            "estimator": probe.get("estimator"),
            "label_encoding": probe.get("label_encoding"),
        }
    if recipe != PROBE_RECIPE:
        raise RuntimeError("v4 RGB probe recipe differs from the frozen v3 recipe")

    thresholds = probe.get("thresholds")
    if thresholds is None:
        negative = _required_mapping(
            probe.get("negative_controls"), label="rgb_history_probe.negative_controls"
        )
        permutation = _required_mapping(
            negative.get("label_permutation"),
            label="rgb_history_probe.negative_controls.label_permutation",
        )
        thresholds = {
            "overall_accuracy_minimum": probe.get("overall_accuracy_minimum"),
            "worst_mode_accuracy_minimum": probe.get(
                "worst_mode_accuracy_minimum"
            ),
            "worst_anchor_family_accuracy_minimum": probe.get(
                "worst_anchor_family_accuracy_minimum"
            ),
            "pair_cluster_bootstrap_lower_bound_minimum": probe.get(
                "pair_cluster_bootstrap_lower_bound_minimum"
            ),
            "label_permutation_mean_accuracy_maximum": permutation.get(
                "mean_accuracy_maximum"
            ),
            "x0_only_accuracy_maximum": negative.get(
                "x0_only_accuracy_maximum"
            ),
            "query_only_accuracy_maximum": negative.get(
                "query_only_accuracy_maximum"
            ),
            "action_only_accuracy_maximum": negative.get(
                "action_only_accuracy_maximum"
            ),
        }
    if thresholds != PROBE_THRESHOLDS:
        raise RuntimeError("v4 RGB probe thresholds differ from v3")
    return {
        "recipe_unchanged_from_v3": True,
        "thresholds_unchanged_from_v3": True,
        "recipe": dict(recipe),
        "thresholds": dict(thresholds),
    }


def _load_json(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{label} is not valid UTF-8 JSON") from error
    return _required_mapping(value, label=label)


def _validate_prior_basis(
    document: Mapping[str, Any], *, artifact_root: Path
) -> dict[str, Any]:
    prior = _required_mapping(
        document.get("prior_episode_exclusion"), label="prior_episode_exclusion"
    )
    basis_entry = _required_mapping(
        prior.get("basis_receipt"), label="prior_episode_exclusion.basis_receipt"
    )
    if basis_entry.get("path") != BASIS_CANONICAL_PATH:
        raise RuntimeError("prior basis receipt canonical path mismatch")
    if basis_entry.get("sha256") != BASIS_RECEIPT_SHA256:
        raise RuntimeError("prior basis receipt frozen SHA256 mismatch")
    verified = _verified_file_entry(
        basis_entry,
        artifact_root=artifact_root,
        label="prior_episode_exclusion.basis_receipt",
    )
    path = _resolve_declared_path(BASIS_CANONICAL_PATH, artifact_root=artifact_root)
    basis = _load_json(path, label="prior episode exclusion basis")
    if basis.get("protocol_id") != PROTOCOL:
        raise RuntimeError("prior basis protocol mismatch")
    if basis.get("status") not in (
        "frozen_before_first_v4_data_build",
        RECEIPT_STATUS,
    ):
        raise RuntimeError("prior basis status mismatch")
    if basis.get("checks_passed") is not True:
        raise RuntimeError("prior basis checks_passed is not true")
    if int(basis.get("excluded_source_episode_count", -1)) != BASIS_EPISODE_COUNT:
        raise RuntimeError("prior basis excluded episode count mismatch")
    if basis.get("excluded_source_episode_ids_sha256") != BASIS_EPISODE_IDS_SHA256:
        raise RuntimeError("prior basis episode-set digest mismatch")
    if prior.get("basis_episode_count") != BASIS_EPISODE_COUNT or prior.get(
        "basis_episode_ids_sha256"
    ) != BASIS_EPISODE_IDS_SHA256:
        raise RuntimeError("prereg prior-basis count/digest mismatch")
    return {
        **verified,
        "checks_passed": True,
        "excluded_source_episode_count": BASIS_EPISODE_COUNT,
        "excluded_source_episode_ids_sha256": BASIS_EPISODE_IDS_SHA256,
    }


def _validate_evidence(
    document: Mapping[str, Any], *, artifact_root: Path
) -> dict[str, dict[str, Any]]:
    evidence = _verified_named_entries(
        document.get("frozen_evidence"),
        required_names=REQUIRED_EVIDENCE_KEYS,
        artifact_root=artifact_root,
        label="frozen_evidence",
    )
    pilot_path = _resolve_declared_path(
        evidence["coupling_pilot"]["path"], artifact_root=artifact_root
    )
    pilot = _load_json(pilot_path, label="coupling pilot")
    design = _required_mapping(pilot.get("design"), label="coupling pilot design")
    couplings = [float(value) for value in design.get("couplings_n", [])]
    if V3_COUPLING_N not in couplings or V4_COUPLING_N not in couplings:
        raise RuntimeError("coupling pilot did not compare 0.30 N and 0.40 N")
    scope = _required_mapping(pilot.get("scope"), label="coupling pilot scope")
    if scope.get("reference_model_training_or_scoring") is not False or scope.get(
        "public_test_opened_read_hashed_or_scored"
    ) is not False:
        raise RuntimeError("coupling pilot scope is contaminated")

    diagnostic_path = _resolve_declared_path(
        evidence["exploratory_diagnostic"]["path"], artifact_root=artifact_root
    )
    diagnostic = _load_json(diagnostic_path, label="exploratory diagnostic")
    diagnostic_scope = _required_mapping(
        diagnostic.get("scope"), label="exploratory diagnostic scope"
    )
    diagnostic_public = _required_mapping(
        diagnostic_scope.get("public_test"), label="diagnostic public_test"
    )
    if any(
        diagnostic_public.get(name) is not False
        for name in ("opened", "read", "hashed", "scored")
    ) or diagnostic_scope.get("reference_model_training_or_scoring") is not False:
        raise RuntimeError("exploratory diagnostic scope is contaminated")

    decision_path = _resolve_declared_path(
        evidence["v3_failed_development_decision"]["path"],
        artifact_root=artifact_root,
    )
    decision = _load_json(decision_path, label="v3 failed development decision")
    if decision.get("status") != "failed_development":
        raise RuntimeError("v3 development decision is not the frozen failure")
    decision_public = _required_mapping(
        decision.get("public_test"), label="v3 failed decision public_test"
    )
    if any(
        decision_public.get(name) is not False
        for name in ("opened", "read", "hashed", "scored")
    ):
        raise RuntimeError("v3 failed decision did not keep Public closed")

    support_path = _resolve_declared_path(
        evidence["v4_action_support_audit"]["path"], artifact_root=artifact_root
    )
    support = _load_json(support_path, label="v4 action-support audit")
    if support.get("passed") is not True or support.get("status") != "passed":
        raise RuntimeError("v4 action-support audit did not pass")
    support_scope = _required_mapping(
        support.get("scope"), label="v4 action-support scope"
    )
    if support_scope.get("public_test_opened") is not False or support_scope.get(
        "total_concrete_profiles"
    ) != 2304:
        raise RuntimeError("v4 action-support audit scope mismatch")
    namespace = _required_mapping(
        support_scope.get("formal_catalog_namespace"),
        label="v4 action-support formal catalog namespace",
    )
    if (
        namespace.get("catalog_index_offset") != 1_000_000
        or namespace.get("offset_modulo_anchor_count") != 0
        or namespace.get("preformal_catalog_indices_excluded") != [0, 1]
    ):
        raise RuntimeError("v4 action-support formal catalog namespace mismatch")

    preformal_path = _resolve_declared_path(
        evidence["v4_preformal_content_receipt"]["path"],
        artifact_root=artifact_root,
    )
    preformal = _load_json(preformal_path, label="v4 preformal content receipt")
    if preformal.get("protocol_id") != PROTOCOL or preformal.get(
        "status"
    ) != RECEIPT_STATUS or preformal.get("checks_passed") is not True:
        raise RuntimeError("v4 preformal content receipt identity mismatch")
    preformal_public = _required_mapping(
        preformal.get("public_test"), label="v4 preformal public_test"
    )
    if any(
        preformal_public.get(name) is not False
        for name in ("opened", "read", "hashed", "scored")
    ) or preformal.get("reference_model_training_or_scoring") is not False:
        raise RuntimeError("v4 preformal content receipt scope is contaminated")
    reconstruction = _required_mapping(
        preformal.get("reconstruction_contract"),
        label="v4 preformal reconstruction contract",
    )
    if reconstruction.get("lance_opened_or_generated") is not False or (
        reconstruction.get("formal_build_attempted") is not False
    ):
        raise RuntimeError("v4 preformal receipt consumed a Lance/formal build")
    if int(preformal.get("excluded_source_episode_count", -1)) != 17:
        raise RuntimeError("v4 preformal episode count mismatch")
    content = _required_mapping(
        preformal.get("prior_content_exclusions"),
        label="v4 preformal content exclusions",
    )
    if any(
        not isinstance(content.get(field), Mapping)
        or int(content[field].get("count", -1)) != 18
        for field in (
            "action_profile_ids",
            "scene_template_content_hashes",
            "pair_content_hashes",
            "query_pixel_hashes",
        )
    ):
        raise RuntimeError("v4 preformal content counts mismatch")
    return evidence


def _stream_action_identity(source_h5: Path, *, chunk_rows: int = 131_072) -> dict[str, Any]:
    all_digest = hashlib.sha256()
    finite_digest = hashlib.sha256()
    count = 0
    mean = np.zeros(5, dtype=np.float64)
    m2 = np.zeros(5, dtype=np.float64)
    with h5py.File(source_h5, "r", swmr=True) as handle:
        if "action" not in handle or "ep_len" not in handle:
            raise RuntimeError("source H5 lacks action or ep_len")
        action = handle["action"]
        if tuple(action.shape) != ACTION_SHAPE or str(action.dtype) != ACTION_DTYPE:
            raise RuntimeError("source H5 action shape/dtype mismatch")
        episodes = int(handle["ep_len"].shape[0])
        for start in range(0, ACTION_SHAPE[0], chunk_rows):
            stop = min(start + chunk_rows, ACTION_SHAPE[0])
            raw = np.asarray(action[start:stop])
            canonical = np.ascontiguousarray(raw, dtype="<f4")
            all_digest.update(canonical.tobytes(order="C"))
            mask = np.isfinite(canonical).all(axis=1)
            finite_f32 = np.ascontiguousarray(canonical[mask], dtype="<f4")
            finite_digest.update(finite_f32.tobytes(order="C"))
            finite = finite_f32.astype(np.float64)
            if not finite.size:
                continue
            batch_count = int(finite.shape[0])
            batch_mean = np.mean(finite, axis=0, dtype=np.float64)
            centered = finite - batch_mean
            batch_m2 = np.sum(centered * centered, axis=0, dtype=np.float64)
            combined = count + batch_count
            delta = batch_mean - mean
            m2 += batch_m2 + delta * delta * (count * batch_count / combined)
            mean += delta * (batch_count / combined)
            count = combined
    if count <= 0:
        raise RuntimeError("source H5 has no finite actions")
    std_f32 = np.sqrt(m2 / count).astype("<f4")
    return {
        "row_count": ACTION_SHAPE[0],
        "episode_count": episodes,
        "action_dataset": {
            "name": "action",
            "shape": list(ACTION_SHAPE),
            "dtype": ACTION_DTYPE,
            "finite_row_count": count,
            "excluded_nonfinite_row_count": ACTION_SHAPE[0] - count,
            "row_major_little_endian_float32_sha256": all_digest.hexdigest(),
            "finite_rows_content_sha256": finite_digest.hexdigest(),
            "population_std_float32": [float(value) for value in std_f32],
            "population_std_float32_sha256": hashlib.sha256(
                std_f32.tobytes(order="C")
            ).hexdigest(),
        },
    }


def _validate_source(
    document: Mapping[str, Any], *, source_h5: Path
) -> dict[str, Any]:
    source_contract = _required_mapping(
        document.get("source_and_catalog"), label="source_and_catalog"
    )
    if source_contract.get("source_symbol") != SOURCE_SYMBOL or source_contract.get(
        "formal_source_must_be_supplied_explicitly"
    ) is not True:
        raise RuntimeError("source path/symbol contract mismatch")
    declared = _required_mapping(
        source_contract.get("frozen_source_identity"),
        label="frozen_source_identity",
    )
    expected_declared = {
        "row_count": SOURCE_ROW_COUNT,
        "episode_count": SOURCE_EPISODE_COUNT,
        "size_bytes": SOURCE_SIZE_BYTES,
        "sha256": SOURCE_FILE_SHA256,
    }
    if {key: declared.get(key) for key in expected_declared} != expected_declared:
        raise RuntimeError("prereg source H5 identity differs from frozen identity")
    action_declared = _required_mapping(
        declared.get("action_dataset"), label="frozen source action_dataset"
    )
    expected_action = {
        "name": "action",
        "shape": list(ACTION_SHAPE),
        "dtype": ACTION_DTYPE,
        "finite_row_count": ACTION_FINITE_ROW_COUNT,
        "excluded_nonfinite_row_count": ACTION_EXCLUDED_NONFINITE_ROW_COUNT,
        "row_major_little_endian_float32_sha256": ACTION_DATA_SHA256,
        "finite_rows_content_sha256": ACTION_FINITE_CONTENT_SHA256,
        "population_std_float32": _canonical_float32_vector(
            ACTION_STD_FLOAT32,
            length=ACTION_SHAPE[1],
            label="frozen ACTION_STD_FLOAT32",
        ),
        "population_std_float32_sha256": ACTION_STD_FLOAT32_SHA256,
    }
    declared_action = {key: action_declared.get(key) for key in expected_action}
    declared_action["population_std_float32"] = _canonical_float32_vector(
        action_declared.get("population_std_float32"),
        length=ACTION_SHAPE[1],
        label="frozen source action_dataset.population_std_float32",
    )
    if declared_action != expected_action:
        raise RuntimeError("prereg action dataset identity differs from frozen identity")
    if source_h5.is_symlink() or not source_h5.is_file():
        raise FileNotFoundError("explicit source H5 must be a regular non-symlink file")
    if source_h5.stat().st_size != SOURCE_SIZE_BYTES:
        raise RuntimeError("source H5 byte size mismatch")
    if file_sha256(source_h5) != SOURCE_FILE_SHA256:
        raise RuntimeError("source H5 full-file SHA256 mismatch")
    observed = _stream_action_identity(source_h5)
    if observed["row_count"] != SOURCE_ROW_COUNT or observed[
        "episode_count"
    ] != SOURCE_EPISODE_COUNT:
        raise RuntimeError("source H5 row/episode count mismatch")
    observed_action = dict(
        _required_mapping(
            observed.get("action_dataset"), label="observed source action_dataset"
        )
    )
    observed_action["population_std_float32"] = _canonical_float32_vector(
        observed_action.get("population_std_float32"),
        length=ACTION_SHAPE[1],
        label="observed source action_dataset.population_std_float32",
    )
    if observed_action != expected_action:
        raise RuntimeError("source H5 action content/statistics identity mismatch")
    return {
        "symbol": SOURCE_SYMBOL,
        "path_recorded": False,
        "sha256": SOURCE_FILE_SHA256,
        "size_bytes": SOURCE_SIZE_BYTES,
        "row_count": SOURCE_ROW_COUNT,
        "episode_count": SOURCE_EPISODE_COUNT,
        "action_dataset": expected_action,
    }


def freeze(
    *,
    prereg_path: Path,
    artifact_root: Path,
    source_h5: Path,
    output: Path,
) -> dict[str, Any]:
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite existing receipt {output}")
    forbidden = _forbidden_public_component(output)
    if forbidden is not None:
        raise RuntimeError(f"refusing Public output path component {forbidden!r}")
    if not prereg_path.is_file() or prereg_path.is_symlink():
        raise FileNotFoundError("prereg must be a regular non-symlink file")
    document = yaml.safe_load(prereg_path.read_text(encoding="utf-8"))
    document = _required_mapping(document, label="preregistration")
    if _contains_placeholder(document):
        raise RuntimeError("Preregistration still contains an identity placeholder")
    _validate_public_and_scope(document)
    scientific_change = _validate_scientific_change(document)
    probe = _validate_probe_contract(document)
    identity = _verified_named_entries(
        document.get("identity"),
        required_names=REQUIRED_IDENTITY_KEYS,
        artifact_root=artifact_root,
        label="identity",
    )
    prior_basis = _validate_prior_basis(document, artifact_root=artifact_root)
    evidence = _validate_evidence(document, artifact_root=artifact_root)
    source = _validate_source(document, source_h5=source_h5)

    # These names and structures are consumed directly by the v4 builder.
    receipt = {
        "schema_version": 1,
        "protocol_id": PROTOCOL,
        "status": RECEIPT_STATUS,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Training_and_Development_data_and_rgb_probe_only",
        "preregistration": {
            "path": prereg_path.relative_to(ROOT).as_posix()
            if prereg_path.is_relative_to(ROOT)
            else prereg_path.as_posix(),
            "sha256": file_sha256(prereg_path),
            "size_bytes": prereg_path.stat().st_size,
        },
        "identity": identity,
        "scientific_change": scientific_change,
        "rgb_history_probe": probe,
        "prior_episode_exclusion_basis": prior_basis,
        "frozen_evidence": evidence,
        "source_h5": source,
        "authorized_splits": list(ACTIVE_SPLITS),
        "public_test": {
            "access_status": "closed_not_read_not_scored",
            "opened": False,
            "read": False,
            "scored": False,
            "hashed": False,
        },
        "reference_model_training_or_scoring_authorized": False,
        "checks_passed": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return receipt


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    values = list(argv) if argv is not None else None
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prereg", type=Path, default=DEFAULT_PREREG)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--source-h5", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(values)
    for path in (args.prereg, args.artifact_root, args.source_h5, args.output):
        forbidden = _forbidden_public_component(path)
        if forbidden is not None:
            raise ValueError(f"refusing Public path component {forbidden!r}")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    receipt = freeze(
        prereg_path=args.prereg.expanduser().resolve(),
        artifact_root=args.artifact_root.expanduser().resolve(),
        source_h5=args.source_h5.expanduser().resolve(),
        output=args.output.expanduser().resolve(),
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "preregistration_sha256": receipt["preregistration"]["sha256"],
                "checks_passed": receipt["checks_passed"],
                "public_test_read": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
