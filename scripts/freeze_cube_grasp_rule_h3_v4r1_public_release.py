#!/usr/bin/env python3
"""Freeze the one-use Cube v4r1 Public generation/scoring authorization."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contextworld.benchmarks.cube_grasp_rule_public_contract import (  # noqa: E402
    DEFAULT_FREEZE_RECEIPT,
    DEFAULT_PREREGISTRATION,
    FREEZE_RECEIPT_ID,
    FREEZE_STATUS,
    PREREGISTRATION_ID,
    PREREGISTRATION_STATUS,
    PROTOCOL_ID,
    PUBLIC_CANDIDATE_ASSIGNMENT_SEED,
    PUBLIC_CATALOG_INDEX_OFFSET,
    PUBLIC_CATALOG_SEED,
    PUBLIC_PAIR_COUNT,
    PUBLIC_PROFILE_SEED,
    PUBLIC_SPLIT,
    file_identity,
    read_json_nofollow,
    read_yaml_nofollow,
    validate_public_preregistration_contract,
    validate_public_freeze_receipt_contract,
)
from contextworld.paths import resolve_contextworld_path  # noqa: E402
from scripts.finalize_cube_grasp_rule_h3_v4r1_prior_exclusions import (  # noqa: E402
    canonical_content_digest,
    excluded_source_episodes_sha256,
)


CONTENT_FIELDS = (
    "action_profile_ids",
    "scene_template_content_hashes",
    "pair_content_hashes",
    "query_pixel_hashes",
)
EXPECTED_ANCHORS = ("endpoint4", "front_hold", "plateau", "ramp4")
EXPECTED_SEEDS = (17321, 17322, 17323)
EXPECTED_RECIPE = "mixed_frozen_image_paired_future_fit_1p00"
EXPECTED_ACTION_MEAN = (
    0.010884696617722511,
    -0.003141433000564575,
    0.002646582666784525,
    0.00042392866453155875,
    0.1592525690793991,
)
EXPECTED_ACTION_STD = (
    0.28941982984542847,
    0.393716961145401,
    0.6431365013122559,
    0.3928016126155853,
    0.2503073513507843,
)


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _identity_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return all(
        left.get(name) == right.get(name)
        for name in ("path", "sha256", "size_bytes")
    )


def _closed_public_state(value: Any, *, label: str) -> None:
    public = _mapping(value, label=label)
    if public.get("access_status") != "closed_not_read_not_scored":
        raise RuntimeError(f"{label} is not closed")
    for name in ("generated", "opened", "read", "hashed", "scored"):
        if public.get(name) is not False:
            raise RuntimeError(f"{label}.{name} must be false")


def _load_bound_json(
    entry: Mapping[str, Any], *, label: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = resolve_contextworld_path(str(entry.get("path", "")))
    observed = file_identity(path, logical_path=str(entry.get("path", "")))
    if not _identity_equal(observed, entry):
        raise RuntimeError(f"{label} identity drifted")
    _, payload = read_json_nofollow(path, label=label)
    return observed, payload


def _validate_implementation_identities(
    prereg: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    identity = _mapping(prereg.get("identity"), label="identity")
    entries = _mapping(
        identity.get("implementation"), label="identity.implementation"
    )
    if not entries:
        raise ValueError("identity.implementation cannot be empty")
    result: dict[str, dict[str, Any]] = {}
    for name, raw in entries.items():
        entry = _mapping(raw, label=f"identity.implementation.{name}")
        path = resolve_contextworld_path(str(entry.get("path", "")))
        observed = file_identity(path, logical_path=str(entry.get("path", "")))
        if not _identity_equal(observed, entry):
            raise RuntimeError(f"implementation identity drifted: {name}")
        result[str(name)] = observed
    return result


def _validate_preregistration_contract(prereg: Mapping[str, Any]) -> None:
    if (
        prereg.get("schema_version") != 1
        or prereg.get("preregistration_id") != PREREGISTRATION_ID
        or prereg.get("protocol_id") != PROTOCOL_ID
        or prereg.get("status") != PREREGISTRATION_STATUS
        or prereg.get("phase") != "public_generation_and_evaluation_only"
    ):
        raise RuntimeError("Cube Public preregistration identity/status mismatch")
    validate_public_preregistration_contract(prereg)
    _closed_public_state(
        prereg.get("public_test_before_freeze"),
        label="preregistration Public state",
    )

    scope = _mapping(prereg.get("scope"), label="scope")
    expected_scope = {
        "environment": "Cube",
        "capability": "does_gripper_lift_move_the_cube",
        "history_tokens": 3,
        "context_transitions": 2,
        "raw_action_dim": 5,
        "raw_steps_per_action_block": 5,
        "flattened_action_input_dim": 25,
        "prediction_horizon_action_blocks": 1,
    }
    if any(scope.get(name) != value for name, value in expected_scope.items()):
        raise RuntimeError("Cube Public scope/action contract drifted")
    if scope.get("grasp_modes") != ["cannot_hold", "can_hold"]:
        raise RuntimeError("Cube Public grasp-mode contract drifted")

    generation = _mapping(
        prereg.get("public_data_generation"), label="public_data_generation"
    )
    expected_generation = {
        "split": PUBLIC_SPLIT,
        "pair_count": PUBLIC_PAIR_COUNT,
        "candidate_pool_count": 2 * PUBLIC_PAIR_COUNT,
        "catalog_index_offset": PUBLIC_CATALOG_INDEX_OFFSET,
        "candidate_assignment_seed": PUBLIC_CANDIDATE_ASSIGNMENT_SEED,
        "catalog_seed": PUBLIC_CATALOG_SEED,
        "profile_seed": PUBLIC_PROFILE_SEED,
        "workers": 16,
        "jpeg_quality": 95,
    }
    if any(
        generation.get(name) != value
        for name, value in expected_generation.items()
    ):
        raise RuntimeError("Cube Public generation recipe drifted")
    if generation.get("action_templates") != list(EXPECTED_ANCHORS):
        raise RuntimeError("Cube Public action-template contract drifted")
    if (
        generation.get("pair_balanced") is not True
        or generation.get("split_disjoint_from_all_non_public_content")
        is not True
        or generation.get("action_profile_sum_zero") is not True
        or generation.get("action_profile_last_zero") is not True
    ):
        raise RuntimeError("Cube Public v4r1 causal/data constraints drifted")

    evaluation = _mapping(
        prereg.get("public_evaluation"), label="public_evaluation"
    )
    checkpoints = evaluation.get("checkpoints")
    if (
        evaluation.get("authorized_model_families") != ["lewm"]
        or not isinstance(checkpoints, list)
        or len(checkpoints) != 3
        or [int(row.get("training_seed", -1)) for row in checkpoints]
        != list(EXPECTED_SEEDS)
        or any(row.get("model_family") != "lewm" for row in checkpoints)
        or any(row.get("training_recipe") != EXPECTED_RECIPE for row in checkpoints)
        or any(int(row.get("checkpoint_step", -1)) != 4096 for row in checkpoints)
        or evaluation.get("checkpoint_or_recipe_selection_after_freeze")
        is not False
        or evaluation.get("training_authorized") is not False
        or evaluation.get("public_data_loaded_once_for_all_checkpoints")
        is not True
    ):
        raise RuntimeError("Cube Public checkpoint/method contract drifted")
    devices = evaluation.get("devices")
    if not isinstance(devices, list) or len(devices) != 3 or len(set(devices)) != 3:
        raise RuntimeError("Cube Public requires three fixed distinct devices")
    if int(evaluation.get("batch_size", -1)) != 64:
        raise RuntimeError("Cube Public batch size drifted")
    if evaluation.get("data_access_contract") != {
        "product_kind": "reproducible_local_public_test_not_sealed_leaderboard",
        "future_target_frame_available_to_evaluator": True,
        "model_visible_fields": ["pixels", "action_block"],
        "adapter_receives_only": ["history_pixels", "query_action_blocks"],
        "privileged_columns_not_exposed_to_model": [
            "physical_state",
            "hidden_grasp_enabled",
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
        ],
    }:
        raise RuntimeError("Cube Public model/evaluator data boundary drifted")
    normalization = _mapping(
        evaluation.get("action_normalization"),
        label="public_evaluation.action_normalization",
    )
    mean = normalization.get("mean")
    std = normalization.get("std_population")
    if (
        set(normalization)
        != {
            "source",
            "finite_action_rows",
            "excluded_nonfinite_rows",
            "mean",
            "std_population",
        }
        or normalization.get("source")
        != "original_cube_h5_finite_actions_population_zscore"
        or int(normalization.get("finite_action_rows", -1)) != 2_000_000
        or int(normalization.get("excluded_nonfinite_rows", -1)) != 10_000
        or not isinstance(mean, list)
        or not isinstance(std, list)
        or len(mean) != 5
        or len(std) != 5
        or tuple(mean) != EXPECTED_ACTION_MEAN
        or tuple(std) != EXPECTED_ACTION_STD
        or not all(math.isfinite(float(value)) for value in [*mean, *std])
        or not all(float(value) > 0.0 for value in std)
    ):
        raise RuntimeError("Cube Public action normalization drifted")

    scoring = _mapping(prereg.get("scoring"), label="scoring")
    hidden = _mapping(
        scoring.get("hidden_future_prediction"),
        label="scoring.hidden_future_prediction",
    )
    gates = _mapping(hidden.get("gates"), label="scoring gates")
    expected_gates = {
        "correct_future_rate_minimum": 0.75,
        "correct_history_rate_minimum": 0.75,
        "context_switch_rate_minimum": 0.90,
        "worst_rule_correct_future_rate_minimum": 0.70,
        "target_latent_separation_required": True,
        "response_gain_minimum": 0.50,
        "normalized_response_error_strict_maximum": 1.00,
    }
    if dict(gates) != expected_gates:
        raise RuntimeError("Cube Public scoring gates drifted")
    uncertainty = _mapping(hidden.get("uncertainty"), label="uncertainty")
    if (
        uncertainty.get("method") != "paired_query_bootstrap"
        or int(uncertainty.get("resamples", -1)) != 10_000
        or int(uncertainty.get("random_seed", -1)) != 2026080314
        or float(uncertainty.get("confidence_level", -1.0)) != 0.95
        or uncertainty.get("lower_bound_minimum")
        != {
            "correct_future_rate": 0.70,
            "correct_history_rate": 0.70,
            "context_switch_rate": 0.85,
        }
    ):
        raise RuntimeError("Cube Public uncertainty contract drifted")


def _validate_basis(
    prereg: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    entries = _mapping(
        prereg.get("authorization_basis"), label="authorization_basis"
    )
    required = {
        "data_readiness_decision",
        "development_build_report",
        "prior_exclusion_receipt",
        "reference_training_preregistration",
        "reference_training_freeze_receipt",
        "reference_training_protocol",
        "reference_development_score",
        "reference_development_decision",
        "retention_preregistration",
        "retention_freeze_receipt",
        "original_task_retention_decision",
    }
    if set(entries) != required:
        raise RuntimeError("Cube Public authorization-basis set drifted")
    identities: dict[str, dict[str, Any]] = {}
    payloads: dict[str, dict[str, Any]] = {}
    for name, raw in entries.items():
        entry = _mapping(raw, label=f"authorization_basis.{name}")
        path = resolve_contextworld_path(str(entry.get("path", "")))
        observed = file_identity(path, logical_path=str(entry.get("path", "")))
        if not _identity_equal(observed, entry):
            raise RuntimeError(f"authorization basis drifted: {name}")
        identities[name] = observed
        if path.suffix == ".json":
            _, payloads[name] = read_json_nofollow(path, label=name)

    data_decision = payloads["data_readiness_decision"]
    if data_decision.get("status") != "passed_development":
        raise RuntimeError("Cube v4r1 data-readiness decision did not pass")
    _closed_public_state(
        data_decision.get("public_test"), label="data-readiness Public state"
    )

    development = payloads["reference_development_decision"]
    families = _mapping(development.get("families"), label="Development families")
    if (
        development.get("status") != "passed_development"
        or development.get("passing_families") != ["lewm"]
        or families.get("lewm", {}).get("passed") is not True
        or int(families.get("lewm", {}).get("checkpoints_passed", -1)) != 3
        or families.get("pldm", {}).get("passed") is not False
        or int(families.get("pldm", {}).get("checkpoints_passed", -1)) != 0
    ):
        raise RuntimeError("Cube reference Development decision is not eligible")
    _closed_public_state(
        development.get("public_test"), label="Development Public state"
    )

    retention = payloads["original_task_retention_decision"]
    comparisons = retention.get("comparisons")
    if (
        retention.get("status") != "passed_retention"
        or retention.get("passing_families") != ["lewm"]
        or not isinstance(comparisons, list)
        or len(comparisons) != 3
        or sorted(int(row.get("training_seed", -1)) for row in comparisons)
        != list(EXPECTED_SEEDS)
        or any(row.get("model_family") != "lewm" for row in comparisons)
        or any(row.get("passed") is not True for row in comparisons)
    ):
        raise RuntimeError("Cube original-task retention decision is not eligible")
    _closed_public_state(
        retention.get("public_test"), label="retention Public state"
    )
    return identities, payloads


def _validate_checkpoint_chain(
    prereg: Mapping[str, Any],
    *,
    payloads: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    checkpoints = prereg["public_evaluation"]["checkpoints"]
    score = payloads["reference_development_score"]
    results = score.get("checkpoint_results")
    if (
        score.get("status") != "completed"
        or score.get("passed") is not True
        or score.get("model_family") != "lewm"
        or score.get("training_seeds") != list(EXPECTED_SEEDS)
        or score.get("training_recipe") != EXPECTED_RECIPE
        or not isinstance(results, list)
        or len(results) != 3
        or any(row.get("gate", {}).get("passed") is not True for row in results)
    ):
        raise RuntimeError("Cube LeWM Development score is incomplete")
    by_seed = {int(row["model"]["training_seed"]): row for row in results}
    retention_by_seed = {
        int(row["training_seed"]): row
        for row in payloads["original_task_retention_decision"]["comparisons"]
    }
    frozen: dict[str, dict[str, Any]] = {}
    for checkpoint in checkpoints:
        seed = int(checkpoint["training_seed"])
        result = by_seed.get(seed)
        if result is None or result.get("gate", {}).get("passed") is not True:
            raise RuntimeError(f"Cube LeWM seed {seed} did not pass Development")
        training_checkpoint = result.get("model", {}).get("training_checkpoint", {})
        expected = {
            name: checkpoint[name]
            for name in ("path", "sha256", "size_bytes", "model_state_sha256")
        }
        observed_from_score = {
            name: training_checkpoint.get(name)
            for name in ("path", "sha256", "size_bytes", "model_state_sha256")
        }
        if observed_from_score != expected:
            raise RuntimeError(f"Cube LeWM seed {seed} Development identity drifted")
        model = result.get("model", {})
        if (
            model.get("family") != "lewm"
            or model.get("training_recipe") != EXPECTED_RECIPE
            or model.get("state_sha256_before") != checkpoint["model_state_sha256"]
            or model.get("state_sha256_after") != checkpoint["model_state_sha256"]
        ):
            raise RuntimeError(f"Cube LeWM seed {seed} model-state chain drifted")
        retention = retention_by_seed.get(seed, {})
        if (
            retention.get("checkpoint") != checkpoint["path"]
            or retention.get("checkpoint_sha256") != checkpoint["sha256"]
            or retention.get("passed") is not True
        ):
            raise RuntimeError(f"Cube LeWM seed {seed} retention identity drifted")
        path = Path(str(checkpoint["path"])).expanduser()
        observed_file = file_identity(path, logical_path=str(checkpoint["path"]))
        if not _identity_equal(observed_file, checkpoint):
            raise RuntimeError(f"Cube LeWM seed {seed} checkpoint file drifted")
        frozen[f"lewm_checkpoint_seed{seed}"] = {
            **observed_file,
            "model_state_sha256": checkpoint["model_state_sha256"],
            "rehash_on_entrypoint": True,
        }
    return frozen


def _union_receipt(
    prereg: Mapping[str, Any],
    *,
    prior: Mapping[str, Any],
    build: Mapping[str, Any],
) -> dict[str, Any]:
    if prior.get("checks_passed") is not True or build.get("passed") is not True:
        raise RuntimeError("Cube exclusion/build prerequisites did not pass")
    if (
        build.get("active_splits") != ["train", "loader_validation"]
        or build.get("public_test_generated") is not False
        or build.get("public_test_opened") is not False
        or build.get("cross_split_audit", {}).get("passed") is not True
    ):
        raise RuntimeError("Cube v4r1 build does not preserve Public isolation")
    splits = _mapping(build.get("splits"), label="v4r1 build splits")
    if set(splits) != {"train", "loader_validation"}:
        raise RuntimeError("Cube v4r1 build split set drifted")
    for split_name, pair_count in (("train", 2048), ("loader_validation", 256)):
        split = _mapping(splits[split_name], label=f"build split {split_name}")
        if (
            split.get("passed") is not True
            or int(split.get("pair_count", -1)) != pair_count
            or split.get("action_anchor_counts")
            != {name: pair_count // 4 for name in EXPECTED_ANCHORS}
            or split.get("prior_episode_and_content_exclusion", {}).get("passed")
            is not True
        ):
            raise RuntimeError(f"Cube v4r1 split failed: {split_name}")

    source_values = set(int(value) for value in prior["excluded_source_episodes"])
    for split in splits.values():
        source_values.update(int(value) for value in split["source_episodes"])
    source = sorted(source_values)
    expected = _mapping(
        prereg["public_data_generation"].get("exclusion_union"),
        label="public_data_generation.exclusion_union",
    )
    source_expected = _mapping(expected.get("source_episodes"), label="source union")
    source_digest = excluded_source_episodes_sha256(source)
    if (
        len(source) != int(source_expected.get("count", -1))
        or source_digest != source_expected.get("sha256")
    ):
        raise RuntimeError("Cube Public source-episode exclusion union drifted")

    content: dict[str, dict[str, Any]] = {}
    build_keys = {
        "action_profile_ids": "action_profile_ids",
        "scene_template_content_hashes": "scene_template_content_hashes",
        "pair_content_hashes": "pair_content_hashes",
        "query_pixel_hashes": "query_hashes",
    }
    for field in CONTENT_FIELDS:
        values = set(
            str(value)
            for value in prior["prior_content_exclusions"][field]["values"]
        )
        for split in splits.values():
            values.update(str(value) for value in split[build_keys[field]])
        normalized = sorted(values)
        digest = canonical_content_digest(normalized, field_name=field)
        field_expected = _mapping(expected.get(field), label=f"{field} union")
        if (
            len(normalized) != int(field_expected.get("count", -1))
            or digest != field_expected.get("sha256")
        ):
            raise RuntimeError(f"Cube Public {field} exclusion union drifted")
        content[field] = {
            "count": len(normalized),
            "sha256": digest,
            "values": normalized,
        }
    return {
        "checks_passed": True,
        "coverage": {
            "historical_prior_receipt": True,
            "v4r1_train": True,
            "v4r1_loader_validation": True,
            "public_content_included": False,
        },
        "excluded_source_episode_count": len(source),
        "excluded_source_episodes_sha256": source_digest,
        "excluded_source_episodes": source,
        "prior_content_exclusions": content,
    }


def _validate_runtime(
    prereg: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    specification = _mapping(
        prereg.get("runtime", {}).get("stable_worldmodel"),
        label="runtime.stable_worldmodel",
    )
    repo = Path(str(specification.get("repo", ""))).expanduser()
    if not repo.is_absolute():
        repo = (ROOT / repo).absolute()
    metadata = os.lstat(repo)
    if not stat.S_ISDIR(metadata.st_mode) or repo.is_symlink():
        raise RuntimeError("pinned Stable-WorldModel repo is missing or aliased")
    environment = os.environ.copy()
    environment["SUDO_UID"] = str(metadata.st_uid)
    git_prefix = ["git", "-c", f"safe.directory={repo}", "-C", str(repo)]
    commit = subprocess.run(
        [*git_prefix, "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    ).stdout.strip()
    dirty = subprocess.run(
        [*git_prefix, "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    ).stdout
    if commit != specification.get("expected_ref") or dirty:
        raise RuntimeError("pinned Stable-WorldModel runtime drifted")
    files: dict[str, dict[str, Any]] = {}
    for name, raw in _mapping(
        specification.get("required_files"), label="runtime required_files"
    ).items():
        entry = _mapping(raw, label=f"runtime required_files.{name}")
        path = repo / str(entry.get("path", ""))
        logical = str(path)
        observed = file_identity(path, logical_path=logical)
        expected = {
            "path": logical,
            "sha256": entry.get("sha256"),
            "size_bytes": entry.get("size_bytes"),
        }
        if not _identity_equal(observed, expected):
            raise RuntimeError(f"Stable-WorldModel runtime file drifted: {name}")
        files[f"stable_worldmodel_{name}"] = {
            **observed,
            "rehash_on_entrypoint": True,
        }
    return {
        "path": str(repo),
        "commit": commit,
        "clean_worktree": True,
        "required_files": {
            name.removeprefix("stable_worldmodel_"): value
            for name, value in files.items()
        },
    }, files


def _source_h5_receipt(prereg: Mapping[str, Any]) -> dict[str, Any]:
    source = _mapping(
        prereg["public_data_generation"].get("source_h5"),
        label="public_data_generation.source_h5",
    )
    path = Path(str(source.get("path", ""))).expanduser()
    metadata = os.lstat(path)
    if not stat.S_ISREG(metadata.st_mode) or path.is_symlink():
        raise RuntimeError("Cube source H5 is missing or aliased")
    if int(metadata.st_size) != int(source.get("size_bytes", -1)):
        raise RuntimeError("Cube source H5 size drifted")
    digest = str(source.get("sha256", ""))
    if len(digest) != 64:
        raise RuntimeError("Cube source H5 frozen digest is malformed")
    return {
        "path": str(source["path"]),
        "sha256": digest,
        "size_bytes": int(metadata.st_size),
        "row_count": int(source["row_count"]),
        "episode_count": int(source["episode_count"]),
        "action_dim": int(source["action_dim"]),
        "content_rehash_deferred_to_public_builder_before_candidate_selection": True,
        "rehash_on_entrypoint": False,
    }


def _assert_absent(path: Path, *, label: str) -> None:
    try:
        os.lstat(path)
    except FileNotFoundError:
        return
    raise FileExistsError(f"one-use {label} already exists: {path}")


def freeze_public_release(
    *, preregistration: Path, output: Path
) -> dict[str, Any]:
    preregistration = preregistration.expanduser()
    if not preregistration.is_absolute():
        preregistration = ROOT / preregistration
    preregistration = preregistration.absolute()
    output = output.expanduser().resolve()
    prereg_raw, prereg = read_yaml_nofollow(
        preregistration, label="Cube Public preregistration"
    )
    _validate_preregistration_contract(prereg)
    planned = _mapping(prereg.get("planned_artifacts"), label="planned_artifacts")
    expected_output = resolve_contextworld_path(str(planned["freeze_receipt"]))
    if output != expected_output:
        raise ValueError("Cube Public freeze output differs from preregistration")
    _assert_absent(output, label="freeze receipt")
    public_root = resolve_contextworld_path(str(planned["public_data_root"]))
    score_root = resolve_contextworld_path(str(planned["public_score_root"]))
    decision_path = resolve_contextworld_path(
        str(planned["public_release_decision"])
    )
    _assert_absent(public_root, label="Public data root")
    _assert_absent(score_root, label="Public score root")
    _assert_absent(decision_path, label="Public decision")

    implementations = _validate_implementation_identities(prereg)
    basis_identities, payloads = _validate_basis(prereg)
    checkpoint_inputs = _validate_checkpoint_chain(prereg, payloads=payloads)
    exclusions = _union_receipt(
        prereg,
        prior=payloads["prior_exclusion_receipt"],
        build=payloads["development_build_report"],
    )
    runtime, runtime_inputs = _validate_runtime(prereg)
    source_h5 = _source_h5_receipt(prereg)

    prereg_identity = {
        "path": str(prereg["identity"]["preregistration_path"]),
        "sha256": hashlib.sha256(prereg_raw).hexdigest(),
        "size_bytes": len(prereg_raw),
    }
    frozen_inputs: dict[str, dict[str, Any]] = {
        name: {**identity, "rehash_on_entrypoint": True}
        for name, identity in basis_identities.items()
    }
    frozen_inputs.update(checkpoint_inputs)
    frozen_inputs.update(runtime_inputs)
    frozen_inputs["source_h5"] = source_h5

    receipt = {
        "schema_version": 1,
        "receipt_id": FREEZE_RECEIPT_ID,
        "receipt_path": str(planned["freeze_receipt"]),
        "preregistration_id": PREREGISTRATION_ID,
        "protocol_id": PROTOCOL_ID,
        "status": FREEZE_STATUS,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "checks_passed": True,
        "preregistration": prereg_identity,
        "implementation_identities": implementations,
        "frozen_inputs": frozen_inputs,
        "runtime": {"stable_worldmodel": runtime},
        "public_exclusions": exclusions,
        "authorization": {
            "public_generation_once": True,
            "public_scoring_once_after_successful_generation": True,
            "authorized_model_families": ["lewm"],
            "training_seeds": list(EXPECTED_SEEDS),
            "training_or_checkpoint_selection": False,
            "threshold_or_recipe_changes": False,
            "public_test_rerun_after_access": False,
            "suite_registration": False,
        },
        "public_test": {
            "access_status": "authorized_not_generated_not_opened_not_read_not_scored",
            "generated": False,
            "opened": False,
            "read": False,
            "hashed": False,
            "scored": False,
        },
        "planned_artifacts": dict(planned),
    }
    validate_public_freeze_receipt_contract(
        prereg=prereg,
        freeze=receipt,
        root=ROOT,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return receipt


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prereg", type=Path, default=DEFAULT_PREREGISTRATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_FREEZE_RECEIPT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    receipt = freeze_public_release(
        preregistration=args.prereg,
        output=args.output,
    )
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "output": str(args.output),
                "authorized_model_families": receipt["authorization"][
                    "authorized_model_families"
                ],
                "public_test": receipt["public_test"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
