from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping

import yaml

from contextworld.paths import artifact_root, repository_root, resolve_contextworld_path


CUBE_REFERENCE_TRAINING_ID = (
    "contextworld_cube_gripper_carry_h3_v4r1_reference_training_v3"
)
CUBE_REFERENCE_TRAINING_PROTOCOL = (
    "cube_gripper_carry_rule_history3_v4r1_reference_training_v3"
)
DEFAULT_CUBE_REFERENCE_TRAINING_PREREG = repository_root() / (
    "configs/benchmark/"
    "cube_gripper_carry_h3_v4r1_reference_training_prereg_v3.yaml"
)
CUBE_REFERENCE_TRAINING_V2_ID = (
    "contextworld_cube_gripper_carry_h3_v4r1_reference_training_v2"
)
CUBE_REFERENCE_TRAINING_V2_PROTOCOL = (
    "cube_gripper_carry_rule_history3_v4r1_reference_training_v2"
)
CUBE_REFERENCE_TRAINING_V2_RECOVERY_EVIDENCE = {
    "preregistration": {
        "path": (
            "configs/benchmark/"
            "cube_gripper_carry_h3_v4r1_reference_training_prereg_v2.yaml"
        ),
        "sha256": "fbf382615a0b26792e789864de0466ead161c153bd546ff6cbbf725b6c44c015",
        "size_bytes": 20897,
    },
    "freeze_receipt": {
        "path": (
            "artifacts/evaluation/history3/"
            "cube_gripper_carry_h3_development_v4r1/"
            "reference_training_freeze_receipt_v2.json"
        ),
        "sha256": "cf075827edb4220b37aa4806bd2fe163c1c5e0597f459e250914025def36cb73",
        "size_bytes": 18382,
    },
    "matrix_request": {
        "path": (
            "artifacts/evaluation/history3/"
            "cube_gripper_carry_h3_development_v4r1/reference_training_v2/"
            "matrix_request.json"
        ),
        "sha256": "666f5a63040044af4c156b61ce89c283740bd13652f7945bc2b9289732b8fe12",
        "size_bytes": 7251,
    },
    "matrix_report": {
        "path": (
            "artifacts/evaluation/history3/"
            "cube_gripper_carry_h3_development_v4r1/reference_training_v2/"
            "matrix_report.json"
        ),
        "sha256": "823b6b067531ced3fb1e979333259befeb84202706b06d0bc847609b95a74950",
        "size_bytes": 2514,
    },
    "failure_receipt": {
        "path": (
            "artifacts/evaluation/history3/"
            "cube_gripper_carry_h3_development_v4r1/reference_training_v2/"
            "reference_training_v2_infrastructure_failure_receipt.json"
        ),
        "sha256": "56815a6b369c076cba4d9414d4312543811e239ac7671e3fb8fe2a97c5d7bf71",
        "size_bytes": 10104,
    },
}
PREREG_STATUS = "preregistered_before_reference_training"
FREEZE_STATUS = "frozen_before_reference_training"
AUTHORIZED_SPLITS = ("train", "loader_validation")
CUBE_TRAINING_PAIR_COUNTS = {"train": 2048, "loader_validation": 256}
CUBE_RAW_ACTION_DIM = 5
CUBE_ACTION_BLOCK_STEPS = 5
CUBE_ACTION_INPUT_DIM = 25


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_file_identity(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"Cube frozen input is not a regular file: {path}")
    before = path.stat()
    digest = file_sha256(path)
    after = path.stat()
    fields = ("st_size", "st_mtime_ns", "st_ino", "st_dev")
    if any(getattr(before, name) != getattr(after, name) for name in fields):
        raise RuntimeError(f"Cube frozen input changed while hashing: {path}")
    return {
        "sha256": digest,
        "size_bytes": int(after.st_size),
        "mtime_ns": int(after.st_mtime_ns),
        "inode": int(after.st_ino),
        "device": int(after.st_dev),
    }


def _regular_tree_files(path: Path) -> list[Path]:
    if not path.is_dir() or path.is_symlink():
        raise FileNotFoundError(f"Cube frozen tree is not a regular directory: {path}")
    files: list[Path] = []
    for child in sorted(path.rglob("*")):
        if child.is_symlink():
            raise RuntimeError(f"Cube frozen tree contains a symlink: {child}")
        if child.is_file():
            files.append(child)
    return files


def _directory_sha256(path: Path, *, excluded: frozenset[str] = frozenset()) -> str:
    digest = hashlib.sha256()
    for child in _regular_tree_files(path):
        relative = child.relative_to(path).as_posix()
        if relative in excluded:
            continue
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_sha256(child).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def cube_reference_data_tree_identity(
    prereg: Mapping[str, Any], *, repo_root: Path | None = None
) -> dict[str, Any]:
    """Verify and return the complete frozen v4r1 generated-data identity."""

    root = (repo_root or repository_root()).resolve()
    data_root = resolve_contextworld_path(
        str(prereg["data"]["artifact_tree"]["root"]), repo_root=root
    )
    files = _regular_tree_files(data_root)
    specification = prereg["data"]["artifact_tree"]
    observed_bytes = sum(path.stat().st_size for path in files)
    if (
        len(files) != int(specification["files"])
        or observed_bytes != int(specification["bytes"])
    ):
        raise RuntimeError("Cube v4r1 generated-data tree size/count drifted")
    without_success = _directory_sha256(
        data_root, excluded=frozenset({"_SUCCESS.json"})
    )
    if without_success != str(specification["tree_sha256_without_success_marker"]):
        raise RuntimeError("Cube v4r1 generated-data tree SHA256 drifted")
    tables: dict[str, Any] = {}
    for split, expected in prereg["data"]["table_tree_sha256"].items():
        table = data_root / str(prereg["data"]["lance_tables"][split])
        observed = _directory_sha256(table)
        if observed != str(expected):
            raise RuntimeError(f"Cube v4r1 {split} table SHA256 drifted")
        table_files = _regular_tree_files(table)
        tables[split] = {
            "path": str(table.relative_to(data_root)),
            "sha256": observed,
            "file_count": len(table_files),
            "bytes": sum(path.stat().st_size for path in table_files),
        }
    return {
        "root": str(specification["root"]),
        "file_count": len(files),
        "bytes": observed_bytes,
        "tree_sha256_without_success_marker": without_success,
        "tables": tables,
    }


def _mapping(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Cube reference-training field {field} must be a mapping")
    return value


def _require_keys(
    value: Mapping[str, Any], *, field: str, keys: tuple[str, ...]
) -> None:
    missing = [key for key in keys if key not in value]
    if missing:
        raise ValueError(
            f"Cube reference-training field {field} is missing: "
            + ", ".join(missing)
        )


def _validate_declared_identity(
    value: Any, *, field: str, size_key: str = "size_bytes"
) -> None:
    entry = _mapping(value, field=field)
    _require_keys(entry, field=field, keys=("sha256", size_key))
    digest = entry["sha256"]
    size = entry[size_key]
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        or isinstance(size, bool)
        or not isinstance(size, int)
        or size <= 0
    ):
        raise ValueError(f"Cube reference-training field {field} has a placeholder identity")


def _validate_closed_public(value: Any, *, field: str = "public_test") -> None:
    public = _mapping(value, field=field)
    if public.get("access_status") != "closed_not_read_not_scored":
        raise ValueError("Cube reference training requires closed Public Test")
    for name in ("generated", "opened", "read", "hashed", "scored"):
        if public.get(name) is not False:
            raise ValueError(f"Cube reference training requires {field}.{name}=false")
    if public.get("validation_lance_access_allowed") is not False:
        raise ValueError(
            "Cube reference training forbids Public validation Lance access"
        )


def _validate_path_specification(value: Any, *, field: str) -> None:
    specification = _mapping(value, field=field)
    if not any(
        specification.get(name)
        for name in (
            "environment_variable",
            "bundled_artifact_path",
            "local_source",
            "path",
            "checkpoint",
        )
    ):
        raise ValueError(f"Cube reference-training field {field} has no path source")


def _validate_shared_resolver_alias(
    value: Any, *, field: str, resolver_key: str
) -> None:
    specification = _mapping(value, field=field)
    _validate_path_specification(specification, field=field)
    local_source = specification.get("local_source")
    alias = specification.get(resolver_key)
    if (
        not isinstance(local_source, str)
        or not local_source
        or not isinstance(alias, str)
        or alias != local_source
    ):
        raise ValueError(
            f"Cube reference-training field {field}.{resolver_key} must equal "
            "its frozen local_source"
        )


def _validate_infrastructure_recovery(value: Any) -> None:
    recovery = _mapping(value, field="infrastructure_recovery")
    expected_header = {
        "source_preregistration_id": CUBE_REFERENCE_TRAINING_V2_ID,
        "source_protocol_id": CUBE_REFERENCE_TRAINING_V2_PROTOCOL,
        "authorization": (
            "new_preregistration_and_namespace_after_zero_step_"
            "infrastructure_failure"
        ),
        "failure_classification": (
            "pinned_runtime_optional_loss_constructor_mismatch_"
            "not_scientific_failure"
        ),
        "failure_stage": (
            "shared_engine_eager_conditional_sigreg_construction_before_"
            "optimizer_and_forward"
        ),
    }
    for name, expected in expected_header.items():
        if recovery.get(name) != expected:
            raise ValueError(f"Cube v3 infrastructure_recovery.{name} drifted")
    if recovery.get("root_cause") != {
        "pinned_class": "stable_worldmodel.wm.loss.ConditionalSIGReg",
        "unsupported_constructor_keyword_observed": "include_unpaired",
        "next_unsupported_constructor_keyword": "complete_haar_population",
        "shared_engine_eagerly_constructs_optional_losses": True,
        "authorized_recipes_use_conditional_sigreg": False,
        "shared_trainer_or_engine_change_required": False,
        "process_local_false_only_constructor_adapter_required": True,
    }:
        raise ValueError("Cube v3 infrastructure root-cause declaration drifted")
    if recovery.get("recovery_change") != {
        "conditional_sigreg_false_only_constructor_adapter_added": True,
        "shared_engine_constructor_preflight_added_to_freezer": True,
        "input_resolver_aliases_changed": False,
        "pinned_runtime_changed": False,
        "shared_trainer_or_engine_changed": False,
        "data_changed": False,
        "model_recipe_changed": False,
        "training_seeds_changed": False,
        "optimizer_or_threshold_changed": False,
        "public_test_access_changed": False,
    }:
        raise ValueError("Cube v3 recovery must change only runtime compatibility")
    prior = _mapping(
        recovery.get("prior_attempt"),
        field="infrastructure_recovery.prior_attempt",
    )
    if prior != CUBE_REFERENCE_TRAINING_V2_RECOVERY_EVIDENCE:
        raise ValueError("Cube v3 prior-attempt identity declaration drifted")
    for name, entry in prior.items():
        _validate_declared_identity(
            entry, field=f"infrastructure_recovery.prior_attempt.{name}"
        )
    if recovery.get("prior_training_state") != {
        "training_data_materialized": True,
        "loader_validation_materialized": True,
        "model_instantiated": True,
        "initial_checkpoint_loaded": True,
        "optimizer_instantiated": False,
        "forward_passes": 0,
        "backward_passes": 0,
        "optimizer_steps": 0,
        "checkpoints_created": 0,
        "training_reports_created": 0,
        "config_files_created": 6,
        "training_provenance_files_created": 6,
    }:
        raise ValueError("Cube v3 recovery requires a zero-step v2 attempt")
    if recovery.get("prior_retry") != {
        "authorized_under_v2": False,
        "v2_output_reusable": False,
        "new_preregistration_and_namespace_required": True,
    }:
        raise ValueError("Cube v3 retry boundary drifted")
    _validate_closed_public(
        recovery.get("public_test"),
        field="infrastructure_recovery.public_test",
    )


def _validate_preregistration(payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported Cube reference-training schema")
    if payload.get("preregistration_id") != CUBE_REFERENCE_TRAINING_ID:
        raise ValueError("Unexpected Cube reference-training preregistration id")
    if payload.get("protocol_id") != CUBE_REFERENCE_TRAINING_PROTOCOL:
        raise ValueError("Unexpected Cube reference-training protocol id")
    if payload.get("status") != PREREG_STATUS:
        raise ValueError("Cube reference-training preregistration is not frozen in intent")
    if payload.get("phase") != "development_only":
        raise ValueError("Cube reference training must be Development-only")

    scope = _mapping(payload.get("scope"), field="scope")
    expected_scope = {
        "environment": "Cube",
        "capability": "does_gripper_lift_move_the_cube",
        "history_tokens": 3,
        "context_transitions": 2,
        "raw_action_dim": CUBE_RAW_ACTION_DIM,
        "raw_steps_per_action_block": CUBE_ACTION_BLOCK_STEPS,
        "flattened_action_input_dim": CUBE_ACTION_INPUT_DIM,
        "prediction_horizon_action_blocks": 1,
        "public_test_included": False,
        "sealed_test_included": False,
    }
    for name, expected in expected_scope.items():
        if scope.get(name) != expected:
            raise ValueError(f"Unexpected Cube reference-training scope.{name}")
    if tuple(scope.get("authorized_splits", ())) != AUTHORIZED_SPLITS:
        raise ValueError("Cube reference training authorizes only Training/Development")
    if tuple(scope.get("grasp_modes", ())) != ("cannot_hold", "can_hold"):
        raise ValueError("Unexpected Cube grasp modes")
    _validate_closed_public(payload.get("public_test"))
    _validate_infrastructure_recovery(payload.get("infrastructure_recovery"))

    runtime = _mapping(payload.get("runtime"), field="runtime")
    stable = _mapping(
        runtime.get("stable_worldmodel"), field="runtime.stable_worldmodel"
    )
    _require_keys(
        stable,
        field="runtime.stable_worldmodel",
        keys=("repo", "expected_ref", "required_files"),
    )
    required_runtime_files = _mapping(
        stable["required_files"],
        field="runtime.stable_worldmodel.required_files",
    )
    if not required_runtime_files:
        raise ValueError("Cube reference training must bind Stable-WorldModel files")
    for name, entry in required_runtime_files.items():
        _require_keys(
            _mapping(
                entry,
                field=f"runtime.stable_worldmodel.required_files.{name}",
            ),
            field=f"runtime.stable_worldmodel.required_files.{name}",
            keys=("path", "sha256", "size_bytes"),
        )
        _validate_declared_identity(
            entry, field=f"runtime.stable_worldmodel.required_files.{name}"
        )

    data = _mapping(payload.get("data"), field="data")
    _require_keys(
        data,
        field="data",
        keys=(
            "protocol",
            "artifact_tree",
            "manifest_sha256",
            "pair_counts",
            "lance_tables",
            "table_tree_sha256",
            "artifacts",
        ),
    )
    if data["protocol"] != "cube_gripper_carry_rule_history3_development_v4":
        raise ValueError("Unexpected Cube v4r1 data protocol")
    if {
        name: int(value) for name, value in data["pair_counts"].items()
    } != CUBE_TRAINING_PAIR_COUNTS:
        raise ValueError("Cube v4r1 reference training requires 2048/256 pairs")
    if tuple(data["lance_tables"]) != AUTHORIZED_SPLITS:
        raise ValueError("Cube v4r1 preregistration has an unexpected Lance split")
    if set(data["lance_tables"]) != set(AUTHORIZED_SPLITS):
        raise ValueError("Cube v4r1 preregistration has an unexpected Lance split")
    if set(data["table_tree_sha256"]) != set(AUTHORIZED_SPLITS):
        raise ValueError("Cube v4r1 preregistration has incomplete table identities")
    for split, digest in data["table_tree_sha256"].items():
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"Cube v4r1 {split} table identity is invalid")
    if (
        tuple(data.get("anchor_families", ()))
        != ("endpoint4", "front_hold", "plateau", "ramp4")
        or data.get("pairs_per_anchor")
        != {"train": 512, "loader_validation": 64}
        or data.get("action_constraints")
        != {
            "temporal_pattern": ["p", "negative_p", "p", "terminal_zero"],
            "sum_p_target": 0.0,
            "final_p_target": 0.0,
        }
        or data.get("split_isolation")
        != {
            "source_episode_overlap": 0,
            "action_profile_overlap": 0,
            "scene_template_overlap": 0,
            "pair_content_overlap": 0,
            "query_pixel_overlap": 0,
        }
    ):
        raise ValueError("Cube v4r1 balanced/split-disjoint data contract drifted")
    if set(data["artifacts"]) != {
        "manifest",
        "build_report",
        "request",
        "success_marker",
        "data_readiness_decision",
        "rgb_history_probe",
        "recovery_freeze_receipt",
        "prior_exclusion_receipt",
    }:
        raise ValueError("Cube reference-training data evidence set is incomplete")
    for name, entry in data["artifacts"].items():
        _require_keys(
            _mapping(entry, field=f"data.artifacts.{name}"),
            field=f"data.artifacts.{name}",
            keys=("path", "sha256", "size_bytes"),
        )
        _validate_declared_identity(entry, field=f"data.artifacts.{name}")

    identity = _mapping(payload.get("identity"), field="identity")
    if not identity:
        raise ValueError("Cube reference-training identity set must not be empty")
    for name, entry in identity.items():
        _require_keys(
            _mapping(entry, field=f"identity.{name}"),
            field=f"identity.{name}",
            keys=("path", "sha256", "size_bytes"),
        )
        _validate_declared_identity(entry, field=f"identity.{name}")

    training = _mapping(payload.get("training"), field="training")
    upstream = _mapping(training.get("upstream"), field="training.upstream")
    _require_keys(
        upstream,
        field="training.upstream",
        keys=("original_h5", "original_lance"),
    )
    for name in ("original_h5", "original_lance"):
        _validate_shared_resolver_alias(
            upstream[name],
            field=f"training.upstream.{name}",
            resolver_key="path",
        )
    h5_identity = _mapping(
        upstream["original_h5"].get("expected_identity"),
        field="training.upstream.original_h5.expected_identity",
    )
    _validate_declared_identity(
        h5_identity, field="training.upstream.original_h5.expected_identity"
    )
    if (
        int(h5_identity.get("row_count", -1)) != 2_010_000
        or int(h5_identity.get("episode_count", -1)) != 10_000
        or int(h5_identity.get("action_dim", -1)) != CUBE_RAW_ACTION_DIM
    ):
        raise ValueError("Cube original H5 structural identity drifted")
    lance_identity = _mapping(
        upstream["original_lance"].get("expected_identity"),
        field="training.upstream.original_lance.expected_identity",
    )
    lance_files = _mapping(
        lance_identity.get("files"),
        field="training.upstream.original_lance.expected_identity.files",
    )
    for name, entry in lance_files.items():
        _validate_declared_identity(
            entry,
            field=f"training.upstream.original_lance.expected_identity.files.{name}",
        )
    if (
        int(lance_identity.get("row_count", -1)) != 2_010_000
        or int(lance_identity.get("action_dim", -1)) != CUBE_RAW_ACTION_DIM
        or int(lance_identity.get("file_count", -1)) != len(lance_files)
        or int(lance_identity.get("bytes", -1))
        != sum(int(entry["size_bytes"]) for entry in lance_files.values())
    ):
        raise ValueError("Cube original Lance structural identity drifted")
    matrix = _mapping(
        training.get("reference_matrix"), field="training.reference_matrix"
    )
    _require_keys(
        matrix,
        field="training.reference_matrix",
        keys=(
            "status",
            "training_seeds",
            "initial_checkpoints",
            "common",
            "models",
            "execution_policy",
        ),
    )
    if matrix["status"] != "planned_not_executed":
        raise ValueError("Cube v4r1 reference matrix must be unexecuted at freeze")
    seeds = matrix["training_seeds"]
    if (
        not isinstance(seeds, list)
        or len(seeds) != 3
        or len(set(seeds)) != 3
        or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds)
    ):
        raise ValueError("Cube v4r1 reference training requires three integer seeds")
    checkpoints = _mapping(
        matrix["initial_checkpoints"],
        field="training.reference_matrix.initial_checkpoints",
    )
    if set(checkpoints) != {"lewm", "pldm"}:
        raise ValueError("Cube reference training requires LeWM and PLDM checkpoints")
    for name, entry in checkpoints.items():
        _validate_shared_resolver_alias(
            entry,
            field=f"training.reference_matrix.initial_checkpoints.{name}",
            resolver_key="checkpoint",
        )
        _require_keys(
            entry,
            field=f"training.reference_matrix.initial_checkpoints.{name}",
            keys=("bytes", "sha256"),
        )
        _validate_declared_identity(
            entry,
            field=f"training.reference_matrix.initial_checkpoints.{name}",
            size_key="bytes",
        )
    common = _mapping(matrix["common"], field="training.reference_matrix.common")
    _require_keys(
        common,
        field="training.reference_matrix.common",
        keys=(
            "optimizer_steps",
            "fixed_checkpoint_step",
            "loader_validation_monitor_steps",
            "batch_size",
            "original_cube_samples_per_batch",
            "gripper_carry_samples_per_batch",
            "complete_gripper_carry_pairs_per_batch",
            "data_loader_workers",
            "loader_validation_batch_size",
            "learning_rate",
            "weight_decay",
            "gradient_clip_norm",
            "mixed_precision",
            "checkpoint_selection",
        ),
    )
    optimizer_steps = int(common["optimizer_steps"])
    monitors = tuple(int(value) for value in common["loader_validation_monitor_steps"])
    if (
        optimizer_steps != 4096
        or int(common["fixed_checkpoint_step"]) != optimizer_steps
        or monitors != (512, 1024, 2048, 4096)
        or int(common["batch_size"]) != 128
        or int(common["original_cube_samples_per_batch"]) != 64
        or int(common["gripper_carry_samples_per_batch"]) != 64
        or int(common["complete_gripper_carry_pairs_per_batch"]) != 32
        or int(common["data_loader_workers"]) != 4
        or int(common["loader_validation_batch_size"]) != 64
        or float(common["learning_rate"]) != 0.00005
        or float(common["weight_decay"]) != 0.001
        or float(common["gradient_clip_norm"]) != 1.0
        or common["mixed_precision"] != "bfloat16"
        or common["checkpoint_selection"] != "fixed_final_optimizer_step"
        or common.get("development_used_for_model_selection") is not False
        or common.get("public_test_used_for_recipe_or_checkpoint_selection")
        is not False
        or common.get("scientific_cli_overrides_allowed") is not False
    ):
        raise ValueError("Cube reference training has an unexpected fixed recipe")
    models = _mapping(matrix["models"], field="training.reference_matrix.models")
    expected_models = {
        "lewm": {
            "model_family": "LeWM",
            "variant": "mixed_frozen_image_paired_future_fit_1p00",
            "image_encoder_and_projector_frozen": True,
            "hidden_labels_used": False,
            "paired_training_rows_required": True,
        },
        "pldm": {
            "model_family": "PLDM",
            "variant": "mixed_pldm_joint",
            "hidden_labels_used": False,
            "paired_training_rows_required": True,
        },
    }
    if models != expected_models:
        raise ValueError("Cube reference-training model variants drifted")
    execution = _mapping(
        matrix["execution_policy"],
        field="training.reference_matrix.execution_policy",
    )
    if (
        execution.get("all_six_fixed_jobs_authorized") is not True
        or execution.get("adaptive_stopping") is not False
        or execution.get("families_decided_independently") is not True
        or execution.get("recipe_or_threshold_changes_after_any_result") is not False
    ):
        raise ValueError("Cube fixed-matrix execution policy drifted")

    evaluation = _mapping(payload.get("evaluation"), field="evaluation")
    if (
        evaluation.get("split") != "loader_validation"
        or evaluation.get("lance_table") != "loader_validation.lance"
        or int(evaluation.get("pair_count", -1)) != 256
        or int(evaluation.get("inference_batch_size", -1)) != 64
        or evaluation.get("public_test_used") is not False
    ):
        raise ValueError("Cube reference training must score only frozen Development")
    normalization = _mapping(
        evaluation.get("action_normalization"),
        field="evaluation.action_normalization",
    )
    mean = normalization.get("mean")
    std = normalization.get("std_population")
    if (
        normalization.get("source")
        != "original_cube_h5_finite_actions_population_zscore"
        or int(normalization.get("finite_action_rows", -1)) != 2_000_000
        or int(normalization.get("excluded_nonfinite_rows", -1)) != 10_000
        or not isinstance(mean, list)
        or not isinstance(std, list)
        or len(mean) != CUBE_RAW_ACTION_DIM
        or len(std) != CUBE_RAW_ACTION_DIM
        or any(not math.isfinite(float(value)) for value in mean)
        or any(
            not math.isfinite(float(value)) or float(value) <= 0
            for value in std
        )
    ):
        raise ValueError("Cube reference-training normalization must have five axes")

    scoring = _mapping(payload.get("scoring"), field="scoring")
    gates = _mapping(
        _mapping(
            scoring.get("hidden_future_prediction"),
            field="scoring.hidden_future_prediction",
        ).get("gates"),
        field="scoring.hidden_future_prediction.gates",
    )
    _require_keys(
        gates,
        field="scoring.hidden_future_prediction.gates",
        keys=(
            "correct_future_rate_minimum",
            "correct_history_rate_minimum",
            "context_switch_rate_minimum",
            "worst_rule_correct_future_rate_minimum",
            "target_latent_separation_required",
            "response_gain_minimum",
            "normalized_response_error_strict_maximum",
        ),
    )
    expected_gates = {
        "correct_future_rate_minimum": 0.75,
        "correct_history_rate_minimum": 0.75,
        "context_switch_rate_minimum": 0.90,
        "worst_rule_correct_future_rate_minimum": 0.70,
        "target_latent_separation_required": True,
        "response_gain_minimum": 0.50,
        "normalized_response_error_strict_maximum": 1.00,
    }
    if gates != expected_gates:
        raise ValueError("Cube Development prediction gates drifted")
    uncertainty = _mapping(
        scoring["hidden_future_prediction"].get("uncertainty"),
        field="scoring.hidden_future_prediction.uncertainty",
    )
    if uncertainty != {
        "method": "paired_query_bootstrap",
        "unit": "rule_matched_query_pair",
        "resamples": 10000,
        "confidence_level": 0.95,
        "random_seed": 2026080314,
        "lower_bound_minimum": {
            "correct_future_rate": 0.70,
            "correct_history_rate": 0.70,
            "context_switch_rate": 0.85,
        },
    }:
        raise ValueError("Cube Development uncertainty contract drifted")
    method = _mapping(scoring.get("method_level"), field="scoring.method_level")
    if (
        int(method.get("training_seeds_required", -1)) != 3
        or method.get("all_three_checkpoints_must_pass_per_family") is not True
        or method.get("at_least_one_family_required_for_release_progression")
        is not True
    ):
        raise ValueError("Cube method-level Development decision drifted")

    planned = _mapping(payload.get("planned_artifacts"), field="planned_artifacts")
    _require_keys(
        planned,
        field="planned_artifacts",
        keys=(
            "freeze_receipt",
            "training_root",
            "development_score_root",
            "development_decision",
        ),
    )
    if planned != {
        "freeze_receipt": (
            "artifacts/evaluation/history3/"
            "cube_gripper_carry_h3_development_v4r1/"
            "reference_training_freeze_receipt_v3.json"
        ),
        "training_root": (
            "artifacts/evaluation/history3/"
            "cube_gripper_carry_h3_development_v4r1/reference_training_v3"
        ),
        "development_score_root": (
            "artifacts/evaluation/history3/"
            "cube_gripper_carry_h3_development_v4r1/"
            "reference_development_score_v3"
        ),
        "development_decision": (
            "artifacts/evaluation/history3/"
            "cube_gripper_carry_h3_development_v4r1/"
            "reference_development_decision_v3.json"
        ),
    }:
        raise ValueError("Cube reference-training artifact namespace drifted")
    if payload.get("prohibited_claims") != [
        "public_test_score_or_generalization",
        "original_task_retention_passed",
        "release_candidate_or_public_release",
        "suite_membership",
        "both_reference_models_passed_before_complete_matrix",
    ]:
        raise ValueError("Cube reference-training prohibited-claim boundary drifted")


def cube_reference_infrastructure_recovery_identity(
    prereg: Mapping[str, Any], *, repo_root: Path | None = None
) -> dict[str, Any]:
    """Verify the immutable zero-step v2 failure that authorizes v3."""

    root = (repo_root or repository_root()).resolve()
    recovery = prereg["infrastructure_recovery"]
    prior = recovery["prior_attempt"]
    observed: dict[str, Any] = {}
    for name, entry in prior.items():
        path = resolve_contextworld_path(str(entry["path"]), repo_root=root)
        identity = _stable_file_identity(path)
        if (
            identity["sha256"] != str(entry["sha256"])
            or identity["size_bytes"] != int(entry["size_bytes"])
        ):
            raise RuntimeError(f"Cube v2 recovery evidence drifted: {name}")
        observed[name] = {
            "path": str(entry["path"]),
            "sha256": identity["sha256"],
            "size_bytes": identity["size_bytes"],
        }

    failure_path = resolve_contextworld_path(
        str(prior["failure_receipt"]["path"]), repo_root=root
    )
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    expected_chain = {
        name: observed[name]
        for name in (
            "preregistration",
            "freeze_receipt",
            "matrix_request",
            "matrix_report",
        )
    }
    expected_jobs = [
        f"{model}_seed{seed}"
        for model in ("lewm", "pldm")
        for seed in (17321, 17322, 17323)
    ]
    jobs = _mapping(failure.get("jobs"), field="v2_failure_receipt.jobs")
    if (
        failure.get("schema_version") != 1
        or failure.get("receipt_id")
        != "cube_gripper_carry_h3_v4r1_reference_training_v2_failure"
        or failure.get("preregistration_id") != CUBE_REFERENCE_TRAINING_V2_ID
        or failure.get("status")
        != "infrastructure_failed_after_model_load_before_forward"
        or failure.get("classification")
        != recovery["failure_classification"]
        or failure.get("failure_stage") != recovery["failure_stage"]
        or failure.get("checks_passed") is not True
        or failure.get("authorization_chain") != expected_chain
        or failure.get("training_state") != recovery["prior_training_state"]
        or failure.get("retry") != recovery["prior_retry"]
        or failure.get("public_test") != recovery["public_test"]
        or failure.get("root_cause") != recovery["root_cause"]
        or set(jobs) != set(expected_jobs)
        or any(
            jobs[name].get("exit_code") != 1
            or jobs[name].get("log", {}).get("expected_error_present") is not True
            or jobs[name].get("artifacts", {}).get("action_input_dim") != 25
            or jobs[name].get("artifacts", {}).get(
                "initial_checkpoint_identity_matched"
            )
            is not True
            or jobs[name].get("artifacts", {}).get("train_pairs_materialized")
            != 2048
            or jobs[name].get("artifacts", {}).get(
                "loader_validation_pairs_materialized"
            )
            != 256
            for name in expected_jobs
        )
    ):
        raise RuntimeError("Cube v2 zero-step recovery receipt drifted")
    scientific = _mapping(
        failure.get("scientific_conclusion"),
        field="v2_failure_receipt.scientific_conclusion",
    )
    if scientific != {
        "data_failure_claim_allowed": False,
        "model_training_failure_claim_allowed": False,
        "development_score_claim_allowed": False,
    }:
        raise RuntimeError("Cube v2 infrastructure failure gained a scientific claim")
    return {
        "source_preregistration_id": CUBE_REFERENCE_TRAINING_V2_ID,
        "failure_classification": recovery["failure_classification"],
        "recovery_change": recovery["recovery_change"],
        "evidence": observed,
        "prior_training_state": recovery["prior_training_state"],
        "prior_retry": recovery["prior_retry"],
        "public_test": recovery["public_test"],
    }


def _resolve_path_specification(
    specification: Mapping[str, Any], *, repo_root: Path, resolver_key: str
) -> Path:
    environment = str(specification.get("environment_variable", ""))
    configured = os.environ.get(environment) if environment else None
    if configured:
        return Path(os.path.abspath(Path(configured).expanduser()))
    bundled = specification.get("bundled_artifact_path")
    if bundled:
        candidate = artifact_root(repo_root) / str(bundled)
        if candidate.exists():
            return Path(os.path.abspath(candidate))
    source = specification.get(resolver_key)
    if not source:
        raise ValueError(
            "Cube reference-training input has no shared-resolver "
            f"{resolver_key!r} source"
        )
    return resolve_contextworld_path(str(source), repo_root=repo_root)


def resolve_cube_reference_training_input(
    prereg: Mapping[str, Any], name: str, *, repo_root: Path | None = None
) -> Path:
    if name not in {"original_h5", "original_lance"}:
        raise ValueError(f"Unsupported Cube training input {name!r}")
    root = (repo_root or repository_root()).resolve()
    return _resolve_path_specification(
        prereg["training"]["upstream"][name],
        repo_root=root,
        resolver_key="path",
    )


def resolve_cube_reference_initial_checkpoint(
    prereg: Mapping[str, Any], family: str, *, repo_root: Path | None = None
) -> Path:
    if family not in {"lewm", "pldm"}:
        raise ValueError("Cube checkpoint family must be lewm or pldm")
    root = (repo_root or repository_root()).resolve()
    return _resolve_path_specification(
        prereg["training"]["reference_matrix"]["initial_checkpoints"][family],
        repo_root=root,
        resolver_key="checkpoint",
    )


def expected_cube_reference_training_cell(
    prereg: Mapping[str, Any],
    *,
    model_family: str,
    training_seed: int,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    if model_family not in {"lewm", "pldm"}:
        raise ValueError("Cube training model family must be lewm or pldm")
    matrix = prereg["training"]["reference_matrix"]
    seed = int(training_seed)
    if seed not in tuple(int(value) for value in matrix["training_seeds"]):
        raise ValueError("Cube training seed is outside the frozen matrix")
    if "_freeze_receipt" in prereg:
        authorized = prereg["_freeze_receipt"]["authorization"]["jobs"]
        if {"model": model_family, "seed": seed} not in authorized:
            raise RuntimeError("Cube training cell is absent from freeze authorization")
    root = (repo_root or repository_root()).resolve()
    training_root = resolve_contextworld_path(
        str(prereg["planned_artifacts"]["training_root"]), repo_root=root
    )
    variant = str(matrix["models"][model_family]["variant"])
    optimizer_steps = int(matrix["common"]["optimizer_steps"])
    job_root = training_root / f"{model_family}_seed{seed}"
    return {
        "model_family": model_family,
        "training_seed": seed,
        "variant": variant,
        "optimizer_steps": optimizer_steps,
        "training_root": training_root,
        "job_root": job_root,
        "report": job_root / "training_report.json",
        "checkpoint": job_root / f"{variant}_step{optimizer_steps}.pt",
    }


def validate_cube_reference_training_report(
    prereg: Mapping[str, Any],
    *,
    model_family: str,
    training_seed: int,
    prereg_path: Path | None = None,
    report_path: Path | None = None,
    checkpoint_path: Path | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Validate one exact frozen training cell and its checkpoint identity."""

    root = (repo_root or repository_root()).resolve()
    cell = expected_cube_reference_training_cell(
        prereg,
        model_family=model_family,
        training_seed=training_seed,
        repo_root=root,
    )
    expected_report = Path(cell["report"])
    expected_checkpoint = Path(cell["checkpoint"])
    report_file = (report_path or expected_report).expanduser().resolve()
    checkpoint_file = (checkpoint_path or expected_checkpoint).expanduser().resolve()
    if report_file != expected_report or checkpoint_file != expected_checkpoint:
        raise RuntimeError("Cube training cell path drifted from preregistration")
    if not report_file.is_file() or report_file.is_symlink():
        raise FileNotFoundError(f"Missing Cube training report: {report_file}")
    checkpoint_identity = _stable_file_identity(checkpoint_file)
    report = json.loads(report_file.read_text(encoding="utf-8"))
    matrix = prereg["training"]["reference_matrix"]
    common = matrix["common"]
    model = str(cell["model_family"])
    seed = int(cell["training_seed"])
    variant = str(cell["variant"])
    optimizer_steps = int(cell["optimizer_steps"])
    config_path = (
        prereg_path
        or Path(str(prereg.get("_config_path", DEFAULT_CUBE_REFERENCE_TRAINING_PREREG)))
    ).expanduser().resolve()
    provenance = _mapping(report.get("provenance"), field="training_report.provenance")
    result = _mapping(report.get("result"), field="training_report.result")
    final = _mapping(
        result.get("final_checkpoint"), field="training_report.result.final_checkpoint"
    )
    batch = _mapping(result.get("batch"), field="training_report.result.batch")
    source_checkpoint = _mapping(
        result.get("source_checkpoint"),
        field="training_report.result.source_checkpoint",
    )
    optimizer = _mapping(
        result.get("optimizer"), field="training_report.result.optimizer"
    )
    expected_initial = matrix["initial_checkpoints"][model]
    expected_data_root = resolve_contextworld_path(
        str(prereg["data"]["artifact_tree"]["root"]), repo_root=root
    )
    expected_h5 = resolve_cube_reference_training_input(
        prereg, "original_h5", repo_root=root
    )
    expected_lance = resolve_cube_reference_training_input(
        prereg, "original_lance", repo_root=root
    )
    expected_initial_path = resolve_cube_reference_initial_checkpoint(
        prereg, model, repo_root=root
    )
    snapshots = result.get("snapshots")
    representation = _mapping(
        result.get("representation_freeze"),
        field="training_report.result.representation_freeze",
    )
    first_gradient = representation.get("first_step_gradient_audit")
    if first_gradient is not None and not isinstance(first_gradient, dict):
        raise RuntimeError("Cube training gradient audit must be a mapping")
    if (
        report.get("schema_version") != 1
        or report.get("status") != "completed"
        or int(report.get("fixed_checkpoint_step", -1)) != optimizer_steps
        or report.get("loader_validation_used_for_selection") is not False
        or report.get("independent_validation_used_for_selection") is not False
        or provenance.get("model") != model
        or int(provenance.get("seed", -1)) != seed
        or provenance.get("variant") != variant
        or int(provenance.get("optimizer_steps", -1)) != optimizer_steps
        or provenance.get("formal_reference_recipe") is not True
        or provenance.get("release", {}).get("release_id")
        != prereg["preregistration_id"]
        or provenance.get("release", {}).get("sha256") != file_sha256(config_path)
        or Path(str(provenance.get("data", {}).get("root", ""))).resolve()
        != expected_data_root
        or provenance.get("data", {}).get("manifest_sha256")
        != prereg["data"]["manifest_sha256"]
        or provenance.get("data", {}).get("release_manifest_sha256")
        != prereg["data"]["manifest_sha256"]
        or provenance.get("data", {}).get("data_root_override") is not False
        or provenance.get("data", {}).get("independent_validation_opened") is not False
        or int(provenance.get("data", {}).get("train_pairs", -1))
        != CUBE_TRAINING_PAIR_COUNTS["train"]
        or int(provenance.get("data", {}).get("loader_validation_pairs", -1))
        != CUBE_TRAINING_PAIR_COUNTS["loader_validation"]
        or Path(
            str(provenance.get("upstream", {}).get("original_h5", {}).get("path", ""))
        ).resolve()
        != expected_h5.resolve()
        or int(
            provenance.get("upstream", {})
            .get("original_h5", {})
            .get("bytes", -1)
        )
        != int(prereg["training"]["upstream"]["original_h5"]["expected_identity"]["size_bytes"])
        or Path(str(provenance.get("upstream", {}).get("original_lance", ""))).resolve()
        != expected_lance.resolve()
        or provenance.get("upstream", {})
        .get("initial_checkpoint", {})
        .get("sha256")
        != expected_initial["sha256"]
        or result.get("variant") != variant
        or int(result.get("seed", -1)) != seed
        or int(result.get("optimizer_steps", -1)) != optimizer_steps
        or int(batch.get("total", -1)) != int(common["batch_size"])
        or int(batch.get("original", -1))
        != int(common["original_cube_samples_per_batch"])
        or int(batch.get("hidden", -1))
        != int(common["gripper_carry_samples_per_batch"])
        or int(batch.get("hidden_pairs", -1))
        != int(common["complete_gripper_carry_pairs_per_batch"])
        or batch.get("ordering") != "original_then_adjacent_hidden_pairs"
        or result.get("hidden_labels_at_model_or_loss_boundary") is not False
        or Path(str(source_checkpoint.get("path", ""))).resolve()
        != expected_initial_path.resolve()
        or source_checkpoint.get("sha256") != expected_initial["sha256"]
        or source_checkpoint.get("loaded_model_config") != model
        or source_checkpoint.get("strict_state_dict_load") is not True
        or float(optimizer.get("learning_rate", -1.0)) != float(common["learning_rate"])
        or float(optimizer.get("weight_decay", -1.0)) != float(common["weight_decay"])
        or float(optimizer.get("gradient_clip_norm", -1.0))
        != float(common["gradient_clip_norm"])
        or result.get("precision") != "bf16_mixed_autocast"
        or not isinstance(snapshots, list)
        or not snapshots
        or int(snapshots[-1].get("optimizer_step", -1)) != optimizer_steps
        or Path(str(final.get("path", ""))).resolve() != expected_checkpoint
        or final.get("sha256") != checkpoint_identity["sha256"]
        or not isinstance(final.get("model_state_sha256"), str)
        or len(final["model_state_sha256"]) != 64
    ):
        raise RuntimeError(f"Frozen Cube training report drifted: {model}/seed{seed}")
    if model == "lewm":
        if (
            result.get("regularizer") != "paired_future_fit"
            or result.get("conditional_population") != "paired_future_fit"
            or representation.get("enabled") is not True
            or representation.get("optimizer_excludes_frozen_parameters") is not True
            or representation.get("frozen_state_unchanged") is not True
            or representation.get("trainable_state_changed") is not True
            or not isinstance(first_gradient, dict)
            or first_gradient.get("frozen_parameters_have_no_gradient") is not True
            or first_gradient.get("trainable_parameters_have_nonzero_gradient")
            is not True
        ):
            raise RuntimeError("Frozen Cube LeWM representation audit drifted")
    elif (
        result.get("regularizer") != "pldm"
        or representation.get("enabled") is not False
        or result.get("pldm_contract") is None
    ):
        raise RuntimeError("Frozen Cube PLDM training contract drifted")
    return {
        **cell,
        "report_payload": report,
        "checkpoint_sha256": checkpoint_identity["sha256"],
        "checkpoint_size_bytes": checkpoint_identity["size_bytes"],
        "model_state_sha256": final["model_state_sha256"],
    }


def _resolve_declared_path(value: str, *, repo_root: Path) -> Path:
    return resolve_contextworld_path(value, repo_root=repo_root)


def _validate_freeze_receipt(
    prereg: dict[str, Any], *, prereg_path: Path, repo_root: Path
) -> dict[str, Any]:
    receipt_path = _resolve_declared_path(
        str(prereg["planned_artifacts"]["freeze_receipt"]),
        repo_root=repo_root,
    )
    if not receipt_path.is_file():
        raise FileNotFoundError(
            "Cube reference training is not authorized: missing freeze receipt "
            f"{receipt_path}"
        )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        receipt.get("schema_version") != 1
        or receipt.get("preregistration_id") != CUBE_REFERENCE_TRAINING_ID
        or receipt.get("protocol_id") != CUBE_REFERENCE_TRAINING_PROTOCOL
        or receipt.get("status") != FREEZE_STATUS
        or receipt.get("checks_passed") is not True
        or receipt.get("training_and_development_scoring_authorized") is not True
    ):
        raise RuntimeError("Invalid Cube reference-training freeze receipt")
    frozen_prereg = _mapping(
        receipt.get("preregistration"), field="freeze_receipt.preregistration"
    )
    if (
        _resolve_declared_path(
            str(frozen_prereg.get("path", "")), repo_root=repo_root
        )
        != prereg_path
        or frozen_prereg.get("sha256") != file_sha256(prereg_path)
        or int(frozen_prereg.get("size_bytes", -1)) != prereg_path.stat().st_size
    ):
        raise RuntimeError("Cube reference-training preregistration drifted after freeze")
    _validate_closed_public(
        receipt.get("public_test"), field="freeze_receipt.public_test"
    )
    observed_recovery = cube_reference_infrastructure_recovery_identity(
        prereg, repo_root=repo_root
    )
    if receipt.get("infrastructure_recovery") != observed_recovery:
        raise RuntimeError("Cube v3 infrastructure-recovery freeze receipt drifted")
    authorization = _mapping(
        receipt.get("authorization"), field="freeze_receipt.authorization"
    )
    matrix = prereg["training"]["reference_matrix"]
    expected_jobs = [
        {"model": model, "seed": int(seed)}
        for model in ("lewm", "pldm")
        for seed in matrix["training_seeds"]
    ]
    if (
        authorization.get("jobs") != expected_jobs
        or int(authorization.get("optimizer_steps_per_job", -1))
        != int(matrix["common"]["optimizer_steps"])
        or int(authorization.get("total_optimizer_steps_authorized", -1))
        != len(expected_jobs) * int(matrix["common"]["optimizer_steps"])
        or authorization.get("authorized_splits") != list(AUTHORIZED_SPLITS)
        or authorization.get("development_scoring_authorized") is not True
        or authorization.get("public_model_scoring_authorized") is not False
        or authorization.get("original_task_retention_authorized") is not False
        or authorization.get("recipe_or_threshold_changes_authorized") is not False
    ):
        raise RuntimeError("Cube reference-training authorization receipt drifted")

    observed_identity_receipts: dict[str, Any] = {}
    for name, entry in prereg["identity"].items():
        path = _resolve_declared_path(str(entry["path"]), repo_root=repo_root)
        if not path.is_file():
            raise FileNotFoundError(f"Cube reference-training identity missing: {path}")
        if (
            path.stat().st_size != int(entry["size_bytes"])
            or file_sha256(path) != str(entry["sha256"])
        ):
            raise RuntimeError(f"Cube reference-training identity drift: {name}")
        observed_identity_receipts[name] = {
            "path": str(entry["path"]),
            "sha256": str(entry["sha256"]),
            "size_bytes": int(entry["size_bytes"]),
        }
    if receipt.get("identity") != observed_identity_receipts:
        raise RuntimeError("Cube reference-training identity freeze receipt drifted")

    observed_evidence_receipts: dict[str, Any] = {}
    for name, entry in prereg["data"]["artifacts"].items():
        path = _resolve_declared_path(str(entry["path"]), repo_root=repo_root)
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != int(entry["size_bytes"])
            or file_sha256(path) != str(entry["sha256"])
        ):
            raise RuntimeError(f"Cube v4r1 data evidence drifted: {name}")
        observed_evidence_receipts[name] = {
            "path": str(entry["path"]),
            "sha256": str(entry["sha256"]),
            "size_bytes": int(entry["size_bytes"]),
        }
    if receipt.get("data_evidence") != observed_evidence_receipts:
        raise RuntimeError("Cube v4r1 data-evidence freeze receipt drifted")
    observed_data_tree = cube_reference_data_tree_identity(
        prereg, repo_root=repo_root
    )
    if receipt.get("data_tree") != observed_data_tree:
        raise RuntimeError("Cube v4r1 generated-data tree drifted after freeze")

    runtime_spec = prereg["runtime"]["stable_worldmodel"]
    stable_repo = Path(str(runtime_spec["repo"])).expanduser()
    if not stable_repo.is_absolute():
        stable_repo = (repo_root / stable_repo).resolve()
    if not stable_repo.is_dir() or stable_repo.is_symlink():
        raise FileNotFoundError("Cube Stable-WorldModel runtime directory drifted")
    git_environment = os.environ.copy()
    git_environment["SUDO_UID"] = str(stable_repo.stat().st_uid)
    commit = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={stable_repo}",
            "-C",
            str(stable_repo),
            "rev-parse",
            "HEAD",
        ],
        check=True,
        text=True,
        capture_output=True,
        env=git_environment,
    ).stdout.strip()
    if commit != str(runtime_spec["expected_ref"]):
        raise RuntimeError("Cube Stable-WorldModel runtime commit drifted")
    if subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={stable_repo}",
            "-C",
            str(stable_repo),
            "status",
            "--porcelain",
        ],
        check=True,
        text=True,
        capture_output=True,
        env=git_environment,
    ).stdout:
        raise RuntimeError("Cube Stable-WorldModel runtime became dirty")
    for name, entry in runtime_spec["required_files"].items():
        path = stable_repo / str(entry["path"])
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != int(entry["size_bytes"])
            or file_sha256(path) != str(entry["sha256"])
        ):
            raise RuntimeError(f"Cube Stable-WorldModel runtime drift: {name}")

    receipt_runtime = _mapping(
        receipt.get("runtime"), field="freeze_receipt.runtime"
    )
    expected_runtime_receipt = {
        "path": str(stable_repo),
        "commit": commit,
        "clean_worktree": True,
        "required_files": {
            name: {
                "path": str(entry["path"]),
                "sha256": str(entry["sha256"]),
                "size_bytes": int(entry["size_bytes"]),
            }
            for name, entry in runtime_spec["required_files"].items()
        },
    }
    if receipt_runtime.get("stable_worldmodel") != expected_runtime_receipt:
        raise RuntimeError("Cube Stable-WorldModel freeze receipt drifted")
    compatibility = _mapping(
        receipt_runtime.get("checkpoint_compatibility"),
        field="freeze_receipt.runtime.checkpoint_compatibility",
    )
    model_compatibility = _mapping(
        compatibility.get("models"),
        field="freeze_receipt.runtime.checkpoint_compatibility.models",
    )
    if set(model_compatibility) != {"lewm", "pldm"}:
        raise RuntimeError("Cube checkpoint/runtime compatibility set drifted")
    for family, row in model_compatibility.items():
        if (
            row.get("strict_state_dict_load") is not True
            or row.get("loaded_model_config") != family
            or int(row.get("action_input_dim", -1)) != CUBE_ACTION_INPUT_DIM
            or not isinstance(row.get("model_state_sha256"), str)
            or len(row["model_state_sha256"]) != 64
            or int(row.get("parameter_count", 0)) <= 0
            or row.get("synthetic_cpu_forward_preflight") is not True
        ):
            raise RuntimeError(
                f"Cube {family} checkpoint/runtime compatibility drifted"
            )
    if compatibility.get("pinned_loss_compatibility") != {
        "conditional_sigreg_constructor_adapter_installed": True,
        "conditional_sigreg_missing_keywords": [
            "include_unpaired",
            "complete_haar_population",
        ],
        "conditional_sigreg_false_only": True,
        "unavailable_eager_diagnostic_sentinels": [
            "DynamicsResponseSIGReg",
            "GroupBalancedSIGReg",
            "ScaleCalibratedConditionalSIGReg",
        ],
        "shared_engine_constructor_preflight": True,
        "constructed_class": "PinnedConditionalSIGReg",
        "include_unpaired": False,
        "complete_haar_population": False,
    }:
        raise RuntimeError("Cube pinned-loss compatibility receipt drifted")

    inputs = _mapping(receipt.get("inputs"), field="freeze_receipt.inputs")
    source_h5 = resolve_cube_reference_training_input(
        prereg, "original_h5", repo_root=repo_root
    )
    source_receipt = _mapping(
        inputs.get("original_h5"), field="freeze_receipt.inputs.original_h5"
    )
    source_identity = _stable_file_identity(source_h5)
    source_expected = prereg["training"]["upstream"]["original_h5"][
        "expected_identity"
    ]
    if (
        source_receipt.get("symbol")
        != prereg["training"]["upstream"]["original_h5"]["source_symbol"]
        or source_receipt.get("path_recorded") is not False
        or source_identity["size_bytes"] != int(source_receipt["size_bytes"])
        or source_identity["sha256"] != str(source_receipt["sha256"])
        or source_identity["size_bytes"] != int(source_expected["size_bytes"])
        or source_identity["sha256"] != str(source_expected["sha256"])
        or int(source_receipt.get("row_count", -1)) != int(source_expected["row_count"])
        or int(source_receipt.get("action_dim", -1)) != CUBE_RAW_ACTION_DIM
        or int(source_receipt.get("episode_count", -1))
        != int(source_expected["episode_count"])
    ):
        raise RuntimeError("Frozen Cube H5 identity drifted")
    original_lance = resolve_cube_reference_training_input(
        prereg, "original_lance", repo_root=repo_root
    )
    lance_receipt = _mapping(
        inputs.get("original_lance"),
        field="freeze_receipt.inputs.original_lance",
    )
    if not original_lance.is_dir() or original_lance.is_symlink():
        raise RuntimeError("Frozen original Cube Lance directory drifted")
    observed_lance_files = {
        child.relative_to(original_lance).as_posix(): child
        for child in original_lance.rglob("*")
        if child.is_file()
    }
    if set(observed_lance_files) != set(lance_receipt["files"]):
        raise RuntimeError("Frozen original Cube Lance file set drifted")
    expected_lance = prereg["training"]["upstream"]["original_lance"]
    expected_lance_identity = expected_lance["expected_identity"]
    if (
        lance_receipt.get("symbol") != expected_lance["source_symbol"]
        or lance_receipt.get("path_recorded") is not False
        or int(lance_receipt.get("row_count", -1))
        != int(expected_lance_identity["row_count"])
        or int(lance_receipt.get("action_dim", -1)) != CUBE_RAW_ACTION_DIM
        or int(lance_receipt.get("file_count", -1))
        != int(expected_lance_identity["file_count"])
        or int(lance_receipt.get("bytes", -1))
        != int(expected_lance_identity["bytes"])
    ):
        raise RuntimeError("Frozen original Cube Lance receipt drifted")
    for relative, entry in lance_receipt["files"].items():
        path = observed_lance_files[relative]
        identity = _stable_file_identity(path)
        expected = expected_lance_identity["files"][relative]
        if (
            identity["size_bytes"] != int(entry["size_bytes"])
            or identity["sha256"] != str(entry["sha256"])
            or identity["size_bytes"] != int(expected["size_bytes"])
            or identity["sha256"] != str(expected["sha256"])
        ):
            raise RuntimeError(f"Frozen original Cube Lance identity drift: {relative}")
    checkpoint_receipts = _mapping(
        inputs.get("initial_checkpoints"),
        field="freeze_receipt.inputs.initial_checkpoints",
    )
    for family in ("lewm", "pldm"):
        checkpoint = resolve_cube_reference_initial_checkpoint(
            prereg, family, repo_root=repo_root
        )
        expected = checkpoint_receipts[family]
        specification = prereg["training"]["reference_matrix"][
            "initial_checkpoints"
        ][family]
        if (
            not checkpoint.is_file()
            or checkpoint.stat().st_size != int(expected["size_bytes"])
            or file_sha256(checkpoint) != str(expected["sha256"])
            or expected.get("source_symbol") != specification["source_symbol"]
            or expected.get("path_recorded") is not False
            or int(expected["size_bytes"]) != int(specification["bytes"])
            or str(expected["sha256"]) != str(specification["sha256"])
        ):
            raise RuntimeError(f"Frozen Cube {family} checkpoint drifted")

    return {**prereg, "_freeze_receipt": receipt, "_freeze_receipt_path": str(receipt_path)}


def load_cube_reference_training_prereg(
    path: Path | str = DEFAULT_CUBE_REFERENCE_TRAINING_PREREG,
    *,
    require_freeze: bool = True,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Cube reference-training preregistration must be a mapping")
    _validate_preregistration(payload)
    result = {
        **payload,
        "release_id": payload["preregistration_id"],
        "release_status": "data_ready_training_in_progress",
        "_config_path": str(config_path),
    }
    if not require_freeze:
        return result
    return _validate_freeze_receipt(
        result,
        prereg_path=config_path,
        repo_root=(repo_root or repository_root()).resolve(),
    )


__all__ = [
    "AUTHORIZED_SPLITS",
    "CUBE_ACTION_BLOCK_STEPS",
    "CUBE_ACTION_INPUT_DIM",
    "CUBE_RAW_ACTION_DIM",
    "CUBE_REFERENCE_TRAINING_ID",
    "CUBE_REFERENCE_TRAINING_PROTOCOL",
    "CUBE_REFERENCE_TRAINING_V2_ID",
    "CUBE_REFERENCE_TRAINING_V2_PROTOCOL",
    "CUBE_REFERENCE_TRAINING_V2_RECOVERY_EVIDENCE",
    "CUBE_TRAINING_PAIR_COUNTS",
    "DEFAULT_CUBE_REFERENCE_TRAINING_PREREG",
    "FREEZE_STATUS",
    "PREREG_STATUS",
    "file_sha256",
    "cube_reference_infrastructure_recovery_identity",
    "load_cube_reference_training_prereg",
    "resolve_cube_reference_initial_checkpoint",
    "resolve_cube_reference_training_input",
]
