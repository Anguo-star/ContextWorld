from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import stat
from typing import Any, Mapping

import yaml

from contextworld.paths import artifact_root, repository_root, resolve_contextworld_path


PREREGISTRATION_ID = "contextworld_cube_gripper_carry_h3_v4r1_public_release_v1"
PROTOCOL_ID = "cube_gripper_carry_rule_history3_v4r1_public_release_v1"
PREREGISTRATION_STATUS = "preregistered_before_public_generation_or_access"
FREEZE_STATUS = "frozen_before_public_generation_or_access"
FREEZE_RECEIPT_ID = (
    "contextworld_cube_gripper_carry_h3_v4r1_public_release_freeze_v1"
)
PUBLIC_SPLIT = "validation"
PUBLIC_PAIR_COUNT = 256
PUBLIC_CATALOG_INDEX_OFFSET = 3_000_000
PUBLIC_CANDIDATE_ASSIGNMENT_SEED = 2026081400
PUBLIC_CATALOG_SEED = 2026081401
PUBLIC_PROFILE_SEED = 2026081402
EXPECTED_TRAINING_SEEDS = (17321, 17322, 17323)
EXPECTED_TRAINING_RECIPE = "mixed_frozen_image_paired_future_fit_1p00"
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
EXPECTED_DATA_ACCESS_CONTRACT = {
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
}
EXPECTED_AUTHORIZATION_BASIS_KEYS = frozenset(
    {
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
)
EXPECTED_IMPLEMENTATION_KEYS = frozenset(
    {
        "public_contract",
        "public_score",
        "public_builder",
        "public_matrix_runner",
        "public_freezer",
        "public_finalizer",
        "public_protocol",
        "public_contract_tests",
        "public_builder_tests",
        "public_score_tests",
        "public_freeze_tests",
        "public_finalizer_tests",
        "base_v2_physics",
        "v3_physics_dependency",
        "v4_physics",
        "common_causal_contract",
        "v4_builder",
        "prior_exclusion_finalizer",
        "cube_data_api",
        "cube_score_api",
        "adapters",
        "paired_latent_response",
        "model_protocol",
        "model_identity",
        "stable_worldmodel_loader",
        "path_resolution",
        "public_api",
        "package",
    }
)
EXPECTED_CONTENT_EXCLUSION_FIELDS = frozenset(
    {
        "action_profile_ids",
        "scene_template_content_hashes",
        "pair_content_hashes",
        "query_pixel_hashes",
    }
)
EXPECTED_RUNTIME_FILE_KEYS = frozenset(
    {
        "lewm_model_config",
        "data_api",
        "dataset_api",
        "lance_loader",
        "lewm_model",
        "lewm_modules",
        "loss_api",
    }
)

DEFAULT_PREREGISTRATION = repository_root() / (
    "configs/benchmark/"
    "cube_gripper_carry_h3_v4r1_public_release_prereg_v1.yaml"
)
DEFAULT_FREEZE_RECEIPT = resolve_contextworld_path(
    "artifacts/evaluation/history3/"
    "cube_gripper_carry_h3_public_release_v1/"
    "public_release_freeze_receipt_v1.json"
)


@dataclass(frozen=True)
class PublicAuthorization:
    preregistration_path: Path
    freeze_receipt_path: Path
    preregistration: dict[str, Any]
    freeze_receipt: dict[str, Any]
    freeze_receipt_identity: dict[str, Any]

    @property
    def public_root(self) -> Path:
        return resolve_contextworld_path(
            self.preregistration["planned_artifacts"]["public_data_root"]
        )

    @property
    def score_root(self) -> Path:
        return resolve_contextworld_path(
            self.preregistration["planned_artifacts"]["public_score_root"]
        )

    @property
    def decision_path(self) -> Path:
        return resolve_contextworld_path(
            self.preregistration["planned_artifacts"]["public_release_decision"]
        )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _absolute_without_following_leaf(path: Path) -> Path:
    value = Path(path).expanduser()
    if not value.is_absolute():
        value = Path.cwd() / value
    return value.absolute()


def file_identity(path: Path, *, logical_path: str | None = None) -> dict[str, Any]:
    value = _absolute_without_following_leaf(path)
    metadata = os.lstat(value)
    if not stat.S_ISREG(metadata.st_mode) or value.is_symlink():
        raise ValueError(f"identity input must be a regular non-symlink file: {value}")
    return {
        "path": logical_path or str(value),
        "sha256": file_sha256(value),
        "size_bytes": int(metadata.st_size),
    }


def _read_bytes_nofollow(path: Path, *, label: str) -> bytes:
    value = _absolute_without_following_leaf(path)
    metadata = os.lstat(value)
    if not stat.S_ISREG(metadata.st_mode) or value.is_symlink():
        raise ValueError(f"{label} must be a regular non-symlink file: {value}")
    descriptor = os.open(value, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(descriptor, "rb") as stream:
        return stream.read()


def read_json_nofollow(path: Path, *, label: str) -> tuple[bytes, dict[str, Any]]:
    raw = _read_bytes_nofollow(path, label=label)
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return raw, value


def read_yaml_nofollow(path: Path, *, label: str) -> tuple[bytes, dict[str, Any]]:
    raw = _read_bytes_nofollow(path, label=label)
    value = yaml.safe_load(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a YAML mapping")
    return raw, value


def _identity_matches(
    observed: Mapping[str, Any], expected: Mapping[str, Any], *, label: str
) -> None:
    fields = ("path", "sha256", "size_bytes")
    if {name: observed.get(name) for name in fields} != {
        name: expected.get(name) for name in fields
    }:
        raise RuntimeError(f"{label} identity mismatch")


def _closed_public_state(value: Any, *, label: str) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    if value.get("access_status") != "closed_not_read_not_scored":
        raise RuntimeError(f"{label} was not closed before freeze")
    for name in ("generated", "opened", "read", "hashed", "scored"):
        if value.get(name) is not False:
            raise RuntimeError(f"{label}.{name} must be false before freeze")
    if value.get("validation_lance_access_allowed") is not False:
        raise RuntimeError(
            f"{label}.validation_lance_access_allowed must be false before freeze"
        )


def validate_public_preregistration_contract(prereg: Mapping[str, Any]) -> None:
    if prereg.get("phase") != "public_generation_and_evaluation_only":
        raise RuntimeError("Cube Public preregistration phase drifted")
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
        "grasp_modes": ["cannot_hold", "can_hold"],
        "public_test_included": True,
        "sealed_test_included": False,
    }
    if any(scope.get(name) != value for name, value in expected_scope.items()):
        raise RuntimeError("Cube Public scope/action contract drifted")

    generation = _mapping(
        prereg.get("public_data_generation"), label="public_data_generation"
    )
    expected_generation = {
        "split": PUBLIC_SPLIT,
        "public_split_name": "Public Test",
        "pair_count": PUBLIC_PAIR_COUNT,
        "candidate_pool_count": 2 * PUBLIC_PAIR_COUNT,
        "catalog_index_offset": PUBLIC_CATALOG_INDEX_OFFSET,
        "candidate_assignment_seed": PUBLIC_CANDIDATE_ASSIGNMENT_SEED,
        "catalog_seed": PUBLIC_CATALOG_SEED,
        "profile_seed": PUBLIC_PROFILE_SEED,
        "action_templates": ["endpoint4", "front_hold", "plateau", "ramp4"],
        "pair_balanced": True,
        "split_disjoint_from_all_non_public_content": True,
        "action_profile_sum_zero": True,
        "action_profile_last_zero": True,
        "workers": 16,
        "jpeg_quality": 95,
        "staging_root": "/tmp",
    }
    if any(
        generation.get(name) != value for name, value in expected_generation.items()
    ):
        raise RuntimeError("Cube Public generation recipe drifted")

    evaluation = _mapping(prereg.get("public_evaluation"), label="public_evaluation")
    if evaluation.get("data_access_contract") != EXPECTED_DATA_ACCESS_CONTRACT:
        raise RuntimeError("Cube Public model/evaluator data boundary drifted")
    if (
        evaluation.get("authorized_model_families") != ["lewm"]
        or evaluation.get("excluded_model_families")
        != {"pldm": "failed_development_0_of_3"}
        or evaluation.get("training_authorized") is not False
        or evaluation.get("checkpoint_or_recipe_selection_after_freeze") is not False
        or evaluation.get("public_data_loaded_once_for_all_checkpoints") is not True
        or evaluation.get("devices") != ["cuda:0", "cuda:1", "cuda:2"]
        or int(evaluation.get("batch_size", -1)) != 64
        or int(evaluation.get("online_environment_calls", -1)) != 0
    ):
        raise RuntimeError("Cube Public evaluation contract drifted")
    normalization = _mapping(
        evaluation.get("action_normalization"), label="action_normalization"
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
        or tuple(mean) != EXPECTED_ACTION_MEAN
        or tuple(std) != EXPECTED_ACTION_STD
        or not all(math.isfinite(float(value)) for value in [*mean, *std])
        or not all(float(value) > 0.0 for value in std)
    ):
        raise RuntimeError("Cube Public action normalization drifted")

    hidden = _mapping(
        prereg.get("scoring", {}).get("hidden_future_prediction"),
        label="scoring.hidden_future_prediction",
    )
    if hidden.get("target") != "each_checkpoint_native_frozen_encoder" or hidden.get(
        "cross_checkpoint_absolute_mse_comparison_allowed"
    ) is not False:
        raise RuntimeError("Cube Public scoring target contract drifted")
    if hidden.get("gates") != {
        "correct_future_rate_minimum": 0.75,
        "correct_history_rate_minimum": 0.75,
        "context_switch_rate_minimum": 0.90,
        "worst_rule_correct_future_rate_minimum": 0.70,
        "target_latent_separation_required": True,
        "response_gain_minimum": 0.50,
        "normalized_response_error_strict_maximum": 1.00,
    }:
        raise RuntimeError("Cube Public scoring gates drifted")
    uncertainty = _mapping(hidden.get("uncertainty"), label="uncertainty")
    if uncertainty != {
        "method": "paired_query_bootstrap",
        "unit": "rule_matched_query_pair",
        "resamples": 10_000,
        "confidence_level": 0.95,
        "random_seed": 2026080314,
        "lower_bound_minimum": {
            "correct_future_rate": 0.70,
            "correct_history_rate": 0.70,
            "context_switch_rate": 0.85,
        },
    }:
        raise RuntimeError("Cube Public uncertainty contract drifted")
    if prereg.get("one_use_policy") != {
        "generation_attempts_authorized": 1,
        "scoring_attempts_authorized_after_successful_generation": 1,
        "access_marker_written_before_public_table_read": True,
        "retry_after_access_authorized": False,
        "new_preregistration_and_namespace_required_after_failure": True,
    }:
        raise RuntimeError("Cube Public one-use policy drifted")


def _resolve_identity_path(path: str, *, root: Path) -> Path:
    value = Path(path).expanduser()
    if value.is_absolute():
        return _absolute_without_following_leaf(value)
    if value.parts and value.parts[0] == "artifacts":
        bundled = (root / value).absolute()
        if bundled.exists() or bundled.is_symlink():
            return bundled
        return artifact_root(root).joinpath(*value.parts[1:]).absolute()
    return (root / value).absolute()


def _validate_frozen_identity_group(
    entries: Any, *, label: str, root: Path, rehash: bool
) -> None:
    if not isinstance(entries, Mapping) or not entries:
        raise ValueError(f"{label} must be a non-empty mapping")
    for name, entry in entries.items():
        if not isinstance(entry, Mapping):
            raise ValueError(f"{label}.{name} must be a mapping")
        path = _resolve_identity_path(str(entry.get("path", "")), root=root)
        if rehash or entry.get("rehash_on_entrypoint") is True:
            observed = file_identity(path, logical_path=str(entry["path"]))
            _identity_matches(observed, entry, label=f"{label}.{name}")
        else:
            metadata = os.lstat(path)
            if not stat.S_ISREG(metadata.st_mode) or path.is_symlink():
                raise ValueError(f"{label}.{name} is not a regular file")
            if int(metadata.st_size) != int(entry.get("size_bytes", -1)):
                raise RuntimeError(f"{label}.{name} size mismatch")


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _require_exact_keys(
    value: Mapping[str, Any], expected: set[str] | frozenset[str], *, label: str
) -> None:
    observed = set(value)
    if observed != set(expected):
        raise RuntimeError(
            f"{label} key set mismatch: "
            f"missing={sorted(set(expected) - observed)}, "
            f"extra={sorted(observed - set(expected))}"
        )


def _identity_core(value: Mapping[str, Any]) -> dict[str, Any]:
    return {name: value.get(name) for name in ("path", "sha256", "size_bytes")}


def _source_episode_digest(values: list[int]) -> str:
    if values != sorted(set(values)) or any(value < 0 for value in values):
        raise RuntimeError("Public source exclusions must be sorted unique nonnegative")
    payload = b"".join(value.to_bytes(8, "little", signed=True) for value in values)
    return hashlib.sha256(
        b"contextworld-cube-prior-source-episodes-v1\0" + payload
    ).hexdigest()


def _content_digest(values: list[str], *, field_name: str) -> str:
    if values != sorted(set(values)):
        raise RuntimeError(f"Public {field_name} exclusions must be sorted and unique")
    decoded: list[bytes] = []
    for value in values:
        if not isinstance(value, str) or len(value) != 64:
            raise RuntimeError(f"Public {field_name} contains an invalid SHA256")
        try:
            decoded.append(bytes.fromhex(value))
        except ValueError as error:
            raise RuntimeError(
                f"Public {field_name} contains an invalid SHA256"
            ) from error
    return hashlib.sha256(
        b"contextworld-cube-prior-content-exclusions-v1\0"
        + field_name.encode("ascii")
        + b"\0"
        + b"".join(decoded)
    ).hexdigest()


def validate_public_freeze_receipt_contract(
    *,
    prereg: Mapping[str, Any],
    freeze: Mapping[str, Any],
    root: Path | None = None,
) -> None:
    root = root or repository_root()
    expected_top_level = {
        "schema_version",
        "receipt_id",
        "receipt_path",
        "preregistration_id",
        "protocol_id",
        "status",
        "frozen_at_utc",
        "checks_passed",
        "preregistration",
        "implementation_identities",
        "frozen_inputs",
        "runtime",
        "public_exclusions",
        "authorization",
        "public_test",
        "planned_artifacts",
    }
    _require_exact_keys(freeze, expected_top_level, label="freeze receipt")
    if (
        freeze.get("schema_version") != 1
        or freeze.get("receipt_id") != FREEZE_RECEIPT_ID
        or freeze.get("preregistration_id") != PREREGISTRATION_ID
        or freeze.get("protocol_id") != PROTOCOL_ID
        or freeze.get("status") != FREEZE_STATUS
        or freeze.get("checks_passed") is not True
    ):
        raise RuntimeError("Cube Public freeze receipt identity/status mismatch")
    if not isinstance(freeze.get("frozen_at_utc"), str) or not freeze["frozen_at_utc"]:
        raise RuntimeError("Cube Public freeze timestamp is missing")

    planned = _mapping(prereg.get("planned_artifacts"), label="planned_artifacts")
    _require_exact_keys(
        planned,
        {
            "freeze_receipt",
            "public_data_root",
            "public_score_root",
            "public_release_decision",
        },
        label="planned_artifacts",
    )
    frozen_planned = _mapping(
        freeze.get("planned_artifacts"), label="freeze planned_artifacts"
    )
    if dict(frozen_planned) != dict(planned):
        raise RuntimeError("Cube Public planned artifact paths drifted")
    if freeze.get("receipt_path") != planned.get("freeze_receipt"):
        raise RuntimeError("Cube Public freeze receipt path binding mismatch")

    implementations = _mapping(
        prereg.get("identity", {}).get("implementation"),
        label="identity.implementation",
    )
    receipt_implementations = _mapping(
        freeze.get("implementation_identities"),
        label="implementation_identities",
    )
    _require_exact_keys(
        implementations, EXPECTED_IMPLEMENTATION_KEYS, label="identity.implementation"
    )
    _require_exact_keys(
        receipt_implementations,
        EXPECTED_IMPLEMENTATION_KEYS,
        label="implementation_identities",
    )
    for name in EXPECTED_IMPLEMENTATION_KEYS:
        if _identity_core(receipt_implementations[name]) != _identity_core(
            _mapping(implementations[name], label=f"implementation {name}")
        ):
            raise RuntimeError(f"implementation receipt binding drifted: {name}")

    basis = _mapping(prereg.get("authorization_basis"), label="authorization_basis")
    _require_exact_keys(
        basis, EXPECTED_AUTHORIZATION_BASIS_KEYS, label="authorization_basis"
    )
    evaluation = _mapping(prereg.get("public_evaluation"), label="public_evaluation")
    checkpoints = evaluation.get("checkpoints")
    if (
        evaluation.get("authorized_model_families") != ["lewm"]
        or not isinstance(checkpoints, list)
        or len(checkpoints) != 3
        or [int(row.get("training_seed", -1)) for row in checkpoints]
        != list(EXPECTED_TRAINING_SEEDS)
        or any(row.get("model_family") != "lewm" for row in checkpoints)
        or any(
            row.get("training_recipe") != EXPECTED_TRAINING_RECIPE
            for row in checkpoints
        )
    ):
        raise RuntimeError("Cube Public frozen checkpoint family/seed set drifted")
    runtime_spec = _mapping(
        prereg.get("runtime", {}).get("stable_worldmodel"),
        label="runtime.stable_worldmodel",
    )
    runtime_files = _mapping(
        runtime_spec.get("required_files"), label="runtime required_files"
    )
    _require_exact_keys(
        runtime_files,
        EXPECTED_RUNTIME_FILE_KEYS,
        label="runtime required_files",
    )
    expected_frozen_keys = (
        set(EXPECTED_AUTHORIZATION_BASIS_KEYS)
        | {f"lewm_checkpoint_seed{seed}" for seed in EXPECTED_TRAINING_SEEDS}
        | {f"stable_worldmodel_{name}" for name in runtime_files}
        | {"source_h5"}
    )
    frozen_inputs = _mapping(freeze.get("frozen_inputs"), label="frozen_inputs")
    _require_exact_keys(frozen_inputs, expected_frozen_keys, label="frozen_inputs")
    for name in EXPECTED_AUTHORIZATION_BASIS_KEYS:
        expected = _mapping(basis[name], label=f"authorization_basis.{name}")
        observed = _mapping(frozen_inputs[name], label=f"frozen_inputs.{name}")
        if (
            _identity_core(observed) != _identity_core(expected)
            or observed.get("rehash_on_entrypoint") is not True
        ):
            raise RuntimeError(f"frozen authorization basis drifted: {name}")
    for checkpoint in checkpoints:
        seed = int(checkpoint["training_seed"])
        observed = _mapping(
            frozen_inputs[f"lewm_checkpoint_seed{seed}"],
            label=f"frozen checkpoint {seed}",
        )
        if (
            _identity_core(observed) != _identity_core(checkpoint)
            or observed.get("model_state_sha256")
            != checkpoint.get("model_state_sha256")
            or observed.get("rehash_on_entrypoint") is not True
        ):
            raise RuntimeError(f"frozen checkpoint receipt drifted: {seed}")
    runtime_repo = Path(str(runtime_spec.get("repo", ""))).expanduser()
    if not runtime_repo.is_absolute():
        runtime_repo = (root / runtime_repo).absolute()
    for name, raw in runtime_files.items():
        specification = _mapping(raw, label=f"runtime file {name}")
        expected = {
            "path": str(runtime_repo / str(specification.get("path", ""))),
            "sha256": specification.get("sha256"),
            "size_bytes": specification.get("size_bytes"),
        }
        observed = _mapping(
            frozen_inputs[f"stable_worldmodel_{name}"],
            label=f"frozen runtime file {name}",
        )
        if (
            _identity_core(observed) != expected
            or observed.get("rehash_on_entrypoint") is not True
        ):
            raise RuntimeError(f"frozen runtime receipt drifted: {name}")
    source_spec = _mapping(
        prereg.get("public_data_generation", {}).get("source_h5"),
        label="public source_h5",
    )
    source_receipt = _mapping(frozen_inputs["source_h5"], label="frozen source_h5")
    source_expected = {
        "path": source_spec.get("path"),
        "sha256": source_spec.get("sha256"),
        "size_bytes": source_spec.get("size_bytes"),
        "row_count": source_spec.get("row_count"),
        "episode_count": source_spec.get("episode_count"),
        "action_dim": source_spec.get("action_dim"),
        "content_rehash_deferred_to_public_builder_before_candidate_selection": True,
        "rehash_on_entrypoint": False,
    }
    if dict(source_receipt) != source_expected:
        raise RuntimeError("frozen source H5 receipt drifted")

    runtime = _mapping(
        freeze.get("runtime", {}).get("stable_worldmodel"),
        label="freeze runtime.stable_worldmodel",
    )
    if (
        runtime.get("path") != str(runtime_repo)
        or runtime.get("commit") != runtime_spec.get("expected_ref")
        or runtime.get("clean_worktree") is not True
        or set(_mapping(runtime.get("required_files"), label="freeze runtime files"))
        != set(runtime_files)
    ):
        raise RuntimeError("Cube Public runtime receipt drifted")

    expected_authorization = {
        "public_generation_once": True,
        "public_scoring_once_after_successful_generation": True,
        "authorized_model_families": ["lewm"],
        "training_seeds": list(EXPECTED_TRAINING_SEEDS),
        "training_or_checkpoint_selection": False,
        "threshold_or_recipe_changes": False,
        "public_test_rerun_after_access": False,
        "suite_registration": False,
    }
    if dict(_mapping(freeze.get("authorization"), label="authorization")) != expected_authorization:
        raise RuntimeError("Cube Public freeze authorization is incomplete or drifted")
    if _mapping(freeze.get("public_test"), label="freeze public_test") != {
        "access_status": "authorized_not_generated_not_opened_not_read_not_scored",
        "generated": False,
        "opened": False,
        "read": False,
        "hashed": False,
        "scored": False,
    }:
        raise RuntimeError("Cube Public freeze receipt has an invalid pre-access state")

    exclusions = _mapping(freeze.get("public_exclusions"), label="public_exclusions")
    _require_exact_keys(
        exclusions,
        {
            "checks_passed",
            "coverage",
            "excluded_source_episode_count",
            "excluded_source_episodes_sha256",
            "excluded_source_episodes",
            "prior_content_exclusions",
        },
        label="public_exclusions",
    )
    if exclusions.get("checks_passed") is not True or exclusions.get("coverage") != {
        "historical_prior_receipt": True,
        "v4r1_train": True,
        "v4r1_loader_validation": True,
        "public_content_included": False,
    }:
        raise RuntimeError("Cube Public exclusion coverage drifted")
    union_spec = _mapping(
        prereg.get("public_data_generation", {}).get("exclusion_union"),
        label="public exclusion union",
    )
    _require_exact_keys(
        union_spec,
        {"source_episodes"} | set(EXPECTED_CONTENT_EXCLUSION_FIELDS),
        label="public exclusion union",
    )
    source_values = exclusions.get("excluded_source_episodes")
    if not isinstance(source_values, list):
        raise RuntimeError("Cube Public source exclusion list is missing")
    source_values = [int(value) for value in source_values]
    source_digest = _source_episode_digest(source_values)
    source_union = _mapping(union_spec["source_episodes"], label="source union")
    if (
        len(source_values) != int(exclusions.get("excluded_source_episode_count", -1))
        or source_digest != exclusions.get("excluded_source_episodes_sha256")
        or len(source_values) != int(source_union.get("count", -1))
        or source_digest != source_union.get("sha256")
    ):
        raise RuntimeError("Cube Public source exclusion identity drifted")
    content = _mapping(
        exclusions.get("prior_content_exclusions"),
        label="prior_content_exclusions",
    )
    _require_exact_keys(
        content,
        EXPECTED_CONTENT_EXCLUSION_FIELDS,
        label="prior_content_exclusions",
    )
    for name in EXPECTED_CONTENT_EXCLUSION_FIELDS:
        entry = _mapping(content[name], label=f"public exclusion {name}")
        _require_exact_keys(entry, {"count", "sha256", "values"}, label=name)
        values = entry.get("values")
        if not isinstance(values, list):
            raise RuntimeError(f"Cube Public {name} values are missing")
        digest = _content_digest(values, field_name=name)
        expected = _mapping(union_spec[name], label=f"union {name}")
        if (
            len(values) != int(entry.get("count", -1))
            or digest != entry.get("sha256")
            or len(values) != int(expected.get("count", -1))
            or digest != expected.get("sha256")
        ):
            raise RuntimeError(f"Cube Public {name} exclusion identity drifted")


def load_public_authorization(
    *,
    preregistration_path: Path = DEFAULT_PREREGISTRATION,
    freeze_receipt_path: Path = DEFAULT_FREEZE_RECEIPT,
    require_public_absent: bool = False,
    validate_implementation_identities: bool = True,
    validate_frozen_inputs: bool = True,
) -> PublicAuthorization:
    root = repository_root()
    preregistration_path = _absolute_without_following_leaf(preregistration_path)
    freeze_receipt_path = _absolute_without_following_leaf(freeze_receipt_path)
    prereg_raw, prereg = read_yaml_nofollow(
        preregistration_path, label="Cube Public preregistration"
    )
    freeze_raw, freeze = read_json_nofollow(
        freeze_receipt_path, label="Cube Public freeze receipt"
    )

    if (
        prereg.get("schema_version") != 1
        or prereg.get("preregistration_id") != PREREGISTRATION_ID
        or prereg.get("protocol_id") != PROTOCOL_ID
        or prereg.get("status") != PREREGISTRATION_STATUS
    ):
        raise RuntimeError("Cube Public preregistration identity/status mismatch")
    _closed_public_state(prereg.get("public_test_before_freeze"), label="preregistration Public state")
    validate_public_preregistration_contract(prereg)
    if (
        freeze.get("schema_version") != 1
        or freeze.get("preregistration_id") != PREREGISTRATION_ID
        or freeze.get("protocol_id") != PROTOCOL_ID
        or freeze.get("status") != FREEZE_STATUS
        or freeze.get("checks_passed") is not True
    ):
        raise RuntimeError("Cube Public freeze receipt identity/status mismatch")
    validate_public_freeze_receipt_contract(
        prereg=prereg, freeze=freeze, root=root
    )

    observed_prereg = {
        "path": str(prereg.get("identity", {}).get("preregistration_path", "")),
        "sha256": hashlib.sha256(prereg_raw).hexdigest(),
        "size_bytes": len(prereg_raw),
    }
    _identity_matches(
        observed_prereg,
        freeze.get("preregistration", {}),
        label="Cube Public preregistration",
    )
    freeze_logical_path = str(freeze.get("receipt_path", ""))
    planned_freeze_path = str(
        prereg.get("planned_artifacts", {}).get("freeze_receipt", "")
    )
    if not freeze_logical_path or freeze_logical_path != planned_freeze_path:
        raise RuntimeError("Cube Public freeze receipt path binding mismatch")
    if (
        _resolve_identity_path(freeze_logical_path, root=root)
        != freeze_receipt_path
    ):
        raise RuntimeError("Cube Public freeze receipt resolved path mismatch")
    freeze_identity = {
        "path": freeze_logical_path,
        "sha256": hashlib.sha256(freeze_raw).hexdigest(),
        "size_bytes": len(freeze_raw),
    }
    if validate_implementation_identities:
        _validate_frozen_identity_group(
            freeze.get("implementation_identities"),
            label="implementation_identities",
            root=root,
            rehash=True,
        )
    if validate_frozen_inputs:
        _validate_frozen_identity_group(
            freeze.get("frozen_inputs"),
            label="frozen_inputs",
            root=root,
            rehash=False,
        )

    public = freeze.get("public_test")
    if not isinstance(public, Mapping) or (
        public.get("access_status")
        != "authorized_not_generated_not_opened_not_read_not_scored"
        or public.get("generated") is not False
        or public.get("opened") is not False
        or public.get("read") is not False
        or public.get("hashed") is not False
        or public.get("scored") is not False
    ):
        raise RuntimeError("Cube Public freeze receipt has an invalid pre-access state")

    result = PublicAuthorization(
        preregistration_path=preregistration_path,
        freeze_receipt_path=freeze_receipt_path,
        preregistration=prereg,
        freeze_receipt=freeze,
        freeze_receipt_identity=freeze_identity,
    )
    if require_public_absent:
        for label, path in (
            ("public_data_root", result.public_root),
            ("public_score_root", result.score_root),
            ("public_release_decision", result.decision_path),
        ):
            try:
                os.lstat(path)
            except FileNotFoundError:
                continue
            raise FileExistsError(
                f"one-use {label} already exists; this authorization cannot be reused: {path}"
            )
    return result


__all__ = [
    "DEFAULT_FREEZE_RECEIPT",
    "DEFAULT_PREREGISTRATION",
    "FREEZE_STATUS",
    "FREEZE_RECEIPT_ID",
    "PREREGISTRATION_ID",
    "PREREGISTRATION_STATUS",
    "PROTOCOL_ID",
    "PUBLIC_CANDIDATE_ASSIGNMENT_SEED",
    "PUBLIC_CATALOG_INDEX_OFFSET",
    "PUBLIC_CATALOG_SEED",
    "PUBLIC_PAIR_COUNT",
    "PUBLIC_PROFILE_SEED",
    "PUBLIC_SPLIT",
    "EXPECTED_TRAINING_RECIPE",
    "EXPECTED_TRAINING_SEEDS",
    "PublicAuthorization",
    "file_identity",
    "file_sha256",
    "load_public_authorization",
    "read_json_nofollow",
    "read_yaml_nofollow",
    "validate_public_preregistration_contract",
    "validate_public_freeze_receipt_contract",
]
