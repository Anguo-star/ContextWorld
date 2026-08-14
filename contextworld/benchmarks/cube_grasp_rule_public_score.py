from __future__ import annotations

import gc
import math
import os
from pathlib import Path
import statistics
from typing import Any, Mapping, Sequence

import numpy as np

from contextworld.benchmarks.adapters import (
    StableWorldModelLeWMCubeGraspRuleAdapter,
)
from contextworld.benchmarks.cube_grasp_rule_icl_data import (
    CubeGraspRuleEvalArrays,
    _read_lance_pairs,
)
from contextworld.benchmarks.cube_grasp_rule_icl_score import (
    _validate_cube_adapter_protocol,
    cube_grasp_rule_prediction_gate,
    cube_grasp_rule_prediction_metrics,
)
from contextworld.benchmarks.cube_grasp_rule_public_contract import (
    PUBLIC_PAIR_COUNT,
    PUBLIC_SPLIT,
    PublicAuthorization,
    file_identity,
    load_public_authorization,
    read_json_nofollow,
)
from contextworld.paths import repository_root
from contextworld.paths import portable_contextworld_path
import scripts.build_cube_grasp_rule_h3_v4_data as development_builder
import scripts.build_cube_grasp_rule_h3_v4r1_public_data as public_builder


BENCHMARK_ID = "cube_history3_gripper_carry_icl_v4r1_public_v1"
MATRIX_STATUS = "completed_one_use_public_scoring"


def _identity_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return all(
        left.get(name) == right.get(name)
        for name in ("path", "sha256", "size_bytes")
    )


def validate_public_publication(
    authorization: PublicAuthorization,
    *,
    verify_published_tree: bool = True,
) -> dict[str, Any]:
    root = authorization.public_root
    failure_path = root / public_builder.GENERATION_FAILURE_MARKER
    try:
        os.lstat(failure_path)
    except FileNotFoundError:
        pass
    else:
        raise RuntimeError("Cube Public generation carries a failure receipt")
    success_path = root / "_SUCCESS.json"
    _, success = read_json_nofollow(success_path, label="Cube Public success marker")
    if (
        success.get("schema_version") != 1
        or success.get("protocol_id")
        != authorization.preregistration["protocol_id"]
        or success.get("status")
        != "public_data_generated_and_integrity_validated_not_model_read_or_scored"
    ):
        raise RuntimeError("Cube Public success marker status mismatch")
    prereg_identity = file_identity(
        authorization.preregistration_path,
        logical_path=authorization.preregistration["identity"][
            "preregistration_path"
        ],
    )
    freeze_identity = file_identity(
        authorization.freeze_receipt_path,
        logical_path=authorization.freeze_receipt_identity["path"],
    )
    if not _identity_equal(success.get("preregistration", {}), prereg_identity):
        raise RuntimeError("Public data does not bind the current preregistration")
    if not _identity_equal(success.get("freeze_receipt", {}), freeze_identity):
        raise RuntimeError("Public data does not bind the current freeze receipt")

    started_path = root / public_builder.GENERATION_STARTED_MARKER
    expected_started_path = portable_contextworld_path(started_path)
    started_entry = success.get("generation_started")
    if (
        not isinstance(started_entry, Mapping)
        or started_entry.get("path") != expected_started_path
    ):
        raise RuntimeError("Public data lacks its canonical generation reservation")
    observed_started = file_identity(
        started_path, logical_path=expected_started_path
    )
    if not _identity_equal(observed_started, started_entry):
        raise RuntimeError("Public generation reservation identity mismatch")
    _, started = read_json_nofollow(
        started_path, label="Cube Public generation reservation"
    )
    if (
        started.get("schema_version") != 1
        or started.get("protocol_id")
        != authorization.preregistration["protocol_id"]
        or started.get("status")
        != "public_generation_attempt_started_one_use_namespace_reserved"
        or int(started.get("generation_attempt", -1)) != 1
        or started.get("output") != portable_contextworld_path(root)
        or started.get("public_table_opened") is not False
        or started.get("public_model_read") is not False
        or started.get("rerun_authorized") is not False
        or not _identity_equal(started.get("preregistration", {}), prereg_identity)
        or not _identity_equal(started.get("freeze_receipt", {}), freeze_identity)
    ):
        raise RuntimeError("Cube Public generation reservation contract mismatch")

    payloads: dict[str, dict[str, Any]] = {}
    for name in ("request", "build_report", "manifest"):
        entry = success.get(name)
        if not isinstance(entry, Mapping):
            raise RuntimeError(f"Public success marker lacks {name}")
        path = root / f"{name}.json"
        canonical_path = portable_contextworld_path(path)
        if entry.get("path") != canonical_path:
            raise RuntimeError(f"Public {name} logical path is not canonical")
        observed = file_identity(path, logical_path=canonical_path)
        if not _identity_equal(observed, entry):
            raise RuntimeError(f"Public {name} identity mismatch")
        _, payloads[name] = read_json_nofollow(path, label=f"Cube Public {name}")

    published = success.get("published_tree_before_success_marker")
    if not isinstance(published, Mapping):
        raise RuntimeError("Public success marker lacks the published tree identity")
    if verify_published_tree:
        observed_tree = public_builder._tree_identity(
            root, excluded_names=frozenset({public_builder.SUCCESS_MARKER})
        )
        if observed_tree != published:
            raise RuntimeError("Cube Public tree changed after publication")

    build = payloads["build_report"]
    split = build.get("splits", {}).get(PUBLIC_SPLIT, {})
    isolation = build.get("cross_split_isolation", {})
    required_isolation = {
        "source_episode_overlap_with_all_prior_content",
        "action_profile_overlap_with_all_prior_content",
        "scene_template_overlap_with_all_prior_content",
        "pair_content_overlap_with_all_prior_content",
        "query_pixel_overlap_with_all_prior_content",
    }
    if (
        build.get("passed") is not True
        or int(build.get("pair_count", -1)) != PUBLIC_PAIR_COUNT
        or split.get("passed") is not True
        or split.get("action_anchor_counts")
        != {
            "endpoint4": 64,
            "front_hold": 64,
            "plateau": 64,
            "ramp4": 64,
        }
        or split.get("prior_episode_and_content_exclusion", {}).get("passed")
        is not True
        or not isinstance(isolation, Mapping)
        or set(isolation) != required_isolation
        or any(int(value) != 0 for value in isolation.values())
    ):
        raise RuntimeError("Cube Public build did not pass frozen data gates")
    state = success.get("public_test")
    if not isinstance(state, Mapping) or (
        state.get("generated") is not True
        or state.get("hashed") is not True
        or state.get("read_by_model") is not False
        or state.get("scored") is not False
    ):
        raise RuntimeError("Cube Public success marker has an invalid access state")
    return {"success": success, **payloads}


def _require_finite_numbers(value: Any, *, label: str) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float, np.integer, np.floating)):
        if not math.isfinite(float(value)):
            raise RuntimeError(f"{label} contains a non-finite numeric value")
        return
    if isinstance(value, Mapping):
        for name, child in value.items():
            _require_finite_numbers(child, label=f"{label}.{name}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        for index, child in enumerate(value):
            _require_finite_numbers(child, label=f"{label}[{index}]")


def validate_public_checkpoint_result(
    result: Mapping[str, Any],
    *,
    authorization: PublicAuthorization,
    checkpoint_specification: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one result and recompute its frozen gate from its metrics."""

    seed = int(checkpoint_specification["training_seed"])
    if (
        result.get("schema_version") != 1
        or result.get("benchmark") != BENCHMARK_ID
        or result.get("submission_kind") != "fixed_public_checkpoint"
        or result.get("status") != "completed"
        or result.get("preregistration_id")
        != authorization.preregistration["preregistration_id"]
        or result.get("freeze_receipt_sha256")
        != authorization.freeze_receipt_identity["sha256"]
    ):
        raise RuntimeError(f"Cube Public seed {seed} result identity drifted")
    model = result.get("model")
    if not isinstance(model, Mapping) or (
        model.get("family") != "lewm"
        or model.get("name") != checkpoint_specification.get("model_name")
        or model.get("training_recipe")
        != checkpoint_specification.get("training_recipe")
        or int(model.get("training_seed", -1)) != seed
        or model.get("checkpoint_path") != checkpoint_specification.get("path")
        or model.get("checkpoint_sha256") != checkpoint_specification.get("sha256")
        or int(model.get("checkpoint_size_bytes", -1))
        != int(checkpoint_specification.get("size_bytes", -2))
        or model.get("state_sha256_before")
        != checkpoint_specification.get("model_state_sha256")
        or model.get("state_sha256_after")
        != checkpoint_specification.get("model_state_sha256")
        or not isinstance(model.get("adapter"), Mapping)
    ):
        raise RuntimeError(f"Cube Public seed {seed} model provenance drifted")
    if result.get("data") != {
        "split": "Public Test",
        "lance_table": "validation.lance",
        "pair_count": PUBLIC_PAIR_COUNT,
        "condition_count": 2 * PUBLIC_PAIR_COUNT,
        "online_environment_calls": 0,
        "model_visible_fields": ["history_pixels", "query_action_blocks"],
        "privileged_columns_passed_to_model": False,
    }:
        raise RuntimeError(f"Cube Public seed {seed} data boundary drifted")
    metrics = result.get("metrics")
    if not isinstance(metrics, Mapping):
        raise RuntimeError(f"Cube Public seed {seed} metrics are missing")
    _require_finite_numbers(metrics, label=f"seed {seed} metrics")
    if (
        int(metrics.get("pair_count", -1)) != PUBLIC_PAIR_COUNT
        or int(metrics.get("decision_count", -1)) != 2 * PUBLIC_PAIR_COUNT
    ):
        raise RuntimeError(f"Cube Public seed {seed} metric cardinality drifted")
    expected_gate = cube_grasp_rule_prediction_gate(
        dict(metrics), release=authorization.preregistration
    )
    if result.get("gate") != expected_gate:
        raise RuntimeError(f"Cube Public seed {seed} gate was not recomputed exactly")
    records = result.get("records")
    if records is not None and (
        not isinstance(records, list) or len(records) != PUBLIC_PAIR_COUNT
    ):
        raise RuntimeError(f"Cube Public seed {seed} record cardinality drifted")
    return dict(result)


def load_public_arrays(
    authorization: PublicAuthorization,
    publication: Mapping[str, Any],
) -> CubeGraspRuleEvalArrays:
    split = publication["build_report"]["splits"][PUBLIC_SPLIT]
    table = authorization.public_root / "validation.lance"
    development_builder._lance_table_identity(
        table,
        expected_row_count=8 * PUBLIC_PAIR_COUNT,
        expected_tree_sha256=str(split["table_sha256"]),
    )
    return _read_lance_pairs(
        table,
        expected_pairs=PUBLIC_PAIR_COUNT,
        expected_split=PUBLIC_SPLIT,
    )


def evaluate_public_checkpoint(
    *,
    adapter: StableWorldModelLeWMCubeGraspRuleAdapter,
    arrays: CubeGraspRuleEvalArrays,
    authorization: PublicAuthorization,
    checkpoint_specification: Mapping[str, Any],
    batch_size: int,
    include_records: bool = True,
) -> dict[str, Any]:
    _validate_cube_adapter_protocol(adapter)
    if arrays.pair_count != PUBLIC_PAIR_COUNT:
        raise RuntimeError("Cube Public arrays have an unexpected pair count")
    histories = np.concatenate(
        [arrays.cannot_hold_pixels[:, :3], arrays.can_hold_pixels[:, :3]]
    )
    actions = np.concatenate(
        [arrays.raw_action_blocks[:, :3], arrays.raw_action_blocks[:, :3]]
    )
    before = adapter.frozen_state_hash()
    expected_state = str(checkpoint_specification["model_state_sha256"])
    if before != expected_state:
        raise RuntimeError("Cube Public adapter state differs from the frozen checkpoint")
    predicted = adapter.rollout_latents(histories, actions, batch_size=batch_size)
    count = arrays.pair_count
    if (
        predicted.ndim != 3
        or predicted.shape[0] != 2 * count
        or predicted.shape[1] != 1
        or not np.isfinite(predicted).all()
    ):
        raise RuntimeError("Cube Public adapter returned invalid predicted latents")
    true_futures = np.concatenate(
        [arrays.cannot_hold_pixels[:, 3], arrays.can_hold_pixels[:, 3]]
    )
    encoded = adapter.encode_pixels(true_futures, batch_size=batch_size)
    if (
        encoded.shape != (2 * count, predicted.shape[2])
        or not np.isfinite(encoded).all()
    ):
        raise RuntimeError("Cube Public encoded targets do not match predictions")
    after = adapter.frozen_state_hash()
    if before != after:
        raise RuntimeError("model state changed during Cube Public scoring")
    metrics, records = cube_grasp_rule_prediction_metrics(
        pair_ids=arrays.pair_ids,
        predicted_cannot_hold=predicted[:count, 0],
        predicted_can_hold=predicted[count:, 0],
        target_cannot_hold=encoded[:count],
        target_can_hold=encoded[count:],
    )
    result = {
        "schema_version": 1,
        "benchmark": BENCHMARK_ID,
        "submission_kind": "fixed_public_checkpoint",
        "status": "completed",
        "preregistration_id": authorization.preregistration[
            "preregistration_id"
        ],
        "freeze_receipt_sha256": authorization.freeze_receipt_identity[
            "sha256"
        ],
        "model": {
            "family": "lewm",
            "name": str(checkpoint_specification["model_name"]),
            "training_recipe": str(checkpoint_specification["training_recipe"]),
            "training_seed": int(checkpoint_specification["training_seed"]),
            "checkpoint_path": str(checkpoint_specification["path"]),
            "checkpoint_sha256": str(checkpoint_specification["sha256"]),
            "checkpoint_size_bytes": int(checkpoint_specification["size_bytes"]),
            "adapter": adapter.metadata,
            "state_sha256_before": before,
            "state_sha256_after": after,
        },
        "data": {
            "split": "Public Test",
            "lance_table": "validation.lance",
            "pair_count": arrays.pair_count,
            "condition_count": 2 * arrays.pair_count,
            "online_environment_calls": 0,
            "model_visible_fields": ["history_pixels", "query_action_blocks"],
            "privileged_columns_passed_to_model": False,
        },
        "metrics": metrics,
        "gate": cube_grasp_rule_prediction_gate(
            metrics, release=authorization.preregistration
        ),
    }
    if include_records:
        result["records"] = records
    return result


def aggregate_public_results(
    results: Sequence[Mapping[str, Any]],
    *,
    authorization: PublicAuthorization,
) -> dict[str, Any]:
    required = authorization.preregistration["public_evaluation"]["checkpoints"]
    expected_seeds = [int(row["training_seed"]) for row in required]
    expected_by_seed = {int(row["training_seed"]): row for row in required}
    observed_seeds = [int(row.get("model", {}).get("training_seed", -1)) for row in results]
    if len(results) != 3 or sorted(observed_seeds) != sorted(expected_seeds):
        raise RuntimeError("Cube Public matrix must contain exactly three frozen seeds")
    validated = [
        validate_public_checkpoint_result(
            row,
            authorization=authorization,
            checkpoint_specification=expected_by_seed[
                int(row["model"]["training_seed"])
            ],
        )
        for row in results
    ]
    names = (
        "correct_future_rate",
        "correct_history_rate",
        "context_switch_rate",
        "worst_rule_correct_future_rate",
        "other_minus_correct_mse_margin_mean",
        "joint_icl_pair_success_rate",
    )
    return {
        "schema_version": 1,
        "benchmark": BENCHMARK_ID,
        "submission_kind": "fixed_three_seed_public_method",
        "status": MATRIX_STATUS,
        "model_family": "lewm",
        "training_recipe": required[0]["training_recipe"],
        "training_seeds": sorted(observed_seeds),
        "checkpoint_results": validated,
        "aggregate": {
            name: {
                "mean": float(statistics.mean(row["metrics"][name] for row in validated)),
                "minimum": float(min(row["metrics"][name] for row in validated)),
                "maximum": float(max(row["metrics"][name] for row in validated)),
            }
            for name in names
        },
        "checkpoints_passed": sum(
            row["gate"]["passed"] is True for row in validated
        ),
        "checkpoints_required": 3,
        "passed": all(row["gate"]["passed"] is True for row in validated),
        "public_test": {
            "generated": True,
            "hashed": True,
            "opened": True,
            "read": True,
            "scored": True,
            "used_for_training_or_selection": False,
        },
    }


def build_adapter(
    *,
    authorization: PublicAuthorization,
    checkpoint: Mapping[str, Any],
    device: str,
) -> StableWorldModelLeWMCubeGraspRuleAdapter:
    prereg = authorization.preregistration
    normalization = prereg["public_evaluation"]["action_normalization"]
    runtime = prereg["runtime"]["stable_worldmodel"]
    path = Path(str(checkpoint["path"])).expanduser().resolve()
    observed = file_identity(path, logical_path=str(checkpoint["path"]))
    if any(
        observed.get(name) != checkpoint.get(name)
        for name in ("path", "sha256", "size_bytes")
    ):
        raise RuntimeError("Cube Public checkpoint identity drifted")
    adapter = StableWorldModelLeWMCubeGraspRuleAdapter.from_checkpoint(
        path,
        action_mean=normalization["mean"],
        action_std=normalization["std_population"],
        repo_root=repository_root(),
        stablewm_repo=runtime["repo"],
        stablewm_ref=runtime["expected_ref"],
        device=device,
    )
    observed_state = adapter.frozen_state_hash()
    if observed_state != str(checkpoint["model_state_sha256"]):
        release_adapter(adapter)
        raise RuntimeError("Cube Public checkpoint model-state identity drifted")
    return adapter


def release_adapter(adapter: Any) -> None:
    del adapter
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


__all__ = [
    "BENCHMARK_ID",
    "MATRIX_STATUS",
    "aggregate_public_results",
    "build_adapter",
    "evaluate_public_checkpoint",
    "load_public_arrays",
    "release_adapter",
    "validate_public_publication",
    "validate_public_checkpoint_result",
]
