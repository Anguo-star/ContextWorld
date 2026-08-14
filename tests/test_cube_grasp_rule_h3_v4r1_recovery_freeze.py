from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest
import yaml

import scripts.freeze_cube_grasp_rule_h3_v4r1_recovery as freezer


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_yaml(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def _identity(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    return {
        "path": path.as_posix(),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
    }


def _public(*, validation_gate: bool = False, generated: bool = False) -> dict[str, Any]:
    value = {
        "access_status": "closed_not_read_not_scored",
        "opened": False,
        "read": False,
        "hashed": False,
        "scored": False,
    }
    if validation_gate:
        value["validation_lance_access_allowed"] = False
    if generated:
        value["generated"] = False
    return value


def _sets(prefix: str, counts: dict[str, int]) -> dict[str, dict[str, Any]]:
    return {
        name: {"count": count, "sha256": _digest(f"{prefix}-{name}")}
        for name, count in counts.items()
    }


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    repo = tmp_path / "repo"
    artifact_root = tmp_path / "artifacts"
    repo.mkdir(parents=True)
    artifact_root.mkdir()
    monkeypatch.setattr(freezer, "ROOT", repo)

    old_sets = _sets(
        "old",
        {
            "source_episodes": 2,
            "action_profile_ids": 2,
            "scene_template_content_hashes": 2,
            "pair_content_hashes": 2,
            "query_pixel_hashes": 2,
        },
    )
    failed_sets = _sets(
        "failed",
        {
            "source_episodes": 2,
            "action_profile_ids": 2,
            "scene_template_content_hashes": 2,
            "pair_content_hashes": 2,
            "query_pixel_hashes": 2,
        },
    )
    union_sets = _sets(
        "union",
        {
            "source_episodes": 4,
            "action_profile_ids": 4,
            "scene_template_content_hashes": 4,
            "pair_content_hashes": 4,
            "query_pixel_hashes": 4,
        },
    )
    monkeypatch.setattr(freezer, "OLD_PRIOR_IDENTITIES", old_sets)
    monkeypatch.setattr(freezer, "FAILED_SET_IDENTITIES", failed_sets)
    monkeypatch.setattr(freezer, "RECOVERY_UNION_IDENTITIES", union_sets)

    source_path = tmp_path / "source.h5"
    with h5py.File(source_path, "w") as handle:
        handle.create_dataset("action", data=np.zeros((2, 5), dtype=np.float32))
        handle.create_dataset("ep_len", data=np.asarray([2], dtype=np.int32))
    monkeypatch.setattr(freezer, "SOURCE_SHA256", freezer.file_sha256(source_path))
    monkeypatch.setattr(freezer, "SOURCE_SIZE_BYTES", source_path.stat().st_size)
    monkeypatch.setattr(freezer, "SOURCE_ROW_COUNT", 2)
    monkeypatch.setattr(freezer, "SOURCE_EPISODE_COUNT", 1)
    monkeypatch.setattr(freezer, "SOURCE_ACTION_SHAPE", (2, 5))
    source = {
        "symbol": freezer.SOURCE_SYMBOL,
        "path_recorded": False,
        "sha256": freezer.SOURCE_SHA256,
        "size_bytes": freezer.SOURCE_SIZE_BYTES,
        "row_count": freezer.SOURCE_ROW_COUNT,
        "episode_count": freezer.SOURCE_EPISODE_COUNT,
    }

    paths = {
        "old_prereg": tmp_path / "old_prereg.yaml",
        "old_freeze": tmp_path / "old_freeze.json",
        "old_prior": tmp_path / "old_prior.json",
        "decision": tmp_path / "decision.json",
        "failed": tmp_path / "failed.json",
        "query": tmp_path / "query.json",
        "support": tmp_path / "support.json",
        "prereg": repo / "configs/benchmark/recovery.yaml",
        "output": tmp_path / "recovery_freeze.json",
    }

    old_prereg = {
        "schema_version": 1,
        "protocol_id": freezer.SCIENTIFIC_PROTOCOL_ID,
        "status": freezer.OLD_PREREG_STATUS,
        "scientific_change": {
            "sole_change": "can_hold_vertical_force_coupling_n",
            "v3_baseline_vertical_force_coupling_n": 0.30,
            "v4_vertical_force_coupling_n": 0.40,
            "capability_semantics_unchanged": True,
            "history3_causal_sequence_unchanged": True,
            "action_profiles_and_constraints_unchanged_except_new_seeds": True,
        },
        "public_test": _public(validation_gate=True),
        "reference_model_phase": {"training_and_scoring_authorized": False},
    }
    _write_yaml(paths["old_prereg"], old_prereg)
    old_prereg_identity = _identity(paths["old_prereg"])
    monkeypatch.setattr(freezer, "EXPECTED_OLD_PREREG_SHA256", old_prereg_identity["sha256"])
    monkeypatch.setattr(freezer, "EXPECTED_OLD_PREREG_SIZE_BYTES", old_prereg_identity["size_bytes"])

    old_code = {
        name: {
            "path": f"artifacts/evaluation/old/{name}.py",
            "sha256": _digest(f"old-code-{name}"),
            "size_bytes": index + 1,
        }
        for index, name in enumerate(
            (
                "v4_builder",
                "v4_physics",
                "v4_probe",
                "v4_probe_tests",
                "v4_action_support_audit",
                "v4_action_support_audit_tests",
            )
        )
    }
    old_freeze = {
        "schema_version": 1,
        "protocol_id": freezer.SCIENTIFIC_PROTOCOL_ID,
        "status": freezer.OLD_FREEZE_STATUS,
        "checks_passed": True,
        "preregistration": old_prereg_identity,
        "scientific_change": {"v4_vertical_force_coupling_n": 0.40},
        "identity": old_code,
        "public_test": _public(),
        "reference_model_training_or_scoring_authorized": False,
    }
    _write_json(paths["old_freeze"], old_freeze)
    old_freeze_identity = _identity(paths["old_freeze"])
    monkeypatch.setattr(freezer, "EXPECTED_OLD_FREEZE_SHA256", old_freeze_identity["sha256"])
    monkeypatch.setattr(freezer, "EXPECTED_OLD_FREEZE_SIZE_BYTES", old_freeze_identity["size_bytes"])

    old_prior = {
        "schema_version": 1,
        "protocol_id": freezer.SCIENTIFIC_PROTOCOL_ID,
        "receipt_id": freezer.OLD_PRIOR_RECEIPT_ID,
        "status": freezer.OLD_FREEZE_STATUS,
        "checks_passed": True,
        "preregistration": old_prereg_identity,
        "freeze_receipt": old_freeze_identity,
        "source_h5": source,
        "excluded_source_episode_count": old_sets["source_episodes"]["count"],
        "excluded_source_episodes_sha256": old_sets["source_episodes"]["sha256"],
        "prior_content_exclusions": {
            name: old_sets[name] for name in freezer.CONTENT_FIELDS
        },
        "public_test": _public(),
        "reference_model_training_or_scoring": False,
    }
    _write_json(paths["old_prior"], old_prior)
    old_prior_identity = _identity(paths["old_prior"])
    monkeypatch.setattr(freezer, "EXPECTED_OLD_PRIOR_SHA256", old_prior_identity["sha256"])
    monkeypatch.setattr(freezer, "EXPECTED_OLD_PRIOR_SIZE_BYTES", old_prior_identity["size_bytes"])

    failed_output = {
        "allowed_inventory_only": True,
        "logical_root": "artifacts/synthesis/failed-v4",
        "inventory": [
            {"path": "artifacts/synthesis/failed-v4/request.json", "type": "regular_file"},
            {"path": "artifacts/synthesis/failed-v4/train.lance/data/a.lance", "type": "regular_file"},
            {"path": "artifacts/synthesis/failed-v4/train.lance/_versions", "type": "empty_directory"},
            {"path": "artifacts/synthesis/failed-v4/train.lance/_transactions", "type": "empty_directory"},
        ],
        "lance_versions_directory_empty": True,
        "lance_transactions_directory_empty": True,
    }
    failed = {
        "schema_version": 1,
        "protocol_id": freezer.SCIENTIFIC_PROTOCOL_ID,
        "receipt_id": freezer.FAILED_RECEIPT_ID,
        "status": freezer.FAILED_STATUS,
        "checks_passed": True,
        "build_passed": False,
        "formal_build_attempt_consumed": True,
        "retry_authorized_under_original_preregistration": False,
        "input_identities": {
            "preregistration": old_prereg_identity,
            "freeze_receipt": old_freeze_identity,
            "prior_exclusion_receipt": old_prior_identity,
            "source_h5": source,
        },
        "failure": {
            "stage": "lance_train_commit_atomic_rename",
            "exit_code": 1,
            "errno_name": "EPERM",
            "errno_number": 1,
        },
        "stage_completion": {
            "train_generation_accepted_pairs": 2048,
            "train_generation_attempted_candidates": 2048,
            "train_lance_data_fragment_written": True,
            "train_lance_commit_completed": False,
            "loader_validation_started": False,
            "build_report_written": False,
            "manifest_written": False,
        },
        "failed_output": failed_output,
        "scope": {
            "public_test": _public(),
            "reference_model_training_or_scoring": False,
            "optimizer_steps": 0,
            "rgb_probe_run": False,
        },
        "failed_attempt_content": {
            "source_episodes": failed_sets["source_episodes"],
            "prior_content_exclusions": {
                name: failed_sets[name] for name in freezer.CONTENT_FIELDS[:3]
            },
            "prior_overlap": {
                "source_episode_count": 0,
                "action_profile_id_count": 0,
                "scene_template_content_hash_count": 0,
                "pair_content_hash_count": 0,
            },
        },
    }
    _write_json(paths["failed"], failed)
    failed_identity = _identity(paths["failed"])
    monkeypatch.setattr(freezer, "EXPECTED_FAILED_SHA256", failed_identity["sha256"])
    monkeypatch.setattr(freezer, "EXPECTED_FAILED_SIZE_BYTES", failed_identity["size_bytes"])

    query = {
        "schema_version": 1,
        "protocol_id": freezer.SCIENTIFIC_PROTOCOL_ID,
        "receipt_id": freezer.QUERY_RECEIPT_ID,
        "status": freezer.QUERY_STATUS,
        "checks_passed": True,
        "failed_attempt_receipt": failed_identity,
        "input_identities": {
            "failed_attempt_receipt": failed_identity,
            "prior_exclusion_receipt": old_prior_identity,
            "source_h5": source,
        },
        "reconstruction_contract": {
            "fragment_read_api": "lance.file.LanceFileReader_single_file",
            "dataset_manifest_opened": False,
            "lance_written": False,
            "replayed_mode": "cannot_hold_only",
            "query_model_step_idx": 2,
            "jpeg_quality": 95,
            "jpeg_reencoding_bitwise_equal_to_fragment": True,
            "builder_snapshot_loaded_by_explicit_path": True,
            "physics_snapshot_loaded_by_explicit_path": True,
            "all_inputs_reverified_unchanged_after_replay": True,
            "raw_query_hashes_unique": True,
            "raw_query_prior_overlap_zero": True,
        },
        "failed_attempt_content": {
            "source_episodes": failed_sets["source_episodes"],
            "prior_content_exclusions": {
                name: failed_sets[name] for name in freezer.CONTENT_FIELDS
            },
        },
        "prior_overlap": {
            **{
                name: {"count": 0, "values": []}
                for name in ("source_episode", *freezer.CONTENT_FIELDS)
            },
            "passed": True,
        },
        "public_test": _public(),
        "rgb_probe": {"opened": False, "run": False, "scored": False},
        "reference_model_training_or_scoring": False,
        "reference_model_optimizer_steps": 0,
    }
    _write_json(paths["query"], query)
    query_identity = _identity(paths["query"])
    monkeypatch.setattr(freezer, "EXPECTED_QUERY_SHA256", query_identity["sha256"])
    monkeypatch.setattr(freezer, "EXPECTED_QUERY_SIZE_BYTES", query_identity["size_bytes"])

    support = {
        "schema_version": 1,
        "protocol": freezer.SCIENTIFIC_PROTOCOL_ID,
        "recovery_authorization_id": freezer.RECOVERY_AUTHORIZATION_ID,
        "audit_id": "cube_gripper_carry_h3_v4r1_action_support_v2",
        "status": "passed",
        "passed": True,
        "scope": {
            "phase": "development_only",
            "active_splits": list(freezer.ACTIVE_SPLITS),
            "frozen_profile_counts": dict(freezer.ACTION_SUPPORT_PROFILE_COUNTS),
            "total_concrete_profiles": 4608,
            "public_test_opened": False,
            "public_test_generated": False,
            "public_test_inputs": [],
            "lance_tables_opened": [],
            "formal_catalog_namespace": {
                "catalog_index_offset": freezer.V4R1_CATALOG_OFFSET,
                "local_index_policy": "zero_based_contiguous_within_each_split",
                "catalog_index_formula": (
                    "FORMAL_CATALOG_INDEX_OFFSET + local_index"
                ),
                "offset_modulo_anchor_count": 0,
                "offset_positive": True,
                "prior_catalog_namespaces_excluded": [
                    {"start_inclusive": 0, "stop_exclusive": 2},
                    {"start_inclusive": 1_000_000, "stop_exclusive": 1_002_048},
                ],
                "per_split_ranges": {
                    "train": {
                        "local_index_start_inclusive": 0,
                        "local_index_stop_exclusive": 4096,
                        "catalog_index_start_inclusive": 2_000_000,
                        "catalog_index_stop_exclusive": 2_004_096,
                    },
                    "loader_validation": {
                        "local_index_start_inclusive": 0,
                        "local_index_stop_exclusive": 512,
                        "catalog_index_start_inclusive": 2_000_000,
                        "catalog_index_stop_exclusive": 2_000_512,
                    },
                },
            },
        },
        "overall": {
            "profile_count": 4608,
            "unique_profile_count": 4608,
            "passed_profile_count": 4608,
            "conservatively_supported_profile_count": 4608,
            "failed_profile_ids": [],
            "passed": True,
        },
        "failed_v4_attempt_exclusion": {
            "failed_action_profile_count": 2048,
            "overlap_count": 0,
            "overlap_values": [],
            "passed": True,
            "recovery_action_profile_count": 4608,
        },
        "splits": {
            "train": {
                "profile_count": 4096,
                "unique_profile_count": 4096,
                "passed_profile_count": 4096,
                "conservatively_supported_profile_count": 4096,
                "action_anchor_counts": {
                    anchor: 1024 for anchor in freezer.ANCHORS
                },
                "failed_profile_ids": [],
                "passed": True,
            },
            "loader_validation": {
                "profile_count": 512,
                "unique_profile_count": 512,
                "passed_profile_count": 512,
                "conservatively_supported_profile_count": 512,
                "action_anchor_counts": {
                    anchor: 128 for anchor in freezer.ANCHORS
                },
                "failed_profile_ids": [],
                "passed": True,
            },
        },
        "cross_split": {
            "passed": True,
            "profile_content_overlap": {"count": 0, "values": []},
            "anchor_family_overlap": {
                "count": 4,
                "values": sorted(freezer.ANCHORS),
            },
        },
    }
    _write_json(paths["support"], support)
    support_identity = _identity(paths["support"])
    monkeypatch.setattr(freezer, "EXPECTED_ACTION_SUPPORT_SHA256", support_identity["sha256"])
    monkeypatch.setattr(freezer, "EXPECTED_ACTION_SUPPORT_SIZE_BYTES", support_identity["size_bytes"])

    decision = {
        "schema_version": 1,
        "protocol_id": freezer.SCIENTIFIC_PROTOCOL_ID,
        "decision_id": "cube_gripper_carry_h3_v4_infrastructure_failure_v1",
        "status": "failed_development",
        "failure_stage": "formal_build_lance_train_commit_atomic_rename",
        "classification": "infrastructure_failure_not_scientific_gate_failure",
        "checks_passed": True,
        "input_identities": {
            "current_old_preregistration": old_prereg_identity,
            "original_preregistration_snapshot": old_prereg_identity,
            "freeze": old_freeze_identity,
            "final_prior": old_prior_identity,
            "failed_receipt": failed_identity,
            "query_receipt": query_identity,
        },
        "formal_build": {
            "attempt_consumed": True,
            "train_generation": {
                "accepted_pairs": 2048,
                "attempted_candidates": 2048,
                "rejected_candidates": 0,
                "action_anchor_counts": {
                    "endpoint4": 512,
                    "front_hold": 512,
                    "plateau": 512,
                    "ramp4": 512,
                },
                "profile_constraints_passed": True,
            },
            "train_lance_commit_completed": False,
            "loader_validation_started": False,
            "development_started": False,
            "build_report_written": False,
            "manifest_written": False,
        },
        "failed_output": {
            **failed_output,
            "immutable_partial_not_canonical_dataset": True,
        },
        "failed_content_exclusions": failed_sets,
        "prior_overlap": {
            **{name: {"count": 0, "values": []} for name in failed_sets},
            "passed": True,
        },
        "recovery_exclusion_union": union_sets,
        "rgb_history_probe": {"run": False, "opened": False, "scored": False},
        "reference_model_phase": {
            "training_or_scoring_authorized": False,
            "trainer_invoked": False,
            "optimizer_steps_run": 0,
            "checkpoints_created": False,
            "lewm_or_pldm_scoring_run": False,
        },
        "public_test": _public(generated=True),
        "claims": {
            "development_ready": False,
            "data_readiness_passed": False,
            "release_claim_allowed": False,
            "suite_registration_allowed": False,
            "public_test_claim_allowed": False,
            "scientific_gate_failure_claimed": False,
        },
        "recovery_policy": {
            "original_v4_formal_attempt_consumed": True,
            "retry_authorized_under_original_preregistration": False,
            "new_frozen_recovery_preregistration_required": True,
            "original_failed_tree_must_remain_immutable": True,
            "partial_output_promotable": False,
        },
        "original_frozen_code": {
            "builder": old_code["v4_builder"],
            "physics": old_code["v4_physics"],
            "rgb_probe": old_code["v4_probe"],
            "rgb_probe_tests": old_code["v4_probe_tests"],
            "action_support": old_code["v4_action_support_audit"],
            "action_support_tests": old_code["v4_action_support_audit_tests"],
        },
        "summary": {
            "formal_build_completed": False,
            "train_generation_completed": True,
            "train_lance_commit_completed": False,
            "development_split_started": False,
            "scientific_data_gates_reached": False,
            "rgb_history_probe_reached": False,
            "development_ready": False,
        },
    }
    _write_json(paths["decision"], decision)
    decision_identity = _identity(paths["decision"])
    monkeypatch.setattr(
        freezer,
        "EXPECTED_FAILURE_DECISION_SHA256",
        decision_identity["sha256"],
    )
    monkeypatch.setattr(
        freezer,
        "EXPECTED_FAILURE_DECISION_SIZE_BYTES",
        decision_identity["size_bytes"],
    )

    identity: dict[str, dict[str, Any]] = {}
    for name in freezer.REQUIRED_IDENTITY_KEYS:
        path = repo / "identity" / f"{name}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{name}\n", encoding="utf-8")
        identity[name] = _identity(path)

    recovery_inputs = {
        "original_v4_preregistration": _identity(paths["old_prereg"]),
        "original_v4_freeze_receipt": _identity(paths["old_freeze"]),
        "old_final_prior_receipt": _identity(paths["old_prior"]),
        "infrastructure_failure_decision": _identity(paths["decision"]),
        "failed_formal_attempt_receipt": _identity(paths["failed"]),
        "query_reconstruction_receipt": _identity(paths["query"]),
        "v4r1_action_support_audit": _identity(paths["support"]),
        "source_h5": source,
    }
    document = {
        "schema_version": 1,
        "protocol_id": freezer.SCIENTIFIC_PROTOCOL_ID,
        "scientific_protocol_id": freezer.SCIENTIFIC_PROTOCOL_ID,
        "recovery_authorization_id": freezer.RECOVERY_AUTHORIZATION_ID,
        "status": freezer.PREREG_STATUS,
        "phase": "development_only",
        "recovery_inputs": recovery_inputs,
        "scientific_protocol_contract": {
            "unchanged_from_original_v4": True,
            "history_tokens": 3,
            "context_transitions": 2,
            "prediction_horizon_action_blocks": 1,
            "raw_steps_per_action_block": 5,
            "can_hold_vertical_force_coupling_n": 0.40,
            "hidden_modes": ["cannot_hold", "can_hold"],
            "action_temporal_pattern": ["p", "negative_p", "p", "terminal_zero"],
            "action_anchor_ids": list(freezer.ANCHORS),
            "sum_p_target": 0.0,
            "final_p_target": 0.0,
            "displacement_moment_weights": [4.0, 3.0, 2.0, 1.0, 0.0],
            "displacement_moment_target": 1.0,
            "constraint_absolute_tolerance": 1.0e-6,
            "jpeg_quality": 95,
            "query_state_and_pixels_equal_across_modes": True,
            "paired_actions_bitwise_equal": True,
            "no_state_installation_after_x0": True,
        },
        "recovery_contract": {
            "failure_class": "infrastructure_lance_atomic_rename_eperm",
            "original_v4_formal_attempt_consumed": True,
            "retry_under_original_v4_preregistration_authorized": False,
            "original_failed_tree_immutable": True,
            "scientific_protocol_changed": False,
            "recovery_build_attempts_authorized": 1,
            "builder_or_lance_smoke_attempts_authorized": 0,
            "rgb_history_probe_attempts_authorized": 1,
            "formal_catalog_index_offset": freezer.V4R1_CATALOG_OFFSET,
            "formal_catalog_offset_four_aligned": True,
            "failed_batch_identities_must_be_excluded": True,
        },
        "data_contract": {
            "logical_output_root": freezer.OUTPUT_LOGICAL_ROOT,
            "authorized_splits": list(freezer.ACTIVE_SPLITS),
            "pair_counts": dict(freezer.PAIR_COUNTS),
            "workers": 16,
            "episodes_per_pair": 2,
            "rows_per_pair": 8,
            "pairs_per_anchor": {"train": 512, "loader_validation": 64},
            "formal_catalog_index_offset": freezer.V4R1_CATALOG_OFFSET,
            "catalog_index_offset_modulo_anchor_count": 0,
            "source_episode_overlap_between_splits_required": 0,
            "action_profile_overlap_between_splits_required": 0,
            "scene_template_overlap_between_splits_required": 0,
            "pair_content_overlap_between_splits_required": 0,
            "query_pixel_overlap_between_splits_required": 0,
        },
        "storage_publication_contract": {
            "staging_root_class": "local_tmp_filesystem",
            "default_staging_root": "/tmp",
            "lance_commit_completed_on_local_staging": True,
            "lance_reopened_and_verified_before_publish": True,
            "local_staging_contains_success_marker": False,
            "destination_creation": "x_exclusive_copytree",
            "copy_function": "shutil.copy2",
            "dirs_exist_ok": False,
            "source_destination_file_receipts_must_match": True,
            "source_destination_lance_identities_must_match": True,
            "nonempty_directory_rename_used": False,
            "success_marker_name": freezer.SUCCESS_MARKER,
            "success_marker_written_last": True,
            "failed_copy_marked_complete": False,
        },
        "recovery_prior_exclusion_contract": {
            "old_prior": old_sets,
            "failed_attempt": failed_sets,
            "required_union": union_sets,
            "old_prior_failed_attempt_overlap_required_zero": True,
            "all_five_identity_classes_required": True,
            "finalizer_required_after_recovery_freeze": True,
        },
        "recovery_capacity_check": {
            "role": "non_scientific_deterministic_population_capacity",
            "eligible_before_old_prior": 9998,
            "removed_by_old_prior": 2321,
            "eligible_after_old_prior": 7677,
            "failed_train_source_episode_count": 2048,
            "failed_train_old_prior_overlap": 0,
            "eligible_after_recovery_union": 5629,
            "candidate_pool_required": 4608,
            "candidate_pool_margin": 1021,
            "runtime_recheck_required": True,
            "used_to_select_scientific_parameters": False,
        },
        "action_support_authorization": {
            "authorizing_audit_id": (
                "cube_gripper_carry_h3_v4r1_action_support_v2"
            ),
            "authorizing_artifact": (
                "artifacts/evaluation/history3/"
                "cube_gripper_carry_h3_development_v4r1/"
                "action_support_audit_v2.json"
            ),
            "candidate_profile_counts": dict(
                freezer.ACTION_SUPPORT_PROFILE_COUNTS
            ),
            "total_candidate_profiles": 4608,
            "v1_audit_id": "cube_gripper_carry_h3_v4r1_action_support_v1",
            "v1_status": (
                "superseded_non_authorizing_incomplete_candidate_pool_coverage"
            ),
            "v1_total_profiles": 2304,
            "v1_authorizes_recovery_freeze": False,
            "v2_is_only_authorizing_action_support_input": True,
        },
        "rgb_history_probe": {
            "attempts_authorized": 1,
            "run_only_after_complete_success_marker": True,
            "recipe_unchanged_from_v4": True,
            "thresholds_unchanged_from_v4": True,
            "recipe": copy.deepcopy(freezer.PROBE_RECIPE),
            "thresholds": copy.deepcopy(freezer.PROBE_THRESHOLDS),
            "trusted_input_contract": copy.deepcopy(
                freezer.PROBE_TRUSTED_INPUT_CONTRACT
            ),
        },
        "public_test": _public(validation_gate=True, generated=True),
        "reference_model_phase": {
            "training_and_scoring_authorized": False,
            "trainer_invoked": False,
            "optimizer_steps_authorized": 0,
            "optimizer_steps_run": 0,
            "checkpoint_creation_authorized": False,
        },
        "reference_model_training_or_scoring_authorized": False,
        "identity": identity,
    }
    monkeypatch.setattr(
        freezer,
        "EXPECTED_INPUT_LOGICAL_PATHS",
        {
            key: str(value["path"])
            for key, value in recovery_inputs.items()
            if key != "source_h5"
        },
    )
    monkeypatch.setattr(
        freezer,
        "EXPECTED_IDENTITY_PATHS",
        {key: str(value["path"]) for key, value in identity.items()},
    )
    _write_yaml(paths["prereg"], document)
    return {
        "repo": repo,
        "artifacts": artifact_root,
        "source": source_path,
        "paths": paths,
        "document": document,
    }


def _run(fixture: dict[str, Any]) -> dict[str, Any]:
    paths = fixture["paths"]
    return freezer.freeze(
        prereg_path=paths["prereg"],
        artifact_root=fixture["artifacts"],
        source_h5=fixture["source"],
        original_v4_prereg=paths["old_prereg"],
        original_v4_freeze_receipt=paths["old_freeze"],
        old_final_prior=paths["old_prior"],
        infrastructure_failure_decision=paths["decision"],
        failed_attempt_receipt=paths["failed"],
        query_reconstruction_receipt=paths["query"],
        action_support_audit=paths["support"],
        output=paths["output"],
    )


def test_success_emits_builder_and_finalizer_compatible_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    receipt = _run(fixture)
    assert receipt["protocol_id"] == freezer.SCIENTIFIC_PROTOCOL_ID
    assert receipt["recovery_authorization_id"] == freezer.RECOVERY_AUTHORIZATION_ID
    assert receipt["status"] == freezer.FREEZE_STATUS
    assert receipt["checks_passed"] is True
    assert receipt["authorized_splits"] == ["train", "loader_validation"]
    assert receipt["identity"]["v4_builder"]
    assert receipt["identity"]["v4_physics"]
    assert receipt["authorization_inputs"]["old_final_prior_receipt"]
    assert receipt["authorization_inputs"]["failed_formal_attempt_receipt"]
    assert receipt["authorization_inputs"]["query_reconstruction_receipt"]
    assert receipt["source_h5"]["path_recorded"] is False
    assert receipt["data_contract"]["formal_catalog_index_offset"] == 2_000_000
    assert receipt["data_contract"]["workers"] == 16
    assert receipt["storage_publication_contract"]["success_marker_written_last"]
    assert receipt["action_support_authorization"]["authorizing_audit_id"] == (
        "cube_gripper_carry_h3_v4r1_action_support_v2"
    )
    assert receipt["action_support_authorization"][
        "v1_authorizes_recovery_freeze"
    ] is False
    assert receipt["rgb_history_probe"]["trusted_input_contract"] == (
        freezer.PROBE_TRUSTED_INPUT_CONTRACT
    )
    assert receipt["public_test"]["read"] is False
    assert receipt["reference_model_optimizer_steps_authorized"] == 0
    assert json.loads(fixture["paths"]["output"].read_text()) == receipt


def test_failure_decision_accepts_canonical_retry_authorization_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    decision = json.loads(fixture["paths"]["decision"].read_text())
    policy = decision["recovery_policy"]
    assert policy["retry_authorized_under_original_preregistration"] is False
    assert "retry_under_original_preregistration_authorized" not in policy

    receipt = _run(fixture)
    assert receipt["checks_passed"] is True


def test_failure_decision_rejects_stale_retry_authorization_alias(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    paths = fixture["paths"]
    decision = json.loads(paths["decision"].read_text())
    policy = decision["recovery_policy"]
    del policy["retry_authorized_under_original_preregistration"]
    policy["retry_under_original_preregistration_authorized"] = False
    _write_json(paths["decision"], decision)
    identity = _identity(paths["decision"])
    monkeypatch.setattr(
        freezer, "EXPECTED_FAILURE_DECISION_SHA256", identity["sha256"]
    )
    monkeypatch.setattr(
        freezer, "EXPECTED_FAILURE_DECISION_SIZE_BYTES", identity["size_bytes"]
    )
    fixture["document"]["recovery_inputs"][
        "infrastructure_failure_decision"
    ] = identity
    _write_yaml(paths["prereg"], fixture["document"])

    with pytest.raises(RuntimeError, match="recovery policy mismatch"):
        _run(fixture)
    assert not paths["output"].exists()


@pytest.mark.parametrize(
    "mutation",
    (
        "status",
        "science",
        "offset",
        "workers",
        "storage",
        "probe_chain",
        "public",
        "public_generated",
        "model",
        "model_top_level",
        "missing_input",
        "old_freeze_binding",
        "failed_retry",
        "query_overlap",
        "support_overlap",
        "support_v1",
        "decision_claim",
        "identity_hash",
        "source_hash",
    ),
)
def test_contract_identity_or_scope_mutation_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    document = fixture["document"]
    paths = fixture["paths"]
    if mutation == "status":
        document["status"] = "draft"
    elif mutation == "science":
        document["scientific_protocol_contract"]["can_hold_vertical_force_coupling_n"] = 0.41
    elif mutation == "offset":
        document["recovery_contract"]["formal_catalog_index_offset"] = 2_000_001
    elif mutation == "workers":
        document["data_contract"]["workers"] = 8
    elif mutation == "storage":
        document["storage_publication_contract"]["success_marker_written_last"] = False
    elif mutation == "probe_chain":
        document["rgb_history_probe"]["trusted_input_contract"][
            "build_report_request_must_equal_request_json"
        ] = False
    elif mutation == "public":
        document["public_test"]["read"] = True
    elif mutation == "public_generated":
        document["public_test"]["generated"] = True
    elif mutation == "model":
        document["reference_model_phase"]["optimizer_steps_authorized"] = 1
    elif mutation == "model_top_level":
        document["reference_model_training_or_scoring_authorized"] = True
    elif mutation == "missing_input":
        del document["recovery_inputs"]["query_reconstruction_receipt"]
    elif mutation == "identity_hash":
        document["identity"]["v4_builder"]["sha256"] = "0" * 64
    elif mutation == "source_hash":
        document["recovery_inputs"]["source_h5"]["sha256"] = "0" * 64
    elif mutation == "old_freeze_binding":
        value = json.loads(paths["old_freeze"].read_text())
        value["preregistration"]["sha256"] = "0" * 64
        _write_json(paths["old_freeze"], value)
        identity = _identity(paths["old_freeze"])
        monkeypatch.setattr(freezer, "EXPECTED_OLD_FREEZE_SHA256", identity["sha256"])
        monkeypatch.setattr(freezer, "EXPECTED_OLD_FREEZE_SIZE_BYTES", identity["size_bytes"])
        document["recovery_inputs"]["original_v4_freeze_receipt"] = identity
    elif mutation == "failed_retry":
        value = json.loads(paths["failed"].read_text())
        value["retry_authorized_under_original_preregistration"] = True
        _write_json(paths["failed"], value)
        identity = _identity(paths["failed"])
        monkeypatch.setattr(freezer, "EXPECTED_FAILED_SHA256", identity["sha256"])
        monkeypatch.setattr(freezer, "EXPECTED_FAILED_SIZE_BYTES", identity["size_bytes"])
        document["recovery_inputs"]["failed_formal_attempt_receipt"] = identity
    elif mutation == "query_overlap":
        value = json.loads(paths["query"].read_text())
        value["prior_overlap"]["query_pixel_hashes"] = {
            "count": 1,
            "values": [_digest("collision")],
        }
        _write_json(paths["query"], value)
        identity = _identity(paths["query"])
        monkeypatch.setattr(freezer, "EXPECTED_QUERY_SHA256", identity["sha256"])
        monkeypatch.setattr(freezer, "EXPECTED_QUERY_SIZE_BYTES", identity["size_bytes"])
        document["recovery_inputs"]["query_reconstruction_receipt"] = identity
    elif mutation == "support_overlap":
        value = json.loads(paths["support"].read_text())
        value["failed_v4_attempt_exclusion"]["overlap_count"] = 1
        _write_json(paths["support"], value)
        identity = _identity(paths["support"])
        monkeypatch.setattr(freezer, "EXPECTED_ACTION_SUPPORT_SHA256", identity["sha256"])
        monkeypatch.setattr(freezer, "EXPECTED_ACTION_SUPPORT_SIZE_BYTES", identity["size_bytes"])
        document["recovery_inputs"]["v4r1_action_support_audit"] = identity
    elif mutation == "support_v1":
        value = json.loads(paths["support"].read_text())
        value["audit_id"] = "cube_gripper_carry_h3_v4r1_action_support_v1"
        _write_json(paths["support"], value)
        identity = _identity(paths["support"])
        monkeypatch.setattr(
            freezer, "EXPECTED_ACTION_SUPPORT_SHA256", identity["sha256"]
        )
        monkeypatch.setattr(
            freezer, "EXPECTED_ACTION_SUPPORT_SIZE_BYTES", identity["size_bytes"]
        )
        document["recovery_inputs"]["v4r1_action_support_audit"] = identity
    else:
        value = json.loads(paths["decision"].read_text())
        value["claims"]["scientific_gate_failure_claimed"] = True
        _write_json(paths["decision"], value)
        identity = _identity(paths["decision"])
        monkeypatch.setattr(
            freezer, "EXPECTED_FAILURE_DECISION_SHA256", identity["sha256"]
        )
        monkeypatch.setattr(
            freezer,
            "EXPECTED_FAILURE_DECISION_SIZE_BYTES",
            identity["size_bytes"],
        )
        document["recovery_inputs"]["infrastructure_failure_decision"] = identity
    _write_yaml(paths["prereg"], document)
    with pytest.raises((RuntimeError, FileNotFoundError)):
        _run(fixture)
    assert not paths["output"].exists()


def test_placeholder_and_output_overwrite_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    fixture["document"]["identity"]["v4_builder"]["sha256"] = "PENDING_SHA256_BUILDER"
    _write_yaml(fixture["paths"]["prereg"], fixture["document"])
    with pytest.raises(RuntimeError, match="placeholder"):
        _run(fixture)
    fixture = _fixture(tmp_path / "second", monkeypatch)
    fixture["paths"]["output"].write_text("owned\n", encoding="utf-8")
    with pytest.raises(FileExistsError):
        _run(fixture)
    assert fixture["paths"]["output"].read_text() == "owned\n"


def test_symlink_input_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    real = fixture["paths"]["decision"]
    link = tmp_path / "decision-link.json"
    link.symlink_to(real)
    with pytest.raises(FileNotFoundError, match="non-symlink"):
        freezer.freeze(
            prereg_path=fixture["paths"]["prereg"],
            artifact_root=fixture["artifacts"],
            source_h5=fixture["source"],
            original_v4_prereg=fixture["paths"]["old_prereg"],
            original_v4_freeze_receipt=fixture["paths"]["old_freeze"],
            old_final_prior=fixture["paths"]["old_prior"],
            infrastructure_failure_decision=link,
            failed_attempt_receipt=fixture["paths"]["failed"],
            query_reconstruction_receipt=fixture["paths"]["query"],
            action_support_audit=fixture["paths"]["support"],
            output=fixture["paths"]["output"],
        )


def test_postflight_mutation_fails_before_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    original = freezer._verify_postflight
    called = False

    def mutate(path: Path, raw: bytes, *, label: str) -> None:
        nonlocal called
        if not called:
            called = True
            fixture["paths"]["query"].write_text("{}\n", encoding="utf-8")
        original(path, raw, label=label)

    monkeypatch.setattr(freezer, "_verify_postflight", mutate)
    with pytest.raises(RuntimeError, match="mutated"):
        _run(fixture)
    assert not fixture["paths"]["output"].exists()


def test_postflight_implementation_identity_mutation_fails_before_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    implementation = Path(
        fixture["document"]["identity"]["base_v2_physics"]["path"]
    )
    original = freezer._verify_postflight
    called = False

    def mutate(path: Path, raw: bytes, *, label: str) -> None:
        nonlocal called
        if not called:
            called = True
            implementation.write_text("mutated\n", encoding="utf-8")
        original(path, raw, label=label)

    monkeypatch.setattr(freezer, "_verify_postflight", mutate)
    with pytest.raises(RuntimeError, match="identity mismatch"):
        _run(fixture)
    assert not fixture["paths"]["output"].exists()


def test_postflight_source_mutation_fails_before_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    original = freezer._verify_postflight
    called = False

    def mutate(path: Path, raw: bytes, *, label: str) -> None:
        nonlocal called
        if not called:
            called = True
            with h5py.File(fixture["source"], "a") as handle:
                handle.attrs["mutated_during_freeze"] = True
        original(path, raw, label=label)

    monkeypatch.setattr(freezer, "_verify_postflight", mutate)
    with pytest.raises(RuntimeError, match="source H5 full-file identity mismatch"):
        _run(fixture)
    assert not fixture["paths"]["output"].exists()


def test_cli_is_explicit_and_rejects_public_paths() -> None:
    with pytest.raises(SystemExit):
        freezer.parse_args([])
    values = [
        "--artifact-root", "artifacts",
        "--source-h5", "source.h5",
        "--original-v4-prereg", "old.yaml",
        "--original-v4-freeze-receipt", "old-freeze.json",
        "--old-final-prior", "old-prior.json",
        "--infrastructure-failure-decision", "decision.json",
        "--failed-attempt-receipt", "failed.json",
        "--query-reconstruction-receipt", "query.json",
        "--action-support-audit", "support.json",
        "--output", "out.json",
    ]
    args = freezer.parse_args(values)
    assert args.action_support_audit == Path("support.json")
    with pytest.raises(RuntimeError, match="Public"):
        freezer.parse_args([*values[:-1], "public/out.json"])


def test_cli_path_normalization_does_not_dereference_symlink(
    tmp_path: Path,
) -> None:
    real = tmp_path / "real.json"
    real.write_text("{}\n", encoding="utf-8")
    link = tmp_path / "link.json"
    link.symlink_to(real)
    normalized = freezer._absolute_without_resolve(link)
    assert normalized == link.absolute()
    assert normalized.is_symlink()


def test_freezer_has_no_build_probe_model_or_lance_execution_surface() -> None:
    source = Path(freezer.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert "lance" not in imports
    assert "torch" not in imports
    assert "subprocess" not in imports
    assert "multiprocessing" not in imports
