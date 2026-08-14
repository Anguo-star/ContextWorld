from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil

import h5py
import numpy as np
import pytest
import yaml

from contextworld.evaluation.cube_grasp_rule_h3_v4 import (
    GRASP_MODES,
    CubeGraspRuleCandidate,
    action_blocks as v4_action_blocks,
    make_v4_candidate,
    make_v4_action_profile,
)
import scripts.build_cube_grasp_rule_h3_v4_data as builder


def test_v4_builder_is_development_only_by_construction() -> None:
    assert builder.PROTOCOL == "cube_gripper_carry_rule_history3_development_v4"
    assert (
        builder.RECOVERY_AUTHORIZATION_ID
        == "cube_gripper_carry_h3_development_v4r1"
    )
    assert builder.ACTIVE_SPLITS == ("train", "loader_validation")
    assert builder.DEFAULT_PAIR_COUNTS == {
        "train": 2048,
        "loader_validation": 256,
    }
    assert builder.FROZEN_JPEG_QUALITY == 95
    assert builder.FROZEN_WORKERS == 16
    assert builder.DEFAULT_OUTPUT_LOGICAL == Path(
        "artifacts/synthesis/cube_gripper_carry_rule_h3_development_v4r1"
    )
    assert builder.DEFAULT_FREEZE_RECEIPT_LOGICAL == Path(
        "artifacts/evaluation/history3/"
        "cube_gripper_carry_h3_development_v4r1/"
        "development_recovery_freeze_receipt_v1.json"
    )
    assert builder.DEFAULT_FREEZE_RECEIPT.name == (
        "development_recovery_freeze_receipt_v1.json"
    )
    assert not hasattr(builder, "DEFAULT_SOURCE")
    assert builder.SOURCE_SYMBOL == "upstream_cube_single_expert_h5"
    assert builder.EVIDENCE_SCOPE == (
        "every accepted pair in Training and Development"
    )
    assert builder.PROFILE_SPLIT_POLICY == (
        "shared_families_disjoint_profiles"
    )


def test_v4_schema_marks_exact_action_profile_and_anchor() -> None:
    assert "action_anchor_id" in builder.SCHEMA.names
    assert "action_profile_id" in builder.SCHEMA.names
    assert "scene_template_content_hash" in builder.SCHEMA.names
    assert "pair_content_hash" in builder.SCHEMA.names
    assert "source_step" in builder.SCHEMA.names
    assert "action_anchor_id" in builder.PRIVILEGED_COLUMNS
    assert "action_profile_id" in builder.PRIVILEGED_COLUMNS
    assert "scene_template_content_hash" in builder.PRIVILEGED_COLUMNS
    assert "pair_content_hash" in builder.PRIVILEGED_COLUMNS
    assert "episode_idx" in builder.PRIVILEGED_COLUMNS
    assert "model_step_idx" in builder.PRIVILEGED_COLUMNS
    assert "source_step" in builder.PRIVILEGED_COLUMNS
    assert builder.SCHEMA.field("action_anchor_id").type == builder.pa.string()
    assert builder.SCHEMA.field("action_profile_id").type == builder.pa.string()
    assert (
        builder.SCHEMA.field("scene_template_content_hash").type
        == builder.pa.string()
    )
    assert builder.SCHEMA.field("pair_content_hash").type == builder.pa.string()


@pytest.mark.parametrize(
    "counts",
    (
        {"train": 0, "loader_validation": 256},
        {"train": -4, "loader_validation": 256},
        {"train": 2046, "loader_validation": 256},
        {"train": 2048, "loader_validation": 2},
        {"train": True, "loader_validation": 256},
        {"train": 2048},
        {"train": 2048, "loader_validation": 256, "validation": 4},
        {"train": 2048, "loader_validation": 256, "public_test": 4},
    ),
)
def test_v4_pair_count_contract_rejects_invalid_or_public_counts(
    counts: dict[str, int],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        builder._validate_pair_counts(counts)


@pytest.mark.parametrize(
    "counts",
    (
        {"train": 4, "loader_validation": 4},
        {"train": 2044, "loader_validation": 256},
        {"train": 2048, "loader_validation": 252},
    ),
)
def test_v4r1_formal_pair_counts_are_exactly_frozen(
    counts: dict[str, int],
) -> None:
    with pytest.raises(ValueError, match="pair counts are frozen"):
        builder._validate_frozen_pair_counts(counts)
    assert builder._validate_frozen_pair_counts(builder.DEFAULT_PAIR_COUNTS) == (
        builder.DEFAULT_PAIR_COUNTS
    )


@pytest.mark.parametrize("quality", (1, 94, 96, 100, True, 95.0))
def test_v4r1_formal_jpeg_quality_is_exactly_frozen(quality: object) -> None:
    with pytest.raises((TypeError, ValueError), match="quality"):
        builder._validate_frozen_jpeg_quality(quality)  # type: ignore[arg-type]
    assert builder._validate_frozen_jpeg_quality(95) == 95


@pytest.mark.parametrize("workers", (0, 1, 8, 15, 17, True, 16.0))
def test_v4r1_formal_worker_count_is_exactly_frozen(workers: object) -> None:
    with pytest.raises((TypeError, ValueError), match="worker"):
        builder._validate_frozen_workers(workers)  # type: ignore[arg-type]
    assert builder._validate_frozen_workers(16) == 16


def test_formal_output_and_local_staging_roots_are_frozen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "artifact-root" / "v4r1"
    monkeypatch.setattr(builder, "DEFAULT_OUTPUT", output)
    assert builder._validate_formal_output(output) == output.resolve()
    with pytest.raises(ValueError, match="formal output is frozen"):
        builder._validate_formal_output(tmp_path / "other")

    staging = tmp_path / "staging"
    staging.mkdir()
    assert builder._validate_local_staging_root(
        staging, output=output.resolve()
    ) == staging.resolve()
    with pytest.raises(ValueError, match="outside the local staging root"):
        builder._validate_local_staging_root(
            staging, output=(staging / "published").resolve()
        )
    with pytest.raises(ValueError, match="must be /tmp"):
        builder._validate_local_staging_root(
            Path("/opt"), output=output.resolve()
        )


def test_formal_output_rejects_symlink_in_parent_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    output = alias / "v4r1"
    monkeypatch.setattr(builder, "DEFAULT_OUTPUT", output)
    with pytest.raises(
        (OSError, ValueError), match="symlink|Too many levels|Not a directory"
    ):
        builder._validate_formal_output(output)


@pytest.mark.parametrize(
    "option",
    ("--public-test-pairs", "--validation-pairs", "--test-pairs"),
)
def test_v4_cli_explicitly_refuses_public_pair_options(option: str) -> None:
    with pytest.raises(ValueError, match="explicitly refuses"):
        builder.parse_args([option, "4"])


def test_v4_cli_requires_explicit_source() -> None:
    with pytest.raises(SystemExit):
        builder.parse_args([])


def test_v4_cli_requires_all_frozen_inputs() -> None:
    with pytest.raises(SystemExit):
        builder.parse_args(["--source", "training-source.h5"])
    args = builder.parse_args(
        [
            "--source",
            "training-source.h5",
            "--prereg",
            "prereg.yaml",
            "--freeze-receipt",
            "freeze.json",
            "--prior-episode-exclusion-receipt",
            "prior.json",
        ]
    )
    assert args.source == Path("training-source.h5")
    assert args.prereg == Path("prereg.yaml")
    assert args.freeze_receipt == Path("freeze.json")
    assert args.prior_episode_exclusion_receipt == Path("prior.json")
    assert args.staging_root == Path("/tmp")


def test_formal_input_gate_rejects_public_alias_and_noncanonical_path(
    tmp_path: Path,
) -> None:
    real = tmp_path / "real.yaml"
    real.write_text("fixture\n", encoding="utf-8")
    alias = tmp_path / "alias.yaml"
    alias.symlink_to(real)
    with pytest.raises(ValueError, match="non-symlink"):
        builder._validate_formal_input_file(alias, label="prereg")
    with pytest.raises(ValueError, match="frozen at"):
        builder._validate_formal_input_file(
            real,
            label="prereg",
            expected=tmp_path / "canonical.yaml",
        )
    with pytest.raises(ValueError, match="forbidden Public"):
        builder.parse_args(
            [
                "--source",
                str(tmp_path / "public" / "source.h5"),
                "--prereg",
                str(real),
                "--freeze-receipt",
                str(real),
                "--prior-episode-exclusion-receipt",
                str(real),
            ]
        )


def test_action_profile_id_hashes_only_canonical_float32_content() -> None:
    blocks = np.arange(4 * 5 * 5, dtype=np.float32).reshape(4, 5, 5) / 101.0
    blocks[3] = 0.0
    expected = hashlib.sha256(np.ascontiguousarray(blocks).tobytes()).hexdigest()
    assert builder.action_profile_content_sha256(blocks) == expected
    assert builder.action_profile_content_sha256(blocks.astype(np.float64)) == expected

    changed = blocks.copy()
    changed[2, 4, 3] = np.nextafter(changed[2, 4, 3], np.float32(np.inf))
    assert builder.action_profile_content_sha256(changed) != expected


@pytest.mark.parametrize(
    "blocks",
    (
        np.zeros((4, 5, 4), dtype=np.float32),
        np.full((4, 5, 5), np.nan, dtype=np.float32),
    ),
)
def test_action_profile_id_rejects_wrong_shape_or_nonfinite(
    blocks: np.ndarray,
) -> None:
    with pytest.raises(ValueError):
        builder.action_profile_content_sha256(blocks)


def test_action_profile_contract_rejects_nonzero_terminal_fourth_block() -> None:
    blocks = np.zeros((4, 5, 5), dtype=np.float32)
    blocks[3, 2, 4] = np.float32(0.25)
    with pytest.raises(ValueError, match="terminal fourth"):
        builder.action_profile_content_sha256(blocks)


def test_scene_template_hash_excludes_split_ids_and_action_metadata() -> None:
    candidate = asdict(make_v4_candidate(_base_candidate("train", 3)))
    expected = builder.scene_template_content_sha256(candidate)

    metadata_changed = dict(candidate)
    metadata_changed["candidate_id"] = "different-identity"
    metadata_changed["split"] = "different-split-label"
    metadata_changed["action_profile"] = {
        "action_anchor_id": "different-anchor",
        "action_profile_id": "f" * 64,
    }
    assert builder.scene_template_content_sha256(metadata_changed) == expected

    content_changed = dict(candidate)
    content_changed["cube_color"] = (0.31, 0.4, 0.5)
    assert builder.scene_template_content_sha256(content_changed) != expected


@pytest.mark.parametrize(
    ("field", "size"),
    (("qpos", 20), ("qpos", 22), ("control", 6), ("control", 8)),
)
def test_scene_template_hash_freezes_source_vector_dimensions(
    field: str,
    size: int,
) -> None:
    candidate = asdict(make_v4_candidate(_base_candidate("train", 0)))
    candidate[field] = tuple(0.0 for _ in range(size))
    with pytest.raises(ValueError, match="must contain"):
        builder.scene_template_content_sha256(candidate)


def test_pair_content_hash_binds_raw_scene_and_action_digest_bytes() -> None:
    candidate = make_v4_candidate(_base_candidate("train", 5))
    scene_hash = builder.scene_template_content_sha256(candidate)
    profile_hash = candidate.action_profile.action_profile_id
    expected = hashlib.sha256(
        bytes.fromhex(scene_hash) + bytes.fromhex(profile_hash)
    ).hexdigest()
    assert builder.pair_content_sha256(scene_hash, profile_hash) == expected
    assert builder.pair_content_sha256("0" * 64, profile_hash) != expected
    assert builder.pair_content_sha256(scene_hash, "0" * 64) != expected


@pytest.mark.parametrize("invalid", ("", "g" * 64, "0" * 63))
def test_pair_content_hash_rejects_invalid_digest_inputs(invalid: str) -> None:
    with pytest.raises(ValueError):
        builder.pair_content_sha256(invalid, "0" * 64)


def test_source_h5_receipt_binds_size_rows_hash_and_selection_rule(
    tmp_path: Path,
) -> None:
    source = tmp_path / "training-source.h5"
    shapes = {
        "qpos": (3, 21),
        "control": (3, 7),
        "action": (3, 5),
        "ep_idx": (3,),
        "step_idx": (3,),
        "proprio_gripper_contact": (3, 1),
        "proprio_gripper_opening": (3, 1),
        "privileged_block_0_pos": (3, 3),
        "proprio_effector_pos": (3, 3),
    }
    with h5py.File(source, "w") as handle:
        for name, shape in shapes.items():
            handle.create_dataset(name, data=np.zeros(shape, dtype=np.float32))
        handle.create_dataset("ep_len", data=np.asarray([1, 2], dtype=np.int32))
    receipt = builder._source_h5_receipt(
        source,
        eligible_episode_count=2,
        frozen_source_identity={
            "size_bytes": source.stat().st_size,
            "row_count": 3,
            "episode_count": 2,
            "sha256": "a" * 64,
        },
    )
    assert receipt["source_size_bytes"] == source.stat().st_size
    assert receipt["source_row_count"] == 3
    assert receipt["source_episode_count"] == 2
    assert receipt["source_file_sha256"] == "a" * 64
    assert receipt["source_symbol"] == builder.SOURCE_SYMBOL
    assert receipt["source_content_hash_reused_from_validated_freeze_receipt"]
    assert receipt[
        "source_content_rehashed_by_builder_before_candidate_selection"
    ] is False
    assert receipt["eligible_source_episode_count"] == 2
    assert receipt["eligible_row_selection_rule"] == (
        builder.ELIGIBLE_ROW_SELECTION_RULE
    )


def _write_freeze_receipt_fixture(tmp_path: Path) -> dict[str, object]:
    prereg = tmp_path / "prereg.yaml"
    physics = tmp_path / "physics.py"
    builder_file = tmp_path / "builder.py"
    source = tmp_path / "source.h5"
    receipt_path = tmp_path / "freeze-receipt.json"
    physics.write_text("V4 = 'physics'\n", encoding="utf-8")
    builder_file.write_text("V4 = 'builder'\n", encoding="utf-8")
    with h5py.File(source, "w") as handle:
        handle.create_dataset(
            "action",
            data=np.zeros((3, 5), dtype=np.float32),
        )
        handle.create_dataset("ep_len", data=np.asarray([1, 2], dtype=np.int32))
    public = {
        "access_status": "closed_not_read_not_scored",
        "validation_lance_access_allowed": False,
        "opened": False,
        "read": False,
        "scored": False,
        "hashed": False,
    }
    scientific = {
        "unchanged_from_original_v4": True,
        "history_tokens": 3,
        "can_hold_vertical_force_coupling_n": 0.40,
        "action_temporal_pattern": ["p", "negative_p", "p", "terminal_zero"],
        "action_anchor_ids": list(builder._anchor_ids()),
        "sum_p_target": 0.0,
        "final_p_target": 0.0,
        "displacement_moment_target": 1.0,
        "jpeg_quality": 95,
    }
    recovery = {
        "original_v4_formal_attempt_consumed": True,
        "retry_under_original_v4_preregistration_authorized": False,
        "scientific_protocol_changed": False,
        "formal_catalog_index_offset": builder.FORMAL_CATALOG_INDEX_OFFSET,
    }
    storage = {
        "staging_root_class": "local_tmp_filesystem",
        "default_staging_root": "/tmp",
        "destination_creation": "x_exclusive_copytree",
        "dirs_exist_ok": False,
        "nonempty_directory_rename_used": False,
        "success_marker_name": builder.SUCCESS_MARKER_NAME,
        "success_marker_written_last": True,
        "failed_copy_marked_complete": False,
    }
    data = {
        "logical_output_root": builder.DEFAULT_OUTPUT_LOGICAL.as_posix(),
        "authorized_splits": list(builder.ACTIVE_SPLITS),
        "pair_counts": dict(builder.DEFAULT_PAIR_COUNTS),
        "workers": builder.FROZEN_WORKERS,
        "episodes_per_pair": 2,
        "rows_per_pair": 8,
        "formal_catalog_index_offset": builder.FORMAL_CATALOG_INDEX_OFFSET,
    }
    action_support = {
        "authorizing_audit_id": "cube_gripper_carry_h3_v4r1_action_support_v2",
        "candidate_profile_counts": {"train": 4096, "loader_validation": 512},
        "total_candidate_profiles": 4608,
        "v2_is_only_authorizing_action_support_input": True,
    }
    canonical_contract = yaml.safe_load(
        builder.DEFAULT_PREREG.read_text(encoding="utf-8")
    )
    scientific = canonical_contract["scientific_protocol_contract"]
    recovery = canonical_contract["recovery_contract"]
    storage = canonical_contract["storage_publication_contract"]
    data = canonical_contract["data_contract"]
    action_support = canonical_contract["action_support_authorization"]
    placeholder_inputs = {
        name: {
            "path": f"artifacts/fixture/{name}.json",
            "sha256": hashlib.sha256(name.encode()).hexdigest(),
            "size_bytes": len(name) + 1,
        }
        for name in builder.REQUIRED_FREEZE_AUTHORIZATION_INPUT_KEYS
    }
    placeholder_inputs["v4r1_action_support_audit"] = {
        "path": "artifacts/evaluation/action_support_audit_v2.json",
        "sha256": builder.ACTION_SUPPORT_AUDIT_SHA256,
        "size_bytes": builder.ACTION_SUPPORT_AUDIT_SIZE_BYTES,
    }
    implementation = {
        name: {
            "path": builder.EXPECTED_FREEZE_IDENTITY_PATHS[name],
            "sha256": builder.file_sha256(
                builder.ROOT / builder.EXPECTED_FREEZE_IDENTITY_PATHS[name]
            ),
            "size_bytes": (
                builder.ROOT / builder.EXPECTED_FREEZE_IDENTITY_PATHS[name]
            ).stat().st_size,
        }
        for name in builder.REQUIRED_FREEZE_IDENTITY_KEYS
    }
    implementation["v4_builder"] = {
        "path": "scripts/build_cube_grasp_rule_h3_v4_data.py",
        "sha256": builder.file_sha256(builder_file),
        "size_bytes": builder_file.stat().st_size,
    }
    implementation["v4_physics"] = {
        "path": "contextworld/evaluation/cube_grasp_rule_h3_v4.py",
        "sha256": builder.file_sha256(physics),
        "size_bytes": physics.stat().st_size,
    }
    prereg_payload = {
        "schema_version": 1,
        "protocol_id": builder.PROTOCOL,
        "scientific_protocol_id": builder.PROTOCOL,
        "recovery_authorization_id": builder.RECOVERY_AUTHORIZATION_ID,
        "status": "preregistered_before_v4r1_recovery_build",
        "phase": "development_only",
        "reference_model_training_or_scoring_authorized": False,
        "public_test": public,
        "scientific_protocol_contract": scientific,
        "recovery_contract": recovery,
        "storage_publication_contract": storage,
        "data_contract": data,
        "recovery_prior_exclusion_contract": {
            "old_prior": builder.EXPECTED_OLD_PRIOR_IDENTITIES,
            "failed_attempt": builder.EXPECTED_FAILED_ATTEMPT_IDENTITIES,
            "required_union": builder.EXPECTED_RECOVERY_UNION_IDENTITIES,
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
        "action_support_authorization": action_support,
        "rgb_history_probe": {
            "attempts_authorized": 1,
            "run_only_after_complete_success_marker": True,
            "recipe_unchanged_from_v4": True,
            "thresholds_unchanged_from_v4": True,
            "recipe": builder.FROZEN_PROBE_RECIPE,
            "thresholds": builder.FROZEN_PROBE_THRESHOLDS,
            "trusted_input_contract": builder.FROZEN_PROBE_TRUSTED_INPUT_CONTRACT,
        },
        "recovery_inputs": placeholder_inputs,
        "identity": implementation,
    }
    prereg.write_text(
        yaml.safe_dump(prereg_payload, sort_keys=False), encoding="utf-8"
    )
    payload = {
        "schema_version": 1,
        "protocol_id": builder.PROTOCOL,
        "scientific_protocol_id": builder.PROTOCOL,
        "recovery_authorization_id": builder.RECOVERY_AUTHORIZATION_ID,
        "status": builder.FREEZE_STATUS,
        "checks_passed": True,
        "authorized_splits": ["train", "loader_validation"],
        "public_test": public,
        "reference_model_training_or_scoring_authorized": False,
        "reference_model_optimizer_steps_authorized": 0,
        "recovery_build_attempts_authorized": 1,
        "rgb_history_probe_attempts_authorized": 1,
        "preregistration": {
            "path": str(prereg),
            "sha256": builder.file_sha256(prereg),
            "size_bytes": prereg.stat().st_size,
        },
        "identity": implementation,
        "authorization_inputs": placeholder_inputs,
        "scientific_protocol_contract": scientific,
        "recovery_contract": recovery,
        "storage_publication_contract": storage,
        "data_contract": data,
        "recovery_prior_exclusion_contract": prereg_payload[
            "recovery_prior_exclusion_contract"
        ],
        "recovery_capacity_check": prereg_payload["recovery_capacity_check"],
        "action_support_authorization": action_support,
        "rgb_history_probe": prereg_payload["rgb_history_probe"],
        "source_h5": {
            "symbol": builder.SOURCE_SYMBOL,
            "path_recorded": False,
            "sha256": builder.file_sha256(source),
            "size_bytes": source.stat().st_size,
            "row_count": 3,
            "episode_count": 2,
        },
    }
    receipt_path.write_text(
        json.dumps(payload, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "prereg": prereg,
        "physics": physics,
        "builder": builder_file,
        "source": source,
        "receipt": receipt_path,
        "payload": payload,
    }


def _prior_content(values: dict[str, list[str]] | None = None) -> dict[str, object]:
    source = values or {
        field_name: [hashlib.sha256(field_name.encode()).hexdigest()]
        for field_name in builder.PRIOR_CONTENT_EXCLUSION_FIELDS
    }
    return {
        field_name: {
            "values": rows,
            "count": len(rows),
            "sha256": builder._canonical_sha256_values_digest(
                rows, field_name=field_name
            ),
        }
        for field_name, rows in source.items()
    }


def _write_prior_exclusion_fixture(
    tmp_path: Path,
    freeze: dict[str, object],
    *,
    episodes: list[int] | None = None,
) -> dict[str, object]:
    episode_ids = [0] if episodes is None else episodes
    artifacts = []
    for index, role in enumerate(builder.REQUIRED_PRIOR_EPISODE_COVERAGE):
        kinds = (
            (
                "failed_formal_attempt_receipt",
                "failed_attempt_query_reconstruction_receipt",
            )
            if role == "v4_failed_formal_attempts"
            else (None,)
        )
        for kind_index, kind in enumerate(kinds):
            row = {
                "role": role,
                "path": (
                    f"artifacts/prior/{role}-{index}-{kind_index}.json"
                ),
                "sha256": hashlib.sha256(
                    f"{role}:{kind}".encode()
                ).hexdigest(),
                "size_bytes": index + kind_index + 1,
            }
            if kind is not None:
                row["artifact_kind"] = kind
            artifacts.append(row)
    payload = {
        "schema_version": 1,
        "protocol_id": builder.PROTOCOL,
        "recovery_authorization_id": builder.RECOVERY_AUTHORIZATION_ID,
        "receipt_id": builder.RECOVERY_PRIOR_RECEIPT_ID,
        "status": builder.PRIOR_EXCLUSION_STATUS,
        "checks_passed": True,
        "public_test": {
            "access_status": "closed_not_read_not_scored",
            "opened": False,
            "read": False,
            "scored": False,
            "hashed": False,
        },
        "preregistration": {
            "path": str(freeze["prereg"]),
            "sha256": builder.file_sha256(freeze["prereg"]),
            "size_bytes": freeze["prereg"].stat().st_size,
        },
        "freeze_receipt": {
            "path": str(freeze["receipt"]),
            "sha256": builder.file_sha256(freeze["receipt"]),
            "size_bytes": freeze["receipt"].stat().st_size,
        },
        "source_h5": {
            "symbol": builder.SOURCE_SYMBOL,
            "sha256": freeze["payload"]["source_h5"]["sha256"],
            "size_bytes": freeze["payload"]["source_h5"]["size_bytes"],
            "row_count": freeze["payload"]["source_h5"]["row_count"],
            "episode_count": freeze["payload"]["source_h5"]["episode_count"],
        },
        "excluded_source_episodes": episode_ids,
        "excluded_source_episode_count": len(episode_ids),
        "excluded_source_episodes_sha256": (
            builder.excluded_source_episodes_sha256(episode_ids)
        ),
        "coverage": {
            role: True for role in builder.REQUIRED_PRIOR_EPISODE_COVERAGE
        },
        "input_artifacts": artifacts,
        "prior_content_exclusions": _prior_content(),
        "recovery_contract": {
            "scientific_protocol_unchanged_from_v4": True,
            "old_prior_preserved_and_extended": True,
            "failed_attempt_source_action_scene_pair_query_all_excluded": True,
            "failed_attempt_raw_queries_deterministically_reconstructed": True,
            "all_inputs_reverified_unchanged_before_output": True,
            "original_v4_attempt_not_retried_or_overwritten": True,
            "lance_opened_or_written": False,
            "public_test_opened_read_hashed_or_scored": False,
            "rgb_probe_run": False,
            "reference_model_training_or_scoring": False,
        },
        "rgb_probe": {"opened": False, "run": False, "scored": False},
        "reference_model_training_or_scoring": False,
        "reference_model_optimizer_steps": 0,
    }
    path = tmp_path / "prior-episode-exclusion.json"
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return {"path": path, "payload": payload}


def _validate_prior_fixture(
    freeze: dict[str, object],
    prior: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, object]:
    payload = prior["payload"]
    monkeypatch.setattr(
        builder,
        "EXPECTED_RECOVERY_EXCLUDED_SOURCE_EPISODE_COUNT",
        payload["excluded_source_episode_count"],
    )
    monkeypatch.setattr(
        builder,
        "EXPECTED_RECOVERY_EXCLUDED_SOURCE_EPISODES_SHA256",
        payload["excluded_source_episodes_sha256"],
    )
    monkeypatch.setattr(
        builder,
        "EXPECTED_RECOVERY_CONTENT",
        {
            field: {
                "count": entry["count"],
                "sha256": entry["sha256"],
            }
            for field, entry in payload["prior_content_exclusions"].items()
        },
    )
    freeze_audit = builder.validate_freeze_receipt(
        receipt_path=freeze["receipt"],
        prereg_path=freeze["prereg"],
        source_h5=freeze["source"],
        builder_path=freeze["builder"],
        physics_path=freeze["physics"],
    )
    return builder.validate_prior_episode_exclusion_receipt(
        receipt_path=prior["path"],
        prereg_path=freeze["prereg"],
        freeze_receipt_path=freeze["receipt"],
        freeze_receipt_audit=freeze_audit,
    )


def test_prior_exclusion_receipt_binds_freeze_source_coverage_and_digests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    freeze = _write_freeze_receipt_fixture(tmp_path)
    prior = _write_prior_exclusion_fixture(tmp_path, freeze)
    audit = _validate_prior_fixture(freeze, prior, monkeypatch)
    assert audit["checks_passed"]
    assert audit["excluded_source_episodes"] == [0]
    assert audit["excluded_source_episode_count"] == 1
    assert audit["coverage"] == {
        role: True for role in builder.REQUIRED_PRIOR_EPISODE_COVERAGE
    }
    assert audit["read_contract"]["receipt_bytes_read_once"]
    assert audit["read_contract"]["prior_artifacts_opened_by_builder"] is False


def test_prior_exclusion_rejects_self_consistent_but_incomplete_union(
    tmp_path: Path,
) -> None:
    freeze = _write_freeze_receipt_fixture(tmp_path)
    prior = _write_prior_exclusion_fixture(tmp_path, freeze)
    freeze_audit = builder.validate_freeze_receipt(
        receipt_path=freeze["receipt"],
        prereg_path=freeze["prereg"],
        source_h5=freeze["source"],
        builder_path=freeze["builder"],
        physics_path=freeze["physics"],
    )
    with pytest.raises(RuntimeError, match="frozen v4r1 union"):
        builder.validate_prior_episode_exclusion_receipt(
            receipt_path=prior["path"],
            prereg_path=freeze["prereg"],
            freeze_receipt_path=freeze["receipt"],
            freeze_receipt_audit=freeze_audit,
        )


@pytest.mark.parametrize(
    "mutation",
    (
        "protocol",
        "status",
        "public",
        "freeze_hash",
        "source_hash",
        "episode_order",
        "episode_digest",
        "coverage",
        "artifact_hash",
        "content_digest",
    ),
)
def test_prior_exclusion_receipt_rejects_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    freeze = _write_freeze_receipt_fixture(tmp_path)
    prior = _write_prior_exclusion_fixture(tmp_path, freeze)
    payload = prior["payload"]
    if mutation == "protocol":
        payload["protocol_id"] = "wrong"
    elif mutation == "status":
        payload["status"] = "draft"
    elif mutation == "public":
        payload["public_test"]["read"] = True
    elif mutation == "freeze_hash":
        payload["freeze_receipt"]["sha256"] = "0" * 64
    elif mutation == "source_hash":
        payload["source_h5"]["sha256"] = "0" * 64
    elif mutation == "episode_order":
        payload["excluded_source_episodes"] = [1, 0]
        payload["excluded_source_episode_count"] = 2
    elif mutation == "episode_digest":
        payload["excluded_source_episodes_sha256"] = "0" * 64
    elif mutation == "coverage":
        payload["coverage"]["v3_pilots"] = False
    elif mutation == "artifact_hash":
        payload["input_artifacts"][0]["sha256"] = "bad"
    else:
        payload["prior_content_exclusions"]["action_profile_ids"]["sha256"] = (
            "0" * 64
        )
    prior["path"].write_text(
        json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises((RuntimeError, TypeError, ValueError)):
        _validate_prior_fixture(freeze, prior, monkeypatch)


def test_freeze_receipt_gate_binds_current_code_prereg_and_source(
    tmp_path: Path,
) -> None:
    fixture = _write_freeze_receipt_fixture(tmp_path)
    audit = builder.validate_freeze_receipt(
        receipt_path=fixture["receipt"],
        prereg_path=fixture["prereg"],
        source_h5=fixture["source"],
        builder_path=fixture["builder"],
        physics_path=fixture["physics"],
    )
    assert audit["protocol_id"] == builder.PROTOCOL
    assert audit["status"] == builder.FREEZE_STATUS
    assert audit["checks_passed"]
    assert audit["authorized_splits"] == ["train", "loader_validation"]
    assert audit["public_test"]["opened"] is False
    assert audit["source_h5"]["row_count"] == 3
    assert audit["source_h5"]["episode_count"] == 2
    assert audit["source_h5"]["symbol"] == builder.SOURCE_SYMBOL
    assert audit["source_h5"][
        "content_rehashed_by_builder_before_candidate_selection"
    ] is True
    assert audit["source_h5"]["observed_sha256"] == builder.file_sha256(
        fixture["source"]
    )
    assert audit["sha256"] == builder.file_sha256(fixture["receipt"])


def test_freeze_gate_rejects_same_size_source_content_replacement(
    tmp_path: Path,
) -> None:
    fixture = _write_freeze_receipt_fixture(tmp_path)
    original_size = fixture["source"].stat().st_size
    with h5py.File(fixture["source"], "r+") as handle:
        handle["action"][0, 0] = np.float32(1.0)
    assert fixture["source"].stat().st_size == original_size
    with pytest.raises(RuntimeError, match="SHA256 differs"):
        builder.validate_freeze_receipt(
            receipt_path=fixture["receipt"],
            prereg_path=fixture["prereg"],
            source_h5=fixture["source"],
            builder_path=fixture["builder"],
            physics_path=fixture["physics"],
        )


def test_source_is_rehashed_after_local_build_before_publish(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.h5"
    source.write_bytes(b"frozen source bytes")
    identity = {
        "sha256": builder.file_sha256(source),
        "size_bytes": source.stat().st_size,
    }
    audit = builder.verify_source_h5_unchanged_after_build(
        source,
        frozen_source_identity=identity,
    )
    assert audit["passed"]
    assert audit["full_content_rehashed_after_local_build_before_publish"]
    source.write_bytes(b"changed source byte")
    with pytest.raises(RuntimeError, match="changed during"):
        builder.verify_source_h5_unchanged_after_build(
            source,
            frozen_source_identity=identity,
        )


@pytest.mark.parametrize(
    "mutation",
    (
        "protocol",
        "status",
        "checks",
        "splits",
        "public",
        "prereg_hash",
        "builder_hash",
        "physics_hash",
        "source_size",
        "source_rows",
        "source_episodes",
        "source_symbol",
    ),
)
def test_freeze_receipt_gate_rejects_invalid_authorization_or_identity(
    tmp_path: Path,
    mutation: str,
) -> None:
    fixture = _write_freeze_receipt_fixture(tmp_path)
    payload = fixture["payload"]
    if mutation == "protocol":
        payload["protocol_id"] = "wrong"
    elif mutation == "status":
        payload["status"] = "draft"
    elif mutation == "checks":
        payload["checks_passed"] = False
    elif mutation == "splits":
        payload["authorized_splits"].append("validation")
    elif mutation == "public":
        payload["public_test"]["opened"] = True
    elif mutation == "prereg_hash":
        payload["preregistration"]["sha256"] = "0" * 64
    elif mutation == "builder_hash":
        payload["identity"]["v4_builder"]["sha256"] = "0" * 64
    elif mutation == "physics_hash":
        payload["identity"]["v4_physics"]["sha256"] = "0" * 64
    elif mutation == "source_size":
        payload["source_h5"]["size_bytes"] += 1
    elif mutation == "source_rows":
        payload["source_h5"]["row_count"] += 1
    elif mutation == "source_episodes":
        payload["source_h5"]["episode_count"] += 1
    else:
        payload["source_h5"]["symbol"] = "wrong_source"
    fixture["receipt"].write_text(
        json.dumps(payload, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError):
        builder.validate_freeze_receipt(
            receipt_path=fixture["receipt"],
            prereg_path=fixture["prereg"],
            source_h5=fixture["source"],
            builder_path=fixture["builder"],
            physics_path=fixture["physics"],
        )


def test_invalid_freeze_receipt_is_rejected_before_output_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _write_freeze_receipt_fixture(tmp_path)
    prior = _write_prior_exclusion_fixture(tmp_path, fixture)
    payload = fixture["payload"]
    payload["status"] = "draft"
    fixture["receipt"].write_text(
        json.dumps(payload, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "must-not-exist"
    staging = tmp_path / "staging"
    staging.mkdir()
    monkeypatch.setattr(builder, "DEFAULT_OUTPUT", output)
    monkeypatch.setattr(builder, "DEFAULT_PREREG", fixture["prereg"])
    monkeypatch.setattr(builder, "DEFAULT_FREEZE_RECEIPT", fixture["receipt"])
    monkeypatch.setattr(
        builder, "DEFAULT_PRIOR_EXCLUSION_RECEIPT", prior["path"]
    )
    with pytest.raises(RuntimeError, match="status"):
        builder.main(
            [
                "--output",
                str(output),
                "--source",
                str(fixture["source"]),
                "--prereg",
                str(fixture["prereg"]),
                "--freeze-receipt",
                str(fixture["receipt"]),
                "--prior-episode-exclusion-receipt",
                str(prior["path"]),
                "--staging-root",
                str(staging),
            ]
        )
    assert not output.exists()


def test_v4_formal_profile_factory_is_balanced_split_and_preformal_disjoint() -> None:
    anchors = builder._anchor_ids()
    assert builder.FORMAL_CATALOG_INDEX_OFFSET == 2_000_000
    assert builder.FORMAL_CATALOG_INDEX_OFFSET > 1_000_000
    assert builder.FORMAL_CATALOG_INDEX_OFFSET % len(anchors) == 0
    profile_ids: dict[str, set[str]] = {}
    for split in builder.ACTIVE_SPLITS:
        counts = {anchor: 0 for anchor in anchors}
        ids: set[str] = set()
        for local_index in range(8):
            catalog_index = builder._formal_catalog_index(local_index)
            profile = make_v4_action_profile(
                split=split,
                catalog_index=catalog_index,
            )
            receipt = asdict(profile)
            blocks = v4_action_blocks(profile)
            assert receipt["action_anchor_id"] == anchors[catalog_index % 4]
            assert receipt["action_profile_id"] == (
                builder.action_profile_content_sha256(blocks)
            )
            counts[receipt["action_anchor_id"]] += 1
            ids.add(receipt["action_profile_id"])
        assert set(counts.values()) == {2}
        assert len(ids) == 8
        profile_ids[split] = ids
    assert not (profile_ids["train"] & profile_ids["loader_validation"])
    preformal_ids = {
        make_v4_action_profile(
            split="train", catalog_index=preformal_index
        ).action_profile_id
        for preformal_index in (0, 1)
    }
    assert not (set().union(*profile_ids.values()) & preformal_ids)


@pytest.mark.parametrize("invalid", (-1, True, 1.5))
def test_formal_catalog_index_rejects_invalid_local_index(invalid: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        builder._formal_catalog_index(invalid)  # type: ignore[arg-type]


def test_candidate_catalog_excludes_prior_episodes_before_seeded_assignment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.h5"
    with h5py.File(source, "w") as handle:
        handle.create_dataset("qpos", data=np.zeros((18, 21), dtype=np.float64))
        handle.create_dataset("control", data=np.zeros((18, 7), dtype=np.float64))
    monkeypatch.setattr(
        builder,
        "_eligible_source_rows",
        lambda _source: [(index, index, 30) for index in range(18)],
    )
    monkeypatch.setattr(
        builder,
        "_source_h5_receipt",
        lambda *_args, eligible_episode_count, **_kwargs: {
            "source_symbol": builder.SOURCE_SYMBOL,
            "eligible_source_episode_count": eligible_episode_count,
        },
    )
    excluded = [0, 1]
    prior = {
        "checks_passed": True,
        "sha256": "9" * 64,
        "excluded_source_episodes": excluded,
        "excluded_source_episode_count": len(excluded),
        "excluded_source_episodes_sha256": (
            builder.excluded_source_episodes_sha256(excluded)
        ),
        "prior_content_exclusions": _prior_content(),
    }
    catalogs, receipt = builder.build_candidate_catalogs(
        source,
        pair_counts={"train": 4, "loader_validation": 4},
        frozen_source_identity={},
        prior_exclusion_audit=prior,
    )
    selected = {
        candidate.source_episode
        for rows in catalogs.values()
        for candidate in rows
    }
    assert not (selected & set(excluded))
    for split in builder.ACTIVE_SPLITS:
        assert [candidate.catalog_index for candidate in catalogs[split]] == [
            builder.FORMAL_CATALOG_INDEX_OFFSET + local_index
            for local_index in range(8)
        ]
        assert [candidate.candidate_id for candidate in catalogs[split]] == [
            f"cube-carry-v4r1-{split}-{local_index:06d}"
            for local_index in range(8)
        ]
    namespace = receipt["formal_catalog_namespace"]
    assert namespace["catalog_index_offset"] == 2_000_000
    assert namespace["local_index_policy"] == (
        builder.FORMAL_CATALOG_LOCAL_INDEX_POLICY
    )
    assert namespace["offset_positive"]
    assert namespace["offset_modulo_anchor_count"] == 0
    assert namespace["scene_rng_task_and_candidate_id_use_local_index"]
    assert namespace["per_split_ranges"] == {
        split: {
            "local_index_start_inclusive": 0,
            "local_index_stop_exclusive": 8,
            "catalog_index_start_inclusive": 2_000_000,
            "catalog_index_stop_exclusive": 2_000_008,
        }
        for split in builder.ACTIVE_SPLITS
    }
    exclusion = receipt["prior_episode_exclusion"]
    assert exclusion["applied_before_candidate_assignment"]
    assert exclusion["catalog_overlap"] == {
        "source_episode_count": 0,
        "action_profile_id_count": 0,
        "scene_template_content_hash_count": 0,
        "pair_content_hash_count": 0,
    }
    assert receipt["eligible_source_episode_count_before_prior_exclusion"] == 18
    assert receipt["eligible_source_episode_count_removed_by_prior_exclusion"] == 2
    assert receipt["eligible_source_episode_count_after_prior_exclusion"] == 16


def test_balanced_acceptance_enforces_four_exact_quotas_and_unique_profiles() -> None:
    tracker = builder._BalancedAcceptance(pair_count=8)
    anchors = builder._anchor_ids()
    assert tracker.consider(anchor=anchors[0], profile_id="a0")
    assert tracker.consider(anchor=anchors[0], profile_id="a1")
    assert not tracker.consider(anchor=anchors[0], profile_id="a2")
    assert not tracker.consider(anchor=anchors[1], profile_id="a1")
    for anchor_index, anchor in enumerate(anchors[1:], start=1):
        assert tracker.consider(anchor=anchor, profile_id=f"{anchor_index}-0")
        assert tracker.consider(anchor=anchor, profile_id=f"{anchor_index}-1")
    assert tracker.complete
    assert set(tracker.counts.values()) == {2}
    assert tracker.quota_full_candidates == 1
    assert tracker.duplicate_profile_candidates == 1


def test_worker_passes_a_distinct_replay_simulator(monkeypatch: pytest.MonkeyPatch) -> None:
    replay = object()

    class _Primary:
        observed_replay: object | None = None

        def build_pair(self, _candidate: object, *, replay_simulator: object):
            self.observed_replay = replay_simulator
            return None

    primary = _Primary()
    monkeypatch.setattr(builder, "_WORKER_SIMULATOR", primary)
    monkeypatch.setattr(builder, "_WORKER_REPLAY_SIMULATOR", replay)
    candidate = make_v4_candidate(_base_candidate("train", 0))
    assert builder._build_candidate(candidate) is None
    assert primary.observed_replay is replay
    assert primary is not replay


def _cross_split_fixture() -> dict[str, dict[str, object]]:
    anchors = list(builder._anchor_ids())
    return {
        "train": {
            "query_hashes": ["query-train"],
            "source_episodes": [1],
            "action_profile_ids": ["profile-train"],
            "scene_template_content_hashes": ["scene-train"],
            "pair_content_hashes": ["content-pair-train"],
            "pair_ids": ["pair-train"],
            "action_anchor_ids": anchors,
        },
        "loader_validation": {
            "query_hashes": ["query-development"],
            "source_episodes": [2],
            "action_profile_ids": ["profile-development"],
            "scene_template_content_hashes": ["scene-development"],
            "pair_content_hashes": ["content-pair-development"],
            "pair_ids": ["pair-development"],
            "action_anchor_ids": anchors,
        },
    }


def test_cross_split_audit_distinguishes_profiles_from_anchor_families() -> None:
    audit = builder._cross_split_audit(_cross_split_fixture())
    assert audit["passed"]
    assert audit["query_pixel_hash_overlap"]["count"] == 0
    assert audit["source_episode_overlap"]["count"] == 0
    assert audit["exact_action_profile_id_overlap"]["count"] == 0
    assert audit["scene_template_content_hash_overlap"]["count"] == 0
    assert audit["pair_content_hash_overlap"]["count"] == 0
    assert audit["action_anchor_family_overlap"]["count"] == 4
    assert audit["action_anchor_family_overlap"]["expected_count"] == 4
    assert "not exact profiles" in audit["action_anchor_family_overlap"][
        "interpretation"
    ]
    assert audit["pair_id_is_content_isolation_evidence"] is False


@pytest.mark.parametrize(
    ("field", "check"),
    (
        ("query_hashes", "query_pixel_hash_overlap_zero"),
        ("source_episodes", "source_episode_overlap_zero"),
        ("action_profile_ids", "exact_action_profile_id_overlap_zero"),
        (
            "scene_template_content_hashes",
            "scene_template_content_hash_overlap_zero",
        ),
        ("pair_content_hashes", "pair_content_hash_overlap_zero"),
    ),
)
def test_cross_split_audit_rejects_required_zero_overlap(
    field: str,
    check: str,
) -> None:
    reports = _cross_split_fixture()
    reports["loader_validation"][field] = list(reports["train"][field])
    audit = builder._cross_split_audit(reports)
    assert not audit["passed"]
    assert not audit["checks"][check]


def test_cross_split_audit_requires_all_four_shared_anchor_families() -> None:
    reports = _cross_split_fixture()
    reports["loader_validation"]["action_anchor_ids"] = list(
        builder._anchor_ids()[:3]
    )
    audit = builder._cross_split_audit(reports)
    assert not audit["passed"]
    assert not audit["checks"]["four_common_action_anchor_families_expected"]


def test_split_prefixed_pair_id_is_not_used_as_content_isolation_evidence() -> None:
    reports = _cross_split_fixture()
    reports["loader_validation"]["pair_ids"] = reports["train"]["pair_ids"]
    audit = builder._cross_split_audit(reports)
    assert audit["passed"]
    assert audit["pair_id_is_content_isolation_evidence"] is False


def _base_candidate(split: str, catalog_index: int) -> CubeGraspRuleCandidate:
    return CubeGraspRuleCandidate(
        candidate_id=f"fixture-{split}-{catalog_index}",
        split=split,
        catalog_index=catalog_index,
        source_row=10 + catalog_index,
        source_episode=20 + catalog_index,
        source_step=30,
        simulator_seed=40,
        task_id=1,
        qpos=tuple(0.0 for _ in range(21)),
        control=tuple(0.0 for _ in range(7)),
        cube_color=(0.3, 0.4, 0.5),
        target_position=(0.4, 0.0, 0.02),
    )


def _built_result_fixture() -> dict[str, object]:
    candidate = make_v4_candidate(_base_candidate("train", 0))
    candidate_receipt = asdict(candidate)
    profile_receipt = asdict(candidate.action_profile)
    blocks = np.asarray(v4_action_blocks(candidate.action_profile), dtype=np.float32)
    profile_id = builder.action_profile_content_sha256(blocks)
    anchor_id = profile_receipt["action_anchor_id"]
    episodes = {
        mode: {
            "pixels": [b"jpeg"] * 4,
            "action_blocks": blocks.copy(),
            "physical_state": np.zeros((4, 7), dtype=np.float32),
            "hidden_value": float(index),
            "action_anchor_id": anchor_id,
            "action_profile_id": profile_id,
        }
        for index, mode in enumerate(GRASP_MODES)
    }
    replay_modes = {
        mode: {
            "passed": True,
            "checks": {"pixels_bitwise_equal": True},
            "maximum_physical_state_gap": 0.0,
            "maximum_simulator_state_gap": 0.0,
            "changed_rgb_values": 0,
            "changed_pixels": 0,
            "hashes": {
                "continuous": {"pixels": f"{mode}-pixels"},
                "fresh_replay": {"pixels": f"{mode}-pixels"},
            },
        }
        for mode in GRASP_MODES
    }
    return {
        "candidate": candidate_receipt,
        "action_profile": profile_receipt,
        "content_hashes": {
            "scene_template_content_hash": builder.scene_template_content_sha256(
                candidate_receipt
            ),
            "action_profile_id": profile_id,
            "pair_content_hash": builder.pair_content_sha256(
                builder.scene_template_content_sha256(candidate_receipt),
                profile_id,
            ),
        },
        "audit": {
            "passed": True,
            "hashes": {"query_pixels": "query-fixture"},
            "v4": {
                "action_anchor_id": anchor_id,
                "action_profile_id": profile_id,
                "profile_constraints": {
                    "maximum_action_abs": float(np.abs(blocks).max()),
                },
                "fresh_simulator_replay": {
                    "passed": True,
                    "independent_simulator_instance": True,
                    "provided_reusable_instance": True,
                    "maximum_physical_state_gap": 0.0,
                    "maximum_simulator_state_gap": 0.0,
                    "total_changed_rgb_values": 0,
                    "total_changed_pixels": 0,
                    "modes": replay_modes,
                },
            },
        },
        "episodes": episodes,
    }


def test_built_result_recomputes_profile_content_identity() -> None:
    row = builder._validate_built_result(_built_result_fixture(), "train")
    assert row["action_anchor_id"] == builder._anchor_ids()[0]
    assert row["action_profile_id"] == builder.action_profile_content_sha256(
        _built_result_fixture()["episodes"][GRASP_MODES[0]]["action_blocks"]
    )
    assert row["scene_template_content_hash"] == (
        builder.scene_template_content_sha256(row["candidate"])
    )
    assert row["pair_content_hash"] == builder.pair_content_sha256(
        row["scene_template_content_hash"],
        row["action_profile_id"],
    )


def test_built_result_rejects_metadata_only_profile_identity() -> None:
    result = _built_result_fixture()
    result["episodes"][GRASP_MODES[1]]["action_blocks"][0, 0, 0] += np.float32(
        0.125
    )
    with pytest.raises(RuntimeError, match="action blocks differ"):
        builder._validate_built_result(result, "train")


def test_built_result_rejects_tampered_scene_pair_content_receipt() -> None:
    result = _built_result_fixture()
    result["content_hashes"]["pair_content_hash"] = "0" * 64
    with pytest.raises(RuntimeError, match="content-hash receipt"):
        builder._validate_built_result(result, "train")


def test_fresh_replay_summary_uses_pair_replay_audits_not_query_gap() -> None:
    first = builder._validate_built_result(_built_result_fixture(), "train")
    second = builder._validate_built_result(_built_result_fixture(), "train")
    summary = builder._fresh_replay_summary([first, second])
    assert summary["passed"]
    assert summary["pair_count"] == 2
    assert summary["mode_replay_count"] == 4
    assert summary["maximum_physical_state_gap"] == 0.0
    assert summary["maximum_simulator_state_gap"] == 0.0
    assert summary["total_changed_rgb_values"] == 0
    assert summary["query_gap_used_as_replay_substitute"] is False

    build_summary = builder._fresh_replay_build_summary(
        {
            split: {"fresh_simulator_replay": summary}
            for split in builder.ACTIVE_SPLITS
        }
    )
    assert build_summary["passed"]
    assert build_summary["pair_count"] == 4
    assert build_summary["mode_replay_count"] == 8
    assert build_summary["query_gap_used_as_replay_substitute"] is False


def test_causal_solver_gate_uses_fresh_replay_not_query_gap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def _capture(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return dict(kwargs)

    monkeypatch.setattr(builder, "audit_causal_data_contract", _capture)
    reports = {
        split: {
            "maximum_query_simulator_state_gap": 0.0,
            "maximum_state_installations_after_x0": 0,
            "minimum_history_cube_height_gap_m": 0.01,
            "minimum_future_cube_height_gap_m": 0.01,
        }
        for split in builder.ACTIVE_SPLITS
    }
    builder._audit_causal_contract_from_real_replay(
        reports,
        {"passed": False},
    )
    assert captured["maximum_query_state_gap"] == 0.0
    assert captured["solver_cache_check_passed"] is False
    assert any(
        "distinct simulator" in value for value in captured["evidence"]
    )


def test_record_batch_persists_privileged_anchor_and_profile_columns() -> None:
    result = _built_result_fixture()
    row = builder._validate_built_result(result, "train")
    batch = builder._record_batch(
        result["episodes"][GRASP_MODES[0]],
        episode_index=0,
        split="train",
        candidate=row["candidate"],
        mode=GRASP_MODES[0],
        action_anchor_id=row["action_anchor_id"],
        action_profile_id=row["action_profile_id"],
        scene_template_content_hash=row["scene_template_content_hash"],
        pair_content_hash=row["pair_content_hash"],
    )
    assert batch.schema == builder.SCHEMA
    values = batch.to_pydict()
    assert set(values["action_anchor_id"]) == {row["action_anchor_id"]}
    assert set(values["action_profile_id"]) == {row["action_profile_id"]}
    assert set(values["scene_template_content_hash"]) == {
        row["scene_template_content_hash"]
    }
    assert set(values["pair_content_hash"]) == {row["pair_content_hash"]}
    assert set(values["source_step"]) == {row["candidate"]["source_step"]}


def test_request_and_manifest_explicitly_keep_public_closed(
    tmp_path: Path,
) -> None:
    freeze_audit = {
        "path": "artifacts/evaluation/freeze.json",
        "sha256": "b" * 64,
        "size_bytes": 789,
        "status": builder.FREEZE_STATUS,
        "checks_passed": True,
        "source_h5": {"sha256": "a" * 64},
    }
    prior_audit = {
        "path": "artifacts/evaluation/prior-exclusion.json",
        "sha256": "c" * 64,
        "size_bytes": 987,
        "status": builder.PRIOR_EXCLUSION_STATUS,
        "checks_passed": True,
        "excluded_source_episode_count": 11,
        "excluded_source_episodes_sha256": "d" * 64,
        "coverage": {
            role: True for role in builder.REQUIRED_PRIOR_EPISODE_COVERAGE
        },
        "prior_content_exclusions": {
            field_name: {"count": 1, "sha256": "e" * 64}
            for field_name in builder.PRIOR_CONTENT_EXCLUSION_FIELDS
        },
    }
    request = builder._request_payload(
        pair_counts=builder.DEFAULT_PAIR_COUNTS,
        source_receipt={
            "source_symbol": builder.SOURCE_SYMBOL,
            "source_size_bytes": 123,
            "source_row_count": 456,
            "source_episode_count": 12,
            "source_file_sha256": "a" * 64,
            "eligible_row_selection_rule": builder.ELIGIBLE_ROW_SELECTION_RULE,
            "prior_episode_exclusion": {
                "receipt_sha256": "c" * 64,
                "catalog_overlap": {
                    "source_episode_count": 0,
                    "action_profile_id_count": 0,
                    "scene_template_content_hash_count": 0,
                    "pair_content_hash_count": 0,
                },
                "passed": True,
            },
        },
        jpeg_quality=95,
        workers=16,
        output=tmp_path,
        freeze_receipt_audit=freeze_audit,
        prior_exclusion_audit=prior_audit,
    )
    assert request["active_splits"] == ["train", "loader_validation"]
    assert request["pair_counts"] == builder.DEFAULT_PAIR_COUNTS
    assert request["evidence_scope"] == builder.EVIDENCE_SCOPE
    assert request["profile_split_policy"] == builder.PROFILE_SPLIT_POLICY
    assert request["public_test_opened"] is False
    assert request["public_test_generated"] is False
    assert request["freeze_receipt"] == {
        "path": "artifacts/evaluation/freeze.json",
        "sha256": "b" * 64,
        "size_bytes": 789,
        "status": builder.FREEZE_STATUS,
        "checks_passed": True,
    }
    assert request["source_content_sha256"] == "a" * 64
    assert request["source"]["source_symbol"] == builder.SOURCE_SYMBOL
    assert request["prior_episode_exclusion_receipt"][
        "excluded_source_episode_count"
    ] == 11
    assert request["prior_episode_exclusion_receipt"][
        "applied_before_candidate_assignment"
    ]
    assert request["action_profile_contract"][
        "exact_profile_ids_split_disjoint"
    ]
    assert request["action_profile_contract"]["terminal_fourth_block"] == {
        "block_index": 3,
        "shape": [5, 5],
        "dtype": "float32",
        "all_values_exactly_zero": True,
        "role": "format-only terminal block; no transition target",
    }
    reproducibility = request["reproducibility_contract"]
    assert reproducibility["candidate_assignment_seed"] == (
        builder.CANDIDATE_ASSIGNMENT_SEED
    )
    assert reproducibility["catalog_seeds"] == builder.CATALOG_SEEDS
    assert reproducibility["profile_split_seeds"] == {
        split: builder.V4_PROFILE_SPLIT_SEEDS[split]
        for split in builder.ACTIVE_SPLITS
    }
    assert reproducibility["candidate_pool_multiplier"] == (
        builder.CANDIDATE_POOL_MULTIPLIER
    )
    assert reproducibility["formal_catalog_index_offset"] == 2_000_000
    assert reproducibility["formal_catalog_local_index_policy"] == (
        builder.FORMAL_CATALOG_LOCAL_INDEX_POLICY
    )
    assert reproducibility["formal_catalog_index_formula"] == (
        "FORMAL_CATALOG_INDEX_OFFSET + local_index"
    )
    assert reproducibility["scene_rng_task_and_candidate_id_use_local_index"]
    assert reproducibility["eligible_row_selection_rule"] == (
        builder.ELIGIBLE_ROW_SELECTION_RULE
    )
    assert reproducibility["source_h5_identity"] == {
        "size_bytes": 123,
        "row_count": 456,
        "episode_count": 12,
        "file_sha256": "a" * 64,
    }
    assert request["action_profile_contract"][
        "formal_catalog_index_offset"
    ] == 2_000_000
    assert request["action_profile_contract"][
        "formal_catalog_offset_positive_and_four_aligned"
    ]
    assert request["action_profile_contract"][
        "failed_v4_formal_catalog_range_excluded"
    ] == {
        "start_inclusive": 1_000_000,
        "stop_exclusive": 1_002_048,
    }
    assert request["content_identity_contract"][
        "scene_pair_profile_and_query_disjoint_from_failed_v4_attempt"
    ]
    assert request["action_profile_contract"][
        "preformal_catalog_indices_excluded"
    ] == [0, 1]
    content_contract = request["content_identity_contract"]
    assert "split" in content_contract["scene_template_content_hash"][
        "excluded_fields"
    ]
    assert content_contract["pair_id_is_content_isolation_evidence"] is False
    assert request["fresh_simulator_replay_contract"] == {
        "required_for_every_accepted_pair": True,
        "primary_and_replay_simulators_distinct": True,
        "environments_not_shared": True,
        "one_reusable_primary_and_one_reusable_replay_instance_per_worker": True,
        "maximum_physical_state_gap": builder.QUERY_STATE_TOLERANCE,
        "maximum_complete_simulator_state_gap": builder.QUERY_STATE_TOLERANCE,
        "pixels_bitwise_equal": True,
        "actions_bitwise_equal": True,
        "query_gap_may_substitute_for_replay": False,
    }

    (tmp_path / "request.json").write_text("{}\n", encoding="utf-8")
    manifest = builder._manifest_payload(
        tmp_path,
        build_report={"passed": True, "request": request},
    )
    assert manifest["active_splits"] == ["train", "loader_validation"]
    assert manifest["evidence_scope"] == builder.EVIDENCE_SCOPE
    assert manifest["profile_split_policy"] == builder.PROFILE_SPLIT_POLICY
    assert manifest["public_test_opened"] is False
    assert manifest["public_test_generated"] is False
    assert manifest["prior_episode_exclusion_receipt"]["sha256"] == "c" * 64
    assert set(manifest["files"]) == {"request.json"}


class _FakeLanceDataset:
    def __init__(self, row_count: int, *, schema: object = builder.SCHEMA) -> None:
        self.schema = schema
        self._row_count = row_count

    def count_rows(self) -> int:
        return self._row_count


def _staged_publication_fixture(
    tmp_path: Path,
) -> tuple[Path, dict[str, dict[str, object]], dict[str, int]]:
    staged = tmp_path / "local" / "cube-v4"
    staged.mkdir(parents=True)
    for name, payload in (
        ("request.json", {"kind": "request"}),
        ("build_report.json", {"kind": "build", "passed": True}),
        ("manifest.json", {"kind": "manifest"}),
    ):
        (staged / name).write_text(
            json.dumps(payload, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    row_counts = {"train.lance": 32, "loader_validation.lance": 16}
    reports: dict[str, dict[str, object]] = {}
    for split in builder.ACTIVE_SPLITS:
        name = f"{split}.lance"
        table = staged / name
        (table / "data").mkdir(parents=True)
        (table / "data" / "fragment.lance").write_bytes(
            f"payload:{split}".encode("ascii")
        )
        receipts = builder.regular_file_receipts(table)
        reports[split] = {
            "table_path": name,
            "model_rows": row_counts[name],
            "table_files": len(receipts),
            "table_bytes": sum(row["size_bytes"] for row in receipts),
            "table_sha256": builder.directory_sha256(table),
        }
    return staged, reports, row_counts


def _fake_lance_open(
    monkeypatch: pytest.MonkeyPatch,
    row_counts: dict[str, int],
) -> None:
    monkeypatch.setattr(
        builder.lance,
        "dataset",
        lambda path: _FakeLanceDataset(row_counts[Path(path).name]),
    )


def test_regular_file_receipts_are_sorted_and_reject_aliases_and_fifo(
    tmp_path: Path,
) -> None:
    root = tmp_path / "tree"
    (root / "nested").mkdir(parents=True)
    (root / "z").write_bytes(b"z")
    (root / "nested" / "a").write_bytes(b"a")
    receipts = builder.regular_file_receipts(root)
    assert [row["path"] for row in receipts] == ["nested/a", "z"]
    assert all(set(row) == {"path", "size_bytes", "sha256"} for row in receipts)

    alias = root / "alias"
    alias.symlink_to(root / "z")
    with pytest.raises(ValueError, match="symlink or special"):
        builder.regular_file_receipts(root)
    alias.unlink()

    fifo = root / "fifo"
    os.mkfifo(fifo)
    with pytest.raises(ValueError, match="symlink or special"):
        builder.regular_file_receipts(root)


def test_publish_refuses_existing_directory_or_symlink_without_touching_it(
    tmp_path: Path,
) -> None:
    staged, reports, _ = _staged_publication_fixture(tmp_path)
    existing = tmp_path / "existing"
    existing.mkdir()
    sentinel = existing / "sentinel"
    sentinel.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        builder._publish_staged_release(
            staged_root=staged,
            output=existing,
            reports=reports,
        )
    assert sentinel.read_text(encoding="utf-8") == "keep"

    target = tmp_path / "target"
    target.mkdir()
    alias = tmp_path / "alias-output"
    alias.symlink_to(target, target_is_directory=True)
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        builder._publish_staged_release(
            staged_root=staged,
            output=alias,
            reports=reports,
        )
    assert alias.is_symlink()


def test_interrupted_copy_leaves_unmarked_incomplete_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staged, reports, row_counts = _staged_publication_fixture(tmp_path)
    output = tmp_path / "published"
    _fake_lance_open(monkeypatch, row_counts)

    def interrupted_copy(source: Path, destination: Path, **_: object) -> None:
        destination = Path(destination)
        destination.mkdir()
        (destination / "partial").write_bytes(b"partial")
        raise RuntimeError("injected copy interruption")

    monkeypatch.setattr(builder.shutil, "copytree", interrupted_copy)
    with pytest.raises(RuntimeError, match="injected copy interruption"):
        builder._publish_staged_release(
            staged_root=staged,
            output=output,
            reports=reports,
        )
    assert (output / "partial").is_file()
    assert not (output / builder.SUCCESS_MARKER_NAME).exists()


def test_success_marker_fsync_failure_is_removed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staged, reports, row_counts = _staged_publication_fixture(tmp_path)
    output = tmp_path / "published"
    _fake_lance_open(monkeypatch, row_counts)

    def fail_fsync(_descriptor: int) -> None:
        raise OSError(5, "injected fsync failure")

    monkeypatch.setattr(builder.os, "fsync", fail_fsync)
    with pytest.raises(OSError, match="injected fsync failure"):
        builder._publish_staged_release(
            staged_root=staged,
            output=output,
            reports=reports,
        )
    assert output.is_dir()
    assert not (output / builder.SUCCESS_MARKER_NAME).exists()


def test_publish_refuses_nested_public_component_before_copy(
    tmp_path: Path,
) -> None:
    staged, reports, _ = _staged_publication_fixture(tmp_path)
    closed = staged / "archive" / "validation.lance"
    closed.mkdir(parents=True)
    (closed / "closed.bin").write_bytes(b"never hash or publish")
    output = tmp_path / "published"
    with pytest.raises((RuntimeError, ValueError), match="namespace|Public"):
        builder._publish_staged_release(
            staged_root=staged,
            output=output,
            reports=reports,
        )
    assert not output.exists()


def test_publish_detects_copy_tampering_before_success_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staged, reports, row_counts = _staged_publication_fixture(tmp_path)
    output = tmp_path / "published"
    _fake_lance_open(monkeypatch, row_counts)
    real_copy2 = shutil.copy2

    def tampered_copy(source: Path, destination: Path, **kwargs: object) -> str:
        result = real_copy2(source, destination, **kwargs)
        if Path(destination).name == "request.json":
            Path(destination).write_bytes(b"tampered")
        return result

    monkeypatch.setattr(builder.shutil, "copy2", tampered_copy)
    with pytest.raises(RuntimeError, match="path, size, or SHA256"):
        builder._publish_staged_release(
            staged_root=staged,
            output=output,
            reports=reports,
        )
    assert not (output / builder.SUCCESS_MARKER_NAME).exists()


def test_publish_reopens_destination_and_leaves_open_failure_unmarked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staged, reports, row_counts = _staged_publication_fixture(tmp_path)
    output = tmp_path / "published"

    def open_dataset(path: str) -> _FakeLanceDataset:
        value = Path(path)
        if output in value.parents or str(value).startswith("/proc/self/fd/"):
            raise OSError("injected destination Lance open failure")
        return _FakeLanceDataset(row_counts[value.name])

    monkeypatch.setattr(builder.lance, "dataset", open_dataset)
    with pytest.raises(OSError, match="destination Lance open failure"):
        builder._publish_staged_release(
            staged_root=staged,
            output=output,
            reports=reports,
        )
    assert output.is_dir()
    assert not (output / builder.SUCCESS_MARKER_NAME).exists()


def test_verified_publish_binds_receipts_lance_and_metadata_then_marks_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staged, reports, row_counts = _staged_publication_fixture(tmp_path)
    output = tmp_path / "published"
    _fake_lance_open(monkeypatch, row_counts)
    staged_receipts = builder.regular_file_receipts(staged)

    result = builder._publish_staged_release(
        staged_root=staged,
        output=output,
        reports=reports,
    )

    marker_path = output / builder.SUCCESS_MARKER_NAME
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    assert result["checks_passed"] is True
    assert result["sha256"] == builder.file_sha256(marker_path)
    assert marker["status"] == "complete"
    assert marker["checks_passed"] is True
    assert marker["protocol"] == builder.PROTOCOL
    assert (
        marker["recovery_authorization_id"]
        == builder.RECOVERY_AUTHORIZATION_ID
    )
    assert marker["publication"]["dirs_exist_ok"] is False
    assert marker["publication"]["nonempty_directory_rename_used"] is False
    assert marker["publication"]["success_marker_written_last"] is True
    assert marker["file_receipts_without_success_marker"] == staged_receipts
    assert marker["publication"]["file_receipts_sha256"] == (
        builder._canonical_json_sha256(staged_receipts)
    )
    assert set(marker["bound_files"]) == {
        "request.json",
        "build_report.json",
        "manifest.json",
    }
    assert set(marker["lance_tables"]) == set(builder.ACTIVE_SPLITS)
    assert all(
        marker["lance_tables"][split]["row_count"]
        == reports[split]["model_rows"]
        for split in builder.ACTIVE_SPLITS
    )


def test_verified_publish_does_not_depend_on_nonempty_directory_rename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staged, reports, row_counts = _staged_publication_fixture(tmp_path)
    output = tmp_path / "published"
    _fake_lance_open(monkeypatch, row_counts)

    def reject_rename(*_: object, **__: object) -> None:
        raise PermissionError("simulated NFS EPERM")

    monkeypatch.setattr(builder.os, "rename", reject_rename)
    monkeypatch.setattr(builder.os, "replace", reject_rename)
    result = builder._publish_staged_release(
        staged_root=staged,
        output=output,
        reports=reports,
    )
    assert result["checks_passed"] is True
    assert (output / builder.SUCCESS_MARKER_NAME).is_file()


def test_main_builds_under_selected_local_staging_but_records_final_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staging_root = tmp_path / "local-staging"
    staging_root.mkdir()
    output = tmp_path / "nfs-like" / "formal-v4"
    source = tmp_path / "source.h5"
    source.write_bytes(b"fixture")
    captured_roots: list[Path] = []
    captured_request_outputs: list[Path] = []
    # This unit fixture exercises storage plumbing with four pairs.  The real
    # CLI gate is tested separately and only authorizes frozen 2048/256.
    monkeypatch.setattr(
        builder,
        "_validate_frozen_pair_counts",
        builder._validate_pair_counts,
    )
    monkeypatch.setattr(builder, "_validate_frozen_workers", lambda value: int(value))
    monkeypatch.setattr(builder, "DEFAULT_OUTPUT", output)
    prereg = tmp_path / "prereg.yaml"
    freeze = tmp_path / "freeze.json"
    prior = tmp_path / "prior.json"
    prereg.write_text("fixture\n", encoding="utf-8")
    freeze.write_text("{}\n", encoding="utf-8")
    prior.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(builder, "DEFAULT_PREREG", prereg)
    monkeypatch.setattr(builder, "DEFAULT_FREEZE_RECEIPT", freeze)
    monkeypatch.setattr(builder, "DEFAULT_PRIOR_EXCLUSION_RECEIPT", prior)

    freeze_audit = {
        "source_h5": {"sha256": "a" * 64},
        "checks_passed": True,
    }
    prior_audit = {"checks_passed": True}
    monkeypatch.setattr(
        builder,
        "validate_freeze_receipt",
        lambda **_: freeze_audit,
    )
    monkeypatch.setattr(
        builder,
        "validate_prior_episode_exclusion_receipt",
        lambda **_: prior_audit,
    )
    monkeypatch.setattr(
        builder,
        "build_candidate_catalogs",
        lambda *_args, **_kwargs: (
            {split: [] for split in builder.ACTIVE_SPLITS},
            {},
        ),
    )

    def fake_request(**kwargs: object) -> dict[str, object]:
        final_output = Path(kwargs["output"])
        captured_request_outputs.append(final_output)
        return {
            "prior_episode_exclusion_receipt": {
                "sha256": "c" * 64,
                "excluded_source_episode_count": 1,
                "excluded_source_episodes_sha256": "d" * 64,
            },
            "reproducibility_contract": {},
            "action_profile_contract": {},
            "content_identity_contract": {},
            "fresh_simulator_replay_contract": {},
            "resolved_output": builder.portable_contextworld_path(final_output),
        }

    monkeypatch.setattr(builder, "_request_payload", fake_request)

    def fake_build_split(
        root: Path,
        *,
        split: str,
        pair_count: int,
        **_: object,
    ) -> dict[str, object]:
        captured_roots.append(root)
        table = root / f"{split}.lance"
        table.mkdir()
        (table / "fragment").write_bytes(split.encode("ascii"))
        receipts = builder.regular_file_receipts(table)
        return {
            "passed": True,
            "table_path": table.name,
            "model_rows": 8 * pair_count,
            "table_files": len(receipts),
            "table_bytes": sum(row["size_bytes"] for row in receipts),
            "table_sha256": builder.directory_sha256(table),
        }

    monkeypatch.setattr(builder, "build_split", fake_build_split)
    monkeypatch.setattr(
        builder,
        "verify_source_h5_unchanged_after_build",
        lambda *_args, **_kwargs: {"passed": True},
    )
    monkeypatch.setattr(
        builder,
        "_cross_split_audit",
        lambda _reports: {"passed": True},
    )
    monkeypatch.setattr(
        builder,
        "_fresh_replay_build_summary",
        lambda _reports: {"passed": True},
    )
    monkeypatch.setattr(
        builder,
        "_audit_causal_contract_from_real_replay",
        lambda *_args: {"passed": True},
    )
    _fake_lance_open(
        monkeypatch,
        {f"{split}.lance": 32 for split in builder.ACTIVE_SPLITS},
    )

    builder.main(
        [
            "--output",
            str(output),
            "--source",
            str(source),
            "--prereg",
            str(prereg),
            "--freeze-receipt",
            str(freeze),
            "--prior-episode-exclusion-receipt",
            str(prior),
            "--train-pairs",
            "4",
            "--development-pairs",
            "4",
            "--workers",
            "1",
            "--staging-root",
            str(staging_root),
        ]
    )

    assert captured_request_outputs == [output]
    assert len(captured_roots) == len(builder.ACTIVE_SPLITS)
    assert len(set(captured_roots)) == 1
    local_root = captured_roots[0]
    assert local_root != output
    assert local_root.name == output.name
    assert local_root.parent.parent == staging_root
    assert local_root.parent.name.startswith("contextworld-cube-v4-")
    assert not local_root.exists()
    request = json.loads((output / "request.json").read_text(encoding="utf-8"))
    assert request["resolved_output"] == builder.portable_contextworld_path(output)
    assert (output / builder.SUCCESS_MARKER_NAME).is_file()
