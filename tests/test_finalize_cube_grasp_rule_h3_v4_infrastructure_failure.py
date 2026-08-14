from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import pytest
import yaml

import scripts.finalize_cube_grasp_rule_h3_v4_infrastructure_failure as finalizer


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _identity(sha256: str, size_bytes: int, path: str = "artifact") -> dict[str, object]:
    return {"path": path, "sha256": sha256, "size_bytes": size_bytes}


def _public(*, generated: bool = False) -> dict[str, object]:
    value: dict[str, object] = {
        "access_status": "closed_not_read_not_scored",
        "opened": False,
        "read": False,
        "hashed": False,
        "scored": False,
    }
    if generated:
        value["generated"] = False
    return value


def _content_entry(values: list[str], field: str) -> dict[str, object]:
    values = sorted(values)
    return {
        "values": values,
        "count": len(values),
        "sha256": finalizer.content_digest(values, field_name=field),
    }


def _source_entry(values: list[int]) -> dict[str, object]:
    values = sorted(values)
    return {
        "values": values,
        "count": len(values),
        "sha256": finalizer.source_episode_digest(values),
    }


def _fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    prior_overlap: bool = False,
) -> dict[str, Path]:
    monkeypatch.setattr(finalizer, "EXPECTED_PAIR_COUNT", 4)
    monkeypatch.setattr(finalizer, "EXPECTED_EPISODE_COUNT", 8)
    monkeypatch.setattr(finalizer, "EXPECTED_ROW_COUNT", 32)
    monkeypatch.setattr(finalizer, "EXPECTED_CATALOG_START", 100)
    monkeypatch.setattr(finalizer, "EXPECTED_CATALOG_STOP", 104)
    monkeypatch.setattr(
        finalizer,
        "EXPECTED_ANCHOR_COUNTS",
        {"endpoint4": 1, "front_hold": 1, "plateau": 1, "ramp4": 1},
    )
    monkeypatch.setattr(finalizer, "SOURCE_SHA256", "ab" * 32)
    monkeypatch.setattr(finalizer, "SOURCE_SIZE_BYTES", 1234)
    monkeypatch.setattr(finalizer, "SOURCE_ROW_COUNT", 12)
    monkeypatch.setattr(finalizer, "SOURCE_EPISODE_COUNT", 100)

    prereg_document = {
        "schema_version": 1,
        "preregistration_id": finalizer.PREREGISTRATION_ID,
        "protocol_id": finalizer.PROTOCOL,
        "status": finalizer.PREREG_STATUS,
        "phase": "development_only",
        "attempt_budget_and_stop_rules": {
            "v4_builder_lance_smoke_attempts_authorized": 0,
            "formal_build_attempts_authorized": 1,
            "rgb_probe_attempts_authorized": 1,
            "model_training_or_scoring_attempts_authorized": 0,
            "public_test_attempts_authorized": 0,
            "on_formal_build_failure": [
                "write_failed_development_with_exact_failure_stage",
                "do_not_run_rgb_probe_when_its_inputs_are_invalid",
                "do_not_rebuild_under_this_preregistration",
                "keep_public_test_closed",
            ],
        },
        "reference_model_phase": {"training_and_scoring_authorized": False},
        "public_test": _public(),
    }
    current_prereg = tmp_path / "repo/configs/prereg_v4.yaml"
    current_prereg.parent.mkdir(parents=True)
    prereg_bytes = yaml.safe_dump(prereg_document, sort_keys=False).encode("utf-8")
    current_prereg.write_bytes(prereg_bytes)
    prereg_snapshot = tmp_path / "evaluation/snapshots/prereg_v4.yaml"
    prereg_snapshot.parent.mkdir(parents=True)
    prereg_snapshot.write_bytes(prereg_bytes)
    prereg_sha = finalizer.file_sha256(current_prereg)
    monkeypatch.setattr(finalizer, "EXPECTED_PREREG_SHA256", prereg_sha)
    monkeypatch.setattr(finalizer, "EXPECTED_PREREG_SIZE_BYTES", len(prereg_bytes))

    snapshot_specs = {
        "builder": ("builder.py", "EXPECTED_BUILDER_SHA256", "EXPECTED_BUILDER_SIZE_BYTES"),
        "physics": ("physics_v4.py", "EXPECTED_PHYSICS_SHA256", "EXPECTED_PHYSICS_SIZE_BYTES"),
        "probe": ("probe_v4.py", "EXPECTED_PROBE_SHA256", "EXPECTED_PROBE_SIZE_BYTES"),
        "probe_tests": ("probe_v4_tests.py", "EXPECTED_PROBE_TESTS_SHA256", "EXPECTED_PROBE_TESTS_SIZE_BYTES"),
        "action_support": ("action_support_v4.py", "EXPECTED_ACTION_SUPPORT_SHA256", "EXPECTED_ACTION_SUPPORT_SIZE_BYTES"),
        "action_support_tests": ("action_support_v4_tests.py", "EXPECTED_ACTION_SUPPORT_TESTS_SHA256", "EXPECTED_ACTION_SUPPORT_TESTS_SIZE_BYTES"),
    }
    paths: dict[str, Path] = {
        "current_old_prereg": current_prereg,
        "original_prereg_snapshot": prereg_snapshot,
    }
    snapshot_identities: dict[str, dict[str, object]] = {}
    for name, (filename, digest_constant, size_constant) in snapshot_specs.items():
        path = tmp_path / "evaluation/snapshots" / filename
        path.write_text(f"# frozen {name}\n", encoding="utf-8")
        digest = finalizer.file_sha256(path)
        monkeypatch.setattr(finalizer, digest_constant, digest)
        monkeypatch.setattr(finalizer, size_constant, path.stat().st_size)
        paths[f"{name}_snapshot"] = path
        snapshot_identities[name] = _identity(digest, path.stat().st_size)

    source = {
        "symbol": finalizer.SOURCE_SYMBOL,
        "sha256": finalizer.SOURCE_SHA256,
        "size_bytes": finalizer.SOURCE_SIZE_BYTES,
        "row_count": finalizer.SOURCE_ROW_COUNT,
        "episode_count": finalizer.SOURCE_EPISODE_COUNT,
    }
    freeze_identity = {
        "v4_builder": {
            **snapshot_identities["builder"],
            "path": "scripts/build_cube_grasp_rule_h3_v4_data.py",
        },
        "v4_physics": {
            **snapshot_identities["physics"],
            "path": "contextworld/evaluation/cube_grasp_rule_h3_v4.py",
        },
        "v4_probe": {
            **snapshot_identities["probe"],
            "path": "scripts/probe_cube_grasp_rule_h3_v4_rgb_history.py",
        },
        "v4_probe_tests": {
            **snapshot_identities["probe_tests"],
            "path": "tests/test_cube_grasp_rule_h3_v4_rgb_history.py",
        },
        "v4_action_support_audit": {
            **snapshot_identities["action_support"],
            "path": "scripts/audit_cube_grasp_rule_h3_v4_action_support.py",
        },
        "v4_action_support_audit_tests": {
            **snapshot_identities["action_support_tests"],
            "path": "tests/test_cube_grasp_rule_h3_v4_action_support.py",
        },
    }
    freeze = {
        "authorized_splits": ["train", "loader_validation"],
        "checks_passed": True,
        "frozen_at_utc": "2026-08-12T00:00:00Z",
        "frozen_evidence": {},
        "identity": freeze_identity,
        "preregistration": _identity(prereg_sha, len(prereg_bytes)),
        "prior_episode_exclusion_basis": {},
        "protocol_id": finalizer.PROTOCOL,
        "public_test": _public(),
        "reference_model_training_or_scoring_authorized": False,
        "rgb_history_probe": {},
        "schema_version": 1,
        "scientific_change": {},
        "scope": "Training_and_Development_data_and_rgb_probe_only",
        "source_h5": source,
        "status": finalizer.FREEZE_STATUS,
    }
    freeze_path = tmp_path / "evaluation/freeze.json"
    _write_json(freeze_path, freeze)
    freeze_sha = finalizer.file_sha256(freeze_path)
    monkeypatch.setattr(finalizer, "EXPECTED_FREEZE_SHA256", freeze_sha)
    monkeypatch.setattr(finalizer, "EXPECTED_FREEZE_SIZE_BYTES", freeze_path.stat().st_size)
    paths["freeze_receipt"] = freeze_path

    old_source = [0] if prior_overlap else [99]
    old_content = {field: [_sha(f"old-{field}")] for field in finalizer.CONTENT_FIELDS}
    prior = {
        "basis_receipt": {},
        "checks_passed": True,
        "coverage": {},
        "excluded_source_episode_count": len(old_source),
        "excluded_source_episodes": old_source,
        "excluded_source_episodes_sha256": finalizer.source_episode_digest(old_source),
        "formal_build_requirement": "newest receipt",
        "freeze_receipt": _identity(freeze_sha, freeze_path.stat().st_size),
        "input_artifacts": [],
        "preregistration": _identity(prereg_sha, len(prereg_bytes)),
        "prior_content_exclusions": {
            field: _content_entry(values, field) for field, values in old_content.items()
        },
        "protocol_id": finalizer.PROTOCOL,
        "public_test": _public(),
        "receipt_id": "cube_gripper_carry_h3_v4_prior_exclusions_final_v1",
        "reference_model_training_or_scoring": False,
        "schema_version": 1,
        "source_h5": source,
        "status": finalizer.FREEZE_STATUS,
        "v4_preformal_build_report_count": 0,
        "v4_preformal_content_receipt": {},
    }
    prior_path = tmp_path / "evaluation/prior.json"
    _write_json(prior_path, prior)
    prior_sha = finalizer.file_sha256(prior_path)
    monkeypatch.setattr(finalizer, "EXPECTED_PRIOR_SHA256", prior_sha)
    monkeypatch.setattr(finalizer, "EXPECTED_PRIOR_SIZE_BYTES", prior_path.stat().st_size)
    paths["prior_exclusion_receipt"] = prior_path

    failed_root = tmp_path / "synthesis/cube_gripper_carry_rule_h3_development_v4"
    (failed_root / "train.lance/data").mkdir(parents=True)
    (failed_root / "train.lance/_versions").mkdir()
    (failed_root / "train.lance/_transactions").mkdir()
    fragment = failed_root / "train.lance/data" / finalizer.EXPECTED_FRAGMENT_NAME
    fragment.write_bytes(b"orphan-lance-fragment")
    fragment_sha = finalizer.file_sha256(fragment)
    monkeypatch.setattr(finalizer, "EXPECTED_FRAGMENT_SHA256", fragment_sha)
    monkeypatch.setattr(finalizer, "EXPECTED_FRAGMENT_SIZE_BYTES", fragment.stat().st_size)
    request = {
        "protocol": finalizer.PROTOCOL,
        "resolved_output": finalizer.EXPECTED_LOGICAL_FAILED_ROOT,
        "logical_default_output": finalizer.EXPECTED_LOGICAL_FAILED_ROOT,
        "pair_counts": {"train": 4, "loader_validation": 256},
        "active_splits": ["train", "loader_validation"],
        "public_test_opened": False,
        "public_test_generated": False,
        "freeze_receipt": _identity(freeze_sha, freeze_path.stat().st_size),
        "prior_episode_exclusion_receipt": _identity(prior_sha, prior_path.stat().st_size),
    }
    request_path = failed_root / "request.json"
    _write_json(request_path, request)
    request_sha = finalizer.file_sha256(request_path)
    monkeypatch.setattr(finalizer, "EXPECTED_REQUEST_SHA256", request_sha)
    monkeypatch.setattr(finalizer, "EXPECTED_REQUEST_SIZE_BYTES", request_path.stat().st_size)
    paths.update(
        failed_output_root=failed_root,
        request_json=request_path,
        partial_train_fragment=fragment,
    )
    inventory = [
        {
            "path": f"{finalizer.EXPECTED_LOGICAL_FAILED_ROOT}/request.json",
            "type": "regular_file",
            "sha256": request_sha,
            "size_bytes": request_path.stat().st_size,
        },
        {
            "path": f"{finalizer.EXPECTED_LOGICAL_FAILED_ROOT}/train.lance/data/{finalizer.EXPECTED_FRAGMENT_NAME}",
            "type": "regular_file",
            "sha256": fragment_sha,
            "size_bytes": fragment.stat().st_size,
        },
        {
            "path": f"{finalizer.EXPECTED_LOGICAL_FAILED_ROOT}/train.lance/_versions",
            "type": "empty_directory",
            "entry_count": 0,
        },
        {
            "path": f"{finalizer.EXPECTED_LOGICAL_FAILED_ROOT}/train.lance/_transactions",
            "type": "empty_directory",
            "entry_count": 0,
        },
    ]

    anchors = ("endpoint4", "plateau", "ramp4", "front_hold")
    failure_pairs: list[dict[str, object]] = []
    query_pairs: list[dict[str, object]] = []
    failed_source = list(range(4))
    failed_sets = {field: [] for field in finalizer.CONTENT_FIELDS}
    jpeg_values: list[str] = []
    for index in range(4):
        action = _sha(f"action-{index}")
        scene = _sha(f"scene-{index}")
        pair_hash = _sha(f"pair-{index}")
        jpeg = _sha(f"jpeg-{index}")
        raw = _sha(f"raw-{index}")
        base = {
            "pair_id": f"cube-carry-v4-train-{index:06d}",
            "catalog_index": 100 + index,
            "source_row": 1000 + index,
            "source_episode": index,
            "source_step": 10 + index,
            "action_anchor_id": anchors[index],
            "action_profile_id": action,
            "scene_template_content_hash": scene,
            "pair_content_hash": pair_hash,
            "query_jpeg_sha256": jpeg,
        }
        failure_pairs.append(base)
        query_pairs.append({**base, "split": "train", "raw_query_pixel_hash": raw})
        failed_sets["action_profile_ids"].append(action)
        failed_sets["scene_template_content_hashes"].append(scene)
        failed_sets["pair_content_hashes"].append(pair_hash)
        failed_sets["query_pixel_hashes"].append(raw)
        jpeg_values.append(jpeg)
    failed_sets = {field: sorted(values) for field, values in failed_sets.items()}
    jpeg_values = sorted(jpeg_values)
    source_entry = _source_entry(failed_source)
    inspectable_sets = {
        field: _content_entry(failed_sets[field], field) for field in finalizer.CONTENT_FIELDS[:3]
    }
    failed_receipt = {
        "build_passed": False,
        "checks_passed": True,
        "failed_attempt_content": {
            "split": "train",
            "row_count": 32,
            "episode_count": 8,
            "pair_count": 4,
            "catalog_index_start_inclusive": 100,
            "catalog_index_stop_exclusive": 104,
            "action_anchor_counts": finalizer.EXPECTED_ANCHOR_COUNTS,
            "source_episodes": source_entry,
            "prior_content_exclusions": inspectable_sets,
            "query_pixel_hash_status": "pending_deterministic_raw_reconstruction_not_present_in_fragment",
            "query_jpeg_sha256": {
                "values": jpeg_values,
                "count": len(jpeg_values),
                "sha256": finalizer.forensic_jpeg_digest(jpeg_values),
                "digest_namespace": finalizer.FORENSIC_JPEG_NAMESPACE,
                "role": "forensic_binding_only_not_raw_query_pixel_hash",
            },
            "pairs": failure_pairs,
            "profile_constraints": {
                "maximum_abs_sum_p": 0,
                "maximum_abs_final_p": 0,
                "maximum_abs_moment_error": 0,
                "terminal_nonzero_value_count": 0,
                "passed": True,
            },
            "prior_overlap": {},
        },
        "failed_output": {
            "logical_root": finalizer.EXPECTED_LOGICAL_FAILED_ROOT,
            "inventory": inventory,
            "allowed_inventory_only": True,
            "lance_versions_directory_empty": True,
            "lance_transactions_directory_empty": True,
        },
        "failure": {
            "exit_code": 1,
            "stage": "lance_train_commit_atomic_rename",
            "errno_name": "EPERM",
            "errno_number": 1,
            "exception_type": "OSError",
            "persistent_log_present": False,
        },
        "formal_build_attempt_consumed": True,
        "frozen_runtime_dependencies_from_original_freeze": {
            "v4_physics": snapshot_identities["physics"]
        },
        "input_identities": {
            "preregistration": _identity(prereg_sha, len(prereg_bytes)),
            "freeze_receipt": _identity(freeze_sha, freeze_path.stat().st_size),
            "prior_exclusion_receipt": _identity(prior_sha, prior_path.stat().st_size),
            "builder_snapshot": snapshot_identities["builder"],
            "request_json": _identity(request_sha, request_path.stat().st_size),
            "partial_train_fragment": _identity(fragment_sha, fragment.stat().st_size),
            "source_h5": source,
        },
        "protocol_id": finalizer.PROTOCOL,
        "raw_query_reconstruction_requirement": {},
        "receipt_id": finalizer.FAILED_RECEIPT_ID,
        "recovery_policy": {
            "original_v4_preregistration_attempt_budget_exhausted": True,
            "original_failed_tree_must_remain_immutable": True,
            "silent_retry_or_overwrite_forbidden": True,
            "newly_frozen_recovery_preregistration_required": True,
            "failed_source_action_scene_pair_and_reconstructed_raw_query_must_be_excluded": True,
        },
        "retry_authorized_under_original_preregistration": False,
        "schema_version": 1,
        "scope": {
            "public_test": _public(),
            "rgb_probe_run": False,
            "reference_model_training_or_scoring": False,
            "optimizer_steps": 0,
        },
        "stage_completion": {
            "train_generation_accepted_pairs": 4,
            "train_generation_attempted_candidates": 4,
            "train_lance_data_fragment_written": True,
            "train_lance_commit_completed": False,
            "loader_validation_started": False,
            "build_report_written": False,
            "manifest_written": False,
            "scientifically_inspectable_partial_output": True,
        },
        "status": finalizer.FAILED_RECEIPT_STATUS,
    }
    failure_path = tmp_path / "evaluation/failure.json"
    _write_json(failure_path, failed_receipt)
    failure_sha = finalizer.file_sha256(failure_path)
    monkeypatch.setattr(finalizer, "EXPECTED_FAILURE_RECEIPT_SHA256", failure_sha)
    monkeypatch.setattr(finalizer, "EXPECTED_FAILURE_RECEIPT_SIZE_BYTES", failure_path.stat().st_size)
    paths["failed_attempt_receipt"] = failure_path

    query_content_sets = {
        field: _content_entry(failed_sets[field], field) for field in finalizer.CONTENT_FIELDS
    }
    zero_overlap = {
        "source_episode": {"count": 0, "values": []},
        **{field: {"count": 0, "values": []} for field in finalizer.CONTENT_FIELDS},
        "passed": True,
    }
    query_receipt = {
        "checks_passed": True,
        "failed_attempt_content": {
            "split": "train",
            "row_count": 32,
            "episode_count": 8,
            "pair_count": 4,
            "pairs": query_pairs,
            "source_episodes": source_entry,
            "prior_content_exclusions": query_content_sets,
        },
        "failed_attempt_receipt": _identity(
            failure_sha, failure_path.stat().st_size
        ),
        "input_identities": {
            "preregistration": _identity(prereg_sha, len(prereg_bytes)),
            "freeze_receipt": _identity(freeze_sha, freeze_path.stat().st_size),
            "prior_exclusion_receipt": _identity(prior_sha, prior_path.stat().st_size),
            "failed_attempt_receipt": _identity(failure_sha, failure_path.stat().st_size),
            "builder_snapshot": snapshot_identities["builder"],
            "physics_snapshot": snapshot_identities["physics"],
            "request_json": _identity(request_sha, request_path.stat().st_size),
            "partial_train_fragment": _identity(fragment_sha, fragment.stat().st_size),
            "source_h5": source,
        },
        "prior_overlap": zero_overlap,
        "protocol_id": finalizer.PROTOCOL,
        "public_test": _public(),
        "receipt_id": finalizer.QUERY_RECEIPT_ID,
        "reconstruction_contract": {
            "all_inputs_reverified_unchanged_after_replay": True,
            "builder_snapshot_loaded_by_explicit_path": True,
            "dataset_manifest_opened": False,
            "fragment_read_api": "lance.file.LanceFileReader_single_file",
            "jpeg_quality": 95,
            "jpeg_reencoding_bitwise_equal_to_fragment": True,
            "lance_written": False,
            "physics_snapshot_loaded_by_explicit_path": True,
            "query_model_step_idx": 2,
            "raw_query_frame": "pixels[2]_before_JPEG",
            "raw_query_hash": "Cube_array_sha256_dtype_shape_bytes",
            "raw_query_hashes_unique": True,
            "raw_query_prior_overlap_zero": True,
            "reencoded_query_jpegs_match_fragment": True,
            "replayed_mode": "cannot_hold_only",
            "stored_paired_query_jpegs_equal": True,
            "workers": 16,
        },
        "reference_model_optimizer_steps": 0,
        "reference_model_training_or_scoring": False,
        "rgb_probe": {"opened": False, "run": False, "scored": False},
        "schema_version": 1,
        "status": finalizer.QUERY_RECEIPT_STATUS,
    }
    query_path = tmp_path / "evaluation/query.json"
    _write_json(query_path, query_receipt)
    query_sha = finalizer.file_sha256(query_path)
    monkeypatch.setattr(finalizer, "EXPECTED_QUERY_RECEIPT_SHA256", query_sha)
    monkeypatch.setattr(finalizer, "EXPECTED_QUERY_RECEIPT_SIZE_BYTES", query_path.stat().st_size)
    paths["query_reconstruction_receipt"] = query_path
    paths["output"] = tmp_path / "evaluation" / finalizer.OUTPUT_NAME
    return paths


def _run(paths: dict[str, Path]) -> dict[str, object]:
    return finalizer.finalize_infrastructure_failure(**paths)


def test_finalizer_writes_complete_failed_development_decision_exclusively(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    decision = _run(paths)
    assert decision["status"] == "failed_development"
    assert decision["failure_stage"] == "formal_build_lance_train_commit_atomic_rename"
    assert decision["classification"] == "infrastructure_failure_not_scientific_gate_failure"
    assert decision["formal_build"]["train_generation"]["accepted_pairs"] == 4
    assert decision["formal_build"]["train_generation"]["rejected_candidates"] == 0
    assert decision["formal_build"]["train_lance_commit_completed"] is False
    assert decision["failed_output"]["allowed_inventory_only"] is True
    assert decision["prior_overlap"]["passed"] is True
    for field in finalizer.ALL_EXCLUSION_FIELDS:
        assert decision["failed_content_exclusions"][field]["count"] == 4
        assert decision["prior_overlap"][field] == {"values": [], "count": 0}
        assert decision["recovery_exclusion_union"][field]["count"] == 5
    assert decision["rgb_history_probe"]["run"] is False
    assert decision["reference_model_phase"]["optimizer_steps_run"] == 0
    assert decision["public_test"] == _public(generated=True)
    assert paths["output"].is_file()
    with pytest.raises(FileExistsError, match="overwrite"):
        _run(paths)


def test_finalizer_rejects_independently_recomputed_prior_overlap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch, prior_overlap=True)
    with pytest.raises(RuntimeError, match="overlaps prior evidence for source_episodes"):
        _run(paths)


def test_finalizer_rejects_unexpected_partial_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    (paths["failed_output_root"] / "build_report.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="inventory mismatch"):
        _run(paths)


def test_finalizer_rejects_query_receipt_tamper_before_schema_use(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    value = json.loads(paths["query_reconstruction_receipt"].read_text(encoding="utf-8"))
    value["checks_passed"] = False
    _write_json(paths["query_reconstruction_receipt"], value)
    with pytest.raises(RuntimeError, match="(size|SHA256) mismatch"):
        _run(paths)


def test_finalizer_rejects_snapshot_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    real_path = paths["physics_snapshot"].with_name("real_physics.py")
    shutil.move(paths["physics_snapshot"], real_path)
    paths["physics_snapshot"].symlink_to(real_path)
    with pytest.raises(FileNotFoundError, match="symlink"):
        _run(paths)


def test_cli_requires_all_inputs_and_rejects_public_component() -> None:
    with pytest.raises(SystemExit):
        finalizer.parse_args([])
    argv: list[str] = []
    for option in (
        "current-old-prereg", "original-prereg-snapshot", "freeze-receipt",
        "prior-exclusion-receipt", "failed-attempt-receipt", "query-reconstruction-receipt",
        "builder-snapshot", "physics-snapshot", "probe-snapshot", "probe-tests-snapshot",
        "action-support-snapshot", "action-support-tests-snapshot", "failed-output-root",
        "request-json", "partial-train-fragment",
    ):
        argv.extend((f"--{option}", f"safe/{option}"))
    argv.extend(("--output", f"public/{finalizer.OUTPUT_NAME}"))
    with pytest.raises(RuntimeError, match="Public"):
        finalizer.parse_args(argv)
