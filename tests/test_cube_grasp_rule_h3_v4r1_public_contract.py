from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from contextworld.benchmarks.cube_grasp_rule_public_contract import (
    EXPECTED_AUTHORIZATION_BASIS_KEYS,
    EXPECTED_ACTION_MEAN,
    EXPECTED_ACTION_STD,
    EXPECTED_CONTENT_EXCLUSION_FIELDS,
    EXPECTED_IMPLEMENTATION_KEYS,
    EXPECTED_DATA_ACCESS_CONTRACT,
    EXPECTED_RUNTIME_FILE_KEYS,
    EXPECTED_TRAINING_RECIPE,
    EXPECTED_TRAINING_SEEDS,
    FREEZE_RECEIPT_ID,
    FREEZE_STATUS,
    PREREGISTRATION_ID,
    PREREGISTRATION_STATUS,
    PROTOCOL_ID,
    file_identity,
    load_public_authorization,
    _content_digest,
    _source_episode_digest,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _authorization_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    public_root = tmp_path / "public_data"
    score_root = tmp_path / "public_score"
    decision_path = tmp_path / "decision.json"
    freeze_path = tmp_path / "freeze.json"
    prereg_path = tmp_path / "prereg.yaml"
    implementations = {}
    for name in sorted(EXPECTED_IMPLEMENTATION_KEYS):
        path = tmp_path / "implementations" / f"{name}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{name}\n", encoding="utf-8")
        implementations[name] = file_identity(path, logical_path=str(path))
    basis = {}
    for name in sorted(EXPECTED_AUTHORIZATION_BASIS_KEYS):
        path = tmp_path / f"basis-{name}.json"
        path.write_text(json.dumps({"name": name}) + "\n", encoding="utf-8")
        basis[name] = file_identity(path, logical_path=str(path))
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    runtime_files = {}
    for name in sorted(EXPECTED_RUNTIME_FILE_KEYS):
        relative = Path("files") / f"{name}.py"
        path = runtime_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{name}\n", encoding="utf-8")
        identity = file_identity(path, logical_path=str(path))
        runtime_files[name] = {
            "path": relative.as_posix(),
            "sha256": identity["sha256"],
            "size_bytes": identity["size_bytes"],
        }
    checkpoints = []
    for index, seed in enumerate(EXPECTED_TRAINING_SEEDS):
        path = tmp_path / f"seed{seed}.pt"
        path.write_bytes(bytes([index + 1]))
        identity = file_identity(path, logical_path=str(path))
        checkpoints.append(
            {
                "model_name": f"cube-seed-{seed}",
                "model_family": "lewm",
                "training_recipe": EXPECTED_TRAINING_RECIPE,
                "training_seed": seed,
                "checkpoint_step": 4096,
                **identity,
                "model_state_sha256": f"{index + 4}" * 64,
            }
        )
    source = tmp_path / "source.h5"
    source.write_bytes(b"source")
    source_identity = file_identity(source, logical_path=str(source))
    source_values = [7]
    content_values = {name: [f"{index + 1}" * 64] for index, name in enumerate(sorted(EXPECTED_CONTENT_EXCLUSION_FIELDS))}
    exclusion_union = {
        "source_episodes": {
            "count": 1,
            "sha256": _source_episode_digest(source_values),
        },
        **{
            name: {
                "count": 1,
                "sha256": _content_digest(values, field_name=name),
            }
            for name, values in content_values.items()
        },
    }
    prereg = {
        "schema_version": 1,
        "preregistration_id": PREREGISTRATION_ID,
        "protocol_id": PROTOCOL_ID,
        "status": PREREGISTRATION_STATUS,
        "phase": "public_generation_and_evaluation_only",
        "scope": {
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
        },
        "identity": {
            "preregistration_path": str(prereg_path),
            "implementation": implementations,
        },
        "authorization_basis": basis,
        "runtime": {
            "stable_worldmodel": {
                "repo": str(runtime_root),
                "expected_ref": "8" * 40,
                "required_files": runtime_files,
            }
        },
        "public_data_generation": {
            "split": "validation",
            "public_split_name": "Public Test",
            "pair_count": 256,
            "candidate_pool_count": 512,
            "catalog_index_offset": 3_000_000,
            "candidate_assignment_seed": 2026081400,
            "catalog_seed": 2026081401,
            "profile_seed": 2026081402,
            "action_templates": [
                "endpoint4",
                "front_hold",
                "plateau",
                "ramp4",
            ],
            "pair_balanced": True,
            "split_disjoint_from_all_non_public_content": True,
            "action_profile_sum_zero": True,
            "action_profile_last_zero": True,
            "workers": 16,
            "jpeg_quality": 95,
            "staging_root": "/tmp",
            "source_h5": {
                **source_identity,
                "row_count": 10,
                "episode_count": 2,
                "action_dim": 5,
            },
            "exclusion_union": exclusion_union,
        },
        "public_evaluation": {
            "authorized_model_families": ["lewm"],
            "excluded_model_families": {"pldm": "failed_development_0_of_3"},
            "training_authorized": False,
            "checkpoint_or_recipe_selection_after_freeze": False,
            "public_data_loaded_once_for_all_checkpoints": True,
            "devices": ["cuda:0", "cuda:1", "cuda:2"],
            "batch_size": 64,
            "online_environment_calls": 0,
            "data_access_contract": EXPECTED_DATA_ACCESS_CONTRACT,
            "action_normalization": {
                "source": "original_cube_h5_finite_actions_population_zscore",
                "finite_action_rows": 2_000_000,
                "excluded_nonfinite_rows": 10_000,
                "mean": list(EXPECTED_ACTION_MEAN),
                "std_population": list(EXPECTED_ACTION_STD),
            },
            "checkpoints": checkpoints,
        },
        "scoring": {
            "hidden_future_prediction": {
                "target": "each_checkpoint_native_frozen_encoder",
                "cross_checkpoint_absolute_mse_comparison_allowed": False,
                "gates": {
                    "correct_future_rate_minimum": 0.75,
                    "correct_history_rate_minimum": 0.75,
                    "context_switch_rate_minimum": 0.90,
                    "worst_rule_correct_future_rate_minimum": 0.70,
                    "target_latent_separation_required": True,
                    "response_gain_minimum": 0.50,
                    "normalized_response_error_strict_maximum": 1.00,
                },
                "uncertainty": {
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
                },
            }
        },
        "one_use_policy": {
            "generation_attempts_authorized": 1,
            "scoring_attempts_authorized_after_successful_generation": 1,
            "access_marker_written_before_public_table_read": True,
            "retry_after_access_authorized": False,
            "new_preregistration_and_namespace_required_after_failure": True,
        },
        "planned_artifacts": {
            "freeze_receipt": str(freeze_path),
            "public_data_root": str(public_root),
            "public_score_root": str(score_root),
            "public_release_decision": str(decision_path),
        },
        "public_test_before_freeze": {
            "access_status": "closed_not_read_not_scored",
            "generated": False,
            "opened": False,
            "read": False,
            "hashed": False,
            "scored": False,
            "validation_lance_access_allowed": False,
        },
    }
    prereg_path.write_text(
        yaml.safe_dump(prereg, sort_keys=False), encoding="utf-8"
    )
    prereg_raw = prereg_path.read_bytes()
    freeze = {
        "schema_version": 1,
        "receipt_id": FREEZE_RECEIPT_ID,
        "receipt_path": str(freeze_path),
        "preregistration_id": PREREGISTRATION_ID,
        "protocol_id": PROTOCOL_ID,
        "status": FREEZE_STATUS,
        "checks_passed": True,
        "frozen_at_utc": "2026-08-14T00:00:00+00:00",
        "preregistration": {
            "path": str(prereg_path),
            "sha256": hashlib.sha256(prereg_raw).hexdigest(),
            "size_bytes": len(prereg_raw),
        },
        "implementation_identities": implementations,
        "frozen_inputs": {},
        "runtime": {
            "stable_worldmodel": {
                "path": str(runtime_root),
                "commit": "8" * 40,
                "clean_worktree": True,
                "required_files": {},
            }
        },
        "public_exclusions": {
            "checks_passed": True,
            "coverage": {
                "historical_prior_receipt": True,
                "v4r1_train": True,
                "v4r1_loader_validation": True,
                "public_content_included": False,
            },
            "excluded_source_episode_count": 1,
            "excluded_source_episodes_sha256": _source_episode_digest(source_values),
            "excluded_source_episodes": source_values,
            "prior_content_exclusions": {
                name: {
                    "count": 1,
                    "sha256": _content_digest(values, field_name=name),
                    "values": values,
                }
                for name, values in content_values.items()
            },
        },
        "authorization": {
            "public_generation_once": True,
            "public_scoring_once_after_successful_generation": True,
            "authorized_model_families": ["lewm"],
            "training_seeds": list(EXPECTED_TRAINING_SEEDS),
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
        "planned_artifacts": prereg["planned_artifacts"],
    }
    for name, identity in basis.items():
        freeze["frozen_inputs"][name] = {
            **identity,
            "rehash_on_entrypoint": True,
        }
    for checkpoint in checkpoints:
        freeze["frozen_inputs"][f"lewm_checkpoint_seed{checkpoint['training_seed']}"] = {
            **{name: checkpoint[name] for name in ("path", "sha256", "size_bytes")},
            "model_state_sha256": checkpoint["model_state_sha256"],
            "rehash_on_entrypoint": True,
        }
    for name, specification in runtime_files.items():
        identity = {
            "path": str(runtime_root / specification["path"]),
            "sha256": specification["sha256"],
            "size_bytes": specification["size_bytes"],
            "rehash_on_entrypoint": True,
        }
        freeze["frozen_inputs"][f"stable_worldmodel_{name}"] = identity
        freeze["runtime"]["stable_worldmodel"]["required_files"][name] = identity
    freeze["frozen_inputs"]["source_h5"] = {
        **source_identity,
        "row_count": 10,
        "episode_count": 2,
        "action_dim": 5,
        "content_rehash_deferred_to_public_builder_before_candidate_selection": True,
        "rehash_on_entrypoint": False,
    }
    _write_json(freeze_path, freeze)
    return prereg_path, freeze_path, public_root, score_root


def test_authorization_uses_external_freeze_identity_without_self_hash(
    tmp_path: Path,
) -> None:
    prereg, freeze, public_root, score_root = _authorization_fixture(tmp_path)
    authorization = load_public_authorization(
        preregistration_path=prereg,
        freeze_receipt_path=freeze,
        require_public_absent=True,
    )
    assert "receipt_identity" not in authorization.freeze_receipt
    assert authorization.freeze_receipt_identity == file_identity(
        freeze, logical_path=str(freeze)
    )
    assert authorization.public_root == public_root
    assert authorization.score_root == score_root


def test_authorization_is_one_use_when_public_root_exists(tmp_path: Path) -> None:
    prereg, freeze, public_root, _ = _authorization_fixture(tmp_path)
    public_root.mkdir()
    with pytest.raises(FileExistsError, match="cannot be reused"):
        load_public_authorization(
            preregistration_path=prereg,
            freeze_receipt_path=freeze,
            require_public_absent=True,
        )


def test_authorization_rejects_frozen_input_drift(tmp_path: Path) -> None:
    prereg, freeze, _, _ = _authorization_fixture(tmp_path)
    frozen_input = tmp_path / "basis-data_readiness_decision.json"
    frozen_input.write_text('{"passed": false}\n', encoding="utf-8")
    with pytest.raises(
        RuntimeError,
        match="frozen_inputs.data_readiness_decision identity mismatch",
    ):
        load_public_authorization(
            preregistration_path=prereg,
            freeze_receipt_path=freeze,
        )


def test_authorization_rejects_incomplete_receipt_key_set(tmp_path: Path) -> None:
    prereg, freeze, _, _ = _authorization_fixture(tmp_path)
    payload = json.loads(freeze.read_text(encoding="utf-8"))
    payload["implementation_identities"].pop("public_builder")
    _write_json(freeze, payload)
    with pytest.raises(RuntimeError, match="implementation_identities key set mismatch"):
        load_public_authorization(
            preregistration_path=prereg,
            freeze_receipt_path=freeze,
        )


def test_file_identity_rejects_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    alias = tmp_path / "alias.txt"
    target.write_text("frozen\n", encoding="utf-8")
    alias.symlink_to(target)
    with pytest.raises(ValueError, match="non-symlink"):
        file_identity(alias)


def test_authorization_rejects_symlinked_preregistration(tmp_path: Path) -> None:
    prereg, freeze, _, _ = _authorization_fixture(tmp_path)
    alias = tmp_path / "prereg-alias.yaml"
    alias.symlink_to(prereg)
    with pytest.raises(ValueError, match="non-symlink"):
        load_public_authorization(
            preregistration_path=alias,
            freeze_receipt_path=freeze,
        )
