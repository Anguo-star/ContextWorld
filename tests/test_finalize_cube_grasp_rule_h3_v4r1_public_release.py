from __future__ import annotations

import json
from pathlib import Path

import pytest

from contextworld.benchmarks.cube_grasp_rule_public_contract import (
    PublicAuthorization,
    file_identity,
)
from contextworld.benchmarks.cube_grasp_rule_public_score import (
    BENCHMARK_ID,
    MATRIX_STATUS,
    aggregate_public_results,
)
from contextworld.benchmarks.cube_grasp_rule_icl_score import (
    cube_grasp_rule_prediction_gate,
)
from contextworld.paths import portable_contextworld_path
import scripts.finalize_cube_grasp_rule_h3_v4r1_public_release as finalizer


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _fixture(tmp_path: Path, *, passed_count: int) -> tuple[PublicAuthorization, Path, Path]:
    public_root = tmp_path / "public"
    score_root = tmp_path / "score"
    decision = tmp_path / "decision.json"
    public_success = public_root / "_SUCCESS.json"
    _write_json(public_success, {"status": "public-data-complete"})
    checkpoints = [
        {
            "model_name": f"cube-gripper-carry-seed{seed}",
            "model_family": "lewm",
            "training_seed": seed,
            "training_recipe": "mixed_frozen_image_paired_future_fit_1p00",
            "path": str(tmp_path / f"seed{seed}.pt"),
            "sha256": f"{index + 1}" * 64,
            "size_bytes": 1,
            "model_state_sha256": f"{index + 4}" * 64,
        }
        for index, seed in enumerate((17321, 17322, 17323))
    ]
    prereg = {
        "preregistration_id": "contextworld_cube_gripper_carry_h3_v4r1_public_release_v1",
        "protocol_id": "cube_gripper_carry_rule_history3_v4r1_public_release_v1",
        "identity": {"preregistration_path": str(tmp_path / "prereg.yaml")},
        "planned_artifacts": {
            "public_data_root": str(public_root),
            "public_score_root": str(score_root),
            "public_release_decision": str(decision),
        },
        "public_evaluation": {
            "checkpoints": checkpoints,
            "devices": ["cuda:0", "cuda:1", "cuda:2"],
            "batch_size": 64,
        },
        "scoring": {
            "hidden_future_prediction": {
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
                    "lower_bound_minimum": {
                        "correct_future_rate": 0.70,
                        "correct_history_rate": 0.70,
                        "context_switch_rate": 0.85,
                    }
                },
            }
        },
        "authorization_basis": {
            "reference_development_decision": {"path": "development.json", "sha256": "1" * 64, "size_bytes": 1},
            "original_task_retention_decision": {"path": "retention.json", "sha256": "2" * 64, "size_bytes": 1},
        },
    }
    prereg_path = tmp_path / "prereg.yaml"
    prereg_path.write_text("frozen: true\n", encoding="utf-8")
    freeze_path = tmp_path / "freeze.json"
    freeze_path.write_text('{"frozen": true}\n', encoding="utf-8")
    authorization = PublicAuthorization(
        preregistration_path=prereg_path,
        freeze_receipt_path=freeze_path,
        preregistration=prereg,
        freeze_receipt={},
        freeze_receipt_identity=file_identity(
            freeze_path, logical_path=str(freeze_path)
        ),
    )
    checkpoint_results = []
    for index, checkpoint in enumerate(checkpoints):
        passed = index < passed_count
        rate = 1.0 if passed else 0.0
        metrics = {
            "pair_count": 256,
            "decision_count": 512,
            "correct_future_rate": rate,
            "correct_history_rate": rate,
            "context_switch_rate": rate,
            "worst_rule_correct_future_rate": rate,
            "other_minus_correct_mse_margin_mean": 1.0 if passed else -1.0,
            "joint_icl_pair_success_rate": rate,
            "paired_bootstrap_95_lower_bound": {
                "correct_future_rate": rate,
                "correct_history_rate": rate,
                "context_switch_rate": rate,
            },
            "latent_response": {
                "response_gain": 1.0 if passed else 0.0,
                "normalized_response_error": 0.0 if passed else 1.0,
                "target_latent_separation": {"passed": True},
            },
        }
        checkpoint_results.append(
            {
                "schema_version": 1,
                "benchmark": BENCHMARK_ID,
                "submission_kind": "fixed_public_checkpoint",
                "status": "completed",
                "preregistration_id": prereg["preregistration_id"],
                "freeze_receipt_sha256": authorization.freeze_receipt_identity[
                    "sha256"
                ],
                "model": {
                    "name": checkpoint["model_name"],
                    "training_seed": checkpoint["training_seed"],
                    "family": "lewm",
                    "training_recipe": checkpoint["training_recipe"],
                    "checkpoint_path": checkpoint["path"],
                    "checkpoint_sha256": checkpoint["sha256"],
                    "checkpoint_size_bytes": checkpoint["size_bytes"],
                    "adapter": {"adapter_id": "fixture"},
                    "state_sha256_before": checkpoint["model_state_sha256"],
                    "state_sha256_after": checkpoint["model_state_sha256"],
                },
                "data": {
                    "split": "Public Test",
                    "lance_table": "validation.lance",
                    "pair_count": 256,
                    "condition_count": 512,
                    "online_environment_calls": 0,
                    "model_visible_fields": [
                        "history_pixels",
                        "query_action_blocks",
                    ],
                    "privileged_columns_passed_to_model": False,
                },
                "metrics": metrics,
                "gate": cube_grasp_rule_prediction_gate(metrics, release=prereg),
            }
        )
    request_path = score_root / "matrix_request.json"
    access_path = score_root / "public_access_started.json"
    request = {
        "schema_version": 1,
        "preregistration": file_identity(
            prereg_path, logical_path=str(prereg_path)
        ),
        "freeze_receipt": authorization.freeze_receipt_identity,
        "public_success_marker": file_identity(
            public_success,
            logical_path=portable_contextworld_path(public_success),
        ),
        "checkpoints": checkpoints,
        "devices": ["cuda:0", "cuda:1", "cuda:2"],
        "batch_size": 64,
        "adapter_runtime_preflight": [
            {
                "training_seed": checkpoint["training_seed"],
                "device": device,
                "model_state_sha256": checkpoint["model_state_sha256"],
                "adapter_id": "fixture-adapter",
                "runtime_preflight_passed": True,
            }
            for checkpoint, device in zip(
                checkpoints, ("cuda:0", "cuda:1", "cuda:2")
            )
        ],
        "model_visible_fields": ["history_pixels", "query_action_blocks"],
        "privileged_columns_passed_to_model": False,
        "training_or_checkpoint_selection": False,
        "threshold_or_recipe_changes": False,
        "rerun_authorized": False,
        "created_at_utc": "2026-08-14T00:00:00+00:00",
    }
    _write_json(request_path, request)
    request_identity = file_identity(
        request_path, logical_path=portable_contextworld_path(request_path)
    )
    _write_json(
        access_path,
        {
            "schema_version": 1,
            "status": "public_access_started_irreversible_one_use_campaign",
            "started_at_utc": "2026-08-14T00:00:01+00:00",
            "request": request_identity,
            "public_data_metadata_preflight_passed": True,
            "checkpoint_identity_preflight_passed": True,
            "adapter_runtime_preflight_passed": True,
        },
    )
    matrix = aggregate_public_results(
        checkpoint_results, authorization=authorization
    )
    matrix["authorization"] = {
        "matrix_request": request_identity,
        "public_access_started": file_identity(
            access_path, logical_path=portable_contextworld_path(access_path)
        ),
    }
    matrix_path = score_root / "matrix_score.json"
    _write_json(matrix_path, matrix)
    result_entries = []
    for result in checkpoint_results:
        result_path = score_root / f"lewm_seed{result['model']['training_seed']}.json"
        _write_json(result_path, result)
        result_entries.append(
            file_identity(
                result_path, logical_path=portable_contextworld_path(result_path)
            )
        )
    _write_json(
        score_root / "_SUCCESS.json",
        {
            "schema_version": 1,
            "status": MATRIX_STATUS,
            "completed_at_utc": "2026-08-14T00:00:02+00:00",
            "rerun_authorized": False,
            "matrix_score": file_identity(
                matrix_path, logical_path=portable_contextworld_path(matrix_path)
            ),
            "checkpoint_results": result_entries,
            "public_test": {
                "generated": True,
                "hashed": True,
                "opened": True,
                "read": True,
                "scored": True,
                "used_for_training_or_selection": False,
            },
        },
    )
    return authorization, score_root, decision


@pytest.mark.parametrize("passed_count", [3, 2])
def test_finalizer_preserves_positive_and_negative_public_branches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    passed_count: int,
) -> None:
    authorization, score_root, output = _fixture(
        tmp_path, passed_count=passed_count
    )
    monkeypatch.setattr(finalizer, "load_public_authorization", lambda **_: authorization)
    monkeypatch.setattr(
        finalizer,
        "validate_public_publication",
        lambda _: {
            "success": {
                "status": "public_data_generated_and_integrity_validated_not_model_read_or_scored"
            },
            "build_report": {"pair_count": 256, "split": "validation"},
        },
    )
    result = finalizer.finalize_public_release(
        preregistration=authorization.preregistration_path,
        freeze_receipt=authorization.freeze_receipt_path,
        score_root=score_root,
        output=output,
    )
    assert result["public_evaluation"]["checkpoints_passed"] == passed_count
    assert result["claims"]["suite_registration_allowed"] is False
    assert result["claims"]["public_test_rerun_allowed"] is False
    assert (
        result["claims"]["positive_reference_public_claim_allowed"]
        is (passed_count == 3)
    )
    assert result["claims"]["local_data_and_scoring_release_packaging_allowed"] is True
    assert output.is_file()
