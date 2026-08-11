from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from contextworld.benchmarks.motion_damping_icl_data import (
    MOTION_DAMPING_RELEASE_ID,
    audit_motion_damping_icl_release,
    load_motion_damping_icl_release,
)
from contextworld.benchmarks.motion_damping_icl_score import (
    motion_damping_prediction_gate,
    motion_damping_prediction_metrics,
)


def test_release_name_and_public_splits_are_explicit() -> None:
    release = load_motion_damping_icl_release()
    assert release["release_id"] == MOTION_DAMPING_RELEASE_ID
    assert release["scope"]["display_name_zh"] == "PushT 运动阻尼 ICL"
    assert release["scope"]["history_tokens"] == 3
    assert release["scope"]["damping_values"] == [0.2, 1.0]
    assert release["scope"]["public_test_included"] is True
    assert release["scope"]["sealed_test_included"] is False
    assert release["data"]["pair_counts"] == {
        "train": 8192,
        "loader_validation": 256,
        "validation": 256,
    }
    matrix = release["training"]["reference_matrix"]
    assert matrix["status"] == "failed_development"
    assert matrix["completed_development_seeds"] == [14321]
    assert matrix["remaining_seeds_run"] is False
    assert matrix["public_model_scoring_opened"] is False
    assert matrix["reported_endpoint"] == {
        "model_family": "LeWM",
        "recipe": "mixed_frozen_image_paired_future_ranking_twin_1p00",
        "training_seed": 14321,
        "optimizer_step": 8192,
    }
    assert release["scoring"]["hidden_future_prediction"]["gates"] == {
        "correct_future_rate_minimum": 0.95,
        "correct_history_rate_minimum": 0.95,
        "context_switch_rate_minimum": 0.95,
        "worst_damping_correct_future_rate_minimum": 0.90,
        "target_latent_separation_required": True,
        "response_gain_minimum": 0.50,
        "normalized_response_error_strict_maximum": 1.00,
    }


def test_release_data_and_public_test_are_auditable() -> None:
    audit = audit_motion_damping_icl_release(full=False)
    assert audit["passed"]
    assert audit["causal_data_contract"]["passed"]
    assert audit["causal_data_contract"]["x0_policy"] == (
        "balanced_visible_start"
    )
    total_pairs = sum(audit["counts"].values())
    assert audit["causal_data_contract"]["evidence_scope"] == (
        f"all_{total_pairs}_pairs_and_{2 * total_pairs}_clean_replays"
    )
    assert audit["row_counts"] == {
        "train": 327680,
        "loader_validation": 10240,
        "validation": 10240,
    }
    assert audit["data_checks"][
        "frozen_evaluation_split_receipts_passed"
    ]
    assert audit["data_checks"][
        "frozen_evaluation_table_hashes_preserved"
    ]
    assert audit["reference_result"]["passed"]
    assert audit["reference_result"]["status"] == "failed_development"
    assert audit["reference_result"]["failed_metrics"] == [
        "correct_history_rate"
    ]
    assert audit["reference_result"]["public_model_scoring_opened"] is False
    assert audit["reference_result"]["positive_reference_claim"] is False


def test_compact_negative_reference_decision_is_public_safe() -> None:
    release = load_motion_damping_icl_release()
    specification = release["reference_results"]["current_decision"]
    path = Path(__file__).resolve().parents[1] / specification["path"]
    decision = json.loads(path.read_text(encoding="utf-8"))
    serialized = json.dumps(decision, sort_keys=True)
    assert "/opt/" not in serialized
    assert "candidates" not in serialized
    assert "attempt" not in serialized
    assert decision["status"] == "failed_development"
    assert decision["data_release"]["published_manifest_sha256"] == release[
        "data"
    ]["manifest_sha256"]
    receipt_path = (
        Path(__file__).resolve().parents[1]
        / release["data"]["artifacts"]["portability_receipt"]["path"]
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert decision["data_release"]["training_manifest_sha256"] == receipt[
        "metadata_sha256"
    ]["manifest.json"]["before"]
    assert decision["data_release"]["portability_receipt_sha256"] == release[
        "data"
    ]["artifacts"]["portability_receipt"]["sha256"]
    endpoint = decision["reported_endpoint"]
    assert endpoint["metrics"] == {
        "correct_future_rate": 0.974609375,
        "correct_history_rate": 0.529296875,
        "context_switch_rate": 0.9765625,
        "worst_damping_correct_future_rate": 0.97265625,
    }
    assert endpoint["failed_metrics"] == ["correct_history_rate"]
    assert endpoint["passed"] is False
    assert decision["public_model_scoring_opened"] is False
    assert decision["additional_training_seeds_run"] is False
    assert decision["original_task_cem_run"] is False
    assert decision["positive_reference_claim"] is False
    legacy_gate = motion_damping_prediction_gate(
        endpoint["metrics"], release=load_motion_damping_icl_release()
    )
    assert legacy_gate["passed"] is False
    assert legacy_gate["checks"]["target_latent_separation"] is False


def test_prediction_metrics_compare_matching_real_futures() -> None:
    faster = np.asarray([[0.0, 0.0], [1.0, 1.0]])
    no_extra = np.asarray([[2.0, 2.0], [3.0, 3.0]])
    metrics, records = motion_damping_prediction_metrics(
        pair_ids=("pair-0", "pair-1"),
        predicted_faster_decay=faster,
        predicted_no_extra_decay=no_extra,
        target_faster_decay=faster,
        target_no_extra_decay=no_extra,
    )
    assert metrics["correct_future_rate"] == 1.0
    assert metrics["correct_history_rate"] == 1.0
    assert metrics["context_switch_rate"] == 1.0
    assert metrics["worst_damping_correct_future_rate"] == 1.0
    assert metrics["joint_icl_pair_success_rate"] == 1.0
    assert len(records) == 2
    gate = motion_damping_prediction_gate(
        metrics, release=load_motion_damping_icl_release()
    )
    assert gate["passed"]
    assert gate["checks"]["target_latent_separation"]
    assert gate["checks"]["response_gain"]
    assert gate["checks"]["normalized_response_error"]
