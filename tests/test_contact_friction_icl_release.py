from __future__ import annotations

import json

import numpy as np

from contextworld.benchmarks.contact_friction_icl_data import (
    CONTACT_FRICTION_RELEASE_ID,
    audit_contact_friction_icl_release,
    load_contact_friction_icl_release,
)
from contextworld.benchmarks.contact_friction_icl_score import (
    contact_friction_prediction_gate,
    contact_friction_prediction_metrics,
)
from contextworld.paths import repository_root, resolve_contextworld_path


def test_release_uses_self_explanatory_contact_friction_name() -> None:
    release = load_contact_friction_icl_release()
    assert release["release_id"] == CONTACT_FRICTION_RELEASE_ID
    assert release["scope"]["display_name_zh"] == "PushT 接触摩擦 ICL"
    assert release["scope"]["history_tokens"] == 3
    assert release["scope"]["friction_values"] == [0.05, 0.80]
    assert release["scope"]["excluded_factor"] == (
        "post_release_motion_damping"
    )
    assert release["scope"]["public_test_included"] is True
    assert release["scope"]["sealed_test_included"] is False


def test_v3_data_release_is_ready_independently_of_model_result() -> None:
    release = load_contact_friction_icl_release()
    assert release["data"]["release_version"] == (
        "pusht_contact_friction_h3_release_v3"
    )
    assert release["data"]["protocol"] == (
        "pusht_contact_friction_history3_strict_continuous_v2"
    )
    assert release["data"]["manifest_sha256"] == (
        "cbb9b1a1c030a3c66ea8acbf25c5e1a302f1c43907beeadcdc9d8bd1e989f3d5"
    )
    assert release["data"]["pair_counts"] == {
        "train": 8192,
        "loader_validation": 256,
        "validation": 256,
    }
    assert all(
        value == 0 for value in release["data"]["isolation"].values()
    )

    audit = audit_contact_friction_icl_release(full=False)
    assert audit["passed"]
    assert audit["identity"]["passed"]
    assert audit["causal_data_contract"]["passed"]
    data_release = audit["data_release"]
    assert data_release["status"] == "ready"
    assert data_release["passed"]
    assert all(data_release["checks"].values())
    assert data_release["artifact_tree"]["passed"]
    assert data_release["causal_data_contract"]["passed"]
    assert data_release["causal_data_contract"]["x0_policy"] == (
        "balanced_visible_start"
    )
    assert data_release["causal_data_contract"]["evidence_scope"] == (
        "all 8,704 Training / Development / Public Test pairs"
    )
    assert data_release["row_counts"] == {
        "train": 327680,
        "loader_validation": 10240,
        "validation": 10240,
    }


def test_expanded_release_audit_covers_all_three_splits() -> None:
    release = load_contact_friction_icl_release()
    specification = release["data"]["artifacts"][
        "expanded_release_audit"
    ]
    report = json.loads(
        resolve_contextworld_path(
            specification["path"],
            repo_root=repository_root(),
        ).read_text(encoding="utf-8")
    )
    assert report["passed"] is True
    assert report["manifest_sha256"] == release["data"][
        "manifest_sha256"
    ]
    assert all(report["checks"].values())
    assert {
        split: details["pair_count"]
        for split, details in report["split_reports"].items()
    } == release["data"]["pair_counts"]
    assert all(
        details["passed"] is True
        for details in report["split_reports"].values()
    )


def test_failed_development_result_is_one_compact_current_decision() -> None:
    release = load_contact_friction_icl_release()
    specification = release["reference_results"]["current_decision"]
    decision = json.loads(
        resolve_contextworld_path(
            specification["path"],
            repo_root=repository_root(),
        ).read_text(encoding="utf-8")
    )
    assert set(decision) == {
        "schema_version",
        "component",
        "status",
        "data_release",
        "reported_endpoint",
        "additional_training_seeds_run",
        "public_model_scoring_opened",
        "original_task_cem_run",
        "positive_reference_claim",
    }
    serialized = json.dumps(decision, sort_keys=True)
    assert "attempt" not in serialized
    assert "root_cause" not in serialized
    assert "/opt/" not in serialized
    assert decision["status"] == "failed_development"
    assert decision["additional_training_seeds_run"] is False
    assert decision["public_model_scoring_opened"] is False
    assert decision["original_task_cem_run"] is False
    assert decision["positive_reference_claim"] is False

    result = decision["reported_endpoint"]
    assert result["training_recipe"] == (
        "mixed_frozen_image_paired_future_matching_1p00"
    )
    assert result["training_seed"] == 13313
    assert result["optimizer_step"] == 8192
    assert result["selection_contract"] == {
        "development_used_for_recipe_selection": True,
        "development_used_for_checkpoint_selection": False,
        "checkpoint_step_was_fixed_before_scoring": True,
        "public_test_used_for_selection": False,
    }
    assert result["metrics"] == {
        "correct_future_rate": 0.9609375,
        "correct_history_rate": 0.90234375,
        "context_switch_rate": 0.99609375,
        "worst_friction_correct_future_rate": 0.94921875,
    }
    assert result["failed_metrics"] == ["correct_history_rate"]
    assert result["passed"] is False

    audit = audit_contact_friction_icl_release(full=False)
    reference = audit["reference_result"]
    assert reference["status"] == "failed_development"
    assert reference["integrity_status"] == "passed"
    assert reference["development_gate_passed"] is False
    assert reference["public_model_scoring_opened"] is False
    assert reference["failed_metrics"] == ["correct_history_rate"]
    assert all(reference["checks"].values())
    assert reference["passed"]
    legacy_gate = contact_friction_prediction_gate(
        result["metrics"], release=release
    )
    assert legacy_gate["passed"] is False
    assert legacy_gate["checks"]["target_latent_separation"] is False


def test_prediction_metrics_use_matching_real_future_and_history() -> None:
    target_low = np.asarray([[0.0, 0.0], [1.0, 1.0]])
    target_high = np.asarray([[2.0, 2.0], [3.0, 3.0]])
    metrics, records = contact_friction_prediction_metrics(
        pair_ids=("pair-0", "pair-1"),
        predicted_low=target_low,
        predicted_high=target_high,
        target_low=target_low,
        target_high=target_high,
    )
    assert metrics["correct_future_rate"] == 1.0
    assert metrics["correct_history_rate"] == 1.0
    assert metrics["context_switch_rate"] == 1.0
    assert metrics["worst_friction_correct_future_rate"] == 1.0
    assert metrics["joint_icl_pair_success_rate"] == 1.0
    assert len(records) == 2
    release = load_contact_friction_icl_release()
    gate = contact_friction_prediction_gate(
        metrics,
        release=release,
    )
    assert gate["passed"]
    assert gate["checks"]["target_latent_separation"]
    assert gate["checks"]["response_gain"]
    assert gate["checks"]["normalized_response_error"]
