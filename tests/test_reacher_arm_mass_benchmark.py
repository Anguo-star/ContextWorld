from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from contextworld.benchmarks.reacher_arm_mass_icl_data import (
    load_reacher_arm_mass_icl_release,
)
from contextworld.benchmarks.reacher_arm_mass_icl_score import (
    reacher_arm_mass_prediction_gate,
    reacher_arm_mass_prediction_metrics,
)


def test_reacher_arm_mass_metrics_reward_the_matching_real_future() -> None:
    target_lighter = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    target_heavier = np.array([[2.0, 0.0], [3.0, 1.0]], dtype=np.float32)
    metrics, records = reacher_arm_mass_prediction_metrics(
        pair_ids=("a", "b"),
        predicted_lighter=target_lighter.copy(),
        predicted_heavier=target_heavier.copy(),
        target_lighter=target_lighter,
        target_heavier=target_heavier,
    )
    assert metrics["correct_future_rate"] == 1.0
    assert metrics["correct_history_rate"] == 1.0
    assert metrics["context_switch_rate"] == 1.0
    assert metrics["worst_mass_correct_future_rate"] == 1.0
    assert metrics["joint_icl_pair_success_rate"] == 1.0
    assert len(records) == 2
    gate = reacher_arm_mass_prediction_gate(
        metrics, release=load_reacher_arm_mass_icl_release()
    )
    assert gate["passed"]
    assert gate["checks"]["target_latent_separation"]
    assert gate["checks"]["response_gain"]
    assert gate["checks"]["normalized_response_error"]


def test_reacher_arm_mass_current_frame_only_prediction_is_chance() -> None:
    target_lighter = np.array([[0.0], [0.0]], dtype=np.float32)
    target_heavier = np.array([[2.0], [2.0]], dtype=np.float32)
    common = np.array([[0.0], [0.0]], dtype=np.float32)
    metrics, _ = reacher_arm_mass_prediction_metrics(
        pair_ids=("a", "b"),
        predicted_lighter=common,
        predicted_heavier=common,
        target_lighter=target_lighter,
        target_heavier=target_heavier,
    )
    assert metrics["correct_future_rate"] == 0.5
    assert metrics["context_switch_rate"] == 0.0
    assert metrics["joint_icl_pair_success_rate"] == 0.0


def test_reacher_release_uses_one_portable_final_result() -> None:
    release = load_reacher_arm_mass_icl_release()
    assert release["release_status"] == "public_test_release_candidate"
    assert tuple(release["reference_results"]) == (
        "final_release_summary",
        "latent_response_summary",
    )
    path = Path(release["reference_results"]["final_release_summary"]["path"])
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["status"] == "passed_public_test_3_of_3"
    assert sum(row["passed"] for row in payload["public_test"]["lewm"]) == 3
    legacy = payload["public_test"]["lewm"][0]
    legacy_metrics = {
        **{
            name: legacy[name]
            for name in (
                "correct_future_rate",
                "correct_history_rate",
                "context_switch_rate",
                "worst_mass_correct_future_rate",
            )
        },
        "paired_bootstrap_95_lower_bound": legacy[
            "bootstrap_lower_bounds"
        ],
    }
    assert not reacher_arm_mass_prediction_gate(
        legacy_metrics, release=release
    )["passed"]
    response_path = Path(
        release["reference_results"]["latent_response_summary"]["path"]
    )
    response = json.loads(response_path.read_text(encoding="utf-8"))
    for response_family, reference_family in (
        ("LeWM", "lewm"),
        ("PLDM", "pldm"),
    ):
        assert {
            row["training_seed"]: row["checkpoint_sha256"]
            for row in response["methods"][response_family]
        } == {
            row["seed"]: row["checkpoint_sha256"]
            for row in payload["public_test"][reference_family]
        }
    assert sum(
        row["response_gate_passed"]
        for row in response["methods"]["LeWM"]
    ) == 3
    assert "/opt/" not in json.dumps(payload, sort_keys=True)
