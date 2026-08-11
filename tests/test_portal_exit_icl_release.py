from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from contextworld.benchmarks.portal_exit_icl_data import (
    PortalExitICLEvalDataset,
    load_portal_exit_icl_release,
)
from contextworld.benchmarks.portal_exit_icl_score import (
    portal_exit_prediction_gate,
    portal_exit_prediction_metrics,
)
from contextworld.evaluation.icl_model import file_sha256
from contextworld.paths import repository_root, resolve_contextworld_path


def test_frozen_public_test_loads_all_pairs() -> None:
    dataset = PortalExitICLEvalDataset()
    arrays = dataset.arrays
    assert arrays.pair_count == 256
    assert arrays.near_border_pixels.shape == (256, 4, 224, 224, 3)
    assert arrays.raw_action_blocks.shape == (256, 4, 5, 2)
    assert np.array_equal(
        arrays.near_border_pixels[:, 2],
        arrays.farther_from_border_pixels[:, 2],
    )


def test_perfect_portal_exit_predictions_pass() -> None:
    pairs = tuple(f"p-{index}" for index in range(8))
    near = np.zeros((8, 4), dtype=np.float32)
    farther = np.ones((8, 4), dtype=np.float32)
    metrics, _ = portal_exit_prediction_metrics(
        pair_ids=pairs,
        predicted_near=near,
        predicted_farther=farther,
        target_near=near,
        target_farther=farther,
    )
    release = PortalExitICLEvalDataset().release
    gate = portal_exit_prediction_gate(metrics, release=release)
    assert gate["passed"]
    assert gate["checks"]["target_latent_separation"]
    assert gate["checks"]["response_gain"]
    assert gate["checks"]["normalized_response_error"]
    assert metrics["uncertainty"]["lower_bounds"]["correct_future_rate"] == 1.0
    assert metrics["joint_icl_pair_success_rate"] == 1.0


def test_portal_release_uses_one_portable_final_result() -> None:
    release = load_portal_exit_icl_release()
    assert tuple(release["reference_results"]) == (
        "final_release_summary",
        "latent_response_summary",
    )
    path = Path(release["reference_results"]["final_release_summary"]["path"])
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["status"] == "failed_public_test_0_of_3"
    assert not any(
        row["passed"]
        for family in ("lewm_paired_real_future", "pldm")
        for row in payload["public_test"][family]
    )
    legacy = payload["public_test"]["lewm_paired_real_future"][0]
    assert not portal_exit_prediction_gate(
        {
            name: legacy[name]
            for name in (
                "correct_future_rate",
                "correct_history_rate",
                "context_switch_rate",
                "worst_exit_correct_future_rate",
            )
        },
        release=load_portal_exit_icl_release(),
    )["passed"]
    response_path = Path(
        release["reference_results"]["latent_response_summary"]["path"]
    )
    response = json.loads(response_path.read_text(encoding="utf-8"))
    for response_family, reference_family in (
        ("LeWM", "lewm_paired_real_future"),
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
    assert not any(
        row["response_gate_passed"]
        for row in response["methods"]["PLDM"]
    )
    assert "/opt/" not in json.dumps(payload, sort_keys=True)


def test_portal_summary_binds_scores_to_selected_lewm_checkpoints() -> None:
    release = load_portal_exit_icl_release()
    summary_path = Path(
        release["reference_results"]["final_release_summary"]["path"]
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    configured = {
        row["seed"]: (row["path"], row["sha256"])
        for row in release["training"]["reference_matrix"]["models"]["LeWM"][
            "checkpoints"
        ]
    }
    reported = {
        row["seed"]: (row["checkpoint"], row["checkpoint_sha256"])
        for row in summary["public_test"]["lewm_paired_real_future"]
    }
    assert reported == configured
    for path, expected_sha256 in configured.values():
        checkpoint = resolve_contextworld_path(
            path,
            repo_root=repository_root(),
        )
        assert file_sha256(checkpoint) == expected_sha256
