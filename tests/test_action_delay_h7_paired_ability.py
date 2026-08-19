from __future__ import annotations

from pathlib import Path

import yaml

from scripts.analyze_tworoom_action_delay_h7_candidate_resolution import (
    score_h1_candidate_resolution,
)
from scripts.analyze_tworoom_action_delay_h7_paired_ability import (
    _paired_noninferiority,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs/benchmark/"
    "tworoom_action_delay_h7_paired_ability_retention_v1.yaml"
)
CURRICULUM_CONFIG = (
    ROOT
    / "configs/benchmark/"
    "tworoom_action_delay_h7_curriculum_ability_retention_v1.yaml"
)


def test_paired_ability_protocol_freezes_full_matrix() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    models = [
        {**row, "role": role}
        for role, rows in config["models"].items()
        for row in rows
    ]
    assert config["status"] == "preregistered_before_cem_scoring"
    assert len(models) == 10
    assert {row["history_size"] for row in models} == {3, 7}
    assert len(config["evaluation"]["domains"]) == 2
    assert len(config["evaluation"]["eval_seeds"]) == 6
    assert config["evaluation"]["expected_jobs"] == 120
    assert (
        config["evaluation"]["expected_independent_planning_evaluations"]
        == 6000
    )
    gate = config["evaluation"]["paired_non_inferiority"]
    assert gate["primary_reference"] == "h3_origheldout_s3072"
    assert gate["success_rate_delta_minimum"] == -0.05
    assert gate["final_distance_delta_px_maximum"] == 5.0
    assert gate["bootstrap_resamples"] == 10_000


def test_curriculum_ability_protocol_is_seven_model_matrix() -> None:
    config = yaml.safe_load(
        CURRICULUM_CONFIG.read_text(encoding="utf-8")
    )
    models = [
        {**row, "role": role}
        for role, rows in config["models"].items()
        for row in rows
    ]

    assert config["status"] == "preregistered_before_cem_scoring"
    assert len(models) == 7
    assert config["evaluation"]["expected_jobs"] == 84
    assert (
        config["evaluation"]["expected_independent_planning_evaluations"]
        == 4200
    )
    assert config["evaluation"]["decision"]["target_role"] == (
        "pldm_reference"
    )


def test_candidate_resolution_separates_coarse_and_fine_selection() -> None:
    selected_delay = {delay: delay for delay in range(11)}
    selected_delay.update({0: 1, 4: 4, 8: 5})
    records = []
    for query_index in range(300):
        query_id = f"query-{query_index:03d}"
        for history_delay in range(11):
            for target_delay in range(11):
                records.append(
                    {
                        "query_id": query_id,
                        "history_delay": history_delay,
                        "target_delay": target_delay,
                        "horizon": 1,
                        "latent_mse": float(
                            abs(
                                target_delay
                                - selected_delay[history_delay]
                            )
                        ),
                    }
                )
    result = score_h1_candidate_resolution(records)
    restricted = result["restricted_three_way"]
    assert restricted["target_selection_units"] == 900
    assert restricted["exact_target_selection_rate"] == 2 / 3
    assert restricted["exact_history_selection_rate"] == 1.0
    confusion = result["full_eleven_way"][
        "selected_target_confusion_counts"
    ]
    assert confusion["0"]["1"] == 300
    assert confusion["4"]["4"] == 300
    assert confusion["8"]["5"] == 300


def test_paired_noninferiority_passes_identical_cem_records() -> None:
    records = {}
    for index in range(300):
        key = f"eval-{index:03d}"
        records[key] = {
            "eval_seed": 42 + index // 50,
            "evaluation_index": index % 50,
            "source_kind": "original_h5",
            "source_path": "tworoom.h5",
            "episode": index,
            "start_step": 0,
            "goal_offset": 50,
            "cem_group_seed": index // 10,
            "stratum": "short",
            "room_relation": "same_room",
            "initial_state": [1.0, 2.0],
            "goal_state": [3.0, 4.0],
            "success": index % 5 != 0,
            "final_distance": float(index % 7),
        }
    config = {
        "evaluation": {
            "paired_non_inferiority": {
                "bootstrap_resamples": 1_000,
                "confidence_level": 0.95,
                "success_rate_delta_minimum": -0.05,
                "final_distance_delta_px_maximum": 5.0,
            }
        }
    }
    result = _paired_noninferiority(
        records, records, seed=1234, config=config
    )
    assert result["passed"] is True
    assert result["candidate_minus_reference_success_rate"] == {
        "point": 0.0,
        "ci_lower": 0.0,
        "ci_upper": 0.0,
    }
    assert result["candidate_minus_reference_final_distance_px"] == {
        "point": 0.0,
        "ci_lower": 0.0,
        "ci_upper": 0.0,
    }
