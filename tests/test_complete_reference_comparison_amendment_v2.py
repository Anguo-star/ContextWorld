from __future__ import annotations

import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
PREREGISTRATION = (
    ROOT
    / "configs/benchmark/contextworld_complete_reference_comparison_execution_amendment_v2.yaml"
)
RECEIPT = (
    ROOT
    / "artifacts/evaluation/complete_reference_comparison_v1/"
    "execution_amendment_v2_freeze_receipt.json"
)


def test_report_all_policy_separates_execution_from_verdict() -> None:
    payload = yaml.safe_load(PREREGISTRATION.read_text(encoding="utf-8"))
    policy = payload["report_all_policy"]
    assert policy["every_frozen_comparator_runs_icl_and_cem"] is True
    assert policy["threshold_controls_execution"] is False
    assert policy["threshold_controls_verdict_only"] is True
    assert policy["failed_results_remain_visible"] is True


def test_amendment_freezes_all_three_previously_missing_pldm_cem_rows() -> None:
    payload = yaml.safe_load(PREREGISTRATION.read_text(encoding="utf-8"))
    cells = payload["execution_cells"]
    assert set(cells) == {
        "action_strength_pldm",
        "reacher_arm_mass_pldm",
        "portal_exit_pldm",
    }
    assert sum(len(cell["checkpoints"]) for cell in cells.values()) == 9
    assert payload["execution_budget"]["newly_authorized_candidate_episodes"] == 2700
    assert payload["decision_rules"]["action_strength_pldm"]["effective_floor"] == 218
    assert payload["decision_rules"]["reacher_arm_mass_pldm"]["effective_floor"] == 233
    assert payload["decision_rules"]["portal_exit_pldm"]["effective_floor"] == 263


def test_amendment_was_frozen_before_candidate_cem_execution() -> None:
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert receipt["status"] == "frozen_before_amended_cem_execution"
    assert receipt["public_access"] == {
        "candidate_cem_episodes_consumed_before_amendment": 0,
        "supplemental_public_scores_observed_before_amendment": False,
    }
    assert sum(
        len(cell["checkpoints"])
        for cell in receipt["execution_cells"].values()
    ) == 9
