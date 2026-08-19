"""Tests for the strict PushT action-strength release builders."""

import inspect

from scripts import build_pusht_replay_matched_confirmatory_h3 as confirm
from scripts import build_pusht_replay_matched_hidden_actuation_h3 as build


def _pair_report(*, gap: float, history: float, future: float):
    return {
        "audit": {
            "state_installations_after_x0": 0,
            "query_simulator_recreated": False,
            "query_physics_max_abs_gap": gap,
            "pair_query_pixel_difference": 0,
            "pair_query_action_difference": 0.0,
            "history_effect": history,
            "true_future_effect": future,
            "query_physics_tolerance": 1e-5,
            "full_state_dimensions": 12,
            "full_state_components": [
                f"state_{index}" for index in range(12)
            ],
        }
    }


def test_training_builder_emits_machine_auditable_strict_summary():
    report = build.strict_causal_chain_audit(
        [
            _pair_report(gap=2e-6, history=12.0, future=3.0),
            _pair_report(gap=7e-6, history=11.0, future=4.0),
        ]
    )

    assert report["passed"] is True
    assert report["pair_count"] == 2
    assert report["state_installations_after_x0"] == 0
    assert report["query_simulator_recreated"] is False
    assert report["max_pair_full_state_gap"] == 7e-6
    assert report["max_pair_query_pixel_difference"] == 0
    assert report["max_pair_query_action_difference"] == 0.0
    assert report["min_history_effect"] == 11.0
    assert report["min_true_future_effect"] == 3.0
    assert report["full_state_dimensions"] == 12


def test_confirmation_builder_reuses_strict_training_generator_and_audit():
    source = inspect.getsource(confirm.main)

    assert "build_split(" in source
    assert 'strict_audit = report["strict_causal_chain_audit"]' in source
    assert 'strict_audit["passed"]' in source
    assert "strict" in confirm.PROTOCOL
    assert "strict" in str(confirm.DEFAULT_OUTPUT)
