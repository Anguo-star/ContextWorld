from __future__ import annotations

from pathlib import Path

import yaml

from contextworld.evaluation.action_delay_validation import (
    EVAL_SEEDS,
    LOSS_RECORDS_PER_CHECKPOINT,
    QUERY_COUNT,
    build_validation_asset,
    select_validation_assignments,
    summarize_validation_records,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs/benchmark/tworoom_action_delay_h3_validation_v1.yaml"
)


def _config() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def test_validation_assigns_50_unique_queries_to_each_of_six_seeds() -> None:
    assignments = select_validation_assignments(_config())

    assert len(assignments) == QUERY_COUNT == 300
    assert len({row.query_id for row in assignments}) == 300
    query_states = {
        (
            row.template.reset_state[0],
            row.template.reset_state[1]
            + (35.0 if row.template.direction == "up" else -35.0),
        )
        for row in assignments
    }
    assert len(query_states) == 300
    for seed in EVAL_SEEDS:
        rows = [row for row in assignments if row.eval_seed == seed]
        assert len(rows) == 50
        assert sum(row.template.direction == "up" for row in rows) == 25
        assert sum(row.template.direction == "down" for row in rows) == 25
        assert {row.evaluation_index for row in rows} == set(range(50))


def test_validation_asset_contains_exact_five_by_five_offline_matrix() -> None:
    assignment = select_validation_assignments(_config())[0]
    arrays, audit = build_validation_asset(
        assignment,
        agent_speed=7.0,
    )

    assert audit["family_passed"] is True
    assert all(audit["family_checks"].values())
    assert arrays["history_pixels"].shape == (5, 3, 224, 224, 3)
    assert arrays["action_blocks"].shape == (5, 3, 5, 2)
    assert arrays["target_pixels"].shape == (5, 224, 224, 3)
    assert arrays["history_states"].shape == (5, 3, 2)
    assert arrays["target_states"].shape == (5, 2)


def test_summary_recovers_a_perfect_five_delay_matrix() -> None:
    records = []
    for seed in EVAL_SEEDS:
        for evaluation_index in range(50):
            query_id = f"s{seed}-q{evaluation_index:02d}"
            direction = "up" if evaluation_index < 25 else "down"
            for history_delay in range(5):
                for target_delay in range(5):
                    records.append(
                        {
                            "query_id": query_id,
                            "eval_seed": seed,
                            "evaluation_index": evaluation_index,
                            "direction": direction,
                            "history_delay": history_delay,
                            "target_delay": target_delay,
                            "target_track": (
                                "training_seen"
                                if target_delay in (0, 2, 4)
                                else "interpolation"
                            ),
                            "latent_mse": float(
                                abs(history_delay - target_delay)
                            ),
                        }
                    )

    assert len(records) == LOSS_RECORDS_PER_CHECKPOINT == 7500
    summary = summarize_validation_records(records)
    assert summary["overall"]["history_selection_accuracy"] == 1.0
    assert summary["overall"]["target_selection_accuracy"] == 1.0
    assert summary["overall"]["matching_history_strict_win_rate"] == 1.0
    assert all(
        value["mean_history_margin"] > 0
        for value in summary["by_target_delay"].values()
    )
