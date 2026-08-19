from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from contextworld.evaluation.action_delay_multistep import (
    DELAYS,
    EVAL_SEEDS,
    HORIZONS,
    QUERY_COUNT,
    build_asset,
    select_assignments,
    summarize_records,
)
from contextworld.paths import resolve_contextworld_path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT
    / "configs/benchmark/tworoom_action_delay_h3_multistep_extrap_v1.yaml"
)
CONFIG_V2_PATH = (
    ROOT
    / "configs/benchmark/tworoom_action_delay_h3_multistep_extrap_v2.yaml"
)


def _config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def _config_v2() -> dict:
    return yaml.safe_load(CONFIG_V2_PATH.read_text(encoding="utf-8"))


def test_frozen_multistep_protocol_counts_and_scope() -> None:
    config = _config()
    assert config["status"] == (
        "preregistered_before_catalog_generation_and_model_scoring"
    )
    assert tuple(config["protocol"]["delay_values"]) == DELAYS
    assert tuple(
        config["protocol"]["target_horizons_action_blocks"]
    ) == HORIZONS
    assert config["evaluation"]["eval_seeds"] == list(EVAL_SEEDS)
    assert config["evaluation"]["unique_queries_per_seed"] == 50
    assert config["evaluation"]["unique_queries"] == QUERY_COUNT
    assert config["evaluation"]["every_history_target_cell_has_queries"] == 300
    assert config["evaluation"]["horizon_loss_records_per_checkpoint"] == 43200
    assert config["protocol"]["high_endpoint_extrapolation_delay_values"] == [5]
    assert "Delays above 5 require" in config["protocol"][
        "high_endpoint_scope_note"
    ]


def test_assignments_are_50x6_balanced_and_disjoint_from_v1() -> None:
    config = _config()
    assignments = select_assignments(config, repo_root=ROOT)
    assert len(assignments) == QUERY_COUNT
    assert len({row.query_id for row in assignments}) == QUERY_COUNT

    for seed in EVAL_SEEDS:
        selected = [row for row in assignments if row.eval_seed == seed]
        assert len(selected) == 50
        assert sum(row.template.direction == "up" for row in selected) == 25
        assert sum(row.template.direction == "down" for row in selected) == 25

    old_catalog_path = resolve_contextworld_path(
        config["source_identity"]["completed_one_step_validation"]["catalog"],
        repo_root=ROOT,
    )
    old_catalog = json.loads(old_catalog_path.read_text(encoding="utf-8"))

    def query_coordinate(template: dict) -> tuple[float, float]:
        sign = 1.0 if template["direction"] == "up" else -1.0
        return (
            float(template["reset_state"][0]),
            float(template["reset_state"][1]) + 35.0 * sign,
        )

    old = {query_coordinate(row["template"]) for row in old_catalog["queries"]}
    new = {
        (
            float(row.template.reset_state[0]),
            float(row.template.reset_state[1])
            + (35.0 if row.template.direction == "up" else -35.0),
        )
        for row in assignments
    }
    assert len(new) == QUERY_COUNT
    assert not (old & new)


def test_one_multistep_family_has_shared_query_and_distinct_futures() -> None:
    assignment = select_assignments(_config(), repo_root=ROOT)[0]
    arrays, audit = build_asset(
        assignment,
        agent_speed=7.0,
        query_action_magnitude=0.35,
    )
    assert audit["family_passed"] is True
    assert all(audit["family_checks"].values())
    assert arrays["history_pixels"].shape == (6, 3, 224, 224, 3)
    assert arrays["action_blocks"].shape == (6, 7, 5, 2)
    assert arrays["target_pixels"].shape == (6, 5, 224, 224, 3)
    assert arrays["history_states"].shape == (6, 3, 2)
    assert arrays["target_states"].shape == (6, 5, 2)
    assert np.all(arrays["history_states"][:, -1] == arrays["history_states"][0, -1])
    for horizon in HORIZONS:
        states = arrays["target_states"][:, horizon - 1]
        assert len({tuple(map(float, value)) for value in states}) == 6


def test_full_action_confirmation_uses_new_300_queries() -> None:
    config = _config_v2()
    assert config["protocol"]["query_action_magnitude"] == 1.0
    assert (
        config["protocol"]["adjacent_delay_endpoint_spacing_at_h1_px"]
        == 7.0
    )
    assignments = select_assignments(config, repo_root=ROOT)
    assert len(assignments) == QUERY_COUNT
    assert all(row.query_id.startswith("action-delay-ms2-") for row in assignments)

    excluded = set()
    identities = [
        config["source_identity"]["completed_one_step_validation"],
        config["generation"]["additional_excluded_catalogs"][0],
    ]
    for identity in identities:
        path = resolve_contextworld_path(
            identity["catalog"], repo_root=ROOT
        )
        catalog = json.loads(path.read_text(encoding="utf-8"))
        excluded.update(
            (
                float(row["query_coordinate"][0]),
                float(row["query_coordinate"][1]),
            )
            if "query_coordinate" in row
            else (
                float(row["template"]["reset_state"][0]),
                float(row["template"]["reset_state"][1])
                + (
                    35.0
                    if row["template"]["direction"] == "up"
                    else -35.0
                ),
            )
            for row in catalog["queries"]
        )
    selected = {
        (
            float(row.template.reset_state[0]),
            float(row.template.reset_state[1])
            + (35.0 if row.template.direction == "up" else -35.0),
        )
        for row in assignments
    }
    assert len(selected) == QUERY_COUNT
    assert not (selected & excluded)

    arrays, audit = build_asset(
        assignments[0],
        agent_speed=7.0,
        query_action_magnitude=1.0,
    )
    assert audit["family_passed"] is True
    h1_y = arrays["target_states"][:, 0, 1]
    assert np.allclose(np.sort(np.abs(np.diff(np.sort(h1_y)))), 7.0)


def test_perfect_loss_matrix_recovers_delay_at_every_horizon() -> None:
    records = []
    for query_index in range(QUERY_COUNT):
        seed = EVAL_SEEDS[query_index // 50]
        direction = "up" if query_index % 2 == 0 else "down"
        for history_delay in DELAYS:
            for target_delay in DELAYS:
                for horizon in HORIZONS:
                    records.append(
                        {
                            "query_id": f"q{query_index:03d}",
                            "eval_seed": seed,
                            "evaluation_index": query_index % 50,
                            "direction": direction,
                            "history_delay": history_delay,
                            "target_delay": target_delay,
                            "target_track": "unused",
                            "horizon": horizon,
                            "latent_mse": float(
                                abs(history_delay - target_delay)
                            ),
                        }
                    )
    summary = summarize_records(records)
    for horizon in HORIZONS:
        overall = summary["by_horizon"][str(horizon)]["overall"]
        assert overall["history_selection_accuracy"] == 1.0
        assert overall["target_selection_accuracy"] == 1.0
        assert overall["matching_history_strict_win_rate"] == 1.0
        assert overall["mean_history_loss_ratio"] == 0.0
