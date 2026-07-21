from pathlib import Path

import numpy as np
import pytest
import yaml

from contextworld.evaluation.icl_sensitive import SensitiveGeometry
from scripts.analyze_tworoom_speed_multistep_extrap_v5 import (
    HORIZONS as ANALYZER_HORIZONS,
    _longest_contiguous,
)
from scripts.build_tworoom_speed_multistep_catalogs import (
    HORIZONS,
    _condition_names,
    _free_motion_residual,
    _query_actions,
)
from scripts.eval_tworoom_speed_multistep_latent import (
    HORIZONS as SCORER_HORIZONS,
    summarize_records,
)


ROOT = Path(__file__).resolve().parents[1]
EVAL_SEEDS = [42, 43, 44, 45, 46, 47]


def test_v5_protocol_freezes_outer_support_multistep_and_counts() -> None:
    config = yaml.safe_load(
        (
            ROOT
            / "configs/benchmark/tworoom_speed_multistep_extrap_v5.yaml"
        ).read_text(encoding="utf-8")
    )
    evaluation = config["evaluation"]
    tracks = config["data"]["tracks"]

    assert config["status"] == (
        "preregistered_before_catalog_generation_and_model_scoring"
    )
    assert evaluation["eval_seeds"] == EVAL_SEEDS
    assert evaluation["unique_queries_per_reference_speed_per_seed"] == 50
    assert evaluation["unique_queries_per_reference_speed"] == 300
    assert evaluation["target_horizons_action_blocks"] == [1, 2, 3, 5]
    assert tuple(evaluation["target_horizons_action_blocks"]) == HORIZONS
    assert SCORER_HORIZONS == HORIZONS
    assert ANALYZER_HORIZONS == HORIZONS
    assert tracks["seen_for_multi"]["speeds"] == [3.1, 5.1, 7.0]
    assert tracks["unseen_interpolation"]["speeds"] == [3.4, 4.8, 6.9]
    assert tracks["extrapolation_low"]["speeds"] == [1.75, 1.95, 2.15, 2.35]
    assert tracks["extrapolation_high"]["speeds"] == [8.25, 8.75, 9.5, 10.25]
    training = yaml.safe_load(
        (ROOT / config["data"]["source_training_protocol"]).read_text(
            encoding="utf-8"
        )
    )
    multi_train = set(training["speed_support"]["multi_synthetic_train"])
    assert set(tracks["unseen_interpolation"]["speeds"]).isdisjoint(
        multi_train
    )
    assert max(tracks["extrapolation_low"]["speeds"]) < min(multi_train)
    assert min(tracks["extrapolation_high"]["speeds"]) > max(multi_train)
    expected_trajectories = 0
    for name, track in tracks.items():
        speed_count = len(track["speeds"])
        matrix = evaluation["matrix_by_track"][name]
        assert matrix["cells"] == speed_count**2
        assert matrix["condition_trajectories"] == speed_count**2 * 300
        expected_trajectories += matrix["condition_trajectories"]
    assert evaluation["condition_trajectories_per_checkpoint_all_tracks"] == 15000
    assert evaluation["horizon_losses_per_checkpoint_all_tracks"] == 60000
    assert expected_trajectories == 15000
    assert expected_trajectories * len(HORIZONS) == 60000
    assert sum(len(rows) for rows in config["models"].values()) == 7
    assert "inferred_speed" in config["metrics"]["excluded"]
    assert "cem_success_rate" in config["metrics"]["excluded"]


def test_query_action_families_are_bounded_and_five_blocks() -> None:
    geometry = SensitiveGeometry(
        template_id="unit",
        distance_bin=72,
        geometry_variant=0,
        reset_state=(160.0, 100.0),
        goal_state=(160.0, 172.0),
        context_direction=(1.0, 0.0),
        query_action=(0.5, 0.0),
    )
    observed = set()
    for family_index in range(3):
        name, actions = _query_actions(
            geometry, magnitude=0.35, family_index=family_index
        )
        observed.add(name)
        assert actions.shape == (5, 5, 2)
        assert np.max(np.abs(actions)) <= 0.35 + 1e-7
        cumulative = np.cumsum(actions.reshape(-1, 2), axis=0)[4::5]
        assert np.all(
            np.linalg.norm(cumulative[np.asarray(HORIZONS) - 1], axis=-1)
            > 1e-6
        )
        assert np.linalg.norm(cumulative[3]) < 1e-6
    assert observed == {
        "bounded_axis_loop_x_first",
        "bounded_axis_loop_y_first",
        "bounded_diagonal_loop",
    }
    family_counts = {}
    for index in range(300):
        name, _ = _query_actions(
            geometry, magnitude=0.35, family_index=index
        )
        family_counts[name] = family_counts.get(name, 0) + 1
    assert set(family_counts.values()) == {100}


def test_free_motion_residual_is_zero_for_exact_dynamics() -> None:
    _, actions = _query_actions(
        SensitiveGeometry(
            template_id="unit",
            distance_bin=72,
            geometry_variant=0,
            reset_state=(160.0, 100.0),
            goal_state=(160.0, 172.0),
            context_direction=(1.0, 0.0),
            query_action=(0.5, 0.0),
        ),
        magnitude=0.35,
        family_index=2,
    )
    reset = np.asarray([160.0, 100.0], dtype=np.float32)
    speed = 7.0
    cumulative = np.cumsum(actions.reshape(-1, 2), axis=0)[4::5]
    next_states = reset[None] + speed * cumulative
    assert _free_motion_residual(tuple(reset), speed, actions, next_states) < 1e-5


def _synthetic_records() -> tuple[list[dict], list[str]]:
    speeds = (1.75, 1.95, 2.15, 2.35)
    conditions = _condition_names(len(speeds))
    records = []
    for speed_index, speed in enumerate(speeds):
        matching = conditions[speed_index]
        for seed_index, eval_seed in enumerate(EVAL_SEEDS):
            for evaluation_index in range(50):
                static_index = seed_index * 50 + evaluation_index
                for condition in conditions:
                    records.append(
                        {
                            "query_id": f"q-{speed:g}-{static_index:03d}",
                            "static_query_id": f"static-{static_index:03d}",
                            "template_id": f"template-{static_index:03d}",
                            "reference_speed": speed,
                            "matching_condition": matching,
                            "action_family": "test",
                            "eval_seed": eval_seed,
                            "evaluation_index": evaluation_index,
                            "condition": condition,
                            "history_speed": speeds[conditions.index(condition)],
                            "latent_mse_by_horizon": {
                                str(horizon): (
                                    float(horizon)
                                    if condition == matching
                                    else float(2 * horizon)
                                )
                                for horizon in HORIZONS
                            },
                        }
                    )
    return records, conditions


def test_full_four_by_four_50_by_6_matrix_and_metric(monkeypatch) -> None:
    monkeypatch.setattr(
        "scripts.eval_tworoom_speed_multistep_latent._bootstrap_ratio",
        lambda rows, seed, samples=10000: {
            "clusters": 300,
            "ci_low": 0.49,
            "ci_high": 0.51,
        },
    )
    records, conditions = _synthetic_records()
    summary = summarize_records(
        records,
        conditions=conditions,
        eval_seeds=EVAL_SEEDS,
        expected_per_seed=50,
        bootstrap_seed=123,
    )
    assert len(records) == 4800
    assert summary["count_audit"]["passed"]
    assert summary["count_audit"]["condition_trajectories"] == 4800
    assert summary["count_audit"]["horizon_loss_records"] == 19200
    for horizon in HORIZONS:
        row = summary["by_horizon"][str(horizon)]
        assert row["formal_within_checkpoint_pass"]
        assert row["strict_each_alternative_pass"]
        assert row["reference_speed_balanced_relative_loss_reduction"] == pytest.approx(
            0.5
        )


def test_longest_contiguous_horizon_stops_at_first_failure() -> None:
    assert _longest_contiguous({"1": True, "2": True, "3": False, "5": True}) == 2
    assert _longest_contiguous({"1": False, "2": True, "3": True, "5": True}) == 0
    assert _longest_contiguous({str(h): True for h in HORIZONS}) == 5


def test_summary_rejects_query_reuse_across_eval_seeds(monkeypatch) -> None:
    monkeypatch.setattr(
        "scripts.eval_tworoom_speed_multistep_latent._bootstrap_ratio",
        lambda rows, seed, samples=10000: {
            "clusters": len({row["static_query_id"] for row in rows}),
            "ci_low": 0.0,
            "ci_high": 1.0,
        },
    )
    records, conditions = _synthetic_records()
    for row in records:
        if row["eval_seed"] == EVAL_SEEDS[1]:
            index = int(row["evaluation_index"])
            row["static_query_id"] = f"static-{index:03d}"
    summary = summarize_records(
        records,
        conditions=conditions,
        eval_seeds=EVAL_SEEDS,
        expected_per_seed=50,
        bootstrap_seed=123,
    )
    assert not summary["count_audit"][
        "eval_seed_query_partitions_are_disjoint"
    ]
    assert not summary["count_audit"]["passed"]


def test_primary_average_gate_is_separate_from_strict_diagnostic(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "scripts.eval_tworoom_speed_multistep_latent._bootstrap_ratio",
        lambda rows, seed, samples=10000: {
            "clusters": 300,
            "ci_low": 0.0,
            "ci_high": 1.0,
        },
    )
    records, conditions = _synthetic_records()
    reference_speed = 1.75
    matching = conditions[0]
    lower_alternative = conditions[1]
    for row in records:
        if row["reference_speed"] != reference_speed:
            continue
        for horizon in HORIZONS:
            if row["condition"] == matching:
                row["latent_mse_by_horizon"][str(horizon)] = 1.0
            elif row["condition"] == lower_alternative:
                row["latent_mse_by_horizon"][str(horizon)] = 0.5
            else:
                row["latent_mse_by_horizon"][str(horizon)] = 3.0
    summary = summarize_records(
        records,
        conditions=conditions,
        eval_seeds=EVAL_SEEDS,
        expected_per_seed=50,
        bootstrap_seed=123,
    )
    for horizon in HORIZONS:
        row = summary["by_horizon"][str(horizon)]
        assert row["formal_within_checkpoint_pass"]
        assert not row["strict_each_alternative_pass"]


def test_one_negative_eval_seed_fails_formal_direction_gate(monkeypatch) -> None:
    monkeypatch.setattr(
        "scripts.eval_tworoom_speed_multistep_latent._bootstrap_ratio",
        lambda rows, seed, samples=10000: {
            "clusters": 300,
            "ci_low": -1.0,
            "ci_high": 1.0,
        },
    )
    records, conditions = _synthetic_records()
    reference_speed = 1.75
    matching = conditions[0]
    for row in records:
        if (
            row["reference_speed"] == reference_speed
            and row["eval_seed"] == EVAL_SEEDS[0]
        ):
            for horizon in HORIZONS:
                row["latent_mse_by_horizon"][str(horizon)] = (
                    3.0 if row["condition"] == matching else 2.0
                )
    summary = summarize_records(
        records,
        conditions=conditions,
        eval_seeds=EVAL_SEEDS,
        expected_per_seed=50,
        bootstrap_seed=123,
    )
    for horizon in HORIZONS:
        row = summary["by_horizon"][str(horizon)]
        assert not row["formal_within_checkpoint_pass"]
        assert not row["by_reference_speed"][str(reference_speed)][
            "all_eval_seed_directions_positive"
        ]
