from pathlib import Path

import numpy as np

from contextworld.evaluation.cem_config_impact_analysis import (
    holm_adjust,
    load_config,
    paired_sign_flip_test,
    seed_stratified_bootstrap,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/benchmark/tworoom_cem_config_impact_v1.yaml"


def test_preregistration_freezes_full_n50x6_single_factor_matrix():
    config = load_config(CONFIG)
    assert config["shared_protocol"]["evaluations_per_condition"] == 300
    assert (
        config["shared_protocol"][
            "require_full_50x6_for_each_eval_and_condition"
        ]
        is True
    )
    assert config["formal_execution"]["new_closed_loop_raw_records"] == 3600
    baseline = config["configurations"]["baseline"]
    expected = {
        "horizon10": "horizon_action_blocks",
        "samples600": "cem_samples",
        "iterations60": "cem_iterations",
    }
    frozen_fields = (
        "horizon_action_blocks",
        "cem_samples",
        "cem_iterations",
        "cem_topk",
    )
    for name, changed in expected.items():
        variant = config["configurations"][name]
        assert variant["changed_parameter"] == changed
        differences = [
            field
            for field in frozen_fields
            if variant[field] != baseline[field]
        ]
        assert differences == [changed]


def test_runner_matches_preregistered_matrix():
    runner = (
        ROOT / "scripts/run_tworoom_cem_config_impact.sh"
    ).read_text()
    assert '"horizon10|10|5|300|30|30"' in runner
    assert '"samples600|5|5|600|30|30"' in runner
    assert '"iterations60|5|5|300|60|30"' in runner
    assert "--num-eval 50" in runner
    assert "--eval-budget 100" in runner


def test_seed_stratified_bootstrap_is_paired_and_deterministic():
    values = [1.0, 3.0, 5.0, 7.0]
    seeds = [42, 42, 43, 43]
    first = seed_stratified_bootstrap(
        values,
        seeds,
        resamples=2_000,
        random_seed=123,
    )
    second = seed_stratified_bootstrap(
        values,
        seeds,
        resamples=2_000,
        random_seed=123,
    )
    assert first == second
    assert first["estimate"] == 4.0
    assert first["stratum_sizes"] == [2, 2]
    assert first["confidence_interval"][0] <= 4.0
    assert first["confidence_interval"][1] >= 4.0


def test_sign_flip_and_holm_statistics():
    test = paired_sign_flip_test(
        np.full(40, -1.0),
        resamples=20_000,
        random_seed=9,
    )
    assert test["two_sided_p"] < 0.001
    adjusted = holm_adjust({"a": 0.01, "b": 0.03, "c": 0.04})
    assert adjusted == {"a": 0.03, "b": 0.06, "c": 0.06}


def test_long_horizon_diagnostic_is_fixed_before_execution():
    config = load_config(CONFIG)
    diagnostic = config["long_horizon_prediction_diagnostic"]
    assert diagnostic["records"] == 300
    assert diagnostic["probe_last_25_raw_steps"] == "zero_raw_actions"
    assert diagnostic["observed_horizons_raw_steps"] == list(
        range(5, 51, 5)
    )
    runner = (
        ROOT / "scripts/run_tworoom_long_horizon_prediction.sh"
    ).read_text()
    assert "--num-eval 50" in runner
    assert "SEED_LIST:-42 43 44 45 46 47" in runner
