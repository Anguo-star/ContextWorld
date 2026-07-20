from __future__ import annotations

from pathlib import Path

import yaml

from contextworld.synthesis.config import scenario_requests
from contextworld.training.tworoom_data import CATALOG_BY_GROUP


ROOT = Path(__file__).resolve().parents[1]


def _load(path: str) -> dict:
    return yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))


def test_unseen_validation_speeds_are_numerically_isolated() -> None:
    benchmark = _load("configs/benchmark/tworoom_speed_isolated_v2.yaml")
    support = benchmark["speed_support"]
    original = set(map(float, support["original_train"]))
    train = set(map(float, support["multi_synthetic_train"]))
    monitor = set(map(float, support["training_monitor_only"]))
    calibration = set(map(float, support["planner_calibration"]))
    seen = set(map(float, support["validation_seen_for_multi"]))
    unseen = set(map(float, support["validation_unseen_interpolation"]))
    sealed_test = set(map(float, support["sealed_test_interpolation"]))

    assert seen <= train
    assert not unseen & (original | train | monitor)
    assert not calibration & (seen | unseen | sealed_test)
    assert not sealed_test & (
        original | train | monitor | calibration | seen | unseen
    )
    assert min(train) < min(unseen) < max(unseen) < max(train)


def test_single_and_multi_training_data_match_except_speed_support() -> None:
    single = _load("configs/synthesis/tworoom_speed_single_matched_v2.yaml")
    multi = _load("configs/synthesis/tworoom_speed_full_v1.yaml")
    single_requests = scenario_requests(single)
    multi_requests = scenario_requests(multi)

    assert len(single_requests) == len(multi_requests) == 608
    assert single["seed"] == multi["seed"]
    assert single["scenario_generation_seed"] == multi[
        "scenario_generation_seed"
    ]
    assert single["collection"] == multi["collection"]

    single_by_group = {request.seed_group: request for request in single_requests}
    multi_by_group = {request.seed_group: request for request in multi_requests}
    assert single_by_group.keys() == multi_by_group.keys()
    for seed_group in single_by_group:
        left = single_by_group[seed_group]
        right = multi_by_group[seed_group]
        assert left.split == right.split
        assert left.regime == right.regime
        assert left.episodes == right.episodes
        assert left.reset_constraints == right.reset_constraints
        assert float(left.atoms[0].value) == 5.0
        assert left.atoms[0].kind == right.atoms[0].kind == "agent_speed"


def test_training_matrix_uses_paired_seeds_and_equal_exposure() -> None:
    benchmark = _load("configs/benchmark/tworoom_speed_isolated_v2.yaml")
    protocol = benchmark["training_protocol"]

    assert protocol["paired_training_seeds"] == [3072, 4096, 5120]
    assert protocol["group_sampling"]["M_speed_single_v2"] == {
        "original": 0.5,
        "speed_single_v2": 0.5,
    }
    assert protocol["group_sampling"]["M_speed_multi_v2"] == {
        "original": 0.5,
        "speed_multi_v2": 0.5,
    }
    assert protocol["exposure_contract"]["optimizer_steps"] == 12840
    assert CATALOG_BY_GROUP["speed_single_v2"] == "speed_single_v2"
    assert CATALOG_BY_GROUP["speed_multi_v2"] == "speed_multi_v2"


def test_every_speed_cube_cell_has_full_50_by_6_count() -> None:
    benchmark = _load("configs/benchmark/tworoom_speed_isolated_v2.yaml")
    evaluation = benchmark["evaluation_protocol"]

    assert evaluation["eval_seeds"] == [42, 43, 44, 45, 46, 47]
    assert evaluation["evaluations_per_matrix_cell_per_seed"] == 50
    assert evaluation["evaluations_per_matrix_cell"] == 300
    assert evaluation["query_speeds_per_track"] == 3
    assert evaluation["history_speeds_per_track"] == 3
