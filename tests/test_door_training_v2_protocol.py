from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import yaml

from contextworld.synthesis.config import (
    build_compiler,
    load_config,
    scenario_requests,
)
from contextworld.synthesis.validator import (
    validate_independent_seed_assignment,
    validate_minimum_episode_start_oracle,
)
from contextworld.synthesis.reset_constraints import (
    apply_tworoom_reset_constraints,
)
from contextworld.training.tworoom_data import CATALOG_BY_GROUP


ROOT = Path(__file__).resolve().parents[1]
FIXED_CONFIG = (
    ROOT / "configs/synthesis/tworoom_door_fixed49_matched_v2.yaml"
)
MULTI_CONFIG = ROOT / "configs/synthesis/tworoom_door_multi_v2.yaml"
TRAINING_CONFIG = ROOT / "configs/benchmark/tworoom_door_training_v2.yaml"


def _yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _requests(path: Path):
    return scenario_requests(_yaml(path))


def test_door_recipes_have_exact_paired_collection_budget() -> None:
    fixed = _yaml(FIXED_CONFIG)
    multi = _yaml(MULTI_CONFIG)
    fixed_requests = _requests(FIXED_CONFIG)
    multi_requests = _requests(MULTI_CONFIG)

    assert fixed["seed"] == multi["seed"]
    assert fixed["scenario_generation_seed"] == multi[
        "scenario_generation_seed"
    ]
    assert fixed["collection"] == multi["collection"]
    assert fixed["controlled_constants"] == multi["controlled_constants"] == {
        "agent_speed": 5.0
    }
    assert len(fixed_requests) == len(multi_requests) == 608
    assert Counter(row.split for row in fixed_requests) == {
        "train": 512,
        "val": 96,
    }
    assert Counter(row.split for row in multi_requests) == {
        "train": 512,
        "val": 96,
    }
    assert sum(row.episodes for row in fixed_requests if row.split == "train") == (
        512 * 32
    )
    assert sum(row.episodes for row in fixed_requests if row.split == "val") == (
        96 * 16
    )
    assert sum(row.episodes for row in multi_requests if row.split == "train") == (
        512 * 32
    )
    assert sum(row.episodes for row in multi_requests if row.split == "val") == (
        96 * 16
    )

    fixed_by_group = {row.seed_group: row for row in fixed_requests}
    multi_by_group = {row.seed_group: row for row in multi_requests}
    assert len(fixed_by_group) == len(multi_by_group) == 608
    assert fixed_by_group.keys() == multi_by_group.keys()
    for seed_group, left in fixed_by_group.items():
        right = multi_by_group[seed_group]
        assert left.split == right.split
        assert left.regime == right.regime
        assert left.episodes == right.episodes
        assert left.reset_constraints == right.reset_constraints
        assert left.atoms[0].kind == right.atoms[0].kind == "door_position"
        assert float(left.atoms[0].value) == 49.0


def test_door_recipes_assign_one_independent_seed_block_per_scenario() -> None:
    for config_path in (FIXED_CONFIG, MULTI_CONFIG):
        config = load_config(config_path)
        requests = scenario_requests(config)
        scenarios = build_compiler(config, ROOT).compile_all(requests)
        audit = validate_independent_seed_assignment(
            scenarios, config["validation"]["independent_seed_assignment"]
        )
        assert audit["passed"]
        assert audit["splits"]["train"]["observed"]["seed_groups"] == 512
        assert audit["splits"]["val"]["observed"]["seed_groups"] == 96
        assert audit["splits"]["train"]["observed"][
            "one_scenario_per_seed_group"
        ]
        assert audit["splits"]["val"]["observed"][
            "one_scenario_per_seed_group"
        ]

    fixed = load_config(FIXED_CONFIG)
    multi = load_config(MULTI_CONFIG)
    fixed_compiled = build_compiler(fixed, ROOT).compile_all(
        scenario_requests(fixed)
    )
    multi_compiled = build_compiler(multi, ROOT).compile_all(
        scenario_requests(multi)
    )
    fixed_seeds = {
        row.seed_group: (row.env_seed, row.policy_seed)
        for row in fixed_compiled
    }
    multi_seeds = {
        row.seed_group: (row.env_seed, row.policy_seed)
        for row in multi_compiled
    }
    assert fixed_seeds == multi_seeds


def test_paired_seed_groups_replay_the_same_reset_and_goal() -> None:
    from stable_worldmodel.envs.two_room.env import TwoRoomEnv

    fixed = load_config(FIXED_CONFIG)
    multi = load_config(MULTI_CONFIG)
    fixed_rows = {
        row.seed_group: row
        for row in build_compiler(fixed, ROOT).compile_all(
            scenario_requests(fixed)
        )
    }
    multi_rows = {
        row.seed_group: row
        for row in build_compiler(multi, ROOT).compile_all(
            scenario_requests(multi)
        )
    }
    env = TwoRoomEnv(render_mode="rgb_array")
    try:
        for seed_group, left in fixed_rows.items():
            right = multi_rows[seed_group]
            apply_tworoom_reset_constraints(env, left.reset_constraints)
            left_observation, _ = env.reset(
                seed=left.env_seed,
                options={
                    "variation": left.variation,
                    "variation_values": left.variation_values,
                },
            )
            apply_tworoom_reset_constraints(env, right.reset_constraints)
            right_observation, _ = env.reset(
                seed=right.env_seed,
                options={
                    "variation": right.variation,
                    "variation_values": right.variation_values,
                },
            )
            np.testing.assert_array_equal(
                left_observation[:4], right_observation[:4]
            )
    finally:
        env.close()


def test_door_recipes_read_back_the_pinned_default_speed() -> None:
    for config_path in (FIXED_CONFIG, MULTI_CONFIG):
        config = load_config(config_path)
        scenario = build_compiler(config, ROOT).compile_all(
            scenario_requests(config)[:1]
        )[0]
        oracle = validate_minimum_episode_start_oracle(
            [scenario], config["validation"]["minimum_episode_start_oracle"]
        )
        assert oracle["passed"]
        assert oracle["expected_agent_speed"] == 5.0
        assert oracle["observed_agent_speeds"] == [5.0]
        assert oracle["agent_speed_readback_passed"]
        assert oracle["agent_speed_mismatches"] == []


def test_loader_validation_uses_training_doors_and_not_eval_holdouts() -> None:
    fixed = _requests(FIXED_CONFIG)
    multi = _requests(MULTI_CONFIG)
    benchmark = _yaml(TRAINING_CONFIG)
    heldout = set(benchmark["door_support"]["eval_heldout_values"])

    fixed_train = {
        int(row.atoms[0].value) for row in fixed if row.split == "train"
    }
    fixed_val = {int(row.atoms[0].value) for row in fixed if row.split == "val"}
    multi_train = {
        int(row.atoms[0].value) for row in multi if row.split == "train"
    }
    multi_val = {int(row.atoms[0].value) for row in multi if row.split == "val"}

    assert fixed_train == fixed_val == {49}
    assert multi_train == multi_val == set(
        benchmark["door_support"]["multi_synthetic_train"]
    )
    assert not heldout & (fixed_train | fixed_val | multi_train | multi_val)

    multi_val_counts = Counter(
        int(row.atoms[0].value) for row in multi if row.split == "val"
    )
    assert set(multi_val_counts.values()) == {6}


def test_door_groups_are_wired_to_the_additive_training_profile() -> None:
    benchmark = _yaml(TRAINING_CONFIG)
    protocol = benchmark["training_protocol"]

    assert protocol["profile"] == "additive"
    assert protocol["paired_training_seeds"] == [3072, 4096, 5120]
    assert protocol["group_sampling"]["M_door_fixed49_v2"] == {
        "original": 0.5,
        "door_fixed49_v2": 0.5,
    }
    assert protocol["group_sampling"]["M_door_multi_v2"] == {
        "original": 0.5,
        "door_multi_v2": 0.5,
    }
    assert benchmark["models"] == [
        {
            "model_id": "M_door_fixed49_v2",
            "display_name": "History-3 固定门位置匹配控制",
            "training_groups": ["original", "door_fixed49_v2"],
        },
        {
            "model_id": "M_door_multi_v2",
            "display_name": "History-3 多门位置目标",
            "training_groups": ["original", "door_multi_v2"],
        },
    ]
    assert CATALOG_BY_GROUP["door_fixed49_v2"] == "door_fixed49_v2"
    assert CATALOG_BY_GROUP["door_multi_v2"] == "door_multi_v2"
    pairing = benchmark["paired_collection_contract"]
    assert pairing["reset_and_goal_pairing"] == "exact_by_seed_group"
    assert "may diverge" in pairing["expert_actions"]
    assert "identical action sequences are not required" in pairing[
        "expert_actions"
    ]
