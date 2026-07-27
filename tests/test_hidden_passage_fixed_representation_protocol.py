from pathlib import Path

import yaml

from scripts.analyze_tworoom_hidden_passage_h3 import _attribution_checks


ROOT = Path(__file__).resolve().parents[1]
TRAINING = (
    ROOT
    / "configs/benchmark/"
    "tworoom_hidden_passage_h3_fixed_representation_training_v1.yaml"
)
TRAIN_SEEN = (
    ROOT
    / "configs/benchmark/"
    "tworoom_hidden_passage_h3_fixed_representation_train_seen_eval_v2.yaml"
)
UNSEEN = (
    ROOT
    / "configs/benchmark/"
    "tworoom_hidden_passage_h3_fixed_representation_validation_v2.yaml"
)
MODEL_ID = "H3_Passage_MixedRules_FrozenRepresentation"
SEEDS = [3072, 4096, 5120]


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_fixed_representation_training_recipe_is_frozen() -> None:
    config = _load(TRAINING)
    protocol = config["training_protocol"]

    assert protocol["frozen_model_modules"] == ["encoder", "projector"]
    assert protocol["force_frozen_modules_eval_mode"] is True
    assert protocol["paired_training_seeds"] == SEEDS
    assert protocol["group_sampling"] == {
        MODEL_ID: {"passage_mixed": 1.0}
    }
    assert protocol["profiles"]["passage_formal"] == {
        "optimizer_steps": 1024,
        "effective_global_batch": 1024,
        "total_logical_draws": 1048576,
    }
    assert protocol["distributed_execution"]["audit_scheduling"] == {
        "policy": "sibling_shared_flock",
        "maximum_concurrency": 8,
        "scope": "per_rank_full_audit_and_fit_start_storage_revalidation",
        "lock_protocol": (
            "contextworld.hidden_passage_h3.audit_scheduling_lock.v2"
        ),
        "lock_order": "release_shared_then_audit_shared",
        "collective_holds_lock": False,
        "topology_scope": "single_node_8gpu",
        "concurrent_training_runs_per_release": 1,
    }
    assert protocol["distributed_execution"]["rank_cpu_affinity"] == {
        "policy": "local_rank_disjoint_contiguous_from_zero",
        "cpus_per_rank": 8,
        "expected_world_size": 8,
        "scope": "full_rank_process",
        "apply_before_stableworldmodel_and_lance_import": True,
    }
    assert config["models"] == [
        {
            "model_id": MODEL_ID,
            "display_name": "固定原始图像表示的双规则模型",
            "training_groups": ["passage_mixed"],
        }
    ]
    assert config["evaluation_gate"]["stage_1_train_seen"][
        "all_three_must_pass"
    ]
    assert config["evaluation_gate"]["stage_2_unseen_door_validation"][
        "locked_until_stage_1_passes"
    ]


def test_each_formal_eval_cell_is_independent_50_by_6() -> None:
    for path in (TRAIN_SEEN, UNSEEN):
        config = _load(path)
        assert config["comparison"]["required_results"] == {
            MODEL_ID: SEEDS
        }
        assert config["comparison"]["checkpoint_training_group"] == {
            MODEL_ID: "passage_mixed"
        }
        assert len(config["evaluation"]["eval_seeds"]) == 6
        assert config["evaluation"]["unique_queries_per_seed"] == 50
        assert config["evaluation"]["unique_queries"] == 300
        assert (
            config["evaluation"]["model_predictions_per_checkpoint"]
            == 900
        )
        assert config["evaluation"]["loss_records_per_checkpoint"] == 1800
        assert config["training_provenance"]["passage_formal"][
            "frozen_model_modules"
        ] == ["encoder", "projector"]
        assert (
            config["gates"]["decision_contract"]
            == "informative_history_rule_switch_v2"
        )
        assert (
            config["metrics"]["no_crossing_attempt_role"]
            == "auxiliary_default_tendency_only"
        )
        assert config["gates"]["paired_bootstrap"][
            "required_metrics"
        ] == [
            "passable/same_vs_other_rule_history",
            "blocked/same_vs_other_rule_history",
            "passable/matching_history_two_target_margin",
            "blocked/matching_history_two_target_margin",
        ]


def test_three_of_three_gate_is_fail_closed() -> None:
    config = _load(TRAIN_SEEN)
    all_pass = {(MODEL_ID, seed): True for seed in SEEDS}
    one_failure = {**all_pass, (MODEL_ID, 4096): False}

    assert all(_attribution_checks(all_pass, config).values())
    assert not all(_attribution_checks(one_failure, config).values())


def test_shell_entry_exposes_the_fixed_recipe() -> None:
    runner = (
        ROOT / "scripts/run_h3_hidden_passage_train.sh"
    ).read_text(encoding="utf-8")
    assert "fixed-mixed)" in runner
    assert "H3_Passage_MixedRules_FrozenRepresentation" in runner
    assert (
        "tworoom_hidden_passage_h3_fixed_representation_training_v1.yaml"
        in runner
    )
