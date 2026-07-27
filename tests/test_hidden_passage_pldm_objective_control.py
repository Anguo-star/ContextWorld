from pathlib import Path

import yaml

from contextworld.benchmarks.adapters import (
    StableWorldModelLeWMAdapter,
    StableWorldModelPLDMAdapter,
)
from scripts.analyze_tworoom_hidden_passage_h3 import _attribution_checks
from scripts.train_tworoom_step1 import _training_method


ROOT = Path(__file__).resolve().parents[1]
JOINT = (
    ROOT
    / "configs/benchmark/"
    "tworoom_hidden_passage_h3_pldm_training_v1.yaml"
)
FIXED = (
    ROOT
    / "configs/benchmark/"
    "tworoom_hidden_passage_h3_pldm_fixed_representation_training_v1.yaml"
)
VALIDATION = (
    ROOT
    / "configs/benchmark/"
    "tworoom_hidden_passage_h3_pldm_validation_v1.yaml"
)
LEWM_JOINT = (
    ROOT
    / "configs/benchmark/"
    "tworoom_hidden_passage_h3_training_v1.yaml"
)
LEWM_FIXED = (
    ROOT
    / "configs/benchmark/"
    "tworoom_hidden_passage_h3_fixed_representation_training_v1.yaml"
)

JOINT_ID = "H3_Passage_MixedRules_PLDMObjective"
FIXED_ID = "H3_Passage_MixedRules_PLDMObjective_FrozenRepresentation"
SEEDS = [3072, 4096, 5120]


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_pldm_control_changes_only_registered_training_factors() -> None:
    pldm_joint = _load(JOINT)
    pldm_fixed = _load(FIXED)
    lewm_joint = _load(LEWM_JOINT)
    lewm_fixed = _load(LEWM_FIXED)

    for pldm, lewm in (
        (pldm_joint, lewm_joint),
        (pldm_fixed, lewm_fixed),
    ):
        assert pldm["stable_worldmodel"] == lewm["stable_worldmodel"]
        assert pldm["data"] == lewm["data"]
        assert pldm["data_quality"] == lewm["data_quality"]
        assert pldm["passage_support"] == lewm["passage_support"]
        assert (
            pldm["training_protocol"]["initialization_checkpoint"]
            == lewm["training_protocol"]["initialization_checkpoint"]
        )
        assert (
            pldm["training_protocol"]["profiles"]
            == lewm["training_protocol"]["profiles"]
        )
        assert (
            pldm["training_protocol"]["paired_training_seeds"]
            == SEEDS
        )
        assert pldm["training_protocol"]["training_method"] == "pldm"

    assert (
        pldm_joint["training_protocol"].get("frozen_model_modules", [])
        == []
    )
    assert pldm_fixed["training_protocol"]["frozen_model_modules"] == [
        "encoder",
        "projector",
    ]
    assert pldm_joint["training_protocol"]["group_sampling"] == {
        JOINT_ID: {"passage_mixed": 1.0}
    }
    assert pldm_fixed["training_protocol"]["group_sampling"] == {
        FIXED_ID: {"passage_mixed": 1.0}
    }


def test_pldm_validation_reuses_frozen_50_by_6_catalog() -> None:
    config = _load(VALIDATION)
    source = _load(
        ROOT
        / "configs/benchmark/"
        "tworoom_hidden_passage_h3_validation_v2.yaml"
    )
    rule_switch = _load(
        ROOT
        / "configs/benchmark/"
        "tworoom_hidden_passage_h3_fixed_representation_validation_v2.yaml"
    )

    assert config["benchmark"] == source["benchmark"]
    assert config["data"] == source["data"]
    assert config["evaluation"] == source["evaluation"]
    assert config["gates"] == rule_switch["gates"]
    assert config["artifacts"]["catalog"] == source["artifacts"]["catalog"]
    assert (
        config["artifacts"]["training_exclusion_manifest"]
        == source["artifacts"]["training_exclusion_manifest"]
    )
    assert config["decision_protocol"]["name"] == (
        "informative_history_rule_switch_v2"
    )
    assert config["metrics"]["no_crossing_attempt_role"] == (
        "auxiliary_default_tendency_only"
    )
    assert config["adapter"]["implementation"] == (
        "StableWorldModelPLDMAdapter"
    )
    assert config["comparison"]["required_results"] == {
        JOINT_ID: SEEDS,
        FIXED_ID: SEEDS,
    }
    assert len(config["evaluation"]["eval_seeds"]) == 6
    assert config["evaluation"]["unique_queries_per_seed"] == 50
    assert config["evaluation"]["unique_queries"] == 300
    assert config["evaluation"]["model_predictions_per_checkpoint"] == 900
    assert config["evaluation"]["loss_records_per_checkpoint"] == 1800
    assert config["comparison"]["attribution_gate"][
        "pldm_joint_and_fixed_required_training_seeds"
    ] == 3


def test_pldm_attribution_requires_both_recipes_three_of_three() -> None:
    config = _load(VALIDATION)
    all_pass = {
        (model_id, seed): True
        for model_id in (JOINT_ID, FIXED_ID)
        for seed in SEEDS
    }
    one_failure = {**all_pass, (JOINT_ID, 4096): False}

    assert all(_attribution_checks(all_pass, config).values())
    assert not all(_attribution_checks(one_failure, config).values())


def test_pldm_provenance_is_model_specific_and_fail_closed() -> None:
    config = _load(VALIDATION)
    provenance = config["training_provenance"]["passage_formal_by_model"]

    assert set(provenance) == {JOINT_ID, FIXED_ID}
    assert provenance[JOINT_ID]["frozen_model_modules"] == []
    assert provenance[FIXED_ID]["frozen_model_modules"] == [
        "encoder",
        "projector",
    ]
    assert provenance[JOINT_ID]["training_benchmark_config"] == (
        "configs/benchmark/"
        "tworoom_hidden_passage_h3_pldm_training_v1.yaml"
    )
    assert provenance[FIXED_ID]["training_benchmark_config"] == (
        "configs/benchmark/"
        "tworoom_hidden_passage_h3_pldm_fixed_representation_training_v1.yaml"
    )


def test_pldm_training_method_and_adapter_are_explicit() -> None:
    assert _training_method(JOINT) == "pldm"
    assert _training_method(FIXED) == "pldm"
    assert _training_method(LEWM_JOINT) == "lewm"
    assert issubclass(
        StableWorldModelPLDMAdapter,
        StableWorldModelLeWMAdapter,
    )
    assert StableWorldModelPLDMAdapter.adapter_id == (
        "stable_worldmodel_pldm_v1"
    )


def test_shell_entry_exposes_both_pldm_controls() -> None:
    runner = (
        ROOT / "scripts/run_h3_hidden_passage_train.sh"
    ).read_text(encoding="utf-8")

    assert "pldm-mixed)" in runner
    assert "pldm-fixed-mixed)" in runner
    assert JOINT_ID in runner
    assert FIXED_ID in runner
    assert "tworoom_hidden_passage_h3_pldm_training_v1.yaml" in runner
    assert (
        "tworoom_hidden_passage_h3_pldm_fixed_representation_training_v1.yaml"
        in runner
    )
