from copy import deepcopy
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


# ccb3526 made the formal H3 metadata portable.  The model-visible Lance
# manifests did not change; only local source/staging paths and the hashes of
# metadata records derived from those paths changed.  Keep both complete,
# recorded bundles explicit so this test does not accidentally accept an
# unregistered data identity as a fair PLDM/LeWM control.
_LEGACY_TRAINING_INPUT_METADATA = {
    "original_read_only": (
        "../../data/world_model/quentinll/lewm-tworooms/tworoom.h5"
    ),
    "normalizer_sha256": (
        "7a5be7ea867bced446c1671b0b2c0ff6450ffc61e1a7bdbbfc5eaa0942f635db"
    ),
    "build_report_sha256": (
        "bd3bde3da3bc97c67c4f9eb1ed87f4a41b50c25c9709ce6b64c4f9a69b9c556a"
    ),
    "initialization_config_sha256": (
        "44b5bde83fbc91634ef84acbdba9a75d3436568ed72266c9a5aa057653f9162b"
    ),
    "group_catalog_sha256": {
        "passage_passable": (
            "0cbcc0c29a54184e6884b1046489bdcda9035cfb3a7c1b5e7f39e0cd3eaed727"
        ),
        "passage_blocked": (
            "8a4ee0c475147a0d1e0af9459720f4c0f8b0d3c755871b7670f296d34609ad59"
        ),
        "passage_mixed": (
            "6a0ecfd6f954d27f33a3f4a517213dbf6cc6b3d1bfa34e31421797c9b9e2d61c"
        ),
    },
    "group_report_sha256": {
        "passage_passable": (
            "39703527973037e3ba520a7116d06db02e59845f7a7b06c5fc339de1fe1f0174"
        ),
        "passage_blocked": (
            "3115887559fd52a7359b6e10cf1ebcfe971d7147fefd7f5c56071f234bea245d"
        ),
        "passage_mixed": (
            "11221fd0bde2bb5bab75639cabc43262bffa68894cf8f4cd29b37ec09006afb8"
        ),
    },
}
_PORTABLE_TRAINING_INPUT_METADATA = {
    "original_read_only": "artifacts/upstream/lewm-tworooms/tworoom.h5",
    "normalizer_sha256": (
        "a9e4b443bbac0d7a4e2d9d9f84d40ac40936556ffadfc7af0b0ce4fe4afed42c"
    ),
    "build_report_sha256": (
        "f69a1c83664af800892c026836bf4a9cd0a0b0703cd9c23f4031e8d4d55efbbe"
    ),
    "initialization_config_sha256": (
        "cd9ef9e2efde4527e36b6467d6a4efd120ab466d6a9f8a4aa41cdb7b56c4eddf"
    ),
    "group_catalog_sha256": {
        "passage_passable": (
            "efeb6ab7775ec56c9b12021dcf2d378e0fa55fa65e58820e8434556f8f6c6cdd"
        ),
        "passage_blocked": (
            "b1c79c70b762711aad95dde9d65c65bbf2bad1efe02bb7bf7ee3d5e1375bb010"
        ),
        "passage_mixed": (
            "187915f7e2771df40e88dc5a3b9d44ac87c204967e1ac4e2632046ce7b41ecb2"
        ),
    },
    "group_report_sha256": {
        "passage_passable": (
            "c63333a4e8aa816a50e93a2740fd3b02bb198c8cfa4107bd5dccc6c81e08593a"
        ),
        "passage_blocked": (
            "63e8c7f72be8c7cd17d4da2403a480af8532de1be9b4cdc430cae118a417fe8d"
        ),
        "passage_mixed": (
            "1974969a5dc75a0f535e2941d5eb967324bf27b4bfea36202e2d6af5be55fb15"
        ),
    },
}


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _training_input_metadata(config: dict) -> dict:
    data = config["data"]
    groups = config["data_quality"]["groups"]
    return {
        "original_read_only": data["original_read_only"],
        "normalizer_sha256": data["frozen_normalizer"]["sha256"],
        "build_report_sha256": data["formal_build_report"]["sha256"],
        "initialization_config_sha256": config["training_protocol"][
            "initialization_checkpoint"
        ]["config_sha256"],
        "group_catalog_sha256": {
            name: groups[name]["required_catalog_sha256"]
            for name in sorted(groups)
        },
        "group_report_sha256": {
            name: groups[name]["required_synthesis_report_sha256"]
            for name in sorted(groups)
        },
    }


def _training_input_semantics(config: dict) -> dict:
    """Compare training inputs after the recorded path-only migration."""

    data = deepcopy(config["data"])
    data.pop("original_read_only")
    data["frozen_normalizer"].pop("sha256")
    data["formal_build_report"].pop("sha256")
    groups = deepcopy(config["data_quality"]["groups"])
    for group in groups.values():
        group.pop("required_catalog_sha256")
        group.pop("required_synthesis_report_sha256")
    return {
        "data": data,
        "data_quality_groups": groups,
        "passage_support": config["passage_support"],
        "paired_collection_contract": config["paired_collection_contract"],
        "initialization": {
            key: value
            for key, value in config["training_protocol"][
                "initialization_checkpoint"
            ].items()
            if key != "config_sha256"
        },
    }


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
        assert _training_input_metadata(pldm) in [
            _LEGACY_TRAINING_INPUT_METADATA,
            _PORTABLE_TRAINING_INPUT_METADATA,
        ]
        assert _training_input_metadata(lewm) in [
            _LEGACY_TRAINING_INPUT_METADATA,
            _PORTABLE_TRAINING_INPUT_METADATA,
        ]
        assert _training_input_semantics(pldm) == _training_input_semantics(
            lewm
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
