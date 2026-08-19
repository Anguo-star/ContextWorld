from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from contextworld.evaluation.action_delay_h7_paired_data import (
    audit_paired_triplets,
    build_paired_shard_plans,
)
from contextworld.training.tworoom_data import _factor_balanced_group


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs/benchmark/"
    "tworoom_action_delay_h7_full_range_training_data_v2.yaml"
)
CORE_TRAINING_CONFIG = (
    ROOT
    / "configs/benchmark/"
    "tworoom_action_delay_h7_core_training_data_v3.yaml"
)
LEWM_CONFIG = (
    ROOT
    / "configs/benchmark/"
    "tworoom_action_delay_h7_full_range_lewm_v2.yaml"
)
PLDM_CONFIG = (
    ROOT
    / "configs/benchmark/"
    "tworoom_action_delay_h7_full_range_pldm_v2.yaml"
)
CORE_LEWM_CONFIG = (
    ROOT
    / "configs/benchmark/"
    "tworoom_action_delay_h7_core_lewm_v3.yaml"
)
CORE_PLDM_CONFIG = (
    ROOT
    / "configs/benchmark/"
    "tworoom_action_delay_h7_core_pldm_v3.yaml"
)
CURRICULUM_LEWM_CONFIG = (
    ROOT
    / "configs/benchmark/"
    "tworoom_action_delay_h7_curriculum_lewm_v4.yaml"
)
CURRICULUM_PLDM_CONFIG = (
    ROOT
    / "configs/benchmark/"
    "tworoom_action_delay_h7_curriculum_pldm_v4.yaml"
)
DELAYS = tuple(range(11))


def _small_config(tmp_path: Path) -> dict:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    config = copy.deepcopy(config)
    config["output_root"] = str(tmp_path / "release")
    for split in ("train", "val"):
        config["counts"][split] = {
            "paired_shards": 1,
            "shards": 11,
            "episodes_per_shard": 160,
            "clips": 1760,
        }
    return config


def _audits(shards: list) -> list[dict]:
    return [
        {
            "initial_pixels_sha256": ["initial"] * 160,
            "query_pixels_sha256": ["query"] * 160,
            "model_action_sha256": ["action"] * 160,
            "target_pixels_sha256": [
                f"target-{min(int(shard.delay_steps), 5)}"
            ]
            * 160,
        }
        for shard in shards
    ]


def test_full_range_plan_is_one_eleven_delay_bundle_per_split(
    tmp_path: Path,
) -> None:
    plans = build_paired_shard_plans(
        _small_config(tmp_path),
        repo_root=ROOT,
    )

    assert len(plans) == 22
    for split in ("train", "val"):
        rows = [row for row in plans if row.split == split]
        assert [row.delay_steps for row in rows] == list(DELAYS)
        assert [row.shard_index for row in rows] == list(range(11))
        reference = [episode.template for episode in rows[0].episodes]
        assert all(
            [episode.template for episode in row.episodes] == reference
            for row in rows[1:]
        )


def test_full_range_audit_uses_true_one_step_physical_groups(
    tmp_path: Path,
) -> None:
    shards = build_paired_shard_plans(
        _small_config(tmp_path),
        repo_root=ROOT,
    )
    audits = _audits(shards)

    result = audit_paired_triplets(
        shards,
        audits,
        delays=DELAYS,
    )

    assert result["query_bundles"] == 320
    assert result["physical_clips"] == 3520
    assert result["physical_next_state_groups"] == [0, 1, 2, 3, 4, 5]
    assert result["passed"] is True

    delay_six = next(
        index
        for index, shard in enumerate(shards)
        if shard.split == "train" and shard.delay_steps == 6
    )
    audits[delay_six]["target_pixels_sha256"][0] = "wrong"
    failed = audit_paired_triplets(
        shards,
        audits,
        delays=DELAYS,
    )
    assert failed["passed"] is False
    assert (
        failed["checks"]["next_frame_physical_equivalence_groups_exact"]
        is False
    )


def test_formal_full_range_counts_are_exact() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))

    assert config["protocol"]["training_delay_values"] == list(DELAYS)
    assert config["counts"]["train"] == {
        "paired_shards": 10,
        "shards": 110,
        "episodes_per_shard": 160,
        "clips": 17600,
    }
    assert config["counts"]["val"] == {
        "paired_shards": 2,
        "shards": 22,
        "episodes_per_shard": 160,
        "clips": 3520,
    }


def test_full_range_lewm_and_pldm_differ_only_by_model_family() -> None:
    lewm = yaml.safe_load(LEWM_CONFIG.read_text(encoding="utf-8"))
    pldm = yaml.safe_load(PLDM_CONFIG.read_text(encoding="utf-8"))

    assert lewm["data"] == pldm["data"]
    assert lewm["data_quality"] == pldm["data_quality"]
    assert lewm["validation"] == pldm["validation"]
    for key in (
        "history_tokens",
        "num_preds",
        "raw_steps_per_action_block",
        "model_visible_fields",
        "initialization_checkpoint",
        "temporal_prediction_loss",
        "paired_training_seeds",
        "early_stopping",
        "checkpoint_selection",
        "distributed_execution",
        "budget",
    ):
        assert lewm["training_protocol"][key] == pldm[
            "training_protocol"
        ][key]
    assert lewm["training_protocol"]["training_method"] == "lewm"
    assert pldm["training_protocol"]["training_method"] == "pldm"
    assert list(
        lewm["training_protocol"]["group_sampling"].values()
    ) == list(pldm["training_protocol"]["group_sampling"].values())


def test_training_sampler_can_balance_one_step_physical_groups() -> None:
    paths = [Path(f"/tmp/delay-{delay}") for delay in DELAYS]
    scenarios = [[delay, delay] for delay in DELAYS]
    factors = {
        path: {"action.delay_steps": delay}
        for path, delay in zip(paths, DELAYS, strict=True)
    }
    groups = [[0], [1], [2], [3], [4], [5, 6, 7, 8, 9, 10]]

    balanced, metadata = _factor_balanced_group(
        paths,
        scenarios,
        factors_by_path=factors,
        factor_key="action.delay_steps",
        factor_value_groups=groups,
    )

    assert len(balanced) == 72
    assert metadata["balance_groups"] == 6
    assert metadata["factor_values"] == 11
    assert sorted(metadata["raw_clips_per_factor"].values()) == [
        2,
        2,
        2,
        2,
        2,
        12,
    ]
    assert [
        sum(
            balanced.locate(index)[0] == group
            for index in range(len(balanced))
        )
        for group in range(6)
    ] == [12] * 6

    with pytest.raises(ValueError, match="multiple balance groups"):
        _factor_balanced_group(
            paths,
            scenarios,
            factors_by_path=factors,
            factor_key="action.delay_steps",
            factor_value_groups=[[0, 1], [1, 2]],
        )


def test_v3_data_restores_query_diversity_before_training() -> None:
    config = yaml.safe_load(
        CORE_TRAINING_CONFIG.read_text(encoding="utf-8")
    )

    assert config["counts"]["train"] == {
        "paired_shards": 32,
        "shards": 352,
        "episodes_per_shard": 160,
        "clips": 56320,
    }
    assert config["counts"]["val"] == {
        "paired_shards": 6,
        "shards": 66,
        "episodes_per_shard": 160,
        "clips": 10560,
    }
    assert config["protocol"]["one_step_physical_groups"] == [
        [0],
        [1],
        [2],
        [3],
        [4],
        [5, 6, 7, 8, 9, 10],
    ]
    assert (
        config["protocol"]["official_training_sampling"]
        == "equal_weight_across_six_physical_groups"
    )


def test_v3_lewm_and_pldm_share_data_sampling_and_budget() -> None:
    lewm = yaml.safe_load(CORE_LEWM_CONFIG.read_text(encoding="utf-8"))
    pldm = yaml.safe_load(CORE_PLDM_CONFIG.read_text(encoding="utf-8"))

    assert lewm["data"] == pldm["data"]
    assert lewm["data_quality"] == pldm["data_quality"]
    assert lewm["validation"] == pldm["validation"]
    for key in (
        "history_tokens",
        "num_preds",
        "raw_steps_per_action_block",
        "model_visible_fields",
        "training_profile",
        "initialization_checkpoint",
        "temporal_prediction_loss",
        "paired_training_seeds",
        "early_stopping",
        "checkpoint_selection",
        "physical_group_sampling",
        "budget",
    ):
        assert lewm["training_protocol"][key] == pldm[
            "training_protocol"
        ][key]
    assert (
        lewm["training_protocol"]["training_method"],
        pldm["training_protocol"]["training_method"],
    ) == ("lewm", "pldm")


def test_v4_curriculum_is_same_data_one_step_family_control() -> None:
    lewm = yaml.safe_load(
        CURRICULUM_LEWM_CONFIG.read_text(encoding="utf-8")
    )
    pldm = yaml.safe_load(
        CURRICULUM_PLDM_CONFIG.read_text(encoding="utf-8")
    )

    assert lewm["data"] == pldm["data"]
    assert lewm["data_quality"] == pldm["data_quality"]
    assert lewm["validation"] == pldm["validation"]
    for config in (lewm, pldm):
        protocol = config["training_protocol"]
        assert protocol["history_tokens"] == 7
        assert protocol["num_preds"] == 1
        assert protocol["curriculum"]["stage_1"]["data_delays"] == [
            0,
            4,
            8,
        ]
        assert protocol["curriculum"]["stage_2"]["data_delays"] == list(
            DELAYS
        )
        assert protocol["curriculum"]["stage_2"][
            "one_step_physical_groups"
        ] == [[0], [1], [2], [3], [4], [5, 6, 7, 8, 9, 10]]
        assert protocol["budget"]["stage_2_optimizer_steps"] == 1024
        assert set(
            protocol["curriculum"]["stage_1"][
                "checkpoint_by_training_seed"
            ]
        ) == {"3072", "4096", "5120"}
    assert (
        lewm["training_protocol"]["training_method"],
        pldm["training_protocol"]["training_method"],
    ) == ("lewm", "pldm")
    assert list(
        lewm["training_protocol"]["group_sampling"].values()
    ) == list(pldm["training_protocol"]["group_sampling"].values())
