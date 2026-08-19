from __future__ import annotations

import copy
from pathlib import Path

import torch
import yaml

from contextworld.evaluation.action_delay_h7_paired_data import (
    DELAYS,
    audit_paired_triplets,
    build_paired_shard_plans,
)
from scripts.train_tworoom_step1 import (
    _temporal_prediction_loss_spec,
    _weighted_transition_mse,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs/benchmark/"
    "tworoom_action_delay_h7_paired_training_data_v1.yaml"
)
LEWM_CONFIG = (
    ROOT
    / "configs/benchmark/"
    "tworoom_action_delay_h7_paired_lewm_v1.yaml"
)
PLDM_CONFIG = (
    ROOT
    / "configs/benchmark/"
    "tworoom_action_delay_h7_paired_pldm_v1.yaml"
)
RELEASE_CONFIG = (
    ROOT
    / "configs/benchmark/"
    "tworoom_action_delay_icl_release_v1.yaml"
)


def _config() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def test_paired_plans_have_one_exact_query_triplet(tmp_path: Path) -> None:
    config = copy.deepcopy(_config())
    config["output_root"] = str(tmp_path / "release")
    for split in ("train", "val"):
        config["counts"][split] = {
            "paired_shards": 1,
            "shards": 3,
            "episodes_per_shard": 160,
            "clips": 480,
        }

    plans = build_paired_shard_plans(config, repo_root=ROOT)

    assert len(plans) == 6
    for split in ("train", "val"):
        rows = [row for row in plans if row.split == split]
        assert [row.delay_steps for row in rows] == list(DELAYS)
        assert [row.shard_index for row in rows] == [0, 1, 2]
        reference = [episode.template for episode in rows[0].episodes]
        assert all(
            [episode.template for episode in row.episodes] == reference
            for row in rows[1:]
        )


def test_pair_audit_requires_same_query_and_distinct_next_frames(
    tmp_path: Path,
) -> None:
    config = copy.deepcopy(_config())
    config["output_root"] = str(tmp_path / "release")
    for split in ("train", "val"):
        config["counts"][split] = {
            "paired_shards": 1,
            "shards": 3,
            "episodes_per_shard": 160,
            "clips": 480,
        }
    shards = build_paired_shard_plans(config, repo_root=ROOT)
    audits = []
    for shard in shards:
        delay = shard.delay_steps
        audits.append(
            {
                "initial_pixels_sha256": ["initial"] * 160,
                "query_pixels_sha256": ["query"] * 160,
                "model_action_sha256": ["action"] * 160,
                "target_pixels_sha256": [f"target-{delay}"] * 160,
            }
        )

    result = audit_paired_triplets(shards, audits)

    assert result["pair_shards"] == 2
    assert result["query_triplets"] == 320
    assert result["physical_clips"] == 960
    assert result["passed"] is True

    audits[1]["query_pixels_sha256"][0] = "changed"
    failed = audit_paired_triplets(shards, audits)
    assert failed["passed"] is False
    assert failed["checks"]["query_pixels_exact"] is False


def test_temporal_prediction_loss_makes_last_transition_majority(
    tmp_path: Path,
) -> None:
    benchmark = tmp_path / "benchmark.yaml"
    benchmark.write_text(
        yaml.safe_dump(
            {
                "training_protocol": {
                    "temporal_prediction_loss": {
                        "mode": "normalized_transition_weights",
                        "transition_weights": [1, 1, 1, 1, 1, 1, 7],
                        "normalization": "divide_by_sum_of_weights",
                        "applies_to": "all_training_groups",
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    specification = _temporal_prediction_loss_spec(
        benchmark,
        predicted_transitions=7,
    )

    assert specification["configured"] is True
    assert specification["normalized_transition_weight"][-1] == 7 / 13
    prediction = torch.zeros(2, 7, 3)
    target = torch.zeros_like(prediction)
    target[:, -1] = 1.0
    losses = _weighted_transition_mse(
        prediction,
        target,
        specification["transition_weights"],
    )
    assert torch.isclose(losses["unweighted"], torch.tensor(1 / 7))
    assert torch.isclose(losses["weighted"], torch.tensor(7 / 13))
    assert torch.isclose(
        losses["final_transition"],
        torch.tensor(1.0),
    )


def test_paired_lewm_and_pldm_change_only_model_objective() -> None:
    lewm = yaml.safe_load(LEWM_CONFIG.read_text(encoding="utf-8"))
    pldm = yaml.safe_load(PLDM_CONFIG.read_text(encoding="utf-8"))
    release = yaml.safe_load(RELEASE_CONFIG.read_text(encoding="utf-8"))

    assert lewm["data"] == pldm["data"]
    assert lewm["data_quality"] == pldm["data_quality"]
    assert lewm["stage_contract"] == pldm["stage_contract"]
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

    # These files specify only the coarse, three-delay training stage.  The
    # complete Public evaluation is intentionally declared once in the
    # capability release, after the full-delay refinement stage.  Adding a
    # second validation block here would change the historical stage recipes
    # without changing what was actually evaluated.
    assert "validation" not in lewm
    assert "validation" not in pldm
    assert "gates" not in lewm
    assert "gates" not in pldm
    assert release["evaluation"]["catalog"] == (
        "artifacts/evaluation/history7/action_delay_validation_v1/catalog.json"
    )
    recipes = release["training"]["recipes"]
    assert recipes["lewm_control"]["stages"][0]["config"] == (
        "configs/benchmark/tworoom_action_delay_h7_paired_lewm_v1.yaml"
    )
    assert recipes["pldm_reference"]["stages"][0]["config"] == (
        "configs/benchmark/tworoom_action_delay_h7_paired_pldm_v1.yaml"
    )
