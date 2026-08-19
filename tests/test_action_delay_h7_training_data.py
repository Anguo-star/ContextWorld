from __future__ import annotations

import copy
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml

from contextworld.evaluation.action_delay_h7_data import (
    DEFAULT_CLIPS_PER_EPISODE,
    apply_formal_clip_filter,
    build_shard_plans,
)
from scripts.train_tworoom_step1 import (
    _apply_initialization_checkpoint,
    _initialization_checkpoint_spec,
    _sha256,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_CONFIG = (
    ROOT
    / "configs/benchmark/tworoom_action_delay_h7_training_data_v1.yaml"
)


def _data_config() -> dict:
    return yaml.safe_load(DATA_CONFIG.read_text(encoding="utf-8"))


def test_h7_single_and_multi_plans_are_exactly_paired(tmp_path: Path) -> None:
    config = copy.deepcopy(_data_config())
    config["output_root"] = str(tmp_path / "release")
    config["counts"] = {
        "train": {
            "shards": 3,
            "episodes_per_shard": 160,
            "clips_per_group": 480,
        },
        "val": {
            "shards": 3,
            "episodes_per_shard": 160,
            "clips_per_group": 480,
        },
    }
    plans = build_shard_plans(config, repo_root=ROOT)

    single = plans["action_delay_single"]
    multi = plans["action_delay_multi"]
    assert {shard.delay_steps for shard in single} == {4}
    assert {shard.delay_steps for shard in multi} == {0, 4, 8}
    assert len(single) == len(multi) == 6
    for left, right in zip(single, multi, strict=True):
        assert left.split == right.split
        assert left.shard_index == right.shard_index
        assert [episode.template for episode in left.episodes] == [
            episode.template for episode in right.episodes
        ]


def test_h7_formal_clip_filter_rejects_ten_misaligned_starts() -> None:
    dataset = SimpleNamespace(
        lengths=[50, 50],
        clip_indices=[
            (episode, start)
            for episode in range(2)
            for start in range(DEFAULT_CLIPS_PER_EPISODE)
        ],
    )

    audit = apply_formal_clip_filter(dataset)

    assert dataset.clip_indices == [(0, 0), (1, 0)]
    assert audit == {
        "default_clip_count": 22,
        "formal_clip_count": 2,
        "allowed_raw_starts": [0],
        "removed_misaligned_clips": 20,
        "passed": True,
    }


class _TemporalModel(torch.nn.Module):
    def __init__(self, history: int) -> None:
        super().__init__()
        self.predictor = torch.nn.Module()
        self.predictor.register_parameter(
            "pos_embedding",
            torch.nn.Parameter(torch.randn(1, history, 4)),
        )
        self.projection = torch.nn.Linear(4, 3)


def test_h3_checkpoint_expands_only_temporal_position_table(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "weights.pt"
    checkpoint.write_bytes(b"model-only-checkpoint")
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    benchmark = tmp_path / "benchmark.yaml"
    benchmark.write_text(
        yaml.safe_dump(
            {
                "training_protocol": {
                    "initialization_checkpoint": {
                        "path": str(checkpoint),
                        "sha256": _sha256(checkpoint),
                        "role": (
                            "model_weight_initialization_only_not_resume"
                        ),
                        "temporal_adaptation": {
                            "parameter": "predictor.pos_embedding",
                            "strategy": "linear_interpolation",
                            "source_history_tokens": 3,
                            "target_history_tokens": 7,
                            "align_corners": True,
                            "source_anchor_target_indices": [0, 3, 6],
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    specification = _initialization_checkpoint_spec(
        Namespace(
            initialization_checkpoint=None,
            initialization_checkpoint_sha256=None,
        ),
        benchmark_config=benchmark,
    )
    assert specification is not None

    torch.manual_seed(21)
    source = _TemporalModel(3)
    torch.manual_seed(22)
    target = _TemporalModel(7)
    fake_swm = SimpleNamespace(
        wm=SimpleNamespace(
            utils=SimpleNamespace(
                load_pretrained=lambda path, cache_dir: source
            )
        )
    )
    audit = _apply_initialization_checkpoint(
        target,
        swm=fake_swm,
        specification=specification,
        cache_dir=tmp_path,
        resume_checkpoint=None,
    )

    assert audit["applied"] is True
    assert audit["state_exact"] is False
    assert audit["temporal_adaptation_audit"]["passed"] is True
    assert audit["temporal_adaptation_audit"][
        "all_other_parameter_tensors_exact"
    ] is True
    assert torch.equal(target.projection.weight, source.projection.weight)
    assert torch.equal(target.projection.bias, source.projection.bias)
    for source_index, target_index in enumerate((0, 3, 6)):
        assert torch.equal(
            target.predictor.pos_embedding[:, target_index],
            source.predictor.pos_embedding[:, source_index],
        )
