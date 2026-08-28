"""Focused tests for the flat, relation-preserving COJA data bridge."""

from __future__ import annotations

import torch

from contextworld.training.groups import LogicalGroupDataset, RelationBatchSampler
from contextworld.training.stablewm_bundle import (
    CONDITIONAL_JOINT_GROUP_COLUMN,
    _RuntimeDataset,
    _paired_episode_relations,
)


class _Rows:
    def __init__(self, count: int, source: int):
        self.count = count
        self.source = source
        self.transform = None

    @property
    def column_names(self):
        return ["pixels", "action"]

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        sample = {
            "pixels": torch.tensor([self.source, int(index)]),
            "action": torch.tensor([float(index)]),
        }
        return self.transform(sample) if self.transform else sample

    def __getitems__(self, indices):
        return [self[index] for index in indices]

    def get_dim(self, column):
        if column != "action":
            raise KeyError(column)
        return 1

    def get_col_data(self, column):
        if column != "action":
            raise KeyError(column)
        return torch.arange(self.count).view(-1, 1).numpy()


def _runtime():
    original = _Rows(24, source=0)
    synthetic = _Rows(12, source=1)
    mixture = LogicalGroupDataset(
        {"original": original, "synthetic": synthetic},
        {"original": 0.5, "synthetic": 0.5},
        epoch_size=48,
    )
    relations = [(index, index + 1) for index in range(0, 12, 2)]
    return _RuntimeDataset(
        mixture,
        [original, synthetic],
        normalizer_source=original,
        conditional_relations=relations,
    )


def test_split_preserves_relations_and_exact_mixture():
    runtime = _runtime()
    train, validation = runtime.split_for_training(
        train_fraction=0.75,
        generator=torch.Generator().manual_seed(7),
    )
    assert len(train) == 36
    assert len(validation) == 12
    assert train.singles.numel() == 18
    assert train.relations.shape == (9, 2)
    assert set(train.column_names) == {
        "pixels",
        "action",
        CONDITIONAL_JOINT_GROUP_COLUMN,
    }


def test_configured_loader_is_flat_and_keeps_each_pair_together(monkeypatch):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    train, _ = _runtime().split_for_training(
        train_fraction=0.75,
        generator=torch.Generator().manual_seed(7),
    )
    config = train.configure_train_loader(
        {
            "batch_size": 8,
            "shuffle": True,
            "drop_last": True,
            "num_workers": 0,
        },
        seed=11,
    )
    batch = next(iter(torch.utils.data.DataLoader(train, **config)))
    assert batch["pixels"].shape == (8, 2)
    assert batch["action"].shape == (8, 1)
    groups = batch[CONDITIONAL_JOINT_GROUP_COLUMN]
    assert int((groups < 0).sum()) == 4
    active = groups[groups >= 0]
    _, counts = torch.unique(active, return_counts=True)
    assert counts.tolist() == [2, 2]


def test_relation_sampler_shards_complete_pairs_across_ranks():
    singles = torch.arange(32)
    relations = torch.arange(32, 64).view(-1, 2)
    left = RelationBatchSampler(
        singles,
        relations,
        batch_size=8,
        epoch_row_count=64,
        seed=5,
        rank=0,
        world_size=2,
    )
    right = RelationBatchSampler(
        singles,
        relations,
        batch_size=8,
        epoch_row_count=64,
        seed=5,
        rank=1,
        world_size=2,
    )
    for left_batch, right_batch in zip(left, right):
        left_pairs = {tuple(left_batch[pos : pos + 2]) for pos in range(4, 8, 2)}
        right_pairs = {tuple(right_batch[pos : pos + 2]) for pos in range(4, 8, 2)}
        assert left_pairs.isdisjoint(right_pairs)
        assert left_pairs | right_pairs <= {tuple(pair) for pair in relations.tolist()}


class _RelationLeaf:
    def episode_relation_keys(self, column):
        assert column == "pair_id"
        return ["a", "a", "b", "b"]

    def episode_clip_range(self, episode):
        return (episode * 3, 3)


def test_relation_index_aligns_offsets_within_public_pairs():
    assert _paired_episode_relations(_RelationLeaf()) == [
        (0, 3),
        (1, 4),
        (2, 5),
        (6, 9),
        (7, 10),
        (8, 11),
    ]
