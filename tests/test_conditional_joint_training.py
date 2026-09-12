"""Focused tests for the flat, relation-preserving COJA data bridge."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from contextworld.training.groups import LogicalGroupDataset, RelationBatchSampler
from contextworld.training.stablewm_bundle import (
    CONDITIONAL_JOINT_GROUP_COLUMN,
    _delay_triplet_relations,
    _paired_episode_relations,
    _RuntimeDataset,
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


def _ternary_runtime():
    """The ActionDelay shape: one relation per query, three delay arms."""

    original = _Rows(36, source=0)
    synthetic = _Rows(18, source=1)
    mixture = LogicalGroupDataset(
        {"original": original, "synthetic": synthetic},
        {"original": 0.5, "synthetic": 0.5},
        epoch_size=72,
    )
    relations = [(index, index + 1, index + 2) for index in range(0, 18, 3)]
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
    assert _paired_episode_relations(_RelationLeaf(), group_width=2) == [
        (0, 3),
        (1, 4),
        (2, 5),
        (6, 9),
        (7, 10),
        (8, 11),
    ]


class _DelayArmLeaf:
    """One published ActionDelay member: anchored, one clip per episode."""

    def __init__(self, queries):
        self.queries = list(queries)

    def __len__(self):
        return len(self.queries)

    def episode_relation_keys(self, column):
        assert column == "id"
        return list(self.queries)

    def episode_clip_range(self, episode):
        return (episode, 1)


def _delay_shard(shard: str, delays=(0, 4, 8), queries=("q0", "q1")):
    members = [
        Path(f"ad-h7-paired-train-p{shard}-d{delay}-0123456789.lance")
        for delay in delays
    ]
    return members, [_DelayArmLeaf(queries) for _ in delays]


def test_delay_relations_join_the_same_query_across_every_delay_arm():
    members, leaves = _delay_shard("000")
    other_members, other_leaves = _delay_shard("001", queries=("q2", "q3"))

    relations = _delay_triplet_relations(
        [*members, *other_members], [*leaves, *other_leaves], group_width=3
    )

    # Member order fixes the flat index space: two episodes per arm, three
    # arms per shard, so shard p001 starts at index six.
    assert relations == [(0, 2, 4), (1, 3, 5), (6, 8, 10), (7, 9, 11)]


def test_delay_relations_reject_a_shard_whose_arms_reorder_their_queries():
    members, leaves = _delay_shard("000")
    leaves[-1] = _DelayArmLeaf(("q1", "q0"))

    with pytest.raises(ValueError, match="one query identity per episode"):
        _delay_triplet_relations(members, leaves, group_width=3)


def test_delay_relations_reject_an_incomplete_delay_shard():
    members, leaves = _delay_shard("000", delays=(0, 4))

    with pytest.raises(ValueError, match="2 delay arms"):
        _delay_triplet_relations(members, leaves, group_width=3)

def test_delay_relations_reject_the_wrong_three_delays():
    members, leaves = _delay_shard("000", delays=(0, 2, 8))

    with pytest.raises(ValueError, match=r"delays \[0, 2, 8\].*\[0, 4, 8\]"):
        _delay_triplet_relations(members, leaves, group_width=3)


def test_delay_relations_reject_a_member_without_a_published_identity():
    members, leaves = _delay_shard("000")
    members[0] = Path("ad-h7-paired-train-0123456789.lance")

    with pytest.raises(ValueError, match="shard and delay identity"):
        _delay_triplet_relations(members, leaves, group_width=3)


def test_ternary_split_keeps_every_delay_arm_in_one_partition():
    runtime = _ternary_runtime()
    train, validation = runtime.split_for_training(
        train_fraction=0.75,
        generator=torch.Generator().manual_seed(7),
    )

    assert len(train) == 54
    assert len(validation) == 18
    assert train.singles.numel() == 27
    assert train.relations.shape == (9, 3)
    assert validation.relations.shape == (3, 3)
    assigned = torch.cat((train.relations.reshape(-1), validation.relations.reshape(-1)))
    assert assigned.unique().numel() == assigned.numel()


def test_ternary_loader_keeps_all_three_arms_inside_one_batch(monkeypatch):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    train, _ = _ternary_runtime().split_for_training(
        train_fraction=0.75,
        generator=torch.Generator().manual_seed(7),
    )
    config = train.configure_train_loader(
        {
            "batch_size": 12,
            "shuffle": True,
            "drop_last": True,
            "num_workers": 0,
        },
        seed=11,
    )

    batch = next(iter(torch.utils.data.DataLoader(train, **config)))

    assert batch["pixels"].shape == (12, 2)
    groups = batch[CONDITIONAL_JOINT_GROUP_COLUMN]
    assert int((groups < 0).sum()) == 6
    _, counts = torch.unique(groups[groups >= 0], return_counts=True)
    assert counts.tolist() == [3, 3]


def test_relation_sampler_shards_complete_triplets_across_ranks():
    singles = torch.arange(24)
    relations = torch.arange(24, 60).view(-1, 3)
    ranks = [
        RelationBatchSampler(
            singles,
            relations,
            batch_size=12,
            epoch_row_count=60,
            seed=5,
            rank=rank,
            world_size=2,
        )
        for rank in (0, 1)
    ]
    published = {tuple(relation) for relation in relations.tolist()}

    for left_batch, right_batch in zip(*ranks):
        left = {tuple(left_batch[pos : pos + 3]) for pos in range(6, 12, 3)}
        right = {tuple(right_batch[pos : pos + 3]) for pos in range(6, 12, 3)}
        assert left.isdisjoint(right)
        assert left | right <= published

def test_ternary_sampler_is_exactly_balanced_over_three_batches():
    singles = torch.arange(500)
    relations = torch.arange(500, 800).view(-1, 3)
    sampler = RelationBatchSampler(
        singles,
        relations,
        batch_size=128,
        epoch_row_count=384,
        seed=5,
    )

    batches = list(sampler)

    assert len(batches) == 3
    relation_rows = [sum(index >= 500 for index in batch) for batch in batches]
    assert relation_rows == [63, 63, 66]
    assert sum(relation_rows) == 192
    assert sum(128 - count for count in relation_rows) == 192
    published = {tuple(relation) for relation in relations.tolist()}
    for batch, relation_count in zip(batches, (21, 21, 22)):
        paired = batch[128 - relation_count * 3 :]
        assert {
            tuple(paired[position : position + 3])
            for position in range(0, len(paired), 3)
        } <= published
