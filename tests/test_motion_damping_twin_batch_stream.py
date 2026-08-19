from __future__ import annotations

import torch

from scripts.run_pusht_motion_damping_h3_train import (
    CompleteTwinPairedBatchStream,
)


def test_motion_damping_batch_keeps_complete_forward_reverse_twins() -> None:
    stream = iter(
        CompleteTwinPairedBatchStream(
            32,
            batch_size=16,
            seed=14321,
        )
    )
    rows = next(stream)

    assert rows.shape == (16,)
    assert torch.unique(rows).numel() == 16
    groups = rows.reshape(-1, 4)
    for group in groups:
        first_pair = int(group[0]) // 2
        second_pair = int(group[2]) // 2
        assert group.tolist() == [
            2 * first_pair,
            2 * first_pair + 1,
            2 * second_pair,
            2 * second_pair + 1,
        ]
        assert first_pair % 2 == 0
        assert second_pair == first_pair + 1


def test_motion_damping_twin_stream_is_seed_reproducible() -> None:
    first = iter(
        CompleteTwinPairedBatchStream(32, batch_size=16, seed=7)
    )
    second = iter(
        CompleteTwinPairedBatchStream(32, batch_size=16, seed=7)
    )

    for _ in range(4):
        assert torch.equal(next(first), next(second))
