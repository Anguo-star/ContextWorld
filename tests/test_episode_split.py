from pathlib import Path

import h5py
import numpy as np

from contextworld.training.episode_split import (
    build_episode_heldout_vds,
    partition_episode_ids,
)


def test_episode_partition_is_deterministic_disjoint_and_complete() -> None:
    first_train, first_heldout = partition_episode_ids(
        100, seed=3072, train_fraction=0.9
    )
    second_train, second_heldout = partition_episode_ids(
        100, seed=3072, train_fraction=0.9
    )

    np.testing.assert_array_equal(first_train, second_train)
    np.testing.assert_array_equal(first_heldout, second_heldout)
    assert len(first_train) == 90
    assert len(first_heldout) == 10
    assert not set(first_train) & set(first_heldout)
    assert set(first_train) | set(first_heldout) == set(range(100))


def test_episode_heldout_vds_reindexes_episodes_without_copying_rows(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.h5"
    output = tmp_path / "heldout.h5"
    with h5py.File(source, "w") as handle:
        handle.create_dataset("ep_len", data=np.asarray([2, 3, 2], dtype=np.int32))
        handle.create_dataset("ep_offset", data=np.asarray([0, 2, 5], dtype=np.int64))
        handle.create_dataset("ep_idx", data=np.asarray([0, 0, 1, 1, 1, 2, 2], dtype=np.int32))
        handle.create_dataset("value", data=np.arange(7, dtype=np.float32))

    result = build_episode_heldout_vds(
        source, output, np.asarray([0, 2], dtype=np.int64)
    )

    with h5py.File(output, "r") as handle:
        np.testing.assert_array_equal(handle["ep_len"][:], [2, 2])
        np.testing.assert_array_equal(handle["ep_offset"][:], [0, 2])
        np.testing.assert_array_equal(handle["ep_idx"][:], [0, 0, 1, 1])
        np.testing.assert_array_equal(handle["value"][:], [0, 1, 5, 6])
        assert handle["value"].is_virtual
    assert result["episodes"] == 2
    assert result["rows"] == 4
