from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np


def partition_episode_ids(
    episode_count: int,
    *,
    seed: int,
    train_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic, disjoint train/held-out episode identifiers."""

    if episode_count < 2:
        raise ValueError("Episode partition requires at least two episodes")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be strictly between zero and one")
    train_count = int(np.floor(episode_count * train_fraction))
    if not 0 < train_count < episode_count:
        raise ValueError("Episode partition produced an empty split")
    permutation = np.random.default_rng(int(seed)).permutation(episode_count)
    train = np.sort(permutation[:train_count].astype(np.int64))
    heldout = np.sort(permutation[train_count:].astype(np.int64))
    return train, heldout


def episode_ids_sha256(values: np.ndarray) -> str:
    canonical = np.asarray(values, dtype="<i8").tobytes()
    return hashlib.sha256(canonical).hexdigest()


def clip_subset_indices(dataset: Any, episode_ids: np.ndarray) -> list[int]:
    selected = set(np.asarray(episode_ids, dtype=np.int64).tolist())
    return [
        index
        for index, (episode_id, _start) in enumerate(dataset.clip_indices)
        if int(episode_id) in selected
    ]


def episode_row_indices(
    lengths: np.ndarray,
    offsets: np.ndarray,
    episode_ids: np.ndarray,
) -> np.ndarray:
    selected = np.asarray(episode_ids, dtype=np.int64)
    return np.concatenate(
        [
            np.arange(
                int(offsets[episode_id]),
                int(offsets[episode_id]) + int(lengths[episode_id]),
                dtype=np.int64,
            )
            for episode_id in selected
        ]
    )


def build_episode_heldout_vds(
    source: Path,
    output: Path,
    episode_ids: np.ndarray,
) -> dict[str, Any]:
    """Create a zero-copy HDF5 virtual view containing selected episodes."""

    import h5py

    source = source.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    selected = np.asarray(episode_ids, dtype=np.int64)
    with h5py.File(source, "r") as source_file:
        lengths = np.asarray(source_file["ep_len"][:], dtype=np.int64)
        offsets = np.asarray(source_file["ep_offset"][:], dtype=np.int64)
        if selected.size == 0 or np.any(selected < 0) or np.any(selected >= len(lengths)):
            raise ValueError("Held-out episode identifiers are empty or out of range")
        selected_lengths = lengths[selected]
        selected_offsets = np.concatenate(
            [np.asarray([0], dtype=np.int64), np.cumsum(selected_lengths[:-1])]
        )
        total_rows = int(selected_lengths.sum())

        with h5py.File(output, "w", libver="latest") as output_file:
            output_file.attrs["contextworld_view"] = "episode_heldout_v1"
            output_file.attrs["source_h5"] = str(source)
            output_file.attrs["source_episode_ids_sha256"] = episode_ids_sha256(
                selected
            )
            output_file.create_dataset(
                "ep_len", data=selected_lengths.astype(source_file["ep_len"].dtype)
            )
            output_file.create_dataset(
                "ep_offset",
                data=selected_offsets.astype(source_file["ep_offset"].dtype),
            )
            output_file.create_dataset(
                "ep_idx",
                data=np.repeat(
                    np.arange(len(selected), dtype=source_file["ep_idx"].dtype),
                    selected_lengths,
                ),
            )

            for key, dataset in source_file.items():
                if key in {"ep_len", "ep_offset", "ep_idx"}:
                    continue
                shape = (total_rows, *dataset.shape[1:])
                layout = h5py.VirtualLayout(shape=shape, dtype=dataset.dtype)
                virtual_source = h5py.VirtualSource(
                    str(source), key, shape=dataset.shape
                )
                cursor = 0
                for episode_id, length in zip(selected, selected_lengths):
                    source_start = int(offsets[episode_id])
                    source_stop = source_start + int(length)
                    layout[cursor : cursor + int(length)] = virtual_source[
                        source_start:source_stop
                    ]
                    cursor += int(length)
                output_file.create_virtual_dataset(key, layout)

    return {
        "source": str(source),
        "output": str(output),
        "episodes": int(len(selected)),
        "rows": total_rows,
        "episode_ids_sha256": episode_ids_sha256(selected),
        "zero_copy_virtual_datasets": True,
    }
