#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import h5py


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from contextworld.paths import artifact_path, resolve_contextworld_path
from contextworld.synthesis.manifest import write_json
from contextworld.training.episode_split import (
    build_episode_heldout_vds,
    episode_ids_sha256,
    partition_episode_ids,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build an episode-held-out zero-copy TwoRoom HDF5 view"
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(
            "../../data/world_model/quentinll/lewm-tworooms/tworoom.h5"
        ),
    )
    parser.add_argument("--seed", type=int, default=3072)
    parser.add_argument("--train-fraction", type=float, default=0.9)
    parser.add_argument(
        "--output",
        type=Path,
        default=artifact_path(
            "splits/tworoom_original_episode_holdout_s3072.h5",
            repo_root=REPO_ROOT,
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=artifact_path(
            "splits/tworoom_original_episode_holdout_s3072.json",
            repo_root=REPO_ROOT,
        ),
    )
    args = parser.parse_args()

    source = resolve_contextworld_path(args.source, repo_root=REPO_ROOT)
    output = resolve_contextworld_path(args.output, repo_root=REPO_ROOT)
    manifest = resolve_contextworld_path(args.manifest, repo_root=REPO_ROOT)
    with h5py.File(source, "r") as handle:
        episode_count = int(len(handle["ep_len"]))
    train, heldout = partition_episode_ids(
        episode_count,
        seed=args.seed,
        train_fraction=args.train_fraction,
    )
    built = build_episode_heldout_vds(source, output, heldout)
    report = {
        "schema_version": 1,
        "protocol": "tworoom_original_episode_heldout_v1",
        "passed": True,
        "partition": {
            "algorithm": "numpy_default_rng_permutation_v1",
            "seed": args.seed,
            "train_fraction": args.train_fraction,
            "source_episodes": episode_count,
            "train_episodes": int(len(train)),
            "heldout_episodes": int(len(heldout)),
            "train_episode_ids_sha256": episode_ids_sha256(train),
            "heldout_episode_ids_sha256": episode_ids_sha256(heldout),
            "episode_overlap": 0,
        },
        "view": built,
        "view_file_sha256": _sha256(output),
    }
    write_json(manifest, report)
    print(report)


if __name__ == "__main__":
    main()
