#!/usr/bin/env python3
"""Build an independent, resolvable replay-matched confirmation set."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np


CONTEXTWORLD_ROOT = Path(__file__).resolve().parents[1]
STABLE_WORLD_MODEL_ROOT = CONTEXTWORLD_ROOT.parent / "stable-worldmodel"
for source_root in (
    CONTEXTWORLD_ROOT,
    STABLE_WORLD_MODEL_ROOT,
    Path(__file__).resolve().parent,
):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from contextworld.evaluation.pusht_hidden_actuation import (  # noqa: E402
    array_sha256,
)
from contextworld.evaluation.pusht_replay_matched_hidden_actuation import (  # noqa: E402
    replay_candidate_rows,
)
from contextworld.paths import artifact_path  # noqa: E402
from build_pusht_replay_matched_hidden_actuation_h3 import (  # noqa: E402
    DEFAULT_ORIGINAL_DATASET,
    build_split,
    canonical_json_sha256,
    file_sha256,
    safe_output_path,
    source_episode_partitions,
)


PROTOCOL = "pusht_action_strength_history3_replay_matched_confirm_strict_v5"
DEFAULT_V2_ROOT = artifact_path(
    "synthesis/pusht_hidden_actuation_replay_matched_h3_strict_v3"
)
DEFAULT_OUTPUT = artifact_path(
    "synthesis/pusht_hidden_actuation_replay_matched_confirm_h3_strict_v4"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--original-dataset",
        type=Path,
        default=DEFAULT_ORIGINAL_DATASET,
    )
    parser.add_argument("--v2-root", type=Path, default=DEFAULT_V2_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pair-count", type=int, default=256)
    parser.add_argument("--partition-seed", type=int, default=20260730)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument(
        "--minimum-future-block-gap-px",
        type=float,
        default=15.0,
    )
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument(
        "--maximum-candidate-attempts",
        type=int,
        default=512,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.original_dataset.expanduser().resolve()
    v2_root = args.v2_root.expanduser().resolve()
    output = safe_output_path(args.output)
    required = (source, v2_root / "manifest.json")
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing input(s):\n" + "\n".join(map(str, missing))
        )
    if args.pair_count <= 0:
        raise ValueError("--pair-count must be positive")
    if args.minimum_future_block_gap_px != 15.0:
        raise ValueError(
            "The registered confirmation gap must remain exactly 15 px"
        )

    v2_manifest = json.loads((v2_root / "manifest.json").read_text())
    with h5py.File(source, "r", swmr=True) as handle:
        states = handle["state"][:]
        actions = handle["action"][:]
        episode_ids_by_row = handle["episode_idx"][:]
        step_indices = handle["step_idx"][:]
        episode_offsets = handle["ep_offset"][:]
        episode_lengths = handle["ep_len"][:]
    partitions = source_episode_partitions(
        len(episode_offsets),
        seed=int(args.partition_seed),
        validation_fraction=float(args.validation_fraction),
    )
    previously_evaluated_rows = np.asarray(
        [
            pair["template"]["source_row_index"]
            for pair in v2_manifest["splits"]["validation"]["pairs"]
        ],
        dtype=np.int64,
    )
    previously_evaluated_episode_positions = set(
        map(
            int,
            np.searchsorted(
                episode_offsets,
                previously_evaluated_rows,
                side="right",
            )
            - 1,
        )
    )
    eligible_episodes = np.asarray(
        [
            int(value)
            for value in partitions["validation"]
            if int(value) not in previously_evaluated_episode_positions
        ],
        dtype=np.int64,
    )
    candidates = replay_candidate_rows(
        states,
        actions,
        episode_offsets,
        episode_lengths,
        eligible_episodes,
    )
    request = {
        "protocol": PROTOCOL,
        "pair_count": int(args.pair_count),
        "partition_seed": int(args.partition_seed),
        "synthesis_seed": int(args.seed),
        "minimum_future_block_gap_px": float(
            args.minimum_future_block_gap_px
        ),
        "resolution": int(args.resolution),
        "jpeg_quality": int(args.jpeg_quality),
        "source": {
            "path": str(source),
            "size_bytes": source.stat().st_size,
            "eligible_episode_count": int(len(eligible_episodes)),
            "eligible_episodes_sha256": array_sha256(
                eligible_episodes
            ),
            "candidate_count": int(len(candidates)),
            "candidate_rows_sha256": array_sha256(
                np.sort(candidates).astype(np.int64)
            ),
        },
        "prior_v2": {
            "root": str(v2_root),
            "manifest_sha256": file_sha256(
                v2_root / "manifest.json"
            ),
            "excluded_evaluated_source_episode_count": len(
                previously_evaluated_episode_positions
            ),
        },
    }

    with tempfile.TemporaryDirectory(
        prefix="pusht-replay-confirm-v3-build-",
        dir="/tmp",
    ) as temporary:
        root = Path(temporary) / output.name
        root.mkdir()
        (root / "request.json").write_text(
            json.dumps(request, indent=2, sort_keys=True) + "\n"
        )
        report = build_split(
            root=root,
            split="validation",
            pair_count=int(args.pair_count),
            candidates=candidates,
            states=states,
            actions=actions,
            episode_ids_by_row=episode_ids_by_row,
            step_indices=step_indices,
            episode_offsets=episode_offsets,
            episode_lengths=episode_lengths,
            seed=int(args.seed),
            resolution=int(args.resolution),
            jpeg_quality=int(args.jpeg_quality),
            maximum_candidate_attempts=int(
                args.maximum_candidate_attempts
            ),
            minimum_future_block_gap_px=float(
                args.minimum_future_block_gap_px
            ),
        )
        strict_audit = report["strict_causal_chain_audit"]
        if not strict_audit["passed"]:
            raise RuntimeError(
                f"Strict causal-chain audit failed: {strict_audit}"
            )
        selected_rows = np.asarray(
            [
                pair["template"]["source_row_index"]
                for pair in report["pairs"]
            ],
            dtype=np.int64,
        )
        selected_episode_positions = set(
            map(
                int,
                np.searchsorted(
                    episode_offsets,
                    selected_rows,
                    side="right",
                )
                - 1,
            )
        )
        prior_query_hashes = set()
        for split in v2_manifest["splits"]:
            prior_query_hashes.update(
                v2_manifest["splits"][split]["query_hashes"]
            )
        new_query_hashes = set(report["query_hashes"])
        overlap = {
            "source_episode_with_prior_replay_validation": len(
                selected_episode_positions
                & previously_evaluated_episode_positions
            ),
            "query_pixel_hash_with_training_or_development": len(
                new_query_hashes & prior_query_hashes
            ),
        }
        cross_audit = {
            **overlap,
            "all_selected_episodes_in_eligible_partition": (
                selected_episode_positions
                <= set(map(int, eligible_episodes))
            ),
        }
        cross_audit["passed"] = (
            not any(overlap.values())
            and cross_audit[
                "all_selected_episodes_in_eligible_partition"
            ]
        )
        if not cross_audit["passed"]:
            raise RuntimeError(
                f"Confirmatory independence audit failed: {cross_audit}"
            )
        manifest = {
            **request,
            "request_sha256": canonical_json_sha256(request),
            "splits": {"validation": report},
            "cross_split_audit": cross_audit,
            "strict_causal_chain_audit": strict_audit,
            "passed": (
                cross_audit["passed"]
                and strict_audit["passed"]
                and all(
                    pair["audit"]["passed"]
                    for pair in report["pairs"]
                )
            ),
        }
        manifest_path = root / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        summary = {
            "protocol": PROTOCOL,
            "root": str(output),
            "manifest_sha256": file_sha256(manifest_path),
            "pair_count": int(args.pair_count),
            "acceptance_rate": report["acceptance_rate"],
            "attempts_total": report["attempts_total"],
            "minimum_future_block_gap_px": float(
                args.minimum_future_block_gap_px
            ),
            "cross_split_audit": cross_audit,
            "strict_causal_chain_audit": strict_audit,
            "passed": manifest["passed"],
        }
        (root / "build_report.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        shutil.copytree(root, output)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
