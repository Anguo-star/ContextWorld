#!/usr/bin/env python3
"""Build frozen planning catalogs for original-heldout and Synth5Matched."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from contextworld.evaluation.protocol import (
    allocate_scenario_evaluations,
    scenario_seed,
    select_episode_balanced_starts,
)
from contextworld.paths import portable_contextworld_path, resolve_contextworld_path
from contextworld.synthesis.manifest import write_json
from contextworld.synthesis.stablewm import load_stable_worldmodel
from contextworld.training.episode_split import partition_episode_ids


PINNED_STABLEWM = "5864b74980f6ed328fd0045e777b3865962eff43"
EVAL_SEEDS = (42, 43, 44, 45, 46, 47)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _original_entries(
    heldout_h5: Path, original_h5: Path, *, num_eval: int
) -> list[dict[str, Any]]:
    import h5py

    with h5py.File(heldout_h5, "r") as handle:
        lengths = np.asarray(handle["ep_len"], dtype=np.int64)
        offsets = np.asarray(handle["ep_offset"], dtype=np.int64)
        ep_idx = np.asarray(handle["ep_idx"], dtype=np.int64)
        step_idx = np.asarray(handle["step_idx"], dtype=np.int64)
    with h5py.File(original_h5, "r") as handle:
        source_count = len(handle["ep_len"])
    _, heldout_source_ids = partition_episode_ids(
        source_count, seed=3072, train_fraction=0.9
    )
    maximum_start = lengths - 25 - 1
    valid = step_idx <= maximum_start[ep_idx]
    valid_indices = np.flatnonzero(valid)
    entries: list[dict[str, Any]] = []
    for seed in EVAL_SEEDS:
        rng = np.random.default_rng(seed)
        positions = rng.choice(
            len(valid_indices) - 1, size=num_eval, replace=False
        )
        rows = np.sort(valid_indices[positions])
        for index, row in enumerate(rows):
            episode = int(ep_idx[row])
            start = int(step_idx[row])
            entries.append(
                {
                    "evaluation_id": f"original-s{seed}-e{index:03d}",
                    "eval_seed": seed,
                    "evaluation_index": index,
                    "source_kind": "original_h5",
                    "source_path": portable_contextworld_path(
                        heldout_h5, repo_root=REPO_ROOT
                    ),
                    "source_episode_id": int(heldout_source_ids[episode]),
                    "episode": episode,
                    "start_step": start,
                    "goal_offset": 25,
                    "cem_group_seed": seed,
                    "stratum": "original_future25",
                }
            )
    return entries


def build_original(args: argparse.Namespace) -> dict[str, Any]:
    heldout = resolve_contextworld_path(args.heldout_h5, repo_root=REPO_ROOT)
    original = resolve_contextworld_path(args.original_h5, repo_root=REPO_ROOT)
    entries = _original_entries(heldout, original, num_eval=args.num_eval)
    payload = {
        "schema_version": 1,
        "catalog": "tworoom_original_heldout_eval_catalog_v1",
        "status": "frozen",
        "protocol": {
            "eval_seeds": list(EVAL_SEEDS),
            "num_eval_per_seed": args.num_eval,
            "goal_offset": 25,
            "eval_budget": 50,
            "horizon": 5,
            "receding_horizon": 5,
            "cem_samples": 300,
            "cem_steps": 30,
            "cem_topk": 30,
            "selection": "pinned_stablewm_valid_row_sampling_v1",
        },
        "sources": {
            "heldout_h5": portable_contextworld_path(
                heldout, repo_root=REPO_ROOT
            ),
            "heldout_h5_sha256": _sha256(heldout),
            "original_h5_sha256": _sha256(original),
        },
        "entries": entries,
    }
    output = resolve_contextworld_path(args.original_output, repo_root=REPO_ROOT)
    write_json(output, payload)
    return payload


def _lance_dataset(swm: Any, path: Path):
    return swm.data.LanceDataset(
        path=path,
        frameskip=1,
        num_steps=1,
        keys_to_load=[
            "pixels",
            "action",
            "proprio",
            "state",
            "goal_state",
            "variation_agent_speed",
        ],
    )


def build_matched(args: argparse.Namespace) -> dict[str, Any]:
    catalog_path = resolve_contextworld_path(
        args.synthesis_catalog, repo_root=REPO_ROOT
    )
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    paths = sorted(
        resolve_contextworld_path(value, repo_root=REPO_ROOT)
        for value in catalog["val"]["synthetic"]
    )
    swm, stable_repo, stable_commit = load_stable_worldmodel(
        REPO_ROOT, args.stablewm_repo, args.stablewm_ref
    )
    entries: list[dict[str, Any]] = []
    for seed in EVAL_SEEDS:
        counts = allocate_scenario_evaluations(
            scenario_count=len(paths), total_evaluations=args.num_eval, seed=seed
        )
        evaluation_index = 0
        for path, count in zip(paths, counts, strict=True):
            dataset = _lance_dataset(swm, path)
            starts = select_episode_balanced_starts(
                dataset.lengths,
                goal_offset=25,
                count=count,
                seed=scenario_seed(seed, path),
            )
            for episode, start in zip(
                starts.episodes, starts.steps, strict=True
            ):
                entries.append(
                    {
                        "evaluation_id": f"matched-s{seed}-e{evaluation_index:03d}",
                        "eval_seed": seed,
                        "evaluation_index": evaluation_index,
                        "source_kind": "synthetic_lance",
                        "source_path": portable_contextworld_path(
                            path, repo_root=REPO_ROOT
                        ),
                        "scenario": path.name,
                        "episode": int(episode),
                        "start_step": int(start),
                        "goal_offset": 25,
                        "cem_group_seed": scenario_seed(seed, path),
                        "stratum": "speed5_original_matched",
                    }
                )
                evaluation_index += 1
    payload = {
        "schema_version": 1,
        "catalog": "tworoom_speed5_matched_eval_catalog_v1",
        "status": "frozen",
        "protocol": {
            "eval_seeds": list(EVAL_SEEDS),
            "num_eval_per_seed": args.num_eval,
            "goal_offset": 25,
            "eval_budget": 50,
            "horizon": 5,
            "receding_horizon": 5,
            "cem_samples": 300,
            "cem_steps": 30,
            "cem_topk": 30,
            "scenario_allocation": "balanced",
            "episode_selection": "round_robin_unique_before_reuse",
        },
        "sources": {
            "synthesis_catalog": portable_contextworld_path(
                catalog_path, repo_root=REPO_ROOT
            ),
            "synthesis_catalog_sha256": _sha256(catalog_path),
            "stable_worldmodel_repo": str(stable_repo),
            "stable_worldmodel_commit": stable_commit,
        },
        "entries": entries,
    }
    output = resolve_contextworld_path(args.matched_output, repo_root=REPO_ROOT)
    write_json(output, payload)
    return payload


def parse_args() -> argparse.Namespace:
    root = Path(
        "artifacts/evaluation/history3/original_ability_reconstruction"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--original-h5",
        type=Path,
        default=Path(
            "../../data/world_model/quentinll/lewm-tworooms/tworoom.h5"
        ),
    )
    parser.add_argument(
        "--heldout-h5",
        type=Path,
        default=Path(
            "artifacts/splits/tworoom_original_episode_holdout_s3072.h5"
        ),
    )
    parser.add_argument(
        "--synthesis-catalog",
        type=Path,
        default=Path(
            "artifacts/synthesis/catalogs/tworoom_synth5_matched_v2.json"
        ),
    )
    parser.add_argument(
        "--original-output",
        type=Path,
        default=root / "original_heldout_eval_catalog.json",
    )
    parser.add_argument(
        "--matched-output",
        type=Path,
        default=root / "speed5_matched_eval_catalog.json",
    )
    parser.add_argument("--num-eval", type=int, default=50)
    parser.add_argument("--stablewm-repo", default="../stable-worldmodel")
    parser.add_argument("--stablewm-ref", default=PINNED_STABLEWM)
    parser.add_argument("--original-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    original = build_original(args)
    output = {
        "original_entries": len(original["entries"]),
        "original_output": str(
            resolve_contextworld_path(args.original_output, repo_root=REPO_ROOT)
        ),
    }
    if not args.original_only:
        matched = build_matched(args)
        output.update(
            {
                "matched_entries": len(matched["entries"]),
                "matched_output": str(
                    resolve_contextworld_path(
                        args.matched_output, repo_root=REPO_ROOT
                    )
                ),
            }
        )
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
