#!/usr/bin/env python3
"""Build the frozen 1/2/3/5-block rollout catalog."""

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


PINNED_STABLEWM = "5864b74980f6ed328fd0045e777b3865962eff43"
EVAL_SEEDS = (42, 43, 44, 45, 46, 47)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _original_entries(path: Path, count: int) -> list[dict[str, Any]]:
    import h5py

    with h5py.File(path, "r") as handle:
        lengths = np.asarray(handle["ep_len"], dtype=np.int64)
    entries = []
    for seed in EVAL_SEEDS:
        starts = select_episode_balanced_starts(
            lengths, goal_offset=35, count=count, seed=seed
        )
        for index, (episode, start) in enumerate(
            zip(starts.episodes, starts.steps, strict=True)
        ):
            entries.append(
                {
                    "evaluation_id": f"rollout-original-s{seed}-e{index:03d}",
                    "eval_seed": seed,
                    "evaluation_index": index,
                    "domain": "original_heldout",
                    "source_kind": "original_h5",
                    "source_path": portable_contextworld_path(
                        path, repo_root=REPO_ROOT
                    ),
                    "episode": int(episode),
                    "start_step": int(start),
                }
            )
    return entries


def _lance_dataset(swm: Any, path: Path):
    return swm.data.LanceDataset(
        path=path,
        frameskip=1,
        num_steps=1,
        keys_to_load=["pixels", "action"],
    )


def _matched_entries(
    swm: Any, paths: list[Path], count: int
) -> list[dict[str, Any]]:
    entries = []
    for seed in EVAL_SEEDS:
        counts = allocate_scenario_evaluations(
            scenario_count=len(paths), total_evaluations=count, seed=seed
        )
        evaluation_index = 0
        for path, scenario_count in zip(paths, counts, strict=True):
            dataset = _lance_dataset(swm, path)
            starts = select_episode_balanced_starts(
                dataset.lengths,
                goal_offset=35,
                count=scenario_count,
                seed=scenario_seed(seed, path) ^ 0x524F4C4C,
            )
            for episode, start in zip(
                starts.episodes, starts.steps, strict=True
            ):
                entries.append(
                    {
                        "evaluation_id": (
                            f"rollout-matched-s{seed}-e{evaluation_index:03d}"
                        ),
                        "eval_seed": seed,
                        "evaluation_index": evaluation_index,
                        "domain": "speed5_matched",
                        "source_kind": "synthetic_lance",
                        "source_path": portable_contextworld_path(
                            path, repo_root=REPO_ROOT
                        ),
                        "scenario": path.name,
                        "episode": int(episode),
                        "start_step": int(start),
                    }
                )
                evaluation_index += 1
    return entries


def run(args: argparse.Namespace) -> dict[str, Any]:
    heldout = resolve_contextworld_path(args.heldout_h5, repo_root=REPO_ROOT)
    synthesis_catalog = resolve_contextworld_path(
        args.synthesis_catalog, repo_root=REPO_ROOT
    )
    catalog = json.loads(synthesis_catalog.read_text(encoding="utf-8"))
    paths = sorted(
        resolve_contextworld_path(value, repo_root=REPO_ROOT)
        for value in catalog["val"]["synthetic"]
    )
    swm, stable_repo, stable_commit = load_stable_worldmodel(
        REPO_ROOT, args.stablewm_repo, args.stablewm_ref
    )
    entries = _original_entries(heldout, args.num_eval)
    entries.extend(_matched_entries(swm, paths, args.num_eval))
    payload = {
        "schema_version": 1,
        "catalog": "tworoom_original_ability_rollout_v1",
        "status": "frozen",
        "protocol": {
            "eval_seeds": list(EVAL_SEEDS),
            "num_eval_per_seed_per_domain": args.num_eval,
            "history_observation_blocks": 3,
            "frameskip": 5,
            "rollout_horizons": [1, 2, 3, 5],
            "required_raw_span": 36,
            "metrics": [
                "native_latent_mse",
                "native_latent_rmse",
                "native_latent_cosine_distance",
            ],
        },
        "sources": {
            "heldout_h5": portable_contextworld_path(
                heldout, repo_root=REPO_ROOT
            ),
            "heldout_h5_sha256": _sha256(heldout),
            "synthesis_catalog": portable_contextworld_path(
                synthesis_catalog, repo_root=REPO_ROOT
            ),
            "synthesis_catalog_sha256": _sha256(synthesis_catalog),
            "stable_worldmodel_repo": str(stable_repo),
            "stable_worldmodel_commit": stable_commit,
        },
        "entries": entries,
    }
    output = resolve_contextworld_path(args.output, repo_root=REPO_ROOT)
    write_json(output, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
        "--output",
        type=Path,
        default=Path(
            "artifacts/evaluation/history3/original_ability_reconstruction/"
            "rollout_catalog.json"
        ),
    )
    parser.add_argument("--num-eval", type=int, default=50)
    parser.add_argument("--stablewm-repo", default="../stable-worldmodel")
    parser.add_argument("--stablewm-ref", default=PINNED_STABLEWM)
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps({"entries": len(result["entries"])}, sort_keys=True))
