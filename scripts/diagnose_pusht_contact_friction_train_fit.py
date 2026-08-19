#!/usr/bin/env python3
"""Measure whether friction checkpoints fit their own frozen training pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contextworld.benchmarks.adapters import (
    StableWorldModelLeWMContactFrictionAdapter,
    StableWorldModelPLDMContactFrictionAdapter,
)
from contextworld.benchmarks.contact_friction_icl_data import (
    DEFAULT_CONTACT_FRICTION_RELEASE_CONFIG,
    _read_lance_pairs,
    load_contact_friction_icl_release,
)
from contextworld.benchmarks.contact_friction_icl_score import (
    contact_friction_prediction_metrics,
)
from contextworld.paths import resolve_contextworld_path


VARIANTS = {
    "lewm": "mixed_native_sigreg_0p09",
    "pldm": "mixed_pldm_joint",
}
ADAPTERS = {
    "lewm": StableWorldModelLeWMContactFrictionAdapter,
    "pldm": StableWorldModelPLDMContactFrictionAdapter,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--release-config",
        type=Path,
        default=DEFAULT_CONTACT_FRICTION_RELEASE_CONFIG,
    )
    parser.add_argument("--device", default="cuda:6")
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    release = load_contact_friction_icl_release(args.release_config)
    data_root = resolve_contextworld_path(
        release["data"]["artifact_tree"]["root"],
        repo_root=ROOT,
    )
    pair_count = int(release["data"]["pair_counts"]["train"])
    print(f"Loading {pair_count} frozen training pairs", flush=True)
    arrays = _read_lance_pairs(
        data_root / release["data"]["lance_tables"]["train"],
        expected_pairs=pair_count,
        expected_split="train",
    )
    histories = np.concatenate(
        [arrays.low_pixels[:, :3], arrays.high_pixels[:, :3]],
        axis=0,
    )
    actions = np.concatenate(
        [
            arrays.raw_action_blocks[:, :3],
            arrays.raw_action_blocks[:, :3],
        ],
        axis=0,
    )
    futures = np.concatenate(
        [arrays.low_pixels[:, 3], arrays.high_pixels[:, 3]],
        axis=0,
    )
    normalization = release["evaluation"]["action_normalization"]
    runtime = release["runtime"]["stable_worldmodel"]
    seeds = tuple(
        int(value)
        for value in release["training"]["reference_matrix"][
            "training_seeds"
        ]
    )
    training_root = args.training_root.expanduser().resolve()
    rows = []
    for model in ("lewm", "pldm"):
        for seed in seeds:
            variant = VARIANTS[model]
            checkpoint = (
                training_root
                / f"{model}_seed{seed}"
                / f"{variant}_step4096.pt"
            )
            print(f"Evaluating {model} seed {seed}", flush=True)
            adapter = ADAPTERS[model].from_checkpoint(
                checkpoint,
                action_mean=normalization["mean"],
                action_std=normalization["std_population"],
                repo_root=ROOT,
                stablewm_repo=runtime["repo"],
                stablewm_ref=runtime["expected_ref"],
                device=args.device,
            )
            predicted = adapter.rollout_latents(
                histories,
                actions,
                batch_size=args.batch_size,
            )[:, 0]
            encoded = adapter.encode_pixels(
                futures,
                batch_size=args.batch_size,
            )
            metrics, _ = contact_friction_prediction_metrics(
                pair_ids=arrays.pair_ids,
                predicted_low=predicted[:pair_count],
                predicted_high=predicted[pair_count:],
                target_low=encoded[:pair_count],
                target_high=encoded[pair_count:],
            )
            rows.append(
                {
                    "model": model,
                    "seed": seed,
                    "checkpoint": str(checkpoint),
                    "metrics": metrics,
                }
            )
            del adapter
            try:
                import torch

                torch.cuda.empty_cache()
            except (ImportError, RuntimeError):
                pass

    payload = {
        "schema_version": 1,
        "status": "completed",
        "diagnostic_only": True,
        "split": "train",
        "pair_count": pair_count,
        "rows": rows,
    }
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "completed", "output": str(output)}))


if __name__ == "__main__":
    main()
