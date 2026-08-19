#!/usr/bin/env python3
"""Estimate frozen response scales from contact-friction Training only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
for source_root in (
    ROOT,
    ROOT.parent / "stable-worldmodel",
    Path(__file__).resolve().parent,
):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from contextworld.benchmarks.contact_friction_icl_data import (  # noqa: E402
    DEFAULT_CONTACT_FRICTION_RELEASE_CONFIG,
    file_sha256,
    load_contact_friction_icl_release,
)
from contextworld.paths import resolve_contextworld_path  # noqa: E402
import run_pusht_contact_friction_h3_train as friction  # noqa: E402
import run_pusht_hidden_actuation_pilot as pilot  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--release-config",
        type=Path,
        default=DEFAULT_CONTACT_FRICTION_RELEASE_CONFIG,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Compatible Training root override for a registered diagnostic",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=friction.DEFAULT_CHECKPOINT,
    )
    parser.add_argument(
        "--original-h5",
        type=Path,
        default=friction.DEFAULT_ORIGINAL_H5,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite output: {output}")
    release_path = args.release_config.expanduser().resolve()
    release = load_contact_friction_icl_release(release_path)
    data_root = (
        resolve_contextworld_path(
            release["data"]["artifact_tree"]["root"],
            repo_root=ROOT,
        )
        if args.data_root is None
        else args.data_root.expanduser().resolve()
    )
    action_stats = pilot.original_action_stats(
        args.original_h5.expanduser().resolve()
    )
    hidden = friction._training_split(
        data_root / release["data"]["lance_tables"]["train"],
        expected_pairs=int(release["data"]["pair_counts"]["train"]),
        action_stats=action_stats,
    )
    device = torch.device(args.device)
    model, checkpoint_receipt = pilot.load_model(
        args.checkpoint.expanduser().resolve(),
        device=device,
    )
    model.eval()
    _, projected = pilot.encode_pixels(
        model,
        hidden.pixels,
        device=device,
        batch_size=args.batch_size,
    )
    pairs = torch.arange(projected.size(0)).reshape(-1, 2)
    contrasts = (
        projected[pairs[:, 0]] - projected[pairs[:, 1]]
    ).double() / (2.0**0.5)
    scales = contrasts.square().mean(dim=(0, 2)).sqrt()
    pair_mse = (
        projected[pairs[:, 0]].double()
        - projected[pairs[:, 1]].double()
    ).square().mean(dim=(0, 2))
    active = (1, 3)
    active_scales = {str(index): float(scales[index]) for index in active}
    if not all(
        torch.isfinite(scales[index]) and scales[index] > 0
        for index in active
    ):
        raise RuntimeError(f"Invalid active scales: {active_scales}")
    payload = {
        "schema_version": 1,
        "status": "frozen_source_response_scales_estimated",
        "benchmark": release["release_id"],
        "method": "dynamics_response_sigreg",
        "release": {
            "path": str(release_path),
            "sha256": file_sha256(release_path),
        },
        "source_checkpoint": checkpoint_receipt,
        "training_split": {
            "path": str(
                data_root / release["data"]["lance_tables"]["train"]
            ),
            "pair_count": hidden.pair_count,
            "manifest_sha256": release["data"]["manifest_sha256"],
            "observed_manifest_sha256": file_sha256(
                data_root / "manifest.json"
            ),
            "data_root_override": args.data_root is not None,
            "evaluation_split_used": False,
        },
        "estimator": {
            "formula": "RMS((z_low-z_high)/sqrt(2))",
            "model_mode": "eval",
            "embedding_dimension": int(projected.size(-1)),
            "active_observation_indices": list(active),
        },
        "contrast_scales_by_observation_index": {
            str(index): float(value)
            for index, value in enumerate(scales)
        },
        "active_contrast_scales": active_scales,
        "pair_mse_by_observation_index": {
            str(index): float(value)
            for index, value in enumerate(pair_mse)
        },
        "frozen_during_training": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "completed",
                "output": str(output),
                "active_contrast_scales": active_scales,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
