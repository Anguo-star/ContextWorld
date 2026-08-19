#!/usr/bin/env python3
"""Export a Lightning-style Push-T LeWM checkpoint for standard CEM eval."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

from omegaconf import OmegaConf, open_dict
import torch


CONTEXTWORLD_ROOT = Path(__file__).resolve().parents[1]
STABLE_WORLD_MODEL_ROOT = CONTEXTWORLD_ROOT.parent / "stable-worldmodel"
for source_root in (
    CONTEXTWORLD_ROOT,
    STABLE_WORLD_MODEL_ROOT,
    Path(__file__).resolve().parent,
):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from contextworld.paths import artifact_path  # noqa: E402
import run_pusht_hidden_actuation_pilot as pilot  # noqa: E402


DEFAULT_OUTPUT = artifact_path(
    "evaluation/history3/pusht_hidden_actuation_h3_v1/"
    "mixed_retention_seed3073_step2048/source_checkpoint"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=pilot.DEFAULT_CHECKPOINT,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    output = Path(os.path.abspath(args.output.expanduser()))
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite output: {output}")
    output.mkdir(parents=True)

    cfg = OmegaConf.load(
        STABLE_WORLD_MODEL_ROOT / "scripts/train/config/lewm.yaml"
    )
    with open_dict(cfg):
        cfg.model.action_encoder.input_dim = 10
    config = OmegaConf.to_container(cfg.model, resolve=True)
    (output / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n"
    )

    state = pilot.checkpoint_model_state(checkpoint)
    weights = output / "source_weights.pt"
    with tempfile.TemporaryDirectory(
        prefix="pusht-source-export-",
        dir="/tmp",
    ) as temporary:
        temporary_weights = Path(temporary) / weights.name
        torch.save(state, temporary_weights)
        shutil.copy2(temporary_weights, weights)
    receipt = {
        "schema_version": 1,
        "source": {
            "path": str(checkpoint),
            "sha256": pilot.file_sha256(checkpoint),
        },
        "export": {
            "path": str(weights),
            "sha256": pilot.file_sha256(weights),
            "model_state_sha256": pilot.state_sha256(state),
            "format": "raw_model_state_dict_with_sibling_config_json",
        },
    }
    receipt_path = output / "receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
