#!/usr/bin/env python3
"""Evaluate frozen checkpoints on replay-matched Push-T validation pairs."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

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
from stable_worldmodel.data import LanceDataset  # noqa: E402


DEFAULT_DATA_ROOT = artifact_path(
    "synthesis/pusht_hidden_actuation_replay_matched_h3_v2"
)
DEFAULT_TRAINING_ROOT = artifact_path(
    "evaluation/history3/"
    "pusht_hidden_actuation_replay_matched_h3_v2/"
    "training_seed13313_step2048_standard64_hidden64"
)
DEFAULT_OUTPUT = DEFAULT_TRAINING_ROOT / "v2_validation_diagnostic.json"


def materialize_validation(
    path: Path,
    *,
    action_stats: dict[str, Any],
) -> dict[str, torch.Tensor]:
    dataset = LanceDataset(
        path=path,
        frameskip=5,
        num_steps=4,
        keys_to_load=["pixels", "action", "state"],
    )
    samples = [dataset[index] for index in range(len(dataset))]
    if not samples or len(samples) % 2:
        raise RuntimeError("Validation must contain adjacent low/high pairs")
    pixels = torch.stack([row["pixels"] for row in samples])
    actions = torch.stack([row["action"] for row in samples]).float()
    actions = pilot.normalize_action_blocks(actions, action_stats)
    states = torch.stack([row["state"] for row in samples]).float()
    low = torch.arange(0, len(samples), 2)
    high = low + 1
    checks = {
        "initial_pixels_equal": bool(
            torch.equal(pixels[low, 0], pixels[high, 0])
        ),
        "query_pixels_equal": bool(
            torch.equal(pixels[low, 2], pixels[high, 2])
        ),
        "actions_equal": bool(
            torch.equal(actions[low], actions[high])
        ),
        "probe_pixels_differ": bool(
            torch.all(
                torch.any(
                    pixels[low, 1] != pixels[high, 1],
                    dim=(1, 2, 3),
                )
            )
        ),
        "future_pixels_differ": bool(
            torch.all(
                torch.any(
                    pixels[low, 3] != pixels[high, 3],
                    dim=(1, 2, 3),
                )
            )
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Validation pair audit failed: {checks}")
    return {
        "low_pixels": pixels[low],
        "high_pixels": pixels[high],
        "action": actions[low],
        "low_states": states[low],
        "high_states": states[high],
        "pair_audit": checks,
    }


@torch.no_grad()
def gap_stratified_diagnostic(
    model: torch.nn.Module,
    evaluation: dict[str, torch.Tensor],
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    """Measure post-hoc decision errors by physical future separation."""

    was_training = model.training
    model.eval()
    low_pixels = evaluation["low_pixels"]
    high_pixels = evaluation["high_pixels"]
    actions = evaluation["action"][:, :3]
    histories = torch.cat([low_pixels[:, :3], high_pixels[:, :3]])
    predictions = pilot.predict_histories(
        model,
        histories,
        torch.cat([actions, actions]),
        device=device,
        batch_size=batch_size,
    )
    count = low_pixels.size(0)
    predicted_low = predictions[:count]
    predicted_high = predictions[count:]
    futures = torch.cat(
        [low_pixels[:, 3:4], high_pixels[:, 3:4]]
    )
    _, projected = pilot.encode_pixels(
        model,
        futures,
        device=device,
        batch_size=batch_size,
    )
    target_low = projected[:count, 0]
    target_high = projected[count:, 0]

    def mse(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return (left - right).square().mean(dim=-1)

    low_to_low = mse(predicted_low, target_low)
    low_to_high = mse(predicted_low, target_high)
    high_to_low = mse(predicted_high, target_low)
    high_to_high = mse(predicted_high, target_high)
    low_correct = low_to_low < low_to_high
    high_correct = high_to_high < high_to_low
    low_history_correct = low_to_low < high_to_low
    high_history_correct = high_to_high < low_to_high
    switch_correct = (
        (
            (predicted_high - predicted_low)
            * (target_high - target_low)
        ).sum(dim=-1)
        > 0
    )
    gaps = torch.linalg.vector_norm(
        evaluation["high_states"][:, 3, 2:4]
        - evaluation["low_states"][:, 3, 2:4],
        dim=-1,
    )

    def summarize(mask: torch.Tensor) -> dict[str, Any]:
        selected = int(mask.sum())
        if selected == 0:
            return {"pair_count": 0}
        low_rate = float(low_correct[mask].float().mean())
        high_rate = float(high_correct[mask].float().mean())
        return {
            "pair_count": selected,
            "decision_count": 2 * selected,
            "target_selection": float(
                torch.cat(
                    [low_correct[mask], high_correct[mask]]
                ).float().mean()
            ),
            "low_gain_target_selection": low_rate,
            "high_gain_target_selection": high_rate,
            "worst_mode_target_selection": min(low_rate, high_rate),
            "correct_history_preference": float(
                torch.cat(
                    [
                        low_history_correct[mask],
                        high_history_correct[mask],
                    ]
                ).float().mean()
            ),
            "correct_rule_switch": float(
                switch_correct[mask].float().mean()
            ),
        }

    thresholds = (2.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0)
    result = {
        "role": (
            "post_hoc_diagnostic_not_a_registered_threshold_change"
        ),
        "overall": summarize(torch.ones_like(gaps, dtype=torch.bool)),
        "minimum_future_block_gap_px": {
            str(int(threshold)): summarize(gaps >= threshold)
            for threshold in thresholds
        },
        "gap_quantiles_px": {
            str(quantile): float(torch.quantile(gaps, quantile))
            for quantile in (0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0)
        },
    }
    model.train(was_training)
    return result


def parse_checkpoint(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or not name or not raw_path:
        raise argparse.ArgumentTypeError(
            "Checkpoint must use NAME=/absolute/or/relative/path"
        )
    return name, Path(raw_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--action-normalizer-source",
        type=Path,
        default=pilot.DEFAULT_ORIGINAL_DATASET,
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        type=parse_checkpoint,
        required=True,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    action_source = args.action_normalizer_source.expanduser().resolve()
    output = Path(os.path.abspath(args.output.expanduser()))
    checkpoints = {
        name: path.expanduser().resolve()
        for name, path in args.checkpoint
    }
    if len(checkpoints) != len(args.checkpoint):
        raise ValueError("Checkpoint names must be unique")
    required = [
        data_root / "manifest.json",
        data_root / "validation.lance",
        action_source,
        *checkpoints.values(),
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing input(s):\n" + "\n".join(map(str, missing))
        )
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but unavailable")

    action_stats = pilot.original_action_stats(action_source)
    evaluation = materialize_validation(
        data_root / "validation.lance",
        action_stats=action_stats,
    )
    pair_audit = evaluation.pop("pair_audit")
    results = {}
    receipts = {}
    for name, checkpoint in checkpoints.items():
        model, receipt = pilot.load_model(checkpoint, device=device)
        results[name] = pilot.evaluate_model(
            model,
            evaluation,
            device=device,
            batch_size=args.batch_size,
        )
        results[name]["gap_stratified_diagnostic"] = (
            gap_stratified_diagnostic(
                model,
                evaluation,
                device=device,
                batch_size=args.batch_size,
            )
        )
        receipts[name] = receipt
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    report = {
        "schema_version": 1,
        "status": "post_registered_v2_failure_validation_diagnostic",
        "question": (
            "Did each checkpoint learn replay-matched validation pairs, "
            "independent of transfer to the frozen narrow v1 evaluation?"
        ),
        "data": {
            "root": str(data_root),
            "manifest_sha256": pilot.file_sha256(
                data_root / "manifest.json"
            ),
            "split": "validation",
            "pair_count": int(evaluation["low_pixels"].size(0)),
            "pair_audit": pair_audit,
        },
        "checkpoints": receipts,
        "results": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "output_sha256": pilot.file_sha256(output),
                "results": {
                    name: {
                        "target_selection": values[
                            "two_real_future_target_selection_rate"
                        ],
                        "history_preference": values[
                            "correct_history_preference_rate"
                        ],
                        "rule_switch": values[
                            "correct_rule_switch_rate"
                        ],
                        "worst_mode": values[
                            "worst_mode_target_selection_rate"
                        ],
                    }
                    for name, values in results.items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
