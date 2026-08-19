#!/usr/bin/env python3
"""Calibrate scale-aware conditional SIGReg on one held-out mixed batch."""

from __future__ import annotations

import argparse
import json
import math
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
import analyze_pusht_hidden_actuation_gradients as base  # noqa: E402
import run_pusht_hidden_actuation_mixed as mixed  # noqa: E402
import run_pusht_hidden_actuation_pilot as pilot  # noqa: E402
from stable_worldmodel.wm.loss import (  # noqa: E402
    ScaleCalibratedConditionalSIGReg,
)


PROTOCOL = CONTEXTWORLD_ROOT / (
    "configs/benchmark/"
    "pusht_hidden_actuation_scale_calibrated_sigreg_v1.yaml"
)
DEFAULT_OUTPUT = artifact_path(
    "evaluation/history3/pusht_hidden_actuation_h3_v1/"
    "scale_calibrated_sigreg_v1/calibration_seed9217/"
    "first_batch_gradient_report.json"
)
CANDIDATE_WEIGHTS = (0.005, 0.01, 0.02, 0.05, 0.09)
DROPOUT_SEEDS = (93101, 93102, 93103, 93104)
VARIANT_BY_WEIGHT = {
    0.005: "mixed_scale_calibrated_sigreg_0p005",
    0.01: "mixed_scale_calibrated_sigreg_0p01",
    0.02: "mixed_scale_calibrated_sigreg_0p02",
    0.05: "mixed_scale_calibrated_sigreg_0p05",
    0.09: "mixed_scale_calibrated_sigreg_0p09",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hidden-data-root",
        type=Path,
        default=pilot.DEFAULT_DATA_ROOT,
    )
    parser.add_argument(
        "--original-lance",
        type=Path,
        default=mixed.DEFAULT_ORIGINAL_LANCE,
    )
    parser.add_argument(
        "--action-normalizer-source",
        type=Path,
        default=pilot.DEFAULT_ORIGINAL_DATASET,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=pilot.DEFAULT_CHECKPOINT,
    )
    parser.add_argument(
        "--contrast-scales",
        type=Path,
        default=mixed.DEFAULT_CONTRAST_SCALES,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=9217)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-projections", type=int, default=1024)
    parser.add_argument("--projection-seed", type=int, default=97033)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def set_seed(seed: int, device: torch.device) -> None:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def pair_distance(
    values: torch.Tensor,
    pairs: torch.Tensor,
    *,
    time_index: int,
) -> torch.Tensor:
    return torch.square(
        values[pairs[:, 0], time_index]
        - values[pairs[:, 1], time_index]
    ).mean()


def correct_future_margin(
    prediction: torch.Tensor,
    targets: torch.Tensor,
    pairs: torch.Tensor,
) -> torch.Tensor:
    low = pairs[:, 0]
    high = pairs[:, 1]
    predicted_low = prediction[low, 2]
    predicted_high = prediction[high, 2]
    target_low = targets[low, 2]
    target_high = targets[high, 2]

    def mse(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return torch.square(left - right).mean(dim=-1)

    return torch.cat(
        [
            mse(predicted_low, target_high)
            - mse(predicted_low, target_low),
            mse(predicted_high, target_low)
            - mse(predicted_high, target_high),
        ]
    ).mean()


def linear_zero_crossing(
    prediction_effect: float,
    regularizer_effect: float,
) -> float | None:
    if regularizer_effect == 0.0:
        return None
    value = -prediction_effect / regularizer_effect
    return value if math.isfinite(value) else None


def main() -> None:
    args = parse_args()
    if args.seed != 9217:
        raise ValueError("The registered calibration seed is 9217")
    if args.batch_size != 128:
        raise ValueError("The registered batch size is 128")
    if args.num_projections != 1024:
        raise ValueError("The registered projection count is 1024")
    if args.projection_seed != 97033:
        raise ValueError("The registered projection seed is 97033")
    output_path = Path(os.path.abspath(args.output.expanduser()))
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite {output_path}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    original_batch_size = args.batch_size // 2
    hidden_batch_size = args.batch_size - original_batch_size
    pilot.set_reproducible_seed(args.seed)
    action_stats = pilot.original_action_stats(
        args.action_normalizer_source
    )
    hidden = pilot.materialize_lance_split(
        args.hidden_data_root / "train.lance",
        action_stats=action_stats,
    )
    original_dataset, original_loader = mixed.original_loader(
        args.original_lance,
        batch_size=original_batch_size,
        seed=args.seed,
        num_workers=0,
    )
    original = next(iter(original_loader))
    hidden_indices = next(
        iter(
            pilot.PairedBatchStream(
                hidden.pair_count,
                batch_size=hidden_batch_size,
                seed=args.seed,
            )
        )
    )
    original_actions = pilot.normalize_action_blocks(
        torch.nan_to_num(original["action"].float(), 0.0),
        action_stats,
    )
    raw_pixels = torch.cat(
        [original["pixels"], hidden.pixels[hidden_indices]],
        dim=0,
    )
    raw_actions = torch.cat(
        [original_actions, hidden.action[hidden_indices]],
        dim=0,
    )
    pixels = pilot.preprocess_pixels(raw_pixels, device)
    actions = raw_actions.to(device=device, non_blocking=True)
    pairs = torch.arange(
        original_batch_size,
        args.batch_size,
        device=device,
    ).reshape(-1, 2)
    active = torch.zeros(
        4,
        pairs.size(0),
        dtype=torch.bool,
        device=device,
    )
    active[1] = True
    active[3] = True
    contrast_scales, scale_receipt = mixed.load_contrast_scales(
        args.contrast_scales.expanduser().resolve()
    )

    model, checkpoint_receipt = pilot.load_model(
        args.checkpoint,
        device=device,
    )
    named_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    names = [name for name, _ in named_parameters]
    parameters = [parameter for _, parameter in named_parameters]

    # Inference-time metrics are constructed before any train-mode BatchNorm
    # update.  They share the exact model parameters with the training loss.
    model.eval()
    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        eval_output = model.encode({"pixels": pixels, "action": actions})
        eval_prediction = model.predict(
            eval_output["emb"][:, :3],
            eval_output["act_emb"][:, :3],
        )
        eval_targets = eval_output["emb"][:, 1:]
        deterministic_metrics = {
            "deterministic_correct_future_margin_x3": (
                correct_future_margin(
                    eval_prediction,
                    eval_targets,
                    pairs,
                )
            ),
            "deterministic_standard_replay_prediction_mse": (
                torch.square(
                    eval_prediction[:original_batch_size]
                    - eval_targets[:original_batch_size]
                ).mean()
            ),
        }

    model.train()
    regularizer = ScaleCalibratedConditionalSIGReg(
        knots=17,
        num_proj=args.num_projections,
        randomize_pair_orientation=True,
    ).to(device)
    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        train_output = model.encode({"pixels": pixels, "action": actions})
        embeddings = train_output["emb"]
        stochastic_losses = []
        for dropout_seed in DROPOUT_SEEDS:
            set_seed(dropout_seed, device)
            stochastic_prediction = model.predict(
                embeddings[:, :3],
                train_output["act_emb"][:, :3],
            )
            stochastic_losses.append(
                torch.square(
                    stochastic_prediction - embeddings[:, 1:]
                ).mean()
            )
        prediction_loss = torch.stack(stochastic_losses).mean()
        metrics = {
            "target_probe_x1_pair_distance": pair_distance(
                embeddings,
                pairs,
                time_index=1,
            ),
            "target_history_future_x3_pair_distance": pair_distance(
                embeddings,
                pairs,
                time_index=3,
            ),
            **deterministic_metrics,
        }
        set_seed(args.projection_seed, device)
        regularizer_loss = regularizer(
            embeddings.transpose(0, 1),
            pairs=pairs,
            active=active,
            contrast_scales=contrast_scales,
        )

    metric_gradients = {
        name: base.gradients(value, parameters)
        for name, value in metrics.items()
    }
    prediction_gradients = base.gradients(prediction_loss, parameters)
    regularizer_gradients = base.gradients(
        regularizer_loss,
        parameters,
    )
    loss_gradients = {
        "dropout_expected_prediction_mse": prediction_gradients,
        "scale_calibrated_sigreg_unweighted": regularizer_gradients,
    }
    for weight in CANDIDATE_WEIGHTS:
        loss_gradients[f"total_weight_{weight:g}"] = (
            base.combine_gradients(
                prediction_gradients,
                regularizer_gradients,
                right_scale=weight,
            )
        )
    effects = {
        metric_name: {
            loss_name: base.gradient_effect(
                names,
                metric_values,
                loss_values,
            )
            for loss_name, loss_values in loss_gradients.items()
        }
        for metric_name, metric_values in metric_gradients.items()
    }

    population_gradient = torch.autograd.grad(
        regularizer_loss,
        embeddings,
    )[0]
    original_gradient_norm = float(
        population_gradient[:original_batch_size, [1, 3]]
        .float()
        .norm()
    )
    hidden_gradient_norm = float(
        population_gradient[original_batch_size:, [1, 3]]
        .float()
        .norm()
    )
    structural_gates = {
        "gradient_reaches_unpaired_standard_rows": (
            original_gradient_norm > 0.0
        ),
        "gradient_reaches_active_hidden_contrasts": (
            hidden_gradient_norm > 0.0
        ),
        "source_scales_are_frozen_and_positive": bool(
            torch.isfinite(contrast_scales).all()
            and (contrast_scales > 0).all()
        ),
    }

    def change(metric_name: str, total_name: str) -> float:
        return float(
            effects[metric_name][total_name]["all"][
                "predicted_distance_change_per_unit_lr"
            ]
        )

    weight_gates: dict[str, dict[str, bool]] = {}
    for weight in CANDIDATE_WEIGHTS:
        total_name = f"total_weight_{weight:g}"
        row = {
            "target_probe_x1_pair_distance_does_not_contract": (
                change(
                    "target_probe_x1_pair_distance",
                    total_name,
                )
                >= 0.0
            ),
            "target_history_future_x3_pair_distance_does_not_contract": (
                change(
                    "target_history_future_x3_pair_distance",
                    total_name,
                )
                >= 0.0
            ),
            "deterministic_correct_future_margin_x3_increases": (
                change(
                    "deterministic_correct_future_margin_x3",
                    total_name,
                )
                > 0.0
            ),
            "deterministic_standard_replay_prediction_mse_decreases": (
                change(
                    "deterministic_standard_replay_prediction_mse",
                    total_name,
                )
                < 0.0
            ),
        }
        row["all_directional_gates_passed"] = all(row.values())
        weight_gates[f"{weight:g}"] = row
    selected_weight = next(
        (
            weight
            for weight in CANDIDATE_WEIGHTS
            if all(structural_gates.values())
            and weight_gates[f"{weight:g}"][
                "all_directional_gates_passed"
            ]
        ),
        None,
    )

    report: dict[str, Any] = {
        "schema_version": 1,
        "benchmark": "pusht_hidden_actuation_history3_action_coverage_v2",
        "diagnostic": "scale_calibrated_sigreg_mixed_first_batch",
        "status": (
            "gradient_gate_passed_weight_selected"
            if selected_weight is not None
            else "gradient_gate_failed_stop_before_training"
        ),
        "method": {
            "display_name_zh": "尺度校准条件 SIGReg",
            "formula": (
                "prediction_mse + lambda * one SIGReg("
                "unpaired rows + source-scaled pair contrasts)"
            ),
            "sigreg_calls_per_optimizer_step": 1,
            "external_regularizer_weight_count": 1,
            "additional_loss_components": 0,
            "native_sigreg_stacked": False,
        },
        "protocol": {
            "path": str(PROTOCOL),
            "sha256": pilot.file_sha256(PROTOCOL),
            "calibration_seed": args.seed,
            "batch_size": args.batch_size,
            "original_samples": original_batch_size,
            "hidden_samples": hidden_batch_size,
            "hidden_pairs": hidden_batch_size // 2,
            "prediction_dropout_expectation_seeds": list(DROPOUT_SEEDS),
            "projection_seed": args.projection_seed,
            "num_projections": args.num_projections,
            "candidate_external_weights": list(CANDIDATE_WEIGHTS),
        },
        "receipts": {
            "source_checkpoint": checkpoint_receipt,
            "source_scales": scale_receipt,
            "hidden_manifest_sha256": pilot.file_sha256(
                args.hidden_data_root / "manifest.json"
            ),
            "original_lance": str(args.original_lance),
            "first_batch_pixels_sha256": base.tensor_sha256([raw_pixels]),
            "first_batch_actions_sha256": base.tensor_sha256([raw_actions]),
            "hidden_pair_indices": [int(value) for value in hidden_indices],
        },
        "scalar_values": {
            "dropout_expected_prediction_mse": float(
                prediction_loss.detach()
            ),
            "dropout_sample_prediction_mse": [
                float(value.detach()) for value in stochastic_losses
            ],
            "scale_calibrated_sigreg_unweighted": float(
                regularizer_loss.detach()
            ),
            "metrics": {
                name: float(value.detach())
                for name, value in metrics.items()
            },
            "population_gradient_norms": {
                "unpaired_standard_active_rows": original_gradient_norm,
                "hidden_active_rows": hidden_gradient_norm,
            },
        },
        "gradient_effects": effects,
        "zero_crossings": {
            metric_name: linear_zero_crossing(
                float(
                    metric_effects[
                        "dropout_expected_prediction_mse"
                    ]["all"]["predicted_distance_change_per_unit_lr"]
                ),
                float(
                    metric_effects[
                        "scale_calibrated_sigreg_unweighted"
                    ]["all"]["predicted_distance_change_per_unit_lr"]
                ),
            )
            for metric_name, metric_effects in effects.items()
        },
        "structural_gates": structural_gates,
        "weight_gates": weight_gates,
        "selected_weight": selected_weight,
        "selected_variant": (
            VARIANT_BY_WEIGHT[selected_weight]
            if selected_weight is not None
            else None
        ),
        "training_authorized": selected_weight is not None,
        "interpretation_scope": [
            (
                "Prediction-loss gradients average the four registered "
                "dropout realizations."
            ),
            (
                "Prediction metrics use deterministic inference semantics "
                "and no candidate-ranking statistic."
            ),
            (
                "The screen authorizes training but cannot establish "
                "prediction or real-environment planning ability."
            ),
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "output": str(output_path),
                "sha256": pilot.file_sha256(output_path),
                "status": report["status"],
                "selected_weight": selected_weight,
                "selected_variant": report["selected_variant"],
                "structural_gates": structural_gates,
                "weight_gates": weight_gates,
                "zero_crossings": report["zero_crossings"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    del original_dataset
    del original_loader


if __name__ == "__main__":
    main()
