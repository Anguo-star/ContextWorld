#!/usr/bin/env python3
"""Calibrate transition-conditional SIGReg on one exact mixed Push-T batch."""

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
    ConditionalSIGReg,
    GroupBalancedSIGReg,
    SIGReg,
)


DEFAULT_OUTPUT = artifact_path(
    "evaluation/history3/pusht_hidden_actuation_h3_v1/"
    "transition_conditional_sigreg_calibration_seed7169/"
    "first_batch_gradient_report.json"
)
PROTOCOL = CONTEXTWORLD_ROOT / (
    "configs/benchmark/"
    "pusht_hidden_actuation_transition_conditional_sigreg_v1.yaml"
)
CANDIDATE_WEIGHTS = (0.005, 0.01, 0.02, 0.05, 0.09)
VARIANT_BY_WEIGHT = {
    0.005: "mixed_transition_conditional_sigreg_0p005",
    0.01: "mixed_transition_conditional_sigreg_0p01",
    0.02: "mixed_transition_conditional_sigreg_0p02",
    0.05: "mixed_transition_conditional_sigreg_0p05",
    0.09: "mixed_transition_conditional_sigreg_0p09",
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
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=7169)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-projections", type=int, default=1024)
    parser.add_argument("--projection-seed", type=int, default=97031)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--model-mode",
        choices=("train", "eval"),
        default="train",
        help=(
            "Use train for the registered optimization-gradient screen. "
            "Eval is an exploratory control that removes predictor dropout "
            "and uses inference-time normalization."
        ),
    )
    return parser.parse_args()


def set_projection_seed(seed: int, device: torch.device) -> None:
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

    low_margin = mse(predicted_low, target_high) - mse(
        predicted_low,
        target_low,
    )
    high_margin = mse(predicted_high, target_low) - mse(
        predicted_high,
        target_high,
    )
    return torch.cat([low_margin, high_margin]).mean()


def linear_zero_crossing(
    prediction_effect: float,
    regularizer_effect: float,
) -> float | None:
    if regularizer_effect == 0.0:
        return None
    crossing = -prediction_effect / regularizer_effect
    return crossing if math.isfinite(crossing) else None


def main() -> None:
    args = parse_args()
    if args.seed != 7169:
        raise ValueError("The registered calibration seed is 7169")
    if args.batch_size != 128:
        raise ValueError("The registered calibration batch size is 128")
    if args.num_projections != 1024:
        raise ValueError("The registered projection count is 1024")
    if args.projection_seed != 97031:
        raise ValueError("The registered projection seed is 97031")
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

    model, checkpoint_receipt = pilot.load_model(
        args.checkpoint,
        device=device,
    )
    if args.model_mode == "train":
        model.train()
    else:
        model.eval()
    named_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    names = [name for name, _ in named_parameters]
    parameters = [parameter for _, parameter in named_parameters]
    marginal = SIGReg(
        knots=17,
        num_proj=args.num_projections,
    ).to(device)
    difference = ConditionalSIGReg(
        knots=17,
        num_proj=args.num_projections,
        randomize_pair_orientation=True,
    ).to(device)
    transition_regularizer = GroupBalancedSIGReg(
        knots=17,
        num_proj=args.num_projections,
        randomize_pair_orientation=True,
    ).to(device)

    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        output = model.encode({"pixels": pixels, "action": actions})
        embeddings = output["emb"]
        prediction = model.predict(
            embeddings[:, :3],
            output["act_emb"][:, :3],
        )
        targets = embeddings[:, 1:]
        prediction_loss = torch.square(prediction - targets).mean()
        standard_prediction_mse = torch.square(
            prediction[:original_batch_size]
            - targets[:original_batch_size]
        ).mean()
        metrics = {
            "target_probe_x1_pair_distance": pair_distance(
                targets,
                pairs,
                time_index=0,
            ),
            "target_history_future_x3_pair_distance": pair_distance(
                targets,
                pairs,
                time_index=2,
            ),
            "predicted_correct_future_margin_x3": correct_future_margin(
                prediction,
                targets,
                pairs,
            ),
            "standard_replay_prediction_mse": standard_prediction_mse,
        }
        irreducible_prediction_x1_pair_distance = pair_distance(
            prediction,
            pairs,
            time_index=0,
        )
        (
            transition_population,
            transition_pairs,
            transition_active,
        ) = mixed.build_transition_conditional_population(
            embeddings,
            prediction,
            pairs,
        )

        set_projection_seed(args.projection_seed, device)
        marginal_loss = marginal(
            transition_population.transpose(0, 1)
        )
        set_projection_seed(args.projection_seed, device)
        difference_loss = difference(
            transition_population.transpose(0, 1),
            pairs=transition_pairs,
            active=transition_active,
        )
        set_projection_seed(args.projection_seed, device)
        regularizer_loss = transition_regularizer(
            transition_population.transpose(0, 1),
            pairs=transition_pairs,
            active=transition_active,
        )

    metric_gradients = {
        name: base.gradients(value, parameters)
        for name, value in metrics.items()
    }
    prediction_gradients = base.gradients(prediction_loss, parameters)
    regularizer_gradients = base.gradients(regularizer_loss, parameters)
    loss_gradients = {
        "prediction_mse": prediction_gradients,
        "transition_conditional_sigreg_unweighted": regularizer_gradients,
    }
    for weight in CANDIDATE_WEIGHTS:
        loss_gradients[f"total_weight_{weight:g}"] = base.combine_gradients(
            prediction_gradients,
            regularizer_gradients,
            right_scale=weight,
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

    marginal_population_gradient = torch.autograd.grad(
        marginal_loss,
        transition_population,
        retain_graph=True,
    )[0]
    difference_population_gradient = torch.autograd.grad(
        difference_loss,
        transition_population,
    )[0]
    batch_size = args.batch_size
    ordinary_marginal_gradient_norm = float(
        torch.cat(
            [
                marginal_population_gradient[:original_batch_size].flatten(),
                marginal_population_gradient[
                    batch_size : batch_size + original_batch_size
                ].flatten(),
            ]
        ).float().norm()
    )
    target_difference_gradient_norm = float(
        difference_population_gradient[
            original_batch_size:batch_size
        ][:, [0, 2]].float().norm()
    )
    prediction_difference_x3_gradient_norm = float(
        difference_population_gradient[
            batch_size + original_batch_size :
        ][:, 2].float().norm()
    )
    prediction_difference_x1_gradient_norm = float(
        difference_population_gradient[
            batch_size + original_batch_size :
        ][:, 0].float().norm()
    )
    structural_gates = {
        "irreducible_prediction_x1_pair_distance_is_zero": (
            float(irreducible_prediction_x1_pair_distance.detach())
            <= 1e-12
        ),
        "irreducible_prediction_x1_pair_not_conditionally_regularized": (
            prediction_difference_x1_gradient_norm == 0.0
        ),
        "marginal_gradient_reaches_ordinary_rows": (
            ordinary_marginal_gradient_norm > 0.0
        ),
        "target_difference_gradient_reaches_hidden_rows": (
            target_difference_gradient_norm > 0.0
        ),
        "prediction_difference_gradient_reaches_hidden_x3_rows": (
            prediction_difference_x3_gradient_norm > 0.0
        ),
    }
    weight_gates: dict[str, dict[str, bool]] = {}
    for weight in CANDIDATE_WEIGHTS:
        total_name = f"total_weight_{weight:g}"

        def change(metric_name: str) -> float:
            return float(
                effects[metric_name][total_name]["all"][
                    "predicted_distance_change_per_unit_lr"
                ]
            )

        weight_gates[f"{weight:g}"] = {
            "target_probe_x1_pair_distance_expands": (
                change("target_probe_x1_pair_distance") > 0.0
            ),
            "target_history_future_x3_pair_distance_expands": (
                change("target_history_future_x3_pair_distance") > 0.0
            ),
            "predicted_correct_future_margin_x3_increases": (
                change("predicted_correct_future_margin_x3") > 0.0
            ),
            "standard_replay_prediction_mse_decreases": (
                change("standard_replay_prediction_mse") < 0.0
            ),
        }
        weight_gates[f"{weight:g}"]["all_directional_gates_passed"] = all(
            weight_gates[f"{weight:g}"].values()
        )

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
    output_path = Path(os.path.abspath(args.output.expanduser()))
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite {output_path}")
    report: dict[str, Any] = {
        "schema_version": 1,
        "benchmark": "pusht_hidden_actuation_history3_action_coverage_v2",
        "diagnostic": (
            "transition_conditional_sigreg_exact_first_mixed_batch"
        ),
        "status": (
            "gradient_gate_passed_weight_selected"
            if selected_weight is not None
            else "gradient_gate_failed_stop_before_training"
        ),
        "method": {
            "display_name_zh": "转移条件 SIGReg",
            "formula": (
                "prediction_mse + lambda * one GroupBalancedSIGReg "
                "on concatenated transition targets and predictions"
            ),
            "native_target_sigreg_replaced_not_stacked": True,
            "additional_loss_components": 0,
            "external_regularizer_weight_count": 1,
        },
        "protocol": {
            "calibration_seed": args.seed,
            "batch_size": args.batch_size,
            "original_samples": original_batch_size,
            "hidden_samples": hidden_batch_size,
            "hidden_pairs": hidden_batch_size // 2,
            "candidate_external_weights": list(CANDIDATE_WEIGHTS),
            "maximum_external_weight": max(CANDIDATE_WEIGHTS),
            "num_projections": args.num_projections,
            "projection_seed": args.projection_seed,
            "target_pair_active_future_times": [1, 3],
            "prediction_pair_active_future_times": [3],
            "hidden_rule_or_class_labels_exposed": False,
            "model_mode": args.model_mode,
        },
        "receipts": {
            "registered_protocol": {
                "path": str(PROTOCOL),
                "sha256": pilot.file_sha256(PROTOCOL),
            },
            "checkpoint": checkpoint_receipt,
            "hidden_manifest_sha256": pilot.file_sha256(
                args.hidden_data_root / "manifest.json"
            ),
            "original_lance": str(args.original_lance),
            "first_batch_pixels_sha256": base.tensor_sha256([raw_pixels]),
            "first_batch_actions_sha256": base.tensor_sha256([raw_actions]),
            "hidden_pair_indices": [int(value) for value in hidden_indices],
        },
        "scalar_values": {
            "prediction_mse": float(prediction_loss.detach()),
            "standard_replay_prediction_mse": float(
                standard_prediction_mse.detach()
            ),
            "marginal_sigreg_unweighted": float(marginal_loss.detach()),
            "paired_difference_sigreg_unweighted": float(
                difference_loss.detach()
            ),
            "transition_conditional_sigreg_unweighted": float(
                regularizer_loss.detach()
            ),
            "metrics": {
                name: float(value.detach())
                for name, value in metrics.items()
            },
            "irreducible_prediction_x1_pair_distance": float(
                irreducible_prediction_x1_pair_distance.detach()
            ),
            "population_gradient_norms": {
                "ordinary_marginal": ordinary_marginal_gradient_norm,
                "hidden_target_difference_x1_x3": (
                    target_difference_gradient_norm
                ),
                "hidden_prediction_difference_x3": (
                    prediction_difference_x3_gradient_norm
                ),
                "hidden_prediction_difference_x1": (
                    prediction_difference_x1_gradient_norm
                ),
            },
        },
        "gradient_effects": effects,
        "zero_crossings": {
            metric_name: linear_zero_crossing(
                float(
                    metric_effects["prediction_mse"]["all"][
                        "predicted_distance_change_per_unit_lr"
                    ]
                ),
                float(
                    metric_effects[
                        "transition_conditional_sigreg_unweighted"
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
                "The screen uses local first-batch derivatives only and "
                "cannot establish trained prediction or planning ability."
            ),
            (
                "The prediction contrast is deliberately inactive at x1 "
                "because the hidden gain is not yet identifiable."
            ),
            (
                "No coefficient may be selected from downstream training or "
                "evaluation results."
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
