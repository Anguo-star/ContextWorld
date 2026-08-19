#!/usr/bin/env python3
"""Diagnose first-batch representation contraction in hidden-actuation Push-T.

The benchmark contains condition-matched low/high-gain pairs.  This script
reconstructs the exact first paired batch used by the registered training
protocol and measures

    d D / d eta at eta=0 for theta <- theta - eta grad_theta L,

where ``D`` is the squared representation distance between the two hidden
gain members.  A negative value means that gradient descent on ``L`` locally
contracts the gain-carrying representation direction.

This is a read-only autograd diagnostic.  It neither updates the checkpoint
nor writes into the synthesis dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

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
from stable_worldmodel.wm.loss import ConditionalSIGReg, SIGReg  # noqa: E402


DEFAULT_OUTPUT = artifact_path(
    "evaluation/history3/pusht_hidden_actuation_h3_v1/"
    "mechanism_seed3073/first_batch_gradient_report.json"
)
MODULE_GROUPS = (
    "encoder",
    "projector",
    "predictor",
    "pred_proj",
    "action_encoder",
)


def tensor_sha256(values: Iterable[torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for value in values:
        tensor = value.detach().cpu().contiguous()
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def parameter_group(name: str) -> str:
    for group in MODULE_GROUPS:
        if name == group or name.startswith(group + "."):
            return group
    return "other"


def combine_gradients(
    left: tuple[torch.Tensor | None, ...],
    right: tuple[torch.Tensor | None, ...],
    *,
    right_scale: float,
) -> tuple[torch.Tensor | None, ...]:
    result: list[torch.Tensor | None] = []
    for left_value, right_value in zip(left, right):
        if left_value is None and right_value is None:
            result.append(None)
        elif left_value is None:
            result.append(right_value * right_scale)
        elif right_value is None:
            result.append(left_value)
        else:
            result.append(left_value + right_scale * right_value)
    return tuple(result)


def gradient_effect(
    names: list[str],
    distance_gradients: tuple[torch.Tensor | None, ...],
    loss_gradients: tuple[torch.Tensor | None, ...],
) -> dict[str, dict[str, float | str | None]]:
    accumulators = {
        group: {
            "dot": 0.0,
            "distance_energy": 0.0,
            "loss_energy": 0.0,
        }
        for group in (*MODULE_GROUPS, "other", "all")
    }
    for name, distance_gradient, loss_gradient in zip(
        names,
        distance_gradients,
        loss_gradients,
    ):
        if distance_gradient is None or loss_gradient is None:
            continue
        distance_value = distance_gradient.detach().float()
        loss_value = loss_gradient.detach().float()
        dot = float(torch.sum(distance_value * loss_value))
        distance_energy = float(torch.square(distance_value).sum())
        loss_energy = float(torch.square(loss_value).sum())
        group = parameter_group(name)
        for key in (group, "all"):
            accumulators[key]["dot"] += dot
            accumulators[key]["distance_energy"] += distance_energy
            accumulators[key]["loss_energy"] += loss_energy

    output: dict[str, dict[str, float | str | None]] = {}
    for group, values in accumulators.items():
        dot = values["dot"]
        denominator = math.sqrt(
            values["distance_energy"] * values["loss_energy"]
        )
        predicted_change = -dot
        if denominator == 0.0:
            effect = "no_shared_gradient"
            cosine = None
        elif predicted_change < 0.0:
            effect = "contracts_pair_distance"
            cosine = predicted_change / denominator
        elif predicted_change > 0.0:
            effect = "expands_pair_distance"
            cosine = predicted_change / denominator
        else:
            effect = "neutral"
            cosine = 0.0
        output[group] = {
            "distance_gradient_norm": math.sqrt(
                values["distance_energy"]
            ),
            "loss_gradient_norm_on_shared_parameters": math.sqrt(
                values["loss_energy"]
            ),
            "gradient_dot": dot,
            "descent_cosine": cosine,
            "predicted_distance_change_per_unit_lr": predicted_change,
            "effect": effect,
        }
    return output


def paired_distance(
    embeddings: torch.Tensor,
    *,
    time_index: int,
) -> torch.Tensor:
    low = embeddings[0::2, time_index]
    high = embeddings[1::2, time_index]
    return torch.square(low - high).mean()


def gradients(
    value: torch.Tensor,
    parameters: list[torch.nn.Parameter],
) -> tuple[torch.Tensor | None, ...]:
    return torch.autograd.grad(
        value,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=pilot.DEFAULT_DATA_ROOT,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=pilot.DEFAULT_CHECKPOINT,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=3073)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--sigreg-weight", type=float, default=0.09)
    parser.add_argument("--num-projections", type=int, default=1024)
    parser.add_argument("--projection-seed", type=int, default=97031)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.batch_size % 2:
        raise ValueError("--batch-size must be a positive even integer")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    pilot.set_reproducible_seed(args.seed)
    action_stats = pilot.original_action_stats(pilot.DEFAULT_ORIGINAL_DATASET)
    train = pilot.materialize_lance_split(
        args.data_root / "train.lance",
        action_stats=action_stats,
    )
    stream = iter(
        pilot.PairedBatchStream(
            train.pair_count,
            batch_size=args.batch_size,
            seed=args.seed,
        )
    )
    indices = next(stream)
    raw_pixels = train.pixels[indices]
    raw_actions = train.action[indices]

    model, checkpoint_receipt = pilot.load_model(
        args.checkpoint,
        device=device,
    )
    model.train()
    parameters_and_names = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    names = [name for name, _ in parameters_and_names]
    parameters = [parameter for _, parameter in parameters_and_names]
    pixels = pilot.preprocess_pixels(raw_pixels, device)
    actions = raw_actions.to(device=device, non_blocking=True)

    pair_indices = torch.arange(
        args.batch_size,
        device=device,
    ).reshape(-1, 2)
    active = torch.zeros(
        4,
        args.batch_size // 2,
        dtype=torch.bool,
        device=device,
    )
    active[1] = True
    active[3] = True
    native_regularizer = SIGReg(
        knots=17,
        num_proj=args.num_projections,
    ).to(device)
    conditional_regularizer = ConditionalSIGReg(
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
        predictions = model.predict(
            embeddings[:, :3],
            output["act_emb"][:, :3],
        )
        prediction_error = torch.square(
            predictions - embeddings[:, 1:]
        )
        prediction_loss = prediction_error.mean()
        prediction_context_branch = torch.square(
            predictions - embeddings[:, 1:].detach()
        ).mean()
        prediction_target_branch = torch.square(
            predictions.detach() - embeddings[:, 1:]
        ).mean()
        prediction_by_transition = [
            prediction_error[:, transition].mean()
            for transition in range(3)
        ]
        distance_t1 = paired_distance(embeddings, time_index=1)
        distance_t3 = paired_distance(embeddings, time_index=3)

        torch.manual_seed(args.projection_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.projection_seed)
        native_sigreg_loss = native_regularizer(
            embeddings.transpose(0, 1)
        )
        torch.manual_seed(args.projection_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.projection_seed)
        conditional_sigreg_loss = conditional_regularizer(
            embeddings.transpose(0, 1),
            pairs=pair_indices,
            active=active,
        )

    distance_gradients = {
        "probe_revealed_t1": gradients(distance_t1, parameters),
        "history_conditioned_future_t3": gradients(
            distance_t3,
            parameters,
        ),
    }
    loss_gradients = {
        "prediction_mse": gradients(prediction_loss, parameters),
        "prediction_context_branch": gradients(
            prediction_context_branch,
            parameters,
        ),
        "prediction_target_branch": gradients(
            prediction_target_branch,
            parameters,
        ),
        "prediction_transition_t0_to_t1": gradients(
            prediction_by_transition[0],
            parameters,
        ),
        "prediction_transition_t1_to_t2": gradients(
            prediction_by_transition[1],
            parameters,
        ),
        "prediction_transition_t2_to_t3": gradients(
            prediction_by_transition[2],
            parameters,
        ),
        "native_sigreg_unweighted": gradients(
            native_sigreg_loss,
            parameters,
        ),
        "conditional_sigreg_unweighted": gradients(
            conditional_sigreg_loss,
            parameters,
        ),
    }
    loss_gradients["native_total"] = combine_gradients(
        loss_gradients["prediction_mse"],
        loss_gradients["native_sigreg_unweighted"],
        right_scale=args.sigreg_weight,
    )
    loss_gradients["conditional_total"] = combine_gradients(
        loss_gradients["prediction_mse"],
        loss_gradients["conditional_sigreg_unweighted"],
        right_scale=args.sigreg_weight,
    )

    effects = {
        distance_name: {
            loss_name: gradient_effect(
                names,
                distance_values,
                loss_values,
            )
            for loss_name, loss_values in loss_gradients.items()
        }
        for distance_name, distance_values in distance_gradients.items()
    }
    manifest = args.data_root / "manifest.json"
    build_report = args.data_root / "build_report.json"
    report: dict[str, Any] = {
        "schema_version": 1,
        "benchmark": "pusht_hidden_actuation_history3_action_coverage_v2",
        "diagnostic": "exact_first_paired_batch_gradient_direction",
        "sign_convention": {
            "quantity": (
                "d(pair_distance)/d(learning_rate) at learning_rate=0 "
                "under gradient descent"
            ),
            "negative": "contracts_hidden_gain_pair_distance",
            "positive": "expands_hidden_gain_pair_distance",
        },
        "protocol": {
            "training_seed": args.seed,
            "batch_size": args.batch_size,
            "pair_count": args.batch_size // 2,
            "sigreg_weight": args.sigreg_weight,
            "num_projections": args.num_projections,
            "projection_seed": args.projection_seed,
            "active_conditional_times": [1, 3],
            "first_batch_indices": [int(value) for value in indices],
            "model_visible_inputs": ["pixels", "normalized_actions"],
            "hidden_mode_labels_exposed": False,
        },
        "receipts": {
            "checkpoint": checkpoint_receipt,
            "data_root": str(args.data_root),
            "manifest_sha256": pilot.file_sha256(manifest),
            "build_report_sha256": pilot.file_sha256(build_report),
            "first_batch_pixels_sha256": tensor_sha256([raw_pixels]),
            "first_batch_actions_sha256": tensor_sha256([raw_actions]),
        },
        "scalar_values": {
            "prediction_mse": float(prediction_loss.detach()),
            "prediction_context_branch": float(
                prediction_context_branch.detach()
            ),
            "prediction_target_branch": float(
                prediction_target_branch.detach()
            ),
            "prediction_by_transition": {
                f"t{index}_to_t{index + 1}": float(value.detach())
                for index, value in enumerate(prediction_by_transition)
            },
            "native_sigreg_unweighted": float(
                native_sigreg_loss.detach()
            ),
            "native_sigreg_weighted": float(
                args.sigreg_weight * native_sigreg_loss.detach()
            ),
            "conditional_sigreg_unweighted": float(
                conditional_sigreg_loss.detach()
            ),
            "conditional_sigreg_weighted": float(
                args.sigreg_weight * conditional_sigreg_loss.detach()
            ),
            "pair_distance": {
                "probe_revealed_t1": float(distance_t1.detach()),
                "history_conditioned_future_t3": float(
                    distance_t3.detach()
                ),
            },
        },
        "gradient_effects": effects,
        "interpretation_scope": [
            (
                "This local diagnostic establishes the initial optimization "
                "direction, not a full training trajectory."
            ),
            (
                "The t0-to-t1 target is irreducibly ambiguous because paired "
                "samples have identical t0 pixels/actions and different t1 "
                "futures."
            ),
            (
                "The t2-to-t3 target is identifiable from the t1 observation "
                "inside History-3 despite the common t2 query."
            ),
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report["scalar_values"], indent=2, sort_keys=True))
    for distance_name in effects:
        print(f"[{distance_name}]")
        for loss_name in (
            "prediction_mse",
            "prediction_transition_t0_to_t1",
            "prediction_transition_t2_to_t3",
            "native_sigreg_unweighted",
            "conditional_sigreg_unweighted",
            "native_total",
            "conditional_total",
        ):
            summary = effects[distance_name][loss_name]["all"]
            print(
                f"  {loss_name}: "
                f"{summary['predicted_distance_change_per_unit_lr']:.6g} "
                f"({summary['effect']})"
            )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
