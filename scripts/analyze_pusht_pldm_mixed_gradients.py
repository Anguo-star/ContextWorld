#!/usr/bin/env python3
"""Compare native LeWM and PLDM on one exact mixed PushT batch.

The diagnostic measures the infinitesimal gradient-descent change of the
condition-matched hidden-actuation representation distance.  Both objectives
use the same source state, mixed standard/hidden batch, stochastic forward
pass, and model parameters.  No optimizer step is taken.
"""

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
import analyze_pusht_hidden_actuation_gradients as base  # noqa: E402
import run_pusht_hidden_actuation_mixed as mixed  # noqa: E402
import run_pusht_hidden_actuation_pilot as pilot  # noqa: E402
from stable_worldmodel.wm.loss import PLDMLoss, SIGReg  # noqa: E402


DEFAULT_OUTPUT = artifact_path(
    "evaluation/history3/pusht_hidden_actuation_h3_v1/"
    "pldm_control_v1/mixed_first_batch_gradient_seed14337/report.json"
)
PLDM_WEIGHTS = {
    "std_loss": 18.0,
    "std_t_loss": 0.7,
    "cov_loss": 12.0,
    "temp_align_loss": 0.2,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=pilot.DEFAULT_CHECKPOINT)
    parser.add_argument("--hidden-data-root", type=Path, default=pilot.DEFAULT_DATA_ROOT)
    parser.add_argument("--original-lance", type=Path, default=mixed.DEFAULT_ORIGINAL_LANCE)
    parser.add_argument(
        "--action-normalizer-source",
        type=Path,
        default=pilot.DEFAULT_ORIGINAL_DATASET,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=14337)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--projection-seed", type=int, default=114337)
    parser.add_argument("--num-projections", type=int, default=1024)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def hidden_pair_distance(
    embeddings: torch.Tensor,
    *,
    original_batch_size: int,
    time_index: int,
) -> torch.Tensor:
    hidden = embeddings[original_batch_size:, time_index]
    return torch.square(hidden[0::2] - hidden[1::2]).mean()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.batch_size % 4:
        raise ValueError("--batch-size must be positive and divisible by four")
    output = Path(os.path.abspath(args.output.expanduser()))
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    pilot.set_reproducible_seed(args.seed)
    action_stats = pilot.original_action_stats(args.action_normalizer_source)
    hidden = pilot.materialize_lance_split(
        args.hidden_data_root / "train.lance",
        action_stats=action_stats,
    )
    original_batch_size = args.batch_size // 2
    hidden_batch_size = args.batch_size - original_batch_size
    original_dataset, original_loader = mixed.original_loader(
        args.original_lance,
        batch_size=original_batch_size,
        seed=args.seed,
        num_workers=0,
    )
    original_iterator = iter(original_loader)
    original = next(original_iterator)
    hidden_iterator = iter(
        pilot.PairedBatchStream(
            hidden.pair_count,
            batch_size=hidden_batch_size,
            seed=args.seed,
        )
    )
    hidden_indices = next(hidden_iterator)
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

    model, checkpoint_receipt = mixed.load_model_for_variant(
        args.checkpoint,
        variant="mixed_pldm_joint",
        device=device,
    )
    model.train()
    named_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    names = [name for name, _ in named_parameters]
    parameters = [parameter for _, parameter in named_parameters]
    pixels = pilot.preprocess_pixels(raw_pixels, device)
    actions = raw_actions.to(device=device, non_blocking=True)
    sigreg = SIGReg(knots=17, num_proj=args.num_projections).to(device)
    pldm = PLDMLoss().to(device)

    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        encoded = model.encode({"pixels": pixels, "action": actions})
        embeddings = encoded["emb"]
        predictions = model.predict(
            embeddings[:, :3],
            encoded["act_emb"][:, :3],
        )
        prediction_loss = torch.square(
            predictions - embeddings[:, 1:]
        ).mean()
        torch.manual_seed(args.projection_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.projection_seed)
        native_regularizer = sigreg(embeddings.transpose(0, 1))
        pldm_components = pldm(embeddings)
        pldm_regularizer = sum(
            weight * pldm_components[name]
            for name, weight in PLDM_WEIGHTS.items()
        )
        native_total = prediction_loss + 0.09 * native_regularizer
        pldm_total = prediction_loss + pldm_regularizer
        distances = {
            "probe_revealed_x1": hidden_pair_distance(
                embeddings,
                original_batch_size=original_batch_size,
                time_index=1,
            ),
            "history_conditioned_future_x3": hidden_pair_distance(
                embeddings,
                original_batch_size=original_batch_size,
                time_index=3,
            ),
        }

    losses = {
        "prediction_mse": prediction_loss,
        "native_sigreg_unweighted": native_regularizer,
        "native_sigreg_weighted": 0.09 * native_regularizer,
        "native_total": native_total,
        "pldm_regularizers_weighted": pldm_regularizer,
        "pldm_total": pldm_total,
    }
    distance_gradients = {
        name: base.gradients(value, parameters)
        for name, value in distances.items()
    }
    loss_gradients = {
        name: base.gradients(value, parameters)
        for name, value in losses.items()
    }
    effects = {
        distance_name: {
            loss_name: base.gradient_effect(
                names,
                distance_values,
                loss_values,
            )
            for loss_name, loss_values in loss_gradients.items()
        }
        for distance_name, distance_values in distance_gradients.items()
    }

    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "exact_mixed_first_batch_gradient_diagnostic",
        "benchmark": "pusht_hidden_actuation_history3_action_coverage_v2",
        "protocol": {
            "seed": args.seed,
            "batch_size": args.batch_size,
            "standard_rows": original_batch_size,
            "hidden_rows": hidden_batch_size,
            "hidden_pairs": hidden_batch_size // 2,
            "model_mode": "train",
            "precision": (
                "bf16_mixed" if device.type == "cuda" else "float32"
            ),
            "hidden_labels_exposed": False,
            "same_forward_graph_for_both_objectives": True,
            "optimizer_step_taken": False,
        },
        "objective_contract": {
            "native_lewm": {
                "prediction_mse": 1.0,
                "sigreg": 0.09,
            },
            "pldm": {
                "prediction_mse": 1.0,
                **PLDM_WEIGHTS,
            },
        },
        "sign_convention": {
            "quantity": (
                "d(mean squared hidden-pair distance)/d(learning rate) "
                "at learning rate zero under gradient descent"
            ),
            "negative": "contracts_hidden_dynamics_direction",
            "positive": "expands_hidden_dynamics_direction",
        },
        "receipts": {
            "checkpoint": checkpoint_receipt,
            "first_batch_pixels_sha256": base.tensor_sha256([raw_pixels]),
            "first_batch_actions_sha256": base.tensor_sha256([raw_actions]),
            "hidden_indices": [int(value) for value in hidden_indices],
        },
        "scalar_values": {
            "losses": {
                name: float(value.detach()) for name, value in losses.items()
            },
            "pldm_components": {
                name: float(value.detach())
                for name, value in pldm_components.items()
            },
            "pair_distances": {
                name: float(value.detach())
                for name, value in distances.items()
            },
        },
        "gradient_effects": effects,
        "interpretation_scope": [
            "This diagnostic establishes the initial local direction only.",
            "The full training trajectory and held-out prediction metrics are "
            "required to establish learned behavior.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": pilot.file_sha256(output),
                "all_parameter_effects": {
                    distance_name: {
                        loss_name: values["all"]
                        for loss_name, values in loss_effects.items()
                    }
                    for distance_name, loss_effects in effects.items()
                },
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )

    del original_iterator
    del original_loader
    del original_dataset


if __name__ == "__main__":
    main()
