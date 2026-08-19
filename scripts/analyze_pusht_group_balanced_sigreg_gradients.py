#!/usr/bin/env python3
"""Pre-training gradient gate for group-balanced SIGReg on mixed Push-T."""

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
from stable_worldmodel.wm.loss import (  # noqa: E402
    ConditionalSIGReg,
    GroupBalancedSIGReg,
    SIGReg,
)


DEFAULT_OUTPUT = artifact_path(
    "evaluation/history3/pusht_hidden_actuation_h3_v1/"
    "group_balanced_sigreg_seed4097_step2048/"
    "first_batch_gradient_report.json"
)
ACTIVE_TIMES = (1, 3)


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
    parser.add_argument("--seed", type=int, default=4097)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--regularizer-weight", type=float, default=0.02)
    parser.add_argument("--num-projections", type=int, default=1024)
    parser.add_argument("--projection-seed", type=int, default=97031)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def set_projection_seed(seed: int, device: torch.device) -> None:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def pair_distance(
    embeddings: torch.Tensor,
    pairs: torch.Tensor,
    *,
    time_index: int,
) -> torch.Tensor:
    return torch.square(
        embeddings[pairs[:, 0], time_index]
        - embeddings[pairs[:, 1], time_index]
    ).mean()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.batch_size % 4:
        raise ValueError("--batch-size must be positive and divisible by 4")
    if args.regularizer_weight not in {0.02, 0.05}:
        raise ValueError(
            "Registered gradient screens freeze weight at 0.02 or 0.05"
        )
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
        hidden_batch_size // 2,
        dtype=torch.bool,
        device=device,
    )
    active[list(ACTIVE_TIMES)] = True

    model, checkpoint_receipt = pilot.load_model(
        args.checkpoint,
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
    native = SIGReg(
        knots=17,
        num_proj=args.num_projections,
    ).to(device)
    difference = ConditionalSIGReg(
        knots=17,
        num_proj=args.num_projections,
        randomize_pair_orientation=True,
    ).to(device)
    balanced = GroupBalancedSIGReg(
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
        prediction_loss = torch.square(
            prediction - embeddings[:, 1:]
        ).mean()
        distances = {
            "probe_revealed_t1": pair_distance(
                embeddings,
                pairs,
                time_index=1,
            ),
            "history_conditioned_future_t3": pair_distance(
                embeddings,
                pairs,
                time_index=3,
            ),
        }

        set_projection_seed(args.projection_seed, device)
        native_loss = native(embeddings.transpose(0, 1))
        set_projection_seed(args.projection_seed, device)
        difference_loss = difference(
            embeddings.transpose(0, 1),
            pairs=pairs,
            active=active,
        )
        set_projection_seed(args.projection_seed, device)
        balanced_loss = balanced(
            embeddings.transpose(0, 1),
            pairs=pairs,
            active=active,
        )

    distance_gradients = {
        name: base.gradients(value, parameters)
        for name, value in distances.items()
    }
    prediction_gradients = base.gradients(prediction_loss, parameters)
    native_gradients = base.gradients(native_loss, parameters)
    difference_gradients = base.gradients(difference_loss, parameters)
    balanced_gradients = base.gradients(balanced_loss, parameters)
    total_gradients = base.combine_gradients(
        prediction_gradients,
        balanced_gradients,
        right_scale=args.regularizer_weight,
    )
    losses = {
        "prediction_mse": prediction_gradients,
        "native_sigreg_unweighted": native_gradients,
        "paired_difference_sigreg_unweighted": difference_gradients,
        "group_balanced_sigreg_unweighted": balanced_gradients,
        "group_balanced_total": total_gradients,
    }
    effects = {
        distance_name: {
            loss_name: base.gradient_effect(
                names,
                distance_values,
                loss_values,
            )
            for loss_name, loss_values in losses.items()
        }
        for distance_name, distance_values in distance_gradients.items()
    }

    native_embedding_gradient = torch.autograd.grad(
        native_loss,
        embeddings,
        retain_graph=True,
    )[0]
    difference_embedding_gradient = torch.autograd.grad(
        difference_loss,
        embeddings,
    )[0]
    active_index = torch.tensor(
        ACTIVE_TIMES,
        device=device,
        dtype=torch.long,
    )
    ordinary_gradient_norm = float(
        native_embedding_gradient[
            :original_batch_size, active_index
        ].float().norm()
    )
    hidden_difference_gradient_norm = float(
        difference_embedding_gradient[
            original_batch_size:, active_index
        ].float().norm()
    )
    total_t1 = effects["probe_revealed_t1"][
        "group_balanced_total"
    ]["all"]["predicted_distance_change_per_unit_lr"]
    total_t3 = effects["history_conditioned_future_t3"][
        "group_balanced_total"
    ]["all"]["predicted_distance_change_per_unit_lr"]
    gates = {
        "total_descent_expands_probe_pair_distance": total_t1 > 0.0,
        "total_descent_expands_history_conditioned_future_pair_distance": (
            total_t3 > 0.0
        ),
        "marginal_branch_gradient_on_original_rows_at_active_times_nonzero": (
            ordinary_gradient_norm > 0.0
        ),
        "paired_difference_branch_gradient_on_hidden_rows_at_active_times_nonzero": (
            hidden_difference_gradient_norm > 0.0
        ),
    }
    report: dict[str, Any] = {
        "schema_version": 1,
        "benchmark": "pusht_hidden_actuation_history3_action_coverage_v2",
        "diagnostic": "group_balanced_sigreg_exact_first_mixed_batch",
        "status": (
            "gradient_gate_passed"
            if all(gates.values())
            else "gradient_gate_failed_stop_before_training"
        ),
        "protocol": {
            "training_seed": args.seed,
            "batch_size": args.batch_size,
            "original_samples": original_batch_size,
            "hidden_samples": hidden_batch_size,
            "hidden_pairs": hidden_batch_size // 2,
            "regularizer_weight": args.regularizer_weight,
            "effective_active_group_weight": (
                args.regularizer_weight * 0.5
            ),
            "num_projections": args.num_projections,
            "projection_seed": args.projection_seed,
            "active_times": list(ACTIVE_TIMES),
            "hidden_rule_or_class_labels_exposed": False,
        },
        "receipts": {
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
            "native_sigreg_unweighted": float(native_loss.detach()),
            "paired_difference_sigreg_unweighted": float(
                difference_loss.detach()
            ),
            "group_balanced_sigreg_unweighted": float(
                balanced_loss.detach()
            ),
            "pair_distance": {
                name: float(value.detach())
                for name, value in distances.items()
            },
            "ordinary_active_embedding_gradient_norm": (
                ordinary_gradient_norm
            ),
            "hidden_difference_active_embedding_gradient_norm": (
                hidden_difference_gradient_norm
            ),
        },
        "gradient_effects": effects,
        "gates": gates,
        "all_gates_passed": all(gates.values()),
        "interpretation_scope": [
            "This local derivative authorizes training but is not a final model result.",
            "Nonzero marginal coverage is necessary but does not establish standard Push-T CEM retention.",
            "The external coefficient was derived before this diagnostic and is not selected from its result.",
        ],
    }
    output_path = Path(os.path.abspath(args.output.expanduser()))
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "output": str(output_path),
                "status": report["status"],
                "scalar_values": report["scalar_values"],
                "gates": gates,
                "gradient_total_t1": total_t1,
                "gradient_total_t3": total_t3,
            },
            indent=2,
            sort_keys=True,
        )
    )
    del original_dataset
    del original_loader


if __name__ == "__main__":
    main()
