#!/usr/bin/env python3
"""Train a low-capacity residual head on frozen PushT History-3 features.

Only Training and Development are opened.  The original Stable-WorldModel
encoder, projector, predictor, action encoder, and prediction projector stay
frozen.  The head receives only three model-visible latents and their three
action embeddings.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
STABLE_WORLD_MODEL_ROOT = ROOT.parent / "stable-worldmodel"
for source_root in (ROOT, STABLE_WORLD_MODEL_ROOT, Path(__file__).parent):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from contextworld.benchmarks.contact_friction_icl_data import (  # noqa: E402
    DEFAULT_CONTACT_FRICTION_RELEASE_CONFIG,
    _read_lance_pairs as read_contact_pairs,
    directory_sha256,
    file_sha256,
    load_contact_friction_icl_release,
)
from contextworld.benchmarks.motion_damping_icl_data import (  # noqa: E402
    DEFAULT_MOTION_DAMPING_RELEASE_CONFIG,
    _read_lance_pairs as read_damping_pairs,
    load_motion_damping_icl_release,
)
from contextworld.paths import resolve_contextworld_path  # noqa: E402
from contextworld.training.pusht_history_residual import (  # noqa: E402
    FrozenHistoryResidualHead,
    complete_pair_batch_stream,
    complete_twin_group_batch_stream,
    paired_center_response_loss,
    paired_prediction_metrics,
)
import run_pusht_hidden_actuation_mixed as mixed  # noqa: E402
import run_pusht_hidden_actuation_pilot as pilot  # noqa: E402


@dataclass(frozen=True)
class FrozenRows:
    latents: torch.Tensor
    action_embeddings: torch.Tensor
    base_prediction: torch.Tensor
    target: torch.Tensor

    @property
    def count(self) -> int:
        return int(self.latents.shape[0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--capability",
        choices=("contact_friction", "motion_damping"),
        required=True,
    )
    parser.add_argument("--release-config", type=Path, default=None)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help=(
            "Optional Training expansion root. Its Development table must "
            "carry a passed byte-reuse receipt from the formal release."
        ),
    )
    parser.add_argument("--source-checkpoint", type=Path, default=pilot.DEFAULT_CHECKPOINT)
    parser.add_argument("--original-lance", type=Path, default=mixed.DEFAULT_ORIGINAL_LANCE)
    parser.add_argument(
        "--action-normalizer-source",
        type=Path,
        default=pilot.DEFAULT_ORIGINAL_DATASET,
    )
    parser.add_argument("--seed", type=int, default=25001)
    parser.add_argument("--optimizer-steps", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--original-batch-size", type=int, default=64)
    parser.add_argument("--standard-pool-count", type=int, default=4096)
    parser.add_argument("--standard-eval-count", type=int, default=512)
    parser.add_argument("--encoder-batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--hidden-loss-weight", type=float, default=1.0)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--device", default="cuda:6")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


@torch.no_grad()
def freeze_model(model: torch.nn.Module) -> None:
    model.requires_grad_(False)
    model.eval()


@torch.no_grad()
def encode_batch(
    model: torch.nn.Module,
    *,
    pixels: torch.Tensor,
    normalized_actions: torch.Tensor,
    device: torch.device,
) -> FrozenRows:
    processed = pilot.preprocess_pixels(pixels, device)
    actions = normalized_actions.to(device=device, non_blocking=True)
    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        output = model.encode({"pixels": processed, "action": actions})
        latents = output["emb"]
        action_embeddings = output["act_emb"]
        base = model.predict(latents[:, :3], action_embeddings[:, :3])[:, -1]
    return FrozenRows(
        latents=latents[:, :3].float().cpu(),
        action_embeddings=action_embeddings[:, :3].float().cpu(),
        base_prediction=base.float().cpu(),
        target=latents[:, 3].float().cpu(),
    )


def concatenate_rows(rows: list[FrozenRows]) -> FrozenRows:
    return FrozenRows(
        latents=torch.cat([row.latents for row in rows]),
        action_embeddings=torch.cat([row.action_embeddings for row in rows]),
        base_prediction=torch.cat([row.base_prediction for row in rows]),
        target=torch.cat([row.target for row in rows]),
    )


def encode_paired_arrays(
    model: torch.nn.Module,
    arrays: Any,
    *,
    action_stats: dict[str, Any],
    device: torch.device,
    batch_pairs: int,
) -> FrozenRows:
    rows = []
    for start in range(0, arrays.pair_count, batch_pairs):
        stop = min(start + batch_pairs, arrays.pair_count)
        pixels = np.stack(
            [arrays.low_pixels[start:stop], arrays.high_pixels[start:stop]],
            axis=1,
        ).reshape(-1, 4, *arrays.low_pixels.shape[2:])
        pixel_tensor = torch.from_numpy(pixels.copy()).permute(0, 1, 4, 2, 3)
        raw_actions = torch.from_numpy(
            arrays.raw_action_blocks[start:stop].copy()
        ).reshape(stop - start, 4, -1)
        normalized = pilot.normalize_action_blocks(
            raw_actions.float(), action_stats
        ).repeat_interleave(2, dim=0)
        rows.append(
            encode_batch(
                model,
                pixels=pixel_tensor,
                normalized_actions=normalized,
                device=device,
            )
        )
    return concatenate_rows(rows)


def encode_standard_replay(
    model: torch.nn.Module,
    *,
    original_lance: Path,
    action_stats: dict[str, Any],
    seed: int,
    sample_count: int,
    batch_size: int,
    device: torch.device,
) -> FrozenRows:
    if sample_count % batch_size:
        raise ValueError("Standard sample count must divide by batch size")
    _, loader = mixed.original_loader(
        original_lance,
        batch_size=batch_size,
        seed=seed,
        num_workers=0,
    )
    iterator = iter(loader)
    rows = []
    for _ in range(sample_count // batch_size):
        raw = next(iterator)
        normalized = pilot.normalize_action_blocks(
            torch.nan_to_num(raw["action"].float(), 0.0),
            action_stats,
        )
        rows.append(
            encode_batch(
                model,
                pixels=raw["pixels"],
                normalized_actions=normalized,
                device=device,
            )
        )
    del iterator
    del loader
    return concatenate_rows(rows)


def take(rows: FrozenRows, indices: torch.Tensor, device: torch.device) -> FrozenRows:
    return FrozenRows(
        latents=rows.latents[indices].to(device),
        action_embeddings=rows.action_embeddings[indices].to(device),
        base_prediction=rows.base_prediction[indices].to(device),
        target=rows.target[indices].to(device),
    )


@torch.no_grad()
def corrected_prediction(
    head: FrozenHistoryResidualHead,
    rows: FrozenRows,
    *,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    head.eval()
    predictions = []
    for start in range(0, rows.count, batch_size):
        indices = torch.arange(start, min(start + batch_size, rows.count))
        batch = take(rows, indices, device)
        predictions.append(
            (
                batch.base_prediction
                + head(batch.latents, batch.action_embeddings)
            ).float().cpu()
        )
    return torch.cat(predictions)


@torch.no_grad()
def evaluate_hidden(
    head: FrozenHistoryResidualHead,
    rows: FrozenRows,
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    prediction = corrected_prediction(
        head, rows, device=device, batch_size=batch_size
    )
    result = paired_prediction_metrics(prediction=prediction, target=rows.target)
    target_delta = rows.target[1::2] - rows.target[0::2]
    predicted_delta = prediction[1::2] - prediction[0::2]
    target_scale = target_delta.square().mean(dim=-1)
    prediction_scale = predicted_delta.square().mean(dim=-1)
    result["prediction_to_target_pair_mse_ratio"] = float(
        prediction_scale.mean() / target_scale.mean().clamp_min(1e-12)
    )
    result["response_cosine_mean"] = float(
        torch.nn.functional.cosine_similarity(
            predicted_delta, target_delta, dim=-1, eps=1e-8
        ).mean()
    )
    return result


@torch.no_grad()
def evaluate_standard(
    head: FrozenHistoryResidualHead,
    rows: FrozenRows,
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    corrected = corrected_prediction(
        head, rows, device=device, batch_size=batch_size
    )
    baseline_mse = float((rows.base_prediction - rows.target).square().mean())
    corrected_mse = float((corrected - rows.target).square().mean())
    return {
        "sample_count": rows.count,
        "base_final_transition_mse": baseline_mse,
        "corrected_final_transition_mse": corrected_mse,
        "corrected_to_base_mse_ratio": corrected_mse / baseline_mse,
    }


def motion_twin_audit(arrays: Any) -> dict[str, Any]:
    if arrays.pair_count % 2:
        raise RuntimeError("Motion damping requires an even pair count")
    forward_low = arrays.low_pixels[0::2, 0]
    forward_high = arrays.high_pixels[0::2, 0]
    reverse_low = arrays.low_pixels[1::2, 0]
    reverse_high = arrays.high_pixels[1::2, 0]
    low_to_reverse_high = np.all(forward_low == reverse_high, axis=(1, 2, 3))
    high_to_reverse_low = np.all(forward_high == reverse_low, axis=(1, 2, 3))
    passed = bool(low_to_reverse_high.all() and high_to_reverse_low.all())
    if not passed:
        raise RuntimeError("Motion forward/reverse x0 twin audit failed")
    return {
        "group_count": arrays.pair_count // 2,
        "four_rows_per_group": True,
        "forward_low_x0_equals_reverse_high_x0": bool(low_to_reverse_high.all()),
        "forward_high_x0_equals_reverse_low_x0": bool(high_to_reverse_low.all()),
        "passed": passed,
    }


def expanded_data_contract(
    *, data_root: Path, formal_release: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, int], dict[str, str]]:
    """Validate a Training expansion while freezing formal Development."""

    manifest_path = data_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("passed") is not True:
        raise RuntimeError("Expanded data manifest did not pass its audit")
    split_rows = manifest.get("splits", {})
    required_splits = {"train", "loader_validation"}
    if not required_splits.issubset(split_rows):
        raise RuntimeError("Expanded data manifest lacks Training/Development")
    counts = {
        split: int(split_rows[split]["pair_count"])
        for split in required_splits
    }
    manifest_counts = manifest.get("pair_counts", {})
    if any(int(manifest_counts.get(split, -1)) != counts[split] for split in counts):
        raise RuntimeError("Expanded manifest pair counts are inconsistent")
    tables = {
        split: str(split_rows[split]["table_path"])
        for split in required_splits
    }
    for split, relative in tables.items():
        observed = directory_sha256(data_root / relative)
        expected = str(split_rows[split]["table_sha256"])
        if observed != expected:
            raise RuntimeError(
                f"Expanded {split} table hash mismatch: {observed} != {expected}"
            )

    development = split_rows["loader_validation"]
    receipt = development.get("frozen_split_reuse", {})
    actual_manifest_sha256 = file_sha256(manifest_path)
    formal_manifest_sha256 = str(formal_release["data"]["manifest_sha256"])
    reused_tables = manifest.get("evaluation_tables_reused_byte_for_byte", {})
    reuse_source = manifest.get("evaluation_reuse_source", {})
    declared_source_manifest_sha256 = str(
        reused_tables.get("source_manifest_sha256")
        or reuse_source.get("manifest_sha256")
        or ""
    )
    formal_is_actual_release = actual_manifest_sha256 == formal_manifest_sha256
    release_identity_is_consistent = (
        formal_is_actual_release
        and bool(declared_source_manifest_sha256)
    ) or (
        not formal_is_actual_release
        and declared_source_manifest_sha256 == formal_manifest_sha256
    )
    required_true = (
        receipt.get("passed") is True
        and receipt.get("pair_identity_preserved") is True
        and receipt.get("model_visible_bytes_preserved") is True
    )
    table_sha256 = str(development["table_sha256"])
    hashes_match = (
        release_identity_is_consistent
        and receipt.get("source_manifest_sha256")
        == declared_source_manifest_sha256
        and receipt.get("source_table_sha256") == table_sha256
        and receipt.get("destination_table_sha256") == table_sha256
    )
    if not required_true or not hashes_match:
        raise RuntimeError("Expanded Development byte-reuse receipt failed")
    file_checks = []
    development_table_root = (data_root / tables["loader_validation"]).resolve()
    for row in receipt.get("file_receipts", []):
        receipt_path = Path(str(row["path"]))
        if receipt_path.is_absolute() or ".." in receipt_path.parts:
            raise RuntimeError(
                "Expanded Development receipt contains an unsafe path: "
                f"{receipt_path}"
            )
        table_relative = Path(tables["loader_validation"])
        if receipt_path.parts[: len(table_relative.parts)] == table_relative.parts:
            path = (data_root / receipt_path).resolve()
            path_base = "data_root"
        else:
            path = (development_table_root / receipt_path).resolve()
            path_base = "development_table"
        try:
            path.relative_to(development_table_root)
        except ValueError as error:
            raise RuntimeError(
                "Expanded Development receipt points outside its table: "
                f"{receipt_path}"
            ) from error
        passed = (
            path.is_file()
            and path.stat().st_size == int(row["bytes"])
            and file_sha256(path) == str(row["sha256"])
        )
        file_checks.append(
            {
                "receipt_path": receipt_path.as_posix(),
                "path_base": path_base,
                "resolved_path": str(path),
                "passed": passed,
            }
        )
    if not file_checks or not all(row["passed"] for row in file_checks):
        raise RuntimeError("Expanded Development file receipt failed")
    audit = {
        "override_enabled": True,
        "root": str(data_root),
        "manifest_sha256": actual_manifest_sha256,
        "formal_manifest_sha256": formal_manifest_sha256,
        "formal_is_actual_release": formal_is_actual_release,
        "declared_evaluation_source_manifest_sha256": (
            declared_source_manifest_sha256
        ),
        "loader_validation_frozen_split_reuse": receipt,
        "destination_file_checks": file_checks,
        "passed": True,
    }
    return audit, counts, tables


def main() -> None:
    args = parse_args()
    output = Path(os.path.abspath(args.output.expanduser()))
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite output: {output}")
    if args.batch_size <= 0 or args.original_batch_size <= 0:
        raise ValueError("Batch sizes must be positive")
    hidden_batch_size = args.batch_size - args.original_batch_size
    if hidden_batch_size <= 0 or hidden_batch_size % 4:
        raise ValueError("Hidden batch size must be a positive multiple of four")
    if args.optimizer_steps <= 0 or args.hidden_loss_weight <= 0:
        raise ValueError("Training controls must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    pilot.set_reproducible_seed(args.seed)

    if args.capability == "contact_friction":
        release_path = args.release_config or DEFAULT_CONTACT_FRICTION_RELEASE_CONFIG
        release = load_contact_friction_icl_release(release_path)
        reader = read_contact_pairs
    else:
        release_path = args.release_config or DEFAULT_MOTION_DAMPING_RELEASE_CONFIG
        release = load_motion_damping_icl_release(release_path)
        reader = read_damping_pairs
    release_path = Path(release_path).expanduser().resolve()
    formal_data_root = resolve_contextworld_path(
        release["data"]["artifact_tree"]["root"], repo_root=ROOT
    )
    if args.data_root is None:
        data_root = formal_data_root
        counts = {
            split: int(release["data"]["pair_counts"][split])
            for split in ("train", "loader_validation")
        }
        tables = {
            split: str(release["data"]["lance_tables"][split])
            for split in ("train", "loader_validation")
        }
        data_contract = {
            "override_enabled": False,
            "root": str(data_root),
            "manifest_sha256": release["data"]["manifest_sha256"],
            "passed": True,
        }
    else:
        data_root = args.data_root.expanduser().resolve()
        data_contract, counts, tables = expanded_data_contract(
            data_root=data_root,
            formal_release=release,
        )
    train_arrays = reader(
        data_root / tables["train"],
        expected_pairs=int(counts["train"]),
        expected_split="train",
    )
    development_arrays = reader(
        data_root / tables["loader_validation"],
        expected_pairs=int(counts["loader_validation"]),
        expected_split="loader_validation",
    )
    twin_audit = (
        motion_twin_audit(train_arrays)
        if args.capability == "motion_damping"
        else None
    )

    source_checkpoint = args.source_checkpoint.expanduser().resolve()
    original_lance = args.original_lance.expanduser().resolve()
    action_source = args.action_normalizer_source.expanduser().resolve()
    action_stats = pilot.original_action_stats(action_source)
    model, source_receipt = pilot.load_model(source_checkpoint, device=device)
    freeze_model(model)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("The base Stable-WorldModel was not fully frozen")

    started = time.monotonic()
    print("Precomputing frozen hidden Training features", flush=True)
    hidden_train = encode_paired_arrays(
        model,
        train_arrays,
        action_stats=action_stats,
        device=device,
        batch_pairs=args.encoder_batch_size // 2,
    )
    print("Precomputing frozen hidden Development features", flush=True)
    hidden_development = encode_paired_arrays(
        model,
        development_arrays,
        action_stats=action_stats,
        device=device,
        batch_pairs=args.encoder_batch_size // 2,
    )
    print("Precomputing frozen standard Training replay", flush=True)
    standard_train = encode_standard_replay(
        model,
        original_lance=original_lance,
        action_stats=action_stats,
        seed=args.seed,
        sample_count=args.standard_pool_count,
        batch_size=args.encoder_batch_size,
        device=device,
    )
    print("Precomputing fixed standard evaluation replay", flush=True)
    standard_evaluation = encode_standard_replay(
        model,
        original_lance=original_lance,
        action_stats=action_stats,
        seed=12289,
        sample_count=args.standard_eval_count,
        batch_size=args.encoder_batch_size,
        device=device,
    )

    latent_dim = hidden_train.latents.shape[-1]
    action_dim = hidden_train.action_embeddings.shape[-1]
    head = FrozenHistoryResidualHead(
        latent_dim=latent_dim,
        action_dim=action_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.optimizer_steps, eta_min=0.0
    )
    baseline_standard_scale = (
        (standard_train.base_prediction - standard_train.target)
        .square()
        .mean()
        .clamp_min(1e-8)
        .to(device)
    )
    hidden_stream = iter(
        complete_twin_group_batch_stream(
            pair_count=train_arrays.pair_count,
            rows_per_batch=hidden_batch_size,
            seed=args.seed,
        )
        if args.capability == "motion_damping"
        else complete_pair_batch_stream(
            pair_count=train_arrays.pair_count,
            rows_per_batch=hidden_batch_size,
            seed=args.seed,
        )
    )
    standard_generator = torch.Generator().manual_seed(args.seed + 1)
    local_pairs = torch.arange(
        hidden_batch_size, device=device, dtype=torch.long
    ).reshape(-1, 2)
    snapshot_steps = {
        value
        for value in (0, 128, 512, 1024, 2048, 4096, 8192)
        if value <= args.optimizer_steps
    } | {args.optimizer_steps}
    snapshots = []
    trace = []

    def snapshot(step: int) -> None:
        row = {
            "optimizer_step": step,
            "training": evaluate_hidden(
                head,
                hidden_train,
                device=device,
                batch_size=args.encoder_batch_size,
            ),
            "development": evaluate_hidden(
                head,
                hidden_development,
                device=device,
                batch_size=args.encoder_batch_size,
            ),
            "standard_replay": evaluate_standard(
                head,
                standard_evaluation,
                device=device,
                batch_size=args.encoder_batch_size,
            ),
        }
        snapshots.append(row)
        dev = row["development"]
        print(
            f"step={step} future={dev['correct_future_rate']:.4f} "
            f"history={dev['correct_history_rate']:.4f} "
            f"switch={dev['context_switch_rate']:.4f} "
            f"worst={dev['worst_mode_correct_future_rate']:.4f} "
            "standard_ratio="
            f"{row['standard_replay']['corrected_to_base_mse_ratio']:.4f}",
            flush=True,
        )

    snapshot(0)
    for step in range(1, args.optimizer_steps + 1):
        head.train()
        hidden_indices = next(hidden_stream)
        standard_indices = torch.randint(
            standard_train.count,
            (args.original_batch_size,),
            generator=standard_generator,
        )
        hidden_batch = take(hidden_train, hidden_indices, device)
        standard_batch = take(standard_train, standard_indices, device)
        hidden_prediction = hidden_batch.base_prediction + head(
            hidden_batch.latents, hidden_batch.action_embeddings
        )
        standard_prediction = standard_batch.base_prediction + head(
            standard_batch.latents, standard_batch.action_embeddings
        )
        standard_mse = (
            standard_prediction - standard_batch.target
        ).square().mean()
        hidden_loss, hidden_components = paired_center_response_loss(
            prediction=hidden_prediction,
            target=hidden_batch.target,
            pair_indices=local_pairs,
        )
        loss = (
            standard_mse / baseline_standard_scale
            + args.hidden_loss_weight * hidden_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            head.parameters(), args.gradient_clip_norm
        )
        optimizer.step()
        scheduler.step()
        if step == 1 or step in snapshot_steps:
            trace.append(
                {
                    "optimizer_step": step,
                    "loss": float(loss.detach()),
                    "standard_mse": float(standard_mse.detach()),
                    "standard_normalized_loss": float(
                        (standard_mse / baseline_standard_scale).detach()
                    ),
                    "hidden_loss": float(hidden_loss.detach()),
                    "hidden_components": {
                        name: float(value.detach())
                        for name, value in hidden_components.items()
                    },
                    "gradient_norm_before_clip": float(gradient_norm),
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                }
            )
        if step in snapshot_steps:
            snapshot(step)

    final_development = snapshots[-1]["development"]
    thresholds = release["scoring"]["hidden_future_prediction"]["gates"]
    gate = {
        "correct_future_rate": final_development["correct_future_rate"]
        >= float(thresholds["correct_future_rate_minimum"]),
        "correct_history_rate": final_development["correct_history_rate"]
        >= float(thresholds["correct_history_rate_minimum"]),
        "context_switch_rate": final_development["context_switch_rate"]
        >= float(thresholds["context_switch_rate_minimum"]),
        "worst_mode_correct_future_rate": final_development[
            "worst_mode_correct_future_rate"
        ]
        >= float(
            thresholds[
                "worst_friction_correct_future_rate_minimum"
                if args.capability == "contact_friction"
                else "worst_damping_correct_future_rate_minimum"
            ]
        ),
    }
    output.mkdir(parents=True)
    head_state = {name: value.detach().cpu() for name, value in head.state_dict().items()}
    head_path = output / "history_residual_head.pt"
    torch.save(head_state, head_path)
    report = {
        "schema_version": 1,
        "status": "completed_training_and_development_only",
        "public_test_opened": False,
        "capability": args.capability,
        "release": {
            "config": str(release_path),
            "config_sha256": file_sha256(release_path),
            "manifest_sha256": release["data"]["manifest_sha256"],
        },
        "actual_data": data_contract,
        "base_model": {
            **source_receipt,
            "all_parameters_frozen": True,
            "frozen_modules": [
                "encoder",
                "projector",
                "predictor",
                "action_encoder",
                "pred_proj",
            ],
        },
        "head": {
            "type": "FrozenHistoryResidualHead",
            "checkpoint": str(head_path),
            "checkpoint_sha256": pilot.file_sha256(head_path),
            "latent_dim": int(latent_dim),
            "action_embedding_dim": int(action_dim),
            "hidden_dim": args.hidden_dim,
            "trainable_parameter_count": head.trainable_parameter_count,
            "input_fields": [
                "three_frozen_latents",
                "three_frozen_action_embeddings",
            ],
            "forbidden_fields": [
                "hidden_label",
                "pair_id",
                "physics_state",
                "catalog_index",
            ],
        },
        "training": {
            "seed": args.seed,
            "optimizer_steps": args.optimizer_steps,
            "batch": {
                "total": args.batch_size,
                "standard_rows": args.original_batch_size,
                "hidden_rows": hidden_batch_size,
                "motion_complete_twin_groups": (
                    args.capability == "motion_damping"
                ),
            },
            "standard_pool_count": args.standard_pool_count,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "hidden_loss_weight": args.hidden_loss_weight,
            "loss": (
                "standard_final_transition_mse_normalized_by_frozen_base_"
                "plus_log_center_and_response_pair_fit"
            ),
            "motion_twin_audit": twin_audit,
        },
        "fixed_standard_evaluation": {
            "seed": 12289,
            "sample_count": args.standard_eval_count,
            "overlap_role": "diagnostic_replay_not_a_sealed_split",
        },
        "gate": {"checks": gate, "passed": all(gate.values())},
        "trace": trace,
        "snapshots": snapshots,
        "elapsed_seconds": time.monotonic() - started,
    }
    report_path = output / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "gate": report["gate"],
                "development": final_development,
                "standard_replay": snapshots[-1]["standard_replay"],
                "report": str(report_path),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
