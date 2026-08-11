#!/usr/bin/env python3
"""Train and score Push-T hidden-actuation History-3 objective controls.

This runner is intentionally self-contained and bounded.  It starts every
variant from the same standard Push-T LeWM checkpoint, materializes the
audited paired dataset without exposing hidden labels to the model, and
reports prediction-space selection between the two *real simulator futures*.

The paired trajectory has an important causal asymmetry:

* at t=0 the first probe outcome is not identifiable from model input;
* the resulting t=1 observation reveals the persistent hidden gain;
* recovery produces a common t=2 query state; and
* the same t=2 action has a gain-dependent t=3 outcome.

Thus the last transition is predictable only from history, while the first
one remains genuinely ambiguous.  Joint representation learning can reduce
the irreducible first-transition loss by collapsing the paired target
direction; frozen image representations and conditional regularization are
controls for that shortcut.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import h5py
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf, open_dict


CONTEXTWORLD_ROOT = Path(__file__).resolve().parents[1]
STABLE_WORLD_MODEL_ROOT = CONTEXTWORLD_ROOT.parent / "stable-worldmodel"
for source_root in (CONTEXTWORLD_ROOT, STABLE_WORLD_MODEL_ROOT):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from contextworld.paths import artifact_path  # noqa: E402


DEFAULT_DATA_ROOT = artifact_path(
    "synthesis/pusht_hidden_actuation_h3_v1"
)
DEFAULT_ORIGINAL_DATASET = Path(
    "/opt/huawei/explorer-env/dataset/ag_data/data/world_model/quentinll/"
    "pusht_expert_train.h5"
)
DEFAULT_CHECKPOINT = Path(
    "/opt/huawei/explorer-env/dataset/ag_data/data/world_model/quentinll/"
    "lewm-pusht/ckpt/pusht_lewm_baseline_seed3073/"
    "pusht_lewm_baseline_seed3073_weights.ckpt"
)
CONDITIONAL_SIGREG_WEIGHTS = {
    "conditional_sigreg_0p01": 0.01,
    "conditional_sigreg_0p03": 0.03,
    "conditional_sigreg_0p05": 0.05,
    "conditional_sigreg_0p09": 0.09,
}
VARIANTS = (
    "native_sigreg_0p09",
    "frozen_image_native",
    "pldm_active",
    *CONDITIONAL_SIGREG_WEIGHTS,
)
SNAPSHOT_STEPS = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
ACTION_DIM = 2
ACTION_INPUT_DIM = 10


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def original_action_stats(path: Path) -> dict[str, Any]:
    """Compute exactly the z-score statistic used by column_normalizer."""

    with h5py.File(path, "r", swmr=True) as handle:
        actions = handle["action"]
        count = int(actions.shape[0])
        total = np.zeros(actions.shape[1:], dtype=np.float64)
        square_total = np.zeros(actions.shape[1:], dtype=np.float64)
        for start in range(0, count, 200_000):
            batch = actions[start : start + 200_000].astype(np.float64)
            total += batch.sum(axis=0)
            square_total += np.square(batch).sum(axis=0)
    mean = total / count
    variance = square_total / count - np.square(mean)
    std = np.sqrt(np.maximum(variance, 0.0))
    return {
        "count": count,
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
        "source": str(path),
        "source_size_bytes": path.stat().st_size,
        "method": "population_zscore_identical_to_numpy_std_ddof0",
    }


def normalize_action_blocks(
    actions: torch.Tensor,
    stats: dict[str, Any],
) -> torch.Tensor:
    """Normalize raw actions, preserving each five-action block."""

    original_shape = actions.shape
    action_dim = int(np.asarray(stats["mean"]).size)
    values = actions.reshape(*original_shape[:-1], -1, action_dim)
    mean = torch.as_tensor(stats["mean"], dtype=values.dtype)
    std = torch.as_tensor(stats["std"], dtype=values.dtype)
    values = (values - mean) / std.clamp_min(1e-8)
    return values.reshape(original_shape)


@dataclass
class MaterializedSplit:
    pixels: torch.Tensor
    action: torch.Tensor
    pair_count: int

    def __post_init__(self) -> None:
        if self.pixels.dtype != torch.uint8:
            raise TypeError("Materialized pixels must remain uint8")
        if self.pixels.ndim != 5 or self.pixels.shape[1:] != (
            4,
            3,
            224,
            224,
        ):
            raise ValueError(
                f"Unexpected pixel shape: {tuple(self.pixels.shape)}"
            )
        if self.action.shape != (
            self.pixels.size(0),
            4,
            ACTION_INPUT_DIM,
        ):
            raise ValueError(
                f"Unexpected action shape: {tuple(self.action.shape)}"
            )
        if self.pixels.size(0) != 2 * self.pair_count:
            raise ValueError("Every hidden-actuation pair needs two samples")


def materialize_lance_split(
    path: Path,
    *,
    action_stats: dict[str, Any],
) -> MaterializedSplit:
    from stable_worldmodel.data import LanceDataset

    dataset = LanceDataset(
        path=path,
        frameskip=5,
        num_steps=4,
        keys_to_load=["pixels", "action"],
    )
    samples = [dataset[index] for index in range(len(dataset))]
    pixels = torch.stack([sample["pixels"] for sample in samples])
    actions = torch.stack([sample["action"] for sample in samples]).float()
    actions = normalize_action_blocks(actions, action_stats)
    split = MaterializedSplit(
        pixels=pixels,
        action=actions,
        pair_count=len(samples) // 2,
    )
    for pair_index in range(split.pair_count):
        low = 2 * pair_index
        high = low + 1
        if not torch.equal(split.pixels[low, 0], split.pixels[high, 0]):
            raise RuntimeError("A train pair has unequal initial pixels")
        if not torch.equal(split.pixels[low, 2], split.pixels[high, 2]):
            raise RuntimeError("A train pair has unequal query pixels")
        if not torch.equal(split.action[low], split.action[high]):
            raise RuntimeError("A train pair has unequal action sequence")
        if torch.equal(split.pixels[low, 1], split.pixels[high, 1]):
            raise RuntimeError("A train pair has no visible probe outcome")
        if torch.equal(split.pixels[low, 3], split.pixels[high, 3]):
            raise RuntimeError("A train pair has no distinct future")
    return split


def load_eval_payloads(
    path: Path,
    *,
    action_stats: dict[str, Any],
) -> dict[str, torch.Tensor]:
    files = sorted(path.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No evaluation payloads found under {path}")
    rows: dict[str, list[torch.Tensor]] = {
        "low_pixels": [],
        "high_pixels": [],
        "action": [],
        "low_states": [],
        "high_states": [],
    }
    for payload_path in files:
        with np.load(payload_path) as payload:
            rows["low_pixels"].append(
                torch.from_numpy(payload["low_pixels"].copy()).permute(
                    0, 3, 1, 2
                )
            )
            rows["high_pixels"].append(
                torch.from_numpy(payload["high_pixels"].copy()).permute(
                    0, 3, 1, 2
                )
            )
            low_actions = torch.from_numpy(
                payload["low_actions"].copy()
            ).reshape(4, ACTION_INPUT_DIM)
            high_actions = torch.from_numpy(
                payload["high_actions"].copy()
            ).reshape(4, ACTION_INPUT_DIM)
            if not torch.equal(low_actions, high_actions):
                raise RuntimeError(
                    f"Unequal paired actions in {payload_path}"
                )
            rows["action"].append(low_actions)
            rows["low_states"].append(
                torch.from_numpy(payload["low_states"].copy())
            )
            rows["high_states"].append(
                torch.from_numpy(payload["high_states"].copy())
            )
    result = {name: torch.stack(values) for name, values in rows.items()}
    result["action"] = normalize_action_blocks(
        result["action"].float(),
        action_stats,
    )
    return result


def preprocess_pixels(pixels: torch.Tensor, device: torch.device) -> torch.Tensor:
    values = pixels.to(device=device, non_blocking=True).float().div_(255.0)
    mean = torch.as_tensor(
        IMAGENET_MEAN,
        device=device,
        dtype=values.dtype,
    ).view(1, 1, 3, 1, 1)
    std = torch.as_tensor(
        IMAGENET_STD,
        device=device,
        dtype=values.dtype,
    ).view(1, 1, 3, 1, 1)
    return (values - mean) / std


def instantiate_model() -> torch.nn.Module:
    cfg = OmegaConf.load(
        STABLE_WORLD_MODEL_ROOT / "scripts/train/config/lewm.yaml"
    )
    with open_dict(cfg):
        cfg.model.action_encoder.input_dim = ACTION_INPUT_DIM
    return hydra.utils.instantiate(cfg.model)


def checkpoint_model_state(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    source = payload.get("state_dict", payload)
    state = {
        name[len("model.") :]: value
        for name, value in source.items()
        if name.startswith("model.")
    }
    if not state:
        state = dict(source)
    return state


def load_model(
    checkpoint: Path,
    *,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    model = instantiate_model()
    state = checkpoint_model_state(checkpoint)
    model.load_state_dict(state, strict=True)
    initial_hash = state_sha256(model.state_dict())
    return model.to(device), {
        "path": str(checkpoint),
        "sha256": file_sha256(checkpoint),
        "model_state_sha256": initial_hash,
    }


@torch.no_grad()
def encode_pixels(
    model: torch.nn.Module,
    pixels: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    raw_rows = []
    projected_rows = []
    for start in range(0, pixels.size(0), batch_size):
        values = preprocess_pixels(pixels[start : start + batch_size], device)
        batch = values.size(0)
        frames = values.size(1)
        flattened = values.flatten(0, 1)
        raw = model.encoder(
            flattened,
            interpolate_pos_encoding=True,
        ).last_hidden_state[:, 0]
        projected = model.projector(raw)
        raw_rows.append(raw.reshape(batch, frames, -1).float().cpu())
        projected_rows.append(
            projected.reshape(batch, frames, -1).float().cpu()
        )
    return torch.cat(raw_rows), torch.cat(projected_rows)


@torch.no_grad()
def predict_histories(
    model: torch.nn.Module,
    pixels: torch.Tensor,
    actions: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    predictions = []
    for start in range(0, pixels.size(0), batch_size):
        batch_pixels = preprocess_pixels(
            pixels[start : start + batch_size],
            device,
        )
        batch_actions = actions[start : start + batch_size].to(
            device=device,
            non_blocking=True,
        )
        output = model.encode(
            {
                "pixels": batch_pixels,
                "action": batch_actions,
            }
        )
        prediction = model.predict(output["emb"], output["act_emb"])[:, -1]
        predictions.append(prediction.float().cpu())
    return torch.cat(predictions)


def _effective_rank(values: torch.Tensor) -> float:
    centered = values.double() - values.double().mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(centered)
    energy = singular.square()
    probability = energy / energy.sum().clamp_min(1e-30)
    entropy = -(probability * probability.clamp_min(1e-30).log()).sum()
    return float(entropy.exp())


def _distance_summary(
    low: torch.Tensor,
    high: torch.Tensor,
) -> dict[str, float]:
    paired = torch.linalg.vector_norm(high - low, dim=-1)
    unrelated = torch.linalg.vector_norm(high.roll(1, 0) - low, dim=-1)
    return {
        "paired_mean": float(paired.mean()),
        "paired_min": float(paired.min()),
        "unrelated_mean": float(unrelated.mean()),
        "paired_to_unrelated_ratio": float(
            paired.mean() / unrelated.mean().clamp_min(1e-12)
        ),
    }


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    evaluation: dict[str, torch.Tensor],
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    was_training = model.training
    model.eval()
    low_pixels = evaluation["low_pixels"]
    high_pixels = evaluation["high_pixels"]
    actions = evaluation["action"][:, :3]
    histories = torch.cat([low_pixels[:, :3], high_pixels[:, :3]])
    history_actions = torch.cat([actions, actions])
    predicted = predict_histories(
        model,
        histories,
        history_actions,
        device=device,
        batch_size=batch_size,
    )
    count = low_pixels.size(0)
    predicted_low = predicted[:count]
    predicted_high = predicted[count:]

    future_pixels = torch.cat(
        [low_pixels[:, 3:4], high_pixels[:, 3:4]]
    )
    raw_future, projected_future = encode_pixels(
        model,
        future_pixels,
        device=device,
        batch_size=batch_size,
    )
    raw_low, raw_high = raw_future[:count, 0], raw_future[count:, 0]
    target_low = projected_future[:count, 0]
    target_high = projected_future[count:, 0]

    def mse(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return (left - right).square().mean(dim=-1)

    low_to_low = mse(predicted_low, target_low)
    low_to_high = mse(predicted_low, target_high)
    high_to_low = mse(predicted_high, target_low)
    high_to_high = mse(predicted_high, target_high)
    target_decisions = torch.cat(
        [low_to_low < low_to_high, high_to_high < high_to_low]
    )
    history_decisions = torch.cat(
        [low_to_low < high_to_low, high_to_high < low_to_high]
    )
    switch_alignment = (
        (predicted_high - predicted_low) * (target_high - target_low)
    ).sum(dim=-1)
    correct_losses = torch.cat([low_to_low, high_to_high])
    incorrect_losses = torch.cat([low_to_high, high_to_low])

    state_gap = torch.linalg.vector_norm(
        evaluation["high_states"][:, 3, 2:4]
        - evaluation["low_states"][:, 3, 2:4],
        dim=-1,
    )
    result = {
        "pair_count": count,
        "decision_count": 2 * count,
        "two_real_future_target_selection_rate": float(
            target_decisions.float().mean()
        ),
        "correct_history_preference_rate": float(
            history_decisions.float().mean()
        ),
        "correct_rule_switch_rate": float(
            (switch_alignment > 0).float().mean()
        ),
        "worst_mode_target_selection_rate": float(
            min(
                (low_to_low < low_to_high).float().mean(),
                (high_to_high < high_to_low).float().mean(),
            )
        ),
        "prediction_mse": {
            "correct_future_mean": float(correct_losses.mean()),
            "incorrect_future_mean": float(incorrect_losses.mean()),
            "incorrect_minus_correct_margin": float(
                (incorrect_losses - correct_losses).mean()
            ),
        },
        "representation_geometry": {
            "raw_encoder": _distance_summary(raw_low, raw_high),
            "prediction_space": _distance_summary(target_low, target_high),
            "future_effective_rank": _effective_rank(
                torch.cat([target_low, target_high])
            ),
            "future_per_dimension_variance": float(
                torch.cat([target_low, target_high]).var(
                    dim=0,
                    unbiased=False,
                ).mean()
            ),
        },
        "physical_future_block_gap_px": {
            "minimum": float(state_gap.min()),
            "mean": float(state_gap.mean()),
            "maximum": float(state_gap.max()),
        },
        "deterministic_current_query_only_accuracy_bound": 0.5,
    }
    model.train(was_training)
    return result


class PairedBatchStream:
    """Yield complete condition-matched pairs without mode metadata."""

    def __init__(
        self,
        pair_count: int,
        *,
        batch_size: int,
        seed: int,
    ) -> None:
        if batch_size <= 0 or batch_size % 2:
            raise ValueError("batch_size must be a positive even integer")
        self.pair_count = pair_count
        self.pairs_per_batch = batch_size // 2
        if pair_count % self.pairs_per_batch:
            raise ValueError(
                "pair_count must divide evenly by pairs_per_batch"
            )
        self.generator = torch.Generator().manual_seed(seed)

    def __iter__(self) -> Iterator[torch.Tensor]:
        while True:
            order = torch.randperm(
                self.pair_count,
                generator=self.generator,
            )
            for start in range(0, self.pair_count, self.pairs_per_batch):
                selected = order[start : start + self.pairs_per_batch]
                indices = torch.stack(
                    [2 * selected, 2 * selected + 1],
                    dim=1,
                ).flatten()
                yield indices


def freeze_image_representation(model: torch.nn.Module) -> list[str]:
    names = []
    for prefix, module in (
        ("encoder", model.encoder),
        ("projector", model.projector),
    ):
        module.requires_grad_(False)
        names.append(prefix)
    return names


def restore_frozen_eval_mode(model: torch.nn.Module) -> None:
    model.encoder.eval()
    model.projector.eval()


def train_variant(
    *,
    variant: str,
    checkpoint: Path,
    train: MaterializedSplit,
    evaluation: dict[str, torch.Tensor],
    output: Path,
    device: torch.device,
    seed: int,
    max_steps: int,
    batch_size: int,
    eval_batch_size: int,
    learning_rate: float,
    weight_decay: float,
    gradient_clip_norm: float,
) -> dict[str, Any]:
    from stable_pretraining.optim.lr_scheduler import (
        LinearWarmupCosineAnnealingLR,
    )
    from stable_worldmodel.wm.loss import (
        ConditionalSIGReg,
        PLDMLoss,
        SIGReg,
    )

    if variant not in VARIANTS:
        raise ValueError(f"Unsupported variant {variant!r}")
    set_reproducible_seed(seed)
    model, checkpoint_receipt = load_model(checkpoint, device=device)
    frozen_modules: list[str] = []
    if variant == "frozen_image_native":
        frozen_modules = freeze_image_representation(model)

    sigreg = SIGReg(knots=17, num_proj=1024).to(device)
    conditional_sigreg = ConditionalSIGReg(
        knots=17,
        num_proj=1024,
        randomize_pair_orientation=True,
    ).to(device)
    pldm = PLDMLoss().to(device)
    parameters = [value for value in model.parameters() if value.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler_max_steps = max(2, max_steps)
    warmup_steps = min(
        scheduler_max_steps - 1,
        max(1, int(0.01 * max_steps)),
    )
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer,
        warmup_steps=warmup_steps,
        max_steps=scheduler_max_steps,
        warmup_start_lr=0.0,
        eta_min=0.0,
    )
    stream = iter(
        PairedBatchStream(
            train.pair_count,
            batch_size=batch_size,
            seed=seed,
        )
    )
    pair_indices = torch.arange(batch_size, device=device).reshape(-1, 2)
    active = torch.zeros(
        4,
        batch_size // 2,
        dtype=torch.bool,
        device=device,
    )
    active[1] = True
    active[3] = True
    snapshot_steps = {
        step for step in SNAPSHOT_STEPS if step <= max_steps
    } | {max_steps}
    snapshots = [
        {
            "optimizer_step": 0,
            "evaluation": evaluate_model(
                model,
                evaluation,
                device=device,
                batch_size=eval_batch_size,
            ),
        }
    ]
    trace: list[dict[str, Any]] = []
    model.train()
    if frozen_modules:
        restore_frozen_eval_mode(model)
    started = time.monotonic()

    for step in range(1, max_steps + 1):
        indices = next(stream)
        batch_pixels = preprocess_pixels(train.pixels[indices], device)
        batch_actions = train.action[indices].to(
            device=device,
            non_blocking=True,
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            output_batch = model.encode(
                {
                    "pixels": batch_pixels,
                    "action": batch_actions,
                }
            )
            embeddings = output_batch["emb"]
            prediction = model.predict(
                embeddings[:, :3],
                output_batch["act_emb"][:, :3],
            )
            # Native LeWM deliberately propagates prediction MSE through
            # both the prediction and online-target branches.
            pred_loss = (prediction - embeddings[:, 1:]).square().mean()
            components: dict[str, torch.Tensor] = {
                "pred_loss": pred_loss,
            }
            if variant in {"native_sigreg_0p09", "frozen_image_native"}:
                components["sigreg_loss"] = sigreg(
                    embeddings.transpose(0, 1)
                )
                loss = pred_loss + 0.09 * components["sigreg_loss"]
            elif variant in CONDITIONAL_SIGREG_WEIGHTS:
                components["conditional_sigreg_loss"] = conditional_sigreg(
                    embeddings.transpose(0, 1),
                    pairs=pair_indices,
                    active=active,
                )
                loss = (
                    pred_loss
                    + CONDITIONAL_SIGREG_WEIGHTS[variant]
                    * components["conditional_sigreg_loss"]
                )
            else:
                pldm_components = pldm(embeddings)
                components.update(pldm_components)
                loss = (
                    pred_loss
                    + 18.0 * components["std_loss"]
                    + 0.7 * components["std_t_loss"]
                    + 12.0 * components["cov_loss"]
                    + 0.2 * components["temp_align_loss"]
                )

        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            parameters,
            gradient_clip_norm,
        )
        learning_rate_used = float(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
        if frozen_modules:
            restore_frozen_eval_mode(model)

        if step == 1 or step in snapshot_steps:
            trace.append(
                {
                    "optimizer_step": step,
                    "learning_rate_used": learning_rate_used,
                    "gradient_norm_before_clip": float(gradient_norm),
                    "loss": float(loss.detach()),
                    "components": {
                        name: float(value.detach())
                        for name, value in components.items()
                    },
                }
            )
        if step in snapshot_steps:
            metrics = evaluate_model(
                model,
                evaluation,
                device=device,
                batch_size=eval_batch_size,
            )
            snapshots.append(
                {
                    "optimizer_step": step,
                    "evaluation": metrics,
                }
            )
            print(
                f"[{variant}] step={step} "
                "target="
                f"{metrics['two_real_future_target_selection_rate']:.3f} "
                "history="
                f"{metrics['correct_history_preference_rate']:.3f} "
                "switch="
                f"{metrics['correct_rule_switch_rate']:.3f} "
                "projected_ratio="
                f"{metrics['representation_geometry']['prediction_space']['paired_to_unrelated_ratio']:.6f}",
                flush=True,
            )
            model.train()
            if frozen_modules:
                restore_frozen_eval_mode(model)

    state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    final_state_hash = state_sha256(state)
    checkpoint_output = output / f"{variant}_step{max_steps}.pt"
    with tempfile.TemporaryDirectory(
        prefix=f"pusht-{variant}-",
        dir="/tmp",
    ) as temporary:
        temporary_path = Path(temporary) / checkpoint_output.name
        torch.save(
            {
                "state_dict": state,
                "variant": variant,
                "seed": seed,
                "optimizer_steps": max_steps,
                "source_checkpoint": checkpoint_receipt,
            },
            temporary_path,
        )
        shutil.copy2(temporary_path, checkpoint_output)
    return {
        "variant": variant,
        "optimizer_steps": max_steps,
        "seed": seed,
        "batch_size": batch_size,
        "batch_mode": "complete_visible_condition_pairs",
        "hidden_labels_at_model_or_loss_boundary": False,
        "conditional_active_times": (
            [1, 3] if variant in CONDITIONAL_SIGREG_WEIGHTS else []
        ),
        "frozen_modules": frozen_modules,
        "source_checkpoint": checkpoint_receipt,
        "optimizer": {
            "type": "AdamW",
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "gradient_clip_norm": gradient_clip_norm,
            "scheduler": "LinearWarmupCosineAnnealingLR",
            "warmup_steps": warmup_steps,
            "max_steps": scheduler_max_steps,
        },
        "precision": (
            "bf16_mixed_autocast" if device.type == "cuda" else "float32"
        ),
        "loss_trace": trace,
        "snapshots": snapshots,
        "final_checkpoint": {
            "path": str(checkpoint_output),
            "sha256": file_sha256(checkpoint_output),
            "model_state_sha256": final_state_hash,
        },
        "elapsed_seconds": time.monotonic() - started,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--original-dataset",
        type=Path,
        default=DEFAULT_ORIGINAL_DATASET,
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--variants",
        default=",".join(VARIANTS),
        help=f"Comma-separated subset of: {', '.join(VARIANTS)}",
    )
    parser.add_argument("--max-steps", type=int, default=256)
    parser.add_argument("--seed", type=int, default=3073)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    variants = tuple(
        value.strip() for value in args.variants.split(",") if value.strip()
    )
    if not variants or len(variants) != len(set(variants)):
        raise ValueError("--variants must be a non-empty unique list")
    unknown = sorted(set(variants) - set(VARIANTS))
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}")
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    if args.batch_size <= 0 or args.batch_size % 2:
        raise ValueError("--batch-size must be a positive even integer")

    data_root = args.data_root.expanduser().resolve()
    original_dataset = args.original_dataset.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    output = Path(os.path.abspath(args.output.expanduser()))
    required = [
        data_root / "manifest.json",
        data_root / "train.lance",
        data_root / "eval_payloads",
        original_dataset,
        checkpoint,
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing input(s):\n" + "\n".join(map(str, missing))
        )
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite output: {output}")
    output.mkdir(parents=True)
    manifest = json.loads(
        (data_root / "manifest.json").read_text(encoding="utf-8")
    )
    if manifest.get("passed") is not True:
        raise RuntimeError("The paired synthesis manifest did not pass")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    action_stats = original_action_stats(original_dataset)
    print("Materializing audited paired Push-T training split", flush=True)
    train = materialize_lance_split(
        data_root / "train.lance",
        action_stats=action_stats,
    )
    evaluation = load_eval_payloads(
        data_root / "eval_payloads",
        action_stats=action_stats,
    )
    if train.pair_count % (args.batch_size // 2):
        raise ValueError(
            "The chosen batch size does not divide the train pair count"
        )

    provenance = {
        "schema_version": 1,
        "status": "bounded_pilot_not_formal_multiseed_confirmation",
        "question": (
            "Does the native joint LeWM objective retain a persistent hidden "
            "Push-T gain that is identifiable only from history?"
        ),
        "data": {
            "root": str(data_root),
            "manifest": str(data_root / "manifest.json"),
            "manifest_sha256": file_sha256(data_root / "manifest.json"),
            "protocol": manifest["protocol"],
            "train_pairs": train.pair_count,
            "eval_pairs": int(evaluation["low_pixels"].size(0)),
        },
        "normalization": {
            "pixels": {
                "scale": "uint8_to_unit_interval",
                "mean": list(IMAGENET_MEAN),
                "std": list(IMAGENET_STD),
            },
            "action": {
                **{
                    key: value
                    for key, value in action_stats.items()
                    if key not in {"mean", "std"}
                },
                "mean": action_stats["mean"].tolist(),
                "std": action_stats["std"].tolist(),
            },
        },
        "model_input": ["pixels", "action"],
        "forbidden_fields": [
            "hidden_mode",
            "hidden_action_scale",
            "pair_id",
            "physics_state",
        ],
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_device_name": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None
        ),
    }
    results = []
    for index, variant in enumerate(variants, start=1):
        print(
            f"[{index}/{len(variants)}] training {variant}",
            flush=True,
        )
        results.append(
            train_variant(
                variant=variant,
                checkpoint=checkpoint,
                train=train,
                evaluation=evaluation,
                output=output,
                device=device,
                seed=args.seed,
                max_steps=args.max_steps,
                batch_size=args.batch_size,
                eval_batch_size=args.eval_batch_size,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                gradient_clip_norm=args.gradient_clip_norm,
            )
        )
        report = {
            "provenance": provenance,
            "training_contract": {
                "seed": args.seed,
                "max_steps": args.max_steps,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "gradient_clip_norm": args.gradient_clip_norm,
                "same_initial_checkpoint": True,
                "same_batch_order_seed": True,
            },
            "results": results,
        }
        (output / "pilot_report.partial.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    report = {
        "provenance": provenance,
        "training_contract": {
            "seed": args.seed,
            "max_steps": args.max_steps,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "gradient_clip_norm": args.gradient_clip_norm,
            "same_initial_checkpoint": True,
            "same_batch_order_seed": True,
        },
        "results": results,
    }
    report_path = output / "pilot_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "report": str(report_path),
                "report_sha256": file_sha256(report_path),
                "variants": {
                    row["variant"]: row["snapshots"][-1]["evaluation"]
                    for row in results
                },
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
