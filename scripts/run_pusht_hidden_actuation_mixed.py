#!/usr/bin/env python3
"""Fine-tune LeWM on an audited original/hidden-actuation Push-T mixture.

The hidden portion of every batch contains complete condition-matched pairs.
The original portion is sampled from the standard Push-T training dataset.
The default partition is 50/50; ``--original-batch-size`` can change the
partition, including a 100% standard-data continuation control.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
import random
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterator

import hydra
import torch
from omegaconf import OmegaConf, open_dict


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
from stable_worldmodel.wm.loss import (  # noqa: E402
    ConditionalSIGReg,
    DynamicsResponseSIGReg,
    GroupBalancedSIGReg,
    PLDMLoss,
    ScaleCalibratedConditionalSIGReg,
    SIGReg,
)


DEFAULT_ORIGINAL_LANCE = Path(
    "/opt/huawei/explorer-env/dataset/ag_data/data/world_model/"
    "lance-format/LeWorldModel/data/lewm_pusht.lance"
)
DEFAULT_OUTPUT = artifact_path(
    "evaluation/history3/pusht_hidden_actuation_h3_v1/"
    "mixed_retention_seed3073_step2048"
)
DEFAULT_CONTRAST_SCALES = artifact_path(
    "evaluation/history3/pusht_hidden_actuation_h3_v1/"
    "scale_calibrated_sigreg_v1/source_scales.json"
)
VARIANT_WEIGHTS = {
    "mixed_native_sigreg_0p09": ("native", 0.09, "marginal"),
    "mixed_pldm_joint": (
        "pldm",
        1.0,
        "official_vcreg_and_temporal_alignment",
    ),
    "mixed_frozen_image_native_0p09": ("native", 0.09, "marginal"),
    "mixed_native_sigreg_0p20": ("native", 0.20, "marginal"),
    "mixed_native_sigreg_0p30": ("native", 0.30, "marginal"),
    "mixed_native_sigreg_0p50": ("native", 0.50, "marginal"),
    "mixed_native_sigreg_0p90": ("native", 0.90, "marginal"),
    "mixed_native_sigreg_2p05": ("native", 2.05, "marginal"),
    "mixed_group_balanced_sigreg_0p02": (
        "group_balanced",
        0.02,
        "separate_marginal_and_highpass",
    ),
    "mixed_group_balanced_sigreg_0p05": (
        "group_balanced",
        0.05,
        "separate_marginal_and_highpass",
    ),
    "mixed_transition_conditional_sigreg_0p005": (
        "transition_conditional",
        0.005,
        "causally_masked_transition_population",
    ),
    "mixed_transition_conditional_sigreg_0p01": (
        "transition_conditional",
        0.01,
        "causally_masked_transition_population",
    ),
    "mixed_transition_conditional_sigreg_0p02": (
        "transition_conditional",
        0.02,
        "causally_masked_transition_population",
    ),
    "mixed_transition_conditional_sigreg_0p05": (
        "transition_conditional",
        0.05,
        "causally_masked_transition_population",
    ),
    "mixed_transition_conditional_sigreg_0p09": (
        "transition_conditional",
        0.09,
        "causally_masked_transition_population",
    ),
    "mixed_scale_calibrated_sigreg_0p005": (
        "scale_calibrated",
        0.005,
        "source_scaled_highpass_with_unpaired",
    ),
    "mixed_scale_calibrated_sigreg_0p01": (
        "scale_calibrated",
        0.01,
        "source_scaled_highpass_with_unpaired",
    ),
    "mixed_scale_calibrated_sigreg_0p02": (
        "scale_calibrated",
        0.02,
        "source_scaled_highpass_with_unpaired",
    ),
    "mixed_scale_calibrated_sigreg_0p05": (
        "scale_calibrated",
        0.05,
        "source_scaled_highpass_with_unpaired",
    ),
    "mixed_scale_calibrated_sigreg_0p09": (
        "scale_calibrated",
        0.09,
        "source_scaled_highpass_with_unpaired",
    ),
    "mixed_dynamics_response_sigreg_0p005": (
        "dynamics_response",
        0.005,
        "deterministic_target_and_prediction_responses",
    ),
    "mixed_dynamics_response_sigreg_0p01": (
        "dynamics_response",
        0.01,
        "deterministic_target_and_prediction_responses",
    ),
    "mixed_dynamics_response_sigreg_0p02": (
        "dynamics_response",
        0.02,
        "deterministic_target_and_prediction_responses",
    ),
    "mixed_dynamics_response_sigreg_0p05": (
        "dynamics_response",
        0.05,
        "deterministic_target_and_prediction_responses",
    ),
    "mixed_dynamics_response_sigreg_0p09": (
        "dynamics_response",
        0.09,
        "deterministic_target_and_prediction_responses",
    ),
    "mixed_conditional_sigreg_0p01": (
        "conditional",
        0.01,
        "highpass",
    ),
    "mixed_conditional_sigreg_0p05": (
        "conditional",
        0.05,
        "highpass",
    ),
    "mixed_conditional_sigreg_0p01_include_unpaired": (
        "conditional",
        0.01,
        "highpass_with_unpaired",
    ),
    "mixed_conditional_sigreg_0p01_complete_haar": (
        "conditional",
        0.01,
        "complete_haar",
    ),
    "mixed_conditional_sigreg_0p05_complete_haar": (
        "conditional",
        0.05,
        "complete_haar",
    ),
}
FROZEN_IMAGE_VARIANTS = {"mixed_frozen_image_native_0p09"}
FROZEN_IMAGE_MODULES = ("encoder", "projector")
TRAINABLE_DYNAMICS_MODULES = ("predictor", "action_encoder", "pred_proj")
SNAPSHOT_STEPS = (0, 1, 8, 32, 128, 512, 1024, 2048)
ACTION_INPUT_DIM = 10


def mixed_prediction_loss(
    *,
    prediction: torch.Tensor,
    embeddings: torch.Tensor,
    original_batch_size: int,
    conditional_population: str,
) -> torch.Tensor:
    """Apply the registered transition supervision without hidden labels."""

    if prediction.ndim != 3 or embeddings.ndim != 3:
        raise ValueError("prediction and embeddings must be rank-3 tensors")
    if prediction.shape[:2] != (
        embeddings.shape[0],
        embeddings.shape[1] - 1,
    ):
        raise ValueError("prediction/embedding transition shapes do not match")
    if not 0 < original_batch_size <= prediction.shape[0]:
        raise ValueError("original_batch_size is outside the batch")
    identifiable_only = conditional_population in {
        "identifiable_future_only",
        "paired_future_ranking",
        "paired_future_matching",
        "paired_future_fit",
        "paired_future_projected_center",
        "paired_future_response_log_norm",
        "paired_future_projected_geometry",
    }
    if not identifiable_only:
        return torch.square(prediction - embeddings[:, 1:]).mean()
    original_prediction_loss = torch.square(
        prediction[:original_batch_size]
        - embeddings[:original_batch_size, 1:]
    ).mean()
    hidden_future_loss = torch.square(
        prediction[original_batch_size:, -1]
        - embeddings[original_batch_size:, -1]
    ).mean()
    return 0.5 * (original_prediction_loss + hidden_future_loss)


def model_config(config_name: str = "lewm") -> dict[str, Any]:
    if config_name not in {"lewm", "pldm"}:
        raise ValueError(f"Unsupported model config {config_name!r}")
    cfg = OmegaConf.load(
        STABLE_WORLD_MODEL_ROOT
        / f"scripts/train/config/{config_name}.yaml"
    )
    with open_dict(cfg):
        cfg.model.action_encoder.input_dim = ACTION_INPUT_DIM
    return OmegaConf.to_container(cfg.model, resolve=True)


def model_config_name_for_variant(variant: str) -> str:
    regularizer_kind = VARIANT_WEIGHTS[variant][0]
    return (
        "pldm"
        if regularizer_kind in {"pldm", "pldm_paired_future_ranking"}
        else "lewm"
    )


def resolve_batch_partition(
    batch_size: int,
    original_batch_size: int | None,
) -> tuple[int, int]:
    """Return an audited standard/paired-hidden batch partition."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if original_batch_size is None:
        original_batch_size = batch_size // 2
    if not 0 < original_batch_size <= batch_size:
        raise ValueError(
            "original_batch_size must be in the interval [1, batch_size]"
        )
    hidden_batch_size = batch_size - original_batch_size
    if hidden_batch_size % 2:
        raise ValueError(
            "hidden_batch_size must be even so complete pairs are retained"
        )
    return original_batch_size, hidden_batch_size


def load_model_for_variant(
    checkpoint: Path,
    *,
    variant: str,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Load the common LeWM state into the registered model architecture."""

    config_name = model_config_name_for_variant(variant)
    cfg = OmegaConf.load(
        STABLE_WORLD_MODEL_ROOT
        / f"scripts/train/config/{config_name}.yaml"
    )
    with open_dict(cfg):
        cfg.model.action_encoder.input_dim = ACTION_INPUT_DIM
    model = hydra.utils.instantiate(cfg.model)
    source_state = pilot.checkpoint_model_state(checkpoint)
    model.load_state_dict(source_state, strict=True)
    model_state_sha256 = pilot.state_sha256(model.state_dict())
    return model.to(device), {
        "path": str(checkpoint),
        "sha256": pilot.file_sha256(checkpoint),
        "model_state_sha256": model_state_sha256,
        "loaded_model_config": config_name,
        "strict_state_dict_load": True,
    }


def original_loader(
    path: Path,
    *,
    batch_size: int,
    seed: int,
    num_workers: int,
) -> tuple[LanceDataset, torch.utils.data.DataLoader]:
    dataset = LanceDataset(
        path=path,
        frameskip=5,
        num_steps=4,
        keys_to_load=["pixels", "action"],
    )
    generator = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=True,
        generator=generator,
    )
    return dataset, loader


def next_original_batch(
    iterator: Iterator[dict[str, torch.Tensor]],
    loader: torch.utils.data.DataLoader,
) -> tuple[dict[str, torch.Tensor], Iterator[dict[str, torch.Tensor]]]:
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def write_checkpoint(
    *,
    state: dict[str, torch.Tensor],
    output: Path,
    variant: str,
    max_steps: int,
) -> dict[str, Any]:
    path = output / f"{variant}_step{max_steps}.pt"
    with tempfile.TemporaryDirectory(
        prefix=f"pusht-mixed-{variant}-",
        dir="/tmp",
    ) as temporary:
        temporary_path = Path(temporary) / path.name
        torch.save(state, temporary_path)
        shutil.copy2(temporary_path, path)
    return {
        "path": str(path),
        "sha256": pilot.file_sha256(path),
        "model_state_sha256": pilot.state_sha256(state),
        "format": "raw_model_state_dict_with_sibling_config_json",
    }


def state_subset_sha256(
    model: torch.nn.Module,
    prefixes: tuple[str, ...],
) -> str:
    """Hash the state owned by exactly the requested top-level modules."""

    selected = {
        name: value
        for name, value in model.state_dict().items()
        if name.split(".", 1)[0] in prefixes
    }
    if not selected:
        raise RuntimeError(f"No model state found for prefixes={prefixes}")
    return pilot.state_sha256(selected)


def gradient_audit(
    model: torch.nn.Module,
    *,
    prefixes: tuple[str, ...],
) -> dict[str, Any]:
    """Summarize gradients for a declared set of top-level modules."""

    selected = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if name.split(".", 1)[0] in prefixes
    ]
    if not selected:
        raise RuntimeError(f"No parameters found for prefixes={prefixes}")
    with_gradient = [
        (name, parameter.grad)
        for name, parameter in selected
        if parameter.grad is not None
    ]
    nonzero = [
        (name, gradient)
        for name, gradient in with_gradient
        if bool(torch.count_nonzero(gradient).item())
    ]
    finite = all(
        bool(torch.isfinite(gradient).all().item())
        for _, gradient in with_gradient
    )
    squared_norm = sum(
        float(gradient.detach().float().square().sum())
        for _, gradient in with_gradient
    )
    return {
        "parameter_tensor_count": len(selected),
        "requires_grad_tensor_count": sum(
            int(parameter.requires_grad) for _, parameter in selected
        ),
        "gradient_tensor_count": len(with_gradient),
        "nonzero_gradient_tensor_count": len(nonzero),
        "all_present_gradients_finite": finite,
        "gradient_l2_norm": squared_norm**0.5,
        "parameter_names": [name for name, _ in selected],
    }


def build_transition_conditional_population(
    embeddings: torch.Tensor,
    prediction: torch.Tensor,
    pairs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the causally masked target/prediction SIGReg population.

    ``embeddings`` contains four observations ``x0..x3`` and ``prediction``
    contains the three transition predictions ``x1_hat..x3_hat``.  The
    returned population concatenates real targets and predictions along the
    batch dimension.  Real target differences are active at ``x1`` and
    ``x3``; prediction differences are active only at History-identifiable
    ``x3``.
    """

    if embeddings.dim() != 3 or prediction.dim() != 3:
        raise ValueError("embeddings and prediction must have shape (B,T,D)")
    if embeddings.size(0) != prediction.size(0):
        raise ValueError("embeddings and prediction batch sizes must match")
    if embeddings.size(1) != prediction.size(1) + 1:
        raise ValueError(
            "embeddings must contain exactly one more time step than "
            "prediction"
        )
    if prediction.size(1) != 3:
        raise ValueError("History-3 transition population expects 3 outputs")
    if embeddings.size(2) != prediction.size(2):
        raise ValueError("embedding and prediction dimensions must match")
    if pairs.dim() != 2 or pairs.size(1) != 2:
        raise ValueError("pairs must have shape (P,2)")
    if pairs.dtype != torch.long:
        raise TypeError("pairs must use torch.long")
    if pairs.numel() and (
        int(pairs.min()) < 0 or int(pairs.max()) >= embeddings.size(0)
    ):
        raise ValueError("pairs contain an index outside the batch")
    if torch.unique(pairs.flatten()).numel() != pairs.numel():
        raise ValueError("pairs must be disjoint")

    targets = embeddings[:, 1:]
    population = torch.cat([targets, prediction], dim=0)
    prediction_pairs = pairs + embeddings.size(0)
    transition_pairs = torch.cat([pairs, prediction_pairs], dim=0)
    pair_count = pairs.size(0)
    active = torch.zeros(
        prediction.size(1),
        2 * pair_count,
        dtype=torch.bool,
        device=prediction.device,
    )
    # The real x1 probe outcomes differ, but their predictions cannot differ
    # because mode is not identifiable from the common x0/action.
    active[0, :pair_count] = True
    # At x3 the x1 observation in History-3 identifies the persistent gain,
    # so both the real and predicted transition differences are protected.
    active[2] = True
    return population, transition_pairs, active


def paired_future_ranking_loss(
    *,
    embeddings: torch.Tensor,
    deterministic_prediction: torch.Tensor,
    pair_indices: torch.Tensor,
) -> torch.Tensor:
    """Rank each identifiable prediction against its two real futures."""

    left = pair_indices[:, 0]
    right = pair_indices[:, 1]
    predicted_left = deterministic_prediction[left, 2]
    predicted_right = deterministic_prediction[right, 2]
    target_left = embeddings[left, 3]
    target_right = embeddings[right, 3]

    def per_row_mse(
        first: torch.Tensor,
        second: torch.Tensor,
    ) -> torch.Tensor:
        return torch.square(first - second).mean(dim=-1)

    correct = torch.cat(
        [
            per_row_mse(predicted_left, target_left),
            per_row_mse(predicted_right, target_right),
        ]
    )
    other = torch.cat(
        [
            per_row_mse(predicted_left, target_right),
            per_row_mse(predicted_right, target_left),
        ]
    )
    pair_scale = per_row_mse(
        target_left,
        target_right,
    ).detach().clamp_min(1e-8)
    scale = torch.cat([pair_scale, pair_scale])
    return torch.nn.functional.softplus(
        (correct - other) / scale + 0.5
    ).mean()


def paired_future_matching_loss(
    *,
    embeddings: torch.Tensor,
    deterministic_prediction: torch.Tensor,
    pair_indices: torch.Tensor,
    include_fit_terms: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Match predictions and real futures in both comparison directions."""

    left = pair_indices[:, 0]
    right = pair_indices[:, 1]
    predicted_left = deterministic_prediction[left, 2]
    predicted_right = deterministic_prediction[right, 2]
    target_left = embeddings[left, 3]
    target_right = embeddings[right, 3]

    def per_row_mse(
        first: torch.Tensor,
        second: torch.Tensor,
    ) -> torch.Tensor:
        return torch.square(first - second).mean(dim=-1)

    left_left = per_row_mse(predicted_left, target_left)
    left_right = per_row_mse(predicted_left, target_right)
    right_left = per_row_mse(predicted_right, target_left)
    right_right = per_row_mse(predicted_right, target_right)
    pair_scale = per_row_mse(
        target_left,
        target_right,
    ).detach().clamp_min(1e-8)
    future_ranking = 0.5 * (
        torch.nn.functional.softplus(
            (left_left - left_right) / pair_scale + 0.5
        ).mean()
        + torch.nn.functional.softplus(
            (right_right - right_left) / pair_scale + 0.5
        ).mean()
    )
    history_ranking = 0.5 * (
        torch.nn.functional.softplus(
            (left_left - right_left) / pair_scale + 0.5
        ).mean()
        + torch.nn.functional.softplus(
            (right_right - left_right) / pair_scale + 0.5
        ).mean()
    )
    target_delta = target_right - target_left
    prediction_delta = predicted_right - predicted_left
    switch_alignment = (
        1.0
        - torch.nn.functional.cosine_similarity(
            prediction_delta.float(),
            target_delta.float(),
            dim=-1,
            eps=1e-8,
        )
    ).mean()
    components = {
        "future_ranking_loss": future_ranking,
        "history_ranking_loss": history_ranking,
        "switch_alignment_loss": switch_alignment,
    }
    if include_fit_terms:
        response_fit = torch.log1p(
            per_row_mse(
                prediction_delta,
                target_delta,
            )
            / pair_scale
        ).mean()
        correct_future_fit = torch.log1p(
            0.5 * (left_left + right_right) / pair_scale
        ).mean()
        components.update(
            {
                "response_fit_loss": response_fit,
                "correct_future_fit_loss": correct_future_fit,
            }
        )
    return sum(components.values()), components


def paired_future_projected_geometry_loss(
    *,
    embeddings: torch.Tensor,
    deterministic_prediction: torch.Tensor,
    pair_indices: torch.Tensor,
    include_projected_center: bool,
    include_response_log_norm: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Match paired futures while calibrating only causal scalar geometry.

    The existing paired matching terms preserve target selection in both
    comparison directions and align the predicted response direction.  The
    optional additions deliberately avoid fitting all latent dimensions:

    * ``projected_center`` anchors the prediction-pair midpoint only along
      the real future-pair direction;
    * ``response_log_norm`` matches the predicted/real response RMS on a
      logarithmic scale.

    Both terms use only predictions and the two real future latents.  No
    hidden condition label is consumed.
    """

    if not include_projected_center and not include_response_log_norm:
        raise ValueError(
            "At least one projected-geometry term must be enabled"
        )
    matching_loss, components = paired_future_matching_loss(
        embeddings=embeddings,
        deterministic_prediction=deterministic_prediction,
        pair_indices=pair_indices,
    )
    left = pair_indices[:, 0]
    right = pair_indices[:, 1]
    predicted_left = deterministic_prediction[left, 2]
    predicted_right = deterministic_prediction[right, 2]
    target_left = embeddings[left, 3]
    target_right = embeddings[right, 3]

    prediction_center = 0.5 * (predicted_left + predicted_right)
    target_center = 0.5 * (target_left + target_right)
    prediction_response = predicted_right - predicted_left
    target_response = target_right - target_left
    target_squared_norm = target_response.float().square().sum(
        dim=-1
    ).clamp_min(1e-12)

    selected_losses = [matching_loss]
    if include_projected_center:
        # Twice the projected offset is measured in units of the real
        # target-pair separation.  Zero is the calibrated midpoint.
        projected_center = 2.0 * (
            (prediction_center.float() - target_center.float())
            * target_response.float()
        ).sum(dim=-1) / target_squared_norm
        center_loss = torch.nn.functional.smooth_l1_loss(
            projected_center,
            torch.zeros_like(projected_center),
        )
        components["projected_center_loss"] = center_loss
        components["projected_center_abs_mean"] = (
            projected_center.detach().abs().mean()
        )
        selected_losses.append(center_loss)

    prediction_rms = prediction_response.float().square().mean(
        dim=-1
    ).clamp_min(1e-12).sqrt()
    target_rms = target_response.float().square().mean(
        dim=-1
    ).clamp_min(1e-12).sqrt()
    response_rms_ratio = prediction_rms / target_rms
    components["response_rms_ratio_mean"] = (
        response_rms_ratio.detach().mean()
    )
    if include_response_log_norm:
        response_log_norm = response_rms_ratio.log()
        response_log_norm_loss = torch.nn.functional.smooth_l1_loss(
            response_log_norm,
            torch.zeros_like(response_log_norm),
        )
        components["response_log_norm_loss"] = response_log_norm_loss
        selected_losses.append(response_log_norm_loss)

    return sum(selected_losses), components


def load_contrast_scales(
    path: Path,
) -> tuple[torch.Tensor, dict[str, Any]]:
    payload = json.loads(path.read_text())
    active = payload.get("active_contrast_scales", {})
    if set(active) != {"1", "3"}:
        raise ValueError(
            "Expected frozen source scales at observation indices 1 and 3"
        )
    values = torch.tensor(
        [1.0, float(active["1"]), 1.0, float(active["3"])],
        dtype=torch.float32,
    )
    if not bool(torch.isfinite(values).all()) or bool((values <= 0).any()):
        raise ValueError(f"Invalid frozen contrast scales: {values.tolist()}")
    return values, {
        "path": str(path),
        "sha256": pilot.file_sha256(path),
        "values_by_observation_index": {
            str(index): float(value)
            for index, value in enumerate(values)
        },
        "frozen_during_training": True,
    }


@contextmanager
def temporary_eval_modules(*modules: torch.nn.Module):
    """Disable stochastic inference modules without disabling gradients."""

    states = {
        submodule: submodule.training
        for module in modules
        for submodule in module.modules()
    }
    try:
        for module in modules:
            module.eval()
        yield
    finally:
        for submodule, training in states.items():
            submodule.training = training


def train_variant(
    *,
    variant: str,
    checkpoint: Path,
    original_path: Path,
    hidden: pilot.MaterializedSplit,
    evaluation: dict[str, torch.Tensor],
    action_stats: dict[str, Any],
    output: Path,
    device: torch.device,
    seed: int,
    max_steps: int,
    batch_size: int,
    original_batch_size: int,
    eval_batch_size: int,
    learning_rate: float,
    weight_decay: float,
    gradient_clip_norm: float,
    num_workers: int,
    contrast_scales: torch.Tensor | None,
    contrast_scale_receipt: dict[str, Any] | None,
) -> dict[str, Any]:
    from stable_pretraining.optim.lr_scheduler import (
        LinearWarmupCosineAnnealingLR,
    )

    (
        regularizer_kind,
        regularizer_weight,
        conditional_population,
    ) = VARIANT_WEIGHTS[variant]
    original_batch_size, hidden_batch_size = resolve_batch_partition(
        batch_size,
        original_batch_size,
    )
    if hidden_batch_size == 0 and regularizer_kind not in {"native", "pldm"}:
        raise ValueError(
            f"{regularizer_kind} requires condition-matched hidden pairs"
        )

    pilot.set_reproducible_seed(seed)
    model, checkpoint_receipt = load_model_for_variant(
        checkpoint,
        variant=variant,
        device=device,
    )
    freeze_image_representation = variant in FROZEN_IMAGE_VARIANTS
    frozen_modules: list[str] = []
    if freeze_image_representation:
        frozen_modules = pilot.freeze_image_representation(model)
        if tuple(frozen_modules) != FROZEN_IMAGE_MODULES:
            raise RuntimeError(
                "Unexpected frozen module set: "
                f"expected={FROZEN_IMAGE_MODULES}, actual={frozen_modules}"
            )
    frozen_state_sha256_before = (
        state_subset_sha256(model, FROZEN_IMAGE_MODULES)
        if freeze_image_representation
        else None
    )
    trainable_state_sha256_before = (
        state_subset_sha256(model, TRAINABLE_DYNAMICS_MODULES)
        if freeze_image_representation
        else None
    )
    native_sigreg = SIGReg(knots=17, num_proj=1024).to(device)
    conditional_sigreg = ConditionalSIGReg(
        knots=17,
        num_proj=1024,
        randomize_pair_orientation=True,
        include_unpaired=(
            conditional_population == "highpass_with_unpaired"
        ),
        complete_haar_population=(
            conditional_population == "complete_haar"
        ),
    ).to(device)
    group_balanced_sigreg = GroupBalancedSIGReg(
        knots=17,
        num_proj=1024,
        randomize_pair_orientation=True,
    ).to(device)
    scale_calibrated_sigreg = ScaleCalibratedConditionalSIGReg(
        knots=17,
        num_proj=1024,
        randomize_pair_orientation=True,
    ).to(device)
    dynamics_response_sigreg = DynamicsResponseSIGReg(
        knots=17,
        num_proj=1024,
        reserve_factor=2.0**0.5,
        randomize_pair_orientation=True,
    ).to(device)
    pldm = PLDMLoss().to(device)
    parameters = [
        parameter for parameter in model.parameters()
        if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    optimizer_parameter_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    frozen_parameter_ids = {
        id(parameter)
        for name, parameter in model.named_parameters()
        if name.split(".", 1)[0] in FROZEN_IMAGE_MODULES
    }
    optimizer_excludes_frozen = not bool(
        optimizer_parameter_ids & frozen_parameter_ids
    )
    if freeze_image_representation and not optimizer_excludes_frozen:
        raise RuntimeError("Optimizer unexpectedly contains frozen parameters")
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

    original_dataset, loader = original_loader(
        original_path,
        batch_size=original_batch_size,
        seed=seed,
        num_workers=num_workers,
    )
    original_iterator = iter(loader)
    hidden_iterator = (
        iter(
            pilot.PairedBatchStream(
                hidden.pair_count,
                batch_size=hidden_batch_size,
                seed=seed,
            )
        )
        if hidden_batch_size
        else None
    )
    pair_indices = torch.arange(
        original_batch_size,
        batch_size,
        device=device,
    ).reshape(-1, 2)
    active = torch.zeros(
        4,
        hidden_batch_size // 2,
        dtype=torch.bool,
        device=device,
    )
    active[1] = True
    active[3] = True
    response_target_active = torch.zeros(
        3,
        hidden_batch_size // 2,
        dtype=torch.bool,
        device=device,
    )
    response_target_active[0] = True
    response_target_active[2] = True
    response_prediction_active = torch.zeros_like(
        response_target_active
    )
    response_prediction_active[2] = True
    snapshot_steps = {
        step for step in SNAPSHOT_STEPS if step <= max_steps
    } | {max_steps}
    model.train()
    if freeze_image_representation:
        pilot.restore_frozen_eval_mode(model)
    snapshots = [
        {
            "optimizer_step": 0,
            "hidden_evaluation": pilot.evaluate_model(
                model,
                evaluation,
                device=device,
                batch_size=eval_batch_size,
            ),
        }
    ]
    if freeze_image_representation:
        pilot.restore_frozen_eval_mode(model)
    trace: list[dict[str, Any]] = []
    first_step_gradient_audit: dict[str, Any] | None = None
    started = time.monotonic()

    for step in range(1, max_steps + 1):
        original, original_iterator = next_original_batch(
            original_iterator,
            loader,
        )
        original_actions = pilot.normalize_action_blocks(
            torch.nan_to_num(original["action"].float(), 0.0),
            action_stats,
        )
        if hidden_iterator is None:
            raw_pixels = original["pixels"]
            raw_actions = original_actions
        else:
            hidden_indices = next(hidden_iterator)
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

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            output_batch = model.encode(
                {"pixels": pixels, "action": actions}
            )
            embeddings = output_batch["emb"]
            prediction = model.predict(
                embeddings[:, :3],
                output_batch["act_emb"][:, :3],
            )
            pred_loss = mixed_prediction_loss(
                prediction=prediction,
                embeddings=embeddings,
                original_batch_size=original_batch_size,
                conditional_population=conditional_population,
            )
            regularizer_components: dict[str, torch.Tensor] = {}
            if regularizer_kind == "native":
                reg_loss = native_sigreg(embeddings.transpose(0, 1))
            elif regularizer_kind == "pldm":
                regularizer_components = pldm(embeddings)
                reg_loss = (
                    18.0 * regularizer_components["std_loss"]
                    + 0.7 * regularizer_components["std_t_loss"]
                    + 12.0 * regularizer_components["cov_loss"]
                    + 0.2 * regularizer_components["temp_align_loss"]
                )
            elif regularizer_kind == "pldm_paired_future_ranking":
                regularizer_components = pldm(embeddings)
                with temporary_eval_modules(
                    model.predictor,
                    model.pred_proj,
                ):
                    deterministic_prediction = model.predict(
                        embeddings[:, :3],
                        output_batch["act_emb"][:, :3],
                    )
                ranking_loss = paired_future_ranking_loss(
                    embeddings=embeddings,
                    deterministic_prediction=deterministic_prediction,
                    pair_indices=pair_indices,
                )
                regularizer_components[
                    "paired_future_ranking_loss"
                ] = ranking_loss
                reg_loss = (
                    18.0 * regularizer_components["std_loss"]
                    + 0.7 * regularizer_components["std_t_loss"]
                    + 12.0 * regularizer_components["cov_loss"]
                    + 0.2 * regularizer_components["temp_align_loss"]
                    + ranking_loss
                )
            elif regularizer_kind == "conditional":
                reg_loss = conditional_sigreg(
                    embeddings.transpose(0, 1),
                    pairs=pair_indices,
                    active=active,
                )
            elif regularizer_kind == "group_balanced":
                reg_loss = group_balanced_sigreg(
                    embeddings.transpose(0, 1),
                    pairs=pair_indices,
                    active=active,
                )
            elif regularizer_kind == "scale_calibrated":
                if contrast_scales is None:
                    raise RuntimeError(
                        "Scale-calibrated SIGReg requires frozen scales"
                    )
                reg_loss = scale_calibrated_sigreg(
                    embeddings.transpose(0, 1),
                    pairs=pair_indices,
                    active=active,
                    contrast_scales=contrast_scales,
                )
            elif regularizer_kind == "dynamics_response":
                if contrast_scales is None:
                    raise RuntimeError(
                        "Dynamics-response SIGReg requires frozen scales"
                    )
                with temporary_eval_modules(
                    model.predictor,
                    model.pred_proj,
                ):
                    deterministic_prediction = model.predict(
                        embeddings[:, :3],
                        output_batch["act_emb"][:, :3],
                    )
                if conditional_population == "paired_response_alignment":
                    scale = contrast_scales[3].to(
                        device=device,
                        dtype=embeddings.dtype,
                    ).clamp_min(1e-8)
                    target_response = (
                        embeddings[pair_indices[:, 0], 3]
                        - embeddings[pair_indices[:, 1], 3]
                    ) / scale
                    predicted_response = (
                        deterministic_prediction[pair_indices[:, 0], 2]
                        - deterministic_prediction[pair_indices[:, 1], 2]
                    ) / scale
                    reg_loss = torch.square(
                        predicted_response - target_response
                    ).mean()
                else:
                    reg_loss = dynamics_response_sigreg(
                        embeddings[:, 1:].transpose(0, 1),
                        deterministic_prediction.transpose(0, 1),
                        pairs=pair_indices,
                        target_active=response_target_active,
                        prediction_active=response_prediction_active,
                        contrast_scales=contrast_scales[1:],
                    )
            elif regularizer_kind == "paired_future_ranking":
                # Each prediction must be closer to its own real future than
                # to the other member's real future.  The target-pair scale
                # makes the margin independent of the encoder's units.
                with temporary_eval_modules(
                    model.predictor,
                    model.pred_proj,
                ):
                    deterministic_prediction = model.predict(
                        embeddings[:, :3],
                        output_batch["act_emb"][:, :3],
                    )
                reg_loss = paired_future_ranking_loss(
                    embeddings=embeddings,
                    deterministic_prediction=deterministic_prediction,
                    pair_indices=pair_indices,
                )
            elif regularizer_kind == "paired_future_matching":
                with temporary_eval_modules(
                    model.predictor,
                    model.pred_proj,
                ):
                    deterministic_prediction = model.predict(
                        embeddings[:, :3],
                        output_batch["act_emb"][:, :3],
                    )
                reg_loss, matching_components = (
                    paired_future_matching_loss(
                        embeddings=embeddings,
                        deterministic_prediction=deterministic_prediction,
                        pair_indices=pair_indices,
                    )
                )
                regularizer_components.update(matching_components)
            elif regularizer_kind == "paired_future_fit":
                with temporary_eval_modules(
                    model.predictor,
                    model.pred_proj,
                ):
                    deterministic_prediction = model.predict(
                        embeddings[:, :3],
                        output_batch["act_emb"][:, :3],
                    )
                reg_loss, matching_components = (
                    paired_future_matching_loss(
                        embeddings=embeddings,
                        deterministic_prediction=deterministic_prediction,
                        pair_indices=pair_indices,
                        include_fit_terms=True,
                    )
                )
                regularizer_components.update(matching_components)
            elif regularizer_kind in {
                "paired_future_projected_center",
                "paired_future_response_log_norm",
                "paired_future_projected_geometry",
            }:
                with temporary_eval_modules(
                    model.predictor,
                    model.pred_proj,
                ):
                    deterministic_prediction = model.predict(
                        embeddings[:, :3],
                        output_batch["act_emb"][:, :3],
                    )
                reg_loss, geometry_components = (
                    paired_future_projected_geometry_loss(
                        embeddings=embeddings,
                        deterministic_prediction=(
                            deterministic_prediction
                        ),
                        pair_indices=pair_indices,
                        include_projected_center=(
                            regularizer_kind
                            in {
                                "paired_future_projected_center",
                                "paired_future_projected_geometry",
                            }
                        ),
                        include_response_log_norm=(
                            regularizer_kind
                            in {
                                "paired_future_response_log_norm",
                                "paired_future_projected_geometry",
                            }
                        ),
                    )
                )
                regularizer_components.update(geometry_components)
            else:
                (
                    transition_population,
                    transition_pairs,
                    transition_active,
                ) = build_transition_conditional_population(
                    embeddings,
                    prediction,
                    pair_indices,
                )
                reg_loss = group_balanced_sigreg(
                    transition_population.transpose(0, 1),
                    pairs=transition_pairs,
                    active=transition_active,
                )
            loss = pred_loss + regularizer_weight * reg_loss

        loss.backward()
        if step == 1 and freeze_image_representation:
            frozen_gradient = gradient_audit(
                model,
                prefixes=FROZEN_IMAGE_MODULES,
            )
            trainable_gradient = gradient_audit(
                model,
                prefixes=TRAINABLE_DYNAMICS_MODULES,
            )
            first_step_gradient_audit = {
                "frozen_modules": frozen_gradient,
                "trainable_modules": trainable_gradient,
                "native_sigreg_requires_grad": bool(reg_loss.requires_grad),
                "frozen_parameters_have_no_gradient": (
                    frozen_gradient["gradient_tensor_count"] == 0
                ),
                "trainable_parameters_have_nonzero_gradient": (
                    trainable_gradient["nonzero_gradient_tensor_count"] > 0
                    and trainable_gradient["gradient_l2_norm"] > 0.0
                ),
            }
            if (
                not first_step_gradient_audit[
                    "frozen_parameters_have_no_gradient"
                ]
                or not first_step_gradient_audit[
                    "trainable_parameters_have_nonzero_gradient"
                ]
                or not trainable_gradient["all_present_gradients_finite"]
            ):
                raise RuntimeError(
                    "First-step frozen-representation gradient audit failed: "
                    f"{first_step_gradient_audit}"
                )
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            parameters,
            gradient_clip_norm,
        )
        learning_rate_used = float(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
        if freeze_image_representation:
            pilot.restore_frozen_eval_mode(model)

        if step == 1 or step in snapshot_steps:
            trace.append(
                {
                    "optimizer_step": step,
                    "learning_rate_used": learning_rate_used,
                    "gradient_norm_before_clip": float(gradient_norm),
                    "loss": float(loss.detach()),
                    "pred_loss": float(pred_loss.detach()),
                    "regularizer_loss": float(reg_loss.detach()),
                    "regularizer_weight": regularizer_weight,
                    "regularizer_components": {
                        name: float(value.detach())
                        for name, value in regularizer_components.items()
                    },
                }
            )
        if step in snapshot_steps:
            hidden_metrics = pilot.evaluate_model(
                model,
                evaluation,
                device=device,
                batch_size=eval_batch_size,
            )
            snapshots.append(
                {
                    "optimizer_step": step,
                    "hidden_evaluation": hidden_metrics,
                }
            )
            print(
                f"[{variant}] step={step} "
                "target="
                f"{hidden_metrics['two_real_future_target_selection_rate']:.3f} "
                "history="
                f"{hidden_metrics['correct_history_preference_rate']:.3f} "
                "switch="
                f"{hidden_metrics['correct_rule_switch_rate']:.3f} "
                "worst="
                f"{hidden_metrics['worst_mode_target_selection_rate']:.3f} "
                "ratio="
                f"{hidden_metrics['representation_geometry']['prediction_space']['paired_to_unrelated_ratio']:.6f}",
                flush=True,
            )
            model.train()
            if freeze_image_representation:
                pilot.restore_frozen_eval_mode(model)

    state = {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
    }
    frozen_state_sha256_after = (
        state_subset_sha256(model, FROZEN_IMAGE_MODULES)
        if freeze_image_representation
        else None
    )
    trainable_state_sha256_after = (
        state_subset_sha256(model, TRAINABLE_DYNAMICS_MODULES)
        if freeze_image_representation
        else None
    )
    frozen_state_unchanged = (
        frozen_state_sha256_before == frozen_state_sha256_after
        if freeze_image_representation
        else None
    )
    trainable_state_changed = (
        trainable_state_sha256_before != trainable_state_sha256_after
        if freeze_image_representation
        else None
    )
    if freeze_image_representation and (
        not frozen_state_unchanged or not trainable_state_changed
    ):
        raise RuntimeError(
            "Frozen-representation final state audit failed: "
            f"frozen_state_unchanged={frozen_state_unchanged}, "
            f"trainable_state_changed={trainable_state_changed}"
        )
    checkpoint_output = write_checkpoint(
        state=state,
        output=output,
        variant=variant,
        max_steps=max_steps,
    )
    del original_iterator
    del loader
    del original_dataset
    return {
        "variant": variant,
        "regularizer": regularizer_kind,
        "regularizer_weight": regularizer_weight,
        "conditional_population": conditional_population,
        "include_unpaired_at_conditional_times": (
            conditional_population
            in {
                "highpass_with_unpaired",
                "complete_haar",
                "source_scaled_highpass_with_unpaired",
                "deterministic_target_and_prediction_responses",
            }
        ),
        "seed": seed,
        "optimizer_steps": max_steps,
        "batch": {
            "total": batch_size,
            "original": original_batch_size,
            "hidden": hidden_batch_size,
            "hidden_pairs": hidden_batch_size // 2,
            "ordering": (
                "standard_only"
                if hidden_batch_size == 0
                else "original_then_adjacent_hidden_pairs"
            ),
        },
        "hidden_labels_at_model_or_loss_boundary": False,
        "prediction_supervision": (
            {
                "standard_rows_transition_indices": [0, 1, 2],
                "hidden_rows_transition_indices": [2],
                "hidden_rows_excluded_transition_indices": [0, 1],
                "reason": (
                    "hidden velocity is not identifiable from x0 alone"
                ),
                "public_test_used": False,
            }
            if conditional_population
            in {
                "identifiable_future_only",
                "paired_future_ranking",
                "paired_future_matching",
                "paired_future_fit",
                "paired_future_projected_center",
                "paired_future_response_log_norm",
                "paired_future_projected_geometry",
            }
            else {
                "standard_rows_transition_indices": [0, 1, 2],
                "hidden_rows_transition_indices": [0, 1, 2],
                "hidden_rows_excluded_transition_indices": [],
                "public_test_used": False,
            }
        ),
        "pldm_contract": (
            {
                "model_architecture": "stable_worldmodel.wm.pldm.pldm.PLDM",
                "prediction_mse_weight": 1.0,
                "active_regularizers": {
                    "std": 18.0,
                    "std_t": 0.7,
                    "cov": 12.0,
                    "temp_align": 0.2,
                },
                "inactive_official_terms": {
                    "idm": 0.0,
                    "cov_t": 0.0,
                    "temp_straight": 0.0,
                },
                "same_as_tworoom_pldm_active_objective": True,
            }
            if regularizer_kind
            in {"pldm", "pldm_paired_future_ranking"}
            else None
        ),
        "conditional_active_times": (
            [1, 3]
            if regularizer_kind
            in {
                "conditional",
                "group_balanced",
                "scale_calibrated",
                "dynamics_response",
                "paired_future_ranking",
                "pldm_paired_future_ranking",
                "paired_future_matching",
                "paired_future_fit",
                "paired_future_projected_center",
                "paired_future_response_log_norm",
                "paired_future_projected_geometry",
            }
            else []
        ),
        "scale_calibrated_contract": (
            {
                "population": (
                    "unpaired_standard_rows_plus_source_scaled_contrasts"
                ),
                "sigreg_calls_per_optimizer_step": 1,
                "native_sigreg_stacked": False,
                "external_regularizer_weight_count": 1,
                "source_scales": contrast_scale_receipt,
            }
            if regularizer_kind == "scale_calibrated"
            else None
        ),
        "dynamics_response_contract": (
            {
                "population": (
                    "standard target/prediction rows plus source-scaled "
                    "real and identifiable deterministic response contrasts"
                ),
                "predictor_and_pred_proj_mode_for_regularizer": "eval",
                "gradients_enabled": True,
                "reserve_factor": 2.0**0.5,
                "target_pair_active_future_times": [1, 3],
                "prediction_pair_active_future_times": [3],
                "irreducible_prediction_future_time_1_included": False,
                "sigreg_calls_per_optimizer_step": 1,
                "native_sigreg_stacked": False,
                "external_regularizer_weight_count": 1,
                "source_scales": contrast_scale_receipt,
            }
            if regularizer_kind == "dynamics_response"
            and conditional_population
            == "deterministic_target_and_prediction_responses"
            else None
        ),
        "paired_response_alignment_contract": (
            {
                "target": (
                    "scaled difference between the two real x3 futures"
                ),
                "prediction": (
                    "scaled difference between the two deterministic x3 predictions"
                ),
                "scale_source": contrast_scale_receipt,
                "hidden_labels_at_loss_boundary": False,
                "pair_order_only": True,
                "active_future_time": 3,
            }
            if regularizer_kind == "dynamics_response"
            and conditional_population == "paired_response_alignment"
            else None
        ),
        "paired_future_ranking_contract": (
            {
                "active_future_time": 3,
                "prediction_target": (
                    "each prediction is ranked against the two real futures"
                ),
                "normalization": "per-pair real-future squared distance",
                "margin": 0.5,
                "hidden_labels_at_loss_boundary": False,
                "pair_order_only": True,
                "ambiguous_early_hidden_transition_excluded": True,
            }
            if regularizer_kind
            in {"paired_future_ranking", "pldm_paired_future_ranking"}
            else None
        ),
        "paired_future_matching_contract": (
            {
                "active_future_time": 3,
                "comparisons": [
                    "each prediction against both real futures",
                    "each real future against both history-conditioned predictions",
                    "predicted change direction against real change direction",
                ],
                "normalization": "per-pair real-future squared distance",
                "margin": 0.5,
                "hidden_labels_at_loss_boundary": False,
                "pair_order_only": True,
                "ambiguous_early_hidden_transition_excluded": True,
            }
            if regularizer_kind == "paired_future_matching"
            else None
        ),
        "paired_future_fit_contract": (
            {
                "active_future_time": 3,
                "comparisons": [
                    "both paired ranking directions",
                    "predicted response versus real response",
                    "each prediction versus its real future",
                ],
                "fit_normalization": (
                    "log-scaled by each pair's real-future squared distance"
                ),
                "hidden_labels_at_loss_boundary": False,
                "pair_order_only": True,
                "ambiguous_early_hidden_transition_excluded": True,
            }
            if regularizer_kind == "paired_future_fit"
            else None
        ),
        "paired_future_projected_geometry_contract": (
            {
                "active_future_time": 3,
                "base_comparisons": [
                    "each prediction against both real futures",
                    "each real future against both predictions",
                    "predicted response direction against real response",
                ],
                "projected_center_enabled": (
                    regularizer_kind
                    in {
                        "paired_future_projected_center",
                        "paired_future_projected_geometry",
                    }
                ),
                "response_log_norm_enabled": (
                    regularizer_kind
                    in {
                        "paired_future_response_log_norm",
                        "paired_future_projected_geometry",
                    }
                ),
                "full_latent_response_fit": False,
                "hidden_labels_at_loss_boundary": False,
                "pair_order_only": True,
                "ambiguous_early_hidden_transition_excluded": True,
            }
            if regularizer_kind
            in {
                "paired_future_projected_center",
                "paired_future_response_log_norm",
                "paired_future_projected_geometry",
            }
            else None
        ),
        "transition_conditional_contract": (
            {
                "population": "concatenated_real_targets_and_predictions",
                "target_pair_active_future_times": [1, 3],
                "prediction_pair_active_future_times": [3],
                "irreducible_prediction_future_time_1_active": False,
                "native_target_sigreg_replaced_not_stacked": True,
                "external_regularizer_weight_count": 1,
            }
            if regularizer_kind == "transition_conditional"
            else None
        ),
        "representation_freeze": {
            "enabled": freeze_image_representation,
            "frozen_modules": frozen_modules,
            "force_frozen_modules_eval_mode": freeze_image_representation,
            "optimizer_excludes_frozen_parameters": (
                optimizer_excludes_frozen
            ),
            "trainable_top_level_modules": sorted(
                {
                    name.split(".", 1)[0]
                    for name, parameter in model.named_parameters()
                    if parameter.requires_grad
                }
            ),
            "trainable_parameter_count": sum(
                parameter.numel() for parameter in parameters
            ),
            "frozen_state_sha256_before": frozen_state_sha256_before,
            "frozen_state_sha256_after": frozen_state_sha256_after,
            "frozen_state_unchanged": frozen_state_unchanged,
            "trainable_state_sha256_before": trainable_state_sha256_before,
            "trainable_state_sha256_after": trainable_state_sha256_after,
            "trainable_state_changed": trainable_state_changed,
            "first_step_gradient_audit": first_step_gradient_audit,
        },
        "source_checkpoint": checkpoint_receipt,
        "optimizer": {
            "type": "AdamW",
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "gradient_clip_norm": gradient_clip_norm,
            "scheduler": "LinearWarmupCosineAnnealingLR",
            "warmup_steps": warmup_steps,
            "scheduler_max_steps": scheduler_max_steps,
        },
        "precision": (
            "bf16_mixed_autocast" if device.type == "cuda" else "float32"
        ),
        "loss_trace": trace,
        "snapshots": snapshots,
        "final_checkpoint": checkpoint_output,
        "elapsed_seconds": time.monotonic() - started,
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
        default=DEFAULT_ORIGINAL_LANCE,
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
        default=DEFAULT_CONTRAST_SCALES,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--variants",
        default=",".join(VARIANT_WEIGHTS),
        help=(
            "Comma-separated subset of: "
            + ", ".join(VARIANT_WEIGHTS)
        ),
    )
    parser.add_argument("--max-steps", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=3073)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--original-batch-size",
        type=int,
        default=None,
        help=(
            "Standard Push-T rows per batch. Defaults to half of "
            "--batch-size; set equal to --batch-size for the standard-only "
            "continuation control."
        ),
    )
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    variants = tuple(
        value.strip() for value in args.variants.split(",") if value.strip()
    )
    unknown = sorted(set(variants) - set(VARIANT_WEIGHTS))
    if not variants or unknown or len(variants) != len(set(variants)):
        raise ValueError(
            f"Invalid --variants; unknown={unknown}, values={variants}"
        )
    if args.batch_size <= 0 or args.batch_size % 4:
        raise ValueError("--batch-size must be positive and divisible by 4")
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative")
    original_batch_size, hidden_batch_size = resolve_batch_partition(
        args.batch_size,
        args.original_batch_size,
    )
    pair_dependent_variants = [
        variant
        for variant in variants
        if VARIANT_WEIGHTS[variant][0] not in {"native", "pldm"}
    ]
    if hidden_batch_size == 0 and pair_dependent_variants:
        raise ValueError(
            "A standard-only batch cannot train pair-dependent variants: "
            f"{pair_dependent_variants}"
        )

    hidden_root = args.hidden_data_root.expanduser().resolve()
    original_lance = args.original_lance.expanduser().resolve()
    action_source = args.action_normalizer_source.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    contrast_scale_path = args.contrast_scales.expanduser().resolve()
    output = Path(os.path.abspath(args.output.expanduser()))
    required = [
        hidden_root / "manifest.json",
        hidden_root / "train.lance",
        hidden_root / "eval_payloads",
        original_lance,
        action_source,
        checkpoint,
    ]
    uses_scale_calibration = any(
        VARIANT_WEIGHTS[variant][0]
        in {"scale_calibrated", "dynamics_response"}
        for variant in variants
    )
    if uses_scale_calibration:
        required.append(contrast_scale_path)
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing input(s):\n" + "\n".join(map(str, missing))
        )
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite output: {output}")
    output.mkdir(parents=True)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    action_stats = pilot.original_action_stats(action_source)
    print("Materializing audited hidden paired split", flush=True)
    hidden = pilot.materialize_lance_split(
        hidden_root / "train.lance",
        action_stats=action_stats,
    )
    evaluation = pilot.load_eval_payloads(
        hidden_root / "eval_payloads",
        action_stats=action_stats,
    )
    if hidden_batch_size and hidden.pair_count % (hidden_batch_size // 2):
        raise ValueError(
            "Hidden pair count must divide by hidden pairs per batch"
        )
    contrast_scales = None
    contrast_scale_receipt = None
    if uses_scale_calibration:
        (
            contrast_scales,
            contrast_scale_receipt,
        ) = load_contrast_scales(contrast_scale_path)

    config_path = output / "config.json"
    model_config_names = {
        model_config_name_for_variant(variant) for variant in variants
    }
    if len(model_config_names) != 1:
        raise ValueError(
            "One output directory cannot mix LeWM and PLDM model configs; "
            "run the PLDM control in its own output directory"
        )
    selected_model_config = next(iter(model_config_names))
    config_path.write_text(
        json.dumps(
            model_config(selected_model_config),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    provenance = {
        "schema_version": 1,
        "status": (
            "standard_only_continuation_control"
            if hidden_batch_size == 0
            else "registered_mixed_retention_pilot"
        ),
        "owner": "ContextWorld",
        "hidden_data": {
            "root": str(hidden_root),
            "manifest_sha256": pilot.file_sha256(
                hidden_root / "manifest.json"
            ),
            "train_pairs": hidden.pair_count,
            "eval_pairs": int(evaluation["low_pixels"].size(0)),
        },
        "original_data": {
            "path": str(original_lance),
            "frameskip": 5,
            "num_steps": 4,
        },
        "action_normalizer": {
            "source": str(action_source),
            "mean": action_stats["mean"].tolist(),
            "std": action_stats["std"].tolist(),
        },
        "source_checkpoint": {
            "path": str(checkpoint),
            "sha256": pilot.file_sha256(checkpoint),
        },
        "model_input": ["pixels", "action"],
        "model_config": selected_model_config,
        "forbidden_fields": [
            "hidden_mode",
            "hidden_action_scale",
            "pair_id",
            "physics_state",
        ],
        "device": str(device),
        "cuda_device_name": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None
        ),
    }
    results = []
    for index, variant in enumerate(variants, start=1):
        print(f"[{index}/{len(variants)}] training {variant}", flush=True)
        results.append(
            train_variant(
                variant=variant,
                checkpoint=checkpoint,
                original_path=original_lance,
                hidden=hidden,
                evaluation=evaluation,
                action_stats=action_stats,
                output=output,
                device=device,
                seed=args.seed,
                max_steps=args.max_steps,
                batch_size=args.batch_size,
                original_batch_size=original_batch_size,
                eval_batch_size=args.eval_batch_size,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                gradient_clip_norm=args.gradient_clip_norm,
                num_workers=args.num_workers,
                contrast_scales=contrast_scales,
                contrast_scale_receipt=contrast_scale_receipt,
            )
        )
        report = {
            "provenance": provenance,
            "training_contract": {
                "seed": args.seed,
                "max_steps": args.max_steps,
                "batch_size": args.batch_size,
                "original_batch_size": original_batch_size,
                "hidden_batch_size": hidden_batch_size,
                "original_fraction": (
                    original_batch_size / args.batch_size
                ),
                "hidden_fraction": hidden_batch_size / args.batch_size,
                "same_source_checkpoint": True,
                "same_original_sampler_seed": True,
                "same_hidden_pair_order_seed": bool(hidden_batch_size),
            },
            "results": results,
        }
        (output / "mixed_report.partial.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )

    report = {
        "provenance": provenance,
        "training_contract": {
            "seed": args.seed,
            "max_steps": args.max_steps,
            "batch_size": args.batch_size,
            "original_batch_size": original_batch_size,
            "hidden_batch_size": hidden_batch_size,
            "original_fraction": original_batch_size / args.batch_size,
            "hidden_fraction": hidden_batch_size / args.batch_size,
            "same_source_checkpoint": True,
            "same_original_sampler_seed": True,
            "same_hidden_pair_order_seed": bool(hidden_batch_size),
        },
        "results": results,
    }
    report_path = output / "mixed_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "report": str(report_path),
                "report_sha256": pilot.file_sha256(report_path),
                "variants": {
                    row["variant"]: row["snapshots"][-1][
                        "hidden_evaluation"
                    ]
                    for row in results
                },
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
