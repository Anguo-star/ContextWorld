#!/usr/bin/env python3
"""Train one Reacher History-3 arm-mass ICL reference checkpoint."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import h5py
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
for value in (ROOT, Path(__file__).resolve().parent):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from contextworld.benchmarks.reacher_arm_mass_icl_data import (  # noqa: E402
    DEFAULT_REACHER_ARM_MASS_RELEASE_CONFIG,
    _read_lance_pairs,
    load_reacher_arm_mass_icl_release,
)
import run_pusht_contact_friction_h3_train as trainer  # noqa: E402


def _finite_action_stats(path: Path) -> dict[str, Any]:
    """Match population z-score normalization while excluding terminal NaNs."""

    count = 0
    total = np.zeros(2, dtype=np.float64)
    square_total = np.zeros(2, dtype=np.float64)
    with h5py.File(path, "r", swmr=True) as handle:
        actions = handle["action"]
        for start in range(0, int(actions.shape[0]), 200_000):
            batch = actions[start : start + 200_000].astype(np.float64)
            batch = batch[np.isfinite(batch).all(axis=1)]
            count += len(batch)
            total += batch.sum(axis=0)
            square_total += np.square(batch).sum(axis=0)
    if not count:
        raise RuntimeError(f"No finite Reacher actions in {path}")
    mean = total / count
    variance = square_total / count - np.square(mean)
    return {
        "count": count,
        "mean": mean.astype(np.float32),
        "std": np.sqrt(np.maximum(variance, 0.0)).astype(np.float32),
        "source": str(path),
        "source_size_bytes": path.stat().st_size,
        "method": "population_zscore_after_excluding_terminal_nan_rows",
    }


def _loader_validation(
    path: Path,
    *,
    expected_pairs: int,
    action_stats: dict[str, Any],
) -> dict[str, torch.Tensor]:
    arrays = _read_lance_pairs(
        path,
        expected_pairs=expected_pairs,
        expected_split="loader_validation",
    )

    def pixels(values: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(values.copy()).permute(0, 1, 4, 2, 3)

    action = torch.from_numpy(
        arrays.raw_action_blocks.copy()
    ).reshape(expected_pairs, 4, 10)
    # The reused evaluator reads physical positions from state[..., 2:4].
    # Put the actual finger x/y there, then rename that diagnostic below so a
    # Reacher report never calls it a PushT block/pixel distance.
    padding = np.zeros_like(arrays.lighter_finger_positions)
    lighter_physical = np.concatenate(
        [padding, arrays.lighter_finger_positions], axis=-1
    )
    heavier_physical = np.concatenate(
        [padding, arrays.heavier_finger_positions], axis=-1
    )
    return {
        "low_pixels": pixels(arrays.lighter_pixels),
        "high_pixels": pixels(arrays.heavier_pixels),
        "action": trainer.pilot.normalize_action_blocks(
            action.float(), action_stats
        ),
        "low_states": torch.from_numpy(lighter_physical),
        "high_states": torch.from_numpy(heavier_physical),
    }


def _install_reacher_diagnostic_name() -> None:
    original = trainer.pilot.evaluate_model

    def evaluate_model(*args: Any, **kwargs: Any) -> dict[str, Any]:
        result = original(*args, **kwargs)
        gap = result.pop("physical_future_block_gap_px")
        result["physical_future_finger_position_gap_world_units"] = gap
        return result

    trainer.pilot.evaluate_model = evaluate_model


def main() -> None:
    frozen_response_variant = (
        "mixed_frozen_image_dynamics_response_sigreg_0p02"
    )
    trainer.mixed.VARIANT_WEIGHTS[frozen_response_variant] = (
        "dynamics_response",
        0.02,
        "deterministic_target_and_prediction_responses",
    )
    trainer.mixed.FROZEN_IMAGE_VARIANTS.add(frozen_response_variant)
    paired_response_variants = {
        "mixed_frozen_image_paired_response_0p05": 0.05,
        "mixed_frozen_image_paired_response_0p20": 0.20,
    }
    for name, weight in paired_response_variants.items():
        trainer.mixed.VARIANT_WEIGHTS[name] = (
            "dynamics_response",
            weight,
            "paired_response_alignment",
        )
        trainer.mixed.FROZEN_IMAGE_VARIANTS.add(name)
    identifiable_future_variant = (
        "mixed_frozen_image_identifiable_future_native_0p09"
    )
    trainer.mixed.VARIANT_WEIGHTS[identifiable_future_variant] = (
        "native",
        0.09,
        "identifiable_future_only",
    )
    trainer.mixed.FROZEN_IMAGE_VARIANTS.add(identifiable_future_variant)
    paired_future_ranking_variant = (
        "mixed_frozen_image_paired_future_ranking_1p00"
    )
    trainer.mixed.VARIANT_WEIGHTS[paired_future_ranking_variant] = (
        "paired_future_ranking",
        1.0,
        "paired_future_ranking",
    )
    trainer.mixed.FROZEN_IMAGE_VARIANTS.add(paired_future_ranking_variant)
    paired_future_matching_variant = (
        "mixed_frozen_image_paired_future_matching_1p00"
    )
    trainer.mixed.VARIANT_WEIGHTS[paired_future_matching_variant] = (
        "paired_future_matching",
        1.0,
        "paired_future_matching",
    )
    trainer.mixed.FROZEN_IMAGE_VARIANTS.add(
        paired_future_matching_variant
    )
    paired_future_fit_variant = (
        "mixed_frozen_image_paired_future_fit_1p00"
    )
    trainer.mixed.VARIANT_WEIGHTS[paired_future_fit_variant] = (
        "paired_future_fit",
        1.0,
        "paired_future_fit",
    )
    trainer.mixed.FROZEN_IMAGE_VARIANTS.add(paired_future_fit_variant)
    pldm_paired_future_ranking_variant = (
        "mixed_pldm_paired_future_ranking_1p00"
    )
    trainer.mixed.VARIANT_WEIGHTS[
        pldm_paired_future_ranking_variant
    ] = (
        "pldm_paired_future_ranking",
        1.0,
        "paired_future_ranking",
    )
    trainer.DEFAULT_CONTACT_FRICTION_RELEASE_CONFIG = (
        DEFAULT_REACHER_ARM_MASS_RELEASE_CONFIG
    )
    trainer.load_contact_friction_icl_release = (
        load_reacher_arm_mass_icl_release
    )
    trainer._read_lance_pairs = _read_lance_pairs
    trainer._loader_validation = _loader_validation
    trainer.MODEL_VARIANTS = {
        "lewm": paired_future_fit_variant,
        "pldm": "mixed_pldm_joint",
    }
    trainer.DIAGNOSTIC_VARIANTS = {
        "lewm": {
            "mixed_native_sigreg_0p09",
            "mixed_frozen_image_native_0p09",
            frozen_response_variant,
            *paired_response_variants,
            identifiable_future_variant,
            paired_future_ranking_variant,
            paired_future_matching_variant,
            paired_future_fit_variant,
        },
        "pldm": {
            "mixed_pldm_joint",
            pldm_paired_future_ranking_variant,
        },
    }
    trainer.CAPABILITY_SLUG = "reacher_arm_mass"
    trainer.CAPABILITY_DISPLAY = "Reacher arm-mass"
    trainer.HIDDEN_FIELD = "hidden_arm_density"
    trainer.TRAINER_DESCRIPTION = __doc__
    trainer.ORIGINAL_BATCH_KEY = "original_reacher_samples_per_batch"
    trainer.ACTION_STATS_LOADER = _finite_action_stats
    _install_reacher_diagnostic_name()
    trainer.main()


if __name__ == "__main__":
    main()
