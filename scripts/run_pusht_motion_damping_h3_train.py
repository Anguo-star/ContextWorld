#!/usr/bin/env python3
"""Train one PushT History-3 motion-damping ICL reference checkpoint."""

from __future__ import annotations

from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
for value in (ROOT, Path(__file__).resolve().parent):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from contextworld.benchmarks.motion_damping_icl_data import (  # noqa: E402
    DEFAULT_MOTION_DAMPING_RELEASE_CONFIG,
    _read_lance_pairs,
    audit_motion_damping_icl_release,
    load_motion_damping_icl_release,
)
import run_pusht_contact_friction_h3_train as trainer  # noqa: E402


LEWM_REFERENCE_VARIANT = (
    "mixed_frozen_image_identifiable_future_native_0p09"
)
PLDM_REFERENCE_VARIANT = "mixed_pldm_identifiable_future_joint"
LEWM_RANKING_VARIANT = "mixed_frozen_image_paired_future_ranking"
LEWM_RANKING_TWIN_VARIANT = (
    "mixed_frozen_image_paired_future_ranking_twin_1p00"
)
PLDM_RANKING_VARIANT = "mixed_pldm_paired_future_ranking"
LEWM_MATCHING_VARIANT = "mixed_frozen_image_paired_future_matching_1p00"
LEWM_FIT_VARIANT = "mixed_frozen_image_paired_future_fit_1p00"
LEWM_FIT_TWIN_VARIANT = (
    "mixed_frozen_image_paired_future_fit_twin_1p00"
)
LEWM_PROJECTED_CENTER_VARIANT = (
    "mixed_frozen_image_paired_future_projected_center_1p00"
)
LEWM_RESPONSE_LOG_NORM_VARIANT = (
    "mixed_frozen_image_paired_future_response_log_norm_1p00"
)
LEWM_PROJECTED_GEOMETRY_VARIANT = (
    "mixed_frozen_image_paired_future_projected_geometry_1p00"
)
LEWM_PROJECTED_GEOMETRY_PRED_PROJ_ONLY_VARIANT = (
    "mixed_pred_proj_only_paired_future_projected_geometry_1p00"
)
LEWM_PROJECTED_GEOMETRY_LAST_BLOCK_VARIANT = (
    "mixed_last_predictor_block_paired_future_projected_geometry_1p00"
)
LEWM_PROJECTED_GEOMETRY_LAST_TWO_BLOCKS_VARIANT = (
    "mixed_last_two_predictor_blocks_paired_future_projected_geometry_1p00"
)
TWIN_GROUP_VARIANTS = {
    LEWM_RANKING_TWIN_VARIANT,
    LEWM_FIT_TWIN_VARIANT,
    LEWM_PROJECTED_CENTER_VARIANT,
    LEWM_RESPONSE_LOG_NORM_VARIANT,
    LEWM_PROJECTED_GEOMETRY_VARIANT,
    LEWM_PROJECTED_GEOMETRY_PRED_PROJ_ONLY_VARIANT,
    LEWM_PROJECTED_GEOMETRY_LAST_BLOCK_VARIANT,
    LEWM_PROJECTED_GEOMETRY_LAST_TWO_BLOCKS_VARIANT,
}
LOW_CAPACITY_LAST_BLOCKS = {
    LEWM_PROJECTED_GEOMETRY_PRED_PROJ_ONLY_VARIANT: 0,
    LEWM_PROJECTED_GEOMETRY_LAST_BLOCK_VARIANT: 1,
    LEWM_PROJECTED_GEOMETRY_LAST_TWO_BLOCKS_VARIANT: 2,
}


class CompleteTwinPairedBatchStream:
    """Yield complete damping pairs in forward/reverse twin groups.

    Strict motion-damping data stores two adjacent condition pairs for each
    rendered geometry.  The second pair reverses the motion direction and
    exchanges the two x0 images across damping modes.  Keeping both pairs in
    one optimizer batch makes the x0-only shortcut cancel within every
    update, while each low/high condition pair remains adjacent for the
    shared paired-future losses.
    """

    def __init__(
        self,
        pair_count: int,
        *,
        batch_size: int,
        seed: int,
    ) -> None:
        if pair_count <= 0 or pair_count % 2:
            raise ValueError("pair_count must contain complete twin pairs")
        if batch_size <= 0 or batch_size % 4:
            raise ValueError(
                "batch_size must be divisible by four so every twin group "
                "contributes four condition rows"
            )
        self.pair_count = int(pair_count)
        self.twin_count = self.pair_count // 2
        self.twin_groups_per_batch = batch_size // 4
        if self.twin_count % self.twin_groups_per_batch:
            raise ValueError(
                "twin_count must divide evenly by twin_groups_per_batch"
            )
        self.generator = torch.Generator().manual_seed(int(seed))

    def __iter__(self):
        while True:
            order = torch.randperm(
                self.twin_count,
                generator=self.generator,
            )
            for start in range(
                0,
                self.twin_count,
                self.twin_groups_per_batch,
            ):
                twins = order[
                    start : start + self.twin_groups_per_batch
                ]
                pair_indices = torch.stack(
                    [2 * twins, 2 * twins + 1],
                    dim=1,
                ).flatten()
                yield torch.stack(
                    [2 * pair_indices, 2 * pair_indices + 1],
                    dim=1,
                ).flatten()


def _install_complete_twin_batching(trainer) -> None:
    """Use complete twin batches only for registered damping variants."""

    original_train_variant = trainer.mixed.train_variant

    def train_variant_with_complete_twins(*args, **kwargs):
        variant = kwargs.get("variant")
        if variant not in TWIN_GROUP_VARIANTS:
            return original_train_variant(*args, **kwargs)
        original_stream = trainer.mixed.pilot.PairedBatchStream
        original_model_loader = trainer.mixed.load_model_for_variant
        trainer.mixed.pilot.PairedBatchStream = CompleteTwinPairedBatchStream
        last_blocks = LOW_CAPACITY_LAST_BLOCKS.get(variant)
        if last_blocks is not None:
            def load_low_capacity_model(*loader_args, **loader_kwargs):
                model, receipt = original_model_loader(
                    *loader_args,
                    **loader_kwargs,
                )
                model.predictor.requires_grad_(False)
                model.action_encoder.requires_grad_(False)
                if last_blocks:
                    layers = model.predictor.transformer.layers
                    if last_blocks > len(layers):
                        raise RuntimeError(
                            "Requested more trainable predictor blocks than "
                            "the model contains"
                        )
                    for layer in layers[-last_blocks:]:
                        layer.requires_grad_(True)
                receipt["motion_damping_capacity_policy"] = {
                    "predictor_blocks_total": len(
                        model.predictor.transformer.layers
                    ),
                    "predictor_blocks_trainable_from_end": last_blocks,
                    "action_encoder_trainable": False,
                    "pred_proj_trainable": True,
                }
                return model, receipt

            trainer.mixed.load_model_for_variant = load_low_capacity_model
        try:
            result = original_train_variant(*args, **kwargs)
        finally:
            trainer.mixed.pilot.PairedBatchStream = original_stream
            trainer.mixed.load_model_for_variant = original_model_loader
        result["batch"]["motion_damping_twin_grouping"] = {
            "enabled": True,
            "condition_rows_per_group": 4,
            "condition_pairs_per_group": 2,
            "pair_order_preserved": True,
            "x0_rgb_label_exchange_complete_in_every_group": True,
        }
        if last_blocks is not None:
            result["motion_damping_capacity_policy"] = {
                "predictor_blocks_trainable_from_end": last_blocks,
                "action_encoder_trainable": False,
                "pred_proj_trainable": True,
            }
        return result

    trainer.mixed.train_variant = train_variant_with_complete_twins


def main() -> None:
    # Both components share the same two-endpoint paired-pixel training
    # boundary.  Only the release loader, data reader, and provenance labels
    # differ; the Stable-WorldModel optimization code remains identical.
    trainer.DEFAULT_CONTACT_FRICTION_RELEASE_CONFIG = (
        DEFAULT_MOTION_DAMPING_RELEASE_CONFIG
    )
    def load_audited_release(path):
        audit = audit_motion_damping_icl_release(
            release_config=path,
            repo_root=ROOT,
            full=False,
        )
        if not audit["passed"]:
            raise RuntimeError(
                "Motion-damping strict causal release audit failed before "
                "training"
            )
        return load_motion_damping_icl_release(path)

    trainer.load_contact_friction_icl_release = load_audited_release
    trainer._read_lance_pairs = _read_lance_pairs
    trainer.CAPABILITY_SLUG = "motion_damping"
    trainer.CAPABILITY_DISPLAY = "motion-damping"
    trainer.HIDDEN_FIELD = "hidden_motion_damping"
    trainer.TRAINER_DESCRIPTION = __doc__
    # Strict causal twins make velocity unknowable from x0 alone. Standard
    # PushT rows still supervise every transition, while paired hidden rows
    # supervise only the final transition for which History=3 identifies the
    # motion decay. This removes an irreducible early-transition target
    # without exposing the hidden label.
    trainer.mixed.VARIANT_WEIGHTS.update(
        {
            LEWM_REFERENCE_VARIANT: (
                "native",
                0.09,
                "identifiable_future_only",
            ),
            PLDM_REFERENCE_VARIANT: (
                "pldm",
                1.0,
                "identifiable_future_only",
            ),
            LEWM_RANKING_VARIANT: (
                "paired_future_ranking",
                1.0,
                "paired_future_ranking",
            ),
            LEWM_RANKING_TWIN_VARIANT: (
                "paired_future_ranking",
                1.0,
                "paired_future_ranking",
            ),
            PLDM_RANKING_VARIANT: (
                "pldm_paired_future_ranking",
                1.0,
                "paired_future_ranking",
            ),
            LEWM_MATCHING_VARIANT: (
                "paired_future_matching",
                1.0,
                "paired_future_matching",
            ),
            LEWM_FIT_VARIANT: (
                "paired_future_fit",
                1.0,
                "paired_future_fit",
            ),
            LEWM_FIT_TWIN_VARIANT: (
                "paired_future_fit",
                1.0,
                "paired_future_fit",
            ),
            LEWM_PROJECTED_CENTER_VARIANT: (
                "paired_future_projected_center",
                1.0,
                "paired_future_projected_center",
            ),
            LEWM_RESPONSE_LOG_NORM_VARIANT: (
                "paired_future_response_log_norm",
                1.0,
                "paired_future_response_log_norm",
            ),
            LEWM_PROJECTED_GEOMETRY_VARIANT: (
                "paired_future_projected_geometry",
                1.0,
                "paired_future_projected_geometry",
            ),
            LEWM_PROJECTED_GEOMETRY_PRED_PROJ_ONLY_VARIANT: (
                "paired_future_projected_geometry",
                1.0,
                "paired_future_projected_geometry",
            ),
            LEWM_PROJECTED_GEOMETRY_LAST_BLOCK_VARIANT: (
                "paired_future_projected_geometry",
                1.0,
                "paired_future_projected_geometry",
            ),
            LEWM_PROJECTED_GEOMETRY_LAST_TWO_BLOCKS_VARIANT: (
                "paired_future_projected_geometry",
                1.0,
                "paired_future_projected_geometry",
            ),
        }
    )
    trainer.mixed.FROZEN_IMAGE_VARIANTS.update(
        {
            LEWM_REFERENCE_VARIANT,
            LEWM_RANKING_VARIANT,
            LEWM_RANKING_TWIN_VARIANT,
            LEWM_MATCHING_VARIANT,
            LEWM_FIT_VARIANT,
            LEWM_FIT_TWIN_VARIANT,
            LEWM_PROJECTED_CENTER_VARIANT,
            LEWM_RESPONSE_LOG_NORM_VARIANT,
            LEWM_PROJECTED_GEOMETRY_VARIANT,
            LEWM_PROJECTED_GEOMETRY_PRED_PROJ_ONLY_VARIANT,
            LEWM_PROJECTED_GEOMETRY_LAST_BLOCK_VARIANT,
            LEWM_PROJECTED_GEOMETRY_LAST_TWO_BLOCKS_VARIANT,
        }
    )
    trainer.DIAGNOSTIC_VARIANTS["lewm"].update(
        {
            LEWM_REFERENCE_VARIANT,
            LEWM_RANKING_VARIANT,
            LEWM_RANKING_TWIN_VARIANT,
            LEWM_MATCHING_VARIANT,
            LEWM_FIT_VARIANT,
            LEWM_FIT_TWIN_VARIANT,
            LEWM_PROJECTED_CENTER_VARIANT,
            LEWM_RESPONSE_LOG_NORM_VARIANT,
            LEWM_PROJECTED_GEOMETRY_VARIANT,
            LEWM_PROJECTED_GEOMETRY_PRED_PROJ_ONLY_VARIANT,
            LEWM_PROJECTED_GEOMETRY_LAST_BLOCK_VARIANT,
            LEWM_PROJECTED_GEOMETRY_LAST_TWO_BLOCKS_VARIANT,
        }
    )
    trainer.DIAGNOSTIC_VARIANTS["pldm"].update(
        {PLDM_REFERENCE_VARIANT, PLDM_RANKING_VARIANT}
    )
    trainer.MODEL_VARIANTS = {
        "lewm": LEWM_REFERENCE_VARIANT,
        "pldm": PLDM_REFERENCE_VARIANT,
    }
    _install_complete_twin_batching(trainer)
    trainer.main()


if __name__ == "__main__":
    main()
