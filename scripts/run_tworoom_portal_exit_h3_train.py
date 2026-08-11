#!/usr/bin/env python3
"""Train one TwoRoom History-3 portal-exit ICL reference checkpoint."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for value in (ROOT, Path(__file__).resolve().parent):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from contextworld.benchmarks.portal_exit_icl_data import (  # noqa: E402
    DEFAULT_PORTAL_EXIT_RELEASE_CONFIG,
    _read_lance_pairs,
    load_portal_exit_icl_release,
)
from contextworld.paths import resolve_contextworld_path  # noqa: E402
import run_pusht_contact_friction_h3_train as trainer  # noqa: E402


NORMALIZER = "artifacts/splits/tworoom_original_train_s3072_normalizer.json"


def tworoom_action_stats(_: Path) -> dict:
    normalizer = resolve_contextworld_path(NORMALIZER, repo_root=ROOT)
    payload = json.loads(normalizer.read_text(encoding="utf-8"))
    action = payload["columns"]["action"]
    return {
        "count": int(action["valid_rows"]),
        "mean": np.asarray(action["mean"], dtype=np.float32),
        "std": np.asarray(action["std_unbiased"], dtype=np.float32),
        "source": str(normalizer),
        "method": "frozen_original_training_split_unbiased_zscore",
    }


def main() -> None:
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
    trainer.mixed.VARIANT_WEIGHTS[pldm_paired_future_ranking_variant] = (
        "pldm_paired_future_ranking",
        1.0,
        "paired_future_ranking",
    )
    trainer.DEFAULT_CONTACT_FRICTION_RELEASE_CONFIG = (
        DEFAULT_PORTAL_EXIT_RELEASE_CONFIG
    )
    trainer.load_contact_friction_icl_release = load_portal_exit_icl_release
    trainer._read_lance_pairs = _read_lance_pairs
    trainer.CAPABILITY_SLUG = "portal_exit"
    trainer.CAPABILITY_DISPLAY = "portal-exit"
    trainer.HIDDEN_FIELD = "hidden_portal_exit"
    trainer.TRAINER_DESCRIPTION = __doc__
    trainer.ORIGINAL_BATCH_KEY = "original_tworoom_samples_per_batch"
    trainer.ACTION_STATS_LOADER = tworoom_action_stats
    trainer.MODEL_VARIANTS = {
        "lewm": paired_future_fit_variant,
        "pldm": "mixed_pldm_joint",
    }
    trainer.DIAGNOSTIC_VARIANTS = {
        "lewm": {
            "mixed_frozen_image_native_0p09",
            paired_future_fit_variant,
        },
        "pldm": {
            "mixed_pldm_joint",
            pldm_paired_future_ranking_variant,
        },
    }
    trainer.main()


if __name__ == "__main__":
    main()
