"""Public bundle scoring path for Motion Damping.

The frozen legacy scorer and its source audit remain unchanged. This entry
point validates the downloaded Test payload and reuses its data reader,
metric kernel, and gate without claiming historical runtime identity.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from contextworld.benchmarks.adapters import MotionDampingICLModelAdapter, validate_adapter_protocol
from contextworld.benchmarks.bundle_test import verify_public_test_bundle
from contextworld.benchmarks.motion_damping_icl_data import (
    DEFAULT_MOTION_DAMPING_RELEASE_CONFIG,
    MotionDampingICLEvalDataset,
    file_sha256,
    load_motion_damping_icl_release,
)
from contextworld.benchmarks.motion_damping_icl_score import (
    motion_damping_prediction_gate,
    motion_damping_prediction_metrics,
)
from contextworld.paths import repository_root


def evaluate_motion_damping_bundle_model(
    *,
    adapter: MotionDampingICLModelAdapter,
    model_name: str,
    training_recipe: str,
    training_seed: int | None,
    release_config: Path | str = DEFAULT_MOTION_DAMPING_RELEASE_CONFIG,
    repo_root: Path | None = None,
    test_bundle_root: Path,
    batch_size: int = 64,
    include_records: bool = True,
) -> dict[str, Any]:
    root = (repo_root or repository_root()).resolve()
    bundle = Path(test_bundle_root).expanduser()
    if not bundle.is_absolute() or bundle.resolve(strict=True) != root:
        raise ValueError("Motion Damping Test bundle root must match repo_root")
    bundle_verification = verify_public_test_bundle(
        bundle, task="motion_damping", release_config=release_config
    )
    release = load_motion_damping_icl_release(release_config)
    dataset = MotionDampingICLEvalDataset(release=release, repo_root=root)
    arrays = dataset.arrays
    if not dataset.is_full_protocol:
        raise RuntimeError("Formal Motion Damping scoring requires 256 pairs")
    validate_adapter_protocol(
        adapter,
        history_tokens=3,
        action_block_raw_steps=5,
        action_dim=2,
        minimum_future_action_blocks=1,
        task_name="Motion Damping v1",
    )
    histories = np.concatenate(
        [
            arrays.faster_decay_pixels[:, :3],
            arrays.no_extra_decay_pixels[:, :3],
        ],
        axis=0,
    )
    actions = np.concatenate(
        [arrays.raw_action_blocks[:, :3], arrays.raw_action_blocks[:, :3]],
        axis=0,
    )
    before = adapter.frozen_state_hash()
    predicted = adapter.rollout_latents(histories, actions, batch_size=batch_size)
    if predicted.ndim != 3 or predicted.shape[1] != 1:
        raise RuntimeError("Motion Damping adapter must return one future")
    true_futures = np.concatenate(
        [
            arrays.faster_decay_pixels[:, 3],
            arrays.no_extra_decay_pixels[:, 3],
        ],
        axis=0,
    )
    encoded = adapter.encode_pixels(true_futures, batch_size=batch_size)
    after = adapter.frozen_state_hash()
    if before != after:
        raise RuntimeError("Model state changed during Motion Damping scoring")
    count = arrays.pair_count
    metrics, records = motion_damping_prediction_metrics(
        pair_ids=arrays.pair_ids,
        predicted_faster_decay=predicted[:count, 0],
        predicted_no_extra_decay=predicted[count:, 0],
        target_faster_decay=encoded[:count],
        target_no_extra_decay=encoded[count:],
    )
    release_path = Path(release["_config_path"])
    result = {
        "schema_version": 1,
        "benchmark": "pusht_history3_motion_damping_icl",
        "submission_kind": "single_checkpoint",
        "status": "completed",
        "release": {
            "release_id": release["release_id"],
            "release_config_sha256": file_sha256(release_path),
            "data_manifest_sha256": release["data"]["manifest_sha256"],
            "sealed_test_included": False,
        },
        "model": {
            "name": str(model_name),
            "training_recipe": str(training_recipe),
            "training_seed": training_seed,
            "adapter": adapter.metadata,
            "state_sha256_before": before,
            "state_sha256_after": after,
        },
        "data": dataset.describe(),
        "metrics": metrics,
        "gate": motion_damping_prediction_gate(metrics, release=release),
    }
    result["bundle_test_verification"] = bundle_verification
    if include_records:
        result["records"] = records
    return result

