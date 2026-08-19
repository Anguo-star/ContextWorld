#!/usr/bin/env python3
"""Run the frozen B1/B2 Motion Damping center-barrier Dev candidates."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Callable

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from contextworld.benchmarks.motion_damping_icl_data import (  # noqa: E402
    DEFAULT_MOTION_DAMPING_RELEASE_CONFIG,
    file_sha256,
    load_motion_damping_icl_release,
)
from contextworld.paths import resolve_contextworld_path  # noqa: E402
from contextworld.training.paired_prediction_geometry import (  # noqa: E402
    paired_batch_prediction_geometry_loss,
    scale_payload_to_spec,
)
import run_pusht_motion_damping_h3_train as motion  # noqa: E402
import run_pusht_hidden_actuation_mixed as mixed  # noqa: E402


SHARED_RUNNER_SHA256 = (
    "ed3f912b716c4090245d2388c11fabcda7962ac8e6c4aa8614d10ea69bcb17b0"
)
SELECTION_SEED = 14321
OPTIMIZER_STEPS = 8192
EXPECTED_MANIFEST_SHA256 = (
    "48246aa4ae4a13d5b1c9677ba37a92fe114129027745f8e258137a016899563b"
)
LOSS_MODULE_PATH = (
    ROOT / "contextworld/training/paired_prediction_geometry.py"
)
EXPECTED_LOSS_MODULE_SHA256 = (
    "3d657ccc3d24fff4a2228974f261273a3a2e4959cb73f67e2255a01866ac6ba5"
)
CANDIDATES = {
    "mixed_frozen_image_history_center_barrier_b1": {
        "candidate": "B1",
        "center_weight": 1.0,
    },
    "mixed_frozen_image_history_center_barrier_b2": {
        "candidate": "B2",
        "center_weight": 2.0,
    },
}


def _runner_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--barrier-scales", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--release-config",
        type=Path,
        default=DEFAULT_MOTION_DAMPING_RELEASE_CONFIG,
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--optimizer-steps", type=int, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=motion.trainer.DEFAULT_CHECKPOINT,
    )
    parsed, _ = parser.parse_known_args()
    if parsed.model != "lewm":
        raise ValueError("Motion B1/B2 is frozen to the LeWM adapter")
    if parsed.seed != SELECTION_SEED:
        raise ValueError(f"Motion B1/B2 requires seed {SELECTION_SEED}")
    if parsed.optimizer_steps != OPTIMIZER_STEPS:
        raise ValueError(
            f"Motion B1/B2 requires {OPTIMIZER_STEPS} optimizer steps"
        )
    forwarded: list[str] = []
    index = 1
    while index < len(sys.argv):
        value = sys.argv[index]
        if value in {
            "--barrier-scales",
            "--data-root",
            "--optimizer-steps",
        }:
            index += 2
            continue
        if any(
            value.startswith(f"{option}=")
            for option in (
                "--barrier-scales",
                "--data-root",
                "--optimizer-steps",
            )
        ):
            index += 1
            continue
        forwarded.append(value)
        index += 1
    return parsed, forwarded


def _register_candidate(variant: str) -> None:
    """Register one candidate without changing shared source code."""

    mixed.VARIANT_WEIGHTS[variant] = (
        "paired_future_matching",
        1.0,
        "paired_future_matching",
    )
    mixed.FROZEN_IMAGE_VARIANTS.add(variant)
    motion.trainer.DIAGNOSTIC_VARIANTS["lewm"].add(variant)
    motion.TWIN_GROUP_VARIANTS.add(variant)


def _augment_provenance(
    *,
    output: Path,
    barrier_provenance: dict[str, Any],
) -> None:
    sidecar = output / "paired_geometry_barrier_provenance.json"
    sidecar.write_text(
        json.dumps(barrier_provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    provenance_path = output / "training_provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["paired_prediction_geometry_barrier"] = barrier_provenance
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_path = output / "training_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["provenance"] = provenance
    report["paired_prediction_geometry_barrier"] = barrier_provenance
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    runner_args, forwarded = _runner_args()
    if runner_args.variant not in CANDIDATES:
        raise ValueError(
            f"Expected one of the two frozen candidates: {tuple(CANDIDATES)}"
        )
    shared_path = SCRIPT_ROOT / "run_pusht_hidden_actuation_mixed.py"
    observed_shared_hash = file_sha256(shared_path)
    if observed_shared_hash != SHARED_RUNNER_SHA256:
        raise RuntimeError(
            "Shared runner changed after B1/B2 preregistration: "
            f"{observed_shared_hash}"
        )
    observed_loss_hash = file_sha256(LOSS_MODULE_PATH)
    if observed_loss_hash != EXPECTED_LOSS_MODULE_SHA256:
        raise RuntimeError(
            "Paired-geometry loss changed after B1/B2 preregistration: "
            f"{observed_loss_hash}"
        )

    scales_path = runner_args.barrier_scales.expanduser().resolve()
    scale_payload = json.loads(scales_path.read_text(encoding="utf-8"))
    if (
        scale_payload.get("role")
        != "training_only_loss_scale_freeze_not_model_selection"
        or scale_payload.get("public_test_opened") is not False
        or scale_payload.get("development_opened") is not False
    ):
        raise RuntimeError("Barrier scales are not a Training-only freeze")
    data_root = runner_args.data_root.expanduser().resolve()
    manifest_path = data_root / "manifest.json"
    release = load_motion_damping_icl_release(
        runner_args.release_config.expanduser().resolve()
    )
    formal_root = resolve_contextworld_path(
        release["data"]["artifact_tree"]["root"], repo_root=ROOT
    )
    formal_steps = int(
        release["training"]["reference_matrix"]["common"][
            "optimizer_steps"
        ]
    )
    manifest_sha256 = file_sha256(manifest_path)
    if (
        data_root != formal_root
        or release["data"]["manifest_sha256"]
        != EXPECTED_MANIFEST_SHA256
        or manifest_sha256 != EXPECTED_MANIFEST_SHA256
        or manifest_sha256 != scale_payload["data"]["manifest_sha256"]
        or formal_steps != OPTIMIZER_STEPS
    ):
        raise RuntimeError("Barrier scales do not match this Training release")
    checkpoint = runner_args.checkpoint.expanduser().resolve()
    if file_sha256(checkpoint) != scale_payload["checkpoint"]["sha256"]:
        raise RuntimeError("Barrier scales use a different initialization")

    candidate = CANDIDATES[runner_args.variant]
    spec = scale_payload_to_spec(
        scale_payload,
        center_weight=float(candidate["center_weight"]),
    )
    original_matching: Callable[..., Any] = mixed.paired_future_matching_loss

    def matching_with_barrier(
        *,
        embeddings: torch.Tensor,
        deterministic_prediction: torch.Tensor,
        pair_indices: torch.Tensor,
        include_fit_terms: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        matching_loss, components = original_matching(
            embeddings=embeddings,
            deterministic_prediction=deterministic_prediction,
            pair_indices=pair_indices,
            include_fit_terms=include_fit_terms,
        )
        barrier_loss, barrier_components = paired_batch_prediction_geometry_loss(
            embeddings=embeddings,
            deterministic_prediction=deterministic_prediction,
            pair_indices=pair_indices,
            spec=spec,
        )
        components.update(barrier_components)
        return matching_loss + barrier_loss, components

    _register_candidate(runner_args.variant)
    barrier_provenance = {
        "schema_version": 1,
        "candidate": candidate["candidate"],
        "candidate_count": 2,
        "variant": runner_args.variant,
        "specification": spec.describe(),
        "scale_freeze": {
            "path": str(scales_path),
            "sha256": file_sha256(scales_path),
            "training_only": True,
        },
        "task_independent_loss_module": {
            "path": str(LOSS_MODULE_PATH),
            "sha256": observed_loss_hash,
        },
        "shared_runner": {
            "path": str(shared_path),
            "sha256": observed_shared_hash,
            "modified_for_candidate": False,
        },
        "motion_damping_batching": {
            "complete_forward_reverse_twins_per_batch": True,
            "condition_rows_per_twin_group": 4,
        },
        "selection_contract": {
            "training_seed": SELECTION_SEED,
            "optimizer_steps": OPTIMIZER_STEPS,
            "fixed_final_checkpoint": True,
            "public_test_opened": False,
            "safe_margin": {
                "correct_future_rate": 0.98,
                "correct_history_rate": 0.98,
                "context_switch_rate": 0.98,
                "worst_damping_correct_future_rate": 0.95,
            },
        },
    }

    original_argv = sys.argv
    mixed.paired_future_matching_loss = matching_with_barrier
    try:
        sys.argv = [original_argv[0], *forwarded]
        motion.main()
    finally:
        sys.argv = original_argv
        mixed.paired_future_matching_loss = original_matching

    output = Path(os.path.abspath(runner_args.output.expanduser()))
    _augment_provenance(
        output=output,
        barrier_provenance=barrier_provenance,
    )
    print(
        json.dumps(
            {
                "status": "completed_with_barrier_provenance",
                "candidate": candidate["candidate"],
                "output": str(output),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
