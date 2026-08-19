#!/usr/bin/env python3
"""Run the frozen B1/B2 Contact Friction center-barrier Dev candidates."""

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

from contextworld.benchmarks.contact_friction_icl_data import (  # noqa: E402
    file_sha256,
)
from contextworld.training.paired_prediction_geometry import (  # noqa: E402
    paired_batch_prediction_geometry_loss,
    scale_payload_to_spec,
)
import run_pusht_contact_friction_h3_train as contact  # noqa: E402
import run_pusht_hidden_actuation_mixed as mixed  # noqa: E402


SHARED_RUNNER_SHA256 = (
    "ed3f912b716c4090245d2388c11fabcda7962ac8e6c4aa8614d10ea69bcb17b0"
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
        "--checkpoint",
        type=Path,
        default=contact.DEFAULT_CHECKPOINT,
    )
    parsed, _ = parser.parse_known_args()
    forwarded: list[str] = []
    index = 1
    while index < len(sys.argv):
        value = sys.argv[index]
        if value == "--barrier-scales":
            index += 2
            continue
        if value.startswith("--barrier-scales="):
            index += 1
            continue
        forwarded.append(value)
        index += 1
    return parsed, forwarded


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
    if (
        file_sha256(manifest_path)
        != scale_payload["data"]["manifest_sha256"]
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
    original_matching: Callable[..., Any] = (
        mixed.paired_future_matching_loss
    )

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
        barrier_loss, barrier_components = (
            paired_batch_prediction_geometry_loss(
                embeddings=embeddings,
                deterministic_prediction=deterministic_prediction,
                pair_indices=pair_indices,
                spec=spec,
            )
        )
        components.update(barrier_components)
        return matching_loss + barrier_loss, components

    mixed.VARIANT_WEIGHTS[runner_args.variant] = (
        "paired_future_matching",
        1.0,
        "paired_future_matching",
    )
    mixed.FROZEN_IMAGE_VARIANTS.add(runner_args.variant)
    contact.DIAGNOSTIC_VARIANTS["lewm"].add(runner_args.variant)
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
            "path": str(
                ROOT
                / "contextworld/training/paired_prediction_geometry.py"
            ),
            "sha256": file_sha256(
                ROOT
                / "contextworld/training/paired_prediction_geometry.py"
            ),
        },
        "shared_runner": {
            "path": str(shared_path),
            "sha256": observed_shared_hash,
            "modified_for_candidate": False,
        },
        "selection_contract": {
            "training_seed": 13313,
            "optimizer_steps": 8192,
            "fixed_final_checkpoint": True,
            "public_test_opened": False,
            "safe_margin": {
                "correct_future_rate": 0.98,
                "correct_history_rate": 0.98,
                "context_switch_rate": 0.98,
                "worst_friction_correct_future_rate": 0.95,
            },
        },
    }

    original_argv = sys.argv
    mixed.paired_future_matching_loss = matching_with_barrier
    try:
        sys.argv = [original_argv[0], *forwarded]
        contact.main()
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
