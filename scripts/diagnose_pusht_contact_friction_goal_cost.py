#!/usr/bin/env python3
"""Compare the model's true-image latent cost with the physical action oracle.

This diagnostic removes prediction error from contact-friction planning.  It
renders the real high-friction future for every candidate action, encodes that
real image with the checkpoint's target encoder, and compares it with the
stored low-friction goal.  If this target-only cost does not prefer the
physical oracle region, adding more predictor training data cannot make the
same latent planning cost pass.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
STABLE_WORLD_MODEL_ROOT = ROOT.parent / "stable-worldmodel"
for source_root in (ROOT, STABLE_WORLD_MODEL_ROOT):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from contextworld.benchmarks.adapters import (  # noqa: E402
    StableWorldModelLeWMContactFrictionAdapter,
)
from contextworld.benchmarks.contact_friction_icl_data import (  # noqa: E402
    DEFAULT_CONTACT_FRICTION_RELEASE_CONFIG,
    ContactFrictionICLEvalDataset,
    file_sha256,
    load_contact_friction_icl_release,
)
from contextworld.evaluation.pusht_contact_friction_h3 import (  # noqa: E402
    ContactFrictionTemplate,
    simulate_query_future,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--release-config",
        type=Path,
        default=DEFAULT_CONTACT_FRICTION_RELEASE_CONFIG,
    )
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pair-count", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.pair_count <= 0:
        raise ValueError("--pair-count must be positive")
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite output: {output}")
    release_path = args.release_config.expanduser().resolve()
    release = load_contact_friction_icl_release(release_path)
    dataset = ContactFrictionICLEvalDataset(
        release=release,
        repo_root=ROOT,
    )
    arrays = dataset.arrays
    pair_count = min(int(args.pair_count), arrays.pair_count)
    oracle_path = args.oracle.expanduser().resolve()
    oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
    if tuple(
        row["pair_id"] for row in oracle["pairs"][:pair_count]
    ) != arrays.pair_ids[:pair_count]:
        raise RuntimeError("Planning oracle and Validation order differ")
    scales = np.asarray(
        oracle["candidate_actions"]["scales"],
        dtype=np.float64,
    )
    manifest = json.loads(
        (dataset.root / "manifest.json").read_text(encoding="utf-8")
    )
    pairs = manifest["splits"]["validation"]["pairs"]
    normalization = release["evaluation"]["action_normalization"]
    runtime = release["runtime"]["stable_worldmodel"]
    checkpoint = args.checkpoint.expanduser().resolve()
    adapter = StableWorldModelLeWMContactFrictionAdapter.from_checkpoint(
        checkpoint,
        action_mean=normalization["mean"],
        action_std=normalization["std_population"],
        repo_root=ROOT,
        stablewm_repo=runtime["repo"],
        stablewm_ref=runtime["expected_ref"],
        device=args.device,
    )
    before = adapter.frozen_state_hash()
    goals = adapter.encode_pixels(
        arrays.low_pixels[:pair_count, 3],
        batch_size=args.batch_size,
    )

    records = []
    selected_indices = []
    started = time.monotonic()
    for index in range(pair_count):
        pair = pairs[index]
        template = ContactFrictionTemplate(**pair["template"])
        if template.template_id != arrays.pair_ids[index]:
            raise RuntimeError("Manifest and Validation order differ")
        base_query = np.asarray(template.query_actions, dtype=np.float64)
        candidate_pixels = []
        for scale in scales:
            candidate_query = np.clip(base_query * scale, -1.0, 1.0)
            candidate = replace(
                template,
                query_actions=tuple(map(tuple, candidate_query.tolist())),
            )
            result = simulate_query_future(
                candidate,
                mode="high_friction",
                canonical_query_snapshot=(
                    template.canonical_query_snapshot
                ),
                resolution=224,
                render_pixels=True,
            )
            candidate_pixels.append(result["future_pixels"])
        encoded = adapter.encode_pixels(
            np.stack(candidate_pixels),
            batch_size=args.batch_size,
        )
        costs = np.square(encoded - goals[index][None]).mean(axis=-1)
        selected = int(np.argmin(costs))
        selected_indices.append(selected)
        physical = oracle["pairs"][index]["modes"]["high_friction"]
        acceptable = set(physical["acceptable_candidate_indices"])
        records.append(
            {
                "pair_id": arrays.pair_ids[index],
                "selected_candidate_index": selected,
                "selected_scale": float(scales[selected]),
                "selected_target_latent_cost": float(costs[selected]),
                "physical_oracle_best_scale": float(
                    physical["best_scale"]
                ),
                "physical_acceptable_candidate_indices": sorted(acceptable),
                "target_latent_cost_selects_physical_region": (
                    selected in acceptable
                ),
                "target_latent_costs": costs.tolist(),
            }
        )
        if (index + 1) % 8 == 0 or index + 1 == pair_count:
            print(
                f"target-only oracle {index + 1}/{pair_count}",
                flush=True,
            )

    selected_scales = scales[np.asarray(selected_indices, dtype=np.int64)]
    agreement = np.asarray(
        [
            row["target_latent_cost_selects_physical_region"]
            for row in records
        ],
        dtype=np.bool_,
    )
    after = adapter.frozen_state_hash()
    if before != after:
        raise RuntimeError("Model state changed during target-only audit")
    payload = {
        "schema_version": 1,
        "status": "completed",
        "diagnostic": (
            "pusht_contact_friction_true_future_target_latent_cost"
        ),
        "interpretation": (
            "Prediction is bypassed: every candidate is a real simulator "
            "future encoded by the checkpoint target encoder."
        ),
        "release": {
            "release_id": release["release_id"],
            "release_config_sha256": file_sha256(release_path),
            "data_manifest_sha256": release["data"]["manifest_sha256"],
        },
        "oracle": {
            "path": str(oracle_path),
            "sha256": file_sha256(oracle_path),
        },
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": file_sha256(checkpoint),
            "adapter": adapter.metadata,
        },
        "sample": {
            "selection": "first_pairs_in_frozen_validation_order",
            "pair_count": pair_count,
            "candidate_count_per_pair": len(scales),
            "real_simulator_futures": pair_count * len(scales),
        },
        "metrics": {
            "physical_region_agreement_rate": float(agreement.mean()),
            "selected_scale": {
                "minimum": float(selected_scales.min()),
                "median": float(np.median(selected_scales)),
                "mean": float(selected_scales.mean()),
                "maximum": float(selected_scales.max()),
            },
        },
        "records": records,
        "elapsed_seconds": time.monotonic() - started,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "completed",
                "output": str(output),
                "metrics": payload["metrics"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
