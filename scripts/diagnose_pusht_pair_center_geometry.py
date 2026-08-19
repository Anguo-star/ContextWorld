#!/usr/bin/env python3
"""Audit paired PushT prediction geometry on Training or Development.

This is a diagnostic utility.  It never opens the Public Test split.  The
reported pair-center quantities explain why target selection and response
direction can be high while the corresponding-history metric remains low.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contextworld.benchmarks.adapters import (  # noqa: E402
    StableWorldModelLeWMContactFrictionAdapter,
    StableWorldModelLeWMMotionDampingAdapter,
    StableWorldModelPLDMContactFrictionAdapter,
    StableWorldModelPLDMMotionDampingAdapter,
)
from contextworld.benchmarks.contact_friction_icl_data import (  # noqa: E402
    DEFAULT_CONTACT_FRICTION_RELEASE_CONFIG,
    _read_lance_pairs as read_contact_pairs,
    load_contact_friction_icl_release,
)
from contextworld.benchmarks.motion_damping_icl_data import (  # noqa: E402
    DEFAULT_MOTION_DAMPING_RELEASE_CONFIG,
    _read_lance_pairs as read_damping_pairs,
    load_motion_damping_icl_release,
)
from contextworld.paths import resolve_contextworld_path  # noqa: E402


ADAPTERS = {
    ("contact_friction", "lewm"): StableWorldModelLeWMContactFrictionAdapter,
    ("contact_friction", "pldm"): StableWorldModelPLDMContactFrictionAdapter,
    ("motion_damping", "lewm"): StableWorldModelLeWMMotionDampingAdapter,
    ("motion_damping", "pldm"): StableWorldModelPLDMMotionDampingAdapter,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--capability",
        choices=("contact_friction", "motion_damping"),
        required=True,
    )
    parser.add_argument("--model", choices=("lewm", "pldm"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--split", choices=("train", "loader_validation"), required=True
    )
    parser.add_argument("--release-config", type=Path, default=None)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda:6")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def percentile_summary(values: np.ndarray) -> dict[str, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return {"count": 0}
    quantiles = np.percentile(finite, [0, 1, 5, 25, 50, 75, 95, 99, 100])
    return {
        "count": int(finite.size),
        "mean": float(finite.mean()),
        "std": float(finite.std()),
        **{
            name: float(value)
            for name, value in zip(
                ("min", "p01", "p05", "p25", "median", "p75", "p95", "p99", "max"),
                quantiles,
                strict=True,
            )
        },
    }


def paired_geometry(
    predicted_low: np.ndarray,
    predicted_high: np.ndarray,
    target_low: np.ndarray,
    target_high: np.ndarray,
) -> dict[str, Any]:
    p0 = np.asarray(predicted_low, dtype=np.float64)
    p1 = np.asarray(predicted_high, dtype=np.float64)
    t0 = np.asarray(target_low, dtype=np.float64)
    t1 = np.asarray(target_high, dtype=np.float64)
    d = t1 - t0
    q = p1 - p0
    target_center = 0.5 * (t0 + t1)
    prediction_center = 0.5 * (p0 + p1)
    center_error = prediction_center - target_center
    d2 = np.square(d).sum(axis=-1)
    q2 = np.square(q).sum(axis=-1)
    dq = (d * q).sum(axis=-1)
    eps = np.finfo(np.float64).eps

    # e_q is exact for the history decision boundary.  When d.q > 0,
    # both history comparisons pass iff -0.5 < e_q < 0.5.
    e_q = (center_error * q).sum(axis=-1) / np.where(
        np.abs(dq) > eps, dq, np.nan
    )
    # e_d is the center displacement projected onto the real response axis.
    e_d = (center_error * d).sum(axis=-1) / np.maximum(d2, eps)
    response_norm_ratio = np.sqrt(q2 / np.maximum(d2, eps))

    def mse(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.square(a - b).mean(axis=-1)

    ll = mse(p0, t0)
    lh = mse(p0, t1)
    hl = mse(p1, t0)
    hh = mse(p1, t1)
    scale = mse(t0, t1)
    future_low_margin = (lh - ll) / np.maximum(scale, eps)
    future_high_margin = (hl - hh) / np.maximum(scale, eps)
    history_low_margin = (hl - ll) / np.maximum(scale, eps)
    history_high_margin = (lh - hh) / np.maximum(scale, eps)
    switch = dq > 0
    bracket = (e_q > -0.5) & (e_q < 0.5) & switch
    exact_history_both = (history_low_margin > 0) & (history_high_margin > 0)
    if not np.array_equal(bracket, exact_history_both):
        raise RuntimeError("Pair-center identity does not match history margins")

    history_low_correct = history_low_margin > 0
    history_high_correct = history_high_margin > 0
    future_low_correct = future_low_margin > 0
    future_high_correct = future_high_margin > 0
    return {
        "pair_count": int(len(d)),
        "rates": {
            "future_selection": float(
                np.concatenate(
                    [future_low_correct, future_high_correct]
                ).mean()
            ),
            "future_low_selection": float(future_low_correct.mean()),
            "future_high_selection": float(future_high_correct.mean()),
            "history_selection": float(
                np.concatenate(
                    [history_low_correct, history_high_correct]
                ).mean()
            ),
            "history_low_selection": float(history_low_correct.mean()),
            "history_high_selection": float(history_high_correct.mean()),
            "switch_alignment": float(switch.mean()),
            "both_histories_bracketed": float(bracket.mean()),
            "center_outside_real_response_interval_given_switch": float(
                ((np.abs(e_q) >= 0.5) & switch).mean()
            ),
        },
        "failure_counts": {
            "future_low": int((~future_low_correct).sum()),
            "future_high": int((~future_high_correct).sum()),
            "history_low": int((~history_low_correct).sum()),
            "history_high": int((~history_high_correct).sum()),
            "switch": int((~switch).sum()),
            "center_below_prediction_interval_given_switch": int(
                ((e_q <= -0.5) & switch).sum()
            ),
            "center_above_prediction_interval_given_switch": int(
                ((e_q >= 0.5) & switch).sum()
            ),
        },
        "target_pair_mse": percentile_summary(scale),
        "prediction_pair_to_target_pair_squared_norm_ratio": percentile_summary(
            q2 / np.maximum(d2, eps)
        ),
        "prediction_response_norm_ratio": percentile_summary(
            response_norm_ratio
        ),
        "normalized_projected_center_error_on_target_axis": percentile_summary(
            e_d
        ),
        "normalized_projected_center_error_on_prediction_axis": percentile_summary(
            e_q
        ),
        "absolute_prediction_axis_center_error": percentile_summary(
            np.abs(e_q)
        ),
        "normalized_margins": {
            "future_low": percentile_summary(future_low_margin),
            "future_high": percentile_summary(future_high_margin),
            "history_low": percentile_summary(history_low_margin),
            "history_high": percentile_summary(history_high_margin),
        },
    }


def main() -> None:
    args = parse_args()
    if args.capability == "contact_friction":
        release_path = args.release_config or DEFAULT_CONTACT_FRICTION_RELEASE_CONFIG
        release = load_contact_friction_icl_release(release_path)
        reader = read_contact_pairs
    else:
        release_path = args.release_config or DEFAULT_MOTION_DAMPING_RELEASE_CONFIG
        release = load_motion_damping_icl_release(release_path)
        reader = read_damping_pairs
    data_root = resolve_contextworld_path(
        release["data"]["artifact_tree"]["root"], repo_root=ROOT
    )
    counts = release["data"]["pair_counts"]
    pair_count = int(counts[args.split])
    arrays = reader(
        data_root / release["data"]["lance_tables"][args.split],
        expected_pairs=pair_count,
        expected_split=args.split,
    )
    if args.max_pairs is not None:
        if not 0 < args.max_pairs <= pair_count:
            raise ValueError("--max-pairs is outside the selected split")
        pair_count = int(args.max_pairs)
    low_pixels = arrays.low_pixels[:pair_count]
    high_pixels = arrays.high_pixels[:pair_count]
    raw_actions = arrays.raw_action_blocks[:pair_count, :3]
    histories = np.concatenate([low_pixels[:, :3], high_pixels[:, :3]])
    actions = np.concatenate([raw_actions, raw_actions])
    futures = np.concatenate([low_pixels[:, 3], high_pixels[:, 3]])
    normalization = release["evaluation"]["action_normalization"]
    runtime = release["runtime"]["stable_worldmodel"]
    adapter = ADAPTERS[(args.capability, args.model)].from_checkpoint(
        args.checkpoint.expanduser().resolve(),
        action_mean=normalization["mean"],
        action_std=normalization["std_population"],
        repo_root=ROOT,
        stablewm_repo=runtime["repo"],
        stablewm_ref=runtime["expected_ref"],
        device=args.device,
    )
    predictions = adapter.rollout_latents(
        histories, actions, batch_size=args.batch_size
    )[:, 0]
    targets = adapter.encode_pixels(futures, batch_size=args.batch_size)
    geometry = paired_geometry(
        predictions[:pair_count],
        predictions[pair_count:],
        targets[:pair_count],
        targets[pair_count:],
    )
    payload = {
        "schema_version": 1,
        "status": "completed_training_or_development_only",
        "public_test_opened": False,
        "capability": args.capability,
        "model": args.model,
        "split": args.split,
        "pair_count": pair_count,
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "geometry": geometry,
    }
    output = Path(os.path.abspath(args.output.expanduser()))
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": payload["status"], "output": str(output)}))


if __name__ == "__main__":
    main()
