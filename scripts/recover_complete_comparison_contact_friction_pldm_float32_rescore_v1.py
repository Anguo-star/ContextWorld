#!/usr/bin/env python3
"""Float32 rescore recovery for the three Contact Friction PLDM receipts.

The frozen generic rescorer (``score_contact_friction_icl_results``) rebuilds
the stored per-record MSE values with NumPy's implicit float64 and therefore
reports a ~4e-9..7e-9 drift on ``other_minus_correct_mse_margin_mean`` for
receipts whose evaluator aggregated in float32.  This mirrors the already
frozen Action Strength recovery
(``pusht_action_strength_pldm_float32_rescore_recovery_v1``).

This launcher:
  * verifies the recovery preregistration, release config, scorer, and the
    three raw receipts byte-for-byte against their frozen identities;
  * recomputes every rate metric in float64 exactly as the frozen rescorer
    does, and the three MSE aggregates in float32 exactly as the frozen
    evaluator does;
  * requires every recomputed value to match the stored receipt within the
    frozen tolerance (rtol=1e-7, atol=1e-9) after the dtype correction;
  * recomputes each gate with the frozen ``contact_friction_prediction_gate``
    and requires equality with the stored gate;
  * writes one additive, x-exclusive method aggregate laid out identically to
    ``score_contact_friction_icl_results``.

It never loads a model, never rewrites a receipt, and never opens Public data.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import statistics
import sys
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contextworld.benchmarks.contact_friction_icl_data import (  # noqa: E402
    load_contact_friction_icl_release,
)
from contextworld.benchmarks.contact_friction_icl_score import (  # noqa: E402
    contact_friction_prediction_gate,
)
from contextworld.benchmarks.paired_latent_response import (  # noqa: E402
    paired_latent_response_summaries_close,
    summarize_paired_latent_response_records,
)

PREREGISTRATION = (
    ROOT
    / "configs/benchmark/"
    "complete_comparison_contact_friction_pldm_float32_rescore_recovery_v1.yaml"
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_identity(row: dict[str, Any], *, label: str) -> Path:
    path = (ROOT / row["path"]).resolve()
    observed = file_sha256(path)
    if observed != row["sha256"]:
        raise RuntimeError(
            f"{label} drifted: expected {row['sha256']}, got {observed}"
        )
    if "size_bytes" in row and path.stat().st_size != int(row["size_bytes"]):
        raise RuntimeError(f"{label} size drifted")
    return path


def _close(a: float, b: float) -> bool:
    return bool(np.isclose(a, b, rtol=1e-7, atol=1e-9))


def _recover_receipt(path: Path, *, release: dict[str, Any]) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema_version") != 1
        or payload.get("benchmark") != "pusht_history3_contact_friction_icl_v1"
        or payload.get("submission_kind") != "single_checkpoint"
        or payload.get("status") != "completed"
    ):
        raise ValueError(f"Unsupported Contact Friction result: {path}")
    expected_release = {
        "release_id": release["release_id"],
        "release_config_sha256": file_sha256(Path(release["_config_path"])),
        "data_manifest_sha256": release["data"]["manifest_sha256"],
        "sealed_test_included": False,
    }
    if payload.get("release") != expected_release:
        raise RuntimeError(f"Release identity mismatch: {path}")
    records = payload["records"]
    expected_pairs = int(release["evaluation"]["pair_count"])
    if not isinstance(records, list) or len(records) != expected_pairs:
        raise ValueError(f"Recovery requires all {expected_pairs} records: {path}")

    def booleans(mode: str, field: str) -> np.ndarray:
        return np.asarray([row[mode][field] for row in records], dtype=bool)

    low_future = booleans("low_friction", "correct_future")
    high_future = booleans("high_friction", "correct_future")
    low_history = booleans("low_friction", "correct_history")
    high_history = booleans("high_friction", "correct_history")
    switch = np.asarray(
        [row["context_switch_correct"] for row in records], dtype=bool
    )
    # float32 is the single deliberate difference from the frozen generic
    # rescorer: it matches the evaluator's aggregation dtype.
    correct_losses = np.asarray(
        [row["low_friction"]["correct_future_mse"] for row in records]
        + [row["high_friction"]["correct_future_mse"] for row in records],
        dtype=np.float32,
    )
    other_losses = np.asarray(
        [row["low_friction"]["other_future_mse"] for row in records]
        + [row["high_friction"]["other_future_mse"] for row in records],
        dtype=np.float32,
    )
    metrics: dict[str, Any] = {
        "pair_count": len(records),
        "decision_count": 2 * len(records),
        "correct_future_rate": float(
            np.concatenate([low_future, high_future]).mean()
        ),
        "correct_history_rate": float(
            np.concatenate([low_history, high_history]).mean()
        ),
        "context_switch_rate": float(switch.mean()),
        "low_friction_correct_future_rate": float(low_future.mean()),
        "high_friction_correct_future_rate": float(high_future.mean()),
        "worst_friction_correct_future_rate": float(
            min(low_future.mean(), high_future.mean())
        ),
        "correct_future_mse_mean": float(correct_losses.mean()),
        "other_future_mse_mean": float(other_losses.mean()),
        "other_minus_correct_mse_margin_mean": float(
            (other_losses - correct_losses).mean()
        ),
        "current_frame_only_accuracy_bound": 0.5,
    }
    if not all(isinstance(row.get("latent_response"), dict) for row in records):
        raise RuntimeError(f"Incomplete latent response records: {path}")
    metrics["latent_response"] = summarize_paired_latent_response_records(
        [{"pair_id": row["pair_id"], **row["latent_response"]} for row in records]
    )
    calibrated = np.asarray(
        [row["latent_response"]["calibrated_response_success"] for row in records],
        dtype=bool,
    )
    joint = low_future & high_future & low_history & high_history & calibrated
    if not all(
        row.get("joint_icl_pair_success") == bool(joint[index])
        for index, row in enumerate(records)
    ):
        raise RuntimeError(f"Invalid joint ICL records: {path}")
    metrics["joint_icl_pair_success_rate"] = float(joint.mean())

    stored = payload["metrics"]
    scalar_names = set(metrics) - {"latent_response"}
    if set(stored) != set(metrics):
        raise RuntimeError(f"Metric name set drifted: {path}")
    mismatched = [
        name
        for name in sorted(scalar_names)
        if not (
            metrics[name] == stored[name]
            if isinstance(metrics[name], int)
            else _close(metrics[name], stored[name])
        )
    ]
    if mismatched or not paired_latent_response_summaries_close(
        stored["latent_response"], metrics["latent_response"]
    ):
        raise RuntimeError(
            f"Stored score does not match float32 reconstruction: {path} "
            f"({mismatched})"
        )
    gate = contact_friction_prediction_gate(metrics, release=release)
    if gate != payload.get("gate"):
        raise RuntimeError(f"Stored gate drifted from frozen gate rule: {path}")
    return payload


def _stats(values: list[float]) -> dict[str, float]:
    rows = [float(value) for value in values]
    return {
        "mean": float(statistics.fmean(rows)),
        "sample_std": float(statistics.stdev(rows)) if len(rows) > 1 else 0.0,
        "minimum": float(min(rows)),
        "maximum": float(max(rows)),
    }


def main() -> None:
    prereg = yaml.safe_load(PREREGISTRATION.read_text(encoding="utf-8"))
    if prereg["recovery_id"] != (
        "complete_comparison_contact_friction_pldm_float32_rescore_recovery_v1"
    ):
        raise RuntimeError("Unexpected recovery preregistration")
    for key, label in (
        ("release_config", "frozen release config"),
        ("frozen_icl_scorer", "frozen ICL scorer"),
        ("paired_latent_metric", "frozen paired latent metric"),
    ):
        _assert_identity(prereg["frozen_inputs"][key], label=label)
    _assert_identity(prereg["precedent"]["preregistration"], label="precedent recovery")

    release = load_contact_friction_icl_release()
    checkpoints_spec = prereg["raw_public_icl"]["checkpoints"]
    paths: list[Path] = []
    results: list[dict[str, Any]] = []
    for row in checkpoints_spec:
        receipt_path = _assert_identity(
            row["raw_result"], label=f"raw receipt seed {row['seed']}"
        )
        payload = _recover_receipt(receipt_path, release=release)
        if payload["model"]["training_seed"] != int(row["seed"]):
            raise RuntimeError(f"Seed mismatch in {receipt_path}")
        if (
            payload["model"]["adapter"]["checkpoint_sha256"]
            != row["checkpoint_sha256"]
        ):
            raise RuntimeError(f"Checkpoint hash mismatch in {receipt_path}")
        paths.append(receipt_path)
        results.append(payload)

    hashes = [r["model"]["adapter"]["checkpoint_sha256"] for r in results]
    seeds = [int(r["model"]["training_seed"]) for r in results]
    recipes = {str(r["model"]["training_recipe"]) for r in results}
    if len(set(hashes)) != 3 or len(set(seeds)) != 3 or len(recipes) != 1:
        raise RuntimeError("Recovery requires three distinct seeds, one recipe")

    metric_names = (
        "correct_future_rate",
        "correct_history_rate",
        "context_switch_rate",
        "worst_friction_correct_future_rate",
        "joint_icl_pair_success_rate",
    )
    checkpoints = [
        {
            "path": str(path),
            "checkpoint_sha256": result["model"]["adapter"]["checkpoint_sha256"],
            "training_seed": result["model"]["training_seed"],
            **{name: result["metrics"][name] for name in metric_names},
            "passed": bool(result["gate"]["passed"]),
        }
        for path, result in zip(paths, results, strict=True)
    ]
    passed = all(row["passed"] for row in checkpoints)
    aggregate = {
        "schema_version": 1,
        "benchmark": "pusht_history3_contact_friction_icl_v1",
        "submission_kind": "three_seed_method",
        "status": "completed",
        "method_name": str(prereg["raw_public_icl"]["method_name"]),
        "release_id": release["release_id"],
        "checkpoints": checkpoints,
        "aggregate": {
            metric: _stats([row[metric] for row in checkpoints])
            for metric in metric_names
        },
        "decision": {
            "passed": passed,
            "formal_method_claim": True,
            "reason": (
                "all_three_training_seeds_passed"
                if passed
                else "one_or_more_training_seeds_failed"
            ),
        },
        "recovery": {
            "recovery_id": prereg["recovery_id"],
            "preregistration": {
                "path": str(PREREGISTRATION.relative_to(ROOT)),
                "sha256": file_sha256(PREREGISTRATION),
            },
            "reason": (
                "frozen_generic_rescorer_rebuilds_float32_mse_aggregates_in_"
                "float64"
            ),
            "dtype_correction": "mse_aggregates_recomputed_in_float32",
            "all_other_metrics_recomputed_in_float64_and_matched": True,
            "gates_recomputed_with_frozen_rule_and_matched": True,
            "model_evaluation_rerun": False,
            "raw_receipts_rewritten": False,
        },
    }

    output = ROOT / prereg["outputs"]["aggregate"]["path"]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(aggregate, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": file_sha256(output),
                "passed": passed,
                "seeds": seeds,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
