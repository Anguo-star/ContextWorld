#!/usr/bin/env python3
"""Audit strict action-strength scores against the prior model-visible run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STRICT_RESULTS = ROOT / (
    "artifacts/evaluation/history3/"
    "pusht_action_strength_h3_strict_results_v1"
)
DEFAULT_LEGACY_RESULTS = Path(
    "/opt/huawei/explorer-env/dataset/ag_data/data/world_model/"
    "context_world/evaluation/history3/"
    "pusht_hidden_actuation_replay_matched_h3_v2"
)
SEEDS = (13313, 13314, 13315)
PUBLIC_TEST_MANIFEST_SHA256 = (
    "5abb504ea57f9525944fdbc75c418cfb"
    "a2bc4afbe1a8f0b3fea2c489b3f62092"
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def legacy_paths(root: Path, seed: int) -> dict[str, Path]:
    run = root / (
        f"training_seed{seed}_step4096_"
        "exposure_matched_standard64_hidden64"
    )
    if seed == 13313:
        planning = run / "confirmatory_v4_hidden_cem_seed13312/aggregate.json"
        prediction = run / "public_test_prediction_v1.json"
        retention = run / "standard_pusht_cem/aggregate.json"
    elif seed == 13314:
        planning = run / (
            "public_test_action_strength_planning_v1/aggregate.json"
        )
        prediction = run / "public_test_prediction_v1.json"
        retention = run / (
            "public_test_standard_pusht_retention_v1_retry1/aggregate.json"
        )
    else:
        planning = run / (
            "public_test_action_strength_planning_v1/aggregate.json"
        )
        prediction = run / "public_test_prediction_v1.json"
        retention = run / (
            "public_test_standard_pusht_retention_v1/aggregate.json"
        )
    return {
        "planning": planning,
        "prediction": prediction,
        "retention": retention,
    }


def planning_receipt(
    *,
    seed: int,
    legacy: Path,
    strict: Path,
    score: Path,
) -> dict[str, Any]:
    old = json.loads(legacy.read_text(encoding="utf-8"))
    new = json.loads(strict.read_text(encoding="utf-8"))
    scored = json.loads(score.read_text(encoding="utf-8"))
    old_records = old["records"]
    new_records = new["records"]
    old_ids = [str(row["condition_id"]) for row in old_records]
    new_ids = [str(row["condition_id"]) for row in new_records]
    old_amplitude = np.asarray(
        [row["execution"]["amplitude"] for row in old_records],
        dtype=np.float64,
    )
    new_amplitude = np.asarray(
        [row["execution"]["amplitude"] for row in new_records],
        dtype=np.float64,
    )
    old_cost = np.asarray(
        [row["selected_predicted_cost"] for row in old_records],
        dtype=np.float64,
    )
    new_cost = np.asarray(
        [row["selected_predicted_cost"] for row in new_records],
        dtype=np.float64,
    )
    old_execution_distance = np.asarray(
        [
            row["execution"]["visible_state_distance"]
            for row in old_records
        ],
        dtype=np.float64,
    )
    new_execution_distance = np.asarray(
        [
            row["execution"]["visible_state_distance"]
            for row in new_records
        ],
        dtype=np.float64,
    )
    maximum_execution_distance_change = float(
        np.max(
            np.abs(old_execution_distance - new_execution_distance)
        )
    )
    strict_execution = all(
        row["execution"].get("state_installations_after_x0") == 0
        and row["execution"].get("query_simulator_recreated") is False
        for row in new_records
    )
    checks = {
        "checkpoint_receipt_identical": old["checkpoint"] == new["checkpoint"],
        "cem_protocol_identical": old["cem"] == new["cem"],
        "condition_ids_identical": old_ids == new_ids,
        "selected_amplitudes_bitwise_identical": np.array_equal(
            old_amplitude, new_amplitude
        ),
        "selected_model_costs_bitwise_identical": np.array_equal(
            old_cost, new_cost
        ),
        "strict_execution_change_below_1e_4_px": (
            maximum_execution_distance_change <= 1.0e-4
        ),
        "strict_execution_replays_natural_prefix": strict_execution,
        "all_512_conditions_scored": (
            len(new_records) == 512
            and scored["summary"]["condition_count"] == 512
        ),
        "action_region_gate_passed": scored["gate"]["passed"] is True,
        "scored_submission_is_strict_result": (
            scored["submission"]["sha256"] == file_sha256(strict)
        ),
    }
    return {
        "training_seed": seed,
        "legacy_result": {
            "path": str(legacy),
            "sha256": file_sha256(legacy),
        },
        "strict_result": {
            "path": str(strict),
            "sha256": file_sha256(strict),
        },
        "strict_score": {
            "path": str(score),
            "sha256": file_sha256(score),
            "correct_action_region_rate": scored["summary"][
                "correct_action_region_rate"
            ],
        },
        "condition_ids_sha256": canonical_sha256(new_ids),
        "selected_amplitudes_sha256": canonical_sha256(
            new_amplitude.tolist()
        ),
        "selected_model_costs_sha256": canonical_sha256(new_cost.tolist()),
        "maximum_selected_amplitude_difference": float(
            np.max(np.abs(old_amplitude - new_amplitude))
        ),
        "maximum_selected_model_cost_difference": float(
            np.max(np.abs(old_cost - new_cost))
        ),
        "maximum_executed_visible_state_distance_change_px": (
            maximum_execution_distance_change
        ),
        "mean_executed_visible_state_distance_change_px": float(
            np.mean(new_execution_distance - old_execution_distance)
        ),
        "checks": checks,
        "passed": all(checks.values()),
    }


def prediction_receipt(
    *, seed: int, legacy: Path, strict: Path
) -> dict[str, Any]:
    old = json.loads(legacy.read_text(encoding="utf-8"))
    new = json.loads(strict.read_text(encoding="utf-8"))
    checks = {
        "checkpoint_hash_identical": (
            old["model"]["adapter"]["checkpoint_sha256"]
            == new["model"]["adapter"]["checkpoint_sha256"]
        ),
        "metrics_identical": old["metrics"] == new["metrics"],
        "pair_records_identical": old["records"] == new["records"],
        "strict_manifest_bound": (
            new["release"]["confirmation_manifest_sha256"]
            == PUBLIC_TEST_MANIFEST_SHA256
        ),
        "gate_passed": new["gate"]["passed"] is True,
    }
    return {
        "training_seed": seed,
        "legacy_result_sha256": file_sha256(legacy),
        "strict_result_sha256": file_sha256(strict),
        "metrics": new["metrics"],
        "checks": checks,
        "passed": all(checks.values()),
    }


def retention_receipt(
    *, seed: int, report: Path, score: Path
) -> dict[str, Any]:
    source = json.loads(report.read_text(encoding="utf-8"))
    scored = json.loads(score.read_text(encoding="utf-8"))
    model_hashes = {
        str(row["checkpoint_sha256"])
        for row in source.get("models", [])
    }
    expected_hash = str(scored["model"]["checkpoint_sha256"])
    checks = {
        "checkpoint_hash_found_in_frozen_report": expected_hash in model_hashes,
        "query_catalog_hash_frozen": (
            source["query_catalog"]["sha256"]
            == "da974c821e3fa0f232ac59538e0c79cc2cb9a80ea4bb4a62b910c66c278ce2d4"
        ),
        "exact_300_episode_protocol": (
            source["protocol"]["num_eval_per_seed"] == 50
            and source["protocol"]["eval_seeds"]
            == [42, 43, 44, 45, 46, 47]
            and scored["score"]["evaluations"] == 300
        ),
        "all_protocol_checks_passed": all(
            scored["protocol_checks"].values()
        ),
        "retention_gate_passed": scored["gate"]["passed"] is True,
        "score_binds_frozen_report_hash": (
            scored["report"]["sha256"] == file_sha256(report)
        ),
    }
    return {
        "training_seed": seed,
        "reuse_reason": (
            "The strict repair changes only Action Strength synthetic data "
            "and its Public Test. This frozen standard PushT CEM report reads "
            "neither artifact; its checkpoint, standard replay protocol, "
            "query catalog, and report bytes are unchanged."
        ),
        "frozen_report": {
            "path": str(report),
            "sha256": file_sha256(report),
        },
        "strict_release_rescore": {
            "path": str(score),
            "sha256": file_sha256(score),
            "successes": scored["score"]["successes"],
            "evaluations": scored["score"]["evaluations"],
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict-results", type=Path, default=DEFAULT_STRICT_RESULTS
    )
    parser.add_argument(
        "--legacy-results", type=Path, default=DEFAULT_LEGACY_RESULTS
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    strict_root = args.strict_results.expanduser().resolve()
    legacy_root = args.legacy_results.expanduser().resolve()
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else strict_root / "strict_result_compatibility_audit.json"
    )
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")

    old_oracle = legacy_root / (
        "training_seed13313_step4096_exposure_matched_standard64_hidden64/"
        "confirmatory_v4_hidden_cem_seed13312/oracle_surface.json"
    )
    new_oracle = strict_root / (
        "planning_oracle_strict_v1/oracle_surface.json"
    )
    old_oracle_rows = json.loads(old_oracle.read_text(encoding="utf-8"))
    new_oracle_rows = json.loads(new_oracle.read_text(encoding="utf-8"))
    oracle_ids_equal = [row["condition_id"] for row in old_oracle_rows] == [
        row["condition_id"] for row in new_oracle_rows
    ]
    oracle_amplitudes_equal = [
        row["best"]["amplitude"] for row in old_oracle_rows
    ] == [row["best"]["amplitude"] for row in new_oracle_rows]
    oracle_receipt = {
        "legacy_path": str(old_oracle),
        "legacy_sha256": file_sha256(old_oracle),
        "strict_path": str(new_oracle),
        "strict_sha256": file_sha256(new_oracle),
        "condition_count": len(new_oracle_rows),
        "condition_ids_identical": oracle_ids_equal,
        "best_amplitudes_identical": oracle_amplitudes_equal,
        "passed": bool(
            len(new_oracle_rows) == 512
            and oracle_ids_equal
            and oracle_amplitudes_equal
        ),
    }

    planning = []
    prediction = []
    retention = []
    for seed in SEEDS:
        old = legacy_paths(legacy_root, seed)
        planning.append(
            planning_receipt(
                seed=seed,
                legacy=old["planning"],
                strict=strict_root / f"planning_seed{seed}/aggregate.json",
                score=strict_root / f"planning_score_seed{seed}.json",
            )
        )
        prediction.append(
            prediction_receipt(
                seed=seed,
                legacy=old["prediction"],
                strict=strict_root / f"prediction_seed{seed}.json",
            )
        )
        retention.append(
            retention_receipt(
                seed=seed,
                report=old["retention"],
                score=strict_root / f"retention_score_seed{seed}.json",
            )
        )

    report = {
        "schema_version": 1,
        "status": "strict_action_strength_result_audit",
        "resource_note": (
            "The three strict prediction and CEM evaluations briefly shared "
            "GPU 0/1/2 with pre-existing workloads; no exclusive GPU claim "
            "is made. Every process returned exit code 0 and emitted a full "
            "result before this audit was run."
        ),
        "oracle_compatibility": oracle_receipt,
        "prediction_three_seed": prediction,
        "planning_three_seed": planning,
        "standard_pusht_cem_reuse": retention,
    }
    report["passed"] = all(
        (
            oracle_receipt["passed"],
            all(row["passed"] for row in prediction),
            all(row["passed"] for row in planning),
            all(row["passed"] for row in retention),
        )
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "passed": report["passed"],
                "prediction": [row["passed"] for row in prediction],
                "planning": [row["passed"] for row in planning],
                "retention": [row["passed"] for row in retention],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
