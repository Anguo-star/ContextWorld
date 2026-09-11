"""Freeze or verify the supplemental original-checkpoint inference results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.eval_dinowm_original_icl_current import (
    ADAPTER_SPEC, METHOD_ID, FREEZE, SOURCE, jobs, read, save, sha, validate_result,
)
from contextworld.benchmarks.reference_decision import reference_decision_for_result

DESTINATION = ROOT / "configs/benchmark/contextworld_dinowm_original_fixed_context_results_v1.json"


def identity(path: Path) -> dict:
    return {"path": str(path.relative_to(ROOT)), "sha256": sha(path)}


def frozen_state_digest(result: dict) -> str:
    model = result["model"]
    audit = result.get("score_audit", {})
    frozen = result.get("frozen_weight_audit", {})
    before = model.get("state_sha256_before", model.get("state_hash_before", audit.get("frozen_state_hash_before", frozen.get("state_hash_before"))))
    after = model.get("state_sha256_after", model.get("state_hash_after", audit.get("frozen_state_hash_after", frozen.get("state_hash_after"))))
    if not before or before != after:
        raise ValueError("Model state absent or changed during evaluation")
    return before


def build() -> dict:
    baseline = read(FREEZE)
    sources = baseline["inputs"]["evaluation_sources"]
    for source in sources:
        if sha(ROOT / source["path"]) != source["sha256"]:
            raise ValueError(f"Frozen evaluation source changed: {source['path']}")
    rows = {}
    for job in jobs():
        payload, score = validate_result(job)
        if payload["task"] != job["task"]:
            raise ValueError("Task identity mismatch")
        result = payload["result"]
        model = result["model"]
        adapter = model.get("adapter", model)
        if model["training_seed"] != job["seed"]:
            raise ValueError("Training seed mismatch")
        before = frozen_state_digest(result)
        if adapter["diagnostic"] is not True or adapter["frozen_v1_compatible"] is not False:
            raise ValueError("Diagnostic inference boundary lost")
        row = rows.setdefault((job["task"], job["seed"]), {
            "component_id": job["task"], "family": "DINO-WM",
            "stage": "original_environment_only", "training_seed": job["seed"],
            "environment": job["environment"],
            "checkpoint": "/".join(Path(job["checkpoint"]).parts[-2:]),
            "checkpoint_sha256": job["checkpoint_sha256"],
            "history_adapter": adapter["history_adapter"],
            "normalized_zero_state_streams": adapter["normalized_zero_state_streams"],
        })
        decision = reference_decision_for_result(job["task"], payload, split=job["split"], repo_root=ROOT)
        if decision["threshold_sources"] != baseline["inputs"]["decision_contract"]:
            raise ValueError("Decision contract differs from v3")
        row[job["split"]] = {
            "main_score": score, "all_gates_passed": decision["passed"],
            "reason_codes": decision["reason_codes"],
            "source_result_path": str(Path(job["output"]).relative_to(ROOT)),
            "source_result_sha256": sha(Path(job["output"])),
            "state_sha256": before,
            "bundle": {k: v for k, v in payload.get("bundle", result.get("bundle", {})).items() if k != "bundle_root"},
            "selection": result.get("selection", payload.get("selection_policy")),
            "result_kind": payload["result_kind"],
        }
    assert len(rows) == 27
    return {
        "schema_version": "contextworld.dinowm-original-fixed-context-results.v1",
        "status": "completed_supplemental_inference_variant",
        "inference_variant": METHOD_ID, "adapter_spec": ADAPTER_SPEC,
        "claim_boundary": {
            "diagnostic": True, "native_checkpoint_input_compatible": False,
            "official_scoreboard_row": False, "data_only_ablation": False,
            "public_inputs": ["pixels", "action"],
            "missing_context_policy": "fixed_model_normalized_zero",
            "checkpoint_weights_modified": False,
            "note": "Task scores use v3 kernels and splits; original state streams are fixed to normalized zero for ICL. CEM reuses native-input runs of the same weights.",
        },
        "inputs": {
            "base_freeze": identity(FREEZE), "original_checkpoint_and_cem_source": identity(SOURCE),
            "adapter_source": identity(ROOT / "contextworld/benchmarks/prejepa_original_context_adapter.py"),
            "evaluation_runner": identity(ROOT / "scripts/eval_dinowm_original_icl_current.py"),
            "freeze_builder": identity(Path(__file__).resolve()),
        },
        "coverage": {"distinct_checkpoints": 12, "checkpoint_task_rows": 27,
                     "development": 27, "test": 21, "test_excluded": ["contact_friction", "motion_damping"]},
        "checkpoint_results": list(rows.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "verify"))
    args = parser.parse_args()
    expected = build()
    if args.action == "verify":
        if read(DESTINATION) != expected:
            raise ValueError("Supplement differs from its raw evaluation evidence")
        print("Verified 48 scores, unchanged weights, v3 scorers and supplemental boundaries")
    else:
        if DESTINATION.exists() and read(DESTINATION) != expected:
            raise ValueError("Existing supplement differs; create a new version instead of overwriting")
        save(DESTINATION, expected)
        print(DESTINATION)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
