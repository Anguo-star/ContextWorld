#!/usr/bin/env python3
"""Evaluate four existing epoch-5 PLDM checkpoints, or summarize archived Test results."""

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
PLAN = ROOT / "docs/research/data/current_reference_test_v1/plan.json"
METRICS = ROOT / "docs/research/data/current_reference_test_v1/metrics.json"
EPOCH_SUMMARY = ROOT / "docs/research/data/native_epoch_checkpoints_v1/summary.csv"
EPOCH_SOURCES = ROOT / "docs/research/data/native_epoch_checkpoints_v1/sources/epoch5"
TASKS = ("contact_friction", "motion_damping")
REGIMES = ("joint", "frozen")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sources():
    plan = json.loads(PLAN.read_text())
    by_id = {run["id"]: run for run in plan["runs"]}
    rows = list(csv.DictReader(EPOCH_SUMMARY.open()))
    for task in TASKS:
        for regime in REGIMES:
            run_id = f"{task}/pldm/{regime}/s3072"
            baseline = by_id[run_id]
            source_path = EPOCH_SOURCES / task / "pldm" / regime / "result.json"
            source = json.loads(source_path.read_text())
            row = next(r for r in rows if r["task"] == task and r["family"] == "pldm"
                       and r["regime"] == regime and r["epoch"] == "5"
                       and r["split"] == "development")
            checkpoint = Path(source["checkpoint"])
            if (source["registered_checkpoint_sha256"] != baseline["checkpoint_sha256"]
                    or source["registered_checkpoint"] != baseline["checkpoint"]
                    or source["checkpoint_sha256"] != row["checkpoint_sha256"]
                    or sha256(checkpoint) != source["checkpoint_sha256"]):
                raise ValueError(f"endpoint checkpoint mismatch: {run_id}")
            yield task, regime, baseline, source_path, source


def evaluate(device, task_filter=None, regime_filter=None):
    plan = json.loads(PLAN.read_text())
    for task, regime, baseline, _, source in sources():
        if (task_filter is not None and task != task_filter or
                regime_filter is not None and regime != regime_filter):
            continue
        result_path = HERE / "results" / task / regime / "epoch5" / "test.json"
        if result_path.exists():
            report = json.loads(result_path.read_text())
            if (report["evaluation_split"] != "test" or
                    report["result"]["model"]["adapter"]["checkpoint_sha256"] !=
                    source["checkpoint_sha256"]):
                raise ValueError(f"existing result mismatch: {result_path}")
            print(f"reuse {result_path}")
            continue
        result_path.parent.mkdir(parents=True, exist_ok=True)
        runner = ("contextworld.benchmarks.external_model_cli" if task == "motion_damping"
                  else "scripts.run_current_reference_test_compat")
        cmd = [sys.executable, "-m", runner,
               "--task", task, "--adapter", "pldm",
               "--checkpoint", source["checkpoint"],
               "--model-name", f"pldm-{regime}-epoch5-s3072",
               "--training-recipe", regime, "--training-seed", "3072",
               "--benchmark-root", plan["benchmark_root"],
               "--stablewm-repo", baseline["stable_repo"],
               "--stablewm-ref", baseline["stable_ref"],
               "--evaluation-split", "test", "--history-adapter", "native",
               "--prejepa-missing-context-policy", "reject", "--batch-size", "32",
               "--device", device, "--output", str(result_path)]
        subprocess.run(cmd, cwd=ROOT, check=True)


def summarize():
    plan = json.loads(PLAN.read_text())
    metrics = json.loads(METRICS.read_text())
    output = {"schema": "contextworld.figure5-checkpoint-test.v1",
              "evaluation_split": "test", "family": "pldm", "training_seed": 3072,
              "protocol": {"plan": str(PLAN.relative_to(ROOT)), "plan_sha256": sha256(PLAN),
                           "benchmark_root": plan["benchmark_root"],
                           "history_adapter": "native", "batch_size": 32,
                           "runner": {"contact_friction": "scripts.run_current_reference_test_compat",
                                      "motion_damping": "contextworld.benchmarks.external_model_cli"},
                           "epoch_source_summary": str(EPOCH_SUMMARY.relative_to(ROOT)),
                           "epoch_source_summary_sha256": sha256(EPOCH_SUMMARY),
                           "epoch10_metrics": str(METRICS.relative_to(ROOT)),
                           "epoch10_metrics_sha256": sha256(METRICS)},
              "rows": []}
    identity_keys = ("bundle_root", "artifact_root", "manifest_sha256", "task_registry_sha256")
    for task, regime, baseline, source_path, source in sources():
        test5_path = HERE / "results" / task / regime / "epoch5" / "test.json"
        test10_path = ROOT / "docs/research/data/current_reference_test_v1/results" / task / "pldm" / regime / "s3072" / "test.json"
        cell = next(c for c in metrics["cells"] if c["task"] == task and
                    c["family"] == "pldm" and c["regime"] == regime)
        assert cell["n"] == 1 and cell["runs"][0]["checkpoint_sha256"] == baseline["checkpoint_sha256"]
        reports = (json.loads(test5_path.read_text()), json.loads(test10_path.read_text()))
        if sha256(test10_path) != cell["runs"][0]["result_sha256"]:
            raise ValueError(f"epoch10 Test report hash mismatch: {test10_path}")
        bundles = [r["bundle"] for r in reports]
        if tuple(bundles[0][k] for k in identity_keys) != tuple(bundles[1][k] for k in identity_keys):
            raise ValueError(f"epoch5/10 Test data identity mismatch: {task} {regime}")
        for epoch, report, path, expected_sha in ((5, reports[0], test5_path, source["checkpoint_sha256"]),
                                                  (10, reports[1], test10_path, baseline["checkpoint_sha256"])):
            result = report["result"]
            if (report["evaluation_split"] != "test" or result["status"] != "completed" or
                    result["model"]["adapter"]["checkpoint_sha256"] != expected_sha or
                    result["model"]["state_sha256_before"] != result["model"]["state_sha256_after"]):
                raise ValueError(f"invalid Test report: {path}")
            measures = result["metrics"]
            output["rows"].append({
                "task": task, "family": "pldm", "regime": regime, "epoch": epoch,
                "split": "test", "training_seed": 3072,
                "NRE": measures["latent_response"]["normalized_response_error"],
                "nre": measures["latent_response"]["normalized_response_error"],
                "accuracy": 100 * measures["correct_future_rate"],
                "history": 100 * measures["correct_history_rate"],
                "checkpoint": result["model"]["adapter"]["checkpoint"],
                "checkpoint_sha256": expected_sha,
                "source_raw_result": str(path.relative_to(ROOT)),
                "source_raw_result_sha256": sha256(path),
                "epoch5_diagnostic_source": str(source_path.relative_to(ROOT)),
                "epoch5_diagnostic_source_sha256": sha256(source_path),
                "epoch10_metrics_source": str(METRICS.relative_to(ROOT)) if epoch == 10 else None,
                "test_bundle": {key: report["bundle"][key] for key in identity_keys},
                "test_payload_sha256_verified": report["bundle"].get("test_payload_sha256_verified"),
            })
    (HERE / "summary.json").write_text(json.dumps(output, indent=2) + "\n")
    print(f"wrote {HERE / 'summary.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--task", choices=TASKS)
    parser.add_argument("--regime", choices=REGIMES)
    args = parser.parse_args()
    summarize() if args.summarize_only else evaluate(args.device, args.task, args.regime)
