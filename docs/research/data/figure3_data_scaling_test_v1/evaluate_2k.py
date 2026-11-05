#!/usr/bin/env python3
"""Reproduce the four existing 2k LeWM checkpoint evaluations on frozen Test."""

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent / "results"
PLAN = ROOT / "docs/research/data/current_reference_test_v1/plan.json"
STUDY = ROOT / "docs/research/data/icl_training_study_v2.json"
METRICS = ROOT / "docs/research/data/current_reference_test_v1/metrics.json"
CHECKPOINT_ROOT = Path("/opt/huawei/explorer-env/dataset/ag_data/ckpt/lewm-contextworld-v1/checkpoints")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def find_study(node, identifier):
    if isinstance(node, dict):
        if node.get("id") == identifier:
            return node
        children = node.values()
    elif isinstance(node, list):
        children = node
    else:
        return None
    for child in children:
        found = find_study(child, identifier)
        if found is not None:
            return found
    return None


def summarize():
    plan = json.loads(PLAN.read_text())
    study = json.loads(STUDY.read_text())
    reference_metrics = json.loads(METRICS.read_text())
    output = {
        "schema": "contextworld.figure3-data-scaling-test-metrics.v1",
        "evaluation_split": "test", "task": "action_strength", "family": "lewm",
        "protocol": {
            "source_plan": str(PLAN.relative_to(ROOT)),
            "source_plan_sha256": sha256(PLAN),
            "benchmark_root": plan["benchmark_root"],
            "benchmark_task_registry_sha256": plan["pinned_files"]["bundle/task_registry.json"],
            "benchmark_manifest_sha256": plan["pinned_files"]["bundle/manifest.jsonl"],
            "history_adapter": "native", "batch_size": 32,
            "evaluation_split": "test",
            "runner": "scripts.run_current_reference_test_compat",
            "training_study": str(STUDY.relative_to(ROOT)),
            "training_study_sha256": sha256(STUDY),
        },
        "cells": [],
    }
    identity_keys = ("bundle_root", "artifact_root", "manifest_sha256", "task_registry_sha256")
    reference_identity = None
    for regime in ("scratch", "joint"):
        source = find_study(study, f"action_strength/lewm/{regime}/scale_2k")
        runs = []
        for rep in source["replicates"]:
            seed = rep["training_seed"]
            report_path = OUT / f"{regime}_s{seed}" / "test.json"
            report = json.loads(report_path.read_text())
            result = report["result"]
            adapter = result["model"]["adapter"]
            bundle = report["bundle"]
            if (report["evaluation_split"] != "test" or result["status"] != "completed" or
                    adapter["checkpoint_sha256"] != rep["checkpoint_sha256"] or
                    result["model"]["state_sha256_before"] != result["model"]["state_sha256_after"] or
                    not bundle.get("test_payload_sha256_verified")):
                raise ValueError(f"invalid 2k Test report: {report_path}")
            identity = tuple(bundle[key] for key in identity_keys)
            if reference_identity is None:
                reference_identity = identity
            elif identity != reference_identity:
                raise ValueError(f"Test bundle differs: {report_path}")
            measures = result["metrics"]
            runs.append({
                "id": f"{source['id']}/s{seed}", "seed": seed,
                "checkpoint": adapter["checkpoint"],
                "checkpoint_sha256": adapter["checkpoint_sha256"],
                "result": str(report_path.relative_to(ROOT)),
                "result_sha256": sha256(report_path),
                "accuracy": 100 * measures["correct_future_rate"],
                "history": 100 * measures["correct_history_rate"],
                "nre": measures["latent_response"]["normalized_response_error"],
                "stable_repo": adapter["stable_worldmodel_repo"],
                "stable_ref": adapter["stable_worldmodel_commit"],
                "bundle_manifest_sha256": bundle["manifest_sha256"],
                "bundle_task_registry_sha256": bundle["task_registry_sha256"],
                "test_payload_bytes": bundle["test_payload_bytes"],
            })
        def score(values):
            return {"mean": statistics.mean(values),
                    "std": statistics.stdev(values) if len(values) > 1 else None,
                    "n": len(values)}
        output["cells"].append({
            "task": "action_strength", "family": "lewm", "regime": regime,
            "scheme": "Mixed" if regime == "scratch" else "Original -> Mixed (Joint)",
            "training_pair_count": 2048, "n": len(runs),
            "scores": {key: score([run[key] for run in runs])
                       for key in ("accuracy", "history", "nre")},
            "runs": runs, "source_metrics": None, "source_metrics_sha256": None,
        })
        reference_cell = next(cell for cell in reference_metrics["cells"]
                              if cell["task"] == "action_strength" and
                              cell["family"] == "lewm" and cell["regime"] == regime)
        source_runs = []
        for run in reference_cell["runs"]:
            source_report = ROOT / run["result"]
            if sha256(source_report) != run["result_sha256"]:
                raise ValueError(f"32k result hash differs: {source_report}")
            archived = json.loads(source_report.read_text())
            if tuple(archived["bundle"][key] for key in identity_keys) != reference_identity:
                raise ValueError(f"32k Test bundle differs: {source_report}")
            source_runs.append({
                "id": run["id"], "seed": run["seed"],
                "checkpoint_sha256": run["checkpoint_sha256"],
                "result": run["result"], "result_sha256": run["result_sha256"],
                "accuracy": run["accuracy"], "history": run["history"],
                "nre": run["nre"],
            })
        output["cells"].append({
            "task": "action_strength", "family": "lewm", "regime": regime,
            "scheme": "Mixed" if regime == "scratch" else "Original -> Mixed (Joint)",
            "training_pair_count": 32768, "n": reference_cell["n"],
            "scores": reference_cell["scores"], "runs": source_runs,
            "source_metrics": str(METRICS.relative_to(ROOT)),
            "source_metrics_sha256": sha256(METRICS),
        })
    destination = OUT.parent / "metrics.json"
    destination.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {destination}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    if args.summarize_only:
        summarize()
        return
    plan = json.loads(PLAN.read_text())
    study = json.loads(STUDY.read_text())
    reference = {run["id"]: run for run in plan["runs"]}
    for regime, seeds in (("scratch", (3072, 3073, 3074)), ("joint", (3072,))):
        source = find_study(study, f"action_strength/lewm/{regime}/scale_2k")
        replicates = {rep["training_seed"]: rep for rep in source["replicates"]}
        baseline = reference[f"action_strength/lewm/{regime}/s3072"]
        for seed in seeds:
            name = (f"action_strength_lewm_joint_scratch_v1_s{seed}"
                    if regime == "scratch" else
                    "action_strength_lewm_init_form_pusht_lewm_baseline_seed3073_weights_s3072")
            checkpoint = CHECKPOINT_ROOT / name / "weights_epoch_10.pt"
            if sha256(checkpoint) != replicates[seed]["checkpoint_sha256"]:
                raise ValueError(f"checkpoint identity mismatch: {regime} {seed}")
            output = OUT / f"{regime}_s{seed}" / "test.json"
            if output.exists():
                report = json.loads(output.read_text())
                if (report["evaluation_split"] != "test" or
                        report["result"]["model"]["adapter"]["checkpoint_sha256"] !=
                        replicates[seed]["checkpoint_sha256"]):
                    raise ValueError(f"existing result identity mismatch: {output}")
                print(f"reuse {output}")
                continue
            output.parent.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, "-m", "scripts.run_current_reference_test_compat",
                       "--task", "action_strength", "--adapter", "lewm",
                       "--checkpoint", str(checkpoint),
                       "--model-name", f"lewm-{regime}-scale2k-s{seed}",
                       "--training-recipe", regime, "--training-seed", str(seed),
                       "--benchmark-root", plan["benchmark_root"],
                       "--stablewm-repo", baseline["stable_repo"],
                       "--stablewm-ref", baseline["stable_ref"],
                       "--evaluation-split", "test", "--history-adapter", "native",
                       "--prejepa-missing-context-policy", "reject",
                       "--batch-size", "32", "--device", args.device,
                       "--output", str(output)]
            subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
