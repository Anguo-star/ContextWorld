#!/usr/bin/env python3
"""Reproduce four existing 10x capacity Test runs and summarize with small baselines."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
PLAN = ROOT / "docs/research/data/current_reference_test_v1/plan.json"
METRICS = ROOT / "docs/research/data/current_reference_test_v1/metrics.json"
DEV = ROOT / "docs/research/data/model_capacity_scaling_v1/final_comparison.json"
CURVES = Path("/opt/huawei/explorer-env/dataset/ag_data/code/stable-worldmodel/research/conditional_dynamics_representation/artifacts/scaling10x_epoch_curves_v1")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def entries():
    plan = json.loads(PLAN.read_text())
    runs = {r["id"]: r for r in plan["runs"]}
    for task in ("contact_friction", "motion_damping"):
        for model in ("lewm", "pldm"):
            source = CURVES / f"sources/{task}/{model}/scaling10x/epoch10/result.json"
            small_source = CURVES / f"sources/{task}/{model}/small/epoch10/result.json"
            endpoint = json.loads(source.read_text())
            small_endpoint = json.loads(small_source.read_text())
            baseline = runs[f"{task}/{model}/scratch/s3072"]
            if (endpoint["checkpoint_sha256"] != sha256(Path(endpoint["checkpoint"])) or
                    small_endpoint["checkpoint_sha256"] != baseline["checkpoint_sha256"]):
                raise ValueError(f"capacity endpoint identity mismatch: {task} {model}")
            yield task, model, baseline, source, endpoint


def evaluate(device):
    plan = json.loads(PLAN.read_text())
    for task, model, baseline, _, endpoint in entries():
        out = HERE / f"results/{task}/{model}/test.json"
        if out.exists():
            report = json.loads(out.read_text())
            if report["result"]["model"]["adapter"]["checkpoint_sha256"] != endpoint["checkpoint_sha256"]:
                raise ValueError(f"saved result mismatch: {out}")
            print(f"reuse {out}")
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        runner = ("contextworld.benchmarks.external_model_cli" if task == "motion_damping"
                  else "scripts.run_current_reference_test_compat")
        cmd = [sys.executable, "-m", runner, "--task", task, "--adapter", model,
               "--checkpoint", endpoint["checkpoint"],
               "--model-name", f"{model}-scaling10x-s3072", "--training-recipe", "scratch",
               "--training-seed", "3072", "--benchmark-root", plan["benchmark_root"],
               "--stablewm-repo", baseline["stable_repo"],
               "--stablewm-ref", baseline["stable_ref"],
               "--evaluation-split", "test", "--history-adapter", "native",
               "--prejepa-missing-context-policy", "reject", "--batch-size", "32",
               "--device", device, "--output", str(out)]
        subprocess.run(cmd, cwd=ROOT, check=True)


def summarize():
    metrics = json.loads(METRICS.read_text())
    dev = json.loads(DEV.read_text())
    result = {"schema": "contextworld.capacity-scaling-test.v1", "split": "test",
              "training_seed": 3072, "data_pairs": 32768, "epochs": 10,
              "test_plan": str(PLAN.relative_to(ROOT)), "test_plan_sha256": sha256(PLAN),
              "test_metrics": str(METRICS.relative_to(ROOT)), "test_metrics_sha256": sha256(METRICS),
              "cem_scope": "original_environment", "rows": []}
    keys = ("bundle_root", "artifact_root", "manifest_sha256", "task_registry_sha256")
    for task, model, baseline, source, endpoint in entries():
        small = next(c for c in metrics["cells"] if c["task"] == task and
                     c["family"] == model and c["regime"] == "scratch")
        assert small["n"] == 1 and small["runs"][0]["checkpoint_sha256"] == baseline["checkpoint_sha256"]
        devrow = next(r for r in dev["rows"] if r["task"] == task and r["model"] == model)
        large_path = HERE / f"results/{task}/{model}/test.json"
        large = json.loads(large_path.read_text())
        small_path = ROOT / small["runs"][0]["result"]
        archived = json.loads(small_path.read_text())
        if (sha256(small_path) != small["runs"][0]["result_sha256"] or
                tuple(archived["bundle"][k] for k in keys) != tuple(large["bundle"][k] for k in keys) or
                large["evaluation_split"] != "test" or
                large["result"]["model"]["adapter"]["checkpoint_sha256"] != endpoint["checkpoint_sha256"] or
                large["result"]["model"]["state_sha256_before"] != large["result"]["model"]["state_sha256_after"]):
            raise ValueError(f"capacity Test identity mismatch: {task} {model}")
        m = large["result"]["metrics"]
        result["rows"].append({"task": task, "model": model, "family": model,
                               "same_training_data": devrow["same_training_data"],
                               "small": {"accuracy_percent": small["scores"]["accuracy"]["mean"],
                                         "main_percent": small["scores"]["accuracy"]["mean"],
                                         "history_percent": small["scores"]["history"]["mean"],
                                         "nre": small["scores"]["nre"]["mean"],
                                         "cem_percent": devrow["small"]["cem_percent"],
                                         "cem_scope": "original_environment",
                                         "checkpoint_sha256": baseline["checkpoint_sha256"],
                                         "test_result": small["runs"][0]["result"],
                                         "test_result_sha256": small["runs"][0]["result_sha256"]},
                               "scaling10x": {"accuracy_percent": 100*m["correct_future_rate"],
                                              "main_percent": 100*m["correct_future_rate"],
                                              "history_percent": 100*m["correct_history_rate"],
                                              "nre": m["latent_response"]["normalized_response_error"],
                                              "cem_percent": devrow["scaling10x"]["cem_percent"],
                                              "cem_scope": "original_environment",
                                              "checkpoint_sha256": endpoint["checkpoint_sha256"],
                                              "test_result": str(large_path.relative_to(ROOT)),
                                              "test_result_sha256": sha256(large_path),
                                              "source_checkpoint_record": str(source),
                                              "source_checkpoint_record_sha256": sha256(source)},
                               "test_bundle": {k: large["bundle"][k] for k in keys}})
    (HERE / "final_comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {HERE / 'final_comparison.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:6")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    summarize() if args.summarize_only else evaluate(args.device)
