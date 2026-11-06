#!/usr/bin/env python3
"""Evaluate existing data-scale checkpoints on frozen Test; summarize saved reports."""

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
PLAN = ROOT / "docs/research/data/current_reference_test_v1/plan.json"
MAIN_METRICS = ROOT / "docs/research/data/current_reference_test_v1/metrics.json"
FIGURE3 = ROOT / "docs/research/data/figure3_data_scaling_test_v1/metrics.json"
MAP = HERE / "checkpoint_map.json"
ADAPTER = {"lewm": "lewm", "pldm": "pldm", "dinowm": "prejepa"}
DINO_PYTHON = Path("/tmp/cw-strength-ablation-20260922/venv/bin/python")
PLDM_JOINT_2K_IDENTITY = HERE / "pldm_joint_2k_identity.json"
PLDM_JOINT_2K_REPORT = HERE / "results/action_strength/pldm/joint/scale_2k/s3072/test.json"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def queue():
    return json.loads(MAP.read_text())["matches"]


def report_path(item):
    task, model, regime = item["study_id"].split("/")[:3]
    size = item["study_id"].split("/")[3]
    seed = item["training_seed"]
    if task == "action_strength" and model == "lewm" and size == "scale_2k":
        return ROOT / f"docs/research/data/figure3_data_scaling_test_v1/results/{regime}_s{seed}/test.json"
    return HERE / f"results/{task}/{model}/{regime}/{size}/s{seed}/test.json"


def pldm_joint_2k_row(main):
    """Return the extra archived Test cell only after its result and identity exist."""
    if not PLDM_JOINT_2K_IDENTITY.exists() and not PLDM_JOINT_2K_REPORT.exists():
        return None
    if not PLDM_JOINT_2K_IDENTITY.exists() or not PLDM_JOINT_2K_REPORT.exists():
        raise ValueError("incomplete PLDM Joint 2k Test archive")
    identity = json.loads(PLDM_JOINT_2K_IDENTITY.read_text())
    report = json.loads(PLDM_JOINT_2K_REPORT.read_text())
    result = report["result"]
    adapter = result["model"]["adapter"]
    expected_id = "action_strength/pldm/joint/scale_2k/s3072"
    if (identity["run_id"] != expected_id or identity["training_pair_count"] != 2048
            or identity["test_pair_count"] != 256 or identity["stable_ref"] !=
            "c34a107e787133468369e08f8f3efa2a43db67a2"
            or sha256(PLDM_JOINT_2K_REPORT) != identity["result_sha256"]
            or adapter["checkpoint_sha256"] != identity["checkpoint_sha256"]
            or adapter["stable_worldmodel_commit"] != identity["stable_ref"]
            or report["evaluation_split"] != "test" or result["status"] != "completed"
            or result["model"]["training_recipe"] != "joint"
            or result["model"]["training_seed"] != 3072
            or result["model"]["state_sha256_before"] != result["model"]["state_sha256_after"]
            or result["data"]["pair_count"] != 256 or result["metrics"]["pair_count"] != 256
            or not report["bundle"].get("test_payload_sha256_verified")):
        raise ValueError("invalid PLDM Joint 2k frozen Test result")
    reference = next(c for c in main["cells"] if c["task"] == "action_strength"
                     and c["family"] == "pldm" and c["regime"] == "joint")
    reference_report = json.loads((ROOT / reference["runs"][0]["result"]).read_text())
    keys = ("bundle_root", "artifact_root", "manifest_sha256", "task_registry_sha256")
    if any(report["bundle"][key] != reference_report["bundle"][key] for key in keys):
        raise ValueError("PLDM Joint 2k Test bundle differs from 32k")
    metrics = result["metrics"]
    run = {"id": expected_id, "seed": 3072, "checkpoint": identity["checkpoint"],
           "checkpoint_sha256": identity["checkpoint_sha256"],
           "result": str(PLDM_JOINT_2K_REPORT.relative_to(ROOT)),
           "result_sha256": identity["result_sha256"],
           "accuracy": 100 * metrics["correct_future_rate"],
           "history": 100 * metrics["correct_history_rate"],
           "nre": metrics["latent_response"]["normalized_response_error"],
           "test_bundle": {key: report["bundle"][key] for key in keys}}
    def one(key):
        return {"mean": run[key], "std": None, "n": 1}
    return {"task": "action_strength", "model": "pldm", "family": "pldm",
            "regime": "joint", "training_pair_count": 2048, "n": 1,
            "accuracy": one("accuracy"), "history": one("history"), "nre": one("nre"),
            "runs": [run]}


def evaluate(worker, workers, device, dino_python, skip_dino=False):
    plan = json.loads(PLAN.read_text())
    references = {r["id"]: r for r in plan["runs"]}
    for index, item in enumerate(queue()):
        if index % workers != worker:
            continue
        task, model, regime = item["study_id"].split("/")[:3]
        if skip_dino and model == "dinowm":
            continue
        seed = item["training_seed"]
        output = report_path(item)
        if output.exists():
            report = json.loads(output.read_text())
            if report["result"]["model"]["adapter"]["checkpoint_sha256"] != item["checkpoint_sha256"]:
                raise ValueError(f"saved report checkpoint mismatch: {output}")
            print(f"reuse {output}", flush=True)
            continue
        checkpoint = Path(item["checkpoint"])
        if not checkpoint.is_file() or sha256(checkpoint) != item["checkpoint_sha256"]:
            raise ValueError(f"checkpoint unavailable or changed: {item['study_id']} s{seed}")
        reference = references[f"{task}/{model}/{regime}/s3072"]
        output.parent.mkdir(parents=True, exist_ok=True)
        runner = ("contextworld.benchmarks.external_model_cli" if task == "motion_damping"
                  else "scripts.run_current_reference_test_compat")
        entry = ([str(dino_python), "-m", "contextworld.benchmarks.external_model_cli"]
                 if model == "dinowm" else [sys.executable, "-m", runner])
        cmd = entry + ["--task", task,
               "--adapter", ADAPTER[model], "--checkpoint", str(checkpoint),
               "--model-name", f"{model}-{regime}-{item['study_id'].split('/')[-1]}-s{seed}",
               "--training-recipe", regime, "--training-seed", str(seed),
               "--benchmark-root", plan["benchmark_root"],
               "--stablewm-repo", reference["stable_repo"],
               "--stablewm-ref", reference["stable_ref"],
               "--evaluation-split", "test", "--history-adapter", "native",
               "--prejepa-missing-context-policy", "reject", "--batch-size", "32",
               "--device", device, "--output", str(output)]
        proc = subprocess.run(cmd, cwd=ROOT, check=False)
        if proc.returncode:
            print(f"failed {item['study_id']} s{seed}: exit {proc.returncode}", flush=True)
            continue
        print(f"completed {item['study_id']} s{seed}", flush=True)


def summarize():
    main = json.loads(MAIN_METRICS.read_text())
    figure3 = json.loads(FIGURE3.read_text())
    items = queue()
    output = {"schema": "contextworld.data-scaling-test.v1", "split": "test",
              "plan": str(PLAN.relative_to(ROOT)), "plan_sha256": sha256(PLAN),
              "source_study": json.loads(MAP.read_text())["source"],
              "source_study_sha256": sha256(ROOT / json.loads(MAP.read_text())["source"]),
              "dino_python": str(DINO_PYTHON),
              "dino_runtime": {"transformers": "4.57.1", "torch": "2.9.1"},
              "checkpoint_map_sha256": sha256(MAP), "rows": []}
    identity_keys = ("bundle_root", "artifact_root", "manifest_sha256", "task_registry_sha256")
    grouped = {}
    for item in items:
        grouped.setdefault(item["study_id"], []).append(item)
    for study_id, group in grouped.items():
        task, model, regime, size = study_id.split("/")
        runs = []
        for item in group:
            path = report_path(item)
            report = json.loads(path.read_text())
            adapter = report["result"]["model"]["adapter"]
            measures = report["result"]["metrics"]
            if (report["evaluation_split"] != "test" or
                    adapter["checkpoint_sha256"] != item["checkpoint_sha256"] or
                    report["result"]["model"]["state_sha256_before"] !=
                    report["result"]["model"]["state_sha256_after"]):
                raise ValueError(f"invalid Test report: {path}")
            if model == "dinowm" and (
                    report["runtime_fingerprint"]["packages"]["transformers"] != "4.57.1" or
                    not report["runtime_fingerprint"]["packages"]["torch"].startswith("2.9.1")):
                raise ValueError(f"DINO runtime differs from archived Test: {path}")
            reference = next(c for c in main["cells"] if c["task"] == task and
                             c["family"] == model and c["regime"] == regime)
            reference_report = json.loads((ROOT / reference["runs"][0]["result"]).read_text())
            if any(report["bundle"][k] != reference_report["bundle"][k]
                   for k in identity_keys):
                raise ValueError(f"Test bundle identity differs: {path}")
            runs.append({"id": f"{study_id}/s{item['training_seed']}",
                         "seed": item["training_seed"], "checkpoint": item["checkpoint"],
                         "checkpoint_sha256": item["checkpoint_sha256"],
                         "result": str(path.relative_to(ROOT)), "result_sha256": sha256(path),
                         "accuracy": 100 * measures["correct_future_rate"],
                         "history": 100 * measures["correct_history_rate"],
                         "nre": measures["latent_response"]["normalized_response_error"],
                         "test_bundle": {k: report["bundle"][k] for k in identity_keys}})
        def stats(key):
            vals = [r[key] for r in runs]
            return {"mean": statistics.mean(vals),
                    "std": statistics.stdev(vals) if len(vals) > 1 else None,
                    "n": len(vals)}
        count = {"scale_2k": 2048, "scale_8k": 8192}[size]
        output["rows"].append({"task": task, "model": model, "family": model,
                               "regime": regime, "training_pair_count": count,
                               "n": len(runs), "accuracy": stats("accuracy"),
                               "history": stats("history"), "nre": stats("nre"),
                               "runs": runs})
    for cell in main["cells"]:
        if cell["task"] not in {"action_strength", "robot_arm_mass", "contact_friction",
                                "motion_damping", "cube_gripper_carry", "portal_exit"}:
            continue
        if cell["regime"] not in {"scratch", "joint"}:
            continue
        count = 10000 if cell["task"] == "cube_gripper_carry" else 32768
        if cell["regime"] == "joint" and (cell["task"] != "action_strength" or
                                          cell["family"] != "lewm"):
            continue
        output["rows"].append({"task": cell["task"], "model": cell["family"],
                               "family": cell["family"], "regime": cell["regime"],
                               "training_pair_count": count, "n": cell["n"],
                               "accuracy": cell["scores"]["accuracy"],
                               "history": cell["scores"]["history"],
                               "nre": cell["scores"]["nre"],
                               "runs": cell["runs"],
                               "source_metrics": str(MAIN_METRICS.relative_to(ROOT)),
                               "source_metrics_sha256": sha256(MAIN_METRICS)})
    extra = pldm_joint_2k_row(main)
    if extra is not None:
        output["rows"].append(extra)
    output["rows"].sort(key=lambda r: (r["task"], r["model"], r["regime"], r["training_pair_count"]))
    (HERE / "summary.json").write_text(json.dumps(output, indent=2) + "\n")
    print(f"wrote {HERE / 'summary.json'} with {len(output['rows'])} rows")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--device", default="cuda:6")
    parser.add_argument("--dino-python", type=Path, default=DINO_PYTHON,
                        help="Python with Transformers 4.57.1 and Torch 2.9.1")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--skip-dino", action="store_true")
    args = parser.parse_args()
    summarize() if args.summarize_only else evaluate(args.worker, args.workers, args.device,
                                                     args.dino_python,
                                                     args.skip_dino)
