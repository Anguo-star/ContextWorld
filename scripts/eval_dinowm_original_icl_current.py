"""Evaluate existing CEM-bound original DINO-WM checkpoints; never train.

Runs the frozen task metrics for the declared fixed-context inference variant.
The original CEM evidence is reused unchanged. All paths and checkpoint hashes
come from the completed original-checkpoint evidence, not a checkpoint search.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from contextworld.benchmarks.prejepa_original_context_adapter import ADAPTER_SPEC, METHOD_ID
from contextworld.benchmarks.reference_decision import reference_decision_for_result
from scripts.freeze_current_reference_baseline import _extract_score

SOURCE = ROOT / "artifacts/evaluation/dinowm_original_diagnostic_v1/summary.json"
FREEZE = ROOT / "configs/benchmark/contextworld_joint_scratch_v1_reference_results_freeze_v3.json"
OUT = ROOT / "artifacts/evaluation/dinowm_original_icl_current_v1"
MODEL_ROOT = Path("/opt/huawei/explorer-env/dataset/ag_data/ckpt/dino-wm/checkpoints")
DATA_ROOT = Path("/opt/huawei/explorer-env/dataset/ag_data/data/world_model")
STABLE_REPO = Path("/opt/huawei/explorer-env/dataset/ag_data/code/stable-worldmodel")
SEEDS = (3072, 3073, 3074)
TASKS = {
    "speed": "tworoom", "action_strength": "pusht", "robot_arm_mass": "reacher",
    "action_delay": "tworoom", "contact_friction": "pusht", "motion_damping": "pusht",
    "cube_gripper_carry": "cube", "door": "tworoom", "portal_exit": "tworoom",
}
DEV_ONLY = {"contact_friction", "motion_damping"}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path):
    return json.loads(path.read_text())


def save(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def jobs():
    source = read(SOURCE)
    assert sha(SOURCE) == read(FREEZE)["inputs"]["dino_original_diagnostic"]["sha256"]
    result = []
    for task, environment in TASKS.items():
        for seed in SEEDS:
            name = f"{environment}_prejepa_original_s{seed}"
            checkpoint = MODEL_ROOT / name / "weights_epoch_10.pt"
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            for split in ("development", "test"):
                if split == "test" and task in DEV_ONLY:
                    continue
                result.append({
                    "task": task, "environment": environment, "seed": seed, "split": split,
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": source["checkpoints"][name],
                    "output": str(OUT / task / f"s{seed}" / f"{split}.json"),
                })
    assert len(result) == 48
    return result


def validate_result(job):
    path = Path(job["output"])
    payload = read(path)
    if payload.get("evaluation_split") != job["split"]:
        raise ValueError(f"Wrong split: {path}")
    result = payload["result"]
    model = result["model"]
    adapter = model.get("adapter", model)  # Speed Test flattens adapter metadata.
    if adapter.get("inference_variant") != METHOD_ID:
        raise ValueError(f"Undeclared inference variant: {path}")
    if adapter.get("checkpoint_sha256") != job["checkpoint_sha256"]:
        raise ValueError(f"Checkpoint does not match original CEM evidence: {path}")
    if adapter.get("checkpoint_weights_modified") is not False:
        raise ValueError(f"Weights must be unchanged: {path}")
    score = _extract_score(payload, job["task"], job["split"])
    if not 0 <= score <= 1:
        raise ValueError(f"Invalid main score: {path}")
    return payload, score


def evaluate(job, *, python: str, gpu: str, batch_size: int):
    output = Path(job["output"])
    if output.exists():
        validate_result(job)
        return {**job, "status": "reused_verified", "sha256": sha(output)}
    command = [
        python, "-m", "contextworld.benchmarks.external_model_cli",
        "--task", job["task"], "--adapter", ADAPTER_SPEC,
        "--checkpoint", job["checkpoint"],
        "--model-name", "DINO-WM-original-fixed-context",
        "--training-recipe", "original_environment_only_fixed_context_v1",
        "--training-seed", str(job["seed"]), "--device", "cuda:0",
        "--batch-size", str(batch_size), "--benchmark-root",
        str(DATA_ROOT / ("ContextWorld-v1" if job["split"] == "development" else "ContextWorld-v1-full")),
        "--stablewm-repo", str(STABLE_REPO),
        "--evaluation-split", job["split"], "--include-records", "--output", str(output),
    ]
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    # Test-split scorers bind repo_root to the benchmark bundle root, which
    # breaks the portable "artifacts/..." fallback in contextworld.paths.
    # Pin the documented override so data resolves to the same canonical
    # artifact root as every repository-local run; scorer-side sha256
    # verification of task data still applies unchanged.
    environment.setdefault(
        "CONTEXTWORLD_ARTIFACT_ROOT",
        str((ROOT.parents[1] / "data/world_model/context_world").resolve()),
    )
    # Released speed catalogs retain artifacts/... payload references. Resolve
    # those through the canonical archive; the scorer verifies every NPZ hash.
    if job["task"] == "speed" and job["split"] == "test":
        environment["CONTEXTWORLD_ARTIFACT_ROOT"] = str(DATA_ROOT / "context_world")
    output.parent.mkdir(parents=True, exist_ok=True)
    log = output.with_suffix(".log")
    receipt = {**job, "command": command, "gpu": gpu,
               "started_utc": datetime.now(timezone.utc).isoformat(), "status": "running"}
    receipt_path = output.with_suffix(".receipt.json")
    save(receipt_path, receipt)
    with log.open("w") as stream:
        run = subprocess.run(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
    receipt.update(returncode=run.returncode, status="failed" if run.returncode else "completed")
    if run.returncode == 0:
        try:
            _, receipt["main_score"] = validate_result(job)
            receipt["sha256"] = sha(output)
        except Exception as exc:
            receipt.update(status="failed", validation_error=str(exc))
    save(receipt_path, receipt)
    print(f"{job['task']}/s{job['seed']}/{job['split']}: {receipt['status']}", flush=True)
    return receipt


def summarize(all_jobs):
    rows = {}
    for job in all_jobs:
        payload, score = validate_result(job)
        key = job["task"], job["seed"]
        row = rows.setdefault(key, {
            "component_id": job["task"], "family": "DINO-WM", "stage": "original_environment_only",
            "training_seed": job["seed"], "environment": job["environment"],
            "checkpoint": job["checkpoint"], "checkpoint_sha256": job["checkpoint_sha256"],
        })
        decision = reference_decision_for_result(job["task"], payload, split=job["split"], repo_root=ROOT)
        row[job["split"]] = {"main_score": score, "decision": decision,
                             "source_result_path": job["output"], "source_result_sha256": sha(Path(job["output"]))}
    assert len(rows) == 27
    source = read(SOURCE)
    adapter = ROOT / "contextworld/benchmarks/prejepa_original_context_adapter.py"
    result = {
        "schema_version": "contextworld.dinowm-original-fixed-context-results.v1",
        "status": "completed", "inference_variant": METHOD_ID,
        "claim_boundary": "Declared fixed-context inference variant, not native state-input scores or a data-only ablation",
        "base_freeze": {"path": str(FREEZE.relative_to(ROOT)), "sha256": sha(FREEZE)},
        "original_checkpoint_and_cem_source": {"path": str(SOURCE.relative_to(ROOT)), "sha256": sha(SOURCE)},
        "adapter_source": {"path": str(adapter.relative_to(ROOT)), "sha256": sha(adapter)},
        "icl_cells": len(all_jobs), "distinct_checkpoints": 12,
        "development_tasks": list(TASKS), "test_tasks": [t for t in TASKS if t not in DEV_ONLY],
        "checkpoint_results": list(rows.values()),
        "original_environment_cem": source["original_environment_cem"],
    }
    save(OUT / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "run", "verify"))
    parser.add_argument("--python", default=None)
    parser.add_argument("--gpus", nargs="+", default=[str(i) for i in range(8)])
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    all_jobs = jobs()
    if args.action == "plan":
        save(OUT / "plan.json", {"inference_variant": METHOD_ID, "jobs": all_jobs})
        print("12 existing checkpoints; 27 Development + 21 Test cells; reuse completed CEM; no training")
        return 0
    if args.action == "verify":
        summarize(all_jobs)
        print("Verified 48 ICL results, 12 original checkpoints and original CEM source")
        return 0
    OUT.mkdir(parents=True, exist_ok=True)
    run_lock = (OUT / "run.lock").open("a")
    try:
        fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("An evaluation batch is already running; use verify after it finishes")
        return 1
    python = args.python or read(ROOT / "artifacts/evaluation/baseline_completion_v2/action_delay/batch_receipt.json")["python"]
    pending = queue.Queue()
    for job in all_jobs:
        pending.put(job)

    def worker(gpu):
        completed = []
        while True:
            try:
                job = pending.get_nowait()
            except queue.Empty:
                return completed
            try:
                completed.append(evaluate(job, python=python, gpu=gpu, batch_size=args.batch_size))
            except Exception as exc:
                completed.append({**job, "status": "failed", "error": str(exc)})
                print(f"{job['task']}/s{job['seed']}/{job['split']}: {exc}", flush=True)

    with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        receipts = [r for group in pool.map(worker, args.gpus) for r in group]
    save(OUT / "batch_receipt.json", receipts)
    if any(r["status"] == "failed" for r in receipts):
        return 1
    summarize(all_jobs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
