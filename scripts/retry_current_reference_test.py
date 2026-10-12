#!/usr/bin/env python3
"""Retry incomplete frozen Test runs through the compatibility wrapper."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_current_reference_test import (ADAPTER, SCHEMA, atomic_json, canonical,
                                                sha256, validate_plan)

WRAPPER = ROOT / "scripts/run_current_reference_test_compat.py"


def _complete(receipt_path: Path, result: Path, run: dict, plan_sha: str) -> bool:
    if not receipt_path.is_file():
        if result.exists():
            raise ValueError(f"result without receipt: {run['id']}")
        return False
    receipt = json.loads(receipt_path.read_text())
    if (receipt.get("id") != run["id"] or receipt.get("plan_sha256") != plan_sha or
            receipt.get("checkpoint_sha256") != run["checkpoint_sha256"]):
        raise ValueError(f"receipt identity mismatch: {run['id']}")
    if receipt.get("returncode") == 0:
        if not result.is_file() or sha256(result) != receipt.get("result_sha256"):
            raise ValueError(f"successful receipt has missing or mismatched result: {run['id']}")
        return True
    if result.exists():
        raise ValueError(f"failed receipt has unexpected result: {run['id']}")
    return False


def _save_previous(run_dir: Path) -> None:
    prior = [run_dir / name for name in ("receipt.json", "stdout.log", "stderr.log", "test.json")]
    present = [path for path in prior if path.exists()]
    if not present:
        return
    attempts = run_dir / "attempts"
    number = 1
    while (attempts / str(number)).exists():
        number += 1
    target = attempts / str(number)
    target.mkdir(parents=True)
    for path in present:
        os.replace(path, target / path.name)


def command_for(run: dict, plan: dict, result: Path, device: str, *,
                artifact_root: Path | None, damping_audit_root: Path | None) -> list[str]:
    command = [sys.executable, "-m", "scripts.run_current_reference_test_compat",
               "--task", run["task"], "--adapter", ADAPTER[run["family"]],
               "--checkpoint", run["checkpoint"],
               "--model-name", f'{run["family"]}-{run["regime"]}-s{run["training_seed"]}',
               "--training-recipe", run["regime"], "--training-seed", str(run["training_seed"]),
               "--benchmark-root", plan["benchmark_root"],
               "--stablewm-repo", run["stable_repo"], "--stablewm-ref", run["stable_ref"],
               "--evaluation-split", "test", "--history-adapter", "native",
               "--prejepa-missing-context-policy", "reject", "--batch-size", "32",
               "--device", device, "--output", str(result)]
    if artifact_root is not None:
        command += ["--artifact-root", str(artifact_root)]
    if damping_audit_root is not None:
        command += ["--damping-audit-root", str(damping_audit_root)]
    return command


def execute(plan_path: Path, output_root: Path, *, device: str = "cuda",
            shard_index: int = 0, num_shards: int = 8,
            artifact_root: Path | None = None,
            damping_audit_root: Path | None = None,
            tasks: set[str] | None = None,
            exclude_tasks: set[str] | None = None) -> dict:
    plan = json.loads(plan_path.read_text())
    if plan.get("schema") != SCHEMA or plan.get("evaluation_split") != "test":
        raise ValueError("unknown frozen plan")
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("invalid shard")
    selected = [run for index, run in enumerate(plan["runs"])
                if index % num_shards == shard_index and
                (tasks is None or run["task"] in tasks) and
                (exclude_tasks is None or run["task"] not in exclude_tasks)]
    plan_sha = hashlib.sha256(canonical(plan)).hexdigest()
    todo = []
    for run in selected:
        directory = output_root / run["id"]
        if not _complete(directory / "receipt.json", directory / "test.json", run, plan_sha):
            todo.append(run)
    validate_plan(plan, selected=todo)
    wrapper_hash = sha256(WRAPPER)
    status = {"plan_sha256": plan_sha, "shard_index": shard_index,
              "num_shards": num_shards, "selected_ids": [run["id"] for run in selected],
              "already_complete": [run["id"] for run in selected if run not in todo],
              "complete": [], "failed": []}
    output_root.mkdir(parents=True, exist_ok=True)
    status_path = output_root / f"compat-shard-{shard_index}-of-{num_shards}.json"
    for run in todo:
        directory = output_root / run["id"]
        directory.mkdir(parents=True, exist_ok=True)
        _save_previous(directory)
        result = directory / "test.json"
        command = command_for(run, plan, result, device,
                              artifact_root=artifact_root,
                              damping_audit_root=damping_audit_root)
        with (directory / "stdout.log").open("w") as stdout, (directory / "stderr.log").open("w") as stderr:
            process = subprocess.run(command, stdout=stdout, stderr=stderr, cwd=ROOT, check=False)
        receipt = {"id": run["id"], "plan_sha256": plan_sha,
                   "checkpoint_sha256": run["checkpoint_sha256"],
                   "returncode": process.returncode,
                   "result": str(result) if result.is_file() else None,
                   "result_sha256": sha256(result) if result.is_file() else None,
                   "runtime_arguments": {"device": device,
                                         "artifact_root": str(artifact_root) if artifact_root else None,
                                         "damping_audit_root": str(damping_audit_root) if damping_audit_root else None},
                   "wrapper_sha256": wrapper_hash}
        atomic_json(directory / "receipt.json", receipt)
        status["complete" if process.returncode == 0 and result.is_file() else "failed"].append(run["id"])
        atomic_json(status_path, status)
        print(json.dumps({"id": run["id"], "returncode": process.returncode}), flush=True)
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--damping-audit-root", type=Path)
    parser.add_argument("--tasks", help="comma-separated task names")
    parser.add_argument("--exclude-task", action="append", default=[])
    args = parser.parse_args()
    execute(args.plan, args.output_root, device=args.device,
            shard_index=args.shard_index, num_shards=args.num_shards,
            artifact_root=args.artifact_root, damping_audit_root=args.damping_audit_root,
            tasks=set(args.tasks.split(",")) if args.tasks else None,
            exclude_tasks=set(args.exclude_task))


if __name__ == "__main__":
    main()
