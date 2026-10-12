#!/usr/bin/env python3
"""Freeze and run every current native checkpoint on offline public Test.

``freeze`` reads checkpoint identities only. ``execute`` is deliberately
separate: it checks the frozen identities before invoking the public CLI.
Each run retains its own result or failure receipt; no score selects a run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.freeze_current_reference_baseline import EVALUATION_SOURCE_PATHS, RELEASE_CONFIG

SCHEMA = "contextworld.current-reference-public-test-plan.v1"
ADAPTER = {"lewm": "lewm", "pldm": "pldm", "dinowm": "prejepa"}
TASKS = {"speed", "door", "action_delay", "portal_exit", "action_strength",
         "contact_friction", "motion_damping", "robot_arm_mass", "cube_gripper_carry"}
SOURCE_FILES = tuple(dict.fromkeys((*EVALUATION_SOURCE_PATHS,
                                    *RELEASE_CONFIG.values(),
                                    "contextworld/training/stablewm_bundle.py",
                                    "contextworld/benchmarks/adapter_registry.py",
                                    "scripts/run_current_reference_test.py")))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(data: object) -> bytes:
    return (json.dumps(data, sort_keys=True, separators=(",", ":")) + "\n").encode()


def atomic_json(path: Path, data: object) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.",
                                     delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(canonical(data))
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)


def pinned_files(bundle_root: Path) -> dict[str, str]:
    paths = {"bundle/task_registry.json": bundle_root / "task_registry.json",
             "bundle/manifest.jsonl": bundle_root / "manifest.jsonl"}
    paths.update({f"source/{name}": ROOT / name for name in SOURCE_FILES})
    return {name: sha256(path) for name, path in paths.items()}


def freeze(models_path: Path, bundle_root: Path) -> dict:
    if not bundle_root.is_absolute() or not bundle_root.is_dir():
        raise ValueError("--benchmark-root must be an existing absolute directory")
    models = json.loads(models_path.read_text())
    if not isinstance(models, list):
        raise ValueError("models manifest must be a list")
    ids: set[str] = set()
    runs = []
    for item in models:
        task, family, regime, seed = (item[k] for k in
                                      ("task", "family", "regime", "training_seed"))
        run_id = f"{task}/{family}/{regime}/s{seed}"
        if task not in TASKS or family not in ADAPTER or item["id"] != run_id or run_id in ids:
            raise ValueError(f"invalid or duplicate current run: {run_id}")
        ids.add(run_id)
        checkpoint = Path(item["checkpoint"])
        stable_repo = Path(item["stable_repo"])
        if not checkpoint.is_absolute() or not checkpoint.is_file():
            raise ValueError(f"missing checkpoint: {checkpoint}")
        if not stable_repo.is_absolute() or not stable_repo.is_dir():
            raise ValueError(f"missing stable repo: {stable_repo}")
        if sha256(checkpoint) != item["checkpoint_sha256"]:
            raise ValueError(f"checkpoint hash mismatch: {run_id}")
        runs.append({"id": run_id, "task": task, "family": family,
                     "regime": regime, "training_seed": seed,
                     "checkpoint": str(checkpoint),
                     "checkpoint_sha256": item["checkpoint_sha256"],
                     "stable_repo": str(stable_repo), "stable_ref": item["stable_ref"],
                     "adapter": ADAPTER[family]})
    if len(runs) != 136 or len({(r["task"], r["family"], r["regime"]) for r in runs}) != 86:
        raise ValueError("current comparison must contain 136 runs in 86 cells")
    runs.sort(key=lambda x: x["id"])
    return {"schema": SCHEMA, "models_path": str(models_path.resolve()),
            "models_sha256": sha256(models_path), "benchmark_root": str(bundle_root),
            "pinned_files": pinned_files(bundle_root),
            "evaluation_split": "test", "history_adapter": "native",
            "prejepa_missing_context_policy": "reject", "batch_size": 32,
            "runs": runs}


def validate_plan(plan: dict, *, selected: list[dict]) -> None:
    if sha256(Path(plan["models_path"])) != plan["models_sha256"]:
        raise ValueError("models manifest changed")
    if pinned_files(Path(plan["benchmark_root"])) != plan["pinned_files"]:
        raise ValueError("public bundle or scorer source changed")
    if len(plan["runs"]) != 136 or len({r["id"] for r in plan["runs"]}) != 136:
        raise ValueError("frozen cohort is incomplete or duplicated")
    manifest = {item["id"]: item for item in json.loads(Path(plan["models_path"]).read_text())}
    if set(manifest) != {r["id"] for r in plan["runs"]}:
        raise ValueError("frozen cohort differs from models manifest")
    for run in plan["runs"]:
        item = manifest[run["id"]]
        expected = {"id": item["id"], "task": item["task"],
                    "family": item["family"], "regime": item["regime"],
                    "training_seed": item["training_seed"],
                    "checkpoint": item["checkpoint"],
                    "checkpoint_sha256": item["checkpoint_sha256"],
                    "stable_repo": item["stable_repo"], "stable_ref": item["stable_ref"],
                    "adapter": ADAPTER[item["family"]]}
        if run != expected:
            raise ValueError(f'frozen run differs from manifest: {run["id"]}')
    for run in selected:
        if sha256(Path(run["checkpoint"])) != run["checkpoint_sha256"]:
            raise ValueError(f'checkpoint hash mismatch: {run["id"]}')


def execute(plan_path: Path, output_root: Path, device: str, *,
            shard_index: int = 0, num_shards: int = 1, retry_failed: bool = False) -> None:
    plan = json.loads(plan_path.read_text())
    if plan.get("schema") != SCHEMA or plan.get("evaluation_split") != "test":
        raise ValueError("unknown plan schema or split")
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("invalid shard index/count")
    selected = [run for index, run in enumerate(plan["runs"])
                if index % num_shards == shard_index]
    validate_plan(plan, selected=selected)
    plan_sha = hashlib.sha256(canonical(plan)).hexdigest()
    output_root.mkdir(parents=True, exist_ok=True)
    status_path = output_root / f"shard-{shard_index}-of-{num_shards}.json"
    status = {"plan_sha256": plan_sha, "shard_index": shard_index,
              "num_shards": num_shards, "selected_ids": [r["id"] for r in selected],
              "complete": [], "failed": []}
    for run in selected:
        run_dir = output_root / run["id"]
        result = run_dir / "test.json"
        receipt_path = run_dir / "receipt.json"
        if run_dir.exists():
            if not receipt_path.is_file():
                raise ValueError(f'ambiguous partial run: {run["id"]}')
            receipt = json.loads(receipt_path.read_text())
            if (receipt.get("id") != run["id"] or receipt.get("plan_sha256") != plan_sha
                    or receipt.get("checkpoint_sha256") != run["checkpoint_sha256"]):
                raise ValueError(f'receipt identity mismatch: {run["id"]}')
            if receipt.get("returncode") == 0 and result.is_file() and sha256(result) == receipt.get("result_sha256"):
                status["complete"].append(run["id"])
                atomic_json(status_path, status)
                continue
            if not retry_failed or receipt.get("returncode") == 0:
                raise ValueError(f'failed or ambiguous prior run: {run["id"]}')
            if result.exists():
                raise ValueError(f'failed run has unexpected result: {run["id"]}')
        else:
            run_dir.mkdir(parents=True)
        command = [sys.executable, "-m", "contextworld.benchmarks.external_model_cli",
                   "--task", run["task"], "--adapter", run["adapter"],
                   "--checkpoint", run["checkpoint"],
                   "--model-name", f'{run["family"]}-{run["regime"]}-s{run["training_seed"]}',
                   "--training-recipe", run["regime"],
                   "--training-seed", str(run["training_seed"]),
                   "--benchmark-root", plan["benchmark_root"],
                   "--stablewm-repo", run["stable_repo"], "--stablewm-ref", run["stable_ref"],
                   "--evaluation-split", "test", "--history-adapter", "native",
                   "--prejepa-missing-context-policy", "reject", "--batch-size", "32",
                   "--device", device, "--output", str(result)]
        with (run_dir / "stdout.log").open("w") as stdout, (run_dir / "stderr.log").open("w") as stderr:
            process = subprocess.run(command, stdout=stdout, stderr=stderr,
                                     cwd=ROOT, check=False)
        receipt = {"id": run["id"], "plan_sha256": plan_sha,
                   "checkpoint_sha256": run["checkpoint_sha256"],
                   "returncode": process.returncode, "result": str(result) if result.is_file() else None,
                   "result_sha256": sha256(result) if result.is_file() else None}
        atomic_json(receipt_path, receipt)
        status["complete" if process.returncode == 0 else "failed"].append(run["id"])
        atomic_json(status_path, status)
        print(json.dumps({"id": run["id"], "returncode": process.returncode}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("freeze", "execute"))
    parser.add_argument("--models", type=Path)
    parser.add_argument("--benchmark-root", type=Path)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    if args.mode == "freeze":
        if args.models is None or args.benchmark_root is None:
            parser.error("freeze requires --models and --benchmark-root")
        plan = freeze(args.models, args.benchmark_root)
        if args.plan.exists():
            raise ValueError("plan already exists")
        args.plan.parent.mkdir(parents=True, exist_ok=True)
        args.plan.write_bytes(canonical(plan))
        print(json.dumps({"plan": str(args.plan), "runs": len(plan["runs"])}))
    else:
        if args.output_root is None:
            parser.error("execute requires --output-root")
        execute(args.plan, args.output_root, args.device,
                shard_index=args.shard_index, num_shards=args.num_shards,
                retry_failed=args.retry_failed)


if __name__ == "__main__":
    main()
