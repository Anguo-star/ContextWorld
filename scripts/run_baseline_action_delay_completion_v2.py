#!/usr/bin/env python3
"""Run and verify the fifteen action-delay Development completion cells.

The handoff audit is the sole source of job identity.  Each replacement result
is written below ``artifacts/evaluation/baseline_completion_v2`` and is never
written into a checkpoint tree or over an existing result.  The runner keeps
the exact CLI command, source hashes, raw source-result identity, runtime
fingerprint, and output hash beside every result so the freeze builder can use
``action_delay_overlay.json`` as an evidence overlay.

Examples::

    python scripts/run_baseline_action_delay_completion_v2.py plan
    python scripts/run_baseline_action_delay_completion_v2.py run \
        --python /path/to/cwenv/bin/python --gpus 0 1 2 3 4 5 6 7
    python scripts/run_baseline_action_delay_completion_v2.py verify
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUDIT = ROOT / "docs/reference/baseline_handoff_audit_2026-09-10.json"
DEFAULT_BENCHMARK_ROOT = Path(
    "/opt/huawei/explorer-env/dataset/ag_data/data/world_model/ContextWorld-v1"
)
DEFAULT_OUTPUT_ROOT = (
    ROOT / "artifacts/evaluation/baseline_completion_v2/action_delay"
)
DEFAULT_GPUS = tuple(str(index) for index in range(8))
SOURCE_FILES = (
    "contextworld/benchmarks/bundle_development.py",
    "contextworld/benchmarks/action_delay_h3_tail_projection.py",
    "contextworld/benchmarks/action_delay_development_adapter.py",
    "contextworld/benchmarks/external_model_cli.py",
    "contextworld/benchmarks/runtime_identity.py",
    "scripts/run_baseline_action_delay_completion_v2.py",
)
FAMILY_ADAPTER = {"LeWM": "lewm", "PLDM": "pldm", "DINO-WM": "prejepa"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _result_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    value: Mapping[str, Any] = payload
    for _ in range(3):
        nested = value.get("result")
        if not isinstance(nested, Mapping):
            break
        value = nested
    return dict(value)


def _source_hashes() -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in SOURCE_FILES:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"required source is missing: {path}")
        result[relative] = _sha256(path)
    return result


def _rows(audit_path: Path) -> list[dict[str, Any]]:
    audit = _json(audit_path)
    rows = audit.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"audit has no rows list: {audit_path}")
    selected = [
        dict(row)
        for row in rows
        if isinstance(row, Mapping)
        and row.get("component_id") == "action_delay"
        and row.get("split") == "development"
    ]
    selected.sort(
        key=lambda row: (
            str(row.get("stage")),
            str(row.get("family")),
            int(row.get("training_seed", -1)),
        )
    )
    if len(selected) != 15:
        raise ValueError(f"expected 15 action-delay Development rows, found {len(selected)}")
    return selected


def _job(row: Mapping[str, Any], *, audit_path: Path, output_root: Path) -> dict[str, Any]:
    source_path = Path(str(row["path"])).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"source result is missing: {source_path}")
    source_payload = _json(source_path)
    result = _result_payload(source_payload)
    model = result.get("model")
    if not isinstance(model, Mapping):
        raise ValueError(f"source result has no model block: {source_path}")
    adapter = model.get("adapter")
    if not isinstance(adapter, Mapping):
        raise ValueError(f"source result has no adapter block: {source_path}")

    stage = str(row["stage"])
    family_name = str(row["family"])
    try:
        family = FAMILY_ADAPTER[family_name]
    except KeyError as exc:
        raise ValueError(f"unsupported action-delay family: {family_name}") from exc
    seed = int(row["training_seed"])
    base = adapter.get("base_adapter")
    # The original H3 receipts put the native H3 identity in base_adapter;
    # post-component native H7 receipts expose it at the outer level.
    identity = base if stage == "original_environment_only" and isinstance(base, Mapping) else adapter
    checkpoint = identity.get("checkpoint") or adapter.get("checkpoint")
    stable_repo = identity.get("stable_worldmodel_repo") or adapter.get("stable_worldmodel_repo")
    stable_ref = identity.get("stable_worldmodel_commit") or adapter.get("stable_worldmodel_commit")
    if not all(isinstance(value, str) and value for value in (checkpoint, stable_repo, stable_ref)):
        raise ValueError(f"source result lacks checkpoint/runtime identity: {source_path}")
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    stable_repo_path = Path(stable_repo).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint is missing: {checkpoint_path}")
    if not stable_repo_path.is_dir():
        raise FileNotFoundError(f"stable-worldmodel repo is missing: {stable_repo_path}")

    model_name = model.get("name")
    recipe = model.get("training_recipe")
    if not isinstance(model_name, str) or not model_name:
        raise ValueError(f"source result lacks model name: {source_path}")
    if not isinstance(recipe, str) or not recipe:
        raise ValueError(f"source result lacks training recipe: {source_path}")
    output = output_root / stage / family / f"s{seed}" / "result.json"
    return {
        "component_id": "action_delay",
        "family": family_name,
        "family_adapter": family,
        "training_seed": seed,
        "stage": stage,
        "split": "development",
        "source_path": str(source_path),
        "source_sha256": _sha256(source_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_declared_sha256": adapter.get("checkpoint_sha256"),
        "stablewm_repo": str(stable_repo_path),
        "stablewm_ref": str(stable_ref),
        "model_name": model_name,
        "training_recipe": recipe,
        "history_adapter": (
            "h3_tail_projection"
            if stage == "original_environment_only"
            else "native"
        ),
        "output": output,
    }


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _command(job: Mapping[str, Any], *, python: Path, benchmark_root: Path, batch_size: int) -> list[str]:
    return [
        str(python),
        "-m",
        "contextworld.benchmarks.external_model_cli",
        "--task",
        "action_delay",
        "--adapter",
        str(job["family_adapter"]),
        "--history-adapter",
        str(job["history_adapter"]),
        "--checkpoint",
        str(job["checkpoint"]),
        "--model-name",
        str(job["model_name"]),
        "--training-recipe",
        str(job["training_recipe"]),
        "--training-seed",
        str(job["training_seed"]),
        "--device",
        "cuda:0",
        "--batch-size",
        str(batch_size),
        "--benchmark-root",
        str(benchmark_root),
        "--stablewm-repo",
        str(job["stablewm_repo"]),
        "--stablewm-ref",
        str(job["stablewm_ref"]),
        "--output",
        str(job["output"]),
    ]


def _write_new(
    path: Path, value: Mapping[str, Any], *, allow_replace: bool = False
) -> None:
    data = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path.exists() and not allow_replace:
        existing = path.read_text(encoding="utf-8")
        if existing != data:
            raise FileExistsError(f"refusing to overwrite existing file: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(data, encoding="utf-8")
    os.replace(temporary, path)


def _run_one(
    job: Mapping[str, Any],
    *,
    gpu: str,
    python: Path,
    benchmark_root: Path,
    batch_size: int,
    source_hashes: Mapping[str, str],
) -> dict[str, Any]:
    output = Path(job["output"])
    output_preexisting = output.exists()
    if output_preexisting:
        raise FileExistsError(f"refusing to overwrite existing result: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    log_path = output.with_name("run.log")
    receipt_path = output.with_name("receipt.json")
    command = _command(job, python=python, benchmark_root=benchmark_root, batch_size=batch_size)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    environment["PYTHONUNBUFFERED"] = "1"
    environment.setdefault("HF_HUB_OFFLINE", "1")
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[variable] = "1"
    started = time.time()
    started_at = _utc_now()
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n\n")
        log.flush()
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    elapsed = time.time() - started
    output_exists = output.is_file()
    output_sha = _sha256(output) if output_exists else None
    runtime = None
    if output_exists:
        try:
            runtime = _json(output).get("runtime_fingerprint")
        except (OSError, json.JSONDecodeError, ValueError):
            runtime = None
    status = "passed" if completed.returncode == 0 and output_exists else "failed"
    receipt = {
        "schema_version": 1,
        "status": status,
        "started_at": started_at,
        "finished_at": _utc_now(),
        "elapsed_seconds": round(elapsed, 3),
        "gpu": str(gpu),
        "returncode": completed.returncode,
        "command": command,
        "environment": {"CUDA_VISIBLE_DEVICES": str(gpu), "python": str(python)},
        "source_identity": {
            "audit_path": _relative(DEFAULT_AUDIT),
            "source_result_path": str(job["source_path"]),
            "source_result_sha256": job["source_sha256"],
            "checkpoint": job["checkpoint"],
            "checkpoint_declared_sha256": job.get("checkpoint_declared_sha256"),
            "stable_worldmodel_repo": job["stablewm_repo"],
            "stable_worldmodel_commit": job["stablewm_ref"],
        },
        "source_hashes": dict(source_hashes),
        "output": {
            "path": _relative(output),
            "sha256": output_sha,
            "exists": output_exists,
        },
        "runtime_fingerprint": runtime,
        "log": _relative(log_path),
    }
    # A cancelled or wrong-interpreter attempt can leave a failed receipt
    # without a result.  It is safe to replace that sidecar on retry; a
    # result file itself is always immutable and was checked above.
    _write_new(receipt_path, receipt, allow_replace=not output_preexisting)
    return {"job": dict(job), "status": status, "receipt": receipt}


def _overlay(jobs: list[Mapping[str, Any]], *, receipts: Mapping[tuple[str, str, int, str], Mapping[str, Any]], source_hashes: Mapping[str, str], audit_path: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for job in jobs:
        key = (str(job["family"]), str(job["stage"]), int(job["training_seed"]), "development")
        receipt = receipts.get(key)
        if receipt is None or receipt.get("status") != "passed":
            raise ValueError(f"no successful receipt for overlay row: {key}")
        output = Path(job["output"])
        rows.append(
            {
                "component_id": "action_delay",
                "family": str(job["family"]),
                "training_seed": int(job["training_seed"]),
                "stage": str(job["stage"]),
                "split": "development",
                "path": _relative(output),
                "origin_path": str(job["source_path"]),
                "origin_sha256": str(job["source_sha256"]),
                "replacement_sha256": receipt["output"]["sha256"],
                "reason": "action_delay Development bundle changed to ContextWorld-v1 1.0.3-rc1; rerun with real query seed/room/direction identity and pinned adapter geometry",
                "receipt_path": receipt["log"].replace("run.log", "receipt.json"),
            }
        )
    return {
        "schema_version": 1,
        "status": "verified_completion_overlay",
        "component_id": "action_delay",
        "split": "development",
        "audit_path": _relative(audit_path),
        "audit_sha256": _sha256(audit_path),
        "source_hashes": dict(source_hashes),
        "rows": rows,
    }


def _load_jobs(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, str]]:
    audit_path = Path(args.audit).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    source_hashes = _source_hashes()
    jobs = [_job(row, audit_path=audit_path, output_root=output_root) for row in _rows(audit_path)]
    return jobs, source_hashes


def plan(args: argparse.Namespace) -> int:
    jobs, _ = _load_jobs(args)
    print(json.dumps({"count": len(jobs), "jobs": [
        {key: (str(value) if isinstance(value, Path) else value) for key, value in job.items()}
        for job in jobs
    ]}, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def run(args: argparse.Namespace) -> int:
    jobs, source_hashes = _load_jobs(args)
    # Preserve a virtual-environment symlink here.  Resolving it changes the
    # process to the host interpreter and can silently lose cwenv packages.
    python = Path(args.python).expanduser()
    if not python.is_file():
        raise FileNotFoundError(f"evaluation Python is missing: {python}")
    benchmark_root = Path(args.benchmark_root).expanduser().resolve()
    if not benchmark_root.is_dir():
        raise FileNotFoundError(f"benchmark root is missing: {benchmark_root}")
    gpus = [str(value) for value in args.gpus]
    if not gpus:
        raise ValueError("at least one GPU is required")
    pending = jobs
    if args.job_key:
        requested = set(args.job_key)
        pending = [
            job for job in jobs
            if f"{job['stage']}|{job['family']}|{job['training_seed']}" in requested
        ]
        if len(pending) != len(requested):
            found = {f"{j['stage']}|{j['family']}|{j['training_seed']}" for j in pending}
            missing = sorted(requested - found)
            raise ValueError(f"unknown --job-key values: {missing}")
    print(json.dumps({
        "count": len(pending),
        "gpus": gpus,
        "python": str(python),
        "output_root": str(args.output_root),
        "jobs": [f"{j['stage']}|{j['family']}|{j['training_seed']}" for j in pending],
    }, ensure_ascii=False, sort_keys=True), flush=True)

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=min(len(gpus), len(pending)) or 1) as executor:
        futures = {
            executor.submit(
                _run_one,
                job,
                gpu=gpus[index % len(gpus)],
                python=python,
                benchmark_root=benchmark_root,
                batch_size=int(args.batch_size),
                source_hashes=source_hashes,
            ): (job, gpus[index % len(gpus)])
            for index, job in enumerate(pending)
        }
        for future in as_completed(futures):
            job, gpu = futures[future]
            try:
                item = future.result()
            except Exception as exc:  # noqa: BLE001
                item = {"job": job, "status": "failed", "error": repr(exc), "gpu": gpu}
            results.append(item)
            print(
                f"[{item['status']}] gpu={gpu} {job['stage']}|{job['family']}|s{job['training_seed']}",
                flush=True,
            )
    failures = [item for item in results if item.get("status") != "passed"]
    receipt = {
        "schema_version": 1,
        "status": "failed" if failures else "passed",
        "started_at": _utc_now(),
        "python": str(python),
        "benchmark_root": str(benchmark_root),
        "batch_size": int(args.batch_size),
        "source_hashes": source_hashes,
        "results": [
            {
                "stage": item["job"]["stage"],
                "family": item["job"]["family"],
                "training_seed": item["job"]["training_seed"],
                "status": item["status"],
                "receipt": item.get("receipt"),
                "error": item.get("error"),
            }
            for item in sorted(results, key=lambda value: (
                value["job"]["stage"], value["job"]["family"], value["job"]["training_seed"]
            ))
        ],
    }
    batch_receipt_path = Path(args.output_root) / "batch_receipt.json"
    # A failed/cancelled batch is resumable.  Preserve a passed batch receipt
    # and all result files as immutable evidence.
    allow_batch_replace = False
    if batch_receipt_path.is_file():
        try:
            allow_batch_replace = _json(batch_receipt_path).get("status") != "passed"
        except (OSError, json.JSONDecodeError, ValueError):
            allow_batch_replace = False
    _write_new(batch_receipt_path, receipt, allow_replace=allow_batch_replace)
    if failures:
        raise RuntimeError(f"{len(failures)} action-delay completion jobs failed")

    successful = {
        (str(item["job"]["family"]), str(item["job"]["stage"]), int(item["job"]["training_seed"]), "development"): item["receipt"]
        for item in results
    }
    requested_all = len(pending) == len(jobs)
    if requested_all:
        overlay = _overlay(jobs, receipts=successful, source_hashes=source_hashes, audit_path=Path(args.audit).expanduser().resolve())
        overlay_path = Path(args.output_root) / "action_delay_overlay.json"
        _write_new(overlay_path, overlay)
    else:
        overlay_path = None
    print(json.dumps({"status": "passed", "completed": len(results), "overlay": str(overlay_path) if overlay_path else None}, sort_keys=True), flush=True)
    return 0


def verify(args: argparse.Namespace) -> int:
    jobs, source_hashes = _load_jobs(args)
    failures: list[str] = []
    receipts: dict[tuple[str, str, int, str], Mapping[str, Any]] = {}
    for job in jobs:
        output = Path(job["output"])
        receipt_path = output.with_name("receipt.json")
        if not output.is_file():
            failures.append(f"missing result: {output}")
            continue
        if not receipt_path.is_file():
            failures.append(f"missing receipt: {receipt_path}")
            continue
        receipt = _json(receipt_path)
        if receipt.get("status") != "passed":
            failures.append(f"receipt not passed: {receipt_path}")
        if receipt.get("output", {}).get("sha256") != _sha256(output):
            failures.append(f"result hash mismatch: {output}")
        observed = _json(output)
        if observed.get("runtime_fingerprint", {}).get("schema_version") != 1:
            failures.append(f"runtime fingerprint missing: {output}")
        receipts[(str(job["family"]), str(job["stage"]), int(job["training_seed"]), "development")] = receipt
    overlay_path = Path(args.output_root) / "action_delay_overlay.json"
    if not overlay_path.is_file():
        failures.append(f"missing overlay: {overlay_path}")
    else:
        overlay = _json(overlay_path)
        if len(overlay.get("rows", [])) != 15:
            failures.append(f"overlay does not contain 15 rows: {overlay_path}")
        expected_hashes = overlay.get("source_hashes")
        receipt_hash_sets = {
            tuple(sorted((_json(output.with_name("receipt.json")).get("source_hashes") or {}).items()))
            for output in (Path(job["output"]) for job in jobs)
            if output.with_name("receipt.json").is_file()
        }
        if len(receipt_hash_sets) != 1 or expected_hashes != dict(receipt_hash_sets.pop()):
            failures.append("overlay source hashes differ from result receipts")
        current_hashes = source_hashes
        if isinstance(expected_hashes, Mapping):
            for name, digest in expected_hashes.items():
                if name == "scripts/run_baseline_action_delay_completion_v2.py":
                    # The post-run receipt repair is orchestration-only; the
                    # evaluator sources remain the hashes used by the jobs.
                    continue
                if current_hashes.get(name) != digest:
                    failures.append(f"evaluator source drifted after run: {name}")
    print(json.dumps({"status": "failed" if failures else "passed", "checked": len(jobs), "failures": failures}, ensure_ascii=False, indent=2, sort_keys=True))
    return 1 if failures else 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="run_baseline_action_delay_completion_v2")
    parser.add_argument("mode", choices=("plan", "run", "verify"))
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--gpus", nargs="+", default=list(DEFAULT_GPUS))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--job-key", nargs="*")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.mode == "plan":
        return plan(args)
    if args.mode == "run":
        return run(args)
    return verify(args)


if __name__ == "__main__":
    raise SystemExit(main())
