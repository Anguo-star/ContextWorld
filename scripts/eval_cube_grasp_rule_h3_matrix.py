#!/usr/bin/env python3
"""Score the frozen Cube v4r1 2-model x 3-seed matrix on Development only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contextworld.benchmarks.cube_grasp_rule_reference_score import (  # noqa: E402
    score_cube_reference_development_results,
)
from contextworld.benchmarks.cube_grasp_rule_reference_training import (  # noqa: E402
    DEFAULT_CUBE_REFERENCE_TRAINING_PREREG,
    file_sha256,
    load_cube_reference_training_prereg,
    validate_cube_reference_training_report,
)
from contextworld.paths import resolve_contextworld_path  # noqa: E402


EVALUATOR_MODULE = "contextworld.benchmarks.cube_grasp_rule_reference_cli"


def _require_cuda_devices(gpus: tuple[str, ...]) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Cube Development matrix requires CUDA; no score output was created"
        )
    count = int(torch.cuda.device_count())
    try:
        indices = tuple(int(value) for value in gpus)
    except ValueError as error:
        raise ValueError("Cube Development GPU indices must be integers") from error
    if any(index < 0 or index >= count for index in indices):
        raise RuntimeError(
            f"Cube Development requested GPUs {indices}, but only {count} are visible"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--prereg", type=Path, default=DEFAULT_CUBE_REFERENCE_TRAINING_PREREG
    )
    parser.add_argument("--gpus", default="0,1,2,3,4,5")
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prereg_path = args.prereg.expanduser().resolve()
    prereg = load_cube_reference_training_prereg(
        prereg_path, require_freeze=True
    )
    matrix = prereg["training"]["reference_matrix"]
    seeds = tuple(int(value) for value in matrix["training_seeds"])
    variants = {
        model: matrix["models"][model]["variant"]
        for model in ("lewm", "pldm")
    }
    optimizer_steps = int(matrix["common"]["optimizer_steps"])
    jobs = [(model, seed) for model in ("lewm", "pldm") for seed in seeds]
    gpus = tuple(value.strip() for value in args.gpus.split(",") if value.strip())
    if len(gpus) != len(jobs) or len(set(gpus)) != len(gpus):
        raise ValueError(f"Expected {len(jobs)} distinct GPU indices, got {gpus}")
    training_root = args.training_root.expanduser().resolve()
    expected_training_root = resolve_contextworld_path(
        prereg["planned_artifacts"]["training_root"], repo_root=ROOT
    )
    if training_root != expected_training_root:
        raise RuntimeError("Cube Development scorer training root drifted")
    output = args.output.expanduser().resolve()
    expected_output = resolve_contextworld_path(
        prereg["planned_artifacts"]["development_score_root"], repo_root=ROOT
    )
    if output != expected_output:
        raise RuntimeError("Cube Development score output does not match prereg")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite output: {output}")
    expected_batch_size = int(prereg["evaluation"]["inference_batch_size"])
    if int(args.batch_size) != expected_batch_size:
        raise ValueError("Cube Development inference batch size drifted")
    _require_cuda_devices(gpus)

    validated_cells = {}
    for model, seed in jobs:
        validated_cells[(model, seed)] = validate_cube_reference_training_report(
            prereg,
            model_family=model,
            training_seed=seed,
            prereg_path=prereg_path,
            repo_root=ROOT,
        )
    training_matrix_path = training_root / "matrix_report.json"
    if not training_matrix_path.is_file() or training_matrix_path.is_symlink():
        raise FileNotFoundError("Cube frozen training matrix report is missing")
    training_matrix = json.loads(training_matrix_path.read_text(encoding="utf-8"))
    freeze_path = Path(prereg["_freeze_receipt_path"])
    if (
        training_matrix.get("schema_version") != 1
        or training_matrix.get("status") != "completed"
        or training_matrix.get("preregistration_id") != prereg["preregistration_id"]
        or Path(str(training_matrix.get("training_root", ""))).resolve()
        != training_root
        or training_matrix.get("failures") != []
        or len(training_matrix.get("reports", ())) != 6
        or training_matrix.get("authorization_chain")
        != {
            "preregistration": {
                "path": str(prereg_path),
                "sha256": file_sha256(prereg_path),
            },
            "freeze_receipt": {
                "path": str(freeze_path),
                "sha256": file_sha256(freeze_path),
            },
        }
        or training_matrix["reports"]
        != [
            validated_cells[(model, seed)]["report_payload"]
            for model, seed in jobs
        ]
    ):
        raise RuntimeError("Cube frozen training matrix provenance drifted")
    logs = output / "logs"
    results_root = output / "checkpoints"
    logs.mkdir(parents=True)
    results_root.mkdir()

    processes: list[dict[str, object]] = []
    for (model, seed), gpu in zip(jobs, gpus, strict=True):
        variant = variants[model]
        job_root = training_root / f"{model}_seed{seed}"
        checkpoint = job_root / f"{variant}_step{optimizer_steps}.pt"
        report_path = job_root / "training_report.json"
        if not checkpoint.is_file() or not report_path.is_file():
            raise FileNotFoundError(
                f"Incomplete frozen Cube training cell: {model}/seed{seed}"
            )
        cell = validated_cells[(model, seed)]
        if checkpoint != cell["checkpoint"] or report_path != cell["report"]:
            raise RuntimeError(f"Cube training cell path drift: {model}/seed{seed}")
        name = f"{model}_seed{seed}"
        result = results_root / f"{name}.json"
        log = logs / f"{name}.log"
        command = [
            sys.executable,
            "-m",
            EVALUATOR_MODULE,
            "--prereg",
            str(prereg_path),
            "eval",
            "--checkpoint",
            str(checkpoint),
            "--model-family",
            model,
            "--model-name",
            f"cube_gripper_carry_{model}_seed{seed}",
            "--training-recipe",
            variant,
            "--training-seed",
            str(seed),
            "--device",
            f"cuda:{gpu}",
            "--batch-size",
            str(args.batch_size),
            "--output",
            str(result),
        ]
        stream = log.open("w", encoding="utf-8")
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
        )
        processes.append(
            {
                "name": name,
                "model": model,
                "seed": seed,
                "gpu": gpu,
                "result": result,
                "log": log,
                "stream": stream,
                "process": process,
            }
        )
    while True:
        status = []
        running = 0
        for row in processes:
            process = row["process"]
            assert isinstance(process, subprocess.Popen)
            code = process.poll()
            running += int(code is None)
            status.append(
                {
                    "name": row["name"],
                    "gpu": row["gpu"],
                    "status": "running" if code is None else f"exit_{code}",
                }
            )
        print(json.dumps({"jobs": status}, sort_keys=True), flush=True)
        if not running:
            break
        time.sleep(15)

    failures = []
    for row in processes:
        row["stream"].close()
        process = row["process"]
        assert isinstance(process, subprocess.Popen)
        if process.returncode:
            failures.append(
                {
                    "name": row["name"],
                    "returncode": process.returncode,
                    "log": str(row["log"]),
                }
            )
    if failures:
        payload = {"schema_version": 1, "status": "failed", "failures": failures}
        (output / "matrix_score.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise RuntimeError(f"Cube Development scoring failed: {failures}")

    methods = {}
    for model in ("lewm", "pldm"):
        paths = [
            Path(row["result"])
            for row in processes
            if row["model"] == model
        ]
        methods[model] = score_cube_reference_development_results(
            result_paths=paths,
            model_family=model,
            prereg_config=prereg_path,
            loaded_prereg=prereg,
        )
        (output / f"{model}_three_seed_score.json").write_text(
            json.dumps(methods[model], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    payload = {
        "schema_version": 1,
        "status": "completed",
        "preregistration_id": prereg["preregistration_id"],
        "training_root": str(training_root),
        "authorization_chain": {
            "preregistration": {
                "path": str(prereg_path),
                "sha256": file_sha256(prereg_path),
            },
            "freeze_receipt": {
                "path": str(freeze_path),
                "sha256": file_sha256(freeze_path),
            },
            "training_matrix": {
                "path": str(training_matrix_path),
                "sha256": file_sha256(training_matrix_path),
            },
        },
        "methods": methods,
        "at_least_one_family_passed": any(
            value["passed"] for value in methods.values()
        ),
        "public_test_opened": False,
    }
    result_path = output / "matrix_score.json"
    result_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "completed", "result": str(result_path)}))


if __name__ == "__main__":
    main()
