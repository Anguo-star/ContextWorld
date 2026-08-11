#!/usr/bin/env python3
"""Evaluate and independently rescore the frozen friction training matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contextworld.benchmarks.contact_friction_icl_data import (
    DEFAULT_CONTACT_FRICTION_RELEASE_CONFIG,
    load_contact_friction_icl_release,
)
from contextworld.benchmarks.contact_friction_icl_score import (
    score_contact_friction_icl_results,
)


VARIANTS = {
    "lewm": "mixed_dynamics_response_sigreg_0p02",
    "pldm": "mixed_pldm_joint",
}
EVALUATOR_MODULE = "contextworld.benchmarks.contact_friction_icl_cli"
MODEL_NAME_PREFIX = "contact_friction"
TRAINING_RECIPE_PREFIX = "contact_friction"
EVALUATION_DESCRIPTION = __doc__


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=EVALUATION_DESCRIPTION)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--release-config",
        type=Path,
        default=DEFAULT_CONTACT_FRICTION_RELEASE_CONFIG,
    )
    parser.add_argument("--gpus", default="0,1,2,3,4,5")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--models",
        default="lewm,pldm",
        help="Comma-separated subset of lewm,pldm",
    )
    parser.add_argument(
        "--lewm-variant",
        default=VARIANTS["lewm"],
    )
    parser.add_argument(
        "--pldm-variant",
        default=VARIANTS["pldm"],
    )
    parser.add_argument(
        "--optimizer-steps",
        type=int,
        default=None,
        help=(
            "Optional diagnostic override. The formal checkpoint step is "
            "read from the release config."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    release_path = args.release_config.expanduser().resolve()
    release = load_contact_friction_icl_release(release_path)
    optimizer_steps = (
        int(release["training"]["reference_matrix"]["common"][
            "optimizer_steps"
        ])
        if args.optimizer_steps is None
        else int(args.optimizer_steps)
    )
    seeds = tuple(
        int(value)
        for value in release["training"]["reference_matrix"][
            "training_seeds"
        ]
    )
    models = tuple(
        value.strip() for value in args.models.split(",") if value.strip()
    )
    if (
        not models
        or len(set(models)) != len(models)
        or any(model not in VARIANTS for model in models)
    ):
        raise ValueError(f"Invalid --models: {models}")
    variants = {
        "lewm": args.lewm_variant,
        "pldm": args.pldm_variant,
    }
    jobs = [
        (model, seed)
        for model in models
        for seed in seeds
    ]
    gpus = tuple(
        value.strip() for value in args.gpus.split(",") if value.strip()
    )
    if len(gpus) != len(jobs) or len(set(gpus)) != len(gpus):
        raise ValueError(
            f"Expected {len(jobs)} distinct GPU indices, got {gpus}"
        )
    training_root = args.training_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite output: {output}")
    logs = output / "logs"
    results_root = output / "checkpoints"
    logs.mkdir(parents=True)
    results_root.mkdir()

    processes: list[dict[str, object]] = []
    for (model, seed), gpu in zip(jobs, gpus):
        variant = variants[model]
        checkpoint = (
            training_root
            / f"{model}_seed{seed}"
            / f"{variant}_step{optimizer_steps}.pt"
        )
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        name = f"{model}_seed{seed}"
        result = results_root / f"{name}.json"
        log = logs / f"{name}.log"
        command = [
            sys.executable,
            "-m",
            EVALUATOR_MODULE,
            "--release-config",
            str(release_path),
            "eval",
            "--checkpoint",
            str(checkpoint),
            "--adapter",
            model,
            "--model-name",
            f"{MODEL_NAME_PREFIX}_{model}_seed{seed}",
            "--training-recipe",
            f"{TRAINING_RECIPE_PREFIX}_{variant}",
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
                "checkpoint": checkpoint,
                "result": result,
                "log": log,
                "stream": stream,
                "process": process,
            }
        )

    failures = []
    while True:
        running = 0
        status = []
        for row in processes:
            process = row["process"]
            assert isinstance(process, subprocess.Popen)
            code = process.poll()
            if code is None:
                running += 1
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
        payload = {
            "schema_version": 1,
            "status": "failed",
            "failures": failures,
        }
        (output / "matrix_score.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise RuntimeError(f"Independent evaluation failed: {failures}")

    methods = {}
    for model in models:
        paths = [
            Path(row["result"])
            for row in processes
            if row["model"] == model
        ]
        methods[model] = score_contact_friction_icl_results(
            result_paths=paths,
            method_name=f"{TRAINING_RECIPE_PREFIX}_{variants[model]}",
            release_config=release_path,
        )
        (output / f"{model}_three_seed_score.json").write_text(
            json.dumps(methods[model], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    payload = {
        "schema_version": 1,
        "status": "completed",
        "release_id": release["release_id"],
        "training_root": str(training_root),
        "methods": methods,
    }
    result_path = output / "matrix_score.json"
    result_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "completed",
                "result": str(result_path),
                "decisions": {
                    model: value.get(
                        "decision", {"passed": value.get("passed")}
                    )
                    for model, value in methods.items()
                },
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
