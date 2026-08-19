#!/usr/bin/env python3
"""Build and execute one exact job from the frozen original-baseline CEM matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contextworld.benchmarks.original_baseline_cem_matrix import (  # noqa: E402
    DEFAULT_PREREG,
    load_preregistration,
)
from contextworld.paths import resolve_contextworld_path  # noqa: E402


DEFAULT_FREEZE = Path(
    "configs/benchmark/contextworld_original_baseline_cem_freeze_v1.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolved(value: str | Path) -> Path:
    return resolve_contextworld_path(value, repo_root=ROOT)


def _option(arguments: list[str], name: str, value: Any) -> None:
    arguments.extend((name, str(value)))


def _freeze_receipt(path: Path, prereg: dict[str, Any]) -> dict[str, Any]:
    resolved = _resolved(path)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    prereg_path = Path(prereg["_config_path"])
    if (
        payload.get("schema_version") != 1
        or payload.get("freeze_id")
        != "contextworld_original_baseline_cem_freeze_v1"
        or payload.get("status") != "frozen_before_first_cem_episode"
        or payload.get("cem_episodes_consumed_before_freeze") != 0
        or payload.get("preregistration", {}).get("sha256") != _sha256(prereg_path)
        or payload.get("preregistration", {}).get("size_bytes")
        != prereg_path.stat().st_size
    ):
        raise RuntimeError("Original-baseline CEM freeze receipt is invalid")
    return payload


def build_jobs(prereg: dict[str, Any]) -> list[dict[str, Any]]:
    """Resolve the seven cells into 17 immutable atomic output jobs."""

    inputs = prereg["frozen_environment_inputs"]
    runtimes = prereg["runtimes"]
    implementations = prereg["_identity_audit"]
    audit = prereg["input_identity_audit"]
    jobs: list[dict[str, Any]] = []
    for cell in prereg["execution_cells"]:
        environment = str(cell["environment"])
        family = str(cell["family"])
        runner_key = str(cell["runner"])
        runner = implementations[f"implementation.{runner_key}"]["path"]
        runtime = runtimes[str(cell["runtime"])]
        source = inputs[environment]
        output_directory = _resolved(cell["output_directory"])
        checkpoint = cell["checkpoint"]
        config = cell["effective_loader_config"]

        common = [
            sys.executable,
            runner,
        ]
        if environment == "tworoom":
            for seed, filename in zip(
                cell["eval_seeds"], cell["output_files"], strict=True
            ):
                argv = [*common, "eval"]
                _option(argv, "--stable-worldmodel-root", runtime["root"])
                _option(argv, "--expected-ref", runtime["expected_commit"])
                _option(argv, "--checkpoint", _resolved(checkpoint["path"]))
                _option(argv, "--expected-checkpoint-sha256", checkpoint["sha256"])
                _option(argv, "--expected-checkpoint-size", checkpoint["size_bytes"])
                _option(argv, "--expected-config-sha256", config["sha256"])
                _option(argv, "--expected-config-size", config["size_bytes"])
                _option(argv, "--catalog", _resolved(source["catalog"]["path"]))
                _option(argv, "--expected-catalog-sha256", source["catalog"]["sha256"])
                _option(argv, "--expected-catalog-size", source["catalog"]["size_bytes"])
                _option(argv, "--normalizer", _resolved(source["normalizer"]["path"]))
                _option(
                    argv,
                    "--expected-normalizer-sha256",
                    source["normalizer"]["sha256"],
                )
                _option(
                    argv,
                    "--expected-normalizer-size",
                    source["normalizer"]["size_bytes"],
                )
                _option(argv, "--expected-source-sha256", source["dataset"]["sha256"])
                _option(argv, "--expected-source-size", source["dataset"]["size_bytes"])
                _option(argv, "--seed", seed)
                _option(argv, "--device", "cuda:0")
                output = output_directory / filename
                _option(argv, "--output", output)
                jobs.append(
                    {
                        "job_id": f"{environment}_{family}_seed{seed}",
                        "cell": [environment, family],
                        "mujoco_gl": cell["mujoco_gl"],
                        "evaluations": int(cell["queries_per_seed"]),
                        "output": str(output),
                        "argv": argv,
                    }
                )
            continue

        output = output_directory / cell["output_files"][0]
        if environment in {"pusht", "reacher"}:
            plan = runtime["plan_configs"][environment]
            argv = [*common, "eval"]
            _option(argv, "--task", environment)
            _option(argv, "--stable-worldmodel-root", runtime["root"])
            _option(argv, "--expected-ref", runtime["expected_commit"])
            _option(argv, "--expected-plan-config-sha256", plan["sha256"])
            _option(argv, "--expected-plan-config-size", plan["size_bytes"])
            _option(
                argv,
                "--model",
                f"{cell['checkpoint_id']}={_resolved(checkpoint['path'])}",
            )
            _option(argv, "--expected-checkpoint-sha256", checkpoint["sha256"])
            _option(argv, "--expected-checkpoint-size", checkpoint["size_bytes"])
            _option(argv, "--expected-config-sha256", config["sha256"])
            _option(argv, "--expected-config-size", config["size_bytes"])
            _option(argv, "--dataset", _resolved(source["dataset"]["path"]))
            _option(argv, "--expected-dataset-sha256", source["dataset"]["sha256"])
            _option(argv, "--expected-dataset-size", source["dataset"]["size_bytes"])
            _option(argv, "--input-identity-audit", _resolved(audit["path"]))
            _option(argv, "--expected-input-identity-audit-sha256", audit["sha256"])
            _option(argv, "--expected-input-identity-audit-size", audit["size_bytes"])
            _option(argv, "--query-catalog", _resolved(source["catalog"]["path"]))
            _option(argv, "--expected-catalog-sha256", source["catalog"]["sha256"])
            _option(argv, "--expected-catalog-size", source["catalog"]["size_bytes"])
            _option(argv, "--eval-seeds", ",".join(str(x) for x in cell["eval_seeds"]))
            _option(argv, "--num-eval", cell["queries_per_seed"])
            _option(argv, "--device", "cuda:0")
            _option(argv, "--output", output_directory)
        elif environment == "cube":
            plan = runtime["plan_configs"]["cube"]
            argv = [*common]
            _option(argv, "--stable-worldmodel-root", runtime["root"])
            _option(argv, "--expected-ref", runtime["expected_commit"])
            _option(argv, "--expected-plan-config-sha256", plan["sha256"])
            _option(argv, "--expected-plan-config-size", plan["size_bytes"])
            _option(argv, "--checkpoint", _resolved(checkpoint["path"]))
            _option(argv, "--expected-checkpoint-sha256", checkpoint["sha256"])
            _option(argv, "--expected-checkpoint-size", checkpoint["size_bytes"])
            _option(argv, "--expected-config-sha256", config["sha256"])
            _option(argv, "--expected-config-size", config["size_bytes"])
            _option(argv, "--dataset", _resolved(source["dataset"]["path"]))
            _option(argv, "--expected-dataset-sha256", source["dataset"]["sha256"])
            _option(argv, "--expected-dataset-size", source["dataset"]["size_bytes"])
            _option(argv, "--input-identity-audit", _resolved(audit["path"]))
            _option(argv, "--expected-input-identity-audit-sha256", audit["sha256"])
            _option(argv, "--expected-input-identity-audit-size", audit["size_bytes"])
            _option(argv, "--query-catalog", _resolved(source["catalog"]["path"]))
            _option(argv, "--expected-catalog-sha256", source["catalog"]["sha256"])
            _option(argv, "--expected-catalog-size", source["catalog"]["size_bytes"])
            _option(argv, "--device", "cuda:0")
            _option(argv, "--output", output_directory)
        else:  # pragma: no cover - the preregistration loader rejects this.
            raise RuntimeError(environment)
        jobs.append(
            {
                "job_id": f"{environment}_{family}",
                "cell": [environment, family],
                "mujoco_gl": cell["mujoco_gl"],
                "evaluations": 300,
                "output": str(output),
                "argv": argv,
            }
        )
    if len(jobs) != 17 or sum(int(row["evaluations"]) for row in jobs) != 2100:
        raise RuntimeError("Frozen CEM job expansion is not 17 jobs / 2100 episodes")
    if len({str(row["job_id"]) for row in jobs}) != 17 or len(
        {str(row["output"]) for row in jobs}
    ) != 17:
        raise RuntimeError("Frozen CEM job identifiers or outputs are duplicated")
    return jobs


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prereg", type=Path, default=DEFAULT_PREREG)
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--list", action="store_true")
    action.add_argument("--job")
    parser.add_argument("--gpu-index", type=int)
    args = parser.parse_args(argv)
    if args.job is not None and (args.gpu_index is None or args.gpu_index < 0):
        parser.error("--job requires a non-negative --gpu-index")
    if args.list and args.gpu_index is not None:
        parser.error("--list does not accept --gpu-index")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    prereg = load_preregistration(args.prereg, require_outputs_absent=False)
    freeze = _freeze_receipt(args.freeze, prereg)
    jobs = build_jobs(prereg)
    if args.list:
        print(
            json.dumps(
                {"freeze": freeze, "job_count": len(jobs), "jobs": jobs},
                indent=2,
                sort_keys=True,
            )
        )
        return
    selected = [row for row in jobs if row["job_id"] == args.job]
    if len(selected) != 1:
        raise ValueError(f"Unknown frozen CEM job: {args.job}")
    job = selected[0]
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(args.gpu_index),
            "GIT_CONFIG_GLOBAL": "/tmp/contextworld-original-baseline-gitconfig",
            "MPLCONFIGDIR": "/tmp/contextworld-original-baseline-cem-mpl",
            "MUJOCO_GL": str(job["mujoco_gl"]),
            "PYTHONHASHSEED": "0",
        }
    )
    subprocess.run(job["argv"], cwd=ROOT, env=environment, check=True)


if __name__ == "__main__":
    main()
