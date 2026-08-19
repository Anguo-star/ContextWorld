#!/usr/bin/env python3
"""Build and execute jobs from the frozen original-baseline seed-completion matrix.

Expands ``contextworld_original_baseline_seed_completion_prereg_v1.yaml`` into
42 immutable atomic jobs (12 aggregate cells + 30 TwoRoom per-eval-seed cells,
5100 episodes) and provides three phases:

  --list                       dump the frozen job expansion
  --preflight CELL --gpu N     run the frozen runner's preflight (0 episodes)
  --job JOB --gpu N            execute exactly one frozen job

Identity policy: the standard/tworoom/cube-PLDM runners close every identity
themselves through their ``--expected-*`` flags.  The cube LeWM cells reuse the
frozen v2 retention evaluator, whose CLI closes only the dataset size, so for
those cells THIS launcher streams sha256 over checkpoint, loader config, query
catalog, and the runtime plan config and refuses to launch on any mismatch
(preregistration field ``launcher_must_verify_checkpoint_identity``).
No runner is modified; outputs are written only by the frozen runners, which
refuse to overwrite existing namespaces.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Sequence

import yaml

ROOT = Path(__file__).resolve().parents[1]
PREREG = (
    ROOT
    / "configs/benchmark/contextworld_original_baseline_seed_completion_prereg_v1.yaml"
)
SWM_ROOT_FLAG = "--stable-worldmodel-" + "root"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolved(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _assert_identity(spec: dict[str, Any], *, label: str) -> None:
    path = _resolved(str(spec["path"]))
    size = path.stat().st_size
    if size != int(spec["size_bytes"]):
        raise RuntimeError(f"{label} size drifted: {path} ({size})")
    observed = _sha256(path)
    if observed != str(spec["sha256"]):
        raise RuntimeError(f"{label} sha256 drifted: {path} ({observed})")


def load_prereg() -> dict[str, Any]:
    prereg = yaml.safe_load(PREREG.read_text(encoding="utf-8"))
    if (
        prereg["preregistration_id"]
        != "contextworld_original_baseline_seed_completion_v1"
        or prereg["schema_version"] != 1
    ):
        raise RuntimeError("Unexpected seed-completion preregistration")
    for key, spec in prereg["implementation"].items():
        if isinstance(spec, dict) and "sha256" in spec:
            _assert_identity(spec, label=f"frozen runner {key}")
    return prereg


def _option(argv: list[str], name: str, value: Any) -> None:
    argv.extend((name, str(value)))


def build_jobs(prereg: dict[str, Any]) -> list[dict[str, Any]]:
    inputs = prereg["frozen_environment_inputs"]
    runtimes = prereg["runtimes"]
    implementation = prereg["implementation"]
    audit = inputs["input_identity_audit"]
    standard_runtime = runtimes["pusht_reacher_cube"]
    jobs: list[dict[str, Any]] = []
    for cell in prereg["new_member_cells"]:
        cell_id = str(cell["cell_id"])
        environment = str(cell["environment"])
        runner = ROOT / str(implementation[str(cell["runner"])]["path"])
        source = inputs[environment]
        checkpoint = cell["checkpoint"]
        config = cell["effective_loader_config"]
        output_directory = _resolved(str(cell["output_directory"]))
        if environment == "tworoom":
            runtime = runtimes["tworoom"]
            for seed in cell["eval_seeds"]:
                argv = [sys.executable, str(runner), "eval"]
                _option(argv, SWM_ROOT_FLAG, runtime["root"])
                _option(argv, "--expected-ref", runtime["expected_commit"])
                _option(argv, "--checkpoint", checkpoint["path"])
                _option(argv, "--expected-checkpoint-sha256", checkpoint["sha256"])
                _option(argv, "--expected-checkpoint-size", checkpoint["size_bytes"])
                _option(argv, "--expected-config-sha256", config["sha256"])
                _option(argv, "--expected-config-size", config["size_bytes"])
                _option(argv, "--catalog", source["catalog"]["path"])
                _option(argv, "--expected-catalog-sha256", source["catalog"]["sha256"])
                _option(argv, "--expected-catalog-size", source["catalog"]["size_bytes"])
                _option(argv, "--normalizer", str(_resolved(source["normalizer"]["path"])))
                _option(argv, "--expected-normalizer-sha256", source["normalizer"]["sha256"])
                _option(argv, "--expected-normalizer-size", source["normalizer"]["size_bytes"])
                _option(argv, "--expected-source-sha256", source["dataset"]["sha256"])
                _option(argv, "--expected-source-size", source["dataset"]["size_bytes"])
                _option(argv, "--seed", seed)
                _option(argv, "--device", "cuda:0")
                _option(argv, "--output", output_directory / f"seed{seed}.json")
                jobs.append(
                    {
                        "job_id": f"{cell_id}_eval{seed}",
                        "cell_id": cell_id,
                        "mujoco_gl": cell["mujoco_gl"],
                        "evaluations": int(cell["queries_per_seed"]),
                        "launcher_verifies": [],
                        "output": str(output_directory / f"seed{seed}.json"),
                        "argv": argv,
                        "preflight_argv": None,
                    }
                )
            preflight_argv = list(jobs[-1]["argv"])
            preflight_argv[2] = "preflight-model"
            preflight_argv = [
                token
                for index, token in enumerate(preflight_argv)
                if not (
                    token in {"--seed", "--output"}
                    or (index > 0 and preflight_argv[index - 1] in {"--seed", "--output"})
                )
            ]
            jobs[-6]["preflight_argv"] = preflight_argv
            continue

        seeds = ",".join(str(seed) for seed in cell["eval_seeds"])
        if str(cell["runner"]) == "standard_runner":
            argv = [sys.executable, str(runner), "eval"]
            _option(argv, "--task", environment)
            _option(argv, SWM_ROOT_FLAG, standard_runtime["root"])
            _option(argv, "--expected-ref", standard_runtime["expected_commit"])
            plan = standard_runtime["plan_configs"][environment]
            _option(argv, "--expected-plan-config-sha256", plan["sha256"])
            _option(argv, "--expected-plan-config-size", plan["size_bytes"])
            _option(argv, "--model", f"{cell_id}={checkpoint['path']}")
            _option(argv, "--expected-checkpoint-sha256", checkpoint["sha256"])
            _option(argv, "--expected-checkpoint-size", checkpoint["size_bytes"])
            _option(argv, "--expected-config-sha256", config["sha256"])
            _option(argv, "--expected-config-size", config["size_bytes"])
            _option(argv, "--dataset", source["dataset"]["path"])
            _option(argv, "--expected-dataset-sha256", source["dataset"]["sha256"])
            _option(argv, "--expected-dataset-size", source["dataset"]["size_bytes"])
            _option(argv, "--input-identity-audit", str(_resolved(audit["path"])))
            _option(argv, "--expected-input-identity-audit-sha256", audit["sha256"])
            _option(argv, "--expected-input-identity-audit-size", audit["size_bytes"])
            _option(argv, "--query-catalog", str(_resolved(source["catalog"]["path"])))
            _option(argv, "--expected-catalog-sha256", source["catalog"]["sha256"])
            _option(argv, "--expected-catalog-size", source["catalog"]["size_bytes"])
            _option(argv, "--eval-seeds", seeds)
            _option(argv, "--num-eval", cell["queries_per_seed"])
            _option(argv, "--device", "cuda:0")
            _option(argv, "--output", output_directory)
            preflight_argv = list(argv)
            preflight_argv[2] = "preflight-model"
            launcher_verifies: list[dict[str, Any]] = []
        elif str(cell["runner"]) == "cube_pldm_wrapper":
            plan = standard_runtime["plan_configs"]["cube"]
            argv = [sys.executable, str(runner)]
            _option(argv, SWM_ROOT_FLAG, standard_runtime["root"])
            _option(argv, "--expected-ref", standard_runtime["expected_commit"])
            _option(argv, "--expected-plan-config-sha256", plan["sha256"])
            _option(argv, "--expected-plan-config-size", plan["size_bytes"])
            _option(argv, "--checkpoint", checkpoint["path"])
            _option(argv, "--expected-checkpoint-sha256", checkpoint["sha256"])
            _option(argv, "--expected-checkpoint-size", checkpoint["size_bytes"])
            _option(argv, "--expected-config-sha256", config["sha256"])
            _option(argv, "--expected-config-size", config["size_bytes"])
            _option(argv, "--dataset", source["dataset"]["path"])
            _option(argv, "--expected-dataset-sha256", source["dataset"]["sha256"])
            _option(argv, "--expected-dataset-size", source["dataset"]["size_bytes"])
            _option(argv, "--input-identity-audit", str(_resolved(audit["path"])))
            _option(argv, "--expected-input-identity-audit-sha256", audit["sha256"])
            _option(argv, "--expected-input-identity-audit-size", audit["size_bytes"])
            _option(argv, "--query-catalog", str(_resolved(source["catalog"]["path"])))
            _option(argv, "--expected-catalog-sha256", source["catalog"]["sha256"])
            _option(argv, "--expected-catalog-size", source["catalog"]["size_bytes"])
            _option(argv, "--device", "cuda:0")
            _option(argv, "--output", output_directory)
            preflight_argv = [*argv[:2], "--preflight", *argv[2:-2]]
            launcher_verifies = []
        elif str(cell["runner"]) == "cube_lewm_evaluator_v2":
            argv = [sys.executable, str(runner), "eval"]
            _option(argv, SWM_ROOT_FLAG, standard_runtime["root"])
            _option(argv, "--expected-ref", standard_runtime["expected_commit"])
            _option(argv, "--model", f"baseline_lewm={checkpoint['path']}")
            _option(argv, "--dataset", source["dataset"]["path"])
            _option(argv, "--expected-dataset-size", source["dataset"]["size_bytes"])
            _option(argv, "--expected-dataset-sha256", source["dataset"]["sha256"])
            _option(argv, "--query-catalog", str(_resolved(source["catalog"]["path"])))
            _option(argv, "--eval-seeds", seeds)
            _option(argv, "--num-eval", cell["queries_per_seed"])
            _option(argv, "--device", "cuda:0")
            _option(argv, "--output", output_directory)
            preflight_argv = [
                *argv[:2],
                "preflight-models",
                *argv[3:9],
            ]
            plan = standard_runtime["plan_configs"]["cube"]
            launcher_verifies = [
                {"label": "cube LeWM checkpoint", "spec": checkpoint},
                {"label": "cube LeWM loader config", "spec": config},
                {"label": "frozen cube query catalog", "spec": source["catalog"]},
                {
                    "label": "frozen cube plan config",
                    "spec": {
                        "path": str(Path(standard_runtime["root"]) / plan["path"]),
                        "sha256": plan["sha256"],
                        "size_bytes": plan["size_bytes"],
                    },
                },
            ]
        else:
            raise RuntimeError(f"Unknown runner for cell {cell_id}")
        jobs.append(
            {
                "job_id": cell_id,
                "cell_id": cell_id,
                "mujoco_gl": cell["mujoco_gl"],
                "evaluations": int(cell["evaluations"]),
                "launcher_verifies": launcher_verifies,
                "output": str(output_directory / "aggregate.json"),
                "argv": argv,
                "preflight_argv": preflight_argv,
            }
        )

    if len(jobs) != 42:
        raise RuntimeError(f"Frozen expansion produced {len(jobs)} jobs, not 42")
    if sum(int(job["evaluations"]) for job in jobs) != 5100:
        raise RuntimeError("Frozen expansion is not 5100 episodes")
    if len({job["job_id"] for job in jobs}) != 42 or len(
        {job["output"] for job in jobs}
    ) != 42:
        raise RuntimeError("Frozen job identifiers or outputs are duplicated")
    return jobs


def job_environment(job: dict[str, Any], gpu_index: int) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu_index),
            "GIT_CONFIG_GLOBAL": "/tmp/contextworld-original-baseline-gitconfig",
            "MPLCONFIGDIR": "/tmp/contextworld-original-baseline-cem-mpl",
            "MUJOCO_GL": str(job["mujoco_gl"]),
            "PYTHONHASHSEED": "0",
            "PYTHONUNBUFFERED": "1",
        }
    )
    return environment


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--list", action="store_true")
    action.add_argument("--preflight")
    action.add_argument("--job")
    parser.add_argument("--gpu-index", type=int)
    args = parser.parse_args(argv)
    if (args.job or args.preflight) and (args.gpu_index is None or args.gpu_index < 0):
        parser.error("--job/--preflight require a non-negative --gpu-index")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    prereg = load_prereg()
    jobs = build_jobs(prereg)
    if args.list:
        print(json.dumps({"job_count": len(jobs), "jobs": jobs}, indent=2, sort_keys=True))
        return
    if args.preflight:
        selected = [job for job in jobs if job["cell_id"] == args.preflight and job["preflight_argv"]]
        if len(selected) != 1:
            raise ValueError(f"Unknown or non-preflight cell: {args.preflight}")
        job = selected[0]
        for check in job["launcher_verifies"]:
            _assert_identity(check["spec"], label=check["label"])
        subprocess.run(
            job["preflight_argv"],
            cwd=ROOT,
            env=job_environment(job, args.gpu_index),
            check=True,
        )
        return
    selected = [job for job in jobs if job["job_id"] == args.job]
    if len(selected) != 1:
        raise ValueError(f"Unknown frozen job: {args.job}")
    job = selected[0]
    for check in job["launcher_verifies"]:
        _assert_identity(check["spec"], label=check["label"])
    subprocess.run(
        job["argv"], cwd=ROOT, env=job_environment(job, args.gpu_index), check=True
    )


if __name__ == "__main__":
    main()
