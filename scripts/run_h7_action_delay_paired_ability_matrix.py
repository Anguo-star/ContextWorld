#!/usr/bin/env python3
"""Run a frozen Action Delay original-ability CEM matrix."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contextworld.evaluation.action_delay_h7_validation import file_sha256
from contextworld.paths import resolve_contextworld_path
from contextworld.synthesis.manifest import write_json


DEFAULT_CONFIG = (
    ROOT
    / "configs/benchmark/"
    "tworoom_action_delay_h7_paired_ability_retention_v1.yaml"
)
EVALUATOR = ROOT / "scripts/eval_tworoom_ability_catalog.py"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _models(config: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {**row, "role": role}
        for role, rows in config["models"].items()
        for row in rows
    ]


def _gpu_count() -> int:
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
        check=True,
        text=True,
        capture_output=True,
    )
    return len(
        [line for line in completed.stdout.splitlines() if line.strip()]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    _require(
        str(config.get("benchmark", "")).startswith(
            "tworoom_action_delay_history7_"
        ),
        "不是 History=7 Action Delay 能力保持配置",
    )
    _require(
        config.get("status") == "preregistered_before_cem_scoring",
        "CEM 协议未在评分前冻结",
    )
    for name, identity in config["source_identity"].items():
        path = resolve_contextworld_path(identity["path"], repo_root=ROOT)
        _require(path.is_file(), f"冻结来源不存在：{name}: {path}")
        _require(
            file_sha256(path) == identity["sha256"],
            f"冻结来源哈希变化：{name}",
        )

    models = _models(config)
    _require(
        len(models) == int(config["evaluation"]["expected_models"]),
        "模型数量与冻结配置不一致",
    )
    for model in models:
        checkpoint = resolve_contextworld_path(
            model["checkpoint"], repo_root=ROOT
        )
        _require(checkpoint.is_file(), f"checkpoint 不存在：{checkpoint}")
        _require(
            file_sha256(checkpoint) == model["checkpoint_sha256"],
            f"checkpoint 哈希变化：{model['slug']}",
        )

    gpu_count = _gpu_count()
    _require(gpu_count == 8, f"正式矩阵需要八张 GPU，实际 {gpu_count}")
    output_root = resolve_contextworld_path(
        config["artifacts"]["results_root"], repo_root=ROOT
    )
    log_root = output_root / "logs"
    matrix_report = resolve_contextworld_path(
        config["artifacts"]["runner_report"], repo_root=ROOT
    )
    log_root.mkdir(parents=True, exist_ok=True)
    _require(not matrix_report.exists(), f"输出已存在：{matrix_report}")

    jobs: list[dict[str, Any]] = []
    for model in models:
        checkpoint = resolve_contextworld_path(
            model["checkpoint"], repo_root=ROOT
        )
        for domain, domain_config in config["evaluation"][
            "domains"
        ].items():
            catalog = resolve_contextworld_path(
                domain_config["catalog"], repo_root=ROOT
            )
            _require(
                file_sha256(catalog) == domain_config["catalog_sha256"],
                f"Eval catalog 哈希变化：{domain}",
            )
            for eval_seed in config["evaluation"]["eval_seeds"]:
                output = (
                    output_root
                    / str(model["slug"])
                    / domain
                    / f"s{int(eval_seed)}.json"
                )
                log_path = (
                    log_root
                    / f"{model['slug']}_{domain}_s{int(eval_seed)}.log"
                )
                _require(
                    not output.exists() and not log_path.exists(),
                    f"拒绝覆盖正式输出：{output}",
                )
                jobs.append(
                    {
                        **model,
                        "domain": domain,
                        "eval_seed": int(eval_seed),
                        "checkpoint_path": checkpoint,
                        "catalog_path": catalog,
                        "output": output,
                        "log": log_path,
                    }
                )
    expected_jobs = int(config["evaluation"]["expected_jobs"])
    expected_evaluations = int(
        config["evaluation"]["expected_independent_planning_evaluations"]
    )
    _require(
        len(jobs) == expected_jobs,
        f"正式矩阵 job 数不一致，实际 {len(jobs)}",
    )
    _require(
        expected_jobs
        * int(
            config["evaluation"][
                "evaluations_per_seed_per_model_per_domain"
            ]
        )
        == expected_evaluations,
        "正式矩阵独立规划次数不一致",
    )

    planning = config["evaluation"]["planner"]
    normalizer = resolve_contextworld_path(
        config["source_identity"]["normalizer"]["path"], repo_root=ROOT
    )
    stablewm_repo = os.environ.get(
        "STABLEWM_REPO", str(config["stable_worldmodel"]["repo"])
    )
    stablewm_ref = str(config["stable_worldmodel"]["commit"])
    pending = list(jobs)
    running: dict[int, dict[str, Any]] = {}
    completed_rows: list[dict[str, Any]] = []
    matrix_started = time.monotonic()
    while pending or running:
        for gpu in [
            value for value in range(gpu_count) if value not in running
        ]:
            if not pending:
                break
            job = pending.pop(0)
            job["output"].parent.mkdir(parents=True, exist_ok=True)
            handle = job["log"].open("w", encoding="utf-8")
            command = [
                sys.executable,
                str(EVALUATOR),
                "--catalog",
                str(job["catalog_path"]),
                "--checkpoint",
                str(job["checkpoint_path"]),
                "--normalizer",
                str(normalizer),
                "--output",
                str(job["output"]),
                "--seed",
                str(job["eval_seed"]),
                "--stablewm-repo",
                stablewm_repo,
                "--stablewm-ref",
                stablewm_ref,
                "--device",
                "cuda:0",
                "--expected-history-size",
                str(job["history_size"]),
                "--eval-budget",
                str(planning["eval_budget_raw_steps"]),
                "--horizon",
                str(planning["horizon_action_blocks"]),
                "--receding-horizon",
                str(planning["receding_horizon_action_blocks"]),
                "--cem-samples",
                str(planning["cem_samples"]),
                "--cem-steps",
                str(planning["cem_steps"]),
                "--cem-topk",
                str(planning["cem_topk"]),
            ]
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env={
                    **os.environ,
                    "CUDA_VISIBLE_DEVICES": str(gpu),
                    "OMP_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "NUMEXPR_NUM_THREADS": "1",
                },
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            running[gpu] = {
                **job,
                "process": process,
                "handle": handle,
                "started": time.monotonic(),
            }
            print(
                f"[paired-ability] start {job['slug']}/{job['domain']}/"
                f"s{job['eval_seed']} gpu={gpu}",
                flush=True,
            )

        time.sleep(max(float(args.poll_seconds), 0.25))
        for gpu, job in list(running.items()):
            returncode = job["process"].poll()
            if returncode is None:
                continue
            job["handle"].close()
            elapsed = time.monotonic() - job["started"]
            if returncode:
                tail = "\n".join(
                    job["log"]
                    .read_text(encoding="utf-8", errors="replace")
                    .splitlines()[-100:]
                )
                raise RuntimeError(
                    f"CEM 评测失败：{job['slug']}/{job['domain']}/"
                    f"s{job['eval_seed']}\n{tail}"
                )
            payload = json.loads(
                job["output"].read_text(encoding="utf-8")
            )
            protocol = payload.get("protocol", {})
            _require(
                payload.get("status") == "passed"
                and int(protocol.get("history_size", -1))
                == int(job["history_size"])
                and int(protocol.get("evaluations", -1)) == 50
                and payload.get("frozen_weight_audit", {}).get("passed")
                is True,
                f"CEM 结果审计失败：{job['output']}",
            )
            completed_rows.append(
                {
                    "slug": job["slug"],
                    "role": job["role"],
                    "domain": job["domain"],
                    "eval_seed": job["eval_seed"],
                    "gpu": gpu,
                    "elapsed_seconds": elapsed,
                    "output": str(job["output"]),
                    "output_sha256": file_sha256(job["output"]),
                    "status": "passed",
                }
            )
            del running[gpu]
            print(
                f"[paired-ability] completed {len(completed_rows)}/"
                f"{len(jobs)} gpu={gpu}",
                flush=True,
            )

    write_json(
        matrix_report,
        {
            "schema_version": 1,
            "benchmark": config["benchmark"],
            "status": "passed",
            "jobs": len(jobs),
            "independent_planning_evaluations": expected_evaluations,
            "gpus": gpu_count,
            "stable_worldmodel_commit": stablewm_ref,
            "elapsed_seconds": time.monotonic() - matrix_started,
            "results": completed_rows,
        },
    )
    print(
        f"[paired-ability] all jobs completed: {matrix_report}", flush=True
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
