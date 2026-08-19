#!/usr/bin/env python3
"""Run paired History-7 original-ability retention jobs on eight GPUs."""

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
    / "configs/benchmark/tworoom_action_delay_h7_ability_retention_v1.yaml"
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


def _checkpoint(config: dict[str, Any], slug: str) -> Path:
    return resolve_contextworld_path(
        str(config["model_artifact_pattern"]["checkpoint"]).format(
            slug=slug
        ),
        repo_root=ROOT,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    _require(
        config.get("benchmark")
        == "tworoom_action_delay_history7_ability_retention_v1",
        "不是 History=7 能力保持配置",
    )
    for name, identity in config["source_identity"].items():
        path = resolve_contextworld_path(identity["path"], repo_root=ROOT)
        _require(
            file_sha256(path) == identity["sha256"],
            f"冻结来源哈希变化：{name}",
        )
    primary = json.loads(
        resolve_contextworld_path(
            config["source_identity"]["primary_prediction_summary"][
                "path"
            ],
            repo_root=ROOT,
        ).read_text(encoding="utf-8")
    )
    _require(
        primary["primary_prediction_gate"]["passed"] is True,
        "主预测门未通过，按冻结顺序不运行能力保持",
    )
    gpu_count = _gpu_count()
    _require(gpu_count == 8, f"需要八张 GPU，实际 {gpu_count}")
    output_root = resolve_contextworld_path(
        config["artifacts"]["results_root"], repo_root=ROOT
    )
    log_root = output_root / "logs"
    matrix_report = output_root / "runner_report.json"
    log_root.mkdir(parents=True, exist_ok=True)
    _require(not matrix_report.exists(), f"输出已存在：{matrix_report}")

    jobs = []
    for model in _models(config):
        slug = str(model["slug"])
        checkpoint = _checkpoint(config, slug)
        _require(checkpoint.is_file(), f"checkpoint 不存在：{checkpoint}")
        for domain, domain_config in config["evaluation"][
            "domains"
        ].items():
            catalog = resolve_contextworld_path(
                domain_config["catalog"], repo_root=ROOT
            )
            for eval_seed in config["evaluation"]["eval_seeds"]:
                output = (
                    output_root
                    / slug
                    / domain
                    / f"s{int(eval_seed)}.json"
                )
                log_path = (
                    log_root
                    / f"{slug}_{domain}_s{int(eval_seed)}.log"
                )
                _require(
                    not output.exists() and not log_path.exists(),
                    f"拒绝覆盖正式能力保持输出：{output}",
                )
                jobs.append(
                    {
                        "slug": slug,
                        "domain": domain,
                        "eval_seed": int(eval_seed),
                        "checkpoint": checkpoint,
                        "catalog": catalog,
                        "output": output,
                        "log": log_path,
                    }
                )
    expected_jobs = (
        len(_models(config))
        * len(config["evaluation"]["domains"])
        * len(config["evaluation"]["eval_seeds"])
    )
    _require(len(jobs) == expected_jobs == 108, "能力保持必须为 108 个 job")

    planning = config["evaluation"]["planner"]
    normalizer = resolve_contextworld_path(
        config["source_identity"]["normalizer"]["path"],
        repo_root=ROOT,
    )
    stablewm_repo = os.environ.get(
        "STABLEWM_REPO", str(config["stable_worldmodel"]["repo"])
    )
    stablewm_ref = str(config["stable_worldmodel"]["commit"])
    pending = list(jobs)
    running: dict[int, dict[str, Any]] = {}
    completed_rows = []
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
                "python",
                str(EVALUATOR),
                "--catalog",
                str(job["catalog"]),
                "--checkpoint",
                str(job["checkpoint"]),
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
                "7",
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
                f"[h7-ability] start {job['slug']}/{job['domain']}/"
                f"s{job['eval_seed']} gpu={gpu}",
                flush=True,
            )

        time.sleep(10)
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
                    f"能力保持失败：{job['slug']}/{job['domain']}/"
                    f"s{job['eval_seed']}\n{tail}"
                )
            payload = json.loads(
                job["output"].read_text(encoding="utf-8")
            )
            protocol = payload.get("protocol", {})
            _require(
                payload.get("status") == "passed"
                and int(protocol.get("history_size", -1)) == 7
                and int(protocol.get("evaluations", -1)) == 50,
                f"能力保持结果审计失败：{job['output']}",
            )
            completed_rows.append(
                {
                    "slug": job["slug"],
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
                f"[h7-ability] completed {len(completed_rows)}/"
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
            "gpus": gpu_count,
            "elapsed_seconds": time.monotonic() - matrix_started,
            "results": completed_rows,
        },
    )
    print(f"[h7-ability] all jobs completed: {matrix_report}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
