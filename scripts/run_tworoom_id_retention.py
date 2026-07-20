#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from contextworld.paths import artifact_path, resolve_contextworld_path
from contextworld.synthesis.manifest import write_json
from contextworld.synthesis.stablewm import load_stable_worldmodel


PINNED_STABLEWM = "5864b74980f6ed328fd0045e777b3865962eff43"
SUCCESS_RATE_PATTERN = re.compile(r"metrics:\s*\{'success_rate':\s*([0-9.]+)")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stablewm_policy_location(checkpoint: Path) -> tuple[Path, str]:
    checkpoint = checkpoint.resolve()
    checkpoint_dir = checkpoint.parent
    checkpoints_dir = checkpoint_dir.parent
    if checkpoints_dir.name != "checkpoints":
        raise ValueError(
            "StableWM native eval expects <STABLEWM_HOME>/checkpoints/<run>/<weights.pt>; "
            f"got {checkpoint}"
        )
    stablewm_home = checkpoints_dir.parent
    policy = str(checkpoint.relative_to(checkpoints_dir))
    return stablewm_home, policy


def _prepare_native_eval_home(
    checkpoint: Path, output_dir: Path, seed: int
) -> tuple[Path, str]:
    """Expose one immutable checkpoint through a seed-local StableWM cache.

    StableWM's native evaluator writes ``env_*.mp4`` beside the selected
    policy.  A seed-local cache preserves its lookup and output semantics while
    preventing parallel seeds from racing on the same video filenames or
    adding evaluation files to the training checkpoint directory.
    """

    _, policy = _stablewm_policy_location(checkpoint)
    checkpoint = checkpoint.resolve()
    config = checkpoint.parent / "config.json"
    if not config.is_file():
        raise FileNotFoundError(config)
    native_home = output_dir / "native_stablewm" / f"seed_{int(seed)}"
    native_run = native_home / "checkpoints" / checkpoint.parent.name
    native_run.mkdir(parents=True, exist_ok=True)
    for source in (checkpoint, config):
        destination = native_run / source.name
        if destination.exists() or destination.is_symlink():
            if destination.resolve() != source.resolve():
                raise FileExistsError(
                    f"Native eval cache path points elsewhere: {destination}"
                )
        else:
            destination.symlink_to(source)
    return native_home, policy


def _parse_success_rate(path: Path) -> float:
    match = SUCCESS_RATE_PATTERN.search(path.read_text(encoding="utf-8"))
    if match is None:
        raise ValueError(f"No StableWM success_rate found in {path}")
    return float(match.group(1))


def _command(
    *,
    python: str,
    stable_repo: Path,
    policy: str,
    original_h5: Path,
    seed: int,
    num_eval: int,
    metrics: Path,
    hydra_dir: Path,
) -> list[str]:
    return [
        python,
        str(stable_repo / "scripts/plan/eval_wm.py"),
        "--config-name=tworoom",
        f"seed={seed}",
        f"policy={policy}",
        f"eval.num_eval={num_eval}",
        f"eval.dataset_name={original_h5}",
        "eval.goal_offset_steps=25",
        "eval.eval_budget=50",
        "plan_config.horizon=5",
        "plan_config.receding_horizon=5",
        "plan_config.action_block=5",
        "solver.num_samples=300",
        "solver.n_steps=30",
        "solver.topk=30",
        f"output.filename={metrics}",
        f"hydra.run.dir={hydra_dir}",
        "hydra.output_subdir=null",
    ]


def run(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = resolve_contextworld_path(args.checkpoint, repo_root=REPO_ROOT)
    original_h5 = resolve_contextworld_path(args.original_h5, repo_root=REPO_ROOT)
    output_dir = resolve_contextworld_path(args.output_dir, repo_root=REPO_ROOT)
    output = resolve_contextworld_path(args.output, repo_root=REPO_ROOT)
    if not checkpoint.is_file() or not original_h5.is_file():
        raise FileNotFoundError(checkpoint if not checkpoint.is_file() else original_h5)
    _, stable_repo, stable_commit = load_stable_worldmodel(
        REPO_ROOT, args.stablewm_repo, args.stablewm_ref
    )
    _stablewm_policy_location(checkpoint)
    seeds = [int(value) for value in args.eval_seeds]
    devices = [str(value) for value in args.devices]
    if len(set(seeds)) != len(seeds) or not devices:
        raise ValueError("Eval seeds must be unique and at least one device is required")

    output_dir.mkdir(parents=True, exist_ok=True)
    pending: list[dict[str, Any]] = []
    completed: list[dict[str, Any]] = []
    for index, seed in enumerate(seeds):
        metrics = output_dir / f"formal_n{args.num_eval}_seed{seed}_metrics.txt"
        log = output_dir / f"formal_n{args.num_eval}_seed{seed}.log"
        native_home, policy = _prepare_native_eval_home(
            checkpoint, output_dir, seed
        )
        if args.reuse_existing and metrics.is_file():
            completed.append(
                {
                    "seed": seed,
                    "metrics": metrics,
                    "log": log,
                    "device": None,
                    "native_home": native_home,
                    "policy": policy,
                }
            )
            continue
        if metrics.exists() or (log.exists() and not args.reuse_existing):
            raise FileExistsError(
                f"Refusing to append/overwrite existing ID output for seed {seed}: {metrics}"
            )
        pending.append(
            {
                "seed": seed,
                "metrics": metrics,
                "log": log,
                "device": devices[index % len(devices)],
                "native_home": native_home,
                "policy": policy,
            }
        )

    env_base = os.environ.copy()
    # ``eval_wm.py`` is executed as a script, so Python otherwise resolves the
    # globally installed package before the pinned checkout.  Prepending the
    # checkout keeps the native evaluator and its library implementation on
    # the same StableWorldModel commit.
    pythonpath = env_base.get("PYTHONPATH")
    env_base["PYTHONPATH"] = (
        f"{stable_repo}{os.pathsep}{pythonpath}" if pythonpath else str(stable_repo)
    )
    env_base.setdefault("MUJOCO_GL", "egl")
    while pending:
        wave = pending[: len(devices)]
        pending = pending[len(devices) :]
        processes = []
        for item in wave:
            hydra_dir = output_dir / f"hydra_seed{item['seed']}"
            command = _command(
                python=args.python,
                stable_repo=stable_repo,
                policy=item["policy"],
                original_h5=original_h5,
                seed=item["seed"],
                num_eval=args.num_eval,
                metrics=item["metrics"],
                hydra_dir=hydra_dir,
            )
            env = env_base.copy()
            env["STABLEWM_HOME"] = str(item["native_home"])
            env["CUDA_VISIBLE_DEVICES"] = item["device"]
            handle = item["log"].open("w", encoding="utf-8")
            process = subprocess.Popen(
                command,
                cwd=stable_repo,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            processes.append((item, command, process, handle))
        failures = []
        for item, command, process, handle in processes:
            returncode = process.wait()
            handle.close()
            if returncode:
                failures.append(
                    {
                        "seed": item["seed"],
                        "returncode": returncode,
                        "command": command,
                        "log": str(item["log"]),
                    }
                )
            else:
                completed.append(item)
        if failures:
            raise RuntimeError(f"StableWM ID eval failures: {failures}")

    completed.sort(key=lambda value: value["seed"])
    seed_rates = {
        str(item["seed"]): _parse_success_rate(item["metrics"])
        for item in completed
    }
    successes = sum(
        round(rate * args.num_eval / 100.0) for rate in seed_rates.values()
    )
    total = args.num_eval * len(seeds)
    rates = list(seed_rates.values())
    mean = sum(rates) / len(rates)
    variance = sum((value - mean) ** 2 for value in rates) / len(rates)
    payload = {
        "schema_version": 1,
        "benchmark": "contextworld_tworoom_history3_id_retention_v1",
        "status": "passed",
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": _sha256(checkpoint),
            "serialization": "stablewm_pretrained",
            "history_size": 3,
            "action_block": 5,
        },
        "runtime": {
            "implementation": "stable_worldmodel.scripts.plan.eval_wm",
            "repo": str(stable_repo),
            "commit": stable_commit,
        },
        "protocol": {
            "eval_seeds": seeds,
            "num_eval_per_seed": args.num_eval,
            "total_evaluations": total,
            "dataset": str(original_h5),
            "goal_offset_steps": 25,
            "eval_budget": 50,
            "horizon": 5,
            "receding_horizon": 5,
            "cem_num_samples": 300,
            "cem_steps": 30,
            "cem_topk": 30,
            "parallel_devices": devices,
        },
        "aggregate": {
            "successes": successes,
            "evaluations": total,
            "pooled_success_rate": 100.0 * successes / total,
            "mean_seed_success_rate": mean,
            "std_seed_success_rate": math.sqrt(variance),
            "sem_seed_success_rate": math.sqrt(variance) / math.sqrt(len(rates)),
            "seed_success_rates": seed_rates,
        },
        "raw_results": [
            {
                "seed": item["seed"],
                "metrics": str(item["metrics"]),
                "log": str(item["log"]),
                "native_stablewm_home": str(item["native_home"]),
            }
            for item in completed
        ],
        "interpretation": {
            "id_retention_only": True,
            "ood_context_adaptation_inferred": False,
        },
    }
    write_json(output, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run StableWM-native original TwoRoom ID retention on one checkpoint"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--original-h5",
        type=Path,
        default=Path(
            "/opt/huawei/explorer-env/dataset/ag_data/data/world_model/quentinll/"
            "lewm-tworooms/tworoom.h5"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46, 47])
    parser.add_argument("--num-eval", type=int, default=50)
    parser.add_argument("--devices", nargs="+", default=["0", "1", "2", "3"])
    parser.add_argument("--stablewm-repo", default="../stable-worldmodel")
    parser.add_argument("--stablewm-ref", default=PINNED_STABLEWM)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--reuse-existing", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps(result["aggregate"], sort_keys=True))
