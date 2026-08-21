#!/usr/bin/env python3
"""Run Stable-WorldModel's original-environment MPC evaluation explicitly.

This is deliberately separate from training.  A training job may opt into it
after a successful run, or an operator may invoke it later against a frozen
checkpoint.  It is not the ContextWorld component ICL/CEM evaluator; published
benchmark results continue to use each component's registered protocol.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILE_CONFIG = REPO_ROOT / "configs/training/stablewm_family_profiles_v1.yaml"


def _env(name: str, fallback: str | None = None) -> str | None:
    value = os.environ.get(name)
    return value if value not in (None, "") else fallback


def _absolute(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise SystemExit(f"{label} must be absolute: {path}")
    return Path(os.path.abspath(path))


def _seeds(value: str) -> tuple[int, ...]:
    tokens = value.replace(",", " ").split()
    if not tokens:
        raise argparse.ArgumentTypeError("at least one eval seed is required")
    try:
        parsed = tuple(int(item) for item in tokens)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"eval seeds must be comma/space-separated integers: {value}") from exc
    if len(parsed) != len(set(parsed)):
        raise argparse.ArgumentTypeError("eval seeds must be unique")
    return parsed


def load_contract() -> dict:
    payload = yaml.safe_load(PROFILE_CONFIG.read_text(encoding="utf-8"))
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    contract = load_contract()
    parser = argparse.ArgumentParser(
        description="Evaluate one StableWM checkpoint with upstream MPC/CEM.")
    parser.add_argument("--family", choices=sorted(contract["families"]), required=True)
    parser.add_argument(
        "--original-env",
        choices=sorted(contract["original_environments"]),
        required=True,
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument(
        "--checkpoint-root",
        default=_env("CW_CHECKPOINT_ROOT", _env("STABLEWM_HOME")),
        required=_env("CW_CHECKPOINT_ROOT", _env("STABLEWM_HOME")) is None,
    )
    parser.add_argument(
        "--stablewm-repo",
        default=_env("CONTEXTWORLD_STABLE_WORLDMODEL_REPO", _env("STABLEWM_REPO")),
        required=_env("CONTEXTWORLD_STABLE_WORLDMODEL_REPO", _env("STABLEWM_REPO"))
        is None,
    )
    parser.add_argument("--num-eval", type=int, default=50)
    parser.add_argument("--eval-seeds", type=_seeds, default=(42, 43, 44))
    parser.add_argument("--history-size", type=int, default=3)
    parser.add_argument("--action-block", type=int, default=5)
    parser.add_argument("--mujoco-gl", default=_env("CW_EVAL_MUJOCO_GL", "osmesa"))
    parser.add_argument("--corruption-type", default="gaussian_noise")
    parser.add_argument("--corruption-std", type=float, default=0.0)
    parser.add_argument("--corruption-factor", type=float, default=1.0)
    parser.add_argument("--corruption-kernel-size", type=int, default=1)
    parser.add_argument("--corruption-apply-to", default="pixels")
    parser.add_argument("--keep-videos", action="store_true")
    parser.add_argument("--print-command", action="store_true")
    return parser.parse_args(argv)


def build_commands(
        args: argparse.Namespace
) -> tuple[Path, list[tuple[int, Path, Path, list[str]]]]:
    contract = load_contract()
    stablewm_repo = _absolute(args.stablewm_repo, "--stablewm-repo")
    entry = stablewm_repo / contract["evaluation"]["entrypoint"]
    if not entry.is_file():
        raise SystemExit(f"Upstream evaluator not found: {entry}")
    dataset = _absolute(args.dataset, "--dataset")
    if not dataset.exists():
        raise SystemExit(f"Evaluation dataset does not exist: {dataset}")
    checkpoint_root = _absolute(args.checkpoint_root, "--checkpoint-root")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", args.run_name):
        raise SystemExit(
            "Run names must be 1-128 characters using letters, numbers, '.', "
            "'_' or '-', and must start with a letter or number.")
    checkpoint = (checkpoint_root / "checkpoints" / args.run_name /
                  f"weights_epoch_{args.epoch}.pt")
    if not checkpoint.is_file() and not args.print_command:
        raise SystemExit(f"Evaluation checkpoint not found: {checkpoint}")
    if (args.epoch <= 0 or args.num_eval <= 0 or args.history_size <= 0
            or args.action_block <= 0 or args.corruption_kernel_size <= 0
            or args.corruption_factor <= 0 or args.corruption_std < 0):
        raise SystemExit(
            "epoch, num-eval, history-size, corruption factor/kernel must be "
            "positive, and corruption std must be non-negative")

    run_dir = checkpoint.parent
    results_dir = run_dir / "eval_results"
    policy = f"{args.run_name}/weights_epoch_{args.epoch}.pt"
    config_name = contract["evaluation"]["original_config_names"][args.original_env]
    commands = []
    for seed in args.eval_seeds:
        label = (f"{args.original_env}_{args.family}_epoch{args.epoch}_"
                 f"num{args.num_eval}_seed{seed}")
        log_path = results_dir / f"{label}.log"
        metrics_path = results_dir / f"{label}_metrics.txt"
        command = [
            sys.executable,
            str(entry),
            f"--config-name={config_name}",
            f"policy={policy}",
            f"eval.dataset_name={dataset}",
            f"eval.num_eval={args.num_eval}",
            f"seed={seed}",
            f"output.filename=eval_results/{metrics_path.name}",
            f"++plan_config.history_len={args.history_size}",
            f"++plan_config.action_block={args.action_block}",
            f"++eval.corruption.type={args.corruption_type}",
            f"++eval.corruption.std={args.corruption_std}",
            f"++eval.corruption.factor={args.corruption_factor}",
            f"++eval.corruption.kernel_size={args.corruption_kernel_size}",
            f"++eval.corruption.apply_to=[{args.corruption_apply_to}]",
        ]
        commands.append((seed, log_path, metrics_path, command))
    return stablewm_repo, commands


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    stablewm_repo, commands = build_commands(args)
    for _, log_path, metrics_path, command in commands:
        print(f"[stablewm-eval] {shlex.join(command)}")
        print(f"[stablewm-eval] log={log_path} metrics={metrics_path}")
    if args.print_command:
        return 0

    environment = dict(os.environ)
    environment["STABLEWM_HOME"] = str(
        _absolute(args.checkpoint_root, "--checkpoint-root"))
    environment["MUJOCO_GL"] = args.mujoco_gl
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(stablewm_repo), environment.get("PYTHONPATH", "")]).strip(os.pathsep)

    for seed, log_path, metrics_path, command in commands:
        if log_path.exists() or metrics_path.exists():
            raise SystemExit("Refusing to overwrite existing evaluation output: "
                             f"{log_path if log_path.exists() else metrics_path}")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        run_dir = log_path.parent.parent
        before_videos = set(run_dir.glob("*.mp4"))
        with log_path.open("x", encoding="utf-8") as log:
            completed = subprocess.run(
                command,
                cwd=str(stablewm_repo),
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode != 0:
            print(
                f"[stablewm-eval] seed {seed} failed with status "
                f"{completed.returncode}; see {log_path}",
                file=sys.stderr,
            )
            return completed.returncode
        if not metrics_path.is_file():
            raise SystemExit(
                f"Upstream eval returned success but wrote no metrics: {metrics_path}")
        if not args.keep_videos:
            for video in set(run_dir.glob("*.mp4")) - before_videos:
                video.unlink()
        receipt = {
            "schema_version": "contextworld.stablewm-original-eval.v1",
            "family": args.family,
            "original_environment": args.original_env,
            "run_name": args.run_name,
            "checkpoint_epoch": args.epoch,
            "eval_seed": seed,
            "num_eval": args.num_eval,
            "history_size": args.history_size,
            "action_block": args.action_block,
            "corruption": {
                "type": args.corruption_type,
                "std": args.corruption_std,
                "factor": args.corruption_factor,
                "kernel_size": args.corruption_kernel_size,
                "apply_to": args.corruption_apply_to,
            },
            "dataset": str(_absolute(args.dataset, "--dataset")),
            "metrics_file": metrics_path.name,
            "log_file": log_path.name,
        }
        receipt_path = metrics_path.with_suffix(".json")
        receipt_path.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"[stablewm-eval] wrote {metrics_path} and {receipt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
