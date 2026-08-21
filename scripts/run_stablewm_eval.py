#!/usr/bin/env python3
"""Evaluate one Stable-WorldModel checkpoint, now or after training.

The same command serves two workflows:

* without ``--suite``, run the upstream original-environment MPC/CEM check;
* with ``--suite``, run every applicable evaluation for one training target:
  the matching original-environment CEM check plus the registered ContextWorld
  ICL scorer(s).

The suite is an orchestrator, not a second implementation of any metric.  It
reuses Stable-WorldModel's planner and ContextWorld's existing task scorers,
stores all outputs beside the checkpoint, and never writes a scoreboard row.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from contextworld.synthesis.manifest import write_json  # noqa: E402


PROFILE_CONFIG = REPO_ROOT / "configs/training/stablewm_family_profiles_v1.yaml"
DEVELOPMENT_ONLY_COMPONENTS = {"contact_friction", "motion_damping"}


def _env(name: str, fallback: str | None = None) -> str | None:
    value = os.environ.get(name)
    return value if value not in (None, "") else fallback


def _env_int(name: str) -> int | None:
    value = _env(name)
    return int(value) if value is not None else None


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
            f"eval seeds must be comma/space-separated integers: {value}"
        ) from exc
    if any(seed < 0 for seed in parsed):
        raise argparse.ArgumentTypeError("eval seeds must be non-negative")
    if len(parsed) != len(set(parsed)):
        raise argparse.ArgumentTypeError("eval seeds must be unique")
    return parsed


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_contract() -> dict[str, Any]:
    payload = yaml.safe_load(PROFILE_CONFIG.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"Invalid StableWM profile contract: {PROFILE_CONFIG}")
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    contract = load_contract()
    default_eval_seeds = ",".join(
        str(seed) for seed in contract["evaluation"]["default_seeds"]
    )
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one StableWM checkpoint with original CEM and, when "
            "--suite is selected, applicable ContextWorld ICL scorers."
        )
    )
    parser.add_argument(
        "--family", choices=sorted(contract["families"]), required=True
    )
    parser.add_argument(
        "--original-env", choices=sorted(contract["original_environments"])
    )
    parser.add_argument(
        "--component", choices=sorted(contract["benchmark_components"])
    )
    parser.add_argument(
        "--suite",
        action="store_true",
        help=(
            "Run the applicable original-environment CEM and benchmark ICL "
            "set. Without this flag, only original CEM is run."
        ),
    )
    parser.add_argument(
        "--dataset",
        default=_env("CW_EVAL_ORIGINAL_DATASET"),
        help="Exact original-environment dataset used by MPC/CEM.",
    )
    parser.add_argument(
        "--checkpoint",
        default=_env("CW_EVAL_CHECKPOINT"),
        help=(
            "Exact .pt/.ckpt checkpoint. If omitted, derive weights_epoch_N.pt "
            "from --checkpoint-root, --run-name and --epoch."
        ),
    )
    parser.add_argument("--run-name", default=_env("CW_RUN_NAME"))
    parser.add_argument("--epoch", type=int, default=_env_int("CW_EVAL_EPOCH"))
    parser.add_argument(
        "--checkpoint-root",
        default=_env("CW_CHECKPOINT_ROOT", _env("STABLEWM_HOME")),
    )
    parser.add_argument(
        "--stablewm-repo",
        default=_env(
            "CONTEXTWORLD_STABLE_WORLDMODEL_REPO", _env("STABLEWM_REPO")
        ),
    )
    parser.add_argument("--stablewm-ref", default=_env("CW_STABLEWM_REF"))
    parser.add_argument(
        "--result-subdir",
        default="",
        help=(
            "Optional safe namespace below the checkpoint's eval_results/. "
            "In suite mode it contains the manifest and every suite output."
        ),
    )
    parser.add_argument("--training-seed", type=int)
    parser.add_argument("--training-recipe")
    parser.add_argument(
        "--num-eval",
        type=int,
        default=int(
            _env("CW_EVAL_NUM", str(contract["evaluation"]["default_num_eval"]))
        ),
    )
    parser.add_argument(
        "--eval-seeds",
        type=_seeds,
        default=_seeds(_env("CW_EVAL_SEEDS", default_eval_seeds)),
    )
    parser.add_argument("--history-size", type=int, default=3)
    parser.add_argument("--action-block", type=int, default=5)
    parser.add_argument("--eval-device", default=_env("CW_EVAL_DEVICE", "cuda:0"))
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=int(_env("CW_EVAL_BATCH_SIZE", "64")),
    )
    parser.add_argument("--mujoco-gl", default=_env("CW_EVAL_MUJOCO_GL", "osmesa"))
    parser.add_argument("--corruption-type", default="gaussian_noise")
    parser.add_argument("--corruption-std", type=float, default=0.0)
    parser.add_argument("--corruption-factor", type=float, default=1.0)
    parser.add_argument("--corruption-kernel-size", type=int, default=1)
    parser.add_argument("--corruption-apply-to", default="pixels")
    parser.add_argument("--keep-videos", action="store_true")
    parser.add_argument("--print-command", action="store_true")
    return parser.parse_args(argv)


@dataclass(frozen=True)
class ResolvedCheckpoint:
    path: Path
    run_name: str
    epoch: int | None
    checkpoint_root: Path
    policy: str


def _run_name(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", value):
        raise SystemExit(
            "Run names must be 1-128 characters using letters, numbers, '.', "
            "'_' or '-', and must start with a letter or number."
        )
    return value


def _resolve_checkpoint(args: argparse.Namespace) -> ResolvedCheckpoint:
    if args.checkpoint:
        checkpoint = _absolute(args.checkpoint, "--checkpoint")
        run_name = _run_name(args.run_name or checkpoint.parent.name)
        if args.checkpoint_root:
            checkpoint_root = _absolute(args.checkpoint_root, "--checkpoint-root")
        elif checkpoint.parent.parent.name == "checkpoints":
            checkpoint_root = checkpoint.parent.parent.parent
        else:
            # Absolute policies do not depend on STABLEWM_HOME for loading.
            # Keep generated videos and cache resolution local to this run.
            checkpoint_root = checkpoint.parent
        policy = str(checkpoint)
        epoch = args.epoch
    else:
        missing = [
            name
            for name, value in (
                ("--checkpoint-root", args.checkpoint_root),
                ("--run-name", args.run_name),
                ("--epoch", args.epoch),
            )
            if value is None
        ]
        if missing:
            raise SystemExit(
                "Checkpoint resolution needs --checkpoint or all of "
                "--checkpoint-root, --run-name and --epoch; missing "
                + ", ".join(missing)
            )
        checkpoint_root = _absolute(args.checkpoint_root, "--checkpoint-root")
        run_name = _run_name(args.run_name)
        epoch = int(args.epoch)
        checkpoint = (
            checkpoint_root
            / "checkpoints"
            / run_name
            / f"weights_epoch_{epoch}.pt"
        )
        policy = f"{run_name}/weights_epoch_{epoch}.pt"
    if epoch is not None and epoch <= 0:
        raise SystemExit("--epoch must be positive")
    if not checkpoint.is_file() and not args.print_command:
        raise SystemExit(f"Evaluation checkpoint not found: {checkpoint}")
    return ResolvedCheckpoint(
        path=checkpoint,
        run_name=run_name,
        epoch=epoch,
        checkpoint_root=checkpoint_root,
        policy=policy,
    )


def _safe_subdir(value: str) -> Path:
    if not value:
        return Path()
    path = Path(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise SystemExit(f"--result-subdir must be a safe relative path: {value}")
    return path


def _stablewm_repo(args: argparse.Namespace) -> Path:
    if not args.stablewm_repo:
        raise SystemExit(
            "Evaluation needs --stablewm-repo or "
            "CONTEXTWORLD_STABLE_WORLDMODEL_REPO."
        )
    repo = _absolute(args.stablewm_repo, "--stablewm-repo")
    if not (repo / "scripts/train").is_dir():
        raise SystemExit(f"Stable-WorldModel checkout is invalid: {repo}")
    return repo


def _stablewm_ref(repo: Path, explicit: str | None) -> str:
    if explicit:
        return explicit
    completed = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    revision = completed.stdout.strip()
    if completed.returncode != 0 or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise SystemExit(
            "Could not identify the Stable-WorldModel checkout revision; pass "
            "--stablewm-ref explicitly."
        )
    return revision


def build_commands(
    args: argparse.Namespace,
) -> tuple[Path, list[tuple[int, Path, Path, list[str]]]]:
    """Build the original-environment MPC/CEM commands."""

    contract = load_contract()
    if not args.original_env:
        raise SystemExit("Original CEM needs --original-env")
    stablewm_repo = _stablewm_repo(args)
    entry = stablewm_repo / contract["evaluation"]["entrypoint"]
    if not entry.is_file():
        raise SystemExit(f"Upstream evaluator not found: {entry}")
    if not args.dataset:
        raise SystemExit("Original CEM needs --dataset")
    dataset = _absolute(args.dataset, "--dataset")
    if not dataset.exists():
        raise SystemExit(f"Evaluation dataset does not exist: {dataset}")
    checkpoint = _resolve_checkpoint(args)
    if (
        args.num_eval <= 0
        or args.history_size <= 0
        or args.action_block <= 0
        or args.corruption_kernel_size <= 0
        or args.corruption_factor <= 0
        or args.corruption_std < 0
    ):
        raise SystemExit(
            "num-eval, history-size, action-block, corruption factor/kernel "
            "must be positive, and corruption std must be non-negative"
        )

    run_dir = checkpoint.path.parent
    results_dir = run_dir / "eval_results" / _safe_subdir(args.result_subdir)
    config_name = contract["evaluation"]["original_config_names"][args.original_env]
    checkpoint_label = (
        f"epoch{checkpoint.epoch}" if checkpoint.epoch is not None else checkpoint.path.stem
    )
    commands = []
    for seed in args.eval_seeds:
        label = (
            f"{args.original_env}_{args.family}_{checkpoint_label}_"
            f"num{args.num_eval}_seed{seed}"
        )
        log_path = results_dir / f"{label}.log"
        metrics_path = results_dir / f"{label}_metrics.txt"
        metrics_relative = metrics_path.relative_to(run_dir).as_posix()
        command = [
            sys.executable,
            str(entry),
            f"--config-name={config_name}",
            f"policy={checkpoint.policy}",
            f"eval.dataset_name={dataset}",
            f"eval.num_eval={args.num_eval}",
            f"seed={seed}",
            f"output.filename={metrics_relative}",
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


def _print_original_commands(
    commands: list[tuple[int, Path, Path, list[str]]]
) -> None:
    for _, log_path, metrics_path, command in commands:
        print(f"[stablewm-eval] original-cem: {shlex.join(command)}")
        print(f"[stablewm-eval] log={log_path} metrics={metrics_path}")


def _run_original(args: argparse.Namespace) -> int:
    stablewm_repo, commands = build_commands(args)
    _print_original_commands(commands)
    if args.print_command:
        return 0

    checkpoint = _resolve_checkpoint(args)
    environment = dict(os.environ)
    environment["STABLEWM_HOME"] = str(checkpoint.checkpoint_root)
    environment["MUJOCO_GL"] = args.mujoco_gl
    environment["PYTHONPATH"] = os.pathsep.join(
        [
            str(REPO_ROOT),
            str(stablewm_repo),
            environment.get("PYTHONPATH", ""),
        ]
    ).strip(os.pathsep)

    for _, log_path, metrics_path, _ in commands:
        receipt_path = metrics_path.with_suffix(".json")
        for output in (log_path, metrics_path, receipt_path):
            if output.exists():
                raise SystemExit(
                    f"Refusing to overwrite existing evaluation output: {output}"
                )

    for seed, log_path, metrics_path, command in commands:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        run_dir = checkpoint.path.parent
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
                f"Upstream eval returned success but wrote no metrics: {metrics_path}"
            )
        if not args.keep_videos:
            for video in set(run_dir.glob("*.mp4")) - before_videos:
                video.unlink()
        receipt = {
            "schema_version": "contextworld.stablewm-original-eval.v2",
            "family": args.family,
            "original_environment": args.original_env,
            "run_name": checkpoint.run_name,
            "checkpoint": str(checkpoint.path),
            "checkpoint_epoch": checkpoint.epoch,
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
            "metrics_file": str(metrics_path),
            "log_file": str(log_path),
        }
        receipt_path = metrics_path.with_suffix(".json")
        write_json(receipt_path, receipt)
        print(f"[stablewm-eval] wrote {metrics_path} and {receipt_path}")
    return 0


@dataclass(frozen=True)
class ICLStep:
    component: str
    output: Path
    command: list[str]


def _suite_environment(args: argparse.Namespace, contract: dict[str, Any]) -> str:
    if bool(args.original_env) == bool(args.component):
        raise SystemExit(
            "--suite requires exactly one training target: --original-env or "
            "--component"
        )
    if args.original_env:
        return args.original_env
    return str(contract["benchmark_components"][args.component]["environment"])


def _suite_components(
    args: argparse.Namespace, contract: dict[str, Any], environment: str
) -> tuple[str, ...]:
    if args.component:
        return (args.component,)
    return tuple(
        component
        for component, specification in contract["benchmark_components"].items()
        if specification["environment"] == environment
    )


def _build_icl_steps(
    args: argparse.Namespace,
    *,
    checkpoint: ResolvedCheckpoint,
    stablewm_repo: Path,
    stablewm_ref: str,
    components: tuple[str, ...],
    eval_root: Path,
) -> list[ICLStep]:
    recipe = args.training_recipe or (
        f"original_{args.original_env}"
        if args.original_env
        else f"contextworld_{args.component}"
    )
    root = eval_root / "benchmark_icl"
    steps = []
    for component in components:
        output = root / component / "result.json"
        command = [
            sys.executable,
            "-m",
            "contextworld.benchmarks.external_model_cli",
            "--task",
            component,
            "--adapter",
            args.family,
            "--checkpoint",
            str(checkpoint.path),
            "--model-name",
            checkpoint.run_name,
            "--training-recipe",
            recipe,
            "--device",
            args.eval_device,
            "--batch-size",
            str(args.eval_batch_size),
            "--stablewm-repo",
            str(stablewm_repo),
            "--stablewm-ref",
            stablewm_ref,
            "--output",
            str(output),
        ]
        if component in DEVELOPMENT_ONLY_COMPONENTS:
            command.extend(("--evaluation-split", "development"))
        if args.training_seed is not None:
            command.extend(("--training-seed", str(args.training_seed)))
        steps.append(ICLStep(component=component, output=output, command=command))
    return steps


def _output_inventory(eval_root: Path, manifest_path: Path) -> list[dict[str, Any]]:
    rows = []
    if not eval_root.is_dir():
        return rows
    for path in sorted(eval_root.rglob("*")):
        if not path.is_file() or path == manifest_path:
            continue
        rows.append(
            {
                "path": path.relative_to(eval_root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
        )
    return rows


def _run_suite(args: argparse.Namespace) -> int:
    contract = load_contract()
    environment_name = _suite_environment(args, contract)
    components = _suite_components(args, contract, environment_name)
    checkpoint = _resolve_checkpoint(args)
    stablewm_repo = _stablewm_repo(args)
    stablewm_ref = _stablewm_ref(stablewm_repo, args.stablewm_ref)
    suite_subdir = _safe_subdir(args.result_subdir)
    eval_root = checkpoint.path.parent / "eval_results" / suite_subdir
    icl_steps = _build_icl_steps(
        args,
        checkpoint=checkpoint,
        stablewm_repo=stablewm_repo,
        stablewm_ref=stablewm_ref,
        components=components,
        eval_root=eval_root,
    )

    cem_args = argparse.Namespace(**vars(args))
    cem_args.original_env = environment_name
    cem_args.component = None
    cem_args.result_subdir = str(
        suite_subdir
        / (
            Path("original_cem")
            if args.original_env
            else Path("benchmark_cem") / str(args.component)
        )
    )
    cem_commands: list[tuple[int, Path, Path, list[str]]] = []
    cem_skip_reason = None
    if args.dataset:
        _, cem_commands = build_commands(cem_args)
    elif args.original_env:
        raise SystemExit(
            "An original-environment evaluation suite needs --dataset or "
            "CW_EVAL_ORIGINAL_DATASET; CEM cannot be skipped for its primary "
            "target."
        )
    else:
        cem_skip_reason = (
            "original-environment dataset was not supplied; set "
            "--dataset/CW_EVAL_ORIGINAL_DATASET to enable CEM retention"
        )

    print(
        f"[stablewm-eval] suite target="
        f"{args.original_env or args.component} family={args.family} "
        f"checkpoint={checkpoint.path}"
    )
    if cem_skip_reason:
        print(f"[stablewm-eval] original-cem skipped: {cem_skip_reason}")
    else:
        _print_original_commands(cem_commands)
    for step in icl_steps:
        print(
            f"[stablewm-eval] benchmark-icl component={step.component}: "
            f"{shlex.join(step.command)}"
        )
        print(f"[stablewm-eval] output={step.output}")
    if args.print_command:
        return 0

    manifest_path = eval_root / "manifest.json"
    if manifest_path.exists():
        raise SystemExit(
            f"Refusing to overwrite existing evaluation manifest: {manifest_path}"
        )
    for step in icl_steps:
        if step.output.exists():
            raise SystemExit(
                f"Refusing to overwrite existing ICL result: {step.output}"
            )

    manifest: dict[str, Any] = {
        "schema_version": "contextworld.stablewm-evaluation-suite.v1",
        "status": "running",
        "target": {
            "kind": "original_environment" if args.original_env else "component",
            "name": args.original_env or args.component,
            "original_environment": environment_name,
        },
        "model": {
            "family": args.family,
            "run_name": checkpoint.run_name,
            "training_seed": args.training_seed,
            "checkpoint": str(checkpoint.path),
            "checkpoint_sha256": _file_sha256(checkpoint.path),
            "stablewm_repo": str(stablewm_repo),
            "stablewm_ref": stablewm_ref,
        },
        "steps": [],
        "outputs": [],
    }
    cem_record: dict[str, Any] = {
        "id": "original_cem",
        "kind": "original_environment_cem",
        "status": "skipped" if cem_skip_reason else "planned",
        "reason": cem_skip_reason,
        "commands": [command for *_, command in cem_commands],
    }
    manifest["steps"].append(cem_record)
    for step in icl_steps:
        manifest["steps"].append(
            {
                "id": f"benchmark_icl/{step.component}",
                "kind": "benchmark_icl",
                "component": step.component,
                "status": "planned",
                "command": step.command,
                "output": str(step.output),
                "official_scoreboard_row": False,
            }
        )
    write_json(manifest_path, manifest)

    try:
        failure_code = 0
        if not cem_skip_reason:
            status = _run_original(cem_args)
            cem_record["status"] = "completed" if status == 0 else "failed"
            cem_record["returncode"] = status
            manifest["outputs"] = _output_inventory(eval_root, manifest_path)
            write_json(manifest_path, manifest)
            if status != 0:
                failure_code = status

        environment = dict(os.environ)
        environment["PYTHONPATH"] = os.pathsep.join(
            [
                str(REPO_ROOT),
                str(stablewm_repo),
                environment.get("PYTHONPATH", ""),
            ]
        ).strip(os.pathsep)
        for index, step in enumerate(icl_steps, start=1):
            print(
                f"[stablewm-eval] ICL {index}/{len(icl_steps)}: "
                f"{step.component}",
                flush=True,
            )
            completed = subprocess.run(
                step.command,
                cwd=str(REPO_ROOT),
                env=environment,
                check=False,
            )
            record = next(
                row
                for row in manifest["steps"]
                if row["id"] == f"benchmark_icl/{step.component}"
            )
            record["status"] = (
                "completed" if completed.returncode == 0 else "failed"
            )
            record["returncode"] = completed.returncode
            manifest["outputs"] = _output_inventory(eval_root, manifest_path)
            write_json(manifest_path, manifest)
            if completed.returncode != 0 and failure_code == 0:
                failure_code = completed.returncode
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        manifest["outputs"] = _output_inventory(eval_root, manifest_path)
        write_json(manifest_path, manifest)
        raise

    manifest["status"] = "failed" if failure_code != 0 else "completed"
    manifest["outputs"] = _output_inventory(eval_root, manifest_path)
    write_json(manifest_path, manifest)
    print(f"[stablewm-eval] suite manifest={manifest_path}")
    return failure_code


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.suite:
        return _run_suite(args)
    if args.component:
        raise SystemExit("--component is only valid with --suite")
    return _run_original(args)


if __name__ == "__main__":
    raise SystemExit(main())
