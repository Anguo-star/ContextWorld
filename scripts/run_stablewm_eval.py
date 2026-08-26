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
import json
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from contextworld.synthesis.manifest import write_json  # noqa: E402
from contextworld.synthesis.stablewm import _git_commit  # noqa: E402


PROFILE_CONFIG = REPO_ROOT / "configs/training/stablewm_family_profiles_v1.yaml"
DEVELOPMENT_ONLY_COMPONENTS = {"contact_friction", "motion_damping"}
SUITE_MANIFEST_SCHEMA = "contextworld.stablewm-evaluation-suite.v3"
STRICT_ICL_PROTOCOL_TRACK = "strict_frozen_v1"
DIAGNOSTIC_ICL_PROTOCOL_TRACK = "diagnostic_normalized_zero_v1"
ORIGINAL_CEM_PROTOCOL_TRACK = "original_cem_v1"


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


def _json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    """Create the initial suite reservation without an overwrite race."""

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as exc:
        raise SystemExit(
            f"Another evaluation already reserved this manifest: {path}"
        ) from exc


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
        "--icl-only",
        action="store_true",
        help=(
            "With --suite, skip original-environment CEM and run only the "
            "registered ICL tracks. This permits an original-environment "
            "checkpoint to be scored without an original CEM dataset."
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


def _prejepa_model_config(checkpoint: Path) -> dict[str, Any] | None:
    """Read the native PreJEPA model config beside a ``.pt`` checkpoint.

    Stable-WorldModel's native save format always places ``config.json`` next
    to the weights.  Dry-run commands are also allowed to name a checkpoint
    that does not exist yet, so absence returns ``None`` and lets profile
    defaults drive command rendering.
    """

    path = checkpoint.parent / "config.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Could not read PreJEPA model config {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"PreJEPA model config must be a JSON object: {path}")
    model = payload.get("model", payload)
    if not isinstance(model, dict):
        raise SystemExit(f"PreJEPA model entry must be a JSON object: {path}")
    return model


def _prejepa_extra_encoder_keys(model_config: dict[str, Any]) -> tuple[str, ...]:
    extra = model_config.get("extra_encoders", {})
    if not isinstance(extra, dict):
        return ()
    modules = extra.get("modules", extra)
    if not isinstance(modules, dict):
        return ()
    return tuple(str(key) for key in modules if not str(key).startswith("_"))


def _prejepa_state_keys(
    *,
    checkpoint: ResolvedCheckpoint,
    original_env: str,
    contract: dict[str, Any],
) -> tuple[str, ...]:
    model_config = _prejepa_model_config(checkpoint.path)
    if model_config is None:
        fallback = contract["original_environments"][original_env].get(
            "encoding_key"
        )
        return (str(fallback),) if fallback else ()
    return tuple(
        key
        for key in _prejepa_extra_encoder_keys(model_config)
        if key != "action"
    )


def _validate_prejepa_cem_geometry(
    *,
    checkpoint: ResolvedCheckpoint,
    original_env: str,
    history_size: int,
    action_block: int,
    contract: dict[str, Any],
) -> None:
    """Reject planner geometry that differs from the trained checkpoint."""

    model_config = _prejepa_model_config(checkpoint.path)
    if model_config is None:
        return
    trained_history = model_config.get("history_size")
    if trained_history is None:
        raise SystemExit(
            "PreJEPA checkpoint config does not declare history_size: "
            f"{checkpoint.path.parent / 'config.json'}"
        )
    try:
        trained_history = int(trained_history)
    except (TypeError, ValueError) as exc:
        raise SystemExit(
            f"Invalid PreJEPA checkpoint history_size: {trained_history!r}"
        ) from exc
    if trained_history != history_size:
        raise SystemExit(
            f"PreJEPA checkpoint was trained with history_size={trained_history}, "
            f"but CEM requested history_size={history_size}."
        )

    extra = model_config.get("extra_encoders", {})
    modules = extra.get("modules", extra) if isinstance(extra, dict) else {}
    action = modules.get("action", {}) if isinstance(modules, dict) else {}
    width = action.get("in_chans") if isinstance(action, dict) else None
    if width is None:
        return
    raw_action_dim = int(
        contract["original_environments"][original_env]["action_dim"]
    )
    try:
        width = int(width)
    except (TypeError, ValueError) as exc:
        raise SystemExit(
            f"Invalid PreJEPA action encoder width: {width!r}"
        ) from exc
    if width <= 0 or width % raw_action_dim:
        raise SystemExit(
            "PreJEPA action encoder width is incompatible with the original "
            f"environment: width={width}, action_dim={raw_action_dim}."
        )
    trained_action_block = width // raw_action_dim
    if trained_action_block != action_block:
        raise SystemExit(
            "PreJEPA checkpoint was trained with action_block="
            f"{trained_action_block}, but CEM requested action_block="
            f"{action_block}."
        )


def _prejepa_objective_overrides(
    *,
    state_keys: tuple[str, ...],
) -> list[str]:
    """Select SWM's split-latent goal objective for a PreJEPA checkpoint.

    A PreJEPA prediction contains pixels, state and action slots.  A goal has
    no action slot, so the generic fused ``goal_mse`` objective is
    dimensionally invalid.  Stable-WorldModel already exposes the correct
    per-source ``WeightedSum`` objective; this function only selects it and,
    for Reacher/Cube, renames the state stream from ``proprio`` to the key
    recorded by the checkpoint.
    """

    if not state_keys:
        return ["objective=goal_mse_pixels"]
    if len(state_keys) != 1:
        raise SystemExit(
            "PreJEPA CEM currently supports zero or one non-action latent "
            f"source; checkpoint declares {state_keys}."
        )

    state_key = state_keys[0]
    overrides = ["objective=goal_mse_pixels_proprio"]
    if state_key != "proprio":
        overrides.extend(
            [
                f"objective.terms.0.1.pred_key=predicted_{state_key}_emb",
                f"objective.terms.0.1.goal_key={state_key}_goal_emb",
            ]
        )
    return overrides


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
        if not re.fullmatch(r"[0-9a-f]{40}", explicit):
            raise SystemExit("--stablewm-ref/CW_STABLEWM_REF must be a 40-digit SHA")
        return explicit

    # Dataset/code mounts are commonly owned by a different uid than the
    # training process. ``git rev-parse`` then refuses the checkout as an
    # unsafe directory even though its metadata is readable. The benchmark
    # adapters already resolve HEAD directly for exactly this reason, so use
    # the same implementation during the suite hand-off.
    try:
        revision = _git_commit(repo)
    except (OSError, RuntimeError) as exc:
        raise SystemExit(
            "Could not read the Stable-WorldModel checkout revision from "
            f"{repo}/.git. Keep Git metadata with the cloud checkout; an "
            "explicit CW_STABLEWM_REF alone cannot verify a source tree that "
            "has no Git metadata."
        ) from exc
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise SystemExit(
            f"Stable-WorldModel HEAD is not a 40-digit SHA: {revision!r}"
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
        f"epoch{checkpoint.epoch}"
        if checkpoint.epoch is not None
        else checkpoint.path.stem
    )
    prejepa_state_keys: tuple[str, ...] = ()
    if args.family == "prejepa":
        _validate_prejepa_cem_geometry(
            checkpoint=checkpoint,
            original_env=args.original_env,
            history_size=args.history_size,
            action_block=args.action_block,
            contract=contract,
        )
        prejepa_state_keys = _prejepa_state_keys(
            checkpoint=checkpoint,
            original_env=args.original_env,
            contract=contract,
        )
        history_keys = ",".join(("pixels", *prejepa_state_keys))
        command_prefix = [
            sys.executable,
            str(REPO_ROOT / "scripts/run_stablewm_plan.py"),
            "--upstream-entry",
            str(entry),
            "--history-keys",
            history_keys,
        ]
    else:
        command_prefix = [sys.executable, str(entry)]
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
            *command_prefix,
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
        if args.family == "prejepa":
            command.extend(
                _prejepa_objective_overrides(state_keys=prejepa_state_keys)
            )
            # Reacher and Cube train PreJEPA's state stream from the
            # ``observation`` column.  Their historical planner configs cache
            # only actions, which would leave both current and goal state
            # unstandardized.  The upstream evaluator already builds the
            # required normalizers for every listed key.
            if prejepa_state_keys == ("observation",):
                command.append("dataset.keys_to_cache=[action,observation]")
        commands.append((seed, log_path, metrics_path, command))
    return stablewm_repo, commands


def _print_original_commands(
    commands: list[tuple[int, Path, Path, list[str]]]
) -> None:
    for _, log_path, metrics_path, command in commands:
        print(f"[stablewm-eval] original-cem: {shlex.join(command)}")
        print(f"[stablewm-eval] log={log_path} metrics={metrics_path}")


def _parse_original_metrics(path: Path, *, num_eval: int) -> dict[str, Any]:
    """Turn the upstream human-readable CEM result into typed evidence."""

    text = path.read_text(encoding="utf-8")
    number = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
    success = re.search(
        rf"['\"]success_rate['\"]\s*:\s*({number})",
        text,
    )
    duration = re.search(rf"evaluation_time:\s*({number})\s*seconds", text)
    if success is None or duration is None:
        raise SystemExit(
            "Could not parse success_rate and evaluation_time from upstream "
            f"metrics: {path}"
        )
    success_rate = float(success.group(1))
    evaluation_time = float(duration.group(1))
    if not 0.0 <= success_rate <= 100.0 or evaluation_time < 0.0:
        raise SystemExit(f"Upstream metrics are outside valid bounds: {path}")
    successful_episodes = success_rate * num_eval / 100.0
    rounded_successes = round(successful_episodes)
    if abs(successful_episodes - rounded_successes) > 1e-6:
        raise SystemExit(
            "Upstream success_rate is inconsistent with eval.num_eval: "
            f"rate={success_rate}, num_eval={num_eval}, path={path}"
        )
    return {
        "success_rate_percent": success_rate,
        "successful_episodes": rounded_successes,
        "evaluation_time_seconds": evaluation_time,
    }


def _failure_returncode(exc: BaseException) -> int:
    """Map a local evaluator exception to a subprocess-like failure code."""

    code = getattr(exc, "code", None)
    if isinstance(code, int) and code != 0:
        return code
    return 1


def _run_original(
    args: argparse.Namespace,
    *,
    on_seed_result: Callable[[dict[str, Any]], None] | None = None,
) -> int:
    """Run every requested original CEM seed and aggregate failures.

    A CEM seed is an independent evaluation cell.  One infrastructure or
    metric-parsing failure must not hide the evidence from subsequent seeds,
    nor must it prevent the suite from reaching its ICL tracks.  ``on_seed``
    persists each terminal cell state in the enclosing suite manifest.
    """

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

    failure_code = 0
    for seed, log_path, metrics_path, command in commands:
        receipt_path = metrics_path.with_suffix(".json")
        result: dict[str, Any] = {
            "eval_seed": seed,
            "log_file": str(log_path),
            "metrics_file": str(metrics_path),
            "receipt_file": str(receipt_path),
        }
        returncode = 0
        try:
            for output in (log_path, metrics_path, receipt_path):
                if output.exists() or output.is_symlink():
                    raise FileExistsError(
                        "Refusing to overwrite existing evaluation output: "
                        f"{output}"
                    )
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
            returncode = int(completed.returncode)
            if returncode != 0:
                print(
                    f"[stablewm-eval] seed {seed} failed with status "
                    f"{returncode}; see {log_path}",
                    file=sys.stderr,
                )
                result.update(status="failed", returncode=returncode)
            else:
                if not metrics_path.is_file() or metrics_path.is_symlink():
                    raise RuntimeError(
                        "Upstream eval returned success without a regular metrics "
                        f"file: {metrics_path}"
                    )
                if not args.keep_videos:
                    for video in set(run_dir.glob("*.mp4")) - before_videos:
                        video.unlink()
                parsed_metrics = _parse_original_metrics(
                    metrics_path,
                    num_eval=args.num_eval,
                )
                receipt = {
                    "schema_version": "contextworld.stablewm-original-eval.v3",
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
                    "metrics": parsed_metrics,
                }
                write_json(receipt_path, receipt)
                print(f"[stablewm-eval] wrote {metrics_path} and {receipt_path}")
                result.update(status="completed", returncode=0)
        except (KeyboardInterrupt, GeneratorExit):
            raise
        except BaseException as exc:
            returncode = _failure_returncode(exc)
            result.update(
                status="failed",
                returncode=returncode,
                error={
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
            )
            print(
                f"[stablewm-eval] seed {seed} failed before a valid receipt: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
        finally:
            if on_seed_result is not None:
                on_seed_result(result)
        if returncode != 0 and failure_code == 0:
            failure_code = returncode
    return failure_code


@dataclass(frozen=True)
class ICLStep:
    identifier: str
    kind: str
    protocol_track: str
    component: str
    output: Path
    command: list[str]
    skip_reason: str | None = None


def _prejepa_icl_skip_reason(
    *,
    checkpoint: ResolvedCheckpoint,
    component: str,
    contract: dict[str, Any],
) -> str | None:
    """Return why a native PreJEPA checkpoint cannot enter the v1 scorer.

    ContextWorld v1 deliberately exposes only RGB history and actions to a
    model adapter.  A PreJEPA checkpoint trained with an additional state
    stream cannot be evaluated faithfully without that stream: zero filling
    changes the trained input, while passing simulator state would silently
    widen the frozen public protocol.  State-free checkpoints remain runnable.
    """

    model_config = _prejepa_model_config(checkpoint.path)
    if model_config is None:
        return None

    reasons = []
    state_keys = tuple(
        key
        for key in _prejepa_extra_encoder_keys(model_config)
        if key != "action"
    )
    if state_keys:
        reasons.append(
            "checkpoint requires context stream(s) "
            f"{list(state_keys)}, but the frozen v1 ICL adapter contract "
            "provides only pixels and actions"
        )

    required_history = int(
        contract["benchmark_components"][component]["history_size"]
    )
    trained_history = model_config.get("history_size")
    if trained_history is not None and int(trained_history) != required_history:
        reasons.append(
            f"checkpoint history_size={int(trained_history)} does not match "
            f"the component's frozen History={required_history} protocol"
        )
    return "; ".join(reasons) if reasons else None


def _prejepa_checkpoint_history_size(
    checkpoint: ResolvedCheckpoint,
) -> int | None:
    """Return the native history geometry when the checkpoint records it."""

    model_config = _prejepa_model_config(checkpoint.path)
    if model_config is None:
        return None
    value = model_config.get("history_size")
    if value is None:
        return None
    try:
        history_size = int(value)
    except (TypeError, ValueError) as exc:
        raise SystemExit(
            f"Invalid PreJEPA checkpoint history_size: {value!r}"
        ) from exc
    if history_size <= 0:
        raise SystemExit(
            f"Invalid PreJEPA checkpoint history_size: {value!r}"
        )
    return history_size


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
    contract: dict[str, Any],
) -> list[ICLStep]:
    recipe = args.training_recipe or (
        f"original_{args.original_env}"
        if args.original_env
        else f"contextworld_{args.component}"
    )
    strict_root = eval_root / "benchmark_icl"
    diagnostic_root = eval_root / "benchmark_icl_diagnostic"
    steps = []
    for component in components:
        output = strict_root / component / "result.json"
        skip_reason = (
            _prejepa_icl_skip_reason(
                checkpoint=checkpoint,
                component=component,
                contract=contract,
            )
            if args.family == "prejepa"
            else None
        )
        model_config = (
            _prejepa_model_config(checkpoint.path)
            if args.family == "prejepa"
            else None
        )
        state_conditioned = bool(
            model_config
            and any(
                key != "action" for key in _prejepa_extra_encoder_keys(model_config)
            )
        )
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
        steps.append(
            ICLStep(
                identifier=f"benchmark_icl/{component}",
                kind="benchmark_icl",
                protocol_track=STRICT_ICL_PROTOCOL_TRACK,
                component=component,
                output=output,
                command=command,
                skip_reason=skip_reason,
            )
        )

        # State-conditioned PreJEPA checkpoints cannot enter the frozen ICL
        # interface.  Preserve that strict result, but also schedule a
        # separately labelled diagnostic track that makes its zero-context
        # imputation explicit.  The diagnostic output can never be confused
        # with the strict frozen-protocol score because its id, path and
        # protocol track are all disjoint.
        if args.family != "prejepa" or not skip_reason or not state_conditioned:
            continue
        diagnostic_output = diagnostic_root / component / "result.json"
        diagnostic_command = list(command)
        diagnostic_command[diagnostic_command.index("--output") + 1] = str(
            diagnostic_output
        )
        diagnostic_command.extend(
            ("--prejepa-missing-context-policy", "normalized_zero")
        )
        if (
            component == "action_delay"
            and _prejepa_checkpoint_history_size(checkpoint) == 3
        ):
            diagnostic_command.extend(
                ("--history-adapter", "h3_tail_projection")
            )
        steps.append(
            ICLStep(
                identifier=f"benchmark_icl_diagnostic/{component}",
                kind="benchmark_icl_diagnostic",
                protocol_track=DIAGNOSTIC_ICL_PROTOCOL_TRACK,
                component=component,
                output=diagnostic_output,
                command=diagnostic_command,
            )
        )
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


def _suite_request(
    args: argparse.Namespace,
    *,
    checkpoint: ResolvedCheckpoint,
    stablewm_repo: Path,
    stablewm_ref: str,
    environment_name: str,
    components: tuple[str, ...],
    suite_subdir: Path,
    cem_commands: list[tuple[int, Path, Path, list[str]]],
    cem_skip_reason: str | None,
    icl_steps: list[ICLStep],
) -> dict[str, Any]:
    """Describe every input that can change a suite result.

    The request is stored with a completed manifest.  A later invocation may
    reuse that manifest only when this complete document and every recorded
    output byte are unchanged.
    """

    return {
        "schema_version": "contextworld.stablewm-evaluation-request.v1",
        "target": {
            "kind": "original_environment" if args.original_env else "component",
            "name": args.original_env or args.component,
            "original_environment": environment_name,
            "components": list(components),
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
        "evaluation": {
            "dataset": (
                str(_absolute(args.dataset, "--dataset")) if args.dataset else None
            ),
            "icl_only": args.icl_only,
            "result_subdir": suite_subdir.as_posix(),
            "training_recipe": args.training_recipe,
            "num_eval": args.num_eval,
            "eval_seeds": list(args.eval_seeds),
            "history_size": args.history_size,
            "action_block": args.action_block,
            "eval_device": args.eval_device,
            "eval_batch_size": args.eval_batch_size,
            "mujoco_gl": args.mujoco_gl,
            "keep_videos": args.keep_videos,
            "corruption": {
                "type": args.corruption_type,
                "std": args.corruption_std,
                "factor": args.corruption_factor,
                "kernel_size": args.corruption_kernel_size,
                "apply_to": args.corruption_apply_to,
            },
        },
        "steps": {
            "original_cem": {
                "skip_reason": cem_skip_reason,
                "commands": [command for *_, command in cem_commands],
            },
            "benchmark_icl": [
                {
                    "id": step.identifier,
                    "kind": step.kind,
                    "protocol_track": step.protocol_track,
                    "component": step.component,
                    "skip_reason": step.skip_reason,
                    "command": step.command,
                    "output": str(step.output),
                }
                for step in icl_steps
            ],
        },
    }


def _reuse_completed_suite(
    *,
    eval_root: Path,
    manifest_path: Path,
    expected_request: dict[str, Any],
) -> None:
    """Accept an immutable completed suite, otherwise fail closed."""

    if manifest_path.is_symlink():
        raise SystemExit(
            f"Refusing symlinked evaluation manifest: {manifest_path}"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(
            f"Could not validate existing evaluation manifest {manifest_path}: {exc}"
        ) from exc
    if not isinstance(manifest, dict):
        raise SystemExit(
            f"Existing evaluation manifest is not a JSON object: {manifest_path}"
        )

    expected_sha256 = _json_sha256(expected_request)
    if (
        manifest.get("schema_version") != SUITE_MANIFEST_SCHEMA
        or manifest.get("status") != "completed"
        or manifest.get("request") != expected_request
        or manifest.get("request_sha256") != expected_sha256
    ):
        raise SystemExit(
            "Refusing to overwrite or reuse an evaluation manifest whose "
            f"request is incomplete, failed, or different: {manifest_path}"
        )

    steps = manifest.get("steps")
    if not isinstance(steps, list) or any(
        not isinstance(step, dict)
        or step.get("status") not in {"completed", "not_compatible", "skipped"}
        or not isinstance(step.get("id"), str)
        or not step["id"]
        or not isinstance(step.get("protocol_track"), str)
        or not step["protocol_track"]
        or (
            step.get("status") == "completed"
            and step.get("returncode") != 0
        )
        for step in steps
    ):
        raise SystemExit(
            f"Completed evaluation manifest has unfinished steps: {manifest_path}"
        )
    step_ids = [step["id"] for step in steps]
    if len(step_ids) != len(set(step_ids)):
        raise SystemExit(
            f"Completed evaluation manifest has duplicate step ids: {manifest_path}"
        )
    for path in eval_root.rglob("*"):
        if path.is_symlink():
            raise SystemExit(
                f"Completed evaluation output contains a symlink: {path}"
            )
    if manifest.get("outputs") != _output_inventory(eval_root, manifest_path):
        raise SystemExit(
            "Completed evaluation outputs no longer match their recorded "
            f"size/SHA256 inventory: {manifest_path}"
        )

    print(
        "[stablewm-eval] exact completed suite already exists; "
        f"reusing immutable manifest={manifest_path}"
    )


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
        contract=contract,
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
    if args.icl_only:
        cem_skip_reason = "ICL-only requested"
    elif args.dataset:
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
        if step.skip_reason:
            print(
                f"[stablewm-eval] {step.kind} "
                f"component={step.component} track={step.protocol_track} "
                f"not compatible: {step.skip_reason}"
            )
        else:
            print(
                f"[stablewm-eval] {step.kind} "
                f"component={step.component} track={step.protocol_track}: "
                f"{shlex.join(step.command)}"
            )
            print(f"[stablewm-eval] output={step.output}")
    if args.print_command:
        return 0

    manifest_path = eval_root / "manifest.json"
    request = _suite_request(
        args,
        checkpoint=checkpoint,
        stablewm_repo=stablewm_repo,
        stablewm_ref=stablewm_ref,
        environment_name=environment_name,
        components=components,
        suite_subdir=suite_subdir,
        cem_commands=cem_commands,
        cem_skip_reason=cem_skip_reason,
        icl_steps=icl_steps,
    )
    if manifest_path.exists():
        _reuse_completed_suite(
            eval_root=eval_root,
            manifest_path=manifest_path,
            expected_request=request,
        )
        return 0
    for step in icl_steps:
        if not step.skip_reason and step.output.exists():
            raise SystemExit(
                f"Refusing to overwrite existing ICL result: {step.output}"
            )

    manifest: dict[str, Any] = {
        "schema_version": SUITE_MANIFEST_SCHEMA,
        "status": "running",
        "request": request,
        "request_sha256": _json_sha256(request),
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
    cem_records: dict[int, dict[str, Any]] = {}
    if cem_skip_reason:
        manifest["steps"].append(
            {
                "id": "original_cem",
                "kind": "original_environment_cem",
                "protocol_track": ORIGINAL_CEM_PROTOCOL_TRACK,
                "status": "skipped",
                "reason": cem_skip_reason,
                "commands": [],
            }
        )
    else:
        for seed, log_path, metrics_path, command in cem_commands:
            record = {
                "id": f"original_cem/seed{seed}",
                "kind": "original_environment_cem",
                "protocol_track": ORIGINAL_CEM_PROTOCOL_TRACK,
                "eval_seed": seed,
                "status": "planned",
                "command": command,
                "log_file": str(log_path),
                "metrics_file": str(metrics_path),
                "receipt_file": str(metrics_path.with_suffix(".json")),
            }
            cem_records[seed] = record
            manifest["steps"].append(record)

    icl_records: dict[str, dict[str, Any]] = {}
    for step in icl_steps:
        record = {
            "id": step.identifier,
            "kind": step.kind,
            "protocol_track": step.protocol_track,
            "component": step.component,
            "status": "not_compatible" if step.skip_reason else "planned",
            "reason": step.skip_reason,
            "command": step.command,
            "output": str(step.output),
            "official_scoreboard_row": False,
        }
        icl_records[step.identifier] = record
        manifest["steps"].append(record)

    step_ids = [str(row["id"]) for row in manifest["steps"]]
    if len(step_ids) != len(set(step_ids)):
        raise SystemExit(
            "Evaluation suite produced duplicate manifest step ids: "
            + ", ".join(step_ids)
        )
    if any("protocol_track" not in row for row in manifest["steps"]):
        raise SystemExit("Every evaluation manifest step needs a protocol_track")
    _write_json_exclusive(manifest_path, manifest)

    def persist_manifest() -> None:
        write_json(manifest_path, manifest)

    try:
        failure_code = 0
        if not cem_skip_reason:
            reported_cem_seeds: set[int] = set()

            def record_cem_result(result: dict[str, Any]) -> None:
                seed = int(result["eval_seed"])
                record = cem_records[seed]
                record.update(result)
                reported_cem_seeds.add(seed)
                persist_manifest()

            try:
                status = _run_original(
                    cem_args,
                    on_seed_result=record_cem_result,
                )
            except (KeyboardInterrupt, GeneratorExit):
                raise
            except BaseException as exc:
                status = _failure_returncode(exc)
                for seed, record in cem_records.items():
                    if seed not in reported_cem_seeds:
                        record.update(
                            status="failed",
                            returncode=status,
                            error={
                                "type": type(exc).__name__,
                                "message": str(exc),
                            },
                        )
                persist_manifest()

            # Keep the private helper easy to monkeypatch in focused tests and
            # still leave every manifest step terminal if an older shim does
            # not invoke ``on_seed_result``.
            for seed, record in cem_records.items():
                if seed in reported_cem_seeds:
                    continue
                record.update(
                    status="completed" if status == 0 else "failed",
                    returncode=status,
                )
            persist_manifest()
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
            if step.skip_reason:
                continue
            print(
                f"[stablewm-eval] ICL {index}/{len(icl_steps)} "
                f"track={step.protocol_track}: {step.component}",
                flush=True,
            )
            record = icl_records[step.identifier]
            returncode = 0
            try:
                completed = subprocess.run(
                    step.command,
                    cwd=str(REPO_ROOT),
                    env=environment,
                    check=False,
                )
                returncode = int(completed.returncode)
                if returncode == 0 and (
                    not step.output.is_file() or step.output.is_symlink()
                ):
                    raise RuntimeError(
                        "ICL evaluator returned success without a regular result "
                        f"file: {step.output}"
                    )
                record["status"] = "completed" if returncode == 0 else "failed"
                record["returncode"] = returncode
            except (KeyboardInterrupt, GeneratorExit):
                raise
            except BaseException as exc:
                returncode = _failure_returncode(exc)
                record.update(
                    status="failed",
                    returncode=returncode,
                    error={
                        "type": type(exc).__name__,
                        "message": str(exc),
                    },
                )
                print(
                    f"[stablewm-eval] ICL {step.identifier} failed before a "
                    f"valid result: {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
            persist_manifest()
            if returncode != 0 and failure_code == 0:
                failure_code = returncode
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
    if args.icl_only and not args.suite:
        raise SystemExit("--icl-only is only valid with --suite")
    if args.suite:
        return _run_suite(args)
    if args.component:
        raise SystemExit("--component is only valid with --suite")
    return _run_original(args)


if __name__ == "__main__":
    raise SystemExit(main())
