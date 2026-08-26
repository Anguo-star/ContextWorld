#!/usr/bin/env python3
"""Run an upstream StableWM trainer with portable SPT resume wiring.

Stable-WorldModel remains the owner of the trainer, model, objective and
optimizer.  This bootstrap only supplies two pieces of orchestration that the
three upstream family entrypoints do not expose consistently:

* identify the StablePretraining UUID run that belongs to a ContextWorld run;
* on a newly submitted job, pass its ``last.ckpt`` back to ``spt.Manager``
  with ``weights_only=False``.

The same-job SLURM requeue path is untouched and remains owned by
StablePretraining.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


RUN_MARKER_FILENAME = "contextworld_run_identity_v1.json"
RUN_MARKER_SCHEMA = "contextworld.stablepretraining-run-identity.v1"


def _prepare_optional_flash_attention() -> bool:
    """Keep a broken optional FlashAttention install from blocking SPT.

    Kornia can use PyTorch's scaled-dot-product attention when FlashAttention
    is absent.  Some shared cloud images nevertheless contain an older
    ``flash-attn`` extension compiled against a different PyTorch ABI.  Kornia
    treats an absent package as optional, but that ABI failure is an
    ``ImportError`` and otherwise aborts StablePretraining during import.

    Return ``True`` only when an unusable installation was masked.  A working
    FlashAttention installation is left untouched.
    """

    try:
        importlib.import_module("flash_attn.modules.mha")
    except ModuleNotFoundError:
        return False
    except (ImportError, OSError) as exc:
        for module_name in tuple(sys.modules):
            if module_name == "flash_attn" or module_name.startswith(
                "flash_attn."
            ):
                sys.modules.pop(module_name, None)
        # ``None`` makes Kornia's optional import raise ModuleNotFoundError,
        # activating its supported PyTorch SDPA fallback.
        sys.modules["flash_attn"] = None
        sys.stderr.write(
            "ContextWorld: optional flash-attn is not loadable "
            f"({type(exc).__name__}); using PyTorch attention instead.\n"
        )
        return True


def _write_run_marker(
    run_dir: Path,
    *,
    run_name: str,
    identity_sha256: str,
    require_existing: bool = False,
) -> None:
    """Bind one StablePretraining UUID directory to one immutable recipe."""

    payload = {
        "schema_version": RUN_MARKER_SCHEMA,
        "run_name": run_name,
        "training_identity_sha256": identity_sha256,
    }
    path = run_dir / RUN_MARKER_FILENAME
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        try:
            observed = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Could not read StablePretraining run marker {path}"
            ) from exc
        if observed != payload:
            raise RuntimeError(
                "StablePretraining run directory is already bound to a "
                f"different ContextWorld recipe: {path}"
            )
        return

    if require_existing:
        raise RuntimeError(
            "StablePretraining requeue selected a run without its immutable "
            f"ContextWorld identity marker: {path}"
        )

    run_dir.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        # All DDP ranks resolve the same UUID directory and write the same
        # payload. Atomic replacement keeps partial JSON hidden and avoids
        # hard links, which are not supported by some cloud mounts.
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _install_manager_bridge(
    *,
    run_name: str,
    identity_sha256: str,
    resume_checkpoint: Path | None,
) -> None:
    """Wrap ``spt.Manager`` without replacing the upstream training loop."""

    _prepare_optional_flash_attention()
    import stable_pretraining as spt

    original_manager = spt.Manager
    parameters = inspect.signature(original_manager.__init__).parameters
    if "weights_only" not in parameters or not hasattr(
        original_manager, "_resolve_run_dir"
    ):
        raise RuntimeError(
            "Installed stable-pretraining does not expose the full-state "
            "Manager interface required by ContextWorld"
        )

    class ContextWorldManager(original_manager):  # type: ignore[misc, valid-type]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            if resume_checkpoint is not None:
                # PreJEPA and PLDM currently pass their inference-weight path
                # with Manager's default weights-only semantics.  A portable
                # SPT last.ckpt is a Lightning trainer checkpoint, so it must
                # replace that path and restore the complete state.
                kwargs["ckpt_path"] = str(resume_checkpoint)
                kwargs["weights_only"] = False
            super().__init__(*args, **kwargs)

        def _resolve_run_dir(self) -> Path | None:
            resolved = super()._resolve_run_dir()
            if resolved is not None:
                try:
                    restart_count = int(
                        os.environ.get("SLURM_RESTART_COUNT", "0")
                    )
                except ValueError as exc:
                    raise RuntimeError(
                        "SLURM_RESTART_COUNT must be an integer"
                    ) from exc
                native_requeue = bool(os.environ.get("SLURM_JOB_ID")) and (
                    restart_count >= 1
                    and not getattr(self, "_early_preempt_fallback", False)
                )
                _write_run_marker(
                    Path(resolved),
                    run_name=run_name,
                    identity_sha256=identity_sha256,
                    require_existing=native_requeue,
                )
            return resolved

    ContextWorldManager.__name__ = original_manager.__name__
    ContextWorldManager.__qualname__ = original_manager.__name__
    ContextWorldManager.__module__ = __name__
    # submitit serializes its Checkpointable callable during requeue. Register
    # the dynamically configured subclass under the module/name advertised to
    # pickle; sitecustomize recreates the same binding in the replacement
    # interpreter before submitit deserializes the callable.
    globals()[ContextWorldManager.__name__] = ContextWorldManager
    spt.Manager = ContextWorldManager


def _parse_args(
    argv: list[str] | None = None,
) -> tuple[argparse.Namespace, list[str]]:
    values = list(sys.argv[1:] if argv is None else argv)
    try:
        separator = values.index("--")
    except ValueError as exc:
        raise SystemExit(
            "StableWM family entry requires '--' before Hydra arguments"
        ) from exc
    own, hydra_args = values[:separator], values[separator + 1 :]
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainer-script", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--identity-sha256", required=True)
    parser.add_argument("--resume-checkpoint", type=Path)
    return parser.parse_args(own), hydra_args


def main(argv: list[str] | None = None) -> int:
    args, hydra_args = _parse_args(argv)
    trainer_script = args.trainer_script.expanduser().resolve()
    if not trainer_script.is_file():
        raise SystemExit(
            f"StableWM trainer script does not exist: {trainer_script}"
        )

    resume_checkpoint = None
    if args.resume_checkpoint is not None:
        resume_checkpoint = args.resume_checkpoint.expanduser().resolve()
        if (
            not resume_checkpoint.is_file()
            or resume_checkpoint.stat().st_size <= 0
        ):
            raise SystemExit(
                "Portable StablePretraining checkpoint is missing or empty: "
                f"{resume_checkpoint}"
            )

    repo_root = Path(__file__).resolve().parents[1]
    bootstrap_dir = repo_root / "scripts/stablewm_bootstrap"
    environment = dict(os.environ)
    environment["CONTEXTWORLD_SPT_BRIDGE"] = "1"
    environment["CONTEXTWORLD_SPT_RUN_NAME"] = args.run_name
    environment["CONTEXTWORLD_SPT_IDENTITY_SHA256"] = args.identity_sha256
    if any("contextworld://v1/" in value for value in hydra_args):
        # sitecustomize runs before the upstream trainer imports its data
        # package, including in every Lightning DDP child interpreter.
        environment["CONTEXTWORLD_STABLEWM_BUNDLE"] = "1"
        # Lance/PyArrow is not fork-safe.  DataLoader workers for the public
        # bundle therefore start as clean Python interpreters.
        environment["CONTEXTWORLD_DATALOADER_START_METHOD"] = "spawn"
    if resume_checkpoint is not None:
        environment["CONTEXTWORLD_SPT_RESUME_CHECKPOINT"] = str(resume_checkpoint)
    environment["PYTHONPATH"] = os.pathsep.join([
        str(bootstrap_dir),
        str(repo_root),
        environment.get("PYTHONPATH", ""),
    ]).strip(os.pathsep)
    completed = subprocess.run(
        [sys.executable, str(trainer_script), *hydra_args],
        cwd=str(trainer_script.parents[2]),
        env=environment,
        check=False,
    )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
