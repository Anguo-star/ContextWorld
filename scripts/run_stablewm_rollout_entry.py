#!/usr/bin/env python3
"""Import an upstream trainer and install ContextWorld rollout supervision."""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from contextworld.training.stablewm_rollout import (
    ROLLOUT_CONTEXT_MODES,
    install_rollout_forward,
)


_TRAINER_ENV = "CONTEXTWORLD_ROLLOUT_TRAINER_SCRIPT"
_FAMILY_ENV = "CONTEXTWORLD_ROLLOUT_FAMILY"
_STEPS_ENV = "CONTEXTWORLD_ROLLOUT_STEPS"
_CONTEXT_ENV = "CONTEXTWORLD_ROLLOUT_CONTEXT"
_HYDRA_MAIN_MODULE_ENV = "HYDRA_MAIN_MODULE"


def _parse_args(
    argv: list[str] | None = None,
) -> tuple[argparse.Namespace, list[str]]:
    values = list(sys.argv[1:] if argv is None else argv)
    if "--" in values:
        separator = values.index("--")
        own, hydra_args = values[:separator], values[separator + 1 :]
    else:
        # Lightning's subprocess DDP launcher re-enters the current script
        # with only Hydra's arguments.  The rank-zero invocation records the
        # immutable overlay selection in inherited environment variables so
        # every child installs the same forward before composing its config.
        own, hydra_args = [], values
    parser = argparse.ArgumentParser()
    trainer_default = os.environ.get(_TRAINER_ENV)
    family_default = os.environ.get(_FAMILY_ENV)
    steps_default = os.environ.get(_STEPS_ENV)
    context_default = os.environ.get(_CONTEXT_ENV, "sliding")
    parser.add_argument(
        "--trainer-script",
        type=Path,
        default=Path(trainer_default) if trainer_default else None,
    )
    parser.add_argument(
        "--rollout-context",
        choices=tuple(sorted(ROLLOUT_CONTEXT_MODES)),
        default=context_default,
    )
    parser.add_argument(
        "--family",
        choices=("lewm", "viswm", "pldm", "prejepa"),
        default=family_default,
    )
    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=int(steps_default) if steps_default else None,
    )
    args = parser.parse_args(own)
    if args.trainer_script is None or args.family is None or args.rollout_steps is None:
        raise SystemExit(
            "StableWM rollout entry needs trainer, family, and rollout steps; "
            "the initial invocation must place them before '--'"
        )
    return args, hydra_args


def _load_trainer(path: Path):
    train_dir = str(path.parent)
    if train_dir not in sys.path:
        sys.path.insert(0, train_dir)
    name = f"contextworld_upstream_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import StableWM trainer: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main(argv: list[str] | None = None) -> int:
    args, hydra_args = _parse_args(argv)
    trainer_script = args.trainer_script.expanduser().resolve()
    if not trainer_script.is_file():
        raise SystemExit(f"StableWM trainer script does not exist: {trainer_script}")
    if args.rollout_steps <= 1:
        raise SystemExit("--rollout-steps must be greater than one")

    os.environ[_TRAINER_ENV] = str(trainer_script)
    os.environ[_FAMILY_ENV] = args.family
    os.environ[_STEPS_ENV] = str(args.rollout_steps)
    os.environ[_CONTEXT_ENV] = str(args.rollout_context)
    # The trainer is imported under a synthetic module name so ContextWorld
    # can patch its forward before Hydra composes the configuration.  Tell
    # Hydra to recover the config path from the decorated function's actual
    # source file; otherwise it interprets ``./config`` as a package path
    # relative to that synthetic module and fails before training starts.
    # Lightning's subprocess DDP children inherit this value and therefore
    # resolve the same upstream config directory when they re-enter here.
    os.environ[_HYDRA_MAIN_MODULE_ENV] = "__main__"
    trainer = _load_trainer(trainer_script)
    install_rollout_forward(
        trainer,
        family=args.family,
        rollout_steps=args.rollout_steps,
        context_mode=args.rollout_context,
    )
    # Hydra sees only its own arguments.  Keep argv[0] on this wrapper so
    # Lightning DDP children re-enter it and install the same overlay on every
    # rank instead of launching the unpatched upstream trainer directly.
    sys.argv = [str(Path(__file__).resolve()), *hydra_args]
    trainer.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
