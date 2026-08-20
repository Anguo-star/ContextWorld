#!/usr/bin/env python3
"""Launch Stable-WorldModel's ``prejepa`` (DINOv2) training for a benchmark task.

ContextWorld is the benchmark entry point, not a trainer. This script composes
Stable-WorldModel's own ``scripts/train/prejepa.py`` configuration, points it
at a task's data, and hands off. The forward pass, loss and optimisation are
entirely upstream's -- nothing here reimplements them.

It lives beside ``train_tworoom_step1.py`` rather than inside it because that
file's bytes are pinned by the speed and door release configs. The same
composition pattern is already used by the portal-exit, cube and reacher
launchers, which extend a pinned trainer by importing it and rebinding
attributes rather than editing it.

``prejepa`` differs structurally from the ``lewm``/``pldm`` configs, which is
the whole of what this script adapts:

======================  ==========================  =========================
                        lewm / pldm                 prejepa
======================  ==========================  =========================
dataset selection       ``data`` defaults group     flat ``dataset_name``
batch size              ``cfg.loader.batch_size``   flat ``cfg.batch_size``
workers                 ``cfg.loader.num_workers``  flat ``cfg.num_workers``
loss config             ``cfg.loss``                (none -- built in code)
======================  ==========================  =========================

Everything else -- history, frameskip, image size, ImageNet preprocessing --
already matches, because ``prejepa.yaml`` and the benchmark were both derived
from the same DINO-WM setup.

Usage::

    python scripts/run_prejepa_train.py --task speed --run-name my_run \\
        --dataset /path/to/data --output /path/to/runs

The training script itself is resolved from the family name and can be
overridden with ``--train-script`` or ``CONTEXTWORLD_TRAIN_SCRIPT``.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from contextworld.paths import artifact_root, repository_root  # noqa: E402


FAMILY = "prejepa"

# Per-task geometry. History and action width belong to the task, not the
# model, so they are stated here and pushed into the upstream config rather
# than left to whatever the family's default config happens to carry.
#
# ``action_input_dim`` is the raw action width times the frameskip, matching
# how the benchmark packs an action block.
TASK_GEOMETRY: dict[str, dict[str, int]] = {
    "speed": {"history": 3, "action_dim": 2},
    "door": {"history": 3, "action_dim": 2},
    "action_delay": {"history": 7, "action_dim": 2},
    "portal_exit": {"history": 3, "action_dim": 2},
    "action_strength": {"history": 3, "action_dim": 2},
    "contact_friction": {"history": 3, "action_dim": 2},
    "motion_damping": {"history": 3, "action_dim": 2},
    "robot_arm_mass": {"history": 3, "action_dim": 2},
    "cube_gripper_carry": {"history": 3, "action_dim": 5},
}

FRAMESKIP = 5

# ``lewm.yaml`` and ``pldm.yaml`` both ship ``batch_size: 128`` with no
# gradient accumulation, so the baselines are aligned by upstream default.
# ``prejepa.yaml`` ships 32, which is the one value worth overriding: pass
# ``--batch-size 128`` to train it like the baselines were trained.
BASELINE_BATCH_SIZE = 128


def resolve_stablewm_repo(explicit: str | None) -> Path:
    """Locate the Stable-WorldModel checkout that owns the training code."""

    candidate = (
        explicit
        or os.environ.get("CONTEXTWORLD_STABLE_WORLDMODEL_REPO")
        or os.environ.get("STABLEWM_REPO")
    )
    if candidate:
        repo = Path(candidate).expanduser().resolve()
    else:
        repo = (
            artifact_root(repository_root())
            / "upstream/stable-worldmodel-875e607fc08aa72e"
        )
    if not (repo / "scripts/train").is_dir():
        raise SystemExit(
            f"Stable-WorldModel checkout not found at {repo}. Set "
            "--stablewm-repo or CONTEXTWORLD_STABLE_WORLDMODEL_REPO."
        )
    return repo


def resolve_train_script(repo: Path, explicit: str | None) -> Path:
    """Resolve the upstream training script.

    Defaults to the family's own script inside the pinned checkout, so the
    common case needs no argument at all. An explicit path (flag or
    environment variable) wins, which is what makes a fork or a new family
    reachable without editing this file.
    """

    candidate = explicit or os.environ.get("CONTEXTWORLD_TRAIN_SCRIPT")
    script = (
        Path(candidate).expanduser().resolve()
        if candidate
        else repo / f"scripts/train/{FAMILY}.py"
    )
    if not script.is_file():
        raise SystemExit(f"Training script not found: {script}")
    return script


def build_overrides(args: argparse.Namespace) -> list[str]:
    """Translate benchmark-level choices into upstream hydra overrides.

    Only values the benchmark actually fixes are set. Anything left alone
    keeps the upstream default, so this does not silently become a second
    copy of the training recipe.
    """

    geometry = TASK_GEOMETRY[args.task]
    overrides = [
        f"dataset_name={args.dataset}",
        f"seed={args.seed}",
        f"frameskip={FRAMESKIP}",
        f"wm.history_size={geometry['history']}",
        f"model.action_encoder.input_dim="
        f"{geometry['action_dim'] * FRAMESKIP}",
        f"output_model_name={args.run_name}",
    ]
    # Hardware-shaped knobs. These change throughput, not the objective, so
    # they are plain pass-throughs with upstream defaults when unset.
    for flag, key in (
        (args.batch_size, "batch_size"),
        (args.num_workers, "num_workers"),
        (args.devices, "trainer.devices"),
        (args.accumulate, "trainer.accumulate_grad_batches"),
        (args.max_epochs, "trainer.max_epochs"),
        (args.precision, "trainer.precision"),
    ):
        if flag is not None:
            overrides.append(f"{key}={flag}")
    if args.output is not None:
        overrides.append(f"hydra.run.dir={Path(args.output).resolve()}")
    overrides.extend(args.override)
    return overrides


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Launch Stable-WorldModel prejepa (DINOv2) training for a "
            "ContextWorld benchmark task."
        )
    )
    parser.add_argument("--task", choices=sorted(TASK_GEOMETRY), required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset name or local path passed to swm.data.load_dataset",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=3072)
    parser.add_argument("--stablewm-repo", default=None)
    parser.add_argument(
        "--train-script",
        default=None,
        help=(
            "Override the upstream training script. Defaults to "
            f"<stablewm-repo>/scripts/train/{FAMILY}.py"
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=(
            f"Defaults to the family config. prejepa.yaml ships 32; the "
            f"lewm/pldm baselines train at {BASELINE_BATCH_SIZE}."
        ),
    )
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--devices", type=int, default=None)
    parser.add_argument(
        "--accumulate",
        type=int,
        default=None,
        help=(
            "Gradient accumulation. Upstream uses none; set this only to "
            "reach the baseline effective batch on fewer GPUs."
        ),
    )
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--precision", default=None)
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra hydra override, repeatable",
    )
    parser.add_argument(
        "--print-command",
        action="store_true",
        help="Print the upstream command without running it",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo = resolve_stablewm_repo(args.stablewm_repo)
    script = resolve_train_script(repo, args.train_script)
    command = [sys.executable, str(script), *build_overrides(args)]

    print(f"[prejepa] task={args.task} run={args.run_name}")
    print(f"[prejepa] stablewm={repo}")
    print(f"[prejepa] script={script}")
    print(f"[prejepa] {' '.join(command)}")
    if args.print_command:
        return 0

    environment = dict(os.environ)
    # Upstream resolves its own imports relative to the checkout.
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(repo), environment.get("PYTHONPATH", "")]
    ).strip(os.pathsep)
    return subprocess.call(command, cwd=str(repo), env=environment)


if __name__ == "__main__":
    raise SystemExit(main())
