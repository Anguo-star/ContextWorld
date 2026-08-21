#!/usr/bin/env python3
"""Compatibility launcher for the four ORIGINAL task datasets.

New public and cloud runs use ``run_stablewm_train.py``. This module remains
for existing commands and for its focused mapping regression tests.

This is the baseline regime, not the benchmark one. ``cloud_train.py`` routes
the nine ContextWorld ICL capabilities; this trains on the unmodified
TwoRoom / PushT / Reacher / Cube data that those capabilities are built on
top of, which is what a baseline column in the comparison table reports.

Everything runs through Stable-WorldModel's own ``scripts/train/<family>.py``.
ContextWorld contributes the environment-to-config mapping and nothing else --
no forward pass, no loss, no optimiser.

Why a mapping is needed at all
------------------------------

The four environments do not name their data the same way, and the two config
shapes disagree about how data is selected:

====================  ==================  =====================  ============
environment           lewm/pldm group     prejepa dataset_name   encoding
====================  ==================  =====================  ============
tworoom               ``data=tworoom``    ``tworoom.h5``         proprio
pusht                 ``data=pusht``      ``pusht_expert…``      proprio
reacher               ``data=dmc``        ``reacher.h5``         observation
cube                  ``data=ogb``        ``ogbench/cube…``      observation
====================  ==================  =====================  ============

Two consequences worth stating, because both fail confusingly:

``lewm``/``pldm`` select data through hydra's ``data`` defaults group, so the
override is ``data=dmc``. ``prejepa`` has no such group -- it reads a flat
``dataset_name`` -- so the same request is spelled differently per family.

``prejepa.yaml`` defaults to ``wm.encoding={proprio: 10, action: 10}`` and
``prejepa.py`` raises if an encoding key is missing from the dataset columns.
Reacher and Cube carry ``observation`` rather than ``proprio``, so the
encoding is remapped per environment rather than left at the default.

Action width is *not* passed. ``lewm.py:285`` and ``prejepa.py:209`` both
derive it from the loaded dataset; overriding it here would be a second,
staler source of truth.

Usage::

    python scripts/run_original_task_train.py --env tworoom --family lewm \\
        --seed 3072

    # all three seeds, printed rather than run
    python scripts/run_original_task_train.py --env pusht --family prejepa \\
        --all-seeds --print-command
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from contextworld.paths import artifact_root  # noqa: E402


# The three seeds the frozen baseline families were completed to, so a new
# family can report mean +/- std against them. See
# ``contextworld_original_baseline_seed_completion_prereg_v1.yaml``.
BASELINE_SEEDS = (3072, 3073, 3074)

# lewm.yaml and pldm.yaml both ship this; prejepa.yaml ships 32.
BASELINE_BATCH_SIZE = 128


@dataclass(frozen=True)
class Environment:
    """One original task, in both config dialects."""

    name: str
    data_group: str
    dataset_name: str
    # prejepa's auxiliary encoders must name columns the dataset actually has.
    encoding_key: str
    # Verified by loading each dataset: action width and the auxiliary column's
    # width. Recorded for the launch banner only -- upstream derives both.
    action_dim: int
    encoding_dim: int

    def prejepa_encoding_overrides(self) -> list[str]:
        """prejepa.yaml hardcodes ``proprio``; two environments lack it."""

        if self.encoding_key == "proprio":
            return []
        return [
            "~wm.encoding.proprio",
            f"+wm.encoding.{self.encoding_key}=10",
        ]


# Dataset identifiers, columns and widths below were verified by loading each
# dataset through ``swm.data.load_dataset``:
#
#   tworoom  cols {action, pixels, proprio}       action 2  proprio 2
#   pusht    cols {action, pixels, proprio,state} action 2  proprio 4
#   reacher  cols {action, pixels, observation}   action 2  observation 6
#   cube     cols {action, pixels, observation}   action 5  observation 28
ENVIRONMENTS: dict[str, Environment] = {
    "tworoom": Environment(
        name="tworoom",
        data_group="tworoom",
        dataset_name="quentinll/tworoom.h5",
        encoding_key="proprio",
        action_dim=2,
        encoding_dim=2,
    ),
    "pusht": Environment(
        name="pusht",
        data_group="pusht",
        dataset_name="quentinll/pusht_expert_train.h5",
        encoding_key="proprio",
        action_dim=2,
        encoding_dim=4,
    ),
    "reacher": Environment(
        name="reacher",
        data_group="dmc",
        dataset_name="quentinll/reacher.h5",
        encoding_key="observation",
        action_dim=2,
        encoding_dim=6,
    ),
    "cube": Environment(
        name="cube",
        data_group="ogb",
        dataset_name="quentinll/ogbench/cube_single_expert.h5",
        encoding_key="observation",
        action_dim=5,
        encoding_dim=28,
    ),
}

FAMILIES = ("lewm", "pldm", "prejepa")


def resolve_dataset_root(explicit: str | None) -> Path | None:
    """Locate the directory ``dataset_name`` is resolved against.

    Stable-WorldModel resolves a relative dataset name under
    ``<STABLEWM_HOME>/datasets/``. On this machine the LeWM data sits one
    level higher, so the root is passed explicitly rather than assumed.

    Returns ``None`` when nothing is configured, which leaves upstream's own
    resolution untouched.  It deliberately does not infer a data location
    from the ContextWorld checkout: source data and generated artifacts are
    separate trees and a coincidental checkout layout is not authority.
    """

    candidate = explicit or os.environ.get("CONTEXTWORLD_DATASET_ROOT")
    if candidate:
        root = Path(candidate).expanduser().resolve()
        if not root.is_dir():
            raise SystemExit(f"Dataset root not found: {root}")
        return root
    return None


def dataset_argument(
    environment: Environment,
    root: Path | None,
    explicit: str | None = None,
) -> str:
    """Give upstream an absolute path when we can resolve one.

    A relative name goes through ``<STABLEWM_HOME>/datasets/<name>``, and an
    empty directory left behind by an interrupted download shadows the real
    file -- upstream then silently re-downloads several GB. An absolute path
    bypasses that lookup entirely (``data/utils.py:110``).
    """

    if explicit:
        resolved = Path(explicit).expanduser().resolve()
        if not resolved.is_file():
            raise SystemExit(f"Dataset file not found: {resolved}")
        return str(resolved)
    if root is None:
        return environment.dataset_name
    resolved = root / environment.dataset_name
    if not resolved.is_file():
        raise SystemExit(
            f"Dataset file not found under explicit root: {resolved}"
        )
    return str(resolved)


def resolve_stablewm_repo(explicit: str | None) -> Path:
    candidate = (
        explicit
        or os.environ.get("CONTEXTWORLD_STABLE_WORLDMODEL_REPO")
        or os.environ.get("STABLEWM_REPO")
    )
    if candidate:
        repo = Path(candidate).expanduser().resolve()
    else:
        repo = (
            artifact_root(REPO_ROOT)
            / "upstream/stable-worldmodel-875e607fc08aa72e"
        )
    if not (repo / "scripts/train").is_dir():
        raise SystemExit(
            f"Stable-WorldModel checkout not found at {repo}. Set "
            "--stablewm-repo or CONTEXTWORLD_STABLE_WORLDMODEL_REPO."
        )
    return repo


def resolve_train_script(repo: Path, family: str, explicit: str | None) -> Path:
    candidate = explicit or os.environ.get("CONTEXTWORLD_TRAIN_SCRIPT")
    script = (
        Path(candidate).expanduser().resolve()
        if candidate
        else repo / f"scripts/train/{family}.py"
    )
    if not script.is_file():
        raise SystemExit(f"Training script not found: {script}")
    return script


def run_name(environment: str, family: str, seed: int) -> str:
    return f"{environment}_{family}_original_s{seed}"


def build_overrides(
    environment: Environment,
    family: str,
    seed: int,
    args: argparse.Namespace,
    dataset_root: Path | None = None,
) -> list[str]:
    """Compose the hydra overrides for one original-task run.

    Only what the baseline regime fixes is set. Action width in particular is
    left to upstream, which derives it from the dataset.
    """

    name = args.run_name or run_name(environment.name, family, seed)
    # All three families name the run the same way. ``exp_name`` is the
    # wm_exp wrapper's spelling and is rejected by these configs.
    overrides = [
        f"seed={seed}",
        f"output_model_name={name}",
        # Upstream writes checkpoints and config.yaml below this directory.
        # Pinning it to the unique run name lets every environment and seed
        # safely share one STABLEWM_HOME/CW_CHECKPOINT_ROOT.
        f"subdir={name}",
    ]

    if family == "prejepa":
        overrides.append(
            f"dataset_name={dataset_argument(environment, dataset_root, args.dataset)}"
        )
        overrides.extend(environment.prejepa_encoding_overrides())
        overrides.append(
            f"batch_size={args.batch_size or BASELINE_BATCH_SIZE}"
        )
        if args.num_workers is not None:
            overrides.append(f"num_workers={args.num_workers}")
    else:
        # lewm/pldm select data through the defaults group, and read batch
        # size and workers from under ``loader``.
        overrides.insert(0, f"data={environment.data_group}")
        if args.dataset or args.dataset_override:
            overrides.append(
                f"data.dataset.name="
                f"{dataset_argument(environment, dataset_root, args.dataset)}"
            )
        if args.batch_size is not None:
            overrides.append(f"loader.batch_size={args.batch_size}")
        if args.num_workers is not None:
            overrides.append(f"loader.num_workers={args.num_workers}")

    for flag, key in (
        (args.devices, "trainer.devices"),
        (args.accumulate, "trainer.accumulate_grad_batches"),
        (args.max_epochs, "trainer.max_epochs"),
        (args.precision, "trainer.precision"),
    ):
        if flag is not None:
            overrides.append(f"{key}={flag}")

    if args.output is not None:
        target = Path(args.output).expanduser().resolve() / name
        overrides.append(f"hydra.run.dir={target}")

    overrides.extend(args.override)
    return overrides


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a world model on one of the four original task datasets "
            "via Stable-WorldModel."
        )
    )
    parser.add_argument(
        "--env",
        choices=sorted(ENVIRONMENTS),
        default=os.environ.get("CW_ENV"),
        help="Original task environment (env: CW_ENV)",
    )
    parser.add_argument(
        "--family",
        choices=FAMILIES,
        default=os.environ.get("CW_FAMILY", "lewm"),
        help="Model family (env: CW_FAMILY)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=int(os.environ.get("CW_SEED", BASELINE_SEEDS[0])),
        help=f"Training seed (env: CW_SEED). Baselines used {BASELINE_SEEDS}",
    )
    parser.add_argument(
        "--all-seeds",
        action="store_true",
        default=bool(os.environ.get("CW_ALL_SEEDS")),
        help=(
            f"Run all three baseline seeds {BASELINE_SEEDS} in sequence "
            "(env: CW_ALL_SEEDS)"
        ),
    )
    parser.add_argument("--run-name", default=os.environ.get("CW_RUN_NAME"))
    parser.add_argument("--output", default=os.environ.get("CW_OUTPUT"))
    parser.add_argument(
        "--dataset",
        default=os.environ.get("CW_DATASET"),
        help=(
            "Exact original dataset path (env: CW_DATASET). An explicit "
            "path overrides the environment's built-in dataset name."
        ),
    )
    parser.add_argument("--stablewm-repo", default=None)
    parser.add_argument("--train-script", default=None)
    parser.add_argument(
        "--dataset-dir",
        default=None,
        help=(
            "Directory holding the LeWM datasets (env: "
            "CONTEXTWORLD_DATASET_ROOT). Used to pass upstream an absolute "
            "path, bypassing the STABLEWM_HOME/datasets lookup."
        ),
    )
    parser.add_argument(
        "--dataset-override",
        action="store_true",
        default=bool(os.environ.get("CW_DATASET_OVERRIDE")),
        help=(
            "Also point lewm/pldm at the resolved path. Off by default so "
            "the data config group stays the source of truth."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=(
            int(os.environ["CW_BATCH_SIZE"])
            if os.environ.get("CW_BATCH_SIZE")
            else None
        ),
        help=(
            "lewm/pldm default to 128 upstream; prejepa defaults to 32 and is "
            f"raised to {BASELINE_BATCH_SIZE} here for comparability"
        ),
    )
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--devices", type=int, default=None)
    parser.add_argument("--accumulate", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--precision", default=None)
    parser.add_argument(
        "--override", action="append", default=[], metavar="KEY=VALUE"
    )
    parser.add_argument(
        "--print-command",
        action="store_true",
        default=bool(os.environ.get("CW_PRINT_ONLY")),
    )
    args = parser.parse_args(argv)
    if not args.env:
        parser.error("no environment: set CW_ENV or pass --env")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    environment = ENVIRONMENTS[args.env]
    repo = resolve_stablewm_repo(args.stablewm_repo)
    script = resolve_train_script(repo, args.family, args.train_script)
    # An exact file is already authoritative and must not be blocked by a
    # stale root variable inherited from a generic cloud job template.
    dataset_root = (
        None if args.dataset else resolve_dataset_root(args.dataset_dir)
    )

    seeds = BASELINE_SEEDS if args.all_seeds else (args.seed,)
    if not args.all_seeds and args.seed not in BASELINE_SEEDS:
        print(
            f"[original] note: seed {args.seed} is outside the baseline set "
            f"{BASELINE_SEEDS}; this run will not be family-comparable."
        )

    environment_variables = dict(os.environ)
    environment_variables["PYTHONPATH"] = os.pathsep.join(
        [str(repo), environment_variables.get("PYTHONPATH", "")]
    ).strip(os.pathsep)

    for seed in seeds:
        overrides = build_overrides(
            environment, args.family, seed, args, dataset_root
        )
        command = [sys.executable, str(script), *overrides]

        print(
            f"[original] env={args.env} family={args.family} seed={seed} "
            f"action_dim={environment.action_dim} "
            f"{environment.encoding_key}_dim={environment.encoding_dim}"
        )
        print(f"[original] {' '.join(command)}")
        if args.print_command:
            continue

        sys.stdout.flush()
        code = subprocess.call(
            command, cwd=str(repo), env=environment_variables
        )
        if code != 0:
            return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
