#!/usr/bin/env python3
"""Route ``(task, family, seed)`` to whichever launcher already trains it.

The cloud job template ends in ``bash ${run_shell_script} "$@"``, so the
platform holds exactly one script path. Pointing that at ``cloud_train.sh``
-- which delegates here -- means switching tasks is an environment variable
rather than a new job configuration.

This is a router, not a trainer. Every command it emits is one an operator
could type by hand; nothing here trains, and nothing here edits a launcher.
Two of the targets (``run_h3_hidden_passage_train.sh`` and
``train_tworoom_step1.py``) are byte-pinned by frozen release configs, which
is precisely why the routing lives in a new file above them.

What it exists to absorb is that the nine tasks do not share an interface.
Three axes diverge, and each divergence is a command a hand-written wrapper
gets silently wrong:

======================  ==================================================
task group              how family and seed are passed
======================  ==================================================
speed                   lewm: ``MODEL_VARIANT`` env + ``TRAINING_SEED`` env
                        pldm: a *different program* entirely
door                    positional variant, spelled ``fixed-mixed`` /
                        ``pldm-mixed`` -- not ``lewm`` / ``pldm``
action_delay            positional ``$1`` family, positional ``$2`` seed,
                        and two stages that must run in order
five hidden-property    ``--model`` / ``--seed`` / ``--output`` flags
action_strength         lewm: ``--variants <recipe string>``
                        pldm: the same different program as speed
prejepa (any task)      uniform, via the public StableWM family profile
======================  ==================================================

Usage::

    CW_TASK=speed CW_FAMILY=lewm bash scripts/cloud_train.sh
    CW_TASK=door CW_FAMILY=pldm CW_MODE=formal bash scripts/cloud_train.sh

Anything after ``--`` is forwarded to the underlying launcher untouched, so
the router never becomes the reason something is unreachable.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from contextworld.paths import artifact_root  # noqa: E402


SCRIPTS = REPO_ROOT / "scripts"

TASKS = (
    "speed",
    "door",
    "action_delay",
    "portal_exit",
    "action_strength",
    "contact_friction",
    "motion_damping",
    "robot_arm_mass",
    "cube_gripper_carry",
)

FAMILIES = ("lewm", "pldm", "prejepa")

# The five tasks that share one training engine and one flag interface.
# Adding a task here is the whole cost of routing it.
SHARED_ENGINE = {
    "contact_friction": "run_pusht_contact_friction_h3_train.py",
    "motion_damping": "run_pusht_motion_damping_h3_train.py",
    "robot_arm_mass": "run_reacher_arm_mass_h3_train.py",
    "portal_exit": "run_tworoom_portal_exit_h3_train.py",
    "cube_gripper_carry": "run_cube_grasp_rule_h3_train.py",
}

# ``run_h3_hidden_passage_train.sh`` names its variants after the data
# mixture, not the model family. The release config's ``shell_variant`` fields
# are the source of this mapping.
DOOR_VARIANT = {"lewm": "fixed-mixed", "pldm": "pldm-mixed"}

# The LeWM recipe of record for action_strength, from the release config's
# ``recipes.reference_method.runner_variant``.
ACTION_STRENGTH_VARIANT = "mixed_dynamics_response_sigreg_0p02"

# ``run_pldm_reference_completion.py`` spells these with a hyphen.
PLDM_COMPLETION_COMPONENT = {
    "speed": "speed",
    "action_strength": "action-strength",
}

# lewm.yaml and pldm.yaml both ship this; prejepa.yaml ships 32. Defaulting it
# here is what lets a prejepa run reach the baselines without the operator
# having to know that one number.
BASELINE_BATCH_SIZE = 128


@dataclass(frozen=True)
class Plan:
    """A resolved launch: what to run, and what it needs in the environment."""

    command: list[str]
    env: dict[str, str] = field(default_factory=dict)
    note: str = ""


def _bash(script: str, *arguments: str) -> list[str]:
    return ["bash", str(SCRIPTS / script), *arguments]


def _python(script: str, *arguments: str) -> list[str]:
    return [sys.executable, str(SCRIPTS / script), *arguments]


def default_run_name(task: str, family: str, seed: int) -> str:
    return f"{task}_{family}_s{seed}"


def default_output(task: str, family: str, seed: int) -> Path:
    root = artifact_root(REPO_ROOT) / "training/runs"
    return root / default_run_name(task, family, seed)


def _prejepa_plan(args: argparse.Namespace) -> Plan:
    """Every task reaches PreJEPA through the public family profile."""

    if not args.dataset:
        raise SystemExit(
            "prejepa needs a dataset: set CW_DATASET or pass --dataset."
        )
    command = _python(
        "run_stablewm_train.py",
        "--component", args.task,
        "--family", "prejepa",
        "--run-name", args.run_name or default_run_name(
            args.task, "prejepa", args.seed
        ),
        "--dataset", args.dataset,
        "--seed", str(args.seed),
        "--batch-size", str(args.batch_size or BASELINE_BATCH_SIZE),
    )
    if args.output:
        command += ["--output", str(args.output)]
    note = ""
    if args.batch_size is None:
        note = (
            f"batch_size defaulted to {BASELINE_BATCH_SIZE} to match the "
            "lewm/pldm baselines; prejepa.yaml ships 32."
        )
    return Plan(command=command, note=note)


def _speed_plan(args: argparse.Namespace) -> Plan:
    if args.family == "pldm":
        return Plan(
            command=_python(
                "run_pldm_reference_completion.py",
                "--component", PLDM_COMPLETION_COMPONENT["speed"],
                "--seed", str(args.seed),
            ),
            note="speed/pldm is a separate completion program, not a mode of "
                 "the lewm launcher.",
        )
    return Plan(
        command=_bash("run_h3_speed_isolated_train.sh", args.mode),
        env={
            "TRAINING_SEED": str(args.seed),
            "MODEL_VARIANT": args.variant or "multi",
        },
    )


def _door_plan(args: argparse.Namespace) -> Plan:
    return Plan(
        command=_bash(
            "run_h3_hidden_passage_train.sh",
            args.variant or DOOR_VARIANT[args.family],
            args.mode,
        ),
        env={"TRAINING_SEED": str(args.seed)},
    )


def _action_delay_plan(args: argparse.Namespace) -> Plan:
    """Two stages, in order. The curriculum stage verifies stage one's
    checkpoint hash, so running it first fails rather than misleads."""

    script = (
        "run_h7_action_delay_curriculum_train.sh"
        if args.stage == "curriculum"
        else "run_h7_action_delay_paired_train.sh"
    )
    return Plan(
        command=_bash(script, args.family, str(args.seed)),
        note=(
            "action_delay trains in two stages: paired, then curriculum. "
            f"This is the {args.stage} stage."
        ),
    )


def _action_strength_plan(args: argparse.Namespace) -> Plan:
    if args.family == "pldm":
        return Plan(
            command=_python(
                "run_pldm_reference_completion.py",
                "--component", PLDM_COMPLETION_COMPONENT["action_strength"],
                "--seed", str(args.seed),
            ),
        )
    command = _python(
        "run_pusht_hidden_actuation_mixed.py",
        "--variants", args.variant or ACTION_STRENGTH_VARIANT,
        "--seed", str(args.seed),
    )
    if args.output:
        command += ["--output", str(args.output)]
    return Plan(command=command)


def _shared_engine_plan(args: argparse.Namespace) -> Plan:
    output = args.output or default_output(args.task, args.family, args.seed)
    command = _python(
        SHARED_ENGINE[args.task],
        "--model", args.family,
        "--seed", str(args.seed),
        "--output", str(output),
    )
    if args.variant:
        command += ["--variant", args.variant]
    return Plan(command=command)


ORIGINAL_ENVIRONMENTS = ("tworoom", "pusht", "reacher", "cube")


def _original_plan(args: argparse.Namespace) -> Plan:
    """The baseline regime: unmodified task data, not an ICL capability.

    Reached with ``CW_TASK=original`` plus ``CW_ENV``, so a job template that
    already sets one task variable can reach both regimes.
    """

    if args.env not in ORIGINAL_ENVIRONMENTS:
        raise SystemExit(
            f"original training needs CW_ENV in {ORIGINAL_ENVIRONMENTS}; "
            f"got {args.env!r}"
        )
    command = _python(
        "run_stablewm_train.py",
        "--original-env", args.env,
        "--family", args.family,
    )
    if args.all_seeds:
        command.append("--all-seeds")
    else:
        command += ["--seed", str(args.seed)]
    if args.dataset:
        command += ["--dataset", str(args.dataset)]
    if args.output:
        command += ["--output", str(args.output)]
    effective_batch_size = args.batch_size
    if effective_batch_size is None and args.family == "prejepa":
        effective_batch_size = BASELINE_BATCH_SIZE
    if effective_batch_size is not None:
        command += ["--batch-size", str(effective_batch_size)]
    note = (
        "original task data, not a benchmark capability; "
        "baseline seeds are 3072/3073/3074"
    )
    if args.family == "prejepa" and args.batch_size is None:
        note += (
            f"; batch_size defaulted to {BASELINE_BATCH_SIZE} for "
            "LeWM/PLDM baseline comparability"
        )
    return Plan(
        command=command,
        note=note,
    )


def build_plan(args: argparse.Namespace) -> Plan:
    """Resolve one launch. Raises SystemExit on a combination that has no
    launcher, rather than emitting a command that would fail later."""

    if args.task == "original":
        if args.family not in FAMILIES:
            raise SystemExit(
                f"Unknown family {args.family!r}; expected one of {FAMILIES}"
            )
        return _original_plan(args)

    if args.task not in TASKS:
        raise SystemExit(f"Unknown task {args.task!r}; expected one of {TASKS}")
    if args.family not in FAMILIES:
        raise SystemExit(
            f"Unknown family {args.family!r}; expected one of {FAMILIES}"
        )

    if args.family == "prejepa":
        return _prejepa_plan(args)

    builders = {
        "speed": _speed_plan,
        "door": _door_plan,
        "action_delay": _action_delay_plan,
        "action_strength": _action_strength_plan,
    }
    builder = builders.get(args.task)
    if builder is not None:
        return builder(args)
    return _shared_engine_plan(args)


def _environment_default(name: str, fallback: str | None = None) -> str | None:
    value = os.environ.get(name)
    return value if value else fallback


def _environment_bool(name: str) -> bool:
    value = _environment_default(name)
    if value is None:
        return False
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise SystemExit(
        f"{name} must be one of 1/0, true/false, yes/no, or on/off"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="cloud_train",
        description=(
            "Route a benchmark task to its existing training launcher. "
            "Every option also reads a CW_* environment variable, because "
            "the cloud job template passes one script path and nothing else."
        ),
    )
    parser.add_argument(
        "--task",
        default=_environment_default("CW_TASK"),
        help=(
            "Benchmark task, or 'original' to train on unmodified task data "
            "(env: CW_TASK)"
        ),
    )
    parser.add_argument(
        "--env",
        default=_environment_default("CW_ENV"),
        help=(
            "Original task environment, used with --task original: "
            f"{', '.join(ORIGINAL_ENVIRONMENTS)} (env: CW_ENV)"
        ),
    )
    parser.add_argument(
        "--all-seeds",
        action="store_true",
        default=_environment_bool("CW_ALL_SEEDS"),
        help="Run all three baseline seeds in sequence (env: CW_ALL_SEEDS)",
    )
    parser.add_argument(
        "--family",
        default=_environment_default("CW_FAMILY", "lewm"),
        help="Model family: lewm, pldm or prejepa (env: CW_FAMILY)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=int(_environment_default("CW_SEED", "3072")),
        help="Training seed (env: CW_SEED)",
    )
    parser.add_argument(
        "--mode",
        default=_environment_default("CW_MODE", "preflight"),
        help=(
            "Mode for the shell-backed tasks. Not validated here -- the "
            "launcher owns its own vocabulary (env: CW_MODE)"
        ),
    )
    parser.add_argument(
        "--stage",
        choices=("paired", "curriculum"),
        default=_environment_default("CW_STAGE", "paired"),
        help="action_delay trains in two stages (env: CW_STAGE)",
    )
    parser.add_argument(
        "--variant",
        default=_environment_default("CW_VARIANT"),
        help=(
            "Override the launcher's variant. Defaults to the recipe of "
            "record for the task and family (env: CW_VARIANT)"
        ),
    )
    parser.add_argument(
        "--dataset",
        default=_environment_default("CW_DATASET"),
        help="Dataset for prejepa runs (env: CW_DATASET)",
    )
    parser.add_argument(
        "--run-name",
        default=_environment_default("CW_RUN_NAME"),
        help="Run name (env: CW_RUN_NAME)",
    )
    parser.add_argument(
        "--output",
        default=_environment_default("CW_OUTPUT"),
        help="Output directory (env: CW_OUTPUT)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=(
            int(os.environ["CW_BATCH_SIZE"])
            if os.environ.get("CW_BATCH_SIZE")
            else None
        ),
        help="Override batch size where the launcher accepts one "
             "(env: CW_BATCH_SIZE)",
    )
    parser.add_argument(
        "--print-command",
        action="store_true",
        default=_environment_bool("CW_PRINT_ONLY"),
        help="Resolve and print without running (env: CW_PRINT_ONLY)",
    )
    parser.add_argument(
        "extra",
        nargs=argparse.REMAINDER,
        help="Arguments after -- are forwarded to the launcher untouched",
    )
    args = parser.parse_args(argv)
    if not args.task:
        parser.error("no task: set CW_TASK or pass --task")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    plan = build_plan(args)

    extra = [item for item in args.extra if item != "--"]
    command = [*plan.command, *extra]
    # The profile launcher performs the useful final resolution: it composes
    # the family-specific YAML dialect and validates the concrete dataset.
    # Let print-only reach it while explicitly keeping it dry. Frozen benchmark
    # launchers own different preflight vocabularies and remain router-only.
    resolve_profile = args.print_command and (
        args.task == "original" or args.family == "prejepa"
    )
    if resolve_profile and "--print-command" not in command:
        command.append("--print-command")

    print(f"[cloud-train] task={args.task} family={args.family} "
          f"seed={args.seed}")
    if plan.note:
        print(f"[cloud-train] note: {plan.note}")
    for key, value in sorted(plan.env.items()):
        print(f"[cloud-train] env {key}={value}")
    print(f"[cloud-train] {' '.join(command)}")
    if args.print_command and not resolve_profile:
        return 0

    environment = {**os.environ, **plan.env}
    # Keep the routing decision ahead of child-process output in buffered
    # cloud logs.
    sys.stdout.flush()
    return subprocess.call(command, cwd=str(REPO_ROOT), env=environment)


if __name__ == "__main__":
    raise SystemExit(main())
