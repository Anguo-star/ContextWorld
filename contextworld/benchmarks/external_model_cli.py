"""Evaluate a model ContextWorld has never seen.

Why this is a separate entry point
----------------------------------

Every task CLI is hash-pinned.  Its ``sha256`` is recorded in the frozen
release configuration that governs the task, and each of those configurations
also declares ``runtime.supported_adapters`` -- the exact model families whose
published numbers that CLI produced.  Editing a task CLI to accept a third
model would invalidate the pin and contradict the frozen declaration, which is
the governance working correctly: the published numbers were produced by those
bytes and should keep meaning that.

So external models get their own door rather than a wider one cut into the
frozen path.  Nothing here is on a published result's provenance chain: the
task CLIs, scorers and release configurations are untouched and unpinned by
this module.  What it reuses is the frozen source of truth -- it reads each
task's release configuration for geometry and action normalization, and calls
that task's existing scorer -- so an external model is evaluated under exactly
the same protocol as the baselines, without being able to alter it.

Ordinary results are labelled ``external_unofficial``.  An explicitly selected
missing-context diagnostic is labelled ``external_diagnostic_non_frozen_v1``.
Neither form can mint an official scoreboard row: that requires the
preregistration and freeze path, and a convenience CLI must not be able to
shortcut it.

Usage
-----

::

    python -m contextworld.benchmarks.external_model_cli --task speed \\
        --adapter my_package.adapter:MyWorldModel \\
        --checkpoint /path/to/weights.pt \\
        --model-name my-world-model \\
        --output result.json

``--adapter`` accepts an import path, an installed ``contextworld.adapters``
entry point, or a built-in family name.  The adapter must implement
:class:`~contextworld.benchmarks.adapters.LatentWorldModelAdapter`; a class
that also implements ``from_contextworld_request`` is constructed through it
and never has to know that Stable-WorldModel exists.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from contextworld.benchmarks.adapter_registry import (
    AdapterRequest,
    add_adapter_argument,
    build_adapter,
)
from contextworld.paths import repository_root, resolve_contextworld_path
from contextworld.synthesis.manifest import write_json


ROOT = repository_root()

RESULT_KIND = "external_unofficial"
DIAGNOSTIC_RESULT_KIND = "external_diagnostic_non_frozen_v1"


@dataclass(frozen=True)
class TaskBinding:
    """How one benchmark task is reached without touching its frozen CLI."""

    task: str
    loader: str
    scorer: str
    builtins: dict[str, str]
    action_source: str
    std_key: str | None = None
    recipe_keyword: str = "training_recipe"
    development_scorer: str | None = None

    def load_release(self) -> dict[str, Any]:
        module_name, _, attribute = self.loader.rpartition(".")
        module = __import__(module_name, fromlist=[attribute])
        return getattr(module, attribute)()

    def load_scorer(self) -> Callable[..., dict[str, Any]]:
        module_name, _, attribute = self.scorer.rpartition(".")
        module = __import__(module_name, fromlist=[attribute])
        return getattr(module, attribute)

    def load_development_scorer(self) -> Callable[..., dict[str, Any]]:
        if self.development_scorer is None:
            raise ValueError(f"{self.task} has no separate Development scorer")
        module_name, _, attribute = self.development_scorer.rpartition(".")
        module = __import__(module_name, fromlist=[attribute])
        return getattr(module, attribute)

    def load_builtins(self) -> dict[str, type]:
        resolved: dict[str, type] = {}
        for family, dotted in self.builtins.items():
            module_name, _, attribute = dotted.rpartition(".")
            module = __import__(module_name, fromlist=[attribute])
            resolved[family] = getattr(module, attribute)
        return resolved


_ADAPTERS = "contextworld.benchmarks.adapters"
_SCORE = "contextworld.benchmarks"

# Built-in families, keyed by the name accepted by ``--adapter``.  Each value
# is the module holding the family's adapters and the infix in their class
# names, which follow the regular pattern
# ``StableWorldModel{infix}{task}Adapter``.  Adding a family that keeps that
# pattern is one entry here rather than one binding per task.
#
# ``prejepa`` lives in its own module because ``adapters.py`` is byte-pinned
# by frozen release configs and must not be edited to add a model family.
_BUILTIN_FAMILIES = {
    "lewm": (_ADAPTERS, "LeWM"),
    "pldm": (_ADAPTERS, "PLDM"),
    "prejepa": (f"{_SCORE}.prejepa_adapters", "PreJEPA"),
}


def _families(prefix: str) -> dict[str, str]:
    return {
        family: f"{module}.StableWorldModel{infix}{prefix}Adapter"
        for family, (module, infix) in _BUILTIN_FAMILIES.items()
    }


TASKS: dict[str, TaskBinding] = {
    "speed": TaskBinding(
        task="speed",
        loader=f"{_SCORE}.speed_icl_data.load_speed_icl_release",
        scorer=f"{_SCORE}.speed_icl_score.evaluate_speed_icl_model",
        builtins=_families(""),
        action_source="normalizer",
        recipe_keyword="training_role",
    ),
    "door": TaskBinding(
        task="door",
        loader=f"{_SCORE}.door_icl_data.load_door_icl_release",
        scorer=f"{_SCORE}.door_icl_score.evaluate_door_icl_model",
        builtins=_families(""),
        action_source="normalizer",
    ),
    "action_delay": TaskBinding(
        task="action_delay",
        loader=f"{_SCORE}.action_delay_icl_data.load_action_delay_icl_release",
        scorer=(
            f"{_SCORE}.action_delay_icl_score.evaluate_action_delay_icl_model"
        ),
        builtins=_families("History7"),
        action_source="normalizer",
    ),
    "action_strength": TaskBinding(
        task="action_strength",
        loader=(
            f"{_SCORE}.action_strength_icl_data."
            "load_action_strength_icl_release"
        ),
        scorer=(
            f"{_SCORE}.action_strength_icl_score."
            "evaluate_action_strength_icl_model"
        ),
        builtins=_families("ActionStrength"),
        action_source="statistics",
        std_key="std_population",
    ),
    "contact_friction": TaskBinding(
        task="contact_friction",
        loader=(
            f"{_SCORE}.contact_friction_icl_data."
            "load_contact_friction_icl_release"
        ),
        scorer=(
            f"{_SCORE}.contact_friction_icl_score."
            "evaluate_contact_friction_icl_model"
        ),
        builtins=_families("ContactFriction"),
        action_source="statistics",
        std_key="std_population",
        development_scorer=(
            f"{_SCORE}.contact_friction_icl_score."
            "evaluate_contact_friction_icl_development_model"
        ),
    ),
    "motion_damping": TaskBinding(
        task="motion_damping",
        loader=(
            f"{_SCORE}.motion_damping_icl_data."
            "load_motion_damping_icl_release"
        ),
        scorer=(
            f"{_SCORE}.motion_damping_icl_score."
            "evaluate_motion_damping_icl_model"
        ),
        builtins=_families("MotionDamping"),
        action_source="statistics",
        std_key="std_population",
        development_scorer=(
            f"{_SCORE}.motion_damping_icl_score."
            "evaluate_motion_damping_icl_development_model"
        ),
    ),
    "portal_exit": TaskBinding(
        task="portal_exit",
        loader=f"{_SCORE}.portal_exit_icl_data.load_portal_exit_icl_release",
        scorer=(
            f"{_SCORE}.portal_exit_icl_score.evaluate_portal_exit_icl_model"
        ),
        builtins=_families("PortalExit"),
        action_source="statistics",
        std_key="std_unbiased",
    ),
    "robot_arm_mass": TaskBinding(
        task="robot_arm_mass",
        loader=(
            f"{_SCORE}.reacher_arm_mass_icl_data."
            "load_reacher_arm_mass_icl_release"
        ),
        scorer=(
            f"{_SCORE}.reacher_arm_mass_icl_score."
            "evaluate_reacher_arm_mass_icl_model"
        ),
        builtins=_families("ReacherArmMass"),
        action_source="statistics",
        std_key="std_population",
    ),
    "cube_gripper_carry": TaskBinding(
        task="cube_gripper_carry",
        loader=(
            f"{_SCORE}.cube_grasp_rule_v4r1_icl_data."
            "load_cube_grasp_rule_v4r1_icl_release"
        ),
        scorer=(
            f"{_SCORE}.cube_grasp_rule_v4r1_icl_score."
            "evaluate_cube_grasp_rule_v4r1_icl_model"
        ),
        builtins=_families("CubeGraspRule"),
        action_source="statistics",
        std_key="std_population",
    ),
}


# The regular family table must continue to resolve only the strict adapters.
# These classes are selected only after an operator explicitly asks for the
# diagnostic missing-context policy.
_DIAGNOSTIC_PREJEPA_CLASSES = {
    "speed": "StableWorldModelPreJEPADiagnosticAdapter",
    "door": "StableWorldModelPreJEPADiagnosticAdapter",
    "action_delay": "StableWorldModelPreJEPADiagnosticHistory7Adapter",
    "action_strength": "StableWorldModelPreJEPADiagnosticActionStrengthAdapter",
    "contact_friction": (
        "StableWorldModelPreJEPADiagnosticContactFrictionAdapter"
    ),
    "motion_damping": "StableWorldModelPreJEPADiagnosticMotionDampingAdapter",
    "portal_exit": "StableWorldModelPreJEPADiagnosticPortalExitAdapter",
    "robot_arm_mass": "StableWorldModelPreJEPADiagnosticReacherArmMassAdapter",
    "cube_gripper_carry": (
        "StableWorldModelPreJEPADiagnosticCubeGraspRuleAdapter"
    ),
}


def _prejepa_missing_context_policy(args: argparse.Namespace) -> str:
    """Read a new option without breaking programmatic callers of ``run``."""

    return str(getattr(args, "prejepa_missing_context_policy", "reject"))


def _history_adapter(args: argparse.Namespace) -> str:
    """Read the Action Delay projection option with its native default."""

    return str(getattr(args, "history_adapter", "native"))


def _validate_diagnostic_options(args: argparse.Namespace) -> None:
    policy = _prejepa_missing_context_policy(args)
    history_adapter = _history_adapter(args)
    if policy not in {"reject", "normalized_zero"}:
        raise ValueError(
            "Unknown --prejepa-missing-context-policy "
            f"{policy!r}; expected 'reject' or 'normalized_zero'"
        )
    if history_adapter not in {"native", "h3_tail_projection"}:
        raise ValueError(
            "Unknown --history-adapter "
            f"{history_adapter!r}; expected 'native' or "
            "'h3_tail_projection'"
        )
    if policy == "normalized_zero" and args.adapter != "prejepa":
        raise ValueError(
            "--prejepa-missing-context-policy normalized_zero is only "
            "available for --adapter prejepa"
        )
    if history_adapter == "h3_tail_projection":
        if args.task != "action_delay":
            raise ValueError(
                "--history-adapter h3_tail_projection is only available "
                "for --task action_delay"
            )
        if args.adapter != "prejepa":
            raise ValueError(
                "--history-adapter h3_tail_projection is only available "
                "for --adapter prejepa in contextworld-external-eval"
            )


def _prejepa_adapter_class_name(
    binding: TaskBinding, args: argparse.Namespace
) -> str | None:
    """Return an explicit PreJEPA override, or ``None`` for strict default."""

    history_adapter = _history_adapter(args)
    if history_adapter == "h3_tail_projection":
        if _prejepa_missing_context_policy(args) == "normalized_zero":
            return "StableWorldModelPreJEPADiagnosticActionDelayH3TailAdapter"
        return "StableWorldModelPreJEPAActionDelayH3TailAdapter"
    if _prejepa_missing_context_policy(args) == "normalized_zero":
        return _DIAGNOSTIC_PREJEPA_CLASSES[binding.task]
    return None


def _builtins_for_run(
    binding: TaskBinding, args: argparse.Namespace
) -> dict[str, type]:
    """Resolve ordinary built-ins, replacing PreJEPA only on explicit opt-in."""

    builtins = binding.load_builtins()
    if args.adapter != "prejepa":
        return builtins
    class_name = _prejepa_adapter_class_name(binding, args)
    if class_name is None:
        return builtins
    module_name = f"{_SCORE}.prejepa_adapters"
    module = __import__(module_name, fromlist=[class_name])
    return {**builtins, "prejepa": getattr(module, class_name)}


def build_request(
    binding: TaskBinding, release: dict[str, Any], args: argparse.Namespace
) -> AdapterRequest:
    """Assemble the construction request from the task's frozen release.

    Action geometry is read from the release configuration rather than from
    the command line, so an external model is normalized exactly as the
    baselines were and cannot quietly evaluate under a different contract.
    """

    runtime = release.get("runtime", {}).get("stable_worldmodel", {})
    common: dict[str, Any] = {
        "task": binding.task,
        "checkpoint": args.checkpoint,
        "device": args.device,
        "repo_root": ROOT,
        "runtime": {
            "stablewm_repo": args.stablewm_repo or runtime.get("repo"),
            "stablewm_ref": (
                args.stablewm_ref or runtime.get("expected_ref", "")
            ),
        },
    }

    if binding.action_source == "normalizer":
        return AdapterRequest(
            action_normalizer=resolve_contextworld_path(
                release["evaluation"]["normalizer"], repo_root=ROOT
            ),
            **common,
        )

    normalization = release["evaluation"]["action_normalization"]
    return AdapterRequest(
        action_mean=normalization["mean"],
        action_std=normalization[binding.std_key],
        **common,
    )


def _scorer_keywords(
    binding: TaskBinding, args: argparse.Namespace
) -> dict[str, Any]:
    keywords: dict[str, Any] = {
        "model_name": args.model_name,
        binding.recipe_keyword: args.training_recipe,
        "training_seed": args.training_seed,
        "repo_root": ROOT,
    }
    if binding.task == "speed":
        # The speed scorer predates the shared shape and takes three batch
        # sizes rather than one.
        keywords["encode_batch_size"] = args.batch_size
        keywords["rollout_batch_size"] = args.batch_size
        keywords["bundle_batch_size"] = args.batch_size
    else:
        keywords["batch_size"] = args.batch_size
    return keywords


def run(args: argparse.Namespace) -> dict[str, Any]:
    _validate_diagnostic_options(args)
    binding = TASKS[args.task]
    release = binding.load_release()
    adapter = build_adapter(
        args.adapter,
        builtins=_builtins_for_run(binding, args),
        request=build_request(binding, release, args),
    )
    evaluation_split = getattr(args, "evaluation_split", "public")
    scorer = (
        binding.load_development_scorer()
        if evaluation_split == "development"
        else binding.load_scorer()
    )
    payload = scorer(
        adapter=adapter, **_scorer_keywords(binding, args)
    )

    # Stamped, not merged into the scorer's own payload keys, so an external
    # run can never be mistaken for -- or replayed as -- a frozen submission.
    policy = _prejepa_missing_context_policy(args)
    history_adapter = _history_adapter(args)
    diagnostic = policy == "normalized_zero"
    payload_envelope: dict[str, Any] = {
        "schema_version": 1,
        "result_kind": (
            DIAGNOSTIC_RESULT_KIND if diagnostic else RESULT_KIND
        ),
        "task": binding.task,
        "evaluation_split": evaluation_split,
        "adapter_spec": args.adapter,
        "model_name": args.model_name,
        "release_id": release.get("release_id"),
        "official_scoreboard_row": False,
        "note": (
            "Produced by contextworld-external-eval. This is an unofficial "
            "result: it uses the frozen task protocol but is not a "
            "preregistered submission and does not enter the public "
            "scoreboard."
        ),
        "result": payload,
    }
    if diagnostic:
        payload_envelope["diagnostic"] = {
            "classification": "diagnostic",
            "prejepa_missing_context_policy": policy,
            "frozen_v1_compatible": False,
            "history_adapter": history_adapter,
        }
        payload_envelope["note"] = (
            "Produced by contextworld-external-eval using normalized-zero "
            "missing context. This is a diagnostic, non-frozen-v1 result; "
            "it is not a preregistered submission and does not enter the "
            "public scoreboard."
        )
    return payload_envelope


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="contextworld-external-eval",
        description=(
            "Evaluate an external latent world model against a frozen "
            "ContextWorld task. Produces an unofficial result, or an "
            "explicitly labelled diagnostic result."
        ),
    )
    parser.add_argument("--task", choices=sorted(TASKS), required=True)
    add_adapter_argument(
        parser,
        # Derived from the family table so the help text cannot drift out of
        # step with what ``--adapter`` actually accepts.  Only the keys are
        # read here; the classes are resolved per task.
        builtins={family: object for family in _BUILTIN_FAMILIES},
        default=None,
        required=True,
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--training-recipe", default="external_method")
    parser.add_argument("--training-seed", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--prejepa-missing-context-policy",
        choices=("reject", "normalized_zero"),
        default="reject",
        help=(
            "Keep the frozen-v1 rejection for state-conditioned PreJEPA "
            "checkpoints (default), or explicitly run the non-frozen-v1 "
            "diagnostic with model-normalized zero state. The latter is "
            "available only for --adapter prejepa."
        ),
    )
    parser.add_argument(
        "--history-adapter",
        choices=("native", "h3_tail_projection"),
        default="native",
        help=(
            "Use native task history (default). h3_tail_projection exposes "
            "a native H3 PreJEPA checkpoint through the Action Delay H7 "
            "boundary and is available only for --task action_delay with "
            "--adapter prejepa."
        ),
    )
    parser.add_argument(
        "--evaluation-split",
        choices=("public", "development"),
        default="public",
        help=(
            "Use Development only when the component's Public Test remains "
            "closed. Components without a separate Development scorer reject it."
        ),
    )
    parser.add_argument("--stablewm-repo")
    parser.add_argument("--stablewm-ref")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        _validate_diagnostic_options(args)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = run(args)
    if args.output is not None:
        target = args.output.expanduser().resolve()
        write_json(target, payload)
        print(
            json.dumps(
                {"result_kind": payload["result_kind"], "output": str(target)},
                sort_keys=True,
            )
        )
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
