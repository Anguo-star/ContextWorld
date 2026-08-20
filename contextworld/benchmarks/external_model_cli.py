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

Results are labelled ``external_unofficial``.  Minting an official scoreboard
row requires the preregistration and freeze path, and a convenience CLI must
not be able to shortcut it.

Usage
-----

::

    contextworld-external-eval --task speed \\
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

    def load_release(self) -> dict[str, Any]:
        module_name, _, attribute = self.loader.rpartition(".")
        module = __import__(module_name, fromlist=[attribute])
        return getattr(module, attribute)()

    def load_scorer(self) -> Callable[..., dict[str, Any]]:
        module_name, _, attribute = self.scorer.rpartition(".")
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
            f"{_SCORE}.cube_grasp_rule_icl_data."
            "load_cube_grasp_rule_icl_release"
        ),
        scorer=(
            f"{_SCORE}.cube_grasp_rule_icl_score."
            "evaluate_cube_grasp_rule_icl_model"
        ),
        builtins=_families("CubeGraspRule"),
        action_source="statistics",
        std_key="std_population",
    ),
}


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
    binding = TASKS[args.task]
    release = binding.load_release()
    adapter = build_adapter(
        args.adapter,
        builtins=binding.load_builtins(),
        request=build_request(binding, release, args),
    )
    payload = binding.load_scorer()(
        adapter=adapter, **_scorer_keywords(binding, args)
    )

    # Stamped, not merged into the scorer's own payload keys, so an external
    # run can never be mistaken for -- or replayed as -- a frozen submission.
    return {
        "schema_version": 1,
        "result_kind": RESULT_KIND,
        "task": binding.task,
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="contextworld-external-eval",
        description=(
            "Evaluate an external latent world model against a frozen "
            "ContextWorld task. Produces an unofficial result."
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
    parser.add_argument("--stablewm-repo")
    parser.add_argument("--stablewm-ref")
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = run(args)
    if args.output is not None:
        target = args.output.expanduser().resolve()
        write_json(target, payload)
        print(
            json.dumps(
                {"result_kind": RESULT_KIND, "output": str(target)},
                sort_keys=True,
            )
        )
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
