"""Evaluate a model ContextWorld has never seen on public Development data.

The command reads only the clean ``ContextWorld-v1`` bundle: its Training and
Development payloads are public, while Public Test remains intentionally
withheld.  This is therefore a reproducible Development evaluator for outside
model families, not a way to reproduce or mint a frozen Public Test result.

Historical task CLIs remain hash-pinned because they are part of published
provenance. This entry point is deliberately separate and unpinned. It can
reuse their model-independent metric kernels, but reconstructs inputs from the
public bundle and never falls back to ``CONTEXTWORLD_ARTIFACT_ROOT``.

Every result is labelled ``development_only_not_public_test`` and has no
formal pass or official-scoreboard status. An explicit missing-context PreJEPA
run is an additional diagnostic within that same Development boundary.

Usage
-----

::

    python -m contextworld.benchmarks.external_model_cli --task speed \\
        --adapter my_package.adapter:MyWorldModel \\
        --checkpoint /path/to/weights.pt \\
        --model-name my-world-model \\
        --benchmark-root /absolute/path/to/ContextWorld-v1 \\
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
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

from contextworld.benchmarks.adapter_registry import (
    AdapterRequest,
    add_adapter_argument,
    build_adapter,
)
from contextworld.benchmarks.bundle_development import (
    DEVELOPMENT_RESULT_KIND,
    development_action_normalization,
    development_action_normalizer_path,
    evaluate_bundle_development_model,
    resolve_development_payload,
)
from contextworld.paths import repository_root, resolve_contextworld_path
from contextworld.synthesis.manifest import write_json


ROOT = repository_root()

RESULT_KIND = DEVELOPMENT_RESULT_KIND
DIAGNOSTIC_RESULT_KIND = "external_diagnostic_non_frozen_v1"
_ARTIFACT_ROOT_ENV = "CONTEXTWORLD_ARTIFACT_ROOT"
_MODEL_CACHE_ROOT_ENV = "CONTEXTWORLD_MODEL_CACHE_ROOT"


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


def _benchmark_root(args: argparse.Namespace) -> Path:
    value = getattr(args, "benchmark_root", None) or os.environ.get(
        "CONTEXTWORLD_BENCHMARK_ROOT"
    )
    if not value:
        raise ValueError(
            "ContextWorld Development evaluation needs --benchmark-root or "
            "CONTEXTWORLD_BENCHMARK_ROOT. Point it at the absolute "
            "ContextWorld-v1 clean-export root."
        )
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError(
            "--benchmark-root must be absolute; received " f"{path}"
        )
    return Path(str(path))


def _public_model_cache_root(args: argparse.Namespace) -> Path:
    """Choose a writable model-cache base without consulting the old archive.

    The sealed LeWM/PLDM ``.pt`` loader predates ``ContextWorld-v1`` and asks
    :func:`contextworld.paths.artifact_path` for a model cache.  Public
    Development evaluation redirects that compatibility call beside the
    checkpoint (or below ``STABLEWM_HOME``) so it never falls back to the
    private ``context_world`` research tree.
    """

    configured = os.environ.get(_MODEL_CACHE_ROOT_ENV)
    if configured:
        root = Path(configured).expanduser()
        if not root.is_absolute():
            raise ValueError(f"{_MODEL_CACHE_ROOT_ENV} must be absolute: {root}")
        return Path(str(root))

    stablewm_home = os.environ.get("STABLEWM_HOME")
    if stablewm_home:
        home = Path(stablewm_home).expanduser()
        if not home.is_absolute():
            raise ValueError(f"STABLEWM_HOME must be absolute: {home}")
        return Path(str(home / ".contextworld-eval-cache"))

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if checkpoint.parent.parent.name == "checkpoints":
        checkpoint_root = checkpoint.parent.parent.parent
    else:
        checkpoint_root = checkpoint.parent
    return checkpoint_root / ".contextworld-eval-cache"


@contextmanager
def _public_model_cache_scope(args: argparse.Namespace) -> Iterator[Path]:
    """Redirect the sealed adapter's cache lookup for one construction call."""

    cache_root = _public_model_cache_root(args)
    previous = os.environ.get(_ARTIFACT_ROOT_ENV)
    os.environ[_ARTIFACT_ROOT_ENV] = str(cache_root)
    try:
        yield cache_root
    finally:
        if previous is None:
            os.environ.pop(_ARTIFACT_ROOT_ENV, None)
        else:
            os.environ[_ARTIFACT_ROOT_ENV] = previous


def build_development_request(
    binding: TaskBinding,
    *,
    bundle_root: Path,
    args: argparse.Namespace,
) -> AdapterRequest:
    """Build an adapter request from public bundle metadata only.

    This is deliberately separate from :func:`build_request`, which remains a
    small compatibility helper for historical frozen-release tooling.  The
    public command path below never calls it or reads a release config.
    """

    development = resolve_development_payload(bundle_root, task=binding.task)
    common: dict[str, Any] = {
        "task": binding.task,
        "checkpoint": args.checkpoint,
        "device": args.device,
        "repo_root": ROOT,
        "runtime": {
            "stablewm_repo": getattr(args, "stablewm_repo", None),
            "stablewm_ref": getattr(args, "stablewm_ref", None) or "",
        },
    }
    if binding.action_source == "normalizer":
        return AdapterRequest(
            action_normalizer=development_action_normalizer_path(development),
            **common,
        )
    action_mean, action_std = development_action_normalization(
        development,
        preferred_std_key=binding.std_key,
    )
    return AdapterRequest(
        action_mean=action_mean,
        action_std=action_std,
        **common,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    _validate_diagnostic_options(args)
    evaluation_split = getattr(args, "evaluation_split", "development")
    if evaluation_split != "development":
        raise ValueError(
            "Public Test is not available through contextworld-external-eval. "
            "This public entry point evaluates only ContextWorld-v1 "
            "Development data."
        )
    binding = TASKS[args.task]
    bundle_root = _benchmark_root(args)
    # The frozen LeWM/PLDM adapter still names its cache through the historical
    # artifact helper.  Scope that compatibility detail to a public,
    # checkpoint-adjacent cache; data resolution remains exclusively bound to
    # ContextWorld-v1.
    with _public_model_cache_scope(args):
        adapter = build_adapter(
            args.adapter,
            builtins=_builtins_for_run(binding, args),
            request=build_development_request(
                binding,
                bundle_root=bundle_root,
                args=args,
            ),
        )
    payload = evaluate_bundle_development_model(
        task=binding.task,
        adapter=adapter,
        model_name=args.model_name,
        training_recipe=args.training_recipe,
        training_seed=args.training_seed,
        benchmark_root=bundle_root,
        batch_size=int(args.batch_size),
        include_records=bool(getattr(args, "include_records", False)),
    )

    # Stamped, not merged into the evaluator payload keys, so a public
    # Development run cannot be replayed as a held-out Public-Test result.
    policy = _prejepa_missing_context_policy(args)
    history_adapter = _history_adapter(args)
    diagnostic = policy == "normalized_zero"
    payload_envelope: dict[str, Any] = {
        "schema_version": 1,
        "result_kind": RESULT_KIND,
        "task": binding.task,
        "evaluation_split": "development",
        "adapter_spec": args.adapter,
        "model_name": args.model_name,
        "official_scoreboard_row": False,
        "note": (
            "Produced by contextworld-external-eval from the public "
            "ContextWorld-v1 Development split. It is not a held-out Public "
            "Test score, has no formal pass decision, and does not enter the "
            "official scoreboard."
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
            "Produced by contextworld-external-eval from the public "
            "ContextWorld-v1 Development split using normalized-zero missing "
            "context. This diagnostic is not a held-out Public Test score, "
            "has no formal pass decision, and does not enter the official "
            "scoreboard."
        )
    return payload_envelope


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="contextworld-external-eval",
        description=(
            "Evaluate an external latent world model on the public "
            "ContextWorld-v1 Development split. Results are explicitly "
            "Development-only and never Public-Test scoreboard rows."
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
        "--benchmark-root",
        help=(
            "Absolute ContextWorld-v1 clean-export root. Defaults to "
            "CONTEXTWORLD_BENCHMARK_ROOT. The evaluator never falls back "
            "to CONTEXTWORLD_ARTIFACT_ROOT."
        ),
    )
    parser.add_argument(
        "--include-records",
        action="store_true",
        help=(
            "Retain per-pair Development diagnostics in the JSON result. "
            "Off by default to keep training post-evaluation artifacts compact."
        ),
    )
    parser.add_argument(
        "--prejepa-missing-context-policy",
        choices=("reject", "normalized_zero"),
        default="reject",
        help=(
            "Reject missing state-conditioned PreJEPA context (default), or "
            "explicitly run the Development-only diagnostic with "
            "model-normalized zero state. The latter is available only for "
            "--adapter prejepa."
        ),
    )
    parser.add_argument(
        "--history-adapter",
        choices=("native", "h3_tail_projection"),
        default="native",
        help=(
            "Use native task history (default). h3_tail_projection exposes "
            "a native H3 PreJEPA checkpoint through the Action Delay H7 "
            "Development boundary and is available only for --task "
            "action_delay with --adapter prejepa."
        ),
    )
    parser.add_argument(
        "--evaluation-split",
        choices=("development", "public"),
        default="development",
        help=(
            "Development is the only executable public option. Passing "
            "public is rejected because Public Test is not shipped in "
            "ContextWorld-v1."
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
