#!/usr/bin/env python3
"""Public Stable-WorldModel training entry for ContextWorld.

The command line is stable; the Hydra keys are not.  LeWM, VIS-WM and PLDM use a
``data`` defaults group and put loader settings below ``loader``.  PreJEPA
(DINO-WM) uses a flat dataset name, batch size and worker count.  This
launcher reads the checked-in family profile and translates only parameters
that the selected trainer actually accepts.

For a benchmark component, the default ``joint_scratch_v1`` track trains any
of the four built-in families from its native initialization on the same
registered ``ContextWorld-v1`` mixture.  The old byte-pinned LeWM/PLDM
launchers remain available only through an explicit ``historical_release``
track, so a current comparison cannot silently become a fine-tuning run.

The selected Stable-WorldModel checkout still owns the model, objective,
forward pass, optimizer and training loop.  ContextWorld owns path validation,
run isolation, logger credentials, family-specific argument mapping and the
optional hand-off to original-environment MPC evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from contextworld.training.seeds import (  # noqa: E402
    DEFAULT_TRAINING_SEEDS,
    parse_training_seeds,
    reject_legacy_seed_environment,
)
from contextworld.training.stablewm_bundle import (  # noqa: E402
    URI_PREFIX as CONTEXTWORLD_DATASET_URI_PREFIX,
    build_contextworld_dataset_uri,
    describe_contextworld_dataset,
)

DEFAULT_PROFILE_CONFIG = (REPO_ROOT /
                          "configs/training/stablewm_family_profiles_v1.yaml")
TRAINING_IDENTITY_FILENAME = "contextworld_training_identity_v1.json"
TRAINING_IDENTITY_SCHEMA = "contextworld.stablewm-training-identity.v1"
FAMILY_ENTRY_SCRIPT = REPO_ROOT / "scripts/run_stablewm_family_entry.py"
STABLEWM_BOOTSTRAP_DIR = REPO_ROOT / "scripts/stablewm_bootstrap"
STABLEWM_SITECUSTOMIZE = STABLEWM_BOOTSTRAP_DIR / "sitecustomize.py"
STABLEWM_BUNDLE_ADAPTER = (
    REPO_ROOT / "contextworld/training/stablewm_bundle.py"
)
SPT_RUN_MARKER_FILENAME = "contextworld_run_identity_v1.json"
SPT_RUN_MARKER_SCHEMA = "contextworld.stablepretraining-run-identity.v1"
RESET_ARCHIVE_DIRNAME = ".contextworld_reset_archive"
RESET_RECEIPT_SCHEMA = "contextworld.stablewm-run-reset.v1"
MINIMUM_STABLE_PRETRAINING_VERSION = (0, 1, 8)


def _env(name: str, fallback: str | None = None) -> str | None:
    value = os.environ.get(name)
    return value if value not in (None, "") else fallback


def _env_int(name: str) -> int | None:
    value = _env(name)
    return int(value) if value is not None else None


def _env_float(name: str) -> float | None:
    value = _env(name)
    return float(value) if value is not None else None


def _env_bool(name: str) -> bool | None:
    value = _env(name)
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise SystemExit(f"{name} must be one of 1/0, true/false, yes/no, or on/off")


def _hydra_bool(value: bool) -> str:
    return "true" if value else "false"


def load_profile_contract(path: Path = DEFAULT_PROFILE_CONFIG) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SystemExit(f"StableWM profile config not found: {path}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"StableWM profile config is not a mapping: {path}")
    expected = "contextworld.stablewm-family-profiles.v1"
    if payload.get("schema_version") != expected:
        raise SystemExit(f"Unsupported StableWM profile schema in {path}: "
                         f"{payload.get('schema_version')!r}")
    return payload


@dataclass(frozen=True)
class Target:
    """One dataset/geometry selection independent of model family."""

    label: str
    dataset: Path | str
    data_group: str | None
    history_size: int
    action_dim: int
    environment: str
    encoding_key: str | None = None
    encoding_dim: int | None = None
    original_env: str | None = None


@dataclass
class SeedOutcome:
    """Execution status for one requested training seed."""

    run_name: str
    training_status: str
    training_returncode: int | None = None
    evaluation_status: str = "not_requested"
    evaluation_returncode: int | None = None


@dataclass(frozen=True)
class RunResetMove:
    """One exact same-filesystem directory rename in a reset plan."""

    kind: str
    source: Path
    archive_namespace: Path
    archive_relative: Path


@dataclass(frozen=True)
class RunResetPlan:
    """All state bound to one public run name."""

    run_name: str
    moves: tuple[RunResetMove, ...]


def _absolute_path(value: str | Path, *, label: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise SystemExit(f"{label} must be an absolute path: {path}")
    # Keep an explicitly supplied symlink spelling visible in the rendered
    # command. The path is already absolute; resolving it would make a cloud
    # mount or operator-selected alias appear to have been silently replaced.
    return Path(os.path.abspath(path))


def _validate_dataset_payload(path: Path) -> None:
    if path.is_file():
        if path.suffix.lower() not in {".h5", ".hdf5"}:
            raise SystemExit("Dataset files must be H5/HDF5; got "
                             f"{path}")
        return
    if path.is_dir() and path.name.endswith(".lance"):
        return
    if path.is_dir():
        tables = sorted(path.rglob("*.lance"))
        preview = ", ".join(str(item) for item in tables[:3])
        suffix = " ..." if len(tables) > 3 else ""
        raise SystemExit(f"Dataset must be an exact H5 file or .lance table, not the "
                         f"collection directory {path}. Found {len(tables)} Lance "
                         f"table(s){(': ' + preview + suffix) if tables else '.'}")
    raise SystemExit(f"Dataset does not exist: {path}")


def _is_contextworld_dataset_uri(value: str | Path) -> bool:
    return str(value).startswith(CONTEXTWORLD_DATASET_URI_PREFIX)


def _family_model_columns(
    *,
    family: str,
    target: "Target",
    stablewm_repo: Path,
) -> tuple[str, ...]:
    """Return the literal model-input columns selected by the family profile."""

    required = ["pixels", "action"]
    # Every current benchmark-component training view follows the same public
    # RGB/action contract, independent of StableWM family. Keep original-
    # environment checkpoints on their native family data profile, but never
    # request privileged simulator state from a component bundle.
    if target.original_env is None:
        return tuple(required)
    if family == "prejepa":
        encoding_key = target.encoding_key
        if encoding_key and encoding_key not in required:
            required.append(encoding_key)
        return tuple(required)

    if not target.data_group:
        raise SystemExit(
            f"{family} needs --data-group before its Lance schema can be "
            "validated."
        )
    data_config = (
        stablewm_repo / "scripts/train/config/data" / f"{target.data_group}.yaml"
    )
    if not data_config.is_file():
        raise SystemExit(
            f"Stable-WorldModel data-group config not found: {data_config}"
        )
    payload = yaml.safe_load(data_config.read_text(encoding="utf-8")) or {}
    dataset_config = payload.get("dataset", {}) if isinstance(payload, dict) else {}
    keys_to_load = dataset_config.get("keys_to_load")
    if not isinstance(keys_to_load, list) or any(
        not isinstance(name, str) or not name for name in keys_to_load
    ):
        raise SystemExit(
            f"Stable-WorldModel data group {target.data_group!r} does not "
            "declare a valid dataset.keys_to_load list."
        )
    for name in keys_to_load:
        if name not in required:
            required.append(name)
    return tuple(required)


def _family_required_lance_columns(
    *,
    family: str,
    target: "Target",
    stablewm_repo: Path,
) -> tuple[str, ...]:
    """Add the temporal indexing contract required by the Lance reader."""

    return (
        "episode_idx",
        "step_idx",
        *_family_model_columns(
            family=family,
            target=target,
            stablewm_repo=stablewm_repo,
        ),
    )


def _validate_lance_column_contract(
    *,
    path: Path,
    columns: set[str],
    required: tuple[str, ...],
    action_width: int | None,
    expected_action_dim: int,
    target_label: str,
    forbidden_string_columns: tuple[str, ...] = (),
) -> None:
    missing = [name for name in required if name not in columns]
    if missing:
        if target_label == "cube_gripper_carry" and {
            "model_step_idx",
            "action_block",
        }.issubset(columns):
            raise SystemExit(
                "The Cube release projection is a blocked-transition table, "
                "not a per-step Stable-WorldModel sequence. It stores "
                "model_step_idx/action_block and requires the audited "
                "cube_block_projection_to_sequence_v1 adapter; renaming "
                "columns would not reconstruct the missing raw time steps."
            )
        raise SystemExit(
            f"Dataset {path} is not compatible with the selected "
            f"Stable-WorldModel data profile. Missing columns: {missing}; "
            f"available columns: {sorted(columns)}"
        )
    if action_width is not None and action_width != expected_action_dim:
        raise SystemExit(
            f"Dataset {path} has raw action width {action_width}, but target "
            f"{target_label!r} declares {expected_action_dim}. The launcher "
            "does not pad, truncate, or hard-code action dimensions."
        )
    if forbidden_string_columns:
        raise SystemExit(
            f"Dataset {path} stores string metadata on every step: "
            f"{list(forbidden_string_columns)}. The selected "
            "Stable-WorldModel reader rejects this layout before selecting "
            "model inputs. First apply "
            "stablewm_step_metadata_to_episode_table_v1; keys_to_load cannot "
            "bypass the reader's schema check."
        )


def _validate_feature_width(
    *,
    path: Path,
    feature: str,
    observed: int | None,
    expected: int | None,
    target_label: str,
) -> None:
    if observed is not None and expected is not None and observed != expected:
        raise SystemExit(
            f"Dataset {path} has {feature} width {observed}, but target "
            f"{target_label!r} declares {expected}. The launcher does not "
            "pad, truncate, or replace model inputs."
        )


def _selected_loader_rejects_step_strings(stablewm_repo: Path) -> bool:
    source = stablewm_repo / "stable_worldmodel/data/formats/lance.py"
    if not source.is_file():
        return False
    text = source.read_text(encoding="utf-8", errors="ignore")
    return "legacy_strings" in text and "Per-step strings" in text


def _validate_contextworld_format_registry(stablewm_repo: Path) -> None:
    """Fail before GPU allocation when the selected checkout lacks URI formats."""

    format_source = stablewm_repo / "stable_worldmodel/data/format.py"
    utils_source = stablewm_repo / "stable_worldmodel/data/utils.py"
    if not format_source.is_file() or not utils_source.is_file():
        raise SystemExit(
            "ContextWorld-v1 StableWM training needs a checkout "
            "with the public data-format registry. Required files are "
            f"missing below {stablewm_repo}."
        )
    format_text = format_source.read_text(encoding="utf-8", errors="ignore")
    utils_text = utils_source.read_text(encoding="utf-8", errors="ignore")
    if (
        "def register_format" not in format_text
        or "FORMATS" not in format_text
        or "if '://' in name" not in utils_text
    ):
        raise SystemExit(
            "The selected Stable-WorldModel checkout does not expose the "
            "scheme-based data-format registry required by the "
            "ContextWorld-v1 runtime reader."
        )
    try:
        import lance  # noqa: F401
        import pyarrow  # noqa: F401
        from PIL import Image  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "ContextWorld-v1 runtime training requires pylance, pyarrow, "
            "and Pillow. Install contextworld[stablewm]."
        ) from exc


def validate_training_dataset_schema(
    *,
    target: "Target",
    family: str,
    stablewm_repo: Path,
) -> None:
    """Validate model columns and action width before allocating a trainer."""

    if _is_contextworld_dataset_uri(target.dataset):
        _validate_contextworld_format_registry(stablewm_repo)
        try:
            identity = describe_contextworld_dataset(str(target.dataset))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise SystemExit(
                f"Invalid ContextWorld-v1 runtime dataset: {exc}"
            ) from exc
        expected = {
            "component": target.label,
            "history_length": target.history_size,
            "action_dimension": target.action_dim,
        }
        observed = {name: identity[name] for name in expected}
        if observed != expected:
            raise SystemExit(
                "ContextWorld-v1 runtime dataset geometry differs from the "
                f"selected component: expected={expected}, observed={observed}"
            )
        return

    dataset = Path(target.dataset)

    if dataset.is_file():
        try:
            import h5py
        except ImportError as exc:
            raise SystemExit(
                "H5 dataset preflight needs h5py. Install "
                "contextworld[stablewm] in the training environment."
            ) from exc
        try:
            with h5py.File(dataset, "r") as handle:
                columns = set(handle.keys())
                required = _family_model_columns(
                    family=family,
                    target=target,
                    stablewm_repo=stablewm_repo,
                )
                action_width = None
                if "action" in handle:
                    action_shape = handle["action"].shape
                    if action_shape:
                        action_width = (
                            int(action_shape[-1]) if len(action_shape) > 1 else 1
                        )
                _validate_lance_column_contract(
                    path=dataset,
                    columns=columns,
                    required=required,
                    action_width=action_width,
                    expected_action_dim=target.action_dim,
                    target_label=target.label,
                )
                encoding_width = None
                if target.encoding_key and target.encoding_key in handle:
                    encoding_shape = handle[target.encoding_key].shape
                    if encoding_shape:
                        encoding_width = (
                            int(encoding_shape[-1])
                            if len(encoding_shape) > 1 else 1
                        )
                _validate_feature_width(
                    path=dataset,
                    feature=target.encoding_key or "auxiliary input",
                    observed=encoding_width,
                    expected=target.encoding_dim,
                    target_label=target.label,
                )
        except SystemExit:
            raise
        except Exception as exc:
            raise SystemExit(
                f"Could not open H5 dataset for schema preflight: "
                f"{dataset}: {exc}"
            ) from exc
        return

    if not (dataset.is_dir() and dataset.name.endswith(".lance")):
        return
    try:
        import lance
    except ImportError as exc:
        raise SystemExit(
            "Lance dataset preflight needs pylance. Install "
            "contextworld[stablewm] in the training environment."
        ) from exc
    try:
        schema = lance.dataset(str(dataset)).schema
    except Exception as exc:
        raise SystemExit(
            f"Could not open Lance dataset for schema preflight: "
            f"{dataset}: {exc}"
        ) from exc

    columns = set(schema.names)
    forbidden_string_columns: tuple[str, ...] = ()
    if _selected_loader_rejects_step_strings(stablewm_repo):
        forbidden_string_columns = tuple(
            field.name
            for field in schema
            if str(field.type) in {"string", "large_string"}
            and field.name not in {"episode_idx", "step_idx"}
        )
    action_width = None
    if "action" in columns:
        field_type = schema.field("action").type
        size = getattr(field_type, "list_size", None)
        if isinstance(size, int):
            action_width = size
    required = _family_required_lance_columns(
        family=family,
        target=target,
        stablewm_repo=stablewm_repo,
    )
    _validate_lance_column_contract(
        path=dataset,
        columns=columns,
        required=required,
        action_width=action_width,
        expected_action_dim=target.action_dim,
        target_label=target.label,
        forbidden_string_columns=forbidden_string_columns,
    )
    encoding_width = None
    if target.encoding_key and target.encoding_key in columns:
        field_type = schema.field(target.encoding_key).type
        size = getattr(field_type, "list_size", None)
        if isinstance(size, int):
            encoding_width = size
    _validate_feature_width(
        path=dataset,
        feature=target.encoding_key or "auxiliary input",
        observed=encoding_width,
        expected=target.encoding_dim,
        target_label=target.label,
    )


def method_profile(contract: dict[str, Any], method: str) -> dict[str, Any]:
    """Return one registered training-method contract."""

    methods = contract.get("methods")
    if not isinstance(methods, dict) or not isinstance(methods.get(method), dict):
        raise SystemExit(
            f"Unknown training method in profile contract: {method!r}"
        )
    return methods[method]


def method_component_recipe(
    contract: dict[str, Any], method: str, component: str
) -> dict[str, Any] | None:
    """Return the method's registered recipe for one benchmark component."""

    if method == "native":
        return None
    profile = method_profile(contract, method)
    unsupported = profile.get("unsupported_components")
    if isinstance(unsupported, dict) and component in unsupported:
        raise SystemExit(
            f"{method} is unavailable for component {component!r}: "
            f"{unsupported[component]}"
        )
    components = profile.get("components")
    recipe = components.get(component) if isinstance(components, dict) else None
    if not isinstance(recipe, dict):
        raise SystemExit(
            f"Component {component!r} has no registered {method} training "
            "relation contract."
        )
    return recipe


def _validate_method(args: argparse.Namespace, contract: dict[str, Any]) -> None:
    """Keep a non-native method inside its registered support envelope."""

    profile = method_profile(contract, args.method)
    if args.method == "native":
        return
    families = profile.get("families")
    if not isinstance(families, list) or args.family not in families:
        raise SystemExit(
            f"{args.method} is registered only for families "
            f"{families!r}; requested {args.family!r}."
        )
    tracks = profile.get("training_tracks")
    if not isinstance(tracks, list) or args.training_track not in tracks:
        raise SystemExit(
            f"{args.method} is registered only for training tracks "
            f"{tracks!r}; requested {args.training_track!r}."
        )
    if args.original_env or not args.component:
        raise SystemExit(
            f"{args.method} conditions on a public benchmark-component "
            "training relation and is unavailable for original-environment "
            "training."
        )
    method_component_recipe(contract, args.method, args.component)
    if args.dataset:
        raise SystemExit(
            f"{args.method} owns its ContextWorld-v1 training view; "
            "CW_DATASET/--dataset cannot be supplied."
        )
    supplied = {
        "--component-payload": args.component_payload,
        "--mix-original-weight": args.mix_original_weight,
        "--mix-synthetic-weight": args.mix_synthetic_weight,
    }
    requested = sorted(name for name, value in supplied.items() if value is not None)
    if requested:
        raise SystemExit(
            f"{args.method} owns its registered payload and mixture; remove "
            + ", ".join(requested)
        )


def resolve_target(args: argparse.Namespace, contract: dict[str, Any]) -> Target:
    if bool(args.original_env) == bool(args.component):
        raise SystemExit("Select exactly one target: --original-env or --component")

    explicit = args.dataset
    if args.original_env:
        environments = contract["original_environments"]
        if args.original_env not in environments:
            raise SystemExit(f"Unknown original environment: {args.original_env}")
        spec = environments[args.original_env]
        if explicit:
            dataset = _absolute_path(explicit, label="--dataset")
        else:
            if not args.dataset_root:
                raise SystemExit("Original training needs --dataset or --dataset-root "
                                 "(CONTEXTWORLD_DATASET_ROOT).")
            root = _absolute_path(args.dataset_root, label="--dataset-root")
            if not root.is_dir():
                raise SystemExit(f"Dataset root does not exist: {root}")
            dataset = Path(os.path.abspath(root / spec["dataset"]))
        _validate_dataset_payload(dataset)
        return Target(
            label=f"original-{args.original_env}",
            dataset=dataset,
            data_group=args.data_group or spec["data_group"],
            history_size=args.history_size or contract["defaults"]["history_size"],
            action_dim=int(spec["action_dim"]),
            environment=args.original_env,
            encoding_key=str(spec["encoding_key"]),
            encoding_dim=int(spec["encoding_dim"]),
            original_env=args.original_env,
        )

    components = contract["benchmark_components"]
    if args.component not in components:
        raise SystemExit(f"Unknown benchmark component: {args.component}")
    spec = components[args.component]
    if spec.get("model_inputs") != [
        "pixels",
        "action",
    ]:
        raise SystemExit(
            f"Benchmark component {args.component!r} must declare "
            "model_inputs: [pixels, action]."
        )

    runtime_identity: dict[str, Any] | None = None
    if explicit:
        if any(
            value is not None
            for value in (
                args.component_payload,
                args.mix_original_weight,
                args.mix_synthetic_weight,
                args.component_epoch_size,
            )
        ):
            raise SystemExit(
                "Component payload, mixture, and epoch-size options build "
                "the automatic ContextWorld-v1 view and cannot be combined "
                "with an explicit --dataset/CW_DATASET."
            )
        if _is_contextworld_dataset_uri(explicit):
            dataset: Path | str = str(explicit)
            try:
                runtime_identity = describe_contextworld_dataset(dataset)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                raise SystemExit(
                    f"Invalid ContextWorld-v1 runtime dataset: {exc}"
                ) from exc
            if runtime_identity["component"] != args.component:
                raise SystemExit(
                    "The contextworld:// dataset component does not match "
                    f"--component: {runtime_identity['component']!r} != "
                    f"{args.component!r}"
                )
        else:
            dataset = _absolute_path(explicit, label="--dataset")
            _validate_dataset_payload(dataset)
    else:
        recipe = spec.get("joint_scratch_training")
        if not isinstance(recipe, dict):
            raise SystemExit(
                f"Benchmark component {args.component!r} has no registered "
                "joint-scratch runtime training recipe."
            )
        method = getattr(args, "method", "native")
        method_recipe = method_component_recipe(contract, method, args.component)
        if method_recipe is not None:
            # The method owns the payload and mixture it was registered with;
            # `_validate_method` has already rejected operator overrides.
            recipe = {**recipe, **method_recipe}
        root_value = args.benchmark_root
        if not root_value and args.dataset_root:
            root_value = str(
                _absolute_path(args.dataset_root, label="--dataset-root")
                / "ContextWorld-v1"
            )
        if not root_value:
            raise SystemExit(
                "Benchmark component training needs --benchmark-root/"
                "CONTEXTWORLD_BENCHMARK_ROOT, or a --dataset-root containing "
                "ContextWorld-v1."
            )
        benchmark_root = _absolute_path(root_value, label="--benchmark-root")
        original_weight = (
            args.mix_original_weight
            if args.mix_original_weight is not None
            else float(recipe["original_weight"])
        )
        synthetic_weight = (
            args.mix_synthetic_weight
            if args.mix_synthetic_weight is not None
            else float(recipe["synthetic_weight"])
        )
        if original_weight < 0 or synthetic_weight <= 0:
            raise SystemExit(
                "Component mixture weights require original >= 0 and "
                "synthetic > 0."
            )
        original_dataset = None
        if original_weight:
            if not args.dataset_root:
                raise SystemExit(
                    "This component recipe mixes original data. Set "
                    "CONTEXTWORLD_DATASET_ROOT/--dataset-root."
                )
            original_root = _absolute_path(
                args.dataset_root, label="--dataset-root"
            )
            original_relative = contract["original_environments"][
                spec["environment"]
            ]["dataset"]
            original_dataset = original_root / original_relative
        try:
            dataset = build_contextworld_dataset_uri(
                benchmark_root,
                component=args.component,
                split="training",
                payload_id=(
                    args.component_payload
                    or str(recipe.get("payload_id") or "data")
                ),
                original_dataset=original_dataset,
                original_weight=original_weight,
                synthetic_weight=synthetic_weight,
                epoch_size=args.component_epoch_size,
                conditional_joint_method=(
                    None if method_recipe is None else method
                ),
            )
            runtime_identity = describe_contextworld_dataset(dataset)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise SystemExit(
                f"Could not construct ContextWorld-v1 training view: {exc}"
            ) from exc
        if method_recipe is not None:
            observed = runtime_identity.get("conditional_joint") or {}
            expected = int(method_recipe["group_width"])
            if int(observed.get("group_width", -1)) != expected:
                raise SystemExit(
                    f"{method} group_width disagrees with the registered "
                    f"contract for {args.component!r}: expected={expected}, "
                    f"observed={observed.get('group_width')}"
                )
            if str(observed.get("relation_kind")) != str(
                method_recipe["relation"]
            ):
                raise SystemExit(
                    f"{method} relation kind disagrees with the registered "
                    f"contract for {args.component!r}: "
                    f"expected={method_recipe['relation']!r}, "
                    f"observed={observed.get('relation_kind')!r}"
                )
    if runtime_identity is not None:
        required_history = int(runtime_identity["history_length"])
        required_frameskip = int(runtime_identity["frameskip"])
        if args.history_size is not None and args.history_size != required_history:
            raise SystemExit(
                f"Component {args.component!r} requires history "
                f"{required_history}, not {args.history_size}."
            )
        if args.frameskip is not None and args.frameskip != required_frameskip:
            raise SystemExit(
                f"Component {args.component!r} requires frameskip "
                f"{required_frameskip}, not {args.frameskip}."
            )

    return Target(
        label=args.component,
        dataset=dataset,
        data_group=args.data_group or spec.get("data_group"),
        history_size=(
            int(runtime_identity["history_length"])
            if runtime_identity is not None
            else args.history_size or int(spec["history_size"])
        ),
        action_dim=int(spec["action_dim"]),
        environment=str(spec["environment"]),
        encoding_key=None,
        encoding_dim=None,
    )


def resolve_stablewm_repo(args: argparse.Namespace) -> Path:
    candidate = (args.stablewm_repo or _env("CONTEXTWORLD_STABLE_WORLDMODEL_REPO")
                 or _env("STABLEWM_REPO"))
    if not candidate:
        raise SystemExit("Set --stablewm-repo or CONTEXTWORLD_STABLE_WORLDMODEL_REPO "
                         "to a Stable-WorldModel source checkout.")
    repo = _absolute_path(candidate, label="Stable-WorldModel checkout")
    if not (repo / "scripts/train").is_dir():
        raise SystemExit(f"Stable-WorldModel training code not found: {repo}")
    return repo


def resolve_checkpoint_root(args: argparse.Namespace) -> Path:
    candidate = args.checkpoint_root or _env("STABLEWM_HOME")
    if not candidate:
        raise SystemExit("Set --checkpoint-root/CW_CHECKPOINT_ROOT or STABLEWM_HOME. "
                         "Stable-WorldModel writes models below <root>/checkpoints/.")
    return _absolute_path(candidate, label="Checkpoint root")


def _nested(mapping: dict[str, Any], *keys: str) -> Any:
    node: Any = mapping
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def _load_upstream_config(stablewm_repo: Path,
                          profile: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    path = (stablewm_repo / "scripts/train/config" / f"{profile['config_name']}.yaml")
    if not path.is_file():
        raise SystemExit(f"Upstream training config not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise SystemExit(f"Upstream training config is not a mapping: {path}")
    # Hydra supports a thin method config inheriting a sibling config. Resolve
    # that one local layer for static capability checks without importing the
    # training environment's Hydra runtime. Config-group defaults remain owned
    # by Hydra and are intentionally ignored here.
    defaults = payload.get("defaults", [])
    inherited = {}
    if isinstance(defaults, list):
        for item in defaults:
            if not isinstance(item, str) or item == "_self_" or "/" in item:
                continue
            base_path = path.with_name(f"{item}.yaml")
            if not base_path.is_file():
                continue
            base_payload = (
                yaml.safe_load(base_path.read_text(encoding="utf-8")) or {}
            )
            if not isinstance(base_payload, dict):
                raise SystemExit(
                    f"Inherited upstream config is not a mapping: {base_path}"
                )
            inherited = _deep_merge(inherited, base_payload)
    payload = _deep_merge(inherited, payload)
    return path, payload


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Return a recursive mapping merge matching Hydra's common map case."""

    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _add(entries: list[str], key: str | None, value: Any) -> None:
    if value is None or key is None:
        return
    if isinstance(value, bool):
        value = _hydra_bool(value)
    entries.append(f"{key}={value}")


def _validate_positive(name: str, value: int | float | None) -> None:
    if value is not None and value <= 0:
        raise SystemExit(f"{name} must be positive; got {value}")


def _validate_args(
    args: argparse.Namespace,
    family: str,
    target: Target | None = None,
) -> None:
    for name in (
            "batch_size",
            "frameskip",
            "history_size",
            "num_preds",
            "max_epochs",
            "accumulate",
            "prefetch_factor",
            "embed_dim",
            "component_epoch_size",
            "eval_batch_size",
            "eval_num",
    ):
        _validate_positive(f"--{name.replace('_', '-')}", getattr(args, name))
    if args.num_workers is not None and args.num_workers <= 0:
        raise SystemExit(
            "--num-workers must be positive. The supported Stable-WorldModel "
            "loaders enable persistent workers and/or prefetching, which "
            "PyTorch rejects when num_workers=0."
        )
    if args.train_split is not None and not 0 < args.train_split < 1:
        raise SystemExit("--train-split must be between 0 and 1")
    if args.learning_rate is not None and args.learning_rate <= 0:
        raise SystemExit("--learning-rate must be positive")
    if args.weight_decay is not None and args.weight_decay < 0:
        raise SystemExit("--weight-decay must be non-negative")
    if args.gradient_clip_val is not None and args.gradient_clip_val < 0:
        raise SystemExit("--gradient-clip-val must be non-negative")
    if args.trainer_default_root_dir is not None:
        _absolute_path(
            args.trainer_default_root_dir,
            label="--trainer-default-root-dir",
        )
    if family == "prejepa":
        unsupported = {
            "--persistent-workers": args.persistent_workers,
            "--prefetch-factor": args.prefetch_factor,
            "--pin-memory": args.pin_memory,
            "--embed-dim": args.embed_dim,
        }
        requested = [name for name, value in unsupported.items() if value is not None]
        if requested:
            raise SystemExit("PreJEPA's trainer does not expose " +
                             ", ".join(requested))
    if family != "lewm" and args.lewm_sigreg_weight is not None:
        raise SystemExit("--lewm-sigreg-weight is only valid with --family lewm")
    viswm_requested = any(value is not None for value in (
        args.viswm_weight,
        args.viswm_num_projections,
        args.viswm_lambda_scale,
        args.viswm_lambda_shape,
        args.viswm_lambda_center,
    ))
    if family != "viswm" and viswm_requested:
        raise SystemExit("VIS-WM loss options are only valid with --family viswm")
    for override in args.override:
        key, separator, value = override.partition("=")
        normalized = key.lstrip("+~")
        if not separator:
            raise SystemExit(f"--override must be KEY=VALUE; got {override!r}")
        if normalized in {
                "seed",
                "output_model_name",
                "subdir",
                "data.dataset.name",
                "dataset_name",
                "hydra.run.dir",
                "loss.regularizer",
        }:
            raise SystemExit(
                f"--override cannot replace launcher-owned identity/path key "
                f"{normalized}; use the corresponding typed option.")
        if (
            "${" in value
            or normalized == "defaults"
            or normalized.startswith("hydra.")
            or "@" in normalized
        ):
            raise SystemExit(
                "Raw overrides cannot use Hydra resolvers, defaults/config "
                "routing, package redirects, or hydra.* keys because their "
                "resolved recipe cannot be bound to the launcher identity: "
                f"{normalized}"
            )
        lowered = normalized.lower()
        if any(word in lowered for word in ("api_key", "password", "secret", "token")):
            raise SystemExit("Secrets must be injected through standard environment "
                             f"variables, not --override {normalized}.")
        if (
            family == "prejepa"
            and target is not None
            and target.original_env is None
            and (normalized == "wm.encoding" or normalized.startswith("wm.encoding."))
        ):
            raise SystemExit(
                "Benchmark PreJEPA fixes model inputs to pixels and action; "
                "--override cannot add or replace wm.encoding streams."
            )
        if (
            target is not None
            and _is_contextworld_dataset_uri(target.dataset)
            and normalized
            in {
                "data.dataset.keys_to_load",
                "data.dataset.keys_to_cache",
                "data.dataset.keys_to_merge",
            }
        ):
            raise SystemExit(
                "ContextWorld-v1 fixes model-visible columns to pixels and "
                "action; raw overrides cannot replace its dataset key contract."
            )
    if target is not None and _is_contextworld_dataset_uri(target.dataset):
        if args.dataset_sampling is not None or args.balance_val is not None:
            raise SystemExit(
                "ContextWorld-v1 owns scenario and mixture balancing; "
                "--dataset-sampling and --balance-val are unavailable."
            )
        if args.dataset_item:
            raise SystemExit(
                "ContextWorld-v1 resolves its manifest-bound members; "
                "--dataset-item cannot replace them."
            )
    if args.post_eval and args.eval_epoch is None and any(
            item.lstrip("+").startswith("trainer.max_epochs=")
            for item in args.override):
        raise SystemExit("Raw trainer.max_epochs with --post-eval also requires an "
                         "explicit --eval-epoch.")


def _logger_overrides(
    args: argparse.Namespace,
    *,
    family: str,
    profile: dict[str, Any],
    upstream_config: dict[str, Any],
    trainer_script: Path,
    run_name: str,
) -> list[str]:
    backend = args.logger
    if backend not in profile["logger_backends"]:
        supported = ", ".join(profile["logger_backends"])
        raise SystemExit(
            f"{family} does not support logger {backend!r}; use {supported}.")

    if family == "prejepa":
        source = trainer_script.read_text(encoding="utf-8")
        uses_common_logger = (
            "build_training_logger" in source
            and "logger_backend" in upstream_config
        )
        if not uses_common_logger:
            if backend == "swanlab":
                raise SystemExit(
                    "The selected PreJEPA trainer does not call "
                    "build_training_logger and therefore cannot consume "
                    "SwanLab settings. Use a compatible Stable-WorldModel "
                    "checkout or --logger none."
                )
            reads_wandb = "cfg.wandb" in source
            if not reads_wandb:
                if backend != "none":
                    raise SystemExit(
                        "This PreJEPA trainer has no logger integration. "
                        "Use --logger none."
                    )
                return []
            entries = [f"++wandb.enabled={_hydra_bool(backend == 'wandb')}"]
            if backend == "wandb":
                for key, value in (
                    ("project", args.wandb_project),
                    ("entity", args.wandb_entity),
                    ("name", args.tracker_name or run_name),
                    ("id", args.tracker_id or run_name),
                ):
                    _add(entries, f"++wandb.config.{key}", value)
            return entries

        # Current compatible checkouts use the same logger factory as
        # LeWM/PLDM, so the common mapping below is authoritative.

    # The pinned public upstream config has no logger block.  A checkout that
    # implements logging declares both keys; otherwise accepting the option
    # would create a valid-looking command that records no metrics.
    has_logger_integration = ("logger_backend" in upstream_config
                              and backend in upstream_config and "build_training_logger"
                              in trainer_script.read_text(encoding="utf-8"))
    if backend != "none" and not has_logger_integration:
        raise SystemExit(f"The selected {family} checkout does not implement {backend} "
                         "logging. Logging is optional for reproduction; use --logger "
                         "none or a checkout/extension that declares the integration.")
    if "logger_backend" not in upstream_config:
        return []

    entries = [f"logger_backend={backend}"]
    if backend == "swanlab":
        entries.append("swanlab.enabled=true")
        for key, value in (
            ("project", args.swanlab_project),
            ("workspace", args.swanlab_workspace),
            ("experiment_name", args.tracker_name or run_name),
            ("id", args.tracker_id or run_name),
            ("logdir", args.swanlab_logdir),
            ("mode", args.swanlab_mode),
        ):
            _add(entries, f"swanlab.config.{key}", value)
        _add(entries, "swanlab.collect_hardware", args.swanlab_collect_hardware)
        _add(entries, "swanlab.hardware_monitor", args.swanlab_hardware_monitor)
        _add(entries, "swanlab.log_hyperparams", args.swanlab_log_hyperparams)
    elif backend == "wandb":
        entries.append("wandb.enabled=true")
        for key, value in (
            ("project", args.wandb_project),
            ("entity", args.wandb_entity),
            ("name", args.tracker_name or run_name),
            ("id", args.tracker_id or run_name),
        ):
            _add(entries, f"wandb.config.{key}", value)
    return entries


def build_overrides(
    args: argparse.Namespace,
    contract: dict[str, Any],
    target: Target,
    *,
    run_name: str,
    seed: int,
    stablewm_repo: Path,
) -> list[str]:
    family = args.family
    profile = contract["families"][family]
    common = contract["common_keys"]
    keys = profile["keys"]
    trainer_script = stablewm_repo / profile["entrypoint"]
    if not trainer_script.is_file():
        raise SystemExit(f"Upstream trainer not found: {trainer_script}")
    _, upstream_config = _load_upstream_config(stablewm_repo, profile)

    _validate_args(args, family, target)
    if profile["data_group"] and not target.data_group:
        raise SystemExit(f"{family} requires --data-group for a non-original dataset")
    if not profile["data_group"] and args.data_group:
        raise SystemExit(f"{family} does not use a Hydra data group")

    entries: list[str] = []
    if profile["data_group"]:
        entries.append(f"data={target.data_group}")
    _add(entries, keys["dataset"], target.dataset)
    if profile["data_group"] and _is_contextworld_dataset_uri(target.dataset):
        # LeWM/PLDM's environment YAMLs normally request privileged simulator
        # state. The public component bundle deliberately exposes only RGB and
        # actions, which are also the only inputs their native model forward
        # methods consume. Own these Hydra keys here so all three families see
        # the same model-visible training data.
        entries.extend(
            [
                "data.dataset.keys_to_load=[pixels,action]",
                "data.dataset.keys_to_cache=[action]",
            ]
        )
        data_config = (
            stablewm_repo
            / "scripts/train/config/data"
            / f"{target.data_group}.yaml"
        )
        data_payload = yaml.safe_load(data_config.read_text(encoding="utf-8")) or {}
        data_dataset = (
            data_payload.get("dataset", {})
            if isinstance(data_payload, dict)
            else {}
        )
        if isinstance(data_dataset, dict) and "keys_to_merge" in data_dataset:
            entries.append("~data.dataset.keys_to_merge")
    _add(entries, common["seed"], seed)
    _add(entries, common["run_name"], run_name)
    _add(entries, common["run_subdir"], run_name)

    frameskip = args.frameskip or contract["defaults"]["frameskip"]
    num_preds = args.num_preds or contract["defaults"]["num_preds"]
    num_workers = args.num_workers
    if num_workers is None and _is_contextworld_dataset_uri(target.dataset):
        # Family defaults range from 6 to 16 workers *per DDP rank*. Keep the
        # public Lance/PyArrow bundle's safe tested default bounded per rank,
        # while preserving CW_NUM_WORKERS as an operator override and leaving
        # original-H5 training unchanged.
        num_workers = int(
            contract["defaults"].get(
                "contextworld_num_workers_per_rank", 2
            )
        )

    precision = args.precision
    if (
        precision is None
        and family == "prejepa"
        and target.label == "action_delay"
    ):
        # History=7 increases the numerical range seen by PreJEPA's predictor.
        # All three component runs made finite progress under 16-mixed and then
        # failed at different epochs with a non-finite loss.  BF16 keeps mixed
        # precision throughput while avoiding FP16's narrow exponent range.
        # Operators can still select another mode explicitly with CW_PRECISION.
        precision = contract["defaults"].get(
            "prejepa_action_delay_precision", "bf16-mixed"
        )

    for name, value in (
        ("batch_size", args.batch_size),
        ("num_workers", num_workers),
        ("persistent_workers", args.persistent_workers),
        ("prefetch_factor", args.prefetch_factor),
        ("pin_memory", args.pin_memory),
        ("frameskip", frameskip),
        ("history_size", target.history_size),
        ("num_preds", num_preds),
        ("embed_dim", args.embed_dim),
    ):
        _add(entries, keys.get(name), value)

    for name, value in (
        ("train_split", args.train_split),
        ("max_epochs", args.max_epochs),
        ("devices", args.devices),
        ("accelerator", args.accelerator),
        ("strategy", args.strategy),
        ("precision", precision),
        ("accumulate_grad_batches", args.accumulate),
        ("gradient_clip_val", args.gradient_clip_val),
        ("fast_dev_run", args.fast_dev_run),
        ("limit_train_batches", args.limit_train_batches),
        ("limit_val_batches", args.limit_val_batches),
        ("trainer_default_root_dir", args.trainer_default_root_dir),
        ("optimizer_lr", args.learning_rate),
        ("optimizer_weight_decay", args.weight_decay),
        ("hydra_job_chdir", args.hydra_job_chdir),
    ):
        _add(entries, common.get(name), value)

    if args.output:
        output = _absolute_path(args.output, label="--output") / run_name
        _add(entries, common["hydra_run_dir"], output)

    if family == "prejepa":
        if target.original_env is None:
            # Benchmark components are evaluated with the frozen RGB/action
            # ICL contract.  Remove the upstream default rather than mapping
            # a component's privileged observation/proprio column into it.
            entries.append("~wm.encoding.proprio")
        # Original-environment DINO-WM keeps its upstream state-conditioned
        # recipe.  Reacher and Cube name that state column ``observation``.
        elif target.encoding_key != "proprio":
            entries.extend([
                "~wm.encoding.proprio",
                f"+wm.encoding.{target.encoding_key}=10",
            ])

    if args.dataset_sampling is not None:
        if not profile["data_group"]:
            raise SystemExit("--dataset-sampling is only valid for grouped data")
        _add(entries, "data.dataset.sampling", args.dataset_sampling)
    if args.balance_val is not None and not profile["data_group"]:
        raise SystemExit("--balance-val is only valid for grouped data")
    _add(
        entries,
        "data.dataset.balance_val" if profile["data_group"] else None,
        args.balance_val,
    )
    if args.dataset_item:
        if not profile["data_group"]:
            raise SystemExit("--dataset-item is only valid for LeWM/PLDM")
        for item in args.dataset_item:
            index, separator, value = item.partition("=")
            if not separator or not index.isdigit():
                raise SystemExit("--dataset-item must be INDEX=/absolute/path/to/table")
            item_path = _absolute_path(value, label="--dataset-item")
            _validate_dataset_payload(item_path)
            _add(entries, f"data.dataset.items.{int(index)}.name", item_path)

    if family == "lewm":
        _add(entries, "loss.sigreg.weight", args.lewm_sigreg_weight)
    if family == "viswm":
        if _nested(upstream_config, "loss", "regularizer") != "visreg":
            raise SystemExit(
                "The selected VIS-WM config must fix loss.regularizer=visreg; "
                "do not emulate VIS-WM through a LeWM objective override."
            )
        for key, value in (
            ("weight", args.viswm_weight),
            ("kwargs.num_projections", args.viswm_num_projections),
            ("kwargs.lambda_scale", args.viswm_lambda_scale),
            ("kwargs.lambda_shape", args.viswm_lambda_shape),
            ("kwargs.lambda_center", args.viswm_lambda_center),
        ):
            if value is not None and _nested(upstream_config, "loss", "visreg") is None:
                raise SystemExit(
                    "This VIS-WM checkout has no VISReg config; use a compatible "
                    "checkout rather than adding ignored Hydra keys.")
            _add(entries, f"loss.visreg.{key}", value)

    entries.extend(
        _method_overrides(
            args,
            contract,
            target,
            upstream_config=upstream_config,
        )
    )

    entries.extend(
        _logger_overrides(
            args,
            family=family,
            profile=profile,
            upstream_config=upstream_config,
            trainer_script=trainer_script,
            run_name=run_name,
        ))
    entries.extend(args.override)
    return entries


def _method_overrides(
    args: argparse.Namespace,
    contract: dict[str, Any],
    target: Target,
    *,
    upstream_config: dict[str, Any],
) -> list[str]:
    """Render the selected method's registered loss keys, or nothing.

    ContextWorld does not implement the objective. It only enables the
    checkout's own one-step conditional-joint interface and tells it how wide
    a relation group is, so a checkout without that interface fails here
    rather than accepting silently ignored Hydra keys.
    """

    method = getattr(args, "method", "native")
    if method == "native":
        return []
    profile = method_profile(contract, method)
    recipe = method_component_recipe(contract, method, target.label)
    assert recipe is not None
    if not _is_contextworld_dataset_uri(target.dataset):
        raise SystemExit(
            f"{method} requires the registered ContextWorld-v1 training view."
        )
    group = profile.get("upstream_config_group")
    if group and _nested(upstream_config, "loss", str(group)) is None:
        raise SystemExit(
            f"This {args.family} checkout does not expose loss.{group}; "
            f"{method} requires a checkout whose one-step conditional-joint "
            "interface is present. ContextWorld does not add a loss family."
        )
    identity = describe_contextworld_dataset(str(target.dataset))
    observed = identity.get("conditional_joint") or {}
    group_width = int(observed.get("group_width", -1))
    if group_width != int(recipe["group_width"]):
        raise SystemExit(
            f"{method} group_width disagrees with the registered contract for "
            f"{target.label!r}: expected={recipe['group_width']}, "
            f"observed={observed.get('group_width')}"
        )
    keys = profile.get("keys")
    if not isinstance(keys, dict) or not {
        "enabled",
        "weight",
        "group_width",
    } <= set(keys):
        raise SystemExit(
            f"Training method {method!r} does not register its enabled/weight/"
            "group_width Hydra keys."
        )
    entries: list[str] = []
    _add(entries, keys["enabled"], True)
    _add(entries, keys["weight"], float(profile["weight"]))
    _add(entries, keys["group_width"], group_width)
    # The relation-aware sampler owns DDP sharding. Some upstream family
    # configs do not predeclare this optional Lightning key, so use Hydra's
    # add-or-override form across every supported family.
    _add(entries, "++trainer.use_distributed_sampler", False)
    return entries


def _run_name(args: argparse.Namespace, target: Target, seed: int,
              seeds: tuple[int, ...]) -> str:
    default_base = (
        f"{target.original_env}_{args.family}_original"
        if target.original_env
        else f"{target.label}_{args.family}_{args.training_track}"
    )
    method = getattr(args, "method", "native")
    if method != "native":
        # A different objective must not share a run directory (and therefore
        # an immutable training identity) with its native counterpart.
        default_base = f"{default_base}_{method}"
    base = args.run_name or default_base
    name = f"{base}_s{seed}" if len(seeds) > 1 or args.run_name is None else base
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", name):
        raise SystemExit(
            "Run names must be 1-128 characters using letters, numbers, '.', "
            "'_' or '-', and must start with a letter or number.")
    return name


def _evaluation_checkpoint(root: Path, run_name: str, epoch: int) -> Path:
    return root / "checkpoints" / run_name / f"weights_epoch_{epoch}.pt"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stablewm_source_identity(
    stablewm_repo: Path,
    trainer_script: Path,
) -> dict[str, Any]:
    """Fingerprint code and YAML inputs that can change the training recipe."""

    candidates = {trainer_script}
    candidates.update(
        path
        for root, pattern in (
            (stablewm_repo / "stable_worldmodel", "*.py"),
            (stablewm_repo / "scripts/train/config", "*.yaml"),
        )
        if root.is_dir()
        for path in root.rglob(pattern)
        if path.is_file()
    )
    pyproject = stablewm_repo / "pyproject.toml"
    if pyproject.is_file():
        candidates.add(pyproject)

    digest = hashlib.sha256()
    for path in sorted(candidates):
        relative = path.relative_to(stablewm_repo).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(_sha256_file(path)))
    return {
        "sha256": digest.hexdigest(),
        "file_count": len(candidates),
    }


def _training_dependency_identity() -> dict[str, Any]:
    """Record direct runtime versions and SPT source used for recovery."""

    result: dict[str, Any] = {}
    for distribution_name in (
        "stable-pretraining",
        "torch",
        "lightning",
        "hydra-core",
        "omegaconf",
        "transformers",
    ):
        try:
            distribution = importlib.metadata.distribution(distribution_name)
        except importlib.metadata.PackageNotFoundError:
            result[distribution_name] = {"version": None}
            continue
        entry: dict[str, Any] = {"version": distribution.version}
        if distribution_name == "stable-pretraining":
            source_files = [
                relative
                for relative in (distribution.files or ())
                if relative.suffix == ".py"
                and "stable_pretraining" in relative.parts
            ]
            digest = hashlib.sha256()
            included = 0
            for relative in sorted(source_files, key=str):
                path = Path(distribution.locate_file(relative))
                if not path.is_file():
                    continue
                digest.update(str(relative).encode("utf-8"))
                digest.update(b"\0")
                digest.update(bytes.fromhex(_sha256_file(path)))
                included += 1
            entry.update(
                source_sha256=digest.hexdigest(),
                source_file_count=included,
            )
        result[distribution_name] = entry
    return result


def validate_stablepretraining_version() -> str:
    """Require the upstream Manager API used for portable full-state resume."""

    try:
        installed = importlib.metadata.version("stable-pretraining")
    except importlib.metadata.PackageNotFoundError as exc:
        raise SystemExit(
            "Stable-WorldModel training requires stable-pretraining>=0.1.8, "
            "but stable-pretraining is not installed."
        ) from exc
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", installed)
    if match is None or tuple(map(int, match.groups())) < (
        MINIMUM_STABLE_PRETRAINING_VERSION
    ):
        raise SystemExit(
            "Stable-WorldModel training requires stable-pretraining>=0.1.8 "
            "for portable full-state resume; installed version is "
            f"{installed}. Keep only a compatible stable_pretraining wheel "
            "in the offline package directory and reinstall it before "
            "starting the job."
        )
    return installed


def _dataset_identity(path: Path | str) -> dict[str, Any]:
    """Record a cheap, fail-closed identity without hashing a multi-GB dataset."""

    if _is_contextworld_dataset_uri(path):
        return {
            "kind": "contextworld_bundle_uri",
            "uri": str(path),
            "contract": describe_contextworld_dataset(str(path)),
            "adapter_source_sha256": _sha256_file(STABLEWM_BUNDLE_ADAPTER),
        }

    path = Path(path)

    if path.is_file():
        stat = path.stat()
        return {
            "kind": "file",
            "path": str(path),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }

    digest = hashlib.sha256()
    count = 0
    total_size = 0
    for item in sorted(candidate for candidate in path.rglob("*")
                       if candidate.is_file()):
        stat = item.stat()
        relative = item.relative_to(path).as_posix()
        digest.update(
            f"{relative}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode("utf-8")
        )
        count += 1
        total_size += stat.st_size
    return {
        "kind": "directory",
        "path": str(path),
        "file_count": count,
        "total_size": total_size,
        "metadata_sha256": digest.hexdigest(),
    }


def _training_identity_document(
    *,
    args: argparse.Namespace,
    target: Target,
    profile: dict[str, Any],
    stablewm_repo: Path,
    trainer_script: Path,
    run_name: str,
    seed: int,
    overrides: list[str],
) -> dict[str, Any]:
    """Create the exact launcher recipe used to authorize automatic skipping."""

    identity = {
        "family": args.family,
        "training_track": args.training_track,
        "method": getattr(args, "method", "native"),
        "run_name": run_name,
        "seed": seed,
        "target": {
            "label": target.label,
            "environment": target.environment,
            "history_size": target.history_size,
            "action_dim": target.action_dim,
            "encoding_key": target.encoding_key,
            "encoding_dim": target.encoding_dim,
            "dataset": _dataset_identity(target.dataset),
        },
        "profile": {
            "path": str(args.profile_config.resolve()),
            "sha256": _sha256_file(args.profile_config.resolve()),
            "config_name": profile["config_name"],
            "entrypoint": profile["entrypoint"],
        },
        "contextworld_family_entry": {
            "path": str(FAMILY_ENTRY_SCRIPT),
            "sha256": _sha256_file(FAMILY_ENTRY_SCRIPT),
            "sitecustomize_path": str(STABLEWM_SITECUSTOMIZE),
            "sitecustomize_sha256": _sha256_file(STABLEWM_SITECUSTOMIZE),
            "role": "upstream_manager_full_state_resume_bridge",
        },
        "stablewm_source": _stablewm_source_identity(
            stablewm_repo,
            trainer_script,
        ),
        "training_dependencies": _training_dependency_identity(),
        "hydra_overrides": overrides,
    }
    serialized = json.dumps(
        identity,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "schema_version": TRAINING_IDENTITY_SCHEMA,
        "identity_sha256": hashlib.sha256(serialized).hexdigest(),
        "identity": identity,
    }


def _read_training_identity(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Could not read training identity {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"Training identity is not a mapping: {path}")
    return payload


def _assert_training_identity(
    path: Path,
    expected: dict[str, Any],
    *,
    context: str,
) -> None:
    if not path.is_file():
        raise SystemExit(
            f"{context} has no launcher training identity: {path}. Use "
            "CW_EVAL_ONLY=1 only after reviewing the checkpoint manually."
        )
    observed = _read_training_identity(path)
    if observed != expected:
        raise SystemExit(
            f"{context} training identity differs from this request: "
            f"saved={observed.get('identity_sha256')!r}, "
            f"requested={expected['identity_sha256']!r}. Use CW_EVAL_ONLY=1 "
            "only after reviewing the difference."
        )


def _install_training_identity(
    checkpoint_root: Path,
    run_name: str,
    expected: dict[str, Any],
    *,
    replace_preflight_reservation: bool = False,
) -> None:
    """Write once for a new run; require an exact match on native requeue."""

    run_dir = checkpoint_root / "checkpoints" / run_name
    path = run_dir / TRAINING_IDENTITY_FILENAME
    if path.exists():
        observed = _read_training_identity(path)
        if observed == expected:
            return
        if (
            replace_preflight_reservation
            and (
                _preflight_reservation_identity(checkpoint_root, run_name)
                is not None
                or _zero_step_failed_training_identity(
                    checkpoint_root,
                    run_name,
                )
                is not None
            )
        ):
            temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
            try:
                with temporary.open("x", encoding="utf-8") as stream:
                    stream.write(
                        json.dumps(expected, indent=2, sort_keys=True) + "\n"
                    )
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)
            print(
                "[stablewm-train] replaced stale zero-progress identity: "
                f"saved={observed.get('identity_sha256')!r} "
                f"requested={expected['identity_sha256']!r}"
            )
            return
        _assert_training_identity(path, expected, context=f"Run {run_name!r}")
    if run_dir.exists() and any(run_dir.iterdir()):
        raise SystemExit(
            f"Existing run {run_name!r} has no {TRAINING_IDENTITY_FILENAME}; "
            "refusing to attach the current recipe to older artifacts."
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        # Cloud object/NFS mounts may reject hard links even when ordinary
        # exclusive file creation works.  Write the immutable target itself
        # with O_EXCL semantics; a crash can leave only a fail-closed partial
        # identity, never a silently replaced recipe.
        with path.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(expected, indent=2, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        _assert_training_identity(path, expected, context=f"Run {run_name!r}")


def _validate_completed_training_identity(
    checkpoint: Path,
    *,
    expected: dict[str, Any],
) -> None:
    """Bind an automatic eval hand-off to the completed training request.

    A non-empty ``weights_epoch_N.pt`` proves only that some run used the same
    directory name. The launcher's immutable identity records the complete
    Hydra override vector, dataset metadata, family profile and relevant
    StableWM source tree. Legacy checkpoints can still be evaluated
    intentionally with ``--eval-only``; they are never accepted here.
    """

    config_path = checkpoint.parent / "config.yaml"
    if not config_path.is_file():
        raise SystemExit(
            "Automatic post-training evaluation found a target checkpoint "
            f"but no resolved training config: {config_path}. Use "
            "CW_EVAL_ONLY=1 only if evaluating that checkpoint is intentional."
        )
    identity_path = checkpoint.parent / TRAINING_IDENTITY_FILENAME
    _assert_training_identity(
        identity_path,
        expected,
        context="Automatic post-training evaluation checkpoint",
    )


def _stablepretraining_native_requeue() -> bool:
    value = os.environ.get("SLURM_RESTART_COUNT", "0")
    try:
        restart_count = int(value)
    except ValueError as exc:
        raise SystemExit("SLURM_RESTART_COUNT must be an integer") from exc
    return bool(os.environ.get("SLURM_JOB_ID")) and restart_count >= 1


def _validate_scheduler_seed_isolation(training_runs: list[str]) -> None:
    """Prevent StablePretraining's job-scoped index crossing seed runs.

    StablePretraining keys native requeue state by SLURM job/array-task, not
    by ContextWorld's run name. Multiple trainers launched serially inside one
    such task would therefore share one recovery index. Non-SLURM launchers do
    not use that index and retain the comma-separated seed convenience.
    """

    if os.environ.get("SLURM_JOB_ID") and len(training_runs) > 1:
        raise SystemExit(
            "A requeue-capable SLURM job may train only one CW_SEEDS value. "
            "Submit one seed per job or array task so StablePretraining's "
            "job-scoped recovery index cannot restore a different seed. "
            f"Planned training runs: {', '.join(training_runs)}"
        )


def _stablepretraining_run_records(
    root: Path,
) -> list[tuple[Path, dict[str, Any]]]:
    """Read every safe ContextWorld binding below the SPT run root."""

    runs_root = root / "runs"
    if not runs_root.is_dir():
        return []
    records: list[tuple[Path, dict[str, Any]]] = []
    # StablePretraining currently uses runs/YYYYMMDD/HHMMSS/<uuid>/, but the
    # marker is the public binding contract and avoids coupling recovery to a
    # particular date-bucket layout.
    for marker in runs_root.rglob(SPT_RUN_MARKER_FILENAME):
        run_dir = marker.parent
        relative = run_dir.relative_to(runs_root)
        cursor = runs_root
        unsafe_ancestor = False
        for part in relative.parts:
            cursor = cursor / part
            if cursor.is_symlink():
                unsafe_ancestor = True
                break
        if unsafe_ancestor or not run_dir.is_dir():
            raise SystemExit(f"Unsafe StablePretraining run directory: {run_dir}")
        if not marker.is_file() or marker.is_symlink():
            raise SystemExit(f"Unsafe StablePretraining run marker: {marker}")
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(
                f"Could not read StablePretraining run marker {marker}: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise SystemExit(f"StablePretraining run marker is not a mapping: {marker}")
        if payload.get("schema_version") != SPT_RUN_MARKER_SCHEMA:
            raise SystemExit(f"Unsupported StablePretraining run marker: {marker}")
        records.append((run_dir, payload))
    return records


def _validate_reset_source(path: Path, *, kind: str) -> bool:
    """Return whether an exact reset target exists, rejecting unsafe shapes."""

    if path.is_symlink():
        raise SystemExit(f"Refusing to reset symlinked {kind}: {path}")
    if not path.exists():
        return False
    if not path.is_dir():
        raise SystemExit(f"Reset target is not a directory ({kind}): {path}")
    return True


def _plan_run_reset(
    checkpoint_root: Path,
    run_name: str,
    *,
    output_root: Path | None,
) -> RunResetPlan:
    """Find only state that is explicitly bound to ``run_name``.

    Planning is read-only so ``CW_PRINT_ONLY=1`` can show the exact scope
    without changing persistent storage. StablePretraining directories are
    selected by ContextWorld's marker rather than by their date/UUID layout.
    """

    checkpoint_namespace = (
        checkpoint_root / RESET_ARCHIVE_DIRNAME / run_name
    )
    moves: list[RunResetMove] = []
    checkpoint_dir = checkpoint_root / "checkpoints" / run_name
    if _validate_reset_source(checkpoint_dir, kind="StableWM checkpoint directory"):
        moves.append(
            RunResetMove(
                kind="stablewm_checkpoint",
                source=checkpoint_dir,
                archive_namespace=checkpoint_namespace,
                archive_relative=Path("stablewm_checkpoint"),
            )
        )

    runs_root = checkpoint_root / "runs"
    if runs_root.is_symlink():
        raise SystemExit(
            f"Refusing to reset through a symlinked StablePretraining root: "
            f"{runs_root}"
        )
    for run_dir, marker in _stablepretraining_run_records(checkpoint_root):
        if marker.get("run_name") != run_name:
            continue
        # _stablepretraining_run_records already rejects symlinked marker
        # paths and unsafe ancestors below runs_root. Keep a second exact
        # shape check here so all targets are validated before any rename.
        if not _validate_reset_source(
            run_dir,
            kind="StablePretraining run directory",
        ):
            continue
        moves.append(
            RunResetMove(
                kind="stablepretraining_run",
                source=run_dir,
                archive_namespace=checkpoint_namespace,
                archive_relative=(
                    Path("stablepretraining") / run_dir.relative_to(runs_root)
                ),
            )
        )

    if output_root is not None:
        hydra_dir = output_root / run_name
        if _validate_reset_source(hydra_dir, kind="Hydra output directory"):
            moves.append(
                RunResetMove(
                    kind="hydra_output",
                    source=hydra_dir,
                    archive_namespace=(
                        output_root / RESET_ARCHIVE_DIRNAME / run_name
                    ),
                    archive_relative=Path("hydra_output"),
                )
            )

    # A custom CW_OUTPUT can make two logical targets identical or nested.
    # Deduplicate an identical directory, but reject nesting because moving an
    # ancestor would make the remaining plan ambiguous and order-dependent.
    unique: list[RunResetMove] = []
    seen_sources: set[Path] = set()
    for move in moves:
        normalized = Path(os.path.abspath(move.source))
        if normalized in seen_sources:
            continue
        seen_sources.add(normalized)
        unique.append(move)
    for index, left in enumerate(unique):
        left_path = Path(os.path.abspath(left.source))
        for right in unique[index + 1:]:
            right_path = Path(os.path.abspath(right.source))
            if left_path in right_path.parents or right_path in left_path.parents:
                raise SystemExit(
                    "Reset targets overlap; choose a CW_OUTPUT outside the "
                    "checkpoint/SPT run directories: "
                    f"{left.source} and {right.source}"
                )

    return RunResetPlan(run_name=run_name, moves=tuple(unique))


def _validate_reset_archive_namespace(namespace: Path) -> None:
    archive_root = namespace.parent
    for path in (archive_root, namespace):
        if path.is_symlink():
            raise SystemExit(f"Refusing symlinked reset archive path: {path}")
        if path.exists() and not path.is_dir():
            raise SystemExit(f"Reset archive path is not a directory: {path}")


def _execute_run_reset(
    plan: RunResetPlan,
    *,
    identity_sha256: str,
) -> tuple[Path, ...]:
    """Archive a run's old state with recoverable same-filesystem renames."""

    if not plan.moves:
        print(
            f"[stablewm-train] reset run={plan.run_name}: "
            "no existing state; starting fresh"
        )
        return ()

    # Validate the complete plan before changing any path.
    for move in plan.moves:
        if not _validate_reset_source(move.source, kind=move.kind):
            raise SystemExit(
                "Reset state changed after planning; refusing a partial reset: "
                f"{move.source}"
            )
        _validate_reset_archive_namespace(move.archive_namespace)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    base_reset_id = f"{stamp}-p{os.getpid()}"
    reset_id = base_reset_id
    namespaces = sorted(
        {move.archive_namespace for move in plan.moves},
        key=str,
    )
    suffix = 0
    while any((namespace / reset_id).exists() for namespace in namespaces):
        suffix += 1
        reset_id = f"{base_reset_id}-{suffix}"

    bundles = tuple(namespace / reset_id for namespace in namespaces)
    destinations = tuple(
        move.archive_namespace / reset_id / move.archive_relative
        for move in plan.moves
    )
    moved: list[tuple[RunResetMove, Path]] = []
    receipts: list[Path] = []
    try:
        for bundle in bundles:
            bundle.mkdir(parents=True, exist_ok=False)
        for move, destination in zip(plan.moves, destinations, strict=True):
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(move.source, destination)
            moved.append((move, destination))

        payload = {
            "schema_version": RESET_RECEIPT_SCHEMA,
            "reset_id": reset_id,
            "run_name": plan.run_name,
            "replacement_training_identity_sha256": identity_sha256,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "moves": [
                {
                    "kind": move.kind,
                    "source": str(move.source),
                    "archive": str(destination),
                }
                for move, destination in moved
            ],
        }
        for bundle in bundles:
            receipt = bundle / "reset_receipt.json"
            with receipt.open("x", encoding="utf-8") as stream:
                stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            receipts.append(receipt)
    except (OSError, ValueError) as exc:
        cleanup_failures: list[str] = []
        for receipt in receipts:
            try:
                receipt.unlink(missing_ok=True)
            except OSError as cleanup_exc:
                cleanup_failures.append(f"{receipt}: {cleanup_exc}")
        rollback_failures: list[str] = []
        for move, destination in reversed(moved):
            try:
                os.replace(destination, move.source)
            except OSError as rollback_exc:
                rollback_failures.append(
                    f"{destination} -> {move.source}: {rollback_exc}"
                )
        details: list[str] = []
        if rollback_failures:
            details.append("rollback failed: " + "; ".join(rollback_failures))
        else:
            details.append("the already moved directories were restored")
        if cleanup_failures:
            details.append(
                "stale reset receipt cleanup failed: "
                + "; ".join(cleanup_failures)
            )
        raise SystemExit(
            f"Could not archive reset state: {exc}. {'; '.join(details)}."
        ) from exc

    for move, destination in moved:
        print(f"[stablewm-train] reset archived {move.kind}: {destination}")
    return tuple(receipts)


def _preflight_reservation_identity(
    root: Path,
    run_name: str,
) -> dict[str, Any] | None:
    """Return an identity only when the upstream trainer never started."""

    run_dir = root / "checkpoints" / run_name
    if not run_dir.is_dir() or run_dir.is_symlink():
        return None
    identity_path = run_dir / TRAINING_IDENTITY_FILENAME
    entries = list(run_dir.iterdir())
    if len(entries) != 1 or entries[0] != identity_path:
        return None
    if not identity_path.is_file() or identity_path.is_symlink():
        raise SystemExit(f"Unsafe training identity reservation: {identity_path}")
    payload = _read_training_identity(identity_path)
    if payload.get("schema_version") != TRAINING_IDENTITY_SCHEMA:
        raise SystemExit(f"Unsupported training identity reservation: {identity_path}")
    if not isinstance(payload.get("identity_sha256"), str):
        raise SystemExit(f"Incomplete training identity reservation: {identity_path}")

    identity = payload.get("identity")
    overrides = identity.get("hydra_overrides") if isinstance(identity, dict) else None
    hydra_values = (
        [
            value.removeprefix("hydra.run.dir=")
            for value in overrides
            if isinstance(value, str) and value.startswith("hydra.run.dir=")
        ]
        if isinstance(overrides, list)
        else []
    )
    if len(hydra_values) != 1:
        return None
    hydra_run_dir = Path(hydra_values[0])
    if not hydra_run_dir.is_absolute():
        return None
    if hydra_run_dir.exists():
        if hydra_run_dir.is_symlink() or not hydra_run_dir.is_dir():
            raise SystemExit(f"Unsafe Hydra run directory: {hydra_run_dir}")
        if any(hydra_run_dir.iterdir()):
            return None

    if any(
        record.get("run_name") == run_name
        for _, record in _stablepretraining_run_records(root)
    ):
        return None
    return payload


def _zero_step_failed_training_identity(
    root: Path,
    run_name: str,
) -> dict[str, Any] | None:
    """Return the saved recipe only for a proven zero-step trainer failure.

    A newly submitted cloud job cannot resume when the trainer failed before
    StablePretraining wrote ``last.ckpt``.  Restarting every incomplete run
    would be unsafe, though: an inference checkpoint may contain useful work
    even when no full trainer state exists.  This predicate therefore accepts
    only the narrow state emitted when setup or Lightning's sanity check
    failed before the first optimizer step:

    * the public checkpoint directory contains only the resolved config and
      immutable ContextWorld identity;
    * every StablePretraining UUID bound to this run has an empty checkpoint
      directory; and
    * at least one rank-zero summary explicitly reports step 0, epoch 0 and no
      metrics.  Worker-rank UUIDs may contain only their binding metadata.

    The old records and logs remain intact.  ``resume=auto`` starts a new SPT
    UUID using the same public run name instead of deleting failed evidence.
    """

    run_dir = root / "checkpoints" / run_name
    if not run_dir.is_dir() or run_dir.is_symlink():
        return None
    identity_path = run_dir / TRAINING_IDENTITY_FILENAME
    allowed_checkpoint_entries = {
        TRAINING_IDENTITY_FILENAME,
        "config.yaml",
    }
    entries = list(run_dir.iterdir())
    if not entries or any(
        entry.name not in allowed_checkpoint_entries for entry in entries
    ):
        return None
    if not identity_path.is_file() or identity_path.is_symlink():
        return None
    for entry in entries:
        if entry.is_symlink() or not entry.is_file():
            return None

    payload = _read_training_identity(identity_path)
    if payload.get("schema_version") != TRAINING_IDENTITY_SCHEMA:
        return None
    if not isinstance(payload.get("identity_sha256"), str):
        return None

    records = [
        (run_path, record)
        for run_path, record in _stablepretraining_run_records(root)
        if record.get("run_name") == run_name
    ]
    if not records:
        return None

    summary_count = 0
    rank_worker_files = {
        SPT_RUN_MARKER_FILENAME,
        "run_meta.json",
    }
    rank_zero_files = rank_worker_files | {
        "hparams.yaml",
        "sidecar.json",
        "summary.json",
    }
    for spt_run, record in records:
        if not isinstance(record.get("training_identity_sha256"), str):
            return None
        checkpoint_dir = spt_run / "checkpoints"
        if checkpoint_dir.exists():
            if checkpoint_dir.is_symlink() or not checkpoint_dir.is_dir():
                return None
            if any(checkpoint_dir.iterdir()):
                return None

        summary_path = spt_run / "summary.json"
        allowed_files = (
            rank_zero_files if summary_path.exists() else rank_worker_files
        )
        for entry in spt_run.iterdir():
            if entry.name == "checkpoints":
                continue
            if (
                entry.name not in allowed_files
                or entry.is_symlink()
                or not entry.is_file()
            ):
                return None

        if not summary_path.exists():
            continue
        if summary_path.is_symlink() or not summary_path.is_file():
            return None
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        step = summary.get("step") if isinstance(summary, dict) else None
        epoch = summary.get("epoch") if isinstance(summary, dict) else None
        if (
            isinstance(step, bool)
            or not isinstance(step, (int, float))
            or step != 0
            or isinstance(epoch, bool)
            or not isinstance(epoch, (int, float))
            or epoch != 0
            or summary.get("metrics") != {}
        ):
            return None
        summary_count += 1

    return payload if summary_count >= 1 else None


def _portable_resume_candidates(
    root: Path,
    run_name: str,
    identity_sha256: str,
) -> list[Path]:
    """Return full-state SPT checkpoints bound to this exact run recipe."""

    candidates: list[Path] = []
    for run_dir, payload in _stablepretraining_run_records(root):
        if (
            payload.get("run_name") != run_name
            or payload.get("training_identity_sha256") != identity_sha256
        ):
            continue
        checkpoint = run_dir / "checkpoints/last.ckpt"
        if (
            checkpoint.is_file()
            and not checkpoint.is_symlink()
            and checkpoint.stat().st_size > 0
        ):
            candidates.append(checkpoint.resolve())
    return sorted(
        candidates,
        key=lambda path: (path.stat().st_mtime_ns, path.stat().st_size, str(path)),
        reverse=True,
    )


def validate_resume(
    root: Path,
    run_name: str,
    policy: str,
    *,
    family: str,
    identity_sha256: str,
) -> Path | None:
    """Resolve either native requeue or portable full-state resume."""

    run_dir = root / "checkpoints" / run_name
    native_requeue = _stablepretraining_native_requeue()
    candidates = _portable_resume_candidates(root, run_name, identity_sha256)
    run_nonempty = run_dir.exists() and any(run_dir.iterdir())
    preflight_reservation = _preflight_reservation_identity(root, run_name)
    zero_step_failure = _zero_step_failed_training_identity(root, run_name)

    if policy == "never":
        if run_nonempty or candidates:
            raise SystemExit(
                f"Fresh training refuses existing state for run {run_name!r}. "
                "Choose another run name, use --resume reset to archive the "
                "same-named state, or set --resume auto/required."
            )
        return None

    if native_requeue:
        # StablePretraining resolves the same-job .slurm_index and forces
        # weights_only=False itself. Supplying a second path would be ignored.
        return None

    if candidates:
        return candidates[0]

    if policy == "required":
        raise SystemExit(
            "Resume was required, but no full-state StablePretraining "
            f"last.ckpt matching {family} run {run_name!r} and its immutable "
            "training identity was found."
        )

    if run_nonempty:
        if preflight_reservation is not None:
            # The prior launcher exited after its immutable O_EXCL identity
            # reservation but before StablePretraining created any state.
            # With no Hydra/SPT output, auto may atomically bind the current
            # recipe even when a dependency repair changed its identity.
            return None
        if zero_step_failure is not None:
            print(
                "[stablewm-train] resume=auto found a proven zero-step "
                f"failed attempt for {run_name}; starting a new "
                "StablePretraining run without deleting the failed logs"
            )
            return None
        raise SystemExit(
            "Resume=auto found an incomplete run but no matching full-state "
            f"StablePretraining checkpoint for {family} run {run_name!r}; "
            "refusing to restart it from epoch zero."
        )
    return None


def _login_swanlab_without_exposing_key(environment: dict[str, str]) -> None:
    key = environment.get("SWANLAB_API_KEY")
    if not key:
        print("[stablewm-train] SwanLab key not injected; the SDK may use "
              "an existing login or offline mode.")
        return
    # The key stays in the inherited environment.  It is never copied into
    # argv, a rendered Hydra command, config.yaml, or a launcher log line.
    code = ("import os, swanlab; "
            "swanlab.login(api_key=os.environ['SWANLAB_API_KEY'])")
    completed = subprocess.run(
        [sys.executable, "-c", code],
        env=environment,
        check=False,
    )
    if completed.returncode != 0:
        raise SystemExit(f"SwanLab login failed with status {completed.returncode}")


def _effective_eval_epoch(
    args: argparse.Namespace,
    stablewm_repo: Path,
    profile: dict[str, Any],
) -> int:
    if args.eval_epoch is not None:
        return args.eval_epoch
    if args.max_epochs is not None:
        return args.max_epochs
    _, upstream = _load_upstream_config(stablewm_repo, profile)
    value = _nested(upstream, "trainer", "max_epochs")
    if not isinstance(value, int) or value <= 0:
        raise SystemExit(
            "Post-train eval needs --eval-epoch or --max-epochs because the "
            "upstream epoch count could not be resolved.")
    return value


def _benchmark_root_for_post_eval(args: argparse.Namespace) -> Path:
    """Resolve the clean bundle used by the public Development ICL suite.

    The original H5 files remain the source for CEM.  ICL must instead name
    the exported ContextWorld-v1 bundle explicitly, so a cloud job cannot
    quietly fall back to the private ``context_world`` research tree.
    """

    root_value = args.benchmark_root
    if not root_value and args.dataset_root:
        root_value = str(
            _absolute_path(args.dataset_root, label="--dataset-root")
            / "ContextWorld-v1"
        )
    if not root_value:
        raise SystemExit(
            "Post-training ContextWorld ICL evaluation needs "
            "--benchmark-root/CONTEXTWORLD_BENCHMARK_ROOT, or a "
            "--dataset-root containing ContextWorld-v1."
        )
    return _absolute_path(root_value, label="--benchmark-root")


def _post_eval_command(
    args: argparse.Namespace,
    *,
    target: Target,
    run_name: str,
    checkpoint_root: Path,
    stablewm_repo: Path,
    profile: dict[str, Any],
    frameskip: int,
    training_seed: int,
    original_dataset: Path | None,
) -> list[str]:
    epoch = _effective_eval_epoch(args, stablewm_repo, profile)
    benchmark_root = _benchmark_root_for_post_eval(args)
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts/run_stablewm_eval.py"),
        "--suite",
        "--family",
        args.family,
        "--run-name",
        run_name,
        "--epoch",
        str(epoch),
        "--checkpoint-root",
        str(checkpoint_root),
        "--stablewm-repo",
        str(stablewm_repo),
        "--benchmark-root",
        str(benchmark_root),
        "--training-seed",
        str(training_seed),
        "--num-eval",
        str(args.eval_num),
        "--eval-seeds",
        args.eval_seeds,
        "--history-size",
        str(target.history_size),
        "--action-block",
        str(frameskip),
        "--mujoco-gl",
        args.eval_mujoco_gl,
        "--corruption-type",
        args.eval_corruption_type,
        "--corruption-std",
        str(args.eval_corruption_std),
        "--corruption-factor",
        str(args.eval_corruption_factor),
        "--corruption-kernel-size",
        str(args.eval_corruption_kernel_size),
        "--corruption-apply-to",
        args.eval_corruption_apply_to,
    ]
    if target.original_env:
        command.extend(("--original-env", target.original_env))
    else:
        # A component suite always attempts its Development ICL cell. When
        # the matching original dataset is available it also runs CEM
        # retention; the evaluator records CEM as skipped only when no such
        # dataset was supplied. ``--icl-only`` is reserved for explicit
        # repair jobs.
        command.extend(("--component", target.label))
    if original_dataset is not None:
        command.extend(("--dataset", str(original_dataset)))
    if args.eval_result_subdir:
        command.extend(("--result-subdir", args.eval_result_subdir))
    if args.eval_device:
        command.extend(("--eval-device", args.eval_device))
    if args.eval_batch_size is not None:
        command.extend(("--eval-batch-size", str(args.eval_batch_size)))
    if args.eval_keep_videos:
        command.append("--keep-videos")
    if args.print_command:
        command.append("--print-command")
    return command


def _safe_eval_result_subdir(value: str) -> Path:
    """Mirror the evaluator's safe result namespace before reading it."""

    if not value:
        return Path()
    path = Path(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise SystemExit(f"--eval-result-subdir must be a safe relative path: {value}")
    return path


def _development_component_icl_failure(
    *,
    args: argparse.Namespace,
    target: Target,
    checkpoint_root: Path,
    run_name: str,
    epoch: int,
) -> str | None:
    """Return an error when a component suite did not finish Development ICL.

    ``run_stablewm_eval.py`` intentionally records incompatible ICL as a
    non-fatal suite row so original state-conditioned checkpoints can retain
    their CEM evidence.  A benchmark-component post-eval is different: its
    RGB/action training contract must yield a completed Development result for
    that exact component, not a successful process containing a skipped ICL
    row. Completion here means only that the Development step ran; it is not
    a Public-Test result or a release decision.
    """

    if target.original_env is not None:
        return None
    checkpoint = _evaluation_checkpoint(checkpoint_root, run_name, epoch)
    manifest = (
        checkpoint.parent
        / "eval_results"
        / _safe_eval_result_subdir(args.eval_result_subdir)
        / "manifest.json"
    )
    if not manifest.is_file() or manifest.is_symlink():
        return f"Development ICL manifest is missing or unsafe: {manifest}"
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return f"could not read Development ICL manifest {manifest}: {exc}"
    if not isinstance(payload, dict):
        return f"Development ICL manifest is not an object: {manifest}"
    if payload.get("status") != "completed":
        return (
            "Development ICL suite did not complete for "
            f"{target.label}: manifest status={payload.get('status')!r}"
        )
    steps = payload.get("steps")
    if not isinstance(steps, list):
        return f"Development ICL manifest has no step list: {manifest}"
    step_id = f"benchmark_icl/{target.label}"
    matching = [
        step
        for step in steps
        if isinstance(step, dict) and step.get("id") == step_id
    ]
    if len(matching) != 1:
        return f"Development ICL manifest is missing exactly one {step_id} step"
    step = matching[0]
    if step.get("status") != "completed":
        return (
            f"Development ICL component={target.label} did not complete: "
            f"status={step.get('status')!r}, reason={step.get('reason')!r}"
        )
    output = step.get("output")
    if not isinstance(output, str):
        return f"Development ICL component={target.label} has no result path"
    output_path = Path(output)
    if not output_path.is_file() or output_path.is_symlink():
        return (
            "Development ICL component="
            f"{target.label} result is missing or unsafe: {output}"
        )
    return None


def _print_seed_summary(outcomes: list[SeedOutcome]) -> None:
    """Render a compact, machine-readable-enough result for every seed."""

    print("[stablewm-train] seed-summary")
    for outcome in outcomes:
        train = outcome.training_status
        if outcome.training_returncode is not None:
            train += f"(returncode={outcome.training_returncode})"
        evaluation = outcome.evaluation_status
        if outcome.evaluation_returncode is not None:
            evaluation += f"(returncode={outcome.evaluation_returncode})"
        print(
            f"[stablewm-train] seed={outcome.run_name} "
            f"training={train} evaluation={evaluation}"
        )


def _original_dataset_for_post_eval(
    args: argparse.Namespace,
    contract: dict[str, Any],
    target: Target,
) -> Path | None:
    """Resolve the original task data used by the CEM-retention step."""

    if target.original_env:
        return target.dataset
    if args.eval_original_dataset:
        dataset = _absolute_path(
            args.eval_original_dataset,
            label="--eval-original-dataset",
        )
        _validate_dataset_payload(dataset)
        return dataset
    if not args.dataset_root:
        raise SystemExit(
            "Complete component post-eval needs the matching original-"
            "environment dataset for CEM retention. Set "
            "CONTEXTWORLD_DATASET_ROOT/--dataset-root or "
            "CW_EVAL_ORIGINAL_DATASET/--eval-original-dataset. For an "
            "explicit ICL-only repair, invoke scripts/run_stablewm_eval.py "
            "--suite --icl-only directly."
        )
    root = _absolute_path(args.dataset_root, label="--dataset-root")
    dataset = Path(
        os.path.abspath(
            root / contract["original_environments"][target.environment]["dataset"]
        )
    )
    _validate_dataset_payload(dataset)
    return dataset


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    contract = load_profile_contract()
    parser = argparse.ArgumentParser(description=(
        "Launch LeWM, PLDM or PreJEPA through a validated family profile."))
    parser.add_argument(
        "--profile-config",
        type=Path,
        default=Path(_env("CW_STABLEWM_PROFILE_CONFIG", str(DEFAULT_PROFILE_CONFIG))),
    )
    parser.add_argument(
        "--family",
        choices=sorted(contract["families"]),
        default=_env("CW_FAMILY", "lewm"),
    )
    parser.add_argument("--original-env", default=_env("CW_ENV"))
    parser.add_argument("--component", default=_env("CW_COMPONENT"))
    parser.add_argument(
        "--training-track",
        choices=("joint_scratch_v1", "historical_release"),
        default=_env("CW_TRAINING_TRACK", "joint_scratch_v1"),
        help=(
            "Component-training route (env: CW_TRAINING_TRACK). The default "
            "uses the public ContextWorld-v1 mixture and no task checkpoint; "
            "historical_release explicitly selects the old frozen LeWM/PLDM "
            "reproduction recipe."
        ),
    )
    parser.add_argument(
        "--method",
        choices=sorted(contract.get("methods") or {"native": {}}),
        default=_env("CW_METHOD", "native"),
        help=(
            "Training method (env: CW_METHOD). 'native' renders the family's "
            "own objective unchanged. 'coja_v1' adds the checkout's one-step "
            "conditional-joint term over complete public training relations."
        ),
    )
    parser.add_argument("--dataset", default=_env("CW_DATASET"))
    parser.add_argument(
        "--mode",
        default=_env("CW_MODE", "preflight"),
        help="Mode passed to a frozen release launcher (env: CW_MODE).",
    )
    parser.add_argument(
        "--stage",
        choices=("paired", "curriculum"),
        default=_env("CW_STAGE", "paired"),
        help="Action-delay release stage (env: CW_STAGE).",
    )
    parser.add_argument(
        "--variant",
        default=_env("CW_VARIANT"),
        help="Frozen release recipe variant override (env: CW_VARIANT).",
    )
    parser.add_argument("--dataset-root", default=_env("CONTEXTWORLD_DATASET_ROOT"))
    parser.add_argument(
        "--benchmark-root",
        default=_env("CONTEXTWORLD_BENCHMARK_ROOT"),
        help=(
            "ContextWorld-v1 clean-export root. For benchmark component runs, "
            "defaults to <dataset-root>/ContextWorld-v1."
        ),
    )
    parser.add_argument(
        "--component-payload",
        default=_env("CW_COMPONENT_PAYLOAD"),
        help=(
            "Registered training payload override. ActionDelay accepts "
            "coarse or full; component defaults are normally sufficient."
        ),
    )
    parser.add_argument(
        "--mix-original-weight",
        type=float,
        default=_env_float("CW_MIX_ORIGINAL_WEIGHT"),
    )
    parser.add_argument(
        "--mix-synthetic-weight",
        type=float,
        default=_env_float("CW_MIX_SYNTHETIC_WEIGHT"),
    )
    parser.add_argument(
        "--component-epoch-size",
        type=int,
        default=_env_int("CW_COMPONENT_EPOCH_SIZE"),
        help=(
            "Optional number of virtual samples per epoch. The runtime "
            "reader otherwise derives a full-coverage balanced epoch."
        ),
    )
    parser.add_argument("--data-group", default=_env("CW_DATA_GROUP"))
    parser.add_argument(
        "--stablewm-repo",
        default=_env("CONTEXTWORLD_STABLE_WORLDMODEL_REPO", _env("STABLEWM_REPO")),
    )
    parser.add_argument(
        "--checkpoint-root",
        default=_env("CW_CHECKPOINT_ROOT", _env("STABLEWM_HOME")),
    )
    parser.add_argument(
        "--dataset-cache-root",
        default=_env("CW_DATASET_CACHE_ROOT", _env("LOCAL_DATASET_DIR")),
    )
    parser.add_argument("--output", default=_env("CW_OUTPUT"))
    parser.add_argument("--run-name", default=_env("CW_RUN_NAME"))
    parser.add_argument(
        "--seeds",
        type=parse_training_seeds,
        default=_env(
            "CW_SEEDS",
            ",".join(str(seed) for seed in DEFAULT_TRAINING_SEEDS),
        ),
        help=(
            "Comma-separated training seeds (env: CW_SEEDS). Defaults to "
            "one run: 3072"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=_env_int("CW_BATCH_SIZE"))
    parser.add_argument("--num-workers", type=int, default=_env_int("CW_NUM_WORKERS"))
    parser.add_argument("--train-split",
                        type=float,
                        default=_env_float("CW_TRAIN_SPLIT"))
    parser.add_argument("--frameskip", type=int, default=_env_int("CW_FRAMESKIP"))
    parser.add_argument("--history-size", type=int, default=_env_int("CW_HISTORY_SIZE"))
    parser.add_argument("--num-preds", type=int, default=_env_int("CW_NUM_PREDS"))
    parser.add_argument("--max-epochs", type=int, default=_env_int("CW_MAX_EPOCHS"))
    parser.add_argument("--devices", default=_env("CW_DEVICES"))
    parser.add_argument("--accelerator", default=_env("CW_ACCELERATOR"))
    parser.add_argument("--strategy", default=_env("CW_STRATEGY"))
    parser.add_argument("--precision", default=_env("CW_PRECISION"))
    parser.add_argument("--accumulate", type=int, default=_env_int("CW_ACCUMULATE"))
    parser.add_argument("--gradient-clip-val",
                        type=float,
                        default=_env_float("CW_GRADIENT_CLIP_VAL"))
    parser.add_argument("--persistent-workers",
                        action=argparse.BooleanOptionalAction,
                        default=_env_bool("CW_PERSISTENT_WORKERS"))
    parser.add_argument("--prefetch-factor",
                        type=int,
                        default=_env_int("CW_PREFETCH_FACTOR"))
    parser.add_argument("--pin-memory",
                        action=argparse.BooleanOptionalAction,
                        default=_env_bool("CW_PIN_MEMORY"))
    parser.add_argument("--fast-dev-run", default=_env("CW_FAST_DEV_RUN"))
    parser.add_argument("--limit-train-batches", default=_env("CW_LIMIT_TRAIN_BATCHES"))
    parser.add_argument("--limit-val-batches", default=_env("CW_LIMIT_VAL_BATCHES"))
    parser.add_argument("--trainer-default-root-dir",
                        default=_env("CW_TRAINER_DEFAULT_ROOT_DIR"))
    parser.add_argument("--learning-rate",
                        type=float,
                        default=_env_float("CW_LEARNING_RATE"))
    parser.add_argument("--weight-decay",
                        type=float,
                        default=_env_float("CW_WEIGHT_DECAY"))
    parser.add_argument("--embed-dim", type=int, default=_env_int("CW_EMBED_DIM"))
    parser.add_argument("--hydra-job-chdir",
                        action=argparse.BooleanOptionalAction,
                        default=_env_bool("CW_HYDRA_JOB_CHDIR"))
    parser.add_argument("--dataset-sampling", default=_env("CW_DATASET_SAMPLING"))
    parser.add_argument("--balance-val",
                        action=argparse.BooleanOptionalAction,
                        default=_env_bool("CW_BALANCE_VAL"))
    parser.add_argument("--dataset-item", action="append", default=[])
    parser.add_argument(
        "--resume",
        choices=("never", "auto", "required", "reset"),
        default=_env("CW_RESUME", contract["defaults"]["resume"]),
        help=(
            "Recovery policy (env: CW_RESUME). 'reset' keeps the same run "
            "name, archives its exact local state, and starts from epoch zero."
        ),
    )

    parser.add_argument(
        "--logger",
        choices=("none", "swanlab", "wandb"),
        default=_env("CW_LOGGER", contract["defaults"]["logger"]),
    )
    parser.add_argument("--tracker-name", default=_env("CW_TRACKER_NAME"))
    parser.add_argument("--tracker-id", default=_env("CW_TRACKER_ID"))
    parser.add_argument("--swanlab-project", default=_env("CW_SWANLAB_PROJECT"))
    parser.add_argument("--swanlab-workspace", default=_env("CW_SWANLAB_WORKSPACE"))
    parser.add_argument("--swanlab-logdir", default=_env("CW_SWANLAB_LOGDIR"))
    parser.add_argument("--swanlab-mode", default=_env("CW_SWANLAB_MODE"))
    parser.add_argument("--swanlab-collect-hardware",
                        action=argparse.BooleanOptionalAction,
                        default=_env_bool("CW_SWANLAB_COLLECT_HARDWARE"))
    parser.add_argument("--swanlab-hardware-monitor",
                        action=argparse.BooleanOptionalAction,
                        default=_env_bool("CW_SWANLAB_HARDWARE_MONITOR"))
    parser.add_argument("--swanlab-log-hyperparams",
                        action=argparse.BooleanOptionalAction,
                        default=_env_bool("CW_SWANLAB_LOG_HYPERPARAMS"))
    parser.add_argument("--wandb-project", default=_env("CW_WANDB_PROJECT"))
    parser.add_argument("--wandb-entity", default=_env("CW_WANDB_ENTITY"))

    parser.add_argument("--lewm-sigreg-weight",
                        type=float,
                        default=_env_float("CW_LEWM_SIGREG_WEIGHT"))
    parser.add_argument("--viswm-weight",
                        type=float,
                        default=_env_float("CW_VISWM_WEIGHT"))
    parser.add_argument("--viswm-num-projections",
                        type=int,
                        default=_env_int("CW_VISWM_NUM_PROJECTIONS"))
    parser.add_argument("--viswm-lambda-scale",
                        type=float,
                        default=_env_float("CW_VISWM_LAMBDA_SCALE"))
    parser.add_argument("--viswm-lambda-shape",
                        type=float,
                        default=_env_float("CW_VISWM_LAMBDA_SHAPE"))
    parser.add_argument("--viswm-lambda-center",
                        type=float,
                        default=_env_float("CW_VISWM_LAMBDA_CENTER"))

    parser.add_argument(
        "--post-eval",
        action="store_true",
        default=bool(_env_bool("CW_POST_TRAIN_EVAL") or False),
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        default=bool(_env_bool("CW_EVAL_ONLY") or False),
        help=(
            "Skip training and run the common evaluation suite against an "
            "existing checkpoint (env: CW_EVAL_ONLY)."
        ),
    )
    parser.add_argument("--eval-epoch", type=int, default=_env_int("CW_EVAL_EPOCH"))
    parser.add_argument(
        "--eval-result-subdir",
        default=_env("CW_EVAL_RESULT_SUBDIR", ""),
        help=(
            "Optional immutable retry namespace below eval_results/ "
            "(env: CW_EVAL_RESULT_SUBDIR)."
        ),
    )
    parser.add_argument("--eval-num",
                        type=int,
                        default=int(
                            _env("CW_EVAL_NUM",
                                 str(contract["evaluation"]["default_num_eval"]))))
    parser.add_argument(
        "--eval-seeds",
        default=_env("CW_EVAL_SEEDS",
                     ",".join(str(x) for x in contract["evaluation"]["default_seeds"])))
    parser.add_argument("--eval-mujoco-gl", default=_env("CW_EVAL_MUJOCO_GL", "osmesa"))
    parser.add_argument(
        "--eval-original-dataset",
        default=_env("CW_EVAL_ORIGINAL_DATASET"),
        help=(
            "Original-environment H5 used by component CEM retention. "
            "CONTEXTWORLD_DATASET_ROOT is used when this is omitted."
        ),
    )
    parser.add_argument("--eval-device", default=_env("CW_EVAL_DEVICE", "cuda:0"))
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=int(_env("CW_EVAL_BATCH_SIZE", "64")),
    )
    parser.add_argument("--eval-corruption-type",
                        default=_env("CW_EVAL_CORRUPTION_TYPE", "gaussian_noise"))
    parser.add_argument("--eval-corruption-std",
                        type=float,
                        default=float(_env("CW_EVAL_CORRUPTION_STD", "0.0")))
    parser.add_argument("--eval-corruption-factor",
                        type=float,
                        default=float(_env("CW_EVAL_CORRUPTION_FACTOR", "1.0")))
    parser.add_argument("--eval-corruption-kernel-size",
                        type=int,
                        default=int(_env("CW_EVAL_CORRUPTION_KERNEL_SIZE", "1")))
    parser.add_argument("--eval-corruption-apply-to",
                        default=_env("CW_EVAL_CORRUPTION_APPLY_TO", "pixels"))
    parser.add_argument("--eval-keep-videos",
                        action="store_true",
                        default=bool(_env_bool("CW_EVAL_KEEP_VIDEOS") or False))
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument(
        "--print-command",
        action="store_true",
        default=bool(_env_bool("CW_PRINT_ONLY") or False),
    )
    args = parser.parse_args(argv)
    reject_legacy_seed_environment(parser, os.environ)
    return args


def _uses_release_recipe(args: argparse.Namespace) -> bool:
    """Return whether this is a frozen LeWM/PLDM component reproduction."""

    return args.training_track == "historical_release"


def _validate_training_track(args: argparse.Namespace) -> None:
    """Keep current joint training and historical release evidence disjoint."""

    if args.training_track != "historical_release":
        return
    if args.resume == "reset":
        raise SystemExit(
            "CW_RESUME=reset is available only on current family-profile "
            "training, not the frozen historical_release launcher."
        )
    if args.original_env or not args.component:
        raise SystemExit(
            "CW_TRAINING_TRACK=historical_release is valid only for a "
            "benchmark component. Original-environment training always uses "
            "the current family profile."
        )
    if args.family not in {"lewm", "pldm"}:
        raise SystemExit(
            "The historical release track exists only for LeWM and PLDM; "
            "PreJEPA has no frozen historical component recipe."
        )
    if args.dataset:
        raise SystemExit(
            "The historical release track owns its frozen dataset mapping; "
            "CW_DATASET/--dataset cannot be supplied."
        )
    if any(
        value is not None
        for value in (
            args.component_payload,
            args.mix_original_weight,
            args.mix_synthetic_weight,
            args.component_epoch_size,
        )
    ):
        raise SystemExit(
            "ContextWorld-v1 payload and mixture options are unavailable on "
            "the historical release track."
        )


def _run_release_reproduction(args: argparse.Namespace) -> int:
    """Resolve and run frozen task launchers without another router process."""

    # cloud_train remains the compatibility CLI and the registry for the
    # historical task-to-recipe mappings. Importing its planner here preserves
    # those mappings while keeping it out of the process chain.
    import cloud_train as release_recipes

    if args.eval_only:
        raise SystemExit(
            "--eval-only is unavailable for a frozen historical component "
            "recipe; use the evaluator named by that release protocol."
        )
    if args.post_eval:
        raise SystemExit(
            "--post-eval is unavailable for a frozen historical component "
            "reproduction. Use the evaluator named by that component's "
            "release protocol, or supply --dataset to select the current "
            "family-profile training and common evaluation suite."
        )

    request = argparse.Namespace(
        task=args.component,
        env=None,
        family=args.family,
        seeds=args.seeds,
        mode=args.mode,
        stage=args.stage,
        variant=args.variant,
        dataset=None,
        run_name=args.run_name,
        output=args.output,
        batch_size=args.batch_size,
        print_command=args.print_command,
        extra=[],
    )
    runs = release_recipes._seed_runs(request)
    print(
        "[stablewm-train] mode=release-reproduction "
        f"component={args.component} family={args.family} "
        f"seeds={','.join(str(seed) for seed in args.seeds)}"
    )
    print(
        "[stablewm-train] recipe=frozen; generic family-profile overrides "
        "are not applied"
    )
    for run in runs:
        plan = release_recipes.build_plan(run)
        print(f"[stablewm-train] seed={run.seed}")
        if plan.note:
            print(f"[stablewm-train] note: {plan.note}")
        for key, value in sorted(plan.env.items()):
            print(f"[stablewm-train] env {key}={value}")
        print(f"[stablewm-train] train: {shlex.join(plan.command)}")
        if args.print_command:
            continue

        environment = {**os.environ, **plan.env}
        sys.stdout.flush()
        completed = subprocess.run(
            plan.command,
            cwd=str(REPO_ROOT),
            env=environment,
            check=False,
        )
        if completed.returncode != 0:
            return completed.returncode
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    contract = load_profile_contract(args.profile_config.resolve())
    if args.family not in contract["families"]:
        raise SystemExit(f"Unknown family in profile contract: {args.family}")
    _validate_training_track(args)
    _validate_method(args, contract)
    if args.resume == "reset" and args.eval_only:
        raise SystemExit(
            "CW_RESUME=reset cannot be combined with CW_EVAL_ONLY=1: reset "
            "archives the checkpoint and starts a new same-named training run."
        )
    if args.resume == "reset" and _stablepretraining_native_requeue():
        raise SystemExit(
            "CW_RESUME=reset cannot run inside a StablePretraining same-job "
            "scheduler requeue. Submit a fresh scheduler job so upstream "
            "native recovery cannot restore the archived state."
        )
    if _uses_release_recipe(args):
        return _run_release_reproduction(args)
    validate_stablepretraining_version()
    target = resolve_target(args, contract)
    stablewm_repo = resolve_stablewm_repo(args)
    validate_training_dataset_schema(
        target=target,
        family=args.family,
        stablewm_repo=stablewm_repo,
    )
    checkpoint_root = resolve_checkpoint_root(args)
    output_root = (
        _absolute_path(args.output, label="--output")
        if args.output
        else None
    )
    profile = contract["families"][args.family]
    trainer_script = stablewm_repo / profile["entrypoint"]
    seeds = args.seeds
    # Both target kinds need the original-environment payload at post-eval:
    # original runs use it for their primary CEM cell, while component runs
    # use the matching environment payload for CEM retention.  Resolving this
    # only for ``target.original_env`` silently reduced component post-eval to
    # ICL-only even when CONTEXTWORLD_DATASET_ROOT was available.
    original_eval_dataset: Path | None = (
        _original_dataset_for_post_eval(args, contract, target)
        if args.post_eval or args.eval_only
        else None
    )

    environment = dict(os.environ)
    for internal_name in (
        "CONTEXTWORLD_SPT_BRIDGE",
        "CONTEXTWORLD_SPT_RUN_NAME",
        "CONTEXTWORLD_SPT_IDENTITY_SHA256",
        "CONTEXTWORLD_SPT_RESUME_CHECKPOINT",
        "CONTEXTWORLD_STABLEWM_BUNDLE",
        "CONTEXTWORLD_DATALOADER_START_METHOD",
    ):
        environment.pop(internal_name, None)
    environment["STABLEWM_HOME"] = str(checkpoint_root)
    # StablePretraining owns full-state checkpoint/requeue. Its default cache
    # is container-local (~/.cache), so bind it to the same persistent root as
    # StableWM unless the caller explicitly selected another durable path.
    configured_spt_cache = environment.get("SPT_CACHE_DIR")
    if configured_spt_cache:
        configured_spt_path = _absolute_path(
            configured_spt_cache,
            label="SPT_CACHE_DIR",
        )
        if configured_spt_path != checkpoint_root:
            raise SystemExit(
                "SPT_CACHE_DIR and --checkpoint-root/CW_CHECKPOINT_ROOT "
                "must identify the same persistent storage root."
            )
    environment["SPT_CACHE_DIR"] = str(checkpoint_root)
    if args.dataset_cache_root:
        cache = _absolute_path(args.dataset_cache_root, label="--dataset-cache-root")
        environment["LOCAL_DATASET_DIR"] = str(cache)
    environment["PYTHONPATH"] = os.pathsep.join([
        str(REPO_ROOT),
        str(stablewm_repo),
        environment.get("PYTHONPATH", ""),
    ]).strip(os.pathsep)

    eval_epoch = (
        _effective_eval_epoch(args, stablewm_repo, profile)
        if args.post_eval or args.eval_only
        else None
    )
    commands: list[
        tuple[
            str,
            list[str] | None,
            list[str] | None,
            str | None,
            dict[str, Any] | None,
            Path | None,
        ]
    ] = []
    reset_plans: dict[str, RunResetPlan] = {}
    for seed in seeds:
        run_name = _run_name(args, target, seed, seeds)
        overrides = None
        training_identity = None
        if not args.eval_only:
            overrides = build_overrides(
                args,
                contract,
                target,
                run_name=run_name,
                seed=seed,
                stablewm_repo=stablewm_repo,
            )
            training_identity = _training_identity_document(
                args=args,
                target=target,
                profile=profile,
                stablewm_repo=stablewm_repo,
                trainer_script=trainer_script,
                run_name=run_name,
                seed=seed,
                overrides=overrides,
            )
        completed_checkpoint = (
            _evaluation_checkpoint(checkpoint_root, run_name, eval_epoch)
            if eval_epoch is not None
            else None
        )
        auto_eval_recovery = bool(
            args.post_eval
            and not args.eval_only
            and args.resume in {"auto", "required"}
            and completed_checkpoint is not None
            and completed_checkpoint.is_file()
            and completed_checkpoint.stat().st_size > 0
        )
        train_command = None
        resume_checkpoint = None
        skip_reason = None
        if args.eval_only:
            skip_reason = "CW_EVAL_ONLY requested"
        elif auto_eval_recovery:
            _validate_completed_training_identity(
                completed_checkpoint,
                expected=training_identity,
            )
            skip_reason = (
                "verified target epoch checkpoint already exists: "
                f"{completed_checkpoint}"
            )
        else:
            assert overrides is not None
            assert training_identity is not None
            if args.resume == "reset":
                reset_plans[run_name] = _plan_run_reset(
                    checkpoint_root,
                    run_name,
                    output_root=output_root,
                )
            else:
                resume_checkpoint = validate_resume(
                    checkpoint_root,
                    run_name,
                    args.resume,
                    family=args.family,
                    identity_sha256=training_identity["identity_sha256"],
                )
            train_command = [
                sys.executable,
                str(trainer_script),
            ]
            train_command.extend([
                f"--config-name={profile['config_name']}",
                *overrides,
            ])
        eval_command = (_post_eval_command(
            args,
            target=target,
            run_name=run_name,
            checkpoint_root=checkpoint_root,
            stablewm_repo=stablewm_repo,
            profile=profile,
            frameskip=(args.frameskip or contract["defaults"]["frameskip"]),
            training_seed=seed,
            original_dataset=original_eval_dataset,
        ) if args.post_eval or args.eval_only else None)
        commands.append(
            (
                run_name,
                train_command,
                eval_command,
                skip_reason,
                training_identity,
                resume_checkpoint,
            )
        )

    _validate_scheduler_seed_isolation(
        [
            run_name
            for run_name, train_command, _, _, _, _ in commands
            if train_command is not None
        ]
    )

    print(f"[stablewm-train] target={target.label} family={args.family} "
          f"dataset={target.dataset}")
    if _is_contextworld_dataset_uri(target.dataset):
        runtime = describe_contextworld_dataset(str(target.dataset))
        print(
            "[stablewm-train] bundle-view="
            f"payload:{runtime['payload_id']} members:{runtime['member_count']} "
            f"original_weight:{runtime['weights']['original']} "
            f"synthetic_weight:{runtime['weights']['synthetic']} "
            f"epoch_size:{runtime['epoch_size'] or '<balanced-full-coverage>'}"
        )
        joint = runtime.get("conditional_joint")
        if joint:
            print(
                "[stablewm-train] conditional-joint="
                f"method:{joint['method']} relation:{joint['relation_kind']} "
                f"group_width:{joint['group_width']}"
            )
    print(f"[stablewm-train] stablewm={stablewm_repo}")
    print(f"[stablewm-train] checkpoint_root={checkpoint_root}")
    print(f"[stablewm-train] spt_cache={environment['SPT_CACHE_DIR']}")
    print("[stablewm-train] dataset_cache="
          f"{environment.get('LOCAL_DATASET_DIR', '<upstream default>')}")
    print(f"[stablewm-train] logger={args.logger} resume={args.resume}")
    print(
        f"[stablewm-train] mode={'eval-only' if args.eval_only else 'train'} "
        f"training_track={args.training_track} method={args.method}"
    )
    for (
        run_name,
        train_command,
        eval_command,
        skip_reason,
        _,
        resume_checkpoint,
    ) in commands:
        print(f"[stablewm-train] run={run_name}")
        if skip_reason:
            print(f"[stablewm-train] training skipped: {skip_reason}")
        elif train_command:
            if args.resume == "reset":
                reset_plan = reset_plans[run_name]
                reset_count = len(reset_plan.moves)
                reset_noun = "directory" if reset_count == 1 else "directories"
                print(
                    "[stablewm-train] reset planned: "
                    f"{reset_count} existing state {reset_noun}"
                )
                for move in reset_plan.moves:
                    print(
                        f"[stablewm-train] reset source {move.kind}: "
                        f"{move.source}"
                    )
                print("[stablewm-train] full_state_resume=fresh-after-reset")
            elif _stablepretraining_native_requeue():
                print("[stablewm-train] full_state_resume=native-scheduler-requeue")
            elif resume_checkpoint is not None:
                print(f"[stablewm-train] full_state_resume={resume_checkpoint}")
            else:
                print("[stablewm-train] full_state_resume=fresh")
        if train_command:
            print(f"[stablewm-train] train: {shlex.join(train_command)}")
        if eval_command:
            print(f"[stablewm-train] eval:  {shlex.join(eval_command)}")
    if args.print_command:
        return 0

    if args.resume == "reset":
        identities = {
            run_name: training_identity
            for run_name, _, _, _, training_identity, _ in commands
        }
        for run_name in reset_plans:
            training_identity = identities[run_name]
            assert training_identity is not None
            # Re-plan immediately before mutation so a path changed after the
            # read-only command preview cannot be silently omitted.
            current_plan = _plan_run_reset(
                checkpoint_root,
                run_name,
                output_root=output_root,
            )
            _execute_run_reset(
                current_plan,
                identity_sha256=training_identity["identity_sha256"],
            )

    if args.eval_only:
        if not checkpoint_root.is_dir():
            raise SystemExit(
                f"Evaluation-only checkpoint root does not exist: {checkpoint_root}"
            )
    else:
        checkpoint_root.mkdir(parents=True, exist_ok=True)
    if output_root is not None:
        output_root.mkdir(parents=True, exist_ok=True)
    if args.dataset_cache_root:
        Path(environment["LOCAL_DATASET_DIR"]).mkdir(parents=True, exist_ok=True)
    if (not args.eval_only and args.logger == "swanlab"
            and args.swanlab_mode != "offline"):
        _login_swanlab_without_exposing_key(environment)

    outcomes = {
        run_name: SeedOutcome(
            run_name=run_name,
            training_status=(
                "not_requested"
                if args.eval_only
                else "recovered" if train_command is None else "pending"
            ),
            evaluation_status="pending" if eval_command else "not_requested",
        )
        for run_name, train_command, eval_command, _, _, _ in commands
    }
    first_nonzero = 0

    # Finish or recover every requested training seed before starting the
    # post-training suite. A failed trainer must not suppress later requested
    # seeds; its own checkpoint is never handed to evaluation.
    for (
        run_name,
        train_command,
        _,
        _,
        training_identity,
        resume_checkpoint,
    ) in commands:
        outcome = outcomes[run_name]
        if train_command:
            assert training_identity is not None
            _install_training_identity(
                checkpoint_root,
                run_name,
                training_identity,
                replace_preflight_reservation=(
                    args.resume == "auto"
                    and not _stablepretraining_native_requeue()
                ),
            )
            training_environment = dict(environment)
            training_environment["PYTHONPATH"] = os.pathsep.join([
                str(STABLEWM_BOOTSTRAP_DIR),
                environment["PYTHONPATH"],
            ])
            training_environment["CONTEXTWORLD_SPT_BRIDGE"] = "1"
            training_environment["CONTEXTWORLD_SPT_RUN_NAME"] = run_name
            training_environment["CONTEXTWORLD_SPT_IDENTITY_SHA256"] = (
                training_identity["identity_sha256"]
            )
            if _is_contextworld_dataset_uri(target.dataset):
                training_environment["CONTEXTWORLD_STABLEWM_BUNDLE"] = "1"
                training_environment[
                    "CONTEXTWORLD_DATALOADER_START_METHOD"
                ] = "spawn"
            else:
                training_environment.pop(
                    "CONTEXTWORLD_STABLEWM_BUNDLE",
                    None,
                )
                training_environment.pop(
                    "CONTEXTWORLD_DATALOADER_START_METHOD",
                    None,
                )
            if resume_checkpoint is not None:
                training_environment["CONTEXTWORLD_SPT_RESUME_CHECKPOINT"] = str(
                    resume_checkpoint
                )
            else:
                training_environment.pop(
                    "CONTEXTWORLD_SPT_RESUME_CHECKPOINT",
                    None,
                )
            sys.stdout.flush()
            completed = subprocess.run(
                train_command,
                cwd=str(stablewm_repo),
                env=training_environment,
                check=False,
            )
            outcome.training_returncode = completed.returncode
            if completed.returncode != 0:
                outcome.training_status = "failed"
                print(
                    f"[stablewm-train] training failed for {run_name}: "
                    f"returncode={completed.returncode}; continuing remaining seeds"
                )
                if first_nonzero == 0:
                    first_nonzero = completed.returncode
            else:
                outcome.training_status = "completed"

    for (
        run_name,
        train_command,
        eval_command,
        _,
        _,
        _,
    ) in commands:
        outcome = outcomes[run_name]
        if eval_command is None:
            continue
        if train_command is not None and outcome.training_status == "failed":
            outcome.evaluation_status = "skipped_training_failed"
            print(
                f"[stablewm-train] evaluation skipped for {run_name}: "
                "training failed for this seed"
            )
            continue

        evaluated = subprocess.run(
            eval_command,
            cwd=str(REPO_ROOT),
            env=environment,
            check=False,
        )
        returncode = evaluated.returncode
        if returncode == 0 and target.original_env is None:
            assert eval_epoch is not None
            development_failure = _development_component_icl_failure(
                args=args,
                target=target,
                checkpoint_root=checkpoint_root,
                run_name=run_name,
                epoch=eval_epoch,
            )
            if development_failure is not None:
                print(
                    "[stablewm-train] Development ICL failed for "
                    f"{run_name}: {development_failure}"
                )
                returncode = 1
        outcome.evaluation_returncode = returncode
        if returncode != 0:
            outcome.evaluation_status = "failed"
            print(
                f"[stablewm-train] evaluation failed for {run_name}: "
                f"returncode={returncode}; continuing remaining seeds"
            )
            if first_nonzero == 0:
                first_nonzero = returncode
        else:
            outcome.evaluation_status = "completed"

    _print_seed_summary(list(outcomes.values()))
    print(f"[stablewm-train] first_nonzero={first_nonzero}")
    return first_nonzero


if __name__ == "__main__":
    raise SystemExit(main())
