#!/usr/bin/env python3
"""Public Stable-WorldModel training entry for ContextWorld.

The command line is stable; the Hydra keys are not.  LeWM and PLDM use a
``data`` defaults group and put loader settings below ``loader``.  PreJEPA
(DINO-WM) uses a flat dataset name, batch size and worker count.  This
launcher reads the checked-in family profile and translates only parameters
that the selected trainer actually accepts.

For a benchmark component, omitting ``--dataset`` with LeWM or PLDM selects
the component's frozen release recipe.  This keeps one public entry without
rewriting the byte-pinned launchers that define historical reproduction.

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

DEFAULT_PROFILE_CONFIG = (REPO_ROOT /
                          "configs/training/stablewm_family_profiles_v1.yaml")
TRAINING_IDENTITY_FILENAME = "contextworld_training_identity_v1.json"
TRAINING_IDENTITY_SCHEMA = "contextworld.stablewm-training-identity.v1"
FAMILY_ENTRY_SCRIPT = REPO_ROOT / "scripts/run_stablewm_family_entry.py"
STABLEWM_BOOTSTRAP_DIR = REPO_ROOT / "scripts/stablewm_bootstrap"
STABLEWM_SITECUSTOMIZE = STABLEWM_BOOTSTRAP_DIR / "sitecustomize.py"
SPT_RUN_MARKER_FILENAME = "contextworld_run_identity_v1.json"
SPT_RUN_MARKER_SCHEMA = "contextworld.stablepretraining-run-identity.v1"


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
    dataset: Path
    data_group: str | None
    history_size: int
    action_dim: int
    environment: str
    encoding_key: str | None = None
    encoding_dim: int | None = None
    original_env: str | None = None


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


def _family_model_columns(
    *,
    family: str,
    target: "Target",
    stablewm_repo: Path,
) -> tuple[str, ...]:
    """Return the literal model-input columns selected by the family profile."""

    required = ["pixels", "action"]
    if family == "prejepa":
        encoding_key = target.encoding_key or "proprio"
        if encoding_key not in required:
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


def validate_training_dataset_schema(
    *,
    target: "Target",
    family: str,
    stablewm_repo: Path,
) -> None:
    """Validate model columns and action width before allocating a trainer."""

    if target.dataset.is_file():
        try:
            import h5py
        except ImportError as exc:
            raise SystemExit(
                "H5 dataset preflight needs h5py. Install "
                "contextworld[stablewm] in the training environment."
            ) from exc
        try:
            with h5py.File(target.dataset, "r") as handle:
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
                    path=target.dataset,
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
                    path=target.dataset,
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
                f"{target.dataset}: {exc}"
            ) from exc
        return

    if not (target.dataset.is_dir() and target.dataset.name.endswith(".lance")):
        return
    try:
        import lance
    except ImportError as exc:
        raise SystemExit(
            "Lance dataset preflight needs pylance. Install "
            "contextworld[stablewm] in the training environment."
        ) from exc
    try:
        schema = lance.dataset(str(target.dataset)).schema
    except Exception as exc:
        raise SystemExit(
            f"Could not open Lance dataset for schema preflight: "
            f"{target.dataset}: {exc}"
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
        path=target.dataset,
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
        path=target.dataset,
        feature=target.encoding_key or "auxiliary input",
        observed=encoding_width,
        expected=target.encoding_dim,
        target_label=target.label,
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
    if not explicit:
        raise SystemExit(
            "Benchmark component training needs an exact --dataset path. "
            "Use task_registry.json from the clean export to select a payload.")
    dataset = _absolute_path(explicit, label="--dataset")
    _validate_dataset_payload(dataset)
    spec = components[args.component]
    return Target(
        label=args.component,
        dataset=dataset,
        data_group=args.data_group or spec.get("data_group"),
        history_size=args.history_size or int(spec["history_size"]),
        action_dim=int(spec["action_dim"]),
        environment=str(spec["environment"]),
        encoding_key=str(spec["encoding_key"]),
        encoding_dim=int(spec["encoding_dim"]),
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
    return path, payload


def _add(entries: list[str], key: str | None, value: Any) -> None:
    if value is None or key is None:
        return
    if isinstance(value, bool):
        value = _hydra_bool(value)
    entries.append(f"{key}={value}")


def _validate_positive(name: str, value: int | float | None) -> None:
    if value is not None and value <= 0:
        raise SystemExit(f"{name} must be positive; got {value}")


def _validate_args(args: argparse.Namespace, family: str) -> None:
    for name in (
            "batch_size",
            "frameskip",
            "history_size",
            "num_preds",
            "max_epochs",
            "accumulate",
            "prefetch_factor",
            "embed_dim",
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
    lewm_requested = any(value is not None for value in (
        args.lewm_regularizer,
        args.lewm_sigreg_weight,
        args.lewm_visreg_weight,
        args.lewm_visreg_num_projections,
        args.lewm_visreg_lambda_scale,
        args.lewm_visreg_lambda_shape,
        args.lewm_visreg_lambda_center,
    ))
    if family != "lewm" and lewm_requested:
        raise SystemExit("LeWM loss options are only valid with --family lewm")
    visreg_requested = any(value is not None for value in (
        args.lewm_visreg_weight,
        args.lewm_visreg_num_projections,
        args.lewm_visreg_lambda_scale,
        args.lewm_visreg_lambda_shape,
        args.lewm_visreg_lambda_center,
    ))
    if visreg_requested and args.lewm_regularizer != "visreg":
        raise SystemExit("VISReg parameters require --lewm-regularizer visreg so they "
                         "cannot be accepted by an inactive objective.")
    if args.lewm_regularizer == "visreg" and args.lewm_sigreg_weight is not None:
        raise SystemExit(
            "--lewm-sigreg-weight is inactive with VISReg and must be omitted.")
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

    _validate_args(args, family)
    if profile["data_group"] and not target.data_group:
        raise SystemExit(f"{family} requires --data-group for a non-original dataset")
    if not profile["data_group"] and args.data_group:
        raise SystemExit(f"{family} does not use a Hydra data group")

    entries: list[str] = []
    if profile["data_group"]:
        entries.append(f"data={target.data_group}")
    _add(entries, keys["dataset"], target.dataset)
    _add(entries, common["seed"], seed)
    _add(entries, common["run_name"], run_name)
    _add(entries, common["run_subdir"], run_name)

    frameskip = args.frameskip or contract["defaults"]["frameskip"]
    num_preds = args.num_preds or contract["defaults"]["num_preds"]
    for name, value in (
        ("batch_size", args.batch_size),
        ("num_workers", args.num_workers),
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
        ("precision", args.precision),
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
        # The upstream default names `proprio`. Reacher and Cube datasets name
        # that column `observation`; this applies to both original datasets and
        # benchmark-component projections. Upstream derives its true width at
        # load, so the profile selects the column but never hard-codes its raw
        # dimension into the action encoder.
        if target.encoding_key != "proprio":
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
        if args.lewm_regularizer is not None:
            if _nested(upstream_config, "loss", "regularizer") is None:
                raise SystemExit(
                    "This LeWM checkout does not expose loss.regularizer; "
                    "VISReg and alternate regularizers require the published "
                    "ContextWorld extension or a compatible checkout.")
            _add(entries, "loss.regularizer", args.lewm_regularizer)
        _add(entries, "loss.sigreg.weight", args.lewm_sigreg_weight)
        for key, value in (
            ("weight", args.lewm_visreg_weight),
            ("kwargs.num_projections", args.lewm_visreg_num_projections),
            ("kwargs.lambda_scale", args.lewm_visreg_lambda_scale),
            ("kwargs.lambda_shape", args.lewm_visreg_lambda_shape),
            ("kwargs.lambda_center", args.lewm_visreg_lambda_center),
        ):
            if value is not None and _nested(upstream_config, "loss", "visreg") is None:
                raise SystemExit(
                    "This LeWM checkout has no VISReg config; use a compatible "
                    "checkout/extension rather than adding ignored Hydra keys.")
            _add(entries, f"loss.visreg.{key}", value)

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


def _run_name(args: argparse.Namespace, target: Target, seed: int,
              seeds: tuple[int, ...]) -> str:
    default_base = (f"{target.original_env}_{args.family}_original"
                    if target.original_env else f"{target.label}_{args.family}")
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


def _dataset_identity(path: Path) -> dict[str, Any]:
    """Record a cheap, fail-closed identity without hashing a multi-GB dataset."""

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
) -> None:
    """Write once for a new run; require an exact match on native requeue."""

    run_dir = checkpoint_root / "checkpoints" / run_name
    path = run_dir / TRAINING_IDENTITY_FILENAME
    if path.exists():
        _assert_training_identity(path, expected, context=f"Run {run_name!r}")
        return
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


def _portable_resume_candidates(
    root: Path,
    run_name: str,
    identity_sha256: str,
) -> list[Path]:
    """Return full-state SPT checkpoints bound to this exact run recipe."""

    runs_root = root / "runs"
    if not runs_root.is_dir():
        return []
    candidates: list[Path] = []
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

    if policy == "never":
        if run_nonempty or candidates:
            raise SystemExit(
                f"Fresh training refuses existing state for run {run_name!r}. "
                "Choose another run name or set --resume auto/required."
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
        return None
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
        choices=("never", "auto", "required"),
        default=_env("CW_RESUME", contract["defaults"]["resume"]),
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

    parser.add_argument("--lewm-regularizer", default=_env("CW_LEWM_REGULARIZER"))
    parser.add_argument("--lewm-sigreg-weight",
                        type=float,
                        default=_env_float("CW_LEWM_SIGREG_WEIGHT"))
    parser.add_argument("--lewm-visreg-weight",
                        type=float,
                        default=_env_float("CW_LEWM_VISREG_WEIGHT"))
    parser.add_argument("--lewm-visreg-num-projections",
                        type=int,
                        default=_env_int("CW_LEWM_VISREG_NUM_PROJECTIONS"))
    parser.add_argument("--lewm-visreg-lambda-scale",
                        type=float,
                        default=_env_float("CW_LEWM_VISREG_LAMBDA_SCALE"))
    parser.add_argument("--lewm-visreg-lambda-shape",
                        type=float,
                        default=_env_float("CW_LEWM_VISREG_LAMBDA_SHAPE"))
    parser.add_argument("--lewm-visreg-lambda-center",
                        type=float,
                        default=_env_float("CW_LEWM_VISREG_LAMBDA_CENTER"))

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

    return bool(
        args.component
        and not args.original_env
        and not args.dataset
        and args.family in {"lewm", "pldm"}
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
    if _uses_release_recipe(args):
        return _run_release_reproduction(args)
    target = resolve_target(args, contract)
    stablewm_repo = resolve_stablewm_repo(args)
    validate_training_dataset_schema(
        target=target,
        family=args.family,
        stablewm_repo=stablewm_repo,
    )
    checkpoint_root = resolve_checkpoint_root(args)
    profile = contract["families"][args.family]
    trainer_script = stablewm_repo / profile["entrypoint"]
    seeds = args.seeds
    original_eval_dataset = (
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
            and args.resume != "never"
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
    print(f"[stablewm-train] stablewm={stablewm_repo}")
    print(f"[stablewm-train] checkpoint_root={checkpoint_root}")
    print(f"[stablewm-train] spt_cache={environment['SPT_CACHE_DIR']}")
    print("[stablewm-train] dataset_cache="
          f"{environment.get('LOCAL_DATASET_DIR', '<upstream default>')}")
    print(f"[stablewm-train] logger={args.logger} resume={args.resume}")
    print(f"[stablewm-train] mode={'eval-only' if args.eval_only else 'train'}")
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
            if _stablepretraining_native_requeue():
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

    if args.eval_only:
        if not checkpoint_root.is_dir():
            raise SystemExit(
                f"Evaluation-only checkpoint root does not exist: {checkpoint_root}"
            )
    else:
        checkpoint_root.mkdir(parents=True, exist_ok=True)
    if args.output:
        _absolute_path(args.output, label="--output").mkdir(parents=True, exist_ok=True)
    if args.dataset_cache_root:
        Path(environment["LOCAL_DATASET_DIR"]).mkdir(parents=True, exist_ok=True)
    if (not args.eval_only and args.logger == "swanlab"
            and args.swanlab_mode != "offline"):
        _login_swanlab_without_exposing_key(environment)

    # Finish or recover every requested training seed before starting the
    # post-training suite.  A CEM/ICL failure for an earlier seed must never
    # prevent a later seed from reaching its requested training checkpoint.
    for (
        run_name,
        train_command,
        _,
        _,
        training_identity,
        resume_checkpoint,
    ) in commands:
        if train_command:
            assert training_identity is not None
            _install_training_identity(
                checkpoint_root,
                run_name,
                training_identity,
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
            if completed.returncode != 0:
                return completed.returncode

    for (
        _,
        _,
        eval_command,
        _,
        _,
        _,
    ) in commands:
        if eval_command:
            evaluated = subprocess.run(
                eval_command,
                cwd=str(REPO_ROOT),
                env=environment,
                check=False,
            )
            if evaluated.returncode != 0:
                return evaluated.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
