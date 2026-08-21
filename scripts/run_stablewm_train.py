#!/usr/bin/env python3
"""Public Stable-WorldModel training entry for ContextWorld.

The command line is stable; the Hydra keys are not.  LeWM and PLDM use a
``data`` defaults group and put loader settings below ``loader``.  PreJEPA
(DINO-WM) uses a flat dataset name, batch size and worker count.  This
launcher reads the checked-in family profile and translates only parameters
that the selected trainer actually accepts.

The selected Stable-WorldModel checkout still owns the model, objective,
forward pass, optimizer and training loop.  ContextWorld owns path validation,
run isolation, logger credentials, family-specific argument mapping and the
optional hand-off to original-environment MPC evaluation.
"""

from __future__ import annotations

import argparse
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
DEFAULT_PROFILE_CONFIG = (REPO_ROOT /
                          "configs/training/stablewm_family_profiles_v1.yaml")


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
        key, separator, _ = override.partition("=")
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
        reads_wandb = "cfg.wandb" in source
        if not reads_wandb:
            if backend != "none":
                raise SystemExit("This PreJEPA trainer has no logger integration. "
                                 "Use --logger none.")
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


def _resume_checkpoint(root: Path, run_name: str) -> Path:
    return root / "checkpoints" / run_name / f"{run_name}_weights.ckpt"


def validate_resume(root: Path, run_name: str, policy: str) -> None:
    run_dir = root / "checkpoints" / run_name
    checkpoint = _resume_checkpoint(root, run_name)
    if policy == "required" and not checkpoint.is_file():
        raise SystemExit(
            f"Resume was required, but the full-state checkpoint is missing: "
            f"{checkpoint}")
    if policy == "never" and run_dir.exists() and any(run_dir.iterdir()):
        raise SystemExit(
            f"Fresh training refuses the non-empty run directory: {run_dir}. "
            "Choose another run name or set --resume auto/required.")


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
) -> list[str]:
    if not target.original_env:
        raise SystemExit(
            "--post-eval is only the original-environment MPC sanity/CEM "
            "stage. Benchmark ICL/CEM uses each component's frozen evaluator.")
    if profile["post_train_eval"] != "validated":
        raise SystemExit(
            f"Post-train eval is not yet checkpoint-smoke-validated for "
            f"{args.family}; train first, then run the explicit evaluator.")
    epoch = _effective_eval_epoch(args, stablewm_repo, profile)
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts/run_stablewm_eval.py"),
        "--family",
        args.family,
        "--original-env",
        target.original_env,
        "--dataset",
        str(target.dataset),
        "--run-name",
        run_name,
        "--epoch",
        str(epoch),
        "--checkpoint-root",
        str(checkpoint_root),
        "--stablewm-repo",
        str(stablewm_repo),
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
    if args.eval_keep_videos:
        command.append("--keep-videos")
    if args.print_command:
        command.append("--print-command")
    return command


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
    parser.add_argument("--seed", type=int, default=int(_env("CW_SEED", "3072")))
    parser.add_argument(
        "--all-seeds",
        action="store_true",
        default=bool(_env_bool("CW_ALL_SEEDS") or False),
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
    parser.add_argument("--eval-epoch", type=int, default=_env_int("CW_EVAL_EPOCH"))
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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    contract = load_profile_contract(args.profile_config.resolve())
    if args.family not in contract["families"]:
        raise SystemExit(f"Unknown family in profile contract: {args.family}")
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
    seeds = (tuple(
        int(seed)
        for seed in contract["defaults"]["baseline_seeds"]) if args.all_seeds else
             (args.seed, ))

    environment = dict(os.environ)
    environment["STABLEWM_HOME"] = str(checkpoint_root)
    if args.dataset_cache_root:
        cache = _absolute_path(args.dataset_cache_root, label="--dataset-cache-root")
        environment["LOCAL_DATASET_DIR"] = str(cache)
    environment["PYTHONPATH"] = os.pathsep.join([
        str(REPO_ROOT),
        str(stablewm_repo),
        environment.get("PYTHONPATH", ""),
    ]).strip(os.pathsep)

    commands: list[tuple[list[str], list[str] | None]] = []
    for seed in seeds:
        run_name = _run_name(args, target, seed, seeds)
        overrides = build_overrides(
            args,
            contract,
            target,
            run_name=run_name,
            seed=seed,
            stablewm_repo=stablewm_repo,
        )
        train_command = [
            sys.executable,
            str(trainer_script),
            f"--config-name={profile['config_name']}",
            *overrides,
        ]
        eval_command = (_post_eval_command(
            args,
            target=target,
            run_name=run_name,
            checkpoint_root=checkpoint_root,
            stablewm_repo=stablewm_repo,
            profile=profile,
            frameskip=(args.frameskip or contract["defaults"]["frameskip"]),
        ) if args.post_eval else None)
        commands.append((train_command, eval_command))

    # Resume policy is read-only to validate, so dry runs should catch a stale
    # or missing run just as a real launch would.
    for train_command, _ in commands:
        run_entry = next(item for item in train_command if item.startswith("subdir="))
        validate_resume(
            checkpoint_root,
            run_entry.split("=", 1)[1],
            args.resume,
        )

    print(f"[stablewm-train] target={target.label} family={args.family} "
          f"dataset={target.dataset}")
    print(f"[stablewm-train] stablewm={stablewm_repo}")
    print(f"[stablewm-train] checkpoint_root={checkpoint_root}")
    print("[stablewm-train] dataset_cache="
          f"{environment.get('LOCAL_DATASET_DIR', '<upstream default>')}")
    print(f"[stablewm-train] logger={args.logger} resume={args.resume}")
    for train_command, eval_command in commands:
        print(f"[stablewm-train] train: {shlex.join(train_command)}")
        if eval_command:
            print(f"[stablewm-train] eval:  {shlex.join(eval_command)}")
    if args.print_command:
        return 0

    checkpoint_root.mkdir(parents=True, exist_ok=True)
    if args.output:
        _absolute_path(args.output, label="--output").mkdir(parents=True, exist_ok=True)
    if args.dataset_cache_root:
        Path(environment["LOCAL_DATASET_DIR"]).mkdir(parents=True, exist_ok=True)
    if args.logger == "swanlab" and args.swanlab_mode != "offline":
        _login_swanlab_without_exposing_key(environment)

    for train_command, eval_command in commands:
        sys.stdout.flush()
        completed = subprocess.run(
            train_command,
            cwd=str(stablewm_repo),
            env=environment,
            check=False,
        )
        if completed.returncode != 0:
            return completed.returncode
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
