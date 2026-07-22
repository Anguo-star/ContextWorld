#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
from functools import partial
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from contextworld.synthesis.manifest import write_json
from contextworld.synthesis.stablewm import load_stable_worldmodel
from contextworld.paths import artifact_path, resolve_contextworld_path
from contextworld.training.tworoom_data import build_tworoom_grouped_data


PINNED_STABLEWM = "5864b74980f6ed328fd0045e777b3865962eff43"

FORMAL_TOPOLOGIES = {
    (4, 128, 2): "4gpu_x_b128_x_accum2",
    (8, 128, 1): "8gpu_x_b128_x_accum1",
}

PROFILE_DEFAULTS = {
    "smoke": {
        "run_kind": "adapter_smoke",
        "epoch_size": 120,
        "validation_epoch_size": 120,
        "max_epochs": 1,
        "batch_size": 4,
        "num_workers": 0,
        "devices": 1,
        "precision": "bf16-mixed",
        "accumulate_grad_batches": 1,
        "limit_train_batches": 2,
        "limit_val_batches": 1,
        "expected_optimizer_steps": 2,
    },
    "formal": {
        "run_kind": "pilot",
        "epoch_size": 1_314_816,
        "validation_epoch_size": 12_288,
        "max_epochs": 5,
        "batch_size": 128,
        "num_workers": 6,
        "devices": 4,
        "precision": "bf16-mixed",
        "accumulate_grad_batches": 2,
        "limit_train_batches": 1.0,
        "limit_val_batches": 1.0,
        "expected_optimizer_steps": 6_420,
    },
    "additive": {
        "run_kind": "confirmation",
        "epoch_size": 2_629_632,
        "validation_epoch_size": 12_288,
        "max_epochs": 5,
        "batch_size": 128,
        "num_workers": 6,
        "devices": 4,
        "precision": "bf16-mixed",
        "accumulate_grad_batches": 2,
        "limit_train_batches": 1.0,
        "limit_val_batches": 1.0,
        "expected_optimizer_steps": 12_840,
    },
}


def _parse_batch_limit(value: str) -> int | float:
    raw = value.strip()
    if raw.lstrip("+").isdigit():
        parsed_integer = int(raw)
        if parsed_integer <= 0:
            raise argparse.ArgumentTypeError("batch limit must be positive")
        return parsed_integer
    parsed = float(raw)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("batch limit must be positive")
    if parsed <= 1:
        return parsed
    if not parsed.is_integer():
        raise argparse.ArgumentTypeError(
            "batch limits above 1 must be integer batch counts"
        )
    return int(parsed)


def _apply_profile(args: argparse.Namespace) -> argparse.Namespace:
    defaults = PROFILE_DEFAULTS[args.profile]
    for key, value in defaults.items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)

    args.expected_warmup_steps = max(
        1, int(0.01 * args.expected_optimizer_steps)
    )
    positive_fields = (
        "epoch_size",
        "validation_epoch_size",
        "max_epochs",
        "batch_size",
        "devices",
        "accumulate_grad_batches",
        "expected_optimizer_steps",
    )
    invalid = {
        name: getattr(args, name)
        for name in positive_fields
        if getattr(args, name) <= 0
    }
    if args.num_workers < 0:
        invalid["num_workers"] = args.num_workers
    if invalid:
        raise ValueError(f"Training profile values are invalid: {invalid}")
    if args.profile in {"formal", "additive"}:
        fixed = {
            key: value
            for key, value in defaults.items()
            if key
            not in {
                "run_kind",
                "num_workers",
                "devices",
                "batch_size",
                "accumulate_grad_batches",
            }
        }
        mismatches = {
            key: {"expected": value, "observed": getattr(args, key)}
            for key, value in fixed.items()
            if getattr(args, key) != value
        }
        if mismatches:
            raise ValueError(
                f"{args.profile.title()} profile fields are frozen: "
                f"{mismatches}"
            )
        if args.run_kind not in {"pilot", "confirmation"}:
            raise ValueError(
                f"{args.profile.title()} profile requires run-kind pilot or confirmation"
            )
        topology = (
            args.devices,
            args.batch_size,
            args.accumulate_grad_batches,
        )
        if topology not in FORMAL_TOPOLOGIES:
            raise ValueError(
                f"{args.profile.title()} profile requires one of the validated execution "
                f"topologies {sorted(FORMAL_TOPOLOGIES)}; observed={topology}"
            )
    return args


def _effective_batches(total: int, limit: int | float) -> int:
    if isinstance(limit, int):
        return min(total, limit)
    if not 0 < limit <= 1:
        raise ValueError(f"Invalid fractional batch limit: {limit}")
    return int(total * limit)


def _build_training_plan(args: argparse.Namespace, data_metadata: dict) -> dict:
    samples_per_rank = math.ceil(args.epoch_size / args.devices)
    batches_per_rank = samples_per_rank // args.batch_size
    executed_batches_per_rank = _effective_batches(
        batches_per_rank, args.limit_train_batches
    )
    if executed_batches_per_rank % args.accumulate_grad_batches:
        raise ValueError(
            "Executed train batches are not divisible by gradient accumulation: "
            f"{executed_batches_per_rank} vs {args.accumulate_grad_batches}"
        )
    optimizer_steps_per_epoch = (
        executed_batches_per_rank // args.accumulate_grad_batches
    )
    computed_steps = optimizer_steps_per_epoch * args.max_epochs
    if computed_steps != args.expected_optimizer_steps:
        raise ValueError(
            "Training plan does not produce the expected optimizer budget: "
            f"computed={computed_steps}, expected={args.expected_optimizer_steps}"
        )

    microbatch_global_size = args.batch_size * args.devices
    global_batch_size = (
        args.batch_size * args.devices * args.accumulate_grad_batches
    )
    full_logical_epochs_executed = (
        executed_batches_per_rank == batches_per_rank
        and args.epoch_size == batches_per_rank * microbatch_global_size
    )
    epoch_group_counts = data_metadata["epoch_group_counts"]
    logical_budget_group_draws = {
        name: int(count) * args.max_epochs
        for name, count in epoch_group_counts.items()
    }
    group_exposure = {}
    for name, draws in logical_budget_group_draws.items():
        group = data_metadata["groups"][name]
        if name == "original":
            raw_clips = int(group["train_clips"])
        else:
            raw_clips = int(group["train_clips_raw"])
        if name == "original":
            unique_raw_clips = min(
                int(epoch_group_counts[name]), raw_clips
            )
        else:
            virtual = data_metadata["epoch_group_coverage"][name]
            if (
                virtual["unique_virtual_slots"]
                == virtual["available_virtual_slots"]
            ):
                unique_raw_clips = raw_clips
            else:
                unique_raw_clips = None
        if not full_logical_epochs_executed:
            unique_raw_clips = None
        group_exposure[name] = {
            "total_draws": draws if full_logical_epochs_executed else None,
            "logical_epoch_budget_draws": draws,
            "exposure_is_exact": full_logical_epochs_executed,
            "raw_train_clips": raw_clips,
            "mean_draws_per_raw_clip": (
                draws / raw_clips if full_logical_epochs_executed else None
            ),
            "logical_budget_mean_draws_per_raw_clip": draws / raw_clips,
            "unique_raw_clips_exposed": unique_raw_clips,
            "raw_clips_never_drawn": (
                raw_clips - unique_raw_clips
                if unique_raw_clips is not None
                else None
            ),
            "run_unique_raw_fraction": (
                unique_raw_clips / raw_clips
                if unique_raw_clips is not None
                else None
            ),
        }

    data_quality_gates = {}
    for name, exposure in group_exposure.items():
        requirements = dict(
            data_metadata["groups"][name].get("quality_requirements", {})
        )
        if not requirements:
            continue
        maximum_reuse = requirements.get(
            "maximum_formal_mean_draws_per_raw_clip"
        )
        gates = {
            "static": all(
                data_metadata["groups"][name]
                .get("static_quality_gates", {"unconfigured": True})
                .values()
            ),
            "formal_reuse": (
                args.profile not in {"formal", "additive"}
                or maximum_reuse is None
                or exposure["logical_budget_mean_draws_per_raw_clip"]
                <= float(maximum_reuse)
            ),
        }
        data_quality_gates[name] = {
            "passed": all(gates.values()),
            "requirements": requirements,
            "observed_mean_draws_per_raw_clip": exposure[
                "logical_budget_mean_draws_per_raw_clip"
            ],
            "gates": gates,
        }
    failed_quality = {
        name: value
        for name, value in data_quality_gates.items()
        if not value["passed"]
    }
    if failed_quality:
        raise ValueError(
            "Training data-quality gates failed: "
            f"{failed_quality}"
        )

    return {
        "profile": args.profile,
        "epoch_size_global_samples": args.epoch_size,
        "logical_epochs": args.max_epochs,
        "batch_size_per_device": args.batch_size,
        "devices": args.devices,
        "execution_topology": FORMAL_TOPOLOGIES.get(
            (
                args.devices,
                args.batch_size,
                args.accumulate_grad_batches,
            ),
            "custom_smoke",
        ),
        "microbatch_global_size": microbatch_global_size,
        "adapter_gradient_accumulation_steps": (
            args.accumulate_grad_batches
        ),
        "gradient_accumulation_implementation": (
            "stablepretraining_manual_optimizer_frequency_with_explicit_loss_scaling"
        ),
        "trainer_accumulate_grad_batches": 1,
        "global_batch_size": global_batch_size,
        "samples_per_rank_before_drop_last": samples_per_rank,
        "batches_per_rank_before_limit": batches_per_rank,
        "executed_batches_per_rank_per_epoch": executed_batches_per_rank,
        "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
        "optimizer_steps_total": computed_steps,
        "microbatches_per_rank_total": (
            computed_steps * args.accumulate_grad_batches
        ),
        "warmup_steps": args.expected_warmup_steps,
        "scheduler_max_steps": args.expected_optimizer_steps,
        "full_logical_epochs_executed": full_logical_epochs_executed,
        "total_global_sample_draws": computed_steps * global_batch_size,
        "group_exposure": group_exposure,
        "data_quality_gates": data_quality_gates,
        "data_split_seed": args.data_split_seed,
        "training_seed": args.seed,
        "sampling_mapping_repeats_each_logical_epoch": True,
        "budget_equivalence_note": (
            "The formal profile preserves the historical total draw budget. "
            "The additive profile preserves that budget independently for "
            "both original and synthetic groups, doubling optimizer steps."
        ),
    }


def _process_is_global_zero() -> bool:
    return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))) == 0


def _lejepa_forward_with_manual_accumulation(
    self,
    batch,
    stage,
    *,
    base_forward,
    cfg,
    accumulation_steps: int,
):
    """Preserve per-rank batch statistics while accumulating exact gradients."""

    state = base_forward(self, batch, stage, cfg)
    if stage == "fit" and accumulation_steps > 1:
        # StablePretraining's manual optimizer frequency delays optimizer.step,
        # but its stock training_step does not call rescale_loss_for_grad_acc.
        # Divide here after the upstream forward has logged the unscaled losses.
        state["loss"] = state["loss"] / accumulation_steps
    return state


def _sample_contract(dataset, count: int = 8) -> dict:
    indices = list(range(min(count, len(dataset))))
    samples = dataset.__getitems__(indices)
    expected = {
        "pixels": [4, 3, 224, 224],
        "action": [4, 10],
        "proprio": [4, 2],
    }
    observed = {
        key: list(samples[0][key].shape) for key in sorted(expected)
    }
    if observed != expected:
        raise RuntimeError(
            f"History-3 training sample contract mismatch: {observed}"
        )
    for sample in samples[1:]:
        shapes = {key: list(sample[key].shape) for key in sorted(expected)}
        if shapes != expected:
            raise RuntimeError(f"Cross-group sample shape mismatch: {shapes}")
    return {
        "passed": True,
        "sample_count": len(samples),
        "shapes": observed,
        "keys": sorted(samples[0]),
        "batched_dataset_read_exercised": True,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_pinned_train_module(stable_repo: Path):
    path = stable_repo / "scripts/train/lewm.py"
    spec = importlib.util.spec_from_file_location(
        "contextworld_pinned_stablewm_lewm_train", path
    )
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _compose_model_config(stable_repo: Path, args):
    import hydra
    from omegaconf import open_dict

    config_dir = str((stable_repo / "scripts/train/config").resolve())
    with hydra.initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = hydra.compose(config_name="lewm", overrides=["data=tworoom"])
    with open_dict(cfg):
        cfg.output_model_name = args.run_name
        cfg.subdir = args.run_name
        cfg.seed = args.seed
        cfg.trainer.max_epochs = args.max_epochs
        cfg.trainer.max_steps = args.expected_optimizer_steps
        cfg.trainer.devices = args.devices
        cfg.trainer.precision = args.precision
        cfg.loader.batch_size = args.batch_size
        cfg.loader.num_workers = args.num_workers
        cfg.model.action_encoder.input_dim = 10
    return cfg


def _validate_resume_policy(
    *, run_dir: Path, checkpoint_path: Path, policy: str
) -> Path | None:
    """Apply launch safety and return the checkpoint Manager should load."""

    if policy not in {"auto", "never", "required"}:
        raise ValueError(f"Unsupported resume policy: {policy}")
    checkpoint_exists = checkpoint_path.is_file()
    run_has_artifacts = run_dir.is_dir() and any(run_dir.iterdir())
    if policy == "required" and not checkpoint_exists:
        raise FileNotFoundError(
            "StableWM resume checkpoint is required but missing: "
            f"{checkpoint_path}"
        )
    if policy == "never" and run_has_artifacts:
        raise FileExistsError(
            f"Fresh training requires an empty run directory: {run_dir}"
        )
    if policy == "auto" and run_has_artifacts and not checkpoint_exists:
        raise FileExistsError(
            "The run directory contains artifacts but no StableWM native "
            f"resume checkpoint at {checkpoint_path}. Refusing to infer "
            "optimizer/scheduler state from model-only weights."
        )
    # stable_pretraining.Manager treats any non-None path as an explicit
    # resume request and rejects a path that does not exist.  Fresh and empty
    # auto launches therefore must pass None, while still retaining the
    # canonical path for checkpoint callbacks and provenance reporting.
    return checkpoint_path if checkpoint_exists else None


def _full_state_checkpoint_metadata(
    checkpoint_path: Path,
    *,
    expected_optimizer_steps: int,
    require_incomplete: bool,
    expected_world_size: int | None = None,
    optimizer_steps_per_epoch: int | None = None,
) -> dict:
    """Fail closed unless ``checkpoint_path`` is a resumable trainer state."""

    import torch

    try:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not read full-state checkpoint {checkpoint_path}: {exc}"
        ) from exc
    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            f"Full-state checkpoint is not a mapping: {checkpoint_path}"
        )
    required = {
        "state_dict",
        "optimizer_states",
        "lr_schedulers",
        "global_step",
        "epoch",
        "loops",
        "contextworld_rng_states_v1",
    }
    missing = sorted(required - set(checkpoint))
    if missing:
        raise RuntimeError(
            "Checkpoint cannot safely resume optimizer/scheduler/RNG state; "
            f"missing={missing}, path={checkpoint_path}"
        )
    if not checkpoint["optimizer_states"] or not checkpoint["lr_schedulers"]:
        raise RuntimeError(
            "Checkpoint has no optimizer or scheduler state: "
            f"{checkpoint_path}"
        )
    rng_states = checkpoint["contextworld_rng_states_v1"]
    if not isinstance(rng_states, list) or not rng_states:
        raise RuntimeError(
            f"Checkpoint has no per-rank RNG state: {checkpoint_path}"
        )
    rng_required = {
        "rank",
        "python",
        "numpy",
        "torch_cpu",
        "torch_cuda",
        "train_loader_generator",
    }
    invalid_rng_rows = [
        index
        for index, row in enumerate(rng_states)
        if not isinstance(row, dict) or not rng_required.issubset(row)
    ]
    if invalid_rng_rows:
        raise RuntimeError(
            "Checkpoint contains incomplete per-rank RNG state: "
            f"rows={invalid_rng_rows}, path={checkpoint_path}"
        )
    ranks = sorted(int(row["rank"]) for row in rng_states)
    if ranks != list(range(len(rng_states))):
        raise RuntimeError(
            f"Checkpoint RNG rank coverage is invalid: {ranks}"
        )
    if expected_world_size is not None and len(ranks) != expected_world_size:
        raise RuntimeError(
            "Checkpoint world size differs from the frozen execution topology: "
            f"checkpoint={len(ranks)}, expected={expected_world_size}"
        )
    global_step = int(checkpoint["global_step"])
    if global_step <= 0:
        raise RuntimeError(
            f"Resume checkpoint has no completed optimizer step: {checkpoint_path}"
        )
    if global_step > expected_optimizer_steps:
        raise RuntimeError(
            "Resume checkpoint exceeds the frozen optimizer budget: "
            f"step={global_step}, expected={expected_optimizer_steps}"
        )
    if require_incomplete and global_step >= expected_optimizer_steps:
        raise RuntimeError(
            "Training is already complete at the frozen optimizer budget; "
            f"refusing to resume step={global_step}"
        )
    if (
        optimizer_steps_per_epoch is not None
        and global_step % optimizer_steps_per_epoch != 0
    ):
        raise RuntimeError(
            "Resume checkpoint is not at a complete epoch boundary: "
            f"step={global_step}, steps_per_epoch={optimizer_steps_per_epoch}, "
            f"path={checkpoint_path}"
        )
    return {
        "path": str(checkpoint_path.resolve()),
        "sha256": _sha256(checkpoint_path),
        "global_step": global_step,
        "epoch": int(checkpoint["epoch"]),
        "optimizer_states": len(checkpoint["optimizer_states"]),
        "lr_schedulers": len(checkpoint["lr_schedulers"]),
        "rng_ranks": ranks,
        "complete_epoch_boundary": (
            global_step % optimizer_steps_per_epoch == 0
            if optimizer_steps_per_epoch is not None
            else None
        ),
        "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
    }


def _normalize_complete_epoch_resume_loop_state(
    checkpoint: dict,
    *,
    optimizer_steps_per_epoch: int,
    accumulation_steps: int,
) -> dict:
    """Start a completed-epoch checkpoint at batch zero of the next epoch.

    Lightning 2.6 restores ``batch_progress.current`` from an epoch-end
    checkpoint.  Without resetting that per-epoch counter, its data fetcher
    fast-forwards by a full epoch and executes zero new batches.  Cumulative
    totals, epoch/global step, optimizer, and scheduler state remain intact.
    """

    global_step = int(checkpoint.get("global_step", -1))
    expected_batches = optimizer_steps_per_epoch * accumulation_steps
    if global_step <= 0 or global_step % optimizer_steps_per_epoch != 0:
        raise RuntimeError(
            "Cannot normalize a checkpoint outside a complete epoch "
            f"boundary: step={global_step}, "
            f"steps_per_epoch={optimizer_steps_per_epoch}"
        )
    expected_epoch = global_step // optimizer_steps_per_epoch
    if int(checkpoint.get("epoch", -1)) != expected_epoch:
        raise RuntimeError(
            "Checkpoint epoch/global-step boundary mismatch: "
            f"epoch={checkpoint.get('epoch')}, expected={expected_epoch}"
        )
    try:
        batch_progress = checkpoint["loops"]["fit_loop"][
            "epoch_loop.batch_progress"
        ]
        current = batch_progress["current"]
        total = batch_progress["total"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError(
            "Checkpoint is missing Lightning train batch-loop progress"
        ) from exc
    counter_names = ("ready", "started", "processed", "completed")
    observed_current = {
        name: int(current.get(name, -1)) for name in counter_names
    }
    expected_current = {name: expected_batches for name in counter_names}
    if observed_current != expected_current or batch_progress.get(
        "is_last_batch"
    ) is not True:
        raise RuntimeError(
            "Checkpoint is not at the end of a complete train epoch: "
            f"current={observed_current}, expected={expected_current}, "
            f"is_last_batch={batch_progress.get('is_last_batch')!r}"
        )
    observed_total = {name: int(total.get(name, -1)) for name in counter_names}
    if any(observed_total[name] < expected_batches for name in counter_names):
        raise RuntimeError(
            f"Checkpoint cumulative batch progress is invalid: {observed_total}"
        )
    batch_progress["current"] = {name: 0 for name in counter_names}
    batch_progress["is_last_batch"] = False
    return {
        "applied": True,
        "global_step": global_step,
        "completed_epochs": expected_epoch,
        "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
        "microbatches_per_epoch_per_rank": expected_batches,
        "previous_current_batch_progress": observed_current,
        "cumulative_batch_progress": observed_total,
        "next_epoch_starts_at_batch_zero": True,
    }


def _configure_spt_for_controlled_checkpointing(spt) -> dict:
    """Keep Manager output under ContextWorld's explicit Trainer run dir.

    The pinned runtime's public ``spt.set(cache_dir=None)`` currently leaves
    the default cache unchanged, despite its documented API.  Assign through
    the typed config property and assert the result so a future runtime change
    fails visibly instead of redirecting checkpoints to an anonymous cache.
    """

    from stable_pretraining._config import get_config

    runtime = get_config()
    runtime.cache_dir = None
    spt.set(default_loggers={"registry": False})
    if runtime.cache_dir is not None:
        raise RuntimeError(
            "stable_pretraining cache_dir must be disabled for controlled "
            f"checkpointing, observed={runtime.cache_dir}"
        )
    return {
        "manager_cache_dir": None,
        "manager_uses_trainer_root": True,
        "registry_logger_disabled": not runtime.default_loggers.get(
            "registry", True
        ),
    }


def _load_distributed_execution_contract(
    benchmark_config: Path, *, devices: int
) -> dict:
    """Load the recovery contract without changing the DDP transport.

    Formal runs deliberately use Lightning/PyTorch's default DDP and NCCL
    selection.  Reproducible recovery is gated on restored training state and
    a measured numerical tolerance, not on benchmark-specific NCCL overrides.
    """

    with benchmark_config.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    declared = (
        config.get("training_protocol", {})
        .get("distributed_execution", {})
    )
    if not isinstance(declared, dict):
        raise ValueError(
            "training_protocol.distributed_execution "
            "must be a mapping"
        )
    transport_keys = {
        str(key) for key in declared if "nccl" in str(key).lower()
    }
    if transport_keys:
        raise ValueError(
            "Door training must not freeze NCCL transport settings; found "
            f"{sorted(transport_keys)}"
        )

    acceptance = declared.get("recovery_acceptance")
    if acceptance is not None:
        expected = {
            "single_gpu_epoch_boundary": {
                "parameter_equivalence": "bitwise",
            },
            "four_gpu_epoch_boundary": {
                "data_order": "exact",
                "rng_state": "exact",
                "global_step": "exact",
                "scheduler_state": "exact",
                "parameter_equivalence": "numerical",
                "maximum_absolute_parameter_difference": 2.0e-9,
                "bytewise_identity_required": False,
            },
        }
        if acceptance != expected:
            raise ValueError(
                "distributed recovery acceptance contract differs from the "
                f"frozen epoch-boundary gate: {acceptance}"
            )

    verification = declared.get("recovery_verification")
    if verification is not None:
        single = verification.get("single_gpu", {})
        multi = verification.get("four_gpu", {})
        maximum = (
            acceptance.get("four_gpu_epoch_boundary", {}).get(
                "maximum_absolute_parameter_difference"
            )
            if acceptance is not None
            else None
        )
        exact_multi_state = all(
            multi.get(key) is True
            for key in (
                "data_order_exact",
                "rng_state_exact",
                "global_step_exact",
                "scheduler_state_exact",
            )
        )
        observed = multi.get(
            "observed_maximum_absolute_parameter_difference"
        )
        if not (
            verification.get("comparison")
            == "continuous_vs_epoch_boundary_restart"
            and single.get("passed") is True
            and single.get("non_bitwise_parameter_tensors") == 0
            and single.get(
                "observed_maximum_absolute_parameter_difference"
            )
            == 0.0
            and single.get("serialized_pretrained_sha256_equal") is True
            and multi.get("passed") is True
            and exact_multi_state
            and isinstance(observed, (int, float))
            and isinstance(maximum, (int, float))
            and 0.0 <= float(observed) <= float(maximum)
            and multi.get("serialized_pretrained_sha256_equal") is False
        ):
            raise ValueError(
                "distributed recovery verification does not satisfy the "
                "frozen epoch-boundary acceptance gate"
            )

    return {
        "devices": devices,
        "runtime_mode": "single_gpu" if devices <= 1 else "multi_gpu",
        "transport_configuration": declared.get(
            "transport_configuration", "framework_defaults"
        ),
        "transport_overrides_applied": False,
        "primary_formal_launch": declared.get("primary_formal_launch"),
        "resume_role": declared.get("resume_role"),
        "resume_scope": declared.get("resume_scope"),
        "recovery_acceptance": acceptance,
        "recovery_verification": verification,
    }


def run(args) -> dict:
    args.output_root = resolve_contextworld_path(
        args.output_root, repo_root=REPO_ROOT
    )
    args.report = resolve_contextworld_path(args.report, repo_root=REPO_ROOT)
    args.benchmark_config = resolve_contextworld_path(
        args.benchmark_config, repo_root=REPO_ROOT
    )
    # Pin the sibling StableWM checkout before importing stable_pretraining,
    # which may otherwise import an unrelated site-packages installation.
    swm, stable_repo, stable_commit = load_stable_worldmodel(
        REPO_ROOT, args.stablewm_repo, args.stablewm_ref
    )

    import hydra
    import lightning as pl
    import stable_pretraining as spt
    import torch
    from lightning.pytorch.callbacks import Callback, ModelCheckpoint
    from lightning.pytorch.trainer.connectors import callback_connector
    from omegaconf import OmegaConf, open_dict
    from stable_worldmodel.wm.loss import SIGReg

    pl.seed_everything(args.seed, workers=True)
    pinned_train = _load_pinned_train_module(stable_repo)
    cfg = _compose_model_config(stable_repo, args)

    grouped = build_tworoom_grouped_data(
        swm,
        repo_root=REPO_ROOT,
        benchmark_config=args.benchmark_config.resolve(),
        model_id=args.model_id,
        epoch_size=args.epoch_size,
        validation_epoch_size=args.validation_epoch_size,
        original_h5=args.original_h5,
        frameskip=5,
        num_steps=4,
        img_size=224,
        seed=args.data_split_seed,
        expected_stablewm_commit=stable_commit,
    )
    training_plan = _build_training_plan(args, grouped.metadata)
    sample_contract = _sample_contract(grouped.train)
    with open_dict(cfg):
        cfg.contextworld_benchmark = {
            "adapter": "ContextWorldGroupedDataModule",
            "benchmark_config": str(args.benchmark_config.resolve()),
            "model_id": args.model_id,
            "profile": args.profile,
            "training_plan": training_plan,
            "data": grouped.metadata,
        }

    if args.preflight_only:
        report = {
            "schema_version": 1,
            "run_kind": "training_data_plan_preflight",
            "passed": True,
            "scope": {
                "data_plan_and_batched_sample_read": True,
                "model_instantiation": False,
                "multi_worker_runtime": False,
                "distributed_runtime": False,
            },
            "model_id": args.model_id,
            "run_name": args.run_name,
            "stable_worldmodel": {
                "repo": str(stable_repo),
                "commit": stable_commit,
                "training_entry": str(stable_repo / "scripts/train/lewm.py"),
            },
            "model_contract": {
                "class": "stable_worldmodel.wm.lewm.lewm.LeWM",
                "action_block": 5,
                "history_size": 3,
            },
            "sample_contract": sample_contract,
            "training_plan": training_plan,
            "data": grouped.metadata,
        }
        write_json(args.report.resolve(), report)
        return report

    spt_runtime = _configure_spt_for_controlled_checkpointing(spt)

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = torch.utils.data.DataLoader(
        grouped.train,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        generator=generator,
        **(
            {"prefetch_factor": int(cfg.loader.prefetch_factor)}
            if args.num_workers > 0
            else {}
        ),
    )
    val_loader = torch.utils.data.DataLoader(
        grouped.val,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        **(
            {"prefetch_factor": int(cfg.loader.prefetch_factor)}
            if args.num_workers > 0
            else {}
        ),
    )

    if args.profile in {"formal", "additive"} and torch.cuda.device_count() < args.devices:
        raise RuntimeError(
            f"{args.profile.title()} training requires {args.devices} visible CUDA devices; "
            f"found {torch.cuda.device_count()}"
        )

    model = hydra.utils.instantiate(cfg.model)
    optimizers = {
        "model_opt": {
            "modules": "model",
            "optimizer": dict(cfg.optimizer),
            # This is the historical M_orig specification.  The current SPT
            # implementation can derive these values from Trainer, but the
            # benchmark records them explicitly and asserts the historical
            # 64/6420 schedule for formal runs.
            "scheduler": {
                "type": "LinearWarmupCosineAnnealingLR",
                "warmup_steps": args.expected_warmup_steps,
                "max_steps": args.expected_optimizer_steps,
                "warmup_start_lr": 0.0,
                "eta_min": 0.0,
            },
            "interval": "step",
            # The pinned SPT module uses manual optimization.  Its frequency
            # gates clipping/step/scheduler/zero_grad, so together with the
            # explicit loss scaling above this is real gradient accumulation.
            "frequency": args.accumulate_grad_batches,
        }
    }
    module = spt.Module(
        model=model,
        sigreg=SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(
            _lejepa_forward_with_manual_accumulation,
            base_forward=pinned_train.lejepa_forward,
            cfg=cfg,
            accumulation_steps=args.accumulate_grad_batches,
        ),
        optim=optimizers,
    )
    # SPT 0.1.6 does not bind the configured nested optimizer name back to
    # optimizer index 0.  Without this, on_train_start creates "default_0"
    # with frequency=1 and silently bypasses the configured accumulation.
    module._optimizer_index_to_name[0] = "model_opt"
    data_module = spt.data.DataModule(train=train_loader, val=val_loader)

    output_root = args.output_root.resolve()
    run_dir = output_root / "checkpoints" / args.run_name
    resume_checkpoint = run_dir / "last.ckpt"
    manager_resume_checkpoint = _validate_resume_policy(
        run_dir=run_dir,
        checkpoint_path=resume_checkpoint,
        policy=args.resume_policy,
    )
    loaded_checkpoint = (
        _full_state_checkpoint_metadata(
            manager_resume_checkpoint,
            expected_optimizer_steps=args.expected_optimizer_steps,
            require_incomplete=True,
            expected_world_size=args.devices,
            optimizer_steps_per_epoch=training_plan[
                "optimizer_steps_per_epoch"
            ],
        )
        if manager_resume_checkpoint is not None
        else None
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    class FullStateRNGCheckpoint(Callback):
        """Persist and restore Python/NumPy/Torch/DataLoader RNG per rank."""

        def __init__(self) -> None:
            self.states: list[dict] | None = None
            self.pending_state: dict | None = None
            self.restored = False
            self.restored_on_fit_start_before_data = False
            self.checkpoint_state_validated_on_load = False
            self.verified_on_train_start = False
            self.loader_generator_advanced_before_train_start = None
            self.loop_state_normalization = None

        @staticmethod
        def _state_for_rank(checkpoint: dict, rank: int) -> dict:
            states = checkpoint.get("contextworld_rng_states_v1")
            if not isinstance(states, list):
                raise RuntimeError(
                    "Resume checkpoint is missing ContextWorld RNG state"
                )
            by_rank = {int(row["rank"]): row for row in states}
            if rank not in by_rank:
                raise RuntimeError(
                    f"Resume checkpoint has no RNG state for rank {rank}"
                )
            return by_rank[rank]

        def _restore(self, trainer, state: dict) -> None:
            import random

            import numpy as np

            random.setstate(state["python"])
            np.random.set_state(state["numpy"])
            torch.set_rng_state(state["torch_cpu"])
            device = trainer.strategy.root_device
            if device.type == "cuda":
                if state["torch_cuda"] is None:
                    raise RuntimeError(
                        "Resume checkpoint has no CUDA RNG state"
                    )
                torch.cuda.set_rng_state(state["torch_cuda"], device=device)
            generator.set_state(state["train_loader_generator"])

        def _observe_loader_generator(self, state: dict) -> None:
            generator_matches = torch.equal(
                generator.get_state().cpu(),
                state["train_loader_generator"].cpu(),
            )
            # Lightning creates the resumed epoch iterator before
            # ``on_train_start``.  Advancing this generator here is expected:
            # the important guarantee is that it advanced from the restored
            # checkpoint state.  End-to-end parameter equivalence is the
            # mechanical test of sampler continuity.
            self.loader_generator_advanced_before_train_start = (
                not generator_matches
            )

        def _capture(self, trainer) -> None:
            import random

            import numpy as np

            device = trainer.strategy.root_device
            local = {
                "rank": int(trainer.global_rank),
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch_cpu": torch.get_rng_state().cpu(),
                "torch_cuda": (
                    torch.cuda.get_rng_state(device).cpu()
                    if device.type == "cuda"
                    else None
                ),
                "train_loader_generator": generator.get_state().cpu(),
            }
            if (
                torch.distributed.is_available()
                and torch.distributed.is_initialized()
            ):
                gathered: list[dict | None] = [None] * int(
                    trainer.world_size
                )
                torch.distributed.all_gather_object(gathered, local)
                if any(row is None for row in gathered):
                    raise RuntimeError("Failed to gather per-rank RNG states")
                self.states = [row for row in gathered if row is not None]
            else:
                self.states = [local]
            ranks = sorted(int(row["rank"]) for row in self.states)
            if ranks != list(range(int(trainer.world_size))):
                raise RuntimeError(
                    f"RNG checkpoint rank coverage mismatch: {ranks}"
                )

        def on_train_epoch_end(self, trainer, pl_module) -> None:
            self._capture(trainer)

        def on_save_checkpoint(
            self, trainer, pl_module, checkpoint
        ) -> None:
            if not self.states:
                raise RuntimeError(
                    "Refusing to save a trainer checkpoint without RNG state"
                )
            checkpoint["contextworld_rng_states_v1"] = self.states

        def on_load_checkpoint(
            self, trainer, pl_module, checkpoint
        ) -> None:
            self.loop_state_normalization = (
                _normalize_complete_epoch_resume_loop_state(
                    checkpoint,
                    optimizer_steps_per_epoch=training_plan[
                        "optimizer_steps_per_epoch"
                    ],
                    accumulation_steps=args.accumulate_grad_batches,
                )
            )
            self.pending_state = self._state_for_rank(
                checkpoint, int(trainer.global_rank)
            )
            self.checkpoint_state_validated_on_load = True

        def on_fit_start(self, trainer, pl_module) -> None:
            if self.pending_state is None:
                return
            # Lightning 2.6 restores callback checkpoint state before
            # ``on_fit_start`` and only creates the FitLoop data iterator
            # afterward.  Restore here: late enough to follow optimizer/model
            # restoration, early enough for the sampler to consume the saved
            # DataLoader generator state.
            self._restore(trainer, self.pending_state)
            self.restored = True
            self.restored_on_fit_start_before_data = True

        def on_train_start(self, trainer, pl_module) -> None:
            if self.pending_state is None:
                return
            if not self.restored_on_fit_start_before_data:
                raise RuntimeError(
                    "Resume RNG was not restored before data setup"
                )
            if not self.checkpoint_state_validated_on_load:
                raise RuntimeError(
                    "Lightning did not validate the preloaded RNG checkpoint"
                )
            self._observe_loader_generator(self.pending_state)
            self.verified_on_train_start = True

    class SavePretrainedAtEpochEnd(Callback):
        def on_train_epoch_end(self, trainer, pl_module) -> None:
            if not trainer.is_global_zero:
                return
            swm.wm.utils.save_pretrained(
                pl_module.model,
                run_name=args.run_name,
                config=cfg,
                filename=f"weights_epoch_{trainer.current_epoch + 1}.pt",
                cache_dir=str(output_root),
            )

    class TrainingContract(Callback):
        def __init__(self) -> None:
            self.microbatches_seen = 0
            self.initial_global_step = 0
            self.initial_epoch = 0

        def on_fit_start(self, trainer, pl_module) -> None:
            if trainer.is_global_zero:
                with (run_dir / "train_config.yaml").open(
                    "w", encoding="utf-8"
                ) as handle:
                    OmegaConf.save(cfg, handle)
            expected_execution_steps = (
                args.stop_after_optimizer_step
                if args.stop_after_optimizer_step is not None
                else args.expected_optimizer_steps
            )
            estimated = int(trainer.estimated_stepping_batches)
            if estimated != expected_execution_steps:
                raise RuntimeError(
                    "Trainer optimizer-step estimate violates the frozen "
                    f"budget: estimated={estimated}, "
                    f"expected={expected_execution_steps}"
                )

        def on_train_start(self, trainer, pl_module) -> None:
            self.initial_global_step = int(trainer.global_step)
            self.initial_epoch = int(trainer.current_epoch)
            if int(trainer.accumulate_grad_batches) != 1:
                raise RuntimeError(
                    "Trainer-level accumulation must remain disabled for "
                    "StablePretraining manual optimization"
                )
            optimizer_objects = pl_module.optimizers()
            optimizer_count = (
                len(optimizer_objects)
                if isinstance(optimizer_objects, (list, tuple))
                else 1
            )
            active_names = [
                pl_module._optimizer_index_to_name[index]
                for index in range(optimizer_count)
            ]
            frequencies = [
                pl_module._optimizer_frequencies[name]
                for name in active_names
            ]
            if active_names != ["model_opt"]:
                raise RuntimeError(
                    f"Manual optimizer index binding mismatch: {active_names}"
                )
            if frequencies != [args.accumulate_grad_batches]:
                raise RuntimeError(
                    "Manual optimizer frequency mismatch: "
                    f"{frequencies} != {[args.accumulate_grad_batches]}"
                )
            configs = list(trainer.lr_scheduler_configs)
            if len(configs) != 1:
                raise RuntimeError(
                    f"Expected one LR scheduler, found {len(configs)}"
                )
            scheduler = configs[0].scheduler
            observed = {
                "warmup_steps": int(scheduler.warmup_steps),
                "max_steps": int(scheduler.max_steps),
            }
            expected = {
                "warmup_steps": args.expected_warmup_steps,
                "max_steps": args.expected_optimizer_steps,
            }
            if observed != expected:
                raise RuntimeError(
                    f"Scheduler contract mismatch: {observed} != {expected}"
                )
            if int(scheduler.last_epoch) != self.initial_global_step:
                raise RuntimeError(
                    "Restored scheduler/global-step mismatch: "
                    f"scheduler={scheduler.last_epoch}, "
                    f"global_step={self.initial_global_step}"
                )

        def on_train_batch_start(
            self, trainer, pl_module, batch, batch_idx
        ) -> None:
            active_name = pl_module._optimizer_index_to_name.get(0)
            active_frequency = pl_module._optimizer_frequencies.get(
                active_name
            )
            if (
                active_name != "model_opt"
                or active_frequency != args.accumulate_grad_batches
            ):
                raise RuntimeError(
                    "Manual accumulation binding changed before a train "
                    f"batch: name={active_name}, frequency={active_frequency}"
                )

        def on_train_batch_end(
            self, trainer, pl_module, outputs, batch, batch_idx
        ) -> None:
            self.microbatches_seen += 1
            expected_steps = (
                self.initial_global_step
                + self.microbatches_seen // args.accumulate_grad_batches
            )
            if int(trainer.global_step) != expected_steps:
                raise RuntimeError(
                    "Optimizer step did not occur at the expected manual "
                    f"accumulation boundary: microbatches={self.microbatches_seen}, "
                    f"global_step={trainer.global_step}, expected={expected_steps}"
                )
            scheduler = trainer.lr_scheduler_configs[0].scheduler
            if int(scheduler.last_epoch) != expected_steps:
                raise RuntimeError(
                    "Scheduler advanced outside the optimizer boundary: "
                    f"last_epoch={scheduler.last_epoch}, "
                    f"expected={expected_steps}"
                )

        def on_train_end(self, trainer, pl_module) -> None:
            target_steps = args.expected_optimizer_steps
            if args.stop_after_optimizer_step is not None:
                target_steps = args.stop_after_optimizer_step
            if int(trainer.global_step) != target_steps:
                raise RuntimeError(
                    "Completed optimizer steps violate the frozen budget: "
                    f"actual={trainer.global_step}, "
                    f"expected={target_steps}"
                )
            expected_microbatches = (
                target_steps - self.initial_global_step
            ) * args.accumulate_grad_batches
            if self.microbatches_seen != expected_microbatches:
                raise RuntimeError(
                    "Manual gradient-accumulation microbatch count mismatch: "
                    f"actual={self.microbatches_seen}, "
                    f"expected={expected_microbatches}"
                )
            scheduler = trainer.lr_scheduler_configs[0].scheduler
            if int(scheduler.last_epoch) != target_steps:
                raise RuntimeError(
                    "Scheduler steps violate the frozen budget: "
                    f"actual={scheduler.last_epoch}, "
                    f"expected={target_steps}"
                )

    rng_checkpoint = FullStateRNGCheckpoint()
    training_contract = TrainingContract()
    if (
        args.stop_after_optimizer_step is not None
        and args.stop_after_optimizer_step
        % training_plan["optimizer_steps_per_epoch"]
        != 0
    ):
        raise ValueError(
            "Controlled resume smoke must stop at a complete epoch boundary: "
            f"stop={args.stop_after_optimizer_step}, "
            "steps_per_epoch="
            f"{training_plan['optimizer_steps_per_epoch']}"
        )
    trainer_max_epochs = args.max_epochs
    if args.stop_after_optimizer_step is not None:
        trainer_max_epochs = (
            args.stop_after_optimizer_step
            // training_plan["optimizer_steps_per_epoch"]
        )
    state_checkpoint = ModelCheckpoint(
        dirpath=str(run_dir),
        filename="state",
        save_top_k=0,
        save_last=True,
        verbose=True,
        enable_version_counter=False,
        every_n_train_steps=None,
        every_n_epochs=1,
        save_on_train_epoch_end=True,
    )
    callbacks = [
        rng_checkpoint,
        SavePretrainedAtEpochEnd(),
        training_contract,
        state_checkpoint,
    ]
    # stable_pretraining registers environment-dump callbacks through a
    # Lightning entry point.  They are unrelated to model training and may
    # write host metadata into the repository, so this benchmark supplies
    # only its explicit callbacks.
    callback_connector._load_external_callbacks = lambda _group: []
    distributed_execution_contract = _load_distributed_execution_contract(
        args.benchmark_config, devices=args.devices
    )
    trainer = pl.Trainer(
        max_epochs=trainer_max_epochs,
        max_steps=args.expected_optimizer_steps,
        accelerator="gpu",
        devices=args.devices,
        precision=args.precision,
        gradient_clip_val=1.0,
        # SPT uses manual optimization, so accumulation is implemented by its
        # optimizer frequency plus the explicit loss scaling above.
        accumulate_grad_batches=1,
        callbacks=callbacks,
        logger=False,
        enable_checkpointing=True,
        enable_model_summary=False,
        default_root_dir=str(run_dir),
        num_sanity_val_steps=0,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        deterministic=True,
    )
    manager = spt.Manager(
        trainer=trainer,
        module=module,
        data=data_module,
        seed=args.seed,
        ckpt_path=manager_resume_checkpoint,
        weights_only=False,
    )
    manager()

    completed_scheduler = trainer.lr_scheduler_configs[0].scheduler

    training_complete = int(trainer.global_step) == args.expected_optimizer_steps
    if args.stop_after_optimizer_step is None and not training_complete:
        raise RuntimeError(
            "Training returned before the frozen optimizer budget without an "
            "intentional smoke stop: "
            f"actual={trainer.global_step}, "
            f"expected={args.expected_optimizer_steps}"
        )
    final_filename = (
        f"weights_final_step_{args.expected_optimizer_steps}.pt"
        if training_complete
        else f"weights_interrupted_step_{int(trainer.global_step)}.pt"
    )
    pretrained_path = run_dir / final_filename
    if trainer.is_global_zero:
        swm.wm.utils.save_pretrained(
            module.model,
            run_name=args.run_name,
            config=cfg,
            filename=final_filename,
            cache_dir=str(output_root),
        )
    trainer.strategy.barrier("contextworld_final_checkpoint")
    if not pretrained_path.is_file() or not (run_dir / "config.json").is_file():
        raise RuntimeError(f"Pretrained save missing under {run_dir}")
    saved_checkpoint = _full_state_checkpoint_metadata(
        resume_checkpoint,
        expected_optimizer_steps=args.expected_optimizer_steps,
        require_incomplete=False,
        expected_world_size=args.devices,
        optimizer_steps_per_epoch=training_plan[
            "optimizer_steps_per_epoch"
        ],
    )
    if saved_checkpoint["global_step"] != int(trainer.global_step):
        raise RuntimeError(
            "Final full-state checkpoint step does not match Trainer: "
            f"checkpoint={saved_checkpoint['global_step']}, "
            f"trainer={trainer.global_step}"
        )
    reloaded = swm.wm.utils.load_pretrained(
        str(pretrained_path), cache_dir=str(output_root)
    )
    state = model.state_dict()
    reloaded_state = reloaded.state_dict()
    reload_equal = state.keys() == reloaded_state.keys() and all(
        torch.equal(state[key].cpu(), reloaded_state[key].cpu()) for key in state
    )
    if not reload_equal:
        raise RuntimeError("Saved pretrained model does not exactly reload")

    report = {
        "schema_version": 1,
        "run_kind": args.run_kind,
        "profile": args.profile,
        "passed": True,
        "model_id": args.model_id,
        "run_name": args.run_name,
        "stable_worldmodel": {
            "repo": str(stable_repo),
            "commit": stable_commit,
            "training_entry": str(
                stable_repo / "scripts/train/lewm.py"
            ),
        },
        "model": {
            "class": f"{type(model).__module__}.{type(model).__name__}",
            "parameters": sum(value.numel() for value in model.parameters()),
            "action_block": 5,
            "history_size": 3,
        },
        "training": {
            "global_step": int(trainer.global_step),
            "current_epoch": int(trainer.current_epoch),
            "max_epochs": args.max_epochs,
            "execution_max_epochs": trainer_max_epochs,
            "seed_before_model_initialization": args.seed,
            "batch_size_per_device": args.batch_size,
            "devices": args.devices,
            "world_size": int(trainer.world_size),
            "strategy": type(trainer.strategy).__name__,
            "ddp_static_graph": False,
            "distributed_execution_contract": (
                distributed_execution_contract
            ),
            "precision": args.precision,
            "num_workers_per_rank": args.num_workers,
            "prefetch_factor": (
                int(cfg.loader.prefetch_factor)
                if args.num_workers > 0
                else None
            ),
            "multi_worker_runtime_exercised": args.num_workers > 0,
            "distributed_runtime_exercised": int(trainer.world_size) > 1,
            "adapter_gradient_accumulation_steps": (
                args.accumulate_grad_batches
            ),
            "trainer_accumulate_grad_batches": int(
                trainer.accumulate_grad_batches
            ),
            "loss_scaling_divisor": args.accumulate_grad_batches,
            "optimizer_frequency": args.accumulate_grad_batches,
            "optimizer_index_binding": {"0": "model_opt"},
            "microbatches_seen_per_rank": (
                training_contract.microbatches_seen
            ),
            "initial_global_step": training_contract.initial_global_step,
            "initial_epoch": training_contract.initial_epoch,
            "resumed_from_checkpoint": (
                training_contract.initial_global_step > 0
            ),
            "loaded_full_state_checkpoint": loaded_checkpoint,
            "restored_global_step": training_contract.initial_global_step,
            "restored_epoch": training_contract.initial_epoch,
            "resume_weights_only": False,
            "rng_state_restored": rng_checkpoint.restored,
            "rng_state_loaded_on_checkpoint": (
                rng_checkpoint.checkpoint_state_validated_on_load
            ),
            "rng_state_restored_on_fit_start_before_data": (
                rng_checkpoint.restored_on_fit_start_before_data
            ),
            "rng_state_verified_on_train_start": (
                rng_checkpoint.verified_on_train_start
            ),
            "loader_generator_advanced_before_train_start": (
                rng_checkpoint.loader_generator_advanced_before_train_start
            ),
            "resume_scope": "complete_epoch_boundary_only",
            "resume_loop_state_normalization": (
                rng_checkpoint.loop_state_normalization
            ),
            "resume_policy": args.resume_policy,
            "training_complete": training_complete,
            "intentional_stop_after_optimizer_step": (
                args.stop_after_optimizer_step
            ),
            "limit_train_batches": args.limit_train_batches,
            "limit_val_batches": args.limit_val_batches,
            "external_callbacks_disabled": True,
            "expected_optimizer_steps": args.expected_optimizer_steps,
            "expected_warmup_steps": args.expected_warmup_steps,
            "scheduler_last_epoch": int(completed_scheduler.last_epoch),
            "scheduler_final_lrs": [
                float(value) for value in completed_scheduler.get_last_lr()
            ],
            "plan": training_plan,
            "stable_pretraining_runtime": spt_runtime,
        },
        "sample_contract": sample_contract,
        "data": grouped.metadata,
        "artifacts": {
            "run_dir": str(run_dir),
            "pretrained": str(pretrained_path),
            "pretrained_sha256": _sha256(pretrained_path),
            "pretrained_config": str(run_dir / "config.json"),
            "pretrained_config_sha256": _sha256(run_dir / "config.json"),
            "stablewm_resume_checkpoint": str(resume_checkpoint),
            "stablewm_resume_checkpoint_exists": (
                resume_checkpoint.is_file()
            ),
            "full_state_checkpoint": saved_checkpoint,
        },
        "save_load_exact": reload_equal,
    }
    if trainer.is_global_zero:
        write_json(args.report.resolve(), report)
    return report


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a TwoRoom Step-1 model with pinned StableWM LeWM"
    )
    parser.add_argument(
        "--model-id",
        required=True,
        help="Model identifier declared by the selected benchmark config.",
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--profile", choices=tuple(PROFILE_DEFAULTS), default="smoke")
    parser.add_argument("--run-kind", choices=("adapter_smoke", "pilot", "confirmation"), default=None)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=artifact_path("training/runs", repo_root=REPO_ROOT),
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--benchmark-config", type=Path, default=REPO_ROOT / "configs/benchmark/tworoom_step1_v1.yaml")
    parser.add_argument(
        "--original-h5",
        type=Path,
        default=None,
        help=(
            "Optional local copy of quentinll/lewm-tworooms/tworoom.h5. "
            "Overrides the machine-layout fallback in the benchmark config."
        ),
    )
    parser.add_argument("--stablewm-repo", default="../stable-worldmodel")
    parser.add_argument("--stablewm-ref", default=PINNED_STABLEWM)
    parser.add_argument(
        "--resume-policy",
        choices=("auto", "never", "required"),
        default="auto",
        help=(
            "Launch policy around StableWM's native checkpoint: auto resumes "
            "when present, never requires a fresh run, required refuses to "
            "start without it."
        ),
    )
    parser.add_argument("--seed", type=int, default=3072)
    parser.add_argument("--data-split-seed", type=int, default=3072)
    parser.add_argument("--epoch-size", type=int, default=None)
    parser.add_argument("--validation-epoch-size", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--devices", type=int, default=None)
    parser.add_argument("--precision", default=None)
    parser.add_argument("--accumulate-grad-batches", type=int, default=None)
    parser.add_argument("--limit-train-batches", type=_parse_batch_limit, default=None)
    parser.add_argument("--limit-val-batches", type=_parse_batch_limit, default=None)
    parser.add_argument("--expected-optimizer-steps", type=int, default=None)
    parser.add_argument(
        "--stop-after-optimizer-step",
        type=int,
        default=None,
        help=(
            "Smoke-only controlled stop on an optimizer boundary, used to "
            "verify native resume."
        ),
    )
    parser.add_argument("--preflight-only", action="store_true")
    args = _apply_profile(parser.parse_args())
    if args.stop_after_optimizer_step is not None:
        if args.profile != "smoke":
            parser.error(
                "--stop-after-optimizer-step is restricted to smoke runs"
            )
        if not (
            0
            < args.stop_after_optimizer_step
            < args.expected_optimizer_steps
        ):
            parser.error(
                "--stop-after-optimizer-step must be between 1 and "
                "expected-optimizer-steps - 1"
            )
    return args


if __name__ == "__main__":
    result = run(parse_args())
    summary = {"passed": result["passed"], "run_kind": result["run_kind"]}
    if "training" in result:
        summary.update(
            {
                "global_step": result["training"]["global_step"],
                "pretrained": result["artifacts"]["pretrained"],
            }
        )
    else:
        summary["training_plan"] = result["training_plan"]
    if _process_is_global_zero():
        print(json.dumps(summary))
