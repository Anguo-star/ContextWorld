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
    from lightning.pytorch.callbacks import Callback
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
    resume_checkpoint = pinned_train.get_resume_checkpoint_path(
        run_dir, args.run_name
    )
    manager_resume_checkpoint = _validate_resume_policy(
        run_dir=run_dir,
        checkpoint_path=resume_checkpoint,
        policy=args.resume_policy,
    )
    run_dir.mkdir(parents=True, exist_ok=True)

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
            estimated = int(trainer.estimated_stepping_batches)
            if estimated != args.expected_optimizer_steps:
                raise RuntimeError(
                    "Trainer optimizer-step estimate violates the frozen "
                    f"budget: estimated={estimated}, "
                    f"expected={args.expected_optimizer_steps}"
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

    class StopAfterOptimizerStepForResumeSmoke(Callback):
        def on_train_batch_end(
            self, trainer, pl_module, outputs, batch, batch_idx
        ) -> None:
            if (
                args.stop_after_optimizer_step is not None
                and int(trainer.global_step)
                >= args.stop_after_optimizer_step
            ):
                trainer.should_stop = True

    training_contract = TrainingContract()
    callbacks = [
        SavePretrainedAtEpochEnd(),
        training_contract,
        StopAfterOptimizerStepForResumeSmoke(),
    ]
    # stable_pretraining registers environment-dump callbacks through a
    # Lightning entry point.  They are unrelated to model training and may
    # write host metadata into the repository, so this benchmark supplies
    # only its explicit callbacks.
    callback_connector._load_external_callbacks = lambda _group: []
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
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
    )
    manager()

    completed_scheduler = trainer.lr_scheduler_configs[0].scheduler

    training_complete = int(trainer.global_step) == args.expected_optimizer_steps
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
            "seed_before_model_initialization": args.seed,
            "batch_size_per_device": args.batch_size,
            "devices": args.devices,
            "world_size": int(trainer.world_size),
            "strategy": type(trainer.strategy).__name__,
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
