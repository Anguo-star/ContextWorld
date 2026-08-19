#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import multiprocessing
import os
import sys
from datetime import timedelta
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from contextworld.synthesis.manifest import write_json
from contextworld.synthesis.stablewm import load_stable_worldmodel
from contextworld.paths import artifact_path, resolve_contextworld_path
from contextworld.evaluation.hidden_passage_h3_data import (
    AUDIT_SCHEDULING_LOCK_PROTOCOL,
    PARALLEL_AUDIT_SCHEDULING_LOCK_PROTOCOL,
    TRAINING_RUN_LOCK_PROTOCOL,
    hidden_passage_audit_scheduling_lock,
    hidden_passage_release_lock,
    hidden_passage_training_run_lock,
    verify_hidden_passage_training_run_parent,
)
from contextworld.training.tworoom_data import (
    build_tworoom_grouped_data,
    hidden_passage_training_release_root,
    revalidate_hidden_passage_training_storage,
)


PINNED_STABLEWM = "5864b74980f6ed328fd0045e777b3865962eff43"
PASSAGE_INTERNAL_ENVIRONMENT = (
    "CONTEXTWORLD_H3_RANK0_ATTESTATION_V1",
    "CONTEXTWORLD_H3_RANK0_ATTESTATION_V2",
    "CONTEXTWORLD_H3_RANK0_SECRET",
    "CONTEXTWORLD_H3_RANK0_ISSUER",
)
PASSAGE_DDP_RENDEZVOUS_TIMEOUT_SECONDS = 7200
TRAINING_RUN_EXCLUSIVITY_CONTRACT = {
    "protocol": TRAINING_RUN_LOCK_PROTOCOL,
    "policy": "one_root_training_run_per_release",
    "maximum_concurrency": 1,
    "blocking": False,
    "scope": "full_root_training_or_preflight_call",
    "held_through_report_snapshot": True,
    "nonzero_rank_admission": "direct_parent_holds_root_training_lock",
}
SERIAL_AUDIT_SCHEDULING_CONTRACT = {
    "policy": "sibling_exclusive_flock",
    "maximum_concurrency": 1,
    "scope": "per_rank_full_audit_and_fit_start_storage_revalidation",
    "lock_protocol": AUDIT_SCHEDULING_LOCK_PROTOCOL,
    "lock_order": "release_shared_then_audit_exclusive",
    "collective_holds_lock": False,
    "topology_scope": "single_node_8gpu",
    "concurrent_training_runs_per_release": 1,
}
PARALLEL_AUDIT_SCHEDULING_CONTRACT = {
    "policy": "sibling_shared_flock",
    "maximum_concurrency": 8,
    "scope": "per_rank_full_audit_and_fit_start_storage_revalidation",
    "lock_protocol": PARALLEL_AUDIT_SCHEDULING_LOCK_PROTOCOL,
    "lock_order": "release_shared_then_audit_shared",
    "collective_holds_lock": False,
    "topology_scope": "single_node_8gpu",
    "concurrent_training_runs_per_release": 1,
}
PARALLEL_RANK_CPU_AFFINITY_CONTRACT = {
    "policy": "local_rank_disjoint_contiguous_from_zero",
    "cpus_per_rank": 8,
    "expected_world_size": 8,
    "scope": "full_rank_process",
    "apply_before_stableworldmodel_and_lance_import": True,
}
AUDIT_SCHEDULING_CONTRACTS = (
    SERIAL_AUDIT_SCHEDULING_CONTRACT,
    PARALLEL_AUDIT_SCHEDULING_CONTRACT,
)

FORMAL_TOPOLOGIES = {
    (4, 128, 2): "4gpu_x_b128_x_accum2",
    (8, 128, 1): "8gpu_x_b128_x_accum1",
}
CONTROLLED_PROFILES = {
    "formal",
    "additive",
    "icl_formal",
    "icl_core_v3",
    "passage_pilot",
    "passage_formal",
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
    "icl_formal": {
        "run_kind": "confirmation",
        "epoch_size": 262_144,
        "validation_epoch_size": 8_192,
        "max_epochs": 4,
        "batch_size": 128,
        "num_workers": 6,
        "devices": 8,
        "precision": "bf16-mixed",
        "accumulate_grad_batches": 1,
        "limit_train_batches": 1.0,
        "limit_val_batches": 1.0,
        "expected_optimizer_steps": 1_024,
    },
    "icl_core_v3": {
        "run_kind": "confirmation",
        "epoch_size": 524_288,
        "validation_epoch_size": 8_192,
        "max_epochs": 2,
        "batch_size": 128,
        "num_workers": 6,
        "devices": 8,
        "precision": "bf16-mixed",
        "accumulate_grad_batches": 1,
        "limit_train_batches": 1.0,
        "limit_val_batches": 1.0,
        "expected_optimizer_steps": 1_024,
    },
    "passage_pilot": {
        "run_kind": "pilot",
        "epoch_size": 65_536,
        "validation_epoch_size": 4_096,
        "max_epochs": 4,
        "batch_size": 128,
        "num_workers": 6,
        "devices": 8,
        "precision": "bf16-mixed",
        "accumulate_grad_batches": 1,
        "limit_train_batches": 1.0,
        "limit_val_batches": 1.0,
        "expected_optimizer_steps": 256,
    },
    "passage_formal": {
        "run_kind": "confirmation",
        "epoch_size": 262_144,
        "validation_epoch_size": 8_192,
        "max_epochs": 4,
        "batch_size": 128,
        "num_workers": 6,
        "devices": 8,
        "precision": "bf16-mixed",
        "accumulate_grad_batches": 1,
        "limit_train_batches": 1.0,
        "limit_val_batches": 1.0,
        "expected_optimizer_steps": 1_024,
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
    if args.profile in CONTROLLED_PROFILES:
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
                f"{args.profile.title()} profile requires run-kind pilot "
                "or confirmation"
            )
        topology = (
            args.devices,
            args.batch_size,
            args.accumulate_grad_batches,
        )
        if topology not in FORMAL_TOPOLOGIES:
            raise ValueError(
                f"{args.profile.title()} profile requires one of the "
                "validated execution "
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
        required_hash_names = {
            "catalog",
            "manifest",
            "synthesis_report",
        }
        frozen_artifact_hashes = set(
            data_metadata["groups"][name]
            .get("catalog_split_audit", {})
            .get("required_artifact_hashes", {})
        )
        artifact_hash_freeze_required = (
            args.profile == "passage_formal"
            and name.startswith("passage_")
        )
        gates = {
            "static": all(
                data_metadata["groups"][name]
                .get("static_quality_gates", {"unconfigured": True})
                .values()
            ),
            "formal_reuse": (
                args.profile not in CONTROLLED_PROFILES
                or maximum_reuse is None
                or exposure["logical_budget_mean_draws_per_raw_clip"]
                <= float(maximum_reuse)
            ),
            "formal_artifact_hashes_frozen": (
                not artifact_hash_freeze_required
                or frozen_artifact_hashes == required_hash_names
            ),
        }
        data_quality_gates[name] = {
            "passed": all(gates.values()),
            "requirements": requirements,
            "observed_mean_draws_per_raw_clip": exposure[
                "logical_budget_mean_draws_per_raw_clip"
            ],
            "required_artifact_hash_names": (
                sorted(required_hash_names)
                if artifact_hash_freeze_required
                else []
            ),
            "frozen_artifact_hash_names": sorted(
                frozen_artifact_hashes
            ),
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
    formal_build = dict(
        data_metadata.get(
            "formal_build_report_audit",
            {"required": False, "passed": True},
        )
    )
    formal_build_report_gate = {
        "required": bool(formal_build.get("required", False)),
        "path": formal_build.get("path"),
        "sha256": formal_build.get("sha256"),
        "passed": formal_build.get("passed") is True,
    }
    if (
        formal_build_report_gate["required"]
        and not formal_build_report_gate["passed"]
    ):
        raise ValueError(
            "Formal hidden-passage build_report gate failed: "
            f"{formal_build_report_gate}"
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
        "formal_build_report_gate": formal_build_report_gate,
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


def _reject_internal_passage_environment() -> None:
    inherited = sorted(
        name for name in PASSAGE_INTERNAL_ENVIRONMENT if name in os.environ
    )
    if inherited:
        for name in inherited:
            os.environ.pop(name, None)
        raise RuntimeError(
            "Internal hidden-passage launch state may not cross a shell or "
            f"pipeline boundary: {inherited}"
        )


def _apply_passage_rank_cpu_affinity(
    *,
    contract: dict | None,
    local_rank: int,
    devices: int,
) -> dict | None:
    """Bound Lance's per-process pools before StableWM imports Lance."""

    if contract is None:
        return None
    if contract != PARALLEL_RANK_CPU_AFFINITY_CONTRACT:
        raise ValueError(
            "Passage rank CPU affinity differs from the frozen parallel "
            f"audit contract: observed={contract}"
        )
    if not hasattr(os, "sched_setaffinity") or not hasattr(
        os, "sched_getaffinity"
    ):
        raise RuntimeError(
            "Parallel passage audits require Linux CPU affinity"
        )
    expected_world_size = int(contract["expected_world_size"])
    cpus_per_rank = int(contract["cpus_per_rank"])
    if int(devices) != expected_world_size:
        raise RuntimeError(
            "Passage CPU affinity is frozen to the 8-rank topology: "
            f"devices={devices}, expected={expected_world_size}"
        )
    if not 0 <= int(local_rank) < expected_world_size:
        raise RuntimeError(
            f"LOCAL_RANK is outside CPU-affinity topology: {local_rank}"
        )
    cpu_count = os.cpu_count()
    required_cpus = expected_world_size * cpus_per_rank
    if not isinstance(cpu_count, int) or cpu_count < required_cpus:
        raise RuntimeError(
            "Host has too few logical CPUs for disjoint passage rank "
            f"affinity: observed={cpu_count}, required={required_cpus}"
        )
    start = int(local_rank) * cpus_per_rank
    target = tuple(range(start, start + cpus_per_rank))
    before = tuple(sorted(os.sched_getaffinity(0)))
    os.sched_setaffinity(0, set(target))
    after = tuple(sorted(os.sched_getaffinity(0)))
    if after != target:
        raise RuntimeError(
            "Passage rank CPU affinity was not applied exactly: "
            f"rank={local_rank}, expected={target}, observed={after}"
        )
    return {
        "policy": contract["policy"],
        "scope": contract["scope"],
        "applied_before_stableworldmodel_and_lance_import": True,
        "local_rank": int(local_rank),
        "cpus_per_rank": cpus_per_rank,
        "cpu_ids": list(target),
        "host_logical_cpu_count": cpu_count,
        "prior_affinity_cpu_count": len(before),
        "passed": True,
    }


class PassageReleaseGatedDataset:
    """A process-shared hard gate at the real Dataset read boundary."""

    def __init__(self, dataset, *, split: str) -> None:
        self.dataset = dataset
        self.split = split
        self._released = multiprocessing.Event()
        self._pre_release_calls = multiprocessing.Value("q", 0)
        self._pre_release_items = multiprocessing.Value("q", 0)
        self._post_release_calls = multiprocessing.Value("q", 0)
        self._post_release_items = multiprocessing.Value("q", 0)

    def __len__(self) -> int:
        return len(self.dataset)

    @staticmethod
    def _increment(counter, amount: int) -> None:
        with counter.get_lock():
            counter.value += int(amount)

    def _authorize(self, item_count: int) -> None:
        if not self._released.is_set():
            self._increment(self._pre_release_calls, 1)
            self._increment(self._pre_release_items, item_count)
            raise RuntimeError(
                f"{self.split} Dataset read attempted before audit consensus"
            )
        self._increment(self._post_release_calls, 1)
        self._increment(self._post_release_items, item_count)

    def __getitem__(self, index):
        self._authorize(1)
        return self.dataset[index]

    def __getitems__(self, indices):
        indices = list(indices)
        self._authorize(len(indices))
        batched = getattr(self.dataset, "__getitems__", None)
        if batched is not None:
            return batched(indices)
        return [self.dataset[index] for index in indices]

    def release(self) -> None:
        self._released.set()

    @property
    def released(self) -> bool:
        return self._released.is_set()

    def receipt(self) -> dict:
        return {
            "split": self.split,
            "released": self.released,
            "pre_release_calls": int(self._pre_release_calls.value),
            "pre_release_items": int(self._pre_release_items.value),
            "post_release_calls": int(self._post_release_calls.value),
            "post_release_items": int(self._post_release_items.value),
        }


PASSAGE_RECEIPT_FIELDS = (
    "rank",
    "passed",
    "full_logical_audit_count",
    "storage_revalidation_count",
    "train_pre_release_calls",
    "train_pre_release_items",
    "val_pre_release_calls",
    "val_pre_release_items",
    "internal_environment_clean",
    "full_audit_lock_acquired",
    "full_audit_lock_released",
    "full_audit_release_shared_held",
    "full_audit_collective_unlocked",
    "full_audit_path_identity_verified",
    "full_audit_path_identity_verified_after_acquire",
    "full_audit_descriptor_noninheritable",
    "full_audit_fork_child_close_registered",
    "full_audit_wait_milliseconds",
    "full_audit_hold_milliseconds",
    "full_audit_sample_contract_reads",
    "revalidation_lock_acquired",
    "revalidation_lock_released",
    "revalidation_release_shared_held",
    "revalidation_collective_unlocked",
    "revalidation_path_identity_verified",
    "revalidation_path_identity_verified_after_acquire",
    "revalidation_descriptor_noninheritable",
    "revalidation_fork_child_close_registered",
    "revalidation_wait_milliseconds",
    "revalidation_hold_milliseconds",
)


def _lock_seconds_to_milliseconds(receipt: dict, field: str) -> int:
    value = receipt.get(field)
    if not isinstance(value, (int, float)) or float(value) < 0:
        return -1
    return int(round(float(value) * 1000))


def _training_run_exclusivity_snapshot(args) -> dict:
    receipt = dict(getattr(args, "_passage_training_run_lock", {}))
    if receipt.get("acquired") is True:
        receipt["held_through_report_snapshot"] = bool(
            receipt.get("released") is False
            and receipt.get("hold_seconds") is None
        )
    return receipt


def _distributed_passage_full_audit_consensus(
    *,
    strategy,
    torch_module,
    device,
    local_receipt: dict,
    expected_world_size: int,
) -> list[dict]:
    """Gather every rank's independent audit before opening either gate."""

    local = torch_module.tensor(
        [int(local_receipt[name]) for name in PASSAGE_RECEIPT_FIELDS],
        dtype=torch_module.int64,
        device=device,
    )
    gathered = strategy.all_gather(local)
    rows = (
        gathered.detach()
        .cpu()
        .reshape(-1, len(PASSAGE_RECEIPT_FIELDS))
        .tolist()
    )
    receipts = [
        {
            name: int(value)
            for name, value in zip(PASSAGE_RECEIPT_FIELDS, row)
        }
        for row in rows
    ]
    ranks = sorted(row["rank"] for row in receipts)
    expected_ranks = list(range(int(expected_world_size)))
    all_passed = (
        ranks == expected_ranks
        and len(receipts) == int(expected_world_size)
        and all(
            row["passed"] == 1
            and row["full_logical_audit_count"] == 1
            and row["storage_revalidation_count"] == 1
            and row["train_pre_release_calls"] == 0
            and row["train_pre_release_items"] == 0
            and row["val_pre_release_calls"] == 0
            and row["val_pre_release_items"] == 0
            and row["internal_environment_clean"] == 1
            and row["full_audit_lock_acquired"] == 1
            and row["full_audit_lock_released"] == 1
            and row["full_audit_release_shared_held"] == 1
            and row["full_audit_collective_unlocked"] == 1
            and row["full_audit_path_identity_verified"] == 1
            and row[
                "full_audit_path_identity_verified_after_acquire"
            ]
            == 1
            and row["full_audit_descriptor_noninheritable"] == 1
            and row["full_audit_fork_child_close_registered"] == 1
            and row["full_audit_wait_milliseconds"] >= 0
            and row["full_audit_hold_milliseconds"] >= 0
            and row["full_audit_sample_contract_reads"] == 8
            and row["revalidation_lock_acquired"] == 1
            and row["revalidation_lock_released"] == 1
            and row["revalidation_release_shared_held"] == 1
            and row["revalidation_collective_unlocked"] == 1
            and row["revalidation_path_identity_verified"] == 1
            and row[
                "revalidation_path_identity_verified_after_acquire"
            ]
            == 1
            and row["revalidation_descriptor_noninheritable"] == 1
            and row["revalidation_fork_child_close_registered"] == 1
            and row["revalidation_wait_milliseconds"] >= 0
            and row["revalidation_hold_milliseconds"] >= 0
            for row in receipts
        )
    )
    if not all_passed:
        raise RuntimeError(
            "Hidden-passage per-rank audit consensus failed before Dataset "
            f"release: receipts={receipts}"
        )
    return sorted(receipts, key=lambda row: row["rank"])


def _project_lewm_model_batch(
    batch,
    *,
    sequence_steps: int = 4,
):
    """Fail closed at the LeWM boundary: only images and actions are visible."""

    import torch

    sequence_steps = int(sequence_steps)
    if sequence_steps < 2:
        raise ValueError("LeWM sequence_steps must be at least 2")
    required = ("pixels", "action")
    missing = [key for key in required if key not in batch]
    if missing:
        raise KeyError(f"LeWM batch is missing required fields: {missing}")
    visible = {key: batch[key] for key in required}
    if not all(torch.is_tensor(value) for value in visible.values()):
        raise TypeError("LeWM pixels and action inputs must be tensors")
    pixels = visible["pixels"]
    actions = visible["action"]
    if (
        pixels.ndim != 5
        or pixels.shape[1] != sequence_steps
        or pixels.shape[2] != 3
    ):
        raise ValueError(
            "LeWM pixels must have shape "
            f"[batch,{sequence_steps},3,height,width], got "
            f"{tuple(pixels.shape)}"
        )
    if actions.ndim != 3 or tuple(actions.shape[1:]) != (
        sequence_steps,
        10,
    ):
        raise ValueError(
            "LeWM actions must have shape "
            f"[batch,{sequence_steps},10], got "
            f"{tuple(actions.shape)}"
        )
    return visible


def _lejepa_forward_with_manual_accumulation(
    self,
    batch,
    stage,
    *,
    base_forward,
    cfg,
    accumulation_steps: int,
    training_method: str | None = None,
    temporal_prediction_loss: dict | None = None,
):
    """Preserve per-rank batch statistics while accumulating exact gradients."""

    sequence_steps = int(batch["pixels"].shape[1])
    if hasattr(cfg, "wm"):
        configured_steps = (
            int(cfg.wm.history_size) + int(cfg.wm.num_preds)
        )
        if sequence_steps != configured_steps:
            raise ValueError(
                "Batch length differs from the configured temporal contract: "
                f"{sequence_steps} != {configured_steps}"
            )
    projected = _project_lewm_model_batch(
        batch,
        sequence_steps=sequence_steps,
    )
    if temporal_prediction_loss and temporal_prediction_loss["configured"]:
        if training_method not in {"lewm", "pldm"}:
            raise ValueError(
                "Temporal prediction weighting requires an explicit "
                "LeWM or PLDM training method"
            )
        state = _temporal_weighted_forward(
            self,
            projected,
            stage,
            cfg,
            training_method=training_method,
            specification=temporal_prediction_loss,
        )
    else:
        state = base_forward(self, projected, stage, cfg)
    if stage == "fit" and accumulation_steps > 1:
        # StablePretraining's manual optimizer frequency delays optimizer.step,
        # but its stock training_step does not call rescale_loss_for_grad_acc.
        # Divide here after the upstream forward has logged the unscaled losses.
        state["loss"] = state["loss"] / accumulation_steps
    return state


def _weighted_transition_mse(prediction, target, weights):
    """Average every transition internally, then apply normalized time weights."""

    import torch

    if prediction.shape != target.shape:
        raise ValueError(
            "Prediction and target shapes differ: "
            f"{tuple(prediction.shape)} != {tuple(target.shape)}"
        )
    if prediction.ndim < 3:
        raise ValueError(
            "Temporal prediction tensors must have batch, time, and feature "
            f"dimensions, got {tuple(prediction.shape)}"
        )
    value = torch.as_tensor(
        weights,
        dtype=prediction.dtype,
        device=prediction.device,
    )
    if value.ndim != 1 or value.numel() != prediction.shape[1]:
        raise ValueError(
            "Temporal weights must match the predicted transition count: "
            f"{tuple(value.shape)} versus {prediction.shape[1]}"
        )
    if not bool(torch.isfinite(value).all()) or not bool((value > 0).all()):
        raise ValueError("Temporal prediction weights must be finite and positive")
    per_transition = (
        (prediction - target)
        .square()
        .flatten(start_dim=2)
        .mean(dim=2)
    )
    weighted = (
        (per_transition * value.unsqueeze(0)).sum(dim=1) / value.sum()
    ).mean()
    return {
        "weighted": weighted,
        "unweighted": per_transition.mean(),
        "final_transition": per_transition[:, -1].mean(),
        "per_transition": per_transition.mean(dim=0),
    }


def _temporal_weighted_forward(
    self,
    batch,
    stage,
    cfg,
    *,
    training_method: str,
    specification: dict,
):
    """Native StableWM forward with only prediction-time reduction changed."""

    import torch

    batch["action"] = torch.nan_to_num(batch["action"], 0.0)
    output = self.model.encode(batch)
    emb = output["emb"]
    act_emb = output["act_emb"]
    history = int(cfg.wm.history_size)
    num_preds = int(cfg.wm.num_preds)
    target = emb[:, num_preds:]
    prediction = self.model.predict(
        emb[:, :history],
        act_emb[:, :history],
    )
    prediction_losses = _weighted_transition_mse(
        prediction,
        target,
        specification["transition_weights"],
    )
    output["pred_loss"] = prediction_losses["weighted"]
    output["pred_loss_unweighted"] = prediction_losses["unweighted"]
    output["pred_loss_final_transition"] = prediction_losses[
        "final_transition"
    ]

    if training_method == "lewm":
        regularizer_name = str(
            cfg.loss.get("regularizer", "sigreg")
        ).lower()
        if regularizer_name == "conditional_sigreg":
            raise ValueError(
                "The paired Action Delay repair uses the native unconditional "
                "LeWM representation regularizer"
            )
        regularizer_cfg = cfg.loss.get(regularizer_name)
        regularizer_loss_key = f"{regularizer_name}_loss"
        output[regularizer_loss_key] = getattr(
            self, regularizer_name
        )(emb.transpose(0, 1))
        output["loss"] = (
            output["pred_loss"]
            + regularizer_cfg.weight * output[regularizer_loss_key]
        )
        active_vcreg_names = [
            name
            for name in ("std", "std_t", "cov", "cov_t")
            if cfg.loss.get(name) is not None
            and cfg.loss.get(name).enabled
        ]
        if active_vcreg_names:
            output.update(self.vc_reg(emb))
        for name in active_vcreg_names:
            output["loss"] = (
                output["loss"]
                + cfg.loss.get(name).weight * output[f"{name}_loss"]
            )
    elif training_method == "pldm":
        output["idm_emb"] = torch.cat(
            [emb[:, 1:], emb[:, :-1]],
            dim=-1,
        )
        output["act_label"] = batch["action"][:, :-1].detach()
        output["act_pred"] = self.idm(output["idm_emb"])
        output["temp_straight_loss"] = self.path_straight(emb)
        output.update(
            self.pldm(emb, output["act_pred"], output["act_label"])
        )
        output["loss"] = output["pred_loss"]
        for name, loss_cfg in cfg.loss.items():
            loss_key = f"{name}_loss"
            if not loss_cfg.enabled or loss_key not in output:
                continue
            output["loss"] = (
                output["loss"] + loss_cfg.weight * output[loss_key]
            )
    else:
        raise ValueError(f"Unsupported training method {training_method!r}")

    losses = {
        f"{stage}/{name}": value.detach()
        for name, value in output.items()
        if "loss" in name
    }
    self.log_dict(losses, on_step=True, sync_dist=True)
    return output


def _sample_contract(
    dataset,
    count: int = 8,
    *,
    history_tokens: int = 3,
    num_preds: int = 1,
) -> dict:
    import torch

    history_tokens = int(history_tokens)
    num_preds = int(num_preds)
    sequence_steps = history_tokens + num_preds
    if history_tokens < 1 or num_preds < 1:
        raise ValueError("history_tokens and num_preds must be positive")
    indices = list(range(min(count, len(dataset))))
    samples = dataset.__getitems__(indices)
    expected = {
        "pixels": [sequence_steps, 3, 224, 224],
        "action": [sequence_steps, 10],
        "proprio": [sequence_steps, 2],
    }
    observed = {
        key: list(samples[0][key].shape) for key in sorted(expected)
    }
    if observed != expected:
        raise RuntimeError(
            "Training sample sequence contract mismatch: "
            f"history={history_tokens}, num_preds={num_preds}, "
            f"observed={observed}"
        )
    for sample in samples[1:]:
        shapes = {key: list(sample[key].shape) for key in sorted(expected)}
        if shapes != expected:
            raise RuntimeError(f"Cross-group sample shape mismatch: {shapes}")
    collated = torch.utils.data.default_collate(samples)
    projected = _project_lewm_model_batch(
        collated,
        sequence_steps=sequence_steps,
    )
    raw_keys = sorted(collated)
    model_boundary_keys = list(projected)
    privileged = {
        "passage.open",
        "passage_open",
        "variation_passage_open",
        "state",
        "proprio",
        "template_id",
        "pair_id",
    }
    leaked = sorted(privileged & set(model_boundary_keys))
    if leaked:
        raise RuntimeError(
            f"Privileged fields reached the LeWM boundary: {leaked}"
        )
    return {
        "passed": True,
        "history_tokens": history_tokens,
        "num_preds": num_preds,
        "sequence_steps": sequence_steps,
        "sample_count": len(samples),
        "shapes": observed,
        "keys": sorted(samples[0]),
        "batched_dataset_read_exercised": True,
        "collated_batch_audit": {
            "raw_keys": raw_keys,
            "raw_shapes": {
                key: list(value.shape)
                for key, value in collated.items()
                if torch.is_tensor(value)
            },
            "model_boundary_keys": model_boundary_keys,
            "model_boundary_shapes": {
                key: list(value.shape)
                for key, value in projected.items()
            },
            "privileged_fields": sorted(privileged),
            "privileged_fields_at_model_boundary": leaked,
            "strict_pixels_action_projection": (
                model_boundary_keys == ["pixels", "action"]
            ),
            "passed": (
                model_boundary_keys == ["pixels", "action"]
                and not leaked
            ),
        },
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _state_dict_sha256(model) -> str:
    import torch

    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(
            tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        )
    return digest.hexdigest()


def _frozen_module_spec(benchmark_config: Path) -> dict:
    """Load an optional, explicit representation-freeze diagnostic."""

    with benchmark_config.open("r", encoding="utf-8") as handle:
        benchmark = yaml.safe_load(handle)
    declared = benchmark.get("training_protocol", {}).get(
        "frozen_model_modules",
        [],
    )
    if declared in (None, []):
        return {
            "configured": False,
            "modules": [],
            "force_eval_mode": False,
        }
    if not isinstance(declared, list) or not all(
        isinstance(value, str) and value for value in declared
    ):
        raise ValueError(
            "training_protocol.frozen_model_modules must be a list of names"
        )
    modules = list(declared)
    if modules != ["encoder", "projector"]:
        raise ValueError(
            "The only audited representation freeze is exactly "
            "['encoder', 'projector']"
        )
    force_eval_mode = benchmark.get("training_protocol", {}).get(
        "force_frozen_modules_eval_mode"
    )
    if force_eval_mode is not True:
        raise ValueError(
            "Frozen encoder/projector requires "
            "force_frozen_modules_eval_mode: true"
        )
    return {
        "configured": True,
        "modules": modules,
        "force_eval_mode": True,
        "purpose": (
            "keep the original target-latent representation fixed while "
            "testing whether the predictor can bind History=3 to the rule"
        ),
    }


def _training_method(benchmark_config: Path) -> str:
    """Return the explicitly selected StableWM training objective."""

    with benchmark_config.open("r", encoding="utf-8") as handle:
        benchmark = yaml.safe_load(handle)
    method = str(
        benchmark.get("training_protocol", {}).get(
            "training_method",
            "lewm",
        )
    ).lower()
    if method not in {"lewm", "pldm"}:
        raise ValueError(
            "training_protocol.training_method must be 'lewm' or 'pldm', "
            f"observed={method!r}"
        )
    return method


def _training_sequence_contract(benchmark_config: Path) -> dict[str, int]:
    """Read the model-visible temporal shape from one benchmark config."""

    with benchmark_config.open("r", encoding="utf-8") as handle:
        benchmark = yaml.safe_load(handle)
    protocol = benchmark.get("training_protocol", {})
    history_tokens = int(protocol.get("history_tokens", 3))
    num_preds = int(protocol.get("num_preds", 1))
    raw_steps_per_action_block = int(
        protocol.get("raw_steps_per_action_block", 5)
    )
    if history_tokens < 1:
        raise ValueError("training_protocol.history_tokens must be positive")
    if num_preds < 1:
        raise ValueError("training_protocol.num_preds must be positive")
    if raw_steps_per_action_block != 5:
        raise ValueError(
            "The controlled TwoRoom runner requires five raw steps per "
            "model action block"
        )
    return {
        "history_tokens": history_tokens,
        "num_preds": num_preds,
        "sequence_steps": history_tokens + num_preds,
        "raw_steps_per_action_block": raw_steps_per_action_block,
    }


def _temporal_prediction_loss_spec(
    benchmark_config: Path,
    *,
    predicted_transitions: int,
) -> dict:
    """Load an optional, preregistered transition-weighting rule."""

    with benchmark_config.open("r", encoding="utf-8") as handle:
        benchmark = yaml.safe_load(handle)
    declared = benchmark.get("training_protocol", {}).get(
        "temporal_prediction_loss"
    )
    if declared is None:
        return {
            "configured": False,
            "mode": "uniform_mean",
            "transition_weights": [1.0] * int(predicted_transitions),
            "normalization": "divide_by_sum_of_weights",
            "applies_to": "all_training_groups",
        }
    if not isinstance(declared, dict):
        raise ValueError(
            "training_protocol.temporal_prediction_loss must be a mapping"
        )
    mode = str(declared.get("mode", ""))
    weights = [
        float(value) for value in declared.get("transition_weights", [])
    ]
    normalization = str(declared.get("normalization", ""))
    applies_to = str(declared.get("applies_to", ""))
    checks = {
        "mode": mode == "normalized_transition_weights",
        "length": len(weights) == int(predicted_transitions),
        "positive": bool(weights) and all(value > 0.0 for value in weights),
        "finite": all(math.isfinite(value) for value in weights),
        "normalization": normalization == "divide_by_sum_of_weights",
        "scope": applies_to == "all_training_groups",
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError(
            "Invalid temporal prediction loss specification: "
            f"{failed}"
        )
    return {
        "configured": True,
        "mode": mode,
        "transition_weights": weights,
        "normalized_transition_weight": [
            value / sum(weights) for value in weights
        ],
        "normalization": normalization,
        "applies_to": applies_to,
    }


def _training_objective_spec(
    training_method: str,
    cfg,
    *,
    temporal_prediction_loss: dict | None = None,
) -> dict:
    """Return the explicit model-loss contract stored in every report."""

    if training_method == "lewm":
        regularizer_name = str(
            cfg.loss.get("regularizer", "sigreg")
        ).lower()
        regularizer_cfg = cfg.loss.get(regularizer_name)
        if regularizer_cfg is None:
            raise ValueError(
                "Missing active LeWM representation regularizer config: "
                f"{regularizer_name}"
            )
        if bool(cfg.loss.std.enabled) and bool(cfg.loss.cov.enabled):
            objective_name = "native_lewm_plus_std_cov"
        elif regularizer_name == "visreg":
            objective_name = "lewm_visreg"
        elif float(regularizer_cfg.weight) != 0.09:
            objective_name = "lewm_sigreg_weight_sweep"
        else:
            objective_name = "native_lewm"
        result = {
            "name": objective_name,
            "prediction_target_detached": False,
            "prediction_weight": 1.0,
            "representation_regularizer": regularizer_name,
            "regularizer_weight": float(regularizer_cfg.weight),
            "regularizer_kwargs": dict(regularizer_cfg.kwargs),
            "sigreg_weight": (
                float(cfg.loss.sigreg.weight)
                if regularizer_name == "sigreg"
                else 0.0
            ),
            "visreg_weight": (
                float(cfg.loss.visreg.weight)
                if regularizer_name == "visreg"
                else 0.0
            ),
            "std_enabled": bool(cfg.loss.std.enabled),
            "std_weight": float(cfg.loss.std.weight),
            "cov_enabled": bool(cfg.loss.cov.enabled),
            "cov_weight": float(cfg.loss.cov.weight),
        }
    else:
        result = {
            "name": "native_pldm",
            "prediction_target_detached": False,
            "prediction_weight": 1.0,
            "std_weight": float(cfg.loss.std.weight),
            "std_t_weight": float(cfg.loss.std_t.weight),
            "cov_weight": float(cfg.loss.cov.weight),
            "cov_t_weight": float(cfg.loss.cov_t.weight),
            "temp_align_weight": float(cfg.loss.temp_align.weight),
            "idm_weight": float(cfg.loss.idm.weight),
        }
    if temporal_prediction_loss and temporal_prediction_loss["configured"]:
        return {
            **result,
            "name": f"{result['name']}_temporal_weighted",
            "temporal_prediction_loss": temporal_prediction_loss,
        }
    return result


def _apply_frozen_modules(model, specification: dict) -> dict:
    modules = list(specification["modules"])
    initial_state_sha256 = {}
    frozen_parameters = 0
    for name in modules:
        if not hasattr(model, name):
            raise AttributeError(
                f"StableWM model has no frozen module {name!r}"
            )
        module = getattr(model, name)
        module.requires_grad_(False)
        module.eval()
        initial_state_sha256[name] = _state_dict_sha256(module)
        frozen_parameters += sum(
            parameter.numel() for parameter in module.parameters()
        )
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    total_parameters = sum(
        parameter.numel() for parameter in model.parameters()
    )
    if specification["configured"] and (
        frozen_parameters <= 0 or trainable_parameters <= 0
    ):
        raise RuntimeError(
            "Representation-freeze diagnostic has no frozen or trainable "
            "parameters"
        )
    return {
        **specification,
        "applied": bool(specification["configured"]),
        "initial_state_sha256": initial_state_sha256,
        "frozen_parameters": frozen_parameters,
        "trainable_parameters": trainable_parameters,
        "total_parameters": total_parameters,
    }


def _finalize_frozen_modules(model, audit: dict) -> dict:
    final_state_sha256 = {
        name: _state_dict_sha256(getattr(model, name))
        for name in audit["modules"]
    }
    unchanged = {
        name: (
            final_state_sha256[name]
            == audit["initial_state_sha256"][name]
        )
        for name in audit["modules"]
    }
    passed = (
        not audit["configured"]
        or (
            bool(unchanged)
            and all(unchanged.values())
            and all(
                not parameter.requires_grad
                for name in audit["modules"]
                for parameter in getattr(model, name).parameters()
            )
        )
    )
    if not passed:
        raise RuntimeError(
            "A frozen representation module changed during training: "
            f"{unchanged}"
        )
    return {
        **audit,
        "final_state_sha256": final_state_sha256,
        "state_unchanged": unchanged,
        "passed": passed,
    }


def _initialization_checkpoint_spec(
    args: argparse.Namespace,
    *,
    benchmark_config: Path,
) -> dict | None:
    """Resolve and hash a model-only initialization, separate from resume."""

    with benchmark_config.open("r", encoding="utf-8") as handle:
        benchmark = yaml.safe_load(handle)
    declared = (
        benchmark.get("training_protocol", {}).get(
            "initialization_checkpoint"
        )
    )
    if declared is not None and not isinstance(declared, dict):
        raise ValueError(
            "training_protocol.initialization_checkpoint must be a mapping"
        )
    declared = dict(declared or {})
    cli_path = getattr(args, "initialization_checkpoint", None)
    cli_sha256 = getattr(
        args,
        "initialization_checkpoint_sha256",
        None,
    )
    raw_path = cli_path or declared.get("path")
    expected_sha256 = cli_sha256 or declared.get("sha256")
    if raw_path is None and expected_sha256 is None:
        return None
    if raw_path is None or expected_sha256 is None:
        raise ValueError(
            "Initialization checkpoint path and sha256 must be provided together"
        )
    path = resolve_contextworld_path(raw_path, repo_root=REPO_ROOT)
    if not path.is_file():
        raise FileNotFoundError(path)
    config_path = path.parent / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    observed_sha256 = _sha256(path)
    if observed_sha256 != str(expected_sha256):
        raise ValueError(
            "Initialization checkpoint hash mismatch: "
            f"expected={expected_sha256}, observed={observed_sha256}"
        )
    observed_config_sha256 = _sha256(config_path)
    expected_config_sha256 = declared.get("config_sha256")
    if (
        expected_config_sha256 is not None
        and observed_config_sha256 != str(expected_config_sha256)
    ):
        raise ValueError(
            "Initialization checkpoint config hash mismatch: "
            f"expected={expected_config_sha256}, "
            f"observed={observed_config_sha256}"
        )
    role = str(
        declared.get(
            "role",
            "model_weight_initialization_only",
        )
    )
    if role not in {
        "model_weight_initialization_only",
        "model_weight_initialization_only_not_resume",
    }:
        raise ValueError(
            "Initialization checkpoint role must explicitly be model-only, "
            f"observed={role!r}"
        )
    temporal_adaptation = declared.get("temporal_adaptation")
    if temporal_adaptation is not None:
        if not isinstance(temporal_adaptation, dict):
            raise ValueError(
                "initialization_checkpoint.temporal_adaptation must be a "
                "mapping"
            )
        required = {
            "parameter",
            "strategy",
            "source_history_tokens",
            "target_history_tokens",
            "align_corners",
            "source_anchor_target_indices",
        }
        missing = sorted(required - set(temporal_adaptation))
        if missing:
            raise ValueError(
                "Initialization temporal adaptation is incomplete: "
                f"missing={missing}"
            )
        if (
            temporal_adaptation["strategy"]
            != "linear_interpolation"
            or temporal_adaptation["align_corners"] is not True
        ):
            raise ValueError(
                "The controlled runner only supports linear temporal "
                "position interpolation with align_corners=true"
            )

    if cli_path is not None and declared.get("path") is not None:
        declared_path = resolve_contextworld_path(
            declared["path"],
            repo_root=REPO_ROOT,
        )
        if declared_path != path:
            raise ValueError(
                "CLI initialization checkpoint differs from benchmark config"
            )
    if (
        cli_sha256 is not None
        and declared.get("sha256") is not None
        and str(cli_sha256) != str(declared["sha256"])
    ):
        raise ValueError(
            "CLI initialization hash differs from benchmark config"
        )
    return {
        "path": str(path),
        "sha256": observed_sha256,
        "config": str(config_path),
        "config_sha256": observed_config_sha256,
        "role": role,
        "resume_state_loaded": False,
        "optimizer_state_loaded": False,
        "scheduler_state_loaded": False,
        "hash_audit_passed": True,
        "temporal_adaptation": temporal_adaptation,
    }


def _tensor_sha256(value) -> str:
    import torch

    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("utf-8"))
    digest.update(str(tuple(tensor.shape)).encode("utf-8"))
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _adapt_temporal_position_embedding(
    source_state: dict,
    target_state: dict,
    *,
    specification: dict[str, Any],
) -> tuple[dict, dict[str, Any]]:
    """Expand one learned temporal position table under a frozen contract."""

    import torch
    import torch.nn.functional as functional

    parameter = str(specification["parameter"])
    source_history = int(specification["source_history_tokens"])
    target_history = int(specification["target_history_tokens"])
    anchors = [
        int(value)
        for value in specification["source_anchor_target_indices"]
    ]
    if source_history < 2 or target_history <= source_history:
        raise ValueError(
            "Temporal initialization requires target history to be longer "
            "than a source history of at least two tokens"
        )
    if set(source_state) != set(target_state):
        raise RuntimeError(
            "Initialization checkpoint state keys differ from the H7 model: "
            f"source_only={sorted(set(source_state) - set(target_state))}, "
            f"target_only={sorted(set(target_state) - set(source_state))}"
        )
    if parameter not in source_state:
        raise KeyError(
            f"Temporal initialization parameter is absent: {parameter}"
        )
    source = source_state[parameter].detach().cpu()
    target = target_state[parameter].detach().cpu()
    expected_source_shape = (1, source_history, source.shape[-1])
    expected_target_shape = (1, target_history, source.shape[-1])
    if tuple(source.shape) != expected_source_shape:
        raise ValueError(
            f"Source temporal position shape differs: {tuple(source.shape)} "
            f"!= {expected_source_shape}"
        )
    if tuple(target.shape) != expected_target_shape:
        raise ValueError(
            f"Target temporal position shape differs: {tuple(target.shape)} "
            f"!= {expected_target_shape}"
        )
    shape_mismatches = {
        name: {
            "source": list(value.shape),
            "target": list(target_state[name].shape),
        }
        for name, value in source_state.items()
        if name != parameter
        and tuple(value.shape) != tuple(target_state[name].shape)
    }
    if shape_mismatches:
        raise RuntimeError(
            "Temporal initialization found additional shape differences: "
            f"{shape_mismatches}"
        )

    numerator = target_history - 1
    denominator = source_history - 1
    if any((index * numerator) % denominator for index in range(source_history)):
        raise ValueError(
            "Every source position must align exactly to a target index"
        )
    expected_anchors = [
        index * numerator // denominator
        for index in range(source_history)
    ]
    if anchors != expected_anchors:
        raise ValueError(
            "Declared temporal anchors differ from align_corners geometry: "
            f"{anchors} != {expected_anchors}"
        )

    interpolated = functional.interpolate(
        source.to(dtype=torch.float32).transpose(1, 2),
        size=target_history,
        mode="linear",
        align_corners=True,
    ).transpose(1, 2)
    interpolated = interpolated.to(dtype=target.dtype)
    if not torch.isfinite(interpolated).all():
        raise RuntimeError("Temporal position interpolation is non-finite")
    if not all(
        torch.equal(interpolated[:, target_index], source[:, source_index])
        for source_index, target_index in enumerate(anchors)
    ):
        raise RuntimeError(
            "Temporal position interpolation did not preserve source anchors"
        )

    adapted = dict(source_state)
    adapted[parameter] = interpolated
    audit = {
        "parameter": parameter,
        "strategy": "linear_interpolation",
        "align_corners": True,
        "source_history_tokens": source_history,
        "target_history_tokens": target_history,
        "source_shape": list(source.shape),
        "target_shape": list(interpolated.shape),
        "source_anchor_target_indices": anchors,
        "source_tensor_sha256": _tensor_sha256(source),
        "adapted_tensor_sha256": _tensor_sha256(interpolated),
        "source_anchors_preserved_exactly": True,
        "only_declared_tensor_shape_changed": True,
        "passed": True,
    }
    return adapted, audit


def _apply_initialization_checkpoint(
    model,
    *,
    swm,
    specification: dict[str, Any] | None,
    cache_dir: Path,
    resume_checkpoint: Path | None,
) -> dict[str, Any]:
    import torch

    if specification is None:
        return {
            "configured": False,
            "applied": False,
            "reason": "from_scratch_or_full_state_resume",
        }
    metadata = {"configured": True, **specification}
    if resume_checkpoint is not None:
        return {
            **metadata,
            "applied": False,
            "reason": "full_state_resume_supersedes_initialization",
        }

    source = swm.wm.utils.load_pretrained(
        specification["path"],
        cache_dir=str(cache_dir),
    )
    source_hash = _state_dict_sha256(source)
    temporal_adaptation = specification.get("temporal_adaptation")
    if temporal_adaptation is None:
        state = source.state_dict()
        adaptation_audit = {
            "required": False,
            "passed": True,
        }
    else:
        state, adaptation_audit = _adapt_temporal_position_embedding(
            source.state_dict(),
            model.state_dict(),
            specification=temporal_adaptation,
        )
        adaptation_audit = {
            "required": True,
            **adaptation_audit,
        }
    result = model.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            "Initialization checkpoint state keys differ from the model: "
            f"missing={result.missing_keys}, unexpected={result.unexpected_keys}"
        )
    initialized_hash = _state_dict_sha256(model)
    if temporal_adaptation is None and initialized_hash != source_hash:
        raise RuntimeError(
            "Initialization checkpoint did not load exactly into the model"
        )
    if temporal_adaptation is not None:
        parameter = str(temporal_adaptation["parameter"])
        source_state = source.state_dict()
        initialized_state = model.state_dict()
        unchanged = {
            name: torch.equal(
                value.detach().cpu(),
                initialized_state[name].detach().cpu(),
            )
            for name, value in source_state.items()
            if name != parameter
        }
        if not unchanged or not all(unchanged.values()):
            raise RuntimeError(
                "Non-temporal initialization tensors changed: "
                f"{[name for name, value in unchanged.items() if not value]}"
            )
        expected_adapted, _ = _adapt_temporal_position_embedding(
            source_state,
            initialized_state,
            specification=temporal_adaptation,
        )
        if not torch.equal(
            initialized_state[parameter].detach().cpu(),
            expected_adapted[parameter].detach().cpu(),
        ):
            raise RuntimeError(
                "Loaded temporal position tensor differs from the frozen "
                "interpolation"
            )
        adaptation_audit["unchanged_parameter_tensors"] = len(unchanged)
        adaptation_audit["all_other_parameter_tensors_exact"] = True
    return {
        **metadata,
        "applied": True,
        "reason": (
            "fresh_model_weight_initialization"
            if temporal_adaptation is None
            else "fresh_model_weight_initialization_with_frozen_temporal_expansion"
        ),
        "source_model_class": (
            f"{type(source).__module__}.{type(source).__name__}"
        ),
        "source_state_sha256": source_hash,
        "initialized_state_sha256": initialized_hash,
        "state_exact": temporal_adaptation is None,
        "temporal_adaptation_audit": adaptation_audit,
    }


def _load_pinned_train_module(stable_repo: Path, training_method: str):
    path = stable_repo / f"scripts/train/{training_method}.py"
    spec = importlib.util.spec_from_file_location(
        f"contextworld_pinned_stablewm_{training_method}_train", path
    )
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _configure_training_logger(cfg, args) -> dict:
    """Apply the shared StableWM logger contract to a composed train config."""

    from omegaconf import OmegaConf, open_dict

    backend = str(args.logger_backend).lower()
    if backend not in {"none", "swanlab", "wandb"}:
        raise ValueError(f"Unsupported logger backend: {backend}")

    with open_dict(cfg):
        if cfg.get("swanlab") is None:
            cfg.swanlab = OmegaConf.create({})
        if cfg.get("wandb") is None:
            cfg.wandb = OmegaConf.create({"config": {}})
        if cfg.swanlab.get("config") is None:
            cfg.swanlab.config = OmegaConf.create({})
        if cfg.wandb.get("config") is None:
            cfg.wandb.config = OmegaConf.create({})

        cfg.logger_backend = backend
        cfg.swanlab.enabled = backend == "swanlab"
        cfg.wandb.enabled = backend == "wandb"
        cfg.swanlab.collect_hardware = bool(
            args.swanlab_collect_hardware
        )
        cfg.swanlab.hardware_monitor = bool(
            args.swanlab_hardware_monitor
        )
        cfg.swanlab.log_hyperparams = bool(
            args.swanlab_log_hyperparams
        )
        cfg.swanlab.config.experiment_name = (
            args.swanlab_experiment_name or args.run_name
        )
        cfg.swanlab.config.id = args.swanlab_id or args.run_name

        optional_swanlab_values = {
            "project": args.swanlab_project,
            "workspace": args.swanlab_workspace,
            "logdir": args.swanlab_logdir,
            "mode": args.swanlab_mode,
        }
        for name, value in optional_swanlab_values.items():
            if value is not None:
                cfg.swanlab.config[name] = value

    selected = cfg.get(backend, {}) if backend != "none" else {}
    selected_config = selected.get("config", {}) if selected else {}
    return {
        "backend": backend,
        "enabled": backend != "none",
        "project": selected_config.get("project"),
        "workspace": selected_config.get("workspace"),
        "experiment_name": selected_config.get(
            "experiment_name",
            selected_config.get("name"),
        ),
        "run_id": selected_config.get("id"),
        "logdir": selected_config.get("logdir"),
        "mode": selected_config.get("mode"),
        "collect_hardware": bool(
            selected.get("collect_hardware", False)
        ),
        "hardware_monitor": bool(
            selected.get("hardware_monitor", False)
        ),
        "log_hyperparams": bool(
            selected.get("log_hyperparams", False)
        ),
    }


def _build_training_logger_preserving_rng(
    cfg,
    *,
    builder,
    torch_module,
):
    """Initialize an external logger without changing training RNG streams."""

    import random

    import numpy as np

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch_module.get_rng_state()
    cuda_states = (
        torch_module.cuda.get_rng_state_all()
        if torch_module.cuda.is_initialized()
        else None
    )
    try:
        return builder(cfg)
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch_module.set_rng_state(torch_state)
        if cuda_states is not None:
            torch_module.cuda.set_rng_state_all(cuda_states)


def _compose_model_config(
    stable_repo: Path,
    args,
    *,
    training_method: str,
    history_tokens: int = 3,
    num_preds: int = 1,
):
    import hydra
    from omegaconf import open_dict

    config_dir = str((stable_repo / "scripts/train/config").resolve())
    with hydra.initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = hydra.compose(
            config_name=training_method,
            overrides=["data=tworoom"],
        )
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
        cfg.wm.history_size = int(history_tokens)
        cfg.wm.num_preds = int(num_preds)
        cfg.model.action_encoder.input_dim = 10
        if training_method == "lewm":
            cfg.loss.regularizer = args.lewm_regularizer
            cfg.loss.sigreg.weight = args.lewm_sigreg_weight
            cfg.loss.visreg.weight = args.lewm_visreg_weight
            cfg.loss.std.enabled = args.lewm_std_weight > 0.0
            cfg.loss.std.weight = args.lewm_std_weight
            cfg.loss.cov.enabled = args.lewm_cov_weight > 0.0
            cfg.loss.cov.weight = args.lewm_cov_weight
        if training_method == "pldm":
            cfg.idm.input_dim = 2 * int(cfg.wm.embed_dim)
            cfg.idm.output_dim = 10
    logger_config = _configure_training_logger(cfg, args)
    with open_dict(cfg):
        cfg.contextworld_logger = logger_config
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


def _canonicalize_complete_epoch_checkpoint_state(
    checkpoint: dict,
    *,
    optimizer_steps_per_epoch: int,
) -> dict:
    """Write Lightning's epoch-end checkpoint in a restart-stable form."""

    global_step = int(checkpoint.get("global_step", -1))
    if global_step <= 0 or global_step % optimizer_steps_per_epoch != 0:
        raise RuntimeError(
            "Cannot save a recovery checkpoint outside a complete epoch "
            f"boundary: step={global_step}, "
            f"steps_per_epoch={optimizer_steps_per_epoch}"
        )
    completed_epochs = global_step // optimizer_steps_per_epoch
    try:
        epoch_progress = checkpoint["loops"]["fit_loop"]["epoch_progress"]
        current = epoch_progress["current"]
        total = epoch_progress["total"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError(
            "Checkpoint is missing Lightning epoch-loop progress"
        ) from exc
    counter_names = ("ready", "started", "processed", "completed")
    previous_current = {
        name: int(current.get(name, -1)) for name in counter_names
    }
    previous_total = {
        name: int(total.get(name, -1)) for name in counter_names
    }
    # ModelCheckpoint serializes inside on_train_epoch_end, before Lightning
    # increments ``epoch`` and ``epoch_progress.completed``.  Canonicalize all
    # cumulative epoch counters to the number proved by global_step.  This is
    # also necessary after a prior resume, where Lightning otherwise keeps the
    # old serialized epoch index while global_step continues to advance.
    canonical = {name: completed_epochs for name in counter_names}
    epoch_progress["current"] = dict(canonical)
    epoch_progress["total"] = dict(canonical)
    previous_epoch = int(checkpoint.get("epoch", -1))
    checkpoint["epoch"] = completed_epochs
    return {
        "applied": True,
        "global_step": global_step,
        "completed_epochs": completed_epochs,
        "previous_epoch": previous_epoch,
        "canonical_epoch": completed_epochs,
        "previous_current_epoch_progress": previous_current,
        "previous_total_epoch_progress": previous_total,
        "canonical_epoch_progress": canonical,
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
    try:
        fit_loop = checkpoint["loops"]["fit_loop"]
        batch_progress = fit_loop[
            "epoch_loop.batch_progress"
        ]
        current = batch_progress["current"]
        total = batch_progress["total"]
        epoch_progress = fit_loop["epoch_progress"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError(
            "Checkpoint is missing Lightning train/epoch-loop progress"
        ) from exc
    stored_epoch = int(checkpoint.get("epoch", -1))
    # ModelCheckpoint runs inside ``on_train_epoch_end``.  Lightning has
    # processed the epoch at this point, but has not yet incremented the
    # serialized ``epoch`` field or ``epoch_progress.completed``.  At restore,
    # FitLoop.on_run_start promotes completed=processed, so the native
    # representation below is the exact boundary before the next epoch.
    native_epoch_end = stored_epoch == expected_epoch - 1
    completed_epoch_end = stored_epoch == expected_epoch
    if not (native_epoch_end or completed_epoch_end):
        raise RuntimeError(
            "Checkpoint epoch/global-step boundary mismatch: "
            f"epoch={stored_epoch}, expected one of "
            f"[{expected_epoch - 1}, {expected_epoch}]"
        )
    epoch_current = epoch_progress.get("current", {})
    observed_epoch_progress = {
        name: int(epoch_current.get(name, -1))
        for name in ("ready", "started", "processed", "completed")
    }
    expected_epoch_progress = {
        "ready": expected_epoch,
        "started": expected_epoch,
        "processed": expected_epoch,
        "completed": (
            expected_epoch - 1 if native_epoch_end else expected_epoch
        ),
    }
    if observed_epoch_progress != expected_epoch_progress:
        raise RuntimeError(
            "Checkpoint epoch progress is not at a complete boundary: "
            f"observed={observed_epoch_progress}, "
            f"expected={expected_epoch_progress}"
        )
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
        "stored_epoch": stored_epoch,
        "native_on_train_epoch_end_state": native_epoch_end,
        "previous_epoch_progress": observed_epoch_progress,
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
    benchmark_config: Path,
    *,
    devices: int,
    passage_model: bool = False,
    audit_concurrency: int | None = None,
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
    expected_transport = (
        "framework_defaults_with_frozen_rendezvous_timeout"
    )
    observed_transport = declared.get("transport_configuration")
    if passage_model and observed_transport != expected_transport:
        raise ValueError(
            "Distributed transport configuration differs from its scope: "
            f"expected={expected_transport}, "
            f"observed={observed_transport}"
        )
    if (
        not passage_model
        and observed_transport not in (None, "framework_defaults")
    ):
        raise ValueError(
            "Non-passage training may only use framework-default transport: "
            f"observed={observed_transport}"
        )
    declared_timeout = declared.get("rendezvous_timeout_seconds")
    declared_scope = declared.get("rendezvous_timeout_scope")
    if passage_model:
        if (
            declared_timeout
            != PASSAGE_DDP_RENDEZVOUS_TIMEOUT_SECONDS
            or declared_scope != "passage_multi_gpu_only"
        ):
            raise ValueError(
                "Passage DDP rendezvous timeout must be frozen to "
                f"{PASSAGE_DDP_RENDEZVOUS_TIMEOUT_SECONDS} seconds with "
                "scope=passage_multi_gpu_only"
            )
    elif declared_timeout is not None or declared_scope is not None:
        raise ValueError(
            "Non-passage training must not declare a passage-only DDP "
            "rendezvous timeout"
        )
    applied_timeout = (
        int(declared_timeout)
        if passage_model and devices > 1
        else None
    )
    if audit_concurrency is not None:
        if not passage_model:
            raise ValueError(
                "Audit concurrency override is passage-only"
            )
        if audit_concurrency not in {1, 8}:
            raise ValueError(
                "Passage audit concurrency must be exactly 1 or 8"
            )
        audit_scheduling = (
            SERIAL_AUDIT_SCHEDULING_CONTRACT
            if audit_concurrency == 1
            else PARALLEL_AUDIT_SCHEDULING_CONTRACT
        )
        rank_cpu_affinity = (
            None
            if audit_concurrency == 1
            else PARALLEL_RANK_CPU_AFFINITY_CONTRACT
        )
        audit_scheduling_source = "controlled_cli_override"
    else:
        audit_scheduling = declared.get("audit_scheduling")
        rank_cpu_affinity = declared.get("rank_cpu_affinity")
        audit_scheduling_source = "benchmark_config"
    if (
        passage_model
        and audit_scheduling not in AUDIT_SCHEDULING_CONTRACTS
    ):
        raise ValueError(
            "Passage audit scheduling must use one single-node audit "
            f"contract from the frozen safe set: "
            f"expected={AUDIT_SCHEDULING_CONTRACTS}, "
            f"observed={audit_scheduling}"
        )
    if not passage_model and audit_scheduling is not None:
        raise ValueError(
            "Non-passage training must not declare passage audit scheduling"
        )
    if (
        passage_model
        and audit_scheduling == PARALLEL_AUDIT_SCHEDULING_CONTRACT
        and rank_cpu_affinity != PARALLEL_RANK_CPU_AFFINITY_CONTRACT
    ):
        raise ValueError(
            "Parallel passage audits require the frozen disjoint rank CPU "
            f"affinity: expected={PARALLEL_RANK_CPU_AFFINITY_CONTRACT}, "
            f"observed={rank_cpu_affinity}"
        )
    if (
        passage_model
        and audit_scheduling == SERIAL_AUDIT_SCHEDULING_CONTRACT
        and rank_cpu_affinity is not None
    ):
        raise ValueError(
            "Serial passage audit config must not declare parallel rank "
            "CPU affinity"
        )
    if not passage_model and rank_cpu_affinity is not None:
        raise ValueError(
            "Non-passage training must not declare passage rank CPU affinity"
        )
    training_run_exclusivity = declared.get(
        "training_run_exclusivity"
    )
    if (
        passage_model
        and training_run_exclusivity
        != TRAINING_RUN_EXCLUSIVITY_CONTRACT
    ):
        raise ValueError(
            "Passage training-run exclusivity differs from the frozen "
            f"single-root contract: expected="
            f"{TRAINING_RUN_EXCLUSIVITY_CONTRACT}, "
            f"observed={training_run_exclusivity}"
        )
    if not passage_model and training_run_exclusivity is not None:
        raise ValueError(
            "Non-passage training must not declare passage root-run "
            "exclusivity"
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
        "rendezvous_timeout_seconds_declared": declared_timeout,
        "rendezvous_timeout_scope": declared_scope,
        "rendezvous_timeout_seconds_applied": applied_timeout,
        "rendezvous_timeout_override_applied": (
            applied_timeout is not None
        ),
        "audit_scheduling": audit_scheduling,
        "audit_scheduling_source": audit_scheduling_source,
        "rank_cpu_affinity": rank_cpu_affinity,
        "training_run_exclusivity": training_run_exclusivity,
        "primary_formal_launch": declared.get("primary_formal_launch"),
        "resume_role": declared.get("resume_role"),
        "resume_scope": declared.get("resume_scope"),
        "recovery_acceptance": acceptance,
        "recovery_verification": verification,
    }


def _trainer_strategy_kwargs(
    distributed_execution_contract: dict,
    *,
    ddp_strategy_class,
) -> dict:
    timeout_seconds = distributed_execution_contract.get(
        "rendezvous_timeout_seconds_applied"
    )
    if timeout_seconds is None:
        return {}
    if (
        int(timeout_seconds)
        != PASSAGE_DDP_RENDEZVOUS_TIMEOUT_SECONDS
    ):
        raise ValueError(
            f"Unexpected passage DDP timeout: {timeout_seconds}"
        )
    return {
        "strategy": ddp_strategy_class(
            timeout=timedelta(seconds=int(timeout_seconds))
        )
    }


def _run_with_release_lock_held(args) -> dict:
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
    from lightning.pytorch.strategies import DDPStrategy
    from lightning.pytorch.trainer.connectors import callback_connector
    from omegaconf import OmegaConf, open_dict
    from stable_worldmodel.wm.loss import (
        PLDMLoss,
        SIGReg,
        TemporalStraighteningLoss,
        VCReg,
    )
    pl.seed_everything(args.seed, workers=True)
    training_method = _training_method(args.benchmark_config.resolve())
    sequence_contract = _training_sequence_contract(
        args.benchmark_config.resolve()
    )
    temporal_prediction_loss = _temporal_prediction_loss_spec(
        args.benchmark_config.resolve(),
        predicted_transitions=sequence_contract["history_tokens"],
    )
    if training_method != "lewm" and (
        args.lewm_regularizer != "sigreg"
        or args.lewm_sigreg_weight != 0.09
        or args.lewm_visreg_weight != 0.09
        or args.lewm_std_weight != 0.0
        or args.lewm_cov_weight != 0.0
    ):
        raise ValueError(
            "LeWM objective overrides cannot be applied to a PLDM training "
            "configuration"
        )
    pinned_train = _load_pinned_train_module(
        stable_repo,
        training_method,
    )
    cfg = _compose_model_config(
        stable_repo,
        args,
        training_method=training_method,
        history_tokens=sequence_contract["history_tokens"],
        num_preds=sequence_contract["num_preds"],
    )
    training_objective = _training_objective_spec(
        training_method,
        cfg,
        temporal_prediction_loss=temporal_prediction_loss,
    )
    initialization_spec = _initialization_checkpoint_spec(
        args,
        benchmark_config=args.benchmark_config.resolve(),
    )
    frozen_module_spec = _frozen_module_spec(
        args.benchmark_config.resolve()
    )
    passage_release_root = getattr(
        args,
        "_passage_release_root",
        None,
    )

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = output_root / "checkpoints" / args.run_name
    resume_checkpoint = run_dir / "last.ckpt"
    manager_resume_checkpoint = _validate_resume_policy(
        run_dir=run_dir,
        checkpoint_path=resume_checkpoint,
        policy=args.resume_policy,
    )

    if passage_release_root is not None:
        release_lock = getattr(args, "_passage_release_lock", {})
        if release_lock.get("mode") != "shared":
            raise RuntimeError(
                "Passage audit lock order requires the release shared lock "
                "to be held first"
            )
    distributed_execution_contract = _load_distributed_execution_contract(
        args.benchmark_config,
        devices=args.devices,
        passage_model=passage_release_root is not None,
        audit_concurrency=args.audit_concurrency,
    )
    local_rank_cpu_affinity = getattr(
        args,
        "_passage_rank_cpu_affinity",
        None,
    )
    distributed_execution_contract["local_rank_cpu_affinity"] = (
        dict(local_rank_cpu_affinity)
        if local_rank_cpu_affinity is not None
        else None
    )
    audit_scheduling_contract = distributed_execution_contract.get(
        "audit_scheduling"
    )
    parallel_passage_audits = bool(
        audit_scheduling_contract
        == PARALLEL_AUDIT_SCHEDULING_CONTRACT
    )
    audit_schedule_context = (
        hidden_passage_audit_scheduling_lock(
            passage_release_root,
            shared=parallel_passage_audits,
        )
        if passage_release_root is not None
        else nullcontext(None)
    )
    with audit_schedule_context as build_audit_schedule:
        grouped = build_tworoom_grouped_data(
            swm,
            repo_root=REPO_ROOT,
            benchmark_config=args.benchmark_config.resolve(),
            model_id=args.model_id,
            epoch_size=args.epoch_size,
            validation_epoch_size=args.validation_epoch_size,
            original_h5=args.original_h5,
            frameskip=sequence_contract["raw_steps_per_action_block"],
            num_steps=sequence_contract["sequence_steps"],
            img_size=224,
            seed=args.data_split_seed,
            expected_stablewm_commit=stable_commit,
        )
        sample_contract = _sample_contract(
            grouped.train,
            history_tokens=sequence_contract["history_tokens"],
            num_preds=sequence_contract["num_preds"],
        )
    build_audit_schedule = (
        dict(build_audit_schedule)
        if build_audit_schedule is not None
        else None
    )
    expected_audit_mode = (
        "shared" if parallel_passage_audits else "exclusive"
    )
    if build_audit_schedule is not None:
        if not (
            build_audit_schedule.get("acquired") is True
            and build_audit_schedule.get("released") is True
            and build_audit_schedule.get("mode")
            == expected_audit_mode
            and build_audit_schedule.get("policy")
            == audit_scheduling_contract["policy"]
            and build_audit_schedule.get("protocol")
            == audit_scheduling_contract["lock_protocol"]
            and build_audit_schedule.get("path_identity_verified") is True
            and build_audit_schedule.get(
                "path_identity_verified_after_acquire"
            )
            is True
            and build_audit_schedule.get("descriptor_inheritable") is False
        ):
            raise RuntimeError(
                "Passage full-audit scheduling lock did not complete its "
                "safe release contract"
            )
        build_audit_schedule.update(
            {
                "release_shared_lock_held": True,
                "collective_unlocked": bool(
                    build_audit_schedule["released"]
                ),
                "sample_contract_reads_inside_lock": int(
                    sample_contract["sample_count"]
                ),
            }
        )
    training_plan = _build_training_plan(args, grouped.metadata)
    pre_release_sample_reads = int(sample_contract["sample_count"])
    if passage_release_root is not None:
        grouped.metadata["distributed_passage_audit"].update(
            {
                "optimization": "disabled_per_rank_full_audit",
                "local_rank": int(os.environ.get("LOCAL_RANK", "0")),
                "full_logical_audit_execution_count": 1,
                "full_audit_scheduling": build_audit_schedule,
                "training_run_exclusivity": (
                    _training_run_exclusivity_snapshot(args)
                ),
                "local_rank_cpu_affinity": (
                    dict(local_rank_cpu_affinity)
                    if local_rank_cpu_affinity is not None
                    else None
                ),
                "preflight_sample_contract_reads": (
                    pre_release_sample_reads
                ),
                "internal_environment_clean": not any(
                    name in os.environ
                    for name in PASSAGE_INTERNAL_ENVIRONMENT
                ),
            }
        )
    with open_dict(cfg):
        cfg.contextworld_benchmark = {
            "adapter": "ContextWorldGroupedDataModule",
            "benchmark_config": str(args.benchmark_config.resolve()),
            "model_id": args.model_id,
            "profile": args.profile,
            "training_method": training_method,
            "training_objective": training_objective,
            "training_plan": training_plan,
            "distributed_execution_contract": (
                distributed_execution_contract
            ),
            "data": grouped.metadata,
            "initialization_checkpoint": initialization_spec,
            "frozen_model_modules": frozen_module_spec,
            "model_input_boundary": sample_contract[
                "collated_batch_audit"
            ],
            "diagnostics": {
                "loss_trace_interval_optimizer_steps": 20,
                "gradient_trace_steps": list(
                    args.diagnostic_checkpoint_step
                ),
                "model_checkpoint_steps": list(
                    args.diagnostic_checkpoint_step
                ),
            },
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
                "training_entry": str(
                    stable_repo
                    / f"scripts/train/{training_method}.py"
                ),
                "training_entry_sha256": _sha256(
                    stable_repo
                    / f"scripts/train/{training_method}.py"
                ),
                "logger_entry": str(
                    stable_repo / "stable_worldmodel/loggers.py"
                ),
                "logger_entry_sha256": _sha256(
                    stable_repo / "stable_worldmodel/loggers.py"
                ),
            },
            "logger": {
                **OmegaConf.to_container(
                    cfg.contextworld_logger,
                    resolve=True,
                ),
                "initialized": False,
                "reason": "preflight_does_not_create_external_logger",
            },
            "model_contract": {
                "class": (
                    "stable_worldmodel.wm.lewm.lewm.LeWM"
                    if training_method == "lewm"
                    else "stable_worldmodel.wm.pldm.pldm.PLDM"
                ),
                "training_method": training_method,
                "training_objective": training_objective,
                "action_block": sequence_contract[
                    "raw_steps_per_action_block"
                ],
                "history_size": sequence_contract["history_tokens"],
                "num_preds": sequence_contract["num_preds"],
                "raw_batch_keys": sample_contract[
                    "collated_batch_audit"
                ]["raw_keys"],
                "model_boundary_keys": sample_contract[
                    "collated_batch_audit"
                ]["model_boundary_keys"],
                "privileged_fields_at_model_boundary": sample_contract[
                    "collated_batch_audit"
                ]["privileged_fields_at_model_boundary"],
            },
            "initialization_checkpoint": (
                {
                    "configured": True,
                    "applied": False,
                    "reason": "preflight_hash_audit_only",
                    **initialization_spec,
                }
                if initialization_spec is not None
                else {"configured": False, "applied": False}
            ),
            "frozen_model_modules": {
                **frozen_module_spec,
                "applied": False,
                "reason": "preflight_does_not_instantiate_model",
            },
            "diagnostics": {
                "initialized": False,
                "reason": "preflight_declares_but_does_not_write_traces",
                "loss_trace_interval_optimizer_steps": 20,
                "gradient_trace_steps": list(
                    args.diagnostic_checkpoint_step
                ),
                "model_checkpoint_steps": list(
                    args.diagnostic_checkpoint_step
                ),
            },
            "sample_contract": sample_contract,
            "training_plan": training_plan,
            "distributed_execution_contract": (
                distributed_execution_contract
            ),
            "data": grouped.metadata,
        }
        write_json(args.report.resolve(), report)
        return report

    passage_train_gate = None
    passage_val_gate = None
    if passage_release_root is not None:
        passage_train_gate = PassageReleaseGatedDataset(
            grouped.train,
            split="train",
        )
        passage_val_gate = PassageReleaseGatedDataset(
            grouped.val,
            split="validation",
        )
        grouped.train = passage_train_gate
        grouped.val = passage_val_gate

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

    if (
        args.profile in CONTROLLED_PROFILES
        and torch.cuda.device_count() < args.devices
    ):
        raise RuntimeError(
            f"{args.profile.title()} training requires {args.devices} "
            "visible CUDA devices; "
            f"found {torch.cuda.device_count()}"
        )

    model = hydra.utils.instantiate(cfg.model)
    initialization_audit = _apply_initialization_checkpoint(
        model,
        swm=swm,
        specification=initialization_spec,
        cache_dir=output_root,
        resume_checkpoint=manager_resume_checkpoint,
    )
    if frozen_module_spec["configured"] and manager_resume_checkpoint is not None:
        raise ValueError(
            "The representation-freeze diagnostic requires a fresh "
            "model-only initialization, not a full-state resume"
        )
    frozen_module_audit = _apply_frozen_modules(
        model,
        frozen_module_spec,
    )
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
    module_components = {"model": model}
    loss_components = {}
    if training_method == "lewm":
        regularizer_name = str(cfg.loss.regularizer).lower()
        regularizer_cfg = cfg.loss.get(regularizer_name)
        if regularizer_name == "sigreg":
            regularizer_class = SIGReg
        elif regularizer_name == "visreg":
            # Some pinned PLDM-only StableWM revisions intentionally predate
            # VISReg.  Do not make that unused optional LeWM loss a runtime
            # dependency of a native PLDM confirmation run.
            from stable_worldmodel.wm.loss import VISRegLoss

            regularizer_class = VISRegLoss
        else:
            raise ValueError(
                "Unsupported LeWM regularizer: "
                f"{regularizer_name!r}"
            )
        loss_components[regularizer_name] = regularizer_class(
            **regularizer_cfg.kwargs
        )
        if any(
            bool(cfg.loss.get(name).enabled)
            for name in ("std", "std_t", "cov", "cov_t")
            if cfg.loss.get(name) is not None
        ):
            loss_components["vc_reg"] = VCReg()
        base_forward = pinned_train.lejepa_forward
    else:
        module_components["idm"] = hydra.utils.instantiate(cfg.idm)
        loss_components.update(
            {
                "pldm": PLDMLoss(),
                "path_straight": TemporalStraighteningLoss(),
            }
        )
        base_forward = pinned_train.pldm_forward

    gradient_trace_path = run_dir / "gradient_trace.jsonl"

    class GradientTraceModule(spt.Module):
        """Record globally averaged, pre-clip module gradient norms."""

        _model_module_names = (
            "encoder",
            "projector",
            "predictor",
            "pred_proj",
            "action_encoder",
        )

        def __init__(
            self,
            *module_args,
            gradient_trace_path: Path,
            diagnostic_steps: list[int],
            accumulation_steps: int,
            **module_kwargs,
        ) -> None:
            super().__init__(*module_args, **module_kwargs)
            self.gradient_trace_path = gradient_trace_path
            self.gradient_diagnostic_steps = tuple(
                sorted(set(int(step) for step in diagnostic_steps))
            )
            self.gradient_accumulation_steps = int(accumulation_steps)
            self._diagnostic_batch_idx: int | None = None
            self._gradient_recorded_steps: set[int] = set()
            if self.gradient_trace_path.is_file():
                for line in self.gradient_trace_path.read_text(
                    encoding="utf-8"
                ).splitlines():
                    if line.strip():
                        self._gradient_recorded_steps.add(
                            int(json.loads(line)["optimizer_step"])
                        )

        @staticmethod
        def _gradient_energy(parameters, *, device):
            energy = torch.zeros((), dtype=torch.float32, device=device)
            for parameter in parameters:
                if parameter.grad is not None:
                    energy = energy + parameter.grad.detach().float().pow(
                        2
                    ).sum()
            return energy

        def training_step(self, batch, batch_idx):
            self._diagnostic_batch_idx = int(batch_idx)
            return super().training_step(batch, batch_idx)

        def after_manual_backward(self) -> None:
            batch_idx = self._diagnostic_batch_idx
            if batch_idx is None:
                raise RuntimeError(
                    "Gradient tracing did not observe the training batch index"
                )
            if (
                batch_idx + 1
            ) % self.gradient_accumulation_steps != 0:
                return

            optimizer_step = int(self.trainer.global_step) + 1
            if (
                optimizer_step not in self.gradient_diagnostic_steps
                or optimizer_step in self._gradient_recorded_steps
            ):
                return

            device = self.trainer.strategy.root_device
            group_names = [
                name
                for name in self._model_module_names
                if hasattr(self.model, name)
            ]
            group_modules = [
                getattr(self.model, name) for name in group_names
            ]
            if hasattr(self, "idm"):
                group_names.append("idm")
                group_modules.append(self.idm)
            energies = torch.stack(
                [
                    self._gradient_energy(
                        component.parameters(),
                        device=device,
                    )
                    for component in group_modules
                ]
                + [
                    self._gradient_energy(
                        self.model.parameters(),
                        device=device,
                    )
                ]
            )
            if (
                torch.distributed.is_available()
                and torch.distributed.is_initialized()
            ):
                torch.distributed.all_reduce(
                    energies,
                    op=torch.distributed.ReduceOp.SUM,
                )
                energies.div_(float(self.trainer.world_size))
            norm_names = [*group_names, "model_total"]
            norms = {
                name: float(value)
                for name, value in zip(
                    norm_names,
                    energies.clamp_min(0).sqrt().cpu().tolist(),
                    strict=True,
                )
            }
            if self.trainer.is_global_zero:
                row = {
                    "schema_version": 1,
                    "optimizer_step": optimizer_step,
                    "training_method": training_method,
                    "training_objective": training_objective["name"],
                    "world_size": int(self.trainer.world_size),
                    "measurement": "after_backward_before_gradient_clip",
                    "gradient_norms": norms,
                }
                with self.gradient_trace_path.open(
                    "a", encoding="utf-8"
                ) as handle:
                    handle.write(
                        json.dumps(row, ensure_ascii=False) + "\n"
                    )
                logger = self.trainer.logger
                if logger is not None:
                    logger.log_metrics(
                        {
                            (
                                "diagnostics/"
                                f"gradient_norm_{name}_pre_clip"
                            ): value
                            for name, value in norms.items()
                        },
                        step=optimizer_step,
                    )
            self._gradient_recorded_steps.add(optimizer_step)

    module = GradientTraceModule(
        **module_components,
        **loss_components,
        gradient_trace_path=gradient_trace_path,
        diagnostic_steps=args.diagnostic_checkpoint_step,
        accumulation_steps=args.accumulate_grad_batches,
        forward=partial(
            _lejepa_forward_with_manual_accumulation,
            base_forward=base_forward,
            cfg=cfg,
            accumulation_steps=args.accumulate_grad_batches,
            training_method=training_method,
            temporal_prediction_loss=temporal_prediction_loss,
        ),
        optim=optimizers,
    )
    # SPT 0.1.6 does not bind the configured nested optimizer name back to
    # optimizer index 0.  Without this, on_train_start creates "default_0"
    # with frequency=1 and silently bypasses the configured accumulation.
    module._optimizer_index_to_name[0] = "model_opt"
    data_module = spt.data.DataModule(train=train_loader, val=val_loader)

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
            self.checkpoint_save_canonicalization = None

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
            self.checkpoint_save_canonicalization = (
                _canonicalize_complete_epoch_checkpoint_state(
                    checkpoint,
                    optimizer_steps_per_epoch=training_plan[
                        "optimizer_steps_per_epoch"
                    ],
                )
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

    class LossTrace(Callback):
        """Persist globally averaged loss components without an external logger."""

        def __init__(self, path: Path, interval: int = 20) -> None:
            self.path = path
            self.interval = int(interval)
            self.last_recorded_step = -1
            self.records_written = 0

        def on_train_batch_end(
            self, trainer, pl_module, outputs, batch, batch_idx
        ) -> None:
            if not isinstance(outputs, dict):
                raise RuntimeError(
                    "Loss tracing requires the StablePretraining state dict"
                )
            step = int(trainer.global_step)
            target = int(args.expected_optimizer_steps)
            if not (
                step == 1
                or step == target
                or step % self.interval == 0
            ):
                return
            if step == self.last_recorded_step:
                return
            names = sorted(
                key
                for key, value in outputs.items()
                if "loss" in key
                and torch.is_tensor(value)
                and value.numel() == 1
            )
            if "loss" not in names or "pred_loss" not in names:
                raise RuntimeError(
                    f"Loss trace is missing total/prediction loss: {names}"
                )
            values = torch.stack(
                [
                    outputs[name].detach().float().reshape(())
                    for name in names
                ]
            )
            if (
                torch.distributed.is_available()
                and torch.distributed.is_initialized()
            ):
                torch.distributed.all_reduce(
                    values,
                    op=torch.distributed.ReduceOp.SUM,
                )
                values.div_(float(trainer.world_size))
            if trainer.is_global_zero:
                row = {
                    "schema_version": 1,
                    "optimizer_step": step,
                    "epoch": int(trainer.current_epoch) + 1,
                    "training_method": training_method,
                    "training_objective": training_objective["name"],
                    "world_size": int(trainer.world_size),
                    "losses": {
                        name: float(value)
                        for name, value in zip(
                            names,
                            values.cpu().tolist(),
                            strict=True,
                        )
                    },
                }
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(row, ensure_ascii=False) + "\n"
                    )
                self.records_written += 1
            self.last_recorded_step = step

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

    class SavePretrainedAtDiagnosticSteps(Callback):
        """Save model-only snapshots at predeclared optimizer steps."""

        def __init__(self, steps: list[int]) -> None:
            self.steps = tuple(sorted(set(int(step) for step in steps)))
            self.saved_steps: list[int] = []

        def on_train_batch_end(
            self, trainer, pl_module, outputs, batch, batch_idx
        ) -> None:
            step = int(trainer.global_step)
            if (
                not trainer.is_global_zero
                or step not in self.steps
                or step in self.saved_steps
            ):
                return
            filename = f"weights_step_{step}.pt"
            swm.wm.utils.save_pretrained(
                pl_module.model,
                run_name=args.run_name,
                config=cfg,
                filename=filename,
                cache_dir=str(output_root),
            )
            path = run_dir / filename
            if not path.is_file():
                raise RuntimeError(
                    f"Diagnostic model checkpoint was not saved: {path}"
                )
            self.saved_steps.append(step)

    class MaintainFrozenModuleEvalMode(Callback):
        """Prevent frozen BatchNorm buffers from changing during training."""

        def _apply(self, pl_module) -> None:
            for name in frozen_module_audit["modules"]:
                getattr(pl_module.model, name).eval()

        def on_fit_start(self, trainer, pl_module) -> None:
            self._apply(pl_module)

        def on_train_batch_start(
            self, trainer, pl_module, batch, batch_idx
        ) -> None:
            self._apply(pl_module)

        def on_validation_batch_start(
            self, trainer, pl_module, batch, batch_idx, dataloader_idx=0
        ) -> None:
            self._apply(pl_module)

    class PassageStorageContract(Callback):
        """Release real Dataset reads after every rank independently audits."""

        def __init__(self) -> None:
            self.required = bool(
                grouped.metadata.get("formal_build_report_audit", {}).get(
                    "required",
                    False,
                )
            )
            self.report: dict | None = None
            self.release_consensus_completed = False

        def on_fit_start(self, trainer, pl_module) -> None:
            if not self.required:
                self.release_consensus_completed = True
                return
            if passage_train_gate is None or passage_val_gate is None:
                raise RuntimeError("Passage Dataset gates were not installed")
            active_groups = [
                name
                for name in grouped.metadata["groups"]
                if name.startswith("passage_")
            ]
            if len(active_groups) != 1:
                raise RuntimeError(
                    "Passage training must have one independently audited "
                    f"group, observed={active_groups}"
                )
            logical = grouped.metadata["groups"][active_groups[0]][
                "catalog_split_audit"
            ]["logical_content_audit"]
            full_audit_passed = bool(
                logical.get("passed") is True
                and int(logical.get("shards_verified", 0)) > 0
                and int(logical.get("episodes_verified", 0)) > 0
            )
            storage_report = None
            storage_error = None
            revalidation_schedule_receipt = None
            try:
                release_lock = getattr(
                    args, "_passage_release_lock", {}
                )
                if release_lock.get("mode") != "shared":
                    raise RuntimeError(
                        "Passage revalidation lock order requires the "
                        "release shared lock to be held first"
                    )
                with hidden_passage_audit_scheduling_lock(
                    passage_release_root,
                    shared=parallel_passage_audits,
                ) as revalidation_schedule:
                    storage_report = (
                        revalidate_hidden_passage_training_storage(
                            args.benchmark_config.resolve(),
                            repo_root=REPO_ROOT,
                        )
                    )
                revalidation_schedule_receipt = dict(
                    revalidation_schedule
                )
            except Exception as exc:
                storage_error = f"{type(exc).__name__}: {exc}"
                if revalidation_schedule_receipt is None and (
                    "revalidation_schedule" in locals()
                ):
                    revalidation_schedule_receipt = dict(
                        revalidation_schedule
                    )
            revalidation_schedule_receipt = dict(
                revalidation_schedule_receipt or {}
            )
            revalidation_schedule_receipt.update(
                {
                    "release_shared_lock_held": (
                        getattr(args, "_passage_release_lock", {}).get(
                            "mode"
                        )
                        == "shared"
                    ),
                    "collective_unlocked": bool(
                        revalidation_schedule_receipt.get("released")
                    ),
                }
            )
            revalidation_lock_ready = bool(
                revalidation_schedule_receipt.get("acquired") is True
                and revalidation_schedule_receipt.get("released") is True
                and revalidation_schedule_receipt.get("mode")
                == expected_audit_mode
                and revalidation_schedule_receipt.get("policy")
                == audit_scheduling_contract["policy"]
                and revalidation_schedule_receipt.get("protocol")
                == audit_scheduling_contract["lock_protocol"]
                and revalidation_schedule_receipt.get(
                    "path_identity_verified"
                )
                is True
                and revalidation_schedule_receipt.get(
                    "path_identity_verified_after_acquire"
                )
                is True
                and revalidation_schedule_receipt.get(
                    "descriptor_inheritable"
                )
                is False
                and revalidation_schedule_receipt.get(
                    "fork_child_close_registered"
                )
                is True
                and revalidation_schedule_receipt[
                    "release_shared_lock_held"
                ]
                is True
                and revalidation_schedule_receipt[
                    "collective_unlocked"
                ]
                is True
            )
            full_audit_schedule = dict(build_audit_schedule or {})
            full_audit_lock_ready = bool(
                full_audit_schedule.get("acquired") is True
                and full_audit_schedule.get("released") is True
                and full_audit_schedule.get("mode")
                == expected_audit_mode
                and full_audit_schedule.get("policy")
                == audit_scheduling_contract["policy"]
                and full_audit_schedule.get("protocol")
                == audit_scheduling_contract["lock_protocol"]
                and full_audit_schedule.get("path_identity_verified") is True
                and full_audit_schedule.get(
                    "path_identity_verified_after_acquire"
                )
                is True
                and full_audit_schedule.get("descriptor_inheritable") is False
                and full_audit_schedule.get(
                    "fork_child_close_registered"
                )
                is True
                and full_audit_schedule.get(
                    "release_shared_lock_held"
                )
                is True
                and full_audit_schedule.get("collective_unlocked") is True
            )
            train_before = passage_train_gate.receipt()
            val_before = passage_val_gate.receipt()
            cpu_affinity_ready = bool(
                (
                    not parallel_passage_audits
                    and local_rank_cpu_affinity is None
                )
                or (
                    parallel_passage_audits
                    and isinstance(local_rank_cpu_affinity, dict)
                    and local_rank_cpu_affinity.get("passed") is True
                    and int(
                        local_rank_cpu_affinity.get("local_rank", -1)
                    )
                    == int(trainer.global_rank)
                    and int(
                        local_rank_cpu_affinity.get(
                            "cpus_per_rank",
                            -1,
                        )
                    )
                    == 8
                    and local_rank_cpu_affinity.get("cpu_ids")
                    == list(
                        range(
                            int(trainer.global_rank) * 8,
                            (int(trainer.global_rank) + 1) * 8,
                        )
                    )
                )
            )
            local_receipt = {
                "rank": int(trainer.global_rank),
                "passed": int(
                    full_audit_passed
                    and storage_report is not None
                    and full_audit_lock_ready
                    and revalidation_lock_ready
                    and cpu_affinity_ready
                    and train_before["pre_release_calls"] == 0
                    and train_before["pre_release_items"] == 0
                    and val_before["pre_release_calls"] == 0
                    and val_before["pre_release_items"] == 0
                    and not any(
                        name in os.environ
                        for name in PASSAGE_INTERNAL_ENVIRONMENT
                    )
                ),
                "full_logical_audit_count": 1,
                "storage_revalidation_count": int(
                    storage_report is not None
                ),
                "train_pre_release_calls": train_before[
                    "pre_release_calls"
                ],
                "train_pre_release_items": train_before[
                    "pre_release_items"
                ],
                "val_pre_release_calls": val_before[
                    "pre_release_calls"
                ],
                "val_pre_release_items": val_before[
                    "pre_release_items"
                ],
                "internal_environment_clean": int(
                    not any(
                        name in os.environ
                        for name in PASSAGE_INTERNAL_ENVIRONMENT
                    )
                ),
                "full_audit_lock_acquired": int(
                    full_audit_schedule.get("acquired") is True
                ),
                "full_audit_lock_released": int(
                    full_audit_schedule.get("released") is True
                ),
                "full_audit_release_shared_held": int(
                    full_audit_schedule.get(
                        "release_shared_lock_held"
                    )
                    is True
                ),
                "full_audit_collective_unlocked": int(
                    full_audit_schedule.get("collective_unlocked") is True
                ),
                "full_audit_path_identity_verified": int(
                    full_audit_schedule.get(
                        "path_identity_verified"
                    )
                    is True
                ),
                "full_audit_path_identity_verified_after_acquire": int(
                    full_audit_schedule.get(
                        "path_identity_verified_after_acquire"
                    )
                    is True
                ),
                "full_audit_descriptor_noninheritable": int(
                    full_audit_schedule.get(
                        "descriptor_inheritable"
                    )
                    is False
                ),
                "full_audit_fork_child_close_registered": int(
                    full_audit_schedule.get(
                        "fork_child_close_registered"
                    )
                    is True
                ),
                "full_audit_wait_milliseconds": (
                    _lock_seconds_to_milliseconds(
                        full_audit_schedule,
                        "wait_seconds",
                    )
                ),
                "full_audit_hold_milliseconds": (
                    _lock_seconds_to_milliseconds(
                        full_audit_schedule,
                        "hold_seconds",
                    )
                ),
                "full_audit_sample_contract_reads": int(
                    full_audit_schedule.get(
                        "sample_contract_reads_inside_lock",
                        -1,
                    )
                ),
                "revalidation_lock_acquired": int(
                    revalidation_schedule_receipt.get("acquired") is True
                ),
                "revalidation_lock_released": int(
                    revalidation_schedule_receipt.get("released") is True
                ),
                "revalidation_release_shared_held": int(
                    revalidation_schedule_receipt.get(
                        "release_shared_lock_held"
                    )
                    is True
                ),
                "revalidation_collective_unlocked": int(
                    revalidation_schedule_receipt.get(
                        "collective_unlocked"
                    )
                    is True
                ),
                "revalidation_path_identity_verified": int(
                    revalidation_schedule_receipt.get(
                        "path_identity_verified"
                    )
                    is True
                ),
                "revalidation_path_identity_verified_after_acquire": int(
                    revalidation_schedule_receipt.get(
                        "path_identity_verified_after_acquire"
                    )
                    is True
                ),
                "revalidation_descriptor_noninheritable": int(
                    revalidation_schedule_receipt.get(
                        "descriptor_inheritable"
                    )
                    is False
                ),
                "revalidation_fork_child_close_registered": int(
                    revalidation_schedule_receipt.get(
                        "fork_child_close_registered"
                    )
                    is True
                ),
                "revalidation_wait_milliseconds": (
                    _lock_seconds_to_milliseconds(
                        revalidation_schedule_receipt,
                        "wait_seconds",
                    )
                ),
                "revalidation_hold_milliseconds": (
                    _lock_seconds_to_milliseconds(
                        revalidation_schedule_receipt,
                        "hold_seconds",
                    )
                ),
            }
            receipts = _distributed_passage_full_audit_consensus(
                strategy=trainer.strategy,
                torch_module=torch,
                device=pl_module.device,
                local_receipt=local_receipt,
                expected_world_size=int(trainer.world_size),
            )
            passage_train_gate.release()
            passage_val_gate.release()
            self.release_consensus_completed = True
            self.report = {
                **dict(storage_report or {}),
                "optimization": "disabled_per_rank_full_audit",
                "local_storage_error": storage_error,
                "local_full_audit_scheduling": full_audit_schedule,
                "local_revalidation_scheduling": (
                    revalidation_schedule_receipt
                ),
                "full_logical_audit_execution_count": int(
                    trainer.world_size
                ),
                "storage_revalidation_execution_count": int(
                    trainer.world_size
                ),
                "rank_receipts": receipts,
                "expected_rank_receipts": int(trainer.world_size),
                "rank_coverage_passed": (
                    [row["rank"] for row in receipts]
                    == list(range(int(trainer.world_size)))
                ),
                "rank_cpu_affinity_contract": (
                    distributed_execution_contract.get(
                        "rank_cpu_affinity"
                    )
                ),
                "local_rank_cpu_affinity": (
                    dict(local_rank_cpu_affinity)
                    if local_rank_cpu_affinity is not None
                    else None
                ),
                "all_rank_cpu_affinity_checked_before_consensus": (
                    parallel_passage_audits
                ),
                "all_ranks_accepted_before_first_batch": True,
                "train_gate_before_release": train_before,
                "validation_gate_before_release": val_before,
                "training_dataloader_reads_before_release": sum(
                    row["train_pre_release_items"] for row in receipts
                ),
                "validation_dataloader_reads_before_release": sum(
                    row["val_pre_release_items"] for row in receipts
                ),
                "gates_opened_only_after_consensus": (
                    passage_train_gate.released
                    and passage_val_gate.released
                ),
                "passed": True,
            }

    class CudaMemoryAudit(Callback):
        def __init__(self) -> None:
            self.report: dict[str, Any] = {
                "configured": True,
                "passed": False,
            }

        def on_fit_start(self, trainer, pl_module) -> None:
            device = torch.device("cuda", torch.cuda.current_device())
            torch.cuda.reset_peak_memory_stats(device)

        def on_fit_end(self, trainer, pl_module) -> None:
            device = torch.device("cuda", torch.cuda.current_device())
            torch.cuda.synchronize(device)
            properties = torch.cuda.get_device_properties(
                device
            )
            local = torch.tensor(
                [
                    int(trainer.global_rank),
                    int(torch.cuda.max_memory_allocated(device)),
                    int(torch.cuda.max_memory_reserved(device)),
                    int(torch.cuda.memory_allocated(device)),
                    int(torch.cuda.memory_reserved(device)),
                    int(properties.total_memory),
                ],
                dtype=torch.int64,
                device=device,
            )
            gathered = trainer.strategy.all_gather(local)
            if gathered.ndim == 1:
                gathered = gathered.unsqueeze(0)
            values = gathered.detach().cpu().tolist()
            rows = []
            for (
                rank,
                peak_allocated,
                peak_reserved,
                final_allocated,
                final_reserved,
                total_memory,
            ) in sorted(values, key=lambda row: int(row[0])):
                rows.append(
                    {
                        "rank": int(rank),
                        "peak_allocated_bytes": int(peak_allocated),
                        "peak_reserved_bytes": int(peak_reserved),
                        "final_allocated_bytes": int(final_allocated),
                        "final_reserved_bytes": int(final_reserved),
                        "total_memory_bytes": int(total_memory),
                    }
                )
            ranks = [row["rank"] for row in rows]
            maximum_reserved = max(
                row["peak_reserved_bytes"] for row in rows
            )
            minimum_total = min(
                row["total_memory_bytes"] for row in rows
            )
            self.report = {
                "configured": True,
                "measurement": (
                    "torch_cuda_peak_from_fit_start_through_fit_end"
                ),
                "bytes_per_mib": 1024 * 1024,
                "rank_count": len(rows),
                "per_rank": rows,
                "maximum_peak_allocated_bytes": max(
                    row["peak_allocated_bytes"] for row in rows
                ),
                "maximum_peak_reserved_bytes": maximum_reserved,
                "minimum_total_memory_bytes": minimum_total,
                "minimum_reserved_headroom_bytes": (
                    minimum_total - maximum_reserved
                ),
                "maximum_peak_reserved_fraction": (
                    maximum_reserved / minimum_total
                ),
                "rank_coverage_exact": ranks
                == list(range(int(trainer.world_size))),
                "all_ranks_below_total_memory": all(
                    row["peak_reserved_bytes"]
                    < row["total_memory_bytes"]
                    for row in rows
                ),
                "passed": (
                    ranks == list(range(int(trainer.world_size)))
                    and all(
                        row["peak_reserved_bytes"]
                        < row["total_memory_bytes"]
                        for row in rows
                    )
                ),
            }

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
            if (
                passage_storage_contract.required
                and not passage_storage_contract.release_consensus_completed
            ):
                raise RuntimeError(
                    "Training batch was read before the passage audit gate"
                )
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
    passage_storage_contract = PassageStorageContract()
    training_contract = TrainingContract()
    loss_trace_path = run_dir / "loss_trace.jsonl"
    loss_trace = LossTrace(loss_trace_path, interval=20)
    diagnostic_checkpoints = SavePretrainedAtDiagnosticSteps(
        args.diagnostic_checkpoint_step
    )
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
        # Lightning 2.6 only calls ``save_last`` after a checkpoint was saved
        # in the same hook.  ``save_top_k=0`` therefore disables both the
        # named checkpoint and ``last.ckpt``.  Keep one rotating named state
        # so every complete epoch also produces the required resumable last
        # checkpoint.
        save_top_k=1,
        save_last=True,
        verbose=True,
        enable_version_counter=False,
        every_n_train_steps=None,
        every_n_epochs=1,
        save_on_train_epoch_end=True,
    )
    cuda_memory_audit = CudaMemoryAudit()
    callbacks = [
        MaintainFrozenModuleEvalMode(),
        cuda_memory_audit,
        rng_checkpoint,
        SavePretrainedAtEpochEnd(),
        diagnostic_checkpoints,
        passage_storage_contract,
        training_contract,
        loss_trace,
        state_checkpoint,
    ]
    # stable_pretraining registers environment-dump callbacks through a
    # Lightning entry point.  They are unrelated to model training and may
    # write host metadata into the repository, so this benchmark supplies
    # only its explicit callbacks.
    callback_connector._load_external_callbacks = lambda _group: []
    trainer_strategy = _trainer_strategy_kwargs(
        distributed_execution_contract,
        ddp_strategy_class=DDPStrategy,
    )
    if str(args.logger_backend).lower() == "none":
        # Older pinned StableWM revisions have no optional external logger
        # module.  An explicitly offline run must not require it.
        experiment_logger = False
    else:
        from stable_worldmodel.loggers import build_training_logger

        experiment_logger = _build_training_logger_preserving_rng(
            cfg,
            builder=build_training_logger,
            torch_module=torch,
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
        logger=experiment_logger,
        enable_checkpointing=True,
        enable_model_summary=False,
        default_root_dir=str(run_dir),
        num_sanity_val_steps=0,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        deterministic=True,
        **trainer_strategy,
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

    if cuda_memory_audit.report.get("passed") is not True:
        raise RuntimeError(
            "CUDA memory audit did not cover every training rank: "
            f"{cuda_memory_audit.report}"
        )
    completed_scheduler = trainer.lr_scheduler_configs[0].scheduler
    frozen_module_audit = _finalize_frozen_modules(
        module.model,
        frozen_module_audit,
    )

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

    if not loss_trace_path.is_file():
        raise RuntimeError(f"Missing loss trace: {loss_trace_path}")
    loss_trace_rows = [
        json.loads(line)
        for line in loss_trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    loss_trace_steps = [
        int(row["optimizer_step"]) for row in loss_trace_rows
    ]
    if (
        not loss_trace_rows
        or loss_trace_steps != sorted(set(loss_trace_steps))
        or loss_trace_steps[-1] != int(trainer.global_step)
    ):
        raise RuntimeError(
            "Loss trace is incomplete or has duplicate/out-of-order steps: "
            f"{loss_trace_steps}"
        )
    loss_trace_audit = {
        "path": str(loss_trace_path),
        "sha256": _sha256(loss_trace_path),
        "records": len(loss_trace_rows),
        "first_optimizer_step": loss_trace_steps[0],
        "last_optimizer_step": loss_trace_steps[-1],
        "interval_optimizer_steps": loss_trace.interval,
        "loss_keys": sorted(
            {
                key
                for row in loss_trace_rows
                for key in row["losses"]
            }
        ),
        "passed": True,
    }
    completed_diagnostic_steps = [
        int(step)
        for step in args.diagnostic_checkpoint_step
        if int(step) <= int(trainer.global_step)
    ]
    diagnostic_checkpoint_rows = []
    for step in completed_diagnostic_steps:
        checkpoint_path = run_dir / f"weights_step_{step}.pt"
        if not checkpoint_path.is_file():
            raise RuntimeError(
                "Missing declared diagnostic model checkpoint: "
                f"{checkpoint_path}"
            )
        diagnostic_checkpoint_rows.append(
            {
                "optimizer_step": step,
                "path": str(checkpoint_path),
                "sha256": _sha256(checkpoint_path),
            }
        )
    diagnostic_checkpoint_audit = {
        "configured": bool(args.diagnostic_checkpoint_step),
        "requested_optimizer_steps": list(
            args.diagnostic_checkpoint_step
        ),
        "completed_optimizer_steps": completed_diagnostic_steps,
        "saved_this_invocation": list(
            diagnostic_checkpoints.saved_steps
        ),
        "checkpoints": diagnostic_checkpoint_rows,
        "passed": True,
    }

    if args.diagnostic_checkpoint_step:
        if not gradient_trace_path.is_file():
            raise RuntimeError(
                f"Missing declared gradient trace: {gradient_trace_path}"
            )
        gradient_trace_rows = [
            json.loads(line)
            for line in gradient_trace_path.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]
        gradient_trace_steps = [
            int(row["optimizer_step"]) for row in gradient_trace_rows
        ]
        if gradient_trace_steps != completed_diagnostic_steps:
            raise RuntimeError(
                "Gradient trace steps do not match the completed diagnostic "
                "checkpoint contract: "
                f"trace={gradient_trace_steps}, "
                f"expected={completed_diagnostic_steps}"
            )
        if any(
            row.get("measurement")
            != "after_backward_before_gradient_clip"
            or "model_total" not in row.get("gradient_norms", {})
            for row in gradient_trace_rows
        ):
            raise RuntimeError(
                "Gradient trace is missing the pre-clip model norm contract"
            )
        gradient_trace_audit = {
            "configured": True,
            "path": str(gradient_trace_path),
            "sha256": _sha256(gradient_trace_path),
            "records": len(gradient_trace_rows),
            "optimizer_steps": gradient_trace_steps,
            "gradient_norm_keys": sorted(
                {
                    key
                    for row in gradient_trace_rows
                    for key in row["gradient_norms"]
                }
            ),
            "measurement": "after_backward_before_gradient_clip",
            "passed": True,
        }
    else:
        gradient_trace_audit = {
            "configured": False,
            "optimizer_steps": [],
            "passed": True,
        }

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
                stable_repo / f"scripts/train/{training_method}.py"
            ),
            "training_entry_sha256": _sha256(
                stable_repo / f"scripts/train/{training_method}.py"
            ),
            "logger_entry": str(
                stable_repo / "stable_worldmodel/loggers.py"
            ),
            "logger_entry_sha256": _sha256(
                stable_repo / "stable_worldmodel/loggers.py"
            ),
        },
        "logger": {
            **OmegaConf.to_container(
                cfg.contextworld_logger,
                resolve=True,
            ),
            "initialized": experiment_logger is not None,
            "local_loss_trace": str(loss_trace_path),
            "local_gradient_trace": (
                str(gradient_trace_path)
                if args.diagnostic_checkpoint_step
                else None
            ),
            "training_rng_preserved_during_initialization": True,
        },
        "model": {
            "class": f"{type(model).__module__}.{type(model).__name__}",
            "training_method": training_method,
            "training_objective": training_objective,
            "parameters": sum(value.numel() for value in model.parameters()),
            "action_block": sequence_contract[
                "raw_steps_per_action_block"
            ],
            "history_size": sequence_contract["history_tokens"],
            "num_preds": sequence_contract["num_preds"],
            "input_boundary": sample_contract[
                "collated_batch_audit"
            ],
        },
        "initialization_checkpoint": initialization_audit,
        "frozen_model_modules": frozen_module_audit,
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
            "cuda_memory": cuda_memory_audit.report,
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
            "checkpoint_save_canonicalization": (
                rng_checkpoint.checkpoint_save_canonicalization
            ),
            "resume_policy": args.resume_policy,
            "training_complete": training_complete,
            "intentional_stop_after_optimizer_step": (
                args.stop_after_optimizer_step
            ),
            "limit_train_batches": args.limit_train_batches,
            "limit_val_batches": args.limit_val_batches,
            "external_callbacks_disabled": True,
            "loss_trace": loss_trace_audit,
            "module_gradient_trace": gradient_trace_audit,
            "diagnostic_checkpoints": diagnostic_checkpoint_audit,
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
        "pre_batch_storage_revalidation": (
            passage_storage_contract.report
            if passage_storage_contract.required
            else {"required": False, "passed": True}
        ),
        "distributed_passage_audit": (
            {
                "required": True,
                "optimization": "disabled_per_rank_full_audit",
                "full_logical_audit_execution_count": int(
                    trainer.world_size
                ),
                "storage_revalidation_execution_count": (
                    passage_storage_contract.report[
                        "storage_revalidation_execution_count"
                    ]
                ),
                "every_rank_executed_full_logical_audit": True,
                "attested_view_rank_count": 0,
                "rank_receipts_before_first_batch": (
                    passage_storage_contract.report["rank_receipts"]
                ),
                "expected_rank_receipts": int(trainer.world_size),
                "rank_coverage_passed": (
                    passage_storage_contract.report[
                        "rank_coverage_passed"
                    ]
                ),
                "all_ranks_accepted_before_first_batch": (
                    passage_storage_contract.report[
                        "all_ranks_accepted_before_first_batch"
                    ]
                ),
                "training_dataloader_reads_before_release": (
                    passage_storage_contract.report[
                        "training_dataloader_reads_before_release"
                    ]
                ),
                "validation_dataloader_reads_before_release": (
                    passage_storage_contract.report[
                        "validation_dataloader_reads_before_release"
                    ]
                ),
                "preflight_sample_contract_reads_per_rank": (
                    pre_release_sample_reads
                ),
                "preflight_sample_contract_reads_are_not_training_reads": True,
                "audit_scheduling_contract": (
                    distributed_execution_contract["audit_scheduling"]
                ),
                "local_full_audit_scheduling": (
                    passage_storage_contract.report[
                        "local_full_audit_scheduling"
                    ]
                ),
                "local_revalidation_scheduling": (
                    passage_storage_contract.report[
                        "local_revalidation_scheduling"
                    ]
                ),
                "training_run_exclusivity": (
                    _training_run_exclusivity_snapshot(args)
                ),
                "train_gate_final": (
                    passage_train_gate.receipt()
                    if passage_train_gate is not None
                    else None
                ),
                "validation_gate_final": (
                    passage_val_gate.receipt()
                    if passage_val_gate is not None
                    else None
                ),
                "internal_attestation_used": False,
                "passed": True,
            }
            if passage_storage_contract.required
            else {"required": False, "passed": True}
        ),
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
            "loss_trace": loss_trace_audit,
            "module_gradient_trace": gradient_trace_audit,
            "diagnostic_checkpoints": diagnostic_checkpoint_audit,
        },
        "save_load_exact": reload_equal,
    }
    if trainer.is_global_zero:
        write_json(args.report.resolve(), report)
    return report


def run(args) -> dict:
    """Hold the passage release read lock from preflight through fit."""

    _reject_internal_passage_environment()
    try:
        benchmark_config = resolve_contextworld_path(
            args.benchmark_config,
            repo_root=REPO_ROOT,
        )
        release_root = hidden_passage_training_release_root(
            benchmark_config,
            repo_root=REPO_ROOT,
            model_id=args.model_id,
        )
        if release_root is None:
            return _run_with_release_lock_held(args)
        args._passage_release_root = release_root
        with hidden_passage_release_lock(
            release_root,
            exclusive=False,
        ) as release_lock:
            args._passage_release_lock = dict(release_lock)
            local_rank_text = os.environ.get("LOCAL_RANK")
            if local_rank_text is None:
                local_rank = 0
            else:
                try:
                    local_rank = int(local_rank_text)
                except ValueError as exc:
                    raise RuntimeError(
                        "LOCAL_RANK must be an integer for the frozen "
                        "single-node passage topology"
                    ) from exc
            if not 0 <= local_rank < int(args.devices):
                raise RuntimeError(
                    "LOCAL_RANK is outside the frozen single-node passage "
                    f"topology: rank={local_rank}, devices={args.devices}"
                )
            child_admission = None
            if local_rank > 0:
                child_admission = (
                    verify_hidden_passage_training_run_parent(
                        release_root
                    )
                )
            early_distributed_contract = (
                _load_distributed_execution_contract(
                    benchmark_config,
                    devices=int(args.devices),
                    passage_model=True,
                    audit_concurrency=args.audit_concurrency,
                )
            )
            rank_cpu_affinity_contract = early_distributed_contract.get(
                "rank_cpu_affinity"
            )
            if local_rank > 0:
                args._passage_training_run_lock = {
                    "role": "lightning_nonzero_local_rank",
                    "coordinator_lock_acquired": False,
                    "coordinator_lock_held_by_local_rank_zero": True,
                    "child_admission": child_admission,
                }
                args._passage_rank_cpu_affinity = (
                    _apply_passage_rank_cpu_affinity(
                        contract=rank_cpu_affinity_contract,
                        local_rank=local_rank,
                        devices=int(args.devices),
                    )
                )
                return _run_with_release_lock_held(args)
            with hidden_passage_training_run_lock(
                release_root
            ) as training_run_lock:
                args._passage_training_run_lock = training_run_lock
                args._passage_rank_cpu_affinity = (
                    _apply_passage_rank_cpu_affinity(
                        contract=rank_cpu_affinity_contract,
                        local_rank=local_rank,
                        devices=int(args.devices),
                    )
                )
                return _run_with_release_lock_held(args)
    finally:
        for name in PASSAGE_INTERNAL_ENVIRONMENT:
            os.environ.pop(name, None)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train a TwoRoom model with a benchmark-selected, pinned "
            "StableWM LeWM or PLDM objective"
        )
    )
    parser.add_argument(
        "--model-id",
        required=True,
        help="Model identifier declared by the selected benchmark config.",
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--profile", choices=tuple(PROFILE_DEFAULTS), default="smoke")
    parser.add_argument(
        "--run-kind",
        choices=("adapter_smoke", "pilot", "confirmation"),
        default=None,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=artifact_path("training/runs", repo_root=REPO_ROOT),
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--benchmark-config",
        type=Path,
        default=REPO_ROOT / "configs/benchmark/tworoom_step1_v1.yaml",
    )
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
        "--audit-concurrency",
        type=int,
        choices=(1, 8),
        default=None,
        help=(
            "Controlled passage audit scheduling override. The selected "
            "safe contract and its source are embedded in the report."
        ),
    )
    parser.add_argument(
        "--lewm-regularizer",
        choices=("sigreg", "visreg"),
        default="sigreg",
        help="Select the single active LeWM marginal regularizer.",
    )
    parser.add_argument(
        "--lewm-sigreg-weight",
        type=float,
        default=0.09,
        help="Weight in prediction + weight * SIGReg.",
    )
    parser.add_argument(
        "--lewm-visreg-weight",
        type=float,
        default=0.09,
        help="Weight in prediction + weight * VISReg.",
    )
    parser.add_argument(
        "--lewm-std-weight",
        type=float,
        default=0.0,
        help=(
            "Optional LeWM VCReg std weight. The root-cause confirmation "
            "candidate uses 18 together with --lewm-cov-weight 12."
        ),
    )
    parser.add_argument(
        "--lewm-cov-weight",
        type=float,
        default=0.0,
        help=(
            "Optional LeWM VCReg covariance weight. Zero preserves the "
            "native LeWM objective."
        ),
    )
    parser.add_argument(
        "--diagnostic-checkpoint-step",
        action="append",
        type=int,
        default=[],
        help=(
            "Save an additional read-only-analysis model checkpoint after "
            "this optimizer step. May be repeated."
        ),
    )
    parser.add_argument(
        "--logger-backend",
        choices=("none", "swanlab", "wandb"),
        default="none",
        help=(
            "Experiment logger. The shell launcher defaults to swanlab; "
            "direct Python calls remain offline unless explicitly enabled."
        ),
    )
    parser.add_argument("--swanlab-project", default=None)
    parser.add_argument("--swanlab-workspace", default=None)
    parser.add_argument("--swanlab-experiment-name", default=None)
    parser.add_argument("--swanlab-id", default=None)
    parser.add_argument("--swanlab-logdir", default=None)
    parser.add_argument("--swanlab-mode", default=None)
    parser.add_argument(
        "--swanlab-collect-hardware",
        action="store_true",
    )
    parser.add_argument(
        "--swanlab-hardware-monitor",
        action="store_true",
    )
    parser.add_argument(
        "--swanlab-log-hyperparams",
        action="store_true",
    )
    parser.add_argument(
        "--initialization-checkpoint",
        type=Path,
        default=None,
        help=(
            "Optional model-only initialization checkpoint. When the "
            "benchmark config declares one, this override must resolve to "
            "the same file."
        ),
    )
    parser.add_argument(
        "--initialization-checkpoint-sha256",
        default=None,
        help=(
            "Required SHA-256 when --initialization-checkpoint is supplied. "
            "This initializes model weights only; it is not a resume."
        ),
    )
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
    if (
        args.lewm_sigreg_weight < 0.0
        or args.lewm_visreg_weight < 0.0
        or args.lewm_std_weight < 0.0
        or args.lewm_cov_weight < 0.0
    ):
        parser.error("LeWM regularizer weights must be non-negative")
    if args.lewm_regularizer == "sigreg":
        if args.lewm_sigreg_weight not in {
            0.09,
            0.3,
            0.9,
            1.3,
            1.65,
            2.05,
        }:
            parser.error(
                "Controlled SIGReg sweep supports weights "
                "0.09, 0.3, 0.9, 1.3, 1.65, or 2.05"
            )
        if args.lewm_visreg_weight != 0.09:
            parser.error(
                "--lewm-visreg-weight may only change when VISReg is active"
            )
    elif args.lewm_sigreg_weight != 0.09:
        parser.error(
            "--lewm-sigreg-weight may only change when SIGReg is active"
        )
    if args.lewm_regularizer == "visreg" and (
        args.lewm_std_weight != 0.0 or args.lewm_cov_weight != 0.0
    ):
        parser.error("VISReg screen cannot be combined with VCReg probes")
    if (args.lewm_std_weight, args.lewm_cov_weight) not in {
        (0.0, 0.0),
        (18.0, 12.0),
    }:
        parser.error(
            "The controlled runner supports only native LeWM weights 0/0 "
            "or the diagnosed std/cov candidate 18/12"
        )
    if (
        (args.lewm_std_weight, args.lewm_cov_weight) == (18.0, 12.0)
        and (
            args.lewm_regularizer != "sigreg"
            or args.lewm_sigreg_weight != 0.09
        )
    ):
        parser.error(
            "The std/cov mechanism probe must retain native SIGReg at 0.09"
        )
    args.diagnostic_checkpoint_step = sorted(
        set(args.diagnostic_checkpoint_step)
    )
    if any(
        step <= 0 or step > args.expected_optimizer_steps
        for step in args.diagnostic_checkpoint_step
    ):
        parser.error(
            "Diagnostic checkpoint steps must be within the optimizer "
            "budget"
        )
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
