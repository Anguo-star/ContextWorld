from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import signal
import socket
import time
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

import scripts.train_tworoom_step1 as train_entry
from contextworld.training.tworoom_data import (
    CATALOG_BY_GROUP,
    _load_frozen_normalizer,
    _load_formal_passage_build_report,
    _load_training_exclusion_manifest,
    _resolve_passage_declared_path,
    _validate_paired_passage_catalogs,
    _verify_passage_shard_logical_content,
)
from contextworld.evaluation.hidden_passage_h3_data import (
    AUDIT_SCHEDULING_LOCK_PROTOCOL,
    PARALLEL_AUDIT_SCHEDULING_LOCK_PROTOCOL,
    LOGICAL_CONTENT_COLUMNS,
    LOGICAL_CONTENT_HASH_KIND,
    SHARD_COMPLETION_PROTOCOL,
    STORAGE_CONTENT_HASH_KIND,
    HiddenPassageShardPlan,
    _publish_hidden_passage_shard_completion,
    hidden_passage_audit_scheduling_lock,
    hidden_passage_audit_scheduling_lock_path,
    hidden_passage_training_run_lock,
    logical_episode_content_hashes,
    logical_shard_content_sha256,
    verify_hidden_passage_training_run_parent,
)
from scripts.train_tworoom_step1 import (
    PARALLEL_RANK_CPU_AFFINITY_CONTRACT,
    PASSAGE_DDP_RENDEZVOUS_TIMEOUT_SECONDS,
    PASSAGE_INTERNAL_ENVIRONMENT,
    PROFILE_DEFAULTS,
    TRAINING_RUN_EXCLUSIVITY_CONTRACT,
    PassageReleaseGatedDataset,
    _apply_passage_rank_cpu_affinity,
    _apply_initialization_checkpoint,
    _apply_profile,
    _build_training_logger_preserving_rng,
    _build_training_plan,
    _configure_training_logger,
    _distributed_passage_full_audit_consensus,
    _initialization_checkpoint_spec,
    _load_distributed_execution_contract,
    _project_lewm_model_batch,
    _reject_internal_passage_environment,
    _sample_contract,
    _state_dict_sha256,
    _trainer_strategy_kwargs,
    _training_objective_spec,
)
from scripts.eval_tworoom_hidden_passage_h3_latent import (
    _audit_formal_build_report,
)


ROOT = Path(__file__).resolve().parents[1]
TRAINING_CONFIG = (
    ROOT
    / "configs/benchmark/tworoom_hidden_passage_h3_training_v1.yaml"
)
TRAINING_RUNNER = ROOT / "scripts/run_h3_hidden_passage_train.sh"
TRAINING_ENTRY = ROOT / "scripts/train_tworoom_step1.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _config() -> dict:
    return yaml.safe_load(TRAINING_CONFIG.read_text(encoding="utf-8"))


class _GlooStrategy:
    def all_gather(self, value):
        gathered = [torch.zeros_like(value) for _ in range(
            torch.distributed.get_world_size()
        )]
        torch.distributed.all_gather(gathered, value)
        return torch.stack(gathered)


class _TinyDataset:
    def __init__(self, length: int = 16) -> None:
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        return torch.tensor(index, dtype=torch.int64)


def _gloo_release_worker(
    rank: int,
    world_size: int,
    init_file: str,
    fail_rank0: bool,
    mismatch_rank1: bool,
    queue,
) -> None:
    # Containerized test runners may forbid hostname/interface discovery even
    # though the loopback interface itself is present.  Pinning Gloo to the
    # local interface keeps this two-process test off the external network.
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    train = PassageReleaseGatedDataset(_TinyDataset(), split="train")
    validation = PassageReleaseGatedDataset(
        _TinyDataset(),
        split="validation",
    )
    local_receipt = {
        "rank": 0 if mismatch_rank1 and rank == 1 else rank,
        "passed": int(not (fail_rank0 and rank == 0)),
        "full_logical_audit_count": 1,
        "storage_revalidation_count": 1,
        "train_pre_release_calls": 0,
        "train_pre_release_items": 0,
        "val_pre_release_calls": 0,
        "val_pre_release_items": 0,
        "internal_environment_clean": 1,
        "full_audit_lock_acquired": 1,
        "full_audit_lock_released": 1,
        "full_audit_release_shared_held": 1,
        "full_audit_collective_unlocked": 1,
        "full_audit_path_identity_verified": 1,
        "full_audit_path_identity_verified_after_acquire": 1,
        "full_audit_descriptor_noninheritable": 1,
        "full_audit_fork_child_close_registered": 1,
        "full_audit_wait_milliseconds": 0,
        "full_audit_hold_milliseconds": 1,
        "full_audit_sample_contract_reads": 8,
        "revalidation_lock_acquired": 1,
        "revalidation_lock_released": 1,
        "revalidation_release_shared_held": 1,
        "revalidation_collective_unlocked": 1,
        "revalidation_path_identity_verified": 1,
        "revalidation_path_identity_verified_after_acquire": 1,
        "revalidation_descriptor_noninheritable": 1,
        "revalidation_fork_child_close_registered": 1,
        "revalidation_wait_milliseconds": 0,
        "revalidation_hold_milliseconds": 1,
    }
    try:
        receipts = _distributed_passage_full_audit_consensus(
            strategy=_GlooStrategy(),
            torch_module=torch,
            device=torch.device("cpu"),
            local_receipt=local_receipt,
            expected_world_size=world_size,
        )
        train.release()
        validation.release()
        value = int(train[0])
        queue.put(
            {
                "rank": rank,
                "passed": True,
                "train_receipt": train.receipt(),
                "validation_receipt": validation.receipt(),
                "value": value,
                "receipts": receipts,
            }
        )
    except Exception as exc:
        queue.put(
            {
                "rank": rank,
                "passed": False,
                "train_receipt": train.receipt(),
                "validation_receipt": validation.receipt(),
                "error": str(exc),
            }
        )
    finally:
        torch.distributed.destroy_process_group()


def _audit_lock_holder(
    release_root: str,
    ready,
    finish,
    queue,
    shared: bool = False,
) -> None:
    with hidden_passage_audit_scheduling_lock(
        release_root,
        shared=shared,
    ) as receipt:
        ready.set()
        finish.wait(timeout=30)
    queue.put(dict(receipt))


def _forking_audit_lock_holder(release_root: str, queue) -> None:
    with hidden_passage_audit_scheduling_lock(release_root):
        child_pid = os.fork()
        if child_pid == 0:
            time.sleep(30)
            os._exit(0)
        queue.put(child_pid)
        while True:
            time.sleep(1)


def _verify_training_parent_worker(release_root: str, queue) -> None:
    try:
        queue.put(
            verify_hidden_passage_training_run_parent(release_root)
        )
    except Exception as exc:
        queue.put({"passed": False, "error": str(exc)})


def _profile_args(profile: str, **overrides) -> Namespace:
    values = {key: None for key in PROFILE_DEFAULTS[profile]}
    values.update(
        {
            "profile": profile,
            "seed": 3072,
            "data_split_seed": 3072,
        }
    )
    values.update(overrides)
    return Namespace(**values)


def _passage_metadata(
    *,
    profile: str,
    raw_clips: int = 7_680,
    hashes_frozen: bool = True,
) -> dict:
    epoch_size = PROFILE_DEFAULTS[profile]["epoch_size"]
    required_hashes = (
        {
            "catalog": "1" * 64,
            "manifest": "2" * 64,
            "synthesis_report": "3" * 64,
        }
        if hashes_frozen
        else {}
    )
    return {
        "formal_build_report_audit": {
            "required": True,
            "path": "/frozen/formal/build_report.json",
            "sha256": "4" * 64,
            "passed": True,
        },
        "epoch_group_counts": {"passage_passable": epoch_size},
        "epoch_group_coverage": {
            "passage_passable": {
                "available_virtual_slots": raw_clips,
                "unique_virtual_slots": raw_clips,
            }
        },
        "groups": {
            "passage_passable": {
                "train_clips_raw": raw_clips,
                "quality_requirements": {
                    "maximum_formal_mean_draws_per_raw_clip": 137.0,
                },
                "static_quality_gates": {"all": True},
                "catalog_split_audit": {
                    "required_artifact_hashes": required_hashes,
                },
            }
        },
    }


def test_training_config_freezes_three_clear_synthetic_only_models() -> None:
    config = _config()

    assert config["status"] == (
        "preregistered_formal_inputs_frozen_before_training"
    )
    assert set(CATALOG_BY_GROUP) >= {
        "passage_passable",
        "passage_blocked",
        "passage_mixed",
    }
    assert config["training_protocol"]["synthetic_only"] is True
    assert config["training_protocol"]["group_sampling"] == {
        "H3_Passage_PassableOnly": {"passage_passable": 1.0},
        "H3_Passage_BlockedOnly": {"passage_blocked": 1.0},
        "H3_Passage_MixedRules": {"passage_mixed": 1.0},
    }
    assert [row["training_groups"] for row in config["models"]] == [
        ["passage_passable"],
        ["passage_blocked"],
        ["passage_mixed"],
    ]
    assert config["training_protocol"]["model_visible_fields"] == [
        "pixels",
        "action",
    ]
    assert config["training_protocol"]["diagnostic_only_fields"] == [
        "proprio"
    ]

    frozen = config["data"]["frozen_normalizer"]
    assert len(frozen["sha256"]) == 64
    assert frozen["statistics_scope"] == (
        "original_9000_train_episodes_only"
    )
    exclusion = config["data"]["training_exclusion_manifest"]
    assert exclusion == {
        "path": (
            "artifacts/evaluation/history3/hidden_passage_validation_v2/"
            "training_exclusion_manifest.json"
        ),
        "sha256": (
            "d732ca66061b7f16436d7897e570da3626957b1bda0a94ab0e7125d98f14eee9"
        ),
        "content_sha256": (
            "c23f079e6e9119fd4320d6209f992ca98b1d81a889c6127ade087a447c2aea0c"
        ),
        "query_count": 300,
    }
    formal_build = config["data"]["formal_build_report"]
    assert formal_build == {
        "path": (
            "artifacts/synthesis/hidden_passage_h3_v1/formal/"
            "build_report.json"
        ),
        "sha256": (
            "bd3bde3da3bc97c67c4f9eb1ed87f4a41b50c25c9709ce6b64c4f9a69b9c556a"
        ),
        "benchmark": (
            "tworoom_hidden_passage_history3_training_data_v1"
        ),
        "scale": "formal",
    }
    initialization = config["training_protocol"][
        "initialization_checkpoint"
    ]
    assert initialization["role"] == (
        "model_weight_initialization_only_not_resume"
    )
    assert len(initialization["sha256"]) == 64
    assert len(initialization["config_sha256"]) == 64

    support = config["passage_support"]
    assert support["passage_passable"] == {
        "train": [1],
        "validation": [1],
    }
    assert support["passage_blocked"] == {
        "train": [0],
        "validation": [0],
    }
    assert support["passage_mixed"] == {
        "train": [0, 1],
        "validation": [0, 1],
    }
    assert set(support["eval_only_door_positions"]) == set(
        range(30, 195, 4)
    )

    quality = config["data_quality"]["groups"]
    expected_counts = {
        "passage_passable": (96, 16, 0, 7_680),
        "passage_blocked": (96, 16, 0, 7_680),
        "passage_mixed": (192, 32, 0, 15_360),
    }
    for group, (train, validation, test, clips) in expected_counts.items():
        row = quality[group]
        assert (
            row["exact_train_scenarios"],
            row["exact_validation_scenarios"],
            row["exact_test_scenarios"],
            row["minimum_raw_train_clips"],
        ) == (train, validation, test, clips)
        assert row["exact_train_clips"] == clips
        assert row["exact_test_clips"] == 0
        assert "expected_by_split" in row["factor_support_contract"]

    assert TRAINING_RUNNER.stat().st_mode & 0o111
    runner = TRAINING_RUNNER.read_text(encoding="utf-8")
    assert "passage_pilot" in runner
    assert "passage_formal" in runner
    assert "smoke-8gpu" in runner
    assert "lewm-std-cov-mixed" in runner
    assert "OBJECTIVE_ARGS=(--lewm-std-weight 18 --lewm-cov-weight 12)" in runner
    assert "lewm-sigreg-0p3-mixed" in runner
    assert "OBJECTIVE_ARGS=(--lewm-sigreg-weight 0.3)" in runner
    assert "lewm-sigreg-0p9-mixed" in runner
    assert "OBJECTIVE_ARGS=(--lewm-sigreg-weight 0.9)" in runner
    assert "lewm-sigreg-1p3-mixed" in runner
    assert "OBJECTIVE_ARGS=(--lewm-sigreg-weight 1.3)" in runner
    assert "lewm-sigreg-1p65-mixed" in runner
    assert "OBJECTIVE_ARGS=(--lewm-sigreg-weight 1.65)" in runner
    assert "lewm-sigreg-2p05-mixed" in runner
    assert "OBJECTIVE_ARGS=(--lewm-sigreg-weight 2.05)" in runner
    assert "lewm-visreg-mixed" in runner
    assert (
        "OBJECTIVE_ARGS=(--lewm-regularizer visreg "
        "--lewm-visreg-weight 0.09)"
    ) in runner
    assert '--diagnostic-checkpoint-step "$diagnostic_step"' in runner
    assert "for diagnostic_step in 1 2 4 8 16 32 64 128 256 512 1024" in runner
    assert "EXTRA_ARGS=(--devices 8 --num-workers 2)" in runner
    assert 'if [[ -n "${LOCAL_RANK+x}" ]]' in runner
    assert "DDP 子进程由 Lightning 创建" in runner
    assert 'logger_backend="${logger_backend:-swanlab}"' in runner
    assert 'swanlab login -k "$SWANLAB_API_KEY"' in runner
    assert '--logger-backend "$logger_backend"' in runner
    assert '--audit-concurrency "$AUDIT_CONCURRENCY"' in runner

    training_entry = TRAINING_ENTRY.read_text(encoding="utf-8")
    assert "class GradientTraceModule(spt.Module)" in training_entry
    assert "after_backward_before_gradient_clip" in training_entry
    assert "weights_step_{step}.pt" in training_entry
    assert '"module_gradient_trace": gradient_trace_audit' in training_entry


def test_lewm_std_cov_objective_is_explicitly_reported() -> None:
    from omegaconf import OmegaConf

    native = OmegaConf.create(
        {
            "loss": {
                "regularizer": "sigreg",
                "sigreg": {"weight": 0.09, "kwargs": {"knots": 17}},
                "visreg": {
                    "weight": 0.09,
                    "kwargs": {"num_projections": 1024},
                },
                "std": {"enabled": False, "weight": 0.0},
                "cov": {"enabled": False, "weight": 0.0},
            }
        }
    )
    candidate = OmegaConf.create(
        {
            "loss": {
                "regularizer": "sigreg",
                "sigreg": {"weight": 0.09, "kwargs": {"knots": 17}},
                "visreg": {
                    "weight": 0.09,
                    "kwargs": {"num_projections": 1024},
                },
                "std": {"enabled": True, "weight": 18.0},
                "cov": {"enabled": True, "weight": 12.0},
            }
        }
    )

    assert _training_objective_spec("lewm", native)["name"] == (
        "native_lewm"
    )
    assert _training_objective_spec("lewm", candidate) == {
        "name": "native_lewm_plus_std_cov",
        "prediction_target_detached": False,
        "prediction_weight": 1.0,
        "representation_regularizer": "sigreg",
        "regularizer_weight": 0.09,
        "regularizer_kwargs": {"knots": 17},
        "sigreg_weight": 0.09,
        "visreg_weight": 0.0,
        "std_enabled": True,
        "std_weight": 18.0,
        "cov_enabled": True,
        "cov_weight": 12.0,
    }


def test_lewm_weight_sweep_and_visreg_objectives_are_explicit() -> None:
    from omegaconf import OmegaConf

    high_sigreg = OmegaConf.create(
        {
            "loss": {
                "regularizer": "sigreg",
                "sigreg": {"weight": 2.05, "kwargs": {"num_proj": 1024}},
                "visreg": {
                    "weight": 0.09,
                    "kwargs": {"num_projections": 1024},
                },
                "std": {"enabled": False, "weight": 0.0},
                "cov": {"enabled": False, "weight": 0.0},
            }
        }
    )
    visreg = OmegaConf.create(
        {
            "loss": {
                "regularizer": "visreg",
                "sigreg": {"weight": 0.09, "kwargs": {"num_proj": 1024}},
                "visreg": {
                    "weight": 0.09,
                    "kwargs": {"num_projections": 1024},
                },
                "std": {"enabled": False, "weight": 0.0},
                "cov": {"enabled": False, "weight": 0.0},
            }
        }
    )

    high_spec = _training_objective_spec("lewm", high_sigreg)
    assert high_spec["name"] == "lewm_sigreg_weight_sweep"
    assert high_spec["representation_regularizer"] == "sigreg"
    assert high_spec["regularizer_weight"] == 2.05
    assert high_spec["sigreg_weight"] == 2.05
    assert high_spec["visreg_weight"] == 0.0

    visreg_spec = _training_objective_spec("lewm", visreg)
    assert visreg_spec["name"] == "lewm_visreg"
    assert visreg_spec["representation_regularizer"] == "visreg"
    assert visreg_spec["regularizer_weight"] == 0.09
    assert visreg_spec["sigreg_weight"] == 0.0
    assert visreg_spec["visreg_weight"] == 0.09


def test_contextworld_logger_uses_stablewm_contract() -> None:
    from omegaconf import OmegaConf

    cfg = OmegaConf.create(
        {
            "logger_backend": "none",
            "swanlab": {
                "enabled": False,
                "collect_hardware": False,
                "hardware_monitor": False,
                "log_hyperparams": False,
                "config": {
                    "project": "stable-wm",
                    "workspace": None,
                    "experiment_name": "old",
                    "id": "old",
                    "logdir": "/tmp/swanlab",
                    "mode": None,
                },
            },
            "wandb": {"enabled": False, "config": {}},
        }
    )
    args = Namespace(
        logger_backend="swanlab",
        run_name="h3_diagnostic_s3072",
        swanlab_project="worldmodels",
        swanlab_workspace="qunteam",
        swanlab_experiment_name=None,
        swanlab_id=None,
        swanlab_logdir=None,
        swanlab_mode=None,
        swanlab_collect_hardware=True,
        swanlab_hardware_monitor=False,
        swanlab_log_hyperparams=True,
    )

    audit = _configure_training_logger(cfg, args)

    assert cfg.logger_backend == "swanlab"
    assert cfg.swanlab.enabled is True
    assert cfg.wandb.enabled is False
    assert cfg.swanlab.config.experiment_name == args.run_name
    assert cfg.swanlab.config.id == args.run_name
    assert audit == {
        "backend": "swanlab",
        "enabled": True,
        "project": "worldmodels",
        "workspace": "qunteam",
        "experiment_name": args.run_name,
        "run_id": args.run_name,
        "logdir": "/tmp/swanlab",
        "mode": None,
        "collect_hardware": True,
        "hardware_monitor": False,
        "log_hyperparams": True,
    }


def test_logger_initialization_preserves_training_rng() -> None:
    import random

    import numpy as np

    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()

    def builder(cfg):
        random.random()
        np.random.random()
        torch.rand(1)
        return {"cfg": cfg}

    logger = _build_training_logger_preserving_rng(
        {"logger_backend": "test"},
        builder=builder,
        torch_module=torch,
    )

    assert logger == {"cfg": {"logger_backend": "test"}}
    assert random.getstate() == python_state
    restored_numpy = np.random.get_state()
    assert restored_numpy[0] == numpy_state[0]
    assert np.array_equal(restored_numpy[1], numpy_state[1])
    assert restored_numpy[2:] == numpy_state[2:]
    assert torch.equal(torch.get_rng_state(), torch_state)


@pytest.mark.parametrize(
    ("profile", "steps", "draws", "warmup"),
    [
        ("passage_pilot", 256, 262_144, 2),
        ("passage_formal", 1_024, 1_048_576, 10),
    ],
)
def test_passage_profiles_use_equal_fixed_logical_budgets(
    profile: str,
    steps: int,
    draws: int,
    warmup: int,
) -> None:
    args = _apply_profile(_profile_args(profile))
    plan = _build_training_plan(
        args,
        _passage_metadata(profile=profile),
    )

    assert plan["devices"] == 8
    assert plan["global_batch_size"] == 1_024
    assert plan["optimizer_steps_total"] == steps
    assert plan["total_global_sample_draws"] == draws
    assert plan["warmup_steps"] == warmup
    assert plan["formal_build_report_gate"] == {
        "required": True,
        "path": "/frozen/formal/build_report.json",
        "sha256": "4" * 64,
        "passed": True,
    }
    assert plan["group_exposure"]["passage_passable"][
        "logical_epoch_budget_draws"
    ] == draws


def test_formal_profile_refuses_unfrozen_synthesis_hashes() -> None:
    args = _apply_profile(_profile_args("passage_formal"))

    with pytest.raises(ValueError, match="data-quality gates"):
        _build_training_plan(
            args,
            _passage_metadata(
                profile="passage_formal",
                hashes_frozen=False,
            ),
        )


def test_eight_gpu_smoke_executes_two_ddp_optimizer_steps() -> None:
    args = _apply_profile(
        _profile_args(
            "smoke",
            devices=8,
            num_workers=2,
        )
    )
    metadata = _passage_metadata(
        profile="smoke",
        raw_clips=100,
        hashes_frozen=False,
    )
    plan = _build_training_plan(args, metadata)

    assert args.devices == 8
    assert args.num_workers == 2
    assert plan["global_batch_size"] == 32
    assert plan["executed_batches_per_rank_per_epoch"] == 2
    assert plan["optimizer_steps_total"] == 2
    assert plan["total_global_sample_draws"] == 64


class _RawHistoryDataset:
    def __len__(self) -> int:
        return 2

    @staticmethod
    def _sample(index: int) -> dict:
        return {
            "pixels": torch.zeros(4, 3, 224, 224),
            "action": torch.zeros(4, 10),
            "proprio": torch.zeros(4, 2),
            "passage.open": torch.tensor(index % 2),
            "pair_id": f"pair-{index}",
        }

    def __getitem__(self, index: int) -> dict:
        return self._sample(index)

    def __getitems__(self, indices: list[int]) -> list[dict]:
        return [self._sample(index) for index in indices]


def test_real_collation_is_projected_to_pixels_and_action_only() -> None:
    audit = _sample_contract(_RawHistoryDataset(), count=2)
    boundary = audit["collated_batch_audit"]

    assert boundary["raw_keys"] == [
        "action",
        "pair_id",
        "passage.open",
        "pixels",
        "proprio",
    ]
    assert boundary["model_boundary_keys"] == ["pixels", "action"]
    assert boundary["privileged_fields_at_model_boundary"] == []
    assert boundary["strict_pixels_action_projection"] is True
    assert boundary["model_boundary_shapes"] == {
        "pixels": [2, 4, 3, 224, 224],
        "action": [2, 4, 10],
    }

    projected = _project_lewm_model_batch(
        {
            "pixels": torch.zeros(1, 4, 3, 224, 224),
            "action": torch.zeros(1, 4, 10),
            "passage.open": torch.ones(1),
            "proprio": torch.zeros(1, 4, 2),
        }
    )
    assert tuple(projected) == ("pixels", "action")


def test_frozen_normalizer_requires_exact_bytes_and_split_identity(
    tmp_path: Path,
) -> None:
    payload = {
        "protocol": "tworoom_original_train_s3072_unbiased_zscore_v1",
        "statistics_scope": "original_9000_train_episodes_only",
        "train_episode_ids_sha256": "split-hash",
        "source_sha256": "source-hash",
        "columns": {
            "action": {
                "mean": [1.0, 2.0],
                "std_unbiased": [3.0, 4.0],
                "valid_rows": 10,
            },
            "proprio": {
                "mean": [5.0, 6.0],
                "std_unbiased": [7.0, 8.0],
                "valid_rows": 10,
            },
        },
    }
    path = tmp_path / "normalizer.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    specification = {
        "path": str(path),
        "sha256": _sha256(path),
        "protocol": payload["protocol"],
        "statistics_scope": payload["statistics_scope"],
    }

    scalers, audit = _load_frozen_normalizer(
        specification,
        repo_root=tmp_path,
        split_metadata={"train_episode_ids_sha256": "split-hash"},
    )
    assert audit["passed"]
    assert scalers["action"].mean.tolist() == [[1.0, 2.0]]
    assert scalers["proprio"].std.tolist() == [[7.0, 8.0]]

    with pytest.raises(ValueError, match="split differ"):
        _load_frozen_normalizer(
            specification,
            repo_root=tmp_path,
            split_metadata={"train_episode_ids_sha256": "other"},
        )
    with pytest.raises(ValueError, match="hash mismatch"):
        _load_frozen_normalizer(
            {**specification, "sha256": "0" * 64},
            repo_root=tmp_path,
            split_metadata={"train_episode_ids_sha256": "split-hash"},
        )


@pytest.mark.parametrize("schema_version", [1, 2])
def test_training_exclusion_manifest_requires_both_hashes_and_exact_doors(
    tmp_path: Path,
    schema_version: int,
) -> None:
    records = [
        {
            "query_id": f"query-{index}",
            "template_id": f"template-{index}",
        }
        for index in range(3)
    ]
    payload = {
        "schema_version": schema_version,
        "content_manifest_sha256": "content-hash",
        "query_count": 3,
        "query_records": records,
        "eval_only_door_positions": [30, 34],
    }
    path = tmp_path / "training_exclusion_manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    specification = {
        "path": str(path),
        "sha256": _sha256(path),
        "content_sha256": "content-hash",
        "query_count": 3,
    }

    audit = _load_training_exclusion_manifest(
        specification,
        repo_root=tmp_path,
        expected_eval_only_door_positions=[30, 34],
    )
    assert audit["passed"]
    assert audit["unique_query_ids"] == 3
    assert audit["unique_template_ids"] == 3

    with pytest.raises(ValueError, match="content hash mismatch"):
        _load_training_exclusion_manifest(
            {**specification, "content_sha256": "wrong"},
            repo_root=tmp_path,
            expected_eval_only_door_positions=[30, 34],
        )
    with pytest.raises(ValueError, match="door support differs"):
        _load_training_exclusion_manifest(
            specification,
            repo_root=tmp_path,
            expected_eval_only_door_positions=[30, 38],
        )


def _formal_build_report_fixture(
    tmp_path: Path,
) -> tuple[dict, Path]:
    artifact_root = tmp_path / "formal"
    counts = {
        "passage_passable": {
            "train": {"shards": 1, "episodes": 2, "clips": 2},
            "val": {"shards": 1, "episodes": 1, "clips": 1},
            "test": {"shards": 0, "episodes": 0, "clips": 0},
        },
        "passage_blocked": {
            "train": {"shards": 1, "episodes": 2, "clips": 2},
            "val": {"shards": 1, "episodes": 1, "clips": 1},
            "test": {"shards": 0, "episodes": 0, "clips": 0},
        },
        "passage_mixed": {
            "train": {"shards": 2, "episodes": 4, "clips": 4},
            "val": {"shards": 2, "episodes": 2, "clips": 2},
            "test": {"shards": 0, "episodes": 0, "clips": 0},
        },
    }
    artifacts = {}
    catalogs = {}
    for group in (
        "passage_passable",
        "passage_blocked",
        "passage_mixed",
    ):
        group_files = {}
        for name, directory, suffix in (
            ("catalog", "catalogs", ".json"),
            ("manifest", "manifests", ".jsonl"),
            ("synthesis_report", "reports", ".json"),
        ):
            path = artifact_root / directory / f"{group}{suffix}"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"{group}:{name}\n", encoding="utf-8")
            group_files[name] = path
        catalogs[group] = str(group_files["catalog"])
        artifacts[group] = {
            "catalog": str(group_files["catalog"]),
            "catalog_sha256": _sha256(group_files["catalog"]),
            "manifest": str(group_files["manifest"]),
            "manifest_sha256": _sha256(group_files["manifest"]),
            "synthesis_report": str(group_files["synthesis_report"]),
            "synthesis_report_sha256": _sha256(
                group_files["synthesis_report"]
            ),
            "counts": counts[group],
        }

    required_checks = {
        name: True
        for name in (
            "all_shards_pass",
            "frozen_validation_exclusion_passes",
            "pair_and_split_audit_passes",
            "three_catalogs_are_same_source",
            "catalog_counts_are_exact",
            "model_columns_are_pixels_and_action_only",
            "catalogs_are_synthetic_only",
            "every_episode_is_exactly_one_h3_clip",
            "no_unreferenced_lance_shards",
            "all_shards_have_valid_completion_markers",
            "no_unreferenced_completion_markers",
        )
    }
    report = {
        "schema_version": 1,
        "benchmark": "fixture_hidden_passage_build",
        "scale": "formal",
        "status": "passed",
        "passed": True,
        "checks": required_checks,
        "identity": {"stable_worldmodel_commit": "stable-commit"},
        "validation_exclusion_audit": {
            "passed": True,
            "selected_query_count": 3,
            "selected_query_pixel_hash_overlap": [],
        },
        "artifacts_by_group": artifacts,
        "history3": {"rows_per_episode": 20},
        "physical_shards": 4,
        "physical_episodes": 6,
        "physical_rows": 120,
    }
    path = artifact_root / "build_report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    config = {
        "stable_worldmodel": {"commit": "stable-commit"},
        "data": {
            "formal_build_report": {
                "path": str(path),
                "sha256": _sha256(path),
                "benchmark": "fixture_hidden_passage_build",
                "scale": "formal",
            },
            "training_exclusion_manifest": {"query_count": 3},
            "catalogs": catalogs,
        },
        "data_quality": {
            "groups": {
                "passage_passable": {
                    "exact_train_scenarios": 1,
                    "exact_validation_scenarios": 1,
                    "exact_test_scenarios": 0,
                    "exact_train_clips": 2,
                    "exact_validation_clips": 1,
                    "exact_test_clips": 0,
                },
                "passage_blocked": {
                    "exact_train_scenarios": 1,
                    "exact_validation_scenarios": 1,
                    "exact_test_scenarios": 0,
                    "exact_train_clips": 2,
                    "exact_validation_clips": 1,
                    "exact_test_clips": 0,
                },
                "passage_mixed": {
                    "exact_train_scenarios": 2,
                    "exact_validation_scenarios": 2,
                    "exact_test_scenarios": 0,
                    "exact_train_clips": 4,
                    "exact_validation_clips": 2,
                    "exact_test_clips": 0,
                },
            }
        },
    }
    return config, path


def test_formal_build_report_is_frozen_and_must_have_passed(
    tmp_path: Path,
) -> None:
    config, path = _formal_build_report_fixture(tmp_path)
    source_payload = json.loads(path.read_text(encoding="utf-8"))
    assert "artifacts_by_group" in source_payload
    assert "active_artifacts" not in source_payload
    assert "physical_counts" not in source_payload
    audit = _load_formal_passage_build_report(
        config,
        repo_root=tmp_path,
    )
    assert audit["passed"]
    assert audit["physical_counts"] == {
        "shards": 4,
        "episodes": 6,
        "rows": 120,
    }

    config["data"]["formal_build_report"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="build_report hash mismatch"):
        _load_formal_passage_build_report(config, repo_root=tmp_path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["status"] = "failed"
    payload["passed"] = False
    path.write_text(json.dumps(payload), encoding="utf-8")
    config["data"]["formal_build_report"]["sha256"] = _sha256(path)
    with pytest.raises(ValueError, match="identity/status failed"):
        _load_formal_passage_build_report(config, repo_root=tmp_path)


@pytest.mark.parametrize(
    ("tamper", "message"),
    (
        ("missing_artifacts_by_group", "exactly the three active"),
        ("wrong_group_counts", "group counts differ"),
        ("wrong_artifact_hash", "active artifact hash mismatch"),
        ("wrong_physical_rows", "physical counts differ"),
    ),
)
def test_formal_build_report_rejects_source_structure_tampering(
    tmp_path: Path,
    tamper: str,
    message: str,
) -> None:
    config, path = _formal_build_report_fixture(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if tamper == "missing_artifacts_by_group":
        del payload["artifacts_by_group"]
    elif tamper == "wrong_group_counts":
        payload["artifacts_by_group"]["passage_passable"]["counts"][
            "train"
        ]["episodes"] += 1
    elif tamper == "wrong_artifact_hash":
        payload["artifacts_by_group"]["passage_passable"][
            "catalog_sha256"
        ] = "0" * 64
    elif tamper == "wrong_physical_rows":
        payload["physical_rows"] += 1
    else:  # pragma: no cover - parametrization is exhaustive.
        raise AssertionError(tamper)
    path.write_text(json.dumps(payload), encoding="utf-8")
    config["data"]["formal_build_report"]["sha256"] = _sha256(path)

    with pytest.raises(ValueError, match=message):
        _load_formal_passage_build_report(
            config,
            repo_root=tmp_path,
        )


def test_eval_reconstructs_and_cross_checks_formal_build_receipt(
    tmp_path: Path,
) -> None:
    config, _ = _formal_build_report_fixture(tmp_path)
    embedded = _load_formal_passage_build_report(
        config,
        repo_root=tmp_path,
    )
    frozen_artifacts = {
        group: {
            "sha256": {
                name: embedded["active_artifacts"][group][name]["sha256"]
                for name in (
                    "catalog",
                    "manifest",
                    "synthesis_report",
                )
            }
        }
        for group in (
            "passage_passable",
            "passage_blocked",
            "passage_mixed",
        )
    }
    plan = {
        "formal_build_report_gate": {
            "required": True,
            "passed": True,
            "path": embedded["path"],
            "sha256": embedded["sha256"],
        }
    }

    observed = _audit_formal_build_report(
        training_config=config,
        checkpoint_data={"formal_build_report_audit": embedded},
        training_plan=plan,
        formal_artifact_hashes=frozen_artifacts,
    )
    assert observed["passed"] is True
    assert observed["physical_counts"] == {
        "shards": 4,
        "episodes": 6,
        "rows": 120,
    }

    for field in ("active_artifacts", "physical_counts"):
        tampered = json.loads(json.dumps(embedded))
        if field == "active_artifacts":
            tampered[field]["passage_passable"]["catalog"][
                "sha256"
            ] = "0" * 64
        else:
            tampered[field]["rows"] += 1
        with pytest.raises(
            ValueError,
            match="Checkpoint does not embed",
        ):
            _audit_formal_build_report(
                training_config=config,
                checkpoint_data={
                    "formal_build_report_audit": tampered
                },
                training_plan=plan,
                formal_artifact_hashes=frozen_artifacts,
            )

    wrong_frozen = json.loads(json.dumps(frozen_artifacts))
    wrong_frozen["passage_passable"]["sha256"]["catalog"] = "0" * 64
    with pytest.raises(ValueError, match="does not bind every"):
        _audit_formal_build_report(
            training_config=config,
            checkpoint_data={"formal_build_report_audit": embedded},
            training_plan=plan,
            formal_artifact_hashes=wrong_frozen,
        )


def _logical_episode() -> dict[str, torch.Tensor]:
    shapes = {
        "pixels": (20, 3, 8, 8),
        "action": (20, 2),
        "proprio": (20, 2),
        "state": (20, 2),
        "goal_state": (20, 2),
        "terminated": (20, 1),
        "truncated": (20, 1),
        "variation_agent_speed": (20, 1),
        "variation_door_number": (20, 1),
        "variation_door_position": (20, 3),
        "variation_passage_open": (20, 1),
    }
    return {
        name: (
            torch.zeros(shape, dtype=torch.uint8)
            if name == "pixels"
            else torch.zeros(shape, dtype=torch.float32)
        )
        for name, shape in shapes.items()
    }


def test_passage_preflight_rejects_one_lance_row_mutation(
    tmp_path: Path,
) -> None:
    episode = _logical_episode()
    hashes = logical_episode_content_hashes(episode)
    episode_row = {
        "episode_index": 0,
        "template_id": "template-0",
        "rule": "passable",
        **hashes,
    }
    sidecar = tmp_path / "episode.jsonl"
    sidecar.write_text(json.dumps(episode_row) + "\n", encoding="utf-8")
    table_path = tmp_path / "shard.lance"
    table_path.mkdir()
    shard = HiddenPassageShardPlan(
        split="train",
        door_position=49,
        rule="passable",
        pair_id="pair-0",
        fingerprint="f" * 64,
        scenario_id="scenario-0",
        table_path=table_path,
        episode_manifest_path=sidecar,
    )
    completion = _publish_hidden_passage_shard_completion(
        shard=shard,
        episode_rows=[episode_row],
    )
    record = {
        "scenario_id": "scenario-0",
        "fingerprint": "f" * 64,
        "output_path": str(table_path),
        "episode_manifest": str(sidecar),
        "episode_manifest_sha256": _sha256(sidecar),
        "episode_count": 1,
        "clip_count": 1,
        "rows_per_episode": 20,
        "raw_rows": 20,
        "content_sha256_kind": LOGICAL_CONTENT_HASH_KIND,
        "content_sha256": logical_shard_content_sha256([episode_row]),
        "storage_sha256_kind": STORAGE_CONTENT_HASH_KIND,
        "storage_sha256": completion["storage_sha256"],
        "completion_protocol": SHARD_COMPLETION_PROTOCOL,
        "completion_marker": completion["path"],
        "completion_marker_sha256": completion["sha256"],
    }

    dataset = SimpleNamespace(
        lengths=[20],
        load_episode=lambda index: episode,
    )
    swm = SimpleNamespace(
        data=SimpleNamespace(
            LanceDataset=lambda path: dataset,
        )
    )
    audit = _verify_passage_shard_logical_content(
        swm,
        record=record,
        repo_root=tmp_path,
    )
    assert audit["passed"]
    assert audit["columns"] == list(LOGICAL_CONTENT_COLUMNS)

    marker_path = Path(completion["path"])
    original_marker = marker_path.read_bytes()
    marker = json.loads(original_marker)
    marker["status"] = "tampered"
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    with pytest.raises(ValueError, match="completion marker hash mismatch"):
        _verify_passage_shard_logical_content(
            swm,
            record=record,
            repo_root=tmp_path,
        )
    marker_path.write_bytes(original_marker)

    episode["action"][0, 0] = 1.0
    with pytest.raises(ValueError, match="logical content differs"):
        _verify_passage_shard_logical_content(
            swm,
            record=record,
            repo_root=tmp_path,
        )


def test_passage_declared_paths_are_checked_lexically_before_resolve(
    tmp_path: Path,
) -> None:
    release = tmp_path / "formal"
    tables = release / "tables"
    tables.mkdir(parents=True)
    outside = tmp_path / "outside.lance"
    outside.mkdir()

    with pytest.raises(ValueError, match="escapes"):
        _resolve_passage_declared_path(
            outside,
            repo_root=tmp_path,
            release_root=release,
            leaf_kind="directory",
            required_subtree="tables",
        )

    alias = tables / "alias.lance"
    alias.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="observed=symlink"):
        _resolve_passage_declared_path(
            alias,
            repo_root=tmp_path,
            release_root=release,
            leaf_kind="directory",
            required_subtree="tables",
        )


@pytest.mark.parametrize("name", PASSAGE_INTERNAL_ENVIRONMENT)
def test_internal_passage_environment_is_rejected_and_cleared(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> None:
    monkeypatch.setenv(name, "forged")
    with pytest.raises(RuntimeError, match="may not cross"):
        _reject_internal_passage_environment()
    assert name not in os.environ


def test_audit_scheduling_lock_is_exclusive_and_releases_normally(
    tmp_path: Path,
) -> None:
    release = tmp_path / "release"
    release.mkdir()
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    finish = context.Event()
    queue = context.Queue()
    process = context.Process(
        target=_audit_lock_holder,
        args=(str(release), ready, finish, queue),
    )
    process.start()
    assert ready.wait(timeout=10)
    with pytest.raises(BlockingIOError):
        with hidden_passage_audit_scheduling_lock(
            release, blocking=False
        ):
            pass
    finish.set()
    process.join(timeout=10)
    assert process.exitcode == 0
    receipt = queue.get(timeout=5)
    assert receipt["protocol"] == AUDIT_SCHEDULING_LOCK_PROTOCOL
    assert receipt["released"] is True
    assert receipt["descriptor_inheritable"] is False
    assert receipt["path_identity_verified_after_acquire"] is True
    with hidden_passage_audit_scheduling_lock(
        release, blocking=False
    ) as reacquired:
        assert reacquired["acquired"] is True


def test_audit_scheduling_lock_releases_on_exception(
    tmp_path: Path,
) -> None:
    release = tmp_path / "release"
    release.mkdir()
    with pytest.raises(RuntimeError, match="injected"):
        with hidden_passage_audit_scheduling_lock(release):
            raise RuntimeError("injected")
    with hidden_passage_audit_scheduling_lock(
        release, blocking=False
    ) as receipt:
        assert receipt["acquired"] is True


def test_parallel_audit_scheduling_allows_shared_readers_and_blocks_writer(
    tmp_path: Path,
) -> None:
    release = tmp_path / "release"
    release.mkdir()
    context = multiprocessing.get_context("spawn")
    finish = context.Event()
    queue = context.Queue()
    readers = []
    ready_events = []
    for _ in range(2):
        ready = context.Event()
        process = context.Process(
            target=_audit_lock_holder,
            args=(str(release), ready, finish, queue, True),
        )
        process.start()
        readers.append(process)
        ready_events.append(ready)
    assert all(ready.wait(timeout=10) for ready in ready_events)
    with pytest.raises(BlockingIOError):
        with hidden_passage_audit_scheduling_lock(
            release,
            blocking=False,
        ):
            pass
    finish.set()
    for process in readers:
        process.join(timeout=10)
        assert process.exitcode == 0
    receipts = [queue.get(timeout=5) for _ in readers]
    assert all(
        receipt["protocol"]
        == PARALLEL_AUDIT_SCHEDULING_LOCK_PROTOCOL
        and receipt["policy"] == "sibling_shared_flock"
        and receipt["mode"] == "shared"
        and receipt["released"] is True
        for receipt in receipts
    )


def test_audit_scheduling_lock_rejects_symlink_and_hardlink(
    tmp_path: Path,
) -> None:
    release = tmp_path / "release"
    release.mkdir()
    lock_path = hidden_passage_audit_scheduling_lock_path(release)
    target = tmp_path / "target.lock"
    target.write_text("", encoding="utf-8")
    target.chmod(0o600)
    lock_path.symlink_to(target)
    with pytest.raises(ValueError, match="unsafe alias"):
        with hidden_passage_audit_scheduling_lock(release):
            pass
    lock_path.unlink()
    os.link(target, lock_path)
    with pytest.raises(ValueError, match="unsafe"):
        with hidden_passage_audit_scheduling_lock(release):
            pass


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX fork")
def test_fork_child_does_not_keep_audit_lock_after_holder_sigkill(
    tmp_path: Path,
) -> None:
    release = tmp_path / "release"
    release.mkdir()
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    holder = context.Process(
        target=_forking_audit_lock_holder,
        args=(str(release), queue),
    )
    holder.start()
    child_pid = int(queue.get(timeout=10))
    try:
        holder.kill()
        holder.join(timeout=10)
        assert holder.exitcode == -signal.SIGKILL
        with hidden_passage_audit_scheduling_lock(
            release, blocking=False
        ) as receipt:
            assert receipt["acquired"] is True
    finally:
        try:
            os.kill(child_pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def test_nonzero_rank_is_admitted_only_when_direct_parent_holds_root_lock(
    tmp_path: Path,
) -> None:
    release = tmp_path / "release"
    release.mkdir()
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    with hidden_passage_training_run_lock(release) as root_receipt:
        assert root_receipt["holder_pid"] == os.getpid()
        assert root_receipt["holder_pid_written"] is True
        child = context.Process(
            target=_verify_training_parent_worker,
            args=(str(release), queue),
        )
        child.start()
        child.join(timeout=10)
        assert child.exitcode == 0
        admission = queue.get(timeout=5)
        assert admission["passed"] is True
        assert admission["holder_pid"] == os.getpid()
        assert admission["parent_pid"] == os.getpid()
        assert admission["lock_is_held"] is True
        assert admission["descriptor_inheritable"] is False


def test_manual_nonzero_local_rank_cannot_bypass_root_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release = tmp_path / "release"
    release.mkdir()
    config = tmp_path / "benchmark.yaml"
    config.write_text("benchmark: test\n", encoding="utf-8")
    args = Namespace(
        benchmark_config=config,
        model_id="H3_Passage_PassableOnly",
        devices=8,
    )
    entered_training = False

    def forbidden_training(_args):
        nonlocal entered_training
        entered_training = True
        return {}

    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setattr(
        train_entry,
        "resolve_contextworld_path",
        lambda value, repo_root: Path(value),
    )
    monkeypatch.setattr(
        train_entry,
        "hidden_passage_training_release_root",
        lambda *args, **kwargs: release,
    )
    monkeypatch.setattr(
        train_entry,
        "_run_with_release_lock_held",
        forbidden_training,
    )
    with pytest.raises(
        RuntimeError,
        match="no active root training lock",
    ):
        train_entry.run(args)
    assert entered_training is False


def test_passage_ddp_timeout_is_frozen_and_multi_gpu_only(
    tmp_path: Path,
) -> None:
    multi = _load_distributed_execution_contract(
        TRAINING_CONFIG,
        devices=8,
        passage_model=True,
    )
    assert (
        multi["rendezvous_timeout_seconds_declared"]
        == PASSAGE_DDP_RENDEZVOUS_TIMEOUT_SECONDS
    )
    assert (
        multi["rendezvous_timeout_seconds_applied"]
        == PASSAGE_DDP_RENDEZVOUS_TIMEOUT_SECONDS
    )
    assert multi["rendezvous_timeout_override_applied"] is True
    assert (
        multi["transport_configuration"]
        == "framework_defaults_with_frozen_rendezvous_timeout"
    )
    assert multi["transport_overrides_applied"] is False
    assert multi["audit_scheduling_source"] == "benchmark_config"
    assert multi["audit_scheduling"] == {
        "policy": "sibling_exclusive_flock",
        "maximum_concurrency": 1,
        "scope": "per_rank_full_audit_and_fit_start_storage_revalidation",
        "lock_protocol": AUDIT_SCHEDULING_LOCK_PROTOCOL,
        "lock_order": "release_shared_then_audit_exclusive",
        "collective_holds_lock": False,
        "topology_scope": "single_node_8gpu",
        "concurrent_training_runs_per_release": 1,
    }
    assert (
        multi["training_run_exclusivity"]
        == TRAINING_RUN_EXCLUSIVITY_CONTRACT
    )
    parallel = _load_distributed_execution_contract(
        TRAINING_CONFIG,
        devices=8,
        passage_model=True,
        audit_concurrency=8,
    )
    assert parallel["audit_scheduling"] == {
        "policy": "sibling_shared_flock",
        "maximum_concurrency": 8,
        "scope": "per_rank_full_audit_and_fit_start_storage_revalidation",
        "lock_protocol": PARALLEL_AUDIT_SCHEDULING_LOCK_PROTOCOL,
        "lock_order": "release_shared_then_audit_shared",
        "collective_holds_lock": False,
        "topology_scope": "single_node_8gpu",
        "concurrent_training_runs_per_release": 1,
    }
    assert parallel["audit_scheduling_source"] == "controlled_cli_override"
    assert (
        parallel["rank_cpu_affinity"]
        == PARALLEL_RANK_CPU_AFFINITY_CONTRACT
    )
    with pytest.raises(ValueError, match="exactly 1 or 8"):
        _load_distributed_execution_contract(
            TRAINING_CONFIG,
            devices=8,
            passage_model=True,
            audit_concurrency=2,
        )

    class FakeDDPStrategy:
        def __init__(self, *, timeout) -> None:
            self.timeout = timeout

    strategy = _trainer_strategy_kwargs(
        multi,
        ddp_strategy_class=FakeDDPStrategy,
    )["strategy"]
    assert strategy.timeout.total_seconds() == 7200

    single = _load_distributed_execution_contract(
        TRAINING_CONFIG,
        devices=1,
        passage_model=True,
    )
    assert single["rendezvous_timeout_seconds_declared"] == 7200
    assert single["rendezvous_timeout_seconds_applied"] is None
    assert single["rendezvous_timeout_override_applied"] is False
    assert (
        _trainer_strategy_kwargs(
            single,
            ddp_strategy_class=FakeDDPStrategy,
        )
        == {}
    )

    payload = _config()
    del payload["training_protocol"]["distributed_execution"][
        "rendezvous_timeout_seconds"
    ]
    broken = tmp_path / "missing-timeout.yaml"
    broken.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="must be frozen"):
        _load_distributed_execution_contract(
            broken,
            devices=8,
            passage_model=True,
        )

    payload = _config()
    payload["training_protocol"]["distributed_execution"][
        "audit_scheduling"
    ]["maximum_concurrency"] = 2
    broken = tmp_path / "unsafe-audit-concurrency.yaml"
    broken.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="one single-node audit"):
        _load_distributed_execution_contract(
            broken,
            devices=8,
            passage_model=True,
        )

    payload = _config()
    del payload["training_protocol"]["distributed_execution"][
        "training_run_exclusivity"
    ]
    broken = tmp_path / "missing-root-run-lock.yaml"
    broken.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="single-root contract"):
        _load_distributed_execution_contract(
            broken,
            devices=8,
            passage_model=True,
        )


def test_parallel_rank_cpu_affinity_is_disjoint_and_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current = set(range(176))

    monkeypatch.setattr(os, "cpu_count", lambda: 176)
    monkeypatch.setattr(
        os,
        "sched_getaffinity",
        lambda _pid: set(current),
    )

    def apply(_pid: int, values: set[int]) -> None:
        current.clear()
        current.update(values)

    monkeypatch.setattr(os, "sched_setaffinity", apply)
    receipt = _apply_passage_rank_cpu_affinity(
        contract=PARALLEL_RANK_CPU_AFFINITY_CONTRACT,
        local_rank=3,
        devices=8,
    )
    assert receipt == {
        "policy": "local_rank_disjoint_contiguous_from_zero",
        "scope": "full_rank_process",
        "applied_before_stableworldmodel_and_lance_import": True,
        "local_rank": 3,
        "cpus_per_rank": 8,
        "cpu_ids": list(range(24, 32)),
        "host_logical_cpu_count": 176,
        "prior_affinity_cpu_count": 176,
        "passed": True,
    }


def test_real_dataloader_cannot_read_before_release_and_reads_afterward(
) -> None:
    closed = PassageReleaseGatedDataset(
        _TinyDataset(),
        split="train",
    )
    closed_loader = torch.utils.data.DataLoader(
        closed,
        batch_size=4,
        num_workers=0,
    )
    closed_iterator = iter(closed_loader)
    with pytest.raises(RuntimeError, match="before audit consensus"):
        next(closed_iterator)
    closed_receipt = closed.receipt()
    assert closed_receipt["pre_release_calls"] > 0
    assert closed_receipt["pre_release_items"] > 0
    assert closed_receipt["post_release_items"] == 0

    opened = PassageReleaseGatedDataset(
        _TinyDataset(),
        split="train",
    )
    assert opened.receipt()["pre_release_items"] == 0
    opened.release()
    opened_loader = torch.utils.data.DataLoader(
        opened,
        batch_size=4,
        num_workers=0,
    )
    opened_iterator = iter(opened_loader)
    batch = next(opened_iterator)
    opened_receipt = opened.receipt()
    assert batch.tolist() == [0, 1, 2, 3]
    assert opened_receipt is not None
    assert opened_receipt["pre_release_items"] == 0
    assert opened_receipt["post_release_items"] > 0


@pytest.mark.parametrize(
    ("fail_rank0", "mismatch_rank1", "expected_passed"),
    [
        (False, False, True),
        (True, False, False),
        (False, True, False),
    ],
)
def test_two_process_gloo_releases_all_ranks_or_none_before_read(
    tmp_path: Path,
    fail_rank0: bool,
    mismatch_rank1: bool,
    expected_passed: bool,
) -> None:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("127.0.0.1", 0))
    except OSError as error:
        pytest.skip(f"local TCP sockets are unavailable for Gloo: {error}")
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    init_file = tmp_path / (
        f"gloo-{int(fail_rank0)}-{int(mismatch_rank1)}"
    )
    processes = [
        context.Process(
            target=_gloo_release_worker,
            args=(
                rank,
                2,
                str(init_file),
                fail_rank0,
                mismatch_rank1,
                queue,
            ),
        )
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=30)
        assert not process.is_alive(), "gloo gate deadlocked"
        assert process.exitcode == 0
    results = sorted(
        (queue.get(timeout=5) for _ in range(2)),
        key=lambda row: row["rank"],
    )
    assert [row["passed"] for row in results] == [
        expected_passed,
        expected_passed,
    ]
    assert all(
        row["train_receipt"]["pre_release_items"] == 0
        for row in results
    )
    assert all(
        row["validation_receipt"]["pre_release_items"] == 0
        for row in results
    )
    if expected_passed:
        assert all(row["value"] == 0 for row in results)
        assert all(
            row["train_receipt"]["post_release_items"] == 1
            for row in results
        )
        assert all(len(row["receipts"]) == 2 for row in results)
    else:
        assert all(
            not row["train_receipt"]["released"] for row in results
        )


def test_initialization_checkpoint_is_hashed_and_is_not_resume(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "weights.pt"
    checkpoint.write_bytes(b"model-only-checkpoint")
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    benchmark = tmp_path / "benchmark.yaml"
    benchmark.write_text(
        yaml.safe_dump(
            {
                "training_protocol": {
                    "initialization_checkpoint": {
                        "path": str(checkpoint),
                        "sha256": _sha256(checkpoint),
                        "role": "model_weight_initialization_only_not_resume",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    args = Namespace(
        initialization_checkpoint=None,
        initialization_checkpoint_sha256=None,
    )

    specification = _initialization_checkpoint_spec(
        args,
        benchmark_config=benchmark,
    )
    assert specification is not None
    assert specification["hash_audit_passed"]
    assert specification["optimizer_state_loaded"] is False
    assert specification["scheduler_state_loaded"] is False
    assert specification["resume_state_loaded"] is False

    torch.manual_seed(11)
    source = torch.nn.Linear(3, 2)
    torch.manual_seed(12)
    target = torch.nn.Linear(3, 2)
    fake_swm = SimpleNamespace(
        wm=SimpleNamespace(
            utils=SimpleNamespace(
                load_pretrained=lambda path, cache_dir: source
            )
        )
    )
    audit = _apply_initialization_checkpoint(
        target,
        swm=fake_swm,
        specification=specification,
        cache_dir=tmp_path,
        resume_checkpoint=None,
    )
    assert audit["applied"] is True
    assert audit["state_exact"] is True
    assert _state_dict_sha256(target) == _state_dict_sha256(source)

    resumed = _apply_initialization_checkpoint(
        torch.nn.Linear(3, 2),
        swm=fake_swm,
        specification=specification,
        cache_dir=tmp_path,
        resume_checkpoint=tmp_path / "last.ckpt",
    )
    assert resumed["applied"] is False
    assert resumed["reason"] == (
        "full_state_resume_supersedes_initialization"
    )


def test_state_hash_supports_scalar_and_bfloat16_state() -> None:
    module = torch.nn.Module()
    module.register_parameter(
        "weight",
        torch.nn.Parameter(torch.ones(2, dtype=torch.bfloat16)),
    )
    module.register_buffer("step", torch.tensor(256, dtype=torch.int64))

    first = _state_dict_sha256(module)
    second = _state_dict_sha256(module)
    assert first == second
    assert len(first) == 64


def _write_catalog(
    root: Path,
    name: str,
    paths_by_split: dict[str, list[Path]],
    manifest_rows: list[dict],
) -> Path:
    catalogs = root / "catalogs"
    manifests = root / "manifests"
    catalogs.mkdir(parents=True, exist_ok=True)
    manifests.mkdir(parents=True, exist_ok=True)
    catalog = catalogs / f"{name}.json"
    catalog.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "train": {
                    "synthetic": [
                        str(path) for path in paths_by_split["train"]
                    ]
                },
                "val": {
                    "synthetic": [
                        str(path) for path in paths_by_split["val"]
                    ]
                },
                "ood_test": {
                    "synthetic": [
                        str(path) for path in paths_by_split["test"]
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    (manifests / f"{name}.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in manifest_rows),
        encoding="utf-8",
    )
    return catalog


def _paired_catalog_fixture(tmp_path: Path) -> tuple[dict, Path]:
    artifact = tmp_path / "formal"
    paths = {
        "passable": {"train": [], "val": [], "test": []},
        "blocked": {"train": [], "val": [], "test": []},
    }
    manifests = {"passable": [], "blocked": []}
    split_doors = {"train": [31], "val": [35], "test": []}
    for split, doors in split_doors.items():
        for door in doors:
            pair_id = f"hp-{split}-door{door:03d}"
            for rule, value in (("passable", 1), ("blocked", 0)):
                output = artifact / "data" / f"{pair_id}-{rule}"
                output.mkdir(parents=True)
                episode_manifest = (
                    artifact
                    / "episodes"
                    / f"{pair_id}-{rule}.jsonl"
                )
                episode_manifest.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )
                episode_manifest.write_text(
                    json.dumps(
                        {
                            "template_id": f"template-{pair_id}",
                            "pair_id": pair_id,
                            "rule": rule,
                            "passage_open": value,
                            "action_sha256": f"action-{pair_id}",
                            "initial_pixels_sha256": (
                                f"initial-{pair_id}"
                            ),
                            "query_pixels_sha256": f"query-{pair_id}",
                            "future_pixels_sha256": (
                                f"future-{pair_id}-{rule}"
                            ),
                            "model_input_keys": ["pixels", "action"],
                            "passed": True,
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
                paths[rule][split].append(output)
                manifests[rule].append(
                    {
                        "split": split,
                        "regime": (
                            "validation_id"
                            if split == "val"
                            else f"{split}_paired"
                        ),
                        "env_id": "contextworld/TwoRoomHiddenPassage-v1",
                        "env_seed": door,
                        "policy_seed": 20260723,
                        "episodes": 1,
                        "task": "tworoom_hidden_passage",
                        "max_episode_steps": 20,
                        "image_shape": [224, 224],
                        "reset_constraints": {"door": door},
                        "pixel_codec": {
                            "format": "png",
                            "compress_level": 1,
                            "lossless": True,
                        },
                        "seed_group": pair_id,
                        "pair_id": pair_id,
                        "episode_manifest": str(episode_manifest),
                        "episode_manifest_sha256": _sha256(
                            episode_manifest
                        ),
                        "episode_count": 1,
                        "clip_count": 1,
                        "rows_per_episode": 20,
                        "raw_rows": 20,
                        "content_sha256": f"content-{pair_id}-{rule}",
                        "factors": {
                            "door.position": door,
                            "passage.open": value,
                        },
                    }
                )

    passable = _write_catalog(
        artifact,
        "passable",
        paths["passable"],
        manifests["passable"],
    )
    blocked = _write_catalog(
        artifact,
        "blocked",
        paths["blocked"],
        manifests["blocked"],
    )
    mixed_paths = {
        split: paths["passable"][split] + paths["blocked"][split]
        for split in ("train", "val", "test")
    }
    mixed = _write_catalog(
        artifact,
        "mixed",
        mixed_paths,
        manifests["passable"] + manifests["blocked"],
    )
    config = {
        "data": {
            "catalogs": {
                "passage_passable": str(passable),
                "passage_blocked": str(blocked),
                "passage_mixed": str(mixed),
            }
        },
        "passage_support": {
            "eval_only_door_positions": [30, 34],
        },
        "paired_collection_contract": {
            "passage_rules": {
                "equal_manifest_fields": [
                    "split",
                    "regime",
                    "env_id",
                    "env_seed",
                    "policy_seed",
                    "episodes",
                    "task",
                    "max_episode_steps",
                    "image_shape",
                    "reset_constraints",
                    "pixel_codec",
                    "seed_group",
                    "pair_id",
                ]
            }
        },
    }
    return config, blocked


def test_paired_passage_catalogs_are_exact_union_and_eval_isolated(
    tmp_path: Path,
) -> None:
    config, blocked_catalog = _paired_catalog_fixture(tmp_path)

    audit = _validate_paired_passage_catalogs(
        config,
        repo_root=tmp_path,
    )
    assert audit["passed"]
    assert audit["paired_single_rule_shards"] == 2
    assert audit["episode_sidecars_verified"] == 4
    assert audit["paired_episode_records"] == 2
    assert audit["equal_episode_fields"] == [
        "pair_id",
        "action_sha256",
        "initial_pixels_sha256",
        "query_pixels_sha256",
        "model_input_keys",
        "passed",
    ]
    assert all(
        row["mixed_is_exact_union"]
        for row in audit["split_path_checks"].values()
    )
    assert audit["door_position_isolation"] == {
        "expected_eval_only": [30, 34],
        "observed": {
            "train": [31],
            "validation": [35],
            "test": [],
        },
        "train_and_loader_validation_exclude_eval_only": True,
        "training_catalog_test_is_empty": True,
        "validation_assets_are_managed_separately": True,
        "passed": True,
    }

    manifest = (
        blocked_catalog.parent.parent
        / "manifests"
        / "blocked.jsonl"
    )
    rows = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
    ]
    rows[0]["policy_seed"] += 1
    manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="differ beyond passage.open"):
        _validate_paired_passage_catalogs(
            config,
            repo_root=tmp_path,
        )

    rows[0]["policy_seed"] -= 1
    sidecar = Path(rows[0]["episode_manifest"])
    episode = json.loads(sidecar.read_text(encoding="utf-8"))
    episode["action_sha256"] = "different-action"
    sidecar.write_text(json.dumps(episode) + "\n", encoding="utf-8")
    rows[0]["episode_manifest_sha256"] = _sha256(sidecar)
    manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="episode.action_sha256"):
        _validate_paired_passage_catalogs(
            config,
            repo_root=tmp_path,
        )
