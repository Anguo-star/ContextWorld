from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

from contextworld.evaluation.hidden_passage_h3_data import (
    hidden_passage_training_run_lock,
    hidden_passage_training_run_lock_path,
)
from contextworld.evaluation.hidden_passage_validation import file_sha256
from scripts.analyze_tworoom_hidden_passage_h3 import (
    _attribution_checks,
    aggregate_validation_results,
)
from scripts.eval_tworoom_hidden_passage_h3_latent import (
    AUDIT_SCHEDULING_CONTRACT,
    TRAINING_RUN_EXCLUSIVITY_CONTRACT,
    _audit_distributed_passage_training,
    _audit_passage_ddp_timeout,
    _audit_training_run_exclusivity,
    validate_training_provenance,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs/benchmark/tworoom_hidden_passage_h3_validation_v2.yaml"
)


def _config() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def _required_identities(config: dict) -> list[tuple[str, int]]:
    return sorted(
        (str(model_id), int(seed))
        for model_id, seeds in config["comparison"][
            "required_results"
        ].items()
        for seed in seeds
    )


def _safe_distributed_audit_fixture(
    world_size: int = 2,
) -> tuple[dict, dict]:
    receipts = [
        {
            "rank": rank,
            "passed": 1,
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
            "full_audit_wait_milliseconds": rank * 10,
            "full_audit_hold_milliseconds": 100,
            "full_audit_sample_contract_reads": 8,
            "revalidation_lock_acquired": 1,
            "revalidation_lock_released": 1,
            "revalidation_release_shared_held": 1,
            "revalidation_collective_unlocked": 1,
            "revalidation_path_identity_verified": 1,
            "revalidation_path_identity_verified_after_acquire": 1,
            "revalidation_descriptor_noninheritable": 1,
            "revalidation_fork_child_close_registered": 1,
            "revalidation_wait_milliseconds": rank * 5,
            "revalidation_hold_milliseconds": 50,
        }
        for rank in range(world_size)
    ]
    audit = {
        "required": True,
        "passed": True,
        "optimization": "disabled_per_rank_full_audit",
        "every_rank_executed_full_logical_audit": True,
        "full_logical_audit_execution_count": world_size,
        "storage_revalidation_execution_count": world_size,
        "attested_view_rank_count": 0,
        "internal_attestation_used": False,
        "expected_rank_receipts": world_size,
        "rank_coverage_passed": True,
        "all_ranks_accepted_before_first_batch": True,
        "training_dataloader_reads_before_release": 0,
        "validation_dataloader_reads_before_release": 0,
        "rank_receipts_before_first_batch": receipts,
        "preflight_sample_contract_reads_per_rank": 8,
        "preflight_sample_contract_reads_are_not_training_reads": True,
        "audit_scheduling_contract": AUDIT_SCHEDULING_CONTRACT,
        "train_gate_final": {
            "released": True,
            "pre_release_calls": 0,
            "pre_release_items": 0,
            "post_release_items": 16,
        },
        "validation_gate_final": {
            "released": True,
            "pre_release_calls": 0,
            "pre_release_items": 0,
            "post_release_items": 4,
        },
    }
    report = {
        "sample_contract": {"sample_count": 8},
        "distributed_passage_audit": audit,
        "pre_batch_storage_revalidation": {
            "passed": True,
            "optimization": "disabled_per_rank_full_audit",
            "rank_coverage_passed": True,
            "gates_opened_only_after_consensus": True,
            "training_dataloader_reads_before_release": 0,
            "validation_dataloader_reads_before_release": 0,
            "rank_receipts": receipts,
        },
    }
    checkpoint_data = {
        "distributed_passage_audit": {
            "required": True,
            "passed": True,
            "optimization": "disabled_per_rank_full_audit",
            "process_mode": "full",
            "full_logical_audit_executed_in_this_process": True,
            "rank0_attestation_required": False,
            "rank0_attestation_used": False,
        }
    }
    return report, checkpoint_data


def test_eval_provenance_requires_safe_distributed_full_audit() -> None:
    report, checkpoint_data = _safe_distributed_audit_fixture()
    observed = _audit_distributed_passage_training(
        report=report,
        checkpoint_data=checkpoint_data,
        world_size=2,
    )
    assert observed["passed"]
    assert observed["world_size"] == 2

    corruptions = []
    value = copy.deepcopy(report)
    del value["distributed_passage_audit"]["optimization"]
    corruptions.append((value, checkpoint_data))
    value = copy.deepcopy(report)
    value["distributed_passage_audit"][
        "full_logical_audit_execution_count"
    ] = 1
    corruptions.append((value, checkpoint_data))
    value = copy.deepcopy(report)
    value["distributed_passage_audit"][
        "rank_receipts_before_first_batch"
    ][1]["train_pre_release_items"] = 1
    corruptions.append((value, checkpoint_data))
    value = copy.deepcopy(report)
    value["distributed_passage_audit"]["train_gate_final"][
        "pre_release_calls"
    ] = 1
    corruptions.append((value, checkpoint_data))
    value = copy.deepcopy(checkpoint_data)
    value["distributed_passage_audit"][
        "rank0_attestation_used"
    ] = True
    corruptions.append((report, value))

    for corrupted_report, corrupted_data in corruptions:
        with pytest.raises(ValueError):
            _audit_distributed_passage_training(
                report=corrupted_report,
                checkpoint_data=corrupted_data,
                world_size=2,
            )


def test_eval_provenance_requires_frozen_passage_ddp_timeout() -> None:
    contract = {
        "rendezvous_timeout_seconds_declared": 7200,
        "rendezvous_timeout_scope": "passage_multi_gpu_only",
        "rendezvous_timeout_seconds_applied": 7200,
        "rendezvous_timeout_override_applied": True,
        "transport_configuration": (
            "framework_defaults_with_frozen_rendezvous_timeout"
        ),
        "transport_overrides_applied": False,
        "audit_scheduling": AUDIT_SCHEDULING_CONTRACT,
        "training_run_exclusivity": (
            TRAINING_RUN_EXCLUSIVITY_CONTRACT
        ),
    }
    assert _audit_passage_ddp_timeout(
        runtime_contract=contract,
        checkpoint_contract=copy.deepcopy(contract),
    )["passed"]

    for key, value in (
        ("rendezvous_timeout_seconds_declared", 1800),
        ("rendezvous_timeout_seconds_applied", None),
        ("rendezvous_timeout_override_applied", False),
    ):
        corrupted = {**contract, key: value}
        with pytest.raises(ValueError, match="timeout provenance"):
            _audit_passage_ddp_timeout(
                runtime_contract=corrupted,
                checkpoint_contract=corrupted,
            )
    corrupted = copy.deepcopy(contract)
    corrupted["training_run_exclusivity"]["maximum_concurrency"] = 2
    with pytest.raises(ValueError, match="timeout provenance"):
        _audit_passage_ddp_timeout(
            runtime_contract=corrupted,
            checkpoint_contract=corrupted,
        )


def _root_run_receipt(release_root: Path) -> dict:
    return {
        "protocol": "contextworld.hidden_passage_h3.training_run_lock.v1",
        "policy": "one_root_training_run_per_release",
        "maximum_concurrency": 1,
        "path": str(
            hidden_passage_training_run_lock_path(release_root)
        ),
        "blocking": False,
        "acquired": True,
        "wait_seconds": 0.125,
        "hold_seconds": None,
        "path_identity_verified": True,
        "path_identity_verified_after_acquire": True,
        "descriptor_inheritable": False,
        "fork_child_close_registered": True,
        "holder_pid": 12345,
        "holder_pid_written": True,
        "released": False,
        "held_through_report_snapshot": True,
    }


def test_training_run_exclusivity_provenance_rejects_each_tamper(
    tmp_path: Path,
) -> None:
    release = tmp_path / "formal"
    release.mkdir()
    receipt = _root_run_receipt(release)
    report = {
        "distributed_passage_audit": {
            "training_run_exclusivity": copy.deepcopy(receipt)
        }
    }
    checkpoint_data = {
        "distributed_passage_audit": {
            "training_run_exclusivity": copy.deepcopy(receipt)
        }
    }
    observed = _audit_training_run_exclusivity(
        report=report,
        checkpoint_data=checkpoint_data,
        release_root=release,
        verify_lock_available=True,
    )
    assert observed["passed"] is True
    assert observed["available_for_scoring"] is True

    corruptions = {
        "protocol": "old",
        "policy": "unlocked",
        "maximum_concurrency": 2,
        "blocking": True,
        "acquired": False,
        "wait_seconds": -1.0,
        "hold_seconds": 1.0,
        "path_identity_verified": False,
        "path_identity_verified_after_acquire": False,
        "descriptor_inheritable": True,
        "fork_child_close_registered": False,
        "holder_pid": 1,
        "holder_pid_written": False,
        "released": True,
        "held_through_report_snapshot": False,
        "path": str(tmp_path / "wrong.lock"),
    }
    for field, value in corruptions.items():
        changed = copy.deepcopy(checkpoint_data)
        changed["distributed_passage_audit"][
            "training_run_exclusivity"
        ][field] = value
        with pytest.raises(ValueError):
            _audit_training_run_exclusivity(
                report=None,
                checkpoint_data=changed,
                release_root=release,
                verify_lock_available=False,
            )

    changed_report = copy.deepcopy(report)
    changed_report["distributed_passage_audit"][
        "training_run_exclusivity"
    ]["wait_seconds"] = 9.0
    with pytest.raises(ValueError, match="differs"):
        _audit_training_run_exclusivity(
            report=changed_report,
            checkpoint_data=checkpoint_data,
            release_root=release,
            verify_lock_available=False,
        )


def test_scoring_rejects_an_active_training_root_lock(
    tmp_path: Path,
) -> None:
    release = tmp_path / "formal"
    release.mkdir()
    receipt = _root_run_receipt(release)
    checkpoint_data = {
        "distributed_passage_audit": {
            "training_run_exclusivity": receipt
        }
    }
    with hidden_passage_training_run_lock(release):
        with pytest.raises(ValueError, match="training is active"):
            _audit_training_run_exclusivity(
                report=None,
                checkpoint_data=checkpoint_data,
                release_root=release,
                verify_lock_available=True,
            )


def test_attribution_requires_mixed_three_of_three_and_negative_controls() -> None:
    config = _config()
    passed = {
        identity: False for identity in _required_identities(config)
    }
    for seed in (3072, 4096, 5120):
        passed[("H3_Passage_MixedRules", seed)] = True
    passed[("H3_Passage_PassableOnly", 3072)] = True

    checks = _attribution_checks(passed, config)
    assert all(checks.values())

    for seed in (3072, 4096, 5120):
        passed[("H3_Passage_PassableOnly", seed)] = True
    checks = _attribution_checks(passed, config)
    assert not checks[
        "passable_only_family_does_not_pass_all_three_seeds"
    ]


def test_analyzer_rejects_an_incomplete_result_matrix(tmp_path: Path) -> None:
    config = _config()
    path = tmp_path / "one.json"
    path.write_text("{}", encoding="utf-8")
    result = {
        "model_id": "H3_Original_LEWM",
        "training_seed": 3072,
    }

    with pytest.raises(ValueError, match="Result matrix is incomplete"):
        aggregate_validation_results(
            results=[result],
            paths=[path],
            config=config,
            config_path=CONFIG,
            expected_catalog_sha256="a" * 64,
        )


def test_analyzer_rejects_mismatched_normalizer_before_comparison(
    tmp_path: Path,
) -> None:
    config = _config()
    catalog_hash = "a" * 64
    results = []
    paths = []
    for index, (model_id, seed) in enumerate(
        _required_identities(config)
    ):
        path = tmp_path / f"result-{index}.json"
        path.write_text("{}", encoding="utf-8")
        paths.append(path)
        results.append(
            {
                "status": "completed",
                "benchmark": config["benchmark"],
                "model_id": model_id,
                "training_seed": seed,
                "identity": {
                    "config_sha256": file_sha256(CONFIG),
                    "catalog_sha256": catalog_hash,
                    "normalizer_sha256": "wrong",
                },
            }
        )

    with pytest.raises(ValueError, match="Normalizer hash mismatch"):
        aggregate_validation_results(
            results=results,
            paths=paths,
            config=config,
            config_path=CONFIG,
            expected_catalog_sha256=catalog_hash,
        )


def _write_incomplete_passage_run(
    tmp_path: Path,
    *,
    profile: str,
    training_complete: bool,
) -> tuple[Path, Path]:
    run_dir = tmp_path / profile
    run_dir.mkdir()
    checkpoint = run_dir / "weights.pt"
    checkpoint.write_bytes(b"not-loaded-by-provenance-audit")
    checkpoint_config = {
        "wm": {"history_size": 3, "num_preds": 1},
        "data": {"dataset": {"frameskip": 5, "num_steps": 4}},
        "model": {"action_encoder": {"input_dim": 10}},
        "contextworld_benchmark": {
            "model_id": "H3_Passage_MixedRules",
            "profile": profile,
            "training_plan": {
                "training_seed": 3072,
                "optimizer_steps_total": 1024,
                "total_global_sample_draws": 1048576,
                "global_batch_size": 1024,
                "devices": 4,
                "batch_size_per_device": 128,
                "adapter_gradient_accumulation_steps": 2,
                "execution_topology": "4gpu_x_b128_x_accum2",
            },
            "data": {},
        },
    }
    checkpoint_config_path = run_dir / "config.json"
    checkpoint_config_path.write_text(
        json.dumps(checkpoint_config),
        encoding="utf-8",
    )
    report = {
        "passed": True,
        "run_kind": "confirmation",
        "save_load_exact": True,
        "model_id": "H3_Passage_MixedRules",
        "profile": profile,
        "stable_worldmodel": {
            "commit": _config()["stable_worldmodel"]["commit"]
        },
        "training": {
            "training_complete": training_complete,
            "seed_before_model_initialization": 3072,
            "global_step": 1024,
            "expected_optimizer_steps": 1024,
            "world_size": 4,
            "devices": 4,
            "batch_size_per_device": 128,
            "adapter_gradient_accumulation_steps": 2,
            "plan": checkpoint_config["contextworld_benchmark"][
                "training_plan"
            ],
        },
        "artifacts": {
            "pretrained": str(checkpoint),
            "pretrained_sha256": file_sha256(checkpoint),
            "pretrained_config": str(checkpoint_config_path),
            "pretrained_config_sha256": file_sha256(
                checkpoint_config_path
            ),
        },
        "data": {},
    }
    report_path = run_dir / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    return checkpoint, report_path


@pytest.mark.parametrize("profile", ["smoke", "passage_pilot"])
def test_eval_provenance_rejects_nonformal_passage_checkpoint(
    tmp_path: Path,
    profile: str,
) -> None:
    checkpoint, report = _write_incomplete_passage_run(
        tmp_path,
        profile=profile,
        training_complete=True,
    )
    with pytest.raises(ValueError, match="passage_formal is required"):
        validate_training_provenance(
            config=_config(),
            model_id="H3_Passage_MixedRules",
            training_seed=3072,
            checkpoint=checkpoint,
            training_report=report,
        )


def test_eval_provenance_rejects_incomplete_training_report(
    tmp_path: Path,
) -> None:
    checkpoint, report = _write_incomplete_passage_run(
        tmp_path,
        profile="passage_formal",
        training_complete=False,
    )
    with pytest.raises(ValueError, match="Training report is incomplete"):
        validate_training_provenance(
            config=_config(),
            model_id="H3_Passage_MixedRules",
            training_seed=3072,
            checkpoint=checkpoint,
            training_report=report,
        )


def test_eval_provenance_rejects_wrong_formal_topology(
    tmp_path: Path,
) -> None:
    checkpoint, report = _write_incomplete_passage_run(
        tmp_path,
        profile="passage_formal",
        training_complete=True,
    )
    with pytest.raises(
        ValueError,
        match="execution topology mismatch",
    ):
        validate_training_provenance(
            config=_config(),
            model_id="H3_Passage_MixedRules",
            training_seed=3072,
            checkpoint=checkpoint,
            training_report=report,
        )
