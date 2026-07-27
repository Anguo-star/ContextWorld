from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

import scripts.eval_tworoom_hidden_passage_h3_latent as passage_score
import scripts.run_tworoom_hidden_passage_h3_pipeline as passage_pipeline
from scripts.run_tworoom_hidden_passage_h3_pipeline import (
    AUDIT_SCHEDULING_CONTRACT,
    ORIGINAL_MODEL_ID,
    PASSAGE_INTERNAL_ENVIRONMENT,
    RECIPES,
    TRAINING_SEEDS,
    _run_command,
    dry_run_plan,
    score_jobs,
    training_jobs,
    validate_existing_preflight,
    validate_partial_training_checkpoint,
    validate_static_contract,
)


ROOT = Path(__file__).resolve().parents[1]
VALIDATION_CONFIG_PATH = (
    ROOT
    / "configs/benchmark/tworoom_hidden_passage_h3_validation_v2.yaml"
)
TRAINING_CONFIG_PATH = (
    ROOT
    / "configs/benchmark/tworoom_hidden_passage_h3_training_v1.yaml"
)


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _availability_receipt(
    path: Path,
    *,
    holder_pid: int,
    wait_seconds: float,
    hold_seconds: float,
) -> dict:
    return {
        "protocol": passage_score.TRAINING_RUN_LOCK_PROTOCOL,
        "policy": "one_root_training_run_per_release",
        "maximum_concurrency": 1,
        "path": str(path),
        "blocking": False,
        "acquired": True,
        "wait_seconds": wait_seconds,
        "hold_seconds": hold_seconds,
        "path_identity_verified": True,
        "path_identity_verified_after_acquire": True,
        "descriptor_inheritable": False,
        "fork_child_close_registered": True,
        "holder_pid": holder_pid,
        "holder_pid_written": True,
        "released": True,
    }


def test_two_real_lock_probe_receipts_have_one_stable_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock_path = tmp_path / ".formal.training-run.lock"
    monkeypatch.setattr(passage_score.os, "getpid", lambda: 101)
    first = passage_score._normalize_training_run_availability_receipt(
        _availability_receipt(
            lock_path,
            holder_pid=101,
            wait_seconds=0.00001,
            hold_seconds=0.01,
        ),
        expected_path=lock_path,
    )
    monkeypatch.setattr(passage_score.os, "getpid", lambda: 202)
    second = passage_score._normalize_training_run_availability_receipt(
        _availability_receipt(
            lock_path,
            holder_pid=202,
            wait_seconds=0.00009,
            hold_seconds=0.03,
        ),
        expected_path=lock_path,
    )

    assert passage_score.canonical_sha256(
        first
    ) == passage_score.canonical_sha256(second)
    assert "holder_pid" not in first
    assert "wait_seconds" not in first
    assert "hold_seconds" not in first
    assert first["holder_pid_is_current_process"] is True
    assert first["wait_seconds_nonnegative_and_finite"] is True
    assert first["hold_seconds_nonnegative_and_finite"] is True


def test_lock_probe_normalization_rejects_tampered_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock_path = tmp_path / ".formal.training-run.lock"
    monkeypatch.setattr(passage_score.os, "getpid", lambda: 101)
    receipt = _availability_receipt(
        lock_path,
        holder_pid=101,
        wait_seconds=0.00001,
        hold_seconds=0.01,
    )
    receipt["released"] = False

    with pytest.raises(
        ValueError,
        match="did not release cleanly",
    ):
        passage_score._normalize_training_run_availability_receipt(
            receipt,
            expected_path=lock_path,
        )


def test_runner_rejects_internal_launch_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    name = PASSAGE_INTERNAL_ENVIRONMENT[0]
    monkeypatch.setenv(name, "forged")
    with pytest.raises(RuntimeError, match="refuses internal"):
        _run_command(label="must-not-run", command=["false"])


def test_runner_freezes_three_recipes_by_three_training_seeds() -> None:
    validation = _load(VALIDATION_CONFIG_PATH)
    training = _load(TRAINING_CONFIG_PATH)
    audit = validate_static_contract(
        validation_config=validation,
        training_config=training,
    )
    jobs = training_jobs(training)

    assert audit["formal_training_jobs"] == 9
    assert len(jobs) == 9
    assert {
        (job.recipe.model_id, job.seed) for job in jobs
    } == {
        (recipe.model_id, seed)
        for recipe in RECIPES
        for seed in TRAINING_SEEDS
    }
    assert all(
        "原始 H3 初始化 +" in job.recipe.display_name
        for job in jobs
    )


def test_runner_score_matrix_has_exactly_ten_explicit_training_reports() -> None:
    validation = _load(VALIDATION_CONFIG_PATH)
    training = _load(TRAINING_CONFIG_PATH)
    trains = training_jobs(training)
    scores = score_jobs(
        validation_config=validation,
        training_config=training,
        train_jobs=trains,
    )
    plan = dry_run_plan(
        stage="score",
        validation_config_path=VALIDATION_CONFIG_PATH,
        validation_config=validation,
        training_config=training,
        python="python",
        device="cuda:0",
    )

    assert len(scores) == 10
    assert scores[0].model_id == ORIGINAL_MODEL_ID
    assert len({job.model_slug for job in scores}) == 10
    assert len({job.output for job in scores}) == 10
    assert len(plan["commands"]) == 10
    for record, job in zip(plan["commands"], scores):
        command = record["command"]
        assert command[command.index("--model-id") + 1] == job.model_id
        assert (
            int(command[command.index("--training-seed") + 1])
            == job.seed
        )
        assert (
            Path(command[command.index("--training-report") + 1])
            == job.training_report
        )
        assert (
            Path(command[command.index("--checkpoint") + 1])
            == job.checkpoint
        )


def test_all_dry_run_is_nine_preflights_nine_trains_ten_scores_one_aggregate() -> None:
    validation = _load(VALIDATION_CONFIG_PATH)
    training = _load(TRAINING_CONFIG_PATH)
    plan = dry_run_plan(
        stage="all",
        validation_config_path=VALIDATION_CONFIG_PATH,
        validation_config=validation,
        training_config=training,
        python="python",
        device="cuda:0",
    )

    by_stage = {}
    for record in plan["commands"]:
        by_stage[record["stage"]] = by_stage.get(record["stage"], 0) + 1
    assert by_stage == {
        "preflight": 9,
        "train": 9,
        "score": 10,
        "aggregate": 1,
    }
    aggregate = plan["commands"][-1]["command"]
    results_index = aggregate.index("--results")
    output_index = aggregate.index("--output")
    assert output_index - results_index - 1 == 10


def test_static_contract_rejects_a_missing_validation_result() -> None:
    validation = copy.deepcopy(_load(VALIDATION_CONFIG_PATH))
    training = _load(TRAINING_CONFIG_PATH)
    validation["comparison"]["required_results"][
        "H3_Passage_MixedRules"
    ] = [3072, 4096]

    with pytest.raises(ValueError, match="结果矩阵"):
        validate_static_contract(
            validation_config=validation,
            training_config=training,
        )


def test_existing_preflight_with_wrong_seed_is_not_silently_reused(
    tmp_path: Path,
) -> None:
    validation = _load(VALIDATION_CONFIG_PATH)
    training = _load(TRAINING_CONFIG_PATH)
    source = training_jobs(training)[0]
    report = tmp_path / "preflight.json"
    job = source.__class__(
        **{
            **source.__dict__,
            "preflight_report": report,
        }
    )
    report.write_text(
        json.dumps(
            {
                "passed": True,
                "run_kind": "training_data_plan_preflight",
                "model_id": job.recipe.model_id,
                "run_name": job.preflight_run_name,
                "stable_worldmodel": {
                    "commit": validation["stable_worldmodel"]["commit"]
                },
                "training_plan": {
                    "profile": "passage_pilot",
                    "training_seed": 9999,
                    "data_split_seed": 3072,
                },
                "data": {
                    "group_weights": {
                        job.recipe.training_group: 1.0
                    },
                    "groups": {job.recipe.training_group: {}},
                    "training_data_scope": {
                        "synthetic_only": True,
                        "original_samples_included": False,
                    },
                },
                "model_contract": {
                    "model_boundary_keys": ["pixels", "action"],
                    "privileged_fields_at_model_boundary": [],
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="profile/seed"):
        validate_existing_preflight(
            job,
            validation_config=validation,
            training_config=training,
        )


def test_complete_training_report_takes_priority_over_last_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validation = _load(VALIDATION_CONFIG_PATH)
    training = _load(TRAINING_CONFIG_PATH)
    jobs = []
    for source in training_jobs(training):
        run_dir = tmp_path / "runs" / source.run_name
        report = tmp_path / "reports" / f"{source.run_name}.json"
        jobs.append(
            source.__class__(
                **{
                    **source.__dict__,
                    "run_dir": run_dir,
                    "report": report,
                    "checkpoint": run_dir / source.checkpoint.name,
                    "preflight_report": (
                        tmp_path
                        / "reports"
                        / source.preflight_report.name
                    ),
                }
            )
        )

    completed = jobs[0]
    completed.run_dir.mkdir(parents=True)
    completed.report.parent.mkdir(parents=True)
    completed.report.write_text('{"passed": true}', encoding="utf-8")
    completed.checkpoint.write_bytes(b"complete-final-weights")
    (completed.run_dir / "last.ckpt").write_bytes(
        b"complete-training-state"
    )

    validation_calls: list[str] = []
    partial_resume_calls: list[str] = []
    launched: list[str] = []
    resume_flags: list[tuple[str, bool]] = []

    def validate_training(job, **_kwargs):
        validation_calls.append(job.run_name)
        return {"passed": True}

    def reject_partial_resume(job, **_kwargs):
        partial_resume_calls.append(job.run_name)
        raise AssertionError("完整报告不得进入 last.ckpt 恢复分支")

    def build_training_command(job, *, python, resume):
        assert python == "python"
        resume_flags.append((job.run_name, resume))
        return (["mock-train", job.run_name], {})

    def record_training_launch(*, label, command, environment=None):
        assert label.startswith("全新正式训练：")
        assert environment == {}
        launched.append(command[-1])

    monkeypatch.setattr(
        passage_pipeline,
        "validate_existing_training",
        validate_training,
    )
    monkeypatch.setattr(
        passage_pipeline,
        "validate_partial_training_checkpoint",
        reject_partial_resume,
    )
    monkeypatch.setattr(
        passage_pipeline,
        "training_command",
        build_training_command,
    )
    monkeypatch.setattr(
        passage_pipeline,
        "_run_command",
        record_training_launch,
    )

    result = passage_pipeline._run_training_stage(
        tuple(jobs),
        training_config_path=TRAINING_CONFIG_PATH,
        validation_config=validation,
        python="python",
    )

    expected_fresh = [job.run_name for job in jobs[1:]]
    assert result == {
        "jobs": 9,
        "reused": 1,
        "resumed": 0,
        "fresh": 8,
    }
    assert validation_calls == [job.run_name for job in jobs]
    assert launched == expected_fresh
    assert resume_flags == [
        (run_name, False) for run_name in expected_fresh
    ]
    assert partial_resume_calls == []


def test_partial_resume_requires_a_same_directory_identity_config(
    tmp_path: Path,
) -> None:
    validation = _load(VALIDATION_CONFIG_PATH)
    training = _load(TRAINING_CONFIG_PATH)
    source = training_jobs(training)[0]
    run_dir = tmp_path / source.run_name
    run_dir.mkdir()
    (run_dir / "last.ckpt").write_bytes(b"not-inspected-before-config")
    job = source.__class__(
        **{
            **source.__dict__,
            "run_dir": run_dir,
            "checkpoint": run_dir / source.checkpoint.name,
        }
    )

    with pytest.raises(FileNotFoundError, match="无法证明断点属于当前配方"):
        validate_partial_training_checkpoint(
            job,
            training_config_path=TRAINING_CONFIG_PATH,
            validation_config=validation,
        )


def test_partial_resume_rejects_last_checkpoint_from_old_lock_protocol(
    tmp_path: Path,
) -> None:
    validation = _load(VALIDATION_CONFIG_PATH)
    training = _load(TRAINING_CONFIG_PATH)
    source = training_jobs(training)[0]
    run_dir = tmp_path / source.run_name
    run_dir.mkdir()
    (run_dir / "last.ckpt").write_bytes(b"old-protocol")
    job = source.__class__(
        **{
            **source.__dict__,
            "run_dir": run_dir,
            "checkpoint": run_dir / source.checkpoint.name,
        }
    )
    checkpoint_config = {
        "output_model_name": job.run_name,
        "subdir": job.run_name,
        "seed": job.seed,
        "wm": {"history_size": 3, "num_preds": 1},
        "data": {
            "dataset": {"frameskip": 5, "num_steps": 4}
        },
        "model": {"action_encoder": {"input_dim": 10}},
        "contextworld_benchmark": {
            "model_id": job.recipe.model_id,
            "profile": "passage_formal",
            "benchmark_config": str(TRAINING_CONFIG_PATH),
            "distributed_execution_contract": {
                "rendezvous_timeout_seconds_declared": 7200,
                "rendezvous_timeout_scope": "passage_multi_gpu_only",
                "rendezvous_timeout_seconds_applied": 7200,
                "rendezvous_timeout_override_applied": True,
                "transport_configuration": (
                    "framework_defaults_with_frozen_rendezvous_timeout"
                ),
                "audit_scheduling": AUDIT_SCHEDULING_CONTRACT,
            },
        },
    }
    (run_dir / "config.json").write_text(
        json.dumps(checkpoint_config),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="旧协议 last.ckpt 禁止恢复"):
        validate_partial_training_checkpoint(
            job,
            training_config_path=TRAINING_CONFIG_PATH,
            validation_config=validation,
        )
