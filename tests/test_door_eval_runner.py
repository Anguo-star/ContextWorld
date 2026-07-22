from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import yaml

from contextworld.evaluation.icl_model import file_sha256
from scripts import run_tworoom_door_eval as runner


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT / "configs/benchmark/tworoom_door_visual_generalization_v1.yaml"
)


def _config() -> dict:
    return runner._load_config(CONFIG_PATH)


def _args(tmp_path: Path, *extra: str):
    return runner.parse_args(
        [
            "--config",
            str(CONFIG_PATH),
            "--artifact-root",
            str(tmp_path / "evaluation"),
            *extra,
        ]
    )


def _option(command: list[str], name: str) -> str:
    index = command.index(name)
    return command[index + 1]


def _valid_common(job: runner.Job, normalizer: Path) -> dict:
    return {
        "status": "passed",
        "config": {
            "path": str(CONFIG_PATH),
            "sha256": file_sha256(CONFIG_PATH),
        },
        "normalizer": {"path": str(normalizer)},
        "model": {"checkpoint": str(job.model.checkpoint)},
        "frozen_weight_audit": {"passed": True},
        "count_audit": {"passed": True},
    }


def test_default_job_matrix_is_one_plus_three_plus_three_and_full_50_by_6(
    tmp_path: Path,
) -> None:
    config = _config()
    validation = runner._jobs(
        _args(tmp_path, "--mode", "all", "--split", "validation"), config
    )
    sealed = runner._jobs(
        _args(tmp_path, "--mode", "all", "--split", "sealed_test"), config
    )

    validation_models = {job.model.slug for job in validation}
    assert validation_models == {
        "h3_origheldout_s3072",
        "h3_door_fixed49_v2_s3072",
        "h3_door_fixed49_v2_s4096",
        "h3_door_fixed49_v2_s5120",
        "h3_door_multi_v2_s3072",
        "h3_door_multi_v2_s4096",
        "h3_door_multi_v2_s5120",
    }
    # Validation has 8 door positions.  Sealed test has 14.  Every planning
    # cell is an independent 50-query evaluation for each of six seeds.
    assert len([job for job in validation if job.mode == "latent"]) == 7
    assert len([job for job in validation if job.mode == "fixed"]) == 7 * 8 * 6
    assert len([job for job in validation if job.mode == "planning"]) == 7 * 8 * 6
    assert len(validation) == 679
    assert len([job for job in sealed if job.mode == "latent"]) == 7
    assert len([job for job in sealed if job.mode == "fixed"]) == 7 * 14 * 6
    assert len([job for job in sealed if job.mode == "planning"]) == 7 * 14 * 6
    assert len(sealed) == 1183


def test_commands_forward_frozen_config_values_without_pooling_cells(
    tmp_path: Path,
) -> None:
    config = _config()
    normalizer = runner._normalizer(config)
    args = _args(
        tmp_path,
        "--mode",
        "all",
        "--split",
        "validation",
        "--models",
        "h3_origheldout_s3072",
        "--tracks",
        "validation_seen",
        "--eval-seeds",
        "42",
    )
    jobs = runner._jobs(args, config)
    by_mode = {job.mode: job for job in jobs if job.door_position in (None, 49)}

    latent = runner._command(
        by_mode["latent"], args=args, config=config, normalizer=normalizer
    )
    assert latent[1].endswith("eval_tworoom_door_true_future_latent.py")
    assert _option(latent, "--artifact-root") == str(tmp_path / "evaluation")
    assert _option(latent, "--group") == "original_reference"

    fixed = runner._command(
        by_mode["fixed"], args=args, config=config, normalizer=normalizer
    )
    frozen_fixed = config["fixed_candidate_evaluation"]
    assert fixed[1].endswith("eval_tworoom_door_fixed_candidates.py")
    assert _option(fixed, "--num-eval") == str(
        frozen_fixed["evaluations_per_door_per_seed"]
    )
    assert _option(fixed, "--candidates") == str(
        frozen_fixed["candidates_per_query"]
    )
    assert _option(fixed, "--horizon") == str(
        frozen_fixed["horizon_action_blocks"]
    )
    assert _option(fixed, "--track") == "validation_seen"
    assert _option(fixed, "--door-position") == "49"
    assert _option(fixed, "--seed") == "42"

    planning = runner._command(
        by_mode["planning"], args=args, config=config, normalizer=normalizer
    )
    frozen_planning = config["closed_loop_planning"]
    assert planning[1].endswith("eval_tworoom_door_planning.py")
    assert _option(planning, "--eval-budget") == str(
        frozen_planning["execution_budget_raw_steps"]
    )
    assert _option(planning, "--horizon") == str(
        frozen_planning["horizon_action_blocks"]
    )
    assert _option(planning, "--receding-horizon") == str(
        frozen_planning["receding_horizon_action_blocks"]
    )
    assert _option(planning, "--cem-num-samples") == str(
        frozen_planning["candidates"]
    )
    assert _option(planning, "--cem-steps") == str(
        frozen_planning["iterations"]
    )
    assert _option(planning, "--cem-topk") == str(frozen_planning["topk"])
    assert _option(planning, "--normalizer") == str(normalizer)
    assert _option(planning, "--catalog").endswith(
        "/planning/validation/catalog.json"
    )


def test_dry_run_needs_no_checkpoints_and_reports_every_command(
    tmp_path: Path,
) -> None:
    args = _args(
        tmp_path,
        "--mode",
        "all",
        "--dry-run",
        "--models",
        "h3_origheldout_s3072",
        "--tracks",
        "validation_seen",
        "--eval-seeds",
        "42",
    )
    result = runner.run(args)

    # One whole-split latent job plus four doors for each planning mode.
    assert result["status"] == "dry_run"
    assert result["jobs"] == 9
    assert result["pending"] == 9
    assert result["skipped_valid"] == 0
    assert len(result["commands"]) == 9
    # Input discovery is reported, but dry-run remains executable even before
    # the formal planning catalog has been built.
    assert result["missing_inputs"]


def test_valid_output_recognizes_latent_and_planning_protocols(
    tmp_path: Path,
) -> None:
    config = _config()
    normalizer = runner._normalizer(config)
    args = _args(
        tmp_path,
        "--mode",
        "all",
        "--models",
        "h3_origheldout_s3072",
        "--tracks",
        "validation_seen",
        "--eval-seeds",
        "42",
    )
    jobs = runner._jobs(args, config)
    latent = next(job for job in jobs if job.mode == "latent")
    planning = next(job for job in jobs if job.mode == "planning")

    latent_payload = _valid_common(latent, normalizer)
    latent_expected = config["evaluation_data"][
        "validation_counts_per_checkpoint"
    ]["scored_sequences"]
    latent_payload.update(
        {
            "evaluation_split": "validation",
            "model": {
                "slug": latent.model.slug,
                "group": latent.model.group,
                "training_seed": latent.model.training_seed,
                "checkpoint": str(latent.model.checkpoint),
            },
            "count_audit": {
                "passed": True,
                "scored_sequences": latent_expected,
            },
        }
    )
    latent.output.parent.mkdir(parents=True, exist_ok=True)
    latent.output.write_text(json.dumps(latent_payload), encoding="utf-8")
    assert runner._valid_output(
        latent,
        config=config,
        config_path=CONFIG_PATH,
        normalizer=normalizer,
    )

    frozen = config["closed_loop_planning"]
    planning_payload = _valid_common(planning, normalizer)
    planning_payload.update(
        {
            "run_kind": "confirmation",
            "track": planning.track,
            "door_position": planning.door_position,
            "eval_seed": planning.eval_seed,
            "catalog": {"path": str(planning.catalog)},
            "protocol": {
                "queries": frozen["evaluations_per_door_per_seed"],
                "agent_speed": frozen["agent_speed"],
                "eval_budget_raw_steps": frozen["execution_budget_raw_steps"],
                "horizon_action_blocks": frozen["horizon_action_blocks"],
                "receding_horizon_action_blocks": frozen[
                    "receding_horizon_action_blocks"
                ],
                "cem_samples": frozen["candidates"],
                "cem_iterations": frozen["iterations"],
                "cem_topk": frozen["topk"],
            },
            "count_audit": {
                "passed": True,
                "records": frozen["evaluations_per_door_per_seed"],
            },
        }
    )
    planning.output.parent.mkdir(parents=True, exist_ok=True)
    planning.output.write_text(json.dumps(planning_payload), encoding="utf-8")
    assert runner._valid_output(
        planning,
        config=config,
        config_path=CONFIG_PATH,
        normalizer=normalizer,
    )

    planning_payload["protocol"]["cem_topk"] += 1
    planning.output.write_text(json.dumps(planning_payload), encoding="utf-8")
    assert not runner._valid_output(
        planning,
        config=config,
        config_path=CONFIG_PATH,
        normalizer=normalizer,
    )


def test_dry_run_skips_a_complete_valid_cell(tmp_path: Path) -> None:
    config = _config()
    normalizer = runner._normalizer(config)
    args = _args(
        tmp_path,
        "--mode",
        "fixed",
        "--dry-run",
        "--models",
        "h3_origheldout_s3072",
        "--tracks",
        "validation_seen",
        "--eval-seeds",
        "42",
    )
    jobs = runner._jobs(args, config)
    assert len(jobs) == 4
    job = jobs[0]
    frozen = config["fixed_candidate_evaluation"]
    payload = _valid_common(job, normalizer)
    payload.update(
        {
            "run_kind": "confirmation",
            "track": job.track,
            "door_position": job.door_position,
            "eval_seed": job.eval_seed,
            "catalog": {"path": str(job.catalog)},
            "protocol": {
                "queries": frozen["evaluations_per_door_per_seed"],
                "agent_speed": config["closed_loop_planning"]["agent_speed"],
                "candidates_per_query": frozen["candidates_per_query"],
                "horizon_action_blocks": frozen["horizon_action_blocks"],
            },
            "count_audit": {
                "passed": True,
                "records": frozen["evaluations_per_door_per_seed"],
            },
        }
    )
    job.output.parent.mkdir(parents=True, exist_ok=True)
    job.output.write_text(json.dumps(payload), encoding="utf-8")

    result = runner.run(args)
    assert result["jobs"] == 4
    assert result["pending"] == 3
    assert result["skipped_valid"] == 1
    assert len(result["commands"]) == 3


def test_job_retries_once_and_accepts_only_a_valid_output(
    tmp_path: Path, monkeypatch
) -> None:
    config = _config()
    normalizer = runner._normalizer(config)
    args = _args(
        tmp_path,
        "--mode",
        "fixed",
        "--models",
        "h3_origheldout_s3072",
        "--tracks",
        "validation_seen",
        "--eval-seeds",
        "42",
        "--retries",
        "1",
    )
    job = runner._jobs(args, config)[0]
    frozen = config["fixed_candidate_evaluation"]
    payload = _valid_common(job, normalizer)
    payload.update(
        {
            "run_kind": "confirmation",
            "track": job.track,
            "door_position": job.door_position,
            "eval_seed": job.eval_seed,
            "catalog": {"path": str(job.catalog)},
            "protocol": {
                "queries": frozen["evaluations_per_door_per_seed"],
                "agent_speed": config["closed_loop_planning"]["agent_speed"],
                "candidates_per_query": frozen["candidates_per_query"],
                "horizon_action_blocks": frozen["horizon_action_blocks"],
            },
            "count_audit": {
                "passed": True,
                "records": frozen["evaluations_per_door_per_seed"],
            },
        }
    )
    calls = []

    def fake_run(*unused_args, **unused_kwargs):
        calls.append(1)
        if len(calls) == 1:
            return SimpleNamespace(returncode=1)
        job.output.write_text(json.dumps(payload), encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    result = runner._run_job(
        job, gpu="7", args=args, config=config, normalizer=normalizer
    )

    assert result["status"] == "passed"
    assert result["attempts"] == 2
    assert result["gpu"] == "7"
    assert len(calls) == 2


def test_normalizer_is_declared_by_door_linked_protocol() -> None:
    config = _config()
    retention_path = runner.resolve_contextworld_path(
        config["ability_retention"]["protocol"], repo_root=ROOT
    )
    retention = yaml.safe_load(retention_path.read_text(encoding="utf-8"))
    expected = runner.resolve_contextworld_path(
        retention["artifacts"]["frozen_normalizer"], repo_root=ROOT
    )
    assert runner._normalizer(config) == expected
