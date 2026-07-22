from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_tworoom_door_ability.py"
CONFIG = ROOT / "configs/benchmark/tworoom_door_visual_generalization_v1.yaml"
SPEC = importlib.util.spec_from_file_location("door_ability_runner", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def _context():
    config = runner._load_door_config(CONFIG)
    protocol_path, protocol = runner._retention_protocol(config)
    frozen = runner._load_frozen_inputs(config, protocol_path, protocol)
    return config, protocol, frozen


def _args(mode: str = "all", **overrides):
    values = {
        "config": CONFIG,
        "mode": mode,
        "artifact_root": None,
        "report": None,
        "models": None,
        "eval_seeds": None,
        "gpus": [str(value) for value in range(8)],
        "retries": 1,
        "python": "python",
        "stablewm_repo": "../stable-worldmodel",
        "rollout_batch_size": 16,
        "force": False,
        "dry_run": True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _option(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


def _common_payload(job, frozen, checkpoint_hash: str) -> dict:
    catalog_hash = (
        frozen.planning_catalog_sha256
        if job.mode == "planning"
        else frozen.rollout_catalog_sha256
    )
    return {
        "status": "passed",
        "checkpoint": {
            "path": str(job.model.checkpoint),
            "sha256": checkpoint_hash,
        },
        "normalizer": {
            "path": str(frozen.normalizer),
            "sha256": frozen.normalizer_sha256,
        },
        "catalog": {"path": str(job.catalog), "sha256": catalog_hash},
        "stable_worldmodel": {"commit": frozen.stablewm_ref},
        "frozen_weight_audit": {
            "passed": True,
            "state_dict_sha256_before": "weights",
            "state_dict_sha256_after": "weights",
        },
    }


def test_frozen_protocol_and_default_matrix_are_exact() -> None:
    config, protocol, frozen = _context()
    models = runner._models(config, protocol, None)
    assert len(models) == 7
    assert [row.slug for row in models] == [
        "h3_origheldout_s3072",
        "h3_door_fixed49_v2_s3072",
        "h3_door_fixed49_v2_s4096",
        "h3_door_fixed49_v2_s5120",
        "h3_door_multi_v2_s3072",
        "h3_door_multi_v2_s4096",
        "h3_door_multi_v2_s5120",
    ]
    assert frozen.eval_seeds == (42, 43, 44, 45, 46, 47)
    assert frozen.num_eval_per_seed == 50
    assert all(len(ids) == 50 for ids in frozen.planning_ids_by_seed.values())
    assert frozen.rollout_record_count == 600


def test_jobs_are_42_independent_planning_cells_plus_7_rollouts() -> None:
    config, protocol, frozen = _context()
    jobs = runner._jobs(_args(), config, protocol, frozen)
    planning = [job for job in jobs if job.mode == "planning"]
    rollout = [job for job in jobs if job.mode == "rollout"]
    assert len(planning) == 7 * 6
    assert len(rollout) == 7
    assert all(job.eval_seed is not None for job in planning)
    assert all("ability_retention" in str(job.output) for job in jobs)


def test_commands_use_only_frozen_retention_parameters_and_artifacts() -> None:
    config, protocol, frozen = _context()
    jobs = runner._jobs(_args(), config, protocol, frozen)
    planning_job = next(job for job in jobs if job.mode == "planning")
    planning = runner._command(planning_job, args=_args(), frozen=frozen)
    assert _option(planning, "--catalog") == str(frozen.planning_catalog)
    assert _option(planning, "--normalizer") == str(frozen.normalizer)
    assert _option(planning, "--eval-budget") == str(
        frozen.planning["eval_budget"]
    )
    assert _option(planning, "--horizon") == str(frozen.planning["horizon"])
    assert _option(planning, "--receding-horizon") == str(
        frozen.planning["receding_horizon"]
    )
    assert _option(planning, "--cem-samples") == str(
        frozen.planning["cem_samples"]
    )
    assert _option(planning, "--cem-steps") == str(frozen.planning["cem_steps"])
    assert _option(planning, "--cem-topk") == str(frozen.planning["cem_topk"])

    rollout_job = next(job for job in jobs if job.mode == "rollout")
    rollout = runner._command(rollout_job, args=_args(), frozen=frozen)
    assert _option(rollout, "--catalog") == str(frozen.rollout_catalog)
    assert "eval_tworoom_rollout_error.py" in rollout[1]


def test_planning_resume_requires_exact_paths_hashes_counts_and_ids(tmp_path) -> None:
    _, _, base = _context()
    checkpoint = tmp_path / "weights.pt"
    normalizer = tmp_path / "normalizer.json"
    catalog = tmp_path / "catalog.json"
    checkpoint.write_bytes(b"checkpoint")
    normalizer.write_text("{}", encoding="utf-8")
    catalog.write_text("{}", encoding="utf-8")
    seed = 42
    ids = frozenset({"q0", "q1"})
    frozen = replace(
        base,
        normalizer=normalizer,
        normalizer_sha256=runner.file_sha256(normalizer),
        planning_catalog=catalog,
        planning_catalog_sha256=runner.file_sha256(catalog),
        eval_seeds=(seed,),
        num_eval_per_seed=2,
        planning_ids_by_seed={seed: ids},
    )
    model = runner.Model("fixed", "model", 3072, checkpoint)
    output = tmp_path / "planning.json"
    job = runner.Job("planning", model, output, tmp_path / "x.log", catalog, seed)
    checkpoint_hash = runner.file_sha256(checkpoint)
    payload = _common_payload(job, frozen, checkpoint_hash)
    payload.update(
        {
            "protocol": {
                "action_block": 5,
                "history_size": 3,
                "eval_seed": seed,
                "evaluations": 2,
                **frozen.planning,
            },
            "aggregate": {"evaluations": 2},
            "raw_records": [
                {"evaluation_id": value, "eval_seed": seed}
                for value in sorted(ids)
            ],
        }
    )
    output.write_text(json.dumps(payload), encoding="utf-8")
    assert runner._valid_output(job, frozen=frozen, checkpoint_hashes={})

    payload["checkpoint"]["sha256"] = "stale"
    output.write_text(json.dumps(payload), encoding="utf-8")
    assert not runner._valid_output(job, frozen=frozen, checkpoint_hashes={})


def test_rollout_resume_requires_all_frozen_records_and_aggregates(tmp_path) -> None:
    _, _, base = _context()
    checkpoint = tmp_path / "weights.pt"
    normalizer = tmp_path / "normalizer.json"
    catalog = tmp_path / "rollout.json"
    checkpoint.write_bytes(b"checkpoint")
    normalizer.write_text("{}", encoding="utf-8")
    catalog.write_text("{}", encoding="utf-8")
    ids = frozenset({"r0", "r1"})
    frozen = replace(
        base,
        normalizer=normalizer,
        normalizer_sha256=runner.file_sha256(normalizer),
        rollout_catalog=catalog,
        rollout_catalog_sha256=runner.file_sha256(catalog),
        eval_seeds=(42,),
        rollout_horizons=(1, 2),
        rollout_ids=ids,
        rollout_counts_by_domain_seed={("original_heldout", 42): 2},
    )
    model = runner.Model("multi", "model", 3072, checkpoint)
    output = tmp_path / "result.json"
    job = runner.Job("rollout", model, output, tmp_path / "x.log", catalog)
    payload = _common_payload(job, frozen, runner.file_sha256(checkpoint))
    payload.update(
        {
            "protocol": {
                "action_block": 5,
                "history_size": 3,
                "horizons_action_blocks": [1, 2],
            },
            "aggregates": [
                {
                    "domain": "original_heldout",
                    "horizon_action_blocks": horizon,
                    "evaluations": 2,
                }
                for horizon in (1, 2)
            ],
            "raw_records": [
                {
                    "evaluation_id": value,
                    "domain": "original_heldout",
                    "eval_seed": 42,
                    "horizons": {"1": {}, "2": {}},
                }
                for value in sorted(ids)
            ],
        }
    )
    output.write_text(json.dumps(payload), encoding="utf-8")
    assert runner._valid_output(job, frozen=frozen, checkpoint_hashes={})

    payload["raw_records"].pop()
    output.write_text(json.dumps(payload), encoding="utf-8")
    assert not runner._valid_output(job, frozen=frozen, checkpoint_hashes={})


def test_retry_runs_again_until_output_is_valid(tmp_path, monkeypatch) -> None:
    config, protocol, frozen = _context()
    original = runner._models(config, protocol, None)[0]
    output = tmp_path / "result.json"
    job = runner.Job(
        "planning",
        original,
        output,
        tmp_path / "result.log",
        frozen.planning_catalog,
        frozen.eval_seeds[0],
    )
    attempts = {"count": 0}

    def fake_subprocess(*_args, **_kwargs):
        attempts["count"] += 1
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner.subprocess, "run", fake_subprocess)
    monkeypatch.setattr(
        runner,
        "_valid_output",
        lambda *_args, **_kwargs: attempts["count"] >= 2,
    )
    result = runner._run_job(
        job,
        gpu="7",
        args=_args(dry_run=False),
        frozen=frozen,
        checkpoint_hashes={},
    )
    assert result["status"] == "passed"
    assert result["attempts"] == 2


def test_cli_defaults_to_all_eight_gpu_queues() -> None:
    args = runner.parse_args(["--mode", "all", "--dry-run"])
    assert args.gpus == [str(value) for value in range(8)]
    assert args.retries == 1
