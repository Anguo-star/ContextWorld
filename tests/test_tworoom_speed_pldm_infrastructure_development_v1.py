from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from contextworld.benchmarks import speed_pldm_infrastructure_development as contract


ROOT = Path(__file__).resolve().parents[1]


def _module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    assert spec is not None and spec.loader is not None
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


freezer = _module(
    "speed_development_manifest_freezer",
    "freeze_tworoom_speed_pldm_development_manifest_v1.py",
)
binding_freezer = _module(
    "speed_development_binding_freezer",
    "freeze_tworoom_speed_pldm_evaluation_binding_v1.py",
)
recovery = _module(
    "speed_development_recovery",
    "recover_tworoom_speed_pldm_formal_icl_v1.py",
)
binding_preparer = _module(
    "speed_development_binding_preparer",
    "prepare_tworoom_speed_pldm_evaluation_binding_v1.py",
)
recovery_preparer = _module(
    "speed_development_recovery_preparer",
    "prepare_tworoom_speed_pldm_formal_icl_recovery_v1.py",
)
development_evaluator = _module(
    "speed_development_evaluator",
    "eval_tworoom_speed_pldm_infrastructure_development_v1.py",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identity(path: Path) -> dict:
    return {
        "path": str(path.resolve()),
        "sha256": _sha(path),
        "size_bytes": path.stat().st_size,
    }


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _readiness_receipt(*, seed: int, config: dict, manifest: dict, checkpoint: dict, state: str) -> dict:
    checks = {
        "strict_native_checkpoint_load": {"passed": True},
        "complete_heldout_manifest_coverage": {"passed": True, "samples": 384, "scenarios": 96},
        "prefix_autoregressive_geometry": {"passed": True},
        "native_future_latent_mse_finiteness": {"passed": True, "mse_value_withheld_not_a_score": True},
        "frozen_weight_audit": {"passed": True, "state_hash_before": state, "state_hash_after": state},
        "public_boundary": {
            "passed": True,
            "public_payload_accessed": False,
            "checkpoint_selection": False,
            "scoreboard_score_emitted": False,
        },
    }
    return {
        "schema_version": 1,
        "development_id": contract.DEVELOPMENT_ID,
        "completion_id": contract.COMPLETION_ID,
        "seed": seed,
        "status": "passed_infrastructure_readiness",
        "passed": True,
        "scope": contract.DEVELOPMENT_SCOPE,
        "development_config": config,
        "development_manifest": manifest,
        "checkpoint": checkpoint,
        "checkpoint_model_state_sha256": state,
        "checks": checks,
    }


def test_length_only_quartile_indices_are_unique_and_nonadaptive() -> None:
    assert contract.deterministic_clip_indices(4) == (0, 1, 2, 3)
    assert contract.deterministic_clip_indices(10) == (0, 3, 6, 9)
    assert contract.deterministic_clip_indices(101) == (0, 33, 66, 100)
    with pytest.raises(ValueError, match="too few clips"):
        contract.deterministic_clip_indices(3)


def test_raw_h3_sample_identity_detects_tampering() -> None:
    pixels = np.arange(4 * 3 * 2 * 2, dtype=np.uint8).reshape(4, 3, 2, 2)
    actions = np.arange(40, dtype=np.float32).reshape(4, 10)
    sample = {"pixels": pixels, "action": actions}
    record = contract.make_record_arrays(sample)
    history, prefix, target = contract.verify_record_arrays(sample, record)
    assert history.shape == (3, 2, 2, 3)
    assert prefix.shape == (3, 5, 2)
    assert target.shape == (2, 2, 3)
    mutated = {"pixels": pixels.copy(), "action": actions.copy()}
    mutated["pixels"][0, 0, 0, 0] ^= 1
    with pytest.raises(RuntimeError, match="identity drifted"):
        contract.verify_record_arrays(mutated, record)


def test_actual_development_yaml_is_preregistered_and_source_pinned() -> None:
    config, identities = freezer._validate_config(freezer.DEFAULT_CONFIG)
    assert config["status"] == "preregistered_during_fixed_training_before_development_manifest_or_inference"
    assert config["sampling"]["index_rule"] == contract.CLIP_INDEX_RULE
    assert set(identities["implementation"]) == {
        "manifest_freezer",
        "development_evaluator",
        "shared_contract",
        "adapter_boundary",
    }


def test_post_interruption_amendment_preserves_base_config_and_registers_active_evaluator() -> None:
    """The evaluator must not silently reinterpret the earlier preregistration."""

    config, implementation, amendment = development_evaluator._config(
        development_evaluator.DEFAULT_CONFIG
    )
    snapshot = ROOT / amendment["pre_interruption_config_snapshot"]["path"]
    base = ROOT / amendment["base_development_config"]["path"]
    assert snapshot.read_bytes() == base.read_bytes()
    assert amendment["historical_manifest_freezer"] == config["implementation"]["manifest_freezer"]
    assert amendment["historical_development_evaluator"] == config["implementation"]["development_evaluator"]
    assert implementation["manifest_freezer"] == amendment["active_manifest_freezer"]
    assert implementation["development_evaluator"] == amendment["active_development_evaluator"]
    active_path = ROOT / "scripts/eval_tworoom_speed_pldm_infrastructure_development_v1.py"
    assert implementation["development_evaluator"]["sha256"] == _sha(active_path)
    assert implementation["development_evaluator"]["size_bytes"] == active_path.stat().st_size
    json.dumps(amendment)


def test_evaluator_audits_disclosure_before_opening_development_manifest(monkeypatch) -> None:
    """Even manifest sample records remain behind the recovery-disclosure gate."""

    import scripts.freeze_tworoom_speed_pldm_execution_disclosure_v1 as disclosure_gate

    config, implementation, amendment = development_evaluator._config(
        development_evaluator.DEFAULT_CONFIG
    )
    events: list[str] = []
    monkeypatch.setattr(
        disclosure_gate,
        "audit_disclosure",
        lambda **_kwargs: events.append("disclosure") or {},
    )
    monkeypatch.setattr(
        development_evaluator,
        "load_json",
        lambda _path: events.append("manifest") or {},
    )
    with pytest.raises(RuntimeError, match="manifest identity or scope"):
        development_evaluator._validate_manifest(
            config,
            development_evaluator.DEFAULT_CONFIG,
            development_evaluator.DEFAULT_MANIFEST,
            implementation,
            amendment,
        )
    assert events == ["disclosure", "manifest"]


def test_manifest_uses_recovery_receipts_and_keeps_preflight_role_narrow(
    tmp_path, monkeypatch
) -> None:
    """A preflight receipt proves readiness and is not accepted as completion evidence."""

    monkeypatch.setattr(freezer, "ROOT", tmp_path)
    entries = []
    completion_rows = []
    report_rows = []
    for seed in contract.EXPECTED_SEEDS:
        run_name = f"speed_pldm_reference_completion_v1_s{seed}"
        run_dir = tmp_path / f"seed_{seed}"
        run_dir.mkdir()
        checkpoint = run_dir / "weights_final_step_12840.pt"
        checkpoint.write_bytes(f"checkpoint-{seed}".encode())
        checkpoint_config = run_dir / "config.json"
        _write_json(checkpoint_config, {"seed": seed})
        trace = run_dir / "loss_trace.jsonl"
        trace.write_text(json.dumps({"optimizer_step": 12840}) + "\n", encoding="utf-8")
        preflight = run_dir / "preflight.json"
        _write_json(
            preflight,
            {
                "completion_id": contract.COMPLETION_ID,
                "status": "passed",
                "seed": seed,
                "training_started": False,
            },
        )
        report = run_dir / "training_report.json"
        _write_json(
            report,
            {
                "schema_version": 1,
                "passed": True,
                "run_kind": "confirmation",
                "profile": "additive",
                "model_id": "H3_Speed_PLDM_ReferenceCompletion",
                "run_name": run_name,
                "model": {
                    "training_method": "pldm",
                    "history_size": 3,
                    "action_block": 5,
                },
                "training": {
                    "training_complete": True,
                    "global_step": 12840,
                    "expected_optimizer_steps": 12840,
                },
                "artifacts": {
                    "pretrained": str(checkpoint),
                    "pretrained_sha256": _sha(checkpoint),
                    "pretrained_config": str(checkpoint_config),
                    "pretrained_config_sha256": _sha(checkpoint_config),
                    "loss_trace": {
                        "sha256": _sha(trace),
                        "last_optimizer_step": 12840,
                    },
                },
                "save_load_exact": True,
            },
        )
        completion = run_dir / "completion_receipt.json"
        _write_json(completion, {"seed": seed, "passed": True})
        entries.append(
            {
                "seed": seed,
                "run_name": run_name,
                "checkpoint": str(checkpoint),
                "checkpoint_config": str(checkpoint_config),
                "training_report": str(report),
                "loss_trace": str(trace),
                "preflight": str(preflight),
            }
        )
        completion_rows.append(
            {"seed": seed, "receipt": freezer.identity(completion, repo_root=tmp_path)}
        )
        report_rows.append(
            {"seed": seed, "report": freezer.identity(report, repo_root=tmp_path)}
        )

    config = {
        "training_artifacts": {
            "fixed_optimizer_steps": 12840,
            "model_id": "H3_Speed_PLDM_ReferenceCompletion",
            "entries": entries,
        }
    }
    disclosure = {
        "completion_receipts": completion_rows,
        "training_reports": report_rows,
    }
    rows = freezer._training_entries(config, disclosure)
    assert all(
        row["fixed_training_contract"]["completion_evidence"]
        == "recovery_completion_receipt_and_final_training_report"
        for row in rows
    )
    assert all(
        row["fixed_training_contract"]["preflight_receipt_role"]
        == "preflight_only_not_training_completion_evidence"
        and row["fixed_training_contract"]["historical_preflight_immutability_claimed"]
        is False
        for row in rows
    )

    rewritten = tmp_path / "seed_3072/preflight.json"
    payload = json.loads(rewritten.read_text(encoding="utf-8"))
    payload["training_completed"] = True
    _write_json(rewritten, payload)
    with pytest.raises(RuntimeError, match="preflight=False"):
        freezer._training_entries(config, disclosure)


def test_binding_rejects_development_receipt_scope_tampering(tmp_path) -> None:
    config_path = tmp_path / "development.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "development_id": contract.DEVELOPMENT_ID,
                "completion_id": contract.COMPLETION_ID,
                "scope": contract.DEVELOPMENT_SCOPE,
            }
        ),
        encoding="utf-8",
    )
    config_identity = _identity(config_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = {
        "schema_version": 1,
        "development_id": contract.DEVELOPMENT_ID,
        "completion_id": contract.COMPLETION_ID,
        "status": "frozen_prepublic_development_manifest",
        "passed": True,
        "scope": contract.DEVELOPMENT_SCOPE,
        "development_config": config_identity,
        "public_payload_accessed": False,
        "formal_public_or_cem_artifacts_present": False,
        "coverage": {
            "validation_scenarios": 96,
            "total_samples": 384,
            "all_actual_indices_unique_per_scenario": True,
            "all_source_spans_continuous": True,
        },
    }
    _write_json(manifest_path, manifest)
    manifest_identity = _identity(manifest_path)
    state = "a" * 64
    checkpoint_entries = []
    declared_receipts = []
    for seed in contract.EXPECTED_SEEDS:
        checkpoint_path = tmp_path / f"checkpoint_{seed}.pt"
        checkpoint_path.write_bytes(f"checkpoint-{seed}".encode())
        checkpoint_identity = _identity(checkpoint_path)
        receipt_path = tmp_path / f"receipt_{seed}.json"
        _write_json(
            receipt_path,
            _readiness_receipt(
                seed=seed,
                config=config_identity,
                manifest=manifest_identity,
                checkpoint=checkpoint_identity,
                state=state,
            ),
        )
        checkpoint_entries.append(
            {
                "seed": seed,
                "checkpoint": {
                    "path": checkpoint_identity["path"],
                    "sha256": checkpoint_identity["sha256"],
                    "model_state_sha256": state,
                },
            }
        )
        declared_receipts.append({"seed": seed, "receipt": _identity(receipt_path)})
    binding = {
        "development": {
            "config": config_identity,
            "manifest": manifest_identity,
            "receipts": declared_receipts,
        },
        "checkpoints": checkpoint_entries,
    }
    checks: dict = {}
    evidence = binding_freezer._validate_development_evidence(binding=binding, checks=checks)
    assert [row["seed"] for row in evidence["receipts"]] == list(contract.EXPECTED_SEEDS)
    assert checks["development_receipt_contract_3072"]["passed"] is True

    first_path = tmp_path / "receipt_3072.json"
    tampered = json.loads(first_path.read_text(encoding="utf-8"))
    tampered["scope"] = {**contract.DEVELOPMENT_SCOPE, "icl_claim": True}
    _write_json(first_path, tampered)
    binding["development"]["receipts"][0]["receipt"] = _identity(first_path)
    changed_checks: dict = {}
    binding_freezer._validate_development_evidence(binding=binding, checks=changed_checks)
    assert changed_checks["development_receipt_contract_3072"]["passed"] is False


def test_recovery_rejects_raw_without_the_bound_development_identity(monkeypatch) -> None:
    boundary_path = ROOT / "configs/benchmark/tworoom_speed_pldm_behavioral_claim_boundary_v1.yaml"
    boundary = yaml.safe_load(boundary_path.read_text(encoding="utf-8"))
    expected_development = {"config": {"path": "x", "sha256": "a"}, "manifest": {"path": "y", "sha256": "b"}, "receipts": []}
    monkeypatch.setattr(recovery, "_development_identities", lambda _prereg: expected_development)
    entry = {
        "seed": 3072,
        "checkpoint_sha256": "c" * 64,
        "model_state_sha256": "d" * 64,
        "raw_gate_passed": True,
    }
    prereg = {
        "frozen_inputs": {
            "release_config": {"sha256": "e" * 64},
            "evaluation_binding_config": {"sha256": "f" * 64},
            "evaluation_binding_receipt": {"sha256": "0" * 64},
            "behavioral_claim_boundary": {
                "path": str(boundary_path),
                "sha256": _sha(boundary_path),
            },
        }
    }
    scope = {
        "public_icl_evaluated": True,
        "action_planning_cem_executed": False,
        "original_tworoom_retention_cem_executed": False,
        "checkpoint_selection_performed": False,
        **recovery._claim_scope(),
    }
    raw = {
        "schema_version": 1,
        "benchmark": "release",
        "submission_kind": "single_model",
        "status": "passed",
        "full_protocol": True,
        "release_config": {"sha256": "e" * 64},
        "model": {"training_seed": 3072, "training_role": "multi_speed_target", "checkpoint_sha256": "c" * 64},
        "frozen_weight_audit": {"state_hash_before": "d" * 64, "state_hash_after": "d" * 64, "passed": True},
        "completion_evaluation": {
            "completion_id": contract.COMPLETION_ID,
            "checkpoint": {"sha256": "c" * 64},
            "checkpoint_model_state_sha256": "d" * 64,
            "binding": {"sha256": "f" * 64},
            "binding_receipt": {"sha256": "0" * 64},
            "development": expected_development,
            "behavioral_claim_boundary": recovery._canonical_identity(prereg["frozen_inputs"]["behavioral_claim_boundary"], label="boundary"),
            "scope": scope,
            "primary_gate": {"id": "unseen_in_range_one_step_strict_history_accuracy", "passed": True},
        },
        "tracks": {},
    }
    recovery._validate_raw(raw, entry=entry, prereg=prereg, release={"release_id": "release", "scope": {"public_tracks": []}})
    raw["completion_evaluation"]["development"] = {"detached": True}
    with pytest.raises(RuntimeError, match="Raw formal ICL contract mismatch"):
        recovery._validate_raw(raw, entry=entry, prereg=prereg, release={"release_id": "release", "scope": {"public_tracks": []}})


def test_post_development_binding_preparer_freezes_all_three_readiness_receipts(
    tmp_path, monkeypatch
) -> None:
    """The dynamic binding YAML is impossible to prepare without 3/3 no-score gates."""

    repo = tmp_path / "ContextWorld"
    repo.mkdir()

    def write_json(logical: str, value: dict) -> Path:
        path = repo / logical
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_json(path, value)
        return path

    def write_yaml(logical: str, value: dict) -> Path:
        path = repo / logical
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(value), encoding="utf-8")
        return path

    def local_identity(path: Path) -> dict:
        return {
            "path": path.relative_to(repo).as_posix(),
            "sha256": _sha(path),
            "size_bytes": path.stat().st_size,
        }

    runtime = repo / "runtime"
    (runtime / ".git").mkdir(parents=True)
    commit = "a" * 40
    (runtime / ".git" / "HEAD").write_text(commit, encoding="utf-8")
    pldm_config = runtime / "scripts/train/config/pldm.yaml"
    pldm_config.parent.mkdir(parents=True)
    pldm_config.write_text("model: {}\n", encoding="utf-8")

    normalizer = write_json("artifacts/normalizer.json", {"mean": 0})
    completion = write_yaml(
        "configs/completion.yaml",
        {
            "completion_id": contract.COMPLETION_ID,
            "training": {
                "seeds": list(contract.EXPECTED_SEEDS),
                "optimizer_steps": 12840,
                "model_id": "H3_Speed_PLDM_ReferenceCompletion",
            },
            "initialization": {"expected_model_state_sha256": "b" * 64},
            "evaluation": {
                "icl": {
                    "release_config": "configs/release.yaml",
                    "tracks": ["seen_for_multi", "unseen_interpolation"],
                }
            },
        },
    )
    release = write_yaml(
        "configs/release.yaml",
        {
            "release_id": "contextworld_tworoom_speed_icl_history3_v1",
            "evaluation": {"normalizer_sha256": _sha(normalizer)},
            "runtime": {"stable_worldmodel": {"expected_ref": commit}},
            "scope": {"public_tracks": ["seen_for_multi", "unseen_interpolation"]},
        },
    )
    source_spec = {"path": "contextworld/benchmarks/speed_icl_score.py", "sha256": "c" * 64}
    boundary = write_yaml(
        "configs/boundary.yaml",
        {
            "completion_id": contract.COMPLETION_ID,
            "release_id": "contextworld_tworoom_speed_icl_history3_v1",
            "frozen_inputs": {
                "completion_config": {k: v for k, v in local_identity(completion).items() if k != "size_bytes"},
                "speed_release": {k: v for k, v in local_identity(release).items() if k != "size_bytes"},
                "public_scorer": source_spec,
            },
        },
    )
    dev_config_path = repo / "configs/development.yaml"
    manifest_path = repo / "artifacts/development/manifest.json"
    receipt_paths = {
        seed: repo / f"artifacts/development/seed_{seed}.json"
        for seed in contract.EXPECTED_SEEDS
    }
    training_entries = []
    frozen_training_entries = []
    state = "d" * 64
    for seed in contract.EXPECTED_SEEDS:
        checkpoint = repo / f"artifacts/training/checkpoint_{seed}.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(f"checkpoint-{seed}".encode())
        checkpoint_config = write_json(f"artifacts/training/config_{seed}.json", {"seed": seed})
        trace = repo / f"artifacts/training/trace_{seed}.jsonl"
        trace.write_text(json.dumps({"optimizer_step": 12840}) + "\n", encoding="utf-8")
        report = write_json(
            f"artifacts/training/report_{seed}.json",
            {
                "schema_version": 1,
                "passed": True,
                "run_name": f"speed_pldm_reference_completion_v1_s{seed}",
                "training": {
                    "training_complete": True,
                    "global_step": 12840,
                    "terminal_report_recovery_optimizer_steps": 0,
                },
                "artifacts": {
                    "pretrained_sha256": _sha(checkpoint),
                    "pretrained_config_sha256": _sha(checkpoint_config),
                    "loss_trace": {"sha256": _sha(trace)},
                },
                "save_load_exact": True,
                "terminal_report_recovery": {
                    "training_or_optimizer_execution": False,
                },
            },
        )
        preflight = write_json(
            f"artifacts/training/preflight_{seed}.json",
            {
                "completion_id": contract.COMPLETION_ID,
                "status": "passed",
                "seed": seed,
                "training_started": False,
            },
        )
        checkpoint_identity = local_identity(checkpoint)
        checkpoint_config_identity = local_identity(checkpoint_config)
        report_identity = local_identity(report)
        trace_identity = local_identity(trace)
        preflight_identity = local_identity(preflight)
        completion_receipt = write_json(
            f"artifacts/training/completion_{seed}.json",
            {
                "completion_id": contract.COMPLETION_ID,
                "seed": seed,
                "status": "completed_fixed_budget_required_resume",
                "passed": True,
                "training_report": report_identity,
                "final_checkpoint": checkpoint_identity,
                "resume_proof": {
                    "initial_global_step": 10272,
                    "final_global_step": 12840,
                },
                "evaluation_executed": False,
                "public_test_accessed": False,
            },
        )
        frozen_training_entries.append(
            {
                "seed": seed,
                "run_name": f"speed_pldm_reference_completion_v1_s{seed}",
                "checkpoint": checkpoint_identity,
                "checkpoint_config": checkpoint_config_identity,
                "training_report": report_identity,
                "loss_trace": trace_identity,
                "preflight": preflight_identity,
                "recovery_completion_receipt": local_identity(completion_receipt),
                "fixed_training_contract": {
                    "passed": True,
                    "checkpoint_selection": "final_fixed_step",
                    "early_stopping": False,
                    "optimizer_steps": 12840,
                    "completion_evidence": (
                        "recovery_completion_receipt_and_final_training_report"
                    ),
                    "preflight_receipt_role": (
                        "preflight_only_not_training_completion_evidence"
                    ),
                    "historical_preflight_immutability_claimed": False,
                },
            }
        )
        training_entries.append(
            {
                "seed": seed,
                "checkpoint": f"artifacts/training/checkpoint_{seed}.pt",
                "checkpoint_config": f"artifacts/training/config_{seed}.json",
                "training_report": f"artifacts/training/report_{seed}.json",
                "loss_trace": f"artifacts/training/trace_{seed}.jsonl",
                "preflight": f"artifacts/training/preflight_{seed}.json",
            }
        )
    dev_config = {
        "development_id": contract.DEVELOPMENT_ID,
        "completion_id": contract.COMPLETION_ID,
        "status": "preregistered_during_fixed_training_before_development_manifest_or_inference",
        "scope": contract.DEVELOPMENT_SCOPE,
        "stable_worldmodel": {
            "worktree": str(runtime),
            "expected_ref": commit,
            "pldm_config": "scripts/train/config/pldm.yaml",
            "pldm_config_sha256": _sha(pldm_config),
        },
        "frozen_inputs": {
            "completion_config": {k: v for k, v in local_identity(completion).items() if k != "size_bytes"},
            "normalizer": {k: v for k, v in local_identity(normalizer).items() if k != "size_bytes"},
            "behavioral_claim_boundary": {k: v for k, v in local_identity(boundary).items() if k != "size_bytes"},
        },
        "outputs": {
            "manifest": "artifacts/development/manifest.json",
            "receipts": {str(seed): f"artifacts/development/seed_{seed}.json" for seed in contract.EXPECTED_SEEDS},
        },
        "training_artifacts": {"entries": training_entries},
    }
    write_yaml("configs/development.yaml", dev_config)
    config_identity = local_identity(dev_config_path)
    manifest = {
        "schema_version": 1,
        "development_id": contract.DEVELOPMENT_ID,
        "completion_id": contract.COMPLETION_ID,
        "status": "frozen_prepublic_development_manifest",
        "passed": True,
        "scope": contract.DEVELOPMENT_SCOPE,
        "development_config": config_identity,
        "public_payload_accessed": False,
        "formal_public_or_cem_artifacts_present": False,
        "training_checkpoints": frozen_training_entries,
        "coverage": {
            "validation_scenarios": 96,
            "total_samples": 384,
            "all_actual_indices_unique_per_scenario": True,
            "all_source_spans_continuous": True,
        },
    }
    write_json(manifest_path.relative_to(repo).as_posix(), manifest)
    manifest_identity = local_identity(manifest_path)
    for entry in training_entries:
        seed = entry["seed"]
        checkpoint_identity = local_identity(repo / entry["checkpoint"])
        write_json(
            receipt_paths[seed].relative_to(repo).as_posix(),
            _readiness_receipt(
                seed=seed,
                config=config_identity,
                manifest=manifest_identity,
                checkpoint=checkpoint_identity,
                state=state,
            ),
        )

    monkeypatch.setattr(binding_preparer, "ROOT", repo)
    monkeypatch.setattr(binding_preparer, "DEVELOPMENT_CONFIG", dev_config_path)
    monkeypatch.setattr(
        binding_preparer,
        "DEFAULT_OUTPUT",
        repo / "configs/benchmark/tworoom_speed_pldm_evaluation_binding_v1.yaml",
    )
    monkeypatch.setattr(
        binding_preparer,
        "DEFAULT_BINDING_RECEIPT",
        repo / "artifacts/evaluation/binding_receipt.json",
    )
    monkeypatch.setattr(binding_preparer, "FORMAL_ROOT", repo / "artifacts/formal")
    monkeypatch.setattr(binding_preparer, "PLANNED_CEM_ROOTS", (repo / "artifacts/planning", repo / "artifacts/retention"))
    monkeypatch.setattr(binding_preparer, "_source_spec", lambda path: source_spec)
    # This isolated test covers the post-Development artifact join, not the
    # separately tested production CEM authority closure.
    monkeypatch.setattr(
        binding_preparer,
        "_cem_prepublic_protocol",
        lambda **_kwargs: {"source_identities": {}},
    )

    binding = binding_preparer.build_binding(dev_config_path)
    assert binding["development"]["manifest"] == manifest_identity
    assert [row["seed"] for row in binding["development"]["receipts"]] == list(contract.EXPECTED_SEEDS)
    assert [row["checkpoint"]["model_state_sha256"] for row in binding["checkpoints"]] == [state] * 3

    receipt_paths[4096].unlink()
    with pytest.raises(FileNotFoundError):
        binding_preparer.build_binding(dev_config_path)


def test_post_raw_recovery_preparer_requires_raw_receipts_to_rebind_development(
    tmp_path, monkeypatch
) -> None:
    repo = tmp_path / "ContextWorld"
    repo.mkdir()

    def write_json(logical: str, value: dict) -> Path:
        path = repo / logical
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_json(path, value)
        return path

    def write_yaml(logical: str, value: dict) -> Path:
        path = repo / logical
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(value), encoding="utf-8")
        return path

    def local_identity(path: Path) -> dict:
        return {
            "path": path.relative_to(repo).as_posix(),
            "sha256": _sha(path),
            "size_bytes": path.stat().st_size,
        }

    completion = write_yaml(
        "configs/completion.yaml",
        {
            "completion_id": contract.COMPLETION_ID,
            "training": {"seeds": list(contract.EXPECTED_SEEDS)},
        },
    )
    release = write_yaml(
        "configs/release.yaml",
        {
            "release_id": "contextworld_tworoom_speed_icl_history3_v1",
            "scope": {"public_tracks": ["seen_for_multi", "unseen_interpolation"]},
        },
    )
    boundary = write_yaml("configs/boundary.yaml", {"boundary": True})
    checkpoints = []
    state = "d" * 64
    for seed in contract.EXPECTED_SEEDS:
        path = repo / f"artifacts/checkpoints/seed_{seed}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(str(seed).encode())
        checkpoints.append(
            {"seed": seed, "checkpoint": {**local_identity(path), "model_state_sha256": state}}
        )
    binding_path = repo / "configs/binding.yaml"
    binding = {
        "schema_version": 1,
        "binding_id": "tworoom_speed_pldm_evaluation_binding_v1",
        "status": "preregistered_after_training_before_formal_public_evaluation",
        "completion": {**local_identity(completion), "completion_id": contract.COMPLETION_ID},
        "release": {**local_identity(release), "release_id": "contextworld_tworoom_speed_icl_history3_v1"},
        "behavioral_claim_boundary": local_identity(boundary),
        "development": {"placeholder": True},
        "evaluator_sources": {
            "speed_icl_score": {
                "path": "contextworld/benchmarks/speed_icl_score.py",
                "sha256": "e" * 64,
            }
        },
        "checkpoints": checkpoints,
    }
    binding_path.parent.mkdir(parents=True, exist_ok=True)
    binding_path.write_text(yaml.safe_dump(binding), encoding="utf-8")
    binding_identity = local_identity(binding_path)
    development = {
        "config": {"path": "configs/development.yaml", "sha256": "f" * 64, "size_bytes": 1},
        "manifest": {"path": "artifacts/development/manifest.json", "sha256": "0" * 64, "size_bytes": 1},
        "receipts": [
            {
                "seed": seed,
                "receipt": {
                    "path": f"artifacts/development/seed_{seed}.json",
                    "sha256": "1" * 64,
                    "size_bytes": 1,
                },
            }
            for seed in contract.EXPECTED_SEEDS
        ],
    }
    receipt_path = write_json(
        "artifacts/binding_receipt.json",
        {
            "status": "passed_evaluation_binding_freeze",
            "passed": True,
            "binding": binding_identity,
            "development": development,
        },
    )
    scope = {
        "public_icl_evaluated": True,
        "action_planning_cem_executed": False,
        "original_tworoom_retention_cem_executed": False,
        "checkpoint_selection_performed": False,
        "paired_single_speed_control_available": False,
        "training_attribution_claim": False,
        "public_test_reopened": False,
        "claim_level": "behavioral_trained_reference_only",
    }
    formal_root = repo / "artifacts/formal"
    for item in checkpoints:
        seed = item["seed"]
        raw = {
            "schema_version": 1,
            "benchmark": "contextworld_tworoom_speed_icl_history3_v1",
            "submission_kind": "single_model",
            "status": "passed",
            "full_protocol": True,
            "release_config": local_identity(release),
            "model": {
                "training_seed": seed,
                "training_role": "multi_speed_target",
                "checkpoint_sha256": item["checkpoint"]["sha256"],
            },
            "frozen_weight_audit": {
                "state_hash_before": state,
                "state_hash_after": state,
                "passed": True,
            },
            "completion_evaluation": {
                "completion_id": contract.COMPLETION_ID,
                "binding": binding_identity,
                "binding_receipt": local_identity(receipt_path),
                "checkpoint": item["checkpoint"],
                "checkpoint_model_state_sha256": state,
                "development": development,
                "scope": scope,
                "primary_gate": {
                    "id": "unseen_in_range_one_step_strict_history_accuracy",
                    "passed": seed != 4096,
                },
            },
            "tracks": {"seen_for_multi": {}, "unseen_interpolation": {}},
        }
        write_json(f"artifacts/formal/seed_{seed}.json", raw)

    monkeypatch.setattr(recovery_preparer, "ROOT", repo)
    monkeypatch.setattr(recovery_preparer, "BINDING_CONFIG", binding_path)
    monkeypatch.setattr(recovery_preparer, "BINDING_RECEIPT", receipt_path)
    monkeypatch.setattr(recovery_preparer, "FORMAL_ROOT", formal_root)
    monkeypatch.setattr(recovery_preparer, "RECOVERY_ROOT", formal_root / "recovery_v1")
    monkeypatch.setattr(recovery_preparer, "AGGREGATE_OUTPUT", formal_root / "three_seed_aggregate.json")
    monkeypatch.setattr(recovery_preparer, "PLANNED_CEM_ROOTS", (repo / "artifacts/planning", repo / "artifacts/retention"))
    monkeypatch.setattr(recovery_preparer, "_development_chain", lambda *_: development)
    monkeypatch.setattr(recovery_preparer, "_source", lambda path: {"path": path, "sha256": "e" * 64, "size_bytes": 1})

    prereg = recovery_preparer.build_preregistration(
        binding_config_path=binding_path, binding_receipt_path=receipt_path
    )
    assert [row["raw_gate_passed"] for row in prereg["raw_public_icl"]["checkpoints"]] == [True, False, True]
    assert prereg["frozen_inputs"]["development_manifest"] == development["manifest"]

    bad_path = formal_root / "seed_4096.json"
    bad = json.loads(bad_path.read_text(encoding="utf-8"))
    bad["completion_evaluation"]["development"] = {"detached": True}
    _write_json(bad_path, bad)
    with pytest.raises(RuntimeError, match="Raw Public ICL contract"):
        recovery_preparer.build_preregistration(
            binding_config_path=binding_path, binding_receipt_path=receipt_path
        )
