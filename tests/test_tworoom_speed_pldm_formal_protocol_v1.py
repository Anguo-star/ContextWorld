from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]


def _module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    assert spec is not None and spec.loader is not None
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


evaluator = _module("speed_formal_eval", "eval_tworoom_speed_pldm_formal_icl_v1.py")
recovery = _module("speed_formal_recovery", "recover_tworoom_speed_pldm_formal_icl_v1.py")
freezer = _module("speed_binding_freezer", "freeze_tworoom_speed_pldm_evaluation_binding_v1.py")
stop_freezer = _module("speed_cem_stop_freezer", "freeze_tworoom_speed_pldm_cem_stop_v1.py")
recovery_preparer = _module(
    "speed_formal_recovery_preparer",
    "prepare_tworoom_speed_pldm_formal_icl_recovery_v1.py",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _spec(path: Path) -> dict:
    return {"path": str(path), "sha256": _sha(path)}


def _identity(path: Path) -> dict:
    return {
        "path": str(path.resolve()),
        "sha256": _sha(path),
        "size_bytes": path.stat().st_size,
    }


def test_recovery_preparer_accepts_equivalent_absolute_and_repository_paths() -> None:
    release = ROOT / "configs/benchmark/tworoom_speed_icl_release_v1.yaml"
    digest = _sha(release)

    assert recovery_preparer._same_path_sha(
        {"path": str(release.resolve()), "sha256": digest},
        {
            "path": "configs/benchmark/tworoom_speed_icl_release_v1.yaml",
            "sha256": digest,
        },
    )
    assert not recovery_preparer._same_path_sha(
        {"path": str(release.resolve()), "sha256": digest},
        {
            "path": "configs/benchmark/tworoom_speed_icl_release_v1.yaml",
            "sha256": "0" * 64,
        },
    )


def _development_fixture(tmp_path: Path, checkpoints: list[dict], state: str) -> dict:
    """Make a complete no-score Development chain for binding-freezer tests."""

    scope = {
        "development_scope": "infrastructure_readiness",
        "icl_claim": False,
        "checkpoint_selection": False,
        "public_payload_accessed": False,
        "scoreboard_score_emitted": False,
        "training_or_recipe_mutation": False,
    }
    config_path = tmp_path / "development.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "development_id": freezer.DEVELOPMENT_ID,
                "completion_id": freezer.COMPLETION_ID,
                "scope": scope,
            }
        ),
        encoding="utf-8",
    )
    config = _identity(config_path)
    manifest_path = tmp_path / "development_manifest.json"
    _write_json(
        manifest_path,
        {
            "schema_version": 1,
            "development_id": freezer.DEVELOPMENT_ID,
            "completion_id": freezer.COMPLETION_ID,
            "status": "frozen_prepublic_development_manifest",
            "passed": True,
            "scope": scope,
            "development_config": config,
            "public_payload_accessed": False,
            "formal_public_or_cem_artifacts_present": False,
            "coverage": {
                "validation_scenarios": 96,
                "total_samples": 384,
                "all_actual_indices_unique_per_scenario": True,
                "all_source_spans_continuous": True,
            },
        },
    )
    manifest = _identity(manifest_path)
    receipts = []
    for entry in checkpoints:
        seed = entry["seed"]
        checkpoint = _identity(Path(entry["checkpoint"]["path"]))
        receipt_path = tmp_path / f"development_receipt_{seed}.json"
        _write_json(
            receipt_path,
            {
                "schema_version": 1,
                "development_id": freezer.DEVELOPMENT_ID,
                "completion_id": freezer.COMPLETION_ID,
                "seed": seed,
                "status": "passed_infrastructure_readiness",
                "passed": True,
                "scope": scope,
                "development_config": config,
                "development_manifest": manifest,
                "checkpoint": checkpoint,
                "checkpoint_model_state_sha256": state,
                "checks": {
                    "strict_native_checkpoint_load": {"passed": True},
                    "complete_heldout_manifest_coverage": {
                        "passed": True,
                        "samples": 384,
                        "scenarios": 96,
                    },
                    "prefix_autoregressive_geometry": {"passed": True},
                    "native_future_latent_mse_finiteness": {
                        "passed": True,
                        "mse_value_withheld_not_a_score": True,
                    },
                    "frozen_weight_audit": {
                        "passed": True,
                        "state_hash_before": state,
                        "state_hash_after": state,
                    },
                    "public_boundary": {
                        "passed": True,
                        "public_payload_accessed": False,
                        "checkpoint_selection": False,
                        "scoreboard_score_emitted": False,
                    },
                },
            },
        )
        receipts.append({"seed": seed, "receipt": _spec(receipt_path)})
    return {"config": _spec(config_path), "manifest": _spec(manifest_path), "receipts": receipts}


def _synthetic_records() -> list[dict]:
    rows = []
    for condition, loss in (("matching", 0.1), ("opposite", 0.2)):
        rows.append(
            {
                "reference_speed": 4.8,
                "eval_seed": 42,
                "query_id": "q0",
                "static_query_id": "s0",
                "matching_condition": "matching",
                "condition": condition,
                "latent_mse_by_horizon": {
                    str(horizon): loss for horizon in (1, 2, 3, 5)
                },
            }
        )
    return rows


def test_recovery_reconstructs_every_track_and_rejects_wrong_output() -> None:
    tracks = [
        "seen_for_multi",
        "unseen_interpolation",
        "extrapolation_low",
        "extrapolation_high",
    ]
    raw = {
        "tracks": {
            track: {"data": {"full_protocol": True}, "records": _synthetic_records()}
            for track in tracks
        }
    }
    reconstructed = recovery._reconstruct_tracks(raw, {"scope": {"public_tracks": tracks}})

    assert recovery._primary_gate(reconstructed) == {
        "id": "unseen_in_range_one_step_strict_history_accuracy",
        "track": "unseen_interpolation",
        "horizon_action_blocks": 1,
        "value": 1.0,
        "passed": True,
    }
    expected = ROOT / "artifacts/evaluation/history3/tworoom_speed_pldm_reference_completion_v1/formal_icl_v1/recovery_v1/seed_3072.json"
    wrong = ROOT / "artifacts/evaluation/history3/not_speed_recovery.json"
    with pytest.raises(ValueError, match="preregistered destination"):
        recovery._assert_output(wrong, expected, label="recovery output")


def test_formal_evaluator_rejects_noncanonical_output_before_public_read(monkeypatch) -> None:
    formal_root = ROOT / "artifacts/evaluation/history3/tworoom_speed_pldm_reference_completion_v1/formal_icl_v1"
    binding = {
        "artifacts": {
            "formal_icl_root": str(formal_root),
            "action_planning_root": str(formal_root.parent / "formal_action_planning_cem_v1"),
            "retention_root": str(formal_root.parent / "formal_original_tworoom_retention_cem_v1"),
        }
    }
    monkeypatch.setattr(
        evaluator,
        "_validate_binding",
        lambda *_args, **_kwargs: (binding, {}, {"checkpoint": {}}),
    )
    with pytest.raises(ValueError, match="preregistered destination"):
        evaluator.evaluate(
            binding_path=ROOT / "configs/benchmark/tworoom_speed_pldm_evaluation_binding_v1.yaml",
            receipt_path=ROOT / "irrelevant.json",
            seed=3072,
            device="cuda:0",
            output=ROOT / "artifacts/evaluation/history3/not_speed_formal_icl.json",
        )


def test_binding_freezer_accepts_a_complete_synthetic_prepublic_contract(
    tmp_path, monkeypatch
) -> None:
    completion = tmp_path / "completion.yaml"
    completion.write_text(
        yaml.safe_dump(
            {
                "completion_id": freezer.COMPLETION_ID,
                "training": {
                    "seeds": [3072, 4096, 5120],
                    "optimizer_steps": 12840,
                    "model_id": "H3_Speed_PLDM_ReferenceCompletion",
                },
            }
        ),
        encoding="utf-8",
    )
    runtime = tmp_path / "stablewm"
    (runtime / ".git").mkdir(parents=True)
    commit = "a" * 40
    (runtime / ".git" / "HEAD").write_text(commit, encoding="utf-8")
    (runtime / "scripts/train/config").mkdir(parents=True)
    pldm_config = runtime / "scripts/train/config/pldm.yaml"
    pldm_config.write_text("model: {}\n", encoding="utf-8")
    normalizer = tmp_path / "normalizer.json"
    _write_json(normalizer, {"normalizer": "synthetic"})
    source = tmp_path / "source.py"
    source.write_text("# synthetic\n", encoding="utf-8")
    release = tmp_path / "release.yaml"
    release.write_text(
        yaml.safe_dump(
            {
                "release_id": "contextworld_tworoom_speed_icl_history3_v1",
                "scope": {
                    "public_test_included": True,
                    "sealed_test_included": False,
                    "public_tracks": ["seen_for_multi", "unseen_interpolation"],
                },
                "evaluation": {"normalizer_sha256": _sha(normalizer)},
                "runtime": {"stable_worldmodel": {"expected_ref": commit}},
            }
        ),
        encoding="utf-8",
    )
    boundary = tmp_path / "behavioral_claim_boundary.yaml"
    boundary.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "amendment_id": "tworoom_speed_pldm_behavioral_claim_boundary_v1",
                "completion_id": freezer.COMPLETION_ID,
                "release_id": "contextworld_tworoom_speed_icl_history3_v1",
                "status": "preregistered_during_fixed_training_before_development_or_public_evaluation",
                "chronology": {
                    "fixed_training_already_running": True,
                    "development_evaluation_started": False,
                    "public_test_opened": False,
                    "checkpoint_selection_changed": False,
                    "training_budget_changed": False,
                },
                "frozen_inputs": {
                    "completion_config": _spec(completion),
                    "speed_release": _spec(release),
                    "public_scorer": _spec(source),
                },
                "claim_boundary": {
                    "paired_single_speed_pldm_controls_trained": False,
                    "training_attribution_claim_authorized": False,
                    "training_attributed_speed_icl_claim_authorized": False,
                    "three_seed_behavioral_reference_authorized": True,
                    "behavioral_reference_requirement": "all three synthetic seeds pass",
                    "comparison_with_lewm_single_speed_controls_authorized": False,
                    "cross_architecture_raw_latent_loss_comparison_authorized": False,
                    "method_name_must_identify_pldm": True,
                    "scoreboard_evidence_scope_if_reported": "behavioral",
                },
                "conditional_evaluation": {
                    "development_must_precede_public": True,
                    "public_test_authorized_by_this_record": False,
                    "public_test_requires_separate_passed_binding": True,
                    "if_three_seed_public_behavioral_gate_passes": {
                        "action_planning_cem_may_be_separately_authorized": True,
                        "original_tworoom_retention_cem_may_be_separately_authorized": True,
                    },
                    "if_any_public_behavioral_gate_fails": {
                        "cem_authorized": False,
                        "cem_executed": False,
                        "terminal_stop_receipt_required": True,
                    },
                },
                "mutation_boundary": {
                    "training_or_checkpoint_selection_authorized": False,
                    "optimizer_step_or_batch_change_authorized": False,
                    "raw_data_mutation_authorized": False,
                    "score_or_threshold_change_authorized": False,
                    "public_test_access_authorized": False,
                },
            }
        ),
        encoding="utf-8",
    )
    state = "b" * 64
    checkpoints = []
    for seed in (3072, 4096, 5120):
        checkpoint = tmp_path / f"checkpoint_{seed}.pt"
        checkpoint.write_bytes(f"checkpoint-{seed}".encode())
        checkpoint_config = tmp_path / f"config_{seed}.json"
        _write_json(checkpoint_config, {"config": "synthetic", "seed": seed})
        trace = tmp_path / f"loss_trace_{seed}.jsonl"
        trace.write_text(json.dumps({"optimizer_step": 12840}) + "\n", encoding="utf-8")
        report = tmp_path / f"training_report_{seed}.json"
        _write_json(
            report,
            {
                "schema_version": 1,
                "passed": True,
                "run_kind": "confirmation",
                "profile": "additive",
                "model_id": "H3_Speed_PLDM_ReferenceCompletion",
                "run_name": f"speed_pldm_reference_completion_v1_s{seed}",
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
                    "pretrained": str(checkpoint.resolve()),
                    "pretrained_sha256": _sha(checkpoint),
                    "pretrained_config": str(checkpoint_config.resolve()),
                    "pretrained_config_sha256": _sha(checkpoint_config),
                    "loss_trace": {"sha256": _sha(trace), "last_optimizer_step": 12840},
                },
                "save_load_exact": True,
            },
        )
        preflight = tmp_path / f"preflight_{seed}.json"
        _write_json(
            preflight,
            {
                "completion_id": freezer.COMPLETION_ID,
                "status": "passed",
                "seed": seed,
                "training_started": True,
                "training_completed": True,
                "strict_load": {"model_state_sha256": "c" * 64},
            },
        )
        checkpoints.append(
            {
                "seed": seed,
                "run_name": f"speed_pldm_reference_completion_v1_s{seed}",
                "checkpoint": {**_spec(checkpoint), "model_state_sha256": state},
                "config": _spec(checkpoint_config),
                "training_report": _spec(report),
                "loss_trace": _spec(trace),
                "preflight": _spec(preflight),
            }
        )
    formal_root = tmp_path / "future_formal_icl"
    binding_path = tmp_path / "binding.yaml"
    binding = {
        "schema_version": 1,
        "binding_id": "tworoom_speed_pldm_evaluation_binding_v1",
        "status": "preregistered_after_training_before_formal_public_evaluation",
        "completion": {
            **_spec(completion),
            "completion_id": freezer.COMPLETION_ID,
            "training_seeds": [3072, 4096, 5120],
            "fixed_optimizer_steps": 12840,
            "model_id": "H3_Speed_PLDM_ReferenceCompletion",
            "initial_model_state_sha256": "c" * 64,
        },
        "release": {
            **_spec(release),
            "release_id": "contextworld_tworoom_speed_icl_history3_v1",
        },
        "normalizer": _spec(normalizer),
        "stable_worldmodel": {
            "worktree": str(runtime),
            "expected_ref": commit,
            "pldm_config": "scripts/train/config/pldm.yaml",
            "pldm_config_sha256": _sha(pldm_config),
        },
        "formal_icl": {"tracks": ["seen_for_multi", "unseen_interpolation"]},
        "behavioral_claim_boundary": _spec(boundary),
        "evaluator_sources": {"speed_icl_score": _spec(source)},
        "checkpoints": checkpoints,
        "development": _development_fixture(tmp_path, checkpoints, state),
        "artifacts": {
            "formal_icl_root": str(formal_root),
            "action_planning_root": str(tmp_path / "future_planning"),
            "retention_root": str(tmp_path / "future_retention"),
        },
    }
    binding_path.write_text(yaml.safe_dump(binding), encoding="utf-8")

    class FakeAdapter:
        def __init__(self, checkpoint_path: Path):
            self.checkpoint_path = checkpoint_path

        def frozen_state_hash(self):
            return state

        @property
        def metadata(self):
            return {
                "checkpoint_sha256": _sha(self.checkpoint_path),
                "stable_worldmodel_commit": commit,
                "adapter_id": "stable_worldmodel_pldm_v1",
                "protocol": {
                    "history_tokens": 3,
                    "action_block_raw_steps": 5,
                    "action_dim": 2,
                    "future_action_blocks": 5,
                },
            }

    class FakeAdapterClass:
        @classmethod
        def from_checkpoint(cls, checkpoint_path, *_args, **_kwargs):
            return FakeAdapter(Path(checkpoint_path))

    monkeypatch.setattr(freezer, "StableWorldModelPLDMAdapter", FakeAdapterClass)
    # This fixture predates the CEM authority closure and isolates the legacy
    # checkpoint/development binding contract.  Dedicated CEM tests exercise
    # the new closure itself below.
    monkeypatch.setattr(
        freezer,
        "_validate_prepublic_cem_protocol",
        lambda **_kwargs: {"synthetic_prepublic_cem_closure": True},
    )
    receipt = freezer.build_receipt(binding_path)
    assert receipt["passed"] is True
    assert receipt["public_test"]["accessed_by_binding"] is False
    assert [row["seed"] for row in receipt["checkpoints"]] == [3072, 4096, 5120]
    assert receipt["claim_boundary"] == {
        "paired_single_speed_control_available": False,
        "training_attribution_claim": False,
        "public_test_reopened": False,
        "claim_level": "behavioral_trained_reference_only",
    }


def test_binding_freezer_rejects_noncanonical_output_and_uses_x_exclusive(tmp_path) -> None:
    with pytest.raises(ValueError, match="Binding output must equal"):
        freezer._assert_output(tmp_path / "other.json")
    output = tmp_path / "exclusive.json"
    freezer._write_exclusive(output, {"first": True})
    with pytest.raises(FileExistsError):
        freezer._write_exclusive(output, {"second": True})


def test_recovery_rejects_wrong_seed_count_before_any_input_read() -> None:
    prereg = {
        "raw_public_icl": {
            "checkpoints": [
                {"seed": 3072},
                {"seed": 4096},
                {"seed": 4096},
            ]
        }
    }
    with pytest.raises(ValueError, match="exactly seeds 3072/4096/5120"):
        recovery._entries(prereg)


def test_recovery_claim_boundary_is_fail_closed(monkeypatch) -> None:
    boundary_path = ROOT / "configs/benchmark/tworoom_speed_pldm_behavioral_claim_boundary_v1.yaml"
    boundary = yaml.safe_load(boundary_path.read_text(encoding="utf-8"))
    malformed = json.loads(json.dumps(boundary))
    malformed["claim_boundary"]["training_attribution_claim_authorized"] = True
    monkeypatch.setattr(recovery, "_load_yaml", lambda _path: malformed)

    with pytest.raises(RuntimeError, match="claim-boundary contract"):
        recovery._validate_behavioral_claim_boundary(
            _spec(boundary_path),
            release_id=boundary["release_id"],
            completion_specification=boundary["frozen_inputs"]["completion_config"],
            release_specification=boundary["frozen_inputs"]["speed_release"],
            scorer_specification=boundary["frozen_inputs"]["public_scorer"],
        )


def test_cem_stop_fixture_requires_exact_failed_three_seed_boundary(tmp_path, monkeypatch) -> None:
    boundary = {
        "path": "configs/benchmark/tworoom_speed_pldm_behavioral_claim_boundary_v1.yaml",
        "sha256": "a" * 64,
        "size_bytes": 1,
    }
    payload = {
        "schema_version": 1,
        "recovery_id": stop_freezer.RECOVERY_ID,
        "completion_id": stop_freezer.COMPLETION_ID,
        "status": "completed",
        "evaluation_kind": "public_icl_recovery_aggregate",
        "submission_kind": "three_seed_method_recovery",
        "checkpoints": [
            {"training_seed": 3072, "passed": True},
            {"training_seed": 4096, "passed": False},
            {"training_seed": 5120, "passed": False},
        ],
        "decision": {
            "formal_evaluation_completed": True,
            "formal_method_claim": False,
            "passed": False,
            "reason": "one_or_more_training_seeds_failed_behavioral_gate",
        },
        "cem": {
            "authorized": False,
            "executed": False,
            "reason": "not_authorized_because_three_seed_icl_gate_failed",
        },
        "behavioral_claim_boundary": boundary,
        "claim_boundary": stop_freezer.CLAIM_BOUNDARY,
    }
    with pytest.raises(ValueError, match="Development chain"):
        stop_freezer._validate_aggregate_payload(payload)

    development = {
        "config": {"path": "development.yaml", "sha256": "b" * 64, "size_bytes": 1},
        "manifest": {"path": "development_manifest.json", "sha256": "c" * 64, "size_bytes": 1},
        "receipts": [],
    }
    monkeypatch.setattr(stop_freezer, "_validate_development_chain", lambda _: development)
    payload["development"] = development
    assert stop_freezer._validate_aggregate_payload(payload) == (1, boundary, development)

    wrong_count = json.loads(json.dumps(payload))
    wrong_count["checkpoints"].pop()
    with pytest.raises(ValueError, match="exactly three"):
        stop_freezer._validate_aggregate_payload(wrong_count)

    attributed = json.loads(json.dumps(payload))
    attributed["claim_boundary"]["training_attribution_claim"] = True
    with pytest.raises(ValueError, match="does not authorize"):
        stop_freezer._validate_aggregate_payload(attributed)

    output = tmp_path / "exclusive_stop.json"
    stop_freezer._write_exclusive(output, {"first": True})
    with pytest.raises(FileExistsError):
        stop_freezer._write_exclusive(output, {"second": True})
