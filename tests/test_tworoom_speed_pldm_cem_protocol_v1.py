from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

import contextworld.benchmarks.pldm_reference_completion_aggregate as finalizer
from scripts import freeze_tworoom_speed_pldm_cem_aggregate_v1 as cem_aggregate
from scripts import freeze_tworoom_speed_pldm_cem_binding_v1 as cem_binding
from scripts import run_tworoom_speed_pldm_cem_v1 as cem_runner


SEEDS = (3072, 4096, 5120)


def _identity(path: str, marker: str) -> dict[str, Any]:
    return {"path": path, "sha256": marker * 64, "size_bytes": 1}


def _criteria() -> dict[str, Any]:
    return {
        "reference": "frozen_original_pldm_cem_6x50",
        "confidence_level": 0.95,
        "paired_bootstrap_seed": 3072,
        "paired_bootstrap_resamples": 10000,
        "success_rate_delta_lower_bound": -0.05,
        "final_distance_delta_upper_bound_px": 5.0,
        "require_no_solvable_room_relation_stratum_collapse": True,
        "stratum_definition": "room_relation",
        "collapse_definition": "candidate_zero_successes_where_baseline_has_at_least_one",
    }


def _action_schedule_and_records() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    schedule: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for index in range(300):
        eval_seed = 42 + index // 50
        evaluation_index = index % 50
        planned = {
            "evaluation_id": f"s{eval_seed}-e{evaluation_index:03d}",
            "eval_seed": eval_seed,
            "evaluation_index": evaluation_index,
            "repeat_index": index // 18,
            "cem_seed": 100000 + index,
            "query_id": f"query-{index % 18}",
            "template_id": f"template-{index % 18}",
            "source_scenario_id": f"scenario-{index % 18}",
            "condition": "history_mid",
        }
        schedule.append(planned)
        records.append(
            {
                **planned,
                "history_relation": "same",
                "query_speed": 5.1,
                "history_speed": 5.1,
                "success": index % 2 == 0,
                "final_distance": float(index % 11),
                "trajectory": {"raw_steps_executed": 100},
            }
        )
    return schedule, records


def _action_terminal(seed: int, records: list[dict[str, Any]]) -> dict[str, Any]:
    successes = sum(bool(row["success"]) for row in records)
    return {
        "training_seed": seed,
        "records": copy.deepcopy(records),
        "aggregate": {
            "evaluations": 300,
            "successes": successes,
            "success_rate": successes / 300,
        },
    }


def test_static_cem_preregistration_is_fully_source_pinned_and_jsonl_only() -> None:
    """This is a no-inference check of the authority sealed before Public ICL."""

    static, sources = cem_binding._validate_static_prereg(cem_binding.CEM_PREREG)

    assert static["cem_preregistration_id"] == "tworoom_speed_pldm_cem_prereg_v1"
    assert set(sources) >= {
        "preregistration",
        "source_protocol",
        "action_catalog",
        "retention_catalog",
        "implementation_formal_runner",
        "implementation_aggregate_freezer",
        "implementation_binding_freezer",
    }
    for name in (
        "source_protocol",
        "action_catalog",
        "retention_catalog",
        "implementation_formal_runner",
        "implementation_aggregate_freezer",
        "implementation_binding_freezer",
    ):
        assert set(sources[name]) == {"path", "sha256", "size_bytes"}
    assert static["outputs"]["action_planning"]["receipts"] == "seed_{training_seed}.jsonl"
    assert static["outputs"]["original_task_retention"]["receipts"] == "seed_{training_seed}.jsonl"


def test_static_identity_rejects_a_byte_size_substitution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    declared = _identity("configs/frozen.yaml", "a")
    monkeypatch.setattr(
        cem_binding,
        "_source",
        lambda _path: {**declared, "size_bytes": declared["size_bytes"] + 1},
    )
    with pytest.raises(RuntimeError, match="identity drifted"):
        cem_binding._require_static_identity(declared, label="synthetic frozen source")


def test_append_only_reservation_rejects_header_substitution_and_duplicate_events(
    tmp_path: Path,
) -> None:
    ledger = tmp_path / "seed_3072.jsonl"
    reservation = cem_runner._reservation(
        binding={},
        binding_identity=_identity("artifacts/binding.json", "a"),
        track="action_planning_cem",
        seed=3072,
        output=ledger,
        snapshot={"identities": []},
    )
    assert cem_runner._reserve_or_resume(output=ledger, expected=reservation) == [reservation]
    assert cem_runner._reserve_or_resume(output=ledger, expected=reservation) == [reservation]

    substituted = {**reservation, "training_seed": 4096}
    with pytest.raises(RuntimeError, match="exact reserved job"):
        cem_runner._reserve_or_resume(output=ledger, expected=substituted)

    event = {"record_type": "evaluation", "record": {"evaluation_id": "only-once"}}
    cem_runner._append_locked(ledger, event)
    cem_runner._append_locked(ledger, event)
    with pytest.raises(RuntimeError, match="duplicate evaluation"):
        cem_runner._completed_records(cem_runner._read_ledger(ledger))


def test_runner_rejects_a_prepublic_authority_replaced_after_evaluation_binding(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The runner must compare all three copies before any CEM construction."""

    preregistration = _identity("configs/cem_prereg.yaml", "p")
    expected_authority = {
        "preregistration": preregistration,
        "source_identities": {"formal_runner": _identity("scripts/runner.py", "r")},
    }
    replaced_authority = {
        **expected_authority,
        "source_identities": {"formal_runner": _identity("scripts/substitute.py", "s")},
    }
    binding = {
        "schema_version": 1,
        "cem_binding_id": cem_runner.CEM_BINDING_ID,
        "completion_id": cem_runner.COMPLETION_ID,
        "status": "frozen_after_passed_three_seed_public_icl_before_cem",
        "passed": True,
        "cem": {"authorized": True, "executed": False},
        "preregistration": preregistration,
        "tracks": {
            "shared": {},
            "action_planning_cem": {},
            "original_task_retention_cem": {},
        },
        "frozen_chain": {
            "evaluation_binding_config": _identity("configs/evaluation_binding.yaml", "b"),
            "evaluation_binding_receipt": _identity("artifacts/evaluation_binding.json", "e"),
            "prepublic_cem_authority": replaced_authority,
        },
        "input_integrity": {
            "all_frozen_inputs_unchanged_during_binding": True,
            "identities_before_binding": {"identities": []},
            "identities_after_binding": {"identities": []},
        },
    }
    evaluation_binding = {"cem_protocol": expected_authority}
    evaluation_receipt = {"cem_protocol": expected_authority}

    monkeypatch.setattr(cem_runner, "resolve_source", lambda path, **_kwargs: Path(path))
    monkeypatch.setattr(
        cem_runner,
        "_load_json",
        lambda path: binding if Path(path).name == "cem_binding.json" else evaluation_receipt,
    )
    monkeypatch.setattr(cem_runner, "_load_yaml", lambda _path: evaluation_binding)
    monkeypatch.setattr(cem_runner, "_source_identity", lambda value, **_kwargs: dict(value))
    monkeypatch.setattr(cem_binding, "_validate_static_prereg", lambda *_args: ({}, {}))
    monkeypatch.setattr(
        cem_binding,
        "_prepublic_cem_authority",
        lambda **_kwargs: copy.deepcopy(expected_authority),
    )

    with pytest.raises(RuntimeError, match="not rooted in the pre-Public CEM closure"):
        cem_runner._validate_binding(
            tmp_path / "cem_binding.json", track="action_planning_cem", seed=3072
        )


def test_ledger_freezer_recomputes_and_requires_the_reserved_input_snapshot(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Ledger receipt snapshots cannot be copied from a different binding state."""

    ledger = tmp_path / "ledger.jsonl"
    logical_ledger = "artifacts/test/ledger.jsonl"
    snapshot = {"identities": [{"path": "configs/frozen.yaml", "sha256": "f" * 64, "size_bytes": 1}]}
    binding_identity = _identity("artifacts/binding.json", "b")
    checkpoint = {
        **_identity("artifacts/checkpoint.pt", "c"),
        "model_state_sha256": "d" * 64,
    }
    normalizer = _identity("artifacts/normalizer.json", "e")
    track = {
        "result_semantics": "EXECUTED_VALID_DESCRIPTIVE",
        "catalog": _identity("artifacts/catalog.json", "1"),
        "protocol": {"cem_samples": 300},
    }
    binding = {
        "frozen_chain": {"checkpoints": [{"seed": 3072, "checkpoint": checkpoint}]},
        "tracks": {"shared": {"normalizer": normalizer}},
    }
    policy = {
        "exclusive_create_before_cem": True,
        "append_only_progress": True,
        "overwrite_permitted": False,
        "resume_requires_exact_binding": True,
    }
    header = {
        "record_type": "reservation",
        "schema_version": 1,
        "completion_id": cem_aggregate.COMPLETION_ID,
        "cem_binding_id": cem_aggregate.CEM_BINDING_ID,
        "binding": binding_identity,
        "track": "action_planning_cem",
        "training_seed": 3072,
        "canonical_ledger": logical_ledger,
        "reservation_policy": policy,
        "input_snapshot_before_reservation": snapshot,
    }
    terminal = {
        "schema_version": 1,
        "completion_id": cem_aggregate.COMPLETION_ID,
        "cem_binding_id": cem_aggregate.CEM_BINDING_ID,
        "status": "completed_exclusive_resumable_cem_ledger",
        "evaluation_kind": "action_planning_cem",
        "training_seed": 3072,
        "binding": binding_identity,
        "canonical_ledger": logical_ledger,
        "result_semantics": track["result_semantics"],
        "checkpoint": checkpoint,
        "normalizer": normalizer,
        "catalog": track["catalog"],
        "protocol": track["protocol"],
        "execution_policy": policy,
        "records": [],
        "frozen_weight_audit": {
            "passed": True,
            "state_dict_sha256_before": checkpoint["model_state_sha256"],
            "state_dict_sha256_after": checkpoint["model_state_sha256"],
            "bound_checkpoint_model_state_sha256": checkpoint["model_state_sha256"],
        },
        "input_integrity": {
            "all_bound_inputs_unchanged_during_cem": True,
            "identities_before_cem": snapshot,
            "identities_after_cem": snapshot,
        },
    }

    ledger_identity = _identity(logical_ledger, "9")
    monkeypatch.setattr(cem_aggregate.runner, "_track_output", lambda *_args: (ledger, tmp_path / "work"))
    monkeypatch.setattr(cem_aggregate.runner, "_bound_snapshot", lambda *_args, **_kwargs: snapshot)
    monkeypatch.setattr(cem_aggregate, "logical_path", lambda *_args, **_kwargs: logical_ledger)
    monkeypatch.setattr(cem_aggregate, "identity", lambda *_args, **_kwargs: ledger_identity)

    def write_receipt(receipt: dict[str, Any]) -> None:
        ledger.write_text(
            "\n".join(
                json.dumps(item, sort_keys=True)
                for item in (header, {"record_type": "completed_receipt", "receipt": receipt})
            )
            + "\n",
            encoding="utf-8",
        )

    write_receipt(terminal)
    observed, observed_identity = cem_aggregate._ledger_rows(
        binding=binding,
        binding_identity=binding_identity,
        track_name="action_planning_cem",
        track=track,
        seed=3072,
    )
    assert observed == terminal
    assert observed_identity == ledger_identity

    tampered = copy.deepcopy(terminal)
    tampered["input_integrity"]["identities_before_cem"] = {"identities": []}
    write_receipt(tampered)
    with pytest.raises(RuntimeError, match="current bound input snapshot"):
        cem_aggregate._ledger_rows(
            binding=binding,
            binding_identity=binding_identity,
            track_name="action_planning_cem",
            track=track,
            seed=3072,
        )


def test_paired_retention_noninferiority_uses_bootstrap_ci_and_room_relation_guard() -> None:
    criteria = _criteria()
    baseline = {
        f"episode-{index}": {
            "success": True,
            "final_distance": 10.0,
            "room_relation": "same_room" if index % 2 == 0 else "different_room",
        }
        for index in range(8)
    }
    candidate = copy.deepcopy(baseline)
    result = cem_aggregate._paired_retention_result(
        baseline=baseline, candidate=candidate, criteria=criteria
    )
    assert result["passed"] is True
    assert result["candidate_minus_reference_success_rate"]["ci_lower"] == 0.0
    assert result["candidate_minus_reference_final_distance_px"]["ci_upper"] == 0.0

    collapsed = copy.deepcopy(candidate)
    for row in collapsed.values():
        row["success"] = False
    collapsed_result = cem_aggregate._paired_retention_result(
        baseline=baseline, candidate=collapsed, criteria=criteria
    )
    assert collapsed_result["passed"] is False
    assert collapsed_result["gates"]["no_solvable_room_relation_stratum_collapse"] is False
    assert collapsed_result["collapsed_solvable_room_relation_strata"] == [
        "different_room",
        "same_room",
    ]


def test_action_aggregate_is_descriptive_and_never_has_a_model_pass_bit() -> None:
    schedule, records = _action_schedule_and_records()
    track = {
        "metric": {"id": "success_rate_by_execution_budget_100_raw_steps"},
        "schedule": schedule,
    }
    binding = {"frozen_chain": {"development": {"fixture": "no-model"}}}
    ledgers = [
        (_action_terminal(seed, records), _identity(f"artifacts/ledger-{seed}.jsonl", str(index + 1)))
        for index, seed in enumerate(SEEDS)
    ]
    payload = cem_aggregate._action_payload(
        binding=binding,
        binding_identity=_identity("artifacts/binding.json", "a"),
        track=track,
        ledgers=ledgers,
    )
    assert payload["status"] == "completed_executed_valid_descriptive"
    assert payload["result_semantics"] == "EXECUTED_VALID_DESCRIPTIVE"
    assert all("passed" not in checkpoint for checkpoint in payload["checkpoints"])
    assert payload["decision"] == {
        "execution_valid": True,
        "model_performance_gate": None,
        "retention_result": "NOT_APPLICABLE",
        "result": "EXECUTED_VALID_DESCRIPTIVE",
    }


def _formal_spec() -> dict[str, Any]:
    return {
        "completion_id": "speed-completion",
        "component_id": "speed",
        "component_name": "Speed",
        "formal_result": {
            "cem_binding": "artifacts/cem_binding.json",
            "cem_not_authorized_stop": "artifacts/cem_stop.json",
            "action_planning_aggregate": "artifacts/action.json",
            "original_task_retention_aggregate": "artifacts/retention.json",
            "method_name": "PLDM",
            "behavioral_claim_boundary": "configs/boundary.yaml",
        },
    }


def _public_icl(*, passed: bool) -> dict[str, Any]:
    gates = (True, True, True) if passed else (True, True, False)
    return {
        "metric": {"id": "metric", "label": "Metric"},
        "records": [
            {"training_seed": seed, "value": 1.0, "passed": gate}
            for seed, gate in zip(SEEDS, gates, strict=True)
        ],
        "aggregate": _identity("artifacts/public.json", "p"),
        "cem": {"authorized": passed, "executed": False},
        "ability_passed": passed,
        "development": {"fixture": "pre-public"},
    }


@pytest.mark.parametrize("passed", (True, False), ids=("three_of_three", "two_of_three"))
def test_finalizer_only_opens_cem_after_exact_three_of_three(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, passed: bool
) -> None:
    calls: list[str] = []
    public = _public_icl(passed=passed)
    monkeypatch.setattr(
        finalizer,
        "_load_completion",
        lambda *_args, **_kwargs: ({}, _identity("configs/completion.yaml", "c"), "configs/release.yaml", list(SEEDS)),
    )
    monkeypatch.setattr(finalizer, "_read_yaml", lambda *_args, **_kwargs: {"release_id": "release"})
    monkeypatch.setattr(finalizer, "_identity", lambda path, **_kwargs: _identity(str(path), "i"))
    monkeypatch.setattr(
        finalizer,
        "_speed_behavioral_claim_boundary",
        lambda **_kwargs: _identity("configs/boundary.yaml", "b"),
    )
    monkeypatch.setattr(finalizer, "_public_icl_from_recovery", lambda *_args, **_kwargs: public)

    def positive_binding(**_kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        calls.append("binding")
        return _identity("artifacts/cem_binding.json", "z"), {"fixture": "binding"}

    def aggregate_result(**kwargs: Any) -> dict[str, Any]:
        calls.append(str(kwargs["expected_kind"]))
        return {
            "metric": {"id": str(kwargs["expected_kind"]), "label": "CEM"},
            "records": [],
            "aggregate": _identity(f"artifacts/{kwargs['expected_kind']}.json", "x"),
        }

    def stop(**_kwargs: Any) -> dict[str, Any]:
        calls.append("stop")
        return _identity("artifacts/cem_stop.json", "s")

    monkeypatch.setattr(finalizer, "_speed_positive_cem_binding", positive_binding)
    monkeypatch.setattr(finalizer, "_cem_aggregate", aggregate_result)
    monkeypatch.setattr(finalizer, "_cem_not_authorized_stop", stop)

    material = finalizer._formal_completion_material("speed", _formal_spec(), repo_root=tmp_path)
    if passed:
        assert calls == ["binding", "action_planning_cem", "original_task_retention_cem"]
        assert material["cem_finalization"]["executed"] is True
    else:
        assert calls == ["stop"]
        assert material["action_planning_cem"] is None
        assert material["original_task_retention_cem"] is None
        assert material["cem_finalization"]["executed"] is False


def test_speed_finalizer_rejects_ledger_source_snapshot_and_paired_ci_tampering(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Exercise the finalizer itself, with synthetic ledgers and no model calls."""

    schedule, records = _action_schedule_and_records()
    development = {"fixture": "development"}
    binding_identity = _identity("artifacts/cem_binding.json", "b")
    action_track = {
        "metric": {"id": "action_metric"},
        "schedule": schedule,
    }
    action_ledgers = {
        seed: (_action_terminal(seed, records), _identity(f"artifacts/action-{seed}.jsonl", str(index + 1)))
        for index, seed in enumerate(SEEDS)
    }
    action_output = tmp_path / "action.json"
    action_payload = {
        "schema_version": 1,
        "completion_id": "speed-completion",
        "cem_binding_id": "tworoom_speed_pldm_cem_binding_v1",
        "binding": binding_identity,
        "status": "completed_executed_valid_descriptive",
        "evaluation_kind": "action_planning_cem",
        "result_semantics": "EXECUTED_VALID_DESCRIPTIVE",
        "metric": {"id": "action_metric", "label": "Action"},
        "checkpoints": [
            {
                "training_seed": seed,
                "value": terminal["aggregate"]["success_rate"],
                "execution_valid": True,
                "source": ledger_identity,
            }
            for seed, (terminal, ledger_identity) in action_ledgers.items()
        ],
        "decision": {
            "execution_valid": True,
            "model_performance_gate": None,
            "retention_result": "NOT_APPLICABLE",
            "result": "EXECUTED_VALID_DESCRIPTIVE",
        },
        "development": development,
        "output": {
            "path": "artifacts/action.json",
            "content_sha256_not_embedded_to_avoid_self_reference": True,
        },
        "input_integrity": {
            "all_bound_inputs_unchanged_during_aggregate_read": True,
            "identities_before_aggregate_read": {"identities": []},
            "identities_after_aggregate_read": {"identities": []},
            "ledgers": [
                {"training_seed": seed, "source": ledger_identity}
                for seed, (_terminal, ledger_identity) in action_ledgers.items()
            ],
        },
    }
    action_identity = _identity("artifacts/action.json", "a")
    identity_by_path = {
        action_identity["path"]: action_identity,
        **{
            item[1]["path"]: item[1] for item in action_ledgers.values()
        },
    }

    monkeypatch.setattr(
        finalizer,
        "_identity",
        lambda path, **_kwargs: copy.deepcopy(identity_by_path[str(path)]),
    )
    monkeypatch.setattr(finalizer, "_read_json", lambda *_args, **_kwargs: copy.deepcopy(action_payload))
    monkeypatch.setattr(finalizer, "resolve_contextworld_path", lambda *_args, **_kwargs: action_output)
    monkeypatch.setattr(cem_aggregate, "_aggregate_output", lambda _track: action_output)
    monkeypatch.setattr(
        cem_aggregate,
        "_ledger_rows",
        lambda **kwargs: action_ledgers[int(kwargs["seed"])],
    )

    action_binding = {"tracks": {"action_planning_cem": action_track}}
    valid = finalizer._speed_cem_aggregate(
        path="artifacts/action.json",
        completion_id="speed-completion",
        expected_kind="action_planning_cem",
        expected_seeds=list(SEEDS),
        expected_development=development,
        speed_cem_binding=(binding_identity, action_binding),
        repo_root=tmp_path,
    )
    assert valid["result_semantics"] == "EXECUTED_VALID_DESCRIPTIVE"

    bad_ledger = copy.deepcopy(action_payload)
    bad_identity = _identity("artifacts/not-the-reserved-ledger.jsonl", "q")
    identity_by_path[bad_identity["path"]] = bad_identity
    bad_ledger["input_integrity"]["ledgers"][0]["source"] = bad_identity
    monkeypatch.setattr(finalizer, "_read_json", lambda *_args, **_kwargs: copy.deepcopy(bad_ledger))
    with pytest.raises(ValueError, match="does not bind seed"):
        finalizer._speed_cem_aggregate(
            path="artifacts/action.json",
            completion_id="speed-completion",
            expected_kind="action_planning_cem",
            expected_seeds=list(SEEDS),
            expected_development=development,
            speed_cem_binding=(binding_identity, action_binding),
            repo_root=tmp_path,
        )

    bad_snapshot = copy.deepcopy(action_payload)
    bad_snapshot["input_integrity"]["identities_after_aggregate_read"] = {"identities": ["changed"]}
    monkeypatch.setattr(finalizer, "_read_json", lambda *_args, **_kwargs: copy.deepcopy(bad_snapshot))
    with pytest.raises(ValueError, match="intact ledger freeze"):
        finalizer._speed_cem_aggregate(
            path="artifacts/action.json",
            completion_id="speed-completion",
            expected_kind="action_planning_cem",
            expected_seeds=list(SEEDS),
            expected_development=development,
            speed_cem_binding=(binding_identity, action_binding),
            repo_root=tmp_path,
        )

    criteria = _criteria()
    baseline = {
        f"pair-{index}": {
            "success": True,
            "final_distance": 10.0,
            "room_relation": "same_room" if index % 2 == 0 else "different_room",
        }
        for index in range(8)
    }
    candidate = copy.deepcopy(baseline)
    comparison = cem_aggregate._paired_retention_result(
        baseline=baseline, candidate=candidate, criteria=criteria
    )
    retention_track = {
        "metric": {"id": "retention_metric", "paired_noninferiority": criteria},
        "paired_baseline": {"frozen": "278-of-300"},
    }
    retention_ledgers = {
        seed: (
            {
                "training_seed": seed,
                "records": [],
                "aggregate": {"success_rate": 1.0},
            },
            _identity(f"artifacts/retention-{seed}.jsonl", str(index + 4)),
        )
        for index, seed in enumerate(SEEDS)
    }
    retention_output = tmp_path / "retention.json"
    retention_identity = _identity("artifacts/retention.json", "r")
    identity_by_path[retention_identity["path"]] = retention_identity
    identity_by_path.update({item[1]["path"]: item[1] for item in retention_ledgers.values()})
    retention_payload = {
        "schema_version": 1,
        "completion_id": "speed-completion",
        "cem_binding_id": "tworoom_speed_pldm_cem_binding_v1",
        "binding": binding_identity,
        "status": "completed_paired_retention_evaluation",
        "evaluation_kind": "original_task_retention_cem",
        "result_semantics": "PAIRED_NONINFERIORITY_RETENTION",
        "metric": {"id": "retention_metric", "label": "Retention"},
        "checkpoints": [
            {
                "training_seed": seed,
                "value": 1.0,
                "passed": True,
                "source": ledger_identity,
                "paired_noninferiority": comparison,
            }
            for seed, (_terminal, ledger_identity) in retention_ledgers.items()
        ],
        "decision": {
            "all_training_seeds_passed": True,
            "passed": True,
            "result": "PASS",
            "criterion": "all_three_fixed_checkpoints_must_pass_paired_noninferiority",
        },
        "paired_baseline": retention_track["paired_baseline"],
        "development": development,
        "output": {
            "path": "artifacts/retention.json",
            "content_sha256_not_embedded_to_avoid_self_reference": True,
        },
        "input_integrity": {
            "all_bound_inputs_unchanged_during_aggregate_read": True,
            "identities_before_aggregate_read": {"identities": []},
            "identities_after_aggregate_read": {"identities": []},
            "ledgers": [
                {"training_seed": seed, "source": ledger_identity}
                for seed, (_terminal, ledger_identity) in retention_ledgers.items()
            ],
        },
    }
    monkeypatch.setattr(finalizer, "_read_json", lambda *_args, **_kwargs: copy.deepcopy(retention_payload))
    monkeypatch.setattr(finalizer, "resolve_contextworld_path", lambda *_args, **_kwargs: retention_output)
    monkeypatch.setattr(cem_aggregate, "_aggregate_output", lambda _track: retention_output)
    monkeypatch.setattr(
        cem_aggregate,
        "_ledger_rows",
        lambda **kwargs: retention_ledgers[int(kwargs["seed"])],
    )
    monkeypatch.setattr(
        cem_aggregate,
        "_expected_retention_records",
        lambda *_args, **_kwargs: (baseline, candidate),
    )
    retention_binding = {"tracks": {"original_task_retention_cem": retention_track}}
    valid_retention = finalizer._speed_cem_aggregate(
        path="artifacts/retention.json",
        completion_id="speed-completion",
        expected_kind="original_task_retention_cem",
        expected_seeds=list(SEEDS),
        expected_development=development,
        speed_cem_binding=(binding_identity, retention_binding),
        repo_root=tmp_path,
    )
    assert valid_retention["decision"]["passed"] is True

    bad_ci = copy.deepcopy(retention_payload)
    bad_ci["checkpoints"][0]["paired_noninferiority"]["gates"][
        "success_rate_non_inferior"
    ] = False
    monkeypatch.setattr(finalizer, "_read_json", lambda *_args, **_kwargs: copy.deepcopy(bad_ci))
    with pytest.raises(ValueError, match="exact paired comparison"):
        finalizer._speed_cem_aggregate(
            path="artifacts/retention.json",
            completion_id="speed-completion",
            expected_kind="original_task_retention_cem",
            expected_seeds=list(SEEDS),
            expected_development=development,
            speed_cem_binding=(binding_identity, retention_binding),
            repo_root=tmp_path,
        )
