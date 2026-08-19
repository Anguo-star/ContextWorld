#!/usr/bin/env python3
"""Freeze one deeply validated three-seed Speed-PLDM CEM aggregate.

The input is not a summary chosen by a caller.  Every source is a completed
append-only ledger reserved by ``run_tworoom_speed_pldm_cem_v1.py`` under the
single passed CEM binding.  This freezer replays the binding's source hashes,
canonical schedules, model-state audits and (for retention) each raw paired
baseline record before it emits an exclusive aggregate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contextworld.benchmarks.speed_pldm_infrastructure_development import (
    EXPECTED_SEEDS,
    identity,
    logical_path,
    resolve_local_output,
    resolve_source,
)
from scripts import run_tworoom_speed_pldm_cem_v1 as runner


COMPLETION_ID = "tworoom_speed_pldm_reference_completion_v1"
CEM_BINDING_ID = "tworoom_speed_pldm_cem_binding_v1"
DEFAULT_BINDING = runner.DEFAULT_BINDING
TRACKS = runner.TRACKS


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _same_identity(left: Any, right: Any) -> bool:
    return bool(
        isinstance(left, Mapping)
        and isinstance(right, Mapping)
        and left.get("path") == right.get("path")
        and left.get("sha256") == right.get("sha256")
        and left.get("size_bytes") == right.get("size_bytes")
    )


def _write_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o644)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _aggregate_output(track: Mapping[str, Any]) -> Path:
    outputs = track.get("outputs")
    if not isinstance(outputs, Mapping) or not isinstance(outputs.get("aggregate"), str):
        raise ValueError("CEM binding lacks an aggregate destination")
    path = resolve_local_output(str(outputs["aggregate"]), repo_root=ROOT)
    root = resolve_local_output(str(outputs["root"]), repo_root=ROOT)
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError("CEM aggregate output escapes its canonical root") from error
    return path


def _ledger_rows(
    *,
    binding: Mapping[str, Any],
    binding_identity: Mapping[str, Any],
    track_name: str,
    track: Mapping[str, Any],
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    output, _work = runner._track_output(track, seed)
    rows = runner._read_ledger(output)
    if not rows:
        raise FileNotFoundError(f"Missing canonical CEM ledger: {output}")
    header = rows[0]
    expected_header = {
        "record_type": "reservation",
        "schema_version": 1,
        "completion_id": COMPLETION_ID,
        "cem_binding_id": CEM_BINDING_ID,
        "binding": dict(binding_identity),
        "track": track_name,
        "training_seed": seed,
        "canonical_ledger": logical_path(output, repo_root=ROOT),
        "reservation_policy": {
            "exclusive_create_before_cem": True,
            "append_only_progress": True,
            "overwrite_permitted": False,
            "resume_requires_exact_binding": True,
        },
    }
    for key, expected in expected_header.items():
        if header.get(key) != expected:
            raise RuntimeError(f"CEM ledger header differs from its frozen reservation: {seed} {key}")
    records, terminal = runner._completed_records(rows)
    if terminal is None:
        raise RuntimeError(f"CEM ledger is reserved but not completed: {output}")
    if not (
        terminal.get("schema_version") == 1
        and terminal.get("completion_id") == COMPLETION_ID
        and terminal.get("cem_binding_id") == CEM_BINDING_ID
        and terminal.get("status") == "completed_exclusive_resumable_cem_ledger"
        and terminal.get("evaluation_kind") == track_name
        and terminal.get("training_seed") == seed
        and terminal.get("binding") == dict(binding_identity)
        and terminal.get("canonical_ledger") == logical_path(output, repo_root=ROOT)
        and terminal.get("result_semantics") == track.get("result_semantics")
        and terminal.get("checkpoint")
        == next(
            row["checkpoint"]
            for row in binding["frozen_chain"]["checkpoints"]
            if row["seed"] == seed
        )
        and terminal.get("normalizer") == binding["tracks"]["shared"]["normalizer"]
        and terminal.get("catalog") == track["catalog"]
        and terminal.get("protocol") == track["protocol"]
        and terminal.get("execution_policy") == expected_header["reservation_policy"]
    ):
        raise RuntimeError(f"CEM terminal receipt is not bound to seed {seed}")
    terminal_records = terminal.get("records")
    if not isinstance(terminal_records, list):
        raise RuntimeError("CEM terminal receipt lacks raw records")
    terminal_by_id = {
        str(row.get("evaluation_id")): row for row in terminal_records if isinstance(row, Mapping)
    }
    if len(terminal_by_id) != len(terminal_records) or terminal_by_id != records:
        raise RuntimeError("CEM terminal receipt does not exactly preserve ledger records")
    audit = terminal.get("frozen_weight_audit")
    checkpoint = terminal["checkpoint"]
    if not (
        isinstance(audit, Mapping)
        and audit.get("passed") is True
        and audit.get("state_dict_sha256_before") == checkpoint.get("model_state_sha256")
        and audit.get("state_dict_sha256_after") == checkpoint.get("model_state_sha256")
        and audit.get("bound_checkpoint_model_state_sha256") == checkpoint.get("model_state_sha256")
    ):
        raise RuntimeError("CEM terminal receipt has an invalid frozen-weight audit")
    current_snapshot = runner._bound_snapshot(binding, track=track_name)
    integrity = terminal.get("input_integrity")
    if not (
        isinstance(integrity, Mapping)
        and integrity.get("all_bound_inputs_unchanged_during_cem") is True
        and integrity.get("identities_after_cem") == integrity.get("identities_before_cem")
        and header.get("input_snapshot_before_reservation")
        == integrity.get("identities_before_cem")
        and integrity.get("identities_before_cem") == current_snapshot
    ):
        raise RuntimeError("CEM terminal receipt is not rooted in the current bound input snapshot")
    return terminal, identity(output, repo_root=ROOT)


def _expected_action_records(track: Mapping[str, Any], records: list[Mapping[str, Any]]) -> None:
    schedule = track.get("schedule")
    if not isinstance(schedule, list) or len(schedule) != 300 or len(records) != 300:
        raise RuntimeError("Action CEM needs exactly 300 scheduled and executed rows")
    actual = {str(row.get("evaluation_id")): row for row in records}
    if len(actual) != 300:
        raise RuntimeError("Action CEM has duplicate evaluation IDs")
    required = (
        "evaluation_id",
        "eval_seed",
        "evaluation_index",
        "repeat_index",
        "cem_seed",
        "query_id",
        "template_id",
        "source_scenario_id",
    )
    for planned in schedule:
        if not isinstance(planned, Mapping):
            raise RuntimeError("Action CEM schedule is malformed")
        evaluation_id = str(planned.get("evaluation_id"))
        record = actual.get(evaluation_id)
        if not isinstance(record, Mapping):
            raise RuntimeError(f"Action CEM is missing scheduled evaluation {evaluation_id}")
        for field in required:
            if record.get(field) != planned.get(field):
                raise RuntimeError(f"Action CEM changed frozen schedule field {field}")
        if not (
            record.get("condition") == "history_mid"
            and record.get("history_relation") == "same"
            and math.isclose(float(record.get("query_speed", float("nan"))), 5.1, abs_tol=1e-6)
            and math.isclose(float(record.get("history_speed", float("nan"))), 5.1, abs_tol=1e-6)
            and isinstance(record.get("success"), bool)
            and isinstance(record.get("final_distance"), (int, float))
            and int(record.get("trajectory", {}).get("raw_steps_executed", -1)) <= 100
        ):
            raise RuntimeError("Action CEM raw record is incomplete or outside the frozen protocol")


def _baseline_records(track: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    baseline = track.get("paired_baseline")
    if not isinstance(baseline, Mapping):
        raise RuntimeError("Retention CEM binding lacks its paired baseline")
    declared = baseline.get("raw_receipts")
    if not isinstance(declared, list) or len(declared) != 6:
        raise RuntimeError("Retention CEM baseline lacks six frozen raw receipts")
    result: dict[str, dict[str, Any]] = {}
    for row in declared:
        if not isinstance(row, Mapping) or not isinstance(row.get("receipt"), Mapping):
            raise RuntimeError("Retention CEM baseline receipt declaration is invalid")
        receipt = row["receipt"]
        source = resolve_source(receipt["path"], repo_root=ROOT)
        if not _same_identity(identity(source, repo_root=ROOT), receipt):
            raise RuntimeError("Retention CEM baseline raw receipt drifted")
        payload = _load_json(source)
        raw = payload.get("raw_records")
        if not isinstance(raw, list) or len(raw) != 50:
            raise RuntimeError("Retention CEM baseline receipt lacks 50 raw records")
        for record in raw:
            if not isinstance(record, Mapping) or not isinstance(record.get("evaluation_id"), str):
                raise RuntimeError("Retention CEM baseline raw record is malformed")
            evaluation_id = str(record["evaluation_id"])
            if evaluation_id in result:
                raise RuntimeError("Retention CEM baseline has duplicate evaluation IDs")
            result[evaluation_id] = dict(record)
    if len(result) != 300 or sum(bool(row["success"]) for row in result.values()) != 278:
        raise RuntimeError("Retention CEM frozen baseline is not the exact 278/300 matrix cell")
    return result


def _expected_retention_records(
    track: Mapping[str, Any], records: list[Mapping[str, Any]]
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    schedule = track.get("schedule")
    if not isinstance(schedule, list) or len(schedule) != 300 or len(records) != 300:
        raise RuntimeError("Retention CEM needs exactly 300 scheduled and executed rows")
    actual = {str(row.get("evaluation_id")): dict(row) for row in records}
    if len(actual) != 300:
        raise RuntimeError("Retention CEM has duplicate evaluation IDs")
    baseline = _baseline_records(track)
    if set(actual) != set(baseline):
        raise RuntimeError("Retention CEM evaluation IDs are not exactly paired to baseline")
    planned = {str(row.get("evaluation_id")): row for row in schedule if isinstance(row, Mapping)}
    if set(planned) != set(actual):
        raise RuntimeError("Retention CEM receipt differs from its frozen schedule")
    paired_fields = (
        "eval_seed",
        "evaluation_index",
        "source_kind",
        "source_path",
        "episode",
        "start_step",
        "goal_offset",
        "cem_group_seed",
        "stratum",
        "room_relation",
        "initial_state",
        "goal_state",
    )
    schedule_fields = (
        "eval_seed",
        "evaluation_index",
        "episode",
        "start_step",
        "goal_offset",
        "cem_group_seed",
    )
    for evaluation_id, candidate in actual.items():
        reference = baseline[evaluation_id]
        for field in paired_fields:
            if candidate.get(field) != reference.get(field):
                raise RuntimeError(f"Retention CEM pairing field drifted at {evaluation_id}: {field}")
        for field in schedule_fields:
            if candidate.get(field) != planned[evaluation_id].get(field):
                raise RuntimeError(f"Retention CEM schedule field drifted at {evaluation_id}: {field}")
        if not (
            isinstance(candidate.get("success"), bool)
            and isinstance(candidate.get("final_distance"), (int, float))
            and candidate.get("execution_track") == "original_task_retention_cem"
        ):
            raise RuntimeError("Retention CEM candidate record lacks an outcome")
    return baseline, actual


def _mean_ci(
    values: np.ndarray, *, seed: int, resamples: int, confidence: float
) -> dict[str, float]:
    if values.ndim != 1 or not len(values):
        raise ValueError("Bootstrap inputs must be a non-empty vector")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(resamples, len(values)))
    means = values[draws].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return {
        "point": float(values.mean()),
        "ci_lower": float(np.quantile(means, alpha)),
        "ci_upper": float(np.quantile(means, 1.0 - alpha)),
    }


def _paired_retention_result(
    *, baseline: Mapping[str, Mapping[str, Any]], candidate: Mapping[str, Mapping[str, Any]], criteria: Mapping[str, Any]
) -> dict[str, Any]:
    keys = sorted(baseline)
    success = np.asarray(
        [float(bool(candidate[key]["success"])) - float(bool(baseline[key]["success"])) for key in keys],
        dtype=np.float64,
    )
    distance = np.asarray(
        [float(candidate[key]["final_distance"]) - float(baseline[key]["final_distance"]) for key in keys],
        dtype=np.float64,
    )
    seed = int(criteria["paired_bootstrap_seed"])
    resamples = int(criteria["paired_bootstrap_resamples"])
    confidence = float(criteria["confidence_level"])
    success_ci = _mean_ci(success, seed=seed, resamples=resamples, confidence=confidence)
    distance_ci = _mean_ci(
        distance, seed=seed ^ 0xD157A, resamples=resamples, confidence=confidence
    )
    strata: dict[str, list[str]] = {}
    for key in keys:
        relation = str(baseline[key]["room_relation"])
        strata.setdefault(relation, []).append(key)
    collapsed: list[str] = []
    stratum_results: dict[str, Any] = {}
    for relation, selected in sorted(strata.items()):
        reference_successes = sum(bool(baseline[key]["success"]) for key in selected)
        candidate_successes = sum(bool(candidate[key]["success"]) for key in selected)
        is_collapsed = reference_successes > 0 and candidate_successes == 0
        if is_collapsed:
            collapsed.append(relation)
        stratum_results[relation] = {
            "evaluations": len(selected),
            "reference_successes": int(reference_successes),
            "candidate_successes": int(candidate_successes),
            "collapsed": is_collapsed,
        }
    gates = {
        "success_rate_non_inferior": success_ci["ci_lower"]
        >= float(criteria["success_rate_delta_lower_bound"]),
        "final_distance_non_inferior": distance_ci["ci_upper"]
        <= float(criteria["final_distance_delta_upper_bound_px"]),
        "no_solvable_room_relation_stratum_collapse": not collapsed,
    }
    return {
        "evaluations": len(keys),
        "candidate_minus_reference_success_rate": success_ci,
        "candidate_minus_reference_final_distance_px": distance_ci,
        "criteria": dict(criteria),
        "room_relation_strata": stratum_results,
        "collapsed_solvable_room_relation_strata": collapsed,
        "gates": gates,
        "passed": all(gates.values()),
    }


def _action_payload(
    *, binding: Mapping[str, Any], binding_identity: Mapping[str, Any], track: Mapping[str, Any], ledgers: list[tuple[dict[str, Any], dict[str, Any]]]
) -> dict[str, Any]:
    checkpoints = []
    for terminal, ledger_identity in ledgers:
        records = terminal["records"]
        _expected_action_records(track, records)
        aggregate = terminal["aggregate"]
        if not (
            isinstance(aggregate, Mapping)
            and aggregate.get("evaluations") == 300
            and aggregate.get("successes") == sum(bool(row["success"]) for row in records)
            and math.isclose(
                float(aggregate.get("success_rate")),
                float(sum(bool(row["success"]) for row in records) / 300),
                rel_tol=0.0,
                abs_tol=0.0,
            )
        ):
            raise RuntimeError("Action CEM ledger aggregate does not match raw rows")
        checkpoints.append(
            {
                "training_seed": terminal["training_seed"],
                "value": float(aggregate["success_rate"]),
                "execution_valid": True,
                "source": ledger_identity,
            }
        )
    checkpoints.sort(key=lambda row: int(row["training_seed"]))
    return {
        "schema_version": 1,
        "completion_id": COMPLETION_ID,
        "cem_binding_id": CEM_BINDING_ID,
        "binding": dict(binding_identity),
        "status": "completed_executed_valid_descriptive",
        "evaluation_kind": "action_planning_cem",
        "result_semantics": "EXECUTED_VALID_DESCRIPTIVE",
        "metric": {
            "id": track["metric"]["id"],
            "label": "Frozen same-speed closed-loop planning success rate (descriptive)",
        },
        "checkpoints": checkpoints,
        "decision": {
            "execution_valid": True,
            "model_performance_gate": None,
            "retention_result": "NOT_APPLICABLE",
            "result": "EXECUTED_VALID_DESCRIPTIVE",
        },
        "development": binding["frozen_chain"]["development"],
    }


def _retention_payload(
    *, binding: Mapping[str, Any], binding_identity: Mapping[str, Any], track: Mapping[str, Any], ledgers: list[tuple[dict[str, Any], dict[str, Any]]]
) -> dict[str, Any]:
    criteria = track.get("metric", {}).get("paired_noninferiority")
    if not isinstance(criteria, Mapping):
        raise RuntimeError("Retention CEM has no frozen paired non-inferiority criteria")
    checkpoints = []
    for terminal, ledger_identity in ledgers:
        baseline, candidate = _expected_retention_records(track, terminal["records"])
        result = _paired_retention_result(
            baseline=baseline, candidate=candidate, criteria=criteria
        )
        aggregate = terminal["aggregate"]
        if not (
            isinstance(aggregate, Mapping)
            and aggregate.get("evaluations") == 300
            and aggregate.get("successes") == sum(bool(row["success"]) for row in candidate.values())
        ):
            raise RuntimeError("Retention CEM ledger aggregate does not match raw rows")
        checkpoints.append(
            {
                "training_seed": terminal["training_seed"],
                "value": float(aggregate["success_rate"]),
                "passed": bool(result["passed"]),
                "source": ledger_identity,
                "paired_noninferiority": result,
            }
        )
    checkpoints.sort(key=lambda row: int(row["training_seed"]))
    all_passed = all(bool(row["passed"]) for row in checkpoints)
    return {
        "schema_version": 1,
        "completion_id": COMPLETION_ID,
        "cem_binding_id": CEM_BINDING_ID,
        "binding": dict(binding_identity),
        "status": "completed_paired_retention_evaluation",
        "evaluation_kind": "original_task_retention_cem",
        "result_semantics": "PAIRED_NONINFERIORITY_RETENTION",
        "metric": {
            "id": track["metric"]["id"],
            "label": "Original TwoRoom CEM paired non-inferiority retention",
        },
        "checkpoints": checkpoints,
        "decision": {
            "all_training_seeds_passed": all_passed,
            "passed": all_passed,
            "result": "PASS" if all_passed else "FAIL",
            "criterion": "all_three_fixed_checkpoints_must_pass_paired_noninferiority",
        },
        "paired_baseline": track["paired_baseline"],
        "development": binding["frozen_chain"]["development"],
    }


def freeze(*, binding_path: Path, track_name: str) -> dict[str, Any]:
    if track_name not in TRACKS:
        raise ValueError(f"Unknown CEM track: {track_name}")
    # runner validates the whole binding source closure before we touch a
    # ledger.  Calling it for every seed additionally verifies canonical
    # destinations and the three fixed checkpoint identities.
    validated = [
        runner._validate_binding(binding_path, track=track_name, seed=seed)
        for seed in EXPECTED_SEEDS
    ]
    binding, track, _checkpoint, _output, _work = validated[0]
    if any(item[0] != binding or item[1] != track for item in validated[1:]):
        raise RuntimeError("CEM binding changed while aggregate inputs were checked")
    resolved_binding = resolve_source(binding_path, repo_root=ROOT)
    binding_identity = identity(resolved_binding, repo_root=ROOT)
    before = runner._bound_snapshot(binding, track=track_name)
    output = _aggregate_output(track)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite CEM aggregate: {output}")
    ledgers = [
        _ledger_rows(
            binding=binding,
            binding_identity=binding_identity,
            track_name=track_name,
            track=track,
            seed=seed,
        )
        for seed in EXPECTED_SEEDS
    ]
    payload = (
        _action_payload(
            binding=binding, binding_identity=binding_identity, track=track, ledgers=ledgers
        )
        if track_name == "action_planning_cem"
        else _retention_payload(
            binding=binding, binding_identity=binding_identity, track=track, ledgers=ledgers
        )
    )
    after = runner._bound_snapshot(binding, track=track_name)
    if before != after:
        raise RuntimeError("Bound CEM inputs changed while aggregate was frozen")
    payload["output"] = {
        "path": logical_path(output, repo_root=ROOT),
        "content_sha256_not_embedded_to_avoid_self_reference": True,
    }
    payload["input_integrity"] = {
        "all_bound_inputs_unchanged_during_aggregate_read": True,
        "identities_before_aggregate_read": before,
        "identities_after_aggregate_read": after,
        "ledgers": [
            {
                "training_seed": terminal["training_seed"],
                "source": ledger_identity,
            }
            for terminal, ledger_identity in ledgers
        ],
    }
    _write_exclusive(output, payload)
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding", type=Path, default=DEFAULT_BINDING)
    parser.add_argument("--track", choices=TRACKS, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    payload = freeze(binding_path=args.binding, track_name=args.track)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "track": payload["evaluation_kind"],
                "output": payload["output"]["path"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
