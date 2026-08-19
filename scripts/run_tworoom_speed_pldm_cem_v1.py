#!/usr/bin/env python3
"""Run one exclusively reserved Speed-PLDM CEM ledger.

This runner is intentionally narrower than the historical planning scripts.
It accepts only a passed ``speed_pldm_cem_binding_v1`` and one of the two
already registered tracks.  The canonical per-seed artifact is an append-only
JSONL ledger: its first line reserves the output *before* model/environment
work begins, every completed evaluation is durably appended, and the last
line is the immutable completed receipt.  Consequently a killed job can be
resumed without overwriting evidence or changing an already executed query.

The script never offers a ``--skip-catalog-replay`` option.  Speed catalog
replay is mandatory, and both tracks verify the bound identities immediately
before and after the execution window.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import socket
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np
import yaml


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


COMPLETION_ID = "tworoom_speed_pldm_reference_completion_v1"
CEM_BINDING_ID = "tworoom_speed_pldm_cem_binding_v1"
DEFAULT_BINDING = (
    ROOT
    / "artifacts/evaluation/history3/tworoom_speed_pldm_reference_completion_v1"
    / "formal_icl_v1/cem_binding_v1.json"
)
TRACKS = ("action_planning_cem", "original_task_retention_cem")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return value


def _same_identity(left: Any, right: Any) -> bool:
    return bool(
        isinstance(left, Mapping)
        and isinstance(right, Mapping)
        and left.get("path") == right.get("path")
        and left.get("sha256") == right.get("sha256")
        and left.get("size_bytes") == right.get("size_bytes")
    )


def _source_identity(value: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    raw = value.get("path")
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{label} lacks a source path")
    observed = identity(resolve_source(raw, repo_root=ROOT), repo_root=ROOT)
    if not _same_identity(observed, value):
        raise RuntimeError(f"{label} identity drifted")
    return observed


def _identity_from_path(path: Path) -> dict[str, Any]:
    return identity(path.resolve(), repo_root=ROOT)


def _read_ledger(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"Ledger line {line_number} is not an object: {path}")
        rows.append(value)
    return rows


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _append_locked(path: Path, payload: Mapping[str, Any]) -> None:
    """Append one durable JSONL event while holding an advisory exclusive lock."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        os.write(descriptor, _json_bytes(payload))
        os.fsync(descriptor)
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _write_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o644)
    try:
        os.write(descriptor, _json_bytes(payload))
        os.fsync(descriptor)
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    finally:
        os.close(descriptor)


@contextmanager
def _seed_lock(work: Path) -> Iterator[None]:
    """Prevent two live processes from extending the same reserved ledger."""

    work.mkdir(parents=True, exist_ok=True)
    lock_path = work / "runner.lock"
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"A live runner already owns {lock_path}") from error
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _all_identity_mappings(value: Any) -> Iterator[dict[str, Any]]:
    """Yield unique, fully specified immutable source mappings recursively."""

    seen: set[tuple[str, str, int]] = set()

    def visit(item: Any) -> Iterator[dict[str, Any]]:
        if isinstance(item, Mapping):
            if (
                isinstance(item.get("path"), str)
                and isinstance(item.get("sha256"), str)
                and isinstance(item.get("size_bytes"), int)
            ):
                key = (str(item["path"]), str(item["sha256"]), int(item["size_bytes"]))
                if key not in seen:
                    seen.add(key)
                    yield dict(item)
            for nested in item.values():
                yield from visit(nested)
        elif isinstance(item, list):
            for nested in item:
                yield from visit(nested)

    yield from visit(value)


def _bound_snapshot(binding: Mapping[str, Any], *, track: str) -> dict[str, Any]:
    """Rehash every input exposed to one run, including its pre-Public chain."""

    selected = {
        "preregistration": binding.get("preregistration"),
        "frozen_chain": binding.get("frozen_chain"),
        "shared": binding.get("tracks", {}).get("shared"),
        "track": binding.get("tracks", {}).get(track),
    }
    observed = []
    for item in _all_identity_mappings(selected):
        observed.append(_source_identity(item, label=f"bound input {item['path']}"))
    observed.sort(key=lambda row: (str(row["path"]), str(row["sha256"])))
    return {"identities": observed}


def _binding_identity(binding_path: Path) -> dict[str, Any]:
    return _identity_from_path(binding_path)


def _track_output(track_payload: Mapping[str, Any], seed: int) -> tuple[Path, Path]:
    outputs = track_payload.get("outputs")
    if not isinstance(outputs, Mapping):
        raise ValueError("CEM track lacks its output contract")
    rows = outputs.get("receipts")
    if not isinstance(rows, list):
        raise ValueError("CEM track lacks receipt destinations")
    matches = [row for row in rows if isinstance(row, Mapping) and row.get("seed") == seed]
    if len(matches) != 1:
        raise ValueError(f"CEM track lacks exactly one output for seed {seed}")
    receipt = matches[0]
    path_value = receipt.get("path")
    work_value = receipt.get("work")
    if not isinstance(path_value, str) or not isinstance(work_value, str):
        raise ValueError("CEM output destination is incomplete")
    output = resolve_local_output(path_value, repo_root=ROOT)
    work = resolve_local_output(work_value, repo_root=ROOT)
    root_value = outputs.get("root")
    if not isinstance(root_value, str):
        raise ValueError("CEM output root is invalid")
    root = resolve_local_output(root_value, repo_root=ROOT)
    try:
        output.relative_to(root)
        work.relative_to(root)
    except ValueError as error:
        raise ValueError("CEM output escapes its canonical track root") from error
    return output, work


def _validate_binding(
    binding_path: Path, *, track: str, seed: int
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Path, Path]:
    """Load the positive branch and fail closed on every frozen identity."""

    if track not in TRACKS:
        raise ValueError(f"Unknown CEM track: {track}")
    if seed not in EXPECTED_SEEDS:
        raise ValueError(f"Training seed is not preregistered: {seed}")
    resolved = resolve_source(binding_path, repo_root=ROOT)
    binding = _load_json(resolved)
    tracks = binding.get("tracks")
    chain = binding.get("frozen_chain")
    if not (
        binding.get("schema_version") == 1
        and binding.get("cem_binding_id") == CEM_BINDING_ID
        and binding.get("completion_id") == COMPLETION_ID
        and binding.get("status") == "frozen_after_passed_three_seed_public_icl_before_cem"
        and binding.get("passed") is True
        and binding.get("cem") == {"authorized": True, "executed": False}
        and isinstance(tracks, Mapping)
        and set(tracks) == {"shared", *TRACKS}
        and isinstance(chain, Mapping)
        and isinstance(binding.get("input_integrity"), Mapping)
        and binding["input_integrity"].get("all_frozen_inputs_unchanged_during_binding")
        is True
        and binding["input_integrity"].get("identities_after_binding")
        == binding["input_integrity"].get("identities_before_binding")
    ):
        raise RuntimeError("Speed CEM binding is not an intact positive pre-CEM branch")
    evaluation_binding_identity = chain.get("evaluation_binding_config")
    evaluation_receipt_identity = chain.get("evaluation_binding_receipt")
    prepublic_authority = chain.get("prepublic_cem_authority")
    if not (
        isinstance(evaluation_binding_identity, Mapping)
        and isinstance(evaluation_receipt_identity, Mapping)
        and isinstance(prepublic_authority, Mapping)
        and _source_identity(
            evaluation_binding_identity, label="CEM-chain evaluation binding"
        )
        and _source_identity(
            evaluation_receipt_identity, label="CEM-chain evaluation binding receipt"
        )
    ):
        raise RuntimeError("Speed CEM binding lacks its immutable pre-Public bridge")
    evaluation_binding = _load_yaml(
        resolve_source(evaluation_binding_identity["path"], repo_root=ROOT)
    )
    evaluation_receipt = _load_json(
        resolve_source(evaluation_receipt_identity["path"], repo_root=ROOT)
    )
    # Reuse the CEM-binding freezer's pure, no-inference closure validator.
    # It proves every CEM runner/core/criterion was selected in the evaluation
    # binding before Public ICL, not reconstructed after the 3/3 outcome.
    from scripts.freeze_tworoom_speed_pldm_cem_binding_v1 import (
        CEM_PREREG,
        _prepublic_cem_authority,
        _validate_static_prereg,
    )

    static, static_sources = _validate_static_prereg(CEM_PREREG)
    validated_prepublic = _prepublic_cem_authority(
        binding=evaluation_binding, static=static, static_sources=static_sources
    )
    if not (
        prepublic_authority == validated_prepublic
        and evaluation_binding.get("cem_protocol") == validated_prepublic
        and evaluation_receipt.get("cem_protocol") == validated_prepublic
        and binding.get("preregistration")
        == validated_prepublic.get("preregistration")
    ):
        raise RuntimeError("Speed CEM binding is not rooted in the pre-Public CEM closure")
    track_payload = tracks.get(track)
    shared = tracks.get("shared")
    if not isinstance(track_payload, Mapping) or not isinstance(shared, Mapping):
        raise RuntimeError("Speed CEM binding lacks a shared or requested track contract")
    if track_payload.get("evaluation_kind") != track:
        raise RuntimeError("Speed CEM track kind differs from requested runner")
    if track == "action_planning_cem":
        if not (
            track_payload.get("result_semantics") == "EXECUTED_VALID_DESCRIPTIVE"
            and track_payload.get("metric", {}).get("performance_threshold") is None
            and track_payload.get("metric", {}).get("pass_threshold") is None
        ):
            raise RuntimeError("Action-planning CEM is not bound as descriptive-only")
    else:
        noninferiority = track_payload.get("metric", {}).get("paired_noninferiority")
        if not (
            track_payload.get("result_semantics") == "PAIRED_NONINFERIORITY_RETENTION"
            and isinstance(noninferiority, Mapping)
            and noninferiority.get("success_rate_delta_lower_bound") == -0.05
            and noninferiority.get("final_distance_delta_upper_bound_px") == 5.0
            and noninferiority.get("confidence_level") == 0.95
            and noninferiority.get("paired_bootstrap_seed") == 3072
            and noninferiority.get("paired_bootstrap_resamples") == 10000
            and noninferiority.get("stratum_definition") == "room_relation"
            and noninferiority.get("require_no_solvable_room_relation_stratum_collapse")
            is True
        ):
            raise RuntimeError("Retention CEM is not bound to its paired non-inferiority rule")
    checkpoints = chain.get("checkpoints")
    by_seed = {
        int(row.get("seed", -1)): row
        for row in checkpoints
        if isinstance(row, Mapping) and isinstance(row.get("seed"), int)
    } if isinstance(checkpoints, list) else {}
    if set(by_seed) != set(EXPECTED_SEEDS):
        raise RuntimeError("Speed CEM binding does not include all fixed checkpoints")
    checkpoint = by_seed[seed]
    checkpoint_identity = checkpoint.get("checkpoint")
    if not isinstance(checkpoint_identity, Mapping):
        raise RuntimeError("Speed CEM binding lacks checkpoint identity")
    _source_identity(checkpoint_identity, label=f"checkpoint {seed}")
    state = checkpoint_identity.get("model_state_sha256")
    if not isinstance(state, str) or len(state) != 64:
        raise RuntimeError("Speed CEM binding lacks checkpoint model-state hash")
    # This recursively rehashes the pre-Public source set before any planner
    # construction.  It never accepts an identity selected after Public ICL.
    _bound_snapshot(binding, track=track)
    output, work = _track_output(track_payload, seed)
    return binding, dict(track_payload), dict(checkpoint), output, work


def _reservation(
    *,
    binding: Mapping[str, Any],
    binding_identity: Mapping[str, Any],
    track: str,
    seed: int,
    output: Path,
    snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "record_type": "reservation",
        "schema_version": 1,
        "completion_id": COMPLETION_ID,
        "cem_binding_id": CEM_BINDING_ID,
        "binding": dict(binding_identity),
        "track": track,
        "training_seed": seed,
        "canonical_ledger": logical_path(output, repo_root=ROOT),
        "reservation_policy": {
            "exclusive_create_before_cem": True,
            "append_only_progress": True,
            "overwrite_permitted": False,
            "resume_requires_exact_binding": True,
        },
        "input_snapshot_before_reservation": dict(snapshot),
        "runner_host": socket.gethostname(),
    }


def _reserve_or_resume(
    *,
    output: Path,
    expected: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if not output.exists():
        _write_exclusive(output, expected)
        return [dict(expected)]
    rows = _read_ledger(output)
    if not rows or rows[0] != dict(expected):
        raise RuntimeError("Existing CEM ledger is not this exact reserved job")
    return rows


def _completed_records(rows: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[str, Any] | None]:
    records: dict[str, dict[str, Any]] = {}
    terminal: dict[str, Any] | None = None
    for row in rows[1:]:
        kind = row.get("record_type")
        if kind == "evaluation":
            record = row.get("record")
            if not isinstance(record, Mapping) or not isinstance(record.get("evaluation_id"), str):
                raise ValueError("CEM ledger has malformed evaluation event")
            key = str(record["evaluation_id"])
            if key in records:
                raise RuntimeError(f"CEM ledger contains duplicate evaluation: {key}")
            records[key] = dict(record)
        elif kind == "completed_receipt":
            if terminal is not None:
                raise RuntimeError("CEM ledger has multiple terminal receipts")
            receipt = row.get("receipt")
            if not isinstance(receipt, Mapping):
                raise ValueError("CEM ledger terminal receipt is malformed")
            terminal = dict(receipt)
        else:
            raise ValueError(f"CEM ledger has unknown event type: {kind!r}")
    return records, terminal


def _action_args(*, device: str, eval_seed: int, protocol: Mapping[str, Any]) -> argparse.Namespace:
    return argparse.Namespace(
        device=device,
        seed=int(eval_seed),
        eval_budget=int(protocol["eval_budget_raw_steps"]),
        img_size=224,
        horizon=int(protocol["horizon_action_blocks"]),
        receding_horizon=int(protocol["receding_horizon_action_blocks"]),
        cem_batch_size=1,
        cem_num_samples=int(protocol["cem_samples"]),
        cem_var_scale=float(protocol["cem_var_scale"]),
        cem_steps=int(protocol["cem_iterations"]),
        cem_topk=int(protocol["cem_topk"]),
    )


def _run_action_track(
    *,
    binding: Mapping[str, Any],
    track: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    existing: Mapping[str, Mapping[str, Any]],
    append: Any,
    device: str,
) -> tuple[list[dict[str, Any]], str, str, dict[str, Any]]:
    """Execute only missing one-at-a-time fixed-context action evaluations."""

    from contextworld.evaluation.icl_catalog import validate_context_query_catalog
    from contextworld.evaluation.icl_model import state_dict_sha256
    from contextworld.evaluation.protocol import frozen_normalizer_process, infer_model_protocol, load_pretrained_cost_model
    from contextworld.evaluation.tworoom import register_tworoom_eval_env
    from contextworld.paths import artifact_path
    from contextworld.synthesis.stablewm import load_stable_worldmodel
    from scripts.eval_tworoom_icl_planning import _load_query_assets, _run_one

    catalog_path = resolve_source(track["catalog"]["path"], repo_root=ROOT)
    validation = validate_context_query_catalog(
        catalog_path, repo_root=ROOT, replay_simulator=True, family="speed"
    )
    if validation.get("passed") is not True:
        raise RuntimeError(f"Mandatory Speed catalog replay failed: {validation.get('failures')}")
    catalog = _load_json(catalog_path)
    bundles = catalog.get("bundles")
    if not isinstance(bundles, list):
        raise RuntimeError("Bound action catalog has no bundles")
    by_query = {str(bundle.get("query_id")): bundle for bundle in bundles if isinstance(bundle, Mapping)}
    schedule = track.get("schedule")
    if not isinstance(schedule, list) or len(schedule) != 300:
        raise RuntimeError("Bound action schedule is not exactly 300 evaluations")
    if len({str(row.get("evaluation_id")) for row in schedule if isinstance(row, Mapping)}) != 300:
        raise RuntimeError("Bound action schedule has duplicate identifiers")
    shared = binding["tracks"]["shared"]
    runtime = shared["stable_worldmodel"]
    checkpoint_path = resolve_source(checkpoint["checkpoint"]["path"], repo_root=ROOT)
    normalizer_path = resolve_source(shared["normalizer"]["path"], repo_root=ROOT)
    swm, stable_repo, stable_commit = load_stable_worldmodel(
        ROOT, runtime["worktree"], runtime["commit"]
    )
    if stable_commit != runtime["commit"]:
        raise RuntimeError("Stable-WorldModel runtime commit drifted during action CEM")
    register_tworoom_eval_env()
    process = frozen_normalizer_process(normalizer_path)
    model = load_pretrained_cost_model(
        checkpoint_path, swm, cache_dir=artifact_path("evaluation/model_cache", repo_root=ROOT)
    )
    protocol = infer_model_protocol(model, action_dim=2)
    if protocol != {"action_block": 5, "history_size": 3}:
        raise RuntimeError(f"Action CEM model protocol drifted: {protocol}")
    model = model.to(device).eval()
    model.requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Action CEM model is not frozen")
    setattr(model, "history_size", 3)
    setattr(model, "interpolate_pos_encoding", True)
    before = state_dict_sha256(model)
    expected_state = checkpoint["checkpoint"]["model_state_sha256"]
    if before != expected_state:
        raise RuntimeError("Action CEM loaded model state differs from the bound checkpoint")

    assets: dict[str, dict[str, Any]] = {}
    records = {key: dict(value) for key, value in existing.items()}
    for position, scheduled in enumerate(schedule, 1):
        if not isinstance(scheduled, Mapping):
            raise RuntimeError("Bound action schedule has a malformed row")
        evaluation_id = str(scheduled.get("evaluation_id", ""))
        if evaluation_id in records:
            continue
        query_id = str(scheduled.get("query_id", ""))
        bundle = by_query.get(query_id)
        if not isinstance(bundle, Mapping):
            raise RuntimeError(f"Bound action schedule query is missing: {query_id}")
        asset = assets.get(query_id)
        if asset is None:
            asset = _load_query_assets(dict(bundle), process=process)
            assets[query_id] = asset
        if not (
            asset["episode"].template_id == scheduled.get("template_id")
            and asset["episode"].scenario_id == scheduled.get("source_scenario_id")
            and np.isclose(float(asset["episode"].speed), 5.1, rtol=0.0, atol=1e-6)
        ):
            raise RuntimeError("Bound action schedule no longer matches its catalog query")
        print(f"[action {position}/300] seed={checkpoint['seed']} eval={evaluation_id}", flush=True)
        result = _run_one(
            args=_action_args(
                device=device,
                eval_seed=int(scheduled["eval_seed"]),
                protocol=track["protocol"],
            ),
            swm=swm,
            model=model,
            process=process,
            protocol=protocol,
            asset=asset,
            condition="history_mid",
            evaluation_id=evaluation_id,
            evaluation_index=int(scheduled["evaluation_index"]),
            repeat_index=int(scheduled["repeat_index"]),
            cem_seed=int(scheduled["cem_seed"]),
        )
        result.update(
            {
                "query_speed": 5.1,
                "history_speed": 5.1,
                "history_relation": "same",
                "execution_track": "action_planning_cem",
            }
        )
        append({"record_type": "evaluation", "record": result})
        records[evaluation_id] = result
    after = state_dict_sha256(model)
    if before != after:
        raise RuntimeError("Action CEM model weights changed during execution")
    ordered = [records[str(row["evaluation_id"])] for row in schedule]
    return ordered, before, after, {
        "catalog_replay": validation,
        "stable_worldmodel": {"repo": str(stable_repo), "commit": stable_commit},
        "model_protocol": protocol,
    }


def _retention_args(*, device: str, protocol: Mapping[str, Any]) -> argparse.Namespace:
    return argparse.Namespace(
        device=device,
        eval_budget=int(protocol["eval_budget_raw_steps"]),
        horizon=int(protocol["horizon_action_blocks"]),
        receding_horizon=int(protocol["receding_horizon_action_blocks"]),
        cem_samples=int(protocol["cem_samples"]),
        cem_steps=int(protocol["cem_iterations"]),
        cem_topk=int(protocol["cem_topk"]),
        expected_history_size=3,
    )


def _run_retention_track(
    *,
    binding: Mapping[str, Any],
    track: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    existing: Mapping[str, Mapping[str, Any]],
    append: Any,
    device: str,
) -> tuple[list[dict[str, Any]], str, str, dict[str, Any]]:
    """Execute complete 50-row seed groups so a resumed run preserves CEM batches."""

    from contextworld.evaluation.icl_model import state_dict_sha256
    from contextworld.evaluation.protocol import frozen_normalizer_process, infer_model_protocol, load_pretrained_cost_model
    from contextworld.evaluation.tworoom import register_tworoom_eval_env
    from contextworld.paths import artifact_path
    from contextworld.synthesis.stablewm import load_stable_worldmodel
    from scripts.eval_tworoom_ability_catalog import _run_group

    catalog_path = resolve_source(track["catalog"]["path"], repo_root=ROOT)
    catalog = _load_json(catalog_path)
    entries = catalog.get("entries")
    if not (
        catalog.get("schema_version") == 1
        and catalog.get("catalog") == "tworoom_original_heldout_eval_catalog_v1"
        and isinstance(entries, list)
        and len(entries) == 300
    ):
        raise RuntimeError("Bound retention catalog does not have the registered 300-row schema")
    schedule = track.get("schedule")
    if not isinstance(schedule, list) or len(schedule) != 300:
        raise RuntimeError("Bound retention schedule is not exactly 300 evaluations")
    by_schedule = {str(row.get("evaluation_id")): row for row in schedule if isinstance(row, Mapping)}
    by_entry = {str(row.get("evaluation_id")): row for row in entries if isinstance(row, Mapping)}
    if set(by_schedule) != set(by_entry) or len(by_schedule) != 300:
        raise RuntimeError("Retention schedule differs from its bound catalog")
    for evaluation_id, scheduled in by_schedule.items():
        entry = by_entry[evaluation_id]
        for field in ("eval_seed", "evaluation_index", "episode", "start_step", "goal_offset", "cem_group_seed"):
            if entry.get(field) != scheduled.get(field):
                raise RuntimeError(f"Retention schedule drifted at {evaluation_id}: {field}")

    shared = binding["tracks"]["shared"]
    runtime = shared["stable_worldmodel"]
    checkpoint_path = resolve_source(checkpoint["checkpoint"]["path"], repo_root=ROOT)
    normalizer_path = resolve_source(shared["normalizer"]["path"], repo_root=ROOT)
    swm, stable_repo, stable_commit = load_stable_worldmodel(
        ROOT, runtime["worktree"], runtime["commit"]
    )
    if stable_commit != runtime["commit"]:
        raise RuntimeError("Stable-WorldModel runtime commit drifted during retention CEM")
    register_tworoom_eval_env()
    process = frozen_normalizer_process(normalizer_path)
    model = load_pretrained_cost_model(
        checkpoint_path, swm, cache_dir=artifact_path("evaluation/model_cache", repo_root=ROOT)
    )
    protocol = infer_model_protocol(model, action_dim=2)
    if protocol != {"action_block": 5, "history_size": 3}:
        raise RuntimeError(f"Retention CEM model protocol drifted: {protocol}")
    model = model.to(device).eval()
    model.requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Retention CEM model is not frozen")
    setattr(model, "history_size", 3)
    setattr(model, "interpolate_pos_encoding", True)
    before = state_dict_sha256(model)
    expected_state = checkpoint["checkpoint"]["model_state_sha256"]
    if before != expected_state:
        raise RuntimeError("Retention CEM loaded model state differs from the bound checkpoint")

    records = {key: dict(value) for key, value in existing.items()}
    args = _retention_args(device=device, protocol=track["protocol"])
    for eval_seed in (42, 43, 44, 45, 46, 47):
        group = [entry for entry in entries if int(entry["eval_seed"]) == eval_seed]
        group.sort(key=lambda row: int(row["evaluation_index"]))
        done = [str(row["evaluation_id"]) in records for row in group]
        if all(done):
            continue
        if any(done):
            # The runner appends a group only after _run_group returns.  A
            # partial group would otherwise silently change batch/CEM RNG.
            raise RuntimeError(f"Retention ledger contains a partial eval-seed group: {eval_seed}")
        print(f"[retention {eval_seed}] seed={checkpoint['seed']} evaluations=50", flush=True)
        generated = _run_group(
            args=args,
            swm=swm,
            model=model,
            protocol=protocol,
            process=process,
            entries=group,
        )
        if len(generated) != 50:
            raise RuntimeError("Retention core did not return its complete 50-row group")
        for record in generated:
            evaluation_id = str(record.get("evaluation_id", ""))
            expected = by_schedule.get(evaluation_id)
            if expected is None:
                raise RuntimeError("Retention core returned an unbound evaluation")
            for field in ("eval_seed", "evaluation_index", "episode", "start_step", "goal_offset", "cem_group_seed"):
                if record.get(field) != expected.get(field):
                    raise RuntimeError(f"Retention core changed bound field {field}")
            record["execution_track"] = "original_task_retention_cem"
            append({"record_type": "evaluation", "record": record})
            records[evaluation_id] = record
    after = state_dict_sha256(model)
    if before != after:
        raise RuntimeError("Retention CEM model weights changed during execution")
    ordered = [records[str(row["evaluation_id"])] for row in schedule]
    return ordered, before, after, {
        "catalog_replay": {
            "required": True,
            "kind": "registered_original_heldout_catalog_schema_and_exact_schedule",
            "passed": True,
        },
        "stable_worldmodel": {"repo": str(stable_repo), "commit": stable_commit},
        "model_protocol": protocol,
    }


def _summary(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    successes = sum(bool(row["success"]) for row in records)
    distances = [float(row["final_distance"]) for row in records]
    return {
        "evaluations": len(records),
        "successes": int(successes),
        "success_rate": float(successes / len(records)),
        "mean_final_distance_px": float(np.mean(distances)),
    }


def run(
    *,
    binding_path: Path,
    track: str,
    seed: int,
    device: str,
    resume: bool,
) -> dict[str, Any]:
    """Run or resume the one exact CEM job selected by the frozen binding."""

    binding, bound_track, checkpoint, output, work = _validate_binding(
        binding_path, track=track, seed=seed
    )
    binding_path = resolve_source(binding_path, repo_root=ROOT)
    binding_identity = _binding_identity(binding_path)
    before_snapshot = _bound_snapshot(binding, track=track)
    reservation = _reservation(
        binding=binding,
        binding_identity=binding_identity,
        track=track,
        seed=seed,
        output=output,
        snapshot=before_snapshot,
    )
    with _seed_lock(work):
        if output.exists() and not resume:
            raise FileExistsError(
                f"Canonical CEM ledger already exists; use --resume only for this exact reservation: {output}"
            )
        rows = _reserve_or_resume(output=output, expected=reservation)
        existing, terminal = _completed_records(rows)
        if terminal is not None:
            return terminal
        append = lambda event: _append_locked(output, event)
        if track == "action_planning_cem":
            records, state_before, state_after, runtime = _run_action_track(
                binding=binding,
                track=bound_track,
                checkpoint=checkpoint,
                existing=existing,
                append=append,
                device=device,
            )
        else:
            records, state_before, state_after, runtime = _run_retention_track(
                binding=binding,
                track=bound_track,
                checkpoint=checkpoint,
                existing=existing,
                append=append,
                device=device,
            )
        expected_count = int(bound_track["expected"]["episodes_per_checkpoint"])
        if len(records) != expected_count or len({row["evaluation_id"] for row in records}) != expected_count:
            raise RuntimeError("CEM execution did not complete its exact frozen schedule")
        after_snapshot = _bound_snapshot(binding, track=track)
        if after_snapshot != before_snapshot:
            raise RuntimeError("A CEM input changed while the execution ledger was open")
        receipt = {
            "schema_version": 1,
            "completion_id": COMPLETION_ID,
            "cem_binding_id": CEM_BINDING_ID,
            "status": "completed_exclusive_resumable_cem_ledger",
            "evaluation_kind": track,
            "training_seed": seed,
            "binding": binding_identity,
            "canonical_ledger": logical_path(output, repo_root=ROOT),
            "result_semantics": bound_track["result_semantics"],
            "checkpoint": checkpoint["checkpoint"],
            "normalizer": binding["tracks"]["shared"]["normalizer"],
            "catalog": bound_track["catalog"],
            "protocol": bound_track["protocol"],
            "records": records,
            "aggregate": _summary(records),
            "frozen_weight_audit": {
                "state_dict_sha256_before": state_before,
                "state_dict_sha256_after": state_after,
                "bound_checkpoint_model_state_sha256": checkpoint["checkpoint"]["model_state_sha256"],
                "passed": state_before == state_after == checkpoint["checkpoint"]["model_state_sha256"],
            },
            "runtime": runtime,
            "input_integrity": {
                "all_bound_inputs_unchanged_during_cem": True,
                "identities_before_cem": before_snapshot,
                "identities_after_cem": after_snapshot,
            },
            "execution_policy": reservation["reservation_policy"],
        }
        append({"record_type": "completed_receipt", "receipt": receipt})
        return receipt


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding", type=Path, default=DEFAULT_BINDING)
    parser.add_argument("--track", choices=TRACKS, required=True)
    parser.add_argument("--seed", type=int, choices=EXPECTED_SEEDS, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    receipt = run(
        binding_path=args.binding,
        track=args.track,
        seed=args.seed,
        device=args.device,
        resume=args.resume,
    )
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "track": receipt["evaluation_kind"],
                "training_seed": receipt["training_seed"],
                "canonical_ledger": receipt["canonical_ledger"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
