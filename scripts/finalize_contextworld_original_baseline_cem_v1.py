#!/usr/bin/env python3
"""Fail-closed finalizer for the 4x2 original-environment CEM matrix.

The CEM matrix is deliberately descriptive: it reports one original-task CEM
result per canonical environment/family checkpoint and never creates a
cross-environment score.  This finalizer consumes already-written evaluator
receipts only.  It does not launch CEM, train, select a checkpoint, or alter a
component release decision.

The result namespace is reserved by an exclusive ``matrix_summary.json``
write.  Every required upstream receipt must already exist and pass its
identity/runtime/model-state closure before the summary is written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from contextworld.paths import repository_root, resolve_contextworld_path


ROOT = repository_root()
DEFAULT_PREREG = Path(
    "configs/benchmark/contextworld_original_baseline_cem_prereg_v1.yaml"
)
DEFAULT_RESULTS_ROOT = Path("artifacts/evaluation/original_baseline_cem_v1")
DEFAULT_CUBE_LEWM_AGGREGATE = Path(
    "/opt/huawei/explorer-env/dataset/ag_data/data/world_model/context_world/"
    "evaluation/history3/cube_gripper_carry_h3_development_v4r1/"
    "original_task_retention_v2/results/baseline_lewm/aggregate.json"
)
DEFAULT_CUBE_LEWM_FREEZE_RECEIPT = Path(
    "/opt/huawei/explorer-env/dataset/ag_data/data/world_model/context_world/"
    "evaluation/history3/cube_gripper_carry_h3_development_v4r1/"
    "original_task_retention_freeze_receipt_v2.json"
)

ENVIRONMENTS = ("tworoom", "pusht", "reacher", "cube")
FAMILIES = ("lewm", "pldm")
TWOROOM_SEEDS = (42, 43, 44, 45, 46, 47)
STANDARD_SEEDS = {
    "pusht": (42, 43, 44, 45, 46, 47),
    "reacher": (42, 43, 44),
    "cube": (42, 43, 44),
}
QUERIES_PER_SEED = {"tworoom": 50, "pusht": 50, "reacher": 100, "cube": 100}
EXPECTED_PROTOCOL = {
    "history_len": 3,
    "goal_offset_steps": 25,
    "eval_budget": 50,
    "horizon": 5,
    "receding_horizon": 5,
    "action_block": 5,
    "cem_samples": 300,
    "cem_iterations": 30,
    "cem_topk": 30,
}


class FinalizationError(RuntimeError):
    """Raised when an input is not a complete frozen CEM result."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(path: str | Path, *, root: Path) -> Path:
    value = Path(path).expanduser()
    if value.is_absolute():
        return value.resolve()
    return resolve_contextworld_path(value, repo_root=root)


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FinalizationError(f"{label} must be a mapping")
    return value


def _list(value: Any, *, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise FinalizationError(f"{label} must be a list")
    return value


def _require(value: Any, *, label: str) -> Any:
    if value is None:
        raise FinalizationError(f"{label} is missing")
    return value


def _require_sha256(value: Any, *, label: str) -> str:
    digest = str(value)
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise FinalizationError(f"{label} must be a lowercase SHA-256 digest")
    return digest


def _read_json(path: Path, *, label: str) -> tuple[Path, dict[str, Any]]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise FileNotFoundError(f"Missing required {label}: {resolved}")
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise FinalizationError(f"{label} is not valid JSON: {resolved}") from error
    return resolved, dict(_mapping(payload, label=label))


def _file_identity(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"Missing required {label}: {path}")
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _identity_from_mapping(value: Any, *, label: str) -> dict[str, Any]:
    row = _mapping(value, label=label)
    sha256 = _require_sha256(row.get("sha256"), label=f"{label}.sha256")
    try:
        size = int(row["size_bytes"])
    except (KeyError, TypeError, ValueError) as error:
        raise FinalizationError(f"{label}.size_bytes is invalid") from error
    if size < 0:
        raise FinalizationError(f"{label}.size_bytes must be non-negative")
    path = row.get("path")
    if not isinstance(path, str) or not path:
        raise FinalizationError(f"{label}.path is missing")
    return {"path": path, "sha256": sha256, "size_bytes": size}


def _assert_identity(
    actual: Any,
    expected: Mapping[str, Any],
    *,
    label: str,
    require_size: bool = True,
) -> dict[str, Any]:
    observed = _identity_from_mapping(actual, label=label)
    expected_identity = _identity_from_mapping(expected, label=f"expected {label}")
    if observed["sha256"] != expected_identity["sha256"]:
        raise FinalizationError(f"{label} SHA-256 drifted")
    if require_size and observed["size_bytes"] != expected_identity["size_bytes"]:
        raise FinalizationError(f"{label} size drifted")
    return observed


def _assert_sha_reference(
    actual: Any, expected: Mapping[str, Any], *, label: str
) -> dict[str, str]:
    """Check a report-level reference that intentionally omits file size.

    The TwoRoom core report predates the frozen wrapper and records its three
    input references as ``path`` + ``sha256`` only.  The wrapper's embedded
    preflight carries the full identity (and is checked separately); this
    helper makes sure the core report still points at that same content rather
    than weakening the wrapper-level identity check.
    """

    row = _mapping(actual, label=label)
    path = row.get("path")
    if not isinstance(path, str) or not path:
        raise FinalizationError(f"{label}.path is missing")
    observed = _require_sha256(row.get("sha256"), label=f"{label}.sha256")
    expected_identity = _identity_from_mapping(expected, label=f"expected {label}")
    if observed != expected_identity["sha256"]:
        raise FinalizationError(f"{label} SHA-256 drifted")
    return {"path": path, "sha256": observed}


def _checkpoint_identity_from_standard(model: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "path": str(_require(model.get("checkpoint"), label="model.checkpoint")),
        "sha256": _require_sha256(
            model.get("checkpoint_sha256"), label="model.checkpoint_sha256"
        ),
        "size_bytes": int(_require(model.get("checkpoint_size_bytes"), label="model.checkpoint_size_bytes")),
    }


def _assert_checkpoint(
    actual: Mapping[str, Any], expected: Mapping[str, Any], *, label: str
) -> dict[str, Any]:
    expected_identity = _identity_from_mapping(expected, label=f"expected {label}")
    observed = _identity_from_mapping(actual, label=label)
    if (
        observed["sha256"] != expected_identity["sha256"]
        or observed["size_bytes"] != expected_identity["size_bytes"]
    ):
        raise FinalizationError(f"{label} does not match the canonical checkpoint")
    return observed


def _assert_runtime(
    runtime: Any, *, expected_commit: str, label: str
) -> dict[str, Any]:
    row = _mapping(runtime, label=label)
    if str(row.get("commit", "")) != expected_commit:
        raise FinalizationError(f"{label} commit drifted")
    if row.get("clean") is not True:
        raise FinalizationError(f"{label} must record a clean checkout")
    root = row.get("root")
    if not isinstance(root, str) or not root:
        raise FinalizationError(f"{label}.root is missing")
    return {"root": root, "commit": expected_commit, "clean": True}


def _successes(value: Any, *, expected_count: int, label: str) -> list[bool]:
    rows = _list(value, label=label)
    if len(rows) != expected_count or any(not isinstance(row, bool) for row in rows):
        raise FinalizationError(f"{label} must contain {expected_count} booleans")
    return [bool(row) for row in rows]


def _validate_aggregate(
    aggregate: Any,
    *,
    successes: int,
    evaluations: int,
    label: str,
) -> dict[str, Any]:
    row = _mapping(aggregate, label=label)
    success_value = row.get("success_count", row.get("successes"))
    evaluation_value = row.get("evaluation_count", row.get("evaluations"))
    if int(_require(success_value, label=f"{label}.success_count")) != successes:
        raise FinalizationError(f"{label} success count drifted")
    if int(_require(evaluation_value, label=f"{label}.evaluation_count")) != evaluations:
        raise FinalizationError(f"{label} evaluation count drifted")
    rate = float(_require(row.get("success_rate"), label=f"{label}.success_rate"))
    if rate != successes / evaluations:
        raise FinalizationError(f"{label} success rate drifted")
    return {
        "success_count": successes,
        "evaluation_count": evaluations,
        "success_rate": rate,
    }


def _assert_protocol(
    protocol: Any,
    *,
    environment: str,
    seeds: Sequence[int],
    queries_per_seed: int,
    label: str,
) -> dict[str, Any]:
    row = _mapping(protocol, label=label)
    aliases = {"history_len": ("history_len", "history_size")}
    for field, expected in EXPECTED_PROTOCOL.items():
        names = aliases.get(field, (field,))
        value = next((row.get(name) for name in names if name in row), None)
        if value is None or int(value) != expected:
            raise FinalizationError(f"{label}.{field} drifted")
    reported_seeds = row.get("eval_seeds")
    if reported_seeds is not None and tuple(int(value) for value in reported_seeds) != tuple(seeds):
        raise FinalizationError(f"{label}.eval_seeds drifted")
    query_value = row.get("queries_per_seed", row.get("num_eval_per_seed"))
    if query_value is not None and int(query_value) != queries_per_seed:
        raise FinalizationError(f"{label}.queries_per_seed drifted")
    evaluation_value = row.get("evaluation_count", row.get("evaluations"))
    if evaluation_value is not None and int(evaluation_value) != len(seeds) * queries_per_seed:
        raise FinalizationError(f"{label}.evaluation_count drifted")
    return dict(row)


def _load_preregistration(path: Path, *, root: Path) -> dict[str, Any]:
    resolved = _resolve(path, root=root)
    if not resolved.is_file() or resolved.is_symlink():
        raise FileNotFoundError(f"Missing CEM preregistration: {resolved}")
    raw = resolved.read_text(encoding="utf-8")
    if "PENDING" in raw:
        raise FinalizationError("CEM preregistration contains pending identities")
    document = dict(_mapping(yaml.safe_load(raw), label="CEM preregistration"))
    if document.get("schema_version") != 1:
        raise FinalizationError("CEM preregistration schema_version drifted")
    if document.get("preregistration_id") != "contextworld_original_baseline_cem_v1":
        raise FinalizationError("unexpected CEM preregistration id")
    if document.get("status") != "frozen_before_cem_execution":
        raise FinalizationError("CEM preregistration is not frozen")
    scope = _mapping(document.get("scientific_scope"), label="scientific_scope")
    if (
        tuple(scope.get("environments", ())) != ENVIRONMENTS
        or tuple(scope.get("families", ())) != FAMILIES
        or int(scope.get("matrix_cells", -1)) != 8
        or int(scope.get("exact_legacy_cells_reused", -1)) != 1
        or int(scope.get("newly_executed_cells", -1)) != 7
        or int(scope.get("total_matrix_episodes", -1)) != 2400
        or int(scope.get("newly_executed_episodes", -1)) != 2100
        or scope.get("formal_suite_scoreboard_eligible") is not False
        or scope.get("cross_environment_average_authorized") is not False
        or scope.get("pass_fail_threshold") is not None
    ):
        raise FinalizationError("CEM matrix scientific scope drifted")
    authority = _mapping(document.get("authority"), label="authority")
    if (
        authority.get("cem_execution_authorized") is not True
        or int(authority.get("authorized_new_cells", -1)) != 7
        or int(authority.get("authorized_new_episodes", -1)) != 2100
    ):
        raise FinalizationError("CEM matrix execution authority/count drifted")
    for field in (
        "training_authorized",
        "finetuning_authorized",
        "checkpoint_selection_authorized",
        "model_or_recipe_change_authorized",
        "result_based_retry_authorized",
        "checkpoint_swap_authorized",
        "public_test_access_authorized",
        "formal_scoreboard_mutation_authorized",
        "component_release_claim_mutation_authorized",
    ):
        if authority.get(field) is not False:
            raise FinalizationError(f"authority.{field} must remain false")

    expected_reuse = {("cube", "lewm")}
    expected_execution = {
        (environment, family)
        for environment in ENVIRONMENTS
        for family in FAMILIES
    } - expected_reuse
    reuse_rows = _list(document.get("reuse_cells"), label="reuse_cells")
    execution_rows = _list(document.get("execution_cells"), label="execution_cells")
    reuse = {
        (
            str(_mapping(row, label="reuse cell").get("environment", "")),
            str(_mapping(row, label="reuse cell").get("family", "")),
        )
        for row in reuse_rows
    }
    execution = {
        (
            str(_mapping(row, label="execution cell").get("environment", "")),
            str(_mapping(row, label="execution cell").get("family", "")),
        )
        for row in execution_rows
    }
    if (
        len(reuse_rows) != 1
        or len(reuse) != 1
        or reuse != expected_reuse
        or len(execution_rows) != 7
        or len(execution) != 7
        or execution != expected_execution
        or reuse & execution
    ):
        raise FinalizationError("CEM reuse/execution cell contract drifted")
    reuse_row = _mapping(reuse_rows[0], label="Cube LeWM reuse cell")
    if reuse_row.get("checkpoint_id") != "cube_lewm_original" or reuse_row.get("exact_reuse_contract_passed") is not True:
        raise FinalizationError("Cube LeWM reuse contract is not frozen")
    protocol = _mapping(document.get("protocol"), label="protocol")
    if (
        int(protocol.get("history_tokens", -1)) != 3
        or int(protocol.get("action_block_raw_steps", -1)) != 5
        or int(protocol.get("goal_offset_raw_steps", -1)) != 25
        or int(protocol.get("execution_budget_raw_steps", -1)) != 50
        or int(protocol.get("horizon_action_blocks", -1)) != 5
        or int(protocol.get("receding_horizon_action_blocks", -1)) != 5
        or int(protocol.get("cem_candidates", -1)) != 300
        or int(protocol.get("cem_iterations", -1)) != 30
        or int(protocol.get("cem_topk", -1)) != 30
        or protocol.get("videos_written") is not False
    ):
        raise FinalizationError("CEM matrix protocol drifted")
    return {"path": resolved, "payload": document}


def _verify_declared_file(
    prereg: Mapping[str, Any], *, name: str, root: Path
) -> tuple[Path, dict[str, Any]]:
    expected = _identity_from_mapping(prereg.get(name), label=name)
    path = _resolve(expected["path"], root=root)
    observed = _file_identity(path, label=name)
    if (
        observed["sha256"] != expected["sha256"]
        or observed["size_bytes"] != expected["size_bytes"]
    ):
        raise FinalizationError(f"{name} identity drifted")
    return path, observed


def _verify_implementation_identities(
    prereg: Mapping[str, Any], *, root: Path
) -> list[dict[str, Any]]:
    entries = _mapping(prereg.get("implementation"), label="implementation")
    if not entries:
        raise FinalizationError("implementation identity set is empty")
    observed: list[dict[str, Any]] = []
    for name, identity in entries.items():
        expected = _identity_from_mapping(identity, label=f"implementation.{name}")
        path = _resolve(expected["path"], root=root)
        actual = _file_identity(path, label=f"implementation.{name}")
        if (
            actual["sha256"] != expected["sha256"]
            or actual["size_bytes"] != expected["size_bytes"]
        ):
            raise FinalizationError(f"implementation.{name} identity drifted")
        observed.append({"name": str(name), **actual})
    return observed


def _cube_lewm_reuse_identities(prereg: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows = _list(prereg.get("reuse_cells"), label="reuse_cells")
    if len(rows) != 1:
        raise FinalizationError("Cube LeWM reuse row is missing")
    row = _mapping(rows[0], label="Cube LeWM reuse cell")
    # The release preregistration names the historical freeze receipt
    # ``source`` and its consumed aggregate ``result``.  Accept the more
    # explicit ``source`` + ``freeze_receipt`` spelling as well so a future
    # recovery preregistration cannot accidentally invert the two artifacts.
    if "result" in row:
        aggregate_value = row.get("result")
        receipt_value = row.get("source")
    else:
        aggregate_value = row.get("source")
        receipt_value = row.get("freeze_receipt")
    return {
        "aggregate": _identity_from_mapping(
            aggregate_value, label="Cube LeWM reuse aggregate"
        ),
        "freeze_receipt": _identity_from_mapping(
            receipt_value, label="Cube LeWM reuse freeze receipt"
        ),
    }


def _assert_declared_source(
    path: Path, expected: Mapping[str, Any], *, label: str, root: Path
) -> dict[str, Any]:
    observed = _file_identity(path, label=label)
    expected_identity = _identity_from_mapping(expected, label=f"expected {label}")
    if (
        observed["sha256"] != expected_identity["sha256"]
        or observed["size_bytes"] != expected_identity["size_bytes"]
    ):
        raise FinalizationError(f"{label} identity drifted from preregistration")
    if path.resolve() != _resolve(expected_identity["path"], root=root):
        raise FinalizationError(f"{label} path differs from preregistration")
    return observed


def _load_checkpoint_registry(
    prereg: Mapping[str, Any], *, root: Path
) -> dict[str, dict[str, Any]]:
    registry_identity = _identity_from_mapping(
        prereg.get("base_checkpoint_registry"), label="base_checkpoint_registry"
    )
    registry_path = _resolve(registry_identity["path"], root=root)
    observed_registry = _file_identity(registry_path, label="base checkpoint registry")
    if (
        observed_registry["sha256"] != registry_identity["sha256"]
        or observed_registry["size_bytes"] != registry_identity["size_bytes"]
    ):
        raise FinalizationError("base checkpoint registry identity drifted")
    registry = dict(
        _mapping(
            yaml.safe_load(registry_path.read_text(encoding="utf-8")),
            label="base checkpoint registry",
        )
    )
    rows: dict[str, dict[str, Any]] = {}
    for raw in _list(registry.get("checkpoints"), label="registry.checkpoints"):
        row = _mapping(raw, label="registry checkpoint")
        checkpoint_id = str(_require(row.get("checkpoint_id"), label="checkpoint_id"))
        if checkpoint_id in rows:
            raise FinalizationError(f"duplicate canonical checkpoint id: {checkpoint_id}")
        rows[checkpoint_id] = {
            "environment": str(_require(row.get("environment"), label=f"{checkpoint_id}.environment")),
            "family": str(_require(row.get("family"), label=f"{checkpoint_id}.family")),
            "weights": _identity_from_mapping(
                row.get("weights"), label=f"{checkpoint_id}.weights"
            ),
        }
    expected_ids = {
        f"{environment}_{family}_original"
        for environment in ENVIRONMENTS
        for family in FAMILIES
    }
    if set(rows) != expected_ids:
        raise FinalizationError("canonical checkpoint registry does not close the 4x2 matrix")
    for checkpoint_id, row in rows.items():
        environment, family, _ = checkpoint_id.split("_", 2)
        if row["environment"] != environment or row["family"] != family:
            raise FinalizationError(f"checkpoint registry environment/family drifted: {checkpoint_id}")
    return rows


def _load_execution_preflight(
    prereg: Mapping[str, Any],
    *,
    registry: Mapping[str, Mapping[str, Any]],
    root: Path,
) -> dict[str, Any]:
    """Close the zero-episode strict-load preflight for all new CEM cells."""

    path, identity = _verify_declared_file(prereg, name="preflight", root=root)
    _, payload = _read_json(path, label="CEM execution preflight")
    if (
        payload.get("schema_version") != 1
        or payload.get("preflight_id") != "contextworld_original_baseline_cem_preflight_v1"
        or payload.get("status") != "passed_without_cem_execution"
        or payload.get("passed") is not True
        or int(payload.get("cem_episodes_consumed", -1)) != 0
        or payload.get("training_performed") is not False
        or payload.get("checkpoint_selection_performed") is not False
    ):
        raise FinalizationError("CEM execution preflight closure drifted")

    runtimes = _mapping(payload.get("runtime_checkouts"), label="preflight.runtime_checkouts")
    for name in ("tworoom", "pusht_reacher_cube"):
        expected = _mapping(
            _mapping(prereg.get("runtimes"), label="runtimes").get(name),
            label=f"runtimes.{name}",
        )
        _assert_runtime(
            runtimes.get(name),
            expected_commit=str(_require(expected.get("expected_commit"), label=f"runtimes.{name}.expected_commit")),
            label=f"preflight.runtime_checkouts.{name}",
        )

    rows = _list(payload.get("new_execution_models"), label="preflight.new_execution_models")
    expected_ids = {
        f"{environment}_{family}_original"
        for environment in ENVIRONMENTS
        for family in FAMILIES
    } - {"cube_lewm_original"}
    records: dict[str, Mapping[str, Any]] = {}
    for raw in rows:
        row = _mapping(raw, label="preflight model")
        checkpoint_id = str(_require(row.get("checkpoint_id"), label="preflight checkpoint_id"))
        if checkpoint_id in records:
            raise FinalizationError(f"duplicate preflight model: {checkpoint_id}")
        if row.get("strict_load") is not True:
            raise FinalizationError(f"preflight strict load failed: {checkpoint_id}")
        expected_checkpoint = registry.get(checkpoint_id)
        if expected_checkpoint is None:
            raise FinalizationError(f"unexpected preflight checkpoint: {checkpoint_id}")
        if _require_sha256(
            row.get("checkpoint_sha256"), label=f"preflight.{checkpoint_id}.checkpoint_sha256"
        ) != expected_checkpoint["weights"]["sha256"]:
            raise FinalizationError(f"preflight checkpoint identity drifted: {checkpoint_id}")
        _require_sha256(
            row.get("config_sha256"), label=f"preflight.{checkpoint_id}.config_sha256"
        )
        records[checkpoint_id] = row
    if len(rows) != len(expected_ids) or set(records) != expected_ids:
        raise FinalizationError("preflight model coverage does not close the seven new CEM cells")
    return {"path": path, "identity": identity, "models": records}


def _load_execution_cells(
    prereg: Mapping[str, Any],
    *,
    registry: Mapping[str, Mapping[str, Any]],
    root: Path,
    results_root: Path,
) -> dict[str, Mapping[str, Any]]:
    """Close all seven planned output locations and loader identities."""

    rows = _list(prereg.get("execution_cells"), label="execution_cells")
    expected_keys = {
        (environment, family)
        for environment in ENVIRONMENTS
        for family in FAMILIES
    } - {("cube", "lewm")}
    expected_seeds = {"tworoom": TWOROOM_SEEDS, **STANDARD_SEEDS}
    expected_mujoco_gl = {
        "tworoom": "egl",
        "pusht": "egl",
        "reacher": "osmesa",
        "cube": "osmesa",
    }
    cells: dict[str, Mapping[str, Any]] = {}
    keys: set[tuple[str, str]] = set()
    for raw in rows:
        row = _mapping(raw, label="execution cell")
        environment = str(_require(row.get("environment"), label="execution cell.environment"))
        family = str(_require(row.get("family"), label="execution cell.family"))
        checkpoint_id = str(_require(row.get("checkpoint_id"), label="execution cell.checkpoint_id"))
        key = (environment, family)
        if key not in expected_keys or checkpoint_id != f"{environment}_{family}_original":
            raise FinalizationError(f"unexpected CEM execution cell: {checkpoint_id}")
        if checkpoint_id in cells:
            raise FinalizationError(f"duplicate CEM execution cell: {checkpoint_id}")
        _assert_checkpoint(
            row.get("checkpoint"),
            registry[checkpoint_id]["weights"],
            label=f"execution cell {checkpoint_id} checkpoint",
        )
        _identity_from_mapping(
            row.get("effective_loader_config"),
            label=f"execution cell {checkpoint_id} effective_loader_config",
        )
        expected_runtime = "tworoom" if environment == "tworoom" else "pusht_reacher_cube"
        if row.get("runtime") != expected_runtime:
            raise FinalizationError(f"execution cell {checkpoint_id} runtime drifted")
        if row.get("mujoco_gl") != expected_mujoco_gl[environment]:
            raise FinalizationError(f"execution cell {checkpoint_id} MuJoCo backend drifted")
        if row.get("environment_inputs") != environment:
            raise FinalizationError(f"execution cell {checkpoint_id} input namespace drifted")
        try:
            seeds = tuple(int(seed) for seed in _list(row.get("eval_seeds"), label=f"execution cell {checkpoint_id}.eval_seeds"))
        except (TypeError, ValueError) as error:
            raise FinalizationError(f"execution cell {checkpoint_id} eval seed is invalid") from error
        if seeds != expected_seeds[environment]:
            raise FinalizationError(f"execution cell {checkpoint_id} seed schedule drifted")
        if int(row.get("queries_per_seed", -1)) != QUERIES_PER_SEED[environment]:
            raise FinalizationError(f"execution cell {checkpoint_id} query count drifted")
        if int(row.get("evaluations", -1)) != 300:
            raise FinalizationError(f"execution cell {checkpoint_id} evaluation count drifted")
        expected_kind = "six_seed_receipts" if environment == "tworoom" else "aggregate"
        if row.get("output_kind") != expected_kind:
            raise FinalizationError(f"execution cell {checkpoint_id} output kind drifted")
        output_dir = _resolve(
            _require(row.get("output_directory"), label=f"execution cell {checkpoint_id}.output_directory"),
            root=root,
        )
        if output_dir != results_root / environment / family:
            raise FinalizationError(f"execution cell {checkpoint_id} output directory drifted")
        output_files = tuple(
            str(item)
            for item in _list(row.get("output_files"), label=f"execution cell {checkpoint_id}.output_files")
        )
        expected_files = (
            tuple(f"seed{seed}.json" for seed in TWOROOM_SEEDS)
            if environment == "tworoom"
            else ("aggregate.json",)
        )
        if output_files != expected_files:
            raise FinalizationError(f"execution cell {checkpoint_id} output file set drifted")
        cells[checkpoint_id] = row
        keys.add(key)
    if len(rows) != 7 or len(cells) != 7 or keys != expected_keys:
        raise FinalizationError("execution cell metadata does not close the seven new CEM cells")
    return cells


def _assert_execution_config(
    execution_cells: Mapping[str, Mapping[str, Any]],
    *,
    checkpoint_id: str,
    actual: Mapping[str, Any],
    label: str,
) -> None:
    row = _mapping(execution_cells.get(checkpoint_id), label=f"execution cell {checkpoint_id}")
    _assert_identity(
        actual,
        _mapping(row.get("effective_loader_config"), label=f"execution cell {checkpoint_id}.effective_loader_config"),
        label=label,
    )


def _load_input_identity_audit(
    prereg: Mapping[str, Any], *, root: Path
) -> dict[str, Any]:
    """Verify the pre-freeze full-file dataset hash receipt once, not per GPU job."""

    path, identity = _verify_declared_file(prereg, name="input_identity_audit", root=root)
    _, payload = _read_json(path, label="CEM input identity audit")
    if (
        payload.get("schema_version") != 1
        or payload.get("audit_id") != "contextworld_original_baseline_cem_input_identity_audit_v1"
        or payload.get("content_hash_authority")
        != "full_file_sha256_streamed_before_cem_freeze"
    ):
        raise FinalizationError("CEM input identity audit closure drifted")
    datasets = _mapping(payload.get("datasets"), label="input_identity_audit.datasets")
    for environment in ENVIRONMENTS:
        expected = _expected_inputs(prereg, environment=environment)["dataset"]
        observed = _mapping(datasets.get(environment), label=f"input audit {environment}")
        _assert_identity(observed, expected, label=f"input audit {environment} dataset")
        if observed.get("content_hash_checked") is not True:
            raise FinalizationError(f"input audit {environment} dataset was not fully hashed")
    return {"path": path, "identity": identity}


def _assert_preflight_config(
    preflight: Mapping[str, Any],
    *,
    checkpoint_id: str,
    config_sha256: Any,
    label: str,
) -> None:
    rows = _mapping(preflight.get("models"), label="preflight.models")
    row = _mapping(rows.get(checkpoint_id), label=f"preflight.{checkpoint_id}")
    expected = _require_sha256(
        row.get("config_sha256"), label=f"preflight.{checkpoint_id}.config_sha256"
    )
    observed = _require_sha256(config_sha256, label=f"{label}.config_sha256")
    if observed != expected:
        raise FinalizationError(f"{label} config identity drifted from strict-load preflight")


def _assert_preflight_state_if_recorded(
    preflight: Mapping[str, Any],
    *,
    checkpoint_id: str,
    observed_state_sha256: str,
    label: str,
) -> None:
    """Bind an evaluator state hash to preflight when that receipt recorded one.

    The historical TwoRoom and Cube receipts only supplied strict-load and a
    parameter count, so the field remains optional for compatibility.  Newer
    standard runners record it and must agree exactly.
    """

    rows = _mapping(preflight.get("models"), label="preflight.models")
    row = _mapping(rows.get(checkpoint_id), label=f"preflight.{checkpoint_id}")
    expected = row.get("state_dict_sha256")
    if expected is None:
        return
    if _require_sha256(expected, label=f"preflight.{checkpoint_id}.state_dict_sha256") != observed_state_sha256:
        raise FinalizationError(f"{label} state identity drifted from strict-load preflight")


def _expected_runtime(prereg: Mapping[str, Any], *, environment: str) -> str:
    runtimes = _mapping(prereg.get("runtimes"), label="runtimes")
    name = "tworoom" if environment == "tworoom" else "pusht_reacher_cube"
    row = _mapping(runtimes.get(name), label=f"runtimes.{name}")
    if row.get("clean_checkout_required") is not True:
        raise FinalizationError(f"runtimes.{name} must require a clean checkout")
    return str(_require(row.get("expected_commit"), label=f"runtimes.{name}.expected_commit"))


def _expected_inputs(prereg: Mapping[str, Any], *, environment: str) -> Mapping[str, Any]:
    all_inputs = _mapping(
        prereg.get("frozen_environment_inputs"), label="frozen_environment_inputs"
    )
    return _mapping(all_inputs.get(environment), label=f"inputs.{environment}")


def _assert_state_pair(value: Any, *, label: str) -> str:
    row = _mapping(value, label=label)
    if row.get("passed") is not True:
        raise FinalizationError(f"{label}.passed must be true")
    before = _mapping(row.get("before"), label=f"{label}.before")
    after = _mapping(row.get("after"), label=f"{label}.after")
    before_sha = _require_sha256(before.get("state_dict_sha256"), label=f"{label}.before.state_dict_sha256")
    after_sha = _require_sha256(after.get("state_dict_sha256"), label=f"{label}.after.state_dict_sha256")
    if before_sha != after_sha:
        raise FinalizationError(f"{label} model state changed during CEM")
    if int(before.get("parameter_count", -1)) != int(after.get("parameter_count", -1)):
        raise FinalizationError(f"{label} parameter count changed during CEM")
    return before_sha


def _assert_tworoom_state_audit(value: Any, *, label: str) -> str:
    """Validate the legacy TwoRoom core's flat before/after audit fields."""

    row = _mapping(value, label=label)
    if row.get("passed") is not True:
        raise FinalizationError(f"{label}.passed must be true")
    before = _require_sha256(
        row.get("state_dict_sha256_before"), label=f"{label}.state_dict_sha256_before"
    )
    after = _require_sha256(
        row.get("state_dict_sha256_after"), label=f"{label}.state_dict_sha256_after"
    )
    if before != after:
        raise FinalizationError(f"{label} model state changed during CEM")
    return before


def _assert_legacy_dataset_protocol(
    protocol: Mapping[str, Any], expected: Mapping[str, Any], *, label: str
) -> None:
    """Validate a v1 aggregate that represented its dataset as flat fields."""

    dataset = protocol.get("dataset")
    if isinstance(dataset, Mapping):
        _assert_identity(dataset, expected, label=label)
        return
    if not isinstance(dataset, str) or not dataset:
        raise FinalizationError(f"{label}.dataset is missing")
    expected_identity = _identity_from_mapping(expected, label=f"expected {label}")
    if Path(dataset).expanduser().resolve() != Path(expected_identity["path"]).expanduser().resolve():
        raise FinalizationError(f"{label}.dataset path drifted")
    if _require_sha256(protocol.get("dataset_sha256"), label=f"{label}.dataset_sha256") != expected_identity["sha256"]:
        raise FinalizationError(f"{label}.dataset SHA-256 drifted")
    try:
        observed_size = int(protocol.get("dataset_size_bytes"))
    except (TypeError, ValueError) as error:
        raise FinalizationError(f"{label}.dataset_size_bytes is invalid") from error
    if observed_size != expected_identity["size_bytes"]:
        raise FinalizationError(f"{label}.dataset size drifted")


def _assert_public_test_closed(
    value: Any, *, label: str, fields: Sequence[str]
) -> None:
    row = _mapping(value, label=label)
    for field in fields:
        if row.get(field) is not False:
            raise FinalizationError(f"{label}.{field} must remain false")


def _validate_tworoom_cell(
    *,
    family: str,
    receipts_root: Path,
    prereg: Mapping[str, Any],
    registry: Mapping[str, Mapping[str, Any]],
    execution_preflight: Mapping[str, Any],
    execution_cells: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    expected_inputs = _expected_inputs(prereg, environment="tworoom")
    expected_checkpoint = registry[f"tworoom_{family}_original"]["weights"]
    expected_commit = _expected_runtime(prereg, environment="tworoom")
    receipt_identities: list[dict[str, Any]] = []
    state_digests: set[str] = set()
    successes = 0

    for seed in TWOROOM_SEEDS:
        path, report = _read_json(
            receipts_root / f"seed{seed}.json",
            label=f"TwoRoom {family} seed {seed} receipt",
        )
        if report.get("status") != "passed":
            raise FinalizationError(f"TwoRoom {family} seed {seed} did not complete")
        protocol = _mapping(report.get("protocol"), label=f"TwoRoom {family} seed {seed}.protocol")
        if (
            int(protocol.get("eval_seed", -1)) != seed
            or int(protocol.get("evaluations", -1)) != 50
            or int(protocol.get("history_size", -1)) != 3
            or int(protocol.get("action_block", -1)) != 5
            or int(protocol.get("eval_budget", -1)) != 50
            or int(protocol.get("horizon", -1)) != 5
            or int(protocol.get("receding_horizon", -1)) != 5
            or int(protocol.get("cem_samples", -1)) != 300
            or int(protocol.get("cem_steps", -1)) != 30
            or int(protocol.get("cem_topk", -1)) != 30
        ):
            raise FinalizationError(f"TwoRoom {family} seed {seed} protocol drifted")

        preflight = _mapping(
            report.get("frozen_input_preflight"),
            label=f"TwoRoom {family} seed {seed}.frozen_input_preflight",
        )
        if preflight.get("passed") is not True:
            raise FinalizationError(f"TwoRoom {family} seed {seed} input preflight failed")
        _assert_runtime(
            preflight.get("runtime"),
            expected_commit=expected_commit,
            label=f"TwoRoom {family} seed {seed}.runtime",
        )
        _assert_checkpoint(
            preflight.get("checkpoint"), expected_checkpoint, label=f"TwoRoom {family} checkpoint"
        )
        _assert_identity(
            preflight.get("catalog"), expected_inputs["catalog"], label=f"TwoRoom {family} catalog"
        )
        _assert_identity(
            preflight.get("normalizer"), expected_inputs["normalizer"], label=f"TwoRoom {family} normalizer"
        )
        _assert_identity(
            preflight.get("source_dataset"), expected_inputs["dataset"], label=f"TwoRoom {family} dataset"
        )
        config = _identity_from_mapping(
            preflight.get("config"), label=f"TwoRoom {family} checkpoint config"
        )
        if not config["path"]:
            raise FinalizationError(f"TwoRoom {family} checkpoint config is missing")
        _assert_preflight_config(
            execution_preflight,
            checkpoint_id=f"tworoom_{family}_original",
            config_sha256=config["sha256"],
            label=f"TwoRoom {family} checkpoint",
        )
        _assert_execution_config(
            execution_cells,
            checkpoint_id=f"tworoom_{family}_original",
            actual=config,
            label=f"TwoRoom {family} checkpoint config",
        )
        _assert_sha_reference(
            report.get("checkpoint"), expected_checkpoint, label=f"TwoRoom {family} report checkpoint"
        )
        _assert_sha_reference(
            report.get("catalog"), expected_inputs["catalog"], label=f"TwoRoom {family} report catalog"
        )
        _assert_sha_reference(
            report.get("normalizer"), expected_inputs["normalizer"], label=f"TwoRoom {family} report normalizer"
        )
        stable = _mapping(report.get("stable_worldmodel"), label=f"TwoRoom {family} stable_worldmodel")
        if str(stable.get("commit", "")) != expected_commit:
            raise FinalizationError(f"TwoRoom {family} seed {seed} report runtime drifted")
        state_digests.add(
            _assert_tworoom_state_audit(
                report.get("frozen_weight_audit"),
                label=f"TwoRoom {family} seed {seed}.frozen_weight_audit",
            )
        )
        raw_records = _list(report.get("raw_records"), label=f"TwoRoom {family} seed {seed}.raw_records")
        if len(raw_records) != 50:
            raise FinalizationError(f"TwoRoom {family} seed {seed} raw record count drifted")
        record_successes: list[bool] = []
        indices: list[int] = []
        for raw in raw_records:
            record = _mapping(raw, label=f"TwoRoom {family} raw record")
            if int(record.get("eval_seed", -1)) != seed:
                raise FinalizationError(f"TwoRoom {family} seed assignment drifted")
            indices.append(int(record.get("evaluation_index", -1)))
            if not isinstance(record.get("success"), bool):
                raise FinalizationError(f"TwoRoom {family} raw success is invalid")
            record_successes.append(bool(record["success"]))
        if sorted(indices) != list(range(50)):
            raise FinalizationError(f"TwoRoom {family} seed {seed} query index coverage drifted")
        aggregate = _validate_aggregate(
            report.get("aggregate"),
            successes=sum(record_successes),
            evaluations=50,
            label=f"TwoRoom {family} seed {seed}.aggregate",
        )
        successes += aggregate["success_count"]
        receipt_identities.append(_file_identity(path, label=f"TwoRoom {family} seed {seed} receipt"))

    if len(state_digests) != 1:
        raise FinalizationError(f"TwoRoom {family} loaded model state differs across seeds")
    loaded_state = next(iter(state_digests))
    _assert_preflight_state_if_recorded(
        execution_preflight,
        checkpoint_id=f"tworoom_{family}_original",
        observed_state_sha256=loaded_state,
        label=f"TwoRoom {family}",
    )
    return {
        "environment": "tworoom",
        "family": family,
        "checkpoint_id": f"tworoom_{family}_original",
        "provenance": "six_seed_receipts",
        "success_count": successes,
        "evaluation_count": 300,
        "success_rate": successes / 300,
        "model_state_audit": {
            "passed": True,
            "loaded_state_dict_sha256": loaded_state,
            "scope": "actual_model_per_seed",
        },
        "sources": receipt_identities,
    }


def _validate_standard_cell(
    *,
    environment: str,
    family: str,
    aggregate_path: Path,
    prereg: Mapping[str, Any],
    registry: Mapping[str, Mapping[str, Any]],
    execution_preflight: Mapping[str, Any],
    execution_cells: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    path, report = _read_json(aggregate_path, label=f"{environment} {family} aggregate")
    if report.get("status") != "standard_original_task_real_environment_cem" or report.get("task") != environment:
        raise FinalizationError(f"{environment} {family} aggregate kind drifted")
    expected_inputs = _expected_inputs(prereg, environment=environment)
    expected_checkpoint = registry[f"{environment}_{family}_original"]["weights"]
    expected_commit = _expected_runtime(prereg, environment=environment)
    _assert_runtime(report.get("runtime"), expected_commit=expected_commit, label=f"{environment} {family}.runtime")
    seeds = STANDARD_SEEDS[environment]
    queries = QUERIES_PER_SEED[environment]
    protocol = _assert_protocol(
        report.get("protocol"),
        environment=environment,
        seeds=seeds,
        queries_per_seed=queries,
        label=f"{environment} {family}.protocol",
    )
    _assert_identity(protocol.get("dataset"), expected_inputs["dataset"], label=f"{environment} {family} dataset")
    query_catalog = _mapping(report.get("query_catalog"), label=f"{environment} {family}.query_catalog")
    _assert_identity(
        query_catalog.get("frozen_source"), expected_inputs["catalog"], label=f"{environment} {family} catalog"
    )
    _assert_public_test_closed(
        report.get("public_test"),
        label=f"{environment} {family}.public_test",
        fields=("contextworld_public_test_read", "contextworld_public_test_scored"),
    )
    model = _mapping(report.get("model"), label=f"{environment} {family}.model")
    _assert_checkpoint(
        _checkpoint_identity_from_standard(model), expected_checkpoint, label=f"{environment} {family} checkpoint"
    )
    loader_config = _identity_from_mapping(
        {
            "path": model.get("config"),
            "sha256": model.get("config_sha256"),
            "size_bytes": model.get("config_size_bytes"),
        },
        label=f"{environment} {family} loader config",
    )
    _assert_preflight_config(
        execution_preflight,
        checkpoint_id=f"{environment}_{family}_original",
        config_sha256=loader_config["sha256"],
        label=f"{environment} {family} loader",
    )
    _assert_execution_config(
        execution_cells,
        checkpoint_id=f"{environment}_{family}_original",
        actual=loader_config,
        label=f"{environment} {family} loader config",
    )
    state_audit = _mapping(model.get("frozen_state_audit"), label=f"{environment} {family}.frozen_state_audit")
    if state_audit.get("passed") is not True or state_audit.get("scope") != "actual_policy_model_per_seed":
        raise FinalizationError(f"{environment} {family} model-state audit did not pass")
    by_seed = _list(state_audit.get("seeds"), label=f"{environment} {family}.state_audit.seeds")
    state_rows = {int(_mapping(row, label="state audit row").get("eval_seed", -1)): row for row in by_seed}
    if len(by_seed) != len(seeds) or len(state_rows) != len(seeds) or set(state_rows) != set(seeds):
        raise FinalizationError(f"{environment} {family} state-audit seed coverage drifted")
    model_seeds = _list(model.get("seeds"), label=f"{environment} {family}.model.seeds")
    rows = {int(_mapping(row, label="model seed row").get("eval_seed", -1)): row for row in model_seeds}
    if len(model_seeds) != len(seeds) or len(rows) != len(seeds) or set(rows) != set(seeds):
        raise FinalizationError(f"{environment} {family} result seed coverage drifted")
    all_successes: list[bool] = []
    state_digests: set[str] = set()
    for seed in seeds:
        row = _mapping(rows[seed], label=f"{environment} {family} seed {seed}")
        outcomes = _successes(row.get("episode_successes"), expected_count=queries, label=f"{environment} {family} seed {seed}.episode_successes")
        _validate_aggregate(
            row,
            successes=sum(outcomes),
            evaluations=queries,
            label=f"{environment} {family} seed {seed}",
        )
        audit = _mapping(state_rows[seed], label=f"{environment} {family} state audit {seed}")
        state_digests.add(_assert_state_pair(audit, label=f"{environment} {family} state audit {seed}"))
        row_audit = _mapping(row.get("frozen_state_audit"), label=f"{environment} {family} seed {seed}.frozen_state_audit")
        state_digests.add(_assert_state_pair(row_audit, label=f"{environment} {family} seed {seed}.frozen_state_audit"))
        all_successes.extend(outcomes)
    if len(state_digests) != 1:
        raise FinalizationError(f"{environment} {family} loaded model state differs across seeds")
    loaded_state = next(iter(state_digests))
    _assert_preflight_state_if_recorded(
        execution_preflight,
        checkpoint_id=f"{environment}_{family}_original",
        observed_state_sha256=loaded_state,
        label=f"{environment} {family}",
    )
    aggregate = _validate_aggregate(
        model.get("aggregate"),
        successes=sum(all_successes),
        evaluations=300,
        label=f"{environment} {family}.aggregate",
    )
    return {
        "environment": environment,
        "family": family,
        "checkpoint_id": f"{environment}_{family}_original",
        "provenance": "aggregate",
        **aggregate,
        "model_state_audit": {
            "passed": True,
            "loaded_state_dict_sha256": loaded_state,
            "scope": "actual_policy_model_per_seed",
        },
        "sources": [_file_identity(path, label=f"{environment} {family} aggregate")],
    }


def _validate_cube_pldm(
    *,
    aggregate_path: Path,
    prereg: Mapping[str, Any],
    registry: Mapping[str, Mapping[str, Any]],
    execution_preflight: Mapping[str, Any],
    execution_cells: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    path, report = _read_json(aggregate_path, label="Cube PLDM aggregate")
    if report.get("status") != "standard_original_task_real_environment_cem" or report.get("task") != "cube":
        raise FinalizationError("Cube PLDM aggregate kind drifted")
    expected_inputs = _expected_inputs(prereg, environment="cube")
    expected_checkpoint = registry["cube_pldm_original"]["weights"]
    expected_commit = _expected_runtime(prereg, environment="cube")
    _assert_runtime(report.get("runtime"), expected_commit=expected_commit, label="Cube PLDM.runtime")
    protocol = _assert_protocol(
        report.get("protocol"),
        environment="cube",
        seeds=STANDARD_SEEDS["cube"],
        queries_per_seed=100,
        label="Cube PLDM.protocol",
    )
    _assert_identity(protocol.get("dataset"), expected_inputs["dataset"], label="Cube PLDM dataset")
    query_catalog = _mapping(report.get("query_catalog"), label="Cube PLDM.query_catalog")
    _assert_identity(query_catalog.get("frozen_source"), expected_inputs["catalog"], label="Cube PLDM catalog")
    _assert_public_test_closed(
        report.get("public_test"),
        label="Cube PLDM.public_test",
        fields=("opened", "read", "hashed", "scored"),
    )
    model = _mapping(report.get("model"), label="Cube PLDM.model")
    if model.get("family") != "pldm" or model.get("strict_load") is not True:
        raise FinalizationError("Cube PLDM strict family/load audit drifted")
    _assert_checkpoint(model.get("checkpoint"), expected_checkpoint, label="Cube PLDM checkpoint")
    loader_config = _identity_from_mapping(model.get("config"), label="Cube PLDM loader config")
    _assert_preflight_config(
        execution_preflight,
        checkpoint_id="cube_pldm_original",
        config_sha256=loader_config["sha256"],
        label="Cube PLDM loader",
    )
    _assert_execution_config(
        execution_cells,
        checkpoint_id="cube_pldm_original",
        actual=loader_config,
        label="Cube PLDM loader config",
    )
    if model.get("loaded_state_consistent_across_seeds") is not True:
        raise FinalizationError("Cube PLDM loaded-state consistency audit failed")
    loaded_state = _require_sha256(
        model.get("loaded_state_dict_sha256"), label="Cube PLDM.loaded_state_dict_sha256"
    )
    model_seeds = _list(model.get("seeds"), label="Cube PLDM.model.seeds")
    rows = {
        int(_mapping(row, label="Cube PLDM seed row").get("eval_seed", -1)): row
        for row in model_seeds
    }
    if (
        len(model_seeds) != len(STANDARD_SEEDS["cube"])
        or len(rows) != len(STANDARD_SEEDS["cube"])
        or set(rows) != set(STANDARD_SEEDS["cube"])
    ):
        raise FinalizationError("Cube PLDM result seed coverage drifted")
    all_successes: list[bool] = []
    for seed in STANDARD_SEEDS["cube"]:
        row = _mapping(rows[seed], label=f"Cube PLDM seed {seed}")
        outcomes = _successes(row.get("episode_successes"), expected_count=100, label=f"Cube PLDM seed {seed}.episode_successes")
        _validate_aggregate(row, successes=sum(outcomes), evaluations=100, label=f"Cube PLDM seed {seed}")
        observed_state = _assert_state_pair(
            row.get("actual_evaluated_model_state"),
            label=f"Cube PLDM seed {seed}.actual_evaluated_model_state",
        )
        if observed_state != loaded_state:
            raise FinalizationError(f"Cube PLDM seed {seed} model state drifted")
        all_successes.extend(outcomes)
    aggregate = _validate_aggregate(
        model.get("aggregate"), successes=sum(all_successes), evaluations=300, label="Cube PLDM.aggregate"
    )
    _assert_preflight_state_if_recorded(
        execution_preflight,
        checkpoint_id="cube_pldm_original",
        observed_state_sha256=loaded_state,
        label="Cube PLDM",
    )
    return {
        "environment": "cube",
        "family": "pldm",
        "checkpoint_id": "cube_pldm_original",
        "provenance": "aggregate",
        **aggregate,
        "model_state_audit": {
            "passed": True,
            "loaded_state_dict_sha256": loaded_state,
            "scope": "actual_model_per_seed",
        },
        "sources": [_file_identity(path, label="Cube PLDM aggregate")],
    }


def _validate_cube_lewm_reuse(
    *,
    aggregate_path: Path,
    freeze_receipt_path: Path,
    prereg: Mapping[str, Any],
    registry: Mapping[str, Mapping[str, Any]],
    reuse_identities: Mapping[str, Mapping[str, Any]],
    root: Path,
) -> dict[str, Any]:
    aggregate_path, aggregate_report = _read_json(aggregate_path, label="Cube LeWM aggregate")
    freeze_receipt_path, freeze = _read_json(freeze_receipt_path, label="Cube LeWM freeze receipt")
    _assert_declared_source(
        aggregate_path,
        reuse_identities["aggregate"],
        label="Cube LeWM aggregate",
        root=root,
    )
    _assert_declared_source(
        freeze_receipt_path,
        reuse_identities["freeze_receipt"],
        label="Cube LeWM freeze receipt",
        root=root,
    )
    if freeze.get("status") != "frozen_authorized":
        raise FinalizationError("Cube LeWM freeze receipt is not frozen_authorized")
    if aggregate_report.get("status") != "standard_original_task_real_environment_cem" or aggregate_report.get("task") != "cube":
        raise FinalizationError("Cube LeWM aggregate kind drifted")
    expected_inputs = _expected_inputs(prereg, environment="cube")
    expected_checkpoint = registry["cube_lewm_original"]["weights"]
    expected_commit = _expected_runtime(prereg, environment="cube")
    _assert_runtime(
        _mapping(freeze.get("model_preflight"), label="Cube LeWM.model_preflight").get("runtime"),
        expected_commit=expected_commit,
        label="Cube LeWM freeze runtime",
    )
    static = _mapping(freeze.get("static_identities"), label="Cube LeWM.static_identities")
    _assert_identity(static.get("dataset"), expected_inputs["dataset"], label="Cube LeWM freeze dataset")
    _assert_identity(freeze.get("query_catalog"), expected_inputs["catalog"], label="Cube LeWM freeze catalog")
    preflight_models = _list(
        _mapping(freeze.get("model_preflight"), label="Cube LeWM.model_preflight").get("models"),
        label="Cube LeWM.model_preflight.models",
    )
    baselines = [
        _mapping(row, label="Cube LeWM preflight model")
        for row in preflight_models
        if _mapping(row, label="Cube LeWM preflight model").get("model")
        == "baseline_lewm"
    ]
    if len(baselines) != 1 or baselines[0].get("strict_load") is not True:
        raise FinalizationError("Cube LeWM frozen baseline strict-load receipt is missing")
    baseline = baselines[0]
    _assert_checkpoint(
        {
            "path": baseline.get("checkpoint"),
            "sha256": baseline.get("checkpoint_sha256"),
            "size_bytes": expected_checkpoint["size_bytes"],
        },
        expected_checkpoint,
        label="Cube LeWM freeze checkpoint",
    )
    _assert_runtime(aggregate_report.get("runtime"), expected_commit=expected_commit, label="Cube LeWM aggregate runtime")
    protocol = _assert_protocol(
        aggregate_report.get("protocol"),
        environment="cube",
        seeds=STANDARD_SEEDS["cube"],
        queries_per_seed=100,
        label="Cube LeWM.protocol",
    )
    _assert_legacy_dataset_protocol(
        protocol, expected_inputs["dataset"], label="Cube LeWM aggregate dataset"
    )
    query_catalog = _mapping(aggregate_report.get("query_catalog"), label="Cube LeWM.query_catalog")
    expected_catalog = _identity_from_mapping(
        expected_inputs["catalog"], label="expected Cube catalog"
    )
    frozen_source = query_catalog.get("frozen_source")
    if (
        not isinstance(frozen_source, str)
        or Path(frozen_source).expanduser().resolve()
        != Path(expected_catalog["path"]).expanduser().resolve()
    ):
        raise FinalizationError("Cube LeWM aggregate catalog source drifted")
    if _require_sha256(query_catalog.get("sha256"), label="Cube LeWM query_catalog.sha256") != expected_catalog["sha256"]:
        raise FinalizationError("Cube LeWM aggregate catalog drifted")
    models = _list(aggregate_report.get("models"), label="Cube LeWM.models")
    if len(models) != 1:
        raise FinalizationError("Cube LeWM aggregate must contain exactly one model")
    model = _mapping(models[0], label="Cube LeWM.model")
    if model.get("model") != "baseline_lewm":
        raise FinalizationError("Cube LeWM aggregate model drifted")
    _assert_checkpoint(
        {
            "path": model.get("checkpoint"),
            "sha256": model.get("checkpoint_sha256"),
            "size_bytes": expected_checkpoint["size_bytes"],
        },
        expected_checkpoint,
        label="Cube LeWM aggregate checkpoint",
    )
    if _require_sha256(model.get("config_sha256"), label="Cube LeWM aggregate config") != _require_sha256(
        baseline.get("config_sha256"), label="Cube LeWM freeze config"
    ):
        raise FinalizationError("Cube LeWM aggregate/config preflight identity drifted")
    model_seeds = _list(model.get("seeds"), label="Cube LeWM.model.seeds")
    rows = {
        int(_mapping(row, label="Cube LeWM seed row").get("eval_seed", -1)): row
        for row in model_seeds
    }
    if (
        len(model_seeds) != len(STANDARD_SEEDS["cube"])
        or len(rows) != len(STANDARD_SEEDS["cube"])
        or set(rows) != set(STANDARD_SEEDS["cube"])
    ):
        raise FinalizationError("Cube LeWM result seed coverage drifted")
    outcomes: list[bool] = []
    for seed in STANDARD_SEEDS["cube"]:
        row = _mapping(rows[seed], label=f"Cube LeWM seed {seed}")
        values = _successes(row.get("episode_successes"), expected_count=100, label=f"Cube LeWM seed {seed}.episode_successes")
        _validate_aggregate(row, successes=sum(values), evaluations=100, label=f"Cube LeWM seed {seed}")
        outcomes.extend(values)
    aggregate = _validate_aggregate(model.get("aggregate"), successes=sum(outcomes), evaluations=300, label="Cube LeWM.aggregate")
    return {
        "environment": "cube",
        "family": "lewm",
        "checkpoint_id": "cube_lewm_original",
        "provenance": "frozen_reuse_receipt_and_aggregate",
        **aggregate,
        "model_state_audit": {
            "passed": True,
            "scope": "frozen_strict_load_receipt",
            "parameter_count": int(_require(baseline.get("parameter_count"), label="Cube LeWM parameter_count")),
        },
        "sources": [
            _file_identity(freeze_receipt_path, label="Cube LeWM freeze receipt"),
            _file_identity(aggregate_path, label="Cube LeWM aggregate"),
        ],
    }


def _write_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite CEM matrix summary: {path}")
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(encoded)
        stream.flush()


def finalize(
    *,
    prereg_path: Path = DEFAULT_PREREG,
    results_root: Path = DEFAULT_RESULTS_ROOT,
    cube_lewm_aggregate: Path = DEFAULT_CUBE_LEWM_AGGREGATE,
    cube_lewm_freeze_receipt: Path = DEFAULT_CUBE_LEWM_FREEZE_RECEIPT,
    output: Path | None = None,
    repo_root: Path = ROOT,
) -> dict[str, Any]:
    """Validate all eight CEM cells and exclusively write their summary."""

    root = repo_root.expanduser().resolve()
    preregistration = _load_preregistration(prereg_path, root=root)
    prereg = preregistration["payload"]
    registry = _load_checkpoint_registry(prereg, root=root)
    _, base_icl_result_freeze = _verify_declared_file(
        prereg, name="base_icl_result_freeze", root=root
    )
    input_identity_audit = _load_input_identity_audit(prereg, root=root)
    implementation = _verify_implementation_identities(prereg, root=root)
    execution_preflight = _load_execution_preflight(
        prereg, registry=registry, root=root
    )
    cube_lewm_reuse = _cube_lewm_reuse_identities(prereg)
    declared_output = _resolve(
        _mapping(prereg.get("execution_policy"), label="execution_policy").get("result_summary"),
        root=root,
    )
    declared_results_root = _resolve(
        _mapping(prereg.get("execution_policy"), label="execution_policy").get("output_root"),
        root=root,
    )
    target = declared_output if output is None else output.expanduser().resolve()
    if target != declared_output:
        raise FinalizationError("matrix_summary output path differs from preregistration")
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Refusing to overwrite CEM matrix summary: {target}")
    root_results = _resolve(results_root, root=root)
    if root_results != declared_results_root:
        raise FinalizationError("CEM results root differs from preregistration")
    execution_cells = _load_execution_cells(
        prereg,
        registry=registry,
        root=root,
        results_root=declared_results_root,
    )
    tw_lewm = _validate_tworoom_cell(
        family="lewm",
        receipts_root=root_results / "tworoom" / "lewm",
        prereg=prereg,
        registry=registry,
        execution_preflight=execution_preflight,
        execution_cells=execution_cells,
    )
    tw_pldm = _validate_tworoom_cell(
        family="pldm",
        receipts_root=root_results / "tworoom" / "pldm",
        prereg=prereg,
        registry=registry,
        execution_preflight=execution_preflight,
        execution_cells=execution_cells,
    )
    cells = [
        tw_lewm,
        tw_pldm,
        _validate_standard_cell(
            environment="pusht",
            family="lewm",
            aggregate_path=root_results / "pusht" / "lewm" / "aggregate.json",
            prereg=prereg,
            registry=registry,
            execution_preflight=execution_preflight,
            execution_cells=execution_cells,
        ),
        _validate_standard_cell(
            environment="pusht",
            family="pldm",
            aggregate_path=root_results / "pusht" / "pldm" / "aggregate.json",
            prereg=prereg,
            registry=registry,
            execution_preflight=execution_preflight,
            execution_cells=execution_cells,
        ),
        _validate_standard_cell(
            environment="reacher",
            family="lewm",
            aggregate_path=root_results / "reacher" / "lewm" / "aggregate.json",
            prereg=prereg,
            registry=registry,
            execution_preflight=execution_preflight,
            execution_cells=execution_cells,
        ),
        _validate_standard_cell(
            environment="reacher",
            family="pldm",
            aggregate_path=root_results / "reacher" / "pldm" / "aggregate.json",
            prereg=prereg,
            registry=registry,
            execution_preflight=execution_preflight,
            execution_cells=execution_cells,
        ),
        _validate_cube_lewm_reuse(
            aggregate_path=_resolve(cube_lewm_aggregate, root=root),
            freeze_receipt_path=_resolve(cube_lewm_freeze_receipt, root=root),
            prereg=prereg,
            registry=registry,
            reuse_identities=cube_lewm_reuse,
            root=root,
        ),
        _validate_cube_pldm(
            aggregate_path=root_results / "cube" / "pldm" / "aggregate.json",
            prereg=prereg,
            registry=registry,
            execution_preflight=execution_preflight,
            execution_cells=execution_cells,
        ),
    ]
    keys = [(row["environment"], row["family"]) for row in cells]
    expected_keys = [(environment, family) for environment in ENVIRONMENTS for family in FAMILIES]
    if sorted(keys) != sorted(expected_keys) or len(keys) != len(set(keys)):
        raise FinalizationError("CEM result matrix is incomplete or duplicated")
    if any(int(row["evaluation_count"]) != 300 for row in cells):
        raise FinalizationError("every CEM matrix cell must contain exactly 300 episodes")
    total = sum(int(row["evaluation_count"]) for row in cells)
    if total != 2400:
        raise FinalizationError("CEM matrix must contain exactly 2400 episodes")
    if any(not row["model_state_audit"]["passed"] for row in cells):
        raise FinalizationError("one or more CEM model-state audits failed")
    summary = {
        "schema_version": 1,
        "summary_id": "contextworld_original_baseline_cem_matrix_v1",
        "status": "completed_descriptive_original_environment_cem_matrix",
        "preregistration": _file_identity(preregistration["path"], label="CEM preregistration"),
        "pre_execution_closure": {
            "base_icl_result_freeze": base_icl_result_freeze,
            "input_identity_audit": input_identity_audit["identity"],
            "execution_preflight": execution_preflight["identity"],
            "implementation": implementation,
            "finalizer": _file_identity(Path(__file__).resolve(), label="CEM finalizer"),
        },
        "scope": {
            "result_kind": "post_release_descriptive_original_environment_baseline",
            "formal_suite_scoreboard_eligible": False,
            "cross_environment_average_authorized": False,
            "cross_environment_average_reported": False,
            "pass_fail_threshold": None,
            "public_test_accessed": False,
        },
        "counts": {
            "matrix_cells": 8,
            "episodes_per_cell": 300,
            "total_matrix_episodes": 2400,
        },
        "cells": cells,
        "interpretation": {
            "allowed": "per-environment original-task CEM descriptive context",
            "prohibited": [
                "cross_environment_average",
                "formal_suite_scoreboard_mutation",
                "component_specific_hidden_rule_inference_claim",
            ],
        },
    }
    _write_exclusive(target, summary)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--prereg", type=Path, default=DEFAULT_PREREG)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--cube-lewm-aggregate", type=Path, default=DEFAULT_CUBE_LEWM_AGGREGATE)
    parser.add_argument(
        "--cube-lewm-freeze-receipt", type=Path, default=DEFAULT_CUBE_LEWM_FREEZE_RECEIPT
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = finalize(
        prereg_path=args.prereg,
        results_root=args.results_root,
        cube_lewm_aggregate=args.cube_lewm_aggregate,
        cube_lewm_freeze_receipt=args.cube_lewm_freeze_receipt,
        output=args.output,
        repo_root=args.repo_root,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "counts": summary["counts"],
                "output": str(
                    _resolve(
                        _mapping(
                            _load_preregistration(args.prereg, root=args.repo_root.expanduser().resolve())["payload"].get("execution_policy"),
                            label="execution_policy",
                        ).get("result_summary"),
                        root=args.repo_root.expanduser().resolve(),
                    )
                ),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
