#!/usr/bin/env python3
"""Fail-closed finalizer for the original-baseline three-seed family matrix.

Consumes the receipts written by the frozen seed-completion launcher
(``run_contextworld_original_baseline_seed_completion_v1.py``) plus the eight
already-evaluated members carried from ``contextworld_original_baseline_cem_v1``
and writes the preregistered ``family_summary.json``: every original-recipe
baseline family (four environments times LeWM/PLDM) as a homogeneous
three-training-seed statistic (mean / sample std / sample variance across
seeds 3072/3073/3074).

Descriptive only: no pass/fail threshold, no cross-environment average, no
formal-scoreboard mutation.  Every required receipt must exist and pass its
identity/protocol/model-state closure before the summary is exclusively
written; a missing or partial cell (including the eval-seed-43 infrastructure
relaunch receipt for tworoom/pldm/3074) fails the finalization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from contextworld.paths import repository_root, resolve_contextworld_path


ROOT = repository_root()
DEFAULT_PREREG = Path(
    "configs/benchmark/contextworld_original_baseline_seed_completion_prereg_v1.yaml"
)
PREREG_ID = "contextworld_original_baseline_seed_completion_v1"
PARENT_RESULTS_FREEZE = Path(
    "configs/benchmark/contextworld_original_baseline_cem_results_freeze_v1.json"
)
EVAL43_RECOVERY_PREREG = Path(
    "configs/benchmark/"
    "original_baseline_seed_completion_tworoom_pldm_seed3074_eval43_"
    "infra_relaunch_recovery_v1.yaml"
)
EVAL43_RECOVERY_ID = (
    "original_baseline_seed_completion_tworoom_pldm_seed3074_eval43_"
    "infra_relaunch_recovery_v1"
)
EVAL43_RELAUNCH_RECEIPT = Path(
    "artifacts/evaluation/original_baseline_seed_completion_v1/tworoom/pldm/"
    "seed3074/eval43_infra_relaunch_recovery_v1.json"
)

ENVIRONMENTS = ("tworoom", "pusht", "reacher", "cube")
FAMILIES = ("lewm", "pldm")
TRAINING_SEEDS = (3072, 3073, 3074)
EXPECTED_TWOROOM_PROTOCOL = {
    "history_size": 3,
    "action_block": 5,
    "eval_budget": 50,
    "horizon": 5,
    "receding_horizon": 5,
    "cem_samples": 300,
    "cem_steps": 30,
    "cem_topk": 30,
}
EXPECTED_STANDARD_PROTOCOL = {
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
    """Raised when an input is not a complete frozen seed-completion result."""


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
    actual: Any, expected: Mapping[str, Any], *, label: str
) -> dict[str, Any]:
    observed = _identity_from_mapping(actual, label=label)
    expected_identity = _identity_from_mapping(expected, label=f"expected {label}")
    if observed["sha256"] != expected_identity["sha256"]:
        raise FinalizationError(f"{label} SHA-256 drifted")
    if observed["size_bytes"] != expected_identity["size_bytes"]:
        raise FinalizationError(f"{label} size drifted")
    return observed


def _assert_file_matches(
    path: Path, expected: Mapping[str, Any], *, label: str
) -> dict[str, Any]:
    return _assert_identity(_file_identity(path, label=label), expected, label=label)


def _successes(value: Any, *, expected_count: int, label: str) -> list[bool]:
    rows = _list(value, label=label)
    if len(rows) != expected_count or any(not isinstance(row, bool) for row in rows):
        raise FinalizationError(f"{label} must contain {expected_count} booleans")
    return [bool(row) for row in rows]


def _validate_aggregate(
    aggregate: Any, *, successes: int, evaluations: int, label: str
) -> dict[str, Any]:
    row = _mapping(aggregate, label=label)
    success_value = row.get("success_count", row.get("successes"))
    evaluation_value = row.get(
        "evaluation_count", row.get("evaluations", row.get("query_count"))
    )
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


def _assert_runtime(runtime: Any, *, expected_commit: str, label: str) -> None:
    row = _mapping(runtime, label=label)
    if str(row.get("commit", "")) != expected_commit:
        raise FinalizationError(f"{label} commit drifted")
    if row.get("clean") is not True:
        raise FinalizationError(f"{label} must record a clean checkout")


def _assert_state_pair(value: Any, *, label: str) -> str:
    row = _mapping(value, label=label)
    if row.get("passed") is not True:
        raise FinalizationError(f"{label}.passed must be true")
    before = _mapping(row.get("before"), label=f"{label}.before")
    after = _mapping(row.get("after"), label=f"{label}.after")
    before_sha = _require_sha256(
        before.get("state_dict_sha256"), label=f"{label}.before.state_dict_sha256"
    )
    after_sha = _require_sha256(
        after.get("state_dict_sha256"), label=f"{label}.after.state_dict_sha256"
    )
    if before_sha != after_sha:
        raise FinalizationError(f"{label} model state changed during CEM")
    if int(before.get("parameter_count", -1)) != int(after.get("parameter_count", -2)):
        raise FinalizationError(f"{label} parameter count changed during CEM")
    return before_sha


def load_preregistration(path: Path, *, root: Path) -> dict[str, Any]:
    resolved = _resolve(path, root=root)
    if not resolved.is_file() or resolved.is_symlink():
        raise FileNotFoundError(f"Missing seed-completion preregistration: {resolved}")
    document = dict(
        _mapping(
            yaml.safe_load(resolved.read_text(encoding="utf-8")),
            label="seed-completion preregistration",
        )
    )
    if document.get("schema_version") != 1:
        raise FinalizationError("seed-completion preregistration schema drifted")
    if document.get("preregistration_id") != PREREG_ID:
        raise FinalizationError("unexpected seed-completion preregistration id")
    if document.get("status") != "frozen_before_cem_execution":
        raise FinalizationError("seed-completion preregistration is not frozen")
    scope = _mapping(document.get("scientific_scope"), label="scientific_scope")
    if (
        tuple(scope.get("environments", ())) != ENVIRONMENTS
        or tuple(scope.get("families", ())) != FAMILIES
        or tuple(scope.get("training_seed_set_per_family", ())) != TRAINING_SEEDS
        or int(scope.get("newly_executed_member_cells", -1)) != 17
        or int(scope.get("newly_executed_episodes", -1)) != 5100
        or scope.get("formal_suite_scoreboard_eligible") is not False
        or scope.get("cross_environment_average_authorized") is not False
        or scope.get("pass_fail_threshold") is not None
    ):
        raise FinalizationError("seed-completion scientific scope drifted")
    authority = _mapping(document.get("authority"), label="authority")
    for field in (
        "training_authorized",
        "finetuning_authorized",
        "checkpoint_selection_authorized",
        "model_or_recipe_change_authorized",
        "result_based_retry_authorized",
        "checkpoint_swap_authorized",
        "public_test_access_authorized",
        "formal_scoreboard_mutation_authorized",
    ):
        if authority.get(field) is not False:
            raise FinalizationError(f"authority.{field} must remain false")
    cells = _list(document.get("new_member_cells"), label="new_member_cells")
    if len(cells) != 17:
        raise FinalizationError("preregistration must expand to 17 new member cells")
    return {"path": resolved, "payload": document}


def _verify_runner_identities(prereg: Mapping[str, Any], *, root: Path) -> list[dict[str, Any]]:
    implementation = _mapping(prereg.get("implementation"), label="implementation")
    observed: list[dict[str, Any]] = []
    for name, spec in implementation.items():
        if not isinstance(spec, Mapping) or "sha256" not in spec:
            continue
        expected = _identity_from_mapping(spec, label=f"implementation.{name}")
        path = _resolve(expected["path"], root=root)
        observed.append(
            {
                "name": str(name),
                **_assert_file_matches(path, expected, label=f"implementation.{name}"),
            }
        )
    if not observed:
        raise FinalizationError("implementation identity set is empty")
    return observed


def _load_carried_members(
    prereg: Mapping[str, Any], *, root: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Bind the eight already-evaluated members to the frozen parent matrix."""

    freeze_path, freeze = _read_json(
        _resolve(PARENT_RESULTS_FREEZE, root=root), label="parent CEM results freeze"
    )
    if freeze.get("freeze_id") != "contextworld_original_baseline_cem_results_freeze_v1":
        raise FinalizationError("parent CEM results freeze id drifted")
    summary_identity = _identity_from_mapping(
        freeze.get("matrix_summary"), label="parent matrix_summary identity"
    )
    summary_path = _resolve(summary_identity["path"], root=root)
    _assert_file_matches(summary_path, summary_identity, label="parent matrix_summary")
    frozen_cells = {
        (str(row["environment"]), str(row["family"])): int(row["success_count"])
        for row in _list(freeze.get("cells"), label="parent freeze cells")
    }
    members: list[dict[str, Any]] = []
    for raw in _list(
        prereg.get("already_evaluated_family_members"),
        label="already_evaluated_family_members",
    ):
        row = _mapping(raw, label="already-evaluated member")
        environment = str(_require(row.get("environment"), label="member.environment"))
        family = str(_require(row.get("family"), label="member.family"))
        successes = int(_require(row.get("successes"), label="member.successes"))
        evaluations = int(_require(row.get("evaluations"), label="member.evaluations"))
        if evaluations != 300:
            raise FinalizationError("carried members must contain 300 episodes")
        if frozen_cells.get((environment, family)) != successes:
            raise FinalizationError(
                f"carried member {environment}/{family} does not match the frozen parent matrix"
            )
        members.append(
            {
                "environment": environment,
                "family": family,
                "training_seed": int(_require(row.get("training_seed"), label="member.training_seed")),
                "success_count": successes,
                "evaluation_count": evaluations,
                "success_rate": successes / evaluations,
                "provenance": f"carried_{row.get('source')}",
                "family_statistics_membership": str(
                    row.get("family_statistics_membership", "included")
                ),
                "reason": str(row["reason"]).strip() if "reason" in row else None,
            }
        )
    if len(members) != 8:
        raise FinalizationError("expected exactly eight already-evaluated members")
    excluded = [
        member
        for member in members
        if member["family_statistics_membership"] == "excluded_lineage_note_only"
    ]
    if len(excluded) != 1 or (
        excluded[0]["environment"],
        excluded[0]["family"],
        excluded[0]["training_seed"],
    ) != ("tworoom", "lewm", 3072):
        raise FinalizationError(
            "exactly the tworoom/lewm h3_origheldout member must be a lineage note"
        )
    return members, {
        "results_freeze": _file_identity(freeze_path, label="parent CEM results freeze"),
        "matrix_summary": _file_identity(summary_path, label="parent matrix_summary"),
    }


def _validate_tworoom_cell(
    cell: Mapping[str, Any], *, prereg: Mapping[str, Any], root: Path
) -> dict[str, Any]:
    cell_id = str(cell["cell_id"])
    inputs = _mapping(
        _mapping(prereg.get("frozen_environment_inputs"), label="inputs").get("tworoom"),
        label="inputs.tworoom",
    )
    runtime = _mapping(
        _mapping(prereg.get("runtimes"), label="runtimes").get("tworoom"),
        label="runtimes.tworoom",
    )
    expected_commit = str(runtime["expected_commit"])
    expected_checkpoint = _identity_from_mapping(
        cell.get("checkpoint"), label=f"{cell_id}.checkpoint"
    )
    expected_config_sha = _require_sha256(
        _mapping(cell.get("effective_loader_config"), label=f"{cell_id}.config").get("sha256"),
        label=f"{cell_id}.config.sha256",
    )
    output_directory = _resolve(str(cell["output_directory"]), root=root)
    state_digests: set[str] = set()
    per_eval_seed: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    total = 0
    for seed in (int(value) for value in cell["eval_seeds"]):
        path, report = _read_json(
            output_directory / f"seed{seed}.json", label=f"{cell_id} seed {seed} receipt"
        )
        if report.get("status") != "passed":
            raise FinalizationError(f"{cell_id} seed {seed} did not complete")
        protocol = _mapping(report.get("protocol"), label=f"{cell_id} seed {seed}.protocol")
        if int(protocol.get("eval_seed", -1)) != seed or int(protocol.get("evaluations", -1)) != 50:
            raise FinalizationError(f"{cell_id} seed {seed} protocol seed/count drifted")
        for field, expected in EXPECTED_TWOROOM_PROTOCOL.items():
            if int(protocol.get(field, -1)) != expected:
                raise FinalizationError(f"{cell_id} seed {seed} protocol.{field} drifted")
        preflight = _mapping(
            report.get("frozen_input_preflight"),
            label=f"{cell_id} seed {seed}.frozen_input_preflight",
        )
        if preflight.get("passed") is not True:
            raise FinalizationError(f"{cell_id} seed {seed} input preflight failed")
        _assert_runtime(
            preflight.get("runtime"),
            expected_commit=expected_commit,
            label=f"{cell_id} seed {seed}.runtime",
        )
        _assert_identity(
            preflight.get("checkpoint"), expected_checkpoint, label=f"{cell_id} checkpoint"
        )
        if (
            _require_sha256(
                _mapping(preflight.get("config"), label=f"{cell_id}.config").get("sha256"),
                label=f"{cell_id}.config.sha256",
            )
            != expected_config_sha
        ):
            raise FinalizationError(f"{cell_id} loader config drifted")
        _assert_identity(
            preflight.get("catalog"), inputs["catalog"], label=f"{cell_id} catalog"
        )
        _assert_identity(
            preflight.get("normalizer"), inputs["normalizer"], label=f"{cell_id} normalizer"
        )
        _assert_identity(
            preflight.get("source_dataset"), inputs["dataset"], label=f"{cell_id} dataset"
        )
        stable = _mapping(
            report.get("stable_worldmodel"), label=f"{cell_id} seed {seed}.stable_worldmodel"
        )
        if str(stable.get("commit", "")) != expected_commit:
            raise FinalizationError(f"{cell_id} seed {seed} report runtime drifted")
        audit = _mapping(
            report.get("frozen_weight_audit"), label=f"{cell_id} seed {seed}.frozen_weight_audit"
        )
        if audit.get("passed") is not True:
            raise FinalizationError(f"{cell_id} seed {seed} weight audit failed")
        before = _require_sha256(
            audit.get("state_dict_sha256_before"),
            label=f"{cell_id} seed {seed}.state_dict_sha256_before",
        )
        if before != _require_sha256(
            audit.get("state_dict_sha256_after"),
            label=f"{cell_id} seed {seed}.state_dict_sha256_after",
        ):
            raise FinalizationError(f"{cell_id} seed {seed} model state changed during CEM")
        state_digests.add(before)
        raw_records = _list(report.get("raw_records"), label=f"{cell_id} seed {seed}.raw_records")
        if len(raw_records) != 50:
            raise FinalizationError(f"{cell_id} seed {seed} raw record count drifted")
        outcomes: list[bool] = []
        indices: list[int] = []
        for raw in raw_records:
            record = _mapping(raw, label=f"{cell_id} raw record")
            if int(record.get("eval_seed", -1)) != seed:
                raise FinalizationError(f"{cell_id} seed assignment drifted")
            indices.append(int(record.get("evaluation_index", -1)))
            if not isinstance(record.get("success"), bool):
                raise FinalizationError(f"{cell_id} raw success is invalid")
            outcomes.append(bool(record["success"]))
        if sorted(indices) != list(range(50)):
            raise FinalizationError(f"{cell_id} seed {seed} query index coverage drifted")
        aggregate = _validate_aggregate(
            report.get("aggregate"),
            successes=sum(outcomes),
            evaluations=50,
            label=f"{cell_id} seed {seed}.aggregate",
        )
        total += aggregate["success_count"]
        per_eval_seed.append({"eval_seed": seed, **aggregate})
        sources.append(_file_identity(path, label=f"{cell_id} seed {seed} receipt"))
    if len(state_digests) != 1:
        raise FinalizationError(f"{cell_id} loaded model state differs across eval seeds")
    return {
        "success_count": total,
        "evaluation_count": 300,
        "success_rate": total / 300,
        "per_eval_seed": per_eval_seed,
        "model_state_audit": {
            "passed": True,
            "loaded_state_dict_sha256": next(iter(state_digests)),
            "scope": "actual_model_per_seed",
        },
        "sources": sources,
    }


def _validate_standard_cell(
    cell: Mapping[str, Any], *, prereg: Mapping[str, Any], root: Path
) -> dict[str, Any]:
    cell_id = str(cell["cell_id"])
    environment = str(cell["environment"])
    inputs = _mapping(
        _mapping(prereg.get("frozen_environment_inputs"), label="inputs").get(environment),
        label=f"inputs.{environment}",
    )
    runtime = _mapping(
        _mapping(prereg.get("runtimes"), label="runtimes").get("pusht_reacher_cube"),
        label="runtimes.pusht_reacher_cube",
    )
    expected_commit = str(runtime["expected_commit"])
    expected_checkpoint = _identity_from_mapping(
        cell.get("checkpoint"), label=f"{cell_id}.checkpoint"
    )
    expected_config = _identity_from_mapping(
        cell.get("effective_loader_config"), label=f"{cell_id}.config"
    )
    seeds = tuple(int(value) for value in cell["eval_seeds"])
    queries = int(cell["queries_per_seed"])
    path, report = _read_json(
        _resolve(str(cell["output_directory"]), root=root) / "aggregate.json",
        label=f"{cell_id} aggregate",
    )
    if (
        report.get("status") != "standard_original_task_real_environment_cem"
        or report.get("task") != environment
    ):
        raise FinalizationError(f"{cell_id} aggregate kind drifted")
    _assert_runtime(report.get("runtime"), expected_commit=expected_commit, label=f"{cell_id}.runtime")
    protocol = _mapping(report.get("protocol"), label=f"{cell_id}.protocol")
    for field, expected in EXPECTED_STANDARD_PROTOCOL.items():
        if int(protocol.get(field, -1)) != expected:
            raise FinalizationError(f"{cell_id} protocol.{field} drifted")
    if tuple(int(value) for value in protocol.get("eval_seeds", ())) != seeds:
        raise FinalizationError(f"{cell_id} protocol seed schedule drifted")
    if int(protocol.get("num_eval_per_seed", protocol.get("queries_per_seed", -1))) != queries:
        raise FinalizationError(f"{cell_id} protocol query count drifted")
    _assert_identity(protocol.get("dataset"), inputs["dataset"], label=f"{cell_id} dataset")
    catalog = _mapping(report.get("query_catalog"), label=f"{cell_id}.query_catalog")
    _assert_identity(catalog.get("frozen_source"), inputs["catalog"], label=f"{cell_id} catalog")
    public = _mapping(report.get("public_test"), label=f"{cell_id}.public_test")
    for field in ("contextworld_public_test_read", "contextworld_public_test_scored"):
        if public.get(field) is not False:
            raise FinalizationError(f"{cell_id}.public_test.{field} must remain false")
    model = _mapping(report.get("model"), label=f"{cell_id}.model")
    if str(model.get("model", "")) != cell_id:
        raise FinalizationError(f"{cell_id} model name drifted")
    _assert_identity(
        {
            "path": model.get("checkpoint"),
            "sha256": model.get("checkpoint_sha256"),
            "size_bytes": model.get("checkpoint_size_bytes"),
        },
        expected_checkpoint,
        label=f"{cell_id} checkpoint",
    )
    _assert_identity(
        {
            "path": model.get("config"),
            "sha256": model.get("config_sha256"),
            "size_bytes": model.get("config_size_bytes"),
        },
        expected_config,
        label=f"{cell_id} loader config",
    )
    state_audit = _mapping(model.get("frozen_state_audit"), label=f"{cell_id}.frozen_state_audit")
    if state_audit.get("passed") is not True or state_audit.get("scope") != "actual_policy_model_per_seed":
        raise FinalizationError(f"{cell_id} model-state audit did not pass")
    audit_rows = {
        int(_mapping(row, label="state audit row").get("eval_seed", -1)): row
        for row in _list(state_audit.get("seeds"), label=f"{cell_id}.state_audit.seeds")
    }
    model_rows = {
        int(_mapping(row, label="model seed row").get("eval_seed", -1)): row
        for row in _list(model.get("seeds"), label=f"{cell_id}.model.seeds")
    }
    if set(audit_rows) != set(seeds) or set(model_rows) != set(seeds):
        raise FinalizationError(f"{cell_id} eval-seed coverage drifted")
    state_digests: set[str] = set()
    per_eval_seed: list[dict[str, Any]] = []
    outcomes: list[bool] = []
    for seed in seeds:
        row = _mapping(model_rows[seed], label=f"{cell_id} seed {seed}")
        values = _successes(
            row.get("episode_successes"),
            expected_count=queries,
            label=f"{cell_id} seed {seed}.episode_successes",
        )
        aggregate = _validate_aggregate(
            row, successes=sum(values), evaluations=queries, label=f"{cell_id} seed {seed}"
        )
        state_digests.add(
            _assert_state_pair(audit_rows[seed], label=f"{cell_id} state audit {seed}")
        )
        state_digests.add(
            _assert_state_pair(
                row.get("frozen_state_audit"), label=f"{cell_id} seed {seed}.frozen_state_audit"
            )
        )
        per_eval_seed.append({"eval_seed": seed, **aggregate})
        outcomes.extend(values)
    if len(state_digests) != 1:
        raise FinalizationError(f"{cell_id} loaded model state differs across eval seeds")
    aggregate = _validate_aggregate(
        model.get("aggregate"), successes=sum(outcomes), evaluations=300, label=f"{cell_id}.aggregate"
    )
    return {
        **aggregate,
        "per_eval_seed": per_eval_seed,
        "model_state_audit": {
            "passed": True,
            "loaded_state_dict_sha256": next(iter(state_digests)),
            "scope": "actual_policy_model_per_seed",
        },
        "sources": [_file_identity(path, label=f"{cell_id} aggregate")],
    }


def _validate_cube_pldm_cell(
    cell: Mapping[str, Any], *, prereg: Mapping[str, Any], root: Path
) -> dict[str, Any]:
    cell_id = str(cell["cell_id"])
    inputs = _mapping(
        _mapping(prereg.get("frozen_environment_inputs"), label="inputs").get("cube"),
        label="inputs.cube",
    )
    runtime = _mapping(
        _mapping(prereg.get("runtimes"), label="runtimes").get("pusht_reacher_cube"),
        label="runtimes.pusht_reacher_cube",
    )
    expected_commit = str(runtime["expected_commit"])
    expected_checkpoint = _identity_from_mapping(
        cell.get("checkpoint"), label=f"{cell_id}.checkpoint"
    )
    expected_config = _identity_from_mapping(
        cell.get("effective_loader_config"), label=f"{cell_id}.config"
    )
    seeds = tuple(int(value) for value in cell["eval_seeds"])
    path, report = _read_json(
        _resolve(str(cell["output_directory"]), root=root) / "aggregate.json",
        label=f"{cell_id} aggregate",
    )
    if (
        report.get("status") != "standard_original_task_real_environment_cem"
        or report.get("task") != "cube"
    ):
        raise FinalizationError(f"{cell_id} aggregate kind drifted")
    _assert_runtime(report.get("runtime"), expected_commit=expected_commit, label=f"{cell_id}.runtime")
    public = _mapping(report.get("public_test"), label=f"{cell_id}.public_test")
    for field in ("opened", "read", "hashed", "scored"):
        if public.get(field) is not False:
            raise FinalizationError(f"{cell_id}.public_test.{field} must remain false")
    model = _mapping(report.get("model"), label=f"{cell_id}.model")
    if model.get("family") != "pldm" or model.get("strict_load") is not True:
        raise FinalizationError(f"{cell_id} strict family/load audit drifted")
    _assert_identity(model.get("checkpoint"), expected_checkpoint, label=f"{cell_id} checkpoint")
    _assert_identity(model.get("config"), expected_config, label=f"{cell_id} loader config")
    if model.get("loaded_state_consistent_across_seeds") is not True:
        raise FinalizationError(f"{cell_id} loaded-state consistency audit failed")
    loaded_state = _require_sha256(
        model.get("loaded_state_dict_sha256"), label=f"{cell_id}.loaded_state_dict_sha256"
    )
    catalog = _mapping(report.get("query_catalog"), label=f"{cell_id}.query_catalog")
    expected_catalog = _identity_from_mapping(inputs["catalog"], label="expected cube catalog")
    if _require_sha256(catalog.get("sha256"), label=f"{cell_id}.query_catalog.sha256") != expected_catalog["sha256"]:
        raise FinalizationError(f"{cell_id} catalog drifted")
    model_rows = {
        int(_mapping(row, label="model seed row").get("eval_seed", -1)): row
        for row in _list(model.get("seeds"), label=f"{cell_id}.model.seeds")
    }
    if set(model_rows) != set(seeds):
        raise FinalizationError(f"{cell_id} eval-seed coverage drifted")
    per_eval_seed: list[dict[str, Any]] = []
    outcomes: list[bool] = []
    for seed in seeds:
        row = _mapping(model_rows[seed], label=f"{cell_id} seed {seed}")
        values = _successes(
            row.get("episode_successes"),
            expected_count=100,
            label=f"{cell_id} seed {seed}.episode_successes",
        )
        aggregate = _validate_aggregate(
            row, successes=sum(values), evaluations=100, label=f"{cell_id} seed {seed}"
        )
        observed_state = _assert_state_pair(
            row.get("actual_evaluated_model_state"),
            label=f"{cell_id} seed {seed}.actual_evaluated_model_state",
        )
        if observed_state != loaded_state:
            raise FinalizationError(f"{cell_id} seed {seed} model state drifted")
        per_eval_seed.append({"eval_seed": seed, **aggregate})
        outcomes.extend(values)
    aggregate = _validate_aggregate(
        model.get("aggregate"), successes=sum(outcomes), evaluations=300, label=f"{cell_id}.aggregate"
    )
    return {
        **aggregate,
        "per_eval_seed": per_eval_seed,
        "model_state_audit": {
            "passed": True,
            "loaded_state_dict_sha256": loaded_state,
            "scope": "actual_model_per_seed",
        },
        "sources": [_file_identity(path, label=f"{cell_id} aggregate")],
    }


def _validate_cube_lewm_cell(
    cell: Mapping[str, Any], *, prereg: Mapping[str, Any], root: Path
) -> dict[str, Any]:
    cell_id = str(cell["cell_id"])
    inputs = _mapping(
        _mapping(prereg.get("frozen_environment_inputs"), label="inputs").get("cube"),
        label="inputs.cube",
    )
    runtime = _mapping(
        _mapping(prereg.get("runtimes"), label="runtimes").get("pusht_reacher_cube"),
        label="runtimes.pusht_reacher_cube",
    )
    expected_commit = str(runtime["expected_commit"])
    plan = _mapping(
        _mapping(runtime.get("plan_configs"), label="plan_configs").get("cube"),
        label="plan_configs.cube",
    )
    expected_checkpoint = _identity_from_mapping(
        cell.get("checkpoint"), label=f"{cell_id}.checkpoint"
    )
    expected_config_sha = _require_sha256(
        _mapping(cell.get("effective_loader_config"), label=f"{cell_id}.config").get("sha256"),
        label=f"{cell_id}.config.sha256",
    )
    seeds = tuple(int(value) for value in cell["eval_seeds"])
    path, report = _read_json(
        _resolve(str(cell["output_directory"]), root=root) / "aggregate.json",
        label=f"{cell_id} aggregate",
    )
    if (
        report.get("status") != "standard_original_task_real_environment_cem"
        or report.get("task") != "cube"
    ):
        raise FinalizationError(f"{cell_id} aggregate kind drifted")
    _assert_runtime(report.get("runtime"), expected_commit=expected_commit, label=f"{cell_id}.runtime")
    protocol = _mapping(report.get("protocol"), label=f"{cell_id}.protocol")
    for field, expected in EXPECTED_STANDARD_PROTOCOL.items():
        if int(protocol.get(field, -1)) != expected:
            raise FinalizationError(f"{cell_id} protocol.{field} drifted")
    if tuple(int(value) for value in protocol.get("eval_seeds", ())) != seeds:
        raise FinalizationError(f"{cell_id} protocol seed schedule drifted")
    if int(protocol.get("num_eval_per_seed", -1)) != 100:
        raise FinalizationError(f"{cell_id} protocol query count drifted")
    expected_dataset = _identity_from_mapping(inputs["dataset"], label="expected cube dataset")
    if (
        Path(str(protocol.get("dataset", ""))).expanduser().resolve()
        != Path(expected_dataset["path"]).expanduser().resolve()
        or _require_sha256(protocol.get("dataset_sha256"), label=f"{cell_id}.dataset_sha256")
        != expected_dataset["sha256"]
        or int(protocol.get("dataset_size_bytes", -1)) != expected_dataset["size_bytes"]
    ):
        raise FinalizationError(f"{cell_id} dataset identity drifted")
    if _require_sha256(protocol.get("source_sha256"), label=f"{cell_id}.plan_config_sha256") != str(
        plan["sha256"]
    ):
        raise FinalizationError(f"{cell_id} frozen plan config drifted")
    public = _mapping(report.get("public_test"), label=f"{cell_id}.public_test")
    for field in ("opened", "read", "hashed", "scored"):
        if public.get(field) is not False:
            raise FinalizationError(f"{cell_id}.public_test.{field} must remain false")
    catalog = _mapping(report.get("query_catalog"), label=f"{cell_id}.query_catalog")
    expected_catalog = _identity_from_mapping(inputs["catalog"], label="expected cube catalog")
    if (
        Path(str(catalog.get("frozen_source", ""))).expanduser().resolve()
        != Path(expected_catalog["path"]).expanduser().resolve()
        or _require_sha256(catalog.get("sha256"), label=f"{cell_id}.query_catalog.sha256")
        != expected_catalog["sha256"]
    ):
        raise FinalizationError(f"{cell_id} catalog drifted")
    models = _list(report.get("models"), label=f"{cell_id}.models")
    if len(models) != 1:
        raise FinalizationError(f"{cell_id} aggregate must contain exactly one model")
    model = _mapping(models[0], label=f"{cell_id}.model")
    if model.get("model") != "baseline_lewm":
        raise FinalizationError(f"{cell_id} evaluator model label drifted")
    if (
        Path(str(model.get("checkpoint", ""))).expanduser().resolve()
        != Path(expected_checkpoint["path"]).expanduser().resolve()
        or _require_sha256(model.get("checkpoint_sha256"), label=f"{cell_id}.checkpoint_sha256")
        != expected_checkpoint["sha256"]
    ):
        raise FinalizationError(f"{cell_id} checkpoint drifted")
    if _require_sha256(model.get("config_sha256"), label=f"{cell_id}.config_sha256") != expected_config_sha:
        raise FinalizationError(f"{cell_id} loader config drifted")
    model_rows = {
        int(_mapping(row, label="model seed row").get("eval_seed", -1)): row
        for row in _list(model.get("seeds"), label=f"{cell_id}.model.seeds")
    }
    if set(model_rows) != set(seeds):
        raise FinalizationError(f"{cell_id} eval-seed coverage drifted")
    per_eval_seed: list[dict[str, Any]] = []
    outcomes: list[bool] = []
    for seed in seeds:
        row = _mapping(model_rows[seed], label=f"{cell_id} seed {seed}")
        values = _successes(
            row.get("episode_successes"),
            expected_count=100,
            label=f"{cell_id} seed {seed}.episode_successes",
        )
        aggregate = _validate_aggregate(
            row, successes=sum(values), evaluations=100, label=f"{cell_id} seed {seed}"
        )
        per_eval_seed.append({"eval_seed": seed, **aggregate})
        outcomes.extend(values)
    aggregate = _validate_aggregate(
        model.get("aggregate"), successes=sum(outcomes), evaluations=300, label=f"{cell_id}.aggregate"
    )
    return {
        **aggregate,
        "per_eval_seed": per_eval_seed,
        "model_state_audit": {
            "passed": True,
            "scope": "launcher_verified_identity_and_evaluator_strict_load",
            "note": (
                "the frozen v2 retention evaluator does not emit a per-seed "
                "state-dict audit; checkpoint/config/catalog/plan identities "
                "were closed by the frozen launcher before launch and are "
                "re-verified against this aggregate above"
            ),
        },
        "sources": [_file_identity(path, label=f"{cell_id} aggregate")],
    }


def _validate_eval43_recovery(*, root: Path, seed43_receipt: Path) -> dict[str, Any]:
    """Close the single authorized infrastructure relaunch for tworoom/pldm/3074."""

    prereg_path = _resolve(EVAL43_RECOVERY_PREREG, root=root)
    if not prereg_path.is_file():
        raise FinalizationError("eval43 infrastructure recovery preregistration is missing")
    document = dict(
        _mapping(
            yaml.safe_load(prereg_path.read_text(encoding="utf-8")),
            label="eval43 recovery preregistration",
        )
    )
    if document.get("recovery_id") != EVAL43_RECOVERY_ID:
        raise FinalizationError("eval43 recovery id drifted")
    scope = _mapping(document.get("scope"), label="recovery scope")
    if (
        scope.get("result_observed_before_failure") is not False
        or scope.get("result_based_retry") is not False
        or int(scope.get("relaunch_count_authorized", -1)) != 1
    ):
        raise FinalizationError("eval43 recovery scope drifted")
    receipt_path, receipt = _read_json(
        _resolve(EVAL43_RELAUNCH_RECEIPT, root=root), label="eval43 relaunch receipt"
    )
    if receipt.get("recovery_id") != EVAL43_RECOVERY_ID:
        raise FinalizationError("eval43 relaunch receipt recovery id drifted")
    relaunch = _mapping(receipt.get("relaunch"), label="relaunch")
    if (
        str(relaunch.get("job_id", "")) != "tworoom_pldm_seed3074_eval43"
        or int(relaunch.get("exit_code", -1)) != 0
    ):
        raise FinalizationError("eval43 relaunch did not complete cleanly")
    output_identity = _identity_from_mapping(
        relaunch.get("output"), label="relaunch.output"
    )
    observed = _file_identity(seed43_receipt, label="relaunched seed43 receipt")
    if (
        observed["sha256"] != output_identity["sha256"]
        or observed["size_bytes"] != output_identity["size_bytes"]
    ):
        raise FinalizationError("relaunched seed43 receipt does not match the relaunch receipt")
    return {
        "recovery_id": EVAL43_RECOVERY_ID,
        "preregistration": _file_identity(prereg_path, label="eval43 recovery preregistration"),
        "relaunch_receipt": _file_identity(receipt_path, label="eval43 relaunch receipt"),
    }


def _family_statistics(rates: Sequence[float]) -> dict[str, Any]:
    values = [float(value) for value in rates]
    count = len(values)
    mean = sum(values) / count
    variance = (
        sum((value - mean) ** 2 for value in values) / (count - 1) if count > 1 else 0.0
    )
    return {
        "n_training_seeds": count,
        "success_rates": values,
        "mean": mean,
        "sample_std": math.sqrt(variance),
        "sample_variance": variance,
        "minimum": min(values),
        "maximum": max(values),
    }


def _write_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite family summary: {path}")
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(encoded)
        stream.flush()


def finalize(
    *, prereg_path: Path = DEFAULT_PREREG, repo_root: Path = ROOT, output: Path | None = None
) -> dict[str, Any]:
    root = repo_root.expanduser().resolve()
    preregistration = load_preregistration(prereg_path, root=root)
    prereg = preregistration["payload"]
    policy = _mapping(prereg.get("execution_policy"), label="execution_policy")
    declared_root = _resolve(str(policy["output_root"]), root=root)
    if not declared_root.is_dir():
        raise FinalizationError("preregistered output namespace does not exist")
    declared_summary = Path(str(policy["result_summary"]))
    if declared_summary.name != "family_summary.json" or not str(
        declared_summary
    ).startswith(str(policy["output_root"])):
        raise FinalizationError("result_summary drifted from the preregistered namespace")
    # Anchor the summary inside the (already existing) receipt namespace so it
    # lands next to the receipts; resolve_contextworld_path would otherwise
    # fall back to the data tree for a not-yet-existing artifacts path.
    declared_output = declared_root / declared_summary.name
    target = declared_output if output is None else output.expanduser().resolve()
    if target != declared_output:
        raise FinalizationError("family summary output path differs from preregistration")
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Refusing to overwrite family summary: {target}")
    implementation = _verify_runner_identities(prereg, root=root)
    preflight_identity = _file_identity(
        _resolve(str(policy["preflight_receipt"]), root=root), label="preflight receipt"
    )
    freeze_identity = _file_identity(
        _resolve(str(policy["freeze_receipt"]), root=root), label="freeze receipt"
    )
    carried_members, carried_binding = _load_carried_members(prereg, root=root)

    validators = {
        "standard_runner": _validate_standard_cell,
        "tworoom_runner": _validate_tworoom_cell,
        "cube_pldm_wrapper": _validate_cube_pldm_cell,
        "cube_lewm_evaluator_v2": _validate_cube_lewm_cell,
    }
    new_members: list[dict[str, Any]] = []
    for raw in _list(prereg.get("new_member_cells"), label="new_member_cells"):
        cell = _mapping(raw, label="new member cell")
        runner = str(cell.get("runner"))
        validator = validators.get(runner)
        if validator is None:
            raise FinalizationError(f"Unknown runner for cell {cell.get('cell_id')}")
        validated = validator(cell, prereg=prereg, root=root)
        new_members.append(
            {
                "environment": str(cell["environment"]),
                "family": str(cell["family"]),
                "training_seed": int(cell["training_seed"]),
                "cell_id": str(cell["cell_id"]),
                "provenance": "new_cell_this_preregistration",
                **validated,
            }
        )
    if len(new_members) != 17:
        raise FinalizationError("expected exactly 17 newly executed member cells")
    if sum(member["evaluation_count"] for member in new_members) != 5100:
        raise FinalizationError("newly executed episodes must total 5100")

    recovery = _validate_eval43_recovery(
        root=root,
        seed43_receipt=_resolve(
            "artifacts/evaluation/original_baseline_seed_completion_v1/tworoom/pldm/"
            "seed3074/seed43.json",
            root=root,
        ),
    )

    families: list[dict[str, Any]] = []
    for environment in ENVIRONMENTS:
        for family in FAMILIES:
            members = [
                member
                for member in new_members
                if member["environment"] == environment and member["family"] == family
            ] + [
                member
                for member in carried_members
                if member["environment"] == environment
                and member["family"] == family
                and member["family_statistics_membership"] != "excluded_lineage_note_only"
            ]
            members.sort(key=lambda member: member["training_seed"])
            if tuple(member["training_seed"] for member in members) != TRAINING_SEEDS:
                raise FinalizationError(
                    f"family {environment}/{family} does not close training seeds 3072/3073/3074"
                )
            if any(member["evaluation_count"] != 300 for member in members):
                raise FinalizationError(
                    f"family {environment}/{family} contains a member without 300 episodes"
                )
            lineage_notes = [
                {
                    "training_seed": member["training_seed"],
                    "success_count": member["success_count"],
                    "evaluation_count": member["evaluation_count"],
                    "success_rate": member["success_rate"],
                    "provenance": member["provenance"],
                    "excluded_from_family_statistics": True,
                    "reason": member["reason"],
                }
                for member in carried_members
                if member["environment"] == environment
                and member["family"] == family
                and member["family_statistics_membership"] == "excluded_lineage_note_only"
            ]
            families.append(
                {
                    "environment": environment,
                    "family": family,
                    "members": [
                        {
                            key: value
                            for key, value in member.items()
                            if key
                            not in {"family_statistics_membership", "reason", "environment", "family"}
                        }
                        for member in members
                    ],
                    "statistics": _family_statistics(
                        [member["success_rate"] for member in members]
                    ),
                    "lineage_notes": lineage_notes,
                }
            )
    if len(families) != 8:
        raise FinalizationError("expected exactly eight baseline families")

    summary = {
        "schema_version": 1,
        "summary_id": "contextworld_original_baseline_seed_completion_family_summary_v1",
        "status": "completed_descriptive_original_environment_baseline_families",
        "preregistration": _file_identity(
            preregistration["path"], label="seed-completion preregistration"
        ),
        "pre_execution_closure": {
            "freeze_receipt": freeze_identity,
            "preflight_receipt": preflight_identity,
            "implementation": implementation,
            "finalizer": _file_identity(Path(__file__).resolve(), label="finalizer"),
        },
        "carried_membership_binding": carried_binding,
        "scope": {
            "result_kind": "post_release_descriptive_original_environment_baseline_seed_completion",
            "formal_suite_scoreboard_eligible": False,
            "cross_environment_average_authorized": False,
            "cross_environment_average_reported": False,
            "pass_fail_threshold": None,
            "public_test_accessed": False,
        },
        "counts": {
            "families": 8,
            "members_per_family": 3,
            "member_cells_total": 24,
            "newly_executed_member_cells": 17,
            "reused_member_cells": 7,
            "lineage_note_cells": 1,
            "episodes_per_member": 300,
            "newly_executed_episodes": 5100,
        },
        "execution_disclosures": [
            {
                "kind": "infrastructure_relaunch",
                "job_id": "tworoom_pldm_seed3074_eval43",
                "detail": (
                    "killed by a CUDA unspecified-launch-failure before any score "
                    "existed; relaunched exactly once under a dedicated recovery "
                    "identity after byte-verified removal of the 46-byte "
                    "pre-CEM reservation stub"
                ),
                **recovery,
            }
        ],
        "families": families,
        "interpretation": {
            "allowed": "per-environment original-task CEM baseline family statistics",
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
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = finalize(prereg_path=args.prereg, repo_root=args.repo_root, output=args.output)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "counts": summary["counts"],
                "families": [
                    {
                        "environment": family["environment"],
                        "family": family["family"],
                        "mean": family["statistics"]["mean"],
                        "sample_std": family["statistics"]["sample_std"],
                    }
                    for family in summary["families"]
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
