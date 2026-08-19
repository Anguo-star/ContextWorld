"""Fail-closed contract for the descriptive 4x2 original-task CEM matrix."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

import yaml

from contextworld.paths import repository_root, resolve_contextworld_path


DEFAULT_PREREG = Path(
    "configs/benchmark/contextworld_original_baseline_cem_prereg_v1.yaml"
)
EXPECTED_CELLS = {
    (environment, family)
    for environment in ("tworoom", "pusht", "reacher", "cube")
    for family in ("lewm", "pldm")
}
EXPECTED_EXECUTION_CELLS = EXPECTED_CELLS - {("cube", "lewm")}
EXPECTED_CHECKPOINT_IDS = {
    (environment, family): f"{environment}_{family}_original"
    for environment, family in EXPECTED_CELLS
}
EXPECTED_PROTOCOLS = {
    "tworoom": ((42, 43, 44, 45, 46, 47), 50, "egl"),
    "pusht": ((42, 43, 44, 45, 46, 47), 50, "egl"),
    "reacher": ((42, 43, 44), 100, "osmesa"),
    "cube": ((42, 43, 44), 100, "osmesa"),
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return value


def _resolve(path: str, *, root: Path) -> Path:
    value = Path(path).expanduser()
    if value.is_absolute():
        return value.resolve()
    return resolve_contextworld_path(value, repo_root=root)


def _same_identity(
    left: Mapping[str, Any], right: Mapping[str, Any], *, root: Path
) -> bool:
    return (
        _resolve(str(left.get("path", "")), root=root)
        == _resolve(str(right.get("path", "")), root=root)
        and str(left.get("sha256", "")) == str(right.get("sha256", ""))
        and int(left.get("size_bytes", -1)) == int(right.get("size_bytes", -1))
    )


def verify_identity(
    identity: Mapping[str, Any],
    *,
    root: Path,
    label: str,
    hash_large_files: bool = False,
) -> dict[str, Any]:
    path = _resolve(str(identity.get("path", "")), root=root)
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"{label}: {path}")
    size = int(identity.get("size_bytes", -1))
    expected = str(identity.get("sha256", ""))
    if path.stat().st_size != size:
        raise RuntimeError(f"{label} size drifted")
    if len(expected) != 64:
        raise ValueError(f"{label} needs a SHA-256 identity")
    hashed = size <= 1024 * 1024 * 1024 or hash_large_files
    observed = file_sha256(path) if hashed else expected
    if observed != expected:
        raise RuntimeError(f"{label} SHA-256 drifted")
    return {
        "path": str(path),
        "sha256": observed,
        "size_bytes": size,
        "content_hash_checked_now": hashed,
    }


def load_preregistration(
    path: Path | str = DEFAULT_PREREG,
    *,
    repo_root: Path | None = None,
    require_outputs_absent: bool = False,
    hash_large_files: bool = False,
) -> dict[str, Any]:
    root = (repo_root or repository_root()).resolve()
    config_path = _resolve(str(path), root=root)
    raw = config_path.read_text(encoding="utf-8")
    if "PENDING" in raw:
        raise RuntimeError("CEM preregistration still contains pending identities")
    document = yaml.safe_load(raw)
    prereg = dict(_mapping(document, "CEM preregistration"))
    if prereg.get("schema_version") != 1:
        raise ValueError("CEM preregistration schema_version must be 1")
    if prereg.get("preregistration_id") != "contextworld_original_baseline_cem_v1":
        raise ValueError("unexpected CEM preregistration id")
    if prereg.get("status") != "frozen_before_cem_execution":
        raise ValueError("CEM preregistration is not frozen")

    scope = _mapping(prereg.get("scientific_scope"), "scientific_scope")
    authority = _mapping(prereg.get("authority"), "authority")
    if (
        int(scope.get("matrix_cells", -1)) != 8
        or int(scope.get("exact_legacy_cells_reused", -1)) != 1
        or int(scope.get("newly_executed_cells", -1)) != 7
        or int(scope.get("newly_executed_episodes", -1)) != 2100
        or authority.get("cem_execution_authorized") is not True
        or int(authority.get("authorized_new_cells", -1)) != 7
        or int(authority.get("authorized_new_episodes", -1)) != 2100
    ):
        raise RuntimeError("CEM matrix authority/count contract drifted")
    for field in (
        "training_authorized",
        "finetuning_authorized",
        "checkpoint_selection_authorized",
        "result_based_retry_authorized",
        "checkpoint_swap_authorized",
        "public_test_access_authorized",
        "formal_scoreboard_mutation_authorized",
    ):
        if authority.get(field) is not False:
            raise RuntimeError(f"authority.{field} must remain false")

    reuse = {
        (str(row["environment"]), str(row["family"]))
        for row in prereg.get("reuse_cells", ())
    }
    execute = {
        (str(row["environment"]), str(row["family"]))
        for row in prereg.get("execution_cells", ())
    }
    if reuse | execute != EXPECTED_CELLS or reuse & execute:
        raise RuntimeError("CEM matrix cells are incomplete or duplicated")
    if execute != EXPECTED_EXECUTION_CELLS:
        raise RuntimeError("CEM execution cell set drifted")

    identities: dict[str, Any] = {}
    for name in (
        "base_checkpoint_registry",
        "base_icl_result_freeze",
        "input_identity_audit",
        "preflight",
    ):
        identities[name] = verify_identity(
            _mapping(prereg.get(name), name),
            root=root,
            label=name,
            hash_large_files=hash_large_files,
        )
    for name, row in _mapping(prereg.get("implementation"), "implementation").items():
        identities[f"implementation.{name}"] = verify_identity(
            _mapping(row, f"implementation.{name}"),
            root=root,
            label=f"implementation.{name}",
            hash_large_files=hash_large_files,
        )
    for row in prereg.get("reuse_cells", ()):
        key = f"reuse.{row['environment']}.{row['family']}"
        identities[key] = verify_identity(
            _mapping(row.get("source"), f"{key}.source"),
            root=root,
            label=key,
            hash_large_files=hash_large_files,
        )
        identities[f"{key}.result"] = verify_identity(
            _mapping(row.get("result"), f"{key}.result"),
            root=root,
            label=f"{key}.result",
            hash_large_files=hash_large_files,
        )
    for environment, values in _mapping(
        prereg.get("frozen_environment_inputs"), "frozen_environment_inputs"
    ).items():
        for name in ("catalog", "dataset"):
            key = f"input.{environment}.{name}"
            identities[key] = verify_identity(
                _mapping(values.get(name), key),
                root=root,
                label=key,
                hash_large_files=hash_large_files,
            )
        if isinstance(values.get("normalizer"), Mapping):
            key = f"input.{environment}.normalizer"
            identities[key] = verify_identity(
                _mapping(values["normalizer"], key),
                root=root,
                label=key,
                hash_large_files=hash_large_files,
            )

    registry_path = Path(identities["base_checkpoint_registry"]["path"])
    registry_document = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    registry_rows = {
        str(row["checkpoint_id"]): row
        for row in registry_document.get("checkpoints", ())
    }
    if set(registry_rows) != set(EXPECTED_CHECKPOINT_IDS.values()):
        raise RuntimeError("Base checkpoint registry is not the canonical 8-checkpoint set")

    implementation = _mapping(prereg.get("implementation"), "implementation")
    output_root = _resolve(
        str(prereg["execution_policy"]["output_root"]), root=root
    )
    observed_outputs: set[Path] = set()
    for row in (*prereg.get("reuse_cells", ()), *prereg.get("execution_cells", ())):
        environment = str(row.get("environment", ""))
        family = str(row.get("family", ""))
        cell = (environment, family)
        if str(row.get("checkpoint_id", "")) != EXPECTED_CHECKPOINT_IDS.get(cell):
            raise RuntimeError(f"CEM checkpoint mapping drifted for {cell}")

    for row in prereg.get("execution_cells", ()):
        environment = str(row["environment"])
        family = str(row["family"])
        checkpoint_id = str(row["checkpoint_id"])
        expected_seeds, expected_count, expected_gl = EXPECTED_PROTOCOLS[environment]
        if (
            tuple(int(value) for value in row.get("eval_seeds", ()))
            != expected_seeds
            or int(row.get("queries_per_seed", -1)) != expected_count
            or int(row.get("evaluations", -1)) != 300
            or str(row.get("mujoco_gl", "")) != expected_gl
            or str(row.get("environment_inputs", "")) != environment
            or str(row.get("runner", "")) not in implementation
            or str(row.get("runtime", "")) not in _mapping(
                prereg.get("runtimes"), "runtimes"
            )
        ):
            raise RuntimeError(f"CEM execution protocol drifted for {(environment, family)}")
        checkpoint = _mapping(row.get("checkpoint"), f"{checkpoint_id}.checkpoint")
        if not _same_identity(
            checkpoint,
            _mapping(registry_rows[checkpoint_id].get("weights"), "registry.weights"),
            root=root,
        ):
            raise RuntimeError(f"CEM checkpoint identity differs from registry: {checkpoint_id}")
        identities[f"execution.{environment}.{family}.checkpoint"] = verify_identity(
            checkpoint,
            root=root,
            label=f"execution.{environment}.{family}.checkpoint",
            hash_large_files=hash_large_files,
        )
        identities[f"execution.{environment}.{family}.config"] = verify_identity(
            _mapping(
                row.get("effective_loader_config"),
                f"{checkpoint_id}.effective_loader_config",
            ),
            root=root,
            label=f"execution.{environment}.{family}.config",
            hash_large_files=hash_large_files,
        )
        directory = _resolve(str(row.get("output_directory", "")), root=root)
        try:
            directory.relative_to(output_root)
        except ValueError as error:
            raise RuntimeError("CEM output escaped the frozen output root") from error
        files = tuple(str(value) for value in row.get("output_files", ()))
        expected_files = (
            tuple(f"seed{seed}.json" for seed in expected_seeds)
            if row.get("output_kind") == "six_seed_receipts"
            else ("aggregate.json",)
        )
        if files != expected_files:
            raise RuntimeError(f"CEM output file contract drifted for {(environment, family)}")
        for name in files:
            path = directory / name
            if path in observed_outputs:
                raise RuntimeError(f"Duplicate CEM output path: {path}")
            observed_outputs.add(path)

    audit_path = Path(identities["input_identity_audit"]["path"])
    audit = yaml.safe_load(audit_path.read_text(encoding="utf-8"))
    if audit.get("audit_id") != "contextworld_original_baseline_cem_input_identity_audit_v1":
        raise RuntimeError("Unexpected CEM input identity audit")
    for environment, values in _mapping(
        prereg.get("frozen_environment_inputs"), "frozen_environment_inputs"
    ).items():
        audit_row = _mapping(audit.get("datasets", {}).get(environment), "audit.dataset")
        if audit_row.get("content_hash_checked") is not True or not _same_identity(
            audit_row, _mapping(values.get("dataset"), "dataset"), root=root
        ):
            raise RuntimeError(f"Dataset audit identity drifted for {environment}")

    if require_outputs_absent and output_root.exists():
        raise FileExistsError(output_root)
    prereg["_config_path"] = str(config_path)
    prereg["_config_sha256"] = file_sha256(config_path)
    prereg["_identity_audit"] = identities
    prereg["_output_root"] = str(output_root)
    return prereg
