from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Iterable

from contextworld.evaluation.icl_model import file_sha256
from contextworld.paths import artifact_path, resolve_contextworld_path


GATE_SCHEMA_VERSION = 1
UNLOCKED_STATUS = "unlocked_after_validation_freeze"
VALIDATION_FREEZE_STATUS = "validation_frozen"
COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
EVIDENCE_ROLES = (
    "prediction",
    "fixed_candidate",
    "closed_loop_planning",
    "original_ability",
)
IMPLEMENTATION_SCOPE_PREFIXES = ("contextworld", "scripts", "configs")


class SealedTestGateError(RuntimeError):
    """Raised before any sealed-Test catalog or payload may be accessed."""


def _same_path(left: Any, right: Path) -> bool:
    try:
        return Path(str(left)).expanduser().resolve() == right.resolve()
    except (OSError, TypeError, ValueError):
        return False


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise SealedTestGateError(f"Missing {label}: {path}") from error
    except (OSError, json.JSONDecodeError) as error:
        raise SealedTestGateError(f"Cannot read {label}: {path}: {error}") from error
    if not isinstance(payload, dict):
        raise SealedTestGateError(f"{label} must be a JSON object: {path}")
    return payload


def _required(mapping: dict[str, Any], key: str, *, where: str) -> Any:
    if key not in mapping:
        raise SealedTestGateError(f"Missing {key!r} in {where}")
    return mapping[key]


def _resolve(value: Any, *, repo_root: Path) -> Path:
    return resolve_contextworld_path(str(value), repo_root=repo_root)


def canonical_door_benchmark_root(
    config: dict[str, Any], *, repo_root: Path
) -> Path:
    benchmark = str(_required(config, "benchmark", where="door config"))
    if not benchmark.startswith("tworoom_"):
        raise SealedTestGateError(f"Unexpected door benchmark name: {benchmark}")
    return artifact_path(
        "evaluation",
        "history3",
        benchmark.removeprefix("tworoom_"),
        repo_root=repo_root,
    ).resolve()


def canonical_door_split_root(
    config: dict[str, Any], *, split: str, repo_root: Path
) -> Path:
    root = canonical_door_benchmark_root(config, repo_root=repo_root)
    if split == "validation":
        return root
    if split == "sealed_test":
        return root / "sealed_test"
    raise ValueError(f"Unknown door evaluation split: {split}")


def canonical_door_planning_catalog(
    config: dict[str, Any], *, split: str, repo_root: Path
) -> Path:
    return (
        canonical_door_benchmark_root(config, repo_root=repo_root)
        / "planning"
        / split
        / "catalog.json"
    )


def require_canonical_split_path(
    value: Path | None,
    *,
    canonical: Path,
    split: str,
    label: str,
) -> Path:
    """Reject path-based split masquerading before the path is accessed.

    Formal Door artifacts have one canonical location per split.  Accepting an
    arbitrary input override would let a sealed-Test catalog (or a byte-for-byte
    copy) be opened while the caller claims ``--split validation`` and only be
    rejected after its metadata had already been read.
    """

    expected = canonical.expanduser().resolve()
    observed = expected if value is None else value.expanduser().resolve()
    if observed != expected:
        raise SealedTestGateError(
            f"Non-canonical {label} is forbidden for {split}: "
            f"observed={observed}, expected={expected}"
        )
    return expected


def require_path_within_split_root(
    value: Path,
    *,
    split_root: Path,
    split: str,
    label: str,
) -> Path:
    observed = value.expanduser().resolve()
    root = split_root.expanduser().resolve()
    try:
        observed.relative_to(root)
    except ValueError as error:
        raise SealedTestGateError(
            f"{label} is outside the canonical {split} artifact root: "
            f"observed={observed}, root={root}"
        ) from error
    return observed


def _hash_rows(rows: Iterable[dict[str, str]]) -> str:
    normalized = sorted(
        ({"path": str(row["path"]), "sha256": str(row["sha256"])} for row in rows),
        key=lambda row: row["path"],
    )
    encoded = json.dumps(
        normalized, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validation_result_manifest_sha256(
    rows: Iterable[dict[str, Any]], *, repo_root: Path, verify_files: bool = True
) -> tuple[str, list[dict[str, str]]]:
    normalized = []
    seen: set[str] = set()
    for row in rows:
        path = _resolve(_required(row, "path", where="Validation result row"), repo_root=repo_root)
        digest = str(_required(row, "sha256", where="Validation result row"))
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise SealedTestGateError(f"Invalid SHA256 for Validation result: {path}")
        key = str(path)
        if key in seen:
            raise SealedTestGateError(f"Repeated Validation result path: {path}")
        seen.add(key)
        if verify_files:
            if not path.is_file():
                raise SealedTestGateError(f"Missing Validation result: {path}")
            if file_sha256(path) != digest:
                raise SealedTestGateError(f"Validation result hash mismatch: {path}")
        normalized.append({"path": key, "sha256": digest})
    return _hash_rows(normalized), sorted(normalized, key=lambda row: row["path"])


def _ability_input_files(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for model in report.get("models", {}).values():
        rows.extend(model.get("planning_files", []))
        rollout = model.get("rollout_file")
        if rollout is not None:
            rows.append(rollout)
    return rows


def _report_input_files(report: dict[str, Any], role: str) -> list[dict[str, Any]]:
    rows = report.get("input_files")
    if rows is None and role == "original_ability":
        rows = _ability_input_files(report)
    if not isinstance(rows, list):
        raise SealedTestGateError(
            f"Validation {role} analysis report does not enumerate input result hashes"
        )
    return rows


def _report_checkpoint_hashes(report: dict[str, Any], role: str) -> set[str]:
    if role == "prediction":
        return {
            str(row["model"]["checkpoint_sha256"])
            for row in report.get("models", {}).values()
        }
    if role in ("fixed_candidate", "closed_loop_planning"):
        bindings = report.get("training_report_binding_audit", {}).get("bindings", {})
        return {str(row["checkpoint_sha256"]) for row in bindings.values()}
    bindings = report.get("training_report_bindings", {})
    return {str(row["checkpoint_sha256"]) for row in bindings.values()}


def _audit_analysis_report(
    report: dict[str, Any],
    *,
    role: str,
    config_hash: str,
    expected_checkpoint_hashes: set[str],
) -> None:
    if report.get("status") != "passed":
        raise SealedTestGateError(f"Validation {role} analysis is not formal/passed")
    if report.get("config", {}).get("sha256") != config_hash:
        raise SealedTestGateError(f"Validation {role} config hash mismatch")
    if role != "original_ability" and report.get("evaluation_split") != "validation":
        raise SealedTestGateError(f"Validation {role} report has the wrong split")
    if role in ("fixed_candidate", "closed_loop_planning"):
        if report.get("formal_analysis") is not True:
            raise SealedTestGateError(f"Validation {role} report is partial")
        matrix = report.get("model_matrix_audit", {})
        if matrix.get("complete_formal_matrix") is not True:
            raise SealedTestGateError(f"Validation {role} matrix is incomplete")
    elif role == "prediction":
        matrix = report.get("model_matrix_audit", {})
        if matrix.get("complete_formal_matrix") is not True:
            raise SealedTestGateError("Validation prediction matrix is incomplete")
        decision = report.get("decision", {})
        if decision.get("visible_geometry_generalization_validation_gate_passed") is not True:
            raise SealedTestGateError("Preregistered Validation prediction gate did not pass")
    else:
        if report.get("formal_analysis") is not True:
            raise SealedTestGateError("Original-ability report is partial")
        if report.get("matrix_audit", {}).get("complete_formal_matrix") is not True:
            raise SealedTestGateError("Original-ability matrix is incomplete")
    observed = _report_checkpoint_hashes(report, role)
    if observed != expected_checkpoint_hashes:
        raise SealedTestGateError(
            f"Validation {role} report is not bound to the seven frozen checkpoints"
        )


def _expected_models(config: dict[str, Any], *, repo_root: Path) -> list[dict[str, Any]]:
    protocol_path = _resolve(config["ability_retention"]["protocol"], repo_root=repo_root)
    import yaml

    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    originals = [
        row
        for row in protocol["models"]
        if list(row.get("training_groups", [])) == ["original"]
    ]
    if len(originals) != 1:
        raise SealedTestGateError("Ability protocol must declare one original reference")
    original_path = _resolve(originals[0]["checkpoint"], repo_root=repo_root)
    steps = int(config["training_protocol"]["optimizer_steps"])
    rows = []
    for group, model in config["models"].items():
        seeds = tuple(int(value) for value in model["required_training_seeds"])
        if group == "original_reference":
            if len(seeds) != 1:
                raise SealedTestGateError("Original reference must declare one seed")
            rows.append(
                {
                    "group": str(group),
                    "slug": original_path.parent.name,
                    "training_seed": seeds[0],
                    "path": original_path,
                }
            )
            continue
        synthetic = sorted(set(model["training_groups"]) - {"original"})
        if len(synthetic) != 1:
            raise SealedTestGateError(f"Cannot derive checkpoint slug for {group}")
        for seed in seeds:
            slug = f"h3_{synthetic[0]}_s{seed}"
            rows.append(
                {
                    "group": str(group),
                    "slug": slug,
                    "training_seed": seed,
                    "path": resolve_contextworld_path(
                        f"artifacts/training/runs/checkpoints/{slug}/"
                        f"weights_final_step_{steps}.pt",
                        repo_root=repo_root,
                    ),
                }
            )
    if len(rows) != 7:
        raise SealedTestGateError(f"Expected seven door checkpoints, found {len(rows)}")
    return rows


def _git(repo_root: Path, *arguments: str, capture: bool = False) -> bytes:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repo_root,
        stdout=subprocess.PIPE if capture else subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise SealedTestGateError(
            f"Git freeze verification failed: git {' '.join(arguments)}: {detail}"
        )
    return completed.stdout if capture else b""


def _verify_git_freeze(
    *,
    repo_root: Path,
    implementation_commit: str,
    freeze_commit: str,
    implementation_files: list[dict[str, str]],
    config_path: Path,
    config_hash: str,
    implementation_tree: str,
    validation_freeze_record_path: Path,
    validation_freeze_record_hash: str,
    gate_manifest_path: Path,
) -> None:
    for value, label in (
        (implementation_commit, "implementation commit"),
        (freeze_commit, "freeze commit"),
    ):
        if COMMIT_PATTERN.fullmatch(value) is None:
            raise SealedTestGateError(f"Invalid {label}: {value!r}")
        _git(repo_root, "cat-file", "-e", f"{value}^{{commit}}")
    observed_tree = _git(
        repo_root,
        "rev-parse",
        f"{implementation_commit}^{{tree}}",
        capture=True,
    ).decode("ascii", errors="strict").strip()
    if observed_tree != implementation_tree:
        raise SealedTestGateError(
            "Implementation commit tree does not match the frozen tree hash"
        )
    _git(repo_root, "merge-base", "--is-ancestor", implementation_commit, freeze_commit)
    _git(repo_root, "merge-base", "--is-ancestor", freeze_commit, "HEAD")
    for row in implementation_files:
        relative = Path(row["path"])
        committed = _git(
            repo_root,
            "show",
            f"{implementation_commit}:{relative.as_posix()}",
            capture=True,
        )
        if hashlib.sha256(committed).hexdigest() != row["sha256"]:
            raise SealedTestGateError(
                f"Implementation file is not bound to commit: {relative}"
            )
    try:
        config_relative = config_path.resolve().relative_to(repo_root.resolve())
    except ValueError as error:
        raise SealedTestGateError("Validation config must live in the repository") from error
    committed_config = _git(
        repo_root,
        "show",
        f"{implementation_commit}:{config_relative.as_posix()}",
        capture=True,
    )
    if hashlib.sha256(committed_config).hexdigest() != config_hash:
        raise SealedTestGateError("Validation config is not bound to implementation commit")

    def repository_relative(path: Path, *, label: str) -> str:
        try:
            return path.resolve().relative_to(repo_root.resolve()).as_posix()
        except ValueError as error:
            raise SealedTestGateError(
                f"{label} must live in the implementation repository"
            ) from error

    freeze_record = repository_relative(
        validation_freeze_record_path,
        label="Validation freeze record",
    )
    gate_manifest = repository_relative(
        gate_manifest_path,
        label="Sealed-Test gate manifest",
    )
    committed_record = _git(
        repo_root,
        "show",
        f"{freeze_commit}:{freeze_record}",
        capture=True,
    )
    if hashlib.sha256(committed_record).hexdigest() != validation_freeze_record_hash:
        raise SealedTestGateError(
            "Current Validation freeze record is not bound to the freeze commit"
        )
    if file_sha256(validation_freeze_record_path) != validation_freeze_record_hash:
        raise SealedTestGateError(
            "Current Validation freeze record differs from the gate-bound record"
        )
    committed_manifest = _git(
        repo_root,
        "show",
        f"HEAD:{gate_manifest}",
        capture=True,
    )
    if hashlib.sha256(committed_manifest).hexdigest() != file_sha256(gate_manifest_path):
        raise SealedTestGateError(
            "Current sealed-Test gate manifest is not committed at HEAD"
        )

    def changed_scope(base: str, target: str) -> set[str]:
        return {
            row
            for row in _git(
                repo_root,
                "diff",
                "--name-only",
                base,
                target,
                "--",
                *IMPLEMENTATION_SCOPE_PREFIXES,
                capture=True,
            )
            .decode("utf-8", errors="strict")
            .splitlines()
            if row
        }

    freeze_changes = changed_scope(implementation_commit, freeze_commit)
    if freeze_changes != {freeze_record}:
        raise SealedTestGateError(
            "Validation freeze commit must change only the freeze record: "
            f"{sorted(freeze_changes)}"
        )
    unlock_changes = changed_scope(freeze_commit, "HEAD")
    if unlock_changes != {gate_manifest}:
        raise SealedTestGateError(
            "Post-freeze unlock must change only the gate manifest: "
            f"{sorted(unlock_changes)}"
        )
    changed = changed_scope(implementation_commit, "HEAD")
    allowed_post_implementation_changes = {freeze_record, gate_manifest}
    if changed != allowed_post_implementation_changes:
        raise SealedTestGateError(
            "Implementation scope after freeze must differ only in the freeze record "
            f"and gate manifest: {sorted(changed)}"
        )
    worktree_changes = {
        row
        for row in _git(
            repo_root,
            "diff",
            "--name-only",
            "HEAD",
            "--",
            *IMPLEMENTATION_SCOPE_PREFIXES,
            capture=True,
        )
        .decode("utf-8", errors="strict")
        .splitlines()
        if row
    }
    if worktree_changes:
        raise SealedTestGateError(
            "Implementation dependency scope has uncommitted changes: "
            f"{sorted(worktree_changes)}"
        )
    untracked = {
        row
        for row in _git(
            repo_root,
            "ls-files",
            "--others",
            "--exclude-standard",
            "--",
            *IMPLEMENTATION_SCOPE_PREFIXES,
            capture=True,
        )
        .decode("utf-8", errors="strict")
        .splitlines()
        if row
    }
    if untracked:
        raise SealedTestGateError(
            "Untracked implementation dependencies are present at Test unlock: "
            f"{sorted(untracked)}"
        )


def validate_sealed_test_gate(
    *,
    config_path: Path,
    config: dict[str, Any],
    manifest_path: Path | None = None,
    repo_root: Path,
    verify_git: bool = True,
) -> dict[str, Any]:
    gate_config = _required(config, "sealed_test_gate", where="door config")
    configured_manifest = _resolve(gate_config["manifest"], repo_root=repo_root)
    manifest_path = (
        manifest_path.expanduser().resolve()
        if manifest_path is not None
        else configured_manifest
    )
    if manifest_path != configured_manifest:
        raise SealedTestGateError(
            "A sealed-Test gate override must resolve to the config-declared manifest"
        )
    manifest = _load_json(manifest_path, label="sealed-Test gate manifest")
    if int(manifest.get("schema_version", -1)) != GATE_SCHEMA_VERSION:
        raise SealedTestGateError("Unsupported sealed-Test gate schema version")
    if manifest.get("benchmark") != config.get("benchmark"):
        raise SealedTestGateError("Gate manifest benchmark mismatch")
    if manifest.get("status") != UNLOCKED_STATUS:
        raise SealedTestGateError(
            f"Sealed Test remains locked: status={manifest.get('status')!r}"
        )
    schema_path = _resolve(gate_config["schema"], repo_root=repo_root)
    if not schema_path.is_file():
        raise SealedTestGateError(f"Missing sealed-Test gate schema: {schema_path}")
    if not _same_path(manifest.get("schema"), schema_path):
        raise SealedTestGateError("Gate manifest schema path mismatch")
    schema = _load_json(schema_path, label="sealed-Test gate JSON schema")
    try:
        from jsonschema import Draft202012Validator
        from jsonschema.exceptions import SchemaError, ValidationError

        Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema).validate(manifest)
    except (SchemaError, ValidationError) as error:
        raise SealedTestGateError(
            f"Sealed-Test gate manifest failed Draft 2020-12 validation: {error.message}"
        ) from error

    configured_record = _resolve(
        _required(
            gate_config,
            "validation_freeze_record",
            where="door sealed-Test gate config",
        ),
        repo_root=repo_root,
    )
    record_pointer = _required(
        manifest,
        "validation_freeze_record",
        where="gate manifest",
    )
    if not _same_path(record_pointer.get("path"), configured_record):
        raise SealedTestGateError("Gate Validation freeze record path mismatch")
    record_hash = str(record_pointer.get("sha256", ""))
    if re.fullmatch(r"[0-9a-f]{64}", record_hash) is None:
        raise SealedTestGateError("Gate Validation freeze record SHA256 is invalid")
    if not configured_record.is_file() or file_sha256(configured_record) != record_hash:
        raise SealedTestGateError("Gate Validation freeze record hash mismatch")
    freeze_record = _load_json(
        configured_record,
        label="Validation freeze record",
    )
    record_schema = {
        "$schema": schema.get(
            "$schema",
            "https://json-schema.org/draft/2020-12/schema",
        ),
        "$defs": schema.get("$defs", {}),
        "$ref": "#/$defs/validation_freeze_record",
    }
    try:
        Draft202012Validator.check_schema(record_schema)
        Draft202012Validator(record_schema).validate(freeze_record)
    except (SchemaError, ValidationError) as error:
        raise SealedTestGateError(
            "Validation freeze record failed Draft 2020-12 validation: "
            f"{error.message}"
        ) from error
    if freeze_record.get("benchmark") != config.get("benchmark"):
        raise SealedTestGateError("Validation freeze record benchmark mismatch")
    if freeze_record.get("status") != VALIDATION_FREEZE_STATUS:
        raise SealedTestGateError(
            "Validation freeze record has not been finalized: "
            f"status={freeze_record.get('status')!r}"
        )

    config_path = config_path.expanduser().resolve()
    config_hash = file_sha256(config_path)
    validation_config = _required(
        freeze_record,
        "validation_config",
        where="Validation freeze record",
    )
    if not _same_path(validation_config.get("path"), config_path):
        raise SealedTestGateError("Gate Validation config path mismatch")
    if validation_config.get("sha256") != config_hash:
        raise SealedTestGateError("Gate Validation config hash mismatch")

    implementation = _required(
        freeze_record,
        "implementation",
        where="Validation freeze record",
    )
    implementation_tree = str(
        _required(implementation, "tree", where="gate implementation")
    )
    if COMMIT_PATTERN.fullmatch(implementation_tree) is None:
        raise SealedTestGateError(
            f"Invalid implementation tree hash: {implementation_tree!r}"
        )
    required_files = tuple(str(value) for value in gate_config["required_implementation_files"])
    observed_files = list(implementation.get("files", []))
    if {str(row.get("path")) for row in observed_files} != set(required_files):
        raise SealedTestGateError("Gate implementation file set is incomplete or changed")
    normalized_implementation = []
    for row in observed_files:
        relative = Path(str(row["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise SealedTestGateError(f"Invalid implementation path: {relative}")
        path = (repo_root / relative).resolve()
        digest = str(row.get("sha256"))
        if not path.is_file() or file_sha256(path) != digest:
            raise SealedTestGateError(f"Implementation file hash mismatch: {relative}")
        normalized_implementation.append({"path": relative.as_posix(), "sha256": digest})

    expected_models = _expected_models(config, repo_root=repo_root)
    checkpoint_rows = list(
        _required(
            freeze_record,
            "checkpoints",
            where="Validation freeze record",
        )
    )
    expected_by_key = {
        (row["group"], row["slug"], int(row["training_seed"])): row
        for row in expected_models
    }
    observed_by_key = {
        (str(row.get("group")), str(row.get("slug")), int(row.get("training_seed", -1))): row
        for row in checkpoint_rows
    }
    if set(observed_by_key) != set(expected_by_key) or len(checkpoint_rows) != 7:
        raise SealedTestGateError("Gate must bind the exact seven-model checkpoint matrix")
    checkpoint_hashes: set[str] = set()
    for key, expected in expected_by_key.items():
        row = observed_by_key[key]
        if not _same_path(row.get("path"), expected["path"]):
            raise SealedTestGateError(f"Checkpoint path mismatch for {key}")
        digest = str(row.get("sha256"))
        if not expected["path"].is_file() or file_sha256(expected["path"]) != digest:
            raise SealedTestGateError(f"Checkpoint hash mismatch for {key}")
        checkpoint_hashes.add(digest)
    if len(checkpoint_hashes) != 7:
        raise SealedTestGateError("The seven model labels must use unique checkpoint hashes")

    evidence_config = gate_config["validation_evidence"]
    evidence = _required(
        freeze_record,
        "validation_evidence",
        where="Validation freeze record",
    )
    if set(evidence) != set(EVIDENCE_ROLES):
        raise SealedTestGateError("Gate Validation evidence roles are incomplete")
    evidence_audit = {}
    for role in EVIDENCE_ROLES:
        row = evidence[role]
        report_row = _required(row, "analysis_report", where=f"gate evidence {role}")
        report_path = _resolve(report_row["path"], repo_root=repo_root)
        if not report_path.is_file() or file_sha256(report_path) != report_row.get("sha256"):
            raise SealedTestGateError(f"Validation {role} analysis report hash mismatch")
        report = _load_json(report_path, label=f"Validation {role} analysis report")
        _audit_analysis_report(
            report,
            role=role,
            config_hash=config_hash,
            expected_checkpoint_hashes=checkpoint_hashes,
        )
        input_rows = _report_input_files(report, role)
        expected_count = int(evidence_config[role]["expected_result_files"])
        if len(input_rows) != expected_count or int(row.get("result_file_count", -1)) != expected_count:
            raise SealedTestGateError(
                f"Validation {role} result set is incomplete: "
                f"expected {expected_count}, observed {len(input_rows)}"
            )
        result_digest, normalized = validation_result_manifest_sha256(
            input_rows, repo_root=repo_root, verify_files=True
        )
        if result_digest != row.get("result_files_manifest_sha256"):
            raise SealedTestGateError(f"Validation {role} result manifest hash mismatch")
        evidence_audit[role] = {
            "analysis_report": str(report_path),
            "analysis_report_sha256": str(report_row["sha256"]),
            "result_file_count": len(normalized),
            "result_files_manifest_sha256": result_digest,
        }

    gate_decision = _required(
        freeze_record,
        "preregistered_gate",
        where="Validation freeze record",
    )
    prediction_hash = str(evidence["prediction"]["analysis_report"]["sha256"])
    if (
        gate_decision.get("name")
        != "visible_geometry_generalization_validation_gate"
        or gate_decision.get("passed") is not True
        or gate_decision.get("source_analysis_report_sha256") != prediction_hash
    ):
        raise SealedTestGateError("Preregistered Validation gate decision is absent or changed")

    implementation_commit = str(implementation.get("commit"))
    freeze = _required(manifest, "freeze", where="gate manifest")
    freeze_commit = str(freeze.get("commit"))
    if freeze.get("immutable") is not True or not str(freeze.get("recorded_at_utc", "")):
        raise SealedTestGateError("Validation freeze commit was not immutably recorded")
    if verify_git:
        _verify_git_freeze(
            repo_root=repo_root,
            implementation_commit=implementation_commit,
            freeze_commit=freeze_commit,
            implementation_files=normalized_implementation,
            config_path=config_path,
            config_hash=config_hash,
            implementation_tree=implementation_tree,
            validation_freeze_record_path=configured_record,
            validation_freeze_record_hash=record_hash,
            gate_manifest_path=manifest_path,
        )
    else:
        if (
            COMMIT_PATTERN.fullmatch(implementation_commit) is None
            or COMMIT_PATTERN.fullmatch(implementation_tree) is None
            or COMMIT_PATTERN.fullmatch(freeze_commit) is None
        ):
            raise SealedTestGateError("Gate commits must be full 40-character SHA1 values")

    return {
        "required": True,
        "passed": True,
        "manifest": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "validation_freeze_record": str(configured_record),
        "validation_freeze_record_sha256": record_hash,
        "implementation_commit": implementation_commit,
        "implementation_tree": implementation_tree,
        "freeze_commit": freeze_commit,
        "config_sha256": config_hash,
        "checkpoint_count": len(checkpoint_rows),
        "evidence": evidence_audit,
        "preregistered_gate_passed": True,
    }


def require_sealed_test_gate(
    *,
    split: str,
    config_path: Path,
    config: dict[str, Any],
    manifest_path: Path | None = None,
    repo_root: Path,
) -> dict[str, Any]:
    if split == "validation":
        return {"required": False, "passed": True, "split": "validation"}
    if split != "sealed_test":
        raise ValueError(f"Unknown door evaluation split: {split}")
    return validate_sealed_test_gate(
        config_path=config_path,
        config=config,
        manifest_path=manifest_path,
        repo_root=repo_root,
        verify_git=True,
    )


__all__ = [
    "EVIDENCE_ROLES",
    "GATE_SCHEMA_VERSION",
    "SealedTestGateError",
    "UNLOCKED_STATUS",
    "canonical_door_benchmark_root",
    "canonical_door_planning_catalog",
    "canonical_door_split_root",
    "require_canonical_split_path",
    "require_path_within_split_root",
    "require_sealed_test_gate",
    "validate_sealed_test_gate",
    "validation_result_manifest_sha256",
]
