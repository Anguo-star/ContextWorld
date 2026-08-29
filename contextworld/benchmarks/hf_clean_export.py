"""Build a reader-facing Hugging Face staging tree from frozen suite data.

This exporter is intentionally separate from the historical suite exporter.
The historical export mixes benchmark payloads with internal evaluation
artifacts.  This module copies only explicitly registered Training,
Development and Test sources into a clean directory.  Test is public for
offline final reporting, but remains excluded from training dataset URIs.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import shutil
import stat
import tempfile
from pathlib import Path
from typing import Any, Iterable

import yaml


EXPECTED_COMPONENTS = (
    "speed",
    "door",
    "action_delay",
    "action_strength",
    "contact_friction",
    "motion_damping",
    "robot_arm_mass",
    "portal_exit",
    "cube_gripper_carry",
)
ALLOWED_SPLITS = {"training", "development", "test"}
PUBLIC_TEST_POLICY = "public_offline_final_reporting"
TEXT_SUFFIXES = {
    ".cfg",
    ".json",
    ".jsonl",
    ".md",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
SECRET_PATTERNS = (
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    re.compile(r"https://[^/\s:]+:[^@\s]+@github\.com"),
)
PRIVATE_PATH_MARKERS = ("/opt/huawei/", "/root/", "explorer-env/")
CATEGORY_LABELS = {
    "instantaneous_continuous_response": "Instantaneous continuous response",
    "temporally_delayed_dynamics": "Temporally delayed dynamics",
    "contact_or_attachment_conditioned_dynamics": (
        "Contact- and attachment-conditioned dynamics"
    ),
    "hidden_structural_transition_rules": (
        "Hidden structural transition rules"
    ),
}
NATIVE_SEQUENCE_SCHEMA = "native_episode_sequence_v1"
ALLOWED_SEQUENCE_SCHEMAS = {
    NATIVE_SEQUENCE_SCHEMA,
    "native_episode_sequence_with_step_metadata_v1",
    "blocked_transition_projection_v1",
}
DEVELOPMENT_EVALUATION_SCHEMA_VERSION = "contextworld.development_evaluation.v1"
DEVELOPMENT_EVALUATION_STATUS = "public_development_only"
PUBLIC_TEST_EVALUATION_SCHEMA_VERSION = (
    "contextworld.public_test_evaluation.v1"
)
PUBLIC_TEST_EVALUATION_STATUS = "public_final_reporting_only"
DEVELOPMENT_NORMALIZATION_TRANSFORM = "zscore"
DEVELOPMENT_NORMALIZATION_ESTIMATORS = {"population", "unbiased"}
TWOROOM_NORMALIZER_RELATIVE_PATH = "normalizers/tworoom_original_train_s3072.json"
TWOROOM_NORMALIZER = {
    "schema_version": 1,
    "protocol": "tworoom_original_train_s3072_unbiased_zscore_v1",
    "statistics_scope": "original_9000_train_episodes_only",
    # Keep the public source label portable.  In particular, do not preserve
    # the developer-machine absolute path from the historical artifact.
    "source": "quentinll/tworoom.h5",
    "source_sha256": "129a36aa93ea0de488d2bcc876e396de9e3907bf66c6aae6394e542ef6a6d623",
    "rows": 828678,
    "train_episode_ids_sha256": (
        "0250d70f46d9fcaa61b3d6627b9048f498669fe69e261f8440e80f1879458325"
    ),
    "columns": {
        "action": {
            "mean": [0.0031402341986976924, -0.051594576296864605],
            "std_unbiased": [0.867571689163936, 0.8688840167517821],
            "valid_rows": 819678,
        },
        "proprio": {
            "mean": [111.7950199284305, 85.03849594298646],
            "std_unbiased": [36.85458874773545, 38.17356572449523],
            "valid_rows": 828678,
        },
    },
}
TWOROOM_NORMALIZER_COMPONENTS = {"speed", "door", "action_delay"}


class CleanExportError(RuntimeError):
    """The requested export is unsafe, incomplete, or ambiguous."""


def _payload_id_for_target(target: str) -> str:
    """Return the stable public payload id for one registered target."""

    return Path(target).name.removesuffix(".lance")


def _validate_development_evaluation(
    component_id: str, component: dict[str, Any]
) -> None:
    """Validate the portable Development-only evaluator input contract.

    The clean bundle intentionally withholds Public Test.  This small contract
    tells a public evaluator exactly which already-distributed Development
    payload and action normalization it may use, without requiring an internal
    artifact tree or a release-config-relative file lookup.
    """

    evaluation = component.get("development_evaluation")
    if not isinstance(evaluation, dict):
        raise CleanExportError(
            f"Missing development_evaluation contract: {component_id}"
        )
    if evaluation.get("schema_version") != DEVELOPMENT_EVALUATION_SCHEMA_VERSION:
        raise CleanExportError(
            f"Invalid development_evaluation schema: {component_id}"
        )
    if evaluation.get("status") != DEVELOPMENT_EVALUATION_STATUS:
        raise CleanExportError(
            f"Development evaluation must be public-development-only: {component_id}"
        )
    if evaluation.get("split") != "development":
        raise CleanExportError(
            f"Development evaluation must select the development split: {component_id}"
        )
    payload_id = evaluation.get("payload_id")
    if not isinstance(payload_id, str) or not payload_id:
        raise CleanExportError(
            f"Development evaluation has no payload_id: {component_id}"
        )
    development_payload_ids = {
        _payload_id_for_target(str(row["target"]))
        for row in component["sources"]
        if row["split"] == "development"
    }
    if payload_id not in development_payload_ids:
        raise CleanExportError(
            f"Development evaluation payload is not a registered development "
            f"payload for {component_id}: {payload_id!r}"
        )
    normalizer_path = evaluation.get("normalizer_path")
    if component_id in TWOROOM_NORMALIZER_COMPONENTS:
        if normalizer_path != TWOROOM_NORMALIZER_RELATIVE_PATH:
            raise CleanExportError(
                f"TwoRoom Development evaluator has no portable normalizer: "
                f"{component_id}"
            )
    elif normalizer_path is not None:
        raise CleanExportError(
            f"Only the frozen TwoRoom adapters may name a normalizer file: "
            f"{component_id}"
        )
    if normalizer_path is not None:
        _relative_path(str(normalizer_path), field="normalizer_path")
    reader_id = evaluation.get("reader_id")
    if not isinstance(reader_id, str) or not reader_id:
        raise CleanExportError(
            f"Development evaluation has no reader_id: {component_id}"
        )
    selection = evaluation.get("selection")
    if not isinstance(selection, dict):
        raise CleanExportError(
            f"Development evaluation has no selection contract: {component_id}"
        )
    if not isinstance(selection.get("method"), str) or not selection["method"]:
        raise CleanExportError(
            f"Development selection has no method: {component_id}"
        )
    for field in ("selected_pair_count", "selected_case_count"):
        if field not in selection:
            continue
        value = selection[field]
        if not isinstance(value, int) or value < 1:
            raise CleanExportError(
                f"Invalid Development selection {field}: {component_id}"
            )
    expected = selection.get("expected_pair_count")
    selected = selection.get("selected_pair_count")
    if expected is not None and (
        not isinstance(expected, int)
        or expected < 1
        or not isinstance(selected, int)
        or selected > expected
    ):
        raise CleanExportError(
            f"Invalid Development expected/selected pair counts: {component_id}"
        )

    input_contract = evaluation.get("input_contract")
    if not isinstance(input_contract, dict):
        raise CleanExportError(
            f"Development evaluation has no input_contract: {component_id}"
        )
    if input_contract.get("context_streams") != ["pixels", "actions"]:
        raise CleanExportError(
            f"Development evaluator must receive pixels and actions: {component_id}"
        )
    if input_contract.get("history_length") != component["history_length"]:
        raise CleanExportError(
            f"Development history length disagrees with component: {component_id}"
        )
    if input_contract.get("action_block_raw_steps") != component["frameskip"]:
        raise CleanExportError(
            f"Development action block disagrees with component: {component_id}"
        )
    horizon = input_contract.get("prediction_horizon_action_blocks")
    if horizon not in component["prediction_horizons_action_blocks"]:
        raise CleanExportError(
            f"Development prediction horizon is not registered: {component_id}"
        )

    normalization = evaluation.get("action_normalization")
    if not isinstance(normalization, dict):
        raise CleanExportError(
            f"Development evaluation has no action_normalization: {component_id}"
        )
    if normalization.get("transform") != DEVELOPMENT_NORMALIZATION_TRANSFORM:
        raise CleanExportError(
            f"Unsupported Development action normalization: {component_id}"
        )
    if not isinstance(normalization.get("source"), str) or not normalization["source"]:
        raise CleanExportError(
            f"Development normalization has no source label: {component_id}"
        )
    if normalization.get("std_estimator") not in DEVELOPMENT_NORMALIZATION_ESTIMATORS:
        raise CleanExportError(
            f"Invalid Development normalization estimator: {component_id}"
        )
    for field, predicate in (
        ("mean", lambda value: math.isfinite(value)),
        ("std", lambda value: math.isfinite(value) and value > 0.0),
    ):
        values = normalization.get(field)
        if (
            not isinstance(values, list)
            or len(values) != component["action_dimension"]
            or any(
                not isinstance(value, (int, float)) or not predicate(float(value))
                for value in values
            )
        ):
            raise CleanExportError(
                f"Invalid Development normalization {field}: {component_id}"
            )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True)
        + "\n"
    ).encode("utf-8")


def _json_line(payload: Any) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _replace_bytes(path: Path, payload: bytes) -> None:
    """Replace one metadata file without touching sibling dataset payloads."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.refresh-", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _relative_path(value: str, *, field: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise CleanExportError(f"{field} must be a safe relative path: {value}")
    return path


def _validate_public_test_evaluation(
    component_id: str, component: dict[str, Any]
) -> None:
    evaluation = component.get("public_test_evaluation")
    if not isinstance(evaluation, dict):
        raise CleanExportError(
            f"Missing public_test_evaluation contract: {component_id}"
        )
    if (
        evaluation.get("schema_version")
        != PUBLIC_TEST_EVALUATION_SCHEMA_VERSION
        or evaluation.get("status") != PUBLIC_TEST_EVALUATION_STATUS
        or evaluation.get("split") != "test"
    ):
        raise CleanExportError(
            f"Invalid public_test_evaluation contract: {component_id}"
        )
    artifact_root = _relative_path(
        str(evaluation.get("artifact_root", "")), field="artifact_root"
    )
    if not artifact_root.parts or artifact_root.parts[0] != "artifacts":
        raise CleanExportError(
            f"Public Test artifact_root must use artifacts/: {component_id}"
        )
    test_rows = [
        row for row in component["sources"] if row.get("split") == "test"
    ]
    if not test_rows:
        raise CleanExportError(f"No Test source registered: {component_id}")
    for row in test_rows:
        target = Path(row["target"])
        if target != artifact_root and artifact_root not in target.parents:
            raise CleanExportError(
                f"Test source is outside artifact_root: {component_id}"
            )


def _excluded_prefixes(row: dict[str, Any]) -> tuple[Path, ...]:
    values = row.get("exclude", [])
    if not isinstance(values, list) or any(
        not isinstance(value, str) or not value for value in values
    ):
        raise CleanExportError("Source exclude must be a list of relative paths")
    prefixes = tuple(_relative_path(value, field="exclude") for value in values)
    if len(prefixes) != len(set(prefixes)):
        raise CleanExportError("Source exclude paths must be unique")
    return prefixes


def load_export_contract(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise CleanExportError("HF export contract must be a mapping")
    if payload.get("public_test_policy") != PUBLIC_TEST_POLICY:
        raise CleanExportError(
            "HF clean export must publish Test for offline final reporting"
        )
    components = payload.get("components")
    if not isinstance(components, dict):
        raise CleanExportError("HF export contract has no components mapping")
    if tuple(components) != EXPECTED_COMPONENTS:
        raise CleanExportError(
            "HF export contract must list the nine canonical components in "
            f"order: {EXPECTED_COMPONENTS}"
        )

    targets: set[Path] = set()
    for component_id, component in components.items():
        if not isinstance(component, dict):
            raise CleanExportError(f"Invalid component: {component_id}")
        dataset_id = component.get("dataset_id")
        if not isinstance(dataset_id, str) or not dataset_id:
            raise CleanExportError(f"Missing dataset_id: {component_id}")
        for field in ("environment", "capability_category", "description"):
            if not isinstance(component.get(field), str) or not component[field]:
                raise CleanExportError(f"Missing {field}: {component_id}")
        if component["capability_category"] not in CATEGORY_LABELS:
            raise CleanExportError(
                f"Unknown capability_category: {component_id}"
            )
        for field in ("history_length", "action_dimension", "frameskip"):
            if not isinstance(component.get(field), int) or component[field] < 1:
                raise CleanExportError(f"Invalid {field}: {component_id}")
        horizons = component.get("prediction_horizons_action_blocks")
        if (
            not isinstance(horizons, list)
            or not horizons
            or any(not isinstance(value, int) or value < 1 for value in horizons)
        ):
            raise CleanExportError(
                f"Invalid prediction_horizons_action_blocks: {component_id}"
            )
        stablewm = component.get("stable_worldmodel")
        if not isinstance(stablewm, dict):
            raise CleanExportError(
                f"Missing stable_worldmodel contract: {component_id}"
            )
        sequence_schema = stablewm.get("sequence_schema")
        if sequence_schema not in ALLOWED_SEQUENCE_SCHEMAS:
            raise CleanExportError(
                f"Invalid Stable-WorldModel sequence schema: {component_id}"
            )
        required_columns = stablewm.get("required_columns")
        if (
            not isinstance(required_columns, list)
            or not required_columns
            or any(not isinstance(name, str) or not name for name in required_columns)
            or len(required_columns) != len(set(required_columns))
        ):
            raise CleanExportError(
                f"Invalid Stable-WorldModel required columns: {component_id}"
            )
        adapter = stablewm.get("adapter_required")
        if sequence_schema == NATIVE_SEQUENCE_SCHEMA and adapter is not None:
            raise CleanExportError(
                f"Native sequence must not require an adapter: {component_id}"
            )
        if sequence_schema != NATIVE_SEQUENCE_SCHEMA and (
            not isinstance(adapter, str) or not adapter
        ):
            raise CleanExportError(
                f"Non-native sequence must name its adapter: {component_id}"
            )
        _relative_path(str(component.get("release_config", "")), field="release_config")
        sources = component.get("sources")
        if not isinstance(sources, list) or not sources:
            raise CleanExportError(f"No sources registered: {component_id}")
        for row in sources:
            if not isinstance(row, dict) or row.get("split") not in ALLOWED_SPLITS:
                raise CleanExportError(f"Invalid split mapping: {component_id}")
            source = _relative_path(str(row.get("source", "")), field="source")
            target = _relative_path(str(row.get("target", "")), field="target")
            if row["split"] != "test" and source.parts[0] != "synthesis":
                raise CleanExportError(
                    f"Training/Development sources must use synthesis/: {source}"
                )
            if row["split"] == "test" and source.parts[0] not in {
                "synthesis",
                "evaluation",
            }:
                raise CleanExportError(
                    f"Test source must use synthesis/ or evaluation/: {source}"
                )
            if row["split"] != "test" and target.parts[:2] != (
                "components",
                dataset_id,
            ):
                raise CleanExportError(
                    f"Target is outside component namespace: {target}"
                )
            if row["split"] == "test" and (
                not target.parts or target.parts[0] != "artifacts"
            ):
                raise CleanExportError(
                    f"Test target must preserve artifacts/ namespace: {target}"
                )
            excluded = _excluded_prefixes(row)
            if excluded and row["split"] != "test":
                raise CleanExportError(
                    f"Only Test mappings may exclude internal paths: {component_id}"
                )
            if target in targets:
                raise CleanExportError(f"Duplicate export target: {target}")
            targets.add(target)
        _validate_development_evaluation(component_id, component)
        _validate_public_test_evaluation(component_id, component)
    return payload


def _is_excluded(relative: Path, prefixes: tuple[Path, ...]) -> bool:
    return any(
        relative == prefix or prefix in relative.parents for prefix in prefixes
    )


def _source_files(
    source: Path, *, exclude: tuple[Path, ...] = ()
) -> Iterable[tuple[Path, Path]]:
    if source.is_symlink():
        raise CleanExportError(f"Symlink source is forbidden: {source}")
    if source.is_file():
        relative = Path(source.name)
        if not _is_excluded(relative, exclude):
            yield source, relative
        return
    if not source.is_dir():
        raise CleanExportError(f"Registered source does not exist: {source}")

    for directory, names, files in os.walk(source, followlinks=False):
        directory_path = Path(directory)
        names.sort()
        files.sort()
        retained_names = []
        for name in names:
            candidate = directory_path / name
            if candidate.is_symlink():
                raise CleanExportError(f"Symlink directory is forbidden: {candidate}")
            relative = candidate.relative_to(source)
            if not _is_excluded(relative, exclude):
                retained_names.append(name)
        names[:] = retained_names
        for name in files:
            candidate = directory_path / name
            mode = candidate.lstat().st_mode
            if stat.S_ISLNK(mode):
                raise CleanExportError(f"Symlink file is forbidden: {candidate}")
            if not stat.S_ISREG(mode):
                raise CleanExportError(f"Non-regular file is forbidden: {candidate}")
            relative = candidate.relative_to(source)
            if not _is_excluded(relative, exclude):
                yield candidate, relative


def _scan_text(path: Path) -> None:
    if path.suffix.lower() not in TEXT_SUFFIXES or path.stat().st_size > 2_000_000:
        return
    text = path.read_text(encoding="utf-8", errors="ignore")
    for pattern in SECRET_PATTERNS:
        if pattern.search(text):
            raise CleanExportError(f"Credential-like text found in {path.name}")
    if any(marker in text for marker in PRIVATE_PATH_MARKERS):
        raise CleanExportError(f"Private absolute path found in {path.name}")


def _is_lance_table(path: Path) -> bool:
    return (
        path.is_dir()
        and path.name.lower().endswith(".lance")
        and (path / "_versions").is_dir()
    )


def _source_inventory(
    source: Path, *, exclude: tuple[Path, ...] = ()
) -> dict[str, Any]:
    """Enumerate and classify a registered payload without copying it."""

    files = list(_source_files(source, exclude=exclude))
    for source_file, _ in files:
        _scan_text(source_file)

    if _is_lance_table(source):
        lance_tables = [source]
    elif source.is_dir():
        lance_tables = sorted(
            path
            for path in source.rglob("*.lance")
            if _is_lance_table(path)
        )
    else:
        lance_tables = []

    entrypoint_member: str | None = None
    if len(lance_tables) == 1 and lance_tables[0] == source:
        payload_kind = "single_lance"
        single_dataset_entrypoint = True
        entrypoint_member = "."
    elif (
        len(lance_tables) == 1
        and lance_tables[0].parent == source
    ):
        payload_kind = "single_lance_container"
        single_dataset_entrypoint = True
        entrypoint_member = lance_tables[0].relative_to(source).as_posix()
    elif lance_tables and all(path.parent == source for path in lance_tables):
        payload_kind = "lance_collection"
        single_dataset_entrypoint = False
    elif lance_tables:
        payload_kind = "nested_lance_collection"
        single_dataset_entrypoint = False
    elif source.is_file() and source.suffix.lower() in {".h5", ".hdf5"}:
        payload_kind = "single_hdf5"
        single_dataset_entrypoint = True
        entrypoint_member = source.name
    elif source.is_file():
        payload_kind = "single_file"
        single_dataset_entrypoint = False
    else:
        payload_kind = "directory"
        single_dataset_entrypoint = False

    members = [
        "." if path == source else path.relative_to(source).as_posix()
        for path in lance_tables
    ]
    return {
        "format": "lance" if lance_tables else "unknown",
        "payload_kind": payload_kind,
        "file_count": len(files),
        "total_bytes": sum(path.stat().st_size for path, _ in files),
        "lance_table_count": len(lance_tables),
        "single_dataset_entrypoint": single_dataset_entrypoint,
        "dataset_entrypoint_member": entrypoint_member,
        "members": members,
    }


def _manifest_row(
    path: Path,
    root: Path,
    *,
    role: str,
    component: str | None = None,
    split: str | None = None,
    source: str | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "role": role,
    }
    if component is not None:
        row["component"] = component
    if split is not None:
        row["split"] = split
    if source is not None:
        row["source_logical_path"] = source
    return row


def _component_card(component_id: str, component: dict[str, Any]) -> str:
    dataset_id = component["dataset_id"]
    rows = []
    direct_paths = []
    has_collection = False
    adapters = set()
    sequence_schemas = set()
    for payload in component["payloads"]:
        direct = payload["direct_stable_worldmodel_load"]
        native = (
            payload["stable_worldmodel_sequence_schema"]
            == NATIVE_SEQUENCE_SCHEMA
        )
        rows.append(
            f"| {payload['split']} | `{payload['payload_id']}` | "
            f"`{payload['public_path']}` | `{payload['payload_kind']}` | "
            f"{payload['lance_table_count']} | {'yes' if native else 'no'} | "
            f"{'yes' if direct else 'no'} |"
        )
        if direct:
            direct_paths.append(payload["cw_dataset_entrypoint"])
        if not payload["single_dataset_entrypoint"]:
            has_collection = True
        if payload["stable_worldmodel_adapter_required"]:
            adapters.add(payload["stable_worldmodel_adapter_required"])
        sequence_schemas.add(payload["stable_worldmodel_sequence_schema"])
    loading = "\n".join(
        f"- `CW_DATASET=<clean-root>/{path}`" for path in direct_paths
    )
    if not loading:
        loading = "- None for the payloads in this staging package."
    collection_note = ""
    if has_collection:
        collection_note = """

Collection payloads contain multiple Lance tables. Do not pass their split
root directly as `CW_DATASET`: Stable-WorldModel cannot infer one table from
that path. Use the component training recipe, or select/compose the registered
members in `task_registry.json` according to the linked release config.
"""
    adapter_note = ""
    if adapters:
        adapter_list = ", ".join(f"`{name}`" for name in sorted(adapters))
        if sequence_schemas == {
            "native_episode_sequence_with_step_metadata_v1"
        }:
            adapter_note = f"""

The temporal columns follow the native episode sequence layout, but this table
also stores string-valued metadata on every step. The pinned public
Stable-WorldModel reader rejects that legacy layout before it selects model
input columns. Move the constant metadata into the episode side table with
the adapter {adapter_list} before using this payload as `CW_DATASET`.
"""
        else:
            adapter_note = f"""

The distributed payload is a benchmark-specific blocked-transition
projection, not a conventional per-step training sequence. It can be opened
as a Lance table, but it must not be passed directly to the standard
Stable-WorldModel trainer. It requires the audited adapter {adapter_list},
which is not part of this staging package.
"""
    return f"""# {dataset_id}

{component['description']}

- Environment: {component['environment']}
- Capability type: {CATEGORY_LABELS[component['capability_category']]}
- History tokens: {component['history_length']}
- Action dimension: {component['action_dimension']}
- Frameskip: {component['frameskip']}
- Prediction horizons (action blocks): {component['prediction_horizons_action_blocks']}

Available splits:

- `training/`: model fitting.
- `development/`: recipe selection and diagnostics.
- `test`: final reporting only; never a training input.

| Split | Payload | Public path | Layout | Lance tables | Native sequence | Direct `CW_DATASET` |
|---|---|---|---|---:|---|---|
{chr(10).join(rows)}

Direct Stable-WorldModel entry points:

{loading}
{collection_note}
{adapter_note}

Test examples are public for reproducible offline final reporting. Use
Development for method and recipe selection, and evaluate Test only after the
choice is fixed. Test payloads are deliberately not direct `CW_DATASET`
entries. This component is part of a staging export, not a hosted leaderboard.
"""


def _root_readme(components: list[dict[str, Any]]) -> str:
    rows = "\n".join(
        f"| `{row['dataset_id']}` | "
        f"{CATEGORY_LABELS[row['capability_category']]} | "
        f"{row['environment']} | {row['history_length']} | "
        f"{row['action_dimension']} |"
        for row in components
    )
    return f"""# ContextWorld dataset

ContextWorld tests whether a latent world model can infer hidden dynamics from
recent interaction history without updating its parameters at evaluation
time. This package contains the Training, Development and Test data for nine
frozen benchmark components. Test is an offline final-reporting split.

This is a staging export, not a public release. Creating it does not by itself
authorize public distribution or benchmark submission.

Test examples and frozen scorers are public. This package does not contain
model checkpoints, training logs, experiment trackers, third-party source
checkouts, or the original LeWM datasets. Development is the only split for
method and recipe selection; Test is for final reporting and must not be fed
back into development. Because no hosted submission service is claimed,
offline Test results are reproducible but not centrally tamper-resistant.

| Component | Capability type | Environment | History | Action dimension |
|---|---|---:|---:|---:|
{rows}

`task_registry.json` is the canonical machine-readable index.
`manifest.jsonl` records every distributed file's size and SHA-256 digest.
Training and Development live below the component directories. Test retains
the frozen `artifacts/...` logical paths required by the existing task
scorers. Internal names are recorded in `task_registry.json` and
`manifest.jsonl`; internal model score receipts are not distributed.

The data contract is model-agnostic. LeWM, PLDM and PreJEPA are the reference
Stable-WorldModel integrations shipped by ContextWorld, not an allow-list.
Other models may consume the registered Training payloads through their own
loader and implement `LatentWorldModelAdapter` for Development evaluation.
Fields prefixed `stable_worldmodel_` describe only the bundled reference
loader's physical-data requirements.

`single_dataset_entrypoint` describes physical layout only. A payload is an
exact `CW_DATASET` training input only when
`direct_stable_worldmodel_load` is also true. Collection payloads need an
explicit member-selection or composition recipe. Five single-table payloads
have native temporal columns but must first move per-step string metadata into
an episode side table for the pinned public loader. The Cube
blocked-transition projection needs a separate raw-sequence adapter. Read the
component card and registry before launching a model.
"""


def _development_evaluation_record(
    component: dict[str, Any], payloads: list[dict[str, Any]]
) -> dict[str, Any]:
    """Attach public payload paths to the portable Development contract."""

    evaluation = copy.deepcopy(component["development_evaluation"])
    matches = [
        payload
        for payload in payloads
        if payload["split"] == evaluation["split"]
        and payload["payload_id"] == evaluation["payload_id"]
    ]
    if len(matches) != 1:
        raise CleanExportError(
            "Development evaluation did not resolve to exactly one payload: "
            f"{component['dataset_id']}"
        )
    payload = matches[0]
    # The registry remains self-contained: an evaluator can resolve its
    # Development input only from this release, without interpreting internal
    # source paths or consulting a private artifact root.
    evaluation["payload"] = {
        "public_path": payload["public_path"],
        "payload_kind": payload["payload_kind"],
        "lance_table_count": payload["lance_table_count"],
        "members": list(payload["members"]),
    }
    return evaluation


def _public_test_evaluation_record(
    component: dict[str, Any], payloads: list[dict[str, Any]]
) -> dict[str, Any]:
    """Attach public Test payload paths to the final-reporting contract."""

    evaluation = copy.deepcopy(component["public_test_evaluation"])
    matches = [payload for payload in payloads if payload["split"] == "test"]
    if not matches:
        raise CleanExportError(
            "Public Test evaluation has no exported payload: "
            f"{component['dataset_id']}"
        )
    evaluation["payloads"] = [
        {
            "public_path": payload["public_path"],
            "payload_kind": payload["payload_kind"],
            "lance_table_count": payload["lance_table_count"],
            "members": list(payload["members"]),
        }
        for payload in matches
    ]
    evaluation["selection_policy"] = (
        "development_only_model_selection_test_final_reporting"
    )
    evaluation["official_scoreboard_row"] = False
    return evaluation


def _build_export_plan_from_registered_paths(
    *,
    contract_path: Path,
    data_root: Path,
    repo_root: Path,
    path_field: str,
) -> dict[str, Any]:
    contract_path = contract_path.resolve()
    data_root = data_root.resolve()
    repo_root = repo_root.resolve()
    contract = load_export_contract(contract_path)
    if path_field not in {"source", "target"}:
        raise ValueError(f"Unsupported registered path field: {path_field}")
    if not data_root.is_dir():
        raise CleanExportError(f"Export data root not found: {data_root}")

    components: list[dict[str, Any]] = []
    total_files = 0
    total_bytes = 0
    for component_id, component in contract["components"].items():
        release_path = repo_root / component["release_config"]
        if not release_path.is_file():
            raise CleanExportError(f"Release config not found: {release_path}")
        source_rows = []
        for row in component["sources"]:
            source = data_root / row[path_field]
            if source.is_symlink() or not source.exists():
                raise CleanExportError(
                    f"Registered {path_field} not found: {row[path_field]}"
                )
            excluded = _excluded_prefixes(row)
            inventory = _source_inventory(source, exclude=excluded)
            target = Path(row["target"])
            entrypoint_member = inventory.pop("dataset_entrypoint_member")
            public_members = [
                target.as_posix()
                if member == "."
                else (target / member).as_posix()
                for member in inventory.pop("members")
            ]
            stablewm = component["stable_worldmodel"]
            native_sequence = (
                stablewm["sequence_schema"] == NATIVE_SEQUENCE_SCHEMA
            )
            direct = (
                row["split"] != "test"
                and inventory["single_dataset_entrypoint"]
                and native_sequence
            )
            if entrypoint_member is None:
                single_dataset_path = None
            elif entrypoint_member == ".":
                single_dataset_path = target.as_posix()
            else:
                single_dataset_path = (target / entrypoint_member).as_posix()
            total_files += inventory["file_count"]
            total_bytes += inventory["total_bytes"]
            source_rows.append(
                {
                    "split": row["split"],
                    "payload_id": _payload_id_for_target(target.as_posix()),
                    "public_path": target.as_posix(),
                    **inventory,
                    "single_dataset_path": single_dataset_path,
                    "stable_worldmodel_sequence_schema": stablewm[
                        "sequence_schema"
                    ],
                    "stable_worldmodel_required_columns": stablewm[
                        "required_columns"
                    ],
                    "stable_worldmodel_adapter_required": stablewm[
                        "adapter_required"
                    ],
                    "direct_stable_worldmodel_load": direct,
                    "cw_dataset_entrypoint": single_dataset_path if direct else None,
                    "members": public_members,
                    "provenance": {
                        "source_logical_path": row["source"],
                        "excluded_paths": [
                            path.as_posix() for path in excluded
                        ],
                    },
                }
            )
        components.append(
            {
                "component_id": component_id,
                "dataset_id": component["dataset_id"],
                "environment": component["environment"],
                "capability_category": component["capability_category"],
                "description": component["description"],
                "history_length": component["history_length"],
                "action_dimension": component["action_dimension"],
                "frameskip": component["frameskip"],
                "prediction_horizons_action_blocks": component[
                    "prediction_horizons_action_blocks"
                ],
                "release_config": component["release_config"],
                "release_config_sha256": _sha256(release_path),
                "payloads": source_rows,
                "development_evaluation": _development_evaluation_record(
                    component, source_rows
                ),
                "public_test_evaluation": _public_test_evaluation_record(
                    component, source_rows
                ),
            }
        )
    return {
        "export_id": contract["export_id"],
        "status": contract["status"],
        "public_test_policy": contract["public_test_policy"],
        "inventory": {
            "file_count": total_files,
            "total_bytes": total_bytes,
        },
        "components": components,
    }


def build_export_plan(
    *, contract_path: Path, suite_export_root: Path, repo_root: Path
) -> dict[str, Any]:
    return _build_export_plan_from_registered_paths(
        contract_path=contract_path,
        data_root=suite_export_root,
        repo_root=repo_root,
        path_field="source",
    )


def _build_staging_metadata_plan(
    *, contract_path: Path, output: Path, repo_root: Path
) -> dict[str, Any]:
    return _build_export_plan_from_registered_paths(
        contract_path=contract_path,
        data_root=output,
        repo_root=repo_root,
        path_field="target",
    )


def _generated_release_metadata(
    plan: dict[str, Any], contract_path: Path
) -> dict[str, bytes]:
    registry = {
        "schema_version": 1,
        "export_id": plan["export_id"],
        "release_status": "staging_not_public_release",
        "public_test": {
            "included": True,
            "policy": PUBLIC_TEST_POLICY,
            "evaluation_interface": "offline_final_reporting",
            "selection_policy": (
                "development_only_model_selection_test_final_reporting"
            ),
        },
        "components": plan["components"],
    }
    return {
        "README.md": _root_readme(plan["components"]).encode("utf-8"),
        "task_registry.json": _json_bytes(registry),
        TWOROOM_NORMALIZER_RELATIVE_PATH: _json_bytes(TWOROOM_NORMALIZER),
        "VERSION.json": _json_bytes(
            {
                "schema_version": 1,
                "dataset_version": "1.0.0-rc2",
                "release_status": "staging_not_public_release",
                "export_contract": contract_path.name,
                "export_contract_sha256": _sha256(contract_path),
                "public_test_included": True,
            }
        ),
    }


def export_hf_clean(
    *,
    contract_path: Path,
    suite_export_root: Path,
    output: Path,
    repo_root: Path,
    atomic_publish: bool = True,
) -> dict[str, Any]:
    """Copy an immutable clean staging tree and return its summary.

    Atomic publication is the default. ``atomic_publish=False`` exists for
    managed dataset mounts that allow creating files but forbid renaming a
    completed directory. The direct mode still refuses an existing target and
    removes its own incomplete target if copying fails.
    """

    output = output.expanduser().resolve()
    if output.exists():
        raise CleanExportError(f"Output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    plan = build_export_plan(
        contract_path=contract_path,
        suite_export_root=suite_export_root,
        repo_root=repo_root,
    )
    contract = load_export_contract(contract_path.resolve())
    suite_export_root = suite_export_root.resolve()
    repo_root = repo_root.resolve()
    if atomic_publish:
        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{output.name}.staging-", dir=output.parent
            )
        )
    else:
        output.mkdir(exist_ok=False)
        staging = output
    manifest: list[dict[str, Any]] = []
    planned_components = {
        row["component_id"]: row for row in plan["components"]
    }
    try:
        for component_id, component in contract["components"].items():
            for row in component["sources"]:
                source = suite_export_root / row["source"]
                target = staging / row["target"]
                target.mkdir(parents=True, exist_ok=False)
                excluded = _excluded_prefixes(row)
                for source_file, relative in _source_files(
                    source, exclude=excluded
                ):
                    destination = target / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source_file, destination, follow_symlinks=False)
                    _scan_text(destination)
                    manifest.append(
                        _manifest_row(
                            destination,
                            staging,
                            role="dataset_payload",
                            component=component_id,
                            split=row["split"],
                            source=(Path(row["source"]) / relative).as_posix(),
                        )
                    )

            card = staging / "components" / component["dataset_id"] / "v1" / "component_card.md"
            _write_bytes(
                card,
                _component_card(
                    component_id, planned_components[component_id]
                ).encode("utf-8"),
            )
            manifest.append(
                _manifest_row(
                    card,
                    staging,
                    role="component_documentation",
                    component=component_id,
                )
            )

        generated = _generated_release_metadata(plan, contract_path)
        for name, payload in generated.items():
            destination = staging / name
            _write_bytes(destination, payload)
            manifest.append(
                _manifest_row(destination, staging, role="release_metadata")
            )

        for name in ("LICENSE", "DATA_LICENSE", "NOTICE"):
            source = repo_root / name
            if not source.is_file():
                raise CleanExportError(f"Required legal file not found: {name}")
            destination = staging / name
            shutil.copy2(source, destination, follow_symlinks=False)
            _scan_text(destination)
            manifest.append(
                _manifest_row(destination, staging, role="legal_metadata")
            )

        manifest.sort(key=lambda row: row["path"])
        manifest_path = staging / "manifest.jsonl"
        _write_bytes(
            manifest_path,
            b"".join(_json_line(row) for row in manifest),
        )
        manifest_digest = _sha256(manifest_path)
        _write_bytes(
            staging / "manifest.sha256",
            f"{manifest_digest}  manifest.jsonl\n".encode("ascii"),
        )
        if atomic_publish:
            os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return {
        "status": "clean_staging_created",
        "output": str(output),
        "component_count": len(EXPECTED_COMPONENTS),
        "manifest_entries": len(manifest),
        "manifest_sha256": manifest_digest,
        "payload_file_count": plan["inventory"]["file_count"],
        "payload_bytes": plan["inventory"]["total_bytes"],
        "public_test_included": True,
    }


def refresh_hf_clean_metadata(
    *,
    contract_path: Path,
    output: Path,
    repo_root: Path,
) -> dict[str, Any]:
    """Refresh a clean staging tree's generated metadata without recopying data.

    The existing manifest must be internally signed and its payload rows must
    match the current export mapping byte-for-byte in path and size. Payload
    hashes are retained; only generated cards, registry, version, README and
    manifest files are replaced.
    """

    output = output.expanduser().resolve()
    contract_path = contract_path.resolve()
    repo_root = repo_root.resolve()
    if not output.is_dir():
        raise CleanExportError(f"Clean staging output does not exist: {output}")

    version_path = output / "VERSION.json"
    registry_path = output / "task_registry.json"
    manifest_path = output / "manifest.jsonl"
    digest_path = output / "manifest.sha256"
    for path in (version_path, registry_path, manifest_path, digest_path):
        if not path.is_file() or path.is_symlink():
            raise CleanExportError(f"Invalid clean staging metadata: {path}")
    version = json.loads(version_path.read_text(encoding="utf-8"))
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if version.get("release_status") != "staging_not_public_release":
        raise CleanExportError("Metadata refresh is restricted to staging exports")
    current_digest = _sha256(manifest_path)
    recorded_digest = digest_path.read_text(encoding="ascii").split()
    if not recorded_digest or recorded_digest[0] != current_digest:
        raise CleanExportError("Existing manifest digest does not match")

    plan = _build_staging_metadata_plan(
        contract_path=contract_path,
        output=output,
        repo_root=repo_root,
    )
    if registry.get("export_id") != plan["export_id"]:
        raise CleanExportError("Existing staging tree has a different export_id")
    contract = load_export_contract(contract_path)

    rows = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    payload_rows = {
        row["path"]: row for row in rows if row.get("role") == "dataset_payload"
    }
    matched_mappings: set[tuple[str, str]] = set()
    for public_path, row in payload_rows.items():
        component_id = row.get("component")
        component = contract["components"].get(component_id)
        if not isinstance(component, dict):
            raise CleanExportError(
                f"Unknown component in staged manifest: {component_id}"
            )
        candidates = []
        for mapping in component["sources"]:
            target = Path(mapping["target"])
            path = Path(public_path)
            if path == target or target in path.parents:
                candidates.append(mapping)
        if len(candidates) != 1:
            raise CleanExportError(
                f"Staged payload path has no unique contract mapping: {public_path}"
            )
        mapping = candidates[0]
        target = Path(mapping["target"])
        relative = Path(public_path).relative_to(target)
        expected = {
            "component": component_id,
            "split": mapping["split"],
            "source_logical_path": (
                Path(mapping["source"]) / relative
            ).as_posix(),
        }
        matched_mappings.add((component_id, mapping["target"]))
        row = payload_rows[public_path]
        actual = output / public_path
        if actual.is_symlink() or not actual.is_file():
            raise CleanExportError(f"Missing staged payload: {public_path}")
        for key, value in expected.items():
            if row.get(key) != value:
                raise CleanExportError(
                    f"Staged payload metadata changed for {public_path}: {key}"
                )
        if actual.stat().st_size != row.get("bytes"):
            raise CleanExportError(f"Staged payload size changed: {public_path}")
    expected_mappings = {
        (component_id, mapping["target"])
        for component_id, component in contract["components"].items()
        for mapping in component["sources"]
    }
    if matched_mappings != expected_mappings:
        raise CleanExportError(
            "Existing staged payloads do not cover every contract mapping"
        )
    if (
        len(payload_rows) != plan["inventory"]["file_count"]
        or sum(int(row["bytes"]) for row in payload_rows.values())
        != plan["inventory"]["total_bytes"]
    ):
        raise CleanExportError("Staged payload inventory does not match the manifest")

    metadata_payloads = _generated_release_metadata(plan, contract_path)
    for component in plan["components"]:
        relative = (
            Path("components")
            / component["dataset_id"]
            / "v1/component_card.md"
        )
        metadata_payloads[relative.as_posix()] = _component_card(
            component["component_id"], component
        ).encode("utf-8")
    for relative, payload in metadata_payloads.items():
        _replace_bytes(output / relative, payload)

    retained = [
        row
        for row in rows
        if row.get("role") in {"dataset_payload", "legal_metadata"}
    ]
    refreshed = list(retained)
    component_by_dataset = {
        component["dataset_id"]: component["component_id"]
        for component in plan["components"]
    }
    for relative in metadata_payloads:
        path = output / relative
        if relative.endswith("component_card.md"):
            dataset_id = Path(relative).parts[1]
            refreshed.append(
                _manifest_row(
                    path,
                    output,
                    role="component_documentation",
                    component=component_by_dataset[dataset_id],
                )
            )
        else:
            refreshed.append(
                _manifest_row(path, output, role="release_metadata")
            )
    refreshed.sort(key=lambda row: row["path"])
    _replace_bytes(
        manifest_path,
        b"".join(_json_line(row) for row in refreshed),
    )
    manifest_digest = _sha256(manifest_path)
    _replace_bytes(
        digest_path,
        f"{manifest_digest}  manifest.jsonl\n".encode("ascii"),
    )
    return {
        "status": "clean_staging_metadata_refreshed",
        "output": str(output),
        "component_count": len(EXPECTED_COMPONENTS),
        "manifest_entries": len(refreshed),
        "manifest_sha256": manifest_digest,
        "payload_file_count": plan["inventory"]["file_count"],
        "payload_bytes": plan["inventory"]["total_bytes"],
        "public_test_included": True,
    }


__all__ = [
    "CleanExportError",
    "EXPECTED_COMPONENTS",
    "build_export_plan",
    "export_hf_clean",
    "load_export_contract",
    "refresh_hf_clean_metadata",
]
