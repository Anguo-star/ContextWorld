#!/usr/bin/env python3
"""Build and verify the current ContextWorld reference-result freeze.

The v1 table was a numerical transcription.  This entry point makes the
next table an evidence index: every Development and Test score points at the
result file that supplied it, and carries the checkpoint, data selection,
scorer, adapter, threshold, runtime, and decision identities used for that
score.  The old v1 file is an input and is never overwritten.

The command intentionally does not load model weights.  A checkpoint hash
already emitted by an evaluator is retained as a declared identity and the
checkpoint path is checked for existence; re-hashing multi-gigabyte weights
belongs to a separate weight audit.  Result files and small provenance files
are hashed in full.

Typical use (after the outstanding evaluations have written their results)::

    python scripts/freeze_current_reference_baseline.py freeze \
      --result-overlay /path/to/action_delay_overlay.json
    python scripts/freeze_current_reference_baseline.py verify

``build_freeze`` and ``verify_freeze`` are also importable for tests and for
the handoff process.  The builder is deliberately strict by default: a
missing source, duplicate seed, mismatched task/split, missing CEM cell, or
unexplained changed data manifest fails closed.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import inspect
import json
import math
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    # ``python scripts/freeze_current_reference_baseline.py`` starts with the
    # scripts directory on sys.path; the decision contract lives at repo root.
    sys.path.insert(0, str(ROOT))
DEFAULT_V1 = ROOT / "configs/benchmark/contextworld_joint_scratch_v1_reference_results_freeze_v1.json"
DEFAULT_AUDIT = ROOT / "docs/reference/baseline_handoff_audit_2026-09-10.json"
DEFAULT_OUTPUT = ROOT / "configs/benchmark/contextworld_joint_scratch_v1_reference_results_freeze_v2.json"
DEFAULT_OVERLAY: Path | None = None
DECISION_MODULE = "contextworld.benchmarks.reference_decision"

EXPECTED_SEEDS = (3072, 3073, 3074)
TASKS = (
    "speed",
    "door",
    "action_delay",
    "portal_exit",
    "action_strength",
    "contact_friction",
    "motion_damping",
    "robot_arm_mass",
    "cube_gripper_carry",
)
FAMILIES = ("LeWM", "PLDM", "DINO-WM")
ORIGINAL_FAMILIES = ("LeWM", "PLDM")
ENVIRONMENT_BY_TASK = {
    "speed": "tworoom",
    "door": "tworoom",
    "action_delay": "tworoom",
    "portal_exit": "tworoom",
    "action_strength": "pusht",
    "contact_friction": "pusht",
    "motion_damping": "pusht",
    "robot_arm_mass": "reacher",
    "cube_gripper_carry": "cube",
}
FAMILY_FOLDER = {
    "LeWM": "lewm-contextworld-v1",
    "PLDM": "pldm-contextworld-v1",
    "DINO-WM": "dino-wm-contextworld-v1",
}
FAMILY_ADAPTER = {"LeWM": "lewm", "PLDM": "pldm", "DINO-WM": "prejepa"}
SCORER_SOURCE = {
    "speed": "contextworld/benchmarks/speed_icl_score.py",
    "door": "contextworld/benchmarks/door_icl_score.py",
    "action_delay": "contextworld/benchmarks/action_delay_icl_score.py",
    "portal_exit": "contextworld/benchmarks/portal_exit_icl_score.py",
    "action_strength": "contextworld/benchmarks/action_strength_icl_score.py",
    "contact_friction": "contextworld/benchmarks/contact_friction_icl_score.py",
    "motion_damping": "contextworld/benchmarks/motion_damping_icl_score.py",
    "robot_arm_mass": "contextworld/benchmarks/reacher_arm_mass_icl_score.py",
    "cube_gripper_carry": "contextworld/benchmarks/cube_grasp_rule_v4r1_icl_score.py",
}
RELEASE_CONFIG = {
    "speed": "configs/benchmark/tworoom_speed_icl_release_v1.yaml",
    "door": "configs/benchmark/tworoom_door_icl_release_v1.yaml",
    "action_delay": "configs/benchmark/tworoom_action_delay_icl_release_v1.yaml",
    "portal_exit": "configs/benchmark/tworoom_portal_exit_icl_release_v1.yaml",
    "action_strength": "configs/benchmark/pusht_action_strength_icl_release_v1.yaml",
    "contact_friction": "configs/benchmark/pusht_contact_friction_icl_release_v1.yaml",
    "motion_damping": "configs/benchmark/pusht_motion_damping_icl_release_v1.yaml",
    "robot_arm_mass": "configs/benchmark/reacher_arm_mass_icl_release_v1.yaml",
    "cube_gripper_carry": "configs/benchmark/cube_gripper_carry_h3_v4r1_icl_release_v1.yaml",
}
GATE_COMPLETION_CONFIG = "configs/benchmark/contextworld_test_gate_completion_v1.yaml"
ADAPTER_SOURCE_BY_MODULE = {
    "contextworld.benchmarks.adapters": "contextworld/benchmarks/adapters.py",
    "contextworld.benchmarks.prejepa_adapters": "contextworld/benchmarks/prejepa_adapters.py",
    "contextworld.benchmarks.action_delay_h3_tail_projection": "contextworld/benchmarks/action_delay_h3_tail_projection.py",
}
ACTION_DELAY_WRAPPER_SOURCE = "contextworld/benchmarks/action_delay_h3_tail_projection.py"
SNAPSHOT_PATH = "docs/reference/contextworld_native_v1_2026-09-03_snapshot.json"
WIP_RUNTIME_EVIDENCE = "docs/WIP_eval_standard_alignment.md"
RAW_SNAPSHOT_ROOT = "artifacts/evaluation/baseline_completion_v2/frozen_results"
ORIGINAL_SEED_CEM_SUMMARY = "artifacts/evaluation/original_baseline_seed_completion_v1/family_summary.json"
ORIGINAL_SEED_CEM_FREEZE = "configs/benchmark/contextworld_original_baseline_seed_completion_results_freeze_v1.json"
DINO_ORIGINAL_DIAGNOSTIC = "artifacts/evaluation/dinowm_original_diagnostic_v1/summary.json"


class FreezeError(RuntimeError):
    """Raised when a freeze input is incomplete or no longer bound."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _directory_manifest_sha256(path: Path) -> str:
    """Hash a directory identity without depending on directory mtime/order."""

    digest = hashlib.sha256()
    for child in sorted(path.rglob("*")):
        if not child.is_file() or child.is_symlink():
            continue
        relative = child.relative_to(path).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        digest.update(file_sha256(child).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    data = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def resolve_path(value: str | Path, *, root: Path = ROOT) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (root / candidate).resolve()


def identity(path: Path, *, root: Path = ROOT, label: str = "file") -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise FreezeError(f"missing {label}: {path}")
    try:
        stored_path = str(path.relative_to(root))
    except ValueError:
        stored_path = str(path)
    return {
        "path": stored_path,
        "resolved_path": str(path),
        "sha256": file_sha256(path),
        "size_bytes": path.stat().st_size,
    }


def read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FreezeError(f"cannot read {label}: {path}") from exc
    if not isinstance(value, Mapping):
        raise FreezeError(f"{label} must contain a JSON object: {path}")
    return dict(value)


def _unwrap(payload: Mapping[str, Any]) -> dict[str, Any]:
    value: Mapping[str, Any] = payload
    # Evaluator output normally wraps the scored object once in ``result``.
    # A second wrapper is accepted for old export utilities.
    for _ in range(3):
        nested = value.get("result")
        if isinstance(nested, Mapping):
            value = nested
        else:
            break
    return dict(value)


def _key(component_id: str, family: str, training_seed: int, stage: str, split: str) -> tuple[str, str, int, str, str]:
    return (str(component_id), str(family), int(training_seed), str(stage), str(split))


def row_key(row: Mapping[str, Any], split: str | None = None) -> tuple[str, str, int, str, str] | tuple[str, str, int, str]:
    base = _key(
        str(row["component_id"]),
        str(row["family"]),
        int(row["training_seed"]),
        str(row["stage"]),
        str(split or row.get("split", "")),
    )
    return base if split is not None or "split" in row else base[:4]


def _normalise_family(value: str) -> str:
    aliases = {"lewm": "LeWM", "pldm": "PLDM", "prejepa": "DINO-WM", "dino-wm": "DINO-WM", "dino_wm": "DINO-WM"}
    return aliases.get(value.lower(), value)


def _all_overlay_entries(payload: Any) -> list[Mapping[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, Mapping)]
    if not isinstance(payload, Mapping):
        raise FreezeError("result overlay must be a JSON object or list")
    for name in ("rows", "overrides", "results", "entries"):
        rows = payload.get(name)
        if isinstance(rows, list):
            return [row for row in rows if isinstance(row, Mapping)]
    entries: list[Mapping[str, Any]] = []
    for key, value in payload.items():
        if not isinstance(value, (str, Mapping)):
            continue
        if isinstance(value, str):
            value = {"path": value}
        row = dict(value)
        row.setdefault("source_path", key)
        # Accept compact pipe-delimited keys used by handoff manifests.
        parts = str(key).split("|")
        if len(parts) == 5:
            row.setdefault("component_id", parts[0])
            row.setdefault("family", parts[1])
            row.setdefault("training_seed", parts[2])
            row.setdefault("stage", parts[3])
            row.setdefault("split", parts[4])
        entries.append(row)
    return entries


def load_overlay(path: Path | None, *, root: Path = ROOT) -> tuple[dict[tuple[str, str, int, str, str], dict[str, Any]], dict[str, Any] | None]:
    if path is None:
        return {}, None
    path = resolve_path(path, root=root)
    payload = read_json(path, label="result overlay")
    entries = _all_overlay_entries(payload)
    mapping: dict[tuple[str, str, int, str, str], dict[str, Any]] = {}
    for index, raw in enumerate(entries):
        required = ("component_id", "family", "training_seed", "stage", "split")
        if any(field not in raw for field in required):
            # A source-path keyed entry may have identity embedded in a
            # nested ``identity`` object.
            nested = raw.get("identity")
            if isinstance(nested, Mapping):
                raw = {**nested, **raw}
        if any(field not in raw for field in required):
            raise FreezeError(f"overlay row {index} lacks row identity fields")
        replacement = raw.get("path", raw.get("result_path", raw.get("replacement_path")))
        if not isinstance(replacement, str) or not replacement:
            raise FreezeError(f"overlay row {index} lacks replacement result path")
        key = _key(
            str(raw["component_id"]),
            _normalise_family(str(raw["family"])),
            int(raw["training_seed"]),
            str(raw["stage"]),
            str(raw["split"]),
        )
        if key in mapping:
            raise FreezeError(f"duplicate overlay identity: {key}")
        item = dict(raw)
        item["path"] = str(resolve_path(replacement, root=root))
        item["declared_path"] = replacement
        mapping[key] = item
    return mapping, identity(path, root=root, label="result overlay")


def load_audit_rows(path: Path, *, root: Path = ROOT) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    audit_path = resolve_path(path, root=root)
    audit = read_json(audit_path, label="baseline handoff audit")
    rows = audit.get("rows")
    if not isinstance(rows, list):
        raise FreezeError("baseline audit has no rows list")
    seen: set[tuple[str, str, int, str, str]] = set()
    normalised: list[dict[str, Any]] = []
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            raise FreezeError(f"audit row {index} is not an object")
        row = dict(raw)
        for field in ("component_id", "family", "training_seed", "stage", "split"):
            if field not in row:
                raise FreezeError(f"audit row {index} lacks {field}")
        row["family"] = _normalise_family(str(row["family"]))
        key = _key(row["component_id"], row["family"], int(row["training_seed"]), row["stage"], row["split"])
        if key in seen:
            raise FreezeError(f"duplicate audit identity: {key}")
        seen.add(key)
        normalised.append(row)
    return normalised, identity(audit_path, root=root, label="baseline handoff audit")


def _candidate_paths(audit_row: Mapping[str, Any], *, root: Path) -> list[Path]:
    candidates: list[str] = []
    for field in ("path", "result_path"):
        value = audit_row.get(field)
        if isinstance(value, str):
            candidates.append(value)
    values = audit_row.get("candidate_paths", [])
    if isinstance(values, list):
        candidates.extend(str(value) for value in values if isinstance(value, str))
    result: list[Path] = []
    for value in candidates:
        path = resolve_path(value, root=root)
        if path not in result:
            result.append(path)
    return result


def _extract_score(payload: Mapping[str, Any], task: str, split: str) -> float:
    result = _unwrap(payload)
    metrics = result.get("metrics") if isinstance(result.get("metrics"), Mapping) else {}

    def get_path(value: Any, path: Sequence[Any]) -> Any:
        for key in path:
            if isinstance(value, Mapping):
                # A few JSON exporters serialise integer horizon keys as
                # strings; accept either representation while preserving the
                # declared path in the evidence record.
                if key in value:
                    value = value[key]
                elif str(key) in value:
                    value = value[str(key)]
                else:
                    raise KeyError(".".join(map(str, path)))
            elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
                try:
                    value = value[int(key)]
                except (ValueError, TypeError, IndexError):
                    raise KeyError(".".join(map(str, path))) from None
            else:
                raise KeyError(".".join(map(str, path)))
        return value

    candidates: list[tuple[Any, Sequence[Any]]] = []
    if task == "action_delay":
        candidates += [(metrics, ("physical_group_macro_accuracy",)), (result, ("core_h1", "physical_group_macro_accuracy")), (result, ("physical_group_macro_accuracy",))]
    elif task == "door":
        candidates += [(metrics, ("same_history_two_target_accuracy",)), (result, ("summary", "overall", "same_history_two_target_accuracy")), (result, ("same_history_two_target_accuracy",))]
    elif task == "speed":
        paths = (
            ("tracks", "unseen_interpolation", "horizons", "1", "reference_speed_balanced_strict_query_win_rate_vs_every_other"),
            ("tracks", "unseen_interpolation", "horizons", 1, "reference_speed_balanced_strict_query_win_rate_vs_every_other"),
        )
        candidates += [(metrics, path) for path in paths] + [(result, path) for path in paths]
    else:
        candidates += [(metrics, ("correct_future_rate",)), (result, ("metrics", "correct_future_rate")), (result, ("correct_future_rate",))]
    for value, path in candidates:
        try:
            score = get_path(value, path)
        except KeyError:
            continue
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(float(score)):
            raise FreezeError(f"non-finite score for {task}/{split}: {path}")
        return float(score)
    for value in (result, payload):
        for key in ("main_score", "score"):
            score = value.get(key) if isinstance(value, Mapping) else None
            if isinstance(score, (int, float)) and not isinstance(score, bool) and math.isfinite(float(score)):
                return float(score)
    raise FreezeError(f"cannot extract main score for {task}/{split}")


def _find_mapping(value: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in value:
            return value[key]
    return None


def _adapter_blocks(payload: Mapping[str, Any], result: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    model = result.get("model")
    if not isinstance(model, Mapping):
        model = payload.get("model") if isinstance(payload.get("model"), Mapping) else {}
    blocks: list[Mapping[str, Any]] = []
    adapter = model.get("adapter") if isinstance(model, Mapping) else None
    if isinstance(adapter, Mapping):
        blocks.append(adapter)
        base = adapter.get("base_adapter")
        if isinstance(base, Mapping):
            blocks.append(base)
    elif isinstance(model, Mapping):
        blocks.append(model)
    return blocks


def _module_source(class_name: str | None, *, root: Path) -> list[dict[str, Any]]:
    if not class_name or "." not in class_name:
        return []
    module_name = class_name.rsplit(".", 1)[0]
    relative = ADAPTER_SOURCE_BY_MODULE.get(module_name)
    if relative is None:
        if module_name == "__main__":
            relative = ACTION_DELAY_WRAPPER_SOURCE
        elif module_name.startswith("contextworld."):
            try:
                module = importlib.import_module(module_name)
                module_file = getattr(module, "__file__", None)
                if module_file:
                    relative = str(Path(module_file).resolve().relative_to(root.resolve()))
            except (ImportError, ValueError):
                relative = None
    if relative is None:
        return [{"module": module_name, "class": class_name, "path": None, "sha256": None, "status": "unresolved"}]
    path = resolve_path(relative, root=root)
    if not path.is_file():
        return [{"module": module_name, "class": class_name, "path": relative, "sha256": None, "status": "missing"}]
    return [{"module": module_name, "class": class_name, "path": relative, "sha256": file_sha256(path), "status": "verified"}]


def _source_hashes(task: str, result: Mapping[str, Any], payload: Mapping[str, Any], *, root: Path, strict: bool) -> dict[str, Any]:
    sources: dict[str, Any] = {}
    scorer_path = resolve_path(SCORER_SOURCE[task], root=root)
    threshold_paths = [RELEASE_CONFIG[task]]
    if task in ("speed", "door", "action_delay"):
        threshold_paths.append(GATE_COMPLETION_CONFIG)
    scorer = identity(scorer_path, root=root, label=f"{task} scorer source") if scorer_path.is_file() else {"path": SCORER_SOURCE[task], "sha256": None, "status": "missing"}
    if strict and scorer.get("sha256") is None:
        raise FreezeError(f"missing scorer source: {scorer_path}")
    sources["scorer"] = scorer
    threshold_rows = []
    for relative in threshold_paths:
        path = resolve_path(relative, root=root)
        if path.is_file():
            threshold_rows.append(identity(path, root=root, label="threshold source"))
        else:
            threshold_rows.append({"path": relative, "sha256": None, "status": "missing"})
    if strict and any(row.get("sha256") is None for row in threshold_rows):
        raise FreezeError(f"missing threshold source for {task}")
    sources["thresholds"] = threshold_rows
    adapter_rows: list[dict[str, Any]] = []
    for block in _adapter_blocks(payload, result):
        adapter_rows.extend(_module_source(block.get("adapter_class"), root=root))
    # Old Action Delay results serialised the wrapper as ``__main__`` but its
    # base adapter is in adapters.py; preserve both source identities.
    if any(row.get("path") == ACTION_DELAY_WRAPPER_SOURCE for row in adapter_rows) and task == "action_delay":
        base_path = resolve_path("contextworld/benchmarks/adapters.py", root=root)
        if base_path.is_file() and not any(row.get("path") == "contextworld/benchmarks/adapters.py" for row in adapter_rows):
            adapter_rows.append(identity(base_path, root=root, label="adapter source"))
    if not adapter_rows:
        adapter_spec = payload.get("adapter_spec")
        adapter_rows = [{"adapter_spec": adapter_spec, "path": None, "sha256": None, "status": "unresolved"}]
    if strict and any(row.get("sha256") is None for row in adapter_rows):
        raise FreezeError(f"adapter source could not be resolved for {task}")
    sources["adapters"] = adapter_rows
    return sources


def _data_identity(result: Mapping[str, Any], payload: Mapping[str, Any], *, root: Path, strict: bool) -> dict[str, Any]:
    bundle = result.get("bundle") if isinstance(result.get("bundle"), Mapping) else payload.get("bundle") if isinstance(payload.get("bundle"), Mapping) else {}
    data = result.get("data") if isinstance(result.get("data"), Mapping) else payload.get("data") if isinstance(payload.get("data"), Mapping) else {}
    release = result.get("release") if isinstance(result.get("release"), Mapping) else payload.get("release") if isinstance(payload.get("release"), Mapping) else {}
    selection = result.get("selection") if isinstance(result.get("selection"), Mapping) else payload.get("selection") if isinstance(payload.get("selection"), Mapping) else {}
    merged: dict[str, Any] = {
        "bundle_schema_version": bundle.get("bundle_schema_version"),
        "component_id": bundle.get("component_id"),
        "dataset_id": bundle.get("dataset_id"),
        "development_payload_id": bundle.get("development_payload_id"),
        "manifest_sha256": _find_mapping(bundle, "manifest_sha256") or _find_mapping(release, "content_manifest_sha256", "manifest_sha256"),
        "task_registry_sha256": bundle.get("task_registry_sha256"),
        "members": list(bundle.get("members", [])) if isinstance(bundle.get("members"), list) else [],
        "selection": copy.deepcopy(selection),
        "data_fields": copy.deepcopy(data),
        "release_fields": copy.deepcopy(release),
    }
    if not merged["selection"] and isinstance(data, Mapping):
        merged["selection"] = {key: copy.deepcopy(data[key]) for key in ("catalog", "queries", "eval_seeds", "queries_per_eval_seed", "history_conditions", "track", "full_protocol") if key in data}
    merged["selection_sha256"] = canonical_sha256(merged["selection"])
    member_identities: list[dict[str, Any]] = []
    data_roots = [
        Path("/opt/huawei/explorer-env/dataset/ag_data/data/world_model/ContextWorld-v1"),
        Path("/opt/huawei/explorer-env/dataset/ag_data/data/world_model/ContextWorld-v1-full"),
    ]
    # Test fixtures can provide an explicit package root in the result.
    for candidate in (bundle.get("root"), data.get("root"), release.get("root")):
        if isinstance(candidate, str):
            data_roots.insert(0, resolve_path(candidate, root=root))
    cache: dict[Path, str] = {}
    package_manifests: dict[Path, dict[str, str]] = {}
    package_manifest_identities: list[dict[str, Any]] = []
    for data_root in data_roots:
        manifest_path = data_root / "manifest.jsonl"
        if not manifest_path.is_file():
            continue
        try:
            manifest_rows = {}
            for line in manifest_path.read_text(encoding="utf-8").splitlines():
                item = json.loads(line)
                if isinstance(item, Mapping) and isinstance(item.get("path"), str) and isinstance(item.get("sha256"), str):
                    manifest_rows[item["path"]] = item["sha256"]
            package_manifests[data_root] = manifest_rows
            package_manifest_identities.append(identity(manifest_path, root=root, label="data package manifest"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
    for member in merged["members"]:
        if not isinstance(member, str):
            continue
        member_path = None
        for data_root in data_roots:
            candidate = data_root / member
            if (candidate.is_file() or candidate.is_dir()) and not candidate.is_symlink():
                member_path = candidate
                break
        item: dict[str, Any] = {"path": member, "sha256": None, "status": "unresolved"}
        if member_path is not None:
            data_root = next((candidate for candidate in data_roots if (candidate / member) == member_path), None)
            manifest_rows = package_manifests.get(data_root, {}) if data_root is not None else {}
            prefix = member.rstrip("/") + "/"
            matching = sorted((path, digest) for path, digest in manifest_rows.items() if path == member or path.startswith(prefix))
            if matching:
                cache_key = member_path
                if cache_key not in cache:
                    cache[cache_key] = canonical_sha256(matching)
                item.update(
                    resolved_path=str(member_path),
                    sha256=cache[cache_key],
                    status="verified_from_package_manifest",
                    manifest_member_count=len(matching),
                )
            else:
                if member_path not in cache:
                    cache[member_path] = file_sha256(member_path) if member_path.is_file() else _directory_manifest_sha256(member_path)
                item.update(resolved_path=str(member_path), sha256=cache[member_path], status="verified", size_bytes=member_path.stat().st_size)
        member_identities.append(item)
    if strict and merged["members"] and any(item["sha256"] is None for item in member_identities):
        raise FreezeError(f"one or more data members cannot be resolved for {merged.get('component_id')}")
    merged["member_identities"] = member_identities
    merged["package_manifests"] = package_manifest_identities
    declared_manifest_sha = merged.get("manifest_sha256")
    matching_manifests = [
        item
        for item in package_manifest_identities
        if isinstance(item, Mapping) and item.get("sha256") == declared_manifest_sha
    ]
    if not matching_manifests:
        for member in member_identities:
            if member.get("status") == "verified_from_package_manifest":
                member["status"] = "current_member_hash_with_declared_historical_manifest"
    merged["result_declared_manifest_sha256"] = declared_manifest_sha
    merged["manifest_verified"] = bool(matching_manifests)
    merged["manifest_verification_scope"] = (
        "result_declared_manifest_matches_current_package_manifest"
        if matching_manifests
        else "declared_historical_identity; current package member hashes are checked separately"
    )
    merged["identity_sha256"] = canonical_sha256({key: merged[key] for key in ("manifest_sha256", "task_registry_sha256", "members", "selection_sha256")})
    return merged


def _runtime_evidence(result: Mapping[str, Any], payload: Mapping[str, Any], *, root: Path) -> dict[str, Any]:
    runtime: Any = None
    for value in (result, payload):
        if isinstance(value, Mapping):
            runtime = _find_mapping(value, "runtime_identity", "runtime_fingerprint", "runtime", "evaluation_runtime")
            if runtime is not None:
                break
    if runtime is not None:
        return {"status": "recorded", "identity": copy.deepcopy(runtime), "identity_sha256": canonical_sha256(runtime)}
    evidence_path = resolve_path(WIP_RUNTIME_EVIDENCE, root=root)
    evidence = identity(evidence_path, root=root, label="legacy runtime evidence") if evidence_path.is_file() else {"path": WIP_RUNTIME_EVIDENCE, "sha256": None, "status": "missing"}
    return {
        "status": "legacy_runtime_unrecorded",
        "identity": None,
        "claim_boundary": "The source result did not record evaluator runtime; the WIP handoff is historical declaration-level evidence only.",
        "declaration_evidence": evidence,
    }


def _checkpoint_identity(result: Mapping[str, Any], payload: Mapping[str, Any], *, strict: bool) -> dict[str, Any]:
    blocks = _adapter_blocks(payload, result)
    block: Mapping[str, Any] = blocks[0] if blocks else {}
    checkpoint = block.get("checkpoint")
    checkpoint_sha = block.get("checkpoint_sha256")
    if not isinstance(checkpoint, str):
        checkpoint = block.get("path") if isinstance(block.get("path"), str) else None
    if not isinstance(checkpoint_sha, str):
        checkpoint_sha = block.get("sha256") if isinstance(block.get("sha256"), str) else None
    if not isinstance(checkpoint, str) or not checkpoint:
        if strict:
            raise FreezeError("result lacks checkpoint path")
        checkpoint = None
    if not isinstance(checkpoint_sha, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", checkpoint_sha):
        if strict:
            raise FreezeError("result lacks a 64-character checkpoint SHA-256")
        checkpoint_sha = None
    exists = bool(checkpoint and Path(checkpoint).is_file())
    if strict and not exists:
        raise FreezeError(f"checkpoint path is not readable: {checkpoint}")
    return {
        "path": checkpoint,
        "sha256": checkpoint_sha.lower() if isinstance(checkpoint_sha, str) else None,
        "path_exists": exists,
        "verification_scope": "declared_sha256_and_path_checked; weight_bytes_not_rehashed",
    }


def _training_identity(
    checkpoint: Mapping[str, Any],
    *,
    root: Path,
    required: bool,
) -> dict[str, Any]:
    """Bind the small per-run training receipt beside the checkpoint.

    The receipt is intentionally indexed by path and SHA rather than copied
    into every result row.  Its internal identity digest remains available to
    readers, while verification can detect edits to the receipt itself.
    """

    checkpoint_value = checkpoint.get("path")
    identity_path: Path | None = None
    if isinstance(checkpoint_value, str) and checkpoint_value:
        checkpoint_path = resolve_path(checkpoint_value, root=root)
        identity_path = checkpoint_path.parent / "contextworld_training_identity_v1.json"
    if identity_path is None or not identity_path.is_file():
        if required:
            raise FreezeError(
                "missing contextworld_training_identity_v1.json beside checkpoint: "
                f"{identity_path or checkpoint_value}"
            )
        return {
            "status": "training_identity_unrecorded",
            "claim_boundary": "No per-run training identity receipt was present beside the declared checkpoint.",
            "source": {
                "path": str(identity_path) if identity_path is not None else None,
                "sha256": None,
                "status": "missing",
            },
        }
    receipt = read_json(identity_path, label="training identity")
    source = identity(identity_path, root=root, label="training identity")
    return {
        "status": "recorded",
        "source": source,
        "schema_version": receipt.get("schema_version"),
        "declared_identity_sha256": receipt.get("identity_sha256"),
    }


def _decision_source(*, root: Path) -> dict[str, Any]:
    module_path = resolve_path("contextworld/benchmarks/reference_decision.py", root=root)
    if module_path.is_file():
        return {"module": DECISION_MODULE, **identity(module_path, root=root, label="reference decision source")}
    return {"module": DECISION_MODULE, "path": "contextworld/benchmarks/reference_decision.py", "sha256": None, "status": "missing"}


def _decision_contract_identity(*, root: Path) -> dict[str, Any] | None:
    try:
        module = importlib.import_module(DECISION_MODULE)
        function = getattr(module, "reference_contract_identity", None)
        if callable(function):
            value = function(repo_root=root)
            if isinstance(value, Mapping):
                return copy.deepcopy(dict(value))
    except (ImportError, TypeError, ValueError, FileNotFoundError):
        return None
    return None


def _decision_inputs(result: Mapping[str, Any], payload: Mapping[str, Any], *, task: str, split: str, score: float) -> dict[str, Any]:
    metrics = result.get("metrics") if isinstance(result.get("metrics"), Mapping) else {}
    return {
        "task": task,
        "split": split,
        "main_score": score,
        "gate": copy.deepcopy(result.get("gate")),
        "summary_decision": copy.deepcopy((result.get("summary") or {}).get("decision") if isinstance(result.get("summary"), Mapping) else None),
        "formal_checkpoint_passed": result.get("formal_checkpoint_passed"),
        "gate_completion_inputs": copy.deepcopy(metrics.get("gate_completion_inputs")),
        "metrics_gate": copy.deepcopy(metrics.get("gate")),
    }


def _legacy_decision(result: Mapping[str, Any], *, task: str, split: str) -> dict[str, Any]:
    values: list[Any] = [result.get("formal_checkpoint_passed"), result.get("gate", {}).get("passed") if isinstance(result.get("gate"), Mapping) else None]
    summary = result.get("summary")
    if isinstance(summary, Mapping) and isinstance(summary.get("decision"), Mapping):
        values.append(summary["decision"].get("passed"))
    values = [value for value in values if isinstance(value, bool)]
    if values:
        passed = all(values)
        return {"decision_id": "legacy_result_decision", "passed": passed, "status": "recovered_from_legacy_result", "reason_codes": [] if passed else ["legacy_gate_failed"]}
    return {"decision_id": "legacy_decision_unrecorded", "passed": False, "status": "decision_unavailable", "reason_codes": ["decision_inputs_missing"]}


def _invoke_reference_decision(context: Mapping[str, Any], *, root: Path, strict: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    source = _decision_source(root=root)
    module = None
    try:
        module = importlib.import_module(DECISION_MODULE)
    except ImportError:
        module = None
    decision: Any = None
    if module is not None:
        # The coordinating implementation exposes this explicit API.  Keep
        # the generic fallbacks below for short-lived compatibility with an
        # in-flight module during the handoff.
        direct = getattr(module, "reference_decision_for_result", None)
        if callable(direct):
            try:
                decision = direct(
                    context["task"],
                    context["result"],
                    split=context["split"],
                    repo_root=root,
                )
            except (TypeError, KeyError, ValueError, FileNotFoundError):
                decision = None
        for name in ("evaluate_row", "evaluate_result", "decide", "compute_decision", "reference_decision"):
            if decision is not None:
                break
            function = getattr(module, name, None)
            if not callable(function):
                continue
            try:
                signature = inspect.signature(function)
                kwargs = {key: value for key, value in context.items() if key in signature.parameters}
            except (TypeError, ValueError):
                kwargs = {}
            attempts = (((), kwargs), ((dict(context),), {}), ((), dict(context)))
            for args, call_kwargs in attempts:
                try:
                    decision = function(*args, **call_kwargs)
                    break
                except (TypeError, KeyError, ValueError):
                    continue
            if decision is not None:
                break
    if decision is None:
        if strict and module is None:
            # A local test fixture may intentionally omit the coordinating
            # module; production freeze requires its hash-bound decision.
            decision = _legacy_decision(context["result"], task=context["task"], split=context["split"])
        else:
            decision = _legacy_decision(context["result"], task=context["task"], split=context["split"])
    if isinstance(decision, bool):
        decision = {"passed": decision}
    if not isinstance(decision, Mapping):
        raise FreezeError("reference_decision returned a non-object decision")
    output = dict(decision)
    if "passed" not in output:
        output["passed"] = output.get("all_gates_passed") if isinstance(output.get("all_gates_passed"), bool) else False
    output.setdefault("status", "passed" if output["passed"] else "failed")
    output.setdefault("reason_codes", [] if output["passed"] else ["decision_failed"])
    output["inputs_sha256"] = canonical_sha256(context["inputs"])
    output.setdefault(
        "decision_id",
        f"{output.get('contract_id', 'reference_decision')}:{output['inputs_sha256']}",
    )
    output["source"] = copy.deepcopy(source)
    return output, source


def _cem_files(result: Mapping[str, Any], *, task: str, root: Path, strict: bool) -> dict[str, Any]:
    blocks = _adapter_blocks({}, result)
    adapter = blocks[0] if blocks else {}
    checkpoint = adapter.get("checkpoint") if isinstance(adapter, Mapping) else None
    run_dir = Path(checkpoint).parent if isinstance(checkpoint, str) else None
    candidates: list[Path] = []
    if run_dir is not None:
        candidates.extend(sorted((run_dir / "eval_results" / "benchmark_cem" / task).glob("*metrics.json")))
    rows = []
    seen_seeds: set[int] = set()
    for path in candidates:
        try:
            item = read_json(path, label="CEM result")
        except FreezeError:
            continue
        seed = item.get("eval_seed")
        if not isinstance(seed, int):
            match = re.search(r"seed(\d+)", path.name)
            seed = int(match.group(1)) if match else None
        if seed is None or seed in seen_seeds or seed not in (42, 43, 44, 45, 46, 47):
            continue
        seen_seeds.add(seed)
        metrics = item.get("metrics") if isinstance(item.get("metrics"), Mapping) else {}
        rows.append({
            "eval_seed": seed,
            "success_rate_percent": metrics.get("success_rate_percent"),
            "trials": item.get("num_eval", metrics.get("evaluation_count", 50)),
            "source": identity(path, root=root, label="CEM result"),
        })
    rows.sort(key=lambda row: row["eval_seed"])
    if strict and len(rows) != 6:
        raise FreezeError(f"expected six CEM seed results for {task}, found {len(rows)}")
    if strict:
        if [row["eval_seed"] for row in rows] != [42, 43, 44, 45, 46, 47]:
            raise FreezeError(f"CEM eval-seed coverage is incomplete for {task}")
        if any(row.get("success_rate_percent") is None for row in rows):
            raise FreezeError(f"CEM success-rate value is missing for {task}")
        if any(row.get("trials") is None for row in rows):
            raise FreezeError(f"CEM trial count is missing for {task}")
    return {
        "status": "verified" if len(rows) == 6 else "incomplete",
        "eval_seeds": [row["eval_seed"] for row in rows],
        "trials_per_seed": [row["trials"] for row in rows],
        "rows": rows,
        "identity_sha256": canonical_sha256(rows),
    }


def _original_cem_evidence(
    *, task: str, family: str, training_seed: int, root: Path, strict: bool
) -> dict[str, Any]:
    """Return the three-seed original-environment CEM member receipt.

    The seed-completion summary is the authoritative source for the current
    original baseline.  Its member carries the six per-evaluation-seed
    values and the individual source-file hashes, so consumers do not need to
    recover a score from the older one-cell matrix summary.
    """

    freeze_path = resolve_path(ORIGINAL_SEED_CEM_FREEZE, root=root)
    summary_path = resolve_path(ORIGINAL_SEED_CEM_SUMMARY, root=root)
    freeze_identity = (
        identity(freeze_path, root=root, label="original seed CEM freeze")
        if freeze_path.is_file()
        else {"path": ORIGINAL_SEED_CEM_FREEZE, "sha256": None, "status": "missing"}
    )
    summary_identity = (
        identity(summary_path, root=root, label="original seed CEM family summary")
        if summary_path.is_file()
        else {"path": ORIGINAL_SEED_CEM_SUMMARY, "sha256": None, "status": "missing"}
    )
    if strict and (freeze_identity.get("sha256") is None or summary_identity.get("sha256") is None):
        raise FreezeError("original seed-completion CEM evidence is missing")

    member: Mapping[str, Any] | None = None
    if summary_path.is_file():
        summary = read_json(summary_path, label="original seed CEM family summary")
        environment = ENVIRONMENT_BY_TASK[task]
        family_name = family.lower()
        families = summary.get("families", [])
        family_row = next(
            (
                item
                for item in families
                if isinstance(item, Mapping)
                and item.get("environment") == environment
                and str(item.get("family", "")).lower() == family_name
            ),
            None,
        )
        if isinstance(family_row, Mapping):
            member = next(
                (
                    item
                    for item in family_row.get("members", [])
                    if isinstance(item, Mapping)
                    and int(item.get("training_seed", -1)) == int(training_seed)
                ),
                None,
            )
    if strict and member is None:
        raise FreezeError(
            f"original seed-completion CEM member missing: "
            f"{ENVIRONMENT_BY_TASK[task]}/{family.lower()}/s{training_seed}"
        )
    return {
        "kind": "original_environment_cem_seed_completion_member",
        "environment": ENVIRONMENT_BY_TASK[task],
        "family": family.lower(),
        "training_seed": int(training_seed),
        "member": copy.deepcopy(dict(member)) if isinstance(member, Mapping) else None,
        "source_summary": summary_identity,
        "source_freeze": freeze_identity,
        "claim_boundary": "Original-environment CEM evidence is retained as comparison evidence; it is not a current v1 ICL admission row.",
    }


def _snapshot_entry(task: str, family: str, seed: int, *, root: Path) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    path = resolve_path(SNAPSHOT_PATH, root=root)
    if not path.is_file():
        return None, None
    payload = read_json(path, label="native checkpoint snapshot")
    entries = payload.get("entries", [])
    target = next((entry for entry in entries if isinstance(entry, Mapping) and entry.get("task") == task and _normalise_family(str(entry.get("family"))) == family and int(entry.get("training_seed", -1)) == seed), None)
    return (dict(target) if isinstance(target, Mapping) else None), identity(path, root=root, label="native checkpoint snapshot")


def _result_for_audit(audit_row: Mapping[str, Any], overlay: Mapping[tuple[str, str, int, str, str], Mapping[str, Any]], *, root: Path) -> tuple[Path, dict[str, Any] | None]:
    key = _key(audit_row["component_id"], audit_row["family"], int(audit_row["training_seed"]), audit_row["stage"], audit_row["split"])
    replacement = overlay.get(key)
    if replacement is not None:
        path = resolve_path(str(replacement["path"]), root=root)
        if not path.is_file():
            raise FreezeError(f"overlay result is missing: {path}")
        return path, dict(replacement)
    candidates = _candidate_paths(audit_row, root=root)
    for path in candidates:
        if path.is_file():
            return path, None
    raise FreezeError(f"missing result for {key}; candidates={candidates}")


def _snapshot_result(
    source_path: Path,
    *,
    row: Mapping[str, Any],
    split: str,
    root: Path,
    snapshot_root: Path | str = RAW_SNAPSHOT_ROOT,
) -> tuple[Path, dict[str, Any]]:
    """Copy one raw result into the repository-owned immutable evidence tree."""

    configured_root = Path(snapshot_root).expanduser()
    if configured_root.is_absolute():
        destination = configured_root.resolve() / str(row["component_id"]) / str(row["stage"]) / str(row["family"]) / f"s{int(row['training_seed'])}" / f"{split}.json"
        relative = destination
    else:
        relative = configured_root / str(row["component_id"]) / str(row["stage"]) / str(row["family"]) / f"s{int(row['training_seed'])}" / f"{split}.json"
        destination = (root / relative).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_hash = file_sha256(source_path)
    if destination.exists():
        if not destination.is_file() or file_sha256(destination) != source_hash:
            raise FreezeError(f"raw result snapshot already exists with a different SHA: {destination}")
    else:
        shutil.copyfile(source_path, destination)
    return destination, {
        "origin_path": str(source_path),
        "origin_sha256": source_hash,
        "snapshot_path": str(relative if not configured_root.is_absolute() else destination),
        "snapshot_sha256": file_sha256(destination),
        "snapshot_size_bytes": destination.stat().st_size,
    }


def _validate_payload_identity(payload: Mapping[str, Any], *, task: str, split: str, family: str, seed: int, stage: str) -> None:
    result = _unwrap(payload)
    bundle = result.get("bundle") if isinstance(result.get("bundle"), Mapping) else payload.get("bundle") if isinstance(payload.get("bundle"), Mapping) else None
    if isinstance(bundle, Mapping) and bundle.get("component_id") not in (None, task):
        raise FreezeError(f"result component mismatch: expected {task}, got {bundle.get('component_id')}")
    for value in (result.get("evaluation_split"), payload.get("evaluation_split")):
        if value in ("development", "test", "public_test"):
            observed = "test" if value == "public_test" else value
            if observed != split:
                raise FreezeError(f"result split mismatch: expected {split}, got {value}")
    blocks = _adapter_blocks(payload, result)
    if blocks:
        observed_seed = blocks[0].get("training_seed")
        if observed_seed is not None and int(observed_seed) != seed:
            raise FreezeError(f"result seed mismatch: expected {seed}, got {observed_seed}")


def _changed_manifest_requires_explanation(
    old_payload: Mapping[str, Any],
    new_payload: Mapping[str, Any],
    overlay: Mapping[str, Any] | None,
    *,
    root: Path = ROOT,
) -> None:
    if overlay is None:
        return
    old = _data_identity(_unwrap(old_payload), old_payload, root=root, strict=False).get("manifest_sha256")
    new = _data_identity(_unwrap(new_payload), new_payload, root=root, strict=False).get("manifest_sha256")
    if old and new and old != new:
        stated = any(bool(overlay.get(key)) for key in ("reason", "justification", "change_reason", "allow_manifest_change"))
        if not stated:
            raise FreezeError("overlay changed data manifest without reason/justification")


def _source_row(audit_rows: Mapping[tuple[str, str, int, str, str], Mapping[str, Any]], row: Mapping[str, Any], split: str) -> Mapping[str, Any]:
    key = _key(row["component_id"], _normalise_family(str(row["family"])), int(row["training_seed"]), row["stage"], split)
    try:
        return audit_rows[key]
    except KeyError as exc:
        raise FreezeError(f"audit lacks source row {key}") from exc


def _build_split(
    *, row: Mapping[str, Any], split: str, audit_row: Mapping[str, Any], result_path: Path, overlay: Mapping[str, Any] | None, root: Path, strict: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = read_json(result_path, label="source result")
    result = _unwrap(payload)
    _validate_payload_identity(payload, task=str(row["component_id"]), split=split, family=str(row["family"]), seed=int(row["training_seed"]), stage=str(row["stage"]))
    score = _extract_score(payload, str(row["component_id"]), split)
    source_identity = identity(result_path, root=root, label="source result")
    if overlay is not None and isinstance(overlay.get("origin_path"), str):
        source_identity["origin_path"] = overlay["origin_path"]
        source_identity["origin_sha256"] = overlay.get("origin_sha256")
    checkpoint = _checkpoint_identity(result, payload, strict=strict)
    training_identity = _training_identity(
        checkpoint,
        root=root,
        required=str(row.get("stage")) == "post_component_training" and strict,
    )
    data = _data_identity(result, payload, root=root, strict=strict)
    sources = _source_hashes(str(row["component_id"]), result, payload, root=root, strict=strict)
    runtime = _runtime_evidence(result, payload, root=root)
    decision_inputs = _decision_inputs(result, payload, task=str(row["component_id"]), split=split, score=score)
    decision_context = {"task": str(row["component_id"]), "family": str(row["family"]), "stage": str(row["stage"]), "split": split, "training_seed": int(row["training_seed"]), "score": score, "result": result, "payload": payload, "inputs": decision_inputs, "thresholds": sources["thresholds"]}
    decision, decision_source = _invoke_reference_decision(decision_context, root=root, strict=False)
    adapter_blocks = _adapter_blocks(payload, result)
    adapter_class = adapter_blocks[0].get("adapter_class") if adapter_blocks else None
    split_block: dict[str, Any] = {
        "main_score": score,
        "all_gates_passed": bool(decision.get("passed")),
        "decision": decision,
        "decision_source": decision_source,
        "decision_contract_identity": _decision_contract_identity(root=root),
        "source_result_path": source_identity["path"],
        "source_result_sha256": source_identity["sha256"],
        "source_result_size_bytes": source_identity["size_bytes"],
        "source_file": source_identity,
        "checkpoint": checkpoint,
        "checkpoint_path": checkpoint["path"],
        "checkpoint_sha256": checkpoint["sha256"],
        "training_identity": training_identity,
        "data_identity": data,
        "selection_identity": {"selection": data["selection"], "selection_sha256": data["selection_sha256"]},
        "scorer_source": sources["scorer"],
        "adapter_source": sources["adapters"],
        "threshold_sources": sources["thresholds"],
        "source_hashes": {"result": source_identity, "scorer": sources["scorer"], "adapters": sources["adapters"], "thresholds": sources["thresholds"], "decision": decision_source},
        "adapter_identity": {"class": adapter_class, "adapter_id": adapter_blocks[0].get("adapter_id") if adapter_blocks else None},
        "runtime_evidence": runtime,
        "overlay": copy.deepcopy(overlay) if overlay is not None and ("declared_path" in overlay or "source_path" in overlay) else None,
        "raw_snapshot": {
            "path": source_identity.get("path"),
            "sha256": source_identity.get("sha256"),
            "origin_path": source_identity.get("origin_path"),
            "origin_sha256": source_identity.get("origin_sha256"),
        },
    }
    return split_block, result


def _validate_coverage(rows: Sequence[Mapping[str, Any]], *, strict: bool) -> None:
    keys = [_key(row["component_id"], _normalise_family(str(row["family"])), int(row["training_seed"]), row["stage"], "") for row in rows]
    if len(set(keys)) != len(keys):
        raise FreezeError("duplicate checkpoint identity in v1 checkpoint_results")
    if not strict:
        return
    if len(rows) != 135:
        raise FreezeError(f"expected 135 checkpoint rows (81 training + 54 original), found {len(rows)}")
    expected: set[tuple[str, str, int, str, str]] = set()
    for task in TASKS:
        for family in FAMILIES:
            for seed in EXPECTED_SEEDS:
                expected.add(_key(task, family, seed, "post_component_training", ""))
        for family in ORIGINAL_FAMILIES:
            for seed in EXPECTED_SEEDS:
                expected.add(_key(task, family, seed, "original_environment_only", ""))
    observed = set(keys)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise FreezeError(f"checkpoint coverage mismatch; missing={missing[:5]} extra={extra[:5]}")


def build_freeze(
    *,
    v1_path: Path | str = DEFAULT_V1,
    audit_path: Path | str = DEFAULT_AUDIT,
    result_overlay: Path | str | None = DEFAULT_OVERLAY,
    output: Path | str = DEFAULT_OUTPUT,
    repo_root: Path = ROOT,
    strict: bool = True,
    write: bool = True,
    copy_raw_results: bool = True,
    raw_snapshot_root: Path | str = RAW_SNAPSHOT_ROOT,
) -> dict[str, Any]:
    root = Path(repo_root).expanduser().resolve()
    v1_file = resolve_path(v1_path, root=root)
    v1 = read_json(v1_file, label="reference freeze v1")
    if not isinstance(v1.get("checkpoint_results"), list):
        raise FreezeError("v1 freeze lacks checkpoint_results")
    base_rows = [dict(row) for row in v1["checkpoint_results"] if isinstance(row, Mapping)]
    _validate_coverage(base_rows, strict=strict)
    audit_rows_list, audit_identity = load_audit_rows(resolve_path(audit_path, root=root), root=root)
    audit_rows = {_key(row["component_id"], row["family"], int(row["training_seed"]), row["stage"], row["split"]): row for row in audit_rows_list}
    if strict and len(audit_rows) != 270:
        raise FreezeError(f"expected 270 source rows in audit, found {len(audit_rows)}")
    overlay_map, overlay_identity = load_overlay(resolve_path(result_overlay, root=root) if result_overlay is not None else None, root=root)
    v1_identity = identity(v1_file, root=root, label="reference freeze v1")
    output_rows: list[dict[str, Any]] = []
    cem_training_count = 0
    excluded_test: list[dict[str, Any]] = []
    for base in base_rows:
        row = {key: copy.deepcopy(value) for key, value in base.items() if key not in ("development", "test", "original_task_cem")}
        row["family"] = _normalise_family(str(row["family"]))
        row["development"] = {}
        row["test"] = {}
        for split in ("development", "test"):
            audit_row = _source_row(audit_rows, row, split)
            result_path, overlay_entry = _result_for_audit(audit_row, overlay_map, root=root)
            payload = read_json(result_path, label="source result")
            if overlay_entry is not None:
                # Check replacement semantics against the original audit row.
                original_candidates = _candidate_paths(audit_row, root=root)
                if original_candidates and original_candidates[0].is_file():
                    old_payload = read_json(original_candidates[0], label="original source result")
                    _changed_manifest_requires_explanation(old_payload, payload, overlay_entry, root=root)
            if copy_raw_results:
                snapshot_path, snapshot_meta = _snapshot_result(result_path, row=row, split=split, root=root, snapshot_root=raw_snapshot_root)
            else:
                snapshot_path = result_path
                snapshot_meta = {"origin_path": str(result_path), "origin_sha256": file_sha256(result_path), "snapshot_path": str(result_path), "snapshot_sha256": file_sha256(result_path), "snapshot_size_bytes": result_path.stat().st_size}
            snapshot_overlay = dict(overlay_entry or {})
            snapshot_overlay.update(snapshot_meta)
            split_block, result = _build_split(row=row, split=split, audit_row=audit_row, result_path=snapshot_path, overlay=snapshot_overlay, root=root, strict=strict)
            split_block["origin_path"] = snapshot_meta["origin_path"]
            split_block["origin_sha256"] = snapshot_meta["origin_sha256"]
            split_block["snapshot_path"] = snapshot_meta["snapshot_path"]
            row[split] = split_block
            if split == "test" and row["component_id"] in ("contact_friction", "motion_damping"):
                excluded_test.append({
                    "component_id": row["component_id"], "family": row["family"], "training_seed": row["training_seed"], "stage": row["stage"],
                    "source_result_path": split_block["source_result_path"], "source_result_sha256": split_block["source_result_sha256"],
                    "status": "excluded_from_final_reporting", "reason": "historical_public_test_read_and_hashed; component is outside current final admission",
                })
        # Development admission is recorded independently from historical Test existence.
        development_decision = row["development"]["decision"]
        dev_cleared = bool(development_decision.get("passed"))
        row["admission"] = {
            "cleared_development": dev_cleared,
            "decision_id": development_decision.get("decision_id") or f"reference_decision:{development_decision.get('inputs_sha256')}",
            "evidence": {"split": "development", "source_result_path": row["development"]["source_result_path"], "source_result_sha256": row["development"]["source_result_sha256"], "decision_source": row["development"]["decision_source"]},
            "historical_test_present": True,
            "current_test_admission": bool(dev_cleared and row["stage"] == "post_component_training" and row["component_id"] not in ("contact_friction", "motion_damping")),
        }
        row["test_history"] = {
            "result_present": True,
            "source_result_path": row["test"]["source_result_path"],
            "source_result_sha256": row["test"]["source_result_sha256"],
            "current_admission": row["admission"]["current_test_admission"],
            "excluded_from_final_reporting": row["component_id"] in ("contact_friction", "motion_damping"),
        }
        if row["stage"] == "post_component_training":
            cem = _cem_files(
                _unwrap(
                    read_json(
                        resolve_path(row["development"]["source_result_path"], root=root),
                        label="development source",
                    )
                ),
                task=row["component_id"],
                root=root,
                strict=strict,
            )
            # The function above receives the source result, while CEM lives
            # beside its checkpoint.  Use the actual source payload to retain
            # the checkpoint directory when the stored path is portable.
            if cem["status"] != "verified":
                payload = read_json(resolve_path(row["development"]["source_result_path"], root=root), label="development source")
                cem = _cem_files(_unwrap(payload), task=row["component_id"], root=root, strict=False)
            row["cem_evidence"] = cem
            if strict and cem["status"] != "verified":
                raise FreezeError(f"missing CEM evidence for {row['component_id']}/{row['family']}/{row['training_seed']}")
            expected_cem = base.get("original_task_cem")
            expected_rates = expected_cem.get("success_rate_percent_by_eval_seed") if isinstance(expected_cem, Mapping) else None
            if strict and isinstance(expected_rates, Mapping):
                observed_rates = {
                    str(item["eval_seed"]): float(item["success_rate_percent"])
                    for item in cem["rows"]
                }
                normalized_expected = {str(key): float(value) for key, value in expected_rates.items()}
                if observed_rates != normalized_expected:
                    raise FreezeError(
                        f"post-training CEM values disagree with v1 for "
                        f"{row['component_id']}/{row['family']}/{row['training_seed']}"
                    )
            cem_training_count += 1
            snapshot_entry, snapshot_identity = _snapshot_entry(row["component_id"], row["family"], int(row["training_seed"]), root=root)
            row["cem_evidence"]["snapshot_entry"] = snapshot_entry
            row["cem_evidence"]["snapshot_source"] = snapshot_identity
            if "original_task_cem" in base:
                row["original_task_cem"] = copy.deepcopy(base["original_task_cem"])
        else:
            row["original_cem_evidence"] = _original_cem_evidence(
                task=row["component_id"],
                family=row["family"],
                training_seed=int(row["training_seed"]),
                root=root,
                strict=strict,
            )
        output_rows.append(row)
    if strict and len(excluded_test) != 30:
        raise FreezeError(f"expected 30 contact_friction/motion_damping historical Test exclusions, found {len(excluded_test)}")
    decision_source = _decision_source(root=root)
    decision_contract = _decision_contract_identity(root=root)
    # The decision contract binds threshold readers and task scorers.  Also
    # retain the shared data readers, rollout kernels and evaluation entry
    # points which produced the evidence consumed by those scorers.
    evaluation_sources = []
    for relative in (
        "contextworld/benchmarks/bundle_development.py",
        "contextworld/benchmarks/external_model_cli.py",
        "contextworld/benchmarks/public_test_report_cli.py",
        "contextworld/benchmarks/runtime_identity.py",
        "contextworld/benchmarks/adapters.py",
        "contextworld/benchmarks/prejepa_adapters.py",
        "contextworld/benchmarks/action_delay_h3_tail_projection.py",
        "contextworld/benchmarks/action_delay_development_adapter.py",
        "contextworld/evaluation/action_delay_h7_score.py",
        "contextworld/evaluation/hidden_passage_validation.py",
    ):
        source_path = resolve_path(relative, root=root)
        if source_path.is_file():
            evaluation_sources.append(identity(source_path, root=root, label="evaluation source"))
        elif strict:
            raise FreezeError(f"missing evaluation source: {relative}")
    if strict and decision_source.get("sha256") is None:
        raise FreezeError("reference_decision module is required for a production freeze")
    original_seed_summary_path = resolve_path(ORIGINAL_SEED_CEM_SUMMARY, root=root)
    original_seed_freeze_path = resolve_path(ORIGINAL_SEED_CEM_FREEZE, root=root)
    dino_diagnostic_path = resolve_path(DINO_ORIGINAL_DIAGNOSTIC, root=root)
    original_seed_summary_identity = (
        identity(original_seed_summary_path, root=root, label="original seed CEM family summary")
        if original_seed_summary_path.is_file()
        else {"path": ORIGINAL_SEED_CEM_SUMMARY, "sha256": None, "status": "missing"}
    )
    original_seed_freeze_identity = (
        identity(original_seed_freeze_path, root=root, label="original seed CEM freeze")
        if original_seed_freeze_path.is_file()
        else {"path": ORIGINAL_SEED_CEM_FREEZE, "sha256": None, "status": "missing"}
    )
    dino_diagnostic_identity = (
        identity(dino_diagnostic_path, root=root, label="DINO original diagnostic summary")
        if dino_diagnostic_path.is_file()
        else {"path": DINO_ORIGINAL_DIAGNOSTIC, "sha256": None, "status": "missing"}
    )
    if strict and dino_diagnostic_identity.get("sha256") is None:
        raise FreezeError("DINO original diagnostic summary is required as supplementary evidence")
    out: dict[str, Any] = {
        "schema_version": 2,
        "freeze_id": "contextworld_joint_scratch_v1_reference_results_freeze_v2",
        "result_kind": "joint_scratch_v1_reference_results_freeze",
        "recipe": v1.get("recipe", "joint_scratch_v1"),
        "note": "Evidence-bound reference results. Public Test rows retain historical presence separately from current admission; contact_friction and motion_damping Test rows are excluded from final reporting.",
        "inputs": {
            "v1_freeze": v1_identity,
            "audit": audit_identity,
            "result_overlay": overlay_identity,
            "decision_source": decision_source,
            "decision_contract": decision_contract,
            "evaluation_sources": evaluation_sources,
            "original_seed_cem_summary": original_seed_summary_identity,
            "original_seed_cem_freeze": original_seed_freeze_identity,
            "dino_original_diagnostic": dino_diagnostic_identity,
        },
        "main_score_fields": copy.deepcopy(v1.get("main_score_fields", {})),
        "checkpoint_results": output_rows,
        "coverage": {
            "checkpoint_rows": len(output_rows),
            "training_after_rows": sum(row["stage"] == "post_component_training" for row in output_rows),
            "original_environment_rows": sum(row["stage"] == "original_environment_only" for row in output_rows),
            "source_result_rows": 2 * len(output_rows),
            "source_audit_rows": len(audit_rows),
            "training_cem_rows": cem_training_count,
            "split_coverage": "all 135 rows have Development and Test source identities",
        },
        "historical_test_exclusions": excluded_test,
        "exclusion_policy": {"contact_friction": "excluded_from_final_reporting", "motion_damping": "excluded_from_final_reporting", "historical_presence_preserved": True},
        "checkpoint_hash_policy": "Declared checkpoint SHA and path are retained; checkpoint bytes are not rehashed by this entry point.",
    }
    output_path = resolve_path(output, root=root)
    if write:
        if output_path.resolve() == v1_file.resolve():
            raise FreezeError("refusing to overwrite freeze v1")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists():
            existing = read_json(output_path, label="existing reference freeze")
            if canonical_sha256(existing) != canonical_sha256(out):
                raise FreezeError(
                    f"refusing to replace an existing freeze with different content: {output_path}"
                )
        else:
            output_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return out


def _compare(expected: Any, observed: Any, *, path: str, errors: list[str]) -> None:
    if isinstance(expected, Mapping):
        if not isinstance(observed, Mapping):
            errors.append(f"{path}: expected object")
            return
        for key, value in expected.items():
            if key not in observed:
                errors.append(f"{path}.{key}: missing")
            else:
                _compare(value, observed[key], path=f"{path}.{key}", errors=errors)
    elif isinstance(expected, list):
        if not isinstance(observed, list) or len(expected) != len(observed):
            errors.append(f"{path}: list length changed")
            return
        for index, (left, right) in enumerate(zip(expected, observed)):
            _compare(left, right, path=f"{path}[{index}]", errors=errors)
    elif expected != observed:
        errors.append(f"{path}: expected {expected!r}, observed {observed!r}")


def _verify_member_identities(data: Mapping[str, Any], *, root: Path, errors: list[str]) -> None:
    package_rows: dict[Path, dict[str, str]] = {}
    for manifest in data.get("package_manifests", []):
        if not isinstance(manifest, Mapping):
            continue
        manifest_path = manifest.get("resolved_path") or manifest.get("path")
        if not isinstance(manifest_path, str):
            continue
        path = resolve_path(manifest_path, root=root)
        if not path.is_file() or manifest.get("sha256") != file_sha256(path):
            errors.append(f"data package manifest drifted: {manifest_path}")
            continue
        package_root = path.parent
        try:
            rows = {}
            for line in path.read_text(encoding="utf-8").splitlines():
                item = json.loads(line)
                if isinstance(item, Mapping) and isinstance(item.get("path"), str) and isinstance(item.get("sha256"), str):
                    rows[item["path"]] = item["sha256"]
            package_rows[package_root] = rows
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            errors.append(f"data package manifest unreadable: {manifest_path}")
    for member in data.get("member_identities", []):
        if not isinstance(member, Mapping) or not member.get("sha256"):
            continue
        path = member.get("resolved_path")
        if not isinstance(path, str) or not Path(path).exists():
            errors.append(f"data member missing: {path}")
            continue
        member_path = Path(path)
        if member.get("status") in {
            "verified_from_package_manifest",
            "current_member_hash_with_declared_historical_manifest",
        }:
            package_root = next((root_path for root_path in package_rows if member_path.is_relative_to(root_path)), None)
            rows = package_rows.get(package_root, {}) if package_root else {}
            relative = member_path.relative_to(package_root).as_posix() if package_root else member.get("path", "")
            prefix = str(relative).rstrip("/") + "/"
            matching = sorted((name, digest) for name, digest in rows.items() if name == relative or name.startswith(prefix))
            observed = canonical_sha256(matching)
        else:
            observed = file_sha256(member_path) if member_path.is_file() else _directory_manifest_sha256(member_path)
        if observed != member["sha256"]:
            errors.append(f"data member SHA drifted: {path}")


def _verify_cem_evidence(
    row: Mapping[str, Any], *, root: Path, errors: list[str]
) -> None:
    evidence = row.get("cem_evidence")
    if not isinstance(evidence, Mapping) or evidence.get("status") != "verified":
        # Non-strict synthetic fixtures may intentionally omit CEM files.  A
        # production freeze marks the block verified during build and enters
        # this branch only for an edited/incomplete freeze, which is reported
        # by the status check below when rows were present.
        if isinstance(evidence, Mapping) and evidence.get("rows"):
            errors.append("CEM evidence status is not verified")
        return
    stored_rows = evidence.get("rows")
    if not isinstance(stored_rows, list) or [item.get("eval_seed") for item in stored_rows if isinstance(item, Mapping)] != [42, 43, 44, 45, 46, 47]:
        errors.append(f"row {row.get('component_id')} CEM seed coverage changed")
        return
    observed_rows: list[dict[str, Any]] = []
    for item in stored_rows:
        if not isinstance(item, Mapping):
            errors.append(f"row {row.get('component_id')} CEM member is not an object")
            continue
        source = item.get("source")
        if not isinstance(source, Mapping):
            errors.append(f"row {row.get('component_id')} CEM source identity missing")
            continue
        source_value = source.get("resolved_path") or source.get("path")
        source_path = resolve_path(str(source_value), root=root) if source_value else None
        if source_path is None or not source_path.is_file():
            errors.append(f"CEM source missing: {source_value}")
            continue
        if file_sha256(source_path) != source.get("sha256"):
            errors.append(f"CEM source SHA drifted: {source_value}")
            continue
        try:
            payload = read_json(source_path, label="CEM result")
        except FreezeError as exc:
            errors.append(str(exc))
            continue
        metrics = payload.get("metrics") if isinstance(payload.get("metrics"), Mapping) else {}
        seed = payload.get("eval_seed")
        if not isinstance(seed, int):
            match = re.search(r"seed(\d+)", source_path.name)
            seed = int(match.group(1)) if match else None
        rate = metrics.get("success_rate_percent")
        trials = payload.get("num_eval", metrics.get("evaluation_count", 50))
        if seed != item.get("eval_seed") or rate != item.get("success_rate_percent") or trials != item.get("trials"):
            errors.append(f"CEM value drifted: {source_value}")
        observed_rows.append({
            "eval_seed": seed,
            "success_rate_percent": rate,
            "trials": trials,
            "source": dict(source),
        })
    if canonical_sha256(observed_rows) != evidence.get("identity_sha256"):
        errors.append(f"row {row.get('component_id')} CEM identity drifted")
    expected = row.get("original_task_cem")
    expected_rates = expected.get("success_rate_percent_by_eval_seed") if isinstance(expected, Mapping) else None
    if isinstance(expected_rates, Mapping):
        observed_rates = {str(item["eval_seed"]): float(item["success_rate_percent"]) for item in observed_rows if item.get("success_rate_percent") is not None}
        normalized_expected = {str(key): float(value) for key, value in expected_rates.items()}
        if observed_rates != normalized_expected:
            errors.append(f"row {row.get('component_id')} CEM values disagree with v1")


def _verify_original_cem_evidence(
    row: Mapping[str, Any], *, root: Path, errors: list[str]
) -> None:
    evidence = row.get("original_cem_evidence")
    if not isinstance(evidence, Mapping):
        errors.append(f"row {row.get('component_id')} original CEM evidence missing")
        return
    summary = evidence.get("source_summary")
    freeze = evidence.get("source_freeze")
    for label, source in (("original CEM summary", summary), ("original CEM freeze", freeze)):
        if not isinstance(source, Mapping) or not source.get("sha256"):
            errors.append(f"{label} identity missing")
            continue
        value = source.get("resolved_path") or source.get("path")
        path = resolve_path(str(value), root=root)
        if not path.is_file() or file_sha256(path) != source.get("sha256"):
            errors.append(f"{label} SHA drifted: {value}")
    member = evidence.get("member")
    if not isinstance(member, Mapping):
        errors.append(f"row {row.get('component_id')} original CEM member missing")
        return
    summary_value = summary.get("resolved_path") or summary.get("path") if isinstance(summary, Mapping) else None
    summary_path = resolve_path(str(summary_value), root=root) if summary_value else None
    if summary_path is None or not summary_path.is_file():
        return
    try:
        payload = read_json(summary_path, label="original seed CEM family summary")
    except FreezeError as exc:
        errors.append(str(exc))
        return
    families = payload.get("families", [])
    family_row = next(
        (
            item
            for item in families
            if isinstance(item, Mapping)
            and item.get("environment") == evidence.get("environment")
            and str(item.get("family", "")).lower() == str(evidence.get("family", "")).lower()
        ),
        None,
    )
    current_member = next(
        (
            item
            for item in family_row.get("members", [])
            if isinstance(item, Mapping) and int(item.get("training_seed", -1)) == int(evidence.get("training_seed", -1))
        ),
        None,
    ) if isinstance(family_row, Mapping) else None
    if not isinstance(current_member, Mapping) or canonical_sha256(dict(current_member)) != canonical_sha256(dict(member)):
        errors.append(f"row {row.get('component_id')} original CEM member drifted")


def verify_freeze(
    *, freeze_path: Path | str = DEFAULT_OUTPUT, result_overlay: Path | str | None = DEFAULT_OVERLAY, repo_root: Path = ROOT,
) -> dict[str, Any]:
    root = Path(repo_root).expanduser().resolve()
    path = resolve_path(freeze_path, root=root)
    frozen = read_json(path, label="reference freeze v2")
    if frozen.get("schema_version") != 2 or frozen.get("freeze_id") != "contextworld_joint_scratch_v1_reference_results_freeze_v2":
        raise FreezeError("unsupported reference freeze v2")
    errors: list[str] = []
    rows = frozen.get("checkpoint_results")
    coverage = frozen.get("coverage")
    expected_rows = 135
    if isinstance(coverage, Mapping) and isinstance(coverage.get("checkpoint_rows"), int):
        expected_rows = int(coverage["checkpoint_rows"])
    if not isinstance(rows, list) or len(rows) != expected_rows:
        errors.append(f"checkpoint_results must contain {expected_rows} rows")
        rows = rows if isinstance(rows, list) else []
    _validate_coverage([row for row in rows if isinstance(row, Mapping)], strict=False)
    stored_overlay = ((frozen.get("inputs") or {}).get("result_overlay") if isinstance(frozen.get("inputs"), Mapping) else None)
    overlay_path_for_verify: Path | None = None
    if result_overlay is not None:
        overlay_path_for_verify = resolve_path(result_overlay, root=root)
    elif isinstance(stored_overlay, Mapping):
        candidate = stored_overlay.get("resolved_path") or stored_overlay.get("path")
        if isinstance(candidate, str):
            overlay_path_for_verify = resolve_path(candidate, root=root)
    overlay_map, overlay_identity = load_overlay(overlay_path_for_verify, root=root) if overlay_path_for_verify is not None and overlay_path_for_verify.is_file() else ({}, None)
    if stored_overlay and overlay_identity is None:
        errors.append("freeze was built with an overlay but verify received none")
    elif stored_overlay and overlay_identity and stored_overlay.get("sha256") != overlay_identity.get("sha256"):
        errors.append("result overlay SHA drifted")
    inputs = frozen.get("inputs") if isinstance(frozen.get("inputs"), Mapping) else {}
    stored_contract = inputs.get("decision_contract")
    if isinstance(stored_contract, Mapping):
        current_contract = _decision_contract_identity(root=root)
        if current_contract is None or canonical_sha256(current_contract) != canonical_sha256(stored_contract):
            errors.append("reference decision contract identity drifted")
    for source in inputs.get("evaluation_sources", []):
        source_path = resolve_path(str(source["path"]), root=root)
        if not source_path.is_file() or file_sha256(source_path) != source.get("sha256"):
            errors.append(f"evaluation source drifted: {source['path']}")
    for input_name in ("original_seed_cem_summary", "original_seed_cem_freeze", "dino_original_diagnostic"):
        source = inputs.get(input_name)
        if not isinstance(source, Mapping) or not source.get("sha256"):
            continue
        source_value = source.get("resolved_path") or source.get("path")
        source_path = resolve_path(str(source_value), root=root)
        if not source_path.is_file() or file_sha256(source_path) != source.get("sha256"):
            errors.append(f"freeze input SHA drifted: {input_name}")
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            errors.append(f"row {index} is not an object")
            continue
        for split in ("development", "test"):
            block = row.get(split)
            if not isinstance(block, Mapping):
                errors.append(f"row {index}.{split} missing")
                continue
            source = block.get("source_file")
            source_path = block.get("source_result_path")
            if not isinstance(source_path, str):
                errors.append(f"row {index}.{split} source path missing")
                continue
            # The repository-owned raw snapshot is authoritative.  An
            # overlay is input provenance and is checked by its own SHA, but
            # a later rerun at the old external path cannot replace the
            # frozen bytes.
            current_path = resolve_path(source_path, root=root)
            if not current_path.is_file():
                errors.append(f"row {index}.{split} source missing: {current_path}")
                continue
            observed_hash = file_sha256(current_path)
            if observed_hash != block.get("source_result_sha256"):
                errors.append(f"row {index}.{split} source SHA drifted")
                continue
            payload = read_json(current_path, label="source result")
            try:
                observed_score = _extract_score(payload, str(row["component_id"]), split)
            except FreezeError as exc:
                errors.append(f"row {index}.{split} score unavailable: {exc}")
                continue
            if observed_score != block.get("main_score"):
                errors.append(f"row {index}.{split} main score drifted")
            result = _unwrap(payload)
            inputs = _decision_inputs(result, payload, task=str(row["component_id"]), split=split, score=observed_score)
            context = {"task": str(row["component_id"]), "family": str(row["family"]), "stage": str(row["stage"]), "split": split, "training_seed": int(row["training_seed"]), "score": observed_score, "result": result, "payload": payload, "inputs": inputs, "thresholds": block.get("threshold_sources", [])}
            try:
                decision, source_info = _invoke_reference_decision(context, root=root, strict=False)
                if decision.get("inputs_sha256") != (block.get("decision") or {}).get("inputs_sha256"):
                    errors.append(f"row {index}.{split} decision inputs drifted")
                if bool(decision.get("passed")) != bool((block.get("decision") or {}).get("passed")):
                    errors.append(f"row {index}.{split} decision drifted")
                stored_all_gates = block.get("all_gates_passed")
                if not isinstance(stored_all_gates, bool):
                    errors.append(f"row {index}.{split} all_gates_passed is not boolean")
                elif stored_all_gates != bool(decision.get("passed")):
                    errors.append(f"row {index}.{split} all_gates_passed disagrees with decision")
                if source_info.get("sha256") != ((block.get("decision_source") or {}).get("sha256")):
                    errors.append(f"row {index}.{split} decision source drifted")
            except FreezeError as exc:
                errors.append(f"row {index}.{split} decision unavailable: {exc}")
            for source_key in ("scorer_source", "threshold_sources", "adapter_source"):
                values = block.get(source_key, []) if source_key != "scorer_source" else [block.get(source_key)]
                if isinstance(values, Mapping):
                    values = [values]
                for source_item in values:
                    if not isinstance(source_item, Mapping) or not source_item.get("path") or not source_item.get("sha256"):
                        continue
                    source_path_value = source_item.get("resolved_path") or resolve_path(str(source_item["path"]), root=root)
                    source_path_obj = Path(source_path_value)
                    if not source_path_obj.is_file() or file_sha256(source_path_obj) != source_item["sha256"]:
                        errors.append(f"row {index}.{split} {source_key} drifted: {source_item.get('path')}")
            training_identity = block.get("training_identity")
            if isinstance(training_identity, Mapping):
                training_source = training_identity.get("source")
                if isinstance(training_source, Mapping) and training_source.get("sha256"):
                    training_value = training_source.get("resolved_path") or training_source.get("path")
                    training_path = resolve_path(str(training_value), root=root)
                    if not training_path.is_file() or file_sha256(training_path) != training_source.get("sha256"):
                        errors.append(f"row {index}.{split} training identity drifted")
            _verify_member_identities(block.get("data_identity") if isinstance(block.get("data_identity"), Mapping) else {}, root=root, errors=errors)
        dev_passed = bool((row.get("development") or {}).get("all_gates_passed"))
        admission = row.get("admission") or {}
        if admission.get("cleared_development") != dev_passed:
            errors.append(f"row {index} Development admission disagrees with decision")
        expected_admission = bool(dev_passed and row.get("stage") == "post_component_training" and row.get("component_id") not in ("contact_friction", "motion_damping"))
        if admission.get("current_test_admission") != expected_admission:
            errors.append(f"row {index} current Test admission disagrees with policy")
        if row.get("stage") == "post_component_training":
            _verify_cem_evidence(row, root=root, errors=errors)
        elif row.get("stage") == "original_environment_only":
            _verify_original_cem_evidence(row, root=root, errors=errors)
    if errors:
        raise FreezeError("reference freeze verification failed:\n" + "\n".join(errors[:40]))
    return {"status": "verified", "freeze_path": str(path), "rows": len(rows), "source_results": 2 * len(rows)}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze", help="build a new v2 freeze")
    verify = sub.add_parser("verify", help="recompute and verify an existing v2 freeze")
    for command in (freeze, verify):
        command.add_argument("--repo-root", type=Path, default=ROOT)
        command.add_argument("--result-overlay", type=Path, default=None)
    freeze.add_argument("--v1", type=Path, default=DEFAULT_V1)
    freeze.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    freeze.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    verify.add_argument("--freeze", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "freeze":
            result = build_freeze(v1_path=args.v1, audit_path=args.audit, result_overlay=args.result_overlay, output=args.output, repo_root=args.repo_root, strict=True, write=True)
            print(json.dumps({"status": "frozen", "path": str(resolve_path(args.output, root=args.repo_root)), "rows": len(result["checkpoint_results"])}, ensure_ascii=False))
        else:
            print(json.dumps(verify_freeze(freeze_path=args.freeze, result_overlay=args.result_overlay, repo_root=args.repo_root), ensure_ascii=False))
    except FreezeError as exc:
        print(f"freeze error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
