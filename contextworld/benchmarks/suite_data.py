from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Iterable

import yaml

from contextworld.benchmarks.door_icl_data import (
    audit_door_icl_release,
    load_door_icl_release,
)
from contextworld.benchmarks.speed_icl_data import (
    audit_speed_icl_release,
    load_speed_icl_release,
    resolve_original_h5,
)
from contextworld.paths import repository_root, resolve_contextworld_path


SUITE_RELEASE_ID = "contextworld_icl_benchmark_suite_v1"
DEFAULT_SUITE_RELEASE_CONFIG = (
    repository_root()
    / "configs/benchmark/contextworld_icl_suite_v1.yaml"
)
COMPONENT_IDS = ("speed", "door")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_icl_suite_release(
    path: Path | str = DEFAULT_SUITE_RELEASE_CONFIG,
) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError(f"Unsupported ICL suite config: {config_path}")
    if payload.get("release_id") != SUITE_RELEASE_ID:
        raise ValueError(f"Unexpected ICL suite release id: {config_path}")
    if payload.get("release_status") not in {
        "validation_release_candidate",
        "validation_release",
    }:
        raise ValueError("Unsupported ICL suite release status")
    if payload.get("scope", {}).get("sealed_test_included") is not False:
        raise ValueError("The public ICL suite must not contain sealed Test")
    components = payload.get("components")
    if not isinstance(components, dict):
        raise ValueError("ICL suite components must be a mapping")
    if tuple(components) != COMPONENT_IDS:
        raise ValueError(
            f"ICL suite components must be ordered as {COMPONENT_IDS}"
        )
    for component_id, component in components.items():
        if component.get("release_id") is None:
            raise ValueError(f"{component_id} is missing release_id")
        expected = str(component.get("release_config_sha256", ""))
        if len(expected) != 64:
            raise ValueError(
                f"{component_id} release hash has not been frozen"
            )
    for logical_path, expected in payload["repository"][
        "source_sha256"
    ].items():
        if len(str(expected)) != 64:
            raise ValueError(f"Source hash is not frozen: {logical_path}")
    document_hash = str(
        payload["repository"]["public_document"]["sha256"]
    )
    if len(document_hash) != 64:
        raise ValueError("Public document hash has not been frozen")
    return {**payload, "_config_path": str(config_path)}


def _audit_file(
    logical_path: str,
    expected_sha256: str,
    *,
    repo_root: Path,
) -> dict[str, Any]:
    path = resolve_contextworld_path(logical_path, repo_root=repo_root)
    exists = path.is_file()
    observed = _sha256(path) if exists else None
    return {
        "logical_path": logical_path,
        "path": str(path),
        "exists": exists,
        "expected_sha256": expected_sha256,
        "observed_sha256": observed,
        "passed": bool(exists and observed == expected_sha256),
    }


def _bundled_component_release(
    suite: dict[str, Any],
    component_id: str,
    *,
    repo_root: Path,
) -> Path:
    suite_path = Path(suite["_config_path"])
    bundled = suite_path.parent / "releases" / f"{component_id}.yaml"
    if bundled.is_file():
        return bundled
    logical = suite["components"][component_id]["release_config"]
    return resolve_contextworld_path(logical, repo_root=repo_root)


def _bundled_readme(suite: dict[str, Any], *, repo_root: Path) -> Path:
    suite_path = Path(suite["_config_path"])
    candidate = suite_path.parent.parent / "README.md"
    if candidate.is_file():
        return candidate
    logical = suite["repository"]["public_document"]["path"]
    return resolve_contextworld_path(logical, repo_root=repo_root)


def audit_icl_suite_release(
    *,
    release_config: Path | str = DEFAULT_SUITE_RELEASE_CONFIG,
    repo_root: Path | None = None,
    components: Iterable[str] | None = None,
    full: bool = False,
    original_h5: Path | str | None = None,
) -> dict[str, Any]:
    root = (repo_root or repository_root()).resolve()
    suite = load_icl_suite_release(release_config)
    selected = tuple(COMPONENT_IDS if components is None else components)
    if not selected or not set(selected).issubset(COMPONENT_IDS):
        raise ValueError(
            f"components must be a non-empty subset of {COMPONENT_IDS}"
        )
    if len(selected) != len(set(selected)):
        raise ValueError("components must be unique")

    code_audits = [
        _audit_file(logical, expected, repo_root=root)
        for logical, expected in suite["repository"][
            "source_sha256"
        ].items()
    ]
    document_path = _bundled_readme(suite, repo_root=root)
    expected_document_hash = suite["repository"]["public_document"]["sha256"]
    document_audit = {
        "path": str(document_path),
        "exists": document_path.is_file(),
        "expected_sha256": expected_document_hash,
        "observed_sha256": (
            _sha256(document_path) if document_path.is_file() else None
        ),
    }
    document_audit["passed"] = bool(
        document_audit["exists"]
        and document_audit["observed_sha256"] == expected_document_hash
    )

    component_audits: dict[str, Any] = {}
    component_release_audits: dict[str, Any] = {}
    for component_id in selected:
        component = suite["components"][component_id]
        component_path = _bundled_component_release(
            suite,
            component_id,
            repo_root=root,
        )
        release_exists = component_path.is_file()
        observed_hash = _sha256(component_path) if release_exists else None
        release_audit = {
            "path": str(component_path),
            "exists": release_exists,
            "expected_sha256": component["release_config_sha256"],
            "observed_sha256": observed_hash,
            "passed": bool(
                release_exists
                and observed_hash == component["release_config_sha256"]
            ),
        }
        component_release_audits[component_id] = release_audit
        if not release_audit["passed"]:
            component_audits[component_id] = {
                "status": "failed",
                "passed": False,
                "reason": "component release config audit failed",
            }
            continue
        if component_id == "speed":
            speed_release = load_speed_icl_release(component_path)
            resolved_original = original_h5
            if resolved_original is None:
                suite_path = Path(suite["_config_path"])
                bundled_original = (
                    suite_path.parent
                    / "upstream/lewm-tworooms/tworoom.h5"
                )
                if bundled_original.is_file():
                    resolved_original = bundled_original
            component_audits[component_id] = audit_speed_icl_release(
                release_config=component_path,
                repo_root=root,
                original_h5=resolved_original,
                verify_all_eval_payloads=full,
            )
            if (
                component_audits[component_id]["release_id"]
                != speed_release["release_id"]
            ):
                component_audits[component_id]["passed"] = False
        else:
            door_release = load_door_icl_release(component_path)
            component_audits[component_id] = audit_door_icl_release(
                release_config=component_path,
                repo_root=root,
                full=full,
            )
            if (
                component_audits[component_id]["release_id"]
                != door_release["release_id"]
            ):
                component_audits[component_id]["passed"] = False

    technical_passed = bool(
        all(row["passed"] for row in code_audits)
        and document_audit["passed"]
        and all(
            row["passed"] for row in component_release_audits.values()
        )
        and all(row.get("passed") is True for row in component_audits.values())
    )
    distribution = suite["distribution"]
    public_ready = bool(
        technical_passed
        and distribution["code_license_status"] == "declared"
        and distribution["generated_data_license_status"] == "declared"
        and distribution["public_download_status"] == "configured"
    )
    return {
        "schema_version": 1,
        "release_id": suite["release_id"],
        "status": "passed" if technical_passed else "failed",
        "release_config": suite["_config_path"],
        "artifact_root_override": os.environ.get(
            "CONTEXTWORLD_ARTIFACT_ROOT"
        ),
        "selected_components": list(selected),
        "full_content_hash_audit": full,
        "sealed_test_included": False,
        "repository_code": code_audits,
        "public_document": document_audit,
        "component_release_configs": component_release_audits,
        "components": component_audits,
        "technical_release_candidate_passed": technical_passed,
        "public_distribution_ready": public_ready,
        "distribution_blockers": [
            label
            for label, passed in (
                (
                    "ContextWorld source license declaration",
                    distribution["code_license_status"] == "declared",
                ),
                (
                    "ContextWorld-generated data license declaration",
                    distribution["generated_data_license_status"]
                    == "declared",
                ),
                (
                    "public artifact download URL",
                    distribution["public_download_status"] == "configured",
                ),
            )
            if not passed
        ],
        "passed": technical_passed,
    }


def _speed_export_entries(release: dict[str, Any]) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    for row in release["training"]["synthetic"].values():
        entries.append((row["data_root"], "directory"))
        for key in ("catalog", "manifest", "report"):
            entries.append((row[key], "file"))
    entries.append((release["evaluation"]["normalizer"], "file"))
    entries.extend(
        [
            (
                "artifacts/evaluation/history3/"
                "speed_multistep_extrap_v5/catalogs",
                "directory",
            ),
            (
                "artifacts/evaluation/history3/"
                "speed_multistep_extrap_v5/payloads",
                "directory",
            ),
        ]
    )
    if release.get("planning"):
        entries.extend(
            [
                (
                    "artifacts/evaluation/history3/"
                    "speed_isolated_v2/catalogs",
                    "directory",
                ),
                (
                    "artifacts/evaluation/history3/"
                    "speed_isolated_v2/payloads",
                    "directory",
                ),
            ]
        )
    return entries


def _door_export_entries(release: dict[str, Any]) -> list[tuple[str, str]]:
    entries = [
        (release["training"]["artifact_tree"]["root"], "directory"),
        (release["evaluation"]["artifact_tree"]["root"], "directory"),
        (release["evaluation"]["normalizer"], "file"),
        (release["training"]["initialization"]["checkpoint"], "file"),
        (
            release["training"]["initialization"]["checkpoint_config"],
            "file",
        ),
    ]
    entries.extend(
        (row["root"], "directory")
        for row in release["reference_results"].values()
    )
    return entries


def _artifact_target(benchmark_root: Path, logical_path: str) -> Path:
    relative = Path(logical_path)
    if not relative.parts or relative.parts[0] != "artifacts":
        raise ValueError(f"Expected artifacts/... path, got {logical_path}")
    return benchmark_root.joinpath(*relative.parts[1:])


def export_icl_suite_artifacts(
    destination: Path | str,
    *,
    release_config: Path | str = DEFAULT_SUITE_RELEASE_CONFIG,
    repo_root: Path | None = None,
    mode: str = "copy",
    include_upstream_original: bool = True,
) -> dict[str, Any]:
    """Export one README plus one integrated Speed-and-Door data directory."""

    if mode not in {"copy", "symlink"}:
        raise ValueError("Export mode must be 'copy' or 'symlink'")
    root = (repo_root or repository_root()).resolve()
    suite = load_icl_suite_release(release_config)
    destination = Path(destination).expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Export destination is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    benchmark_root = destination / "benchmark"
    benchmark_root.mkdir()

    speed_config = resolve_contextworld_path(
        suite["components"]["speed"]["release_config"],
        repo_root=root,
    )
    door_config = resolve_contextworld_path(
        suite["components"]["door"]["release_config"],
        repo_root=root,
    )
    speed_release = load_speed_icl_release(speed_config)
    door_release = load_door_icl_release(door_config)

    entries = (
        _speed_export_entries(speed_release)
        + _door_export_entries(door_release)
    )
    seen: set[str] = set()
    inventory_entries: list[dict[str, Any]] = []
    for logical_path, kind in entries:
        if logical_path in seen:
            continue
        seen.add(logical_path)
        source = resolve_contextworld_path(logical_path, repo_root=root)
        target = _artifact_target(benchmark_root, logical_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if kind == "file":
            if not source.is_file():
                raise FileNotFoundError(source)
            if mode == "symlink":
                target.symlink_to(source)
            else:
                shutil.copy2(source, target)
            inventory_entries.append(
                {
                    "logical_path": logical_path,
                    "kind": kind,
                    "bytes": source.stat().st_size,
                    "sha256": _sha256(source),
                }
            )
        else:
            if not source.is_dir():
                raise FileNotFoundError(source)
            if mode == "symlink":
                target.symlink_to(source, target_is_directory=True)
            else:
                shutil.copytree(source, target)
            inventory_entries.append(
                {
                    "logical_path": logical_path,
                    "kind": kind,
                    "export_mode": mode,
                }
            )

    original_entry: dict[str, Any] | None = None
    if include_upstream_original:
        original_source = resolve_original_h5(speed_release, repo_root=root)
        if not original_source.is_file():
            raise FileNotFoundError(original_source)
        original_target = (
            benchmark_root / "upstream/lewm-tworooms/tworoom.h5"
        )
        original_target.parent.mkdir(parents=True, exist_ok=True)
        if mode == "symlink":
            original_target.symlink_to(original_source)
        else:
            shutil.copy2(original_source, original_target)
        original_entry = {
            "logical_path": "upstream/lewm-tworooms/tworoom.h5",
            "kind": "file",
            "bytes": original_source.stat().st_size,
            "sha256": _sha256(original_source),
            "source": speed_release["training"]["original"]["source"],
            "license_reported_upstream": speed_release["training"][
                "original"
            ]["license"],
        }
        inventory_entries.append(original_entry)

    releases_dir = benchmark_root / "releases"
    releases_dir.mkdir()
    shutil.copy2(speed_config, releases_dir / "speed.yaml")
    shutil.copy2(door_config, releases_dir / "door.yaml")
    shutil.copy2(Path(suite["_config_path"]), benchmark_root / "suite.yaml")
    document = resolve_contextworld_path(
        suite["repository"]["public_document"]["path"],
        repo_root=root,
    )
    shutil.copy2(document, destination / "README.md")

    payload = {
        "schema_version": 1,
        "release_id": suite["release_id"],
        "status": "passed",
        "release_kind": "local_technical_release_candidate",
        "mode": mode,
        "top_level_entries": ["README.md", "benchmark"],
        "benchmark_root": "benchmark",
        "suite_config": "benchmark/suite.yaml",
        "component_release_configs": {
            "speed": "benchmark/releases/speed.yaml",
            "door": "benchmark/releases/door.yaml",
        },
        "components": ["speed", "door"],
        "includes_upstream_original_h5": include_upstream_original,
        "sealed_test_included": False,
        "redistribution_granted_by_export": False,
        "distribution": suite["distribution"],
        "entries": inventory_entries,
    }
    inventory_path = benchmark_root / "inventory.json"
    inventory_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        **payload,
        "destination": str(destination),
        "readme": str(destination / "README.md"),
        "benchmark_root_path": str(benchmark_root),
        "suite_config_path": str(benchmark_root / "suite.yaml"),
        "inventory": str(inventory_path),
        "upstream_original": (
            str(benchmark_root / "upstream/lewm-tworooms/tworoom.h5")
            if original_entry is not None
            else None
        ),
    }


__all__ = [
    "COMPONENT_IDS",
    "DEFAULT_SUITE_RELEASE_CONFIG",
    "SUITE_RELEASE_ID",
    "audit_icl_suite_release",
    "export_icl_suite_artifacts",
    "load_icl_suite_release",
]
