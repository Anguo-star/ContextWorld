from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Iterable

import yaml

from contextworld.benchmarks.action_delay_icl_data import (
    audit_action_delay_icl_release,
    load_action_delay_icl_release,
)
from contextworld.benchmarks.action_strength_icl_data import (
    audit_action_strength_icl_release,
    load_action_strength_icl_release,
    resolve_action_strength_initial_checkpoint,
    resolve_action_strength_original_h5,
    resolve_action_strength_original_lance,
)
from contextworld.benchmarks.contact_friction_icl_data import (
    audit_contact_friction_icl_release,
    load_contact_friction_icl_release,
)
from contextworld.benchmarks.motion_damping_icl_data import (
    audit_motion_damping_icl_release,
    load_motion_damping_icl_release,
)
from contextworld.benchmarks.portal_exit_icl_data import (
    audit_portal_exit_icl_release,
    load_portal_exit_icl_release,
    resolve_portal_original_lance,
)
from contextworld.benchmarks.public_score import (
    make_public_scoreboard_from_spec,
)
from contextworld.benchmarks.reacher_arm_mass_icl_data import (
    audit_reacher_arm_mass_icl_release,
    load_reacher_arm_mass_icl_release,
    resolve_reacher_initial_checkpoint,
    resolve_reacher_initial_checkpoint_config,
    resolve_reacher_original_h5,
    resolve_reacher_original_lance,
)
from contextworld.benchmarks.door_icl_data import (
    audit_door_icl_release,
    door_icl_export_entries,
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
COMPONENT_IDS = (
    "speed",
    "door",
    "action_delay",
    "action_strength",
    "contact_friction",
    "motion_damping",
    "robot_arm_mass",
    "portal_exit",
)
REFERENCE_RESULT_STATUSES = {
    "speed": "passed_public_test_3_of_3",
    "door": "passed_public_test_3_of_3",
    "action_delay": "passed_public_test_3_of_3",
    "action_strength": "passed_public_test_3_of_3",
    "contact_friction": "failed_development",
    "motion_damping": "failed_development",
    "robot_arm_mass": "passed_public_test_3_of_3",
    "portal_exit": "failed_public_test_0_of_3",
}


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
        "public_test_release_candidate",
        "public_test_release",
    }:
        raise ValueError("Unsupported ICL suite release status")
    scope = payload.get("scope", {})
    if scope.get("public_test_included") is not True:
        raise ValueError("The public ICL suite must include Public Test")
    if scope.get("sealed_test_included") is not False:
        raise ValueError("The public ICL suite must not contain sealed Test")
    model_interface = payload.get("model_interface", {})
    if (
        model_interface.get("primary_model_type") != "latent_world_model"
        or model_interface.get("decoder_required") is not False
        or model_interface.get(
            "raw_latent_loss_cross_model_comparison_allowed"
        )
        is not False
    ):
        raise ValueError(
            "The Suite must use the decoder-free latent-world-model contract"
        )
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
        if component.get("benchmark_component_status") != "ready":
            raise ValueError(
                f"{component_id} benchmark component is not ready"
            )
        if (
            component.get("reference_result_status")
            != REFERENCE_RESULT_STATUSES[component_id]
        ):
            raise ValueError(
                f"{component_id} reference result status is not frozen"
            )
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
    public_results = payload.get("public_results")
    if not isinstance(public_results, dict):
        raise ValueError("Suite public_results must be registered")
    for key in ("specification", "scoreboard"):
        specification = public_results.get(key)
        if (
            not isinstance(specification, dict)
            or not str(specification.get("path", "")).startswith(
                "artifacts/"
            )
            or len(str(specification.get("sha256", ""))) != 64
        ):
            raise ValueError(f"Public result {key!r} is not frozen")
    if int(public_results.get("formal_reference_rows", 0)) <= 0:
        raise ValueError("Suite public result row count must be positive")
    formal_components = public_results.get(
        "components_with_formal_results"
    )
    if (
        not isinstance(formal_components, list)
        or not formal_components
        or not set(formal_components).issubset(COMPONENT_IDS)
    ):
        raise ValueError("Invalid components_with_formal_results")
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


def _bundled_public_result(
    suite: dict[str, Any],
    key: str,
    *,
    repo_root: Path,
) -> Path:
    """Resolve a Suite-level result in both source and exported layouts."""

    specification = suite["public_results"][key]
    logical = Path(str(specification["path"]))
    suite_path = Path(suite["_config_path"])
    if logical.parts and logical.parts[0] == "artifacts":
        bundled = suite_path.parent.joinpath(*logical.parts[1:])
        if bundled.is_file():
            return bundled
    return resolve_contextworld_path(logical, repo_root=repo_root)


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _audit_public_results(
    suite: dict[str, Any],
    *,
    repo_root: Path,
) -> dict[str, Any]:
    """Verify the compact public table against its formal seed-level spec."""

    rows: dict[str, Any] = {}
    for key in ("specification", "scoreboard"):
        specification = suite["public_results"][key]
        path = _bundled_public_result(suite, key, repo_root=repo_root)
        exists = path.is_file()
        observed = _sha256(path) if exists else None
        rows[key] = {
            "path": str(path),
            "exists": exists,
            "expected_sha256": specification["sha256"],
            "observed_sha256": observed,
            "passed": bool(
                exists and observed == specification["sha256"]
            ),
        }

    reproduction: dict[str, Any]
    if all(row["passed"] for row in rows.values()):
        try:
            source = _read_json_object(
                Path(rows["specification"]["path"])
            )
            observed_scoreboard = _read_json_object(
                Path(rows["scoreboard"]["path"])
            )
            expected_scoreboard = make_public_scoreboard_from_spec(source)
            result_rows = observed_scoreboard.get("component_results", [])
            expected_components = set(
                suite["public_results"]["components_with_formal_results"]
            )
            observed_components = {
                row.get("component_id")
                for row in result_rows
                if isinstance(row, dict)
            }
            reproduction = {
                "scoreboard_exactly_reproduced": (
                    observed_scoreboard == expected_scoreboard
                ),
                "expected_reference_rows": suite["public_results"][
                    "formal_reference_rows"
                ],
                "observed_reference_rows": len(result_rows),
                "expected_components": sorted(expected_components),
                "observed_components": sorted(observed_components),
            }
            reproduction["passed"] = bool(
                reproduction["scoreboard_exactly_reproduced"]
                and reproduction["observed_reference_rows"]
                == reproduction["expected_reference_rows"]
                and observed_components == expected_components
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            reproduction = {"passed": False, "error": str(error)}
    else:
        reproduction = {
            "passed": False,
            "error": "public result file hash audit failed",
        }
    return {
        "files": rows,
        "reproduction": reproduction,
        "passed": bool(
            all(row["passed"] for row in rows.values())
            and reproduction["passed"]
        ),
    }


def load_public_scoreboard(
    release_config: Path | str = DEFAULT_SUITE_RELEASE_CONFIG,
    *,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Load the frozen compact result table and reject stale metadata."""

    root = (repo_root or repository_root()).resolve()
    suite = load_icl_suite_release(release_config)
    audit = _audit_public_results(suite, repo_root=root)
    if not audit["passed"]:
        raise RuntimeError("Public scoreboard audit failed")
    return _read_json_object(Path(audit["files"]["scoreboard"]["path"]))


def _audit_public_document_template(
    document_path: Path,
    suite: dict[str, Any],
) -> dict[str, Any]:
    template = suite["extension"]["public_document_template"]
    expected_subsections = list(template["subsections"])
    section_titles = template["component_sections"]
    lines = document_path.read_text(encoding="utf-8").splitlines()
    observed: dict[str, list[str]] = {}
    for component_id in COMPONENT_IDS:
        section_heading = f"### {section_titles[component_id]}"
        try:
            start = lines.index(section_heading) + 1
        except ValueError:
            observed[component_id] = []
            continue
        end = next(
            (
                index
                for index in range(start, len(lines))
                if lines[index].startswith("### ")
            ),
            len(lines),
        )
        observed[component_id] = [
            line.removeprefix("#### ")
            for line in lines[start:end]
            if line.startswith("#### ")
        ]
    return {
        "expected_subsections": expected_subsections,
        "observed_subsections": observed,
        "passed": all(
            observed[component_id] == expected_subsections
            for component_id in COMPONENT_IDS
        ),
    }


def _assert_frozen_export_inputs(
    suite: dict[str, Any],
    *,
    repo_root: Path,
) -> None:
    """Reject a bundle when the frozen public entry points are stale.

    Component content audits remain available through ``audit``.  Export has
    a smaller fail-closed gate of its own so it cannot silently copy a changed
    document, source file, or component release YAML under an old Suite hash.
    """

    failures: list[str] = []
    repository = suite.get("repository", {})
    for logical, expected in repository.get("source_sha256", {}).items():
        result = _audit_file(logical, str(expected), repo_root=repo_root)
        if not result["passed"]:
            failures.append(
                f"source {logical}: {result['observed_sha256']} != {expected}"
            )

    document = repository.get("public_document")
    if isinstance(document, dict) and document.get("sha256"):
        path = _bundled_readme(suite, repo_root=repo_root)
        observed = _sha256(path) if path.is_file() else None
        if observed != document["sha256"]:
            failures.append(
                f"public document {path}: {observed} != {document['sha256']}"
            )
        elif not _audit_public_document_template(path, suite)["passed"]:
            failures.append("public document component template is invalid")

    for component_id, component in suite.get("components", {}).items():
        expected = component.get("release_config_sha256")
        if not expected:
            continue
        path = _bundled_component_release(
            suite,
            component_id,
            repo_root=repo_root,
        )
        observed = _sha256(path) if path.is_file() else None
        if observed != expected:
            failures.append(
                f"component {component_id}: {observed} != {expected}"
            )

    if "public_results" in suite:
        public_results = _audit_public_results(suite, repo_root=repo_root)
        if not public_results["passed"]:
            failures.append("public scoreboard is stale or not reproducible")

    if failures:
        raise RuntimeError(
            "Suite export inputs are not frozen:\n- " + "\n- ".join(failures)
        )


_PORTABLE_TEXT_SUFFIXES = {
    ".csv",
    ".json",
    ".jsonl",
    ".log",
    ".md",
    ".txt",
    ".yaml",
    ".yml",
}
_NON_PORTABLE_MARKERS = (
    "/opt/",
    "/tmp/",
    "/home/",
    "/root/",
    "../../data/",
    "\\Users\\",
)


def _artifact_text_files(path: Path, *, kind: str) -> Iterable[Path]:
    candidates = (path,) if kind == "file" else path.rglob("*")
    for candidate in candidates:
        if (
            candidate.is_file()
            and candidate.suffix.lower() in _PORTABLE_TEXT_SUFFIXES
        ):
            yield candidate


def _assert_portable_export_entries(
    entries: Iterable[tuple[str, str]],
    *,
    repo_root: Path,
) -> None:
    """Reject machine-specific paths in files copied into the public bundle."""

    violations: list[str] = []
    seen: set[str] = set()
    for logical_path, kind in entries:
        if logical_path in seen:
            continue
        seen.add(logical_path)
        source = resolve_contextworld_path(logical_path, repo_root=repo_root)
        if kind == "file" and not source.is_file():
            raise FileNotFoundError(source)
        if kind == "directory" and not source.is_dir():
            raise FileNotFoundError(source)
        for path in _artifact_text_files(source, kind=kind):
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            markers = [
                marker for marker in _NON_PORTABLE_MARKERS if marker in text
            ]
            if markers:
                violations.append(
                    f"{logical_path}:{path.relative_to(source) if kind == 'directory' else path.name} "
                    f"contains {', '.join(markers)}"
                )
    if violations:
        raise RuntimeError(
            "Suite export contains machine-specific paths:\n- "
            + "\n- ".join(violations)
        )


def _assert_portable_source_files(
    files: Iterable[tuple[str, Path]],
) -> None:
    """Apply the public-path policy to files copied outside artifact entries."""

    violations: list[str] = []
    for label, path in files:
        if not path.is_file():
            raise FileNotFoundError(path)
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        markers = [
            marker for marker in _NON_PORTABLE_MARKERS if marker in content
        ]
        if markers:
            violations.append(f"{label} contains {', '.join(markers)}")
    if violations:
        raise RuntimeError(
            "Suite export contains machine-specific paths:\n- "
            + "\n- ".join(violations)
        )


def _enforce_component_causal_gate(
    component_audit: dict[str, Any],
) -> dict[str, bool]:
    """Require every Suite component to publish the same causal-data proof."""

    causal_contract = component_audit.get("causal_data_contract")
    causal_gate = {
        "present": isinstance(causal_contract, dict),
        "passed": bool(
            isinstance(causal_contract, dict)
            and causal_contract.get("passed") is True
        ),
    }
    component_audit["suite_causal_data_gate"] = causal_gate
    if not causal_gate["passed"]:
        component_audit["passed"] = False
        component_audit["status"] = "failed"
        component_audit["reason"] = (
            "component is missing a passed causal data contract"
        )
    return causal_gate


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
    document_audit["component_template"] = (
        _audit_public_document_template(document_path, suite)
        if document_path.is_file()
        else {"passed": False}
    )
    document_audit["passed"] = bool(
        document_audit["exists"]
        and document_audit["observed_sha256"] == expected_document_hash
        and document_audit["component_template"]["passed"]
    )
    public_results_audit = _audit_public_results(suite, repo_root=root)

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
        elif component_id == "door":
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
        elif component_id == "action_delay":
            action_delay_release = load_action_delay_icl_release(
                component_path
            )
            component_audits[
                component_id
            ] = audit_action_delay_icl_release(
                release_config=component_path,
                repo_root=root,
                full=full,
            )
            if (
                component_audits[component_id]["release_id"]
                != action_delay_release["release_id"]
            ):
                component_audits[component_id]["passed"] = False
        elif component_id == "action_strength":
            action_strength_release = load_action_strength_icl_release(
                component_path
            )
            component_audits[
                component_id
            ] = audit_action_strength_icl_release(
                release_config=component_path,
                repo_root=root,
                full=full,
            )
            if (
                component_audits[component_id]["release_id"]
                != action_strength_release["release_id"]
            ):
                component_audits[component_id]["passed"] = False
        elif component_id == "contact_friction":
            contact_friction_release = load_contact_friction_icl_release(
                component_path
            )
            component_audits[
                component_id
            ] = audit_contact_friction_icl_release(
                release_config=component_path,
                repo_root=root,
                full=full,
            )
            if (
                component_audits[component_id]["release_id"]
                != contact_friction_release["release_id"]
            ):
                component_audits[component_id]["passed"] = False
        elif component_id == "motion_damping":
            motion_damping_release = load_motion_damping_icl_release(
                component_path
            )
            component_audits[
                component_id
            ] = audit_motion_damping_icl_release(
                release_config=component_path,
                repo_root=root,
                full=full,
            )
            if (
                component_audits[component_id]["release_id"]
                != motion_damping_release["release_id"]
            ):
                component_audits[component_id]["passed"] = False
        elif component_id == "robot_arm_mass":
            robot_arm_mass_release = load_reacher_arm_mass_icl_release(
                component_path
            )
            component_audits[
                component_id
            ] = audit_reacher_arm_mass_icl_release(
                release_config=component_path,
                repo_root=root,
                full=full,
            )
            if (
                component_audits[component_id]["release_id"]
                != robot_arm_mass_release["release_id"]
            ):
                component_audits[component_id]["passed"] = False
        elif component_id == "portal_exit":
            portal_exit_release = load_portal_exit_icl_release(
                component_path
            )
            component_audits[component_id] = audit_portal_exit_icl_release(
                release_config=component_path,
                repo_root=root,
                full=full,
            )
            if (
                component_audits[component_id]["release_id"]
                != portal_exit_release["release_id"]
            ):
                component_audits[component_id]["passed"] = False
        else:  # pragma: no cover - selected ids are validated above
            raise AssertionError(f"Unhandled benchmark component: {component_id}")

        _enforce_component_causal_gate(component_audits[component_id])

    technical_passed = bool(
        all(row["passed"] for row in code_audits)
        and document_audit["passed"]
        and public_results_audit["passed"]
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
        "public_test_included": True,
        "sealed_test_included": False,
        "repository_code": code_audits,
        "public_document": document_audit,
        "public_results": public_results_audit,
        "component_release_configs": component_release_audits,
        "components": component_audits,
        "causal_data_contract_required_for_every_component": True,
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
    causal_audit = release["evaluation"].get("causal_data_audit")
    if causal_audit:
        entries.append((str(causal_audit), "file"))
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
    entries.extend(
        (specification["path"], "file")
        for specification in release.get("reference_results", {}).values()
        if isinstance(specification, dict)
        and "path" in specification
        and "sha256" in specification
        and str(specification["path"]).startswith("artifacts/")
    )
    return entries


def _door_export_entries(release: dict[str, Any]) -> list[tuple[str, str]]:
    return door_icl_export_entries(release)


def _action_delay_export_entries(
    release: dict[str, Any],
) -> list[tuple[str, str]]:
    stages = release["training"].get("stages")
    if isinstance(stages, dict) and stages:
        training_roots = [
            (stage["artifact_tree"]["root"], "directory")
            for stage in stages.values()
        ]
    else:
        training_roots = [
            (release["training"]["artifact_tree"]["root"], "directory")
        ]
    evaluation_root = str(
        release["evaluation"]["artifact_tree"]["root"]
    ).rstrip("/")
    entries = training_roots + [
        (evaluation_root, "directory"),
        (release["evaluation"]["normalizer"], "file"),
    ]
    initialization = release["training"].get("initialization")
    if isinstance(initialization, dict):
        entries.extend(
            [
                (initialization["checkpoint"], "file"),
                (initialization["checkpoint_config"], "file"),
            ]
        )
    entries.extend(
        _external_artifact_files(
            {
                "evaluation_artifacts": release["evaluation"].get(
                    "artifacts", {}
                ),
                "reference_results": release.get("reference_results", {}),
            },
            bundled_roots=(
                *(str(path).rstrip("/") for path, _ in training_roots),
                evaluation_root,
            ),
        )
    )
    return entries


def _action_strength_export_entries(
    release: dict[str, Any],
) -> list[tuple[str, str]]:
    training_root = str(
        release["training"]["artifact_tree"]["root"]
    ).rstrip("/")
    evaluation_root = str(
        release["evaluation"]["artifact_tree"]["root"]
    ).rstrip("/")
    entries = [
        (training_root, "directory"),
        (evaluation_root, "directory"),
    ]
    bundled_roots = [training_root, evaluation_root]
    reference_method = release.get("reference_method", {})
    reference_tree = reference_method.get("artifact_tree", {})
    reference_root = reference_tree.get("root")
    if isinstance(reference_root, str):
        reference_root = reference_root.rstrip("/")
        entries.append((reference_root, "directory"))
        bundled_roots.append(reference_root)
    for section, key in (
        (release.get("training", {}), "contrast_scales"),
        (release.get("evaluation", {}), "planning_oracle"),
    ):
        specification = section.get(key)
        if not isinstance(specification, dict):
            continue
        path = specification.get("path")
        if isinstance(path, str) and not any(
            path == root or path.startswith(root + "/")
            for root in bundled_roots
        ):
            entries.append((path, "file"))
    for _root_path, artifacts in (
        (training_root, release["training"].get("artifacts", {})),
        (evaluation_root, release["evaluation"].get("artifacts", {})),
    ):
        entries.extend(
            (specification["path"], "file")
            for specification in artifacts.values()
            if isinstance(specification, dict)
            and str(specification.get("path", "")).startswith("artifacts/")
            and not any(
                str(specification["path"]) == bundled_root
                or str(specification["path"]).startswith(
                    bundled_root + "/"
                )
                for bundled_root in bundled_roots
            )
        )
    entries.extend(
        _external_artifact_files(
            release.get("reference_results", {}),
            bundled_roots=bundled_roots,
        )
    )
    return entries


def _external_artifact_files(
    value: Any,
    *,
    bundled_roots: Iterable[str] = (),
) -> list[tuple[str, str]]:
    """Collect hashed artifact files that sit outside bundled data trees.

    A component may organize receipts differently as its reference method
    evolves.  The Suite exporter therefore follows the stable public contract
    (an ``artifacts/...`` path plus its SHA-256) instead of depending on a
    method-specific diagnostic hierarchy.
    """

    normalized_roots = tuple(str(root).rstrip("/") for root in bundled_roots)
    entries: list[tuple[str, str]] = []

    def visit(item: Any) -> None:
        if isinstance(item, dict):
            path = item.get("path")
            digest = item.get("sha256")
            if (
                isinstance(path, str)
                and path.startswith("artifacts/")
                and isinstance(digest, str)
                and len(digest) == 64
                and not any(
                    path == root or path.startswith(root + "/")
                    for root in normalized_roots
                )
            ):
                entries.append((path, "file"))
            for child in item.values():
                visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)

    visit(value)
    return entries


def _contact_friction_export_entries(
    release: dict[str, Any],
) -> list[tuple[str, str]]:
    data_root = str(release["data"]["artifact_tree"]["root"]).rstrip("/")
    return [(data_root, "directory")] + _external_artifact_files(
        {
            "data_artifacts": release["data"].get("artifacts", {}),
            "reference_results": release.get("reference_results", {}),
        },
        bundled_roots=(data_root,),
    )


def _motion_damping_export_entries(
    release: dict[str, Any],
) -> list[tuple[str, str]]:
    data_root = str(release["data"]["artifact_tree"]["root"]).rstrip("/")
    return [(data_root, "directory")] + _external_artifact_files(
        {
            "data_artifacts": release["data"].get("artifacts", {}),
            "reference_results": release.get("reference_results", {}),
        },
        bundled_roots=(data_root,),
    )


def _robot_arm_mass_export_entries(
    release: dict[str, Any],
) -> list[tuple[str, str]]:
    data_root = str(release["data"]["artifact_tree"]["root"]).rstrip("/")
    entries = [(data_root, "directory")]
    entries.extend(
        (specification["path"], "file")
        for specification in release["data"].get("artifacts", {}).values()
        if isinstance(specification, dict)
        and "path" in specification
        and str(specification["path"]).startswith("artifacts/")
        and not str(specification["path"]).startswith(data_root + "/")
    )
    entries.extend(
        (specification["path"], "file")
        for specification in release.get("reference_results", {}).values()
        if isinstance(specification, dict)
        and "path" in specification
        and "sha256" in specification
        and str(specification["path"]).startswith("artifacts/")
    )
    return entries


def _portal_exit_export_entries(
    release: dict[str, Any],
) -> list[tuple[str, str]]:
    data_root = str(release["data"]["artifact_tree"]["root"]).rstrip("/")
    entries = [(data_root, "directory")]
    entries.extend(
        (specification["path"], "file")
        for specification in release["data"].get("artifacts", {}).values()
        if isinstance(specification, dict)
        and "path" in specification
        and str(specification["path"]).startswith("artifacts/")
        and not str(specification["path"]).startswith(data_root + "/")
    )
    entries.extend(
        (specification["path"], "file")
        for specification in release.get("reference_results", {}).values()
        if isinstance(specification, dict)
        and "path" in specification
        and "sha256" in specification
        and str(specification["path"]).startswith("artifacts/")
    )
    entries.extend(
        [
            (release["training"]["initialization"]["checkpoint"], "file"),
            (
                release["training"]["initialization"]["frozen_normalizer"],
                "file",
            ),
        ]
    )
    entries.extend(
        _external_artifact_files(
            release.get("scoring", {}).get(
                "original_task_retention", {}
            ),
            bundled_roots=(data_root,),
        )
    )
    return entries


def _artifact_target(benchmark_root: Path, logical_path: str) -> Path:
    relative = Path(logical_path)
    if not relative.parts or relative.parts[0] != "artifacts":
        raise ValueError(f"Expected artifacts/... path, got {logical_path}")
    return benchmark_root.joinpath(*relative.parts[1:])


def _deduplicate_export_entries(
    entries: Iterable[tuple[str, str]],
    *,
    repo_root: Path,
) -> list[tuple[str, str]]:
    """Collapse exact duplicates and files already covered by a data tree.

    A directory entry exports its complete subtree.  Listing one of its files
    again would either create the same target twice or, more seriously, hide
    a conflict between two different sources.  Descendants are skipped only
    when their resolved source is exactly the corresponding member of the
    exported directory; every other overlap fails closed.
    """

    unique: list[tuple[str, str]] = []
    kinds_by_path: dict[str, str] = {}
    for logical_path, kind in entries:
        if kind not in {"file", "directory"}:
            raise ValueError(f"Unsupported export entry kind: {kind!r}")
        path = Path(logical_path)
        if (
            path.is_absolute()
            or not path.parts
            or path.parts[0] != "artifacts"
            or ".." in path.parts
        ):
            raise ValueError(
                f"Export entry must be a normalized artifacts/... path: "
                f"{logical_path!r}"
            )
        normalized = path.as_posix()
        previous_kind = kinds_by_path.get(normalized)
        if previous_kind is not None:
            if previous_kind != kind:
                raise RuntimeError(
                    "Export target is registered as both a file and a "
                    f"directory: {normalized}"
                )
            continue
        kinds_by_path[normalized] = kind
        unique.append((normalized, kind))

    directory_paths = tuple(
        Path(logical_path)
        for logical_path, kind in unique
        if kind == "directory"
    )
    result: list[tuple[str, str]] = []
    for logical_path, kind in unique:
        path = Path(logical_path)
        ancestors = [
            directory
            for directory in directory_paths
            if directory != path and directory in path.parents
        ]
        if not ancestors:
            result.append((logical_path, kind))
            continue
        covering = min(ancestors, key=lambda value: len(value.parts))
        source = resolve_contextworld_path(
            logical_path,
            repo_root=repo_root,
        )
        covering_source = resolve_contextworld_path(
            covering.as_posix(),
            repo_root=repo_root,
        )
        expected_source = covering_source.joinpath(
            *path.relative_to(covering).parts
        )
        if source.resolve() != expected_source.resolve():
            raise RuntimeError(
                "Overlapping export entries resolve to different sources: "
                f"{covering.as_posix()} covers {logical_path}, but "
                f"{expected_source} != {source}"
            )
        if kind == "file" and not expected_source.is_file():
            raise RuntimeError(
                f"Covered export file is missing: {expected_source}"
            )
        if kind == "directory" and not expected_source.is_dir():
            raise RuntimeError(
                f"Covered export directory is missing: {expected_source}"
            )
    return result


def export_icl_suite_artifacts(
    destination: Path | str,
    *,
    release_config: Path | str = DEFAULT_SUITE_RELEASE_CONFIG,
    repo_root: Path | None = None,
    mode: str = "copy",
    include_upstream_original: bool = True,
) -> dict[str, Any]:
    """Export one README plus one integrated benchmark data directory."""

    if mode not in {"copy", "symlink"}:
        raise ValueError("Export mode must be 'copy' or 'symlink'")
    root = (repo_root or repository_root()).resolve()
    suite = load_icl_suite_release(release_config)
    _assert_frozen_export_inputs(suite, repo_root=root)
    destination = Path(destination).expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Export destination is not empty: {destination}")

    speed_config = resolve_contextworld_path(
        suite["components"]["speed"]["release_config"],
        repo_root=root,
    )
    door_config = resolve_contextworld_path(
        suite["components"]["door"]["release_config"],
        repo_root=root,
    )
    action_delay_config = resolve_contextworld_path(
        suite["components"]["action_delay"]["release_config"],
        repo_root=root,
    )
    action_strength_config = resolve_contextworld_path(
        suite["components"]["action_strength"]["release_config"],
        repo_root=root,
    )
    contact_friction_config = resolve_contextworld_path(
        suite["components"]["contact_friction"]["release_config"],
        repo_root=root,
    )
    motion_damping_config = resolve_contextworld_path(
        suite["components"]["motion_damping"]["release_config"],
        repo_root=root,
    )
    robot_arm_mass_config = resolve_contextworld_path(
        suite["components"]["robot_arm_mass"]["release_config"],
        repo_root=root,
    )
    portal_exit_config = resolve_contextworld_path(
        suite["components"]["portal_exit"]["release_config"],
        repo_root=root,
    )
    speed_release = load_speed_icl_release(speed_config)
    door_release = load_door_icl_release(door_config)
    action_delay_release = load_action_delay_icl_release(
        action_delay_config
    )
    action_strength_release = load_action_strength_icl_release(
        action_strength_config
    )
    contact_friction_release = load_contact_friction_icl_release(
        contact_friction_config
    )
    motion_damping_release = load_motion_damping_icl_release(
        motion_damping_config
    )
    robot_arm_mass_release = load_reacher_arm_mass_icl_release(
        robot_arm_mass_config
    )
    portal_exit_release = load_portal_exit_icl_release(portal_exit_config)

    reacher_checkpoint_configs = tuple(
        (
            f"robot_arm_mass.{family}_checkpoint_config",
            resolve_reacher_initial_checkpoint_config(
                robot_arm_mass_release,
                family,
                repo_root=root,
            ),
        )
        for family in ("lewm", "pldm")
    )
    _assert_portable_source_files(reacher_checkpoint_configs)

    entries = (
        _speed_export_entries(speed_release)
        + _door_export_entries(door_release)
        + _action_delay_export_entries(action_delay_release)
        + _action_strength_export_entries(action_strength_release)
        + _contact_friction_export_entries(contact_friction_release)
        + _motion_damping_export_entries(motion_damping_release)
        + _robot_arm_mass_export_entries(robot_arm_mass_release)
        + _portal_exit_export_entries(portal_exit_release)
        + [
            (suite["public_results"][key]["path"], "file")
            for key in ("specification", "scoreboard")
            if "public_results" in suite
        ]
    )
    entries = _deduplicate_export_entries(entries, repo_root=root)
    _assert_portable_export_entries(entries, repo_root=root)

    destination.mkdir(parents=True, exist_ok=True)
    benchmark_root = destination / "benchmark"
    benchmark_root.mkdir()
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
    tworoom_lance_entry: dict[str, Any] | None = None
    pusht_upstream_entries: list[dict[str, Any]] = []
    reacher_upstream_entries: list[dict[str, Any]] = []
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

        portal_upstream = portal_exit_release.get("training", {}).get(
            "upstream", {}
        )
        portal_lance_specification = portal_upstream.get("original_lance")
        if portal_lance_specification is not None:
            portal_lance_source = resolve_portal_original_lance(
                portal_exit_release,
                repo_root=root,
            )
            if not portal_lance_source.is_dir():
                raise FileNotFoundError(portal_lance_source)
            portal_lance_target = (
                benchmark_root
                / "upstream/stable-worldmodel/lewm_tworoom.lance"
            )
            portal_lance_target.parent.mkdir(parents=True, exist_ok=True)
            if mode == "symlink":
                portal_lance_target.symlink_to(
                    portal_lance_source,
                    target_is_directory=True,
                )
            else:
                shutil.copytree(portal_lance_source, portal_lance_target)
            tworoom_lance_entry = {
                "logical_path": (
                    "upstream/stable-worldmodel/lewm_tworoom.lance"
                ),
                "kind": "directory",
                "bytes": int(portal_lance_specification["bytes"]),
                "source_role": portal_lance_specification["role"],
            }
            inventory_entries.append(tworoom_lance_entry)

        pusht_sources = (
            (
                "original_h5",
                resolve_action_strength_original_h5(
                    action_strength_release,
                    repo_root=root,
                ),
                benchmark_root
                / "upstream/stable-worldmodel/pusht_expert_train.h5",
                "file",
            ),
            (
                "original_lance",
                resolve_action_strength_original_lance(
                    action_strength_release,
                    repo_root=root,
                ),
                benchmark_root
                / "upstream/stable-worldmodel/lewm_pusht.lance",
                "directory",
            ),
            (
                "initial_checkpoint",
                resolve_action_strength_initial_checkpoint(
                    action_strength_release,
                    repo_root=root,
                ),
                benchmark_root
                / "upstream/stable-worldmodel/"
                "pusht_lewm_baseline_seed3073_weights.ckpt",
                "file",
            ),
        )
        for name, source, target, kind in pusht_sources:
            if not source.exists():
                raise FileNotFoundError(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            if mode == "symlink":
                target.symlink_to(
                    source,
                    target_is_directory=(kind == "directory"),
                )
            elif kind == "directory":
                shutil.copytree(source, target)
            else:
                shutil.copy2(source, target)
            specification = (
                action_strength_release["training"]["initialization"]
                if name == "initial_checkpoint"
                else action_strength_release["training"]["upstream"][name]
            )
            entry = {
                "logical_path": (
                    "upstream/stable-worldmodel/" + target.name
                ),
                "kind": kind,
                "bytes": int(specification["bytes"]),
                "source_role": specification["role"],
            }
            if specification.get("sha256"):
                entry["sha256"] = specification["sha256"]
            pusht_upstream_entries.append(entry)
            inventory_entries.append(entry)

        reacher_sources = (
            (
                "original_h5",
                resolve_reacher_original_h5(
                    robot_arm_mass_release,
                    repo_root=root,
                ),
                benchmark_root / "upstream/stable-worldmodel/reacher.h5",
                "file",
                robot_arm_mass_release["training"]["upstream"][
                    "original_h5"
                ],
            ),
            (
                "original_lance",
                resolve_reacher_original_lance(
                    robot_arm_mass_release,
                    repo_root=root,
                ),
                benchmark_root
                / "upstream/stable-worldmodel/lewm_reacher.lance",
                "directory",
                robot_arm_mass_release["training"]["upstream"][
                    "original_lance"
                ],
            ),
            (
                "lewm_checkpoint",
                resolve_reacher_initial_checkpoint(
                    robot_arm_mass_release,
                    "lewm",
                    repo_root=root,
                ),
                benchmark_root
                / "upstream/stable-worldmodel/reacher_lewm/"
                "reacher_lewm_weights.ckpt",
                "file",
                robot_arm_mass_release["training"]["reference_matrix"][
                    "initial_checkpoints"
                ]["lewm"],
            ),
            (
                "pldm_checkpoint",
                resolve_reacher_initial_checkpoint(
                    robot_arm_mass_release,
                    "pldm",
                    repo_root=root,
                ),
                benchmark_root
                / "upstream/stable-worldmodel/reacher_pldm_baseline/"
                "reacher_pldm_baseline_weights.ckpt",
                "file",
                robot_arm_mass_release["training"]["reference_matrix"][
                    "initial_checkpoints"
                ]["pldm"],
            ),
        )
        for name, source, target, kind, specification in reacher_sources:
            if not source.exists():
                raise FileNotFoundError(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            if mode == "symlink":
                target.symlink_to(
                    source,
                    target_is_directory=(kind == "directory"),
                )
            elif kind == "directory":
                shutil.copytree(source, target)
            else:
                shutil.copy2(source, target)
            entry = {
                "logical_path": target.relative_to(benchmark_root).as_posix(),
                "kind": kind,
                "bytes": int(specification["bytes"]),
                "source_role": specification["role"],
            }
            if specification.get("sha256"):
                entry["sha256"] = specification["sha256"]
            reacher_upstream_entries.append(entry)
            inventory_entries.append(entry)

        for family in ("lewm", "pldm"):
            specification = robot_arm_mass_release["training"][
                "reference_matrix"
            ]["initial_checkpoints"][family]
            source = resolve_reacher_initial_checkpoint_config(
                robot_arm_mass_release,
                family,
                repo_root=root,
            )
            target = benchmark_root / specification[
                "config_bundled_artifact_path"
            ]
            target.parent.mkdir(parents=True, exist_ok=True)
            if mode == "symlink":
                target.symlink_to(source)
            else:
                shutil.copy2(source, target)
            entry = {
                "logical_path": target.relative_to(benchmark_root).as_posix(),
                "kind": "file",
                "bytes": int(specification["config_bytes"]),
                "sha256": specification["config_sha256"],
                "source_role": f"{family}_checkpoint_configuration",
            }
            reacher_upstream_entries.append(entry)
            inventory_entries.append(entry)

    releases_dir = benchmark_root / "releases"
    releases_dir.mkdir()
    shutil.copy2(speed_config, releases_dir / "speed.yaml")
    shutil.copy2(door_config, releases_dir / "door.yaml")
    shutil.copy2(
        action_delay_config,
        releases_dir / "action_delay.yaml",
    )
    shutil.copy2(
        action_strength_config,
        releases_dir / "action_strength.yaml",
    )
    shutil.copy2(
        contact_friction_config,
        releases_dir / "contact_friction.yaml",
    )
    shutil.copy2(
        motion_damping_config,
        releases_dir / "motion_damping.yaml",
    )
    shutil.copy2(
        robot_arm_mass_config,
        releases_dir / "robot_arm_mass.yaml",
    )
    shutil.copy2(
        portal_exit_config,
        releases_dir / "portal_exit.yaml",
    )
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
            "action_delay": "benchmark/releases/action_delay.yaml",
            "action_strength": (
                "benchmark/releases/action_strength.yaml"
            ),
            "contact_friction": (
                "benchmark/releases/contact_friction.yaml"
            ),
            "motion_damping": (
                "benchmark/releases/motion_damping.yaml"
            ),
            "robot_arm_mass": (
                "benchmark/releases/robot_arm_mass.yaml"
            ),
            "portal_exit": "benchmark/releases/portal_exit.yaml",
        },
        "public_results": (
            {
                key: str(
                    Path("benchmark").joinpath(
                        *Path(
                            suite["public_results"][key]["path"]
                        ).parts[1:]
                    )
                )
                for key in ("specification", "scoreboard")
            }
            if "public_results" in suite
            else {}
        ),
        "components": list(COMPONENT_IDS),
        "includes_upstream_original_h5": include_upstream_original,
        "public_test_included": True,
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
        "tworoom_upstream": (
            {
                "original_h5": str(
                    benchmark_root / "upstream/lewm-tworooms/tworoom.h5"
                ),
                "original_lance": str(
                    benchmark_root
                    / "upstream/stable-worldmodel/lewm_tworoom.lance"
                ),
            }
            if original_entry is not None and tworoom_lance_entry is not None
            else None
        ),
        "pusht_upstream": (
            {
                "original_h5": str(
                    benchmark_root
                    / "upstream/stable-worldmodel/pusht_expert_train.h5"
                ),
                "original_lance": str(
                    benchmark_root
                    / "upstream/stable-worldmodel/lewm_pusht.lance"
                ),
                "initial_checkpoint": str(
                    benchmark_root
                    / "upstream/stable-worldmodel/"
                    "pusht_lewm_baseline_seed3073_weights.ckpt"
                ),
            }
            if pusht_upstream_entries
            else None
        ),
        "reacher_upstream": (
            {
                "original_h5": str(
                    benchmark_root
                    / "upstream/stable-worldmodel/reacher.h5"
                ),
                "original_lance": str(
                    benchmark_root
                    / "upstream/stable-worldmodel/lewm_reacher.lance"
                ),
                "lewm_checkpoint": str(
                    benchmark_root
                    / "upstream/stable-worldmodel/reacher_lewm/"
                    "reacher_lewm_weights.ckpt"
                ),
                "pldm_checkpoint": str(
                    benchmark_root
                    / "upstream/stable-worldmodel/reacher_pldm_baseline/"
                    "reacher_pldm_baseline_weights.ckpt"
                ),
            }
            if reacher_upstream_entries
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
