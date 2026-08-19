"""Additive integrity-reseal support for a changed Suite v2 working tree.

Recovery-v2 and the documentation amendment are historical commit markers.  A
later edit to a component release YAML must not make either marker look as if
it signed the new bytes.  This module therefore records the historical chain
separately and builds a new, exact manifest for the current Suite material.
The public Suite activation is wired fail-closed to this layer: activation
remains unavailable until the current baseline-CEM freeze and public document
are final and the one-use reseal decision has been written.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from contextworld.benchmarks.suite_data import (
    SUITE_V2_COMPONENT_IDS,
    load_icl_suite_release,
)
from contextworld.paths import repository_root, resolve_contextworld_path


RESEAL_ID = "contextworld_icl_suite_v2_integrity_reseal_v1"
RESEAL_CONFIG = (
    repository_root()
    / "configs/benchmark/contextworld_icl_suite_v2_integrity_reseal_v1.yaml"
)
DESCRIPTIVE_RESULT_FREEZE_SPECS = {
    "original_baseline_icl": {
        "path": (
            "configs/benchmark/"
            "contextworld_original_baseline_matrix_results_freeze_v1.json"
        ),
        "freeze_id": "contextworld_original_baseline_matrix_results_freeze_v1",
    },
    "original_baseline_cem": {
        "path": (
            "configs/benchmark/"
            "contextworld_original_baseline_cem_results_freeze_v1.json"
        ),
        "freeze_id": "contextworld_original_baseline_cem_results_freeze_v1",
    },
}
RESEAL_AUTHORITY_IMPLEMENTATION_PATHS = (
    "contextworld/benchmarks/suite_v2_integrity_reseal.py",
    "scripts/freeze_contextworld_icl_suite_v2_integrity_reseal_v1.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(logical_path: str, *, repo_root: Path) -> dict[str, Any]:
    path = resolve_contextworld_path(logical_path, repo_root=repo_root)
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": logical_path,
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _config_identity(path: Path, *, repo_root: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    try:
        logical = str(resolved.relative_to(repo_root))
    except ValueError as error:
        raise ValueError("reseal config must live under the repository root") from error
    return _identity(logical, repo_root=repo_root)


def _require_identity(
    observed: Mapping[str, Any],
    expected: Any,
    *,
    label: str,
) -> None:
    if not isinstance(expected, Mapping):
        raise ValueError(f"{label} identity is invalid")
    if dict(observed) != dict(expected):
        raise ValueError(f"{label} identity drifted")


def load_integrity_reseal_preregistration(
    path: Path | str = RESEAL_CONFIG,
    *,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Load the non-active reseal plan and verify its historical anchors."""

    root = (repo_root or repository_root()).resolve()
    config_path = Path(path).expanduser().resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    expected_keys = {
        "schema_version",
        "release_id",
        "release_status",
        "candidate_date",
        "integrity_reseal",
        "membership_authority",
    }
    expected_reseal_keys = {
        "reseal_id",
        "status",
        "scope",
        "historical_chain",
        "allowed_changes",
        "prohibited_changes",
        "descriptive_result_freezes",
        "decision_contract",
    }
    reseal = payload.get("integrity_reseal") if isinstance(payload, dict) else None
    if (
        not isinstance(payload, dict)
        or set(payload) != expected_keys
        or payload.get("schema_version") != 1
        or payload.get("release_id") != "contextworld_icl_benchmark_suite_v2"
        or not isinstance(reseal, dict)
        or set(reseal) != expected_reseal_keys
        or reseal.get("reseal_id") != RESEAL_ID
        or reseal.get("status")
        != "preregistered_pending_final_identity_freeze"
        or reseal.get("scope") != "current_nine_component_identity_reseal"
    ):
        raise ValueError("Suite v2 integrity-reseal preregistration is invalid")

    chain = reseal.get("historical_chain")
    expected_chain_paths = {
        "documentation_amendment_config": (
            "configs/benchmark/"
            "contextworld_icl_suite_v2_public_document_amendment_v1.yaml"
        ),
        "documentation_amendment_decision": (
            "configs/benchmark/"
            "contextworld_icl_suite_v2_public_document_amendment_decision_v1.json"
        ),
        "recovery_v2_config": (
            "configs/benchmark/contextworld_icl_suite_v2_recovery_v2.yaml"
        ),
        "recovery_v2_decision": (
            "artifacts/evaluation/history3/"
            "cube_gripper_carry_h3_v4r1_suite_registration_recovery_v2/"
            "registration_decision_v2.json"
        ),
    }
    if not isinstance(chain, dict) or set(chain) != set(expected_chain_paths):
        raise ValueError("Suite v2 integrity-reseal historical chain is invalid")
    for name, logical_path in expected_chain_paths.items():
        _require_identity(
            _identity(logical_path, repo_root=root),
            chain[name],
            label=f"historical chain {name}",
        )

    contract = reseal.get("decision_contract")
    if (
        not isinstance(contract, dict)
        or contract.get("decision_path")
        != "configs/benchmark/"
        "contextworld_icl_suite_v2_integrity_reseal_decision_v1.json"
        or contract.get("exclusive_creation_required") is not True
        or contract.get("historical_chain_must_remain_byte_identical")
        is not True
        or contract.get("old_membership_may_not_be_silently_reactivated")
        is not True
        or contract.get("activation_requires_new_decision") is not True
    ):
        raise ValueError("Suite v2 integrity-reseal decision contract is invalid")
    result_freezes = reseal.get("descriptive_result_freezes")
    if result_freezes != DESCRIPTIVE_RESULT_FREEZE_SPECS:
        raise ValueError(
            "Suite v2 integrity-reseal descriptive-result freeze contract is invalid"
        )
    return {**reseal, "_config_path": str(config_path)}


def _descriptive_result_freeze_identities(
    preregistration: Mapping[str, Any], *, repo_root: Path
) -> dict[str, Any]:
    """Require both descriptive-result freezes and bind their exact bytes."""

    specifications = preregistration["descriptive_result_freezes"]
    identities: dict[str, Any] = {}
    for name, specification in specifications.items():
        logical_path = specification["path"]
        try:
            identity = _identity(logical_path, repo_root=repo_root)
        except FileNotFoundError as error:
            raise ValueError(
                "required descriptive result freeze is missing: "
                f"{logical_path}"
            ) from error
        try:
            payload = json.loads(
                resolve_contextworld_path(logical_path, repo_root=repo_root).read_text(
                    encoding="utf-8"
                )
            )
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(
                "required descriptive result freeze is unreadable: "
                f"{logical_path}"
            ) from error
        if (
            not isinstance(payload, dict)
            or payload.get("freeze_id") != specification["freeze_id"]
        ):
            raise ValueError(
                "required descriptive result freeze has the wrong freeze_id: "
                f"{logical_path}"
            )
        identities[name] = identity
    return identities


def _audit_descriptive_result_freezes(
    preregistration: Mapping[str, Any], *, repo_root: Path
) -> tuple[dict[str, Any], list[str]]:
    """Report each freeze independently while keeping formal creation strict."""

    identities: dict[str, Any] = {}
    missing: list[str] = []
    for name, specification in preregistration[
        "descriptive_result_freezes"
    ].items():
        try:
            identities.update(
                _descriptive_result_freeze_identities(
                    {"descriptive_result_freezes": {name: specification}},
                    repo_root=repo_root,
                )
            )
        except ValueError as error:
            missing.append(str(error))
    return identities, missing


def _current_material_identities_without_result_freezes(
    *, repo_root: Path | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Collect live Suite bytes without relaxing the formal freeze gate."""

    root = (repo_root or repository_root()).resolve()
    prereg = load_integrity_reseal_preregistration(repo_root=root)
    chain = prereg["historical_chain"]
    amendment_path = resolve_contextworld_path(
        chain["documentation_amendment_config"]["path"], repo_root=root
    )
    # Loading does structural validation only; it deliberately does not claim
    # the old amendment remains active after its recorded inputs drifted.
    suite = load_icl_suite_release(amendment_path)
    sources = {
        logical_path: _identity(logical_path, repo_root=root)
        for logical_path in sorted(suite["repository"]["source_sha256"])
    }
    components = {
        component_id: _identity(
            suite["components"][component_id]["release_config"],
            repo_root=root,
        )
        for component_id in SUITE_V2_COMPONENT_IDS
    }
    public_results = {
        key: _identity(
            suite["public_results"][key]["path"], repo_root=root
        )
        for key in ("specification", "scoreboard")
    }
    return {
        "sources": sources,
        "authority_implementation": {
            logical_path: _identity(logical_path, repo_root=root)
            for logical_path in RESEAL_AUTHORITY_IMPLEMENTATION_PATHS
        },
        "public_document": _identity(
            suite["repository"]["public_document"]["path"], repo_root=root
        ),
        "components": components,
        "public_results": public_results,
    }, prereg


def current_material_identities(*, repo_root: Path | None = None) -> dict[str, Any]:
    """Return exact material that a new decision may bind, never a partial set."""

    root = (repo_root or repository_root()).resolve()
    materials, prereg = _current_material_identities_without_result_freezes(
        repo_root=root
    )
    materials["descriptive_result_freezes"] = (
        _descriptive_result_freeze_identities(prereg, repo_root=root)
    )
    return materials


def audit_current_identity_drift(
    *, repo_root: Path | None = None
) -> dict[str, Any]:
    """Compare live bytes with the last documentation-amendment expectations."""

    root = (repo_root or repository_root()).resolve()
    prereg = load_integrity_reseal_preregistration(repo_root=root)
    amendment_path = resolve_contextworld_path(
        prereg["historical_chain"]["documentation_amendment_config"]["path"],
        repo_root=root,
    )
    suite = load_icl_suite_release(amendment_path)
    current, _ = _current_material_identities_without_result_freezes(
        repo_root=root
    )
    result_freezes, missing_result_freezes = _audit_descriptive_result_freezes(
        prereg, repo_root=root
    )
    source_drift = {
        logical_path: {
            "expected_sha256": expected,
            "observed_sha256": current["sources"][logical_path]["sha256"],
        }
        for logical_path, expected in suite["repository"]["source_sha256"].items()
        if current["sources"][logical_path]["sha256"] != expected
    }
    component_drift = {
        component_id: {
            "expected_sha256": suite["components"][component_id][
                "release_config_sha256"
            ],
            "observed_sha256": current["components"][component_id]["sha256"],
        }
        for component_id in SUITE_V2_COMPONENT_IDS
        if current["components"][component_id]["sha256"]
        != suite["components"][component_id]["release_config_sha256"]
    }
    expected_document = suite["repository"]["public_document"]["sha256"]
    observed_document = current["public_document"]["sha256"]
    return {
        "schema_version": 1,
        "reseal_id": RESEAL_ID,
        "source_drift": source_drift,
        "component_release_drift": component_drift,
        "public_document_drift": {
            "expected_sha256": expected_document,
            "observed_sha256": observed_document,
            "drifted": observed_document != expected_document,
        },
        "descriptive_result_freezes": result_freezes,
        "missing_required_descriptive_result_freezes": missing_result_freezes,
        "requires_additive_reseal": bool(
            source_drift
            or component_drift
            or observed_document != expected_document
            or missing_result_freezes
        ),
    }


def build_integrity_reseal_decision(
    *,
    reseal_config: Path | str = RESEAL_CONFIG,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Build, but do not write, the exact decision for stable current bytes."""

    root = (repo_root or repository_root()).resolve()
    prereg = load_integrity_reseal_preregistration(
        reseal_config, repo_root=root
    )
    return {
        "schema_version": 1,
        "reseal_id": RESEAL_ID,
        "suite_release_id": "contextworld_icl_benchmark_suite_v2",
        "status": "integrity_reseal_passed",
        "passed": True,
        "reseal_config": _config_identity(
            Path(prereg["_config_path"]), repo_root=root
        ),
        "historical_chain": prereg["historical_chain"],
        "release_materials": current_material_identities(repo_root=root),
        "claims": {
            "historical_chain_preserved": True,
            "old_membership_silently_reactivated": False,
            "formal_scoreboard_mutated": False,
            "public_test_rerun": False,
            "training_or_checkpoint_selection": False,
            "raw_dataset_or_checkpoint_mutation": False,
            "current_nine_component_identities_resealed": True,
        },
    }


def validate_integrity_reseal_decision(
    decision: Mapping[str, Any],
    *,
    reseal_config: Path | str = RESEAL_CONFIG,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Fail closed unless a decision binds the current bytes exactly."""

    expected = build_integrity_reseal_decision(
        reseal_config=reseal_config, repo_root=repo_root
    )
    if not isinstance(decision, Mapping) or dict(decision) != expected:
        raise ValueError("Suite v2 integrity-reseal decision does not match current bytes")
    return {"passed": True, "reseal_id": RESEAL_ID}


def write_integrity_reseal_decision(
    output: Path | str,
    *,
    reseal_config: Path | str = RESEAL_CONFIG,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Create a one-use decision file; never overwrite an existing decision."""

    decision = build_integrity_reseal_decision(
        reseal_config=reseal_config, repo_root=repo_root
    )
    target = Path(output).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as stream:
        json.dump(decision, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    return decision
