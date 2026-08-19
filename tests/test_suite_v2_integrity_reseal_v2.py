from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

import contextworld.benchmarks.suite_v2_integrity_reseal_v2 as reseal_v2
from contextworld.benchmarks.public_score import make_public_scoreboard_from_spec
from contextworld.paths import resolve_contextworld_path


ROOT = Path(__file__).resolve().parents[1]


def _identity(path: Path, logical_path: str) -> dict[str, Any]:
    return {
        "path": logical_path,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size_bytes": path.stat().st_size,
    }


def _copy(root: Path, logical_path: str) -> Path:
    source = resolve_contextworld_path(logical_path, repo_root=ROOT)
    target = root / logical_path
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return target


def _write_json(root: Path, logical_path: str, payload: dict[str, Any]) -> Path:
    target = root / logical_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return target


def _completion_ids(config: dict[str, Any]) -> dict[str, str]:
    specifications = config["integrity_reseal"]["required_evidence"][
        "pldm_completion_preregistrations"
    ]
    return {
        component_id: specification["completion_id"]
        for component_id, specification in specifications.items()
    }


def _valid_aggregate_validation() -> dict[str, Any]:
    """Minimal stand-in for the production aggregate validator in fixtures.

    Fixture trees intentionally do not copy the large frozen training and
    Public-Test receipts.  Production code calls the real validator; these
    fields model the exact semantic summary it must return.
    """

    return {
        "passed": True,
        "aggregate_id": reseal_v2.FINAL_PLDMS_AGGREGATE_FREEZE_ID,
        "formal_reference_rows": 13,
        "formal_reference_rows_added": 2,
        "components_added": ["speed", "action_strength"],
        "development_only_components_not_added": [
            "contact_friction",
            "motion_damping",
        ],
        "speed_evidence_scope": "behavioral",
        "speed_training_attribution_claim": False,
        "action_strength_formal_row_included": True,
        "action_strength_ability_passed": False,
        "local_outputs": {
            "aggregate_freeze": {
                "path": reseal_v2.FINAL_PLDMS_AGGREGATE_FREEZE_PATH,
                "sha256": "a" * 64,
                "size_bytes": 1,
            },
            "addendum_specification": {
                "path": (
                    "artifacts/evaluation/"
                    "contextworld_icl_suite_v2_release_addendum_v1/"
                    "public_scoreboard_spec.json"
                ),
                "sha256": "b" * 64,
                "size_bytes": 1,
            },
            "addendum_scoreboard": {
                "path": (
                    "artifacts/evaluation/"
                    "contextworld_icl_suite_v2_release_addendum_v1/"
                    "public_scoreboard.json"
                ),
                "sha256": "c" * 64,
                "size_bytes": 1,
            },
            "scoreboard_resolution_decision": {
                "path": (
                    "configs/benchmark/"
                    "contextworld_icl_suite_v2_scoreboard_addendum_decision_v1.json"
                ),
                "sha256": "d" * 64,
                "size_bytes": 1,
            },
        },
    }


@pytest.fixture(autouse=True)
def _stub_action_strength_release_audit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fixture tree contains identities, not the large Lance release."""

    monkeypatch.setattr(
        reseal_v2,
        "audit_action_strength_icl_release",
        lambda **_kwargs: {
            "release_id": "contextworld_pusht_action_strength_icl_history3_v1",
            "status": "passed",
            "passed": True,
            "full_content_hash_audit": False,
        },
    )
    monkeypatch.setattr(
        reseal_v2,
        "audit_archived_original_baseline_matrix",
        lambda **_kwargs: {
            "audit_id": "contextworld_original_baseline_archive_audit_v1",
            "status": "passed",
            "archive_scope": "immutable_frozen_results_only",
            "live_release_rederivation_performed": False,
        },
    )


@pytest.fixture
def finalized_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path]:
    """Install exact v1 ancestors plus synthetic, final v2-only evidence."""

    monkeypatch.setattr(
        reseal_v2,
        "validate_written_completion_aggregate_and_scoreboard",
        lambda **_kwargs: _valid_aggregate_validation(),
    )

    root = tmp_path / "ContextWorld"
    config_path = _copy(
        root,
        "configs/benchmark/contextworld_icl_suite_v2_integrity_reseal_v2.yaml",
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    evidence = config["integrity_reseal"]["required_evidence"]

    for specification in config["integrity_reseal"][
        "historical_predecessor"
    ].values():
        _copy(root, specification["path"])
    v1_prereg = yaml.safe_load(
        (
            root
            / "configs/benchmark/contextworld_icl_suite_v2_integrity_reseal_v1.yaml"
        ).read_text(encoding="utf-8")
    )
    for specification in v1_prereg["integrity_reseal"]["historical_chain"].values():
        _copy(root, specification["path"])
    _copy(root, evidence["engineering_identity_amendment"]["path"])
    _copy(root, evidence["capability_taxonomy_documentation_amendment"]["path"])
    _copy(root, evidence["current_results_overlay"]["path"])
    _copy(root, evidence["action_strength_float32_consistency_amendment"]["path"])
    _copy(root, evidence["action_strength_float32_consistency_amendment"]["scorer_path"])
    _copy(
        root,
        evidence["action_strength_float32_consistency_amendment"][
            "release_auditor_path"
        ],
    )
    _copy(root, evidence["original_baseline_archive_auditor"]["module"]["path"])
    _copy(root, evidence["original_baseline_archive_auditor"]["cli"]["path"])
    _copy(
        root,
        evidence["public_scoreboard"]["additive_resolution_preregistration"][
            "path"
        ],
    )
    _copy(root, evidence["registered_suite_sources"]["source_manifest"])
    _copy(root, "contextworld/benchmarks/suite_v2_integrity_reseal_v2.py")
    _copy(root, "scripts/freeze_contextworld_icl_suite_v2_integrity_reseal_v2.py")

    for logical_path in evidence["registered_suite_sources"]["paths"]:
        _copy(root, logical_path)
    for logical_path in evidence["final_public_documents"].values():
        _copy(root, logical_path)
    for logical_path in evidence["current_component_release_configs"].values():
        _copy(root, logical_path)
    for specification in evidence["original_baseline_result_freezes"].values():
        _copy(root, specification["path"])
    for specification in evidence["pldm_completion_preregistrations"].values():
        _copy(root, specification["path"])

    amendment = yaml.safe_load(
        (root / evidence["engineering_identity_amendment"]["path"]).read_text(
            encoding="utf-8"
        )
    )["engineering_identity_amendment"]
    for update in amendment["approved_component_identity_updates"].values():
        for logical_path in update["source_paths"]:
            _copy(root, logical_path)

    aggregate_spec = evidence["final_pldm_completion_aggregate_results_freeze"]
    aggregate_path = _write_json(
        root,
        aggregate_spec["path"],
        {
            "schema_version": 1,
            "freeze_id": aggregate_spec["freeze_id"],
            "status": aggregate_spec["required_status"],
            "all_four_completion_outcomes_finalized": True,
            "completion_results": {
                component_id: {
                    "completion_id": completion_id,
                    "finalized": True,
                    "outcome": "synthetic_final_fixture",
                }
                for component_id, completion_id in _completion_ids(config).items()
            },
        },
    )

    scoreboard_spec = evidence["public_scoreboard"]
    base_specification = scoreboard_spec["historical_base_specification"]
    base_scoreboard = scoreboard_spec["historical_base_scoreboard"]
    base_specification_path = _copy(root, base_specification["path"])
    base_scoreboard_path = _copy(root, base_scoreboard["path"])
    specification_path = root / scoreboard_spec["addendum_specification"]
    scoreboard_path = root / scoreboard_spec["addendum_scoreboard"]
    specification_path.parent.mkdir(parents=True, exist_ok=True)
    scoreboard_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(base_specification_path, specification_path)
    shutil.copy2(base_scoreboard_path, scoreboard_path)
    resolution_spec = scoreboard_spec["additive_resolution_decision"]
    _write_json(
        root,
        resolution_spec["path"],
        {
            "schema_version": 1,
            "decision_id": resolution_spec["decision_id"],
            "status": "no_additive_scoreboard_extension",
            "passed": True,
            "scoreboard_extension_authorized": False,
            "formal_reference_rows": 11,
            "formal_reference_rows_added": 0,
            "historical_base_specification": _identity(
                base_specification_path, base_specification["path"]
            ),
            "historical_base_scoreboard": _identity(
                base_scoreboard_path, base_scoreboard["path"]
            ),
            "addendum_specification": _identity(
                specification_path, scoreboard_spec["addendum_specification"]
            ),
            "addendum_scoreboard": _identity(
                scoreboard_path, scoreboard_spec["addendum_scoreboard"]
            ),
            "final_pldm_completion_aggregate_results_freeze": _identity(
                aggregate_path, aggregate_spec["path"]
            ),
        },
    )
    return root, config_path


def _build(root: Path, config_path: Path) -> dict[str, Any]:
    return reseal_v2.build_integrity_reseal_v2_decision(
        reseal_config=config_path,
        repo_root=root,
    )


def test_v2_preregistration_leaves_dynamic_public_documents_unfrozen() -> None:
    payload = yaml.safe_load(reseal_v2.RESEAL_CONFIG.read_text(encoding="utf-8"))
    documents = payload["integrity_reseal"]["required_evidence"][
        "final_public_documents"
    ]

    assert set(documents) == {
        "public_benchmark",
        "repository_readme",
        "docs_navigation",
        "public_release_readiness",
    }
    assert all(isinstance(path, str) for path in documents.values())
    assert all("sha256" not in str(value) for value in documents.values())
    decision = json.loads(
        (ROOT / reseal_v2.RESEAL_DECISION_PATH).read_text(encoding="utf-8")
    )
    assert reseal_v2.validate_integrity_reseal_v2_decision(
        decision, repo_root=ROOT
    ) == {"passed": True, "reseal_id": reseal_v2.RESEAL_ID}


def test_check_only_reports_the_final_repository_ready_without_rewriting_decision() -> None:
    audit = reseal_v2.audit_integrity_reseal_v2_readiness(repo_root=ROOT)

    assert audit["ready"] is True
    assert audit["decision_created"] is False
    assert audit["blockers"] == []
    assert audit["status"] == "ready_for_explicit_one_use_decision_creation"


def test_check_only_reports_multiple_missing_evidence_paths(
    finalized_fixture: tuple[Path, Path],
) -> None:
    root, config_path = finalized_fixture
    aggregate = root / reseal_v2.FINAL_PLDMS_AGGREGATE_FREEZE_PATH
    resolution = root / (
        "configs/benchmark/contextworld_icl_suite_v2_scoreboard_addendum_decision_v1.json"
    )
    aggregate.unlink()
    resolution.unlink()

    audit = reseal_v2.audit_integrity_reseal_v2_readiness(
        reseal_config=config_path, repo_root=root
    )

    assert audit["ready"] is False
    assert any(str(aggregate.relative_to(root)) in blocker for blocker in audit["blockers"])
    assert any(str(resolution.relative_to(root)) in blocker for blocker in audit["blockers"])


def test_final_fixture_binds_all_evidence_and_rejects_current_material_drift(
    finalized_fixture: tuple[Path, Path],
) -> None:
    root, config_path = finalized_fixture
    decision = _build(root, config_path)

    assert reseal_v2.validate_integrity_reseal_v2_decision(
        decision, reseal_config=config_path, repo_root=root
    ) == {"passed": True, "reseal_id": reseal_v2.RESEAL_ID}
    materials = decision["release_materials"]
    assert set(materials["current_component_release_configs"]) == set(
        reseal_v2.COMPONENT_IDS
    )
    assert set(materials["pldm_completion_preregistrations"]) == set(
        reseal_v2.PLDM_COMPLETION_COMPONENT_IDS
    )
    assert set(materials["capability_taxonomy_documentation_amendment"][
        "final_public_documents"
    ]) == {
        "public_benchmark",
        "repository_readme",
        "docs_navigation",
        "public_release_readiness",
    }
    action_strength = materials["action_strength_float32_consistency_amendment"]
    assert action_strength["amendment"] == {
        "path": reseal_v2.ACTION_STRENGTH_FLOAT32_AMENDMENT_PATH,
        "sha256": "07ea18d4ebf9df5f798e5b3d0761b19145c5caab162ef513ffdb4150bc368d0a",
        "size_bytes": 3676,
    }
    assert action_strength["current_release_audit"]["passed"] is True
    assert action_strength["current_release_audit"]["status"] == "passed"
    assert materials["original_baseline_archive_auditor"]["audit"] == {
        "audit_id": "contextworld_original_baseline_archive_audit_v1",
        "status": "passed",
        "archive_scope": "immutable_frozen_results_only",
        "live_release_rederivation_performed": False,
    }

    readme = root / "README.md"
    readme.write_text(readme.read_text(encoding="utf-8") + "\ncurrent drift\n", encoding="utf-8")
    with pytest.raises(ValueError, match="does not match current bytes"):
        reseal_v2.validate_integrity_reseal_v2_decision(
            decision, reseal_config=config_path, repo_root=root
        )


@pytest.mark.parametrize(
    "logical_path",
    [
        "configs/benchmark/contextworld_icl_suite_v2_engineering_identity_amendment_prereg_v1.yaml",
        "configs/benchmark/contextworld_icl_suite_v2_capability_taxonomy_documentation_amendment_v1.yaml",
        "configs/benchmark/pusht_action_strength_score_float32_consistency_amendment_v1.yaml",
        "contextworld/benchmarks/original_baseline_archive.py",
        "scripts/audit_contextworld_original_baseline_matrix_freeze_v1.py",
        "configs/benchmark/contextworld_icl_suite_v2_current_results_overlay_v1.yaml",
        "docs/README.md",
        "configs/benchmark/tworoom_speed_icl_release_v1.yaml",
        "contextworld/benchmarks/suite_data.py",
        "configs/benchmark/contextworld_original_baseline_matrix_results_freeze_v1.json",
        "configs/benchmark/tworoom_speed_pldm_reference_completion_v1.yaml",
        reseal_v2.FINAL_PLDMS_AGGREGATE_FREEZE_PATH,
        "configs/benchmark/contextworld_icl_suite_v2_scoreboard_addendum_decision_v1.json",
    ],
)
def test_every_required_evidence_category_blocks_when_missing(
    finalized_fixture: tuple[Path, Path], logical_path: str
) -> None:
    root, config_path = finalized_fixture
    (root / logical_path).unlink()

    audit = reseal_v2.audit_integrity_reseal_v2_readiness(
        reseal_config=config_path, repo_root=root
    )

    assert audit["ready"] is False
    assert any(logical_path in blocker for blocker in audit["blockers"])
    with pytest.raises(reseal_v2.ResealBlocked):
        _build(root, config_path)


def test_v1_predecessor_drift_is_rejected_before_any_v2_material_is_read(
    finalized_fixture: tuple[Path, Path],
) -> None:
    root, config_path = finalized_fixture
    predecessor = root / "configs/benchmark/contextworld_icl_suite_v2_integrity_reseal_v1.yaml"
    predecessor.write_text(
        predecessor.read_text(encoding="utf-8") + "\n# drift\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="historical predecessor v1_preregistration identity drifted"):
        reseal_v2.load_integrity_reseal_v2_preregistration(
            config_path, repo_root=root
        )


def test_v1_downstream_historical_chain_drift_is_rejected(
    finalized_fixture: tuple[Path, Path],
) -> None:
    root, config_path = finalized_fixture
    historical = root / (
        "configs/benchmark/contextworld_icl_suite_v2_public_document_amendment_v1.yaml"
    )
    historical.write_text(
        historical.read_text(encoding="utf-8") + "\n# drift\n", encoding="utf-8"
    )

    with pytest.raises(
        ValueError, match="v1 historical chain documentation_amendment_config identity drifted"
    ):
        reseal_v2.load_integrity_reseal_v2_preregistration(
            config_path, repo_root=root
        )


def test_historical_eleven_row_scoreboard_bytes_must_not_drift(
    finalized_fixture: tuple[Path, Path],
) -> None:
    root, config_path = finalized_fixture
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    base_path = root / config["integrity_reseal"]["required_evidence"][
        "public_scoreboard"
    ]["historical_base_specification"]["path"]
    base_path.write_bytes(base_path.read_bytes() + b"\n")

    with pytest.raises(reseal_v2.ResealBlocked, match="historical base specification identity drifted"):
        _build(root, config_path)


def test_self_consistent_twelve_row_extension_is_rejected_before_reseal(
    finalized_fixture: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, config_path = finalized_fixture
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    evidence = config["integrity_reseal"]["required_evidence"]
    scoreboard_spec = evidence["public_scoreboard"]
    specification_path = root / scoreboard_spec["addendum_specification"]
    scoreboard_path = root / scoreboard_spec["addendum_scoreboard"]
    base_specification_path = root / scoreboard_spec["historical_base_specification"][
        "path"
    ]
    base_scoreboard_path = root / scoreboard_spec["historical_base_scoreboard"][
        "path"
    ]
    specification = json.loads(specification_path.read_text(encoding="utf-8"))
    # The public scoreboard is sorted by component and method, so an added
    # Action Strength method is deliberately not a byte-prefix extension of
    # the archived 11-row scoreboard.  Preservation must be checked by the
    # stable (component_id, method_name) key, not by row position.
    additive_row = copy.deepcopy(specification["components"][5])
    assert additive_row["component_id"] == "action_strength"
    additive_row["method_name"] = "PLDM (fixture additive)"
    specification["components"].append(additive_row)
    scoreboard = make_public_scoreboard_from_spec(specification)
    specification_path.write_text(
        json.dumps(specification, indent=2) + "\n",
        encoding="utf-8",
    )
    scoreboard_path.write_text(
        json.dumps(scoreboard, indent=2) + "\n",
        encoding="utf-8",
    )
    aggregate_spec = evidence["final_pldm_completion_aggregate_results_freeze"]
    aggregate_path = root / aggregate_spec["path"]
    resolution_spec = scoreboard_spec["additive_resolution_decision"]
    _write_json(
        root,
        resolution_spec["path"],
        {
            "schema_version": 1,
            "decision_id": resolution_spec["decision_id"],
            "status": "additive_scoreboard_extension_authorized",
            "passed": True,
            "scoreboard_extension_authorized": True,
            "formal_reference_rows": 12,
            "formal_reference_rows_added": 1,
            "historical_base_specification": _identity(
                base_specification_path,
                scoreboard_spec["historical_base_specification"]["path"],
            ),
            "historical_base_scoreboard": _identity(
                base_scoreboard_path,
                scoreboard_spec["historical_base_scoreboard"]["path"],
            ),
            "addendum_specification": _identity(
                specification_path, scoreboard_spec["addendum_specification"]
            ),
            "addendum_scoreboard": _identity(
                scoreboard_path, scoreboard_spec["addendum_scoreboard"]
            ),
            "final_pldm_completion_aggregate_results_freeze": _identity(
                aggregate_path, aggregate_spec["path"]
            ),
        },
    )

    invalid = _valid_aggregate_validation()
    invalid.update(
        {
            "formal_reference_rows": 12,
            "formal_reference_rows_added": 1,
            "components_added": ["speed"],
        }
    )
    monkeypatch.setattr(
        reseal_v2,
        "validate_written_completion_aggregate_and_scoreboard",
        lambda **_kwargs: invalid,
    )

    with pytest.raises(
        reseal_v2.ResealBlocked,
        match="final PLDM aggregate semantic validation is incomplete",
    ):
        _build(root, config_path)


def test_addendum_scoreboard_must_be_reproduced_from_its_specification(
    finalized_fixture: tuple[Path, Path],
) -> None:
    root, config_path = finalized_fixture
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    scoreboard_path = root / config["integrity_reseal"]["required_evidence"][
        "public_scoreboard"
    ]["addendum_scoreboard"]
    scoreboard = json.loads(scoreboard_path.read_text(encoding="utf-8"))
    scoreboard["aggregation_policy"] = "forged_fixture_value"
    scoreboard_path.write_text(
        json.dumps(scoreboard, indent=2) + "\n", encoding="utf-8"
    )

    with pytest.raises(
        reseal_v2.ResealBlocked, match="final addendum scoreboard cannot be reproduced"
    ):
        _build(root, config_path)


def test_action_strength_amendment_requires_a_passed_current_release_audit(
    finalized_fixture: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, config_path = finalized_fixture
    monkeypatch.setattr(
        reseal_v2,
        "audit_action_strength_icl_release",
        lambda **_kwargs: {
            "release_id": "contextworld_pusht_action_strength_icl_history3_v1",
            "status": "failed",
            "passed": False,
            "full_content_hash_audit": False,
        },
    )

    with pytest.raises(
        reseal_v2.ResealBlocked,
        match="current Action Strength release audit did not pass",
    ):
        _build(root, config_path)


def test_no_extension_decision_cannot_claim_twelve_rows(
    finalized_fixture: tuple[Path, Path],
) -> None:
    root, config_path = finalized_fixture
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    resolution_path = root / config["integrity_reseal"]["required_evidence"][
        "public_scoreboard"
    ]["additive_resolution_decision"]["path"]
    payload = json.loads(resolution_path.read_text(encoding="utf-8"))
    payload["formal_reference_rows"] = 12
    payload["formal_reference_rows_added"] = 1
    resolution_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(reseal_v2.ResealBlocked, match="row count"):
        _build(root, config_path)


def test_reseal_uses_the_production_aggregate_validator_as_a_gate(
    finalized_fixture: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, config_path = finalized_fixture
    calls: list[dict[str, Any]] = []

    def reject(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        raise ValueError("fixture aggregate semantic mismatch")

    monkeypatch.setattr(
        reseal_v2,
        "validate_written_completion_aggregate_and_scoreboard",
        reject,
    )

    with pytest.raises(
        reseal_v2.ResealBlocked,
        match="fixture aggregate semantic mismatch",
    ):
        _build(root, config_path)
    assert calls == [
        {
            "aggregate_config": (
                root
                / reseal_v2.FINAL_PLDMS_AGGREGATE_PREREGISTRATION_PATH
            ),
            "repo_root": root,
        }
    ]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("speed_evidence_scope", "training_attributed"),
        ("speed_training_attribution_claim", True),
        ("action_strength_formal_row_included", False),
        ("action_strength_ability_passed", True),
        (
            "development_only_components_not_added",
            ["contact_friction"],
        ),
        ("components_added", ["speed", "contact_friction"]),
    ],
)
def test_reseal_rejects_invalid_formal_addendum_semantics(
    finalized_fixture: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: Any,
) -> None:
    root, config_path = finalized_fixture
    invalid = _valid_aggregate_validation()
    invalid[field] = value
    monkeypatch.setattr(
        reseal_v2,
        "validate_written_completion_aggregate_and_scoreboard",
        lambda **_kwargs: invalid,
    )

    with pytest.raises(
        reseal_v2.ResealBlocked,
        match="final PLDM aggregate semantic validation is incomplete",
    ):
        _build(root, config_path)


def test_decision_writer_rejects_noncanonical_output_before_building(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "ContextWorld"
    root.mkdir()
    called = False

    def should_not_build(**_kwargs: Any) -> dict[str, Any]:
        nonlocal called
        called = True
        raise AssertionError("writer should validate its output path first")

    monkeypatch.setattr(
        reseal_v2,
        "build_integrity_reseal_v2_decision",
        should_not_build,
    )

    with pytest.raises(ValueError, match="canonical output path"):
        reseal_v2.write_integrity_reseal_v2_decision(
            root / "outside-decision.json", repo_root=root
        )
    assert called is False


def test_decision_writer_rejects_a_symlinked_canonical_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "ContextWorld"
    root.mkdir()
    external = tmp_path / "external-configs"
    external.mkdir()
    (root / "configs").symlink_to(external, target_is_directory=True)
    monkeypatch.setattr(
        reseal_v2,
        "build_integrity_reseal_v2_decision",
        lambda **_kwargs: pytest.fail("writer must reject the symlink before building"),
    )

    with pytest.raises(RuntimeError, match="traverses a symlink"):
        reseal_v2.write_integrity_reseal_v2_decision(
            reseal_v2.RESEAL_DECISION_PATH, repo_root=root
        )


def test_decision_writer_creates_only_the_canonical_local_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "ContextWorld"
    target = root / reseal_v2.RESEAL_DECISION_PATH
    target.parent.mkdir(parents=True)
    decision = {"schema_version": 1, "reseal_id": reseal_v2.RESEAL_ID}
    monkeypatch.setattr(
        reseal_v2,
        "build_integrity_reseal_v2_decision",
        lambda **_kwargs: decision,
    )

    observed = reseal_v2.write_integrity_reseal_v2_decision(
        target, repo_root=root
    )

    assert observed == decision
    assert json.loads(target.read_text(encoding="utf-8")) == decision
    with pytest.raises(FileExistsError):
        reseal_v2.write_integrity_reseal_v2_decision(target, repo_root=root)
