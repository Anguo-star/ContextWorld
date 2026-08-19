from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import pytest

import contextworld.benchmarks.suite_data as suite_data
import contextworld.benchmarks.suite_v2_integrity_reseal as integrity_reseal
from contextworld.benchmarks.suite_data import (
    COMPONENT_IDS,
    DEFAULT_SUITE_V2_RELEASE_CONFIG,
    SUITE_V2_INTEGRITY_RESEAL_ID,
    SUITE_V2_RECOVERY_CONFIG,
    SUITE_V2_COMPONENT_IDS,
    audit_icl_suite_release,
    export_icl_suite_artifacts,
    load_icl_suite_release,
    load_public_scoreboard,
    require_suite_membership_activation,
)
from contextworld.paths import resolve_contextworld_path
from contextworld.benchmarks.suite_v2_integrity_reseal import (
    build_integrity_reseal_decision,
)


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _install_fake_cem_result_freeze(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    logical_path = integrity_reseal.DESCRIPTIVE_RESULT_FREEZE_SPECS[
        "original_baseline_cem"
    ]["path"]
    fake = tmp_path / "contextworld_original_baseline_cem_results_freeze_v1.json"
    fake.write_text(
        json.dumps(
            {
                "freeze_id": integrity_reseal.DESCRIPTIVE_RESULT_FREEZE_SPECS[
                    "original_baseline_cem"
                ]["freeze_id"],
                "status": "frozen_test_fixture",
            }
        ),
        encoding="utf-8",
    )
    original_resolve = integrity_reseal.resolve_contextworld_path

    def resolve_with_fake_cem(value: str | Path, **kwargs: object) -> Path:
        if str(value) == logical_path:
            return fake
        return original_resolve(value, **kwargs)

    monkeypatch.setattr(
        integrity_reseal,
        "resolve_contextworld_path",
        resolve_with_fake_cem,
    )


def _install_temporary_reseal_decision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    decision: dict[str, object] | None = None,
) -> tuple[dict[str, object], Path]:
    _install_fake_cem_result_freeze(tmp_path, monkeypatch)
    suite = load_icl_suite_release(DEFAULT_SUITE_V2_RELEASE_CONFIG)
    payload = decision or build_integrity_reseal_decision(repo_root=ROOT)
    decision_path = tmp_path / "integrity-reseal-decision.json"
    decision_path.write_text(json.dumps(payload), encoding="utf-8")
    original_resolve = suite_data.resolve_no_symlink_contextworld_path

    def resolve_with_temporary_decision(
        value: str | Path,
        *,
        repo_root: Path | None = None,
        label: str,
        allow_missing: bool = False,
    ) -> Path:
        if str(value) == suite["membership_authority"]["decision_path"]:
            return decision_path
        return original_resolve(
            value,
            repo_root=repo_root,
            label=label,
            allow_missing=allow_missing,
        )

    monkeypatch.setattr(
        suite_data,
        "resolve_no_symlink_contextworld_path",
        resolve_with_temporary_decision,
    )
    monkeypatch.setattr(
        suite_data,
        "_validate_integrity_reseal_current_document",
        lambda _path: {"passed": True, "fixture": True},
    )
    return suite, decision_path


def test_suite_v2_adds_cube_without_rewriting_suite_v1() -> None:
    suite_v1 = load_icl_suite_release()
    suite_v2 = load_icl_suite_release(DEFAULT_SUITE_V2_RELEASE_CONFIG)

    assert tuple(suite_v1["components"]) == COMPONENT_IDS
    assert len(suite_v1["components"]) == 8
    assert suite_v1["public_results"]["formal_reference_rows"] == 10
    assert tuple(suite_v2["components"]) == SUITE_V2_COMPONENT_IDS
    assert len(suite_v2["components"]) == 9
    assert suite_v2["public_results"]["formal_reference_rows"] == 11
    assert len(suite_v2["public_results"]["components_with_formal_results"]) == 7
    authority = suite_v2["membership_authority"]
    assert authority["config_alone_grants_membership"] is False
    assert authority["activation_condition"] == (
        "passed_integrity_reseal_decision_v1"
    )
    assert authority["reseal_id"] == SUITE_V2_INTEGRITY_RESEAL_ID
    assert authority["decision_path"].endswith(
        "contextworld_icl_suite_v2_integrity_reseal_decision_v1.json"
    )
    assert authority["decision_is_commit_marker"] is True
    assert authority["partial_outputs_grant_membership"] is False
    assert authority["historical_chain_must_remain_byte_identical"] is True
    assert authority["old_membership_may_not_be_silently_reactivated"] is True
    assert authority["formal_scoreboard_mutation_authorized"] is False
    assert authority["public_test_rerun_authorized"] is False

    recovery = load_icl_suite_release(SUITE_V2_RECOVERY_CONFIG)
    assert recovery["membership_authority"]["activation_condition"] == (
        "passed_registration_decision_v2"
    )


def test_suite_v2_scoreboard_contains_one_lewm_only_cube_row() -> None:
    suite = load_icl_suite_release(DEFAULT_SUITE_V2_RELEASE_CONFIG)
    scoreboard_path = resolve_contextworld_path(
        suite["public_results"]["scoreboard"]["path"], repo_root=ROOT
    )
    scoreboard = json.loads(scoreboard_path.read_text(encoding="utf-8"))
    rows = scoreboard["component_results"]
    cube_rows = [row for row in rows if row["component_id"] == "cube_gripper_carry"]

    assert len(rows) == 11
    assert len(cube_rows) == 1
    assert cube_rows[0]["method_name"].startswith("LeWM")
    assert "PLDM" not in cube_rows[0]["method_name"]
    assert cube_rows[0]["icl_ability"]["result"] == "PASS"
    assert cube_rows[0]["icl_ability"]["primary_metric"]["mean"] == (
        0.7845052083333334
    )
    assert not any(
        key in cube_rows[0]
        for key in (
            "submission_kind",
            "claim_boundary",
            "external_result",
            "formal_scoreboard_eligible",
        )
    )


def test_suite_v2_integrity_reseal_activates_only_with_a_new_decision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    suite, _ = _install_temporary_reseal_decision(tmp_path, monkeypatch)
    activation = require_suite_membership_activation(suite, repo_root=ROOT)

    assert activation["status"] == (
        "suite_registration_passed_with_integrity_reseal_v1"
    )
    assert activation["historical_chain_preserved"] is True
    assert activation["old_membership_silently_reactivated"] is False
    assert activation["current_document_contract"] == {
        "passed": True,
        "fixture": True,
    }
    assert set(activation["release_materials"]["components"]) == set(
        SUITE_V2_COMPONENT_IDS
    )

    document = (ROOT / "docs/ContextWorld_ICL_Benchmark.md").read_text(
        encoding="utf-8"
    )
    assert "#### 6.3.3 Cube 夹爪携带规则" in document
    assert suite_data._audit_public_document_template(
        ROOT / "docs/ContextWorld_ICL_Benchmark.md", suite
    )["passed"] is True
    assert len(
        load_public_scoreboard(
            DEFAULT_SUITE_V2_RELEASE_CONFIG, repo_root=ROOT
        )["component_results"]
    ) == 11


def test_suite_v2_reseal_decision_rebinds_current_identities(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    suite, _ = _install_temporary_reseal_decision(tmp_path, monkeypatch)
    require_suite_membership_activation(suite, repo_root=ROOT)
    for logical, expected in suite["repository"]["source_sha256"].items():
        assert _sha256(ROOT / logical) == expected
    document = suite["repository"]["public_document"]
    assert _sha256(ROOT / document["path"]) == document["sha256"]
    for component in suite["components"].values():
        assert _sha256(ROOT / component["release_config"]) == component[
            "release_config_sha256"
        ]


def test_suite_v2_reseal_rejects_a_drifted_temporary_decision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    suite, decision_path = _install_temporary_reseal_decision(
        tmp_path, monkeypatch
    )
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    decision["release_materials"]["components"]["speed"]["sha256"] = "0" * 64
    decision_path.write_text(json.dumps(decision), encoding="utf-8")

    with pytest.raises(RuntimeError, match="integrity reseal decision drifted"):
        require_suite_membership_activation(suite, repo_root=ROOT)


def test_suite_v2_reseal_current_document_contract_uses_new_section5_shape(
    tmp_path: Path,
) -> None:
    def table(rows: int) -> str:
        body = "\n".join(f"| r{index} | value |" for index in range(rows))
        return "| task | result |\n|---|---|\n" + body

    document = tmp_path / "benchmark.md"
    document.write_text(
        "\n\n".join(
            (
                "### 5.1 Original ICL\n" + table(18),
                "### 5.2 Original environment CEM\n" + table(8),
                "### 5.3 Trained 18 slots\n" + table(18),
                "### 5.4 Cube external slots\n" + table(3),
                "### 5.5 Cube recovery\nclosed",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    audit = suite_data._validate_integrity_reseal_current_document(document)

    assert audit["passed"] is True
    assert audit["table_rows"] == {
        "### 5.1 ": 18,
        "### 5.2 ": 8,
        "### 5.3 ": 18,
        "### 5.4 ": 3,
        "### 5.5 ": 0,
    }

    document.write_text(
        document.read_text(encoding="utf-8").replace("| r17 | value |\n", ""),
        encoding="utf-8",
    )
    assert suite_data._validate_integrity_reseal_current_document(document)[
        "passed"
    ] is False


def test_suite_v2_membership_fails_closed_while_historical_results_remain_readable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    suite = load_icl_suite_release(DEFAULT_SUITE_V2_RELEASE_CONFIG)
    original_resolve = suite_data.resolve_no_symlink_contextworld_path
    partial_root = tmp_path / "cube-registration"
    partial_root.mkdir()
    for name in (
        "component_release_audit_v2.json",
        "suite_v2_audit_v2.json",
        "suite_v2_export_audit_v2.json",
        "export_reservation_v2.json",
        "suite_v2_copy_complete_v2.json",
    ):
        (partial_root / name).write_text("{}\n", encoding="utf-8")
    (partial_root / "suite_v2_copy_export_v2").mkdir()

    def resolve_without_decision(
        value: str | Path,
        *,
        repo_root: Path | None = None,
        label: str,
        allow_missing: bool = False,
    ) -> Path:
        if str(value) == suite["membership_authority"]["decision_path"]:
            return partial_root / "registration_decision_v2.json"
        return original_resolve(
            value,
            repo_root=repo_root,
            label=label,
            allow_missing=allow_missing,
        )

    monkeypatch.setattr(
        suite_data,
        "resolve_no_symlink_contextworld_path",
        resolve_without_decision,
    )
    with pytest.raises(RuntimeError, match="decision is missing"):
        require_suite_membership_activation(suite, repo_root=ROOT)
    archive = load_public_scoreboard(DEFAULT_SUITE_V2_RELEASE_CONFIG, repo_root=ROOT)
    assert len(archive["component_results"]) == 11


def test_suite_v2_membership_rejects_a_symlinked_decision(
    tmp_path: Path,
) -> None:
    suite = load_icl_suite_release(DEFAULT_SUITE_V2_RELEASE_CONFIG)
    logical = Path(suite["membership_authority"]["decision_path"])
    decision = tmp_path / logical
    decision.parent.mkdir(parents=True)
    target = tmp_path / "redirected-decision.json"
    target.write_text("{}\n", encoding="utf-8")
    decision.symlink_to(target)

    with pytest.raises(RuntimeError, match="traverses a symlink"):
        require_suite_membership_activation(suite, repo_root=tmp_path)


def test_public_suite_apis_do_not_expose_a_pending_registration_bypass() -> None:
    for function in (
        require_suite_membership_activation,
        load_public_scoreboard,
        audit_icl_suite_release,
        export_icl_suite_artifacts,
    ):
        parameters = inspect.signature(function).parameters
        assert "allow_pending_registration" not in parameters
        assert "registration_capability" not in parameters
