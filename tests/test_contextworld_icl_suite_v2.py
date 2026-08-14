from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import pytest

import contextworld.benchmarks.suite_data as suite_data
from contextworld.benchmarks.suite_data import (
    COMPONENT_IDS,
    DEFAULT_SUITE_V2_RELEASE_CONFIG,
    SUITE_V2_COMPONENT_IDS,
    audit_icl_suite_release,
    export_icl_suite_artifacts,
    load_icl_suite_release,
    load_public_scoreboard,
    require_suite_membership_activation,
)
from contextworld.paths import resolve_contextworld_path


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    assert authority["activation_condition"] == "passed_registration_decision_v2"
    assert authority["registration_id"].endswith("suite_registration_recovery_v2")
    assert authority["decision_path"].endswith(
        "suite_registration_recovery_v2/registration_decision_v2.json"
    )
    assert authority["decision_is_commit_marker"] is True
    assert authority["partial_outputs_grant_membership"] is False
    assert authority["directory_rename_authorized"] is False
    assert authority["prior_failed_staging_reuse_authorized"] is False


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


def test_suite_v2_repository_identities_match_current_sources() -> None:
    suite = load_icl_suite_release(DEFAULT_SUITE_V2_RELEASE_CONFIG)
    for logical, expected in suite["repository"]["source_sha256"].items():
        assert _sha256(ROOT / logical) == expected
    document = suite["repository"]["public_document"]
    assert _sha256(ROOT / document["path"]) == document["sha256"]
    for component in suite["components"].values():
        assert _sha256(ROOT / component["release_config"]) == component[
            "release_config_sha256"
        ]


def test_suite_v2_membership_fails_closed_without_canonical_decision(
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
    with pytest.raises(RuntimeError, match="decision is missing"):
        load_public_scoreboard(DEFAULT_SUITE_V2_RELEASE_CONFIG, repo_root=ROOT)


def test_suite_v2_membership_rejects_a_symlinked_decision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    suite = load_icl_suite_release(DEFAULT_SUITE_V2_RELEASE_CONFIG)
    monkeypatch.setenv("CONTEXTWORLD_ARTIFACT_ROOT", str(tmp_path))
    decision = tmp_path.joinpath(
        *Path(suite["membership_authority"]["decision_path"]).parts[1:]
    )
    decision.parent.mkdir(parents=True)
    target = tmp_path / "redirected-decision.json"
    target.write_text("{}\n", encoding="utf-8")
    decision.symlink_to(target)

    with pytest.raises(RuntimeError, match="traverses a symlink"):
        require_suite_membership_activation(suite, repo_root=ROOT)


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
