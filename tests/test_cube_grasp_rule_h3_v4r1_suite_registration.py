from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from contextworld.benchmarks.cube_grasp_rule_suite_registration import (
    EXPECTED_BASIS_KEYS,
    EXPECTED_SEEDS,
    read_yaml,
    require_no_symlink_components,
    validate_historical_evidence,
    validate_registration_preregistration_contract,
)
from contextworld.paths import repository_root
import scripts.freeze_cube_grasp_rule_h3_v4r1_suite_registration as freezer
import scripts.finalize_cube_grasp_rule_h3_v4r1_suite_registration as finalizer
import scripts.package_cube_grasp_rule_h3_v4r1_icl_release as packager


PREREGISTRATION = repository_root() / (
    "configs/benchmark/"
    "cube_gripper_carry_h3_v4r1_suite_registration_prereg_v1.yaml"
)


def test_registration_preregistration_is_fail_closed() -> None:
    prereg = read_yaml(PREREGISTRATION, label="registration preregistration")
    validate_registration_preregistration_contract(
        prereg, preregistration_path=PREREGISTRATION
    )
    assert set(prereg["authorization_basis"]) == EXPECTED_BASIS_KEYS
    assert prereg["packaging_contract"]["projection_contains"] == [
        "_PACKAGING_STARTED.json",
        "train.lance",
        "loader_validation.lance",
        "validation.lance",
        "portable_provenance.json",
        "_SUCCESS.json",
    ]
    assert prereg["packaging_contract"]["rerun_in_same_namespace_authorized"] is False


def test_historical_chain_and_three_public_gates_recompute() -> None:
    prereg = read_yaml(PREREGISTRATION, label="registration preregistration")
    evidence = validate_historical_evidence(prereg)
    reference = evidence["public_reference"]
    assert reference["passed"] is True
    assert tuple(int(seed) for seed in reference["checkpoint_results"]) == EXPECTED_SEEDS
    assert all(
        row["gate"]["passed"] is True
        for row in reference["checkpoint_results"].values()
    )
    assert evidence["original_task_retention"]["passed"] is True
    assert evidence["data_contract"]["causal_data_contract"] == {
        "development_passed": True,
        "public_passed": True,
        "passed": True,
    }
    assert set(evidence["data_contract"]["source_table_report_bindings"]) == {
        "train",
        "loader_validation",
        "validation",
    }


@pytest.mark.parametrize(
    ("section", "key", "replacement"),
    [
        (
            "authorization_basis",
            "public_matrix_score",
            {"path": "x", "sha256": "0" * 64, "size_bytes": 1},
        ),
        (
            "source_tables",
            "train",
            {
                "source": "artifacts/x",
                "pair_count": 2048,
                "files": 3,
                "bytes": 1,
                "sha256": "0" * 64,
            },
        ),
        ("registration_gates", "full_suite_audit_required", False),
        ("planned_repository_outputs", "release_config", "configs/benchmark/wrong.yaml"),
        ("planned_artifacts", "component_audit", "artifacts/wrong.json"),
        ("allowed_claim_after_all_gates_pass", "distribution", "public"),
    ],
)
def test_registration_contract_rejects_every_mutable_authority_surface(
    section: str, key: str, replacement: object
) -> None:
    prereg = read_yaml(PREREGISTRATION, label="registration preregistration")
    changed = deepcopy(prereg)
    changed[section][key] = replacement
    with pytest.raises(RuntimeError):
        validate_registration_preregistration_contract(
            changed, preregistration_path=PREREGISTRATION
        )


def test_historical_tables_are_bound_to_frozen_build_reports() -> None:
    prereg = read_yaml(PREREGISTRATION, label="registration preregistration")
    changed = deepcopy(prereg)
    changed["source_tables"]["train"]["sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="bound to its build report"):
        validate_historical_evidence(changed)


def test_symlink_components_are_rejected(tmp_path: Path) -> None:
    trusted = tmp_path / "trusted"
    actual = tmp_path / "actual"
    trusted.mkdir()
    actual.mkdir()
    (actual / "receipt.json").write_text("{}\n", encoding="utf-8")
    (trusted / "linked").symlink_to(actual, target_is_directory=True)
    with pytest.raises(RuntimeError, match="traverses a symlink"):
        require_no_symlink_components(
            trusted / "linked" / "receipt.json",
            anchor=trusted,
            label="test receipt",
        )


def test_packager_reserves_the_target_before_historical_reads() -> None:
    source = Path(packager.__file__).read_text(encoding="utf-8")
    reservation = source.index("output.mkdir()")
    started = source.index('"_PACKAGING_STARTED.json"')
    historical = source.index("validate_historical_evidence(prereg", reservation)
    copy = source.index("shutil.copytree", historical)
    assert reservation < started < historical < copy
    assert "tempfile" not in source
    assert "shutil.rmtree" not in source
    assert '"data_contract": evidence["data_contract"]' in source


def test_freezer_binds_all_packaging_implementations() -> None:
    assert set(freezer.IMPLEMENTATION_PATHS) == {
        "registration_contract",
        "packaging_script",
        "registration_freezer",
    }


def test_finalizer_rejects_noncanonical_formal_inputs(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="canonical path"):
        finalizer.finalize_registration(preregistration=tmp_path / "copy.yaml")


def test_finalizer_commits_only_after_all_audits_pass() -> None:
    source = Path(finalizer.__file__).read_text(encoding="utf-8")
    component_audit = source.index("component_audit = audit_cube")
    suite_audit = source.index(
        "suite_audit = _audit_icl_suite_release_for_registration"
    )
    bundle_audit = source.index(
        "bundle_audit = _audit_icl_suite_release_for_registration"
    )
    atomic_commit = source.index("os.replace(export_staging, export_destination)")
    first_formal_write = source.index(
        '_write_json(outputs["component_audit"], component_audit)'
    )
    decision_write = source.index(
        '_write_json(outputs["registration_decision"], decision)'
    )

    assert component_audit < suite_audit < bundle_audit < atomic_commit
    assert atomic_commit < first_formal_write < decision_write
    assert '"registration_decision_is_commit_marker": True' in source
    assert '"partial_outputs_grant_membership": False' in source
    assert 'freeze.get("receipt_id") != FREEZE_RECEIPT_ID' in source
    assert (
        'freeze.get("implementation_identities") != implementation_identities'
        in source
    )


def test_finalizer_records_the_local_dependency_boundary() -> None:
    source = Path(finalizer.__file__).read_text(encoding="utf-8")
    for claim in (
        '"self_contained": False',
        '"requires_contextworld_checkout": True',
        '"external_upstream_dependencies_required": True',
        '"upstream_not_redistributed": True',
        '"cube_formal_checkpoint_opened": False',
        '"non_cube_checkpoint_identity_audit": True',
    ):
        assert claim in source
    assert '"model_or_checkpoint_read"' not in source
