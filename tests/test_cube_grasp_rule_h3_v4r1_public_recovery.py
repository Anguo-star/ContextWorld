from __future__ import annotations

import copy
import inspect
from pathlib import Path

import pytest
import yaml

from contextworld.benchmarks.cube_grasp_rule_public_contract import (
    PROTOCOL_ID,
    file_identity,
)
from contextworld.benchmarks.cube_grasp_rule_public_recovery_contract import (
    DECISION_ID,
    DEFAULT_FREEZE_RECEIPT,
    DEFAULT_PREREGISTRATION,
    EXPECTED_PLANNED_ARTIFACTS,
    FREEZE_RECEIPT_ID,
    PREREGISTRATION_ID,
    RECOVERY_AUTHORIZATION_ID,
    validate_recovery_lineage,
    validate_recovery_preregistration_contract,
)
import scripts.build_cube_grasp_rule_h3_v4r1_public_data as base_builder
import scripts.finalize_cube_grasp_rule_h3_v4r1_public_release as base_finalizer
import scripts.run_cube_grasp_rule_h3_v4r1_public_matrix as base_runner


ROOT = Path(__file__).resolve().parents[1]
FAILED_PREREG = (
    ROOT
    / "configs/benchmark/"
    "cube_gripper_carry_h3_v4r1_public_release_prereg_v1.yaml"
)
CURRENT_RELEASE_CONFIG = (
    ROOT
    / "configs/benchmark/"
    "cube_gripper_carry_h3_v4r1_icl_release_v1.yaml"
)
ENGINEERING_IDENTITY_AMENDMENT = (
    ROOT
    / "configs/benchmark/"
    "contextworld_icl_suite_v2_engineering_identity_amendment_prereg_v1.yaml"
)


def _load(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_recovery_preregistration_preserves_science_and_uses_new_namespaces() -> None:
    failed = _load(FAILED_PREREG)
    recovery = _load(DEFAULT_PREREGISTRATION)
    validate_recovery_preregistration_contract(recovery)

    assert recovery["preregistration_id"] == PREREGISTRATION_ID
    assert recovery["recovery_authorization_id"] == RECOVERY_AUTHORIZATION_ID
    assert recovery["protocol_id"] == PROTOCOL_ID
    assert recovery["planned_artifacts"] == EXPECTED_PLANNED_ARTIFACTS
    assert str(DEFAULT_FREEZE_RECEIPT).endswith(
        "cube_gripper_carry_h3_public_recovery_v1/"
        "public_recovery_freeze_receipt_v1.json"
    )
    assert FREEZE_RECEIPT_ID.endswith("public_recovery_freeze_v1")
    assert DECISION_ID.endswith("public_recovery_decision_v1")

    for name in (
        "scope",
        "public_data_generation",
        "public_evaluation",
        "scoring",
        "one_use_policy",
        "outcomes",
    ):
        assert recovery[name] == failed[name]


def test_recovery_lineage_is_exact_and_semantically_closed() -> None:
    recovery = _load(DEFAULT_PREREGISTRATION)
    observed = validate_recovery_lineage(recovery, root=ROOT)
    assert observed == recovery["recovery_lineage"]

    drifted = copy.deepcopy(recovery)
    drifted["recovery"]["failed_public_model_read"] = True
    with pytest.raises(RuntimeError, match="lineage declaration drifted"):
        validate_recovery_preregistration_contract(drifted)


def test_recovery_preregistration_identities_are_historical_or_amended() -> None:
    """Keep recovery evidence immutable while checking the active Cube chain.

    The recovery preregistration is historical evidence and must retain the
    code identities present when its one-use campaign was frozen.  A later,
    additive suite amendment may register a current implementation identity;
    it must not be simulated by editing that historical preregistration.
    """

    recovery = _load(DEFAULT_PREREGISTRATION)
    amendment = _load(ENGINEERING_IDENTITY_AMENDMENT)[
        "engineering_identity_amendment"
    ]
    cube_update = amendment["approved_component_identity_updates"][
        "cube_gripper_carry"
    ]
    assert cube_update["release_config"] == (
        "configs/benchmark/cube_gripper_carry_h3_v4r1_icl_release_v1.yaml"
    )
    active_release = _load(CURRENT_RELEASE_CONFIG)
    active_by_path = {
        expected["path"]: expected
        for expected in active_release["identity"].values()
    }

    amended = set()
    groups = (
        recovery["identity"]["implementation"],
        recovery["identity"]["recovery_implementation"],
    )
    for group in groups:
        for name, expected in group.items():
            assert "PLACEHOLDER" not in str(expected), name
            path = ROOT / expected["path"]
            observed = file_identity(path, logical_path=expected["path"])
            if observed == expected:
                continue
            amended.add(name)
            assert expected["path"] in cube_update["source_paths"]
            active_identity = active_by_path.get(expected["path"])
            if active_identity is not None:
                assert active_identity["sha256"] == observed["sha256"]

    # The four current identities are precisely the Cube-scoped changes
    # permitted by the additive engineering amendment.  The fifth is this
    # assertion: it records that the amendment, rather than a rewrite of the
    # recovery preregistration, authorizes the current identity check.
    assert amended == {
        "adapters",
        "cube_score_api",
        "package",
        "public_api",
        "recovery_contract_tests",
    }


def test_shared_entrypoints_expose_recovery_loader_injection() -> None:
    assert "authorization_loader" in inspect.signature(
        base_builder.build_public_data
    ).parameters
    assert "authorization_loader" in inspect.signature(
        base_runner.run_public_matrix
    ).parameters
    finalizer_parameters = inspect.signature(
        base_finalizer.finalize_public_release
    ).parameters
    assert "authorization_loader" in finalizer_parameters
    assert "decision_id" in finalizer_parameters
