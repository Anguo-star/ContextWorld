from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path

import pytest
import yaml

from contextworld.benchmarks import cube_grasp_rule_v4r1_icl_score as score_api
from contextworld.benchmarks.cube_grasp_rule_v4r1_icl_cli import (
    _assert_external_output,
)
from contextworld.benchmarks.cube_grasp_rule_v4r1_icl_data import (
    CubeGraspRuleV4R1ICLEvalDataset,
    audit_cube_grasp_rule_v4r1_icl_release,
    directory_identity,
    load_cube_grasp_rule_v4r1_icl_release,
    recompute_cube_grasp_rule_v4r1_public_reference,
)
from contextworld.paths import resolve_contextworld_path


def test_v4r1_release_has_exact_five_dimensional_contract() -> None:
    release = load_cube_grasp_rule_v4r1_icl_release()
    assert release["component_id"] == "cube_gripper_carry"
    assert release["scope"] | {} == {
        "environment": "Cube",
        "capability": "infer_hidden_gripper_carry_rule_from_recent_interaction",
        "display_name_zh": "Cube 夹爪携带规则 ICL",
        "history_tokens": 3,
        "context_transitions": 2,
        "raw_action_dim": 5,
        "raw_steps_per_action_block": 5,
        "flattened_action_input_dim": 25,
        "prediction_horizon_action_blocks": 1,
        "grasp_modes": ["cannot_hold", "can_hold"],
        "hidden_values": {"cannot_hold": 0.0, "can_hold": 1.0},
        "public_test_included": True,
        "sealed_test_included": False,
    }


def test_v4r1_release_loader_rejects_dimension_drift(tmp_path: Path) -> None:
    source = load_cube_grasp_rule_v4r1_icl_release()
    payload = {
        key: value for key, value in source.items() if not key.startswith("_")
    }
    payload["scope"] = {**payload["scope"], "raw_action_dim": 2}
    config = tmp_path / "release.yaml"
    config.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="5D History=3"):
        load_cube_grasp_rule_v4r1_icl_release(config)


def test_v4r1_bundle_recomputes_canonical_reference_and_retention() -> None:
    release = load_cube_grasp_rule_v4r1_icl_release()
    reference = recompute_cube_grasp_rule_v4r1_public_reference(
        release, layout="bundle"
    )
    assert reference["passed"] is True
    assert reference["external_result"] is False
    assert reference["training_seeds"] == [17321, 17322, 17323]
    assert [
        reference["per_seed"][str(seed)]["metrics"]["correct_future_rate"]
        for seed in reference["training_seeds"]
    ] == [0.77734375, 0.791015625, 0.78515625]
    assert [
        row["candidate_successes"]
        for row in reference["original_task_retention"]["comparisons"]
    ] == [186, 183, 185]


def test_v4r1_full_bundle_audit_and_public_shape() -> None:
    audit = audit_cube_grasp_rule_v4r1_icl_release(
        layout="bundle", full=True
    )
    failed_files = {
        name for name, result in audit["files"].items() if not result["passed"]
    }
    assert failed_files == {"identity.package"}
    package = audit["files"]["identity.package"]
    root = Path(__file__).resolve().parents[1]
    release_path = (
        root
        / "configs/benchmark/cube_gripper_carry_h3_v4r1_icl_release_v1.yaml"
    )
    correction = yaml.safe_load(
        (
            root
            / "configs/benchmark/contextworld_historical_package_pin_correction_v1.yaml"
        ).read_text(encoding="utf-8")
    )
    package_row = next(
        row
        for row in correction["affected_records"]
        if row["config"]["path"]
        == "configs/benchmark/cube_gripper_carry_h3_v4r1_icl_release_v1.yaml"
    )
    assert package_row["field"] == "identity.package.sha256"
    assert package_row["config"] == {
        "path": "configs/benchmark/cube_gripper_carry_h3_v4r1_icl_release_v1.yaml",
        "sha256": hashlib.sha256(release_path.read_bytes()).hexdigest(),
        "size_bytes": release_path.stat().st_size,
    }
    assert package["expected_sha256"] == correction["finding"]["invalid_sha256"]
    assert package["observed_sha256"] != package["expected_sha256"]
    assert audit["source_historical_evidence_revalidated"] is False
    assert audit["artifact_tree"]["passed"] is True
    assert all(result["passed"] for result in audit["tables"].values())
    assert audit["causal_data_contract"]["passed"] is True
    assert audit["public_reference"]["passed"] is True
    assert audit["public_test"]["raw_action_dim"] == 5
    dataset = CubeGraspRuleV4R1ICLEvalDataset(layout="bundle")
    assert dataset.arrays.raw_action_blocks.shape == (256, 4, 5, 5)


def test_directory_identity_rejects_nested_symlink(tmp_path: Path) -> None:
    tree = tmp_path / "tree"
    real = tmp_path / "real"
    tree.mkdir()
    real.mkdir()
    (real / "value.txt").write_text("value\n", encoding="utf-8")
    (tree / "nested").symlink_to(real, target_is_directory=True)
    with pytest.raises(RuntimeError, match="symlink"):
        directory_identity(tree)


def test_external_policy_fails_before_adapter_or_public_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release = load_cube_grasp_rule_v4r1_icl_release()
    unauthorized = deepcopy(release)
    unauthorized["claim_boundary"]["external_evaluation"] = {
        **unauthorized["claim_boundary"]["external_evaluation"],
        "reference_rerun": True,
    }
    monkeypatch.setattr(
        score_api,
        "load_cube_grasp_rule_v4r1_icl_release",
        lambda *args, **kwargs: unauthorized,
    )
    with pytest.raises(RuntimeError, match="not authorized"):
        score_api.evaluate_cube_grasp_rule_v4r1_icl_model(
            adapter=None,  # type: ignore[arg-type]
            model_name="must-not-open",
            training_recipe="must-not-open",
            training_seed=1,
        )


def test_external_result_claim_is_formally_ineligible() -> None:
    assert score_api.EXTERNAL_RESULT_CLAIM_BOUNDARY == {
        "external_result": True,
        "external_evaluation_allowed": True,
        "formal_reference_mutation": False,
        "formal_scoreboard_eligible": False,
        "reference_rerun": False,
    }


@pytest.mark.parametrize("identity_kind", ("hash", "path", "state"))
def test_external_api_rejects_every_canonical_checkpoint_identity(
    identity_kind: str,
) -> None:
    release = load_cube_grasp_rule_v4r1_icl_release()
    canonical = release["reference_method"]["checkpoints"][0]
    arguments = {
        "checkpoint_sha256": "0" * 64,
        "checkpoint_path": "/tmp/external-cube-checkpoint.pt",
        "model_state_sha256": "1" * 64,
    }
    arguments[
        {"hash": "checkpoint_sha256", "path": "checkpoint_path", "state": "model_state_sha256"}[
            identity_kind
        ]
    ] = {
        "hash": canonical["sha256"],
        "path": canonical["path"],
        "state": canonical["model_state_sha256"],
    }[identity_kind]
    with pytest.raises(RuntimeError, match="cannot be rerun"):
        score_api.validate_cube_grasp_rule_v4r1_external_checkpoint_identity(
            release, **arguments
        )


@pytest.mark.parametrize(
    "logical",
    [
        "configs/external-result.json",
        "contextworld/external-result.json",
        "scripts/external-result.json",
        "tests/external-result.json",
        "docs/external-result.json",
        "artifacts/evaluation/history3/cube_gripper_carry_h3_public_recovery_v1/external.json",
    ],
)
def test_cli_rejects_repository_and_frozen_outputs(logical: str) -> None:
    release = load_cube_grasp_rule_v4r1_icl_release()
    root = Path(release["_config_path"]).parents[2]
    output = (
        resolve_contextworld_path(logical, repo_root=root)
        if logical.startswith("artifacts/")
        else root / logical
    )
    with pytest.raises(RuntimeError, match="cannot"):
        _assert_external_output(output, release, repo_root=root)
