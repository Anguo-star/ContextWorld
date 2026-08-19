from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import yaml

import contextworld.benchmarks as public_benchmarks
from contextworld.benchmarks.adapters import (
    AdapterProtocol,
    StableWorldModelLeWMCubeGraspRuleAdapter,
    StableWorldModelLeWMReacherArmMassAdapter,
    StableWorldModelPLDMCubeGraspRuleAdapter,
)
from contextworld.evaluation.cube_grasp_rule_h3 import (
    QUERY_STATE_TOLERANCE,
    CubeGraspRuleCandidate,
    CubeGraspRuleSimulator,
)
from contextworld.benchmarks.cube_grasp_rule_icl_score import (
    _validate_cube_adapter_protocol,
    cube_grasp_rule_prediction_metrics,
    score_cube_grasp_rule_icl_results,
)
from contextworld.benchmarks.cube_grasp_rule_icl_data import (
    audit_cube_grasp_rule_icl_release,
    file_sha256,
    load_cube_grasp_rule_icl_release,
)
from contextworld.evaluation.protocol import ColumnStandardizer


CUBE_SOURCE = Path(
    "/opt/huawei/explorer-env/dataset/ag_data/data/world_model/"
    "quentinll/lewm-cube/ogbench/cube_single_expert.h5"
)
ROOT = Path(__file__).resolve().parents[1]


def test_legacy_cube_v1_stays_out_of_suite_v1_while_v4r1_is_public() -> None:
    assert "CubeGraspRuleV4R1ICLEvalDataset" in public_benchmarks.__all__
    assert "audit_cube_grasp_rule_v4r1_icl_release" in public_benchmarks.__all__
    assert "contextworld-cube-gripper-carry" in (
        ROOT / "pyproject.toml"
    ).read_text(encoding="utf-8")
    assert "cube" not in (
        ROOT / "configs/benchmark/contextworld_icl_suite_v1.yaml"
    ).read_text(encoding="utf-8").lower()


class _FakeCubeWorldModel:
    def __init__(self) -> None:
        self.action_encoder = SimpleNamespace(
            patch_embed=SimpleNamespace(in_channels=25)
        )

    def to(self, _device: str) -> "_FakeCubeWorldModel":
        return self

    def eval(self) -> "_FakeCubeWorldModel":
        return self

    def requires_grad_(self, _requires_grad: bool) -> "_FakeCubeWorldModel":
        return self


class _FakeTwoAxisWorldModel(_FakeCubeWorldModel):
    def __init__(self) -> None:
        self.action_encoder = SimpleNamespace(
            patch_embed=SimpleNamespace(in_channels=10)
        )


@pytest.mark.parametrize(
    "adapter_class",
    (
        StableWorldModelLeWMCubeGraspRuleAdapter,
        StableWorldModelPLDMCubeGraspRuleAdapter,
    ),
)
def test_cube_adapter_uses_five_raw_action_axes(
    tmp_path: Path,
    adapter_class: type[StableWorldModelLeWMCubeGraspRuleAdapter],
) -> None:
    checkpoint = tmp_path / "cube.pt"
    checkpoint.write_bytes(b"cube-adapter-fixture")
    adapter = adapter_class(
        model=_FakeCubeWorldModel(),
        checkpoint=checkpoint,
        stable_repo=tmp_path,
        stable_commit="fixture",
        action_standardizer=ColumnStandardizer(
            np.zeros((1, 5), dtype=np.float32),
            np.ones((1, 5), dtype=np.float32),
        ),
        device="cpu",
    )

    assert adapter.protocol.action_dim == 5
    assert adapter.protocol.action_block_raw_steps == 5
    blocks = np.arange(2 * 3 * 5 * 5, dtype=np.float32).reshape(
        2, 3, 5, 5
    )
    normalized = adapter._normalize_actions(blocks)
    assert normalized.shape == (2, 3, 25)
    np.testing.assert_array_equal(normalized, blocks.reshape(2, 3, 25))


def test_cube_adapter_rejects_two_axis_normalizer(tmp_path: Path) -> None:
    checkpoint = tmp_path / "cube.pt"
    checkpoint.write_bytes(b"cube-adapter-fixture")
    with pytest.raises(ValueError, match="normalizer dimension"):
        StableWorldModelLeWMCubeGraspRuleAdapter(
            model=_FakeCubeWorldModel(),
            checkpoint=checkpoint,
            stable_repo=tmp_path,
            stable_commit="fixture",
            action_standardizer=ColumnStandardizer(
                np.zeros((1, 2), dtype=np.float32),
                np.ones((1, 2), dtype=np.float32),
            ),
            device="cpu",
        )


def test_two_axis_adapter_protocol_remains_compatible(tmp_path: Path) -> None:
    checkpoint = tmp_path / "two-axis.pt"
    checkpoint.write_bytes(b"two-axis-adapter-fixture")
    adapter = StableWorldModelLeWMReacherArmMassAdapter(
        model=_FakeTwoAxisWorldModel(),
        checkpoint=checkpoint,
        stable_repo=tmp_path,
        stable_commit="fixture",
        action_standardizer=ColumnStandardizer(
            np.zeros((1, 2), dtype=np.float32),
            np.ones((1, 2), dtype=np.float32),
        ),
        device="cpu",
    )

    assert adapter.protocol.action_dim == 2
    assert adapter.protocol.action_block_raw_steps == 5


def test_cube_scorer_rejects_two_axis_protocol_before_data_access() -> None:
    incompatible = SimpleNamespace(
        protocol=AdapterProtocol(
            history_tokens=3,
            action_block_raw_steps=5,
            action_dim=2,
            future_action_blocks=1,
        )
    )
    with pytest.raises(ValueError, match="5x5 raw-action blocks"):
        _validate_cube_adapter_protocol(incompatible)


def test_cube_scorer_only_requires_public_geometry_fields() -> None:
    compatible = SimpleNamespace(
        protocol=AdapterProtocol(
            history_tokens=3,
            action_block_raw_steps=5,
            action_dim=5,
            future_action_blocks=1,
            native_target_encoder=False,
            decoder_required=True,
        )
    )

    _validate_cube_adapter_protocol(compatible)


def test_cube_grasp_rule_metric_rewards_matching_real_futures() -> None:
    cannot_hold = np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    can_hold = np.asarray([[0.0, 2.0], [1.0, 2.0]], dtype=np.float32)
    metrics, records = cube_grasp_rule_prediction_metrics(
        pair_ids=("a", "b"),
        predicted_cannot_hold=cannot_hold.copy(),
        predicted_can_hold=can_hold.copy(),
        target_cannot_hold=cannot_hold,
        target_can_hold=can_hold,
    )
    assert metrics["correct_future_rate"] == 1.0
    assert metrics["correct_history_rate"] == 1.0
    assert metrics["context_switch_rate"] == 1.0
    assert metrics["worst_rule_correct_future_rate"] == 1.0
    assert len(records) == 2


def test_cube_grasp_rule_current_frame_only_prediction_is_chance() -> None:
    cannot_hold = np.zeros((2, 1), dtype=np.float32)
    can_hold = np.full((2, 1), 2.0, dtype=np.float32)
    common = np.zeros((2, 1), dtype=np.float32)
    metrics, _ = cube_grasp_rule_prediction_metrics(
        pair_ids=("a", "b"),
        predicted_cannot_hold=common,
        predicted_can_hold=common,
        target_cannot_hold=cannot_hold,
        target_can_hold=can_hold,
    )
    assert metrics["correct_future_rate"] == 0.5
    assert metrics["context_switch_rate"] == 0.0


@pytest.mark.parametrize(
    ("pair_ids", "changed", "match"),
    [
        (("a", "a"), None, "non-empty and unique"),
        (("a", "b"), "shape", "latent shapes must match"),
        (("a", "b"), "nan", "latents must be finite"),
    ],
)
def test_cube_grasp_rule_metric_rejects_invalid_inputs(
    pair_ids: tuple[str, ...],
    changed: str | None,
    match: str,
) -> None:
    arrays = [np.zeros((2, 3), dtype=np.float32) for _ in range(4)]
    if changed == "shape":
        arrays[1] = np.zeros((2, 4), dtype=np.float32)
    elif changed == "nan":
        arrays[2][0, 0] = np.nan
    with pytest.raises(ValueError, match=match):
        cube_grasp_rule_prediction_metrics(
            pair_ids=pair_ids,
            predicted_cannot_hold=arrays[0],
            predicted_can_hold=arrays[1],
            target_cannot_hold=arrays[2],
            target_can_hold=arrays[3],
        )


def _write_method_scoring_fixture(tmp_path: Path) -> tuple[Path, list[Path]]:
    release_path = tmp_path / "release.yaml"
    release = {
        "schema_version": 1,
        "release_id": "contextworld_cube_gripper_carry_icl_history3_v1",
        "release_status": "data_ready_training_in_progress",
        "scope": {
            "history_tokens": 3,
            "public_test_included": True,
            "sealed_test_included": False,
            "grasp_modes": ["cannot_hold", "can_hold"],
        },
        "runtime": {
            "stable_worldmodel": {"repo": "../stable-worldmodel", "expected_ref": "abc"}
        },
        "identity": {
            "fixture": {"path": "fixture.py", "sha256": "fixture-sha"}
        },
        "data": {
            "protocol": "cube_gripper_carry_rule_history3_release_v1",
            "artifact_tree": {
                "root": "artifacts/synthesis/fixture",
                "files": 1,
                "bytes": 1,
                "sha256": "tree-sha",
            },
            "artifacts": {
                "manifest": {"path": "manifest.json", "sha256": "manifest-sha"}
            },
            "manifest_sha256": "manifest-sha",
            "pair_counts": {
                "train": 2048,
                "loader_validation": 256,
                "validation": 256,
            },
            "lance_tables": {
                "train": "train.lance",
                "loader_validation": "loader_validation.lance",
                "validation": "validation.lance",
            },
        },
        "training": {
            "upstream": {
                "original_h5": {"local_source": "/fixture.h5"},
                "original_lance": {"local_source": "/fixture.lance"},
            },
            "reference_matrix": {
                "status": "planned_not_executed",
                "training_seeds": [3072, 3073, 3074],
                "initial_checkpoints": {
                    "lewm": {"local_source": "/lewm.ckpt"},
                    "pldm": {"local_source": "/pldm.ckpt"},
                },
                "common": {
                    "optimizer_steps": 4,
                    "fixed_checkpoint_step": 4,
                    "loader_validation_monitor_steps": [2, 4],
                    "batch_size": 4,
                    "original_cube_samples_per_batch": 2,
                    "learning_rate": 5.0e-5,
                    "weight_decay": 1.0e-3,
                    "gradient_clip_norm": 1.0,
                },
            },
        },
        "evaluation": {
            "pair_count": 256,
            "lance_table": "validation.lance",
            "action_normalization": {
                "mean": [0.0, 0.0, 0.0, 0.0, 0.0],
                "std_population": [1.0, 1.0, 1.0, 1.0, 1.0],
            },
        },
        "scoring": {
            "hidden_future_prediction": {
                "gates": {
                    "correct_future_rate_minimum": 0.75,
                    "correct_history_rate_minimum": 0.75,
                    "context_switch_rate_minimum": 0.90,
                    "worst_rule_correct_future_rate_minimum": 0.70,
                    "target_latent_separation_required": True,
                    "response_gain_minimum": 0.50,
                    "normalized_response_error_strict_maximum": 1.0,
                }
            },
            "method_level": {"training_seeds_required": 3},
            "original_task_retention": {"status": "planned_not_executed"},
        },
    }
    release_path.write_text(yaml.safe_dump(release), encoding="utf-8")
    receipt = {
        "release_id": release["release_id"],
        "release_config_sha256": file_sha256(release_path),
        "data_manifest_sha256": "manifest-sha",
    }
    result_paths = []
    for seed in (3072, 3073, 3074):
        result = {
            "schema_version": 1,
            "benchmark": "cube_history3_gripper_carry_icl_v1",
            "submission_kind": "single_checkpoint",
            "status": "completed",
            "release": receipt,
            "model": {"training_seed": seed},
            "metrics": {
                "correct_future_rate": 1.0,
                "correct_history_rate": 1.0,
                "context_switch_rate": 1.0,
                "worst_rule_correct_future_rate": 1.0,
                "other_minus_correct_mse_margin_mean": 1.0,
                "joint_icl_pair_success_rate": 1.0,
                "latent_response": {},
            },
            "gate": {"passed": True},
        }
        path = tmp_path / f"seed{seed}.json"
        path.write_text(json.dumps(result), encoding="utf-8")
        result_paths.append(path)
    return release_path, result_paths


def test_cube_method_scoring_binds_all_results_to_release(tmp_path: Path) -> None:
    release_path, result_paths = _write_method_scoring_fixture(tmp_path)
    summary = score_cube_grasp_rule_icl_results(
        result_paths=result_paths,
        method_name="fixture",
        release_config=release_path,
    )
    assert summary["passed"]
    assert summary["training_seeds"] == [3072, 3073, 3074]

    payload = json.loads(result_paths[0].read_text(encoding="utf-8"))
    payload["release"]["data_manifest_sha256"] = "wrong"
    result_paths[0].write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="release identity mismatch"):
        score_cube_grasp_rule_icl_results(
            result_paths=result_paths,
            method_name="fixture",
            release_config=release_path,
        )


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("missing_ref", "expected_ref"),
        ("zero_action_std", "five finite axes"),
        ("duplicate_seed", "distinct integer training seeds"),
    ],
)
def test_cube_release_loader_rejects_invalid_nested_contract(
    tmp_path: Path,
    mutation: str,
    match: str,
) -> None:
    release_path, _ = _write_method_scoring_fixture(tmp_path)
    release = yaml.safe_load(release_path.read_text(encoding="utf-8"))
    if mutation == "missing_ref":
        del release["runtime"]["stable_worldmodel"]["expected_ref"]
    elif mutation == "zero_action_std":
        release["evaluation"]["action_normalization"]["std_population"][0] = 0.0
    else:
        release["training"]["reference_matrix"]["training_seeds"][-1] = 3073
    release_path.write_text(yaml.safe_dump(release), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        load_cube_grasp_rule_icl_release(release_path)


def test_cube_release_audit_reports_partial_data_without_crashing(
    tmp_path: Path,
) -> None:
    release_path, _ = _write_method_scoring_fixture(tmp_path)
    release = yaml.safe_load(release_path.read_text(encoding="utf-8"))
    release["release_status"] = "public_test_release_candidate"
    release["training"]["reference_matrix"]["status"] = "failed_development"
    release["scoring"]["original_task_retention"]["status"] = (
        "not_run_after_failed_development"
    )
    release_path.write_text(yaml.safe_dump(release), encoding="utf-8")

    audit = audit_cube_grasp_rule_icl_release(
        release_config=release_path,
        repo_root=tmp_path,
    )
    assert not audit["passed"]
    assert audit["status"] == "failed"
    assert audit["build_report"]["error"].startswith("FileNotFoundError")
    assert not audit["public_test"]["passed"]
    assert all(audit["release_checks"].values())


@pytest.mark.skipif(not CUBE_SOURCE.is_file(), reason="Cube source is unavailable")
def test_cube_grasp_rule_pair_has_a_strict_shared_query() -> None:
    with h5py.File(CUBE_SOURCE, "r", swmr=True) as handle:
        row = 30
        candidate = CubeGraspRuleCandidate(
            candidate_id="cube-grasp-unit-row30",
            split="unit",
            catalog_index=0,
            source_row=row,
            source_episode=int(handle["ep_idx"][row]),
            source_step=int(handle["step_idx"][row]),
            simulator_seed=123,
            task_id=1,
            qpos=tuple(float(value) for value in handle["qpos"][row]),
            control=tuple(float(value) for value in handle["control"][row]),
            cube_color=(0.5, 0.4, 0.3),
            target_position=(0.4, 0.0, 0.02),
        )

    simulator = CubeGraspRuleSimulator()
    try:
        pair = simulator.build_pair(candidate)
    finally:
        simulator.close()

    assert pair is not None
    assert pair["audit"]["passed"]
    assert pair["audit"]["maximum_initial_simulator_state_gap"] == 0.0
    assert (
        pair["audit"]["maximum_query_simulator_state_gap"]
        <= QUERY_STATE_TOLERANCE
    )
    assert pair["audit"]["state_installations_after_x0"] == 0
    assert pair["audit"]["checks"]["no_state_installation_after_x0"]
    assert (
        pair["audit"]["maximum_prequery_object_state_residual"]
        <= QUERY_STATE_TOLERANCE
    )
    assert np.array_equal(
        pair["cannot_hold"]["pixels"][2],
        pair["can_hold"]["pixels"][2],
    )
    assert np.array_equal(
        pair["cannot_hold"]["action_blocks"],
        pair["can_hold"]["action_blocks"],
    )
    assert not np.array_equal(
        pair["cannot_hold"]["pixels"][1],
        pair["can_hold"]["pixels"][1],
    )
    assert not np.array_equal(
        pair["cannot_hold"]["pixels"][3],
        pair["can_hold"]["pixels"][3],
    )
