from __future__ import annotations

from collections import Counter
from dataclasses import replace
import hashlib
from pathlib import Path

import h5py
import numpy as np
import pytest

from contextworld.evaluation.cube_grasp_rule_h3_v3 import (
    GRASP_MODES,
    QUERY_STATE_TOLERANCE,
    CubeGraspRuleCandidate,
    CubeGraspRuleV3Candidate,
    CubeGraspRuleV3Simulator,
    V3_ACTION_ANCHORS,
    V3_ANCHOR_GRIPPER_ACTIONS,
    V3_ANCHOR_PROFILES,
    V3_PERTURBATION_COEFFICIENT_LIMIT,
    V3_PROFILE_SPLIT_SEEDS,
    V3_PROTOCOL,
    action_blocks,
    action_profile_content_sha256,
    make_v3_action_profile,
    make_v3_candidate,
    validate_v3_action_profile,
)


CUBE_SOURCE = Path(
    "/opt/huawei/explorer-env/dataset/ag_data/data/world_model/"
    "quentinll/lewm-cube/ogbench/cube_single_expert.h5"
)
MOMENT_WEIGHTS = np.asarray([4.0, 3.0, 2.0, 1.0, 0.0])


class _FakeV3Simulator(CubeGraspRuleV3Simulator):
    def __init__(
        self,
        *,
        resolution: int = 8,
        mismatch_mode: str | None = None,
    ) -> None:
        self.resolution = resolution
        self.env = object()
        self.mismatch_mode = mismatch_mode
        self.run_calls: list[str] = []
        self.closed = False

    def close(self) -> None:
        self.closed = True

    def _run_mode(
        self,
        candidate: CubeGraspRuleV3Candidate,
        *,
        mode: str,
        blocks: np.ndarray,
    ) -> dict[str, object]:
        del candidate
        self.run_calls.append(mode)
        physical_state = np.zeros((4, 7), dtype=np.float32)
        simulator_state = np.zeros((4, 12), dtype=np.float64)
        pixels = np.zeros((4, 8, 8, 3), dtype=np.uint8)
        if mode == "can_hold":
            physical_state[[1, 3], 4] = 0.009
            simulator_state[[1, 3], 1] = 0.009
            pixels[[1, 3]] = 1
        if mode == self.mismatch_mode:
            pixels[0, 0, 0, 0] = 7
        return {
            "pixels": pixels,
            "physical_state": physical_state,
            "simulator_state": simulator_state,
            "action_blocks": np.asarray(blocks, dtype=np.float32).copy(),
            "hidden_value": 1.0 if mode == "can_hold" else 0.0,
            "prequery_residual": 0.0,
            "state_installations_after_x0": 0,
            "external_force_updates_after_x0": 20,
        }


def _base_candidate(*, split: str = "train", catalog_index: int = 0):
    return CubeGraspRuleCandidate(
        candidate_id=f"cube-v3-unit-{split}-{catalog_index}",
        split=split,
        catalog_index=catalog_index,
        source_row=10 + catalog_index,
        source_episode=catalog_index,
        source_step=10,
        simulator_seed=123 + catalog_index,
        task_id=1,
        qpos=(0.0,),
        control=(0.0,),
        cube_color=(0.5, 0.4, 0.3),
        target_position=(0.4, 0.0, 0.02),
    )


def test_v3_declares_four_fixed_anchor_families() -> None:
    assert V3_ACTION_ANCHORS == (
        "endpoint4",
        "plateau",
        "ramp4",
        "front_hold",
    )
    assert V3_ANCHOR_PROFILES == {
        "endpoint4": (1.0 / 3.0, 0.0, 0.0, -1.0 / 3.0, 0.0),
        "plateau": (0.25, 0.25, -0.25, -0.25, 0.0),
        "ramp4": (0.3, 0.1, -0.1, -0.3, 0.0),
        "front_hold": (0.2, 0.2, 0.0, -0.4, 0.0),
    }
    assert V3_ANCHOR_GRIPPER_ACTIONS == {
        "endpoint4": 0.4,
        "plateau": 0.4,
        "ramp4": 0.5,
        "front_hold": 0.5,
    }


@pytest.mark.parametrize("split", tuple(V3_PROFILE_SPLIT_SEEDS))
def test_v3_profiles_are_exactly_balanced_and_constrained(split: str) -> None:
    profiles = [
        make_v3_action_profile(split=split, catalog_index=index)
        for index in range(64)
    ]
    assert Counter(value.action_anchor_id for value in profiles) == {
        anchor_id: 16 for anchor_id in V3_ACTION_ANCHORS
    }
    for profile in profiles:
        probe = np.asarray(profile.probe_profile, dtype=np.float64)
        assert probe.sum() == 0.0
        assert probe[-1] == 0.0
        assert np.dot(MOMENT_WEIGHTS, probe) == 1.0
        assert validate_v3_action_profile(profile)["passed"]


def test_v3_profile_generation_is_deterministic_and_content_addressed() -> None:
    first = make_v3_action_profile(split="train", catalog_index=17)
    second = make_v3_action_profile(split="train", catalog_index=17)
    assert first == second

    blocks = action_blocks(first)
    assert blocks.shape == (4, 5, 5)
    assert blocks.dtype == np.float32
    expected = hashlib.sha256(
        np.ascontiguousarray(blocks, dtype=np.float32).tobytes()
    ).hexdigest()
    assert first.action_profile_id == expected
    assert action_profile_content_sha256(blocks) == expected
    assert len(first.action_profile_id) == 64
    assert first.split not in first.action_profile_id


def test_v3_training_and_development_content_ids_are_disjoint() -> None:
    training = {
        make_v3_action_profile(
            split="train", catalog_index=index
        ).action_profile_id
        for index in range(2048)
    }
    development = {
        make_v3_action_profile(
            split="loader_validation", catalog_index=index
        ).action_profile_id
        for index in range(256)
    }
    assert len(training) == 2048
    assert len(development) == 256
    assert training.isdisjoint(development)


def test_v3_perturbations_are_small_and_use_both_coefficients() -> None:
    nonzero_first = False
    nonzero_second = False
    for index in range(32):
        profile = make_v3_action_profile(split="train", catalog_index=index)
        anchor = np.asarray(
            V3_ANCHOR_PROFILES[profile.action_anchor_id], dtype=np.float64
        )
        actual = np.asarray(profile.probe_profile, dtype=np.float64)
        coefficients = np.asarray(profile.perturbation_coefficients)
        assert np.max(np.abs(coefficients)) <= (
            V3_PERTURBATION_COEFFICIENT_LIMIT
        )
        assert np.linalg.norm(actual - anchor) < 0.04
        nonzero_first |= coefficients[0] != 0.0
        nonzero_second |= coefficients[1] != 0.0
    assert nonzero_first and nonzero_second


def test_v3_action_blocks_encode_probe_recovery_query_and_format_tail() -> None:
    profile = make_v3_action_profile(split="train", catalog_index=2)
    blocks = action_blocks(profile)
    probe = np.asarray(profile.probe_profile, dtype=np.float32)

    np.testing.assert_array_equal(blocks[0, :, 2], probe)
    np.testing.assert_array_equal(blocks[1, :, 2], -probe)
    np.testing.assert_array_equal(blocks[2, :, 2], probe)
    np.testing.assert_array_equal(blocks[:3, :, 4], profile.gripper_action)
    np.testing.assert_array_equal(blocks[:3, :, (0, 1, 3)], 0.0)
    np.testing.assert_array_equal(blocks[3], 0.0)


@pytest.mark.parametrize(
    ("split", "catalog_index", "exception"),
    [
        ("development", 0, ValueError),
        ("validation", 0, ValueError),
        ("train", -1, ValueError),
        ("train", True, TypeError),
    ],
)
def test_v3_profile_generation_rejects_invalid_coordinates(
    split: str, catalog_index: int, exception: type[Exception]
) -> None:
    with pytest.raises(exception):
        make_v3_action_profile(split=split, catalog_index=catalog_index)


def test_v3_profile_audit_rejects_metadata_or_content_tampering() -> None:
    profile = make_v3_action_profile(split="train", catalog_index=0)
    wrong_hash = replace(profile, action_profile_id="0" * 64)
    hash_audit = validate_v3_action_profile(wrong_hash)
    assert not hash_audit["passed"]
    assert not hash_audit["checks"]["action_profile_id_is_content_hash"]
    with pytest.raises(ValueError, match="action_profile_id_is_content_hash"):
        action_blocks(wrong_hash)

    changed_probe = list(profile.probe_profile)
    changed_probe[-1] = 0.01
    wrong_probe = replace(profile, probe_profile=tuple(changed_probe))
    probe_audit = validate_v3_action_profile(wrong_probe)
    assert not probe_audit["passed"]
    assert not probe_audit["checks"]["probe_final_z_exact_zero"]


def test_v3_candidate_factory_binds_split_index_and_profile() -> None:
    base = _base_candidate(split="loader_validation", catalog_index=9)
    candidate = make_v3_candidate(base)
    assert isinstance(candidate, CubeGraspRuleV3Candidate)
    assert candidate.action_profile.split == base.split
    assert candidate.action_profile.catalog_index == base.catalog_index
    assert make_v3_candidate(candidate) is candidate

    wrong = make_v3_action_profile(split="train", catalog_index=9)
    with pytest.raises(ValueError, match="splits differ"):
        CubeGraspRuleV3Candidate(
            **{field: getattr(base, field) for field in base.__dataclass_fields__},
            action_profile=wrong,
        )


def test_v3_simulator_rejects_unprofiled_v2_candidate_without_running() -> None:
    simulator = object.__new__(CubeGraspRuleV3Simulator)
    with pytest.raises(TypeError, match="CubeGraspRuleV3Candidate"):
        simulator.build_pair(_base_candidate())


def test_v3_protocol_is_the_frozen_development_only_identifier() -> None:
    assert V3_PROTOCOL == "cube_gripper_carry_rule_history3_development_v3"


def test_v3_build_pair_requires_exact_fresh_simulator_replay() -> None:
    candidate = make_v3_candidate(_base_candidate())
    primary = _FakeV3Simulator()
    replay = _FakeV3Simulator()
    pair = primary.build_pair(candidate, replay_simulator=replay)

    assert pair is not None
    assert pair["audit"]["passed"]
    assert primary.run_calls == list(GRASP_MODES)
    assert replay.run_calls == list(GRASP_MODES)
    fresh = pair["audit"]["v3"]["fresh_simulator_replay"]
    assert fresh["passed"]
    assert fresh["provided_reusable_instance"]
    assert fresh["independent_simulator_instance"]
    assert fresh["fresh_candidate_reset_before_each_mode"]
    assert not fresh["replay_payload_retained_as_training_target"]
    assert fresh["maximum_physical_state_gap"] == 0.0
    assert fresh["maximum_simulator_state_gap"] == 0.0
    assert fresh["total_changed_rgb_values"] == 0
    assert fresh["total_changed_pixels"] == 0
    for mode in GRASP_MODES:
        mode_audit = fresh["modes"][mode]
        assert mode_audit["passed"]
        assert all(mode_audit["checks"].values())
        assert mode_audit["changed_rgb_values"] == 0
        assert mode_audit["changed_pixels"] == 0
        assert (
            mode_audit["continuous_action_profile_id"]
            == mode_audit["fresh_replay_action_profile_id"]
            == candidate.action_profile.action_profile_id
        )
        assert (
            mode_audit["hashes"]["continuous"]
            == mode_audit["hashes"]["fresh_replay"]
        )


def test_v3_fresh_replay_audit_detects_pixel_nondeterminism() -> None:
    candidate = make_v3_candidate(_base_candidate())
    primary = _FakeV3Simulator()
    blocks = action_blocks(candidate.action_profile)
    continuous = {
        mode: primary._run_mode(candidate, mode=mode, blocks=blocks)
        for mode in GRASP_MODES
    }
    for episode in continuous.values():
        episode["action_anchor_id"] = candidate.action_profile.action_anchor_id
        episode["action_profile_id"] = candidate.action_profile.action_profile_id
    replay = _FakeV3Simulator(mismatch_mode="can_hold")

    audit = primary.audit_fresh_simulator_replay(
        candidate,
        continuous,
        replay_simulator=replay,
    )
    assert not audit["passed"]
    assert audit["total_changed_rgb_values"] == 1
    assert audit["total_changed_pixels"] == 1
    assert audit["modes"]["cannot_hold"]["passed"]
    changed = audit["modes"]["can_hold"]
    assert not changed["passed"]
    assert not changed["checks"]["pixels_bitwise_equal"]
    assert not changed["checks"]["pixels_hash_equal"]
    assert changed["changed_rgb_values"] == 1
    assert changed["changed_pixels"] == 1


def test_v3_fresh_replay_requires_distinct_unshared_simulator() -> None:
    candidate = make_v3_candidate(_base_candidate())
    primary = _FakeV3Simulator()
    blocks = action_blocks(candidate.action_profile)
    continuous = {
        mode: primary._run_mode(candidate, mode=mode, blocks=blocks)
        for mode in GRASP_MODES
    }
    with pytest.raises(ValueError, match="distinct instance"):
        primary.audit_fresh_simulator_replay(
            candidate,
            continuous,
            replay_simulator=primary,
        )

    shared = _FakeV3Simulator()
    shared.env = primary.env
    with pytest.raises(ValueError, match="must not share"):
        primary.audit_fresh_simulator_replay(
            candidate,
            continuous,
            replay_simulator=shared,
        )


@pytest.mark.skipif(not CUBE_SOURCE.is_file(), reason="Cube source is unavailable")
def test_v3_mujoco_pair_has_shared_query_and_bitwise_equal_actions() -> None:
    with h5py.File(CUBE_SOURCE, "r", swmr=True) as handle:
        row = 30
        base = CubeGraspRuleCandidate(
            candidate_id="cube-v3-smoke-row30",
            split="train",
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
    candidate = make_v3_candidate(base)

    simulator = CubeGraspRuleV3Simulator()
    replay_simulator = CubeGraspRuleV3Simulator()
    try:
        pair = simulator.build_pair(
            candidate,
            replay_simulator=replay_simulator,
        )
    finally:
        simulator.close()
        replay_simulator.close()

    assert pair is not None
    assert pair["audit"]["passed"]
    assert pair["audit"]["v3"]["action_anchor_id"] == "endpoint4"
    assert (
        pair["audit"]["v3"]["action_profile_id"]
        == candidate.action_profile.action_profile_id
    )
    assert pair["audit"]["v3"]["profile_constraints"]["passed"]
    replay = pair["audit"]["v3"]["fresh_simulator_replay"]
    assert replay["passed"]
    assert replay["provided_reusable_instance"]
    assert replay["maximum_physical_state_gap"] == 0.0
    assert replay["maximum_simulator_state_gap"] == 0.0
    assert replay["total_changed_rgb_values"] == 0
    assert replay["total_changed_pixels"] == 0
    assert (
        pair["audit"]["maximum_query_simulator_state_gap"]
        <= QUERY_STATE_TOLERANCE
    )
    assert (
        pair["audit"]["maximum_prequery_object_state_residual"]
        <= QUERY_STATE_TOLERANCE
    )
    np.testing.assert_array_equal(
        pair["cannot_hold"]["action_blocks"],
        pair["can_hold"]["action_blocks"],
    )
    assert (
        action_profile_content_sha256(
            pair["cannot_hold"]["action_blocks"]
        )
        == candidate.action_profile.action_profile_id
    )
    assert (
        pair["cannot_hold"]["action_profile_id"]
        == pair["can_hold"]["action_profile_id"]
    )
