from __future__ import annotations

from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

import h5py
import numpy as np
import pytest

from contextworld.evaluation.cube_grasp_rule_h3 import (
    VERTICAL_FORCE_COUPLING_N as V3_VERTICAL_FORCE_COUPLING_N,
    CubeGraspRuleCandidate,
)
from contextworld.evaluation.cube_grasp_rule_h3_v3 import (
    CubeGraspRuleV3Simulator,
    make_v3_action_profile,
)
from contextworld.evaluation.cube_grasp_rule_h3_v4 import (
    GRASP_MODES,
    QUERY_STATE_TOLERANCE,
    CubeGraspRuleV4Candidate,
    CubeGraspRuleV4Profile,
    CubeGraspRuleV4Simulator,
    V4_ACTION_ANCHORS,
    V4_ANCHOR_GRIPPER_ACTIONS,
    V4_ANCHOR_PROFILES,
    V4_FORMAL_CATALOG_INDEX_OFFSET,
    V4R1_FORMAL_CATALOG_INDEX_OFFSET,
    V4_PERTURBATION_COEFFICIENT_LIMIT,
    V4_PROFILE_SPLIT_SEEDS,
    V4_PROTOCOL,
    V4_VERTICAL_FORCE_COUPLING_N,
    action_blocks,
    action_profile_content_sha256,
    make_v4_action_profile,
    make_v4_candidate,
    validate_v4_action_profile,
)


CUBE_SOURCE = Path(
    "/opt/huawei/explorer-env/dataset/ag_data/data/world_model/"
    "quentinll/lewm-cube/ogbench/cube_single_expert.h5"
)
MOMENT_WEIGHTS = np.asarray([4.0, 3.0, 2.0, 1.0, 0.0])


class _FakeV4Simulator(CubeGraspRuleV4Simulator):
    def __init__(self, *, resolution: int = 8) -> None:
        self.resolution = resolution
        self.env = object()
        self.run_calls: list[str] = []
        self.closed = False

    def close(self) -> None:
        self.closed = True

    def _run_mode(
        self,
        candidate: CubeGraspRuleV4Candidate,
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
            physical_state[[1, 3], 4] = 0.013
            simulator_state[[1, 3], 1] = 0.013
            pixels[[1, 3]] = 1
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
        candidate_id=f"cube-v4-unit-{split}-{catalog_index}",
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


def _string_receipt_fragments(value: Any) -> Iterator[str]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key)
            yield from _string_receipt_fragments(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _string_receipt_fragments(child)
    elif isinstance(value, str):
        yield value


def test_v4_freezes_protocol_seeds_and_unchanged_anchor_templates() -> None:
    assert V4_PROTOCOL == "cube_gripper_carry_rule_history3_development_v4"
    assert V4_PROFILE_SPLIT_SEEDS == {
        "train": 2026081201,
        "loader_validation": 2026081202,
    }
    assert V4_FORMAL_CATALOG_INDEX_OFFSET == 1_000_000
    assert V4_FORMAL_CATALOG_INDEX_OFFSET > 0
    assert V4_FORMAL_CATALOG_INDEX_OFFSET % len(V4_ACTION_ANCHORS) == 0
    assert V4R1_FORMAL_CATALOG_INDEX_OFFSET == 2_000_000
    assert V4R1_FORMAL_CATALOG_INDEX_OFFSET > V4_FORMAL_CATALOG_INDEX_OFFSET
    assert V4R1_FORMAL_CATALOG_INDEX_OFFSET % len(V4_ACTION_ANCHORS) == 0
    assert V4_ACTION_ANCHORS == (
        "endpoint4",
        "plateau",
        "ramp4",
        "front_hold",
    )
    assert V4_ANCHOR_PROFILES == {
        "endpoint4": (1.0 / 3.0, 0.0, 0.0, -1.0 / 3.0, 0.0),
        "plateau": (0.25, 0.25, -0.25, -0.25, 0.0),
        "ramp4": (0.3, 0.1, -0.1, -0.3, 0.0),
        "front_hold": (0.2, 0.2, 0.0, -0.4, 0.0),
    }
    assert V4_ANCHOR_GRIPPER_ACTIONS == {
        "endpoint4": 0.4,
        "plateau": 0.4,
        "ramp4": 0.5,
        "front_hold": 0.5,
    }


@pytest.mark.parametrize("split", tuple(V4_PROFILE_SPLIT_SEEDS))
def test_v4_profiles_are_balanced_and_exactly_constrained(split: str) -> None:
    profiles = [
        make_v4_action_profile(split=split, catalog_index=index)
        for index in range(64)
    ]
    assert Counter(value.action_anchor_id for value in profiles) == {
        anchor_id: 16 for anchor_id in V4_ACTION_ANCHORS
    }
    for profile in profiles:
        probe = np.asarray(profile.probe_profile, dtype=np.float64)
        coefficients = np.asarray(profile.perturbation_coefficients)
        assert probe.sum() == 0.0
        assert probe[-1] == 0.0
        assert np.dot(MOMENT_WEIGHTS, probe) == 1.0
        assert np.max(np.abs(coefficients)) <= (
            V4_PERTURBATION_COEFFICIENT_LIMIT
        )
        assert validate_v4_action_profile(profile)["passed"]


def test_v4_profiles_use_exact_p_negative_p_p_blocks_and_content_ids() -> None:
    profile = make_v4_action_profile(split="train", catalog_index=17)
    assert profile == make_v4_action_profile(split="train", catalog_index=17)
    blocks = action_blocks(profile)
    probe = np.asarray(profile.probe_profile, dtype=np.float32)

    assert blocks.shape == (4, 5, 5)
    assert blocks.dtype == np.float32
    np.testing.assert_array_equal(blocks[0, :, 2], probe)
    np.testing.assert_array_equal(blocks[1, :, 2], -probe)
    np.testing.assert_array_equal(blocks[2, :, 2], probe)
    np.testing.assert_array_equal(blocks[:3, :, 4], profile.gripper_action)
    np.testing.assert_array_equal(blocks[:3, :, (0, 1, 3)], 0.0)
    np.testing.assert_array_equal(blocks[3], 0.0)
    assert profile.action_profile_id == action_profile_content_sha256(blocks)


def test_v4_formal_profile_sets_are_fully_disjoint_from_each_other_and_v3() -> None:
    v4_training = {
        make_v4_action_profile(
            split="train",
            catalog_index=V4_FORMAL_CATALOG_INDEX_OFFSET + local_index,
        ).action_profile_id
        for local_index in range(2048)
    }
    v4_development = {
        make_v4_action_profile(
            split="loader_validation",
            catalog_index=V4_FORMAL_CATALOG_INDEX_OFFSET + local_index,
        ).action_profile_id
        for local_index in range(256)
    }
    v3_training = {
        make_v3_action_profile(
            split="train", catalog_index=index
        ).action_profile_id
        for index in range(2048)
    }
    v3_development = {
        make_v3_action_profile(
            split="loader_validation", catalog_index=index
        ).action_profile_id
        for index in range(256)
    }

    assert len(v4_training) == 2048
    assert len(v4_development) == 256
    assert v4_training.isdisjoint(v4_development)
    assert (v4_training | v4_development).isdisjoint(
        v3_training | v3_development
    )

    preformal_v4 = {
        make_v4_action_profile(
            split="train", catalog_index=preformal_index
        ).action_profile_id
        for preformal_index in (0, 1)
    }
    assert len(preformal_v4) == 2
    assert (v4_training | v4_development).isdisjoint(preformal_v4)


def test_v4_profile_audit_rejects_metadata_or_content_tampering() -> None:
    profile = make_v4_action_profile(split="train", catalog_index=0)
    wrong_hash = replace(profile, action_profile_id="0" * 64)
    hash_audit = validate_v4_action_profile(wrong_hash)
    assert not hash_audit["passed"]
    assert not hash_audit["checks"]["action_profile_id_is_content_hash"]
    with pytest.raises(ValueError, match="action_profile_id_is_content_hash"):
        action_blocks(wrong_hash)

    changed_probe = list(profile.probe_profile)
    changed_probe[-1] = 0.01
    wrong_probe = replace(profile, probe_profile=tuple(changed_probe))
    probe_audit = validate_v4_action_profile(wrong_probe)
    assert not probe_audit["passed"]
    assert not probe_audit["checks"]["probe_final_z_exact_zero"]


def test_v4_candidate_factory_uses_only_v4_types() -> None:
    base = _base_candidate(split="loader_validation", catalog_index=9)
    candidate = make_v4_candidate(base)
    assert isinstance(candidate, CubeGraspRuleV4Candidate)
    assert isinstance(candidate.action_profile, CubeGraspRuleV4Profile)
    assert candidate.action_profile.split == base.split
    assert candidate.action_profile.catalog_index == base.catalog_index
    assert make_v4_candidate(candidate) is candidate

    wrong = make_v4_action_profile(split="train", catalog_index=9)
    with pytest.raises(ValueError, match="splits differ"):
        CubeGraspRuleV4Candidate(
            **{field: getattr(base, field) for field in base.__dataclass_fields__},
            action_profile=wrong,
        )


def test_v4_force_override_is_0_40_n_and_leaves_v3_at_0_30_n() -> None:
    assert V4_VERTICAL_FORCE_COUPLING_N == 0.40
    assert V3_VERTICAL_FORCE_COUPLING_N == 0.30

    qfrc_applied = np.full(8, 99.0, dtype=np.float64)
    base = SimpleNamespace(
        _data=SimpleNamespace(qfrc_applied=qfrc_applied),
        _model=SimpleNamespace(
            opt=SimpleNamespace(gravity=np.asarray([0.0, 0.0, -9.81])),
            body_mass=np.asarray([2.0]),
        ),
    )
    v4 = object.__new__(CubeGraspRuleV4Simulator)
    v4.base = base
    v4.object_body_id = 0
    v4.object_dof_address = 1
    v4._apply_transition_force(mode="can_hold", action_z=0.5)
    expected_gravity_compensation = 2.0 * 9.81
    assert qfrc_applied[3] == pytest.approx(
        expected_gravity_compensation + 0.40 * 0.5
    )
    np.testing.assert_array_equal(qfrc_applied[np.arange(8) != 3], 0.0)

    v3 = object.__new__(CubeGraspRuleV3Simulator)
    v3.base = base
    v3.object_body_id = 0
    v3.object_dof_address = 1
    v3._apply_transition_force(mode="can_hold", action_z=0.5)
    assert qfrc_applied[3] == pytest.approx(
        expected_gravity_compensation + 0.30 * 0.5
    )


def test_v4_fake_pair_has_v4_receipts_and_exact_fresh_replay() -> None:
    candidate = make_v4_candidate(_base_candidate())
    primary = _FakeV4Simulator()
    replay = _FakeV4Simulator()
    pair = primary.build_pair(candidate, replay_simulator=replay)

    assert pair is not None
    assert pair["audit"]["passed"]
    assert primary.run_calls == list(GRASP_MODES)
    assert replay.run_calls == list(GRASP_MODES)
    assert "v4" in pair["audit"]
    assert "v3" not in pair["audit"]
    receipt = pair["audit"]["v4"]
    assert receipt["protocol"] == V4_PROTOCOL
    assert receipt["vertical_force_coupling_n"] == 0.40
    assert receipt["action_profile_id"] == (
        candidate.action_profile.action_profile_id
    )
    fresh = receipt["fresh_simulator_replay"]
    assert fresh["passed"]
    assert fresh["provided_reusable_instance"]
    assert fresh["independent_simulator_instance"]
    assert fresh["maximum_physical_state_gap"] == 0.0
    assert fresh["maximum_simulator_state_gap"] == 0.0
    assert fresh["total_changed_rgb_values"] == 0
    assert fresh["total_changed_pixels"] == 0
    assert all(
        "v3" not in fragment.lower()
        for fragment in _string_receipt_fragments(pair)
    )


def test_v4_simulator_rejects_unprofiled_candidate_without_running() -> None:
    simulator = object.__new__(CubeGraspRuleV4Simulator)
    with pytest.raises(TypeError, match="CubeGraspRuleV4Candidate"):
        simulator.build_pair(_base_candidate())


@pytest.mark.skipif(not CUBE_SOURCE.is_file(), reason="Cube source is unavailable")
def test_v4_mujoco_pairs_have_12_60mm_signal_and_exact_shared_query() -> None:
    rows = (30, 31)
    with h5py.File(CUBE_SOURCE, "r", swmr=True) as handle:
        bases = [
            CubeGraspRuleCandidate(
                candidate_id=f"cube-v4-smoke-row{row}",
                split="train",
                catalog_index=index,
                source_row=row,
                source_episode=int(handle["ep_idx"][row]),
                source_step=int(handle["step_idx"][row]),
                simulator_seed=123 + index,
                task_id=1,
                qpos=tuple(float(value) for value in handle["qpos"][row]),
                control=tuple(float(value) for value in handle["control"][row]),
                cube_color=(0.5, 0.4, 0.3),
                target_position=(0.4, 0.0, 0.02),
            )
            for index, row in enumerate(rows)
        ]

    # These are frozen preformal evidence indices; formal catalogs use the
    # disjoint V4_FORMAL_CATALOG_INDEX_OFFSET namespace.
    assert [base.catalog_index for base in bases] == [0, 1]
    assert all(
        base.catalog_index < V4_FORMAL_CATALOG_INDEX_OFFSET for base in bases
    )

    simulator = CubeGraspRuleV4Simulator()
    replay_simulator = CubeGraspRuleV4Simulator()
    try:
        pairs = [
            simulator.build_pair(
                make_v4_candidate(base),
                replay_simulator=replay_simulator,
            )
            for base in bases
        ]
    finally:
        simulator.close()
        replay_simulator.close()

    for pair in pairs:
        assert pair is not None
        audit = pair["audit"]
        assert audit["passed"]
        assert audit["history_cube_height_gap_m"] == pytest.approx(
            0.01260081, abs=2e-6
        )
        assert audit["future_cube_height_gap_m"] == pytest.approx(
            0.01260081, abs=2e-6
        )
        assert audit["maximum_query_simulator_state_gap"] <= (
            QUERY_STATE_TOLERANCE
        )
        assert audit["maximum_prequery_object_state_residual"] <= (
            QUERY_STATE_TOLERANCE
        )
        assert audit["checks"]["query_pixels_identical"]
        assert audit["checks"]["no_state_installation_after_x0"]
        assert audit["v4"]["protocol"] == V4_PROTOCOL
        assert audit["v4"]["fresh_simulator_replay"]["passed"]
        np.testing.assert_array_equal(
            pair["cannot_hold"]["action_blocks"],
            pair["can_hold"]["action_blocks"],
        )
