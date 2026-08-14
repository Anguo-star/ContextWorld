from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from contextworld.evaluation.cube_grasp_rule_h3_v3 import (
    GRASP_MODES,
    CubeGraspRuleCandidate,
    action_blocks as v3_action_blocks,
    make_v3_candidate,
    make_v3_action_profile,
)
import scripts.build_cube_grasp_rule_h3_v3_data as builder


def test_v3_builder_is_development_only_by_construction() -> None:
    assert builder.PROTOCOL == "cube_gripper_carry_rule_history3_development_v3"
    assert builder.ACTIVE_SPLITS == ("train", "loader_validation")
    assert builder.DEFAULT_PAIR_COUNTS == {
        "train": 2048,
        "loader_validation": 256,
    }
    assert builder.DEFAULT_OUTPUT_LOGICAL == Path(
        "artifacts/synthesis/cube_gripper_carry_rule_h3_development_v3"
    )
    assert builder.DEFAULT_FREEZE_RECEIPT_LOGICAL == Path(
        "artifacts/evaluation/history3/"
        "cube_gripper_carry_h3_development_v3/"
        "development_prereg_freeze_receipt_v2.json"
    )
    assert builder.DEFAULT_FREEZE_RECEIPT.name == (
        "development_prereg_freeze_receipt_v2.json"
    )
    assert not hasattr(builder, "DEFAULT_SOURCE")
    assert builder.SOURCE_SYMBOL == "upstream_cube_single_expert_h5"
    assert builder.EVIDENCE_SCOPE == (
        "every accepted pair in Training and Development"
    )
    assert builder.PROFILE_SPLIT_POLICY == (
        "shared_families_disjoint_profiles"
    )


def test_v3_schema_marks_exact_action_profile_and_anchor() -> None:
    assert "action_anchor_id" in builder.SCHEMA.names
    assert "action_profile_id" in builder.SCHEMA.names
    assert "scene_template_content_hash" in builder.SCHEMA.names
    assert "pair_content_hash" in builder.SCHEMA.names
    assert "source_step" in builder.SCHEMA.names
    assert "action_anchor_id" in builder.PRIVILEGED_COLUMNS
    assert "action_profile_id" in builder.PRIVILEGED_COLUMNS
    assert "scene_template_content_hash" in builder.PRIVILEGED_COLUMNS
    assert "pair_content_hash" in builder.PRIVILEGED_COLUMNS
    assert "episode_idx" in builder.PRIVILEGED_COLUMNS
    assert "model_step_idx" in builder.PRIVILEGED_COLUMNS
    assert "source_step" in builder.PRIVILEGED_COLUMNS
    assert builder.SCHEMA.field("action_anchor_id").type == builder.pa.string()
    assert builder.SCHEMA.field("action_profile_id").type == builder.pa.string()
    assert (
        builder.SCHEMA.field("scene_template_content_hash").type
        == builder.pa.string()
    )
    assert builder.SCHEMA.field("pair_content_hash").type == builder.pa.string()


@pytest.mark.parametrize(
    "counts",
    (
        {"train": 0, "loader_validation": 256},
        {"train": -4, "loader_validation": 256},
        {"train": 2046, "loader_validation": 256},
        {"train": 2048, "loader_validation": 2},
        {"train": True, "loader_validation": 256},
        {"train": 2048},
        {"train": 2048, "loader_validation": 256, "validation": 4},
        {"train": 2048, "loader_validation": 256, "public_test": 4},
    ),
)
def test_v3_pair_count_contract_rejects_invalid_or_public_counts(
    counts: dict[str, int],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        builder._validate_pair_counts(counts)


@pytest.mark.parametrize(
    "option",
    ("--public-test-pairs", "--validation-pairs", "--test-pairs"),
)
def test_v3_cli_explicitly_refuses_public_pair_options(option: str) -> None:
    with pytest.raises(ValueError, match="explicitly refuses"):
        builder.parse_args([option, "4"])


def test_v3_cli_requires_explicit_source() -> None:
    with pytest.raises(SystemExit):
        builder.parse_args([])


def test_v3_cli_defaults_to_the_frozen_preregistration_receipt() -> None:
    args = builder.parse_args(["--source", "training-source.h5"])
    assert args.source == Path("training-source.h5")
    assert args.prereg == builder.DEFAULT_PREREG
    assert args.freeze_receipt == builder.DEFAULT_FREEZE_RECEIPT


def test_action_profile_id_hashes_only_canonical_float32_content() -> None:
    blocks = np.arange(4 * 5 * 5, dtype=np.float32).reshape(4, 5, 5) / 101.0
    blocks[3] = 0.0
    expected = hashlib.sha256(np.ascontiguousarray(blocks).tobytes()).hexdigest()
    assert builder.action_profile_content_sha256(blocks) == expected
    assert builder.action_profile_content_sha256(blocks.astype(np.float64)) == expected

    changed = blocks.copy()
    changed[2, 4, 3] = np.nextafter(changed[2, 4, 3], np.float32(np.inf))
    assert builder.action_profile_content_sha256(changed) != expected


@pytest.mark.parametrize(
    "blocks",
    (
        np.zeros((4, 5, 4), dtype=np.float32),
        np.full((4, 5, 5), np.nan, dtype=np.float32),
    ),
)
def test_action_profile_id_rejects_wrong_shape_or_nonfinite(
    blocks: np.ndarray,
) -> None:
    with pytest.raises(ValueError):
        builder.action_profile_content_sha256(blocks)


def test_action_profile_contract_rejects_nonzero_terminal_fourth_block() -> None:
    blocks = np.zeros((4, 5, 5), dtype=np.float32)
    blocks[3, 2, 4] = np.float32(0.25)
    with pytest.raises(ValueError, match="terminal fourth"):
        builder.action_profile_content_sha256(blocks)


def test_scene_template_hash_excludes_split_ids_and_action_metadata() -> None:
    candidate = asdict(make_v3_candidate(_base_candidate("train", 3)))
    expected = builder.scene_template_content_sha256(candidate)

    metadata_changed = dict(candidate)
    metadata_changed["candidate_id"] = "different-identity"
    metadata_changed["split"] = "different-split-label"
    metadata_changed["action_profile"] = {
        "action_anchor_id": "different-anchor",
        "action_profile_id": "f" * 64,
    }
    assert builder.scene_template_content_sha256(metadata_changed) == expected

    content_changed = dict(candidate)
    content_changed["cube_color"] = (0.31, 0.4, 0.5)
    assert builder.scene_template_content_sha256(content_changed) != expected


@pytest.mark.parametrize(
    ("field", "size"),
    (("qpos", 20), ("qpos", 22), ("control", 6), ("control", 8)),
)
def test_scene_template_hash_freezes_source_vector_dimensions(
    field: str,
    size: int,
) -> None:
    candidate = asdict(make_v3_candidate(_base_candidate("train", 0)))
    candidate[field] = tuple(0.0 for _ in range(size))
    with pytest.raises(ValueError, match="must contain"):
        builder.scene_template_content_sha256(candidate)


def test_pair_content_hash_binds_raw_scene_and_action_digest_bytes() -> None:
    candidate = make_v3_candidate(_base_candidate("train", 5))
    scene_hash = builder.scene_template_content_sha256(candidate)
    profile_hash = candidate.action_profile.action_profile_id
    expected = hashlib.sha256(
        bytes.fromhex(scene_hash) + bytes.fromhex(profile_hash)
    ).hexdigest()
    assert builder.pair_content_sha256(scene_hash, profile_hash) == expected
    assert builder.pair_content_sha256("0" * 64, profile_hash) != expected
    assert builder.pair_content_sha256(scene_hash, "0" * 64) != expected


@pytest.mark.parametrize("invalid", ("", "g" * 64, "0" * 63))
def test_pair_content_hash_rejects_invalid_digest_inputs(invalid: str) -> None:
    with pytest.raises(ValueError):
        builder.pair_content_sha256(invalid, "0" * 64)


def test_source_h5_receipt_binds_size_rows_hash_and_selection_rule(
    tmp_path: Path,
) -> None:
    source = tmp_path / "training-source.h5"
    shapes = {
        "qpos": (3, 21),
        "control": (3, 7),
        "action": (3, 5),
        "ep_idx": (3,),
        "step_idx": (3,),
        "proprio_gripper_contact": (3, 1),
        "proprio_gripper_opening": (3, 1),
        "privileged_block_0_pos": (3, 3),
        "proprio_effector_pos": (3, 3),
    }
    with h5py.File(source, "w") as handle:
        for name, shape in shapes.items():
            handle.create_dataset(name, data=np.zeros(shape, dtype=np.float32))
        handle.create_dataset("ep_len", data=np.asarray([1, 2], dtype=np.int32))
    receipt = builder._source_h5_receipt(
        source,
        eligible_episode_count=2,
        frozen_source_identity={
            "size_bytes": source.stat().st_size,
            "row_count": 3,
            "episode_count": 2,
            "sha256": "a" * 64,
        },
    )
    assert receipt["source_size_bytes"] == source.stat().st_size
    assert receipt["source_row_count"] == 3
    assert receipt["source_episode_count"] == 2
    assert receipt["source_file_sha256"] == "a" * 64
    assert receipt["source_symbol"] == builder.SOURCE_SYMBOL
    assert receipt["source_content_hash_reused_from_validated_freeze_receipt"]
    assert receipt["source_content_rehashed_by_builder"] is False
    assert receipt["eligible_source_episode_count"] == 2
    assert receipt["eligible_row_selection_rule"] == (
        builder.ELIGIBLE_ROW_SELECTION_RULE
    )


def _write_freeze_receipt_fixture(tmp_path: Path) -> dict[str, object]:
    prereg = tmp_path / "prereg.yaml"
    physics = tmp_path / "physics.py"
    builder_file = tmp_path / "builder.py"
    source = tmp_path / "source.h5"
    receipt_path = tmp_path / "freeze-receipt.json"
    prereg.write_text("protocol: cube-v3\n", encoding="utf-8")
    physics.write_text("V3 = 'physics'\n", encoding="utf-8")
    builder_file.write_text("V3 = 'builder'\n", encoding="utf-8")
    with h5py.File(source, "w") as handle:
        handle.create_dataset(
            "action",
            data=np.zeros((3, 5), dtype=np.float32),
        )
        handle.create_dataset("ep_len", data=np.asarray([1, 2], dtype=np.int32))
    payload = {
        "schema_version": 1,
        "protocol_id": builder.PROTOCOL,
        "status": "frozen_before_first_v3_data_build",
        "checks_passed": True,
        "authorized_splits": ["train", "loader_validation"],
        "public_test": {
            "access_status": "closed_not_read_not_scored",
            "opened": False,
            "read": False,
            "scored": False,
            "hashed": False,
        },
        "reference_model_training_or_scoring_authorized": False,
        "preregistration": {
            "path": str(prereg),
            "sha256": builder.file_sha256(prereg),
            "size_bytes": prereg.stat().st_size,
        },
        "identity": {
            "v3_builder": {
                "path": str(builder_file),
                "sha256": builder.file_sha256(builder_file),
                "size_bytes": builder_file.stat().st_size,
            },
            "v3_physics": {
                "path": str(physics),
                "sha256": builder.file_sha256(physics),
                "size_bytes": physics.stat().st_size,
            },
        },
        "source_h5": {
            "symbol": builder.SOURCE_SYMBOL,
            "path_recorded": False,
            "sha256": builder.file_sha256(source),
            "size_bytes": source.stat().st_size,
            "row_count": 3,
            "episode_count": 2,
        },
    }
    receipt_path.write_text(
        json.dumps(payload, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "prereg": prereg,
        "physics": physics,
        "builder": builder_file,
        "source": source,
        "receipt": receipt_path,
        "payload": payload,
    }


def test_freeze_receipt_gate_binds_current_code_prereg_and_source(
    tmp_path: Path,
) -> None:
    fixture = _write_freeze_receipt_fixture(tmp_path)
    audit = builder.validate_freeze_receipt(
        receipt_path=fixture["receipt"],
        prereg_path=fixture["prereg"],
        source_h5=fixture["source"],
        builder_path=fixture["builder"],
        physics_path=fixture["physics"],
    )
    assert audit["protocol_id"] == builder.PROTOCOL
    assert audit["status"] == "frozen_before_first_v3_data_build"
    assert audit["checks_passed"]
    assert audit["authorized_splits"] == ["train", "loader_validation"]
    assert audit["public_test"]["opened"] is False
    assert audit["source_h5"]["row_count"] == 3
    assert audit["source_h5"]["episode_count"] == 2
    assert audit["source_h5"]["symbol"] == builder.SOURCE_SYMBOL
    assert audit["source_h5"]["content_rehashed_by_builder"] is False
    assert audit["sha256"] == builder.file_sha256(fixture["receipt"])


@pytest.mark.parametrize(
    "mutation",
    (
        "protocol",
        "status",
        "checks",
        "splits",
        "public",
        "prereg_hash",
        "builder_hash",
        "physics_hash",
        "source_size",
        "source_rows",
        "source_episodes",
        "source_symbol",
    ),
)
def test_freeze_receipt_gate_rejects_invalid_authorization_or_identity(
    tmp_path: Path,
    mutation: str,
) -> None:
    fixture = _write_freeze_receipt_fixture(tmp_path)
    payload = fixture["payload"]
    if mutation == "protocol":
        payload["protocol_id"] = "wrong"
    elif mutation == "status":
        payload["status"] = "draft"
    elif mutation == "checks":
        payload["checks_passed"] = False
    elif mutation == "splits":
        payload["authorized_splits"].append("validation")
    elif mutation == "public":
        payload["public_test"]["opened"] = True
    elif mutation == "prereg_hash":
        payload["preregistration"]["sha256"] = "0" * 64
    elif mutation == "builder_hash":
        payload["identity"]["v3_builder"]["sha256"] = "0" * 64
    elif mutation == "physics_hash":
        payload["identity"]["v3_physics"]["sha256"] = "0" * 64
    elif mutation == "source_size":
        payload["source_h5"]["size_bytes"] += 1
    elif mutation == "source_rows":
        payload["source_h5"]["row_count"] += 1
    elif mutation == "source_episodes":
        payload["source_h5"]["episode_count"] += 1
    else:
        payload["source_h5"]["symbol"] = "wrong_source"
    fixture["receipt"].write_text(
        json.dumps(payload, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError):
        builder.validate_freeze_receipt(
            receipt_path=fixture["receipt"],
            prereg_path=fixture["prereg"],
            source_h5=fixture["source"],
            builder_path=fixture["builder"],
            physics_path=fixture["physics"],
        )


def test_invalid_freeze_receipt_is_rejected_before_output_creation(
    tmp_path: Path,
) -> None:
    fixture = _write_freeze_receipt_fixture(tmp_path)
    payload = fixture["payload"]
    payload["status"] = "draft"
    fixture["receipt"].write_text(
        json.dumps(payload, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "must-not-exist"
    with pytest.raises(RuntimeError, match="status"):
        builder.main(
            [
                "--output",
                str(output),
                "--source",
                str(fixture["source"]),
                "--prereg",
                str(fixture["prereg"]),
                "--freeze-receipt",
                str(fixture["receipt"]),
            ]
        )
    assert not output.exists()


def test_v3_profile_factory_is_balanced_and_split_disjoint() -> None:
    anchors = builder._anchor_ids()
    profile_ids: dict[str, set[str]] = {}
    for split in builder.ACTIVE_SPLITS:
        counts = {anchor: 0 for anchor in anchors}
        ids: set[str] = set()
        for catalog_index in range(8):
            profile = make_v3_action_profile(
                split=split,
                catalog_index=catalog_index,
            )
            receipt = asdict(profile)
            blocks = v3_action_blocks(profile)
            assert receipt["action_anchor_id"] == anchors[catalog_index % 4]
            assert receipt["action_profile_id"] == (
                builder.action_profile_content_sha256(blocks)
            )
            counts[receipt["action_anchor_id"]] += 1
            ids.add(receipt["action_profile_id"])
        assert set(counts.values()) == {2}
        assert len(ids) == 8
        profile_ids[split] = ids
    assert not (profile_ids["train"] & profile_ids["loader_validation"])


def test_balanced_acceptance_enforces_four_exact_quotas_and_unique_profiles() -> None:
    tracker = builder._BalancedAcceptance(pair_count=8)
    anchors = builder._anchor_ids()
    assert tracker.consider(anchor=anchors[0], profile_id="a0")
    assert tracker.consider(anchor=anchors[0], profile_id="a1")
    assert not tracker.consider(anchor=anchors[0], profile_id="a2")
    assert not tracker.consider(anchor=anchors[1], profile_id="a1")
    for anchor_index, anchor in enumerate(anchors[1:], start=1):
        assert tracker.consider(anchor=anchor, profile_id=f"{anchor_index}-0")
        assert tracker.consider(anchor=anchor, profile_id=f"{anchor_index}-1")
    assert tracker.complete
    assert set(tracker.counts.values()) == {2}
    assert tracker.quota_full_candidates == 1
    assert tracker.duplicate_profile_candidates == 1


def test_worker_passes_a_distinct_replay_simulator(monkeypatch: pytest.MonkeyPatch) -> None:
    replay = object()

    class _Primary:
        observed_replay: object | None = None

        def build_pair(self, _candidate: object, *, replay_simulator: object):
            self.observed_replay = replay_simulator
            return None

    primary = _Primary()
    monkeypatch.setattr(builder, "_WORKER_SIMULATOR", primary)
    monkeypatch.setattr(builder, "_WORKER_REPLAY_SIMULATOR", replay)
    candidate = make_v3_candidate(_base_candidate("train", 0))
    assert builder._build_candidate(candidate) is None
    assert primary.observed_replay is replay
    assert primary is not replay


def _cross_split_fixture() -> dict[str, dict[str, object]]:
    anchors = list(builder._anchor_ids())
    return {
        "train": {
            "query_hashes": ["query-train"],
            "source_episodes": [1],
            "action_profile_ids": ["profile-train"],
            "scene_template_content_hashes": ["scene-train"],
            "pair_content_hashes": ["content-pair-train"],
            "pair_ids": ["pair-train"],
            "action_anchor_ids": anchors,
        },
        "loader_validation": {
            "query_hashes": ["query-development"],
            "source_episodes": [2],
            "action_profile_ids": ["profile-development"],
            "scene_template_content_hashes": ["scene-development"],
            "pair_content_hashes": ["content-pair-development"],
            "pair_ids": ["pair-development"],
            "action_anchor_ids": anchors,
        },
    }


def test_cross_split_audit_distinguishes_profiles_from_anchor_families() -> None:
    audit = builder._cross_split_audit(_cross_split_fixture())
    assert audit["passed"]
    assert audit["query_pixel_hash_overlap"]["count"] == 0
    assert audit["source_episode_overlap"]["count"] == 0
    assert audit["exact_action_profile_id_overlap"]["count"] == 0
    assert audit["scene_template_content_hash_overlap"]["count"] == 0
    assert audit["pair_content_hash_overlap"]["count"] == 0
    assert audit["action_anchor_family_overlap"]["count"] == 4
    assert audit["action_anchor_family_overlap"]["expected_count"] == 4
    assert "not exact profiles" in audit["action_anchor_family_overlap"][
        "interpretation"
    ]
    assert audit["pair_id_is_content_isolation_evidence"] is False


@pytest.mark.parametrize(
    ("field", "check"),
    (
        ("query_hashes", "query_pixel_hash_overlap_zero"),
        ("source_episodes", "source_episode_overlap_zero"),
        ("action_profile_ids", "exact_action_profile_id_overlap_zero"),
        (
            "scene_template_content_hashes",
            "scene_template_content_hash_overlap_zero",
        ),
        ("pair_content_hashes", "pair_content_hash_overlap_zero"),
    ),
)
def test_cross_split_audit_rejects_required_zero_overlap(
    field: str,
    check: str,
) -> None:
    reports = _cross_split_fixture()
    reports["loader_validation"][field] = list(reports["train"][field])
    audit = builder._cross_split_audit(reports)
    assert not audit["passed"]
    assert not audit["checks"][check]


def test_cross_split_audit_requires_all_four_shared_anchor_families() -> None:
    reports = _cross_split_fixture()
    reports["loader_validation"]["action_anchor_ids"] = list(
        builder._anchor_ids()[:3]
    )
    audit = builder._cross_split_audit(reports)
    assert not audit["passed"]
    assert not audit["checks"]["four_common_action_anchor_families_expected"]


def test_split_prefixed_pair_id_is_not_used_as_content_isolation_evidence() -> None:
    reports = _cross_split_fixture()
    reports["loader_validation"]["pair_ids"] = reports["train"]["pair_ids"]
    audit = builder._cross_split_audit(reports)
    assert audit["passed"]
    assert audit["pair_id_is_content_isolation_evidence"] is False


def _base_candidate(split: str, catalog_index: int) -> CubeGraspRuleCandidate:
    return CubeGraspRuleCandidate(
        candidate_id=f"fixture-{split}-{catalog_index}",
        split=split,
        catalog_index=catalog_index,
        source_row=10 + catalog_index,
        source_episode=20 + catalog_index,
        source_step=30,
        simulator_seed=40,
        task_id=1,
        qpos=tuple(0.0 for _ in range(21)),
        control=tuple(0.0 for _ in range(7)),
        cube_color=(0.3, 0.4, 0.5),
        target_position=(0.4, 0.0, 0.02),
    )


def _built_result_fixture() -> dict[str, object]:
    candidate = make_v3_candidate(_base_candidate("train", 0))
    candidate_receipt = asdict(candidate)
    profile_receipt = asdict(candidate.action_profile)
    blocks = np.asarray(v3_action_blocks(candidate.action_profile), dtype=np.float32)
    profile_id = builder.action_profile_content_sha256(blocks)
    anchor_id = profile_receipt["action_anchor_id"]
    episodes = {
        mode: {
            "pixels": [b"jpeg"] * 4,
            "action_blocks": blocks.copy(),
            "physical_state": np.zeros((4, 7), dtype=np.float32),
            "hidden_value": float(index),
            "action_anchor_id": anchor_id,
            "action_profile_id": profile_id,
        }
        for index, mode in enumerate(GRASP_MODES)
    }
    replay_modes = {
        mode: {
            "passed": True,
            "checks": {"pixels_bitwise_equal": True},
            "maximum_physical_state_gap": 0.0,
            "maximum_simulator_state_gap": 0.0,
            "changed_rgb_values": 0,
            "changed_pixels": 0,
            "hashes": {
                "continuous": {"pixels": f"{mode}-pixels"},
                "fresh_replay": {"pixels": f"{mode}-pixels"},
            },
        }
        for mode in GRASP_MODES
    }
    return {
        "candidate": candidate_receipt,
        "action_profile": profile_receipt,
        "content_hashes": {
            "scene_template_content_hash": builder.scene_template_content_sha256(
                candidate_receipt
            ),
            "action_profile_id": profile_id,
            "pair_content_hash": builder.pair_content_sha256(
                builder.scene_template_content_sha256(candidate_receipt),
                profile_id,
            ),
        },
        "audit": {
            "passed": True,
            "hashes": {"query_pixels": "query-fixture"},
            "v3": {
                "action_anchor_id": anchor_id,
                "action_profile_id": profile_id,
                "profile_constraints": {
                    "maximum_action_abs": float(np.abs(blocks).max()),
                },
                "fresh_simulator_replay": {
                    "passed": True,
                    "independent_simulator_instance": True,
                    "provided_reusable_instance": True,
                    "maximum_physical_state_gap": 0.0,
                    "maximum_simulator_state_gap": 0.0,
                    "total_changed_rgb_values": 0,
                    "total_changed_pixels": 0,
                    "modes": replay_modes,
                },
            },
        },
        "episodes": episodes,
    }


def test_built_result_recomputes_profile_content_identity() -> None:
    row = builder._validate_built_result(_built_result_fixture(), "train")
    assert row["action_anchor_id"] == builder._anchor_ids()[0]
    assert row["action_profile_id"] == builder.action_profile_content_sha256(
        _built_result_fixture()["episodes"][GRASP_MODES[0]]["action_blocks"]
    )
    assert row["scene_template_content_hash"] == (
        builder.scene_template_content_sha256(row["candidate"])
    )
    assert row["pair_content_hash"] == builder.pair_content_sha256(
        row["scene_template_content_hash"],
        row["action_profile_id"],
    )


def test_built_result_rejects_metadata_only_profile_identity() -> None:
    result = _built_result_fixture()
    result["episodes"][GRASP_MODES[1]]["action_blocks"][0, 0, 0] += np.float32(
        0.125
    )
    with pytest.raises(RuntimeError, match="action blocks differ"):
        builder._validate_built_result(result, "train")


def test_built_result_rejects_tampered_scene_pair_content_receipt() -> None:
    result = _built_result_fixture()
    result["content_hashes"]["pair_content_hash"] = "0" * 64
    with pytest.raises(RuntimeError, match="content-hash receipt"):
        builder._validate_built_result(result, "train")


def test_fresh_replay_summary_uses_pair_replay_audits_not_query_gap() -> None:
    first = builder._validate_built_result(_built_result_fixture(), "train")
    second = builder._validate_built_result(_built_result_fixture(), "train")
    summary = builder._fresh_replay_summary([first, second])
    assert summary["passed"]
    assert summary["pair_count"] == 2
    assert summary["mode_replay_count"] == 4
    assert summary["maximum_physical_state_gap"] == 0.0
    assert summary["maximum_simulator_state_gap"] == 0.0
    assert summary["total_changed_rgb_values"] == 0
    assert summary["query_gap_used_as_replay_substitute"] is False

    build_summary = builder._fresh_replay_build_summary(
        {
            split: {"fresh_simulator_replay": summary}
            for split in builder.ACTIVE_SPLITS
        }
    )
    assert build_summary["passed"]
    assert build_summary["pair_count"] == 4
    assert build_summary["mode_replay_count"] == 8
    assert build_summary["query_gap_used_as_replay_substitute"] is False


def test_causal_solver_gate_uses_fresh_replay_not_query_gap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def _capture(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return dict(kwargs)

    monkeypatch.setattr(builder, "audit_causal_data_contract", _capture)
    reports = {
        split: {
            "maximum_query_simulator_state_gap": 0.0,
            "maximum_state_installations_after_x0": 0,
            "minimum_history_cube_height_gap_m": 0.01,
            "minimum_future_cube_height_gap_m": 0.01,
        }
        for split in builder.ACTIVE_SPLITS
    }
    builder._audit_causal_contract_from_real_replay(
        reports,
        {"passed": False},
    )
    assert captured["maximum_query_state_gap"] == 0.0
    assert captured["solver_cache_check_passed"] is False
    assert any(
        "distinct simulator" in value for value in captured["evidence"]
    )


def test_record_batch_persists_privileged_anchor_and_profile_columns() -> None:
    result = _built_result_fixture()
    row = builder._validate_built_result(result, "train")
    batch = builder._record_batch(
        result["episodes"][GRASP_MODES[0]],
        episode_index=0,
        split="train",
        candidate=row["candidate"],
        mode=GRASP_MODES[0],
        action_anchor_id=row["action_anchor_id"],
        action_profile_id=row["action_profile_id"],
        scene_template_content_hash=row["scene_template_content_hash"],
        pair_content_hash=row["pair_content_hash"],
    )
    assert batch.schema == builder.SCHEMA
    values = batch.to_pydict()
    assert set(values["action_anchor_id"]) == {row["action_anchor_id"]}
    assert set(values["action_profile_id"]) == {row["action_profile_id"]}
    assert set(values["scene_template_content_hash"]) == {
        row["scene_template_content_hash"]
    }
    assert set(values["pair_content_hash"]) == {row["pair_content_hash"]}
    assert set(values["source_step"]) == {row["candidate"]["source_step"]}


def test_request_and_manifest_explicitly_keep_public_closed(
    tmp_path: Path,
) -> None:
    freeze_audit = {
        "path": "artifacts/evaluation/freeze.json",
        "sha256": "b" * 64,
        "size_bytes": 789,
        "status": "frozen_before_first_v3_data_build",
        "checks_passed": True,
        "source_h5": {"sha256": "a" * 64},
    }
    request = builder._request_payload(
        pair_counts=builder.DEFAULT_PAIR_COUNTS,
        source_receipt={
            "source_symbol": builder.SOURCE_SYMBOL,
            "source_size_bytes": 123,
            "source_row_count": 456,
            "source_episode_count": 12,
            "source_file_sha256": "a" * 64,
            "eligible_row_selection_rule": builder.ELIGIBLE_ROW_SELECTION_RULE,
        },
        jpeg_quality=95,
        workers=1,
        output=tmp_path,
        freeze_receipt_audit=freeze_audit,
    )
    assert request["active_splits"] == ["train", "loader_validation"]
    assert request["pair_counts"] == builder.DEFAULT_PAIR_COUNTS
    assert request["evidence_scope"] == builder.EVIDENCE_SCOPE
    assert request["profile_split_policy"] == builder.PROFILE_SPLIT_POLICY
    assert request["public_test_opened"] is False
    assert request["public_test_generated"] is False
    assert request["freeze_receipt"] == {
        "path": "artifacts/evaluation/freeze.json",
        "sha256": "b" * 64,
        "size_bytes": 789,
        "status": "frozen_before_first_v3_data_build",
        "checks_passed": True,
    }
    assert request["source_content_sha256"] == "a" * 64
    assert request["source"]["source_symbol"] == builder.SOURCE_SYMBOL
    assert request["action_profile_contract"][
        "exact_profile_ids_split_disjoint"
    ]
    assert request["action_profile_contract"]["terminal_fourth_block"] == {
        "block_index": 3,
        "shape": [5, 5],
        "dtype": "float32",
        "all_values_exactly_zero": True,
        "role": "format-only terminal block; no transition target",
    }
    reproducibility = request["reproducibility_contract"]
    assert reproducibility["candidate_assignment_seed"] == (
        builder.CANDIDATE_ASSIGNMENT_SEED
    )
    assert reproducibility["catalog_seeds"] == builder.CATALOG_SEEDS
    assert reproducibility["profile_split_seeds"] == {
        split: builder.V3_PROFILE_SPLIT_SEEDS[split]
        for split in builder.ACTIVE_SPLITS
    }
    assert reproducibility["candidate_pool_multiplier"] == (
        builder.CANDIDATE_POOL_MULTIPLIER
    )
    assert reproducibility["eligible_row_selection_rule"] == (
        builder.ELIGIBLE_ROW_SELECTION_RULE
    )
    assert reproducibility["source_h5_identity"] == {
        "size_bytes": 123,
        "row_count": 456,
        "episode_count": 12,
        "file_sha256": "a" * 64,
    }
    content_contract = request["content_identity_contract"]
    assert "split" in content_contract["scene_template_content_hash"][
        "excluded_fields"
    ]
    assert content_contract["pair_id_is_content_isolation_evidence"] is False
    assert request["fresh_simulator_replay_contract"] == {
        "required_for_every_accepted_pair": True,
        "primary_and_replay_simulators_distinct": True,
        "environments_not_shared": True,
        "one_reusable_primary_and_one_reusable_replay_instance_per_worker": True,
        "maximum_physical_state_gap": builder.QUERY_STATE_TOLERANCE,
        "maximum_complete_simulator_state_gap": builder.QUERY_STATE_TOLERANCE,
        "pixels_bitwise_equal": True,
        "actions_bitwise_equal": True,
        "query_gap_may_substitute_for_replay": False,
    }

    (tmp_path / "request.json").write_text("{}\n", encoding="utf-8")
    manifest = builder._manifest_payload(
        tmp_path,
        build_report={"passed": True},
    )
    assert manifest["active_splits"] == ["train", "loader_validation"]
    assert manifest["evidence_scope"] == builder.EVIDENCE_SCOPE
    assert manifest["profile_split_policy"] == builder.PROFILE_SPLIT_POLICY
    assert manifest["public_test_opened"] is False
    assert manifest["public_test_generated"] is False
    assert set(manifest["files"]) == {"request.json"}
