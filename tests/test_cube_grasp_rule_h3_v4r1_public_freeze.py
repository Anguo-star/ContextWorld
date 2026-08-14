from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest
import yaml

from scripts.finalize_cube_grasp_rule_h3_v4r1_prior_exclusions import (
    canonical_content_digest,
    excluded_source_episodes_sha256,
)
import scripts.freeze_cube_grasp_rule_h3_v4r1_public_release as freezer
from contextworld.benchmarks.cube_grasp_rule_public_contract import (
    validate_public_preregistration_contract,
)


ROOT = Path(__file__).resolve().parents[1]
PREREG = (
    ROOT
    / "configs/benchmark/cube_gripper_carry_h3_v4r1_public_release_prereg_v1.yaml"
)


def _digest(index: int) -> str:
    return hashlib.sha256(f"public-freeze-{index}".encode()).hexdigest()


def _split(name: str, pair_count: int, offset: int) -> dict:
    return {
        "split": name,
        "pair_count": pair_count,
        "passed": True,
        "source_episodes": list(range(offset, offset + pair_count)),
        "action_profile_ids": [_digest(10_000 + offset + index) for index in range(pair_count)],
        "scene_template_content_hashes": [
            _digest(20_000 + offset + index) for index in range(pair_count)
        ],
        "pair_content_hashes": [_digest(30_000 + offset + index) for index in range(pair_count)],
        "query_hashes": [_digest(40_000 + offset + index) for index in range(pair_count)],
        "action_anchor_counts": {
            "endpoint4": pair_count // 4,
            "front_hold": pair_count // 4,
            "plateau": pair_count // 4,
            "ramp4": pair_count // 4,
        },
        "prior_episode_and_content_exclusion": {"passed": True},
    }


def _union_fixture() -> tuple[dict, dict, dict]:
    prior = {
        "checks_passed": True,
        "excluded_source_episodes": [9_999],
        "prior_content_exclusions": {
            name: {"values": [_digest(index + 1)]}
            for index, name in enumerate(freezer.CONTENT_FIELDS)
        },
    }
    build = {
        "passed": True,
        "active_splits": ["train", "loader_validation"],
        "public_test_generated": False,
        "public_test_opened": False,
        "cross_split_audit": {"passed": True},
        "splits": {
            "train": _split("train", 2048, 0),
            "loader_validation": _split("loader_validation", 256, 2048),
        },
    }
    source = sorted(
        {9_999}
        | set(build["splits"]["train"]["source_episodes"])
        | set(build["splits"]["loader_validation"]["source_episodes"])
    )
    key_map = {
        "action_profile_ids": "action_profile_ids",
        "scene_template_content_hashes": "scene_template_content_hashes",
        "pair_content_hashes": "pair_content_hashes",
        "query_pixel_hashes": "query_hashes",
    }
    expected = {
        "source_episodes": {
            "count": len(source),
            "sha256": excluded_source_episodes_sha256(source),
        }
    }
    for field, build_key in key_map.items():
        values = sorted(
            set(prior["prior_content_exclusions"][field]["values"])
            | set(build["splits"]["train"][build_key])
            | set(build["splits"]["loader_validation"][build_key])
        )
        expected[field] = {
            "count": len(values),
            "sha256": canonical_content_digest(values, field_name=field),
        }
    prereg = {"public_data_generation": {"exclusion_union": expected}}
    return prereg, prior, build


def test_public_exclusion_union_binds_all_non_public_content() -> None:
    prereg, prior, build = _union_fixture()
    receipt = freezer._union_receipt(prereg, prior=prior, build=build)
    assert receipt["checks_passed"] is True
    assert receipt["excluded_source_episode_count"] == 2305
    assert receipt["coverage"] == {
        "historical_prior_receipt": True,
        "v4r1_train": True,
        "v4r1_loader_validation": True,
        "public_content_included": False,
    }
    assert all(
        receipt["prior_content_exclusions"][name]["count"] == 2305
        for name in freezer.CONTENT_FIELDS
    )


def test_public_exclusion_union_fails_closed_on_digest_drift() -> None:
    prereg, prior, build = _union_fixture()
    mutated = copy.deepcopy(prereg)
    mutated["public_data_generation"]["exclusion_union"][
        "query_pixel_hashes"
    ]["sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="query_pixel_hashes exclusion union drifted"):
        freezer._union_receipt(mutated, prior=prior, build=build)


def test_formal_public_preregistration_is_closed_and_dimension_safe() -> None:
    prereg = yaml.safe_load(PREREG.read_text(encoding="utf-8"))
    validate_public_preregistration_contract(prereg)
    freezer._validate_preregistration_contract(prereg)
    assert prereg["scope"]["flattened_action_input_dim"] == 25
    assert prereg["public_evaluation"]["authorized_model_families"] == ["lewm"]
    assert prereg["public_test_before_freeze"]["read"] is False
    assert prereg["public_test_before_freeze"]["scored"] is False
    assert prereg["one_use_policy"]["retry_after_access_authorized"] is False
    assert prereg["public_evaluation"]["data_access_contract"][
        "model_visible_fields"
    ] == ["pixels", "action_block"]


def test_public_preregistration_rejects_normalization_or_data_boundary_drift() -> None:
    prereg = yaml.safe_load(PREREG.read_text(encoding="utf-8"))
    bad_std = copy.deepcopy(prereg)
    bad_std["public_evaluation"]["action_normalization"]["std_population"][2] = 0.0
    with pytest.raises(RuntimeError, match="action normalization drifted"):
        freezer._validate_preregistration_contract(bad_std)

    leaked = copy.deepcopy(prereg)
    leaked["public_evaluation"]["data_access_contract"][
        "adapter_receives_only"
    ].append("hidden_mode")
    with pytest.raises(RuntimeError, match="model/evaluator data boundary drifted"):
        freezer._validate_preregistration_contract(leaked)
