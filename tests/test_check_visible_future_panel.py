from __future__ import annotations

import numpy as np

from scripts import check_visible_future_panel as check


def test_geometry_uses_full_bounds_and_margin() -> None:
    bounds = np.zeros((2, 3, 25, 2, 4), dtype=np.float64)
    bounds[..., 0] = 20.0
    bounds[..., 1] = 30.0
    bounds[..., 2] = 40.0
    bounds[..., 3] = 50.0

    ok, candidate_safe, _ = check._geometry_safety(bounds)
    assert ok
    assert candidate_safe.tolist() == [True, True, True]

    # The center can be inside while the full shape edge crosses the margin.
    bounds[1, 0, 12, 0, 0] = 1.9
    ok, candidate_safe, _ = check._geometry_safety(bounds)
    assert not ok
    assert candidate_safe.tolist() == [False, True, True]


def test_factor_ladder_has_ordered_scale_choices() -> None:
    assert check.CANDIDATE_FACTORS.tolist() == [1.0, 0.75, 0.5, 0.25, 0.125, 0.0625, 0.0]
    safe = np.asarray([False, True, True, True, True, True, True])
    selected, found = check._select_maximum_safe_factor(safe)
    assert found
    assert selected == 0.75
    no_safe = np.zeros(7, dtype=bool)
    fallback, found = check._select_maximum_safe_factor(no_safe)
    assert not found
    # The builder's factor-0 trajectory remains represented when no factor is safe.
    assert fallback == 0.0


def test_adjacent_factor_search_receipt_is_validated_with_trajectory_receipt() -> None:
    row = {
        "trajectory_receipt": {
            "replayed_original_history": True,
            "query_state_injected": False,
            "all_candidates_continuous": True,
            "future_raw_steps": 25,
            "prefix_pixels_exact": [True, True],
            "prefix_state_gap": [0.0, 0.0],
        },
        "factor_search_receipt": {
            "factor_search_complete": True,
            "factors": [1.0, 0.75, 0.5, 0.25, 0.125, 0.0625, 0.0],
        },
    }
    receipt = check._receipt_for({}, row, "scene")
    failures: list[dict[str, str]] = []
    checks = check._receipt_checks("action_strength", "scene", receipt, failures)
    assert all(checks.values())
    assert not failures


def test_frozen_manifest_source_group_metadata_must_be_preserved() -> None:
    old = [
        {"scene_id": "a", "row": {"bootstrap_cluster": "group-1"}},
        {"scene_id": "b", "row": {"bootstrap_cluster": "group-2"}},
    ]
    matching = [
        {"scene_id": "a", "row": {"bootstrap_cluster": "group-1"}},
        {"scene_id": "b", "row": {"bootstrap_cluster": "group-2"}},
    ]
    dropped = [
        {"scene_id": "a", "row": {"bootstrap_cluster": "group-1"}},
        {"scene_id": "b", "row": {}},
    ]
    assert check._manifest_metadata_mismatches(old, matching) == {"bootstrap_cluster": []}
    assert check._manifest_metadata_mismatches(old, dropped) == {"bootstrap_cluster": ["b"]}


def test_action_dedup_uses_exact_float32_bytes() -> None:
    actions = np.zeros((3, 5, 5, 2), dtype=np.float32)
    actions[1] = actions[0]
    actions[2, 0, 0, 0] = np.nextafter(np.float32(0), np.float32(1))
    fingerprints, representatives, groups = check._candidate_fingerprints(actions)
    assert representatives == [0, 2]
    assert groups[fingerprints[0]] == [0, 1]
    assert fingerprints[0] != fingerprints[2]


def test_action_dedup_canonicalizes_signed_zero() -> None:
    actions = np.zeros((2, 5, 5, 2), dtype=np.float32)
    actions[1, 0, 0, 0] = np.float32(-0.0)
    assert np.signbit(actions[1, 0, 0, 0])
    fingerprints, representatives, groups = check._candidate_fingerprints(actions)
    assert representatives == [0]
    assert fingerprints[0] == fingerprints[1]
    assert groups[fingerprints[0]] == [0, 1]
    assert check.canonical_action_bytes(actions[0]) == check.canonical_action_bytes(actions[1])


def test_unique_candidate_axis_maps_all_legacy_slots_once() -> None:
    old_actions = np.zeros((11, 5, 5, 2), dtype=np.float32)
    slot_actions = np.zeros_like(old_actions)
    new = {
        "candidate_actions": np.zeros((1, 5, 5, 2), dtype=np.float32),
        "candidate_slot_actions": slot_actions,
        "candidate_parent_index": np.arange(11, dtype=np.int64),
        "candidate_factor": np.ones(11, dtype=np.float32),
        "candidate_safe": np.ones(11, dtype=bool),
        "factor_search_safe": np.ones((11, 7), dtype=bool),
        "candidate_slot_to_unique": np.zeros(11, dtype=np.int64),
    }
    failures: list[dict[str, str]] = []
    result = check._check_bank("action_strength", "scene", {"candidate_actions": old_actions}, new, {}, failures)
    assert result["candidate_count"] == 1
    assert result["legacy_slot_count"] == 11
    assert result["unique_action_count"] == 1
    assert not failures


def test_no_safe_factor_is_a_failure_and_slot_is_not_removed() -> None:
    old_actions = np.zeros((11, 5, 5, 2), dtype=np.float32)
    new = {
        "candidate_actions": np.zeros((1, 5, 5, 2), dtype=np.float32),
        "candidate_slot_actions": np.zeros((11, 5, 5, 2), dtype=np.float32),
        "candidate_parent_index": np.arange(11, dtype=np.int64),
        "candidate_factor": np.asarray([0.0] + [1.0] * 10, dtype=np.float32),
        "candidate_safe": np.asarray([False] + [True] * 10, dtype=bool),
        "factor_search_safe": np.ones((11, 7), dtype=bool),
        "candidate_slot_to_unique": np.zeros(11, dtype=np.int64),
    }
    new["factor_search_safe"][0] = False
    failures: list[dict[str, str]] = []
    result = check._check_bank("action_strength", "scene", {"candidate_actions": old_actions}, new, {}, failures)
    assert result["candidate_count"] == 1
    assert result["legacy_slot_count"] == 11
    assert any(item["check"] == "no_safe_candidate_factor" for item in failures)


def test_state_gap_threshold_is_two_render_pixels() -> None:
    left = np.zeros(7, dtype=np.float64)
    right = left.copy()
    right[0] = check.PHYSICAL_DIFF_THRESHOLD_WORLD - 1e-5
    gap = check._state_gaps("action_strength", left, right)
    assert gap["agent_world_gap"] < check.PHYSICAL_DIFF_THRESHOLD_WORLD
    right[0] = check.PHYSICAL_DIFF_THRESHOLD_WORLD + 1e-5
    gap = check._state_gaps("action_strength", left, right)
    assert gap["agent_world_gap"] > check.PHYSICAL_DIFF_THRESHOLD_WORLD


def test_exact_pixel_alias_requires_identical_rgb_and_material_state_gap() -> None:
    pixels = np.arange(2 * 1 * 5 * 3, dtype=np.uint8).reshape(2, 1, 5, 1, 1, 3)
    pixels[1, 0, 0] = pixels[0, 0, 0]
    states = np.zeros((2, 1, 25, 7), dtype=np.float64)
    states[1, 0, 4, 0] = check.PHYSICAL_DIFF_THRESHOLD_WORLD + 1.0
    result = check._find_exact_pixel_aliases(pixels, states, np.zeros((1, 5, 5, 2), dtype=np.float32), "action_strength")
    assert result["qualifying_alias_pair_count"] == 1
    assert result["qualifying_alias_pair_counts"]["same_horizon_across_conditions"] == 1

    # A one-channel pixel change is not an identical-image counterexample.
    pixels[1, 0, 0, 0, 0, 0] += 1
    result = check._find_exact_pixel_aliases(pixels, states, np.zeros((1, 5, 5, 2), dtype=np.float32), "action_strength")
    assert result["qualifying_alias_pair_count"] == 0


def test_temporal_same_candidate_alias_is_reported_by_horizon() -> None:
    pixels = np.arange(2 * 1 * 5 * 3, dtype=np.uint8).reshape(2, 1, 5, 1, 1, 3)
    pixels[0, 0, 4] = pixels[0, 0, 3]
    states = np.zeros((2, 1, 25, 7), dtype=np.float64)
    states[0, 0, 24, 2] = check.PHYSICAL_DIFF_THRESHOLD_WORLD + 1.0
    result = check._find_exact_pixel_aliases(pixels, states, np.zeros((1, 5, 5, 2), dtype=np.float32), "action_strength")
    assert result["qualifying_alias_pair_counts"]["same_candidate_across_horizons"] == 1
    assert result["examples"][0]["physical_step_a"] != result["examples"][0]["physical_step_b"]
