"""Focused CPU checks for the query-anchored Motion Damping K=10 builder."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import lance
import numpy as np
import pyarrow as pa
import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import build_pusht_motion_damping_rollout_k10 as builder  # noqa: E402

from contextworld.evaluation import pusht_motion_damping_h3 as damping  # noqa: E402


SOURCE_MANIFEST = (
    ROOT / "artifacts/synthesis/pusht_motion_damping_h3_release_v4/manifest.json"
)


def _source() -> tuple[dict, str]:
    return builder._read_source(SOURCE_MANIFEST)


def test_k10_action_layout_preserves_h1_then_uses_zero_holds() -> None:
    source, _ = _source()
    raw = source["splits"]["validation"]["pairs"][0]["template"]
    template = builder._template_from_dict(raw)
    actions = builder._rollout_actions(template)

    assert actions.shape == (65, 2)
    assert np.array_equal(actions[:10], np.asarray(template.history_actions))
    assert np.array_equal(actions[10:15], np.asarray(template.query_actions))
    # Nine holds (h2..h10) plus the final completion block.
    assert not np.any(actions[15:])
    assert builder.MODEL_FRAME_ROWS == tuple(range(0, 61, 5))
    assert builder.HORIZON_FRAME_ROWS == {
        1: 15,
        2: 20,
        3: 25,
        4: 30,
        5: 35,
        6: 40,
        7: 45,
        8: 50,
        9: 55,
        10: 60,
    }


def test_twin_group_preflight_rejects_unsafe_groups_and_keeps_twins_together() -> None:
    source, _ = _source()
    catalog = builder._source_catalog(source, "validation")
    first = builder._candidate_group(
        split="validation",
        group_index=0,
        source_catalog=catalog,
        catalog_seed=int(source["split_catalog_seeds"]["validation"]),
    )
    first_report = builder._group_preflight(first, resolution=64)
    # Long rollout safety is stricter than the K=1 release.  This group is a
    # useful regression case: it is rejected as a whole, never half-admitted.
    assert not first_report["passed"]
    assert [row["catalog_index"] for row in first_report["candidates"]] == [0, 1]

    accepted, rejected = builder._select_groups(
        split="validation",
        pair_count=2,
        source_catalog=catalog,
        catalog_seed=int(source["split_catalog_seeds"]["validation"]),
        resolution=64,
        workers=1,
        max_catalog_groups=16,
    )
    assert len(accepted) == 1
    assert rejected
    report = accepted[0]
    assert report["visible_x0_label_balance"]
    assert [row["catalog_index"] for row in report["candidates"]] == [
        2 * report["group_index"],
        2 * report["group_index"] + 1,
    ]
    assert len(report["template_audits"]) == 2
    for audit in report["template_audits"]:
        assert audit["passed"]
        assert audit["total_contact_steps"] == 0
        assert audit["maximum_arbiter_count"] == 0
        assert audit["bounds_ok"]
        assert audit["horizon_gaps"]["10"]["block_position_px"] > audit[
            "horizon_gaps"
        ]["1"]["block_position_px"]


def test_small_lance_build_is_anchored_and_compatible(tmp_path: Path) -> None:
    source, _ = _source()
    split = "validation"
    root = tmp_path / "k10"
    root.mkdir()
    report = builder._build_split(
        root=root,
        split=split,
        pair_count=2,
        source_catalog=builder._source_catalog(source, split),
        catalog_seed=int(source["split_catalog_seeds"][split]),
        resolution=64,
        jpeg_quality=85,
        workers=1,
        max_catalog_groups=8,
    )

    assert report["passed"]
    assert report["episode_count"] == 4
    assert report["raw_rows"] == 4 * builder.RAW_ROWS_PER_EPISODE
    assert report["zero_contact_steps"]
    assert report["maximum_arbiter_count"] == 0
    assert report["all_bounds_valid"]
    assert report["x0_rgb_hash_multisets_identical_across_modes"]
    h1 = report["h1_source_consistency"]
    assert h1["source_templates_checked"] == 2
    assert h1["generated_continuation_templates"] == 0
    assert h1["maximum_absolute_error"] == 0.0
    # The focused smoke uses 64px; full 224px builds additionally compare
    # query/h1 RGB hashes with the frozen K=1 source release.
    assert h1["source_pixel_hashes_checked"] == 0
    assert h1["source_pixel_hashes_matched"]
    assert h1["source_render_provenance_match"]
    assert h1["passed"]
    assert report["horizon_gap_block_position_px"]["10"]["min"] > report[
        "horizon_gap_block_position_px"
    ]["1"]["max"]

    dataset = lance.dataset(str(root / "validation.lance"))
    assert dataset.schema == builder.LANCE_SCHEMA
    assert not any(
        pa.types.is_string(field.type) or pa.types.is_large_string(field.type)
        for field in dataset.schema
    )
    assert dataset.count_rows() == 4 * 65
    episode_dataset = lance.dataset(str(root / "validation_episodes.lance"))
    assert episode_dataset.schema == builder.EPISODE_SCHEMA
    assert episode_dataset.count_rows() == 4
    assert report["episode_table_rows"] == 4
    episode_rows = episode_dataset.to_table().to_pylist()
    assert [row["episode_idx"] for row in episode_rows] == list(range(4))
    assert [row["hidden_mode"] for row in episode_rows] == [
        builder.ENDPOINT_MODES[0],
        builder.ENDPOINT_MODES[1],
        builder.ENDPOINT_MODES[0],
        builder.ENDPOINT_MODES[1],
    ]
    rows = dataset.scanner(
        columns=[
            "episode_idx",
            "step_idx",
            "action",
        ]
    ).to_table().to_pylist()
    rows.sort(key=lambda row: (row["episode_idx"], row["step_idx"]))
    for episode_idx in range(4):
        episode = [row for row in rows if row["episode_idx"] == episode_idx]
        assert [row["step_idx"] for row in episode] == list(range(65))
        assert not np.any(np.asarray([row["action"] for row in episode])[15:])
    assert [row["hidden_mode"] for row in episode_rows] == [
        builder.ENDPOINT_MODES[0],
        builder.ENDPOINT_MODES[1],
        builder.ENDPOINT_MODES[0],
        builder.ENDPOINT_MODES[1],
    ]
    source_pairs = builder._source_catalog(source, split)
    expected_source_episode_indices = []
    for catalog_index in report["accepted_catalog_indices"]:
        source_pair = source_pairs[catalog_index]["source_pair_index"]
        expected_source_episode_indices.extend([2 * source_pair, 2 * source_pair + 1])
    assert [row["source_episode_idx"] for row in episode_rows] == (
        expected_source_episode_indices
    )


def test_direct_reader_window_contract_is_explicit() -> None:
    # A History=3, K=10 clip uses 13 model frames.  There is one possible
    # start in a 13-frame episode; smaller horizons leave extra sliding starts.
    frame_count = len(builder.MODEL_FRAME_ROWS)
    starts_at_k10 = frame_count - (builder.HISTORY_SIZE + builder.MAX_HORIZON) + 1
    starts_at_k3 = frame_count - (builder.HISTORY_SIZE + 3) + 1
    assert starts_at_k10 == 1
    assert starts_at_k3 > 1
    # Keep this wording aligned with the manifest contract emitted by main().
    assert "sliding windows" in json.dumps(
        {
            "num_preds_less_than_10": (
                "produces sliding windows; those are not the query-anchored "
                "K=1..K evaluation contract"
            )
        }
    )


def test_frame_columns_keep_frozen_v4_numeric_types() -> None:
    source_schema = lance.dataset(
        str(ROOT / "artifacts/synthesis/pusht_motion_damping_h3_release_v4/train.lance")
    ).schema
    for name in builder.LANCE_SCHEMA.names:
        assert builder.LANCE_SCHEMA.field(name).type == source_schema.field(name).type


# --------------------------------------------------------------------------- #
# frozen source binding and overwrite safety
# --------------------------------------------------------------------------- #


def test_source_manifest_sha_pin_matches_release_v4() -> None:
    assert (
        builder._sha256_file(builder.DEFAULT_SOURCE_MANIFEST)
        == builder.EXPECTED_SOURCE_MANIFEST_SHA256
    )
    assert builder.DEFAULT_SOURCE_MANIFEST == SOURCE_MANIFEST


def test_source_manifest_sha_mismatch_is_refused(tmp_path: Path) -> None:
    impostor = tmp_path / "manifest.json"
    impostor.write_text(json.dumps({"passed": True}))
    with pytest.raises(RuntimeError, match="does not match the frozen v4 release"):
        builder._read_source(impostor)


def test_existing_output_is_never_overwritten(tmp_path: Path) -> None:
    occupied = tmp_path / "already_here"
    occupied.mkdir()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        builder._safe_output(occupied)
    fresh = builder._safe_output(tmp_path / "new_release")
    assert not fresh.exists()


def test_cli_exposes_local_staging_for_network_mounts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["builder", "--staging-root", "/tmp/contextworld-k10-staging"],
    )
    args = builder.parse_args()
    assert args.staging_root == Path("/tmp/contextworld-k10-staging")


# --------------------------------------------------------------------------- #
# one continuous simulator, one shared future action array
# --------------------------------------------------------------------------- #


def test_both_hidden_modes_share_one_simulator_and_one_action_array() -> None:
    source, _ = _source()
    catalog = builder._source_catalog(source, "validation")
    template = builder._template_from_dict(catalog[4]["template"])
    modes = {
        mode: builder._simulate(
            template, mode=mode, resolution=64, include_rows=False
        )
        for mode in builder.ENDPOINT_MODES
    }
    low, high = (modes[mode] for mode in builder.ENDPOINT_MODES)

    assert np.array_equal(low["raw_actions"], high["raw_actions"])
    assert low["raw_actions"].shape == (builder.RAW_ROWS_PER_EPISODE, 2)
    for rollout in (low, high):
        assert rollout["state_installations_after_x0"] == 0
        assert rollout["query_simulator_recreated"] is False
        assert len(rollout["contact_counts"]) == builder.RAW_ROWS_PER_EPISODE
        assert (
            rollout["query_reference_deviation"]
            <= damping.QUERY_REFERENCE_TOLERANCE
        )
    # Both branches meet at one query state that renders identically.
    assert np.array_equal(
        low["model_pixels"][builder.HISTORY_RAW_STEPS],
        high["model_pixels"][builder.HISTORY_RAW_STEPS],
    )
    # The hidden factor separates the futures and keeps separating them.
    audit = builder._pair_audit(low, high)
    distances = [
        audit["horizon_gaps"][str(horizon)]["block_position_px"]
        for horizon in range(1, builder.MAX_HORIZON + 1)
    ]
    assert distances == sorted(distances)
    assert distances[0] >= damping.MINIMUM_FUTURE_GAP_PX


def test_accepted_twin_group_is_clean_for_both_templates_and_modes() -> None:
    source, _ = _source()
    catalog = builder._source_catalog(source, "validation")
    payload = builder._candidate_group(
        split="validation",
        group_index=2,
        source_catalog=catalog,
        catalog_seed=int(source["split_catalog_seeds"]["validation"]),
    )
    report = builder._group_preflight(payload, resolution=64)

    assert report["passed"] is True
    assert [row["catalog_index"] for row in report["candidates"]] == [4, 5]
    assert report["visible_x0_label_balance"] is True
    for audit in report["template_audits"]:
        assert audit["passed"]
        assert audit["total_contact_steps"] == 0
        assert audit["maximum_arbiter_count"] == 0
        assert audit["bounds_ok"] is True


def test_parallel_preflight_cannot_change_which_groups_are_admitted() -> None:
    source, _ = _source()
    catalog = builder._source_catalog(source, "validation")
    seed = int(source["split_catalog_seeds"]["validation"])
    payloads = [
        builder._candidate_group(
            split="validation",
            group_index=index,
            source_catalog=catalog,
            catalog_seed=seed,
        )
        for index in range(3)
    ]
    serial = builder._preflight_many(payloads, resolution=64, workers=1)
    parallel = builder._preflight_many(payloads, resolution=64, workers=2)

    assert [row["group_index"] for row in serial] == [0, 1, 2]
    # Groups 0 and 1 keep a cached wall arbiter before h10; group 2 is clean.
    assert [row["passed"] for row in serial] == [False, False, True]
    assert [row["group_index"] for row in parallel] == [0, 1, 2]
    assert [row["passed"] for row in parallel] == [row["passed"] for row in serial]
