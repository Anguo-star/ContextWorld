from __future__ import annotations

from dataclasses import replace
import copy
import hashlib
from io import BytesIO
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import pytest

import scripts.probe_cube_grasp_rule_h3_v4_rgb_history as probe


def _jpeg(value: int) -> bytes:
    image = Image.new("RGB", (23, 19), (value, value + 1, value + 2))
    stream = BytesIO()
    image.save(stream, format="JPEG", quality=95)
    image.close()
    return stream.getvalue()


def _png(value: int) -> bytes:
    image = Image.new("RGB", (17, 21), (value, value, value))
    stream = BytesIO()
    image.save(stream, format="PNG")
    image.close()
    return stream.getvalue()


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_success_marker(root: Path) -> None:
    for name in ("request.json", "build_report.json", "manifest.json"):
        path = root / name
        if not path.exists():
            path.write_text(f"fixture:{name}\n", encoding="utf-8")
    receipts = probe._regular_release_file_receipts(root)
    by_path = {row["path"]: row for row in receipts}
    lance_tables: dict[str, dict[str, Any]] = {}
    for split, table_name in probe.TABLE_NAMES.items():
        prefix = f"{table_name}/"
        rows = [
            {
                "path": row["path"][len(prefix) :],
                "size_bytes": row["size_bytes"],
                "sha256": row["sha256"],
            }
            for row in receipts
            if row["path"].startswith(prefix)
        ]
        lance_tables[split] = {
            "table": table_name,
            "schema_equals_frozen_v4": True,
            "row_count": probe.EXPECTED_MODEL_ROWS[split],
            "file_count": len(rows),
            "size_bytes": sum(int(row["size_bytes"]) for row in rows),
            "tree_sha256": probe._tree_sha256_from_receipts(rows),
            "file_receipts_sha256": probe._canonical_json_sha256(rows),
            "passed": True,
        }
    payload = {
        "schema_version": 1,
        "protocol": probe.PROTOCOL,
        "recovery_authorization_id": probe.RECOVERY_AUTHORIZATION_ID,
        "status": "complete",
        "checks_passed": True,
        "public_test_opened": False,
        "public_test_generated": False,
        "publication": {
            "method": "verified_x_exclusive_copytree",
            "nonempty_directory_rename_used": False,
            "success_marker_written_last": True,
            "failed_copy_is_never_marked_complete": True,
            "source_and_destination_file_receipts_equal": True,
            "file_count_without_success_marker": len(receipts),
            "bytes_without_success_marker": sum(
                int(row["size_bytes"]) for row in receipts
            ),
            "tree_sha256_without_success_marker": (
                probe._tree_sha256_from_receipts(receipts)
            ),
            "file_receipts_sha256": probe._canonical_json_sha256(receipts),
        },
        "bound_files": {
            name: by_path[name]
            for name in ("request.json", "build_report.json", "manifest.json")
        },
        "lance_tables": lance_tables,
        "file_receipts_without_success_marker": receipts,
    }
    (root / probe.SUCCESS_MARKER_NAME).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _pair_id(split: str, pair_index: int) -> str:
    return f"cube-carry-v4r1-{split}-{pair_index:06d}"


def _fixture_rows(split: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    split_offset = 0 if split == "train" else 40
    for pair_index, anchor in enumerate(probe.ACTION_ANCHORS):
        pair_id = _pair_id(split, pair_index)
        catalog_index = probe.V4R1_FORMAL_CATALOG_INDEX_OFFSET + pair_index
        profile = probe.make_v4_action_profile(
            split=split, catalog_index=catalog_index
        )
        assert profile.action_anchor_id == anchor
        blocks = probe.frozen_v4_action_blocks(profile)
        profile_id = profile.action_profile_id
        scene_hash = _digest(f"scene:{split}:{pair_index}")
        pair_hash = probe.pair_content_sha256(scene_hash, profile_id)
        x0 = _jpeg(90 + split_offset + pair_index)
        query = _jpeg(100 + split_offset + pair_index)
        for mode in probe.HIDDEN_MODES:
            history = _jpeg(25 if mode == "cannot_hold" else 225)
            # The future deliberately carries a very strong label shortcut.
            # The frozen probe must never decode or use it.
            future = _jpeg(5 if mode == "cannot_hold" else 245)
            frames = (x0, history, query, future)
            for step in probe.MODEL_STEPS:
                rows.append(
                    {
                        "model_step_idx": step,
                        "pixels": frames[step],
                        "action_block": blocks[step].reshape(-1).tolist(),
                        "hidden_grasp_enabled": [
                            float(probe.LABEL_ENCODING[mode])
                        ],
                        "pair_id": pair_id,
                        "hidden_mode": mode,
                        "split": split,
                        "catalog_index": catalog_index,
                        "source_episode": split_offset + pair_index,
                        "action_anchor_id": anchor,
                        "action_profile_id": profile_id,
                        "scene_template_content_hash": scene_hash,
                        "pair_content_hash": pair_hash,
                    }
                )
    order = np.random.default_rng(71).permutation(len(rows))
    return [rows[int(index)] for index in order]


def _condition_rows(
    rows: list[dict[str, Any]], pair_id: str, mode: str
) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row["pair_id"] == pair_id and row["hidden_mode"] == mode
    ]


def _fixture_authorization() -> dict[str, Any]:
    return {
        "identities": {},
        "prior_sets": {
            name: frozenset()
            for name in (
                "source_episodes",
                "action_profile_ids",
                "scene_template_content_hashes",
                "pair_content_hashes",
                "query_pixel_hashes",
            )
        },
    }


def _stub_run_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    metadata = {
        "raw": {name: name.encode("ascii") for name in probe.METADATA_FILE_NAMES},
        "identities": {
            name: probe._raw_identity(name.encode("ascii"))
            for name in probe.METADATA_FILE_NAMES
        },
    }
    monkeypatch.setattr(
        probe,
        "validate_release_metadata",
        lambda _root, *, marker, authorization: metadata,
    )
    monkeypatch.setattr(
        probe, "_reverify_authorization_inputs", lambda _authorization: None
    )


def _main_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    output_name: str,
) -> tuple[Path, Path, list[str]]:
    artifact = tmp_path / "artifact"
    artifact.mkdir(exist_ok=True)
    prereg = tmp_path / "prereg.yaml"
    freeze = tmp_path / "freeze.json"
    prior = tmp_path / "prior.json"
    output = tmp_path / output_name
    monkeypatch.setattr(probe, "CANONICAL_ARTIFACT_ROOT", artifact)
    monkeypatch.setattr(probe, "CANONICAL_PREREG_PATH", prereg)
    monkeypatch.setattr(probe, "CANONICAL_FREEZE_RECEIPT_PATH", freeze)
    monkeypatch.setattr(probe, "CANONICAL_PRIOR_EXCLUSION_PATH", prior)
    monkeypatch.setattr(probe, "CANONICAL_OUTPUT_PATH", output)
    argv = [
        "--artifact-root",
        str(artifact),
        "--prereg",
        str(prereg),
        "--freeze-receipt",
        str(freeze),
        "--prior-exclusion-receipt",
        str(prior),
        "--output",
        str(output),
    ]
    return artifact, output, argv


def _valid_build_report() -> dict[str, Any]:
    overlap_names = (
        "query_pixel_hash_overlap",
        "source_episode_overlap",
        "exact_action_profile_id_overlap",
        "scene_template_content_hash_overlap",
        "pair_content_hash_overlap",
    )
    splits = {}
    for split in probe.ACTIVE_SPLITS:
        count = probe.EXPECTED_PAIR_COUNTS[split]
        splits[split] = {
            "passed": True,
            "pair_count": count,
            "episode_count": 2 * count,
            "model_rows": probe.EXPECTED_MODEL_ROWS[split],
            "table_path": probe.TABLE_NAMES[split],
            "all_causal_checks_passed": True,
            "unique_action_profile_count": count,
            "unique_scene_template_content_hash_count": count,
            "unique_pair_content_hash_count": count,
            "action_anchor_counts": {
                anchor: count // 4 for anchor in probe.ACTION_ANCHORS
            },
            "action_anchor_expected_count_each": count // 4,
            "fresh_simulator_replay": {
                "passed": True,
                "pair_count": count,
                "mode_replay_count": 2 * count,
                "query_gap_used_as_replay_substitute": False,
            },
            "prior_episode_and_content_exclusion": {
                "passed": True,
                "candidate_catalog_source_episode_overlap_count": 0,
                "accepted_overlap": {
                    "source_episode_count": 0,
                    "action_profile_id_count": 0,
                    "scene_template_content_hash_count": 0,
                    "pair_content_hash_count": 0,
                    "query_pixel_hash_count": 0,
                },
            },
        }
    return {
        "passed": True,
        "source_h5_post_build_integrity": {
            "passed": True,
            "expected_sha256": "a" * 64,
            "observed_sha256": "a" * 64,
        },
        "cross_split_audit": {
            "passed": True,
            **{name: {"count": 0, "values": []} for name in overlap_names},
        },
        "fresh_simulator_replay": {
            "passed": True,
            "pair_count": sum(probe.EXPECTED_PAIR_COUNTS.values()),
            "mode_replay_count": 2 * sum(probe.EXPECTED_PAIR_COUNTS.values()),
            "query_gap_used_as_replay_substitute": False,
        },
        "causal_data_contract": {"passed": True},
        "splits": splits,
    }


def test_frozen_probe_contract_is_exact_and_development_only() -> None:
    assert probe.PROTOCOL == "cube_gripper_carry_rule_history3_development_v4"
    assert (
        probe.RECOVERY_AUTHORIZATION_ID
        == "cube_gripper_carry_h3_development_v4r1"
    )
    assert probe.PROBE_ID == "cube_gripper_carry_h3_v4r1_rgb_history_probe_v1"
    assert probe.SUCCESS_MARKER_NAME == "_SUCCESS.json"
    assert probe.ACTIVE_SPLITS == ("train", "loader_validation")
    assert probe.EXPECTED_PAIR_COUNTS == {
        "train": 2048,
        "loader_validation": 256,
    }
    assert probe.EXPECTED_MODEL_ROWS == {
        "train": 16384,
        "loader_validation": 2048,
    }
    assert probe.TABLE_NAMES == {
        "train": "train.lance",
        "loader_validation": "loader_validation.lance",
    }
    assert probe.LABEL_ENCODING == {"cannot_hold": 0, "can_hold": 1}
    assert probe.BOOTSTRAP_RESAMPLES == 10_000
    assert probe.BOOTSTRAP_SEED == 2026081203
    assert probe.BOOTSTRAP_LOWER_QUANTILE == 0.025
    assert probe.PERMUTATION_REPETITIONS == 16
    assert probe.PERMUTATION_SEED == 2026081204
    assert probe.OVERALL_ACCURACY_MINIMUM == 0.75
    assert probe.WORST_MODE_ACCURACY_MINIMUM == 0.70
    assert probe.WORST_ANCHOR_ACCURACY_MINIMUM == 0.70
    assert probe.BOOTSTRAP_LOWER_BOUND_MINIMUM == 0.70
    assert probe.PERMUTATION_MEAN_ACCURACY_MAXIMUM == 0.60
    assert probe.SHORTCUT_ACCURACY_MAXIMUM == 0.51
    assert probe.RESIZE_SHAPE == (16, 16)
    assert probe.MAIN_FEATURE_COLUMNS == ("pixels",)
    assert probe.NEGATIVE_CONTROL_ONLY_COLUMNS == ("action_block",)
    assert probe.METADATA_ACTION_COLUMNS == tuple(
        name for name in probe.TABLE_COLUMNS if name != "pixels"
    )
    assert probe.PIXEL_FILTER == "model_step_idx <= 2"
    assert probe.PIXEL_JOIN_COLUMNS == (
        "pair_id",
        "hidden_mode",
        "split",
        "model_step_idx",
        "pixels",
    )
    assert not (
        set(probe.MAIN_FEATURE_COLUMNS)
        & set(probe.PRIVILEGED_COLUMNS_EXCLUDED_FROM_MAIN_FEATURE)
    )
    assert "physical_state" not in probe.TABLE_COLUMNS
    assert "hidden_grasp_enabled" in probe.TABLE_COLUMNS
    assert "hidden_grasp_enabled" in probe.AUDIT_ONLY_COLUMNS
    assert "hidden_grasp_enabled" not in probe.MAIN_FEATURE_COLUMNS
    assert "hidden_grasp_enabled" not in probe.NEGATIVE_CONTROL_ONLY_COLUMNS
    assert {"episode_idx", "model_step_idx", "source_step"} <= set(
        probe.PRIVILEGED_COLUMNS_EXCLUDED_FROM_MAIN_FEATURE
    )


def test_v4_does_not_adopt_exploratory_feature_changes() -> None:
    source = Path(probe.__file__).read_text(encoding="utf-8")
    assert "signed_temporal_deltas" not in source
    assert "absolute_temporal_deltas" not in source
    assert "training_localized_roi" not in source
    assert "full32" not in source
    assert "2.0 * frames[1] - frames[0] - frames[2]" in source


def test_pillow_decoder_and_float64_c_order_feature_are_frozen() -> None:
    decoded = probe._decode_rgb_frame(_jpeg(80))
    assert decoded.shape == (16, 16, 3)
    assert decoded.dtype == np.float64
    assert decoded.flags.c_contiguous
    with pytest.raises(ValueError, match="JPEG container"):
        probe._decode_rgb_frame(_png(80))

    x0 = np.arange(16 * 16 * 3, dtype=np.float64).reshape(16, 16, 3)
    x1 = x0 + 3.0
    x2 = x0 - 2.0
    feature = probe.rgb_history_feature(x0, x1, x2)
    expected = (2.0 * x1 - x0 - x2).flatten(order="C")
    assert feature.dtype == np.float64
    assert feature.flags.c_contiguous
    assert np.array_equal(feature, expected)


def test_prepare_split_groups_shuffled_four_row_conditions_and_ignores_x3() -> None:
    rows = _fixture_rows("train")
    prepared = probe.prepare_split(rows, expected_split="train")
    assert prepared.row_count == 32
    assert prepared.pair_count == 4
    assert prepared.condition_count == 8
    assert prepared.main_features.shape == (8, 768)
    assert prepared.x0_features.shape == (8, 768)
    assert prepared.query_features.shape == (8, 768)
    assert prepared.action_features.shape == (8, 100)
    assert prepared.labels.tolist() == [0, 1] * 4
    assert prepared.anchor_pair_counts == {
        anchor: 1 for anchor in probe.ACTION_ANCHORS
    }

    changed_future = [dict(row) for row in rows]
    for row in changed_future:
        if row["model_step_idx"] == 3:
            row["pixels"] = b"not-even-an-image"
    replay = probe.prepare_split(changed_future, expected_split="train")
    assert np.array_equal(prepared.main_features, replay.main_features)
    assert np.array_equal(prepared.x0_features, replay.x0_features)
    assert np.array_equal(prepared.query_features, replay.query_features)


@pytest.mark.parametrize(
    ("step", "message"),
    ((0, "paired x0"), (2, "paired query/x2")),
)
def test_prepare_split_rejects_paired_pixel_mismatch(
    step: int, message: str
) -> None:
    rows = _fixture_rows("train")
    pair_id = _pair_id("train", 0)
    target = next(
        row
        for row in rows
        if row["pair_id"] == pair_id
        and row["hidden_mode"] == "can_hold"
        and row["model_step_idx"] == step
    )
    target["pixels"] = _jpeg(211)
    with pytest.raises(ValueError, match=message):
        probe.prepare_split(rows, expected_split="train")


def test_prepare_split_rejects_incomplete_condition_and_action_hash_tamper() -> None:
    rows = _fixture_rows("train")
    incomplete = rows[:-1]
    with pytest.raises(ValueError, match="expected exactly model_step_idx"):
        probe.prepare_split(incomplete, expected_split="train")

    rows = _fixture_rows("train")
    condition = _condition_rows(rows, _pair_id("train", 0), "can_hold")
    for row in condition:
        row["action_profile_id"] = "0" * 64
        row["pair_content_hash"] = probe.pair_content_sha256(
            row["scene_template_content_hash"], row["action_profile_id"]
        )
    with pytest.raises(ValueError, match="does not match actual float32 actions"):
        probe.prepare_split(rows, expected_split="train")


def test_prepare_split_recomputes_pair_content_hash() -> None:
    rows = _fixture_rows("loader_validation")
    condition = _condition_rows(
        rows, _pair_id("loader_validation", 0), "cannot_hold"
    )
    for row in condition:
        row["pair_content_hash"] = "f" * 64
    with pytest.raises(ValueError, match="does not bind scene/profile"):
        probe.prepare_split(rows, expected_split="loader_validation")


def test_prepare_split_rejects_catalog_pair_and_hidden_label_tampering() -> None:
    rows = _fixture_rows("train")
    condition = _condition_rows(rows, _pair_id("train", 0), "cannot_hold")
    condition[0]["pair_id"] = "forged-pair"
    with pytest.raises(ValueError, match="pair_id does not match"):
        probe.prepare_split(rows, expected_split="train")

    rows = _fixture_rows("train")
    condition = _condition_rows(rows, _pair_id("train", 0), "cannot_hold")
    condition[0]["hidden_grasp_enabled"] = [1.0]
    with pytest.raises(ValueError, match="does not match hidden_mode"):
        probe.prepare_split(rows, expected_split="train")

    rows = _fixture_rows("train")
    condition = _condition_rows(rows, _pair_id("train", 0), "cannot_hold")
    for row in condition:
        row["catalog_index"] += 4
    with pytest.raises(
        ValueError, match="pair_id does not match|profile ID does not match"
    ):
        probe.prepare_split(rows, expected_split="train")


def test_cross_split_content_gate_uses_content_not_pair_id() -> None:
    train = probe.prepare_split(_fixture_rows("train"), expected_split="train")
    development = probe.prepare_split(
        _fixture_rows("loader_validation"),
        expected_split="loader_validation",
    )
    audit = probe.cross_split_content_audit(train, development)
    assert audit["passed"]
    assert audit["pair_id_is_content_isolation_evidence"] is False
    assert audit["evidence_source"]["manifest_read"] is False



@pytest.mark.parametrize(
    ("attribute", "check", "report_key"),
    (
        (
            "action_profile_ids",
            "exact_action_profile_id_overlap_zero",
            "exact_action_profile_id_overlap",
        ),
        (
            "scene_template_content_hashes",
            "scene_template_content_hash_overlap_zero",
            "scene_template_content_hash_overlap",
        ),
        (
            "pair_content_hashes",
            "pair_content_hash_overlap_zero",
            "pair_content_hash_overlap",
        ),
    ),
)
def test_each_cross_split_content_overlap_fails_its_gate(
    attribute: str, check: str, report_key: str
) -> None:
    train = probe.prepare_split(_fixture_rows("train"), expected_split="train")
    development = probe.prepare_split(
        _fixture_rows("loader_validation"),
        expected_split="loader_validation",
    )
    leaked_values = frozenset(
        set(getattr(development, attribute))
        | {next(iter(getattr(train, attribute)))}
    )
    leaked = replace(development, **{attribute: leaked_values})
    failed = probe.cross_split_content_audit(train, leaked)
    assert not failed["passed"]
    assert not failed["checks"][check]
    assert failed[report_key]["count"] == 1


@pytest.mark.parametrize(
    "attribute",
    ("source_episodes", "query_pixel_hashes"),
)
def test_source_and_raw_query_cross_split_overlap_fail(attribute: str) -> None:
    train = probe.prepare_split(_fixture_rows("train"), expected_split="train")
    development = probe.prepare_split(
        _fixture_rows("loader_validation"), expected_split="loader_validation"
    )
    leaked = replace(
        development,
        **{
            attribute: frozenset(
                {*getattr(development, attribute), next(iter(getattr(train, attribute)))}
            )
        },
    )
    assert not probe.cross_split_content_audit(train, leaked)["passed"]


@pytest.mark.parametrize(
    "field_name",
    (
        "source_episodes",
        "action_profile_ids",
        "scene_template_content_hashes",
        "pair_content_hashes",
        "query_pixel_hashes",
    ),
)
def test_prior_exclusion_audit_rejects_each_identity_class(field_name: str) -> None:
    train = probe.prepare_split(_fixture_rows("train"), expected_split="train")
    development = probe.prepare_split(
        _fixture_rows("loader_validation"), expected_split="loader_validation"
    )
    prior = {
        "source_episodes": frozenset(),
        "action_profile_ids": frozenset(),
        "scene_template_content_hashes": frozenset(),
        "pair_content_hashes": frozenset(),
        "query_pixel_hashes": frozenset(),
    }
    prior[field_name] = frozenset({next(iter(getattr(train, field_name)))})
    with pytest.raises(ValueError, match="overlap frozen prior exclusions"):
        probe.prior_exclusion_audit(train, development, prior_sets=prior)


def test_pair_cluster_bootstrap_is_anchor_stratified_and_deterministic() -> None:
    development = probe.prepare_split(
        _fixture_rows("loader_validation"),
        expected_split="loader_validation",
    )
    predictions = development.labels.copy()
    first_pair = development.pair_ids[0]
    predictions[development.pair_ids == first_pair] ^= 1
    first = probe.stratified_pair_cluster_bootstrap(
        development.labels,
        predictions,
        development.pair_ids,
        development.action_anchors,
        resamples=500,
        seed=19,
    )
    second = probe.stratified_pair_cluster_bootstrap(
        development.labels,
        predictions,
        development.pair_ids,
        development.action_anchors,
        resamples=500,
        seed=19,
    )
    assert first == second
    assert first["unit"] == "pair_cluster"
    assert first["stratification"] == "action_anchor_id"
    assert first["stratum_pair_counts"] == {
        anchor: 1 for anchor in probe.ACTION_ANCHORS
    }
    assert first["overall_accuracy"] == 0.75
    assert first["lower_bound_2_5_percent"] == 0.75


def test_full_fixture_probe_passes_all_frozen_gates() -> None:
    report = probe.evaluate_fixture_rows(
        _fixture_rows("train"),
        _fixture_rows("loader_validation"),
    )
    assert report["status"] == "passed"
    assert report["passed"]
    assert (
        report["recovery_authorization_id"]
        == probe.RECOVERY_AUTHORIZATION_ID
    )
    assert all(report["gates"].values())

    metrics = report["primary_probe"]["metrics"]
    assert metrics["overall_accuracy"] == 1.0
    assert metrics["worst_mode"]["accuracy"] == 1.0
    assert metrics["worst_anchor_family"]["accuracy"] == 1.0
    bootstrap = report["pair_cluster_anchor_stratified_bootstrap"]
    assert bootstrap["resamples"] == 10_000
    assert bootstrap["seed"] == 2026081203
    assert bootstrap["lower_bound_2_5_percent"] == 1.0

    controls = report["negative_controls"]
    assert controls["label_permutation"]["repetitions"] == 16
    assert controls["label_permutation"]["seed"] == 2026081204
    assert controls["label_permutation"]["mean_accuracy"] <= 0.60
    assert controls["x0_only"]["accuracy"] == 0.5
    assert controls["query_x2_only"]["accuracy"] == 0.5
    assert controls["action_only"]["accuracy"] == 0.5
    assert (
        report["fit_contract"]["primary_fit_receipt"]["standard_scaler"][
            "fit_split"
        ]
        == "train"
    )
    for name in ("x0_only", "query_x2_only", "action_only"):
        assert controls[name]["fit_receipt"]["standard_scaler"]["fit_split"] == (
            "train"
        )
    contract = report["decoder_and_feature_contract"]
    assert contract["x3_decoded_or_used"] is False
    assert contract["x3_pixel_bytes_not_read_from_lance"] is True
    assert contract["fixed_main_feature"] == "flatten(2*x1-x0-x2)_C_order"
    assert contract["arithmetic_dtype"] == "float64"
    assert contract["ids_labels_metadata_and_row_order_used_as_main_feature"] is False
    assert set(report["package_versions"]) >= {
        "python",
        "numpy",
        "Pillow",
        "scikit-learn",
        "scipy",
        "lance",
        "pyarrow",
    }


def test_allowed_table_resolver_rejects_public_or_extra_lance_tables(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifact"
    root.mkdir()
    (root / "train.lance").mkdir()
    (root / "loader_validation.lance").mkdir()
    resolved, tables = probe.resolve_allowed_tables(root)
    assert resolved == root.resolve()
    assert set(tables) == set(probe.ACTIVE_SPLITS)

    (root / "validation.lance").mkdir()
    with pytest.raises(ValueError, match="validation/Public"):
        probe.resolve_allowed_tables(root)
    with pytest.raises(ValueError, match="validation/Public path"):
        probe.resolve_allowed_tables(tmp_path / "validation" / "artifact")


def test_resolver_rejects_nested_public_before_any_content_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "artifact"
    (root / "train.lance").mkdir(parents=True)
    (root / "loader_validation.lance").mkdir()
    (root / "train.lance" / "Public").mkdir()
    read_called = False

    def forbidden_read(*_args: Any, **_kwargs: Any) -> bytes:
        nonlocal read_called
        read_called = True
        raise AssertionError("content read must not occur")

    monkeypatch.setattr(probe, "_read_bytes_nofollow", forbidden_read)
    with pytest.raises(ValueError, match="validation/Public component"):
        probe.resolve_allowed_tables(root)
    assert not read_called


@pytest.mark.parametrize(
    "option",
    ("--validation", "--validation-lance", "--public-test", "--test-table"),
)
def test_cli_explicitly_refuses_closed_split_options(option: str) -> None:
    with pytest.raises(ValueError, match="explicitly refuses"):
        probe.parse_args(
            [
                "--artifact-root",
                "/unused",
                "--output",
                "/unused.json",
                option,
                "/closed",
            ]
        )


def test_lance_reader_uses_split_projection_and_never_reads_x3_pixels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_rows = _fixture_rows("train")
    calls: list[tuple[tuple[str, ...], str | None]] = []

    class _Table:
        def __init__(self, rows: list[dict[str, Any]]) -> None:
            self._rows = rows

        def to_pylist(self) -> list[dict[str, Any]]:
            return self._rows

    class _Dataset:
        schema = probe.FROZEN_ARROW_SCHEMA

        def count_rows(self) -> int:
            return len(source_rows)

        def to_table(
            self,
            *,
            columns: list[str],
            filter: str | None = None,
        ) -> _Table:
            calls.append((tuple(columns), filter))
            selected = source_rows
            if filter is not None:
                assert filter == "model_step_idx <= 2"
                selected = [
                    row for row in source_rows if row["model_step_idx"] <= 2
                ]
            return _Table(
                [{name: row[name] for name in columns} for row in selected]
            )

    monkeypatch.setattr(probe.lance, "dataset", lambda _path: _Dataset())
    monkeypatch.setattr(
        probe,
        "EXPECTED_MODEL_ROWS",
        {"train": len(source_rows), "loader_validation": len(source_rows)},
    )
    merged = probe._read_lance_rows(
        tmp_path / "train.lance",
        expected_split="train",
    )
    assert calls == [
        (tuple(probe.METADATA_ACTION_COLUMNS), None),
        (tuple(probe.PIXEL_JOIN_COLUMNS), "model_step_idx <= 2"),
    ]
    assert len(merged) == len(source_rows)
    assert all(
        ("pixels" in row) == (row["model_step_idx"] <= 2) for row in merged
    )
    assert all(
        "pixels" not in row for row in merged if row["model_step_idx"] == 3
    )


def test_lance_reader_rejects_validation_before_opening(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened = False

    def forbidden_open(_path: str) -> None:
        nonlocal opened
        opened = True
        raise AssertionError("closed table was opened")

    monkeypatch.setattr(probe.lance, "dataset", forbidden_open)
    with pytest.raises(ValueError, match="validation/Public path"):
        probe._read_lance_rows(
            tmp_path / "validation.lance",
            expected_split="validation",
        )
    assert not opened


def test_lance_reader_rejects_schema_with_same_names_wrong_types(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wrong = probe.FROZEN_ARROW_SCHEMA.set(
        0, probe.pa.field("episode_idx", probe.pa.int64())
    )

    class Dataset:
        schema = wrong

    monkeypatch.setattr(probe.lance, "dataset", lambda _path: Dataset())
    with pytest.raises(ValueError, match="Arrow schema differs"):
        probe._read_lance_rows(tmp_path / "train.lance", expected_split="train")


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (
            lambda report: report["source_h5_post_build_integrity"].__setitem__(
                "observed_sha256", "b" * 64
            ),
            "source H5 post-build integrity",
        ),
        (
            lambda report: report["cross_split_audit"][
                "source_episode_overlap"
            ].__setitem__("count", 1),
            "cross-split isolation",
        ),
        (
            lambda report: report["fresh_simulator_replay"].__setitem__(
                "pair_count", 1
            ),
            "aggregate fresh replay",
        ),
        (
            lambda report: report["causal_data_contract"].__setitem__(
                "passed", False
            ),
            "causal data contract",
        ),
        (
            lambda report: report["splits"]["train"][
                "prior_episode_and_content_exclusion"
            ]["accepted_overlap"].__setitem__("query_pixel_hash_count", 1),
            "train identity mismatch",
        ),
        (
            lambda report: report["splits"]["loader_validation"][
                "action_anchor_counts"
            ].__setitem__("endpoint4", 0),
            "loader_validation identity mismatch",
        ),
    ),
)
def test_release_metadata_rejects_critical_build_gate_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutate: Any,
    message: str,
) -> None:
    build = _valid_build_report()
    mutate(build)
    request = {
        "protocol": probe.PROTOCOL,
        "recovery_authorization_id": probe.RECOVERY_AUTHORIZATION_ID,
        "active_splits": list(probe.ACTIVE_SPLITS),
        "public_test_opened": False,
        "public_test_generated": False,
        "pair_counts": dict(probe.EXPECTED_PAIR_COUNTS),
        "workers": 16,
        "jpeg_quality": 95,
        "freeze_receipt": {
            "sha256": "1" * 64,
            "size_bytes": 1,
            "status": probe.FREEZE_STATUS,
            "checks_passed": True,
        },
        "prior_episode_exclusion_receipt": {
            "sha256": "2" * 64,
            "size_bytes": 1,
            "status": probe.FREEZE_STATUS,
            "checks_passed": True,
        },
    }
    build.update(
        {
            "protocol": probe.PROTOCOL,
            "recovery_authorization_id": probe.RECOVERY_AUTHORIZATION_ID,
            "active_splits": list(probe.ACTIVE_SPLITS),
            "public_test_opened": False,
            "public_test_generated": False,
            "request": request,
        }
    )
    docs = {
        "request.json": request,
        "build_report.json": build,
        "manifest.json": {
            "protocol": probe.PROTOCOL,
            "recovery_authorization_id": probe.RECOVERY_AUTHORIZATION_ID,
            "active_splits": list(probe.ACTIVE_SPLITS),
            "public_test_opened": False,
            "public_test_generated": False,
        },
    }
    raw = {
        name: (json.dumps(value) + "\n").encode("utf-8")
        for name, value in docs.items()
    }
    monkeypatch.setattr(
        probe,
        "_read_release_file_nofollow",
        lambda _root, name: raw[name],
    )
    marker = {
        "payload": {
            "bound_files": {
                name: probe._raw_identity(value) for name, value in raw.items()
            }
        }
    }
    authorization = {
        "identities": {
            "freeze_receipt": {"sha256": "1" * 64, "size_bytes": 1},
            "prior_exclusion_receipt": {"sha256": "2" * 64, "size_bytes": 1},
        },
        "documents": {
            "prior_exclusion_receipt": {
                "excluded_source_episode_count": 0,
                "excluded_source_episodes_sha256": "3" * 64,
            }
        },
    }
    with pytest.raises(ValueError, match=message):
        probe.validate_release_metadata(
            tmp_path, marker=marker, authorization=authorization
        )


def test_run_probe_reads_only_the_two_whitelisted_tables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    monkeypatch.setattr(
        probe,
        "EXPECTED_MODEL_ROWS",
        {"train": 32, "loader_validation": 32},
    )
    for table_name in probe.TABLE_NAMES.values():
        table = artifact / table_name
        table.mkdir()
        (table / "fixture.bin").write_bytes(table_name.encode("ascii"))
    _write_success_marker(artifact)
    opened: list[tuple[str, str]] = []

    _stub_run_contract(monkeypatch)

    def fixture_reader(
        path: Path, *, expected_split: str, fd_anchored: bool
    ) -> list[dict[str, Any]]:
        assert path.parent == Path("/proc/self/fd")
        assert fd_anchored is True
        opened.append((probe.TABLE_NAMES[expected_split], expected_split))
        return _fixture_rows(expected_split)

    monkeypatch.setattr(probe, "_read_lance_rows", fixture_reader)
    report = probe.run_probe(artifact, authorization=_fixture_authorization())
    assert opened == [
        ("train.lance", "train"),
        ("loader_validation.lance", "loader_validation"),
    ]
    assert report["inputs"]["only_authorized_lance_tables_opened"] == [
        "train.lance",
        "loader_validation.lance",
    ]
    assert report["inputs"]["manifest_or_build_report_parsed"] is True
    assert report["inputs"]["manifest_and_build_report_bytes_hashed"] is True
    assert (
        report["inputs"]["completed_publication_verified_before_lance_open"]
        is True
    )
    assert report["inputs"]["success_marker"]["checks_passed"] is True
    assert report["inputs"]["release_identity_unchanged_during_reads"] is True
    assert (
        report["inputs"]["success_marker_preflight"]
        == report["inputs"]["success_marker_postflight"]
    )
    assert report["inputs"]["validation_or_public_table_read"] is False
    for split in probe.ACTIVE_SPLITS:
        table = report["inputs"]["tables"][split]
        assert table["table_directory_hashed"] is True
        assert "directory_sha256" not in table
        assert table["x3_pixel_bytes_read"] is False
        assert table["projections"][0] == {
            "columns": list(probe.METADATA_ACTION_COLUMNS),
            "filter": None,
            "row_scope": "all_four_model_steps",
        }
        assert table["projections"][1] == {
            "columns": list(probe.PIXEL_JOIN_COLUMNS),
            "filter": "model_step_idx <= 2",
            "row_scope": "x0_x1_x2_only",
        }


def test_run_probe_requires_untampered_success_marker_before_lance_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    monkeypatch.setattr(
        probe,
        "EXPECTED_MODEL_ROWS",
        {"train": 32, "loader_validation": 32},
    )
    for table_name in probe.TABLE_NAMES.values():
        table = artifact / table_name
        table.mkdir()
        (table / "fixture.bin").write_bytes(table_name.encode("ascii"))

    opened = False

    def forbidden_reader(
        _path: Path, *, expected_split: str, **_kwargs: Any
    ) -> list[dict[str, Any]]:
        del expected_split
        nonlocal opened
        opened = True
        raise AssertionError("Lance must remain unopened")

    monkeypatch.setattr(probe, "_read_lance_rows", forbidden_reader)
    _stub_run_contract(monkeypatch)
    with pytest.raises(ValueError, match="artifact root must contain exactly"):
        probe.run_probe(artifact, authorization=_fixture_authorization())
    assert not opened

    _write_success_marker(artifact)
    (artifact / "train.lance" / "fixture.bin").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="identities differ"):
        probe.run_probe(artifact, authorization=_fixture_authorization())
    assert not opened


def test_run_probe_rejects_release_mutation_during_lance_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    monkeypatch.setattr(
        probe,
        "EXPECTED_MODEL_ROWS",
        {"train": 32, "loader_validation": 32},
    )
    for table_name in probe.TABLE_NAMES.values():
        table = artifact / table_name
        table.mkdir()
        (table / "fixture.bin").write_bytes(table_name.encode("ascii"))
    _write_success_marker(artifact)
    calls = 0

    def mutating_reader(
        _path: Path, *, expected_split: str, **_kwargs: Any
    ) -> list[dict[str, Any]]:
        nonlocal calls
        calls += 1
        if calls == 1:
            (artifact / "train.lance" / "fixture.bin").write_bytes(b"mutated")
        return _fixture_rows(expected_split)

    monkeypatch.setattr(probe, "_read_lance_rows", mutating_reader)
    _stub_run_contract(monkeypatch)
    with pytest.raises(RuntimeError, match="fd-anchored Lance identity mismatch"):
        probe.run_probe(artifact, authorization=_fixture_authorization())
    assert calls == 1


def test_main_refuses_overwrite_before_probe_reads_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _artifact, output, argv = _main_fixture(
        tmp_path, monkeypatch, output_name="existing.json"
    )
    output.write_text("do not replace\n", encoding="utf-8")
    called = False

    def forbidden_probe(_root: Path, *, authorization: Any) -> dict[str, Any]:
        del authorization
        nonlocal called
        called = True
        raise AssertionError("probe should not run")

    monkeypatch.setattr(probe, "run_probe", forbidden_probe)
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        probe.main(argv)
    assert not called
    assert output.read_text(encoding="utf-8") == "do not replace\n"


def test_main_writes_json_exclusively_without_touching_fixture_tables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact, output, argv = _main_fixture(
        tmp_path, monkeypatch, output_name="probe.json"
    )
    (artifact / "train.lance").mkdir()
    (artifact / "loader_validation.lance").mkdir()
    payload = {"status": "passed", "passed": True}
    monkeypatch.setattr(
        probe, "validate_authorization_chain", lambda **_kwargs: {"ok": True}
    )
    monkeypatch.setattr(
        probe,
        "run_probe",
        lambda _root, *, authorization: dict(payload),
    )
    assert probe.main(argv) == 0
    assert json.loads(output.read_text(encoding="utf-8")) == payload
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        probe.main(argv)


def test_main_persists_complete_scientific_failure_before_exit_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact, output, argv = _main_fixture(
        tmp_path, monkeypatch, output_name="failed_probe.json"
    )
    (artifact / "train.lance").mkdir()
    (artifact / "loader_validation.lance").mkdir()
    payload = {
        "schema_version": 1,
        "probe_id": probe.PROBE_ID,
        "protocol": probe.PROTOCOL,
        "status": "failed",
        "public_test": {
            "access_status": "closed_not_read_not_scored",
            "opened": False,
            "read": False,
            "hashed": False,
            "scored": False,
        },
        "decoder_and_feature_contract": {
            "decoded_steps": [0, 1, 2],
            "x3_decoded_or_used": False,
            "x3_pixel_bytes_not_read_from_lance": True,
            "fixed_main_feature": "flatten(2*x1-x0-x2)_C_order",
        },
        "gates": {"overall_accuracy_at_least_0_75": False},
        "passed": False,
    }
    monkeypatch.setattr(
        probe, "validate_authorization_chain", lambda **_kwargs: {"ok": True}
    )
    monkeypatch.setattr(
        probe,
        "run_probe",
        lambda _root, *, authorization: dict(payload),
    )
    assert probe.main(argv) == 1
    assert json.loads(output.read_text(encoding="utf-8")) == payload
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        probe.main(argv)


def test_exclusive_output_rolls_back_failed_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "probe.json"
    original_fsync = probe.os.fsync
    calls = 0

    def failing_fsync(descriptor: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("fixture fsync failure")
        original_fsync(descriptor)

    monkeypatch.setattr(probe.os, "fsync", failing_fsync)
    with pytest.raises(OSError, match="fixture fsync failure"):
        probe._write_json_exclusive(output, {"passed": True})
    assert not output.exists()


def test_authorization_chain_rejects_prereg_freeze_identity_divergence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prereg = {
        "schema_version": 1,
        "protocol_id": probe.PROTOCOL,
        "recovery_authorization_id": probe.RECOVERY_AUTHORIZATION_ID,
        "status": probe.PREREG_STATUS,
        "phase": "development_only",
        "public_test": {
            "access_status": "closed_not_read_not_scored",
            "validation_lance_access_allowed": False,
            "opened": False,
            "read": False,
            "hashed": False,
            "scored": False,
        },
        "reference_model_training_or_scoring_authorized": False,
        "reference_model_phase": {
            "training_and_scoring_authorized": False,
            "trainer_invoked": False,
            "optimizer_steps_authorized": 0,
            "optimizer_steps_run": 0,
            "checkpoint_creation_authorized": False,
        },
        "identity": {"declared": {"sha256": "0" * 64, "size_bytes": 1}},
    }
    freeze = {
        "schema_version": 1,
        "protocol_id": probe.PROTOCOL,
        "recovery_authorization_id": probe.RECOVERY_AUTHORIZATION_ID,
        "status": probe.FREEZE_STATUS,
        "checks_passed": True,
        "authorized_splits": list(probe.ACTIVE_SPLITS),
        "recovery_build_attempts_authorized": 1,
        "rgb_history_probe_attempts_authorized": 1,
        "public_test": {
            "access_status": "closed_not_read_not_scored",
            "opened": False,
            "read": False,
            "hashed": False,
            "scored": False,
        },
        "reference_model_training_or_scoring_authorized": False,
        "reference_model_optimizer_steps_authorized": 0,
        "identity": {"frozen": {"sha256": "1" * 64, "size_bytes": 1}},
    }
    prior = {
        "schema_version": 1,
        "protocol_id": probe.PROTOCOL,
        "recovery_authorization_id": probe.RECOVERY_AUTHORIZATION_ID,
        "receipt_id": probe.PRIOR_RECEIPT_ID,
        "status": probe.FREEZE_STATUS,
        "checks_passed": True,
        "public_test": {
            "access_status": "closed_not_read_not_scored",
            "opened": False,
            "read": False,
            "hashed": False,
            "scored": False,
        },
        "reference_model_training_or_scoring": False,
        "reference_model_optimizer_steps": 0,
        "rgb_probe": {"opened": False, "run": False, "scored": False},
    }
    documents = {
        "preregistration": prereg,
        "freeze_receipt": freeze,
        "prior_exclusion_receipt": prior,
    }
    raw = {
        name: json.dumps(value).encode("utf-8")
        for name, value in documents.items()
    }
    identities = {name: probe._raw_identity(value) for name, value in raw.items()}
    freeze["preregistration"] = identities["preregistration"]
    prior["preregistration"] = identities["preregistration"]
    raw["freeze_receipt"] = json.dumps(freeze).encode("utf-8")
    identities["freeze_receipt"] = probe._raw_identity(raw["freeze_receipt"])
    prior["freeze_receipt"] = identities["freeze_receipt"]
    raw["prior_exclusion_receipt"] = json.dumps(prior).encode("utf-8")
    paths = {
        "preregistration": probe.CANONICAL_PREREG_PATH,
        "freeze_receipt": probe.CANONICAL_FREEZE_RECEIPT_PATH,
        "prior_exclusion_receipt": probe.CANONICAL_PRIOR_EXCLUSION_PATH,
    }
    monkeypatch.setattr(
        probe,
        "_read_bytes_nofollow",
        lambda path, *, label: raw[
            next(name for name, candidate in paths.items() if candidate == path)
        ],
    )
    with pytest.raises(ValueError, match="implementation identities differ"):
        probe.validate_authorization_chain(
            prereg_path=paths["preregistration"],
            freeze_receipt_path=paths["freeze_receipt"],
            prior_exclusion_path=paths["prior_exclusion_receipt"],
        )
