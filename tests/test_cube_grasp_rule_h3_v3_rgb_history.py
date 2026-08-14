from __future__ import annotations

from dataclasses import replace
import hashlib
from io import BytesIO
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import pytest

import scripts.probe_cube_grasp_rule_h3_v3_rgb_history as probe


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


def _action_blocks(split: str, pair_index: int) -> np.ndarray:
    split_offset = 1 if split == "train" else 101
    value = np.float32((split_offset + pair_index) / 256.0)
    blocks = np.zeros(probe.ACTION_PROFILE_SHAPE, dtype=np.float32)
    blocks[0, 0, 0] = value
    blocks[1, 1, 1] = np.float32(-value)
    blocks[2, 2, 2] = np.float32(value / 2.0)
    return blocks


def _fixture_rows(split: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pair_index, anchor in enumerate(probe.ACTION_ANCHORS):
        pair_id = f"fixture-{split}-{pair_index}"
        blocks = _action_blocks(split, pair_index)
        profile_id = probe.action_profile_content_sha256(blocks)
        scene_hash = _digest(f"scene:{split}:{pair_index}")
        pair_hash = probe.pair_content_sha256(scene_hash, profile_id)
        x0 = _jpeg(90 + pair_index)
        query = _jpeg(100 + pair_index)
        for mode in probe.HIDDEN_MODES:
            history = _jpeg(65 if mode == "cannot_hold" else 145)
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
                        "pair_id": pair_id,
                        "hidden_mode": mode,
                        "split": split,
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


def test_frozen_probe_contract_is_exact_and_development_only() -> None:
    assert probe.PROTOCOL == "cube_gripper_carry_rule_history3_development_v3"
    assert probe.ACTIVE_SPLITS == ("train", "loader_validation")
    assert probe.TABLE_NAMES == {
        "train": "train.lance",
        "loader_validation": "loader_validation.lance",
    }
    assert probe.LABEL_ENCODING == {"cannot_hold": 0, "can_hold": 1}
    assert probe.BOOTSTRAP_RESAMPLES == 10_000
    assert probe.BOOTSTRAP_SEED == 2026081103
    assert probe.BOOTSTRAP_LOWER_QUANTILE == 0.025
    assert probe.PERMUTATION_REPETITIONS == 16
    assert probe.PERMUTATION_SEED == 2026081104
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
    assert "hidden_grasp_enabled" not in probe.TABLE_COLUMNS
    assert {"episode_idx", "model_step_idx", "source_step"} <= set(
        probe.PRIVILEGED_COLUMNS_EXCLUDED_FROM_MAIN_FEATURE
    )


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
    pair_id = "fixture-train-0"
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
    condition = _condition_rows(rows, "fixture-train-0", "can_hold")
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
        rows, "fixture-loader_validation-0", "cannot_hold"
    )
    for row in condition:
        row["pair_content_hash"] = "f" * 64
    with pytest.raises(ValueError, match="does not bind scene/profile"):
        probe.prepare_split(rows, expected_split="loader_validation")


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
    assert all(report["gates"].values())

    metrics = report["primary_probe"]["metrics"]
    assert metrics["overall_accuracy"] == 1.0
    assert metrics["worst_mode"]["accuracy"] == 1.0
    assert metrics["worst_anchor_family"]["accuracy"] == 1.0
    bootstrap = report["pair_cluster_anchor_stratified_bootstrap"]
    assert bootstrap["resamples"] == 10_000
    assert bootstrap["seed"] == 2026081103
    assert bootstrap["lower_bound_2_5_percent"] == 1.0

    controls = report["negative_controls"]
    assert controls["label_permutation"]["repetitions"] == 16
    assert controls["label_permutation"]["seed"] == 2026081104
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

    class _Schema:
        names = list(probe.TABLE_COLUMNS)

    class _Dataset:
        schema = _Schema()

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


def test_run_probe_reads_only_the_two_whitelisted_tables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    for table_name in probe.TABLE_NAMES.values():
        table = artifact / table_name
        table.mkdir()
        (table / "fixture.bin").write_bytes(table_name.encode("ascii"))
    opened: list[tuple[str, str]] = []

    def fixture_reader(path: Path, *, expected_split: str) -> list[dict[str, Any]]:
        opened.append((path.name, expected_split))
        return _fixture_rows(expected_split)

    monkeypatch.setattr(probe, "_read_lance_rows", fixture_reader)
    report = probe.run_probe(artifact)
    assert opened == [
        ("train.lance", "train"),
        ("loader_validation.lance", "loader_validation"),
    ]
    assert report["inputs"]["only_authorized_lance_tables_opened"] == [
        "train.lance",
        "loader_validation.lance",
    ]
    assert report["inputs"]["manifest_or_build_report_read"] is False
    assert report["inputs"]["validation_or_public_table_read"] is False
    for split in probe.ACTIVE_SPLITS:
        table = report["inputs"]["tables"][split]
        assert table["table_directory_hashed"] is False
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


def test_main_refuses_overwrite_before_probe_reads_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "existing.json"
    output.write_text("do not replace\n", encoding="utf-8")
    called = False

    def forbidden_probe(_root: Path) -> dict[str, Any]:
        nonlocal called
        called = True
        raise AssertionError("probe should not run")

    monkeypatch.setattr(probe, "run_probe", forbidden_probe)
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        probe.main(
            [
                "--artifact-root",
                str(tmp_path / "missing"),
                "--output",
                str(output),
            ]
        )
    assert not called
    assert output.read_text(encoding="utf-8") == "do not replace\n"


def test_main_writes_json_exclusively_without_touching_fixture_tables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "train.lance").mkdir()
    (artifact / "loader_validation.lance").mkdir()
    output = tmp_path / "probe.json"
    payload = {"status": "passed", "passed": True}
    monkeypatch.setattr(probe, "run_probe", lambda _root: dict(payload))
    assert probe.main(
        [
            "--artifact-root",
            str(artifact),
            "--output",
            str(output),
        ]
    ) == 0
    assert json.loads(output.read_text(encoding="utf-8")) == payload
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        probe.main(
            [
                "--artifact-root",
                str(artifact),
                "--output",
                str(output),
            ]
        )
