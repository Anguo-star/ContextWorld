from __future__ import annotations

from io import BytesIO
import hashlib
import json
from pathlib import Path
import shutil

import h5py
import lance
import numpy as np
import pyarrow as pa
from PIL import Image
import pytest
import yaml

import scripts.freeze_cube_grasp_rule_h3_v4_failed_formal_attempt as freezer


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _jpeg(value: int) -> bytes:
    image = Image.fromarray(np.full((4, 4, 3), value, dtype=np.uint8), mode="RGB")
    stream = BytesIO()
    image.save(stream, format="JPEG", quality=95)
    return stream.getvalue()


def _profile(index: int, *, invalid_constraint: bool) -> np.ndarray:
    profiles = (
        [0.25, 0, 0.25, -0.5, 0],
        [0.25, 0.25, -0.25, -0.25, 0],
        [0.375, 0, -0.125, -0.25, 0],
        [0.125, 0.375, -0.125, -0.375, 0],
    )
    p = np.asarray(profiles[index], dtype=np.float32)
    if invalid_constraint and index == 0:
        p[-1] = np.float32(0.125)
    blocks = np.zeros((4, 5, 5), dtype=np.float32)
    blocks[0, :, 2] = p
    blocks[1, :, 2] = -p
    blocks[2, :, 2] = p
    blocks[:3, :, 4] = np.float32(0.4 if index < 2 else 0.5)
    return blocks


def _prior_entry(values: list[str], field: str) -> dict[str, object]:
    values = sorted(values)
    return {
        "values": values,
        "count": len(values),
        "sha256": freezer.canonical_content_digest(values, field_name=field),
    }


def _fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    invalid_constraint: bool = False,
    prior_overlap: bool = False,
) -> dict[str, Path]:
    monkeypatch.setattr(freezer, "EXPECTED_PAIR_COUNT", 4)
    monkeypatch.setattr(freezer, "EXPECTED_ROW_COUNT", 32)
    monkeypatch.setattr(freezer, "EXPECTED_EPISODE_COUNT", 8)
    monkeypatch.setattr(freezer, "EXPECTED_CATALOG_INDEX_OFFSET", 100)
    monkeypatch.setattr(freezer, "EXPECTED_CATALOG_INDEX_STOP_EXCLUSIVE", 104)
    monkeypatch.setattr(freezer, "EXPECTED_IMAGE_SIZE", (4, 4))

    prereg = tmp_path / "repo/configs/prereg.yaml"
    prereg.parent.mkdir(parents=True)
    prereg.write_text(
        yaml.safe_dump(
            {
                "protocol_id": freezer.PROTOCOL,
                "status": freezer.PREREG_STATUS,
                "public_test": {
                    "access_status": "closed_not_read_not_scored",
                    "opened": False,
                    "read": False,
                    "hashed": False,
                    "scored": False,
                },
                "reference_model_phase": {"training_and_scoring_authorized": False},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    prereg_sha = freezer.file_sha256(prereg)
    monkeypatch.setattr(freezer, "EXPECTED_PREREG_SHA256", prereg_sha)

    builder = tmp_path / "artifacts/evaluation/builder_snapshot.py"
    builder.parent.mkdir(parents=True)
    builder.write_text("# immutable builder snapshot\n", encoding="utf-8")
    builder_sha = freezer.file_sha256(builder)
    monkeypatch.setattr(freezer, "EXPECTED_BUILDER_SHA256", builder_sha)

    source = tmp_path / "source.h5"
    with h5py.File(source, "w") as handle:
        handle.create_dataset("action", data=np.zeros((12, 5), dtype=np.float32))
        handle.create_dataset("ep_len", data=np.ones(10, dtype=np.int32))
    source_sha = freezer.file_sha256(source)
    monkeypatch.setattr(freezer, "EXPECTED_SOURCE_SHA256", source_sha)
    monkeypatch.setattr(freezer, "EXPECTED_SOURCE_SIZE_BYTES", source.stat().st_size)
    monkeypatch.setattr(freezer, "EXPECTED_SOURCE_ROW_COUNT", 12)
    monkeypatch.setattr(freezer, "EXPECTED_SOURCE_EPISODE_COUNT", 10)

    freeze_path = tmp_path / "artifacts/evaluation/freeze.json"
    freeze = {
        "schema_version": 1,
        "protocol_id": freezer.PROTOCOL,
        "status": freezer.FREEZE_STATUS,
        "checks_passed": True,
        "preregistration": {"sha256": prereg_sha},
        "identity": {
            "v4_builder": {"path": "scripts/builder.py", "sha256": builder_sha, "size_bytes": builder.stat().st_size},
            "v4_physics": {"path": "contextworld/v4.py", "sha256": "11" * 32, "size_bytes": 1},
            "v3_physics_dependency": {"path": "contextworld/v3.py", "sha256": "22" * 32, "size_bytes": 1},
            "common_causal_contract": {"path": "contextworld/causal.py", "sha256": "33" * 32, "size_bytes": 1},
        },
        "source_h5": {
            "symbol": freezer.SOURCE_SYMBOL,
            "sha256": source_sha,
            "size_bytes": source.stat().st_size,
            "row_count": 12,
            "episode_count": 10,
        },
        "public_test": {
            "access_status": "closed_not_read_not_scored",
            "opened": False,
            "read": False,
            "hashed": False,
            "scored": False,
        },
        "reference_model_training_or_scoring_authorized": False,
    }
    _write_json(freeze_path, freeze)
    freeze_sha = freezer.file_sha256(freeze_path)
    monkeypatch.setattr(freezer, "EXPECTED_FREEZE_SHA256", freeze_sha)

    old_episode = 0 if prior_overlap else 9
    old_episodes = [old_episode]
    prior_path = tmp_path / "artifacts/evaluation/prior.json"
    prior = {
        "schema_version": 1,
        "protocol_id": freezer.PROTOCOL,
        "status": freezer.FREEZE_STATUS,
        "checks_passed": True,
        "preregistration": {"sha256": prereg_sha},
        "freeze_receipt": {"sha256": freeze_sha},
        "source_h5": freeze["source_h5"],
        "excluded_source_episodes": old_episodes,
        "excluded_source_episode_count": 1,
        "excluded_source_episodes_sha256": freezer.excluded_source_episodes_sha256(old_episodes),
        "prior_content_exclusions": {
            field: _prior_entry([hashlib.sha256(("old-" + field).encode()).hexdigest()], field)
            for field in (*freezer.CONTENT_FIELDS, "query_pixel_hashes")
        },
        "public_test": freeze["public_test"],
        "reference_model_training_or_scoring": False,
    }
    _write_json(prior_path, prior)
    prior_sha = freezer.file_sha256(prior_path)
    monkeypatch.setattr(freezer, "EXPECTED_PRIOR_EXCLUSION_SHA256", prior_sha)

    fields: dict[str, list[object]] = {name: [] for name in freezer.EXPECTED_SCHEMA.names}
    for pair_index in range(4):
        blocks = _profile(pair_index, invalid_constraint=invalid_constraint)
        profile_id = freezer.action_profile_content_sha256(blocks)
        scene_hash = hashlib.sha256(f"scene-{pair_index}".encode()).hexdigest()
        pair_hash = freezer.pair_content_sha256(scene_hash, profile_id)
        pair_id = f"cube-carry-v4-train-{pair_index:06d}"
        for mode_index, mode in enumerate(("cannot_hold", "can_hold")):
            for step in range(4):
                physical = np.zeros(7, dtype=np.float32)
                if step in (1, 3) and mode == "can_hold":
                    physical[4] = np.float32(0.01)
                image_value = 20 * pair_index + 3 * step
                if step not in (0, 2):
                    image_value += mode_index
                fields["episode_idx"].append(2 * pair_index + mode_index)
                fields["model_step_idx"].append(step)
                fields["pixels"].append(_jpeg(image_value))
                fields["action_block"].append(blocks[step].reshape(-1).tolist())
                fields["physical_state"].append(physical.tolist())
                fields["hidden_grasp_enabled"].append([float(mode_index)])
                fields["pair_id"].append(pair_id)
                fields["hidden_mode"].append(mode)
                fields["split"].append("train")
                fields["catalog_index"].append(100 + pair_index)
                fields["source_row"].append(1000 + pair_index)
                fields["source_episode"].append(pair_index)
                fields["source_step"].append(10 + pair_index)
                fields["action_anchor_id"].append(freezer.EXPECTED_ANCHORS[pair_index])
                fields["action_profile_id"].append(profile_id)
                fields["scene_template_content_hash"].append(scene_hash)
                fields["pair_content_hash"].append(pair_hash)
    arrays = [pa.array(fields[field.name], type=field.type) for field in freezer.EXPECTED_SCHEMA]
    table = pa.Table.from_arrays(arrays, schema=freezer.EXPECTED_SCHEMA)
    local_dataset = tmp_path / "local.lance"
    lance.write_dataset(table, str(local_dataset), mode="create")
    local_fragment = next((local_dataset / "data").glob("*.lance"))

    failed_root = tmp_path / "artifacts/synthesis/cube_gripper_carry_rule_h3_development_v4"
    fragment = failed_root / "train.lance/data" / local_fragment.name
    fragment.parent.mkdir(parents=True)
    (failed_root / "train.lance/_versions").mkdir()
    (failed_root / "train.lance/_transactions").mkdir()
    shutil.copy2(local_fragment, fragment)
    monkeypatch.setattr(freezer, "EXPECTED_FRAGMENT_SHA256", freezer.file_sha256(fragment))
    monkeypatch.setattr(freezer, "EXPECTED_FRAGMENT_SIZE_BYTES", fragment.stat().st_size)

    request_path = failed_root / "request.json"
    request = {
        "protocol": freezer.PROTOCOL,
        "resolved_output": freezer.EXPECTED_LOGICAL_OUTPUT,
        "logical_default_output": freezer.EXPECTED_LOGICAL_OUTPUT,
        "pair_counts": {"train": 4, "loader_validation": 256},
        "active_splits": ["train", "loader_validation"],
        "jpeg_quality": 95,
        "workers": 16,
        "public_test_opened": False,
        "public_test_generated": False,
        "freeze_receipt": {"sha256": freeze_sha},
        "prior_episode_exclusion_receipt": {"sha256": prior_sha},
        "source": {
            "source_symbol": freezer.SOURCE_SYMBOL,
            "source_file_sha256": source_sha,
            "source_size_bytes": source.stat().st_size,
            "source_row_count": 12,
            "source_episode_count": 10,
            "formal_catalog_namespace": {
                "per_split_ranges": {"train": {"catalog_index_start_inclusive": 100}}
            },
        },
    }
    _write_json(request_path, request)
    monkeypatch.setattr(freezer, "EXPECTED_REQUEST_SHA256", freezer.file_sha256(request_path))

    output = tmp_path / "artifacts/evaluation/failure.json"
    return {
        "failed_root": failed_root,
        "prereg": prereg,
        "freeze": freeze_path,
        "prior": prior_path,
        "builder": builder,
        "source": source,
        "request": request_path,
        "fragment": fragment,
        "output": output,
    }


def _run(paths: dict[str, Path]) -> dict[str, object]:
    return freezer.freeze_failed_attempt(
        failed_output_root=paths["failed_root"],
        prereg_path=paths["prereg"],
        freeze_receipt_path=paths["freeze"],
        prior_exclusion_receipt_path=paths["prior"],
        builder_snapshot=paths["builder"],
        builder_snapshot_logical_path=freezer.EXPECTED_BUILDER_SNAPSHOT_LOGICAL_PATH,
        source_h5=paths["source"],
        request_json=paths["request"],
        partial_train_fragment=paths["fragment"],
        output=paths["output"],
    )


def test_failed_attempt_receipt_is_complete_exclusive_and_raw_query_safe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    receipt = _run(paths)
    assert receipt["status"] == freezer.RECEIPT_STATUS
    assert receipt["checks_passed"] is True
    assert receipt["formal_build_attempt_consumed"] is True
    assert receipt["failure"]["stage"] == "lance_train_commit_atomic_rename"
    assert receipt["failure"]["errno_name"] == "EPERM"
    assert receipt["failure"]["persistent_log_present"] is False
    content = receipt["failed_attempt_content"]
    assert content["row_count"] == 32
    assert content["pair_count"] == 4
    assert content["action_anchor_counts"] == {
        "endpoint4": 1,
        "front_hold": 1,
        "plateau": 1,
        "ramp4": 1,
    }
    assert content["source_episodes"]["values"] == [0, 1, 2, 3]
    assert len(content["pairs"]) == 4
    assert content["profile_constraints"]["passed"] is True
    assert content["prior_overlap"]["passed_for_directly_inspectable_identities"] is True
    assert content["prior_overlap"]["query_pixel_hash_count"] is None
    assert content["query_pixel_hash_status"].startswith("pending_deterministic")
    assert content["query_jpeg_sha256"]["role"] == (
        "forensic_binding_only_not_raw_query_pixel_hash"
    )
    assert receipt["scope"]["public_test"]["read"] is False
    assert receipt["scope"]["rgb_probe_run"] is False
    assert receipt["scope"]["reference_model_training_or_scoring"] is False
    assert paths["output"].is_file()
    with pytest.raises(FileExistsError):
        _run(paths)


def test_failed_attempt_rejects_unexpected_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    (paths["failed_root"] / "build_report.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="inventory mismatch"):
        _run(paths)


def test_failed_attempt_rejects_invalid_action_constraint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch, invalid_constraint=True)
    with pytest.raises(RuntimeError, match="action recovery constraints"):
        _run(paths)


def test_failed_attempt_rejects_prior_source_overlap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch, prior_overlap=True)
    with pytest.raises(RuntimeError, match="overlaps prior evidence"):
        _run(paths)


def test_cli_rejects_public_path_and_required_inputs() -> None:
    with pytest.raises(SystemExit):
        freezer.parse_args([])
    argv = [
        "--failed-output-root", "safe/root",
        "--prereg", "safe/prereg.yaml",
        "--freeze-receipt", "safe/freeze.json",
        "--prior-exclusion-receipt", "safe/prior.json",
        "--builder-snapshot", "safe/builder.py",
        "--builder-snapshot-logical-path", freezer.EXPECTED_BUILDER_SNAPSHOT_LOGICAL_PATH,
        "--source-h5", "safe/source.h5",
        "--request-json", "safe/request.json",
        "--partial-train-fragment", "safe/train.lance/data/x.lance",
        "--output", "public/failure.json",
    ]
    with pytest.raises(RuntimeError, match="Public"):
        freezer.parse_args(argv)


def test_failed_attempt_rejects_fragment_symlink_before_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    real_fragment = paths["fragment"].with_name("real.lance")
    paths["fragment"].rename(real_fragment)
    paths["fragment"].symlink_to(real_fragment)
    monkeypatch.setattr(freezer, "EXPECTED_FRAGMENT_SHA256", freezer.file_sha256(real_fragment))
    monkeypatch.setattr(freezer, "EXPECTED_FRAGMENT_SIZE_BYTES", real_fragment.stat().st_size)
    with pytest.raises(FileNotFoundError, match="symlink"):
        _run(paths)
