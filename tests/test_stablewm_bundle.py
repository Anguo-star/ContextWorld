"""The clean HF bundle is exposed to StableWM without copying its payloads."""

from __future__ import annotations

import hashlib
import io
import json
import pickle
import sys
from pathlib import Path

import lance
import h5py
import numpy as np
import pyarrow as pa
import pytest
from PIL import Image

from contextworld.benchmarks import bundle_cli
from contextworld.training.stablewm_bundle import (
    URI_PREFIX,
    build_contextworld_dataset_uri,
    describe_contextworld_dataset,
    register_stablewm_bundle_format,
    resolve_contextworld_bundle,
    resolve_contextworld_component,
    resolve_contextworld_development_payload,
    resolve_contextworld_development_payload_members,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_stablewm_train as launcher  # noqa: E402


def _png() -> bytes:
    stream = io.BytesIO()
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(
        stream, format="PNG"
    )
    return stream.getvalue()


def _write_release(
    root: Path,
    *,
    component: str,
    dataset_id: str,
    action_dim: int,
    adapter: str | None,
    cube: bool = False,
    tworoom_normalizer: bool = False,
) -> None:
    member = root / f"components/{dataset_id}/v1/training/data.lance"
    development_member = root / f"components/{dataset_id}/v1/development/data.lance"
    member.parent.mkdir(parents=True)
    development_member.parent.mkdir(parents=True)
    if cube:
        rows = 4
        table = pa.table(
            {
                "episode_idx": pa.array([0] * rows, type=pa.int64()),
                "model_step_idx": pa.array(range(rows), type=pa.int64()),
                "pixels": pa.array([_png()] * rows, type=pa.binary()),
                "action_block": pa.array(
                    np.arange(rows * 25, dtype=np.float32)
                    .reshape(rows, 25)
                    .tolist(),
                    type=pa.list_(pa.float32(), 25),
                ),
                "pair_id": pa.array(["pair"] * rows),
            }
        )
        sequence_schema = "blocked_transition_projection_v1"
    else:
        rows = 20
        table = pa.table(
            {
                "episode_idx": pa.array([0] * rows, type=pa.int64()),
                "step_idx": pa.array(range(rows), type=pa.int64()),
                "pixels": pa.array([_png()] * rows, type=pa.binary()),
                "action": pa.array(
                    np.arange(rows * action_dim, dtype=np.float32).reshape(
                        rows, action_dim
                    ).tolist(),
                    type=pa.list_(pa.float32(), action_dim),
                ),
                "pair_id": pa.array(["pair"] * rows),
            }
        )
        sequence_schema = "native_episode_sequence_with_step_metadata_v1"
    lance.write_dataset(table, str(member))
    lance.write_dataset(table, str(development_member))

    relative = member.relative_to(root).as_posix()
    development_relative = development_member.relative_to(root).as_posix()
    development_payload = {
        "split": "development",
        "payload_id": "data",
        "payload_kind": "single_lance",
        "public_path": development_relative,
        "members": [development_relative],
        "lance_table_count": 1,
        "stable_worldmodel_sequence_schema": sequence_schema,
        "stable_worldmodel_adapter_required": adapter,
    }
    action_normalization = {
        "transform": "zscore",
        "source": "test_original_data",
        "mean": [0.0] * action_dim,
        "std": [1.0] * action_dim,
        "std_estimator": "population",
    }
    normalizer_path = None
    normalizer_file = None
    if tworoom_normalizer:
        normalizer_path = "normalizers/tworoom_original_train_s3072.json"
        action_normalization = {
            "transform": "zscore",
            "source": "original_tworoom_training_split",
            "mean": [0.0031402341986976924, -0.051594576296864605],
            "std": [0.867571689163936, 0.8688840167517821],
            "std_estimator": "unbiased",
        }
        normalizer_file = root / normalizer_path
        normalizer_file.parent.mkdir(parents=True)
        normalizer_file.write_text(
            json.dumps(
                {
                    "protocol": "tworoom_original_train_s3072_unbiased_zscore_v1",
                    "statistics_scope": "original_9000_train_episodes_only",
                    "columns": {
                        "action": {
                            "mean": action_normalization["mean"],
                            "std_unbiased": action_normalization["std"],
                        },
                        "proprio": {
                            "mean": [111.7950199284305, 85.03849594298646],
                            "std_unbiased": [36.85458874773545, 38.17356572449523],
                        },
                    },
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    development_evaluation = {
        "schema_version": "contextworld.development_evaluation.v1",
        "status": "public_development_only",
        "split": "development",
        "payload_id": "data",
        "reader_id": "test_paired_condition_lance_v1",
        "selection": {
            "expected_pair_count": 1,
            "selected_pair_count": 1,
            "method": "all_pairs_v1",
        },
        "input_contract": {
            "context_streams": ["pixels", "actions"],
            "history_length": 3,
            "action_block_raw_steps": 5,
            "prediction_horizon_action_blocks": 1,
        },
        "action_normalization": action_normalization,
        "payload": {
            "public_path": development_relative,
            "payload_kind": "single_lance",
            "lance_table_count": 1,
            "members": [development_relative],
        },
    }
    if normalizer_path is not None:
        development_evaluation["normalizer_path"] = normalizer_path
    registry = {
        "schema_version": "contextworld.hf-task-registry.v1",
        "release_status": "staging_not_public_release",
        "public_test": {"included": False, "policy": "withheld"},
        "components": [
            {
                "component_id": component,
                "dataset_id": dataset_id,
                "environment": "Cube" if cube else "PushT",
                "history_length": 3,
                "action_dimension": action_dim,
                "frameskip": 5,
                "payloads": [
                    {
                        "split": "training",
                        "payload_id": "data",
                        "payload_kind": "single_lance",
                        "public_path": relative,
                        "members": [relative],
                        "lance_table_count": 1,
                        "stable_worldmodel_sequence_schema": sequence_schema,
                        "stable_worldmodel_adapter_required": adapter,
                    },
                    development_payload,
                ],
                "development_evaluation": development_evaluation,
            }
        ],
    }
    registry_path = root / "task_registry.json"
    registry_path.write_text(
        json.dumps(registry, sort_keys=True) + "\n", encoding="utf-8"
    )
    registry_sha = hashlib.sha256(registry_path.read_bytes()).hexdigest()
    manifest = root / "manifest.jsonl"
    rows = [
        {
            "path": "task_registry.json",
            "sha256": registry_sha,
            "size": registry_path.stat().st_size,
        }
    ]
    for payload_file in sorted(development_member.rglob("*")):
        if not payload_file.is_file():
            continue
        rows.append(
            {
                "path": payload_file.relative_to(root).as_posix(),
                "sha256": hashlib.sha256(payload_file.read_bytes()).hexdigest(),
                "size": payload_file.stat().st_size,
                "role": "dataset_payload",
                "component": component,
                "split": "development",
            }
        )
    if normalizer_file is not None:
        rows.append(
            {
                "path": normalizer_file.relative_to(root).as_posix(),
                "sha256": hashlib.sha256(normalizer_file.read_bytes()).hexdigest(),
                "size": normalizer_file.stat().st_size,
                "role": "release_metadata",
            }
        )
    manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    (root / "manifest.sha256").write_text(
        f"{manifest_sha}  manifest.jsonl\n", encoding="utf-8"
    )


def _refresh_registry_receipt(root: Path) -> None:
    """Rebind a deliberately edited test registry into its tiny release."""

    registry = root / "task_registry.json"
    registry_sha = hashlib.sha256(registry.read_bytes()).hexdigest()
    manifest = root / "manifest.jsonl"
    rows = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
    ]
    for row in rows:
        if row.get("path") == "task_registry.json":
            row["sha256"] = registry_sha
            row["size"] = registry.stat().st_size
            break
    else:  # pragma: no cover - test fixture invariant
        raise AssertionError("test release manifest did not bind task_registry.json")
    manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    (root / "manifest.sha256").write_text(
        f"{manifest_sha}  manifest.jsonl\n", encoding="utf-8"
    )


def test_step_metadata_projection_is_a_stablewm_sequence(tmp_path: Path) -> None:
    root = tmp_path / "ContextWorld-v1"
    _write_release(
        root,
        component="action_strength",
        dataset_id="pusht-action-strength",
        action_dim=2,
        adapter="stablewm_step_metadata_to_episode_table_v1",
    )
    uri = build_contextworld_dataset_uri(
        root,
        component="action_strength",
        synthetic_weight=1.0,
        epoch_size=5,
    )
    identity = describe_contextworld_dataset(uri)

    assert uri.startswith(URI_PREFIX)
    assert identity["member_count"] == 1
    assert len(identity["member_list_sha256"]) == 64
    assert "members" not in identity

    register_stablewm_bundle_format()
    import stable_worldmodel as swm

    dataset = swm.data.load_dataset(
        uri,
        num_steps=4,
        frameskip=5,
        transform=None,
        keys_to_load=["pixels", "action"],
        keys_to_cache=["action"],
    )
    # Native Lance handles must not survive construction in the parent that
    # later starts DataLoader workers.
    assert dataset.leaves[0]._dataset is None
    sample = dataset[0]
    assert dataset.leaves[0]._dataset is not None
    assert len(dataset) == 5
    assert dataset.column_names == ["pixels", "action"]
    assert dataset.get_dim("action") == 2
    assert sample["pixels"].shape == (4, 3, 8, 8)
    assert sample["action"].shape == (4, 10)
    assert len(dataset.__getitems__([0, 1])) == 2
    restored = pickle.loads(pickle.dumps(dataset))
    assert restored.leaves[0]._dataset is None


def test_cube_projection_preserves_five_raw_actions_per_model_step(
    tmp_path: Path,
) -> None:
    root = tmp_path / "ContextWorld-v1"
    _write_release(
        root,
        component="cube_gripper_carry",
        dataset_id="cube-gripper-carry",
        action_dim=5,
        adapter="cube_block_projection_to_sequence_v1",
        cube=True,
    )
    uri = build_contextworld_dataset_uri(
        root,
        component="cube_gripper_carry",
        synthetic_weight=1.0,
        epoch_size=1,
    )
    register_stablewm_bundle_format()
    import stable_worldmodel as swm

    dataset = swm.data.load_dataset(
        uri,
        num_steps=4,
        frameskip=5,
        transform=None,
        keys_to_load=["pixels", "action"],
        keys_to_cache=["action"],
    )
    assert dataset.leaves[0]._dataset is None
    sample = dataset[0]
    assert dataset.leaves[0]._dataset is not None
    assert dataset.get_dim("action") == 5
    assert dataset.get_col_data("action").shape == (20, 5)
    assert sample["pixels"].shape == (4, 3, 8, 8)
    assert sample["action"].shape == (4, 25)
    assert len(dataset.__getitems__([0, 0])) == 2
    restored = pickle.loads(pickle.dumps(dataset))
    assert restored.leaves[0]._dataset is None


def test_development_resolver_is_manifest_bound_and_needs_no_private_tree(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "ContextWorld-v1"
    _write_release(
        root,
        component="door",
        dataset_id="tworoom-door",
        action_dim=2,
        adapter="",
        tworoom_normalizer=True,
    )
    monkeypatch.setenv("CONTEXTWORLD_ARTIFACT_ROOT", "/definitely/not/used")

    bundle = resolve_contextworld_bundle(root)
    component = resolve_contextworld_component(root, component="door")
    payload = resolve_contextworld_development_payload(root, component="door")
    members = resolve_contextworld_development_payload_members(root, component="door")

    assert bundle["bundle_root"] == str(root)
    assert len(bundle["manifest_sha256"]) == 64
    assert len(bundle["task_registry_sha256"]) == 64
    assert component["development_evaluation"]["payload_id"] == "data"
    assert payload["relative_members"] == (
        "components/tworoom-door/v1/development/data.lance",
    )
    assert members == (
        root / "components/tworoom-door/v1/development/data.lance",
    )
    assert payload["normalizer_path"] == str(
        root / "normalizers/tworoom_original_train_s3072.json"
    )


def test_synthetic_only_door_uses_registered_original_action_normalization(
    tmp_path: Path,
) -> None:
    root = tmp_path / "ContextWorld-v1"
    _write_release(
        root,
        component="door",
        dataset_id="tworoom-door",
        action_dim=2,
        adapter=None,
        tworoom_normalizer=True,
    )
    uri = build_contextworld_dataset_uri(
        root,
        component="door",
        synthetic_weight=1.0,
    )

    register_stablewm_bundle_format()
    import stable_worldmodel as swm

    dataset = swm.data.load_dataset(
        uri,
        num_steps=4,
        frameskip=5,
        transform=None,
        keys_to_load=["pixels", "action"],
        keys_to_cache=["action"],
    )
    scaler = swm.data.ZScoreScaler().fit(dataset.get_col_data("action"))
    expected_mean = np.asarray(
        [0.0031402341986976924, -0.051594576296864605], dtype=np.float64
    )
    expected_std = np.asarray(
        [0.867571689163936, 0.8688840167517821], dtype=np.float64
    )

    # StableWM fits population standard deviation.  The two contract-derived
    # rows are deliberately symmetric, so they reproduce the frozen TwoRoom
    # stats without loading ``quentinll/tworoom.h5``.
    np.testing.assert_allclose(scaler.mean.reshape(-1), expected_mean)
    np.testing.assert_allclose(scaler.std.reshape(-1), expected_std)


def test_door_mixture_keeps_original_dataset_as_normalizer_source(
    tmp_path: Path,
) -> None:
    root = tmp_path / "ContextWorld-v1"
    _write_release(
        root,
        component="door",
        dataset_id="tworoom-door",
        action_dim=2,
        adapter=None,
        tworoom_normalizer=True,
    )
    original = tmp_path / "quentinll/tworoom.h5"
    original.parent.mkdir()
    original_actions = np.arange(40, dtype=np.float32).reshape(20, 2)
    with h5py.File(original, "w") as handle:
        handle.create_dataset("ep_len", data=np.asarray([20], dtype=np.int32))
        handle.create_dataset("ep_offset", data=np.asarray([0], dtype=np.int64))
        handle.create_dataset(
            "pixels", data=np.zeros((20, 8, 8, 3), dtype=np.uint8)
        )
        handle.create_dataset("action", data=original_actions)
    uri = build_contextworld_dataset_uri(
        root,
        component="door",
        original_dataset=original,
        original_weight=0.5,
        synthetic_weight=0.5,
    )

    register_stablewm_bundle_format()
    import stable_worldmodel as swm

    dataset = swm.data.load_dataset(
        uri,
        num_steps=4,
        frameskip=5,
        transform=None,
        keys_to_load=["pixels", "action"],
        keys_to_cache=["action"],
    )

    np.testing.assert_array_equal(dataset.get_col_data("action"), original_actions)


def test_synthetic_only_normalizer_rejects_malformed_action_contract(
    tmp_path: Path,
) -> None:
    root = tmp_path / "ContextWorld-v1"
    _write_release(
        root,
        component="door",
        dataset_id="tworoom-door",
        action_dim=2,
        adapter=None,
        tworoom_normalizer=True,
    )
    registry_path = root / "task_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    normalization = registry["components"][0]["development_evaluation"][
        "action_normalization"
    ]
    normalization["mean"] = [0.0, 0.0, 0.0]
    normalization["std"] = [1.0, 1.0, 1.0]
    registry_path.write_text(
        json.dumps(registry, sort_keys=True) + "\n", encoding="utf-8"
    )
    _refresh_registry_receipt(root)
    uri = build_contextworld_dataset_uri(
        root,
        component="door",
        synthetic_weight=1.0,
    )

    register_stablewm_bundle_format()
    import stable_worldmodel as swm

    with pytest.raises(ValueError, match="dimensions must match action_dimension=2"):
        swm.data.load_dataset(
            uri,
            num_steps=4,
            frameskip=5,
            transform=None,
            keys_to_load=["pixels", "action"],
            keys_to_cache=["action"],
        )


@pytest.mark.parametrize("family", ["lewm", "viswm", "pldm", "prejepa"])
def test_all_builtin_families_build_the_same_registered_mixture_without_cw_dataset(
    tmp_path: Path,
    family: str,
) -> None:
    root = tmp_path / "ContextWorld-v1"
    _write_release(
        root,
        component="action_strength",
        dataset_id="pusht-action-strength",
        action_dim=2,
        adapter="stablewm_step_metadata_to_episode_table_v1",
    )
    original_root = tmp_path / "world_model"
    original = original_root / "quentinll/pusht_expert_train.h5"
    original.parent.mkdir(parents=True)
    original.write_bytes(b"registered original payload")
    args = launcher.parse_args(
        [
            "--family",
            family,
            "--component",
            "action_strength",
            "--benchmark-root",
            str(root),
            "--dataset-root",
            str(original_root),
            "--checkpoint-root",
            str(tmp_path / "checkpoints"),
        ]
    )

    target = launcher.resolve_target(args, launcher.load_profile_contract())
    identity = describe_contextworld_dataset(str(target.dataset))

    assert str(target.dataset).startswith(URI_PREFIX)
    assert target.history_size == 3
    assert target.action_dim == 2
    assert identity["weights"] == {"original": 0.5, "synthetic": 0.5}
    assert identity["original_dataset"] == str(original)


@pytest.mark.parametrize("family", ["lewm", "viswm", "pldm", "prejepa"])
def test_training_process_registers_the_bundle_in_every_child_environment(
    tmp_path: Path,
    monkeypatch,
    family: str,
) -> None:
    root = tmp_path / "ContextWorld-v1"
    _write_release(
        root,
        component="action_strength",
        dataset_id="pusht-action-strength",
        action_dim=2,
        adapter="stablewm_step_metadata_to_episode_table_v1",
    )
    original_root = tmp_path / "world_model"
    original = original_root / "quentinll/pusht_expert_train.h5"
    original.parent.mkdir(parents=True)
    with h5py.File(original, "w") as handle:
        handle.create_dataset("pixels", shape=(20, 8, 8, 3), dtype="uint8")
        handle.create_dataset("action", shape=(20, 2), dtype="float32")
        handle.create_dataset("proprio", shape=(20, 4), dtype="float32")

    stablewm = tmp_path / "stable-worldmodel"
    config = stablewm / "scripts/train/config"
    config.mkdir(parents=True)
    for family_name in ("lewm", "viswm", "pldm", "prejepa"):
        (config.parent / f"{family_name}.py").write_text(
            "pass\n", encoding="utf-8"
        )
        payload = "trainer:\n  max_epochs: 1\n"
        if family_name == "viswm":
            payload += "loss:\n  regularizer: visreg\n  visreg:\n    weight: 4.5\n"
        (config / f"{family_name}.yaml").write_text(payload, encoding="utf-8")
    data_config = config / "data"
    data_config.mkdir()
    (data_config / "pusht.yaml").write_text(
        "dataset:\n"
        "  keys_to_load: [pixels, action, proprio, state]\n"
        "  keys_to_cache: [action, proprio, state]\n",
        encoding="utf-8",
    )
    data_package = stablewm / "stable_worldmodel/data"
    data_package.mkdir(parents=True)
    (data_package / "format.py").write_text(
        "FORMATS = {}\ndef register_format(value): return value\n",
        encoding="utf-8",
    )
    (data_package / "utils.py").write_text(
        "def load_dataset(name):\n    if '://' in name:\n        return name\n",
        encoding="utf-8",
    )
    calls = []

    class Completed:
        returncode = 0

    def run(command, **kwargs):
        calls.append((command, kwargs["env"]))
        return Completed()

    monkeypatch.delenv("SPT_CACHE_DIR", raising=False)
    monkeypatch.setattr(launcher.subprocess, "run", run)
    status = launcher.main(
        [
            "--family",
            family,
            "--component",
            "action_strength",
            "--benchmark-root",
            str(root),
            "--dataset-root",
            str(original_root),
            "--stablewm-repo",
            str(stablewm),
            "--checkpoint-root",
            str(tmp_path / "checkpoints"),
            "--seeds",
            "9901",
        ]
    )

    assert status == 0
    assert len(calls) == 1
    command, environment = calls[0]
    dataset_key = "dataset_name=" if family == "prejepa" else "data.dataset.name="
    worker_key = "num_workers=2" if family == "prejepa" else "loader.num_workers=2"
    assert any(value.startswith(dataset_key + "contextworld://v1/") for value in command)
    assert worker_key in command
    if family in {"lewm", "viswm", "pldm"}:
        assert "data.dataset.keys_to_load=[pixels,action]" in command
        assert "data.dataset.keys_to_cache=[action]" in command
    assert environment["CONTEXTWORLD_STABLEWM_BUNDLE"] == "1"
    assert environment["CONTEXTWORLD_DATALOADER_START_METHOD"] == "spawn"
    assert environment["PYTHONPATH"].split(":")[0] == str(
        launcher.STABLEWM_BOOTSTRAP_DIR
    )


def test_public_bundle_cli_info_and_audit_do_not_need_artifact_root(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    root = tmp_path / "ContextWorld-v1"
    _write_release(
        root,
        component="action_strength",
        dataset_id="pusht-action-strength",
        action_dim=2,
        adapter="stablewm_step_metadata_to_episode_table_v1",
    )
    (root / "VERSION.json").write_text(
        json.dumps(
            {
                "dataset_version": "1.0.0-rc1",
                "release_status": "staging_not_public_release",
                "public_test_included": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("CONTEXTWORLD_BENCHMARK_ROOT", str(root))
    monkeypatch.setenv("CONTEXTWORLD_ARTIFACT_ROOT", "/definitely/not/used")

    assert bundle_cli.main(["info"]) == 0
    info = json.loads(capsys.readouterr().out)
    assert info["component_count"] == 1
    assert info["components"][0]["component_id"] == "action_strength"

    assert bundle_cli.main(["audit"]) == 0
    audit = json.loads(capsys.readouterr().out)
    assert audit["status"] == "passed"
    assert audit["training_view_count"] == 1
    assert audit["public_test_included"] is False
