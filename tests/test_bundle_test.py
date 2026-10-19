"""Selected-task Test verification and the explicit bundle scoring boundary."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from contextworld.benchmarks import (
    bundle_test,
    motion_damping_bundle_score as bundle_damping,
    motion_damping_icl_score as damping,
)


@pytest.fixture
def tiny_bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "bundle"
    payload = root / "artifacts/task/validation.lance/data/part.lance"
    payload.parent.mkdir(parents=True)
    payload.write_bytes(b"native test bytes")
    config = tmp_path / "release.yaml"
    config.write_bytes(b"fixed scoring contract")
    config_hash = hashlib.sha256(config.read_bytes()).hexdigest()
    registry = {
        "public_test": {"included": True, "policy": bundle_test.PUBLIC_TEST_POLICY},
        "components": [{
            "component_id": "motion_damping",
            "release_config": "configs/benchmark/pusht_motion_damping_icl_release_v1.yaml",
            "release_config_sha256": config_hash,
            "public_test_evaluation": {
                "schema_version": bundle_test.PUBLIC_TEST_EVALUATION_SCHEMA_VERSION,
                "status": bundle_test.PUBLIC_TEST_EVALUATION_STATUS,
                "split": "test",
                "artifact_root": "artifacts/task",
                "payloads": [{"public_path": "artifacts/task/validation.lance"}],
                "selection_policy": "development_only_model_selection_test_final_reporting",
            },
        }],
    }
    (root / "task_registry.json").write_text(json.dumps(registry))
    row = {
        "path": payload.relative_to(root).as_posix(),
        "role": "dataset_payload", "component": "motion_damping", "split": "test",
        "bytes": payload.stat().st_size,
        "sha256": hashlib.sha256(payload.read_bytes()).hexdigest(),
    }
    (root / "manifest.jsonl").write_text(json.dumps(row) + "\n")
    monkeypatch.setattr(bundle_test, "resolve_contextworld_bundle", lambda root: {
        "manifest_sha256": hashlib.sha256((root / "manifest.jsonl").read_bytes()).hexdigest(),
        "task_registry_sha256": hashlib.sha256((root / "task_registry.json").read_bytes()).hexdigest(),
    })
    return root, config, payload


def test_selected_test_files_and_config_are_checked(tiny_bundle):
    root, config, payload = tiny_bundle
    receipt = bundle_test.verify_public_test_bundle(root, task="motion_damping", release_config=config)
    assert receipt["manifest_payload_files"] == 1
    assert receipt["test_payload_sha256_verified"] is True
    payload.write_bytes(b"corrupt test bytes")
    with pytest.raises(ValueError, match="differs from manifest"):
        bundle_test.verify_public_test_bundle(root, task="motion_damping", release_config=config)


def test_unlisted_file_and_wrong_config_are_rejected(tiny_bundle):
    root, config, payload = tiny_bundle
    wrong = config.with_name("wrong.yaml")
    wrong.write_bytes(b"different scoring contract")
    with pytest.raises(ValueError, match="release config differs"):
        bundle_test.verify_public_test_bundle(root, task="motion_damping", release_config=wrong)
    (payload.parent / "unlisted.lance").write_bytes(b"not in manifest")
    with pytest.raises(ValueError, match="files differ from manifest"):
        bundle_test.verify_public_test_bundle(root, task="motion_damping", release_config=config)


def test_hf_cache_file_symlink_is_accepted(tiny_bundle, tmp_path):
    root, config, payload = tiny_bundle
    blob = tmp_path / "cache-blob"
    blob.write_bytes(payload.read_bytes())
    payload.unlink()
    payload.symlink_to(blob)
    receipt = bundle_test.verify_public_test_bundle(root, task="motion_damping", release_config=config)
    assert receipt["test_payload_sha256_verified"] is True


def test_direct_motion_damping_still_invokes_legacy_audit(monkeypatch):
    def stopped(**kwargs):
        raise RuntimeError("legacy audit was called")
    monkeypatch.setattr(damping, "audit_motion_damping_icl_release", stopped)
    with pytest.raises(RuntimeError, match="legacy audit was called"):
        damping.evaluate_motion_damping_icl_model(
            adapter=object(), model_name="toy", training_recipe="test", training_seed=None
        )


def test_motion_damping_bundle_and_legacy_use_same_metrics(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    config.write_text("contract")
    root = tmp_path / "bundle"
    root.mkdir()
    release = {"_config_path": str(config), "release_id": "toy", "data": {"manifest_sha256": "fixed"}}
    pixels_fast = np.zeros((2, 4, 1, 1, 1), dtype=np.float32)
    pixels_slow = np.ones((2, 4, 1, 1, 1), dtype=np.float32)
    arrays = SimpleNamespace(
        pair_ids=("a", "b"), pair_count=2,
        faster_decay_pixels=pixels_fast, no_extra_decay_pixels=pixels_slow,
        raw_action_blocks=np.zeros((2, 4, 1, 1), dtype=np.float32),
    )
    class Dataset:
        def __init__(self, **kwargs):
            self.arrays = arrays
        is_full_protocol = True
        def describe(self):
            return {"pair_count": 2}
    class Adapter:
        metadata = {"name": "toy"}
        def frozen_state_hash(self):
            return "fixed"
        def rollout_latents(self, histories, actions, *, batch_size):
            value = histories[:, 0, 0, 0, 0]
            return np.stack((value, value + 0.5), axis=-1)[:, None, :]
        def encode_pixels(self, pixels, *, batch_size):
            value = pixels[:, 0, 0, 0]
            return np.stack((value, value + 0.5), axis=-1)
    for module in (damping, bundle_damping):
        monkeypatch.setattr(module, "load_motion_damping_icl_release", lambda _: release)
        monkeypatch.setattr(module, "MotionDampingICLEvalDataset", Dataset)
        monkeypatch.setattr(module, "validate_adapter_protocol", lambda *a, **k: None)
        monkeypatch.setattr(module, "motion_damping_prediction_gate", lambda metrics, **k: {"passed": True})
    monkeypatch.setattr(damping, "audit_motion_damping_icl_release", lambda **k: {"passed": True})
    monkeypatch.setattr(bundle_damping, "verify_public_test_bundle", lambda *a, **k: {"test_payload_sha256_verified": True})
    kwargs = dict(adapter=Adapter(), model_name="toy", training_recipe="test", training_seed=None,
                  release_config=config, repo_root=root, include_records=False)
    legacy = damping.evaluate_motion_damping_icl_model(**kwargs)
    bundled = bundle_damping.evaluate_motion_damping_bundle_model(**kwargs, test_bundle_root=root)
    assert legacy["metrics"] == bundled["metrics"]
    assert "bundle_test_verification" not in legacy
    assert bundled["bundle_test_verification"]["test_payload_sha256_verified"] is True
    with pytest.raises(ValueError, match="must match repo_root"):
        bundle_damping.evaluate_motion_damping_bundle_model(**kwargs, test_bundle_root=tmp_path)
