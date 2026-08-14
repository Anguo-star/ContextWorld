from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

import contextworld.evaluation.cube_grasp_rule_h3_v4 as v4_physics
import scripts.build_cube_grasp_rule_h3_v4_data as development_builder
import scripts.build_cube_grasp_rule_h3_v4r1_public_data as public_builder


def _exclusions() -> dict:
    return {
        "checks_passed": True,
        "excluded_source_episode_count": 1,
        "excluded_source_episodes_sha256": "a" * 64,
        "excluded_source_episodes": [999_999],
        "prior_content_exclusions": {
            name: {"count": 1, "sha256": "b" * 64, "values": ["f" * 64]}
            for name in (
                "action_profile_ids",
                "scene_template_content_hashes",
                "pair_content_hashes",
                "query_pixel_hashes",
            )
        },
    }


def test_public_exclusion_audit_injects_observed_freeze_identity() -> None:
    freeze = {
        "status": "frozen_before_public_generation_or_access",
        "public_exclusions": _exclusions(),
    }
    identity = {"path": "artifacts/freeze.json", "sha256": "1" * 64, "size_bytes": 42}
    audit = public_builder._public_exclusion_audit(
        freeze, freeze_receipt_identity=identity
    )
    assert audit["path"] == identity["path"]
    assert audit["sha256"] == identity["sha256"]
    assert audit["size_bytes"] == identity["size_bytes"]
    assert audit["checks_passed"] is True


def test_public_runtime_patch_is_process_local_and_restored() -> None:
    original_profile_seeds = dict(v4_physics.V4_PROFILE_SPLIT_SEEDS)
    original_splits = development_builder.ACTIVE_SPLITS
    original_catalog_seeds = development_builder.CATALOG_SEEDS
    original_offset = development_builder.FORMAL_CATALOG_INDEX_OFFSET
    original_initializer = development_builder._worker_initialize
    original_acceptance = development_builder._BalancedAcceptance
    with public_builder._public_v4_runtime():
        assert development_builder.ACTIVE_SPLITS == ("validation",)
        assert development_builder.CATALOG_SEEDS == {
            "validation": public_builder.PUBLIC_CATALOG_SEED
        }
        assert (
            development_builder.FORMAL_CATALOG_INDEX_OFFSET
            == public_builder.PUBLIC_CATALOG_INDEX_OFFSET
        )
        assert (
            v4_physics.V4_PROFILE_SPLIT_SEEDS["validation"]
            == public_builder.PUBLIC_PROFILE_SEED
        )
        assert development_builder._worker_initialize is public_builder._public_worker_initialize
        assert (
            development_builder._BalancedAcceptance
            is public_builder._PublicBalancedAcceptance
        )
        tracker = development_builder._BalancedAcceptance(
            public_builder.PUBLIC_PAIR_COUNT
        )
        assert tracker.quota == 64
        assert tracker.counts == {
            "endpoint4": 0,
            "front_hold": 0,
            "plateau": 0,
            "ramp4": 0,
        }
    assert dict(v4_physics.V4_PROFILE_SPLIT_SEEDS) == original_profile_seeds
    assert development_builder.ACTIVE_SPLITS == original_splits
    assert development_builder.CATALOG_SEEDS == original_catalog_seeds
    assert development_builder.FORMAL_CATALOG_INDEX_OFFSET == original_offset
    assert development_builder._worker_initialize is original_initializer
    assert development_builder._BalancedAcceptance is original_acceptance


def test_public_catalog_is_deterministic_balanced_and_prior_disjoint(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "source.h5"
    rows = 2 * public_builder.PUBLIC_PAIR_COUNT
    with h5py.File(source, "w") as handle:
        handle.create_dataset(
            "qpos", data=np.arange(rows * 21, dtype=np.float64).reshape(rows, 21) / 1000
        )
        handle.create_dataset(
            "control", data=np.zeros((rows, 7), dtype=np.float64)
        )
    eligible = [(index, index, 0) for index in range(rows)]
    monkeypatch.setattr(development_builder, "_eligible_source_rows", lambda _: eligible)
    with public_builder._public_v4_runtime():
        first, first_receipt = public_builder.build_public_catalog(
            source,
            source_identity={"path": str(source), "sha256": "0" * 64, "size_bytes": 1},
            exclusions=_exclusions(),
        )
        second, second_receipt = public_builder.build_public_catalog(
            source,
            source_identity={"path": str(source), "sha256": "0" * 64, "size_bytes": 1},
            exclusions=_exclusions(),
        )
    assert first == second
    assert first_receipt == second_receipt
    assert len(first) == rows
    assert len({row.source_episode for row in first}) == rows
    assert first_receipt["catalog_action_anchor_counts"] == {
        "endpoint4": 128,
        "front_hold": 128,
        "plateau": 128,
        "ramp4": 128,
    }
    assert all(row.split == "validation" for row in first)
    assert all(
        row.catalog_index >= public_builder.PUBLIC_CATALOG_INDEX_OFFSET
        for row in first
    )


def test_public_generation_success_binds_authorization_identities(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prereg = tmp_path / "prereg.yaml"
    freeze = tmp_path / "freeze.json"
    source = tmp_path / "source.h5"
    prereg.write_text("frozen: true\n", encoding="utf-8")
    freeze.write_text('{"frozen": true}\n', encoding="utf-8")
    source.write_bytes(b"source")
    output = tmp_path / "public"
    prereg_logical_path = "configs/benchmark/public_recovery.yaml"
    freeze_logical_path = "artifacts/evaluation/public_recovery/freeze.json"
    authorization = SimpleNamespace(
        public_root=output,
        preregistration_path=prereg,
        freeze_receipt_path=freeze,
        preregistration={
            "identity": {"preregistration_path": prereg_logical_path}
        },
        freeze_receipt={
            "frozen_inputs": {
                "source_h5": {"path": str(source), "sha256": "0" * 64}
            }
        },
        freeze_receipt_identity={
            "path": freeze_logical_path,
            "sha256": "1" * 64,
            "size_bytes": freeze.stat().st_size,
        },
    )
    monkeypatch.setattr(
        public_builder, "load_public_authorization", lambda **_: authorization
    )
    monkeypatch.setattr(
        public_builder,
        "_source_identity",
        lambda *_: {"path": str(source), "sha256": "0" * 64, "size_bytes": 6},
    )
    monkeypatch.setattr(
        public_builder,
        "_public_exclusion_audit",
        lambda *_args, **_kwargs: _exclusions(),
    )
    monkeypatch.setattr(
        public_builder,
        "build_public_catalog",
        lambda *_args, **_kwargs: ([], {"candidate_pool_count": 0}),
    )
    monkeypatch.setattr(
        development_builder,
        "build_split",
        lambda *_args, **_kwargs: {
            "prior_episode_and_content_exclusion": {
                "accepted_overlap": {
                    "source_episode_count": 0,
                    "action_profile_id_count": 0,
                    "scene_template_content_hash_count": 0,
                    "pair_content_hash_count": 0,
                    "query_pixel_hash_count": 0,
                }
            },
            "all_causal_checks_passed": True,
            "fresh_simulator_replay": {"passed": True},
            "maximum_query_physical_gap": 0.0,
            "maximum_query_simulator_state_gap": 0.0,
            "maximum_state_installations_after_x0": 0,
            "passed": True,
        },
    )
    captured: dict[str, dict] = {}

    def fake_publish(staged: Path, _output: Path, *, success_payload: dict) -> dict:
        captured["request"] = json.loads(
            (staged / "request.json").read_text(encoding="utf-8")
        )
        captured["success"] = dict(success_payload)
        return {"success_marker": {}, "published_tree": {}}

    monkeypatch.setattr(public_builder, "_publish", fake_publish)

    result = public_builder.build_public_data(
        source=source,
        preregistration=prereg,
        freeze_receipt=freeze,
        output=output,
        staging_root=Path("/tmp"),
        workers=16,
        jpeg_quality=95,
    )

    expected_preregistration = public_builder.file_identity(
        prereg, logical_path=prereg_logical_path
    )
    expected_freeze = public_builder.file_identity(
        freeze, logical_path=public_builder.portable_contextworld_path(freeze)
    )
    assert result["status"] == "public_data_generated_not_model_scored"
    assert captured["request"]["preregistration"] == expected_preregistration
    assert captured["request"]["freeze_receipt"] == expected_freeze
    assert captured["success"]["preregistration"] == expected_preregistration
    assert captured["success"]["freeze_receipt"] == expected_freeze


def test_public_generation_failure_persists_consumed_namespace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prereg = tmp_path / "prereg.yaml"
    freeze = tmp_path / "freeze.json"
    source = tmp_path / "source.h5"
    prereg.write_text("frozen: true\n", encoding="utf-8")
    freeze.write_text('{"frozen": true}\n', encoding="utf-8")
    source.write_bytes(b"source")
    output = tmp_path / "public"
    authorization = SimpleNamespace(
        public_root=output,
        preregistration_path=prereg,
        freeze_receipt_path=freeze,
        preregistration={"identity": {"preregistration_path": str(prereg)}},
        freeze_receipt={
            "frozen_inputs": {
                "source_h5": {"path": str(source), "sha256": "0" * 64}
            }
        },
        freeze_receipt_identity={
            "path": str(freeze),
            "sha256": "1" * 64,
            "size_bytes": freeze.stat().st_size,
        },
    )
    monkeypatch.setattr(
        public_builder, "load_public_authorization", lambda **_: authorization
    )
    monkeypatch.setattr(
        public_builder,
        "_source_identity",
        lambda *_: (_ for _ in ()).throw(RuntimeError("forced source failure")),
    )
    with pytest.raises(RuntimeError, match="forced source failure"):
        public_builder.build_public_data(
            source=source,
            preregistration=prereg,
            freeze_receipt=freeze,
            output=output,
            staging_root=Path("/tmp"),
            workers=16,
            jpeg_quality=95,
        )
    started = output / public_builder.GENERATION_STARTED_MARKER
    failure = output / public_builder.GENERATION_FAILURE_MARKER
    assert started.is_file()
    assert failure.is_file()
    payload = json.loads(failure.read_text(encoding="utf-8"))
    assert payload["rerun_authorized"] is False
    assert payload["public_model_read"] is False
    with pytest.raises(FileExistsError):
        public_builder.build_public_data(
            source=source,
            preregistration=prereg,
            freeze_receipt=freeze,
            output=output,
            staging_root=Path("/tmp"),
            workers=16,
            jpeg_quality=95,
        )
