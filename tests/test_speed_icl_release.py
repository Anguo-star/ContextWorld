from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from contextworld.benchmarks.adapters import (
    AdapterProtocol,
    SpeedICLModelAdapter,
)
from contextworld.benchmarks import speed_icl_cli
from contextworld.benchmarks.speed_icl_data import (
    SpeedICLEvalDataset,
    _array_sha256,
    _audit_file,
    _sha256,
    _tree_fingerprint,
    export_speed_icl_artifacts,
    load_speed_icl_release,
)
from contextworld.benchmarks.speed_release_shadow import (
    SpeedMetadataPathError,
    absolute_json_paths,
    audit_speed_portable_shadow,
    copy_speed_catalog_shadow,
    sanitize_speed_release_metadata,
)
from contextworld.benchmarks.speed_icl_score import (
    aggregate_speed_icl_method,
    aggregate_speed_icl_planning,
    evaluate_speed_icl_model,
)


class FakeAdapter(SpeedICLModelAdapter):
    @property
    def protocol(self) -> AdapterProtocol:
        return AdapterProtocol(
            history_tokens=3,
            action_block_raw_steps=5,
            action_dim=2,
            future_action_blocks=5,
        )

    @property
    def metadata(self):
        return {"adapter_id": "fake", "checkpoint_sha256": "fake"}

    def encode_pixels(self, pixels, *, batch_size):
        del batch_size
        return np.asarray(pixels, dtype=np.float32).mean(
            axis=(1, 2, 3), keepdims=False
        )[:, None]

    def rollout_latents(
        self, input_pixels, raw_action_blocks, *, batch_size
    ):
        del batch_size
        future = raw_action_blocks.shape[1] - 2
        speed_signal = np.asarray(input_pixels, dtype=np.float32)[
            :, 0
        ].mean(axis=(1, 2, 3))
        return np.repeat(speed_signal[:, None, None], future, axis=1)

    def frozen_state_hash(self) -> str:
        return "frozen"


def _make_release(tmp_path: Path) -> Path:
    artifact_root = tmp_path / "artifact_root"
    payload_dir = artifact_root / "eval/payloads/track"
    payload_dir.mkdir(parents=True)
    query = np.zeros((2, 2, 3), dtype=np.uint8)
    targets = np.full((5, 2, 2, 3), 10, dtype=np.uint8)
    future_actions = np.zeros((5, 5, 2), dtype=np.float32)
    payload_values = {
        "query_pixels": query,
        "future_actions": future_actions,
        "future_pixels": np.concatenate([query[None], targets[:-1]], axis=0),
        "future_next_pixels": targets,
    }
    conditions = {"history_low": 10, "history_mid": 20, "history_high": 30}
    for condition, value in conditions.items():
        prefix = f"context_b2_{condition}"
        pixels = np.full((2, 2, 2, 3), value, dtype=np.uint8)
        next_pixels = np.stack([pixels[1], query])
        payload_values[f"{prefix}_pixels"] = pixels
        payload_values[f"{prefix}_actions"] = np.zeros(
            (2, 5, 2), dtype=np.float32
        )
        payload_values[f"{prefix}_next_pixels"] = next_pixels
    payload_path = payload_dir / "q0.npz"
    np.savez(payload_path, **payload_values)
    bundle = {
        "query_id": "q0",
        "static_query_id": "static0",
        "track": "seen_for_multi",
        "query_factors": {"agent.speed": 10.0},
        "conditions": {
            key: {"factors": {"agent.speed": float(value)}}
            for key, value in conditions.items()
        },
        "matching_condition": "history_low",
        "query_action_family": "test",
        "payload": str(payload_path),
        "payload_sha256": _sha256(payload_path),
        "query_pixels_sha256": _array_sha256(query),
        "future_actions_sha256": _array_sha256(future_actions),
        "target_pixels_sha256_by_horizon": {
            str(horizon): _array_sha256(targets[horizon - 1])
            for horizon in (1, 2, 3, 5)
        },
        "eval_seed": 42,
        "evaluation_index": 0,
    }
    catalog = {
        "track": "seen_for_multi",
        "summary": {
            "passed": True,
            "history_conditions": list(conditions),
        },
        "bundles": [bundle],
    }
    catalog_path = artifact_root / "eval/catalog.json"
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
    release = {
        "schema_version": 1,
        "release_id": "contextworld_tworoom_speed_icl_history3_v1",
        "scope": {
            "public_tracks": ["seen_for_multi"],
            "history_tokens": 3,
            "action_block_raw_steps": 5,
        },
        "training": {"paired_training_seeds": [1, 2, 3]},
        "runtime": {
            "stable_worldmodel": {"expected_ref": "fake-stablewm"}
        },
        "scoring": {
            "core_claim_tracks": ["seen_for_multi"],
            "extrapolation_tracks": [],
        },
        "evaluation": {
            "normalizer_sha256": "fake-normalizer",
            "eval_seeds": [42],
            "queries_per_reference_speed_per_seed": 1,
            "tracks": {
                "seen_for_multi": {
                    "speeds": [10.0],
                    "catalog": str(catalog_path),
                    "catalog_sha256": _sha256(catalog_path),
                }
            },
        },
        "planning": {
            "eval_seeds": [42],
            "evaluations_per_speed_condition_per_seed": 1,
            "tracks": {
                "seen_for_multi": {
                    "speeds": [10.0],
                    "catalog_sha256": "fake-planning-catalog",
                },
            },
            "fixed_candidate": {
                "candidates": 300,
                "horizon_action_blocks": 10,
            },
            "cem": {"deadline_budgets_raw_steps": [50, 75, 100]},
        },
    }
    release_path = tmp_path / "release.yaml"
    release_path.write_text(yaml.safe_dump(release), encoding="utf-8")
    return release_path


def test_public_release_config_loads() -> None:
    release = load_speed_icl_release()
    assert release["release_status"] == "public_test_release_candidate"
    assert release["runtime"]["supported_adapters"] == [
        "stable_worldmodel_lewm",
        "stable_worldmodel_pldm",
    ]
    assert release["scope"]["public_test_included"] is True
    assert release["scope"]["sealed_test_included"] is False


def test_speed_cli_keeps_lewm_default_and_routes_pldm(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class FakeLeWM:
        @classmethod
        def from_checkpoint(cls, *args, **kwargs):
            del args, kwargs
            return "lewm"

    class FakePLDM:
        @classmethod
        def from_checkpoint(cls, *args, **kwargs):
            del args, kwargs
            return "pldm"

    monkeypatch.setattr(speed_icl_cli, "StableWorldModelLeWMAdapter", FakeLeWM)
    monkeypatch.setattr(speed_icl_cli, "StableWorldModelPLDMAdapter", FakePLDM)
    monkeypatch.setattr(
        speed_icl_cli,
        "load_speed_icl_release",
        lambda _: {
            "runtime": {
                "stable_worldmodel": {
                    "repo": "stable-worldmodel",
                    "expected_ref": "test-ref",
                }
            },
            "evaluation": {"normalizer": "normalizer.json"},
        },
    )
    monkeypatch.setattr(
        speed_icl_cli,
        "resolve_contextworld_path",
        lambda *args, **kwargs: tmp_path / "normalizer.json",
    )
    common = [
        "eval",
        "--checkpoint",
        "checkpoint.pt",
        "--model-name",
        "external-baseline",
        "--output",
        "result.json",
    ]
    legacy = speed_icl_cli.parse_args(common)
    pldm = speed_icl_cli.parse_args([*common, "--adapter", "pldm"])

    assert legacy.adapter == "lewm"
    assert speed_icl_cli._adapter(legacy) == "lewm"
    assert speed_icl_cli._adapter(pldm) == "pldm"


def test_training_tree_fingerprint_detects_content_changes(
    tmp_path: Path,
) -> None:
    root = tmp_path / "tree"
    root.mkdir()
    (root / "a.bin").write_bytes(b"abc")
    (root / "b.bin").write_bytes(b"defg")
    first = _tree_fingerprint(root, hash_contents=True)
    assert first["files"] == 2
    assert first["bytes"] == 7
    assert first["full_hash_verified"] is True
    (root / "b.bin").write_bytes(b"DEFG")
    second = _tree_fingerprint(root, hash_contents=True)
    assert second["bytes"] == first["bytes"]
    assert second["sha256"] != first["sha256"]


def test_speed_metadata_sanitizer_is_fail_closed(tmp_path: Path) -> None:
    repo = tmp_path / "ContextWorld"
    canonical = tmp_path / "context_world"
    original = tmp_path / "quentinll/tworoom.h5"
    stablewm = tmp_path / "stable-worldmodel"
    value = {
        "artifact": str(canonical / "evaluation/result.json"),
        "config": str(repo / "configs/release.yaml"),
        "original": str(original),
        "stablewm": str(stablewm),
        "metric": 0.25,
    }
    portable = sanitize_speed_release_metadata(
        value,
        repo_root=repo,
        canonical_artifact_root=canonical,
        original_h5=original,
        stable_worldmodel_root=stablewm,
    )
    assert portable == {
        "artifact": "artifacts/evaluation/result.json",
        "config": "configs/release.yaml",
        "original": "upstream/lewm-tworooms/tworoom.h5",
        "stablewm": "upstream/stable-worldmodel",
        "metric": 0.25,
    }
    with pytest.raises(SpeedMetadataPathError, match="unknown absolute"):
        sanitize_speed_release_metadata(
            {"bad": str(tmp_path / "unknown/file.json")},
            repo_root=repo,
            canonical_artifact_root=canonical,
            original_h5=original,
            stable_worldmodel_root=stablewm,
        )


def test_catalog_shadow_is_complete_and_preserves_payloads(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "ContextWorld"
    canonical = tmp_path / "context_world"
    original = tmp_path / "quentinll/tworoom.h5"
    stablewm = tmp_path / "stable-worldmodel"
    source = canonical / "evaluation/catalogs"
    source.mkdir(parents=True)
    report = {
        "config": {"path": str(repo / "configs/release.yaml")},
        "catalog": str(source / "track.json"),
        "stable_worldmodel": {"repo": str(stablewm)},
        "score": 0.75,
    }
    (source / "build_report.json").write_text(
        json.dumps(report), encoding="utf-8"
    )
    payload = {"bundles": [], "passed": True}
    (source / "track.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    source_payload_hash = _sha256(source / "track.json")
    target = repo / "artifacts/evaluation/catalogs"
    result = copy_speed_catalog_shadow(
        source,
        target,
        repo_root=repo,
        canonical_artifact_root=canonical,
        original_h5=original,
        stable_worldmodel_root=stablewm,
    )
    assert sorted(path.name for path in target.iterdir()) == [
        "build_report.json",
        "track.json",
    ]
    assert _sha256(target / "track.json") == source_payload_hash
    assert result["payload_sha256"] == {"track.json": source_payload_hash}
    assert absolute_json_paths(
        json.loads((target / "build_report.json").read_text())
    ) == []
    assert len(
        absolute_json_paths(
            json.loads((source / "build_report.json").read_text())
        )
    ) == 3


def test_file_audit_reports_logical_path(tmp_path: Path) -> None:
    path = tmp_path / "artifacts/value.json"
    path.parent.mkdir(parents=True)
    path.write_text("{}\n", encoding="utf-8")
    audit = _audit_file(
        "artifacts/value.json", _sha256(path), repo_root=tmp_path
    )
    assert audit["path"] == "artifacts/value.json"
    assert audit["passed"] is True


def test_export_inventory_contains_only_portable_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    data_root = repo / "artifacts/synthetic/data"
    data_root.mkdir(parents=True)
    (data_root / "payload.bin").write_bytes(b"payload")
    files = {
        "artifacts/synthetic/catalog.json": "{}\n",
        "artifacts/synthetic/manifest.jsonl": "{}\n",
        "artifacts/synthetic/report.json": "{}\n",
        "artifacts/splits/normalizer.json": "{}\n",
    }
    for logical, content in files.items():
        path = repo / logical
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    for logical in (
        "artifacts/evaluation/history3/"
        "speed_multistep_extrap_v5/catalogs",
        "artifacts/evaluation/history3/"
        "speed_multistep_extrap_v5/payloads",
    ):
        path = repo / logical
        path.mkdir(parents=True)
        (path / "value.json").write_text("{}\n", encoding="utf-8")
    release_path = repo / "configs/release.yaml"
    release_path.parent.mkdir(parents=True)
    release_path.write_text("schema_version: 1\n", encoding="utf-8")
    release = {
        "release_id": "contextworld_tworoom_speed_icl_history3_v1",
        "_config_path": str(release_path),
        "training": {
            "original": {"source": "https://example.test/tworoom.h5"},
            "synthetic": {
                "multi_speed_target": {
                    "data_root": "artifacts/synthetic/data",
                    "catalog": "artifacts/synthetic/catalog.json",
                    "manifest": "artifacts/synthetic/manifest.jsonl",
                    "report": "artifacts/synthetic/report.json",
                    "data_tree_files": 1,
                    "data_tree_bytes": 7,
                    "data_tree_sha256": "unused",
                }
            },
        },
        "evaluation": {"normalizer": "artifacts/splits/normalizer.json"},
        "reference_results": {},
    }
    monkeypatch.setattr(
        "contextworld.benchmarks.speed_icl_data.load_speed_icl_release",
        lambda _path: release,
    )
    destination = tmp_path / "export"
    result = export_speed_icl_artifacts(
        destination,
        release_config=release_path,
        repo_root=repo,
        include_single_speed_control=False,
    )
    inventory = json.loads(
        (destination / "release/inventory.json").read_text(encoding="utf-8")
    )
    assert result["artifact_root"] == "."
    assert result["inventory"] == "release/inventory.json"
    assert absolute_json_paths(inventory) == []


def test_formal_speed_shadow_is_portable_and_exact() -> None:
    release = load_speed_icl_release()
    root = Path(__file__).resolve().parents[1]
    audit = audit_speed_portable_shadow(release, repo_root=root)
    assert audit["owned_file_tree"]["missing"] == []
    assert audit["owned_file_tree"]["files"] == 19
    for name in (
        "tworoom_speed_single_matched_v2.jsonl",
        "tworoom_speed_full_v1.jsonl",
    ):
        assert (root / "artifacts/synthesis/manifests" / name).is_file()
    assert audit["absolute_json_paths"] == []
    assert all(
        row["passed"] for row in audit["catalog_directories"].values()
    )
    assert audit["formal_results_only"] is True


def test_lazy_eval_dataset_and_model_score(tmp_path: Path) -> None:
    release_path = _make_release(tmp_path)
    dataset = SpeedICLEvalDataset(
        release_config=release_path,
        track="seen_for_multi",
        repo_root=tmp_path,
    )
    assert len(dataset) == 1
    bundle = dataset[0]
    assert set(bundle.histories) == {
        "history_low",
        "history_mid",
        "history_high",
    }
    assert bundle.histories["history_low"].input_pixels.shape == (3, 2, 2, 3)
    result = evaluate_speed_icl_model(
        adapter=FakeAdapter(),
        model_name="fake-target",
        training_role="multi_speed_target",
        training_seed=1,
        release_config=release_path,
        repo_root=tmp_path,
        bundle_batch_size=1,
    )
    assert result["full_protocol"] is True
    h1 = result["tracks"]["seen_for_multi"]["horizons"]["1"]
    assert h1["formal_within_checkpoint_pass"] is True
    assert h1[
        "reference_speed_balanced_matching_to_other_loss_ratio"
    ] == 0.0
    assert h1[
        "reference_speed_balanced_strict_query_win_rate_vs_every_other"
    ] == 1.0


def test_complete_method_aggregation_requires_paired_seeds(
    tmp_path: Path,
) -> None:
    release_path = _make_release(tmp_path)
    base = evaluate_speed_icl_model(
        adapter=FakeAdapter(),
        model_name="fake",
        training_role="multi_speed_target",
        training_seed=1,
        release_config=release_path,
        repo_root=tmp_path,
        bundle_batch_size=1,
        include_records=False,
    )
    target_paths = []
    control_paths = []
    for seed in (1, 2, 3):
        target = json.loads(json.dumps(base))
        target["model"]["training_seed"] = seed
        target_path = tmp_path / f"target-{seed}.json"
        target_path.write_text(json.dumps(target), encoding="utf-8")
        target_paths.append(target_path)

        control = json.loads(json.dumps(target))
        control["model"]["training_role"] = "single_speed_control"
        for horizon in ("1", "2", "3", "5"):
            row = control["tracks"]["seen_for_multi"]["horizons"][horizon]
            row["reference_speed_balanced_relative_loss_reduction"] = -0.1
        control_path = tmp_path / f"control-{seed}.json"
        control_path.write_text(json.dumps(control), encoding="utf-8")
        control_paths.append(control_path)
    method = aggregate_speed_icl_method(
        target_results=target_paths,
        control_results=control_paths,
        method_name="fake-method",
        release_config=release_path,
    )
    assert method["formal_claim_level"] == "training_attributed_speed_icl"
    assert method["decision"]["longest_contiguous_passing_horizon_by_track"] == {
        "seen_for_multi": 5
    }


def test_planning_support_is_aggregated_without_becoming_primary_claim(
    tmp_path: Path,
) -> None:
    release_path = _make_release(tmp_path)
    result = {
        "schema_version": 1,
        "benchmark": "tworoom_history3_speed_fixed_candidate_v2",
        "status": "passed",
        "track": "seen_for_multi",
        "query_speed": 10.0,
        "eval_seed": 42,
        "count_audit": {
            "passed": True,
            "records": 1,
        },
        "contextworld_release": {
            "release_id": "contextworld_tworoom_speed_icl_history3_v1",
            "release_config_sha256": _sha256(release_path),
            "planning_mode": "fixed_candidate",
            "catalog_sha256": "fake-planning-catalog",
        },
        "model": {"sha256": "fake-model"},
        "normalizer": {"sha256": "fake-normalizer"},
        "stable_worldmodel": {"commit": "fake-stablewm"},
        "frozen_weight_audit": {"passed": True},
        "protocol": {
            "action_block": 5,
            "history_size": 3,
            "candidates": 300,
            "horizon_action_blocks": 10,
            "same_candidate_bank_across_conditions": True,
            "regret_uses_exact_query_dynamics": True,
        },
        "records": [
            {
                "conditions": {
                    "history_low": {
                        "history_speed": 10.0,
                        "history_relation": "same",
                        "exact_query_dynamics_regret_px": 2.0,
                        "cost_vs_true_distance_spearman": 0.8,
                    },
                    "history_high": {
                        "history_speed": 20.0,
                        "history_relation": "faster",
                        "exact_query_dynamics_regret_px": 5.0,
                        "cost_vs_true_distance_spearman": 0.3,
                    },
                }
            }
        ],
    }
    path = tmp_path / "fixed.json"
    path.write_text(json.dumps(result), encoding="utf-8")
    summary = aggregate_speed_icl_planning(
        result_paths=[path], release_config=release_path
    )
    assert summary["full_protocol"] is True
    assert summary["formal_claim_level"] == "supporting_utility_metrics"
    assert summary["tracks"]["seen_for_multi"]["10.0"]["conditions"][
        "history_low"
    ]["mean_exact_query_dynamics_regret_px"] == 2.0
