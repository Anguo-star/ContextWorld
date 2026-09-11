"""Small, local contract tests for the current reference freeze entry point.

These tests use tiny synthetic result envelopes.  They exercise the evidence
and verification paths without copying the real 270-result handoff or
touching model weights.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/freeze_current_reference_baseline.py"


def _module():
    spec = importlib.util.spec_from_file_location("current_reference_freeze", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _result(path: Path, *, score: float, checkpoint: Path) -> None:
    payload = {
        "result": {
            "schema_version": 1,
            "bundle": {
                "component_id": "speed",
                "dataset_id": "fixture-speed",
                "manifest_sha256": "f" * 64,
                "members": [],
            },
            "model": {
                "adapter": {
                    "adapter_class": "contextworld.benchmarks.adapters.StableWorldModelLeWMAdapter",
                    "adapter_id": "fixture_adapter",
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": _sha(checkpoint),
                    "training_seed": 3072,
                }
            },
            "metrics": {
                "tracks": {
                    "unseen_interpolation": {
                        "horizons": {
                            "1": {
                                "reference_speed_balanced_strict_query_win_rate_vs_every_other": score
                            }
                        }
                    }
                }
            },
            "selection": {"queries": 2, "eval_seeds": [42]},
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _fixture(tmp_path: Path, *, overlay: bool = False):
    module = _module()
    checkpoint = tmp_path / "weights.pt"
    checkpoint.write_bytes(b"tiny fixture checkpoint")
    source = tmp_path / "source.json"
    _result(source, score=0.25, checkpoint=checkpoint)
    overlay_path = None
    replacement = None
    if overlay:
        replacement = tmp_path / "replacement.json"
        _result(replacement, score=0.75, checkpoint=checkpoint)
        overlay_path = tmp_path / "overlay.json"
        overlay_path.write_text(
            json.dumps(
                {
                    "rows": [
                        {
                            "component_id": "speed",
                            "family": "LeWM",
                            "training_seed": 3072,
                            "stage": "post_component_training",
                            "split": "development",
                            "path": str(replacement),
                            "reason": "fixture replacement",
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
    v1 = tmp_path / "v1.json"
    v1.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "recipe": "joint_scratch_v1",
                "main_score_fields": {"speed": "fixture"},
                "checkpoint_results": [
                    {
                        "component_id": "speed",
                        "family": "LeWM",
                        "stage": "post_component_training",
                        "training_seed": 3072,
                        "development": {"main_score": 0.25, "all_gates_passed": None},
                        "test": {"main_score": 0.25, "all_gates_passed": False},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    audit = tmp_path / "audit.json"
    audit.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "component_id": "speed",
                        "family": "LeWM",
                        "training_seed": 3072,
                        "stage": "post_component_training",
                        "split": split,
                        "path": str(source),
                    }
                    for split in ("development", "test")
                ]
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "v2.json"
    return module, v1, audit, output, overlay_path, source, replacement


def test_build_copies_raw_snapshot_and_verify_uses_snapshot(tmp_path: Path) -> None:
    module, v1, audit, output, _, source, _ = _fixture(tmp_path)
    frozen = module.build_freeze(
        v1_path=v1,
        audit_path=audit,
        output=output,
        repo_root=tmp_path,
        strict=False,
        write=True,
    )
    development = frozen["checkpoint_results"][0]["development"]
    snapshot = tmp_path / development["source_result_path"]
    assert snapshot.is_file()
    assert development["raw_snapshot"]["origin_path"] == str(source)
    assert development["all_gates_passed"] is False
    assert module.verify_freeze(freeze_path=output, repo_root=tmp_path)["status"] == "verified"

    # A later mutation of the external origin cannot destroy the repository
    # snapshot or change the verified score.
    source.write_text(source.read_text(encoding="utf-8").replace("0.25", "0.99"), encoding="utf-8")
    assert module.verify_freeze(freeze_path=output, repo_root=tmp_path)["status"] == "verified"

    snapshot.write_text(snapshot.read_text(encoding="utf-8").replace("0.25", "0.99"), encoding="utf-8")
    with pytest.raises(module.FreezeError, match="source SHA drifted"):
        module.verify_freeze(freeze_path=output, repo_root=tmp_path)


@pytest.mark.parametrize("field", ["all_gates_passed", "cleared_development"])
def test_frozen_pass_or_admission_cannot_disagree_with_evidence(tmp_path: Path, field: str) -> None:
    module, v1, audit, output, _, _, _ = _fixture(tmp_path)
    frozen = module.build_freeze(
        v1_path=v1, audit_path=audit, output=output, repo_root=tmp_path, strict=False,
    )
    row = frozen["checkpoint_results"][0]
    if field == "all_gates_passed":
        row["development"][field] = True
    else:
        row["admission"][field] = True
    output.write_text(json.dumps(frozen))
    with pytest.raises(module.FreezeError, match="disagrees"):
        module.verify_freeze(freeze_path=output, repo_root=tmp_path)


def test_overlay_is_recorded_but_snapshot_is_authoritative(tmp_path: Path) -> None:
    module, v1, audit, output, overlay, _, replacement = _fixture(tmp_path, overlay=True)
    frozen = module.build_freeze(
        v1_path=v1,
        audit_path=audit,
        result_overlay=overlay,
        output=output,
        repo_root=tmp_path,
        strict=False,
        write=True,
    )
    development = frozen["checkpoint_results"][0]["development"]
    assert development["main_score"] == 0.75
    assert development["overlay"] is not None
    assert development["raw_snapshot"]["origin_path"] == str(replacement)
    assert module.verify_freeze(freeze_path=output, repo_root=tmp_path)["status"] == "verified"

    # Changing the replacement after freeze does not affect verification.
    replacement.write_text(replacement.read_text(encoding="utf-8").replace("0.75", "0.11"), encoding="utf-8")
    assert module.verify_freeze(freeze_path=output, repo_root=tmp_path)["status"] == "verified"

    # The overlay manifest itself is an input identity and is checked.
    overlay.write_text(overlay.read_text(encoding="utf-8").replace("fixture replacement", "edited"), encoding="utf-8")
    with pytest.raises(module.FreezeError, match="overlay SHA drifted"):
        module.verify_freeze(freeze_path=output, repo_root=tmp_path)


def test_existing_different_output_is_not_silently_replaced(tmp_path: Path) -> None:
    module, v1, audit, output, _, _, _ = _fixture(tmp_path)
    output.write_text(json.dumps({"freeze_id": "other"}), encoding="utf-8")
    with pytest.raises(module.FreezeError, match="different content"):
        module.build_freeze(
            v1_path=v1,
            audit_path=audit,
            output=output,
            repo_root=tmp_path,
            strict=False,
            write=True,
        )


def _native_fixture(tmp_path: Path, module: Any, monkeypatch: pytest.MonkeyPatch):
    """Make a small native validation release with real table/manifest bytes."""

    artifact = tmp_path / "artifact"
    table = artifact / "validation.lance"
    table.mkdir(parents=True)
    (table / "data.bin").write_bytes(b"native-table-bytes")
    manifest = artifact / "manifest.json"
    manifest.write_bytes(b'{"release":"fixture"}\n')
    table_sha = module._native_table_sha256(table)[0]
    manifest_sha = _sha(manifest)
    config = {
        "release_id": "fixture-native-release",
        "data": {
            "artifact_tree": {"root": "artifact"},
            "pair_counts": {"validation": 256},
            "lance_tables": {"validation": "validation.lance"},
            "table_sha256": {"validation": table_sha},
            "manifest_sha256": manifest_sha,
            "artifacts": {"manifest": {"path": "manifest.json", "sha256": manifest_sha}},
        },
    }
    config_path = tmp_path / "release.yaml"
    # JSON is valid YAML and keeps this fixture independent of a serializer.
    config_path.write_text(json.dumps(config), encoding="utf-8")
    payload = {
        "bundle": {
            "artifact_root": str(artifact),
            "selection_policy": "fixture_public_test",
        },
        "result": {
            "data": {"root": str(artifact), "pair_count": 256, "condition_count": 512},
            "release": {
                "release_id": "fixture-native-release",
                "data_manifest_sha256": manifest_sha,
            },
        },
        "selection_policy": "fixture_public_test",
    }
    monkeypatch.setitem(module.RELEASE_CONFIG, "action_strength", str(config_path))
    return payload, artifact, table, config_path


def test_native_binding_hashes_actual_table_and_rejects_table_byte_drift(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module()
    payload, _, table, _ = _native_fixture(tmp_path, module, monkeypatch)
    binding = module._native_validation_identity(
        task="action_strength",
        result=payload["result"],
        payload=payload,
        root=tmp_path,
        strict=True,
    )
    assert binding["status"] == "verified"
    assert binding["selection"]["pair_count"] == 256
    assert binding["members"][0]["status"] == "verified_native_table"
    table_sha = binding["members"][0]["sha256"]
    assert table_sha == module._native_table_sha256(table)[0]

    (table / "data.bin").write_bytes(b"changed-table-bytes")
    with pytest.raises(module.FreezeError, match="table SHA disagrees"):
        module._native_validation_identity(
            task="action_strength",
            result=payload["result"],
            payload=payload,
            root=tmp_path,
            strict=True,
        )


def test_native_binding_rejects_conflicting_result_manifest_claims(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module()
    payload, _, _, _ = _native_fixture(tmp_path, module, monkeypatch)
    payload["result"]["release"]["confirmation_manifest_sha256"] = "b" * 64
    with pytest.raises(module.FreezeError, match="result manifest identities conflict"):
        module._native_validation_identity(
            task="action_strength",
            result=payload["result"],
            payload=payload,
            root=tmp_path,
            strict=True,
        )


def test_native_binding_rejects_selection_member_drift(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module()
    payload, _, _, config_path = _native_fixture(tmp_path, module, monkeypatch)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["data"]["pair_counts"]["validation"] = 255
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(module.FreezeError, match="pair count disagrees"):
        module._native_validation_identity(
            task="action_strength",
            result=payload["result"],
            payload=payload,
            root=tmp_path,
            strict=True,
        )


def test_native_package_manifest_must_list_every_table_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module()
    payload, artifact, _, _ = _native_fixture(tmp_path, module, monkeypatch)
    (artifact / "manifest.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(module.FreezeError, match="absent from package manifest"):
        module._native_validation_identity(
            task="action_strength",
            result=payload["result"],
            payload=payload,
            root=tmp_path,
            strict=True,
        )


def test_native_provenance_sha_mismatch_is_not_waived_by_package_chain(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module()
    payload, artifact, _, config_path = _native_fixture(tmp_path, module, monkeypatch)
    provenance = artifact / "portable_provenance.json"
    provenance.write_bytes(b"actual provenance")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected = "a" * 64
    config["data"]["artifacts"]["portable_provenance"] = {"path": "portable_provenance.json", "sha256": expected}
    config_path.write_text(json.dumps(config), encoding="utf-8")
    payload["result"]["release"]["portable_provenance_sha256"] = expected
    with pytest.raises(module.FreezeError, match="provenance SHA disagrees"):
        module._native_validation_identity(
            task="action_strength",
            result=payload["result"],
            payload=payload,
            root=tmp_path,
            strict=True,
        )


def test_v3_requires_parent_and_decision_contract_for_production_receipt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module, v1, audit, output, _, _, _ = _fixture(tmp_path)
    frozen = module.build_freeze(
        v1_path=v1,
        audit_path=audit,
        output=output,
        repo_root=tmp_path,
        strict=False,
        write=True,
    )
    frozen["coverage"]["checkpoint_rows"] = 135
    output.write_text(json.dumps(frozen), encoding="utf-8")
    with pytest.raises(module.FreezeError, match="v3 parent freeze identity is missing"):
        module.verify_freeze(freeze_path=output, repo_root=tmp_path)

    parent = tmp_path / "parent.json"
    parent.write_text("{}", encoding="utf-8")
    frozen["inputs"]["parent_freeze"] = module.identity(parent, root=tmp_path)
    output.write_text(json.dumps(frozen), encoding="utf-8")
    with pytest.raises(module.FreezeError, match="v3 decision contract identity is missing"):
        module.verify_freeze(freeze_path=output, repo_root=tmp_path)


def test_v3_production_inventory_rejects_critical_source_drift(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module, v1, audit, output, _, _, _ = _fixture(tmp_path)
    frozen = module.build_freeze(
        v1_path=v1,
        audit_path=audit,
        output=output,
        repo_root=tmp_path,
        strict=False,
        write=True,
    )
    parent = tmp_path / "parent.json"
    parent.write_text("{}", encoding="utf-8")
    frozen["inputs"]["parent_freeze"] = module.identity(parent, root=tmp_path)
    monkeypatch.setattr(module, "_decision_contract_identity", lambda *, root: {})
    frozen["inputs"]["decision_contract"] = {}
    source_identities = []
    for relative in module.EVALUATION_SOURCE_PATHS:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")
        source_identities.append(module.identity(path, root=tmp_path))
    frozen["inputs"]["evaluation_sources"] = source_identities
    output.write_text(json.dumps(frozen), encoding="utf-8")
    assert module.verify_freeze(freeze_path=output, repo_root=tmp_path)["status"] == "verified"

    drifted = tmp_path / module.EVALUATION_SOURCE_PATHS[0]
    drifted.write_text("changed", encoding="utf-8")
    with pytest.raises(module.FreezeError, match="evaluation source drifted"):
        module.verify_freeze(freeze_path=output, repo_root=tmp_path)
