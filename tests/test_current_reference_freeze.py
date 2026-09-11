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
