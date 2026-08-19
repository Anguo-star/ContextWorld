from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/recover_tworoom_speed_pldm_training_interruption_v1.py"
PREREGISTRATION = (
    ROOT
    / "configs/benchmark/"
    "tworoom_speed_pldm_training_interruption_recovery_v1.yaml"
)


def _module():
    specification = importlib.util.spec_from_file_location(
        "speed_training_interruption_recovery_test", SCRIPT
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


recovery = _module()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _payload(steps: list[int]) -> bytes:
    return b"".join(
        (
            json.dumps(
                {
                    "schema_version": 1,
                    "optimizer_step": step,
                    "losses": {"loss": 1.0, "pred_loss": 1.0},
                }
            )
            + "\n"
        ).encode("utf-8")
        for step in steps
    )


def _byte_spec(payload: bytes) -> dict:
    return {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def test_preregistration_binds_exact_recovery_sources_and_three_runs() -> None:
    payload = yaml.safe_load(PREREGISTRATION.read_text(encoding="utf-8"))
    assert payload["status"] == (
        "preregistered_after_external_interruption_before_recovery"
    )
    assert payload["checkpoint_step"] == 10272
    assert payload["final_optimizer_step"] == 12840
    assert payload["scope"] == {
        "training_recipe_changed": False,
        "checkpoint_selected_by_metric": False,
        "public_test_accessed": False,
        "development_or_cem_executed": False,
        "full_state_required_resume_only": True,
        "uncommitted_trace_tail_preserved": True,
    }
    assert [row["seed"] for row in payload["interrupted_runs"]] == [
        3072,
        4096,
        5120,
    ]
    for specification in (
        payload["frozen_inputs"] | payload["implementation"]
    ).values():
        path = ROOT / specification["path"]
        assert _sha(path) == specification["sha256"]
        assert path.stat().st_size == specification["size_bytes"]
    for row in payload["interrupted_runs"]:
        assert row["last_checkpoint"]["sha256"] == row["state_checkpoint"][
            "sha256"
        ]
        assert row["last_checkpoint"]["size_bytes"] == row["state_checkpoint"][
            "size_bytes"
        ]
        assert row["discarded_uncommitted_tail"]["first_optimizer_step"] > 10272
        assert row["canonical_prefix"]["last_optimizer_step"] <= 10272


def test_trace_recovery_preserves_original_and_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(recovery, "ROOT", tmp_path)
    run = tmp_path / "run"
    run.mkdir()
    trace = run / "loss_trace.jsonl"
    archive = tmp_path / "attempt" / "original.jsonl"
    original = _payload([1, 20, 10260, 10280, 10520])
    prefix = _payload([1, 20, 10260])
    trace.write_bytes(original)
    entry = {
        "loss_trace": {"path": "run/loss_trace.jsonl", **_byte_spec(original)},
        "canonical_prefix": {
            **_byte_spec(prefix),
            "rows": 3,
            "last_optimizer_step": 10260,
        },
        "discarded_uncommitted_tail": {
            "rows": 2,
            "first_optimizer_step": 10280,
            "last_optimizer_step": 10520,
        },
        "archive_path": "attempt/original.jsonl",
    }

    first = recovery.prepare_trace_recovery(entry)
    second = recovery.prepare_trace_recovery(entry)

    assert first == second
    assert archive.read_bytes() == original
    assert trace.read_bytes() == prefix
    assert first["excluded_uncommitted_tail"] == {
        "rows": 2,
        "first_optimizer_step": 10280,
        "last_optimizer_step": 10520,
        "preserved_in_original_trace": True,
    }


def test_trace_recovery_rejects_changed_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(recovery, "ROOT", tmp_path)
    run = tmp_path / "run"
    attempt = tmp_path / "attempt"
    run.mkdir()
    attempt.mkdir()
    original = _payload([1, 10260, 10280])
    prefix = _payload([1, 10260])
    (run / "loss_trace.jsonl").write_bytes(prefix)
    (attempt / "original.jsonl").write_bytes(b"changed\n")
    entry = {
        "loss_trace": {"path": "run/loss_trace.jsonl", **_byte_spec(original)},
        "canonical_prefix": {
            **_byte_spec(prefix),
            "rows": 2,
            "last_optimizer_step": 10260,
        },
        "discarded_uncommitted_tail": {
            "rows": 1,
            "first_optimizer_step": 10280,
            "last_optimizer_step": 10280,
        },
        "archive_path": "attempt/original.jsonl",
    }
    with pytest.raises(RuntimeError, match="archive identity changed"):
        recovery.prepare_trace_recovery(entry)


def test_training_invocation_is_required_resume_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(recovery, "ROOT", tmp_path)
    (tmp_path / "completion.yaml").write_text("schema_version: 1\n")
    initialization = tmp_path / "initial.pt"
    initialization.write_bytes(b"weights")
    output = tmp_path / "training"
    preregistration = {
        "frozen_inputs": {
            "completion_config": {"path": "completion.yaml"},
            "initialization_checkpoint": {
                "path": "initial.pt",
                "sha256": _sha(initialization),
            },
        },
        "training_invocation": {
            "model_id": "H3_Speed_PLDM_ReferenceCompletion",
            "data_split_seed": 3072,
            "original_h5": "/frozen/tworoom.h5",
            "output_root": "training",
        },
        "stable_worldmodel": {"worktree": "/stablewm", "commit": "abc"},
    }
    entry = {
        "seed": 3072,
        "run_name": "speed_pldm_reference_completion_v1_s3072",
        "training_report": "seed_3072/training_report.json",
    }
    module = type("Training", (), {"__file__": "/trainer.py"})

    arguments = recovery._training_arguments(preregistration, entry, module)

    assert arguments[arguments.index("--resume-policy") + 1] == "required"
    assert "never" not in arguments
    assert arguments[arguments.index("--seed") + 1] == "3072"
    assert arguments[arguments.index("--output-root") + 1] == str(output)
