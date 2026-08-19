from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/freeze_tworoom_speed_pldm_execution_disclosure_v1.py"


def _module():
    spec = importlib.util.spec_from_file_location("speed_execution_disclosure", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_registered_disclosure_boundary_is_static_and_explicit_about_nonclaims() -> None:
    gate = _module()
    config, state = gate._validate_config(gate.DEFAULT_CONFIG)

    assert config["status"] == (
        "registered_after_terminal_report_failure_before_report_recovery_and_execution_disclosure"
    )
    assert state["implementation"]["execution_disclosure_freezer"]["path"] == (
        "scripts/freeze_tworoom_speed_pldm_execution_disclosure_v1.py"
    )
    assert config["assertion_boundary"]["asserted"] == {
        "full_trainer_state_resume_from_step_10272": True,
        "fixed_training_recipe_and_optimizer_budget": True,
        "terminal_reports_recovered_with_zero_optimizer_steps": True,
    }
    assert (
        state["frozen_inputs"]["terminal_report_recovery_preregistration"]["path"]
        == "configs/benchmark/tworoom_speed_pldm_terminal_report_recovery_v1.yaml"
    )
    assert config["assertion_boundary"]["not_asserted"] == {
        "worker_rng_bitwise_equivalence": False,
        "sample_order_bitwise_equivalence": False,
        "batch_composition_bitwise_equivalence": False,
        "loss_trace_bitwise_equivalence_after_resume": False,
        "parameter_tensor_bitwise_equivalence_to_an_uninterrupted_counterfactual": False,
    }


def test_development_lineage_snapshot_is_byte_identical_and_active_sources_are_registered() -> None:
    manifest_path = ROOT / "scripts/freeze_tworoom_speed_pldm_development_manifest_v1.py"
    spec = importlib.util.spec_from_file_location("speed_development_manifest", manifest_path)
    assert spec is not None and spec.loader is not None
    freezer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(freezer)

    _config, state = freezer._validate_config(freezer.DEFAULT_CONFIG)
    amendment = state["execution_disclosure_amendment"]
    snapshot = ROOT / amendment["pre_interruption_config_snapshot"]["path"]
    base = ROOT / amendment["base_development_config"]["path"]
    assert snapshot.read_bytes() == base.read_bytes()
    assert amendment["active_manifest_freezer"] == state["implementation"]["manifest_freezer"]


def test_archive_copies_match_source_bytes_without_reusing_source_paths() -> None:
    gate = _module()
    source_last = {"path": "training/last.ckpt", "sha256": "a" * 64, "size_bytes": 17}
    source_state = {"path": "training/state.ckpt", "sha256": "a" * 64, "size_bytes": 17}
    archived_checkpoint = {
        "path": "attempts/seed_3072/resume_source_step_10272.ckpt",
        "sha256": "a" * 64,
        "size_bytes": 17,
    }
    assert gate._valid_checkpoint_archive_copy(
        archived_checkpoint,
        source_last,
        source_state,
        expected_archive_path=archived_checkpoint["path"],
    )
    assert not gate._valid_checkpoint_archive_copy(
        {**archived_checkpoint, "path": "attempts/wrong.ckpt"},
        source_last,
        source_state,
        expected_archive_path=archived_checkpoint["path"],
    )
    assert not gate._valid_checkpoint_archive_copy(
        {**archived_checkpoint, "sha256": "b" * 64},
        source_last,
        source_state,
        expected_archive_path=archived_checkpoint["path"],
    )

    source_trace = {"path": "training/loss_trace.jsonl", "sha256": "c" * 64, "size_bytes": 23}
    prefix = {"rows": 515}
    tail = {"rows": 13, "last_optimizer_step": 10520}
    archived_trace = {
        "path": "attempts/seed_3072/loss_trace_interrupted_after_state_10272.jsonl",
        "sha256": "c" * 64,
        "size_bytes": 23,
        "rows": 528,
        "last_optimizer_step": 10520,
    }
    assert gate._valid_trace_archive_copy(
        archived_trace,
        source_trace,
        prefix,
        tail,
        expected_archive_path=archived_trace["path"],
    )
    assert not gate._valid_trace_archive_copy(
        {**archived_trace, "path": source_trace["path"]},
        source_trace,
        prefix,
        tail,
        expected_archive_path=archived_trace["path"],
    )
    assert not gate._valid_trace_archive_copy(
        {**archived_trace, "rows": 527},
        source_trace,
        prefix,
        tail,
        expected_archive_path=archived_trace["path"],
    )
