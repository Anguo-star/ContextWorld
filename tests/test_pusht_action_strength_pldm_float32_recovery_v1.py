from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/recover_pusht_action_strength_pldm_float32_rescore_v1.py"
PREREG = ROOT / "configs/benchmark/pusht_action_strength_pldm_float32_rescore_recovery_v1.yaml"

SPEC = importlib.util.spec_from_file_location("action_strength_float32_recovery", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
recovery = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(recovery)


def test_implementation_hash_drift_is_fail_closed() -> None:
    """A changed recovery implementation cannot create an evidence receipt."""

    prereg = recovery._load_yaml(PREREG)
    drifted = deepcopy(prereg)
    drifted["implementation"]["recovery_launcher"]["sha256"] = "0" * 64

    with pytest.raises(RuntimeError, match="Implementation changed"):
        recovery._matched_identities(
            drifted["implementation"], category="Implementation"
        )


def test_recovery_and_aggregate_reject_nonpreregistered_outputs() -> None:
    """Neither command may write an arbitrary repository JSON path."""

    prereg = recovery._load_yaml(PREREG)
    entries = recovery._checkpoint_entries(prereg)
    rejected = ROOT / "artifacts/evaluation/history3/not_recovery_namespace.json"
    with pytest.raises(ValueError, match="preregistered exclusive destination"):
        expected, _ = recovery._expected_recovery_output(
            prereg, entries, 13313
        )
        recovery._assert_output(rejected, expected, label="recovery output")
    with pytest.raises(ValueError, match="preregistered exclusive destination"):
        expected, _ = recovery._expected_aggregate_output(prereg)
        recovery._assert_output(rejected, expected, label="aggregate output")


def test_frozen_seed_receipt_records_an_exact_reconstruction() -> None:
    """The completed receipt remains the immutable historical evidence."""

    prereg = recovery._load_yaml(PREREG)
    entries = recovery._checkpoint_entries(prereg)
    output, logical_output = recovery._expected_recovery_output(
        prereg, entries, 13313
    )
    receipt = json.loads(output.read_text(encoding="utf-8"))

    assert receipt["output"]["path"] == logical_output
    assert receipt["bindings"]["checkpoint"]["matched"] is True
    assert receipt["verification"]["passed"] is True
    assert receipt["verification"]["float32_scalar_aggregates_bitwise_equal"] is True
    assert receipt["verification"]["gate_exact_equal"] is True
    assert receipt["input_integrity"]["all_frozen_inputs_unchanged_during_recovery"]


def test_recovery_write_is_x_exclusive(tmp_path) -> None:
    """A completed receipt cannot be silently replaced."""

    output = tmp_path / "receipt.json"
    recovery._write_exclusive(output, {"status": "first"})
    with pytest.raises(FileExistsError):
        recovery._write_exclusive(output, {"status": "replacement"})
    assert output.read_text(encoding="utf-8").strip() == '{\n  "status": "first"\n}'
