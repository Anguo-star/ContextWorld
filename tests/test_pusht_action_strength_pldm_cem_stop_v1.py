from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/freeze_pusht_action_strength_pldm_cem_stop_v1.py"

SPEC = importlib.util.spec_from_file_location("action_strength_cem_stop", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
stop = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stop)


def test_failed_three_seed_aggregate_builds_a_non_cem_terminal_receipt() -> None:
    output = stop._resolve(stop.STOP_OUTPUT, label="CEM stop output")
    receipt = json.loads(output.read_text(encoding="utf-8"))

    assert receipt["completion_id"] == stop.COMPLETION_ID
    assert receipt["public_icl"] == {
        "passed": False,
        "passed_checkpoints": 0,
        "evaluated_checkpoints": 3,
        "raw_public_gate_passed": {"13313": False, "13314": False, "13315": False},
        "float32_recovered_gate_passed": {
            "13313": False,
            "13314": False,
            "13315": False,
        },
        "reason": "all_three_raw_and_float32_recovered_gates_are_false",
    }
    assert receipt["cem"]["authorized"] is False
    assert receipt["cem"]["executed"] is False
    assert receipt["input_integrity"]["all_frozen_inputs_unchanged_during_stop_freeze"]


def test_stop_freezer_rejects_a_noncanonical_output() -> None:
    with pytest.raises(ValueError, match="CEM stop output must equal"):
        stop.freeze(ROOT / "artifacts/evaluation/history3/not_the_stop_receipt.json")
