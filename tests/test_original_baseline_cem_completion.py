from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from contextworld.benchmarks.original_baseline_cem_completion import (
    EXPECTED_COMPONENTS,
    EXPECTED_EXCLUDED_COMPONENTS,
    EXPECTED_UNIT_COUNTS,
    audit_original_baseline_cem_completion_draft,
    enumerate_atomic_execution_units,
    load_original_baseline_cem_completion_draft,
)
from contextworld.benchmarks.original_baseline_archive import (
    audit_archived_original_baseline_matrix,
)


def test_draft_is_complete_but_non_executable() -> None:
    draft = load_original_baseline_cem_completion_draft()

    assert draft["status"] == "draft_blocked"
    assert draft["freeze_generated"] is False
    assert draft["authority"] == {
        "cem_execution_authorized": False,
        "training_authorized": False,
        "finetuning_authorized": False,
        "checkpoint_selection_authorized": False,
        "result_based_retry_or_checkpoint_swap_authorized": False,
        "formal_scoreboard_mutation": False,
        "legacy_cem_result_reuse_authorized": False,
    }
    assert len(draft["canonical_checkpoint_ids"]) == 8
    assert len(draft["component_protocols"]) == len(EXPECTED_COMPONENTS) == 7
    assert len(draft["cem_cells"]) == 14
    assert len(draft["excluded_cem_cells"]) == 4
    assert {
        cell["capability_id"] for cell in draft["excluded_cem_cells"]
    } == set(EXPECTED_EXCLUDED_COMPONENTS)
    assert all(
        cell["execution_status"] == "blocked_before_execution"
        and cell["cem_execution_authorized"] is False
        for cell in draft["cem_cells"]
    )
    assert all(
        component["executable_argv"] is None
        for component in draft["component_protocols"]
    )


def test_atomic_units_are_planning_only_and_all_blocked() -> None:
    draft = load_original_baseline_cem_completion_draft()
    units = enumerate_atomic_execution_units(draft)

    assert len(units) == sum(EXPECTED_UNIT_COUNTS.values()) == 132
    assert all(unit["command"] is None for unit in units)
    assert all(unit["result_status"] == "not_started" for unit in units)
    assert all(unit["cem_execution_authorized"] is False for unit in units)
    assert {
        component: sum(unit["capability_id"] == component for unit in units)
        for component in EXPECTED_COMPONENTS
    } == EXPECTED_UNIT_COUNTS


def test_obsolete_static_draft_refuses_live_release_rederivation() -> None:
    # This planning-only draft predates the later component-independent CEM
    # execution.  It remains readable, but its old live-release identities must
    # not be treated as a current result authority.
    with pytest.raises(RuntimeError, match="identity mismatch"):
        audit_original_baseline_cem_completion_draft()
    assert audit_archived_original_baseline_matrix()["status"] == "passed"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda draft: draft["authority"].__setitem__(
                "cem_execution_authorized", True
            ),
            "authority.cem_execution_authorized",
        ),
        (
            lambda draft: draft["component_protocols"][0].__setitem__(
                "executable_argv", ["python", "unsafe.py"]
            ),
            "must not contain an executable command",
        ),
        (
            lambda draft: draft["cem_cells"][0].__setitem__(
                "checkpoint_id", "pusht_lewm_original"
            ),
            "expected tworoom_lewm_original",
        ),
    ],
)
def test_validator_rejects_authority_command_or_checkpoint_drift(
    mutation, message: str, tmp_path: Path
) -> None:
    draft = deepcopy(load_original_baseline_cem_completion_draft())
    draft.pop("_config_path")
    mutation(draft)
    path = tmp_path / "mutated.yaml"
    path.write_text(yaml.safe_dump(draft, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_original_baseline_cem_completion_draft(path)
