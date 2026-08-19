from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]

COMPONENT_ENVIRONMENT = {
    "speed": "tworoom",
    "door": "tworoom",
    "action_delay": "tworoom",
    "portal_exit": "tworoom",
    "action_strength": "pusht",
    "contact_friction": "pusht",
    "motion_damping": "pusht",
    "robot_arm_mass": "reacher",
    "cube_gripper_carry": "cube",
}
FAMILY_COUNTS = {
    ("tworoom", "lewm"): (276, 279, 277),
    ("tworoom", "pldm"): (278, 254, 265),
    ("pusht", "lewm"): (248, 235, 257),
    ("pusht", "pldm"): (233, 219, 229),
    ("reacher", "lewm"): (164, 170, 169),
    ("reacher", "pldm"): (248, 240, 139),
    ("cube", "lewm"): (197, 198, 194),
    ("cube", "pldm"): (158, 159, 164),
}
V1_BASELINE_COUNTS = {
    ("tworoom", "lewm"): 273,
    ("tworoom", "pldm"): 278,
    ("pusht", "lewm"): 235,
    ("pusht", "pldm"): 233,
    ("reacher", "lewm"): 164,
    ("reacher", "pldm"): 248,
    ("cube", "lewm"): 198,
    ("cube", "pldm"): 159,
}


def _load_builder():
    path = ROOT / "scripts/build_contextworld_complete_comparison_v2.py"
    spec = importlib.util.spec_from_file_location("comparison_v2_builder_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return _sha256(path)


def _v1_document() -> dict[str, Any]:
    rows = []
    for component, environment in COMPONENT_ENVIRONMENT.items():
        for label in ("LeWM", "PLDM"):
            count = V1_BASELINE_COUNTS[(environment, label.lower())]
            rows.append(
                {
                    "component_id": component,
                    "family": label,
                    "environment": environment,
                    "icl": {"correct_future_rate": 0.5},
                    "original_task_cem": {
                        "family_baseline_successes": count,
                        "family_baseline_evaluations": 300,
                        "family_baseline_rate": count / 300,
                    },
                }
            )
    assert len(rows) == 18
    return {
        "comparison_id": "contextworld_complete_reference_comparison_v1",
        "row_count": 18,
        "rows": rows,
        "claim_boundary": {"formal_suite_scoreboard_eligible": False},
        "historical_scoreboard": {"rows": 13, "sha256": "78dec56c" + "0" * 56},
        "report_all_policy": {"failed_scores_published": True},
    }


def _summary_document() -> dict[str, Any]:
    families = []
    for (environment, family), counts in FAMILY_COUNTS.items():
        rates = [count / 300 for count in counts]
        mean = sum(rates) / 3
        variance = sum((rate - mean) ** 2 for rate in rates) / 2
        lineage_notes = []
        if (environment, family) == ("tworoom", "lewm"):
            lineage_notes.append(
                {
                    "training_seed": 3072,
                    "success_count": 273,
                    "evaluation_count": 300,
                    "success_rate": 0.91,
                    "provenance": "carried_original_baseline_cem_v1",
                    "excluded_from_family_statistics": True,
                    "reason": "repo-trained h3_origheldout lineage",
                }
            )
        families.append(
            {
                "environment": environment,
                "family": family,
                "members": [
                    {
                        "training_seed": 3072 + index,
                        "cell_id": f"{environment}_{family}_seed{3072 + index}",
                        "success_count": count,
                        "evaluation_count": 300,
                        "success_rate": count / 300,
                        "provenance": (
                            "carried_original_baseline_cem_v1"
                            if index == 0 and (environment, family) != ("tworoom", "lewm")
                            else "new_cell_this_preregistration"
                        ),
                    }
                    for index, count in enumerate(counts)
                ],
                "statistics": {
                    "n_training_seeds": 3,
                    "success_rates": rates,
                    "mean": mean,
                    "sample_std": math.sqrt(variance),
                    "sample_variance": variance,
                    "minimum": min(rates),
                    "maximum": max(rates),
                },
                "lineage_notes": lineage_notes,
            }
        )
    return {
        "summary_id": "contextworld_original_baseline_seed_completion_family_summary_v1",
        "families": families,
    }


class Fixture:
    def __init__(self, tmp_path: Path):
        self.module = _load_builder()
        self.root = tmp_path
        self.v1_path = tmp_path / "reference/complete_comparison.json"
        self.v1_payload = _v1_document()
        self.v1_sha256 = _write_json(self.v1_path, self.v1_payload)
        self.summary_path = tmp_path / "seed_completion/family_summary.json"
        self.summary_payload = _summary_document()
        _write_json(self.summary_path, self.summary_payload)
        self.prereg_path = tmp_path / "prereg.yaml"
        self.prereg_path.write_text("preregistration_id: seed_completion\n", encoding="utf-8")
        self.output_path = tmp_path / "reference/complete_comparison_v2.json"

    def rewrite_v1(self) -> None:
        self.v1_sha256 = _write_json(self.v1_path, self.v1_payload)

    def build(self, **overrides: Any) -> dict[str, Any]:
        options: dict[str, Any] = {
            "v1_path": self.v1_path,
            "family_summary_path": self.summary_path,
            "seed_completion_prereg_path": self.prereg_path,
            "output": self.output_path,
            "expected_v1_sha256": self.v1_sha256,
            "repo_root": self.root,
        }
        options.update(overrides)
        return self.module.build(**options)


@pytest.fixture()
def fixture(tmp_path: Path) -> Fixture:
    return Fixture(tmp_path)


def test_build_adds_family_baseline_v2_and_keeps_v1_untouched(fixture: Fixture) -> None:
    before = fixture.v1_sha256
    document = fixture.build()
    assert fixture.output_path.is_file()
    assert json.loads(fixture.output_path.read_text(encoding="utf-8")) == document
    assert _sha256(fixture.v1_path) == before
    assert document["comparison_id"] == "contextworld_complete_reference_comparison_v2"
    assert document["row_count"] == 18 and len(document["rows"]) == 18
    assert document["derived_from_v1"]["sha256"] == before
    assert document["derived_from_v1"]["v1_file_unmodified"] is True
    assert (
        document["baseline_family_summary"]["summary_id"]
        == "contextworld_original_baseline_seed_completion_family_summary_v1"
    )
    claim = document["claim_boundary"]
    assert claim["formal_suite_scoreboard_eligible"] is False
    assert claim["historical_scoreboard_rows_unchanged"] is True
    assert claim["baseline_columns_reported_as_three_training_seed_statistics"] is True
    assert document["historical_scoreboard"] == fixture.v1_payload["historical_scoreboard"]

    families = {
        (family["environment"], family["family"]): family
        for family in fixture.summary_payload["families"]
    }
    for original, row in zip(fixture.v1_payload["rows"], document["rows"]):
        environment = COMPONENT_ENVIRONMENT[original["component_id"]]
        family = families[(environment, original["family"].lower())]
        block = row["original_task_cem"]["family_baseline_v2"]
        assert block["environment"] == environment
        assert [entry["training_seed"] for entry in block["per_training_seed"]] == [
            3072,
            3073,
            3074,
        ]
        assert [entry["success_count"] for entry in block["per_training_seed"]] == [
            member["success_count"] for member in family["members"]
        ]
        statistics = family["statistics"]
        assert block["mean"] == pytest.approx(statistics["mean"])
        assert block["sample_std"] == pytest.approx(statistics["sample_std"])
        assert block["sample_variance"] == pytest.approx(statistics["sample_variance"])
        assert block["minimum"] == statistics["minimum"]
        assert block["maximum"] == statistics["maximum"]
        assert (
            block["historical_family_baseline_rate_v1"]
            == original["original_task_cem"]["family_baseline_rate"]
        )
        assert block["lineage_notes"] == family["lineage_notes"]
        assert block["source"]["sha256"] == _sha256(fixture.summary_path)
        # Every v1 field must be carried verbatim; the v2 block is additive.
        stripped = copy.deepcopy(row)
        del stripped["original_task_cem"]["family_baseline_v2"]
        assert stripped == original


def test_v1_identity_drift_fails(fixture: Fixture) -> None:
    with pytest.raises(fixture.module.ComparisonBuildError, match="frozen v1 identity"):
        fixture.build(expected_v1_sha256="0" * 64)
    assert not fixture.output_path.exists()


def test_family_summary_id_drift_fails(fixture: Fixture) -> None:
    fixture.summary_payload["summary_id"] = "something_else"
    _write_json(fixture.summary_path, fixture.summary_payload)
    with pytest.raises(fixture.module.ComparisonBuildError, match="summary id"):
        fixture.build()


def test_missing_family_fails(fixture: Fixture) -> None:
    fixture.summary_payload["families"] = fixture.summary_payload["families"][:7]
    _write_json(fixture.summary_path, fixture.summary_payload)
    with pytest.raises(fixture.module.ComparisonBuildError, match="eight families"):
        fixture.build()


def test_unknown_component_fails(fixture: Fixture) -> None:
    fixture.v1_payload["rows"][0]["component_id"] = "unknown_component"
    fixture.rewrite_v1()
    with pytest.raises(fixture.module.ComparisonBuildError, match="unknown component"):
        fixture.build()


def test_row_with_existing_v2_block_fails(fixture: Fixture) -> None:
    fixture.v1_payload["rows"][3]["original_task_cem"]["family_baseline_v2"] = {}
    fixture.rewrite_v1()
    with pytest.raises(fixture.module.ComparisonBuildError, match="family_baseline_v2"):
        fixture.build()


def test_refuses_to_overwrite_output(fixture: Fixture) -> None:
    fixture.output_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(FileExistsError):
        fixture.build()
