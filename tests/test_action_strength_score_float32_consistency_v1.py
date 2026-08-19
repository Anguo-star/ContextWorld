from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from contextworld.benchmarks.action_strength_icl_score import (
    score_action_strength_icl_results,
)


ROOT = Path(__file__).resolve().parents[1]
AMENDMENT = ROOT / (
    "configs/benchmark/"
    "pusht_action_strength_score_float32_consistency_amendment_v1.yaml"
)
RELEASE = ROOT / "configs/benchmark/pusht_action_strength_icl_release_v1.yaml"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_yaml(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_amendment_preserves_every_frozen_formal_input() -> None:
    amendment = _load_yaml(AMENDMENT)
    evidence = amendment["frozen_formal_evidence"]
    identities = list(evidence["raw_public_icl"])
    recovery = evidence["independent_float32_recovery"]
    identities.extend(recovery["seed_receipts"])
    identities.extend([recovery["aggregate"], recovery["cem_stop"]])

    for identity in identities:
        path = ROOT / identity["path"]
        assert path.is_file()
        assert _sha256(path) == identity["sha256"]


def test_current_release_binds_the_float32_consistent_scorer() -> None:
    release = _load_yaml(RELEASE)
    score_identity = release["identity"]["score_api"]
    scorer = ROOT / score_identity["path"]
    assert _sha256(scorer) == score_identity["sha256"]

    maintenance = release["maintenance_amendments"]
    assert len(maintenance) == 1
    assert maintenance[0]["path"] == str(AMENDMENT.relative_to(ROOT))
    assert _sha256(AMENDMENT) == maintenance[0]["sha256"]


def test_public_score_command_reproduces_the_frozen_negative_result() -> None:
    amendment = _load_yaml(AMENDMENT)
    raw_rows = amendment["frozen_formal_evidence"]["raw_public_icl"]
    result = score_action_strength_icl_results(
        result_paths=[ROOT / row["path"] for row in raw_rows],
        method_name="stable_worldmodel_pldm_reference_completion_v1",
        release_config=RELEASE,
    )

    assert result["submission_kind"] == "three_seed_method"
    assert result["decision"] == {
        "passed": False,
        "formal_method_claim": True,
        "reason": "one_or_more_training_seeds_failed",
    }
    assert sum(row["passed"] for row in result["checkpoints"]) == 0
    assert result["aggregate"]["correct_future_rate"]["mean"] == (
        0.9427083333333334
    )

    recovery = amendment["frozen_formal_evidence"][
        "independent_float32_recovery"
    ]
    recovery_by_seed = {
        row["seed"]: _load_json(ROOT / row["path"])
        for row in recovery["seed_receipts"]
    }
    for checkpoint in result["checkpoints"]:
        recovered = recovery_by_seed[checkpoint["training_seed"]]
        assert checkpoint["passed"] is recovered["reconstruction"]["gate"][
            "passed"
        ]
