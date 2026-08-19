from __future__ import annotations

import hashlib
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
READINESS_CONFIG = (
    ROOT
    / "configs/benchmark/contextworld_public_v1_release_readiness_draft_v1.yaml"
)
READINESS_DOCUMENT = ROOT / "docs/ContextWorld_Public_v1_Release_Readiness.md"
FROZEN_PUBLIC_DOCUMENT = ROOT / "docs/ContextWorld_ICL_Benchmark.md"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_readiness_draft_cannot_activate_or_mutate_release() -> None:
    payload = yaml.safe_load(READINESS_CONFIG.read_text(encoding="utf-8"))

    assert payload["status"] == "draft_not_release_authority"
    authority = payload["authority"]
    assert {
        key: authority[key]
        for key in (
            "activates_release",
            "modifies_formal_scoreboard",
            "authorizes_public_test_access",
            "authorizes_reference_rerun",
            "frozen_candidate_commit",
        )
    } == {
        "activates_release": False,
        "modifies_formal_scoreboard": False,
        "authorizes_public_test_access": False,
        "authorizes_reference_rerun": False,
        "frozen_candidate_commit": (
            "6607552ac37ba26d6b5e7f053416b1e792ddae91"
        ),
    }
    assert authority["base_frozen_public_document"] == {
        "path": "docs/ContextWorld_ICL_Benchmark.md",
        "sha256": (
            "72031232d008b77f809d387348f8bc320532f80517e387837571a2995932cccc"
        ),
    }
    assert authority["canonical_registration_decision"] == {
        "path": (
            "artifacts/evaluation/history3/"
            "cube_gripper_carry_h3_v4r1_suite_registration_recovery_v2/"
            "registration_decision_v2.json"
        ),
        "sha256": (
            "0c6d38ec4304d0ffa078fe8680a7b6d90b851eb69780d22936a456bf0bb7859d"
        ),
    }
    amendment = authority["historical_document_amendment_v1"]
    assert amendment["status"] == "historical_exact_identity_preserved"
    assert amendment["scope"] == "documentation_only_reference_table_expansion"
    assert amendment["modifies_formal_scoreboard"] is False
    assert amendment["authorizes_public_test_access"] is False
    assert _sha256(ROOT / amendment["config"]["path"]) == amendment[
        "config"
    ]["sha256"]
    assert _sha256(ROOT / amendment["decision"]["path"]) == amendment[
        "decision"
    ]["sha256"]
    assert _sha256(FROZEN_PUBLIC_DOCUMENT) != amendment["public_document"][
        "sha256"
    ]

    current = authority["current_documentation"]
    assert current["status"] == (
        "edited_for_external_readers_pending_integrity_reseal_v2"
    )
    assert current["current_bytes_are_not_bound_by_historical_amendment_v1"] is True
    taxonomy = current["capability_taxonomy_record"]
    taxonomy_payload = yaml.safe_load(
        (ROOT / taxonomy["path"]).read_text(encoding="utf-8")
    )
    assert taxonomy_payload["capability_taxonomy_documentation_amendment"][
        "record_id"
    ] == taxonomy["record_id"]
    reseal = current["final_identity_authority"]
    reseal_payload = yaml.safe_load(
        (ROOT / reseal["path"]).read_text(encoding="utf-8")
    )
    assert reseal_payload["integrity_reseal"]["reseal_id"] == reseal["reseal_id"]


def test_cube_comparison_reports_only_existing_evidence() -> None:
    payload = yaml.safe_load(READINESS_CONFIG.read_text(encoding="utf-8"))
    rows = {
        row["comparison_id"]: row
        for row in payload["cube_current_evidence"]["comparison_rows"]
    }

    assert rows["original_lewm"]["icl_score"] == 0.509765625
    assert rows["original_lewm"]["cem_successes"] == 198
    assert rows["original_pldm"]["icl_score"] == 0.509765625
    assert rows["original_pldm"]["cem_successes"] == 159
    assert rows["trained_lewm"]["public_correct_future_rate_mean"] == (
        0.7845052083333334
    )
    assert rows["trained_lewm"]["cem_successes_by_training_seed"] == [
        186,
        183,
        185,
    ]
    assert rows["trained_pldm"]["development_correct_future_rate_mean"] == (
        0.5013020833333334
    )
    assert rows["trained_pldm"]["public_score"] == "not_authorized_not_run"

    document = READINESS_DOCUMENT.read_text(encoding="utf-8")
    assert "独立 v4r1 描述性基线" in document
    assert "78.45%；77.73%、79.10%、78.52%" in document
    assert "50.13%；50.20%、50.20%、50.00%" in document
    assert "正式 scoreboard 目前仍只有一条 Cube LeWM Public 行" in document


def test_external_model_matrix_is_empty_and_fail_closed() -> None:
    payload = yaml.safe_load(READINESS_CONFIG.read_text(encoding="utf-8"))
    validation = payload["external_model_validation"]

    assert validation["status"] == "not_started"
    assert validation["proposed_v1_minimum"][
        "additional_open_source_model_slots"
    ] == 3
    assert validation["proposed_v1_minimum"]["distribution_claim_scope"] == (
        "cube_external_pilot_not_suite_wide_validation"
    )
    suite_claim = validation["suite_wide_cross_architecture_claim"]
    assert suite_claim["actual_results_required_not_future_plan"] is True
    assert len(suite_claim["capability_types_required"]) == 4
    assert suite_claim["independent_upstream_model_families_minimum"] == 2
    assert suite_claim["common_task_overlap_between_two_families_minimum"] == 1
    policy = validation["selection_and_scoring_policy"]
    assert policy["same_dataset_and_query_identities_across_models"] is True
    assert policy["evaluation_parameter_updates_allowed"] is False
    assert policy["common_budget_track_definition_required_before_runs"] is True
    assert len(validation["slots"]) == 3
    for slot in validation["slots"]:
        assert slot["model_name"] == "TBD"
        assert slot["upstream_repository"] == "TBD"
        assert slot["license"] == "TBD"
        assert slot["adapter_status"] == "not_started"
        assert slot["development_status"] == "not_run"
        assert slot["public_status"] == "not_authorized_not_run"
        assert slot["cem_status"] == "not_run"

    document = READINESS_DOCUMENT.read_text(encoding="utf-8")
    assert "多开源模型对比表（待补齐）" in document
    assert document.count("| External-0") == 3
    assert "空白项必须由真实运行和冻结证据填写" in document
    assert "全 Suite 跨架构覆盖声明" in document
    assert "只有后续计划而没有真实结果" in document


def test_all_public_v1_blockers_remain_explicit_and_navigable() -> None:
    payload = yaml.safe_load(READINESS_CONFIG.read_text(encoding="utf-8"))
    gates = payload["release_gates"]
    expected_blockers = {
        "external_model_coverage",
        "code_license",
        "generated_data_license",
        "upstream_redistribution_and_attribution",
        "stable_public_artifact_urls",
        "checksummed_download_manifest",
        "clean_room_install_and_reproduction",
        "external_submission_and_scoreboard_governance",
        "common_cross_model_training_budget",
        "citation_metadata",
        "final_public_v1_release_decision",
    }
    assert {name for name, gate in gates.items() if gate["blocking"]} == (
        expected_blockers
    )
    assert gates["reference_table_document_amendment"] == {
        "status": "completed_v1_without_formal_scoreboard_mutation",
        "blocking": False,
    }

    root_readme = (ROOT / "README.md").read_text(encoding="utf-8")
    docs_readme = (ROOT / "docs/README.md").read_text(encoding="utf-8")
    assert "ContextWorld_Public_v1_Release_Readiness.md" in root_readme
    assert "ContextWorld_Public_v1_Release_Readiness.md" in docs_readme
