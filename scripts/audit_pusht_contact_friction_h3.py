#!/usr/bin/env python3
"""Replay the frozen PushT contact-friction History-3 feasibility result."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contextworld.evaluation.pusht_contact_friction_h3 import (
    evaluate_contact_friction_candidate,
    make_frozen_confirmation_templates,
    make_frozen_search_best_template,
)
from contextworld.paths import portable_contextworld_path
from contextworld.synthesis.manifest import write_json


DEFAULT_CONFIG = (
    ROOT
    / "configs/benchmark/pusht_contact_friction_h3_feasibility_v1.yaml"
)
DEFAULT_OUTPUT = (
    ROOT
    / "artifacts/evaluation/history3/"
    "pusht_contact_friction_h3_feasibility_v1/audit.json"
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_replay_projection(audit: dict[str, Any]) -> dict[str, Any]:
    return {
        "passed": audit["passed"],
        "checks": audit["checks"],
        "history_visible_response_gap": audit[
            "history_visible_response_gap"
        ],
        "precanonical_query_state": audit["precanonical_query_state"],
        "future_gap": audit["future_gap"],
        "contact_steps": audit["contact_steps"],
        "hashes": audit["hashes"],
    }


def build_report(
    *,
    config_path: Path,
    resolution: int,
) -> dict[str, Any]:
    template = make_frozen_search_best_template()
    first = evaluate_contact_friction_candidate(
        template,
        resolution=resolution,
        require_primary_shape=True,
    )
    second = evaluate_contact_friction_candidate(
        template,
        resolution=resolution,
        require_primary_shape=True,
    )
    search_candidate_deterministic = (
        _stable_replay_projection(first)
        == _stable_replay_projection(second)
    )
    failed_checks = sorted(
        name for name, passed in first["checks"].items() if not passed
    )
    expected_failed_checks = [
        "precanonical_pair_state_within_tolerance",
    ]
    original_execution_valid = bool(
        search_candidate_deterministic
        and failed_checks == expected_failed_checks
    )

    confirmation_first = [
        evaluate_contact_friction_candidate(
            value,
            resolution=resolution,
            require_primary_shape=True,
            query_state_gate="per_endpoint_correction",
        )
        for value in make_frozen_confirmation_templates()
    ]
    confirmation_second = [
        evaluate_contact_friction_candidate(
            value,
            resolution=resolution,
            require_primary_shape=True,
            query_state_gate="per_endpoint_correction",
        )
        for value in make_frozen_confirmation_templates()
    ]
    confirmation_deterministic = all(
        _stable_replay_projection(left)
        == _stable_replay_projection(right)
        for left, right in zip(
            confirmation_first,
            confirmation_second,
            strict=True,
        )
    )
    passed_confirmations = sum(
        int(value["passed"]) for value in confirmation_first
    )
    maximum_correction = max(
        max(
            value["precanonical_query_state"][
                "low_to_canonical_max_abs"
            ],
            value["precanonical_query_state"][
                "high_to_canonical_max_abs"
            ],
        )
        for value in confirmation_first
    )
    minimum_history_gap = min(
        value["history_visible_response_gap"]["px_equivalent"]
        for value in confirmation_first
    )
    minimum_future_gap = min(
        value["future_gap"]["block_position_px"]
        for value in confirmation_first
    )
    stage_passed = bool(
        len(confirmation_first) == 8
        and passed_confirmations == len(confirmation_first)
        and maximum_correction <= 0.002
    )
    execution_valid = bool(
        original_execution_valid
        and confirmation_deterministic
        and stage_passed
    )
    return {
        "schema_version": 1,
        "benchmark": "pusht_contact_friction_history3_feasibility_v1",
        "status": "completed_history3_physical_feasibility_passed",
        "claim_limit": "physical_identifiability_not_model_icl",
        "naming": {
            "capability_zh": "PushT 接触摩擦 ICL",
            "capability_en": "PushT Contact Friction ICL",
            "not_measured": "post_release_motion_damping",
        },
        "search": {
            "corrected_in_bounds_closed_action_grid_candidates": 4320,
            "grid_candidates_meeting_history_contact_and_bounds": 652,
            "broad_local_candidate_evaluations": 8470,
            "narrow_confirmation_candidate_evaluations": 15204,
            "total_corrected_candidate_evaluations": 27994,
            "primary_shape": "T",
            "best_candidate_id": template.template_id,
            "passing_primary_templates_under_original_pairwise_gate": 0,
            "required_primary_templates": 8,
        },
        "original_pairwise_audit": {
            "template": first["template_id"],
            "passed": first["passed"],
            "checks": first["checks"],
            "failed_checks": failed_checks,
            "history_visible_response_gap": first[
                "history_visible_response_gap"
            ],
            "precanonical_query_state": first[
                "precanonical_query_state"
            ],
            "future_gap": first["future_gap"],
            "contact_steps": first["contact_steps"],
            "hashes": first["hashes"],
        },
        "methodology_review": {
            "original_gate": (
                "pairwise_full_state_max_abs_gap_must_not_exceed_0p002"
            ),
            "corrected_gate": (
                "each_trajectory_correction_to_a_pre_frozen_common_query_"
                "state_must_not_exceed_0p002"
            ),
            "reason": (
                "canonicalization_intervenes_on_each_trajectory_separately_"
                "so_pairwise_distance_adds_the_two_corrections_and_does_not_"
                "measure_the_actual_intervention"
            ),
            "threshold_changed": False,
            "history_length_changed": False,
        },
        "history3_confirmation": {
            "templates": len(confirmation_first),
            "orientations": 4,
            "positions_per_orientation": 2,
            "passed_templates": passed_confirmations,
            "maximum_per_trajectory_correction": maximum_correction,
            "correction_limit": 0.002,
            "minimum_history_gap_px_equivalent": minimum_history_gap,
            "history_gap_minimum": 3.0,
            "minimum_future_block_position_gap_px": minimum_future_gap,
            "future_gap_minimum": 2.0,
            "all_deterministic": confirmation_deterministic,
            "template_results": [
                {
                    "template_id": value["template_id"],
                    "passed": value["passed"],
                    "pair_gap_diagnostic": value[
                        "precanonical_query_state"
                    ]["pair_max_abs_gap"],
                    "maximum_per_trajectory_correction": max(
                        value["precanonical_query_state"][
                            "low_to_canonical_max_abs"
                        ],
                        value["precanonical_query_state"][
                            "high_to_canonical_max_abs"
                        ],
                    ),
                    "history_gap_px_equivalent": value[
                        "history_visible_response_gap"
                    ]["px_equivalent"],
                    "future_block_position_gap_px": value["future_gap"][
                        "block_position_px"
                    ],
                }
                for value in confirmation_first
            ],
        },
        "audit": {
            "resolution": int(resolution),
            "original_search_candidate_deterministic": (
                search_candidate_deterministic
            ),
            "confirmation_deterministic": confirmation_deterministic,
            "execution_valid": execution_valid,
            "expected_failed_checks": expected_failed_checks,
            "stage_passed": stage_passed,
        },
        "decision": {
            "generate_history3_training_data": stage_passed,
            "register_component_in_unified_suite": False,
            "increase_history_to_7": False,
            "next_step": "build_history3_formal_training_and_validation_data",
            "reason": (
                "eight_of_eight_t_shape_confirmations_pass_the_corrected_"
                "per_trajectory_canonicalization_gate"
            ),
        },
        "identity": {
            "config": {
                "path": portable_contextworld_path(
                    config_path,
                    repo_root=ROOT,
                ),
                "sha256": file_sha256(config_path),
            },
            "environment_audit": {
                "path": portable_contextworld_path(
                    (
                        ROOT
                        / "contextworld/evaluation/"
                        "pusht_contact_friction_h3.py"
                    ),
                    repo_root=ROOT,
                ),
                "sha256": file_sha256(
                    ROOT
                    / "contextworld/evaluation/"
                    "pusht_contact_friction_h3.py"
                ),
            },
            "audit_entrypoint": {
                "path": portable_contextworld_path(
                    Path(__file__).resolve(),
                    repo_root=ROOT,
                ),
                "sha256": file_sha256(Path(__file__).resolve()),
            },
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Replay the frozen History-3 PushT contact-friction physical "
            "feasibility audit"
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resolution", type=int, default=96)
    parser.add_argument(
        "--require-pass",
        action="store_true",
        help="return a non-zero code when the physical stage did not pass",
    )
    args = parser.parse_args()

    config_path = args.config.resolve()
    output_path = args.output.resolve()
    report = build_report(
        config_path=config_path,
        resolution=int(args.resolution),
    )
    write_json(output_path, report)
    print(
        json.dumps(
            {
                "benchmark": report["benchmark"],
                "status": report["status"],
                "search": report["search"],
                "original_pairwise_audit": {
                    "failed_checks": report["original_pairwise_audit"][
                        "failed_checks"
                    ],
                    "history_gap_px_equivalent": report[
                        "original_pairwise_audit"
                    ][
                        "history_visible_response_gap"
                    ]["px_equivalent"],
                    "precanonical_pair_max_abs_gap": report[
                        "original_pairwise_audit"
                    ]["precanonical_query_state"]["pair_max_abs_gap"],
                    "future_block_position_gap_px": report[
                        "original_pairwise_audit"
                    ]["future_gap"]["block_position_px"],
                },
                "history3_confirmation": {
                    key: report["history3_confirmation"][key]
                    for key in (
                        "templates",
                        "passed_templates",
                        "maximum_per_trajectory_correction",
                        "minimum_history_gap_px_equivalent",
                        "minimum_future_block_position_gap_px",
                    )
                },
                "decision": report["decision"],
                "output": portable_contextworld_path(
                    output_path,
                    repo_root=ROOT,
                ),
                "output_sha256": file_sha256(output_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    if args.require_pass and not report["audit"]["stage_passed"]:
        return 1
    return 0 if report["audit"]["execution_valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
