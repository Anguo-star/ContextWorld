#!/usr/bin/env python3
"""Freeze the positive Speed PLDM CEM branch after a passed 3/3 ICL gate.

This is a binding freezer, not a CEM runner.  It is deliberately the only
bridge from the immutable recovered Public-ICL outcome to either CEM
namespace.  It snapshots the full pre-CEM chain, both registered planning
protocols, the three final checkpoints and model-state hashes, and the exact
canonical destinations.  It never constructs an environment or calls a CEM
solver.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from contextworld.benchmarks.speed_pldm_infrastructure_development import (
    COMPLETION_ID,
    EXPECTED_SEEDS,
    identity,
    logical_path,
    resolve_local_output,
    resolve_source,
    root,
)


ROOT = root()
CEM_PREREG = ROOT / "configs/benchmark/tworoom_speed_pldm_cem_prereg_v1.yaml"
AGGREGATE_PREREG = (
    ROOT / "configs/benchmark/contextworld_pldm_reference_completion_aggregate_prereg_v1.yaml"
)
EVALUATION_BINDING = (
    ROOT / "configs/benchmark/tworoom_speed_pldm_evaluation_binding_v1.yaml"
)
EVALUATION_BINDING_RECEIPT = (
    ROOT
    / "artifacts/evaluation/history3/tworoom_speed_pldm_reference_completion_v1"
    / "evaluation_binding_v1/evaluation_binding_receipt.json"
)
RECOVERY_PREREG = (
    ROOT / "configs/benchmark/tworoom_speed_pldm_formal_icl_recovery_v1.yaml"
)
FORMAL_ROOT = (
    ROOT
    / "artifacts/evaluation/history3/tworoom_speed_pldm_reference_completion_v1"
    / "formal_icl_v1"
)
DEFAULT_OUTPUT = FORMAL_ROOT / "cem_binding_v1.json"
CEM_ROOTS = {
    "action_planning_cem": (
        ROOT
        / "artifacts/evaluation/history3/tworoom_speed_pldm_reference_completion_v1"
        / "formal_action_planning_cem_v1"
    ),
    "original_task_retention_cem": (
        ROOT
        / "artifacts/evaluation/history3/tworoom_speed_pldm_reference_completion_v1"
        / "formal_original_tworoom_retention_cem_v1"
    ),
}

CEM_BINDING_ID = "tworoom_speed_pldm_cem_binding_v1"
RELEASE_ID = "contextworld_tworoom_speed_icl_history3_v1"

# These are deliberate literal pre-Public identities, not values discovered
# after the ICL gate.  The preregistration repeats them so its human-readable
# declaration and this fail-closed validator independently reject a path-only
# or silently substituted planning implementation.
STATIC_IDENTITY_EXPECTATIONS: dict[str, dict[str, Any]] = {
    "source_protocol": {
        "path": "configs/benchmark/tworoom_speed_cube_eval_v2.yaml",
        "sha256": "387e5bd678ccf04e5ced9cd967b218f2014feefade09a0c5ab6989e2df62f2e9",
        "size_bytes": 5487,
    },
    "aggregate_preregistration": {
        "path": "configs/benchmark/contextworld_pldm_reference_completion_aggregate_prereg_v1.yaml",
        "sha256": "84bbfff198663846f1e597ecfb65151697cd3754d651fce608023ff92d5b106a",
        "size_bytes": 11219,
    },
    "action_catalog": {
        "path": "artifacts/evaluation/history3/speed_isolated_v2/catalogs/seen_for_multi.json",
        "sha256": "f8bb83e008398378e71c9345de13144b52f08b9bcb6ad02653deddc2020a9152",
        "size_bytes": 176987,
    },
    "action_catalog_validator": {
        "path": "contextworld/evaluation/icl_catalog.py",
        "sha256": "82a76a70c6380a4109d3909c06f04e2fdc1a6b7f94ce572a30df0925c7e5398f",
        "size_bytes": 30306,
    },
    "action_episode_oracle": {
        "path": "scripts/eval_tworoom_icl_planning.py",
        "sha256": "44f912dd73afdb93003272a498c8c474ab4f6d961bd85a3d007a1c9217544a4d",
        "size_bytes": 35457,
    },
    "action_runner_core": {
        "path": "scripts/eval_tworoom_speed_cube_planning.py",
        "sha256": "22184323ec39ab049c94df2c73af31465c6ebcf881b2d351d112756afe7e794c",
        "size_bytes": 12971,
    },
    "speed_cli": {
        "path": "contextworld/benchmarks/speed_icl_cli.py",
        "sha256": "886aa06cb57cf9f660868b89ab1c334a411ac82f84adf3f14ea3115956b878b0",
        "size_bytes": 16298,
    },
    "speed_score": {
        "path": "contextworld/benchmarks/speed_icl_score.py",
        "sha256": "0d7d52667eea0a29badcac1cee1047f3a2d56d55f65a59a0fd7a086a77601a1e",
        "size_bytes": 32831,
    },
    "retention_catalog": {
        "path": "artifacts/evaluation/history3/original_ability_reconstruction/original_heldout_eval_catalog.json",
        "sha256": "08233c6755c8f6f358f5fe35f84e7cfed626b922be0eeaba8f7392c7741b4b34",
        "size_bytes": 119993,
    },
    "retention_catalog_builder": {
        "path": "scripts/build_tworoom_original_ability_catalogs.py",
        "sha256": "534b131929de3d005439c8dbe7a9f60a8f2877b6d760ac26568642f6b6aa90c3",
        "size_bytes": 9768,
    },
    "retention_episode_oracle": {
        "path": "scripts/eval_tworoom_ability_catalog.py",
        "sha256": "8bec7a6d4315850f985141697ad1a04c21fd60478c6439e40edb5c506a11bcea",
        "size_bytes": 12569,
    },
    "retention_runner_core": {
        "path": "scripts/eval_tworoom_ability_catalog.py",
        "sha256": "8bec7a6d4315850f985141697ad1a04c21fd60478c6439e40edb5c506a11bcea",
        "size_bytes": 12569,
    },
    "retention_frozen_baseline_wrapper": {
        "path": "scripts/eval_tworoom_original_baseline_cem_frozen_v1.py",
        "sha256": "03e3863c6f3f9dc559095eeb01654f6b8e49675ce8dc90548b6aec0fdef106a6",
        "size_bytes": 12353,
    },
    "implementation_paired_retention_comparator": {
        "path": "scripts/analyze_tworoom_original_ability.py",
        "sha256": "92de93da3c96de7de758c29959b3491958a47d8a2d740667a921608ce1d6db73",
        "size_bytes": 23276,
    },
    "implementation_formal_runner": {
        "path": "scripts/run_tworoom_speed_pldm_cem_v1.py",
        "sha256": "670231be8796a7cc2a03ab0ea1906ace20f2812fc57144504f46583d1db441af",
        "size_bytes": 35648,
    },
    "implementation_aggregate_freezer": {
        "path": "scripts/freeze_tworoom_speed_pldm_cem_aggregate_v1.py",
        "sha256": "c02c2116bc8724d9d0296ef80ec6d7152140bd1f16639467c5459dfd64f89c67",
        "size_bytes": 23809,
    },
    "implementation_adapter_boundary": {
        "path": "contextworld/benchmarks/adapters.py",
        "sha256": "cc9e758b7081a57251e8cd026e9ac9ff8a17e3f300d52f464bdad871edcf26b2",
        "size_bytes": 24085,
    },
    "implementation_development_contract": {
        "path": "contextworld/benchmarks/speed_pldm_infrastructure_development.py",
        "sha256": "89e12e8d8b7effbf14445335fa9e013d58e746ae3cd31dd6c233470b3e8c3c14",
        "size_bytes": 11633,
    },
    "retention_noninferiority_protocol": {
        "path": "configs/benchmark/tworoom_original_ability_reconstruction_v1.yaml",
        "sha256": "65e3da088a3cff18a66f407b0d03c740b250ead86da5f94d2456f67588fb9bf0",
        "size_bytes": 3749,
    },
}


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return value


def _source(path: str | Path) -> dict[str, Any]:
    return identity(resolve_source(path, repo_root=ROOT), repo_root=ROOT)


def _same_identity(left: Any, right: Any) -> bool:
    return bool(
        isinstance(left, Mapping)
        and isinstance(right, Mapping)
        and left.get("path") == right.get("path")
        and left.get("sha256") == right.get("sha256")
        and left.get("size_bytes") == right.get("size_bytes")
    )


def _require_static_identity(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an immutable identity")
    path = value.get("path")
    expected = value.get("sha256")
    expected_size = value.get("size_bytes")
    if not (
        isinstance(path, str)
        and path
        and isinstance(expected, str)
        and isinstance(expected_size, int)
    ):
        raise ValueError(f"{label} needs path, SHA-256, and byte size")
    observed = _source(path)
    if observed["sha256"] != expected or observed["size_bytes"] != expected_size:
        raise RuntimeError(f"{label} identity drifted")
    return observed


def _require_expected_static_identity(
    value: Any, *, key: str, label: str
) -> dict[str, Any]:
    expected = STATIC_IDENTITY_EXPECTATIONS[key]
    if not _same_identity(value, expected):
        raise ValueError(f"{label} differs from its preregistered exact identity")
    return _require_static_identity(value, label=label)


def _write_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o644)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _assert_output(path: Path) -> Path:
    expected = DEFAULT_OUTPUT.resolve()
    actual = resolve_local_output(path, repo_root=ROOT)
    if actual != expected:
        raise ValueError(
            "CEM binding output must equal its dedicated destination "
            f"{logical_path(expected, repo_root=ROOT)}"
        )
    return actual


def _git_head(worktree: Path) -> str:
    pointer = worktree / ".git"
    if pointer.is_file():
        text = pointer.read_text(encoding="utf-8").strip()
        if not text.startswith("gitdir: "):
            raise RuntimeError(f"Unsupported git pointer: {pointer}")
        gitdir = Path(text[len("gitdir: ") :]).expanduser()
    elif pointer.is_dir():
        gitdir = pointer
    else:
        raise FileNotFoundError(f"Stable-WorldModel worktree lacks .git: {worktree}")
    head = (gitdir / "HEAD").read_text(encoding="utf-8").strip()
    if head.startswith("ref: "):
        ref = head[len("ref: ") :]
        target = gitdir / ref
        if not target.is_file():
            common = gitdir / "commondir"
            if not common.is_file():
                raise RuntimeError(f"Cannot resolve Stable-WorldModel ref: {ref}")
            target = (gitdir / common.read_text(encoding="utf-8").strip() / ref).resolve()
        head = target.read_text(encoding="utf-8").strip()
    if len(head) != 40:
        raise RuntimeError("Stable-WorldModel HEAD is not a commit SHA")
    return head


def _validate_static_prereg(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate the pre-outcome declaration without consuming CEM results."""

    payload = _load_yaml(path)
    static_identity = _source(path)
    chronology = payload.get("chronology")
    authorization = payload.get("authorization")
    common = payload.get("common")
    tracks = payload.get("tracks")
    outputs = payload.get("outputs")
    if not (
        payload.get("schema_version") == 1
        and payload.get("cem_preregistration_id") == "tworoom_speed_pldm_cem_prereg_v1"
        and payload.get("completion_id") == COMPLETION_ID
        and payload.get("release_id") == RELEASE_ID
        and payload.get("status")
        == "preregistered_during_fixed_training_before_public_icl_or_cem"
        and chronology
        == {
            "fixed_training_already_running": True,
            "development_evaluation_started": False,
            "public_test_opened": False,
            "cem_started": False,
            "checkpoint_selection_changed": False,
            "training_budget_changed": False,
        }
        and authorization
        == {
            "requires_passed_three_seed_public_icl_recovery_aggregate": True,
            "requires_cem_binding_after_public_icl_before_any_cem": True,
            "all_three_fixed_training_seeds_required": list(EXPECTED_SEEDS),
            "only_fixed_final_checkpoints": True,
            "retry_or_checkpoint_selection_authorized": False,
            "output_overwrite_authorized": False,
        }
        and isinstance(common, Mapping)
        and common.get("eval_seeds") == [42, 43, 44, 45, 46, 47]
        and common.get("episodes_per_eval_seed") == 50
        and common.get("episodes_per_checkpoint") == 300
        and common.get("successful_episode_definition", {}).get("radius_px") == 16
        and isinstance(tracks, Mapping)
        and set(tracks) == {"action_planning_cem", "original_task_retention_cem"}
        and outputs
        == {
            "cem_binding": logical_path(DEFAULT_OUTPUT, repo_root=ROOT),
            "action_planning": {
                "root": logical_path(CEM_ROOTS["action_planning_cem"], repo_root=ROOT),
                "receipts": "seed_{training_seed}.jsonl",
                "work": "work/seed_{training_seed}",
                "aggregate": "three_seed_aggregate.json",
            },
            "original_task_retention": {
                "root": logical_path(CEM_ROOTS["original_task_retention_cem"], repo_root=ROOT),
                "receipts": "seed_{training_seed}.jsonl",
                "work": "work/seed_{training_seed}",
                "aggregate": "three_seed_aggregate.json",
            },
        }
    ):
        raise ValueError("Speed positive-CEM preregistration contract is invalid")

    frozen = payload.get("frozen_sources")
    if not isinstance(frozen, Mapping):
        raise ValueError("Speed positive-CEM preregistration lacks frozen sources")
    completion = _require_static_identity(frozen.get("completion_config"), label="completion config")
    release = _require_static_identity(frozen.get("speed_release"), label="Speed release")
    boundary = _require_static_identity(
        frozen.get("behavioral_claim_boundary"), label="behavioral claim boundary"
    )
    source_protocol = _require_expected_static_identity(
        frozen.get("source_protocol"), key="source_protocol", label="Speed source protocol"
    )
    aggregate_prereg = _require_expected_static_identity(
        frozen.get("aggregate_preregistration"),
        key="aggregate_preregistration",
        label="aggregate preregistration",
    )
    retention_noninferiority = _require_expected_static_identity(
        frozen.get("retention_noninferiority_protocol"),
        key="retention_noninferiority_protocol",
        label="retention non-inferiority protocol",
    )
    action = tracks["action_planning_cem"]
    retention = tracks["original_task_retention_cem"]
    action_source = action.get("source", {})
    retention_source = retention.get("source", {})
    if not isinstance(action_source, Mapping) or not isinstance(retention_source, Mapping):
        raise ValueError("Speed CEM tracks lack source mappings")
    action_catalog = _require_expected_static_identity(
        action_source.get("catalog"), key="action_catalog", label="action catalog"
    )
    action_catalog_validator = _require_expected_static_identity(
        action_source.get("catalog_validator"),
        key="action_catalog_validator",
        label="action catalog validator",
    )
    action_episode_oracle = _require_expected_static_identity(
        action_source.get("episode_oracle"),
        key="action_episode_oracle",
        label="action episode oracle",
    )
    action_runner_core = _require_expected_static_identity(
        action_source.get("runner_core"), key="action_runner_core", label="action runner core"
    )
    speed_cli = _require_expected_static_identity(
        action_source.get("speed_cli"), key="speed_cli", label="Speed CLI"
    )
    speed_score = _require_expected_static_identity(
        action_source.get("speed_score"), key="speed_score", label="Speed scorer"
    )
    if not _same_identity(action_source.get("source_protocol"), source_protocol):
        raise ValueError("Action source protocol differs from frozen source protocol")
    retention_catalog = _require_expected_static_identity(
        retention_source.get("catalog"), key="retention_catalog", label="retention catalog"
    )
    retention_catalog_builder = _require_expected_static_identity(
        retention_source.get("catalog_builder"),
        key="retention_catalog_builder",
        label="retention catalog builder",
    )
    retention_episode_oracle = _require_expected_static_identity(
        retention_source.get("episode_oracle"),
        key="retention_episode_oracle",
        label="retention episode oracle",
    )
    retention_runner_core = _require_expected_static_identity(
        retention_source.get("runner_core"),
        key="retention_runner_core",
        label="retention runner core",
    )
    retention_frozen_baseline_wrapper = _require_expected_static_identity(
        retention_source.get("frozen_baseline_wrapper"),
        key="retention_frozen_baseline_wrapper",
        label="retention frozen baseline wrapper",
    )
    implementation = payload.get("runtime_and_implementation")
    if not isinstance(implementation, Mapping):
        raise ValueError("Speed CEM preregistration lacks implementation identities")
    implementation_identities = {}
    for name in (
        "formal_runner",
        "aggregate_freezer",
        "binding_freezer",
        "adapter_boundary",
        "development_contract",
        "paired_retention_comparator",
    ):
        key = f"implementation_{name}"
        # Pinning this freezer's *own* SHA in its source would be a self-hash
        # cycle.  The outer preregistration is its authority: it stores the
        # complete path/SHA/size and this code verifies that declaration with
        # _require_static_identity.  Every other implementation additionally
        # has a literal independent expectation above.
        implementation_identities[name] = (
            _require_static_identity(implementation.get(name), label=f"implementation {name}")
            if name == "binding_freezer"
            else _require_expected_static_identity(
                implementation.get(name), key=key, label=f"implementation {name}"
            )
        )
    if not (
        action.get("evaluation_kind") == "action_planning_cem"
        and action_source.get("release_planning_track") == "seen_for_multi"
        and action_catalog["path"]
        == "artifacts/evaluation/history3/speed_isolated_v2/catalogs/seen_for_multi.json"
        and action.get("grouping", {}).get("query_speed") == 5.1
        and action.get("grouping", {}).get("history_condition") == "history_mid"
        and action.get("grouping", {}).get("same_speed_condition_required") is True
        and action.get("grouping", {}).get("expected_base_queries") == 18
        and action.get("grouping", {}).get("expected_records_per_checkpoint") == 300
        and action.get("grouping", {}).get("expected_records_per_eval_seed") == 50
        and action.get("protocol")
        == {
            "eval_budget_raw_steps": 100,
            "deadline_budgets_raw_steps": [50, 75, 100],
            "horizon_action_blocks": 10,
            "receding_horizon_action_blocks": 5,
            "cem_samples": 300,
            "cem_iterations": 30,
            "cem_topk": 30,
            "cem_var_scale": 1.0,
        }
        and action.get("metric", {}).get("id")
        == "success_rate_by_execution_budget_100_raw_steps"
        and action.get("metric", {}).get("performance_threshold") is None
        and action.get("metric", {}).get("result_semantics")
        == "EXECUTED_VALID_DESCRIPTIVE"
        and retention.get("evaluation_kind") == "original_task_retention_cem"
        and retention_source.get("completion_field")
        == "evaluation.original_task_retention"
        and retention.get("grouping", {}).get("catalog_kind")
        == "tworoom_original_heldout_eval_catalog_v1"
        and retention.get("grouping", {}).get("expected_records_per_checkpoint") == 300
        and retention.get("grouping", {}).get("expected_records_per_eval_seed") == 50
        and retention.get("protocol")
        == {
            "eval_budget_raw_steps": 50,
            "horizon_action_blocks": 5,
            "receding_horizon_action_blocks": 5,
            "cem_samples": 300,
            "cem_iterations": 30,
            "cem_topk": 30,
        }
        and retention.get("metric", {}).get("id") == "original_tworoom_cem_success_rate"
        and retention.get("metric", {}).get("result_semantics")
        == "PAIRED_NONINFERIORITY_RETENTION"
        and retention.get("metric", {}).get("paired_noninferiority")
        == {
            "reference": "frozen_original_pldm_cem_6x50",
            "confidence_level": 0.95,
            "paired_bootstrap_seed": 3072,
            "paired_bootstrap_resamples": 10000,
            "success_rate_delta_lower_bound": -0.05,
            "final_distance_delta_upper_bound_px": 5.0,
            "require_no_solvable_room_relation_stratum_collapse": True,
            "stratum_definition": "room_relation",
            "collapse_definition": "candidate_zero_successes_where_baseline_has_at_least_one",
        }
    ):
        raise ValueError("Speed CEM-track preregistration drifted")
    return payload, {
        "preregistration": static_identity,
        "completion_config": completion,
        "release_config": release,
        "behavioral_claim_boundary": boundary,
        "source_protocol": source_protocol,
        "aggregate_preregistration": aggregate_prereg,
        "retention_noninferiority_protocol": retention_noninferiority,
        "action_catalog": action_catalog,
        "action_catalog_validator": action_catalog_validator,
        "action_episode_oracle": action_episode_oracle,
        "action_runner_core": action_runner_core,
        "speed_cli": speed_cli,
        "speed_score": speed_score,
        "retention_catalog": retention_catalog,
        "retention_catalog_builder": retention_catalog_builder,
        "retention_episode_oracle": retention_episode_oracle,
        "retention_runner_core": retention_runner_core,
        "retention_frozen_baseline_wrapper": retention_frozen_baseline_wrapper,
        **{f"implementation_{name}": value for name, value in implementation_identities.items()},
    }


def _action_schedule(catalog: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Recreate the declared 6x50 balanced schedule without environment work."""

    bundles = catalog.get("bundles")
    if not isinstance(bundles, list):
        raise ValueError("Action-planning catalog lacks bundles")
    selected = [
        bundle
        for bundle in bundles
        if isinstance(bundle, Mapping)
        and bundle.get("family") == "speed"
        and np.isclose(
            float(bundle.get("query_factors", {}).get("agent.speed", float("nan"))),
            5.1,
            rtol=0.0,
            atol=1e-6,
        )
    ]
    selected.sort(key=lambda row: str(row.get("template", {}).get("template_id", "")))
    if len(selected) != 18 or len({row.get("query_id") for row in selected}) != 18:
        raise ValueError("Action-planning catalog does not contain its 18 fixed query bundles")
    for bundle in selected:
        if not (
            bundle.get("track") == "seen_for_multi"
            and bundle.get("same_speed_condition") == "history_mid"
            and bundle.get("conditions", {}).get("history_mid", {}).get("factors", {}).get(
                "agent.speed"
            )
            == 5.1
        ):
            raise ValueError("Action-planning same-speed catalog condition drifted")

    rows: list[dict[str, Any]] = []
    for eval_seed in (42, 43, 44, 45, 46, 47):
        rng = np.random.default_rng(eval_seed)
        indices: list[int] = []
        while len(indices) < 50:
            order = rng.permutation(len(selected))
            indices.extend(int(value) for value in order[: 50 - len(indices)])
        occurrences: dict[str, int] = {}
        for evaluation_index, asset_index in enumerate(indices):
            bundle = selected[asset_index]
            query_id = str(bundle["query_id"])
            repeat_index = occurrences.get(query_id, 0)
            occurrences[query_id] = repeat_index + 1
            cem_seed = int(
                np.random.SeedSequence(
                    [int(eval_seed), int(evaluation_index), int(asset_index)]
                ).generate_state(1)[0]
            )
            rows.append(
                {
                    "eval_seed": eval_seed,
                    "evaluation_index": evaluation_index,
                    "evaluation_id": f"s{eval_seed}-e{evaluation_index:03d}-{query_id}",
                    "repeat_index": repeat_index,
                    "cem_seed": cem_seed,
                    "query_id": query_id,
                    "template_id": bundle["template"]["template_id"],
                    "source_scenario_id": bundle["source_scenario_id"],
                    "condition": "history_mid",
                }
            )
    if len(rows) != 300:
        raise AssertionError("Action-planning schedule is not 300 episodes")
    return rows


def _retention_catalog(catalog: Mapping[str, Any]) -> list[dict[str, Any]]:
    entries = catalog.get("entries")
    if not (
        catalog.get("schema_version") == 1
        and catalog.get("catalog") == "tworoom_original_heldout_eval_catalog_v1"
        and isinstance(entries, list)
        and len(entries) == 300
    ):
        raise ValueError("Retention catalog must be the registered 300-row original-heldout catalog")
    rows: list[dict[str, Any]] = []
    for eval_seed in (42, 43, 44, 45, 46, 47):
        selected = [row for row in entries if int(row.get("eval_seed", -1)) == eval_seed]
        selected.sort(key=lambda row: int(row.get("evaluation_index", -1)))
        if not (
            len(selected) == 50
            and [int(row.get("evaluation_index", -1)) for row in selected] == list(range(50))
            and {str(row.get("source_kind", "")) for row in selected} == {"original_h5"}
            and {int(row.get("goal_offset", -1)) for row in selected} == {25}
            and {int(row.get("cem_group_seed", -1)) for row in selected} == {eval_seed}
        ):
            raise ValueError(f"Retention catalog grouping drifted for eval seed {eval_seed}")
        rows.extend(
            {
                "eval_seed": eval_seed,
                "evaluation_index": int(row["evaluation_index"]),
                "evaluation_id": str(row["evaluation_id"]),
                "episode": int(row["episode"]),
                "start_step": int(row["start_step"]),
                "goal_offset": int(row["goal_offset"]),
                "cem_group_seed": int(row["cem_group_seed"]),
            }
            for row in selected
        )
    if len(rows) != 300 or len({row["evaluation_id"] for row in rows}) != 300:
        raise ValueError("Retention catalog does not provide exactly 300 unique evaluations")
    return rows


def _baseline_retention(
    retention: Mapping[str, Any],
    *,
    catalog_identity: Mapping[str, Any],
    retention_schedule: list[dict[str, Any]],
) -> dict[str, Any]:
    """Validate the frozen original-PLDM 6x50 records used for pairing.

    The matrix result is descriptive in its own release, but it is an exact
    paired reference for this separately preregistered retention comparison.
    We reject a summary-only baseline: all six raw receipts must be present,
    immutable, and match the candidate catalog key-for-key.
    """

    baseline = retention.get("frozen_paired_baseline")
    if not isinstance(baseline, Mapping):
        raise ValueError("Retention preregistration lacks its frozen paired baseline")
    results_freeze = _require_static_identity(
        baseline.get("results_freeze"), label="retention baseline results freeze"
    )
    matrix_summary = _require_static_identity(
        baseline.get("matrix_summary"), label="retention baseline matrix summary"
    )
    freeze_payload = _load_json(resolve_source(results_freeze["path"], repo_root=ROOT))
    summary_payload = _load_json(resolve_source(matrix_summary["path"], repo_root=ROOT))
    if not (
        freeze_payload.get("freeze_id")
        == "contextworld_original_baseline_cem_results_freeze_v1"
        and freeze_payload.get("status") == "frozen_after_completed_descriptive_matrix"
        and freeze_payload.get("matrix_summary") == {
            **matrix_summary,
            "summary_id": "contextworld_original_baseline_cem_matrix_v1",
            "status": "completed_descriptive_original_environment_cem_matrix",
            "matrix_cells": 8,
            "episodes_per_cell": 300,
            "total_matrix_episodes": 2400,
            "newly_executed_cells": 7,
            "newly_executed_episodes": 2100,
            "strictly_reused_cells": 1,
            "receipt_identities_embedded_in_summary": True,
            "all_model_state_audits_passed": True,
        }
        and summary_payload.get("summary_id") == "contextworld_original_baseline_cem_matrix_v1"
        and summary_payload.get("status")
        == "completed_descriptive_original_environment_cem_matrix"
    ):
        raise RuntimeError("Retention baseline results-freeze/summary chain is invalid")
    cells = summary_payload.get("cells")
    if not isinstance(cells, list):
        raise RuntimeError("Retention baseline matrix lacks cells")
    matches = [
        row
        for row in cells
        if isinstance(row, Mapping)
        and row.get("environment") == "tworoom"
        and row.get("family") == "pldm"
    ]
    if len(matches) != 1:
        raise RuntimeError("Retention baseline matrix lacks exactly one TwoRoom PLDM cell")
    cell = matches[0]
    if not (
        cell.get("checkpoint_id") == baseline.get("checkpoint_id")
        and cell.get("success_count") == baseline.get("expected_successes") == 278
        and cell.get("evaluation_count") == baseline.get("expected_evaluations") == 300
        and cell.get("success_rate") == 278 / 300
        and cell.get("model_state_audit", {}).get("passed") is True
        and cell.get("model_state_audit", {}).get("loaded_state_dict_sha256")
        == baseline.get("checkpoint_model_state_sha256")
        and cell.get("provenance") == "six_seed_receipts"
    ):
        raise RuntimeError("Retention baseline TwoRoom PLDM cell drifted")
    declared = baseline.get("raw_receipts")
    if not isinstance(declared, list) or len(declared) != 6:
        raise RuntimeError("Retention baseline must declare six raw receipts")
    declared_by_seed = {
        int(row.get("eval_seed", -1)): row for row in declared if isinstance(row, Mapping)
    }
    summary_by_path = {
        str(row.get("path")): row for row in cell.get("sources", []) if isinstance(row, Mapping)
    }
    if set(declared_by_seed) != {42, 43, 44, 45, 46, 47}:
        raise RuntimeError("Retention baseline seed set is invalid")
    rows: list[dict[str, Any]] = []
    candidate_by_id = {row["evaluation_id"]: row for row in retention_schedule}
    for eval_seed in (42, 43, 44, 45, 46, 47):
        item = declared_by_seed[eval_seed]
        receipt_identity = _require_static_identity(
            item, label=f"retention baseline raw receipt {eval_seed}"
        )
        summary_item = summary_by_path.get(str(resolve_source(item["path"], repo_root=ROOT)))
        # Matrix summaries may retain absolute locations; identity equality is
        # the robust check and prevents a path-only rebinding.
        if summary_item != receipt_identity:
            raise RuntimeError(f"Retention baseline summary disagrees for eval seed {eval_seed}")
        payload = _load_json(resolve_source(receipt_identity["path"], repo_root=ROOT))
        raw_records = payload.get("raw_records")
        if not (
            payload.get("schema_version") == 1
            and payload.get("benchmark") == "tworoom_original_ability_planning_v1"
            and payload.get("status") == "passed"
            and payload.get("catalog", {}).get("kind")
            == "tworoom_original_heldout_eval_catalog_v1"
            and payload.get("catalog", {}).get("sha256") == catalog_identity["sha256"]
            and payload.get("normalizer", {}).get("sha256")
            == "a9e4b443bbac0d7a4e2d9d9f84d40ac40936556ffadfc7af0b0ce4fe4afed42c"
            and payload.get("stable_worldmodel", {}).get("commit")
            == "5864b74980f6ed328fd0045e777b3865962eff43"
            and payload.get("protocol")
            == {
                "action_block": 5,
                "cem_samples": 300,
                "cem_steps": 30,
                "cem_topk": 30,
                "eval_budget": 50,
                "eval_seed": eval_seed,
                "evaluations": 50,
                "history_size": 3,
                "horizon": 5,
                "receding_horizon": 5,
            }
            and payload.get("frozen_weight_audit", {}).get("passed") is True
            and payload.get("frozen_weight_audit", {}).get("state_dict_sha256_before")
            == baseline.get("checkpoint_model_state_sha256")
            and payload.get("frozen_weight_audit", {}).get("state_dict_sha256_after")
            == baseline.get("checkpoint_model_state_sha256")
            and isinstance(raw_records, list)
            and len(raw_records) == 50
        ):
            raise RuntimeError(f"Retention baseline raw receipt is invalid for eval seed {eval_seed}")
        raw_records.sort(key=lambda row: int(row.get("evaluation_index", -1)))
        if [int(row.get("evaluation_index", -1)) for row in raw_records] != list(range(50)):
            raise RuntimeError(f"Retention baseline ordering is invalid for eval seed {eval_seed}")
        for record in raw_records:
            evaluation_id = str(record.get("evaluation_id", ""))
            candidate = candidate_by_id.get(evaluation_id)
            matching = {
                "eval_seed": int(record.get("eval_seed", -1)),
                "evaluation_index": int(record.get("evaluation_index", -1)),
                "episode": int(record.get("episode", -1)),
                "start_step": int(record.get("start_step", -1)),
                "goal_offset": int(record.get("goal_offset", -1)),
                "cem_group_seed": int(record.get("cem_group_seed", -1)),
            }
            if candidate is None or matching != {
                key: candidate[key]
                for key in (
                    "eval_seed",
                    "evaluation_index",
                    "episode",
                    "start_step",
                    "goal_offset",
                    "cem_group_seed",
                )
            }:
                raise RuntimeError(
                    f"Retention baseline query schedule is not paired for {evaluation_id}"
                )
            if not (
                isinstance(record.get("success"), bool)
                and isinstance(record.get("final_distance"), (int, float))
                and isinstance(record.get("room_relation"), str)
            ):
                raise RuntimeError(f"Retention baseline record is incomplete for {evaluation_id}")
            rows.append(
                {
                    "evaluation_id": evaluation_id,
                    "success": bool(record["success"]),
                    "final_distance": float(record["final_distance"]),
                    "room_relation": str(record["room_relation"]),
                }
            )
    if len(rows) != 300 or len({row["evaluation_id"] for row in rows}) != 300:
        raise RuntimeError("Retention baseline does not expose exactly 300 paired records")
    if sum(row["success"] for row in rows) != 278:
        raise RuntimeError("Retention baseline success count does not equal frozen 278/300")
    return {
        "results_freeze": results_freeze,
        "matrix_summary": matrix_summary,
        "checkpoint_id": baseline["checkpoint_id"],
        "checkpoint_model_state_sha256": baseline["checkpoint_model_state_sha256"],
        "raw_receipts": [
            {"eval_seed": int(row["eval_seed"]), "receipt": _require_static_identity(row, label="retention baseline receipt")}
            for row in sorted(declared, key=lambda item: int(item["eval_seed"]))
        ],
        "expected": {"successes": 278, "evaluations": 300},
    }


def _passed_public_chain() -> dict[str, Any]:
    """Use the finalizer's independent validators for the raw/recovery chain."""

    from contextworld.benchmarks import pldm_reference_completion_aggregate as finalizer

    preregistration = finalizer.load_completion_aggregate_preregistration(
        AGGREGATE_PREREG, repo_root=ROOT
    )
    specification = preregistration["completion_inputs"]["speed"]
    completion, completion_identity, release_path, seeds = finalizer._load_completion(
        specification, repo_root=ROOT
    )
    if tuple(seeds) != EXPECTED_SEEDS:
        raise RuntimeError("Speed finalization does not use the registered seed set")
    release = finalizer._read_yaml(str(release_path), repo_root=ROOT)
    public = finalizer._public_icl_from_recovery(
        "speed",
        specification,
        completion_id=COMPLETION_ID,
        release_id=RELEASE_ID,
        expected_seeds=seeds,
        repo_root=ROOT,
    )
    if not (
        completion.get("completion_id") == COMPLETION_ID
        and release.get("release_id") == RELEASE_ID
        and public.get("ability_passed") is True
        and public.get("cem") == {"authorized": True, "executed": False}
        and len(public.get("records", [])) == len(EXPECTED_SEEDS)
    ):
        raise RuntimeError("A passed 3/3 recovered Public-ICL chain is required before CEM")
    return {
        "specification": specification,
        "completion": completion_identity,
        "release": finalizer._identity(str(release_path), repo_root=ROOT),
        "public": public,
    }


def _prepublic_cem_authority(
    *,
    binding: Mapping[str, Any],
    static: Mapping[str, Any],
    static_sources: Mapping[str, Any],
) -> dict[str, Any]:
    """Reject any source/protocol selected after the Public ICL result.

    The only accepted CEM source closure is the one copied into the evaluation
    binding before Public ICL.  This routine compares identities as values;
    later ``_snapshot`` calls merely rehash those already-selected identities.
    """

    authority = binding.get("cem_protocol")
    evaluator_sources = binding.get("evaluator_sources")
    if not isinstance(authority, Mapping) or not isinstance(evaluator_sources, Mapping):
        raise RuntimeError("Evaluation binding lacks a pre-Public CEM authority closure")
    source_identities = authority.get("source_identities")
    expected_sources = {
        name: static_sources[name]
        for name in (
            "preregistration",
            "source_protocol",
            "aggregate_preregistration",
            "retention_noninferiority_protocol",
            "action_catalog",
            "action_catalog_validator",
            "action_episode_oracle",
            "action_runner_core",
            "speed_cli",
            "speed_score",
            "retention_catalog",
            "retention_catalog_builder",
            "retention_episode_oracle",
            "retention_runner_core",
            "retention_frozen_baseline_wrapper",
            "implementation_formal_runner",
            "implementation_aggregate_freezer",
            "implementation_binding_freezer",
            "implementation_adapter_boundary",
            "implementation_development_contract",
            "implementation_paired_retention_comparator",
        )
    }
    if not (
        authority.get("cem_preregistration_id") == "tworoom_speed_pldm_cem_prereg_v1"
        and authority.get("status") == "frozen_prepublic_cem_execution_and_decision_authority"
        and authority.get("preregistration") == expected_sources["preregistration"]
        and source_identities == expected_sources
        and all(
            evaluator_sources.get(f"cem_{name}") == item
            for name, item in expected_sources.items()
        )
        and authority.get("completion") == binding.get("completion")
        and authority.get("release") == binding.get("release")
        and authority.get("behavioral_claim_boundary")
        == binding.get("behavioral_claim_boundary")
        and authority.get("normalizer") == binding.get("normalizer")
        and authority.get("stable_worldmodel") == binding.get("stable_worldmodel")
        and authority.get("outputs") == static.get("outputs")
        and authority.get("authority")
        == {
            "all_source_identities_frozen_before_public_icl": True,
            "post_icl_cem_binding_may_only_validate_and_rebind_this_closure": True,
            "action_planning_outcomes_are_descriptive_not_a_model_gate": True,
            "retention_pass_fail_uses_only_frozen_paired_noninferiority": True,
        }
    ):
        raise RuntimeError("Pre-Public CEM authority differs from the static preregistration")
    action = authority.get("tracks", {}).get("action_planning_cem")
    retention = authority.get("tracks", {}).get("original_task_retention_cem")
    static_action = static["tracks"]["action_planning_cem"]
    static_retention = static["tracks"]["original_task_retention_cem"]
    expected_action_source = {
        "release_planning_track": "seen_for_multi",
        "catalog": expected_sources["action_catalog"],
        "source_protocol": expected_sources["source_protocol"],
        "catalog_validator": expected_sources["action_catalog_validator"],
        "episode_oracle": expected_sources["action_episode_oracle"],
        "runner_core": expected_sources["action_runner_core"],
        "speed_cli": expected_sources["speed_cli"],
        "speed_score": expected_sources["speed_score"],
    }
    expected_retention_source = {
        "completion_field": "evaluation.original_task_retention",
        "catalog": expected_sources["retention_catalog"],
        "catalog_builder": expected_sources["retention_catalog_builder"],
        "episode_oracle": expected_sources["retention_episode_oracle"],
        "runner_core": expected_sources["retention_runner_core"],
        "frozen_baseline_wrapper": expected_sources["retention_frozen_baseline_wrapper"],
    }
    if not (
        isinstance(action, Mapping)
        and isinstance(retention, Mapping)
        and action.get("evaluation_kind") == "action_planning_cem"
        and action.get("source") == expected_action_source
        and action.get("grouping") == static_action.get("grouping")
        and action.get("protocol") == static_action.get("protocol")
        and action.get("metric") == static_action.get("metric")
        and retention.get("evaluation_kind") == "original_task_retention_cem"
        and retention.get("source") == expected_retention_source
        and retention.get("grouping") == static_retention.get("grouping")
        and retention.get("protocol") == static_retention.get("protocol")
        and retention.get("metric") == static_retention.get("metric")
        and retention.get("frozen_paired_baseline")
        == static_retention.get("frozen_paired_baseline")
        and authority.get("implementation")
        == {
            "formal_runner": expected_sources["implementation_formal_runner"],
            "aggregate_freezer": expected_sources["implementation_aggregate_freezer"],
            "binding_freezer": expected_sources["implementation_binding_freezer"],
            "adapter_boundary": expected_sources["implementation_adapter_boundary"],
            "development_contract": expected_sources["implementation_development_contract"],
            "paired_retention_comparator": expected_sources[
                "implementation_paired_retention_comparator"
            ],
        }
    ):
        raise RuntimeError("Pre-Public CEM track protocols are not the registered closure")
    return dict(authority)


def _binding_chain(static: dict[str, Any], static_sources: dict[str, Any]) -> dict[str, Any]:
    """Validate and flatten every identity a CEM result must later rebind."""

    passed = _passed_public_chain()
    specification = passed["specification"]
    public = passed["public"]
    if not (
        _same_identity(static_sources["completion_config"], passed["completion"])
        and _same_identity(static_sources["release_config"], passed["release"])
    ):
        raise RuntimeError("Static CEM preregistration does not bind finalization's completion/release")

    binding_config = _source(EVALUATION_BINDING)
    binding_receipt = _source(EVALUATION_BINDING_RECEIPT)
    recovery_prereg = _source(RECOVERY_PREREG)
    binding_payload = _load_yaml(resolve_source(binding_config["path"], repo_root=ROOT))
    receipt_payload = _load_json(resolve_source(binding_receipt["path"], repo_root=ROOT))
    prepublic_cem = _prepublic_cem_authority(
        binding=binding_payload, static=static, static_sources=static_sources
    )
    if not (
        binding_payload.get("binding_id") == "tworoom_speed_pldm_evaluation_binding_v1"
        and binding_payload.get("completion", {}).get("completion_id") == COMPLETION_ID
        and receipt_payload.get("status") == "passed_evaluation_binding_freeze"
        and receipt_payload.get("passed") is True
        and receipt_payload.get("binding", {}).get("sha256") == binding_config["sha256"]
        and receipt_payload.get("development") == public["development"]
        and receipt_payload.get("behavioral_claim_boundary")
        == static_sources["behavioral_claim_boundary"]
        and binding_payload.get("development") == public["development"]
        and binding_payload.get("behavioral_claim_boundary")
        == static_sources["behavioral_claim_boundary"]
    ):
        raise RuntimeError("Evaluation binding does not preserve the passed Public-ICL chain")

    raw_rows = []
    recovery_rows = []
    checkpoints = []
    binding_by_seed = {
        int(row["seed"]): row
        for row in binding_payload.get("checkpoints", [])
        if isinstance(row, Mapping) and isinstance(row.get("seed"), int)
    }
    if set(binding_by_seed) != set(EXPECTED_SEEDS):
        raise RuntimeError("Evaluation binding lacks one of the three checkpoints")
    public_rows = {int(row["training_seed"]): row for row in public["records"]}
    if set(public_rows) != set(EXPECTED_SEEDS):
        raise RuntimeError("Recovered Public-ICL aggregate lacks one of the three checkpoints")
    for seed in EXPECTED_SEEDS:
        bound = binding_by_seed[seed]
        checkpoint = bound.get("checkpoint")
        if not isinstance(checkpoint, Mapping):
            raise RuntimeError(f"Evaluation binding lacks checkpoint identity for seed {seed}")
        checkpoint_identity = _source(checkpoint.get("path", ""))
        state = checkpoint.get("model_state_sha256")
        public_row = public_rows[seed]
        raw_identity = public_row["source"]
        recovery_identity = public_row["recovery_receipt"]
        raw_payload = _load_json(resolve_source(raw_identity["path"], repo_root=ROOT))
        recovery_payload = _load_json(resolve_source(recovery_identity["path"], repo_root=ROOT))
        if not (
            _same_identity(checkpoint_identity, checkpoint)
            and isinstance(state, str)
            and len(state) == 64
            and raw_payload.get("model", {}).get("checkpoint_sha256")
            == checkpoint_identity["sha256"]
            and raw_payload.get("completion_evaluation", {}).get(
                "checkpoint_model_state_sha256"
            )
            == state
            and raw_payload.get("completion_evaluation", {}).get("development")
            == public["development"]
            and raw_payload.get("completion_evaluation", {}).get("behavioral_claim_boundary")
            == static_sources["behavioral_claim_boundary"]
            and recovery_payload.get("training_seed") == seed
            and recovery_payload.get("checkpoint_sha256") == checkpoint_identity["sha256"]
            and recovery_payload.get("development") == public["development"]
            and recovery_payload.get("behavioral_claim_boundary")
            == static_sources["behavioral_claim_boundary"]
            and recovery_payload.get("preregistration") == recovery_prereg
        ):
            raise RuntimeError(f"Raw/recovery chain drifted for seed {seed}")
        checkpoints.append(
            {
                "seed": seed,
                "run_name": bound.get("run_name"),
                "checkpoint": {**checkpoint_identity, "model_state_sha256": state},
                "checkpoint_config": bound.get("config"),
                "training_report": bound.get("training_report"),
                "loss_trace": bound.get("loss_trace"),
                "preflight": bound.get("preflight"),
            }
        )
        raw_rows.append({"seed": seed, "receipt": raw_identity})
        recovery_rows.append({"seed": seed, "receipt": recovery_identity})
    return {
        "completion": passed["completion"],
        "release": passed["release"],
        "evaluation_binding_config": binding_config,
        "evaluation_binding_receipt": binding_receipt,
        "recovery_preregistration": recovery_prereg,
        "public_icl_aggregate": public["aggregate"],
        "behavioral_claim_boundary": static_sources["behavioral_claim_boundary"],
        "development": public["development"],
        "raw_public_icl": raw_rows,
        "recovery_receipts": recovery_rows,
        "checkpoints": checkpoints,
        "claim_boundary": {
            "paired_single_speed_control_available": False,
            "training_attribution_claim": False,
            "public_test_reopened": False,
            "claim_level": "behavioral_trained_reference_only",
        },
        "prepublic_cem_authority": prepublic_cem,
    }


def _runtime(binding_chain: Mapping[str, Any]) -> dict[str, Any]:
    binding = _load_yaml(
        resolve_source(binding_chain["evaluation_binding_config"]["path"], repo_root=ROOT)
    )
    normalizer = binding.get("normalizer")
    runtime = binding.get("stable_worldmodel")
    if not isinstance(normalizer, Mapping) or not isinstance(runtime, Mapping):
        raise RuntimeError("Evaluation binding lacks normalizer/runtime identities")
    normalizer_identity = _source(normalizer.get("path", ""))
    worktree = Path(str(runtime.get("worktree", ""))).expanduser().resolve()
    pldm_config = worktree / str(runtime.get("pldm_config", ""))
    if not (
        _same_identity(normalizer_identity, normalizer)
        and runtime.get("expected_ref") == _git_head(worktree)
        and pldm_config.is_file()
    ):
        raise RuntimeError("Bound normalizer or Stable-WorldModel runtime drifted")
    pldm_identity = identity(pldm_config, repo_root=ROOT)
    if runtime.get("pldm_config_sha256") != pldm_identity["sha256"]:
        raise RuntimeError("Bound PLDM runtime config drifted")
    return {
        "normalizer": normalizer_identity,
        "stable_worldmodel": {
            "worktree": str(worktree),
            "commit": runtime["expected_ref"],
            "pldm_config": pldm_identity,
        },
    }


def _track_bindings(
    static: Mapping[str, Any],
    runtime: Mapping[str, Any],
    *,
    static_sources: Mapping[str, Any],
) -> dict[str, Any]:
    source_protocol_path = resolve_source(
        static["frozen_sources"]["source_protocol"]["path"], repo_root=ROOT
    )
    source_protocol = _load_yaml(source_protocol_path)
    release_path = resolve_source(static["frozen_sources"]["speed_release"]["path"], repo_root=ROOT)
    release = _load_yaml(release_path)
    tracks = static["tracks"]
    implementation = static["runtime_and_implementation"]
    action = tracks["action_planning_cem"]
    retention = tracks["original_task_retention_cem"]
    action_catalog_path = resolve_source(action["source"]["catalog"]["path"], repo_root=ROOT)
    action_catalog = _load_json(action_catalog_path)
    action_catalog_identity = identity(action_catalog_path, repo_root=ROOT)
    if not _same_identity(action_catalog_identity, static_sources["action_catalog"]):
        raise RuntimeError("Action-planning catalog SHA differs from preregistration")
    schedule = _action_schedule(action_catalog)
    source_planner = source_protocol.get("formal_eval", {}).get("planner")
    release_planning = release.get("planning", {})
    if not (
        source_protocol.get("benchmark") == "tworoom_history3_speed_cube_eval_v2"
        and source_planner
        == {
            "eval_budget_raw_steps": 100,
            "deadline_budgets_raw_steps": [50, 75, 100],
            "horizon_action_blocks": 10,
            "receding_horizon_action_blocks": 5,
            "cem_samples": 300,
            "cem_steps": 30,
            "cem_topk": 30,
            "cem_var_scale": 1.0,
            "success_radius_px": 16,
        }
        and release_planning.get("role") == "supporting_utility_metrics"
        and release_planning.get("required_for_speed_icl_prediction_claim") is False
        and release_planning.get("eval_seeds") == [42, 43, 44, 45, 46, 47]
        and release_planning.get("evaluations_per_speed_condition_per_seed") == 50
        and release_planning.get("tracks", {}).get("seen_for_multi", {}).get("catalog")
        == action["source"]["catalog"]["path"]
        and release_planning.get("tracks", {}).get("seen_for_multi", {}).get("catalog_sha256")
        == action_catalog_identity["sha256"]
    ):
        raise RuntimeError("Action-planning source protocol is not the registered Speed protocol")

    retention_catalog_path = resolve_source(
        retention["source"]["catalog"]["path"], repo_root=ROOT
    )
    retention_catalog = _load_json(retention_catalog_path)
    retention_catalog_identity = identity(retention_catalog_path, repo_root=ROOT)
    if not _same_identity(retention_catalog_identity, static_sources["retention_catalog"]):
        raise RuntimeError("Retention catalog SHA differs from preregistration")
    retention_schedule = _retention_catalog(retention_catalog)
    paired_baseline = _baseline_retention(
        retention,
        catalog_identity=retention_catalog_identity,
        retention_schedule=retention_schedule,
    )
    completion = _load_yaml(
        resolve_source(static["frozen_sources"]["completion_config"]["path"], repo_root=ROOT)
    )
    completion_retention = completion.get("evaluation", {}).get("original_task_retention", {})
    if not (
        completion_retention.get("catalog") == retention["source"]["catalog"]["path"]
        and completion_retention.get("eval_seeds") == [42, 43, 44, 45, 46, 47]
        and completion_retention.get("queries_per_seed") == 50
        and completion_retention.get("total_cem_episodes") == 300
        and completion_retention.get("cem")
        == {
            "samples": 300,
            "iterations": 30,
            "topk": 30,
            "horizon_action_blocks": 5,
            "receding_horizon_action_blocks": 5,
        }
    ):
        raise RuntimeError("Retention source protocol is not the registered Speed completion protocol")

    noninferiority_path = resolve_source(
        static_sources["retention_noninferiority_protocol"]["path"], repo_root=ROOT
    )
    noninferiority = _load_yaml(noninferiority_path).get("evaluation_protocol", {}).get(
        "non_inferiority", {}
    )
    if noninferiority != {
        "reference": "H3-OrigHeldout",
        "confidence_level": 0.95,
        "paired_bootstrap_seed": 3072,
        "paired_bootstrap_resamples": 10000,
        "success_margin_percentage_points": -5.0,
        "final_distance_margin_px": 5.0,
        "require_no_solvable_stratum_collapse": True,
    }:
        raise RuntimeError("Registered original-ability non-inferiority protocol drifted")

    formal_runner = static_sources["implementation_formal_runner"]
    aggregate_freezer = static_sources["implementation_aggregate_freezer"]
    binding_freezer = static_sources["implementation_binding_freezer"]
    adapter = static_sources["implementation_adapter_boundary"]
    development_contract = static_sources["implementation_development_contract"]
    paired_retention_comparator = static_sources["implementation_paired_retention_comparator"]
    action_oracle = static_sources["action_episode_oracle"]
    action_core = static_sources["action_runner_core"]
    action_validator = static_sources["action_catalog_validator"]
    retention_oracle = static_sources["retention_episode_oracle"]
    retention_core = static_sources["retention_runner_core"]
    retention_builder = static_sources["retention_catalog_builder"]
    retention_frozen_baseline_wrapper = static_sources["retention_frozen_baseline_wrapper"]

    def output_for(track: str, *, canonical: str) -> dict[str, Any]:
        root_path = CEM_ROOTS[track]
        if logical_path(root_path, repo_root=ROOT) != canonical:
            raise RuntimeError("CEM root differs from preregistered canonical destination")
        return {
            "root": canonical,
            "receipts": [
                {
                    "seed": seed,
                    "path": f"{canonical}/seed_{seed}.jsonl",
                    "work": f"{canonical}/work/seed_{seed}",
                }
                for seed in EXPECTED_SEEDS
            ],
            "aggregate": f"{canonical}/three_seed_aggregate.json",
        }

    return {
        "shared": {
            **runtime,
            "formal_runner": formal_runner,
            "aggregate_freezer": aggregate_freezer,
            "binding_freezer": binding_freezer,
            "adapter_boundary": adapter,
            "development_contract": development_contract,
            "paired_retention_comparator": paired_retention_comparator,
        },
        "action_planning_cem": {
            "evaluation_kind": action["evaluation_kind"],
            "catalog": action_catalog_identity,
            "oracle": action_oracle,
            "runner_core": action_core,
            "catalog_validator": action_validator,
            "speed_cli": static_sources["speed_cli"],
            "speed_score": static_sources["speed_score"],
            "protocol": action["protocol"],
            "metric": action["metric"],
            "result_semantics": "EXECUTED_VALID_DESCRIPTIVE",
            "expected": {
                "episodes_per_checkpoint": 300,
                "episodes_per_eval_seed": 50,
                "eval_seeds": [42, 43, 44, 45, 46, 47],
                "base_queries": 18,
                "same_speed_condition": "history_mid",
            },
            "schedule": schedule,
            "outputs": output_for(
                "action_planning_cem",
                canonical=static["outputs"]["action_planning"]["root"],
            ),
        },
        "original_task_retention_cem": {
            "evaluation_kind": retention["evaluation_kind"],
            "catalog": retention_catalog_identity,
            "catalog_builder": retention_builder,
            "oracle": retention_oracle,
            "runner_core": retention_core,
            "frozen_baseline_wrapper": retention_frozen_baseline_wrapper,
            "protocol": retention["protocol"],
            "metric": retention["metric"],
            "result_semantics": "PAIRED_NONINFERIORITY_RETENTION",
            "paired_baseline": paired_baseline,
            "expected": {
                "episodes_per_checkpoint": 300,
                "episodes_per_eval_seed": 50,
                "eval_seeds": [42, 43, 44, 45, 46, 47],
                "goal_offset_raw_steps": 25,
                "catalog_kind": "tworoom_original_heldout_eval_catalog_v1",
            },
            "schedule": retention_schedule,
            "outputs": output_for(
                "original_task_retention_cem",
                canonical=static["outputs"]["original_task_retention"]["root"],
            ),
        },
    }


def _snapshot(
    *,
    static_sources: Mapping[str, Any],
    chain: Mapping[str, Any],
    tracks: Mapping[str, Any],
) -> dict[str, Any]:
    """Rehash all file identities that the CEM binding exposes."""

    def observed(value: Mapping[str, Any], label: str) -> dict[str, Any]:
        path = value.get("path")
        if not isinstance(path, str):
            raise ValueError(f"{label} lacks a path")
        item = _source(path)
        if not _same_identity(item, value):
            raise RuntimeError(f"{label} drifted while freezing CEM binding")
        return item

    def observed_tree(value: Any, label: str) -> Any:
        """Preserve and rehash every identity nested in the CEM authority.

        ``prepublic_cem_authority`` is deliberately a rich, nested contract:
        it carries both source identities and the semantic policy they govern.
        Storing it as an opaque copied mapping would make the binding's
        before/after proof blind to a source replacement inside that closure.
        This walker retains the full declared structure while independently
        rehashing every identity-shaped mapping it encounters.
        """

        if isinstance(value, Mapping):
            if {
                "path",
                "sha256",
                "size_bytes",
            }.issubset(value):
                refreshed = observed(value, label)
                return {**dict(value), **refreshed}
            return {
                str(key): observed_tree(item, f"{label}.{key}")
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [observed_tree(item, f"{label}[{index}]") for index, item in enumerate(value)]
        return value

    raw = [
        {"seed": row["seed"], "receipt": observed(row["receipt"], f"raw {row['seed']}")}
        for row in chain["raw_public_icl"]
    ]
    recovery = [
        {
            "seed": row["seed"],
            "receipt": observed(row["receipt"], f"recovery {row['seed']}")
        }
        for row in chain["recovery_receipts"]
    ]
    checkpoints = []
    for row in chain["checkpoints"]:
        checkpoint = observed(row["checkpoint"], f"checkpoint {row['seed']}")
        checkpoints.append(
            {
                "seed": row["seed"],
                "checkpoint": {**checkpoint, "model_state_sha256": row["checkpoint"]["model_state_sha256"]},
            }
        )
    track_snapshots = {}
    for name in ("action_planning_cem", "original_task_retention_cem"):
        track = tracks[name]
        snapshot = {
            "catalog": observed(track["catalog"], f"{name} catalog"),
            "oracle": observed(track["oracle"], f"{name} oracle"),
            "runner_core": observed(track["runner_core"], f"{name} runner core"),
        }
        for field in (
            "catalog_validator",
            "catalog_builder",
            "speed_cli",
            "speed_score",
            "frozen_baseline_wrapper",
        ):
            if field in track:
                snapshot[field] = observed(track[field], f"{name} {field}")
        if "paired_baseline" in track:
            baseline = track["paired_baseline"]
            snapshot["paired_baseline"] = {
                "results_freeze": observed(
                    baseline["results_freeze"], "retention baseline results freeze"
                ),
                "matrix_summary": observed(
                    baseline["matrix_summary"], "retention baseline matrix summary"
                ),
                "raw_receipts": [
                    {
                        "eval_seed": row["eval_seed"],
                        "receipt": observed(
                            row["receipt"],
                            f"retention baseline receipt {row['eval_seed']}",
                        ),
                    }
                    for row in baseline["raw_receipts"]
                ],
            }
        track_snapshots[name] = snapshot
    shared = tracks["shared"]
    return {
        "static_sources": {
            name: observed(value, f"static {name}")
            for name, value in static_sources.items()
        },
        "chain": {
            "completion": observed(chain["completion"], "completion"),
            "release": observed(chain["release"], "release"),
            "evaluation_binding_config": observed(
                chain["evaluation_binding_config"], "evaluation binding config"
            ),
            "evaluation_binding_receipt": observed(
                chain["evaluation_binding_receipt"], "evaluation binding receipt"
            ),
            "prepublic_cem_authority": observed_tree(
                chain["prepublic_cem_authority"], "pre-Public CEM authority"
            ),
            "recovery_preregistration": observed(
                chain["recovery_preregistration"], "recovery preregistration"
            ),
            "public_icl_aggregate": observed(
                chain["public_icl_aggregate"], "Public ICL aggregate"
            ),
            "behavioral_claim_boundary": observed(
                chain["behavioral_claim_boundary"], "behavioral claim boundary"
            ),
            "raw_public_icl": raw,
            "recovery_receipts": recovery,
            "checkpoints": checkpoints,
        },
        "shared": {
            "normalizer": observed(shared["normalizer"], "normalizer"),
            "pldm_config": observed(
                shared["stable_worldmodel"]["pldm_config"], "PLDM runtime config"
            ),
            "formal_runner": observed(shared["formal_runner"], "formal runner"),
            "aggregate_freezer": observed(shared["aggregate_freezer"], "aggregate freezer"),
            "binding_freezer": observed(shared["binding_freezer"], "binding freezer"),
            "adapter_boundary": observed(shared["adapter_boundary"], "adapter boundary"),
            "development_contract": observed(
                shared["development_contract"], "development contract"
            ),
            "paired_retention_comparator": observed(
                shared["paired_retention_comparator"], "paired retention comparator"
            ),
        },
        "tracks": track_snapshots,
    }


def build_binding(prereg_path: Path = CEM_PREREG) -> dict[str, Any]:
    prereg_path = resolve_source(prereg_path, repo_root=ROOT)
    static, static_sources = _validate_static_prereg(prereg_path)
    if DEFAULT_OUTPUT.exists() or any(path.exists() for path in CEM_ROOTS.values()):
        raise RuntimeError("CEM binding or CEM output namespace already exists")
    chain = _binding_chain(static, static_sources)
    runtime = _runtime(chain)
    tracks = _track_bindings(static, runtime, static_sources=static_sources)
    before = _snapshot(static_sources=static_sources, chain=chain, tracks=tracks)
    after = _snapshot(static_sources=static_sources, chain=chain, tracks=tracks)
    if before != after:
        raise RuntimeError("A positive-CEM input changed while its binding was frozen")
    return {
        "schema_version": 1,
        "cem_binding_id": CEM_BINDING_ID,
        "completion_id": COMPLETION_ID,
        "release_id": RELEASE_ID,
        "status": "frozen_after_passed_three_seed_public_icl_before_cem",
        "passed": True,
        "cem": {"authorized": True, "executed": False},
        "output": {
            "path": logical_path(DEFAULT_OUTPUT, repo_root=ROOT),
            "content_sha256_not_embedded_to_avoid_self_reference": True,
        },
        "preregistration": static_sources["preregistration"],
        "frozen_chain": chain,
        "tracks": tracks,
        "claim_boundary": chain["claim_boundary"],
        "scope": {
            "model_or_environment_execution_performed": False,
            "public_test_reopened": False,
            "checkpoint_selection_performed": False,
            "action_planning_cem_executed": False,
            "original_tworoom_retention_cem_executed": False,
        },
        "input_integrity": {
            "all_frozen_inputs_unchanged_during_binding": True,
            "identities_before_binding": before,
            "identities_after_binding": after,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", type=Path, default=CEM_PREREG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = _assert_output(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite CEM binding: {output}")
    payload = build_binding(args.preregistration)
    _write_exclusive(output, payload)
    print(
        json.dumps(
            {
                "cem_binding_id": payload["cem_binding_id"],
                "status": payload["status"],
                "output": payload["output"]["path"],
                "cem_executed": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
