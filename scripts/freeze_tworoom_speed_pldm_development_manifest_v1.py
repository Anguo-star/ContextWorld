#!/usr/bin/env python3
"""Freeze the non-Public Speed PLDM Development sample manifest.

The manifest is intentionally created only after all three fixed-step training
runs have completed and before any Development model inference.  It selects
the same four *index-defined* raw clips from each of the 96 synthetic
``speed_multi_v2`` validation scenarios.  It does not load a Public release,
run ICL scoring, or instantiate a model checkpoint.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from contextworld.benchmarks.speed_pldm_infrastructure_development import (
    COMPLETION_ID,
    DEVELOPMENT_ID,
    DEVELOPMENT_SCOPE,
    EXPECTED_ACTION_BLOCK_RAW_STEPS,
    EXPECTED_ACTION_DIM,
    EXPECTED_FUTURE_ACTION_BLOCKS,
    EXPECTED_HISTORY_TOKENS,
    EXPECTED_OBSERVATION_STEPS,
    EXPECTED_SEEDS,
    CLIP_INDEX_RULE,
    SAMPLES_PER_SCENARIO,
    deterministic_clip_indices,
    identity,
    logical_path,
    make_record_arrays,
    require_identity,
    resolve_local_output,
    resolve_source,
    root,
    sha256_file,
    write_json_exclusive,
)
from contextworld.synthesis.stablewm import load_stable_worldmodel


ROOT = root()
DEFAULT_CONFIG = (
    ROOT / "configs/benchmark/tworoom_speed_pldm_infrastructure_development_v1.yaml"
)
DEFAULT_EXECUTION_DISCLOSURE_AMENDMENT = (
    ROOT
    / "configs/benchmark/"
    "tworoom_speed_pldm_development_execution_disclosure_amendment_v1.yaml"
)
DEFAULT_OUTPUT = (
    ROOT
    / "artifacts/evaluation/history3/tworoom_speed_pldm_reference_completion_v1"
    / "infrastructure_development_v1/development_manifest.json"
)


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return value


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON mapping: {path}")
    return value


def _file_specification(path: Path) -> dict[str, Any]:
    value = identity(path, repo_root=ROOT)
    return {"path": value["path"], "sha256": value["sha256"]}


def _assert_exact(value: Any, expected: Any, *, label: str) -> None:
    if value != expected:
        raise ValueError(f"{label} differs from the preregistered contract")


def _require_output(config: dict[str, Any], output: Path) -> Path:
    outputs = config.get("outputs")
    if not isinstance(outputs, dict) or not isinstance(outputs.get("manifest"), str):
        raise ValueError("Development config lacks outputs.manifest")
    expected = resolve_local_output(outputs["manifest"], repo_root=ROOT)
    if expected != DEFAULT_OUTPUT.resolve():
        raise ValueError("Development manifest output is not its dedicated namespace")
    if output.resolve() != expected:
        raise ValueError(
            "Development manifest output must equal its preregistered destination"
        )
    return expected


def _same_path_sha(value: Any, expected: Any) -> bool:
    return bool(
        isinstance(value, dict)
        and isinstance(expected, dict)
        and value.get("path") == expected.get("path")
        and value.get("sha256") == expected.get("sha256")
    )


def _validate_post_interruption_amendment(
    *, config_path: Path, config: dict[str, Any]
) -> dict[str, Any]:
    """Validate the later disclosure amendment without changing the prereg.

    The original Development YAML remains byte-identical and is archived as a
    separate byte-for-byte snapshot.  This amendment is the only place that
    authorizes the active manifest freezer to add the recovery disclosure.
    """

    path = resolve_source(DEFAULT_EXECUTION_DISCLOSURE_AMENDMENT, repo_root=ROOT)
    amendment = _load_yaml(path)
    if not (
        amendment.get("schema_version") == 1
        and amendment.get("amendment_id")
        == "tworoom_speed_pldm_development_execution_disclosure_amendment_v1"
        and amendment.get("development_id") == DEVELOPMENT_ID
        and amendment.get("completion_id") == COMPLETION_ID
        and amendment.get("status")
        == "registered_after_interruption_recovery_before_development_manifest"
        and amendment.get("chronology")
        == {
            "base_development_preregistration_preserved": True,
            "registered_after_external_interruption": True,
            "registered_after_recovery_preparation": True,
            "not_represented_as_pretraining_preregistration": True,
            "development_public_or_cem_executed_at_registration": False,
        }
        and amendment.get("scope")
        == {
            "training_recipe_changed": False,
            "checkpoint_selection_changed": False,
            "development_or_cem_executed": False,
            "public_test_accessed": False,
            "adds_predevelopment_execution_disclosure_gate": True,
            "existing_preflight_receipts_used_only_as_preflight_records": True,
            "recovery_completion_receipts_are_authoritative_for_training_completion": True,
        }
    ):
        raise ValueError("Execution-disclosure amendment chronology is invalid")
    base = amendment.get("base_development_config")
    snapshot = amendment.get("pre_interruption_config_snapshot")
    if not isinstance(base, dict) or not isinstance(snapshot, dict):
        raise ValueError("Execution-disclosure amendment lacks base-config lineage")
    observed_base = identity(config_path, repo_root=ROOT)
    if base != observed_base:
        raise RuntimeError("Development config no longer matches the amended lineage base")
    snapshot_identity = require_identity(
        snapshot,
        label="pre_interruption_config_snapshot",
        repo_root=ROOT,
    )
    snapshot_path = resolve_source(snapshot["path"], repo_root=ROOT)
    if snapshot_path.read_bytes() != config_path.read_bytes():
        raise RuntimeError("Pre-interruption Development-config snapshot is not byte-identical")

    source_lineage = amendment.get("source_lineage")
    if not isinstance(source_lineage, dict) or set(source_lineage) != {
        "historical_manifest_freezer",
        "active_manifest_freezer",
        "historical_development_evaluator",
        "active_development_evaluator",
    }:
        raise ValueError("Execution-disclosure amendment source lineage is incomplete")
    historical = config.get("implementation", {}).get("manifest_freezer")
    if not _same_path_sha(historical, source_lineage["historical_manifest_freezer"]):
        raise RuntimeError("Execution-disclosure amendment does not preserve the historical freezer identity")
    active = require_identity(
        source_lineage["active_manifest_freezer"],
        label="active_manifest_freezer",
        repo_root=ROOT,
    )
    if active != identity(Path(__file__).resolve(), repo_root=ROOT):
        raise RuntimeError("Active Development manifest freezer is not registered by the amendment")
    historical_evaluator = config.get("implementation", {}).get("development_evaluator")
    if not _same_path_sha(
        historical_evaluator, source_lineage["historical_development_evaluator"]
    ):
        raise RuntimeError("Execution-disclosure amendment does not preserve the historical evaluator identity")
    active_evaluator = require_identity(
        source_lineage["active_development_evaluator"],
        label="active_development_evaluator",
        repo_root=ROOT,
    )

    disclosure = amendment.get("execution_disclosure")
    if not isinstance(disclosure, dict) or set(disclosure) != {
        "config",
        "output",
    }:
        raise ValueError("Execution-disclosure amendment lacks its disclosure contract")
    disclosure_config = require_identity(
        disclosure["config"],
        label="execution_disclosure.config",
        repo_root=ROOT,
    )
    disclosure_payload = _load_yaml(
        resolve_source(disclosure["config"]["path"], repo_root=ROOT)
    )
    expected_output = (
        "artifacts/evaluation/history3/tworoom_speed_pldm_reference_completion_v1/"
        "attempts/training_interruption_recovery_v1/execution_disclosure_v1.json"
    )
    if not (
        disclosure_payload.get("execution_disclosure_id")
        == "tworoom_speed_pldm_training_interruption_execution_disclosure_v1"
        and disclosure_payload.get("completion_id") == COMPLETION_ID
        and disclosure_payload.get("outputs", {}).get("disclosure") == expected_output
        and disclosure.get("output") == expected_output
    ):
        raise ValueError("Execution-disclosure amendment points to the wrong disclosure contract")
    if amendment.get("outputs") != {
        "development_manifest": "artifacts/evaluation/history3/tworoom_speed_pldm_reference_completion_v1/infrastructure_development_v1/development_manifest.json"
    }:
        raise ValueError("Execution-disclosure amendment changes the Development output namespace")
    return {
        "amendment": identity(path, repo_root=ROOT),
        "base_development_config": observed_base,
        "pre_interruption_config_snapshot": snapshot_identity,
        "historical_manifest_freezer": dict(source_lineage["historical_manifest_freezer"]),
        "active_manifest_freezer": active,
        "historical_development_evaluator": dict(
            source_lineage["historical_development_evaluator"]
        ),
        "active_development_evaluator": active_evaluator,
        "execution_disclosure_config": disclosure_config,
        "execution_disclosure_output": str(disclosure["output"]),
    }


def _validate_implementation(
    config: dict[str, Any], amendment: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    implementation = config.get("implementation")
    required = {
        "manifest_freezer",
        "development_evaluator",
        "shared_contract",
        "adapter_boundary",
    }
    if not isinstance(implementation, dict) or set(implementation) != required:
        raise ValueError("Development config implementation identities are incomplete")
    observed = {
        name: require_identity(specification, label=f"implementation.{name}", repo_root=ROOT)
        for name, specification in implementation.items()
        if name not in {"manifest_freezer", "development_evaluator"}
    }
    observed["manifest_freezer"] = amendment["active_manifest_freezer"]
    observed["development_evaluator"] = amendment["active_development_evaluator"]
    return observed


def _execution_disclosure_evidence(amendment: dict[str, Any]) -> dict[str, Any]:
    """Audit the completed recovery disclosure before Development data access."""

    disclosure_source = (
        ROOT / "scripts/freeze_tworoom_speed_pldm_execution_disclosure_v1.py"
    )
    specification = importlib.util.spec_from_file_location(
        "_tworoom_speed_pldm_execution_disclosure_gate",
        disclosure_source,
    )
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Cannot import execution-disclosure gate: {disclosure_source}")
    disclosure_gate = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(disclosure_gate)

    evidence = disclosure_gate.audit_disclosure(
        config_path=resolve_source(
            amendment["execution_disclosure_config"]["path"], repo_root=ROOT
        ),
        disclosure_path=amendment["execution_disclosure_output"],
    )
    if not (
        isinstance(evidence, dict)
        and isinstance(evidence.get("receipt"), dict)
        and isinstance(evidence.get("completion_receipts"), list)
        and tuple(int(row.get("seed", -1)) for row in evidence["completion_receipts"])
        == EXPECTED_SEEDS
        and isinstance(evidence.get("training_reports"), list)
        and tuple(int(row.get("seed", -1)) for row in evidence["training_reports"])
        == EXPECTED_SEEDS
    ):
        raise RuntimeError("Execution disclosure did not bind all completed recovery runs")
    return evidence


def _validate_config(config_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config_path = resolve_source(config_path, repo_root=ROOT)
    config = _load_yaml(config_path)
    if not (
        config.get("schema_version") == 1
        and config.get("development_id") == DEVELOPMENT_ID
        and config.get("completion_id") == COMPLETION_ID
        and config.get("status")
        == "preregistered_during_fixed_training_before_development_manifest_or_inference"
        and config.get("scope") == DEVELOPMENT_SCOPE
    ):
        raise ValueError("Unexpected Speed Development preregistration")

    sampling = config.get("sampling")
    if not isinstance(sampling, dict):
        raise ValueError("Development config lacks sampling contract")
    _assert_exact(
        sampling,
        {
            "catalog_split": "val.synthetic",
            "scenarios": 96,
            "samples_per_scenario": SAMPLES_PER_SCENARIO,
            "index_rule": CLIP_INDEX_RULE,
            "index_rule_inputs": "dataset_length_only_no_pixels_actions_labels_model_or_loss",
            "history_tokens": EXPECTED_HISTORY_TOKENS,
            "future_action_blocks": EXPECTED_FUTURE_ACTION_BLOCKS,
            "raw_steps_per_action_block": EXPECTED_ACTION_BLOCK_RAW_STEPS,
            "action_dim": EXPECTED_ACTION_DIM,
            "observation_steps_per_clip": EXPECTED_OBSERVATION_STEPS,
        },
        label="sampling contract",
    )
    expected_coverage = {
        "speed_values": [2.5, 3.2, 4.3, 4.6, 4.9, 5.4, 5.6, 5.8],
        "scenario_regime_counts": {
            "validation_broad_cross": 32,
            "validation_broad_same": 32,
            "validation_template_s0": 8,
            "validation_template_s1": 8,
            "validation_template_s2": 8,
            "validation_template_s3": 8,
        },
    }
    _assert_exact(config.get("expected_coverage"), expected_coverage, label="coverage contract")
    amendment = _validate_post_interruption_amendment(
        config_path=config_path,
        config=config,
    )
    implementation = _validate_implementation(config, amendment)

    frozen_inputs = config.get("frozen_inputs")
    required_inputs = {
        "completion_config",
        "behavioral_claim_boundary",
        "normalizer",
        "speed_catalog",
        "speed_manifest",
        "speed_synthesis_report",
    }
    if not isinstance(frozen_inputs, dict) or set(frozen_inputs) != required_inputs:
        raise ValueError("Development config frozen inputs are incomplete")
    inputs = {
        name: require_identity(specification, label=f"frozen_inputs.{name}", repo_root=ROOT)
        for name, specification in frozen_inputs.items()
    }
    completion = _load_yaml(resolve_source(frozen_inputs["completion_config"]["path"], repo_root=ROOT))
    if not (
        completion.get("completion_id") == COMPLETION_ID
        and completion.get("training", {}).get("seeds") == list(EXPECTED_SEEDS)
        and int(completion.get("training", {}).get("optimizer_steps", -1)) == 12840
        and completion.get("training", {}).get("model_id")
        == "H3_Speed_PLDM_ReferenceCompletion"
        and completion.get("training", {}).get("checkpoint_selection") == "final_fixed_step"
        and completion.get("training", {}).get("early_stopping") is False
    ):
        raise ValueError("Completion config does not match the fixed Development contract")

    boundary = _load_yaml(resolve_source(frozen_inputs["behavioral_claim_boundary"]["path"], repo_root=ROOT))
    if not (
        boundary.get("completion_id") == COMPLETION_ID
        and boundary.get("chronology", {}).get("development_evaluation_started") is False
        and boundary.get("chronology", {}).get("public_test_opened") is False
        and boundary.get("conditional_evaluation", {}).get("development_must_precede_public") is True
        and boundary.get("mutation_boundary", {}).get("public_test_access_authorized") is False
    ):
        raise ValueError("Behavioral claim boundary does not permit this pre-Public stage")

    runtime = config.get("stable_worldmodel")
    if not isinstance(runtime, dict) or set(runtime) != {
        "worktree",
        "expected_ref",
        "pldm_config",
        "pldm_config_sha256",
    }:
        raise ValueError("Development config stable_worldmodel section is invalid")
    worktree = Path(runtime["worktree"]).resolve()
    pldm_config = worktree / str(runtime["pldm_config"])
    if not pldm_config.is_file() or sha256_file(pldm_config) != runtime["pldm_config_sha256"]:
        raise RuntimeError("Pinned Stable-WorldModel PLDM config drifted")
    return config, {
        "implementation": implementation,
        "frozen_inputs": inputs,
        "execution_disclosure_amendment": amendment,
    }


def _training_entries(
    config: dict[str, Any], execution_disclosure: dict[str, Any]
) -> list[dict[str, Any]]:
    training = config.get("training_artifacts")
    if not isinstance(training, dict) or set(training) != {
        "fixed_optimizer_steps",
        "model_id",
        "entries",
    }:
        raise ValueError("Development training-artifact declaration is invalid")
    if int(training["fixed_optimizer_steps"]) != 12840 or training["model_id"] != "H3_Speed_PLDM_ReferenceCompletion":
        raise ValueError("Development training-artifact budget/model is invalid")
    rows = training["entries"]
    if not isinstance(rows, list) or len(rows) != len(EXPECTED_SEEDS):
        raise ValueError("Development requires exactly three training artifacts")
    if tuple(sorted(int(row.get("seed", -1)) for row in rows)) != EXPECTED_SEEDS:
        raise ValueError("Development training artifacts use the wrong seeds")

    completion_rows = execution_disclosure.get("completion_receipts")
    report_rows = execution_disclosure.get("training_reports")
    if not (
        isinstance(completion_rows, list)
        and isinstance(report_rows, list)
        and tuple(int(row.get("seed", -1)) for row in completion_rows)
        == EXPECTED_SEEDS
        and tuple(int(row.get("seed", -1)) for row in report_rows)
        == EXPECTED_SEEDS
    ):
        raise RuntimeError("Execution disclosure lacks three completed recovery runs")
    completion_by_seed = {
        int(row["seed"]): row["receipt"] for row in completion_rows
    }
    report_by_seed = {int(row["seed"]): row["report"] for row in report_rows}

    output = []
    for row in rows:
        seed = int(row["seed"])
        expected_run_name = f"speed_pldm_reference_completion_v1_s{seed}"
        if row.get("run_name") != expected_run_name:
            raise ValueError(f"Unexpected run name for seed {seed}")
        required_paths = {
            "checkpoint",
            "checkpoint_config",
            "training_report",
            "loss_trace",
            "preflight",
        }
        if set(row) != {"seed", "run_name", *required_paths}:
            raise ValueError(f"Training entry fields are invalid for seed {seed}")
        paths = {name: resolve_source(row[name], repo_root=ROOT) for name in required_paths}
        if any(not path.is_file() for path in paths.values()):
            missing = [name for name, path in paths.items() if not path.is_file()]
            raise FileNotFoundError(f"Missing final training artifacts for seed {seed}: {missing}")
        report = _load_json(paths["training_report"])
        preflight = _load_json(paths["preflight"])
        trace = [
            json.loads(line)
            for line in paths["loss_trace"].read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        artifacts = report.get("artifacts", {})
        training_state = report.get("training", {})
        report_contract = (
            report.get("schema_version") == 1
            and report.get("passed") is True
            and report.get("run_kind") == "confirmation"
            and report.get("profile") == "additive"
            and report.get("model_id") == training["model_id"]
            and report.get("run_name") == expected_run_name
            and report.get("model", {}).get("training_method") == "pldm"
            and report.get("model", {}).get("history_size") == EXPECTED_HISTORY_TOKENS
            and report.get("model", {}).get("action_block") == EXPECTED_ACTION_BLOCK_RAW_STEPS
            and training_state.get("training_complete") is True
            and int(training_state.get("global_step", -1)) == int(training["fixed_optimizer_steps"])
            and int(training_state.get("expected_optimizer_steps", -1)) == int(training["fixed_optimizer_steps"])
            and artifacts.get("pretrained") == str(paths["checkpoint"])
            and artifacts.get("pretrained_sha256") == sha256_file(paths["checkpoint"])
            and artifacts.get("pretrained_config") == str(paths["checkpoint_config"])
            and artifacts.get("pretrained_config_sha256") == sha256_file(paths["checkpoint_config"])
            and artifacts.get("loss_trace", {}).get("sha256") == sha256_file(paths["loss_trace"])
            and int(artifacts.get("loss_trace", {}).get("last_optimizer_step", -1))
            == int(training["fixed_optimizer_steps"])
            and report.get("save_load_exact") is True
        )
        # These existing files are accepted only as preflight records.  No
        # historical byte-immutability claim is made here; completion is
        # proven by the separately frozen recovery receipt and final report.
        preflight_contract = (
            preflight.get("completion_id") == COMPLETION_ID
            and preflight.get("status") == "passed"
            and int(preflight.get("seed", -1)) == seed
            and preflight.get("training_started") is False
            and "training_completed" not in preflight
            and "training_failed" not in preflight
            and report_by_seed.get(seed)
            == identity(paths["training_report"], repo_root=ROOT)
            and isinstance(completion_by_seed.get(seed), dict)
        )
        trace_contract = bool(
            trace
            and [int(item["optimizer_step"]) for item in trace]
            == sorted({int(item["optimizer_step"]) for item in trace})
            and int(trace[-1]["optimizer_step"]) == int(training["fixed_optimizer_steps"])
        )
        if not (report_contract and preflight_contract and trace_contract):
            raise RuntimeError(
                f"Fixed-step training completion contract failed for seed {seed}: "
                f"report={report_contract}, preflight={preflight_contract}, trace={trace_contract}"
            )
        output.append(
            {
                "seed": seed,
                "run_name": expected_run_name,
                "checkpoint": identity(paths["checkpoint"], repo_root=ROOT),
                "checkpoint_config": identity(paths["checkpoint_config"], repo_root=ROOT),
                "training_report": identity(paths["training_report"], repo_root=ROOT),
                "loss_trace": identity(paths["loss_trace"], repo_root=ROOT),
                "preflight": identity(paths["preflight"], repo_root=ROOT),
                "recovery_completion_receipt": completion_by_seed[seed],
                "fixed_training_contract": {
                    "passed": True,
                    "checkpoint_selection": "final_fixed_step",
                    "early_stopping": False,
                    "optimizer_steps": int(training["fixed_optimizer_steps"]),
                    "completion_evidence": (
                        "recovery_completion_receipt_and_final_training_report"
                    ),
                    "preflight_receipt_role": (
                        "preflight_only_not_training_completion_evidence"
                    ),
                    "historical_preflight_immutability_claimed": False,
                },
            }
        )
    return sorted(output, key=lambda item: item["seed"])


def _scenario_report_map(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = report.get("scenarios")
    if not isinstance(rows, list):
        raise ValueError("Synthesis report has no scenario list")
    result = {
        str(row.get("scenario_id")): row
        for row in rows
        if isinstance(row, dict) and isinstance(row.get("scenario_id"), str)
    }
    if len(result) != len(rows):
        raise ValueError("Synthesis report scenario IDs are incomplete or duplicate")
    return result


def _catalog_validation_paths(config: dict[str, Any]) -> tuple[list[str], dict[str, Any], dict[str, Any]]:
    frozen = config["frozen_inputs"]
    catalog = _load_json(resolve_source(frozen["speed_catalog"]["path"], repo_root=ROOT))
    report = _load_json(resolve_source(frozen["speed_synthesis_report"]["path"], repo_root=ROOT))
    val = catalog.get("val")
    if not isinstance(val, dict) or set(val) != {"synthetic"}:
        raise ValueError("Development loader must use only catalog.val.synthetic")
    paths = val["synthetic"]
    if not isinstance(paths, list) or len(paths) != 96 or any(not isinstance(path, str) for path in paths):
        raise ValueError("Development validation catalog must contain exactly 96 paths")
    if len(set(paths)) != len(paths):
        raise ValueError("Development validation catalog paths are duplicated")
    forbidden = tuple(config.get("forbidden_public_path_fragments", ()))
    if not forbidden or any(any(fragment in path for fragment in forbidden) for path in paths):
        raise ValueError("Development catalog is missing its Public-exclusion contract")
    return paths, catalog, report


def _build_records(config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    paths, _catalog, report = _catalog_validation_paths(config)
    scenario_map = _scenario_report_map(report)
    runtime = config["stable_worldmodel"]
    swm, stable_repo, stable_commit = load_stable_worldmodel(
        ROOT,
        str(runtime["worktree"]),
        str(runtime["expected_ref"]),
    )
    if stable_repo.resolve() != Path(runtime["worktree"]).resolve() or stable_commit != runtime["expected_ref"]:
        raise RuntimeError("Development data loader did not use the pinned StableWM worktree")

    records: list[dict[str, Any]] = []
    speed_counts: Counter[float] = Counter()
    regime_counts: Counter[str] = Counter()
    for scenario_index, raw_path in enumerate(paths):
        source = resolve_source(raw_path, repo_root=ROOT)
        if not source.is_dir() or source.suffix != ".lance":
            raise FileNotFoundError(f"Development scenario is not a Lance directory: {source}")
        scenario_id = source.stem
        report_row = scenario_map.get(scenario_id)
        if not isinstance(report_row, dict) or report_row.get("split") != "val":
            raise ValueError(f"Development scenario is absent from the held-out split: {scenario_id}")
        expected_speed = report_row.get("factor_checks", {}).get("agent.speed", {}).get("expected")
        if not isinstance(expected_speed, list) or len(expected_speed) != 1:
            raise ValueError(f"Development scenario has no unambiguous speed: {scenario_id}")
        speed = float(expected_speed[0])
        regime = source.parent.name
        dataset = swm.data.LanceDataset(
            path=source,
            frameskip=EXPECTED_ACTION_BLOCK_RAW_STEPS,
            num_steps=EXPECTED_OBSERVATION_STEPS,
            keys_to_load=["pixels", "action"],
            transform=None,
        )
        clip_indices = deterministic_clip_indices(len(dataset))
        if len(set(clip_indices)) != SAMPLES_PER_SCENARIO:
            raise RuntimeError(f"Development clip indices are not unique: {scenario_id}")
        for clip_index in clip_indices:
            episode_index, start_raw_step = dataset.clip_indices[clip_index]
            episode_length = int(dataset.lengths[episode_index])
            span = EXPECTED_OBSERVATION_STEPS * EXPECTED_ACTION_BLOCK_RAW_STEPS
            if int(start_raw_step) < 0 or int(start_raw_step) + span > episode_length:
                raise RuntimeError(f"Non-contiguous Development clip: {scenario_id}:{clip_index}")
            arrays = make_record_arrays(dataset[clip_index])
            records.append(
                {
                    "scenario_index": scenario_index,
                    "scenario_id": scenario_id,
                    "scenario_path": raw_path,
                    "scenario_regime": regime,
                    "speed": speed,
                    "dataset_clip_count": int(len(dataset)),
                    "clip_index": int(clip_index),
                    "episode_index": int(episode_index),
                    "start_raw_step": int(start_raw_step),
                    "episode_length_raw_steps": episode_length,
                    "span_raw_steps": span,
                    "continuous_raw_span": True,
                    **arrays,
                }
            )
        speed_counts[speed] += 1
        regime_counts[regime] += 1
    expected_coverage = config["expected_coverage"]
    observed_speeds = sorted(speed_counts)
    if observed_speeds != list(expected_coverage["speed_values"]):
        raise ValueError(f"Development speed support drifted: {observed_speeds}")
    if dict(sorted(regime_counts.items())) != expected_coverage["scenario_regime_counts"]:
        raise ValueError(f"Development regime coverage drifted: {dict(regime_counts)}")
    if len(records) != 96 * SAMPLES_PER_SCENARIO:
        raise RuntimeError("Development manifest did not cover every fixed sample")
    per_scenario = {
        row["scenario_id"]: []
        for row in records
    }
    for row in records:
        per_scenario[row["scenario_id"]].append(row)
    distinct_episode_counts = [
        len({int(row["episode_index"]) for row in rows})
        for rows in per_scenario.values()
    ]
    raw_start_spans = [
        max(int(row["start_raw_step"]) for row in rows)
        - min(int(row["start_raw_step"]) for row in rows)
        for rows in per_scenario.values()
    ]
    return records, {
        "validation_scenarios": len(paths),
        "samples_per_scenario": SAMPLES_PER_SCENARIO,
        "total_samples": len(records),
        "speed_values": observed_speeds,
        "scenario_regime_counts": dict(sorted(regime_counts.items())),
        "index_rule": CLIP_INDEX_RULE,
        "all_actual_indices_unique_per_scenario": all(
            len({int(row["clip_index"]) for row in rows}) == SAMPLES_PER_SCENARIO
            for rows in per_scenario.values()
        ),
        "all_source_spans_continuous": all(row["continuous_raw_span"] for row in records),
        "episode_and_time_coverage": {
            "total_distinct_episode_references": int(sum(distinct_episode_counts)),
            "minimum_distinct_episodes_per_scenario": int(min(distinct_episode_counts)),
            "maximum_distinct_episodes_per_scenario": int(max(distinct_episode_counts)),
            "scenarios_with_multiple_selected_episodes": int(sum(value > 1 for value in distinct_episode_counts)),
            "minimum_selected_raw_start_span": int(min(raw_start_spans)),
            "maximum_selected_raw_start_span": int(max(raw_start_spans)),
        },
    }


def build_manifest(config_path: Path) -> dict[str, Any]:
    config_path = resolve_source(config_path, repo_root=ROOT)
    config, identities = _validate_config(config_path)
    # This audit touches only the already-completed recovery evidence.  It
    # precedes StableWM loading and therefore keeps Development data access
    # closed until the execution disclosure has passed.
    execution_disclosure = _execution_disclosure_evidence(
        identities["execution_disclosure_amendment"]
    )
    training = _training_entries(config, execution_disclosure)
    records, coverage = _build_records(config)
    formal_roots = [
        ROOT / "artifacts/evaluation/history3/tworoom_speed_pldm_reference_completion_v1/formal_icl_v1",
        ROOT / "artifacts/evaluation/history3/tworoom_speed_pldm_reference_completion_v1/formal_action_planning_cem_v1",
        ROOT / "artifacts/evaluation/history3/tworoom_speed_pldm_reference_completion_v1/formal_original_tworoom_retention_cem_v1",
    ]
    if any(path.exists() for path in formal_roots):
        raise RuntimeError("Public or CEM artifacts exist before Development manifest freeze")
    return {
        "schema_version": 1,
        "development_id": DEVELOPMENT_ID,
        "completion_id": COMPLETION_ID,
        "status": "frozen_prepublic_development_manifest",
        "passed": True,
        "scope": DEVELOPMENT_SCOPE,
        "development_config": identity(config_path, repo_root=ROOT),
        "implementation": identities["implementation"],
        "frozen_inputs": identities["frozen_inputs"],
        "post_interruption_execution_disclosure_amendment": identities[
            "execution_disclosure_amendment"
        ],
        "execution_disclosure": execution_disclosure,
        "training_checkpoints": training,
        "sampling": config["sampling"],
        "coverage": coverage,
        "records": records,
        "public_payload_accessed": False,
        "formal_public_or_cem_artifacts_present": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    config_path = resolve_source(args.config, repo_root=ROOT)
    config = _load_yaml(config_path)
    output = _require_output(config, resolve_local_output(args.output, repo_root=ROOT))
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite Development manifest: {output}")
    manifest = build_manifest(config_path)
    write_json_exclusive(output, manifest)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "output": logical_path(output, repo_root=ROOT),
                "total_samples": manifest["coverage"]["total_samples"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
