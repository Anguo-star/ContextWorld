#!/usr/bin/env python3
"""Run one non-Public infrastructure-readiness check for a Speed PLDM seed.

This is not an ICL evaluation and intentionally emits no capability score.
It reconstructs the frozen 384 raw held-out samples, uses the native model
rollout path for one future latent block, and accepts only infrastructure
facts: complete coverage, finite native latent MSE, correct prefix geometry,
and unchanged frozen weights.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from contextworld.benchmarks.adapters import StableWorldModelPLDMAdapter
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
    canonical_array_identity,
    deterministic_clip_indices,
    identity,
    load_json,
    logical_path,
    require_identity,
    resolve_local_output,
    resolve_source,
    root,
    same_identity,
    verify_record_arrays,
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
DEFAULT_MANIFEST = (
    ROOT
    / "artifacts/evaluation/history3/tworoom_speed_pldm_reference_completion_v1"
    / "infrastructure_development_v1/development_manifest.json"
)
DEFAULT_OUTPUT_ROOT = DEFAULT_MANIFEST.parent


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return value


def _scope_contract(value: Any) -> bool:
    return value == DEVELOPMENT_SCOPE


def _same_path_sha(value: Any, expected: Any) -> bool:
    return bool(
        isinstance(value, dict)
        and isinstance(expected, dict)
        and value.get("path") == expected.get("path")
        and value.get("sha256") == expected.get("sha256")
    )


def _execution_disclosure_amendment(
    *, config_path: Path, config: dict[str, Any]
) -> dict[str, Any]:
    """Load the explicit post-interruption lineage for the active sources."""

    amendment_path = resolve_source(
        DEFAULT_EXECUTION_DISCLOSURE_AMENDMENT, repo_root=ROOT
    )
    amendment = _load_yaml(amendment_path)
    if not (
        amendment.get("schema_version") == 1
        and amendment.get("amendment_id")
        == "tworoom_speed_pldm_development_execution_disclosure_amendment_v1"
        and amendment.get("development_id") == DEVELOPMENT_ID
        and amendment.get("completion_id") == COMPLETION_ID
        and amendment.get("status")
        == "registered_after_interruption_recovery_before_development_manifest"
        and amendment.get("chronology", {}).get(
            "base_development_preregistration_preserved"
        )
        is True
        and amendment.get("chronology", {}).get(
            "not_represented_as_pretraining_preregistration"
        )
        is True
        and amendment.get("scope", {}).get(
            "adds_predevelopment_execution_disclosure_gate"
        )
        is True
    ):
        raise RuntimeError("Development execution-disclosure amendment is invalid")
    config_identity = identity(config_path, repo_root=ROOT)
    snapshot = amendment.get("pre_interruption_config_snapshot")
    if amendment.get("base_development_config") != config_identity or not isinstance(snapshot, dict):
        raise RuntimeError("Development execution-disclosure amendment base lineage drifted")
    snapshot_identity = require_identity(
        snapshot,
        label="pre_interruption_config_snapshot",
        repo_root=ROOT,
    )
    snapshot_path = resolve_source(snapshot["path"], repo_root=ROOT)
    if snapshot_path.read_bytes() != config_path.read_bytes():
        raise RuntimeError("Development pre-interruption config snapshot is not byte-identical")
    lineage = amendment.get("source_lineage")
    if not isinstance(lineage, dict) or set(lineage) != {
        "historical_manifest_freezer",
        "active_manifest_freezer",
        "historical_development_evaluator",
        "active_development_evaluator",
    }:
        raise RuntimeError("Development execution-disclosure source lineage is incomplete")
    historical = config.get("implementation", {})
    if not (
        _same_path_sha(historical.get("manifest_freezer"), lineage["historical_manifest_freezer"])
        and _same_path_sha(
            historical.get("development_evaluator"),
            lineage["historical_development_evaluator"],
        )
    ):
        raise RuntimeError("Development execution-disclosure amendment does not preserve historical sources")
    active_manifest = require_identity(
        lineage["active_manifest_freezer"],
        label="active_manifest_freezer",
        repo_root=ROOT,
    )
    active_evaluator = require_identity(
        lineage["active_development_evaluator"],
        label="active_development_evaluator",
        repo_root=ROOT,
    )
    if active_evaluator != identity(Path(__file__).resolve(), repo_root=ROOT):
        raise RuntimeError("Development evaluator source differs from its registered amendment identity")
    disclosure = amendment.get("execution_disclosure")
    if not isinstance(disclosure, dict) or set(disclosure) != {"config", "output"}:
        raise RuntimeError("Development execution-disclosure contract is incomplete")
    disclosure_config = require_identity(
        disclosure["config"], label="execution_disclosure.config", repo_root=ROOT
    )
    if disclosure.get("output") != (
        "artifacts/evaluation/history3/tworoom_speed_pldm_reference_completion_v1/"
        "attempts/training_interruption_recovery_v1/execution_disclosure_v1.json"
    ):
        raise RuntimeError("Development execution-disclosure output drifted")
    return {
        "amendment": identity(amendment_path, repo_root=ROOT),
        "base_development_config": config_identity,
        "pre_interruption_config_snapshot": snapshot_identity,
        "historical_manifest_freezer": dict(lineage["historical_manifest_freezer"]),
        "active_manifest_freezer": active_manifest,
        "historical_development_evaluator": dict(
            lineage["historical_development_evaluator"]
        ),
        "active_development_evaluator": active_evaluator,
        "execution_disclosure_config": disclosure_config,
        "execution_disclosure_output": str(disclosure["output"]),
    }


def _config(
    config_path: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, Any]]:
    config = _load_yaml(config_path)
    if not (
        config.get("schema_version") == 1
        and config.get("development_id") == DEVELOPMENT_ID
        and config.get("completion_id") == COMPLETION_ID
        and config.get("status")
        == "preregistered_during_fixed_training_before_development_manifest_or_inference"
        and _scope_contract(config.get("scope"))
    ):
        raise ValueError("Unexpected Speed Development preregistration")
    implementation = config.get("implementation")
    required = {
        "manifest_freezer",
        "development_evaluator",
        "shared_contract",
        "adapter_boundary",
    }
    if not isinstance(implementation, dict) or set(implementation) != required:
        raise ValueError("Development implementation identities are incomplete")
    amendment = _execution_disclosure_amendment(
        config_path=config_path,
        config=config,
    )
    observed = {
        name: require_identity(specification, label=f"implementation.{name}", repo_root=ROOT)
        for name, specification in implementation.items()
        if name not in {"manifest_freezer", "development_evaluator"}
    }
    observed["manifest_freezer"] = amendment["active_manifest_freezer"]
    observed["development_evaluator"] = amendment["active_development_evaluator"]
    frozen = config.get("frozen_inputs")
    if not isinstance(frozen, dict):
        raise ValueError("Development config lacks frozen inputs")
    required_inputs = {
        "completion_config",
        "behavioral_claim_boundary",
        "normalizer",
        "speed_catalog",
        "speed_manifest",
        "speed_synthesis_report",
    }
    if set(frozen) != required_inputs:
        raise ValueError("Development config frozen inputs are incomplete")
    for name, specification in frozen.items():
        require_identity(specification, label=f"frozen_inputs.{name}", repo_root=ROOT)
    return config, observed, amendment


def _expected_output(config: dict[str, Any], seed: int, output: Path) -> Path:
    if seed not in EXPECTED_SEEDS:
        raise ValueError(f"Development seed is not registered: {seed}")
    outputs = config.get("outputs")
    receipts = outputs.get("receipts") if isinstance(outputs, dict) else None
    if not isinstance(receipts, dict) or set(receipts) != {str(item) for item in EXPECTED_SEEDS}:
        raise ValueError("Development outputs lack the complete three-seed receipt map")
    expected = resolve_local_output(receipts[str(seed)], repo_root=ROOT)
    if expected.parent != DEFAULT_OUTPUT_ROOT.resolve() or expected.name != f"seed_{seed}.json":
        raise ValueError("Development receipt output is not in its dedicated namespace")
    if output.resolve() != expected:
        raise ValueError("Development receipt must equal its preregistered destination")
    return expected


def _checkpoint_entry(manifest: dict[str, Any], seed: int) -> dict[str, Any]:
    rows = manifest.get("training_checkpoints")
    if not isinstance(rows, list) or len(rows) != len(EXPECTED_SEEDS):
        raise ValueError("Development manifest lacks three checkpoint entries")
    matches = [row for row in rows if isinstance(row, dict) and int(row.get("seed", -1)) == seed]
    if len(matches) != 1:
        raise ValueError(f"Development manifest does not uniquely bind seed {seed}")
    return matches[0]


def _validate_manifest(
    config: dict[str, Any],
    config_path: Path,
    manifest_path: Path,
    implementation: dict[str, dict[str, Any]],
    amendment: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    outputs = config.get("outputs")
    if not isinstance(outputs, dict) or not isinstance(outputs.get("manifest"), str):
        raise ValueError("Development config lacks outputs.manifest")
    expected_manifest = resolve_local_output(outputs["manifest"], repo_root=ROOT)
    if manifest_path.resolve() != expected_manifest or expected_manifest != DEFAULT_MANIFEST.resolve():
        raise ValueError("Development manifest is not its preregistered path")
    # Revalidate the immutable recovery disclosure before opening even the
    # Development manifest.  The manifest contains sample records and must not
    # become a side channel around the post-interruption pre-evaluation gate.
    from scripts.freeze_tworoom_speed_pldm_execution_disclosure_v1 import (
        audit_disclosure,
    )

    execution_disclosure = audit_disclosure(
        config_path=resolve_source(
            amendment["execution_disclosure_config"]["path"], repo_root=ROOT
        ),
        disclosure_path=amendment["execution_disclosure_output"],
    )
    manifest = load_json(manifest_path)
    if not (
        manifest.get("schema_version") == 1
        and manifest.get("development_id") == DEVELOPMENT_ID
        and manifest.get("completion_id") == COMPLETION_ID
        and manifest.get("status") == "frozen_prepublic_development_manifest"
        and manifest.get("passed") is True
        and _scope_contract(manifest.get("scope"))
        and manifest.get("public_payload_accessed") is False
        and manifest.get("formal_public_or_cem_artifacts_present") is False
        and same_identity(manifest.get("development_config"), identity(config_path, repo_root=ROOT))
        and manifest.get("implementation") == implementation
        and manifest.get("post_interruption_execution_disclosure_amendment") == amendment
    ):
        raise RuntimeError("Development manifest identity or scope is invalid")
    if manifest.get("execution_disclosure") != execution_disclosure:
        raise RuntimeError("Development manifest execution disclosure drifted")
    sampling = manifest.get("sampling")
    if not isinstance(sampling, dict) or sampling != config.get("sampling"):
        raise RuntimeError("Development manifest sampling contract drifted")
    coverage = manifest.get("coverage")
    if not (
        isinstance(coverage, dict)
        and coverage.get("validation_scenarios") == 96
        and coverage.get("samples_per_scenario") == SAMPLES_PER_SCENARIO
        and coverage.get("total_samples") == 96 * SAMPLES_PER_SCENARIO
        and coverage.get("index_rule") == CLIP_INDEX_RULE
        and coverage.get("all_actual_indices_unique_per_scenario") is True
        and coverage.get("all_source_spans_continuous") is True
        and coverage.get("speed_values") == config.get("expected_coverage", {}).get("speed_values")
        and coverage.get("scenario_regime_counts")
        == config.get("expected_coverage", {}).get("scenario_regime_counts")
    ):
        raise RuntimeError("Development manifest coverage contract drifted")
    records = manifest.get("records")
    if not isinstance(records, list) or len(records) != 96 * SAMPLES_PER_SCENARIO:
        raise RuntimeError("Development manifest does not have exactly 384 samples")
    by_scenario: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise RuntimeError("Development manifest contains a non-object record")
        scenario_id = record.get("scenario_id")
        if not isinstance(scenario_id, str):
            raise RuntimeError("Development manifest record lacks scenario_id")
        by_scenario.setdefault(scenario_id, []).append(record)
    if len(by_scenario) != 96:
        raise RuntimeError("Development manifest does not cover exactly 96 scenarios")
    for scenario_id, rows in by_scenario.items():
        if len({int(row.get("clip_index", -1)) for row in rows}) != SAMPLES_PER_SCENARIO:
            raise RuntimeError(f"Development manifest has duplicate clip indices: {scenario_id}")
        if not all(row.get("continuous_raw_span") is True for row in rows):
            raise RuntimeError(f"Development manifest has a non-contiguous source span: {scenario_id}")
    return manifest, execution_disclosure


def _load_record_arrays(manifest: dict[str, Any], config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    runtime = config["stable_worldmodel"]
    swm, stable_repo, stable_commit = load_stable_worldmodel(
        ROOT,
        str(runtime["worktree"]),
        str(runtime["expected_ref"]),
    )
    if stable_repo.resolve() != Path(runtime["worktree"]).resolve() or stable_commit != runtime["expected_ref"]:
        raise RuntimeError("Development evaluator did not load the pinned StableWM worktree")
    datasets: dict[str, Any] = {}
    histories = []
    action_prefixes = []
    targets = []
    observed_scenarios: Counter[str] = Counter()
    for record in manifest["records"]:
        raw_path = record.get("scenario_path")
        if not isinstance(raw_path, str):
            raise RuntimeError("Development manifest record lacks scenario_path")
        source = resolve_source(raw_path, repo_root=ROOT)
        key = str(source)
        dataset = datasets.get(key)
        if dataset is None:
            dataset = swm.data.LanceDataset(
                path=source,
                frameskip=EXPECTED_ACTION_BLOCK_RAW_STEPS,
                num_steps=EXPECTED_OBSERVATION_STEPS,
                keys_to_load=["pixels", "action"],
                transform=None,
            )
            datasets[key] = dataset
        clip_index = int(record.get("clip_index", -1))
        if not 0 <= clip_index < len(dataset):
            raise RuntimeError("Development manifest clip index is not available")
        if int(record.get("dataset_clip_count", -1)) != len(dataset):
            raise RuntimeError("Development manifest dataset length drifted")
        if clip_index not in deterministic_clip_indices(len(dataset)):
            raise RuntimeError("Development manifest clip index differs from length-only rule")
        episode_index, start_raw_step = dataset.clip_indices[clip_index]
        if not (
            int(record.get("episode_index", -1)) == int(episode_index)
            and int(record.get("start_raw_step", -1)) == int(start_raw_step)
            and int(record.get("episode_length_raw_steps", -1)) == int(dataset.lengths[episode_index])
            and int(record.get("span_raw_steps", -1))
            == EXPECTED_OBSERVATION_STEPS * EXPECTED_ACTION_BLOCK_RAW_STEPS
        ):
            raise RuntimeError("Development manifest source-continuity identity drifted")
        history, actions, target = verify_record_arrays(dataset[clip_index], record)
        histories.append(history)
        action_prefixes.append(actions)
        targets.append(target)
        observed_scenarios[str(record["scenario_id"])] += 1
    if len(observed_scenarios) != 96 or set(observed_scenarios.values()) != {SAMPLES_PER_SCENARIO}:
        raise RuntimeError("Development evaluator did not reconstruct every manifest sample")
    return (
        np.stack(histories),
        np.stack(action_prefixes),
        np.stack(targets),
        {
            "scenarios": len(observed_scenarios),
            "samples": len(histories),
            "source_lance_datasets": len(datasets),
            "all_manifest_array_hashes_verified": True,
        },
    )


def evaluate(
    *,
    config_path: Path,
    manifest_path: Path,
    seed: int,
    device: str,
    output: Path,
) -> dict[str, Any]:
    config_path = resolve_source(config_path, repo_root=ROOT)
    config, implementation, amendment = _config(config_path)
    output = _expected_output(config, seed, resolve_local_output(output, repo_root=ROOT))
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite Development receipt: {output}")
    manifest_path = resolve_local_output(manifest_path, repo_root=ROOT)
    manifest, execution_disclosure = _validate_manifest(
        config,
        config_path,
        manifest_path,
        implementation,
        amendment,
    )
    checkpoint_entry = _checkpoint_entry(manifest, seed)
    checkpoint = resolve_source(checkpoint_entry["checkpoint"]["path"], repo_root=ROOT)
    checkpoint_identity = identity(checkpoint, repo_root=ROOT)
    if checkpoint_identity != checkpoint_entry["checkpoint"]:
        raise RuntimeError("Development checkpoint bytes differ from frozen manifest")
    public_paths = [
        ROOT / "artifacts/evaluation/history3/tworoom_speed_pldm_reference_completion_v1/formal_icl_v1",
        ROOT / "artifacts/evaluation/history3/tworoom_speed_pldm_reference_completion_v1/formal_action_planning_cem_v1",
        ROOT / "artifacts/evaluation/history3/tworoom_speed_pldm_reference_completion_v1/formal_original_tworoom_retention_cem_v1",
    ]
    if any(path.exists() for path in public_paths):
        raise RuntimeError("Public/CEM artifacts exist before Development inference")

    histories, action_prefixes, targets, sample_audit = _load_record_arrays(manifest, config)
    if not (
        histories.shape[1] == EXPECTED_HISTORY_TOKENS
        and action_prefixes.shape[1:]
        == (
            EXPECTED_HISTORY_TOKENS,
            EXPECTED_ACTION_BLOCK_RAW_STEPS,
            EXPECTED_ACTION_DIM,
        )
        and len(targets) == len(histories) == 96 * SAMPLES_PER_SCENARIO
    ):
        raise RuntimeError("Development rollout prefix geometry is invalid before model load")
    runtime = config["stable_worldmodel"]
    normalizer = resolve_source(config["frozen_inputs"]["normalizer"]["path"], repo_root=ROOT)
    adapter = StableWorldModelPLDMAdapter.from_checkpoint(
        checkpoint,
        normalizer=normalizer,
        repo_root=ROOT,
        stablewm_repo=str(runtime["worktree"]),
        stablewm_ref=str(runtime["expected_ref"]),
        device=device,
    )
    protocol = adapter.protocol
    protocol_ok = (
        int(protocol.history_tokens) == EXPECTED_HISTORY_TOKENS
        and int(protocol.action_block_raw_steps) == EXPECTED_ACTION_BLOCK_RAW_STEPS
        and int(protocol.action_dim) == EXPECTED_ACTION_DIM
        and int(protocol.future_action_blocks) >= EXPECTED_FUTURE_ACTION_BLOCKS
    )
    state_before = adapter.frozen_state_hash()
    metadata = adapter.metadata
    strict_load_ok = (
        protocol_ok
        and metadata.get("checkpoint_sha256") == checkpoint_identity["sha256"]
        and metadata.get("stable_worldmodel_commit") == runtime["expected_ref"]
        and metadata.get("adapter_id") == "stable_worldmodel_pldm_v1"
        and getattr(adapter.model, "training", True) is False
        and all(not parameter.requires_grad for parameter in adapter.model.parameters())
    )
    if not strict_load_ok:
        raise RuntimeError("Development strict checkpoint-load contract failed")
    predicted = adapter.rollout_latents(
        histories,
        action_prefixes,
        batch_size=int(config["runtime"]["rollout_batch_size"]),
    )
    native_target = adapter.encode_pixels(
        targets,
        batch_size=int(config["runtime"]["encode_batch_size"]),
    )
    expected_prediction_shape = (len(targets), EXPECTED_FUTURE_ACTION_BLOCKS, *native_target.shape[1:])
    geometry_ok = tuple(predicted.shape) == expected_prediction_shape
    finite_inputs = bool(
        np.isfinite(histories).all()
        and np.isfinite(action_prefixes).all()
        and np.isfinite(targets).all()
    )
    if geometry_ok:
        difference = predicted[:, 0].astype(np.float64) - native_target.astype(np.float64)
        mse = np.mean(np.square(difference), axis=tuple(range(1, difference.ndim)))
    else:
        mse = np.asarray([], dtype=np.float64)
    finite_mse = bool(geometry_ok and finite_inputs and np.isfinite(predicted).all() and np.isfinite(native_target).all() and np.isfinite(mse).all())
    state_after = adapter.frozen_state_hash()
    weight_ok = state_before == state_after
    checks = {
        "strict_native_checkpoint_load": {
            "passed": strict_load_ok,
            "checkpoint_sha256": checkpoint_identity["sha256"],
            "model_state_sha256_before": state_before,
            "protocol": {
                "history_tokens": int(protocol.history_tokens),
                "raw_steps_per_action_block": int(protocol.action_block_raw_steps),
                "action_dim": int(protocol.action_dim),
                "future_action_blocks": int(protocol.future_action_blocks),
            },
        },
        "complete_heldout_manifest_coverage": {"passed": sample_audit["samples"] == 384 and sample_audit["scenarios"] == 96, **sample_audit},
        "prefix_autoregressive_geometry": {
            "passed": geometry_ok,
            "input_history_tokens": EXPECTED_HISTORY_TOKENS,
            "input_action_blocks": EXPECTED_HISTORY_TOKENS,
            "future_action_blocks": EXPECTED_FUTURE_ACTION_BLOCKS,
            "target_observation_index": EXPECTED_HISTORY_TOKENS,
            "prediction_shape": [int(size) for size in predicted.shape],
            "native_target_shape": [int(size) for size in native_target.shape],
        },
        "native_future_latent_mse_finiteness": {
            "passed": finite_mse,
            "samples": int(len(mse)),
            "mse_value_withheld_not_a_score": True,
            "per_sample_mse_identity": canonical_array_identity(mse) if finite_mse else None,
        },
        "frozen_weight_audit": {
            "passed": weight_ok,
            "state_hash_before": state_before,
            "state_hash_after": state_after,
        },
        "public_boundary": {
            "passed": True,
            "public_payload_accessed": False,
            "formal_public_icl_executed": False,
            "cem_executed": False,
            "checkpoint_selection": False,
            "scoreboard_score_emitted": False,
        },
    }
    passed = all(value["passed"] for value in checks.values())
    return {
        "schema_version": 1,
        "development_id": DEVELOPMENT_ID,
        "completion_id": COMPLETION_ID,
        "seed": int(seed),
        "status": "passed_infrastructure_readiness" if passed else "failed_infrastructure_readiness",
        "passed": passed,
        "scope": DEVELOPMENT_SCOPE,
        "development_config": identity(config_path, repo_root=ROOT),
        "development_manifest": identity(manifest_path, repo_root=ROOT),
        "implementation": implementation,
        "post_interruption_execution_disclosure_amendment": amendment,
        "execution_disclosure": execution_disclosure,
        "checkpoint": checkpoint_identity,
        "checkpoint_model_state_sha256": state_before,
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(
        config_path=args.config,
        manifest_path=args.manifest,
        seed=int(args.seed),
        device=str(args.device),
        output=args.output,
    )
    output = resolve_local_output(args.output, repo_root=ROOT)
    write_json_exclusive(output, result)
    print(json.dumps({"status": result["status"], "output": logical_path(output, repo_root=ROOT)}, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
