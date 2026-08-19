#!/usr/bin/env python3
"""Freeze the post-Development, pre-Public Speed PLDM binding configuration.

Unlike the binding *receipt* freezer, this program creates the only dynamic
YAML input to that freezer.  It may run only after the three fixed-step
checkpoints and the three no-score Development readiness receipts exist.  It
never resolves, opens, hashes, or scores a Public ICL payload.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping

import yaml

from contextworld.benchmarks.speed_pldm_infrastructure_development import (
    COMPLETION_ID,
    DEVELOPMENT_ID,
    DEVELOPMENT_SCOPE,
    EXPECTED_SEEDS,
    identity,
    load_json,
    logical_path,
    resolve_local_output,
    resolve_source,
    root,
    same_identity,
    sha256_file,
)


ROOT = root()
DEVELOPMENT_CONFIG = (
    ROOT / "configs/benchmark/tworoom_speed_pldm_infrastructure_development_v1.yaml"
)
CEM_PREREG = ROOT / "configs/benchmark/tworoom_speed_pldm_cem_prereg_v1.yaml"
DEFAULT_OUTPUT = ROOT / "configs/benchmark/tworoom_speed_pldm_evaluation_binding_v1.yaml"
DEFAULT_BINDING_RECEIPT = (
    ROOT
    / "artifacts/evaluation/history3/tworoom_speed_pldm_reference_completion_v1"
    / "evaluation_binding_v1/evaluation_binding_receipt.json"
)
FORMAL_ROOT = (
    ROOT
    / "artifacts/evaluation/history3/tworoom_speed_pldm_reference_completion_v1"
    / "formal_icl_v1"
)
PLANNED_CEM_ROOTS = (
    ROOT
    / "artifacts/evaluation/history3/tworoom_speed_pldm_reference_completion_v1"
    / "formal_action_planning_cem_v1",
    ROOT
    / "artifacts/evaluation/history3/tworoom_speed_pldm_reference_completion_v1"
    / "formal_original_tworoom_retention_cem_v1",
)


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return value


def _git_head(worktree: Path) -> str:
    marker = worktree / ".git"
    if marker.is_file():
        text = marker.read_text(encoding="utf-8").strip()
        if not text.startswith("gitdir: "):
            raise RuntimeError(f"Unsupported .git pointer: {marker}")
        gitdir = Path(text[len("gitdir: ") :]).expanduser()
    elif marker.is_dir():
        gitdir = marker
    else:
        raise FileNotFoundError(f"Pinned Stable-WorldModel worktree has no .git: {worktree}")
    head = (gitdir / "HEAD").read_text(encoding="utf-8").strip()
    if head.startswith("ref: "):
        ref = head[len("ref: ") :]
        target = gitdir / ref
        if not target.is_file():
            common = gitdir / "commondir"
            if not common.is_file():
                raise RuntimeError(f"Cannot resolve Stable-WorldModel HEAD ref: {ref}")
            target = (gitdir / common.read_text(encoding="utf-8").strip() / ref).resolve()
        head = target.read_text(encoding="utf-8").strip()
    if len(head) != 40:
        raise RuntimeError("Pinned Stable-WorldModel HEAD is not a commit SHA")
    return head


def _spec(path: Path) -> dict[str, Any]:
    return identity(path, repo_root=ROOT)


def _sha_spec(path: Path) -> dict[str, str]:
    item = _spec(path)
    return {"path": item["path"], "sha256": item["sha256"]}


def _source_spec(path: str) -> dict[str, str]:
    resolved = resolve_source(path, repo_root=ROOT)
    return _sha_spec(resolved)


def _write_yaml_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o644)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            yaml.safe_dump(
                payload,
                stream,
                allow_unicode=True,
                default_flow_style=False,
                sort_keys=False,
            )
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
            "Binding configuration output must equal its dedicated destination "
            f"{logical_path(expected, repo_root=ROOT)}"
        )
    return actual


def _load_development_evidence(config_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return immutable Development evidence, tied to final training artifacts."""

    config = _load_yaml(config_path)
    if not (
        config.get("development_id") == DEVELOPMENT_ID
        and config.get("completion_id") == COMPLETION_ID
        and config.get("scope") == DEVELOPMENT_SCOPE
        and config.get("status")
        == "preregistered_during_fixed_training_before_development_manifest_or_inference"
    ):
        raise ValueError("Unexpected Speed Development preregistration")
    outputs = config.get("outputs")
    training = config.get("training_artifacts")
    if not isinstance(outputs, dict) or not isinstance(training, dict):
        raise ValueError("Development preregistration lacks output/training declarations")
    manifest_path = resolve_source(outputs.get("manifest", ""), repo_root=ROOT)
    manifest = load_json(manifest_path)
    config_identity = _spec(config_path)
    manifest_identity = _spec(manifest_path)
    if not (
        manifest.get("schema_version") == 1
        and manifest.get("development_id") == DEVELOPMENT_ID
        and manifest.get("completion_id") == COMPLETION_ID
        and manifest.get("status") == "frozen_prepublic_development_manifest"
        and manifest.get("passed") is True
        and manifest.get("scope") == DEVELOPMENT_SCOPE
        and manifest.get("development_config") == config_identity
        and manifest.get("public_payload_accessed") is False
        and manifest.get("formal_public_or_cem_artifacts_present") is False
        and manifest.get("coverage", {}).get("validation_scenarios") == 96
        and manifest.get("coverage", {}).get("total_samples") == 384
        and manifest.get("coverage", {}).get("all_actual_indices_unique_per_scenario")
        is True
        and manifest.get("coverage", {}).get("all_source_spans_continuous")
        is True
    ):
        raise RuntimeError("Development manifest is not an intact pre-Public gate")

    manifest_training = manifest.get("training_checkpoints")
    if not (
        isinstance(manifest_training, list)
        and tuple(int(row.get("seed", -1)) for row in manifest_training)
        == EXPECTED_SEEDS
    ):
        raise RuntimeError("Development manifest lacks the frozen three-seed training chain")
    manifest_training_by_seed = {
        int(row["seed"]): row for row in manifest_training
    }

    entries = training.get("entries")
    receipt_paths = outputs.get("receipts")
    if not (
        isinstance(entries, list)
        and isinstance(receipt_paths, dict)
        and tuple(sorted(int(row.get("seed", -1)) for row in entries)) == EXPECTED_SEEDS
        and {int(seed) for seed in receipt_paths} == set(EXPECTED_SEEDS)
    ):
        raise ValueError("Development checkpoint/receipt declarations are incomplete")
    entries_by_seed = {int(row["seed"]): row for row in entries}
    receipts = []
    checkpoints = []
    for seed in EXPECTED_SEEDS:
        entry = entries_by_seed[seed]
        required = {"checkpoint", "checkpoint_config", "training_report", "loss_trace", "preflight"}
        if not required.issubset(entry):
            raise ValueError(f"Development training declaration is incomplete for seed {seed}")
        paths = {name: resolve_source(entry[name], repo_root=ROOT) for name in required}
        if any(not path.is_file() for path in paths.values()):
            raise FileNotFoundError(f"Missing final Speed training artifact for seed {seed}")
        checkpoint_identity = _spec(paths["checkpoint"])
        checkpoint_config_identity = _spec(paths["checkpoint_config"])
        report_identity = _spec(paths["training_report"])
        trace_identity = _spec(paths["loss_trace"])
        preflight_identity = _spec(paths["preflight"])
        report = load_json(paths["training_report"])
        preflight = load_json(paths["preflight"])
        trace = [
            json.loads(line)
            for line in paths["loss_trace"].read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        manifest_entry = manifest_training_by_seed[seed]
        completion_identity = manifest_entry.get("recovery_completion_receipt")
        if not isinstance(completion_identity, dict):
            raise RuntimeError(f"Development manifest lacks recovery completion evidence for seed {seed}")
        completion_path = resolve_source(completion_identity.get("path", ""), repo_root=ROOT)
        if _spec(completion_path) != completion_identity:
            raise RuntimeError(f"Recovery completion receipt identity drifted for seed {seed}")
        completion = load_json(completion_path)
        fixed_contract = manifest_entry.get("fixed_training_contract")
        if not (
            report.get("schema_version") == 1
            and report.get("passed") is True
            and report.get("run_name") == f"speed_pldm_reference_completion_v1_s{seed}"
            and report.get("training", {}).get("training_complete") is True
            and int(report.get("training", {}).get("global_step", -1)) == 12840
            and report.get("artifacts", {}).get("pretrained_sha256")
            == checkpoint_identity["sha256"]
            and report.get("artifacts", {}).get("pretrained_config_sha256")
            == checkpoint_config_identity["sha256"]
            and report.get("artifacts", {}).get("loss_trace", {}).get("sha256")
            == trace_identity["sha256"]
            and trace
            and int(trace[-1].get("optimizer_step", -1)) == 12840
            and report.get("save_load_exact") is True
            and report.get("training", {}).get("terminal_report_recovery_optimizer_steps")
            == 0
            and report.get("terminal_report_recovery", {}).get(
                "training_or_optimizer_execution"
            )
            is False
            and preflight.get("completion_id") == COMPLETION_ID
            and preflight.get("status") == "passed"
            and preflight.get("seed") == seed
            and preflight.get("training_started") is False
            and "training_completed" not in preflight
            and "training_failed" not in preflight
            and manifest_entry.get("checkpoint") == checkpoint_identity
            and manifest_entry.get("checkpoint_config") == checkpoint_config_identity
            and manifest_entry.get("training_report") == report_identity
            and manifest_entry.get("loss_trace") == trace_identity
            and manifest_entry.get("preflight") == preflight_identity
            and fixed_contract
            == {
                "passed": True,
                "checkpoint_selection": "final_fixed_step",
                "early_stopping": False,
                "optimizer_steps": 12840,
                "completion_evidence": (
                    "recovery_completion_receipt_and_final_training_report"
                ),
                "preflight_receipt_role": (
                    "preflight_only_not_training_completion_evidence"
                ),
                "historical_preflight_immutability_claimed": False,
            }
            and completion.get("completion_id") == COMPLETION_ID
            and completion.get("seed") == seed
            and completion.get("status") == "completed_fixed_budget_required_resume"
            and completion.get("passed") is True
            and completion.get("training_report") == report_identity
            and completion.get("final_checkpoint") == checkpoint_identity
            and completion.get("resume_proof", {}).get("initial_global_step")
            == 10272
            and completion.get("resume_proof", {}).get("final_global_step")
            == 12840
            and completion.get("evaluation_executed") is False
            and completion.get("public_test_accessed") is False
        ):
            raise RuntimeError(f"Final fixed-step training contract is invalid for seed {seed}")
        receipt_path = resolve_source(receipt_paths[str(seed)], repo_root=ROOT)
        receipt_identity = _spec(receipt_path)
        receipt = load_json(receipt_path)
        checks = receipt.get("checks", {})
        state = receipt.get("checkpoint_model_state_sha256")
        if not (
            receipt.get("schema_version") == 1
            and receipt.get("development_id") == DEVELOPMENT_ID
            and receipt.get("completion_id") == COMPLETION_ID
            and receipt.get("seed") == seed
            and receipt.get("status") == "passed_infrastructure_readiness"
            and receipt.get("passed") is True
            and receipt.get("scope") == DEVELOPMENT_SCOPE
            and receipt.get("development_config") == config_identity
            and receipt.get("development_manifest") == manifest_identity
            and receipt.get("checkpoint") == checkpoint_identity
            and isinstance(state, str)
            and len(state) == 64
            and isinstance(checks, dict)
            and all(
                checks.get(name, {}).get("passed") is True
                for name in (
                    "strict_native_checkpoint_load",
                    "complete_heldout_manifest_coverage",
                    "prefix_autoregressive_geometry",
                    "native_future_latent_mse_finiteness",
                    "frozen_weight_audit",
                    "public_boundary",
                )
            )
            and checks.get("native_future_latent_mse_finiteness", {}).get(
                "mse_value_withheld_not_a_score"
            )
            is True
            and checks.get("frozen_weight_audit", {}).get("state_hash_before") == state
            and checks.get("frozen_weight_audit", {}).get("state_hash_after") == state
            and checks.get("public_boundary", {}).get("public_payload_accessed") is False
            and checks.get("public_boundary", {}).get("checkpoint_selection") is False
            and checks.get("public_boundary", {}).get("scoreboard_score_emitted") is False
        ):
            raise RuntimeError(f"Development readiness receipt is invalid for seed {seed}")
        receipts.append({"seed": seed, "receipt": receipt_identity})
        checkpoints.append(
            {
                "seed": seed,
                "run_name": f"speed_pldm_reference_completion_v1_s{seed}",
                "checkpoint": {**checkpoint_identity, "model_state_sha256": state},
                "config": checkpoint_config_identity,
                "training_report": report_identity,
                "loss_trace": trace_identity,
                "preflight": preflight_identity,
            }
        )
    return (
        {"config": config_identity, "manifest": manifest_identity, "receipts": receipts},
        {"checkpoints": checkpoints, "development_config": config},
    )


def _cem_prepublic_protocol(
    *,
    completion: Mapping[str, Any],
    completion_identity: Mapping[str, Any],
    release: Mapping[str, Any],
    release_identity: Mapping[str, Any],
    boundary_identity: Mapping[str, Any],
    normalizer_identity: Mapping[str, Any],
    stable_worldmodel: Mapping[str, Any],
) -> dict[str, Any]:
    """Copy the full CEM authority into the binding *before* Public ICL.

    ``freeze_tworoom_speed_pldm_cem_binding_v1`` may later validate and bind
    these inputs, but it is prohibited from deciding which runner, catalog,
    threshold, output namespace, or comparison code to use after seeing 3/3.
    The CEM preregistration validator is deliberately pure: it hashes static
    files only and never opens Public ICL inputs or constructs an environment.
    """

    from scripts.freeze_tworoom_speed_pldm_cem_binding_v1 import _validate_static_prereg

    static, sources = _validate_static_prereg(CEM_PREREG)
    action = static["tracks"]["action_planning_cem"]
    retention = static["tracks"]["original_task_retention_cem"]
    outputs = static["outputs"]
    if not (
        static.get("completion_id") == COMPLETION_ID
        and static.get("release_id") == release.get("release_id")
        and same_identity(sources["completion_config"], completion_identity)
        and same_identity(sources["release_config"], release_identity)
        and same_identity(sources["behavioral_claim_boundary"], boundary_identity)
        and outputs.get("cem_binding")
        == logical_path(
            FORMAL_ROOT / "cem_binding_v1.json", repo_root=ROOT
        )
        and outputs.get("action_planning", {}).get("root")
        == logical_path(PLANNED_CEM_ROOTS[0], repo_root=ROOT)
        and outputs.get("original_task_retention", {}).get("root")
        == logical_path(PLANNED_CEM_ROOTS[1], repo_root=ROOT)
        and action.get("metric", {}).get("result_semantics")
        == "EXECUTED_VALID_DESCRIPTIVE"
        and action.get("metric", {}).get("performance_threshold") is None
        and action.get("metric", {}).get("pass_threshold") is None
        and retention.get("metric", {}).get("result_semantics")
        == "PAIRED_NONINFERIORITY_RETENTION"
        and retention.get("metric", {}).get("paired_noninferiority", {}).get(
            "success_rate_delta_lower_bound"
        )
        == -0.05
        and retention.get("metric", {}).get("paired_noninferiority", {}).get(
            "final_distance_delta_upper_bound_px"
        )
        == 5.0
    ):
        raise RuntimeError("Pre-Public Speed CEM authority disagrees with fixed training inputs")

    source_identities = {
        name: source
        for name, source in sources.items()
        if name
        not in {
            "completion_config",
            "release_config",
            "behavioral_claim_boundary",
        }
    }
    required_sources = {
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
    }
    if set(source_identities) != required_sources:
        raise RuntimeError("Pre-Public Speed CEM source closure is incomplete")
    return {
        "cem_preregistration_id": static["cem_preregistration_id"],
        "status": "frozen_prepublic_cem_execution_and_decision_authority",
        "preregistration": sources["preregistration"],
        "completion": {
            **dict(completion_identity),
            "completion_id": COMPLETION_ID,
            "training_seeds": list(EXPECTED_SEEDS),
            "fixed_optimizer_steps": 12840,
            "model_id": "H3_Speed_PLDM_ReferenceCompletion",
            "initial_model_state_sha256": completion["initialization"][
                "expected_model_state_sha256"
            ],
        },
        "release": {**dict(release_identity), "release_id": release["release_id"]},
        "behavioral_claim_boundary": dict(boundary_identity),
        "normalizer": dict(normalizer_identity),
        "stable_worldmodel": dict(stable_worldmodel),
        "source_identities": source_identities,
        "common": static["common"],
        "tracks": {
            "action_planning_cem": {
                "evaluation_kind": action["evaluation_kind"],
                "source": {
                    "release_planning_track": action["source"]["release_planning_track"],
                    "catalog": sources["action_catalog"],
                    "source_protocol": sources["source_protocol"],
                    "catalog_validator": sources["action_catalog_validator"],
                    "episode_oracle": sources["action_episode_oracle"],
                    "runner_core": sources["action_runner_core"],
                    "speed_cli": sources["speed_cli"],
                    "speed_score": sources["speed_score"],
                },
                "grouping": action["grouping"],
                "protocol": action["protocol"],
                "metric": action["metric"],
            },
            "original_task_retention_cem": {
                "evaluation_kind": retention["evaluation_kind"],
                "source": {
                    "completion_field": retention["source"]["completion_field"],
                    "catalog": sources["retention_catalog"],
                    "catalog_builder": sources["retention_catalog_builder"],
                    "episode_oracle": sources["retention_episode_oracle"],
                    "runner_core": sources["retention_runner_core"],
                    "frozen_baseline_wrapper": sources[
                        "retention_frozen_baseline_wrapper"
                    ],
                },
                "grouping": retention["grouping"],
                "protocol": retention["protocol"],
                "metric": retention["metric"],
                "frozen_paired_baseline": retention["frozen_paired_baseline"],
            },
        },
        "implementation": {
            "formal_runner": sources["implementation_formal_runner"],
            "aggregate_freezer": sources["implementation_aggregate_freezer"],
            "binding_freezer": sources["implementation_binding_freezer"],
            "adapter_boundary": sources["implementation_adapter_boundary"],
            "development_contract": sources["implementation_development_contract"],
            "paired_retention_comparator": sources[
                "implementation_paired_retention_comparator"
            ],
        },
        "outputs": outputs,
        "authority": {
            "all_source_identities_frozen_before_public_icl": True,
            "post_icl_cem_binding_may_only_validate_and_rebind_this_closure": True,
            "action_planning_outcomes_are_descriptive_not_a_model_gate": True,
            "retention_pass_fail_uses_only_frozen_paired_noninferiority": True,
        },
    }


def build_binding(config_path: Path = DEVELOPMENT_CONFIG) -> dict[str, Any]:
    config_path = resolve_source(config_path, repo_root=ROOT)
    development, training = _load_development_evidence(config_path)
    completion_path = resolve_source(
        training["development_config"]["frozen_inputs"]["completion_config"]["path"],
        repo_root=ROOT,
    )
    completion = _load_yaml(completion_path)
    completion_identity = _spec(completion_path)
    release_path = resolve_source(
        completion.get("evaluation", {}).get("icl", {}).get("release_config", ""),
        repo_root=ROOT,
    )
    release = _load_yaml(release_path)
    release_identity = _spec(release_path)
    normalizer_path = resolve_source(
        training["development_config"]["frozen_inputs"]["normalizer"]["path"],
        repo_root=ROOT,
    )
    normalizer_identity = _spec(normalizer_path)
    boundary_path = resolve_source(
        training["development_config"]["frozen_inputs"]["behavioral_claim_boundary"]["path"],
        repo_root=ROOT,
    )
    boundary_identity = _spec(boundary_path)
    boundary = _load_yaml(boundary_path)
    runtime = training["development_config"].get("stable_worldmodel", {})
    runtime_root = Path(runtime.get("worktree", "")).resolve()
    pldm_config = runtime_root / str(runtime.get("pldm_config", ""))
    expected_ref = runtime.get("expected_ref")
    tracks = completion.get("evaluation", {}).get("icl", {}).get("tracks")
    scorer = _source_spec("contextworld/benchmarks/speed_icl_score.py")
    if not (
        completion.get("completion_id") == COMPLETION_ID
        and completion.get("training", {}).get("seeds") == list(EXPECTED_SEEDS)
        and int(completion.get("training", {}).get("optimizer_steps", -1)) == 12840
        and completion.get("training", {}).get("model_id")
        == "H3_Speed_PLDM_ReferenceCompletion"
        and release.get("release_id") == "contextworld_tworoom_speed_icl_history3_v1"
        and release.get("evaluation", {}).get("normalizer_sha256") == normalizer_identity["sha256"]
        and release.get("runtime", {}).get("stable_worldmodel", {}).get("expected_ref")
        == expected_ref
        and tracks == release.get("scope", {}).get("public_tracks")
        and _git_head(runtime_root) == expected_ref
        and pldm_config.is_file()
        and sha256_file(pldm_config) == runtime.get("pldm_config_sha256")
        and boundary.get("completion_id") == COMPLETION_ID
        and boundary.get("release_id") == release.get("release_id")
        # The behavioral boundary predates generated size metadata, so compare
        # the observed richer identity against its path/SHA-only declaration.
        and same_identity(
            completion_identity, boundary.get("frozen_inputs", {}).get("completion_config")
        )
        and same_identity(
            release_identity, boundary.get("frozen_inputs", {}).get("speed_release")
        )
        and same_identity(scorer, boundary.get("frozen_inputs", {}).get("public_scorer"))
    ):
        raise RuntimeError("Post-Development Speed binding inputs are inconsistent")
    if DEFAULT_BINDING_RECEIPT.exists() or FORMAL_ROOT.exists() or any(
        path.exists() for path in PLANNED_CEM_ROOTS
    ):
        raise RuntimeError("Public/CEM or binding artifacts exist before binding configuration freeze")
    stable_worldmodel = {
        "worktree": str(runtime_root),
        "expected_ref": expected_ref,
        "pldm_config": str(runtime["pldm_config"]),
        "pldm_config_sha256": runtime["pldm_config_sha256"],
    }
    cem_protocol = _cem_prepublic_protocol(
        completion=completion,
        completion_identity=completion_identity,
        release=release,
        release_identity=release_identity,
        boundary_identity=boundary_identity,
        normalizer_identity=normalizer_identity,
        stable_worldmodel=stable_worldmodel,
    )
    cem_evaluator_sources = {
        f"cem_{name}": specification
        for name, specification in cem_protocol["source_identities"].items()
    }
    return {
        "schema_version": 1,
        "binding_id": "tworoom_speed_pldm_evaluation_binding_v1",
        "status": "preregistered_after_training_before_formal_public_evaluation",
        "completion": {
            **completion_identity,
            "completion_id": COMPLETION_ID,
            "training_seeds": list(EXPECTED_SEEDS),
            "fixed_optimizer_steps": 12840,
            "model_id": "H3_Speed_PLDM_ReferenceCompletion",
            "initial_model_state_sha256": completion["initialization"][
                "expected_model_state_sha256"
            ],
        },
        "release": {**release_identity, "release_id": release["release_id"]},
        "normalizer": normalizer_identity,
        "stable_worldmodel": stable_worldmodel,
        "development": development,
        "formal_icl": {
            "tracks": tracks,
            "encode_batch_size": 64,
            "rollout_batch_size": 64,
            "bundle_batch_size": 64,
        },
        "behavioral_claim_boundary": boundary_identity,
        "cem_protocol": cem_protocol,
        "evaluator_sources": {
            "speed_icl_score": scorer,
            "formal_icl_evaluator": _source_spec(
                "scripts/eval_tworoom_speed_pldm_formal_icl_v1.py"
            ),
            "binding_freezer": _source_spec(
                "scripts/freeze_tworoom_speed_pldm_evaluation_binding_v1.py"
            ),
            "development_contract": _source_spec(
                "contextworld/benchmarks/speed_pldm_infrastructure_development.py"
            ),
            "adapter_boundary": _source_spec("contextworld/benchmarks/adapters.py"),
            **cem_evaluator_sources,
        },
        "checkpoints": training["checkpoints"],
        "artifacts": {
            "formal_icl_root": logical_path(FORMAL_ROOT, repo_root=ROOT),
            "action_planning_root": logical_path(PLANNED_CEM_ROOTS[0], repo_root=ROOT),
            "retention_root": logical_path(PLANNED_CEM_ROOTS[1], repo_root=ROOT),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-config", type=Path, default=DEVELOPMENT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = _assert_output(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite binding configuration: {output}")
    binding = build_binding(args.development_config)
    _write_yaml_exclusive(output, binding)
    print(
        json.dumps(
            {
                "binding_id": binding["binding_id"],
                "status": binding["status"],
                "output": logical_path(output, repo_root=ROOT),
                "public_payload_accessed": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
