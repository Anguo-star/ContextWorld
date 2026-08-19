#!/usr/bin/env python3
"""Fail-closed freezer for the ActionStrength PLDM formal-evaluation binding.

It validates only preregistered code, completed training artifacts, and
CPU-only runtime-injection receipts.  In particular, it never opens or hashes
the Public Test payload; Public reads remain unavailable until this receipt
has passed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BINDING = ROOT / "configs/benchmark/pusht_action_strength_pldm_evaluation_binding_v1.yaml"
DEFAULT_OUTPUT = (
    ROOT
    / "artifacts/evaluation/history3/pusht_action_strength_pldm_reference_completion_v1"
    / "evaluation_binding_v1/evaluation_binding_receipt.json"
)


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML object: {path}")
    return payload


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
        raise FileNotFoundError(f"Missing .git in runtime worktree: {worktree}")
    head = (gitdir / "HEAD").read_text(encoding="utf-8").strip()
    if head.startswith("ref: "):
        ref = head[len("ref: ") :]
        target = gitdir / ref
        if not target.is_file():
            common = (gitdir / "commondir").read_text(encoding="utf-8").strip()
            target = (gitdir / common / ref).resolve()
        head = target.read_text(encoding="utf-8").strip()
    if len(head) != 40:
        raise RuntimeError(f"Could not resolve commit at {gitdir / 'HEAD'}")
    return head


def _nested(mapping: dict[str, Any], *keys: str) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _record(
    checks: dict[str, dict[str, Any]],
    name: str,
    passed: bool,
    **details: Any,
) -> None:
    checks[name] = {"passed": bool(passed), **details}


def _matches_metric(observed: Any, expected: Any) -> bool:
    if isinstance(expected, float):
        return isinstance(observed, (int, float)) and math.isclose(
            float(observed), expected, rel_tol=1e-12, abs_tol=1e-12
        )
    return observed == expected


def _file_check(
    checks: dict[str, dict[str, Any]],
    name: str,
    specification: dict[str, Any],
) -> Path:
    path = _resolve(specification["path"])
    observed = _sha256(path) if path.is_file() else None
    _record(
        checks,
        name,
        observed == specification["sha256"],
        path=str(path),
        expected_sha256=specification["sha256"],
        observed_sha256=observed,
    )
    return path


def _validate_runtime_receipt(
    *,
    binding: dict[str, Any],
    checkpoint: dict[str, Any],
    track: str,
    checks: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    root = _resolve(binding["artifacts"]["root"])
    seed = int(checkpoint["seed"])
    path = root / "evaluation_binding_v1/runtime_preflight" / f"{track}_seed_{seed}.json"
    try:
        receipt = _load_json(path)
        sources = binding["evaluator_sources"]
        expected_runner = sources[f"{track}_runner"]
        strict = receipt.get("strict_load", {})
        expected_checkpoint = checkpoint["checkpoint"]
        expected = bool(
            receipt.get("track") == track
            and receipt.get("runtime_assignment_injected_in_memory_only") is True
            and receipt.get("original_runner_modified") is False
            and receipt.get("stable_worldmodel_root")
            == str(_resolve(binding["stable_worldmodel"]["worktree"]))
            and receipt.get("stable_worldmodel_expected_ref")
            == binding["stable_worldmodel"]["expected_ref"]
            and receipt.get("stable_worldmodel_observed_ref")
            == binding["stable_worldmodel"]["expected_ref"]
            and _nested(receipt, "runner", "sha256") == expected_runner["sha256"]
            and strict.get("checkpoint_sha256") == expected_checkpoint["sha256"]
            and strict.get("model_state_sha256")
            == expected_checkpoint["model_state_sha256"]
            and int(strict.get("parameters", -1)) == 18034478
        )
        if track == "planning":
            loader = sources["planning_shared_model_loader"]
            expected = expected and (
                _nested(receipt, "shared_model_loader", "sha256")
                == loader["sha256"]
            )
        _record(
            checks,
            f"{track}_runtime_preflight_seed_{seed}",
            expected,
            path=str(path),
            sha256=_sha256(path),
            strict_load=strict,
        )
        return receipt
    except Exception as error:
        _record(
            checks,
            f"{track}_runtime_preflight_seed_{seed}",
            False,
            path=str(path),
            error=f"{type(error).__name__}: {error}",
        )
        return None


def _validate_checkpoint(
    *,
    binding: dict[str, Any],
    entry: dict[str, Any],
    checks: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    seed = int(entry["seed"])
    checkpoint_path = _file_check(checks, f"checkpoint_{seed}", entry["checkpoint"])
    _file_check(checks, f"checkpoint_config_{seed}", entry["config"])
    report_path = _file_check(checks, f"training_report_{seed}", entry["training_report"])
    gate_path = _file_check(checks, f"development_gate_{seed}", entry["development_gate"])
    report = _load_json(report_path)
    gate = _load_json(gate_path)
    result_rows = report.get("results")
    result = result_rows[0] if isinstance(result_rows, list) and len(result_rows) == 1 else {}
    final = result.get("final_checkpoint", {})
    snapshots = result.get("snapshots")
    final_snapshot = snapshots[-1] if isinstance(snapshots, list) and snapshots else {}
    metrics = final_snapshot.get("hidden_evaluation", {})
    expected_metrics = entry["final_development_metrics"]
    metrics_match = all(
        _matches_metric(metrics.get(key), value)
        for key, value in expected_metrics.items()
        if key
        not in {
            "optimizer_step",
            "correct_future_mean",
            "incorrect_future_mean",
        }
    ) and _matches_metric(
        _nested(metrics, "prediction_mse", "correct_future_mean"),
        expected_metrics["correct_future_mean"],
    ) and _matches_metric(
        _nested(metrics, "prediction_mse", "incorrect_future_mean"),
        expected_metrics["incorrect_future_mean"],
    )
    _record(
        checks,
        f"fixed_training_report_and_development_metrics_{seed}",
        bool(
            int(result.get("seed", -1)) == seed
            and int(result.get("optimizer_steps", -1))
            == int(binding["completion"]["fixed_optimizer_steps"])
            and final.get("path") == str(checkpoint_path)
            and final.get("sha256") == entry["checkpoint"]["sha256"]
            and final.get("model_state_sha256")
            == entry["checkpoint"]["model_state_sha256"]
            and final_snapshot.get("optimizer_step")
            == int(binding["completion"]["fixed_optimizer_steps"])
            and metrics_match
        ),
        final_checkpoint=final,
        final_development_metrics=metrics,
    )
    hidden = _nested(report, "provenance", "hidden_data") or {}
    supervision = result.get("prediction_supervision", {})
    _record(
        checks,
        f"training_public_exclusion_{seed}",
        bool(
            hidden.get("evaluation_source") == "development_validation_lance"
            and hidden.get("public_test_used") is False
            and supervision.get("public_test_used") is False
            and int(hidden.get("eval_pairs", -1)) == 256
        ),
        provenance_hidden_data=hidden,
        prediction_supervision=supervision,
    )
    gate_checkpoint = gate.get("checkpoint", {})
    _record(
        checks,
        f"development_gate_public_exclusion_{seed}",
        bool(
            gate.get("status") == "passed_development_only_checkpoint_gate"
            and gate.get("public_test_used") is False
            and gate.get("formal_icl_or_cem_executed") is False
            and gate.get("seed") == seed
            and gate_checkpoint.get("sha256") == entry["checkpoint"]["sha256"]
            and gate_checkpoint.get("config_sha256") == entry["config"]["sha256"]
            and gate_checkpoint.get("model_state_sha256")
            == entry["checkpoint"]["model_state_sha256"]
            and _nested(gate, "training_report", "sha256")
            == entry["training_report"]["sha256"]
        ),
        development_gate=gate,
    )
    runtime = {
        track: _validate_runtime_receipt(
            binding=binding,
            checkpoint=entry,
            track=track,
            checks=checks,
        )
        for track in ("planning", "retention")
    }
    return {
        "seed": seed,
        "checkpoint": entry["checkpoint"],
        "config": entry["config"],
        "training_report": entry["training_report"],
        "development_gate": entry["development_gate"],
        "runtime_preflight": runtime,
    }


def _validate_planning_equivalence(
    *,
    binding: dict[str, Any],
    binding_path: Path,
    entry: dict[str, Any],
    checks: dict[str, dict[str, Any]],
) -> None:
    seed = int(entry["seed"])
    path = (
        _resolve(binding["artifacts"]["planning_equivalence_root"])
        / f"seed_{seed}.json"
    )
    try:
        payload = _load_json(path)
        tensors = payload.get("planner_tensor_equivalence", {})
        expected_tensor_names = {
            "history_embedding",
            "goal_embedding",
            "action_embedding",
            "prediction",
        }
        states = _nested(payload, "models", "state_sha256") or {}
        passed = bool(
            payload.get("status") == "passed"
            and payload.get("passed") is True
            and _nested(payload, "binding", "sha256") == _sha256(binding_path)
            and payload.get("seed") == seed
            and _nested(payload, "development_input", "public_test_used") is False
            and _nested(payload, "checkpoint", "observed_sha256")
            == entry["checkpoint"]["sha256"]
            and _nested(payload, "checkpoint", "model_state_sha256")
            == entry["checkpoint"]["model_state_sha256"]
            and states.get("pldm") == entry["checkpoint"]["model_state_sha256"]
            and states.get("planning_shell")
            == entry["checkpoint"]["model_state_sha256"]
            and _nested(payload, "planner_code_equivalence", "passed") is True
            and set(tensors) == expected_tensor_names
            and all(
                row.get("allclose_rtol_1e-5_atol_1e-6") is True
                and float(row.get("max_absolute_difference", float("inf"))) <= 1e-6
                for row in tensors.values()
            )
        )
        _record(
            checks,
            f"planning_pldm_math_equivalence_seed_{seed}",
            passed,
            path=str(path),
            sha256=_sha256(path),
            model_classes=payload.get("models"),
            tensor_checks=tensors,
        )
    except Exception as error:
        _record(
            checks,
            f"planning_pldm_math_equivalence_seed_{seed}",
            False,
            path=str(path),
            error=f"{type(error).__name__}: {error}",
        )


def build_receipt(binding_path: Path) -> dict[str, Any]:
    binding = _load_yaml(binding_path)
    checks: dict[str, dict[str, Any]] = {}
    _record(
        checks,
        "binding_shape",
        bool(
            binding.get("schema_version") == 1
            and binding.get("binding_id")
            == "pusht_action_strength_pldm_evaluation_binding_v1"
            and binding.get("status")
            == "preregistered_after_training_before_formal_public_evaluation"
            and binding.get("scope", {}).get("model_family") == "PLDM"
            and binding.get("completion", {}).get("training_seeds")
            == [13313, 13314, 13315]
        ),
    )
    completion_path = _file_check(checks, "completion_config", binding["completion"])
    release_path = _file_check(checks, "release_config", binding["release"])
    completion = _load_yaml(completion_path)
    release = _load_yaml(release_path)
    _record(
        checks,
        "completion_identity",
        bool(
            completion.get("completion_id") == binding["completion"]["completion_id"]
            and completion.get("training", {}).get("recipe")
            == binding["completion"]["fixed_recipe"]
            and completion.get("training", {}).get("optimizer_steps")
            == binding["completion"]["fixed_optimizer_steps"]
            and completion.get("training", {}).get("seeds")
            == binding["completion"]["training_seeds"]
        ),
    )
    _record(
        checks,
        "release_identity_and_public_contract",
        bool(
            release.get("release_id") == binding["release"]["release_id"]
            and _nested(release, "runtime", "stable_worldmodel", "expected_ref")
            == binding["stable_worldmodel"]["expected_ref"]
            and _nested(release, "evaluation", "manifest_sha256")
            == _nested(binding, "public_test_identity", "manifest", "sha256")
            and _nested(release, "evaluation", "artifact_tree", "sha256")
            == _nested(binding, "public_test_identity", "query_table", "artifact_tree_sha256")
            and _nested(release, "evaluation", "pair_count")
            == _nested(binding, "public_test_identity", "query_table", "pair_count")
            and _nested(release, "evaluation", "condition_count")
            == _nested(binding, "public_test_identity", "query_table", "condition_count")
            and _nested(release, "evaluation", "planning_oracle", "sha256")
            == _nested(binding, "public_test_identity", "planning_oracle", "sha256")
            and _nested(release, "scoring", "original_task_retention", "query_catalog_sha256")
            == binding["public_test_identity"]["original_pusht_query_catalog_sha256"]
        ),
        public_payload_accessed=False,
    )
    runtime = binding["stable_worldmodel"]
    runtime_root = _resolve(runtime["worktree"])
    runtime_config = runtime_root / runtime["pldm_config"]
    _record(
        checks,
        "pinned_stable_worldmodel",
        bool(
            _git_head(runtime_root) == runtime["expected_ref"]
            and _sha256(runtime_config) == runtime["pldm_config_sha256"]
        ),
        worktree=str(runtime_root),
        expected_ref=runtime["expected_ref"],
        observed_ref=_git_head(runtime_root),
        pldm_config=str(runtime_config),
        pldm_config_sha256=_sha256(runtime_config),
    )
    for name, specification in binding["evaluator_sources"].items():
        _file_check(checks, f"evaluator_source_{name}", specification)
    release_identity = release.get("identity", {})
    mapped = {
        "adapters": "adapters",
        "data": "data_api",
        "score": "score_api",
        "cli": "command_line",
        "published_training_runner": "published_reference_runner",
        "planning_runner": "planning_runner",
        "retention_runner": "retention_runner",
    }
    _record(
        checks,
        "release_evaluator_sources_match_binding",
        all(
            _nested(release_identity, release_name, "sha256")
            == binding["evaluator_sources"][binding_name]["sha256"]
            for binding_name, release_name in mapped.items()
        ),
    )
    checkpoint_receipts = [
        _validate_checkpoint(binding=binding, entry=entry, checks=checks)
        for entry in binding["checkpoints"]
    ]
    for entry in binding["checkpoints"]:
        _validate_planning_equivalence(
            binding=binding,
            binding_path=binding_path,
            entry=entry,
            checks=checks,
        )
    formal_paths = [
        _resolve(binding["artifacts"][key])
        for key in ("formal_icl_root", "action_planning_root", "retention_root")
    ]
    _record(
        checks,
        "formal_public_evaluation_not_started_before_binding",
        not any(path.exists() for path in formal_paths),
        paths=[str(path) for path in formal_paths],
    )
    passed = all(row["passed"] for row in checks.values())
    return {
        "schema_version": 1,
        "binding_id": binding["binding_id"],
        "status": "passed_evaluation_binding_freeze" if passed else "failed_evaluation_binding_freeze",
        "passed": passed,
        "binding": {"path": str(binding_path), "sha256": _sha256(binding_path)},
        "release": binding["release"],
        "completion": binding["completion"],
        "stable_worldmodel": binding["stable_worldmodel"],
        "public_test": {
            **binding["public_test_identity"],
            "accessed_by_binding": False,
            "scored_by_binding": False,
        },
        "checkpoints": checkpoint_receipts,
        "checks": checks,
        "next_stage": (
            {
                "formal_public_icl_authorized": True,
                "action_planning_cem_authorized_after_three_seed_icl_gate": True,
                "original_pusht_cem_authorized_after_three_seed_icl_gate": True,
            }
            if passed
            else {
                "formal_public_icl_authorized": False,
                "action_planning_cem_authorized": False,
                "original_pusht_cem_authorized": False,
            }
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding", type=Path, default=DEFAULT_BINDING)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    binding_path = _resolve(args.binding)
    output = _resolve(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite additive freeze: {output}")
    receipt = build_receipt(binding_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"status": receipt["status"], "output": str(output)}))
    if not receipt["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
