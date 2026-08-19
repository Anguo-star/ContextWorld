#!/usr/bin/env python3
"""Fail-closed launcher for the additive Speed/ActionStrength PLDM completions.

It derives a raw ``model.*`` state mapping from the legacy TwoRoom Lightning
checkpoint for StableWM's ``.pt + config.json`` loader.  Source identity and
strict state-dict loading are checked before any optimizer state is created.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contextworld.paths import resolve_contextworld_path
from contextworld.synthesis.manifest import write_json


CONFIGS = {
    "speed": ROOT / "configs/benchmark/tworoom_speed_pldm_reference_completion_v1.yaml",
    "action-strength": ROOT / "configs/benchmark/pusht_action_strength_pldm_reference_completion_v1.yaml",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _state_mapping_sha256(state: dict[str, Any]) -> str:
    """Hash a raw state mapping with the adapter's model-state contract."""

    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _git_head(repo: Path) -> str:
    marker = repo / ".git"
    git_dir = marker if marker.is_dir() else (
        repo / marker.read_text(encoding="utf-8").strip().removeprefix("gitdir: ")
    )
    head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
    if not head.startswith("ref: "):
        return head
    ref = head.removeprefix("ref: ")
    loose = git_dir / ref
    if loose.is_file():
        return loose.read_text(encoding="utf-8").strip()
    for line in (git_dir / "packed-refs").read_text(encoding="utf-8").splitlines():
        if line and not line.startswith(("#", "^")):
            commit, _, name = line.partition(" ")
            if name == ref:
                return commit
    raise RuntimeError(f"Cannot resolve Git HEAD for {repo}")


def _load(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError(f"Unsupported completion config: {path}")
    if payload.get("status") != "preregistered_before_training":
        raise ValueError("Completion configuration is not preregistered")
    return payload


def _resolved(value: str | Path) -> Path:
    return resolve_contextworld_path(value, repo_root=ROOT)


def _completion_root(config: dict[str, Any]) -> Path:
    """Keep new receipts in this writable checkout, never a read-only mount."""

    raw = Path(config["artifacts"]["root"])
    return raw.resolve() if raw.is_absolute() else (ROOT / raw).resolve()


def _require_hash(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    observed = _sha256(path)
    if observed != expected:
        raise RuntimeError(
            f"{label} SHA-256 drifted: expected={expected}, observed={observed}"
        )
    return observed


def _strict_adapter(component: str, config: dict[str, Any]) -> dict[str, Any]:
    init = config["initialization"]
    runtime = config["runtime"]["stable_worldmodel"]
    checkpoint = Path(init["checkpoint"])
    if component == "speed":
        from contextworld.benchmarks.adapters import StableWorldModelPLDMAdapter

        normalizer = _resolved(config["data"]["normalizer"]["path"])
        adapter = StableWorldModelPLDMAdapter.from_checkpoint(
            checkpoint,
            normalizer=normalizer,
            repo_root=ROOT,
            stablewm_repo=str(runtime["repo"]),
            stablewm_ref=str(runtime["commit"]),
            device="cpu",
        )
    else:
        from contextworld.benchmarks.adapters import (
            StableWorldModelPLDMActionStrengthAdapter,
        )

        adapter = StableWorldModelPLDMActionStrengthAdapter.from_checkpoint(
            checkpoint,
            action_mean=[-0.007812564261257648, 0.006860687397420406],
            action_std=[0.20846743881702423, 0.2067486196756363],
            repo_root=ROOT,
            stablewm_repo=str(runtime["repo"]),
            stablewm_ref=str(runtime["commit"]),
            device="cpu",
        )
    state = adapter.frozen_state_hash()
    if state != init["expected_model_state_sha256"]:
        raise RuntimeError("Strict-load model-state receipt differs from preregistration")
    protocol = adapter.protocol
    if not (
        protocol.history_tokens == 3
        and protocol.action_block_raw_steps == 5
        and protocol.action_dim == 2
    ):
        raise RuntimeError(f"Unexpected PLDM adapter protocol: {protocol}")
    return {"metadata": adapter.metadata, "model_state_sha256": state, "protocol": vars(protocol)}


def _stage_speed_initialization(config: dict[str, Any]) -> dict[str, Any]:
    """Materialize a raw-state ``.pt + config.json`` view of legacy weights."""

    init = config["initialization"]
    source = Path(init["checkpoint"])
    raw_target = Path(init["staged_checkpoint"])
    target = (
        raw_target.resolve()
        if raw_target.is_absolute()
        else (ROOT / raw_target).resolve()
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    import torch

    source_payload = torch.load(source, map_location="cpu", weights_only=False)
    source_state = (
        source_payload.get("state_dict", source_payload)
        if isinstance(source_payload, dict)
        else source_payload
    )
    if not isinstance(source_state, dict):
        raise RuntimeError("Legacy PLDM checkpoint has no state dictionary")
    raw_state = {
        name.removeprefix("model."): tensor
        for name, tensor in source_state.items()
        if name.startswith("model.")
    }
    if not raw_state:
        raise RuntimeError("Legacy PLDM checkpoint has no model.* state tensors")
    expected_state = init["expected_model_state_sha256"]
    observed_state = _state_mapping_sha256(raw_state)
    if observed_state != expected_state:
        raise RuntimeError(
            "Legacy checkpoint raw model state differs from strict-load preregistration"
        )
    replace = True
    if target.exists():
        try:
            existing = torch.load(target, map_location="cpu", weights_only=False)
            replace = _state_mapping_sha256(existing) != expected_state
        except Exception:
            replace = True
    if replace:
        temporary = target.with_suffix(".tmp")
        torch.save(raw_state, temporary)
        os.replace(temporary, target)

    # StableWM's native loader needs a resolved Hydra model mapping.  It is
    # generated only from the pinned PLDM YAML; the legacy YAML remains part
    # of the immutable source identity checked above.
    from omegaconf import OmegaConf, open_dict

    runtime = config["runtime"]["stable_worldmodel"]
    source_cfg = OmegaConf.load(Path(runtime["pldm_config"]))
    with open_dict(source_cfg):
        source_cfg.model.action_encoder.input_dim = 10
        source_cfg.idm.input_dim = 384
        source_cfg.idm.output_dim = 10
    model = OmegaConf.to_container(source_cfg.model, resolve=True)
    model_config = target.parent / "config.json"
    rendered = json.dumps({"model": model}, indent=2, sort_keys=True) + "\n"
    if model_config.exists():
        if model_config.read_text(encoding="utf-8") != rendered:
            raise RuntimeError("Staged initialization config already exists with different bytes")
    else:
        model_config.write_text(rendered, encoding="utf-8")
    checkpoint_sha256 = _sha256(target)
    model_config_sha256 = _sha256(model_config)
    # These values are frozen in the preregistration only after the zero-step
    # staging pass has materialized the deterministic raw-state representation.
    # Never allow a later run to silently accept different staged bytes.
    if checkpoint_sha256 != str(init["staged_checkpoint_sha256"]):
        raise RuntimeError("Staged raw-state checkpoint differs from preregistration")
    if model_config_sha256 != str(init["staged_config_sha256"]):
        raise RuntimeError("Staged initialization config differs from preregistration")
    return {
        "checkpoint": str(target),
        "checkpoint_sha256": checkpoint_sha256,
        "model_state_sha256": observed_state,
        "model_config": str(model_config),
        "model_config_sha256": model_config_sha256,
        "source_checkpoint": str(source),
        "source_checkpoint_sha256": str(init["checkpoint_sha256"]),
    }


def preflight(component: str, config: dict[str, Any], *, seed: int) -> dict[str, Any]:
    training = config["training"]
    if int(seed) not in {int(value) for value in training["seeds"]}:
        raise ValueError(f"Seed {seed} is not registered for {component}")
    runtime = config["runtime"]["stable_worldmodel"]
    if _git_head(Path(runtime["repo"])) != runtime["commit"]:
        raise RuntimeError("Pinned Stable-WorldModel worktree drifted")
    _require_hash(Path(runtime["pldm_config"]), runtime["pldm_config_sha256"], "pinned PLDM config")
    init = config["initialization"]
    _require_hash(Path(init["checkpoint"]), init["checkpoint_sha256"], "original PLDM checkpoint")
    _require_hash(Path(init["source_config"]), init["source_config_sha256"], "original PLDM config")
    data = config["data"]
    action_training_inputs = None
    if component == "speed":
        _require_hash(_resolved(data["normalizer"]["path"]), data["normalizer"]["sha256"], "frozen normalizer")
        for name in ("catalog", "manifest", "report"):
            _require_hash(_resolved(data["speed_multi"][name]), data["speed_multi"][f"{name}_sha256"], f"speed {name}")
        if not Path(data["speed_multi"]["data_root"]).is_dir():
            raise FileNotFoundError(data["speed_multi"]["data_root"])
        staged = _stage_speed_initialization(config)
    else:
        training_root = _resolved(data["training_root"])
        public_root = _resolved(data["public_root"])
        for path in (
            Path(data["original_h5"]),
            Path(data["original_lance"]),
            training_root,
            public_root,
        ):
            if not path.exists():
                raise FileNotFoundError(path)
        _require_hash(training_root / "manifest.json", data["training_manifest_sha256"], "ActionStrength training manifest")
        _require_hash(public_root / "manifest.json", data["public_manifest_sha256"], "ActionStrength public manifest")
        train_lance = training_root / "train.lance"
        legacy_eval = training_root / "eval_payloads"
        validation_lance = training_root / "validation.lance"
        if not train_lance.is_dir():
            raise FileNotFoundError(train_lance)
        if legacy_eval.is_dir():
            evaluation_input = legacy_eval
            evaluation_source = "legacy_eval_payloads"
        elif validation_lance.is_dir():
            evaluation_input = validation_lance
            evaluation_source = "development_validation_lance"
        else:
            raise FileNotFoundError(
                "ActionStrength training root has neither legacy eval_payloads "
                f"nor development validation.lance: {training_root}"
            )
        action_training_inputs = {
            "root": str(training_root),
            "train_lance": str(train_lance),
            "evaluation_input": str(evaluation_input),
            "evaluation_source": evaluation_source,
            "public_root_not_used_for_training": str(public_root),
        }
        staged = None
    strict = _strict_adapter(component, config)
    root = _completion_root(config)
    receipt = {
        "schema_version": 1,
        "completion_id": config["completion_id"],
        "status": "passed",
        "seed": int(seed),
        "training_started": False,
        "runtime": {"repo": runtime["repo"], "commit": runtime["commit"]},
        "strict_load": strict,
        "staged_initialization": staged,
        "action_training_inputs": action_training_inputs,
    }
    write_json(root / "preflight" / f"seed_{seed}.json", receipt)
    return receipt


def _import_script(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _run_speed(config: dict[str, Any], seed: int) -> None:
    root = _completion_root(config)
    staged = _stage_speed_initialization(config)
    run_name = f"speed_pldm_reference_completion_v1_s{seed}"
    output_root = root / "training"
    report = root / f"seed_{seed}" / "training_report.json"
    train = _import_script(ROOT / "scripts/train_tworoom_step1.py", "_completion_speed_train")
    train.FORMAL_TOPOLOGIES[(1, 128, 8)] = "1gpu_x_b128_x_accum8"
    config_path = CONFIGS["speed"]
    argv = [
        str(train.__file__), "--model-id", config["training"]["model_id"],
        "--benchmark-config", str(config_path), "--run-name", run_name,
        "--profile", "additive", "--run-kind", "confirmation",
        "--resume-policy", "never", "--seed", str(seed),
        "--data-split-seed", "3072", "--stablewm-repo", config["runtime"]["stable_worldmodel"]["repo"],
        "--stablewm-ref", config["runtime"]["stable_worldmodel"]["commit"],
        "--original-h5", config["data"]["original_h5"], "--output-root", str(output_root),
        "--report", str(report), "--devices", "1", "--batch-size", "128",
        "--accumulate-grad-batches", "8", "--num-workers", "6", "--logger-backend", "none",
        "--initialization-checkpoint", staged["checkpoint"],
        "--initialization-checkpoint-sha256", staged["checkpoint_sha256"],
    ]
    previous = sys.argv
    try:
        sys.argv = argv
        result = train.run(train.parse_args())
    finally:
        sys.argv = previous
    if result.get("passed") is not True:
        raise RuntimeError("Speed PLDM training did not pass")


def _run_action_strength(config: dict[str, Any], seed: int) -> None:
    runtime = config["runtime"]["stable_worldmodel"]
    pinned = str(Path(runtime["repo"]).resolve())
    sys.path.insert(0, pinned)
    import stable_worldmodel  # noqa: F401  # pin before legacy scripts alter sys.path

    scripts_dir = str(ROOT / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    import run_pusht_hidden_actuation_pilot as pilot
    import run_pusht_hidden_actuation_mixed as mixed

    pilot.STABLE_WORLD_MODEL_ROOT = Path(pinned)
    mixed.STABLE_WORLD_MODEL_ROOT = Path(pinned)
    root = _completion_root(config)
    output = root / f"seed_{seed}" / "training"
    data = config["data"]
    argv = [
        str(mixed.__file__), "--hidden-data-root", str(_resolved(data["training_root"])),
        "--original-lance", data["original_lance"], "--action-normalizer-source", data["original_h5"],
        "--checkpoint", config["initialization"]["checkpoint"], "--output", str(output),
        "--variants", config["training"]["recipe"], "--max-steps", str(config["training"]["optimizer_steps"]),
        "--seed", str(seed), "--batch-size", str(config["training"]["batch_size"]),
        "--original-batch-size", str(config["training"]["original_batch_size"]),
        "--device", "cuda:0", "--num-workers", "6",
    ]
    previous = sys.argv
    try:
        sys.argv = argv
        mixed.main()
    finally:
        sys.argv = previous
    report = output / "mixed_report.json"
    if not report.is_file():
        raise RuntimeError("ActionStrength PLDM runner did not write mixed_report.json")


def _action_strength_development_gate(
    config: dict[str, Any],
    *,
    seed: int,
) -> dict[str, Any]:
    """Freeze one completed ActionStrength checkpoint before formal scoring.

    This is deliberately a development-only integrity gate.  It verifies the
    final fixed-step artifact and its local validation-lance diagnostic, but
    does not read the Public ICL tree or execute any CEM evaluation.
    """

    root = _completion_root(config)
    training_root = root / f"seed_{seed}" / "training"
    report_path = training_root / "mixed_report.json"
    if not report_path.is_file():
        raise FileNotFoundError(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    results = report.get("results")
    if not isinstance(results, list) or len(results) != 1:
        raise RuntimeError("Expected exactly one completed mixed PLDM result")
    result = results[0]
    if (
        result.get("variant") != config["training"]["recipe"]
        or int(result.get("optimizer_steps", -1))
        != int(config["training"]["optimizer_steps"])
    ):
        raise RuntimeError("Completed ActionStrength run differs from preregistered recipe")
    final = result.get("final_checkpoint")
    if not isinstance(final, dict):
        raise RuntimeError("Mixed report has no final checkpoint receipt")
    checkpoint = Path(str(final["path"])).resolve()
    checkpoint_sha = _require_hash(
        checkpoint,
        str(final["sha256"]),
        "ActionStrength final checkpoint",
    )
    model_config = checkpoint.parent / "config.json"
    if not model_config.is_file():
        raise FileNotFoundError(model_config)
    from contextworld.benchmarks.adapters import (
        StableWorldModelPLDMActionStrengthAdapter,
    )

    runtime = config["runtime"]["stable_worldmodel"]
    adapter = StableWorldModelPLDMActionStrengthAdapter.from_checkpoint(
        checkpoint,
        action_mean=[-0.007812564261257648, 0.006860687397420406],
        action_std=[0.20846743881702423, 0.2067486196756363],
        repo_root=ROOT,
        stablewm_repo=str(runtime["repo"]),
        stablewm_ref=str(runtime["commit"]),
        device="cpu",
    )
    observed_state = adapter.frozen_state_hash()
    if observed_state != str(final["model_state_sha256"]):
        raise RuntimeError("Final checkpoint strict-load state hash differs from training receipt")
    hidden_data = report.get("provenance", {}).get("hidden_data", {})
    if (
        hidden_data.get("evaluation_source")
        != "development_validation_lance"
        or hidden_data.get("public_test_used") is not False
    ):
        raise RuntimeError("Development gate detected Public data use or wrong evaluation source")
    return {
        "schema_version": 1,
        "completion_id": config["completion_id"],
        "seed": int(seed),
        "status": "passed_development_only_checkpoint_gate",
        "formal_icl_or_cem_executed": False,
        "public_test_used": False,
        "training_report": {
            "path": str(report_path),
            "sha256": _sha256(report_path),
        },
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": checkpoint_sha,
            "model_state_sha256": observed_state,
            "config": str(model_config),
            "config_sha256": _sha256(model_config),
        },
        "strict_load": {
            "metadata": adapter.metadata,
            "protocol": vars(adapter.protocol),
        },
        "development_data": {
            "evaluation_source": hidden_data["evaluation_source"],
            "evaluation_pairs": int(hidden_data["eval_pairs"]),
            "training_manifest_sha256": hidden_data["manifest_sha256"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--component", choices=tuple(CONFIGS), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--mode",
        choices=("preflight", "train", "development-gate"),
        required=True,
    )
    args = parser.parse_args()
    config = _load(CONFIGS[args.component])
    if args.mode == "development-gate":
        if args.component != "action-strength":
            raise ValueError("Speed development gate is available after its final checkpoint")
        gate = _action_strength_development_gate(config, seed=args.seed)
        write_json(
            _completion_root(config) / "development_gate" / f"seed_{args.seed}.json",
            gate,
        )
        print(json.dumps(gate, ensure_ascii=False, sort_keys=True))
        return
    receipt = preflight(args.component, config, seed=args.seed)
    if args.mode == "train":
        receipt["training_started"] = True
        receipt["training_completed"] = False
        receipt["process_id"] = os.getpid()
        receipt_path = (
            _completion_root(config) / "preflight" / f"seed_{args.seed}.json"
        )
        write_json(receipt_path, receipt)
        try:
            if args.component == "speed":
                _run_speed(config, args.seed)
            else:
                _run_action_strength(config, args.seed)
        except Exception as error:
            receipt["training_failed"] = True
            receipt["failure"] = {
                "type": type(error).__name__,
                "message": str(error),
            }
            write_json(receipt_path, receipt)
            raise
        receipt["training_completed"] = True
        write_json(receipt_path, receipt)
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
