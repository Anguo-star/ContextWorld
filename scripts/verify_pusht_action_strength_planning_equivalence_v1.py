#!/usr/bin/env python3
"""Prove that the frozen ActionStrength planning loader preserves PLDM math.

The historical planning runner instantiates its compatible ``LeWM`` container
while loading a PLDM state mapping.  Class names alone are not evidence of
equivalence.  This CPU-only verifier uses fixed Loader-Validation inputs and
compares exactly the tensors actually consumed by the frozen planner:
encoder/projector history and goal embeddings, action embedding, and
``predict`` output.  It neither opens nor scores Public Test data.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_pusht_action_strength_pinned_cem as pinned_cem  # noqa: E402


DEFAULT_BINDING = ROOT / "configs/benchmark/pusht_action_strength_pldm_evaluation_binding_v1.yaml"


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _state_sha256(model: Any) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _tensor_summary(value: Any) -> dict[str, Any]:
    tensor = value.detach().cpu().contiguous()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "sha256": hashlib.sha256(tensor.numpy().tobytes()).hexdigest(),
    }


def _module_identity(module: Any) -> dict[str, Any]:
    source_path = Path(sys.modules[type(module).__module__].__file__).resolve()
    return {
        "qualified_type": f"{type(module).__module__}.{type(module).__name__}",
        "class_name": type(module).__name__,
        "source_path": str(source_path),
        "source_sha256": _sha256(source_path),
        "parameter_and_buffer_state_sha256": _state_sha256(module),
    }


def _model_outputs(model: Any, loader: Any, pixels: Any, actions: Any) -> dict[str, Any]:
    import torch

    model.eval()
    with torch.no_grad():
        transformed = loader.preprocess_pixels(pixels, device=torch.device("cpu"))
        batch, frames = transformed.shape[:2]
        encoded = model.encoder(
            transformed.flatten(0, 1), interpolate_pos_encoding=True
        ).last_hidden_state[:, 0]
        projected = model.projector(encoded).reshape(batch, frames, -1)
        action_embedding = model.action_encoder(actions)
        prediction = model.predict(projected[:, :3], action_embedding)
    return {
        "history_embedding": projected[:, :3].float().cpu(),
        "goal_embedding": projected[:, 3:].float().cpu(),
        "action_embedding": action_embedding.float().cpu(),
        "prediction": prediction.float().cpu(),
    }


def _comparison(left: Any, right: Any) -> dict[str, Any]:
    import torch

    delta = (left - right).abs()
    return {
        "left": _tensor_summary(left),
        "right": _tensor_summary(right),
        "exact_equal": bool(torch.equal(left, right)),
        "allclose_rtol_1e-5_atol_1e-6": bool(
            torch.allclose(left, right, rtol=1e-5, atol=1e-6)
        ),
        "max_absolute_difference": float(delta.max().item()),
    }


def _load_binding(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Binding must be a YAML mapping")
    return payload


def verify(binding_path: Path, seed: int) -> dict[str, Any]:
    binding = _load_binding(binding_path)
    matching = [row for row in binding["checkpoints"] if int(row["seed"]) == seed]
    if len(matching) != 1:
        raise ValueError(f"Seed {seed} is not registered exactly once")
    checkpoint = matching[0]["checkpoint"]
    runtime = binding["stable_worldmodel"]
    sources = binding["evaluator_sources"]
    worktree = _resolve(runtime["worktree"])
    wrapper_path = _resolve(sources["pinned_runtime_launcher"]["path"])
    if _sha256(wrapper_path) != sources["pinned_runtime_launcher"]["sha256"]:
        raise RuntimeError("Pinned CEM wrapper changed after binding preregistration")
    runner, injection = pinned_cem._prepare_runtime(
        track="planning",
        stablewm_root=worktree,
        expected_ref=runtime["expected_ref"],
        runner_sha256=sources["planning_runner"]["sha256"],
        dependency_sha256=sources["planning_shared_model_loader"]["sha256"],
    )
    del runner
    loader = sys.modules["eval_pusht_hidden_actuation_cem"]

    import hydra
    import torch
    from omegaconf import OmegaConf, open_dict
    from stable_worldmodel.data import LanceDataset

    # The two containers are evaluated with identical one-thread CPU kernels
    # and deterministic algorithms so a residual is attributable to their
    # graph, not parallel reduction scheduling.  We retain exact and allclose
    # evidence separately; allclose is never the sole structural proof.
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.mkldnn.enabled = False

    configuration = OmegaConf.load(worktree / runtime["pldm_config"])
    with open_dict(configuration):
        configuration.model.action_encoder.input_dim = 10
    pldm = hydra.utils.instantiate(configuration.model)
    raw_checkpoint = _resolve(checkpoint["path"])
    payload = torch.load(raw_checkpoint, map_location="cpu", weights_only=False)
    source_state = payload.get("state_dict", payload)
    state = {
        name[len("model.") :]: value
        for name, value in source_state.items()
        if name.startswith("model.")
    } or dict(source_state)
    pldm.load_state_dict(state, strict=True)
    planning_shell = loader.load_model(raw_checkpoint, torch.device("cpu"))

    completion = _load_binding(_resolve(binding["completion"]["path"]))
    development_root = _resolve(completion["data"]["training_root"])
    dataset = LanceDataset(
        path=development_root / "validation.lance",
        frameskip=5,
        num_steps=4,
        keys_to_load=["pixels", "action"],
    )
    sample_indices = (0, 1)
    rows = [dataset[index] for index in sample_indices]
    pixels = torch.stack([row["pixels"] for row in rows]).contiguous()
    original_h5 = _resolve(completion["data"]["original_h5"])
    mean, std = loader.action_stats(original_h5)
    actions = loader.normalize_action(
        torch.stack([row["action"][:3].float() for row in rows]),
        mean=mean,
        std=std,
    ).contiguous()
    outputs_pldm = _model_outputs(pldm, loader, pixels, actions)
    outputs_shell = _model_outputs(planning_shell, loader, pixels, actions)
    comparisons = {
        name: _comparison(outputs_pldm[name], outputs_shell[name])
        for name in outputs_pldm
    }
    states = {
        "pldm": _state_sha256(pldm),
        "planning_shell": _state_sha256(planning_shell),
    }
    component_names = (
        "encoder",
        "projector",
        "action_encoder",
        "predictor",
        "pred_proj",
    )
    components = {
        name: {
            "pldm": _module_identity(getattr(pldm, name)),
            "planning_shell": _module_identity(getattr(planning_shell, name)),
        }
        for name in component_names
    }
    for row in components.values():
        left, right = row["pldm"], row["planning_shell"]
        row["class_name_matches"] = left["class_name"] == right["class_name"]
        row["source_sha256_matches"] = (
            left["source_sha256"] == right["source_sha256"]
        )
        row["parameter_and_buffer_state_matches"] = (
            left["parameter_and_buffer_state_sha256"]
            == right["parameter_and_buffer_state_sha256"]
        )
    code_equivalence = {
        "predict_method_source_sha256": {
            "pldm": hashlib.sha256(
                inspect.getsource(type(pldm).predict).encode("utf-8")
            ).hexdigest(),
            "planning_shell": hashlib.sha256(
                inspect.getsource(type(planning_shell).predict).encode("utf-8")
            ).hexdigest(),
        },
        "actual_planner_components": components,
        "planner_invokes_direct_predict_path": True,
    }
    code_equivalence["passed"] = bool(
        code_equivalence["predict_method_source_sha256"]["pldm"]
        == code_equivalence["predict_method_source_sha256"]["planning_shell"]
        and all(
            row["class_name_matches"]
            and row["source_sha256_matches"]
            and row["parameter_and_buffer_state_matches"]
            for row in components.values()
        )
    )
    passed = bool(
        states["pldm"] == checkpoint["model_state_sha256"]
        and states["planning_shell"] == checkpoint["model_state_sha256"]
        and code_equivalence["passed"]
        and all(row["allclose_rtol_1e-5_atol_1e-6"] for row in comparisons.values())
    )
    return {
        "schema_version": 1,
        "status": "passed" if passed else "failed",
        "passed": passed,
        "binding": {"path": str(binding_path), "sha256": _sha256(binding_path)},
        "seed": seed,
        "development_input": {
            "root": str(development_root),
            "table": "validation.lance",
            "sample_indices": list(sample_indices),
            "pixels": _tensor_summary(pixels),
            "normalized_actions": _tensor_summary(actions),
            "public_test_used": False,
        },
        "checkpoint": {
            **checkpoint,
            "observed_sha256": _sha256(raw_checkpoint),
        },
        "runtime_injection": injection,
        "models": {
            "pldm_class": f"{type(pldm).__module__}.{type(pldm).__name__}",
            "planning_shell_class": (
                f"{type(planning_shell).__module__}.{type(planning_shell).__name__}"
            ),
            "state_sha256": states,
        },
        "planner_code_equivalence": code_equivalence,
        "planner_tensor_equivalence": comparisons,
        "interpretation": (
            "The planning shell is a frozen runner compatibility container; "
            "its direct predict method and component modules are source-equal, "
            "and all tensors used by its encoder/projector/action_encoder/"
            "predict CEM path are compared on fixed Development inputs."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding", type=Path, default=DEFAULT_BINDING)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    binding_path = _resolve(args.binding)
    output = _resolve(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite equivalence receipt: {output}")
    payload = verify(binding_path, int(args.seed))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"status": payload["status"], "output": str(output)}))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
