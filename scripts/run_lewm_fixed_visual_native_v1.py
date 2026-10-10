#!/usr/bin/env python3
"""Run a T0-start LeWM control through the pinned native trainer.

The only training intervention is the requested frozen module scope.  The
saved T3 config, native loss, loader, optimizer and scheduler remain in force.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import importlib
import json
import os
from pathlib import Path
import sys

OLD = "/opt/huawei/dataset/ag_data"
LOCAL = "/opt/huawei/explorer-env/dataset/ag_data"
URI = "contextworld://v1/"


def mapped(value):
    if isinstance(value, str):
        if value.startswith(URI):
            raw = value[len(URI):]
            data = json.loads(base64.urlsafe_b64decode(raw + "=" * (-len(raw) % 4)))
            data = mapped(data)
            return URI + base64.urlsafe_b64encode(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).decode().rstrip("=")
        return LOCAL + value[len(OLD):] if value.startswith(OLD + "/") else value
    if isinstance(value, list):
        return [mapped(item) for item in value]
    if isinstance(value, dict):
        return {key: mapped(item) for key, item in value.items()}
    return value


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def prepare_optional_flash_attention():
    """Mask only an installed optional extension with an incompatible ABI."""
    try:
        importlib.import_module("flash_attn.modules.mha")
    except ModuleNotFoundError:
        pass
    except (ImportError, OSError):
        for name in tuple(sys.modules):
            if name == "flash_attn" or name.startswith("flash_attn."):
                sys.modules.pop(name, None)
        sys.modules["flash_attn"] = None


def tensor_hash(module):
    h = hashlib.sha256()
    count = 0
    for name, tensor in sorted(module.state_dict().items()):
        cpu = tensor.detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str(cpu.dtype).encode())
        h.update(str(tuple(cpu.shape)).encode())
        h.update(cpu.numpy().tobytes())
        count += 1
    return {"sha256": h.hexdigest(), "tensors": count}


def initialize_and_freeze(model, checkpoint, scope):
    import torch
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = state.get("state_dict", state)
    target = model.state_dict()
    source = {name.removeprefix("model."): value for name, value in state.items()
              if name.startswith("model.")}
    if set(source) != set(target):
        raise RuntimeError(f"T0 state mismatch: missing={sorted(set(target)-set(source))[:8]}, extra={sorted(set(source)-set(target))[:8]}")
    model.load_state_dict(source, strict=True)
    names = ("encoder",) if scope == "encoder" else ("encoder", "projector")
    frozen = {name: getattr(model, name) for name in names}
    for module in frozen.values():
        module.requires_grad_(False)
        module.eval()
    for name in ("action_encoder", "predictor", "pred_proj"):
        module = getattr(model, name)
        parameters = list(module.parameters())
        if not parameters or not all(p.requires_grad for p in parameters):
            raise RuntimeError(f"{name} must remain fully trainable")
    return frozen, {name: tensor_hash(module) for name, module in frozen.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--stablewm-repo", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--freeze-scope", choices=("encoder", "visual"), required=True)
    parser.add_argument("--smoke-max-steps", type=int)
    parser.add_argument("--devices", type=int)
    args = parser.parse_args()
    if not args.run_name or "/" in args.run_name or args.run_name in (".", ".."):
        parser.error("--run-name must be a single directory name")
    reference = args.reference_run.resolve()
    repo = args.stablewm_repo.resolve()
    output = args.output_root.resolve()
    run_dir = output / "checkpoints" / args.run_name
    config_file = reference / "config.json"
    identity_file = reference / "contextworld_training_identity_v1.json"
    identity_record = json.loads(identity_file.read_text())
    identity = identity_record["identity"]
    t0 = Path(mapped(identity["initialization"]["checkpoint"]))
    if not t0.is_file() or not (repo / "scripts/train/lewm.py").is_file():
        raise RuntimeError("T0 checkpoint or pinned native LeWM entry is missing")
    if sha256(t0) != identity["initialization"]["sha256"]:
        raise RuntimeError("T0 checkpoint SHA differs from T3 identity")
    if identity["family"] != "lewm":
        raise RuntimeError("reference run is not LeWM")
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    if "LOCAL_RANK" not in os.environ:
        if run_dir.exists():
            raise FileExistsError(run_dir)
        run_dir.mkdir(parents=True)
    if any(run_dir.glob("*.ckpt")):
        raise RuntimeError("new control output unexpectedly contains a resume checkpoint")
    os.environ["STABLEWM_HOME"] = str(output)
    os.environ["CONTEXTWORLD_STABLEWM_BUNDLE"] = "1"
    os.environ["CW_FIXED_VISUAL_BOOTSTRAP"] = "1"
    contextworld_root = Path(__file__).resolve().parents[1]
    bootstrap = contextworld_root / "scripts/fixed_visual_bootstrap"
    os.environ["PYTHONPATH"] = os.pathsep.join((str(bootstrap), str(repo), str(contextworld_root), os.environ.get("PYTHONPATH", "")))
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    sys.path.insert(0, str(repo))
    sys.path.insert(1, str(contextworld_root))
    prepare_optional_flash_attention()
    from contextworld.training.stablewm_bundle import register_stablewm_bundle_format
    register_stablewm_bundle_format()
    import torch
    import stable_pretraining as spt
    from lightning.pytorch.callbacks import Callback
    from lightning.pytorch.loggers import CSVLogger
    from omegaconf import OmegaConf
    cfg = OmegaConf.create(mapped(json.loads(config_file.read_text())))
    cfg.subdir = args.run_name
    cfg.output_model_name = args.run_name
    cfg.logger_backend = "csv"
    if cfg.get("swanlab") is not None:
        cfg.swanlab.enabled = False
    if cfg.get("wandb") is not None:
        cfg.wandb.enabled = False
    if args.devices is not None:
        cfg.trainer.devices = args.devices
    if args.smoke_max_steps is not None:
        if args.smoke_max_steps <= 0:
            raise ValueError("smoke steps must be positive")
        cfg.trainer.max_steps = args.smoke_max_steps
        cfg.trainer.limit_val_batches = 1
    if rank == 0:
        recipe = {"reference_run": str(reference), "native_config_source": str(config_file),
                  "native_config_sha256": sha256(config_file), "identity_source": str(identity_file),
                  "identity_file_sha256": sha256(identity_file),
                  "identity_sha256": identity_record.get("identity_sha256"),
                  "stablewm_repo": str(repo), "native_entry": str(repo / "scripts/train/lewm.py"),
                  "native_entry_sha256": sha256(repo / "scripts/train/lewm.py"),
                  "runner_sha256": sha256(Path(__file__).resolve()),
                  "full_t0_checkpoint": str(t0), "full_t0_sha256": sha256(t0),
                  "freeze_scope": args.freeze_scope, "frozen_modules": ["encoder"] if args.freeze_scope == "encoder" else ["encoder", "projector"],
                  "changes_from_T3": ["T0 visual freeze scope", "local CSV logger", "new output path"] + (["smoke max_steps"] if args.smoke_max_steps else []) + (["device count"] if args.devices is not None else []),
                  "smoke_max_steps": args.smoke_max_steps, "devices_override": args.devices}
        (run_dir / "recipe.json").write_text(json.dumps(recipe, indent=2) + "\n")
    spec = importlib.util.spec_from_file_location("cw_native_lewm_fixed_visual", repo / "scripts/train/lewm.py")
    native = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = native
    spec.loader.exec_module(native)
    native.build_logger = lambda cfg: CSVLogger(str(run_dir / "csv"), name="native", version="0")
    original_loaders = native.build_data_loaders
    def loaders(*a, **kw):
        train, val = original_loaders(*a, **kw)
        if rank == 0:
            (run_dir / "loader_constructed.json").write_text(json.dumps({"train_loader_len_before_trainer": len(train), "val_loader_len_before_trainer": len(val), "native_scheduler_max_steps": len(train) * int(cfg.trainer.max_epochs), "batch_size_per_rank": int(cfg.loader.batch_size)}, indent=2) + "\n")
        return train, val
    native.build_data_loaders = loaders
    original_manager = spt.Manager
    class FrozenManager(original_manager):
        def __init__(self, *a, **kw):
            module = kw["module"]
            trainer = kw["trainer"]
            frozen, initial = initialize_and_freeze(module.model, t0, args.freeze_scope)
            callback = FrozenAudit(frozen, initial, run_dir)
            trainer.callbacks.append(callback)
            if rank == 0:
                (run_dir / "initial_frozen_hashes.json").write_text(json.dumps(initial, indent=2) + "\n")
            super().__init__(*a, **kw)
    class FrozenAudit(Callback):
        def __init__(self, modules, initial, directory):
            self.modules, self.initial, self.directory = modules, initial, directory
        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
            for module in self.modules.values():
                module.eval()
        def on_train_start(self, trainer, pl_module):
            if trainer.is_global_zero:
                receipt = {"world_size": trainer.world_size,
                           "actual_num_training_batches_per_rank": trainer.num_training_batches,
                           "estimated_optimizer_steps": trainer.estimated_stepping_batches,
                           "max_epochs": trainer.max_epochs,
                           "max_steps": trainer.max_steps,
                           "accumulate_grad_batches": trainer.accumulate_grad_batches}
                (self.directory / "loader_actual.json").write_text(json.dumps(receipt, indent=2) + "\n")
        def on_train_epoch_end(self, trainer, pl_module):
            current = {name: tensor_hash(module) for name, module in self.modules.items()}
            if current != self.initial:
                raise RuntimeError(f"frozen tensors/buffers changed: {current}")
            if trainer.is_global_zero:
                epoch = trainer.current_epoch + 1
                receipt = {"epoch": epoch, "global_step": trainer.global_step,
                           "world_size": trainer.world_size, "frozen_hashes": current,
                           "frozen_state_unchanged": True}
                (self.directory / f"epoch_{epoch:02d}_audit.json").write_text(json.dumps(receipt, indent=2) + "\n")
        def on_train_end(self, trainer, pl_module):
            current = {name: tensor_hash(module) for name, module in self.modules.items()}
            if current != self.initial:
                raise RuntimeError("frozen tensors/buffers changed at train end")
            if trainer.is_global_zero:
                receipt = {"global_step": trainer.global_step, "world_size": trainer.world_size,
                           "frozen_hashes": current, "frozen_state_unchanged": True}
                (self.directory / "train_end_audit.json").write_text(json.dumps(receipt, indent=2) + "\n")
    spt.Manager = FrozenManager
    native.run_training(cfg)


if __name__ == "__main__":
    main()
