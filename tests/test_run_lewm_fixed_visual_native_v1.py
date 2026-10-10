"""Focused CPU checks for the standalone native LeWM control."""
import importlib.util
from pathlib import Path

import torch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_lewm_fixed_visual_native_v1.py"
SPEC = importlib.util.spec_from_file_location("fixed_visual_native", SCRIPT)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class ToyLeWM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.BatchNorm1d(2)
        self.projector = torch.nn.BatchNorm1d(2)
        self.action_encoder = torch.nn.Linear(2, 2)
        self.predictor = torch.nn.Linear(2, 2)
        self.pred_proj = torch.nn.Linear(2, 2)


def test_full_t0_load_and_visual_freeze(tmp_path):
    source = ToyLeWM()
    with torch.no_grad():
        source.projector.running_mean.fill_(3)
        source.predictor.weight.fill_(7)
    checkpoint = tmp_path / "t0.ckpt"
    torch.save({"state_dict": {f"model.{key}": value for key, value in source.state_dict().items()},
                "optimizer_states": [{"ignored": True}]}, checkpoint)
    target = ToyLeWM()
    frozen, hashes = RUNNER.initialize_and_freeze(target, checkpoint, "visual")
    assert all(torch.equal(target.state_dict()[key], value)
               for key, value in source.state_dict().items())
    assert set(frozen) == {"encoder", "projector"}
    assert all(not p.requires_grad for module in frozen.values() for p in module.parameters())
    assert all(not module.training for module in frozen.values())
    assert all(p.requires_grad for name in ("action_encoder", "predictor", "pred_proj")
               for p in getattr(target, name).parameters())
    target.train()
    for module in frozen.values():
        module.eval()
    before = {name: RUNNER.tensor_hash(module) for name, module in frozen.items()}
    x = torch.randn(8, 2)
    loss = (target.predictor(target.projector(target.encoder(x))) + target.action_encoder(x) + target.pred_proj(x)).square().mean()
    loss.backward()
    torch.optim.SGD((p for p in target.parameters() if p.requires_grad), lr=0.1).step()
    assert {name: RUNNER.tensor_hash(module) for name, module in frozen.items()} == before == hashes
    assert target.predictor.weight.grad is not None


def test_old_prefix_inside_dataset_uri_is_mapped():
    from base64 import urlsafe_b64encode
    import json
    payload = {"root": "/opt/huawei/dataset/ag_data/data/world_model/release",
               "original_dataset": "/opt/huawei/dataset/ag_data/data/source.h5"}
    uri = RUNNER.URI + urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    rewritten = RUNNER.mapped(uri)
    from base64 import urlsafe_b64decode
    encoded = rewritten[len(RUNNER.URI):]
    decoded = json.loads(urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)))
    assert decoded["root"].startswith(RUNNER.LOCAL)
    assert decoded["original_dataset"].startswith(RUNNER.LOCAL)
