"""Original weights retain their input adaptation and frozen-state boundary."""
from types import SimpleNamespace

import numpy as np
import pytest

from contextworld.benchmarks.prejepa_original_context_adapter import (
    METHOD_ID, OriginalDINOFixedContextAdapter, _TASK_CLASSES,
)
from contextworld.benchmarks import prejepa_adapters
from test_prejepa_adapters import _adapter, _fake_model
from scripts.freeze_dinowm_original_icl import DESTINATION, frozen_state_digest
from scripts.eval_dinowm_original_icl_current import SOURCE, read


def test_fixed_context_wrapper_preserves_zeros_and_frozen_weights():
    model = _fake_model(state=True)
    base = _adapter(model, cls=prejepa_adapters.StableWorldModelPreJEPADiagnosticAdapter)
    adapter = OriginalDINOFixedContextAdapter(base, task="speed")
    before = adapter.frozen_state_hash()
    adapter.rollout_latents(np.zeros((2, 3, 8, 8, 3), dtype=np.uint8),
                           np.zeros((2, 5, 5, 2), dtype=np.float32), batch_size=2)
    assert adapter.frozen_state_hash() == before
    assert model.rollout_calls[0][0]["proprio"].count_nonzero() == 0
    assert adapter.metadata["diagnostic"] is True
    assert adapter.metadata["frozen_v1_compatible"] is False
    assert adapter.metadata["base_adapter"] == base.metadata
    assert adapter.metadata["inference_variant"] == METHOD_ID
    assert adapter.metadata["checkpoint_sha256"] == base.metadata["checkpoint_sha256"]
    assert adapter.protocol == base.protocol


@pytest.mark.parametrize("task", [*_TASK_CLASSES, "action_delay"])
def test_task_request_reaches_the_existing_adapter(monkeypatch, task):
    if task == "action_delay":
        from contextworld.benchmarks.action_delay_development_adapter import (
            StableWorldModelPreJEPADiagnosticActionDelayH3TailAdapter as cls,
        )
    else:
        cls = getattr(prejepa_adapters, _TASK_CLASSES[task])
    request = SimpleNamespace(task=task)
    seen = []
    base = object()
    def load(cls, incoming):
        seen.append(incoming)
        return base
    monkeypatch.setattr(cls, "from_contextworld_request", classmethod(load))
    adapter = OriginalDINOFixedContextAdapter.from_contextworld_request(request)
    assert seen == [request]
    assert adapter.base is base


@pytest.mark.parametrize("model", [{}, {"state_hash_before": "a", "state_hash_after": "b"}])
def test_supplement_rejects_missing_or_changed_model_state(model):
    with pytest.raises(ValueError, match="state"):
        frozen_state_digest({"model": model})


def test_supplement_covers_original_cem_checkpoints_and_only_reportable_splits():
    supplement = read(DESTINATION)
    source = read(SOURCE)
    rows = supplement["checkpoint_results"]
    assert len(rows) == 27
    assert len({(r["component_id"], r["training_seed"]) for r in rows}) == 27
    assert {r["checkpoint_sha256"] for r in rows} == set(source["checkpoints"].values())
    assert sum("development" in r for r in rows) == 27
    assert sum("test" in r for r in rows) == 21
    for row in rows:
        assert ("test" in row) == (row["component_id"] not in {"contact_friction", "motion_damping"})
    assert supplement["claim_boundary"]["diagnostic"] is True
    assert supplement["claim_boundary"]["official_scoreboard_row"] is False
