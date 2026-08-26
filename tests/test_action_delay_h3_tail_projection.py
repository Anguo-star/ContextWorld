from __future__ import annotations

import numpy as np
import pytest
import yaml

from contextworld.benchmarks import action_delay_original_baseline_recovery_cli
from contextworld.benchmarks.action_delay_h3_tail_projection import (
    H3TailProjectionActionDelayAdapter,
)
from contextworld.benchmarks.adapters import AdapterProtocol
from contextworld.evaluation.action_delay_h7_validation import file_sha256
from contextworld.paths import repository_root


class _NativeH3Adapter:
    def __init__(self) -> None:
        self.protocol = AdapterProtocol(
            history_tokens=3,
            action_block_raw_steps=5,
            action_dim=2,
            future_action_blocks=3,
        )
        self.metadata = {
            "adapter_id": "stable_worldmodel_lewm_v1",
            "checkpoint_sha256": "a" * 64,
            "stable_worldmodel_commit": "b" * 40,
        }
        self.received: tuple[np.ndarray, np.ndarray, int] | None = None

    def rollout_latents(
        self,
        pixels: np.ndarray,
        actions: np.ndarray,
        *,
        batch_size: int,
    ) -> np.ndarray:
        self.received = (pixels.copy(), actions.copy(), batch_size)
        return np.zeros((len(pixels), 3, 4), dtype=np.float32)

    def encode_pixels(self, pixels: np.ndarray, *, batch_size: int) -> np.ndarray:
        return np.asarray(pixels)[:, 0, 0, :1]

    def frozen_state_hash(self) -> str:
        return "frozen-native-h3"


def test_h3_tail_projection_slices_latest_aligned_context_and_actions() -> None:
    base = _NativeH3Adapter()
    adapter = H3TailProjectionActionDelayAdapter(base)
    pixels = np.arange(2 * 7 * 1 * 1 * 3, dtype=np.uint8).reshape(2, 7, 1, 1, 3)
    actions = np.arange(2 * 9 * 5 * 2, dtype=np.float32).reshape(2, 9, 5, 2)

    predicted = adapter.rollout_latents(pixels, actions, batch_size=11)

    assert predicted.shape == (2, 3, 4)
    assert base.received is not None
    received_pixels, received_actions, batch_size = base.received
    assert batch_size == 11
    assert np.array_equal(received_pixels, pixels[:, -3:])
    assert np.array_equal(received_actions, actions[:, -5:])
    assert adapter.protocol == AdapterProtocol(
        history_tokens=7,
        action_block_raw_steps=5,
        action_dim=2,
        future_action_blocks=3,
    )


def test_h3_tail_projection_discloses_projection_without_weight_change() -> None:
    base = _NativeH3Adapter()
    adapter = H3TailProjectionActionDelayAdapter(base)

    metadata = adapter.metadata

    assert metadata["adapter_id"] == (
        "stable_worldmodel_h3_tail_projection_action_delay_v1"
    )
    assert metadata["checkpoint_sha256"] == "a" * 64
    assert metadata["stable_worldmodel_commit"] == "b" * 40
    assert metadata["history_adapter"] == "h3_tail_projection"
    assert metadata["weights_modified"] is False
    assert metadata["protocol"]["history_tokens"] == 7
    assert metadata["projection"] == {
        "source_history_tokens": 7,
        "native_checkpoint_history_tokens": 3,
        "scorer_future_action_blocks": 3,
        "native_future_action_blocks_requested": 3,
        "input_pixels": "input_pixels[:, -3:]",
        "raw_action_blocks": "raw_action_blocks[:, -5:]",
        "source_action_block_count": 9,
        "projected_action_block_count": 5,
        "projected_context_action_blocks": 2,
        "projected_future_action_blocks": 3,
        "positional_embedding_interpolation": False,
    }
    assert metadata["base_adapter"] == base.metadata
    assert adapter.frozen_state_hash() == "frozen-native-h3"


def test_h3_tail_projection_rejects_unaligned_h7_action_sequence() -> None:
    adapter = H3TailProjectionActionDelayAdapter(_NativeH3Adapter())
    pixels = np.zeros((1, 7, 2, 2, 3), dtype=np.uint8)
    actions = np.zeros((1, 8, 5, 2), dtype=np.float32)

    with pytest.raises(ValueError, match=r"expects \[B,9,5,A\] action blocks"):
        adapter.rollout_latents(pixels, actions, batch_size=1)


def test_recovery_cli_defaults_to_native_h7_and_tail_projection_is_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class NativeLeWM:
        @classmethod
        def from_checkpoint(cls, *args, **kwargs):
            del args, kwargs
            calls.append("native_h7")
            return "native_h7_adapter"

    class NativePLDM:
        @classmethod
        def from_checkpoint(cls, *args, **kwargs):
            del args, kwargs
            raise AssertionError("PLDM should not be selected in this test")

    class NativeH3LeWM:
        @classmethod
        def from_checkpoint(cls, *args, **kwargs):
            del args, kwargs
            calls.append("native_h3")
            return "native_h3_adapter"

    class NativeH3PLDM:
        @classmethod
        def from_checkpoint(cls, *args, **kwargs):
            del args, kwargs
            raise AssertionError("PLDM should not be selected in this test")

    monkeypatch.setattr(
        action_delay_original_baseline_recovery_cli,
        "StableWorldModelLeWMHistory7Adapter",
        NativeLeWM,
    )
    monkeypatch.setattr(
        action_delay_original_baseline_recovery_cli,
        "StableWorldModelPLDMHistory7Adapter",
        NativePLDM,
    )
    monkeypatch.setattr(
        action_delay_original_baseline_recovery_cli,
        "StableWorldModelLeWMAdapter",
        NativeH3LeWM,
    )
    monkeypatch.setattr(
        action_delay_original_baseline_recovery_cli,
        "StableWorldModelPLDMAdapter",
        NativeH3PLDM,
    )
    monkeypatch.setattr(
        action_delay_original_baseline_recovery_cli,
        "H3TailProjectionActionDelayAdapter",
        lambda base: ("tail_projection", base),
    )
    monkeypatch.setattr(
        action_delay_original_baseline_recovery_cli,
        "load_action_delay_icl_release",
        lambda *args, **kwargs: {
            "evaluation": {"normalizer": "normalizer.json"},
            "runtime": {
                "stable_worldmodel": {"repo": "stable-worldmodel", "expected_ref": "ref"}
            },
        },
    )
    monkeypatch.setattr(
        action_delay_original_baseline_recovery_cli,
        "resolve_contextworld_path",
        lambda *args, **kwargs: "normalizer.json",
    )
    common = [
        "eval",
        "--checkpoint",
        "checkpoint.pt",
        "--adapter",
        "lewm",
        "--model-name",
        "baseline",
        "--output",
        "result.json",
    ]

    default_args = action_delay_original_baseline_recovery_cli.parse_args(common)
    projected_args = action_delay_original_baseline_recovery_cli.parse_args(
        [*common, "--history-adapter", "h3_tail_projection"]
    )

    assert default_args.history_adapter == "native_h7"
    assert (
        action_delay_original_baseline_recovery_cli._adapter(default_args)
        == "native_h7_adapter"
    )
    assert action_delay_original_baseline_recovery_cli._adapter(projected_args) == (
        "tail_projection",
        "native_h3_adapter",
    )
    assert calls == ["native_h7", "native_h3"]


def test_original_cli_stays_frozen_and_release_change_is_additively_scoped() -> None:
    root = repository_root()

    assert file_sha256(root / "contextworld/benchmarks/action_delay_icl_cli.py") == (
        "5c73e71a36bcebdb04840c5e4929881573d7f3478b9fc543acf2bbaf73d392f1"
    )
    release_path = root / "configs/benchmark/tworoom_action_delay_icl_release_v1.yaml"
    historical = yaml.safe_load(
        (
            root
            / "configs/benchmark/contextworld_action_delay_original_baseline_recovery_prereg_v1.yaml"
        ).read_text(encoding="utf-8")
    )
    historical_release = historical["release_bindings"]["original_release"]
    assert historical_release["sha256"] == (
        "303d27e9163ce435fdceff11d3d2ef2e6f7a99d61b1bf3c2f9737122611e47bb"
    )
    assert file_sha256(release_path) != historical_release["sha256"]

    amendment = yaml.safe_load(
        (
            root
            / "configs/benchmark/contextworld_icl_suite_v2_engineering_identity_amendment_prereg_v1.yaml"
        ).read_text(encoding="utf-8")
    )["engineering_identity_amendment"]
    update = amendment["approved_component_identity_updates"]["action_delay"]
    assert update["release_config"] == (
        "configs/benchmark/tworoom_action_delay_icl_release_v1.yaml"
    )
    release = yaml.safe_load(release_path.read_text(encoding="utf-8"))
    correction = yaml.safe_load(
        (
            root
            / "configs/benchmark/contextworld_historical_package_pin_correction_v1.yaml"
        ).read_text(encoding="utf-8")
    )
    package_row = next(
        row
        for row in correction["affected_records"]
        if row["config"]["path"]
        == "configs/benchmark/tworoom_action_delay_icl_release_v1.yaml"
    )
    assert package_row["field"] == "identity.package.sha256"
    assert package_row["config"] == {
        "path": "configs/benchmark/tworoom_action_delay_icl_release_v1.yaml",
        "sha256": file_sha256(release_path),
        "size_bytes": release_path.stat().st_size,
    }
    for identity in release["identity"].values():
        if identity["path"] == "pyproject.toml":
            assert identity["sha256"] == correction["finding"]["invalid_sha256"]
            assert correction["finding"]["role_after_correction"] == (
                "historical_packaging_metadata_not_runtime_source"
            )
            continue
        assert file_sha256(root / identity["path"]) == identity["sha256"]
