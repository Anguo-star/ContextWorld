"""Explicit History-7-to-History-3 inference adapter for Action Delay.

This adapter is deliberately a projection, not a positional-embedding
interpolation scheme.  It lets the frozen History-7 Action Delay scorer call
an original History-3 checkpoint without changing its weights, its native
history setting, or its three-token positional embedding.  The model sees
only the trailing three visible frames and the five action blocks aligned to
them (two context transitions plus three requested futures).
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np

from contextworld.benchmarks.adapters import (
    AdapterProtocol,
    ActionDelayICLModelAdapter,
)


SOURCE_HISTORY_TOKENS = 7
PROJECTED_HISTORY_TOKENS = 3
FUTURE_ACTION_BLOCKS = 3


class H3TailProjectionActionDelayAdapter(ActionDelayICLModelAdapter):
    """Expose a frozen H3 checkpoint through the H7 Action Delay boundary.

    ``protocol`` intentionally reports the scorer-facing H7 contract.  The
    full native H3 contract remains in ``metadata["base_adapter"]`` along
    with an explicit, immutable description of both slices.
    """

    adapter_id = "stable_worldmodel_h3_tail_projection_action_delay_v1"

    def __init__(self, base_adapter: ActionDelayICLModelAdapter) -> None:
        protocol = base_adapter.protocol
        if int(protocol.history_tokens) != PROJECTED_HISTORY_TOKENS:
            raise ValueError(
                "h3_tail_projection requires a native History-3 base "
                f"adapter, got {protocol}"
            )
        if int(protocol.action_block_raw_steps) != 5:
            raise ValueError(
                "h3_tail_projection requires five raw steps per action "
                f"block, got {protocol}"
            )
        if int(protocol.future_action_blocks) < FUTURE_ACTION_BLOCKS:
            raise ValueError(
                "h3_tail_projection requires at least three native future "
                f"action blocks, got {protocol}"
            )
        self._base_adapter = base_adapter
        self._protocol = AdapterProtocol(
            history_tokens=SOURCE_HISTORY_TOKENS,
            action_block_raw_steps=int(protocol.action_block_raw_steps),
            action_dim=int(protocol.action_dim),
            future_action_blocks=FUTURE_ACTION_BLOCKS,
            native_target_encoder=bool(protocol.native_target_encoder),
            decoder_required=bool(protocol.decoder_required),
        )

    @property
    def base_adapter(self) -> ActionDelayICLModelAdapter:
        """The unchanged native H3 adapter, exposed for audit/testing."""

        return self._base_adapter

    @property
    def protocol(self) -> AdapterProtocol:
        return self._protocol

    @property
    def metadata(self) -> dict[str, Any]:
        base_metadata = self._base_adapter.metadata
        action_blocks = (
            PROJECTED_HISTORY_TOKENS - 1 + FUTURE_ACTION_BLOCKS
        )
        return {
            **base_metadata,
            "adapter_id": self.adapter_id,
            "adapter_class": f"{type(self).__module__}.{type(self).__name__}",
            "protocol": asdict(self.protocol),
            "history_adapter": "h3_tail_projection",
            "weights_modified": False,
            "projection": {
                "source_history_tokens": SOURCE_HISTORY_TOKENS,
                "native_checkpoint_history_tokens": PROJECTED_HISTORY_TOKENS,
                "scorer_future_action_blocks": FUTURE_ACTION_BLOCKS,
                "native_future_action_blocks_requested": FUTURE_ACTION_BLOCKS,
                "input_pixels": "input_pixels[:, -3:]",
                "raw_action_blocks": "raw_action_blocks[:, -5:]",
                "source_action_block_count": (
                    SOURCE_HISTORY_TOKENS - 1 + FUTURE_ACTION_BLOCKS
                ),
                "projected_action_block_count": action_blocks,
                "projected_context_action_blocks": (
                    PROJECTED_HISTORY_TOKENS - 1
                ),
                "projected_future_action_blocks": FUTURE_ACTION_BLOCKS,
                "positional_embedding_interpolation": False,
            },
            "base_adapter": base_metadata,
        }

    def _project_inputs(
        self,
        input_pixels: np.ndarray,
        raw_action_blocks: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        pixels = np.asarray(input_pixels)
        actions = np.asarray(raw_action_blocks)
        if pixels.ndim != 5 or pixels.shape[1] != SOURCE_HISTORY_TOKENS:
            raise ValueError(
                "h3_tail_projection expects "
                f"[B,{SOURCE_HISTORY_TOKENS},H,W,3] pixels, got "
                f"{pixels.shape}"
            )
        if pixels.shape[-1] != 3:
            raise ValueError(
                "h3_tail_projection expects RGB pixels, got "
                f"{pixels.shape}"
            )
        expected_action_blocks = (
            SOURCE_HISTORY_TOKENS - 1 + FUTURE_ACTION_BLOCKS
        )
        expected_action_shape = (
            expected_action_blocks,
            self.protocol.action_block_raw_steps,
            self.protocol.action_dim,
        )
        if actions.ndim != 4 or tuple(actions.shape[1:]) != expected_action_shape:
            raise ValueError(
                "h3_tail_projection expects [B,9,5,A] action blocks, got "
                f"{actions.shape}; expected trailing shape "
                f"{expected_action_shape}"
            )
        if len(pixels) != len(actions):
            raise ValueError("Pixel/action batch sizes differ")
        projected_action_blocks = (
            PROJECTED_HISTORY_TOKENS - 1 + FUTURE_ACTION_BLOCKS
        )
        return (
            pixels[:, -PROJECTED_HISTORY_TOKENS:],
            actions[:, -projected_action_blocks:],
        )

    def encode_pixels(
        self, pixels: np.ndarray, *, batch_size: int
    ) -> np.ndarray:
        return self._base_adapter.encode_pixels(pixels, batch_size=batch_size)

    def rollout_latents(
        self,
        input_pixels: np.ndarray,
        raw_action_blocks: np.ndarray,
        *,
        batch_size: int,
    ) -> np.ndarray:
        projected_pixels, projected_actions = self._project_inputs(
            input_pixels, raw_action_blocks
        )
        predicted = self._base_adapter.rollout_latents(
            projected_pixels,
            projected_actions,
            batch_size=batch_size,
        )
        values = np.asarray(predicted)
        if values.ndim < 3 or values.shape[1] != FUTURE_ACTION_BLOCKS:
            raise RuntimeError(
                "Native H3 adapter returned an incompatible projected "
                f"rollout: {values.shape}"
            )
        return values

    def frozen_state_hash(self) -> str:
        return self._base_adapter.frozen_state_hash()


__all__ = [
    "FUTURE_ACTION_BLOCKS",
    "PROJECTED_HISTORY_TOKENS",
    "SOURCE_HISTORY_TOKENS",
    "H3TailProjectionActionDelayAdapter",
]
