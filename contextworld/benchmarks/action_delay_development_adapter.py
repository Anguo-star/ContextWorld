"""Development request surface for the frozen History-3 tail projection.

The Public-Test recovery adapter in :mod:`action_delay_h3_tail_projection`
has a deliberately stable source contract: nine H7 action blocks request
three H3 futures.  Development Action Delay h1 asks for seven blocks and one
future.  This module adds that request variant without changing the frozen
adapter's source bytes or its Test metadata.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from contextworld.benchmarks.action_delay_h3_tail_projection import (
    FUTURE_ACTION_BLOCKS,
    H3TailProjectionActionDelayAdapter,
    PROJECTED_HISTORY_TOKENS,
    SOURCE_HISTORY_TOKENS,
)


DEVELOPMENT_FUTURE_ACTION_BLOCKS = 1
DEVELOPMENT_SOURCE_ACTION_BLOCKS = (
    SOURCE_HISTORY_TOKENS - 1 + DEVELOPMENT_FUTURE_ACTION_BLOCKS
)
TEST_SOURCE_ACTION_BLOCKS = SOURCE_HISTORY_TOKENS - 1 + FUTURE_ACTION_BLOCKS
SUPPORTED_SOURCE_ACTION_BLOCK_COUNTS = (
    DEVELOPMENT_SOURCE_ACTION_BLOCKS,
    TEST_SOURCE_ACTION_BLOCKS,
)


class DevelopmentH3TailProjectionActionDelayAdapter(
    H3TailProjectionActionDelayAdapter
):
    """Expose an H3 checkpoint through both Dev h1 and Test h3 requests."""

    adapter_id = "stable_worldmodel_h3_tail_projection_action_delay_dev_v1"

    def _requested_future_blocks(self, raw_action_blocks: np.ndarray) -> int:
        actions = np.asarray(raw_action_blocks)
        if actions.ndim != 4:
            raise ValueError(
                "h3_tail_projection expects [B,9,5,A] action blocks for Test "
                "or [B,7,5,A] for Development, got "
                f"{actions.shape}"
            )
        count = int(actions.shape[1])
        if count == DEVELOPMENT_SOURCE_ACTION_BLOCKS:
            return DEVELOPMENT_FUTURE_ACTION_BLOCKS
        if count == TEST_SOURCE_ACTION_BLOCKS:
            return FUTURE_ACTION_BLOCKS
        raise ValueError(
            "h3_tail_projection expects [B,9,5,A] action blocks for Test "
            "or [B,7,5,A] for Development, got "
            f"{actions.shape}"
        )

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
        requested = self._requested_future_blocks(actions)
        expected = (
            int(actions.shape[1]),
            self.protocol.action_block_raw_steps,
            self.protocol.action_dim,
        )
        if tuple(actions.shape[1:]) != expected:
            raise ValueError(
                "h3_tail_projection expects [B,9,5,A] action blocks for Test "
                "or [B,7,5,A] for Development, got "
                f"{actions.shape}; expected trailing shape {expected}"
            )
        if len(pixels) != len(actions):
            raise ValueError("Pixel/action batch sizes differ")
        projected_blocks = PROJECTED_HISTORY_TOKENS - 1 + requested
        return (
            pixels[:, -PROJECTED_HISTORY_TOKENS:],
            actions[:, -projected_blocks:],
        )

    @property
    def metadata(self) -> dict[str, Any]:
        metadata = dict(super().metadata)
        metadata["adapter_id"] = self.adapter_id
        metadata["adapter_class"] = (
            f"{type(self).__module__}.{type(self).__name__}"
        )
        metadata["supported_request_conventions"] = {
            "development_h1": {
                "source_action_block_count": DEVELOPMENT_SOURCE_ACTION_BLOCKS,
                "projected_action_block_count": (
                    PROJECTED_HISTORY_TOKENS - 1
                    + DEVELOPMENT_FUTURE_ACTION_BLOCKS
                ),
                "future_action_blocks": DEVELOPMENT_FUTURE_ACTION_BLOCKS,
            },
            "test_h3": {
                "source_action_block_count": TEST_SOURCE_ACTION_BLOCKS,
                "projected_action_block_count": (
                    PROJECTED_HISTORY_TOKENS - 1 + FUTURE_ACTION_BLOCKS
                ),
                "future_action_blocks": FUTURE_ACTION_BLOCKS,
            },
        }
        return metadata

    def rollout_latents(
        self,
        input_pixels: np.ndarray,
        raw_action_blocks: np.ndarray,
        *,
        batch_size: int,
    ) -> np.ndarray:
        requested = self._requested_future_blocks(raw_action_blocks)
        projected_pixels, projected_actions = self._project_inputs(
            input_pixels, raw_action_blocks
        )
        predicted = np.asarray(
            self._base_adapter.rollout_latents(
                projected_pixels,
                projected_actions,
                batch_size=batch_size,
            )
        )
        if predicted.ndim < 3 or predicted.shape[1] != requested:
            raise RuntimeError(
                "Native H3 adapter returned an incompatible projected "
                f"rollout: {predicted.shape}"
            )
        return predicted


class StableWorldModelLeWMH3TailProjectionAdapter(
    DevelopmentH3TailProjectionActionDelayAdapter
):
    """Built-in LeWM constructor for the explicit projection route."""

    adapter_id = "stable_worldmodel_lewm_h3_tail_projection_action_delay_v2"

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: Path,
        *,
        normalizer: Path,
        repo_root: Path,
        stablewm_repo: str,
        stablewm_ref: str,
        device: str,
    ) -> "StableWorldModelLeWMH3TailProjectionAdapter":
        from contextworld.benchmarks.adapters import StableWorldModelLeWMAdapter

        return cls(
            StableWorldModelLeWMAdapter.from_checkpoint(
                checkpoint,
                normalizer=normalizer,
                repo_root=repo_root,
                stablewm_repo=stablewm_repo,
                stablewm_ref=stablewm_ref,
                device=device,
            )
        )


class StableWorldModelPLDMH3TailProjectionAdapter(
    DevelopmentH3TailProjectionActionDelayAdapter
):
    """Built-in PLDM constructor for the explicit projection route."""

    adapter_id = "stable_worldmodel_pldm_h3_tail_projection_action_delay_v2"

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: Path,
        *,
        normalizer: Path,
        repo_root: Path,
        stablewm_repo: str,
        stablewm_ref: str,
        device: str,
    ) -> "StableWorldModelPLDMH3TailProjectionAdapter":
        from contextworld.benchmarks.adapters import StableWorldModelPLDMAdapter

        return cls(
            StableWorldModelPLDMAdapter.from_checkpoint(
                checkpoint,
                normalizer=normalizer,
                repo_root=repo_root,
                stablewm_repo=stablewm_repo,
                stablewm_ref=stablewm_ref,
                device=device,
            )
        )


class StableWorldModelPreJEPAActionDelayH3TailAdapter(
    DevelopmentH3TailProjectionActionDelayAdapter
):
    """Built-in regular PreJEPA constructor for the projection route."""

    adapter_id = "stable_worldmodel_prejepa_h3_tail_projection_action_delay_v2"
    required_history_tokens = SOURCE_HISTORY_TOKENS
    maximum_future_action_blocks = FUTURE_ACTION_BLOCKS
    raw_action_dim = 2

    @classmethod
    def from_contextworld_request(cls, request: Any):
        from contextworld.benchmarks.prejepa_adapters import (
            StableWorldModelPreJEPAAdapter,
        )

        return cls(StableWorldModelPreJEPAAdapter.from_contextworld_request(request))


class StableWorldModelPreJEPADiagnosticActionDelayH3TailAdapter(
    DevelopmentH3TailProjectionActionDelayAdapter
):
    """Built-in normalized-zero PreJEPA projection constructor."""

    adapter_id = (
        "stable_worldmodel_prejepa_diagnostic_h3_tail_projection_"
        "action_delay_v2"
    )
    required_history_tokens = SOURCE_HISTORY_TOKENS
    maximum_future_action_blocks = FUTURE_ACTION_BLOCKS
    raw_action_dim = 2

    @classmethod
    def from_contextworld_request(cls, request: Any):
        from contextworld.benchmarks.prejepa_adapters import (
            StableWorldModelPreJEPADiagnosticAdapter,
        )

        return cls(
            StableWorldModelPreJEPADiagnosticAdapter.from_contextworld_request(
                request
            )
        )


__all__ = [
    "DEVELOPMENT_FUTURE_ACTION_BLOCKS",
    "DEVELOPMENT_SOURCE_ACTION_BLOCKS",
    "TEST_SOURCE_ACTION_BLOCKS",
    "SUPPORTED_SOURCE_ACTION_BLOCK_COUNTS",
    "DevelopmentH3TailProjectionActionDelayAdapter",
    "StableWorldModelLeWMH3TailProjectionAdapter",
    "StableWorldModelPLDMH3TailProjectionAdapter",
    "StableWorldModelPreJEPAActionDelayH3TailAdapter",
    "StableWorldModelPreJEPADiagnosticActionDelayH3TailAdapter",
]
