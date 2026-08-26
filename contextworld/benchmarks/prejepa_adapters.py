"""Stable-WorldModel PreJEPA adapters for the public ICL input contract.

The frozen ContextWorld v1 scorers expose RGB history and raw actions only.
That is sufficient for a PreJEPA checkpoint trained with pixel and action
encoders, but not for a checkpoint whose predictor also consumes additional
``proprio`` or ``observation`` streams. The default adapters reject those
checkpoints rather than silently inventing state or adding a simulator side
channel. Separate, explicitly named diagnostic adapters can instead supply
normalized-space zero state; their metadata makes clear that they are not
frozen-v1-compatible scores.

This module stays separate from ``adapters.py`` because published component
releases pin that file byte-for-byte.  It loads PreJEPA through Stable-
WorldModel's native ``load_pretrained`` path and adapts only its public
``encode``/``rollout`` surfaces.  Predictions and targets are compared in the
same visual DINO feature space; action slots are never scored as state.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from contextworld.benchmarks.adapters import (
    AdapterProtocol,
    StableWorldModelLeWMActionStrengthAdapter,
    StableWorldModelLeWMAdapter,
    StableWorldModelLeWMContactFrictionAdapter,
    StableWorldModelLeWMCubeGraspRuleAdapter,
    StableWorldModelLeWMHistory7Adapter,
    StableWorldModelLeWMMotionDampingAdapter,
    StableWorldModelLeWMPortalExitAdapter,
    StableWorldModelLeWMReacherArmMassAdapter,
    _preprocess_pixels,
)
from contextworld.benchmarks.action_delay_h3_tail_projection import (
    H3TailProjectionActionDelayAdapter,
)
from contextworld.evaluation.icl_model import file_sha256
from contextworld.evaluation.protocol import (
    ColumnStandardizer,
    frozen_normalizer_process,
)
from contextworld.paths import artifact_path
from contextworld.synthesis.stablewm import load_stable_worldmodel


class PreJEPAInputContractError(ValueError):
    """A checkpoint requires inputs the frozen v1 scorer does not expose."""


def _action_standardizer(request: Any) -> ColumnStandardizer:
    if request.action_normalizer is not None:
        process = frozen_normalizer_process(
            Path(request.action_normalizer).expanduser().resolve()
        )
        return process["action"]
    if request.action_mean is None or request.action_std is None:
        raise PreJEPAInputContractError(
            "PreJEPA evaluation needs either a frozen action normalizer or "
            "explicit action mean/std statistics"
        )
    return ColumnStandardizer(
        np.asarray(request.action_mean, dtype=np.float32)[None],
        np.asarray(request.action_std, dtype=np.float32)[None],
    )


class _PreJEPAAdapterMixin:
    """Implement the native PreJEPA load and visual-rollout contract."""

    model_config_name = "prejepa"
    # ``reject`` is the public frozen-v1 behavior.  The value is deliberately
    # a class-level policy, rather than a request field, so the generic
    # AdapterRequest remains an evaluation contract and only the explicitly
    # selected diagnostic adapter can opt into synthesized inputs.
    missing_context_strategy = "reject"

    def __init__(
        self,
        *,
        model: Any,
        checkpoint: Path,
        stable_repo: Path,
        stable_commit: str,
        action_standardizer: ColumnStandardizer,
        device: str,
    ) -> None:
        import torch

        self.model = model.to(device).eval()
        self.model.requires_grad_(False)
        setattr(self.model, "interpolate_pos_encoding", True)

        extra_encoders = getattr(self.model, "extra_encoders", None)
        if extra_encoders is None:
            raise PreJEPAInputContractError(
                "Loaded PreJEPA model exposes no extra_encoders mapping"
            )
        keys = tuple(str(key) for key in extra_encoders.keys())
        state_keys = tuple(key for key in keys if key != "action")
        if self.missing_context_strategy not in {"reject", "normalized_zero"}:
            raise PreJEPAInputContractError(
                "Unknown PreJEPA missing-context strategy "
                f"{self.missing_context_strategy!r}"
            )
        if state_keys and self.missing_context_strategy == "reject":
            raise PreJEPAInputContractError(
                "This PreJEPA checkpoint requires context stream(s) "
                f"{list(state_keys)}, but the frozen ContextWorld v1 ICL "
                "contract supplies only pixels and actions. Zero-filled state "
                "or simulator side channels would not be a valid score."
            )
        if "action" not in extra_encoders:
            raise PreJEPAInputContractError(
                "PreJEPA ICL rollout requires an action encoder"
            )

        state_streams: list[tuple[str, int]] = []
        if self.missing_context_strategy == "normalized_zero":
            for key in state_keys:
                raw_in_chans = getattr(extra_encoders[key], "in_chans", 0)
                try:
                    in_chans = int(raw_in_chans)
                except (TypeError, ValueError) as exc:
                    raise PreJEPAInputContractError(
                        "Cannot synthesize normalized-zero context for "
                        f"PreJEPA stream {key!r}: invalid "
                        f"in_chans={raw_in_chans!r}"
                    ) from exc
                if in_chans <= 0:
                    raise PreJEPAInputContractError(
                        "Cannot synthesize normalized-zero context for "
                        f"PreJEPA stream {key!r}: invalid in_chans={in_chans}"
                    )
                state_streams.append((key, in_chans))
        self._state_streams = tuple(state_streams)

        trained_history = int(getattr(self.model, "history_size", 0))
        if trained_history != int(self.required_history_tokens):
            raise PreJEPAInputContractError(
                f"Checkpoint history_size={trained_history} does not match "
                f"the frozen History={self.required_history_tokens} task protocol"
            )

        action_dim = int(self.raw_action_dim)
        action_width = int(getattr(extra_encoders["action"], "in_chans", 0))
        if action_dim <= 0 or action_width <= 0 or action_width % action_dim:
            raise PreJEPAInputContractError(
                "Cannot infer PreJEPA action block from action encoder width "
                f"{action_width} and raw action dimension {action_dim}"
            )
        action_block = action_width // action_dim

        mean = np.asarray(action_standardizer.mean)
        std = np.asarray(action_standardizer.std)
        if (
            mean.ndim == 0
            or std.ndim == 0
            or mean.shape[-1] != action_dim
            or std.shape[-1] != action_dim
        ):
            raise PreJEPAInputContractError(
                "Action normalizer dimension does not match the adapter "
                f"protocol: expected {action_dim}, got mean={mean.shape}, "
                f"std={std.shape}"
            )

        self._protocol = AdapterProtocol(
            history_tokens=trained_history,
            action_block_raw_steps=action_block,
            action_dim=action_dim,
            future_action_blocks=int(self.maximum_future_action_blocks),
        )
        if action_block != 5:
            raise PreJEPAInputContractError(
                "ContextWorld v1 requires five raw actions per block, got "
                f"{action_block}"
            )

        self.checkpoint = Path(checkpoint).resolve()
        self.checkpoint_sha256 = file_sha256(self.checkpoint)
        self.stable_repo = Path(stable_repo).resolve()
        self.stable_commit = str(stable_commit)
        self.action_standardizer = action_standardizer
        self.device = str(device)

        # Ensure at least one parameter exists before rollout uses its dtype.
        try:
            next(self.model.parameters())
        except StopIteration as exc:
            raise PreJEPAInputContractError(
                "Loaded PreJEPA model has no parameters"
            ) from exc
        if not isinstance(next(self.model.parameters()), torch.Tensor):
            raise PreJEPAInputContractError("Invalid PreJEPA parameter surface")

    @classmethod
    def from_contextworld_request(cls, request: Any):
        checkpoint = Path(request.checkpoint).expanduser().resolve()
        if checkpoint.suffix.lower() != ".pt":
            raise PreJEPAInputContractError(
                f"PreJEPA requires StableWM's native .pt format, got {checkpoint}"
            )
        if not (checkpoint.parent / "config.json").is_file():
            raise FileNotFoundError(checkpoint.parent / "config.json")

        runtime = dict(request.runtime)
        swm, stable_repo, stable_commit = load_stable_worldmodel(
            request.repo_root,
            runtime.get("stablewm_repo"),
            runtime.get("stablewm_ref"),
        )
        model = swm.wm.utils.load_pretrained(
            str(checkpoint),
            cache_dir=str(
                artifact_path(
                    "evaluation/model_cache",
                    repo_root=request.repo_root,
                ).resolve()
            ),
        )
        return cls(
            model=model,
            checkpoint=checkpoint,
            stable_repo=stable_repo,
            stable_commit=stable_commit,
            action_standardizer=_action_standardizer(request),
            device=request.device,
        )

    @property
    def protocol(self) -> AdapterProtocol:
        return self._protocol

    @property
    def metadata(self) -> dict[str, Any]:
        metadata = super().metadata
        if self.missing_context_strategy != "normalized_zero":
            return metadata
        return {
            **metadata,
            "diagnostic": True,
            "diagnostic_reason": "missing_context_normalized_zero",
            "frozen_v1_compatible": False,
            "prejepa_missing_context_policy": "normalized_zero",
            "missing_context_strategy": "normalized_zero",
            "normalized_zero_state_streams": [
                {"key": key, "in_chans": in_chans}
                for key, in_chans in self._state_streams
            ],
        }

    def encode_pixels(
        self, pixels: np.ndarray, *, batch_size: int
    ) -> np.ndarray:
        import torch

        values = np.asarray(pixels, dtype=np.uint8)
        output = []
        with torch.inference_mode():
            for start in range(0, len(values), int(batch_size)):
                transformed = _preprocess_pixels(
                    values[start : start + int(batch_size)],
                    device=self.device,
                ).unsqueeze(1)
                encoded = self.model.encode(
                    {"pixels": transformed}, emb_keys=[]
                )["pixels_emb"][:, 0]
                # ContextWorld's adapter contract exposes one latent vector
                # per frame. DINO keeps a patch axis internally, so flatten
                # patch × channel identically for targets and predictions.
                encoded = encoded.reshape(encoded.shape[0], -1)
                output.append(encoded.detach().float().cpu().numpy())
        if not output:
            raise ValueError("Cannot encode an empty pixel batch")
        return np.concatenate(output, axis=0)

    def rollout_latents(
        self,
        input_pixels: np.ndarray,
        raw_action_blocks: np.ndarray,
        *,
        batch_size: int,
    ) -> np.ndarray:
        import torch

        pixels = np.asarray(input_pixels, dtype=np.uint8)
        actions = np.asarray(raw_action_blocks, dtype=np.float32)
        history = self.protocol.history_tokens
        if pixels.ndim != 5 or pixels.shape[1] != history:
            raise ValueError(
                f"Expected [B,{history},H,W,3] input pixels, got {pixels.shape}"
            )
        if len(pixels) != len(actions):
            raise ValueError("Pixel/action batch sizes differ")

        expected_future = actions.shape[1] - (history - 1)
        if not 1 <= expected_future <= self.protocol.future_action_blocks:
            raise ValueError(
                f"Requested unsupported future length {expected_future}"
            )

        outputs = []
        parameter_dtype = next(self.model.parameters()).dtype
        with torch.inference_mode():
            for start in range(0, len(pixels), int(batch_size)):
                pixel_chunk = pixels[start : start + int(batch_size)]
                batch, frames = pixel_chunk.shape[:2]
                transformed = _preprocess_pixels(
                    pixel_chunk.reshape(-1, *pixel_chunk.shape[2:]),
                    device=self.device,
                ).reshape(
                    batch,
                    frames,
                    3,
                    pixel_chunk.shape[2],
                    pixel_chunk.shape[3],
                )
                normalized = self._normalize_actions(
                    actions[start : start + int(batch_size)]
                )
                action_tensor = torch.from_numpy(normalized).to(
                    device=self.device,
                    dtype=parameter_dtype,
                )
                action_history = action_tensor[:, : history - 1]
                future_actions = action_tensor[:, history - 1 :]

                rollout_info: dict[str, Any] = {
                    "pixels": transformed[:, None],
                    "action_history": action_history[:, None],
                }
                # These tensors are already in the model's normalized input
                # space.  They are a diagnostic missing-context substitute,
                # never a reconstruction of simulator observations.
                for key, in_chans in self._state_streams:
                    rollout_info[key] = torch.zeros(
                        (batch, 1, frames, in_chans),
                        device=transformed.device,
                        dtype=parameter_dtype,
                    )

                # The upstream cache is keyed by environment id/step.  ICL
                # bundles have neither and are independent, so no cache may
                # cross a batch boundary.
                if hasattr(self.model, "_init_cached_info"):
                    delattr(self.model, "_init_cached_info")
                result = self.model.rollout(
                    rollout_info,
                    future_actions[:, None],
                )
                predicted = result["predicted_pixels_emb"][:, 0, history:]
                if predicted.shape[1] != expected_future:
                    raise RuntimeError(
                        "PreJEPA returned an unexpected future length: "
                        f"{tuple(predicted.shape)}"
                    )
                predicted = predicted.reshape(
                    predicted.shape[0], predicted.shape[1], -1
                )
                outputs.append(predicted.detach().float().cpu().numpy())
        if not outputs:
            raise ValueError("Cannot rollout an empty batch")
        return np.concatenate(outputs, axis=0)


class StableWorldModelPreJEPAAdapter(
    _PreJEPAAdapterMixin, StableWorldModelLeWMAdapter
):
    """History-3 adapter for state-free StableWM PreJEPA checkpoints."""

    adapter_id = "stable_worldmodel_prejepa_v2"


class StableWorldModelPreJEPAHistory7Adapter(
    _PreJEPAAdapterMixin, StableWorldModelLeWMHistory7Adapter
):
    """History-7 adapter for a state-free PreJEPA checkpoint trained at H=7."""

    adapter_id = "stable_worldmodel_prejepa_history7_v2"


class StableWorldModelPreJEPAActionStrengthAdapter(
    _PreJEPAAdapterMixin, StableWorldModelLeWMActionStrengthAdapter
):
    adapter_id = "stable_worldmodel_prejepa_action_strength_v2"


class StableWorldModelPreJEPAContactFrictionAdapter(
    _PreJEPAAdapterMixin, StableWorldModelLeWMContactFrictionAdapter
):
    adapter_id = "stable_worldmodel_prejepa_contact_friction_v2"


class StableWorldModelPreJEPAMotionDampingAdapter(
    _PreJEPAAdapterMixin, StableWorldModelLeWMMotionDampingAdapter
):
    adapter_id = "stable_worldmodel_prejepa_motion_damping_v2"


class StableWorldModelPreJEPAPortalExitAdapter(
    _PreJEPAAdapterMixin, StableWorldModelLeWMPortalExitAdapter
):
    adapter_id = "stable_worldmodel_prejepa_portal_exit_v2"


class StableWorldModelPreJEPAReacherArmMassAdapter(
    _PreJEPAAdapterMixin, StableWorldModelLeWMReacherArmMassAdapter
):
    adapter_id = "stable_worldmodel_prejepa_reacher_arm_mass_v2"


class StableWorldModelPreJEPACubeGraspRuleAdapter(
    _PreJEPAAdapterMixin, StableWorldModelLeWMCubeGraspRuleAdapter
):
    adapter_id = "stable_worldmodel_prejepa_cube_grasp_rule_v2"


class StableWorldModelPreJEPADiagnosticAdapter(
    StableWorldModelPreJEPAAdapter
):
    """Explicit non-frozen-v1 adapter for missing state context.

    This is intentionally a different adapter class from the ordinary
    PreJEPA adapter.  Callers must opt in to the diagnostic result before a
    state-conditioned checkpoint can receive normalized-zero state streams.
    """

    adapter_id = "stable_worldmodel_prejepa_diagnostic_normalized_zero_v1"
    missing_context_strategy = "normalized_zero"


class StableWorldModelPreJEPADiagnosticHistory7Adapter(
    StableWorldModelPreJEPAHistory7Adapter
):
    """History-7 diagnostic adapter for state-conditioned PreJEPA."""

    adapter_id = (
        "stable_worldmodel_prejepa_diagnostic_history7_normalized_zero_v1"
    )
    missing_context_strategy = "normalized_zero"


class StableWorldModelPreJEPADiagnosticActionStrengthAdapter(
    StableWorldModelPreJEPAActionStrengthAdapter
):
    adapter_id = (
        "stable_worldmodel_prejepa_diagnostic_action_strength_"
        "normalized_zero_v1"
    )
    missing_context_strategy = "normalized_zero"


class StableWorldModelPreJEPADiagnosticContactFrictionAdapter(
    StableWorldModelPreJEPAContactFrictionAdapter
):
    adapter_id = (
        "stable_worldmodel_prejepa_diagnostic_contact_friction_"
        "normalized_zero_v1"
    )
    missing_context_strategy = "normalized_zero"


class StableWorldModelPreJEPADiagnosticMotionDampingAdapter(
    StableWorldModelPreJEPAMotionDampingAdapter
):
    adapter_id = (
        "stable_worldmodel_prejepa_diagnostic_motion_damping_"
        "normalized_zero_v1"
    )
    missing_context_strategy = "normalized_zero"


class StableWorldModelPreJEPADiagnosticPortalExitAdapter(
    StableWorldModelPreJEPAPortalExitAdapter
):
    adapter_id = (
        "stable_worldmodel_prejepa_diagnostic_portal_exit_normalized_zero_v1"
    )
    missing_context_strategy = "normalized_zero"


class StableWorldModelPreJEPADiagnosticReacherArmMassAdapter(
    StableWorldModelPreJEPAReacherArmMassAdapter
):
    adapter_id = (
        "stable_worldmodel_prejepa_diagnostic_reacher_arm_mass_"
        "normalized_zero_v1"
    )
    missing_context_strategy = "normalized_zero"


class StableWorldModelPreJEPADiagnosticCubeGraspRuleAdapter(
    StableWorldModelPreJEPACubeGraspRuleAdapter
):
    adapter_id = (
        "stable_worldmodel_prejepa_diagnostic_cube_grasp_rule_"
        "normalized_zero_v1"
    )
    missing_context_strategy = "normalized_zero"


class StableWorldModelPreJEPAActionDelayH3TailAdapter(
    H3TailProjectionActionDelayAdapter
):
    """Expose a native H3 PreJEPA checkpoint at Action Delay's H7 edge."""

    adapter_id = "stable_worldmodel_prejepa_h3_tail_projection_action_delay_v1"
    required_history_tokens = 7
    maximum_future_action_blocks = 3
    raw_action_dim = 2

    @classmethod
    def from_contextworld_request(cls, request: Any):
        return cls(
            StableWorldModelPreJEPAAdapter.from_contextworld_request(request)
        )


class StableWorldModelPreJEPADiagnosticActionDelayH3TailAdapter(
    StableWorldModelPreJEPAActionDelayH3TailAdapter
):
    """Expose a diagnostic PreJEPA H3 checkpoint at Action Delay's H7 edge."""

    adapter_id = (
        "stable_worldmodel_prejepa_diagnostic_h3_tail_projection_"
        "action_delay_v1"
    )

    @classmethod
    def from_contextworld_request(cls, request: Any):
        return cls(
            StableWorldModelPreJEPADiagnosticAdapter.from_contextworld_request(
                request
            )
        )


__all__ = [
    "PreJEPAInputContractError",
    "StableWorldModelPreJEPAActionStrengthAdapter",
    "StableWorldModelPreJEPAAdapter",
    "StableWorldModelPreJEPAActionDelayH3TailAdapter",
    "StableWorldModelPreJEPAContactFrictionAdapter",
    "StableWorldModelPreJEPACubeGraspRuleAdapter",
    "StableWorldModelPreJEPADiagnosticActionDelayH3TailAdapter",
    "StableWorldModelPreJEPADiagnosticActionStrengthAdapter",
    "StableWorldModelPreJEPADiagnosticAdapter",
    "StableWorldModelPreJEPADiagnosticContactFrictionAdapter",
    "StableWorldModelPreJEPADiagnosticCubeGraspRuleAdapter",
    "StableWorldModelPreJEPADiagnosticHistory7Adapter",
    "StableWorldModelPreJEPADiagnosticMotionDampingAdapter",
    "StableWorldModelPreJEPADiagnosticPortalExitAdapter",
    "StableWorldModelPreJEPADiagnosticReacherArmMassAdapter",
    "StableWorldModelPreJEPAHistory7Adapter",
    "StableWorldModelPreJEPAMotionDampingAdapter",
    "StableWorldModelPreJEPAPortalExitAdapter",
    "StableWorldModelPreJEPAReacherArmMassAdapter",
]
