"""Declare the existing original DINO-WM's RGB/action inference variant.

The checkpoint is loaded unchanged. Its unavailable state streams are fixed
to zero in model-normalized coordinates, exactly as in the archived original
diagnostic. This is an explicitly named model adapter, not a new task metric
or a claim that the checkpoint was trained without state inputs. The native
adapter's incompatibility/diagnostic metadata remains in ``base_adapter``.

Only the ordinary public AdapterRequest and RGB/action prediction arguments
are accepted. No simulator state, labels, or query targets are supplied to
the model. Action Delay uses the already implemented H3 tail projection.
"""

from __future__ import annotations

from typing import Any

from contextworld.benchmarks.adapters import LatentWorldModelAdapter


METHOD_ID = "dinowm_original_fixed_context_v1"
ADAPTER_SPEC = __name__ + ":OriginalDINOFixedContextAdapter"

_TASK_CLASSES = {
    "speed": "StableWorldModelPreJEPADiagnosticAdapter",
    "door": "StableWorldModelPreJEPADiagnosticAdapter",
    "action_strength": "StableWorldModelPreJEPADiagnosticActionStrengthAdapter",
    "contact_friction": "StableWorldModelPreJEPADiagnosticContactFrictionAdapter",
    "motion_damping": "StableWorldModelPreJEPADiagnosticMotionDampingAdapter",
    "portal_exit": "StableWorldModelPreJEPADiagnosticPortalExitAdapter",
    "robot_arm_mass": "StableWorldModelPreJEPADiagnosticReacherArmMassAdapter",
    "cube_gripper_carry": "StableWorldModelPreJEPADiagnosticCubeGraspRuleAdapter",
}


class OriginalDINOFixedContextAdapter(LatentWorldModelAdapter):
    """A declared, fixed-weight inference adapter for existing checkpoints."""

    adapter_id = METHOD_ID

    def __init__(self, base: LatentWorldModelAdapter, *, task: str) -> None:
        self.base = base
        self.task = task

    @classmethod
    def from_contextworld_request(cls, request: Any):
        if request.task == "action_delay":
            from contextworld.benchmarks.action_delay_development_adapter import (
                StableWorldModelPreJEPADiagnosticActionDelayH3TailAdapter,
            )

            adapter_class = StableWorldModelPreJEPADiagnosticActionDelayH3TailAdapter
        else:
            from contextworld.benchmarks import prejepa_adapters

            try:
                adapter_class = getattr(prejepa_adapters, _TASK_CLASSES[request.task])
            except KeyError as exc:
                raise ValueError(f"Unsupported original DINO task: {request.task}") from exc
        return cls(adapter_class.from_contextworld_request(request), task=request.task)

    @property
    def protocol(self):
        return self.base.protocol

    def frozen_state_hash(self) -> str:
        return self.base.frozen_state_hash()

    @property
    def metadata(self) -> dict[str, Any]:
        base = self.base.metadata
        # Keep complete native provenance, including its original diagnostic
        # label. A declared inference variant does not rewrite that history.
        return {
            **base,
            "adapter_id": self.adapter_id,
            "adapter_class": f"{type(self).__module__}.{type(self).__name__}",
            "checkpoint": base.get("checkpoint"),
            "checkpoint_sha256": base.get("checkpoint_sha256"),
            "stable_worldmodel_repo": base.get("stable_worldmodel_repo"),
            "stable_worldmodel_commit": base.get("stable_worldmodel_commit"),
            "public_inputs": ["pixels", "action"],
            "inference_variant": METHOD_ID,
            "missing_context_policy": "fixed_model_normalized_zero",
            "checkpoint_weights_modified": False,
            "native_checkpoint_input_compatible": False,
            "history_adapter": "h3_tail_projection" if self.task == "action_delay" else "native",
            "interpretation": (
                "Score of the declared fixed-context inference variant; not a "
                "native state-input score or a data-only training ablation."
            ),
            "base_adapter": base,
        }

    def encode_pixels(self, pixels, *, batch_size: int):
        return self.base.encode_pixels(pixels, batch_size=batch_size)

    def rollout_latents(self, input_pixels, raw_action_blocks, *, batch_size: int):
        return self.base.rollout_latents(
            input_pixels, raw_action_blocks, batch_size=batch_size
        )
