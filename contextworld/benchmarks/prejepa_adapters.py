"""DINOv2 (``prejepa``) adapters for Stable-WorldModel checkpoints.

These live outside ``adapters.py`` because that module's bytes are pinned by
frozen release configurations -- the speed and door releases both record its
sha256, so editing it would invalidate the provenance of published results.
Adding a model family is not a reason to break that seal, and a separate
module costs nothing.

``prejepa`` is Stable-WorldModel's DINOv2 world model.  Its training config
(``scripts/train/config/prejepa.yaml``) matches the published DINO-WM
architecture: ``dinov2_small``, patch size 14, image size 224, history 3,
frameskip 5, and a predictor of depth 6 with 16 heads.  Because it is a
Stable-WorldModel family, checkpoint loading, ImageNet preprocessing and the
frozen-state hash are all inherited unchanged from the LeWM adapters, and the
evaluation stays aligned with the baselines by construction.

Only the rollout call differs, in three ways that ``_PreJEPARolloutMixin``
absorbs.  Everything else -- geometry, normalization, batching, the
future-length contract -- is inherited.
"""

from __future__ import annotations

from typing import Any

from contextworld.benchmarks.adapters import (
    StableWorldModelLeWMActionStrengthAdapter,
    StableWorldModelLeWMAdapter,
    StableWorldModelLeWMContactFrictionAdapter,
    StableWorldModelLeWMCubeGraspRuleAdapter,
    StableWorldModelLeWMHistory7Adapter,
    StableWorldModelLeWMMotionDampingAdapter,
    StableWorldModelLeWMPortalExitAdapter,
    StableWorldModelLeWMReacherArmMassAdapter,
)


class _PreJEPARolloutShim:
    """Present ``prejepa``'s rollout under the ``lewm``/``pldm`` interface.

    The base adapter calls
    ``model.rollout(info, actions, history_size=H)["predicted_emb"]``.  The
    ``prejepa`` module differs in three ways, reconciled here so that the
    inherited ``rollout_latents`` -- with its preprocessing, batching and
    future-length contract -- runs unmodified:

    * ``rollout`` accepts no ``history_size`` argument; it reads the value
      from its own attribute.
    * The result is the mutated info dict, and the visual stream lives under
      ``predicted_visual``.  The benchmark compares against ``encode_pixels``
      targets, which are visual-only, so scoring the concatenated
      action/proprio slots would compare a model's own action encoding with
      itself.
    * ``rollout`` caches its initial embedding against ``info['id']`` and
      ``info['step_idx']``.  ContextWorld scores independent query bundles
      that carry neither key, so the cache is dropped per call; a stale hit
      would silently score one bundle from another's initial state.

    Wrapping is used rather than subclassing the adapter because
    ``adapters.py`` is byte-pinned by frozen release configs and offers no
    override seam inside ``rollout_latents``.
    """

    def __init__(self, model: Any) -> None:
        self._model = model

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)

    def rollout(self, info: Any, action_sequence: Any, **_: Any) -> Any:
        if hasattr(self._model, "_init_cached_info"):
            delattr(self._model, "_init_cached_info")
        result = self._model.rollout(info, action_sequence)
        if "predicted_visual" not in result:
            raise RuntimeError(
                "prejepa rollout returned no predicted_visual stream; got "
                f"keys {sorted(result)}"
            )
        return {"predicted_emb": result["predicted_visual"]}


class _PreJEPARolloutMixin:
    """Load a ``prejepa`` checkpoint and wrap it for the benchmark contract."""

    model_config_name = "prejepa"

    def __init__(self, *, model: Any, **keywords: Any) -> None:
        super().__init__(model=_PreJEPARolloutShim(model), **keywords)


class StableWorldModelPreJEPAAdapter(
    _PreJEPARolloutMixin, StableWorldModelLeWMAdapter
):
    """History-3 adapter for Stable-WorldModel ``prejepa`` checkpoints."""

    adapter_id = "stable_worldmodel_prejepa_v1"


class StableWorldModelPreJEPAHistory7Adapter(
    _PreJEPARolloutMixin, StableWorldModelLeWMHistory7Adapter
):
    """History-7 ``prejepa`` adapter for the Action Delay benchmark."""

    adapter_id = "stable_worldmodel_prejepa_history7_v1"


class StableWorldModelPreJEPAActionStrengthAdapter(
    _PreJEPARolloutMixin, StableWorldModelLeWMActionStrengthAdapter
):
    """History-3 ``prejepa`` adapter with frozen PushT action statistics."""

    adapter_id = "stable_worldmodel_prejepa_action_strength_v1"


class StableWorldModelPreJEPAContactFrictionAdapter(
    _PreJEPARolloutMixin, StableWorldModelLeWMContactFrictionAdapter
):
    """History-3 ``prejepa`` adapter for PushT Contact Friction."""

    adapter_id = "stable_worldmodel_prejepa_contact_friction_v1"


class StableWorldModelPreJEPAMotionDampingAdapter(
    _PreJEPARolloutMixin, StableWorldModelLeWMMotionDampingAdapter
):
    """History-3 ``prejepa`` adapter for PushT Motion Damping."""

    adapter_id = "stable_worldmodel_prejepa_motion_damping_v1"


class StableWorldModelPreJEPAPortalExitAdapter(
    _PreJEPARolloutMixin, StableWorldModelLeWMPortalExitAdapter
):
    """History-3 ``prejepa`` adapter for TwoRoom Portal Exit."""

    adapter_id = "stable_worldmodel_prejepa_portal_exit_v1"


class StableWorldModelPreJEPAReacherArmMassAdapter(
    _PreJEPARolloutMixin, StableWorldModelLeWMReacherArmMassAdapter
):
    """History-3 ``prejepa`` adapter for Reacher Arm Mass."""

    adapter_id = "stable_worldmodel_prejepa_reacher_arm_mass_v1"


class StableWorldModelPreJEPACubeGraspRuleAdapter(
    _PreJEPARolloutMixin, StableWorldModelLeWMCubeGraspRuleAdapter
):
    """History-3 ``prejepa`` adapter for the Cube grasp-rule benchmark."""

    adapter_id = "stable_worldmodel_prejepa_cube_grasp_rule_v1"


__all__ = [
    "StableWorldModelPreJEPAActionStrengthAdapter",
    "StableWorldModelPreJEPAAdapter",
    "StableWorldModelPreJEPAContactFrictionAdapter",
    "StableWorldModelPreJEPACubeGraspRuleAdapter",
    "StableWorldModelPreJEPAHistory7Adapter",
    "StableWorldModelPreJEPAMotionDampingAdapter",
    "StableWorldModelPreJEPAPortalExitAdapter",
    "StableWorldModelPreJEPAReacherArmMassAdapter",
]
