"""Freeze only a pretrained visual encoder during a model warmstart.

This policy is intentionally separate from the historical ``visual``
initialization scope and ``freeze_visual`` policy.  A current encoder-freeze
run first loads the *complete* checkpoint state, then freezes only the
encoder/backbone.  Projectors, predictors, action heads and auxiliary heads
therefore retain the checkpoint initialization and remain trainable.
"""

from __future__ import annotations

from typing import Any, Dict

__all__ = [
    "FREEZE_ENCODER_FAMILIES",
    "FREEZE_ENCODER_POLICY_NAME",
    "attach_encoder_freeze",
    "freeze_encoder",
    "freeze_encoder_identity",
    "validate_encoder_freeze_policy",
]


FREEZE_ENCODER_FAMILIES = frozenset({"lewm", "pldm", "prejepa"})
FREEZE_ENCODER_POLICY_NAME = "freeze_encoder_v1"


def _module_names(family: str) -> tuple[str, ...]:
    if family in {"lewm", "pldm"}:
        return ("encoder",)
    if family == "prejepa":
        # Stable-WorldModel's PreJEPA stores the frozen visual module under
        # ``backbone`` (usually an EvalOnly wrapper around the pretrained
        # DINO encoder).  ``encode`` also detaches its output natively.
        return ("backbone",)
    raise ValueError(
        "CW_FREEZE_ENCODER currently supports only families %s; got %r"
        % (sorted(FREEZE_ENCODER_FAMILIES), family)
    )


def freeze_encoder_identity(*, family: str) -> Dict[str, Any]:
    """Return the stable receipt fragment for an encoder-freeze request."""
    names = _module_names(family)
    return {
        "policy": FREEZE_ENCODER_POLICY_NAME,
        "family": family,
        "modules": list(names),
        "train_behavior": "eval_mode_every_training_batch_including_buffers",
        "semantics": "full_model_warmstart_encoder_only_frozen",
    }


def validate_encoder_freeze_policy(
    *, family: Any, checkpoint: Any,
) -> Dict[str, Any]:
    """Validate a freeze request; full-model initialization is mandatory."""
    if not checkpoint:
        raise ValueError(
            "CW_FREEZE_ENCODER=1 requires a pretrained CW_INIT_CHECKPOINT"
        )
    return freeze_encoder_identity(family=family)


def freeze_encoder(model: Any, *, family: str) -> Dict[str, Any]:
    """Freeze only ``encoder`` (LeWM/PLDM) or ``backbone`` (PreJEPA).

    This must run before StablePretraining constructs optimizer groups.  The
    returned record includes buffers for auditable BN/statistics handling;
    downstream modules are deliberately untouched.
    """
    names = _module_names(family)
    record: Dict[str, Any] = {
        "policy": FREEZE_ENCODER_POLICY_NAME,
        "family": family,
        "modules": list(names),
        "frozen_parameters": [],
        "frozen_buffers": [],
        "training_flags": {},
    }
    for name in names:
        submodule = getattr(model, name, None)
        if submodule is None:
            raise ValueError("model has no encoder module %r" % name)
        for pname, parameter in submodule.named_parameters():
            parameter.requires_grad_(False)
            record["frozen_parameters"].append(f"{name}.{pname}")
        record["frozen_buffers"].extend(
            f"{name}.{bname}" for bname, _ in submodule.named_buffers()
        )
        submodule.eval()
        record["training_flags"][name] = bool(submodule.training)
    return record


try:
    from lightning.pytorch.callbacks import Callback as _CallbackBase
except Exception:  # pragma: no cover - Lightning-free import
    class _CallbackBase:  # type: ignore[no-redef]
        pass


class EncoderFreezeCallback(_CallbackBase):
    """Re-enforce encoder eval mode after Lightning calls ``model.train()``."""

    def __init__(self, model: Any, *, family: str):
        self.model = model
        self.module_names = _module_names(family)

    def _enforce_eval(self) -> None:
        for name in self.module_names:
            getattr(self.model, name).eval()

    def on_train_start(self, trainer, pl_module) -> None:
        self._enforce_eval()

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx) -> None:
        self._enforce_eval()


def attach_encoder_freeze(
    trainer: Any, model: Any, *, family: str,
) -> EncoderFreezeCallback:
    """Attach eval enforcement to the caller-owned Lightning trainer."""
    callbacks = getattr(trainer, "callbacks", None)
    if not isinstance(callbacks, list):
        raise RuntimeError(
            "CW_FREEZE_ENCODER=1 requires a Lightning trainer with a callbacks list"
        )
    callback = EncoderFreezeCallback(model, family=family)
    callbacks.append(callback)
    return callback
