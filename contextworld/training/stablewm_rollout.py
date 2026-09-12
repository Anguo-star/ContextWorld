"""Autoregressive multi-step loss overlay for Stable-WorldModel trainers.

Stable-WorldModel's ``wm.num_preds`` selects a target offset; it does not feed
one prediction back into the next prediction during training.  This module
keeps ``wm.num_preds=1`` and replaces only the native one-step prediction term
with the mean loss over an actual differentiable rollout.  The family trainer
continues to own its model, representation losses, optimizer, logging, and
checkpoint format.

The overlay is deliberately installed only by the ContextWorld family entry
when ``rollout_steps > 1``.  A one-step run never imports or patches it.
"""

from __future__ import annotations

from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Iterator

import torch
from torch.nn import functional as F


_METADATA_KEYS = frozenset(
    {
        "conditional_joint_group",
        "conditional_pairs",
        "conditional_active",
    }
)

ROLLOUT_CONTEXT_MODES = frozenset({"sliding", "expanding"})


@contextmanager
def _temporary_attribute(
    owner: Any, name: str, value: Any
) -> Iterator[None]:
    """Temporarily shadow one method and restore the exact prior state."""

    namespace = getattr(owner, "__dict__", {})
    had_instance_value = name in namespace
    instance_value = namespace.get(name)
    setattr(owner, name, value)
    try:
        yield
    finally:
        if had_instance_value:
            setattr(owner, name, instance_value)
        else:
            delattr(owner, name)


def _slice_sequence_mapping(
    values: dict[str, Any], *, sequence_length: int, prefix_length: int
) -> dict[str, Any]:
    """Copy a batch/output mapping and truncate only temporal tensors."""

    result: dict[str, Any] = {}
    for key, value in values.items():
        if (
            torch.is_tensor(value)
            and value.ndim >= 2
            and int(value.shape[1]) == sequence_length
        ):
            result[key] = value[:, :prefix_length]
        else:
            result[key] = value
    return result


def _model_batch(batch: dict[str, Any]) -> dict[str, Any]:
    """Return only inputs that may cross the StableWM model boundary."""

    values = {
        key: value for key, value in batch.items() if key not in _METADATA_KEYS
    }
    if "action" in values:
        values["action"] = torch.nan_to_num(values["action"], 0.0)
    return values


def _run_native_prefix(
    original_forward: Callable[..., dict[str, Any]],
    module: Any,
    batch: dict[str, Any],
    stage: str,
    cfg: Any,
    *,
    encoded_full: dict[str, Any],
    history_size: int,
    original_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Run the family objective on its unchanged H+1 prefix.

    Encoding is reused from the full H+K sequence.  Suppressing the original
    log call avoids publishing a one-step total immediately before the
    overlay replaces it with the rollout total.
    """

    sequence_length = int(batch["pixels"].shape[1])
    prefix_length = history_size + 1
    prefix_batch = _slice_sequence_mapping(
        batch,
        sequence_length=sequence_length,
        prefix_length=prefix_length,
    )
    encoded_prefix = _slice_sequence_mapping(
        encoded_full,
        sequence_length=sequence_length,
        prefix_length=prefix_length,
    )

    def cached_encode(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return dict(encoded_prefix)

    with _temporary_attribute(module.model, "encode", cached_encode):
        with _temporary_attribute(module, "log_dict", lambda *_a, **_k: None):
            return original_forward(
                module,
                prefix_batch,
                stage,
                cfg,
                **original_kwargs,
            )


def _log_losses(module: Any, output: dict[str, Any], stage: str) -> None:
    losses = {
        f"{stage}/{key}": value.detach()
        for key, value in output.items()
        if "loss" in key and torch.is_tensor(value)
    }
    module.log_dict(losses, on_step=True, sync_dist=True)


def _latent_rollout(
    model: Any,
    embeddings: torch.Tensor,
    action_embeddings: torch.Tensor,
    *,
    history_size: int,
    rollout_steps: int,
    detach_targets: bool,
    context_mode: str = "sliding",
) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
    """Roll out LeWM/PLDM one step at a time with unbroken gradients.

    ``sliding`` reproduces StableWM inference: every step sees at most H
    states. ``expanding`` is the memory-retention diagnostic: the original H
    observed states remain visible while autoregressive predictions are
    appended. The latter requires a predictor positional capacity of at least
    ``H + K - 1``; the typed launcher installs that capacity in the model
    config before construction.
    """

    if context_mode not in ROLLOUT_CONTEXT_MODES:
        raise ValueError(
            f"Unsupported rollout context mode {context_mode!r}; expected "
            f"one of {sorted(ROLLOUT_CONTEXT_MODES)}"
        )

    states = list(embeddings[:, :history_size].unbind(dim=1))
    predictions: list[torch.Tensor] = []
    losses: list[torch.Tensor] = []
    for step in range(rollout_steps):
        start = 0 if context_mode == "expanding" else len(states) - history_size
        context = torch.stack(states[start:], dim=1)
        actions = action_embeddings[:, start : len(states)]
        prediction = model.predict(context, actions)[:, -1]
        target = embeddings[:, history_size + step]
        if detach_targets:
            target = target.detach()
        step_loss = F.mse_loss(prediction, target)
        predictions.append(prediction)
        losses.append(step_loss)
        # No detach: later rollout losses train through every earlier step.
        states.append(prediction)
    return torch.stack(losses).mean(), predictions, losses


def _strip_action_dims(
    tensor: torch.Tensor, action_range: tuple[int, int]
) -> torch.Tensor:
    lo, hi = action_range
    return torch.cat([tensor[..., :lo], tensor[..., hi:]], dim=-1)


def _prejepa_action_range(
    model: Any, encoded: dict[str, Any]
) -> tuple[int, int]:
    start = int(encoded["pixels_emb"].shape[-1])
    for key in model.extra_encoders:
        width = int(encoded[f"{key}_emb"].shape[-1])
        if key == "action":
            return start, start + width
        start += width
    raise ValueError("PreJEPA rollout requires the model's action encoder")


def _replace_prejepa_action(
    model: Any, prediction: torch.Tensor, action: torch.Tensor
) -> torch.Tensor:
    """Install the action leaving a predicted frame before the next step."""

    # replace_action_in_embedding expects (B,N,T,P,D) and (B,N,T,A).
    return model.replace_action_in_embedding(
        prediction.unsqueeze(1), action.unsqueeze(1)
    ).squeeze(1)


def _prejepa_rollout(
    model: Any,
    encoded: dict[str, Any],
    raw_actions: torch.Tensor,
    *,
    history_size: int,
    rollout_steps: int,
) -> tuple[
    torch.Tensor,
    list[torch.Tensor],
    list[torch.Tensor],
    tuple[int, int],
]:
    embeddings = encoded["emb"]
    action_range = _prejepa_action_range(model, encoded)
    states = list(embeddings[:, :history_size].unbind(dim=1))
    predictions: list[torch.Tensor] = []
    losses: list[torch.Tensor] = []
    for step in range(rollout_steps):
        context = torch.stack(states[-history_size:], dim=1)
        prediction = model.predict(context)[:, -1:]
        target = embeddings[
            :, history_size + step : history_size + step + 1
        ].detach()
        step_loss = F.mse_loss(
            _strip_action_dims(prediction, action_range),
            _strip_action_dims(target, action_range),
        )
        predictions.append(prediction)
        losses.append(step_loss)
        if step + 1 < rollout_steps:
            next_action = raw_actions[
                :, history_size + step : history_size + step + 1
            ]
            prediction = _replace_prejepa_action(
                model, prediction, next_action
            )
        states.append(prediction[:, 0])
    return torch.stack(losses).mean(), predictions, losses, action_range


def _build_lewm_like_forward(
    original_forward: Callable[..., dict[str, Any]],
    *,
    rollout_steps: int,
    context_mode: str = "sliding",
) -> Callable[..., dict[str, Any]]:
    @wraps(original_forward)
    def forward(
        module: Any,
        batch: dict[str, Any],
        stage: str,
        cfg: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        history_size = int(cfg.wm.history_size)
        model_inputs = _model_batch(batch)
        encoded_full = module.model.encode(model_inputs)
        output = _run_native_prefix(
            original_forward,
            module,
            batch,
            stage,
            cfg,
            encoded_full=encoded_full,
            history_size=history_size,
            original_kwargs=kwargs,
        )
        rollout_loss, _predictions, step_losses = _latent_rollout(
            module.model,
            encoded_full["emb"],
            encoded_full["act_emb"],
            history_size=history_size,
            rollout_steps=rollout_steps,
            # Preserve the native LeWM/PLDM target-gradient convention.
            detach_targets=False,
            context_mode=context_mode,
        )
        native_one_step = output["pred_loss"]
        output["loss"] = output["loss"] - native_one_step + rollout_loss
        output["pred_loss"] = rollout_loss
        output["rollout_loss"] = rollout_loss
        for index, loss in enumerate(step_losses, 1):
            output[f"rollout_step_{index}_loss"] = loss
        _log_losses(module, output, stage)
        return output

    return forward


def _build_prejepa_forward(
    original_forward: Callable[..., dict[str, Any]],
    *,
    rollout_steps: int,
) -> Callable[..., dict[str, Any]]:
    @wraps(original_forward)
    def forward(
        module: Any,
        batch: dict[str, Any],
        stage: str,
        cfg: Any,
    ) -> dict[str, Any]:
        history_size = int(cfg.wm.history_size)
        model_inputs = _model_batch(batch)
        for key in module.model.extra_encoders:
            model_inputs[key] = torch.nan_to_num(
                model_inputs[key], 0.0
            ).squeeze()
        encoded_full = module.model.encode(
            model_inputs,
            target="emb",
            is_video=cfg.backbone.get("is_video_encoder", False),
        )
        output = _run_native_prefix(
            original_forward,
            module,
            batch,
            stage,
            cfg,
            encoded_full=encoded_full,
            history_size=history_size,
            original_kwargs={},
        )
        rollout_loss, predictions, step_losses, action_range = (
            _prejepa_rollout(
                module.model,
                encoded_full,
                model_inputs["action"],
                history_size=history_size,
                rollout_steps=rollout_steps,
            )
        )
        native_one_step = F.mse_loss(
            output["actionless_pred_emb"],
            output["actionless_target_emb"].detach(),
        )
        output["loss"] = output["loss"] - native_one_step + rollout_loss
        output["pred_loss"] = rollout_loss
        output["rollout_loss"] = rollout_loss
        for index, loss in enumerate(step_losses, 1):
            output[f"rollout_step_{index}_loss"] = loss

        pixels_width = int(encoded_full["pixels_emb"].shape[-1])
        output["pixels_loss"] = torch.stack(
            [
                F.mse_loss(
                    prediction[..., :pixels_width],
                    encoded_full["emb"][:, history_size + index : history_size + index + 1, ..., :pixels_width].detach(),
                )
                for index, prediction in enumerate(predictions)
            ]
        ).mean()
        # Keep the range visible to downstream probes exactly as in the
        # upstream one-step forward.
        output["actionless_emb"] = _strip_action_dims(
            encoded_full["emb"], action_range
        )
        _log_losses(module, output, stage)
        if not torch.isfinite(output["loss"]):
            raise ValueError("Non-finite autoregressive rollout loss encountered")
        return output

    return forward


def install_rollout_forward(
    trainer_module: Any,
    *,
    family: str,
    rollout_steps: int,
    context_mode: str = "sliding",
) -> None:
    """Patch one imported upstream trainer module, failing closed by family."""

    if rollout_steps <= 1:
        raise ValueError("Rollout overlay requires rollout_steps > 1")
    if context_mode not in ROLLOUT_CONTEXT_MODES:
        raise ValueError(
            f"Unsupported rollout context mode {context_mode!r}; expected "
            f"one of {sorted(ROLLOUT_CONTEXT_MODES)}"
        )
    if context_mode == "expanding" and family not in {"lewm", "pldm"}:
        raise ValueError(
            "Expanding rollout context is currently implemented only for "
            "LeWM and PLDM predictors"
        )
    if family in {"lewm", "pldm"}:
        attribute = "lejepa_forward" if family == "lewm" else "pldm_forward"
        original = getattr(trainer_module, attribute, None)
        if not callable(original):
            raise RuntimeError(
                f"The {family} trainer does not expose {attribute}; "
                "rollout training cannot be installed safely"
            )
        setattr(
            trainer_module,
            attribute,
            _build_lewm_like_forward(
                original,
                rollout_steps=rollout_steps,
                context_mode=context_mode,
            ),
        )
        return
    if family == "viswm":
        # VIS-WM imports LeWM's run_training function; that function resolves
        # lejepa_forward in the imported lewm module at call time.
        import lewm as lewm_trainer

        install_rollout_forward(
            lewm_trainer,
            family="lewm",
            rollout_steps=rollout_steps,
            context_mode=context_mode,
        )
        return
    if family == "prejepa":
        original = getattr(trainer_module, "dinowm_forward", None)
        if not callable(original):
            raise RuntimeError(
                "The PreJEPA trainer does not expose dinowm_forward; "
                "rollout training cannot be installed safely"
            )
        trainer_module.dinowm_forward = _build_prejepa_forward(
            original, rollout_steps=rollout_steps
        )
        return
    raise ValueError(f"Unsupported StableWM rollout family: {family!r}")


__all__ = ["ROLLOUT_CONTEXT_MODES", "install_rollout_forward"]
