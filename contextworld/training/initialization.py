"""Deterministic weight-initialization helpers for ContextWorld training.

This module owns exactly two responsibilities:

1. :func:`validate_initialization` -- pure validation/bookkeeping of an
   optional checkpoint request, with an automatically recorded SHA-256.
2. :func:`apply_initialization` -- load model weights according to a validated
   specification, without ever touching optimizer state, epoch counters, RNG
   state, or parameter ``requires_grad`` flags.

Only the Python standard library is imported at module scope; ``torch`` is
imported lazily inside :func:`apply_initialization` (and its private helpers,
which are only reachable from there) so importing this helper never pulls in
the training stack and never performs a random draw or reseeding.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict

__all__ = [
    "apply_initialization",
    "validate_initialization",
    "INITIALIZATION_FAMILIES",
    "VISUAL_MODULE_NAMES",
]

# Add a family only after validating its model ownership, state formats and
# resume/DDP behavior. This is a capability declaration, not a task allowlist.
INITIALIZATION_FAMILIES = frozenset({"lewm", "pldm", "viswm", "prejepa"})

# Submodules copied by the optional ``visual`` initialization scope. Both
# LeWM and PLDM expose exactly this visual representation; the freeze policy
# in contextworld.training.visual_policy uses the same names.
VISUAL_MODULE_NAMES = ("encoder", "projector")

_MODE_SCRATCH = "scratch"
_MODE_WARMSTART = "warmstart"
_SCOPE_MODEL = "model"
_SCOPE_VISUAL = "visual"
_SEMANTICS = "model_weights_only_fresh_optimizer"
_PREFIX = "model."
_READ_CHUNK_BYTES = 1 << 20
_SPEC_FIELDS = frozenset({"mode", "checkpoint", "sha256", "semantics"})


def _resolve_checkpoint(checkpoint: Any) -> Path:
    """Resolve ``checkpoint`` to an existing, non-empty, absolute regular file."""
    if checkpoint is None:
        raise ValueError("warmstart initialization requires a checkpoint file")
    if isinstance(checkpoint, bytes) or not isinstance(checkpoint, (str, os.PathLike)):
        raise ValueError(
            "checkpoint must be a filesystem path (str or os.PathLike), got %r"
            % (checkpoint,)
        )
    try:
        candidate = Path(os.fspath(checkpoint)).expanduser()
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise ValueError("checkpoint must be a filesystem path") from exc
    if not candidate.is_absolute():
        raise ValueError("checkpoint must be supplied as an absolute path")
    resolved = candidate.resolve()
    if not resolved.is_absolute():
        raise ValueError(
            "checkpoint could not be resolved to an absolute path: %r" % (checkpoint,)
        )
    if not resolved.exists():
        raise ValueError("checkpoint does not exist: %s" % resolved)
    if not resolved.is_file():
        raise ValueError("checkpoint is not a regular file: %s" % resolved)
    if resolved.stat().st_size == 0:
        raise ValueError("checkpoint is empty: %s" % resolved)
    return resolved


def _stream_sha256(path: Path) -> str:
    """Stream-hash ``path`` in fixed-size chunks; return the lowercase digest."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(_READ_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def validate_initialization(
    checkpoint: Any = None, scope: str = _SCOPE_MODEL
) -> Dict[str, Any]:
    """Validate an initialization request and return its canonical description.

    Returns ``{'mode': mode, 'checkpoint': resolved_absolute_path_or_None,
    'sha256': lowercase_digest_or_None,
    'semantics': 'model_weights_only_fresh_optimizer'}``. For the optional
    ``scope='visual'`` the canonical spec carries one extra field
    ``'scope': 'visual'``; the ``model`` scope keeps the canonical four fields
    exactly as before.

    * No path (None or an empty string) preserves default model initialization.
      The current LeWM default is recorded internally as ``scratch``. The
      ``visual`` scope is meaningless without a checkpoint and is rejected.
    * A supplied path selects ``warmstart`` and must be an existing, non-empty,
      absolute file. Its SHA256 is computed and recorded automatically.

    Any invalid input raises :class:`ValueError`.
    """
    if scope not in (_SCOPE_MODEL, _SCOPE_VISUAL):
        raise ValueError("initialization scope must be 'model' or 'visual'")
    if checkpoint is None or checkpoint == "":
        if scope == _SCOPE_VISUAL:
            raise ValueError("visual initialization scope requires a checkpoint")
        return {
            "mode": _MODE_SCRATCH,
            "checkpoint": None,
            "sha256": None,
            "semantics": _SEMANTICS,
        }
    resolved = _resolve_checkpoint(checkpoint)
    computed = _stream_sha256(resolved)
    canonical = {
        "mode": _MODE_WARMSTART,
        "checkpoint": str(resolved),
        "sha256": computed,
        "semantics": _SEMANTICS,
    }
    if scope == _SCOPE_VISUAL:
        canonical["scope"] = _SCOPE_VISUAL
    return canonical


def _model_state_sha256(model: Any) -> str:
    """Deterministically hash ``model.state_dict()`` (names/dtype/shape/bytes)."""
    import torch

    state = model.state_dict()
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name]
        if not isinstance(tensor, torch.Tensor):
            raise ValueError("state_dict entry %r is not a tensor" % name)
        # reshape(-1) first so 0-dim buffers (e.g. num_batches_tracked) can be
        # byte-viewed as uint8.
        flat = tensor.detach().to("cpu").contiguous().reshape(-1)
        digest.update(name.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(b"\x00")
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(b"\x00")
        digest.update(flat.view(torch.uint8).numpy().tobytes())
        digest.update(b"\x00")
    return digest.hexdigest()


def _extract_model_state(payload: Any) -> Dict[str, Any]:
    """Normalize a torch.load() payload to a tensor-only ``model.``-free state dict.

    Accepts a raw tensor-only state dict or a Lightning-style checkpoint
    (``payload['state_dict']``; optimizer/epoch/RNG metadata is ignored).
    As in the historical loader, select the complete ``model.`` subtree
    when present. Other Lightning module state (e.g. SIGReg buffers) belongs
    to the training wrapper and is not a world-model parameter.
    """
    import torch

    state = payload
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict) or not state:
        raise ValueError("malformed state: expected a non-empty mapping of tensors")
    if any(not isinstance(key, str) for key in state):
        raise ValueError("malformed state: state keys must be strings")
    prefixed = [key for key in state if key.startswith(_PREFIX)]
    if prefixed:
        state = {key[len(_PREFIX):]: state[key] for key in prefixed}
    for key, value in state.items():
        if not isinstance(value, torch.Tensor):
            raise ValueError("malformed state: entry %r is not a tensor" % key)
    return state


def _select_visual_state(model_state: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Split a full ``model.``-free state dict into per-visual-module states."""
    selected: Dict[str, Dict[str, Any]] = {}
    for name in VISUAL_MODULE_NAMES:
        prefix = name + "."
        incoming = {
            key[len(prefix):]: value
            for key, value in model_state.items()
            if key.startswith(prefix)
        }
        if not incoming:
            raise ValueError(
                "checkpoint lacks visual state for module %r" % name
            )
        selected[name] = incoming
    return selected


def _apply_visual_scope(model: Any, model_state: Dict[str, Any]) -> Dict[str, int]:
    """Copy only ``encoder``/``projector`` state (buffers included) into ``model``.

    Every requested module's keys and shapes are validated against the target
    submodule before any tensor is copied, so a mismatch leaves the model
    untouched. Nothing outside the visual modules is read or written, no RNG
    state is consumed and ``requires_grad`` flags are never modified.
    """
    selected = _select_visual_state(model_state)
    for name, incoming in selected.items():
        submodule = getattr(model, name, None)
        if submodule is None:
            raise ValueError("model has no visual module %r" % name)
        target = submodule.state_dict()
        if set(target) != set(incoming) or any(
            target[key].shape != incoming[key].shape for key in target
        ):
            raise RuntimeError(
                "visual state mismatch for module %r: target keys %d vs "
                "incoming keys %d" % (name, len(target), len(incoming))
            )
    copied: Dict[str, int] = {}
    for name, incoming in selected.items():
        getattr(model, name).load_state_dict(incoming, strict=True)
        copied[name] = len(incoming)
    return copied


def _canonicalize_spec(spec: Any) -> Dict[str, Any]:
    """Require exactly the identity produced by launcher validation."""
    if not isinstance(spec, Mapping):
        raise ValueError("initialization spec must be a mapping")
    scope = spec.get("scope", _SCOPE_MODEL)
    if scope == _SCOPE_MODEL:
        expected_fields = _SPEC_FIELDS
    elif scope == _SCOPE_VISUAL:
        expected_fields = _SPEC_FIELDS | {"scope"}
    else:
        raise ValueError("initialization spec scope must be 'model' or 'visual'")
    if set(spec) != expected_fields:
        raise ValueError("initialization spec must contain the canonical four fields")
    canonical = validate_initialization(spec["checkpoint"], scope=scope)
    if dict(spec) != canonical:
        raise ValueError("initialization spec must equal the canonical validated identity")
    return canonical


def apply_initialization(model: Any, spec: Mapping[str, Any]) -> Dict[str, Any]:
    """Initialize ``model`` according to ``spec`` and return a JSON-simple receipt.

    Receipt keys: ``initialization`` (the canonical spec), and
    ``model_state_sha256_before`` / ``model_state_sha256_after``,
    ``trainable_parameter_count``, ``total_parameter_count``.

    ``scratch`` never touches the model. ``warmstart`` re-validates the SHA-256
    (via :func:`validate_initialization`), then
    ``torch.load(path, map_location='cpu', weights_only=True)`` and
    ``load_state_dict(strict=True)`` after a supported explicit model-state
    adaptation, if needed. PreJEPA state-input checkpoints migrate visual/action
    channels with a non-equivalence receipt. A spec with
    ``scope='visual'`` copies only the ``encoder``/``projector`` tensors
    (BatchNorm buffers included) and leaves every other module untouched.
    Optimizer state, epoch counters, RNG and
    ``requires_grad`` flags are never modified; no random draws or reseeding.
    """
    import torch  # noqa: F401  -- lazy import, per contract

    canonical = _canonicalize_spec(spec)
    for attribute in ("state_dict", "load_state_dict", "parameters"):
        if not callable(getattr(model, attribute, None)):
            raise ValueError("model must implement %s()" % attribute)

    state_before = _model_state_sha256(model)
    trainable_before = sum(
        1 for parameter in model.parameters() if parameter.requires_grad
    )
    total_parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    adaptation = None
    visual_modules = None
    if canonical["mode"] == _MODE_WARMSTART:
        # validate_initialization (inside _canonicalize_spec) already streamed
        # the file and matched the digest again.
        with open(canonical["checkpoint"], "rb") as handle:
            payload = torch.load(handle, map_location="cpu", weights_only=True)
            handle.seek(0)
            digest = hashlib.sha256()
            for chunk in iter(lambda: handle.read(_READ_CHUNK_BYTES), b""):
                digest.update(chunk)
            if digest.hexdigest() != canonical["sha256"]:
                raise ValueError("initialization checkpoint changed during loading")
        model_state = _extract_model_state(payload)
        if canonical.get("scope") == _SCOPE_VISUAL:
            # Visual-only scope: copy encoder+projector tensors (BatchNorm
            # buffers included) and leave every other module at its native
            # RNG initialization. Mismatches raise before any mutation.
            visual_modules = _apply_visual_scope(model, model_state)
        else:
            from contextworld.training.prejepa_initialization import adapt_prejepa_state
            model_state, adaptation = adapt_prejepa_state(model, model_state)
            # Validate everything before mutation; strict PyTorch loading can otherwise
            # copy matching tensors before reporting an incompatible tensor.
            expected = model.state_dict()
            if set(expected) != set(model_state) or any(
                expected[k].shape != model_state[k].shape for k in expected
            ):
                raise RuntimeError("Initialization state keys/shapes do not match target model")
            model.load_state_dict(model_state, strict=True)

    state_after = _model_state_sha256(model)
    trainable_after = sum(
        1 for parameter in model.parameters() if parameter.requires_grad
    )
    if trainable_after != trainable_before:
        raise RuntimeError("initialization changed parameter requires_grad flags")
    if canonical["mode"] == _MODE_SCRATCH and state_after != state_before:
        raise RuntimeError("scratch initialization must not modify model state")

    return {
        "initialization": canonical,
        "adaptation": adaptation,
        "visual_scope": visual_modules,
        "model_state_sha256_before": state_before,
        "model_state_sha256_after": state_after,
        "trainable_parameter_count": trainable_parameter_count,
        "total_parameter_count": total_parameter_count,
    }
