from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from contextworld.evaluation.icl_model import file_sha256, state_dict_sha256
from contextworld.evaluation.protocol import (
    ColumnStandardizer,
    frozen_normalizer_process,
    infer_model_protocol,
    load_pretrained_cost_model,
)
from contextworld.paths import artifact_path
from contextworld.synthesis.stablewm import load_stable_worldmodel


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _load_model(
    checkpoint: Path,
    *,
    stable_worldmodel: Any,
    stable_repo: Path,
    repo_root: Path,
    model_config_name: str,
    action_input_dim: int,
) -> Any:
    """Load either StableWM's native or a legacy Lightning checkpoint.

    Native ``.pt`` checkpoints keep the long-standing
    ``.pt + config.json`` loading path exactly.  The original TwoRoom and
    PushT baselines instead use Lightning ``.ckpt`` payloads.  Those payloads
    are instantiated from the family-specific pinned StableWM training
    configuration, then loaded only from their ``model.*`` state entries with
    strict key matching.
    """

    checkpoint = Path(checkpoint).expanduser().resolve()
    if checkpoint.suffix.lower() == ".pt":
        return load_pretrained_cost_model(
            checkpoint,
            stable_worldmodel,
            cache_dir=artifact_path(
                "evaluation/model_cache",
                repo_root=repo_root,
            ),
        )
    if checkpoint.suffix.lower() != ".ckpt":
        raise ValueError(
            "Stable-WorldModel adapter expects a .pt or legacy .ckpt "
            f"checkpoint, got {checkpoint}"
        )

    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf, open_dict

    config = OmegaConf.load(
        stable_repo / f"scripts/train/config/{model_config_name}.yaml"
    )
    with open_dict(config):
        config.model.action_encoder.input_dim = int(action_input_dim)
    model = instantiate(config.model)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("state_dict") if isinstance(payload, dict) else None
    if not isinstance(state, dict):
        raise ValueError(
            f"Legacy checkpoint has no state_dict mapping: {checkpoint}"
        )
    model_state = {
        name.removeprefix("model."): value
        for name, value in state.items()
        if name.startswith("model.")
    }
    if not model_state:
        raise ValueError(
            f"Legacy checkpoint has no model.* tensors: {checkpoint}"
        )
    model.load_state_dict(model_state, strict=True)
    return model


@dataclass(frozen=True)
class AdapterProtocol:
    """Input/output contract for a latent-world-model evaluator.

    A model encodes simulator-rendered target frames and predicts future
    latents in that same checkpoint-native space.  ContextWorld never asks it
    to reconstruct pixels, so a decoder is neither required nor called.
    """

    history_tokens: int
    action_block_raw_steps: int
    action_dim: int
    future_action_blocks: int
    native_target_encoder: bool = True
    decoder_required: bool = False


class LatentWorldModelAdapter(ABC):
    """Model-independent latent boundary used by ContextWorld scorers.

    Adapters receive raw uint8 RGB frames and raw environment actions.  This
    keeps model-specific image transforms, action normalization and latent
    representation choices outside the benchmark dataset.  The evaluator
    compares predictions only with targets encoded by the same frozen model;
    it never requires a generative image decoder.  Individual tasks validate
    their required history, action geometry, and future horizon at scoring
    time, so one implementation can serve every compatible task.
    """

    @property
    @abstractmethod
    def protocol(self) -> AdapterProtocol:
        """Return the model input and rollout contract."""

    @property
    @abstractmethod
    def metadata(self) -> dict[str, Any]:
        """Return serializable model/runtime provenance."""

    @abstractmethod
    def encode_pixels(
        self, pixels: np.ndarray, *, batch_size: int
    ) -> np.ndarray:
        """Encode ``[batch,height,width,3]`` uint8 images into native targets."""

    @abstractmethod
    def rollout_latents(
        self,
        input_pixels: np.ndarray,
        raw_action_blocks: np.ndarray,
        *,
        batch_size: int,
    ) -> np.ndarray:
        """Return future latents for the supplied input history and actions."""

    @abstractmethod
    def frozen_state_hash(self) -> str:
        """Hash model parameters and buffers without changing model state."""


# ``SpeedICLModelAdapter`` was the original public name.  Keep it as an
# identity alias so existing subclasses and isinstance/issubclass checks keep
# working while integrations can use the task-neutral name above.
SpeedICLModelAdapter = LatentWorldModelAdapter


def validate_adapter_protocol(
    adapter: LatentWorldModelAdapter,
    *,
    history_tokens: int,
    action_block_raw_steps: int,
    action_dim: int,
    minimum_future_action_blocks: int,
    task_name: str,
) -> AdapterProtocol:
    """Validate one task's geometry without requiring a task-specific class.

    A generic adapter may expose a larger rollout horizon than a task needs,
    but it must match that task's history and raw-action representation.  The
    check deliberately does not constrain model family, latent width, or
    framework, allowing external implementations to reuse the public scorer.
    """

    expected = {
        "history_tokens": int(history_tokens),
        "action_block_raw_steps": int(action_block_raw_steps),
        "action_dim": int(action_dim),
        "minimum_future_action_blocks": int(minimum_future_action_blocks),
    }
    if any(value <= 0 for value in expected.values()):
        raise ValueError(
            "Adapter protocol requirements must be positive: "
            f"{expected}"
        )

    protocol = adapter.protocol
    try:
        observed = {
            "history_tokens": int(protocol.history_tokens),
            "action_block_raw_steps": int(protocol.action_block_raw_steps),
            "action_dim": int(protocol.action_dim),
            "future_action_blocks": int(protocol.future_action_blocks),
        }
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(
            f"{task_name} adapter must expose a complete protocol; got "
            f"{protocol!r}"
        ) from exc

    checks = {
        "history_tokens": (
            observed["history_tokens"] == expected["history_tokens"]
        ),
        "action_block_raw_steps": (
            observed["action_block_raw_steps"]
            == expected["action_block_raw_steps"]
        ),
        "action_dim": observed["action_dim"] == expected["action_dim"],
        "future_action_blocks": (
            observed["future_action_blocks"]
            >= expected["minimum_future_action_blocks"]
        ),
    }
    if not all(checks.values()):
        requirement = (
            f"History={expected['history_tokens']}, "
            f"action blocks={expected['action_block_raw_steps']}x"
            f"{expected['action_dim']}, and at least "
            f"{expected['minimum_future_action_blocks']} future block(s)"
        )
        raise ValueError(
            f"{task_name} adapter protocol is incompatible; requires "
            f"{requirement}; got {protocol} (checks={checks})"
        )
    return protocol


def _preprocess_pixels(pixels: np.ndarray, *, device: str):
    import torch

    values = np.asarray(pixels, dtype=np.uint8)
    if values.ndim != 4 or values.shape[-1] != 3:
        raise ValueError(f"Expected [B,H,W,3] uint8 pixels, got {values.shape}")
    tensor = torch.from_numpy(values).to(device=device)
    tensor = tensor.permute(0, 3, 1, 2).float().div_(255.0)
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    return (tensor - mean) / std


class StableWorldModelLeWMAdapter(LatentWorldModelAdapter):
    """Tested adapter for the pinned Stable-WorldModel LeWM checkpoint format."""

    adapter_id = "stable_worldmodel_lewm_v1"
    required_history_tokens = 3
    maximum_future_action_blocks = 5
    raw_action_dim = 2
    model_config_name = "lewm"
    action_input_dim = 10

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
        self.model = model.to(device).eval()
        self.model.requires_grad_(False)
        setattr(self.model, "history_size", self.required_history_tokens)
        setattr(self.model, "interpolate_pos_encoding", True)
        action_dim = int(self.raw_action_dim)
        if action_dim <= 0:
            raise ValueError(f"Invalid raw action dimension: {action_dim}")
        normalizer_mean = np.asarray(action_standardizer.mean)
        normalizer_std = np.asarray(action_standardizer.std)
        if (
            normalizer_mean.ndim == 0
            or normalizer_std.ndim == 0
            or normalizer_mean.shape[-1] != action_dim
            or normalizer_std.shape[-1] != action_dim
        ):
            raise ValueError(
                "Action normalizer dimension does not match the adapter "
                f"protocol: expected {action_dim}, got "
                f"mean={normalizer_mean.shape}, std={normalizer_std.shape}"
            )
        inferred = infer_model_protocol(self.model, action_dim=action_dim)
        self._protocol = AdapterProtocol(
            history_tokens=int(inferred["history_size"]),
            action_block_raw_steps=int(inferred["action_block"]),
            action_dim=action_dim,
            future_action_blocks=self.maximum_future_action_blocks,
        )
        if self._protocol.history_tokens != self.required_history_tokens:
            raise RuntimeError(
                f"{self.adapter_id} requires History-"
                f"{self.required_history_tokens}, got {self._protocol}"
            )
        if self._protocol.action_block_raw_steps != 5:
            raise RuntimeError(
                f"{self.adapter_id} requires action block 5, got "
                f"{self._protocol}"
            )
        self.checkpoint = checkpoint.resolve()
        self.checkpoint_sha256 = file_sha256(self.checkpoint)
        self.stable_repo = stable_repo.resolve()
        self.stable_commit = str(stable_commit)
        self.action_standardizer = action_standardizer
        self.device = str(device)

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
    ) -> "StableWorldModelLeWMAdapter":
        checkpoint = Path(checkpoint).expanduser().resolve()
        normalizer = Path(normalizer).expanduser().resolve()
        swm, stable_repo, stable_commit = load_stable_worldmodel(
            repo_root, stablewm_repo, stablewm_ref
        )
        model = _load_model(
            checkpoint,
            stable_worldmodel=swm,
            stable_repo=stable_repo,
            repo_root=repo_root,
            model_config_name=cls.model_config_name,
            action_input_dim=cls.action_input_dim,
        )
        process = frozen_normalizer_process(normalizer)
        return cls(
            model=model,
            checkpoint=checkpoint,
            stable_repo=stable_repo,
            stable_commit=stable_commit,
            action_standardizer=process["action"],
            device=device,
        )

    @property
    def protocol(self) -> AdapterProtocol:
        return self._protocol

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "adapter_id": self.adapter_id,
            "adapter_class": f"{type(self).__module__}.{type(self).__name__}",
            "checkpoint": str(self.checkpoint),
            "checkpoint_sha256": self.checkpoint_sha256,
            "stable_worldmodel_repo": str(self.stable_repo),
            "stable_worldmodel_commit": self.stable_commit,
            "device": self.device,
            "protocol": asdict(self.protocol),
            "model_class": f"{type(self.model).__module__}.{type(self.model).__name__}",
            "parameters": sum(
                parameter.numel() for parameter in self.model.parameters()
            ),
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
                    values[start : start + int(batch_size)], device=self.device
                ).unsqueeze(1)
                encoded = self.model.encode({"pixels": transformed})["emb"][:, 0]
                output.append(encoded.detach().float().cpu().numpy())
        if not output:
            raise ValueError("Cannot encode an empty pixel batch")
        return np.concatenate(output, axis=0)

    def _normalize_actions(self, blocks: np.ndarray) -> np.ndarray:
        values = np.asarray(blocks, dtype=np.float32)
        expected = (
            self.protocol.action_block_raw_steps,
            self.protocol.action_dim,
        )
        if values.ndim != 4 or tuple(values.shape[-2:]) != expected:
            raise ValueError(
                f"Expected [B,T,{expected[0]},{expected[1]}] actions, "
                f"got {values.shape}"
            )
        normalized = self.action_standardizer.transform(
            values.reshape(-1, self.protocol.action_dim)
        ).astype(np.float32)
        return normalized.reshape(values.shape[0], values.shape[1], -1)

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
        if pixels.ndim != 5 or pixels.shape[1] != self.protocol.history_tokens:
            raise ValueError(
                "Expected [B,history,H,W,3] input pixels, got "
                f"{pixels.shape}"
            )
        if len(pixels) != len(actions):
            raise ValueError("Pixel/action batch sizes differ")
        outputs = []
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
                    dtype=next(self.model.parameters()).dtype,
                )
                rollout_code = getattr(self.model.rollout, "__code__", None)
                explicit_action_history = bool(
                    rollout_code is not None
                    and "action_history" in rollout_code.co_consts
                )
                if explicit_action_history:
                    history_actions = action_tensor[
                        :, : self.protocol.history_tokens - 1
                    ]
                    future_actions = action_tensor[
                        :, self.protocol.history_tokens - 1 :
                    ]
                    rollout_info = {
                        "pixels": transformed[:, None],
                        "action_history": history_actions[:, None],
                    }
                    rollout_actions = future_actions[:, None]
                    expected_future = future_actions.shape[1]
                else:
                    # Historical StableWM checkouts encode the first H-1
                    # executed blocks in the action sequence itself.  Keep
                    # that frozen protocol reproducible while current models
                    # use the explicit action_history field above.
                    rollout_info = {"pixels": transformed[:, None]}
                    rollout_actions = action_tensor[:, None]
                    expected_future = (
                        action_tensor.shape[1]
                        - (self.protocol.history_tokens - 1)
                    )
                result = self.model.rollout(
                    rollout_info,
                    rollout_actions,
                    history_size=self.protocol.history_tokens,
                )["predicted_emb"][:, 0]
                predicted = result[:, self.protocol.history_tokens :]
                # ``action_tensor`` has one token per transition.  History H
                # consumes H-1 context actions, so T action tokens request
                # T-(H-1) future predictions.
                if not 1 <= expected_future <= self.protocol.future_action_blocks:
                    raise ValueError(
                        f"Requested unsupported future length {expected_future}"
                    )
                if predicted.shape[1] != expected_future:
                    raise RuntimeError(
                        "Adapter returned an unexpected future length: "
                        f"{predicted.shape}"
                    )
                outputs.append(predicted.detach().float().cpu().numpy())
        if not outputs:
            raise ValueError("Cannot rollout an empty batch")
        return np.concatenate(outputs, axis=0)

    def frozen_state_hash(self) -> str:
        return state_dict_sha256(self.model)


class StableWorldModelPLDMAdapter(StableWorldModelLeWMAdapter):
    """History-3 adapter for Stable-WorldModel PLDM checkpoints."""

    adapter_id = "stable_worldmodel_pldm_v1"
    model_config_name = "pldm"


class StableWorldModelLeWMHistory7Adapter(StableWorldModelLeWMAdapter):
    """LeWM adapter for the History-7 Action Delay benchmark."""

    adapter_id = "stable_worldmodel_lewm_history7_v1"
    required_history_tokens = 7
    maximum_future_action_blocks = 3


class StableWorldModelPLDMHistory7Adapter(StableWorldModelPLDMAdapter):
    """PLDM adapter for the History-7 Action Delay benchmark."""

    adapter_id = "stable_worldmodel_pldm_history7_v1"
    required_history_tokens = 7
    maximum_future_action_blocks = 3


class StableWorldModelLeWMActionStrengthAdapter(
    StableWorldModelLeWMAdapter
):
    """History-3 LeWM adapter with the frozen PushT action statistics."""

    adapter_id = "stable_worldmodel_lewm_action_strength_v1"

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: Path,
        *,
        action_mean: tuple[float, float] | list[float],
        action_std: tuple[float, float] | list[float],
        repo_root: Path,
        stablewm_repo: str,
        stablewm_ref: str,
        device: str,
    ) -> "StableWorldModelLeWMActionStrengthAdapter":
        checkpoint = Path(checkpoint).expanduser().resolve()
        swm, stable_repo, stable_commit = load_stable_worldmodel(
            repo_root, stablewm_repo, stablewm_ref
        )
        model = _load_model(
            checkpoint,
            stable_worldmodel=swm,
            stable_repo=stable_repo,
            repo_root=repo_root,
            model_config_name=cls.model_config_name,
            action_input_dim=cls.action_input_dim,
        )
        return cls(
            model=model,
            checkpoint=checkpoint,
            stable_repo=stable_repo,
            stable_commit=stable_commit,
            action_standardizer=ColumnStandardizer(
                np.asarray(action_mean, dtype=np.float32)[None],
                np.asarray(action_std, dtype=np.float32)[None],
            ),
            device=device,
        )


class StableWorldModelPLDMActionStrengthAdapter(
    StableWorldModelLeWMActionStrengthAdapter
):
    """History-3 PLDM adapter for PushT Action Strength checkpoints."""

    adapter_id = "stable_worldmodel_pldm_action_strength_v1"
    model_config_name = "pldm"


class StableWorldModelLeWMContactFrictionAdapter(
    StableWorldModelLeWMActionStrengthAdapter
):
    """History-3 LeWM adapter for PushT Contact Friction checkpoints."""

    adapter_id = "stable_worldmodel_lewm_contact_friction_v1"


class StableWorldModelPLDMContactFrictionAdapter(
    StableWorldModelLeWMContactFrictionAdapter
):
    """History-3 PLDM adapter for PushT Contact Friction checkpoints."""

    adapter_id = "stable_worldmodel_pldm_contact_friction_v1"
    model_config_name = "pldm"


class StableWorldModelLeWMMotionDampingAdapter(
    StableWorldModelLeWMActionStrengthAdapter
):
    """History-3 LeWM adapter for PushT Motion Damping checkpoints."""

    adapter_id = "stable_worldmodel_lewm_motion_damping_v1"


class StableWorldModelPLDMMotionDampingAdapter(
    StableWorldModelLeWMMotionDampingAdapter
):
    """History-3 PLDM adapter for PushT Motion Damping checkpoints."""

    adapter_id = "stable_worldmodel_pldm_motion_damping_v1"
    model_config_name = "pldm"


class StableWorldModelLeWMPortalExitAdapter(
    StableWorldModelLeWMActionStrengthAdapter
):
    """History-3 LeWM adapter for the TwoRoom Portal Exit benchmark."""

    adapter_id = "stable_worldmodel_lewm_portal_exit_v1"


class StableWorldModelPLDMPortalExitAdapter(
    StableWorldModelLeWMPortalExitAdapter
):
    """History-3 PLDM adapter for the TwoRoom Portal Exit benchmark."""

    adapter_id = "stable_worldmodel_pldm_portal_exit_v1"
    model_config_name = "pldm"


class StableWorldModelLeWMReacherArmMassAdapter(
    StableWorldModelLeWMActionStrengthAdapter
):
    """History-3 LeWM adapter for Reacher arm-mass checkpoints."""

    adapter_id = "stable_worldmodel_lewm_reacher_arm_mass_v1"
    model_config_name = "lewm"
    action_input_dim = 10


class StableWorldModelPLDMReacherArmMassAdapter(
    StableWorldModelLeWMReacherArmMassAdapter
):
    """History-3 PLDM adapter for Reacher arm-mass checkpoints."""

    adapter_id = "stable_worldmodel_pldm_reacher_arm_mass_v1"
    model_config_name = "pldm"


class StableWorldModelLeWMCubeGraspRuleAdapter(
    StableWorldModelLeWMReacherArmMassAdapter
):
    """History-3 LeWM adapter for the Cube grasp-rule benchmark."""

    adapter_id = "stable_worldmodel_lewm_cube_grasp_rule_v1"
    model_config_name = "lewm"
    raw_action_dim = 5
    action_input_dim = 25


class StableWorldModelPLDMCubeGraspRuleAdapter(
    StableWorldModelLeWMCubeGraspRuleAdapter
):
    """History-3 PLDM adapter for the Cube grasp-rule benchmark."""

    adapter_id = "stable_worldmodel_pldm_cube_grasp_rule_v1"
    model_config_name = "pldm"


# These task names remain identity aliases for source and runtime backwards
# compatibility.  Task-specific geometry is validated by each scorer rather
# than encoded in nominal Python subclasses, so an external implementation of
# ``LatentWorldModelAdapter`` can participate in every compatible task.
DoorICLModelAdapter = LatentWorldModelAdapter
ActionDelayICLModelAdapter = LatentWorldModelAdapter
ActionStrengthICLModelAdapter = LatentWorldModelAdapter
ContactFrictionICLModelAdapter = LatentWorldModelAdapter
MotionDampingICLModelAdapter = LatentWorldModelAdapter
PortalExitICLModelAdapter = LatentWorldModelAdapter
ReacherArmMassICLModelAdapter = LatentWorldModelAdapter
CubeGraspRuleICLModelAdapter = LatentWorldModelAdapter


__all__ = [
    "ActionStrengthICLModelAdapter",
    "ContactFrictionICLModelAdapter",
    "MotionDampingICLModelAdapter",
    "PortalExitICLModelAdapter",
    "ReacherArmMassICLModelAdapter",
    "CubeGraspRuleICLModelAdapter",
    "LatentWorldModelAdapter",
    "AdapterProtocol",
    "validate_adapter_protocol",
    "ActionDelayICLModelAdapter",
    "DoorICLModelAdapter",
    "SpeedICLModelAdapter",
    "StableWorldModelLeWMActionStrengthAdapter",
    "StableWorldModelLeWMContactFrictionAdapter",
    "StableWorldModelLeWMAdapter",
    "StableWorldModelLeWMHistory7Adapter",
    "StableWorldModelPLDMActionStrengthAdapter",
    "StableWorldModelPLDMContactFrictionAdapter",
    "StableWorldModelLeWMMotionDampingAdapter",
    "StableWorldModelLeWMPortalExitAdapter",
    "StableWorldModelLeWMReacherArmMassAdapter",
    "StableWorldModelPLDMMotionDampingAdapter",
    "StableWorldModelPLDMPortalExitAdapter",
    "StableWorldModelPLDMReacherArmMassAdapter",
    "StableWorldModelLeWMCubeGraspRuleAdapter",
    "StableWorldModelPLDMCubeGraspRuleAdapter",
    "StableWorldModelPLDMAdapter",
    "StableWorldModelPLDMHistory7Adapter",
]
