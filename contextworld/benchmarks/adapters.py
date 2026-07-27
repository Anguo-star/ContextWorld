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


@dataclass(frozen=True)
class AdapterProtocol:
    """Input/output contract required by the Speed ICL evaluator."""

    history_tokens: int
    action_block_raw_steps: int
    action_dim: int
    future_action_blocks: int
    native_target_encoder: bool = True


class SpeedICLModelAdapter(ABC):
    """Model-independent boundary used by the public Speed ICL scorer.

    Adapters receive raw uint8 RGB frames and raw environment actions.  This
    keeps model-specific image transforms, action normalization and latent
    representation choices outside the benchmark dataset.
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
        """Return future latents for a History-3 input and raw action blocks."""

    @abstractmethod
    def frozen_state_hash(self) -> str:
        """Hash model parameters and buffers without changing model state."""


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


class StableWorldModelLeWMAdapter(SpeedICLModelAdapter):
    """Tested adapter for the pinned Stable-WorldModel LeWM checkpoint format."""

    adapter_id = "stable_worldmodel_lewm_v1"

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
        setattr(self.model, "history_size", 3)
        setattr(self.model, "interpolate_pos_encoding", True)
        inferred = infer_model_protocol(self.model, action_dim=2)
        self._protocol = AdapterProtocol(
            history_tokens=int(inferred["history_size"]),
            action_block_raw_steps=int(inferred["action_block"]),
            action_dim=2,
            future_action_blocks=5,
        )
        if self._protocol.history_tokens != 3:
            raise RuntimeError(
                f"Speed ICL v1 requires History-3, got {self._protocol}"
            )
        if self._protocol.action_block_raw_steps != 5:
            raise RuntimeError(
                f"Speed ICL v1 requires action block 5, got {self._protocol}"
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
        model = load_pretrained_cost_model(
            checkpoint,
            swm,
            cache_dir=artifact_path("evaluation/model_cache", repo_root=repo_root),
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
                result = self.model.rollout(
                    {"pixels": transformed[:, None]},
                    action_tensor[:, None],
                    history_size=self.protocol.history_tokens,
                )["predicted_emb"][:, 0]
                predicted = result[:, self.protocol.history_tokens :]
                # ``action_tensor`` has one action token per transition.  A
                # History-3 input consumes two context actions, so T actions
                # request T-2 future predictions.
                expected_future = action_tensor.shape[1] - 2
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


# Door and Speed use the same raw-pixel/action interface.  This public alias
# lets integrations describe their purpose without inheriting a speed-named
# type; existing Speed integrations remain fully compatible.
DoorICLModelAdapter = SpeedICLModelAdapter


__all__ = [
    "AdapterProtocol",
    "DoorICLModelAdapter",
    "SpeedICLModelAdapter",
    "StableWorldModelLeWMAdapter",
    "StableWorldModelPLDMAdapter",
]
