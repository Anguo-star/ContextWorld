from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np


CONTEXT_PIXELS_KEY = "pixels_context"
CONTEXT_ACTIONS_KEY = "actions_context"


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(tuple(array.shape)).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


class FixedContextCostModel:
    """Prepend a fixed context prompt without exposing it to CEM.

    CEM still optimizes exactly ``horizon`` action blocks.  The two context
    action blocks are concatenated only inside ``get_cost`` together with the
    corresponding context observations.
    """

    def __init__(self, model: Any, *, history_size: int) -> None:
        import torch

        if not isinstance(model, torch.nn.Module):
            raise TypeError("FixedContextCostModel requires a torch module")
        if history_size < 2:
            raise ValueError("Context planning requires history_size >= 2")
        self.model = model
        self.history_size = int(history_size)
        self.get_cost_calls = 0

    def parameters(self, *args: Any, **kwargs: Any):
        return self.model.parameters(*args, **kwargs)

    def get_cost(self, info_dict: dict[str, Any], action_candidates: Any):
        import torch

        self.get_cost_calls += 1

        model_info = dict(info_dict)
        try:
            context_pixels = model_info.pop(CONTEXT_PIXELS_KEY)
            context_actions = model_info.pop(CONTEXT_ACTIONS_KEY)
        except KeyError as exc:
            raise KeyError(f"Missing fixed context input {exc.args[0]!r}") from exc

        current_pixels = model_info.get("pixels")
        if not all(
            torch.is_tensor(value)
            for value in (context_pixels, context_actions, current_pixels, action_candidates)
        ):
            raise TypeError("Context cost inputs must be torch tensors")
        if context_pixels.ndim != current_pixels.ndim:
            raise ValueError(
                "Context/current pixel ranks differ: "
                f"{context_pixels.shape} vs {current_pixels.shape}"
            )
        if context_actions.ndim != action_candidates.ndim:
            raise ValueError(
                "Context/candidate action ranks differ: "
                f"{context_actions.shape} vs {action_candidates.shape}"
            )
        if context_pixels.shape[:2] != current_pixels.shape[:2]:
            raise ValueError("Context/current pixel batch dimensions differ")
        if context_actions.shape[:2] != action_candidates.shape[:2]:
            raise ValueError("Context/candidate action batch dimensions differ")
        expected_context = self.history_size - 1
        if context_pixels.shape[2] != expected_context:
            raise ValueError(
                f"Expected {expected_context} context observations, got "
                f"{context_pixels.shape[2]}"
            )
        if context_actions.shape[2] != expected_context:
            raise ValueError(
                f"Expected {expected_context} fixed context actions, got "
                f"{context_actions.shape[2]}"
            )
        if current_pixels.shape[2] != 1:
            raise ValueError(
                "Paired context planning expects one live query observation, "
                f"got {current_pixels.shape[2]}"
            )

        model_info["pixels"] = torch.cat(
            [context_pixels, current_pixels], dim=2
        )
        prompted_actions = torch.cat(
            [context_actions, action_candidates], dim=2
        )
        return self.model.get_cost(model_info, prompted_actions)


class FixedContextPolicy:
    """Add the same per-query prompt at every MPC replan."""

    def __init__(
        self,
        policy: Any,
        *,
        context_pixels: np.ndarray,
        context_actions: np.ndarray,
        trace_steps: list[dict[str, Any]] | None = None,
    ) -> None:
        pixels = np.asarray(context_pixels, dtype=np.uint8)
        actions = np.asarray(context_actions, dtype=np.float32)
        if pixels.ndim != 5 or pixels.shape[1] != 2:
            raise ValueError(
                f"Expected context pixels (N,2,H,W,C), got {pixels.shape}"
            )
        if actions.ndim != 3 or actions.shape[1:] != (2, 10):
            raise ValueError(
                f"Expected normalized context actions (N,2,10), got {actions.shape}"
            )
        if pixels.shape[0] != actions.shape[0]:
            raise ValueError("Context pixel/action query counts differ")
        self.policy = policy
        self.context_pixels = pixels.copy()
        self.context_actions = actions.copy()
        self.trace_steps = trace_steps
        self.type = "fixed_context_world_model"
        self.env = None

    def set_env(self, env: Any) -> None:
        if env.num_envs != self.context_pixels.shape[0]:
            raise ValueError(
                f"Context prompt has {self.context_pixels.shape[0]} rows, "
                f"world has {env.num_envs} envs"
            )
        self.env = env
        self.policy.set_env(env)

    def get_action(self, info_dict: dict[str, Any], **kwargs: Any) -> np.ndarray:
        augmented = dict(info_dict)
        augmented[CONTEXT_PIXELS_KEY] = self.context_pixels
        augmented[CONTEXT_ACTIONS_KEY] = self.context_actions
        actions = self.policy.get_action(augmented, **kwargs)
        if self.trace_steps is not None:
            state_value = info_dict.get("state", info_dict.get("proprio"))
            if state_value is None:
                raise KeyError("Trajectory tracing requires state or proprio")
            states = np.asarray(state_value)
            state = states[0, -1] if states.ndim >= 3 else states[0]
            action = np.asarray(actions)[0]
            self.trace_steps.append(
                {
                    "state": np.asarray(state, dtype=np.float32).tolist(),
                    "action": np.asarray(action, dtype=np.float32).tolist(),
                }
            )
        return actions


class RollingContextPolicy:
    """Supply a causal, live History-3 prompt at every MPC replan.

    The initial solve uses the catalog's two contiguous context transitions.
    Later solves use the two most recent complete action blocks and the live
    observations at their boundaries.  This keeps the final context action
    aligned with the current observation instead of reconnecting a stale
    episode prefix to a new state.

    ``WorldModelPolicy`` asks for one raw action on every environment step but
    invokes CEM only when its internal action buffer is empty.  This wrapper
    observes every pre-action frame and returned raw action, and replaces the
    context only at those replan boundaries.
    """

    def __init__(
        self,
        policy: Any,
        *,
        initial_context_pixels: np.ndarray,
        initial_context_raw_actions: np.ndarray,
        initial_context_normalized_actions: np.ndarray,
        initial_query_pixels: np.ndarray,
        action_block: int,
        action_transform: Any,
        trace_steps: list[dict[str, Any]] | None = None,
    ) -> None:
        pixels = np.asarray(initial_context_pixels, dtype=np.uint8)
        raw_actions = np.asarray(
            initial_context_raw_actions, dtype=np.float32
        )
        normalized_actions = np.asarray(
            initial_context_normalized_actions, dtype=np.float32
        )
        query_pixels = np.asarray(initial_query_pixels, dtype=np.uint8)
        block = int(action_block)
        if block <= 0:
            raise ValueError("action_block must be positive")
        if pixels.ndim != 5 or pixels.shape[1] != 2:
            raise ValueError(
                "Expected initial context pixels (N,2,H,W,C), got "
                f"{pixels.shape}"
            )
        if raw_actions.ndim != 4 or raw_actions.shape[1:3] != (2, block):
            raise ValueError(
                "Expected initial raw context actions (N,2,B,A), got "
                f"{raw_actions.shape}"
            )
        action_dim = int(raw_actions.shape[-1])
        expected_normalized = (pixels.shape[0], 2, block * action_dim)
        if normalized_actions.shape != expected_normalized:
            raise ValueError(
                "Expected initial normalized actions "
                f"{expected_normalized}, got {normalized_actions.shape}"
            )
        if query_pixels.shape != (pixels.shape[0], *pixels.shape[2:]):
            raise ValueError(
                "Expected initial query pixels (N,H,W,C) matching context, "
                f"got {query_pixels.shape}"
            )
        if not callable(getattr(action_transform, "transform", None)):
            raise TypeError("action_transform must expose transform(array)")

        self.policy = policy
        self.initial_context_pixels = pixels.copy()
        self.initial_context_raw_actions = raw_actions.copy()
        self.initial_context_normalized_actions = normalized_actions.copy()
        self.initial_query_pixels = query_pixels.copy()
        self.action_block = block
        self.action_dim = action_dim
        self.action_transform = action_transform
        self.trace_steps = trace_steps
        self.type = "rolling_context_world_model"
        self.env = None

        self._raw_step = 0
        self._observed_pixels: list[np.ndarray] = []
        self._executed_raw_actions: list[np.ndarray] = []
        self._active_context_pixels = self.initial_context_pixels.copy()
        self._active_context_normalized_actions = (
            self.initial_context_normalized_actions.copy()
        )
        self._context_uses: list[dict[str, Any]] = []

    def set_env(self, env: Any) -> None:
        if env.num_envs != self.initial_context_pixels.shape[0]:
            raise ValueError(
                "Initial context query count differs from world env count"
            )
        self.env = env
        self.policy.set_env(env)

    def _replan_due(self) -> bool:
        buffers = getattr(self.policy, "_action_buffer", None)
        if buffers is None:
            raise RuntimeError(
                "RollingContextPolicy requires WorldModelPolicy action buffers"
            )
        empty = [len(buffer) == 0 for buffer in buffers]
        if not (all(empty) or not any(empty)):
            raise RuntimeError(
                "Rolling context currently requires synchronized replans"
            )
        return all(empty)

    @staticmethod
    def _current_pixels(info_dict: dict[str, Any]) -> np.ndarray:
        value = np.asarray(info_dict.get("pixels"))
        if value.ndim != 5 or value.shape[1] != 1 or value.shape[-1] != 3:
            raise ValueError(
                "Rolling context requires live pixels (N,1,H,W,C), got "
                f"{value.shape}"
            )
        return np.asarray(value[:, -1], dtype=np.uint8).copy()

    @staticmethod
    def _current_states(info_dict: dict[str, Any]) -> np.ndarray | None:
        value = info_dict.get("state", info_dict.get("proprio"))
        if value is None:
            return None
        states = np.asarray(value, dtype=np.float32)
        if states.ndim < 2:
            raise ValueError(f"Unexpected live state shape {states.shape}")
        return (states[:, -1] if states.ndim >= 3 else states).copy()

    def _rolling_context(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        end = int(self._raw_step)
        middle = end - self.action_block
        start = middle - self.action_block
        if start < 0 or len(self._observed_pixels) != end:
            raise RuntimeError(
                "A rolling replan occurred before two complete action blocks "
                f"were available: raw_step={end}"
            )
        pixels = np.stack(
            [self._observed_pixels[start], self._observed_pixels[middle]],
            axis=1,
        ).astype(np.uint8, copy=False)
        first = np.stack(
            self._executed_raw_actions[start:middle], axis=1
        )
        second = np.stack(
            self._executed_raw_actions[middle:end], axis=1
        )
        raw_actions = np.stack([first, second], axis=1).astype(
            np.float32, copy=False
        )
        normalized = np.asarray(
            self.action_transform.transform(
                raw_actions.reshape(-1, self.action_dim)
            ),
            dtype=np.float32,
        ).reshape(
            raw_actions.shape[0],
            2,
            self.action_block * self.action_dim,
        )
        return pixels, raw_actions, normalized

    def _activate_context(
        self,
        *,
        current_pixels: np.ndarray,
        current_states: np.ndarray | None,
    ) -> None:
        if self._raw_step == 0:
            if not np.array_equal(current_pixels, self.initial_query_pixels):
                raise RuntimeError(
                    "Initial live query pixels differ from the frozen catalog"
                )
            source = "catalog_initial_contiguous_history"
            pixels = self.initial_context_pixels
            raw_actions = self.initial_context_raw_actions
            normalized = self.initial_context_normalized_actions
            frame_steps = [-2 * self.action_block, -self.action_block]
        else:
            source = "rolling_live_environment_history"
            pixels, raw_actions, normalized = self._rolling_context()
            frame_steps = [
                self._raw_step - 2 * self.action_block,
                self._raw_step - self.action_block,
            ]
        self._active_context_pixels = np.asarray(pixels).copy()
        self._active_context_normalized_actions = np.asarray(
            normalized
        ).copy()
        current_step = int(self._raw_step)
        self._context_uses.append(
            {
                "replan_index": len(self._context_uses),
                "source": source,
                "current_raw_step": current_step,
                "context_frame_raw_steps": frame_steps,
                "context_action_raw_step_ranges": [
                    [frame_steps[0], frame_steps[1]],
                    [frame_steps[1], current_step],
                ],
                "context_pixels_sha256": _array_sha256(pixels),
                "context_raw_actions_sha256": _array_sha256(raw_actions),
                "context_normalized_actions_sha256": _array_sha256(
                    normalized
                ),
                "current_pixels_sha256": _array_sha256(current_pixels),
                "current_states_sha256": (
                    None
                    if current_states is None
                    else _array_sha256(current_states)
                ),
                "causal_alignment_passed": bool(
                    frame_steps[0] + self.action_block == frame_steps[1]
                    and frame_steps[1] + self.action_block == current_step
                ),
            }
        )

    def get_action(self, info_dict: dict[str, Any], **kwargs: Any) -> np.ndarray:
        current_pixels = self._current_pixels(info_dict)
        current_states = self._current_states(info_dict)
        if self._replan_due():
            self._activate_context(
                current_pixels=current_pixels,
                current_states=current_states,
            )

        augmented = dict(info_dict)
        augmented[CONTEXT_PIXELS_KEY] = self._active_context_pixels
        augmented[CONTEXT_ACTIONS_KEY] = (
            self._active_context_normalized_actions
        )
        actions = np.asarray(
            self.policy.get_action(augmented, **kwargs), dtype=np.float32
        )
        expected = (self.initial_context_pixels.shape[0], self.action_dim)
        if actions.shape != expected:
            raise ValueError(
                f"Expected returned raw actions {expected}, got {actions.shape}"
            )

        self._observed_pixels.append(current_pixels)
        self._executed_raw_actions.append(actions.copy())
        if self.trace_steps is not None:
            if current_states is None:
                raise KeyError("Trajectory tracing requires state or proprio")
            if current_states.shape[0] != 1:
                raise ValueError("Trajectory tracing currently requires one env")
            self.trace_steps.append(
                {
                    "state": current_states[0].tolist(),
                    "action": actions[0].tolist(),
                }
            )
        self._raw_step += 1
        return actions

    def runtime_audit(self) -> dict[str, Any]:
        rolling = [
            row
            for row in self._context_uses
            if row["source"] == "rolling_live_environment_history"
        ]
        passed = bool(
            self._context_uses
            and all(row["causal_alignment_passed"] for row in self._context_uses)
        )
        return {
            "passed": passed,
            "mode": "rolling_live_history3",
            "action_block_raw_steps": self.action_block,
            "raw_steps_observed": self._raw_step,
            "replans": len(self._context_uses),
            "rolling_replans": len(rolling),
            "all_replans_causally_aligned": passed,
            "uses": list(self._context_uses),
        }


@dataclass(frozen=True)
class QueryEpisode:
    query_id: str
    scenario_id: str
    template_id: str
    speed: float
    door_position: int
    simulator_seed: int
    query_pixels: np.ndarray
    goal_pixels: np.ndarray
    query_state: np.ndarray
    goal_state: np.ndarray


class PairedQueryDataset:
    """Minimal dataset interface consumed by StableWM's dataset evaluator."""

    def __init__(self, episodes: list[QueryEpisode]) -> None:
        import torch

        from .tworoom import DOOR_COLUMN, SPEED_COLUMN

        if not episodes:
            raise ValueError("At least one query episode is required")
        self.query_ids = [episode.query_id for episode in episodes]
        self.scenario_ids = [episode.scenario_id for episode in episodes]
        self.speeds = [episode.speed for episode in episodes]
        self.column_names = [
            "pixels",
            "action",
            "proprio",
            "state",
            "goal_state",
            SPEED_COLUMN,
            DOOR_COLUMN,
        ]
        self.lengths = np.full(len(episodes), 2, dtype=np.int64)
        self.offsets = np.arange(len(episodes) + 1, dtype=np.int64) * 2
        self._episodes: list[dict[str, torch.Tensor]] = []
        for episode in episodes:
            query_pixels = np.asarray(episode.query_pixels, dtype=np.uint8)
            goal_pixels = np.asarray(episode.goal_pixels, dtype=np.uint8)
            if query_pixels.shape != goal_pixels.shape or query_pixels.ndim != 3:
                raise ValueError("Query/goal pixels must be matching HWC images")
            pixels = np.stack([query_pixels, goal_pixels]).transpose(0, 3, 1, 2)
            states = np.stack(
                [episode.query_state, episode.goal_state]
            ).astype(np.float32)
            self._episodes.append(
                {
                    "pixels": torch.from_numpy(pixels.copy()),
                    "action": torch.zeros((2, 2), dtype=torch.float32),
                    "proprio": torch.from_numpy(states.copy()),
                    "state": torch.from_numpy(states.copy()),
                    "goal_state": torch.from_numpy(
                        np.repeat(episode.goal_state[None], 2, axis=0).astype(
                            np.float32
                        )
                    ),
                    SPEED_COLUMN: torch.full(
                        (2, 1), float(episode.speed), dtype=torch.float32
                    ),
                    DOOR_COLUMN: torch.full(
                        (2, 3), int(episode.door_position), dtype=torch.int64
                    ),
                }
            )

    def load_chunk(
        self,
        episodes_idx: np.ndarray,
        start: np.ndarray,
        end: np.ndarray,
    ) -> list[dict[str, Any]]:
        results = []
        for episode, begin, stop in zip(episodes_idx, start, end):
            index = int(episode)
            begin_value = int(begin)
            stop_value = int(stop)
            results.append(
                {
                    key: value[begin_value:stop_value].clone()
                    for key, value in self._episodes[index].items()
                }
            )
        return results

    def get_row_data(self, row_index: int) -> dict[str, np.ndarray]:
        episode = int(row_index) // 2
        step = int(row_index) % 2
        return {
            key: value[step].detach().cpu().numpy()
            for key, value in self._episodes[episode].items()
        }


__all__ = [
    "CONTEXT_ACTIONS_KEY",
    "CONTEXT_PIXELS_KEY",
    "FixedContextCostModel",
    "FixedContextPolicy",
    "RollingContextPolicy",
    "PairedQueryDataset",
    "QueryEpisode",
]
