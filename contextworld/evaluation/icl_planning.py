from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


CONTEXT_PIXELS_KEY = "pixels_context"
CONTEXT_ACTIONS_KEY = "actions_context"


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
    "PairedQueryDataset",
    "QueryEpisode",
]
