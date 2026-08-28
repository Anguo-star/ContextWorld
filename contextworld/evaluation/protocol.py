from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from contextworld.paths import resolve_contextworld_path

from .tworoom import (
    DOOR_COLUMN,
    SPEED_COLUMN,
    compile_tworoom_eval_variations,
    validate_tworoom_factor_columns,
)


@dataclass(frozen=True)
class EvaluationStarts:
    episodes: list[int]
    steps: list[int]


class ColumnStandardizer:
    """Training-stat standardizer with sklearn-compatible methods."""

    def __init__(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        if np.any(~np.isfinite(self.mean)) or np.any(~np.isfinite(self.std)):
            raise ValueError("Non-finite normalization statistics")
        if np.any(self.std <= 0):
            raise ValueError(f"Normalization std must be positive: {self.std}")

    def transform(self, value: np.ndarray) -> np.ndarray:
        return (np.asarray(value) - self.mean) / self.std

    def inverse_transform(self, value: np.ndarray) -> np.ndarray:
        return np.asarray(value) * self.std + self.mean


def original_h5_process(path: Path) -> dict[str, ColumnStandardizer]:
    """Reproduce training normalization from the read-only original H5.

    Training used ``torch.mean`` and the default unbiased ``torch.std``.
    Test/OOD rows are never used to estimate these statistics.
    """

    import h5py
    import torch

    process: dict[str, ColumnStandardizer] = {}
    with h5py.File(path, "r") as handle:
        for column in ("action", "proprio"):
            values = torch.from_numpy(np.asarray(handle[column]))
            valid = values[~torch.isnan(values).any(dim=1)]
            mean = valid.mean(0, keepdim=True).numpy()
            std = valid.std(0, keepdim=True).numpy()
            process[column] = ColumnStandardizer(mean, std)
            if column != "action":
                process[f"goal_{column}"] = process[column]
    return process


def frozen_normalizer_process(path: Path) -> dict[str, ColumnStandardizer]:
    """Load the preregistered original-train normalization artifact."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("protocol") != "tworoom_original_train_s3072_unbiased_zscore_v1":
        raise ValueError(f"Unsupported frozen normalizer protocol in {path}")
    if payload.get("statistics_scope") != "original_9000_train_episodes_only":
        raise ValueError(f"Frozen normalizer has an invalid statistics scope: {path}")
    process: dict[str, ColumnStandardizer] = {}
    for column in ("action", "proprio"):
        values = payload["columns"][column]
        process[column] = ColumnStandardizer(
            np.asarray(values["mean"], dtype=np.float32)[None],
            np.asarray(values["std_unbiased"], dtype=np.float32)[None],
        )
        if column != "action":
            process[f"goal_{column}"] = process[column]
    return process


def load_catalog_regime(
    catalog_path: Path,
    regime: str,
    *,
    repo_root: Path,
) -> list[Path]:
    with catalog_path.open("r", encoding="utf-8") as handle:
        catalog = json.load(handle)
    try:
        entries = catalog["by_regime"][regime]
    except KeyError as exc:
        available = sorted(catalog.get("by_regime", {}))
        raise KeyError(
            f"Unknown regime {regime!r} in {catalog_path}; available={available}"
        ) from exc
    paths = []
    for entry in entries:
        path = resolve_contextworld_path(entry, repo_root=repo_root)
        if not path.is_dir():
            raise FileNotFoundError(path)
        paths.append(path)
    return sorted(paths)


def select_episode_balanced_starts(
    lengths: np.ndarray,
    *,
    goal_offset: int,
    count: int,
    seed: int,
) -> EvaluationStarts:
    """Select unique starts while balancing episode reuse in rounds.

    This prevents longer trajectories from receiving more evaluation weight.
    Every eligible episode is used once before an episode is used again.  A
    particular ``(episode, start_step)`` pair is never repeated.
    """

    lengths = np.asarray(lengths, dtype=np.int64)
    eligible = np.flatnonzero(lengths >= goal_offset + 1)
    if count <= 0:
        raise ValueError("Evaluation count must be positive")
    if not len(eligible):
        raise ValueError(
            f"No eligible episodes have at least {goal_offset + 1} rows"
        )

    available_starts = {
        int(episode): int(lengths[episode]) - goal_offset
        for episode in eligible
    }
    total_unique_starts = sum(available_starts.values())
    if count > total_unique_starts:
        raise ValueError(
            f"Need {count} unique evaluation starts, but only "
            f"{total_unique_starts} satisfy goal_offset={goal_offset}"
        )

    rng = np.random.default_rng(seed)
    remaining_steps = {
        episode: list(rng.permutation(start_count).astype(int))
        for episode, start_count in available_starts.items()
    }
    episodes: list[int] = []
    steps: list[int] = []
    while len(episodes) < count:
        active = [
            episode for episode in eligible if remaining_steps[int(episode)]
        ]
        for episode_value in rng.permutation(active):
            episode = int(episode_value)
            episodes.append(episode)
            steps.append(int(remaining_steps[episode].pop()))
            if len(episodes) == count:
                break
    return EvaluationStarts(
        episodes=episodes,
        steps=steps,
    )


def allocate_scenario_evaluations(
    *,
    scenario_count: int,
    total_evaluations: int,
    seed: int,
) -> list[int]:
    """Split a fixed total budget as evenly as possible across scenarios."""

    if scenario_count <= 0:
        raise ValueError("Scenario count must be positive")
    if total_evaluations < scenario_count:
        raise ValueError(
            "Total evaluations must cover every selected scenario at least once: "
            f"total_evaluations={total_evaluations}, scenarios={scenario_count}"
        )
    quotient, remainder = divmod(total_evaluations, scenario_count)
    counts = np.full(scenario_count, quotient, dtype=np.int64)
    if remainder:
        rng = np.random.default_rng(seed)
        counts[rng.permutation(scenario_count)[:remainder]] += 1
    return [int(value) for value in counts]


def scenario_seed(seed: int, scenario_path: Path) -> int:
    digest = hashlib.sha256(scenario_path.name.encode("utf-8")).digest()
    scenario_part = int.from_bytes(digest[:4], "little")
    return int(np.random.SeedSequence([seed, scenario_part]).generate_state(1)[0])


def load_legacy_cost_model(checkpoint: Path, legacy_code_root: Path):
    """Load an object checkpoint and locate the module exposing ``get_cost``."""

    import torch

    legacy = str(legacy_code_root.resolve())
    if legacy not in sys.path:
        sys.path.insert(0, legacy)
    loaded = torch.load(checkpoint, map_location="cpu", weights_only=False)

    visited: set[int] = set()

    def scan(module: Any):
        identity = id(module)
        if identity in visited:
            return None
        visited.add(identity)
        if hasattr(module, "get_cost"):
            return module
        children = getattr(module, "children", None)
        if children is None:
            return None
        for child in children():
            result = scan(child)
            if result is not None:
                return result
        return None

    model = scan(loaded)
    if model is None:
        raise RuntimeError(f"No get_cost module found in {checkpoint}")
    return model


def load_pretrained_cost_model(
    checkpoint: Path,
    stable_worldmodel: Any,
    *,
    cache_dir: Path,
):
    """Load StableWM's native ``.pt + config.json`` checkpoint format."""

    checkpoint = checkpoint.resolve()
    if checkpoint.suffix.lower() != ".pt":
        raise ValueError(f"Expected a StableWM .pt checkpoint, got {checkpoint}")
    if not (checkpoint.parent / "config.json").is_file():
        raise FileNotFoundError(checkpoint.parent / "config.json")
    model = stable_worldmodel.wm.utils.load_pretrained(
        str(checkpoint), cache_dir=str(cache_dir.resolve())
    )
    # Stable-WorldModel's current LeWM/PLDM API deliberately separates the
    # latent dynamics from planner objectives: the checkpoint exposes
    # ``encode``/``rollout``, while ``ShootingCostEvaluator`` supplies
    # ``get_cost`` only for MPC.  ContextWorld's ICL adapters exercise the raw
    # dynamics and therefore must accept both the legacy monolithic surface
    # and the current compositional one.
    if not hasattr(model, "get_cost") and not (
        hasattr(model, "encode") and hasattr(model, "rollout")
    ):
        raise RuntimeError(
            "Loaded checkpoint exposes neither the legacy get_cost API nor "
            f"the current encode/rollout dynamics API: {checkpoint}"
        )
    return model


def infer_model_protocol(model: Any, action_dim: int = 2) -> dict[str, int]:
    patch_embed = getattr(getattr(model, "action_encoder", None), "patch_embed", None)
    in_channels = getattr(patch_embed, "in_channels", None)
    if in_channels is None or int(in_channels) % action_dim:
        raise ValueError("Cannot infer action block from model.action_encoder")
    action_block = int(in_channels) // action_dim

    history_size = None
    for attribute in ("history_size", "history_len"):
        value = getattr(model, attribute, None)
        if value is not None:
            history_size = int(value)
            break
    if history_size is None:
        position = getattr(getattr(model, "predictor", None), "pos_embedding", None)
        if position is not None and getattr(position, "ndim", 0) >= 2:
            history_size = int(position.shape[1])
    if history_size is None:
        raise ValueError("Cannot infer model history size")
    return {"action_block": action_block, "history_size": history_size}


def factor_readback_audit(
    dataset: Any,
    world: Any,
    starts: EvaluationStarts,
) -> dict[str, Any]:
    columns = validate_tworoom_factor_columns(dataset.column_names)
    failures: list[dict[str, Any]] = []
    checked = 0
    for env_index, (episode, step) in enumerate(
        zip(starts.episodes, starts.steps)
    ):
        row_index = int(dataset.offsets[episode]) + int(step)
        row = dataset.get_row_data(row_index)
        expected = compile_tworoom_eval_variations(
            agent_speed=(row[SPEED_COLUMN][0] if SPEED_COLUMN in columns else None),
            door_position=(row[DOOR_COLUMN][0] if DOOR_COLUMN in columns else None),
        )
        env = world.envs.envs[env_index].unwrapped
        observed = getattr(env, "_contextworld_variation_readback", None)
        if observed is None:
            failures.append(
                {"env_index": env_index, "reason": "missing_readback"}
            )
            continue
        for factor, value in expected.items():
            checked += 1
            actual = observed.get(factor)
            if actual is None or not np.array_equal(actual, value):
                failures.append(
                    {
                        "env_index": env_index,
                        "factor": factor,
                        "expected": np.asarray(value).tolist(),
                        "observed": (
                            None if actual is None else np.asarray(actual).tolist()
                        ),
                    }
                )
    return {"passed": not failures and checked > 0, "checked": checked, "failures": failures}
