from __future__ import annotations

import hashlib
import io
import json
import math
import multiprocessing
import os
import shutil
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
from PIL import Image

from contextworld.paths import (
    portable_contextworld_path,
    resolve_contextworld_path,
)
from contextworld.synthesis.lance import build_lance_writer, encode_frame
from contextworld.synthesis.manifest import write_json
from contextworld.synthesis.stablewm import load_stable_worldmodel
from contextworld.synthesis.validator import validate_loader_mix

from .action_delay import array_sha256, canonical_sha256
from .action_delay_env import (
    ACTION_DELAY_FACTOR,
    make_extended_action_delay_env,
)
from .action_delay_h7_validation import file_sha256
from .action_delay_long_history import (
    ACTION_BLOCK,
    LongHistoryDelayTemplate,
    simulate_template,
)


H7_ACTION_DELAY_ENV_ID = "contextworld/TwoRoomActionDelayH7-v1"
HISTORY_TOKENS = 7
NUM_PREDS = 1
TRAIN_SEQUENCE_STEPS = HISTORY_TOKENS + NUM_PREDS
EPISODE_MODEL_FRAMES = 10
RAW_STEPS = EPISODE_MODEL_FRAMES * ACTION_BLOCK
DEFAULT_CLIPS_PER_EPISODE = RAW_STEPS - (
    TRAIN_SEQUENCE_STEPS * ACTION_BLOCK
) + 1
FORMAL_CLIP_STARTS = (0,)
SINGLE_DELAY = 4
MULTI_DELAYS = (0, 4, 8)
MAXIMUM_DELAY_STEPS = 10
GROUPS = ("action_delay_single", "action_delay_multi")
SPLITS = ("train", "val")
MODEL_KEYS = ("pixels", "action")
DIAGNOSTIC_KEYS = ("pixels", "action", "proprio")
WATCHED_VARIATIONS = ("agent.speed", ACTION_DELAY_FACTOR)
REQUIRED_COLUMNS = {
    "pixels",
    "action",
    "proprio",
    "state",
    "goal_state",
    "terminated",
    "truncated",
    "variation_agent_speed",
    "variation_action_delay_steps",
}


@dataclass(frozen=True)
class ActionDelayH7EpisodePlan:
    template: LongHistoryDelayTemplate
    delay_steps: int
    split: str
    shard_index: int
    episode_index: int


@dataclass(frozen=True)
class ActionDelayH7ShardPlan:
    group: str
    split: str
    shard_index: int
    delay_steps: int
    scenario_id: str
    fingerprint: str
    table_path: Path
    episodes: tuple[ActionDelayH7EpisodePlan, ...]


_WORKER_SWM: Any | None = None
_WORKER_CONFIG: dict[str, Any] | None = None


def directory_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    root = Path(path)
    for value in sorted(
        (item for item in root.rglob("*") if item.is_file()),
        key=lambda item: item.relative_to(root).as_posix(),
    ):
        digest.update(value.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with value.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def make_action_delay_h7_env(**kwargs: Any):
    return make_extended_action_delay_env(
        max_delay_steps=MAXIMUM_DELAY_STEPS,
        **kwargs,
    )


def register_action_delay_h7_env(
    env_id: str = H7_ACTION_DELAY_ENV_ID,
) -> str:
    import gymnasium as gym

    if env_id in gym.registry:
        entry_point = gym.spec(env_id).entry_point
        accepted = {
            make_action_delay_h7_env,
            (
                "contextworld.evaluation.action_delay_h7_data:"
                "make_action_delay_h7_env"
            ),
        }
        if entry_point in accepted:
            return env_id
        raise RuntimeError(
            f"Gym id {env_id!r} is registered by another entry point"
        )
    gym.register(id=env_id, entry_point=make_action_delay_h7_env)
    return env_id


def _validate_config(config: dict[str, Any]) -> dict[str, bool]:
    protocol = config["protocol"]
    counts = config["counts"]
    checks = {
        "history_tokens_are_seven": int(protocol["history_tokens"])
        == HISTORY_TOKENS,
        "num_preds_is_one": int(protocol["num_preds"]) == NUM_PREDS,
        "action_block_is_five": int(
            protocol["raw_steps_per_action_block"]
        )
        == ACTION_BLOCK,
        "episode_has_ten_model_frames": int(
            protocol["episode_model_frames"]
        )
        == EPISODE_MODEL_FRAMES,
        "episode_has_fifty_raw_rows": int(protocol["rows_per_episode"])
        == RAW_STEPS,
        "strict_clip_starts_at_zero": int(
            protocol["strict_training_clip_start_raw_step"]
        )
        == FORMAL_CLIP_STARTS[0],
        "default_reader_would_produce_eleven_clips": int(
            protocol["default_reader_clip_count_before_filter"]
        )
        == DEFAULT_CLIPS_PER_EPISODE,
        "formal_reader_keeps_one_clip": int(
            protocol["formal_training_clips_per_episode"]
        )
        == len(FORMAL_CLIP_STARTS),
        "single_delay_is_four": int(protocol["single_control_delay"])
        == SINGLE_DELAY,
        "multi_delays_are_zero_four_eight": tuple(
            map(int, protocol["training_delay_values"])
        )
        == MULTI_DELAYS,
        "model_fields_are_pixels_and_action": tuple(
            protocol["model_visible_fields"]
        )
        == MODEL_KEYS,
        "train_shards_balance_three_delays": int(
            counts["train"]["shards"]
        )
        % len(MULTI_DELAYS)
        == 0,
        "val_shards_balance_three_delays": int(counts["val"]["shards"])
        % len(MULTI_DELAYS)
        == 0,
    }
    if not all(checks.values()):
        failed = [name for name, value in checks.items() if not value]
        raise ValueError(f"Invalid H7 training-data protocol: {failed}")
    return checks


@lru_cache(maxsize=None)
def _coordinate_permutation(
    catalog_seed: int,
    room: str,
    direction: str,
) -> tuple[tuple[float, float], ...]:
    if room == "left":
        x_values = np.arange(28.25, 92.0, 0.5, dtype=np.float32)
    elif room == "right":
        x_values = np.arange(133.25, 197.0, 0.5, dtype=np.float32)
    else:
        raise ValueError(f"Unknown room {room!r}")
    if direction == "up":
        y_values = np.arange(30.0, 145.5, 0.5, dtype=np.float32)
    elif direction == "down":
        y_values = np.arange(78.25, 194.0, 0.5, dtype=np.float32)
    else:
        raise ValueError(f"Unknown direction {direction!r}")
    values = [
        (float(x_position), float(y_position))
        for x_position in x_values
        for y_position in y_values
    ]
    rng = np.random.default_rng(
        np.random.SeedSequence(
            [
                int(catalog_seed),
                0 if room == "left" else 1,
                0 if direction == "up" else 1,
                0xC001,
            ]
        )
    )
    permutation = rng.permutation(len(values))
    return tuple(values[index] for index in permutation)


def training_template(
    *,
    catalog_seed: int,
    split: str,
    shard_index: int,
    episode_index: int,
) -> LongHistoryDelayTemplate:
    """Return one unique, paired geometry outside the Validation grid."""

    if split not in SPLITS:
        raise ValueError(f"Unknown H7 action-delay split {split!r}")
    global_index = int(shard_index) * 160 + int(episode_index)
    direction = "up" if global_index % 2 == 0 else "down"
    room = "left" if (global_index // 2) % 2 == 0 else "right"
    local_index = global_index // 4
    split_offset = 0 if split == "train" else 10_000
    pool = _coordinate_permutation(catalog_seed, room, direction)
    coordinate_index = split_offset + local_index
    if coordinate_index >= len(pool):
        raise RuntimeError(
            "H7 action-delay coordinate pool is too small for the "
            f"configured shard count: {coordinate_index} >= {len(pool)}"
        )
    reset_state = pool[coordinate_index]
    y_position = reset_state[1]
    goal_state = (
        (190.0, 200.0 if y_position < 112.0 else 24.0)
        if room == "left"
        else (30.0, 200.0 if y_position < 112.0 else 24.0)
    )
    split_index = SPLITS.index(split)
    simulator_seed = int(
        np.random.SeedSequence(
            [
                int(catalog_seed),
                0xA7D7,
                split_index,
                int(shard_index),
                int(episode_index),
            ]
        ).generate_state(1)[0]
    )
    return LongHistoryDelayTemplate(
        template_id=(
            f"action-delay-h7-{split}-s{shard_index:03d}-"
            f"e{episode_index:03d}"
        ),
        direction=direction,
        reset_state=reset_state,
        goal_state=goal_state,
        simulator_seed=simulator_seed,
    )


def _delay_for_shard(group: str, shard_index: int) -> int:
    if group == "action_delay_single":
        return SINGLE_DELAY
    if group == "action_delay_multi":
        return MULTI_DELAYS[int(shard_index) % len(MULTI_DELAYS)]
    raise ValueError(f"Unknown H7 action-delay group {group!r}")


def build_shard_plans(
    config: dict[str, Any],
    *,
    repo_root: Path,
) -> dict[str, list[ActionDelayH7ShardPlan]]:
    _validate_config(config)
    output_root = resolve_contextworld_path(
        config["output_root"],
        repo_root=repo_root,
    )
    catalog_seed = int(config["catalog_seed"])
    plans: dict[str, list[ActionDelayH7ShardPlan]] = {
        group: [] for group in GROUPS
    }
    for group in GROUPS:
        for split in SPLITS:
            shard_count = int(config["counts"][split]["shards"])
            episodes_per_shard = int(
                config["counts"][split]["episodes_per_shard"]
            )
            if episodes_per_shard != 160:
                raise ValueError(
                    "The frozen H7 geometry indexing requires 160 "
                    "episodes per shard"
                )
            for shard_index in range(shard_count):
                delay_steps = _delay_for_shard(group, shard_index)
                episodes = tuple(
                    ActionDelayH7EpisodePlan(
                        template=training_template(
                            catalog_seed=catalog_seed,
                            split=split,
                            shard_index=shard_index,
                            episode_index=episode_index,
                        ),
                        delay_steps=delay_steps,
                        split=split,
                        shard_index=shard_index,
                        episode_index=episode_index,
                    )
                    for episode_index in range(episodes_per_shard)
                )
                fingerprint = canonical_sha256(
                    {
                        "benchmark": config["benchmark"],
                        "group": group,
                        "split": split,
                        "shard_index": shard_index,
                        "delay_steps": delay_steps,
                        "episode_templates": [
                            asdict(value.template) for value in episodes
                        ],
                    }
                )
                scenario_id = (
                    f"ad-h7-{group.removeprefix('action_delay_')}-"
                    f"{split}-s{shard_index:03d}-{fingerprint[:10]}"
                )
                plans[group].append(
                    ActionDelayH7ShardPlan(
                        group=group,
                        split=split,
                        shard_index=shard_index,
                        delay_steps=delay_steps,
                        scenario_id=scenario_id,
                        fingerprint=fingerprint,
                        table_path=(
                            output_root
                            / "tables"
                            / group
                            / split
                            / f"{scenario_id}.lance"
                        ),
                        episodes=episodes,
                    )
                )
    return plans


def _reference_rollout(
    plan: ActionDelayH7EpisodePlan,
    *,
    config: dict[str, Any],
) -> dict[str, Any]:
    return simulate_template(
        plan.template,
        history_tokens=HISTORY_TOKENS,
        delay_steps=plan.delay_steps,
        agent_speed=float(config["protocol"]["agent_speed"]),
        action_magnitude=float(config["protocol"]["action_magnitude"]),
        maximum_delay_steps=MAXIMUM_DELAY_STEPS,
    )


def _model_blocks(reference: dict[str, Any]) -> np.ndarray:
    blocks = np.concatenate(
        [
            np.asarray(reference["action_blocks"], dtype=np.float32),
            np.zeros((1, ACTION_BLOCK, 2), dtype=np.float32),
        ],
        axis=0,
    )
    if blocks.shape != (EPISODE_MODEL_FRAMES, ACTION_BLOCK, 2):
        raise RuntimeError(f"Unexpected H7 model blocks: {blocks.shape}")
    return blocks


def _collection_actions(reference: dict[str, Any]) -> np.ndarray:
    """Compensate for World.collect's one-row action convention."""

    desired = _model_blocks(reference).reshape(RAW_STEPS, 2)
    value = np.concatenate(
        [np.zeros((1, 2), dtype=np.float32), desired[:-1]],
        axis=0,
    )
    if value.shape != (RAW_STEPS, 2):
        raise RuntimeError(f"Unexpected H7 collection actions: {value.shape}")
    return value


class _ResettableScriptedPolicy:
    def __init__(self) -> None:
        self.actions: np.ndarray | None = None
        self.step = 0
        self.env: Any | None = None

    def set_env(self, env: Any) -> None:
        self.env = env

    def reset_actions(self, actions: np.ndarray) -> None:
        value = np.asarray(actions, dtype=np.float32)
        if value.shape != (RAW_STEPS, 2):
            raise ValueError(
                f"Expected {(RAW_STEPS, 2)} actions, got {value.shape}"
            )
        self.actions = value
        self.step = 0

    def get_action(self, _: dict[str, Any]) -> np.ndarray:
        if self.env is None or self.actions is None:
            raise RuntimeError("H7 scripted policy is not ready")
        if self.step >= RAW_STEPS:
            raise RuntimeError("World requested too many H7 actions")
        action = self.actions[self.step]
        self.step += 1
        return np.repeat(action[None], self.env.num_envs, axis=0)


class _OneEpisodeCapture:
    def __init__(self) -> None:
        self.episodes: list[dict[str, Any]] = []

    def __enter__(self) -> _OneEpisodeCapture:
        return self

    def __exit__(self, *exc: Any) -> None:
        return None

    def write_episodes(
        self,
        episodes: Iterable[dict[str, Any]],
    ) -> None:
        self.episodes.extend(episodes)

    def one(self) -> dict[str, Any]:
        if len(self.episodes) != 1:
            raise RuntimeError(
                f"Expected one H7 episode, got {len(self.episodes)}"
            )
        return self.episodes[0]


def _episode_iterator(
    swm: Any,
    *,
    shard: ActionDelayH7ShardPlan,
    config: dict[str, Any],
) -> Iterator[dict[str, Any]]:
    register_action_delay_h7_env()
    world = swm.World(
        H7_ACTION_DELAY_ENV_ID,
        num_envs=1,
        max_episode_steps=RAW_STEPS,
        image_shape=(224, 224),
        render_mode="rgb_array",
    )
    policy = _ResettableScriptedPolicy()
    world.set_policy(policy)
    speed = float(config["protocol"]["agent_speed"])
    try:
        for plan in shard.episodes:
            reference = _reference_rollout(plan, config=config)
            policy.reset_actions(_collection_actions(reference))
            capture = _OneEpisodeCapture()
            world.collect(
                episodes=1,
                seed=int(plan.template.simulator_seed),
                options={
                    "variation": WATCHED_VARIATIONS,
                    "variation_values": {
                        "agent.speed": np.asarray(
                            [speed],
                            dtype=np.float32,
                        ),
                        ACTION_DELAY_FACTOR: int(plan.delay_steps),
                    },
                    "state": np.asarray(
                        plan.template.reset_state,
                        dtype=np.float32,
                    ),
                    "target_state": np.asarray(
                        plan.template.goal_state,
                        dtype=np.float32,
                    ),
                },
                writer=capture,
                progress=False,
            )
            if policy.step != RAW_STEPS:
                raise RuntimeError(
                    f"{plan.template.template_id}: World used "
                    f"{policy.step} actions"
                )
            yield capture.one()
    finally:
        world.close()


def collect_shard(
    swm: Any,
    *,
    shard: ActionDelayH7ShardPlan,
    config: dict[str, Any],
) -> None:
    if shard.table_path.exists():
        raise FileExistsError(shard.table_path)
    shard.table_path.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(config["collection"]["staging_root"])
    staging_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="contextworld-action-delay-h7-",
        dir=staging_root,
    ) as temporary:
        staged = Path(temporary) / shard.table_path.name
        writer = build_lance_writer(
            swm,
            staged,
            pixel_codec=dict(config["storage"]["pixel_codec"]),
        )
        with writer as opened:
            opened.write_episodes(
                _episode_iterator(
                    swm,
                    shard=shard,
                    config=config,
                )
            )
        shutil.copytree(staged, shard.table_path)


def _initialize_worker(
    repo_root: str,
    config: dict[str, Any],
) -> None:
    global _WORKER_SWM, _WORKER_CONFIG
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[variable] = "1"
    _WORKER_SWM, _, observed = load_stable_worldmodel(
        Path(repo_root),
        str(config["stable_worldmodel"]["repo"]),
        str(config["stable_worldmodel"]["commit"]),
    )
    try:
        import torch

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except (ImportError, RuntimeError):
        pass
    if observed != str(config["stable_worldmodel"]["commit"]):
        raise RuntimeError("H7 worker loaded the wrong Stable-WorldModel")
    _WORKER_CONFIG = config


def _collect_worker(shard: ActionDelayH7ShardPlan) -> str:
    if _WORKER_SWM is None or _WORKER_CONFIG is None:
        raise RuntimeError("H7 collection worker is not initialized")
    collect_shard(_WORKER_SWM, shard=shard, config=_WORKER_CONFIG)
    return shard.scenario_id


def _decode_blob(blob: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(blob)) as image:
        return np.asarray(image.convert("RGB")).copy()


def apply_formal_clip_filter(dataset: Any) -> dict[str, Any]:
    """Keep only raw-start 0 from each 50-row H7 episode."""

    before = list(dataset.clip_indices)
    expected_before = len(dataset.lengths) * DEFAULT_CLIPS_PER_EPISODE
    if len(before) != expected_before:
        raise ValueError(
            "Unexpected default H7 clip count before alignment filter: "
            f"{len(before)} != {expected_before}"
        )
    allowed = {
        (episode_index, start)
        for episode_index in range(len(dataset.lengths))
        for start in FORMAL_CLIP_STARTS
    }
    dataset.clip_indices = [
        value for value in before if value in allowed
    ]
    expected_after = len(dataset.lengths) * len(FORMAL_CLIP_STARTS)
    if len(dataset.clip_indices) != expected_after:
        raise RuntimeError(
            "H7 aligned clip filter produced the wrong count: "
            f"{len(dataset.clip_indices)} != {expected_after}"
        )
    return {
        "default_clip_count": len(before),
        "formal_clip_count": len(dataset.clip_indices),
        "allowed_raw_starts": list(FORMAL_CLIP_STARTS),
        "removed_misaligned_clips": len(before) - len(dataset.clip_indices),
        "passed": True,
    }


def audit_shard(
    swm: Any,
    *,
    shard: ActionDelayH7ShardPlan,
    config: dict[str, Any],
) -> dict[str, Any]:
    raw = swm.data.LanceDataset(path=shard.table_path)
    columns = set(raw.column_names)
    missing = sorted(REQUIRED_COLUMNS - columns)
    if missing:
        raise ValueError(f"{shard.scenario_id} missing columns {missing}")
    lengths = [int(value) for value in raw.lengths]
    expected_episodes = len(shard.episodes)
    if lengths != [RAW_STEPS] * expected_episodes:
        raise ValueError(f"{shard.scenario_id} has invalid episode lengths")
    delays = np.asarray(
        raw.get_col_data("variation_action_delay_steps")
    ).reshape(-1)
    speeds = np.asarray(
        raw.get_col_data("variation_agent_speed")
    ).reshape(-1)
    if not np.array_equal(
        delays,
        np.full_like(delays, shard.delay_steps),
    ):
        raise ValueError(f"{shard.scenario_id} delay readback differs")
    speed = float(config["protocol"]["agent_speed"])
    if not np.allclose(speeds, speed, atol=0.0, rtol=0.0):
        raise ValueError(f"{shard.scenario_id} speed readback differs")

    actions = np.asarray(raw.get_col_data("action"), dtype=np.float32)
    states = np.asarray(raw.get_col_data("proprio"), dtype=np.float32)
    goals = np.asarray(raw.get_col_data("goal_state"), dtype=np.float32)
    pixels = raw.get_col_data("pixels")
    terminated = np.asarray(raw.get_col_data("terminated")).reshape(-1)
    truncated = np.asarray(raw.get_col_data("truncated")).reshape(-1)
    codec = dict(config["storage"]["pixel_codec"])
    state_mismatches = 0
    pixel_mismatches = 0
    flag_mismatches = 0
    analytical_mismatches = 0
    initial_hashes: list[str] = []
    query_hashes: list[str] = []
    target_hashes: list[str] = []
    action_hashes: list[str] = []

    for plan, offset, length in zip(
        shard.episodes,
        raw.offsets,
        raw.lengths,
        strict=True,
    ):
        start = int(offset)
        stop = start + int(length)
        env = make_action_delay_h7_env(render_mode="rgb_array")
        try:
            observation, _ = env.reset(
                seed=int(plan.template.simulator_seed),
                options={
                    "variation": (),
                    "variation_values": {
                        "agent.speed": np.asarray(
                            [speed],
                            dtype=np.float32,
                        ),
                        ACTION_DELAY_FACTOR: int(plan.delay_steps),
                    },
                    "state": np.asarray(
                        plan.template.reset_state,
                        dtype=np.float32,
                    ),
                    "target_state": np.asarray(
                        plan.template.goal_state,
                        dtype=np.float32,
                    ),
                },
            )
            state_mismatches += int(
                not np.array_equal(
                    np.asarray(observation[:2], dtype=np.float32),
                    states[start],
                )
            )
            rendered = np.asarray(env.render(), dtype=np.uint8)
            pixel_mismatches += int(
                encode_frame(rendered, codec) != pixels[start]
            )
            for row in range(start, stop - 1):
                observation, _, ended, cut, _ = env.step(actions[row])
                state_mismatches += int(
                    not np.array_equal(
                        np.asarray(observation[:2], dtype=np.float32),
                        states[row + 1],
                    )
                )
                pixel_mismatches += int(
                    encode_frame(
                        np.asarray(env.render(), dtype=np.uint8),
                        codec,
                    )
                    != pixels[row + 1]
                )
                expected_cut = bool(
                    cut or row - start + 2 >= RAW_STEPS
                )
                flag_mismatches += int(
                    bool(terminated[row + 1]) != bool(ended)
                    or bool(truncated[row + 1]) != expected_cut
                )
        finally:
            env.close()

        reference = _reference_rollout(plan, config=config)
        model_indices = np.asarray(
            [
                start + block * ACTION_BLOCK
                for block in range(EPISODE_MODEL_FRAMES)
            ],
            dtype=np.int64,
        )
        expected_states = np.concatenate(
            [
                reference["history_states"],
                reference["future_states"],
            ],
            axis=0,
        )
        analytical_mismatches += int(
            not np.array_equal(states[model_indices], expected_states)
        )
        expected_actions = _model_blocks(reference).reshape(
            EPISODE_MODEL_FRAMES,
            -1,
        )
        observed_actions = np.stack(
            [
                actions[
                    start
                    + block
                    * ACTION_BLOCK : start
                    + (block + 1)
                    * ACTION_BLOCK
                ].reshape(-1)
                for block in range(EPISODE_MODEL_FRAMES)
            ]
        )
        analytical_mismatches += int(
            not np.array_equal(observed_actions, expected_actions)
        )
        decoded_frames = [
            _decode_blob(pixels[index]) for index in model_indices
        ]
        expected_pixels = np.concatenate(
            [
                reference["history_pixels"],
                reference["future_pixels"],
            ],
            axis=0,
        )
        analytical_mismatches += int(
            not np.array_equal(
                np.stack(decoded_frames),
                expected_pixels,
            )
        )
        initial_hashes.append(array_sha256(decoded_frames[0]))
        query_hashes.append(
            array_sha256(decoded_frames[HISTORY_TOKENS - 1])
        )
        target_hashes.append(
            array_sha256(decoded_frames[HISTORY_TOKENS])
        )
        action_hashes.append(array_sha256(expected_actions))

    model_dataset = swm.data.LanceDataset(
        path=shard.table_path,
        frameskip=ACTION_BLOCK,
        num_steps=TRAIN_SEQUENCE_STEPS,
        keys_to_load=list(DIAGNOSTIC_KEYS),
    )
    filter_audit = apply_formal_clip_filter(model_dataset)
    model_shapes = {
        key: list(model_dataset[0][key].shape)
        for key in DIAGNOSTIC_KEYS
    }
    expected_shapes = {
        "pixels": [TRAIN_SEQUENCE_STEPS, 3, 224, 224],
        "action": [TRAIN_SEQUENCE_STEPS, ACTION_BLOCK * 2],
        "proprio": [TRAIN_SEQUENCE_STEPS, 2],
    }
    checks = {
        "required_columns_present": not missing,
        "exact_episode_count": len(lengths) == expected_episodes,
        "every_episode_has_fifty_rows": lengths
        == [RAW_STEPS] * expected_episodes,
        "delay_readback_exact": True,
        "speed_readback_exact": True,
        "raw_state_replay_exact": state_mismatches == 0,
        "raw_pixel_replay_exact": pixel_mismatches == 0,
        "termination_flags_exact": flag_mismatches == 0,
        "analytical_ten_model_frames_exact": analytical_mismatches == 0,
        "default_reader_has_eleven_clips_per_episode": (
            filter_audit["default_clip_count"]
            == expected_episodes * DEFAULT_CLIPS_PER_EPISODE
        ),
        "formal_reader_has_one_aligned_clip_per_episode": (
            filter_audit["formal_clip_count"] == expected_episodes
        ),
        "formal_clip_starts_at_raw_zero": all(
            start == 0 for _, start in model_dataset.clip_indices
        ),
        "model_shapes_exact": model_shapes == expected_shapes,
    }
    return {
        "scenario_id": shard.scenario_id,
        "group": shard.group,
        "split": shard.split,
        "delay_steps": shard.delay_steps,
        "passed": all(checks.values()),
        "checks": checks,
        "episodes": expected_episodes,
        "raw_rows": sum(lengths),
        "default_model_clips": filter_audit["default_clip_count"],
        "formal_model_clips": filter_audit["formal_clip_count"],
        "clip_filter": filter_audit,
        "initial_pixels_sha256": initial_hashes,
        "query_pixels_sha256": query_hashes,
        "target_pixels_sha256": target_hashes,
        "model_action_sha256": action_hashes,
        "model_shapes": model_shapes,
        "storage_sha256": directory_sha256(shard.table_path),
    }


def _audit_worker(shard: ActionDelayH7ShardPlan) -> dict[str, Any]:
    if _WORKER_SWM is None or _WORKER_CONFIG is None:
        raise RuntimeError("H7 audit worker is not initialized")
    return audit_shard(
        _WORKER_SWM,
        shard=shard,
        config=_WORKER_CONFIG,
    )


def _manifest_record(
    shard: ActionDelayH7ShardPlan,
    *,
    config: dict[str, Any],
    repo_root: Path,
    audit: dict[str, Any],
    stable_commit: str,
    collection_status: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "benchmark": config["benchmark"],
        "group": shard.group,
        "split": shard.split,
        "regime": (
            "train_action_delay_history7"
            if shard.split == "train"
            else "validation_action_delay_history7"
        ),
        "scenario_id": shard.scenario_id,
        "fingerprint": shard.fingerprint,
        "output_path": portable_contextworld_path(
            shard.table_path,
            repo_root=repo_root,
        ),
        "stable_worldmodel_commit": stable_commit,
        "pixel_codec": dict(config["storage"]["pixel_codec"]),
        "collection_status": collection_status,
        "seed_group": (
            f"action-delay-h7-{shard.split}-s"
            f"{shard.shard_index:03d}"
        ),
        "factors": {ACTION_DELAY_FACTOR: shard.delay_steps},
        "episodes": len(shard.episodes),
        "rows_per_episode": RAW_STEPS,
        "default_model_clips": audit["default_model_clips"],
        "formal_model_clips": audit["formal_model_clips"],
        "formal_clip_start_raw_steps": list(FORMAL_CLIP_STARTS),
        "storage_sha256": audit["storage_sha256"],
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(
                    json.dumps(
                        row,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                )
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _catalog_payload(
    *,
    config: dict[str, Any],
    group: str,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    by_split = {
        split: [
            record["output_path"]
            for record in records
            if record["split"] == split
        ]
        for split in SPLITS
    }
    delay_support = (
        [SINGLE_DELAY] if group == "action_delay_single" else list(MULTI_DELAYS)
    )
    return {
        "schema_version": 1,
        "benchmark": config["benchmark"],
        "group": group,
        "pixel_codec": dict(config["storage"]["pixel_codec"]),
        "model_columns": list(MODEL_KEYS),
        "raw_privileged_columns_excluded_from_model": [
            "variation_action_delay_steps",
            "proprio",
            "state",
        ],
        "delay_support": delay_support,
        "temporal_contract": {
            "history_tokens": HISTORY_TOKENS,
            "num_preds": NUM_PREDS,
            "frameskip": ACTION_BLOCK,
            "num_steps": TRAIN_SEQUENCE_STEPS,
            "episode_raw_rows": RAW_STEPS,
            "allowed_clip_start_raw_steps": list(FORMAL_CLIP_STARTS),
        },
        "train": {
            "original": config["original_dataset"]["path"],
            "synthetic": by_split["train"],
        },
        "val": {
            "original": config["original_dataset"]["path"],
            "synthetic": by_split["val"],
        },
        "ood_test": {"synthetic": []},
        "by_regime": {
            "train_action_delay_history7": by_split["train"],
            "validation_action_delay_history7": by_split["val"],
        },
        "mixing": {"strategy": "configured_by_training_recipe"},
    }


def _set_by_split(
    plans: list[ActionDelayH7ShardPlan],
    audits: list[dict[str, Any]],
    field: str,
) -> dict[str, set[Any]]:
    values: dict[str, set[Any]] = {split: set() for split in SPLITS}
    for shard, audit in zip(plans, audits, strict=True):
        if field == "template_id":
            values[shard.split].update(
                plan.template.template_id for plan in shard.episodes
            )
        elif field == "simulator_seed":
            values[shard.split].update(
                int(plan.template.simulator_seed) for plan in shard.episodes
            )
        elif field == "reset_state":
            values[shard.split].update(
                tuple(plan.template.reset_state) for plan in shard.episodes
            )
        else:
            values[shard.split].update(audit[field])
    return values


def _pair_audit(
    plans_by_group: dict[str, list[ActionDelayH7ShardPlan]],
    audits_by_group: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    single = {
        (shard.split, shard.shard_index): (shard, audit)
        for shard, audit in zip(
            plans_by_group["action_delay_single"],
            audits_by_group["action_delay_single"],
            strict=True,
        )
    }
    multi = {
        (shard.split, shard.shard_index): (shard, audit)
        for shard, audit in zip(
            plans_by_group["action_delay_multi"],
            audits_by_group["action_delay_multi"],
            strict=True,
        )
    }
    keys_exact = set(single) == set(multi)
    templates_exact = keys_exact and all(
        [
            asdict(left.template) == asdict(right.template)
            for left, right in zip(
                single[key][0].episodes,
                multi[key][0].episodes,
                strict=True,
            )
        ]
        for key in single
    )
    actions_exact = keys_exact and all(
        single[key][1]["model_action_sha256"]
        == multi[key][1]["model_action_sha256"]
        for key in single
    )
    query_pixels_exact = keys_exact and all(
        single[key][1]["query_pixels_sha256"]
        == multi[key][1]["query_pixels_sha256"]
        for key in single
    )
    paired_episodes = sum(
        len(shard.episodes) for shard, _ in single.values()
    )
    return {
        "paired_shards": len(single),
        "paired_episodes": paired_episodes,
        "keys_exact": keys_exact,
        "templates_exact": templates_exact,
        "commanded_actions_exact": actions_exact,
        "query_pixels_exact": query_pixels_exact,
        "passed": bool(
            keys_exact
            and templates_exact
            and actions_exact
            and query_pixels_exact
        ),
    }


def _isolation_audit(
    *,
    config: dict[str, Any],
    repo_root: Path,
    plans: list[ActionDelayH7ShardPlan],
    audits: list[dict[str, Any]],
) -> dict[str, Any]:
    exclusion_path = resolve_contextworld_path(
        config["validation_exclusion"]["manifest"],
        repo_root=repo_root,
    )
    observed_exclusion_sha = file_sha256(exclusion_path)
    expected_exclusion_sha = str(
        config["validation_exclusion"]["manifest_sha256"]
    )
    if observed_exclusion_sha != expected_exclusion_sha:
        raise ValueError(
            "H7 Validation exclusion manifest changed: "
            f"{observed_exclusion_sha} != {expected_exclusion_sha}"
        )
    exclusion = json.loads(exclusion_path.read_text(encoding="utf-8"))
    expected_content = str(
        config["validation_exclusion"]["content_manifest_sha256"]
    )
    if exclusion.get("content_manifest_sha256") != expected_content:
        raise ValueError("H7 Validation exclusion content identity changed")
    if int(exclusion.get("query_count", -1)) != int(
        config["validation_exclusion"]["query_count"]
    ):
        raise ValueError("H7 Validation exclusion query count changed")

    training = {
        field: _set_by_split(plans, audits, field)
        for field in (
            "template_id",
            "simulator_seed",
            "reset_state",
            "initial_pixels_sha256",
            "query_pixels_sha256",
        )
    }
    eval_values = {
        "template_id": {
            str(row["template_id"])
            for row in exclusion["query_records"]
        },
        "simulator_seed": {
            int(row["simulator_seed"])
            for row in exclusion["query_records"]
        },
        "reset_state": {
            tuple(map(float, row["reset_state"]))
            for row in exclusion["query_records"]
        },
        "initial_pixels_sha256": {
            str(row["initial_pixels_sha256"])
            for row in exclusion["query_records"]
        },
        "query_pixels_sha256": {
            str(row["query_pixels_sha256"])
            for row in exclusion["query_records"]
        },
    }
    validation_overlap = {}
    split_overlap = {}
    for field, by_split in training.items():
        combined = set().union(*by_split.values())
        validation_overlap[field] = sorted(
            combined & eval_values[field],
            key=str,
        )
        split_overlap[field] = sorted(
            by_split["train"] & by_split["val"],
            key=str,
        )
    base_episode_count = sum(len(shard.episodes) for shard in plans)
    uniqueness = {
        field: len(set().union(*by_split.values())) == base_episode_count
        for field, by_split in training.items()
    }
    checks = {
        "validation_manifest_hash_exact": observed_exclusion_sha
        == expected_exclusion_sha,
        "validation_content_identity_exact": exclusion.get(
            "content_manifest_sha256"
        )
        == expected_content,
        "validation_overlap_zero_for_every_frozen_field": all(
            not values for values in validation_overlap.values()
        ),
        "train_internal_validation_overlap_zero": all(
            not values for values in split_overlap.values()
        ),
        "base_template_ids_unique": uniqueness["template_id"],
        "base_simulator_seeds_unique": uniqueness["simulator_seed"],
        "base_reset_states_unique": uniqueness["reset_state"],
        "base_initial_pixels_unique": uniqueness[
            "initial_pixels_sha256"
        ],
        "base_query_pixels_unique": uniqueness["query_pixels_sha256"],
    }
    return {
        "manifest": portable_contextworld_path(
            exclusion_path,
            repo_root=repo_root,
        ),
        "manifest_sha256": observed_exclusion_sha,
        "eval_queries": len(exclusion["query_records"]),
        "base_training_and_internal_validation_episodes": (
            base_episode_count
        ),
        "validation_overlap": validation_overlap,
        "train_internal_validation_overlap": split_overlap,
        "unique_counts": {
            field: len(set().union(*by_split.values()))
            for field, by_split in training.items()
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def _delay_balance(
    plans: list[ActionDelayH7ShardPlan],
) -> dict[str, Any]:
    counts = {
        split: Counter(
            plan.delay_steps
            for shard in plans
            if shard.split == split
            for plan in shard.episodes
        )
        for split in SPLITS
    }
    expected = {
        split: {
            delay: sum(
                len(shard.episodes)
                for shard in plans
                if shard.split == split
            )
            // len(MULTI_DELAYS)
            for delay in MULTI_DELAYS
        }
        for split in SPLITS
    }
    passed = all(dict(counts[split]) == expected[split] for split in SPLITS)
    return {
        "counts": {
            split: {
                str(delay): counts[split][delay] for delay in MULTI_DELAYS
            }
            for split in SPLITS
        },
        "expected": {
            split: {
                str(delay): expected[split][delay]
                for delay in MULTI_DELAYS
            }
            for split in SPLITS
        },
        "passed": passed,
    }


def _collect_missing(
    swm: Any,
    *,
    missing: list[ActionDelayH7ShardPlan],
    config: dict[str, Any],
    repo_root: Path,
    workers: int,
) -> None:
    if not missing:
        return
    if workers == 1:
        for index, shard in enumerate(missing, start=1):
            collect_shard(swm, shard=shard, config=config)
            print(
                f"[action-delay-h7-data] collected "
                f"{index}/{len(missing)} {shard.scenario_id}",
                flush=True,
            )
        return
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=context,
        initializer=_initialize_worker,
        initargs=(str(repo_root), config),
    ) as executor:
        for index, (shard, observed_id) in enumerate(
            zip(
                missing,
                executor.map(_collect_worker, missing, chunksize=1),
                strict=True,
            ),
            start=1,
        ):
            if observed_id != shard.scenario_id:
                raise RuntimeError("H7 collection worker identity changed")
            print(
                f"[action-delay-h7-data] collected "
                f"{index}/{len(missing)} {observed_id}",
                flush=True,
            )


def _audit_group(
    swm: Any,
    *,
    shards: list[ActionDelayH7ShardPlan],
    config: dict[str, Any],
    repo_root: Path,
    workers: int,
) -> list[dict[str, Any]]:
    if workers == 1:
        audits = []
        for index, shard in enumerate(shards, start=1):
            audits.append(audit_shard(swm, shard=shard, config=config))
            print(
                f"[action-delay-h7-data] audited {shard.group} "
                f"{index}/{len(shards)}",
                flush=True,
            )
        return audits
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=context,
        initializer=_initialize_worker,
        initargs=(str(repo_root), config),
    ) as executor:
        audits = []
        for index, (shard, audit) in enumerate(
            zip(
                shards,
                executor.map(_audit_worker, shards, chunksize=1),
                strict=True,
            ),
            start=1,
        ):
            if audit["scenario_id"] != shard.scenario_id:
                raise RuntimeError("H7 audit worker identity changed")
            audits.append(audit)
            print(
                f"[action-delay-h7-data] audited {shard.group} "
                f"{index}/{len(shards)}",
                flush=True,
            )
        return audits


def build_training_release(
    *,
    config: dict[str, Any],
    repo_root: Path,
    workers: int,
    resume: bool,
) -> dict[str, Any]:
    protocol_checks = _validate_config(config)
    if workers < 1:
        raise ValueError("workers must be positive")
    output_root = resolve_contextworld_path(
        config["output_root"],
        repo_root=repo_root,
    )
    output_root.mkdir(parents=True, exist_ok=resume)
    swm, stable_repo, stable_commit = load_stable_worldmodel(
        repo_root,
        str(config["stable_worldmodel"]["repo"]),
        str(config["stable_worldmodel"]["commit"]),
    )
    plans_by_group = build_shard_plans(config, repo_root=repo_root)
    all_shards = [
        shard
        for group in GROUPS
        for shard in plans_by_group[group]
    ]
    missing = [
        shard for shard in all_shards if not shard.table_path.exists()
    ]
    if missing and not resume and len(missing) != len(all_shards):
        raise FileExistsError(
            "H7 release is partially populated; use --resume to finish it"
        )
    _collect_missing(
        swm,
        missing=missing,
        config=config,
        repo_root=repo_root,
        workers=workers,
    )

    audits_by_group: dict[str, list[dict[str, Any]]] = {}
    artifacts: dict[str, Any] = {}
    for group in GROUPS:
        shards = plans_by_group[group]
        audits = _audit_group(
            swm,
            shards=shards,
            config=config,
            repo_root=repo_root,
            workers=workers,
        )
        failed = [
            audit["scenario_id"] for audit in audits if not audit["passed"]
        ]
        if failed:
            raise RuntimeError(f"H7 shard audits failed: {failed}")
        audits_by_group[group] = audits
        records = [
            _manifest_record(
                shard,
                config=config,
                repo_root=repo_root,
                audit=audit,
                stable_commit=stable_commit,
                collection_status=(
                    "collected" if shard in missing else "reused"
                ),
            )
            for shard, audit in zip(shards, audits, strict=True)
        ]
        stem = str(config["catalog_stems"][group])
        catalog_path = output_root / "catalogs" / f"{stem}.json"
        manifest_path = output_root / "manifests" / f"{stem}.jsonl"
        report_path = output_root / "reports" / f"{stem}.json"
        _write_jsonl(manifest_path, records)
        catalog = _catalog_payload(
            config=config,
            group=group,
            records=records,
        )
        write_json(catalog_path, catalog)
        loader = validate_loader_mix(
            swm,
            original_dataset=resolve_contextworld_path(
                config["original_dataset"]["path"],
                repo_root=repo_root,
            ),
            synthetic_dataset=shards[0].table_path,
            cache_dir=Path("/tmp/contextworld-action-delay-h7-loader"),
        )
        group_report = {
            "schema_version": 1,
            "experiment": stem,
            "benchmark": config["benchmark"],
            "group": group,
            "compile_only": False,
            "preflight_passed": bool(loader["passed"]),
            "passed": bool(
                loader["passed"] and all(audit["passed"] for audit in audits)
            ),
            "stable_worldmodel": {
                "repo": str(stable_repo),
                "commit": stable_commit,
            },
            "catalog": str(catalog_path.resolve()),
            "manifest": str(manifest_path.resolve()),
            "collection_status": {
                record["scenario_id"]: record["collection_status"]
                for record in records
            },
            "loader_compatibility": loader,
            "temporal_contract": catalog["temporal_contract"],
            "scenarios": [
                {
                    "scenario_id": audit["scenario_id"],
                    "passed": audit["passed"],
                    "checks": audit["checks"],
                }
                for audit in audits
            ],
            "counts": {
                split: {
                    "shards": sum(
                        shard.split == split for shard in shards
                    ),
                    "episodes": sum(
                        len(shard.episodes)
                        for shard in shards
                        if shard.split == split
                    ),
                    "formal_model_clips": sum(
                        audit["formal_model_clips"]
                        for shard, audit in zip(
                            shards,
                            audits,
                            strict=True,
                        )
                        if shard.split == split
                    ),
                }
                for split in SPLITS
            },
        }
        write_json(report_path, group_report)
        artifacts[group] = {
            "catalog": portable_contextworld_path(
                catalog_path,
                repo_root=repo_root,
            ),
            "catalog_sha256": file_sha256(catalog_path),
            "manifest": portable_contextworld_path(
                manifest_path,
                repo_root=repo_root,
            ),
            "manifest_sha256": file_sha256(manifest_path),
            "synthesis_report": portable_contextworld_path(
                report_path,
                repo_root=repo_root,
            ),
            "synthesis_report_sha256": file_sha256(report_path),
            "counts": group_report["counts"],
        }

    pair_audit = _pair_audit(plans_by_group, audits_by_group)
    isolation = _isolation_audit(
        config=config,
        repo_root=repo_root,
        plans=plans_by_group["action_delay_single"],
        audits=audits_by_group["action_delay_single"],
    )
    delay_balance = _delay_balance(
        plans_by_group["action_delay_multi"]
    )
    delay_support = {
        group: {
            split: sorted(
                {
                    shard.delay_steps
                    for shard in plans_by_group[group]
                    if shard.split == split
                }
            )
            for split in SPLITS
        }
        for group in GROUPS
    }
    expected_clips = {
        group: {
            split: int(config["counts"][split]["clips_per_group"])
            for split in SPLITS
        }
        for group in GROUPS
    }
    observed_clips = {
        group: {
            split: sum(
                audit["formal_model_clips"]
                for shard, audit in zip(
                    plans_by_group[group],
                    audits_by_group[group],
                    strict=True,
                )
                if shard.split == split
            )
            for split in SPLITS
        }
        for group in GROUPS
    }
    checks = {
        **protocol_checks,
        "all_shards_pass_full_raw_replay": all(
            audit["passed"]
            for audits in audits_by_group.values()
            for audit in audits
        ),
        "single_and_multi_are_exactly_paired": pair_audit["passed"],
        "multi_delay_counts_are_exactly_balanced": delay_balance[
            "passed"
        ],
        "validation_and_split_isolation_pass": isolation["passed"],
        "single_delay_support_exact": delay_support[
            "action_delay_single"
        ]
        == {"train": [SINGLE_DELAY], "val": [SINGLE_DELAY]},
        "multi_delay_support_exact": delay_support[
            "action_delay_multi"
        ]
        == {
            "train": list(MULTI_DELAYS),
            "val": list(MULTI_DELAYS),
        },
        "formal_clip_counts_exact": observed_clips == expected_clips,
        "catalogs_and_manifests_published": set(artifacts) == set(GROUPS),
        "model_projection_is_pixels_and_actions_only": MODEL_KEYS
        == ("pixels", "action"),
    }
    report = {
        "schema_version": 1,
        "benchmark": config["benchmark"],
        "status": "passed" if all(checks.values()) else "failed",
        "passed": all(checks.values()),
        "scope": (
            "paired_history7_training_data_and_full_physical_replay; "
            "no_model_training"
        ),
        "checks": checks,
        "identity": {
            "stable_worldmodel_repo": str(stable_repo),
            "stable_worldmodel_commit": stable_commit,
        },
        "temporal_contract": {
            "history_tokens": HISTORY_TOKENS,
            "num_preds": NUM_PREDS,
            "raw_steps_per_action_block": ACTION_BLOCK,
            "episode_model_frames": EPISODE_MODEL_FRAMES,
            "rows_per_episode": RAW_STEPS,
            "default_clips_per_episode": DEFAULT_CLIPS_PER_EPISODE,
            "formal_clip_start_raw_steps": list(FORMAL_CLIP_STARTS),
            "formal_clips_per_episode": len(FORMAL_CLIP_STARTS),
        },
        "delay_support": delay_support,
        "delay_balance": delay_balance,
        "paired_geometry_and_actions": pair_audit,
        "isolation": isolation,
        "formal_clip_counts": observed_clips,
        "artifacts_by_group": artifacts,
        "physical_counts": {
            "shards": len(all_shards),
            "episodes": sum(
                len(shard.episodes) for shard in all_shards
            ),
            "raw_rows_replayed": sum(
                len(shard.episodes) * RAW_STEPS
                for shard in all_shards
            ),
            "formal_model_clips": sum(
                audit["formal_model_clips"]
                for audits in audits_by_group.values()
                for audit in audits
            ),
            "misaligned_default_clips_rejected": sum(
                audit["clip_filter"]["removed_misaligned_clips"]
                for audits in audits_by_group.values()
                for audit in audits
            ),
        },
    }
    write_json(output_root / "build_report.json", report)
    return report


__all__ = [
    "DEFAULT_CLIPS_PER_EPISODE",
    "DIAGNOSTIC_KEYS",
    "EPISODE_MODEL_FRAMES",
    "FORMAL_CLIP_STARTS",
    "GROUPS",
    "H7_ACTION_DELAY_ENV_ID",
    "HISTORY_TOKENS",
    "MODEL_KEYS",
    "MAXIMUM_DELAY_STEPS",
    "MULTI_DELAYS",
    "RAW_STEPS",
    "REQUIRED_COLUMNS",
    "SINGLE_DELAY",
    "SPLITS",
    "TRAIN_SEQUENCE_STEPS",
    "ActionDelayH7EpisodePlan",
    "ActionDelayH7ShardPlan",
    "apply_formal_clip_filter",
    "audit_shard",
    "build_shard_plans",
    "build_training_release",
    "collect_shard",
    "directory_sha256",
    "make_action_delay_h7_env",
    "register_action_delay_h7_env",
    "training_template",
]
