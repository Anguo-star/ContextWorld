from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
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

from .action_delay import (
    ACTION_BLOCK,
    TRAIN_DELAY_VALUES,
    ActionDelayTemplate,
    array_sha256,
    canonical_sha256,
    simulate_template,
)
from .action_delay_env import (
    ACTION_DELAY_ENV_ID,
    ACTION_DELAY_FACTOR,
    make_action_delay_env,
    register_action_delay_env,
)
from .action_delay_validation import file_sha256


RAW_STEPS = 20
MODEL_STEPS = 4
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
GROUPS = ("action_delay_single", "action_delay_multi")
SPLITS = ("train", "val")


@dataclass(frozen=True)
class ActionDelayEpisodePlan:
    template: ActionDelayTemplate
    delay_steps: int
    split: str
    shard_index: int
    episode_index: int


@dataclass(frozen=True)
class ActionDelayShardPlan:
    group: str
    split: str
    shard_index: int
    delay_steps: int
    scenario_id: str
    fingerprint: str
    table_path: Path
    episodes: tuple[ActionDelayEpisodePlan, ...]


_WORKER_SWM: Any | None = None
_WORKER_CONFIG: dict[str, Any] | None = None


def directory_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    root = Path(path)
    for value in sorted(
        (item for item in root.rglob("*") if item.is_file()),
        key=lambda item: item.relative_to(root).as_posix(),
    ):
        relative = value.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with value.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _direction_action(direction: str) -> np.ndarray:
    if direction == "up":
        return np.asarray([0.0, 1.0], dtype=np.float32)
    if direction == "down":
        return np.asarray([0.0, -1.0], dtype=np.float32)
    raise ValueError(f"Unknown direction {direction!r}")


def _reserved_training_coordinates() -> tuple[list[int], list[int], list[int]]:
    eval_left_x = set(range(25, 94, 3))
    eval_right_x = set(range(133, 201, 3))
    eval_y = set(range(60, 166, 5))
    left = [value for value in range(25, 93) if value not in eval_left_x]
    right = [
        value for value in range(133, 201) if value not in eval_right_x
    ]
    y_values = [value for value in range(60, 166) if value not in eval_y]
    return left, right, y_values


def training_template(
    *,
    catalog_seed: int,
    split: str,
    shard_index: int,
    episode_index: int,
) -> ActionDelayTemplate:
    """Generate geometry that is raster-disjoint from formal Validation."""

    if split not in SPLITS:
        raise ValueError(f"Unknown action-delay split {split!r}")
    left_x, right_x, y_values = _reserved_training_coordinates()
    split_index = SPLITS.index(split)
    rng = np.random.default_rng(
        np.random.SeedSequence(
            [
                int(catalog_seed),
                split_index,
                int(shard_index),
                int(episode_index),
            ]
        )
    )
    global_index = shard_index * 1_000_000 + episode_index
    direction = "up" if global_index % 2 == 0 else "down"
    room = "left" if (global_index // 2) % 2 == 0 else "right"
    x_position = float(rng.choice(left_x if room == "left" else right_x))
    query_y = float(rng.choice(y_values))
    action = _direction_action(direction)
    reset = (
        np.asarray([x_position, query_y], dtype=np.float32)
        - 7.0 * ACTION_BLOCK * action
    )
    goal = (
        (200.0, 190.0 if query_y < 113.0 else 35.0)
        if room == "left"
        else (25.0, 190.0 if query_y < 113.0 else 35.0)
    )
    simulator_seed = int(
        np.random.SeedSequence(
            [
                int(catalog_seed),
                0xA71D,
                split_index,
                int(shard_index),
                int(episode_index),
            ]
        ).generate_state(1)[0]
    )
    return ActionDelayTemplate(
        template_id=(
            f"action-delay-{split}-s{shard_index:03d}-"
            f"e{episode_index:03d}"
        ),
        direction=direction,
        reset_state=tuple(map(float, reset)),
        goal_state=tuple(map(float, goal)),
        simulator_seed=simulator_seed,
    )


def _delay_for_shard(group: str, shard_index: int) -> int:
    if group == "action_delay_single":
        return 2
    if group == "action_delay_multi":
        return TRAIN_DELAY_VALUES[shard_index % len(TRAIN_DELAY_VALUES)]
    raise ValueError(f"Unknown action-delay data group {group!r}")


def build_shard_plans(
    config: dict[str, Any],
    *,
    repo_root: Path,
) -> dict[str, list[ActionDelayShardPlan]]:
    output_root = resolve_contextworld_path(
        config["output_root"],
        repo_root=repo_root,
    )
    catalog_seed = int(config["catalog_seed"])
    plans: dict[str, list[ActionDelayShardPlan]] = {
        group: [] for group in GROUPS
    }
    for group in GROUPS:
        for split in SPLITS:
            shard_count = int(config["counts"][split]["shards"])
            episodes_per_shard = int(
                config["counts"][split]["episodes_per_shard"]
            )
            if (
                group == "action_delay_multi"
                and shard_count % len(TRAIN_DELAY_VALUES)
            ):
                raise ValueError(
                    f"{split} shard count must balance delays 0,2,4"
                )
            for shard_index in range(shard_count):
                delay = _delay_for_shard(group, shard_index)
                episode_plans = tuple(
                    ActionDelayEpisodePlan(
                        template=training_template(
                            catalog_seed=catalog_seed,
                            split=split,
                            shard_index=shard_index,
                            episode_index=episode_index,
                        ),
                        delay_steps=delay,
                        split=split,
                        shard_index=shard_index,
                        episode_index=episode_index,
                    )
                    for episode_index in range(episodes_per_shard)
                )
                fingerprint_payload = {
                    "benchmark": config["benchmark"],
                    "group": group,
                    "split": split,
                    "shard_index": shard_index,
                    "delay_steps": delay,
                    "episode_templates": [
                        asdict(plan.template) for plan in episode_plans
                    ],
                }
                fingerprint = canonical_sha256(fingerprint_payload)
                scenario_id = (
                    f"ad-{group.removeprefix('action_delay_')}-"
                    f"{split}-s{shard_index:03d}-{fingerprint[:10]}"
                )
                plans[group].append(
                    ActionDelayShardPlan(
                        group=group,
                        split=split,
                        shard_index=shard_index,
                        delay_steps=delay,
                        scenario_id=scenario_id,
                        fingerprint=fingerprint,
                        table_path=(
                            output_root
                            / "tables"
                            / group
                            / split
                            / f"{scenario_id}.lance"
                        ),
                        episodes=episode_plans,
                    )
                )
    return plans


def _model_blocks(reference: dict[str, Any]) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(reference["history_actions"], dtype=np.float32),
            np.asarray(reference["query_action"], dtype=np.float32)[None],
            np.zeros((1, ACTION_BLOCK, 2), dtype=np.float32),
        ],
        axis=0,
    )


def _collection_actions(reference: dict[str, Any]) -> np.ndarray:
    """Add one zero lead-in for World.collect's row/action convention."""

    return np.concatenate(
        [
            np.zeros((1, 2), dtype=np.float32),
            np.asarray(reference["history_actions"], dtype=np.float32).reshape(
                -1,
                2,
            ),
            np.asarray(reference["query_action"], dtype=np.float32),
            np.zeros((4, 2), dtype=np.float32),
        ],
        axis=0,
    )


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
            raise RuntimeError("Action-delay scripted policy is not ready")
        if self.step >= RAW_STEPS:
            raise RuntimeError("World requested too many scripted actions")
        action = self.actions[self.step]
        self.step += 1
        return np.repeat(action[None, :], self.env.num_envs, axis=0)


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
                f"Expected one captured episode, got {len(self.episodes)}"
            )
        return self.episodes[0]


def _episode_iterator(
    swm: Any,
    *,
    shard: ActionDelayShardPlan,
    agent_speed: float,
) -> Iterator[dict[str, Any]]:
    register_action_delay_env()
    world = swm.World(
        ACTION_DELAY_ENV_ID,
        num_envs=1,
        max_episode_steps=RAW_STEPS,
        image_shape=(224, 224),
        render_mode="rgb_array",
    )
    policy = _ResettableScriptedPolicy()
    world.set_policy(policy)
    try:
        for plan in shard.episodes:
            reference = simulate_template(
                plan.template,
                delay_steps=plan.delay_steps,
                agent_speed=agent_speed,
            )
            policy.reset_actions(_collection_actions(reference))
            capture = _OneEpisodeCapture()
            world.collect(
                episodes=1,
                seed=int(plan.template.simulator_seed),
                options={
                    "variation": WATCHED_VARIATIONS,
                    "variation_values": {
                        "agent.speed": np.asarray(
                            [agent_speed],
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
    shard: ActionDelayShardPlan,
    config: dict[str, Any],
) -> None:
    if shard.table_path.exists():
        raise FileExistsError(shard.table_path)
    shard.table_path.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(config["collection"]["staging_root"])
    staging_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="contextworld-action-delay-h3-",
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
                    agent_speed=float(config["protocol"]["agent_speed"]),
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
    if observed != str(config["stable_worldmodel"]["commit"]):
        raise RuntimeError("Action-delay worker loaded the wrong StableWM")
    _WORKER_CONFIG = config


def _collect_worker(shard: ActionDelayShardPlan) -> str:
    if _WORKER_SWM is None or _WORKER_CONFIG is None:
        raise RuntimeError("Action-delay shard worker was not initialized")
    collect_shard(_WORKER_SWM, shard=shard, config=_WORKER_CONFIG)
    return shard.scenario_id


def _decode_blob(blob: bytes) -> np.ndarray:
    import io

    with Image.open(io.BytesIO(blob)) as image:
        return np.asarray(image.convert("RGB")).copy()


def audit_shard(
    swm: Any,
    *,
    shard: ActionDelayShardPlan,
    config: dict[str, Any],
) -> dict[str, Any]:
    raw = swm.data.LanceDataset(path=shard.table_path)
    columns = set(raw.column_names)
    missing = sorted(REQUIRED_COLUMNS - columns)
    if missing:
        raise ValueError(
            f"{shard.scenario_id} is missing columns {missing}"
        )
    lengths = [int(value) for value in raw.lengths]
    expected_episodes = len(shard.episodes)
    if lengths != [RAW_STEPS] * expected_episodes:
        raise ValueError(
            f"{shard.scenario_id} has invalid episode lengths"
        )
    delay_column = np.asarray(
        raw.get_col_data("variation_action_delay_steps")
    ).reshape(-1)
    speed_column = np.asarray(
        raw.get_col_data("variation_agent_speed")
    ).reshape(-1)
    if not np.array_equal(
        delay_column,
        np.full_like(delay_column, shard.delay_steps),
    ):
        raise ValueError(
            f"{shard.scenario_id} delay readback differs"
        )
    if not np.allclose(
        speed_column,
        float(config["protocol"]["agent_speed"]),
        atol=0.0,
        rtol=0.0,
    ):
        raise ValueError(
            f"{shard.scenario_id} speed readback differs"
        )

    actions = np.asarray(raw.get_col_data("action"), dtype=np.float32)
    states = np.asarray(raw.get_col_data("proprio"), dtype=np.float32)
    goals = np.asarray(raw.get_col_data("goal_state"), dtype=np.float32)
    pixels = raw.get_col_data("pixels")
    terminated = np.asarray(raw.get_col_data("terminated")).reshape(-1)
    truncated = np.asarray(raw.get_col_data("truncated")).reshape(-1)
    state_mismatches = 0
    pixel_mismatches = 0
    flag_mismatches = 0
    analytical_mismatches = 0
    query_hashes: list[str] = []
    codec = dict(config["storage"]["pixel_codec"])
    speed = float(config["protocol"]["agent_speed"])

    for plan, offset, length in zip(
        shard.episodes,
        raw.offsets,
        raw.lengths,
        strict=True,
    ):
        start = int(offset)
        stop = start + int(length)
        env = make_action_delay_env(render_mode="rgb_array")
        try:
            env.reset(
                seed=int(plan.template.simulator_seed),
                options={
                    "variation": (),
                    "variation_values": {
                        "agent.speed": np.asarray(
                            [speed], dtype=np.float32
                        ),
                        ACTION_DELAY_FACTOR: int(plan.delay_steps),
                    },
                    "state": states[start].copy(),
                    "target_state": goals[start].copy(),
                },
            )
            rendered = np.asarray(env.render(), dtype=np.uint8)
            pixel_mismatches += int(
                encode_frame(rendered, codec) != pixels[start]
            )
            for row in range(start, stop - 1):
                observation, _, ended, cut, _ = env.step(actions[row])
                observed_state = np.asarray(
                    observation[:2],
                    dtype=np.float32,
                )
                state_mismatches += int(
                    not np.array_equal(observed_state, states[row + 1])
                )
                rendered = np.asarray(env.render(), dtype=np.uint8)
                pixel_mismatches += int(
                    encode_frame(rendered, codec) != pixels[row + 1]
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

        reference = simulate_template(
            plan.template,
            delay_steps=plan.delay_steps,
            agent_speed=speed,
        )
        model_indices = np.asarray(
            [start, start + 5, start + 10, start + 15],
            dtype=np.int64,
        )
        expected_states = np.concatenate(
            [
                reference["history_states"],
                reference["target_state"][None],
            ],
            axis=0,
        )
        analytical_mismatches += int(
            not np.array_equal(states[model_indices], expected_states)
        )
        expected_actions = _model_blocks(reference).reshape(MODEL_STEPS, -1)
        observed_action_blocks = np.stack(
            [
                actions[start + block * ACTION_BLOCK : start + (block + 1) * ACTION_BLOCK].reshape(
                    -1
                )
                for block in range(MODEL_STEPS)
            ]
        )
        analytical_mismatches += int(
            not np.array_equal(observed_action_blocks, expected_actions)
        )
        query_pixels = _decode_blob(pixels[start + 10])
        if not np.array_equal(query_pixels, reference["query_pixels"]):
            analytical_mismatches += 1
        query_hashes.append(array_sha256(query_pixels))

    model_dataset = swm.data.LanceDataset(
        path=shard.table_path,
        frameskip=ACTION_BLOCK,
        num_steps=MODEL_STEPS,
        keys_to_load=list(DIAGNOSTIC_KEYS),
    )
    model_shapes = {
        key: list(model_dataset[0][key].shape)
        for key in DIAGNOSTIC_KEYS
    }
    expected_shapes = {
        "pixels": [4, 3, 224, 224],
        "action": [4, 10],
        "proprio": [4, 2],
    }
    checks = {
        "required_columns": not missing,
        "exact_episode_count": len(lengths) == expected_episodes,
        "every_episode_is_one_h3_clip": lengths
        == [RAW_STEPS] * expected_episodes,
        "delay_readback_exact": True,
        "speed_readback_exact": True,
        "raw_state_replay_exact": state_mismatches == 0,
        "raw_pixel_replay_exact": pixel_mismatches == 0,
        "termination_flags_exact": flag_mismatches == 0,
        "analytical_model_rows_exact": analytical_mismatches == 0,
        "model_clip_count_exact": len(model_dataset) == expected_episodes,
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
        "model_clips": len(model_dataset),
        "query_pixels_sha256": query_hashes,
        "model_shapes": model_shapes,
        "storage_sha256": directory_sha256(shard.table_path),
    }


def _audit_worker(shard: ActionDelayShardPlan) -> dict[str, Any]:
    if _WORKER_SWM is None or _WORKER_CONFIG is None:
        raise RuntimeError("Action-delay audit worker was not initialized")
    return audit_shard(
        _WORKER_SWM,
        shard=shard,
        config=_WORKER_CONFIG,
    )


def _manifest_record(
    shard: ActionDelayShardPlan,
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
            "train_action_delay_history3"
            if shard.split == "train"
            else "validation_action_delay_history3"
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
            f"action-delay-{shard.split}-s{shard.shard_index:03d}"
        ),
        "factors": {ACTION_DELAY_FACTOR: shard.delay_steps},
        "episodes": len(shard.episodes),
        "rows_per_episode": RAW_STEPS,
        "model_clips": audit["model_clips"],
        "storage_sha256": audit["storage_sha256"],
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, sort_keys=True, separators=(",", ":"))
            )
            handle.write("\n")
    temporary.replace(path)


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
        "delay_support": (
            [2] if group == "action_delay_single" else list(TRAIN_DELAY_VALUES)
        ),
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
            "train_action_delay_history3": by_split["train"],
            "validation_action_delay_history3": by_split["val"],
        },
        "mixing": {"strategy": "configured_by_training_recipe"},
    }


def build_training_release(
    *,
    config: dict[str, Any],
    repo_root: Path,
    workers: int,
    resume: bool,
) -> dict[str, Any]:
    if tuple(map(int, config["protocol"]["training_delay_values"])) != (
        0,
        2,
        4,
    ):
        raise ValueError("Formal action-delay training support changed")
    if int(config["protocol"]["history_tokens"]) != 3:
        raise ValueError("Action-delay training requires History=3")
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
    if missing and resume is False and len(missing) != len(all_shards):
        raise FileExistsError(
            "Action-delay release is partially populated; use --resume "
            "to validate and finish it"
        )
    if missing:
        if workers <= 1:
            for index, shard in enumerate(missing, start=1):
                print(
                    f"[action-delay-data] collect {index}/{len(missing)} "
                    f"{shard.scenario_id}",
                    flush=True,
                )
                collect_shard(swm, shard=shard, config=config)
        else:
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
                        raise RuntimeError(
                            "Shard worker returned a different identity"
                        )
                    print(
                        f"[action-delay-data] collected "
                        f"{index}/{len(missing)} {observed_id}",
                        flush=True,
                    )

    exclusion_path = resolve_contextworld_path(
        config["validation_exclusion"]["manifest"],
        repo_root=repo_root,
    )
    exclusion = json.loads(exclusion_path.read_text(encoding="utf-8"))
    expected_exclusion_hash = str(
        config["validation_exclusion"]["manifest_sha256"]
    )
    if file_sha256(exclusion_path) != expected_exclusion_hash:
        raise ValueError("Validation exclusion manifest hash changed")
    eval_query_hashes = {
        str(row["query_pixels_sha256"])
        for row in exclusion["query_records"]
    }

    audits_by_group: dict[str, list[dict[str, Any]]] = {}
    artifacts: dict[str, Any] = {}
    for group in GROUPS:
        group_shards = plans_by_group[group]
        if workers <= 1:
            audits = []
            for index, shard in enumerate(group_shards, start=1):
                audits.append(
                    audit_shard(swm, shard=shard, config=config)
                )
                print(
                    f"[action-delay-data] audited {group} "
                    f"{index}/{len(group_shards)}",
                    flush=True,
                )
        else:
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
                        group_shards,
                        executor.map(
                            _audit_worker,
                            group_shards,
                            chunksize=1,
                        ),
                        strict=True,
                    ),
                    start=1,
                ):
                    if audit["scenario_id"] != shard.scenario_id:
                        raise RuntimeError(
                            "Audit worker returned a different identity"
                        )
                    audits.append(audit)
                    print(
                        f"[action-delay-data] audited {group} "
                        f"{index}/{len(group_shards)}",
                        flush=True,
                    )
        if not all(audit["passed"] for audit in audits):
            failed = [
                audit["scenario_id"]
                for audit in audits
                if not audit["passed"]
            ]
            raise RuntimeError(
                f"Action-delay shard audits failed: {failed}"
            )
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
            for shard, audit in zip(
                plans_by_group[group], audits, strict=True
            )
        ]
        catalog_stem = str(config["catalog_stems"][group])
        catalog_path = output_root / "catalogs" / f"{catalog_stem}.json"
        manifest_path = output_root / "manifests" / f"{catalog_stem}.jsonl"
        report_path = output_root / "reports" / f"{catalog_stem}.json"
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
            synthetic_dataset=plans_by_group[group][0].table_path,
            cache_dir=Path("/tmp/contextworld-action-delay-loader"),
        )
        scenario_reports = [
            {
                "scenario_id": audit["scenario_id"],
                "passed": audit["passed"],
                "checks": audit["checks"],
            }
            for audit in audits
        ]
        collection_status = {
            record["scenario_id"]: record["collection_status"]
            for record in records
        }
        group_report = {
            "schema_version": 1,
            "experiment": catalog_stem,
            "benchmark": config["benchmark"],
            "group": group,
            "passed": bool(
                loader["passed"]
                and all(audit["passed"] for audit in audits)
            ),
            "compile_only": False,
            "preflight_passed": True,
            "stable_worldmodel": {
                "repo": str(stable_repo),
                "commit": stable_commit,
            },
            "catalog": str(catalog_path.resolve()),
            "manifest": str(manifest_path.resolve()),
            "collection_status": collection_status,
            "scenarios": scenario_reports,
            "loader_compatibility": loader,
            "counts": {
                split: {
                    "shards": sum(
                        shard.split == split
                        for shard in plans_by_group[group]
                    ),
                    "episodes": sum(
                        len(shard.episodes)
                        for shard in plans_by_group[group]
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

    single_by_key = {
        (shard.split, shard.shard_index): (shard, audit)
        for shard, audit in zip(
            plans_by_group["action_delay_single"],
            audits_by_group["action_delay_single"],
            strict=True,
        )
    }
    multi_by_key = {
        (shard.split, shard.shard_index): (shard, audit)
        for shard, audit in zip(
            plans_by_group["action_delay_multi"],
            audits_by_group["action_delay_multi"],
            strict=True,
        )
    }
    paired_geometry_exact = set(single_by_key) == set(multi_by_key) and all(
        [
            asdict(left.template) == asdict(right.template)
            for left, right in zip(
                single_by_key[key][0].episodes,
                multi_by_key[key][0].episodes,
                strict=True,
            )
        ]
        for key in single_by_key
    )
    train_query_hashes = {
        value
        for group_audits in audits_by_group.values()
        for audit in group_audits
        for value in audit["query_pixels_sha256"]
    }
    overlap = sorted(train_query_hashes & eval_query_hashes)
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
    checks = {
        "all_shards_pass": all(
            audit["passed"]
            for audits in audits_by_group.values()
            for audit in audits
        ),
        "single_and_multi_geometry_exactly_paired": paired_geometry_exact,
        "validation_query_pixels_excluded": not overlap,
        "single_delay_support_exact": delay_support[
            "action_delay_single"
        ]
        == {"train": [2], "val": [2]},
        "multi_delay_support_exact": delay_support[
            "action_delay_multi"
        ]
        == {"train": [0, 2, 4], "val": [0, 2, 4]},
        "model_columns_are_pixels_and_actions_only": True,
        "each_episode_is_exactly_one_h3_clip": all(
            audit["checks"]["every_episode_is_one_h3_clip"]
            for audits in audits_by_group.values()
            for audit in audits
        ),
        "catalogs_and_manifests_published": set(artifacts) == set(GROUPS),
    }
    report = {
        "schema_version": 1,
        "benchmark": config["benchmark"],
        "status": "passed" if all(checks.values()) else "failed",
        "passed": all(checks.values()),
        "checks": checks,
        "identity": {
            "stable_worldmodel_repo": str(stable_repo),
            "stable_worldmodel_commit": stable_commit,
        },
        "history3": {
            "history_tokens": 3,
            "raw_steps_per_action_block": ACTION_BLOCK,
            "rows_per_episode": RAW_STEPS,
        },
        "delay_support": delay_support,
        "paired_geometry_audit": {
            "paired_shards": len(single_by_key),
            "paired_episodes": sum(
                len(shard.episodes)
                for shard, _ in single_by_key.values()
            ),
            "passed": paired_geometry_exact,
        },
        "validation_exclusion_audit": {
            "manifest": portable_contextworld_path(
                exclusion_path,
                repo_root=repo_root,
            ),
            "manifest_sha256": file_sha256(exclusion_path),
            "eval_query_hashes": len(eval_query_hashes),
            "training_query_hashes": len(train_query_hashes),
            "overlap": overlap,
            "passed": not overlap,
        },
        "artifacts_by_group": artifacts,
        "physical_counts": {
            "shards": len(all_shards),
            "episodes": sum(
                len(shard.episodes) for shard in all_shards
            ),
            "raw_rows": sum(
                len(shard.episodes) * RAW_STEPS
                for shard in all_shards
            ),
        },
    }
    write_json(output_root / "build_report.json", report)
    return report


__all__ = [
    "DIAGNOSTIC_KEYS",
    "GROUPS",
    "MODEL_KEYS",
    "MODEL_STEPS",
    "RAW_STEPS",
    "REQUIRED_COLUMNS",
    "SPLITS",
    "WATCHED_VARIATIONS",
    "ActionDelayEpisodePlan",
    "ActionDelayShardPlan",
    "audit_shard",
    "build_shard_plans",
    "build_training_release",
    "collect_shard",
    "directory_sha256",
    "training_template",
]
