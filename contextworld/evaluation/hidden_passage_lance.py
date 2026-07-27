from __future__ import annotations

import hashlib
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from contextworld.benchmarks.adapters import StableWorldModelLeWMAdapter
from contextworld.synthesis.lance import build_lance_writer, encode_frame

from .hidden_passage import (
    ACTION_BLOCK,
    HiddenPassageTemplate,
    simulate_template,
)
from .hidden_passage_env import (
    HIDDEN_PASSAGE_ENV_ID,
    PASSAGE_FACTOR,
    PASSAGE_RULES,
    make_hidden_passage_env,
    passage_open_value,
    register_hidden_passage_env,
)


RAW_STEPS = 20
MODEL_STEPS = 4
MODEL_KEYS = ("pixels", "action")
DIAGNOSTIC_KEYS = ("pixels", "action", "proprio")
WATCHED_VARIATIONS = (
    "agent.speed",
    "door.number",
    "door.position",
    PASSAGE_FACTOR,
)
REQUIRED_COLUMNS = {
    "pixels",
    "action",
    "proprio",
    "state",
    "goal_state",
    "terminated",
    "truncated",
    "variation_agent_speed",
    "variation_door_number",
    "variation_door_position",
    "variation_passage_open",
}


@dataclass(frozen=True)
class HiddenPassageLanceCase:
    case_id: str
    direction: str
    rule: str
    table_path: Path


class _ScriptedPolicy:
    def __init__(self, actions: np.ndarray) -> None:
        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape != (RAW_STEPS, 2):
            raise ValueError(
                f"Expected {(RAW_STEPS, 2)} scripted actions, got {actions.shape}"
            )
        self.actions = actions
        self.step = 0
        self.env: Any | None = None

    def set_env(self, env: Any) -> None:
        self.env = env

    def get_action(self, _: dict[str, Any]) -> np.ndarray:
        if self.env is None:
            raise RuntimeError("Scripted policy has no environment")
        if self.step >= len(self.actions):
            raise RuntimeError("World requested more than 20 scripted actions")
        action = self.actions[self.step]
        self.step += 1
        return np.repeat(action[None, :], self.env.num_envs, axis=0)


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
    """Account for World.collect dropping reset and rotating actions left."""

    return np.concatenate(
        [
            np.zeros((1, 2), dtype=np.float32),
            np.asarray(reference["history_actions"], dtype=np.float32).reshape(
                -1, 2
            ),
            np.asarray(reference["query_action"], dtype=np.float32),
            np.zeros((4, 2), dtype=np.float32),
        ],
        axis=0,
    )


def _variation_values(
    template: HiddenPassageTemplate,
    *,
    rule: str,
) -> dict[str, Any]:
    return {
        "agent.speed": np.asarray([5.0], dtype=np.float32),
        "door.number": 1,
        "door.position": np.asarray(
            [template.door_position] * 3,
            dtype=np.int64,
        ),
        PASSAGE_FACTOR: PASSAGE_RULES[rule],
    }


def collect_hidden_passage_lance_case(
    swm: Any,
    *,
    template: HiddenPassageTemplate,
    rule: str,
    table_path: Path,
    pixel_codec: dict[str, Any],
) -> dict[str, Any]:
    """Collect one exact History-3 episode through StableWM World."""

    if rule not in PASSAGE_RULES:
        raise ValueError(f"Unknown hidden-passage rule {rule!r}")
    if table_path.exists():
        raise FileExistsError(table_path)

    reference = simulate_template(template, rule=rule)
    actions = _collection_actions(reference)
    register_hidden_passage_env()
    table_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="contextworld-h3-lance-",
        dir="/tmp",
    ) as temporary:
        staged_path = Path(temporary) / table_path.name
        world = swm.World(
            HIDDEN_PASSAGE_ENV_ID,
            num_envs=1,
            max_episode_steps=RAW_STEPS,
            image_shape=(224, 224),
            render_mode="rgb_array",
        )
        policy = _ScriptedPolicy(actions)
        world.set_policy(policy)
        try:
            writer = build_lance_writer(
                swm,
                staged_path,
                pixel_codec=pixel_codec,
            )
            world.collect(
                episodes=1,
                seed=int(template.simulator_seed),
                options={
                    "variation": WATCHED_VARIATIONS,
                    "variation_values": _variation_values(
                        template,
                        rule=rule,
                    ),
                    "state": np.asarray(
                        template.reset_state,
                        dtype=np.float32,
                    ),
                    "target_state": np.asarray(
                        template.goal_state,
                        dtype=np.float32,
                    ),
                },
                writer=writer,
                progress=False,
            )
        finally:
            world.close()
        shutil.copytree(staged_path, table_path)

    if policy.step != RAW_STEPS:
        raise RuntimeError(
            f"World used {policy.step} actions instead of {RAW_STEPS}"
        )
    return reference


def _decoded_pixels(episode: dict[str, Any]) -> np.ndarray:
    pixels = episode["pixels"].detach().cpu().numpy()
    return np.transpose(pixels, (0, 2, 3, 1)).astype(np.uint8)


def _tensor_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _allclose_constant(value: np.ndarray, expected: Any) -> bool:
    rows = np.asarray(value).reshape(len(value), -1)
    target = np.asarray(expected).reshape(-1)
    return bool(
        rows.shape[1] == target.size
        and np.allclose(rows, target[None, :], atol=0.0, rtol=0.0)
    )


def _replay_rows(
    *,
    pixels: np.ndarray,
    states: np.ndarray,
    goals: np.ndarray,
    actions: np.ndarray,
    passage_open: int,
    speed: float,
    door_position: np.ndarray,
    door_number: int,
    pixel_blobs: np.ndarray,
    pixel_codec: dict[str, Any],
    opposite_rule: bool = False,
) -> dict[str, Any]:
    rule = 1 - int(passage_open) if opposite_rule else int(passage_open)
    env = make_hidden_passage_env(render_mode="rgb_array")
    state_mismatches = 0
    decoded_pixel_mismatches = 0
    encoded_pixel_mismatches = 0
    maximum_state_error = 0.0
    try:
        env.reset(
            seed=0,
            options={
                "variation": (),
                "variation_values": {
                    "agent.speed": np.asarray([speed], dtype=np.float32),
                    "door.number": int(door_number),
                    "door.position": np.asarray(
                        door_position,
                        dtype=np.int64,
                    ),
                },
                "state": states[0],
                "target_state": goals[0],
            },
        )
        env.restore_contextworld_hidden_passage(
            passage_open=rule,
            state=states[0],
            goal_state=goals[0],
        )
        if env.passage_open != rule:
            raise RuntimeError("Hidden-passage rule readback differs")

        rendered = np.asarray(env.render(), dtype=np.uint8)
        decoded_pixel_mismatches += int(
            not np.array_equal(rendered, pixels[0])
        )
        encoded_pixel_mismatches += int(
            encode_frame(rendered, pixel_codec) != pixel_blobs[0]
        )
        for row in range(RAW_STEPS - 1):
            observation, _, _, _, _ = env.step(actions[row])
            observed_state = np.asarray(observation[:2], dtype=np.float32)
            error = float(np.max(np.abs(observed_state - states[row + 1])))
            maximum_state_error = max(maximum_state_error, error)
            state_mismatches += int(
                not np.array_equal(observed_state, states[row + 1])
            )
            rendered = np.asarray(env.render(), dtype=np.uint8)
            decoded_pixel_mismatches += int(
                not np.array_equal(rendered, pixels[row + 1])
            )
            encoded_pixel_mismatches += int(
                encode_frame(rendered, pixel_codec)
                != pixel_blobs[row + 1]
            )
    finally:
        env.close()
    passed = not any(
        (
            state_mismatches,
            decoded_pixel_mismatches,
            encoded_pixel_mismatches,
        )
    )
    return {
        "passed": passed,
        "rule_restored": rule,
        "state_mismatches": state_mismatches,
        "decoded_pixel_mismatches": decoded_pixel_mismatches,
        "encoded_pixel_mismatches": encoded_pixel_mismatches,
        "maximum_state_absolute_error": maximum_state_error,
    }


def audit_hidden_passage_lance_case(
    swm: Any,
    *,
    case: HiddenPassageLanceCase,
    template: HiddenPassageTemplate,
    reference: dict[str, Any],
    pixel_codec: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Reload, project and replay one collected Lance table."""

    raw = swm.data.LanceDataset(path=case.table_path)
    columns = set(raw.column_names)
    episode_lengths = [int(value) for value in raw.lengths]
    episode = raw.load_episode(0)
    pixels = _decoded_pixels(episode)
    actions = _tensor_numpy(episode["action"]).astype(np.float32)
    states = _tensor_numpy(episode["state"]).astype(np.float32)
    proprio = _tensor_numpy(episode["proprio"]).astype(np.float32)
    goals = _tensor_numpy(episode["goal_state"]).astype(np.float32)
    truncated = _tensor_numpy(episode["truncated"]).reshape(-1).astype(bool)
    terminated = _tensor_numpy(episode["terminated"]).reshape(-1).astype(bool)
    pixel_blobs = raw.get_col_data("pixels")

    passage = raw.get_col_data("variation_passage_open")
    speed = raw.get_col_data("variation_agent_speed")
    door_number = raw.get_col_data("variation_door_number")
    door_position = raw.get_col_data("variation_door_position")

    diagnostic = swm.data.LanceDataset(
        path=case.table_path,
        keys_to_load=list(DIAGNOSTIC_KEYS),
        frameskip=ACTION_BLOCK,
        num_steps=MODEL_STEPS,
    )
    strict_model = swm.data.LanceDataset(
        path=case.table_path,
        keys_to_load=list(MODEL_KEYS),
        frameskip=ACTION_BLOCK,
        num_steps=MODEL_STEPS,
    )
    diagnostic_sample = diagnostic[0]
    model_sample = strict_model[0]
    model_pixels = np.transpose(
        _tensor_numpy(model_sample["pixels"]),
        (0, 2, 3, 1),
    ).astype(np.uint8)
    model_actions = _tensor_numpy(model_sample["action"]).reshape(
        MODEL_STEPS,
        ACTION_BLOCK,
        2,
    )
    model_proprio = _tensor_numpy(diagnostic_sample["proprio"]).astype(
        np.float32
    )

    expected_model_pixels = np.concatenate(
        [
            np.asarray(reference["history_pixels"], dtype=np.uint8),
            np.asarray(reference["target_pixels"], dtype=np.uint8)[None],
        ],
        axis=0,
    )
    expected_model_states = np.concatenate(
        [
            np.asarray(reference["history_states"], dtype=np.float32),
            np.asarray(reference["target_state"], dtype=np.float32)[None],
        ],
        axis=0,
    )
    expected_model_actions = _model_blocks(reference)

    exact_replay = _replay_rows(
        pixels=pixels,
        states=states,
        goals=goals,
        actions=actions,
        passage_open=passage_open_value(passage[0]),
        speed=float(np.asarray(speed[0]).reshape(-1)[0]),
        door_position=np.asarray(door_position[0]).reshape(-1),
        door_number=int(np.asarray(door_number[0]).reshape(-1)[0]),
        pixel_blobs=pixel_blobs,
        pixel_codec=pixel_codec,
    )
    wrong_rule_replay = _replay_rows(
        pixels=pixels,
        states=states,
        goals=goals,
        actions=actions,
        passage_open=passage_open_value(passage[0]),
        speed=float(np.asarray(speed[0]).reshape(-1)[0]),
        door_position=np.asarray(door_position[0]).reshape(-1),
        door_number=int(np.asarray(door_number[0]).reshape(-1)[0]),
        pixel_blobs=pixel_blobs,
        pixel_codec=pixel_codec,
        opposite_rule=True,
    )

    checks = {
        "required_columns_present": REQUIRED_COLUMNS <= columns,
        "one_episode_of_20_rows": episode_lengths == [RAW_STEPS],
        "stored_rule_constant_and_exact": _allclose_constant(
            passage,
            PASSAGE_RULES[case.rule],
        ),
        "stored_speed_constant_and_exact": _allclose_constant(speed, [5.0]),
        "stored_door_number_constant_and_exact": _allclose_constant(
            door_number,
            [1],
        ),
        "stored_door_position_constant_and_exact": _allclose_constant(
            door_position,
            [template.door_position] * 3,
        ),
        "state_and_proprio_identical": np.array_equal(states, proprio),
        "goal_constant_and_exact": _allclose_constant(
            goals,
            template.goal_state,
        ),
        "only_last_row_truncated": bool(
            len(truncated) == RAW_STEPS
            and not truncated[:-1].any()
            and truncated[-1]
        ),
        "never_terminated": bool(not terminated.any()),
        "diagnostic_loader_has_one_clip": len(diagnostic) == 1,
        "strict_model_loader_has_one_clip": len(strict_model) == 1,
        "diagnostic_loader_keys_exact": (
            tuple(diagnostic_sample) == DIAGNOSTIC_KEYS
        ),
        "strict_model_loader_keys_exact": tuple(model_sample) == MODEL_KEYS,
        "strict_model_loader_excludes_rule": not any(
            key.startswith("variation") for key in model_sample
        ),
        "model_pixel_shape": tuple(model_sample["pixels"].shape)
        == (4, 3, 224, 224),
        "model_action_shape": tuple(model_sample["action"].shape) == (4, 10),
        "diagnostic_proprio_shape": tuple(
            diagnostic_sample["proprio"].shape
        )
        == (4, 2),
        "history3_and_future_pixels_exact": np.array_equal(
            model_pixels,
            expected_model_pixels,
        ),
        "history3_and_future_states_exact": np.array_equal(
            model_proprio,
            expected_model_states,
        ),
        "history3_action_blocks_exact": np.array_equal(
            model_actions,
            expected_model_actions,
        ),
        "unused_fourth_action_block_zero": bool(
            np.array_equal(model_actions[-1], np.zeros((5, 2), np.float32))
        ),
        "restored_rule_exact_replay": exact_replay["passed"],
        "opposite_rule_fails_replay": not wrong_rule_replay["passed"],
    }
    report = {
        "case_id": case.case_id,
        "direction": case.direction,
        "rule": case.rule,
        "passed": all(checks.values()),
        "checks": checks,
        "table": str(case.table_path),
        "raw_columns": sorted(columns),
        "episode_lengths": episode_lengths,
        "model_sample_shapes": {
            key: list(model_sample[key].shape) for key in MODEL_KEYS
        },
        "diagnostic_sample_shapes": {
            key: list(diagnostic_sample[key].shape)
            for key in DIAGNOSTIC_KEYS
        },
        "exact_replay": exact_replay,
        "wrong_rule_replay": wrong_rule_replay,
    }
    assets = {
        "model_pixels": model_pixels,
        "model_actions": model_actions,
        "model_proprio": model_proprio,
    }
    return report, assets


def audit_hidden_passage_lance_pairs(
    case_assets: dict[str, dict[str, np.ndarray]],
) -> dict[str, Any]:
    by_direction: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for case_id, assets in case_assets.items():
        direction, rule = case_id.rsplit("-", 1)
        by_direction.setdefault(direction, {})[rule] = assets

    results: dict[str, Any] = {}
    for direction, rules in sorted(by_direction.items()):
        passable = rules["passable"]
        blocked = rules["blocked"]
        middle_gap = float(
            np.linalg.norm(
                passable["model_proprio"][1]
                - blocked["model_proprio"][1]
            )
        )
        future_gap = float(
            np.linalg.norm(
                passable["model_proprio"][3]
                - blocked["model_proprio"][3]
            )
        )
        checks = {
            "initial_pixels_identical": np.array_equal(
                passable["model_pixels"][0],
                blocked["model_pixels"][0],
            ),
            "probe_result_pixels_different": not np.array_equal(
                passable["model_pixels"][1],
                blocked["model_pixels"][1],
            ),
            "query_pixels_identical": np.array_equal(
                passable["model_pixels"][2],
                blocked["model_pixels"][2],
            ),
            "future_pixels_different": not np.array_equal(
                passable["model_pixels"][3],
                blocked["model_pixels"][3],
            ),
            "all_model_actions_identical": np.array_equal(
                passable["model_actions"],
                blocked["model_actions"],
            ),
            "middle_state_gap_is_8_5_px": middle_gap == 8.5,
            "future_state_gap_is_25_px": future_gap == 25.0,
        }
        results[direction] = {
            "passed": all(checks.values()),
            "checks": checks,
            "middle_state_gap_px": middle_gap,
            "future_state_gap_px": future_gap,
        }
    return {
        "passed": bool(results) and all(
            result["passed"] for result in results.values()
        ),
        "directions": results,
    }


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(str(tuple(array.shape)).encode("utf-8"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def audit_actual_lewm_adapter(
    *,
    checkpoint: Path,
    normalizer: Path,
    repo_root: Path,
    stablewm_repo: str,
    stablewm_ref: str,
    device: str,
    case_assets: dict[str, dict[str, np.ndarray]],
) -> dict[str, Any]:
    """Run the real public adapter without exposing privileged Lance fields."""

    adapter = StableWorldModelLeWMAdapter.from_checkpoint(
        checkpoint,
        normalizer=normalizer,
        repo_root=repo_root,
        stablewm_repo=stablewm_repo,
        stablewm_ref=stablewm_ref,
        device=device,
    )
    ordered = [case_assets[key] for key in sorted(case_assets)]
    input_pixels = np.stack([asset["model_pixels"][:3] for asset in ordered])
    raw_actions = np.stack([asset["model_actions"][:3] for asset in ordered])
    target_pixels = np.stack([asset["model_pixels"][3] for asset in ordered])

    encode_keys: list[list[str]] = []
    tensor_inputs: dict[str, list[list[int]]] = {
        "encoder": [],
        "action_encoder": [],
        "predictor": [],
    }
    model = adapter.model
    original_encode = model.encode

    def recording_encode(info: dict[str, Any]):
        encode_keys.append(sorted(info))
        return original_encode(info)

    model.encode = recording_encode
    hooks = [
        model.encoder.register_forward_pre_hook(
            lambda _module, args: tensor_inputs["encoder"].append(
                list(args[0].shape)
            )
        ),
        model.action_encoder.register_forward_pre_hook(
            lambda _module, args: tensor_inputs["action_encoder"].append(
                list(args[0].shape)
            )
        ),
        model.predictor.register_forward_pre_hook(
            lambda _module, args: tensor_inputs["predictor"].append(
                [list(value.shape) for value in args]
            )
        ),
    ]
    before = adapter.frozen_state_hash()
    try:
        prediction = adapter.rollout_latents(
            input_pixels,
            raw_actions,
            batch_size=len(input_pixels),
        )
        target = adapter.encode_pixels(
            target_pixels,
            batch_size=len(target_pixels),
        )
    finally:
        model.encode = original_encode
        for hook in hooks:
            hook.remove()
    after = adapter.frozen_state_hash()

    forbidden = {
        "passage.open",
        "passage_open",
        "variation_passage_open",
        "state",
        "proprio",
        "template_id",
    }
    observed_keys = sorted({key for keys in encode_keys for key in keys})
    checks = {
        "protocol_is_history3_action_block5": (
            adapter.protocol.history_tokens == 3
            and adapter.protocol.action_block_raw_steps == 5
        ),
        "model_encode_keys_are_pixels_and_action": (
            bool(encode_keys)
            and all(set(keys) <= {"pixels", "action"} for keys in encode_keys)
            and {"pixels", "action"} in [set(keys) for keys in encode_keys]
        ),
        "forbidden_fields_never_reach_model": not (
            forbidden & set(observed_keys)
        ),
        "prediction_shape": tuple(prediction.shape)
        == (len(ordered), 1, prediction.shape[-1]),
        "target_shape": tuple(target.shape)
        == (len(ordered), target.shape[-1]),
        "prediction_finite": bool(np.isfinite(prediction).all()),
        "target_finite": bool(np.isfinite(target).all()),
        "model_state_unchanged": before == after,
        "encoder_was_called": bool(tensor_inputs["encoder"]),
        "action_encoder_was_called": bool(tensor_inputs["action_encoder"]),
        "predictor_was_called": bool(tensor_inputs["predictor"]),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "adapter": adapter.metadata,
        "model_encode_keys_by_call": encode_keys,
        "observed_model_encode_keys": observed_keys,
        "forbidden_fields": sorted(forbidden),
        "tensor_inputs": tensor_inputs,
        "prediction_shape": list(prediction.shape),
        "target_shape": list(target.shape),
        "prediction_sha256": _array_sha256(prediction),
        "target_sha256": _array_sha256(target),
        "model_state_sha256_before": before,
        "model_state_sha256_after": after,
    }


def case_identity(case: HiddenPassageLanceCase) -> dict[str, Any]:
    payload = asdict(case)
    payload["table_path"] = str(case.table_path)
    return payload


__all__ = [
    "DIAGNOSTIC_KEYS",
    "HiddenPassageLanceCase",
    "MODEL_KEYS",
    "RAW_STEPS",
    "audit_actual_lewm_adapter",
    "audit_hidden_passage_lance_case",
    "audit_hidden_passage_lance_pairs",
    "case_identity",
    "collect_hidden_passage_lance_case",
]
