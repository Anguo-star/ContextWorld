"""Public Development ICL readers stay contract-driven and non-formal."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from contextworld.benchmarks import bundle_development as development
from contextworld.benchmarks.adapters import AdapterProtocol, LatentWorldModelAdapter


class _Adapter(LatentWorldModelAdapter):
    def __init__(self, *, history: int, action_dim: int) -> None:
        self._protocol = AdapterProtocol(
            history_tokens=history,
            action_block_raw_steps=5,
            action_dim=action_dim,
            future_action_blocks=1,
        )

    @property
    def protocol(self) -> AdapterProtocol:
        return self._protocol

    @property
    def metadata(self) -> dict[str, str]:
        return {"adapter_id": "test-development-adapter"}

    def encode_pixels(self, pixels: np.ndarray, *, batch_size: int) -> np.ndarray:
        return np.asarray(pixels, dtype=np.float32).mean(axis=(1, 2))

    def rollout_latents(
        self,
        input_pixels: np.ndarray,
        raw_action_blocks: np.ndarray,
        *,
        batch_size: int,
    ) -> np.ndarray:
        return np.asarray(input_pixels, dtype=np.float32).mean(axis=(1, 2, 3))[
            :, None, :
        ]

    def frozen_state_hash(self) -> str:
        return "a" * 64


def _payload(
    task: str,
    *,
    history: int = 3,
    action_dim: int = 2,
    selection: dict[str, int] | None = None,
    members: tuple[Path, ...] = (),
) -> development.DevelopmentPayload:
    root = Path("/tmp/contextworld-bundle")
    return development.DevelopmentPayload(
        root=root,
        task=task,
        component={
            "component_id": task,
            "dataset_id": task,
            "history_length": history,
            "action_dimension": action_dim,
            "frameskip": 5,
        },
        evaluation={
            "selection": selection or {},
            "action_normalization": {
                "mean": [0.0] * action_dim,
                "std": [1.0] * action_dim,
            },
        },
        payload={"payload_id": "data", "payload_kind": "test"},
        members=members,
        manifest_sha256="b" * 64,
        task_registry_sha256="c" * 64,
        normalizer_path=None,
    )


def _pixels(value: int, *, frames: int) -> np.ndarray:
    result = np.zeros((frames, 2, 2, 3), dtype=np.uint8)
    result[1] = value
    result[-1] = value + 10
    return result


def test_speed_prefix_window_selection_skips_early_stops() -> None:
    """Speed selects valid 20-step prefixes from variable-length rollouts."""

    table = pa.table(
        {
            # Episode 0 is longer than the scoring window, episode 1 ends
            # early, episode 2 is exactly one window, and episode 3 is a
            # truncated tail.
            "episode_idx": pa.array(
                [0] * 24 + [1] * 19 + [2] * 20 + [3] * 2
            ),
            "step_idx": pa.array(
                list(range(24)) + list(range(19)) + list(range(20)) + [0, 1]
            ),
        }
    )

    complete, rows, _, _ = development._episode_rows(
        table,
        expected_steps=20,
        allow_prefix_clip=True,
    )

    assert complete == (0, 2)
    assert tuple(rows) == (0, 2)
    assert len(rows[0]) == 20
    assert len(rows[2]) == 20
    with pytest.raises(RuntimeError, match="valid 20-step prefix window"):
        development._episode_rows(
            table,
            expected_steps=20,
            selected_episode_ids=(1,),
            allow_prefix_clip=True,
        )


def test_door_selection_is_16_times_18_and_contract_checked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Path("/tmp/contextworld-bundle")
    members = tuple(
        root / f"hp-val-d{index:03d}-{mode}-deadbeef.lance"
        for index in range(16)
        for mode in ("blocked", "passable")
    )
    payload = _payload(
        "door",
        selection={
            "door_positions": 16,
            "complete_episodes_per_position": 18,
            "selected_pair_count": 288,
        },
        members=members,
    )

    def fake_read(path: Path, **_: object):
        available = tuple(range(80))
        is_blocked = "-blocked-" in path.name
        pixels = _pixels(1 if is_blocked else 2, frames=4)
        # Initial and query must match, while the history and target diverge.
        pixels[0] = 0
        pixels[2] = 0
        actions = np.zeros((4, 5, 2), dtype=np.float32)
        return available, {episode: (pixels, actions, None) for episode in available}

    monkeypatch.setattr(development, "_read_tworoom_episodes", fake_read)
    arrays = development._door_arrays(payload)

    assert len(arrays.pair_ids) == 288
    assert arrays.selection["groups"] == 16
    assert arrays.selection["pairs_per_group"] == 18


def test_action_delay_selection_is_6_times_10_times_5(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Path("/tmp/contextworld-bundle")
    members = tuple(
        root / f"ad-h7-paired-val-p{profile:03d}-d{delay}-deadbeef.lance"
        for profile in range(6)
        for delay in range(11)
    )
    payload = _payload(
        "action_delay",
        history=7,
        selection={
            "reference_condition": 0,
            "contrasts": list(range(1, 11)),
            "profiles": 6,
            "pairs_per_contrast_per_profile": 5,
            "selected_pair_count": 300,
        },
        members=members,
    )

    def fake_read(path: Path, **kwargs: object):
        delay = int(path.name.split("-d", 1)[1].split("-", 1)[0])
        available = tuple(range(160))
        selected = tuple(kwargs.get("selected_episode_ids") or available)
        pixels = _pixels(1 + delay, frames=8)
        pixels[0] = 0
        pixels[6] = 0
        actions = np.zeros((10, 5, 2), dtype=np.float32)
        return available, {episode: (pixels, actions, None) for episode in selected}

    monkeypatch.setattr(development, "_read_tworoom_episodes", fake_read)
    arrays = development._action_delay_arrays(payload)

    assert len(arrays.query_ids) == 30
    assert arrays.delay_values == tuple(range(11))
    assert arrays.member_pixels.shape[:3] == (30, 11, 8)
    assert arrays.selection["profiles"] == 6
    assert arrays.selection["queries_per_profile"] == 5
    assert arrays.selection["selected_contrast_pairs"] == 300


def test_speed_selection_is_96_times_3_and_is_not_counterfactual(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Path("/tmp/contextworld-bundle")
    payload = _payload(
        "speed",
        selection={
            "member_count": 96,
            "complete_windows_per_member": 3,
            "selected_case_count": 288,
        },
        members=tuple(root / f"speed-{index}.lance" for index in range(96)),
    )

    read_options: list[dict[str, object]] = []

    def fake_read(path: Path, **kwargs: object):
        read_options.append(dict(kwargs))
        available = tuple(range(16))
        speed = float(int(path.stem.split("-")[1]) % 8)
        pixels = _pixels(1, frames=4)
        actions = np.zeros((20, 5, 2), dtype=np.float32)
        return available, {episode: (pixels, actions, speed) for episode in available}

    monkeypatch.setattr(development, "_read_tworoom_episodes", fake_read)
    _, _, _, cases, selection = development._speed_cases(payload)

    assert len(cases) == 288
    assert selection["members"] == 96
    assert selection["windows_per_member"] == 3
    assert all(options["expected_steps"] == 20 for options in read_options)
    assert all(options["allow_prefix_clip"] is True for options in read_options)


@pytest.mark.parametrize(
    ("task", "action_dim"),
    [
        ("action_strength", 2),
        ("contact_friction", 2),
        ("motion_damping", 2),
        ("robot_arm_mass", 2),
        ("portal_exit", 2),
        ("cube_gripper_carry", 5),
    ],
)
def test_all_six_single_table_components_emit_256_development_pairs(
    monkeypatch: pytest.MonkeyPatch, task: str, action_dim: int
) -> None:
    payload = _payload(
        task,
        action_dim=action_dim,
        selection={"expected_pair_count": 256, "selected_pair_count": 256},
    )
    pair_ids = tuple(f"pair-{index}" for index in range(256))
    first = np.zeros((256, 4, 2, 2, 3), dtype=np.uint8)
    second = np.full((256, 4, 2, 2, 3), 2, dtype=np.uint8)
    actions = np.zeros((256, 4, 5, action_dim), dtype=np.float32)
    arrays = development._PairedArrays(
        pair_ids=pair_ids,
        first_pixels=first,
        second_pixels=second,
        raw_action_blocks=actions,
        first_label="first",
        second_label="second",
        selection={"pair_count": 256},
    )
    monkeypatch.setattr(development, "resolve_development_payload", lambda *a, **k: payload)
    monkeypatch.setattr(development, "_single_table_arrays", lambda *a, **k: arrays)

    result = development.evaluate_bundle_development_model(
        task=task,
        adapter=_Adapter(history=3, action_dim=action_dim),
        model_name="test-model",
        training_recipe="test",
        training_seed=1,
        benchmark_root="/tmp/contextworld-bundle",
    )

    assert result["result_kind"] == development.DEVELOPMENT_RESULT_KIND
    assert result["record_count"] == 256
    assert result["metrics"]["pair_count"] == 256
    assert result["protocol"]["official_scoreboard_row"] is False
    assert result["protocol"]["formal_pass_available"] is False
    assert "gate" not in result


class _ConstantAdapter(_Adapter):
    """A degenerate model whose futures ignore the history entirely."""

    def rollout_latents(
        self,
        input_pixels: np.ndarray,
        raw_action_blocks: np.ndarray,
        *,
        batch_size: int,
    ) -> np.ndarray:
        count = int(np.asarray(input_pixels).shape[0])
        return np.zeros((count, 1, 3), dtype=np.float32)


def _action_delay_payload() -> development.DevelopmentPayload:
    root = Path("/tmp/contextworld-bundle")
    members = tuple(
        root / f"ad-h7-paired-val-p{profile:03d}-d{delay}-deadbeef.lance"
        for profile in range(6)
        for delay in range(11)
    )
    return _payload(
        "action_delay",
        history=7,
        selection={
            "reference_condition": 0,
            "contrasts": list(range(1, 11)),
            "profiles": 6,
            "pairs_per_contrast_per_profile": 5,
            "selected_pair_count": 300,
        },
        members=members,
    )


def _delay_family_pixels(delay: int) -> np.ndarray:
    """Frames whose channel means identify the delay under mean pooling.

    Frames 0 and 6 stay shared across the family (initial and query frames),
    the interior history frames carry ``8 * delay``, and the future frame is
    the reference-history mean rounded to uint8, so a mean-pooling adapter
    maps every history to the encoded future of its own delay.
    """

    pixels = np.zeros((8, 2, 2, 3), dtype=np.uint8)
    pixels[1:6] = 8 * delay
    pixels[7] = int(round(40 * delay / 7))
    return pixels


def _fake_delay_family_read(path: Path, **kwargs: object):
    delay = int(path.name.split("-d", 1)[1].split("-", 1)[0])
    available = tuple(range(160))
    selected = tuple(kwargs.get("selected_episode_ids") or available)
    pixels = _delay_family_pixels(delay)
    actions = np.zeros((10, 5, 2), dtype=np.float32)
    return available, {episode: (pixels, actions, None) for episode in selected}


def test_action_delay_physical_groups_follow_the_formal_six_group_scheme(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _action_delay_payload()
    monkeypatch.setattr(
        development, "resolve_development_payload", lambda *a, **k: payload
    )
    monkeypatch.setattr(
        development, "_read_tworoom_episodes", _fake_delay_family_read
    )

    result = development.evaluate_bundle_development_model(
        task="action_delay",
        adapter=_Adapter(history=7, action_dim=2),
        model_name="oracle",
        training_recipe="test",
        training_seed=1,
        benchmark_root="/tmp/contextworld-bundle",
    )

    metrics = result["metrics"]
    assert (
        metrics["aggregation"]
        == "equal physical-group mean within query, then query mean"
    )
    assert metrics["physical_group_macro_accuracy"] == 1.0
    assert metrics["minimum_physical_group_accuracy"] == 1.0
    assert metrics["queries"] == 30
    assert metrics["history_conditions"] == 330
    expected_delays = {
        "0": [0],
        "1": [1],
        "2": [2],
        "3": [3],
        "4": [4],
        "5": [5, 6, 7, 8, 9, 10],
    }
    assert set(metrics["by_physical_group"]) == set(expected_delays)
    for group, delays in expected_delays.items():
        row = metrics["by_physical_group"][group]
        assert row["delays"] == delays
        assert row["accuracy"] == 1.0
        assert row["history_conditions"] == 30 * len(delays)
    assert set(metrics["confusion_counts"]) == {
        str(group) for group in range(6)
    }
    assert all(
        set(row) == {str(group) for group in range(6)}
        for row in metrics["confusion_counts"].values()
    )
    assert all(
        metrics["confusion_counts"][str(group)][str(group)]
        == 30 * len(delays)
        for group, delays in expected_delays.items()
    )
    assert "delay_0_correct_future_rate" not in metrics
    assert "delayed_correct_future_rate" not in metrics
    assert "gate" not in result
    assert result["record_count"] == 330
    assert result["protocol"]["formal_pass_available"] is False


def test_action_delay_constant_prediction_scores_chance_level_macro(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _action_delay_payload()
    monkeypatch.setattr(
        development, "resolve_development_payload", lambda *a, **k: payload
    )
    monkeypatch.setattr(
        development, "_read_tworoom_episodes", _fake_delay_family_read
    )

    result = development.evaluate_bundle_development_model(
        task="action_delay",
        adapter=_ConstantAdapter(history=7, action_dim=2),
        model_name="constant",
        training_recipe="test",
        training_seed=1,
        benchmark_root="/tmp/contextworld-bundle",
    )

    metrics = result["metrics"]
    # The constant zero future is always nearest delay 0's encoded future, so
    # exactly one of the six groups is ever credited: the 1/6 chance level.
    assert metrics["physical_group_macro_accuracy"] == pytest.approx(1 / 6)
    assert metrics["minimum_physical_group_accuracy"] == 0.0
    assert metrics["confusion_counts"]["0"]["0"] == 30
    assert (
        sum(metrics["confusion_counts"][str(group)]["0"] for group in range(6))
        == 330
    )
    assert "delay_0_correct_future_rate" not in metrics
    assert "delayed_correct_future_rate" not in metrics
    assert "gate" not in result
