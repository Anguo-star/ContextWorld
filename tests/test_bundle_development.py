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


def test_read_dev_episodes_decodes_each_episodes_own_frames(
    tmp_path: Path,
) -> None:
    """Two episodes in one real Lance table keep their own frames.

    Regression guard: an earlier reader version indexed the pixel column by
    the raw step values, which silently gave every episode the first
    episode's frames; the per-episode rows must select the blobs instead.
    """

    try:
        import lance
        from PIL import Image
    except ImportError:  # pragma: no cover - environment dependent
        pytest.skip("lance/Pillow not available")

    def png(value: int) -> bytes:
        import io

        buffer = io.BytesIO()
        Image.fromarray(
            np.full((2, 2, 3), value, dtype=np.uint8), mode="RGB"
        ).save(buffer, format="PNG")
        return buffer.getvalue()

    steps = 4
    table = pa.table(
        {
            "episode_idx": pa.array(
                [0] * steps + [1] * steps, type=pa.int64()
            ),
            "step_idx": pa.array(
                list(range(steps)) * 2, type=pa.int64()
            ),
            "pixels": [png(10 + step) for step in range(steps)]
            + [png(50 + step) for step in range(steps)],
            "action": pa.array(
                [[0.0, 0.0]] * (2 * steps),
                type=pa.list_(pa.float32(), 2),
            ),
            "dev_query_id": ["q-a"] * steps + ["q-b"] * steps,
            "dev_eval_seed": [[42.0]] * (2 * steps),
        }
    )
    path = tmp_path / "regression.lance"
    lance.write_dataset(table, path)

    records = development._read_dev_episodes(
        path,
        expected_steps=steps,
        frame_steps=(0, 3),
        string_columns=("dev_query_id",),
        scalar_columns=("dev_eval_seed",),
    )

    assert [record["episode"] for record in records] == [0, 1]
    assert records[0]["dev_query_id"] == "q-a"
    assert records[1]["dev_query_id"] == "q-b"
    assert records[0]["dev_eval_seed"] == 42.0
    # Each episode reads its own pixel values at the requested steps.
    assert records[0]["frames"][0].mean() == 10
    assert records[0]["frames"][1].mean() == 13
    assert records[1]["frames"][0].mean() == 50
    assert records[1]["frames"][1].mean() == 53
    assert records[0]["actions"].shape == (steps, 2)


_DOOR_CONDITIONS = (
    "observed_passable",
    "observed_blocked",
    "did_not_attempt_crossing",
)


def _door_dev_payload() -> development.DevelopmentPayload:
    root = Path("/tmp/contextworld-bundle")
    members = tuple(
        root / f"hpdev-d{index:03d}-{condition}-deadbeef.lance"
        for index in range(40)
        for condition in _DOOR_CONDITIONS
    )
    return _payload("door", members=members)


def _fake_door_read(path: Path, **_: object):
    """Synthetic reset-isolated door episodes keyed by channel-mean values.

    ``passable``/``blocked`` encode the rule targets and each condition's
    history; the no-attempt history uses a third value and repeats the
    passable target at its final frame, exactly like the real payload.
    """

    door = int(path.name.split("-")[1][1:])
    condition = path.name.split("-")[2]
    records = []
    for index in range(300):
        if index % 40 != door:
            continue
        seed = 42 + index // 50
        direction = (
            "left_to_right" if (index % 50) < 25 else "right_to_left"
        )
        passable = 30 + (index % 7)
        blocked = 100 + (index % 11)
        no_attempt = 160 + (index % 5)
        values = {
            "observed_passable": (passable, passable),
            "observed_blocked": (blocked, blocked),
            "did_not_attempt_crossing": (no_attempt, passable),
        }
        history_value, target_value = values[condition]
        frames = np.zeros((4, 2, 2, 3), dtype=np.uint8)
        frames[:3] = history_value
        frames[3] = target_value
        records.append(
            {
                "episode": len(records),
                "frames": frames,
                "actions": np.zeros((20, 2), dtype=np.float32),
                "dev_query_id": f"s{seed}-e{index % 50:03d}-static-{index}",
                "dev_static_query_id": f"static-{index}",
                "dev_direction": direction,
                "dev_env_rule": (
                    "blocked"
                    if condition == "observed_blocked"
                    else "passable"
                ),
                "dev_template_id": f"hp-d{door:03d}-t{index}",
                "dev_eval_seed": float(seed),
                "dev_evaluation_index": float(index % 50),
                "dev_door_position": float(door),
            }
        )
    return records


def test_door_structural_selection_covers_the_public_test_stratification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _door_dev_payload()
    monkeypatch.setattr(development, "_read_dev_episodes", _fake_door_read)
    assets, selection = development._door_structural_assets(payload)

    assert len(assets) == 300
    assert selection["unique_queries"] == 300
    assert selection["per_direction_per_eval_seed"] == 25
    strata: dict[tuple[int, str], int] = {}
    for asset in assets:
        key = (asset["eval_seed"], asset["direction"])
        strata[key] = strata.get(key, 0) + 1
    assert len(strata) == 12
    assert set(strata.values()) == {25}
    first = assets[0]
    assert set(first["histories"]) == set(_DOOR_CONDITIONS)
    assert first["histories"]["observed_passable"].shape == (3, 2, 2, 3)
    assert list(first["actions"].values())[0].shape == (3, 5, 2)


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


def test_action_delay_dev_reader_preserves_real_lance_identity() -> None:
    """The released 1.0.3-rc1 Lance rows keep query seed/room/direction."""

    bundle_root = Path(
        "/opt/huawei/explorer-env/dataset/ag_data/data/world_model/ContextWorld-v1"
    )
    if not bundle_root.is_dir():
        pytest.skip("ContextWorld-v1 bundle is not mounted")
    try:
        payload = development.resolve_development_payload(
            bundle_root, task="action_delay"
        )
        available, episodes = development._read_tworoom_episodes(
            payload.members[0],
            expected_steps=50,
            frame_steps=tuple(range(0, 50, 5)),
            selected_episode_ids=(0,),
            metadata_columns=(
                "dev_eval_seed",
                "dev_room",
                "dev_direction",
                "dev_query_id",
                "dev_delay",
            ),
        )
    except (ImportError, RuntimeError) as exc:
        pytest.skip(f"Lance/Pillow or released payload unavailable: {exc}")

    assert 0 in available
    pixels, actions, _, metadata = episodes[0]
    assert pixels.shape[0] == 10
    assert actions.shape[0] == 10
    assert metadata == {
        "dev_eval_seed": 52.0,
        "dev_room": "right",
        "dev_direction": "down",
        "dev_query_id": "action-delay-h7-val-s52-q00",
        "dev_delay": 0.0,
    }


def test_action_delay_auxiliary_h2_h3_reuses_test_horizon_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A three-future adapter receives the 9-block prefix once for h1/h2/h3."""

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
            "pairs_per_contrast_per_profile": 50,
            "selected_pair_count": 3000,
        },
        members=members,
    )

    def fake_read(path: Path, **kwargs: object):
        delay = int(path.name.split("-d", 1)[1].split("-", 1)[0])
        profile = int(path.name.split("val-p", 1)[1][:3])
        available = tuple(range(160))
        selected = tuple(kwargs.get("selected_episode_ids") or available)
        records = {}
        for episode in selected:
            frames = np.zeros((10, 2, 2, 3), dtype=np.uint8)
            frames[1] = 1 + delay
            frames[6] = 0
            frames[7] = 10 + delay
            frames[8] = 20 + delay
            frames[9] = 30 + delay
            metadata = {
                "dev_eval_seed": float(52 + profile),
                "dev_room": "left" if episode % 2 else "right",
                "dev_direction": "up" if episode % 2 else "down",
                "dev_query_id": (
                    f"action-delay-h7-val-s{52 + profile}"
                    f"-q{episode:02d}"
                ),
                "dev_delay": float(delay),
            }
            actions = np.zeros((10, 5, 2), dtype=np.float32)
            records[episode] = (frames, actions, None, metadata)
        return available, records

    class ThreeFutureAdapter(_Adapter):
        def __init__(self) -> None:
            super().__init__(history=7, action_dim=2)
            self._protocol = AdapterProtocol(
                history_tokens=7,
                action_block_raw_steps=5,
                action_dim=2,
                future_action_blocks=3,
            )

        def rollout_latents(
            self,
            input_pixels: np.ndarray,
            raw_action_blocks: np.ndarray,
            *,
            batch_size: int,
        ) -> np.ndarray:
            del raw_action_blocks, batch_size
            values = np.asarray(input_pixels, dtype=np.float32)[:, -1].mean(
                axis=(1, 2)
            )
            return np.repeat(values[:, None, :], 3, axis=1)

    monkeypatch.setattr(
        development, "resolve_development_payload", lambda *a, **k: payload
    )
    monkeypatch.setattr(development, "_read_tworoom_episodes", fake_read)
    result = development.evaluate_bundle_development_model(
        task="action_delay",
        adapter=ThreeFutureAdapter(),
        model_name="three-future",
        training_recipe="test",
        training_seed=1,
        benchmark_root="/tmp/contextworld-bundle",
        batch_size=32,
    )

    metrics = result["metrics"]
    assert result["record_count"] == 3300
    assert metrics["auxiliary_horizons"]["available"] == [2, 3]
    assert set(metrics["by_horizon"]) == {"1", "2", "3"}
    assert metrics["by_horizon"]["2"]["overall"]["query_target_units"] == 3300
    assert metrics["by_horizon"]["3"]["overall"]["query_target_units"] == 3300
    assert metrics["eval_seed_query_counts"] == {
        "52": 50,
        "53": 50,
        "54": 50,
        "55": 50,
        "56": 50,
        "57": 50,
    }


_SPEED_TRACK = "seen_for_multi"
_SPEED_CONDITION_SPEEDS = {
    "history_low": 10.0,
    "history_mid": 20.0,
    "history_high": 30.0,
}


def _speed_dev_payload() -> development.DevelopmentPayload:
    root = Path("/tmp/contextworld-bundle")
    members = tuple(
        root / f"twmsdev-{_SPEED_TRACK}-{condition}-deadbeef.lance"
        for condition in _SPEED_CONDITION_SPEEDS
    )
    return _payload("speed", members=members)


def _fake_speed_read(path: Path, **_: object):
    """Synthetic track episodes whose token-2 mean is the condition speed.

    Frame 0 is the shared reset, frame 5 carries the condition speed, frame
    10 is the shared query, and frames 15-35 all encode the reference speed,
    mirroring the counterfactual splice of the real payload.
    """

    condition = path.name.split("-")[2]
    condition_speed = _SPEED_CONDITION_SPEEDS[condition]
    records = []
    for static_index in range(300):
        seed = 42 + static_index // 50
        for reference_speed in (10.0, 20.0, 30.0):
            frames = np.zeros((8, 2, 2, 3), dtype=np.uint8)
            frames[0] = 5
            frames[1] = int(condition_speed)
            # Kept in 100..199 so it never collides with the speed values
            # (10/20/30) after the uint8 cast; the legacy context-free
            # diagnostic compares against exactly those values.
            frames[2] = 100 + (static_index % 100)
            frames[3:] = int(reference_speed)
            records.append(
                {
                    "episode": len(records),
                    "frames": frames,
                    "actions": np.zeros((40, 2), dtype=np.float32),
                    "dev_query_id": f"s{seed}-q{static_index}-v{reference_speed:g}",
                    "dev_static_query_id": f"static-{static_index}",
                    "dev_track": _SPEED_TRACK,
                    "dev_condition": condition,
                    "dev_template_id": f"tpl-{static_index}",
                    "dev_eval_seed": float(seed),
                    "dev_evaluation_index": float(static_index % 50),
                    "dev_reference_speed": reference_speed,
                    "dev_condition_speed": condition_speed,
                }
            )
    return records


def test_speed_structural_families_group_every_reference_speed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _speed_dev_payload()
    monkeypatch.setattr(development, "_read_dev_episodes", _fake_speed_read)
    tracks, selection = development._speed_structural_tracks(payload)

    assert set(tracks) == {_SPEED_TRACK}
    families = tracks[_SPEED_TRACK]
    assert len(families) == 300 * 3
    for (static_id, reference_speed), slot in families.items():
        assert set(slot) == set(_SPEED_CONDITION_SPEEDS)
        matching = [
            condition
            for condition in _SPEED_CONDITION_SPEEDS
            if _SPEED_CONDITION_SPEEDS[condition] == reference_speed
        ]
        assert len(matching) == 1
    facts = selection["tracks"][_SPEED_TRACK]
    assert facts["static_queries"] == 300
    assert facts["reference_speeds"] == [10.0, 20.0, 30.0]
    assert facts["condition_speeds"] == _SPEED_CONDITION_SPEEDS


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


_FORBIDDEN_RESULT_KEYS = {
    "gate",
    "gates",
    "decision",
    "passed",
    "verdict",
    "checks",
}


def _assert_no_decision_fields(value: object) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            assert key not in _FORBIDDEN_RESULT_KEYS, key
            _assert_no_decision_fields(item)
    elif isinstance(value, list):
        for item in value:
            _assert_no_decision_fields(item)


def _functional_adapter(
    *,
    futures: int,
    rollout,
) -> _Adapter:
    adapter = _Adapter(history=3, action_dim=2)
    adapter._protocol = AdapterProtocol(
        history_tokens=3,
        action_block_raw_steps=5,
        action_dim=2,
        future_action_blocks=futures,
    )
    adapter.rollout_latents = rollout
    return adapter


def _perfect_token_rollout(futures: int):
    """A model that reads the middle history token and nothing else."""

    def rollout(
        input_pixels: np.ndarray,
        raw_action_blocks: np.ndarray,
        *,
        batch_size: int,
    ) -> np.ndarray:
        value = np.asarray(input_pixels, dtype=np.float32)[:, 1].mean(
            axis=(1, 2)
        )
        return np.repeat(value[:, None, :], futures, axis=1)

    return rollout


def _constant_rollout(futures: int):
    def rollout(
        input_pixels: np.ndarray,
        raw_action_blocks: np.ndarray,
        *,
        batch_size: int,
    ) -> np.ndarray:
        count = int(np.asarray(input_pixels).shape[0])
        return np.zeros((count, futures, 3), dtype=np.float32)

    return rollout


def _noise_rollout(futures: int, seed: int = 20260909):
    state = np.random.default_rng(seed)

    def rollout(
        input_pixels: np.ndarray,
        raw_action_blocks: np.ndarray,
        *,
        batch_size: int,
    ) -> np.ndarray:
        count = int(np.asarray(input_pixels).shape[0])
        return state.normal(size=(count, futures, 3)).astype(np.float32)

    return rollout


def _evaluate_structural(
    monkeypatch: pytest.MonkeyPatch,
    *,
    task: str,
    payload: development.DevelopmentPayload,
    fake_read,
    adapter,
) -> dict[str, object]:
    monkeypatch.setattr(
        development, "resolve_development_payload", lambda *a, **k: payload
    )
    monkeypatch.setattr(development, "_read_dev_episodes", fake_read)
    return development.evaluate_bundle_development_model(
        task=task,
        adapter=adapter,
        model_name="test-model",
        training_recipe="test",
        training_seed=1,
        benchmark_root="/tmp/contextworld-bundle",
    )


def test_speed_structural_primary_score_matches_the_public_test_definition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _evaluate_structural(
        monkeypatch,
        task="speed",
        payload=_speed_dev_payload(),
        fake_read=_fake_speed_read,
        adapter=_functional_adapter(
            futures=5, rollout=_perfect_token_rollout(5)
        ),
    )

    metrics = result["metrics"]
    # Same field name and same per-track, per-horizon address as the Public
    # Test value path tracks.<track>.horizons.1.<primary>.
    assert (
        metrics["primary_metric"]
        == "reference_speed_balanced_strict_query_win_rate_vs_every_other"
    )
    assert metrics["reference_speed_balanced_strict_query_win_rate_vs_every_other"][
        _SPEED_TRACK
    ] == pytest.approx(1.0)
    track = metrics["tracks"][_SPEED_TRACK]
    assert (
        track["horizons"]["1"][
            "reference_speed_balanced_strict_query_win_rate_vs_every_other"
        ]
        == pytest.approx(1.0)
    )
    gate_inputs = track["gate_completion_inputs"]
    assert gate_inputs["metrics"]["correct_history_rate"] == pytest.approx(1.0)
    assert gate_inputs["metrics"]["context_switch_rate"] == pytest.approx(1.0)
    assert gate_inputs["metrics"]["latent_response"]["response_gain"] == (
        pytest.approx(1.0)
    )
    assert gate_inputs["metrics"]["latent_response"][
        "normalized_response_error"
    ] == pytest.approx(0.0, abs=1e-9)
    assert (
        gate_inputs["metrics"]["worst_reference_speed_correct_history_rate"]
        == pytest.approx(1.0)
    )
    assert gate_inputs["uncertainty"]["lower_bounds"][
        "correct_history_rate"
    ] == pytest.approx(1.0)
    legacy = metrics["legacy_history_utility_diagnostics"]
    assert legacy["history_better_rate"] == pytest.approx(1.0)
    assert result["record_count"] == 2700
    assert "gate" not in result
    _assert_no_decision_fields(result)


def test_speed_structural_constant_prediction_scores_zero_strict_wins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _evaluate_structural(
        monkeypatch,
        task="speed",
        payload=_speed_dev_payload(),
        fake_read=_fake_speed_read,
        adapter=_functional_adapter(futures=5, rollout=_constant_rollout(5)),
    )

    primary = result["metrics"][
        "reference_speed_balanced_strict_query_win_rate_vs_every_other"
    ]
    # A constant prediction ties every history condition; strict
    # "beats every other" comparisons do not credit ties.
    assert primary[_SPEED_TRACK] == 0.0
    assert "gate" not in result


def test_speed_structural_noise_prediction_scores_the_chance_level(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _evaluate_structural(
        monkeypatch,
        task="speed",
        payload=_speed_dev_payload(),
        fake_read=_fake_speed_read,
        adapter=_functional_adapter(futures=5, rollout=_noise_rollout(5)),
    )

    primary = result["metrics"][
        "reference_speed_balanced_strict_query_win_rate_vs_every_other"
    ]
    # One matching history out of three conditions is the chance level of the
    # strict win rate for this track.
    assert primary[_SPEED_TRACK] == pytest.approx(1 / 3, abs=0.1)


def test_door_structural_primary_score_matches_the_public_test_definition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _evaluate_structural(
        monkeypatch,
        task="door",
        payload=_door_dev_payload(),
        fake_read=_fake_door_read,
        adapter=_functional_adapter(
            futures=1, rollout=_perfect_token_rollout(1)
        ),
    )

    metrics = result["metrics"]
    assert metrics["primary_metric"] == "same_history_two_target_accuracy"
    assert metrics["same_history_two_target_accuracy"] == pytest.approx(1.0)
    for rule in ("passable", "blocked"):
        assert metrics["by_true_rule"][rule]["overall"][
            "same_history_two_target_accuracy"
        ] == pytest.approx(1.0)
        assert metrics["by_true_rule"][rule]["overall"]["strict_win_rate"] == (
            pytest.approx(1.0)
        )
    bootstrap = metrics["paired_static_query_bootstrap"]
    assert bootstrap["unique_static_queries"] == 300
    assert set(bootstrap["strata"].values()) == {25}
    assert all(row["lower"] > 0.0 for row in bootstrap["metrics"].values())
    gate_inputs = metrics["gate_completion_inputs"]
    assert gate_inputs["metrics"]["context_switch_rate"] == pytest.approx(1.0)
    assert gate_inputs["metrics"]["worst_rule_correct_target_choice_rate"] == (
        pytest.approx(1.0)
    )
    assert gate_inputs["metrics"]["latent_response"]["response_gain"] == (
        pytest.approx(1.0)
    )
    legacy = metrics["legacy_paired_diagnostics"]
    assert legacy["correct_future_rate"] == pytest.approx(1.0)
    assert legacy["correct_history_rate"] == pytest.approx(1.0)
    assert result["record_count"] == 1800
    assert "gate" not in result
    _assert_no_decision_fields(result)


def test_door_structural_constant_prediction_scores_the_chance_level(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _evaluate_structural(
        monkeypatch,
        task="door",
        payload=_door_dev_payload(),
        fake_read=_fake_door_read,
        adapter=_functional_adapter(futures=1, rollout=_constant_rollout(1)),
    )

    # The constant prediction picks the same rule for both rows of every
    # query, so exactly one of the two true-rule rows is credited: 0.5.
    assert result["metrics"]["same_history_two_target_accuracy"] == (
        pytest.approx(0.5)
    )
    assert "gate" not in result
