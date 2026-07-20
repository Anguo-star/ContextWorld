#!/usr/bin/env python3
"""Create evidence plots for the first ContextWorld TwoRoom benchmark step.

Every image in the output is derived from the original H5, accepted v2 Lance
tables, or a controlled replay of the pinned Stable-WorldModel TwoRoom env.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import h5py
import matplotlib
import numpy as np


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
STABLE_WORLD_MODEL = (ROOT.parent / "stable-worldmodel").resolve()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(STABLE_WORLD_MODEL))

from contextworld.paths import artifact_path, resolve_contextworld_path  # noqa: E402
import stable_worldmodel as swm  # noqa: E402
from stable_worldmodel.envs.two_room.env import TwoRoomEnv  # noqa: E402


ORIGINAL_H5 = (
    ROOT
    / "../../data/world_model/quentinll/lewm-tworooms/tworoom.h5"
).resolve()
MANIFEST = artifact_path(
    "synthesis/manifests/tworoom_speed_pixel_v2.jsonl", repo_root=ROOT
)
REPORT = artifact_path(
    "synthesis/reports/tworoom_speed_pixel_v2.json", repo_root=ROOT
)
OUTPUT = artifact_path(
    "synthesis/visualizations/tworoom_benchmark_step1", repo_root=ROOT
)


def _load_manifest() -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in MANIFEST.read_text().splitlines()
        if line.strip()
    ]


def _scenario_for_speed(
    manifest: list[dict[str, Any]], speed: float
) -> dict[str, Any]:
    return next(
        item
        for item in manifest
        if np.isclose(float(item["factors"]["agent.speed"]), speed)
    )


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _load_lance_episode(
    scenario: dict[str, Any], episode_index: int = 0
) -> dict[str, np.ndarray]:
    dataset = swm.data.LanceDataset(
        path=resolve_contextworld_path(
            scenario["output_path"], repo_root=ROOT
        ),
        keys_to_load=["pixels", "proprio", "action", "goal_state"],
    )
    episode = dataset.load_episode(episode_index)
    result = {key: _numpy(value) for key, value in episode.items()}
    result["pixels"] = np.transpose(result["pixels"], (0, 2, 3, 1))
    return result


def _speed_values(scenario: dict[str, Any]) -> dict[str, np.ndarray]:
    return {
        "agent.speed": np.asarray(
            scenario["variation_values"]["agent.speed"], dtype=np.float32
        )
    }


def _paired_reset(
    scenario: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    env = TwoRoomEnv(render_mode="rgb_array")
    try:
        _, info = env.reset(
            seed=int(scenario["env_seed"]),
            options={
                "variation": tuple(scenario["variation"]),
                "variation_values": _speed_values(scenario),
            },
        )
        return (
            np.asarray(info["proprio"]).copy(),
            np.asarray(info["goal_state"]).copy(),
            env.render().copy(),
        )
    finally:
        env.close()


def _save(fig: plt.Figure, name: str) -> Path:
    path = OUTPUT / name
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def original_data_evidence() -> dict[str, Any]:
    with h5py.File(ORIGINAL_H5, "r") as handle:
        count = min(20_000, len(handle["proprio"]))
        proprio = handle["proprio"][:count]
        action = handle["action"][:count]
        episode_id = handle["ep_idx"][:count]
        same_episode = episode_id[1:] == episode_id[:-1]
        delta = proprio[1:] - proprio[:-1]
        residual = np.linalg.norm(delta - 5.0 * action[:-1], axis=1)
        action_norm = np.linalg.norm(action[:-1], axis=1)
        projected_speed = np.sum(delta * action[:-1], axis=1) / np.maximum(
            np.sum(action[:-1] ** 2, axis=1), 1e-8
        )
        exact = same_episode & (residual <= 1e-4) & (action_norm > 1e-3)
        observation = handle["observation"][:count]
        return {
            "path": str(ORIGINAL_H5),
            "rows": int(len(handle["proprio"])),
            "episodes": int(len(handle["ep_len"])),
            "episode_length_min": int(handle["ep_len"][:].min()),
            "episode_length_mean": float(handle["ep_len"][:].mean()),
            "episode_length_max": int(handle["ep_len"][:].max()),
            "inferred_speed_median_on_exact_transitions": float(
                np.median(projected_speed[exact])
            ),
            "speed_5_exact_fraction": float(
                np.mean(residual[same_episode] <= 1e-4)
            ),
            "door_x": float(np.unique(observation[:, 4]).item()),
            "door_y": float(np.unique(observation[:, 5]).item()),
        }


def plot_original_vs_synthetic(
    manifest: list[dict[str, Any]], summary: dict[str, Any]
) -> Path:
    scenario = _scenario_for_speed(manifest, 5.0)
    synthetic = _load_lance_episode(scenario)

    with h5py.File(ORIGINAL_H5, "r") as handle:
        offset = int(handle["ep_offset"][0])
        length = int(handle["ep_len"][0])
        original_indices = np.linspace(0, length - 1, 4, dtype=int)
        original_frames = handle["pixels"][
            offset + original_indices
        ]
        original_states = handle["proprio"][
            offset + original_indices
        ]

    synthetic_indices = np.linspace(
        0, len(synthetic["pixels"]) - 1, 4, dtype=int
    )
    synthetic_frames = synthetic["pixels"][synthetic_indices]
    synthetic_states = synthetic["proprio"][synthetic_indices]

    fig, axes = plt.subplots(2, 4, figsize=(14.5, 7.2))
    for column in range(4):
        axes[0, column].imshow(original_frames[column])
        axes[0, column].set_title(
            f"H5 step {original_indices[column]}\n"
            f"state=({original_states[column, 0]:.1f}, "
            f"{original_states[column, 1]:.1f})",
            fontsize=9,
        )
        axes[1, column].imshow(synthetic_frames[column])
        axes[1, column].set_title(
            f"Lance step {synthetic_indices[column]}\n"
            f"state=({synthetic_states[column, 0]:.1f}, "
            f"{synthetic_states[column, 1]:.1f})",
            fontsize=9,
        )
        axes[0, column].axis("off")
        axes[1, column].axis("off")

    axes[0, 0].text(
        -0.08,
        0.5,
        "Original H5\ndefault speed=5\ndoor=49",
        transform=axes[0, 0].transAxes,
        ha="right",
        va="center",
        fontsize=11,
        fontweight="bold",
    )
    axes[1, 0].text(
        -0.08,
        0.5,
        "Synthetic v2\nspeed=5\ndoor=49",
        transform=axes[1, 0].transAxes,
        ha="right",
        va="center",
        fontsize=11,
        fontweight="bold",
    )
    fig.suptitle(
        "Actual dataset frames: original H5 vs accepted synthetic Lance",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.01,
        "Different reset seeds; this panel checks renderer/layout and storage "
        "compatibility, not pixel identity.",
        ha="center",
        fontsize=10,
    )
    summary["original_vs_synthetic"] = {
        "original_episode": 0,
        "original_steps": original_indices.tolist(),
        "synthetic_scenario_id": scenario["scenario_id"],
        "synthetic_steps": synthetic_indices.tolist(),
    }
    return _save(fig, "01_original_h5_vs_synthetic_speed5.png")


def plot_paired_speed_effect(
    manifest: list[dict[str, Any]], summary: dict[str, Any]
) -> Path:
    low_scenario = _scenario_for_speed(manifest, 1.75)
    high_scenario = _scenario_for_speed(manifest, 10.25)
    low = _load_lance_episode(low_scenario)
    high = _load_lance_episode(high_scenario)
    low_reset, low_goal, reset_frame = _paired_reset(low_scenario)
    high_reset, high_goal, _ = _paired_reset(high_scenario)
    reset_equal = bool(np.array_equal(low_reset, high_reset))
    goal_equal = bool(np.array_equal(low_goal, high_goal))
    if not (reset_equal and goal_equal):
        raise RuntimeError("Paired low/high scenarios do not share reset/goal")

    frame_steps = [0, 5, 10, 15]
    fig = plt.figure(figsize=(15.5, 10.5))
    grid = fig.add_gridspec(3, 4, height_ratios=[1, 1, 1.25])
    for row, (label, episode, color) in enumerate(
        (
            ("speed=1.75", low, "#1976d2"),
            ("speed=10.25", high, "#d32f2f"),
        )
    ):
        for column, step in enumerate(frame_steps):
            axis = fig.add_subplot(grid[row, column])
            axis.imshow(episode["pixels"][step])
            axis.set_title(
                f"{label} | stored step {step}\n"
                f"state=({episode['proprio'][step, 0]:.1f}, "
                f"{episode['proprio'][step, 1]:.1f})",
                color=color,
                fontsize=9,
            )
            axis.axis("off")

    low_path = np.vstack([low_reset, low["proprio"]])
    high_path = np.vstack([high_reset, high["proprio"]])
    spatial = fig.add_subplot(grid[2, :2])
    spatial.imshow(reset_frame)
    spatial.plot(
        low_path[:, 0], low_path[:, 1], "-o", color="#1976d2",
        markersize=2.5, linewidth=1.6, label="speed=1.75",
    )
    spatial.plot(
        high_path[:, 0], high_path[:, 1], "-o", color="#d32f2f",
        markersize=2.5, linewidth=1.6, label="speed=10.25",
    )
    spatial.scatter(
        [low_reset[0]], [low_reset[1]], marker="s", s=80,
        color="gold", edgecolor="black", label="shared reset",
    )
    spatial.scatter(
        [low_goal[0]], [low_goal[1]], marker="*", s=170,
        color="#00a650", edgecolor="black", label="shared goal",
    )
    spatial.set_title("Paired trajectories on the same room/reset/goal")
    spatial.legend(loc="lower left", fontsize=8)
    spatial.set_xlim(0, 223)
    spatial.set_ylim(223, 0)

    distance_axis = fig.add_subplot(grid[2, 2:])
    low_distance = np.linalg.norm(low_path - low_goal[None, :], axis=1)
    high_distance = np.linalg.norm(high_path - high_goal[None, :], axis=1)
    distance_axis.plot(
        low_distance, color="#1976d2", linewidth=2, label="speed=1.75"
    )
    distance_axis.plot(
        high_distance, color="#d32f2f", linewidth=2, label="speed=10.25"
    )
    distance_axis.axhline(
        16.0, color="black", linestyle="--", linewidth=1,
        label="success radius",
    )
    distance_axis.set_xlabel("Environment step (0 = shared reset)")
    distance_axis.set_ylabel("Distance to goal (pixels)")
    distance_axis.set_title("High speed changes frame-to-frame progress")
    distance_axis.grid(alpha=0.25)
    distance_axis.legend(fontsize=8)

    fig.suptitle(
        "Accepted Lance data: one-factor paired comparison (only speed differs)",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.015,
        f"shared reset=({low_reset[0]:.2f}, {low_reset[1]:.2f}); "
        f"low episode={len(low['proprio'])} rows, "
        f"high episode={len(high['proprio'])} rows",
        ha="center",
        fontsize=10,
    )
    summary["paired_speed_effect"] = {
        "low_scenario_id": low_scenario["scenario_id"],
        "high_scenario_id": high_scenario["scenario_id"],
        "shared_env_seed": int(low_scenario["env_seed"]),
        "reset_equal": reset_equal,
        "goal_equal": goal_equal,
        "reset": low_reset.tolist(),
        "goal": low_goal.tolist(),
        "low_episode_rows": int(len(low["proprio"])),
        "high_episode_rows": int(len(high["proprio"])),
    }
    return _save(fig, "02_paired_speed_1p75_vs_10p25.png")


def plot_speed_frame_skip_oracle(summary: dict[str, Any]) -> Path:
    state = np.asarray([40.0, 40.0], dtype=np.float32)
    target = np.asarray([190.0, 190.0], dtype=np.float32)
    action = np.asarray([0.5, 0.25], dtype=np.float32)

    def options(speed: float) -> dict[str, Any]:
        return {
            "variation": ("agent.speed",),
            "variation_values": {
                "agent.speed": np.asarray([speed], dtype=np.float32)
            },
            "state": state.copy(),
            "target_state": target.copy(),
        }

    slow = TwoRoomEnv(render_mode="rgb_array")
    fast = TwoRoomEnv(render_mode="rgb_array")
    try:
        slow.reset(seed=314159, options=options(3.0))
        fast.reset(seed=314159, options=options(6.0))
        initial = slow.render().copy()
        initial_fast = fast.render().copy()
        slow.step(action)
        slow_middle = slow.render().copy()
        slow_observation, *_ = slow.step(action)
        slow_final = slow.render().copy()
        fast_observation, *_ = fast.step(action)
        fast_final = fast.render().copy()
    finally:
        slow.close()
        fast.close()

    absolute_difference = np.max(
        np.abs(slow_final.astype(np.int16) - fast_final.astype(np.int16)),
        axis=-1,
    )
    frames = [initial, slow_middle, slow_final, fast_final]
    titles = [
        "Shared initial frame",
        "speed=3 after 1 step\n(intermediate frame)",
        "speed=3 after 2 steps",
        "speed=6 after 1 step",
    ]
    fig, axes = plt.subplots(1, 5, figsize=(17, 3.8))
    for axis, frame, title in zip(axes[:4], frames, titles):
        axis.imshow(frame)
        axis.set_title(title, fontsize=10)
        axis.axis("off")
    axes[4].imshow(absolute_difference, cmap="magma", vmin=0, vmax=255)
    axes[4].set_title(
        "|slow step 2 - fast step 1|\nmax pixel difference = 0",
        fontsize=10,
    )
    axes[4].axis("off")
    fig.suptitle(
        "Controlled speed oracle: doubling speed equals dropping the slow middle frame",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.01,
        "Same reset, same open-loop action [0.5, 0.25], collision-free. "
        "This equivalence is not assumed for changing closed-loop actions.",
        ha="center",
        fontsize=9,
    )
    summary["speed_frame_skip_oracle"] = {
        "initial_pixels_equal": bool(np.array_equal(initial, initial_fast)),
        "slow_final_state": np.asarray(slow_observation[:2]).tolist(),
        "fast_final_state": np.asarray(fast_observation[:2]).tolist(),
        "final_states_equal": bool(
            np.array_equal(
                np.asarray(slow_observation[:2]),
                np.asarray(fast_observation[:2]),
            )
        ),
        "final_pixels_equal": bool(np.array_equal(slow_final, fast_final)),
        "maximum_pixel_difference": int(absolute_difference.max()),
        "middle_frame_differs": bool(
            not np.array_equal(slow_middle, fast_final)
        ),
    }
    return _save(fig, "03_speed_frame_skip_oracle.png")


def plot_door_position_oracle(summary: dict[str, Any]) -> Path:
    baseline_position = 49
    changed_position = 154
    door_half_size = 14
    wall_thickness = 10
    state = np.asarray([40.0, 40.0], dtype=np.float32)
    target = np.asarray([190.0, 190.0], dtype=np.float32)

    def options(position: int) -> dict[str, Any]:
        return {
            "variation": (
                "door.position",
                "door.size",
                "door.number",
                "wall.axis",
                "wall.thickness",
            ),
            "variation_values": {
                "door.position": np.asarray([position] * 3, dtype=np.int64),
                "door.size": np.asarray(
                    [door_half_size] * 3, dtype=np.int64
                ),
                "door.number": 1,
                "wall.axis": 1,
                "wall.thickness": wall_thickness,
            },
            "state": state.copy(),
            "target_state": target.copy(),
        }

    baseline_env = TwoRoomEnv(render_mode="rgb_array")
    changed_env = TwoRoomEnv(render_mode="rgb_array")
    try:
        baseline_observation, baseline_info = baseline_env.reset(
            seed=271828, options=options(baseline_position)
        )
        changed_observation, changed_info = changed_env.reset(
            seed=271828, options=options(changed_position)
        )
        baseline = baseline_env.render().copy()
        changed = changed_env.render().copy()
    finally:
        baseline_env.close()
        changed_env.close()

    actual_mask = np.any(baseline != changed, axis=-1)
    height, width = actual_mask.shape
    grid_y, grid_x = np.mgrid[:height, :width]
    half = wall_thickness // 2
    wall = (
        (grid_x >= TwoRoomEnv.WALL_CENTER - half)
        & (grid_x <= TwoRoomEnv.WALL_CENTER + half)
    )
    baseline_span = (
        (grid_y >= baseline_position - door_half_size)
        & (grid_y <= baseline_position + door_half_size)
    )
    changed_span = (
        (grid_y >= changed_position - door_half_size)
        & (grid_y <= changed_position + door_half_size)
    )
    expected_mask = wall & (baseline_span ^ changed_span)
    mismatch = actual_mask ^ expected_mask
    overlay = np.zeros((height, width, 3), dtype=np.uint8)
    overlay[actual_mask & expected_mask] = [0, 210, 80]
    overlay[actual_mask & ~expected_mask] = [255, 0, 0]
    overlay[~actual_mask & expected_mask] = [0, 100, 255]
    absolute_difference = np.abs(
        baseline.astype(np.int16) - changed.astype(np.int16)
    ).astype(np.uint8)

    fig, axes = plt.subplots(1, 4, figsize=(14.5, 3.9))
    axes[0].imshow(baseline)
    axes[0].set_title("Original default\ndoor.position=49")
    axes[1].imshow(changed)
    axes[1].set_title("Changed parameter\ndoor.position=154")
    axes[2].imshow(absolute_difference)
    axes[2].set_title(
        f"Absolute RGB difference\nchanged pixels={int(actual_mask.sum())}"
    )
    axes[3].imshow(overlay)
    axes[3].set_title(
        "Expected vs actual mask\ngreen=exact match, red/blue=error"
    )
    for axis in axes:
        axis.axis("off")
    fig.suptitle(
        "Controlled geometry oracle: moving the door changes only its old/new opening",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.01,
        "Same state, goal, wall, door size, colors, and seed; only door.position changes.",
        ha="center",
        fontsize=9,
    )
    unchanged_exact = bool(
        np.array_equal(baseline[~expected_mask], changed[~expected_mask])
    )
    summary["door_position_oracle"] = {
        "baseline_position": baseline_position,
        "changed_position": changed_position,
        "actual_changed_pixels": int(actual_mask.sum()),
        "expected_changed_pixels": int(expected_mask.sum()),
        "change_mask_exact": bool(np.array_equal(actual_mask, expected_mask)),
        "mask_mismatch_pixels": int(mismatch.sum()),
        "unchanged_pixels_exact": unchanged_exact,
        "state_unchanged": bool(
            np.array_equal(baseline_info["proprio"], changed_info["proprio"])
        ),
        "observation_readback": bool(
            baseline_observation[5] == baseline_position
            and changed_observation[5] == changed_position
        ),
    }
    return _save(fig, "04_door_position_pixel_oracle.png")


def plot_benchmark_split(
    manifest: list[dict[str, Any]], summary: dict[str, Any]
) -> Path:
    lanes = {
        "train": 4,
        "validation_interp": 3,
        "test_interp": 2,
        "test_extrap_low": 1,
        "test_extrap_high": 0,
    }
    colors = {
        "train": "#1976d2",
        "validation_interp": "#6a1b9a",
        "test_interp": "#ef6c00",
        "test_extrap_low": "#00897b",
        "test_extrap_high": "#c62828",
    }
    values_by_regime: dict[str, list[float]] = defaultdict(list)
    meta_by_id = {}
    for item in manifest:
        regime = item.get("regime") or item["split"]
        values_by_regime[regime].append(
            float(item["factors"]["agent.speed"])
        )
        meta_by_id[item["scenario_id"]] = item

    report = json.loads(REPORT.read_text())
    rows_by_regime: dict[str, int] = defaultdict(int)
    for scenario_report in report["scenarios"]:
        regime = meta_by_id[scenario_report["scenario_id"]].get("regime")
        rows_by_regime[regime] += int(scenario_report["rows"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    for regime, y in lanes.items():
        values = sorted(values_by_regime[regime])
        axes[0].scatter(
            values,
            [y] * len(values),
            s=55,
            color=colors[regime],
            edgecolor="white",
            linewidth=0.6,
            label=f"{regime} ({len(values)})",
        )
    axes[0].set_yticks(list(lanes.values()), list(lanes.keys()))
    axes[0].set_xlabel("agent.speed")
    axes[0].set_title("Scenario values: train / validation / test isolation")
    axes[0].grid(axis="x", alpha=0.25)
    axes[0].legend(loc="lower right", fontsize=8)

    regimes = list(lanes)
    rows = [rows_by_regime[name] for name in regimes]
    bars = axes[1].barh(
        regimes, rows, color=[colors[name] for name in regimes]
    )
    axes[1].bar_label(bars, labels=[f"{value:,}" for value in rows], padding=4)
    axes[1].set_xlabel("Accepted rows")
    axes[1].set_title("Actual accepted data volume by regime")
    axes[1].grid(axis="x", alpha=0.25)

    fig.suptitle(
        "Benchmark step 1 dataset: simple speed change with separated evaluation regimes",
        fontsize=14,
        fontweight="bold",
    )
    summary["benchmark_split"] = {
        "speed_values_by_regime": {
            key: sorted(value) for key, value in values_by_regime.items()
        },
        "rows_by_regime": dict(rows_by_regime),
        "minimum_cross_split_gap": report["numeric_atom_isolation"][
            "minimum_observed_cross_split_gap"
        ],
    }
    return _save(fig, "05_benchmark_speed_split_and_rows.png")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest()
    summary: dict[str, Any] = {
        "original_data": original_data_evidence(),
        "accepted_v2_report": str(REPORT),
        "stable_worldmodel_commit": "5864b74980f6ed328fd0045e777b3865962eff43",
    }
    outputs = [
        plot_original_vs_synthetic(manifest, summary),
        plot_paired_speed_effect(manifest, summary),
        plot_speed_frame_skip_oracle(summary),
        plot_door_position_oracle(summary),
        plot_benchmark_split(manifest, summary),
    ]
    summary_path = OUTPUT / "visualization_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({
        "output_directory": str(OUTPUT),
        "figures": [str(path) for path in outputs],
        "summary": str(summary_path),
    }, indent=2))


if __name__ == "__main__":
    main()
