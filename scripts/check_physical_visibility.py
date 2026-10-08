#!/usr/bin/env python3
"""Audit PushT future-state visibility in the fixed three-task panel release.

The audit is model-free: it reads future_states for every Development scene and
reads future_pixels only until one exact-pixel counterexample is found per task
(or 16 off-canvas scenes have been checked).  The input panel root and output
path are explicit CLI arguments so the fixed source can be replayed elsewhere.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

TASKS = {
    "action_strength": {
        "state_dim": 7,
        "pusher_slice": (0, 2),
        "block_slice": (2, 4),
        "mapping": "state[:2] is pusher center xy; state[2:4] is block center xy",
    },
    "contact_friction": {
        "state_dim": 12,
        "pusher_slice": (0, 2),
        "block_slice": (6, 8),
        "mapping": "state[:2] is pusher center xy; state[6:8] is block center xy",
    },
    "motion_damping": {
        "state_dim": 12,
        "pusher_slice": (0, 2),
        "block_slice": (6, 8),
        "mapping": "state[:2] is pusher center xy; state[6:8] is block center xy",
    },
}
HORIZONS = (5, 10, 15, 20, 25)
CANVAS_LOW = 0.0
CANVAS_HIGH = 512.0
EXPECTED_SCENES = 256
MAX_IMAGE_SCENES = 16
MIN_COORDINATE_DELTA_PX = 1.0
EXPECTED_PIXEL_SHAPE = (224, 224, 3)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_npz(task_root: Path, entry: dict[str, Any]) -> Path:
    path = Path(entry["path"])
    return path if path.is_absolute() else task_root / path


def out_of_canvas(xy: np.ndarray) -> np.ndarray:
    return np.any((xy < CANVAS_LOW) | (xy > CANVAS_HIGH), axis=-1)


def ratio(count: int, rows: int) -> float:
    return float(count / rows) if rows else 0.0


def aggregate(counts: list[int], rows_per_horizon: int) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    for horizon, count in zip(HORIZONS, counts):
        result[str(horizon)] = {
            "rows": rows_per_horizon,
            "out_count": int(count),
            "out_fraction": ratio(int(count), rows_per_horizon),
            "out_percent": 100.0 * ratio(int(count), rows_per_horizon),
        }
    all_count = int(sum(counts))
    rows_all = rows_per_horizon * len(HORIZONS)
    result["all_horizons"] = {
        "rows": rows_all,
        "out_count": all_count,
        "out_fraction": ratio(all_count, rows_all),
        "out_percent": 100.0 * ratio(all_count, rows_all),
    }
    return result


def canvas_sources(panels_root: Path) -> list[dict[str, Any]]:
    """Locate the pinned PushT implementation without scanning unrelated paths."""
    candidates = (
        panels_root
        / "contact_friction/.swm_6ab823fdc692/stable_worldmodel/envs/pusht/env.py",
        panels_root
        / "motion_damping/.swm_875e607fc08a/stable_worldmodel/envs/pusht/env.py",
        panels_root.parent
        / "pusht/scratch/cf/.swm_6ab823fdc692/stable_worldmodel/envs/pusht/env.py",
        panels_root.parent
        / "pusht/scratch/md/.swm_875e607fc08a/stable_worldmodel/envs/pusht/env.py",
    )
    sources = []
    for path in candidates:
        if not path.is_file():
            continue
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if "self.window_size = ws = 512" in line:
                sources.append(
                    {
                        "path": str(path),
                        "line": line_number,
                        "evidence": line.strip(),
                    }
                )
                break
    return sources


def coordinate_counterexample(
    pixels: np.ndarray,
    coordinates: dict[str, np.ndarray],
    out_masks: dict[str, np.ndarray],
) -> dict[str, Any] | None:
    """Find exact-pixel matches with >=1 px differing off-canvas coordinates."""
    # The bytes are retained in addition to SHA-256, so equality is genuinely
    # bitwise and does not depend on a hash collision assumption.
    seen: dict[str, list[dict[str, Any]]] = {}
    for k in range(pixels.shape[0]):
        for c in range(pixels.shape[1]):
            for t in range(pixels.shape[2]):
                index = (k, c, t)
                frame = np.ascontiguousarray(pixels[k, c, t])
                frame_bytes = frame.tobytes(order="C")
                digest = hashlib.sha256(frame_bytes).hexdigest()
                for previous in seen.get(digest, []):
                    if previous["bytes"] != frame_bytes:
                        continue
                    prior = previous["index"]
                    # Pusher is checked first, prioritizing the relevant
                    # off-canvas pusher example when one exists.
                    for body in ("pusher", "block"):
                        a = np.asarray(coordinates[body][prior], dtype=np.float64)
                        b = np.asarray(coordinates[body][index], dtype=np.float64)
                        delta = b - a
                        distance = float(np.linalg.norm(delta))
                        if distance < MIN_COORDINATE_DELTA_PX:
                            continue
                        a_out = bool(out_masks[body][prior])
                        b_out = bool(out_masks[body][index])
                        if not (a_out or b_out):
                            continue
                        return {
                            "physical_body": body,
                            "group_a": {"k": int(prior[0]), "c": int(prior[1]), "t": int(prior[2])},
                            "group_b": {"k": int(index[0]), "c": int(index[1]), "t": int(index[2])},
                            "coordinate_a": [float(x) for x in a],
                            "coordinate_b": [float(x) for x in b],
                            "coordinate_delta_b_minus_a": [float(x) for x in delta],
                            "coordinate_delta_norm_px": distance,
                            "minimum_coordinate_delta_px": MIN_COORDINATE_DELTA_PX,
                            "group_a_out_of_canvas": a_out,
                            "group_b_out_of_canvas": b_out,
                            "pixel_shape": [int(x) for x in frame.shape],
                            "pixel_dtype": str(frame.dtype),
                            "pixel_sha256": digest,
                            "pixel_equal_bitwise": True,
                        }
                seen.setdefault(digest, []).append({"index": index, "bytes": frame_bytes})
    return None


def audit_task(task: str, panels_root: Path) -> dict[str, Any]:
    config = TASKS[task]
    task_root = panels_root / task
    manifest_path = task_root / "manifest.json"
    manifest_digest = sha256_file(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest["scenes"]
    if len(entries) != EXPECTED_SCENES:
        raise ValueError(f"{task}: expected {EXPECTED_SCENES} scenes, got {len(entries)}")

    scene_info: list[dict[str, Any]] = []
    pusher_counts: list[np.ndarray] = []
    block_counts: list[np.ndarray] = []
    any_counts: list[np.ndarray] = []
    shape: tuple[int, int, int, int] | None = None

    for manifest_index, entry in enumerate(entries):
        npz_path = resolve_npz(task_root, entry)
        with np.load(npz_path, allow_pickle=False) as data:
            states = np.asarray(data["future_states"])
        if states.ndim != 4 or states.shape[-1] != config["state_dim"]:
            raise ValueError(f"{task} {npz_path}: unexpected future_states shape {states.shape}")
        if shape is None:
            shape = tuple(int(x) for x in states.shape)
        elif tuple(states.shape) != shape:
            raise ValueError(f"{task}: inconsistent future_states shape {states.shape}; expected {shape}")
        if shape[2] != len(HORIZONS):
            raise ValueError(f"{task}: expected {len(HORIZONS)} horizons, got {shape[2]}")

        pusher = np.asarray(states[..., slice(*config["pusher_slice"])], dtype=np.float64)
        block = np.asarray(states[..., slice(*config["block_slice"])], dtype=np.float64)
        pusher_out = out_of_canvas(pusher)
        block_out = out_of_canvas(block)
        any_out = pusher_out | block_out
        pusher_counts.append(pusher_out.sum(axis=(0, 1)).astype(np.int64))
        block_counts.append(block_out.sum(axis=(0, 1)).astype(np.int64))
        any_counts.append(any_out.sum(axis=(0, 1)).astype(np.int64))
        scene_info.append(
            {
                "manifest_index": manifest_index,
                "scene_id": str(entry.get("scene_id", manifest_index)),
                "path": str(npz_path),
                "pusher": pusher,
                "block": block,
                "pusher_out": pusher_out,
                "block_out": block_out,
                "any_out": any_out,
                "has_any_out": bool(any_out.any()),
            }
        )

    assert shape is not None
    conditions, candidates, horizons, _ = shape
    rows_per_horizon = EXPECTED_SCENES * conditions * candidates
    pusher_by_t = np.sum(np.stack(pusher_counts), axis=0).astype(int).tolist()
    block_by_t = np.sum(np.stack(block_counts), axis=0).astype(int).tolist()
    any_by_t = np.sum(np.stack(any_counts), axis=0).astype(int).tolist()

    candidate_scenes = [item for item in scene_info if item["has_any_out"]]
    checked_ids: list[str] = []
    checked_indices: list[int] = []
    counterexample: dict[str, Any] | None = None
    for info in candidate_scenes[:MAX_IMAGE_SCENES]:
        checked_ids.append(info["scene_id"])
        checked_indices.append(int(info["manifest_index"]))
        with np.load(info["path"], allow_pickle=False) as data:
            pixels = np.asarray(data["future_pixels"])
        expected_shape = (conditions, candidates, horizons) + EXPECTED_PIXEL_SHAPE
        if tuple(pixels.shape) != expected_shape:
            raise ValueError(f"{task} {info['path']}: unexpected future_pixels shape {pixels.shape}")
        counterexample = coordinate_counterexample(
            pixels,
            {"pusher": info["pusher"], "block": info["block"]},
            {"pusher": info["pusher_out"], "block": info["block_out"]},
        )
        if counterexample is not None:
            counterexample.update(
                {
                    "scene_id": info["scene_id"],
                    "manifest_index": int(info["manifest_index"]),
                    "npz_path": info["path"],
                }
            )
            break

    checked = len(checked_ids)
    return {
        "scene_count": len(entries),
        "manifest": {"path": str(manifest_path), "sha256": manifest_digest},
        "condition_count": int(conditions),
        "candidate_count": int(candidates),
        "horizons": list(HORIZONS),
        "state_dim": int(shape[3]),
        "future_state_rows_all_horizons": int(rows_per_horizon * horizons),
        "future_pixel_shape": list(EXPECTED_PIXEL_SHAPE),
        "physical_mapping": config["mapping"],
        "pusher_out_of_canvas": aggregate(pusher_by_t, rows_per_horizon),
        "block_out_of_canvas": aggregate(block_by_t, rows_per_horizon),
        "any_center_out_of_canvas": aggregate(any_by_t, rows_per_horizon),
        "out_of_canvas_scene_count": len(candidate_scenes),
        "out_of_canvas_scene_ids_manifest_order": [item["scene_id"] for item in candidate_scenes],
        "future_pixel_counterexample": {
            "found": counterexample is not None,
            "offscreen_scene_candidates_available": len(candidate_scenes),
            "offscreen_scenes_checked_for_pixels": checked,
            "checked_scene_ids_manifest_order": checked_ids,
            "checked_manifest_indices": checked_indices,
            "max_offscreen_scenes_checked": MAX_IMAGE_SCENES,
            "minimum_coordinate_delta_px": MIN_COORDINATE_DELTA_PX,
            "stopped_after_first_counterexample": counterexample is not None,
            "example": counterexample,
        },
        "all_future_state_rows_retained": True,
        "all_horizons_retained": True,
    }


def build_report(panels_root: Path) -> dict[str, Any]:
    tasks = {task: audit_task(task, panels_root) for task in TASKS}
    return {
        "schema_version": 2,
        "scope": {
            "panels_root": str(panels_root.resolve()),
            "fixed_tasks": list(TASKS),
            "scene_count_per_task": EXPECTED_SCENES,
            "evaluation_split": "development",
            "future_source": "future_states and future_pixels in the fixed Development NPZ panels",
            "horizons": list(HORIZONS),
            "canvas_range_xy_inclusive": [CANVAS_LOW, CANVAS_HIGH],
            "canvas_implementation": {
                "window_size": 512,
                "source_files": canvas_sources(panels_root),
                "evidence": "PushT env.py sets self.window_size = ws = 512; center coordinates are checked against the original 512x512 canvas.",
            },
            "queries_retained": True,
            "model_free": True,
            "gpu_used": False,
            "training_or_generation": False,
            "image_read_limit": "At most 16 manifest-order scenes containing an off-canvas future center per task; stop after the first exact-pixel counterexample.",
            "counterexample_requirement": "same-scene uint8 future_pixels are bitwise equal and selected physical coordinate distance is at least 1 px, with at least one coordinate off canvas; pusher is checked before block",
        },
        "tasks": tasks,
        "checks": {
            "all_tasks_have_256_scenes": all(item["scene_count"] == EXPECTED_SCENES for item in tasks.values()),
            "all_rows_retained": all(item["all_future_state_rows_retained"] for item in tasks.values()),
            "all_horizons_complete": all(item["all_horizons_retained"] for item in tasks.values()),
            "counterexample_found_each_task": all(item["future_pixel_counterexample"]["found"] for item in tasks.values()),
        },
        "interpretation": [
            "The 0..512 range check is a necessary visibility condition only: in-range coordinates do not prove pixel identifiability, and off-canvas coordinates do not prove that pixels contain no state evidence.",
            "A rendered latent frame need not encode an off-canvas center. Low physical readout error on in-manifold rows therefore cannot establish generalizable off-manifold physical readout.",
            "No query, candidate, condition, or horizon rows were filtered; this is a model-free visibility audit of the fixed Development panels.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panels-root", required=True, type=Path, help="Fixed panel root containing the three task directories")
    parser.add_argument("--output", required=True, type=Path, help="JSON report path")
    args = parser.parse_args()
    if not args.panels_root.is_dir():
        parser.error(f"panels root does not exist: {args.panels_root}")
    report = build_report(args.panels_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
