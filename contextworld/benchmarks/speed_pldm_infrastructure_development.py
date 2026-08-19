"""Shared contracts for the pre-Public Speed PLDM infrastructure gate.

This module deliberately contains no evaluator, environment, or Public-suite
loader.  It only defines portable identities and the narrow data geometry used
by the Development-only readiness receipt.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from contextworld.paths import portable_contextworld_path, repository_root, resolve_contextworld_path


DEVELOPMENT_ID = "tworoom_speed_pldm_infrastructure_development_v1"
COMPLETION_ID = "tworoom_speed_pldm_reference_completion_v1"
EXPECTED_SEEDS = (3072, 4096, 5120)
EXPECTED_HISTORY_TOKENS = 3
EXPECTED_ACTION_BLOCK_RAW_STEPS = 5
EXPECTED_ACTION_DIM = 2
EXPECTED_FUTURE_ACTION_BLOCKS = 1
EXPECTED_OBSERVATION_STEPS = 4
SAMPLES_PER_SCENARIO = 4
CLIP_INDEX_RULE = "zero_floor_n_minus_one_over_three_floor_two_n_minus_one_over_three_n_minus_one"

# This is intentionally a no-score readiness contract.  Keep these fields
# literal so later stages can reject a receipt that silently grows a capability
# or selection claim.
DEVELOPMENT_SCOPE = {
    "development_scope": "infrastructure_readiness",
    "icl_claim": False,
    "checkpoint_selection": False,
    "public_payload_accessed": False,
    "scoreboard_score_emitted": False,
    "training_or_recipe_mutation": False,
}


def root() -> Path:
    return repository_root().resolve()


def resolve_source(value: str | Path, *, repo_root: Path | None = None) -> Path:
    """Resolve an input path using ContextWorld's artifact-root convention."""

    return resolve_contextworld_path(value, repo_root=(repo_root or root())).resolve()


def resolve_local_output(value: str | Path, *, repo_root: Path | None = None) -> Path:
    """Resolve a new output strictly beneath the checkout.

    Generated freeze receipts must never be redirected to the artifact mount by
    an environment setting or a pre-existing symlink.
    """

    base = (repo_root or root()).resolve()
    raw = Path(value).expanduser()
    candidate = raw.resolve() if raw.is_absolute() else (base / raw).resolve()
    try:
        candidate.relative_to(base)
    except ValueError as error:
        raise ValueError(f"Output must remain inside the repository: {value}") from error
    return candidate


def logical_path(path: Path, *, repo_root: Path | None = None) -> str:
    """Return a portable ContextWorld path and reject an unexpected escape."""

    base = (repo_root or root()).resolve()
    resolved = Path(path).resolve()
    try:
        return portable_contextworld_path(resolved, repo_root=base)
    except ValueError as error:  # defensive: current helper does not raise
        raise ValueError(f"Cannot serialize path: {path}") from error


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(path: Path, *, repo_root: Path | None = None) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": logical_path(path, repo_root=repo_root),
        "sha256": sha256_file(path),
        "size_bytes": int(path.stat().st_size),
    }


def same_identity(value: Any, expected: Mapping[str, Any]) -> bool:
    """Compare the required immutable identity fields, accepting extra data."""

    return bool(
        isinstance(value, Mapping)
        and value.get("path") == expected.get("path")
        and value.get("sha256") == expected.get("sha256")
        and (
            "size_bytes" not in expected
            or value.get("size_bytes") == expected.get("size_bytes")
        )
    )


def require_identity(
    specification: Mapping[str, Any],
    *,
    label: str,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Verify a path/SHA specification and return its observed identity."""

    if not isinstance(specification, Mapping):
        raise ValueError(f"{label} needs a path/SHA mapping")
    raw_path = specification.get("path")
    expected_sha256 = specification.get("sha256")
    if not isinstance(raw_path, str) or not raw_path or not isinstance(expected_sha256, str):
        raise ValueError(f"{label} needs non-empty path and sha256")
    observed = identity(resolve_source(raw_path, repo_root=repo_root), repo_root=repo_root)
    if observed["sha256"] != expected_sha256:
        raise RuntimeError(
            f"{label} identity drifted: expected={expected_sha256}, "
            f"observed={observed['sha256']}"
        )
    if "size_bytes" in specification and int(specification["size_bytes"]) != observed["size_bytes"]:
        raise RuntimeError(f"{label} size drifted")
    return observed


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def canonical_array_identity(value: Any) -> dict[str, Any]:
    """Hash an array with dtype and shape, preventing byte-layout ambiguity."""

    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(str(tuple(int(size) for size in array.shape)).encode("utf-8"))
    digest.update(array.tobytes())
    return {
        "dtype": str(array.dtype),
        "shape": [int(size) for size in array.shape],
        "sha256": digest.hexdigest(),
    }


def canonical_array_matches(value: Any, expected: Mapping[str, Any]) -> bool:
    if not isinstance(expected, Mapping):
        return False
    return canonical_array_identity(value) == {
        "dtype": expected.get("dtype"),
        "shape": expected.get("shape"),
        "sha256": expected.get("sha256"),
    }


def raw_sample_arrays(sample: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Normalize one raw H3 PLDM clip to public-API array geometry.

    A StableWM Lance clip has four RGB observations in CHW order and four
    flattened five-step action blocks.  The readiness check uses observations
    0..2 and action blocks 0..2 to predict observation 3.
    """

    if not isinstance(sample, Mapping) or set(sample) != {"pixels", "action"}:
        raise ValueError("Development loader sample must contain exactly pixels/action")
    pixels = sample["pixels"]
    actions = sample["action"]
    if hasattr(pixels, "detach"):
        pixels = pixels.detach().cpu().numpy()
    if hasattr(actions, "detach"):
        actions = actions.detach().cpu().numpy()
    pixels = np.asarray(pixels)
    actions = np.asarray(actions)
    if pixels.shape[:2] == (EXPECTED_OBSERVATION_STEPS, 3):
        pixels = np.transpose(pixels, (0, 2, 3, 1))
    if (
        pixels.ndim != 4
        or pixels.shape[0] != EXPECTED_OBSERVATION_STEPS
        or pixels.shape[-1] != 3
        or pixels.dtype != np.uint8
    ):
        raise ValueError(
            "Expected raw [4,H,W,3] uint8 pixels (or CHW source), got "
            f"shape={pixels.shape}, dtype={pixels.dtype}"
        )
    expected_action_width = EXPECTED_ACTION_BLOCK_RAW_STEPS * EXPECTED_ACTION_DIM
    if actions.shape != (EXPECTED_OBSERVATION_STEPS, expected_action_width):
        raise ValueError(
            "Expected raw [4,10] action blocks, got "
            f"shape={actions.shape}, dtype={actions.dtype}"
        )
    actions = np.asarray(actions, dtype=np.float32)
    if not np.isfinite(actions).all():
        raise ValueError("Development action blocks contain non-finite values")
    return np.ascontiguousarray(pixels), np.ascontiguousarray(actions)


def deterministic_clip_indices(dataset_length: int) -> tuple[int, int, int, int]:
    """Choose four evenly spaced clip indices from length alone.

    This deliberately cannot inspect pixels, actions, labels, model outputs,
    losses, or episode identities.  The actual resulting indices and their
    structural episode coverage are frozen into the manifest afterward.
    """

    length = int(dataset_length)
    if length < SAMPLES_PER_SCENARIO:
        raise ValueError(
            "Development scenario has too few clips for the preregistered "
            f"four-sample rule: {length}"
        )
    last = length - 1
    values = (0, last // 3, (2 * last) // 3, last)
    if len(set(values)) != SAMPLES_PER_SCENARIO:
        raise RuntimeError(
            "Length-only Development clip rule did not produce four unique "
            f"indices: length={length}, indices={values}"
        )
    return values


def make_record_arrays(sample: Mapping[str, Any]) -> dict[str, Any]:
    """Produce the frozen hash fields for one deterministic raw clip."""

    pixels, actions = raw_sample_arrays(sample)
    action_blocks = actions.reshape(
        EXPECTED_OBSERVATION_STEPS,
        EXPECTED_ACTION_BLOCK_RAW_STEPS,
        EXPECTED_ACTION_DIM,
    )
    history = pixels[:EXPECTED_HISTORY_TOKENS]
    target = pixels[EXPECTED_HISTORY_TOKENS]
    prefix_actions = action_blocks[: EXPECTED_HISTORY_TOKENS]
    return {
        "pixels": canonical_array_identity(pixels),
        "actions": canonical_array_identity(actions),
        "history_pixels": canonical_array_identity(history),
        "target_pixels": canonical_array_identity(target),
        "prefix_action_blocks": canonical_array_identity(prefix_actions),
        "geometry": {
            "history_pixel_indices": [0, 1, 2],
            "target_pixel_index": 3,
            "action_block_indices": [0, 1, 2],
            "history_tokens": EXPECTED_HISTORY_TOKENS,
            "future_action_blocks": EXPECTED_FUTURE_ACTION_BLOCKS,
            "raw_steps_per_action_block": EXPECTED_ACTION_BLOCK_RAW_STEPS,
            "action_dim": EXPECTED_ACTION_DIM,
        },
    }


def verify_record_arrays(sample: Mapping[str, Any], record: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct a manifest record and return its rollout-ready arrays."""

    expected = make_record_arrays(sample)
    for name in (
        "pixels",
        "actions",
        "history_pixels",
        "target_pixels",
        "prefix_action_blocks",
        "geometry",
    ):
        if record.get(name) != expected[name]:
            raise RuntimeError(f"Development manifest sample identity drifted: {name}")
    pixels, actions = raw_sample_arrays(sample)
    return (
        pixels[:EXPECTED_HISTORY_TOKENS],
        actions[:EXPECTED_HISTORY_TOKENS].reshape(
            EXPECTED_HISTORY_TOKENS,
            EXPECTED_ACTION_BLOCK_RAW_STEPS,
            EXPECTED_ACTION_DIM,
        ),
        pixels[EXPECTED_HISTORY_TOKENS],
    )


def write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    """Write an immutable receipt without following an existing output."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite immutable output: {path}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o644)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
