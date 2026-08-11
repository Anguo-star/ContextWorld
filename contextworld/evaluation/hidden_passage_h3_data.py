from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import multiprocessing
import os
import shutil
import stat
import tempfile
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np

from contextworld.paths import (
    ARTIFACT_ROOT_ENV,
    LEGACY_ARTIFACT_PREFIX,
    portable_contextworld_path,
    resolve_contextworld_path,
)
from contextworld.synthesis.lance import build_lance_writer
from contextworld.synthesis.manifest import write_json
from contextworld.synthesis.stablewm import load_stable_worldmodel

from .hidden_passage import (
    DIRECTIONS,
    RULE_NAMES,
    HiddenPassageTemplate,
    simulate_template,
    validate_pair,
)
from .hidden_passage_env import (
    HIDDEN_PASSAGE_ENV_ID,
    PASSAGE_FACTOR,
    PASSAGE_RULES,
    register_hidden_passage_env,
)
from .hidden_passage_lance import (
    DIAGNOSTIC_KEYS,
    MODEL_KEYS,
    MODEL_STEPS,
    RAW_STEPS,
    REQUIRED_COLUMNS,
    WATCHED_VARIATIONS,
    _collection_actions,
    _model_blocks,
)
from .speed_door_rule_composition import (
    simulate_template as simulate_speed_door_rule_template,
    validate_factor_grid as validate_speed_door_rule_factor_grid,
)
from .speed_door_rule_v2_feasibility import (
    validate_v2_training_factor_grid,
)


GROUP_RULES: dict[str, tuple[str, ...]] = {
    "passage_passable": ("passable",),
    "passage_blocked": ("blocked",),
    "passage_mixed": ("passable", "blocked"),
}
GROUP_CATALOG_SUFFIX = {
    "passage_passable": "passable",
    "passage_blocked": "blocked",
    "passage_mixed": "mixed",
}
SPLITS = ("train", "val", "test")
COLLECTED_SPLITS = ("train", "val")
SPLIT_LABELS = {
    "train": "train",
    "val": "loader_val",
    "test": "final_eval_candidates",
}
LOGICAL_CONTENT_COLUMNS = (
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
)
LOGICAL_CONTENT_HASH_KIND = "ordered_episode_training_rows_v1"
SHARD_COMPLETION_PROTOCOL = (
    "contextworld.hidden_passage_h3.shard_completion.v1"
)
SHARD_COMPLETION_SUFFIX = ".complete.json"
STORAGE_CONTENT_HASH_KIND = "lance_directory_bytes_v1"
RELEASE_LOCK_PROTOCOL = "contextworld.hidden_passage_h3.release_lock.v1"
AUDIT_SCHEDULING_LOCK_PROTOCOL = (
    "contextworld.hidden_passage_h3.audit_scheduling_lock.v1"
)
PARALLEL_AUDIT_SCHEDULING_LOCK_PROTOCOL = (
    "contextworld.hidden_passage_h3.audit_scheduling_lock.v2"
)
TRAINING_RUN_LOCK_PROTOCOL = (
    "contextworld.hidden_passage_h3.training_run_lock.v1"
)
_ACTIVE_SIBLING_LOCK_DESCRIPTORS: set[int] = set()


def _close_sibling_lock_descriptors_after_fork() -> None:
    """Prevent forked workers from extending a parent's flock lifetime."""

    for descriptor in tuple(_ACTIVE_SIBLING_LOCK_DESCRIPTORS):
        try:
            os.close(descriptor)
        except OSError:
            pass
    _ACTIVE_SIBLING_LOCK_DESCRIPTORS.clear()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(
        after_in_child=_close_sibling_lock_descriptors_after_fork
    )


def lexical_absolute_path(path: str | Path) -> Path:
    """Normalize ``.``/``..`` without resolving any symbolic links."""

    return Path(os.path.abspath(os.fspath(Path(path).expanduser())))


def lexical_contextworld_path(
    value: str | Path,
    *,
    repo_root: Path,
) -> Path:
    """Map a portable path without canonicalizing aliases."""

    path = Path(value).expanduser()
    if path.is_absolute():
        return lexical_absolute_path(path)
    root = lexical_absolute_path(repo_root)
    if path.parts and path.parts[0] == LEGACY_ARTIFACT_PREFIX:
        configured = os.environ.get(ARTIFACT_ROOT_ENV)
        if configured:
            artifacts = lexical_absolute_path(configured)
        else:
            artifacts = lexical_absolute_path(
                root.parents[1] / "data/world_model/context_world"
            )
        return lexical_absolute_path(
            artifacts.joinpath(*path.parts[1:])
        )
    return lexical_absolute_path(root / path)


def require_lexical_containment(
    path: str | Path,
    root: str | Path,
) -> Path:
    """Return a lexical absolute path only when it is inside ``root``."""

    candidate = lexical_absolute_path(path)
    boundary = lexical_absolute_path(root)
    try:
        candidate.relative_to(boundary)
    except ValueError as exc:
        raise ValueError(
            "Hidden-passage path escapes its sealed release root: "
            f"path={candidate}, root={boundary}"
        ) from exc
    return candidate


def _path_components(path: Path) -> list[Path]:
    absolute = lexical_absolute_path(path)
    current = Path(absolute.anchor)
    values = [current]
    for part in absolute.parts[1:]:
        current = current / part
        values.append(current)
    return values


def _node_kind(mode: int) -> str:
    if stat.S_ISDIR(mode):
        return "directory"
    if stat.S_ISREG(mode):
        return "regular_file"
    if stat.S_ISLNK(mode):
        return "symlink"
    if stat.S_ISFIFO(mode):
        return "fifo"
    if stat.S_ISSOCK(mode):
        return "socket"
    if stat.S_ISCHR(mode):
        return "character_device"
    if stat.S_ISBLK(mode):
        return "block_device"
    return "unsupported"


def require_safe_path(
    path: str | Path,
    *,
    leaf_kind: str,
    containment_root: str | Path | None = None,
) -> Path:
    """Reject symlinks and special nodes at every existing path component."""

    candidate = lexical_absolute_path(path)
    if containment_root is not None:
        candidate = require_lexical_containment(
            candidate,
            containment_root,
        )
    components = _path_components(candidate)
    for index, component in enumerate(components):
        try:
            metadata = os.lstat(component)
        except FileNotFoundError:
            raise FileNotFoundError(component) from None
        observed = _node_kind(metadata.st_mode)
        expected = leaf_kind if index == len(components) - 1 else "directory"
        if observed != expected:
            raise ValueError(
                "Unsafe hidden-passage path component: "
                f"path={candidate}, component={component}, "
                f"expected={expected}, observed={observed}"
            )
    return candidate


def require_safe_directory(
    path: str | Path,
    *,
    containment_root: str | Path | None = None,
) -> Path:
    return require_safe_path(
        path,
        leaf_kind="directory",
        containment_root=containment_root,
    )


def require_safe_regular_file(
    path: str | Path,
    *,
    containment_root: str | Path | None = None,
) -> Path:
    return require_safe_path(
        path,
        leaf_kind="regular_file",
        containment_root=containment_root,
    )


def require_safe_missing_or_directory(path: str | Path) -> Path:
    """Validate all ancestors and allow only a missing or real directory leaf."""

    candidate = lexical_absolute_path(path)
    components = _path_components(candidate)
    missing_seen = False
    for index, component in enumerate(components):
        try:
            metadata = os.lstat(component)
        except FileNotFoundError:
            missing_seen = True
            continue
        if missing_seen:
            raise ValueError(
                "A hidden-passage path exists below a missing ancestor: "
                f"{candidate}"
            )
        observed = _node_kind(metadata.st_mode)
        expected = "directory"
        if observed != expected:
            raise ValueError(
                "Unsafe hidden-passage output path component: "
                f"path={candidate}, component={component}, "
                f"expected={expected}, observed={observed}"
            )
        if index == len(components) - 1:
            return candidate
    return candidate


def _regular_tree_files(path: Path) -> list[Path]:
    """List regular files without following any alias or special node."""

    root = require_safe_directory(path)
    files: list[Path] = []

    def visit(directory: Path) -> None:
        with os.scandir(directory) as iterator:
            entries = sorted(iterator, key=lambda value: value.name)
        for entry in entries:
            entry_path = directory / entry.name
            metadata = entry.stat(follow_symlinks=False)
            kind = _node_kind(metadata.st_mode)
            if kind == "directory":
                visit(entry_path)
            elif kind == "regular_file":
                files.append(entry_path)
            else:
                raise ValueError(
                    "Hidden-passage directory tree contains an unsafe node: "
                    f"path={entry_path}, kind={kind}"
                )

    visit(root)
    return sorted(files, key=lambda value: value.relative_to(root).as_posix())


def validate_regular_directory_tree(path: str | Path) -> dict[str, Any]:
    """Validate a directory tree without changing its byte-hash semantics."""

    root = require_safe_directory(path)
    files = _regular_tree_files(root)
    return {
        "path": str(root),
        "regular_files": len(files),
        "symlinks_or_special_nodes": 0,
        "passed": True,
    }


@contextmanager
def hidden_passage_release_lock(
    release_root: str | Path,
    *,
    exclusive: bool,
) -> Iterator[dict[str, Any]]:
    """Hold one cooperative lock adjacent to a release for an entire run."""

    root = require_safe_missing_or_directory(release_root)
    parent = root.parent
    require_safe_directory(parent)
    lock_path = parent / f".{root.name}.hidden-passage.lock"
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(lock_path, flags, 0o600)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(
                f"Hidden-passage release lock is not regular: {lock_path}"
            )
        operation = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        fcntl.flock(descriptor, operation)
        yield {
            "protocol": RELEASE_LOCK_PROTOCOL,
            "path": str(lock_path),
            "mode": "exclusive" if exclusive else "shared",
        }
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
                os.close(descriptor)


def hidden_passage_audit_scheduling_lock_path(
    release_root: str | Path,
) -> Path:
    root = require_safe_missing_or_directory(release_root)
    require_safe_directory(root.parent)
    return (
        root.parent
        / f".{root.name}.hidden-passage-audit-scheduling.lock"
    )


@contextmanager
def _hidden_passage_sibling_exclusive_lock(
    release_root: str | Path,
    *,
    suffix: str,
    protocol: str,
    policy: str,
    blocking: bool = True,
    write_holder_pid: bool = False,
    shared: bool = False,
) -> Iterator[dict[str, Any]]:
    root = require_safe_missing_or_directory(release_root)
    require_safe_directory(root.parent)
    lock_path = root.parent / f".{root.name}.{suffix}"
    if (
        not hasattr(os, "O_CLOEXEC")
        or not hasattr(os, "O_NOFOLLOW")
        or not hasattr(os, "register_at_fork")
    ):
        raise RuntimeError(
            "Hidden-passage sibling locks require Linux O_CLOEXEC and "
            "O_NOFOLLOW plus register_at_fork"
        )
    flags = os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as exc:
        if exc.errno in {errno.ELOOP, errno.EMLINK}:
            raise ValueError(
                "Hidden-passage sibling lock is an unsafe alias: "
                f"{lock_path}"
            ) from exc
        raise
    receipt = {
        "protocol": protocol,
        "policy": policy,
        "maximum_concurrency": 1,
        "path": str(lock_path),
        "blocking": blocking,
        "mode": "shared" if shared else "exclusive",
        "acquired": False,
        "wait_seconds": None,
        "hold_seconds": None,
        "path_identity_verified": False,
        "path_identity_verified_after_acquire": False,
        "descriptor_inheritable": None,
        "fork_child_close_registered": hasattr(os, "register_at_fork"),
        "holder_pid": None,
        "holder_pid_written": False,
        "released": False,
    }
    try:
        def verify_path_identity() -> None:
            metadata = os.fstat(descriptor)
            path_metadata = os.lstat(lock_path)
            parent_metadata = os.lstat(root.parent)
            unsafe = (
                not stat.S_ISREG(metadata.st_mode)
                or not stat.S_ISREG(path_metadata.st_mode)
                or int(metadata.st_dev) != int(path_metadata.st_dev)
                or int(metadata.st_ino) != int(path_metadata.st_ino)
                or int(metadata.st_uid) != int(parent_metadata.st_uid)
                or int(metadata.st_gid) != int(parent_metadata.st_gid)
                or int(metadata.st_nlink) != 1
                or stat.S_IMODE(metadata.st_mode) != 0o600
            )
            if unsafe:
                raise ValueError(
                    "Hidden-passage sibling lock is unsafe: "
                    f"path={lock_path}, "
                    f"mode={oct(stat.S_IMODE(metadata.st_mode))}, "
                    f"owner={metadata.st_uid}, links={metadata.st_nlink}"
                )

        verify_path_identity()
        receipt["path_identity_verified"] = True
        os.set_inheritable(descriptor, False)
        receipt["descriptor_inheritable"] = os.get_inheritable(descriptor)
        _ACTIVE_SIBLING_LOCK_DESCRIPTORS.add(descriptor)
        queued_at = time.monotonic()
        operation = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
        if not blocking:
            operation |= fcntl.LOCK_NB
        fcntl.flock(descriptor, operation)
        acquired_at = time.monotonic()
        verify_path_identity()
        receipt["path_identity_verified_after_acquire"] = True
        receipt["acquired"] = True
        receipt["wait_seconds"] = acquired_at - queued_at
        if write_holder_pid:
            holder_pid = int(os.getpid())
            payload = f"{holder_pid}\n".encode("ascii")
            os.ftruncate(descriptor, 0)
            os.lseek(descriptor, 0, os.SEEK_SET)
            if os.write(descriptor, payload) != len(payload):
                raise RuntimeError(
                    "Failed to publish the passage root-lock holder PID"
                )
            os.fsync(descriptor)
            receipt["holder_pid"] = holder_pid
            receipt["holder_pid_written"] = True
        try:
            yield receipt
        finally:
            receipt["hold_seconds"] = time.monotonic() - acquired_at
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            receipt["released"] = True
    finally:
        _ACTIVE_SIBLING_LOCK_DESCRIPTORS.discard(descriptor)
        os.close(descriptor)


@contextmanager
def hidden_passage_audit_scheduling_lock(
    release_root: str | Path,
    *,
    blocking: bool = True,
    shared: bool = False,
) -> Iterator[dict[str, Any]]:
    """Schedule read-only audits under the sealed-release read lock."""

    with _hidden_passage_sibling_exclusive_lock(
        release_root,
        suffix="hidden-passage-audit-scheduling.lock",
        protocol=(
            PARALLEL_AUDIT_SCHEDULING_LOCK_PROTOCOL
            if shared
            else AUDIT_SCHEDULING_LOCK_PROTOCOL
        ),
        policy=(
            "sibling_shared_flock"
            if shared
            else "sibling_exclusive_flock"
        ),
        blocking=blocking,
        shared=shared,
    ) as receipt:
        yield receipt


def hidden_passage_training_run_lock_path(
    release_root: str | Path,
) -> Path:
    root = require_safe_missing_or_directory(release_root)
    require_safe_directory(root.parent)
    return root.parent / f".{root.name}.hidden-passage-training-run.lock"


@contextmanager
def hidden_passage_training_run_lock(
    release_root: str | Path,
) -> Iterator[dict[str, Any]]:
    """Reject a second root launcher for the same single-node release."""

    with _hidden_passage_sibling_exclusive_lock(
        release_root,
        suffix="hidden-passage-training-run.lock",
        protocol=TRAINING_RUN_LOCK_PROTOCOL,
        policy="one_root_training_run_per_release",
        blocking=False,
        write_holder_pid=True,
    ) as receipt:
        yield receipt


def verify_hidden_passage_training_run_parent(
    release_root: str | Path,
) -> dict[str, Any]:
    """Admit only a child whose direct parent holds this release's lock."""

    root = require_safe_missing_or_directory(release_root)
    require_safe_directory(root.parent)
    lock_path = hidden_passage_training_run_lock_path(root)
    if (
        not hasattr(os, "O_CLOEXEC")
        or not hasattr(os, "O_NOFOLLOW")
        or not hasattr(os, "pread")
    ):
        raise RuntimeError(
            "Passage child admission requires Linux O_CLOEXEC, O_NOFOLLOW, "
            "and pread"
        )
    flags = os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        descriptor = os.open(lock_path, flags)
    except OSError as exc:
        if exc.errno in {errno.ELOOP, errno.EMLINK}:
            raise ValueError(
                "Hidden-passage training-run lock is an unsafe alias: "
                f"{lock_path}"
            ) from exc
        raise RuntimeError(
            "A nonzero LOCAL_RANK has no active root training lock: "
            f"{lock_path}"
        ) from exc
    try:
        metadata = os.fstat(descriptor)
        path_metadata = os.lstat(lock_path)
        parent_metadata = os.lstat(root.parent)
        safe = (
            stat.S_ISREG(metadata.st_mode)
            and stat.S_ISREG(path_metadata.st_mode)
            and int(metadata.st_dev) == int(path_metadata.st_dev)
            and int(metadata.st_ino) == int(path_metadata.st_ino)
            and int(metadata.st_uid) == int(parent_metadata.st_uid)
            and int(metadata.st_gid) == int(parent_metadata.st_gid)
            and int(metadata.st_nlink) == 1
            and stat.S_IMODE(metadata.st_mode) == 0o600
        )
        if not safe:
            raise ValueError(
                "Hidden-passage training-run parent lock is unsafe: "
                f"{lock_path}"
            )
        os.set_inheritable(descriptor, False)
        payload = os.pread(descriptor, 64, 0)
        try:
            holder_text = payload.decode("ascii")
        except UnicodeDecodeError as exc:
            raise RuntimeError(
                "Training-run lock holder PID is not ASCII"
            ) from exc
        if (
            not holder_text.endswith("\n")
            or not holder_text[:-1].isdigit()
            or len(holder_text) > 32
        ):
            raise RuntimeError(
                "Training-run lock has no valid holder PID"
            )
        holder_pid = int(holder_text[:-1])
        parent_pid = int(os.getppid())
        if holder_pid != parent_pid:
            raise RuntimeError(
                "A nonzero LOCAL_RANK is not a direct child of the root "
                f"lock holder: holder_pid={holder_pid}, parent_pid={parent_pid}"
            )
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock_is_held = True
        else:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            lock_is_held = False
        if not lock_is_held:
            raise RuntimeError(
                "A nonzero LOCAL_RANK cannot prove an active root lock"
            )
        final_metadata = os.fstat(descriptor)
        final_path_metadata = os.lstat(lock_path)
        if (
            int(final_metadata.st_dev) != int(final_path_metadata.st_dev)
            or int(final_metadata.st_ino) != int(final_path_metadata.st_ino)
        ):
            raise ValueError(
                "Training-run lock identity changed during child admission"
            )
        return {
            "protocol": TRAINING_RUN_LOCK_PROTOCOL,
            "policy": "direct_parent_holds_root_training_lock",
            "holder_pid": holder_pid,
            "parent_pid": parent_pid,
            "lock_is_held": True,
            "path_identity_verified": True,
            "descriptor_inheritable": os.get_inheritable(descriptor),
            "passed": True,
        }
    finally:
        os.close(descriptor)


@dataclass(frozen=True)
class HiddenPassageDoorSplits:
    train: tuple[int, ...]
    val: tuple[int, ...]
    test: tuple[int, ...]
    guard: tuple[int, ...]

    def as_report(self) -> dict[str, list[int]]:
        return {
            "train": list(self.train),
            "loader_val": list(self.val),
            "final_eval_candidates": list(self.test),
            "guard": list(self.guard),
        }


@dataclass
class HiddenPassageEpisodePlan:
    template: HiddenPassageTemplate
    rule: str
    collection_actions: np.ndarray
    expected_hashes: dict[str, str]
    pair_metrics: dict[str, Any]
    agent_speed: float = 5.0

    @property
    def template_id(self) -> str:
        return self.template.template_id


@dataclass(frozen=True)
class HiddenPassageShardPlan:
    split: str
    door_position: int
    rule: str
    pair_id: str
    fingerprint: str
    scenario_id: str
    table_path: Path
    episode_manifest_path: Path


@dataclass(frozen=True)
class HiddenPassageShardCollectionJob:
    shard: HiddenPassageShardPlan
    plans: tuple[HiddenPassageEpisodePlan, ...]


_SHARD_WORKER_SWM: Any | None = None
_SHARD_WORKER_CONFIG: dict[str, Any] | None = None


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    path = require_safe_regular_file(path)
    digest = hashlib.sha256()
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    with os.fdopen(descriptor, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_sha256(path: Path) -> str:
    path = require_safe_directory(path)
    digest = hashlib.sha256()
    files = _regular_tree_files(path)
    for value in files:
        relative = value.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(value, flags)
        with os.fdopen(descriptor, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _fsync_path(path: Path) -> None:
    """Request persistence without rejecting filesystems that lack fsync."""

    flags = os.O_RDONLY
    metadata = os.lstat(path)
    kind = _node_kind(metadata.st_mode)
    if kind == "directory":
        flags |= getattr(os, "O_DIRECTORY", 0)
    elif kind != "regular_file":
        raise ValueError(
            f"Refusing to fsync unsafe hidden-passage node: {path}"
        )
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        if exc.errno in {
            errno.EINVAL,
            errno.ENOTSUP,
            errno.EOPNOTSUPP,
        }:
            return
        raise
    try:
        try:
            os.fsync(descriptor)
        except OSError as exc:
            if exc.errno not in {
                errno.EINVAL,
                errno.ENOTSUP,
                errno.EOPNOTSUPP,
            }:
                raise
    finally:
        os.close(descriptor)


def _fsync_tree(path: Path) -> None:
    """Sync files before directories; completion is published afterwards."""

    path = require_safe_directory(path)
    files = _regular_tree_files(path)
    directories = sorted(
        {
            parent
            for value in files
            for parent in value.parents
            if parent != path and path in parent.parents
        },
        key=lambda value: len(value.parts),
        reverse=True,
    )
    for value in files:
        _fsync_path(value)
    for value in [*directories, path]:
        _fsync_path(value)


def shard_completion_marker_path(table_path: Path) -> Path:
    return table_path.with_name(
        f"{table_path.name}{SHARD_COMPLETION_SUFFIX}"
    )


def _episode_manifest_rows(path: Path) -> list[dict[str, Any]]:
    path = require_safe_regular_file(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "Invalid hidden-passage episode manifest JSON: "
                    f"{path}:{line_number}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(
                    "Invalid hidden-passage episode manifest row: "
                    f"{path}:{line_number}"
                )
            rows.append(row)
    if not rows:
        raise ValueError(
            f"Empty hidden-passage episode manifest: {path}"
        )
    return rows


def verify_hidden_passage_shard_completion(
    *,
    table_path: Path,
    episode_manifest_path: Path,
    expected_scenario_id: str,
    expected_fingerprint: str,
    expected_content_sha256: str | None = None,
    expected_storage_sha256: str | None = None,
    expected_episode_manifest_sha256: str | None = None,
    expected_marker_sha256: str | None = None,
) -> dict[str, Any]:
    """Fail closed unless a shard's last-written completion marker is valid."""

    marker_path = shard_completion_marker_path(table_path)
    try:
        table_path = require_safe_directory(table_path)
    except (FileNotFoundError, ValueError) as exc:
        raise ValueError(
            f"Hidden-passage shard is not a real directory: {table_path}; "
            f"{exc}"
        ) from exc
    try:
        episode_manifest_path = require_safe_regular_file(
            episode_manifest_path
        )
    except (FileNotFoundError, ValueError) as exc:
        raise ValueError(
            "Hidden-passage shard has no regular episode manifest: "
            f"{episode_manifest_path}; {exc}"
        ) from exc
    try:
        marker_path = require_safe_regular_file(marker_path)
    except (FileNotFoundError, ValueError) as exc:
        raise ValueError(
            "Hidden-passage shard has no valid completion marker: "
            f"{marker_path}; {exc}"
        ) from exc

    marker_sha256 = file_sha256(marker_path)
    if (
        expected_marker_sha256 is not None
        and marker_sha256 != expected_marker_sha256
    ):
        raise ValueError(
            "Hidden-passage completion marker hash mismatch: "
            f"expected={expected_marker_sha256}, "
            f"observed={marker_sha256}, path={marker_path}"
        )
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid hidden-passage completion marker: {marker_path}"
        ) from exc
    if not isinstance(marker, dict):
        raise ValueError(
            f"Invalid hidden-passage completion marker: {marker_path}"
        )

    identity_checks = {
        "schema_version": marker.get("schema_version") == 1,
        "protocol": (
            marker.get("protocol") == SHARD_COMPLETION_PROTOCOL
        ),
        "status": marker.get("status") == "complete",
        "scenario_id": marker.get("scenario_id")
        == expected_scenario_id,
        "fingerprint": marker.get("fingerprint")
        == expected_fingerprint,
        "table_directory_name": marker.get("table_directory_name")
        == table_path.name,
        "episode_manifest_name": marker.get("episode_manifest_name")
        == episode_manifest_path.name,
        "content_sha256_kind": marker.get("content_sha256_kind")
        == LOGICAL_CONTENT_HASH_KIND,
        "storage_sha256_kind": marker.get("storage_sha256_kind")
        == STORAGE_CONTENT_HASH_KIND,
        "rows_per_episode": marker.get("rows_per_episode") == RAW_STEPS,
        "published_after_full_audit": (
            marker.get("published_after_full_audit") is True
        ),
    }
    if not all(identity_checks.values()):
        raise ValueError(
            "Hidden-passage completion marker identity failed: "
            f"path={marker_path}, checks={identity_checks}"
        )

    actual_storage_sha256 = directory_sha256(table_path)
    actual_episode_manifest_sha256 = file_sha256(
        episode_manifest_path
    )
    episode_rows = _episode_manifest_rows(episode_manifest_path)
    actual_content_sha256 = logical_shard_content_sha256(episode_rows)
    expected_values = {
        "storage_sha256": (
            marker.get("storage_sha256"),
            actual_storage_sha256,
            expected_storage_sha256,
        ),
        "episode_manifest_sha256": (
            marker.get("episode_manifest_sha256"),
            actual_episode_manifest_sha256,
            expected_episode_manifest_sha256,
        ),
        "content_sha256": (
            marker.get("content_sha256"),
            actual_content_sha256,
            expected_content_sha256,
        ),
    }
    mismatches: dict[str, dict[str, Any]] = {}
    for field, (declared, actual, externally_expected) in (
        expected_values.items()
    ):
        if declared != actual or (
            externally_expected is not None
            and declared != externally_expected
        ):
            mismatches[field] = {
                "marker": declared,
                "actual": actual,
                "expected": externally_expected,
            }
    if int(marker.get("episode_count", -1)) != len(episode_rows):
        mismatches["episode_count"] = {
            "marker": marker.get("episode_count"),
            "actual": len(episode_rows),
        }
    expected_raw_rows = len(episode_rows) * RAW_STEPS
    if int(marker.get("raw_rows", -1)) != expected_raw_rows:
        mismatches["raw_rows"] = {
            "marker": marker.get("raw_rows"),
            "actual": expected_raw_rows,
        }
    if mismatches:
        raise ValueError(
            "Hidden-passage completion marker content binding failed: "
            f"path={marker_path}, mismatches={mismatches}"
        )
    return {
        "path": str(marker_path),
        "sha256": marker_sha256,
        "protocol": SHARD_COMPLETION_PROTOCOL,
        "scenario_id": expected_scenario_id,
        "fingerprint": expected_fingerprint,
        "storage_sha256": actual_storage_sha256,
        "episode_manifest_sha256": actual_episode_manifest_sha256,
        "content_sha256": actual_content_sha256,
        "episode_count": len(episode_rows),
        "raw_rows": expected_raw_rows,
        "passed": True,
    }


def _publish_hidden_passage_shard_completion(
    *,
    shard: HiddenPassageShardPlan,
    episode_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Write the marker last, after copied bytes and logical audit are sealed."""

    marker_path = shard_completion_marker_path(shard.table_path)
    if marker_path.exists() or marker_path.is_symlink():
        raise FileExistsError(marker_path)
    require_safe_directory(shard.table_path)
    require_safe_regular_file(shard.episode_manifest_path)

    content_sha256 = logical_shard_content_sha256(episode_rows)
    sidecar_rows = _episode_manifest_rows(shard.episode_manifest_path)
    if sidecar_rows != episode_rows:
        raise ValueError(
            "Refusing to publish a completion marker unless the sidecar "
            f"exactly equals the audited episode rows: {shard.scenario_id}"
        )
    if logical_shard_content_sha256(sidecar_rows) != content_sha256:
        raise ValueError(
            "Refusing to publish a completion marker for a sidecar that "
            f"differs from the audited rows: {shard.scenario_id}"
        )
    _fsync_tree(shard.table_path)
    _fsync_path(shard.episode_manifest_path)
    _fsync_path(shard.episode_manifest_path.parent)
    payload = {
        "schema_version": 1,
        "protocol": SHARD_COMPLETION_PROTOCOL,
        "status": "complete",
        "scenario_id": shard.scenario_id,
        "fingerprint": shard.fingerprint,
        "table_directory_name": shard.table_path.name,
        "episode_manifest_name": shard.episode_manifest_path.name,
        "storage_sha256_kind": STORAGE_CONTENT_HASH_KIND,
        "storage_sha256": directory_sha256(shard.table_path),
        "content_sha256_kind": LOGICAL_CONTENT_HASH_KIND,
        "content_sha256": content_sha256,
        "episode_manifest_sha256": file_sha256(
            shard.episode_manifest_path
        ),
        "episode_count": len(episode_rows),
        "rows_per_episode": RAW_STEPS,
        "raw_rows": len(episode_rows) * RAW_STEPS,
        "published_after_full_audit": True,
    }
    write_json(marker_path, payload)
    _fsync_path(marker_path.parent)
    return verify_hidden_passage_shard_completion(
        table_path=shard.table_path,
        episode_manifest_path=shard.episode_manifest_path,
        expected_scenario_id=shard.scenario_id,
        expected_fingerprint=shard.fingerprint,
        expected_content_sha256=content_sha256,
        expected_storage_sha256=payload["storage_sha256"],
        expected_episode_manifest_sha256=payload[
            "episode_manifest_sha256"
        ],
        expected_marker_sha256=file_sha256(marker_path),
    )


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(f"{array.dtype.str}:{array.shape}".encode("utf-8"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _integer_range(specification: dict[str, Any]) -> list[int]:
    start = int(specification["start"])
    stop = int(specification["stop_inclusive"])
    step = int(specification.get("step", 1))
    if step <= 0 or stop < start:
        raise ValueError(f"Invalid integer range: {specification}")
    return list(range(start, stop + 1, step))


def door_splits_for_scale(
    config: dict[str, Any],
    scale: str,
) -> HiddenPassageDoorSplits:
    scales = config.get("scales")
    if not isinstance(scales, dict) or scale not in scales:
        raise ValueError(
            f"Unknown scale {scale!r}; expected one of "
            f"{sorted(scales or {})}"
        )
    raw = scales[scale]
    explicit = raw.get("door_splits")
    if explicit is not None:
        splits = HiddenPassageDoorSplits(
            train=tuple(map(int, explicit["train"])),
            val=tuple(map(int, explicit["loader_val"])),
            test=tuple(map(int, explicit["final_eval_candidates"])),
            guard=tuple(map(int, explicit.get("guard", ()))),
        )
    else:
        selection = raw["door_selection"]
        safe = set(_integer_range(selection["safe_integer_range"]))
        final_eval = set(
            _integer_range(selection["final_eval_candidate_range"])
        )
        eligible = np.asarray(sorted(safe - final_eval), dtype=np.int64)
        rng = np.random.default_rng(int(selection["split_seed"]))
        shuffled = rng.permutation(eligible).tolist()
        train_count = int(selection["train_count"])
        val_count = int(selection["loader_val_count"])
        expected_guard = int(selection["expected_guard_count"])
        if train_count + val_count + expected_guard != len(shuffled):
            raise ValueError(
                "Formal hidden-passage door counts do not exhaust the "
                f"eligible pool: train={train_count}, val={val_count}, "
                f"guard={expected_guard}, eligible={len(shuffled)}"
            )
        splits = HiddenPassageDoorSplits(
            train=tuple(sorted(map(int, shuffled[:train_count]))),
            val=tuple(
                sorted(
                    map(
                        int,
                        shuffled[train_count : train_count + val_count],
                    )
                )
            ),
            test=tuple(sorted(final_eval)),
            guard=tuple(
                sorted(map(int, shuffled[train_count + val_count :]))
            ),
        )
        observed_intersection = safe & final_eval
        expected_intersection = int(
            selection["expected_safe_final_eval_intersection_count"]
        )
        if len(observed_intersection) != expected_intersection:
            raise ValueError(
                "Safe/final-Eval door intersection changed: "
                f"expected {expected_intersection}, "
                f"observed {len(observed_intersection)}"
            )

    named = {
        "train": set(splits.train),
        "loader_val": set(splits.val),
        "final_eval_candidates": set(splits.test),
        "guard": set(splits.guard),
    }
    if any(len(values) != len(getattr(splits, key)) for key, values in {
        "train": named["train"],
        "val": named["loader_val"],
        "test": named["final_eval_candidates"],
        "guard": named["guard"],
    }.items()):
        raise ValueError("A door split contains duplicate positions")
    overlaps: dict[str, list[int]] = {}
    items = list(named.items())
    for index, (left_name, left) in enumerate(items):
        for right_name, right in items[index + 1 :]:
            overlap = sorted(left & right)
            if overlap:
                overlaps[f"{left_name}__{right_name}"] = overlap
    if overlaps:
        raise ValueError(f"Hidden-passage door splits overlap: {overlaps}")

    expected_counts = raw.get("expected_door_counts", {})
    observed_counts = {
        "train": len(splits.train),
        "loader_val": len(splits.val),
        "final_eval_candidates": len(splits.test),
        "guard": len(splits.guard),
    }
    mismatches = {
        name: {
            "expected": int(expected),
            "observed": observed_counts.get(name),
        }
        for name, expected in expected_counts.items()
        if observed_counts.get(name) != int(expected)
    }
    if mismatches:
        raise ValueError(f"Door split counts differ: {mismatches}")
    return splits


def audit_validation_exclusion(
    config: dict[str, Any],
    *,
    scale: str,
    door_splits: HiddenPassageDoorSplits,
    repo_root: Path,
) -> tuple[dict[str, Any], set[str]]:
    frozen = config["validation_exclusion"]
    catalog_path = resolve_contextworld_path(
        frozen["catalog"],
        repo_root=repo_root,
    )
    manifest_path = resolve_contextworld_path(
        frozen["manifest"],
        repo_root=repo_root,
    )
    if not catalog_path.is_file():
        raise FileNotFoundError(catalog_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    observed_catalog_sha = file_sha256(catalog_path)
    observed_manifest_sha = file_sha256(manifest_path)
    if observed_catalog_sha != str(frozen["catalog_sha256"]):
        raise RuntimeError(
            "Frozen hidden-passage Validation catalog hash mismatch: "
            f"expected {frozen['catalog_sha256']}, "
            f"observed {observed_catalog_sha}"
        )
    if observed_manifest_sha != str(frozen["manifest_sha256"]):
        raise RuntimeError(
            "Frozen hidden-passage training exclusion hash mismatch: "
            f"expected {frozen['manifest_sha256']}, "
            f"observed {observed_manifest_sha}"
        )
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_benchmark = str(frozen["benchmark"])
    if (
        str(catalog.get("benchmark")) != expected_benchmark
        or str(manifest.get("benchmark")) != expected_benchmark
    ):
        raise RuntimeError(
            "Frozen hidden-passage Validation benchmark identity mismatch"
        )
    content_sha = str(frozen["content_manifest_sha256"])
    if (
        str(catalog.get("content_manifest_sha256")) != content_sha
        or str(manifest.get("content_manifest_sha256")) != content_sha
    ):
        raise RuntimeError(
            "Frozen hidden-passage Validation content identity mismatch"
        )
    eval_only = set(map(int, manifest["eval_only_door_positions"]))
    train = set(door_splits.train)
    val = set(door_splits.val)
    test = set(door_splits.test)
    exact_test_required = scale in set(
        map(str, frozen.get("exact_test_door_scales", ["formal"]))
    )
    checks = {
        "benchmark_identity_exact": True,
        "catalog_sha256_exact": True,
        "training_exclusion_manifest_sha256_exact": True,
        "content_manifest_sha256_exact": True,
        "manifest_query_count_exact": (
            int(manifest["query_count"])
            == len(manifest["query_records"])
            == int(frozen["query_count"])
        ),
        "train_doors_exclude_all_eval_only_doors": not (train & eval_only),
        "loader_val_doors_exclude_all_eval_only_doors": not (
            val & eval_only
        ),
        "test_doors_are_eval_only": test <= eval_only,
        "eval_only_door_count_exact": (
            len(eval_only) == int(frozen["eval_only_door_count"])
        ),
        "test_doors_equal_all_eval_only_when_required": (
            test == eval_only if exact_test_required else True
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(
            "Hidden-passage Validation exclusion failed: "
            f"{checks}; train_overlap={sorted(train & eval_only)}, "
            f"val_overlap={sorted(val & eval_only)}, "
            f"test_missing={sorted(eval_only - test)}, "
            f"test_extra={sorted(test - eval_only)}"
        )
    query_hashes = {
        str(row["query_pixels_sha256"])
        for row in manifest["query_records"]
    }
    return (
        {
            "passed": True,
            "checks": checks,
            "catalog": portable_contextworld_path(
                catalog_path,
                repo_root=repo_root,
            ),
            "catalog_sha256": observed_catalog_sha,
            "manifest": portable_contextworld_path(
                manifest_path,
                repo_root=repo_root,
            ),
            "manifest_sha256": observed_manifest_sha,
            "content_manifest_sha256": content_sha,
            "eval_only_door_positions": sorted(eval_only),
            "eval_only_door_count": len(eval_only),
            "selected_query_count": len(query_hashes),
            "exact_test_door_set_required": exact_test_required,
        },
        query_hashes,
    )


def _scale_axes(
    config: dict[str, Any],
    scale: str,
) -> tuple[list[str], list[float], list[float]]:
    geometry = dict(config["geometry"])
    geometry.update(config["scales"][scale].get("geometry", {}))
    directions = list(map(str, geometry["directions"]))
    offsets = list(map(float, geometry["doorway_offsets_px"]))
    distances = list(map(float, geometry["wall_distances_px"]))
    if tuple(directions) != DIRECTIONS:
        raise ValueError(
            "History-3 hidden passage requires both directions in the "
            f"frozen order {DIRECTIONS}, got {directions}"
        )
    if len(offsets) != len(set(offsets)):
        raise ValueError("Duplicate doorway offsets")
    if len(distances) != len(set(distances)):
        raise ValueError("Duplicate wall distances")
    return directions, offsets, distances


def templates_for_door(
    config: dict[str, Any],
    *,
    scale: str,
    door_position: int,
) -> list[HiddenPassageTemplate]:
    directions, offsets, distances = _scale_axes(config, scale)
    catalog_seed = int(config["catalog_seed"])
    wall = dict(config["geometry"]["wall_geometry"])
    left_contact_x = float(wall["left_contact_x"])
    right_contact_x = float(wall["right_contact_x"])
    left_goal_x = float(wall["left_to_right_goal_x"])
    right_goal_x = float(wall["right_to_left_goal_x"])
    vertical_sign = 1.0 if int(door_position) <= 112 else -1.0
    templates: list[HiddenPassageTemplate] = []
    for direction_index, direction in enumerate(directions):
        left_to_right = direction == "left_to_right"
        for distance_index, distance in enumerate(distances):
            reset_x = (
                left_contact_x - distance
                if left_to_right
                else right_contact_x + distance
            )
            goal_x = left_goal_x if left_to_right else right_goal_x
            for offset_index, offset in enumerate(offsets):
                reset_y = float(door_position) + vertical_sign * offset
                simulator_seed = int(
                    np.random.SeedSequence(
                        [
                            catalog_seed,
                            int(door_position),
                            direction_index,
                            distance_index,
                            offset_index,
                        ]
                    ).generate_state(1)[0]
                )
                direction_slug = "ltr" if left_to_right else "rtl"
                template_id = (
                    f"hp-d{int(door_position):03d}-{direction_slug}-"
                    f"w{distance_index:02d}-o{offset_index:02d}"
                )
                templates.append(
                    HiddenPassageTemplate(
                        template_id=template_id,
                        door_position=int(door_position),
                        direction=direction,
                        doorway_offset_px=float(offset),
                        reset_state=(float(reset_x), float(reset_y)),
                        goal_state=(float(goal_x), float(reset_y)),
                        simulator_seed=simulator_seed,
                    )
                )
    expected = len(directions) * len(offsets) * len(distances)
    if len(templates) != expected:
        raise RuntimeError("Hidden-passage template construction failed")
    if len({value.template_id for value in templates}) != len(templates):
        raise RuntimeError("Hidden-passage template IDs are not unique")
    return templates


def _expected_hashes(reference: dict[str, Any]) -> dict[str, str]:
    model_pixels = np.concatenate(
        [
            np.asarray(reference["history_pixels"], dtype=np.uint8),
            np.asarray(reference["target_pixels"], dtype=np.uint8)[None],
        ],
        axis=0,
    )
    model_actions = _model_blocks(reference).astype(np.float32)
    model_proprio = np.concatenate(
        [
            np.asarray(reference["history_states"], dtype=np.float32),
            np.asarray(reference["target_state"], dtype=np.float32)[None],
        ],
        axis=0,
    )
    return {
        "model_pixels": array_sha256(model_pixels),
        "model_actions": array_sha256(model_actions),
        "model_proprio": array_sha256(model_proprio),
        "initial_pixels": array_sha256(model_pixels[0]),
        "middle_pixels": array_sha256(model_pixels[1]),
        "query_pixels": array_sha256(model_pixels[2]),
        "future_pixels": array_sha256(model_pixels[3]),
        "goal_pixels": array_sha256(
            np.asarray(reference["goal_pixels"], dtype=np.uint8)
        ),
    }


def episode_plans_for_door(
    config: dict[str, Any],
    *,
    scale: str,
    door_position: int,
) -> dict[str, list[HiddenPassageEpisodePlan]]:
    if str(config["protocol"].get("task", "hidden_passage")) in {
        "speed_door_rule_composition",
        "speed_door_rule_composition_v2",
    }:
        return _composition_episode_plans_for_door(
            config,
            scale=scale,
            door_position=door_position,
        )
    gates = config["gates"]
    plans = {rule: [] for rule in RULE_NAMES}
    for template in templates_for_door(
        config,
        scale=scale,
        door_position=door_position,
    ):
        references = {
            rule: simulate_template(
                template,
                rule=rule,
                agent_speed=float(config["protocol"]["agent_speed"]),
                door_number=int(config["protocol"]["door_number"]),
            )
            for rule in RULE_NAMES
        }
        pair = validate_pair(
            template,
            references["passable"],
            references["blocked"],
            minimum_middle_state_gap_px=float(
                gates["minimum_middle_state_gap_px"]
            ),
            minimum_future_state_gap_px=float(
                gates["minimum_future_state_gap_px"]
            ),
        )
        if not pair["passed"]:
            failed = sorted(
                key for key, passed in pair["checks"].items() if not passed
            )
            raise RuntimeError(
                f"Hidden-passage pair failed for {template.template_id}: "
                f"{failed}"
            )
        pair_metrics = {
            "middle_state_gap_px": pair["middle_state_gap_px"],
            "future_state_gap_px": pair["future_state_gap_px"],
            "collision_projection_used_to_restore_query": pair[
                "collision_projection_used_to_restore_query"
            ],
        }
        for rule in RULE_NAMES:
            plans[rule].append(
                HiddenPassageEpisodePlan(
                    template=template,
                    rule=rule,
                    collection_actions=_collection_actions(
                        references[rule]
                    ).astype(np.float32),
                    expected_hashes=_expected_hashes(references[rule]),
                    pair_metrics=pair_metrics,
                    agent_speed=float(config["protocol"]["agent_speed"]),
                )
            )
    return plans


def _speed_slug(speed: float) -> str:
    return f"{float(speed):05.2f}".replace(".", "p")


def _composition_episode_plans_for_door(
    config: dict[str, Any],
    *,
    scale: str,
    door_position: int,
) -> dict[str, list[HiddenPassageEpisodePlan]]:
    protocol = config["protocol"]
    speeds = tuple(map(float, protocol["agent_speeds"]))
    if not speeds or tuple(sorted(speeds)) != speeds:
        raise ValueError(
            "Composition agent_speeds must be non-empty and increasing"
        )
    plans = {rule: [] for rule in RULE_NAMES}
    for base_template in templates_for_door(
        config,
        scale=scale,
        door_position=door_position,
    ):
        references = {
            (speed, rule): simulate_speed_door_rule_template(
                base_template,
                speed=speed,
                rule=rule,
                protocol=protocol,
            )
            for speed in speeds
            for rule in RULE_NAMES
        }
        if protocol["task"] == "speed_door_rule_composition_v2":
            validation = validate_v2_training_factor_grid(
                base_template,
                references,
                speeds=speeds,
                thresholds=config["gates"],
            )
        else:
            validation = validate_speed_door_rule_factor_grid(
                base_template,
                references,
                speeds=speeds,
                thresholds=config["gates"],
            )
        if not validation["passed"]:
            failed = sorted(
                key
                for key, passed in validation["checks"].items()
                if not passed
            )
            raise RuntimeError(
                "Speed-door-rule factor grid failed for "
                f"{base_template.template_id}: {failed}"
            )
        pair_metrics = {
            "minimum_middle_rule_centroid_gap_px": validation[
                "minimum_middle_rule_centroid_gap_px"
            ],
            "minimum_middle_adjacent_speed_centroid_gap_px": validation[
                "minimum_middle_adjacent_speed_centroid_gap_px"
            ],
            "minimum_future_rule_state_gap_px": validation[
                "minimum_future_rule_state_gap_px"
            ],
            "minimum_future_adjacent_speed_centroid_gap_px": validation[
                "minimum_future_adjacent_speed_centroid_gap_px"
            ],
            "query_state": validation["query_state"],
        }
        for speed_index, speed in enumerate(speeds):
            template = replace(
                base_template,
                template_id=(
                    f"{base_template.template_id}-"
                    f"s{speed_index:02d}v{_speed_slug(speed)}"
                ),
            )
            for rule in RULE_NAMES:
                reference = references[(speed, rule)]
                plans[rule].append(
                    HiddenPassageEpisodePlan(
                        template=template,
                        rule=rule,
                        collection_actions=_collection_actions(
                            reference
                        ).astype(np.float32),
                        expected_hashes=_expected_hashes(reference),
                        pair_metrics=pair_metrics,
                        agent_speed=speed,
                    )
                )
    return plans


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
            raise RuntimeError("Hidden-passage policy is not ready")
        if self.step >= RAW_STEPS:
            raise RuntimeError("World requested more than 20 actions")
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


def _variation_values(
    template: HiddenPassageTemplate,
    *,
    rule: str,
    config: dict[str, Any],
    agent_speed: float | None = None,
) -> dict[str, Any]:
    speed = (
        float(config["protocol"]["agent_speed"])
        if agent_speed is None
        else float(agent_speed)
    )
    return {
        "agent.speed": np.asarray(
            [speed],
            dtype=np.float32,
        ),
        "door.number": int(config["protocol"]["door_number"]),
        "door.position": np.asarray(
            [template.door_position] * 3,
            dtype=np.int64,
        ),
        PASSAGE_FACTOR: PASSAGE_RULES[rule],
    }


def _episode_iterator(
    swm: Any,
    *,
    plans: list[HiddenPassageEpisodePlan],
    config: dict[str, Any],
) -> Iterator[dict[str, Any]]:
    register_hidden_passage_env()
    world = swm.World(
        HIDDEN_PASSAGE_ENV_ID,
        num_envs=1,
        max_episode_steps=RAW_STEPS,
        image_shape=(224, 224),
        render_mode="rgb_array",
    )
    policy = _ResettableScriptedPolicy()
    world.set_policy(policy)
    try:
        for plan in plans:
            policy.reset_actions(plan.collection_actions)
            capture = _OneEpisodeCapture()
            world.collect(
                episodes=1,
                seed=int(plan.template.simulator_seed),
                options={
                    "variation": WATCHED_VARIATIONS,
                    "variation_values": _variation_values(
                        plan.template,
                        rule=plan.rule,
                        config=config,
                        agent_speed=plan.agent_speed,
                    ),
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
                    f"{plan.template_id}: World used {policy.step} actions"
                )
            yield capture.one()
    finally:
        world.close()


def collect_hidden_passage_shard(
    swm: Any,
    *,
    plans: list[HiddenPassageEpisodePlan],
    table_path: Path,
    config: dict[str, Any],
) -> None:
    if not plans:
        raise ValueError("Cannot collect an empty hidden-passage shard")
    if table_path.exists():
        raise FileExistsError(table_path)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(config["collection"]["staging_root"]).expanduser()
    staging_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="contextworld-h3-data-",
        dir=staging_root,
    ) as temporary:
        staged = Path(temporary) / table_path.name
        writer = build_lance_writer(
            swm,
            staged,
            pixel_codec=dict(config["storage"]["pixel_codec"]),
        )
        with writer as opened:
            opened.write_episodes(
                _episode_iterator(swm, plans=plans, config=config)
            )
        # Some shared filesystems reject rename(2) for non-empty
        # directories. Copy directly into the final planned directory and
        # leave it deliberately incomplete until the main process finishes
        # the full logical audit and writes the sibling completion marker.
        # A crash can therefore leave bytes behind, but never a shard that a
        # resume or training process is allowed to treat as complete.
        shutil.copytree(staged, table_path)
        _fsync_tree(table_path)


def _initialize_shard_worker(
    repo_root: Path,
    config: dict[str, Any],
    stable_worldmodel_commit: str,
) -> None:
    global _SHARD_WORKER_CONFIG, _SHARD_WORKER_SWM
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[variable] = "1"
    swm, _, observed_commit = load_stable_worldmodel(
        repo_root,
        str(config["stable_worldmodel"]["repo"]),
        stable_worldmodel_commit,
    )
    try:
        import torch

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except (ImportError, RuntimeError):
        pass
    if observed_commit != stable_worldmodel_commit:
        raise RuntimeError(
            "Shard worker loaded the wrong Stable-WorldModel commit: "
            f"expected {stable_worldmodel_commit}, got {observed_commit}"
        )
    _SHARD_WORKER_SWM = swm
    _SHARD_WORKER_CONFIG = config


def _collect_shard_worker(job: HiddenPassageShardCollectionJob) -> str:
    if _SHARD_WORKER_SWM is None or _SHARD_WORKER_CONFIG is None:
        raise RuntimeError("Hidden-passage shard worker was not initialized")
    collect_hidden_passage_shard(
        _SHARD_WORKER_SWM,
        plans=list(job.plans),
        table_path=job.shard.table_path,
        config=_SHARD_WORKER_CONFIG,
    )
    return job.shard.scenario_id


def _collect_missing_shards(
    swm: Any,
    *,
    jobs: list[HiddenPassageShardCollectionJob],
    config: dict[str, Any],
    repo_root: Path,
    stable_worldmodel_commit: str,
    workers: int,
) -> None:
    if not jobs:
        return
    if workers == 1:
        for job in jobs:
            print(
                f"[h3-data] collect {job.shard.scenario_id} "
                f"episodes={len(job.plans)}",
                flush=True,
            )
            collect_hidden_passage_shard(
                swm,
                plans=list(job.plans),
                table_path=job.shard.table_path,
                config=config,
            )
        return

    print(
        f"[h3-data] collect {len(jobs)} shards with {workers} workers",
        flush=True,
    )
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=context,
        initializer=_initialize_shard_worker,
        initargs=(
            repo_root,
            config,
            stable_worldmodel_commit,
        ),
    ) as executor:
        for job, scenario_id in zip(
            jobs,
            executor.map(_collect_shard_worker, jobs),
            strict=True,
        ):
            if scenario_id != job.shard.scenario_id:
                raise RuntimeError(
                    "Shard worker returned an unexpected scenario: "
                    f"expected {job.shard.scenario_id}, got {scenario_id}"
                )
            print(
                f"[h3-data] collected {scenario_id}",
                flush=True,
            )


def _tensor_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def logical_episode_content_hashes(
    episode: dict[str, Any],
) -> dict[str, str]:
    missing = [
        column for column in LOGICAL_CONTENT_COLUMNS if column not in episode
    ]
    if missing:
        raise ValueError(
            f"Hidden-passage episode is missing logical columns: {missing}"
        )
    return {
        f"raw_{column}_sha256": array_sha256(
            _tensor_numpy(episode[column])
        )
        for column in LOGICAL_CONTENT_COLUMNS
    }


def logical_shard_content_sha256(
    episode_rows: Iterable[dict[str, Any]],
) -> str:
    logical_rows = []
    for row in episode_rows:
        hashes = {
            key: value
            for key, value in sorted(row.items())
            if key.startswith("raw_") and key.endswith("_sha256")
        }
        expected_keys = {
            f"raw_{column}_sha256"
            for column in LOGICAL_CONTENT_COLUMNS
        }
        if set(hashes) != expected_keys:
            raise ValueError(
                "Hidden-passage logical row hash fields differ: "
                f"missing={sorted(expected_keys - set(hashes))}, "
                f"extra={sorted(set(hashes) - expected_keys)}"
            )
        logical_rows.append(
            {
                "episode_index": int(row["episode_index"]),
                "template_id": str(row["template_id"]),
                "rule": str(row["rule"]),
                "hashes": hashes,
            }
        )
    if not logical_rows:
        raise ValueError("Cannot hash an empty hidden-passage shard")
    return canonical_sha256(logical_rows)


def _decoded_model_pixels(sample: dict[str, Any]) -> np.ndarray:
    return np.transpose(
        _tensor_numpy(sample["pixels"]),
        (0, 2, 3, 1),
    ).astype(np.uint8)


def _allclose_constant(value: Any, expected: Any) -> bool:
    array = _tensor_numpy(value)
    rows = np.asarray(array).reshape(len(array), -1)
    target = np.asarray(expected).reshape(-1)
    return bool(
        rows.shape[1] == target.size
        and np.allclose(rows, target[None, :], atol=0.0, rtol=0.0)
    )


def audit_hidden_passage_shard(
    swm: Any,
    *,
    shard: HiddenPassageShardPlan,
    plans: list[HiddenPassageEpisodePlan],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raw = swm.data.LanceDataset(path=shard.table_path)
    strict = swm.data.LanceDataset(
        path=shard.table_path,
        keys_to_load=list(MODEL_KEYS),
        frameskip=5,
        num_steps=MODEL_STEPS,
    )
    diagnostic = swm.data.LanceDataset(
        path=shard.table_path,
        keys_to_load=list(DIAGNOSTIC_KEYS),
        frameskip=5,
        num_steps=MODEL_STEPS,
    )
    columns = set(raw.column_names)
    checks = {
        "required_raw_columns_present": REQUIRED_COLUMNS <= columns,
        "template_not_stored_in_lance": not any(
            "template" in column for column in columns
        ),
        "episode_count_exact": len(raw.lengths) == len(plans),
        "every_episode_has_20_rows": all(
            int(value) == RAW_STEPS for value in raw.lengths
        ),
        "one_model_clip_per_episode": len(strict) == len(plans),
        "one_diagnostic_clip_per_episode": len(diagnostic) == len(plans),
    }
    failures: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for index, plan in enumerate(plans):
        model_sample = strict[index]
        diagnostic_sample = diagnostic[index]
        episode = raw.load_episode(index)
        model_pixels = _decoded_model_pixels(model_sample)
        model_actions = _tensor_numpy(model_sample["action"]).reshape(
            MODEL_STEPS,
            5,
            2,
        ).astype(np.float32)
        model_proprio = _tensor_numpy(
            diagnostic_sample["proprio"]
        ).astype(np.float32)
        observed_hashes = {
            "model_pixels": array_sha256(model_pixels),
            "model_actions": array_sha256(model_actions),
            "model_proprio": array_sha256(model_proprio),
            "initial_pixels": array_sha256(model_pixels[0]),
            "middle_pixels": array_sha256(model_pixels[1]),
            "query_pixels": array_sha256(model_pixels[2]),
            "future_pixels": array_sha256(model_pixels[3]),
        }
        terminated = _tensor_numpy(
            episode["terminated"]
        ).reshape(-1).astype(bool)
        truncated = _tensor_numpy(
            episode["truncated"]
        ).reshape(-1).astype(bool)
        raw_actions = _tensor_numpy(episode["action"]).astype(np.float32)
        raw_content_hashes = logical_episode_content_hashes(episode)
        episode_checks = {
            "model_keys_exact": tuple(model_sample) == MODEL_KEYS,
            "diagnostic_keys_exact": (
                tuple(diagnostic_sample) == DIAGNOSTIC_KEYS
            ),
            "model_has_no_privileged_fields": not any(
                key.startswith("variation")
                or "template" in key
                or key in {"state", "proprio"}
                for key in model_sample
            ),
            "model_pixels_shape": tuple(model_sample["pixels"].shape)
            == (4, 3, 224, 224),
            "model_actions_shape": tuple(model_sample["action"].shape)
            == (4, 10),
            "model_proprio_shape": tuple(
                diagnostic_sample["proprio"].shape
            )
            == (4, 2),
            "independent_simulation_hashes_exact": all(
                observed_hashes[key] == expected
                for key, expected in plan.expected_hashes.items()
                if key in observed_hashes
            ),
            "raw_actions_exact": np.array_equal(
                raw_actions,
                model_actions.reshape(RAW_STEPS, 2),
            ),
            "stored_rule_constant": _allclose_constant(
                episode["variation_passage_open"],
                PASSAGE_RULES[plan.rule],
            ),
            "stored_speed_constant": _allclose_constant(
                episode["variation_agent_speed"],
                np.asarray([plan.agent_speed], dtype=np.float32),
            ),
            "stored_door_number_constant": _allclose_constant(
                episode["variation_door_number"],
                [1],
            ),
            "stored_door_position_constant": _allclose_constant(
                episode["variation_door_position"],
                [plan.template.door_position] * 3,
            ),
            "goal_state_constant": _allclose_constant(
                episode["goal_state"],
                plan.template.goal_state,
            ),
            "never_terminated": bool(not terminated.any()),
            "only_last_row_truncated": bool(
                len(truncated) == RAW_STEPS
                and not truncated[:-1].any()
                and truncated[-1]
            ),
        }
        if not all(episode_checks.values()):
            failures.append(
                {
                    "template_id": plan.template_id,
                    "failed_checks": sorted(
                        key
                        for key, passed in episode_checks.items()
                        if not passed
                    ),
                }
            )
        rows.append(
            {
                "schema_version": 1,
                "episode_index": index,
                "template_id": plan.template_id,
                "pair_id": plan.template_id,
                "rule": plan.rule,
                "passage_open": PASSAGE_RULES[plan.rule],
                "agent_speed": float(plan.agent_speed),
                "direction": plan.template.direction,
                "door_position": plan.template.door_position,
                "doorway_offset_px": plan.template.doorway_offset_px,
                "wall_distance_px": (
                    99.5 - plan.template.reset_state[0]
                    if plan.template.direction == "left_to_right"
                    else plan.template.reset_state[0] - 124.5
                ),
                "reset_state": list(plan.template.reset_state),
                "goal_state": list(plan.template.goal_state),
                "simulator_seed": plan.template.simulator_seed,
                "rows": RAW_STEPS,
                "model_clips": 1,
                "model_input_keys": list(MODEL_KEYS),
                "action_sha256": observed_hashes["model_actions"],
                "initial_pixels_sha256": observed_hashes[
                    "initial_pixels"
                ],
                "middle_pixels_sha256": observed_hashes["middle_pixels"],
                "query_pixels_sha256": observed_hashes["query_pixels"],
                "future_pixels_sha256": observed_hashes["future_pixels"],
                "goal_pixels_sha256": plan.expected_hashes["goal_pixels"],
                "model_pixels_sha256": observed_hashes["model_pixels"],
                "model_proprio_sha256": observed_hashes[
                    "model_proprio"
                ],
                **raw_content_hashes,
                "pair_metrics": plan.pair_metrics,
                "passed": all(episode_checks.values()),
            }
        )
    checks["every_episode_passed"] = not failures
    return (
        {
            "passed": all(checks.values()),
            "checks": checks,
            "scenario_id": shard.scenario_id,
            "split": shard.split,
            "door_position": shard.door_position,
            "rule": shard.rule,
            "episode_count": len(plans),
            "clip_count": len(strict),
            "raw_rows": int(sum(map(int, raw.lengths))),
            "model_loader_keys": list(MODEL_KEYS),
            "diagnostic_loader_keys": list(DIAGNOSTIC_KEYS),
            "raw_columns": sorted(columns),
            "failure_count": len(failures),
            "failures": failures[:20],
        },
        rows,
    )


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(
                    json.dumps(record, sort_keys=True, separators=(",", ":"))
                )
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_path(path.parent)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _shard_plan(
    *,
    output_root: Path,
    scale: str,
    split: str,
    door_position: int,
    rule: str,
    templates: list[HiddenPassageTemplate],
    stable_worldmodel_commit: str,
    pixel_codec: dict[str, Any],
    protocol: dict[str, Any],
) -> HiddenPassageShardPlan:
    pair_id = f"hp-{split}-door{door_position:03d}"
    fingerprint = canonical_sha256(
        {
            "schema_version": 1,
            "scale": scale,
            "split": split,
            "door_position": door_position,
            "rule": rule,
            "templates": [asdict(value) for value in templates],
            "stable_worldmodel_commit": stable_worldmodel_commit,
            "pixel_codec": pixel_codec,
            "protocol": protocol,
            "rows_per_episode": RAW_STEPS,
        }
    )
    scenario_id = (
        f"hp-{split}-d{door_position:03d}-{rule}-{fingerprint[:10]}"
    )
    return HiddenPassageShardPlan(
        split=split,
        door_position=door_position,
        rule=rule,
        pair_id=pair_id,
        fingerprint=fingerprint,
        scenario_id=scenario_id,
        table_path=output_root / "tables" / split / f"{scenario_id}.lance",
        episode_manifest_path=(
            output_root
            / "episode_manifests"
            / split
            / f"{scenario_id}.jsonl"
        ),
    )


def _planned_shard_assets(shard: HiddenPassageShardPlan) -> tuple[Path, ...]:
    return (
        shard_completion_marker_path(shard.table_path),
        shard.episode_manifest_path,
        shard.table_path,
    )


def audit_hidden_passage_release_assets(
    *,
    release_root: Path,
    expected_tables: Iterable[Path],
    expected_markers: Iterable[Path],
    expected_sidecars: Iterable[Path],
) -> dict[str, Any]:
    """Reject unsafe nodes and require an exact physical release tree."""

    root = require_safe_directory(release_root)
    expected = {
        "tables": {
            require_lexical_containment(value, root)
            for value in expected_tables
        },
        "markers": {
            require_lexical_containment(value, root)
            for value in expected_markers
        },
        "sidecars": {
            require_lexical_containment(value, root)
            for value in expected_sidecars
        },
    }
    observed: dict[str, set[Path]] = {
        "tables": set(),
        "markers": set(),
        "sidecars": set(),
    }

    def visit(directory: Path, *, section: str) -> None:
        require_safe_directory(directory, containment_root=root)
        with os.scandir(directory) as iterator:
            entries = sorted(iterator, key=lambda value: value.name)
        for entry in entries:
            value = require_lexical_containment(
                directory / entry.name,
                root,
            )
            metadata = entry.stat(follow_symlinks=False)
            kind = _node_kind(metadata.st_mode)
            if kind == "directory":
                if section == "tables" and value.name.endswith(".lance"):
                    observed["tables"].add(value)
                visit(value, section=section)
            elif kind == "regular_file":
                if (
                    section == "tables"
                    and value.name.endswith(SHARD_COMPLETION_SUFFIX)
                ):
                    observed["markers"].add(value)
                elif section == "sidecars" and value.name.endswith(".jsonl"):
                    observed["sidecars"].add(value)
            else:
                raise ValueError(
                    "Hidden-passage release tree contains an unsafe node "
                    "before alias resolution: "
                    f"path={value}, kind={kind}"
                )

    tables_root = root / "tables"
    sidecars_root = root / "episode_manifests"
    visit(tables_root, section="tables")
    visit(sidecars_root, section="sidecars")

    differences: dict[str, dict[str, list[str]]] = {}
    for name in ("tables", "markers", "sidecars"):
        extras = sorted(str(value) for value in observed[name] - expected[name])
        missing = sorted(str(value) for value in expected[name] - observed[name])
        if extras or missing:
            differences[name] = {
                "extra": extras,
                "missing": missing,
            }
    if differences:
        raise ValueError(
            "Hidden-passage release assets differ from the sealed catalogs: "
            f"{differences}"
        )
    return {
        "release_root": str(root),
        "tables": len(observed["tables"]),
        "markers": len(observed["markers"]),
        "sidecars": len(observed["sidecars"]),
        "lexical_paths_checked_before_resolution": True,
        "unsafe_nodes_rejected": True,
        "passed": True,
    }


def _path_exists_or_is_symlink(path: Path) -> bool:
    try:
        os.lstat(path)
    except FileNotFoundError:
        return False
    return True


def _remove_incomplete_planned_shard(
    shard: HiddenPassageShardPlan,
) -> None:
    """Remove only the three exact paths derived from the current plan."""

    release_root = lexical_absolute_path(shard.table_path.parents[2])
    require_safe_directory(release_root)
    marker_path, episode_manifest_path, table_path = (
        _planned_shard_assets(shard)
    )
    for path in (marker_path, episode_manifest_path, table_path):
        require_lexical_containment(path, release_root)
    for path in (marker_path, episode_manifest_path):
        if not _path_exists_or_is_symlink(path):
            continue
        require_safe_regular_file(path, containment_root=release_root)
        if path.is_file():
            path.unlink()
    if not _path_exists_or_is_symlink(table_path):
        return
    require_safe_directory(table_path, containment_root=release_root)
    _regular_tree_files(table_path)
    if table_path.is_dir():
        shutil.rmtree(table_path)


def _audit_reusable_hidden_passage_shard(
    swm: Any,
    *,
    job: HiddenPassageShardCollectionJob,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    shard = job.shard
    completion = verify_hidden_passage_shard_completion(
        table_path=shard.table_path,
        episode_manifest_path=shard.episode_manifest_path,
        expected_scenario_id=shard.scenario_id,
        expected_fingerprint=shard.fingerprint,
    )
    audit, episode_rows = audit_hidden_passage_shard(
        swm,
        shard=shard,
        plans=list(job.plans),
    )
    if not audit["passed"]:
        raise ValueError(
            f"Reusable shard loader audit failed: {shard.scenario_id}"
        )
    observed_content_sha256 = logical_shard_content_sha256(episode_rows)
    if observed_content_sha256 != completion["content_sha256"]:
        raise ValueError(
            "Reusable shard logical content differs from its completion "
            f"marker: {shard.scenario_id}"
        )
    existing_episode_rows = _episode_manifest_rows(
        shard.episode_manifest_path
    )
    if existing_episode_rows != episode_rows:
        raise ValueError(
            "Reusable shard episode manifest differs from the current full "
            f"audit: {shard.scenario_id}"
        )
    return audit, episode_rows, completion


def _scenario_manifest_record(
    *,
    shard: HiddenPassageShardPlan,
    shard_audit: dict[str, Any],
    episode_rows: list[dict[str, Any]],
    completion: dict[str, Any],
    collection_status: str,
    stable_worldmodel_commit: str,
    pixel_codec: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    regime = {
        "train": "train_hidden_passage_history3",
        "val": "validation_hidden_passage_history3",
        "test": "test_hidden_passage_history3",
    }[shard.split]
    return {
        "schema_version": 1,
        "scenario_id": shard.scenario_id,
        "fingerprint": shard.fingerprint,
        "split": shard.split,
        "regime": regime,
        "seed_group": shard.pair_id,
        "pair_id": shard.pair_id,
        "factors": {
            PASSAGE_FACTOR: PASSAGE_RULES[shard.rule],
            "door.position": [shard.door_position] * 3,
            "agent.speed": sorted(
                {
                    float(row["agent_speed"])
                    for row in episode_rows
                }
            ),
        },
        "rule": shard.rule,
        "output_path": portable_contextworld_path(
            shard.table_path,
            repo_root=repo_root,
        ),
        "collection_status": collection_status,
        "stable_worldmodel_commit": stable_worldmodel_commit,
        "pixel_codec": pixel_codec,
        "episode_count": shard_audit["episode_count"],
        "clip_count": shard_audit["clip_count"],
        "rows_per_episode": RAW_STEPS,
        "raw_rows": shard_audit["raw_rows"],
        "content_sha256_kind": LOGICAL_CONTENT_HASH_KIND,
        "content_sha256": logical_shard_content_sha256(episode_rows),
        "storage_sha256_kind": STORAGE_CONTENT_HASH_KIND,
        "storage_sha256": completion["storage_sha256"],
        "episode_manifest": portable_contextworld_path(
            shard.episode_manifest_path,
            repo_root=repo_root,
        ),
        "episode_manifest_sha256": file_sha256(
            shard.episode_manifest_path
        ),
        "completion_protocol": SHARD_COMPLETION_PROTOCOL,
        "completion_marker": portable_contextworld_path(
            shard_completion_marker_path(shard.table_path),
            repo_root=repo_root,
        ),
        "completion_marker_sha256": completion["sha256"],
        "passed": shard_audit["passed"],
    }


def _pair_and_split_audit(
    *,
    episode_rows: list[dict[str, Any]],
    door_splits: HiddenPassageDoorSplits,
) -> dict[str, Any]:
    by_template: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in episode_rows:
        by_template[row["template_id"]][row["rule"]] = row
    pair_failures = []
    action_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for template_id, rules in sorted(by_template.items()):
        checks = {
            "both_rules_present": set(rules) == set(RULE_NAMES),
        }
        if checks["both_rules_present"]:
            passable = rules["passable"]
            blocked = rules["blocked"]
            checks.update(
                {
                    "actions_identical": (
                        passable["action_sha256"]
                        == blocked["action_sha256"]
                    ),
                    "initial_pixels_identical": (
                        passable["initial_pixels_sha256"]
                        == blocked["initial_pixels_sha256"]
                    ),
                    "query_pixels_identical": (
                        passable["query_pixels_sha256"]
                        == blocked["query_pixels_sha256"]
                    ),
                    "goal_pixels_identical": (
                        passable["goal_pixels_sha256"]
                        == blocked["goal_pixels_sha256"]
                    ),
                    "middle_pixels_different": (
                        passable["middle_pixels_sha256"]
                        != blocked["middle_pixels_sha256"]
                    ),
                    "future_pixels_different": (
                        passable["future_pixels_sha256"]
                        != blocked["future_pixels_sha256"]
                    ),
                }
            )
            for rule in RULE_NAMES:
                action_counts[rules[rule]["action_sha256"]][rule] += 1
        if not all(checks.values()):
            pair_failures.append(
                {
                    "template_id": template_id,
                    "failed_checks": sorted(
                        key
                        for key, passed in checks.items()
                        if not passed
                    ),
                }
            )

    total = sum(sum(values.values()) for values in action_counts.values())
    majority = sum(max(values.values()) for values in action_counts.values())
    action_accuracy = float(majority / total) if total else 1.0
    action_balanced = all(
        set(values) == set(RULE_NAMES)
        and values["passable"] == values["blocked"]
        for values in action_counts.values()
    )
    split_sets = {
        "train": set(door_splits.train),
        "loader_val": set(door_splits.val),
        "final_eval_candidates": set(door_splits.test),
        "guard": set(door_splits.guard),
    }
    split_overlaps: dict[str, list[int]] = {}
    items = list(split_sets.items())
    for index, (left_name, left) in enumerate(items):
        for right_name, right in items[index + 1 :]:
            overlap = sorted(left & right)
            if overlap:
                split_overlaps[f"{left_name}__{right_name}"] = overlap
    checks = {
        "all_templates_have_exact_rule_pair": not pair_failures,
        "action_signature_balanced_across_rules": action_balanced,
        "action_signature_only_accuracy_is_chance": action_accuracy == 0.5,
        "door_splits_have_zero_overlap": not split_overlaps,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "paired_templates": len(by_template),
        "pair_failure_count": len(pair_failures),
        "pair_failures": pair_failures[:20],
        "action_signatures": len(action_counts),
        "best_action_signature_only_rule_accuracy": action_accuracy,
        "door_splits": door_splits.as_report(),
        "door_split_overlaps": split_overlaps,
    }


def _catalog_stem(config: dict[str, Any], group: str) -> str:
    prefix = str(config["artifact_names"]["catalog_prefix"])
    version = str(config["artifact_names"]["version"])
    return f"{prefix}_{GROUP_CATALOG_SUFFIX[group]}_{version}"


def _catalog_payload(
    *,
    group: str,
    records: list[dict[str, Any]],
    config: dict[str, Any],
    repo_root: Path,
    scale: str,
) -> dict[str, Any]:
    rules = set(GROUP_RULES[group])
    selected = [record for record in records if record["rule"] in rules]
    paths_by_split = {
        split: [
            record["output_path"]
            for record in selected
            if record["split"] == split
        ]
        for split in SPLITS
    }
    return {
        "schema_version": 1,
        "benchmark": config["benchmark"],
        "scale": scale,
        "group": group,
        "mixing": "logical_concat_at_load_time",
        "sampling_recommendation": "concatenated_raw_clips",
        "sampling_contract": {
            "synthetic_only": True,
            "original_samples_included": False,
            "original_dataset_is_normalization_reference_only": True,
        },
        "pixel_codec": dict(config["storage"]["pixel_codec"]),
        "model_columns": list(MODEL_KEYS),
        "raw_privileged_columns_excluded_from_model": [
            "variation_passage_open",
            "variation_door_position",
            "variation_agent_speed",
            "variation_door_number",
        ],
        "original_dataset_read_only": portable_contextworld_path(
            resolve_contextworld_path(
                config["original_dataset"]["path"],
                repo_root=repo_root,
            ),
            repo_root=repo_root,
        ),
        "train": {
            "original": [],
            "synthetic": paths_by_split["train"],
        },
        "val": {"synthetic": paths_by_split["val"]},
        "ood_test": {"synthetic": paths_by_split["test"]},
        "by_regime": {
            "train_hidden_passage_history3": paths_by_split["train"],
            "validation_hidden_passage_history3": paths_by_split["val"],
            "test_hidden_passage_history3": paths_by_split["test"],
        },
        "rule_support": {
            "names": list(GROUP_RULES[group]),
            "passage_open_values": [
                PASSAGE_RULES[rule] for rule in GROUP_RULES[group]
            ],
        },
        "speed_support": {
            split: sorted(
                {
                    float(speed)
                    for record in selected
                    if record["split"] == split
                    for speed in record["factors"]["agent.speed"]
                }
            )
            for split in SPLITS
        },
        "counts": {
            split: {
                "shards": sum(
                    record["split"] == split for record in selected
                ),
                "episodes": sum(
                    int(record["episode_count"])
                    for record in selected
                    if record["split"] == split
                ),
                "clips": sum(
                    int(record["clip_count"])
                    for record in selected
                    if record["split"] == split
                ),
            }
            for split in SPLITS
        },
    }


def write_group_artifacts(
    *,
    output_root: Path,
    config: dict[str, Any],
    scale: str,
    records: list[dict[str, Any]],
    shard_audits: dict[str, dict[str, Any]],
    stable_worldmodel_commit: str,
    repo_root: Path,
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for group, rules in GROUP_RULES.items():
        stem = _catalog_stem(config, group)
        catalog_path = output_root / "catalogs" / f"{stem}.json"
        manifest_path = output_root / "manifests" / f"{stem}.jsonl"
        report_path = output_root / "reports" / f"{stem}.json"
        selected = [
            record for record in records if record["rule"] in set(rules)
        ]
        _write_jsonl(manifest_path, selected)
        write_json(
            catalog_path,
            _catalog_payload(
                group=group,
                records=records,
                config=config,
                repo_root=repo_root,
                scale=scale,
            ),
        )
        scenarios = [
            {
                "scenario_id": record["scenario_id"],
                "passed": shard_audits[record["scenario_id"]]["passed"],
                "split": record["split"],
                "rule": record["rule"],
                "episode_count": record["episode_count"],
                "clip_count": record["clip_count"],
            }
            for record in selected
        ]
        collection_status = {
            record["scenario_id"]: record["collection_status"]
            for record in selected
        }
        loader_compatibility = {
            "passed": all(row["passed"] for row in scenarios),
            "frameskip": 5,
            "num_steps": MODEL_STEPS,
            "model_loader_keys": list(MODEL_KEYS),
            "diagnostic_loader_keys": list(DIAGNOSTIC_KEYS),
            "episodes_equal_clips": all(
                row["episode_count"] == row["clip_count"]
                for row in scenarios
            ),
            "total_episodes": sum(
                row["episode_count"] for row in scenarios
            ),
            "total_clips": sum(row["clip_count"] for row in scenarios),
        }
        report = {
            "schema_version": 1,
            "benchmark": config["benchmark"],
            "scale": scale,
            "group": group,
            "passed": (
                bool(scenarios)
                and all(row["passed"] for row in scenarios)
                and loader_compatibility["passed"]
            ),
            "compile_only": False,
            "preflight_passed": True,
            "catalog": str(catalog_path.resolve()),
            "manifest": str(manifest_path.resolve()),
            "stable_worldmodel_commit": stable_worldmodel_commit,
            "loader_compatibility": loader_compatibility,
            "scenarios": scenarios,
            "collection_status": collection_status,
            "claim_limit": (
                "Training data construction and loader compatibility only; "
                "this report is not an ICL model result"
            ),
        }
        write_json(report_path, report)
        output[group] = {
            "catalog": portable_contextworld_path(
                catalog_path,
                repo_root=repo_root,
            ),
            "catalog_absolute": str(catalog_path.resolve()),
            "catalog_sha256": file_sha256(catalog_path),
            "manifest": portable_contextworld_path(
                manifest_path,
                repo_root=repo_root,
            ),
            "manifest_absolute": str(manifest_path.resolve()),
            "manifest_sha256": file_sha256(manifest_path),
            "synthesis_report": portable_contextworld_path(
                report_path,
                repo_root=repo_root,
            ),
            "synthesis_report_absolute": str(report_path.resolve()),
            "synthesis_report_sha256": file_sha256(report_path),
            "counts": _catalog_payload(
                group=group,
                records=records,
                config=config,
                repo_root=repo_root,
                scale=scale,
            )["counts"],
        }
    return output


def _group_union_audit(
    *,
    output_root: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    catalogs = {}
    for group in GROUP_RULES:
        path = (
            output_root
            / "catalogs"
            / f"{_catalog_stem(config, group)}.json"
        )
        catalogs[group] = json.loads(path.read_text(encoding="utf-8"))

    def paths(group: str, split: str) -> set[str]:
        section = {
            "train": "train",
            "val": "val",
            "test": "ood_test",
        }[split]
        return set(catalogs[group][section]["synthetic"])

    per_split = {}
    for split in SPLITS:
        passable = paths("passage_passable", split)
        blocked = paths("passage_blocked", split)
        mixed = paths("passage_mixed", split)
        checks = {
            "single_rule_paths_disjoint": not (passable & blocked),
            "mixed_is_exact_union": mixed == passable | blocked,
            "single_rule_shard_counts_equal": len(passable) == len(blocked),
            "mixed_shard_count_is_double": (
                len(mixed) == 2 * len(passable)
            ),
        }
        per_split[split] = {
            "passed": all(checks.values()),
            "checks": checks,
            "passable_shards": len(passable),
            "blocked_shards": len(blocked),
            "mixed_shards": len(mixed),
        }
    return {
        "passed": all(value["passed"] for value in per_split.values()),
        "splits": per_split,
    }


def _catalog_count_audit(
    *,
    config: dict[str, Any],
    scale: str,
    group_artifacts: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    expected_single = config["scales"][scale][
        "expected_single_rule_catalog_counts"
    ]
    results = {}
    for group in GROUP_RULES:
        multiplier = len(GROUP_RULES[group])
        mismatches = {}
        for split in SPLITS:
            observed = group_artifacts[group]["counts"][split]
            expected = expected_single[split]
            for field in ("shards", "clips"):
                target = int(expected[field]) * multiplier
                if int(observed[field]) != target:
                    mismatches[f"{split}.{field}"] = {
                        "expected": target,
                        "observed": int(observed[field]),
                    }
            if int(observed["episodes"]) != int(observed["clips"]):
                mismatches[f"{split}.episodes_equal_clips"] = {
                    "expected": int(observed["clips"]),
                    "observed": int(observed["episodes"]),
                }
        results[group] = {
            "passed": not mismatches,
            "rule_multiplier": multiplier,
            "counts": group_artifacts[group]["counts"],
            "mismatches": mismatches,
        }
    return {
        "passed": all(value["passed"] for value in results.values()),
        "groups": results,
    }


def build_hidden_passage_h3_data(
    swm: Any,
    *,
    config: dict[str, Any],
    scale: str,
    output_root: Path,
    repo_root: Path,
    stable_worldmodel_commit: str,
    resume_partial: bool = False,
    workers: int = 1,
) -> dict[str, Any]:
    workers = int(workers)
    if workers < 1:
        raise ValueError(f"workers must be at least 1, got {workers}")
    door_splits = door_splits_for_scale(config, scale)
    validation_exclusion, frozen_query_hashes = audit_validation_exclusion(
        config,
        scale=scale,
        door_splits=door_splits,
        repo_root=repo_root,
    )
    split_doors = {
        "train": door_splits.train,
        "val": door_splits.val,
        "test": door_splits.test,
    }
    pixel_codec = dict(config["storage"]["pixel_codec"])
    records: list[dict[str, Any]] = []
    shard_audits: dict[str, dict[str, Any]] = {}
    all_episode_rows: list[dict[str, Any]] = []
    ordered_jobs: list[HiddenPassageShardCollectionJob] = []
    for split in COLLECTED_SPLITS:
        for door_position in split_doors[split]:
            print(
                f"[h3-data] prepare {scale} {split} door={door_position}",
                flush=True,
            )
            by_rule = episode_plans_for_door(
                config,
                scale=scale,
                door_position=door_position,
            )
            templates = [value.template for value in by_rule["passable"]]
            for rule in RULE_NAMES:
                shard = _shard_plan(
                    output_root=output_root,
                    scale=scale,
                    split=split,
                    door_position=door_position,
                    rule=rule,
                    templates=templates,
                    stable_worldmodel_commit=stable_worldmodel_commit,
                    pixel_codec=pixel_codec,
                    protocol=dict(config["protocol"]),
                )
                ordered_jobs.append(
                    HiddenPassageShardCollectionJob(
                        shard=shard,
                        plans=tuple(by_rule[rule]),
                    )
                )

    missing_jobs: list[HiddenPassageShardCollectionJob] = []
    reusable: dict[
        str,
        tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]],
    ] = {}
    for job in ordered_jobs:
        existing_assets = [
            path
            for path in _planned_shard_assets(job.shard)
            if _path_exists_or_is_symlink(path)
        ]
        if existing_assets:
            if not resume_partial:
                raise FileExistsError(
                    "Refusing to overwrite existing planned shard assets: "
                    f"{existing_assets}"
                )
            try:
                print(
                    f"[h3-data] verify reusable {job.shard.scenario_id}",
                    flush=True,
                )
                reusable[job.shard.scenario_id] = (
                    _audit_reusable_hidden_passage_shard(
                        swm,
                        job=job,
                    )
                )
            except Exception as exc:
                print(
                    f"[h3-data] rebuild incomplete "
                    f"{job.shard.scenario_id}: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
                _remove_incomplete_planned_shard(job.shard)
                missing_jobs.append(job)
        else:
            missing_jobs.append(job)
    _collect_missing_shards(
        swm,
        jobs=missing_jobs,
        config=config,
        repo_root=repo_root,
        stable_worldmodel_commit=stable_worldmodel_commit,
        workers=workers,
    )

    for job in ordered_jobs:
        shard = job.shard
        plans = list(job.plans)
        if shard.scenario_id in reusable:
            audit, episode_rows, completion = reusable[
                shard.scenario_id
            ]
            collection_status = "reused"
        else:
            print(
                f"[h3-data] audit {shard.scenario_id}",
                flush=True,
            )
            audit, episode_rows = audit_hidden_passage_shard(
                swm,
                shard=shard,
                plans=plans,
            )
            if not audit["passed"]:
                raise RuntimeError(
                    f"Shard audit failed: {shard.scenario_id}: "
                    f"{audit}"
                )
            _write_jsonl(shard.episode_manifest_path, episode_rows)
            completion = _publish_hidden_passage_shard_completion(
                shard=shard,
                episode_rows=episode_rows,
            )
            collection_status = "collected"
        record = _scenario_manifest_record(
            shard=shard,
            shard_audit=audit,
            episode_rows=episode_rows,
            completion=completion,
            collection_status=collection_status,
            stable_worldmodel_commit=stable_worldmodel_commit,
            pixel_codec=pixel_codec,
            repo_root=repo_root,
        )
        records.append(record)
        shard_audits[shard.scenario_id] = audit
        all_episode_rows.extend(episode_rows)

    pair_and_split = _pair_and_split_audit(
        episode_rows=all_episode_rows,
        door_splits=door_splits,
    )
    if not pair_and_split["passed"]:
        raise RuntimeError(
            f"Pair/split audit failed: {pair_and_split}"
        )
    expected_tables = {
        lexical_absolute_path(job.shard.table_path)
        for job in ordered_jobs
    }
    expected_markers = {
        shard_completion_marker_path(path)
        for path in expected_tables
    }
    expected_sidecars = {
        lexical_absolute_path(job.shard.episode_manifest_path)
        for job in ordered_jobs
    }
    release_asset_audit = audit_hidden_passage_release_assets(
        release_root=output_root,
        expected_tables=expected_tables,
        expected_markers=expected_markers,
        expected_sidecars=expected_sidecars,
    )
    training_query_hashes = {
        str(row["query_pixels_sha256"])
        for row in all_episode_rows
        if int(row["door_position"])
        in set(door_splits.train) | set(door_splits.val)
    }
    selected_query_overlap = sorted(
        training_query_hashes & frozen_query_hashes
    )
    validation_exclusion["checks"][
        "train_and_loader_val_exclude_selected_query_pixels"
    ] = not selected_query_overlap
    validation_exclusion["selected_query_pixel_hash_overlap"] = (
        selected_query_overlap
    )
    validation_exclusion["passed"] = all(
        validation_exclusion["checks"].values()
    )
    if not validation_exclusion["passed"]:
        raise RuntimeError(
            "Training data overlaps frozen Validation queries: "
            f"{selected_query_overlap[:20]}"
        )
    group_artifacts = write_group_artifacts(
        output_root=output_root,
        config=config,
        scale=scale,
        records=records,
        shard_audits=shard_audits,
        stable_worldmodel_commit=stable_worldmodel_commit,
        repo_root=repo_root,
    )
    group_union = _group_union_audit(
        output_root=output_root,
        config=config,
    )
    catalog_counts = _catalog_count_audit(
        config=config,
        scale=scale,
        group_artifacts=group_artifacts,
    )
    checks = {
        "all_shards_pass": all(
            value["passed"] for value in shard_audits.values()
        ),
        "frozen_validation_exclusion_passes": validation_exclusion[
            "passed"
        ],
        "pair_and_split_audit_passes": pair_and_split["passed"],
        "three_catalogs_are_same_source": group_union["passed"],
        "catalog_counts_are_exact": catalog_counts["passed"],
        "model_columns_are_pixels_and_action_only": (
            MODEL_KEYS == ("pixels", "action")
        ),
        "catalogs_are_synthetic_only": all(
            not json.loads(
                (
                    output_root
                    / "catalogs"
                    / f"{_catalog_stem(config, group)}.json"
                ).read_text(encoding="utf-8")
            )["train"]["original"]
            for group in GROUP_RULES
        ),
        "every_episode_is_exactly_one_h3_clip": all(
            record["episode_count"] == record["clip_count"]
            for record in records
        ),
        "no_unreferenced_lance_shards": release_asset_audit["passed"],
        "all_shards_have_valid_completion_markers": all(
            record.get("completion_protocol")
            == SHARD_COMPLETION_PROTOCOL
            and isinstance(record.get("completion_marker_sha256"), str)
            and len(record["completion_marker_sha256"]) == 64
            for record in records
        ),
        "no_unreferenced_completion_markers": release_asset_audit["passed"],
        "no_unreferenced_episode_sidecars": release_asset_audit["passed"],
    }
    report = {
        "schema_version": 1,
        "benchmark": config["benchmark"],
        "status": "passed" if all(checks.values()) else "failed",
        "passed": all(checks.values()),
        "scale": scale,
        "checks": checks,
        "catalog_keys": {
            group: {
                "catalog": artifact["catalog"],
                "catalog_sha256": artifact["catalog_sha256"],
            }
            for group, artifact in group_artifacts.items()
        },
        "artifacts_by_group": group_artifacts,
        "door_splits": door_splits.as_report(),
        "door_split_counts": {
            name: len(values)
            for name, values in door_splits.as_report().items()
        },
        "axes": {
            "directions": _scale_axes(config, scale)[0],
            "doorway_offsets_px": _scale_axes(config, scale)[1],
            "wall_distances_px": _scale_axes(config, scale)[2],
            "agent_speeds": list(
                map(
                    float,
                    config["protocol"].get(
                        "agent_speeds",
                        [config["protocol"]["agent_speed"]],
                    ),
                )
            ),
        },
        "pair_and_split_audit": pair_and_split,
        "validation_exclusion_audit": validation_exclusion,
        "group_union_audit": group_union,
        "catalog_count_audit": catalog_counts,
        "release_asset_audit": release_asset_audit,
        "physical_shards": len(records),
        "physical_episodes": sum(
            int(record["episode_count"]) for record in records
        ),
        "physical_rows": sum(int(record["raw_rows"]) for record in records),
        "resume_partial": bool(resume_partial),
        "collection_status_counts": dict(
            sorted(
                Counter(
                    record["collection_status"] for record in records
                ).items()
            )
        ),
        "collection_workers_requested": workers,
        "collection_workers_used": min(workers, len(missing_jobs)),
        "parallel_collection": workers > 1 and len(missing_jobs) > 1,
        "history3": {
            "history_tokens": 3,
            "frameskip": 5,
            "num_steps": 4,
            "rows_per_episode": RAW_STEPS,
            "clips_per_episode": 1,
            "model_input_keys": list(MODEL_KEYS),
        },
        "identity": {
            "config": portable_contextworld_path(
                Path(config["_config_path"]),
                repo_root=repo_root,
            ),
            "config_sha256": file_sha256(Path(config["_config_path"])),
            "config_canonical_sha256": canonical_sha256(
                {
                    key: value
                    for key, value in config.items()
                    if key != "_config_path"
                }
            ),
            "stable_worldmodel_commit": stable_worldmodel_commit,
        },
        "claim_limit": (
            "This report proves data construction, split isolation, paired "
            "hidden-rule support and StableWM loader compatibility. It does "
            "not by itself prove model ICL."
        ),
    }
    write_json(output_root / "build_report.json", report)
    return report


__all__ = [
    "AUDIT_SCHEDULING_LOCK_PROTOCOL",
    "GROUP_RULES",
    "LOGICAL_CONTENT_COLUMNS",
    "LOGICAL_CONTENT_HASH_KIND",
    "PARALLEL_AUDIT_SCHEDULING_LOCK_PROTOCOL",
    "SHARD_COMPLETION_PROTOCOL",
    "SHARD_COMPLETION_SUFFIX",
    "STORAGE_CONTENT_HASH_KIND",
    "TRAINING_RUN_LOCK_PROTOCOL",
    "HiddenPassageDoorSplits",
    "HiddenPassageEpisodePlan",
    "HiddenPassageShardCollectionJob",
    "HiddenPassageShardPlan",
    "array_sha256",
    "audit_hidden_passage_release_assets",
    "audit_validation_exclusion",
    "build_hidden_passage_h3_data",
    "canonical_sha256",
    "directory_sha256",
    "door_splits_for_scale",
    "episode_plans_for_door",
    "file_sha256",
    "hidden_passage_release_lock",
    "hidden_passage_audit_scheduling_lock",
    "hidden_passage_audit_scheduling_lock_path",
    "hidden_passage_training_run_lock",
    "hidden_passage_training_run_lock_path",
    "verify_hidden_passage_training_run_parent",
    "lexical_absolute_path",
    "lexical_contextworld_path",
    "logical_episode_content_hashes",
    "logical_shard_content_sha256",
    "require_lexical_containment",
    "require_safe_directory",
    "require_safe_missing_or_directory",
    "require_safe_regular_file",
    "shard_completion_marker_path",
    "templates_for_door",
    "validate_regular_directory_tree",
    "verify_hidden_passage_shard_completion",
]
