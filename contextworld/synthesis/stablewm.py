from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType


def _git_commit(repo: Path) -> str:
    """Resolve HEAD without changing the user's global Git safety config."""

    marker = repo / ".git"
    if marker.is_dir():
        git_dir = marker
    elif marker.is_file():
        value = marker.read_text(encoding="utf-8").strip()
        prefix = "gitdir: "
        if not value.startswith(prefix):
            raise RuntimeError(f"Unrecognized Git marker at {marker}")
        git_dir = (repo / value[len(prefix) :]).resolve()
    else:
        raise FileNotFoundError(f"Git metadata not found at {marker}")

    head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
    if not head.startswith("ref: "):
        return head
    ref = head[len("ref: ") :]
    loose_ref = git_dir / ref
    if loose_ref.is_file():
        return loose_ref.read_text(encoding="utf-8").strip()
    packed_refs = git_dir / "packed-refs"
    if packed_refs.is_file():
        for line in packed_refs.read_text(encoding="utf-8").splitlines():
            if line.startswith(("#", "^")):
                continue
            commit, _, candidate = line.partition(" ")
            if candidate == ref:
                return commit
    raise RuntimeError(f"Cannot resolve {ref!r} under {git_dir}")


def load_stable_worldmodel(
    repo_root: Path,
    configured_repo: str,
    expected_ref: str | None = None,
) -> tuple[ModuleType, Path, str]:
    """Import the explicitly configured sibling checkout and verify its ref."""

    stable_repo = (repo_root / configured_repo).resolve()
    package_dir = stable_repo / "stable_worldmodel"
    if not package_dir.is_dir():
        raise FileNotFoundError(
            f"Stable-WorldModel package not found at {package_dir}"
        )

    commit = _git_commit(stable_repo)
    if expected_ref and commit != expected_ref:
        raise RuntimeError(
            "Stable-WorldModel checkout differs from the smoke pin: "
            f"expected {expected_ref}, found {commit}"
        )

    repo_string = str(stable_repo)
    if repo_string not in sys.path:
        sys.path.insert(0, repo_string)
    swm = importlib.import_module("stable_worldmodel")
    imported_path = Path(swm.__file__).resolve()
    if stable_repo not in imported_path.parents:
        raise RuntimeError(
            f"Imported stable_worldmodel from {imported_path}, not {stable_repo}"
        )
    return swm, stable_repo, commit
