from __future__ import annotations

import os
from pathlib import Path


ARTIFACT_ROOT_ENV = "CONTEXTWORLD_ARTIFACT_ROOT"
LEGACY_ARTIFACT_PREFIX = "artifacts"


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def artifact_root(repo_root: Path | None = None) -> Path:
    """Return the canonical root for generated data and experiment outputs.

    The default follows the workspace layout requested for ContextWorld:
    ``<ag_data>/data/world_model/context_world``.  The environment override is
    useful when the same code checkout is run on another mounted filesystem.
    """

    configured = os.environ.get(ARTIFACT_ROOT_ENV)
    if configured:
        return Path(configured).expanduser().resolve()
    root = (repo_root or repository_root()).resolve()
    return (root.parents[1] / "data/world_model/context_world").resolve()


def artifact_path(*parts: str | Path, repo_root: Path | None = None) -> Path:
    return artifact_root(repo_root).joinpath(*parts)


def resolve_contextworld_path(
    value: str | Path, *, repo_root: Path | None = None
) -> Path:
    """Resolve repo paths and stable ``artifacts/...`` logical references."""

    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    root = (repo_root or repository_root()).resolve()
    if path.parts and path.parts[0] == LEGACY_ARTIFACT_PREFIX:
        return artifact_root(root).joinpath(*path.parts[1:]).resolve()
    return (root / path).resolve()


def portable_contextworld_path(
    value: str | Path, *, repo_root: Path | None = None
) -> str:
    """Serialize artifact paths without tying manifests to a mount prefix."""

    path = Path(value).expanduser().resolve()
    root = (repo_root or repository_root()).resolve()
    artifacts = artifact_root(root)
    try:
        relative = path.relative_to(artifacts)
        return str(Path(LEGACY_ARTIFACT_PREFIX) / relative)
    except ValueError:
        pass
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)
