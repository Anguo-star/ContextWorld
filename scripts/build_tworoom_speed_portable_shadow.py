#!/usr/bin/env python3
"""Build the repo-local portable metadata shadow for Speed ICL v1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contextworld.benchmarks.speed_release_shadow import (  # noqa: E402
    build_speed_portable_shadow,
)
from contextworld.paths import artifact_root  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--canonical-artifact-root", type=Path)
    parser.add_argument("--original-h5", type=Path)
    parser.add_argument("--stable-worldmodel-root", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = args.repo_root.expanduser().resolve()
    canonical = (
        args.canonical_artifact_root.expanduser().resolve()
        if args.canonical_artifact_root is not None
        else artifact_root(repo)
    )
    original = (
        args.original_h5.expanduser().resolve()
        if args.original_h5 is not None
        else canonical.parent / "quentinll/lewm-tworooms/tworoom.h5"
    )
    stablewm = (
        args.stable_worldmodel_root.expanduser().resolve()
        if args.stable_worldmodel_root is not None
        else repo.parent / "stable-worldmodel"
    )
    result = build_speed_portable_shadow(
        repo_root=repo,
        canonical_artifact_root=canonical,
        original_h5=original,
        stable_worldmodel_root=stablewm,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
