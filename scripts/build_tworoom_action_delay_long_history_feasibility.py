#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contextworld.evaluation.action_delay_long_history import (
    build_long_history_feasibility,
)
from contextworld.paths import (
    portable_contextworld_path,
    resolve_contextworld_path,
)
from contextworld.synthesis.config import load_config
from contextworld.synthesis.manifest import write_json
from contextworld.synthesis.stablewm import load_stable_worldmodel


DEFAULT_CONFIG = (
    ROOT
    / "configs/benchmark/"
    "tworoom_action_delay_long_history_feasibility_v1.yaml"
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Select the shortest physically valid long history for "
            "TwoRoom action delay"
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--refresh-existing", action="store_true")
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = load_config(config_path)
    _, stable_repo, stable_commit = load_stable_worldmodel(
        ROOT,
        str(config["stable_worldmodel"]["repo"]),
        str(config["stable_worldmodel"]["commit"]),
    )
    output_root = resolve_contextworld_path(
        (
            args.output_root
            if args.output_root is not None
            else config["output_root"]
        ),
        repo_root=ROOT,
    )
    if output_root.exists() and not args.refresh_existing:
        raise FileExistsError(
            f"Refusing to overwrite feasibility output {output_root}"
        )
    output_root.mkdir(parents=True, exist_ok=True)

    catalog, report = build_long_history_feasibility(
        config=config,
        repo_root=ROOT,
        output_root=output_root,
    )
    with tempfile.TemporaryDirectory(
        prefix="contextworld-action-delay-long-rebuild-"
    ) as temporary:
        shadow_catalog, shadow_report = build_long_history_feasibility(
            config=config,
            repo_root=ROOT,
            output_root=Path(temporary) / "shadow",
        )
    report["checks"]["deterministic_rebuild"] = bool(
        catalog["content_manifest_sha256"]
        == shadow_catalog["content_manifest_sha256"]
        and report["content_manifest_sha256"]
        == shadow_report["content_manifest_sha256"]
    )
    report["identity"] = {
        "config": portable_contextworld_path(
            config_path,
            repo_root=ROOT,
        ),
        "config_sha256": file_sha256(config_path),
        "stable_worldmodel_repo": str(stable_repo),
        "stable_worldmodel_commit": stable_commit,
        "contextworld_sources": {
            name: {
                "path": portable_contextworld_path(path, repo_root=ROOT),
                "sha256": file_sha256(path),
            }
            for name, path in {
                "environment": (
                    ROOT / "contextworld/evaluation/action_delay_env.py"
                ),
                "feasibility": (
                    ROOT
                    / "contextworld/evaluation/"
                    "action_delay_long_history.py"
                ),
                "entrypoint": Path(__file__).resolve(),
            }.items()
        },
    }
    catalog_path = output_root / "catalog.json"
    write_json(catalog_path, catalog)
    report["catalog"] = portable_contextworld_path(
        catalog_path,
        repo_root=ROOT,
    )
    report["catalog_sha256"] = file_sha256(catalog_path)
    report["checks"]["catalog_reopens"] = (
        resolve_contextworld_path(
            report["catalog"],
            repo_root=ROOT,
        )
        == catalog_path.resolve()
    )
    report["status"] = (
        "passed" if all(report["checks"].values()) else "failed"
    )
    report_path = output_root / "build_report.json"
    write_json(report_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
