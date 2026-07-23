#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contextworld.evaluation.hidden_passage import build_feasibility_catalog
from contextworld.evaluation.icl_model import file_sha256
from contextworld.paths import (
    portable_contextworld_path,
    resolve_contextworld_path,
)
from contextworld.synthesis.config import load_config
from contextworld.synthesis.manifest import write_json
from contextworld.synthesis.stablewm import load_stable_worldmodel


DEFAULT_CONFIG = (
    ROOT
    / "configs/benchmark/tworoom_hidden_passage_h3_feasibility_v1.yaml"
)


def _output_root(config: dict, override: Path | None) -> Path:
    if override is not None:
        return resolve_contextworld_path(override, repo_root=ROOT)
    return resolve_contextworld_path(
        config["output_root"],
        repo_root=ROOT,
    )


def _assert_stable_worldmodel_tracked_clean(stable_repo: Path) -> None:
    result = subprocess.run(
        [
            "git",
            "-C",
            str(stable_repo),
            "status",
            "--porcelain",
            "--untracked-files=no",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    if result.stdout.strip():
        raise RuntimeError(
            "Stable-WorldModel has tracked working-tree changes; "
            "the feasibility build requires the pinned commit exactly"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build and audit the History-3 hidden-passage physical "
            "feasibility catalog"
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--refresh-existing",
        action="store_true",
        help=(
            "Rebuild an existing output only when its report has the same "
            "benchmark identity; the default remains fail-closed"
        ),
    )
    args = parser.parse_args()

    config = load_config(args.config)
    _, stable_repo, stable_commit = load_stable_worldmodel(
        ROOT,
        str(config["stable_worldmodel"]["repo"]),
        str(config["stable_worldmodel"]["commit"]),
    )
    _assert_stable_worldmodel_tracked_clean(stable_repo)
    output_root = _output_root(config, args.output_root)
    if output_root.exists():
        if not args.refresh_existing:
            raise FileExistsError(
                "Refusing to overwrite existing feasibility output "
                f"{output_root}"
            )
        existing_report_path = output_root / "build_report.json"
        if not existing_report_path.is_file():
            raise FileNotFoundError(
                "--refresh-existing requires the prior build report at "
                f"{existing_report_path}"
            )
        with existing_report_path.open("r", encoding="utf-8") as handle:
            existing_report = json.load(handle)
        if existing_report.get("benchmark") != config["benchmark"]:
            raise ValueError(
                "Refusing to refresh output owned by benchmark "
                f"{existing_report.get('benchmark')!r}"
            )
    else:
        output_root.mkdir(parents=True, exist_ok=False)

    catalog, report = build_feasibility_catalog(
        config=config,
        repo_root=ROOT,
        output_root=output_root,
    )
    with tempfile.TemporaryDirectory(
        prefix="contextworld-hidden-passage-rebuild-"
    ) as temporary:
        shadow_catalog, shadow_report = build_feasibility_catalog(
            config=config,
            repo_root=ROOT,
            output_root=Path(temporary) / "shadow",
        )
    deterministic_rebuild = (
        catalog["content_manifest_sha256"]
        == shadow_catalog["content_manifest_sha256"]
        and report["content_manifest_sha256"]
        == shadow_report["content_manifest_sha256"]
    )
    report["checks"]["deterministic_rebuild"] = deterministic_rebuild
    report["status"] = (
        "passed" if all(report["checks"].values()) else "failed"
    )
    report["identity"] = {
        "config": portable_contextworld_path(
            Path(args.config).resolve(),
            repo_root=ROOT,
        ),
        "config_sha256": file_sha256(Path(args.config).resolve()),
        "stable_worldmodel_repo": str(stable_repo),
        "stable_worldmodel_commit": stable_commit,
        "stable_worldmodel_tracked_tree_clean": True,
        "stable_worldmodel_sources": {
            name: {
                "path": str(path.relative_to(stable_repo)),
                "sha256": file_sha256(path),
            }
            for name, path in {
                "two_room_environment": (
                    stable_repo
                    / "stable_worldmodel/envs/two_room/env.py"
                ),
                "variation_spaces": (
                    stable_repo / "stable_worldmodel/spaces.py"
                ),
            }.items()
        },
        "contextworld_sources": {
            name: {
                "path": portable_contextworld_path(path, repo_root=ROOT),
                "sha256": file_sha256(path),
            }
            for name, path in {
                "environment": (
                    ROOT
                    / "contextworld/evaluation/hidden_passage_env.py"
                ),
                "catalog_builder": (
                    ROOT / "contextworld/evaluation/hidden_passage.py"
                ),
                "entrypoint": Path(__file__).resolve(),
            }.items()
        },
    }

    catalog_path = output_root / "catalog.json"
    report_path = output_root / "build_report.json"
    write_json(catalog_path, catalog)
    catalog_reference = portable_contextworld_path(
        catalog_path,
        repo_root=ROOT,
    )
    resolved_catalog = resolve_contextworld_path(
        catalog_reference,
        repo_root=ROOT,
    )
    if resolved_catalog != catalog_path.resolve():
        raise RuntimeError(
            "Catalog reference does not resolve back to its output: "
            f"{catalog_reference!r} -> {resolved_catalog}, "
            f"expected {catalog_path.resolve()}"
        )
    report["catalog"] = catalog_reference
    report["catalog_sha256"] = file_sha256(resolved_catalog)
    report["checks"]["catalog_path_reopens"] = resolved_catalog.is_file()
    report["status"] = (
        "passed" if all(report["checks"].values()) else "failed"
    )
    write_json(report_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
