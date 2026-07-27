#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contextworld.evaluation.hidden_passage_h3_data import (
    build_hidden_passage_h3_data,
    hidden_passage_release_lock,
    lexical_absolute_path,
    lexical_contextworld_path,
    require_safe_directory,
    require_safe_missing_or_directory,
    validate_regular_directory_tree,
)
from contextworld.synthesis.config import load_config
from contextworld.synthesis.stablewm import load_stable_worldmodel


DEFAULT_CONFIG = (
    ROOT
    / "configs/benchmark/tworoom_hidden_passage_h3_training_data_v1.yaml"
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
            "Stable-WorldModel has tracked changes; hidden-passage data "
            "must be collected with the pinned implementation exactly"
        )


def _prepare_output(
    output_root: Path,
    *,
    benchmark: str,
    scale: str,
    refresh_existing: bool,
    resume_partial: bool,
) -> None:
    output_root = require_safe_missing_or_directory(output_root)
    if not output_root.exists():
        output_root.mkdir(parents=True)
        require_safe_directory(output_root)
        return
    require_safe_directory(output_root)
    validate_regular_directory_tree(output_root)
    if resume_partial:
        report_path = output_root / "build_report.json"
        if report_path.is_file():
            prior = json.loads(report_path.read_text(encoding="utf-8"))
            if (
                prior.get("benchmark") != benchmark
                or prior.get("scale") != scale
            ):
                raise ValueError(
                    "Existing completed output belongs to another benchmark "
                    "or scale"
                )
        elif not (output_root / "tables").is_dir():
            raise FileNotFoundError(
                "--resume-partial requires prior fingerprinted tables or "
                "a completed build_report.json"
            )
        return
    if not refresh_existing:
        raise FileExistsError(
            f"Refusing to overwrite hidden-passage output {output_root}"
        )
    report_path = output_root / "build_report.json"
    if not report_path.is_file():
        raise FileNotFoundError(
            "--refresh-existing requires the prior build_report.json"
        )
    prior = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        prior.get("benchmark") != benchmark
        or prior.get("scale") != scale
    ):
        raise ValueError(
            "Refusing to replace output belonging to another benchmark "
            f"or scale: {prior.get('benchmark')!r}/{prior.get('scale')!r}"
        )
    # The complete tree was lstat-validated above.  Never let rmtree follow a
    # replaced release root or an alias hidden anywhere below it.
    require_safe_directory(output_root)
    validate_regular_directory_tree(output_root)
    shutil.rmtree(output_root)
    output_root.mkdir(parents=True)
    require_safe_directory(output_root)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build paired, sharded History-3 hidden-passage training data "
            "and the passable/blocked/mixed StableWM catalogs"
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--scale",
        choices=("small", "formal"),
        required=True,
        help="small validates the complete pipeline; formal builds release data",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help=(
            "Exact output directory. By default, append the selected scale "
            "to output_root_base from the config."
        ),
    )
    parser.add_argument("--refresh-existing", action="store_true")
    parser.add_argument(
        "--resume-partial",
        action="store_true",
        help=(
            "Reuse only fingerprint-matching Lance shards after re-running "
            "their complete loader audit."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Number of shard collection processes. The default 1 preserves "
            "serial behavior; use 4 for the formal build. Audits and "
            "catalog/manifest writes always remain in the main process."
        ),
    )
    args = parser.parse_args()
    if args.refresh_existing and args.resume_partial:
        parser.error(
            "--refresh-existing and --resume-partial are mutually exclusive"
        )
    if args.workers < 1:
        parser.error("--workers must be at least 1")

    config_path = args.config.expanduser().resolve()
    config = load_config(config_path)
    swm, stable_repo, stable_commit = load_stable_worldmodel(
        ROOT,
        str(config["stable_worldmodel"]["repo"]),
        str(config["stable_worldmodel"]["commit"]),
    )
    _assert_stable_worldmodel_tracked_clean(stable_repo)
    if args.output_root is None:
        base = lexical_contextworld_path(
            config["output_root_base"],
            repo_root=ROOT,
        )
        output_root = base / args.scale
    else:
        output_root = lexical_absolute_path(args.output_root)
    with hidden_passage_release_lock(
        output_root,
        exclusive=True,
    ) as release_lock:
        _prepare_output(
            output_root,
            benchmark=str(config["benchmark"]),
            scale=args.scale,
            refresh_existing=bool(args.refresh_existing),
            resume_partial=bool(args.resume_partial),
        )
        print(
            f"[h3-data] pinned StableWM {stable_commit}; "
            f"scale={args.scale}; output={output_root}; "
            f"lock={release_lock['mode']}",
            flush=True,
        )
        report = build_hidden_passage_h3_data(
            swm,
            config=config,
            scale=args.scale,
            output_root=output_root,
            repo_root=ROOT,
            stable_worldmodel_commit=stable_commit,
            resume_partial=bool(args.resume_partial),
            workers=int(args.workers),
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
