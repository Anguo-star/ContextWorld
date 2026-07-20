#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from contextworld.synthesis.manifest import write_json
from contextworld.paths import resolve_contextworld_path
from contextworld.synthesis.stablewm import load_stable_worldmodel
from contextworld.training.tworoom_data import build_tworoom_grouped_data


PINNED_STABLEWM = "5864b74980f6ed328fd0045e777b3865962eff43"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="M_composed")
    parser.add_argument(
        "--benchmark-config",
        type=Path,
        default=REPO_ROOT / "configs/benchmark/tworoom_step1_v1.yaml",
    )
    parser.add_argument("--stablewm-repo", default="../stable-worldmodel")
    parser.add_argument("--stablewm-ref", default=PINNED_STABLEWM)
    parser.add_argument("--epoch-size", type=int, default=120)
    parser.add_argument("--validation-epoch-size", type=int, default=120)
    parser.add_argument("--sample-count", type=int, default=12)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output = resolve_contextworld_path(args.output, repo_root=REPO_ROOT)
    args.benchmark_config = resolve_contextworld_path(
        args.benchmark_config, repo_root=REPO_ROOT
    )

    swm, stable_repo, commit = load_stable_worldmodel(
        REPO_ROOT, args.stablewm_repo, args.stablewm_ref
    )
    data = build_tworoom_grouped_data(
        swm,
        repo_root=REPO_ROOT,
        benchmark_config=args.benchmark_config.resolve(),
        model_id=args.model_id,
        epoch_size=args.epoch_size,
        validation_epoch_size=args.validation_epoch_size,
    )

    group_counts = {}
    sample_shapes = None
    for index in range(min(args.sample_count, len(data.train))):
        group, _ = data.train.locate(index)
        group_counts[group] = group_counts.get(group, 0) + 1
        sample = data.train[index]
        shapes = {key: list(value.shape) for key, value in sample.items()}
        sample_shapes = sample_shapes or shapes
        if shapes != sample_shapes:
            raise RuntimeError(
                f"Cross-group sample shape mismatch at index {index}: "
                f"expected={sample_shapes}, observed={shapes}"
            )

    report = {
        "schema_version": 1,
        "audit": "tworoom_logical_group_loader",
        "passed": True,
        "stable_worldmodel": {"repo": str(stable_repo), "commit": commit},
        "metadata": data.metadata,
        "sample_count": min(args.sample_count, len(data.train)),
        "sampled_group_counts": group_counts,
        "sample_shapes": sample_shapes,
    }
    write_json(args.output.resolve(), report)
    print(json.dumps({"passed": True, **data.metadata["epoch_group_counts"]}))


if __name__ == "__main__":
    main()
