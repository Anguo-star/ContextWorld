#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from contextworld.synthesis.collector import collect_scenario
from contextworld.synthesis.config import (
    build_compiler,
    load_config,
    scenario_requests,
)
from contextworld.synthesis.stablewm import load_stable_worldmodel


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect one deterministic shard of a synthesis config"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    args = parser.parse_args()
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError(
            f"Invalid shard {args.shard_index}/{args.num_shards}"
        )

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    import torch

    torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
    torch.set_num_interop_threads(1)

    config = load_config(args.config)
    specification = config["stable_worldmodel"]
    swm, _, _ = load_stable_worldmodel(
        REPO_ROOT,
        specification["repo"],
        specification.get("expected_ref"),
    )
    compiler = build_compiler(config, REPO_ROOT)
    scenarios = compiler.compile_all(scenario_requests(config))
    selected = scenarios[args.shard_index :: args.num_shards]
    status = {"collected": 0, "reused": 0}
    for scenario in selected:
        result = collect_scenario(
            swm,
            scenario,
            config["collection"],
            resume=True,
        )
        status[result] += 1
    print(
        json.dumps(
            {
                "shard_index": args.shard_index,
                "num_shards": args.num_shards,
                "selected": len(selected),
                **status,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
