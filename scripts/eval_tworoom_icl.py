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

from contextworld.evaluation.icl_catalog import validate_context_query_catalog
from contextworld.evaluation.icl_model import evaluate_frozen_tworoom_icl
from contextworld.evaluation.protocol import (
    frozen_normalizer_process,
    load_legacy_cost_model,
    load_pretrained_cost_model,
    original_h5_process,
)
from contextworld.synthesis.manifest import write_json
from contextworld.paths import artifact_path, resolve_contextworld_path
from contextworld.synthesis.stablewm import load_stable_worldmodel


PINNED_STABLEWM = "5864b74980f6ed328fd0045e777b3865962eff43"


def run(args: argparse.Namespace) -> dict:
    args.catalog = resolve_contextworld_path(args.catalog, repo_root=REPO_ROOT)
    args.output = resolve_contextworld_path(args.output, repo_root=REPO_ROOT)
    os.environ.setdefault("MUJOCO_GL", "egl")
    swm, stable_repo, stable_commit = load_stable_worldmodel(
        REPO_ROOT, args.stablewm_repo, args.stablewm_ref
    )
    catalog_validation = validate_context_query_catalog(
        args.catalog.resolve(),
        repo_root=REPO_ROOT,
        replay_simulator=not args.skip_catalog_replay,
        family=args.family,
    )
    if not catalog_validation["passed"]:
        raise RuntimeError(
            f"Catalog validation failed: {catalog_validation['failures'][:5]}"
        )

    checkpoint = args.checkpoint.resolve()
    if checkpoint.suffix.lower() == ".pt":
        model = load_pretrained_cost_model(
            checkpoint,
            swm,
            cache_dir=artifact_path(
                "evaluation/model_cache", repo_root=REPO_ROOT
            ),
        )
        checkpoint_serialization = "stablewm_pretrained"
    else:
        model = load_legacy_cost_model(
            checkpoint, args.legacy_code_root.resolve()
        )
        checkpoint_serialization = "legacy_object"
    process = (
        frozen_normalizer_process(args.normalizer.resolve())
        if args.normalizer is not None
        else original_h5_process(args.original_h5.resolve())
    )
    result = evaluate_frozen_tworoom_icl(
        model=model,
        checkpoint_path=args.checkpoint.resolve(),
        catalog_path=args.catalog.resolve(),
        repo_root=REPO_ROOT,
        original_h5=args.original_h5.resolve(),
        action_standardizer=process["action"],
        device=args.device,
        encode_batch_size=args.encode_batch_size,
        predictor_batch_size=args.predictor_batch_size,
        seed=args.seed,
        family=args.family,
    )
    result["stable_worldmodel"] = {
        "repo": str(stable_repo),
        "commit": stable_commit,
    }
    result["checkpoint"]["serialization"] = checkpoint_serialization
    result["data"]["normalization_source"] = str(
        args.normalizer.resolve()
        if args.normalizer is not None
        else args.original_h5.resolve()
    )
    result["catalog_validation"] = catalog_validation
    write_json(args.output.resolve(), result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Frozen-weight TwoRoom paired-context latent prediction"
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=artifact_path(
            "evaluation/icl/tworoom_icl_v1_validation_context_query_catalog.json",
            repo_root=REPO_ROOT,
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "/opt/workspace/explorer-env/dataset/ag_data/data/world_model/"
            "quentinll/lewm-tworooms/ckpt/tworoom_lewm_20260430/"
            "tworoom_lewm_20260430_epoch_10_object.ckpt"
        ),
    )
    parser.add_argument(
        "--legacy-code-root",
        type=Path,
        default=Path(
            "/opt/workspace/explorer-env/dataset/ag_data/code/wm_exp"
        ),
    )
    parser.add_argument(
        "--original-h5",
        type=Path,
        default=Path(
            "/opt/workspace/explorer-env/dataset/ag_data/data/world_model/"
            "quentinll/lewm-tworooms/tworoom.h5"
        ),
    )
    parser.add_argument("--normalizer", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=artifact_path(
            "evaluation/icl/tworoom_m_orig_validation_t0_t1.json",
            repo_root=REPO_ROOT,
        ),
    )
    parser.add_argument("--stablewm-repo", default="../stable-worldmodel")
    parser.add_argument("--stablewm-ref", default=PINNED_STABLEWM)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--encode-batch-size", type=int, default=64)
    parser.add_argument("--predictor-batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=3072)
    parser.add_argument(
        "--family",
        choices=("speed", "door", "speed_door_composition"),
        help="Evaluate only one factor family from the frozen catalog.",
    )
    parser.add_argument("--skip-catalog-replay", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = run(args)
    compact = {
        "status": result["status"],
        "frozen": result["frozen_weight_audit"]["passed"],
        "signals": result["diagnostic_signals"],
        "output": str(args.output.resolve()),
    }
    print(json.dumps(compact, ensure_ascii=False, sort_keys=True))
