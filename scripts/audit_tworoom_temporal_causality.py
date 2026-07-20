#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contextworld.evaluation.icl_model import (
    file_sha256,
    state_dict_sha256,
)
from contextworld.evaluation.protocol import (
    infer_model_protocol,
    load_pretrained_cost_model,
)
from contextworld.paths import artifact_path, resolve_contextworld_path
from contextworld.synthesis.manifest import write_json
from contextworld.synthesis.stablewm import load_stable_worldmodel


DEFAULT_CONFIG = (
    ROOT / "configs/benchmark/tworoom_speed_cube_eval_v2.yaml"
)


def temporal_causality_probe(
    predictor: Any,
    *,
    seed: int,
    trials: int,
    sequence_length: int,
    tolerance: float,
) -> dict[str, Any]:
    import torch

    if sequence_length < 2:
        raise ValueError("sequence_length must be at least two")
    predictor = predictor.cpu().float().eval()
    input_dim = int(predictor.input_dim)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    maximum_earlier_change = 0.0
    maximum_future_change = 0.0
    rows = []
    with torch.inference_mode():
        for trial in range(int(trials)):
            observations = torch.randn(
                2,
                sequence_length,
                input_dim,
                generator=generator,
            )
            actions = torch.randn(
                2,
                sequence_length,
                input_dim,
                generator=generator,
            )
            baseline = predictor(observations, actions)
            for boundary in range(sequence_length - 1):
                modified_observations = observations.clone()
                modified_actions = actions.clone()
                modified_observations[:, boundary + 1 :] = (
                    7.0
                    * torch.randn(
                        modified_observations[
                            :, boundary + 1 :
                        ].shape,
                        generator=generator,
                    )
                )
                modified_actions[:, boundary + 1 :] = (
                    7.0
                    * torch.randn(
                        modified_actions[:, boundary + 1 :].shape,
                        generator=generator,
                    )
                )
                changed = predictor(
                    modified_observations, modified_actions
                )
                earlier_change = float(
                    torch.max(
                        torch.abs(
                            changed[:, : boundary + 1]
                            - baseline[:, : boundary + 1]
                        )
                    )
                )
                future_change = float(
                    torch.max(
                        torch.abs(
                            changed[:, boundary + 1 :]
                            - baseline[:, boundary + 1 :]
                        )
                    )
                )
                maximum_earlier_change = max(
                    maximum_earlier_change, earlier_change
                )
                maximum_future_change = max(
                    maximum_future_change, future_change
                )
                rows.append(
                    {
                        "trial": trial,
                        "boundary": boundary,
                        "maximum_change_at_or_before_boundary": (
                            earlier_change
                        ),
                        "maximum_change_after_boundary": future_change,
                    }
                )
    future_changed = bool(maximum_future_change > tolerance)
    return {
        "seed": int(seed),
        "trials": int(trials),
        "sequence_length": int(sequence_length),
        "input_dim": input_dim,
        "tolerance": float(tolerance),
        "maximum_change_at_or_before_boundary": (
            maximum_earlier_change
        ),
        "maximum_change_after_boundary": maximum_future_change,
        "future_perturbation_changed_a_future_output": future_changed,
        "passed": bool(
            maximum_earlier_change <= tolerance and future_changed
        ),
        "rows": rows,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = yaml.safe_load(
        args.config.resolve().read_text(encoding="utf-8")
    )
    swm, stable_repo, stable_commit = load_stable_worldmodel(
        ROOT,
        (ROOT / config["stable_worldmodel"]["repo"]).resolve(),
        config["stable_worldmodel"]["expected_ref"],
    )
    models = [
        (group, dict(model))
        for group, members in config["models"].items()
        for model in members
    ]
    results = {}
    for index, (group, model_row) in enumerate(models):
        slug = str(model_row["slug"])
        checkpoint = resolve_contextworld_path(
            model_row["checkpoint"], repo_root=ROOT
        )
        model = load_pretrained_cost_model(
            checkpoint,
            swm,
            cache_dir=artifact_path(
                "evaluation/model_cache", repo_root=ROOT
            ),
        )
        protocol = infer_model_protocol(model, action_dim=2)
        if protocol != {"action_block": 5, "history_size": 3}:
            raise RuntimeError(
                f"Unexpected model protocol for {slug}: {protocol}"
            )
        weight_before = state_dict_sha256(model)
        probe = temporal_causality_probe(
            model.predictor,
            seed=int(args.seed) + index,
            trials=args.trials,
            sequence_length=protocol["history_size"],
            tolerance=args.tolerance,
        )
        weight_after = state_dict_sha256(model)
        results[slug] = {
            "model_group": group,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": file_sha256(checkpoint),
            "protocol": protocol,
            "probe": probe,
            "frozen_weight_audit": {
                "state_dict_sha256_before": weight_before,
                "state_dict_sha256_after": weight_after,
                "passed": weight_before == weight_after,
            },
            "passed": bool(
                probe["passed"] and weight_before == weight_after
            ),
        }
        del model
    output = {
        "schema_version": 1,
        "benchmark": "tworoom_history3_temporal_causality_audit",
        "status": (
            "passed"
            if all(row["passed"] for row in results.values())
            else "failed"
        ),
        "stable_worldmodel": {
            "repo": str(stable_repo),
            "commit": stable_commit,
        },
        "interpretation": (
            "Changing model-frame observation and action tokens strictly "
            "after a boundary must not change predictor outputs at or "
            "before that boundary."
        ),
        "models": results,
    }
    write_json(
        resolve_contextworld_path(args.output, repo_root=ROOT), output
    )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/evaluation/history3/speed_isolated_v2/"
            "temporal_causality_audit.json"
        ),
    )
    parser.add_argument("--seed", type=int, default=2026072024)
    parser.add_argument("--trials", type=int, default=4)
    parser.add_argument("--tolerance", type=float, default=1.0e-6)
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(
        json.dumps(
            {
                "status": result["status"],
                "models": {
                    slug: {
                        "passed": row["passed"],
                        "maximum_change_at_or_before_boundary": row[
                            "probe"
                        ]["maximum_change_at_or_before_boundary"],
                    }
                    for slug, row in result["models"].items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
