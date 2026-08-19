#!/usr/bin/env python3
"""Score one History-7 checkpoint on the three training-replay tracks.

This is a post-hoc root-cause diagnostic.  It deliberately does not reuse the
formal evaluator's final-checkpoint receipt because intermediate epoch weights
are the object being inspected.  The frozen catalog, physical futures,
normalizer, adapter boundary, and latent scoring implementation are reused
unchanged.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contextworld.benchmarks.adapters import (
    StableWorldModelLeWMHistory7Adapter,
    StableWorldModelPLDMAdapter,
)
from contextworld.evaluation.action_delay_h7_domain_score import (
    load_domain_catalog,
    load_domain_track_assets,
    score_domain_track,
)
from contextworld.evaluation.action_delay_h7_validation import file_sha256
from contextworld.paths import resolve_contextworld_path
from contextworld.synthesis.manifest import write_json


DEFAULT_CONFIG = (
    ROOT
    / "configs/benchmark/"
    "tworoom_action_delay_h7_domain_diagnostic_scoring_v1.yaml"
)
DELAYS = (0, 4, 8)
TRACKS_BY_SCOPE = {
    "training_replay": tuple(
        f"training_replay_delay_{delay}" for delay in DELAYS
    ),
    "loader_validation": tuple(
        f"loader_validation_delay_{delay}" for delay in DELAYS
    ),
}


class StableWorldModelPLDMHistory7DiagnosticAdapter(
    StableWorldModelPLDMAdapter
):
    """History-7 PLDM adapter used only by this diagnostic."""

    adapter_id = "stable_worldmodel_pldm_history7_diagnostic_v1"
    required_history_tokens = 7
    maximum_future_action_blocks = 3


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _checkpoint_protocol(path: Path) -> dict[str, int]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return {
        "history_size": int(value["wm"]["history_size"]),
        "num_preds": int(value["wm"]["num_preds"]),
        "frameskip": int(value["data"]["dataset"]["frameskip"]),
        "num_steps": int(value["data"]["dataset"]["num_steps"]),
        "action_encoder_input_dim": int(
            value["model"]["action_encoder"]["input_dim"]
        ),
    }


def _source_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    source_delay = int(result["source_delay"])
    return [
        row
        for row in result["by_horizon"]["1"]["query_metrics"]
        if int(row["target_delay"]) == source_delay
    ]


def _track_summary(result: dict[str, Any]) -> dict[str, Any]:
    rows = _source_rows(result)
    _require(len(rows) == 300, "Each source track must contain 300 h1 units")
    counts = Counter(int(row["selected_target"]) for row in rows)
    return {
        "source_delay": int(result["source_delay"]),
        "source_h1_units": len(rows),
        "selected_target_counts": {
            str(delay): int(counts[delay]) for delay in DELAYS
        },
        "selected_target_rates": {
            str(delay): float(counts[delay] / len(rows))
            for delay in DELAYS
        },
        "exact_target_selection_rate": float(
            sum(row["exact_target_selection_correct"] for row in rows)
            / len(rows)
        ),
        "exact_history_selection_rate": float(
            sum(row["exact_history_selection_correct"] for row in rows)
            / len(rows)
        ),
        "matching_history_strict_win_rate": float(
            sum(row["matching_history_strict_win"] for row in rows)
            / len(rows)
        ),
        "mean_matching_history_loss": float(
            sum(row["matching_history_loss"] for row in rows) / len(rows)
        ),
        "mean_other_history_loss": float(
            sum(row["other_history_mean_loss"] for row in rows)
            / len(rows)
        ),
        "latent_alignment_h1": result["latent_alignment"]["h1"],
    }


def _aggregate_tracks(
    summaries: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    units = sum(row["source_h1_units"] for row in summaries.values())
    selected = Counter()
    for row in summaries.values():
        selected.update(
            {
                int(delay): int(count)
                for delay, count in row["selected_target_counts"].items()
            }
        )
    return {
        "source_h1_units": units,
        "selected_target_counts": {
            str(delay): int(selected[delay]) for delay in DELAYS
        },
        "selected_target_rates": {
            str(delay): float(selected[delay] / units) for delay in DELAYS
        },
        "exact_target_selection_rate": float(
            sum(
                row["exact_target_selection_rate"]
                * row["source_h1_units"]
                for row in summaries.values()
            )
            / units
        ),
        "exact_history_selection_rate": float(
            sum(
                row["exact_history_selection_rate"]
                * row["source_h1_units"]
                for row in summaries.values()
            )
            / units
        ),
        "matching_history_strict_win_rate": float(
            sum(
                row["matching_history_strict_win_rate"]
                * row["source_h1_units"]
                for row in summaries.values()
            )
            / units
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--model-family",
        choices=("lewm", "pldm"),
        required=True,
    )
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--stablewm-repo")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--track-scope",
        choices=tuple(TRACKS_BY_SCOPE),
        default="training_replay",
        help=(
            "Use replayed training-source queries or independent "
            "same-distribution loader-validation queries."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    checkpoint = args.checkpoint.expanduser().resolve()
    _require(checkpoint.is_file(), f"Checkpoint does not exist: {checkpoint}")
    checkpoint_config = checkpoint.parent / "config.json"
    _require(
        checkpoint_config.is_file(),
        f"Checkpoint config does not exist: {checkpoint_config}",
    )
    protocol = _checkpoint_protocol(checkpoint_config)
    _require(
        protocol
        == {
            "history_size": 7,
            "num_preds": 1,
            "frameskip": 5,
            "num_steps": 8,
            "action_encoder_input_dim": 10,
        },
        f"Checkpoint protocol is not History-7 Action Delay: {protocol}",
    )

    catalog_path = resolve_contextworld_path(
        config["source_identity"]["diagnostic_catalog"]["path"],
        repo_root=ROOT,
    )
    catalog = load_domain_catalog(catalog_path)
    normalizer = resolve_contextworld_path(
        config["source_identity"]["normalizer"]["path"],
        repo_root=ROOT,
    )
    adapter_class = (
        StableWorldModelLeWMHistory7Adapter
        if args.model_family == "lewm"
        else StableWorldModelPLDMHistory7DiagnosticAdapter
    )
    adapter = adapter_class.from_checkpoint(
        checkpoint,
        normalizer=normalizer,
        repo_root=ROOT,
        stablewm_repo=str(
            args.stablewm_repo or config["stable_worldmodel"]["repo"]
        ),
        stablewm_ref=str(config["stable_worldmodel"]["commit"]),
        device=args.device,
    )
    state_before = adapter.frozen_state_hash()
    summaries = {}
    tracks = TRACKS_BY_SCOPE[str(args.track_scope)]
    for track in tracks:
        assets = load_domain_track_assets(
            catalog,
            track=track,
            repo_root=ROOT,
        )
        result = score_domain_track(
            adapter,
            assets,
            batch_size=int(args.batch_size),
        )
        summaries[track] = _track_summary(result)
        print(f"[h7-checkpoint-diagnostic] completed {track}", flush=True)
    state_after = adapter.frozen_state_hash()
    _require(state_before == state_after, "Model state changed during scoring")

    output = args.output.expanduser().resolve()
    payload = {
        "schema_version": 1,
        "benchmark": (
            "tworoom_action_delay_history7_checkpoint_trajectory_diagnostic_v1"
        ),
        "status": "completed_post_hoc_diagnostic",
        "claim_boundary": {
            "changes_formal_benchmark_result": False,
            "track_scope": str(args.track_scope),
            "uses_training_replay_only": (
                args.track_scope == "training_replay"
            ),
            "uses_loader_validation_only": (
                args.track_scope == "loader_validation"
            ),
            "hidden_test_used": False,
            "formal_validation_used": False,
        },
        "label": str(args.label),
        "model_family": str(args.model_family),
        "identity": {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": file_sha256(checkpoint),
            "checkpoint_config": str(checkpoint_config),
            "checkpoint_config_sha256": file_sha256(checkpoint_config),
            "checkpoint_protocol": protocol,
            "frozen_catalog": str(catalog_path),
            "frozen_catalog_sha256": file_sha256(catalog_path),
            "normalizer": str(normalizer),
            "normalizer_sha256": file_sha256(normalizer),
            "entrypoint": str(Path(__file__).resolve()),
            "entrypoint_sha256": file_sha256(Path(__file__).resolve()),
        },
        "model": adapter.metadata,
        "model_state_sha256_before": state_before,
        "model_state_sha256_after": state_after,
        "tracks": summaries,
        "aggregate_source_h1": _aggregate_tracks(summaries),
    }
    write_json(output, payload)
    print(
        json.dumps(
            {
                "label": args.label,
                "model_family": args.model_family,
                "output": str(output),
                "aggregate_source_h1": payload["aggregate_source_h1"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
