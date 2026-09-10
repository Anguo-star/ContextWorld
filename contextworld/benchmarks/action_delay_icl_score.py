from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from contextworld.benchmarks.action_delay_icl_data import (
    DEFAULT_ACTION_DELAY_RELEASE_CONFIG,
    ActionDelayICLEvalDataset,
    load_action_delay_icl_release,
)
from contextworld.benchmarks.adapters import ActionDelayICLModelAdapter
from contextworld.benchmarks.paired_latent_response import (
    paired_latent_response_gate_checks,
    paired_latent_response_metrics,
    summarize_paired_latent_response_records,
)
from contextworld.benchmarks.test_gate_completion import (
    load_test_gate_completion_config,
    percentile_lower_bound,
    threshold_source_block,
    unit_bootstrap_draws,
)
from contextworld.evaluation.action_delay_h7_core import (
    DELAYS,
    summarize_action_delay_h1_physical,
)
from contextworld.evaluation.action_delay_h7_score import (
    physical_future_group,
    score_h7_validation_assets,
    summarize_h7_validation_records,
)
from contextworld.evaluation.action_delay_h7_validation import file_sha256
from contextworld.paths import repository_root


def _gate(
    core: dict[str, Any],
    *,
    release: dict[str, Any],
) -> dict[str, Any]:
    thresholds = release["scoring"]["primary_gate"]
    checks = {
        "physical_group_macro_accuracy": (
            core["physical_group_macro_accuracy"]
            >= float(
                thresholds[
                    "physical_group_macro_accuracy_minimum"
                ]
            )
        ),
        "minimum_physical_group_accuracy": (
            core["minimum_physical_group_accuracy"]
            >= float(
                thresholds["minimum_physical_group_accuracy"]
            )
        ),
        "bootstrap_lower_bound": (
            core["paired_query_bootstrap_95_percent_interval"]["lower"]
            >= float(
                thresholds[
                    "paired_query_bootstrap_95_percent_lower_bound_minimum"
                ]
            )
        ),
    }
    return {"checks": checks, "passed": all(checks.values())}


def action_delay_gate_completion_metrics(
    *,
    predicted_h1: np.ndarray,
    encoded_h1: np.ndarray,
    query_ids: list[str],
    history_strict_wins: np.ndarray,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Additive anti-shortcut gates for Action Delay at horizon 1.

    The frozen assets hold the true future of every query under each of the
    eleven counterfactual delays, and the model rolls out a prediction under
    each of the eleven delay histories.  Pairing two delays whose one-step
    physical futures are distinguishable (``physical_future_group`` at h1;
    delays 5..10 share one stationary future and are never paired with each
    other) reproduces the paired hidden-dynamics probe of the six fully gated
    components: swapping only the demonstrated history must move the
    prediction along the true inter-delay future displacement.

    ``history_strict_wins`` is the per-(query, target delay) matching-history
    strict-win matrix at h1, i.e. the correct-history decisions the existing
    scorer already computes.  All thresholds come from the additive
    gate-completion config; the frozen primary gate stays untouched.
    """

    section = config["action_delay"]
    gates = section["gates"]
    bootstrap = section["bootstrap"]
    predicted = np.asarray(predicted_h1, dtype=np.float64)
    encoded = np.asarray(encoded_h1, dtype=np.float64)
    if predicted.shape != encoded.shape or predicted.ndim != 3:
        raise ValueError(
            "Action Delay gate latents must share a (queries, delays, dim) shape:"
            f" {predicted.shape} vs {encoded.shape}"
        )
    if len(predicted) != len(query_ids) or len(set(query_ids)) != len(query_ids):
        raise ValueError("One unique query id per latent row is required")
    wins = np.asarray(history_strict_wins, dtype=bool)
    if wins.shape != (len(predicted), len(DELAYS)):
        raise ValueError(
            "History strict wins must be (queries, delays):"
            f" {wins.shape} vs {(len(predicted), len(DELAYS))}"
        )

    delay_pairs = [
        (delay_a, delay_b)
        for delay_a in DELAYS
        for delay_b in DELAYS
        if delay_a < delay_b
        and physical_future_group(delay_a, 1)
        != physical_future_group(delay_b, 1)
    ]
    response_records: list[dict[str, Any]] = []
    switch_unit_rows = []
    for query_index, query_id in enumerate(query_ids):
        pair_rows = []
        switch_values = []
        for delay_a, delay_b in delay_pairs:
            predicted_first = predicted[query_index, delay_a]
            predicted_second = predicted[query_index, delay_b]
            target_first = encoded[query_index, delay_a]
            target_second = encoded[query_index, delay_b]
            pair_rows.append(
                (
                    f"{query_id}:d{delay_a}>d{delay_b}",
                    predicted_first,
                    predicted_second,
                    target_first,
                    target_second,
                )
            )
            switch_values.append(
                bool(
                    np.sum(
                        (predicted_second - predicted_first)
                        * (target_second - target_first)
                    )
                    > 0.0
                )
            )
        _family_metrics, family_records = paired_latent_response_metrics(
            pair_ids=[row[0] for row in pair_rows],
            predicted_first=np.stack([row[1] for row in pair_rows]),
            predicted_second=np.stack([row[2] for row in pair_rows]),
            target_first=np.stack([row[3] for row in pair_rows]),
            target_second=np.stack([row[4] for row in pair_rows]),
        )
        response_records.extend(family_records)
        switch_unit_rows.append(
            [float(np.mean(switch_values)), float(wins[query_index].mean())]
        )
    latent_response = summarize_paired_latent_response_records(response_records)
    switch_rate = float(np.mean([row[0] for row in switch_unit_rows]))
    correct_history_rate = float(np.mean(wins))

    draws = unit_bootstrap_draws(
        np.asarray(switch_unit_rows, dtype=np.float64),
        resamples=int(bootstrap["resamples"]),
        random_seed=int(bootstrap["random_seed"]),
    )
    lower_bounds = percentile_lower_bound(
        draws,
        confidence_level=float(bootstrap["confidence_level"]),
    )
    uncertainty = {
        "method": "paired_query_bootstrap",
        "unit": str(bootstrap["unit"]),
        "resamples": int(bootstrap["resamples"]),
        "confidence_level": float(bootstrap["confidence_level"]),
        "random_seed": int(bootstrap["random_seed"]),
        "lower_bounds": {
            "context_switch_rate": float(lower_bounds[0]),
            "correct_history_rate": float(lower_bounds[1]),
        },
    }

    metrics = {
        "gate_horizon": int(section.get("gate_horizon", 1)),
        "queries": len(query_ids),
        "pair_count": len(query_ids) * len(delay_pairs),
        "delay_pairs": [f"{a}>{b}" for a, b in delay_pairs],
        "correct_history_rate": correct_history_rate,
        "correct_history_rate_definition": (
            "matching-history latent loss strictly below every"
            " physically distinguishable delay at the gate horizon"
        ),
        "context_switch_rate": switch_rate,
        "latent_response": latent_response,
    }
    checks = {
        "correct_history_rate": bool(
            correct_history_rate
            >= float(gates["correct_history_rate_minimum"])
        ),
        "context_switch_rate": bool(
            switch_rate >= float(gates["context_switch_rate_minimum"])
        ),
    }
    checks.update(
        paired_latent_response_gate_checks(metrics, thresholds=gates)
    )
    uncertainty_checks = {
        name: bool(
            uncertainty["lower_bounds"][name] >= float(minimum)
        )
        for name, minimum in dict(bootstrap["lower_bound_minimum"]).items()
    }
    return {
        "schema_version": 1,
        "component": "action_delay",
        "threshold_source": threshold_source_block(config),
        "threshold_provenance": str(section["threshold_provenance"]).strip(),
        "additive_only": True,
        "metrics": metrics,
        "uncertainty": uncertainty,
        "checks": {**checks, **uncertainty_checks},
        "passed": bool(all(checks.values()) and all(uncertainty_checks.values())),
    }


def _summaries(
    records: list[dict[str, Any]],
    *,
    release: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    summary = summarize_h7_validation_records(records)
    uncertainty = release["scoring"]["uncertainty"]
    core = summarize_action_delay_h1_physical(
        summary["by_horizon"]["1"]["query_metrics"],
        bootstrap_resamples=int(uncertainty["resamples"]),
        bootstrap_seed=int(uncertainty["random_seed"]),
    )
    return summary, core, _gate(core, release=release)


def evaluate_action_delay_icl_model(
    *,
    adapter: ActionDelayICLModelAdapter,
    model_name: str,
    training_recipe: str,
    training_seed: int | None,
    release_config: Path | str = DEFAULT_ACTION_DELAY_RELEASE_CONFIG,
    repo_root: Path | None = None,
    batch_size: int = 128,
    include_records: bool = True,
) -> dict[str, Any]:
    """Run the complete frozen 300-query Action Delay Public Test."""

    root = (repo_root or repository_root()).resolve()
    release = load_action_delay_icl_release(release_config)
    dataset = ActionDelayICLEvalDataset(
        release=release,
        repo_root=root,
    )
    if not dataset.is_full_protocol:
        raise RuntimeError(
            "Formal Action Delay scoring requires all 300 frozen queries"
        )
    before = adapter.frozen_state_hash()
    scored = score_h7_validation_assets(
        adapter,
        dataset.raw_assets,
        batch_size=int(batch_size),
        return_latents=True,
    )
    after = adapter.frozen_state_hash()
    if before != after:
        raise RuntimeError("Model state changed during Action Delay scoring")
    summary, core, gate = _summaries(
        scored["records"],
        release=release,
    )
    # The gate's history-use input compares the matching history only against
    # PHYSICALLY DISTINGUISHABLE alternatives.  At horizon 1
    # ``physical_future_group`` collapses delays 5..10 into one stationary
    # group, so the frozen ``matching_history_strict_win`` flag -- which
    # requires beating all ten other delays, including the five identical
    # twins -- is capped at (5 + 6 * 1/6) / 11 = 6/11 for a perfect model.
    # Scoring against same-group delays would measure a distinction the task
    # does not contain, so they are excluded here, exactly as ``delay_pairs``
    # already excludes them for the context-switch metric.  The frozen
    # summary's own ``matching_history_strict_win_rate`` is left untouched.
    h1_losses: dict[tuple[str, int], dict[int, float]] = {}
    for row in scored["records"]:
        if int(row["horizon"]) != 1:
            continue
        key = (str(row["query_id"]), int(row["target_delay"]))
        h1_losses.setdefault(key, {})[int(row["history_delay"])] = float(
            row["latent_mse"]
        )
    wins_by_query: dict[str, list[bool | None]] = {}
    for (query_id, target_delay), losses in h1_losses.items():
        matching = losses[target_delay]
        distinguishable = [
            loss
            for history_delay, loss in losses.items()
            if physical_future_group(history_delay, 1)
            != physical_future_group(target_delay, 1)
        ]
        if not distinguishable:
            raise RuntimeError(
                "No physically distinguishable alternative for delay"
                f" {target_delay}"
            )
        slots = wins_by_query.setdefault(query_id, [None] * len(DELAYS))
        slots[target_delay] = bool(matching < min(distinguishable))
    query_ids = [str(asset["query_id"]) for asset in dataset.raw_assets]
    history_strict_wins = np.asarray(
        [wins_by_query[query_id] for query_id in query_ids], dtype=bool
    )
    latents = scored["latents"]
    gate_completion = action_delay_gate_completion_metrics(
        predicted_h1=latents["predicted"][:, :, 0],
        encoded_h1=latents["encoded"][:, :, 0],
        query_ids=query_ids,
        history_strict_wins=history_strict_wins,
        config=load_test_gate_completion_config(),
    )
    release_path = Path(release["_config_path"])
    payload = {
        "schema_version": 1,
        "benchmark": "tworoom_history7_action_delay_icl_v1",
        "submission_kind": "single_checkpoint",
        "status": "completed",
        "release": {
            "release_id": release["release_id"],
            "release_config_sha256": file_sha256(release_path),
            "catalog_sha256": release["evaluation"][
                "catalog_sha256"
            ],
            "content_manifest_sha256": release["evaluation"][
                "content_manifest_sha256"
            ],
            "normalizer_sha256": release["evaluation"][
                "normalizer_sha256"
            ],
            "sealed_test_included": False,
        },
        "model": {
            "name": str(model_name),
            "training_recipe": str(training_recipe),
            "training_seed": (
                None if training_seed is None else int(training_seed)
            ),
            "adapter": adapter.metadata,
            "state_sha256_before": before,
            "state_sha256_after": after,
        },
        "data": dataset.describe(),
        "score_audit": scored["score_audit"],
        "core_h1": core,
        "gate": gate,
        "gate_completion": gate_completion,
    }
    if include_records:
        payload["records"] = scored["records"]
    return payload


def _load_and_rescore(
    path: Path,
    *,
    release: dict[str, Any],
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema_version") != 1
        or payload.get("benchmark")
        != "tworoom_history7_action_delay_icl_v1"
        or payload.get("submission_kind") != "single_checkpoint"
        or payload.get("status") != "completed"
    ):
        raise ValueError(f"Unsupported Action Delay result: {path}")
    expected_identity = {
        "release_id": release["release_id"],
        "release_config_sha256": file_sha256(
            Path(release["_config_path"])
        ),
        "catalog_sha256": release["evaluation"]["catalog_sha256"],
        "content_manifest_sha256": release["evaluation"][
            "content_manifest_sha256"
        ],
        "normalizer_sha256": release["evaluation"][
            "normalizer_sha256"
        ],
        "sealed_test_included": False,
    }
    if payload.get("release") != expected_identity:
        raise RuntimeError(f"Release identity mismatch: {path}")
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError(
            f"Independent rescoring requires retained records: {path}"
        )
    _, core, gate = _summaries(records, release=release)
    if core != payload.get("core_h1") or gate != payload.get("gate"):
        raise RuntimeError(f"Stored Action Delay score changed: {path}")
    return payload


def _stats(values: Iterable[float]) -> dict[str, float]:
    rows = [float(value) for value in values]
    if not rows:
        raise ValueError("Cannot summarize an empty metric")
    return {
        "mean": float(statistics.fmean(rows)),
        "sample_std": (
            float(statistics.stdev(rows)) if len(rows) > 1 else 0.0
        ),
        "minimum": float(min(rows)),
        "maximum": float(max(rows)),
    }


def score_action_delay_icl_results(
    *,
    result_paths: Iterable[Path | str],
    method_name: str,
    release_config: Path | str = DEFAULT_ACTION_DELAY_RELEASE_CONFIG,
) -> dict[str, Any]:
    """Independently rescore one checkpoint or one three-seed method."""

    release = load_action_delay_icl_release(release_config)
    paths = [Path(value).expanduser().resolve() for value in result_paths]
    if len(paths) not in {1, 3}:
        raise ValueError(
            "Provide one result for a descriptive checkpoint or three "
            "results for a method-level claim"
        )
    results = [
        _load_and_rescore(path, release=release) for path in paths
    ]
    checkpoint_hashes = [
        str(result["model"]["adapter"].get("checkpoint_sha256", ""))
        for result in results
    ]
    if (
        any(len(value) != 64 for value in checkpoint_hashes)
        or len(set(checkpoint_hashes)) != len(checkpoint_hashes)
    ):
        raise ValueError(
            "Every result must bind a distinct checkpoint SHA-256"
        )
    seeds = [result["model"]["training_seed"] for result in results]
    if len(paths) == 3:
        if any(seed is None for seed in seeds) or len(set(seeds)) != 3:
            raise ValueError(
                "A method score requires three distinct training seeds"
            )
        recipes = {
            str(result["model"]["training_recipe"])
            for result in results
        }
        adapters = {
            str(result["model"]["adapter"].get("adapter_id"))
            for result in results
        }
        if len(recipes) != 1 or len(adapters) != 1:
            raise ValueError(
                "A method score cannot mix recipes or adapter families"
            )
    per_checkpoint = [
        {
            "path": str(path),
            "checkpoint_sha256": result["model"]["adapter"][
                "checkpoint_sha256"
            ],
            "training_seed": result["model"]["training_seed"],
            "physical_group_macro_accuracy": result["core_h1"][
                "physical_group_macro_accuracy"
            ],
            "minimum_physical_group_accuracy": result["core_h1"][
                "minimum_physical_group_accuracy"
            ],
            "bootstrap_lower_bound": result["core_h1"][
                "paired_query_bootstrap_95_percent_interval"
            ]["lower"],
            "passed": bool(result["gate"]["passed"]),
        }
        for path, result in zip(paths, results, strict=True)
    ]
    formal_method = len(paths) == 3
    return {
        "schema_version": 1,
        "benchmark": "tworoom_history7_action_delay_icl_v1",
        "submission_kind": (
            "three_seed_method"
            if formal_method
            else "descriptive_checkpoint"
        ),
        "status": "completed",
        "method_name": str(method_name),
        "release_id": release["release_id"],
        "checkpoints": per_checkpoint,
        "aggregate": {
            metric: _stats(row[metric] for row in per_checkpoint)
            for metric in (
                "physical_group_macro_accuracy",
                "minimum_physical_group_accuracy",
                "bootstrap_lower_bound",
            )
        },
        "decision": {
            "passed": (
                all(row["passed"] for row in per_checkpoint)
                if formal_method
                else None
            ),
            "passed_checkpoints": sum(
                row["passed"] for row in per_checkpoint
            ),
            "required_checkpoints": 3 if formal_method else None,
            "claim": (
                "method_level_action_delay_icl"
                if formal_method
                else "descriptive_checkpoint_only"
            ),
        },
    }


__all__ = [
    "evaluate_action_delay_icl_model",
    "score_action_delay_icl_results",
]
