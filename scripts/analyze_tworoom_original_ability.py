#!/usr/bin/env python3
"""Analyze the preregistered four-model original-ability reconstruction study."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from contextworld.paths import resolve_contextworld_path
from contextworld.synthesis.manifest import write_json


EVAL_SEEDS = (42, 43, 44, 45, 46, 47)
MODEL_SLUGS = {
    "H3-OrigHeldout": "h3_origheldout_s3072",
    "H3-Synth5Matched": "h3_synth5matched_s3072",
    "H3-OrigPlusSynth5": "h3_origplus_synth5_s3072",
    "H3-SpeedFull": "h3_speedfull_s3072",
}
DOMAIN_STEMS = {
    "original_heldout": "planning_original_heldout",
    "speed5_matched": "planning_speed5_matched",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mean_ci(
    values: np.ndarray,
    *,
    rng: np.random.Generator,
    resamples: int,
    confidence: float,
) -> dict[str, float]:
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("Bootstrap input must be a non-empty vector")
    draws = rng.integers(0, len(values), size=(resamples, len(values)))
    means = values[draws].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return {
        "point": float(values.mean()),
        "ci_lower": float(np.quantile(means, alpha)),
        "ci_upper": float(np.quantile(means, 1.0 - alpha)),
    }


def _planning_paths(root: Path, model: str, domain: str) -> list[Path]:
    stem = DOMAIN_STEMS[domain]
    return [
        root / MODEL_SLUGS[model] / f"{stem}_s{seed}.json"
        for seed in EVAL_SEEDS
    ]


def _load_planning(paths: list[Path]) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    records: dict[str, dict[str, Any]] = {}
    provenance: dict[str, Any] = {"files": []}
    catalog_sha = None
    normalizer_sha = None
    checkpoint_sha = None
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "passed":
            raise ValueError(f"Planning result did not pass: {path}")
        observed_catalog = payload["catalog"]["sha256"]
        observed_normalizer = payload["normalizer"]["sha256"]
        observed_checkpoint = payload["checkpoint"]["sha256"]
        if catalog_sha is None:
            catalog_sha = observed_catalog
            normalizer_sha = observed_normalizer
            checkpoint_sha = observed_checkpoint
        if observed_catalog != catalog_sha:
            raise ValueError(f"Catalog changed across seeds for {path}")
        if observed_normalizer != normalizer_sha:
            raise ValueError(f"Normalizer changed across seeds for {path}")
        if observed_checkpoint != checkpoint_sha:
            raise ValueError(f"Checkpoint changed across seeds for {path}")
        provenance["files"].append(
            {"path": str(path), "sha256": _sha256(path)}
        )
        for record in payload["raw_records"]:
            key = str(record["evaluation_id"])
            if key in records:
                raise ValueError(f"Duplicate planning record: {key}")
            records[key] = record
    if len(records) != 300:
        raise ValueError(
            f"Expected 300 planning records, observed {len(records)}"
        )
    provenance.update(
        {
            "catalog_sha256": catalog_sha,
            "normalizer_sha256": normalizer_sha,
            "checkpoint_sha256": checkpoint_sha,
        }
    )
    return provenance, records


def _aggregate(records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    values = list(records.values())
    successes = sum(bool(record["success"]) for record in values)
    by_stratum: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in values:
        key = f"{record['stratum']}|{record['room_relation']}"
        by_stratum[key].append(record)
        by_seed[int(record["eval_seed"])].append(record)

    def summarize(selected: list[dict[str, Any]]) -> dict[str, Any]:
        selected_successes = sum(bool(row["success"]) for row in selected)
        return {
            "evaluations": len(selected),
            "successes": int(selected_successes),
            "success_rate": float(selected_successes / len(selected)),
            "mean_final_distance": float(
                np.mean([float(row["final_distance"]) for row in selected])
            ),
        }

    return {
        "evaluations": len(values),
        "successes": int(successes),
        "success_rate": float(successes / len(values)),
        "mean_final_distance": float(
            np.mean([float(record["final_distance"]) for record in values])
        ),
        "by_stratum": {
            key: summarize(selected)
            for key, selected in sorted(by_stratum.items())
        },
        "by_seed": {
            str(seed): summarize(selected)
            for seed, selected in sorted(by_seed.items())
        },
    }


def _paired_planning_comparison(
    reference: dict[str, dict[str, Any]],
    candidate: dict[str, dict[str, Any]],
    *,
    bootstrap_seed: int,
    resamples: int,
    confidence: float,
    success_margin: float,
    distance_margin: float,
) -> dict[str, Any]:
    if set(reference) != set(candidate):
        raise ValueError(
            "Planning evaluation IDs differ across paired models: "
            f"reference_only={len(set(reference) - set(candidate))}, "
            f"candidate_only={len(set(candidate) - set(reference))}"
        )
    keys = sorted(reference)
    paired_fields = (
        "eval_seed",
        "evaluation_index",
        "source_kind",
        "source_path",
        "episode",
        "start_step",
        "goal_offset",
        "cem_group_seed",
        "stratum",
        "room_relation",
        "initial_state",
        "goal_state",
    )
    for key in keys:
        mismatches = [
            field
            for field in paired_fields
            if reference[key][field] != candidate[key][field]
        ]
        if mismatches:
            raise ValueError(
                f"Paired planning metadata mismatch for {key}: {mismatches}"
            )

    success_deltas = np.asarray(
        [
            float(bool(candidate[key]["success"]))
            - float(bool(reference[key]["success"]))
            for key in keys
        ],
        dtype=np.float64,
    )
    distance_deltas = np.asarray(
        [
            float(candidate[key]["final_distance"])
            - float(reference[key]["final_distance"])
            for key in keys
        ],
        dtype=np.float64,
    )
    success_ci = _mean_ci(
        success_deltas,
        rng=np.random.default_rng(bootstrap_seed),
        resamples=resamples,
        confidence=confidence,
    )
    distance_ci = _mean_ci(
        distance_deltas,
        rng=np.random.default_rng(bootstrap_seed ^ 0xD157A),
        resamples=resamples,
        confidence=confidence,
    )

    strata: dict[str, list[str]] = defaultdict(list)
    for key in keys:
        record = reference[key]
        strata[f"{record['stratum']}|{record['room_relation']}"].append(key)
    stratum_results = {}
    collapsed = []
    for stratum, selected_keys in sorted(strata.items()):
        reference_successes = sum(
            bool(reference[key]["success"]) for key in selected_keys
        )
        candidate_successes = sum(
            bool(candidate[key]["success"]) for key in selected_keys
        )
        is_collapsed = reference_successes > 0 and candidate_successes == 0
        if is_collapsed:
            collapsed.append(stratum)
        stratum_results[stratum] = {
            "evaluations": len(selected_keys),
            "reference_successes": int(reference_successes),
            "candidate_successes": int(candidate_successes),
            "collapsed": is_collapsed,
        }

    gates = {
        "success_rate_non_inferior": (
            success_ci["ci_lower"] >= success_margin
        ),
        "final_distance_non_inferior": (
            distance_ci["ci_upper"] <= distance_margin
        ),
        "no_solvable_stratum_collapse": not collapsed,
    }
    return {
        "evaluations": len(keys),
        "candidate_minus_reference_success_rate": success_ci,
        "candidate_minus_reference_final_distance_px": distance_ci,
        "margins": {
            "success_rate": success_margin,
            "final_distance_px": distance_margin,
        },
        "strata": stratum_results,
        "collapsed_solvable_strata": collapsed,
        "gates": gates,
        "passed": all(gates.values()),
    }


def _load_rollout(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "passed":
        raise ValueError(f"Rollout result did not pass: {path}")
    records = {
        str(record["evaluation_id"]): record
        for record in payload["raw_records"]
    }
    if len(records) != 600:
        raise ValueError(f"Expected 600 rollout records in {path}")
    return records


def _rollout_comparison(
    reference: dict[str, dict[str, Any]],
    candidate: dict[str, dict[str, Any]],
    *,
    bootstrap_seed: int,
    resamples: int,
    confidence: float,
) -> list[dict[str, Any]]:
    if set(reference) != set(candidate):
        raise ValueError("Rollout evaluation IDs differ across paired models")
    output = []
    domains = sorted({record["domain"] for record in reference.values()})
    for domain in domains:
        keys = sorted(
            key
            for key, record in reference.items()
            if record["domain"] == domain
        )
        for horizon in (1, 2, 3, 5):
            result: dict[str, Any] = {
                "domain": domain,
                "horizon_action_blocks": horizon,
                "evaluations": len(keys),
            }
            for offset, metric in enumerate(
                ("latent_mse", "latent_rmse", "latent_cosine_distance")
            ):
                deltas = np.asarray(
                    [
                        float(candidate[key]["horizons"][str(horizon)][metric])
                        - float(
                            reference[key]["horizons"][str(horizon)][metric]
                        )
                        for key in keys
                    ],
                    dtype=np.float64,
                )
                result[f"candidate_minus_reference_{metric}"] = _mean_ci(
                    deltas,
                    rng=np.random.default_rng(
                        bootstrap_seed
                        ^ (horizon << 8)
                        ^ (offset << 16)
                        ^ sum(map(ord, domain))
                    ),
                    resamples=resamples,
                    confidence=confidence,
                )
            output.append(result)
    return output


def _context_evaluation_summary(model_dir: Path) -> dict[str, Any]:
    e1_path = model_dir / "e1_speed_paired.json"
    e4_path = model_dir / "e4_speed_ctx_n50x6.json"
    no_context_paths = [
        model_dir / f"e4_speed_noctx_n50_s{seed}.json"
        for seed in EVAL_SEEDS
    ]
    e1 = json.loads(e1_path.read_text(encoding="utf-8"))
    e4 = json.loads(e4_path.read_text(encoding="utf-8"))
    if e1.get("status") != "passed" or e4.get("status") != "passed":
        raise ValueError(f"E1/E4 result did not pass under {model_dir}")

    def e1_row(condition: str, budget: int) -> dict[str, Any]:
        matches = [
            row
            for row in e1["aggregates"]
            if row["condition"] == condition
            and int(row["context_budget"]) == budget
            and row["family"] == "speed"
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Expected one E1 row condition={condition}, budget={budget}"
            )
        row = matches[0]
        return {
            "latent_mse": row["latent_mse"],
            "latent_cosine_distance": row["latent_cosine_distance"],
            "counterfactual_accuracy": row["counterfactual_accuracy"],
        }

    no_context_records = []
    no_context_provenance = []
    for path in no_context_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "passed":
            raise ValueError(f"E4 no-context result did not pass: {path}")
        no_context_records.extend(payload["records"])
        no_context_provenance.append(
            {"path": str(path), "sha256": _sha256(path)}
        )
    if len(no_context_records) != 300:
        raise ValueError(
            f"Expected 300 E4 no-context records under {model_dir}"
        )
    no_context_ids = {
        (record["evaluation_id"], int(record["cem_seed"]))
        for record in no_context_records
    }
    paired_ids = set()
    for raw_path in e4["protocol"]["raw_results"]:
        raw = json.loads(Path(raw_path).read_text(encoding="utf-8"))
        paired_ids.update(
            (record["evaluation_id"], int(record["cem_seed"]))
            for record in raw["records"]
            if record["condition"] == "correct"
        )
    if no_context_ids != paired_ids:
        raise ValueError(
            f"E4 no-context schedule is not paired under {model_dir}"
        )
    no_context_successes = sum(
        bool(record["success"]) for record in no_context_records
    )
    no_context_distance = float(
        np.mean(
            [float(record["final_distance"]) for record in no_context_records]
        )
    )
    paired = e4["aggregate"]
    correct_rate = float(paired["correct"]["pooled_success_rate"])
    wrong_rate = float(paired["wrong"]["pooled_success_rate"])
    no_context_rate_points = 100.0 * no_context_successes / 300
    sign_p = float(paired["paired_sign_test"]["two_sided_p_value"])
    return {
        "provenance": {
            "e1": {"path": str(e1_path), "sha256": _sha256(e1_path)},
            "e4_paired": {"path": str(e4_path), "sha256": _sha256(e4_path)},
            "e4_no_context": no_context_provenance,
        },
        "one_step_prediction": {
            "no_context": e1_row("none", 0),
            "correct_context_k2": e1_row("correct", 2),
            "wrong_context_k2": e1_row("wrong", 2),
            "diagnostic_signals": e1["diagnostic_signals"],
        },
        "planning": {
            "no_context": {
                "evaluations": 300,
                "successes": int(no_context_successes),
                "success_rate_points": no_context_rate_points,
                "mean_final_distance": no_context_distance,
            },
            "correct_context": {
                **paired["correct"],
                "mean_final_distance": float(
                    np.mean(
                        [
                            record["final_distance"]
                            for raw_path in e4["protocol"]["raw_results"]
                            for record in json.loads(
                                Path(raw_path).read_text(encoding="utf-8")
                            )["records"]
                            if record["condition"] == "correct"
                        ]
                    )
                ),
            },
            "wrong_context": {
                **paired["wrong"],
                "mean_final_distance": float(
                    np.mean(
                        [
                            record["final_distance"]
                            for raw_path in e4["protocol"]["raw_results"]
                            for record in json.loads(
                                Path(raw_path).read_text(encoding="utf-8")
                            )["records"]
                            if record["condition"] == "wrong"
                        ]
                    )
                ),
            },
            "correct_minus_wrong_success_rate_points": float(
                paired["correct_minus_wrong_success_rate_points"]
            ),
            "correct_minus_no_context_success_rate_points": (
                correct_rate - no_context_rate_points
            ),
            "wrong_minus_no_context_success_rate_points": (
                wrong_rate - no_context_rate_points
            ),
            "paired_sign_test_two_sided_p_value": sign_p,
            "planning_icl_positive": (
                correct_rate > wrong_rate and sign_p < 0.05
            ),
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = resolve_contextworld_path(args.root, repo_root=REPO_ROOT)
    output = resolve_contextworld_path(args.output, repo_root=REPO_ROOT)
    planning: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    provenance: dict[str, dict[str, Any]] = {}
    aggregates: dict[str, dict[str, Any]] = {}
    for model in MODEL_SLUGS:
        planning[model] = {}
        provenance[model] = {}
        aggregates[model] = {}
        for domain in DOMAIN_STEMS:
            model_provenance, records = _load_planning(
                _planning_paths(root, model, domain)
            )
            planning[model][domain] = records
            provenance[model][domain] = model_provenance
            aggregates[model][domain] = _aggregate(records)

    comparisons: dict[str, Any] = {}
    comparison_pairs = {
        "synth5matched_vs_origheldout": (
            "H3-OrigHeldout",
            "H3-Synth5Matched",
        ),
        "origplus_synth5_vs_origheldout": (
            "H3-OrigHeldout",
            "H3-OrigPlusSynth5",
        ),
        "speedfull_vs_origheldout": (
            "H3-OrigHeldout",
            "H3-SpeedFull",
        ),
        "speedfull_vs_origplus_synth5": (
            "H3-OrigPlusSynth5",
            "H3-SpeedFull",
        ),
    }
    for name, (reference_model, candidate_model) in comparison_pairs.items():
        comparisons[name] = {
            "reference": reference_model,
            "candidate": candidate_model,
            "domains": {
                domain: _paired_planning_comparison(
                    planning[reference_model][domain],
                    planning[candidate_model][domain],
                    bootstrap_seed=args.bootstrap_seed
                    ^ sum(map(ord, name + domain)),
                    resamples=args.bootstrap_resamples,
                    confidence=args.confidence,
                    success_margin=args.success_margin,
                    distance_margin=args.distance_margin,
                )
                for domain in DOMAIN_STEMS
            },
        }
        comparisons[name]["all_domains_passed"] = all(
            result["passed"]
            for result in comparisons[name]["domains"].values()
        )

    rollout: dict[str, Any] = {}
    rollout_paths = {
        model: root / slug / "rollout_error.json"
        for model, slug in MODEL_SLUGS.items()
    }
    if all(path.is_file() for path in rollout_paths.values()):
        rollout_records = {
            model: _load_rollout(path)
            for model, path in rollout_paths.items()
        }
        for name, (reference_model, candidate_model) in comparison_pairs.items():
            rollout[name] = _rollout_comparison(
                rollout_records[reference_model],
                rollout_records[candidate_model],
                bootstrap_seed=args.bootstrap_seed ^ sum(map(ord, name)),
                resamples=args.bootstrap_resamples,
                confidence=args.confidence,
            )

    context_paths = [
        root / slug / filename
        for slug in MODEL_SLUGS.values()
        for filename in (
            "e1_speed_paired.json",
            "e4_speed_ctx_n50x6.json",
            *[
                f"e4_speed_noctx_n50_s{seed}.json"
                for seed in EVAL_SEEDS
            ],
        )
    ]
    context_evaluations = {}
    if all(path.is_file() for path in context_paths):
        context_evaluations = {
            model: _context_evaluation_summary(root / slug)
            for model, slug in MODEL_SLUGS.items()
        }

    reconstruction_passed = comparisons[
        "synth5matched_vs_origheldout"
    ]["all_domains_passed"]
    mixture_passed = comparisons[
        "origplus_synth5_vs_origheldout"
    ]["all_domains_passed"]
    payload = {
        "schema_version": 1,
        "benchmark": "tworoom_original_ability_reconstruction_v1",
        "status": "complete",
        "protocol": {
            "eval_seeds": list(EVAL_SEEDS),
            "evaluations_per_seed_per_domain": 50,
            "paired_bootstrap_seed": args.bootstrap_seed,
            "paired_bootstrap_resamples": args.bootstrap_resamples,
            "confidence_level": args.confidence,
            "success_rate_margin": args.success_margin,
            "final_distance_margin_px": args.distance_margin,
            "stratum_collapse_definition": (
                "candidate has zero successes in a stratum where the "
                "reference has at least one success"
            ),
        },
        "provenance": provenance,
        "model_aggregates": aggregates,
        "paired_planning_comparisons": comparisons,
        "paired_rollout_comparisons": rollout,
        "context_evaluations": context_evaluations,
        "formal_decisions": {
            "synthetic_only_reconstructs_original_ability": {
                "passed": reconstruction_passed,
                "interpretation": (
                    "preregistered non-inferiority established"
                    if reconstruction_passed
                    else "preregistered non-inferiority not established"
                ),
            },
            "mixed_training_preserves_original_ability": {
                "passed": mixture_passed,
                "interpretation": (
                    "no material interference under preregistered margins"
                    if mixture_passed
                    else "preservation criterion not established"
                ),
            },
        },
    }
    write_json(output, payload)
    return payload


def parse_args() -> argparse.Namespace:
    root = Path(
        "artifacts/evaluation/history3/original_ability_reconstruction"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "original_ability_reconstruction_n50x6.json",
    )
    parser.add_argument("--bootstrap-seed", type=int, default=3072)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--success-margin", type=float, default=-0.05)
    parser.add_argument("--distance-margin", type=float, default=5.0)
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps(result["formal_decisions"], indent=2, sort_keys=True))
