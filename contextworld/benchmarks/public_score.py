"""Minimal, model-facing result contract for the public benchmark table.

Component scorers intentionally retain richer diagnostics.  This module does
not replace those scorers or their frozen gates; it creates the small view a
benchmark reader normally needs:

* one ICL ability metric and the frozen PASS/FAIL decision;
* stability across the required independent training seeds;
* one separate original-task-retention result, or an explicit reason why it
  is not applicable or has not yet been evaluated.

Scores from different components have different meanings and are therefore
never averaged into a suite-wide percentage.
"""

from __future__ import annotations

import statistics
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Literal

import numpy as np


EvidenceScope = Literal["behavioral", "training_attributed"]
RetentionResult = Literal["PASS", "FAIL", "N/A", "NOT_EVALUATED"]

_EVIDENCE_SCOPES = {"behavioral", "training_attributed"}
_RETENTION_RESULTS = {"PASS", "FAIL", "N/A", "NOT_EVALUATED"}
_UNSCORED_RETENTION_RESULTS = {"N/A", "NOT_EVALUATED"}


def _finite_fraction(value: float, *, name: str) -> float:
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be a finite fraction in [0, 1]")
    return number


def _fraction_summary(values: Iterable[float]) -> dict[str, Any]:
    rows = [
        _finite_fraction(value, name="primary metric") for value in values
    ]
    if not rows:
        raise ValueError("At least one primary-metric value is required")
    return {
        "mean": float(statistics.fmean(rows)),
        "minimum": float(min(rows)),
        "maximum": float(max(rows)),
        "unit": "fraction",
    }


def _validate_uncertainty(
    uncertainty: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if uncertainty is None:
        return None
    kind = str(uncertainty.get("kind", ""))
    if kind != "paired_bootstrap_95_percent_interval":
        raise ValueError(
            "Public uncertainty must be a paired-query bootstrap 95% interval"
        )
    lower = _finite_fraction(uncertainty.get("lower"), name="lower bound")
    upper = _finite_fraction(uncertainty.get("upper"), name="upper bound")
    if lower > upper:
        raise ValueError("Uncertainty lower bound exceeds upper bound")
    return {"kind": kind, "lower": lower, "upper": upper}


def make_retention_result(
    *,
    result: RetentionResult,
    reason: str | None = None,
    metric_id: str | None = None,
    metric_label: str | None = None,
    per_seed_values: Sequence[float] | None = None,
    baseline_value: float | None = None,
) -> dict[str, Any]:
    """Build the separate original-task/CEM part of a public result.

    ``N/A`` means that no meaningful original-task retention track exists.
    ``NOT_EVALUATED`` means that a retention track is applicable but has not
    been completed.  Both require an explanation and neither counts as PASS.
    """

    if result not in _RETENTION_RESULTS:
        raise ValueError(f"Unsupported retention result: {result!r}")
    if result in _UNSCORED_RETENTION_RESULTS:
        if not reason or not str(reason).strip():
            raise ValueError(f"A {result} retention result requires a reason")
        if any(
            value is not None
            for value in (
                metric_id,
                metric_label,
                per_seed_values,
                baseline_value,
            )
        ):
            raise ValueError(
                f"A {result} retention result cannot contain a score"
            )
        return {"result": result, "reason": str(reason)}

    if not metric_id or not metric_label or per_seed_values is None:
        raise ValueError("PASS/FAIL retention requires one named metric")
    values = list(per_seed_values)
    if not values:
        raise ValueError("Retention requires at least one evaluated checkpoint")
    metric = {
        "id": str(metric_id),
        "label": str(metric_label),
        **_fraction_summary(values),
    }
    if baseline_value is not None:
        metric["baseline"] = _finite_fraction(
            baseline_value, name="retention baseline"
        )
    return {
        "result": result,
        "primary_metric": metric,
        "evaluated_checkpoints": len(values),
    }


def _validate_retention_result(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = payload.get("result")
    if result not in _RETENTION_RESULTS:
        raise ValueError("Invalid original-task-retention result")
    if result in _UNSCORED_RETENTION_RESULTS:
        unknown = set(payload) - {"result", "reason"}
        if unknown:
            raise ValueError(
                f"A {result} retention result cannot contain a score: "
                f"{sorted(unknown)}"
            )
        reason = payload.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"A {result} retention result requires a reason")
        if "primary_metric" in payload:
            raise ValueError(
                f"A {result} retention result cannot contain a score"
            )
        return dict(payload)
    metric = payload.get("primary_metric")
    if not isinstance(metric, Mapping):
        raise ValueError("PASS/FAIL retention requires one primary metric")
    for key in ("id", "label", "mean", "minimum", "maximum", "unit"):
        if key not in metric:
            raise ValueError(f"Retention primary metric is missing {key!r}")
    if metric["unit"] != "fraction":
        raise ValueError("Retention primary metric must use fraction units")
    for key in ("mean", "minimum", "maximum"):
        _finite_fraction(metric[key], name=f"retention {key}")
    return dict(payload)


def make_component_result(
    *,
    component_id: str,
    component_name: str,
    method_name: str,
    primary_metric_id: str,
    primary_metric_label: str,
    per_seed_primary_values: Sequence[float],
    per_seed_gate_passes: Sequence[bool],
    ability_passed: bool,
    required_training_seeds: int,
    evidence_scope: EvidenceScope,
    original_task_retention: Mapping[str, Any],
    uncertainty: Mapping[str, Any] | None = None,
    training_attribution: Mapping[str, Any] | None = None,
    diagnostics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create one canonical component result with public and diagnostic views.

    ``behavioral`` means that the checkpoint responds correctly to the history
    intervention.  ``training_attributed`` additionally means that matched
    models trained without the varying factor were compared seed by seed.
    """

    if not component_id.strip() or not component_name.strip():
        raise ValueError("Component id and name must be non-empty")
    if not method_name.strip():
        raise ValueError("Method name must be non-empty")
    if evidence_scope not in _EVIDENCE_SCOPES:
        raise ValueError(f"Unsupported evidence scope: {evidence_scope!r}")
    if required_training_seeds <= 0:
        raise ValueError("required_training_seeds must be positive")

    values = list(per_seed_primary_values)
    seed_passes = [bool(value) for value in per_seed_gate_passes]
    if len(values) != len(seed_passes):
        raise ValueError("Metric values and seed decisions must have equal size")
    if len(values) != int(required_training_seeds):
        raise ValueError(
            "A formal public result must contain exactly the required "
            "independent training seeds"
        )
    stable = all(seed_passes)
    if ability_passed and not stable:
        raise ValueError(
            "A method cannot pass when one or more required training seeds fail"
        )

    attribution: dict[str, Any] | None = None
    if evidence_scope == "training_attributed":
        if training_attribution is None:
            raise ValueError(
                "A training-attributed claim requires matched control evidence"
            )
        attribution = dict(training_attribution)
        effects_favor_target = attribution.get(
            "all_paired_effects_favor_target"
        )
        if (
            attribution.get("control_kind")
            != "matched_no_factor_training_control"
            or int(attribution.get("paired_training_seeds", -1))
            != int(required_training_seeds)
            or not isinstance(effects_favor_target, bool)
        ):
            raise ValueError(
                "Training attribution requires matched no-factor controls, "
                "the required paired seeds, and a per-seed effect decision"
            )
        if ability_passed and not effects_favor_target:
            raise ValueError(
                "A training-attributed pass requires positive effects for "
                "every paired training seed"
            )
    elif training_attribution is not None:
        raise ValueError(
            "Control evidence must use evidence_scope='training_attributed'"
        )

    retention = _validate_retention_result(original_task_retention)

    public_metric_text = f"{primary_metric_id} {primary_metric_label}".lower()
    normalized_metric_text = public_metric_text.replace("_", " ").replace(
        "-", " "
    )
    if "latent" in normalized_metric_text and any(
        forbidden in normalized_metric_text for forbidden in ("mse", "loss")
    ):
        raise ValueError(
            "Raw latent loss is a within-checkpoint diagnostic, not a public "
            "cross-model ability metric"
        )
    metric = {
        "id": str(primary_metric_id),
        "label": str(primary_metric_label),
        **_fraction_summary(values),
    }
    interval = _validate_uncertainty(uncertainty)
    if interval is not None:
        metric["uncertainty"] = interval

    result_word = "PASS" if ability_passed else "FAIL"
    claim = (
        "training_attributed_icl_demonstrated"
        if ability_passed and evidence_scope == "training_attributed"
        else (
            "behavioral_icl_demonstrated"
            if ability_passed
            else "icl_not_demonstrated"
        )
    )
    public = {
        "component_id": str(component_id),
        "component_name": str(component_name),
        "method_name": str(method_name),
        "icl_ability": {
            "result": result_word,
            "primary_metric": metric,
            "training_seed_stability": {
                "passed_checkpoints": sum(seed_passes),
                "evaluated_checkpoints": len(seed_passes),
                "required_checkpoints": int(required_training_seeds),
                "all_required_seeds_passed": stable,
            },
            "evidence_scope": evidence_scope,
            "claim": claim,
        },
        "original_task_retention": retention,
    }
    internal = dict(diagnostics or {})
    if attribution is not None:
        internal["training_attribution"] = attribution
    return {
        "schema_version": 1,
        "result_kind": "contextworld_component_result",
        "public": public,
        "diagnostics": internal,
    }


def make_public_scoreboard(
    component_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return a compact suite view without inventing a cross-task average."""

    rows: list[dict[str, Any]] = []
    identities: set[tuple[str, str]] = set()
    for result in component_results:
        if (
            result.get("schema_version") != 1
            or result.get("result_kind")
            != "contextworld_component_result"
        ):
            raise ValueError("Unsupported component result contract")
        public = dict(result["public"])
        identity = (
            str(public["component_id"]),
            str(public["method_name"]),
        )
        if identity in identities:
            raise ValueError(f"Duplicate component/method result: {identity}")
        identities.add(identity)
        rows.append(public)
    rows.sort(key=lambda row: (row["component_id"], row["method_name"]))
    return {
        "schema_version": 1,
        "result_kind": "contextworld_public_scoreboard",
        "aggregation_policy": (
            "Each component is judged independently; component metrics are "
            "not averaged into a suite-wide score."
        ),
        "component_results": rows,
    }


def make_public_scoreboard_from_spec(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Generate a scoreboard from manually supplied formal seed results.

    This is deliberately a presentation adapter, not a scorer.  Native
    component scorers remain responsible for producing calibrated per-query
    correctness and frozen gate decisions.  The input has no decoder or raw
    latent-loss field and the returned scoreboard contains no diagnostics.
    """

    if (
        payload.get("schema_version") != 1
        or payload.get("result_kind")
        != "contextworld_public_scoreboard_spec"
    ):
        raise ValueError("Unsupported public scoreboard input spec")
    components = payload.get("components")
    if not isinstance(components, Sequence) or isinstance(
        components, (str, bytes)
    ) or not components:
        raise ValueError("Scoreboard spec requires at least one component")

    results = []
    allowed_component_keys = {
        "component_id",
        "component_name",
        "method_name",
        "primary_metric",
        "per_seed_gate_passes",
        "ability_passed",
        "required_training_seeds",
        "evidence_scope",
        "training_attribution",
        "original_task_retention",
    }
    for index, raw_component in enumerate(components):
        if not isinstance(raw_component, Mapping):
            raise ValueError(f"Component {index} must be an object")
        unknown = set(raw_component) - allowed_component_keys
        if unknown:
            raise ValueError(
                f"Component {index} contains unsupported fields: "
                f"{sorted(unknown)}"
            )
        raw_metric = raw_component.get("primary_metric")
        if not isinstance(raw_metric, Mapping):
            raise ValueError(f"Component {index} has no primary metric")
        allowed_metric_keys = {
            "id",
            "label",
            "per_seed_values",
            "uncertainty",
        }
        metric_unknown = set(raw_metric) - allowed_metric_keys
        if metric_unknown:
            raise ValueError(
                f"Component {index} primary metric contains unsupported "
                f"fields: {sorted(metric_unknown)}"
            )
        raw_retention = raw_component.get("original_task_retention")
        if not isinstance(raw_retention, Mapping):
            raise ValueError(
                f"Component {index} has no original-task-retention result"
            )
        retention_result = str(raw_retention.get("result", ""))
        if retention_result in _UNSCORED_RETENTION_RESULTS:
            retention_unknown = set(raw_retention) - {"result", "reason"}
            if retention_unknown:
                raise ValueError(
                    f"Component {index} {retention_result} retention cannot "
                    f"contain a score: {sorted(retention_unknown)}"
                )
            retention = make_retention_result(
                result=retention_result,  # type: ignore[arg-type]
                reason=raw_retention.get("reason"),
            )
        else:
            allowed_retention_keys = {
                "result",
                "metric_id",
                "metric_label",
                "per_seed_values",
                "baseline_value",
            }
            retention_unknown = set(raw_retention) - allowed_retention_keys
            if retention_unknown:
                raise ValueError(
                    f"Component {index} retention contains unsupported "
                    f"fields: {sorted(retention_unknown)}"
                )
            retention = make_retention_result(
                result=retention_result,  # type: ignore[arg-type]
                metric_id=raw_retention.get("metric_id"),
                metric_label=raw_retention.get("metric_label"),
                per_seed_values=raw_retention.get("per_seed_values"),
                baseline_value=raw_retention.get("baseline_value"),
            )
        results.append(
            make_component_result(
                component_id=str(raw_component.get("component_id", "")),
                component_name=str(raw_component.get("component_name", "")),
                method_name=str(raw_component.get("method_name", "")),
                primary_metric_id=str(raw_metric.get("id", "")),
                primary_metric_label=str(raw_metric.get("label", "")),
                per_seed_primary_values=raw_metric.get(
                    "per_seed_values", []
                ),
                per_seed_gate_passes=raw_component.get(
                    "per_seed_gate_passes", []
                ),
                ability_passed=bool(raw_component.get("ability_passed")),
                required_training_seeds=int(
                    raw_component.get("required_training_seeds", 0)
                ),
                evidence_scope=str(  # type: ignore[arg-type]
                    raw_component.get("evidence_scope", "")
                ),
                original_task_retention=retention,
                uncertainty=raw_metric.get("uncertainty"),
                training_attribution=raw_component.get(
                    "training_attribution"
                ),
            )
        )
    return make_public_scoreboard(results)


def _calibrated_query_submission(
    payload: Mapping[str, Any],
    *,
    side: str,
) -> tuple[str, str, str, str, dict[str, int]]:
    component_id = str(payload.get("component_id", "")).strip()
    public_test_id = str(payload.get("public_test_id", "")).strip()
    public_test_sha256 = str(payload.get("public_test_sha256", "")).strip()
    model_name = str(payload.get("model_name", "")).strip()
    if not component_id or not public_test_id or not model_name:
        raise ValueError(
            f"{side} must identify its component, Public Test, and model"
        )
    if (
        len(public_test_sha256) != 64
        or any(character not in "0123456789abcdef" for character in public_test_sha256)
    ):
        raise ValueError(f"{side} Public Test SHA-256 is invalid")
    if payload.get("decision_metric") != "calibrated_icl_correct":
        raise ValueError(
            "Cross-model comparison accepts only calibrated 0/1 ICL "
            "decisions; raw latent MSE is not comparable across models"
        )
    raw_decisions = payload.get("query_decisions")
    if not isinstance(raw_decisions, Mapping) or not raw_decisions:
        raise ValueError(f"{side} query_decisions must be a non-empty mapping")
    decisions: dict[str, int] = {}
    for raw_query_id, raw_value in raw_decisions.items():
        query_id = str(raw_query_id).strip()
        if not query_id:
            raise ValueError(f"{side} contains an empty query id")
        if raw_value not in (0, 1):
            raise ValueError(
                f"{side} query {query_id!r} is not a calibrated 0/1 decision"
            )
        decisions[query_id] = int(raw_value)
    return (
        component_id,
        public_test_id,
        public_test_sha256,
        model_name,
        decisions,
    )


def compare_paired_query_decisions(
    *,
    model_a: Mapping[str, Any],
    model_b: Mapping[str, Any],
    bootstrap_resamples: int = 10_000,
    bootstrap_seed: int = 20260804,
) -> dict[str, Any]:
    """Compare two models on identical calibrated Public Test decisions.

    The comparison is intentionally query-paired.  Model A is called superior
    only when the 95% bootstrap lower bound of ``accuracy(A)-accuracy(B)`` is
    strictly positive.  Callers must first convert each task's native score to
    its frozen per-query 0/1 correctness decision; latent distances themselves
    are never compared across models.
    """

    if int(bootstrap_resamples) <= 0:
        raise ValueError("bootstrap_resamples must be positive")
    a = _calibrated_query_submission(model_a, side="model_a")
    b = _calibrated_query_submission(model_b, side="model_b")
    a_component, a_test, a_hash, a_name, a_decisions = a
    b_component, b_test, b_hash, b_name, b_decisions = b
    if a_component != b_component:
        raise ValueError(
            "Cross-component model comparison is not allowed: "
            f"{a_component!r} != {b_component!r}"
        )
    if a_test != b_test or a_hash != b_hash:
        raise ValueError("Models must use the same frozen Public Test release")
    if a_name == b_name:
        raise ValueError("Compared model names must be distinct")
    if set(a_decisions) != set(b_decisions):
        only_a = sorted(set(a_decisions) - set(b_decisions))
        only_b = sorted(set(b_decisions) - set(a_decisions))
        raise ValueError(
            "Paired comparison requires identical query ids; "
            f"only_model_a={only_a[:3]}, only_model_b={only_b[:3]}"
        )

    query_ids = sorted(a_decisions)
    a_values = np.asarray(
        [a_decisions[query_id] for query_id in query_ids], dtype=np.float64
    )
    b_values = np.asarray(
        [b_decisions[query_id] for query_id in query_ids], dtype=np.float64
    )
    paired_differences = a_values - b_values
    rng = np.random.default_rng(int(bootstrap_seed))
    bootstrap = np.empty(int(bootstrap_resamples), dtype=np.float64)
    for start in range(0, len(bootstrap), 1_000):
        count = min(1_000, len(bootstrap) - start)
        indices = rng.integers(
            0,
            len(query_ids),
            size=(count, len(query_ids)),
        )
        bootstrap[start : start + count] = paired_differences[indices].mean(
            axis=1
        )
    lower, upper = np.quantile(bootstrap, [0.025, 0.975])
    difference = float(paired_differences.mean())
    model_a_superior = bool(float(lower) > 0.0)
    return {
        "schema_version": 1,
        "result_kind": "contextworld_paired_model_comparison",
        "component_id": a_component,
        "public_test": {
            "id": a_test,
            "sha256": a_hash,
            "paired_queries": len(query_ids),
        },
        "decision_metric": "calibrated_icl_correct",
        "models": {
            "model_a": {
                "name": a_name,
                "accuracy": float(a_values.mean()),
            },
            "model_b": {
                "name": b_name,
                "accuracy": float(b_values.mean()),
            },
        },
        "paired_accuracy_difference": {
            "direction": "model_a_minus_model_b",
            "value": difference,
            "paired_bootstrap_95_percent_interval": {
                "lower": float(lower),
                "upper": float(upper),
                "resamples": int(bootstrap_resamples),
                "random_seed": int(bootstrap_seed),
            },
        },
        "superiority": {
            "criterion": "only_if_lower_bound_gt_0",
            "rule": (
                "model_a_is_superior_only_if_the_paired_bootstrap_95_percent_"
                "lower_bound_of_a_minus_b_is_strictly_greater_than_zero"
            ),
            "model_a_superior": model_a_superior,
            "superior_model": a_name if model_a_superior else None,
        },
    }


__all__ = [
    "compare_paired_query_decisions",
    "make_component_result",
    "make_public_scoreboard",
    "make_public_scoreboard_from_spec",
    "make_retention_result",
]
