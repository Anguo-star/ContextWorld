"""Additive anti-shortcut gate completion for the three under-gated components.

The Speed, Action Delay, and Door Rule Public Test releases froze their main
scores and (for action delay and door) part of their gate set, but their
release yamls define no
``correct_history`` / ``context_switch`` / latent-response thresholds.  This
module loads those thresholds from a NEW configuration file so the ten frozen
``configs/benchmark/*_icl_release_v1.yaml`` files stay byte-identical, and
provides the shared paired-unit bootstrap used by the three completions.

Every threshold here is additive: no existing gate is lowered, replaced, or
reinterpreted, and the provenance of each value is recorded in the
configuration file itself.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from contextworld.paths import repository_root


DEFAULT_GATE_COMPLETION_CONFIG = (
    repository_root()
    / "configs/benchmark/contextworld_test_gate_completion_v1.yaml"
)

GATE_KEYS_BY_COMPONENT: dict[str, tuple[str, ...]] = {
    "speed": (
        "correct_history_rate_minimum",
        "context_switch_rate_minimum",
        "worst_reference_speed_correct_history_rate_minimum",
        "target_latent_separation_required",
        "response_gain_minimum",
        "normalized_response_error_strict_maximum",
    ),
    "action_delay": (
        "correct_history_rate_minimum",
        "context_switch_rate_minimum",
        "target_latent_separation_required",
        "response_gain_minimum",
        "normalized_response_error_strict_maximum",
    ),
    "door": (
        "context_switch_rate_minimum",
        "worst_rule_correct_target_choice_rate_minimum",
        "target_latent_separation_required",
        "response_gain_minimum",
        "normalized_response_error_strict_maximum",
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_test_gate_completion_config(
    path: Path | str | None = None,
    *,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Load and validate the additive gate-completion thresholds."""

    config_path = (
        Path(path).expanduser().resolve()
        if path is not None
        else (
            (repo_root or repository_root()).resolve()
            / "configs/benchmark/contextworld_test_gate_completion_v1.yaml"
        )
    )
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError(f"Unsupported gate-completion config: {config_path}")
    if payload.get("release_id") != "contextworld_test_gate_completion_v1":
        raise ValueError(f"Unexpected gate-completion release id: {config_path}")
    if payload.get("status") not in {
        "additive_test_gate_completion",
        "accepted_additive_test_gate_completion",
    }:
        raise ValueError(
            "Gate-completion config must declare accepted additive_test_gate_completion"
        )
    missing = set(GATE_KEYS_BY_COMPONENT) - set(payload)
    if missing:
        raise ValueError(f"Gate-completion config lacks sections: {sorted(missing)}")
    for component, keys in GATE_KEYS_BY_COMPONENT.items():
        gates = payload[component].get("gates")
        if not isinstance(gates, dict):
            raise ValueError(f"Gate-completion section {component} lacks gates")
        absent = set(keys) - set(gates)
        if absent:
            raise ValueError(
                f"Gate-completion gates for {component} lack: {sorted(absent)}"
            )
        if gates["target_latent_separation_required"] is not True:
            raise ValueError(
                f"Gate completion for {component} must require latent separation"
            )
        if float(gates["response_gain_minimum"]) < 0.50:
            raise ValueError(
                f"Gate completion for {component} lowers the shared response-gain"
                " standard below 0.50"
            )
        if float(gates["normalized_response_error_strict_maximum"]) > 1.00:
            raise ValueError(
                f"Gate completion for {component} relaxes the shared"
                " normalized-response-error standard above 1.00"
            )
        bootstrap = payload[component].get("bootstrap")
        if not isinstance(bootstrap, dict):
            raise ValueError(f"Gate-completion section {component} lacks bootstrap")
        for key in ("unit", "resamples", "random_seed", "lower_bound_minimum"):
            if key not in bootstrap:
                raise ValueError(
                    f"Gate-completion bootstrap for {component} lacks {key}"
                )
        if int(bootstrap["resamples"]) <= 0:
            raise ValueError("Bootstrap resamples must be positive")
    return {
        **payload,
        "_config_path": str(config_path),
        "_config_sha256": _sha256(config_path),
    }


def unit_bootstrap_draws(
    unit_rows: np.ndarray,
    *,
    resamples: int,
    random_seed: int,
) -> np.ndarray:
    """Return per-draw unit means from a paired unit-level bootstrap.

    ``unit_rows`` has shape ``(units, metrics)``; whole units are resampled
    with replacement so per-unit decisions stay paired, exactly like the
    paired bootstrap in the six fully gated components.  The result has shape
    ``(resamples, metrics)``; callers apply their own aggregation (plain
    quantiles, or min-over-condition as the worst-condition bound) and take
    quantiles.
    """

    rows = np.asarray(unit_rows, dtype=np.float64)
    if rows.ndim == 1:
        rows = rows[:, None]
    if rows.ndim != 2 or not len(rows):
        raise ValueError("Bootstrap unit rows must have shape (units, metrics)")
    if not np.isfinite(rows).all():
        raise ValueError("Bootstrap unit rows must be finite")
    if int(resamples) <= 0:
        raise ValueError("Bootstrap resamples must be positive")
    rng = np.random.default_rng(int(random_seed))
    draws = np.empty((int(resamples), rows.shape[1]), dtype=np.float64)
    remaining = int(resamples)
    start = 0
    while remaining:
        count = min(1_000, remaining)
        indices = rng.integers(0, len(rows), size=(count, len(rows)))
        draws[start : start + count] = rows[indices].mean(axis=1)
        start += count
        remaining -= count
    return draws


def percentile_lower_bound(
    draws: np.ndarray,
    *,
    confidence_level: float = 0.95,
) -> np.ndarray:
    """Per-column percentile bootstrap lower bound."""

    tail = 0.5 * (1.0 - float(confidence_level))
    return np.quantile(np.asarray(draws, dtype=np.float64), tail, axis=0)


def threshold_source_block(config: dict[str, Any]) -> dict[str, Any]:
    """Provenance echoed into every gate-completion payload."""

    return {
        "config_path": str(config["_config_path"]),
        "config_sha256": str(config["_config_sha256"]),
        "release_id": str(config["release_id"]),
        "status": str(config["status"]),
    }


__all__ = [
    "DEFAULT_GATE_COMPLETION_CONFIG",
    "GATE_KEYS_BY_COMPONENT",
    "load_test_gate_completion_config",
    "percentile_lower_bound",
    "threshold_source_block",
    "unit_bootstrap_draws",
]
