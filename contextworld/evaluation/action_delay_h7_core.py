from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import numpy as np


DELAYS = tuple(range(11))
PHYSICAL_GROUPS = tuple(range(6))


def physical_group(delay: int) -> int:
    """Map delay values to distinguishable one-step physical outcomes."""

    return min(int(delay), 5)


def summarize_action_delay_h1_physical(
    query_metrics: list[dict[str, Any]],
    *,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    """Score six one-step physical outcomes with equal per-query weight."""

    if bootstrap_resamples <= 0:
        raise ValueError("Bootstrap resamples must be positive")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in query_metrics:
        if int(row["horizon"]) == 1:
            grouped[str(row["query_id"])].append(row)
    if not grouped:
        raise ValueError("No h1 query metrics found")
    for query_id, rows in grouped.items():
        if sorted(int(row["target_delay"]) for row in rows) != list(DELAYS):
            raise ValueError(
                f"Incomplete h1 delay family for {query_id}"
            )

    group_correct: dict[int, list[float]] = {
        group: [] for group in PHYSICAL_GROUPS
    }
    query_macros: list[float] = []
    confusion: Counter[tuple[int, int]] = Counter()
    eval_seed_counts: Counter[int] = Counter()
    by_query: list[dict[str, Any]] = []
    for query_id in sorted(grouped):
        rows = sorted(
            grouped[query_id],
            key=lambda row: int(row["target_delay"]),
        )
        eval_seeds = {int(row["eval_seed"]) for row in rows}
        if len(eval_seeds) != 1:
            raise ValueError(
                f"Eval seed changed within query {query_id}"
            )
        eval_seed = next(iter(eval_seeds))
        eval_seed_counts[eval_seed] += 1
        query_groups: dict[int, list[float]] = {
            group: [] for group in PHYSICAL_GROUPS
        }
        for row in rows:
            true_group = physical_group(int(row["target_delay"]))
            selected_group = physical_group(int(row["selected_target"]))
            correct = float(selected_group == true_group)
            query_groups[true_group].append(correct)
            group_correct[true_group].append(correct)
            confusion[(true_group, selected_group)] += 1
        if not all(query_groups[group] for group in PHYSICAL_GROUPS):
            raise ValueError(
                f"Incomplete physical groups for {query_id}"
            )
        group_scores = {
            group: float(np.mean(query_groups[group]))
            for group in PHYSICAL_GROUPS
        }
        macro = float(np.mean(list(group_scores.values())))
        query_macros.append(macro)
        by_query.append(
            {
                "query_id": query_id,
                "eval_seed": eval_seed,
                "physical_group_accuracy": {
                    str(group): score
                    for group, score in group_scores.items()
                },
                "physical_group_macro_accuracy": macro,
            }
        )

    values = np.asarray(query_macros, dtype=np.float64)
    rng = np.random.default_rng(int(bootstrap_seed))
    bootstrap = np.empty(int(bootstrap_resamples), dtype=np.float64)
    for start in range(0, bootstrap_resamples, 1000):
        count = min(1000, bootstrap_resamples - start)
        indices = rng.integers(
            0,
            len(values),
            size=(count, len(values)),
        )
        bootstrap[start : start + count] = values[indices].mean(axis=1)
    lower, upper = np.quantile(bootstrap, [0.025, 0.975])
    by_group = {
        str(group): {
            "history_conditions": len(group_correct[group]),
            "accuracy": float(np.mean(group_correct[group])),
            "delays": (
                [group] if group < 5 else [5, 6, 7, 8, 9, 10]
            ),
        }
        for group in PHYSICAL_GROUPS
    }
    return {
        "queries": len(grouped),
        "history_conditions": sum(
            len(rows) for rows in group_correct.values()
        ),
        "physical_groups": len(PHYSICAL_GROUPS),
        "aggregation": (
            "equal physical-group mean within query, then query mean"
        ),
        "physical_group_macro_accuracy": float(values.mean()),
        "minimum_physical_group_accuracy": float(
            min(row["accuracy"] for row in by_group.values())
        ),
        "paired_query_bootstrap_95_percent_interval": {
            "lower": float(lower),
            "upper": float(upper),
            "resamples": int(bootstrap_resamples),
            "random_seed": int(bootstrap_seed),
        },
        "by_physical_group": by_group,
        "confusion_counts": {
            str(true_group): {
                str(selected_group): int(
                    confusion[(true_group, selected_group)]
                )
                for selected_group in PHYSICAL_GROUPS
            }
            for true_group in PHYSICAL_GROUPS
        },
        "eval_seed_query_counts": {
            str(seed): int(count)
            for seed, count in sorted(eval_seed_counts.items())
        },
        "query_metrics": by_query,
    }


__all__ = [
    "DELAYS",
    "PHYSICAL_GROUPS",
    "physical_group",
    "summarize_action_delay_h1_physical",
]
