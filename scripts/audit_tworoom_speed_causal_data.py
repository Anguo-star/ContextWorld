#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from contextworld.paths import repository_root, resolve_contextworld_path


CATALOGS = (
    "seen_for_multi",
    "unseen_interpolation",
    "extrapolation_low",
    "extrapolation_high",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _equal(values: list[np.ndarray]) -> bool:
    return all(np.array_equal(values[0], value) for value in values[1:])


def _minimum_pair_gap(values: list[np.ndarray]) -> float:
    return min(
        float(np.linalg.norm(values[left].astype(np.float64) - values[right]))
        for left in range(len(values))
        for right in range(left + 1, len(values))
    )


def audit_speed_causal_data(
    *,
    catalog_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    failures: list[dict[str, str]] = []
    bundles = 0
    condition_trajectories = 0
    maximum_transition_residual = 0.0
    minimum_history_state_gap = float("inf")
    grouped: dict[str, list[dict[str, Any]]] = {}
    catalog_receipts: dict[str, dict[str, Any]] = {}

    for track in CATALOGS:
        catalog_path = catalog_root / f"{track}.json"
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        catalog_receipts[track] = {
            "path": str(catalog_path),
            "sha256": _sha256(catalog_path),
            "bundles": len(catalog["bundles"]),
        }
        for row in catalog["bundles"]:
            bundles += 1
            payload_path = resolve_contextworld_path(
                row["payload"], repo_root=repo_root
            )
            if _sha256(payload_path) != row["payload_sha256"]:
                failures.append(
                    {"query_id": row["query_id"], "reason": "payload_hash"}
                )
                continue
            with np.load(payload_path, allow_pickle=False) as arrays:
                conditions = list(row["conditions"])
                condition_trajectories += len(conditions)
                x0_pixels: list[np.ndarray] = []
                x0_states: list[np.ndarray] = []
                context_actions: list[np.ndarray] = []
                x2_pixels: list[np.ndarray] = []
                x2_states: list[np.ndarray] = []
                middle_states: list[np.ndarray] = []
                for condition in conditions:
                    prefix = f"context_b2_{condition}"
                    pixels = arrays[f"{prefix}_pixels"]
                    actions = arrays[f"{prefix}_actions"]
                    next_pixels = arrays[f"{prefix}_next_pixels"]
                    states = arrays[f"{prefix}_states"]
                    next_states = arrays[f"{prefix}_next_states"]
                    x0_pixels.append(pixels[0])
                    x0_states.append(states[0])
                    context_actions.append(actions)
                    x2_pixels.append(next_pixels[-1])
                    x2_states.append(next_states[-1])
                    middle_states.append(next_states[0])
                    speed = float(
                        row["conditions"][condition]["factors"][
                            "agent.speed"
                        ]
                    )
                    for index in range(actions.shape[0]):
                        expected = (
                            states[index].astype(np.float64)
                            + speed
                            * actions[index].astype(np.float64).sum(axis=0)
                        )
                        residual = float(
                            np.linalg.norm(
                                next_states[index].astype(np.float64)
                                - expected
                            )
                        )
                        maximum_transition_residual = max(
                            maximum_transition_residual, residual
                        )
                    if not (
                        np.array_equal(next_pixels[0], pixels[1])
                        and np.array_equal(next_states[0], states[1])
                    ):
                        failures.append(
                            {
                                "query_id": row["query_id"],
                                "reason": f"broken_context_chain:{condition}",
                            }
                        )
                checks = {
                    "x0_pixels_identical": _equal(x0_pixels),
                    "x0_states_identical": _equal(x0_states),
                    "history_actions_identical": _equal(context_actions),
                    "x2_pixels_identical": _equal(x2_pixels),
                    "x2_states_identical": _equal(x2_states),
                    "x2_is_query_pixels": all(
                        np.array_equal(value, arrays["query_pixels"])
                        for value in x2_pixels
                    ),
                    "x2_is_future_start_state": all(
                        np.array_equal(value, arrays["future_states"][0])
                        for value in x2_states
                    ),
                }
                for name, passed in checks.items():
                    if not passed:
                        failures.append(
                            {
                                "query_id": row["query_id"],
                                "reason": name,
                            }
                        )
                history_gap = _minimum_pair_gap(middle_states)
                minimum_history_state_gap = min(
                    minimum_history_state_gap, history_gap
                )
                query_speed = float(row["query_factors"]["agent.speed"])
                future_actions = arrays["future_actions"]
                future_states = arrays["future_states"]
                future_next_states = arrays["future_next_states"]
                for index in range(future_actions.shape[0]):
                    expected = (
                        future_states[index].astype(np.float64)
                        + query_speed
                        * future_actions[index].astype(np.float64).sum(axis=0)
                    )
                    residual = float(
                        np.linalg.norm(
                            future_next_states[index].astype(np.float64)
                            - expected
                        )
                    )
                    maximum_transition_residual = max(
                        maximum_transition_residual, residual
                    )
                grouped.setdefault(row["static_query_id"], []).append(
                    {
                        "query_pixels_sha256": row["query_pixels_sha256"],
                        "future_actions_sha256": row["future_actions_sha256"],
                        "x0_pixels": x0_pixels[0].copy(),
                        "x0_state": x0_states[0].copy(),
                        "future_state": future_next_states[0].copy(),
                    }
                )

    minimum_future_state_gap = float("inf")
    for static_query_id, rows in grouped.items():
        group_checks = {
            "query_pixels_identical_across_speeds": len(
                {row["query_pixels_sha256"] for row in rows}
            )
            == 1,
            "future_actions_identical_across_speeds": len(
                {row["future_actions_sha256"] for row in rows}
            )
            == 1,
            "x0_pixels_identical_across_speeds": _equal(
                [row["x0_pixels"] for row in rows]
            ),
            "x0_states_identical_across_speeds": _equal(
                [row["x0_state"] for row in rows]
            ),
        }
        for name, passed in group_checks.items():
            if not passed:
                failures.append(
                    {"query_id": static_query_id, "reason": name}
                )
        minimum_future_state_gap = min(
            minimum_future_state_gap,
            _minimum_pair_gap([row["future_state"] for row in rows]),
        )

    checks = {
        "all_payloads_reopened_and_hashed": not any(
            row["reason"] == "payload_hash" for row in failures
        ),
        "every_history_is_a_continuous_two_transition_chain": not any(
            row["reason"].startswith("broken_context_chain")
            for row in failures
        ),
        "common_x0_is_exact_across_history_conditions_and_speeds": not any(
            row["reason"].startswith("x0_") for row in failures
        ),
        "history_actions_are_exactly_paired": not any(
            row["reason"] == "history_actions_identical"
            for row in failures
        ),
        "x2_query_is_exactly_paired": not any(
            row["reason"].startswith("x2_") for row in failures
        ),
        "future_actions_are_exactly_paired": not any(
            row["reason"] == "future_actions_identical_across_speeds"
            for row in failures
        ),
        "history_reveals_speed": minimum_history_state_gap > 0.0,
        "real_future_depends_on_speed": minimum_future_state_gap > 0.0,
        "natural_transition_residual_within_1e_4_px": (
            maximum_transition_residual <= 1e-4
        ),
        "all_registered_tracks_are_present": set(catalog_receipts)
        == set(CATALOGS),
    }
    return {
        "schema_version": 1,
        "benchmark": "tworoom_speed_continuous_causal_data_audit",
        "status": "passed" if all(checks.values()) else "failed",
        "catalogs": catalog_receipts,
        "counts": {
            "bundles": bundles,
            "condition_trajectories": condition_trajectories,
            "static_query_groups": len(grouped),
        },
        "measurements": {
            "maximum_natural_transition_residual_px": (
                maximum_transition_residual
            ),
            "minimum_history_state_gap_px": minimum_history_state_gap,
            "minimum_true_future_state_gap_px": minimum_future_state_gap,
            "state_installations_after_x0": 0,
            "query_simulator_recreated": False,
        },
        "checks": checks,
        "failure_count": len(failures),
        "failures": failures[:100],
        "passed": all(checks.values()) and not failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--catalog-root",
        default=(
            "artifacts/evaluation/history3/"
            "speed_multistep_extrap_v5/catalogs"
        ),
    )
    parser.add_argument(
        "--output",
        default=(
            "artifacts/evaluation/history3/"
            "speed_continuous_causal_audit.json"
        ),
    )
    args = parser.parse_args()
    root = repository_root().resolve()
    catalog_root = resolve_contextworld_path(
        args.catalog_root, repo_root=root
    )
    report = audit_speed_causal_data(
        catalog_root=catalog_root,
        repo_root=root,
    )
    output = Path(args.output).expanduser()
    if not output.is_absolute():
        output = root / output
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(json.dumps({"output": str(output), **report["counts"], "passed": report["passed"]}))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
