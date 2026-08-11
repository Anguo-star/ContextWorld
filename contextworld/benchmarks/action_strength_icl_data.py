from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from contextworld.benchmarks.causal_data_contract import (
    audit_causal_data_contract,
)
from contextworld.paths import (
    artifact_root,
    repository_root,
    resolve_contextworld_path,
)


ACTION_STRENGTH_RELEASE_ID = (
    "contextworld_pusht_action_strength_icl_history3_v1"
)
DEFAULT_ACTION_STRENGTH_RELEASE_CONFIG = (
    repository_root()
    / "configs/benchmark/pusht_action_strength_icl_release_v1.yaml"
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for child in sorted(value for value in path.rglob("*") if value.is_file()):
        relative = child.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_sha256(child).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def load_action_strength_icl_release(
    path: Path | str = DEFAULT_ACTION_STRENGTH_RELEASE_CONFIG,
) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError(f"Unsupported Action Strength release: {config_path}")
    if payload.get("release_id") != ACTION_STRENGTH_RELEASE_ID:
        raise ValueError(f"Unexpected Action Strength release id: {config_path}")
    if payload.get("release_status") not in {
        "validation_release_candidate",
            "validation_release",
            "public_test_release_candidate",
            "public_test_release",
            "data_ready_reference_requires_latent_response_rescore",
        }:
        raise ValueError("Unsupported Action Strength release status")
    scope = payload.get("scope", {})
    if scope.get("history_tokens") != 3:
        raise ValueError("Action Strength v1 requires History=3")
    if str(payload.get("release_status")).startswith("public_test_") and (
        scope.get("public_test_included") is not True
    ):
        raise ValueError("Action Strength v1 must include Public Test")
    if scope.get("sealed_test_included") is not False:
        raise ValueError("The public release must not include sealed Test")
    if scope.get("strength_values") != [60, 140]:
        raise ValueError("Action Strength v1 requires strengths [60, 140]")
    return {**payload, "_config_path": str(config_path)}


def _resolve_upstream(
    specification: dict[str, Any],
    *,
    repo_root: Path,
) -> Path:
    environment = str(specification.get("environment_variable", ""))
    configured = os.environ.get(environment) if environment else None
    if configured:
        return Path(configured).expanduser().resolve()
    bundled = specification.get("bundled_artifact_path")
    if bundled:
        candidate = artifact_root(repo_root) / str(bundled)
        if candidate.exists():
            return candidate.resolve()
    symbol = specification.get("source_symbol", "unspecified_upstream")
    raise ValueError(
        f"Upstream input {symbol!r} is not installed; set "
        f"{environment!r} or provide the bundled artifact"
    )


def resolve_action_strength_original_h5(
    release: dict[str, Any],
    *,
    repo_root: Path,
) -> Path:
    return _resolve_upstream(
        release["training"]["upstream"]["original_h5"],
        repo_root=repo_root,
    )


def resolve_action_strength_original_lance(
    release: dict[str, Any],
    *,
    repo_root: Path,
) -> Path:
    return _resolve_upstream(
        release["training"]["upstream"]["original_lance"],
        repo_root=repo_root,
    )


def resolve_action_strength_initial_checkpoint(
    release: dict[str, Any],
    *,
    repo_root: Path,
) -> Path:
    return _resolve_upstream(
        release["training"]["initialization"],
        repo_root=repo_root,
    )


@dataclass(frozen=True)
class ActionStrengthEvalArrays:
    pair_ids: tuple[str, ...]
    low_pixels: np.ndarray
    high_pixels: np.ndarray
    raw_action_blocks: np.ndarray
    low_states: np.ndarray
    high_states: np.ndarray

    @property
    def pair_count(self) -> int:
        return len(self.pair_ids)


def _decode_rgb(value: bytes) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Reading Action Strength pixels requires Pillow"
        ) from exc
    with Image.open(BytesIO(value)) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _read_lance_pairs(
    path: Path,
    *,
    expected_pairs: int,
) -> ActionStrengthEvalArrays:
    try:
        import lance
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Reading Action Strength data requires the Python lance package"
        ) from exc

    table = lance.dataset(path).to_table(
        columns=[
            "episode_idx",
            "step_idx",
            "pixels",
            "action",
            "state",
            "pair_id",
            "hidden_mode",
        ]
    )
    episode_indices = np.asarray(
        table["episode_idx"].to_numpy(), dtype=np.int64
    )
    step_indices = np.asarray(table["step_idx"].to_numpy(), dtype=np.int64)
    actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    states = np.asarray(table["state"].to_pylist(), dtype=np.float32)
    pixel_bytes = table["pixels"].to_pylist()
    pair_ids = table["pair_id"].to_pylist()
    modes = table["hidden_mode"].to_pylist()

    episodes: dict[str, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    for episode in np.unique(episode_indices):
        rows = np.flatnonzero(episode_indices == episode)
        order = np.argsort(step_indices[rows])
        rows = rows[order]
        if not np.array_equal(step_indices[rows], np.arange(20)):
            raise RuntimeError(f"Episode {episode} is not a complete 20-row clip")
        pair_values = {str(pair_ids[index]) for index in rows}
        mode_values = {str(modes[index]) for index in rows}
        if len(pair_values) != 1 or len(mode_values) != 1:
            raise RuntimeError(f"Episode {episode} changes pair or strength")
        frame_rows = rows[[0, 5, 10, 15]]
        pixels = np.stack([_decode_rgb(pixel_bytes[index]) for index in frame_rows])
        action_blocks = actions[rows].reshape(4, 5, 2)
        frame_states = states[frame_rows]
        pair_id = pair_values.pop()
        mode = mode_values.pop()
        if mode not in {"low_gain", "high_gain"}:
            raise RuntimeError(f"Unexpected action-strength mode {mode!r}")
        if mode in episodes.setdefault(pair_id, {}):
            raise RuntimeError(f"Duplicate {mode} episode for {pair_id}")
        episodes[pair_id][mode] = (pixels, action_blocks, frame_states)

    if len(episodes) != expected_pairs:
        raise RuntimeError(
            f"Expected {expected_pairs} Action Strength pairs, got {len(episodes)}"
        )
    ordered_ids = tuple(sorted(episodes))
    low_pixels: list[np.ndarray] = []
    high_pixels: list[np.ndarray] = []
    action_blocks: list[np.ndarray] = []
    low_states: list[np.ndarray] = []
    high_states: list[np.ndarray] = []
    for pair_id in ordered_ids:
        pair = episodes[pair_id]
        if set(pair) != {"low_gain", "high_gain"}:
            raise RuntimeError(f"Incomplete low/high pair: {pair_id}")
        low = pair["low_gain"]
        high = pair["high_gain"]
        if not np.array_equal(low[0][0], high[0][0]):
            raise RuntimeError(f"Initial image differs within pair {pair_id}")
        if not np.array_equal(low[0][2], high[0][2]):
            raise RuntimeError(f"Current query image differs within pair {pair_id}")
        if not np.array_equal(low[1], high[1]):
            raise RuntimeError(f"Actions differ within pair {pair_id}")
        if np.array_equal(low[0][1], high[0][1]):
            raise RuntimeError(f"History does not reveal strength for {pair_id}")
        if np.array_equal(low[0][3], high[0][3]):
            raise RuntimeError(f"True futures do not differ for {pair_id}")
        low_pixels.append(low[0])
        high_pixels.append(high[0])
        action_blocks.append(low[1])
        low_states.append(low[2])
        high_states.append(high[2])
    return ActionStrengthEvalArrays(
        pair_ids=ordered_ids,
        low_pixels=np.stack(low_pixels),
        high_pixels=np.stack(high_pixels),
        raw_action_blocks=np.stack(action_blocks),
        low_states=np.stack(low_states),
        high_states=np.stack(high_states),
    )


class ActionStrengthICLEvalDataset:
    """Frozen 256-pair PushT Action Strength confirmation set."""

    def __init__(
        self,
        *,
        release: dict[str, Any] | None = None,
        release_config: Path | str = DEFAULT_ACTION_STRENGTH_RELEASE_CONFIG,
        repo_root: Path | None = None,
    ) -> None:
        self.repo_root = (repo_root or repository_root()).resolve()
        self.release = release or load_action_strength_icl_release(
            release_config
        )
        self.root = resolve_contextworld_path(
            self.release["evaluation"]["artifact_tree"]["root"],
            repo_root=self.repo_root,
        )
        self._arrays: ActionStrengthEvalArrays | None = None

    @property
    def arrays(self) -> ActionStrengthEvalArrays:
        if self._arrays is None:
            self._arrays = _read_lance_pairs(
                self.root / self.release["evaluation"]["lance_table"],
                expected_pairs=int(self.release["evaluation"]["pair_count"]),
            )
        return self._arrays

    @property
    def is_full_protocol(self) -> bool:
        return self.arrays.pair_count == int(
            self.release["evaluation"]["pair_count"]
        )

    def describe(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "pair_count": self.arrays.pair_count,
            "condition_count": 2 * self.arrays.pair_count,
            "history_tokens": 3,
            "strength_values": [60, 140],
            "online_environment_calls": 0,
        }


def _verified_file(
    specification: dict[str, Any],
    *,
    repo_root: Path,
) -> dict[str, Any]:
    path = resolve_contextworld_path(specification["path"], repo_root=repo_root)
    exists = path.is_file()
    observed = file_sha256(path) if exists else None
    expected = str(specification["sha256"])
    return {
        "path": str(path),
        "exists": exists,
        "expected_sha256": expected,
        "observed_sha256": observed,
        "passed": bool(exists and observed == expected),
    }


def _lance_rows(path: Path) -> int:
    try:
        import lance
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("Action Strength audit requires lance") from exc
    return int(lance.dataset(path).count_rows())


def _reference_method_release_audit(
    release: dict[str, Any],
    *,
    repo_root: Path,
) -> dict[str, Any]:
    """Audit the complete, self-contained three-seed reference package."""

    reference = release["reference_method"]
    reference_root = resolve_contextworld_path(
        reference["artifact_tree"]["root"], repo_root=repo_root
    )
    summary_specification = release["reference_results"][
        "reference_method_summary"
    ]
    summary_path = resolve_contextworld_path(
        summary_specification["path"], repo_root=repo_root
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    response_specification = release["reference_results"][
        "latent_response_summary"
    ]
    response_path = resolve_contextworld_path(
        response_specification["path"], repo_root=repo_root
    )
    response_summary = json.loads(
        response_path.read_text(encoding="utf-8")
    )
    training_receipt_sha256 = release["training"]["artifacts"][
        "portability_receipt"
    ]["sha256"]
    public_test_receipt_sha256 = release["evaluation"]["artifacts"][
        "portability_receipt"
    ]["sha256"]
    reference_training_scales = json.loads(
        resolve_contextworld_path(
            summary["method"]["reference_training_scales"]["path"],
            repo_root=repo_root,
        ).read_text(encoding="utf-8")
    )

    artifact_specifications: dict[str, dict[str, str]] = {
        "reference_training_scales": summary["method"][
            "reference_training_scales"
        ],
        "latent_response_summary": response_specification,
    }
    for name, specification in summary["data_binding"].items():
        if isinstance(specification, dict) and {
            "path",
            "sha256",
        }.issubset(specification):
            artifact_specifications[f"data_binding.{name}"] = specification
    for seed, seed_summary in summary["per_seed"].items():
        for name, specification in seed_summary["artifacts"].items():
            artifact_specifications[f"seed_{seed}.{name}"] = specification

    verified_artifacts: dict[str, Any] = {}
    expected_paths = {summary_path.resolve(), response_path.resolve()}
    all_artifacts_inside_package = True
    for name, specification in artifact_specifications.items():
        path = resolve_contextworld_path(
            specification["path"], repo_root=repo_root
        ).resolve()
        try:
            path.relative_to(reference_root.resolve())
        except ValueError:
            all_artifacts_inside_package = False
        expected_paths.add(path)
        verified_artifacts[name] = _verified_file(
            specification, repo_root=repo_root
        )

    actual_paths = {
        path.resolve() for path in reference_root.rglob("*") if path.is_file()
    }
    forbidden_tokens = (
        "/opt/",
        "data/world_model/context_world/evaluation/history3/"
        "pusht_hidden_actuation",
        "research",
    )
    contaminated_files: dict[str, list[str]] = {}
    for path in sorted(actual_paths):
        if path.suffix == ".pt":
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            contaminated_files[path.relative_to(reference_root).as_posix()] = [
                "non_utf8_public_metadata"
            ]
            continue
        found = [token for token in forbidden_tokens if token in text]
        if found:
            contaminated_files[path.relative_to(reference_root).as_posix()] = found

    config_text = Path(release["_config_path"]).read_text(encoding="utf-8")
    config_forbidden = (
        "/opt/",
        "local_source",
        "data/world_model/context_world/evaluation/history3/"
        "pusht_hidden_actuation",
    )
    config_contamination = [
        token for token in config_forbidden if token in config_text
    ]

    strict_data = json.loads(
        resolve_contextworld_path(
            summary["data_binding"][
                "strict_causal_data_compatibility_audit"
            ]["path"],
            repo_root=repo_root,
        ).read_text(encoding="utf-8")
    )
    strict_results = json.loads(
        resolve_contextworld_path(
            summary["data_binding"]["strict_result_compatibility_audit"][
                "path"
            ],
            repo_root=repo_root,
        ).read_text(encoding="utf-8")
    )

    expected_seeds = [str(value) for value in reference["training_seeds"]]
    prediction_gates = release["scoring"]["hidden_future_prediction"][
        "gates"
    ]
    anti_spoof_gate_names = {
        "target_latent_separation_required",
        "response_gain_minimum",
        "normalized_response_error_strict_maximum",
    }
    legacy_prediction_gates = {
        name: value
        for name, value in prediction_gates.items()
        if name not in anti_spoof_gate_names
    }
    expected_gates = {
        **legacy_prediction_gates,
        "correct_action_region_rate_minimum": release["scoring"][
            "action_planning"
        ]["correct_action_region_rate_minimum"],
        "standard_pusht_cem_successes_minimum": release["scoring"][
            "original_task_retention"
        ]["noninferiority_minimum_successes"],
        "standard_pusht_cem_evaluations": release["scoring"][
            "original_task_retention"
        ]["independent_cem_episodes"],
    }
    prediction_metric_names = (
        "correct_future_rate",
        "correct_history_rate",
        "rule_switch_rate",
        "worst_strength_correct_future_rate",
    )
    response_rows = {
        str(row["training_seed"]): row
        for row in response_summary.get("models", [])
    }

    def close(left: Any, right: Any) -> bool:
        return bool(
            np.isclose(float(left), float(right), rtol=0.0, atol=1.0e-12)
        )

    per_seed: dict[str, Any] = {}
    metric_values = {name: [] for name in prediction_metric_names}
    action_rates: list[float] = []
    cem_successes = 0
    cem_evaluations = 0
    for seed in expected_seeds:
        seed_summary = summary["per_seed"][seed]
        configured = reference["per_seed"][int(seed)]
        artifacts = seed_summary["artifacts"]
        checkpoint_hash = artifacts["checkpoint"]["sha256"]
        training_report = json.loads(
            resolve_contextworld_path(
                artifacts["training_report"]["path"], repo_root=repo_root
            ).read_text(encoding="utf-8")
        )
        model_config = json.loads(
            resolve_contextworld_path(
                artifacts["model_config"]["path"], repo_root=repo_root
            ).read_text(encoding="utf-8")
        )
        prediction = json.loads(
            resolve_contextworld_path(
                artifacts["prediction_result"]["path"], repo_root=repo_root
            ).read_text(encoding="utf-8")
        )
        planning = json.loads(
            resolve_contextworld_path(
                artifacts["action_planning_result"]["path"],
                repo_root=repo_root,
            ).read_text(encoding="utf-8")
        )
        retention = json.loads(
            resolve_contextworld_path(
                artifacts["standard_pusht_cem_result"]["path"],
                repo_root=repo_root,
            ).read_text(encoding="utf-8")
        )
        compatibility = strict_results["per_seed"][seed]
        response = response_rows.get(seed, {})

        metric_match = all(
            close(seed_summary["prediction_metrics"][name], prediction["metrics"][name])
            and close(seed_summary["prediction_metrics"][name], configured[name])
            for name in prediction_metric_names
        )
        checks = {
            "exact_six_artifacts": set(artifacts)
            == {
                "checkpoint",
                "model_config",
                "training_report",
                "prediction_result",
                "action_planning_result",
                "standard_pusht_cem_result",
            },
            "all_artifact_hashes_verified": all(
                verified_artifacts[f"seed_{seed}.{name}"]["passed"]
                for name in artifacts
            ),
            "checkpoint_bound_everywhere": all(
                payload.get("checkpoint_sha256") == checkpoint_hash
                for payload in (prediction, planning, retention)
            )
            and training_report["checkpoint"]["sha256"] == checkpoint_hash
            and configured["checkpoint"]["sha256"] == checkpoint_hash,
            "training_receipt_matches_recipe": (
                training_report.get("status")
                == "completed_reference_training_receipt"
                and training_report.get("training_seed") == int(seed)
                and training_report.get("training_recipe")
                == reference["training_recipe"]
                and training_report.get("optimizer_steps")
                == reference["optimizer_steps"]
                and training_report.get("data", {}).get(
                    "formal_training_manifest_sha256"
                )
                == release["training"]["manifest_sha256"]
                and training_report.get("data", {}).get(
                    "formal_training_portability_receipt_sha256"
                )
                == training_receipt_sha256
                and training_report.get("data", {}).get(
                    "model_visible_bytes_compatible_with_training_source"
                )
                is True
            ),
            "history3_model_config": model_config.get("predictor", {}).get(
                "num_frames"
            )
            == 3,
            "prediction_receipt_matches_summary": (
                prediction.get("training_seed") == int(seed)
                and prediction.get("training_manifest_sha256")
                == release["training"]["manifest_sha256"]
                and prediction.get("training_portability_receipt_sha256")
                == training_receipt_sha256
                and prediction.get("public_test_manifest_sha256")
                == release["evaluation"]["manifest_sha256"]
                and prediction.get(
                    "public_test_portability_receipt_sha256"
                )
                == public_test_receipt_sha256
                and prediction.get("pair_count")
                == release["evaluation"]["pair_count"]
                and prediction.get("decision_count")
                == release["evaluation"]["condition_count"]
                and metric_match
                and prediction.get("gate", {}).get("passed") is True
            ),
            "latent_response_receipt_passes_current_gate": bool(
                response.get("checkpoint_sha256") == checkpoint_hash
                and close(
                    response.get("correct_future_rate"),
                    seed_summary["prediction_metrics"][
                        "correct_future_rate"
                    ],
                )
                and float(response.get("minimum_target_response_mse", 0.0))
                > 0.0
                and float(response.get("response_gain", -np.inf))
                >= float(prediction_gates["response_gain_minimum"])
                and float(
                    response.get("normalized_response_error", np.inf)
                )
                < float(
                    prediction_gates[
                        "normalized_response_error_strict_maximum"
                    ]
                )
                and response.get("response_gate_passed") is True
            ),
            "planning_receipt_matches_summary": (
                planning.get("training_seed") == int(seed)
                and planning.get("public_test_manifest_sha256")
                == release["evaluation"]["manifest_sha256"]
                and planning.get(
                    "public_test_portability_receipt_sha256"
                )
                == public_test_receipt_sha256
                and planning.get("oracle_sha256")
                == release["evaluation"]["planning_oracle"]["sha256"]
                and planning.get("protocol", {}).get("condition_count")
                == release["evaluation"]["condition_count"]
                and planning.get("protocol", {}).get("cem_candidates")
                == release["scoring"]["action_planning"]["candidates"]
                and planning.get("protocol", {}).get("cem_iterations")
                == release["scoring"]["action_planning"]["iterations"]
                and planning.get("protocol", {}).get("cem_topk")
                == release["scoring"]["action_planning"]["topk"]
                and close(
                    planning.get("correct_action_region_rate"),
                    seed_summary["correct_action_region_rate"],
                )
                and close(
                    planning.get("correct_action_region_rate"),
                    configured["correct_action_region_rate"],
                )
                and planning.get("receipts", {}).get(
                    "strict_and_source_selections_identical"
                )
                is True
                and planning.get("passed") is True
            ),
            "standard_cem_receipt_matches_summary": (
                retention.get("training_seed") == int(seed)
                and retention.get("dataset") == "upstream_original_h5"
                and retention.get("query_catalog", {}).get("sha256")
                == release["scoring"]["original_task_retention"][
                    "query_catalog_sha256"
                ]
                and retention.get("protocol", {}).get("eval_seeds")
                == release["scoring"]["original_task_retention"]["eval_seeds"]
                and retention.get("protocol", {}).get("queries_per_seed")
                == release["scoring"]["original_task_retention"][
                    "queries_per_seed"
                ]
                and retention.get("successes")
                == seed_summary["standard_pusht_cem"]["successes"]
                == configured["standard_cem_successes"]
                and retention.get("evaluations")
                == seed_summary["standard_pusht_cem"]["evaluations"]
                == configured["standard_cem_evaluations"]
                and retention.get("passed") is True
            ),
            "strict_result_compatibility_passed": (
                compatibility["prediction"]["strict_source_result_sha256"]
                == prediction["source_result_sha256"]
                and compatibility["prediction"]["passed"] is True
                and compatibility["action_planning"][
                    "strict_source_result_sha256"
                ]
                == planning["source_result_sha256"]
                and compatibility["action_planning"]["passed"] is True
                and compatibility["standard_pusht_cem"][
                    "source_result_sha256"
                ]
                == retention["source_result_sha256"]
                and compatibility["standard_pusht_cem"]["successes"]
                == retention["successes"]
                and compatibility["standard_pusht_cem"]["evaluations"]
                == retention["evaluations"]
                and compatibility["standard_pusht_cem"]["passed"] is True
            ),
            "seed_gate_passed": (
                seed_summary.get("passed") is True
                and configured.get("all_gates_passed") is True
            ),
        }
        per_seed[seed] = {"checks": checks, "passed": all(checks.values())}
        for name in prediction_metric_names:
            metric_values[name].append(
                float(seed_summary["prediction_metrics"][name])
            )
        action_rates.append(float(seed_summary["correct_action_region_rate"]))
        cem_successes += int(seed_summary["standard_pusht_cem"]["successes"])
        cem_evaluations += int(seed_summary["standard_pusht_cem"]["evaluations"])

    aggregate = summary["aggregate"]
    configured_aggregate = reference["aggregate"]
    aggregate_checks = {
        **{
            f"{name}_mean": close(
                aggregate[f"{name}_mean"], np.mean(metric_values[name])
            )
            and close(
                aggregate[f"{name}_mean"],
                configured_aggregate[f"{name}_mean"],
            )
            for name in prediction_metric_names
        },
        "correct_action_region_rate_mean": close(
            aggregate["correct_action_region_rate_mean"],
            np.mean(action_rates),
        )
        and close(
            aggregate["correct_action_region_rate_mean"],
            configured_aggregate["correct_action_region_rate_mean"],
        ),
        "standard_cem_totals": (
            aggregate["standard_pusht_cem_successes"] == cem_successes
            == configured_aggregate["standard_cem_successes"]
            and aggregate["standard_pusht_cem_evaluations"] == cem_evaluations
            == configured_aggregate["standard_cem_evaluations"]
        ),
        "all_three_seeds_passed": (
            aggregate.get("all_three_seeds_passed") is True
            and configured_aggregate.get("all_three_seeds_passed") is True
            and all(row["passed"] for row in per_seed.values())
        ),
    }

    byte_compatibility = strict_data.get(
        "model_visible_byte_compatibility", {}
    )
    root_checks = {
        "summary_contract": (
            summary.get("release_id") == release["release_id"]
            and summary.get("status") == "passed"
            and summary.get("method", {}).get("training_recipe")
            == reference["training_recipe"]
            and summary.get("method", {}).get("optimizer_steps")
            == reference["optimizer_steps"]
            and summary.get("method", {}).get("training_seeds")
            == reference["training_seeds"]
        ),
        "summary_gates_match_config": summary.get("gates") == expected_gates,
        "latent_response_summary_matches_config": bool(
            response_summary.get("public_test_manifest_sha256")
            == release["evaluation"]["manifest_sha256"]
            and response_summary.get("gate")
            == {
                "response_gain_minimum": prediction_gates[
                    "response_gain_minimum"
                ],
                "normalized_response_error_exclusive_maximum": (
                    prediction_gates[
                        "normalized_response_error_strict_maximum"
                    ]
                ),
                "target_latent_separation_required": prediction_gates[
                    "target_latent_separation_required"
                ],
            }
            and set(response_rows) == set(expected_seeds)
            and response_summary.get("all_three_response_gates_passed")
            is True
        ),
        "summary_binds_portable_data_releases": (
            summary.get("data_binding", {}).get("training_manifest_sha256")
            == release["training"]["manifest_sha256"]
            and summary.get("data_binding", {}).get(
                "training_portability_receipt_sha256"
            )
            == training_receipt_sha256
            and summary.get("data_binding", {}).get(
                "public_test_manifest_sha256"
            )
            == release["evaluation"]["manifest_sha256"]
            and summary.get("data_binding", {}).get(
                "public_test_portability_receipt_sha256"
            )
            == public_test_receipt_sha256
        ),
        "reference_training_scales_bind_portable_training": (
            reference_training_scales.get(
                "formal_training_manifest_sha256"
            )
            == release["training"]["manifest_sha256"]
            and reference_training_scales.get(
                "formal_training_portability_receipt_sha256"
            )
            == training_receipt_sha256
        ),
        "exact_seed_set": set(summary.get("per_seed", {}))
        == set(expected_seeds),
        "all_artifacts_inside_package": all_artifacts_inside_package,
        "exact_package_file_set": actual_paths == expected_paths,
        "all_package_artifact_hashes_verified": all(
            row["passed"] for row in verified_artifacts.values()
        ),
        "package_metadata_is_self_contained": not contaminated_files,
        "component_config_has_no_machine_paths": not config_contamination,
        "strict_causal_receipt_passed": (
            strict_data.get("status") == "passed"
            and strict_data.get("passed") is True
            and strict_data.get("formal_manifests", {}).get(
                "training_and_development"
            )
            == release["training"]["manifest_sha256"]
            and strict_data.get("formal_manifests", {}).get("public_test")
            == release["evaluation"]["manifest_sha256"]
            and strict_data.get("formal_portability_receipts", {}).get(
                "training_and_development"
            )
            == training_receipt_sha256
            and strict_data.get("formal_portability_receipts", {}).get(
                "public_test"
            )
            == public_test_receipt_sha256
            and strict_data.get("strict_causal_contract", {}).get(
                "audited_pairs"
            )
            == 2560
            and strict_data.get("strict_causal_contract", {}).get("passed")
            is True
            and all(
                byte_compatibility.get(split, {}).get("identical") is True
                for split in ("training", "development", "public_test")
            )
            and strict_data.get("existing_checkpoint_retraining_required")
            is False
        ),
        "strict_result_receipt_passed": (
            strict_results.get("status") == "passed"
            and strict_results.get("passed") is True
            and strict_results.get("oracle", {}).get("strict_sha256")
            == release["evaluation"]["planning_oracle"]["sha256"]
            and strict_results.get("oracle", {}).get("passed") is True
        ),
        "reference_method_config_status_passed": (
            reference.get("status") == "passed"
            and reference.get("current_anti_spoof_gate_status")
            == "passed_3_of_3"
        ),
        "all_seed_receipts_passed": all(
            row["passed"] for row in per_seed.values()
        ),
        "aggregate_receipt_passed": all(aggregate_checks.values()),
    }
    return {
        "root": str(reference_root),
        "artifact_files": verified_artifacts,
        "contaminated_files": contaminated_files,
        "config_contamination": config_contamination,
        "per_seed": per_seed,
        "aggregate_checks": aggregate_checks,
        "checks": root_checks,
        "passed": all(root_checks.values()),
    }


def _strict_causal_manifest_checks(
    manifest: dict[str, Any],
    *,
    splits: tuple[str, ...],
    prefix: str,
) -> dict[str, bool]:
    """Return hard gates for the natural x0 -> x1 -> x2 -> x3 chain."""

    def checks(audit: dict[str, Any]) -> dict[str, bool]:
        return {
            "passed": audit.get("passed") is True,
            "no_state_installation_after_x0": (
                audit.get("state_installations_after_x0") == 0
            ),
            "single_simulator_per_trajectory": (
                audit.get("query_simulator_recreated") is False
            ),
            "full_x2_state_compared": (
                audit.get("full_state_dimensions") == 12
            ),
            "x2_full_state_within_tolerance": (
                audit.get("max_pair_full_state_gap", float("inf"))
                <= audit.get("full_state_tolerance", -1.0)
            ),
            "x2_pixels_identical": (
                audit.get("max_pair_query_pixel_difference") == 0
            ),
            "query_actions_identical": (
                audit.get("max_pair_query_action_difference") == 0.0
            ),
        }

    result = {
        f"{prefix}_strict_{name}": value
        for name, value in checks(
            manifest.get("strict_causal_chain_audit", {})
        ).items()
    }
    for split in splits:
        split_audit = manifest.get("splits", {}).get(split, {}).get(
            "strict_causal_chain_audit", {}
        )
        result.update(
            {
                f"{prefix}_{split}_strict_{name}": value
                for name, value in checks(split_audit).items()
            }
        )
    return result


def _cross_release_isolation_checks(
    training: dict[str, Any],
    evaluation: dict[str, Any],
) -> dict[str, bool]:
    groups = {
        "training": training["splits"]["train"],
        "development": training["splits"]["validation"],
        "public_test": evaluation["splits"]["validation"],
    }
    episodes = {
        name: {
            int(row["template"]["source_episode_index"])
            for row in split["pairs"]
        }
        for name, split in groups.items()
    }
    queries = {
        name: set(split["query_hashes"])
        for name, split in groups.items()
    }
    result: dict[str, bool] = {}
    names = tuple(groups)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            result[f"{left}_{right}_source_episodes_disjoint"] = not bool(
                episodes[left] & episodes[right]
            )
            result[f"{left}_{right}_query_pixels_disjoint"] = not bool(
                queries[left] & queries[right]
            )
    return result


def _action_strength_pair_causal_coverage(
    *,
    pair_groups: dict[str, list[dict[str, Any]]],
    split_audits: dict[str, dict[str, Any]],
    expected_counts: dict[str, int],
) -> dict[str, Any]:
    """Independently verify every paired x0 -> x1 -> x2 -> x3 receipt."""

    gate_names = (
        "pair_audit_passed",
        "common_x0",
        "continuous_trajectory",
        "x2_full_state_is_12d",
        "x2_full_state_within_tolerance",
        "query_rgb_exact",
        "query_action_exact",
        "history_effect_present",
        "true_future_effect_present",
        "no_state_installation_after_x0",
        "query_simulator_not_recreated",
    )
    passed_counts = {name: 0 for name in gate_names}
    failed_examples: dict[str, list[str]] = {
        name: [] for name in gate_names
    }
    split_reports: dict[str, Any] = {}
    maximum_query_gap = 0.0
    minimum_query_tolerance = float("inf")
    maximum_state_installations = 0
    any_query_simulator_recreated = False

    for split, rows in pair_groups.items():
        split_counts = {name: 0 for name in gate_names}
        qualified_ids: list[str] = []
        for index, row in enumerate(rows):
            audit = row.get("audit", {})
            checks = audit.get("checks", {})
            template_id = str(
                audit.get(
                    "template_id",
                    row.get("template", {}).get(
                        "template_id", f"row-{index}"
                    ),
                )
            )
            qualified_id = f"{split}/{template_id}"
            qualified_ids.append(qualified_id)
            gap = float(audit.get("query_physics_max_abs_gap", float("inf")))
            tolerance = float(
                audit.get("query_physics_tolerance", -1.0)
            )
            installations = int(
                audit.get("state_installations_after_x0", -1)
            )
            recreated = audit.get("query_simulator_recreated") is not False
            maximum_query_gap = max(maximum_query_gap, gap)
            minimum_query_tolerance = min(
                minimum_query_tolerance, tolerance
            )
            maximum_state_installations = max(
                maximum_state_installations, installations
            )
            any_query_simulator_recreated = bool(
                any_query_simulator_recreated or recreated
            )
            pair_checks = {
                "pair_audit_passed": audit.get("passed") is True,
                "common_x0": (
                    checks.get("initial_pixels_identical") is True
                    and checks.get("initial_state_identical") is True
                ),
                "continuous_trajectory": (
                    checks.get("low_recovery_natural") is True
                    and checks.get("high_recovery_natural") is True
                ),
                "x2_full_state_is_12d": (
                    audit.get("full_state_dimensions") == 12
                    and len(audit.get("full_state_components", ())) == 12
                ),
                "x2_full_state_within_tolerance": (
                    checks.get("query_physics_within_numerical_tolerance")
                    is True
                    and gap >= 0.0
                    and tolerance >= 0.0
                    and gap <= tolerance
                ),
                "query_rgb_exact": (
                    checks.get("query_pixels_identical") is True
                    and checks.get("query_matches_initial_low") is True
                    and checks.get("query_matches_initial_high") is True
                    and audit.get("pair_query_pixel_difference") == 0
                ),
                "query_action_exact": (
                    checks.get("actions_identical") is True
                    and audit.get("pair_query_action_difference") == 0.0
                ),
                "history_effect_present": (
                    checks.get("middle_pixels_different") is True
                    and float(audit.get("history_effect", 0.0)) > 0.0
                ),
                "true_future_effect_present": (
                    checks.get("future_pixels_different") is True
                    and float(audit.get("true_future_effect", 0.0)) > 0.0
                ),
                "no_state_installation_after_x0": (
                    checks.get("no_state_installations_after_x0") is True
                    and installations == 0
                ),
                "query_simulator_not_recreated": not recreated,
            }
            for name, passed in pair_checks.items():
                if passed:
                    passed_counts[name] += 1
                    split_counts[name] += 1
                elif len(failed_examples[name]) < 5:
                    failed_examples[name].append(qualified_id)

        expected = int(expected_counts[split])
        split_audit = split_audits[split]
        split_reports[split] = {
            "expected_pairs": expected,
            "manifest_pairs": len(rows),
            "manifest_pair_ids_unique": (
                len(qualified_ids) == len(set(qualified_ids))
            ),
            "strict_split_audit_pair_count": split_audit.get(
                "pair_count"
            ),
            "strict_split_audit_passed": (
                split_audit.get("passed") is True
            ),
            "passed_pair_counts": split_counts,
            "passed": bool(
                len(rows) == expected
                and len(qualified_ids) == len(set(qualified_ids))
                and split_audit.get("pair_count") == expected
                and split_audit.get("passed") is True
                and all(count == expected for count in split_counts.values())
            ),
        }

    expected_total = sum(int(value) for value in expected_counts.values())
    observed_total = sum(len(rows) for rows in pair_groups.values())
    coverage_checks = {
        "all_three_splits_present": (
            set(pair_groups) == set(expected_counts) == set(split_audits)
        ),
        "exactly_2560_pairs_covered": (
            expected_total == 2560 and observed_total == expected_total
        ),
        "all_split_receipts_match_expected_counts": all(
            report["passed"] for report in split_reports.values()
        ),
        **{
            f"all_pairs_{name}": passed_counts[name] == expected_total
            for name in gate_names
        },
    }
    return {
        "schema_version": 1,
        "expected_pair_count": expected_total,
        "audited_pair_count": observed_total,
        "split_reports": split_reports,
        "passed_pair_counts": passed_counts,
        "failed_pair_examples": failed_examples,
        "measurements": {
            "maximum_query_state_gap": maximum_query_gap,
            "minimum_query_state_tolerance": minimum_query_tolerance,
            "maximum_state_installations_after_x0": (
                maximum_state_installations
            ),
            "any_query_simulator_recreated": (
                any_query_simulator_recreated
            ),
        },
        "checks": coverage_checks,
        "passed": all(coverage_checks.values()),
    }


def audit_action_strength_icl_release(
    *,
    release_config: Path | str = DEFAULT_ACTION_STRENGTH_RELEASE_CONFIG,
    repo_root: Path | None = None,
    full: bool = False,
) -> dict[str, Any]:
    root = (repo_root or repository_root()).resolve()
    release = load_action_strength_icl_release(release_config)

    files: dict[str, Any] = {}
    for section_name in ("identity",):
        for name, specification in release[section_name].items():
            files[f"{section_name}.{name}"] = _verified_file(
                specification, repo_root=root
            )
    for section_name, specifications in (
        ("training", release["training"]["artifacts"]),
        ("evaluation", release["evaluation"]["artifacts"]),
        ("reference", release.get("reference_results", {})),
    ):
        for name, specification in specifications.items():
            key = f"{section_name}.{name}"
            valid_contract = bool(
                isinstance(specification, dict)
                and {
                    "path",
                    "sha256",
                }.issubset(specification)
                and isinstance(specification.get("path"), str)
                and specification["path"] == specification["path"].strip()
                and isinstance(specification.get("sha256"), str)
                and len(specification["sha256"]) == 64
            )
            if valid_contract:
                files[key] = _verified_file(
                    specification, repo_root=root
                )
            else:
                files[key] = {
                    "path": (
                        specification.get("path")
                        if isinstance(specification, dict)
                        else None
                    ),
                    "exists": False,
                    "expected_sha256": (
                        specification.get("sha256")
                        if isinstance(specification, dict)
                        else None
                    ),
                    "observed_sha256": None,
                    "contract_valid": False,
                    "passed": False,
                }
            if isinstance(specification, dict) and {
                "path",
                "sha256",
            }.issubset(specification) and key in files:
                files[key]["contract_valid"] = valid_contract

    training_root = resolve_contextworld_path(
        release["training"]["artifact_tree"]["root"], repo_root=root
    )
    evaluation_root = resolve_contextworld_path(
        release["evaluation"]["artifact_tree"]["root"], repo_root=root
    )
    reference_root = resolve_contextworld_path(
        release["reference_method"]["artifact_tree"]["root"],
        repo_root=root,
    )
    training_manifest = json.loads(
        (training_root / "manifest.json").read_text(encoding="utf-8")
    )
    evaluation_manifest = json.loads(
        (evaluation_root / "manifest.json").read_text(encoding="utf-8")
    )
    distribution = json.loads(
        (training_root / "distribution_audit.json").read_text(encoding="utf-8")
    )
    strict_release_audit_specification = release["training"]["artifacts"][
        "strict_release_audit"
    ]
    strict_release_audit_path = resolve_contextworld_path(
        strict_release_audit_specification["path"], repo_root=root
    )
    strict_release_audit = json.loads(
        strict_release_audit_path.read_text(encoding="utf-8")
    )
    train_rows = _lance_rows(training_root / "train.lance")
    validation_rows = _lance_rows(training_root / "validation.lance")
    confirmation_rows = _lance_rows(evaluation_root / "validation.lance")
    oracle_path = resolve_contextworld_path(
        release["evaluation"]["planning_oracle"]["path"],
        repo_root=root,
    )
    oracle_rows = json.loads(oracle_path.read_text(encoding="utf-8"))
    oracle_report_specification = release["evaluation"]["artifacts"][
        "planning_oracle_report"
    ]
    oracle_report_path = resolve_contextworld_path(
        oracle_report_specification["path"], repo_root=root
    )
    oracle_report = json.loads(
        oracle_report_path.read_text(encoding="utf-8")
    )
    oracle_ids = [str(row.get("condition_id")) for row in oracle_rows]
    oracle_pairs: dict[int, set[str]] = {}
    for row in oracle_rows:
        oracle_pairs.setdefault(int(row["pair_index"]), set()).add(
            str(row["mode"])
        )

    expected_train_pairs = int(release["training"]["train_pairs"])
    expected_validation_pairs = int(release["training"]["validation_pairs"])
    expected_confirmation_pairs = int(release["evaluation"]["pair_count"])
    pair_groups = {
        "training": training_manifest["splits"]["train"]["pairs"],
        "development": training_manifest["splits"]["validation"]["pairs"],
        "public_test": evaluation_manifest["splits"]["validation"]["pairs"],
    }
    split_audits = {
        "training": training_manifest["splits"]["train"][
            "strict_causal_chain_audit"
        ],
        "development": training_manifest["splits"]["validation"][
            "strict_causal_chain_audit"
        ],
        "public_test": evaluation_manifest["splits"]["validation"][
            "strict_causal_chain_audit"
        ],
    }
    pair_causal_coverage = _action_strength_pair_causal_coverage(
        pair_groups=pair_groups,
        split_audits=split_audits,
        expected_counts={
            "training": expected_train_pairs,
            "development": expected_validation_pairs,
            "public_test": expected_confirmation_pairs,
        },
    )
    pair_measurements = pair_causal_coverage["measurements"]
    causal_data = audit_causal_data_contract(
        component_id="pusht_action_strength_icl",
        evidence_scope=(
            f"{expected_train_pairs + expected_validation_pairs + expected_confirmation_pairs} "
            "Training / Development / Public Test pairs"
        ),
        continuous_environment_trajectory=bool(
            pair_causal_coverage["checks"][
                "all_pairs_continuous_trajectory"
            ]
            and pair_causal_coverage["checks"][
                "all_pairs_pair_audit_passed"
            ]
        ),
        state_installations_after_x0=int(
            pair_measurements["maximum_state_installations_after_x0"]
        ),
        query_simulator_recreated=bool(
            pair_measurements["any_query_simulator_recreated"]
        ),
        maximum_query_state_gap=float(
            pair_measurements["maximum_query_state_gap"]
        ),
        query_state_tolerance=float(
            pair_measurements["minimum_query_state_tolerance"]
        ),
        query_pixels_exact=bool(
            pair_causal_coverage["checks"]["all_pairs_query_rgb_exact"]
        ),
        query_actions_exact=bool(
            pair_causal_coverage["checks"][
                "all_pairs_query_action_exact"
            ]
        ),
        history_effect_present=bool(
            pair_causal_coverage["checks"][
                "all_pairs_history_effect_present"
            ]
        ),
        true_future_effect_present=bool(
            pair_causal_coverage["checks"][
                "all_pairs_true_future_effect_present"
            ]
        ),
        x0_policy="shared_visible_start",
        x0_static_leakage_check_passed=bool(
            pair_causal_coverage["checks"]["all_pairs_common_x0"]
        ),
        evidence=(
            release["training"]["artifacts"]["manifest"]["path"],
            release["evaluation"]["artifacts"]["manifest"]["path"],
            strict_release_audit_specification["path"],
        ),
    )
    causal_data["pair_coverage"] = pair_causal_coverage
    causal_data["checks"]["all_pairs_individually_audited"] = bool(
        pair_causal_coverage["passed"]
    )
    causal_data["checks"]["query_state_is_12d_for_every_pair"] = bool(
        pair_causal_coverage["checks"]["all_pairs_x2_full_state_is_12d"]
    )
    causal_data["passed"] = all(causal_data["checks"].values())
    data_checks = {
        "all_reference_results_enumerated": (
            {
                key.removeprefix("reference.")
                for key in files
                if key.startswith("reference.")
            }
            == set(release.get("reference_results", {}))
        ),
        "all_reference_result_contracts_valid": all(
            files[f"reference.{name}"].get("contract_valid") is True
            for name in release.get("reference_results", {})
        ),
        "causal_data_contract_passed": causal_data["passed"] is True,
        "causal_data_contract_covers_all_2560_pairs": (
            pair_causal_coverage["passed"] is True
            and pair_causal_coverage["audited_pair_count"] == 2560
        ),
        "training_manifest_passed": training_manifest.get("passed") is True,
        "training_cross_split_passed": (
            training_manifest.get("cross_split_audit", {}).get("passed") is True
        ),
        "train_pair_count": (
            training_manifest["pair_counts"]["train"] == expected_train_pairs
        ),
        "validation_pair_count": (
            training_manifest["pair_counts"]["validation"]
            == expected_validation_pairs
        ),
        "train_lance_rows": train_rows == expected_train_pairs * 2 * 20,
        "validation_lance_rows": (
            validation_rows == expected_validation_pairs * 2 * 20
        ),
        "distribution_audit_passed": distribution.get("passed") is True,
        "distribution_strict_causal_chain_passed": (
            distribution.get("strict_causal_chain", {}).get("passed")
            is True
        ),
        "distribution_source_population_identity_passed": (
            distribution.get("source_population_identity", {}).get(
                "passed"
            )
            is True
        ),
        "strict_release_audit_passed": (
            strict_release_audit.get("passed") is True
        ),
        "strict_release_training_bytes_compatible": (
            strict_release_audit.get(
                "model_visible_byte_compatibility", {}
            )
            .get("training", {})
            .get("identical")
            is True
        ),
        "strict_release_development_bytes_compatible": (
            strict_release_audit.get(
                "model_visible_byte_compatibility", {}
            )
            .get("development", {})
            .get("identical")
            is True
        ),
        "strict_release_public_test_bytes_compatible": (
            strict_release_audit.get(
                "model_visible_byte_compatibility", {}
            )
            .get("public_test", {})
            .get("identical")
            is True
        ),
        "strict_release_existing_checkpoints_remain_valid": (
            strict_release_audit.get(
                "existing_checkpoint_retraining_required"
            )
            is False
        ),
        "strict_release_all_splits_isolated": (
            strict_release_audit.get("cross_split_isolation", {}).get(
                "passed"
            )
            is True
        ),
        "confirmation_manifest_passed": evaluation_manifest.get("passed") is True,
        "confirmation_overlap_audit_passed": (
            evaluation_manifest.get("cross_split_audit", {}).get("passed")
            is True
        ),
        "confirmation_pair_count": (
            evaluation_manifest.get("pair_count") == expected_confirmation_pairs
        ),
        "confirmation_lance_rows": (
            confirmation_rows == expected_confirmation_pairs * 2 * 20
        ),
        "planning_oracle_condition_count": (
            len(oracle_rows)
            == int(release["evaluation"]["planning_oracle"]["condition_count"])
        ),
        "planning_oracle_condition_ids_unique": (
            len(oracle_ids) == len(set(oracle_ids))
        ),
        "planning_oracle_pairs_complete": (
            len(oracle_pairs) == expected_confirmation_pairs
            and all(
                modes == {"low_gain", "high_gain"}
                for modes in oracle_pairs.values()
            )
        ),
        "planning_oracle_report_passed": (
            oracle_report.get("passed") is True
        ),
        "planning_oracle_bound_to_public_test_manifest": (
            oracle_report.get("public_test_manifest_sha256")
            == release["evaluation"]["manifest_sha256"]
        ),
        "planning_oracle_bound_to_public_test_portability_receipt": (
            oracle_report.get("public_test_portability_receipt_sha256")
            == release["evaluation"]["artifacts"][
                "portability_receipt"
            ]["sha256"]
        ),
        "planning_oracle_replays_natural_prefix": (
            oracle_report.get("causal_execution", {}).get(
                "prefix_replayed_before_each_candidate"
            )
            is True
        ),
        "planning_oracle_no_state_installation_after_x0": (
            oracle_report.get("causal_execution", {}).get(
                "state_installations_after_x0"
            )
            == 0
        ),
        "planning_oracle_single_simulator_per_trajectory": (
            oracle_report.get("causal_execution", {}).get(
                "query_simulator_recreated"
            )
            is False
        ),
        **_strict_causal_manifest_checks(
            training_manifest,
            splits=("train", "validation"),
            prefix="training",
        ),
        **_strict_causal_manifest_checks(
            evaluation_manifest,
            splits=("validation",),
            prefix="public_test",
        ),
        **_cross_release_isolation_checks(
            training_manifest,
            evaluation_manifest,
        ),
    }

    tree_audits: dict[str, Any] = {}
    for name, path, specification in (
        (
            "training",
            training_root,
            release["training"]["artifact_tree"],
        ),
        (
            "evaluation",
            evaluation_root,
            release["evaluation"]["artifact_tree"],
        ),
        (
            "reference_method",
            reference_root,
            release["reference_method"]["artifact_tree"],
        ),
    ):
        children = [value for value in path.rglob("*") if value.is_file()]
        observed_files = len(children)
        observed_bytes = sum(value.stat().st_size for value in children)
        observed_hash = directory_sha256(path) if full else None
        tree_audits[name] = {
            "path": str(path),
            "exists": path.is_dir(),
            "expected_files": int(specification["files"]),
            "observed_files": observed_files,
            "expected_bytes": int(specification["bytes"]),
            "observed_bytes": observed_bytes,
            "expected_sha256": specification["sha256"],
            "observed_sha256": observed_hash,
            "full_hash_checked": full,
            "passed": bool(
                path.is_dir()
                and observed_files == int(specification["files"])
                and observed_bytes == int(specification["bytes"])
                and (
                    not full
                    or observed_hash == specification["sha256"]
                )
            ),
        }

    payload_audit: dict[str, Any] = {
        "decoded_pairs": 0,
        "passed": True,
    }
    if full:
        dataset = ActionStrengthICLEvalDataset(
            release=release,
            repo_root=root,
        )
        expected_condition_ids = {
            f"{pair_id}/{mode}"
            for pair_id in dataset.arrays.pair_ids
            for mode in ("low_gain", "high_gain")
        }
        payload_audit = {
            "decoded_pairs": dataset.arrays.pair_count,
            "condition_count": 2 * dataset.arrays.pair_count,
            "planning_oracle_matches_pairs": (
                expected_condition_ids == set(oracle_ids)
            ),
            "passed": (
                dataset.is_full_protocol
                and expected_condition_ids == set(oracle_ids)
            ),
        }

    upstream_specifications = {
        "original_h5": release["training"]["upstream"]["original_h5"],
        "original_lance": release["training"]["upstream"][
            "original_lance"
        ],
        "initial_checkpoint": release["training"]["initialization"],
    }
    upstream_audits = {
        name: {
            "source_symbol": specification["source_symbol"],
            "environment_variable": specification["environment_variable"],
            "bundled_artifact_path": specification[
                "bundled_artifact_path"
            ],
            "required_for_release_audit": False,
            "passed": True,
        }
        for name, specification in upstream_specifications.items()
    }
    reference_method_audit = _reference_method_release_audit(
        release,
        repo_root=root,
    )

    passed = bool(
        all(row["passed"] for row in files.values())
        and all(data_checks.values())
        and all(row["passed"] for row in tree_audits.values())
        and payload_audit["passed"]
        and reference_method_audit["passed"]
        and causal_data["passed"]
    )
    return {
        "schema_version": 1,
        "release_id": release["release_id"],
        "status": "passed" if passed else "failed",
        "full_content_hash_audit": full,
        "files": files,
        "data_checks": data_checks,
        "artifact_trees": tree_audits,
        "payload_audit": payload_audit,
        "causal_data_contract": causal_data,
        "reference_method": reference_method_audit,
        "upstream_inputs": upstream_audits,
        "counts": {
            "training_pairs": expected_train_pairs,
            "development_pairs": expected_validation_pairs,
            "confirmation_pairs": expected_confirmation_pairs,
        },
        "sealed_test_included": False,
        "passed": passed,
    }


def action_strength_icl_training_plan(
    recipe: str,
    *,
    training_seed: int,
    output: Path | str,
    release_config: Path | str = DEFAULT_ACTION_STRENGTH_RELEASE_CONFIG,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Return a runnable Stable-WorldModel reference/control command.

    The reference recipe is an example proving learnability.  The benchmark
    does not require external methods to use its regularizer.
    """

    root = (repo_root or repository_root()).resolve()
    release = load_action_strength_icl_release(release_config)
    recipes = release["training"]["recipes"]
    if recipe not in recipes:
        raise KeyError(
            f"Unknown Action Strength recipe {recipe!r}; "
            f"available={sorted(recipes)}"
        )
    specification = recipes[recipe]
    hidden_root = resolve_contextworld_path(
        release["training"]["artifact_tree"]["root"], repo_root=root
    )
    original_lance = resolve_action_strength_original_lance(
        release, repo_root=root
    )
    original_h5 = resolve_action_strength_original_h5(release, repo_root=root)
    checkpoint = resolve_action_strength_initial_checkpoint(
        release, repo_root=root
    )
    contrast = resolve_contextworld_path(
        release["training"]["contrast_scales"]["path"], repo_root=root
    )
    command = [
        "python",
        "scripts/run_pusht_hidden_actuation_mixed.py",
        "--hidden-data-root",
        str(hidden_root),
        "--original-lance",
        str(original_lance),
        "--action-normalizer-source",
        str(original_h5),
        "--checkpoint",
        str(checkpoint),
        "--output",
        str(Path(output).expanduser().resolve()),
        "--variants",
        str(specification["runner_variant"]),
        "--max-steps",
        str(specification["optimizer_steps"]),
        "--seed",
        str(int(training_seed)),
        "--batch-size",
        str(specification["batch_size"]),
        "--original-batch-size",
        str(specification["original_batch_size"]),
    ]
    if specification.get("uses_contrast_scales"):
        command.extend(["--contrast-scales", str(contrast)])
    return {
        "schema_version": 1,
        "release_id": release["release_id"],
        "recipe": recipe,
        "display_name": specification["display_name"],
        "training_seed": int(training_seed),
        "method_specific_reference_recipe": bool(
            specification.get("method_specific_reference_recipe", False)
        ),
        "benchmark_requires_this_objective": False,
        "command": command,
        "shell": " ".join(command),
        "data": {
            "paired_training": str(hidden_root / "train.lance"),
            "development_validation": str(
                hidden_root / "validation.lance"
            ),
            "independent_confirmation": str(
                resolve_contextworld_path(
                    release["evaluation"]["artifact_tree"]["root"],
                    repo_root=root,
                )
                / "validation.lance"
            ),
            "standard_replay": str(original_lance),
        },
    }


def action_strength_icl_evaluation_plans(
    *,
    checkpoint: Path | str,
    model_name: str,
    output_root: Path | str,
    release_config: Path | str = DEFAULT_ACTION_STRENGTH_RELEASE_CONFIG,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Return the frozen Stable-WorldModel planning and retention commands."""

    root = (repo_root or repository_root()).resolve()
    release = load_action_strength_icl_release(release_config)
    checkpoint = Path(checkpoint).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    confirmation = resolve_contextworld_path(
        release["evaluation"]["artifact_tree"]["root"],
        repo_root=root,
    )
    original_h5 = resolve_action_strength_original_h5(
        release,
        repo_root=root,
    )
    planning_output = output_root / "action_strength_planning"
    retention_output = output_root / "standard_pusht_retention"
    planning = [
        "python",
        "scripts/eval_pusht_replay_matched_hidden_cem.py",
        "--data-root",
        str(confirmation),
        "--original-dataset",
        str(original_h5),
        "--checkpoint",
        str(checkpoint),
        "--output",
        str(planning_output),
    ]
    retention = [
        "python",
        "scripts/eval_pusht_standard_cem_retention.py",
        "--model",
        f"{model_name}={checkpoint}",
        "--dataset",
        str(original_h5),
        "--output",
        str(retention_output),
    ]
    return {
        "schema_version": 1,
        "release_id": release["release_id"],
        "model_name": str(model_name),
        "checkpoint": str(checkpoint),
        "commands": {
            "action_strength_planning": {
                "argv": planning,
                "shell": " ".join(planning),
                "result": str(planning_output / "aggregate.json"),
                "score_command": (
                    "contextworld-action-strength score-planning "
                    f"--submission {planning_output / 'aggregate.json'} "
                    f"--output {output_root / 'planning-score.json'}"
                ),
            },
            "standard_pusht_retention": {
                "argv": retention,
                "shell": " ".join(retention),
                "result": str(retention_output / "aggregate.json"),
                "score_command": (
                    "contextworld-action-strength score-retention "
                    f"--report {retention_output / 'aggregate.json'} "
                    f"--model-name {model_name} "
                    f"--output {output_root / 'retention-score.json'}"
                ),
            },
        },
        "note": (
            "The two tracks are reported separately; neither replaces the "
            "frozen real-future prediction score."
        ),
    }


__all__ = [
    "ACTION_STRENGTH_RELEASE_ID",
    "DEFAULT_ACTION_STRENGTH_RELEASE_CONFIG",
    "ActionStrengthEvalArrays",
    "ActionStrengthICLEvalDataset",
    "action_strength_icl_evaluation_plans",
    "action_strength_icl_training_plan",
    "audit_action_strength_icl_release",
    "directory_sha256",
    "file_sha256",
    "load_action_strength_icl_release",
    "resolve_action_strength_initial_checkpoint",
    "resolve_action_strength_original_h5",
    "resolve_action_strength_original_lance",
]
