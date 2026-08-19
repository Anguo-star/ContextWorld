from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]


def _load() -> object:
    path = ROOT / "scripts/eval_cube_original_baseline_cem_frozen_v1.py"
    spec = importlib.util.spec_from_file_location(
        "test_cube_original_baseline_cem_frozen_v1", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _args(*, preflight: bool = True) -> list[str]:
    values = [
        "--stable-worldmodel-root",
        "/tmp/stable-worldmodel",
        "--expected-ref",
        "a" * 40,
        "--expected-plan-config-sha256",
        "f" * 64,
        "--expected-plan-config-size",
        "5",
        "--checkpoint",
        "/tmp/cube_pldm.ckpt",
        "--expected-checkpoint-sha256",
        "b" * 64,
        "--expected-checkpoint-size",
        "1",
        "--expected-config-sha256",
        "c" * 64,
        "--expected-config-size",
        "2",
        "--dataset",
        "/tmp/cube.h5",
        "--expected-dataset-sha256",
        "d" * 64,
        "--expected-dataset-size",
        "3",
        "--input-identity-audit",
        "/tmp/input-audit.json",
        "--expected-input-identity-audit-sha256",
        "1" * 64,
        "--expected-input-identity-audit-size",
        "6",
        "--query-catalog",
        "/tmp/catalog.json",
        "--expected-catalog-sha256",
        "e" * 64,
        "--expected-catalog-size",
        "4",
    ]
    if preflight:
        return ["--preflight", *values]
    return [*values, "--output", "/tmp/output"]


def test_preflight_cli_requires_no_output_and_evaluation_requires_one() -> None:
    runner = _load()
    args = runner.parse_args(_args(preflight=True))
    assert args.preflight is True
    assert args.output is None

    evaluated = runner.parse_args(_args(preflight=False))
    assert evaluated.preflight is False
    assert evaluated.output == Path("/tmp/output")

    with pytest.raises(SystemExit):
        runner.parse_args([*_args(preflight=True), "--output", "/tmp/output"])
    with pytest.raises(SystemExit):
        runner.parse_args(_args(preflight=False)[:-2])


def test_identity_check_hashes_content_and_rejects_drift(tmp_path: Path) -> None:
    runner = _load()
    source = tmp_path / "catalog.json"
    source.write_text("frozen", encoding="utf-8")
    digest = runner.base.file_sha256(source)
    identity = runner._assert_file_identity(
        source,
        expected_sha256=digest,
        expected_size=source.stat().st_size,
        label="catalog",
    )
    assert identity["content_hash_checked_in_job"] is True

    with pytest.raises(RuntimeError, match="size drifted"):
        runner._assert_file_identity(
            source,
            expected_sha256=digest,
            expected_size=source.stat().st_size + 1,
            label="catalog",
        )


def test_actual_model_state_identity_is_stable_and_output_is_exclusive(
    tmp_path: Path,
) -> None:
    runner = _load()
    model = torch.nn.Linear(3, 2)
    before = runner._model_state(model)
    after = runner._model_state(model)
    assert before == after
    assert len(before["state_dict_sha256"]) == 64

    output = tmp_path / "exclusive"
    assert runner._reserve_output(output) == output.resolve()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        runner._reserve_output(output)


def test_fixed_three_seed_query_contract_is_explicit() -> None:
    runner = _load()
    assert runner.EVAL_SEEDS == (42, 43, 44)
    assert runner.QUERIES_PER_SEED == 100
    assert runner.TOTAL_EVALUATIONS == 300

    payload = {
        "selection": {
            "algorithm": "numpy_default_rng_choice_sorted_valid_rows",
            "historical_final_index_exclusion": True,
            "goal_offset_steps": 25,
            "eval_seeds": [42, 43, 44],
            "queries_per_seed": 100,
        }
    }
    queries = {
        seed: {"row_indices": list(range(100))}
        for seed in runner.EVAL_SEEDS
    }
    runner._assert_query_contract(payload, queries)

    payload["selection"]["queries_per_seed"] = 50
    with pytest.raises(RuntimeError, match="selection contract drifted"):
        runner._assert_query_contract(payload, queries)


def test_direct_cube_row_validation_reads_only_identity_columns() -> None:
    runner = _load()
    episode = np.asarray([0] * 30 + [1] * 30, dtype=np.int64)
    step = np.asarray(list(range(30)) * 2, dtype=np.int64)
    dataset = type(
        "Dataset",
        (),
        {
            "column_names": ("ep_idx", "step_idx", "pixels"),
            "h5_file": {"ep_idx": episode, "step_idx": step},
        },
    )()
    runner._verify_direct_rows(
        dataset,
        rows=np.asarray([0, 30]),
        episodes=np.asarray([0, 1]),
        starts=np.asarray([0, 0]),
    )
    with pytest.raises(RuntimeError, match="non-eligible starts"):
        runner._verify_direct_rows(
            dataset,
            rows=np.asarray([5]),
            episodes=np.asarray([0]),
            starts=np.asarray([5]),
        )
