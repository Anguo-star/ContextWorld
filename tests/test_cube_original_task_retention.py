from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest
import yaml

from contextworld.benchmarks import cube_original_task_retention as retention


ROOT = Path(__file__).resolve().parents[1]


def _outcomes(count: int) -> list[bool]:
    return [True] * count + [False] * (100 - count)


def _validated_row(
    name: str, counts: tuple[int, int, int], *, checkpoint: str
) -> dict[str, object]:
    seeds = [
        {"eval_seed": seed, "episode_successes": _outcomes(count)}
        for seed, count in zip(retention.EVAL_SEEDS, counts, strict=True)
    ]
    return {
        "model_name": name,
        "checkpoint": Path(checkpoint),
        "checkpoint_sha256": name.ljust(64, "0")[:64],
        "report_identity": {
            "path": f"/{name}/aggregate.json",
            "sha256": name.ljust(64, "1")[:64],
            "size_bytes": 1,
        },
        "model_payload": {"seeds": seeds},
        "success_count": sum(counts),
    }


def test_cube_cem_noninferiority_uses_paired_300_query_margin() -> None:
    baseline = _validated_row("baseline_lewm", (60, 55, 50), checkpoint="/b.ckpt")
    edge = _validated_row("lewm_seed17321", (55, 50, 45), checkpoint="/c.pt")
    failed = _validated_row("lewm_seed17322", (54, 50, 45), checkpoint="/d.pt")

    edge_result = retention.paired_cube_cem_noninferiority(
        baseline, edge, training_seed=17321
    )
    failed_result = retention.paired_cube_cem_noninferiority(
        baseline, failed, training_seed=17322
    )

    assert edge_result["evaluation_count"] == 300
    assert edge_result["success_delta"] == -15
    assert edge_result["passed"] is True
    assert failed_result["success_delta"] == -16
    assert failed_result["passed"] is False
    assert [row["eval_seed"] for row in edge_result["by_eval_seed"]] == [42, 43, 44]


def test_cube_cem_prereg_keeps_public_closed_and_only_lewm_authorized(
    tmp_path: Path,
) -> None:
    prereg = retention.load_cube_cem_retention_prereg(require_freeze=False)
    assert prereg["scope"]["passing_families"] == ["lewm"]
    assert prereg["public_test"] == retention.closed_public_contract()
    assert [
        row["training_seed"] for row in prereg["authorization"]["candidates"]
    ] == [17321, 17322, 17323]

    contaminated = copy.deepcopy(prereg)
    contaminated.pop("_config_path")
    contaminated["public_test"]["opened"] = True
    path = tmp_path / "contaminated.yaml"
    path.write_text(yaml.safe_dump(contaminated), encoding="utf-8")
    with pytest.raises(ValueError, match="contract drifted"):
        retention.load_cube_cem_retention_prereg(path, require_freeze=False)


def test_cube_cem_preregistered_source_identities_match() -> None:
    prereg = retention.load_cube_cem_retention_prereg(require_freeze=False)
    for name, entry in prereg["identity"].items():
        path = retention.resolve_declared_path(entry["path"], repo_root=ROOT)
        assert path.is_file(), name
        assert not path.is_symlink(), name
        assert retention.file_sha256(path) == entry["sha256"], name
        assert path.stat().st_size == entry["size_bytes"], name


def test_frozen_cube_evaluator_requires_explicit_runtime_and_shared_catalog() -> None:
    path = ROOT / "scripts/eval_cube_original_task_cem_frozen.py"
    source = path.read_text(encoding="utf-8")
    assert "--stable-worldmodel-root" in source
    assert "--query-catalog" in source
    assert "ROOT.parent" not in source
    assert "history_len=3" in source
    assert "action_block=5" in source
    assert "num_samples=300" in source
    assert "n_steps=30" in source
    assert "topk=30" in source


def test_cube_query_catalog_selection_is_deterministic() -> None:
    script = ROOT / "scripts/eval_cube_original_task_cem_frozen.py"
    spec = importlib.util.spec_from_file_location("cube_cem_frozen", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class Dataset:
        column_names = ["episode_idx", "step_idx"]

        def __init__(self) -> None:
            self.episode = [0] * 40 + [1] * 40 + [2] * 40
            self.step = list(range(40)) * 3

        def get_col_data(self, name: str):
            import numpy as np

            return np.asarray(self.episode if name == "episode_idx" else self.step)

        def get_row_data(self, rows):
            import numpy as np

            values = [int(value) for value in rows]
            return {
                "episode_idx": np.asarray([self.episode[index] for index in values]),
                "step_idx": np.asarray([self.step[index] for index in values]),
            }

    dataset = Dataset()
    first = module._catalog_payload(dataset, seeds=(42,), count=10)
    second = module._catalog_payload(dataset, seeds=(42,), count=10)
    assert first == second
    assert first["selection"]["historical_final_index_exclusion"] is True
    assert len(first["queries"]["42"]["row_indices"]) == 10
