from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
E4_SPEEDS = {3.1, 3.3, 3.5, 4.1, 5.0, 5.1, 5.9, 7.0}
REPLACED_SPEEDS = {3.0, 3.4, 3.6, 4.0, 4.8, 5.2, 6.0, 7.1}


def _load(path: str) -> dict:
    return yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))


def _scenario(config: dict, split: str) -> dict:
    return next(value for value in config["scenario_sets"] if value["split"] == split)


def test_speedseen_changes_only_exact_speed_support() -> None:
    clean = _load("configs/synthesis/tworoom_speed_clean_v1.yaml")
    seen = _load("configs/synthesis/tworoom_speed_seen_v1.yaml")
    clean_train = _scenario(clean, "train")
    seen_train = _scenario(seen, "train")
    clean_speeds = set(clean_train["values"]["values"])
    seen_speeds = set(seen_train["values"]["values"])

    assert len(clean_speeds) == len(seen_speeds) == 32
    assert seen_speeds & E4_SPEEDS == E4_SPEEDS
    assert clean_speeds & E4_SPEEDS == set()
    assert clean_speeds - seen_speeds == REPLACED_SPEEDS
    assert seen_speeds - clean_speeds == E4_SPEEDS
    assert clean_train["episodes_per_scenario"] == seen_train["episodes_per_scenario"] == 32
    assert clean_train["seed_groups"] == seen_train["seed_groups"]
    assert clean["collection"] == seen["collection"]
    assert clean["scenario_generation_seed"] == seen["scenario_generation_seed"]
    assert _scenario(clean, "val") == _scenario(seen, "val")


def test_speedseen_training_budget_remains_half_original_half_synthetic() -> None:
    benchmark = _load("configs/benchmark/tworoom_speed_seen_v1.yaml")
    assert benchmark["training_protocol"]["group_sampling"]["M_speed"] == {
        "original": 0.5,
        "speed": 0.5,
    }
    assert benchmark["models"][0]["training_groups"] == ["original", "speed"]
    assert (
        benchmark["training_protocol"]["formal_e4_role"]
        == "seen_speed_diagnostic_not_heldout_validation"
    )


def test_speedseen_launchers_use_distinct_run_and_artifact_names() -> None:
    training = (ROOT / "scripts/run_h3_speedseen_train.sh").read_text(encoding="utf-8")
    evaluation = (ROOT / "scripts/run_h3_speedseen_eval.sh").read_text(encoding="utf-8")
    parallel_e4 = (ROOT / "scripts/run_h3_speedseen_e4_parallel.sh").read_text(encoding="utf-8")
    assert "h3_speedseen_s${TRAINING_SEED}" in training
    assert "tworoom_speed_seen_v1.yaml" in training
    assert "h3_speedseen_s3072/weights_final_step_6420.pt" in evaluation
    assert '$OUTPUT_DIR/id_retention_n50x6.json' in evaluation
    assert "--reuse-existing" in evaluation
    assert "CUDA_VISIBLE_DEVICES" in parallel_e4
    assert "--skip-catalog-replay" in parallel_e4
    assert '"$seed" != "42"' in parallel_e4
    assert "--suite e4-context-plan" in parallel_e4
    assert "--reuse-existing" in parallel_e4
