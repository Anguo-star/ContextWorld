from __future__ import annotations

import json
from pathlib import Path

import yaml

from contextworld.benchmarks import action_delay_icl_data
from contextworld.benchmarks.action_delay_icl_data import (
    action_delay_icl_training_plan,
    audit_action_delay_icl_release,
)
from contextworld.evaluation.action_delay_h7_validation import file_sha256
from contextworld.benchmarks.suite_data import _action_delay_export_entries
from contextworld.paths import repository_root
from contextworld.training.tworoom_data import resolve_tworoom_original_h5


def test_public_training_plan_exposes_both_one_step_stages(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo = tmp_path / "repo"
    (repo / "configs").mkdir(parents=True)
    (repo / "scripts").mkdir()
    coarse_config = repo / "configs/coarse.yaml"
    refinement_config = repo / "configs/refinement.yaml"
    coarse_launcher = repo / "scripts/coarse.sh"
    refinement_launcher = repo / "scripts/refinement.sh"
    for path, content in (
        (coarse_config, "stage: coarse\n"),
        (refinement_config, "stage: refinement\n"),
        (coarse_launcher, "#!/usr/bin/env bash\n"),
        (refinement_launcher, "#!/usr/bin/env bash\n"),
    ):
        path.write_text(content, encoding="utf-8")

    release = {
        "release_id": action_delay_icl_data.RELEASE_ID,
        "training": {
            "recipes": {
                "pldm_reference": {
                    "model_family": "pldm",
                    "model_id": "H7_ActionDelay_Reference_PLDM",
                    "training_seeds": [3072, 4096, 5120],
                    "stages": [
                        {
                            "name": "coarse_delay_learning",
                            "config": "configs/coarse.yaml",
                            "config_sha256": file_sha256(coarse_config),
                            "launcher": "scripts/coarse.sh",
                        },
                        {
                            "name": "full_delay_refinement",
                            "config": "configs/refinement.yaml",
                            "config_sha256": file_sha256(
                                refinement_config
                            ),
                            "launcher": "scripts/refinement.sh",
                        },
                    ],
                }
            }
        },
    }
    monkeypatch.setattr(
        action_delay_icl_data,
        "load_action_delay_icl_release",
        lambda *args, **kwargs: release,
    )

    plan = action_delay_icl_training_plan(
        "pldm_reference",
        training_seed=4096,
        repo_root=repo,
    )

    assert plan["model_family"] == "pldm"
    assert [row["stage"] for row in plan["stages"]] == [
        "coarse_delay_learning",
        "full_delay_refinement",
    ]
    assert plan["commands"] == [
        ["bash", str(coarse_launcher), "pldm", "4096"],
        ["bash", str(refinement_launcher), "pldm", "4096"],
    ]


def test_release_audit_checks_every_training_stage(
    tmp_path: Path,
    monkeypatch,
) -> None:
    release_config = tmp_path / "release.yaml"
    release_config.write_text("release: test\n", encoding="utf-8")
    stages = {}
    for name, query_key in (
        ("coarse_delay_learning", "query_triplets"),
        ("full_delay_refinement", "query_bundles"),
    ):
        root = tmp_path / name
        shard = root / "shard"
        shard.mkdir(parents=True)
        build = root / "build.json"
        manifest = root / "manifest.jsonl"
        build.write_text(
            json.dumps(
                {
                    "passed": True,
                    "checks": {
                        "all_shards_pass_raw_physical_replay": True,
                    },
                    "pairing": {
                        "passed": True,
                        "checks": {
                            "query_pixels_exact": True,
                            "commanded_actions_exact": True,
                            "initial_pixels_exact": True,
                        },
                    },
                    "physical_counts": {
                        "shards": 1,
                        "episodes": 11,
                        "raw_rows_replayed": 550,
                        query_key: 7,
                    },
                }
            ),
            encoding="utf-8",
        )
        manifest.write_text(
            json.dumps(
                {
                    "output_path": str(shard),
                    "storage_sha256": "0" * 64,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        stages[name] = {
            "artifact_tree": {"root": str(root)},
            "artifacts": {
                "build_report": {
                    "path": str(build),
                    "sha256": file_sha256(build),
                },
                "manifest": {
                    "path": str(manifest),
                    "sha256": file_sha256(manifest),
                },
            },
        }
    evaluation_build = tmp_path / "evaluation-build.json"
    evaluation_build.write_text(
        json.dumps(
            {
                "status": "passed",
                "checks": {"every_physical_family_passed": True},
                "physical_equivalence": {
                    "horizon1": {"distinct_groups": 6},
                },
            }
        ),
        encoding="utf-8",
    )
    evaluation_audit = tmp_path / "evaluation-audit.json"
    evaluation_audit.write_text(
        json.dumps(
            {
                "status": "passed",
                "checks": {"full_physical_replay_completed": True},
            }
        ),
        encoding="utf-8",
    )
    validation_contract = tmp_path / "validation-contract.py"
    validation_contract.write_text("# test contract\n", encoding="utf-8")
    release = {
        "release_id": action_delay_icl_data.RELEASE_ID,
        "_config_path": str(release_config),
        "identity": {
            "validation_contract": {
                "path": str(validation_contract),
                "sha256": file_sha256(validation_contract),
            }
        },
        "training": {"stages": stages},
        "evaluation": {
            "artifacts": {
                "build_report": {
                    "path": str(evaluation_build),
                    "sha256": file_sha256(evaluation_build),
                },
                "audit_report": {
                    "path": str(evaluation_audit),
                    "sha256": file_sha256(evaluation_audit),
                },
            }
        },
        "reference_results": {},
    }
    monkeypatch.setattr(
        action_delay_icl_data,
        "load_action_delay_icl_release",
        lambda *args, **kwargs: release,
    )

    audit = audit_action_delay_icl_release(
        release_config=release_config,
        repo_root=tmp_path,
    )

    assert audit["passed"] is True
    assert set(audit["training_data"]["stages"]) == set(stages)
    assert audit["training_data"]["episodes"] == 22
    assert audit["training_data"]["raw_rows_replayed"] == 1100
    assert audit["training_data"]["shards_present"] == 2


def test_suite_export_bundles_action_delay_receipts_with_public_test() -> None:
    evaluation_root = "artifacts/evaluation/action-delay-public-test"
    release = {
        "training": {
            "stages": {
                "coarse": {
                    "artifact_tree": {
                        "root": "artifacts/synthesis/action-delay-coarse"
                    }
                },
                "refinement": {
                    "artifact_tree": {
                        "root": "artifacts/synthesis/action-delay-refinement"
                    }
                },
            },
            "initialization": {
                "checkpoint": "artifacts/training/init.pt",
                "checkpoint_config": "artifacts/training/init.json",
            },
        },
        "evaluation": {
            "artifact_tree": {"root": evaluation_root},
            "normalizer": "artifacts/splits/normalizer.json",
            "artifacts": {
                "current_result": {
                    "path": (
                        evaluation_root
                        + "/score_receipts/model_results/result.json"
                    ),
                    "sha256": "a" * 64,
                }
            },
        },
        "reference_results": {
            "core": {
                "path": evaluation_root + "/score_receipts/core_summary.json",
                "sha256": "b" * 64,
            },
            "external": {
                "path": "artifacts/evaluation/external-receipt.json",
                "sha256": "c" * 64,
            },
        },
    }

    entries = _action_delay_export_entries(release)

    assert entries == [
        ("artifacts/synthesis/action-delay-coarse", "directory"),
        ("artifacts/synthesis/action-delay-refinement", "directory"),
        (evaluation_root, "directory"),
        ("artifacts/splits/normalizer.json", "file"),
        ("artifacts/training/init.pt", "file"),
        ("artifacts/training/init.json", "file"),
        ("artifacts/evaluation/external-receipt.json", "file"),
    ]


def test_current_public_release_is_repo_local_and_current_only() -> None:
    audit = audit_action_delay_icl_release(full=False)

    assert audit["passed"] is True
    assert all(
        row["repository_local"] and not row["symlinks"]
        for row in audit["artifact_trees"].values()
    )
    assert audit["portability"] == {
        "repository_local_artifacts": True,
        "json_files_checked": 22,
        "violations": [],
        "passed": True,
        "skipped": False,
    }
    assert audit["public_test_layout"]["assets"] == 300
    assert audit["public_test_layout"]["score_receipts"] == 10
    assert audit["public_test_layout"]["forbidden_files"] == []
    assert audit["reference_results"]["current_only"] is True
    assert audit["reference_results"]["model_results"] == 6
    assert audit["reference_results"]["pldm_training_seeds_passed"] == 3
    assert audit["reference_results"]["cem_target_seeds_passed"] == 3


def test_public_training_recipes_exclude_internal_iteration_narrative() -> None:
    root = repository_root()
    recipe_names = (
        "tworoom_action_delay_h7_paired_lewm_v1.yaml",
        "tworoom_action_delay_h7_paired_pldm_v1.yaml",
        "tworoom_action_delay_h7_curriculum_lewm_v4.yaml",
        "tworoom_action_delay_h7_curriculum_pldm_v4.yaml",
    )
    forbidden = (
        "../../data/",
        "diagnostic",
        "post_hoc",
        "pilot_",
        "evidence_before_freeze",
        "not_started",
        "preregistered_before",
        "frozen_after",
    )
    for name in recipe_names:
        path = root / "configs/benchmark" / name
        text = path.read_text(encoding="utf-8")
        payload = yaml.safe_load(text)
        assert payload["data"]["original_read_only"] == (
            "upstream/lewm-tworooms/tworoom.h5"
        )
        assert payload["status"].startswith("public_reference_")
        assert not any(term in text for term in forbidden)

    core_path = (
        root / "configs/benchmark/tworoom_action_delay_h7_core_icl_v2.yaml"
    )
    core_text = core_path.read_text(encoding="utf-8")
    assert "diagnostics:" not in core_text
    assert yaml.safe_load(core_text)["status"] == "public_evaluation_contract"


def test_tworoom_upstream_symbol_resolves_inside_public_bundle(
    tmp_path: Path,
) -> None:
    upstream = tmp_path / "upstream/lewm-tworooms/tworoom.h5"
    upstream.parent.mkdir(parents=True)
    upstream.write_bytes(b"public-two-room-source")

    resolved = resolve_tworoom_original_h5(
        "upstream/lewm-tworooms/tworoom.h5",
        repo_root=tmp_path,
    )

    assert resolved == upstream.resolve()
