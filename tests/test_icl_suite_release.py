from __future__ import annotations

from pathlib import Path

import pytest

from contextworld.benchmarks import suite_data
from contextworld.benchmarks.suite_data import (
    COMPONENT_IDS,
    REFERENCE_RESULT_STATUSES,
    SUITE_RELEASE_ID,
    export_icl_suite_artifacts,
    load_icl_suite_release,
)


def test_default_suite_registers_all_released_components() -> None:
    suite = load_icl_suite_release()
    assert suite["release_id"] == SUITE_RELEASE_ID
    assert tuple(suite["components"]) == COMPONENT_IDS
    assert suite["bundle"]["top_level_entries"] == [
        "README.md",
        "benchmark",
    ]
    assert suite["scope"]["public_test_included"] is True
    assert suite["scope"]["sealed_test_included"] is False
    assert {
        component_id: component["reference_result_status"]
        for component_id, component in suite["components"].items()
    } == REFERENCE_RESULT_STATUSES
    assert all(
        component["benchmark_component_status"] == "ready"
        for component in suite["components"].values()
    )


def test_suite_uses_a_decoder_free_latent_model_contract() -> None:
    suite = load_icl_suite_release()
    interface = suite["model_interface"]
    assert interface["primary_model_type"] == "latent_world_model"
    assert interface["decoder_required"] is False
    assert (
        interface["raw_latent_loss_cross_model_comparison_allowed"]
        is False
    )


def test_frozen_public_scoreboard_is_exactly_reproducible() -> None:
    suite = load_icl_suite_release()
    audit = suite_data._audit_public_results(
        suite,
        repo_root=Path.cwd(),
    )
    assert audit["passed"] is True
    assert audit["reproduction"]["scoreboard_exactly_reproduced"] is True
    assert audit["reproduction"]["observed_reference_rows"] == 10


def test_public_document_uses_one_template_for_every_component() -> None:
    suite = load_icl_suite_release()
    document = Path(suite["repository"]["public_document"]["path"])
    audit = suite_data._audit_public_document_template(document, suite)
    assert audit["passed"] is True


def test_suite_requires_a_passed_causal_contract_per_component() -> None:
    passed = {
        "passed": True,
        "causal_data_contract": {"passed": True},
    }
    gate = suite_data._enforce_component_causal_gate(passed)
    assert gate == {"present": True, "passed": True}
    assert passed["passed"] is True

    missing = {"passed": True}
    gate = suite_data._enforce_component_causal_gate(missing)
    assert gate == {"present": False, "passed": False}
    assert missing["passed"] is False

    failed = {
        "passed": True,
        "causal_data_contract": {"passed": False},
    }
    gate = suite_data._enforce_component_causal_gate(failed)
    assert gate == {"present": True, "passed": False}
    assert failed["passed"] is False


def test_export_gate_rejects_a_stale_component_config(tmp_path: Path) -> None:
    component = tmp_path / "component.yaml"
    component.write_text("current", encoding="utf-8")
    suite = {
        "_config_path": str(tmp_path / "suite.yaml"),
        "repository": {},
        "components": {
            "speed": {
                "release_config": str(component),
                "release_config_sha256": "0" * 64,
            }
        },
    }
    with pytest.raises(RuntimeError, match="component speed"):
        suite_data._assert_frozen_export_inputs(suite, repo_root=tmp_path)


def test_export_gate_rejects_machine_paths_in_metadata(
    tmp_path: Path,
    monkeypatch,
) -> None:
    artifact = tmp_path / "release"
    artifact.mkdir()
    (artifact / "manifest.json").write_text(
        '{"source": "/opt/private-machine/data.h5"}',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        suite_data,
        "resolve_contextworld_path",
        lambda *args, **kwargs: artifact,
    )
    with pytest.raises(RuntimeError, match="machine-specific paths"):
        suite_data._assert_portable_export_entries(
            [("artifacts/release", "directory")],
            repo_root=tmp_path,
        )


def test_export_entries_skip_a_file_already_covered_by_a_directory(
    tmp_path: Path,
    monkeypatch,
) -> None:
    artifact_root = tmp_path / "artifacts"
    release = artifact_root / "release"
    release.mkdir(parents=True)
    (release / "receipt.json").write_text("{}\n", encoding="utf-8")

    def resolve(logical_path: str, **_kwargs) -> Path:
        return artifact_root.joinpath(*Path(logical_path).parts[1:])

    monkeypatch.setattr(suite_data, "resolve_contextworld_path", resolve)
    entries = suite_data._deduplicate_export_entries(
        [
            ("artifacts/release", "directory"),
            ("artifacts/release/receipt.json", "file"),
            ("artifacts/release", "directory"),
        ],
        repo_root=tmp_path,
    )
    assert entries == [("artifacts/release", "directory")]


def test_export_entries_reject_conflicting_directory_coverage(
    tmp_path: Path,
    monkeypatch,
) -> None:
    artifact_root = tmp_path / "artifacts"
    release = artifact_root / "release"
    release.mkdir(parents=True)
    (release / "receipt.json").write_text("inside\n", encoding="utf-8")
    conflicting = tmp_path / "other-receipt.json"
    conflicting.write_text("outside\n", encoding="utf-8")

    def resolve(logical_path: str, **_kwargs) -> Path:
        if logical_path.endswith("receipt.json"):
            return conflicting
        return release

    monkeypatch.setattr(suite_data, "resolve_contextworld_path", resolve)
    with pytest.raises(RuntimeError, match="different sources"):
        suite_data._deduplicate_export_entries(
            [
                ("artifacts/release", "directory"),
                ("artifacts/release/receipt.json", "file"),
            ],
            repo_root=tmp_path,
        )


def test_suite_export_includes_causal_evidence_outside_data_trees() -> None:
    speed_entries = suite_data._speed_export_entries(
        {
            "training": {"synthetic": {}},
            "evaluation": {
                "normalizer": "artifacts/normalizer.json",
                "causal_data_audit": "artifacts/speed-causal.json",
            },
            "reference_results": {},
        }
    )
    assert ("artifacts/speed-causal.json", "file") in speed_entries

    action_entries = suite_data._action_strength_export_entries(
        {
            "training": {
                "artifact_tree": {"root": "artifacts/action/train"},
                "contrast_scales": {"path": "artifacts/scales.json"},
                "artifacts": {
                    "manifest": {
                        "path": "artifacts/action/train/manifest.json"
                    },
                    "strict_release_audit": {
                        "path": "artifacts/action-causal.json"
                    },
                },
            },
            "evaluation": {
                "artifact_tree": {"root": "artifacts/action/test"},
                "planning_oracle": {"path": "artifacts/oracle.json"},
                "artifacts": {},
            },
            "reference_method": {
                "artifact_tree": {
                    "root": "artifacts/action/reference"
                }
            },
            "reference_results": {},
        }
    )
    assert ("artifacts/action-causal.json", "file") in action_entries
    assert (
        "artifacts/action/train/manifest.json",
        "file",
    ) not in action_entries

    release_with_nested_reference_receipt = {
        "training": {
            "artifact_tree": {"root": "artifacts/action/train"},
            "contrast_scales": {"path": "artifacts/scales.json"},
            "artifacts": {
                "compatibility": {
                    "path": (
                        "artifacts/action/reference/compatibility.json"
                    )
                }
            },
        },
        "evaluation": {
            "artifact_tree": {"root": "artifacts/action/test"},
            "planning_oracle": {"path": "artifacts/oracle.json"},
            "artifacts": {},
        },
        "reference_method": {
            "artifact_tree": {"root": "artifacts/action/reference"}
        },
        "reference_results": {},
    }
    assert (
        "artifacts/action/reference/compatibility.json",
        "file",
    ) not in suite_data._action_strength_export_entries(
        release_with_nested_reference_receipt
    )

    contact_entries = suite_data._contact_friction_export_entries(
        {
            "data": {
                "artifact_tree": {
                    "root": "artifacts/contact/data"
                },
                "artifacts": {
                    "manifest": {
                        "path": "artifacts/contact/data/manifest.json",
                        "sha256": "a" * 64,
                    },
                    "causal_audit": {
                        "path": "artifacts/contact/causal-audit.json",
                        "sha256": "b" * 64,
                    },
                },
            },
            "reference_results": {
                "current_decision": {
                    "path": "artifacts/contact/development-decision.json",
                    "sha256": "c" * 64,
                }
            },
        }
    )
    assert (
        "artifacts/contact/development-decision.json",
        "file",
    ) in contact_entries
    assert (
        "artifacts/contact/causal-audit.json",
        "file",
    ) in contact_entries
    assert (
        "artifacts/contact/data/manifest.json",
        "file",
    ) not in contact_entries

    motion_entries = suite_data._motion_damping_export_entries(
        {
            "data": {
                "artifact_tree": {
                    "root": "artifacts/motion/data"
                },
                "artifacts": {},
            },
            "reference_results": {
                "current_decision": {
                    "path": "artifacts/motion/development-decision.json",
                    "sha256": "d" * 64,
                }
            },
        }
    )
    assert motion_entries == [
        ("artifacts/motion/data", "directory"),
        ("artifacts/motion/development-decision.json", "file"),
    ]


def test_integrated_export_has_only_readme_and_benchmark(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo = tmp_path / "repo"
    artifacts = tmp_path / "artifacts"
    destination = tmp_path / "bundle"
    (repo / "configs/benchmark").mkdir(parents=True)
    (repo / "docs").mkdir()
    (repo / "docs/ContextWorld_ICL_Benchmark.md").write_text(
        "# Unified benchmark\n",
        encoding="utf-8",
    )
    suite_config = repo / "configs/benchmark/suite.yaml"
    speed_config = repo / "configs/benchmark/speed.yaml"
    door_config = repo / "configs/benchmark/door.yaml"
    action_delay_config = repo / "configs/benchmark/action_delay.yaml"
    action_strength_config = (
        repo / "configs/benchmark/action_strength.yaml"
    )
    contact_friction_config = (
        repo / "configs/benchmark/contact_friction.yaml"
    )
    motion_damping_config = (
        repo / "configs/benchmark/motion_damping.yaml"
    )
    robot_arm_mass_config = (
        repo / "configs/benchmark/robot_arm_mass.yaml"
    )
    portal_exit_config = repo / "configs/benchmark/portal_exit.yaml"
    suite_config.write_text("suite\n", encoding="utf-8")
    speed_config.write_text("speed\n", encoding="utf-8")
    door_config.write_text("door\n", encoding="utf-8")
    action_delay_config.write_text("action-delay\n", encoding="utf-8")
    action_strength_config.write_text(
        "action-strength\n",
        encoding="utf-8",
    )
    contact_friction_config.write_text(
        "contact-friction\n",
        encoding="utf-8",
    )
    motion_damping_config.write_text(
        "motion-damping\n",
        encoding="utf-8",
    )
    robot_arm_mass_config.write_text(
        "robot-arm-mass\n",
        encoding="utf-8",
    )
    portal_exit_config.write_text("portal-exit\n", encoding="utf-8")

    directory_paths = (
        "synthesis/speed",
        "evaluation/history3/speed_multistep_extrap_v5/catalogs",
        "evaluation/history3/speed_multistep_extrap_v5/payloads",
        "evaluation/history3/speed_isolated_v2/catalogs",
        "evaluation/history3/speed_isolated_v2/payloads",
        "synthesis/door",
        "evaluation/door",
        "evaluation/door-reference",
        "synthesis/action-delay",
        "evaluation/action-delay",
        "synthesis/action-strength",
        "evaluation/action-strength",
        "synthesis/contact-friction",
        "synthesis/motion-damping",
        "synthesis/robot-arm-mass",
        "synthesis/portal-exit",
    )
    for relative in directory_paths:
        path = artifacts / relative
        path.mkdir(parents=True)
        (path / "payload.bin").write_bytes(relative.encode("utf-8"))
    file_paths = (
        "synthesis/speed.json",
        "synthesis/speed.jsonl",
        "synthesis/speed-report.json",
        "splits/normalizer.json",
        "training/init.pt",
        "training/config.json",
        "evaluation/action-delay-core.json",
        "evaluation/action-delay-cem.json",
        "evaluation/speed-one-step.json",
        "evaluation/speed-multistep.json",
        "evaluation/speed-planning.json",
        "evaluation/action-strength-scales.json",
        "evaluation/action-strength-oracle.json",
        "evaluation/action-strength-result.json",
        "evaluation/contact-friction-result.json",
        "evaluation/robot-arm-mass-result.json",
        "evaluation/portal-exit-result.json",
    )
    for relative in file_paths:
        path = artifacts / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode("utf-8"))
    original_h5 = tmp_path / "tworoom.h5"
    original_h5.write_bytes(b"original")
    tworoom_lance = tmp_path / "tworoom.lance"
    tworoom_lance.mkdir()
    (tworoom_lance / "data.bin").write_bytes(b"tworoom-lance")
    pusht_h5 = tmp_path / "pusht.h5"
    pusht_h5.write_bytes(b"pusht-h5")
    pusht_lance = tmp_path / "pusht.lance"
    pusht_lance.mkdir()
    (pusht_lance / "data.bin").write_bytes(b"pusht-lance")
    pusht_checkpoint = tmp_path / "pusht.pt"
    pusht_checkpoint.write_bytes(b"pusht-checkpoint")
    reacher_h5 = tmp_path / "reacher.h5"
    reacher_h5.write_bytes(b"reacher-h5")
    reacher_lance = tmp_path / "reacher.lance"
    reacher_lance.mkdir()
    (reacher_lance / "data.bin").write_bytes(b"reacher-lance")
    reacher_lewm = tmp_path / "reacher-lewm.ckpt"
    reacher_lewm.write_bytes(b"reacher-lewm")
    reacher_pldm = tmp_path / "reacher-pldm.ckpt"
    reacher_pldm.write_bytes(b"reacher-pldm")
    reacher_lewm_config = tmp_path / "reacher-lewm.yaml"
    reacher_lewm_config.write_bytes(b"lewm-config")
    reacher_pldm_config = tmp_path / "reacher-pldm.yaml"
    reacher_pldm_config.write_bytes(b"pldm-config")

    suite = {
        "release_id": SUITE_RELEASE_ID,
        "_config_path": str(suite_config),
        "components": {
            "speed": {"release_config": str(speed_config)},
            "door": {"release_config": str(door_config)},
            "action_delay": {
                "release_config": str(action_delay_config)
            },
            "action_strength": {
                "release_config": str(action_strength_config)
            },
            "contact_friction": {
                "release_config": str(contact_friction_config)
            },
            "motion_damping": {
                "release_config": str(motion_damping_config)
            },
            "robot_arm_mass": {
                "release_config": str(robot_arm_mass_config)
            },
            "portal_exit": {
                "release_config": str(portal_exit_config)
            },
        },
        "repository": {
            "public_document": {
                "path": "docs/ContextWorld_ICL_Benchmark.md"
            }
        },
        "distribution": {},
    }
    speed = {
        "training": {
            "synthetic": {
                "single": {
                    "data_root": "artifacts/synthesis/speed",
                    "catalog": "artifacts/synthesis/speed.json",
                    "manifest": "artifacts/synthesis/speed.jsonl",
                    "report": "artifacts/synthesis/speed-report.json",
                }
            },
            "original": {"source": "upstream", "license": "MIT"},
        },
        "evaluation": {"normalizer": "artifacts/splits/normalizer.json"},
        "planning": {"enabled": True},
        "reference_results": {
            "one_step": {
                "path": "artifacts/evaluation/speed-one-step.json",
                "sha256": "a" * 64,
            },
            "multistep": {
                "path": "artifacts/evaluation/speed-multistep.json",
                "sha256": "b" * 64,
            },
            "planning": {
                "path": "artifacts/evaluation/speed-planning.json",
                "sha256": "c" * 64,
            },
        },
    }
    door = {
        "training": {
            "artifact_tree": {"root": "artifacts/synthesis/door"},
            "initialization": {
                "checkpoint": "artifacts/training/init.pt",
                "checkpoint_config": "artifacts/training/config.json",
            },
        },
        "evaluation": {
            "artifact_tree": {"root": "artifacts/evaluation/door"},
            "normalizer": "artifacts/splits/normalizer.json",
        },
        "reference_results": {
            "reference": {
                "root": "artifacts/evaluation/door-reference"
            }
        },
    }
    action_delay = {
        "training": {
            "artifact_tree": {
                "root": "artifacts/synthesis/action-delay"
            }
        },
        "evaluation": {
            "artifact_tree": {
                "root": "artifacts/evaluation/action-delay"
            },
            "normalizer": "artifacts/splits/normalizer.json",
        },
        "reference_results": {
            "core": {
                "path": "artifacts/evaluation/action-delay-core.json",
                "sha256": "0" * 64,
            },
            "cem": {
                "path": "artifacts/evaluation/action-delay-cem.json",
                "sha256": "1" * 64,
            },
        },
    }
    action_strength = {
        "training": {
            "artifact_tree": {
                "root": "artifacts/synthesis/action-strength"
            },
            "contrast_scales": {
                "path": "artifacts/evaluation/action-strength-scales.json"
            },
            "upstream": {
                "original_h5": {
                    "bytes": pusht_h5.stat().st_size,
                    "role": "source",
                },
                "original_lance": {
                    "bytes": sum(
                        path.stat().st_size
                        for path in pusht_lance.rglob("*")
                        if path.is_file()
                    ),
                    "role": "source",
                },
            },
            "initialization": {
                "bytes": pusht_checkpoint.stat().st_size,
                "role": "initialization",
            },
        },
        "evaluation": {
            "artifact_tree": {
                "root": "artifacts/evaluation/action-strength"
            },
            "planning_oracle": {
                "path": "artifacts/evaluation/action-strength-oracle.json"
            },
        },
        "reference_results": {
            "result": {
                "path": "artifacts/evaluation/action-strength-result.json",
                "sha256": "2" * 64,
            }
        },
    }
    contact_friction = {
        "data": {
            "artifact_tree": {
                "root": "artifacts/synthesis/contact-friction"
            }
        },
        "reference_results": {
            "result": {
                "path": (
                    "artifacts/evaluation/"
                    "contact-friction-result.json"
                ),
                "sha256": "4" * 64,
            }
        },
    }
    motion_damping = {
        "data": {
            "artifact_tree": {
                "root": "artifacts/synthesis/motion-damping"
            }
        },
        "reference_results": {},
    }
    robot_arm_mass = {
        "data": {
            "artifact_tree": {
                "root": "artifacts/synthesis/robot-arm-mass"
            },
            "artifacts": {},
        },
        "training": {
            "upstream": {
                "original_h5": {
                    "bytes": reacher_h5.stat().st_size,
                    "role": "source",
                },
                "original_lance": {
                    "bytes": sum(
                        path.stat().st_size
                        for path in reacher_lance.rglob("*")
                        if path.is_file()
                    ),
                    "role": "source",
                },
            },
            "reference_matrix": {
                "initial_checkpoints": {
                    "lewm": {
                        "bytes": reacher_lewm.stat().st_size,
                        "role": "initialization",
                        "config_relative_to_checkpoint": "config.yaml",
                        "config_bundled_artifact_path": (
                            "upstream/stable-worldmodel/reacher_lewm/"
                            "config.yaml"
                        ),
                        "config_bytes": reacher_lewm_config.stat().st_size,
                        "config_sha256": "5" * 64,
                    },
                    "pldm": {
                        "bytes": reacher_pldm.stat().st_size,
                        "role": "initialization",
                        "config_relative_to_checkpoint": "config.yaml",
                        "config_bundled_artifact_path": (
                            "upstream/stable-worldmodel/"
                            "reacher_pldm_baseline/config.yaml"
                        ),
                        "config_bytes": reacher_pldm_config.stat().st_size,
                        "config_sha256": "6" * 64,
                    },
                }
            },
        },
        "reference_results": {
            "result": {
                "path": "artifacts/evaluation/robot-arm-mass-result.json",
                "sha256": "7" * 64,
            }
        },
    }
    portal_exit = {
        "data": {
            "artifact_tree": {
                "root": "artifacts/synthesis/portal-exit"
            },
            "artifacts": {},
        },
        "training": {
            "initialization": {
                "checkpoint": "artifacts/training/init.pt",
                "frozen_normalizer": "artifacts/splits/normalizer.json",
            },
            "upstream": {
                "original_lance": {
                    "bytes": sum(
                        path.stat().st_size
                        for path in tworoom_lance.rglob("*")
                        if path.is_file()
                    ),
                    "role": "source",
                }
            },
        },
        "reference_results": {
            "result": {
                "path": "artifacts/evaluation/portal-exit-result.json",
                "sha256": "8" * 64,
            }
        },
    }

    original_resolver = suite_data.resolve_contextworld_path

    def fake_resolve(value, *, repo_root=None):
        path = Path(value)
        if path.parts and path.parts[0] == "artifacts":
            return artifacts.joinpath(*path.parts[1:])
        return original_resolver(value, repo_root=repo_root)

    monkeypatch.setattr(suite_data, "load_icl_suite_release", lambda *a, **k: suite)
    monkeypatch.setattr(suite_data, "load_speed_icl_release", lambda *a, **k: speed)
    monkeypatch.setattr(suite_data, "load_door_icl_release", lambda *a, **k: door)
    monkeypatch.setattr(
        suite_data,
        "load_action_delay_icl_release",
        lambda *a, **k: action_delay,
    )
    monkeypatch.setattr(
        suite_data,
        "load_action_strength_icl_release",
        lambda *a, **k: action_strength,
    )
    monkeypatch.setattr(
        suite_data,
        "load_contact_friction_icl_release",
        lambda *a, **k: contact_friction,
    )
    monkeypatch.setattr(
        suite_data,
        "load_motion_damping_icl_release",
        lambda *a, **k: motion_damping,
    )
    monkeypatch.setattr(
        suite_data,
        "load_reacher_arm_mass_icl_release",
        lambda *a, **k: robot_arm_mass,
    )
    monkeypatch.setattr(
        suite_data,
        "load_portal_exit_icl_release",
        lambda *a, **k: portal_exit,
    )
    monkeypatch.setattr(suite_data, "resolve_contextworld_path", fake_resolve)
    monkeypatch.setattr(
        suite_data,
        "resolve_original_h5",
        lambda *a, **k: original_h5,
    )
    monkeypatch.setattr(
        suite_data,
        "resolve_portal_original_lance",
        lambda *a, **k: tworoom_lance,
    )
    monkeypatch.setattr(
        suite_data,
        "resolve_action_strength_original_h5",
        lambda *a, **k: pusht_h5,
    )
    monkeypatch.setattr(
        suite_data,
        "resolve_action_strength_original_lance",
        lambda *a, **k: pusht_lance,
    )
    monkeypatch.setattr(
        suite_data,
        "resolve_action_strength_initial_checkpoint",
        lambda *a, **k: pusht_checkpoint,
    )
    monkeypatch.setattr(
        suite_data,
        "resolve_reacher_original_h5",
        lambda *a, **k: reacher_h5,
    )
    monkeypatch.setattr(
        suite_data,
        "resolve_reacher_original_lance",
        lambda *a, **k: reacher_lance,
    )
    monkeypatch.setattr(
        suite_data,
        "resolve_reacher_initial_checkpoint",
        lambda release, family, **kwargs: (
            reacher_lewm if family == "lewm" else reacher_pldm
        ),
    )
    monkeypatch.setattr(
        suite_data,
        "resolve_reacher_initial_checkpoint_config",
        lambda release, family, **kwargs: (
            reacher_lewm_config if family == "lewm" else reacher_pldm_config
        ),
    )

    result = export_icl_suite_artifacts(
        destination,
        repo_root=repo,
        mode="copy",
    )
    assert sorted(path.name for path in destination.iterdir()) == [
        "README.md",
        "benchmark",
    ]
    assert (destination / "benchmark/suite.yaml").is_file()
    assert (destination / "benchmark/releases/speed.yaml").is_file()
    assert (destination / "benchmark/releases/door.yaml").is_file()
    assert (
        destination / "benchmark/releases/action_delay.yaml"
    ).is_file()
    assert (
        destination / "benchmark/releases/action_strength.yaml"
    ).is_file()
    assert (
        destination / "benchmark/releases/contact_friction.yaml"
    ).is_file()
    assert (
        destination / "benchmark/releases/motion_damping.yaml"
    ).is_file()
    assert (
        destination / "benchmark/releases/robot_arm_mass.yaml"
    ).is_file()
    assert (
        destination / "benchmark/releases/portal_exit.yaml"
    ).is_file()
    assert (
        destination / "benchmark/evaluation/speed-one-step.json"
    ).is_file()
    assert (
        destination / "benchmark/evaluation/speed-multistep.json"
    ).is_file()
    assert (
        destination / "benchmark/evaluation/speed-planning.json"
    ).is_file()
    assert (
        destination / "benchmark/upstream/lewm-tworooms/tworoom.h5"
    ).is_file()
    assert (
        destination
        / "benchmark/upstream/stable-worldmodel/lewm_tworoom.lance"
    ).is_dir()
    assert result["components"] == [
        "speed",
        "door",
        "action_delay",
        "action_strength",
        "contact_friction",
        "motion_damping",
        "robot_arm_mass",
        "portal_exit",
    ]
    assert result["includes_upstream_original_h5"] is True
