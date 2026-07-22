from argparse import ArgumentTypeError, Namespace

import pytest

from scripts.train_tworoom_step1 import (
    PROFILE_DEFAULTS,
    _apply_profile,
    _build_training_plan,
    _full_state_checkpoint_metadata,
    _lejepa_forward_with_manual_accumulation,
    _normalize_complete_epoch_resume_loop_state,
    _parse_batch_limit,
    _validate_resume_policy,
    parse_args,
)


def _full_state_payload(*, global_step: int = 3, world_size: int = 1):
    import random

    import numpy as np
    import torch

    rng_states = []
    for rank in range(world_size):
        generator = torch.Generator().manual_seed(100 + rank)
        rng_states.append(
            {
                "rank": rank,
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch_cpu": torch.get_rng_state(),
                "torch_cuda": torch.tensor([rank], dtype=torch.uint8),
                "train_loader_generator": generator.get_state(),
            }
        )
    return {
        "state_dict": {"weight": torch.ones(1)},
        "optimizer_states": [{"state": {}, "param_groups": []}],
        "lr_schedulers": [{"last_epoch": global_step}],
        "global_step": global_step,
        "epoch": 0,
        "loops": {"fit_loop": {}},
        "contextworld_rng_states_v1": rng_states,
    }


def _profile_args(profile: str, **overrides) -> Namespace:
    values = {key: None for key in PROFILE_DEFAULTS[profile]}
    values.update(
        {
            "profile": profile,
            "seed": 3072,
            "data_split_seed": 3072,
        }
    )
    values.update(overrides)
    return Namespace(**values)


def test_formal_history3_mixspeed_budget_matches_baseline() -> None:
    args = _apply_profile(_profile_args("formal"))
    metadata = {
        "epoch_group_counts": {
            "original": 657_408,
            "speed": 657_408,
        },
        "epoch_group_coverage": {
            "original": {
                "available_virtual_slots": 657_729,
                "unique_virtual_slots": 657_408,
            },
            "speed": {
                "available_virtual_slots": 15_936,
                "unique_virtual_slots": 15_936,
            },
        },
        "groups": {
            "original": {"train_clips": 657_729},
            "speed": {"train_clips_raw": 11_895},
        },
    }

    plan = _build_training_plan(args, metadata)

    assert plan["global_batch_size"] == 1_024
    assert plan["optimizer_steps_per_epoch"] == 1_284
    assert plan["optimizer_steps_total"] == 6_420
    assert plan["warmup_steps"] == 64
    assert plan["total_global_sample_draws"] == 6_574_080
    assert plan["group_exposure"]["original"]["total_draws"] == 3_287_040
    assert plan["group_exposure"]["original"]["raw_clips_never_drawn"] == 321
    assert plan["group_exposure"]["speed"]["total_draws"] == 3_287_040
    assert plan["group_exposure"]["speed"]["raw_clips_never_drawn"] == 0


def test_additive_profile_preserves_full_budget_for_each_group() -> None:
    args = _apply_profile(_profile_args("additive"))
    metadata = {
        "epoch_group_counts": {
            "original": 1_314_816,
            "speed": 1_314_816,
        },
        "epoch_group_coverage": {
            "original": {
                "available_virtual_slots": 657_000,
                "unique_virtual_slots": 657_000,
            },
            "speed": {
                "available_virtual_slots": 1_200_000,
                "unique_virtual_slots": 1_200_000,
            },
        },
        "groups": {
            "original": {"train_clips": 657_000},
            "speed": {"train_clips_raw": 1_000_000},
        },
    }

    plan = _build_training_plan(args, metadata)

    assert plan["optimizer_steps_total"] == 12_840
    assert plan["warmup_steps"] == 128
    assert plan["total_global_sample_draws"] == 13_148_160
    assert plan["group_exposure"]["original"]["total_draws"] == 6_574_080
    assert plan["group_exposure"]["speed"]["total_draws"] == 6_574_080


def test_native_resume_policy_accepts_fresh_auto_run(tmp_path) -> None:
    run_dir = tmp_path / "run"
    checkpoint = run_dir / "run_weights.ckpt"

    assert (
        _validate_resume_policy(
            run_dir=run_dir, checkpoint_path=checkpoint, policy="auto"
        )
        is None
    )


def test_native_resume_policy_rejects_model_only_run(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "weights_epoch_1.pt").write_bytes(b"model-only")

    with pytest.raises(FileExistsError, match="optimizer/scheduler"):
        _validate_resume_policy(
            run_dir=run_dir,
            checkpoint_path=run_dir / "run_weights.ckpt",
            policy="auto",
        )


def test_native_resume_policy_requires_native_checkpoint(tmp_path) -> None:
    run_dir = tmp_path / "run"
    checkpoint = run_dir / "run_weights.ckpt"

    with pytest.raises(FileNotFoundError, match="required but missing"):
        _validate_resume_policy(
            run_dir=run_dir,
            checkpoint_path=checkpoint,
            policy="required",
        )

    run_dir.mkdir()
    checkpoint.write_bytes(b"trainer-state")
    assert (
        _validate_resume_policy(
            run_dir=run_dir, checkpoint_path=checkpoint, policy="required"
        )
        == checkpoint
    )


def test_full_state_checkpoint_metadata_accepts_complete_trainer_state(
    tmp_path,
) -> None:
    import torch

    checkpoint = tmp_path / "last.ckpt"
    torch.save(_full_state_payload(global_step=3, world_size=4), checkpoint)

    metadata = _full_state_checkpoint_metadata(
        checkpoint,
        expected_optimizer_steps=10,
        require_incomplete=True,
        expected_world_size=4,
    )

    assert metadata["global_step"] == 3
    assert metadata["optimizer_states"] == 1
    assert metadata["lr_schedulers"] == 1
    assert metadata["rng_ranks"] == [0, 1, 2, 3]
    assert len(metadata["sha256"]) == 64


def test_full_state_checkpoint_rejects_model_only_weights(tmp_path) -> None:
    import torch

    checkpoint = tmp_path / "last.ckpt"
    torch.save({"state_dict": {"weight": torch.ones(1)}}, checkpoint)

    with pytest.raises(RuntimeError, match="cannot safely resume"):
        _full_state_checkpoint_metadata(
            checkpoint,
            expected_optimizer_steps=10,
            require_incomplete=True,
            expected_world_size=4,
        )


def test_full_state_checkpoint_rejects_topology_or_completed_run(
    tmp_path,
) -> None:
    import torch

    checkpoint = tmp_path / "last.ckpt"
    torch.save(_full_state_payload(global_step=10, world_size=1), checkpoint)

    with pytest.raises(RuntimeError, match="world size differs"):
        _full_state_checkpoint_metadata(
            checkpoint,
            expected_optimizer_steps=10,
            require_incomplete=True,
            expected_world_size=4,
        )
    with pytest.raises(RuntimeError, match="already complete"):
        _full_state_checkpoint_metadata(
            checkpoint,
            expected_optimizer_steps=10,
            require_incomplete=True,
            expected_world_size=1,
        )


def test_full_state_checkpoint_requires_complete_epoch_boundary(
    tmp_path,
) -> None:
    import torch

    checkpoint = tmp_path / "last.ckpt"
    torch.save(_full_state_payload(global_step=3, world_size=1), checkpoint)

    with pytest.raises(RuntimeError, match="complete epoch boundary"):
        _full_state_checkpoint_metadata(
            checkpoint,
            expected_optimizer_steps=10,
            require_incomplete=True,
            expected_world_size=1,
            optimizer_steps_per_epoch=2,
        )

    torch.save(_full_state_payload(global_step=4, world_size=1), checkpoint)
    metadata = _full_state_checkpoint_metadata(
        checkpoint,
        expected_optimizer_steps=10,
        require_incomplete=True,
        expected_world_size=1,
        optimizer_steps_per_epoch=2,
    )
    assert metadata["complete_epoch_boundary"] is True
    assert metadata["optimizer_steps_per_epoch"] == 2


def test_complete_epoch_loop_state_restarts_at_next_epoch_batch_zero() -> None:
    checkpoint = {
        "global_step": 4,
        "epoch": 2,
        "loops": {
            "fit_loop": {
                "epoch_loop.batch_progress": {
                    "total": {
                        "ready": 8,
                        "started": 8,
                        "processed": 8,
                        "completed": 8,
                    },
                    "current": {
                        "ready": 4,
                        "started": 4,
                        "processed": 4,
                        "completed": 4,
                    },
                    "is_last_batch": True,
                }
            }
        },
    }

    audit = _normalize_complete_epoch_resume_loop_state(
        checkpoint,
        optimizer_steps_per_epoch=2,
        accumulation_steps=2,
    )

    progress = checkpoint["loops"]["fit_loop"][
        "epoch_loop.batch_progress"
    ]
    assert progress["current"] == {
        "ready": 0,
        "started": 0,
        "processed": 0,
        "completed": 0,
    }
    assert progress["is_last_batch"] is False
    assert progress["total"]["completed"] == 8
    assert audit["next_epoch_starts_at_batch_zero"] is True


def test_complete_epoch_loop_state_rejects_partial_epoch() -> None:
    checkpoint = {
        "global_step": 2,
        "epoch": 1,
        "loops": {
            "fit_loop": {
                "epoch_loop.batch_progress": {
                    "total": {
                        "ready": 3,
                        "started": 3,
                        "processed": 3,
                        "completed": 3,
                    },
                    "current": {
                        "ready": 3,
                        "started": 3,
                        "processed": 3,
                        "completed": 3,
                    },
                    "is_last_batch": False,
                }
            }
        },
    }

    with pytest.raises(RuntimeError, match="not at the end"):
        _normalize_complete_epoch_resume_loop_state(
            checkpoint,
            optimizer_steps_per_epoch=2,
            accumulation_steps=2,
        )


def test_formal_data_quality_gate_rejects_excessive_reuse() -> None:
    args = _apply_profile(_profile_args("formal"))
    metadata = {
        "epoch_group_counts": {"original": 657_408, "speed": 657_408},
        "epoch_group_coverage": {
            "original": {
                "available_virtual_slots": 657_729,
                "unique_virtual_slots": 657_408,
            },
            "speed": {
                "available_virtual_slots": 20_000,
                "unique_virtual_slots": 20_000,
            },
        },
        "groups": {
            "original": {"train_clips": 657_729},
            "speed": {
                "train_clips_raw": 20_000,
                "quality_requirements": {
                    "maximum_formal_mean_draws_per_raw_clip": 20.0
                },
                "static_quality_gates": {"all": True},
            },
        },
    }

    with pytest.raises(ValueError, match="data-quality gates"):
        _build_training_plan(args, metadata)


def test_training_cli_model_id_is_config_driven(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "train_tworoom_step1.py",
            "--model-id",
            "M_synth5matched",
            "--run-name",
            "config-driven-model",
            "--report",
            "/tmp/config-driven-model.json",
        ],
    )
    assert parse_args().model_id == "M_synth5matched"


def test_formal_profile_rejects_budget_changing_override() -> None:
    with pytest.raises(ValueError, match="validated execution topologies"):
        _apply_profile(_profile_args("formal", batch_size=64))


def test_formal_four_gpu_topology_preserves_per_rank_microbatch() -> None:
    args = _apply_profile(_profile_args("formal"))

    assert args.devices == 4
    assert args.batch_size == 128
    assert args.accumulate_grad_batches == 2


def test_formal_eight_gpu_topology_remains_supported() -> None:
    args = _apply_profile(
        _profile_args(
            "formal",
            devices=8,
            batch_size=128,
            accumulate_grad_batches=1,
        )
    )

    assert args.devices == 8
    assert args.accumulate_grad_batches == 1


def test_batch_limit_distinguishes_integer_batch_from_fraction() -> None:
    assert _parse_batch_limit("1") == 1
    assert isinstance(_parse_batch_limit("1"), int)
    assert _parse_batch_limit("1.0") == 1.0
    assert isinstance(_parse_batch_limit("1.0"), float)
    assert _parse_batch_limit("0.25") == 0.25
    with pytest.raises(ArgumentTypeError):
        _parse_batch_limit("1.5")


def test_manual_accumulation_scales_only_fit_loss() -> None:
    import torch

    calls = []

    def base_forward(module, batch, stage, cfg):
        calls.append((module, batch, stage, cfg))
        return {"loss": torch.tensor(6.0), "pred_loss": torch.tensor(4.0)}

    module = object()
    fit = _lejepa_forward_with_manual_accumulation(
        module,
        {"x": 1},
        "fit",
        base_forward=base_forward,
        cfg={"history": 3},
        accumulation_steps=2,
    )
    validation = _lejepa_forward_with_manual_accumulation(
        module,
        {"x": 2},
        "validate",
        base_forward=base_forward,
        cfg={"history": 3},
        accumulation_steps=2,
    )

    assert fit["loss"].item() == 3.0
    assert fit["pred_loss"].item() == 4.0
    assert validation["loss"].item() == 6.0
    assert len(calls) == 2


def test_two_scaled_microbatch_backwards_match_mean_objective_update() -> None:
    import torch

    torch.manual_seed(7)
    reference = torch.nn.Linear(3, 2)
    accumulated = torch.nn.Linear(3, 2)
    accumulated.load_state_dict(reference.state_dict())
    inputs = [torch.randn(5, 3), torch.randn(5, 3)]
    targets = [torch.randn(5, 2), torch.randn(5, 2)]
    reference_opt = torch.optim.SGD(reference.parameters(), lr=0.1)
    accumulated_opt = torch.optim.SGD(accumulated.parameters(), lr=0.1)

    reference_loss = sum(
        torch.nn.functional.mse_loss(reference(x), y)
        for x, y in zip(inputs, targets)
    ) / 2
    reference_loss.backward()
    reference_opt.step()

    for x, y in zip(inputs, targets):
        loss = torch.nn.functional.mse_loss(accumulated(x), y) / 2
        loss.backward()
    accumulated_opt.step()

    for expected, observed in zip(
        reference.parameters(), accumulated.parameters()
    ):
        assert torch.allclose(expected, observed, atol=1e-7, rtol=1e-6)
