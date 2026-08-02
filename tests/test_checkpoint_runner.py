from __future__ import annotations

import random
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from rwkv_lm import checkpoint_runner as runner_module
from rwkv_lm.checkpoint import (
    BackendIdentity,
    CheckpointContractError,
    select_checkpoint_loader,
)
from rwkv_lm.checkpoint_runner import EpochCheckpointRunnerAdapter
from rwkv_lm.dataset import MyDataset
from rwkv_lm.trainer import scheduled_learning_rate


def _backend(*, version: str = torch.__version__) -> BackendIdentity:
    return BackendIdentity(
        name="pytorch",
        version=version,
        strategy="single_process",
        world_size=1,
        state_dict_type="full",
    )


def _adapter(*, version: str = torch.__version__) -> EpochCheckpointRunnerAdapter:
    return EpochCheckpointRunnerAdapter(
        backend=_backend(version=version),
        training_config={
            "batch_size": 4,
            "epoch_steps": 3,
            "model": "tiny-linear",
            "seed": 20260801,
        },
        samples_per_epoch=12,
    )


def _pytorch_adapter() -> EpochCheckpointRunnerAdapter:
    return EpochCheckpointRunnerAdapter(
        backend=BackendIdentity(
            name="pytorch",
            version=torch.__version__,
            strategy="single_process",
            world_size=1,
            state_dict_type="full",
        ),
        training_config={
            "batch_size": 4,
            "epoch_steps": 3,
            "model": "tiny-linear-cosine",
            "scheduler": "CosineAnnealingLR",
            "seed": 20260801,
        },
        samples_per_epoch=12,
    )


def _new_runner() -> tuple[nn.Module, torch.optim.Optimizer]:
    model = nn.Sequential(nn.Linear(3, 5), nn.Tanh(), nn.Linear(5, 1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.05)
    return model, optimizer


def _seed_all() -> None:
    random.seed(20260801)
    np.random.seed(20260801)
    torch.manual_seed(20260801)


def _train_steps(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    start_step: int,
    step_count: int,
) -> None:
    for global_step in range(start_step, start_step + step_count):
        inputs = torch.randn(4, 3)
        python_offset = random.random()
        numpy_offset = float(np.random.random())
        targets = inputs.sum(dim=1, keepdim=True) + python_offset + numpy_offset
        optimizer.param_groups[0]["lr"] = 0.05 / (global_step + 1)
        optimizer.zero_grad(set_to_none=True)
        loss = (model(inputs) - targets).square().mean()
        loss.backward()
        optimizer.step()


def _train_scheduled_steps(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    *,
    step_count: int,
) -> None:
    for _ in range(step_count):
        inputs = torch.randn(4, 3)
        targets = inputs.sum(dim=1, keepdim=True)
        optimizer.zero_grad(set_to_none=True)
        loss = (model(inputs) - targets).square().mean()
        loss.backward()
        optimizer.step()
        scheduler.step()


def _assert_nested_equal(actual: object, expected: object) -> None:
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, Mapping):
        assert isinstance(actual, Mapping)
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_nested_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert isinstance(actual, type(expected))
        assert len(actual) == len(expected)
        for actual_value, expected_value in zip(actual, expected, strict=True):
            _assert_nested_equal(actual_value, expected_value)
    else:
        assert actual == expected


def test_interrupted_resume_matches_uninterrupted_training(tmp_path: Path) -> None:
    _seed_all()
    uninterrupted_model, uninterrupted_optimizer = _new_runner()
    _train_steps(
        uninterrupted_model,
        uninterrupted_optimizer,
        start_step=0,
        step_count=6,
    )

    _seed_all()
    interrupted_model, interrupted_optimizer = _new_runner()
    _train_steps(
        interrupted_model,
        interrupted_optimizer,
        start_step=0,
        step_count=3,
    )
    checkpoint = tmp_path / "epoch-00000001"
    manifest = _adapter().save(
        checkpoint,
        model=interrupted_model,
        optimizer=interrupted_optimizer,
        global_step=3,
        next_epoch=1,
    )

    random.random()
    np.random.random()
    torch.rand(9)
    resumed_model, resumed_optimizer = _new_runner()
    progress = _adapter().restore(
        checkpoint,
        model=resumed_model,
        optimizer=resumed_optimizer,
    )
    _train_steps(
        resumed_model,
        resumed_optimizer,
        start_step=progress.global_step,
        step_count=3,
    )

    assert progress == manifest.progress
    assert progress.epoch == 1
    assert progress.step_in_epoch == 0
    _assert_nested_equal(resumed_model.state_dict(), uninterrupted_model.state_dict())
    _assert_nested_equal(
        resumed_optimizer.state_dict(),
        uninterrupted_optimizer.state_dict(),
    )
    assert select_checkpoint_loader(
        checkpoint,
        expected_backend=_backend(),
    ).complete_resume


def test_resume_migrates_implicit_accumulation_default(tmp_path: Path) -> None:
    model, optimizer = _new_runner()
    legacy_adapter = _adapter()
    checkpoint = tmp_path / "epoch-00000001"
    legacy_adapter.save(
        checkpoint,
        model=model,
        optimizer=optimizer,
        global_step=3,
        next_epoch=1,
    )
    requested_config = dict(legacy_adapter.training_config)
    requested_config["accumulate_grad_batches"] = 1
    current_adapter = EpochCheckpointRunnerAdapter(
        backend=legacy_adapter.backend,
        training_config=requested_config,
        samples_per_epoch=legacy_adapter.samples_per_epoch,
    )

    resumed_model, resumed_optimizer = _new_runner()
    progress = current_adapter.restore(
        checkpoint,
        model=resumed_model,
        optimizer=resumed_optimizer,
    )

    assert progress.global_step == 3
    _assert_nested_equal(resumed_model.state_dict(), model.state_dict())
    _assert_nested_equal(resumed_optimizer.state_dict(), optimizer.state_dict())


def test_callback_schedule_uses_absolute_global_step() -> None:
    args = SimpleNamespace(
        ctx_len=16,
        lr_final=1e-4,
        lr_init=1e-3,
        my_exit_tokens=0,
        real_bsz=4,
        warmup_steps=10,
    )

    first_lr, first_stop = scheduled_learning_rate(args, 0)
    resumed_lr, resumed_stop = scheduled_learning_rate(args, 10)

    assert first_lr == pytest.approx(1e-5)
    assert resumed_lr == pytest.approx(args.lr_init)
    assert first_stop is False
    assert resumed_stop is False


def test_dataset_length_covers_every_accumulated_microbatch() -> None:
    dataset = object.__new__(MyDataset)
    dataset.args = SimpleNamespace(
        accumulate_grad_batches=4,
        epoch_steps=3,
        micro_bsz=2,
    )

    assert len(dataset) == 24


def test_transaction_failure_leaves_no_published_or_partial_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, optimizer = _new_runner()
    original_torch_save = runner_module._torch_save
    calls = 0

    def fail_optimizer(value: object, path: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise CheckpointContractError("synthetic optimizer write failure")
        original_torch_save(value, path)

    monkeypatch.setattr(runner_module, "_torch_save", fail_optimizer)
    destination = tmp_path / "epoch-00000001"

    with pytest.raises(CheckpointContractError, match="optimizer write failure"):
        _adapter().save(
            destination,
            model=model,
            optimizer=optimizer,
            global_step=3,
            next_epoch=1,
        )

    assert not destination.exists()
    assert list(tmp_path.iterdir()) == []


def test_torch_cosine_scheduler_resume_matches_uninterrupted_training(
    tmp_path: Path,
) -> None:
    _seed_all()
    uninterrupted_model, uninterrupted_optimizer = _new_runner()
    uninterrupted_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        uninterrupted_optimizer,
        T_max=6,
        eta_min=1e-4,
    )
    _train_scheduled_steps(
        uninterrupted_model,
        uninterrupted_optimizer,
        uninterrupted_scheduler,
        step_count=6,
    )

    _seed_all()
    interrupted_model, interrupted_optimizer = _new_runner()
    interrupted_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        interrupted_optimizer,
        T_max=6,
        eta_min=1e-4,
    )
    _train_scheduled_steps(
        interrupted_model,
        interrupted_optimizer,
        interrupted_scheduler,
        step_count=3,
    )
    checkpoint = tmp_path / "epoch-00000001"
    manifest = _pytorch_adapter().save(
        checkpoint,
        model=interrupted_model,
        optimizer=interrupted_optimizer,
        scheduler=interrupted_scheduler,
        global_step=3,
        next_epoch=1,
    )

    resumed_model, resumed_optimizer = _new_runner()
    resumed_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        resumed_optimizer,
        T_max=6,
        eta_min=1e-4,
    )
    progress = _pytorch_adapter().restore(
        checkpoint,
        model=resumed_model,
        optimizer=resumed_optimizer,
        scheduler=resumed_scheduler,
    )
    _train_scheduled_steps(
        resumed_model,
        resumed_optimizer,
        resumed_scheduler,
        step_count=3,
    )

    assert progress.global_step == 3
    assert manifest.states["scheduler"].serialization == "torch-lr-scheduler-json-v1"
    _assert_nested_equal(resumed_model.state_dict(), uninterrupted_model.state_dict())
    _assert_nested_equal(
        resumed_optimizer.state_dict(),
        uninterrupted_optimizer.state_dict(),
    )
    _assert_nested_equal(
        resumed_scheduler.state_dict(),
        uninterrupted_scheduler.state_dict(),
    )


def test_fp16_gradient_scaler_state_round_trips(tmp_path: Path) -> None:
    base_adapter = _pytorch_adapter()
    training_config = dict(base_adapter.training_config)
    training_config["precision"] = 16
    adapter = EpochCheckpointRunnerAdapter(
        backend=base_adapter.backend,
        training_config=training_config,
        samples_per_epoch=base_adapter.samples_per_epoch,
    )
    model, optimizer = _new_runner()
    gradient_scaler = torch.amp.GradScaler(
        "cpu",
        init_scale=128.0,
        growth_interval=2,
    )
    inputs = torch.randn(4, 3)
    optimizer.zero_grad(set_to_none=True)
    gradient_scaler.scale(model(inputs).square().mean()).backward()
    gradient_scaler.step(optimizer)
    gradient_scaler.update()
    expected_scaler_state = dict(gradient_scaler.state_dict())

    checkpoint = tmp_path / "epoch-00000001"
    manifest = adapter.save(
        checkpoint,
        model=model,
        optimizer=optimizer,
        gradient_scaler=gradient_scaler,
        global_step=3,
        next_epoch=1,
    )
    resumed_model, resumed_optimizer = _new_runner()
    resumed_gradient_scaler = torch.amp.GradScaler("cpu")
    adapter.restore(
        checkpoint,
        model=resumed_model,
        optimizer=resumed_optimizer,
        gradient_scaler=resumed_gradient_scaler,
    )

    assert (
        manifest.states["gradient_scaler"].serialization == "torch-grad-scaler-json-v1"
    )
    assert resumed_gradient_scaler.state_dict() == expected_scaler_state


def test_restore_rejects_backend_or_config_drift_before_model_mutation(
    tmp_path: Path,
) -> None:
    _seed_all()
    model, optimizer = _new_runner()
    _train_steps(model, optimizer, start_step=0, step_count=3)
    checkpoint = tmp_path / "epoch-00000001"
    _adapter().save(
        checkpoint,
        model=model,
        optimizer=optimizer,
        global_step=3,
        next_epoch=1,
    )

    candidate, candidate_optimizer = _new_runner()
    before = {name: tensor.clone() for name, tensor in candidate.state_dict().items()}
    with pytest.raises(CheckpointContractError, match="backend identity"):
        _adapter(version="incompatible-version").restore(
            checkpoint,
            model=candidate,
            optimizer=candidate_optimizer,
        )
    _assert_nested_equal(candidate.state_dict(), before)

    mismatched_config = EpochCheckpointRunnerAdapter(
        backend=_backend(),
        training_config={"model": "different"},
        samples_per_epoch=12,
    )
    with pytest.raises(CheckpointContractError, match="training config"):
        mismatched_config.restore(
            checkpoint,
            model=candidate,
            optimizer=candidate_optimizer,
        )
    _assert_nested_equal(candidate.state_dict(), before)


@pytest.mark.parametrize(
    "backend",
    [
        BackendIdentity(
            name="pytorch-lightning",
            version="1.9.5",
            strategy="ddp",
            world_size=2,
            state_dict_type="full",
        ),
        BackendIdentity(
            name="deepspeed",
            version="0.17.6",
            strategy="deepspeed_stage_2",
            world_size=2,
            state_dict_type="full",
        ),
    ],
)
def test_runner_rejects_backend_without_complete_state_collection(
    backend: BackendIdentity,
) -> None:
    with pytest.raises(CheckpointContractError, match="only supports"):
        EpochCheckpointRunnerAdapter(
            backend=backend,
            training_config={"model": "tiny-linear"},
            samples_per_epoch=12,
        )


def test_runner_never_publishes_new_training_state_as_pth(tmp_path: Path) -> None:
    model, optimizer = _new_runner()
    with pytest.raises(CheckpointContractError, match="not .pth"):
        _adapter().save(
            tmp_path / "rwkv-1.pth",
            model=model,
            optimizer=optimizer,
            global_step=3,
            next_epoch=1,
        )
    assert not (tmp_path / "rwkv-1.pth").exists()
