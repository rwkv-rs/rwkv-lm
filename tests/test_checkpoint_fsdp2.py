from __future__ import annotations

import copy
import json
import random
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor

from rwkv_lm.activation_checkpointing import (
    FSDP2_ACTIVATION_CHECKPOINT_POLICY,
    FSDP2ActivationCheckpointing,
)
from rwkv_lm.checkpoint import (
    ArtifactRecord,
    CheckpointContractError,
    CheckpointManifest,
    select_checkpoint_loader,
)
from rwkv_lm.checkpoint_fsdp2 import FSDP2CheckpointRunnerAdapter
from rwkv_lm.fsdp2_trainer import (
    _accumulation_windows,
    _gradient_scaler,
    _run_accumulated_optimizer_step,
)


class _RankDivergentScheduler:
    def __init__(self, rank: int) -> None:
        self.rank = rank

    def state_dict(self) -> dict[str, int]:
        return {"rank": self.rank}

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        self.rank = int(state_dict["rank"])


class _CountingTransform(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.forward_calls = 0

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self.forward_calls += 1
        return inputs * 0.5


class _ActivationBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.full((4,), 0.75))
        self.forward_calls = 0

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self.forward_calls += 1
        return torch.tanh(inputs * self.scale)


class _CountingHead(nn.Linear):
    def __init__(self) -> None:
        super().__init__(4, 2, bias=False)
        self.forward_calls = 0

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self.forward_calls += 1
        return super().forward(inputs)


class _ActivationCheckpointModel(nn.Module):
    def __init__(self, *, activation_checkpointing: bool) -> None:
        super().__init__()
        self.input = _CountingTransform()
        self.blocks = nn.ModuleList([_ActivationBlock(), _ActivationBlock()])
        self.head = _CountingHead()
        self.activation_checkpointing = (
            FSDP2ActivationCheckpointing.for_rwkv_blocks(
                self.blocks,
                enabled=activation_checkpointing,
            )
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.activation_checkpointing.run(
            self.input,
            self.input,
            inputs,
        )
        for block in self.blocks:
            hidden = self.activation_checkpointing.run(
                block,
                block,
                hidden,
            )
        return self.activation_checkpointing.run(
            self.head,
            self.head,
            hidden,
        )

    def reset_forward_calls(self) -> None:
        self.input.forward_calls = 0
        self.head.forward_calls = 0
        for block in self.blocks:
            block.forward_calls = 0


def _new_fsdp2_runner() -> tuple[
    nn.Module,
    torch.optim.Optimizer,
    torch.optim.lr_scheduler.LRScheduler,
]:
    torch.manual_seed(20260801)
    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.Tanh(),
        nn.Linear(8, 2),
    )
    fully_shard(model, mesh=init_device_mesh("cpu", (dist.get_world_size(),)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.03)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=6,
        eta_min=1e-4,
    )
    return model, optimizer, scheduler


def _new_activation_checkpoint_runner(
    *,
    activation_checkpointing: bool = True,
) -> tuple[
    _ActivationCheckpointModel,
    torch.optim.Optimizer,
]:
    torch.manual_seed(20260802)
    model = _ActivationCheckpointModel(
        activation_checkpointing=activation_checkpointing,
    )
    mesh = init_device_mesh("cpu", (dist.get_world_size(),))
    for block in model.blocks:
        fully_shard(block, mesh=mesh)
    fully_shard(model, mesh=mesh)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
    return model, optimizer


def _seed_rank_training_rng(rank: int) -> None:
    random.seed(20260900 + rank)
    np.random.seed(20260900 + rank)
    torch.manual_seed(20260900 + rank)


def _train_steps(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    *,
    step_count: int,
    accumulation_steps: int,
    gradient_scaler: torch.amp.GradScaler | None = None,
) -> list[float]:
    losses = []
    for _ in range(step_count):
        microbatches = []
        for _ in range(accumulation_steps):
            inputs = torch.randn(2, 4)
            offset = random.random() + float(np.random.random())
            targets = inputs[:, :2] * 0.25 + offset
            microbatches.append((inputs, targets))

        loss = _run_accumulated_optimizer_step(
            model,
            optimizer,
            microbatches,
            forward_loss=lambda batch, _: (model(batch[0]) - batch[1])
            .square()
            .mean(),
            grad_clip=1.0,
            gradient_scaler=gradient_scaler,
        )
        scheduler.step()
        losses.append(float(loss.detach()))
    return losses


def _train_activation_checkpoint_steps(
    model: _ActivationCheckpointModel,
    optimizer: torch.optim.Optimizer,
    *,
    start_step: int,
    step_count: int,
) -> list[float]:
    losses = []
    for step in range(start_step, start_step + step_count):
        inputs = torch.arange(8, dtype=torch.float32).reshape(2, 4)
        inputs = inputs / 8 + step * 0.05
        targets = torch.tensor(
            [[0.25, -0.5], [-0.25, 0.5]],
            dtype=torch.float32,
        )
        loss = _run_accumulated_optimizer_step(
            model,
            optimizer,
            ((inputs, targets),),
            forward_loss=lambda batch, _: (
                model(batch[0]) - batch[1]
            ).square().mean(),
            grad_clip=1.0,
        )
        losses.append(float(loss.detach()))
    return losses


def _snapshot(value: object) -> object:
    if isinstance(value, DTensor):
        return value.to_local().detach().clone()
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, Mapping):
        return {key: _snapshot(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_snapshot(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_snapshot(item) for item in value)
    return value


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


def _assert_model_state_close(actual: nn.Module, expected: nn.Module) -> None:
    actual_state = _snapshot(actual.state_dict())
    expected_state = _snapshot(expected.state_dict())
    assert isinstance(actual_state, Mapping)
    assert isinstance(expected_state, Mapping)
    assert actual_state.keys() == expected_state.keys()
    for name in expected_state:
        torch.testing.assert_close(
            actual_state[name],
            expected_state[name],
            rtol=1e-6,
            atol=1e-7,
            msg=lambda message, name=name: f"{name}: {message}",
        )


def _fsdp2_worker(rank: int, root: str) -> None:
    root_path = Path(root)
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{root_path / 'process-group-init'}",
        rank=rank,
        world_size=2,
    )
    try:
        accumulated_model, accumulated_optimizer, _ = _new_fsdp2_runner()
        reference_model, reference_optimizer, _ = _new_fsdp2_runner()
        first_inputs = torch.randn(2, 4)
        second_inputs = torch.randn(2, 4)
        first_targets = first_inputs[:, :2] * 0.25 + 0.75
        second_targets = second_inputs[:, :2] * 0.25 - 0.25
        accumulated_loss = _run_accumulated_optimizer_step(
            accumulated_model,
            accumulated_optimizer,
            (
                (first_inputs, first_targets),
                (second_inputs, second_targets),
            ),
            forward_loss=lambda batch, _: (
                accumulated_model(batch[0]) - batch[1]
            )
            .square()
            .mean(),
            grad_clip=0.05,
        )
        reference_inputs = torch.cat((first_inputs, second_inputs))
        reference_targets = torch.cat((first_targets, second_targets))
        reference_loss = _run_accumulated_optimizer_step(
            reference_model,
            reference_optimizer,
            ((reference_inputs, reference_targets),),
            forward_loss=lambda batch, _: (
                reference_model(batch[0]) - batch[1]
            )
            .square()
            .mean(),
            grad_clip=0.05,
        )
        torch.testing.assert_close(
            accumulated_loss,
            reference_loss,
            rtol=1e-6,
            atol=1e-7,
        )
        _assert_model_state_close(accumulated_model, reference_model)

        overflow_model, overflow_optimizer, _ = _new_fsdp2_runner()
        overflow_gradient_scaler = _gradient_scaler(16, device="cpu")
        assert overflow_gradient_scaler is not None
        overflow_model_state = _snapshot(overflow_model.state_dict())
        initial_scale = overflow_gradient_scaler.get_scale()
        overflow_factor = float("inf") if rank == 0 else 1.0
        _run_accumulated_optimizer_step(
            overflow_model,
            overflow_optimizer,
            ((torch.ones(2, 4),),),
            forward_loss=lambda batch, _: (
                overflow_model(batch[0]).square().mean() * overflow_factor
            ),
            grad_clip=1.0,
            gradient_scaler=overflow_gradient_scaler,
        )
        _assert_nested_equal(
            _snapshot(overflow_model.state_dict()),
            overflow_model_state,
        )
        assert overflow_gradient_scaler.get_scale() == (
            initial_scale * overflow_gradient_scaler.get_backoff_factor()
        )

        activation_probe_model, activation_probe_optimizer = (
            _new_activation_checkpoint_runner()
        )
        assert activation_probe_model.activation_checkpointing.enabled
        assert not activation_probe_model.activation_checkpointing.use_reentrant
        assert not activation_probe_model.activation_checkpointing.selects(
            activation_probe_model
        )
        assert not activation_probe_model.activation_checkpointing.selects(
            activation_probe_model.input
        )
        assert not activation_probe_model.activation_checkpointing.selects(
            activation_probe_model.head
        )
        assert all(
            activation_probe_model.activation_checkpointing.selects(block)
            for block in activation_probe_model.blocks
        )
        activation_probe_model.activation_checkpointing.require_rwkv_blocks(
            activation_probe_model.blocks,
            enabled=True,
        )
        with pytest.raises(
            CheckpointContractError,
            match="select exactly RWKV blocks",
        ):
            activation_probe_model.activation_checkpointing.require_rwkv_blocks(
                (
                    activation_probe_model.input,
                    *activation_probe_model.blocks,
                ),
                enabled=True,
            )
        activation_probe_model.reset_forward_calls()
        activation_probe_losses = _train_activation_checkpoint_steps(
            activation_probe_model,
            activation_probe_optimizer,
            start_step=0,
            step_count=1,
        )
        assert activation_probe_model.input.forward_calls == 1
        assert activation_probe_model.head.forward_calls == 1
        assert all(
            block.forward_calls == 2
            for block in activation_probe_model.blocks
        )
        assert all(
            block.scale.grad is not None
            and torch.count_nonzero(block.scale.grad) > 0
            for block in activation_probe_model.blocks
        )

        activation_reference_model, activation_reference_optimizer = (
            _new_activation_checkpoint_runner(
                activation_checkpointing=False,
            )
        )
        activation_reference_model.reset_forward_calls()
        activation_reference_losses = _train_activation_checkpoint_steps(
            activation_reference_model,
            activation_reference_optimizer,
            start_step=0,
            step_count=1,
        )
        assert all(
            block.forward_calls == 1
            for block in activation_reference_model.blocks
        )
        torch.testing.assert_close(
            torch.tensor(activation_probe_losses),
            torch.tensor(activation_reference_losses),
            rtol=0,
            atol=0,
        )
        _assert_nested_equal(
            _snapshot(activation_probe_model.state_dict()),
            _snapshot(activation_reference_model.state_dict()),
        )

        (
            uninterrupted_activation_model,
            uninterrupted_activation_optimizer,
        ) = _new_activation_checkpoint_runner()
        uninterrupted_activation_losses = _train_activation_checkpoint_steps(
            uninterrupted_activation_model,
            uninterrupted_activation_optimizer,
            start_step=0,
            step_count=4,
        )
        uninterrupted_activation_model_state = _snapshot(
            uninterrupted_activation_model.state_dict()
        )
        uninterrupted_activation_optimizer_state = _snapshot(
            uninterrupted_activation_optimizer.state_dict()
        )

        interrupted_activation_model, interrupted_activation_optimizer = (
            _new_activation_checkpoint_runner()
        )
        _train_activation_checkpoint_steps(
            interrupted_activation_model,
            interrupted_activation_optimizer,
            start_step=0,
            step_count=2,
        )
        activation_adapter = FSDP2CheckpointRunnerAdapter.from_process_group(
            training_config={
                "accumulate_grad_batches": 1,
                "batch_size": 4,
                "epoch_steps": 2,
                "fsdp2_activation_checkpointing": (
                    FSDP2_ACTIVATION_CHECKPOINT_POLICY
                ),
                "grad_cp": 1,
                "model": "tiny-activation-checkpoint-fsdp2",
                "precision": 32,
                "seed": 20260802,
            },
            samples_per_epoch=8,
        )
        activation_checkpoint = (
            root_path / "checkpoints" / "activation-epoch-00000001"
        )
        activation_manifest = activation_adapter.save(
            activation_checkpoint,
            model=interrupted_activation_model,
            optimizer=interrupted_activation_optimizer,
            global_step=2,
            next_epoch=1,
        )

        resumed_activation_model, resumed_activation_optimizer = (
            _new_activation_checkpoint_runner()
        )
        activation_progress = activation_adapter.restore(
            activation_checkpoint,
            model=resumed_activation_model,
            optimizer=resumed_activation_optimizer,
        )
        resumed_activation_losses = _train_activation_checkpoint_steps(
            resumed_activation_model,
            resumed_activation_optimizer,
            start_step=2,
            step_count=2,
        )
        assert activation_progress.global_step == 2
        assert activation_progress.epoch == 1
        torch.testing.assert_close(
            torch.tensor(resumed_activation_losses),
            torch.tensor(uninterrupted_activation_losses[2:]),
            rtol=0,
            atol=0,
        )
        _assert_nested_equal(
            _snapshot(resumed_activation_model.state_dict()),
            uninterrupted_activation_model_state,
        )
        _assert_nested_equal(
            _snapshot(resumed_activation_optimizer.state_dict()),
            uninterrupted_activation_optimizer_state,
        )
        assert (
            activation_manifest.states["gradient_scaler"].serialization
            == "torch-grad-scaler-json-v1"
        )

        drifted_activation_config = dict(activation_adapter.training_config)
        drifted_activation_config["fsdp2_activation_checkpointing"] = "disabled"
        drifted_activation_adapter = FSDP2CheckpointRunnerAdapter(
            backend=activation_adapter.backend,
            training_config=drifted_activation_config,
            samples_per_epoch=activation_adapter.samples_per_epoch,
        )
        with pytest.raises(
            CheckpointContractError,
            match="training config",
        ):
            drifted_activation_adapter.restore(
                activation_checkpoint,
                model=resumed_activation_model,
                optimizer=resumed_activation_optimizer,
            )

        uninterrupted_model, uninterrupted_optimizer, uninterrupted_scheduler = (
            _new_fsdp2_runner()
        )
        uninterrupted_gradient_scaler = _gradient_scaler(16, device="cpu")
        assert uninterrupted_gradient_scaler is not None
        _seed_rank_training_rng(rank)
        uninterrupted_losses = _train_steps(
            uninterrupted_model,
            uninterrupted_optimizer,
            uninterrupted_scheduler,
            step_count=6,
            accumulation_steps=2,
            gradient_scaler=uninterrupted_gradient_scaler,
        )
        uninterrupted_model_state = _snapshot(uninterrupted_model.state_dict())
        uninterrupted_optimizer_state = _snapshot(uninterrupted_optimizer.state_dict())
        uninterrupted_scheduler_state = _snapshot(uninterrupted_scheduler.state_dict())
        uninterrupted_gradient_scaler_state = _snapshot(
            uninterrupted_gradient_scaler.state_dict()
        )

        interrupted_model, interrupted_optimizer, interrupted_scheduler = (
            _new_fsdp2_runner()
        )
        interrupted_gradient_scaler = _gradient_scaler(16, device="cpu")
        assert interrupted_gradient_scaler is not None
        _seed_rank_training_rng(rank)
        _train_steps(
            interrupted_model,
            interrupted_optimizer,
            interrupted_scheduler,
            step_count=3,
            accumulation_steps=2,
            gradient_scaler=interrupted_gradient_scaler,
        )
        adapter = FSDP2CheckpointRunnerAdapter.from_process_group(
            training_config={
                "batch_size": 8,
                "accumulate_grad_batches": 2,
                "epoch_steps": 3,
                "model": "tiny-fsdp2",
                "precision": 16,
                "scheduler": "CosineAnnealingLR",
                "seed": 20260801,
            },
            samples_per_epoch=24,
        )
        checkpoint = root_path / "checkpoints" / "epoch-00000001"
        manifest = adapter.save(
            checkpoint,
            model=interrupted_model,
            optimizer=interrupted_optimizer,
            scheduler=interrupted_scheduler,
            gradient_scaler=interrupted_gradient_scaler,
            global_step=3,
            next_epoch=1,
        )

        random.random()
        np.random.random()
        torch.rand(9)
        resumed_model, resumed_optimizer, resumed_scheduler = _new_fsdp2_runner()
        resumed_model_before_restore = _snapshot(resumed_model.state_dict())
        with pytest.raises(
            CheckpointContractError,
            match="gradient scaler enabled state does not match",
        ):
            adapter.restore(
                checkpoint,
                model=resumed_model,
                optimizer=resumed_optimizer,
                scheduler=resumed_scheduler,
            )
        _assert_nested_equal(
            _snapshot(resumed_model.state_dict()),
            resumed_model_before_restore,
        )
        resumed_gradient_scaler = _gradient_scaler(16, device="cpu")
        assert resumed_gradient_scaler is not None
        progress = adapter.restore(
            checkpoint,
            model=resumed_model,
            optimizer=resumed_optimizer,
            scheduler=resumed_scheduler,
            gradient_scaler=resumed_gradient_scaler,
        )
        resumed_losses = _train_steps(
            resumed_model,
            resumed_optimizer,
            resumed_scheduler,
            step_count=3,
            accumulation_steps=2,
            gradient_scaler=resumed_gradient_scaler,
        )

        assert progress.global_step == 3
        assert progress.epoch == 1
        torch.testing.assert_close(
            torch.tensor(resumed_losses),
            torch.tensor(uninterrupted_losses[3:]),
            rtol=0,
            atol=0,
        )
        _assert_nested_equal(
            _snapshot(resumed_model.state_dict()),
            uninterrupted_model_state,
        )
        _assert_nested_equal(
            _snapshot(resumed_optimizer.state_dict()),
            uninterrupted_optimizer_state,
        )
        _assert_nested_equal(
            _snapshot(resumed_scheduler.state_dict()),
            uninterrupted_scheduler_state,
        )
        _assert_nested_equal(
            _snapshot(resumed_gradient_scaler.state_dict()),
            uninterrupted_gradient_scaler_state,
        )
        assert manifest.backend.strategy == "fsdp2"
        assert manifest.backend.state_dict_type == "sharded"
        assert manifest.backend.world_size == 2
        assert (
            manifest.states["model"].serialization
            == "torch-distributed-checkpoint-model-v1"
        )
        assert (
            manifest.states["optimizer"].serialization
            == "torch-distributed-checkpoint-optimizer-v1"
        )
        assert manifest.states["rng"].serialization == "torch-rng-state-per-rank-v1"
        assert (
            manifest.states["gradient_scaler"].serialization
            == "torch-grad-scaler-json-v1"
        )

        divergent_checkpoint = root_path / "checkpoints" / "divergent-scheduler"
        try:
            adapter.save(
                divergent_checkpoint,
                model=resumed_model,
                optimizer=resumed_optimizer,
                scheduler=_RankDivergentScheduler(rank),
                gradient_scaler=resumed_gradient_scaler,
                global_step=6,
                next_epoch=2,
            )
        except CheckpointContractError as error:
            assert "scheduler state must be identical" in str(error)
        else:
            raise AssertionError("rank-divergent scheduler state was published")
        assert not divergent_checkpoint.exists()
        assert not list(
            divergent_checkpoint.parent.glob(f".{divergent_checkpoint.name}.partial-*")
        )

        divergent_progress_checkpoint = root_path / "checkpoints" / "divergent-progress"
        try:
            adapter.save(
                divergent_progress_checkpoint,
                model=resumed_model,
                optimizer=resumed_optimizer,
                scheduler=resumed_scheduler,
                gradient_scaler=resumed_gradient_scaler,
                global_step=6 if rank == 0 else 9,
                next_epoch=2 if rank == 0 else 3,
            )
        except CheckpointContractError as error:
            assert "training progress must be identical" in str(error)
        else:
            raise AssertionError("rank-divergent training progress was published")
        assert not divergent_progress_checkpoint.exists()
        assert not list(
            divergent_progress_checkpoint.parent.glob(
                f".{divergent_progress_checkpoint.name}.partial-*"
            )
        )

        if rank == 0:
            plan = select_checkpoint_loader(checkpoint)
            assert plan.complete_resume
            cursor = json.loads(
                (checkpoint / manifest.states["data_cursor"].path).read_bytes()
            )
            ownership = cursor["rank_ownership"]
            assert ownership["coordinator_rank"] == 0
            assert set(ownership["model"]["shards"]) == {"0", "1"}
            assert set(ownership["optimizer"]["shards"]) == {"0", "1"}
            assert ownership["rng"] == {
                "0": "rng/rank-00000.pt",
                "1": "rng/rank-00001.pt",
            }
            assert (checkpoint / ownership["rng"]["0"]).read_bytes() != (
                checkpoint / ownership["rng"]["1"]
            ).read_bytes()
            assert not list(checkpoint.rglob("*.pth"))
    finally:
        dist.destroy_process_group()


def test_fsdp2_world_size_two_resume_matches_uninterrupted_trajectory(
    tmp_path: Path,
) -> None:
    mp.start_processes(
        _fsdp2_worker,
        args=(str(tmp_path),),
        nprocs=2,
        join=True,
        start_method="spawn",
    )

    checkpoint = tmp_path / "checkpoints" / "epoch-00000001"
    manifest = CheckpointManifest.read(checkpoint / "manifest.json")
    partial_sharded_state = copy.deepcopy(manifest.to_dict())
    partial_sharded_state["states"]["rng"]["serialization"] = "torch-rng-state"
    with pytest.raises(CheckpointContractError, match="selected together"):
        CheckpointManifest.from_dict(partial_sharded_state)

    cursor_path = checkpoint / manifest.states["data_cursor"].path
    original_cursor = cursor_path.read_bytes()
    cursor = json.loads(original_cursor)
    cursor["rank_ownership"]["rng"]["1"] = "rng/rank-00000.pt"
    cursor_path.write_text(
        json.dumps(cursor, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    states = dict(manifest.states)
    states["data_cursor"] = ArtifactRecord.from_file(
        checkpoint,
        manifest.states["data_cursor"].path,
        serialization=manifest.states["data_cursor"].serialization,
    )
    invalid_ownership = CheckpointManifest(
        backend=manifest.backend,
        progress=manifest.progress,
        training_config=manifest.training_config,
        states=states,
    )
    with pytest.raises(CheckpointContractError, match="RNG rank 1 ownership"):
        invalid_ownership.verify_artifacts(checkpoint)

    cursor_path.write_bytes(original_cursor)
    model_shard = next((checkpoint / manifest.states["model"].path).glob("*.distcp"))
    model_shard.write_bytes(model_shard.read_bytes() + b"tampered")
    with pytest.raises(CheckpointContractError, match="artifact (size|digest)"):
        select_checkpoint_loader(checkpoint)


def test_accumulation_windows_reject_partial_effective_batch() -> None:
    assert list(_accumulation_windows(range(6), 3)) == [(0, 1, 2), (3, 4, 5)]
    with pytest.raises(CheckpointContractError, match="inside a gradient accumulation"):
        list(_accumulation_windows(range(5), 3))
