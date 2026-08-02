"""Standalone FSDP2 training owner used by the existing RWKV CLI entrypoint."""

from __future__ import annotations

import contextlib
import os
from collections.abc import Callable, Iterable, Iterator, Sequence
from pathlib import Path
from typing import TypeVar

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy, fully_shard
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.utils.data import DataLoader

from .checkpoint import CheckpointContractError
from .checkpoint_fsdp2 import FSDP2CheckpointRunnerAdapter
from .trainer import _checkpoint_training_config, scheduled_learning_rate

_FSDP2_WRAP_POLICY = "rwkv-blocks-then-root-v1"
_T = TypeVar("_T")


def run_fsdp2_training(args, model, train_data, *, resume_checkpoint=None) -> None:
    """Run the existing RWKV model and dataset under composable FSDP2.

    This path is intentionally separate from PyTorch Lightning 1.9.5, which
    predates composable FSDP2. It reuses the repository's one RWKV model,
    optimizer group owner, dataset, loss, and stateless LR schedule.
    """

    initialized_here = False
    if not dist.is_initialized():
        prepare_fsdp2_launch_device()
        dist.init_process_group(backend="nccl", init_method="env://")
        initialized_here = True
    try:
        _run_initialized(args, model, train_data, resume_checkpoint=resume_checkpoint)
    finally:
        if initialized_here and dist.is_initialized():
            dist.destroy_process_group()


def prepare_fsdp2_launch_device() -> None:
    """Bind each torchrun process before RWKV imports or allocates CUDA state."""

    _require_torchrun_environment()
    torch.cuda.set_device(_local_cuda_rank())


def _run_initialized(args, model, train_data, *, resume_checkpoint) -> None:
    if not torch.cuda.is_available():
        raise CheckpointContractError(
            "the RWKV FSDP2 training entry requires CUDA; CPU is supported only "
            "for the tiny checkpoint conformance test"
        )
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    expected_world_size = int(args.num_nodes) * int(args.devices)
    if world_size != expected_world_size:
        raise CheckpointContractError(
            "torchrun WORLD_SIZE must equal num_nodes * devices for FSDP2"
        )
    accumulation_steps = _gradient_accumulation_steps(args)

    local_rank = _local_cuda_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    mesh = init_device_mesh("cuda", (world_size,))
    mixed_precision = _mixed_precision_policy(args.precision)

    model.to(device)
    if isinstance(model, FSDPModule):
        raise CheckpointContractError("RWKV model was already wrapped by fully_shard")
    for block in model.blocks:
        fully_shard(block, mesh=mesh, mp_policy=mixed_precision)
    fully_shard(model, mesh=mesh, mp_policy=mixed_precision)

    optimizer_groups = model.build_optimizer_groups(is_global_zero=rank == 0)
    optimizer_type = torch.optim.AdamW if args.weight_decay > 0 else torch.optim.Adam
    optimizer = optimizer_type(
        optimizer_groups,
        lr=args.lr_init,
        betas=args.betas,
        eps=args.adam_eps,
    )
    gradient_scaler = _gradient_scaler(args.precision)
    checkpoint_config = dict(_checkpoint_training_config(args))
    checkpoint_config.update(
        {
            "distributed_backend": dist.get_backend(),
            "fsdp2_wrap_policy": _FSDP2_WRAP_POLICY,
            "optimizer_backend": f"{optimizer_type.__module__}.{optimizer_type.__name__}",
        }
    )
    checkpoint_adapter = FSDP2CheckpointRunnerAdapter.from_process_group(
        training_config=checkpoint_config,
        samples_per_epoch=int(args.epoch_steps) * int(args.real_bsz),
    )

    global_step = int(args.epoch_begin) * int(args.epoch_steps)
    if resume_checkpoint is not None:
        progress = checkpoint_adapter.restore(
            Path(resume_checkpoint),
            model=model,
            optimizer=optimizer,
            gradient_scaler=gradient_scaler,
        )
        if (
            progress.epoch != int(args.epoch_begin)
            or progress.global_step != global_step
        ):
            raise CheckpointContractError(
                "standard resume progress does not match the epoch schedule"
            )

    data_loader = DataLoader(
        train_data,
        shuffle=False,
        pin_memory=True,
        batch_size=args.micro_bsz,
        num_workers=1,
        persistent_workers=False,
        drop_last=True,
    )
    autocast = _autocast_context(args.precision)
    wandb_run = None
    if args.wandb:

        def initialize_wandb():
            import wandb

            return wandb.init(
                project=args.wandb,
                name=args.run_name + " " + args.my_timestamp,
                config=vars(args),
                save_code=False,
            )

        wandb_run = _rank_zero_phase("initialize W&B", initialize_wandb)
    model.train()
    for epoch in range(int(args.epoch_begin), int(args.epoch_count)):
        train_data.global_rank = rank
        train_data.real_epoch = epoch
        train_data.world_size = world_size
        stop_after_epoch = False
        epoch_loss = 0.0
        optimizer_steps_this_epoch = 0

        for optimizer_step_idx, microbatches in enumerate(
            _accumulation_windows(data_loader, accumulation_steps)
        ):
            if optimizer_step_idx >= int(args.epoch_steps):
                raise CheckpointContractError(
                    "FSDP2 data loader produced more microbatches than the "
                    "declared epoch_steps and accumulation contract"
                )
            lr, reached_token_limit = scheduled_learning_rate(args, global_step)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr * param_group["my_lr_scale"]
                if param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = args.weight_decay

            def forward_loss(batch, microbatch_offset):
                inputs, targets = (
                    tensor.to(device, non_blocking=True) for tensor in batch
                )
                batch_idx = (
                    optimizer_step_idx * accumulation_steps + microbatch_offset
                )
                with autocast():
                    return model.training_step((inputs, targets), batch_idx)

            loss = _run_accumulated_optimizer_step(
                model,
                optimizer,
                microbatches,
                forward_loss=forward_loss,
                grad_clip=args.grad_clip,
                gradient_scaler=gradient_scaler,
            )
            global_step += 1
            optimizer_steps_this_epoch += 1

            reduced_loss = loss.detach().float()
            dist.all_reduce(reduced_loss, op=dist.ReduceOp.SUM)
            epoch_loss += float(reduced_loss / world_size)

            magic_limit = args.magic_prime > 0 and global_step == int(
                args.magic_prime // args.real_bsz
            )
            if reached_token_limit or magic_limit:
                if optimizer_step_idx + 1 != int(args.epoch_steps):
                    raise CheckpointContractError(
                        "standard checkpoint v1 cannot stop and save inside an epoch"
                    )
                stop_after_epoch = True

        next_epoch = epoch + 1
        if (
            optimizer_steps_this_epoch != int(args.epoch_steps)
            or global_step != next_epoch * int(args.epoch_steps)
        ):
            raise CheckpointContractError(
                "FSDP2 runner did not finish the declared epoch boundary"
            )
        save_due = (
            args.epoch_save > 0 and epoch % args.epoch_save == 0
        ) or next_epoch == int(args.epoch_count)
        if save_due or stop_after_epoch:
            checkpoint_adapter.save(
                Path(args.proj_dir) / "checkpoints" / f"epoch-{next_epoch:08d}",
                model=model,
                optimizer=optimizer,
                gradient_scaler=gradient_scaler,
                global_step=global_step,
                next_epoch=next_epoch,
            )
        mean_loss = epoch_loss / int(args.epoch_steps)
        if rank == 0:
            print(f"FSDP2 epoch={epoch} global_step={global_step} loss={mean_loss:.6f}")
        if args.wandb:
            _rank_zero_phase(
                "log W&B epoch",
                lambda loss=mean_loss, step=global_step: wandb_run.log(
                    {"loss": loss, "lr": optimizer.param_groups[0]["lr"]},
                    step=step,
                ),
            )
        if stop_after_epoch:
            break
    dist.barrier()
    if args.wandb:
        _rank_zero_phase("finish W&B", lambda: wandb_run.finish())


def _gradient_accumulation_steps(args) -> int:
    accumulation_steps = getattr(args, "accumulate_grad_batches", 1)
    if (
        isinstance(accumulation_steps, bool)
        or not isinstance(accumulation_steps, int)
        or accumulation_steps <= 0
    ):
        raise CheckpointContractError(
            "accumulate_grad_batches must be a positive integer"
        )
    return accumulation_steps


def _accumulation_windows(
    batches: Iterable[_T],
    accumulation_steps: int,
) -> Iterator[tuple[_T, ...]]:
    """Group a complete epoch into fixed optimizer-step windows."""

    iterator = iter(batches)
    while True:
        window = []
        for _ in range(accumulation_steps):
            try:
                window.append(next(iterator))
            except StopIteration:
                if window:
                    raise CheckpointContractError(
                        "FSDP2 data loader ended inside a gradient accumulation window"
                    ) from None
                return
        yield tuple(window)


def _run_accumulated_optimizer_step(
    model: FSDPModule,
    optimizer: torch.optim.Optimizer,
    microbatches: Sequence[_T],
    *,
    forward_loss: Callable[[_T, int], torch.Tensor],
    grad_clip: float,
    gradient_scaler: torch.amp.GradScaler | None = None,
) -> torch.Tensor:
    """Run one FSDP2 optimizer step over one effective global batch."""

    if not isinstance(model, FSDPModule):
        raise CheckpointContractError(
            "gradient accumulation requires a fully_shard-wrapped model"
        )
    if not microbatches:
        raise CheckpointContractError(
            "gradient accumulation requires at least one microbatch"
        )

    optimizer.zero_grad(set_to_none=True)
    mean_loss = None
    accumulation_steps = len(microbatches)
    for microbatch_offset, batch in enumerate(microbatches):
        synchronize = microbatch_offset + 1 == accumulation_steps
        model.set_requires_gradient_sync(synchronize)
        model.set_reshard_after_backward(synchronize)
        loss = forward_loss(batch, microbatch_offset)
        if not isinstance(loss, torch.Tensor) or loss.ndim != 0:
            raise CheckpointContractError(
                "FSDP2 training_step must return a scalar Torch loss"
            )
        scaled_loss = loss / accumulation_steps
        if gradient_scaler is not None:
            scaled_loss = gradient_scaler.scale(scaled_loss)
        scaled_loss.backward()
        detached_loss = loss.detach().float() / accumulation_steps
        mean_loss = (
            detached_loss if mean_loss is None else mean_loss + detached_loss
        )

    if gradient_scaler is not None:
        gradient_scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    if gradient_scaler is None:
        optimizer.step()
    else:
        gradient_scaler.step(optimizer)
        gradient_scaler.update()
    assert mean_loss is not None
    return mean_loss


def _require_torchrun_environment() -> None:
    missing = [
        name
        for name in (
            "RANK",
            "WORLD_SIZE",
            "LOCAL_RANK",
            "MASTER_ADDR",
            "MASTER_PORT",
        )
        if not os.environ.get(name)
    ]
    if missing:
        raise CheckpointContractError(
            "FSDP2 strategy must be launched with torchrun; missing: "
            + ", ".join(missing)
        )


def _local_cuda_rank() -> int:
    local_rank = int(
        os.environ.get("LOCAL_RANK", dist.get_rank() if dist.is_initialized() else 0)
    )
    if not torch.cuda.is_available():
        raise CheckpointContractError("FSDP2 training requires CUDA")
    if local_rank < 0 or local_rank >= torch.cuda.device_count():
        raise CheckpointContractError(
            "LOCAL_RANK does not identify a visible CUDA device"
        )
    return local_rank


def _rank_zero_phase(owner: str, action: Callable[[], _T]) -> _T | None:
    value = None
    error = None
    if dist.get_rank() == 0:
        try:
            value = action()
        except Exception as caught:  # noqa: BLE001 - synchronize rank-zero failure
            error = f"{type(caught).__name__}: {caught}"
    errors: list[str | None] = [None] * dist.get_world_size()
    dist.all_gather_object(errors, error)
    if any(message is not None for message in errors):
        raise CheckpointContractError(f"{owner} failed on rank 0: {errors[0]}")
    return value


def _mixed_precision_policy(precision) -> MixedPrecisionPolicy:
    if precision == "bf16":
        return MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            output_dtype=torch.bfloat16,
        )
    if precision == 16:
        return MixedPrecisionPolicy(
            param_dtype=torch.float16,
            reduce_dtype=torch.float32,
            output_dtype=torch.float16,
        )
    if precision == 32:
        return MixedPrecisionPolicy()
    raise CheckpointContractError(f"unsupported FSDP2 precision: {precision!r}")


def _autocast_context(precision):
    if precision == "bf16":
        return lambda: torch.autocast("cuda", dtype=torch.bfloat16)
    if precision == 16:
        return lambda: torch.autocast("cuda", dtype=torch.float16)
    if precision == 32:
        return contextlib.nullcontext
    raise CheckpointContractError(f"unsupported FSDP2 precision: {precision!r}")


def _gradient_scaler(
    precision,
    *,
    device: str = "cuda",
) -> ShardedGradScaler | None:
    if precision == 16:
        return ShardedGradScaler(
            device=device,
            process_group=dist.group.WORLD,
        )
    if precision in {"bf16", 32}:
        return None
    raise CheckpointContractError(f"unsupported FSDP2 precision: {precision!r}")


__all__ = ["prepare_fsdp2_launch_device", "run_fsdp2_training"]
