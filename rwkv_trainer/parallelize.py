"""RWKV-specific FSDP2 and activation-checkpoint policy."""

from __future__ import annotations

from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, fully_shard
from torchtitan.config import TORCH_DTYPE_MAP, CompileConfig, ParallelismConfig, TrainingConfig
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.full_dtensor import resolve_fsdp_mesh

from .model import RwkvModelAdapter


def parallelize_rwkv(
    model: RwkvModelAdapter,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config,
    dump_folder: str,
) -> RwkvModelAdapter:
    unsupported = {
        "tensor parallel": parallel_dims.tp_enabled,
        "pipeline parallel": parallel_dims.pp_enabled,
        "context parallel": parallel_dims.cp_enabled,
        "expert parallel": parallel_dims.ep_enabled,
    }
    enabled = [name for name, value in unsupported.items() if value]
    if enabled:
        raise NotImplementedError(f"RWKV v1 supports only data parallel/FSDP2; enabled: {enabled}.")
    if compile_config.enable and "model" in compile_config.components:
        raise NotImplementedError("RWKV v1 has not validated torch.compile model wrapping.")

    rwkv = model.rwkv_model
    if ac_config is not None:
        if model.config.infctx:
            raise ValueError(
                "infctx owns non-reentrant chunk checkpointing and rejects nested "
                "block activation checkpointing."
            )
        policy = ac_config.build(dump_folder=dump_folder)
        for index, block in enumerate(rwkv.model.blocks):
            rwkv.model.blocks[index] = policy._wrap_block(
                block, base_fqn=f"hf_model.model.blocks.{index}"
            )

    if parallelism.spmd_backend in ("full_dtensor", "spmd_types"):
        dp_mesh, dp_mesh_dims = resolve_fsdp_mesh(parallel_dims)
    else:
        names = ["dp_replicate", "fsdp"] if parallel_dims.dp_replicate_enabled else ["fsdp"]
        dp_mesh = parallel_dims.get_mesh(names)
        dp_mesh_dims = None
    mp_policy = MixedPrecisionPolicy(
        param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
        reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
        cast_forward_inputs=False,
    )
    fsdp_kwargs = {"mesh": dp_mesh, "mp_policy": mp_policy}
    if dp_mesh_dims is not None:
        fsdp_kwargs["dp_mesh_dims"] = dp_mesh_dims
    if training.enable_cpu_offload:
        fsdp_kwargs["offload_policy"] = CPUOffloadPolicy()
    reshard = parallelism.fsdp_reshard_after_forward != "never"

    for block in rwkv.model.blocks:
        fully_shard(block, **fsdp_kwargs, reshard_after_forward=reshard)
    fully_shard(rwkv.model, **fsdp_kwargs, reshard_after_forward=reshard)
    fully_shard(rwkv, **fsdp_kwargs, reshard_after_forward=reshard)
    return model


__all__ = ["parallelize_rwkv"]
