"""Apply TorchTitan parallelism and training transforms to RWKV-7."""

from __future__ import annotations

from torchtitan.config import (
    TORCH_DTYPE_MAP,
    CompileConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import ActivationCheckpointingConfig
from torchtitan.distributed.compile import apply_compile
from torchtitan.distributed.fsdp import apply_fsdp_to_decoder
from torchtitan.distributed.full_dtensor import resolve_fsdp_mesh, validate_config

from .model import Rwkv7Model


def parallelize_rwkv7(
    model: Rwkv7Model,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointingConfig,
    dump_folder: str,
) -> Rwkv7Model:
    """Apply activation checkpointing, compile, then FSDP in TorchTitan order."""
    if parallel_dims.tp_enabled or parallel_dims.cp_enabled:
        raise ValueError("RWKV-7 currently supports TorchTitan data parallelism only")

    if parallelism.spmd_backend in ("full_dtensor", "spmd_types"):
        validate_config(parallel_dims, model)
        model.parallelize(parallel_dims)

    if ac_config is not None:
        ac_config.build(dump_folder=dump_folder).apply(model)

    if compile_config.enable and "model" in compile_config.components:
        apply_compile(model, compile_config)

    if parallelism.spmd_backend in ("full_dtensor", "spmd_types"):
        dp_mesh, dp_mesh_dims = resolve_fsdp_mesh(parallel_dims)
    else:
        mesh_names = (
            ["dp_replicate", "fsdp"] if parallel_dims.dp_replicate_enabled else ["fsdp"]
        )
        dp_mesh = parallel_dims.get_mesh(mesh_names)
        dp_mesh_dims = None

    apply_fsdp_to_decoder(
        model,
        dp_mesh,
        param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
        reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
        pp_enabled=parallel_dims.pp_enabled,
        cpu_offload=training.enable_cpu_offload,
        reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
        dp_mesh_dims=dp_mesh_dims,
        enable_symm_mem=parallelism.enable_fsdp_symm_mem,
    )
    return model
