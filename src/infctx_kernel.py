"""Minimal source-pinned loader for RWKV7 state-passing training kernels."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


CHUNK_SIZE = 16


class _StatePassing(torch.autograd.Function):
    @staticmethod
    def forward(ctx, extension, state, r, w, k, v, a, b):
        output, final_state, snapshots, sa = extension.forward(
            state, r, w, k, v, a, b
        )
        ctx.extension = extension
        ctx.state_shape = tuple(state.shape)
        ctx.save_for_backward(r, w, k, v, a, b, snapshots, sa)
        return output, final_state

    @staticmethod
    def backward(ctx, grad_output, grad_final_state):
        r, w, k, v, a, b, snapshots, sa = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        if grad_final_state is None:
            grad_final_state = torch.zeros(
                ctx.state_shape,
                dtype=torch.float32,
                device=grad_output.device,
            )
        gradients = ctx.extension.backward(
            r,
            w,
            k,
            v,
            a,
            b,
            grad_output,
            grad_final_state.contiguous().float(),
            snapshots,
            sa,
        )
        return (None, *gradients)


@lru_cache(maxsize=4)
def load_statepassing_kernel(head_size: int, chunk_size: int = CHUNK_SIZE):
    head_size = int(head_size)
    chunk_size = int(chunk_size)
    if head_size <= 0 or head_size % 4 or head_size > 128:
        raise ValueError(
            "RWKV7 state-passing head_size must be a positive multiple of 4 up to 128"
        )
    if chunk_size <= 0:
        raise ValueError("RWKV7 state-passing chunk_size must be positive")
    checkout = Path(__file__).resolve().parents[1]
    extension = load(
        name=f"rwkv7_statepassing_n{head_size}_c{chunk_size}",
        sources=[
            str(checkout / "cuda/rwkv7_statepassing_clampw.cu"),
            str(checkout / "cuda/rwkv7_statepassing_pybind.cpp"),
        ],
        extra_cflags=[
            "-O3",
            f"-D_N_={head_size}",
            f"-D_CHUNK_LEN_={chunk_size}",
        ],
        extra_cuda_cflags=[
            "-res-usage",
            f"-D_N_={head_size}",
            f"-D_CHUNK_LEN_={chunk_size}",
            "--use_fast_math",
            "-O3",
            "-Xptxas",
            "-O3",
            "--extra-device-vectorization",
        ],
        verbose=True,
    )

    def operation(state, r, w, k, v, a, b):
        batch, tokens, channels = r.shape
        if channels % head_size:
            raise ValueError("RWKV7 flattened width must divide head_size")
        heads = channels // head_size
        vectors = tuple(
            value.view(batch, tokens, heads, head_size)
            for value in (r, w, k, v, a, b)
        )
        output, final_state = _StatePassing.apply(
            extension, state, *vectors
        )
        return output.view(batch, tokens, channels), final_state

    return operation
