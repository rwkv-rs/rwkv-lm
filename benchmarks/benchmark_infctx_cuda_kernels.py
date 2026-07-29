#!/usr/bin/env python3
"""Benchmark RWKV infctx state-aware CUDA kernels against existing fused kernels."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Callable

import torch


def _prepare_import() -> None:
    os.environ.setdefault("RWKV_MY_TESTING", "x070")
    os.environ.setdefault("RWKV_KERNEL", "")
    os.environ.setdefault("RWKV_HEAD_SIZE", "64")
    os.environ.setdefault("RWKV_HEAD_L2WRAP_CE_CHUNK", "0")
    os.environ.setdefault("RWKV_FLOAT_MODE", "bf16")
    os.environ.setdefault("RWKV_JIT_ON", "1")
    os.environ["RWKV_TRAIN_TYPE"] = "infctx"
    os.environ.setdefault("RWKV_CHUNK_CTX", "16")
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))


def _bf16_rand(
    shape: tuple[int, ...],
    *,
    requires_grad: bool = False,
    scale: float = 1.0,
) -> torch.Tensor:
    tensor = torch.randn(*shape, device="cuda", dtype=torch.bfloat16) * scale
    return tensor.requires_grad_(requires_grad)


def _zero_grads(tensors: list[torch.Tensor]) -> None:
    for tensor in tensors:
        tensor.grad = None


def _time_cuda(
    name: str,
    fn: Callable[[], None],
    *,
    warmup: int,
    iterations: int,
    repeats: int,
) -> dict[str, float | int | str]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) / iterations)

    return {
        "name": name,
        "warmup": warmup,
        "iterations": iterations,
        "repeats": repeats,
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.fmean(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def _ratio(state: dict[str, float | int | str], old: dict[str, float | int | str]) -> float:
    return float(state["median_ms"]) / float(old["median_ms"])


def _build_cases(model, *, batch_size: int, chunk_ctx: int, channels: int):
    total_len = chunk_ctx * 2
    heads = channels // 64
    head_size = 64

    tmix_x = _bf16_rand((batch_size, chunk_ctx, channels))
    tmix_state = torch.randn(batch_size, channels, device="cuda", dtype=torch.bfloat16)
    tmix_params = [_bf16_rand((channels,)) for _ in range(6)]
    tmix_full_x = _bf16_rand((batch_size, total_len, channels))

    tmix_x_bwd = _bf16_rand((batch_size, chunk_ctx, channels), requires_grad=True)
    tmix_state_bwd = _bf16_rand((batch_size, channels), requires_grad=True)
    tmix_params_bwd = [_bf16_rand((channels,), requires_grad=True) for _ in range(6)]

    tmix_full_x_bwd = _bf16_rand((batch_size, total_len, channels), requires_grad=True)
    tmix_chunk_state_bwd = _bf16_rand((batch_size, channels), requires_grad=True)
    tmix_full_params_bwd = [_bf16_rand((channels,), requires_grad=True) for _ in range(6)]
    tmix_chunk_params_bwd = [param.detach().clone().requires_grad_(True) for param in tmix_full_params_bwd]

    cmix_x = _bf16_rand((batch_size, chunk_ctx, channels))
    cmix_state = _bf16_rand((batch_size, channels))
    cmix_x_k = _bf16_rand((channels,))
    cmix_key = _bf16_rand((channels * 4, channels))
    cmix_value = _bf16_rand((channels, channels * 4))
    cmix_full_x = _bf16_rand((batch_size, total_len, channels))

    cmix_x_bwd = _bf16_rand((batch_size, chunk_ctx, channels), requires_grad=True)
    cmix_state_bwd = _bf16_rand((batch_size, channels), requires_grad=True)
    cmix_x_k_bwd = _bf16_rand((channels,), requires_grad=True)
    cmix_key_bwd = _bf16_rand((channels * 4, channels), requires_grad=True)
    cmix_value_bwd = _bf16_rand((channels, channels * 4), requires_grad=True)

    cmix_full_x_bwd = _bf16_rand((batch_size, total_len, channels), requires_grad=True)
    cmix_chunk_state_bwd = _bf16_rand((batch_size, channels), requires_grad=True)
    cmix_full_x_k_bwd = _bf16_rand((channels,), requires_grad=True)
    cmix_full_key_bwd = _bf16_rand((channels * 4, channels), requires_grad=True)
    cmix_full_value_bwd = _bf16_rand((channels, channels * 4), requires_grad=True)
    cmix_chunk_x_k_bwd = cmix_full_x_k_bwd.detach().clone().requires_grad_(True)
    cmix_chunk_key_bwd = cmix_full_key_bwd.detach().clone().requires_grad_(True)
    cmix_chunk_value_bwd = cmix_full_value_bwd.detach().clone().requires_grad_(True)

    wkv_tensors = [_bf16_rand((batch_size, chunk_ctx, channels), scale=0.05) for _ in range(6)]
    wkv_state = torch.randn(batch_size, heads, head_size, head_size, device="cuda", dtype=torch.float32)
    wkv_full_tensors = [_bf16_rand((batch_size, total_len, channels), scale=0.05) for _ in range(6)]

    wkv_tensors_bwd = [
        _bf16_rand((batch_size, chunk_ctx, channels), requires_grad=True, scale=0.05) for _ in range(6)
    ]
    wkv_state_bwd = torch.randn(
        batch_size, heads, head_size, head_size, device="cuda", dtype=torch.float32, requires_grad=True
    )

    wkv_full_tensors_bwd = [
        _bf16_rand((batch_size, total_len, channels), requires_grad=True, scale=0.05) for _ in range(6)
    ]
    wkv_chunk_tensors_bwd = [tensor.detach().clone().requires_grad_(True) for tensor in wkv_full_tensors_bwd]
    wkv_chunk_state_bwd = torch.randn(
        batch_size, heads, head_size, head_size, device="cuda", dtype=torch.float32, requires_grad=True
    )

    def tmix_old_forward():
        with torch.no_grad():
            model.tmix_mix6_bf16_v5(tmix_x, *tmix_params)

    def tmix_state_forward():
        with torch.no_grad():
            model.tmix_mix6_state_bf16_v5(tmix_x, tmix_state, *tmix_params)

    def tmix_old_backward():
        _zero_grads([tmix_x_bwd, *tmix_params_bwd])
        outs = model.tmix_mix6_bf16_v5(tmix_x_bwd, *tmix_params_bwd)
        loss = sum(out.float().square().mean() for out in outs)
        loss.backward()

    def tmix_state_backward():
        _zero_grads([tmix_x_bwd, tmix_state_bwd, *tmix_params_bwd])
        outs = model.tmix_mix6_state_bf16_v5(tmix_x_bwd, tmix_state_bwd, *tmix_params_bwd)
        loss = sum(out.float().square().mean() for out in outs)
        loss.backward()

    def tmix_old_full_forward():
        with torch.no_grad():
            model.tmix_mix6_bf16_v5(tmix_full_x, *tmix_params)

    def tmix_state_chunked_forward():
        with torch.no_grad():
            first = model.tmix_mix6_state_bf16_v5(tmix_full_x[:, :chunk_ctx], tmix_state, *tmix_params)
            model.tmix_mix6_state_bf16_v5(tmix_full_x[:, chunk_ctx:], first[6], *tmix_params)

    def tmix_old_full_backward():
        _zero_grads([tmix_full_x_bwd, *tmix_full_params_bwd])
        outs = model.tmix_mix6_bf16_v5(tmix_full_x_bwd, *tmix_full_params_bwd)
        loss = sum(out.float().square().mean() for out in outs)
        loss.backward()

    def tmix_state_chunked_backward():
        _zero_grads([tmix_full_x_bwd, tmix_chunk_state_bwd, *tmix_chunk_params_bwd])
        first = model.tmix_mix6_state_bf16_v5(
            tmix_full_x_bwd[:, :chunk_ctx],
            tmix_chunk_state_bwd,
            *tmix_chunk_params_bwd,
        )
        second = model.tmix_mix6_state_bf16_v5(
            tmix_full_x_bwd[:, chunk_ctx:],
            first[6],
            *tmix_chunk_params_bwd,
        )
        loss = sum(torch.cat([first[idx], second[idx]], dim=1).float().square().mean() for idx in range(6))
        loss.backward()

    def cmix_old_forward():
        with torch.no_grad():
            model._CmixLayerV2Fn.apply(cmix_x, cmix_x_k, cmix_key, cmix_value)

    def cmix_state_forward():
        with torch.no_grad():
            model._CmixStateLayerV2Fn.apply(cmix_x, cmix_state, cmix_x_k, cmix_key, cmix_value)

    def cmix_old_backward():
        _zero_grads([cmix_x_bwd, cmix_x_k_bwd, cmix_key_bwd, cmix_value_bwd])
        out = model._CmixLayerV2Fn.apply(cmix_x_bwd, cmix_x_k_bwd, cmix_key_bwd, cmix_value_bwd)
        out.float().square().mean().backward()

    def cmix_state_backward():
        _zero_grads([cmix_x_bwd, cmix_state_bwd, cmix_x_k_bwd, cmix_key_bwd, cmix_value_bwd])
        out, new_state = model._CmixStateLayerV2Fn.apply(
            cmix_x_bwd,
            cmix_state_bwd,
            cmix_x_k_bwd,
            cmix_key_bwd,
            cmix_value_bwd,
        )
        (out.float().square().mean() + new_state.float().square().mean()).backward()

    def cmix_old_full_forward():
        with torch.no_grad():
            model._CmixLayerV2Fn.apply(cmix_full_x, cmix_x_k, cmix_key, cmix_value)

    def cmix_state_chunked_forward():
        with torch.no_grad():
            first, state = model._CmixStateLayerV2Fn.apply(
                cmix_full_x[:, :chunk_ctx], cmix_state, cmix_x_k, cmix_key, cmix_value
            )
            del first
            model._CmixStateLayerV2Fn.apply(cmix_full_x[:, chunk_ctx:], state, cmix_x_k, cmix_key, cmix_value)

    def cmix_old_full_backward():
        _zero_grads([cmix_full_x_bwd, cmix_full_x_k_bwd, cmix_full_key_bwd, cmix_full_value_bwd])
        out = model._CmixLayerV2Fn.apply(
            cmix_full_x_bwd,
            cmix_full_x_k_bwd,
            cmix_full_key_bwd,
            cmix_full_value_bwd,
        )
        out.float().square().mean().backward()

    def cmix_state_chunked_backward():
        _zero_grads([cmix_full_x_bwd, cmix_chunk_state_bwd, cmix_chunk_x_k_bwd, cmix_chunk_key_bwd, cmix_chunk_value_bwd])
        first, state = model._CmixStateLayerV2Fn.apply(
            cmix_full_x_bwd[:, :chunk_ctx],
            cmix_chunk_state_bwd,
            cmix_chunk_x_k_bwd,
            cmix_chunk_key_bwd,
            cmix_chunk_value_bwd,
        )
        second, _ = model._CmixStateLayerV2Fn.apply(
            cmix_full_x_bwd[:, chunk_ctx:],
            state,
            cmix_chunk_x_k_bwd,
            cmix_chunk_key_bwd,
            cmix_chunk_value_bwd,
        )
        torch.cat([first, second], dim=1).float().square().mean().backward()

    def wkv_old_forward():
        with torch.no_grad():
            model.RWKV7_CLAMPW_CUDA(*wkv_tensors)

    def wkv_state_forward():
        with torch.no_grad():
            model.RWKV7_STATEPASSING_CLAMPW_CUDA(wkv_state, *wkv_tensors)

    def wkv_old_backward():
        _zero_grads(wkv_tensors_bwd)
        out = model.RWKV7_CLAMPW_CUDA(*wkv_tensors_bwd)
        out.float().square().mean().backward()

    def wkv_state_backward():
        _zero_grads([wkv_state_bwd, *wkv_tensors_bwd])
        out, new_state = model.RWKV7_STATEPASSING_CLAMPW_CUDA(wkv_state_bwd, *wkv_tensors_bwd)
        (out.float().square().mean() + new_state.float().square().mean()).backward()

    def wkv_old_full_forward():
        with torch.no_grad():
            model.RWKV7_CLAMPW_CUDA(*wkv_full_tensors)

    def wkv_state_chunked_forward():
        with torch.no_grad():
            first_y, state = model.RWKV7_STATEPASSING_CLAMPW_CUDA(
                wkv_state,
                *[tensor[:, :chunk_ctx] for tensor in wkv_full_tensors],
            )
            del first_y
            model.RWKV7_STATEPASSING_CLAMPW_CUDA(
                state,
                *[tensor[:, chunk_ctx:] for tensor in wkv_full_tensors],
            )

    def wkv_old_full_backward():
        _zero_grads(wkv_full_tensors_bwd)
        out = model.RWKV7_CLAMPW_CUDA(*wkv_full_tensors_bwd)
        out.float().square().mean().backward()

    def wkv_state_chunked_backward():
        _zero_grads([wkv_chunk_state_bwd, *wkv_chunk_tensors_bwd])
        first_y, state = model.RWKV7_STATEPASSING_CLAMPW_CUDA(
            wkv_chunk_state_bwd,
            *[tensor[:, :chunk_ctx] for tensor in wkv_chunk_tensors_bwd],
        )
        second_y, new_state = model.RWKV7_STATEPASSING_CLAMPW_CUDA(
            state,
            *[tensor[:, chunk_ctx:] for tensor in wkv_chunk_tensors_bwd],
        )
        (torch.cat([first_y, second_y], dim=1).float().square().mean() + new_state.float().square().mean()).backward()

    return [
        ("tmix_forward_old_chunk", tmix_old_forward),
        ("tmix_forward_state_chunk", tmix_state_forward),
        ("tmix_backward_old_chunk", tmix_old_backward),
        ("tmix_backward_state_chunk", tmix_state_backward),
        ("tmix_forward_old_full", tmix_old_full_forward),
        ("tmix_forward_state_chunked", tmix_state_chunked_forward),
        ("tmix_backward_old_full", tmix_old_full_backward),
        ("tmix_backward_state_chunked", tmix_state_chunked_backward),
        ("cmix_forward_old_chunk", cmix_old_forward),
        ("cmix_forward_state_chunk", cmix_state_forward),
        ("cmix_backward_old_chunk", cmix_old_backward),
        ("cmix_backward_state_chunk", cmix_state_backward),
        ("cmix_forward_old_full", cmix_old_full_forward),
        ("cmix_forward_state_chunked", cmix_state_chunked_forward),
        ("cmix_backward_old_full", cmix_old_full_backward),
        ("cmix_backward_state_chunked", cmix_state_chunked_backward),
        ("wkv_forward_old_chunk", wkv_old_forward),
        ("wkv_forward_state_chunk", wkv_state_forward),
        ("wkv_backward_old_chunk", wkv_old_backward),
        ("wkv_backward_state_chunk", wkv_state_backward),
        ("wkv_forward_old_full", wkv_old_full_forward),
        ("wkv_forward_state_chunked", wkv_state_chunked_forward),
        ("wkv_backward_old_full", wkv_old_full_backward),
        ("wkv_backward_state_chunked", wkv_state_chunked_backward),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--chunk-ctx", type=int, default=512)
    parser.add_argument("--channels", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--backward-iterations", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    if args.channels % 64 != 0:
        raise ValueError("--channels must be divisible by 64.")
    if args.chunk_ctx % 16 != 0:
        raise ValueError("--chunk-ctx must be divisible by 16.")

    _prepare_import()
    import src.model as model

    torch.manual_seed(123)
    torch.cuda.set_device(0)
    torch.cuda.synchronize()

    cases = _build_cases(model, batch_size=args.batch_size, chunk_ctx=args.chunk_ctx, channels=args.channels)
    metrics = []
    for name, fn in cases:
        iterations = args.backward_iterations if "backward" in name else args.iterations
        metrics.append(
            _time_cuda(
                name,
                fn,
                warmup=args.warmup,
                iterations=iterations,
                repeats=args.repeats,
            )
        )

    by_name = {metric["name"]: metric for metric in metrics}
    ratios = {}
    for prefix in ("tmix", "cmix", "wkv"):
        for stage in ("forward", "backward"):
            ratios[f"{prefix}_{stage}_state_vs_old_chunk"] = _ratio(
                by_name[f"{prefix}_{stage}_state_chunk"],
                by_name[f"{prefix}_{stage}_old_chunk"],
            )
            ratios[f"{prefix}_{stage}_state_chunked_vs_old_full"] = _ratio(
                by_name[f"{prefix}_{stage}_state_chunked"],
                by_name[f"{prefix}_{stage}_old_full"],
            )

    result = {
        "metadata": {
            "device": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "batch_size": args.batch_size,
            "chunk_ctx": args.chunk_ctx,
            "total_ctx_for_chunked": args.chunk_ctx * 2,
            "channels": args.channels,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "backward_iterations": args.backward_iterations,
            "repeats": args.repeats,
            "RWKV_JIT_ON": os.environ.get("RWKV_JIT_ON"),
            "RWKV_TRAIN_TYPE": os.environ.get("RWKV_TRAIN_TYPE"),
        },
        "metrics": metrics,
        "ratios": ratios,
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
