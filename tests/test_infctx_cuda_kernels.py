import importlib
import os
from types import SimpleNamespace

import pytest

from rwkv_lm.infctx import InfctxBoundary

torch = pytest.importorskip("torch")


def _import_model():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for RWKV infctx kernel smoke tests.")
    os.environ.setdefault("RWKV_MY_TESTING", "x070")
    os.environ.setdefault("RWKV_KERNEL", "")
    os.environ.setdefault("RWKV_HEAD_SIZE", "64")
    os.environ.setdefault("RWKV_HEAD_L2WRAP_CE_CHUNK", "0")
    os.environ.setdefault("RWKV_FLOAT_MODE", "bf16")
    os.environ.setdefault("RWKV_JIT_ON", "1")
    os.environ["RWKV_TRAIN_TYPE"] = "infctx"
    os.environ.setdefault("RWKV_CHUNK_CTX", "16")
    return importlib.import_module("rwkv_lm.model")


def _rand_bf16(*shape, requires_grad=False, scale=1.0):
    tensor = torch.randn(*shape, device="cuda", dtype=torch.bfloat16) * scale
    return tensor.requires_grad_(requires_grad)


def _assert_close(actual, expected):
    torch.testing.assert_close(actual.float(), expected.float(), rtol=0, atol=2e-2)


def _assert_grad_close(actual, expected):
    actual = actual.float()
    expected = expected.float()
    try:
        torch.testing.assert_close(actual, expected, rtol=5e-2, atol=5e-2)
    except AssertionError:
        max_abs = (actual - expected).abs().max()
        scale = expected.abs().max().clamp_min(1.0)
        assert max_abs / scale <= 2e-2


def _clone_with_grad(tensor):
    return tensor.detach().clone().requires_grad_(True)


def _tiny_rwkv(model_module):
    args = SimpleNamespace(
        chunk_ctx=16,
        ctx_len=32,
        dim_att=64,
        dim_ffn=224,
        grad_cp=0,
        head_size=64,
        my_testing="x070",
        n_embd=64,
        n_layer=2,
        strategy="fsdp2",
        train_type="infctx",
        vocab_size=64,
        weight_decay=0.0,
    )
    instance = model_module.RWKV(args).to(device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        for name, parameter in instance.named_parameters():
            if name.endswith(("att.output.weight", "ffn.value.weight")):
                parameter.normal_(mean=0.0, std=0.02)
    return instance


def test_tmix_state_cuda_matches_full_sequence_and_backpropagates_state():
    model = _import_model()
    batch_size, total_len, chunk_ctx, channels = 1, 32, 16, 64
    x_seed = _rand_bf16(batch_size, total_len, channels)
    param_seeds = [_rand_bf16(channels) for _ in range(6)]

    full_x = _clone_with_grad(x_seed)
    full_params = [_clone_with_grad(param) for param in param_seeds]
    full = model.tmix_mix6_bf16_v5(full_x, *full_params)
    full_loss = sum(out.float().square().mean() for out in full)
    full_loss.backward()

    chunk_x = _clone_with_grad(x_seed)
    chunk_params = [_clone_with_grad(param) for param in param_seeds]
    shift0 = torch.zeros(batch_size, channels, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    first = model.tmix_mix6_state_bf16_v5(chunk_x[:, :chunk_ctx], shift0, *chunk_params)
    second = model.tmix_mix6_state_bf16_v5(chunk_x[:, chunk_ctx:], first[6], *chunk_params)
    for idx in range(6):
        _assert_close(torch.cat([first[idx], second[idx]], dim=1), full[idx])

    loss = sum(torch.cat([first[idx], second[idx]], dim=1).float().square().mean() for idx in range(6))
    loss.backward()

    _assert_grad_close(chunk_x.grad, full_x.grad)
    for chunk_param, full_param in zip(chunk_params, full_params):
        _assert_grad_close(chunk_param.grad, full_param.grad)
    assert shift0.grad is not None


def test_cmix_state_cuda_matches_full_sequence_and_backpropagates_state():
    model = _import_model()
    batch_size, total_len, chunk_ctx, channels = 1, 32, 16, 64
    x_seed = _rand_bf16(batch_size, total_len, channels)
    x_k_seed = _rand_bf16(channels)
    key_weight_seed = _rand_bf16(channels * 4, channels)
    value_weight_seed = _rand_bf16(channels, channels * 4)

    full_x = _clone_with_grad(x_seed)
    full_x_k = _clone_with_grad(x_k_seed)
    full_key_weight = _clone_with_grad(key_weight_seed)
    full_value_weight = _clone_with_grad(value_weight_seed)
    full = model._CmixLayerV2Fn.apply(full_x, full_x_k, full_key_weight, full_value_weight)
    full_loss = full.float().square().mean()
    full_loss.backward()

    chunk_x = _clone_with_grad(x_seed)
    chunk_x_k = _clone_with_grad(x_k_seed)
    chunk_key_weight = _clone_with_grad(key_weight_seed)
    chunk_value_weight = _clone_with_grad(value_weight_seed)
    shift0 = torch.zeros(batch_size, channels, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    first, shift1 = model._CmixStateLayerV2Fn.apply(
        chunk_x[:, :chunk_ctx],
        shift0,
        chunk_x_k,
        chunk_key_weight,
        chunk_value_weight,
    )
    second, _shift2 = model._CmixStateLayerV2Fn.apply(
        chunk_x[:, chunk_ctx:],
        shift1,
        chunk_x_k,
        chunk_key_weight,
        chunk_value_weight,
    )
    _assert_close(torch.cat([first, second], dim=1), full)

    loss = torch.cat([first, second], dim=1).float().square().mean()
    loss.backward()

    _assert_grad_close(chunk_x.grad, full_x.grad)
    _assert_grad_close(chunk_x_k.grad, full_x_k.grad)
    _assert_grad_close(chunk_key_weight.grad, full_key_weight.grad)
    _assert_grad_close(chunk_value_weight.grad, full_value_weight.grad)
    assert shift0.grad is not None


def test_wkv_statepassing_cuda_matches_full_sequence_and_backpropagates_state():
    model = _import_model()
    batch_size, total_len, chunk_ctx, channels = 1, 32, 16, 64
    heads, head_size = 1, 64
    tensor_seeds = [_rand_bf16(batch_size, total_len, channels, scale=0.05) for _ in range(6)]
    full_s0 = torch.zeros(batch_size, heads, head_size, head_size, device="cuda", dtype=torch.float32, requires_grad=True)
    full_tensors = [_clone_with_grad(tensor) for tensor in tensor_seeds]
    full_y, full_sT = model.RWKV7_STATEPASSING_CLAMPW_CUDA(full_s0, *full_tensors)
    full_loss = full_y.float().square().mean() + full_sT.float().square().mean()
    full_loss.backward()

    s0 = torch.zeros(batch_size, heads, head_size, head_size, device="cuda", dtype=torch.float32, requires_grad=True)
    tensors = [_clone_with_grad(tensor) for tensor in tensor_seeds]
    first_y, first_sT = model.RWKV7_STATEPASSING_CLAMPW_CUDA(
        s0,
        *[tensor[:, :chunk_ctx] for tensor in tensors],
    )
    second_y, second_sT = model.RWKV7_STATEPASSING_CLAMPW_CUDA(
        first_sT,
        *[tensor[:, chunk_ctx:] for tensor in tensors],
    )
    _assert_close(torch.cat([first_y, second_y], dim=1), full_y)
    torch.testing.assert_close(second_sT, full_sT, rtol=0, atol=2e-2)

    loss = torch.cat([first_y, second_y], dim=1).float().square().mean() + second_sT.float().square().mean()
    loss.backward()

    torch.testing.assert_close(s0.grad, full_s0.grad, rtol=0, atol=2e-2)
    for chunk_tensor, full_tensor in zip(tensors, full_tensors):
        _assert_grad_close(chunk_tensor.grad, full_tensor.grad)


def test_rwkv_infctx_sequence_matches_full_values_and_response_gradients():
    model_module = _import_model()
    torch.manual_seed(20260801)
    full_model = _tiny_rwkv(model_module)
    chunk_model = _tiny_rwkv(model_module)
    response_reference = _tiny_rwkv(model_module)
    chunk_model.load_state_dict(full_model.state_dict())
    response_reference.load_state_dict(full_model.state_dict())
    input_ids = torch.randint(0, 64, (1, 32), device="cuda")
    targets = torch.randint(0, 64, (1, 16), device="cuda")

    full_logits = full_model(input_ids)
    chunked = chunk_model.forward_infctx_sequence(
        input_ids,
        chunk_ctx=16,
        boundary=InfctxBoundary.RESET,
    )
    _assert_close(chunked.output, full_logits)
    full_loss = torch.nn.functional.cross_entropy(
        full_logits[:, 16:].reshape(-1, full_logits.shape[-1]).float(),
        targets.reshape(-1),
    )
    chunk_loss = torch.nn.functional.cross_entropy(
        chunked.output[:, 16:].reshape(-1, chunked.output.shape[-1]).float(),
        targets.reshape(-1),
    )
    torch.testing.assert_close(chunk_loss, full_loss, rtol=5e-3, atol=5e-3)
    assert not chunked.state.shift_states.requires_grad
    assert not chunked.state.wkv_states.requires_grad

    prefix = response_reference.forward_infctx_features(
        input_ids[:, :16],
        chunk_ctx=16,
        boundary=InfctxBoundary.RESET,
    )
    response = response_reference.forward_infctx_sequence(
        input_ids[:, 16:],
        chunk_ctx=16,
        boundary=InfctxBoundary.CONTINUE,
        state=prefix.state,
    )
    reference_loss = torch.nn.functional.cross_entropy(
        response.output.reshape(-1, response.output.shape[-1]).float(),
        targets.reshape(-1),
    )
    _assert_close(chunked.output[:, 16:], response.output)
    torch.testing.assert_close(chunk_loss, reference_loss, rtol=0, atol=0)

    chunk_loss.backward()
    reference_loss.backward()
    for name in (
        "head.weight",
        "blocks.0.att.receptance.weight",
        "blocks.0.ffn.key.weight",
    ):
        chunk_gradient = dict(chunk_model.named_parameters())[name].grad
        reference_gradient = dict(response_reference.named_parameters())[name].grad
        assert chunk_gradient is not None
        assert torch.count_nonzero(chunk_gradient) > 0
        _assert_grad_close(chunk_gradient, reference_gradient)
