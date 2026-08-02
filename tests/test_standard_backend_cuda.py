"""CUDA integration tests for the standard RWKV-7 backend boundary.

Kernel-level correctness belongs to fla-rwkv and FlashRWKV. These tests cover
the rwkv-lm contract at the standard Transformers model boundary: real backend
selection, recurrent state handoff, TBPTT detach, and PEFT gradients.
"""

from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from rwkv_lm.infctx import InfctxBoundary
from rwkv_lm.peft import LoraConfig, lora_parameter_names
from rwkv_lm.standard_model import (
    configure_standard_rwkv7_peft,
    create_standard_rwkv7_model,
    standard_rwkv7_blocks,
    standard_rwkv7_infctx_forward,
    standard_rwkv7_training_loss,
)


def _require_standard_flash_stack():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the standard FlashRWKV integration test")
    pytest.importorskip("transformers.models.rwkv7.modeling_rwkv7")
    pytest.importorskip("flash_rwkv")
    pytest.importorskip("fla.ops.rwkv7")
    from fla.ops.rwkv7 import get_last_rwkv7_provider
    from fla.ops.rwkv7.backends.flash_rwkv import FlashRWKVBackend

    if not FlashRWKVBackend.is_available():
        pytest.skip(
            "the installed fla-rwkv / FlashRWKV provider contract is unavailable"
        )
    return get_last_rwkv7_provider


def _args(backend: str) -> SimpleNamespace:
    return SimpleNamespace(
        ctx_len=32,
        dim_ffn=224,
        head_size=64,
        n_embd=64,
        n_layer=2,
        vocab_size=64,
        wkv_backend=backend,
    )


def _model_pair():
    torch.manual_seed(20260802)
    reference = create_standard_rwkv7_model(_args("reference"))
    with torch.no_grad():
        for block in standard_rwkv7_blocks(reference):
            block.att.g1.weight.normal_(mean=0.0, std=0.02)
            block.att.g2.weight.normal_(mean=0.0, std=0.02)
            block.att.output.weight.normal_(mean=0.0, std=0.02)
    flash = create_standard_rwkv7_model(_args("flash_rwkv"))
    flash.load_state_dict(reference.state_dict(), strict=True)
    reference.to(device="cuda", dtype=torch.bfloat16)
    flash.to(device="cuda", dtype=torch.bfloat16)
    return reference, flash


def _assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(
        actual.float(),
        expected.float(),
        rtol=3e-2,
        atol=3e-2,
    )


def _assert_response_gradient_close(
    actual: torch.Tensor | None,
    expected: torch.Tensor | None,
) -> None:
    assert actual is not None
    assert expected is not None
    torch.testing.assert_close(
        actual.float(),
        expected.float(),
        rtol=8e-2,
        atol=5e-2,
    )


def test_standard_flash_infctx_matches_detached_reference_and_response_gradients():
    get_last_rwkv7_provider = _require_standard_flash_stack()
    reference, flash = _model_pair()
    input_ids = torch.arange(1, 33, device="cuda").reshape(1, 32)
    targets = torch.arange(17, 33, device="cuda").reshape(1, 16).roll(-1, dims=1)

    full_reference = reference(
        input_ids=input_ids,
        use_cache=True,
        return_dict=True,
    )
    chunked = standard_rwkv7_infctx_forward(
        flash,
        input_ids,
        chunk_ctx=16,
        ctx_len=32,
        boundary=InfctxBoundary.RESET,
    )
    _assert_close(chunked.output, full_reference.logits)
    assert chunked.state.tokens_seen == 32
    assert not chunked.state.shift_states.requires_grad
    assert not chunked.state.wkv_states.requires_grad
    assert get_last_rwkv7_provider() == "flash_rwkv"
    assert all(
        block.att.last_wkv_backend == "flash_rwkv"
        for block in standard_rwkv7_blocks(flash)
    )

    reference.zero_grad(set_to_none=True)
    prefix = reference(
        input_ids=input_ids[:, :16],
        use_cache=True,
        return_dict=True,
    )
    response = reference(
        input_ids=input_ids[:, 16:],
        state=tuple(value.detach() for value in prefix.state),
        use_cache=True,
        return_dict=True,
    )
    _assert_close(chunked.output[:, 16:], response.logits)

    flash_loss = F.cross_entropy(
        chunked.output[:, 16:].float().reshape(-1, 64),
        targets.reshape(-1),
    )
    reference_loss = F.cross_entropy(
        response.logits.float().reshape(-1, 64),
        targets.reshape(-1),
    )
    torch.testing.assert_close(
        flash_loss,
        reference_loss,
        rtol=3e-2,
        atol=3e-2,
    )
    flash_loss.backward()
    reference_loss.backward()

    flash_parameters = dict(flash.named_parameters())
    reference_parameters = dict(reference.named_parameters())
    for name in (
        "head.weight",
        "model.blocks.0.att.output.weight",
        "model.blocks.0.ffn.value.weight",
    ):
        _assert_response_gradient_close(
            flash_parameters[name].grad,
            reference_parameters[name].grad,
        )
    embedding_gradient = flash.model.embeddings.weight.grad
    assert embedding_gradient is not None
    assert torch.count_nonzero(embedding_gradient[1:17]) == 0
    assert torch.count_nonzero(embedding_gradient[17:33]) > 0


def test_standard_flash_peft_backward_is_owned_by_public_projections():
    get_last_rwkv7_provider = _require_standard_flash_stack()
    _reference, model = _model_pair()
    config = LoraConfig(
        rank=2,
        alpha=4.0,
        dropout=0.0,
        target_modules=("time_mix.output", "channel_mix.value"),
    )
    configure_standard_rwkv7_peft(model, config)
    trainable = lora_parameter_names(model)
    assert trainable
    assert trainable == tuple(
        sorted(
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        )
    )

    input_ids = torch.arange(1, 33, device="cuda").reshape(1, 32)
    labels = input_ids.roll(-1, dims=1)
    loss = standard_rwkv7_training_loss(model, input_ids, labels)
    loss.backward()

    parameters = dict(model.named_parameters())
    assert all(parameters[name].grad is not None for name in trainable)
    assert all(
        torch.count_nonzero(parameters[name].grad) > 0
        for name in trainable
        if name.endswith(".lora_B")
    )
    assert all(
        parameter.grad is None
        for name, parameter in model.named_parameters()
        if name not in trainable
    )
    assert get_last_rwkv7_provider() == "flash_rwkv"
    assert all(
        block.att.last_wkv_backend == "flash_rwkv"
        for block in standard_rwkv7_blocks(model)
    )
