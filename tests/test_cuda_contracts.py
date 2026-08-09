from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest
import torch
from transformers import RwkvConfig, RwkvForCausalLM, RwkvTrainingState

from rwkv_trainer.loss import RwkvL2WrapLoss
from rwkv_trainer.model import RwkvModelAdapter
from rwkv_trainer.optimizer import RwkvOptimizersContainer

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or importlib.util.find_spec("flashrwkv2") is None,
    reason="RWKV CUDA contracts require CUDA and the locked FlashRWKV2 package",
)


def _config() -> RwkvConfig:
    return RwkvConfig(
        vocab_size=256,
        context_length=32,
        hidden_size=128,
        num_hidden_layers=2,
        intermediate_size=512,
        head_size=64,
        decay_low_rank_dim=32,
        a_low_rank_dim=32,
        v_low_rank_dim=32,
        gate_low_rank_dim=32,
    )


def _assets(path: Path, config: RwkvConfig | None = None) -> Path:
    (config or _config()).save_pretrained(path)
    return path


def _assert_model_close(
    direct: RwkvForCausalLM,
    adapted: RwkvForCausalLM,
    *,
    compare_gradients: bool,
) -> None:
    direct_parameters = dict(direct.named_parameters())
    adapted_parameters = dict(adapted.named_parameters())
    assert direct_parameters.keys() == adapted_parameters.keys()
    for name in direct_parameters:
        torch.testing.assert_close(
            direct_parameters[name], adapted_parameters[name], atol=2e-2, rtol=2e-2
        )
        if compare_gradients:
            direct_gradient = direct_parameters[name].grad
            adapted_gradient = adapted_parameters[name].grad
            assert (direct_gradient is None) == (adapted_gradient is None), name
            if direct_gradient is not None:
                torch.testing.assert_close(direct_gradient, adapted_gradient, atol=2e-2, rtol=2e-2)


def test_adapter_loss_gradients_and_optimizer_match_direct_hf(tmp_path: Path) -> None:
    torch.manual_seed(1234)
    config = _config()
    direct = RwkvForCausalLM(config).cuda().to(torch.bfloat16).train()
    adapter = RwkvModelAdapter.Config(hf_assets_path=str(_assets(tmp_path))).build()
    adapter.rwkv_model.load_state_dict(direct.state_dict())
    adapter = adapter.cuda().to(torch.bfloat16).train()
    tokens = torch.randint(0, config.vocab_size, (1, 16), device="cuda")
    labels = torch.randint(0, config.vocab_size, (1, 16), device="cuda")
    direct_logits = direct(input_ids=tokens, use_cache=False, return_dict=True).logits
    adapter_logits = adapter(tokens)
    torch.testing.assert_close(direct_logits, adapter_logits, atol=0, rtol=0)

    loss_fn = RwkvL2WrapLoss(RwkvL2WrapLoss.Config())
    global_valid_tokens = float(labels.numel())
    direct_loss, _ = loss_fn(direct_logits, labels, global_valid_tokens)
    adapter_loss, _ = loss_fn(adapter_logits, labels, global_valid_tokens)
    torch.testing.assert_close(direct_loss, adapter_loss, atol=0, rtol=0)
    direct_loss.backward()
    adapter_loss.backward()
    _assert_model_close(direct, adapter.rwkv_model, compare_gradients=True)

    direct_norm = torch.nn.utils.clip_grad_norm_(direct.parameters(), 1.0)
    adapter_norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
    torch.testing.assert_close(direct_norm, adapter_norm, atol=2e-2, rtol=2e-2)
    optimizer_config = RwkvOptimizersContainer.Config(
        implementation="for-loop", lr=1e-4, weight_decay=0.01
    )
    direct_optimizer = optimizer_config.build(model_parts=[direct])
    adapter_optimizer = optimizer_config.build(model_parts=[adapter])
    direct_optimizer.step()
    adapter_optimizer.step()
    _assert_model_close(direct, adapter.rwkv_model, compare_gradients=False)


def test_infctx_chunk_state_tail_and_detach_contract(tmp_path: Path) -> None:
    torch.manual_seed(4321)
    config = _config()
    adapter = RwkvModelAdapter.Config(
        hf_assets_path=str(_assets(tmp_path)), infctx=True, chunk_ctx=7
    ).build()
    adapter = adapter.cuda().to(torch.bfloat16).train()
    tokens = torch.randint(0, config.vocab_size, (1, 16), device="cuda")
    initial = RwkvTrainingState.zeros(config, 1, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        whole = adapter.forward_stateful(tokens, initial.clone())
        first = adapter.forward_stateful(tokens[:, :7], initial.clone())
        second = adapter.forward_stateful(tokens[:, 7:], first.training_state)
    assert whole.training_state is not None
    assert first.training_state is not None
    assert second.training_state is not None
    torch.testing.assert_close(
        whole.logits,
        torch.cat((first.logits, second.logits), dim=1),
        atol=2e-2,
        rtol=2e-2,
    )
    for actual, expected in zip(
        second.training_state.tensors(), whole.training_state.tensors(), strict=True
    ):
        torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
    assert second.training_state.wkv.dtype == torch.float32

    first_with_grad = adapter.forward_stateful(tokens[:, :7], initial.clone())
    assert first_with_grad.training_state is not None
    detached = first_with_grad.training_state.clone_detach()
    assert all(not tensor.requires_grad for tensor in detached.tensors())
    tail = adapter.forward_stateful(tokens[:, 7:], detached)
    tail.logits.float().mean().backward()
    gradient = adapter.rwkv_model.model.blocks[0].att.receptance.weight.grad
    assert gradient is not None and torch.isfinite(gradient).all()

    invalid = initial.clone()
    invalid.wkv = invalid.wkv.to(torch.bfloat16)
    with pytest.raises(TypeError, match=r"training_state\.wkv must have dtype"):
        adapter.forward_stateful(tokens[:, :1], invalid)


def test_canonical_artifact_adapter_forward_matches_direct_hf() -> None:
    artifact_value = os.environ.get("RWKV7_HF_ARTIFACT")
    if not artifact_value:
        pytest.skip("set RWKV7_HF_ARTIFACT for the canonical checkpoint smoke")
    assert artifact_value is not None
    artifact = Path(artifact_value)
    direct = (
        RwkvForCausalLM.from_pretrained(artifact, local_files_only=True, dtype=torch.bfloat16)
        .cuda()
        .train()
    )
    adapter_config = RwkvModelAdapter.Config(hf_assets_path=str(artifact))
    with torch.device("meta"):
        adapter = adapter_config.build()
    adapter.to_empty(device="cuda")
    adapter = adapter.to(torch.bfloat16).train()
    adapter.rwkv_model.load_state_dict(direct.state_dict())
    tokens = torch.arange(16, device="cuda", dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        direct_logits = direct(input_ids=tokens, use_cache=False, return_dict=True).logits
        adapter_logits = adapter(tokens)
    torch.testing.assert_close(direct_logits, adapter_logits, atol=0, rtol=0)
