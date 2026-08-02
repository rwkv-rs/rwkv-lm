import importlib
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")


def _import_model():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the RWKV LoRA integration test.")
    os.environ.setdefault("RWKV_MY_TESTING", "x070")
    os.environ.setdefault("RWKV_KERNEL", "")
    os.environ.setdefault("RWKV_HEAD_SIZE", "64")
    os.environ.setdefault("RWKV_HEAD_L2WRAP_CE_CHUNK", "0")
    os.environ.setdefault("RWKV_FLOAT_MODE", "bf16")
    os.environ.setdefault("RWKV_JIT_ON", "1")
    os.environ.setdefault("RWKV_TRAIN_TYPE", "standard")
    return importlib.import_module("rwkv_lm.model")


def _tiny_lora_rwkv(model_module):
    args = SimpleNamespace(
        chunk_ctx=0,
        ctx_len=16,
        dim_att=64,
        dim_ffn=224,
        grad_cp=0,
        head_size=64,
        lora_alpha=4.0,
        lora_dropout=0.0,
        lora_rank=2,
        lora_target_modules=("time_mix.output", "channel_mix.value"),
        my_testing="x070",
        n_embd=64,
        n_layer=2,
        strategy="fsdp2",
        train_type="standard",
        vocab_size=64,
        weight_decay=0.0,
    )
    return model_module.RWKV(args).to(device="cuda", dtype=torch.bfloat16)


def _optimizer(model):
    groups = model.build_optimizer_groups(is_global_zero=False)
    return torch.optim.AdamW(groups, lr=0.03)


def _step(model, optimizer, input_ids, targets):
    optimizer.zero_grad(set_to_none=True)
    logits = model(input_ids)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(),
        targets.reshape(-1),
    )
    loss.backward()
    optimizer.step()
    return loss.detach()


def test_rwkv_lora_trains_only_selected_projections_and_reloads_logits(
    tmp_path: Path,
):
    model_module = _import_model()
    torch.manual_seed(20260801)
    model = _tiny_lora_rwkv(model_module)
    base_state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.base_state_dict().items()
    }
    trainable = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    assert trainable == set(model.adapter_state_dict())
    assert len(trainable) == 2 * 2 * 2
    assert all(
        ".att.output_lora." in name or ".ffn.value_lora." in name for name in trainable
    )
    assert not model.blocks[0].att.output.weight.requires_grad
    assert not model.blocks[0].ffn.value.weight.requires_grad

    input_ids = torch.randint(0, 64, (1, 16), device="cuda")
    targets = torch.randint(0, 64, (1, 16), device="cuda")
    optimizer = _optimizer(model)
    _step(model, optimizer, input_ids, targets)
    _step(model, optimizer, input_ids, targets)
    assert model.blocks[0].att.output.weight.grad is None
    assert model.blocks[0].ffn.value.weight.grad is None
    for name, parameter in model.named_parameters():
        if name in trainable:
            assert parameter.grad is not None
            assert torch.count_nonzero(parameter.grad) > 0

    model.eval()
    expected_logits = model(input_ids).detach()
    artifact = model.save_lora_adapter(tmp_path / "adapter.pt")

    torch.manual_seed(7)
    reloaded = _tiny_lora_rwkv(model_module)
    reloaded.load_base_state_dict(base_state)
    reloaded.load_lora_adapter(artifact)
    reloaded.eval()
    torch.testing.assert_close(
        reloaded(input_ids).float(),
        expected_logits.float(),
        rtol=0,
        atol=0,
    )
