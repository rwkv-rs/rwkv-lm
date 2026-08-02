from __future__ import annotations

import json
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from rwkv_lm import standard_model
from rwkv_lm.checkpoint import BackendIdentity
from rwkv_lm.checkpoint_runner import EpochCheckpointRunnerAdapter
from rwkv_lm.infctx import InfctxBoundary
from rwkv_lm.peft import LoraConfig, lora_base_state_dict, lora_parameter_names
from rwkv_lm.standard_model import (
    StandardModelContractError,
    configure_standard_rwkv7_peft,
    convert_legacy_rwkv7_checkpoint,
    create_standard_rwkv7_model,
    prepare_standard_rwkv7_for_fsdp2,
    save_standard_rwkv7_lora_adapter,
    save_standard_rwkv7_merged_model,
    save_standard_rwkv7_model,
    standard_rwkv7_blocks,
    standard_rwkv7_infctx_forward,
    standard_rwkv7_optimizer_groups,
    standard_rwkv7_training_loss,
)


class _FakeConfig:
    model_type = "rwkv7"

    def __init__(self, **kwargs) -> None:
        for name, value in kwargs.items():
            setattr(self, name, value)


class _FakeAttention(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.w0 = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.receptance = nn.Linear(hidden_size, hidden_size, bias=False)
        self.key = nn.Linear(hidden_size, hidden_size, bias=False)
        self.value = nn.Linear(hidden_size, hidden_size, bias=False)
        self.output = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        mixed = (
            self.receptance(hidden)
            + self.key(hidden)
            + self.value(hidden)
            + self.w0
        )
        return self.output(torch.tanh(mixed))


class _FakeChannelMix(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.key = nn.Linear(hidden_size, hidden_size * 2, bias=False)
        self.value = nn.Linear(hidden_size * 2, hidden_size, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.value(F.relu(self.key(hidden)).square())


class _FakeBlock(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.att = _FakeAttention(hidden_size)
        self.ffn = _FakeChannelMix(hidden_size)
        self.forward_calls = 0

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        self.forward_calls += 1
        return torch.tanh(self.ffn(self.att(hidden)))


class _FakeBody(nn.Module):
    def __init__(self, config: _FakeConfig) -> None:
        super().__init__()
        self.config = config
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList(
            [_FakeBlock(config.hidden_size) for _ in range(config.num_hidden_layers)]
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        state: tuple[torch.Tensor, ...] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...] | None]:
        hidden = self.embeddings(input_ids)
        batch_size = input_ids.shape[0]
        if state is None:
            attention_state = hidden.new_zeros(
                len(self.blocks), batch_size, hidden.shape[-1]
            )
            wkv_state = torch.zeros(
                len(self.blocks),
                batch_size,
                self.config.num_attention_heads,
                self.config.head_size,
                self.config.head_size,
                device=hidden.device,
                dtype=torch.float32,
            )
            ffn_state = torch.zeros_like(attention_state)
        else:
            attention_state, wkv_state, ffn_state = state
        next_attention = []
        next_ffn = []
        for layer_id, block in enumerate(self.blocks):
            block_inputs = hidden
            hidden = block(
                block_inputs.cumsum(dim=1)
                + attention_state[layer_id].unsqueeze(1)
            )
            next_attention.append(
                attention_state[layer_id] + block_inputs.sum(dim=1)
            )
            next_ffn.append(ffn_state[layer_id] + hidden.sum(dim=1) * 0)
        next_state = None
        if use_cache:
            next_state = (
                torch.stack(next_attention),
                wkv_state + hidden.sum() * 0,
                torch.stack(next_ffn),
            )
        return hidden, next_state


class _FakeCausalLM(nn.Module):
    base_model_prefix = "model"
    config_class = _FakeConfig

    def __init__(self, config: _FakeConfig) -> None:
        super().__init__()
        self.config = config
        self.model = _FakeBody(config)
        self.head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        **_kwargs,
    ) -> SimpleNamespace:
        hidden, next_state = self.model(
            input_ids,
            state=_kwargs.get("state"),
            use_cache=bool(_kwargs.get("use_cache", False)),
        )
        logits = self.head(hidden)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, self.config.vocab_size),
                labels[:, 1:].reshape(-1),
            )
        return SimpleNamespace(loss=loss, logits=logits, state=next_state)

    def save_pretrained(
        self,
        destination: str,
        *,
        safe_serialization: bool,
    ) -> None:
        assert safe_serialization
        path = Path(destination)
        path.mkdir(parents=True)
        config = {
            name: value
            for name, value in vars(self.config).items()
            if isinstance(value, (bool, int, float, str)) or value is None
        }
        (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
        torch.save(self.state_dict(), path / "model.safetensors")

    @classmethod
    def from_pretrained(cls, source: str) -> _FakeCausalLM:
        path = Path(source)
        config = _FakeConfig(
            **json.loads((path / "config.json").read_text(encoding="utf-8"))
        )
        model = cls(config)
        model.load_state_dict(
            torch.load(path / "model.safetensors", weights_only=True)
        )
        return model


def _args(**overrides) -> SimpleNamespace:
    values = {
        "ctx_len": 8,
        "dim_ffn": 8,
        "head_size": 2,
        "n_embd": 4,
        "n_layer": 2,
        "vocab_size": 11,
        "wkv_backend": "reference",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.fixture
def standard_bindings(monkeypatch):
    config_module = ModuleType("fake_configuration_rwkv7")
    config_module.Rwkv7Config = _FakeConfig
    model_module = ModuleType("fake_modeling_rwkv7")
    model_module.Rwkv7ForCausalLM = _FakeCausalLM
    converter_module = ModuleType("fake_convert_rwkv7_checkpoint_to_hf")

    def convert(source: str, destination: str, **kwargs):
        raw = torch.load(source, map_location="cpu", weights_only=True)
        path = Path(destination)
        path.mkdir(parents=True)
        (path / "config.json").write_text(
            json.dumps({"model_type": "rwkv7", "wkv_backend": kwargs["wkv_backend"]}),
            encoding="utf-8",
        )
        torch.save(raw, path / "model.safetensors")
        return {
            "tensor_count": len(raw),
            "wkv_backend": kwargs["wkv_backend"],
        }

    converter_module.convert_rwkv7_checkpoint_to_hf_format = convert
    modules = {
        standard_model._CONFIG_MODULE: config_module,
        standard_model._MODEL_MODULE: model_module,
        standard_model._CONVERTER_MODULE: converter_module,
    }
    monkeypatch.setattr(standard_model, "import_module", modules.__getitem__)
    return modules


def test_standard_model_owns_real_loss_gradients_and_optimizer_groups(
    standard_bindings,
) -> None:
    torch.manual_seed(7)
    model = create_standard_rwkv7_model(_args())
    input_ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
    labels = torch.tensor([[2, 3, 4, 5], [3, 2, 1, 0]])

    logits = model(input_ids=input_ids).logits
    loss = standard_rwkv7_training_loss(model, input_ids, labels)
    expected_loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
    )
    loss.backward()

    assert torch.isfinite(loss)
    torch.testing.assert_close(loss, expected_loss, rtol=0, atol=0)
    assert model.config.wkv_backend == "reference"
    assert len(standard_rwkv7_blocks(model)) == 2
    assert model.head.weight.grad is not None
    assert torch.count_nonzero(model.head.weight.grad) > 0
    assert all(block.ffn.key.weight.grad is not None for block in model.model.blocks)
    groups = standard_rwkv7_optimizer_groups(model, weight_decay=0.1)
    assert {group["my_lr_scale"] for group in groups} == {1.0, 2.0}
    assert {group["weight_decay"] for group in groups} == {0.0, 0.1}


def test_standard_model_save_reload_preserves_tensor_output(
    standard_bindings,
    tmp_path: Path,
) -> None:
    torch.manual_seed(11)
    model = create_standard_rwkv7_model(_args())
    model.config.pad_token_id = None
    input_ids = torch.tensor([[1, 2, 3, 4]])
    expected = model(input_ids=input_ids).logits.detach()
    artifact = save_standard_rwkv7_model(model, tmp_path / "standard-model")

    reloaded = create_standard_rwkv7_model(_args(), model_source=artifact)

    assert reloaded.config.pad_token_id is None
    torch.testing.assert_close(
        reloaded(input_ids=input_ids).logits,
        expected,
        rtol=0,
        atol=0,
    )


def test_standard_model_applies_non_reentrant_checkpointing_to_only_blocks(
    standard_bindings,
) -> None:
    torch.manual_seed(19)
    model = create_standard_rwkv7_model(_args())
    reference = create_standard_rwkv7_model(_args())
    reference.load_state_dict(model.state_dict())
    blocks = standard_rwkv7_blocks(model)
    state_names = set(model.state_dict())
    prepare_standard_rwkv7_for_fsdp2(model, activation_checkpointing=True)
    input_ids = torch.tensor([[1, 2, 3, 4]])

    loss = standard_rwkv7_training_loss(model, input_ids, input_ids)
    reference_loss = standard_rwkv7_training_loss(reference, input_ids, input_ids)
    loss.backward()
    reference_loss.backward()

    assert set(model.state_dict()) == state_names
    torch.testing.assert_close(loss, reference_loss, rtol=0, atol=0)
    assert all(block.forward_calls == 2 for block in blocks)
    reference_parameters = dict(reference.named_parameters())
    for name, parameter in model.named_parameters():
        reference_name = name.replace("._checkpoint_wrapped_module", "")
        reference_parameter = reference_parameters[reference_name]
        torch.testing.assert_close(
            parameter.grad,
            reference_parameter.grad,
            rtol=0,
            atol=0,
        )
    policy = model._fsdp2_activation_checkpointing
    assert policy.enabled
    assert not policy.use_reentrant
    policy.require_rwkv_blocks(
        standard_rwkv7_blocks(model),
        enabled=True,
    )


def test_standard_peft_train_save_load_merge_and_inference(
    standard_bindings,
    tmp_path: Path,
) -> None:
    torch.manual_seed(23)
    args = _args()
    model = create_standard_rwkv7_model(args)
    base_dir = save_standard_rwkv7_model(model, tmp_path / "base")
    config = LoraConfig(
        rank=2,
        alpha=4.0,
        dropout=0.0,
        target_modules=("time_mix.output", "channel_mix.key"),
    )
    configure_standard_rwkv7_peft(model, config)
    trainable = tuple(
        sorted(name for name, parameter in model.named_parameters() if parameter.requires_grad)
    )
    assert trainable == lora_parameter_names(model)
    assert trainable
    assert all(name.endswith((".lora_A", ".lora_B")) for name in trainable)
    base_before = {
        name: tensor.detach().clone()
        for name, tensor in lora_base_state_dict(model).items()
    }
    optimizer = torch.optim.AdamW(
        standard_rwkv7_optimizer_groups(model, weight_decay=0.0),
        lr=0.05,
    )
    input_ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
    labels = torch.tensor([[2, 3, 4, 5], [3, 2, 1, 0]])
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss = standard_rwkv7_training_loss(model, input_ids, labels)
        loss.backward()
        optimizer.step()
    assert all(parameter.grad is None for name, parameter in model.named_parameters() if name not in trainable)
    for name, tensor in lora_base_state_dict(model).items():
        torch.testing.assert_close(tensor, base_before[name], rtol=0, atol=0)

    model.eval()
    expected = model(input_ids=input_ids).logits.detach()
    adapter = save_standard_rwkv7_lora_adapter(
        model,
        tmp_path / "adapter.pt",
    )
    reloaded = create_standard_rwkv7_model(args, model_source=base_dir)
    configure_standard_rwkv7_peft(reloaded, config, adapter_path=adapter)
    reloaded.eval()
    torch.testing.assert_close(
        reloaded(input_ids=input_ids).logits,
        expected,
        rtol=0,
        atol=0,
    )

    merged_dir = save_standard_rwkv7_merged_model(
        model,
        tmp_path / "merged",
    )
    merged = create_standard_rwkv7_model(args, model_source=merged_dir)
    merged.eval()
    assert lora_parameter_names(merged) == ()
    torch.testing.assert_close(
        merged(input_ids=input_ids).logits,
        expected,
        rtol=1e-6,
        atol=1e-7,
    )


def test_standard_peft_checkpoint_resume_restores_adapter_and_optimizer(
    standard_bindings,
    tmp_path: Path,
) -> None:
    torch.manual_seed(29)
    args = _args()
    config = LoraConfig(
        rank=2,
        alpha=4.0,
        target_modules=("time_mix.key",),
    )
    model = create_standard_rwkv7_model(args)
    configure_standard_rwkv7_peft(model, config)
    optimizer = torch.optim.AdamW(
        standard_rwkv7_optimizer_groups(model, weight_decay=0.0),
        lr=0.03,
    )
    input_ids = torch.tensor([[1, 2, 3, 4]])
    labels = torch.tensor([[2, 3, 4, 5]])

    def step(candidate, candidate_optimizer) -> None:
        candidate_optimizer.zero_grad(set_to_none=True)
        loss = standard_rwkv7_training_loss(candidate, input_ids, labels)
        loss.backward()
        candidate_optimizer.step()

    step(model, optimizer)
    adapter = EpochCheckpointRunnerAdapter(
        backend=BackendIdentity(
            name="pytorch",
            version=torch.__version__,
            strategy="single_process",
            world_size=1,
            state_dict_type="full",
        ),
        training_config={"model": "standard-rwkv7-lora", **config.to_dict()},
        samples_per_epoch=1,
    )
    checkpoint = tmp_path / "epoch-00000001"
    adapter.save(
        checkpoint,
        model=model,
        optimizer=optimizer,
        global_step=1,
        next_epoch=1,
    )
    resumed = create_standard_rwkv7_model(args)
    configure_standard_rwkv7_peft(resumed, config)
    resumed_optimizer = torch.optim.AdamW(
        standard_rwkv7_optimizer_groups(resumed, weight_decay=0.0),
        lr=0.03,
    )
    progress = adapter.restore(
        checkpoint,
        model=resumed,
        optimizer=resumed_optimizer,
    )
    assert (progress.global_step, progress.epoch) == (1, 1)
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(resumed.state_dict()[name], tensor, rtol=0, atol=0)

    step(model, optimizer)
    step(resumed, resumed_optimizer)
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(resumed.state_dict()[name], tensor, rtol=0, atol=0)


def test_standard_peft_artifacts_keep_names_after_activation_wrapping(
    standard_bindings,
    tmp_path: Path,
) -> None:
    torch.manual_seed(30)
    args = _args()
    model = create_standard_rwkv7_model(args)
    base_dir = save_standard_rwkv7_model(model, tmp_path / "base")
    config = LoraConfig(
        rank=2,
        alpha=4.0,
        target_modules=("time_mix.value",),
    )
    configure_standard_rwkv7_peft(model, config)
    canonical_names = lora_parameter_names(model)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith(".lora_B"):
                parameter.fill_(0.125)
    prepare_standard_rwkv7_for_fsdp2(model, activation_checkpointing=True)
    assert lora_parameter_names(model) == canonical_names
    input_ids = torch.tensor([[1, 2, 3, 4]])
    model.eval()
    expected = model(input_ids=input_ids).logits.detach()

    adapter = save_standard_rwkv7_lora_adapter(
        model,
        tmp_path / "wrapped-adapter.pt",
    )
    reloaded = create_standard_rwkv7_model(args, model_source=base_dir)
    configure_standard_rwkv7_peft(reloaded, config, adapter_path=adapter)
    reloaded.eval()
    torch.testing.assert_close(
        reloaded(input_ids=input_ids).logits,
        expected,
        rtol=0,
        atol=0,
    )

    merged_dir = save_standard_rwkv7_merged_model(
        model,
        tmp_path / "wrapped-merged",
    )
    merged = create_standard_rwkv7_model(args, model_source=merged_dir)
    merged.eval()
    torch.testing.assert_close(
        merged(input_ids=input_ids).logits,
        expected,
        rtol=1e-6,
        atol=1e-7,
    )


def test_standard_infctx_matches_full_and_detached_response_reference(
    standard_bindings,
) -> None:
    torch.manual_seed(31)
    args = _args(ctx_len=64, vocab_size=64)
    model = create_standard_rwkv7_model(args)
    reference = create_standard_rwkv7_model(args)
    reference.load_state_dict(model.state_dict())
    input_ids = torch.arange(1, 33).reshape(1, 32)

    full = model(input_ids=input_ids, use_cache=True).logits.detach()
    prepare_standard_rwkv7_for_fsdp2(model, activation_checkpointing=True)
    chunked = standard_rwkv7_infctx_forward(
        model,
        input_ids,
        chunk_ctx=16,
        ctx_len=64,
        boundary=InfctxBoundary.RESET,
    )
    torch.testing.assert_close(chunked.output, full, rtol=1e-6, atol=1e-7)
    assert chunked.state.tokens_seen == 32
    assert not chunked.state.shift_states.requires_grad
    assert not chunked.state.wkv_states.requires_grad

    first = standard_rwkv7_infctx_forward(
        model,
        input_ids[:, :16],
        chunk_ctx=16,
        ctx_len=64,
        boundary=InfctxBoundary.RESET,
    )
    continued = standard_rwkv7_infctx_forward(
        model,
        input_ids[:, 16:],
        chunk_ctx=16,
        ctx_len=64,
        boundary=InfctxBoundary.CONTINUE,
        state=first.state,
    )
    reset = standard_rwkv7_infctx_forward(
        model,
        input_ids[:, 16:],
        chunk_ctx=16,
        ctx_len=64,
        boundary=InfctxBoundary.RESET,
    )
    torch.testing.assert_close(
        torch.cat((first.output, continued.output), dim=1),
        full,
        rtol=1e-6,
        atol=1e-7,
    )
    assert continued.state.tokens_seen == 32
    assert reset.state.tokens_seen == 16
    assert not torch.equal(continued.output, reset.output)

    chunk_loss = chunked.output[:, 16:].square().mean()
    prefix = reference(input_ids=input_ids[:, :16], use_cache=True)
    detached_provider_state = tuple(value.detach() for value in prefix.state)
    response = reference(
        input_ids=input_ids[:, 16:],
        state=detached_provider_state,
        use_cache=True,
    )
    reference_loss = response.logits.square().mean()
    torch.testing.assert_close(
        chunked.output[:, 16:],
        response.logits,
        rtol=1e-6,
        atol=1e-7,
    )
    torch.testing.assert_close(chunk_loss, reference_loss, rtol=1e-6, atol=1e-7)
    chunk_loss.backward()
    reference_loss.backward()
    for parameter, reference_parameter in zip(
        model.parameters(),
        reference.parameters(),
        strict=True,
    ):
        if reference_parameter.grad is None:
            assert parameter.grad is None
        else:
            torch.testing.assert_close(
                parameter.grad,
                reference_parameter.grad,
                rtol=1e-5,
                atol=1e-7,
            )
    embedding_grad = model.model.embeddings.weight.grad
    assert embedding_grad is not None
    assert torch.count_nonzero(embedding_grad[1:17]) == 0
    assert torch.count_nonzero(embedding_grad[17:33]) > 0

    infctx_loss = standard_rwkv7_training_loss(
        model,
        input_ids,
        input_ids.roll(-1, dims=1),
        train_type="infctx",
        chunk_ctx=16,
        ctx_len=64,
    )
    assert infctx_loss.ndim == 0


def test_legacy_converter_delegates_real_tensor_artifact(
    standard_bindings,
    tmp_path: Path,
) -> None:
    source = tmp_path / "legacy.pth"
    raw = {"emb.weight": torch.arange(12).reshape(3, 4)}
    torch.save(raw, source)

    result = convert_legacy_rwkv7_checkpoint(
        source,
        tmp_path / "standard",
        wkv_backend="reference",
    )

    assert result == {"tensor_count": 1, "wkv_backend": "reference"}
    converted = torch.load(
        tmp_path / "standard" / "model.safetensors",
        weights_only=True,
    )
    torch.testing.assert_close(converted["emb.weight"], raw["emb.weight"])


def test_model_loader_rejects_legacy_pth_before_import(
    monkeypatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "legacy.pth"
    torch.save({"weight": torch.ones(1)}, source)
    monkeypatch.setattr(
        standard_model,
        "load_standard_rwkv7_bindings",
        lambda **_kwargs: pytest.fail("legacy load must fail before model import"),
    )

    with pytest.raises(StandardModelContractError, match="convert it"):
        create_standard_rwkv7_model(_args(), model_source=source)


def test_missing_standard_dependency_fails_closed(monkeypatch) -> None:
    attempted = []

    def missing(name: str):
        attempted.append(name)
        raise ImportError(f"missing {name}")

    monkeypatch.setattr(standard_model, "import_module", missing)

    with pytest.raises(ImportError) as error:
        create_standard_rwkv7_model(_args())

    message = str(error.value)
    assert standard_model._CONFIG_MODULE in message
    assert "transformers@eb8248eb9083288e7769518077a1be9c0f7cf7b8" in message
    assert "flash-linear-attention@1bc262c8c81241e1d339419a31f0aadffa20c210" in message
    assert "flash-rwkv@866aafd2eed146b0eda1ce03444009ae030f89e3" in message
    assert attempted == ["transformers.models.rwkv7.configuration_rwkv7"]


def test_standard_flash_backend_failure_propagates_without_fallback(
    standard_bindings,
) -> None:
    class _FailClosedCausalLM(_FakeCausalLM):
        def forward(self, **_kwargs):
            raise RuntimeError(
                "Explicit FlashRWKV request failed closed: FLA public "
                "chunk_rwkv7 did not select FlashRWKV; fallback is disabled"
            )

    standard_bindings[
        standard_model._MODEL_MODULE
    ].Rwkv7ForCausalLM = _FailClosedCausalLM
    model = create_standard_rwkv7_model(_args(wkv_backend="flash_rwkv"))
    input_ids = torch.tensor([[1, 2, 3, 4]])

    with pytest.raises(
        RuntimeError,
        match="FLA public chunk_rwkv7 did not select FlashRWKV; fallback is disabled",
    ):
        standard_rwkv7_training_loss(model, input_ids, input_ids)
