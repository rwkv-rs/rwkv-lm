from __future__ import annotations

import json
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from rwkv_lm import standard_model
from rwkv_lm.standard_model import (
    StandardModelContractError,
    convert_legacy_rwkv7_checkpoint,
    create_standard_rwkv7_model,
    prepare_standard_rwkv7_for_fsdp2,
    save_standard_rwkv7_model,
    standard_rwkv7_blocks,
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
        self.output = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.output(hidden + self.w0)


class _FakeBlock(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.att = _FakeAttention(hidden_size)
        self.ffn = nn.Linear(hidden_size, hidden_size, bias=False)
        self.forward_calls = 0

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        self.forward_calls += 1
        return torch.tanh(self.ffn(self.att(hidden)))


class _FakeBody(nn.Module):
    def __init__(self, config: _FakeConfig) -> None:
        super().__init__()
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList(
            [_FakeBlock(config.hidden_size) for _ in range(config.num_hidden_layers)]
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.embeddings(input_ids)
        for block in self.blocks:
            hidden = block(hidden)
        return hidden


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
        logits = self.head(self.model(input_ids))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, self.config.vocab_size),
                labels[:, 1:].reshape(-1),
            )
        return SimpleNamespace(loss=loss, logits=logits, state=None)

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


def test_standard_model_owns_real_loss_gradients_and_optimizer_groups(
    standard_bindings,
) -> None:
    torch.manual_seed(7)
    model = create_standard_rwkv7_model(_args())
    input_ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])

    loss = standard_rwkv7_training_loss(model, input_ids, input_ids)
    loss.backward()

    assert torch.isfinite(loss)
    assert model.config.wkv_backend == "reference"
    assert len(standard_rwkv7_blocks(model)) == 2
    assert model.head.weight.grad is not None
    assert torch.count_nonzero(model.head.weight.grad) > 0
    assert all(block.ffn.weight.grad is not None for block in model.model.blocks)
    groups = standard_rwkv7_optimizer_groups(model, weight_decay=0.1)
    assert {group["my_lr_scale"] for group in groups} == {1.0, 2.0}
    assert {group["weight_decay"] for group in groups} == {0.0, 0.1}


def test_standard_model_save_reload_preserves_tensor_output(
    standard_bindings,
    tmp_path: Path,
) -> None:
    torch.manual_seed(11)
    model = create_standard_rwkv7_model(_args())
    input_ids = torch.tensor([[1, 2, 3, 4]])
    expected = model(input_ids=input_ids).logits.detach()
    artifact = save_standard_rwkv7_model(model, tmp_path / "standard-model")

    reloaded = create_standard_rwkv7_model(_args(), model_source=artifact)

    torch.testing.assert_close(
        reloaded(input_ids=input_ids).logits,
        expected,
        rtol=0,
        atol=0,
    )


def test_standard_model_applies_non_reentrant_checkpointing_to_only_blocks(
    standard_bindings,
) -> None:
    model = create_standard_rwkv7_model(_args())
    blocks = standard_rwkv7_blocks(model)
    prepare_standard_rwkv7_for_fsdp2(model, activation_checkpointing=True)
    input_ids = torch.tensor([[1, 2, 3, 4]])

    standard_rwkv7_training_loss(model, input_ids, input_ids).backward()

    assert all(block.forward_calls == 2 for block in blocks)
    policy = model._fsdp2_activation_checkpointing
    assert policy.enabled
    assert not policy.use_reentrant
    policy.require_rwkv_blocks(
        standard_rwkv7_blocks(model),
        enabled=True,
    )


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
    def missing(name: str):
        raise ImportError(f"missing {name}")

    monkeypatch.setattr(standard_model, "import_module", missing)

    with pytest.raises(
        StandardModelContractError,
        match="requires transformers-rwkv",
    ):
        create_standard_rwkv7_model(_args())
