import pytest
import torch
from torchtitan.components.lora import LoRAConverter

from rwkv_lm.models.rwkv7 import model_registry
from rwkv_lm.models.rwkv7.state_dict_adapter import Rwkv7StateDictAdapter


def test_transformers_state_dict_names_round_trip() -> None:
    adapter = Rwkv7StateDictAdapter(model_registry("debugmodel").model, None)
    native = {
        "tok_embeddings.weight": torch.tensor([1.0]),
        "layers.0.att.receptance.weight": torch.tensor([2.0]),
        "norm.weight": torch.tensor([3.0]),
        "lm_head.weight": torch.tensor([4.0]),
    }

    hf = adapter.to_hf(native)

    assert set(hf) == {
        "model.embeddings.weight",
        "model.blocks.0.att.receptance.weight",
        "model.ln_out.weight",
        "head.weight",
    }
    assert adapter.from_hf(hf) == native


def test_full_69_key_model_state_round_trips_without_loss() -> None:
    model = model_registry("debugmodel").model.build()
    model.init_states()
    adapter = Rwkv7StateDictAdapter(model_registry("debugmodel").model, None)
    native = model.state_dict()

    assert len(native) == 69
    restored = adapter.from_hf(adapter.to_hf(native))
    assert restored.keys() == native.keys()
    assert all(torch.equal(restored[name], native[name]) for name in native)


def test_unknown_state_dict_name_fails_closed() -> None:
    adapter = Rwkv7StateDictAdapter(model_registry("debugmodel").model, None)

    with pytest.raises(KeyError, match="unrecognized RWKV-7"):
        adapter.to_hf({"native_fallback.weight": torch.tensor([1.0])})


def test_lora_state_exports_as_merged_transformers_weight() -> None:
    spec = model_registry(
        "debugmodel",
        converters=[
            LoRAConverter.Config(
                rank=2,
                alpha=4.0,
                target_modules=["receptance"],
            )
        ],
    )
    model = spec.model.build()
    model.init_states()
    adapter = Rwkv7StateDictAdapter(spec.model, None)
    projection = model.layers["0"].att.receptance
    with torch.no_grad():
        projection.lora_a.weight.fill_(0.25)
        projection.lora_b.weight.fill_(0.5)
    expected = projection.weight + 2.0 * (
        projection.lora_b.weight @ projection.lora_a.weight
    )

    hf_state = adapter.to_hf(model.state_dict())

    assert torch.equal(
        hf_state["model.blocks.0.att.receptance.weight"],
        expected,
    )
    assert not any("lora_" in name for name in hf_state)


def test_lora_adapter_only_state_loads_and_preserves_inference() -> None:
    spec = model_registry(
        "debugmodel",
        converters=[
            LoRAConverter.Config(
                rank=2,
                alpha=4.0,
                target_modules=["receptance"],
            )
        ],
    )
    source = spec.model.build()
    source.init_states()
    target = spec.model.build()
    target.init_states()
    adapter = Rwkv7StateDictAdapter(spec.model, None)
    with torch.no_grad():
        source.layers["0"].att.receptance.lora_a.weight.fill_(0.25)
        source.layers["0"].att.receptance.lora_b.weight.fill_(0.5)
        target.load_state_dict(source.state_dict(), strict=True)
        target.layers["0"].att.receptance.lora_a.weight.zero_()
        target.layers["0"].att.receptance.lora_b.weight.zero_()
    adapter_state = adapter.adapter_state_dict(source.state_dict())
    adapter.load_adapter_state_dict(target, adapter_state)
    inputs = torch.randn(2, 3, spec.model.hidden_size)

    assert adapter_state
    assert all(".lora_" in name for name in adapter_state)
    assert torch.equal(
        source.layers["0"].att.receptance(inputs),
        target.layers["0"].att.receptance(inputs),
    )
