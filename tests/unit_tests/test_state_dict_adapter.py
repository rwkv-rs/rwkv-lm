import copy
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch
from torchtitan.components.lora import LoRAConverter

from rwkv_lm.artifacts.rwkv7 import AdapterCheckpointError
from rwkv_lm.models.rwkv7 import model_registry
from rwkv_lm.models.rwkv7.state_dict_adapter import Rwkv7StateDictAdapter

_SOURCE_REVISION = "1" * 40
_OTHER_REVISION = "2" * 40


def _lora_spec():
    return model_registry(
        "debugmodel",
        converters=[
            LoRAConverter.Config(
                rank=2,
                alpha=4.0,
                target_modules=["receptance"],
            )
        ],
    )


def _initialized_lora_model():
    model = _lora_spec().model.build()
    model.init_states()
    return model


def _state_snapshot(model) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().clone() for name, tensor in model.state_dict().items()
    }


def _assert_state_unchanged(
    model,
    snapshot: dict[str, torch.Tensor],
) -> None:
    assert model.state_dict().keys() == snapshot.keys()
    assert all(
        torch.equal(tensor, snapshot[name])
        for name, tensor in model.state_dict().items()
    )


def _source_and_exact_base_target():
    source = _initialized_lora_model()
    target = _initialized_lora_model()
    with torch.no_grad():
        source.layers["0"].att.receptance.lora_a.weight.fill_(0.25)
        source.layers["0"].att.receptance.lora_b.weight.fill_(0.5)
        source.layers["1"].att.receptance.lora_a.weight.fill_(0.75)
        source.layers["1"].att.receptance.lora_b.weight.fill_(1.0)
        target.load_state_dict(source.state_dict(), strict=True)
        for name, parameter in target.named_parameters():
            if ".lora_" in name:
                parameter.zero_()
    return source, target


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
    spec = _lora_spec()
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


def test_adapter_checkpoint_binds_full_base_and_preserves_inference(
    tmp_path: Path,
    rwkv7_artifact_factory,
) -> None:
    artifact_path, model_identity = rwkv7_artifact_factory()
    spec = _lora_spec()
    source, target = _source_and_exact_base_target()
    adapter = Rwkv7StateDictAdapter(spec.model, str(artifact_path))

    checkpoint = adapter.adapter_checkpoint(source)
    checkpoint_path = tmp_path / "adapter.pt"
    torch.save(checkpoint, checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    adapter.load_adapter_checkpoint(target, checkpoint)
    inputs = torch.randn(2, 3, spec.model.hidden_size)

    assert checkpoint["base"]["model_identity"] == model_identity
    assert checkpoint["base"]["source_revision"] == _SOURCE_REVISION
    assert len(checkpoint["base"]["state_inventory"]) == 69
    assert list(checkpoint["state_dict"]) == [
        "layers.0.att.receptance.lora_a.weight",
        "layers.0.att.receptance.lora_b.weight",
        "layers.1.att.receptance.lora_a.weight",
        "layers.1.att.receptance.lora_b.weight",
    ]
    assert all(
        torch.equal(source.state_dict()[name], target.state_dict()[name])
        for name in checkpoint["state_dict"]
    )
    assert torch.equal(
        source.layers["0"].att.receptance(inputs),
        target.layers["0"].att.receptance(inputs),
    )


@pytest.mark.parametrize(
    "corrupt",
    [
        pytest.param(
            lambda checkpoint: checkpoint["base"].__setitem__(
                "model_identity", "4" * 64
            ),
            id="wrong-base-identity",
        ),
        pytest.param(
            lambda checkpoint: checkpoint["base"].__setitem__(
                "source_revision", _OTHER_REVISION
            ),
            id="wrong-source-revision",
        ),
        pytest.param(
            lambda checkpoint: checkpoint["state_dict"].__setitem__(
                next(iter(checkpoint["state_dict"])),
                next(iter(checkpoint["state_dict"].values())).double(),
            ),
            id="float64",
        ),
        pytest.param(
            lambda checkpoint: checkpoint["state_dict"].__setitem__(
                next(iter(checkpoint["state_dict"])),
                next(iter(checkpoint["state_dict"].values())).to(torch.int32),
            ),
            id="integer-dtype",
        ),
        pytest.param(
            lambda checkpoint: checkpoint["adapter"].__setitem__(
                "storage_device", "cuda"
            ),
            id="wrong-storage-device-policy",
        ),
        pytest.param(
            lambda checkpoint: checkpoint["state_dict"].pop(
                next(iter(checkpoint["state_dict"]))
            ),
            id="missing-key",
        ),
        pytest.param(
            lambda checkpoint: checkpoint["state_dict"].__setitem__(
                "layers.0.att.receptance.lora_extra.weight",
                torch.zeros(1),
            ),
            id="extra-key",
        ),
        pytest.param(
            lambda checkpoint: checkpoint["state_dict"].__setitem__(
                "layers.1.att.receptance.lora_b.weight",
                checkpoint["state_dict"]["layers.1.att.receptance.lora_b.weight"][:-1],
            ),
            id="lexical-last-wrong-shape",
        ),
    ],
)
def test_invalid_adapter_checkpoint_fails_before_any_model_mutation(
    rwkv7_artifact_factory,
    corrupt: Callable[[dict[str, Any]], Any],
) -> None:
    artifact_path, _model_identity = rwkv7_artifact_factory()
    source, target = _source_and_exact_base_target()
    adapter = Rwkv7StateDictAdapter(_lora_spec().model, str(artifact_path))
    checkpoint = copy.deepcopy(adapter.adapter_checkpoint(source))
    corrupt(checkpoint)
    before = _state_snapshot(target)

    with pytest.raises(AdapterCheckpointError):
        adapter.load_adapter_checkpoint(target, checkpoint)

    _assert_state_unchanged(target, before)


def test_adapter_checkpoint_rejects_different_base_without_mutation(
    rwkv7_artifact_factory,
) -> None:
    artifact_path, _model_identity = rwkv7_artifact_factory()
    source, _exact_target = _source_and_exact_base_target()
    different_base = _initialized_lora_model()
    adapter = Rwkv7StateDictAdapter(_lora_spec().model, str(artifact_path))
    checkpoint = adapter.adapter_checkpoint(source)
    before = _state_snapshot(different_base)

    with pytest.raises(AdapterCheckpointError, match="base state digest"):
        adapter.load_adapter_checkpoint(different_base, checkpoint)

    _assert_state_unchanged(different_base, before)
