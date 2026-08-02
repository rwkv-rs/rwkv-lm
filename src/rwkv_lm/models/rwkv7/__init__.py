"""TorchTitan model registry for RWKV-7."""

from __future__ import annotations

from torchtitan.models.utils import validate_converter_order
from torchtitan.protocols.model import ModelConfigConverter
from torchtitan.protocols.model_spec import ModelSpec

from .model import Rwkv7Model, build_rwkv7_config
from .parallelize import parallelize_rwkv7
from .state_dict_adapter import Rwkv7StateDictAdapter


def _debug_model() -> Rwkv7Model.Config:
    return build_rwkv7_config(
        vocab_size=1_024,
        context_length=128,
        hidden_size=128,
        num_hidden_layers=2,
        intermediate_size=448,
        head_size=64,
    )


def _g1h_1_5b() -> Rwkv7Model.Config:
    return build_rwkv7_config(
        vocab_size=65_536,
        context_length=10_240,
        hidden_size=2_048,
        num_hidden_layers=24,
        intermediate_size=7_168,
        head_size=64,
    )


rwkv7_configs = {
    "debugmodel": _debug_model,
    "g1h-1.5b": _g1h_1_5b,
}


def model_registry(
    flavor: str,
    converters: list[ModelConfigConverter.Config] | None = None,
) -> ModelSpec:
    try:
        model_config = rwkv7_configs[flavor]()
    except KeyError as error:
        raise ValueError(f"unknown RWKV-7 model flavor: {flavor}") from error
    if converters is not None:
        validate_converter_order(converters)
        for converter_config in converters:
            model_config = converter_config.build().convert(model_config)
    return ModelSpec(
        name="rwkv7",
        flavor=flavor,
        model=model_config,
        parallelize_fn=parallelize_rwkv7,
        pipelining_fn=None,
        post_optimizer_build_fn=None,
        state_dict_adapter=Rwkv7StateDictAdapter,
    )


__all__ = ["Rwkv7Model", "model_registry", "rwkv7_configs"]
