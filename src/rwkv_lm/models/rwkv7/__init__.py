"""TorchTitan model registry for RWKV-7."""

from __future__ import annotations

import math
from functools import partial

from torch import nn
from torchtitan.models.common.embedding import Embedding
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.nn_modules import GroupNorm, Identity, LayerNorm
from torchtitan.models.utils import validate_converter_order
from torchtitan.protocols.model import ModelConfigConverter
from torchtitan.protocols.model_spec import ModelSpec

from .dataloader import RwkvDataLoader
from .model import Rwkv7Block, Rwkv7ChannelMix, Rwkv7Model, Rwkv7TimeMix
from .parallelize import parallelize_rwkv7
from .state_dict_adapter import Rwkv7StateDictAdapter
from .tokenizer import RwkvPretokenizedTokenizer

_LINEAR_INIT = {
    "weight": partial(nn.init.normal_, mean=0.0, std=0.02),
}
_ZERO_LINEAR_INIT = {"weight": nn.init.zeros_}
_NORM_INIT = {"weight": nn.init.ones_, "bias": nn.init.zeros_}


def _linear_config(
    in_features: int,
    out_features: int,
    *,
    zero: bool = False,
) -> Linear.Config:
    return Linear.Config(
        in_features=in_features,
        out_features=out_features,
        bias=False,
        param_init=_ZERO_LINEAR_INIT if zero else _LINEAR_INIT,
    )


def _layer_norm_config(hidden_size: int, epsilon: float) -> LayerNorm.Config:
    return LayerNorm.Config(
        normalized_shape=hidden_size,
        eps=epsilon,
        param_init=_NORM_INIT,
    )


def _build_rwkv7_config(
    *,
    vocab_size: int,
    context_length: int,
    hidden_size: int,
    num_hidden_layers: int,
    intermediate_size: int,
    head_size: int,
    layer_norm_epsilon: float = 1e-5,
    group_norm_epsilon: float = 64e-5,
    recurrent_chunk_size: int | None = None,
    detach_state_between_chunks: bool = False,
) -> Rwkv7Model.Config:
    """Build the declarative TorchTitan config tree for one RWKV-7 flavor."""
    if hidden_size % head_size != 0:
        raise ValueError("RWKV-7 hidden_size must be divisible by head_size")
    decay_rank = max(32, round(2.5 * math.sqrt(hidden_size) / 32) * 32)
    value_rank = max(32, round(1.7 * math.sqrt(hidden_size) / 32) * 32)
    gate_rank = max(32, round(5.0 * math.sqrt(hidden_size) / 32) * 32)

    model_config = Rwkv7Model.Config(
        vocab_size=vocab_size,
        context_length=context_length,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        intermediate_size=intermediate_size,
        head_size=head_size,
        layer_norm_epsilon=layer_norm_epsilon,
        group_norm_epsilon=group_norm_epsilon,
        recurrent_chunk_size=recurrent_chunk_size,
        detach_state_between_chunks=detach_state_between_chunks,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=hidden_size,
            param_init=_LINEAR_INIT,
        ),
        layers=[],
        norm=_layer_norm_config(hidden_size, layer_norm_epsilon),
        lm_head=_linear_config(hidden_size, vocab_size),
    )
    for layer_id in range(num_hidden_layers):
        direct_parameter_names = [
            "x_r",
            "x_w",
            "x_k",
            "x_v",
            "x_a",
            "x_g",
            "w0",
            "a0",
            "k_k",
            "k_a",
            "r_k",
        ]
        if layer_id > 0:
            direct_parameter_names.append("v0")
        time_mix = Rwkv7TimeMix.Config(
            layer_id=layer_id,
            hidden_size=hidden_size,
            head_size=head_size,
            param_init={name: nn.init.zeros_ for name in direct_parameter_names},
            w1=_linear_config(hidden_size, decay_rank, zero=True),
            w2=_linear_config(decay_rank, hidden_size, zero=True),
            a1=_linear_config(hidden_size, decay_rank, zero=True),
            a2=_linear_config(decay_rank, hidden_size, zero=True),
            v1=(
                _linear_config(hidden_size, value_rank, zero=True)
                if layer_id > 0
                else None
            ),
            v2=(
                _linear_config(value_rank, hidden_size, zero=True)
                if layer_id > 0
                else None
            ),
            g1=_linear_config(hidden_size, gate_rank, zero=True),
            g2=_linear_config(gate_rank, hidden_size, zero=True),
            receptance=_linear_config(hidden_size, hidden_size),
            key=_linear_config(hidden_size, hidden_size),
            value=_linear_config(hidden_size, hidden_size),
            output=_linear_config(hidden_size, hidden_size),
            ln_x=GroupNorm.Config(
                num_groups=hidden_size // head_size,
                num_channels=hidden_size,
                eps=group_norm_epsilon,
                param_init=_NORM_INIT,
            ),
        )
        channel_mix = Rwkv7ChannelMix.Config(
            hidden_size=hidden_size,
            param_init={"x_k": nn.init.zeros_},
            key=_linear_config(hidden_size, intermediate_size),
            value=_linear_config(intermediate_size, hidden_size),
        )
        model_config.layers.append(
            Rwkv7Block.Config(
                ln0=(
                    _layer_norm_config(hidden_size, layer_norm_epsilon)
                    if layer_id == 0
                    else Identity.Config()
                ),
                ln1=_layer_norm_config(hidden_size, layer_norm_epsilon),
                ln2=_layer_norm_config(hidden_size, layer_norm_epsilon),
                att=time_mix,
                ffn=channel_mix,
            )
        )
    return model_config


def _debug_model() -> Rwkv7Model.Config:
    return _build_rwkv7_config(
        vocab_size=1_024,
        context_length=128,
        hidden_size=128,
        num_hidden_layers=2,
        intermediate_size=448,
        head_size=64,
    )


def _g1h_1_5b() -> Rwkv7Model.Config:
    return _build_rwkv7_config(
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


__all__ = [
    "Rwkv7Model",
    "RwkvDataLoader",
    "RwkvPretokenizedTokenizer",
    "model_registry",
    "parallelize_rwkv7",
    "rwkv7_configs",
]
