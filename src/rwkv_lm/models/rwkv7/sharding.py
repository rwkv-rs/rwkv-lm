"""Declarative TorchTitan sharding for RWKV-7."""

from __future__ import annotations

import spmd_types as spmd
from torchtitan.models.common.decoder_sharding import (
    dense_activation_placement,
    dense_param_placement,
)
from torchtitan.protocols.sharding import ShardingConfig

from .model import Rwkv7Model


def _replicated_parameter_config(*names: str) -> ShardingConfig:
    return ShardingConfig(
        state_shardings={name: dense_param_placement(tp=spmd.I) for name in names}
    )


def _linear_config() -> ShardingConfig:
    activation = dense_activation_placement(tp=spmd.I, cp=spmd.I)
    return ShardingConfig(
        state_shardings={
            "weight": dense_param_placement(tp=spmd.I),
            "bias": dense_param_placement(tp=spmd.I),
        },
        in_src_shardings={"input": activation},
        out_src_shardings=activation,
    )


def _norm_config() -> ShardingConfig:
    activation = dense_activation_placement(tp=spmd.I, cp=spmd.I)
    return ShardingConfig(
        state_shardings={
            "weight": dense_param_placement(tp=spmd.I),
            "bias": dense_param_placement(tp=spmd.I),
        },
        in_src_shardings={"input": activation},
        out_src_shardings=activation,
    )


def set_rwkv7_sharding_config(
    config: Rwkv7Model.Config,
    *,
    enable_sp: bool,
) -> None:
    """Populate every RWKV-7 parameter and recurrent-state placement."""
    if enable_sp:
        raise ValueError("RWKV-7 sequence parallelism is not implemented")

    activation = dense_activation_placement(tp=spmd.I, cp=spmd.I)
    recurrent_state = dense_activation_placement(tp=spmd.I, cp=spmd.I)
    config.sharding_config = ShardingConfig(
        in_src_shardings={"tokens": activation, "positions": activation},
        out_src_shardings=activation,
    )
    config.tok_embeddings.sharding_config = ShardingConfig(
        state_shardings={"weight": dense_param_placement(tp=spmd.I)},
        in_src_shardings={"input": activation},
        out_src_shardings=activation,
    )
    config.norm.sharding_config = _norm_config()
    config.lm_head.sharding_config = _linear_config()

    for layer in config.layers:
        layer.sharding_config = ShardingConfig(
            in_src_shardings={
                "hidden_states": activation,
                "v_first": activation,
                "att_shift": recurrent_state,
                "wkv_state": recurrent_state,
                "ffn_shift": recurrent_state,
            },
            out_src_shardings=(
                activation,
                activation,
                recurrent_state,
                recurrent_state,
                recurrent_state,
            ),
        )
        layer.ln0.sharding_config = (
            _norm_config()
            if hasattr(layer.ln0, "normalized_shape")
            else ShardingConfig(
                in_src_shardings={"input": activation},
                out_src_shardings=activation,
            )
        )
        layer.ln1.sharding_config = _norm_config()
        layer.ln2.sharding_config = _norm_config()

        time_mix = layer.att
        time_mix_parameter_names = [
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
        if time_mix.layer_id > 0:
            time_mix_parameter_names.append("v0")
        time_mix.sharding_config = _replicated_parameter_config(
            *time_mix_parameter_names
        )
        time_mix.sharding_config.in_src_shardings = {
            "hidden_states": activation,
            "v_first": activation,
            "previous_hidden_state": recurrent_state,
            "wkv_state": recurrent_state,
        }
        time_mix.sharding_config.out_src_shardings = (
            activation,
            activation,
            recurrent_state,
            recurrent_state,
        )
        for linear in (
            time_mix.w1,
            time_mix.w2,
            time_mix.a1,
            time_mix.a2,
            time_mix.v1,
            time_mix.v2,
            time_mix.g1,
            time_mix.g2,
            time_mix.receptance,
            time_mix.key,
            time_mix.value,
            time_mix.output,
        ):
            if linear is not None:
                linear.sharding_config = _linear_config()
        time_mix.ln_x.sharding_config = _norm_config()

        channel_mix = layer.ffn
        channel_mix.sharding_config = _replicated_parameter_config("x_k")
        channel_mix.sharding_config.in_src_shardings = {
            "hidden_states": activation,
            "previous_hidden_state": recurrent_state,
        }
        channel_mix.sharding_config.out_src_shardings = (
            activation,
            recurrent_state,
        )
        channel_mix.key.sharding_config = _linear_config()
        channel_mix.value.sharding_config = _linear_config()


__all__ = ["set_rwkv7_sharding_config"]
