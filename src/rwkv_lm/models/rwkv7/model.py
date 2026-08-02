"""Single-device RWKV-7 model used by TorchTitan."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torchtitan.models.common.embedding import Embedding
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.nn_modules import GroupNorm, Identity, LayerNorm
from torchtitan.protocols import BaseModel
from torchtitan.protocols.module import Module


def _normal_parameter(parameter: nn.Parameter) -> None:
    nn.init.normal_(parameter, mean=0.0, std=0.02)


def _zero_parameter(parameter: nn.Parameter) -> None:
    nn.init.zeros_(parameter)


def _one_parameter(parameter: nn.Parameter) -> None:
    nn.init.ones_(parameter)


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
        param_init={"weight": _zero_parameter if zero else _normal_parameter},
    )


def _layer_norm_config(hidden_size: int, epsilon: float) -> LayerNorm.Config:
    return LayerNorm.Config(
        normalized_shape=hidden_size,
        eps=epsilon,
        param_init={"weight": _one_parameter, "bias": _zero_parameter},
    )


class Rwkv7TimeMix(Module):
    """RWKV-7 time mixing with the public FLA/FlashRWKV stateful kernel."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        layer_id: int
        hidden_size: int
        head_size: int
        wkv_backend: str
        w1: Linear.Config
        w2: Linear.Config
        a1: Linear.Config
        a2: Linear.Config
        v1: Linear.Config | None
        v2: Linear.Config | None
        g1: Linear.Config
        g2: Linear.Config
        receptance: Linear.Config
        key: Linear.Config
        value: Linear.Config
        output: Linear.Config
        ln_x: GroupNorm.Config

    def __init__(self, config: Config) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        num_heads = hidden_size // config.head_size

        self.layer_id = config.layer_id
        self.hidden_size = hidden_size
        self.head_size = config.head_size
        self.num_heads = num_heads
        self.wkv_backend = config.wkv_backend
        for name in ("x_r", "x_w", "x_k", "x_v", "x_a", "x_g"):
            setattr(self, name, nn.Parameter(torch.empty(1, 1, hidden_size)))
        self.w0 = nn.Parameter(torch.empty(1, 1, hidden_size))
        self.a0 = nn.Parameter(torch.empty(1, 1, hidden_size))
        if config.layer_id > 0:
            self.v0 = nn.Parameter(torch.empty(1, 1, hidden_size))
        self.k_k = nn.Parameter(torch.empty(1, 1, hidden_size))
        self.k_a = nn.Parameter(torch.empty(1, 1, hidden_size))
        self.r_k = nn.Parameter(torch.empty(num_heads, config.head_size))

        self.w1 = config.w1.build()
        self.w2 = config.w2.build()
        self.a1 = config.a1.build()
        self.a2 = config.a2.build()
        if config.v1 is not None and config.v2 is not None:
            self.v1 = config.v1.build()
            self.v2 = config.v2.build()
        self.g1 = config.g1.build()
        self.g2 = config.g2.build()
        self.receptance = config.receptance.build()
        self.key = config.key.build()
        self.value = config.value.build()
        self.output = config.output.build()
        self.ln_x = config.ln_x.build()

    def forward(
        self,
        hidden_states: torch.Tensor,
        v_first: torch.Tensor,
        previous_hidden_state: torch.Tensor,
        wkv_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.wkv_backend != "flash_rwkv":
            raise RuntimeError("RWKV-7 training requires the FlashRWKV provider")

        batch_size, sequence_length, hidden_size = hidden_states.shape
        shifted = torch.cat(
            (previous_hidden_state[:, None], hidden_states[:, :-1]),
            dim=1,
        )
        shifted = shifted - hidden_states
        final_hidden_state = hidden_states[:, -1]
        inputs = {
            name: hidden_states + shifted * getattr(self, f"x_{name}")
            for name in ("r", "w", "k", "v", "a", "g")
        }
        receptance = self.receptance(inputs["r"])
        key = self.key(inputs["k"])
        value = self.value(inputs["v"])
        raw_decay = self.w0 + self.w2(torch.tanh(self.w1(inputs["w"])))
        if self.layer_id == 0:
            v_first = value
        else:
            value = value + (v_first - value) * torch.sigmoid(
                self.v0 + self.v2(self.v1(inputs["v"]))
            )
        learning_rate = torch.sigmoid(self.a0 + self.a2(self.a1(inputs["a"])))
        gate = self.g2(torch.sigmoid(self.g1(inputs["g"])))
        normalized_key = F.normalize(
            (key * self.k_k).view(
                batch_size,
                sequence_length,
                self.num_heads,
                self.head_size,
            ),
            dim=-1,
        ).view(batch_size, sequence_length, hidden_size)
        key = key * (1 + (learning_rate - 1) * self.k_a)

        from fla.ops.rwkv7 import chunk_rwkv7, get_last_rwkv7_provider

        kernel_inputs = [
            tensor.view(
                batch_size,
                sequence_length,
                self.num_heads,
                self.head_size,
            ).contiguous()
            for tensor in (
                receptance,
                raw_decay,
                key,
                value,
                -normalized_key,
                normalized_key * learning_rate,
            )
        ]
        kernel_inputs[1] = (-F.softplus(-kernel_inputs[1]) - 0.5).contiguous()
        output, final_wkv_state = chunk_rwkv7(
            *kernel_inputs,
            initial_state=wkv_state,
            output_final_state=True,
            cu_seqlens=None,
            state_indices=None,
            mode="fp32io16",
        )
        if get_last_rwkv7_provider() != "flash_rwkv":
            raise RuntimeError(
                "FLA public chunk_rwkv7 did not select FlashRWKV; fallback is disabled"
            )
        if output.shape != kernel_inputs[3].shape:
            raise RuntimeError("FLA public chunk_rwkv7 returned an invalid output")
        if final_wkv_state.shape != wkv_state.shape:
            raise RuntimeError("FLA public chunk_rwkv7 returned an invalid state")

        output = output.reshape(batch_size, sequence_length, hidden_size)
        output = self.ln_x(output.flatten(0, 1)).view_as(output)
        local = (
            (
                receptance.view(
                    batch_size,
                    sequence_length,
                    self.num_heads,
                    self.head_size,
                )
                * key.view(
                    batch_size,
                    sequence_length,
                    self.num_heads,
                    self.head_size,
                )
                * self.r_k
            ).sum(-1, keepdim=True)
            * value.view(
                batch_size,
                sequence_length,
                self.num_heads,
                self.head_size,
            )
        ).view_as(output)
        return (
            self.output((output + local) * gate),
            v_first,
            final_hidden_state,
            final_wkv_state,
        )


class Rwkv7ChannelMix(Module):
    """RWKV-7 channel mixing module."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        hidden_size: int
        key: Linear.Config
        value: Linear.Config

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.x_k = nn.Parameter(torch.empty(1, 1, config.hidden_size))
        self.key = config.key.build()
        self.value = config.value.build()

    def forward(
        self,
        hidden_states: torch.Tensor,
        previous_hidden_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shifted = torch.cat(
            (previous_hidden_state[:, None], hidden_states[:, :-1]),
            dim=1,
        )
        shifted = shifted - hidden_states
        output = self.value(
            F.relu(self.key(hidden_states + shifted * self.x_k)).square()
        )
        return output, hidden_states[:, -1]


class Rwkv7Block(Module):
    """One TorchTitan-configurable RWKV-7 block."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        ln0: LayerNorm.Config | Identity.Config
        ln1: LayerNorm.Config
        ln2: LayerNorm.Config
        att: Rwkv7TimeMix.Config
        ffn: Rwkv7ChannelMix.Config

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.ln0 = config.ln0.build()
        self.ln1 = config.ln1.build()
        self.ln2 = config.ln2.build()
        self.att = config.att.build()
        self.ffn = config.ffn.build()

    def forward(
        self,
        hidden_states: torch.Tensor,
        v_first: torch.Tensor,
        att_shift: torch.Tensor,
        wkv_state: torch.Tensor,
        ffn_shift: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        hidden_states = self.ln0(hidden_states)
        output, v_first, att_shift, wkv_state = self.att(
            self.ln1(hidden_states),
            v_first,
            att_shift,
            wkv_state,
        )
        hidden_states = hidden_states + output
        output, ffn_shift = self.ffn(self.ln2(hidden_states), ffn_shift)
        return hidden_states + output, v_first, att_shift, wkv_state, ffn_shift


class Rwkv7Model(BaseModel):
    """Readable single-device RWKV-7 model."""

    @dataclass(kw_only=True, slots=True)
    class Config(BaseModel.Config):
        vocab_size: int
        context_length: int
        hidden_size: int
        num_hidden_layers: int
        intermediate_size: int
        head_size: int
        tok_embeddings: Embedding.Config
        layers: list[Rwkv7Block.Config]
        norm: LayerNorm.Config
        lm_head: Linear.Config
        layer_norm_epsilon: float = 1e-5
        group_norm_epsilon: float = 64e-5
        wkv_backend: str = "flash_rwkv"

        def update_from_config(self, *, config: Any, **kwargs: Any) -> None:
            self.context_length = config.training.seq_len
            from .sharding import set_rwkv7_sharding_config

            set_rwkv7_sharding_config(
                self,
                enable_sp=config.parallelism.enable_sequence_parallel,
            )

        def get_nparams_and_flops(
            self,
            model: BaseModel,
            seq_len: int,
        ) -> tuple[int, int]:
            del seq_len
            parameter_count = sum(parameter.numel() for parameter in model.parameters())
            return parameter_count, 6 * parameter_count

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.enable_weight_tying = False
        self.tok_embeddings = config.tok_embeddings.build()
        self.layers = nn.ModuleDict(
            {
                str(layer_index): layer_config.build()
                for layer_index, layer_config in enumerate(config.layers)
            }
        )
        self.norm = config.norm.build()
        self.lm_head = config.lm_head.build()

    def _init_self_parameters(self) -> None:
        pass

    def _initial_state(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_heads = self.config.hidden_size // self.config.head_size
        return (
            torch.zeros(
                self.config.num_hidden_layers,
                batch_size,
                self.config.hidden_size,
                dtype=dtype,
                device=device,
            ),
            torch.zeros(
                self.config.num_hidden_layers,
                batch_size,
                num_heads,
                self.config.head_size,
                self.config.head_size,
                dtype=torch.float32,
                device=device,
            ),
            torch.zeros(
                self.config.num_hidden_layers,
                batch_size,
                self.config.hidden_size,
                dtype=dtype,
                device=device,
            ),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        hidden_states = self.tok_embeddings(tokens)
        if state is None:
            state = self._initial_state(
                hidden_states.shape[0],
                hidden_states.dtype,
                hidden_states.device,
            )

        v_first = torch.zeros_like(hidden_states)
        for layer_index, block in enumerate(self.layers.values()):
            hidden_states, v_first, _, _, _ = block(
                hidden_states,
                v_first,
                state[0][layer_index],
                state[1][layer_index],
                state[2][layer_index],
            )
        return self.lm_head(self.norm(hidden_states))


def build_rwkv7_config(
    *,
    vocab_size: int,
    context_length: int,
    hidden_size: int,
    num_hidden_layers: int,
    intermediate_size: int,
    head_size: int,
    layer_norm_epsilon: float = 1e-5,
    group_norm_epsilon: float = 64e-5,
    wkv_backend: str = "flash_rwkv",
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
        wkv_backend=wkv_backend,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=hidden_size,
            param_init={"weight": _normal_parameter},
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
            wkv_backend=wkv_backend,
            param_init={name: _zero_parameter for name in direct_parameter_names},
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
                param_init={"weight": _one_parameter, "bias": _zero_parameter},
            ),
        )
        channel_mix = Rwkv7ChannelMix.Config(
            hidden_size=hidden_size,
            param_init={"x_k": _zero_parameter},
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


__all__ = [
    "Rwkv7Block",
    "Rwkv7ChannelMix",
    "Rwkv7Model",
    "Rwkv7TimeMix",
    "build_rwkv7_config",
]
