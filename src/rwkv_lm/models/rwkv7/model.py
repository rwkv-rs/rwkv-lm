"""Single-device RWKV-7 model used by TorchTitan."""

from __future__ import annotations

import inspect
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torchtitan.models.common.embedding import Embedding
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.nn_modules import GroupNorm, Identity, LayerNorm
from torchtitan.protocols import BaseModel
from torchtitan.protocols.module import Module, ModuleDict

RWKV7_FLA_RECURRENT_REVISION = "6c4c52c5632e954a8f67d91e4652d551fc57387e"
RWKV7_FLA_REVISION = "88e8ff9d29dcebadb89ebad62ee76951729ea0df"
RWKV7_FLASH_RWKV_REVISION = "c637985558c398de1db6a3c0523b1eec206a88d4"
_RWKV7_RECURRENT_PARAMETERS = {
    "initial_state",
    "output_final_state",
    "cu_seqlens",
    "state_indices",
    "mode",
}


@dataclass(frozen=True, slots=True)
class _Rwkv7Runtime:
    recurrent_rwkv7: Callable[..., tuple[torch.Tensor, torch.Tensor]]
    get_last_provider: Callable[[], str | None]


_RWKV7_RUNTIME: _Rwkv7Runtime | None = None


def initialize_rwkv7_runtime() -> None:
    """Validate and bind the pinned public FLA/FlashRWKV runtime once."""
    global _RWKV7_RUNTIME
    if _RWKV7_RUNTIME is not None:
        return

    from transformers.models.rwkv7 import validate_rwkv7_runtime_provenance

    try:
        provenance = validate_rwkv7_runtime_provenance()
    except (ImportError, RuntimeError) as error:
        raise RuntimeError(
            "RWKV-LM recurrent runtime requires transformers-rwkv provenance for "
            f"FLA {RWKV7_FLA_REVISION} (semantic contract "
            f"{RWKV7_FLA_RECURRENT_REVISION}) and FlashRWKV "
            f"{RWKV7_FLASH_RWKV_REVISION}"
        ) from error
    if not isinstance(provenance, Mapping):
        raise TypeError("transformers-rwkv runtime provenance must be a mapping")
    expected_provenance = {
        "revision": RWKV7_FLA_REVISION,
        "flash_rwkv_revision": RWKV7_FLASH_RWKV_REVISION,
    }
    observed_provenance = {name: provenance.get(name) for name in expected_provenance}
    if observed_provenance != expected_provenance:
        raise RuntimeError(
            "transformers-rwkv runtime provenance does not match RWKV-LM: "
            f"expected={expected_provenance}, observed={observed_provenance}"
        )

    from fla.ops.rwkv7 import (
        FLASH_RWKV_SOURCE_REVISION,
        get_last_rwkv7_provider,
        recurrent_rwkv7,
    )

    if not callable(recurrent_rwkv7) or not callable(get_last_rwkv7_provider):
        raise TypeError(
            "FLA must expose public recurrent_rwkv7 and "
            "get_last_rwkv7_provider callables"
        )
    if FLASH_RWKV_SOURCE_REVISION != RWKV7_FLASH_RWKV_REVISION:
        raise RuntimeError(
            "FLA public recurrent provider pins the wrong FlashRWKV revision: "
            f"expected={RWKV7_FLASH_RWKV_REVISION}, "
            f"observed={FLASH_RWKV_SOURCE_REVISION}"
        )
    try:
        recurrent_parameters = inspect.signature(recurrent_rwkv7).parameters
    except (TypeError, ValueError) as error:
        raise TypeError(
            "FLA public recurrent_rwkv7 signature is not inspectable"
        ) from error
    missing_parameters = sorted(
        _RWKV7_RECURRENT_PARAMETERS - recurrent_parameters.keys()
    )
    if missing_parameters:
        raise TypeError(
            "FLA public recurrent_rwkv7 lacks required parameters: "
            f"{missing_parameters}"
        )
    _RWKV7_RUNTIME = _Rwkv7Runtime(
        recurrent_rwkv7=recurrent_rwkv7,
        get_last_provider=get_last_rwkv7_provider,
    )


@dataclass(frozen=True, slots=True)
class Rwkv7RecurrentState:
    """Per-layer recurrent state carried across RWKV-7 chunks."""

    attention_shift: torch.Tensor
    wkv: torch.Tensor
    ffn_shift: torch.Tensor

    def detached(self) -> Rwkv7RecurrentState:
        """Stop gradients through recurrence while preserving future-token gradients."""
        return Rwkv7RecurrentState(
            attention_shift=self.attention_shift.detach(),
            wkv=self.wkv.detach(),
            ffn_shift=self.ffn_shift.detach(),
        )

    def reset_rows(self, reset_mask: torch.Tensor) -> Rwkv7RecurrentState:
        """Reset selected batch rows without modifying continuing requests."""
        if reset_mask.ndim != 1 or reset_mask.dtype != torch.bool:
            raise ValueError("RWKV reset mask must be a rank-one boolean tensor")

        def reset(tensor: torch.Tensor) -> torch.Tensor:
            if tensor.shape[1] != reset_mask.shape[0]:
                raise ValueError("RWKV reset mask batch size does not match state")
            shape = (1, reset_mask.shape[0], *((1,) * (tensor.ndim - 2)))
            return torch.where(
                reset_mask.view(shape),
                torch.zeros((), dtype=tensor.dtype, device=tensor.device),
                tensor,
            )

        return Rwkv7RecurrentState(
            attention_shift=reset(self.attention_shift),
            wkv=reset(self.wkv),
            ffn_shift=reset(self.ffn_shift),
        )


@dataclass(frozen=True, slots=True)
class Rwkv7ModelOutput:
    """Logits and the final state for explicit recurrent inference/chunking."""

    logits: torch.Tensor
    state: Rwkv7RecurrentState


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

        runtime = _RWKV7_RUNTIME
        if runtime is None:
            raise RuntimeError(
                "RWKV-7 runtime is not initialized; canonical training must run "
                "model Config.update_from_config before the first forward"
            )

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
        recurrent_result = runtime.recurrent_rwkv7(
            *kernel_inputs,
            initial_state=wkv_state,
            output_final_state=True,
            cu_seqlens=None,
            state_indices=None,
            mode="fp32io16",
        )
        if not isinstance(recurrent_result, tuple) or len(recurrent_result) != 2:
            raise RuntimeError("FLA public recurrent_rwkv7 returned an invalid result")
        output, final_wkv_state = recurrent_result
        if runtime.get_last_provider() != "flash_rwkv":
            raise RuntimeError(
                "FLA public recurrent_rwkv7 did not select FlashRWKV; "
                "fallback is disabled"
            )
        if (
            not isinstance(output, torch.Tensor)
            or output.shape != kernel_inputs[3].shape
        ):
            raise RuntimeError("FLA public recurrent_rwkv7 returned an invalid output")
        if (
            not isinstance(final_wkv_state, torch.Tensor)
            or final_wkv_state.shape != wkv_state.shape
        ):
            raise RuntimeError("FLA public recurrent_rwkv7 returned an invalid state")

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
        recurrent_chunk_size: int | None = None
        detach_state_between_chunks: bool = False

        def update_from_config(self, *, config: Any, **kwargs: Any) -> None:
            initialize_rwkv7_runtime()
            from .state_dict_adapter import validate_rwkv7_hf_checkpoint

            validate_rwkv7_hf_checkpoint(config)
            self.context_length = config.training.seq_len
            if self.recurrent_chunk_size is not None and (
                self.recurrent_chunk_size <= 0
                or self.recurrent_chunk_size >= self.context_length
                or self.recurrent_chunk_size % 16 != 0
            ):
                raise ValueError(
                    "RWKV recurrent_chunk_size must be positive, shorter than "
                    "training.seq_len, and divisible by 16"
                )
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
        self.layers = ModuleDict(
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
    ) -> Rwkv7RecurrentState:
        num_heads = self.config.hidden_size // self.config.head_size
        return Rwkv7RecurrentState(
            attention_shift=torch.zeros(
                self.config.num_hidden_layers,
                batch_size,
                self.config.hidden_size,
                dtype=dtype,
                device=device,
            ),
            wkv=torch.zeros(
                self.config.num_hidden_layers,
                batch_size,
                num_heads,
                self.config.head_size,
                self.config.head_size,
                dtype=torch.float32,
                device=device,
            ),
            ffn_shift=torch.zeros(
                self.config.num_hidden_layers,
                batch_size,
                self.config.hidden_size,
                dtype=dtype,
                device=device,
            ),
        )

    def _validate_state(
        self,
        state: Rwkv7RecurrentState,
        *,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        num_heads = self.config.hidden_size // self.config.head_size
        expected_shapes = {
            "attention_shift": (
                self.config.num_hidden_layers,
                batch_size,
                self.config.hidden_size,
            ),
            "wkv": (
                self.config.num_hidden_layers,
                batch_size,
                num_heads,
                self.config.head_size,
                self.config.head_size,
            ),
            "ffn_shift": (
                self.config.num_hidden_layers,
                batch_size,
                self.config.hidden_size,
            ),
        }
        for name, expected_shape in expected_shapes.items():
            tensor = getattr(state, name)
            if tensor.shape != expected_shape:
                raise ValueError(
                    f"RWKV {name} state has shape {tuple(tensor.shape)}, "
                    f"expected {expected_shape}"
                )
            if tensor.device != device:
                raise ValueError(f"RWKV {name} state must be on {device}")
        if state.attention_shift.dtype != dtype or state.ffn_shift.dtype != dtype:
            raise ValueError(f"RWKV shift state must use activation dtype {dtype}")
        if state.wkv.dtype != torch.float32:
            raise ValueError("RWKV WKV state must use torch.float32")

    def _forward_chunk(
        self,
        hidden_states: torch.Tensor,
        state: Rwkv7RecurrentState,
    ) -> tuple[torch.Tensor, Rwkv7RecurrentState]:
        v_first = torch.zeros_like(hidden_states)
        attention_shift = []
        wkv = []
        ffn_shift = []
        for layer_index, block in enumerate(self.layers.values()):
            hidden_states, v_first, next_att, next_wkv, next_ffn = block(
                hidden_states,
                v_first,
                state.attention_shift[layer_index],
                state.wkv[layer_index],
                state.ffn_shift[layer_index],
            )
            attention_shift.append(next_att)
            wkv.append(next_wkv)
            ffn_shift.append(next_ffn)
        return hidden_states, Rwkv7RecurrentState(
            attention_shift=torch.stack(attention_shift),
            wkv=torch.stack(wkv),
            ffn_shift=torch.stack(ffn_shift),
        )

    @staticmethod
    def _validate_positions(tokens: torch.Tensor, positions: torch.Tensor) -> None:
        if positions.shape != tokens.shape:
            raise ValueError(
                f"RWKV positions shape {tuple(positions.shape)} must match "
                f"tokens shape {tuple(tokens.shape)}"
            )
        if positions.dtype not in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        ):
            raise TypeError("RWKV positions must use an integer dtype")
        if positions.device != tokens.device:
            raise ValueError("RWKV positions and tokens must be on the same device")

    @staticmethod
    def _segment_boundaries(
        *,
        sequence_length: int,
        chunk_size: int | None,
    ) -> list[int]:
        boundaries = {0, sequence_length}
        if chunk_size is not None:
            boundaries.update(range(chunk_size, sequence_length, chunk_size))
        return sorted(boundaries)

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        positions: torch.Tensor | None = None,
        state: Rwkv7RecurrentState
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        | None = None,
        reset_state: bool = False,
        reset_mask: torch.Tensor | None = None,
        chunk_size: int | None = None,
        detach_state_between_chunks: bool | None = None,
        return_state: bool = False,
        **extra_kwargs: Any,
    ) -> torch.Tensor | Rwkv7ModelOutput:
        if extra_kwargs:
            raise TypeError(f"unsupported RWKV model inputs: {sorted(extra_kwargs)}")
        if tokens.ndim != 2 or tokens.shape[1] == 0:
            raise ValueError("RWKV tokens must have shape [batch, non-empty sequence]")
        if positions is not None:
            self._validate_positions(tokens, positions)
        if state is not None and not isinstance(state, Rwkv7RecurrentState):
            if len(state) != 3:
                raise ValueError(
                    "RWKV recurrent state tuple must contain three tensors"
                )
            state = Rwkv7RecurrentState(*state)

        hidden_states = self.tok_embeddings(tokens)
        if reset_state:
            state = None
        if state is None:
            state = self._initial_state(
                hidden_states.shape[0],
                hidden_states.dtype,
                hidden_states.device,
            )
        self._validate_state(
            state,
            batch_size=hidden_states.shape[0],
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        if reset_mask is not None:
            if reset_mask.device != hidden_states.device:
                raise ValueError(
                    "RWKV reset_mask and tokens must be on the same device"
                )
            state = state.reset_rows(reset_mask)

        chunk_size = (
            self.config.recurrent_chunk_size if chunk_size is None else chunk_size
        )
        if chunk_size is not None and (
            chunk_size <= 0 or chunk_size >= tokens.shape[1] or chunk_size % 16 != 0
        ):
            raise ValueError(
                "RWKV recurrent chunk_size must be positive, shorter than the "
                "sequence, and divisible by 16"
            )
        detach_chunks = (
            self.config.detach_state_between_chunks
            if detach_state_between_chunks is None
            else detach_state_between_chunks
        )
        boundaries = self._segment_boundaries(
            sequence_length=tokens.shape[1],
            chunk_size=chunk_size,
        )
        chunk_outputs = []
        for segment_index, (start, end) in enumerate(pairwise(boundaries)):
            chunk_output, state = self._forward_chunk(
                hidden_states[:, start:end],
                state,
            )
            chunk_outputs.append(chunk_output)
            if detach_chunks and segment_index + 1 < len(boundaries) - 1:
                state = state.detached()

        logits = self.lm_head(self.norm(torch.cat(chunk_outputs, dim=1)))
        if return_state:
            return Rwkv7ModelOutput(logits=logits, state=state)
        return logits


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
    "RWKV7_FLASH_RWKV_REVISION",
    "RWKV7_FLA_RECURRENT_REVISION",
    "RWKV7_FLA_REVISION",
    "Rwkv7Block",
    "Rwkv7ChannelMix",
    "Rwkv7Model",
    "Rwkv7ModelOutput",
    "Rwkv7RecurrentState",
    "Rwkv7TimeMix",
    "build_rwkv7_config",
    "initialize_rwkv7_runtime",
]
