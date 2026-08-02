"""Single-device RWKV-7 model used by TorchTitan."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torchtitan.protocols import BaseModel


class Rwkv7Model(BaseModel):
    """TorchTitan model container built from transformers-rwkv RWKV-7 blocks."""

    @dataclass(kw_only=True, slots=True)
    class Config(BaseModel.Config):
        vocab_size: int = 65_536
        context_length: int = 4_096
        hidden_size: int = 2_048
        num_hidden_layers: int = 24
        intermediate_size: int = 7_168
        head_size: int = 64
        layer_norm_epsilon: float = 1e-5
        group_norm_epsilon: float = 64e-5
        wkv_backend: str = "flash_rwkv"

        def update_from_config(self, *, config: Any, **kwargs: Any) -> None:
            self.context_length = config.training.seq_len

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
        from transformers.models.rwkv7.configuration_rwkv7 import (
            Rwkv7Config as TransformersRwkv7Config,
        )
        from transformers.models.rwkv7.modeling_rwkv7 import Rwkv7ForCausalLM

        self.config = config
        transformers_config = TransformersRwkv7Config(
            vocab_size=config.vocab_size,
            context_length=config.context_length,
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            intermediate_size=config.intermediate_size,
            head_size=config.head_size,
            layer_norm_epsilon=config.layer_norm_epsilon,
            group_norm_epsilon=config.group_norm_epsilon,
            use_cache=False,
            wkv_backend=config.wkv_backend,
            wkv_state_dtype="float32",
        )
        transformers_model = Rwkv7ForCausalLM(transformers_config)

        self.enable_weight_tying = False
        self.tok_embeddings = transformers_model.model.embeddings
        self.layers = nn.ModuleDict(
            {
                str(layer_index): layer
                for layer_index, layer in enumerate(transformers_model.model.blocks)
            }
        )
        self.norm = transformers_model.model.ln_out
        self.lm_head = transformers_model.head

    def verify_module_protocol(self) -> None:
        """Validate the intentional transformers-rwkv module boundary."""
        if not self.layers:
            raise RuntimeError("RWKV-7 requires at least one transformers-rwkv block")
        if len(self.layers) != self.config.num_hidden_layers:
            raise RuntimeError("transformers-rwkv returned an unexpected block count")

    def init_states(self, *, buffer_device: torch.device | None = None) -> None:
        del buffer_device
        from transformers.models.rwkv7.modeling_rwkv7 import Rwkv7PreTrainedModel

        self.apply(lambda module: Rwkv7PreTrainedModel._init_weights(self, module))

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

        v_first = torch.empty(
            0,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        for layer_index, block in enumerate(self.layers.values()):
            hidden_states, v_first, _, _, _ = block(
                hidden_states,
                v_first,
                state[0][layer_index],
                state[1][layer_index],
                state[2][layer_index],
            )
        return self.lm_head(self.norm(hidden_states))
