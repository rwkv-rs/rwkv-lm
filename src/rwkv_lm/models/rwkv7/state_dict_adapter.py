"""TorchTitan and transformers-rwkv checkpoint name conversion."""

from __future__ import annotations

from typing import Any

import torch
from torchtitan.models.common.linear import Linear
from torchtitan.protocols.module import Module
from torchtitan.protocols.state_dict_adapter import StateDictAdapter

from .model import Rwkv7Model

_NATIVE_TO_HF_PREFIXES = (
    ("tok_embeddings.", "model.embeddings."),
    ("layers.", "model.blocks."),
    ("norm.", "model.ln_out."),
    ("lm_head.", "head."),
)


def _rename_prefix(
    name: str,
    prefixes: tuple[tuple[str, str], ...],
) -> str:
    for source, destination in prefixes:
        if name.startswith(source):
            return destination + name.removeprefix(source)
    raise KeyError(f"unrecognized RWKV-7 state-dict key: {name}")


class Rwkv7StateDictAdapter(StateDictAdapter):
    """Map TorchTitan RWKV-7 FQNs to standard transformers-rwkv FQNs."""

    def __init__(
        self,
        model_config: Rwkv7Model.Config,
        hf_assets_path: str | None,
    ) -> None:
        super().__init__(model_config, hf_assets_path)
        self._lora_scales = {
            fqn: float(config.alpha) / int(config.rank)
            for fqn, config, _, _ in model_config.traverse(
                Module.Config,
                recurse=True,
            )
            if isinstance(config, Linear.Config)
            and hasattr(config, "rank")
            and hasattr(config, "alpha")
        }

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        state_dict = dict(state_dict)
        for fqn, scale in self._lora_scales.items():
            weight_name = f"{fqn}.weight"
            lora_a_name = f"{fqn}.lora_a.weight"
            lora_b_name = f"{fqn}.lora_b.weight"
            if lora_a_name not in state_dict and lora_b_name not in state_dict:
                continue
            if not all(
                name in state_dict for name in (weight_name, lora_a_name, lora_b_name)
            ):
                raise KeyError(f"incomplete LoRA state for {fqn}")
            weight = state_dict[weight_name]
            lora_a = state_dict.pop(lora_a_name)
            lora_b = state_dict.pop(lora_b_name)
            if not all(
                isinstance(tensor, torch.Tensor) for tensor in (weight, lora_a, lora_b)
            ):
                raise TypeError(f"LoRA state for {fqn} must contain tensors")
            state_dict[weight_name] = weight + scale * (lora_b @ lora_a)
        return {
            _rename_prefix(name, _NATIVE_TO_HF_PREFIXES): value
            for name, value in state_dict.items()
        }

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        hf_to_native = tuple(
            (hf_prefix, native_prefix)
            for native_prefix, hf_prefix in _NATIVE_TO_HF_PREFIXES
        )
        return {
            _rename_prefix(name, hf_to_native): value
            for name, value in hf_state_dict.items()
        }
