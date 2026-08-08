"""Only outer-wrapper prefix handling; tensor names remain canonical HF RWKV names."""

from __future__ import annotations

from typing import Any

from torchtitan.protocols.state_dict_adapter import StateDictAdapter

from .model import RwkvModelAdapter


class RwkvStateDictAdapter(StateDictAdapter):
    def __init__(self, model_config: RwkvModelAdapter.Config, hf_assets_path: str | None):
        super().__init__(model_config, hf_assets_path)
        self.peft_enabled = model_config.peft.enabled

    @property
    def _native_prefix(self) -> str:
        return "hf_model.base_model.model." if self.peft_enabled else "hf_model."

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        prefix = self._native_prefix
        for key, value in state_dict.items():
            if not key.startswith(prefix):
                continue
            hf_key = key.removeprefix(prefix)
            if self.peft_enabled and "lora_" in hf_key:
                continue
            result[hf_key] = value
        return result

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        prefix = self._native_prefix
        return {f"{prefix}{key}": value for key, value in hf_state_dict.items()}


__all__ = ["RwkvStateDictAdapter"]
