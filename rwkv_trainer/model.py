"""TorchTitan model protocol adapter around the canonical HF RWKV implementation."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn
from torchtitan.protocols import BaseModel
from transformers import AutoConfig, AutoModelForCausalLM, RwkvConfig, RwkvForCausalLM


@dataclass(kw_only=True, slots=True)
class PeftSettings:
    enabled: bool = False
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.0
    target_modules: list[str] = field(
        default_factory=lambda: ["receptance", "key", "value", "output"]
    )


class RwkvModelAdapter(BaseModel):
    """Thin positional-input adapter; all RWKV layers remain HF modules."""

    @dataclass(kw_only=True, slots=True)
    class Config(BaseModel.Config):
        hf_assets_path: str = ""
        peft: PeftSettings = field(default_factory=PeftSettings)
        infctx: bool = False
        chunk_ctx: int = 0

        def update_from_config(self, *, config, **kwargs: Any) -> None:
            del kwargs
            self.hf_assets_path = config.hf_assets_path

        def get_nparams_and_flops(self, model: BaseModel, seq_len: int) -> tuple[int, int]:
            del seq_len
            count = sum(parameter.numel() for parameter in model.parameters())
            return count, 0

    def __init__(self, config: Config):
        super().__init__()
        if not config.hf_assets_path:
            raise ValueError("RWKV model construction requires a non-empty hf_assets_path.")
        hf_config = AutoConfig.from_pretrained(config.hf_assets_path, local_files_only=True)
        if not isinstance(hf_config, RwkvConfig) or hf_config.architecture_version != "rwkv7":
            raise TypeError(
                "hf_assets_path must contain a native Transformers RWKV-7 RwkvConfig; "
                f"got {type(hf_config).__name__}."
            )
        if hf_config.head_size != 64:
            raise ValueError(
                f"The first rwkv-trainer release requires head_size=64, got {hf_config.head_size}."
            )
        self.config = config
        model = AutoModelForCausalLM.from_config(hf_config)
        if not isinstance(model, RwkvForCausalLM):
            raise TypeError(
                f"AutoModelForCausalLM constructed {type(model).__name__}, "
                "expected RwkvForCausalLM."
            )
        self.hf_model = self._apply_peft(model) if config.peft.enabled else model

    def _apply_peft(self, model: RwkvForCausalLM) -> nn.Module:
        from peft import LoraConfig, TaskType, get_peft_model

        settings = self.config.peft
        if settings.rank <= 0 or settings.alpha <= 0:
            raise ValueError("LoRA rank and alpha must be positive.")
        if not 0 <= settings.dropout < 1:
            raise ValueError("LoRA dropout must be in [0, 1).")
        if not settings.target_modules:
            raise ValueError("LoRA target_modules must not be empty.")
        target_pattern = (
            r".*\.att\.(?:" + "|".join(re.escape(name) for name in settings.target_modules) + r")$"
        )
        wrapped = get_peft_model(
            model,
            LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=settings.rank,
                lora_alpha=settings.alpha,
                lora_dropout=settings.dropout,
                target_modules=target_pattern,
                bias="none",
            ),
        )
        trainable = {
            name for name, parameter in wrapped.named_parameters() if parameter.requires_grad
        }
        if not trainable or any("lora_" not in name for name in trainable):
            raise RuntimeError(
                "PEFT must freeze the base model and expose only LoRA parameters as trainable."
            )
        return wrapped

    @property
    def rwkv_model(self) -> RwkvForCausalLM:
        model = self.hf_model
        while not isinstance(model, RwkvForCausalLM):
            next_model = getattr(model, "model", None)
            if next_model is None or next_model is model:
                raise TypeError("Unable to locate RwkvForCausalLM inside the PEFT wrapper.")
            model = next_model
        return model

    @property
    def lm_head(self) -> nn.Module:
        return self.rwkv_model.head

    def verify_module_protocol(self) -> None:
        """HF and PEFT children are intentionally ordinary ``nn.Module`` objects."""

    def init_states(self, *, buffer_device: torch.device | None = None) -> None:
        del buffer_device
        model = self.rwkv_model
        model.model.reset_parameters()
        model.reset_head_parameters()
        if self.config.peft.enabled:
            from peft.tuners.lora import LoraLayer

            for module in self.hf_model.modules():
                if isinstance(module, LoraLayer):
                    for adapter_name in module.lora_A:
                        module.reset_lora_parameters(adapter_name, init_lora_weights=True)

    def forward(self, tokens: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        kwargs.pop("positions", None)
        if kwargs:
            raise TypeError(f"Unsupported RWKV model forward arguments: {sorted(kwargs)}")
        return self.hf_model(input_ids=tokens, use_cache=False, return_dict=True).logits

    def forward_stateful(self, tokens: torch.Tensor, training_state):
        if not self.config.infctx:
            raise RuntimeError("forward_stateful is available only for an infctx model config.")
        return self.hf_model(
            input_ids=tokens,
            training_state=training_state,
            use_cache=False,
            return_dict=True,
        )


__all__ = ["PeftSettings", "RwkvModelAdapter"]
