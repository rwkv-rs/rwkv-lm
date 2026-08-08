"""Canonical train_temp optimizer grouping on TorchTitan's container lifecycle."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torchtitan.components.optimizer import OptimizersContainer, ParamGroupConfig


class RwkvOptimizersContainer(OptimizersContainer):
    @dataclass(kw_only=True, slots=True)
    class Config(OptimizersContainer.Config):
        optimizer_name: Literal["Adam", "AdamW"] = "AdamW"
        lr: float = 6e-4
        betas: tuple[float, float] = (0.9, 0.99)
        eps: float = 1e-18
        weight_decay: float = 0.01

    def __init__(self, config: Config, *, model_parts: list[torch.nn.Module]) -> None:
        if config.lr <= 0 or config.eps <= 0 or config.weight_decay < 0:
            raise ValueError(
                "RWKV optimizer lr/eps must be positive and weight_decay non-negative."
            )
        self._rwkv_config = config
        # Parent construction invokes the overridden shape-aware grouping method.
        config.param_groups = [
            ParamGroupConfig(
                pattern=r".*",
                optimizer_name=config.optimizer_name,
                optimizer_kwargs={
                    "lr": config.lr,
                    "betas": config.betas,
                    "eps": config.eps,
                    "weight_decay": 0.0,
                },
            )
        ]
        super().__init__(config, model_parts=model_parts)

    def _build_param_groups(self, model, param_group_configs, impl_kwargs):
        del param_group_configs
        config = self._rwkv_config
        groups = {"w0": [], "matrix": [], "other": []}
        names = {key: [] for key in groups}
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            if "att.w0" in name:
                key = "w0"
            elif parameter.ndim >= 2 and name.endswith(".weight"):
                key = "matrix"
            else:
                key = "other"
            groups[key].append(parameter)
            names[key].append(name)
        result = []
        patterns = []
        for key, lr_scale, decay in (
            ("w0", 2.0, 0.0),
            ("matrix", 1.0, config.weight_decay),
            ("other", 1.0, 0.0),
        ):
            if not groups[key]:
                continue
            result.append(
                {
                    "params": groups[key],
                    "param_names": names[key],
                    **impl_kwargs,
                    "lr": config.lr * lr_scale,
                    "betas": config.betas,
                    "eps": config.eps,
                    "weight_decay": decay,
                }
            )
            patterns.append(key)
        return {config.optimizer_name: result}, {config.optimizer_name: patterns}


def rwkv_optimizer(*, lr: float, weight_decay: float = 0.01) -> RwkvOptimizersContainer.Config:
    return RwkvOptimizersContainer.Config(lr=lr, weight_decay=weight_decay)


__all__ = ["RwkvOptimizersContainer", "rwkv_optimizer"]
