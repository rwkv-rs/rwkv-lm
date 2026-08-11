"""Canonical train_temp optimizer grouping on TorchTitan's container lifecycle."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

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
        # Parent construction invokes the overridden shape-aware grouping method.
        config.param_groups = [
            ParamGroupConfig(
                pattern=r".*",
                optimizer_name=config.optimizer_name,
                optimizer_kwargs={
                    "lr": config.lr,
                    "betas": config.betas,
                    "eps": config.eps,
                    "weight_decay": config.weight_decay,
                },
            )
        ]
        super().__init__(config, model_parts=model_parts)

    @staticmethod
    def _build_param_groups(model, param_group_configs, impl_kwargs):
        if len(param_group_configs) != 1:
            raise ValueError("RWKV optimizer requires exactly one canonical parameter config.")
        parameter_config = param_group_configs[0]
        optimizer_kwargs = parameter_config.optimizer_kwargs
        optimizer_name = parameter_config.optimizer_name
        lr = optimizer_kwargs["lr"]
        weight_decay = optimizer_kwargs["weight_decay"]
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
            ("matrix", 1.0, weight_decay),
            ("other", 1.0, 0.0),
        ):
            if not groups[key]:
                continue
            result.append(
                {
                    "params": groups[key],
                    "param_names": names[key],
                    **impl_kwargs,
                    "lr": lr * lr_scale,
                    "betas": optimizer_kwargs["betas"],
                    "eps": optimizer_kwargs["eps"],
                    "weight_decay": decay,
                }
            )
            patterns.append(key)
        return {optimizer_name: result}, {optimizer_name: patterns}

    @staticmethod
    def _initialize_missing_states(optimizer: torch.optim.Optimizer) -> None:
        """Materialize states for parameters unused by the current RWKV step."""

        missing = [
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
            if parameter.requires_grad and not optimizer.state[parameter]
        ]
        if not missing:
            return
        parameters = [
            parameter for group in optimizer.param_groups for parameter in group["params"]
        ]
        saved_grads = [parameter.grad for parameter in parameters]
        saved_lrs = [group["lr"] for group in optimizer.param_groups]
        try:
            # TorchTitan checkpoints immediately after optimizer.step(), before
            # clearing the completed step's gradients. Hide those gradients so
            # this synthetic step touches only parameters with lazy state.
            for parameter in parameters:
                parameter.grad = None
            for group in optimizer.param_groups:
                group["lr"] = 0.0
            for parameter in missing:
                parameter.grad = torch.zeros_like(parameter)
            optimizer.step()
            for parameter in missing:
                step = optimizer.state[parameter].get("step")
                if isinstance(step, torch.Tensor):
                    step.zero_()
                elif step is not None:
                    optimizer.state[parameter]["step"] = 0
        finally:
            for parameter, gradient in zip(parameters, saved_grads, strict=True):
                parameter.grad = gradient
            for group, lr in zip(optimizer.param_groups, saved_lrs, strict=True):
                group["lr"] = lr

    def state_dict(self) -> dict[str, Any]:
        for optimizer in self.optimizers:
            self._initialize_missing_states(optimizer)
        return super().state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        for optimizer in self.optimizers:
            self._initialize_missing_states(optimizer)
        super().load_state_dict(state_dict)


def rwkv_optimizer(*, lr: float, weight_decay: float = 0.01) -> RwkvOptimizersContainer.Config:
    return RwkvOptimizersContainer.Config(lr=lr, weight_decay=weight_decay)


__all__ = ["RwkvOptimizersContainer", "rwkv_optimizer"]
