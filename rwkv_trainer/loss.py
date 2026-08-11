"""Global-token-normalized train_temp L2Wrap loss."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torchtitan.components.loss import IGNORE_INDEX, BaseLoss


class RwkvL2WrapLoss(BaseLoss):
    @dataclass(kw_only=True, slots=True)
    class Config(BaseLoss.Config):
        factor: float = 1e-4

    def __init__(self, config: Config, *, compile_config=None):
        del compile_config
        if config.factor != 1e-4:
            raise ValueError(
                "FlashRWKV2's canonical L2Wrap operator fixes the train_temp factor at 1e-4."
            )

    def __call__(self, pred, labels, global_valid_tokens=None):
        if labels.dtype != torch.long:
            raise TypeError(f"RWKV labels must have dtype torch.long, got {labels.dtype}.")
        if pred.ndim != 3 or labels.shape != pred.shape[:2]:
            raise ValueError(
                "RWKV logits/labels must have shapes [batch,time,vocab] and [batch,time], "
                f"got {tuple(pred.shape)} and {tuple(labels.shape)}."
            )
        if torch.any((labels < 0) | (labels >= pred.shape[-1])).item():
            raise ValueError(
                "RWKV v1 binidx loss requires every label to be valid; "
                f"masked or out-of-vocabulary labels such as {IGNORE_INDEX} are unsupported."
            )
        if global_valid_tokens is None or float(global_valid_tokens) <= 0:
            raise ValueError("RwkvL2WrapLoss requires TorchTitan's positive global_valid_tokens.")
        from flashrwkv2 import pretrain_l2wrap_ce_bf16

        local_tokens = labels.numel()
        local_mean = pretrain_l2wrap_ce_bf16(pred.contiguous(), labels.contiguous())
        return local_mean * (local_tokens / global_valid_tokens), {}


__all__ = ["RwkvL2WrapLoss"]
