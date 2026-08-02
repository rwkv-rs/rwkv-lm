"""Activation-checkpoint policy shared by RWKV's FSDP2 block paths."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import TypeVar

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from .checkpoint import CheckpointContractError

FSDP2_ACTIVATION_CHECKPOINT_POLICY = "rwkv-blocks-non-reentrant-v1"

_T = TypeVar("_T")


@dataclass(frozen=True)
class FSDP2ActivationCheckpointing:
    """Select only RWKV blocks for non-reentrant recomputation.

    Non-reentrant checkpointing is required when frozen embeddings feed blocks
    containing the only trainable parameters, as in LoRA training: gradients
    must still be recorded even though none of the block inputs requires grad.
    """

    _selected_modules: tuple[nn.Module, ...] = field(repr=False)
    use_reentrant: bool = field(default=False, init=False)

    @classmethod
    def for_rwkv_blocks(
        cls,
        blocks: Iterable[nn.Module],
        *,
        enabled: bool,
    ) -> FSDP2ActivationCheckpointing:
        selected_modules = tuple(blocks) if enabled else ()
        if any(not isinstance(module, nn.Module) for module in selected_modules):
            raise CheckpointContractError(
                "FSDP2 activation checkpoint selection must contain modules"
            )
        if len({id(module) for module in selected_modules}) != len(selected_modules):
            raise CheckpointContractError(
                "FSDP2 activation checkpoint selection contains duplicate modules"
            )
        if enabled and not selected_modules:
            raise CheckpointContractError(
                "FSDP2 activation checkpointing requires at least one RWKV block"
            )
        return cls(_selected_modules=selected_modules)

    @property
    def enabled(self) -> bool:
        return bool(self._selected_modules)

    def selects(self, module: nn.Module) -> bool:
        return any(module is selected for selected in self._selected_modules)

    def require_rwkv_blocks(
        self,
        blocks: Iterable[nn.Module],
        *,
        enabled: bool,
    ) -> None:
        expected_modules = tuple(blocks) if enabled else ()
        if len(expected_modules) != len(self._selected_modules) or any(
            actual is not expected
            for actual, expected in zip(
                self._selected_modules,
                expected_modules,
                strict=True,
            )
        ):
            raise CheckpointContractError(
                "FSDP2 activation checkpoint policy must select exactly RWKV blocks"
            )

    def run(
        self,
        module: nn.Module,
        forward: Callable[..., _T],
        *inputs: object,
    ) -> _T:
        """Run selected blocks through the fixed non-reentrant policy."""

        if not self.selects(module) or not torch.is_grad_enabled():
            return forward(*inputs)
        return torch_checkpoint(
            forward,
            *inputs,
            use_reentrant=self.use_reentrant,
        )


__all__ = [
    "FSDP2_ACTIVATION_CHECKPOINT_POLICY",
    "FSDP2ActivationCheckpointing",
]
