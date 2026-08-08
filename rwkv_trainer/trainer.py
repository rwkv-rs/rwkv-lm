"""Minimal TorchTitan Trainer override for RWKV-PEFT-style batch-local TBPTT."""

from __future__ import annotations

import spmd_types as spmd
import torch
from torch.utils.checkpoint import checkpoint
from torchtitan.trainer import Trainer
from transformers import RwkvTrainingState

from .model import RwkvModelAdapter


class RwkvTrainer(Trainer):
    def forward_backward_step(
        self,
        *,
        input_dict: dict[str, torch.Tensor] | list[dict[str, torch.Tensor]],
        labels: torch.Tensor | list[torch.Tensor],
        global_valid_tokens: float,
    ) -> torch.Tensor:
        model_config = self.config.model_spec.model
        if not isinstance(model_config, RwkvModelAdapter.Config) or not model_config.infctx:
            return super().forward_backward_step(
                input_dict=input_dict,
                labels=labels,
                global_valid_tokens=global_valid_tokens,
            )
        if self.parallel_dims.pp_enabled:
            raise NotImplementedError("RWKV infctx does not support pipeline parallelism.")
        if not isinstance(input_dict, dict) or not isinstance(labels, torch.Tensor):
            raise TypeError("RWKV infctx expects one non-pipeline microbatch.")
        tokens, labels, extra_kwargs = self.post_dataloading_process(input_dict, labels)
        extra_kwargs.pop("positions", None)
        if extra_kwargs:
            raise TypeError(f"RWKV infctx received unsupported inputs: {sorted(extra_kwargs)}")
        chunk_ctx = model_config.chunk_ctx
        logical_length = tokens.shape[1]
        if chunk_ctx <= 0 or chunk_ctx > logical_length:
            raise ValueError(
                "chunk_ctx must be positive and no greater than logical length "
                f"{logical_length}, got {chunk_ctx}."
            )
        if tokens.shape != labels.shape:
            raise ValueError(
                f"RWKV infctx input/label shapes must match, got {tokens.shape} and {labels.shape}."
            )
        model = self.model_parts[0]
        if not isinstance(model, RwkvModelAdapter):
            raise TypeError(f"RWKV infctx expected RwkvModelAdapter, got {type(model).__name__}.")
        dtype = next(model.parameters()).dtype
        state = RwkvTrainingState.zeros(
            model.rwkv_model.config,
            tokens.shape[0],
            device=tokens.device,
            dtype=dtype,
        )
        accumulated = torch.zeros((), device=tokens.device, dtype=torch.float32)

        def run_chunk(chunk_tokens, time_shift, wkv, channel_shift):
            chunk_state = RwkvTrainingState(time_shift, wkv, channel_shift)
            outputs = model.forward_stateful(chunk_tokens, chunk_state)
            next_state = outputs.training_state
            if next_state is None:
                raise RuntimeError(
                    "Transformers RWKV stateful forward did not return a training state."
                )
            return outputs.logits, *next_state.tensors()

        with self.train_context():
            for start in range(0, logical_length, chunk_ctx):
                end = min(start + chunk_ctx, logical_length)
                result = checkpoint(
                    run_chunk,
                    tokens[:, start:end],
                    *state.tensors(),
                    use_reentrant=False,
                    preserve_rng_state=True,
                )
                logits, *next_tensors = result
                chunk_loss, _ = self.loss_fn(logits, labels[:, start:end], global_valid_tokens)
                with spmd.no_typecheck():
                    chunk_loss.backward()
                accumulated += chunk_loss.detach()
                state = RwkvTrainingState(*next_tensors).clone_detach()
                del logits, result, chunk_loss
        del state
        return accumulated


__all__ = ["RwkvTrainer"]
