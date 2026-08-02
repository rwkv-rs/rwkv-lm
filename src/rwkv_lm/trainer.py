"""Training configuration and schedule shared by the FSDP2 runner."""

from __future__ import annotations

import math

from .checkpoint import CheckpointContractError

_CHECKPOINT_CONFIG_FIELDS = (
    "accumulate_grad_batches",
    "adam_eps",
    "beta1",
    "beta2",
    "chunk_ctx",
    "ctx_len",
    "data_file",
    "data_type",
    "dim_att",
    "dim_ffn",
    "epoch_steps",
    "grad_clip",
    "grad_cp",
    "head_chunk",
    "head_size",
    "kernel",
    "lora_alpha",
    "lora_dropout",
    "lora_rank",
    "lora_target_modules",
    "lr_final",
    "lr_init",
    "magic_prime",
    "micro_bsz",
    "my_exit_tokens",
    "my_testing",
    "n_embd",
    "n_layer",
    "precision",
    "random_seed",
    "real_bsz",
    "train_stage",
    "train_type",
    "vocab_size",
    "warmup_steps",
    "weight_decay",
    "wkv_backend",
)


def _checkpoint_training_config(args: object) -> dict[str, object]:
    missing = [
        name for name in _CHECKPOINT_CONFIG_FIELDS if not hasattr(args, name)
    ]
    if missing:
        raise CheckpointContractError(
            f"training arguments are missing checkpoint fields: {missing}"
        )
    return {name: getattr(args, name) for name in _CHECKPOINT_CONFIG_FIELDS}


def scheduled_learning_rate(
    args: object,
    global_step: int,
) -> tuple[float, bool]:
    """Return the stateless RWKV learning rate and whether its token limit hit."""

    lr = args.lr_init
    reached_token_limit = False
    warmup_steps = args.warmup_steps
    if args.my_exit_tokens != 0:
        real_tokens = global_step * args.ctx_len * args.real_bsz
        warmup_tokens = warmup_steps * args.ctx_len * args.real_bsz
        decay_tokens = abs(args.my_exit_tokens) - warmup_tokens
        if decay_tokens <= 0:
            raise CheckpointContractError(
                "my_exit_tokens must exceed the configured warmup token count"
            )
        progress = (real_tokens - warmup_tokens) / decay_tokens
        progress = max(0, min(1, progress))
        lr_final_factor = args.lr_final / args.lr_init
        lr_mult = (0.5 + lr_final_factor / 2) + (
            0.5 - lr_final_factor / 2
        ) * math.cos(math.pi * progress)
        if args.my_exit_tokens > 0:
            lr = args.lr_init * lr_mult
        else:
            lr = (lr + args.lr_init * lr_mult) / 2
        reached_token_limit = progress >= 1
    if warmup_steps > 0 and global_step < warmup_steps:
        lr *= 0.01 + 0.99 * global_step / warmup_steps
    return lr, reached_token_limit


__all__ = ["scheduled_learning_rate"]
