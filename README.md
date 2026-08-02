# rwkv-lm

This package trains the standard `transformers-rwkv` RWKV-7 causal language
model with composable PyTorch FSDP2. Model layers and recurrent state are owned
by `transformers-rwkv`; accelerated WKV dispatch is owned by the public
`fla-rwkv` / `FlashRWKV` provider interface. The package does not build or ship
a second RWKV model or private CUDA kernels.

The training environment must provide compatible builds of those three model
and backend projects in addition to this package. An explicit
`--wkv_backend flash_rwkv` request fails closed if that provider cannot accept
the real training tensors. `--wkv_backend reference` selects the differentiable
Transformers reference recurrence.

## Convert an existing checkpoint once

Raw RWKV `.pth` files are migration inputs, not training checkpoints. Convert
one to a standard model directory before training:

```bash
rwkv-convert-legacy-checkpoint legacy-model.pth out/rwkv-init \
  --wkv-backend flash_rwkv
```

`rwkv-train` intentionally refuses raw `.pth` input. The converter remains the
only legacy model entry point; subsequent save, load, resume, merge, and
inference use standard model directories or complete FSDP2 checkpoints.

## Train

`demo-training-run.sh` and `demo-training-run-v7-pile.sh` contain complete
single-node examples. They require a standard model directory at the configured
`PROJ_DIR/rwkv-init` path and launch the canonical entry point with `torchrun`:

```bash
torchrun --standalone --nproc-per-node=8 train.py \
  --load_model out/rwkv-init \
  --strategy fsdp2 --accelerator gpu --devices 8 --precision bf16 \
  --wkv_backend flash_rwkv \
  --data_type binidx --data_file /path/to/dataset \
  --proj_dir out/run --ctx_len 4096 --n_layer 24 --n_embd 2048 \
  --head_size 64 --vocab_size 65536 --micro_bsz 1 \
  --magic_prime 40320
```

The FSDP2 checkpoint transaction includes model, optimizer, mixed-precision
scaler, scheduler position, RNG, and data cursor state. Resuming from its
directory restores those states together.

## PEFT and infctx

LoRA is attached to public Transformers projections. For example:

```text
--lora_rank 8 --lora_alpha 16 \
--lora_target_modules time_mix.output,channel_mix.value
```

Adapter-only save/load, checkpoint resume, merged standard model export, and
inference are handled by the same standard adapter. Recurrent chunk training is
selected with:

```text
--train_type infctx --chunk_ctx 16
```

The infctx path passes the public Transformers recurrent state between chunks,
detaches it at TBPTT boundaries, preserves response-token gradients, and resets
state only at an explicit reset boundary.
