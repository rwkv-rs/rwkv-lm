#!/usr/bin/env bash
set -euo pipefail

: "${RWKV_HF_ASSETS_PATH:?set this to a standard transformers-rwkv directory}"
: "${RWKV_BINIDX_PATH:?set this to the .bin/.idx path prefix}"
: "${RWKV_MAGIC_PRIME:?set this to the dataset prime congruent to 2 mod 3}"

RWKV_NPROC_PER_NODE="${RWKV_NPROC_PER_NODE:-8}"
RWKV_DUMP_FOLDER="${RWKV_DUMP_FOLDER:-outputs/rwkv7-1.5b-infctx}"

torchrun --standalone --nproc-per-node="$RWKV_NPROC_PER_NODE" \
  -m torchtitan.train \
  --module rwkv_lm.models.rwkv7 \
  --config rwkv7_1_5b_infctx \
  --hf-assets-path "$RWKV_HF_ASSETS_PATH" \
  --checkpoint.initial-load-path "$RWKV_HF_ASSETS_PATH" \
  --dataloader.dataset-path "$RWKV_BINIDX_PATH" \
  --dataloader.magic-prime "$RWKV_MAGIC_PRIME" \
  --dump-folder "$RWKV_DUMP_FOLDER"
