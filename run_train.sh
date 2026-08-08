#!/usr/bin/env bash
set -euo pipefail

MODULE="${RWKV_TORCHTITAN_MODULE:-rwkv_trainer}"
CONFIG="${RWKV_TORCHTITAN_CONFIG:-rwkv7_debug}"
NGPU="${NGPU:-1}"

exec torchrun --nproc_per_node="${NGPU}" -m torchtitan.train \
  --module "${MODULE}" --config "${CONFIG}" "$@"
