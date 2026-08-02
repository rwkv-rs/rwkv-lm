#!/usr/bin/bash

set -euo pipefail

NGPU="${NGPU:-8}"
export LOG_RANK="${LOG_RANK:-0}"
MODULE="${MODULE:-rwkv_lm.models.rwkv7}"
CONFIG="${CONFIG:-rwkv7_debugmodel}"
COMM_MODE="${COMM_MODE:-}"

if [[ -n "${COMM_MODE}" ]]; then
    NGPU="${NGPU}" LOCAL_RANK=0 python3 -m torchtitan.train \
        --module "${MODULE}" \
        --config "${CONFIG}" \
        "$@" \
        --comm.mode="${COMM_MODE}" \
        --training.steps 1
else
    PYTORCH_ALLOC_CONF="expandable_segments:True" \
        torchrun \
        --nproc_per_node="${NGPU}" \
        --rdzv_backend c10d \
        --rdzv_endpoint="localhost:0" \
        --local-ranks-filter "${LOG_RANK}" \
        --role rank \
        --tee 3 \
        -m torchtitan.train \
        --module "${MODULE}" \
        --config "${CONFIG}" \
        "$@"
fi
