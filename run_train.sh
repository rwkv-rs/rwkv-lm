#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODULE="${RWKV_TORCHTITAN_MODULE:-rwkv_trainer}"
CONFIG="${RWKV_TORCHTITAN_CONFIG:-rwkv7_debug}"
NGPU="${NGPU:-1}"
TORCHRUN="${RWKV_TORCHRUN:-${SCRIPT_DIR}/.venv/bin/torchrun}"

if [[ ! -x "${TORCHRUN}" ]]; then
  echo "rwkv-trainer requires an executable torchrun at ${TORCHRUN}; run 'uv sync --all-extras'." >&2
  exit 1
fi

exec "${TORCHRUN}" --nproc_per_node="${NGPU}" -m torchtitan.train \
  --module "${MODULE}" --config "${CONFIG}" "$@"
