# rwkv-lm

This package provides a TorchTitan-native RWKV-7 model integration. TorchTitan
owns the trainer, FSDP, activation checkpointing, optimizer, metrics, LoRA
conversion, and DCP checkpoint lifecycle. RWKV-specific code owns the readable
single-device model, recurrent state, binidx cursor, declarative sharding, and
the standard Transformers state-dict mapping.

The WKV call uses the public `fla.ops.rwkv7.recurrent_rwkv7` interface after
`transformers-rwkv` validates the pinned FLA/FlashRWKV runtime provenance. No
reference or native-kernel fallback exists in this package.

## Convert an existing checkpoint once

Raw RWKV `.pth` files are migration inputs, not training checkpoints. Convert
one to a standard model directory before training:

```bash
rwkv-convert-legacy-checkpoint legacy-model.pth out/rwkv-init
```

## Train

The debug flavor is a self-contained synthetic-data configuration that starts
from random weights and enables complete TorchTitan checkpoints. The canonical
launcher follows TorchTitan's `MODULE`/`CONFIG` contract:

```bash
NGPU=1 CONFIG=rwkv7_debugmodel ./run_train.sh
```

The production flavor requires an existing standard Transformers model
directory and an RWKV binidx dataset. It fails closed when either input is
missing:

```bash
export RWKV_HF_ASSETS_PATH=/path/to/standard-rwkv7
export RWKV_BINIDX_PATH=/path/to/tokenized-dataset-prefix
export RWKV_MAGIC_PRIME=81082817
CONFIG=rwkv7_1_5b ./run_train.sh \
  --hf-assets-path "${RWKV_HF_ASSETS_PATH}" \
  --checkpoint.initial-load-path "${RWKV_HF_ASSETS_PATH}" \
  --dataloader.dataset-path "${RWKV_BINIDX_PATH}" \
  --dataloader.magic-prime "${RWKV_MAGIC_PRIME}"
```

TorchTitan checkpoints own the model, optimizer, scheduler, training step, and
stateful data cursor. The newest complete DCP checkpoint is resumed by the
TorchTitan checkpoint manager.

## LoRA and recurrent chunks

`rwkv7_debugmodel_lora` applies TorchTitan's `LoRAConverter`; only adapter
parameters are trainable, DCP resume retains the adapter and optimizer state,
and the RWKV state adapter exports merged standard Transformers weights.

`rwkv7_debugmodel_infctx` and `rwkv7_1_5b_infctx` split each training sequence
into aligned recurrent chunks. Every layer's attention shift, FP32 WKV matrix,
and channel-mix shift are carried forward and detached at TBPTT boundaries.
`positions` is accepted as TorchTitan dataloader metadata without host
synchronization; explicit `state`, `reset_state`, and `reset_mask` inputs own
reset/continue semantics and callers can request the final recurrent state.
