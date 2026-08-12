# RWKV Trainer

`rwkv-trainer` is a thin external [TorchTitan](https://github.com/pytorch/torchtitan)
extension for dense text RWKV-7 training. It deliberately does not contain an RWKV
model, tokenizer vocabulary, checkpoint converter, CUDA/Triton kernel, or fallback
recurrent implementation.

The ownership boundary is strict:

- `transformers-rwkv` owns `RwkvConfig`, `RwkvForCausalLM`, initialization, state,
  tokenizer behavior, checkpoint keys, and safetensors artifacts.
- FlashRWKV2 owns every high-performance RWKV operator.
- Hugging Face PEFT owns LoRA injection, adapter serialization, and merging.
- TorchTitan owns FSDP2, DCP, accumulation, scheduling, clipping, metrics, and lifecycle.
- This package owns only the adapters and policies needed to connect those projects.

## Launch

```bash
uv sync --all-extras
RWKV_TORCHTITAN_CONFIG=rwkv7_debug ./run_train.sh \
  --hf-assets-path /path/to/rwkv7-hf-artifact
```

Available configs are `rwkv7_debug`, `rwkv7_pretrain`,
`rwkv7_pretrain_infctx`, `rwkv7_lora`, and `rwkv7_lora_infctx`. Production model
shape always comes from the HF artifact given by `hf_assets_path`.

Version provenance is recorded in [`DEPENDENCIES.md`](DEPENDENCIES.md). Unsupported
in the first release: TP, PP, CP, MoE, RWKV-VL, SFT/masked labels, QLoRA,
MiSS/DiSHA, PiSSA, AdaLoRA, prefix tuning, and state tuning.

The current distinction between validated LoRA training artifacts and the
remaining FP16 merged-inference parity limitation is recorded in
[`docs/known-limitations.md`](docs/known-limitations.md).
