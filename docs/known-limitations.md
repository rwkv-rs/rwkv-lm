# Known limitations

## FP16 unmerged LoRA versus merged-model logits

Status: recorded on 2026-08-12 and temporarily deprioritized. This limitation
does not indicate a failure of RWKV training, DCP recovery, or adapter-only PEFT
serialization. It does prevent claiming that the FP16 merged artifact has
passed the repository's strict full-model parity gate.

The two artifact contracts are intentionally separate:

1. The training contract compares the BF16 DCP-restored PEFT model with an
   adapter-only artifact reloaded on the same BF16 training path.
2. The inference contract compares an FP16 base with an active, unmerged LoRA
   adapter against a model produced by FP32 `merge_and_unload()`, stored as FP16
   safetensors, reloaded with `AutoModelForCausalLM`, and prepared through the
   same Transformers/FlashRWKV2 inference path.

The unresolved result concerns only the second contract. It is not a direct
comparison between BF16 training logits and FP16 inference logits.

### Reproduced environment

- rwkv-trainer: `e0af1f6383aa822143fff41db4671d2a6e516f40`
- transformers-rwkv: `204974d2a8cec8d31a57cd8c4737a2ee3a32549a`
- FlashRWKV2: `0.1.0a6`
- PEFT: `0.19.1`
- TorchTitan: `96276d86577cf3e3bd29de72586e76af62010a55`
- model: canonical dense RWKV7 7.2B artifact
- inference dtype: FP16, with the provider's BF16 embedding layout
- requested tolerance: `atol=2e-2, rtol=2e-2`

With deterministic PyTorch algorithms and
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, the result was reproducible:

- compared logits: `1,048,576`
- elements outside tolerance: `87`
- maximum absolute difference: `0.080078125`
- maximum relative difference: `4.101369857788086`

Setting `CUBLAS_WORKSPACE_CONFIG` before Python startup produced the same
numbers, so this is not being treated as a flaky pass/fail result.

Layer-level diagnostics did not identify a weight-mapping or LoRA-scaling
error. Sampled TimeMix projections from layers 0, 1, and 31 differed by at most
approximately `0.001953125` between the optional-LoRA and merged FlashRWKV2
paths. The larger final-logit difference appears after the small FP16
representation differences propagate through the recurrent model.

### Current policy

- BF16 training, infctx, DCP, and adapter-only validation may be evaluated
  independently of this limitation.
- `rwkv-export --merge` remains fail closed: it does not write success metadata
  when the configured full-logit tolerance is exceeded.
- The tolerance is not relaxed after observing the result.
- A merged artifact must not be described as having passed strict full-model
  parity until a provider or artifact-format change satisfies the gate, or the
  acceptance contract is deliberately revised in a separate reviewed change.
- No trainer-local projection, kernel, or inference fallback is introduced.

Possible future investigations are a provider-level numerical alignment for
the optional-LoRA path, a higher-precision merged artifact, or an explicitly
reviewed layer/weight-level merge contract. None is selected by this record.
