# Immutable dependency provenance

| Component | Repository | Locked revision |
|---|---|---|
| TorchTitan | `pytorch/torchtitan` | `96276d86577cf3e3bd29de72586e76af62010a55` |
| Transformers RWKV | `rwkv-rs/transformers-rwkv` | `fe9df14d52d96ead6a5c9e03e49d3076d3ece01a` (temporary prerequisite baseline; update to the stateful API commit before release) |
| tokenizers-rwkv | `rwkv-rs/tokenizers-rwkv` | `c5d8dde5ff49c70e4656199d5033a84e03c21b2b` |
| FlashRWKV2 | `rwkv-rs/FlashRWKV2` | `8494189a426f1cfec3e623a46efb81627bef64be` |
| PEFT | `huggingface/peft` | `v0.18.0` / `abefcce659b892b42271831504b66f3f2340b655` |
| RWKV-PEFT reference | `JL-er/RWKV-PEFT` | `5704c39f8ab1d2ac63936ab392aadb6ba526e1a5` |

The Transformers entry is intentionally marked temporary until its prerequisite
state API change is committed and pushed. A release or GPU evidence commit must not
retain the temporary revision.
