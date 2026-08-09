# Immutable dependency provenance

| Component | Repository | Locked revision |
|---|---|---|
| TorchTitan | `pytorch/torchtitan` | `96276d86577cf3e3bd29de72586e76af62010a55` |
| Transformers RWKV | `rwkv-rs/transformers-rwkv` | `10052072bd3ca172957d97e11b24ae297e0e0072` |
| tokenizers-rwkv | `rwkv-rs/tokenizers-rwkv` | `c5d8dde5ff49c70e4656199d5033a84e03c21b2b` |
| FlashRWKV2 | `rwkv-rs/FlashRWKV2` | `v0.1.0a5` / `046257e7918d93a0fefce868e2ab580fbf6078da` |
| PEFT | `huggingface/peft` | `v0.18.0` / `abefcce659b892b42271831504b66f3f2340b655` |
| RWKV-PEFT reference | `JL-er/RWKV-PEFT` | `5704c39f8ab1d2ac63936ab392aadb6ba526e1a5` |

`uv.lock` fixes the published FlashRWKV2 source artifact and its hash. The upstream
tag commit is recorded separately so exported training artifacts retain source
provenance as well as the installed package version.
