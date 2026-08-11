# Immutable dependency provenance

| Component | Repository | Locked revision |
|---|---|---|
| TorchTitan | `pytorch/torchtitan` | `96276d86577cf3e3bd29de72586e76af62010a55` |
| Transformers RWKV | `rwkv-rs/transformers-rwkv` | `204974d2a8cec8d31a57cd8c4737a2ee3a32549a` |
| tokenizers-rwkv | `rwkv-rs/tokenizers-rwkv` | `c5d8dde5ff49c70e4656199d5033a84e03c21b2b` |
| FlashRWKV2 | `rwkv-rs/FlashRWKV2` | `v0.1.0a6` / `255df16b85edeac69ce512bb4a5ad1122a11863d` |
| PEFT | `huggingface/peft` | `v0.19.1` / `ba6a19060d6ab54a87538a6e77e3e4d5a907375b` |
| RWKV-PEFT reference | `JL-er/RWKV-PEFT` | `5704c39f8ab1d2ac63936ab392aadb6ba526e1a5` |

`uv.lock` fixes the published FlashRWKV2 source artifact and its hash. The upstream
tag commit is recorded separately so exported training artifacts retain source
provenance as well as the installed package version.

FlashRWKV2 `0.1.0a6` artifacts are pinned to SHA-256
`73d7ff2f055d03c0ae092a39e6387f8a372a320f31e6262a204a4eae9f2ec274`
for the CPython 3.12 x86_64 wheel and
`d2c0cf7c55fb3a6732c1e674fd9e3615719c685c2e0b50828e592c489e86eb29`
for the source distribution.
