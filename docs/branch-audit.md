# Legacy branch capability audit

This integration does not merge or cherry-pick either legacy implementation.
The comparison is against `main@5a2b3d4` and records the immutable tips that
were reviewed.

## `codex/rwkv-lm-standard-checkpoint-contract@8b9eb5a`

Capabilities retained as behavioral contracts:

- TorchTitan-native DCP ownership of model, optimizer, scheduler, step, RNG,
  and dataloader cursor.
- FQN-keyed optimizer state and complete parameter assignment checks.
- binidx magic-prime ordering, cursor round trip, and resume identity rejection.
- explicit launcher defaults that are isolated from generic ambient `MODULE`
  and `CONFIG` values.
- adapter-only and merged PEFT artifact validation.

Implementation rejected from direct reuse:

- its copied RWKV model and state-dict tensor mapping;
- its pretokenized-only tokenizer boundary;
- historical FLA/operator routing;
- any whole-fork TorchTitan layout.

The new package instead delegates model/config/tokenizer/checkpoint keys to
Transformers, operators to FlashRWKV2, and LoRA to PEFT.

## `feature/rwkv7-statepassing-head-size@81908e5`

The branch adds private `cuda/` state-passing, TimeMix, and ChannelMix sources
plus a local model. Those sources are not imported. Its required capability is
covered by FlashRWKV2's public `statetune_recurrent_fp32io16` contract at
`8494189a426f1cfec3e623a46efb81627bef64be`, including differentiable FP32 WKV
state. The trainer contains no `_C` binding calls or copied CUDA/Triton source.

Although the public operator supports additional head sizes, the first trainer
release rejects anything except canonical `head_size=64` until the complete
model/artifact/FSDP acceptance matrix is extended.

## Deletion gate

Neither remote branch may be deleted until the replacement PR is merged from
an exact final `torchtitan` OID, all CPU/single-GPU/8-GPU gates and artifact
reload checks pass, and remote `main` is verified at the merge OID.
