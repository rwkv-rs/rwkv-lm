from __future__ import annotations

from types import SimpleNamespace

import pytest

from rwkv_lm.cli import canonicalize_training_capabilities
from rwkv_lm.infctx import InfctxContractError
from rwkv_lm.peft import PeftContractError


def _args(**overrides) -> SimpleNamespace:
    values = {
        "chunk_ctx": 0,
        "ctx_len": 64,
        "lora_adapter": "",
        "lora_alpha": 0.0,
        "lora_dropout": 0.0,
        "lora_rank": 0,
        "lora_target_modules": "",
        "train_stage": 0,
        "train_type": "standard",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize(
    "args",
    [
        _args(),
        _args(
            lora_rank=2,
            lora_alpha=4.0,
            lora_target_modules="time_mix.key,channel_mix.value",
            lora_adapter="adapter.pt",
        ),
        _args(train_type="infctx", chunk_ctx=16),
    ],
    ids=("sft", "peft", "infctx"),
)
def test_standard_cli_accepts_supported_training_capabilities(args) -> None:
    config = canonicalize_training_capabilities(args)

    assert config.enabled is (args.lora_rank > 0)
    if args.train_type == "infctx":
        assert args.chunk_ctx == 16


def test_standard_cli_still_rejects_ambiguous_capability_requests() -> None:
    with pytest.raises(PeftContractError, match="requires an enabled"):
        canonicalize_training_capabilities(_args(lora_adapter="adapter.pt"))
    with pytest.raises(InfctxContractError, match="only valid"):
        canonicalize_training_capabilities(_args(chunk_ctx=16))
