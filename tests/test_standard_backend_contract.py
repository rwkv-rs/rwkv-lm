from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch

from rwkv_lm.standard_model import (
    create_standard_rwkv7_model,
    standard_rwkv7_training_loss,
)

_PUBLIC_CHUNK_RWKV7_PARAMETERS = frozenset(
    {
        "r",
        "w",
        "k",
        "v",
        "a",
        "b",
        "initial_state",
        "output_final_state",
        "cu_seqlens",
        "state_indices",
        "mode",
    }
)


def test_fla_rwkv7_public_api_exposes_stateful_dispatch_and_telemetry() -> None:
    rwkv7 = pytest.importorskip("fla.ops.rwkv7")
    chunk_rwkv7 = getattr(rwkv7, "chunk_rwkv7", None)
    get_last_provider = getattr(rwkv7, "get_last_rwkv7_provider", None)

    assert callable(chunk_rwkv7)
    assert callable(get_last_provider)
    parameters = inspect.signature(chunk_rwkv7).parameters
    assert not (_PUBLIC_CHUNK_RWKV7_PARAMETERS - parameters.keys())
    assert not inspect.signature(get_last_provider).parameters


def test_standard_training_loss_reaches_public_fla_boundary_without_fallback(
    monkeypatch,
) -> None:
    rwkv7 = pytest.importorskip("fla.ops.rwkv7")
    pytest.importorskip("transformers.models.rwkv7")
    monkeypatch.setenv("FLA_FLASH_RWKV", "1")

    model = create_standard_rwkv7_model(
        SimpleNamespace(
            ctx_len=16,
            dim_ffn=224,
            head_size=64,
            n_embd=64,
            n_layer=2,
            vocab_size=64,
            wkv_backend="flash_rwkv",
        )
    )
    input_ids = torch.arange(16).reshape(1, 16)

    with pytest.raises(
        RuntimeError,
        match=(
            "Explicit FlashRWKV request failed closed: "
            "FLA public chunk_rwkv7 execution failed"
        ),
    ):
        standard_rwkv7_training_loss(model, input_ids, input_ids)

    assert rwkv7.get_last_rwkv7_provider() is None
