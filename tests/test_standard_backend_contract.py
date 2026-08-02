from __future__ import annotations

import inspect

import pytest

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
