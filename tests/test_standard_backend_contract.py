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


def _standard_flash_model():
    return create_standard_rwkv7_model(
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


@pytest.fixture
def _isolated_fla_contract(monkeypatch):
    modeling = pytest.importorskip("transformers.models.rwkv7.modeling_rwkv7")
    validate_provenance = getattr(
        modeling,
        "validate_rwkv7_runtime_provenance",
        None,
    )
    if callable(validate_provenance):
        monkeypatch.setattr(
            modeling,
            "validate_rwkv7_runtime_provenance",
            dict,
        )
    contract_loader = modeling._load_fla_rwkv7_contract
    contract_loader.cache_clear()
    yield
    contract_loader.cache_clear()


def test_fla_rwkv7_public_api_exposes_stateful_dispatch_and_telemetry() -> None:
    rwkv7 = pytest.importorskip("fla.ops.rwkv7")
    chunk_rwkv7 = getattr(rwkv7, "chunk_rwkv7", None)
    get_last_provider = getattr(rwkv7, "get_last_rwkv7_provider", None)

    assert callable(chunk_rwkv7)
    assert callable(get_last_provider)
    parameters = inspect.signature(chunk_rwkv7).parameters
    assert not (_PUBLIC_CHUNK_RWKV7_PARAMETERS - parameters.keys())
    assert not inspect.signature(get_last_provider).parameters


def test_transformers_rwkv7_public_provenance_contract_matches_manifest() -> None:
    modeling = pytest.importorskip("transformers.models.rwkv7.modeling_rwkv7")
    validate_provenance = getattr(
        modeling,
        "validate_rwkv7_runtime_provenance",
        None,
    )

    assert callable(validate_provenance)
    assert not inspect.signature(validate_provenance).parameters
    assert modeling.RWKV7_FLA_EXTRA == "flash-rwkv"
    assert modeling.RWKV7_FLA_REPOSITORY == "https://github.com/rwkv-rs/fla-rwkv.git"
    assert modeling.RWKV7_FLA_REVISION == "a4a8aa98df6ec5322f194a80ec57363dd045adfc"
    assert modeling.RWKV7_FLA_REQUIREMENT == (
        "flash-linear-attention[flash-rwkv] @ "
        "git+https://github.com/rwkv-rs/fla-rwkv.git@"
        "a4a8aa98df6ec5322f194a80ec57363dd045adfc"
    )
    assert (
        modeling.RWKV7_FLASH_RWKV_REPOSITORY
        == "https://github.com/rwkv-rs/FlashRWKV.git"
    )
    assert (
        modeling.RWKV7_FLASH_RWKV_REVISION
        == "866aafd2eed146b0eda1ce03444009ae030f89e3"
    )


def test_transformers_rwkv7_public_provenance_matches_installed_runtime() -> None:
    modeling = pytest.importorskip("transformers.models.rwkv7.modeling_rwkv7")
    validate_provenance = getattr(
        modeling,
        "validate_rwkv7_runtime_provenance",
        None,
    )

    assert callable(validate_provenance)
    provenance = validate_provenance()
    assert provenance["distribution"] == "flash-linear-attention"
    assert provenance["extra"] == "flash-rwkv"
    assert provenance["repository"] == "https://github.com/rwkv-rs/fla-rwkv.git"
    assert provenance["revision"] == "a4a8aa98df6ec5322f194a80ec57363dd045adfc"
    assert provenance["flash_rwkv_distribution"] == "flash-rwkv"
    assert (
        provenance["flash_rwkv_repository"]
        == "https://github.com/rwkv-rs/FlashRWKV.git"
    )
    assert (
        provenance["flash_rwkv_revision"]
        == "866aafd2eed146b0eda1ce03444009ae030f89e3"
    )
    assert provenance["source_kind"] in {"editable", "vcs"}
    assert provenance["flash_rwkv_source_kind"] in {"editable", "vcs"}


def test_standard_training_loss_reaches_public_fla_boundary_without_fallback(
    monkeypatch,
    _isolated_fla_contract,
) -> None:
    rwkv7 = pytest.importorskip("fla.ops.rwkv7")
    pytest.importorskip("transformers.models.rwkv7")
    monkeypatch.setenv("FLA_FLASH_RWKV", "1")

    model = _standard_flash_model()
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


def test_standard_training_rejects_non_flash_fla_provider(
    monkeypatch,
    _isolated_fla_contract,
) -> None:
    rwkv7 = pytest.importorskip("fla.ops.rwkv7")
    monkeypatch.setenv("FLA_FLASH_RWKV", "1")

    def non_flash_chunk(
        r,
        w,
        k,
        v,
        a,
        b,
        *,
        initial_state,
        output_final_state,
        cu_seqlens,
        state_indices,
        mode,
    ):
        del r, w, k, a, b, output_final_state, cu_seqlens, state_indices, mode
        return v, initial_state

    monkeypatch.setattr(rwkv7, "chunk_rwkv7", non_flash_chunk)
    monkeypatch.setattr(rwkv7, "get_last_rwkv7_provider", lambda: "triton")
    model = _standard_flash_model()
    input_ids = torch.arange(16).reshape(1, 16)

    with pytest.raises(
        RuntimeError,
        match=(
            "Explicit FlashRWKV request failed closed: "
            "FLA public chunk_rwkv7 did not select FlashRWKV; fallback is disabled"
        ),
    ):
        standard_rwkv7_training_loss(model, input_ids, input_ids)
