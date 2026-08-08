"""TorchTitan ModelSpec registration for the external RWKV module."""

from __future__ import annotations

from torchtitan.protocols.model_spec import ModelSpec

from .model import PeftSettings, RwkvModelAdapter
from .parallelize import parallelize_rwkv
from .state_dict import RwkvStateDictAdapter

_FLAVORS = {
    "debug": (False, False, 0),
    "pretrain": (False, False, 0),
    "pretrain_infctx": (False, True, 1024),
    "lora": (True, False, 0),
    "lora_infctx": (True, True, 1024),
}


def model_registry(flavor: str) -> ModelSpec:
    try:
        peft, infctx, chunk_ctx = _FLAVORS[flavor]
    except KeyError as error:
        raise ValueError(
            f"Unknown RWKV flavor {flavor!r}; expected one of {sorted(_FLAVORS)}"
        ) from error
    return ModelSpec(
        name="rwkv7",
        flavor=flavor,
        model=RwkvModelAdapter.Config(
            peft=PeftSettings(enabled=peft),
            infctx=infctx,
            chunk_ctx=chunk_ctx,
        ),
        parallelize_fn=parallelize_rwkv,
        pipelining_fn=None,
        post_optimizer_build_fn=None,
        state_dict_adapter=RwkvStateDictAdapter,
    )


__all__ = ["model_registry"]
