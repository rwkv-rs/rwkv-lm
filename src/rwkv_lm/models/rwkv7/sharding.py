"""RWKV-7 declarative sharding configuration."""

from __future__ import annotations

from .model import Rwkv7Model


def set_rwkv7_sharding_config(
    config: Rwkv7Model.Config,
    *,
    enable_sp: bool,
) -> None:
    """Reject unsupported tensor/sequence parallel declarations explicitly."""
    if enable_sp:
        raise ValueError("RWKV-7 sequence parallelism is not implemented")
