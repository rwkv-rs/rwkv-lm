"""Thin external TorchTitan extension for Transformers RWKV-7."""

from .compat import ensure_distributed_set_timeout
from .model import RwkvModelAdapter
from .model_spec import model_registry

ensure_distributed_set_timeout()

__all__ = ["RwkvModelAdapter", "model_registry"]
