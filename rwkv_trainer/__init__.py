"""Thin external TorchTitan extension for Transformers RWKV-7."""

from .model import RwkvModelAdapter
from .model_spec import model_registry

__all__ = ["RwkvModelAdapter", "model_registry"]
