"""TorchTitan tokenizer boundary for pretokenized RWKV data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torchtitan.components.tokenizer import BaseTokenizer

from rwkv_lm.models.rwkv7.state_dict_adapter import validate_rwkv7_artifact


class RwkvPretokenizedTokenizer(BaseTokenizer):
    """Identity boundary for token IDs produced before TorchTitan training."""

    @dataclass(kw_only=True, slots=True)
    class Config(BaseTokenizer.Config):
        vocab_size: int

    def __init__(self, config: Config, *, tokenizer_path: str) -> None:
        super().__init__()
        if config.vocab_size <= 1:
            raise ValueError("RWKV tokenizer vocab_size must be greater than one")
        identity = validate_rwkv7_artifact(tokenizer_path)
        if identity.vocab_size != config.vocab_size:
            raise ValueError(
                "RWKV pretokenized vocabulary does not match the model artifact"
            )
        self.vocab_size = config.vocab_size
        self.eos_id = 0
        self.model_identity = identity.model_identity
        self.source_revision = identity.source_revision

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        del text, add_bos, add_eos
        raise RuntimeError(
            "RWKV TorchTitan data is pretokenized-only; encode text with the "
            "bound transformers-rwkv tokenizer before building binidx data"
        )

    def decode(self, token_ids: list[int], **kwargs: Any) -> str:
        del token_ids, kwargs
        raise RuntimeError(
            "RWKV TorchTitan data is pretokenized-only and does not decode text"
        )

    def get_vocab_size(self) -> int:
        return self.vocab_size


__all__ = ["RwkvPretokenizedTokenizer"]
