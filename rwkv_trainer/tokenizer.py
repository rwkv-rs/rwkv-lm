"""TorchTitan tokenizer component backed directly by Transformers AutoTokenizer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torchtitan.components.tokenizer import BaseTokenizer
from transformers import AutoTokenizer


class RwkvAutoTokenizer(BaseTokenizer):
    @dataclass(kw_only=True, slots=True)
    class Config(BaseTokenizer.Config):
        pass

    def __init__(self, config: Config, *, tokenizer_path: str):
        del config
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        self.eos_id = self.tokenizer.eos_token_id

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if add_bos:
            if self.tokenizer.bos_token_id is None:
                raise ValueError("RWKV tokenizer has no BOS token.")
            ids.insert(0, self.tokenizer.bos_token_id)
        if add_eos:
            if self.eos_id is None:
                raise ValueError("RWKV tokenizer has no EOS token.")
            ids.append(self.eos_id)
        return ids

    def decode(self, token_ids: list[int], **kwargs: Any) -> str:
        return self.tokenizer.decode(token_ids, **kwargs)

    def get_vocab_size(self) -> int:
        return len(self.tokenizer)

    def apply_chat_template(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        return self.tokenizer.apply_chat_template(messages, tokenize=False, **kwargs)


__all__ = ["RwkvAutoTokenizer"]
