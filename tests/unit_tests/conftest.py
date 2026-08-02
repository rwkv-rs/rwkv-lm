import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

SOURCE_REVISION = "1" * 40


@pytest.fixture
def rwkv7_artifact_factory(
    tmp_path: Path,
) -> Callable[..., tuple[Path, str]]:
    def write_artifact(
        name: str = "artifact",
        *,
        source_revision: str = SOURCE_REVISION,
        vocab_size: int = 1_024,
    ) -> tuple[Path, str]:
        path = tmp_path / name
        path.mkdir()
        vocab = {"<|endoftext|>": 0, "<unk>": 1}
        vocab.update({f"token_{index}": index for index in range(2, vocab_size)})
        backend = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
        backend.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend,
            bos_token="<|endoftext|>",
            eos_token="<|endoftext|>",
            pad_token="<|endoftext|>",
            unk_token="<unk>",
        )
        tokenizer.save_pretrained(path)
        tokenizer_files = {
            tokenizer_path.name: hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()
            for tokenizer_path in sorted(path.iterdir())
            if tokenizer_path.is_file()
        }
        config = {
            "architectures": ["Rwkv7ForCausalLM"],
            "hidden_size": 128,
            "model_type": "rwkv7",
            "vocab_size": vocab_size,
        }
        (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
        identity_payload = {
            "checkpoint_sha256": "3" * 64,
            "config": config,
            "source_revision": source_revision,
            "tokenizer_files": tokenizer_files,
        }
        model_identity = hashlib.sha256(
            json.dumps(
                identity_payload,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode()
        ).hexdigest()
        conversion = {**identity_payload, "model_identity": model_identity}
        (path / "rwkv7_conversion.json").write_text(
            json.dumps(conversion),
            encoding="utf-8",
        )
        return path, model_identity

    return write_artifact
