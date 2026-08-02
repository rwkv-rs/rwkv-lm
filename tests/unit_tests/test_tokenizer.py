import json
from pathlib import Path

import pytest

from rwkv_lm.artifacts.rwkv7 import AdapterCheckpointError
from rwkv_lm.components.tokenizer import RwkvPretokenizedTokenizer


def _tokenizer(
    tokenizer_path: Path,
    vocab_size: int = 1_024,
) -> RwkvPretokenizedTokenizer:
    return RwkvPretokenizedTokenizer.Config(vocab_size=vocab_size).build(
        tokenizer_path=str(tokenizer_path)
    )


def test_pretokenized_tokenizer_requires_canonical_assets_and_rejects_text(
    tmp_path: Path,
    rwkv7_artifact_factory,
) -> None:
    artifact_path, model_identity = rwkv7_artifact_factory()
    tokenizer = _tokenizer(artifact_path)

    assert tokenizer.model_identity == model_identity
    assert tokenizer.get_vocab_size() == 1_024
    with pytest.raises(RuntimeError, match="pretokenized-only"):
        tokenizer.encode("must not be byte-mapped")
    with pytest.raises(RuntimeError, match="pretokenized-only"):
        tokenizer.decode([1, 2, 3])
    with pytest.raises(AdapterCheckpointError, match="requires readable"):
        _tokenizer(tmp_path / "missing")


def test_pretokenized_tokenizer_rejects_tampered_assets(
    rwkv7_artifact_factory,
) -> None:
    artifact_path, _model_identity = rwkv7_artifact_factory()
    (artifact_path / "tokenizer.json").write_text("{}", encoding="utf-8")

    with pytest.raises(AdapterCheckpointError, match="digest does not match"):
        _tokenizer(artifact_path)


def test_pretokenized_tokenizer_rejects_forged_model_identity(
    rwkv7_artifact_factory,
) -> None:
    artifact_path, _model_identity = rwkv7_artifact_factory()
    conversion_path = artifact_path / "rwkv7_conversion.json"
    conversion = json.loads(conversion_path.read_text(encoding="utf-8"))
    conversion["model_identity"] = "f" * 64
    conversion_path.write_text(json.dumps(conversion), encoding="utf-8")

    with pytest.raises(AdapterCheckpointError, match="canonical conversion"):
        _tokenizer(artifact_path)
