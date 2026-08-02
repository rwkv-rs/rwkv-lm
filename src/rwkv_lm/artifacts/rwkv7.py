"""Canonical transformers-rwkv artifact identity and validation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class AdapterCheckpointError(ValueError):
    """Raised before an unsafe or incompatible artifact is consumed."""


@dataclass(frozen=True, slots=True)
class Rwkv7ArtifactIdentity:
    """Canonical transformers-rwkv identity consumed by training artifacts."""

    model_identity: str
    source_revision: str
    vocab_size: int


def validate_rwkv7_artifact(
    hf_assets_path: str | Path | None,
) -> Rwkv7ArtifactIdentity:
    if hf_assets_path is None:
        raise AdapterCheckpointError(
            "RWKV artifact validation requires a transformers-rwkv artifact path"
        )
    artifact_path = Path(hf_assets_path)
    conversion_path = artifact_path / "rwkv7_conversion.json"
    config_path = artifact_path / "config.json"
    try:
        conversion = json.loads(conversion_path.read_text(encoding="utf-8"))
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise AdapterCheckpointError(
            "RWKV artifact requires readable config.json and rwkv7_conversion.json"
        ) from error
    conversion = _string_mapping(conversion, "rwkv7_conversion.json")
    config = _string_mapping(config, "RWKV config.json")
    if config.get("model_type") != "rwkv7" or "Rwkv7ForCausalLM" not in config.get(
        "architectures", []
    ):
        raise AdapterCheckpointError(
            "RWKV artifact config must describe Rwkv7ForCausalLM"
        )
    vocab_size = config.get("vocab_size")
    if type(vocab_size) is not int or vocab_size <= 1:
        raise AdapterCheckpointError(
            "RWKV artifact config requires vocab_size greater than one"
        )
    normalized_config = dict(config)
    normalized_config.pop("_name_or_path", None)
    normalized_config.pop("transformers_version", None)
    if conversion.get("config") != normalized_config:
        raise AdapterCheckpointError(
            "RWKV artifact config does not match conversion metadata"
        )
    checkpoint_sha256 = _lower_hex(
        conversion.get("checkpoint_sha256"),
        length=64,
        owner="RWKV raw checkpoint digest",
    )
    source_revision = _lower_hex(
        conversion.get("source_revision"),
        length=40,
        owner="RWKV source revision",
    )
    tokenizer_files = _string_mapping(
        conversion.get("tokenizer_files"),
        "RWKV tokenizer digest inventory",
    )
    if "tokenizer.json" not in tokenizer_files:
        raise AdapterCheckpointError("RWKV conversion metadata requires tokenizer.json")
    verified_tokenizer_files = {}
    for name, raw_digest in sorted(tokenizer_files.items()):
        if Path(name).name != name:
            raise AdapterCheckpointError(
                f"RWKV tokenizer digest has an unsafe file name: {name}"
            )
        expected_digest = _lower_hex(
            raw_digest,
            length=64,
            owner=f"RWKV tokenizer digest for {name}",
        )
        tokenizer_path = artifact_path / name
        try:
            observed_digest = hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()
        except OSError as error:
            raise AdapterCheckpointError(
                f"RWKV tokenizer artifact is missing or unreadable: {name}"
            ) from error
        if observed_digest != expected_digest:
            raise AdapterCheckpointError(
                f"RWKV tokenizer artifact digest does not match: {name}"
            )
        verified_tokenizer_files[name] = expected_digest
    identity_payload = {
        "checkpoint_sha256": checkpoint_sha256,
        "config": normalized_config,
        "source_revision": source_revision,
        "tokenizer_files": verified_tokenizer_files,
    }
    encoded_identity = json.dumps(
        identity_payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    expected_identity = hashlib.sha256(encoded_identity).hexdigest()
    model_identity = _lower_hex(
        conversion.get("model_identity"),
        length=64,
        owner="RWKV model identity",
    )
    if model_identity != expected_identity:
        raise AdapterCheckpointError(
            "RWKV model identity does not match canonical conversion metadata"
        )
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            artifact_path,
            local_files_only=True,
            trust_remote_code=False,
            use_fast=True,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise AdapterCheckpointError(
            "RWKV artifact requires a valid local fast tokenizer"
        ) from error
    if not tokenizer.is_fast or len(tokenizer) != vocab_size:
        raise AdapterCheckpointError(
            "RWKV tokenizer must be fast and match the model vocabulary"
        )
    special_token_ids = {
        tokenizer.bos_token_id,
        tokenizer.eos_token_id,
        tokenizer.pad_token_id,
    }
    if special_token_ids != {0}:
        raise AdapterCheckpointError(
            "RWKV tokenizer BOS/EOS/PAD token ids must all be zero"
        )
    return Rwkv7ArtifactIdentity(
        model_identity=model_identity,
        source_revision=source_revision,
        vocab_size=vocab_size,
    )


def _string_mapping(raw: Any, owner: str) -> Mapping[str, Any]:
    if not isinstance(raw, Mapping) or any(not isinstance(key, str) for key in raw):
        raise AdapterCheckpointError(f"{owner} must be a mapping with string keys")
    return raw


def _lower_hex(raw: Any, *, length: int, owner: str) -> str:
    if (
        not isinstance(raw, str)
        or len(raw) != length
        or raw != raw.lower()
        or any(character not in "0123456789abcdef" for character in raw)
    ):
        raise AdapterCheckpointError(
            f"{owner} must be a lowercase {length}-character hexadecimal string"
        )
    return raw


__all__ = [
    "AdapterCheckpointError",
    "Rwkv7ArtifactIdentity",
    "validate_rwkv7_artifact",
]
