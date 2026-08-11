"""Pinned upstream revisions used by builds and exported artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path

TORCHTITAN_OID = "96276d86577cf3e3bd29de72586e76af62010a55"
TRANSFORMERS_OID = "204974d2a8cec8d31a57cd8c4737a2ee3a32549a"
TOKENIZERS_OID = "c5d8dde5ff49c70e4656199d5033a84e03c21b2b"
FLASHRWKV2_VERSION = "0.1.0a6"
FLASHRWKV2_OID = "255df16b85edeac69ce512bb4a5ad1122a11863d"
PEFT_VERSION = "0.19.1"
PEFT_OID = "ba6a19060d6ab54a87538a6e77e3e4d5a907375b"
RWKV_PEFT_REFERENCE_OID = "5704c39f8ab1d2ac63936ab392aadb6ba526e1a5"


def dependency_metadata() -> dict[str, str]:
    return {
        "torchtitan_oid": TORCHTITAN_OID,
        "transformers_oid": TRANSFORMERS_OID,
        "tokenizers_oid": TOKENIZERS_OID,
        "flashrwkv2_version": FLASHRWKV2_VERSION,
        "flashrwkv2_oid": FLASHRWKV2_OID,
        "peft_version": PEFT_VERSION,
        "peft_oid": PEFT_OID,
        "rwkv_peft_reference_oid": RWKV_PEFT_REFERENCE_OID,
    }


def artifact_hashes(path: str | Path) -> dict[str, str]:
    root = Path(path)
    if not root.is_dir():
        raise ValueError(f"RWKV base artifact must be a directory, got {root}.")
    result = {}
    for artifact in sorted(item for item in root.rglob("*") if item.is_file()):
        digest = hashlib.sha256()
        with artifact.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        result[str(artifact.relative_to(root))] = digest.hexdigest()
    if not result:
        raise ValueError(f"RWKV base artifact directory is empty: {root}.")
    return result


__all__ = ["artifact_hashes", "dependency_metadata"]
