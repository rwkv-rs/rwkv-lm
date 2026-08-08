"""Pinned upstream revisions used by builds and exported artifacts."""

TORCHTITAN_OID = "96276d86577cf3e3bd29de72586e76af62010a55"
TRANSFORMERS_BASELINE_OID = "fe9df14d52d96ead6a5c9e03e49d3076d3ece01a"
FLASHRWKV2_OID = "8494189a426f1cfec3e623a46efb81627bef64be"
PEFT_VERSION = "0.18.0"
RWKV_PEFT_REFERENCE_OID = "5704c39f8ab1d2ac63936ab392aadb6ba526e1a5"


def dependency_metadata() -> dict[str, str]:
    return {
        "torchtitan_oid": TORCHTITAN_OID,
        "transformers_oid": TRANSFORMERS_BASELINE_OID,
        "flashrwkv2_oid": FLASHRWKV2_OID,
        "peft_version": PEFT_VERSION,
        "rwkv_peft_reference_oid": RWKV_PEFT_REFERENCE_OID,
    }


__all__ = ["dependency_metadata"]
