"""Pinned upstream revisions used by builds and exported artifacts."""

TORCHTITAN_OID = "96276d86577cf3e3bd29de72586e76af62010a55"
TRANSFORMERS_OID = "10052072bd3ca172957d97e11b24ae297e0e0072"
FLASHRWKV2_VERSION = "0.1.0a5"
FLASHRWKV2_OID = "046257e7918d93a0fefce868e2ab580fbf6078da"
PEFT_VERSION = "0.18.0"
RWKV_PEFT_REFERENCE_OID = "5704c39f8ab1d2ac63936ab392aadb6ba526e1a5"


def dependency_metadata() -> dict[str, str]:
    return {
        "torchtitan_oid": TORCHTITAN_OID,
        "transformers_oid": TRANSFORMERS_OID,
        "flashrwkv2_version": FLASHRWKV2_VERSION,
        "flashrwkv2_oid": FLASHRWKV2_OID,
        "peft_version": PEFT_VERSION,
        "rwkv_peft_reference_oid": RWKV_PEFT_REFERENCE_OID,
    }


__all__ = ["dependency_metadata"]
