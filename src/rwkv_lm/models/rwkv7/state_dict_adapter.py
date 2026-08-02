"""TorchTitan and transformers-rwkv checkpoint name conversion."""

from __future__ import annotations

from typing import Any

from torchtitan.protocols.state_dict_adapter import StateDictAdapter

from .model import Rwkv7Model

_NATIVE_TO_HF_PREFIXES = (
    ("tok_embeddings.", "model.embeddings."),
    ("layers.", "model.blocks."),
    ("norm.", "model.ln_out."),
    ("lm_head.", "head."),
)


def _rename_prefix(
    name: str,
    prefixes: tuple[tuple[str, str], ...],
) -> str:
    for source, destination in prefixes:
        if name.startswith(source):
            return destination + name.removeprefix(source)
    raise KeyError(f"unrecognized RWKV-7 state-dict key: {name}")


class Rwkv7StateDictAdapter(StateDictAdapter):
    """Map TorchTitan RWKV-7 FQNs to standard transformers-rwkv FQNs."""

    def __init__(
        self,
        model_config: Rwkv7Model.Config,
        hf_assets_path: str | None,
    ) -> None:
        super().__init__(model_config, hf_assets_path)

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        return {
            _rename_prefix(name, _NATIVE_TO_HF_PREFIXES): value
            for name, value in state_dict.items()
        }

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        hf_to_native = tuple(
            (hf_prefix, native_prefix)
            for native_prefix, hf_prefix in _NATIVE_TO_HF_PREFIXES
        )
        return {
            _rename_prefix(name, hf_to_native): value
            for name, value in hf_state_dict.items()
        }


def converter_main(argv: list[str] | None = None) -> None:
    """Delegate legacy `.pth` conversion to transformers-rwkv's standard CLI."""
    from transformers.models.rwkv7.convert_rwkv7_checkpoint_to_hf import main

    main(argv)
