"""Training adapter for the standard Transformers RWKV-7 model.

The model definition, recurrent state, backend dispatch, and legacy tensor-name
conversion belong to ``transformers-rwkv``.  This module owns only the small
surface that the rwkv-lm training runner needs around that public model.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from functools import partial
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import Any

import torch
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)

from .activation_checkpointing import FSDP2ActivationCheckpointing
from .checkpoint import CheckpointContractError

_CONFIG_MODULE = "transformers.models.rwkv7.configuration_rwkv7"
_MODEL_MODULE = "transformers.models.rwkv7.modeling_rwkv7"
_CONVERTER_MODULE = "transformers.models.rwkv7.convert_rwkv7_checkpoint_to_hf"


class StandardModelContractError(CheckpointContractError):
    """Raised when the standard RWKV-7 model interface is unavailable or invalid."""


@dataclass(frozen=True)
class StandardRwkv7Bindings:
    """Imported public classes and converter supplied by transformers-rwkv."""

    config_type: type
    model_type: type[nn.Module]
    convert_checkpoint: Callable[..., Mapping[str, object]] | None = None


@dataclass(frozen=True)
class StandardRwkv7Config:
    """Architecture fields translated from the existing rwkv-lm CLI contract."""

    vocab_size: int
    context_length: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    head_size: int
    wkv_backend: str

    def __post_init__(self) -> None:
        for name in (
            "vocab_size",
            "context_length",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "head_size",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise StandardModelContractError(
                    f"standard RWKV-7 {name} must be a positive integer"
                )
        if self.hidden_size % self.head_size:
            raise StandardModelContractError(
                "standard RWKV-7 hidden_size must be divisible by head_size"
            )
        if self.wkv_backend not in {"reference", "flash_rwkv"}:
            raise StandardModelContractError(
                "standard RWKV-7 training requires an explicit reference or "
                "flash_rwkv backend"
            )

    @classmethod
    def from_namespace(cls, args: object) -> StandardRwkv7Config:
        return cls(
            vocab_size=getattr(args, "vocab_size"),
            context_length=getattr(args, "ctx_len"),
            hidden_size=getattr(args, "n_embd"),
            intermediate_size=getattr(args, "dim_ffn"),
            num_hidden_layers=getattr(args, "n_layer"),
            head_size=getattr(args, "head_size"),
            wkv_backend=getattr(args, "wkv_backend", "flash_rwkv"),
        )

    def external_kwargs(self) -> dict[str, object]:
        return {
            "bos_token_id": 0,
            "context_length": self.context_length,
            "eos_token_id": 0,
            "head_size": self.head_size,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_attention_heads": self.hidden_size // self.head_size,
            "num_hidden_layers": self.num_hidden_layers,
            "pad_token_id": 0,
            "use_cache": True,
            "vocab_size": self.vocab_size,
            "wkv_backend": self.wkv_backend,
            "wkv_state_dtype": "float32",
        }


def load_standard_rwkv7_bindings(
    *,
    require_converter: bool = False,
) -> StandardRwkv7Bindings:
    """Import the published transformers-rwkv interface or fail closed."""

    config_module = _import_required_module(_CONFIG_MODULE)
    model_module = _import_required_module(_MODEL_MODULE)
    config_type = _required_attribute(config_module, "Rwkv7Config")
    model_type = _required_attribute(model_module, "Rwkv7ForCausalLM")
    if not isinstance(config_type, type) or not isinstance(model_type, type):
        raise StandardModelContractError(
            "transformers-rwkv RWKV-7 config and model exports must be classes"
        )
    if getattr(config_type, "model_type", None) != "rwkv7":
        raise StandardModelContractError(
            "transformers-rwkv Rwkv7Config must declare model_type='rwkv7'"
        )
    if not issubclass(model_type, nn.Module):
        raise StandardModelContractError(
            "transformers-rwkv Rwkv7ForCausalLM must be a Torch module"
        )

    converter = None
    if require_converter:
        converter_module = _import_required_module(_CONVERTER_MODULE)
        converter = _required_attribute(
            converter_module,
            "convert_rwkv7_checkpoint_to_hf_format",
        )
        if not callable(converter):
            raise StandardModelContractError(
                "transformers-rwkv legacy checkpoint converter must be callable"
            )
    return StandardRwkv7Bindings(
        config_type=config_type,
        model_type=model_type,
        convert_checkpoint=converter,
    )


def create_standard_rwkv7_model(
    args: object,
    *,
    model_source: Path | None = None,
) -> nn.Module:
    """Create or strict-load the one standard model owned by the training runner."""

    requested = StandardRwkv7Config.from_namespace(args)
    source = None if model_source is None else Path(model_source)
    if source is not None and source.suffix == ".pth":
        raise StandardModelContractError(
            "legacy .pth cannot be loaded by the training runner; convert it "
            "with rwkv-convert-legacy-checkpoint first"
        )
    bindings = load_standard_rwkv7_bindings()
    if model_source is None:
        config = bindings.config_type(**requested.external_kwargs())
        model = bindings.model_type(config)
    else:
        assert source is not None
        if source.is_symlink() or not source.is_dir():
            raise StandardModelContractError(
                f"standard RWKV-7 model source must be a regular directory: {source}"
            )
        if not (source / "config.json").is_file():
            raise StandardModelContractError(
                f"standard RWKV-7 model source is missing config.json: {source}"
            )
        from_pretrained = getattr(bindings.model_type, "from_pretrained", None)
        if not callable(from_pretrained):
            raise StandardModelContractError(
                "transformers-rwkv Rwkv7ForCausalLM is missing from_pretrained"
            )
        model = from_pretrained(str(source))
    _require_standard_model(model, requested)
    return model


def save_standard_rwkv7_model(model: nn.Module, destination: Path) -> Path:
    """Save the canonical model with the Transformers serialization contract."""

    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        raise StandardModelContractError(
            f"standard RWKV-7 model destination exists: {destination}"
        )
    save_pretrained = getattr(model, "save_pretrained", None)
    if not callable(save_pretrained):
        raise StandardModelContractError(
            "standard RWKV-7 model is missing save_pretrained"
        )
    save_pretrained(str(destination), safe_serialization=True)
    if not (destination / "config.json").is_file():
        raise StandardModelContractError(
            "standard RWKV-7 save did not produce config.json"
        )
    return destination


def standard_rwkv7_blocks(model: nn.Module) -> tuple[nn.Module, ...]:
    """Return the Transformers-owned RWKV blocks selected by FSDP2."""

    base_model = getattr(model, "model", None)
    blocks = getattr(base_model, "blocks", None)
    if not isinstance(blocks, Iterable) or isinstance(blocks, (str, bytes)):
        raise StandardModelContractError(
            "standard RWKV-7 model must expose model.blocks"
        )
    result = tuple(blocks)
    if not result or any(not isinstance(block, nn.Module) for block in result):
        raise StandardModelContractError(
            "standard RWKV-7 model.blocks must contain Torch modules"
        )
    return result


def prepare_standard_rwkv7_for_fsdp2(
    model: nn.Module,
    *,
    activation_checkpointing: bool,
) -> tuple[nn.Module, ...]:
    """Apply non-reentrant checkpoint wrappers to exactly the standard blocks."""

    blocks = standard_rwkv7_blocks(model)
    if activation_checkpointing:
        selected = FSDP2ActivationCheckpointing.for_rwkv_blocks(
            blocks,
            enabled=True,
        )
        apply_activation_checkpointing(
            model,
            checkpoint_wrapper_fn=partial(
                checkpoint_wrapper,
                checkpoint_impl=CheckpointImpl.NO_REENTRANT,
            ),
            check_fn=selected.selects,
        )
        blocks = standard_rwkv7_blocks(model)
    policy = FSDP2ActivationCheckpointing.for_rwkv_blocks(
        blocks,
        enabled=activation_checkpointing,
    )
    policy.require_rwkv_blocks(blocks, enabled=activation_checkpointing)
    setattr(model, "_fsdp2_activation_checkpointing", policy)
    return blocks


def standard_rwkv7_training_loss(
    model: nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Run the public CausalLM loss path used by both SFT and pretraining."""

    outputs = model(
        input_ids=input_ids,
        labels=labels,
        use_cache=False,
        return_dict=True,
    )
    loss = getattr(outputs, "loss", None)
    if not isinstance(loss, torch.Tensor) or loss.ndim != 0:
        raise StandardModelContractError(
            "standard RWKV-7 forward must return a scalar loss"
        )
    return loss


def standard_rwkv7_optimizer_groups(
    model: nn.Module,
    *,
    weight_decay: float,
    is_global_zero: bool = False,
) -> list[dict[str, object]]:
    """Preserve RWKV's decay and doubled time-decay learning-rate groups."""

    decay: list[str] = []
    regular: list[str] = []
    doubled: list[str] = []
    parameters = dict(model.named_parameters())
    for name, parameter in parameters.items():
        if not parameter.requires_grad:
            continue
        if ".att.w0" in name:
            doubled.append(name)
        elif parameter.squeeze().ndim >= 2 and weight_decay > 0 and name.endswith(
            ".weight"
        ):
            decay.append(name)
        else:
            regular.append(name)
    groups: list[dict[str, object]] = []
    for names, decay_value, scale in (
        (sorted(regular), 0.0, 1.0),
        (sorted(doubled), 0.0, 2.0),
        (sorted(decay), weight_decay, 1.0),
    ):
        if names:
            groups.append(
                {
                    "params": [parameters[name] for name in names],
                    "weight_decay": decay_value,
                    "my_lr_scale": scale,
                }
            )
    if not groups:
        raise StandardModelContractError(
            "standard RWKV-7 optimizer has no trainable parameters"
        )
    if is_global_zero:
        print("decay", sorted(decay))
        print("1x", sorted(regular))
        print("2x", sorted(doubled))
    return groups


def convert_legacy_rwkv7_checkpoint(
    checkpoint: Path,
    output_dir: Path,
    *,
    dtype: str | None = None,
    tokenizer_name_or_path: str | None = None,
    wkv_backend: str = "flash_rwkv",
) -> Mapping[str, object]:
    """Delegate legacy tensor conversion to the standard model's converter."""

    source = Path(checkpoint)
    destination = Path(output_dir)
    if source.is_symlink() or not source.is_file() or source.suffix != ".pth":
        raise StandardModelContractError(
            f"legacy RWKV-7 checkpoint must be a regular .pth file: {source}"
        )
    if destination.exists() or destination.is_symlink():
        raise StandardModelContractError(
            f"standard RWKV-7 conversion destination exists: {destination}"
        )
    if wkv_backend not in {"reference", "flash_rwkv"}:
        raise StandardModelContractError(
            "legacy conversion requires an explicit reference or flash_rwkv backend"
        )
    bindings = load_standard_rwkv7_bindings(require_converter=True)
    assert bindings.convert_checkpoint is not None
    result = bindings.convert_checkpoint(
        str(source),
        str(destination),
        dtype=dtype,
        safe_serialization=True,
        tokenizer_name_or_path=tokenizer_name_or_path,
        wkv_backend=wkv_backend,
    )
    if not isinstance(result, Mapping):
        raise StandardModelContractError(
            "transformers-rwkv converter must return a result mapping"
        )
    if not (destination / "config.json").is_file():
        raise StandardModelContractError(
            "transformers-rwkv converter did not produce config.json"
        )
    return result


def converter_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Convert a legacy RWKV-7 .pth into a standard Transformers checkpoint."
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"))
    parser.add_argument("--tokenizer-name-or-path")
    parser.add_argument(
        "--wkv-backend",
        choices=("reference", "flash_rwkv"),
        default="flash_rwkv",
    )
    args = parser.parse_args(argv)
    result = convert_legacy_rwkv7_checkpoint(
        args.checkpoint,
        args.output_dir,
        dtype=args.dtype,
        tokenizer_name_or_path=args.tokenizer_name_or_path,
        wkv_backend=args.wkv_backend,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def _require_standard_model(
    model: object,
    requested: StandardRwkv7Config,
) -> None:
    if not isinstance(model, nn.Module):
        raise StandardModelContractError(
            "transformers-rwkv model factory must return a Torch module"
        )
    config = getattr(model, "config", None)
    if getattr(config, "model_type", None) != "rwkv7":
        raise StandardModelContractError(
            "standard model config must declare model_type='rwkv7'"
        )
    expected = requested.external_kwargs()
    mismatches = {
        name: (getattr(config, name, None), value)
        for name, value in expected.items()
        if getattr(config, name, None) != value
    }
    if mismatches:
        details = ", ".join(
            f"{name}={actual!r} (expected {wanted!r})"
            for name, (actual, wanted) in sorted(mismatches.items())
        )
        raise StandardModelContractError(
            "standard RWKV-7 config does not match the training request: " + details
        )
    standard_rwkv7_blocks(model)
    if not callable(getattr(model, "forward", None)):
        raise StandardModelContractError(
            "standard RWKV-7 model must expose forward"
        )


def _import_required_module(name: str) -> ModuleType:
    try:
        return import_module(name)
    except ImportError as error:
        raise StandardModelContractError(
            "standard RWKV-7 support requires transformers-rwkv with the "
            "Rwkv7Config/Rwkv7ForCausalLM interface and its fla-rwkv/FlashRWKV "
            f"dependencies; failed to import {name}: {error}"
        ) from error


def _required_attribute(module: ModuleType, name: str) -> Any:
    try:
        return getattr(module, name)
    except AttributeError as error:
        raise StandardModelContractError(
            f"standard RWKV-7 dependency {module.__name__} is missing {name}"
        ) from error


__all__ = [
    "StandardModelContractError",
    "StandardRwkv7Bindings",
    "StandardRwkv7Config",
    "convert_legacy_rwkv7_checkpoint",
    "create_standard_rwkv7_model",
    "load_standard_rwkv7_bindings",
    "prepare_standard_rwkv7_for_fsdp2",
    "save_standard_rwkv7_model",
    "standard_rwkv7_blocks",
    "standard_rwkv7_optimizer_groups",
    "standard_rwkv7_training_loss",
]
