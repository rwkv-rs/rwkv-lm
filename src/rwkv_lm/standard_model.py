"""Training adapter for the standard Transformers RWKV-7 model.

The model definition, recurrent state, backend dispatch, and legacy tensor-name
conversion belong to ``transformers-rwkv``.  This module owns only the small
surface that the rwkv-lm training runner needs around that public model.
"""

from __future__ import annotations

import argparse
import copy
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
from torch.nn import functional as F

from .activation_checkpointing import FSDP2ActivationCheckpointing
from .checkpoint import CheckpointContractError
from .infctx import (
    InfctxBoundary,
    InfctxContractError,
    InfctxResult,
    InfctxState,
    recurrent_chunk_forward,
)
from .peft import (
    LoraConfig,
    PeftContractError,
    bind_lora_base_provenance,
    build_lora_delta,
    freeze_base_for_lora,
    load_lora_adapter,
    lora_merged_state_dict,
    save_lora_adapter,
)

_CONFIG_MODULE = "transformers.models.rwkv7.configuration_rwkv7"
_MODEL_MODULE = "transformers.models.rwkv7.modeling_rwkv7"
_CONVERTER_MODULE = "transformers.models.rwkv7.convert_rwkv7_checkpoint_to_hf"
_STANDARD_RUNTIME_REVISIONS = {
    "transformers": "2696927df9363b5fa175076bb827ba4da2c4e581",
    "flash-linear-attention": "a4a8aa98df6ec5322f194a80ec57363dd045adfc",
    "flash-rwkv": "866aafd2eed146b0eda1ce03444009ae030f89e3",
}
_STANDARD_LORA_TARGETS = {
    "time_mix.receptance": ("att", "receptance"),
    "time_mix.key": ("att", "key"),
    "time_mix.value": ("att", "value"),
    "time_mix.output": ("att", "output"),
    "channel_mix.key": ("ffn", "key"),
    "channel_mix.value": ("ffn", "value"),
}
_STANDARD_LORA_BASE_CONFIG_FIELDS = (
    "bos_token_id",
    "context_length",
    "eos_token_id",
    "head_size",
    "hidden_size",
    "intermediate_size",
    "model_type",
    "num_attention_heads",
    "num_hidden_layers",
    "use_cache",
    "vocab_size",
    "wkv_backend",
    "wkv_state_dtype",
)


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
            vocab_size=args.vocab_size,
            context_length=args.ctx_len,
            hidden_size=args.n_embd,
            intermediate_size=args.dim_ffn,
            num_hidden_layers=args.n_layer,
            head_size=args.head_size,
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


def configure_standard_rwkv7_peft(
    model: nn.Module,
    config: LoraConfig,
    *,
    adapter_path: Path | None = None,
) -> nn.Module:
    """Attach the existing RWKV LoRA contract to Transformers projections.

    The public model keeps ownership of every base projection.  Selected LoRA
    deltas are registered as sibling modules and contribute through projection
    hooks, so adapter artifacts and merge logic retain their existing canonical
    parameter names without copying the Transformers implementation.
    """

    if not isinstance(config, LoraConfig):
        raise PeftContractError("standard RWKV-7 PEFT requires LoraConfig")
    if hasattr(model, "_standard_rwkv7_lora_hook_handles"):
        raise PeftContractError("standard RWKV-7 model already has PEFT configured")
    bind_lora_base_provenance(
        model,
        model_config={
            name: getattr(model.config, name)
            for name in _STANDARD_LORA_BASE_CONFIG_FIELDS
        },
        source_revision=_STANDARD_RUNTIME_REVISIONS["transformers"],
    )
    model.lora_config = config
    handles = []
    if config.enabled:
        for block_id, block in enumerate(standard_rwkv7_blocks(model)):
            for target in config.target_modules:
                owner_name, projection_name = _STANDARD_LORA_TARGETS[target]
                owner = getattr(block, owner_name, None)
                projection = getattr(owner, projection_name, None)
                if not isinstance(projection, nn.Linear):
                    raise PeftContractError(
                        "standard RWKV-7 LoRA target is not a linear projection: "
                        f"model.blocks.{block_id}.{owner_name}.{projection_name}"
                    )
                adapter_name = f"{projection_name}_lora"
                if hasattr(owner, adapter_name):
                    raise PeftContractError(
                        "standard RWKV-7 LoRA target is already configured: "
                        f"model.blocks.{block_id}.{owner_name}.{projection_name}"
                    )
                adapter = build_lora_delta(
                    config,
                    target,
                    in_features=projection.in_features,
                    out_features=projection.out_features,
                )
                setattr(owner, adapter_name, adapter)
                handles.append(
                    projection.register_forward_hook(
                        partial(_add_lora_projection_output, adapter=adapter)
                    )
                )
        freeze_base_for_lora(model, config)
        if adapter_path is not None:
            load_lora_adapter(model, Path(adapter_path))
    elif adapter_path is not None:
        raise PeftContractError(
            "loading a LoRA adapter requires an enabled standard model"
        )
    model._standard_rwkv7_lora_hook_handles = handles
    return model


def save_standard_rwkv7_lora_adapter(model: nn.Module, path: Path) -> Path:
    """Save the adapter-only artifact shared with the existing PEFT contract."""

    return save_lora_adapter(model, Path(path))


def save_standard_rwkv7_merged_model(
    model: nn.Module,
    destination: Path,
) -> Path:
    """Merge LoRA deltas into a strict-loadable Transformers model directory."""

    merged_state = lora_merged_state_dict(model)
    bindings = load_standard_rwkv7_bindings()
    merged_model = bindings.model_type(copy.deepcopy(model.config))
    incompatible = merged_model.load_state_dict(merged_state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise PeftContractError(
            "merged standard RWKV-7 state did not strict-load into a base model"
        )
    return save_standard_rwkv7_model(merged_model, Path(destination))


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
    model._fsdp2_activation_checkpointing = policy
    return blocks


def standard_rwkv7_training_loss(
    model: nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    *,
    train_type: str = "standard",
    chunk_ctx: int = 0,
    ctx_len: int | None = None,
) -> torch.Tensor:
    """Run SFT/pretraining loss with the existing explicit-target dataset."""

    if train_type == "standard":
        outputs = model(
            input_ids=input_ids,
            use_cache=False,
            return_dict=True,
        )
        logits = getattr(outputs, "logits", None)
    elif train_type == "infctx":
        if ctx_len is None:
            raise InfctxContractError("standard RWKV-7 infctx requires ctx_len")
        logits = standard_rwkv7_infctx_forward(
            model,
            input_ids,
            chunk_ctx=chunk_ctx,
            ctx_len=ctx_len,
            boundary=InfctxBoundary.RESET,
        ).output
    else:
        raise StandardModelContractError(
            f"unsupported standard RWKV-7 train_type: {train_type!r}"
        )
    if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
        raise StandardModelContractError(
            "standard RWKV-7 forward must return [B, T, vocab] logits"
        )
    if labels.shape != input_ids.shape or logits.shape[:2] != labels.shape:
        raise StandardModelContractError(
            "standard RWKV-7 labels and logits must preserve the input [B, T] axes"
        )
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=-100,
    )


def standard_rwkv7_infctx_forward(
    model: nn.Module,
    input_ids: torch.Tensor,
    *,
    chunk_ctx: int,
    ctx_len: int,
    boundary: InfctxBoundary | str,
    state: InfctxState | None = None,
) -> InfctxResult:
    """Run the public Transformers recurrent state through TBPTT boundaries."""

    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
        raise InfctxContractError(
            "standard RWKV-7 infctx input_ids must have shape [B, T]"
        )
    batch_size = input_ids.shape[0]
    return recurrent_chunk_forward(
        input_ids,
        chunk_ctx=chunk_ctx,
        ctx_len=ctx_len,
        boundary=boundary,
        state=state,
        reset_state=lambda: _reset_standard_rwkv7_infctx_state(
            model,
            batch_size=batch_size,
            device=input_ids.device,
        ),
        validate_state=lambda candidate: _validate_standard_rwkv7_infctx_state(
            model,
            candidate,
            batch_size=batch_size,
            device=input_ids.device,
        ),
        forward_chunk=lambda chunk, candidate: _forward_standard_rwkv7_chunk(
            model,
            chunk,
            candidate,
        ),
    )


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


def _add_lora_projection_output(
    _projection: nn.Module,
    inputs: tuple[object, ...],
    output: torch.Tensor,
    *,
    adapter: nn.Module,
) -> torch.Tensor:
    if len(inputs) != 1 or not isinstance(inputs[0], torch.Tensor):
        raise PeftContractError(
            "standard RWKV-7 LoRA projection must receive one Torch tensor"
        )
    if not isinstance(output, torch.Tensor):
        raise PeftContractError(
            "standard RWKV-7 LoRA projection must return one Torch tensor"
        )
    return output + adapter(inputs[0])


def _reset_standard_rwkv7_infctx_state(
    model: nn.Module,
    *,
    batch_size: int,
    device: torch.device,
) -> InfctxState:
    config = getattr(model, "config", None)
    embeddings = getattr(getattr(model, "model", None), "embeddings", None)
    if not isinstance(embeddings, nn.Embedding):
        raise InfctxContractError(
            "standard RWKV-7 infctx requires model.embeddings"
        )
    layers = int(getattr(config, "num_hidden_layers", 0))
    hidden_size = int(getattr(config, "hidden_size", 0))
    num_heads = int(getattr(config, "num_attention_heads", 0))
    head_size = int(getattr(config, "head_size", 0))
    if (
        layers <= 0
        or hidden_size <= 0
        or num_heads <= 0
        or head_size <= 0
        or num_heads * head_size != hidden_size
    ):
        raise InfctxContractError(
            "standard RWKV-7 infctx config has an invalid recurrent layout"
        )
    return InfctxState(
        shift_states=torch.zeros(
            layers,
            2,
            batch_size,
            hidden_size,
            dtype=_standard_rwkv7_hidden_dtype(model, device),
            device=device,
        ),
        wkv_states=torch.zeros(
            layers,
            batch_size,
            num_heads,
            head_size,
            head_size,
            dtype=torch.float32,
            device=device,
        ),
        tokens_seen=0,
    )


def _validate_standard_rwkv7_infctx_state(
    model: nn.Module,
    state: InfctxState,
    *,
    batch_size: int,
    device: torch.device,
) -> None:
    if not isinstance(state, InfctxState):
        raise InfctxContractError(
            "standard RWKV-7 recurrent state must be InfctxState"
        )
    config = model.config
    embeddings = getattr(getattr(model, "model", None), "embeddings", None)
    if not isinstance(embeddings, nn.Embedding):
        raise InfctxContractError(
            "standard RWKV-7 infctx requires model.embeddings"
        )
    expected_shift = (
        int(config.num_hidden_layers),
        2,
        batch_size,
        int(config.hidden_size),
    )
    expected_wkv = (
        int(config.num_hidden_layers),
        batch_size,
        int(config.num_attention_heads),
        int(config.head_size),
        int(config.head_size),
    )
    if tuple(state.shift_states.shape) != expected_shift:
        raise InfctxContractError(
            "standard RWKV-7 infctx shift state shape is invalid"
        )
    if tuple(state.wkv_states.shape) != expected_wkv:
        raise InfctxContractError(
            "standard RWKV-7 infctx WKV state shape is invalid"
        )
    if state.shift_states.device != device or state.wkv_states.device != device:
        raise InfctxContractError(
            "standard RWKV-7 infctx state must be on the input device"
        )
    if state.shift_states.dtype != _standard_rwkv7_hidden_dtype(model, device):
        raise InfctxContractError(
            "standard RWKV-7 infctx shift state dtype must match embeddings"
        )
    if state.wkv_states.dtype != torch.float32:
        raise InfctxContractError(
            "standard RWKV-7 infctx WKV state must use FP32"
        )
    if not state.shift_states.is_contiguous() or not state.wkv_states.is_contiguous():
        raise InfctxContractError(
            "standard RWKV-7 infctx state must be contiguous"
        )


def _forward_standard_rwkv7_chunk(
    model: nn.Module,
    input_ids: torch.Tensor,
    state: InfctxState,
) -> InfctxResult:
    provider_state = (
        state.shift_states[:, 0],
        state.wkv_states,
        state.shift_states[:, 1],
    )
    outputs = model(
        input_ids=input_ids,
        state=provider_state,
        use_cache=True,
        return_dict=True,
    )
    logits = getattr(outputs, "logits", None)
    next_state = getattr(outputs, "state", None)
    if not isinstance(logits, torch.Tensor):
        raise InfctxContractError(
            "standard RWKV-7 infctx forward must return logits"
        )
    if (
        not isinstance(next_state, (tuple, list))
        or len(next_state) != 3
        or any(not isinstance(value, torch.Tensor) for value in next_state)
    ):
        raise InfctxContractError(
            "standard RWKV-7 infctx forward must return attention, WKV, and "
            "FFN state tensors"
        )
    attention_shift, wkv_state, ffn_shift = next_state
    return InfctxResult(
        output=logits,
        state=InfctxState(
            shift_states=torch.stack(
                (attention_shift, ffn_shift),
                dim=1,
            ).contiguous(),
            wkv_states=wkv_state.contiguous(),
            tokens_seen=state.tokens_seen + input_ids.shape[1],
        ),
    )


def _standard_rwkv7_hidden_dtype(
    model: nn.Module,
    device: torch.device,
) -> torch.dtype:
    if torch.is_autocast_enabled(device.type):
        return torch.get_autocast_dtype(device.type)
    embeddings = getattr(getattr(model, "model", None), "embeddings", None)
    if not isinstance(embeddings, nn.Embedding):
        raise InfctxContractError(
            "standard RWKV-7 infctx requires model.embeddings"
        )
    return embeddings.weight.dtype


def _import_required_module(name: str) -> ModuleType:
    try:
        return import_module(name)
    except ImportError as error:
        requirements = ", ".join(
            f"{package}@{revision}"
            for package, revision in _STANDARD_RUNTIME_REVISIONS.items()
        )
        raise ImportError(
            "standard RWKV-7 runtime requires exact revisions "
            f"({requirements}); failed to import {name}: {error}"
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
    "configure_standard_rwkv7_peft",
    "convert_legacy_rwkv7_checkpoint",
    "create_standard_rwkv7_model",
    "load_standard_rwkv7_bindings",
    "prepare_standard_rwkv7_for_fsdp2",
    "save_standard_rwkv7_lora_adapter",
    "save_standard_rwkv7_merged_model",
    "save_standard_rwkv7_model",
    "standard_rwkv7_blocks",
    "standard_rwkv7_infctx_forward",
    "standard_rwkv7_optimizer_groups",
    "standard_rwkv7_training_loss",
]
