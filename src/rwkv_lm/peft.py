"""RWKV LoRA configuration, trainable-state, and artifact contracts."""

from __future__ import annotations

import math
import os
import pickle
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

LORA_TARGET_MODULES = (
    "time_mix.receptance",
    "time_mix.key",
    "time_mix.value",
    "time_mix.output",
    "channel_mix.key",
    "channel_mix.value",
)
_ADAPTER_FORMAT = "rwkv-lora-adapter"
_ADAPTER_SCHEMA_VERSION = 1
_CHECKPOINT_WRAPPER_SEGMENT = "._checkpoint_wrapped_module"


class PeftContractError(ValueError):
    """Raised when a PEFT request would be ambiguous or incomplete."""


@dataclass(frozen=True)
class LoraConfig:
    """Complete, canonical LoRA configuration owned by the RWKV model."""

    rank: int = 0
    alpha: float = 0.0
    dropout: float = 0.0
    target_modules: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.rank, bool) or not isinstance(self.rank, int):
            raise PeftContractError("LoRA rank must be an integer")
        if self.rank < 0:
            raise PeftContractError("LoRA rank must be non-negative")
        alpha = _finite_float(self.alpha, "alpha")
        dropout = _finite_float(self.dropout, "dropout")
        if dropout < 0 or dropout >= 1:
            raise PeftContractError("LoRA dropout must be in the range [0, 1)")
        targets = _canonical_targets(self.target_modules)
        if self.rank == 0:
            if alpha != 0 or dropout != 0 or targets:
                raise PeftContractError(
                    "disabled LoRA requires alpha=0, dropout=0, and no targets"
                )
        else:
            if alpha <= 0:
                raise PeftContractError("enabled LoRA requires alpha > 0")
            if not targets:
                raise PeftContractError(
                    "enabled LoRA requires at least one explicit target module"
                )
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "dropout", dropout)
        object.__setattr__(self, "target_modules", targets)

    @property
    def enabled(self) -> bool:
        return self.rank > 0

    @classmethod
    def from_namespace(cls, args: object) -> LoraConfig:
        """Read the ordinary CLI/model fields without relying on globals."""

        return cls(
            rank=getattr(args, "lora_rank", 0),
            alpha=getattr(args, "lora_alpha", 0.0),
            dropout=getattr(args, "lora_dropout", 0.0),
            target_modules=_target_values(getattr(args, "lora_target_modules", ())),
        )

    @classmethod
    def from_dict(cls, raw: object) -> LoraConfig:
        value = _exact_mapping(
            raw,
            {"rank", "alpha", "dropout", "target_modules"},
            "LoRA config",
        )
        return cls(
            rank=value["rank"],
            alpha=value["alpha"],
            dropout=value["dropout"],
            target_modules=_target_values(value["target_modules"]),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "target_modules": list(self.target_modules),
        }


class LoraDelta(nn.Module):
    """One trainable low-rank delta; the frozen base projection stays separate."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: LoraConfig,
        target_module: str,
    ) -> None:
        super().__init__()
        if not config.enabled or target_module not in config.target_modules:
            raise PeftContractError("LoraDelta requires an enabled selected target")
        self.enabled = True
        self.target_module = target_module
        self.rank = config.rank
        self.scaling = config.alpha / config.rank
        self.dropout = nn.Dropout(config.dropout)
        self.lora_A = nn.Parameter(torch.empty(config.rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, config.rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = F.linear(self.dropout(inputs), self.lora_A)
        return F.linear(hidden, self.lora_B) * self.scaling


class DisabledLoraDelta(nn.Module):
    """Parameter-free marker used when a projection is not selected."""

    def __init__(self) -> None:
        super().__init__()
        self.enabled = False

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("a disabled LoRA delta cannot be evaluated")


def build_lora_delta(
    config: LoraConfig,
    target_module: str,
    *,
    in_features: int,
    out_features: int,
) -> LoraDelta | DisabledLoraDelta:
    """Build an adapter only for an explicitly selected projection."""

    if target_module not in LORA_TARGET_MODULES:
        raise PeftContractError(f"unsupported LoRA target module: {target_module}")
    if config.enabled and target_module in config.target_modules:
        return LoraDelta(in_features, out_features, config, target_module)
    return DisabledLoraDelta()


def freeze_base_for_lora(model: nn.Module, config: LoraConfig) -> None:
    """Freeze every base parameter and expose only selected adapter parameters."""

    if not config.enabled:
        return
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(_is_lora_parameter_name(name))
    validate_lora_trainable_parameters(model, config)


def validate_lora_trainable_parameters(
    model: nn.Module,
    config: LoraConfig,
) -> tuple[str, ...]:
    """Return adapter names after proving no base parameter is trainable."""

    adapter_names = set(lora_parameter_names(model))
    if config.enabled and not adapter_names:
        raise PeftContractError("enabled LoRA did not create adapter parameters")
    unexpected = sorted(
        _canonical_parameter_name(name)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and _canonical_parameter_name(name) not in adapter_names
    )
    frozen_adapters = sorted(
        _canonical_parameter_name(name)
        for name, parameter in model.named_parameters()
        if _canonical_parameter_name(name) in adapter_names
        and not parameter.requires_grad
    )
    if unexpected:
        raise PeftContractError(
            "LoRA left base parameters trainable: " + ", ".join(unexpected)
        )
    if frozen_adapters:
        raise PeftContractError(
            "LoRA adapter parameters are frozen: " + ", ".join(frozen_adapters)
        )
    return tuple(sorted(adapter_names))


def lora_parameter_names(model: nn.Module) -> tuple[str, ...]:
    """Return the exact trainable tensor names owned by all LoRA deltas."""

    return tuple(
        sorted(
            _canonical_parameter_name(name)
            for name, _parameter in model.named_parameters()
            if _is_lora_parameter_name(name)
        )
    )


def lora_adapter_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Copy only adapter tensors to CPU for a standalone artifact."""

    names = lora_parameter_names(model)
    if not names:
        raise PeftContractError("cannot serialize an adapter from LoRA-disabled model")
    parameters = _canonical_parameter_map(model)
    return {
        name: parameters[name].detach().cpu().contiguous().clone() for name in names
    }


def lora_base_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return the model state excluding adapter tensors.

    The returned tensors have the same reference semantics as ``state_dict``;
    callers that need an immutable snapshot should clone them.
    """

    adapter_names = set(lora_parameter_names(model))
    base_state = {}
    for name, tensor in model.state_dict().items():
        canonical_name = _canonical_parameter_name(name)
        if canonical_name in base_state:
            raise PeftContractError(
                f"LoRA base state contains duplicate canonical key: {canonical_name}"
            )
        if canonical_name not in adapter_names:
            base_state[canonical_name] = tensor
    return base_state


def lora_merged_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return standalone base weights with every trained LoRA delta merged."""

    config = _model_lora_config(model)
    validate_lora_trainable_parameters(model, config)
    merged_state = {
        name: tensor.detach().cpu().contiguous().clone()
        for name, tensor in lora_base_state_dict(model).items()
    }
    modules = dict(model.named_modules())
    merged_projection_names = set()
    for adapter_name, adapter in modules.items():
        if not isinstance(adapter, LoraDelta):
            continue
        if not adapter_name.endswith("_lora"):
            raise PeftContractError(
                f"LoRA adapter module name cannot identify its base: {adapter_name}"
            )
        projection_name = adapter_name.removesuffix("_lora")
        projection = modules.get(projection_name)
        if not isinstance(projection, nn.Linear):
            raise PeftContractError(
                f"LoRA adapter base is not a linear projection: {adapter_name}"
            )
        if projection_name in merged_projection_names:
            raise PeftContractError(
                f"multiple LoRA adapters target one projection: {projection_name}"
            )
        merged_projection_names.add(projection_name)
        weight_name = _canonical_parameter_name(f"{projection_name}.weight")
        if weight_name not in merged_state:
            raise PeftContractError(
                f"LoRA base state is missing projection weight: {weight_name}"
            )
        delta = (adapter.lora_B.detach() @ adapter.lora_A.detach()) * adapter.scaling
        if delta.shape != projection.weight.shape:
            raise PeftContractError(
                f"LoRA merged delta shape does not match: {adapter_name}"
            )
        merged_state[weight_name] = (
            projection.weight.detach() + delta.to(projection.weight.dtype)
        ).cpu().contiguous()
    if not merged_projection_names:
        raise PeftContractError("enabled LoRA model has no mergeable projections")
    return merged_state


def load_lora_base_state_dict(
    model: nn.Module,
    state_dict: Mapping[str, torch.Tensor],
    *,
    allow_partial: bool = False,
) -> None:
    """Load a base checkpoint while requiring adapter tensors to stay separate."""

    if not isinstance(state_dict, Mapping) or any(
        not isinstance(name, str) for name in state_dict
    ):
        raise PeftContractError("LoRA base state must be a mapping with string keys")
    adapter_names = set(lora_parameter_names(model))
    if not adapter_names:
        raise PeftContractError("LoRA base loading requires an enabled model")
    expected_base = set(lora_base_state_dict(model))
    actual = set(state_dict)
    unexpected = sorted(actual - expected_base)
    missing_base = sorted(expected_base - actual)
    if unexpected:
        raise PeftContractError(
            "LoRA base state contains unknown or adapter tensors: "
            + ", ".join(unexpected)
        )
    if missing_base and not allow_partial:
        raise PeftContractError(
            "LoRA base state is incomplete: " + ", ".join(missing_base)
        )
    incompatible = model.load_state_dict(dict(state_dict), strict=False)
    expected_missing = adapter_names | set(missing_base)
    if set(incompatible.missing_keys) != expected_missing:
        raise PeftContractError("LoRA base loading reported inconsistent missing keys")
    if incompatible.unexpected_keys:
        raise PeftContractError("LoRA base loading reported unexpected keys")


def save_lora_adapter(model: nn.Module, path: Path) -> Path:
    """Atomically publish a standalone adapter-only Torch artifact."""

    config = _model_lora_config(model)
    validate_lora_trainable_parameters(model, config)
    payload = {
        "format": _ADAPTER_FORMAT,
        "schema_version": _ADAPTER_SCHEMA_VERSION,
        "config": config.to_dict(),
        "state_dict": lora_adapter_state_dict(model),
    }
    return _atomic_torch_save(payload, Path(path), "LoRA adapter")


def save_lora_merged_model(model: nn.Module, path: Path) -> Path:
    """Atomically export weights strict-loadable by a LoRA-disabled model."""

    return _atomic_torch_save(
        lora_merged_state_dict(model),
        Path(path),
        "LoRA merged model",
    )


def _atomic_torch_save(payload: object, destination: Path, owner: str) -> Path:
    """Publish one Torch artifact without replacing an existing destination."""

    if destination.exists() or destination.is_symlink():
        raise PeftContractError(f"{owner} destination exists: {destination}")
    parent = destination.parent
    parent.mkdir(parents=True, exist_ok=True)
    if parent.is_symlink() or not parent.is_dir():
        raise PeftContractError(
            f"{owner} parent must be a regular directory: {parent}"
        )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.partial-",
        dir=parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError as error:
            raise PeftContractError(
                f"{owner} destination exists: {destination}"
            ) from error
        destination.chmod(0o600)
        temporary.unlink()
        _fsync_directory(parent)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def load_lora_adapter(model: nn.Module, path: Path) -> LoraConfig:
    """Verify and load adapter tensors into an already loaded frozen base model."""

    config = _model_lora_config(model)
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise PeftContractError(f"LoRA adapter must be a regular file: {source}")
    try:
        raw = torch.load(source, map_location="cpu", weights_only=True)
    except (
        OSError,
        RuntimeError,
        EOFError,
        ValueError,
        pickle.UnpicklingError,
    ) as error:
        raise PeftContractError(f"LoRA adapter is unreadable: {source}") from error
    payload = _exact_mapping(
        raw,
        {"format", "schema_version", "config", "state_dict"},
        "LoRA adapter",
    )
    if payload["format"] != _ADAPTER_FORMAT:
        raise PeftContractError("LoRA adapter format is unsupported")
    if payload["schema_version"] != _ADAPTER_SCHEMA_VERSION:
        raise PeftContractError("LoRA adapter schema version is unsupported")
    artifact_config = LoraConfig.from_dict(payload["config"])
    if artifact_config != config:
        raise PeftContractError(
            "LoRA adapter config does not match the initialized model"
        )
    state = payload["state_dict"]
    if not isinstance(state, Mapping) or any(
        not isinstance(name, str) or not isinstance(tensor, torch.Tensor)
        for name, tensor in state.items()
    ):
        raise PeftContractError("LoRA adapter state_dict must contain tensors")
    expected_names = set(lora_parameter_names(model))
    actual_names = set(state)
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        unknown = sorted(actual_names - expected_names)
        details = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if unknown:
            details.append("unknown: " + ", ".join(unknown))
        raise PeftContractError(
            "LoRA adapter tensors do not match the model"
            + (" (" + "; ".join(details) + ")" if details else "")
        )
    parameters = _canonical_parameter_map(model)
    with torch.no_grad():
        for name in sorted(expected_names):
            source_tensor = state[name]
            destination_tensor = parameters[name]
            if source_tensor.shape != destination_tensor.shape:
                raise PeftContractError(
                    f"LoRA adapter tensor shape does not match: {name}"
                )
            if source_tensor.dtype != destination_tensor.dtype:
                raise PeftContractError(
                    f"LoRA adapter tensor dtype does not match: {name}"
                )
            destination_tensor.copy_(source_tensor.to(destination_tensor.device))
    validate_lora_trainable_parameters(model, config)
    return artifact_config


def _target_values(raw: object) -> tuple[str, ...]:
    if isinstance(raw, str):
        if not raw.strip():
            return ()
        return tuple(value.strip() for value in raw.split(","))
    if isinstance(raw, Sequence) and not isinstance(raw, (bytes, bytearray)):
        return tuple(raw)
    raise PeftContractError(
        "LoRA target modules must be a comma-separated string or sequence"
    )


def _canonical_targets(raw: Sequence[str]) -> tuple[str, ...]:
    if any(not isinstance(value, str) or not value.strip() for value in raw):
        raise PeftContractError("LoRA target module names must be non-empty strings")
    values = tuple(value.strip() for value in raw)
    if len(values) != len(set(values)):
        raise PeftContractError("LoRA target modules must not contain duplicates")
    unknown = sorted(set(values) - set(LORA_TARGET_MODULES))
    if unknown:
        raise PeftContractError(
            "unsupported LoRA target modules: " + ", ".join(unknown)
        )
    selected = set(values)
    return tuple(name for name in LORA_TARGET_MODULES if name in selected)


def _finite_float(raw: object, name: str) -> float:
    if isinstance(raw, bool) or not isinstance(raw, Real):
        raise PeftContractError(f"LoRA {name} must be a real number")
    value = float(raw)
    if not math.isfinite(value):
        raise PeftContractError(f"LoRA {name} must be finite")
    return value


def _model_lora_config(model: nn.Module) -> LoraConfig:
    config = getattr(model, "lora_config", None)
    if not isinstance(config, LoraConfig) or not config.enabled:
        raise PeftContractError("model does not own an enabled LoRA config")
    return config


def _is_lora_parameter_name(name: str) -> bool:
    return name.endswith((".lora_A", ".lora_B"))


def _canonical_parameter_name(name: str) -> str:
    return name.replace(_CHECKPOINT_WRAPPER_SEGMENT, "")


def _canonical_parameter_map(model: nn.Module) -> dict[str, nn.Parameter]:
    parameters = {}
    for name, parameter in model.named_parameters():
        canonical_name = _canonical_parameter_name(name)
        if canonical_name in parameters:
            raise PeftContractError(
                f"LoRA model contains duplicate canonical parameter: {canonical_name}"
            )
        parameters[canonical_name] = parameter
    return parameters


def _exact_mapping(
    raw: object,
    expected_fields: set[str],
    owner: str,
) -> Mapping[str, object]:
    if not isinstance(raw, Mapping) or any(not isinstance(key, str) for key in raw):
        raise PeftContractError(f"{owner} must be a mapping with string keys")
    actual_fields = set(raw)
    if actual_fields != expected_fields:
        missing = sorted(expected_fields - actual_fields)
        unknown = sorted(actual_fields - expected_fields)
        details = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if unknown:
            details.append("unknown: " + ", ".join(unknown))
        raise PeftContractError(
            f"{owner} fields are invalid"
            + (" (" + "; ".join(details) + ")" if details else "")
        )
    return raw


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "LORA_TARGET_MODULES",
    "DisabledLoraDelta",
    "LoraConfig",
    "LoraDelta",
    "PeftContractError",
    "build_lora_delta",
    "freeze_base_for_lora",
    "load_lora_adapter",
    "load_lora_base_state_dict",
    "lora_adapter_state_dict",
    "lora_base_state_dict",
    "lora_merged_state_dict",
    "lora_parameter_names",
    "save_lora_adapter",
    "save_lora_merged_model",
    "validate_lora_trainable_parameters",
]
