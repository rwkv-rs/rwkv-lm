"""TorchTitan and transformers-rwkv checkpoint name conversion."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torchtitan.models.common.linear import Linear
from torchtitan.protocols.module import Module
from torchtitan.protocols.state_dict_adapter import StateDictAdapter

from .model import Rwkv7Model

_NATIVE_TO_HF_PREFIXES = (
    ("tok_embeddings.", "model.embeddings."),
    ("layers.", "model.blocks."),
    ("norm.", "model.ln_out."),
    ("lm_head.", "head."),
)
_ADAPTER_FORMAT = "rwkv7-torchtitan-lora-adapter"
_ADAPTER_SCHEMA_VERSION = 1
_ADAPTER_MARKERS = (".lora_a.", ".lora_b.")
_ALLOWED_ADAPTER_DTYPES = {
    torch.float16,
    torch.bfloat16,
    torch.float32,
}


class AdapterCheckpointError(ValueError):
    """Raised before an unsafe or incompatible adapter checkpoint is applied."""


def validate_rwkv7_hf_checkpoint(config: Any) -> None:
    """Fail before model construction when an HF initial checkpoint is incomplete."""
    if not config.checkpoint.initial_load_in_hf:
        return
    checkpoint_path = Path(
        config.checkpoint.initial_load_path or config.hf_assets_path
    ).resolve()
    required = [checkpoint_path / "config.json"]
    has_weights = (checkpoint_path / "model.safetensors").is_file() or (
        checkpoint_path / "model.safetensors.index.json"
    ).is_file()
    missing = [str(path) for path in required if not path.is_file()]
    if not has_weights:
        missing.append(f"{checkpoint_path}/model.safetensors[.index.json]")
    if missing:
        raise FileNotFoundError(
            "RWKV initial_load_in_hf requires a standard transformers-rwkv "
            f"checkpoint; missing {missing}"
        )
    _rwkv7_artifact_identity(checkpoint_path)


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
        self._lora_scales = {
            fqn: float(config.alpha) / int(config.rank)
            for fqn, config, _, _ in model_config.traverse(
                Module.Config,
                recurse=True,
            )
            if isinstance(config, Linear.Config)
            and hasattr(config, "rank")
            and hasattr(config, "alpha")
        }

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        state_dict = dict(state_dict)
        for fqn, scale in self._lora_scales.items():
            weight_name = f"{fqn}.weight"
            lora_a_name = f"{fqn}.lora_a.weight"
            lora_b_name = f"{fqn}.lora_b.weight"
            if lora_a_name not in state_dict and lora_b_name not in state_dict:
                continue
            if not all(
                name in state_dict for name in (weight_name, lora_a_name, lora_b_name)
            ):
                raise KeyError(f"incomplete LoRA state for {fqn}")
            weight = state_dict[weight_name]
            lora_a = state_dict.pop(lora_a_name)
            lora_b = state_dict.pop(lora_b_name)
            if not all(
                isinstance(tensor, torch.Tensor) for tensor in (weight, lora_a, lora_b)
            ):
                raise TypeError(f"LoRA state for {fqn} must contain tensors")
            state_dict[weight_name] = weight + scale * (lora_b @ lora_a)
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

    def adapter_checkpoint(self, model: Rwkv7Model) -> dict[str, Any]:
        """Build an adapter-only checkpoint bound to one immutable base artifact."""
        model_identity, source_revision = _rwkv7_artifact_identity(self.hf_assets_path)
        state_dict = model.state_dict()
        adapter_names = self._adapter_names()
        _validate_state_key_inventory(state_dict, adapter_names)
        base_state = {
            name: tensor
            for name, tensor in state_dict.items()
            if name not in adapter_names
        }
        adapter_state = {name: state_dict[name] for name in adapter_names}
        serialized_adapter = {
            name: tensor.detach().cpu().contiguous().clone()
            for name, tensor in adapter_state.items()
        }
        _validate_adapter_tensors(serialized_adapter, adapter_state)
        return {
            "format": _ADAPTER_FORMAT,
            "schema_version": _ADAPTER_SCHEMA_VERSION,
            "base": {
                "model_identity": model_identity,
                "source_revision": source_revision,
                "state_digest": _state_digest(base_state, owner="RWKV base state"),
                "state_inventory": _state_inventory(base_state),
            },
            "adapter": {
                "state_digest": _state_digest(
                    serialized_adapter,
                    owner="RWKV adapter state",
                ),
                "state_inventory": _state_inventory(serialized_adapter),
                "storage_device": "cpu",
                "load_device_policy": "target",
            },
            "state_dict": serialized_adapter,
        }

    def load_adapter_checkpoint(
        self,
        model: Rwkv7Model,
        checkpoint: Mapping[str, Any],
    ) -> None:
        """Validate a complete adapter checkpoint, then commit it atomically."""
        payload = _exact_mapping(
            checkpoint,
            {"format", "schema_version", "base", "adapter", "state_dict"},
            "RWKV adapter checkpoint",
        )
        if payload["format"] != _ADAPTER_FORMAT:
            raise AdapterCheckpointError(
                "RWKV adapter checkpoint format is unsupported"
            )
        if (
            type(payload["schema_version"]) is not int
            or payload["schema_version"] != _ADAPTER_SCHEMA_VERSION
        ):
            raise AdapterCheckpointError(
                "RWKV adapter checkpoint schema version is unsupported"
            )
        base_metadata = _validate_base_metadata(payload["base"])
        adapter_metadata = _validate_adapter_metadata(payload["adapter"])
        model_identity, source_revision = _rwkv7_artifact_identity(self.hf_assets_path)
        if base_metadata["model_identity"] != model_identity:
            raise AdapterCheckpointError(
                "RWKV adapter base model identity does not match the loaded artifact"
            )
        if base_metadata["source_revision"] != source_revision:
            raise AdapterCheckpointError(
                "RWKV adapter base source revision does not match the loaded artifact"
            )

        source_state = _tensor_mapping(payload["state_dict"], "RWKV adapter state")
        target_state = model.state_dict()
        adapter_names = self._adapter_names()
        _validate_state_key_inventory(target_state, adapter_names)
        if set(source_state) != set(adapter_names):
            missing = sorted(set(adapter_names) - set(source_state))
            unexpected = sorted(set(source_state) - set(adapter_names))
            raise AdapterCheckpointError(
                "RWKV adapter tensor inventory does not match the model: "
                f"missing={missing}, unexpected={unexpected}"
            )
        target_adapter = {name: target_state[name] for name in adapter_names}
        target_base = {
            name: tensor
            for name, tensor in target_state.items()
            if name not in adapter_names
        }

        # Complete every schema, inventory, shape, dtype, device, and base identity
        # check before staging or mutating a single destination tensor.
        _validate_adapter_tensors(source_state, target_adapter)
        if adapter_metadata["state_inventory"] != _state_inventory(source_state):
            raise AdapterCheckpointError(
                "RWKV adapter tensor inventory does not match checkpoint metadata"
            )
        if base_metadata["state_inventory"] != _state_inventory(target_base):
            raise AdapterCheckpointError(
                "RWKV base tensor inventory does not match checkpoint metadata"
            )
        if adapter_metadata["state_digest"] != _state_digest(
            source_state,
            owner="RWKV adapter state",
        ):
            raise AdapterCheckpointError("RWKV adapter state digest does not match")
        if base_metadata["state_digest"] != _state_digest(
            target_base,
            owner="RWKV base state",
        ):
            raise AdapterCheckpointError(
                "RWKV adapter base state digest does not match the loaded model"
            )

        prepared = {}
        originals = {}
        try:
            for name in adapter_names:
                destination = target_adapter[name]
                prepared[name] = source_state[name].to(
                    device=destination.device,
                    copy=True,
                )
                originals[name] = destination.detach().clone()
        except (RuntimeError, TypeError, ValueError) as error:
            raise AdapterCheckpointError(
                "RWKV adapter tensors cannot be staged on the target device"
            ) from error
        try:
            with torch.no_grad():
                for name in adapter_names:
                    target_adapter[name].copy_(prepared[name])
        except Exception as error:
            with torch.no_grad():
                for name in adapter_names:
                    target_adapter[name].copy_(originals[name])
            raise AdapterCheckpointError(
                "RWKV adapter tensor commit failed and was rolled back"
            ) from error

    def _adapter_names(self) -> tuple[str, ...]:
        names = tuple(
            sorted(
                f"{fqn}.{adapter}.weight"
                for fqn in self._lora_scales
                for adapter in ("lora_a", "lora_b")
            )
        )
        if not names:
            raise AdapterCheckpointError(
                "RWKV model config does not contain LoRA adapters"
            )
        return names


def _rwkv7_artifact_identity(
    hf_assets_path: str | Path | None,
) -> tuple[str, str]:
    if hf_assets_path is None:
        raise AdapterCheckpointError(
            "RWKV adapter checkpoints require a transformers-rwkv artifact path"
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
    return model_identity, source_revision


def _validate_state_key_inventory(
    state_dict: Mapping[str, Any],
    adapter_names: tuple[str, ...],
) -> None:
    observed_adapters = {
        name
        for name in state_dict
        if any(marker in name for marker in _ADAPTER_MARKERS)
    }
    expected_adapters = set(adapter_names)
    if observed_adapters != expected_adapters:
        raise AdapterCheckpointError(
            "RWKV model LoRA tensor inventory is inconsistent: "
            f"missing={sorted(expected_adapters - observed_adapters)}, "
            f"unexpected={sorted(observed_adapters - expected_adapters)}"
        )


def _validate_adapter_tensors(
    source_state: Mapping[str, Any],
    target_state: Mapping[str, Any],
) -> None:
    for name in sorted(source_state):
        source = source_state[name]
        target = target_state[name]
        if not isinstance(source, torch.Tensor) or not isinstance(target, torch.Tensor):
            raise AdapterCheckpointError(
                f"RWKV adapter state value must be a tensor: {name}"
            )
        if source.device.type != "cpu":
            raise AdapterCheckpointError(
                f"RWKV adapter checkpoint tensor must be stored on CPU: {name}"
            )
        if source.dtype not in _ALLOWED_ADAPTER_DTYPES:
            raise AdapterCheckpointError(
                f"RWKV adapter checkpoint tensor dtype is unsupported: {name}"
            )
        if target.dtype not in _ALLOWED_ADAPTER_DTYPES:
            raise AdapterCheckpointError(
                f"RWKV target adapter tensor dtype is unsupported: {name}"
            )
        if source.dtype != target.dtype:
            raise AdapterCheckpointError(
                f"RWKV adapter tensor dtype does not match the target: {name}"
            )
        if source.shape != target.shape:
            raise AdapterCheckpointError(
                f"RWKV adapter tensor shape does not match the target: {name}"
            )
        if target.is_meta:
            raise AdapterCheckpointError(
                f"RWKV adapter target tensor must be materialized: {name}"
            )
        if source.layout != torch.strided or target.layout != torch.strided:
            raise AdapterCheckpointError(
                f"RWKV adapter tensor layout is unsupported: {name}"
            )


def _validate_base_metadata(raw: Any) -> Mapping[str, Any]:
    metadata = _exact_mapping(
        raw,
        {"model_identity", "source_revision", "state_digest", "state_inventory"},
        "RWKV adapter base metadata",
    )
    _lower_hex(metadata["model_identity"], length=64, owner="RWKV model identity")
    _lower_hex(metadata["source_revision"], length=40, owner="RWKV source revision")
    _lower_hex(metadata["state_digest"], length=64, owner="RWKV base state digest")
    _validate_inventory(metadata["state_inventory"], "RWKV base state inventory")
    return metadata


def _validate_adapter_metadata(raw: Any) -> Mapping[str, Any]:
    metadata = _exact_mapping(
        raw,
        {
            "state_digest",
            "state_inventory",
            "storage_device",
            "load_device_policy",
        },
        "RWKV adapter metadata",
    )
    _lower_hex(metadata["state_digest"], length=64, owner="RWKV adapter state digest")
    _validate_inventory(metadata["state_inventory"], "RWKV adapter state inventory")
    if metadata["storage_device"] != "cpu":
        raise AdapterCheckpointError("RWKV adapter storage device must be CPU")
    if metadata["load_device_policy"] != "target":
        raise AdapterCheckpointError("RWKV adapter load device policy must be target")
    return metadata


def _validate_inventory(raw: Any, owner: str) -> None:
    if not isinstance(raw, list):
        raise AdapterCheckpointError(f"{owner} must be a list")
    previous_name = None
    for item in raw:
        item = _exact_mapping(item, {"name", "shape", "dtype"}, owner)
        name = item["name"]
        shape = item["shape"]
        dtype = item["dtype"]
        if not isinstance(name, str) or not name:
            raise AdapterCheckpointError(f"{owner} contains an invalid name")
        if previous_name is not None and name <= previous_name:
            raise AdapterCheckpointError(f"{owner} must have unique sorted names")
        if not isinstance(shape, list) or any(
            type(dimension) is not int or dimension < 0 for dimension in shape
        ):
            raise AdapterCheckpointError(f"{owner} contains an invalid shape")
        if not isinstance(dtype, str) or not dtype:
            raise AdapterCheckpointError(f"{owner} contains an invalid dtype")
        previous_name = name


def _state_inventory(state_dict: Mapping[str, Any]) -> list[dict[str, Any]]:
    inventory = []
    for name, tensor in sorted(state_dict.items()):
        if not isinstance(tensor, torch.Tensor):
            raise AdapterCheckpointError(f"RWKV state value must be a tensor: {name}")
        inventory.append(
            {
                "name": name,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype).removeprefix("torch."),
            }
        )
    return inventory


def _state_digest(state_dict: Mapping[str, Any], *, owner: str) -> str:
    digest = hashlib.sha256()
    digest.update(b"rwkv7-state-digest-v1\0")
    for item in _state_inventory(state_dict):
        name = item["name"]
        tensor = state_dict[name]
        if tensor.layout != torch.strided or tensor.is_quantized or tensor.is_meta:
            raise AdapterCheckpointError(f"{owner} tensor is unsupported: {name}")
        try:
            canonical = tensor.detach().contiguous().view(torch.uint8).cpu()
            value = canonical.numpy().tobytes(order="C")
        except (RuntimeError, TypeError, ValueError) as error:
            raise AdapterCheckpointError(
                f"{owner} tensor cannot be canonicalized: {name}"
            ) from error
        for field in (
            name.encode(),
            item["dtype"].encode(),
            json.dumps(item["shape"], separators=(",", ":")).encode(),
            value,
        ):
            digest.update(len(field).to_bytes(8, "big"))
            digest.update(field)
    return digest.hexdigest()


def _tensor_mapping(raw: Any, owner: str) -> Mapping[str, torch.Tensor]:
    mapping = _string_mapping(raw, owner)
    if any(not isinstance(value, torch.Tensor) for value in mapping.values()):
        raise AdapterCheckpointError(f"{owner} must contain only tensors")
    return mapping


def _string_mapping(raw: Any, owner: str) -> Mapping[str, Any]:
    if not isinstance(raw, Mapping) or any(not isinstance(key, str) for key in raw):
        raise AdapterCheckpointError(f"{owner} must be a mapping with string keys")
    return raw


def _exact_mapping(
    raw: Any,
    expected_fields: set[str],
    owner: str,
) -> Mapping[str, Any]:
    mapping = _string_mapping(raw, owner)
    actual_fields = set(mapping)
    if actual_fields != expected_fields:
        raise AdapterCheckpointError(
            f"{owner} fields are invalid: "
            f"missing={sorted(expected_fields - actual_fields)}, "
            f"unexpected={sorted(actual_fields - expected_fields)}"
        )
    return mapping


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
    "Rwkv7StateDictAdapter",
    "validate_rwkv7_hf_checkpoint",
]
