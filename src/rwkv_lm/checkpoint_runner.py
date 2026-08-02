"""Single-process RWKV trainer adapter for standard epoch checkpoints."""

from __future__ import annotations

import json
import os
import random
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Protocol

import numpy as np
import torch

from .checkpoint import (
    ArtifactRecord,
    BackendIdentity,
    CheckpointContractError,
    CheckpointManifest,
    TrainingProgress,
    select_checkpoint_loader,
)

_RUNNER_BACKEND_PROFILES = {
    ("pytorch", "single_process", "full"),
    ("pytorch-lightning", "single_device", "full"),
}
_RNG_SCHEMA_VERSION = 1
_STATE_PATHS = {
    "model": ("model.pt", "torch-state-dict"),
    "optimizer": ("optimizer.pt", "torch-optimizer-state"),
    "scheduler": ("scheduler.json", "rwkv-callback-schedule-v1"),
    "rng": ("rng.pt", "torch-rng-state"),
    "data_cursor": ("data-cursor.json", "rwkv-binidx-epoch-cursor-v1"),
}


class Stateful(Protocol):
    """Minimal state-dict surface owned by the trainer adapter."""

    def state_dict(self) -> Mapping[str, object]: ...

    def load_state_dict(self, state_dict: Mapping[str, object], **kwargs): ...


@dataclass(frozen=True)
class EpochCheckpointRunnerAdapter:
    """Save and restore the current callback scheduler at epoch boundaries.

    The v1 adapter intentionally supports only single-process PyTorch and
    Lightning backends. DDP and DeepSpeed need per-rank RNG and backend-owned
    optimizer collection; treating rank-zero ``state_dict()`` output as
    complete would make the manifest dishonest.
    """

    backend: BackendIdentity
    training_config: Mapping[str, object]
    samples_per_epoch: int

    def __post_init__(self) -> None:
        _require_runner_backend(self.backend)
        if (
            isinstance(self.samples_per_epoch, bool)
            or not isinstance(self.samples_per_epoch, int)
            or self.samples_per_epoch <= 0
        ):
            raise CheckpointContractError(
                "runner samples_per_epoch must be a positive integer"
            )
        canonical_config = _canonical_json_bytes(
            self.training_config,
            "training config",
        )
        object.__setattr__(
            self,
            "training_config",
            MappingProxyType(json.loads(canonical_config)),
        )

    def save(
        self,
        checkpoint_dir: Path,
        *,
        model: Stateful,
        optimizer: Stateful,
        global_step: int,
        next_epoch: int,
        scheduler: Stateful | None = None,
    ) -> CheckpointManifest:
        """Publish all resume state together, or leave no final checkpoint."""

        progress = TrainingProgress(
            global_step=global_step,
            epoch=next_epoch,
            step_in_epoch=0,
        )
        destination = Path(checkpoint_dir)
        if destination.suffix == ".pth":
            raise CheckpointContractError(
                "new training checkpoints must be standard directories, not .pth"
            )
        parent = destination.parent
        parent.mkdir(parents=True, exist_ok=True)
        if parent.is_symlink() or not parent.is_dir():
            raise CheckpointContractError(
                f"checkpoint parent must be a regular directory: {parent}"
            )
        if destination.exists() or destination.is_symlink():
            raise CheckpointContractError(
                f"checkpoint destination already exists: {destination}"
            )

        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{destination.name}.partial-",
                dir=parent,
            )
        )
        try:
            _torch_save(model.state_dict(), staging / _STATE_PATHS["model"][0])
            _torch_save(
                optimizer.state_dict(),
                staging / _STATE_PATHS["optimizer"][0],
            )
            scheduler_serialization = "rwkv-callback-schedule-v1"
            scheduler_payload: dict[str, object] = {"global_step": progress.global_step}
            if scheduler is not None:
                scheduler_serialization = "torch-lr-scheduler-json-v1"
                scheduler_payload["state_dict"] = scheduler.state_dict()
            _write_canonical_json(
                staging / _STATE_PATHS["scheduler"][0],
                scheduler_payload,
            )
            _torch_save(_capture_rng_state(), staging / _STATE_PATHS["rng"][0])
            _write_canonical_json(
                staging / _STATE_PATHS["data_cursor"][0],
                {
                    "next_epoch": progress.epoch,
                    "samples_per_epoch": self.samples_per_epoch,
                    "step_in_epoch": progress.step_in_epoch,
                    "world_size": self.backend.world_size,
                },
            )
            _write_canonical_json(
                staging / "training-config.json",
                self.training_config,
            )

            states = {}
            for name, (relative_path, serialization) in _STATE_PATHS.items():
                if name == "scheduler":
                    serialization = scheduler_serialization
                states[name] = ArtifactRecord.from_file(
                    staging,
                    relative_path,
                    serialization=serialization,
                )
            manifest = CheckpointManifest(
                backend=self.backend,
                progress=progress,
                training_config=ArtifactRecord.from_file(
                    staging,
                    "training-config.json",
                    serialization="canonical-json",
                ),
                states=states,
            )
            manifest.write(staging)
            _fsync_directory(staging)
            staging.rename(destination)
            _fsync_directory(parent)
        except Exception:
            if staging.exists():
                shutil.rmtree(staging)
            raise
        return manifest

    def restore(
        self,
        checkpoint: Path,
        *,
        model: Stateful,
        optimizer: Stateful,
        scheduler: Stateful | None = None,
    ) -> TrainingProgress:
        """Restore a verified complete checkpoint into an initialized runner."""

        plan = select_checkpoint_loader(
            Path(checkpoint),
            expected_backend=self.backend,
        )
        manifest = plan.manifest
        if manifest is None:
            raise CheckpointContractError(
                "runner restore requires a complete standard checkpoint"
            )
        checkpoint_dir = plan.source
        expected_config = _canonical_json_bytes(
            self.training_config,
            "training config",
        )
        actual_config = (
            (checkpoint_dir / manifest.training_config.path).read_bytes().rstrip(b"\n")
        )
        if actual_config != expected_config:
            raise CheckpointContractError(
                "checkpoint training config does not match the requested runner"
            )

        model_state = _torch_load(
            checkpoint_dir / manifest.states["model"].path,
            "model",
        )
        optimizer_state = _torch_load(
            checkpoint_dir / manifest.states["optimizer"].path,
            "optimizer",
        )
        rng_state = _torch_load(
            checkpoint_dir / manifest.states["rng"].path,
            "rng",
        )
        scheduler_payload = json.loads(
            (checkpoint_dir / manifest.states["scheduler"].path).read_bytes()
        )
        data_cursor = json.loads(
            (checkpoint_dir / manifest.states["data_cursor"].path).read_bytes()
        )
        if data_cursor["samples_per_epoch"] != self.samples_per_epoch:
            raise CheckpointContractError(
                "checkpoint data cursor samples_per_epoch does not match the runner"
            )
        if not isinstance(model_state, Mapping):
            raise CheckpointContractError("checkpoint model state must be a mapping")
        if not isinstance(optimizer_state, Mapping):
            raise CheckpointContractError(
                "checkpoint optimizer state must be a mapping"
            )
        normalized_rng = _validate_rng_state(rng_state)
        scheduler_serialization = manifest.states["scheduler"].serialization
        if scheduler_serialization == "torch-lr-scheduler-json-v1":
            if scheduler is None:
                raise CheckpointContractError(
                    "checkpoint requires a Torch LR scheduler instance"
                )
            scheduler_state = scheduler_payload.get("state_dict")
            if not isinstance(scheduler_state, Mapping):
                raise CheckpointContractError(
                    "checkpoint scheduler state_dict must be a mapping"
                )
        elif scheduler is not None:
            raise CheckpointContractError(
                "RWKV callback schedule checkpoint does not accept a Torch scheduler"
            )

        model.load_state_dict(model_state, strict=True)
        optimizer.load_state_dict(optimizer_state)
        if scheduler is not None:
            scheduler.load_state_dict(scheduler_state)
        _restore_rng_state(normalized_rng)
        return manifest.progress


def find_latest_training_checkpoint(project_dir: Path) -> Path:
    """Return the newest verified published epoch checkpoint."""

    checkpoint_root = Path(project_dir) / "checkpoints"
    if checkpoint_root.is_symlink() or not checkpoint_root.is_dir():
        raise CheckpointContractError(
            f"standard checkpoint directory does not exist: {checkpoint_root}"
        )
    candidates: list[tuple[int, Path]] = []
    for candidate in sorted(checkpoint_root.iterdir()):
        if candidate.name.startswith("."):
            continue
        plan = select_checkpoint_loader(candidate)
        manifest = plan.manifest
        if manifest is None:
            raise CheckpointContractError(
                f"published checkpoint is incomplete: {candidate}"
            )
        expected_name = f"epoch-{manifest.progress.epoch:08d}"
        if candidate.name != expected_name:
            raise CheckpointContractError(
                "checkpoint directory name does not match its next epoch: "
                f"{candidate.name} != {expected_name}"
            )
        candidates.append((manifest.progress.epoch, plan.source))
    if not candidates:
        raise CheckpointContractError(
            f"no published standard checkpoints found in: {checkpoint_root}"
        )
    return max(candidates, key=lambda item: item[0])[1]


def _require_runner_backend(backend: BackendIdentity) -> None:
    profile = (backend.name, backend.strategy, backend.state_dict_type)
    if profile not in _RUNNER_BACKEND_PROFILES or backend.world_size != 1:
        raise CheckpointContractError(
            "standard runner checkpoint adapter only supports "
            "pytorch/single_process/full or "
            "pytorch-lightning/single_device/full with world_size=1; "
            f"got {'/'.join(profile)} world_size={backend.world_size}"
        )


def _canonical_json_bytes(value: object, owner: str) -> bytes:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise CheckpointContractError(f"{owner} must be a mapping with string keys")
    try:
        return json.dumps(
            dict(value),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise CheckpointContractError(
            f"{owner} must contain canonical JSON values"
        ) from error


def _write_canonical_json(path: Path, value: object) -> None:
    payload = _canonical_json_bytes(value, path.name)
    with path.open("xb") as stream:
        stream.write(payload)
        stream.write(b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o600)


def _torch_save(value: object, path: Path) -> None:
    try:
        torch.save(value, path)
        path.chmod(0o600)
        with path.open("rb") as stream:
            os.fsync(stream.fileno())
    except Exception as error:
        path.unlink(missing_ok=True)
        raise CheckpointContractError(
            f"failed to serialize checkpoint state: {path.name}"
        ) from error


def _torch_load(path: Path, owner: str) -> object:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise CheckpointContractError(
            f"failed to load checkpoint {owner} state"
        ) from error


def _capture_rng_state(
    *,
    include_cuda: bool = True,
    cuda_device: int | None = None,
) -> dict[str, object]:
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    cuda_states = []
    if include_cuda and torch.cuda.is_available():
        cuda_states = (
            torch.cuda.get_rng_state_all()
            if cuda_device is None
            else [torch.cuda.get_rng_state(cuda_device)]
        )
    return {
        "schema_version": _RNG_SCHEMA_VERSION,
        "python": {
            "version": python_state[0],
            "state": list(python_state[1]),
            "gauss_next": python_state[2],
        },
        "numpy": {
            "bit_generator": numpy_state[0],
            "state": torch.from_numpy(numpy_state[1].copy()),
            "position": numpy_state[2],
            "has_gauss": numpy_state[3],
            "cached_gaussian": numpy_state[4],
        },
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": cuda_states,
    }


def _validate_rng_state(
    raw: object,
    *,
    expected_cuda_device_count: int | None = None,
) -> dict[str, object]:
    if not isinstance(raw, dict) or set(raw) != {
        "schema_version",
        "python",
        "numpy",
        "torch_cpu",
        "torch_cuda",
    }:
        raise CheckpointContractError("checkpoint RNG state has invalid fields")
    if raw["schema_version"] != _RNG_SCHEMA_VERSION:
        raise CheckpointContractError("checkpoint RNG schema_version is unsupported")

    python_state = raw["python"]
    if not isinstance(python_state, dict) or set(python_state) != {
        "version",
        "state",
        "gauss_next",
    }:
        raise CheckpointContractError("checkpoint Python RNG state is invalid")
    if (
        not isinstance(python_state["version"], int)
        or not isinstance(python_state["state"], list)
        or not python_state["state"]
        or any(not isinstance(value, int) for value in python_state["state"])
        or (
            python_state["gauss_next"] is not None
            and not isinstance(python_state["gauss_next"], float)
        )
    ):
        raise CheckpointContractError("checkpoint Python RNG values are invalid")

    numpy_state = raw["numpy"]
    if not isinstance(numpy_state, dict) or set(numpy_state) != {
        "bit_generator",
        "state",
        "position",
        "has_gauss",
        "cached_gaussian",
    }:
        raise CheckpointContractError("checkpoint NumPy RNG state is invalid")
    if (
        not isinstance(numpy_state["bit_generator"], str)
        or not isinstance(numpy_state["state"], torch.Tensor)
        or numpy_state["state"].dtype != torch.uint32
        or numpy_state["state"].ndim != 1
        or not isinstance(numpy_state["position"], int)
        or not isinstance(numpy_state["has_gauss"], int)
        or not isinstance(numpy_state["cached_gaussian"], float)
    ):
        raise CheckpointContractError("checkpoint NumPy RNG values are invalid")

    torch_cpu = raw["torch_cpu"]
    torch_cuda = raw["torch_cuda"]
    if (
        not isinstance(torch_cpu, torch.Tensor)
        or torch_cpu.dtype != torch.uint8
        or torch_cpu.ndim != 1
        or not isinstance(torch_cuda, list)
        or any(
            not isinstance(state, torch.Tensor)
            or state.dtype != torch.uint8
            or state.ndim != 1
            for state in torch_cuda
        )
    ):
        raise CheckpointContractError("checkpoint Torch RNG values are invalid")
    if expected_cuda_device_count is None:
        expected_cuda_device_count = (
            torch.cuda.device_count() if torch.cuda.is_available() else 0
        )
    if len(torch_cuda) != expected_cuda_device_count:
        raise CheckpointContractError(
            "checkpoint CUDA RNG device count does not match the runtime"
        )
    return raw


def _restore_rng_state(
    state: Mapping[str, object],
    *,
    cuda_device: int | None = None,
) -> None:
    python_state = state["python"]
    numpy_state = state["numpy"]
    random.setstate(
        (
            python_state["version"],
            tuple(python_state["state"]),
            python_state["gauss_next"],
        )
    )
    np.random.set_state(
        (
            numpy_state["bit_generator"],
            numpy_state["state"].numpy().astype(np.uint32, copy=False),
            numpy_state["position"],
            numpy_state["has_gauss"],
            numpy_state["cached_gaussian"],
        )
    )
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"]:
        if cuda_device is None:
            torch.cuda.set_rng_state_all(state["torch_cuda"])
        else:
            if len(state["torch_cuda"]) != 1:
                raise CheckpointContractError(
                    "per-rank CUDA RNG state must contain exactly one device"
                )
            torch.cuda.set_rng_state(state["torch_cuda"][0], cuda_device)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = ["EpochCheckpointRunnerAdapter", "find_latest_training_checkpoint"]
