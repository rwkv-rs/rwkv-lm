"""Collective FSDP2 checkpoint owner backed by Torch DCP sharded state."""

from __future__ import annotations

import json
import re
import shutil
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Protocol, TypeVar

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)
from torch.distributed.fsdp import FSDPModule

from .checkpoint import (
    ArtifactRecord,
    BackendIdentity,
    CheckpointContractError,
    CheckpointManifest,
    TrainingProgress,
    select_checkpoint_loader,
)
from .checkpoint_runner import (
    _canonical_json_bytes,
    _capture_rng_state,
    _fsync_directory,
    _require_compatible_training_config,
    _restore_rng_state,
    _torch_load,
    _torch_save,
    _validate_rng_state,
    _write_canonical_json,
)

_DCP_MODEL_SERIALIZATION = "torch-distributed-checkpoint-model-v1"
_DCP_OPTIMIZER_SERIALIZATION = "torch-distributed-checkpoint-optimizer-v1"
_PER_RANK_RNG_SERIALIZATION = "torch-rng-state-per-rank-v1"
_SHARDED_CURSOR_SERIALIZATION = "rwkv-binidx-epoch-cursor-sharded-v1"
_DCP_SHARD_PATTERN = re.compile(r"^__(?P<rank>[0-9]+)_[0-9]+\.distcp$")
_STATE_DICT_OPTIONS = StateDictOptions(
    full_state_dict=False,
    cpu_offload=False,
    strict=True,
)

_T = TypeVar("_T")


class Stateful(Protocol):
    def state_dict(self) -> Mapping[str, object]: ...

    def load_state_dict(self, state_dict: Mapping[str, object]): ...


@dataclass(frozen=True)
class FSDP2CheckpointRunnerAdapter:
    """Save and restore one collective FSDP2 training transaction.

    Model and optimizer state use ``torch.distributed.checkpoint`` sharded
    trees. Scheduler state is required to be identical on every rank, while
    Python, NumPy, Torch CPU, and Torch CUDA RNG state is owned per rank.
    Rank 0 alone publishes the manifest and atomically renames the completed
    staging directory.
    """

    backend: BackendIdentity
    training_config: Mapping[str, object]
    samples_per_epoch: int

    def __post_init__(self) -> None:
        if (
            self.backend.name,
            self.backend.strategy,
            self.backend.state_dict_type,
        ) != ("pytorch", "fsdp2", "sharded"):
            raise CheckpointContractError(
                "FSDP2 runner requires pytorch/fsdp2/sharded backend identity"
            )
        if self.backend.version != torch.__version__:
            raise CheckpointContractError(
                "FSDP2 backend version must exactly match torch.__version__"
            )
        if (
            isinstance(self.samples_per_epoch, bool)
            or not isinstance(self.samples_per_epoch, int)
            or self.samples_per_epoch <= 0
        ):
            raise CheckpointContractError(
                "runner samples_per_epoch must be a positive integer"
            )
        epoch_steps = self.training_config.get("epoch_steps")
        if (
            isinstance(epoch_steps, bool)
            or not isinstance(epoch_steps, int)
            or epoch_steps <= 0
        ):
            raise CheckpointContractError(
                "FSDP2 training config must define positive integer epoch_steps"
            )
        global_batch_size = self.training_config.get(
            "real_bsz",
            self.training_config.get("batch_size"),
        )
        if (
            isinstance(global_batch_size, bool)
            or not isinstance(global_batch_size, int)
            or global_batch_size <= 0
            or epoch_steps * global_batch_size != self.samples_per_epoch
        ):
            raise CheckpointContractError(
                "FSDP2 samples_per_epoch must equal epoch_steps * global batch size"
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

    @classmethod
    def from_process_group(
        cls,
        *,
        training_config: Mapping[str, object],
        samples_per_epoch: int,
    ) -> FSDP2CheckpointRunnerAdapter:
        _require_distributed()
        backend = BackendIdentity(
            name="pytorch",
            version=torch.__version__,
            strategy="fsdp2",
            world_size=dist.get_world_size(),
            state_dict_type="sharded",
        )
        return _collective_phase(
            "construct FSDP2 checkpoint adapter",
            lambda: cls(
                backend=backend,
                training_config=training_config,
                samples_per_epoch=samples_per_epoch,
            ),
        )

    def save(
        self,
        checkpoint_dir: Path,
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        global_step: int,
        next_epoch: int,
        scheduler: Stateful | None = None,
    ) -> CheckpointManifest:
        """Collectively publish a complete sharded epoch checkpoint."""

        rank = self._require_runtime(model)
        device_type = _collective_phase(
            "identify FSDP2 model device",
            lambda: _model_device_type(model),
        )
        _require_consensus(device_type.encode("utf-8"), "model device type")
        cuda_device = (
            _collective_phase(
                "identify FSDP2 CUDA device",
                torch.cuda.current_device,
            )
            if device_type == "cuda"
            else None
        )
        progress = _collective_phase(
            "validate FSDP2 training progress",
            lambda: _epoch_boundary_progress(
                global_step=global_step,
                next_epoch=next_epoch,
                epoch_steps=self.training_config["epoch_steps"],
            ),
        )
        _require_consensus(
            _canonical_json_bytes(progress.to_dict(), "training progress"),
            "training progress",
        )
        destination = _consensus_destination(Path(checkpoint_dir))
        if destination.suffix == ".pth":
            raise CheckpointContractError(
                "new training checkpoints must be standard directories, not .pth"
            )
        _require_consensus(
            _canonical_json_bytes(self.training_config, "training config"),
            "training config",
        )
        scheduler_serialization, scheduler_payload = _collective_phase(
            "serialize scheduler state",
            lambda: _scheduler_payload(
                scheduler,
                global_step=progress.global_step,
            ),
        )
        scheduler_bytes = _collective_phase(
            "canonicalize scheduler state",
            lambda: _canonical_json_bytes(
                scheduler_payload,
                "scheduler state",
            ),
        )
        _require_consensus(scheduler_bytes, "scheduler state")

        staging = _create_staging(destination)
        try:
            model_state = _collective_phase(
                "collect FSDP2 model state",
                lambda: get_model_state_dict(model, options=_STATE_DICT_OPTIONS),
            )
            _collective_phase(
                "write FSDP2 model state",
                lambda: dcp.save(
                    {"model": model_state},
                    checkpoint_id=staging / "model",
                ),
            )

            optimizer_state = _collective_phase(
                "collect FSDP2 optimizer state",
                lambda: get_optimizer_state_dict(
                    model,
                    optimizer,
                    options=_STATE_DICT_OPTIONS,
                ),
            )
            _collective_phase(
                "write FSDP2 optimizer state",
                lambda: dcp.save(
                    {"optimizer": optimizer_state},
                    checkpoint_id=staging / "optimizer",
                ),
            )

            _rank_zero_phase(
                "create per-rank RNG state directory",
                lambda: (staging / "rng").mkdir(mode=0o700),
            )
            _collective_phase(
                "write per-rank RNG state",
                lambda: _torch_save(
                    _capture_rng_state(
                        include_cuda=device_type == "cuda",
                        cuda_device=cuda_device,
                    ),
                    staging / "rng" / f"rank-{rank:05d}.pt",
                ),
            )

            def write_manifest() -> None:
                _chmod_tree_private(staging / "model")
                _chmod_tree_private(staging / "optimizer")
                _chmod_tree_private(staging / "rng")
                _write_canonical_json(
                    staging / "scheduler.json",
                    scheduler_payload,
                )
                _write_canonical_json(
                    staging / "training-config.json",
                    self.training_config,
                )
                _write_canonical_json(
                    staging / "data-cursor.json",
                    {
                        "next_epoch": progress.epoch,
                        "rank_ownership": _rank_ownership(
                            staging,
                            self.backend.world_size,
                        ),
                        "samples_per_epoch": self.samples_per_epoch,
                        "step_in_epoch": progress.step_in_epoch,
                        "world_size": self.backend.world_size,
                    },
                )
                states = {
                    "model": ArtifactRecord.from_tree(
                        staging,
                        "model",
                        serialization=_DCP_MODEL_SERIALIZATION,
                    ),
                    "optimizer": ArtifactRecord.from_tree(
                        staging,
                        "optimizer",
                        serialization=_DCP_OPTIMIZER_SERIALIZATION,
                    ),
                    "scheduler": ArtifactRecord.from_file(
                        staging,
                        "scheduler.json",
                        serialization=scheduler_serialization,
                    ),
                    "rng": ArtifactRecord.from_tree(
                        staging,
                        "rng",
                        serialization=_PER_RANK_RNG_SERIALIZATION,
                    ),
                    "data_cursor": ArtifactRecord.from_file(
                        staging,
                        "data-cursor.json",
                        serialization=_SHARDED_CURSOR_SERIALIZATION,
                    ),
                }
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

            _rank_zero_phase("write FSDP2 checkpoint manifest", write_manifest)
            _rank_zero_phase(
                "verify staged FSDP2 checkpoint",
                lambda: select_checkpoint_loader(
                    staging,
                    expected_backend=self.backend,
                ),
            )
            _rank_zero_phase(
                "publish FSDP2 checkpoint",
                lambda: _publish_staging(staging, destination),
            )
        except Exception:
            _rank_zero_phase(
                "clean failed FSDP2 checkpoint staging",
                lambda: shutil.rmtree(staging, ignore_errors=True),
            )
            raise

        plan = _collective_phase(
            "verify published FSDP2 checkpoint",
            lambda: select_checkpoint_loader(
                destination,
                expected_backend=self.backend,
            ),
        )
        if plan.manifest is None:
            raise CheckpointContractError(
                "published FSDP2 checkpoint is missing its manifest"
            )
        return plan.manifest

    def restore(
        self,
        checkpoint: Path,
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Stateful | None = None,
    ) -> TrainingProgress:
        """Collectively restore verified FSDP2 and per-rank runtime state."""

        rank = self._require_runtime(model)
        device_type = _collective_phase(
            "identify FSDP2 model device",
            lambda: _model_device_type(model),
        )
        _require_consensus(device_type.encode("utf-8"), "model device type")
        cuda_device = (
            _collective_phase(
                "identify FSDP2 CUDA device",
                torch.cuda.current_device,
            )
            if device_type == "cuda"
            else None
        )
        source = _consensus_destination(Path(checkpoint))
        plan = _collective_phase(
            "verify FSDP2 checkpoint",
            lambda: select_checkpoint_loader(
                source,
                expected_backend=self.backend,
            ),
        )
        manifest = plan.manifest
        if manifest is None:
            raise CheckpointContractError(
                "FSDP2 restore requires a complete standard checkpoint"
            )
        checkpoint_dir = plan.source
        expected_config = _canonical_json_bytes(
            self.training_config,
            "training config",
        )
        actual_config = _collective_phase(
            "read FSDP2 training config",
            lambda: (
                (checkpoint_dir / manifest.training_config.path)
                .read_bytes()
                .rstrip(b"\n")
            ),
        )
        _collective_phase(
            "validate FSDP2 training config",
            lambda: _require_equal_training_config(
                actual_config,
                expected_config,
            ),
        )

        scheduler_payload = _collective_phase(
            "read replicated scheduler state",
            lambda: json.loads(
                (checkpoint_dir / manifest.states["scheduler"].path).read_bytes()
            ),
        )
        data_cursor = _collective_phase(
            "read FSDP2 data cursor",
            lambda: json.loads(
                (checkpoint_dir / manifest.states["data_cursor"].path).read_bytes()
            ),
        )
        _collective_phase(
            "validate FSDP2 data cursor",
            lambda: _require_samples_per_epoch(
                data_cursor,
                self.samples_per_epoch,
            ),
        )
        scheduler_state = _collective_phase(
            "validate replicated scheduler state",
            lambda: _validate_scheduler_restore(
                manifest.states["scheduler"].serialization,
                scheduler_payload,
                scheduler,
            ),
        )
        normalized_rng = _collective_phase(
            "validate per-rank RNG state",
            lambda: _validate_rng_state(
                _torch_load(
                    checkpoint_dir
                    / manifest.states["rng"].path
                    / f"rank-{rank:05d}.pt",
                    f"rank {rank} RNG",
                ),
                expected_cuda_device_count=(1 if device_type == "cuda" else 0),
            ),
        )

        model_state = _collective_phase(
            "prepare FSDP2 model restore",
            lambda: {"model": get_model_state_dict(model, options=_STATE_DICT_OPTIONS)},
        )
        _collective_phase(
            "read FSDP2 model state",
            lambda: dcp.load(
                model_state,
                checkpoint_id=checkpoint_dir / manifest.states["model"].path,
            ),
        )

        optimizer_state = _collective_phase(
            "prepare FSDP2 optimizer restore",
            lambda: {
                "optimizer": get_optimizer_state_dict(
                    model,
                    optimizer,
                    options=_STATE_DICT_OPTIONS,
                )
            },
        )
        _collective_phase(
            "read FSDP2 optimizer state",
            lambda: dcp.load(
                optimizer_state,
                checkpoint_id=checkpoint_dir / manifest.states["optimizer"].path,
            ),
        )
        _collective_phase(
            "apply FSDP2 model state",
            lambda: set_model_state_dict(
                model,
                model_state["model"],
                options=_STATE_DICT_OPTIONS,
            ),
        )
        _collective_phase(
            "apply FSDP2 optimizer state",
            lambda: set_optimizer_state_dict(
                model,
                optimizer,
                optimizer_state["optimizer"],
                options=_STATE_DICT_OPTIONS,
            ),
        )
        if scheduler is not None:
            _collective_phase(
                "apply replicated scheduler state",
                lambda: scheduler.load_state_dict(scheduler_state),
            )
        _collective_phase(
            "restore per-rank RNG state",
            lambda: _restore_rng_state(
                normalized_rng,
                cuda_device=cuda_device,
            ),
        )
        return manifest.progress

    def _require_runtime(self, model: torch.nn.Module) -> int:
        _require_distributed()
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        def validate_runtime() -> None:
            if world_size != self.backend.world_size:
                raise CheckpointContractError(
                    "FSDP2 runtime world_size does not match backend identity"
                )
            if not isinstance(model, FSDPModule):
                raise CheckpointContractError(
                    "FSDP2 checkpoint model must be wrapped by fully_shard"
                )

        _collective_phase("validate FSDP2 runtime", validate_runtime)
        return rank


def _require_distributed() -> None:
    if not dist.is_available() or not dist.is_initialized():
        raise CheckpointContractError(
            "FSDP2 checkpoint requires an initialized process group"
        )


def _model_device_type(model: torch.nn.Module) -> str:
    try:
        parameter = next(model.parameters())
    except StopIteration as error:
        raise CheckpointContractError(
            "FSDP2 checkpoint model must contain parameters"
        ) from error
    device_mesh = getattr(parameter, "device_mesh", None)
    if device_mesh is not None:
        return device_mesh.device_type
    return parameter.device.type


def _collective_phase(owner: str, action: Callable[[], _T]) -> _T:
    value = None
    error = None
    try:
        value = action()
    except Exception as caught:  # noqa: BLE001 - every rank must report failures
        error = f"{type(caught).__name__}: {caught}"
    errors: list[str | None] = [None] * dist.get_world_size()
    dist.all_gather_object(errors, error)
    failures = [
        f"rank {rank}: {message}"
        for rank, message in enumerate(errors)
        if message is not None
    ]
    if failures:
        raise CheckpointContractError(
            f"{owner} failed collectively (" + "; ".join(failures) + ")"
        )
    return value


def _rank_zero_phase(owner: str, action: Callable[[], object]) -> None:
    _collective_phase(owner, action if dist.get_rank() == 0 else lambda: None)


def _require_consensus(payload: bytes, owner: str) -> None:
    gathered: list[bytes | None] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, payload)
    if any(candidate != gathered[0] for candidate in gathered[1:]):
        raise CheckpointContractError(f"FSDP2 {owner} must be identical on every rank")


def _consensus_destination(path: Path) -> Path:
    destination = path.absolute()
    gathered: list[str | None] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, str(destination))
    if any(candidate != gathered[0] for candidate in gathered[1:]):
        raise CheckpointContractError(
            "FSDP2 checkpoint path must be identical on every rank"
        )
    return destination


def _create_staging(destination: Path) -> Path:
    result: list[dict[str, str] | None] = [None]
    if dist.get_rank() == 0:
        try:
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
            result[0] = {"path": str(staging)}
        except Exception as error:  # noqa: BLE001 - broadcast rank-zero failure
            result[0] = {"error": f"{type(error).__name__}: {error}"}
    dist.broadcast_object_list(result, src=0)
    if result[0] is None or "error" in result[0]:
        message = "missing rank-zero staging result"
        if result[0] is not None:
            message = result[0]["error"]
        raise CheckpointContractError(
            f"create FSDP2 checkpoint staging failed: {message}"
        )
    return Path(result[0]["path"])


def _scheduler_payload(
    scheduler: Stateful | None,
    *,
    global_step: int,
) -> tuple[str, dict[str, object]]:
    payload: dict[str, object] = {"global_step": global_step}
    if scheduler is None:
        return "rwkv-callback-schedule-v1", payload
    payload["state_dict"] = scheduler.state_dict()
    return "torch-lr-scheduler-json-v1", payload


def _epoch_boundary_progress(
    *,
    global_step: int,
    next_epoch: int,
    epoch_steps: int,
) -> TrainingProgress:
    progress = TrainingProgress(
        global_step=global_step,
        epoch=next_epoch,
        step_in_epoch=0,
    )
    if progress.global_step != progress.epoch * epoch_steps:
        raise CheckpointContractError(
            "FSDP2 checkpoint global_step must match its epoch boundary"
        )
    return progress


def _require_equal_training_config(actual: bytes, expected: bytes) -> None:
    expected_config = json.loads(expected)
    _require_compatible_training_config(actual, expected_config)


def _require_samples_per_epoch(
    data_cursor: Mapping[str, object],
    expected: int,
) -> None:
    if data_cursor["samples_per_epoch"] != expected:
        raise CheckpointContractError(
            "checkpoint data cursor samples_per_epoch does not match the runner"
        )


def _validate_scheduler_restore(
    serialization: str,
    payload: object,
    scheduler: Stateful | None,
) -> Mapping[str, object] | None:
    if not isinstance(payload, dict):
        raise CheckpointContractError("checkpoint scheduler state must be a mapping")
    if serialization == "torch-lr-scheduler-json-v1":
        if scheduler is None:
            raise CheckpointContractError(
                "checkpoint requires a Torch LR scheduler instance"
            )
        state = payload.get("state_dict")
        if not isinstance(state, Mapping):
            raise CheckpointContractError(
                "checkpoint scheduler state_dict must be a mapping"
            )
        return state
    if scheduler is not None:
        raise CheckpointContractError(
            "RWKV callback schedule checkpoint does not accept a Torch scheduler"
        )
    return None


def _rank_ownership(staging: Path, world_size: int) -> dict[str, object]:
    ownership: dict[str, object] = {
        "coordinator_rank": 0,
        "rng": {str(rank): f"rng/rank-{rank:05d}.pt" for rank in range(world_size)},
    }
    for component in ("model", "optimizer"):
        shards = {str(rank): [] for rank in range(world_size)}
        for path in sorted((staging / component).glob("*.distcp")):
            match = _DCP_SHARD_PATTERN.fullmatch(path.name)
            if match is None:
                raise CheckpointContractError(
                    f"unrecognized Torch DCP shard path: {path.name}"
                )
            rank = match.group("rank")
            if rank not in shards:
                raise CheckpointContractError(
                    f"Torch DCP shard owner rank is outside world_size: {path.name}"
                )
            shards[rank].append(f"{component}/{path.name}")
        if any(not rank_paths for rank_paths in shards.values()):
            raise CheckpointContractError(
                f"Torch DCP {component} must contain a shard for every rank"
            )
        ownership[component] = {
            "metadata": f"{component}/.metadata",
            "shards": shards,
        }
    return ownership


def _chmod_tree_private(path: Path) -> None:
    path.chmod(0o700)
    directories = [path]
    for entry in path.rglob("*"):
        if entry.is_dir():
            entry.chmod(0o700)
            directories.append(entry)
        else:
            entry.chmod(0o600)
    for directory in sorted(
        directories,
        key=lambda item: len(item.parts),
        reverse=True,
    ):
        _fsync_directory(directory)


def _publish_staging(staging: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        raise CheckpointContractError(
            f"checkpoint destination already exists: {destination}"
        )
    staging.rename(destination)
    _fsync_directory(destination.parent)


__all__ = ["FSDP2CheckpointRunnerAdapter"]
