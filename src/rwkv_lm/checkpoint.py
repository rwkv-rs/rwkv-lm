"""Fail-closed identities for complete RWKV training checkpoints.

This module defines the storage boundary only. It does not save framework state,
restore a trainer, convert legacy weights, or run distributed collectives.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath
from types import MappingProxyType

CHECKPOINT_FORMAT = "rwkv-lm-training-checkpoint"
CHECKPOINT_MANIFEST_FILENAME = "manifest.json"
CHECKPOINT_SCHEMA_VERSION = 1

REQUIRED_STATE_COMPONENTS = (
    "model",
    "optimizer",
    "scheduler",
    "gradient_scaler",
    "rng",
    "data_cursor",
)
SUPPORTED_STATE_DICT_TYPES = frozenset({"full", "sharded"})
SUPPORTED_STATE_SERIALIZATIONS = MappingProxyType(
    {
        "model": "torch-state-dict",
        "optimizer": "torch-optimizer-state",
        "scheduler": "rwkv-callback-schedule-v1",
        "gradient_scaler": "torch-grad-scaler-json-v1",
        "rng": "torch-rng-state",
        "data_cursor": "rwkv-binidx-epoch-cursor-v1",
    }
)
SUPPORTED_MODEL_SERIALIZATIONS = frozenset(
    {
        "torch-state-dict",
        "torch-distributed-checkpoint-model-v1",
    }
)
SUPPORTED_OPTIMIZER_SERIALIZATIONS = frozenset(
    {
        "torch-optimizer-state",
        "torch-distributed-checkpoint-optimizer-v1",
    }
)
SUPPORTED_SCHEDULER_SERIALIZATIONS = frozenset(
    {
        "rwkv-callback-schedule-v1",
        "torch-lr-scheduler-json-v1",
    }
)
SUPPORTED_GRADIENT_SCALER_SERIALIZATIONS = frozenset(
    {
        "torch-grad-scaler-json-v1",
    }
)
SUPPORTED_RNG_SERIALIZATIONS = frozenset(
    {
        "torch-rng-state",
        "torch-rng-state-per-rank-v1",
    }
)
SUPPORTED_DATA_CURSOR_SERIALIZATIONS = frozenset(
    {
        "rwkv-binidx-epoch-cursor-v1",
        "rwkv-binidx-epoch-cursor-sharded-v1",
    }
)
_SUPPORTED_STATE_SERIALIZATION_SETS = MappingProxyType(
    {
        "model": SUPPORTED_MODEL_SERIALIZATIONS,
        "optimizer": SUPPORTED_OPTIMIZER_SERIALIZATIONS,
        "scheduler": SUPPORTED_SCHEDULER_SERIALIZATIONS,
        "gradient_scaler": SUPPORTED_GRADIENT_SCALER_SERIALIZATIONS,
        "rng": SUPPORTED_RNG_SERIALIZATIONS,
        "data_cursor": SUPPORTED_DATA_CURSOR_SERIALIZATIONS,
    }
)
_TREE_STATE_SERIALIZATIONS = frozenset(
    {
        "torch-distributed-checkpoint-model-v1",
        "torch-distributed-checkpoint-optimizer-v1",
        "torch-rng-state-per-rank-v1",
    }
)
SUPPORTED_STANDARD_PROFILES = frozenset(
    {
        ("deepspeed", "deepspeed_stage_2", "full"),
        ("pytorch", "fsdp2", "sharded"),
        ("pytorch", "single_process", "full"),
        ("pytorch-lightning", "ddp", "full"),
        ("pytorch-lightning", "single_device", "full"),
    }
)


class CheckpointContractError(ValueError):
    """Raised when a checkpoint cannot satisfy the declared storage contract."""


@dataclass(frozen=True)
class ArtifactRecord:
    """Immutable identity for one serialized checkpoint artifact."""

    path: str
    serialization: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        _validate_relative_path(self.path, "artifact path")
        _validate_trimmed_string(self.serialization, "artifact serialization")
        _validate_sha256(self.sha256, "artifact sha256")
        _validate_non_negative_int(self.size_bytes, "artifact size_bytes")
        if self.size_bytes == 0:
            raise CheckpointContractError("artifact size_bytes must be positive")

    @classmethod
    def from_file(
        cls,
        checkpoint_dir: Path,
        relative_path: str,
        *,
        serialization: str,
    ) -> ArtifactRecord:
        root = _checkpoint_root(checkpoint_dir)
        path = _artifact_path(root, relative_path)
        size_bytes, sha256 = _file_identity(path)
        return cls(
            path=relative_path,
            serialization=serialization,
            sha256=sha256,
            size_bytes=size_bytes,
        )

    @classmethod
    def from_tree(
        cls,
        checkpoint_dir: Path,
        relative_path: str,
        *,
        serialization: str,
    ) -> ArtifactRecord:
        if serialization not in _TREE_STATE_SERIALIZATIONS:
            raise CheckpointContractError(
                "artifact tree requires a registered tree serialization"
            )
        root = _checkpoint_root(checkpoint_dir)
        path = _artifact_tree_path(root, relative_path)
        size_bytes, sha256 = _tree_identity(path)
        return cls(
            path=relative_path,
            serialization=serialization,
            sha256=sha256,
            size_bytes=size_bytes,
        )

    @classmethod
    def from_dict(cls, raw: object) -> ArtifactRecord:
        value = _exact_mapping(
            raw,
            {"path", "serialization", "sha256", "size_bytes"},
            "artifact",
        )
        return cls(
            path=value["path"],
            serialization=value["serialization"],
            sha256=value["sha256"],
            size_bytes=value["size_bytes"],
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "serialization": self.serialization,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True)
class BackendIdentity:
    """Framework identity that owns the serialized optimizer/model layout."""

    name: str
    version: str
    strategy: str
    world_size: int
    state_dict_type: str

    def __post_init__(self) -> None:
        _validate_trimmed_string(self.name, "backend name")
        _validate_trimmed_string(self.version, "backend version")
        _validate_trimmed_string(self.strategy, "backend strategy")
        _validate_non_negative_int(self.world_size, "backend world_size")
        if self.world_size == 0:
            raise CheckpointContractError("backend world_size must be positive")
        if self.state_dict_type not in SUPPORTED_STATE_DICT_TYPES:
            raise CheckpointContractError(
                "backend state_dict_type must be one of: "
                + ", ".join(sorted(SUPPORTED_STATE_DICT_TYPES))
            )

    @classmethod
    def from_dict(cls, raw: object) -> BackendIdentity:
        value = _exact_mapping(
            raw,
            {"name", "version", "strategy", "world_size", "state_dict_type"},
            "backend identity",
        )
        return cls(
            name=value["name"],
            version=value["version"],
            strategy=value["strategy"],
            world_size=value["world_size"],
            state_dict_type=value["state_dict_type"],
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "version": self.version,
            "strategy": self.strategy,
            "world_size": self.world_size,
            "state_dict_type": self.state_dict_type,
        }


@dataclass(frozen=True)
class TrainingProgress:
    """Resume position supported by the epoch-boundary v1 contract."""

    global_step: int
    epoch: int
    step_in_epoch: int

    def __post_init__(self) -> None:
        _validate_non_negative_int(self.global_step, "progress global_step")
        _validate_non_negative_int(self.epoch, "progress epoch")
        _validate_non_negative_int(self.step_in_epoch, "progress step_in_epoch")
        if self.step_in_epoch != 0:
            raise CheckpointContractError(
                "checkpoint schema v1 only supports epoch-boundary data cursors"
            )

    @classmethod
    def from_dict(cls, raw: object) -> TrainingProgress:
        value = _exact_mapping(
            raw,
            {"global_step", "epoch", "step_in_epoch"},
            "training progress",
        )
        return cls(
            global_step=value["global_step"],
            epoch=value["epoch"],
            step_in_epoch=value["step_in_epoch"],
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "step_in_epoch": self.step_in_epoch,
        }


@dataclass(frozen=True)
class CheckpointManifest:
    """Complete, backend-owned training state required for an exact resume."""

    backend: BackendIdentity
    progress: TrainingProgress
    training_config: ArtifactRecord
    states: Mapping[str, ArtifactRecord]
    format: str = CHECKPOINT_FORMAT
    schema_version: int = CHECKPOINT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.backend, BackendIdentity):
            raise CheckpointContractError(
                "checkpoint backend must be a BackendIdentity"
            )
        if not isinstance(self.progress, TrainingProgress):
            raise CheckpointContractError(
                "checkpoint progress must be TrainingProgress"
            )
        if not isinstance(self.training_config, ArtifactRecord):
            raise CheckpointContractError(
                "checkpoint training_config must be an ArtifactRecord"
            )
        if self.format != CHECKPOINT_FORMAT:
            raise CheckpointContractError(
                f"checkpoint format must be {CHECKPOINT_FORMAT!r}"
            )
        if (
            isinstance(self.schema_version, bool)
            or self.schema_version != CHECKPOINT_SCHEMA_VERSION
        ):
            raise CheckpointContractError(
                f"checkpoint schema_version must be {CHECKPOINT_SCHEMA_VERSION}"
            )
        if not isinstance(self.states, Mapping):
            raise CheckpointContractError("checkpoint states must be a mapping")
        if any(not isinstance(name, str) for name in self.states):
            raise CheckpointContractError(
                "checkpoint state component names must be strings"
            )
        expected = set(REQUIRED_STATE_COMPONENTS)
        actual = set(self.states)
        if actual != expected:
            missing = sorted(expected - actual)
            unknown = sorted(actual - expected)
            details = []
            if missing:
                details.append("missing: " + ", ".join(missing))
            if unknown:
                details.append("unknown: " + ", ".join(unknown))
            raise CheckpointContractError(
                "checkpoint states must contain exactly the required components"
                + (" (" + "; ".join(details) + ")" if details else "")
            )
        if not all(
            isinstance(record, ArtifactRecord) for record in self.states.values()
        ):
            raise CheckpointContractError(
                "checkpoint states must contain ArtifactRecord values"
            )
        if self.training_config.serialization != "canonical-json":
            raise CheckpointContractError(
                "training_config serialization must be canonical-json"
            )
        for (
            name,
            supported_serializations,
        ) in _SUPPORTED_STATE_SERIALIZATION_SETS.items():
            actual_serialization = self.states[name].serialization
            if actual_serialization not in supported_serializations:
                raise CheckpointContractError(
                    f"checkpoint {name} serialization must be one of: "
                    + ", ".join(sorted(supported_serializations))
                )
        sharded_serializations = {
            "model": "torch-distributed-checkpoint-model-v1",
            "optimizer": "torch-distributed-checkpoint-optimizer-v1",
            "rng": "torch-rng-state-per-rank-v1",
            "data_cursor": "rwkv-binidx-epoch-cursor-sharded-v1",
        }
        sharded_components = {
            name
            for name, serialization in sharded_serializations.items()
            if self.states[name].serialization == serialization
        }
        sharded_backend = (
            self.backend.name,
            self.backend.strategy,
            self.backend.state_dict_type,
        ) == ("pytorch", "fsdp2", "sharded")
        if sharded_components and sharded_components != set(sharded_serializations):
            raise CheckpointContractError(
                "FSDP2 sharded state serializations must be selected together"
            )
        if bool(sharded_components) != sharded_backend:
            raise CheckpointContractError(
                "FSDP2 sharded state serializations must match backend identity"
            )
        paths = [self.training_config.path]
        paths.extend(record.path for record in self.states.values())
        if len(paths) != len(set(paths)):
            raise CheckpointContractError("checkpoint artifact paths must be unique")
        object.__setattr__(self, "states", MappingProxyType(dict(self.states)))

    @classmethod
    def from_dict(cls, raw: object) -> CheckpointManifest:
        value = _exact_mapping(
            raw,
            {
                "format",
                "schema_version",
                "backend",
                "progress",
                "training_config",
                "states",
            },
            "checkpoint manifest",
        )
        states = _mapping(value["states"], "checkpoint states")
        return cls(
            format=value["format"],
            schema_version=value["schema_version"],
            backend=BackendIdentity.from_dict(value["backend"]),
            progress=TrainingProgress.from_dict(value["progress"]),
            training_config=ArtifactRecord.from_dict(value["training_config"]),
            states={
                name: ArtifactRecord.from_dict(record)
                for name, record in states.items()
            },
        )

    @classmethod
    def read(cls, path: Path) -> CheckpointManifest:
        if path.is_symlink() or not path.is_file():
            raise CheckpointContractError(
                f"checkpoint manifest must be a regular file: {path}"
            )
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise CheckpointContractError(
                f"checkpoint manifest is not valid UTF-8 JSON: {path}"
            ) from error
        return cls.from_dict(raw)

    def to_dict(self) -> dict[str, object]:
        return {
            "format": self.format,
            "schema_version": self.schema_version,
            "backend": self.backend.to_dict(),
            "progress": self.progress.to_dict(),
            "training_config": self.training_config.to_dict(),
            "states": {
                name: self.states[name].to_dict() for name in REQUIRED_STATE_COMPONENTS
            },
        }

    def canonical_json(self) -> bytes:
        return json.dumps(
            self.to_dict(),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")

    def identity_sha256(self) -> str:
        return hashlib.sha256(self.canonical_json()).hexdigest()

    def verify_artifacts(self, checkpoint_dir: Path) -> None:
        root = _checkpoint_root(checkpoint_dir)
        records = [self.training_config, *self.states.values()]
        for record in records:
            size_bytes, sha256 = _artifact_record_identity(root, record)
            if size_bytes != record.size_bytes:
                raise CheckpointContractError(
                    f"checkpoint artifact size does not match manifest: {record.path}"
                )
            if sha256 != record.sha256:
                raise CheckpointContractError(
                    f"checkpoint artifact digest does not match manifest: {record.path}"
                )
        training_config = _read_canonical_json_object(
            _artifact_path(root, self.training_config.path),
            "training config",
        )
        scheduler = _read_canonical_json_object(
            _artifact_path(root, self.states["scheduler"].path),
            "scheduler state",
        )
        scheduler_serialization = self.states["scheduler"].serialization
        expected_scheduler_fields = (
            {"global_step"}
            if scheduler_serialization == "rwkv-callback-schedule-v1"
            else {"global_step", "state_dict"}
        )
        if set(scheduler) != expected_scheduler_fields:
            raise CheckpointContractError("scheduler state has invalid fields")
        scheduler_global_step = scheduler["global_step"]
        _validate_non_negative_int(
            scheduler_global_step,
            "scheduler state global_step",
        )
        if scheduler_global_step != self.progress.global_step:
            raise CheckpointContractError(
                "scheduler state must exactly match progress global_step"
            )
        if scheduler_serialization == "torch-lr-scheduler-json-v1":
            scheduler_state_dict = scheduler["state_dict"]
            if not isinstance(scheduler_state_dict, dict) or any(
                not isinstance(key, str) for key in scheduler_state_dict
            ):
                raise CheckpointContractError(
                    "torch scheduler state_dict must be a JSON object"
                )
        gradient_scaler = _read_canonical_json_object(
            _artifact_path(root, self.states["gradient_scaler"].path),
            "gradient scaler state",
        )
        _validate_gradient_scaler_artifact(gradient_scaler)
        if "precision" in training_config:
            requires_gradient_scaler = training_config["precision"] == 16
            if gradient_scaler["enabled"] != requires_gradient_scaler:
                raise CheckpointContractError(
                    "gradient scaler enabled state must match training precision"
                )
        data_cursor = _read_canonical_json_object(
            _artifact_path(root, self.states["data_cursor"].path),
            "data cursor",
        )
        cursor_fields = {
            "next_epoch",
            "samples_per_epoch",
            "step_in_epoch",
            "world_size",
        }
        sharded_cursor = (
            self.states["data_cursor"].serialization
            == "rwkv-binidx-epoch-cursor-sharded-v1"
        )
        if sharded_cursor:
            cursor_fields.add("rank_ownership")
        if set(data_cursor) != cursor_fields:
            raise CheckpointContractError("data cursor has invalid fields")
        for field in (
            "next_epoch",
            "samples_per_epoch",
            "step_in_epoch",
            "world_size",
        ):
            _validate_non_negative_int(
                data_cursor[field],
                f"data cursor {field}",
            )
        samples_per_epoch = data_cursor["samples_per_epoch"]
        if (
            data_cursor["next_epoch"] != self.progress.epoch
            or data_cursor["step_in_epoch"] != self.progress.step_in_epoch
            or data_cursor["world_size"] != self.backend.world_size
            or samples_per_epoch <= 0
        ):
            raise CheckpointContractError(
                "data cursor must match progress and backend world_size"
            )
        if sharded_cursor:
            _verify_sharded_rank_ownership(root, self, data_cursor["rank_ownership"])

    def write(self, checkpoint_dir: Path) -> Path:
        root = _checkpoint_root(checkpoint_dir)
        self.verify_artifacts(root)
        path = root / CHECKPOINT_MANIFEST_FILENAME
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError as error:
            raise CheckpointContractError(
                f"checkpoint manifest already exists: {path}"
            ) from error
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(self.canonical_json())
                stream.write(b"\n")
                stream.flush()
                os.fsync(stream.fileno())
        except Exception:
            path.unlink(missing_ok=True)
            raise
        return path


class CheckpointLoadKind(str, Enum):
    STANDARD_V1 = "standard_v1"
    LEGACY_PTH_FOR_CONVERSION = "legacy_pth_for_conversion"


@dataclass(frozen=True)
class CheckpointLoadPlan:
    """Validated loader boundary without backend-specific restore behavior."""

    kind: CheckpointLoadKind
    source: Path
    manifest: CheckpointManifest | None
    complete_resume: bool
    requires_conversion: bool


def select_checkpoint_loader(
    path: Path,
    *,
    expected_backend: BackendIdentity | None = None,
    allow_legacy_pth_for_conversion: bool = False,
) -> CheckpointLoadPlan:
    """Select a standard loader profile or an opaque legacy converter input."""

    if path.is_symlink():
        raise CheckpointContractError(
            f"checkpoint source must not be a symlink: {path}"
        )
    if path.is_dir():
        checkpoint_dir = _checkpoint_root(path)
        manifest_path = checkpoint_dir / CHECKPOINT_MANIFEST_FILENAME
    elif path.is_file() and path.name == CHECKPOINT_MANIFEST_FILENAME:
        manifest_path = path
        checkpoint_dir = _checkpoint_root(path.parent)
    elif path.is_file() and path.suffix == ".pth":
        if not allow_legacy_pth_for_conversion:
            raise CheckpointContractError(
                "legacy .pth content is unverified; opt in explicitly to the "
                "legacy converter boundary"
            )
        if expected_backend is not None:
            raise CheckpointContractError(
                "legacy .pth content has no verifiable backend identity"
            )
        return CheckpointLoadPlan(
            kind=CheckpointLoadKind.LEGACY_PTH_FOR_CONVERSION,
            source=path.resolve(),
            manifest=None,
            complete_resume=False,
            requires_conversion=True,
        )
    elif path.exists():
        raise CheckpointContractError(f"unsupported checkpoint source: {path}")
    else:
        raise CheckpointContractError(f"checkpoint source does not exist: {path}")

    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise CheckpointContractError(
            "standard checkpoint directory requires a regular "
            f"{CHECKPOINT_MANIFEST_FILENAME}"
        )
    manifest = CheckpointManifest.read(manifest_path)
    loader_profile = (
        manifest.backend.name,
        manifest.backend.strategy,
        manifest.backend.state_dict_type,
    )
    if loader_profile not in SUPPORTED_STANDARD_PROFILES:
        raise CheckpointContractError(
            "checkpoint backend profile has no registered standard loader: "
            + "/".join(loader_profile)
        )
    if expected_backend is not None and manifest.backend != expected_backend:
        raise CheckpointContractError(
            "checkpoint backend identity does not match the requested runtime"
        )
    manifest.verify_artifacts(checkpoint_dir)
    return CheckpointLoadPlan(
        kind=CheckpointLoadKind.STANDARD_V1,
        source=checkpoint_dir,
        manifest=manifest,
        complete_resume=True,
        requires_conversion=False,
    )


def _mapping(raw: object, owner: str) -> Mapping[str, object]:
    if not isinstance(raw, dict) or any(not isinstance(key, str) for key in raw):
        raise CheckpointContractError(f"{owner} must be a JSON object")
    return raw


def _exact_mapping(
    raw: object,
    expected_fields: set[str],
    owner: str,
) -> Mapping[str, object]:
    value = _mapping(raw, owner)
    if set(value) != expected_fields:
        missing = sorted(expected_fields - set(value))
        unknown = sorted(set(value) - expected_fields)
        details = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if unknown:
            details.append("unknown: " + ", ".join(unknown))
        raise CheckpointContractError(
            f"{owner} has invalid fields"
            + (" (" + "; ".join(details) + ")" if details else "")
        )
    return value


def _validate_trimmed_string(value: object, name: str) -> None:
    if not isinstance(value, str) or not value or value != value.strip():
        raise CheckpointContractError(f"{name} must be a non-empty trimmed string")


def _validate_non_negative_int(value: object, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise CheckpointContractError(f"{name} must be a non-negative integer")


def _validate_sha256(value: object, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise CheckpointContractError(f"{name} must be a lowercase SHA-256")


def _validate_relative_path(value: object, name: str) -> None:
    if not isinstance(value, str) or not value or "\\" in value:
        raise CheckpointContractError(f"{name} must be a normalized relative path")
    path = PurePosixPath(value)
    if (
        value != value.strip()
        or path.is_absolute()
        or "." in path.parts
        or ".." in path.parts
        or str(path) != value
    ):
        raise CheckpointContractError(f"{name} must be a normalized relative path")


def _checkpoint_root(path: Path) -> Path:
    if path.is_symlink() or not path.is_dir():
        raise CheckpointContractError(
            f"checkpoint root must be a regular directory: {path}"
        )
    return path.resolve()


def _artifact_path(root: Path, relative_path: str) -> Path:
    _validate_relative_path(relative_path, "artifact path")
    path = root
    for part in PurePosixPath(relative_path).parts:
        path /= part
        if path.is_symlink():
            raise CheckpointContractError(
                f"checkpoint artifact path must not use symlinks: {relative_path}"
            )
    try:
        status = path.stat()
        path.resolve().relative_to(root)
    except (FileNotFoundError, OSError, ValueError) as error:
        raise CheckpointContractError(
            f"checkpoint artifact is missing or outside the checkpoint: {relative_path}"
        ) from error
    if not stat.S_ISREG(status.st_mode):
        raise CheckpointContractError(
            f"checkpoint artifact must be a regular file: {relative_path}"
        )
    return path


def _artifact_tree_path(root: Path, relative_path: str) -> Path:
    _validate_relative_path(relative_path, "artifact path")
    path = root
    for part in PurePosixPath(relative_path).parts:
        path /= part
        if path.is_symlink():
            raise CheckpointContractError(
                f"checkpoint artifact tree must not use symlinks: {relative_path}"
            )
    try:
        path.resolve().relative_to(root)
    except (FileNotFoundError, OSError, ValueError) as error:
        raise CheckpointContractError(
            "checkpoint artifact tree is missing or outside the checkpoint: "
            f"{relative_path}"
        ) from error
    if not path.is_dir():
        raise CheckpointContractError(
            f"checkpoint artifact tree must be a directory: {relative_path}"
        )
    return path


def _tree_files(tree: Path) -> list[tuple[str, Path]]:
    files = []
    for path in sorted(tree.rglob("*")):
        if path.is_symlink():
            raise CheckpointContractError(
                f"checkpoint artifact tree must not use symlinks: {path}"
            )
        if path.is_dir():
            continue
        if not path.is_file():
            raise CheckpointContractError(
                f"checkpoint artifact tree entry must be regular: {path}"
            )
        files.append((path.relative_to(tree).as_posix(), path))
    if not files:
        raise CheckpointContractError(
            f"checkpoint artifact tree must contain files: {tree}"
        )
    return files


def _tree_identity(tree: Path) -> tuple[int, str]:
    inventory = []
    size_bytes = 0
    for relative_path, path in _tree_files(tree):
        file_size, sha256 = _file_identity(path)
        size_bytes += file_size
        inventory.append(
            {
                "path": relative_path,
                "sha256": sha256,
                "size_bytes": file_size,
            }
        )
    if size_bytes <= 0:
        raise CheckpointContractError(
            f"checkpoint artifact tree must contain non-empty data: {tree}"
        )
    identity = json.dumps(
        {"format": "rwkv-checkpoint-tree-v1", "files": inventory},
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return size_bytes, hashlib.sha256(identity).hexdigest()


def _artifact_record_identity(
    root: Path,
    record: ArtifactRecord,
) -> tuple[int, str]:
    if record.serialization in _TREE_STATE_SERIALIZATIONS:
        return _tree_identity(_artifact_tree_path(root, record.path))
    return _file_identity(_artifact_path(root, record.path))


def _tree_checkpoint_paths(root: Path, relative_path: str) -> set[str]:
    tree = _artifact_tree_path(root, relative_path)
    return {
        str(PurePosixPath(relative_path) / file_path)
        for file_path, _ in _tree_files(tree)
    }


def _verify_sharded_rank_ownership(
    root: Path,
    manifest: CheckpointManifest,
    raw: object,
) -> None:
    if (
        manifest.backend.name,
        manifest.backend.strategy,
        manifest.backend.state_dict_type,
    ) != ("pytorch", "fsdp2", "sharded"):
        raise CheckpointContractError(
            "sharded data cursor requires a pytorch/fsdp2/sharded backend"
        )
    expected_serializations = {
        "model": "torch-distributed-checkpoint-model-v1",
        "optimizer": "torch-distributed-checkpoint-optimizer-v1",
        "rng": "torch-rng-state-per-rank-v1",
    }
    for component, serialization in expected_serializations.items():
        if manifest.states[component].serialization != serialization:
            raise CheckpointContractError(
                f"sharded data cursor requires {component} serialization "
                f"{serialization}"
            )

    ownership = _exact_mapping(
        raw,
        {"coordinator_rank", "model", "optimizer", "rng"},
        "rank ownership",
    )
    if (
        isinstance(ownership["coordinator_rank"], bool)
        or ownership["coordinator_rank"] != 0
    ):
        raise CheckpointContractError("rank ownership coordinator_rank must be 0")
    expected_ranks = {str(rank) for rank in range(manifest.backend.world_size)}

    for component in ("model", "optimizer"):
        record = manifest.states[component]
        component_ownership = _exact_mapping(
            ownership[component],
            {"metadata", "shards"},
            f"{component} rank ownership",
        )
        metadata_path = f"{record.path}/.metadata"
        if component_ownership["metadata"] != metadata_path:
            raise CheckpointContractError(
                f"{component} rank ownership metadata path is invalid"
            )
        shards = _mapping(
            component_ownership["shards"],
            f"{component} rank shards",
        )
        if set(shards) != expected_ranks:
            raise CheckpointContractError(
                f"{component} rank ownership must cover every rank"
            )
        declared_paths = {metadata_path}
        for rank in sorted(expected_ranks, key=int):
            rank_paths = shards[rank]
            if (
                not isinstance(rank_paths, list)
                or not rank_paths
                or any(not isinstance(path, str) for path in rank_paths)
            ):
                raise CheckpointContractError(
                    f"{component} rank {rank} must own a non-empty shard list"
                )
            for path in rank_paths:
                _validate_relative_path(path, f"{component} rank shard path")
                if (
                    not path.startswith(f"{record.path}/")
                    or not PurePosixPath(path).name.startswith(f"__{rank}_")
                    or not path.endswith(".distcp")
                ):
                    raise CheckpointContractError(
                        f"{component} rank {rank} shard ownership is invalid"
                    )
                if path in declared_paths:
                    raise CheckpointContractError(
                        f"{component} rank shard ownership must be unique"
                    )
                declared_paths.add(path)
        if declared_paths != _tree_checkpoint_paths(root, record.path):
            raise CheckpointContractError(
                f"{component} rank ownership does not match its artifact tree"
            )

    rng_record = manifest.states["rng"]
    rng_ownership = _mapping(ownership["rng"], "RNG rank ownership")
    if set(rng_ownership) != expected_ranks:
        raise CheckpointContractError("RNG rank ownership must cover every rank")
    rng_paths = set()
    for rank in sorted(expected_ranks, key=int):
        expected_path = f"{rng_record.path}/rank-{int(rank):05d}.pt"
        if rng_ownership[rank] != expected_path:
            raise CheckpointContractError(f"RNG rank {rank} ownership path is invalid")
        rng_paths.add(expected_path)
    if rng_paths != _tree_checkpoint_paths(root, rng_record.path):
        raise CheckpointContractError(
            "RNG rank ownership does not match its artifact tree"
        )


def _file_identity(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size_bytes = 0
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                size_bytes += len(chunk)
                digest.update(chunk)
    except OSError as error:
        raise CheckpointContractError(
            f"checkpoint artifact cannot be read: {path}"
        ) from error
    return size_bytes, digest.hexdigest()


def _read_canonical_json_object(path: Path, owner: str) -> Mapping[str, object]:
    try:
        payload = path.read_bytes()
        raw = json.loads(payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CheckpointContractError(f"{owner} must be valid UTF-8 JSON") from error
    value = _mapping(raw, owner)
    try:
        canonical = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise CheckpointContractError(
            f"{owner} must contain canonical JSON values"
        ) from error
    if payload not in {canonical, canonical + b"\n"}:
        raise CheckpointContractError(f"{owner} must use canonical JSON")
    return value


def _validate_gradient_scaler_artifact(
    payload: Mapping[str, object],
) -> None:
    if set(payload) != {"enabled", "state_dict"}:
        raise CheckpointContractError("gradient scaler state has invalid fields")
    enabled = payload["enabled"]
    state_dict = payload["state_dict"]
    if not isinstance(enabled, bool) or not isinstance(state_dict, dict):
        raise CheckpointContractError("gradient scaler state has invalid values")
    if not enabled:
        if state_dict:
            raise CheckpointContractError(
                "disabled gradient scaler state_dict must be empty"
            )
        return
    expected_fields = {
        "scale",
        "growth_factor",
        "backoff_factor",
        "growth_interval",
        "_growth_tracker",
    }
    if set(state_dict) != expected_fields:
        raise CheckpointContractError(
            "enabled gradient scaler state_dict has invalid fields"
        )
    scale = state_dict["scale"]
    growth_factor = state_dict["growth_factor"]
    backoff_factor = state_dict["backoff_factor"]
    growth_interval = state_dict["growth_interval"]
    growth_tracker = state_dict["_growth_tracker"]
    if (
        isinstance(scale, bool)
        or not isinstance(scale, (int, float))
        or not scale > 0
        or isinstance(growth_factor, bool)
        or not isinstance(growth_factor, (int, float))
        or not growth_factor > 1
        or isinstance(backoff_factor, bool)
        or not isinstance(backoff_factor, (int, float))
        or not 0 < backoff_factor < 1
        or isinstance(growth_interval, bool)
        or not isinstance(growth_interval, int)
        or growth_interval <= 0
        or isinstance(growth_tracker, bool)
        or not isinstance(growth_tracker, int)
        or growth_tracker < 0
    ):
        raise CheckpointContractError(
            "enabled gradient scaler state_dict has invalid values"
        )


__all__ = [
    "CHECKPOINT_FORMAT",
    "CHECKPOINT_MANIFEST_FILENAME",
    "CHECKPOINT_SCHEMA_VERSION",
    "REQUIRED_STATE_COMPONENTS",
    "SUPPORTED_DATA_CURSOR_SERIALIZATIONS",
    "SUPPORTED_GRADIENT_SCALER_SERIALIZATIONS",
    "SUPPORTED_MODEL_SERIALIZATIONS",
    "SUPPORTED_OPTIMIZER_SERIALIZATIONS",
    "SUPPORTED_RNG_SERIALIZATIONS",
    "SUPPORTED_SCHEDULER_SERIALIZATIONS",
    "SUPPORTED_STANDARD_PROFILES",
    "SUPPORTED_STATE_SERIALIZATIONS",
    "ArtifactRecord",
    "BackendIdentity",
    "CheckpointContractError",
    "CheckpointLoadKind",
    "CheckpointLoadPlan",
    "CheckpointManifest",
    "TrainingProgress",
    "select_checkpoint_loader",
]
