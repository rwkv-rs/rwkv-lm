"""Fail-closed identities for complete RWKV training checkpoints.

This module defines the storage boundary only. It does not save framework state,
restore a trainer, convert legacy weights, or implement FSDP2.
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
    "rng",
    "data_cursor",
)
SUPPORTED_STATE_DICT_TYPES = frozenset({"full", "sharded"})
SUPPORTED_STATE_SERIALIZATIONS = MappingProxyType(
    {
        "model": "torch-state-dict",
        "optimizer": "torch-optimizer-state",
        "scheduler": "rwkv-callback-schedule-v1",
        "rng": "torch-rng-state",
        "data_cursor": "rwkv-binidx-epoch-cursor-v1",
    }
)
SUPPORTED_SCHEDULER_SERIALIZATIONS = frozenset(
    {
        "rwkv-callback-schedule-v1",
        "torch-lr-scheduler-json-v1",
    }
)
SUPPORTED_STANDARD_PROFILES = frozenset(
    {
        ("deepspeed", "deepspeed_stage_2", "full"),
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
        for name, expected_serialization in SUPPORTED_STATE_SERIALIZATIONS.items():
            actual_serialization = self.states[name].serialization
            if name == "scheduler":
                valid = actual_serialization in SUPPORTED_SCHEDULER_SERIALIZATIONS
            else:
                valid = actual_serialization == expected_serialization
            if not valid:
                raise CheckpointContractError(
                    f"checkpoint {name} serialization must be "
                    + (
                        "one of: "
                        + ", ".join(sorted(SUPPORTED_SCHEDULER_SERIALIZATIONS))
                        if name == "scheduler"
                        else expected_serialization
                    )
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
            path = _artifact_path(root, record.path)
            size_bytes, sha256 = _file_identity(path)
            if size_bytes != record.size_bytes:
                raise CheckpointContractError(
                    f"checkpoint artifact size does not match manifest: {record.path}"
                )
            if sha256 != record.sha256:
                raise CheckpointContractError(
                    f"checkpoint artifact digest does not match manifest: {record.path}"
                )
        _read_canonical_json_object(
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
        data_cursor = _read_canonical_json_object(
            _artifact_path(root, self.states["data_cursor"].path),
            "data cursor",
        )
        if set(data_cursor) != {
            "next_epoch",
            "samples_per_epoch",
            "step_in_epoch",
            "world_size",
        }:
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


__all__ = [
    "CHECKPOINT_FORMAT",
    "CHECKPOINT_MANIFEST_FILENAME",
    "CHECKPOINT_SCHEMA_VERSION",
    "REQUIRED_STATE_COMPONENTS",
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
