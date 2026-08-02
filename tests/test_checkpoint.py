from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from rwkv_lm.checkpoint import (
    ArtifactRecord,
    BackendIdentity,
    CHECKPOINT_MANIFEST_FILENAME,
    CheckpointContractError,
    CheckpointLoadKind,
    CheckpointManifest,
    TrainingProgress,
    select_checkpoint_loader,
)


STATE_FILES = {
    "model": ("model.pt", "torch-state-dict", b"model-state"),
    "optimizer": ("optimizer.pt", "torch-optimizer-state", b"optimizer-state"),
    "scheduler": (
        "scheduler.json",
        "rwkv-callback-schedule-v1",
        b'{"global_step":12}',
    ),
    "rng": ("rng.pt", "torch-rng-state", b"rng-state"),
    "data_cursor": (
        "data-cursor.json",
        "rwkv-binidx-epoch-cursor-v1",
        (
            b'{"next_epoch":3,"samples_per_epoch":40320,'
            b'"step_in_epoch":0,"world_size":2}'
        ),
    ),
}


def _backend() -> BackendIdentity:
    return BackendIdentity(
        name="deepspeed",
        version="0.17.6",
        strategy="deepspeed_stage_2",
        world_size=2,
        state_dict_type="full",
    )


def _manifest(checkpoint_dir: Path) -> CheckpointManifest:
    config_path = checkpoint_dir / "training-config.json"
    config_path.write_text('{"ctx_len":1024,"precision":"bf16"}', encoding="utf-8")
    states = {}
    for name, (relative_path, serialization, payload) in STATE_FILES.items():
        (checkpoint_dir / relative_path).write_bytes(payload)
        states[name] = ArtifactRecord.from_file(
            checkpoint_dir,
            relative_path,
            serialization=serialization,
        )
    return CheckpointManifest(
        backend=_backend(),
        progress=TrainingProgress(global_step=12, epoch=3, step_in_epoch=0),
        training_config=ArtifactRecord.from_file(
            checkpoint_dir,
            "training-config.json",
            serialization="canonical-json",
        ),
        states=states,
    )


def test_standard_manifest_round_trip_and_loader_selection(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    manifest_path = manifest.write(tmp_path)

    loaded_from_dir = select_checkpoint_loader(
        tmp_path,
        expected_backend=_backend(),
    )
    loaded_from_manifest = select_checkpoint_loader(
        manifest_path,
        expected_backend=_backend(),
    )

    assert loaded_from_dir == loaded_from_manifest
    assert loaded_from_dir.kind is CheckpointLoadKind.STANDARD_V1
    assert loaded_from_dir.complete_resume is True
    assert loaded_from_dir.requires_conversion is False
    assert loaded_from_dir.manifest == manifest
    assert CheckpointManifest.from_dict(
        json.loads(manifest.canonical_json())
    ) == manifest
    assert len(manifest.identity_sha256()) == 64


def test_legacy_pth_is_only_an_explicit_opaque_converter_input(
    tmp_path: Path,
) -> None:
    weights = tmp_path / "rwkv-3.pth"
    weights.write_bytes(b"legacy model weights")

    with pytest.raises(CheckpointContractError, match="unverified"):
        select_checkpoint_loader(weights)

    plan = select_checkpoint_loader(
        weights,
        allow_legacy_pth_for_conversion=True,
    )

    assert plan.kind is CheckpointLoadKind.LEGACY_PTH_FOR_CONVERSION
    assert plan.complete_resume is False
    assert plan.requires_conversion is True
    assert plan.manifest is None


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda raw: raw.update(format="unknown"), "format"),
        (lambda raw: raw.update(schema_version=2), "schema_version"),
        (lambda raw: raw.update(extra=True), "invalid fields"),
        (lambda raw: raw["states"].pop("optimizer"), "required components"),
        (
            lambda raw: raw["states"]["rng"].update(path="../rng.pt"),
            "relative path",
        ),
        (
            lambda raw: raw["states"]["rng"].update(sha256="A" * 64),
            "lowercase SHA-256",
        ),
        (
            lambda raw: raw["states"]["scheduler"].update(
                serialization="torch-state-dict"
            ),
            "scheduler serialization",
        ),
        (
            lambda raw: raw["backend"].update(world_size=True),
            "world_size",
        ),
        (
            lambda raw: raw["progress"].update(step_in_epoch=1),
            "epoch-boundary",
        ),
    ],
)
def test_manifest_decision_table_rejects_ambiguous_or_incomplete_state(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    raw = copy.deepcopy(_manifest(tmp_path).to_dict())
    mutation(raw)

    with pytest.raises(CheckpointContractError, match=message):
        CheckpointManifest.from_dict(raw)


def test_standard_loader_rejects_tampering_and_backend_mismatch(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    manifest.write(tmp_path)
    (tmp_path / "optimizer.pt").write_bytes(b"tampered")

    with pytest.raises(CheckpointContractError, match="artifact (size|digest)"):
        select_checkpoint_loader(tmp_path)

    (tmp_path / "optimizer.pt").write_bytes(STATE_FILES["optimizer"][2])
    incompatible = BackendIdentity(
        name="deepspeed",
        version="0.17.6",
        strategy="deepspeed_stage_3",
        world_size=2,
        state_dict_type="sharded",
    )
    with pytest.raises(CheckpointContractError, match="backend identity"):
        select_checkpoint_loader(tmp_path, expected_backend=incompatible)


def test_standard_loader_cross_checks_scheduler_and_data_cursor(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    cursor_path = tmp_path / "data-cursor.json"
    cursor_path.write_text(
        (
            '{"next_epoch":4,"samples_per_epoch":40320,'
            '"step_in_epoch":0,"world_size":2}'
        ),
        encoding="utf-8",
    )
    states = dict(manifest.states)
    states["data_cursor"] = ArtifactRecord.from_file(
        tmp_path,
        "data-cursor.json",
        serialization="rwkv-binidx-epoch-cursor-v1",
    )
    inconsistent = CheckpointManifest(
        backend=manifest.backend,
        progress=manifest.progress,
        training_config=manifest.training_config,
        states=states,
    )

    with pytest.raises(CheckpointContractError, match="data cursor must match"):
        inconsistent.write(tmp_path)


@pytest.mark.parametrize(
    ("component", "payload", "message"),
    [
        ("scheduler", b'{"global_step":12.0}', "scheduler state global_step"),
        (
            "data_cursor",
            (
                b'{"next_epoch":3,"samples_per_epoch":40320,'
                b'"step_in_epoch":false,"world_size":2}'
            ),
            "data cursor step_in_epoch",
        ),
    ],
)
def test_standard_loader_rejects_non_integer_artifact_progress(
    tmp_path: Path,
    component: str,
    payload: bytes,
    message: str,
) -> None:
    manifest = _manifest(tmp_path)
    relative_path, serialization, _ = STATE_FILES[component]
    (tmp_path / relative_path).write_bytes(payload)
    states = dict(manifest.states)
    states[component] = ArtifactRecord.from_file(
        tmp_path,
        relative_path,
        serialization=serialization,
    )
    malformed = CheckpointManifest(
        backend=manifest.backend,
        progress=manifest.progress,
        training_config=manifest.training_config,
        states=states,
    )

    with pytest.raises(CheckpointContractError, match=message):
        malformed.write(tmp_path)


def test_loader_does_not_fallback_from_invalid_standard(
    tmp_path: Path,
) -> None:
    (tmp_path / "rwkv-3.pth").write_bytes(b"legacy model weights")
    with pytest.raises(CheckpointContractError, match=CHECKPOINT_MANIFEST_FILENAME):
        select_checkpoint_loader(
            tmp_path,
            allow_legacy_pth_for_conversion=True,
        )


@pytest.mark.parametrize(
    "backend_update",
    [
        {"name": "fsdp2", "strategy": "fsdp2"},
        {"strategy": "unknown"},
        {"state_dict_type": "sharded"},
    ],
)
def test_standard_loader_rejects_unregistered_backend_profile(
    tmp_path: Path,
    backend_update: dict[str, str],
) -> None:
    manifest = _manifest(tmp_path)
    raw = manifest.to_dict()
    raw["backend"].update(backend_update)
    (tmp_path / CHECKPOINT_MANIFEST_FILENAME).write_text(
        json.dumps(raw),
        encoding="utf-8",
    )
    with pytest.raises(CheckpointContractError, match="no registered standard loader"):
        select_checkpoint_loader(tmp_path)
