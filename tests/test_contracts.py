from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.tensor import DeviceMesh, Shard, distribute_tensor
from transformers import RwkvConfig

from rwkv_trainer.binidx import RwkvDataLoader
from rwkv_trainer.config_registry import (
    rwkv7_debug,
    rwkv7_lora,
    rwkv7_lora_infctx,
    rwkv7_pretrain,
    rwkv7_pretrain_infctx,
)
from rwkv_trainer.model import PeftSettings, RwkvModelAdapter
from rwkv_trainer.model_spec import model_registry
from rwkv_trainer.optimizer import RwkvOptimizersContainer
from rwkv_trainer.provenance import dependency_metadata
from rwkv_trainer.state_dict import RwkvStateDictAdapter
from rwkv_trainer.trainer import RwkvTrainer
from rwkv_trainer.validate import validate_tree


def _assets(path: Path) -> Path:
    config = RwkvConfig(
        vocab_size=256,
        context_length=16,
        hidden_size=128,
        num_hidden_layers=2,
        intermediate_size=512,
        head_size=64,
        decay_low_rank_dim=32,
        a_low_rank_dim=32,
        v_low_rank_dim=32,
        gate_low_rank_dim=32,
    )
    config.save_pretrained(path)
    return path


def _binidx(prefix: Path, tokens: np.ndarray) -> None:
    Path(f"{prefix}.bin").write_bytes(tokens.tobytes())
    index = bytearray(b"MMIDIDX\x00\x00")
    index += struct.pack("<Q", 1)
    index += struct.pack("<B", 8)
    index += struct.pack("<Q", 1)
    index += struct.pack("<Q", 1)
    index += np.asarray([tokens.size], dtype=np.int32).tobytes()
    index += np.asarray([0], dtype=np.int64).tobytes()
    index += np.asarray([0], dtype=np.int64).tobytes()
    Path(f"{prefix}.idx").write_bytes(index)


def test_all_torchtitan_registrations_are_available() -> None:
    functions = (
        rwkv7_debug,
        rwkv7_pretrain,
        rwkv7_pretrain_infctx,
        rwkv7_lora,
        rwkv7_lora_infctx,
    )
    assert [fn().model_spec.flavor for fn in functions] == [
        "debug",
        "pretrain",
        "pretrain_infctx",
        "lora",
        "lora_infctx",
    ]
    assert rwkv7_debug()._owner is RwkvTrainer
    with pytest.raises(ValueError, match="Unknown RWKV flavor"):
        model_registry("0.4B")


def test_meta_build_and_canonical_initialization(tmp_path: Path) -> None:
    config = RwkvModelAdapter.Config(hf_assets_path=str(_assets(tmp_path)))
    with torch.device("meta"):
        model = config.build()
    model.verify_module_protocol()
    assert all(parameter.is_meta for parameter in model.parameters())
    model.to_empty(device="cpu")
    model.init_states()
    assert not any(parameter.is_meta for parameter in model.parameters())
    assert torch.count_nonzero(model.rwkv_model.model.blocks[0].att.output.weight) == 0


def test_canonical_initialization_supports_fsdp_dtensors(tmp_path: Path) -> None:
    if not dist.is_available():
        pytest.skip("torch.distributed is unavailable")
    rendezvous = tmp_path / "dtensor-init"
    dist.init_process_group("fake", rank=0, world_size=1, init_method=f"file://{rendezvous}")
    try:
        config = RwkvModelAdapter.Config(hf_assets_path=str(_assets(tmp_path)))
        with torch.device("meta"):
            model = config.build()
        model.to_empty(device="cpu")
        mesh = DeviceMesh("cpu", [0])
        for module in model.modules():
            for name, parameter in tuple(module.named_parameters(recurse=False)):
                full = torch.empty(tuple(parameter.shape), dtype=parameter.dtype)
                sharded = distribute_tensor(full, mesh, [Shard(0)])
                setattr(
                    module,
                    name,
                    torch.nn.Parameter(sharded, requires_grad=parameter.requires_grad),
                )

        model.init_states()

        parameters = tuple(model.parameters())
        assert parameters and all(parameter.isfinite().all() for parameter in parameters)
        output = model.rwkv_model.model.blocks[0].att.output.weight.full_tensor()
        assert torch.count_nonzero(output) == 0
        assert torch.count_nonzero(
            model.rwkv_model.model.blocks[0].att.receptance.weight.full_tensor()
        )
    finally:
        dist.destroy_process_group()


def test_state_dict_adapter_only_changes_outer_prefix(tmp_path: Path) -> None:
    config = RwkvModelAdapter.Config(hf_assets_path=str(_assets(tmp_path)))
    adapter = RwkvStateDictAdapter(config, str(tmp_path))
    tensor = torch.ones(2)
    native = adapter.from_hf({"model.emb.weight": tensor})
    assert native == {"hf_model.model.emb.weight": tensor}
    assert adapter.to_hf(native) == {"model.emb.weight": tensor}


def test_peft_trains_only_four_attention_projection_families(tmp_path: Path) -> None:
    pytest.importorskip("peft")
    model = RwkvModelAdapter(
        RwkvModelAdapter.Config(
            hf_assets_path=str(_assets(tmp_path)),
            peft=PeftSettings(enabled=True),
        )
    )
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    assert len(trainable) == 2 * 4 * 2
    assert all(".att." in name and "lora_" in name for name in trainable)
    assert {name.split(".att.", 1)[1].split(".", 1)[0] for name in trainable} == {
        "receptance",
        "key",
        "value",
        "output",
    }
    optimizer = RwkvOptimizersContainer.Config(implementation="for-loop").build(model_parts=[model])
    optimized = {
        name
        for inner in optimizer.optimizers
        for group in inner.param_groups
        for name in group["param_names"]
    }
    assert optimized == set(trainable)


def test_meta_peft_initialization_restores_lora_defaults(tmp_path: Path) -> None:
    pytest.importorskip("peft")
    config = RwkvModelAdapter.Config(
        hf_assets_path=str(_assets(tmp_path)),
        peft=PeftSettings(enabled=True),
    )
    with torch.device("meta"):
        model = config.build()
    model.to_empty(device="cpu")
    model.init_states()
    lora_a = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and ".lora_A." in name
    ]
    lora_b = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and ".lora_B." in name
    ]
    assert lora_a and all(torch.isfinite(parameter).all() for parameter in lora_a)
    assert all(torch.count_nonzero(parameter) > 0 for parameter in lora_a)
    assert lora_b and all(torch.count_nonzero(parameter) == 0 for parameter in lora_b)


def test_binidx_cursor_round_trip_and_identity_rejection(tmp_path: Path) -> None:
    prefix = tmp_path / "tokens"
    tokens = np.arange(34, dtype=np.uint16)
    _binidx(prefix, tokens)
    config = RwkvDataLoader.Config(
        dataset="binidx",
        dataset_path=str(prefix),
        vocab_size=256,
        magic_prime=11,
    )
    loader = RwkvDataLoader(
        config,
        dp_world_size=1,
        dp_rank=0,
        tokenizer=object(),
        seq_len=3,
        local_batch_size=2,
    )
    first = next(iter(loader))
    state = loader.state_dict()
    resumed = RwkvDataLoader(
        config,
        dp_world_size=1,
        dp_rank=0,
        tokenizer=object(),
        seq_len=3,
        local_batch_size=2,
    )
    resumed.load_state_dict(state)
    assert resumed.cursor == 1
    assert first[0]["input"].shape == first[1].shape == (2, 3)
    Path(f"{prefix}.bin").write_bytes((tokens + 1).tobytes())
    changed = RwkvDataLoader(
        config,
        dp_world_size=1,
        dp_rank=0,
        tokenizer=object(),
        seq_len=3,
        local_batch_size=2,
    )
    with pytest.raises(ValueError, match="identity mismatch"):
        changed.load_state_dict(state)


def test_dcp_round_trip_restores_cursor(tmp_path: Path) -> None:
    config = RwkvDataLoader.Config(dataset="synthetic", vocab_size=64)
    loader = RwkvDataLoader(
        config,
        dp_world_size=1,
        dp_rank=0,
        tokenizer=object(),
        seq_len=4,
        local_batch_size=1,
    )
    next(iter(loader))
    checkpoint = tmp_path / "dcp"
    dcp.save({"dataloader": loader}, checkpoint_id=str(checkpoint))
    resumed = RwkvDataLoader(
        config,
        dp_world_size=1,
        dp_rank=0,
        tokenizer=object(),
        seq_len=4,
        local_batch_size=1,
    )
    dcp.load({"dataloader": resumed}, checkpoint_id=str(checkpoint))
    assert resumed.cursor == loader.cursor == 1


class _OptimizerFixture(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.att = torch.nn.Module()
        self.att.w0 = torch.nn.Parameter(torch.ones(1, 1, 4))
        self.matrix = torch.nn.Linear(4, 4, bias=False)
        self.norm = torch.nn.LayerNorm(4)


def test_optimizer_grouping_matches_train_temp() -> None:
    model = _OptimizerFixture()
    config = RwkvOptimizersContainer.Config(
        implementation="for-loop",
        lr=1e-3,
        weight_decay=0.1,
    )
    container = config.build(model_parts=[model])
    groups = {group["param_names"][0]: group for group in container.optimizers[0].param_groups}
    assert groups["att.w0"]["lr"] == 2e-3
    assert groups["att.w0"]["weight_decay"] == 0
    assert groups["matrix.weight"]["weight_decay"] == 0.1
    assert groups["norm.weight"]["weight_decay"] == 0
    assert all(group["eps"] == 1e-18 for group in groups.values())


def test_no_duplicate_model_tokenizer_or_kernel_sources() -> None:
    assert validate_tree(Path(__file__).parents[1]) == []


def test_launcher_ignores_hostile_generic_environment_names() -> None:
    script = (Path(__file__).parents[1] / "run_train.sh").read_text()
    assert "${MODULE:-" not in script
    assert "${CONFIG:-" not in script
    assert "RWKV_TORCHTITAN_MODULE" in script
    assert "RWKV_TORCHTITAN_CONFIG" in script
    assert "RWKV_TORCHRUN:-${SCRIPT_DIR}/.venv/bin/torchrun" in script


def test_dependency_manifest_has_no_floating_main() -> None:
    manifest = (Path(__file__).parents[1] / "pyproject.toml").read_text()
    assert "@main" not in manifest
    assert "10052072bd3ca172957d97e11b24ae297e0e0072" in manifest
    assert "flashrwkv2==0.1.0a5" in manifest
    lock = (Path(__file__).parents[1] / "uv.lock").read_text()
    assert "sha256:bb8a565084addabf1071c06d398ac7f451919fb4cc3e30f314a8e1442124efdd" in lock
    assert dependency_metadata() == {
        "torchtitan_oid": "96276d86577cf3e3bd29de72586e76af62010a55",
        "transformers_oid": "10052072bd3ca172957d97e11b24ae297e0e0072",
        "flashrwkv2_version": "0.1.0a5",
        "flashrwkv2_oid": "046257e7918d93a0fefce868e2ab580fbf6078da",
        "peft_version": "0.18.0",
        "rwkv_peft_reference_oid": "5704c39f8ab1d2ac63936ab392aadb6ba526e1a5",
    }
    metadata = json.loads(json.dumps(model_registry("pretrain").model.to_dict()))
    assert "hf_assets_path" in metadata
