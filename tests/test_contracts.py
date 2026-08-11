from __future__ import annotations

import json
import os
import struct
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

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
from rwkv_trainer.export import (
    _assert_logits_close,
    _dcp_model_state,
    _deterministic_cuda_validation,
    _require_uniform_floating_dtype,
)
from rwkv_trainer.loss import RwkvL2WrapLoss
from rwkv_trainer.model import PeftSettings, RwkvModelAdapter
from rwkv_trainer.model_spec import model_registry
from rwkv_trainer.optimizer import RwkvOptimizersContainer
from rwkv_trainer.parallelize import parallelize_rwkv
from rwkv_trainer.provenance import artifact_hashes, dependency_metadata
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
    nparams, flops_per_token = config.get_nparams_and_flops(model, seq_len=16)
    assert nparams == sum(parameter.numel() for parameter in model.parameters())
    assert flops_per_token == 6 * (nparams - model.rwkv_model.model.emb.weight.numel())
    assert flops_per_token > 0


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
        assert hasattr(dist, "set_timeout")

        class TimeoutGroup:
            timeout: timedelta | None = None

            def set_timeout(self, timeout: timedelta) -> None:
                self.timeout = timeout

        timeout_group = TimeoutGroup()
        dist.set_timeout(  # type: ignore[attr-defined, arg-type]
            timedelta(seconds=10), timeout_group
        )
        assert timeout_group.timeout == timedelta(seconds=10)
    finally:
        dist.destroy_process_group()


def test_single_rank_parallelize_skips_fsdp_without_active_dp_mesh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = RwkvModelAdapter(RwkvModelAdapter.Config(hf_assets_path=str(_assets(tmp_path))))
    sharded = False

    def record_shard(*args, **kwargs) -> None:
        del args, kwargs
        nonlocal sharded
        sharded = True

    monkeypatch.setattr("rwkv_trainer.parallelize.fully_shard", record_shard)
    parallel_dims = SimpleNamespace(
        tp_enabled=False,
        pp_enabled=False,
        cp_enabled=False,
        ep_enabled=False,
        dp_replicate_enabled=False,
        dp_shard_enabled=False,
    )
    compile_config = SimpleNamespace(enable=False, components=[])
    result = parallelize_rwkv(
        model,
        parallel_dims=parallel_dims,  # type: ignore[arg-type]
        training=SimpleNamespace(),  # type: ignore[arg-type]
        parallelism=SimpleNamespace(),  # type: ignore[arg-type]
        compile_config=compile_config,  # type: ignore[arg-type]
        ac_config=None,
        dump_folder=str(tmp_path),
    )
    assert result is model
    assert not sharded


@pytest.mark.parametrize("peft_enabled", [False, True])
def test_fsdp_marks_only_forward_aligned_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, peft_enabled: bool
) -> None:
    model = RwkvModelAdapter(
        RwkvModelAdapter.Config(
            hf_assets_path=str(_assets(tmp_path)),
            peft=PeftSettings(enabled=peft_enabled),
        )
    )
    sharded: list[torch.nn.Module] = []

    def record_shard(module, **kwargs) -> None:
        del kwargs
        sharded.append(module)

    monkeypatch.setattr("rwkv_trainer.parallelize.fully_shard", record_shard)
    mesh = object()
    monkeypatch.setattr(
        "rwkv_trainer.parallelize.resolve_fsdp_mesh", lambda parallel_dims: (mesh, None)
    )
    parallel_dims = SimpleNamespace(
        tp_enabled=False,
        pp_enabled=False,
        cp_enabled=False,
        ep_enabled=False,
        dp_replicate_enabled=False,
        dp_shard_enabled=True,
    )
    parallelism = SimpleNamespace(spmd_backend="full_dtensor", fsdp_reshard_after_forward="default")
    training = SimpleNamespace(
        mixed_precision_param="bfloat16",
        mixed_precision_reduce="float32",
        enable_cpu_offload=False,
    )
    compile_config = SimpleNamespace(enable=False, components=[])
    result = parallelize_rwkv(
        model,
        parallel_dims=parallel_dims,  # type: ignore[arg-type]
        training=training,  # type: ignore[arg-type]
        parallelism=parallelism,  # type: ignore[arg-type]
        compile_config=compile_config,  # type: ignore[arg-type]
        ac_config=None,
        dump_folder=str(tmp_path),
    )
    assert result is model
    assert model.rwkv_model.model.blocks[0] not in sharded
    assert all(block in sharded for block in model.rwkv_model.model.blocks[1:])
    assert sharded[-1] is model
    assert (model.rwkv_model in sharded) is not peft_enabled
    if peft_enabled:
        assert sharded[-2] is model.rwkv_model.model
    else:
        assert sharded[-2] is model.rwkv_model
        assert sharded[-3] is model.rwkv_model.model


def test_state_dict_adapter_only_changes_outer_prefix(tmp_path: Path) -> None:
    config = RwkvModelAdapter.Config(hf_assets_path=str(_assets(tmp_path)))
    adapter = RwkvStateDictAdapter(config, str(tmp_path))
    tensor = torch.ones(2)
    native = adapter.from_hf({"model.emb.weight": tensor})
    assert native == {"hf_model.model.emb.weight": tensor}
    assert adapter.to_hf(native) == {"model.emb.weight": tensor}

    peft_config = RwkvModelAdapter.Config(
        hf_assets_path=str(tmp_path), peft=PeftSettings(enabled=True)
    )
    peft_adapter = RwkvStateDictAdapter(peft_config, str(tmp_path))
    hf_state = {
        "model.blocks.0.att.key.weight": tensor,
        "model.blocks.0.att.gate.weight": tensor,
    }
    peft_native = peft_adapter.from_hf(hf_state)
    assert peft_native == {
        "hf_model.base_model.model.model.blocks.0.att.key.base_layer.weight": tensor,
        "hf_model.base_model.model.model.blocks.0.att.gate.weight": tensor,
    }
    assert peft_adapter.to_hf(peft_native) == hf_state


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


def test_optimizer_checkpoint_materializes_unused_parameter_state(tmp_path: Path) -> None:
    model = RwkvModelAdapter(RwkvModelAdapter.Config(hf_assets_path=str(_assets(tmp_path))))
    optimizer = RwkvOptimizersContainer.Config(implementation="for-loop").build(model_parts=[model])
    target_name = "hf_model.model.blocks.0.att.v0"
    target = dict(model.named_parameters())[target_name]
    for parameter in model.parameters():
        if parameter is not target:
            parameter.grad = torch.zeros_like(parameter)
    optimizer.step()
    saved_grads = {
        name: parameter.grad
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }

    state = optimizer.state_dict()

    assert f"state.{target_name}.step" in state
    assert state[f"state.{target_name}.step"].item() == 0
    assert all(
        dict(model.named_parameters())[name].grad is gradient
        for name, gradient in saved_grads.items()
    )
    assert target.grad is None
    restored = RwkvOptimizersContainer.Config(implementation="for-loop").build(model_parts=[model])
    restored.load_state_dict(state)
    assert len(restored.optimizers[0].state) == sum(
        parameter.requires_grad for parameter in model.parameters()
    )


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


def test_dataloader_checkpoint_identity_is_shared_across_dp_ranks() -> None:
    config = RwkvDataLoader.Config(dataset="synthetic", vocab_size=64)
    rank_zero = RwkvDataLoader(
        config,
        dp_world_size=2,
        dp_rank=0,
        tokenizer=object(),
        seq_len=4,
        local_batch_size=1,
    )
    next(iter(rank_zero))
    state = rank_zero.state_dict()
    rank_one = RwkvDataLoader(
        config,
        dp_world_size=2,
        dp_rank=1,
        tokenizer=object(),
        seq_len=4,
        local_batch_size=1,
    )

    rank_one.load_state_dict(state)

    assert rank_one.cursor == 1
    expected = torch.randint(64, (1, 5), generator=torch.Generator().manual_seed(45))
    actual = next(iter(rank_one))
    torch.testing.assert_close(actual[0]["input"], expected[:, :-1])
    different_world = RwkvDataLoader(
        config,
        dp_world_size=1,
        dp_rank=0,
        tokenizer=object(),
        seq_len=4,
        local_batch_size=1,
    )
    with pytest.raises(ValueError, match="identity mismatch"):
        different_world.load_state_dict(state)


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


@pytest.mark.parametrize("invalid", [-100, -1, 4])
def test_l2wrap_rejects_masked_or_invalid_labels(invalid: int) -> None:
    loss = RwkvL2WrapLoss(RwkvL2WrapLoss.Config())
    logits = torch.zeros(1, 1, 4, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="masked or out-of-vocabulary"):
        loss(logits, torch.tensor([[invalid]]), 1.0)


def test_l2wrap_rejects_non_integer_or_misaligned_labels() -> None:
    loss = RwkvL2WrapLoss(RwkvL2WrapLoss.Config())
    logits = torch.zeros(1, 2, 4, dtype=torch.bfloat16)
    with pytest.raises(TypeError, match=r"torch\.long"):
        loss(logits, torch.zeros(1, 2), 2.0)
    with pytest.raises(ValueError, match="shapes"):
        loss(logits, torch.zeros(1, 1, dtype=torch.long), 1.0)


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
    assert "204974d2a8cec8d31a57cd8c4737a2ee3a32549a" in manifest
    assert "flashrwkv2==0.1.0a6" in manifest
    lock = (Path(__file__).parents[1] / "uv.lock").read_text()
    assert "sha256:73d7ff2f055d03c0ae092a39e6387f8a372a320f31e6262a204a4eae9f2ec274" in lock
    assert "sha256:d2c0cf7c55fb3a6732c1e674fd9e3615719c685c2e0b50828e592c489e86eb29" in lock
    assert dependency_metadata() == {
        "torchtitan_oid": "96276d86577cf3e3bd29de72586e76af62010a55",
        "transformers_oid": "204974d2a8cec8d31a57cd8c4737a2ee3a32549a",
        "tokenizers_oid": "c5d8dde5ff49c70e4656199d5033a84e03c21b2b",
        "flashrwkv2_version": "0.1.0a6",
        "flashrwkv2_oid": "255df16b85edeac69ce512bb4a5ad1122a11863d",
        "peft_version": "0.19.1",
        "peft_oid": "ba6a19060d6ab54a87538a6e77e3e4d5a907375b",
        "rwkv_peft_reference_oid": "5704c39f8ab1d2ac63936ab392aadb6ba526e1a5",
    }
    metadata = json.loads(json.dumps(model_registry("pretrain").model.to_dict()))
    assert "hf_assets_path" in metadata


def test_export_maps_canonical_and_activation_checkpoint_keys(tmp_path: Path) -> None:
    tensors = {
        "hf_model.model.blocks.0.att.key.weight": torch.ones(1),
        "hf_model.head.weight": torch.ones(1),
    }
    source_keys = {
        "hf_model.model.blocks.0._checkpoint_wrapped_module.att.key.weight",
        "hf_model.head.weight",
    }
    mapped = _dcp_model_state(tensors, source_keys)
    assert set(mapped) == source_keys
    assert mapped["hf_model.head.weight"] is tensors["hf_model.head.weight"]
    with pytest.raises(KeyError, match="does not contain"):
        _dcp_model_state(tensors, {"hf_model.head.weight"})

    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "config.json").write_text("{}")
    assert artifact_hashes(artifact) == {
        "config.json": "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a"
    }


def test_export_rejects_mixed_floating_dtypes() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(2, 2).to(torch.bfloat16))
    model.register_buffer("fp32_buffer", torch.ones(1))
    with pytest.raises(TypeError, match=r"buffer fp32_buffer=torch.float32"):
        _require_uniform_floating_dtype(model, torch.bfloat16)
    model.fp32_buffer = model.fp32_buffer.to(torch.bfloat16)
    _require_uniform_floating_dtype(model, torch.bfloat16)


def test_export_reports_strict_same_path_logit_error() -> None:
    expected = torch.tensor([1.0, 2.0], dtype=torch.float32)
    actual = torch.tensor([1.01, 1.98], dtype=torch.float32)
    stats = _assert_logits_close(actual, expected, atol=2e-2, rtol=2e-2)
    assert stats == {
        "elements": 2,
        "max_absolute_difference": pytest.approx(0.02),
        "max_relative_difference": pytest.approx(0.01),
    }
    with pytest.raises(AssertionError, match="not close"):
        _assert_logits_close(torch.tensor([1.1]), torch.tensor([1.0]), atol=2e-2, rtol=2e-2)


def test_export_deterministic_validation_is_scoped(monkeypatch: pytest.MonkeyPatch) -> None:
    original_enabled = torch.are_deterministic_algorithms_enabled()
    original_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    with _deterministic_cuda_validation():
        assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
        assert torch.are_deterministic_algorithms_enabled()
        assert not torch.is_deterministic_algorithms_warn_only_enabled()
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":16:8"
    assert torch.are_deterministic_algorithms_enabled() == original_enabled
    assert torch.is_deterministic_algorithms_warn_only_enabled() == original_warn_only
