from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest
import torch
from torch import nn

from rwkv_lm.checkpoint import BackendIdentity
from rwkv_lm.checkpoint_runner import EpochCheckpointRunnerAdapter
from rwkv_lm.peft import (
    LoraConfig,
    PeftContractError,
    bind_lora_base_provenance,
    build_lora_delta,
    freeze_base_for_lora,
    load_lora_adapter,
    load_lora_base_state_dict,
    lora_base_state_dict,
    lora_merged_state_dict,
    lora_parameter_names,
    save_lora_adapter,
    save_lora_merged_model,
)

_TINY_SOURCE_REVISION = "0123456789abcdef0123456789abcdef01234567"
_TINY_MODEL_CONFIG = {
    "hidden_size": 4,
    "model_type": "tiny-rwkv-lora",
    "output_size": 3,
}


class _TinyLoraModel(nn.Module):
    """Small real linear owner using the same adapter and artifact contract."""

    def __init__(
        self,
        config: LoraConfig,
        *,
        source_revision: str = _TINY_SOURCE_REVISION,
    ) -> None:
        super().__init__()
        self.lora_config = config
        self.projection = nn.Linear(4, 3, bias=False)
        self.projection_lora = build_lora_delta(
            config,
            "time_mix.key",
            in_features=4,
            out_features=3,
        )
        bind_lora_base_provenance(
            self,
            model_config=_TINY_MODEL_CONFIG,
            source_revision=source_revision,
        )
        freeze_base_for_lora(self, config)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.projection(inputs)
        if self.projection_lora.enabled:
            output = output + self.projection_lora(inputs)
        return output


def _config() -> LoraConfig:
    return LoraConfig(
        rank=2,
        alpha=4.0,
        dropout=0.0,
        target_modules=("time_mix.key",),
    )


def _optimizer(model: nn.Module) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=0.03,
    )


def _adapter() -> EpochCheckpointRunnerAdapter:
    config = _config()
    return EpochCheckpointRunnerAdapter(
        backend=BackendIdentity(
            name="pytorch",
            version=torch.__version__,
            strategy="single_process",
            world_size=1,
            state_dict_type="full",
        ),
        training_config={
            "epoch_steps": 2,
            "lora_alpha": config.alpha,
            "lora_dropout": config.dropout,
            "lora_rank": config.rank,
            "lora_target_modules": config.target_modules,
            "model": "tiny-lora",
        },
        samples_per_epoch=4,
    )


def _step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    optimizer.zero_grad(set_to_none=True)
    loss = (model(inputs) - targets).square().mean()
    loss.backward()
    optimizer.step()
    return loss.detach()


def _clone_state(state: Mapping[str, object]) -> dict[str, object]:
    cloned = {}
    for name, value in state.items():
        if isinstance(value, torch.Tensor):
            cloned[name] = value.detach().clone()
        elif isinstance(value, Mapping):
            cloned[name] = _clone_state(value)
        elif isinstance(value, list):
            cloned[name] = [
                _clone_state(item) if isinstance(item, Mapping) else item
                for item in value
            ]
        else:
            cloned[name] = value
    return cloned


def _assert_nested_equal(actual: object, expected: object) -> None:
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, Mapping):
        assert isinstance(actual, Mapping)
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_nested_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert isinstance(actual, type(expected))
        assert len(actual) == len(expected)
        for actual_value, expected_value in zip(actual, expected, strict=True):
            _assert_nested_equal(actual_value, expected_value)
    else:
        assert actual == expected


def test_lora_freezes_base_and_adapter_artifact_fresh_reload_matches_logits(
    tmp_path: Path,
) -> None:
    torch.manual_seed(20260801)
    model = _TinyLoraModel(_config())
    base_state = {
        name: tensor.detach().clone()
        for name, tensor in lora_base_state_dict(model).items()
    }
    trainable_names = tuple(
        sorted(
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        )
    )
    assert trainable_names == lora_parameter_names(model)
    assert trainable_names == (
        "projection_lora.lora_A",
        "projection_lora.lora_B",
    )
    assert not model.projection.weight.requires_grad

    inputs = torch.tensor([[0.1, -0.2, 0.3, -0.4], [0.5, 0.6, -0.7, -0.8]])
    targets = torch.tensor([[0.25, -0.5, 0.75], [-0.1, 0.2, -0.3]])
    optimizer = _optimizer(model)
    _step(model, optimizer, inputs, targets)
    _step(model, optimizer, inputs, targets)
    assert model.projection.weight.grad is None
    assert torch.count_nonzero(model.projection_lora.lora_A.grad) > 0
    assert torch.count_nonzero(model.projection_lora.lora_B.grad) > 0

    model.eval()
    expected_logits = model(inputs).detach()
    artifact = save_lora_adapter(model, tmp_path / "adapter.pt")
    payload = torch.load(artifact, map_location="cpu", weights_only=True)
    assert payload["schema_version"] == 2
    assert payload["base_identity"]["model_config"] == _TINY_MODEL_CONFIG
    assert payload["base_identity"]["source_revision"] == _TINY_SOURCE_REVISION
    assert len(payload["base_identity"]["state_sha256"]) == 64

    torch.manual_seed(7)
    reloaded = _TinyLoraModel(_config())
    load_lora_base_state_dict(reloaded, base_state)
    load_lora_adapter(reloaded, artifact)
    reloaded.eval()
    torch.testing.assert_close(reloaded(inputs), expected_logits, rtol=0, atol=0)


def test_adapter_rejects_wrong_base_state_or_source_revision(tmp_path: Path) -> None:
    torch.manual_seed(20260801)
    model = _TinyLoraModel(_config())
    base_state = {
        name: tensor.detach().clone()
        for name, tensor in lora_base_state_dict(model).items()
    }
    artifact = save_lora_adapter(model, tmp_path / "adapter.pt")

    torch.manual_seed(7)
    wrong_base = _TinyLoraModel(_config())
    with pytest.raises(PeftContractError, match="base identity"):
        load_lora_adapter(wrong_base, artifact)

    wrong_revision = _TinyLoraModel(_config(), source_revision="1" * 40)
    load_lora_base_state_dict(wrong_revision, base_state)
    with pytest.raises(PeftContractError, match="base identity"):
        load_lora_adapter(wrong_revision, artifact)


def test_adapter_validation_failure_does_not_modify_any_parameter(
    tmp_path: Path,
) -> None:
    torch.manual_seed(20260801)
    model = _TinyLoraModel(_config())
    base_state = {
        name: tensor.detach().clone()
        for name, tensor in lora_base_state_dict(model).items()
    }
    artifact = save_lora_adapter(model, tmp_path / "adapter.pt")
    payload = torch.load(artifact, map_location="cpu", weights_only=True)
    payload["state_dict"]["projection_lora.lora_B"] = payload["state_dict"][
        "projection_lora.lora_B"
    ].double()
    invalid_artifact = tmp_path / "invalid-adapter.pt"
    torch.save(payload, invalid_artifact)

    torch.manual_seed(7)
    destination = _TinyLoraModel(_config())
    load_lora_base_state_dict(destination, base_state)
    before = _clone_state(destination.state_dict())
    assert not torch.equal(
        before["projection_lora.lora_A"],
        payload["state_dict"]["projection_lora.lora_A"],
    )

    with pytest.raises(PeftContractError, match="dtype"):
        load_lora_adapter(destination, invalid_artifact)

    _assert_nested_equal(destination.state_dict(), before)


def test_standard_checkpoint_restores_adapter_and_optimizer_state(
    tmp_path: Path,
) -> None:
    torch.manual_seed(20260801)
    model = _TinyLoraModel(_config())
    optimizer = _optimizer(model)
    inputs = torch.randn(2, 4)
    targets = torch.randn(2, 3)
    _step(model, optimizer, inputs, targets)
    _step(model, optimizer, inputs, targets)

    checkpoint = tmp_path / "epoch-00000001"
    _adapter().save(
        checkpoint,
        model=model,
        optimizer=optimizer,
        global_step=2,
        next_epoch=1,
    )
    expected_model = _clone_state(model.state_dict())
    expected_optimizer = _clone_state(optimizer.state_dict())

    resumed = _TinyLoraModel(_config())
    resumed_optimizer = _optimizer(resumed)
    progress = _adapter().restore(
        checkpoint,
        model=resumed,
        optimizer=resumed_optimizer,
    )

    assert progress.global_step == 2
    assert progress.epoch == 1
    _assert_nested_equal(resumed.state_dict(), expected_model)
    _assert_nested_equal(resumed_optimizer.state_dict(), expected_optimizer)

    _step(model, optimizer, inputs, targets)
    _step(resumed, resumed_optimizer, inputs, targets)
    _assert_nested_equal(resumed.state_dict(), model.state_dict())
    _assert_nested_equal(resumed_optimizer.state_dict(), optimizer.state_dict())


def test_merged_model_fresh_reload_matches_adapter_inference(tmp_path: Path) -> None:
    torch.manual_seed(20260801)
    model = _TinyLoraModel(_config())
    optimizer = _optimizer(model)
    inputs = torch.randn(5, 4)
    targets = torch.randn(5, 3)
    _step(model, optimizer, inputs, targets)
    _step(model, optimizer, inputs, targets)
    model.eval()
    expected_output = model(inputs).detach()
    base_before_merge = {
        name: tensor.detach().clone()
        for name, tensor in lora_base_state_dict(model).items()
    }

    merged_state = lora_merged_state_dict(model)
    assert set(merged_state) == set(base_before_merge)
    assert not any(name.endswith((".lora_A", ".lora_B")) for name in merged_state)
    _assert_nested_equal(lora_base_state_dict(model), base_before_merge)
    torch.testing.assert_close(model(inputs), expected_output, rtol=0, atol=0)

    artifact = save_lora_merged_model(model, tmp_path / "merged-model.pth")
    fresh_state = torch.load(artifact, map_location="cpu", weights_only=True)
    fresh_model = _TinyLoraModel(LoraConfig())
    fresh_model.load_state_dict(fresh_state, strict=True)
    fresh_model.eval()

    torch.testing.assert_close(
        fresh_model(inputs),
        expected_output,
        rtol=1e-6,
        atol=1e-7,
    )


def test_disabled_or_incomplete_lora_does_not_create_trainable_adapters() -> None:
    model = _TinyLoraModel(LoraConfig())
    assert lora_parameter_names(model) == ()
    with pytest.raises(PeftContractError, match="enabled LoRA config"):
        save_lora_adapter(model, Path("unused-adapter.pt"))
    with pytest.raises(PeftContractError, match="explicit target"):
        LoraConfig(rank=2, alpha=4.0)
