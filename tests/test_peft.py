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
    build_lora_delta,
    freeze_base_for_lora,
    load_lora_adapter,
    load_lora_base_state_dict,
    lora_base_state_dict,
    lora_parameter_names,
    save_lora_adapter,
)


class _TinyLoraModel(nn.Module):
    """Small real linear owner using the same adapter and artifact contract."""

    def __init__(self, config: LoraConfig) -> None:
        super().__init__()
        self.lora_config = config
        self.projection = nn.Linear(4, 3, bias=False)
        self.projection_lora = build_lora_delta(
            config,
            "time_mix.key",
            in_features=4,
            out_features=3,
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

    torch.manual_seed(7)
    reloaded = _TinyLoraModel(_config())
    load_lora_base_state_dict(reloaded, base_state)
    load_lora_adapter(reloaded, artifact)
    reloaded.eval()
    torch.testing.assert_close(reloaded(inputs), expected_logits, rtol=0, atol=0)


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


def test_disabled_or_incomplete_lora_does_not_create_trainable_adapters() -> None:
    model = _TinyLoraModel(LoraConfig())
    assert lora_parameter_names(model) == ()
    with pytest.raises(PeftContractError, match="enabled LoRA config"):
        save_lora_adapter(model, Path("unused-adapter.pt"))
    with pytest.raises(PeftContractError, match="explicit target"):
        LoraConfig(rank=2, alpha=4.0)
