import json

from torchtitan.components.lora import LoRAConverter
from torchtitan.models.common.linear import Linear
from torchtitan.protocols.model_spec import ModelSpec
from torchtitan.protocols.module import Module
from torchtitan.trainer import Trainer

from rwkv_lm.models.rwkv7 import model_registry
from rwkv_lm.models.rwkv7.config_registry import rwkv7_debugmodel


def test_model_registry_returns_torchtitan_model_spec() -> None:
    spec = model_registry("debugmodel")

    assert isinstance(spec, ModelSpec)
    assert spec.name == "rwkv7"
    assert spec.flavor == "debugmodel"
    assert spec.pipelining_fn is None


def test_config_registry_returns_trainer_config() -> None:
    config = rwkv7_debugmodel()

    assert isinstance(config, Trainer.Config)
    assert config.model_spec is not None
    assert config.model_spec.name == "rwkv7"
    assert config.parallelism.enable_sequence_parallel is False


def test_model_config_tree_is_acyclic_with_unique_linear_fqns() -> None:
    model_config = model_registry("debugmodel").model
    configs = list(model_config.traverse(Module.Config, recurse=True))
    linear_entries = [
        (fqn, config)
        for fqn, config, _, _ in configs
        if isinstance(config, Linear.Config)
    ]

    assert len(configs) < 100
    assert len(linear_entries) > 0
    assert len({fqn for fqn, _ in linear_entries}) == len(linear_entries)
    assert len({id(config) for _, config in linear_entries}) == len(linear_entries)
    json.dumps(model_config.to_dict())


def test_update_from_trainer_config_populates_all_linear_sharding() -> None:
    trainer_config = rwkv7_debugmodel()
    model_config = trainer_config.model_spec.model

    model_config.update_from_config(config=trainer_config)

    linear_configs = [
        config
        for _, config, _, _ in model_config.traverse(Module.Config, recurse=True)
        if isinstance(config, Linear.Config)
    ]
    assert model_config.sharding_config is not None
    assert model_config.tok_embeddings.sharding_config is not None
    assert all(config.sharding_config is not None for config in linear_configs)
    assert "v0" not in model_config.layers[0].att.sharding_config.state_shardings
    assert "v0" in model_config.layers[1].att.sharding_config.state_shardings


def test_torchtitan_lora_converter_builds_adapter_only_training_model() -> None:
    spec = model_registry(
        "debugmodel",
        converters=[
            LoRAConverter.Config(
                rank=4,
                alpha=8.0,
                target_modules=["receptance", "output"],
            )
        ],
    )
    model = spec.model.build()
    model.init_states()
    trainable = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }

    assert trainable
    assert all(".lora_a." in name or ".lora_b." in name for name in trainable)
    assert any(".receptance.lora_a." in name for name in trainable)
    assert any(".output.lora_b." in name for name in trainable)
