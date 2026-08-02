from torchtitan.protocols.model_spec import ModelSpec
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
