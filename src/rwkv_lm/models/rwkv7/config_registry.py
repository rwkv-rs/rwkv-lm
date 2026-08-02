"""TorchTitan training configurations for RWKV-7."""

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import default_adamw
from torchtitan.config import ParallelismConfig, TrainingConfig
from torchtitan.distributed.activation_checkpoint import FullAC
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.trainer import Trainer

from . import model_registry


def rwkv7_debugmodel() -> Trainer.Config:
    model_spec = model_registry("debugmodel")
    return Trainer.Config(
        model_spec=model_spec,
        loss=CrossEntropyLoss.Config(global_vocab_size=model_spec.model.vocab_size),
        optimizer=default_adamw(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            local_batch_size=2,
            seq_len=128,
            steps=10,
        ),
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4_test"),
        metrics=MetricsProcessor.Config(log_freq=1),
        parallelism=ParallelismConfig(enable_sequence_parallel=False),
        checkpoint=CheckpointManager.Config(
            interval=10,
            last_save_model_only=False,
        ),
        activation_checkpoint=FullAC.Config(),
    )


def rwkv7_1_5b() -> Trainer.Config:
    config = rwkv7_debugmodel()
    config.model_spec = model_registry("g1h-1.5b")
    config.training.seq_len = 10_240
    return config
