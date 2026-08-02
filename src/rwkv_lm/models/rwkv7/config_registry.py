"""TorchTitan training configurations for RWKV-7."""

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.lora import LoRAConverter
from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import default_adamw
from torchtitan.config import ParallelismConfig, TrainingConfig
from torchtitan.distributed.activation_checkpoint import FullAC
from torchtitan.trainer import Trainer

from . import model_registry
from .data import RwkvDataLoader, RwkvPretokenizedTokenizer


def _training_config(
    *,
    flavor: str,
    vocab_size: int,
    seq_len: int,
    local_batch_size: int,
    steps: int,
    dataloader: RwkvDataLoader.Config,
    checkpoint: CheckpointManager.Config,
) -> Trainer.Config:
    model_spec = model_registry(flavor)
    return Trainer.Config(
        model_spec=model_spec,
        hf_assets_path="",
        tokenizer=RwkvPretokenizedTokenizer.Config(vocab_size=vocab_size),
        loss=CrossEntropyLoss.Config(global_vocab_size=model_spec.model.vocab_size),
        optimizer=default_adamw(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            local_batch_size=local_batch_size,
            seq_len=seq_len,
            steps=steps,
        ),
        dataloader=dataloader,
        metrics=MetricsProcessor.Config(log_freq=1),
        parallelism=ParallelismConfig(enable_sequence_parallel=False),
        checkpoint=checkpoint,
        activation_checkpoint=FullAC.Config(),
    )


def rwkv7_debugmodel() -> Trainer.Config:
    return _training_config(
        flavor="debugmodel",
        vocab_size=1_024,
        seq_len=128,
        local_batch_size=2,
        steps=10,
        dataloader=RwkvDataLoader.Config(
            dataset="synthetic",
            vocab_size=1_024,
            infinite=False,
            num_batches=10,
        ),
        checkpoint=CheckpointManager.Config(
            enable=True,
            interval=5,
            initial_load_model_only=False,
            last_save_model_only=False,
        ),
    )


def rwkv7_1_5b() -> Trainer.Config:
    return _training_config(
        flavor="g1h-1.5b",
        vocab_size=65_536,
        seq_len=10_240,
        local_batch_size=1,
        steps=1_000,
        dataloader=RwkvDataLoader.Config(
            dataset="binidx",
            dataset_path=None,
            vocab_size=65_536,
            magic_prime=None,
        ),
        checkpoint=CheckpointManager.Config(
            enable=True,
            interval=100,
            initial_load_path=None,
            initial_load_in_hf=True,
            initial_load_model_only=True,
            last_save_model_only=False,
        ),
    )


def rwkv7_debugmodel_lora() -> Trainer.Config:
    config = rwkv7_debugmodel()
    config.model_spec = model_registry(
        "debugmodel",
        converters=[
            LoRAConverter.Config(
                rank=8,
                alpha=16.0,
                target_modules=["receptance", "key", "value", "output"],
            )
        ],
    )
    return config


def rwkv7_debugmodel_infctx() -> Trainer.Config:
    config = rwkv7_debugmodel()
    config.model_spec.model.recurrent_chunk_size = 16
    config.model_spec.model.detach_state_between_chunks = True
    return config


def rwkv7_1_5b_infctx() -> Trainer.Config:
    config = rwkv7_1_5b()
    config.model_spec.model.recurrent_chunk_size = 1_024
    config.model_spec.model.detach_state_between_chunks = True
    return config
