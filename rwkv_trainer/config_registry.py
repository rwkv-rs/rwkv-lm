"""Standard TorchTitan config registry. Model shape is always read from HF assets."""

from __future__ import annotations

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.validate import Validator
from torchtitan.config import ParallelismConfig, TrainingConfig
from torchtitan.distributed.activation_checkpoint import SelectiveAC

from .binidx import RwkvDataLoader
from .loss import RwkvL2WrapLoss
from .model_spec import model_registry
from .optimizer import rwkv_optimizer
from .tokenizer import RwkvAutoTokenizer
from .trainer import RwkvTrainer


def _config(flavor: str, *, debug: bool = False) -> RwkvTrainer.Config:
    infctx = flavor.endswith("infctx")
    return RwkvTrainer.Config(
        model_spec=model_registry(flavor),
        tokenizer=RwkvAutoTokenizer.Config(),
        dataloader=RwkvDataLoader.Config(
            dataset="synthetic" if debug else "binidx",
            vocab_size=1024,
            infinite=True,
        ),
        loss=RwkvL2WrapLoss.Config(),
        optimizer=rwkv_optimizer(lr=6e-4, weight_decay=0.01),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=10 if debug else 100,
            decay_type="linear",
            min_lr_factor=0.1,
        ),
        training=TrainingConfig(
            local_batch_size=1,
            global_batch_size=1,
            seq_len=16 if debug else 4096,
            steps=2 if debug else 10000,
            dtype="bfloat16",
            mixed_precision_param="bfloat16",
            mixed_precision_reduce="float32",
            max_norm=1.0,
        ),
        parallelism=ParallelismConfig(
            data_parallel_replicate_degree=1,
            data_parallel_shard_degree=-1,
            tensor_parallel_degree=1,
            pipeline_parallel_degree=1,
            context_parallel_degree=1,
            expert_parallel_degree=1,
        ),
        checkpoint=CheckpointManager.Config(
            enable=True,
            interval=1 if debug else 500,
            initial_load_in_hf=True,
            initial_load_model_only=True,
            last_save_model_only=False,
        ),
        activation_checkpoint=None if infctx else SelectiveAC.Config(),
        metrics=MetricsProcessor.Config(log_freq=1 if debug else 10),
        validator=Validator.Config(enable=False),
    )


def rwkv7_debug() -> RwkvTrainer.Config:
    return _config("debug", debug=True)


def rwkv7_pretrain() -> RwkvTrainer.Config:
    return _config("pretrain")


def rwkv7_pretrain_infctx() -> RwkvTrainer.Config:
    return _config("pretrain_infctx")


def rwkv7_lora() -> RwkvTrainer.Config:
    return _config("lora")


def rwkv7_lora_infctx() -> RwkvTrainer.Config:
    return _config("lora_infctx")


__all__ = [
    "rwkv7_debug",
    "rwkv7_lora",
    "rwkv7_lora_infctx",
    "rwkv7_pretrain",
    "rwkv7_pretrain_infctx",
]
