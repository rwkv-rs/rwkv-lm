import os, math, time, datetime, subprocess
from importlib import metadata
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.strategies import (
    DDPStrategy,
    DeepSpeedStrategy,
    SingleDeviceStrategy,
)
from pytorch_lightning.utilities import rank_zero_info, rank_zero_only

from .checkpoint import BackendIdentity, CheckpointContractError
from .checkpoint_runner import EpochCheckpointRunnerAdapter


_CHECKPOINT_CONFIG_FIELDS = (
    "adam_eps",
    "beta1",
    "beta2",
    "chunk_ctx",
    "ctx_len",
    "data_file",
    "data_type",
    "dim_att",
    "dim_ffn",
    "epoch_steps",
    "grad_clip",
    "grad_cp",
    "head_chunk",
    "head_size",
    "kernel",
    "lr_final",
    "lr_init",
    "magic_prime",
    "micro_bsz",
    "my_exit_tokens",
    "my_testing",
    "n_embd",
    "n_layer",
    "precision",
    "random_seed",
    "real_bsz",
    "train_stage",
    "train_type",
    "vocab_size",
    "warmup_steps",
    "weight_decay",
)


def _checkpoint_training_config(args):
    missing = [name for name in _CHECKPOINT_CONFIG_FIELDS if not hasattr(args, name)]
    if missing:
        raise CheckpointContractError(
            f"training arguments are missing checkpoint fields: {missing}"
        )
    return {name: getattr(args, name) for name in _CHECKPOINT_CONFIG_FIELDS}


def _checkpoint_backend_identity(trainer):
    strategy = trainer.strategy
    world_size = int(trainer.world_size)
    if isinstance(strategy, DeepSpeedStrategy):
        name = "deepspeed"
        version = (
            f"{metadata.version('deepspeed')};"
            f"pytorch-lightning={pl.__version__};torch={torch.__version__}"
        )
        zero_optimization = strategy.config.get("zero_optimization", {})
        stage = zero_optimization.get("stage")
        if stage == 2:
            normalized_strategy = "deepspeed_stage_2"
        elif stage == 3:
            normalized_strategy = "deepspeed_stage_3"
        else:
            raise CheckpointContractError(
                "standard checkpoint cannot identify the active DeepSpeed ZeRO stage"
            )
    elif isinstance(strategy, DDPStrategy):
        name = "pytorch-lightning"
        version = f"{pl.__version__};torch={torch.__version__}"
        normalized_strategy = "ddp"
    elif isinstance(strategy, SingleDeviceStrategy):
        name = "pytorch-lightning"
        version = f"{pl.__version__};torch={torch.__version__}"
        normalized_strategy = "single_device"
    else:
        raise CheckpointContractError(
            "standard checkpoint does not recognize the active Lightning strategy: "
            f"{type(strategy).__module__}.{type(strategy).__qualname__}"
        )
    return BackendIdentity(
        name=name,
        version=version,
        strategy=normalized_strategy,
        world_size=world_size,
        state_dict_type="full",
    )


def scheduled_learning_rate(args, global_step):
    """Return the stateless RWKV learning rate and whether its token limit hit."""

    lr = args.lr_init
    reached_token_limit = False
    warmup_steps = args.warmup_steps
    if args.my_exit_tokens != 0:
        real_tokens = global_step * args.ctx_len * args.real_bsz
        warmup_tokens = warmup_steps * args.ctx_len * args.real_bsz
        decay_tokens = abs(args.my_exit_tokens) - warmup_tokens
        if decay_tokens <= 0:
            raise CheckpointContractError(
                "my_exit_tokens must exceed the configured warmup token count"
            )
        progress = (real_tokens - warmup_tokens) / decay_tokens
        progress = max(0, min(1, progress))
        lr_final_factor = args.lr_final / args.lr_init
        lr_mult = (0.5 + lr_final_factor / 2) + (
            0.5 - lr_final_factor / 2
        ) * math.cos(math.pi * progress)
        if args.my_exit_tokens > 0:
            lr = args.lr_init * lr_mult
        else:
            lr = (lr + args.lr_init * lr_mult) / 2
        reached_token_limit = progress >= 1
    if warmup_steps > 0 and global_step < warmup_steps:
        lr *= 0.01 + 0.99 * global_step / warmup_steps
    return lr, reached_token_limit


class train_callback(pl.Callback):
    def __init__(self, args, *, resume_checkpoint=None):
        super().__init__()
        self.args = args
        self.resume_checkpoint = (
            Path(resume_checkpoint) if resume_checkpoint is not None else None
        )
        self.checkpoint_adapter = None
        self.save_at_epoch_end = False

    def on_fit_start(self, trainer, pl_module):
        args = self.args
        if int(getattr(args, "accumulate_grad_batches", 1)) != 1:
            raise CheckpointContractError(
                "standard checkpoint v1 requires accumulate_grad_batches=1"
            )
        self.checkpoint_adapter = EpochCheckpointRunnerAdapter(
            backend=_checkpoint_backend_identity(trainer),
            training_config=_checkpoint_training_config(args),
            samples_per_epoch=int(args.epoch_steps) * int(args.real_bsz),
        )
        if self.resume_checkpoint is None:
            return
        if trainer.global_step != 0:
            raise CheckpointContractError(
                "standard resume must run before the new Trainer advances"
            )
        progress = self.checkpoint_adapter.restore(
            self.resume_checkpoint,
            model=pl_module,
            optimizer=trainer.optimizers[0],
        )
        expected_global_step = int(progress.epoch) * int(args.epoch_steps)
        if (
            progress.epoch != int(args.epoch_begin)
            or progress.global_step != expected_global_step
        ):
            raise CheckpointContractError(
                "standard resume progress does not match the epoch schedule"
            )

    def _request_epoch_boundary_stop(self, trainer, batch_idx):
        if int(batch_idx) + 1 != int(self.args.epoch_steps):
            raise CheckpointContractError(
                "standard checkpoint v1 cannot stop and save inside an epoch"
            )
        self.save_at_epoch_end = True
        trainer.should_stop = True

    def _save_epoch_checkpoint(self, trainer, pl_module):
        args = self.args
        if self.checkpoint_adapter is None:
            raise CheckpointContractError("checkpoint adapter was not initialized")
        next_epoch = int(args.epoch_begin) + int(trainer.current_epoch) + 1
        global_step = int(trainer.global_step) + int(args.epoch_begin) * int(
            args.epoch_steps
        )
        if global_step != next_epoch * int(args.epoch_steps):
            raise CheckpointContractError(
                "standard checkpoint v1 can only publish a complete epoch"
            )
        checkpoint_dir = (
            Path(args.proj_dir)
            / "checkpoints"
            / f"epoch-{next_epoch:08d}"
        )
        self.checkpoint_adapter.save(
            checkpoint_dir,
            model=pl_module,
            optimizer=trainer.optimizers[0],
            global_step=global_step,
            next_epoch=next_epoch,
        )

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        args = self.args

        real_step = trainer.global_step + args.epoch_begin * args.epoch_steps

        lr, reached_token_limit = scheduled_learning_rate(args, real_step)
        if reached_token_limit:
            self._request_epoch_boundary_stop(trainer, batch_idx)

        wd_now = args.weight_decay

        for param_group in trainer.optimizers[0].param_groups:
            if param_group["weight_decay"] > 0:
                param_group["weight_decay"] = wd_now
            param_group["lr"] = lr * param_group["my_lr_scale"]

        trainer.my_lr = lr
        trainer.my_wd = wd_now

        if trainer.global_step == 0:
            if trainer.is_global_zero:  # logging
                trainer.my_loss_sum = 0
                trainer.my_loss_count = 0
                trainer.my_log = open(args.proj_dir + "/train_log.txt", "a")
                trainer.my_log.write(f"NEW RUN {args.my_timestamp}\n{vars(self.args)}\n")
                try:
                    print(f"\n{trainer.strategy.config}\n")
                    trainer.my_log.write(f"{trainer.strategy.config}\n")
                except:
                    pass
                trainer.my_log.flush()
                if len(args.wandb) > 0:
                    print("Login to wandb...")
                    import wandb
                    wandb.init(
                        project=args.wandb,
                        name=args.run_name + " " + args.my_timestamp,
                        config=args,
                        save_code=False,
                    )
                    trainer.my_wandb = wandb

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        args = self.args
        token_per_step = args.ctx_len * args.real_bsz
        real_step = trainer.global_step + args.epoch_begin * args.epoch_steps

        if trainer.is_global_zero:  # logging
            t_now = time.time_ns()
            kt_s = 0
            try:
                t_cost = (t_now - trainer.my_time_ns) / 1e9
                kt_s = token_per_step / t_cost / 1000
                self.log("REAL it/s", 1.0 / t_cost, prog_bar=True, on_step=True)
                self.log("Kt/s", kt_s, prog_bar=True, on_step=True)
            except:
                pass
            trainer.my_time_ns = t_now
            trainer.my_loss = trainer.my_loss_all.float().mean().item()
            trainer.my_loss_sum += trainer.my_loss
            trainer.my_loss_count += 1
            trainer.my_epoch_loss = trainer.my_loss_sum / trainer.my_loss_count
            self.log("lr", trainer.my_lr, prog_bar=True, on_step=True)
            self.log("loss", trainer.my_epoch_loss, prog_bar=True, on_step=True)

            if len(args.wandb) > 0:
                lll = {"loss": trainer.my_loss, "lr": trainer.my_lr, "wd": trainer.my_wd, "Gtokens": real_step * token_per_step / 1e9}
                if kt_s > 0:
                    lll["kt/s"] = kt_s
                trainer.my_wandb.log(lll, step=int(real_step))

        if args.magic_prime > 0:
            if int(real_step) == int(args.magic_prime // args.real_bsz) - 1:
                self._request_epoch_boundary_stop(trainer, batch_idx)
                

    def on_train_epoch_start(self, trainer, pl_module):
        args = self.args
        dataset = trainer.train_dataloader.dataset.datasets
        assert "MyDataset" in str(dataset)
        dataset.global_rank = trainer.global_rank
        dataset.real_epoch = int(args.epoch_begin + trainer.current_epoch)
        dataset.world_size = trainer.world_size
        # print(f'########## world_size {dataset.world_size} global_rank {dataset.global_rank} real_epoch {dataset.real_epoch} ##########')

    def on_train_epoch_end(self, trainer, pl_module):
        args = self.args
        save_due = (
            args.epoch_save > 0
            and trainer.current_epoch % args.epoch_save == 0
        ) or (trainer.current_epoch == args.epoch_count - 1)
        if save_due or self.save_at_epoch_end:
            self._save_epoch_checkpoint(trainer, pl_module)
            self.save_at_epoch_end = False

        if trainer.is_global_zero:  # logging
            trainer.my_log.write(f"{args.epoch_begin + trainer.current_epoch} {trainer.my_epoch_loss:.6f} {math.exp(trainer.my_epoch_loss):.4f} {trainer.my_lr:.8f} {datetime.datetime.now()} {trainer.current_epoch}\n")
            trainer.my_log.flush()

            trainer.my_loss_sum = 0
            trainer.my_loss_count = 0

@rank_zero_only
def generate_init_weight(model, init_weight_name):
    mm = model.generate_init_weight()

    if model.args.train_stage == 1:
        if len(model.args.load_model) > 0:
            print(f"Combine weights from {model.args.load_model}...")
            load_dict = torch.load(model.args.load_model, map_location="cpu")
            for k in load_dict:
                try:
                    assert k in mm
                except:
                    print('missing', k)
                    exit(0)
                src = load_dict[k]
                try:
                    mm[k] = src.reshape(mm[k].shape)
                except:
                    tmp = mm[k].squeeze().clone()
                    print(k, src.shape, '-->', mm[k].shape)
                    ss = src.shape[0]
                    dd = tmp.shape[0]
                    for i in range(dd):
                        pos = i / dd * ss
                        if pos >= ss - 1:
                            tmp[i] = src[ss-1]
                        else:
                            p0 = int(math.floor(pos))
                            ii = pos - p0
                            tmp[i] = src[p0] * (1-ii) + src[p0+1] * (ii)
                    mm[k] = tmp.reshape(mm[k].shape)
                    sss = src.squeeze().float().cpu().numpy()
                    print(sss[:10], '...', sss[-10:])
                    mmm = mm[k].squeeze().float().cpu().numpy()
                    print(mmm[:10], '...', mmm[-10:])

    # This is a legacy model-only initialization input, never a resume checkpoint.
    print(f"Save to {init_weight_name}...")
    torch.save(mm, init_weight_name)

    if model.args.train_stage == 1:
        print("Done. Now go for stage 2.")
        exit(0)
