########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import logging


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    from argparse import ArgumentParser
    from pathlib import Path
    from pytorch_lightning import Trainer
    from pytorch_lightning.utilities import rank_zero_info, rank_zero_only
    import pytorch_lightning as pl

    rank_zero_info("########## work in progress ##########")

    parser = ArgumentParser()

    parser.add_argument(
        "--load_model",
        default="",
        type=str,
    )  # standard checkpoint directory or legacy model-input .pth
    parser.add_argument("--wandb", default="", type=str)  # wandb project name. if "" then don't use wandb
    parser.add_argument("--proj_dir", default="out", type=str)
    parser.add_argument("--random_seed", default="-1", type=int)

    parser.add_argument("--data_file", default="", type=str)
    parser.add_argument("--data_type", default="utf-8", type=str)
    parser.add_argument("--vocab_size", default=0, type=int)  # vocab_size = 0 means auto (for char-level LM and .txt data)

    parser.add_argument("--ctx_len", default=1024, type=int)
    parser.add_argument(
        "--train_type",
        choices=("standard", "infctx"),
        default="standard",
    )
    parser.add_argument("--chunk_ctx", default=0, type=int)
    parser.add_argument("--lora_rank", default=0, type=int)
    parser.add_argument("--lora_alpha", default=0.0, type=float)
    parser.add_argument("--lora_dropout", default=0.0, type=float)
    parser.add_argument(
        "--lora_target_modules",
        default="",
        type=str,
        help="comma-separated explicit TimeMix/ChannelMix projection targets",
    )
    parser.add_argument(
        "--lora_adapter",
        default="",
        type=str,
        help="adapter-only artifact loaded after the base model",
    )
    parser.add_argument("--epoch_steps", default=1000, type=int)  # a mini "epoch" has [epoch_steps] steps
    parser.add_argument("--epoch_count", default=500, type=int)  # train for this many "epochs". will continue afterwards with lr = lr_final
    parser.add_argument("--epoch_begin", default=0, type=int)  # if you load a model trained for x "epochs", set epoch_begin = x
    parser.add_argument("--epoch_save", default=5, type=int)  # save the model every [epoch_save] "epochs"

    parser.add_argument("--micro_bsz", default=12, type=int)  # micro batch size (batch size per GPU)
    parser.add_argument("--n_layer", default=6, type=int)
    parser.add_argument("--n_embd", default=512, type=int)
    parser.add_argument("--dim_att", default=0, type=int)
    parser.add_argument("--dim_ffn", default=0, type=int)

    parser.add_argument("--lr_init", default=6e-4, type=float)  # 6e-4 for L12-D768, 4e-4 for L24-D1024, 3e-4 for L24-D2048
    parser.add_argument("--lr_final", default=1e-5, type=float)
    parser.add_argument("--warmup_steps", default=-1, type=int)  # try 10 if you load a model
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.99, type=float)
    parser.add_argument("--adam_eps", default=1e-18, type=float)
    parser.add_argument("--grad_cp", default=0, type=int)  # gradient checkpt: saves VRAM, but slower
    parser.add_argument("--weight_decay", default=0, type=float) # try 0.1
    parser.add_argument("--grad_clip", default=1.0, type=float) # reduce it to 0.7 / 0.5 / 0.3 / 0.2 for problematic samples

    parser.add_argument("--train_stage", default=0, type=int)  # my special pile mode
    parser.add_argument("--ds_bucket_mb", default=200, type=int)  # deepspeed bucket size in MB. 200 seems enough

    parser.add_argument("--head_size", default=64, type=int) # can try larger values for larger models
    parser.add_argument("--head_chunk", default=0, type=int) # 0 = fast, takes more VRAM; 65536 = saves 70% VRAM (when your bsz is large), slower; 4096 = saves 80% VRAM (when your bsz is large), slower
    parser.add_argument("--load_partial", default=0, type=int)
    parser.add_argument("--magic_prime", default=0, type=int)
    parser.add_argument("--my_testing", default='x070', type=str)
    parser.add_argument("--kernel", default="", type=str)
    parser.add_argument("--my_exit_tokens", default=0, type=int)

    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    fsdp2_training = str(args.strategy).lower() == "fsdp2"

    ########################################################################################################

    import os, warnings, math, datetime, sys, time
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from .checkpoint import (
        CheckpointContractError,
        CheckpointLoadKind,
        select_checkpoint_loader,
    )
    from .infctx import InfctxContractError, validate_infctx_chunk_ctx
    from .peft import LoraConfig, PeftContractError
    if "deepspeed" in args.strategy:
        import deepspeed
    from pytorch_lightning import seed_everything

    if args.random_seed >= 0:
        print(f"########## WARNING: GLOBAL SEED {args.random_seed} THIS WILL AFFECT MULTIGPU SAMPLING ##########\n" * 3)
        seed_everything(args.random_seed)

    np.set_printoptions(precision=4, suppress=True, linewidth=200)
    warnings.filterwarnings("ignore", ".*Consider increasing the value of the `num_workers` argument*")
    warnings.filterwarnings("ignore", ".*The progress bar already tracks a metric with the*")
    # os.environ["WDS_SHOW_SEED"] = "1"

    args.my_timestamp = datetime.datetime.today().strftime("%Y-%m-%d-%H-%M-%S")
    args.enable_checkpointing = False
    args.replace_sampler_ddp = False
    args.logger = False
    args.gradient_clip_val = args.grad_clip
    args.num_sanity_val_steps = 0
    args.check_val_every_n_epoch = int(1e20)
    args.log_every_n_steps = int(1e20)
    args.max_epochs = -1  # continue forever
    args.betas = (args.beta1, args.beta2)
    accumulation_steps = args.accumulate_grad_batches
    if (
        isinstance(accumulation_steps, bool)
        or not isinstance(accumulation_steps, int)
        or accumulation_steps <= 0
    ):
        raise CheckpointContractError(
            "accumulate_grad_batches must be a positive integer"
        )
    args.real_bsz = (
        int(args.num_nodes)
        * int(args.devices)
        * args.micro_bsz
        * accumulation_steps
    )
    lora_config = LoraConfig.from_namespace(args)
    args.lora_rank = lora_config.rank
    args.lora_alpha = lora_config.alpha
    args.lora_dropout = lora_config.dropout
    args.lora_target_modules = lora_config.target_modules
    if args.lora_adapter and not lora_config.enabled:
        raise PeftContractError(
            "lora_adapter requires an enabled and explicitly configured LoRA model"
        )
    if lora_config.enabled and args.train_stage == 1:
        raise PeftContractError(
            "LoRA training requires an existing base checkpoint, not train_stage=1"
        )
    if args.train_type == "infctx":
        validate_infctx_chunk_ctx(args.chunk_ctx, ctx_len=args.ctx_len)
    elif args.chunk_ctx != 0:
        raise InfctxContractError(
            "chunk_ctx is only valid when train_type is infctx"
        )
    os.environ["RWKV_MY_TESTING"] = args.my_testing
    os.environ["RWKV_KERNEL"] = args.kernel
    os.environ["RWKV_CTXLEN"] = str(args.ctx_len)
    os.environ["RWKV_HEAD_SIZE"] = str(args.head_size)
    os.environ["RWKV_HEAD_L2WRAP_CE_CHUNK"] = str(args.head_chunk)
    os.environ["RWKV_TRAIN_TYPE"] = args.train_type
    if args.dim_att <= 0:
        args.dim_att = args.n_embd
    if args.dim_ffn <= 0:
        args.dim_ffn = int((args.n_embd * 3.5) // 32 * 32) # default = 3.5x emb size

    args.run_name = f"{args.vocab_size} ctx{args.ctx_len} L{args.n_layer} D{args.n_embd}"
    if fsdp2_training:
        Path(args.proj_dir).mkdir(parents=True, exist_ok=True)
    elif not os.path.exists(args.proj_dir):
        os.makedirs(args.proj_dir)

    args.epoch_count = args.magic_prime // 40320
    args.epoch_steps = 40320 // args.real_bsz
    assert args.epoch_steps * args.real_bsz == 40320

    from .checkpoint_runner import find_latest_training_checkpoint

    resume_checkpoint = None
    if args.train_stage >= 2:
        explicit_source = Path(args.load_model) if args.load_model else None
        if explicit_source is None or explicit_source.suffix != ".pth":
            if explicit_source is not None:
                resume_source = explicit_source
            else:
                checkpoint_root = Path(args.proj_dir) / "checkpoints"
                init_weight = Path(args.proj_dir) / "rwkv-init.pth"
                if checkpoint_root.exists():
                    resume_source = find_latest_training_checkpoint(
                        Path(args.proj_dir)
                    )
                elif init_weight.is_file() and int(args.epoch_begin) == 0:
                    args.load_model = str(init_weight)
                    resume_source = None
                else:
                    resume_source = find_latest_training_checkpoint(
                        Path(args.proj_dir)
                    )
            if resume_source is not None:
                resume_plan = select_checkpoint_loader(resume_source)
                if resume_plan.manifest is None:
                    raise CheckpointContractError(
                        "standard resume checkpoint is missing its manifest"
                    )
                resume_checkpoint = resume_plan.source
                args.load_model = str(resume_plan.source)
                args.epoch_begin = resume_plan.manifest.progress.epoch
    elif args.load_model:
        explicit_source = Path(args.load_model)
        if explicit_source.is_dir() or explicit_source.name == "manifest.json":
            resume_plan = select_checkpoint_loader(explicit_source)
            if resume_plan.manifest is None:
                raise CheckpointContractError(
                    "standard resume checkpoint is missing its manifest"
                )
            resume_checkpoint = resume_plan.source
            args.load_model = str(resume_plan.source)
            args.epoch_begin = resume_plan.manifest.progress.epoch

    if resume_checkpoint is not None and args.load_partial == 1:
        raise CheckpointContractError(
            "standard resume does not allow partial model loading"
        )
    if resume_checkpoint is not None and args.lora_adapter:
        raise PeftContractError(
            "standard resume restores its own adapter and does not accept lora_adapter"
        )
    if resume_checkpoint is None and lora_config.enabled and not args.load_model:
        raise PeftContractError(
            "fresh LoRA training requires a legacy base model checkpoint"
        )
    if (
        resume_checkpoint is None
        and args.load_model
        and Path(args.load_model).suffix == ".pth"
        and int(args.epoch_begin) != 0
        and args.train_stage != 1
    ):
        raise CheckpointContractError(
            "legacy .pth is a model initialization input, not a training resume; "
            "epoch_begin must be 0"
        )

    samples_per_epoch = args.epoch_steps * args.real_bsz
    tokens_per_epoch = samples_per_epoch * args.ctx_len
    try:
        deepspeed_version = deepspeed.__version__
    except:
        deepspeed_version = None
        pass
    rank_zero_info(
        f"""
############################################################################
#
# RWKV-7 {args.precision.upper()} on {args.num_nodes}x{args.devices} {args.accelerator.upper()}, bsz {args.num_nodes}x{args.devices}x{args.micro_bsz}xaccum{args.accumulate_grad_batches}={args.real_bsz}, {args.strategy} {'with grad_cp' if args.grad_cp > 0 else ''}
#
# Data = {args.data_file} ({args.data_type}), ProjDir = {args.proj_dir}
#
# Epoch = {args.epoch_begin} to {args.epoch_begin + args.epoch_count - 1} (will continue afterwards), save every {args.epoch_save} epoch
#
# Each "epoch" = {args.epoch_steps} steps, {samples_per_epoch} samples, {tokens_per_epoch} tokens
#
# Model = {args.n_layer} n_layer, {args.n_embd} n_embd, {args.ctx_len} ctx_len
#
# Adam = lr {args.lr_init} to {args.lr_final}, warmup {args.warmup_steps} steps, beta {args.betas}, eps {args.adam_eps}
#
# Found torch {torch.__version__}, recommend latest torch
# Found deepspeed {deepspeed_version}, recommend latest deepspeed
# Found pytorch_lightning {pl.__version__}, recommend 1.9.5
#
############################################################################
"""
    )
    rank_zero_info(str(vars(args)) + "\n")

    assert args.data_type in ["binidx"]

    if args.lr_final == 0 or args.lr_init == 0:
        rank_zero_info("\n\nNote: lr_final = 0 or lr_init = 0. Using linear LR schedule instead.\n\n")

    assert args.precision in ["fp32", "tf32", "fp16", "bf16"]
    os.environ["RWKV_FLOAT_MODE"] = args.precision
    if args.precision == "fp32":
        for i in range(10):
            rank_zero_info("\n\nNote: you are using fp32 (very slow). Try bf16 / tf32 for faster training.\n\n")
    if args.precision == "fp16":
        rank_zero_info("\n\nNote: you are using fp16 (might overflow). Try bf16 / tf32 for stable training.\n\n")

    os.environ["RWKV_JIT_ON"] = "1"
    if "deepspeed_stage_3" in args.strategy:
        os.environ["RWKV_JIT_ON"] = "0" # somehow incompatible

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    if args.precision == "fp32":
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    else:
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True

    if "32" in args.precision:
        args.precision = 32
    elif args.precision == "fp16":
        args.precision = 16
    else:
        args.precision = "bf16"

    ########################################################################################################

    if fsdp2_training:
        from .fsdp2_trainer import prepare_fsdp2_launch_device

        prepare_fsdp2_launch_device()

    from .trainer import train_callback, generate_init_weight
    from .dataset import MyDataset

    train_data = MyDataset(args)
    args.vocab_size = train_data.vocab_size

    from .model import RWKV
    model = RWKV(args)

    if len(args.load_model) == 0 or args.train_stage == 1:  # shall we build the initial weights?
        init_weight_name = f"{args.proj_dir}/rwkv-init.pth"
        generate_init_weight(model, init_weight_name)  # save initial weights
        args.load_model = init_weight_name

    rank_zero_info(f"########## Loading {args.load_model}... ##########")
    if resume_checkpoint is None:
        legacy_plan = select_checkpoint_loader(
            Path(args.load_model),
            allow_legacy_pth_for_conversion=True,
        )
        if legacy_plan.kind is not CheckpointLoadKind.LEGACY_PTH_FOR_CONVERSION:
            raise CheckpointContractError(
                "model initialization requires a legacy .pth input"
            )
        load_dict = torch.load(
            legacy_plan.source,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        load_keys = list(load_dict.keys())
        for k in load_keys:
            if k.startswith('_forward_module.'):
                load_dict[k.replace('_forward_module.','')] = load_dict[k]
                del load_dict[k]
        if model.lora_config.enabled:
            model.load_base_state_dict(
                load_dict,
                allow_partial=args.load_partial == 1,
            )
            if args.lora_adapter:
                model.load_lora_adapter(Path(args.lora_adapter))
        else:
            if args.load_partial == 1:
                load_keys = load_dict.keys()
                for k in model.state_dict():
                    if k not in load_keys:
                        load_dict[k] = model.state_dict()[k]
            model.load_state_dict(load_dict)
    else:
        rank_zero_info(
            "Model, optimizer, scheduler, RNG, and data cursor will be restored "
            "together when the training runtime is initialized."
        )

    if fsdp2_training:
        from .fsdp2_trainer import run_fsdp2_training

        run_fsdp2_training(
            args,
            model,
            train_data,
            resume_checkpoint=resume_checkpoint,
        )
        return

    trainer = Trainer.from_argparse_args(
        args,
        callbacks=[
            train_callback(
                args,
                resume_checkpoint=resume_checkpoint,
            )
        ],
    )

    if trainer.global_rank == 0:
        for n in model.state_dict():
            shape = model.state_dict()[n].shape
            s0 = str(shape[0]) if len(shape) > 0 else ""
            s1 = str(shape[1]) if len(shape) > 1 else ""
            s2 = str(shape[2]) if len(shape) > 2 else ""
            s3 = str(shape[3]) if len(shape) > 3 else ""
            print(f"{s0.ljust(5)} {s1.ljust(5)} {s2.ljust(5)} {s3.ljust(5)} {n}")

    if "deepspeed" in args.strategy:
        trainer.strategy.config["zero_optimization"]["allgather_bucket_size"] = args.ds_bucket_mb * 1000 * 1000
        trainer.strategy.config["zero_optimization"]["reduce_bucket_size"] = args.ds_bucket_mb * 1000 * 1000

    # must set shuffle=False, persistent_workers=False (because worker is in another thread)
    data_loader = DataLoader(train_data, shuffle=False, pin_memory=True, batch_size=args.micro_bsz, num_workers=1, persistent_workers=False, drop_last=True)

    if trainer.global_rank == 0:
        print(f'### Preparing for training (loaded {args.load_model}). Please wait...')
    trainer.fit(model, data_loader)
