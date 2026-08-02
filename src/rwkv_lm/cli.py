########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import logging
import os


def canonicalize_training_capabilities(args: object):
    """Validate SFT, PEFT, and infctx CLI fields without selecting a model path."""

    from .infctx import InfctxContractError, validate_infctx_chunk_ctx
    from .peft import LoraConfig, PeftContractError

    lora_config = LoraConfig.from_namespace(args)
    args.lora_rank = lora_config.rank
    args.lora_alpha = lora_config.alpha
    args.lora_dropout = lora_config.dropout
    args.lora_target_modules = lora_config.target_modules
    if getattr(args, "lora_adapter", "") and not lora_config.enabled:
        raise PeftContractError(
            "lora_adapter requires an enabled and explicitly configured LoRA model"
        )
    if lora_config.enabled and getattr(args, "train_stage", 0) == 1:
        raise PeftContractError(
            "LoRA training requires an existing base checkpoint, not train_stage=1"
        )
    if getattr(args, "train_type", "standard") == "infctx":
        validate_infctx_chunk_ctx(args.chunk_ctx, ctx_len=args.ctx_len)
    elif getattr(args, "chunk_ctx", 0) != 0:
        raise InfctxContractError(
            "chunk_ctx is only valid when train_type is infctx"
        )
    return lora_config


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    from argparse import ArgumentParser
    from pathlib import Path

    def rank_zero_info(message: object) -> None:
        if int(os.environ.get("RANK", "0")) == 0:
            logging.info(message)

    parser = ArgumentParser()

    parser.add_argument(
        "--load_model",
        default="",
        type=str,
    )  # standard Transformers model directory or complete training checkpoint
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
    parser.add_argument("--head_size", default=64, type=int) # can try larger values for larger models
    parser.add_argument("--head_chunk", default=0, type=int) # 0 = fast, takes more VRAM; 65536 = saves 70% VRAM (when your bsz is large), slower; 4096 = saves 80% VRAM (when your bsz is large), slower
    parser.add_argument("--load_partial", default=0, type=int)
    parser.add_argument("--magic_prime", default=0, type=int)
    parser.add_argument("--my_testing", default='x070', type=str)
    parser.add_argument("--kernel", default="", type=str)
    parser.add_argument("--my_exit_tokens", default=0, type=int)
    parser.add_argument(
        "--wkv_backend",
        choices=("reference", "flash_rwkv"),
        default="flash_rwkv",
        help="explicit transformers-rwkv WKV backend; accelerated requests fail closed",
    )

    parser.add_argument("--accelerator", choices=("gpu", "cuda"), default="gpu")
    parser.add_argument("--devices", default=1, type=int)
    parser.add_argument("--num_nodes", default=1, type=int)
    parser.add_argument(
        "--precision",
        choices=("fp32", "tf32", "fp16", "bf16"),
        default="bf16",
    )
    parser.add_argument("--strategy", choices=("fsdp2",), default="fsdp2")
    parser.add_argument("--accumulate_grad_batches", default=1, type=int)
    parser.add_argument("--enable_progress_bar", default=False)
    args = parser.parse_args()
    fsdp2_training = str(args.strategy).lower() == "fsdp2"

    ########################################################################################################

    import datetime, random, warnings
    import numpy as np
    import torch
    from .checkpoint import (
        CheckpointContractError,
        select_checkpoint_loader,
    )
    from .peft import PeftContractError
    from .standard_model import StandardModelContractError

    if not fsdp2_training:
        raise StandardModelContractError(
            "the standard RWKV-7 training runner requires --strategy fsdp2; "
            "the native Lightning/DeepSpeed model path has been retired"
        )

    if args.random_seed >= 0:
        print(f"########## WARNING: GLOBAL SEED {args.random_seed} THIS WILL AFFECT MULTIGPU SAMPLING ##########\n" * 3)
        random.seed(args.random_seed)
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)

    np.set_printoptions(precision=4, suppress=True, linewidth=200)
    warnings.filterwarnings("ignore", ".*Consider increasing the value of the `num_workers` argument*")
    warnings.filterwarnings("ignore", ".*The progress bar already tracks a metric with the*")
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
    lora_config = canonicalize_training_capabilities(args)
    if args.dim_att <= 0:
        args.dim_att = args.n_embd
    elif args.dim_att != args.n_embd:
        raise StandardModelContractError(
            "transformers-rwkv requires dim_att to equal n_embd"
        )
    if args.dim_ffn <= 0:
        args.dim_ffn = int((args.n_embd * 3.5) // 32 * 32) # default = 3.5x emb size
    if args.load_partial != 0:
        raise StandardModelContractError(
            "standard transformers-rwkv checkpoints require strict model loading"
        )
    if args.head_chunk != 0 or args.kernel:
        raise StandardModelContractError(
            "head_chunk and kernel select the retired native model path; use "
            "wkv_backend with the standard model"
        )

    Path(args.proj_dir).mkdir(parents=True, exist_ok=True)

    args.epoch_count = args.magic_prime // 40320
    args.epoch_steps = 40320 // args.real_bsz
    assert args.epoch_steps * args.real_bsz == 40320

    from .checkpoint_runner import find_latest_training_checkpoint

    explicit_source = Path(args.load_model) if args.load_model else None
    if explicit_source is not None and explicit_source.suffix == ".pth":
        raise StandardModelContractError(
            "legacy .pth cannot be loaded by rwkv-train; convert it with "
            "rwkv-convert-legacy-checkpoint first"
        )
    model_source = None
    resume_checkpoint = None
    if explicit_source is not None:
        manifest_path = (
            explicit_source
            if explicit_source.name == "manifest.json"
            else explicit_source / "manifest.json"
        )
        if manifest_path.is_file():
            resume_plan = select_checkpoint_loader(explicit_source)
            if resume_plan.manifest is None:
                raise CheckpointContractError(
                    "standard resume checkpoint is missing its manifest"
                )
            resume_checkpoint = resume_plan.source
            args.load_model = str(resume_plan.source)
            args.epoch_begin = resume_plan.manifest.progress.epoch
        else:
            model_source = explicit_source
    elif args.train_stage >= 2:
        checkpoint_root = Path(args.proj_dir) / "checkpoints"
        init_model = Path(args.proj_dir) / "rwkv-init"
        if checkpoint_root.exists():
            resume_source = find_latest_training_checkpoint(Path(args.proj_dir))
            resume_plan = select_checkpoint_loader(resume_source)
            if resume_plan.manifest is None:
                raise CheckpointContractError(
                    "standard resume checkpoint is missing its manifest"
                )
            resume_checkpoint = resume_plan.source
            args.load_model = str(resume_plan.source)
            args.epoch_begin = resume_plan.manifest.progress.epoch
        elif init_model.is_dir() and int(args.epoch_begin) == 0:
            model_source = init_model
            args.load_model = str(init_model)
        else:
            raise CheckpointContractError(
                "train_stage >= 2 requires a complete training checkpoint or "
                "standard rwkv-init model"
            )
    if resume_checkpoint is not None and args.lora_adapter:
        raise PeftContractError(
            "standard resume restores its own adapter and does not accept "
            "lora_adapter"
        )
    if lora_config.enabled and resume_checkpoint is None and model_source is None:
        raise PeftContractError(
            "fresh LoRA training requires a standard base model directory"
        )

    samples_per_epoch = args.epoch_steps * args.real_bsz
    tokens_per_epoch = samples_per_epoch * args.ctx_len
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
# Model owner = transformers-rwkv Rwkv7ForCausalLM ({args.wkv_backend})
#
############################################################################
"""
    )
    rank_zero_info(str(vars(args)) + "\n")

    assert args.data_type in ["binidx"]

    if args.lr_final == 0 or args.lr_init == 0:
        rank_zero_info("\n\nNote: lr_final = 0 or lr_init = 0. Using linear LR schedule instead.\n\n")

    assert args.precision in ["fp32", "tf32", "fp16", "bf16"]
    if args.precision == "fp32":
        for i in range(10):
            rank_zero_info("\n\nNote: you are using fp32 (very slow). Try bf16 / tf32 for faster training.\n\n")
    if args.precision == "fp16":
        rank_zero_info("\n\nNote: you are using fp16 (might overflow). Try bf16 / tf32 for stable training.\n\n")

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

    from .fsdp2_trainer import prepare_fsdp2_launch_device

    prepare_fsdp2_launch_device()

    from .dataset import MyDataset
    from .standard_model import (
        configure_standard_rwkv7_peft,
        create_standard_rwkv7_model,
        save_standard_rwkv7_model,
    )

    train_data = MyDataset(args)
    args.vocab_size = train_data.vocab_size
    args.run_name = (
        f"{args.vocab_size} ctx{args.ctx_len} L{args.n_layer} D{args.n_embd}"
    )

    if resume_checkpoint is not None:
        model = create_standard_rwkv7_model(args)
        rank_zero_info(
            "Model, optimizer, scheduler, RNG, and data cursor will be restored "
            "together when the FSDP2 runtime is initialized."
        )
    elif model_source is not None:
        rank_zero_info(f"########## Loading {model_source}... ##########")
        model = create_standard_rwkv7_model(args, model_source=model_source)
    else:
        model = create_standard_rwkv7_model(args)
        init_model = Path(args.proj_dir) / "rwkv-init"
        save_standard_rwkv7_model(model, init_model)
        args.load_model = str(init_model)
        rank_zero_info(
            f"########## Initialized standard model at {init_model} ##########"
        )
    configure_standard_rwkv7_peft(
        model,
        lora_config,
        adapter_path=Path(args.lora_adapter) if args.lora_adapter else None,
    )

    from .fsdp2_trainer import run_fsdp2_training

    run_fsdp2_training(
        args,
        model,
        train_data,
        resume_checkpoint=resume_checkpoint,
    )
