#!/bin/bash
#######################################################################################################################
#
# Convert a legacy .pth once with rwkv-convert-legacy-checkpoint, or use an
# existing standard transformers-rwkv model directory at PROJ_DIR/rwkv-init.
#
# The trainer resumes from the newest verified checkpoints/epoch-* directory.
# PROJ_DIR/rwkv-init must be a standard model directory, never a raw .pth file.
# FSDP2 writes one verified sharded checkpoint transaction across all ranks.
#
#######################################################################################################################
#
MODEL_TYPE="x070" # x070 => rwkv-7.0
#
N_LAYER="12"
N_EMBD="768"
#
CTX_LEN="512" # !!! change magic_prime if you change ctx_len !!!
PROJ_DIR="out/L"$N_LAYER"-D"$N_EMBD"-"$MODEL_TYPE # set output folder
#
#######################################################################################################################
#
# Note bsz & lr affects model & training performance
# Small data => use smaller bsz & slightly smaller LR
# Large data => use larger bsz & slightly larger LR
# Larger model => use smaller LR
# Finetuning => use very small LR, such as 1e-5
#
M_BSZ="16" # takes ~7G VRAM => reduce this to save VRAM, increase this for faster speed
LR_INIT="6e-4"
LR_FINAL="6e-5"
GRAD_CP=1 # 1 => slower, save VRAM; 0 => faster, more VRAM
EPOCH_SAVE=10 # save every 10 "miniepochs" (1 miniepoch = 40320 * ctx_len tokens) => decrease if your GPU is weak
#
#######################################################################################################################
#
# magic_prime = the largest 3n+2 prime smaller than datalen/ctxlen-1 (= 1498226207/512-1 = 2926222.06 in this case) = 2926181 in this case
# use https://www.dcode.fr/prime-numbers-search
#
N_NODE=1 # number of nodes
GPU_PER_NODE=1 # number of GPUs per node
#
torchrun --standalone --nnodes=$N_NODE --nproc-per-node=$GPU_PER_NODE train.py --wandb "Test" --proj_dir $PROJ_DIR \
 --ctx_len $CTX_LEN --train_stage 3 --epoch_count 999999 --epoch_begin 0 \
 --data_file "data/minipile" --my_exit_tokens 1498226207 --magic_prime 2926181 \
 --num_nodes $N_NODE --micro_bsz $M_BSZ --n_layer $N_LAYER --n_embd $N_EMBD \
 --lr_init $LR_INIT --lr_final $LR_FINAL --warmup_steps 10 --beta1 0.9 --beta2 0.99 --adam_eps 1e-18 --data_type "binidx" --vocab_size 65536 \
 --weight_decay 0.001 --epoch_save $EPOCH_SAVE --head_size 64 --wkv_backend flash_rwkv \
 --accelerator gpu --devices $GPU_PER_NODE --precision bf16 --strategy fsdp2 --grad_cp $GRAD_CP
