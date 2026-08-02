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
CTX_LEN="4096" # !!! change magic_prime if you change ctx_len !!!
PROJ_DIR="out/L"$N_LAYER"-D"$N_EMBD"-"$MODEL_TYPE # set output folder
#
#######################################################################################################################
#
M_BSZ="30" # for 80G VRAM GPUs
LR_INIT="8e-4"
LR_FINAL="3e-5"
#
W_DECAY="0.1"
BETA_2="0.99"
ADAM_EPS="1e-18"
#
GRAD_CP=1 # 1 => slower, save VRAM; 0 => faster, more VRAM
EPOCH_SAVE=50 # save every 50 "miniepochs" (1 miniepoch = 40320 * ctx_len tokens) => decrease if your GPU is weak
#
#######################################################################################################################
#
N_NODE=1 # number of nodes
GPU_PER_NODE=8 # number of GPUs per node
#
torchrun --standalone --nnodes=$N_NODE --nproc-per-node=$GPU_PER_NODE train.py --wandb "RWKV-7-Pile" --proj_dir $PROJ_DIR \
 --ctx_len $CTX_LEN --train_stage 3 --epoch_count 999999 --epoch_begin 0 \
 --data_file "/mnt/nvme0n1/pile/pile_20B_tokenizer_text_document" --my_exit_tokens 332115325534 --magic_prime 81082817 \
 --num_nodes $N_NODE --micro_bsz $M_BSZ --n_layer $N_LAYER --n_embd $N_EMBD \
 --lr_init $LR_INIT --lr_final $LR_FINAL --warmup_steps 10 --beta1 0.9 --beta2 $BETA_2 --adam_eps $ADAM_EPS --data_type "binidx" --vocab_size 50304 \
 --weight_decay $W_DECAY --epoch_save $EPOCH_SAVE --head_size 64 --wkv_backend flash_rwkv \
 --accelerator gpu --devices $GPU_PER_NODE --precision bf16 --strategy fsdp2 --grad_cp $GRAD_CP
