#!/bin/bash
# DreamZero PushT training -- ~300M image+text-to-video (i2v) DiT, FROM SCRATCH.
#
# Single top-down RGB view (resized 96x96 -> 256x256), 2D absolute target-position action.
# The ~300M DiT trains from random init (the 300m action-head config sets skip_component_loading=true,
# so only the frozen T5 + CLIP + VAE load, from the local Wan2.1-I2V-14B-480P checkpoint).
# Single view => frame_seqlen = (256/8/2)*(256/8/2) = 16*16 = 256 (overrides the default 880).
#
# Usage:
#   NUM_GPUS=4 bash scripts/train/pusht_training_300m.sh
#   # extra hydra overrides pass through, e.g.:  bash scripts/train/pusht_training_300m.sh max_steps=2 report_to=none

export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export ATTENTION_BACKEND=${ATTENTION_BACKEND:-torch}   # SDPA self-attn (no flash_attn needed)
export TOKENIZERS_PARALLELISM=false

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DREAMZERO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$DREAMZERO_ROOT"

# ============ CONFIG ============
DATA_ROOT=${DATA_ROOT:-"$DREAMZERO_ROOT/data/pusht_lerobot"}
OUTPUT_DIR=${OUTPUT_DIR:-"$DREAMZERO_ROOT/checkpoints/dreamzero_pusht_300m"}
WAN21_CKPT_DIR=${WAN21_CKPT_DIR:-"$DREAMZERO_ROOT/checkpoints/Wan2.1-I2V-14B-480P"}
TOKENIZER_DIR=${TOKENIZER_DIR:-"$WAN21_CKPT_DIR/google/umt5-xxl"}
NUM_GPUS=${NUM_GPUS:-4}
PER_DEVICE_BS=${PER_DEVICE_BS:-1}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-$((NUM_GPUS * PER_DEVICE_BS))}
MAX_STEPS=${MAX_STEPS:-60000}
SAVE_STEPS=${SAVE_STEPS:-2000}
LR=${LR:-1e-4}
REPORT_TO=${REPORT_TO:-wandb}
DEEPSPEED_CFG=${DEEPSPEED_CFG:-zero2_offload}
VENVPY="$DREAMZERO_ROOT/.venv/bin/python"
# ================================

if [ ! -d "$DATA_ROOT" ]; then echo "ERROR: PushT dataset not found at $DATA_ROOT (run scripts/data/convert_pusht_to_lerobot.py)"; exit 1; fi
if [ ! -d "$WAN21_CKPT_DIR" ]; then echo "ERROR: Wan2.1-I2V-14B-480P (frozen T5/CLIP/VAE) not found at $WAN21_CKPT_DIR"; exit 1; fi

"$VENVPY" -m torch.distributed.run --nproc_per_node "$NUM_GPUS" --standalone \
    "$DREAMZERO_ROOT/groot/vla/experiment/experiment.py" \
    report_to=$REPORT_TO \
    data=dreamzero/pusht_relative \
    wandb_project=dreamzero_pusht \
    train_architecture=full \
    num_frames=33 \
    action_horizon=24 \
    num_views=1 \
    model=dreamzero/vla \
    model/dreamzero/action_head=wan_flow_matching_action_tf_300m \
    model/dreamzero/transform=dreamzero_cotrain \
    num_frame_per_block=2 \
    num_action_per_block=24 \
    num_state_per_block=1 \
    frame_seqlen=256 \
    image_resolution_width=256 \
    image_resolution_height=256 \
    seed=42 \
    training_args.learning_rate=$LR \
    training_args.deepspeed="groot/vla/configs/deepspeed/${DEEPSPEED_CFG}.json" \
    save_steps=$SAVE_STEPS \
    training_args.warmup_ratio=0.05 \
    output_dir=$OUTPUT_DIR \
    per_device_train_batch_size=$PER_DEVICE_BS \
    global_batch_size=$GLOBAL_BATCH_SIZE \
    max_steps=$MAX_STEPS \
    weight_decay=1e-5 \
    save_total_limit=10 \
    upload_checkpoints=false \
    bf16=true \
    tf32=true \
    eval_bf16=true \
    dataloader_pin_memory=true \
    dataloader_num_workers=4 \
    save_lora_only=false \
    max_chunk_size=4 \
    save_strategy=steps \
    pusht_data_root=$DATA_ROOT \
    dit_version=$WAN21_CKPT_DIR \
    text_encoder_pretrained_path=$WAN21_CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth \
    image_encoder_pretrained_path=$WAN21_CKPT_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
    vae_pretrained_path=$WAN21_CKPT_DIR/Wan2.1_VAE.pth \
    tokenizer_path=$TOKENIZER_DIR \
    "$@"
