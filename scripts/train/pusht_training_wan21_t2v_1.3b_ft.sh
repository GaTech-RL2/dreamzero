#!/bin/bash
# DreamZero PushT training -- Wan2.1-T2V 1.3B DiT, FULL fine-tune of the PRETRAINED backbone.
#
# This is the on-recipe DreamZero 1.3B setup (matches droid_training_wan21_t2v_1.3b.sh): the pretrained
# Wan2.1-T2V-1.3B video DiT is LOADED (skip_component_loading=false, dit_version -> the downloaded
# Wan2.1-T2V-1.3B/) and ALL its params are fine-tuned (train_architecture=full) jointly with the new
# action head. Contrast: pusht_training_1.3b_lora.sh freezes the DiT + trains LoRA adapters. Full-FT
# keeps the pretrained video prior but adapts every weight -> LR is LOW (1e-5) to avoid wrecking it
# (vs 1e-4 for LoRA).
#
# T2V backbone: no CLIP, no first-frame latent concat; observation rides the clean_x causal path.
# The frozen umt5-xxl T5 + Wan2.1 z=16 VAE are shared across Wan2.1 sizes, so they reuse the local
# Wan2.1-I2V-14B-480P copy. Single view at NATIVE 96x96 => frame_seqlen = (96/8/2)^2 = 6*6 = 36.
#
# Usage:
#   NUM_GPUS=4 bash scripts/train/pusht_training_wan21_t2v_1.3b_ft.sh
#   bash scripts/train/pusht_training_wan21_t2v_1.3b_ft.sh max_steps=2 report_to=none

export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export ATTENTION_BACKEND=${ATTENTION_BACKEND:-torch}
export TOKENIZERS_PARALLELISM=false

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DREAMZERO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$DREAMZERO_ROOT"

# ============ CONFIG ============
DATA_ROOT=${DATA_ROOT:-"$DREAMZERO_ROOT/data/pusht_lerobot"}
OUTPUT_DIR=${OUTPUT_DIR:-"$DREAMZERO_ROOT/checkpoints/dreamzero_pusht_1.3b_ft"}
WAN21_CKPT_DIR=${WAN21_CKPT_DIR:-"$DREAMZERO_ROOT/checkpoints/Wan2.1-I2V-14B-480P"}   # shared T5 + VAE
DIT_CKPT_DIR=${DIT_CKPT_DIR:-"$DREAMZERO_ROOT/checkpoints/Wan2.1-T2V-1.3B"}            # pretrained 1.3B DiT
TOKENIZER_DIR=${TOKENIZER_DIR:-"$WAN21_CKPT_DIR/google/umt5-xxl"}
NUM_GPUS=${NUM_GPUS:-4}
PER_DEVICE_BS=${PER_DEVICE_BS:-1}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-$((NUM_GPUS * PER_DEVICE_BS))}
MAX_STEPS=${MAX_STEPS:-20000}
SAVE_STEPS=${SAVE_STEPS:-500}
LR=${LR:-1e-5}                # full fine-tune of pretrained weights -> low LR (matches droid_training_wan21_t2v_1.3b.sh)
REPORT_TO=${REPORT_TO:-wandb}
DEEPSPEED_CFG=${DEEPSPEED_CFG:-zero2}
VENVPY="$DREAMZERO_ROOT/.venv/bin/python"
# ================================

if [ ! -d "$DATA_ROOT" ]; then echo "ERROR: PushT dataset not found at $DATA_ROOT"; exit 1; fi
if [ ! -d "$WAN21_CKPT_DIR" ]; then echo "ERROR: Wan2.1-I2V-14B-480P (shared T5/VAE) not found at $WAN21_CKPT_DIR"; exit 1; fi
if [ ! -f "$DIT_CKPT_DIR/diffusion_pytorch_model.safetensors" ]; then
    echo "ERROR: pretrained Wan2.1-T2V-1.3B DiT not found at $DIT_CKPT_DIR (need diffusion_pytorch_model.safetensors)."
    echo "  download: .venv/bin/huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --include diffusion_pytorch_model.safetensors config.json --local-dir $DIT_CKPT_DIR"
    exit 1
fi

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
    model/dreamzero/action_head=wan_flow_matching_action_tf_wan21_t2v \
    model/dreamzero/transform=dreamzero_cotrain \
    ++action_head_cfg.config.skip_component_loading=false \
    num_frame_per_block=2 \
    num_action_per_block=24 \
    num_state_per_block=1 \
    frame_seqlen=36 \
    fixed_num_chunks=${FIXED_NUM_CHUNKS:-null} \
    image_resolution_width=96 \
    image_resolution_height=96 \
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
    save_total_limit=5 \
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
    dit_version=$DIT_CKPT_DIR \
    text_encoder_pretrained_path=$WAN21_CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth \
    vae_pretrained_path=$WAN21_CKPT_DIR/Wan2.1_VAE.pth \
    tokenizer_path=$TOKENIZER_DIR \
    "$@"
