#!/bin/bash
# DreamZero PushT training -- Wan2.2-TI2V-5B DiT, LoRA fine-tune of the PRETRAINED backbone.
#
# Like pusht_training_1.3b_lora.sh but on the 5B (Wan2.2-TI2V-5B) backbone:
#   - DiT loads pretrained + FROZEN; LoRA adapters injected; the action head (state/action
#     encoder+decoder) stays fully trainable. save_lora_only=false -> self-contained eval checkpoints.
#   - Wan2.2-TI2V-5B is text+image-to-video: in_dim=48 latent (no [x;y] concat), first frame via CLIP.
#     Uses the Wan2.2 VAE (WanVideoVAE38, z=48, 16x spatial downscale) -- NOT the Wan2.1 z=16 VAE.
#   - The umt5-xxl T5 and CLIP are shared, so they reuse the local Wan2.1-I2V-14B-480P copy; only the
#     5B DiT (+ Wan2.2_VAE.pth) come from the Wan2.2-TI2V-5B download.
#   - Single view 256x256: WanVideoVAE38 16x -> latent 16x16, DiT patch stride 2 -> 8x8 => frame_seqlen=64.
#
# Per-device batching only (NO gradient accumulation): GLOBAL_BATCH_SIZE = PER_DEVICE_BS * NUM_GPUS.
# Effective batch 16 (per_device 4) or 32 (per_device 8) on 4 GPUs.
#
# Usage:
#   NUM_GPUS=4 PER_DEVICE_BS=8 bash scripts/train/pusht_training_5b_lora.sh
#   bash scripts/train/pusht_training_5b_lora.sh max_steps=2 report_to=none

export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export ATTENTION_BACKEND=${ATTENTION_BACKEND:-torch}
export TOKENIZERS_PARALLELISM=false

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DREAMZERO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$DREAMZERO_ROOT"

# ============ CONFIG ============
DATA_ROOT=${DATA_ROOT:-"$DREAMZERO_ROOT/data/pusht_lerobot"}
OUTPUT_DIR=${OUTPUT_DIR:-"$DREAMZERO_ROOT/checkpoints/dreamzero_pusht_5b_lora"}
WAN21_CKPT_DIR=${WAN21_CKPT_DIR:-"$DREAMZERO_ROOT/checkpoints/Wan2.1-I2V-14B-480P"}   # shared T5 + CLIP
WAN22_CKPT_DIR=${WAN22_CKPT_DIR:-"$DREAMZERO_ROOT/checkpoints/Wan2.2-TI2V-5B"}        # pretrained 5B DiT + z=48 VAE
TOKENIZER_DIR=${TOKENIZER_DIR:-"$WAN21_CKPT_DIR/google/umt5-xxl"}
NUM_GPUS=${NUM_GPUS:-4}
PER_DEVICE_BS=${PER_DEVICE_BS:-8}
# NO grad accum: global batch is exactly per_device * num_gpus.
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-$((NUM_GPUS * PER_DEVICE_BS))}
MAX_STEPS=${MAX_STEPS:-20000}
SAVE_STEPS=${SAVE_STEPS:-500}
LR=${LR:-1e-4}                # LoRA default (matches droid_training_lora.sh)
REPORT_TO=${REPORT_TO:-wandb}
DEEPSPEED_CFG=${DEEPSPEED_CFG:-zero2}
VENVPY="$DREAMZERO_ROOT/.venv/bin/python"
# ================================

if [ ! -d "$DATA_ROOT" ]; then echo "ERROR: PushT dataset not found at $DATA_ROOT"; exit 1; fi
if [ ! -d "$WAN21_CKPT_DIR" ]; then echo "ERROR: Wan2.1-I2V-14B-480P (shared T5/CLIP) not found at $WAN21_CKPT_DIR"; exit 1; fi
if [ ! -f "$WAN22_CKPT_DIR/diffusion_pytorch_model.safetensors.index.json" ]; then
    echo "ERROR: pretrained Wan2.2-TI2V-5B DiT not found at $WAN22_CKPT_DIR (need the sharded DiT + Wan2.2_VAE.pth)."
    echo "  download: .venv/bin/huggingface-cli download Wan-AI/Wan2.2-TI2V-5B --include 'diffusion_pytorch_model*' config.json Wan2.2_VAE.pth --local-dir $WAN22_CKPT_DIR"
    exit 1
fi
if [ ! -f "$WAN22_CKPT_DIR/Wan2.2_VAE.pth" ]; then echo "ERROR: $WAN22_CKPT_DIR/Wan2.2_VAE.pth missing"; exit 1; fi

"$VENVPY" -m torch.distributed.run --nproc_per_node "$NUM_GPUS" --standalone \
    "$DREAMZERO_ROOT/groot/vla/experiment/experiment.py" \
    report_to=$REPORT_TO \
    data=dreamzero/pusht_relative \
    wandb_project=dreamzero_pusht \
    train_architecture=lora \
    num_frames=33 \
    action_horizon=24 \
    num_views=1 \
    model=dreamzero/vla \
    model/dreamzero/action_head=wan_flow_matching_action_tf_wan22 \
    model/dreamzero/transform=dreamzero_cotrain \
    ++action_head_cfg.config.skip_component_loading=false \
    ++action_head_cfg.config.target_video_height=256 \
    ++action_head_cfg.config.target_video_width=256 \
    num_frame_per_block=2 \
    num_action_per_block=24 \
    num_state_per_block=1 \
    frame_seqlen=64 \
    fixed_num_chunks=${FIXED_NUM_CHUNKS:-2} \
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
    dit_version=$WAN22_CKPT_DIR \
    text_encoder_pretrained_path=$WAN21_CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth \
    image_encoder_pretrained_path=$WAN21_CKPT_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
    vae_pretrained_path=$WAN22_CKPT_DIR/Wan2.2_VAE.pth \
    tokenizer_path=$TOKENIZER_DIR \
    "$@"
