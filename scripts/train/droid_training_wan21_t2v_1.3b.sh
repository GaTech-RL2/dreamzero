#!/bin/bash
# DreamZero DROID Full Fine-Tuning Script with Wan2.1-T2V-1.3B backbone
#
# Usage:
#   bash scripts/train/droid_training_wan21_t2v_1.3b.sh
#
# Wan2.1-T2V-1.3B is a pure TEXT-to-video model: no image encoder, no first-frame input channels.
# DreamZero conditions on the current observation through its intrinsic clean_x causal path
# (clean observed latents fed through the causal DiT) + text cross-attention, so CLIP and the
# first-frame latent concat are disabled. The pretrained 16-channel patch_embedding loads exactly.
#
# Prerequisites:
#   - DROID dataset in LeRobot format at DROID_DATA_ROOT
#     Download: huggingface-cli download GEAR-Dreams/DreamZero-DROID-Data --repo-type dataset --local-dir ./data/droid_lerobot
#   - Wan2.1-T2V-1.3B weights (auto-downloaded or pre-downloaded from HuggingFace)
#     huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir ./checkpoints/Wan2.1-T2V-1.3B
#     (This repo bundles the DiT, the umt5-xxl T5 encoder, and the Wan2.1 VAE; no CLIP needed.)
#   - umt5-xxl tokenizer (auto-downloaded or pre-downloaded)
#     huggingface-cli download google/umt5-xxl --local-dir ./checkpoints/umt5-xxl

export HYDRA_FULL_ERROR=1
# Reduce CUDA fragmentation. Helps full fine-tune fit on a single 44-48GB card (the frozen umt5-xxl
# encoder is ~11GB resident); no-op/harmless on multi-GPU where ZeRO-2 already shards optimizer state.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# Repo root (same logic as droid_training_wan22.sh)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
if [ -n "$DREAMZERO_ROOT" ] && [ -d "$DREAMZERO_ROOT/groot" ]; then
    :
elif [ -d "/root/yejink/dreamzero/groot" ]; then
    DREAMZERO_ROOT=/root/yejink/dreamzero
elif [ -d "/root/dreamzero/groot" ]; then
    DREAMZERO_ROOT=/root/dreamzero
elif [ -d "$SCRIPT_REPO_ROOT/groot" ]; then
    DREAMZERO_ROOT="$SCRIPT_REPO_ROOT"
else
    DREAMZERO_ROOT="${DREAMZERO_ROOT:-/root/yejink/dreamzero}"
fi
if [ ! -d "$DREAMZERO_ROOT/groot" ]; then
    echo "ERROR: No groot/ under $DREAMZERO_ROOT. Set DREAMZERO_ROOT to the dreamzero repo root that contains groot/."
    exit 1
fi

# ============ USER CONFIGURATION ============
DROID_DATA_ROOT=${DROID_DATA_ROOT:-"$DREAMZERO_ROOT/data/droid_lerobot"}
if [ "$DROID_DATA_ROOT" = "./data/droid_lerobot" ]; then
    DROID_DATA_ROOT="$DREAMZERO_ROOT/data/droid_lerobot"
fi
OUTPUT_DIR=${OUTPUT_DIR:-"$DREAMZERO_ROOT/checkpoints/dreamzero_droid_wan21_t2v_1.3b_full_finetune"}

NUM_GPUS=${NUM_GPUS:-4}
PER_DEVICE_BS=${PER_DEVICE_BS:-1}
# Global batch: default = NUM_GPUS * PER_DEVICE_BS. Override for larger effective batch via grad accum.
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-$((NUM_GPUS * PER_DEVICE_BS))}

# Wan2.1-T2V-1.3B checkpoint (bundles DiT, T5 text encoder, and Wan2.1 VAE). No CLIP image encoder.
WAN_CKPT_DIR=${WAN_CKPT_DIR:-"$DREAMZERO_ROOT/checkpoints/Wan2.1-T2V-1.3B"}
TOKENIZER_DIR=${TOKENIZER_DIR:-"$DREAMZERO_ROOT/checkpoints/umt5-xxl"}
# 1.3B is small; default to ZeRO-2 (no offload). Set DEEPSPEED_CFG=zero2_offload to trade speed for memory.
DEEPSPEED_CFG=${DEEPSPEED_CFG:-zero2}
# =============================================

# ============ AUTO-DOWNLOAD WEIGHTS ============
if [ ! -d "$WAN_CKPT_DIR" ] || [ -z "$(ls -A "$WAN_CKPT_DIR" 2>/dev/null)" ]; then
    echo "Wan2.1-T2V-1.3B not found at $WAN_CKPT_DIR. Downloading from HuggingFace..."
    huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir "$WAN_CKPT_DIR"
fi

if [ ! -d "$TOKENIZER_DIR" ] || [ -z "$(ls -A "$TOKENIZER_DIR" 2>/dev/null)" ]; then
    echo "umt5-xxl tokenizer not found at $TOKENIZER_DIR. Downloading from HuggingFace..."
    huggingface-cli download google/umt5-xxl --local-dir "$TOKENIZER_DIR"
fi
# ================================================

if [ ! -d "$DROID_DATA_ROOT" ]; then
    echo "ERROR: DROID dataset not found at $DROID_DATA_ROOT"
    echo "Download with: huggingface-cli download GEAR-Dreams/DreamZero-DROID-Data --repo-type dataset --local-dir $DROID_DATA_ROOT"
    exit 1
fi

EXPERIMENT_PY="$DREAMZERO_ROOT/groot/vla/experiment/experiment.py"
if [ ! -f "$EXPERIMENT_PY" ]; then
    echo "ERROR: Not found: $EXPERIMENT_PY"
    exit 1
fi
PYTHON_311="/usr/bin/python3.11"
if [ -x "$PYTHON_311" ]; then
    if [ -n "${FIX_NUMPY_IN_SCRIPT:-}" ]; then
        "$PYTHON_311" -m pip install "numpy==1.26.4" --force-reinstall -q 2>/dev/null || true
    fi
    RUN_CMD=( "$PYTHON_311" -m torch.distributed.run --nproc_per_node "$NUM_GPUS" --standalone "$EXPERIMENT_PY" )
    echo "Using image Python 3.11: $PYTHON_311"
else
    RUN_CMD=( python3 -m torch.distributed.run --nproc_per_node "$NUM_GPUS" --standalone "$EXPERIMENT_PY" )
    echo "Using: $(command -v python3)"
fi
cd "$DREAMZERO_ROOT"

# Full fine-tune: train_architecture=full, save_lora_only=false.
# T2V-1.3B shares the Wan2.1 VAE (z=16, 8x) / patch / 320x176 resolution with the 14B path, so
# frame_seqlen stays 880 (inherited from the action_head config) and data=droid_relative is reused.
# Note: no image_encoder_pretrained_path (the T2V config sets image_encoder_cfg=null).
"${RUN_CMD[@]}" \
    report_to=wandb \
    data=dreamzero/droid_relative \
    wandb_project=dreamzero \
    train_architecture=full \
    num_frames=33 \
    action_horizon=24 \
    num_views=3 \
    model=dreamzero/vla \
    model/dreamzero/action_head=wan_flow_matching_action_tf_wan21_t2v \
    model/dreamzero/transform=dreamzero_cotrain \
    num_frame_per_block=2 \
    num_action_per_block=24 \
    num_state_per_block=1 \
    seed=42 \
    training_args.learning_rate=1e-5 \
    training_args.deepspeed="groot/vla/configs/deepspeed/${DEEPSPEED_CFG}.json" \
    save_steps=500 \
    training_args.warmup_ratio=0.05 \
    output_dir=$OUTPUT_DIR \
    per_device_train_batch_size=$PER_DEVICE_BS \
    global_batch_size=$GLOBAL_BATCH_SIZE \
    max_steps=200000 \
    weight_decay=1e-5 \
    save_total_limit=10 \
    upload_checkpoints=false \
    bf16=true \
    tf32=true \
    eval_bf16=true \
    dataloader_pin_memory=true \
    dataloader_num_workers=4 \
    image_resolution_width=320 \
    image_resolution_height=176 \
    save_lora_only=false \
    max_chunk_size=4 \
    save_strategy=steps \
    droid_data_root=$DROID_DATA_ROOT \
    dit_version=$WAN_CKPT_DIR \
    text_encoder_pretrained_path=$WAN_CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth \
    vae_pretrained_path=$WAN_CKPT_DIR/Wan2.1_VAE.pth \
    tokenizer_path=$TOKENIZER_DIR
