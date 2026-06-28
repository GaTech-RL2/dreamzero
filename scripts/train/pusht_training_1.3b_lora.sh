#!/bin/bash
# DreamZero PushT training -- Wan2.1-T2V 1.3B DiT, LoRA fine-tune of the PRETRAINED backbone.
#
# Unlike pusht_training_wan21_t2v_1.3b.sh (from scratch, dit_version=null), this ADAPTS the released
# Wan2.1-T2V-1.3B video prior: the DiT loads pretrained (skip_component_loading=false, dit_version ->
# the downloaded Wan2.1-T2V-1.3B/) and is FROZEN; only small LoRA adapters (rank 4, targets
# q,k,v,o,ffn.0,ffn.2) are injected on it. The action head (state_encoder/action_encoder/action_decoder)
# stays FULLY trainable (see wan_flow_matching_action_tf.py:385-387) so the policy still learns actions
# from scratch on top of an adapted, pretrained video model. save_lora_only=false -> self-contained
# checkpoints loadable by the existing GrootSimPolicy eval path (same as the full runs).
#
# Pretrained DiT load order is correct as-is: __init__ loads pretrained weights (line ~335) BEFORE LoRA
# injection (line ~378), so defer_lora_injection stays false (default). The frozen umt5-xxl T5 and
# Wan2.1 z=16 VAE are shared across Wan2.1 sizes, so they reuse the local Wan2.1-I2V-14B-480P copy.
# Single view => frame_seqlen = (256/8/2)*(256/8/2) = 256.
#
# Usage:
#   NUM_GPUS=4 bash scripts/train/pusht_training_1.3b_lora.sh
#   bash scripts/train/pusht_training_1.3b_lora.sh max_steps=2 report_to=none

export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export ATTENTION_BACKEND=${ATTENTION_BACKEND:-torch}
export TOKENIZERS_PARALLELISM=false

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DREAMZERO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$DREAMZERO_ROOT"

# ============ CONFIG ============
DATA_ROOT=${DATA_ROOT:-"$DREAMZERO_ROOT/data/pusht_lerobot"}
OUTPUT_DIR=${OUTPUT_DIR:-"$DREAMZERO_ROOT/checkpoints/dreamzero_pusht_1.3b_lora"}
WAN21_CKPT_DIR=${WAN21_CKPT_DIR:-"$DREAMZERO_ROOT/checkpoints/Wan2.1-I2V-14B-480P"}        # shared T5 + VAE
DIT_CKPT_DIR=${DIT_CKPT_DIR:-"$DREAMZERO_ROOT/checkpoints/Wan2.1-T2V-1.3B"}                # pretrained 1.3B DiT
TOKENIZER_DIR=${TOKENIZER_DIR:-"$WAN21_CKPT_DIR/google/umt5-xxl"}
NUM_GPUS=${NUM_GPUS:-4}
PER_DEVICE_BS=${PER_DEVICE_BS:-1}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-$((NUM_GPUS * PER_DEVICE_BS))}
MAX_STEPS=${MAX_STEPS:-60000}
SAVE_STEPS=${SAVE_STEPS:-2000}
LR=${LR:-1e-4}                # LoRA default (matches droid_training_lora.sh); adapters tolerate a higher LR than full-finetune's 1e-5
REPORT_TO=${REPORT_TO:-wandb}
DEEPSPEED_CFG=${DEEPSPEED_CFG:-zero2}
VENVPY="$DREAMZERO_ROOT/.venv/bin/python"
# ================================

if [ ! -d "$DATA_ROOT" ]; then echo "ERROR: PushT dataset not found at $DATA_ROOT"; exit 1; fi
if [ ! -d "$WAN21_CKPT_DIR" ]; then echo "ERROR: Wan2.1-I2V-14B-480P (frozen T5/VAE) not found at $WAN21_CKPT_DIR"; exit 1; fi
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
    train_architecture=lora \
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
    frame_seqlen=256 \
    fixed_num_chunks=${FIXED_NUM_CHUNKS:-null} \
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
    dit_version=$DIT_CKPT_DIR \
    text_encoder_pretrained_path=$WAN21_CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth \
    vae_pretrained_path=$WAN21_CKPT_DIR/Wan2.1_VAE.pth \
    tokenizer_path=$TOKENIZER_DIR \
    "$@"
