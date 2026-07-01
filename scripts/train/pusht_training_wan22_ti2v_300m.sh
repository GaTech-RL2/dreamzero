#!/bin/bash
# DreamZero PushT training -- ~300M text+image-to-video (ti2v) DiT on the Wan2.2 VAE, FROM SCRATCH.
#
# Sibling of pusht_training_300m.sh, but built on the Wan2.2-TI2V stack instead of Wan2.1 i2v:
#   - VAE: WanVideoVAE38 (z=48, 16x spatial) loaded from Wan2.2-TI2V-5B/Wan2.2_VAE.pth
#   - DiT: model_type=ti2v, in_dim=out_dim=48, concat_first_frame_latent=false (config
#     wan_flow_matching_action_tf_wan22_300m); first frame conditions via CLIP + in-context latent.
# Single top-down RGB view kept at NATIVE 96x96 (no upsampling), 2D absolute target-position action.
# The ~300M DiT trains from random init (the wan22_300m config sets skip_component_loading=true,
# so only the frozen T5 + CLIP + Wan2.2 VAE load; T5/CLIP/tokenizer reuse the Wan2.1-I2V-14B-480P
# checkpoint -- the Wan2.2-TI2V-5B dir on disk holds only the VAE).
# Single view, Wan2.2 16x VAE + DiT stride-2 patch => effective stride 32:
#   frame_seqlen = (96/16/2)*(96/16/2) = 3*3 = 9  (native 96x96). Latent 6x6 (even).
#   (256x256 would give 8*8 = 64; we use native 96 -- VAE recon is ~equal vs the native frame.)
#
# Usage:
#   NUM_GPUS=4 bash scripts/train/pusht_training_wan22_ti2v_300m.sh
#   # extra hydra overrides pass through, e.g.:  bash scripts/train/pusht_training_wan22_ti2v_300m.sh max_steps=2 report_to=none

export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export ATTENTION_BACKEND=${ATTENTION_BACKEND:-torch}   # SDPA self-attn (no flash_attn needed)
export TOKENIZERS_PARALLELISM=false

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DREAMZERO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$DREAMZERO_ROOT"

# ============ CONFIG ============
DATA_ROOT=${DATA_ROOT:-"$DREAMZERO_ROOT/data/pusht_lerobot"}
OUTPUT_DIR=${OUTPUT_DIR:-"$DREAMZERO_ROOT/checkpoints/dreamzero_pusht_300m_wan22"}
WAN22_CKPT_DIR=${WAN22_CKPT_DIR:-"$DREAMZERO_ROOT/checkpoints/Wan2.2-TI2V-5B"}    # Wan2.2 VAE
WAN21_CKPT_DIR=${WAN21_CKPT_DIR:-"$DREAMZERO_ROOT/checkpoints/Wan2.1-I2V-14B-480P"} # frozen T5 + CLIP + tokenizer
TOKENIZER_DIR=${TOKENIZER_DIR:-"$WAN21_CKPT_DIR/google/umt5-xxl"}
NUM_GPUS=${NUM_GPUS:-4}
PER_DEVICE_BS=${PER_DEVICE_BS:-1}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-32}
MAX_STEPS=${MAX_STEPS:-60000}
SAVE_STEPS=${SAVE_STEPS:-2000}
EVAL_STRATEGY=${EVAL_STRATEGY:-steps}   # held-out val-loss on episodes [val_episode_start, num_episodes); set "no" to disable
EVAL_STEPS=${EVAL_STEPS:-2000}
LR=${LR:-1e-4}
REPORT_TO=${REPORT_TO:-wandb}
DEEPSPEED_CFG=${DEEPSPEED_CFG:-zero2}   # zero2 (NOT zero2_offload: offload needs a node CUDA toolchain it lacks)
VENVPY="$DREAMZERO_ROOT/.venv/bin/python"
# ================================

if [ ! -d "$DATA_ROOT" ]; then echo "ERROR: PushT dataset not found at $DATA_ROOT (run scripts/data/convert_pusht_to_lerobot.py)"; exit 1; fi
if [ ! -f "$WAN22_CKPT_DIR/Wan2.2_VAE.pth" ]; then echo "ERROR: Wan2.2 VAE not found at $WAN22_CKPT_DIR/Wan2.2_VAE.pth (huggingface-cli download Wan-AI/Wan2.2-TI2V-5B --include Wan2.2_VAE.pth --local-dir $WAN22_CKPT_DIR)"; exit 1; fi
if [ ! -d "$WAN21_CKPT_DIR" ]; then echo "ERROR: Wan2.1-I2V-14B-480P (frozen T5/CLIP/tokenizer) not found at $WAN21_CKPT_DIR"; exit 1; fi

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
    model/dreamzero/action_head=wan_flow_matching_action_tf_wan22_300m \
    model/dreamzero/transform=dreamzero_cotrain \
    num_frame_per_block=2 \
    num_action_per_block=24 \
    num_state_per_block=1 \
    frame_seqlen=9 \
    fixed_num_chunks=${FIXED_NUM_CHUNKS:-null} \
    image_resolution_width=96 \
    image_resolution_height=96 \
    seed=42 \
    training_args.learning_rate=$LR \
    training_args.deepspeed="groot/vla/configs/deepspeed/${DEEPSPEED_CFG}.json" \
    save_steps=$SAVE_STEPS \
    do_eval=true \
    eval_strategy=$EVAL_STRATEGY \
    eval_steps=$EVAL_STEPS \
    per_device_eval_batch_size=1 \
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
    dit_version=$WAN22_CKPT_DIR \
    text_encoder_pretrained_path=$WAN21_CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth \
    image_encoder_pretrained_path=$WAN21_CKPT_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
    vae_pretrained_path=$WAN22_CKPT_DIR/Wan2.2_VAE.pth \
    tokenizer_path=$TOKENIZER_DIR \
    "$@"
