#!/bin/bash
# DreamZero (Wan2.2-TI2V-5B) training on RoboTwin: baseline vs RICL retrieved-frame grounding.
#
# Usage (submit both arms):
#   MODE=baseline  ROBOTWIN_ROOT=/path/to/robotwin sbatch external/dreamzero/scripts/train/robotwin_dreamzero.sbatch
#   MODE=treatment ROBOTWIN_ROOT=/path/to/robotwin K=2 X=4 sbatch external/dreamzero/scripts/train/robotwin_dreamzero.sbatch
#
# 5B full training needs >=80GB VRAM -> gpu-h200 / gpu-h100. Probe scheduling with:
#   sbatch --test-only --partition=gpu-h100 ... ; submit to the soonest torch-compatible one.
#SBATCH --job-name=rt_dz
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --partition=gpu-h200
#SBATCH --account=gts-dxu345-rl2
#SBATCH --gres=gpu:h200:1
#SBATCH --qos=inferno
#SBATCH --time=08:00:00
#SBATCH --output=external/dreamzero/outputs/rt_dz_%x_%j.out
#SBATCH --error=external/dreamzero/outputs/rt_dz_%x_%j.err

set -euo pipefail
export HYDRA_FULL_ERROR=1

# --- repo + env (egomimic must be importable: RoboTwinCorpus + retrieval live there) ---
REPO=/storage/project/r-dxu345-0/rco3/EgoVerse2
DZ=$REPO/external/dreamzero
# Use the env that has BOTH dreamzero deps and egomimic on the path. Override PY as needed.
PY=${PY:-$REPO/emimic/bin/python}
# _dzdeps = PYTHONPATH overlay with peft/accelerate/ftfy/regex/dm-tree/albumentations/sentencepiece
# (added to emimic without touching it; see RICL_README). Must precede so it's importable.
export PYTHONPATH="$DZ/_dzdeps:$DZ:$REPO:${PYTHONPATH:-}"
cd "$DZ"
mkdir -p "$DZ/outputs"

# --- args ---
MODE=${MODE:-baseline}            # baseline | treatment
ROBOTWIN_ROOT=${ROBOTWIN_ROOT:?set ROBOTWIN_ROOT to a RoboTwin task dir/parent}
K=${K:-2}; X=${X:-4}
NUM_GPUS=${NUM_GPUS:-1}
MAX_STEPS=${MAX_STEPS:-4000}
BS=${BS:-1}
LR=${LR:-1e-5}
EMBED=${EMBED:-dinov2}
OUTPUT_DIR=${OUTPUT_DIR:-$DZ/outputs/rt_dz_${MODE}}

# --- pretrained Wan2.2-5B base (NOT a DreamZero DROID/AgiBot checkpoint) ---
WAN22_CKPT_DIR=${WAN22_CKPT_DIR:-$DZ/checkpoints/Wan2.2-TI2V-5B}
IMAGE_ENCODER_DIR=${IMAGE_ENCODER_DIR:-$DZ/checkpoints/Wan2.1-I2V-14B-480P}
TOKENIZER_DIR=${TOKENIZER_DIR:-$DZ/checkpoints/umt5-xxl}

if [ "$MODE" = "treatment" ]; then
  RICL=(enable_retrieved_context=true num_retrieved_demos=$K frames_per_demo=$X)
else
  RICL=(enable_retrieved_context=false num_retrieved_demos=0)
fi

"$PY" -m torch.distributed.run --nproc_per_node "$NUM_GPUS" --standalone \
  groot/vla/experiment/train_robotwin_dreamzero.py \
  report_to=wandb wandb_project=dreamzero_robotwin \
  data=dreamzero/robotwin_wan22 \
  robotwin_root="$ROBOTWIN_ROOT" \
  embed=$EMBED \
  train_architecture=lora \
  num_frames=9 action_horizon=24 num_views=3 \
  model=dreamzero/vla \
  model/dreamzero/action_head=wan_flow_matching_action_tf_wan22_ricl \
  model/dreamzero/transform=dreamzero_cotrain \
  "${RICL[@]}" \
  num_frame_per_block=2 num_action_per_block=24 num_state_per_block=1 \
  max_chunk_size=8 \
  seed=42 \
  training_args.learning_rate=$LR \
  training_args.deepspeed=null \
  training_args.warmup_ratio=0.05 \
  save_steps=1000 save_total_limit=10 save_strategy=steps \
  output_dir="$OUTPUT_DIR" \
  per_device_train_batch_size=$BS \
  max_steps=$MAX_STEPS \
  weight_decay=1e-5 bf16=true tf32=true eval_bf16=true \
  dataloader_pin_memory=false dataloader_num_workers=2 \
  save_lora_only=true \
  pretrained_model_path=null \
  dit_version="$WAN22_CKPT_DIR" \
  text_encoder_pretrained_path="$WAN22_CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth" \
  image_encoder_pretrained_path="$IMAGE_ENCODER_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
  vae_pretrained_path="$WAN22_CKPT_DIR/Wan2.2_VAE.pth" \
  tokenizer_path="$TOKENIZER_DIR"
