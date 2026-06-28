#!/bin/bash
#SBATCH --job-name=dz_pusht
#SBATCH --partition=overcap
#SBATCH --gres=gpu:a40:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --output=%x_%j.out
#
# Preemption-safe full training launcher for DreamZero PushT.
#   sbatch --job-name=dz_pusht_300m scripts/train/sbatch_pusht.sh 300m
#   sbatch --job-name=dz_pusht_1.3b scripts/train/sbatch_pusht.sh 1.3b
#
# overcap is preemptible; --requeue + the trainer's auto-resume (get_checkpoint_path on
# output_dir) means a requeued job continues from the latest checkpoint. Logs stream to the
# fixed training log (so monitoring survives requeues), in addition to the per-job %x_%j.out.

set -euo pipefail
SIZE="${1:?usage: sbatch scripts/train/sbatch_pusht.sh <300m|1.3b>}"
# Under sbatch, BASH_SOURCE points at the spool copy; use the submit dir (the repo root) instead.
DREAMZERO_ROOT="${DREAMZERO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
if [ ! -d "$DREAMZERO_ROOT/groot" ]; then
    echo "ERROR: $DREAMZERO_ROOT is not the dreamzero repo root (no groot/). Submit from the repo root."; exit 1
fi
cd "$DREAMZERO_ROOT"

case "$SIZE" in
  300m) TRAIN_SH="scripts/train/pusht_training_300m.sh";        OUT="$DREAMZERO_ROOT/checkpoints/dreamzero_pusht_300m" ;;
  1.3b) TRAIN_SH="scripts/train/pusht_training_wan21_t2v_1.3b.sh"; OUT="$DREAMZERO_ROOT/checkpoints/dreamzero_pusht_1.3b" ;;
  *) echo "unknown SIZE '$SIZE' (use 300m or 1.3b)"; exit 1 ;;
esac

NGPU="${SLURM_GPUS_ON_NODE:-4}"
OUT="${OUTPUT_DIR:-$OUT}"   # allow caller to override the output dir
mkdir -p "$OUT"
LOG="$OUT/train.log"
echo "=== $(date) launching $SIZE on $NGPU GPUs (job $SLURM_JOB_ID), out=$OUT, per_device_bs=${PER_DEVICE_BS:-1} global_bs=${GLOBAL_BATCH_SIZE:-64} ===" | tee -a "$LOG"

NUM_GPUS="$NGPU" \
OUTPUT_DIR="$OUT" \
PER_DEVICE_BS="${PER_DEVICE_BS:-1}" \
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}" \
MAX_STEPS="${MAX_STEPS:-60000}" \
SAVE_STEPS="${SAVE_STEPS:-2000}" \
REPORT_TO="${REPORT_TO:-none}" \
DEEPSPEED_CFG="${DEEPSPEED_CFG:-zero2_offload}" \
  bash "$TRAIN_SH" 2>&1 | tee -a "$LOG"
