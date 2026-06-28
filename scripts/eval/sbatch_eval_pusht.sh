#!/bin/bash
#SBATCH --job-name=dz_pusht_eval
#SBATCH --partition=overcap
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
#
# Evaluate a DreamZero PushT checkpoint on gym-pusht (coverage success + env/pred mp4s).
#   sbatch scripts/eval/sbatch_eval_pusht.sh <checkpoint_dir> [num_episodes]
# If <checkpoint_dir> is a training output dir (no config.json), the latest checkpoint-* is used.

set -euo pipefail
CKPT="${1:?usage: sbatch scripts/eval/sbatch_eval_pusht.sh <checkpoint_dir> [num_episodes]}"
N="${2:-50}"
# Under sbatch, BASH_SOURCE points at the spool copy; use the submit dir (the repo root).
DREAMZERO_ROOT="${DREAMZERO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$DREAMZERO_ROOT"

# If given an output dir, pick the highest-step checkpoint-*.
if [ ! -f "$CKPT/config.json" ]; then
  LATEST="$(ls -d "$CKPT"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1 || true)"
  [ -n "$LATEST" ] && CKPT="$LATEST"
fi
echo "=== evaluating $CKPT ($N episodes) ==="

.venv/bin/python scripts/eval/run_pusht_eval.py \
  --model_path "$CKPT" --num_episodes "$N" \
  --n_action_steps "${N_ACTION_STEPS:-8}" --max_steps "${MAX_STEPS:-300}" \
  --save_pred_video
