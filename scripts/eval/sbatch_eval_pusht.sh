#!/bin/bash
#SBATCH --job-name=dz_pusht_eval
#SBATCH --partition=overcap
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out   # SLURM won't mkdir this; launch from repo root where logs/ exists
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

# Log eval/* to the training wandb run (same entity/project as sbatch_pusht.sh). WANDB_EVAL=0 disables.
export WANDB_ENTITY="${WANDB_ENTITY:-rl2-group}"
export WANDB_PROJECT="${WANDB_PROJECT:-world-value}"
WANDB_FLAG=""
[ "${WANDB_EVAL:-1}" = "1" ] && WANDB_FLAG="--wandb"
# Explicit run id overrides derive_wandb_run_id — needed when the training run used a wandb
# auto-id instead of the dzpusht_<size> scheme (e.g. the *_96 runs -> v49jbytw / 818ysd8k).
[ -n "${WANDB_RUN_ID:-}" ] && WANDB_FLAG="$WANDB_FLAG --wandb_run_id ${WANDB_RUN_ID}"
# Unique distributed port per job so two evals co-located on one node don't collide on the
# hardcoded 29577 (init_distributed uses os.environ.setdefault, so this env wins).
export MASTER_PORT="${MASTER_PORT:-$((20000 + ${SLURM_JOB_ID:-$$} % 40000))}"

.venv/bin/python scripts/eval/run_pusht_eval.py \
  --model_path "$CKPT" --num_episodes "$N" \
  --n_action_steps "${N_ACTION_STEPS:-8}" --max_steps "${MAX_STEPS:-300}" \
  --save_pred_video $WANDB_FLAG
