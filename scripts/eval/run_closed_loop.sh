#!/bin/bash
# Closed-loop RoboTwin eval for DreamZero: dz_plain vs dz_ricl success rate over N episodes/task.
# For each task: (1) build a per-task DINOv2 bank index (for dz_ricl retrieval), (2) run
# record_rollout.py which steps the SAPIEN sim (get_obs -> policy.get_action -> take_action(qpos)
# -> eval_success) and writes an mp4 + per-episode success/fail.
# Usage: SPLIT=eval N=10 bash run_closed_loop.sh turn_switch place_shoe ...
# Env: SPLIT (train|eval, default eval), N (max episodes/model, default 10), TASK_CONFIG (demo_clean).
set -uo pipefail
REPO=/storage/project/r-dxu345-0/rco3/EgoVerse2
cd "$REPO"
source emimic/bin/activate
export PYTHONPATH=external/dreamzero/_dzdeps:external/dreamzero:.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TORCH_COMPILE_DISABLE=1
export PYTHONWARNINGS=ignore::UserWarning DISPLAY='' HF_HUB_OFFLINE=1 ATTENTION_BACKEND=FA2
module load cuda/12.6.1 2>/dev/null || true   # curobo runtime needs the CUDA toolkit on PATH

SPLIT=${SPLIT:-eval}
N=${N:-10}
export TASK_CONFIG=${TASK_CONFIG:-demo_clean}
DATA=$REPO/egomimic/ricl/outputs/robotwin_10x10/$SPLIT
IDXBASE=$REPO/external/dreamzero/outputs/cl_bank_index/$SPLIT
OUT=$REPO/external/dreamzero/outputs/eval_results/closed_loop/$SPLIT
mkdir -p "$IDXBASE" "$OUT"

export DZ_PLAIN_DIR=$REPO/external/dreamzero/outputs/rt_dz_baseline
export DZ_RICL_DIR=$REPO/external/dreamzero/outputs/rt_dz_treatment
export DZ_BANK_ROOT=$DATA
export DZ_BANK_INDEX=$IDXBASE

for T in "$@"; do
  echo "===== closed-loop task=$T split=$SPLIT N=$N ====="
  if [ ! -f "$IDXBASE/$T/manifest.json" ]; then
    echo "--- build DINOv2 bank index: $T ---"
    python egomimic/ricl/scripts/build_robotwin_bank_index.py \
      --root "$DATA/$T" --out "$IDXBASE/$T" --embed dinov2 \
      || { echo "[!] index build failed for $T; skipping dz_ricl"; }
  fi
  python egomimic/ricl/scripts/record_rollout.py \
    --task_name "$T" --task_config "$TASK_CONFIG" \
    --models dz_plain dz_ricl --max_episodes "$N" \
    --out_dir "$OUT/videos" 2>&1 | tee "$OUT/${T}.log" | grep -aE "====|ep[0-9]+ seed|SUCCESS|FAIL|wrote|SUMMARY|build FAILED"
done
echo "ALL CLOSED-LOOP DONE ($SPLIT): $*"
