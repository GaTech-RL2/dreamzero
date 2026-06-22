#!/bin/bash
# Fire all 4 held-out-loss evals in parallel (separate a100 srun jobs):
#   Eval 1 (train_heldout): treatment + baseline on the TRAIN tasks (in-distribution).
#   Eval 2 (newtask_loo):   treatment + baseline on the 10 UNSEEN tasks (new-skill test).
# Each writes a JSON under external/dreamzero/outputs/eval_results/.
# Quantiles: both runs' quantiles.json are over the TRAIN root (identical data) -> train quantiles,
# correct for Eval 1 AND Eval 2 (new-task states normalized with TRAIN quantiles, by design).
set -uo pipefail
REPO=/storage/project/r-dxu345-0/rco3/EgoVerse2
cd "$REPO"
TR=$REPO/external/dreamzero/outputs/rt_dz_treatment
BL=$REPO/external/dreamzero/outputs/rt_dz_baseline
TRAIN=$REPO/egomimic/ricl/outputs/robotwin_10x10/train
EVAL=$REPO/egomimic/ricl/outputs/robotwin_10x10/eval
OUT=$REPO/external/dreamzero/outputs/eval_results
mkdir -p "$OUT"
PER_TASK=${PER_TASK:-25}

run() {  # name scenario mode ckpt root quantiles
  local name=$1 scen=$2 mode=$3 ckpt=$4 root=$5 q=$6
  srun --account=gts-dxu345-rl2 --qos=inferno --partition=gpu-a100 --gres=gpu:a100:1 \
    --cpus-per-task=4 --mem=100G -t 1:30:00 -J dz_$name bash -lc "
    cd $REPO && source emimic/bin/activate &&
    export PYTHONPATH=external/dreamzero/_dzdeps:external/dreamzero:. &&
    export TORCH_COMPILE_DISABLE=1 HF_HUB_OFFLINE=1 ATTENTION_BACKEND=FA2 &&
    python external/dreamzero/scripts/eval/heldout_loss.py \
      --checkpoint $ckpt --robotwin-root $root --quantiles $q \
      --scenario $scen --mode $mode --embed dinov2 --per-task $PER_TASK \
      --out $OUT/$name.json
  " > "$OUT/$name.log" 2>&1 &
  echo "launched $name (pid $!)"
}

run eval1_treatment train_heldout treatment "$TR" "$TRAIN" "$TR/quantiles.json"
run eval1_baseline  train_heldout baseline  "$BL" "$TRAIN" "$BL/quantiles.json"
run eval2_treatment newtask_loo   treatment "$TR" "$EVAL"  "$TR/quantiles.json"
run eval2_baseline  newtask_loo   baseline  "$BL" "$EVAL"  "$BL/quantiles.json"
wait
echo "ALL 4 EVALS DONE"
