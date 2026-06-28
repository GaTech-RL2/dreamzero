#!/bin/bash
# Auto-eval watcher: submits a PushT eval for each new checkpoint (every INTERVAL_STEPS steps) of
# both models, then aggregates completed summary.json files into a single curve TSV.
# Run on the login node in the background; tail its stdout for submissions + results.
# Repo root, derived from this script's location (scripts/eval/auto_eval_watch.sh).
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
INTERVAL_STEPS=${INTERVAL_STEPS:-2000}   # eval every checkpoint = continuous eval during training
N_EP=${N_EP:-25}
STATE="$ROOT/checkpoints/.autoeval_state"
CURVE="$ROOT/checkpoints/pusht_curve.tsv"
touch "$STATE"
[ -f "$CURVE" ] || printf "model\tstep\tsuccess_rate\tmean_score\tmean_coverage\n" > "$CURVE"

while true; do
  for model in dreamzero_pusht_300m dreamzero_pusht_1.3b; do
    d="$ROOT/checkpoints/$model"
    [ -d "$d" ] || continue
    for ckpt in "$d"/checkpoint-*; do
      [ -d "$ckpt" ] || continue
      step=$(basename "$ckpt" | sed 's/checkpoint-//')
      [ "$step" -eq "$step" ] 2>/dev/null || continue
      [ $((step % INTERVAL_STEPS)) -eq 0 ] || continue
      # Require the checkpoint to be fully finalized: weights + the trainer's experiment_cfg copy
      # (metadata.json is the last file CheckpointFormatCallback writes). Submitting earlier would
      # make the eval try to create experiment_cfg itself, racing the trainer's copytree.
      [ -f "$ckpt/config.json" ] || continue
      [ -f "$ckpt/experiment_cfg/metadata.json" ] || continue
      key="sub:$model:$step"
      grep -qF "$key" "$STATE" && continue
      echo "$key" >> "$STATE"
      jid=$(sbatch --parsable --exclude=sonny scripts/eval/sbatch_eval_pusht.sh "$ckpt" "$N_EP" 2>/dev/null || true)
      echo "[autoeval] $(date +%H:%M) submitted $model step=$step job=$jid"
    done
  done
  # Aggregate completed summaries into the curve (once each).
  for s in "$ROOT"/checkpoints/dreamzero_pusht_*/checkpoint-*/pusht_eval/summary.json; do
    [ -f "$s" ] || continue
    model=$(echo "$s" | sed -E 's#.*/checkpoints/([^/]+)/.*#\1#')
    step=$(echo "$s" | sed -E 's#.*/checkpoint-([0-9]+)/.*#\1#')
    key="curve:$model:$step"
    grep -qF "$key" "$STATE" && continue
    vals=$(python3 -c "import json;d=json.load(open('$s'));print(d['success_rate'],d['mean_score'],d['mean_max_coverage'])" 2>/dev/null || true)
    [ -n "$vals" ] || continue
    echo "$key" >> "$STATE"
    printf "%s\t%s\t%s\t%s\t%s\n" "$model" "$step" $vals >> "$CURVE"
    echo "[autoeval] $(date +%H:%M) RESULT $model step=$step -> success_rate=$(echo $vals|cut -d' ' -f1) cov=$(echo $vals|cut -d' ' -f3)"
  done
  sleep 300
done
