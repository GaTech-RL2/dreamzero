#!/bin/bash
# Auto-eval watcher: submits a PushT eval for each new checkpoint (every INTERVAL_STEPS steps) of
# both models, then aggregates completed summary.json files into a single curve TSV.
# Run on the login node in the background; tail its stdout for submissions + results.
# Repo root, derived from this script's location (scripts/eval/auto_eval_watch.sh).
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
INTERVAL_STEPS=${INTERVAL_STEPS:-2000}   # eval every checkpoint = continuous eval during training
N_EP=${N_EP:-25}
# Which model output dirs to watch. Default = all known PushT runs (standalone login-node use);
# sbatch_pusht.sh overrides MODELS to just the run it launched so each training job scopes its own watcher.
MODELS=${MODELS:-"dreamzero_pusht_300m dreamzero_pusht_1.3b dreamzero_pusht_1.3b_lora dreamzero_pusht_5b_lora dreamzero_pusht_1.3b_ft"}
# Per-model wandb "project run_id". Runs that used a wandb auto-id (not the dzpusht_<size>
# scheme, e.g. the *_96 runs) must be listed here or eval logs to an orphan run. Models absent
# from this map fall back to the eval script's defaults (world-value + derive_wandb_run_id).
declare -A RUNS=(
  [dreamzero_pusht_1p3b_wan21_t2v_ft_96]="dreamzero_pusht v49jbytw"
  [dreamzero_pusht_300m_wan22_96]="dreamzero_pusht 818ysd8k"
)
# Per-tag submit-marker file so concurrent per-job watchers don't fight over one state file
# (STATE_TAG=all for the shared login-node watcher). The curve TSV stays shared: one curve, all models.
STATE="$ROOT/checkpoints/.autoeval_state_${STATE_TAG:-all}"
CURVE="$ROOT/checkpoints/pusht_curve.tsv"
touch "$STATE"
[ -f "$CURVE" ] || printf "model\tstep\tsuccess_rate\tmean_score\tmean_coverage\n" > "$CURVE"

while true; do
  for model in $MODELS; do
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
      # Record the marker ONLY on a successful submit (non-empty job id). Recording before submit means
      # a transient sbatch failure (e.g. unresponsive SLURM controller -> empty job id) is never retried.
      # Pass the correct wandb project + run id for this model (empty for unmapped models).
      read -r WPROJ WRID <<<"${RUNS[$model]:-}"
      # EXCLUDE_NODES: overcap nodes with flaky GPUs where torch.cuda.is_available()==False even
      # after a gres/gpu:a40 alloc (e.g. ig-88). A submitted job that dies there is NOT retried
      # (marker is set on submit), so keep known-bad nodes out. Append new offenders as found.
      jid=$(WANDB_PROJECT="${WPROJ:-}" WANDB_RUN_ID="${WRID:-}" \
            sbatch --parsable --exclude="${EXCLUDE_NODES:-sonny,ig-88}" scripts/eval/sbatch_eval_pusht.sh "$ckpt" "$N_EP" 2>/dev/null || true)
      if [ -n "$jid" ]; then
        echo "$key" >> "$STATE"
        echo "[autoeval] $(date +%H:%M) submitted $model step=$step job=$jid"
      else
        echo "[autoeval] $(date +%H:%M) SUBMIT FAILED $model step=$step (SLURM busy?) -- will retry next loop"
      fi
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
