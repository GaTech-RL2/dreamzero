#!/usr/bin/env bash
# Convert a collected DP-rollout PushT zarr into a DreamZero/GEAR LeRobot dataset.
# Runs BOTH converter stages under the dreamzero venv (which has pyarrow).
# Writes to a NEW dataset dir -- never touches the original pusht_lerobot.
#
# Usage (from repo root or anywhere):
#   bash baselines/dreamzero/scripts/data/convert_dp_demos.sh \
#       [ZARR=baselines/diffusion_policy/data/pusht_dp_demos.zarr] \
#       [OUT=baselines/dreamzero/data/pusht_dp_lerobot] [FPS=10]
set -euo pipefail

DZ_ROOT="/coc/flash7/rco3/world-value/baselines/dreamzero"
PY="$DZ_ROOT/.venv/bin/python"
ZARR="${1:-/coc/flash7/rco3/world-value/baselines/diffusion_policy/data/pusht_dp_demos.zarr}"
OUT="${2:-$DZ_ROOT/data/pusht_dp_lerobot}"
FPS="${3:-10}"

echo "[convert] zarr   = $ZARR"
echo "[convert] output = $OUT  (separate from data/pusht_lerobot)"

if [ -e "$OUT" ]; then
  echo "[convert] ERROR: $OUT already exists; remove it or pass a different OUT path." >&2
  exit 1
fi

# Stage 1: zarr -> LeRobot v2.0 (parquet + mp4 + meta)
"$PY" "$DZ_ROOT/scripts/data/convert_pusht_to_lerobot.py" \
    --zarr-path "$ZARR" --output-path "$OUT" --fps "$FPS"

# Stage 2: build modality.json + stats.json in-place (embodiment-tag pusht)
"$PY" "$DZ_ROOT/scripts/data/convert_lerobot_to_gear.py" \
    --dataset-path "$OUT" --embodiment-tag pusht \
    --state-keys '{"agent_pos":[0,2]}' --action-keys '{"target_pos":[0,2]}' \
    --task-key annotation.language.action_text --fps "$FPS"

echo "[convert] DONE -> $OUT"
"$PY" - "$OUT" <<'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
info = json.loads((out / "meta" / "info.json").read_text())
print(f"[convert] episodes={info['total_episodes']} frames={info['total_frames']} fps={info['fps']}")
print(f"[convert] modality.json: {(out/'meta'/'modality.json').exists()}  stats.json: {(out/'meta'/'stats.json').exists()}")
PY
