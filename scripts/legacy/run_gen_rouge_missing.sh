#!/usr/bin/env bash
set -euo pipefail

EP=ep5
ROOT="runs/$EP"
LOG_DIR="logs/gen_rouge_missing"
mkdir -p "$LOG_DIR"

# Load model list and compute missing
python3 - <<'PY' > /tmp/missing_models.txt
import json
from pathlib import Path
models=json.load(open('runs/ep5/model_list.json'))
if isinstance(models, dict):
    models=models['ep5_models']
models=[m for m in models if m not in ('full','retain')]
root=Path('runs/ep5/gen_rouge')
existing=[p.name for p in root.iterdir() if p.is_dir()]
missing=[m for m in models if m not in existing]
print("\n".join(missing))
PY

TOTAL=$(wc -l < /tmp/missing_models.txt | tr -d ' ')
if [[ "$TOTAL" -eq 0 ]]; then
  echo "No missing models" | tee -a "$LOG_DIR/run.log"
  exit 0
fi

MID=$((TOTAL/2))

# GPU0 half
python3 scripts/compute_generation_metrics_all.py \
  --ep_dirs runs/ep5 \
  --gpu 0 --batch_size 32 \
  --models_file /tmp/missing_models.txt \
  --model_start 0 --model_end "$MID" \
  --purge_cache \
  > "$LOG_DIR/gpu0.log" 2>&1 &
PID0=$!

# GPU1 half
python3 scripts/compute_generation_metrics_all.py \
  --ep_dirs runs/ep5 \
  --gpu 1 --batch_size 32 \
  --models_file /tmp/missing_models.txt \
  --model_start "$MID" --model_end "$TOTAL" \
  --purge_cache \
  > "$LOG_DIR/gpu1.log" 2>&1 &
PID1=$!

wait $PID0
wait $PID1
