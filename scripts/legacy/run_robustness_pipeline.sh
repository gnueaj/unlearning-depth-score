#!/usr/bin/env bash
set -euo pipefail

# Wait for gen_rouge summaries for ep5/ep10 to finish, then build filter list and run robustness.
# Uses GPU0 by default.

EP5_DIR="runs/ep5"
EP10_DIR="runs/ep10"
PN_RESULTS="runs/meta_eval/combined_faithfulness_v3/results.json"
OUT_DIR="runs/meta_eval/robustness_pipeline"
LOG_DIR="logs/robustness_pipeline"

mkdir -p "$OUT_DIR" "$LOG_DIR"

count_models() {
  local ep_dir="$1"
  python3 - "$ep_dir" <<'PY'
import json
from pathlib import Path
import sys

ep_dir = Path(sys.argv[1])
model_list = ep_dir / 'model_list.json'
if not model_list.exists():
    print(0)
    raise SystemExit

data = json.load(open(model_list))
if isinstance(data, dict):
    key = 'ep5_models' if 'ep5' in ep_dir.name else 'ep10_models'
    models = data.get(key, [])
else:
    models = data
models = [m for m in models if m not in ('full','retain')]
print(len(models))
PY
}

count_done() {
  local ep_dir="$1"
  python3 - "$ep_dir" <<'PY'
from pathlib import Path
import sys

ep_dir = Path(sys.argv[1])
root = ep_dir / 'gen_rouge'
if not root.exists():
    print(0)
    raise SystemExit
count = 0
for p in root.iterdir():
    if p.is_dir() and (p / 'summary.json').exists():
        count += 1
print(count)
PY
}

EP5_TOTAL=$(count_models "$EP5_DIR")
EP10_TOTAL=$(count_models "$EP10_DIR")

# Wait loop
while true; do
  EP5_DONE=$(count_done "$EP5_DIR")
  EP10_DONE=$(count_done "$EP10_DIR")
  echo "[wait] ep5 ${EP5_DONE}/${EP5_TOTAL}, ep10 ${EP10_DONE}/${EP10_TOTAL}" | tee -a "$LOG_DIR/wait.log"
  if [[ "$EP5_DONE" -ge "$EP5_TOTAL" && "$EP10_DONE" -ge "$EP10_TOTAL" ]]; then
    break
  fi
  sleep 120
 done

# Build filter list
python3 scripts/build_robustness_filter_list.py \
  --ep_dirs "$EP5_DIR,$EP10_DIR" \
  --pos_results runs/meta_eval/table2_faithfulness_v3_gpu0/results.json \
  --neg_results runs/meta_eval/table2_faithfulness_v3_gpu1/results.json \
  --combined_out "$PN_RESULTS" \
  --out_dir runs/meta_eval/robustness_filter_list \
  > "$LOG_DIR/build_filter.log" 2>&1

# Run robustness (metric filtering inside)
python3 scripts/meta_eval_robustness.py \
  --gpu 0 --batch_size 32 \
  --metrics table2,uds \
  --filter_insufficient \
  --faithfulness_pn_results "$PN_RESULTS" \
  --filter_utility \
  --utility_epoch runs/ep5 \
  --out_dir "$OUT_DIR" \
  > "$LOG_DIR/robustness.log" 2>&1
