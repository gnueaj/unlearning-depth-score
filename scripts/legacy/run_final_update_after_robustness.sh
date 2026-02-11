#!/usr/bin/env bash
set -euo pipefail

LOG_DIR="logs/final_update"
mkdir -p "$LOG_DIR"

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

EP5_TOTAL=$(count_models runs/ep5)
EP10_TOTAL=$(count_models runs/ep10)

while true; do
  EP5_DONE=$(count_done runs/ep5)
  EP10_DONE=$(count_done runs/ep10)
  echo "[wait] ep5 ${EP5_DONE}/${EP5_TOTAL}, ep10 ${EP10_DONE}/${EP10_TOTAL}" | tee -a "$LOG_DIR/wait.log"
  if [[ "$EP5_DONE" -ge "$EP5_TOTAL" && "$EP10_DONE" -ge "$EP10_TOTAL" ]]; then
    break
  fi
  sleep 120
 done

# Wait for robustness merged summary
ROBUST_SUM="runs/meta_eval/robustness_parallel/merged/summary.json"
while [[ ! -f "$ROBUST_SUM" ]]; do
  echo "[wait] robustness summary not found yet" | tee -a "$LOG_DIR/wait.log"
  sleep 120
 done

# Rebuild HTML/MD (legacy builder removed)
echo "[done] skipped HTML/MD rebuild (scripts/legacy/build_openunlearning_alpha_all.py removed)" | tee -a "$LOG_DIR/build_html.log"
