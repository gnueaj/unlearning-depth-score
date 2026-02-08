#!/usr/bin/env bash
set -euo pipefail

EP5_DIR="runs/ep5"
EP10_DIR="runs/ep10"
PN_RESULTS="runs/faithfulness/results.json"
OUT_ROOT="runs/meta_eval/robustness_parallel"
LOG_DIR="logs/robustness_parallel"
MODEL_LIST="runs/meta_eval/robustness_filter_list/model_list_150.json"

mkdir -p "$OUT_ROOT" "$LOG_DIR"

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

# wait for gen_rouge completion
while true; do
  EP5_DONE=$(count_done "$EP5_DIR")
  EP10_DONE=$(count_done "$EP10_DIR")
  echo "[wait] ep5 ${EP5_DONE}/${EP5_TOTAL}, ep10 ${EP10_DONE}/${EP10_TOTAL}" | tee -a "$LOG_DIR/wait.log"
  if [[ "$EP5_DONE" -ge "$EP5_TOTAL" && "$EP10_DONE" -ge "$EP10_TOTAL" ]]; then
    break
  fi
  sleep 120
 done

# build filter list (uses gen_rouge + faithfulness P/N + utility)
python3 scripts/build_robustness_filter_list.py \
  --ep_dirs "$EP5_DIR,$EP10_DIR" \
  --pn_results "$PN_RESULTS" \
  --out_dir runs/meta_eval/robustness_filter_list \
  > "$LOG_DIR/build_filter.log" 2>&1

# build 150-model list (ep5+ep10)
python3 scripts/build_robustness_model_list.py \
  --ep_dirs "$EP5_DIR,$EP10_DIR" \
  --out "$MODEL_LIST" \
  > "$LOG_DIR/build_model_list.log" 2>&1

# compute split for robustness models (150 list)
TOTAL=$(python3 -c "import json; from pathlib import Path; ml=Path('runs/meta_eval/robustness_filter_list/model_list_150.json'); print(len(json.load(open(ml))))")
MID=$((TOTAL/2))

PART0="$OUT_ROOT/part0"
PART1="$OUT_ROOT/part1"
MERGED="$OUT_ROOT/merged"
mkdir -p "$PART0" "$PART1" "$MERGED"

# run two robustness jobs in parallel (GPU0 & GPU1)
python3 scripts/meta_eval_robustness.py \
  --gpu 0 --batch_size 32 \
  --metrics table2,uds \
  --filter_insufficient \
  --faithfulness_pn_results "$PN_RESULTS" \
  --filter_utility \
  --utility_epoch runs/ep5 \
  --models_file "$MODEL_LIST" \
  --model_start 0 --model_end "$MID" \
  --out_dir "$PART0" \
  > "$LOG_DIR/robustness_part0.log" 2>&1 &
PID0=$!

python3 scripts/meta_eval_robustness.py \
  --gpu 1 --batch_size 32 \
  --metrics table2,uds \
  --filter_insufficient \
  --faithfulness_pn_results "$PN_RESULTS" \
  --filter_utility \
  --utility_epoch runs/ep5 \
  --models_file "$MODEL_LIST" \
  --model_start "$MID" --model_end "$TOTAL" \
  --out_dir "$PART1" \
  > "$LOG_DIR/robustness_part1.log" 2>&1 &
PID1=$!

wait $PID0
wait $PID1

# merge
python3 scripts/merge_robustness_results.py \
  --part0 "$PART0" \
  --part1 "$PART1" \
  --out_dir "$MERGED" \
  --metrics table2,uds \
  --faithfulness_pn_results "$PN_RESULTS" \
  --filter_insufficient \
  --filter_utility \
  --utility_epoch runs/ep5 \
  --utility_drop 0.20 \
  > "$LOG_DIR/merge.log" 2>&1
