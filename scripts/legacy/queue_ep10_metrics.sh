#!/bin/bash
# Queue: Wait for faithfulness v3 -> Delete P/N cache -> Run EP10 metrics (priv/mem/uds/utility)
# Usage: nohup bash scripts/queue_ep10_metrics.sh > logs/queue_ep10.log 2>&1 &

set -e
cd /home/jaeung/activation-patching-unlearning

echo "$(date) === Queue EP10 Metrics Started ==="
echo "Waiting for Faithfulness v3 to complete..."

# ============================================================
# Wait for Faithfulness v3 to complete (30 models each GPU)
# ============================================================
while true; do
  g0=$(python3 -c "import json; print(len(json.load(open('runs/meta_eval/table2_faithfulness_v3_gpu0/results.json'))))" 2>/dev/null || echo 0)
  g1=$(python3 -c "import json; print(len(json.load(open('runs/meta_eval/table2_faithfulness_v3_gpu1/results.json'))))" 2>/dev/null || echo 0)
  echo "$(date '+%H:%M:%S') Faithfulness v3: GPU0=$g0/30 GPU1=$g1/30"

  if [ "$g0" = "30" ] && [ "$g1" = "30" ]; then
    echo "$(date) Faithfulness v3 complete!"
    break
  fi
  sleep 60
done

sleep 10

# ============================================================
# Delete P/N pool model caches (60 models)
# ============================================================
echo ""
echo "$(date) Deleting P/N pool model caches..."

python3 << 'EOF'
import shutil
from pathlib import Path

cache_dir = Path.home() / ".cache" / "huggingface" / "hub"

# P/N pool patterns (epoch5 and epoch10)
patterns = ["pos_tofu_", "neg_tofu_"]

deleted = 0
for model_dir in cache_dir.glob("models--open-unlearning--*"):
    name = model_dir.name
    if any(p in name for p in patterns):
        try:
            shutil.rmtree(model_dir, ignore_errors=True)
            deleted += 1
            print(f"  Deleted: {name}")
        except:
            pass

print(f"\nDeleted {deleted} P/N pool model caches")
EOF

# ============================================================
# Run EP10 metrics: priv/mem/uds/utility on 75 unlearn models
# ============================================================
echo ""
echo "$(date) Starting EP10 metrics evaluation..."

# Split 75 models: GPU0 gets 0-37, GPU1 gets 38-74
# Output goes to fixed dirs: runs/ep10/{memorization,privacy,utility,uds}/
python3 scripts/run_ep10_experiments.py --gpu 0 --start 0 --end 38 \
  > logs/ep10_metrics_gpu0.log 2>&1 &
PID_G0=$!

python3 scripts/run_ep10_experiments.py --gpu 1 --start 38 --end 75 \
  > logs/ep10_metrics_gpu1.log 2>&1 &
PID_G1=$!

echo "EP10 PIDs: GPU0=$PID_G0, GPU1=$PID_G1"

# Wait for completion
wait $PID_G0 || echo "WARNING: GPU0 EP10 error"
wait $PID_G1 || echo "WARNING: GPU1 EP10 error"

echo ""
echo "$(date) === All EP10 metrics complete! ==="
