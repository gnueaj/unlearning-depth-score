#!/bin/bash
# Queue EP10 experiments after robustness v2 completes
# This script waits for robustness processes to finish, then starts EP10

cd /home/jaeung/activation-patching-unlearning

echo "=============================================="
echo "EP10 Queue Script"
echo "Waiting for robustness v2 to complete..."
echo "=============================================="

# Wait for robustness processes to finish
while pgrep -f "meta_eval_robustness" > /dev/null; do
    # Show progress
    GPU0=$(grep -E "^\[[0-9]+/37\]" logs/robustness_v2_filtered_gpu0.log 2>/dev/null | tail -n 1 | grep -oE "\[[0-9]+/37\]" || echo "[?/37]")
    GPU1=$(grep -E "^\[[0-9]+/38\]" logs/robustness_v2_filtered_gpu1.log 2>/dev/null | tail -n 1 | grep -oE "\[[0-9]+/38\]" || echo "[?/38]")
    echo "$(date '+%H:%M:%S') - Robustness: GPU0=$GPU0, GPU1=$GPU1"
    sleep 300  # Check every 5 minutes
done

echo ""
echo "=============================================="
echo "Robustness completed! Starting EP10..."
echo "Time: $(date)"
echo "=============================================="

# Merge robustness results first
echo "Merging robustness results..."
python3 << 'EOF'
import json
from pathlib import Path

# Merge GPU0 and GPU1 results
results = {}
for gpu in [0, 1]:
    f = Path(f'runs/meta_eval/table2_robustness_v2_filtered_gpu{gpu}/results.json')
    if f.exists():
        data = json.load(open(f))
        results.update(data)

out_file = Path('runs/meta_eval/table2_robustness_v2_filtered/results.json')
out_file.parent.mkdir(parents=True, exist_ok=True)
with open(out_file, 'w') as f:
    json.dump(results, f, indent=2)
print(f"Merged {len(results)} models to {out_file}")
EOF

# Start EP10 experiments
echo ""
echo "Starting EP10 experiments on both GPUs..."

mkdir -p logs/ep10

nohup bash scripts/run_ep10_gpu0.sh > logs/ep10/gpu0.log 2>&1 &
PID0=$!
echo "Started GPU0: PID=$PID0"

nohup bash scripts/run_ep10_gpu1.sh > logs/ep10/gpu1.log 2>&1 &
PID1=$!
echo "Started GPU1: PID=$PID1"

echo ""
echo "EP10 experiments queued!"
echo "Monitor with:"
echo "  tail -f logs/ep10/gpu0.log"
echo "  tail -f logs/ep10/gpu1.log"
