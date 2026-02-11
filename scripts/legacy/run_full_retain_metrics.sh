#!/bin/bash
# FULL and RETAIN metrics (MEM/PRIV/UTIL only, no UDS)
# Runs after EP10 completes

cd /home/jaeung/activation-patching-unlearning

echo "=== FULL/RETAIN Metrics ==="
echo "Started: $(date)"

# Wait for EP10 to complete
while pgrep -f "run_ep10_all_metrics" > /dev/null; do
    echo "Waiting for EP10 to complete... ($(date +%H:%M))"
    sleep 60
done

echo "EP10 completed. Starting FULL/RETAIN metrics..."

export CUDA_VISIBLE_DEVICES=0

# FULL model
echo ""
echo "=== FULL Model ==="
MODEL="full"
OUT_BASE="runs/ep10"

echo "[MEM] Running..."
python -m uds.memorization_eval \
    --model "$MODEL" \
    --hf_dataset locuslab/TOFU \
    --hf_config forget10_perturbed \
    --use_chat_template \
    --batch_size 32 \
    --out_dir "$OUT_BASE/memorization/$MODEL" 2>&1 | tail -3

echo "[PRIV] Running..."
python -m uds.privacy_eval \
    --model "$MODEL" \
    --use_chat_template \
    --batch_size 32 \
    --out_dir "$OUT_BASE/privacy/$MODEL" 2>&1 | tail -3

echo "[UTIL] Running..."
python -m uds.utility_eval \
    --model "$MODEL" \
    --use_chat_template \
    --batch_size 32 \
    --out_dir "$OUT_BASE/utility/$MODEL" 2>&1 | tail -3

# RETAIN model
echo ""
echo "=== RETAIN Model ==="
MODEL="retain"

echo "[MEM] Running..."
python -m uds.memorization_eval \
    --model "$MODEL" \
    --hf_dataset locuslab/TOFU \
    --hf_config forget10_perturbed \
    --use_chat_template \
    --batch_size 32 \
    --out_dir "$OUT_BASE/memorization/$MODEL" 2>&1 | tail -3

echo "[PRIV] Running..."
python -m uds.privacy_eval \
    --model "$MODEL" \
    --use_chat_template \
    --batch_size 32 \
    --out_dir "$OUT_BASE/privacy/$MODEL" 2>&1 | tail -3

echo "[UTIL] Running..."
python -m uds.utility_eval \
    --model "$MODEL" \
    --use_chat_template \
    --batch_size 32 \
    --out_dir "$OUT_BASE/utility/$MODEL" 2>&1 | tail -3

echo ""
echo "=== COMPLETED ==="
echo "Finished: $(date)"
