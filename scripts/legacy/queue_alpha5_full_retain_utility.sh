#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/jaeung/activation-patching-unlearning"
cd "$ROOT"

echo "[queue] waiting for ongoing evals to finish..."
while pgrep -f "patchscope\\.(memorization|privacy|utility)_eval" >/dev/null 2>&1; do
  sleep 60
done

export CUDA_VISIBLE_DEVICES=0
echo "[queue] starting alpha5 full/retain utility rerun..."
python3 -m patchscope.utility_eval \
  --model full \
  --batch_size 64 \
  --max_length 512 \
  --max_new_tokens 200 \
  --use_chat_template \
  --system_prompt "" \
  --out_dir runs/utility_eval/alpha5/full

python3 -m patchscope.utility_eval \
  --model retain \
  --batch_size 64 \
  --max_length 512 \
  --max_new_tokens 200 \
  --use_chat_template \
  --system_prompt "" \
  --out_dir runs/utility_eval/alpha5/retain

echo "[queue] done"
