#!/usr/bin/env bash
set -euo pipefail
BATCH_SIZE=${BATCH_SIZE:-64}
ALPHAS=(alpha1_fix alpha2_fix alpha5_fix)
for alpha in "${ALPHAS[@]}"; do
  OUT_PRIV="runs/privacy_eval/${alpha}"
  CACHE_PATH="$OUT_PRIV/retain_full_cache_v2.json"
  mkdir -p "$OUT_PRIV" logs
  LOG="logs/privacy_rerun_${alpha}_v2.log"
  echo "[start] $alpha" | tee -a "$LOG"
  for m in $(ls -1 "runs/memorization_eval/${alpha}" | sort); do
    if [ ! -d "runs/memorization_eval/${alpha}/$m" ]; then
      continue
    fi
    echo "[$alpha] $m" | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES=0 python3 -m patchscope.privacy_eval \
      --model "$m" \
      --reference_model retain \
      --full_model full \
      --attack all \
      --retain_full_cache "$CACHE_PATH" \
      --k 0.4 \
      --batch_size "$BATCH_SIZE" \
      --max_length 512 \
      --use_chat_template \
      --system_prompt "" \
      --out_dir "$OUT_PRIV/$m" \
      >> "$LOG" 2>&1
  done
  echo "[done] $alpha" | tee -a "$LOG"
done
