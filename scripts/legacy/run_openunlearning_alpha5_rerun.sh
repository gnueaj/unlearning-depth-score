#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/jaeung/activation-patching-unlearning"
cd "$ROOT"

OUT_MEM="runs/memorization_eval/alpha5_fix"
OUT_PRIV="runs/privacy_eval/alpha5_fix"
OUT_UTIL="runs/utility_eval/alpha5_fix"
CACHE_PATH="$OUT_PRIV/retain_full_cache.json"
BATCH_SIZE="${BATCH_SIZE:-64}"

mkdir -p "$OUT_MEM" "$OUT_PRIV" "$OUT_UTIL" logs

# Use existing alpha5 model list (32: 30 + full/retain)
mapfile -t MODELS < <(ls -1 "runs/memorization_eval/alpha5" | sort)

run_worker() {
  local gpu="$1"
  local start="$2"
  local step=2

  export CUDA_VISIBLE_DEVICES="$gpu"

  for ((i=start; i<${#MODELS[@]}; i+=step)); do
    m="${MODELS[$i]}"
    echo "[gpu${gpu}][mem] $m"
    python3 -m patchscope.memorization_eval \
      --hf_dataset locuslab/TOFU \
      --hf_config forget10_perturbed \
      --model "$m" \
      --batch_size "$BATCH_SIZE" \
      --max_length 512 \
      --use_chat_template \
      --system_prompt "" \
      --out_dir "$OUT_MEM/$m"

    echo "[gpu${gpu}][privacy] $m"
    python3 -m patchscope.privacy_eval \
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
      --out_dir "$OUT_PRIV/$m"

    echo "[gpu${gpu}][utility] $m"
    python3 -m patchscope.utility_eval \
      --model "$m" \
      --batch_size "$BATCH_SIZE" \
      --max_length 512 \
      --max_new_tokens 200 \
      --use_chat_template \
      --system_prompt "" \
      --out_dir "$OUT_UTIL/$m"
  done
}

run_worker 0 0 &
run_worker 1 1 &
wait

echo "[done] open-unlearning evals (alpha5_fix, 2 GPUs)"
