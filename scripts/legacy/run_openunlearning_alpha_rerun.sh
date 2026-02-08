#!/usr/bin/env bash
set -euo pipefail

# Usage: ./scripts/run_openunlearning_alpha_rerun.sh alpha1
#   alpha1 | alpha2 | alpha5

if [ $# -lt 1 ]; then
  echo "Usage: $0 <alpha1|alpha2|alpha5>"
  exit 1
fi

ALPHA_TAG="$1"

ROOT="/home/jaeung/activation-patching-unlearning"
cd "$ROOT"

OUT_MEM="runs/memorization_eval/${ALPHA_TAG}_fix"
OUT_PRIV="runs/privacy_eval/${ALPHA_TAG}_fix"
OUT_UTIL="runs/utility_eval/${ALPHA_TAG}_fix"
CACHE_PATH="$OUT_PRIV/retain_full_cache.json"
BATCH_SIZE="${BATCH_SIZE:-64}"

mkdir -p "$OUT_MEM" "$OUT_PRIV" "$OUT_UTIL" logs

# Build model list by mapping alpha5 list -> alpha1/alpha2
mapfile -t MODELS < <(python3 - <<PY
from pathlib import Path
alpha = "${ALPHA_TAG}".replace("alpha", "")
models = sorted([p.name for p in Path("runs/memorization_eval/alpha5").iterdir() if p.is_dir()])
out=[]
for m in models:
    if m in ("full","retain"):
        out.append(m); continue
    if "_a5_" in m:
        out.append(m.replace("_a5_", f"_a{alpha}_"))
    else:
        out.append(m)
print("\\n".join(out))
PY)

run_worker() {
  local gpu="$1"
  local start="$2"
  local step=2

  export CUDA_VISIBLE_DEVICES="$gpu"

  for ((i=start; i<${#MODELS[@]}; i+=step)); do
    m="${MODELS[$i]}"
    echo "[${ALPHA_TAG}][gpu${gpu}][mem] $m"
    python3 -m patchscope.memorization_eval \
      --hf_dataset locuslab/TOFU \
      --hf_config forget10_perturbed \
      --model "$m" \
      --batch_size "$BATCH_SIZE" \
      --max_length 512 \
      --use_chat_template \
      --system_prompt "" \
      --out_dir "$OUT_MEM/$m"

    echo "[${ALPHA_TAG}][gpu${gpu}][privacy] $m"
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

    echo "[${ALPHA_TAG}][gpu${gpu}][utility] $m"
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

echo "[done] open-unlearning evals (${ALPHA_TAG}_fix, 2 GPUs)"
