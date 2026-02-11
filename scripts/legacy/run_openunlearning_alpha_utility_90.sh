#!/usr/bin/env bash
set -euo pipefail

# Run Open-Unlearning utility eval for 30 unlearned models per alpha (alpha1/2/5),
# excluding full/retain and skipping simnpo/rmu for alpha2/5.
#
# Usage: ./scripts/run_openunlearning_alpha_utility_90.sh

ROOT="/home/jaeung/activation-patching-unlearning"
cd "$ROOT"

ALPHAS=("alpha1" "alpha2" "alpha5")
OUT_ROOT="runs/utility_eval"
BATCH_SIZE="${BATCH_SIZE:-64}"

build_models() {
  local alpha="$1"
  python3 - <<PY
from pathlib import Path
alpha = "${alpha}".replace("alpha", "")
models = sorted([p.name for p in Path("runs/memorization_eval/alpha5").iterdir() if p.is_dir()])
out = []
for m in models:
    if m in ("full", "retain"):
        continue
    if m.startswith(("simnpo_", "rmu_")) and alpha != "1":
        continue
    if "_a5_" in m:
        out.append(m.replace("_a5_", f"_a{alpha}_"))
    else:
        out.append(m)
print("\\n".join(out))
PY
}

for alpha in "${ALPHAS[@]}"; do
  OUT_DIR="${OUT_ROOT}/${alpha}_fix"
  mkdir -p "$OUT_DIR"

  mapfile -t MODELS < <(build_models "$alpha")
  echo "[alpha=${alpha}] models=${#MODELS[@]}"

  for m in "${MODELS[@]}"; do
    echo "[${alpha}] utility $m"
    CUDA_VISIBLE_DEVICES=0 python3 -m uds.utility_eval \
      --model "$m" \
      --batch_size "$BATCH_SIZE" \
      --max_length 512 \
      --max_new_tokens 200 \
      --use_chat_template \
      --system_prompt "" \
      --out_dir "$OUT_DIR/$m"
  done
done

echo "[done] utility evals for alpha1/2/5 (30 each, simnpo/rmu only alpha1)"
