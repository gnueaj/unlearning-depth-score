#!/usr/bin/env bash
set -euo pipefail

COMMON_ARGS=(
  --em_scope entity
  --entity_source full
  --patch_scope boundary
  --metric logprob
  --layers 0-15
  --delta_threshold 0.02
)

run_one() {
  local gpu=$1
  local model=$2
  python exp_s1_teacher_forcing.py --unlearn_model "$model" --gpu "$gpu" "${COMMON_ARGS[@]}"
}

run_pair() {
  local m0=$1
  local m1=$2
  run_one 0 "$m0" &
  local p0=$!
  run_one 1 "$m1" &
  local p1=$!
  wait "$p0" "$p1"
}

run_pair idknll_lr2e5_a10_ep5 idknll_lr3e5_a1_ep5
run_pair idknll_lr3e5_a1_ep10 simnpo_lr1e5_b35_ep5
run_pair simnpo_lr2e5_b35_ep10 simnpo_lr5e5_b45_ep10
run_pair graddiff_a5_ep5 graddiff_lr2e5_a5_ep10
run_one 0 graddiff_lr5e5_a2_ep10
