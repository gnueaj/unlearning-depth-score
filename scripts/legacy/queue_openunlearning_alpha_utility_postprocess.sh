#!/usr/bin/env bash
set -euo pipefail

LOG="logs/openunlearning_alpha125_utility.log"

echo "[queue] waiting for utility runs to finish..."
while true; do
  if [ -f "$LOG" ] && rg -n "\\[done\\] open-unlearning evals" "$LOG" >/dev/null 2>&1; then
    break
  fi
  sleep 60
done

echo "[queue] utility runs done. Building combined table/HTML..."
python3 scripts/build_openunlearning_alpha_all.py --out_dir docs/0202
echo "[queue] done"
