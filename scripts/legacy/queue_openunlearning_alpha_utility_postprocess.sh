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

echo "[queue] utility runs done."
echo "[queue] skipped: scripts/legacy/build_openunlearning_alpha_all.py was removed."
echo "[queue] done"
