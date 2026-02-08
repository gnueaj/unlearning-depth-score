#!/usr/bin/env bash
set -euo pipefail

# Runs alpha1 then alpha2 with shared settings (2 GPUs).

ROOT="/home/jaeung/activation-patching-unlearning"
cd "$ROOT"

echo "[start] alpha1"
./scripts/run_openunlearning_alpha_rerun.sh alpha1
echo "[done] alpha1"

echo "[start] alpha2"
./scripts/run_openunlearning_alpha_rerun.sh alpha2
echo "[done] alpha2"
