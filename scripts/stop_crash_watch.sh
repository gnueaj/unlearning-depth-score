#!/usr/bin/env bash
set -euo pipefail

# Stop crash-watch background loggers.
# Usage:
#   scripts/stop_crash_watch.sh
#   scripts/stop_crash_watch.sh logs/crash_watch/<timestamp>

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_RUN_DIR="$(ls -1dt "$ROOT_DIR"/logs/crash_watch/* 2>/dev/null | head -n 1 || true)"
RUN_DIR="${1:-$DEFAULT_RUN_DIR}"

if [[ -z "$RUN_DIR" ]]; then
  echo "No crash watch run directory found."
  exit 1
fi

PID_FILE="$RUN_DIR/pids.txt"
if [[ ! -f "$PID_FILE" ]]; then
  echo "PID file not found: $PID_FILE"
  exit 1
fi

echo "Stopping crash watch in: $RUN_DIR"

# Reverse order is safer for pipeline trees.
tac "$PID_FILE" | while read -r pid name; do
  if [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null; then
    kill "$pid" 2>/dev/null || true
    echo "  stopped $name (pid=$pid)"
  else
    echo "  already dead $name (pid=${pid:-unknown})"
  fi
done

echo "Done."
