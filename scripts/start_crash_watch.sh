#!/usr/bin/env bash
set -euo pipefail

# Start background loggers for sudden server/GPU crashes.
# Usage:
#   scripts/start_crash_watch.sh
#   scripts/start_crash_watch.sh /custom/log/root

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_ROOT="${1:-$ROOT_DIR/logs/crash_watch}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$LOG_ROOT/$TIMESTAMP"
PID_FILE="$RUN_DIR/pids.txt"

mkdir -p "$RUN_DIR"
touch "$PID_FILE"

log() {
  printf "[%s] %s\n" "$(date +'%F %T')" "$*"
}

start_job() {
  local name="$1"
  shift
  (
    exec stdbuf -oL -eL "$@"
  ) >> "$RUN_DIR/$name.log" 2>&1 &
  local pid=$!
  printf "%s %s\n" "$pid" "$name" >> "$PID_FILE"
  log "started $name (pid=$pid)"
}

{
  echo "run_dir=$RUN_DIR"
  echo "started_at=$(date +'%F %T %Z')"
  echo "host=$(hostname)"
  echo "user=$(whoami)"
  echo "kernel=$(uname -r)"
} > "$RUN_DIR/meta.txt"

KERNEL_FILTER='nvrm|xid|mmu|ctx|oom|killed process|panic|segfault|watchdog|pcie|aer|fallen off the bus|illegal memory access'

if command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null; then
  start_job kernel_full sudo journalctl -q -kf -o short-iso
  start_job kernel_filtered bash -lc "sudo journalctl -q -kf -o short-iso | rg -i --line-buffered \"$KERNEL_FILTER\""
else
  echo "kernel_log_scope=limited (run with sudo for full kernel logs)" >> "$RUN_DIR/meta.txt"
  start_job kernel_full journalctl -q -kf -o short-iso
  start_job kernel_filtered bash -lc "journalctl -q -kf -o short-iso | rg -i --line-buffered \"$KERNEL_FILTER\""
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  start_job nvidia_dmon nvidia-smi dmon -s pucvmet -d 1
  start_job nvidia_query bash -lc "while true; do date +'%F %T'; nvidia-smi --query-gpu=index,uuid,pci.bus_id,temperature.gpu,fan.speed,pstate,power.draw,memory.used,utilization.gpu --format=csv,noheader; echo; sleep 5; done"
fi

PROC_FILTER='meta_eval_robustness.py|meta_eval_faithfulness.py|exp_s1_teacher_forcing.py|patchscope\\.|python .*meta_eval_|python .*exp_s1_teacher_forcing.py'
start_job process_watch bash -lc "self=\$\$; while true; do date +'%F %T'; ps -eo pid,ppid,etime,%cpu,%mem,cmd | awk -v self=\"\$self\" 'NR==1 || (\$1 != self && \$0 !~ /awk -v self=/ && \$0 ~ /$PROC_FILTER/)'; echo; sleep 5; done"

echo "Crash watch started."
echo "Run dir: $RUN_DIR"
echo "PID file: $PID_FILE"
echo "Stop command: scripts/stop_crash_watch.sh \"$RUN_DIR\""
