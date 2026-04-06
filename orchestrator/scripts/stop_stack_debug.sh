#!/usr/bin/env bash
set -euo pipefail

kill_pid_file() {
  local file="$1"
  if [[ -f "$file" ]]; then
    local pid
    pid=$(cat "$file")
    if kill -0 "$pid" >/dev/null 2>&1; then
      kill "$pid" || true
      echo "[stop_stack_debug] 已停止 pid=$pid"
    fi
    rm -f "$file"
  fi
}

kill_pid_file "/home/aidlux/2026/orchestrator/pids/orchestrator.pid"
kill_pid_file "/home/aidlux/2026/VISTA/pids/vision.pid"
kill_pid_file "/home/aidlux/2026/Voice/pids/voice.pid"

echo "[stop_stack_debug] 完成。"
