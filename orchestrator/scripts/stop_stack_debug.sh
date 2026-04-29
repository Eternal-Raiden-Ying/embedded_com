#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

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

kill_pid_file "$REPO_ROOT/orchestrator/pids/orchestrator.pid"
kill_pid_file "$REPO_ROOT/VISTA/pids/vision.pid"

echo "[stop_stack_debug] 完成。"
