#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DEFAULT_VISION_ROOT="$REPO_ROOT/VISTA"

VISION_ROOT="${VISION_ROOT:-$DEFAULT_VISION_ROOT}"
LOG_DIR="$VISION_ROOT/logs"
PID_DIR="$VISION_ROOT/pids"
PID_FILE="$PID_DIR/vision.pid"
MAIN_MODULE="${VISION_MAIN_MODULE:-vision_module.app.app}"
mkdir -p "$LOG_DIR" "$PID_DIR"

is_running() {
  [[ -f "$1" ]] || return 1
  local pid
  pid=$(cat "$1" 2>/dev/null || true)
  [[ -n "$pid" ]] || return 1
  kill -0 "$pid" >/dev/null 2>&1
}

cd "$VISION_ROOT"

if is_running "$PID_FILE"; then
  echo "[start_vision_service] vision 已在运行, pid=$(cat "$PID_FILE")"
  exit 0
fi
rm -f "$PID_FILE"

if [[ "${1:-fg}" == "bg" ]]; then
  nohup python3 -m "$MAIN_MODULE" > "$LOG_DIR/vision.out" 2>&1 &
  echo $! > "$PID_FILE"
  echo "[start_vision_service] 已后台启动 vision, pid=$(cat "$PID_FILE")"
  echo "[start_vision_service] 日志: $LOG_DIR/vision.out"
else
  echo "[start_vision_service] 前台启动 vision"
  python3 -m "$MAIN_MODULE"
fi
