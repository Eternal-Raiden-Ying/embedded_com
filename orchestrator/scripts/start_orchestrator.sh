#!/usr/bin/env bash
set -euo pipefail

ORCH_ROOT="${ORCH_ROOT:-/home/aidlux/2026/orchestrator}"
LOG_DIR="$ORCH_ROOT/logs"
PID_DIR="$ORCH_ROOT/pids"
RUNS_DIR="$ORCH_ROOT/runs"
MAIN_MODULE="orchestrator_service.app.main"
PID_FILE="$PID_DIR/orchestrator.pid"
LOG_FILE="$LOG_DIR/orchestrator.out"
mkdir -p "$LOG_DIR" "$PID_DIR" "$RUNS_DIR"

latest_run_dir() {
  ls -1dt "$RUNS_DIR"/run_* 2>/dev/null | head -n 1 || true
}

is_running() {
  [[ -f "$PID_FILE" ]] || return 1
  local pid
  pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  [[ -n "$pid" ]] || return 1
  kill -0 "$pid" 2>/dev/null
}

print_summary() {
  echo "[orchestrator] root=$ORCH_ROOT"
  echo "[orchestrator] log =$LOG_FILE"
  echo "[orchestrator] run =$(latest_run_dir)"
}

start_bg() {
  if is_running; then
    echo "[orchestrator] 已在运行, pid=$(cat "$PID_FILE")"
    print_summary
    return 0
  fi
  cd "$ORCH_ROOT"
  echo "[orchestrator] 后台启动中..."
  nohup python3 -m "$MAIN_MODULE" > "$LOG_FILE" 2>&1 &
  echo $! > "$PID_FILE"
  sleep 1
  echo "[orchestrator] 已后台启动, pid=$(cat "$PID_FILE")"
  print_summary
}

start_fg() {
  cd "$ORCH_ROOT"
  echo "[orchestrator] 前台启动"
  print_summary
  exec python3 -m "$MAIN_MODULE"
}

stop_orchestrator() {
  if ! is_running; then
    echo "[orchestrator] 未运行"
    rm -f "$PID_FILE"
    return 0
  fi
  local pid
  pid="$(cat "$PID_FILE")"
  echo "[orchestrator] 停止中, pid=$pid"
  kill "$pid" 2>/dev/null || true
  sleep 1
  if kill -0 "$pid" 2>/dev/null; then
    echo "[orchestrator] 进程仍在运行，发送 SIGKILL" >&2
    kill -9 "$pid" 2>/dev/null || true
  fi
  rm -f "$PID_FILE"
  echo "[orchestrator] 已停止"
}

status_orchestrator() {
  if is_running; then
    echo "[orchestrator] 正在运行, pid=$(cat "$PID_FILE")"
  else
    echo "[orchestrator] 未运行"
  fi
  print_summary
}

case "${1:-fg}" in
  fg)
    start_fg
    ;;
  bg)
    start_bg
    ;;
  stop)
    stop_orchestrator
    ;;
  restart)
    stop_orchestrator || true
    start_bg
    ;;
  status)
    status_orchestrator
    ;;
  tail)
    touch "$LOG_FILE"
    echo "[orchestrator] tail $LOG_FILE"
    tail -f "$LOG_FILE"
    ;;
  *)
    echo "用法: $0 {fg|bg|stop|restart|status|tail}" >&2
    exit 1
    ;;
esac
