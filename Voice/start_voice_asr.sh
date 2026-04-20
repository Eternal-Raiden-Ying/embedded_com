#!/usr/bin/env bash
set -euo pipefail

VOICE_ROOT="${VOICE_ROOT:-/home/aidlux/2026/Voice}"
LOG_DIR="$VOICE_ROOT/logs"
PID_DIR="$VOICE_ROOT/pids"
RUNS_DIR="$VOICE_ROOT/runs"
ASR_ENV_NAME="${ASR_ENV_NAME:-asr}"
MAIN_MODULE="voice_service.app.main"
PID_FILE="$PID_DIR/voice.pid"
LOG_FILE="$LOG_DIR/voice.out"
mkdir -p "$LOG_DIR" "$PID_DIR" "$RUNS_DIR"

load_conda() {
  local candidates=(
    "/home/aidlux/env/miniconda3/etc/profile.d/conda.sh"
    "$HOME/miniconda3/etc/profile.d/conda.sh"
    "$HOME/anaconda3/etc/profile.d/conda.sh"
    "/opt/conda/etc/profile.d/conda.sh"
  )
  for p in "${candidates[@]}"; do
    if [[ -f "$p" ]]; then
      # shellcheck disable=SC1090
      source "$p"
      return 0
    fi
  done
  echo "[voice] 找不到 conda.sh，请手动修改脚本中的 conda 路径。" >&2
  return 1
}

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
  echo "[voice] root=$VOICE_ROOT"
  echo "[voice] env =$ASR_ENV_NAME"
  echo "[voice] log =$LOG_FILE"
  echo "[voice] run =$(latest_run_dir)"
  echo "[voice] send_mode=${VOICE_TASK_SEND_MODE:-oneshot} ack_timeout=${VOICE_TASK_ACK_TIMEOUT_S:-0.60}"
}

start_bg() {
  if is_running; then
    echo "[voice] 已在运行, pid=$(cat "$PID_FILE")"
    print_summary
    return 0
  fi
  load_conda
  conda activate "$ASR_ENV_NAME"
  cd "$VOICE_ROOT"
  export VOICE_TASK_SEND_MODE="${VOICE_TASK_SEND_MODE:-oneshot}"
  export VOICE_TASK_ACK_TIMEOUT_S="${VOICE_TASK_ACK_TIMEOUT_S:-0.60}"
  echo "[voice] 后台启动中..."
  nohup python3 -m "$MAIN_MODULE" > "$LOG_FILE" 2>&1 &
  echo $! > "$PID_FILE"
  sleep 1
  echo "[voice] 已后台启动, pid=$(cat "$PID_FILE")"
  print_summary
}

start_fg() {
  load_conda
  conda activate "$ASR_ENV_NAME"
  cd "$VOICE_ROOT"
  export VOICE_TASK_SEND_MODE="${VOICE_TASK_SEND_MODE:-oneshot}"
  export VOICE_TASK_ACK_TIMEOUT_S="${VOICE_TASK_ACK_TIMEOUT_S:-0.60}"
  echo "[voice] 前台启动"
  print_summary
  exec python3 -m "$MAIN_MODULE"
}

stop_voice() {
  if ! is_running; then
    echo "[voice] 未运行"
    rm -f "$PID_FILE"
    return 0
  fi
  local pid
  pid="$(cat "$PID_FILE")"
  echo "[voice] 停止中, pid=$pid"
  kill "$pid" 2>/dev/null || true
  sleep 1
  if kill -0 "$pid" 2>/dev/null; then
    echo "[voice] 进程仍在运行，发送 SIGKILL" >&2
    kill -9 "$pid" 2>/dev/null || true
  fi
  rm -f "$PID_FILE"
  echo "[voice] 已停止"
}

status_voice() {
  if is_running; then
    echo "[voice] 正在运行, pid=$(cat "$PID_FILE")"
  else
    echo "[voice] 未运行"
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
    stop_voice
    ;;
  restart)
    stop_voice || true
    start_bg
    ;;
  status)
    status_voice
    ;;
  tail)
    touch "$LOG_FILE"
    echo "[voice] tail $LOG_FILE"
    tail -f "$LOG_FILE"
    ;;
  *)
    echo "用法: $0 {fg|bg|stop|restart|status|tail}" >&2
    exit 1
    ;;
esac
