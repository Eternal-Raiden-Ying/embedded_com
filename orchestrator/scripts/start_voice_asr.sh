#!/usr/bin/env bash
set -euo pipefail

VOICE_ROOT="/home/aidlux/2026/Voice"
LOG_DIR="$VOICE_ROOT/logs"
PID_DIR="$VOICE_ROOT/pids"
PID_FILE="$PID_DIR/voice.pid"
ASR_ENV_NAME="asr"
mkdir -p "$LOG_DIR" "$PID_DIR"

is_running() {
  [[ -f "$1" ]] || return 1
  local pid
  pid=$(cat "$1" 2>/dev/null || true)
  [[ -n "$pid" ]] || return 1
  kill -0 "$pid" >/dev/null 2>&1
}

load_conda() {
  local candidates=(
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
  echo "[start_voice_asr] 找不到 conda.sh，请手动修改脚本中的 conda 路径。" >&2
  return 1
}

if is_running "$PID_FILE"; then
  echo "[start_voice_asr] voice 已在运行, pid=$(cat "$PID_FILE")"
  exit 0
fi
rm -f "$PID_FILE"

load_conda
conda activate "$ASR_ENV_NAME"
cd "$VOICE_ROOT"

if [[ "${1:-fg}" == "bg" ]]; then
  nohup python3 -m voice_service.app.main > "$LOG_DIR/voice.out" 2>&1 &
  echo $! > "$PID_FILE"
  echo "[start_voice_asr] 已后台启动 voice, pid=$(cat "$PID_FILE")"
  echo "[start_voice_asr] 日志: $LOG_DIR/voice.out"
else
  echo "[start_voice_asr] 前台启动 voice (conda env: $ASR_ENV_NAME)"
  python3 -m voice_service.app.main
fi
