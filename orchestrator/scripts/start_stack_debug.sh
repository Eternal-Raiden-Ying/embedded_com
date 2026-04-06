#!/usr/bin/env bash
set -euo pipefail

WITH_VOICE=0
WITH_VISION=1

for arg in "$@"; do
  if [[ "$arg" == "--with-voice" ]]; then
    WITH_VOICE=1
  elif [[ "$arg" == "--without-vision" ]]; then
    WITH_VISION=0
  fi
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "$SCRIPT_DIR/start_orchestrator.sh" bg
sleep 1

if [[ "$WITH_VISION" == "1" ]]; then
  bash "$SCRIPT_DIR/start_vision_service.sh" bg
  sleep 1
fi

if [[ "$WITH_VOICE" == "1" ]]; then
  bash "$SCRIPT_DIR/start_voice_asr.sh" bg
  sleep 1
fi

echo "[start_stack_debug] 启动完成。"
echo "[start_stack_debug] 默认已带上 vision；加 --with-voice 可一并启动语音。"
echo "[start_stack_debug] 如需停止，请执行: bash $SCRIPT_DIR/stop_stack_debug.sh"
