#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export REPO_ROOT
PYTHONPATH="${PYTHONPATH:-}"
for path in "$REPO_ROOT" "$REPO_ROOT/Voice" "$REPO_ROOT/common" "$REPO_ROOT/orchestrator"; do
  case ":$PYTHONPATH:" in
    *":$path:"*) ;;
    *) PYTHONPATH="$path${PYTHONPATH:+:$PYTHONPATH}" ;;
  esac
done
export PYTHONPATH
PYTHON_BIN="${VOICE_PYTHON:-python3}"
PID_FILE="$SCRIPT_DIR/voice.pid"
LOG_FILE="$SCRIPT_DIR/voice.out"
usage(){ echo "Usage: $0 {fg|start|stop|status|tail} [--profile PATH] [--dry-run-text]"; exit 2; }
ACTION="${1:-}"; shift || true
PROFILE=""; EXTRA=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --profile) PROFILE="${2:-}"; [ -n "$PROFILE" ] || usage; shift 2;;
    --dry-run-text) EXTRA+=("$1"); shift;;
    *) EXTRA+=("$1"); shift;;
  esac
done
[ -n "$ACTION" ] || usage
if [ -n "$PROFILE" ] && [ ! -f "$REPO_ROOT/$PROFILE" ] && [ ! -f "$PROFILE" ]; then echo "[VOICE][BOOT][ERROR] profile not found: $PROFILE"; exit 2; fi
[ -n "$PROFILE" ] && { [ -f "$REPO_ROOT/$PROFILE" ] && PROFILE="$REPO_ROOT/$PROFILE"; }
boot(){
 echo "[VOICE][BOOT] repo_root=$REPO_ROOT"; echo "[VOICE][BOOT] profile=${PROFILE:-default}"; echo "[VOICE][BOOT] python_bin=$PYTHON_BIN"; "$PYTHON_BIN" --version; echo "[VOICE][BOOT] PYTHONPATH=$PYTHONPATH"
 "$PYTHON_BIN" -c 'import voice_service; print("[VOICE][BOOT] voice_service="+voice_service.__file__)' || { echo "[VOICE][BOOT][ERROR] voice_service import failed; PYTHONPATH=$PYTHONPATH"; exit 1; }
 local inspect_args=()
 [ -n "$PROFILE" ] && inspect_args+=(--profile "$PROFILE")
 "$PYTHON_BIN" -m voice_service.app.main "${inspect_args[@]}" --inspect-config \
  | awk '/^(debug_input_only|arecord_device|task_transport|disable_tts|mobile_feedback_transport)=/ {print "[VOICE][BOOT] "$0}'
}
args=(); [ -n "$PROFILE" ] && args+=(--profile "$PROFILE"); args+=("${EXTRA[@]}")
case "$ACTION" in
 fg) boot; cd "$REPO_ROOT"; exec "$PYTHON_BIN" -m voice_service.app.main "${args[@]}";;
 start) boot; cd "$REPO_ROOT"; nohup "$PYTHON_BIN" -m voice_service.app.main "${args[@]}" >"$LOG_FILE" 2>&1 & echo $! >"$PID_FILE"; echo "[VOICE][BOOT] background_pid=$(cat "$PID_FILE")";;
 stop) [ -f "$PID_FILE" ] && kill "$(cat "$PID_FILE")" 2>/dev/null || true; rm -f "$PID_FILE";;
 status) [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null && echo running || { echo stopped; exit 3; };;
 tail) tail -f "$LOG_FILE";;
 *) usage;;
esac
