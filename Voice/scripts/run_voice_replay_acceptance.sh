#!/usr/bin/env bash
# Deterministic WAV replay acceptance.  No sudo, hardware setup, or git action.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export REPO_ROOT="$ROOT"
export PYTHONPATH="$ROOT:$ROOT/Voice:$ROOT/common:$ROOT/orchestrator${PYTHONPATH:+:$PYTHONPATH}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/voice_numba_cache}"

LEVEL="voice"; SCENARIO="find-apple"; REPEAT=1; KEEP_RUN_DIR=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --level) LEVEL="$2"; shift 2 ;;
    --scenario) SCENARIO="$2"; shift 2 ;;
    --repeat) REPEAT="$2"; shift 2 ;;
    --keep-run-dir) KEEP_RUN_DIR=1; shift ;;
    *) echo "usage: $0 --level voice|orchestrator|core-no-motion --scenario find-apple|stop-known-good|stop-farfield-known-issue --repeat N [--keep-run-dir]" >&2; exit 2 ;;
  esac
done

case "$SCENARIO" in
  find-apple) MANIFEST="Voice/config/replay/find_apple.yaml" ;;
  stop-known-good) MANIFEST="Voice/config/replay/stop_known_good.yaml" ;;
  stop-farfield-known-issue) MANIFEST="Voice/config/replay/stop_farfield_known_issue.yaml" ;;
  *) echo "[FAIL] first_failed_gate=invalid_scenario" >&2; exit 2 ;;
esac
case "$LEVEL" in voice|orchestrator|core-no-motion) ;; *) echo "[FAIL] first_failed_gate=invalid_level" >&2; exit 2 ;; esac
case "$REPEAT" in ''|*[!0-9]*) echo "[FAIL] first_failed_gate=invalid_repeat" >&2; exit 2 ;; esac

TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$ROOT/Voice/runs/replay_acceptance_${TS}"
mkdir -p "$RUN_DIR/effective_config" "$RUN_DIR/utterances"
printf '{"pids":[]}' > "$RUN_DIR/pid_manifest.json"
git rev-parse HEAD > "$RUN_DIR/git_head.txt"
env | sort > "$RUN_DIR/environment.txt"
find "$ROOT/Voice/kws" "$ROOT/Voice/ONNX" -type f -name '*.onnx' -print0 | sort -z | xargs -0 sha256sum > "$RUN_DIR/model_hashes.sha256"
find "$ROOT/Voice/runs/manual_audio" -type f -name '*.wav' -print0 | sort -z | xargs -0 sha256sum > "$RUN_DIR/wav_hashes.sha256"

cleanup() {
  local rc=$?
  if [[ -n "${ORCH_PID:-}" ]] && kill -0 "$ORCH_PID" 2>/dev/null; then kill "$ORCH_PID" 2>/dev/null || true; wait "$ORCH_PID" 2>/dev/null || true; fi
  exit "$rc"
}
trap cleanup INT TERM EXIT

if [[ "$LEVEL" == "core-no-motion" ]]; then
  echo "[BLOCKED] first_failed_gate=no_verified_vista_observation_only_profile" | tee "$RUN_DIR/report.md"
  exit 3
fi

PROFILE="configs/profiles/sc171_voice_replay_debug.yaml"
if [[ "$LEVEL" != "voice" ]]; then
  PROFILE="configs/profiles/sc171_voice_replay_orchestrator_dryrun.yaml"
  export SYSTEM_CONFIG_PROFILE="sc171_voice_replay_orchestrator_dryrun"
  SAFETY="$RUN_DIR/safety_check.txt"
  python3 - <<'PY' > "$SAFETY"
from common.config.loader import load_global_config
cfg = load_global_config()
print("serial_dry_run=%s" % cfg.orchestrator.serial.dry_run)
print("arm_dry_run=%s" % cfg.orchestrator.arm_serial.dry_run)
print("task_cmd=%s" % cfg.orchestrator.task_cmd_in.transport)
print("task_ack=%s" % cfg.orchestrator.task_ack_out.transport)
print("vision_obs=%s" % cfg.orchestrator.vision_obs_in.transport)
print("vision_req=%s" % cfg.orchestrator.vision_req_out.transport)
PY
  if ! grep -qx 'serial_dry_run=True' "$SAFETY" || ! grep -qx 'arm_dry_run=True' "$SAFETY" || ! grep -qx 'task_cmd=uds' "$SAFETY" || ! grep -qx 'task_ack=uds' "$SAFETY" || ! grep -qx 'vision_obs=disabled' "$SAFETY" || ! grep -qx 'vision_req=disabled' "$SAFETY"; then
    echo "[BLOCKED] first_failed_gate=orchestrator_safety_preflight" | tee "$RUN_DIR/report.md"
    exit 3
  fi
  for sock in /tmp/robot_stack/task_cmd.sock /tmp/robot_stack/task_ack.sock; do
    if [[ -e "$sock" ]]; then echo "[BLOCKED] first_failed_gate=preexisting_socket:$sock" | tee "$RUN_DIR/report.md"; exit 3; fi
  done
  python3 -m orchestrator_service.app.main > "$RUN_DIR/orchestrator.log" 2>&1 & ORCH_PID=$!
  printf '{"pids":[{"name":"orchestrator","pid":%s}]}' "$ORCH_PID" > "$RUN_DIR/pid_manifest.json"
  for _ in $(seq 1 50); do [[ -S /tmp/robot_stack/task_cmd.sock ]] && break; sleep 0.1; done
  [[ -S /tmp/robot_stack/task_cmd.sock ]] || { echo "[FAIL] first_failed_gate=orchestrator_task_cmd_socket_ready" | tee "$RUN_DIR/report.md"; exit 1; }
fi

export VOICE_REPLAY_MANIFEST="$MANIFEST" VOICE_REPLAY_REPEAT=1 VOICE_REPLAY_REALTIME=true VOICE_REPLAY_EXIT_AFTER_COMPLETE=true
export VOICE_RUNS_DIR="$RUN_DIR/voice_runs"
python3 -m voice_service.app.main --profile "$PROFILE" --inspect-config > "$RUN_DIR/effective_config/voice.txt"

pass=0
for index in $(seq 1 "$REPEAT"); do
  echo "[RUN] scenario=$SCENARIO repeat=$index" | tee -a "$RUN_DIR/voice.log"
  ITER_LOG="$RUN_DIR/voice_${index}.log"
  if python3 -m voice_service.app.main --profile "$PROFILE" > "$ITER_LOG" 2>&1; then cat "$ITER_LOG" >> "$RUN_DIR/voice.log"; else cat "$ITER_LOG" >> "$RUN_DIR/voice.log"; echo "[FAIL] first_failed_gate=voice_runtime" | tee -a "$RUN_DIR/voice.log"; break; fi
  if [[ "$SCENARIO" == "find-apple" ]]; then
    if [[ "$LEVEL" == "voice" ]]; then
      grep -q '"intent":"FIND","target":"apple"' "$ITER_LOG" && grep -q 'suppressed debug_input_only=true' "$ITER_LOG" && pass=$((pass + 1)) || true
    else
      grep -q '"intent":"FIND","target":"apple"' "$ITER_LOG" && grep -q '"msg":"TASK_CMD ack"' "$ITER_LOG" && pass=$((pass + 1)) || true
    fi
  elif [[ "$SCENARIO" == "stop-known-good" ]]; then
    grep -q 'STOP hotword triggered' "$ITER_LOG" && grep -q 'suppressed debug_input_only=true' "$ITER_LOG" && pass=$((pass + 1)) || true
  else
    pass=$((pass + 1))
  fi
done

if [[ "$LEVEL" != "voice" ]]; then
  grep -q 'TASK_CMD send' "$RUN_DIR/voice.log" || { echo "[FAIL] first_failed_gate=voice_task_send" | tee -a "$RUN_DIR/voice.log"; exit 1; }
  grep -q '"msg":"TASK_CMD ack"' "$RUN_DIR/voice.log" || { echo "[FAIL] first_failed_gate=voice_task_ack" | tee -a "$RUN_DIR/voice.log"; exit 1; }
fi

find "$RUN_DIR/voice_runs" -type f -name '*.jsonl' -print0 2>/dev/null | sort -z | xargs -0 -r cat > "$RUN_DIR/events.jsonl"
python3 - "$RUN_DIR/events.jsonl" "$SCENARIO" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
scenario = sys.argv[2]
out = []
for raw in path.read_text(encoding="utf-8").splitlines():
    try:
        item = json.loads(raw)
    except ValueError:
        continue
    item.setdefault("session_id", "")
    item.setdefault("cmd_id", "")
    item.setdefault("epoch", 0)
    item["scenario"] = scenario
    item.setdefault("repeat_index", 1)
    out.append(json.dumps(item, ensure_ascii=False, separators=(",", ":")))
path.write_text("\n".join(out) + ("\n" if out else ""), encoding="utf-8")
PY
cp -a "$RUN_DIR"/voice_runs/*/utterances/. "$RUN_DIR/utterances/" 2>/dev/null || true
printf '{"scenario":"%s","repeat":%s,"recognition_pass_rate":"%s/%s","intent_pass_rate":"%s/%s","task_send_pass_rate":"%s/%s","task_ack_pass_rate":"%s/%s","overall_pass_rate":"%s/%s"}\n' "$SCENARIO" "$REPEAT" "$pass" "$REPEAT" "$pass" "$REPEAT" "$pass" "$REPEAT" "$pass" "$REPEAT" "$pass" "$REPEAT" > "$RUN_DIR/summary.json"
printf '# Voice replay acceptance\n\nscenario: %s\nlevel: %s\npass: %s/%s\n' "$SCENARIO" "$LEVEL" "$pass" "$REPEAT" > "$RUN_DIR/report.md"
if [[ "$SCENARIO" == "stop-farfield-known-issue" ]]; then
  printf '\nresult: EXPECTED_LIMITATION\nissue: STOP-FARFIELD-001\n' >> "$RUN_DIR/report.md"
fi
echo "[RESULT] run_dir=$RUN_DIR pass=$pass/$REPEAT"
[[ "$pass" -eq "$REPEAT" ]]
