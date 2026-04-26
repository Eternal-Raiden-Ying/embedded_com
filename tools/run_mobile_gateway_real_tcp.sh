#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ORCH_DIR="${REPO_ROOT}/orchestrator"

if [[ ! -f "${ORCH_DIR}/orchestrator_service/mobile_gateway/app/main.py" ]]; then
  echo "[run_mobile_gateway_real_tcp] cannot find orchestrator gateway entry" >&2
  exit 1
fi

cd "${ORCH_DIR}"
export MOBILE_GATEWAY_BACKEND="${MOBILE_GATEWAY_BACKEND:-orchestrator_tcp}"
export MOBILE_GATEWAY_STATUS_OUT_TRANSPORT="${MOBILE_GATEWAY_STATUS_OUT_TRANSPORT:-tcp}"
export MOBILE_GATEWAY_STATUS_OUT_HOST="${MOBILE_GATEWAY_STATUS_OUT_HOST:-127.0.0.1}"
export MOBILE_GATEWAY_STATUS_OUT_PORT="${MOBILE_GATEWAY_STATUS_OUT_PORT:-9102}"
export MOBILE_GATEWAY_ORCH_TASK_CMD_HOST="${MOBILE_GATEWAY_ORCH_TASK_CMD_HOST:-127.0.0.1}"
export MOBILE_GATEWAY_ORCH_TASK_CMD_PORT="${MOBILE_GATEWAY_ORCH_TASK_CMD_PORT:-9001}"
export MOBILE_GATEWAY_ORCH_TASK_ACK_TRANSPORT="${MOBILE_GATEWAY_ORCH_TASK_ACK_TRANSPORT:-tcp}"
export MOBILE_GATEWAY_ORCH_TASK_ACK_HOST="${MOBILE_GATEWAY_ORCH_TASK_ACK_HOST:-127.0.0.1}"
export MOBILE_GATEWAY_ORCH_TASK_ACK_PORT="${MOBILE_GATEWAY_ORCH_TASK_ACK_PORT:-9103}"

echo "[run_mobile_gateway_real_tcp] repo_root=${REPO_ROOT}"
echo "[run_mobile_gateway_real_tcp] backend=${MOBILE_GATEWAY_BACKEND}"
echo "[run_mobile_gateway_real_tcp] task_cmd=${MOBILE_GATEWAY_ORCH_TASK_CMD_HOST}:${MOBILE_GATEWAY_ORCH_TASK_CMD_PORT}"
echo "[run_mobile_gateway_real_tcp] task_ack listen=${MOBILE_GATEWAY_ORCH_TASK_ACK_HOST}:${MOBILE_GATEWAY_ORCH_TASK_ACK_PORT}"
exec python3 -m orchestrator_service.mobile_gateway.app.main

