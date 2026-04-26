#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ORCH_DIR="${REPO_ROOT}/orchestrator"

if [[ ! -f "${ORCH_DIR}/orchestrator_service/mobile_gateway/app/main.py" ]]; then
  echo "[run_mobile_gateway_mock] cannot find orchestrator_service/mobile_gateway/app/main.py" >&2
  echo "[run_mobile_gateway_mock] expected repo root: ${REPO_ROOT}" >&2
  exit 1
fi

cd "${ORCH_DIR}"
export MOBILE_GATEWAY_BACKEND="${MOBILE_GATEWAY_BACKEND:-mock}"
export MOBILE_GATEWAY_STATUS_OUT_TRANSPORT="${MOBILE_GATEWAY_STATUS_OUT_TRANSPORT:-tcp}"
export MOBILE_GATEWAY_STATUS_OUT_HOST="${MOBILE_GATEWAY_STATUS_OUT_HOST:-127.0.0.1}"
export MOBILE_GATEWAY_STATUS_OUT_PORT="${MOBILE_GATEWAY_STATUS_OUT_PORT:-9102}"

echo "[run_mobile_gateway_mock] repo_root=${REPO_ROOT}"
echo "[run_mobile_gateway_mock] backend=${MOBILE_GATEWAY_BACKEND}"
echo "[run_mobile_gateway_mock] status_out=${MOBILE_GATEWAY_STATUS_OUT_HOST}:${MOBILE_GATEWAY_STATUS_OUT_PORT}"
exec python3 -m orchestrator_service.mobile_gateway.app.main

