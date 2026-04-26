#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STACK_ROOT="$SCRIPT_DIR"

# ======================================================
# robot stack launcher v5 (simple edition)
# 使用方式只保留两种：
#   1) ./start_robot_stack.sh          # 开启（默认）
#   2) ./start_robot_stack.sh stop     # 结束
#
# 平时不要记命令行参数，直接修改下面这些配置即可。
# ======================================================

# =========================
# 这里改你最常用的配置
# =========================
VOICE_ROOT="${VOICE_ROOT:-$STACK_ROOT/Voice}"
VISION_ROOT="${VISION_ROOT:-$STACK_ROOT/VISTA}"
ORCH_ROOT="${ORCH_ROOT:-$STACK_ROOT/orchestrator}"
ASR_ENV_NAME="asr"
CONDA_SH=""

# 运行模式：dryrun / full
# STACK_PROFILE="full"
STACK_PROFILE="dryrun"

# 是否接扬声器：0=不接  1=接
SPEAKER_ENABLED=0

# orchestrator 是否使用 sudo：auto / 0 / 1
# 建议：保持 auto
ORCH_USE_SUDO="auto"

# 串口设备（正式接车时用）
UART_DEV="/dev/ttyHS1"

# ready-check 超时时间（秒）
READY_TIMEOUT_S=35

# STOP 后按 Ctrl+C 只退出当前摘要显示，不停止服务
FOLLOW_ORCH_SUMMARY_AFTER_START=1

# 当前终端默认只看状态机摘要，不打印视觉/语音日志
ORCH_SUMMARY_PATTERN='状态切换|MODE |stop_class=|搜索超时|已发现目标|目标丢失|开始寻找|AUTOEXPLORE|AUTOSEARCH|SEARCH|RETURN|收到 STOP 命令|热待机|重新发现目标|自动搜索超时'
LOG_TAIL_N=80

# voice ready 判断：不要只靠端口，优先看日志
VOICE_READY_PATTERN='voice service ready'

# 端口配置
STACK_PORTS="9001 9002 9003 9011 9012"
STACK_SOCK_DIR="/tmp/robot_stack"
VISION_READY_PORTS="9003"
ORCH_READY_PORTS="9001 9002"

# 额外等待（秒）
VISION_READY_EXTRA_S=2
ORCH_READY_EXTRA_S=1
VOICE_READY_EXTRA_S=1
STOP_GRACE_S=3

# =========================
# 以下一般不用改
# =========================
VOICE_LOG_DIR="$VOICE_ROOT/logs"
VISION_LOG_DIR="$VISION_ROOT/logs"
ORCH_LOG_DIR="$ORCH_ROOT/logs"
VOICE_PID_DIR="$VOICE_ROOT/pids"
VISION_PID_DIR="$VISION_ROOT/pids"
ORCH_PID_DIR="$ORCH_ROOT/pids"

VOICE_PID_FILE="$VOICE_PID_DIR/voice.pid"
VISION_PID_FILE="$VISION_PID_DIR/vision.pid"
ORCH_PID_FILE="$ORCH_PID_DIR/orchestrator.pid"

VOICE_LOG_FILE="$VOICE_LOG_DIR/voice.out"
VISION_LOG_FILE="$VISION_LOG_DIR/vision.out"
ORCH_LOG_FILE="$ORCH_LOG_DIR/orchestrator.out"

mkdir -p "$VOICE_LOG_DIR" "$VISION_LOG_DIR" "$ORCH_LOG_DIR" \
         "$VOICE_PID_DIR" "$VISION_PID_DIR" "$ORCH_PID_DIR" "$STACK_SOCK_DIR"

# ---------- pretty output ----------
if [[ -t 1 ]]; then
  C_RESET="\033[0m"; C_BOLD="\033[1m"
  C_BLUE="\033[34m"; C_CYAN="\033[36m"; C_GREEN="\033[32m"; C_YELLOW="\033[33m"; C_RED="\033[31m"; C_MAGENTA="\033[35m"
else
  C_RESET=""; C_BOLD=""; C_BLUE=""; C_CYAN=""; C_GREEN=""; C_YELLOW=""; C_RED=""; C_MAGENTA=""
fi

mark() {
  local level="$1"; shift
  local color="$C_BLUE" tag="INFO"
  case "$level" in
    ok)   color="$C_GREEN"; tag="OK" ;;
    wait) color="$C_CYAN"; tag="WAIT" ;;
    warn) color="$C_YELLOW"; tag="WARN" ;;
    err)  color="$C_RED"; tag="FAIL" ;;
    run)  color="$C_MAGENTA"; tag="RUN" ;;
    note) color="$C_BLUE"; tag="NOTE" ;;
  esac
  printf '%b[%s]%b %s\n' "$color" "$tag" "$C_RESET" "$*"
}

headline() { printf '\n%b== %s ==%b\n' "$C_BOLD$C_BLUE" "$*" "$C_RESET"; }
log() { mark note "$*"; }
warn() { mark warn "$*" >&2; }
die() { mark err "$*" >&2; exit 1; }

apply_profile_defaults() {
  case "$STACK_PROFILE" in
    dryrun)
      ORCH_SERIAL_DRY_RUN=1
      ORCH_TTS_EVENT_OUT_TRANSPORT="disabled"
      ORCH_DRY_RUN_ECHO_STDOUT=1
      VOICE_TASK_SEND_MODE="oneshot"
      ;;
    full)
      ORCH_SERIAL_DRY_RUN=0
      ORCH_TTS_EVENT_OUT_TRANSPORT="tcp"
      ORCH_DRY_RUN_ECHO_STDOUT=0
      VOICE_TASK_SEND_MODE="oneshot"
      ;;
    *)
      die "STACK_PROFILE 只支持 dryrun/full，当前=$STACK_PROFILE"
      ;;
  esac

  if [[ "$SPEAKER_ENABLED" == "1" ]]; then
    VOICE_TEST_PROFILE="normal"
    VOICE_DISABLE_TTS=0
    VOICE_TTS_EVENT_TRANSPORT="tcp"
  else
    VOICE_TEST_PROFILE="nospeaker"
    VOICE_DISABLE_TTS=1
    VOICE_TTS_EVENT_TRANSPORT="disabled"
  fi

  PYTHONUNBUFFERED=1
}

show_banner() {
  headline "robot stack controller v5"
  printf '%bprofile%b      : %s\n' "$C_BOLD" "$C_RESET" "$STACK_PROFILE"
  printf '%bspeaker%b      : %s\n' "$C_BOLD" "$C_RESET" "$SPEAKER_ENABLED"
  printf '%bvoice root%b   : %s\n' "$C_BOLD" "$C_RESET" "$VOICE_ROOT"
  printf '%bvision root%b  : %s\n' "$C_BOLD" "$C_RESET" "$VISION_ROOT"
  printf '%borch root%b    : %s\n' "$C_BOLD" "$C_RESET" "$ORCH_ROOT"
  printf '%bready timeout%b: %ss\n' "$C_BOLD" "$C_RESET" "$READY_TIMEOUT_S"
  printf '%bsudo%b         : requested=%s effective=%s\n' "$C_BOLD" "$C_RESET" "$ORCH_USE_SUDO" "$([[ $(orch_use_sudo_effective; echo $?) -eq 0 ]] && echo 1 || echo 0)"
  printf '%bvoice profile%b: %s\n' "$C_BOLD" "$C_RESET" "$VOICE_TEST_PROFILE"
  printf '%bdry-run%b      : ORCH_SERIAL_DRY_RUN=%s  ORCH_TTS_EVENT_OUT_TRANSPORT=%s\n' "$C_BOLD" "$C_RESET" "$ORCH_SERIAL_DRY_RUN" "$ORCH_TTS_EVENT_OUT_TRANSPORT"
}

path_user_writable_or_creatable() {
  local p="$1"
  if [[ -e "$p" ]]; then
    [[ -w "$p" ]]
    return $?
  fi
  [[ -w "$(dirname "$p")" ]]
}

orch_runs_path() {
  echo "$ORCH_ROOT/runs"
}

orch_use_sudo_effective() {
  case "$ORCH_USE_SUDO" in
    1|true|yes) return 0 ;;
    0|false|no) return 1 ;;
    auto|"")
      if [[ "$ORCH_SERIAL_DRY_RUN" != "1" ]]; then
        return 0
      fi
      if ! path_user_writable_or_creatable "$(orch_runs_path)"; then
        return 0
      fi
      return 1
      ;;
    *)
      warn "未知 ORCH_USE_SUDO=$ORCH_USE_SUDO，按 auto 处理"
      if [[ "$ORCH_SERIAL_DRY_RUN" != "1" ]]; then
        return 0
      fi
      if ! path_user_writable_or_creatable "$(orch_runs_path)"; then
        return 0
      fi
      return 1
      ;;
  esac
}

need_sudo() {
  orch_use_sudo_effective
}

ensure_sudo_ready() {
  if ! need_sudo; then
    return 0
  fi
  if sudo -n true 2>/dev/null; then
    return 0
  fi
  headline "sudo 授权"
  log "需要一次 sudo 授权，用于 orchestrator 串口访问与清理 root 进程。"
  sudo -v
}

pid_alive() {
  local pid_file="$1"
  local use_sudo="${2:-0}"
  [[ -f "$pid_file" ]] || return 1
  local pid
  pid="$(cat "$pid_file" 2>/dev/null || true)"
  [[ -n "$pid" ]] || return 1
  if [[ "$use_sudo" == "1" ]]; then
    sudo -n kill -0 "$pid" 2>/dev/null
  else
    kill -0 "$pid" 2>/dev/null
  fi
}

rotate_log() {
  local f="$1"
  if [[ -f "$f" && -s "$f" ]]; then
    mv "$f" "$f.$(date +%Y%m%d_%H%M%S)"
  fi
  : > "$f"
}

launch_bg_user() {
  local name="$1" pid_file="$2" log_file="$3" cmd="$4"
  rotate_log "$log_file"
  nohup setsid bash -lc "$cmd" > "$log_file" 2>&1 &
  echo $! > "$pid_file"
  mark ok "$name 已拉起: pid=$(cat "$pid_file")  log=$log_file"
}

launch_bg_sudo() {
  local name="$1" pid_file="$2" log_file="$3" cmd="$4"
  rotate_log "$log_file"
  local pid_dir log_dir root_cmd
  pid_dir="$(dirname "$pid_file")"
  log_dir="$(dirname "$log_file")"
  root_cmd=$(cat <<CMD
set -euo pipefail
mkdir -p "$pid_dir" "$log_dir"
cd /
nohup setsid bash -lc '$cmd' > "$log_file" 2>&1 &
echo \$! > "$pid_file"
CMD
)
  sudo bash -lc "$root_cmd"
  mark ok "$name 已拉起: pid=$(cat "$pid_file")  log=$log_file"
}

start_voice_bg() {
  if pid_alive "$VOICE_PID_FILE" 0; then
    log "voice 已在运行, pid=$(cat "$VOICE_PID_FILE")"
    return 0
  fi
  headline "启动 voice"
  mark run "voice 使用 conda env=$ASR_ENV_NAME"
  local cmd
  cmd=$(cat <<CMD
set -euo pipefail
load_conda_inner() {
  local candidates=(
    "$CONDA_SH"
    "/home/aidlux/env/miniconda3/etc/profile.d/conda.sh"
    "\$HOME/miniconda3/etc/profile.d/conda.sh"
    "\$HOME/anaconda3/etc/profile.d/conda.sh"
    "/opt/conda/etc/profile.d/conda.sh"
  )
  local p
  for p in "\${candidates[@]}"; do
    [[ -n "\$p" ]] || continue
    if [[ -f "\$p" ]]; then
      source "\$p"
      return 0
    fi
  done
  echo "[voice-launch] 找不到 conda.sh" >&2
  exit 1
}
load_conda_inner
conda activate "$ASR_ENV_NAME"
cd "$VOICE_ROOT"
export PYTHONUNBUFFERED="$PYTHONUNBUFFERED"
export VOICE_TASK_SEND_MODE="$VOICE_TASK_SEND_MODE"
export VOICE_TEST_PROFILE="$VOICE_TEST_PROFILE"
export VOICE_DISABLE_TTS="$VOICE_DISABLE_TTS"
export VOICE_TTS_EVENT_TRANSPORT="$VOICE_TTS_EVENT_TRANSPORT"
exec stdbuf -oL -eL python3 -m voice_service.app.main
CMD
)
  launch_bg_user "voice" "$VOICE_PID_FILE" "$VOICE_LOG_FILE" "$cmd"
}

start_vision_bg() {
  if pid_alive "$VISION_PID_FILE" 0; then
    log "vision 已在运行, pid=$(cat "$VISION_PID_FILE")"
    return 0
  fi
  headline "启动 vision"
  mark run "vision 使用 /usr/bin/python3"
  local cmd
  cmd=$(cat <<CMD
set -euo pipefail
cd "$VISION_ROOT"
export PYTHONUNBUFFERED="$PYTHONUNBUFFERED"
exec stdbuf -oL -eL /usr/bin/python3 -m vision_module.app.app
CMD
)
  launch_bg_user "vision" "$VISION_PID_FILE" "$VISION_LOG_FILE" "$cmd"
}

start_orch_bg() {
  local sudo_flag=0
  if need_sudo; then
    sudo_flag=1
    ensure_sudo_ready
  fi
  if pid_alive "$ORCH_PID_FILE" "$sudo_flag"; then
    log "orchestrator 已在运行, pid=$(cat "$ORCH_PID_FILE")"
    return 0
  fi
  if orch_use_sudo_effective; then
    mark warn "orchestrator 将以 sudo 方式启动（原因：串口/目录权限或 full 模式需要）"
  else
    mark ok "orchestrator 将以普通用户方式启动"
  fi
  headline "启动 orchestrator"
  mark run "orchestrator 模式=$([[ $(orch_use_sudo_effective; echo $?) -eq 0 ]] && echo sudo || echo user)  python=/usr/bin/python3"
  local cmd
  cmd=$(cat <<CMD
set -euo pipefail
cd "$ORCH_ROOT"
export PYTHONUNBUFFERED="$PYTHONUNBUFFERED"
export ORCH_SERIAL_DRY_RUN="$ORCH_SERIAL_DRY_RUN"
export ORCH_TTS_EVENT_OUT_TRANSPORT="$ORCH_TTS_EVENT_OUT_TRANSPORT"
export ORCH_DRY_RUN_ECHO_STDOUT="$ORCH_DRY_RUN_ECHO_STDOUT"
exec stdbuf -oL -eL /usr/bin/python3 -m orchestrator_service.app.main
CMD
)
  if need_sudo; then
    launch_bg_sudo "orchestrator" "$ORCH_PID_FILE" "$ORCH_LOG_FILE" "$cmd"
  else
    launch_bg_user "orchestrator" "$ORCH_PID_FILE" "$ORCH_LOG_FILE" "$cmd"
  fi
}

is_port_listening() {
  local port="$1"
  if command -v ss >/dev/null 2>&1; then
    ss -ltn 2>/dev/null | awk '{print $4}' | grep -qE "(^|:)$port$"
    return $?
  fi
  if command -v netstat >/dev/null 2>&1; then
    netstat -ltn 2>/dev/null | awk '{print $4}' | grep -qE "(^|:)$port$"
    return $?
  fi
  (echo > "/dev/tcp/127.0.0.1/$port") >/dev/null 2>&1
}

wait_for_ports() {
  local name="$1" ports_str="$2" timeout_s="$3" extra_s="$4"
  local start_ts now_ts all_ok p
  start_ts=$(date +%s)
  while true; do
    all_ok=1
    for p in $ports_str; do
      if ! is_port_listening "$p"; then
        all_ok=0
        break
      fi
    done
    if [[ "$all_ok" == "1" ]]; then
      [[ "$extra_s" -gt 0 ]] && sleep "$extra_s"
      mark ok "$name ready  ports=[$ports_str]"
      return 0
    fi
    now_ts=$(date +%s)
    if (( now_ts - start_ts >= timeout_s )); then
      mark err "$name ready-check 超时  ports=[$ports_str]"
      return 1
    fi
    sleep 0.5
  done
}

wait_for_log_pattern() {
  local name="$1" file="$2" pattern="$3" timeout_s="$4" extra_s="$5"
  local start_ts now_ts
  start_ts=$(date +%s)
  while true; do
    if [[ -f "$file" ]] && grep -qE "$pattern" "$file" 2>/dev/null; then
      [[ "$extra_s" -gt 0 ]] && sleep "$extra_s"
      mark ok "$name ready  pattern=/$pattern/"
      return 0
    fi
    now_ts=$(date +%s)
    if (( now_ts - start_ts >= timeout_s )); then
      mark err "$name ready-check 超时  pattern=/$pattern/"
      return 1
    fi
    sleep 0.5
  done
}

tail_last_logs_on_failure() {
  local name="$1" file="$2"
  headline "$name 启动失败"
  warn "最近日志如下:"
  tail -n 80 "$file" 2>/dev/null || true
}

kill_pid_group() {
  local pid_file="$1" use_sudo="${2:-0}" name="$3"
  [[ -f "$pid_file" ]] || { mark note "$name 未运行"; return 0; }
  local pid
  pid="$(cat "$pid_file" 2>/dev/null || true)"
  if [[ -z "$pid" ]]; then
    rm -f "$pid_file"
    return 0
  fi
  mark wait "停止 $name, pid=$pid (优先 SIGINT 到进程组)"
  if [[ "$use_sudo" == "1" ]]; then
    sudo kill -INT -- "-$pid" 2>/dev/null || sudo kill -INT "$pid" 2>/dev/null || true
  else
    kill -INT -- "-$pid" 2>/dev/null || kill -INT "$pid" 2>/dev/null || true
  fi
  sleep "$STOP_GRACE_S"
  if [[ "$use_sudo" == "1" ]]; then
    if sudo kill -0 "$pid" 2>/dev/null; then
      warn "$name 仍在运行，发送 SIGTERM 到进程组"
      sudo kill -TERM -- "-$pid" 2>/dev/null || sudo kill -TERM "$pid" 2>/dev/null || true
      sleep 1
    fi
    if sudo kill -0 "$pid" 2>/dev/null; then
      warn "$name 仍在运行，发送 SIGKILL 到进程组"
      sudo kill -KILL -- "-$pid" 2>/dev/null || sudo kill -KILL "$pid" 2>/dev/null || true
    fi
  else
    if kill -0 "$pid" 2>/dev/null; then
      warn "$name 仍在运行，发送 SIGTERM 到进程组"
      kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
      sleep 1
    fi
    if kill -0 "$pid" 2>/dev/null; then
      warn "$name 仍在运行，发送 SIGKILL 到进程组"
      kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
    fi
  fi
  rm -f "$pid_file"
}

kill_by_ports() {
  local p
  for p in $STACK_PORTS; do
    if command -v fuser >/dev/null 2>&1; then
      if fuser -n tcp "$p" >/dev/null 2>&1; then
        log "清理端口 $p 占用"
        if need_sudo; then
          sudo fuser -k -n tcp "$p" >/dev/null 2>&1 || true
        else
          fuser -k -n tcp "$p" >/dev/null 2>&1 || true
        fi
      fi
    fi
  done
}

cleanup_sockets() {
  [[ -d "$STACK_SOCK_DIR" ]] && rm -f "$STACK_SOCK_DIR"/*.sock 2>/dev/null || true
}

stop_all() {
  if need_sudo; then
    ensure_sudo_ready
  fi
  headline "停止与清理"
  kill_pid_group "$VOICE_PID_FILE" 0 "voice"
  kill_pid_group "$VISION_PID_FILE" 0 "vision"
  kill_pid_group "$ORCH_PID_FILE" $([[ $(orch_use_sudo_effective; echo $?) -eq 0 ]] && echo 1 || echo 0) "orchestrator"
  kill_by_ports
  cleanup_sockets
}

status_one() {
  local name="$1" pid_file="$2" use_sudo="$3" log_file="$4"
  if pid_alive "$pid_file" "$use_sudo"; then
    mark ok "$name RUNNING  pid=$(cat "$pid_file")  log=$log_file"
  else
    mark warn "$name STOPPED  log=$log_file"
  fi
}

status_all() {
  headline "当前状态"
  status_one "vision" "$VISION_PID_FILE" 0 "$VISION_LOG_FILE"
  status_one "orchestrator" "$ORCH_PID_FILE" $([[ $(orch_use_sudo_effective; echo $?) -eq 0 ]] && echo 1 || echo 0) "$ORCH_LOG_FILE"
  status_one "voice" "$VOICE_PID_FILE" 0 "$VOICE_LOG_FILE"
}

tail_orch_summary() {
  touch "$ORCH_LOG_FILE"
  local pattern="$ORCH_SUMMARY_PATTERN"
  headline "状态机摘要"
  log "当前终端只显示状态机状态切换摘要。"
  log "按 Ctrl+C 仅退出摘要显示，不会停止服务；需要结束时请执行：./start_robot_stack.sh stop"
  tail -n "$LOG_TAIL_N" -F "$ORCH_LOG_FILE" | grep --line-buffered -E "$pattern" | sed -u 's/^/[orch] /'
}

start_stack() {
  show_banner
  stop_all || true

  start_vision_bg
  if ! wait_for_ports "vision" "$VISION_READY_PORTS" "$READY_TIMEOUT_S" "$VISION_READY_EXTRA_S"; then
    tail_last_logs_on_failure "vision" "$VISION_LOG_FILE"
    stop_all || true
    exit 1
  fi

  start_orch_bg
  if ! wait_for_ports "orchestrator" "$ORCH_READY_PORTS" "$READY_TIMEOUT_S" "$ORCH_READY_EXTRA_S"; then
    tail_last_logs_on_failure "orchestrator" "$ORCH_LOG_FILE"
    stop_all || true
    exit 1
  fi

  start_voice_bg
  if ! wait_for_log_pattern "voice" "$VOICE_LOG_FILE" "$VOICE_READY_PATTERN" "$READY_TIMEOUT_S" "$VOICE_READY_EXTRA_S"; then
    tail_last_logs_on_failure "voice" "$VOICE_LOG_FILE"
    stop_all || true
    exit 1
  fi

  headline "启动完成"
  mark ok "vision / orchestrator / voice 均已通过 ready-check，可以开始测试。"
  status_all

  if [[ "$FOLLOW_ORCH_SUMMARY_AFTER_START" == "1" ]]; then
    trap 'echo; mark note "退出状态机摘要显示，服务继续运行。"; exit 0' INT TERM
    tail_orch_summary
  fi
}

stop_stack() {
  show_banner
  stop_all
  status_all
}

main() {
  apply_profile_defaults
  local action="${1:-start}"
  case "$action" in
    start|on|up|run|开启|开)
      start_stack
      ;;
    stop|off|down|结束|关)
      stop_stack
      ;;
    *)
      echo "用法只保留两种："
      echo "  ./start_robot_stack.sh        # 开启（默认）"
      echo "  ./start_robot_stack.sh stop   # 结束"
      exit 1
      ;;
  esac
}

main "$@"
