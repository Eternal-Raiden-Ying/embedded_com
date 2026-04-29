#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STACK_ROOT="$SCRIPT_DIR"

# ======================================================
# robot stack launcher v6 (mobile control edition)
# 使用方式：
#   1) ./start_robot_stack.sh          # 开启（默认）
#   2) ./start_robot_stack.sh stop     # 结束
#   3) ./start_robot_stack.sh status   # 查看状态
#
# 当前主链路：Mobile Gateway -> Orchestrator/Controller -> VISTA
# voice 默认不再启动；手机小程序现在是任务入口。
# ======================================================

# =========================
# 这里改你最常用的配置
# =========================
VISION_ROOT="${VISION_ROOT:-$STACK_ROOT/VISTA}"
ORCH_ROOT="${ORCH_ROOT:-$STACK_ROOT/orchestrator}"
GATEWAY_CONFIG="${GATEWAY_CONFIG:-$STACK_ROOT/configs/mobile_gateway.mqtt.yaml}"
VISION_LD_PRELOAD="${VISION_LD_PRELOAD:-/lib/aarch64-linux-gnu/libGLdispatch.so.0}"
VISTA_TABLE_BBOX_ENABLE="${VISTA_TABLE_BBOX_ENABLE:-0}"
VISTA_TABLE_MODEL="${VISTA_TABLE_MODEL:-yolov7_detect}"
VISTA_PREVIEW_RGB="${VISTA_PREVIEW_RGB:-1}"
VISTA_MOCK_TABLE_BBOX="${VISTA_MOCK_TABLE_BBOX:-}"

# 运行模式：dryrun / full
# dryrun：不连接小车，只打印将要发送到 UART 的实际控制信号。
# full：连接小车串口，真实下发控制。
# STACK_PROFILE="full"
STACK_PROFILE="${STACK_PROFILE:-dryrun}"

# orchestrator 是否使用 sudo：auto / 0 / 1
# full 模式通常需要 sudo 访问串口；dryrun 一般不需要。
ORCH_USE_SUDO="${ORCH_USE_SUDO:-auto}"

# 串口设备（正式接车时用）
UART_DEV="${UART_DEV:-/dev/ttyHS1}"
UART_BAUDRATE="${UART_BAUDRATE:-115200}"

# ready-check 超时时间（秒）
READY_TIMEOUT_S="${READY_TIMEOUT_S:-35}"

# STOP 后按 Ctrl+C 只退出当前日志显示，不停止服务
FOLLOW_STACK_LOGS_AFTER_START="${FOLLOW_STACK_LOGS_AFTER_START:-1}"

# 当前终端显示的状态摘要和手机链路关键字
ORCH_SUMMARY_PATTERN='状态切换|MODE |DRY_RUN|UART|stop_class=|搜索超时|已发现目标|目标丢失|开始寻找|AUTOEXPLORE|AUTOSEARCH|SEARCH|RETURN|收到 STOP 命令|热待机|重新发现目标|自动搜索超时|TASK_CMD|task_cmd|vision_req'
GATEWAY_SUMMARY_PATTERN='gateway online|mqtt connected|mqtt disconnected|cmd received|gateway_ack sent|task_cmd forwarded|task_ack forwarded|status changed|heartbeat running|fetch_object|mobile_cmd|stop'
VISION_SUMMARY_PATTERN='SERVICE_READY|vision runtime started|mode switched|camera enabled|model loaded|vision_obs|vision_req|target_obs|table_edge_obs|FAILED|ERROR'
LOG_TAIL_N="${LOG_TAIL_N:-80}"

# TCP/UDS 端口配置
STACK_PORTS="${STACK_PORTS:-9001 9002 9003 9011 9012 9101 9102}"
STACK_SOCK_DIR="${STACK_SOCK_DIR:-/tmp/robot_stack}"
VISION_READY_PORTS="${VISION_READY_PORTS:-9003}"
ORCH_READY_PORTS="${ORCH_READY_PORTS:-9001 9002}"
GATEWAY_READY_PATTERN="${GATEWAY_READY_PATTERN:-gateway online|SERVICE_READY|mqtt connected|mqtt disabled}"

# 额外等待（秒）
VISION_READY_EXTRA_S="${VISION_READY_EXTRA_S:-2}"
ORCH_READY_EXTRA_S="${ORCH_READY_EXTRA_S:-1}"
GATEWAY_READY_EXTRA_S="${GATEWAY_READY_EXTRA_S:-1}"
STOP_GRACE_S="${STOP_GRACE_S:-3}"

# =========================
# 以下一般不用改
# =========================
VISION_LOG_DIR="$VISION_ROOT/logs"
ORCH_LOG_DIR="$ORCH_ROOT/logs"
GATEWAY_LOG_DIR="$STACK_ROOT/logs"
VISION_PID_DIR="$VISION_ROOT/pids"
ORCH_PID_DIR="$ORCH_ROOT/pids"
GATEWAY_PID_DIR="$STACK_ROOT/pids"

VISION_PID_FILE="$VISION_PID_DIR/vision.pid"
ORCH_PID_FILE="$ORCH_PID_DIR/orchestrator.pid"
GATEWAY_PID_FILE="$GATEWAY_PID_DIR/mobile_gateway.pid"

VISION_LOG_FILE="$VISION_LOG_DIR/vision.out"
ORCH_LOG_FILE="$ORCH_LOG_DIR/orchestrator.out"
GATEWAY_LOG_FILE="$GATEWAY_LOG_DIR/mobile_gateway.out"

mkdir -p "$VISION_LOG_DIR" "$ORCH_LOG_DIR" "$GATEWAY_LOG_DIR" \
         "$VISION_PID_DIR" "$ORCH_PID_DIR" "$GATEWAY_PID_DIR" "$STACK_SOCK_DIR"

# ---------- pretty output ----------
if [[ -t 1 ]]; then
  C_RESET="\033[0m"; C_BOLD="\033[1m"
  C_BLUE="\033[34m"; C_CYAN="\033[36m"; C_GREEN="\033[32m"; C_YELLOW="\033[33m"; C_RED="\033[31m"; C_MAGENTA="\033[35m"; C_GRAY="\033[90m"; C_BRIGHT_GREEN="\033[92m"; C_BOLD_CYAN="\033[1;36m"
else
  C_RESET=""; C_BOLD=""; C_BLUE=""; C_CYAN=""; C_GREEN=""; C_YELLOW=""; C_RED=""; C_MAGENTA=""; C_GRAY=""; C_BRIGHT_GREEN=""; C_BOLD_CYAN=""
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

divider() {
  printf '%b%s%b\n' "$C_BLUE" "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" "$C_RESET"
}

env_truthy() {
  case "${!1:-}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

console_color_enabled() {
  [[ -t 1 ]] || return 1
  env_truthy NO_COLOR && return 1
  env_truthy FORCE_COLOR && return 0
  local mode="${ROBOT_CONSOLE_COLOR:-auto}"
  case "$mode" in
    never) return 1 ;;
    always) return 0 ;;
    auto|"") ;;
    *) mode="auto" ;;
  esac
  [[ "$mode" == "auto" ]]
}

colorize_line() {
  local line="$1"
  if ! console_color_enabled; then
    printf '%s\n' "$line"
    return 0
  fi

  case "$line" in
    *ERROR*|*FATAL*)
      printf '%b%s%b\n' "$C_RED" "$line" "$C_RESET"
      return 0
      ;;
    *WARN*|*WARNING*)
      printf '%b%s%b\n' "$C_YELLOW" "$line" "$C_RESET"
      return 0
      ;;
    *"phone command path"*)
      printf '%b%s%b\n' "$C_GRAY" "$line" "$C_RESET"
      return 0
      ;;
  esac

  case "$line" in
    *STATE*)
      printf '%b%s%b\n' "$C_BOLD_CYAN" "$line" "$C_RESET"
      return 0
      ;;
    *MODE*|*"mode switched"*)
      printf '%b%s%b\n' "$C_BLUE" "$line" "$C_RESET"
      return 0
      ;;
    *CTRL*)
      printf '%b%s%b\n' "$C_GREEN" "$line" "$C_RESET"
      return 0
      ;;
    *CAR*)
      printf '%b%s%b\n' "$C_MAGENTA" "$line" "$C_RESET"
      return 0
      ;;
    *EDGE*)
      printf '%b%s%b\n' "$C_CYAN" "$line" "$C_RESET"
      return 0
      ;;
    *TARGET*)
      printf '%b%s%b\n' "$C_YELLOW" "$line" "$C_RESET"
      return 0
      ;;
  esac

  case "$line" in
    "[orchestrator]"*|"[ORCH]"*)
      printf '%b%s%b\n' "$C_GREEN" "$line" "$C_RESET"
      ;;
    "[vista]"*|"[VISTA]"*)
      printf '%b%s%b\n' "$C_BLUE" "$line" "$C_RESET"
      ;;
    "[phone-gateway]"*)
      printf '%b%s%b\n' "$C_YELLOW" "$line" "$C_RESET"
      ;;
    *)
      printf '%s\n' "$line"
      ;;
  esac
}

headline() {
  printf '\n'
  divider
  printf '%b== %s ==%b\n' "$C_BOLD$C_BLUE" "$*" "$C_RESET"
  divider
}

log() { mark note "$*"; }
warn() { mark warn "$*" >&2; }
die() { mark err "$*" >&2; exit 1; }

apply_profile_defaults() {
  case "$STACK_PROFILE" in
    dryrun)
      ORCH_SERIAL_DRY_RUN=1
      ORCH_TTS_EVENT_OUT_TRANSPORT="disabled"
      ORCH_DRY_RUN_ECHO_STDOUT=1
      ;;
    full)
      ORCH_SERIAL_DRY_RUN=0
      ORCH_TTS_EVENT_OUT_TRANSPORT="disabled"
      ORCH_DRY_RUN_ECHO_STDOUT=0
      ;;
    *)
      die "STACK_PROFILE 只支持 dryrun/full，当前=$STACK_PROFILE"
      ;;
  esac

  PYTHONUNBUFFERED=1
}

show_banner() {
  headline "robot stack controller v6"
  printf '%bprofile%b        : %s\n' "$C_BOLD" "$C_RESET" "$STACK_PROFILE"
  printf '%bmobile input%b   : mobile_gateway (voice disabled)\n' "$C_BOLD" "$C_RESET"
  printf '%bvision root%b    : %s\n' "$C_BOLD" "$C_RESET" "$VISION_ROOT"
  printf '%bvision preload%b : %s\n' "$C_BOLD" "$C_RESET" "${VISION_LD_PRELOAD:-<none>}"
  printf '%btable bbox%b     : enable=%s model=%s mock=%s preview_rgb=%s\n' \
    "$C_BOLD" "$C_RESET" "$VISTA_TABLE_BBOX_ENABLE" "$VISTA_TABLE_MODEL" "${VISTA_MOCK_TABLE_BBOX:-<none>}" "$VISTA_PREVIEW_RGB"
  printf '%borch root%b      : %s\n' "$C_BOLD" "$C_RESET" "$ORCH_ROOT"
  printf '%bgateway config%b : %s\n' "$C_BOLD" "$C_RESET" "$GATEWAY_CONFIG"
  printf '%bready timeout%b  : %ss\n' "$C_BOLD" "$C_RESET" "$READY_TIMEOUT_S"
  printf '%buart device%b    : %s @ %s\n' "$C_BOLD" "$C_RESET" "$UART_DEV" "$UART_BAUDRATE"
  printf '%bsudo%b           : requested=%s effective=%s\n' "$C_BOLD" "$C_RESET" "$ORCH_USE_SUDO" "$([[ $(orch_use_sudo_effective; echo $?) -eq 0 ]] && echo 1 || echo 0)"
  printf '%bdry-run%b        : ORCH_SERIAL_DRY_RUN=%s  ORCH_DRY_RUN_ECHO_STDOUT=%s\n' "$C_BOLD" "$C_RESET" "$ORCH_SERIAL_DRY_RUN" "$ORCH_DRY_RUN_ECHO_STDOUT"
  printf '%bports%b          : %s\n' "$C_BOLD" "$C_RESET" "$STACK_PORTS"
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

start_vision_bg() {
  if pid_alive "$VISION_PID_FILE" 0; then
    log "vision 已在运行, pid=$(cat "$VISION_PID_FILE")"
    return 0
  fi
  headline "启动 vision / VISTA"
  mark run "vision 使用 /usr/bin/python3"
  local cmd
  cmd=$(cat <<CMD
set -euo pipefail
cd "$VISION_ROOT"
export PYTHONUNBUFFERED="$PYTHONUNBUFFERED"
export ROBOT_CONSOLE_COLOR=never
unset FORCE_COLOR
export VISTA_TABLE_BBOX_ENABLE="$VISTA_TABLE_BBOX_ENABLE"
export VISTA_TABLE_MODEL="$VISTA_TABLE_MODEL"
export VISTA_PREVIEW_RGB="$VISTA_PREVIEW_RGB"
export VISTA_MOCK_TABLE_BBOX="$VISTA_MOCK_TABLE_BBOX"
if [[ -n "$VISION_LD_PRELOAD" && -e "$VISION_LD_PRELOAD" ]]; then
  export LD_PRELOAD="$VISION_LD_PRELOAD\${LD_PRELOAD:+:\$LD_PRELOAD}"
fi
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
  headline "启动 orchestrator / controller"
  mark run "orchestrator 模式=$([[ $(orch_use_sudo_effective; echo $?) -eq 0 ]] && echo sudo || echo user)  python=/usr/bin/python3"
  local cmd
  cmd=$(cat <<CMD
set -euo pipefail
cd "$ORCH_ROOT"
export PYTHONUNBUFFERED="$PYTHONUNBUFFERED"
export ROBOT_CONSOLE_COLOR=never
unset FORCE_COLOR
export ORCH_SERIAL_DRY_RUN="$ORCH_SERIAL_DRY_RUN"
export ORCH_SERIAL_PORT="$UART_DEV"
export ORCH_SERIAL_BAUDRATE="$UART_BAUDRATE"
export ORCH_TTS_EVENT_OUT_TRANSPORT="$ORCH_TTS_EVENT_OUT_TRANSPORT"
export ORCH_DRY_RUN_ECHO_STDOUT="$ORCH_DRY_RUN_ECHO_STDOUT"
export ORCH_TASK_CMD_IN_HOST="127.0.0.1"
export ORCH_TASK_CMD_IN_PORT="9001"
export ORCH_TASK_ACK_OUT_HOST="127.0.0.1"
export ORCH_TASK_ACK_OUT_PORT="9012"
export ORCH_VISION_OBS_IN_HOST="127.0.0.1"
export ORCH_VISION_OBS_IN_PORT="9002"
export ORCH_VISION_REQ_OUT_HOST="127.0.0.1"
export ORCH_VISION_REQ_OUT_PORT="9003"
exec stdbuf -oL -eL /usr/bin/python3 -m orchestrator_service.app.main
CMD
)
  if need_sudo; then
    launch_bg_sudo "orchestrator" "$ORCH_PID_FILE" "$ORCH_LOG_FILE" "$cmd"
  else
    launch_bg_user "orchestrator" "$ORCH_PID_FILE" "$ORCH_LOG_FILE" "$cmd"
  fi
}

start_gateway_bg() {
  if pid_alive "$GATEWAY_PID_FILE" 0; then
    log "mobile_gateway 已在运行, pid=$(cat "$GATEWAY_PID_FILE")"
    return 0
  fi
  if [[ ! -f "$GATEWAY_CONFIG" ]]; then
    die "找不到 mobile gateway 配置: $GATEWAY_CONFIG"
  fi
  headline "启动 mobile gateway"
  mark run "gateway config=$GATEWAY_CONFIG"
  local cmd
  cmd=$(cat <<CMD
set -euo pipefail
cd "$STACK_ROOT"
export PYTHONUNBUFFERED="$PYTHONUNBUFFERED"
export ROBOT_CONSOLE_COLOR=never
unset FORCE_COLOR
export PYTHONPATH="$ORCH_ROOT"
export MOBILE_GATEWAY_ORCH_TASK_CMD_HOST="127.0.0.1"
export MOBILE_GATEWAY_ORCH_TASK_CMD_PORT="9001"
export MOBILE_GATEWAY_ORCH_TASK_ACK_HOST="127.0.0.1"
export MOBILE_GATEWAY_ORCH_TASK_ACK_PORT="9012"
exec stdbuf -oL -eL /usr/bin/python3 -m orchestrator_service.mobile_gateway.runtime.service --config "$GATEWAY_CONFIG"
CMD
)
  launch_bg_user "mobile_gateway" "$GATEWAY_PID_FILE" "$GATEWAY_LOG_FILE" "$cmd"
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
  kill_pid_group "$GATEWAY_PID_FILE" 0 "mobile_gateway"
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
  status_one "orchestrator/controller" "$ORCH_PID_FILE" $([[ $(orch_use_sudo_effective; echo $?) -eq 0 ]] && echo 1 || echo 0) "$ORCH_LOG_FILE"
  status_one "mobile_gateway" "$GATEWAY_PID_FILE" 0 "$GATEWAY_LOG_FILE"
}

tail_stack_summary() {
  touch "$ORCH_LOG_FILE" "$GATEWAY_LOG_FILE" "$VISION_LOG_FILE"
  headline "单终端日志"
  log "显示 mobile_gateway / orchestrator / VISTA 的关键日志。"
  log "按 Ctrl+C 只退出日志显示，不会停止服务；需要结束时执行：./start_robot_stack.sh stop"
  log "彩色显示测试：ROBOT_CONSOLE_COLOR=always ./start_robot_stack.sh"
  divider
  tail -n "$LOG_TAIL_N" -F "$GATEWAY_LOG_FILE" "$ORCH_LOG_FILE" "$VISION_LOG_FILE" 2>/dev/null | \
  awk -v orch_pat="$ORCH_SUMMARY_PATTERN" \
      -v gw_pat="$GATEWAY_SUMMARY_PATTERN" \
      -v vista_pat="$VISION_SUMMARY_PATTERN" '
    /^==> .* <==$/ {
      src=$0
      next
    }
    {
      tag="[stack]"
      show=0
      if (src ~ /mobile_gateway\.out/) {
        tag="[phone-gateway]"
        show=($0 ~ gw_pat)
      } else if (src ~ /orchestrator\.out/) {
        tag="[orchestrator]"
        show=($0 ~ orch_pat)
      } else if (src ~ /vision\.out/) {
        tag="[vista]"
        show=($0 ~ vista_pat)
      }
      if (show) {
        if ($0 ~ /(mobile_cmd|cmd received|gateway_ack|task_ack|fetch_object|status changed|stop)/) {
          print "···· phone command path ····"
        }
        print tag " " $0
      }
    }
  ' | while IFS= read -r line; do
    colorize_line "$line"
  done
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

  start_gateway_bg
  if ! wait_for_log_pattern "mobile_gateway" "$GATEWAY_LOG_FILE" "$GATEWAY_READY_PATTERN" "$READY_TIMEOUT_S" "$GATEWAY_READY_EXTRA_S"; then
    tail_last_logs_on_failure "mobile_gateway" "$GATEWAY_LOG_FILE"
    stop_all || true
    exit 1
  fi

  headline "启动完成"
  mark ok "vision / orchestrator(controller) / mobile_gateway 均已通过 ready-check，可以用手机端发指令。"
  status_all

  if [[ "$FOLLOW_STACK_LOGS_AFTER_START" == "1" ]]; then
    trap 'echo; mark note "退出日志显示，服务继续运行。"; exit 0' INT TERM
    tail_stack_summary
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
    status|状态)
      status_all
      ;;
    *)
      echo "用法："
      echo "  ./start_robot_stack.sh        # 开启（默认）"
      echo "  ./start_robot_stack.sh stop   # 结束"
      echo "  ./start_robot_stack.sh status # 查看状态"
      exit 1
      ;;
  esac
}

main "$@"
