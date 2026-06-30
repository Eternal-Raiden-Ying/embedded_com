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
SYSTEM_CONFIG_FILE="${SYSTEM_CONFIG_FILE:-$STACK_ROOT/configs/system_config.yaml}"
if [[ -z "${GATEWAY_CONFIG+x}" && -f "$STACK_ROOT/configs/mobile_gateway.mqtt.yaml" ]]; then
  GATEWAY_CONFIG="$STACK_ROOT/configs/mobile_gateway.mqtt.yaml"
else
  GATEWAY_CONFIG="${GATEWAY_CONFIG:-$SYSTEM_CONFIG_FILE}"
fi
VISION_LD_PRELOAD="${VISION_LD_PRELOAD:-/lib/aarch64-linux-gnu/libGLdispatch.so.0}"
VISTA_TABLE_BBOX_ENABLE="${VISTA_TABLE_BBOX_ENABLE:-0}"
VISTA_TABLE_MODEL="${VISTA_TABLE_MODEL:-yolo26s_detect}"
VISTA_PREVIEW_RGB="${VISTA_PREVIEW_RGB:-1}"
VISTA_MOCK_TABLE_BBOX="${VISTA_MOCK_TABLE_BBOX:-}"

# 运行模式：dryrun / full
# dryrun：不连接小车，只打印将要发送到 UART 的实际控制信号。
# full：连接小车串口，真实下发控制。
# STACK_PROFILE="full"
STACK_PROFILE="${STACK_PROFILE:-full}"

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
ROBOT_CONSOLE_LEVEL="${ROBOT_CONSOLE_LEVEL:-normal}"
ENABLE_GATEWAY_LOGS="${ENABLE_GATEWAY_LOGS:-false}"

# 当前终端显示的状态摘要和手机链路关键字
ORCH_SUMMARY_PATTERN='状态切换|MODE |DRY_RUN|UART|stop_class=|搜索超时|已发现目标|目标丢失|开始寻找|AUTOEXPLORE|AUTOSEARCH|SEARCH|RETURN|收到 STOP 命令|热待机|重新发现目标|自动搜索超时|TASK_CMD|task_cmd|vision_req|TASK_DONE|DEMO|IDLE_HOT|preview kept alive|waiting for next command|target        :|session_id    :|result        :|final_state   :|reason        :|total_time_s  :|next_state    :|preview       :|waiting       :|━━━━━━━━'
GATEWAY_SUMMARY_PATTERN='gateway online|mqtt connected|mqtt disconnected|received mobile command|forwarded task_cmd|gateway_ack sent|task_ack forwarded|status changed|heartbeat running|fetch_object|mobile_cmd|stop'
VISION_SUMMARY_PATTERN='SERVICE_READY|vision runtime started|mode switched|camera enabled|model loaded|vision_obs|vision_req|target_obs|table_edge_obs|FAILED|ERROR'
LOG_TAIL_N="${LOG_TAIL_N:-80}"

# TCP/UDS 端口与 Socket 配置
STACK_PORTS="${STACK_PORTS:-9001 9002 9003 9011 9012 9101 9102}"
STACK_SOCK_DIR="${STACK_SOCK_DIR:-/tmp/robot_stack}"
VISION_READY_SOCKETS="${VISION_READY_SOCKETS:-/tmp/robot_stack/vision_req.sock}"
ORCH_READY_SOCKETS="${ORCH_READY_SOCKETS:-/tmp/robot_stack/task_cmd.sock /tmp/robot_stack/vision_obs.sock}"
GATEWAY_READY_PATTERN="${GATEWAY_READY_PATTERN:-gateway online|SERVICE_READY|mqtt connected|http listening}"

# 额外等待（秒）
VISION_READY_EXTRA_S="${VISION_READY_EXTRA_S:-2}"
ORCH_READY_EXTRA_S="${ORCH_READY_EXTRA_S:-1}"
GATEWAY_READY_EXTRA_S="${GATEWAY_READY_EXTRA_S:-1}"
STOP_GRACE_S="${STOP_GRACE_S:-3}"

# Stop 后自动生成本次 run summary。
# 关闭：RUN_SUMMARY_ENABLE=0 ./start_robot_stack.sh stop
RUN_SUMMARY_ENABLE="${RUN_SUMMARY_ENABLE:-1}"
RUN_SUMMARY_SCRIPT="${RUN_SUMMARY_SCRIPT:-$STACK_ROOT/tools/run_summary.py}"
RUN_SUMMARY_NO_PLOTS="${RUN_SUMMARY_NO_PLOTS:-0}"

# =========================
# 以下一般不用改
# =========================
STACK_RUNS_ROOT="$STACK_ROOT/logs/runs"
LATEST_RUN_ID_FILE="$STACK_RUNS_ROOT/latest_run_id"
VISION_PID_DIR="$VISION_ROOT/pids"
ORCH_PID_DIR="$ORCH_ROOT/pids"
GATEWAY_PID_DIR="$STACK_ROOT/pids"

VISION_PID_FILE="$VISION_PID_DIR/vision.pid"
ORCH_PID_FILE="$ORCH_PID_DIR/orchestrator.pid"
GATEWAY_PID_FILE="$GATEWAY_PID_DIR/mobile_gateway.pid"

VISION_LOG_DIR=""
ORCH_LOG_DIR=""
GATEWAY_LOG_DIR=""
VISION_LOG_FILE=""
ORCH_LOG_FILE=""
GATEWAY_LOG_FILE=""

gateway_logs_enabled() {
  case "${ENABLE_GATEWAY_LOGS:-false}" in
    *1*|*true*|*yes*|*on*) return 0 ;;
    *) return 1 ;;
  esac
}

mkdir -p "$STACK_RUNS_ROOT" "$VISION_PID_DIR" "$ORCH_PID_DIR" "$GATEWAY_PID_DIR" "$STACK_SOCK_DIR"
if gateway_logs_enabled; then
  mkdir -p "$GATEWAY_LOG_DIR"
fi

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
    *"[DEMO][DRY_RUN]"*)
      printf '%b%s%b\n' "$C_YELLOW" "$line" "$C_RESET"
      return 0
      ;;
    *"[DEMO][SUCCESS]"*|*"DEMO SUCCESS"*)
      printf '%b%s%b\n' "$C_BRIGHT_GREEN" "$line" "$C_RESET"
      return 0
      ;;
    *"[DEMO][FAILED]"*|*"DEMO FAILED"*)
      printf '%b%s%b\n' "$C_RED" "$line" "$C_RESET"
      return 0
      ;;
    *"[DEMO][IDLE"*|*"IDLE_HOT"*|*"preview kept alive, waiting for next command"*)
      printf '%b%s%b\n' "$C_BOLD_CYAN" "$line" "$C_RESET"
      return 0
      ;;
    *"[DEMO][HEALTH]"*)
      printf '%b%s%b\n' "$C_CYAN" "$line" "$C_RESET"
      return 0
      ;;
    *"[DEMO][PHONE]"*|*"[DEMO][START]"*)
      printf '%b%s%b\n' "$C_MAGENTA" "$line" "$C_RESET"
      return 0
      ;;
    *"[DEMO][PREVIEW]"*|*"[DEMO][WARN]"*|*"[DEMO][RECOVER]"*)
      printf '%b%s%b\n' "$C_YELLOW" "$line" "$C_RESET"
      return 0
      ;;
    *"[DEMO]"*)
      printf '%b%s%b\n' "$C_BOLD_CYAN" "$line" "$C_RESET"
      return 0
      ;;
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
    dryrun|dry_run)
      ORCH_SERIAL_DRY_RUN=1
      ORCH_TTS_EVENT_OUT_TRANSPORT="disabled"
      ORCH_DRY_RUN_ECHO_STDOUT=0
      CONFIG_PROFILE_EFFECTIVE="dry_run"
      ;;
    full|sc171_board)
      ORCH_SERIAL_DRY_RUN=0
      ORCH_TTS_EVENT_OUT_TRANSPORT="disabled"
      ORCH_DRY_RUN_ECHO_STDOUT=0
      CONFIG_PROFILE_EFFECTIVE="sc171_board"
      ;;
    windows_dev)
      ORCH_SERIAL_DRY_RUN=1
      ORCH_TTS_EVENT_OUT_TRANSPORT="disabled"
      ORCH_DRY_RUN_ECHO_STDOUT=0
      CONFIG_PROFILE_EFFECTIVE="windows_dev"
      ;;
    *)
      die "STACK_PROFILE 只支持 dryrun/full/sc171_board/windows_dev，当前=$STACK_PROFILE"
      ;;
  esac

  SYSTEM_CONFIG_PROFILE="${SYSTEM_CONFIG_PROFILE:-$CONFIG_PROFILE_EFFECTIVE}"
  PYTHONUNBUFFERED=1
}

show_banner() {
  headline "robot stack controller v6"
  [[ -n "${STACK_RUN_ID:-}" ]] && printf '%brun id%b         : %s\n' "$C_BOLD" "$C_RESET" "$STACK_RUN_ID"
  printf '%bprofile%b        : %s\n' "$C_BOLD" "$C_RESET" "$STACK_PROFILE"
  printf '%bconfig profile%b : %s\n' "$C_BOLD" "$C_RESET" "$SYSTEM_CONFIG_PROFILE"
  printf '%bmobile input%b   : mobile_gateway (voice disabled)\n' "$C_BOLD" "$C_RESET"
  printf '%bvision root%b    : %s\n' "$C_BOLD" "$C_RESET" "$VISION_ROOT"
  printf '%bvision preload%b : %s\n' "$C_BOLD" "$C_RESET" "${VISION_LD_PRELOAD:-<none>}"
  printf '%btable bbox%b     : enable=%s model=%s mock=%s preview_rgb=%s\n' \
    "$C_BOLD" "$C_RESET" "$VISTA_TABLE_BBOX_ENABLE" "$VISTA_TABLE_MODEL" "${VISTA_MOCK_TABLE_BBOX:-<none>}" "$VISTA_PREVIEW_RGB"
  printf '%borch root%b      : %s\n' "$C_BOLD" "$C_RESET" "$ORCH_ROOT"
  printf '%bsystem config%b  : %s\n' "$C_BOLD" "$C_RESET" "$SYSTEM_CONFIG_FILE"
  printf '%bgateway config%b : %s\n' "$C_BOLD" "$C_RESET" "$GATEWAY_CONFIG"
  printf '%bready timeout%b  : %ss\n' "$C_BOLD" "$C_RESET" "$READY_TIMEOUT_S"
  printf '%buart device%b    : %s @ %s\n' "$C_BOLD" "$C_RESET" "$UART_DEV" "$UART_BAUDRATE"
  printf '%bsudo%b           : requested=%s effective=%s\n' "$C_BOLD" "$C_RESET" "$ORCH_USE_SUDO" "$([[ $(orch_use_sudo_effective; echo $?) -eq 0 ]] && echo 1 || echo 0)"
  printf '%bdry-run%b        : ORCH_SERIAL_DRY_RUN=%s  ORCH_DRY_RUN_ECHO_STDOUT=%s\n' "$C_BOLD" "$C_RESET" "$ORCH_SERIAL_DRY_RUN" "$ORCH_DRY_RUN_ECHO_STDOUT"
  printf '%bconsole%b        : ROBOT_CONSOLE_LEVEL=%s  ROBOT_CONSOLE_COLOR=%s\n' "$C_BOLD" "$C_RESET" "$ROBOT_CONSOLE_LEVEL" "${ROBOT_CONSOLE_COLOR:-auto}"
  printf '%bgateway logs%b   : enabled=%s\n' "$C_BOLD" "$C_RESET" "$ENABLE_GATEWAY_LOGS"
  printf '%bports%b          : %s\n' "$C_BOLD" "$C_RESET" "$STACK_PORTS"
}

make_run_id() {
  printf 'run_%s_%06x\n' "$(date +%Y%m%d_%H%M%S)" "$((RANDOM * RANDOM % 16777216))"
}

apply_run_log_paths() {
  local run_id="$1"
  local run_dir="$STACK_RUNS_ROOT/$run_id"
  STACK_RUN_ID="$run_id"
  STACK_RUN_DIR="$run_dir"
  VISION_LOG_DIR="$run_dir/vision"
  ORCH_LOG_DIR="$run_dir/orchestrator"
  GATEWAY_LOG_DIR="$run_dir/mobile_gateway"
  VISION_LOG_FILE="$VISION_LOG_DIR/vision.out"
  ORCH_LOG_FILE="$ORCH_LOG_DIR/orchestrator.out"
  if gateway_logs_enabled; then
    GATEWAY_LOG_FILE="$GATEWAY_LOG_DIR/mobile_gateway.out"
  else
    GATEWAY_LOG_FILE="/dev/null"
  fi
  mkdir -p "$VISION_LOG_DIR" "$ORCH_LOG_DIR" "$STACK_RUNS_ROOT"
  if gateway_logs_enabled; then
    mkdir -p "$GATEWAY_LOG_DIR"
  fi
}

prepare_start_run_paths() {
  local run_id="${STACK_RUN_ID:-}"
  [[ -n "$run_id" ]] || run_id="$(make_run_id)"
  apply_run_log_paths "$run_id"
  printf '%s\n' "$run_id" > "$LATEST_RUN_ID_FILE"
  ln -sfn "$run_id" "$STACK_RUNS_ROOT/latest" 2>/dev/null || true
}

prepare_latest_run_paths() {
  if [[ -z "${STACK_RUN_ID:-}" && -f "$LATEST_RUN_ID_FILE" ]]; then
    STACK_RUN_ID="$(cat "$LATEST_RUN_ID_FILE" 2>/dev/null || true)"
  fi
  if [[ -n "${STACK_RUN_ID:-}" && -d "$STACK_RUNS_ROOT/$STACK_RUN_ID" ]]; then
    apply_run_log_paths "$STACK_RUN_ID"
  fi
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
  if [[ "$log_file" != "/dev/null" ]]; then
    rotate_log "$log_file"
  fi
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
export PYTHONPATH="$STACK_ROOT:$VISION_ROOT\${PYTHONPATH:+:\$PYTHONPATH}"
export SYSTEM_CONFIG_FILE="$SYSTEM_CONFIG_FILE"
export SYSTEM_CONFIG_PROFILE="$SYSTEM_CONFIG_PROFILE"
export PYTHONUNBUFFERED="$PYTHONUNBUFFERED"
export ROBOT_CONSOLE_COLOR=never
export ROBOT_CONSOLE_LEVEL="$ROBOT_CONSOLE_LEVEL"
export ROBOT_RUN_MODULE_SUBDIRS=1
export STACK_RUN_ID="$STACK_RUN_ID"
export VISION_LOG_DIR="$VISION_LOG_DIR"
export VISION_RUNS_DIR="$STACK_RUNS_ROOT"
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
export PYTHONPATH="$STACK_ROOT:$ORCH_ROOT\${PYTHONPATH:+:\$PYTHONPATH}"
export SYSTEM_CONFIG_FILE="$SYSTEM_CONFIG_FILE"
export SYSTEM_CONFIG_PROFILE="$SYSTEM_CONFIG_PROFILE"
export PYTHONUNBUFFERED="$PYTHONUNBUFFERED"
export ROBOT_CONSOLE_COLOR=never
export ROBOT_RUN_MODULE_SUBDIRS=1
export STACK_RUN_ID="$STACK_RUN_ID"
export ORCH_LOG_DIR="$ORCH_LOG_DIR"
export ORCH_RUNS_DIR="$STACK_RUNS_ROOT"
unset FORCE_COLOR
export ORCH_SERIAL_DRY_RUN="$ORCH_SERIAL_DRY_RUN"
if [[ "$ORCH_SERIAL_DRY_RUN" == "0" ]]; then
  unset ENV
fi
export ROBOT_CONSOLE_LEVEL="$ROBOT_CONSOLE_LEVEL"
export ORCH_SERIAL_PORT="$UART_DEV"
export ORCH_SERIAL_BAUDRATE="$UART_BAUDRATE"
export ORCH_TTS_EVENT_OUT_TRANSPORT="$ORCH_TTS_EVENT_OUT_TRANSPORT"
export ORCH_DRY_RUN_ECHO_STDOUT="$ORCH_DRY_RUN_ECHO_STDOUT"
export ORCH_TASK_CMD_IN_SOCKET_PATH="/tmp/robot_stack/task_cmd.sock"
export ORCH_TASK_ACK_OUT_SOCKET_PATH="/tmp/robot_stack/task_ack.sock"
export ORCH_VISION_OBS_IN_SOCKET_PATH="/tmp/robot_stack/vision_obs.sock"
export ORCH_VISION_REQ_OUT_SOCKET_PATH="/tmp/robot_stack/vision_req.sock"
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
export PYTHONPATH="$STACK_ROOT:$ORCH_ROOT\${PYTHONPATH:+:\$PYTHONPATH}"
export SYSTEM_CONFIG_FILE="$SYSTEM_CONFIG_FILE"
export SYSTEM_CONFIG_PROFILE="$SYSTEM_CONFIG_PROFILE"
export PYTHONUNBUFFERED="$PYTHONUNBUFFERED"
export ROBOT_CONSOLE_COLOR=never
export ROBOT_RUN_MODULE_SUBDIRS=1
export STACK_RUN_ID="$STACK_RUN_ID"
unset FORCE_COLOR
export ROBOT_CONSOLE_LEVEL="$ROBOT_CONSOLE_LEVEL"
export MOBILE_GATEWAY_LOG_ENABLED="$(gateway_logs_enabled && echo 1 || echo 0)"
export MOBILE_GATEWAY_STATUS_STDOUT="$(gateway_logs_enabled && echo 1 || echo 0)"
export MOBILE_GATEWAY_LOG_DIR="$GATEWAY_LOG_DIR"
export MOBILE_GATEWAY_LOG_FILE="$(gateway_logs_enabled && echo "$GATEWAY_LOG_DIR/mobile_gateway.log" || echo "/dev/null")"
export MOBILE_GATEWAY_RUNS_DIR="$STACK_RUNS_ROOT"
export MOBILE_GATEWAY_ORCH_RUNS_DIR="$STACK_RUNS_ROOT"
export MOBILE_GATEWAY_ORCH_STATE_BLOCKS_PATH="$STACK_RUN_DIR/orchestrator/state_blocks.jsonl"
export MOBILE_GATEWAY_BACKEND="\${MOBILE_GATEWAY_BACKEND:-tcp_no_ack}"
export MOBILE_GATEWAY_COMMAND_IN_TRANSPORT="\${MOBILE_GATEWAY_COMMAND_IN_TRANSPORT:-http}"
export MOBILE_GATEWAY_COMMAND_IN_HOST="\${MOBILE_GATEWAY_COMMAND_IN_HOST:-0.0.0.0}"
export MOBILE_GATEWAY_COMMAND_IN_PORT="\${MOBILE_GATEWAY_COMMAND_IN_PORT:-9001}"
export MOBILE_GATEWAY_LEGACY_TCP_ENABLED="\${MOBILE_GATEWAY_LEGACY_TCP_ENABLED:-1}"
export MOBILE_GATEWAY_LEGACY_TCP_HOST="\${MOBILE_GATEWAY_LEGACY_TCP_HOST:-0.0.0.0}"
export MOBILE_GATEWAY_LEGACY_TCP_PORT="\${MOBILE_GATEWAY_LEGACY_TCP_PORT:-9101}"
export MOBILE_GATEWAY_ORCH_TASK_CMD_TRANSPORT="\${MOBILE_GATEWAY_ORCH_TASK_CMD_TRANSPORT:-uds}"
export MOBILE_GATEWAY_ORCH_TASK_CMD_SOCKET_PATH="/tmp/robot_stack/task_cmd.sock"
export MOBILE_GATEWAY_ORCH_TASK_ACK_SOCKET_PATH="/tmp/robot_stack/task_ack.sock"
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

endpoint_spec() {
  local endpoint="$1"
  SYSTEM_CONFIG_FILE="$SYSTEM_CONFIG_FILE" SYSTEM_CONFIG_PROFILE="$SYSTEM_CONFIG_PROFILE" PYTHONPATH="$STACK_ROOT:${PYTHONPATH:-}" \
    /usr/bin/python3 -c '
import sys
from common.config_loader import get_config

cfg = get_config()
name = sys.argv[1]
if name == "vision_req":
    ep = cfg.vision.req_in
elif name == "orchestrator_task_cmd":
    ep = cfg.orchestrator.task_cmd_in
elif name == "orchestrator_vision_obs":
    ep = cfg.orchestrator.vision_obs_in
else:
    raise SystemExit(f"unknown endpoint: {name}")

mode = str(getattr(ep, "transport", "disabled") or "disabled").strip().lower()
path = str(getattr(ep, "ipc_socket_path", "") or getattr(ep, "uds_path", "") or "")
host = str(getattr(ep, "tcp_host", "") or getattr(ep, "host", "") or "127.0.0.1")
port = int(getattr(ep, "tcp_port", 0) or getattr(ep, "port", 0) or 0)
print(f"{mode}|{path}|{host}|{port}")
' "$endpoint"
}

endpoint_connect_ok() {
  local mode="$1" path="$2" host="$3" port="$4"
  /usr/bin/python3 -c '
import socket
import sys

mode, path, host, port = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4] or 0)
try:
    if mode == "uds":
        if not hasattr(socket, "AF_UNIX"):
            raise RuntimeError("socket.AF_UNIX unavailable")
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(0.5)
        sock.connect(path)
    elif mode == "tcp":
        sock = socket.create_connection((host, port), timeout=0.5)
    else:
        raise RuntimeError(f"unsupported connect mode: {mode}")
    sock.close()
except Exception:
    raise SystemExit(1)
' "$mode" "$path" "$host" "$port"
}

ready_pid_alive() {
  local name="$1" pid_file="" use_sudo=0
  case "$name" in
    vision)
      pid_file="$VISION_PID_FILE"
      ;;
    orchestrator)
      pid_file="$ORCH_PID_FILE"
      if orch_use_sudo_effective; then
        use_sudo=1
      fi
      ;;
    mobile_gateway)
      pid_file="$GATEWAY_PID_FILE"
      ;;
  esac
  [[ -n "$pid_file" ]] && pid_alive "$pid_file" "$use_sudo"
}

ready_log_file_for() {
  case "$1" in
    vision) printf '%s\n' "$VISION_LOG_FILE" ;;
    orchestrator) printf '%s\n' "$ORCH_LOG_FILE" ;;
    mobile_gateway) printf '%s\n' "$GATEWAY_LOG_FILE" ;;
    *) printf '%s\n' "" ;;
  esac
}

wait_for_endpoint() {
  local service="$1" endpoint="$2" timeout_s="$3" extra_s="$4"
  local spec mode path host port start_ts now_ts log_file
  spec="$(endpoint_spec "$endpoint")" || return 1
  IFS='|' read -r mode path host port <<< "$spec"
  case "$mode" in
    uds)
      mark note "[READY_CHECK] $endpoint endpoint mode=uds path=$path"
      ;;
    tcp)
      mark note "[READY_CHECK] $endpoint endpoint mode=tcp host=$host port=$port"
      ;;
    disabled)
      mark note "[READY_CHECK] $endpoint endpoint mode=disabled; checking process + READY log"
      ;;
    *)
      mark err "$service ready-check 不支持 endpoint=$endpoint mode=$mode"
      return 1
      ;;
  esac

  start_ts=$(date +%s)
  while true; do
    if [[ "$mode" == "disabled" ]]; then
      log_file="$(ready_log_file_for "$service")"
      if ready_pid_alive "$service" && [[ -f "$log_file" ]] && grep -qE 'READY|SERVICE_READY' "$log_file" 2>/dev/null; then
        [[ "$extra_s" -gt 0 ]] && sleep "$extra_s"
        mark ok "$service ready  endpoint=$endpoint mode=disabled"
        return 0
      fi
    elif [[ "$mode" == "uds" ]]; then
      if [[ -S "$path" ]] && endpoint_connect_ok "$mode" "$path" "$host" "$port"; then
        [[ "$extra_s" -gt 0 ]] && sleep "$extra_s"
        if ! ready_pid_alive "$service"; then
          mark err "$service ready-check 失败：endpoint 曾经可用，但进程已退出"
          return 1
        fi
        mark ok "$service ready  endpoint=$endpoint mode=uds path=$path"
        return 0
      fi
    elif [[ "$mode" == "tcp" ]]; then
      if endpoint_connect_ok "$mode" "$path" "$host" "$port"; then
        [[ "$extra_s" -gt 0 ]] && sleep "$extra_s"
        if ! ready_pid_alive "$service"; then
          mark err "$service ready-check 失败：endpoint 曾经可用，但进程已退出"
          return 1
        fi
        mark ok "$service ready  endpoint=$endpoint mode=tcp host=$host port=$port"
        return 0
      fi
    fi

    now_ts=$(date +%s)
    if (( now_ts - start_ts >= timeout_s )); then
      case "$mode" in
        uds) mark err "$service ready-check 超时  endpoint=$endpoint mode=uds path=$path" ;;
        tcp) mark err "$service ready-check 超时  endpoint=$endpoint mode=tcp host=$host port=$port" ;;
        disabled) mark err "$service ready-check 超时  endpoint=$endpoint mode=disabled pattern=/READY|SERVICE_READY/" ;;
      esac
      return 1
    fi
    sleep 0.5
  done
}

wait_for_endpoint_group() {
  local service="$1" timeout_s="$2" extra_s="$3"; shift 3
  local endpoint
  for endpoint in "$@"; do
    wait_for_endpoint "$service" "$endpoint" "$timeout_s" "$extra_s" || return 1
  done
}

gateway_spec() {
  SYSTEM_CONFIG_FILE="$SYSTEM_CONFIG_FILE" SYSTEM_CONFIG_PROFILE="$SYSTEM_CONFIG_PROFILE" PYTHONPATH="$STACK_ROOT:${PYTHONPATH:-}" \
    MOBILE_GATEWAY_BACKEND="${MOBILE_GATEWAY_BACKEND:-tcp_no_ack}" \
    MOBILE_GATEWAY_COMMAND_IN_TRANSPORT="${MOBILE_GATEWAY_COMMAND_IN_TRANSPORT:-http}" \
    MOBILE_GATEWAY_COMMAND_IN_HOST="${MOBILE_GATEWAY_COMMAND_IN_HOST:-0.0.0.0}" \
    MOBILE_GATEWAY_COMMAND_IN_PORT="${MOBILE_GATEWAY_COMMAND_IN_PORT:-9001}" \
    MOBILE_GATEWAY_ORCH_TASK_CMD_TRANSPORT="${MOBILE_GATEWAY_ORCH_TASK_CMD_TRANSPORT:-uds}" \
    MOBILE_GATEWAY_ORCH_TASK_CMD_SOCKET_PATH="${MOBILE_GATEWAY_ORCH_TASK_CMD_SOCKET_PATH:-/tmp/robot_stack/task_cmd.sock}" \
    /usr/bin/python3 -c '
from common.config_loader import get_config

cfg = get_config().gateway
cmd = cfg.command_in
orch = cfg.orchestrator_task_cmd_out
print("|".join([
    str(cfg.backend.mode or ""),
    str(cmd.transport or ""),
    str(getattr(cmd, "tcp_host", "") or getattr(cmd, "host", "") or ""),
    str(getattr(cmd, "tcp_port", 0) or getattr(cmd, "port", 0) or 0),
    str(getattr(cmd, "ipc_socket_path", "") or getattr(cmd, "uds_path", "") or ""),
    "1" if bool(cfg.mqtt.enabled) else "0",
    str(orch.transport or ""),
    str(getattr(orch, "ipc_socket_path", "") or getattr(orch, "uds_path", "") or ""),
    str(getattr(orch, "tcp_host", "") or getattr(orch, "host", "") or ""),
    str(getattr(orch, "tcp_port", 0) or getattr(orch, "port", 0) or 0),
]))
'
}

gateway_http_health_ok() {
  local host="$1" port="$2"
  [[ "$host" == "0.0.0.0" || "$host" == "::" || -z "$host" ]] && host="127.0.0.1"
  /usr/bin/python3 -c '
import http.client
import json
import sys

host, port = sys.argv[1], int(sys.argv[2] or 0)
try:
    conn = http.client.HTTPConnection(host, port, timeout=0.8)
    conn.request("GET", "/health")
    resp = conn.getresponse()
    data = json.loads(resp.read().decode("utf-8") or "{}")
    conn.close()
    if resp.status != 200 or not data.get("ok"):
        raise RuntimeError(f"bad health status={resp.status} payload={data!r}")
    if str(data.get("backend_mode", "")).lower() == "mock":
        raise RuntimeError("gateway backend_mode is mock")
except Exception as exc:
    print(str(exc), file=sys.stderr)
    raise SystemExit(1)
' "$host" "$port"
}

wait_for_gateway_ready() {
  local timeout_s="$1" extra_s="$2"
  local spec backend cmd_mode cmd_host cmd_port cmd_path mqtt_enabled orch_mode orch_path orch_host orch_port
  spec="$(gateway_spec)" || return 1
  IFS='|' read -r backend cmd_mode cmd_host cmd_port cmd_path mqtt_enabled orch_mode orch_path orch_host orch_port <<< "$spec"
  backend="$(printf '%s' "$backend" | tr '[:upper:]' '[:lower:]')"
  cmd_mode="$(printf '%s' "$cmd_mode" | tr '[:upper:]' '[:lower:]')"
  orch_mode="$(printf '%s' "$orch_mode" | tr '[:upper:]' '[:lower:]')"

  mark note "[READY_CHECK] gateway backend=$backend command_in=$cmd_mode host=$cmd_host port=$cmd_port path=$cmd_path"
  mark note "[READY_CHECK] gateway orchestrator_task_cmd_out=$orch_mode host=$orch_host port=$orch_port path=$orch_path"

  if [[ "$backend" == "mock" && "${MOBILE_GATEWAY_ALLOW_MOCK:-0}" != "1" ]]; then
    mark err "mobile_gateway backend_mode=mock；dry_run/full 默认禁止 mock，请显式 MOBILE_GATEWAY_ALLOW_MOCK=1 才能调试使用"
    return 1
  fi
  if [[ "$orch_mode" != "uds" && "$orch_mode" != "tcp" ]]; then
    mark err "mobile_gateway orchestrator_task_cmd_out 未启用：transport=$orch_mode"
    return 1
  fi

  local start_ts now_ts
  start_ts=$(date +%s)
  while true; do
    if endpoint_connect_ok "$orch_mode" "$orch_path" "${orch_host:-127.0.0.1}" "${orch_port:-0}"; then
      if [[ "$cmd_mode" == "http" ]]; then
        if gateway_http_health_ok "$cmd_host" "$cmd_port" 2>/dev/null; then
          [[ "$extra_s" -gt 0 ]] && sleep "$extra_s"
          mark ok "mobile_gateway ready  backend=$backend http=${cmd_host}:${cmd_port} orch_task_cmd=$orch_mode"
          return 0
        fi
      elif [[ "$mqtt_enabled" == "1" ]]; then
        if [[ -f "$GATEWAY_LOG_FILE" ]] && grep -qE "mqtt connected" "$GATEWAY_LOG_FILE" 2>/dev/null; then
          [[ "$extra_s" -gt 0 ]] && sleep "$extra_s"
          mark ok "mobile_gateway ready  backend=$backend mqtt=connected orch_task_cmd=$orch_mode"
          return 0
        fi
      else
        mark err "mobile_gateway command_in=$cmd_mode 且 mqtt 未启用；需要 HTTP 或 MQTT 真实入口"
        return 1
      fi
    fi

    now_ts=$(date +%s)
    if (( now_ts - start_ts >= timeout_s )); then
      mark err "mobile_gateway ready-check 超时 backend=$backend command_in=$cmd_mode http=${cmd_host}:${cmd_port} orch_task_cmd=$orch_mode"
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
  if [[ -d "$STACK_SOCK_DIR" ]]; then
    if ! rm -f "$STACK_SOCK_DIR"/*.sock 2>/dev/null; then
      sudo rm -f \
        "$STACK_SOCK_DIR/vision_req.sock" \
        "$STACK_SOCK_DIR/task_cmd.sock" \
        "$STACK_SOCK_DIR/vision_obs.sock" \
        "$STACK_SOCK_DIR/task_ack.sock" 2>/dev/null || true
    fi
  fi
  mkdir -p "$STACK_SOCK_DIR"
  chmod 1777 "$STACK_SOCK_DIR" 2>/dev/null || sudo chmod 1777 "$STACK_SOCK_DIR" 2>/dev/null || true
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

run_latest_summary() {
  [[ "${RUN_SUMMARY_ENABLE:-1}" == "1" ]] || return 0
  [[ -n "${STACK_RUN_DIR:-}" && -d "$STACK_RUN_DIR" ]] || { mark warn "run summary skipped: STACK_RUN_DIR not found"; return 0; }
  [[ -f "$RUN_SUMMARY_SCRIPT" ]] || { mark warn "run summary skipped: script not found: $RUN_SUMMARY_SCRIPT"; return 0; }

  headline "生成 run summary"
  local plot_flag=""
  [[ "${RUN_SUMMARY_NO_PLOTS:-0}" == "1" ]] && plot_flag="--no-plots"
  if /usr/bin/python3 "$RUN_SUMMARY_SCRIPT" "$STACK_RUN_DIR" $plot_flag; then
    mark ok "run summary 已生成: $STACK_RUN_DIR/summary/run_summary_auto.md"
  else
    mark warn "run summary 生成失败，但 stop 流程继续"
  fi
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
  log "console level: ROBOT_CONSOLE_LEVEL=$ROBOT_CONSOLE_LEVEL (demo/normal/debug)"
  log "按 Ctrl+C 只退出日志显示，不会停止服务；需要结束时执行：./start_robot_stack.sh stop"
  log "彩色显示测试：ROBOT_CONSOLE_COLOR=always ./start_robot_stack.sh"
  divider
  tail -n "$LOG_TAIL_N" -F "$GATEWAY_LOG_FILE" "$ORCH_LOG_FILE" "$VISION_LOG_FILE" 2>/dev/null | \
  awk -v orch_pat="$ORCH_SUMMARY_PATTERN" \
      -v gw_pat="$GATEWAY_SUMMARY_PATTERN" \
      -v vista_pat="$VISION_SUMMARY_PATTERN" \
      -v console_level="$ROBOT_CONSOLE_LEVEL" '
    function is_demo_line(s) {
      return (s ~ /\[DEMO\]/ || s ~ /DEMO SUCCESS/ || s ~ /DEMO FAILED/ || s ~ /^━/ ||
              s ~ /^target        :/ || s ~ /^result        :/ || s ~ /^final_state   :/ ||
              s ~ /^reason        :/ || s ~ /^next_state    :/ || s ~ /^preview       :/ ||
              s ~ /^waiting       :/)
    }
    function is_warn_error(s) {
      return (s ~ /WARN/ || s ~ /WARNING/ || s ~ /ERROR/ || s ~ /FATAL/ || s ~ /FAILED/)
    }
    function is_key_state(s) {
      return (s ~ /\[ORCH\] STATE/ || s ~ /SERVICE_READY/ || s ~ /\[VISTA\] READY/ ||
              s ~ /gateway online/ || s ~ /mqtt connected/ || s ~ /mqtt disabled/)
    }
    /^==> .* <==$/ {
      src=$0
      next
    }
    {
      if ($0 ~ /resizeWindow/ && $0 ~ /No UI backends available/) {
        if (!preview_warned) {
          preview_warned=1
          preview_unavailable=1
          print "[vista] [DEMO][PREVIEW] unavailable: no UI backend, continuing without preview"
        }
        next
      }
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
      if (console_level == "debug") {
        show=1
      } else if (console_level == "demo") {
        show=(is_demo_line($0) || is_warn_error($0))
      } else {
        show=(is_demo_line($0) || is_warn_error($0) || is_key_state($0))
      }
      if (show) {
        if (preview_unavailable && $0 ~ /\[DEMO\]\[HEALTH\]/) {
          gsub(/preview=n\/a/, "preview=unavailable", $0)
        }
        if (console_level == "debug" && $0 ~ /(mobile_cmd|cmd received|gateway_ack|task_ack|fetch_object|status changed|stop)/) {
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
  prepare_start_run_paths
  show_banner
  stop_all || true

  start_vision_bg
  if ! wait_for_endpoint "vision" "vision_req" "$READY_TIMEOUT_S" "$VISION_READY_EXTRA_S"; then
    tail_last_logs_on_failure "vision" "$VISION_LOG_FILE"
    stop_all || true
    exit 1
  fi

  start_orch_bg
  if ! wait_for_endpoint_group "orchestrator" "$READY_TIMEOUT_S" "$ORCH_READY_EXTRA_S" "orchestrator_task_cmd" "orchestrator_vision_obs"; then
    tail_last_logs_on_failure "orchestrator" "$ORCH_LOG_FILE"
    stop_all || true
    exit 1
  fi

  # Check connection to orchestrator task_cmd socket for mobile_gateway (normal user check)
  if [[ "$STACK_PROFILE" == "full" ]]; then
    local gateway_task_cmd_socket="/tmp/robot_stack/task_cmd.sock"
    headline "mobile_gateway 启动前连接检查"
    mark wait "检查普通用户连接 orchestrator task_cmd UDS 能力..."
    
    if [[ -S "$gateway_task_cmd_socket" ]]; then
      local cmd_user=""
      if [[ "$(id -u)" -eq 0 ]]; then
        cmd_user="sudo -u aidlux "
      fi
      local conn_err
      conn_err=$(${cmd_user}/usr/bin/python3 -c "
import socket, sys
try:
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(2.0)
    s.connect('$gateway_task_cmd_socket')
    s.close()
    sys.exit(0)
except Exception as e:
    print(f'{type(e).__name__}: {e}', end='')
    sys.exit(1)
" 2>&1 || true)
      
      if [[ -n "$conn_err" ]]; then
        mark err "连接检查失败："
        mark err "  socket path: $gateway_task_cmd_socket"
        mark err "  ls -l parent: $(ls -l \"$(dirname "$gateway_task_cmd_socket")\" 2>&1 || true)"
        mark err "  stat socket: $(stat "$gateway_task_cmd_socket" 2>&1 || true)"
        mark err "  connect error: $conn_err"
        mark err "  提示：请检查 UDS socket 文件拥有者及权限是否正确（当前由 sudo 启动可能导致普通用户无权访问）"
        stop_all || true
        exit 1
      else
        mark ok "普通用户连接 orchestrator task_cmd 成功！"
      fi
    else
      mark err "UDS socket 文件不存在或不是 socket: $gateway_task_cmd_socket"
      mark err "  ls -l parent: $(ls -l \"$(dirname "$gateway_task_cmd_socket")\" 2>&1 || true)"
      stop_all || true
      exit 1
    fi
  fi

  start_gateway_bg
  if ! wait_for_gateway_ready "$READY_TIMEOUT_S" "$GATEWAY_READY_EXTRA_S"; then
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
  prepare_latest_run_paths
  show_banner
  stop_all
  status_all
  run_latest_summary
}

main() {
  local action="${1:-start}"
  case "$action" in
    dryrun|dry_run|full|sc171_board|windows_dev)
      STACK_PROFILE="$action"
      shift || true
      action="${1:-start}"
      ;;
  esac
  apply_profile_defaults
  case "$action" in
    start|on|up|run|开启|开)
      start_stack
      ;;
    stop|off|down|结束|关)
      stop_stack
      ;;
    status|状态)
      prepare_latest_run_paths
      status_all
      ;;
    *)
      echo "用法："
      echo "  ./start_robot_stack.sh        # 开启（默认）"
      echo "  ./start_robot_stack.sh full   # 使用 full/sc171_board 配置开启"
      echo "  ./start_robot_stack.sh dryrun # 使用 dry_run 配置开启"
      echo "  ./start_robot_stack.sh stop   # 结束"
      echo "  ./start_robot_stack.sh status # 查看状态"
      exit 1
      ;;
  esac
}

main "$@"
