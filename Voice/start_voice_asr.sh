#!/bin/bash
# -*- coding: utf-8 -*-

# Determine repo root structural directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VOICE_REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Exports
export VOICE_REPO_ROOT
export PYTHONPATH="$VOICE_REPO_ROOT:$VOICE_REPO_ROOT/common:$VOICE_REPO_ROOT/orchestrator"

PID_FILE="$SCRIPT_DIR/voice.pid"
LOG_FILE="$SCRIPT_DIR/voice.out"

ACTION="$1"
PROFILE="$2"
DRY_RUN_TEXT_FLAG="$3"

# Read extra arguments
shift
shift
EXTRA_ARGS="$@"

usage() {
    echo "Usage: $0 {start|stop|restart|status|tail} [profile_path] [--dry-run-text]"
    exit 1
}

if [ -z "$ACTION" ]; then
    usage
fi

get_pid() {
    if [ -f "$PID_FILE" ]; then
        cat "$PID_FILE"
    else
        echo ""
    fi
}

is_running() {
    local pid=$(get_pid)
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        return 0
    else
        return 1
    fi
}

start_service() {
    if is_running; then
        echo "Voice Gateway is already running (PID: $(get_pid))"
        return 0
    fi

    echo "Starting Voice Gateway with profile: ${PROFILE:-default}..."
    
    # Construct args list
    local args=""
    if [ -n "$PROFILE" ]; then
        args="$args --profile $PROFILE"
    fi
    if [ "$DRY_RUN_TEXT_FLAG" = "--dry-run-text" ]; then
        args="$args --dry-run-text"
    fi
    args="$args $EXTRA_ARGS"

    cd "$SCRIPT_DIR" || exit 1
    # Run in background and redirect output
    nohup python3 -m voice_service.app.main $args > "$LOG_FILE" 2>&1 &
    local new_pid=$!
    echo "$new_pid" > "$PID_FILE"
    
    # Wait briefly and verify start
    sleep 0.5
    if kill -0 "$new_pid" 2>/dev/null; then
        echo "Voice Gateway started successfully (PID: $new_pid)"
        echo "Logs redirected to $LOG_FILE"
    else
        echo "Voice Gateway failed to start. Last log lines:"
        tail -n 10 "$LOG_FILE"
        exit 1
    fi
}

stop_service() {
    if ! is_running; then
        echo "Voice Gateway is not running"
        # Clean up stale pid file if present
        rm -f "$PID_FILE"
        return 0
    fi

    local pid=$(get_pid)
    echo "Stopping Voice Gateway (PID: $pid)..."
    kill "$pid"
    
    # Grace wait
    for i in {1..30}; do
        if kill -0 "$pid" 2>/dev/null; then
            sleep 0.1
        else
            break
        fi
    done
    
    if kill -0 "$pid" 2>/dev/null; then
        echo "Force killing Voice Gateway (PID: $pid)..."
        kill -9 "$pid"
    fi
    rm -f "$PID_FILE"
    echo "Voice Gateway stopped"
}

status_service() {
    if is_running; then
        echo "Voice Gateway is running (PID: $(get_pid))"
        exit 0
    else
        echo "Voice Gateway is stopped"
        exit 3
    fi
}

tail_logs() {
    if [ -f "$LOG_FILE" ]; then
        tail -f "$LOG_FILE"
    else
        echo "Log file $LOG_FILE does not exist"
        exit 1
    fi
}

case "$ACTION" in
    start)
        start_service
        ;;
    stop)
        stop_service
        ;;
    restart)
        stop_service
        start_service
        ;;
    status)
        status_service
        ;;
    tail)
        tail_logs
        ;;
    *)
        usage
        ;;
esac
