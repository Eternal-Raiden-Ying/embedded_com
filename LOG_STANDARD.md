# LOG_STANDARD.md

## 1. Overview

This document defines the unified logging schema for the Robot Stack 2026, designed for ARM/AidLux edge environment deployment.

## 2. Current Audit

### 2.1 Orchestrator

| Aspect | Pattern |
|--------|---------|
| **Stdout** | `logging.basicConfig` with format `%(asctime)s \| %(levelname)s \| %(name)s \| %(message)s` |
| **File** | JSONL files: `events.log`, `{name}.jsonl` (timeline, ipc, state_blocks, heartbeat, etc.) |
| **Mode** | Configurable via `configure_logging(mode)` - "full" (DEBUG) or "concise" (INFO) |
| **Module Tag** | Logger name: `OrchestratorService`, `OrchestratorCore` |
| **RunLogger** | Creates timestamped `runs/` directories with structured JSONL output |

### 2.2 VISTA

| Aspect | Pattern |
|--------|---------|
| **Stdout** | Text console via Python `logging` with format `%(asctime)s \| %(levelname)-5s \| %(name)s \| %(message)s` |
| **Structured Files** | Per-run files under `VISTA/runs/<stack_run_id>/`: `meta.json`, `event.jsonl`, `ipc.jsonl`, optional `heartbeat.jsonl` |
| **Console Mirror** | `VISTA/logs/vision.log` is the stdout/stderr mirror created by launcher scripts |
| **Mode** | `VISION_LOG_MODE=concise/full`; heartbeat off by default and enabled by `VISION_HEARTBEAT_ENABLED=1` |
| **Module Tag** | Logger names such as `vision.runtime`, `vision.engine`, `vision.ipc`, `vision.stage` |
| **IPC Logging** | Unified into `ipc.jsonl` with fixed fields and ordered top-level columns |

### 2.3 Voice

| Aspect | Pattern |
|--------|---------|
| **Stdout** | JSON via `jlog()` - writes JSON to stdout/stderr |
| **File** | None (stdout only) |
| **Mode** | `should_emit()` filter - "full" (all) or "concise" (filtered) |
| **Module Tag** | Field `src` in JSON payload: `boot`, `loop`, `oww`, `seg`, `decision`, `tts`, `mic`, etc. |
| **Filtering** | Level-based + source-based filtering |

## 3. Unified Logging Schema

### 3.1 Structured File Format

Structured run artifacts MUST use JSON or JSONL.

- `meta.json` stores one run's startup metadata and effective configuration
- `event.jsonl` stores the ordered runtime event stream
- `ipc.jsonl` stores ordered inter-process communication records
- `heartbeat.jsonl` is optional and low-frequency; disabled by default in VISTA

Console stdout MAY remain human-readable text while structured files remain machine-readable.

Common top-level fields for JSONL records:

```json
{
  "ts": 1234567890.123,
  "level": "info",
  "module": "vision",
  "stack_run_id": "run_20260413_123456_ab12cd",
  "data": { "optional": "context fields" }
}
```

### 3.2 Field Definitions

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `ts` | float | Yes | Unix timestamp (wall clock) with millisecond precision |
| `level` | string | No | One of: `debug`, `info`, `warn`, `error`, `critical` |
| `module` | string | Yes | Module identifier such as `vision`, `orch`, `voice` |
| `stack_run_id` | string | Yes | Run identifier shared across one stack execution |
| `data` | object | No | Additional structured context for debugging |

### 3.3 VISTA Event Schema

`event.jsonl` field order:

1. `ts`
2. `level`
3. `module`
4. `stack_run_id`
5. `event`
6. `stage`
7. `mode`
8. `trigger`
9. `session_id`
10. `req_id`
11. `epoch`
12. `interaction_id`
13. `data`

Typical VISTA events:

- `SERVICE_STARTING`
- `SERVICE_READY`
- `SERVICE_STOPPING`
- `SERVICE_STOPPED`
- `VISION_REQ`
- `VISION_STOP`
- `STAGE_TRANSITION`
- `INTERACTION_RESPONSE_HANDLED`
- `INTERACTION_STATE_CHANGED`
- backend events such as `BACKEND_LIFECYCLE_CHANGED`, `BACKEND_MODE_CHANGED`, `CAPABILITY_CHANGED`, `BACKEND_FAILURE`

### 3.4 VISTA IPC Schema

`ipc.jsonl` field order:

1. `ts`
2. `level`
3. `module`
4. `stack_run_id`
5. `direction`
6. `channel`
7. `event`
8. `msg_type`
9. `session_id`
10. `req_id`
11. `epoch`
12. `ok`
13. `peer`
14. `error`
15. `data`

Typical VISTA IPC events:

- `listening`
- `recv_ok`
- `received`
- `enqueue_ok`
- `send_attempt`
- `connected`
- `send_ok`
- `connect_failed`
- `send_failed`
- `queue_drop_oldest`
- `invalid_json`

### 3.5 Level Guidelines

| Level | Usage |
|-------|-------|
| `debug` | Detailed flow tracing, per-frame metrics, verbose IPC dumps |
| `info` | State transitions, task start/stop, heartbeat, milestone events |
| `warn` | Recoverable anomalies: stale data, retry attempts, queue full |
| `error` | Operation failures: send failed, parse error, exception |
| `critical` | Fatal: service crash, unhandled exception |

### 3.6 Source Module Tags

| Module | Recommended logger/module tags |
|--------|-------------------------------|
| Orchestrator | `orch`, `orchestrator`, `state_machine`, `controller`, `ipc`, `uart` |
| VISTA | `vision`, `vision.runtime`, `vision.engine`, `vision.stage`, `vision.ipc`, `vision.camera.mock` |
| Voice | `voice`, `vad`, `asr`, `kws`, `tts`, `mic` |

## 4. Configuration

### 4.1 Log Mode

Two modes supported via environment variable `LOG_MODE`:

| Mode | Level | Output Volume |
|------|-------|----------------|
| `concise` | INFO + WARN + ERROR | Minimal, milestone events only |
| `full` | DEBUG + INFO + WARN + ERROR | Verbose, for debugging |

### 4.2 Output Destinations

| Environment | Stdout | File |
|-------------|--------|------|
| Development | Human-readable text to stdout | Structured JSON/JSONL under `runs/<stack_run_id>/`, optional console mirror under `logs/` |
| Production (AidLux) | Human-readable text to stdout | Structured JSON/JSONL under `/data/runs/<stack_run_id>/`, optional console mirror under `/data/logs/` |

### 4.3 Rotation Policy

- **Max file size**: 10MB per JSONL file
- **Max retention**: 7 days or 100 runs (whichever first)
- **Compression**: Gzip rotated files

## 5. Implementation Guidelines

### 5.1 Python Logging Integration

```python
import logging
import json

class JsonFormatter(logging.Formatter):
    def format(self, record):
        payload = {
            "ts": record.created + record.msecs / 1000,
            "level": record.levelname.lower(),
            "module": getattr(record, "module_name", record.name),
            "stack_run_id": getattr(record, "stack_run_id", ""),
            "event": getattr(record, "event_name", "LOG"),
        }
        if hasattr(record, "data"):
            payload["data"] = record.data
        return json.dumps(payload, ensure_ascii=False)

# Usage
logger = logging.getLogger("orchestrator")
handler.setFormatter(JsonFormatter())
```

### 5.2 VISTA Writer Contract

- VISTA console remains text only
- VISTA structured files are written through a common `RunLogger`
- Callers pass dictionaries; writer normalizes field order before output
- Unknown extra fields are folded into `data`
- `meta.json` is written once per run
- `event.jsonl` and `ipc.jsonl` are the default required structured files
- `heartbeat.jsonl` is optional and should be enabled only for debugging slow or silent services

## 6. IPC Logging Convention

All inter-module messages SHOULD be reflected in `ipc.jsonl` for both directions:

```json
{
  "ts": 1234567890.123,
  "level": "info",
  "module": "vision",
  "stack_run_id": "run_20260413_123456_ab12cd",
  "direction": "TX",
  "channel": "obs_out",
  "event": "send_ok",
  "msg_type": "vision_obs",
  "session_id": "abc123",
  "req_id": "req_001",
  "epoch": 1,
  "ok": true,
  "data": {
    "stage": "SEARCH",
    "mode": "TRACK_LOCAL",
    "status": "RUNNING"
  }
}
```

## 7. Migration Checklist

- [ ] Orchestrator: align structured file names with `meta/event/ipc`
- [x] VISTA: console text plus structured `meta.json`, `event.jsonl`, `ipc.jsonl`, optional `heartbeat.jsonl`
- [ ] Voice: replace ad-hoc JSON stdout/file patterns with the same `meta/event/ipc` run layout
- [ ] Add `LOG_MODE` env var support to all modules
- [ ] Configure file rotation in production deployment
