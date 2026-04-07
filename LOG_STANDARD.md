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
| **Stdout** | StreamHandler with format `%(asctime)s \| %(levelname)-5s \| %(message)s` |
| **File** | None (stdout only) |
| **Mode** | Fixed INFO level |
| **Module Tag** | Logger name: `AppLayer`, embedded in messages like `🔄`, `🎯`, `💤` |
| **IPC Logging** | Lambda callbacks: `lambda x: log.info(f"[IPC-RX] {x['msg']}")` |

### 2.3 Voice

| Aspect | Pattern |
|--------|---------|
| **Stdout** | JSON via `jlog()` - writes JSON to stdout/stderr |
| **File** | None (stdout only) |
| **Mode** | `should_emit()` filter - "full" (all) or "concise" (filtered) |
| **Module Tag** | Field `src` in JSON payload: `boot`, `loop`, `oww`, `seg`, `decision`, `tts`, `mic`, etc. |
| **Filtering** | Level-based + source-based filtering |

## 3. Unified Logging Schema

### 3.1 Log Entry Format (JSON)

All modules MUST emit JSON-formatted log entries with the following schema:

```json
{
  "ts": 1234567890.123,
  "level": "info",
  "src": "module_name",
  "msg": "human readable message",
  "data": { "optional": "context fields" }
}
```

### 3.2 Field Definitions

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `ts` | float | Yes | Unix timestamp (wall clock) with millisecond precision |
| `level` | string | Yes | One of: `debug`, `info`, `warn`, `error`, `critical` |
| `src` | string | Yes | Source module identifier (e.g., `orchestrator`, `vista`, `voice`, `ipc`, `uart`) |
| `msg` | string | Yes | Human-readable message (Chinese acceptable) |
| `data` | object | No | Additional structured context for debugging |

### 3.3 Level Guidelines

| Level | Usage |
|-------|-------|
| `debug` | Detailed flow tracing, per-frame metrics, verbose IPC dumps |
| `info` | State transitions, task start/stop, heartbeat, milestone events |
| `warn` | Recoverable anomalies: stale data, retry attempts, queue full |
| `error` | Operation failures: send failed, parse error, exception |
| `critical` | Fatal: service crash, unhandled exception |

### 3.4 Source Module Tags

| Module | Recommended `src` Values |
|--------|--------------------------|
| Orchestrator | `orchestrator`, `state_machine`, `controller`, `ipc`, `uart` |
| VISTA | `vista`, `engine`, `camera`, `inference`, `ipc` |
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
| Development | JSON to stdout | Optional JSONL to `./logs/` |
| Production (AidLux) | JSON to stdout | JSONL to `/data/runs/{timestamp}/` |

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
            "src": record.name,
            "msg": record.getMessage(),
        }
        if hasattr(record, "data"):
            payload["data"] = record.data
        return json.dumps(payload, ensure_ascii=False)

# Usage
logger = logging.getLogger("orchestrator")
handler.setFormatter(JsonFormatter())
```

### 5.2 Backward Compatibility

- Keep existing `RunLogger` for file output (JSONL already compliant)
- Add JSON stdout fallback for container log aggregation
- Migrate VISTA and Voice to use Python `logging` module

## 6. IPC Logging Convention

All inter-module messages MUST log both directions:

```json
{
  "ts": 1234567890.123,
  "level": "info",
  "src": "ipc",
  "msg": "vision_req sent",
  "data": {
    "channel": "vision_req_out",
    "session_id": "abc123",
    "epoch": 1,
    "sent": true
  }
}
```

## 7. Migration Checklist

- [ ] Orchestrator: Already compliant (RunLogger outputs JSONL)
- [ ] VISTA: Add `data` field to log entries, migrate to JSON format
- [ ] Voice: Replace `jlog()` with structured Python logging
- [ ] Add `LOG_MODE` env var support to all modules
- [ ] Configure file rotation in production deployment