# VISTA Logging Rules

## Goal

VISTA keeps console output for operators, but all structured debugging data is written into a small fixed set of files under one `runs/<stack_run_id>/` directory.

The default logging target set is:

- `meta.json`: one file per run, static startup config and file locations
- `event.jsonl`: one ordered event stream for VISTA runtime milestones
- `ipc.jsonl`: one ordered IPC stream for communication with other modules
- `heartbeat.jsonl`: optional low-frequency health snapshots, disabled by default
- `logs/vision.log`: plain text console mirror, not a structured data file

Deprecated by default for VISTA:

- `timeline.jsonl`
- `config.jsonl`
- `stage.jsonl`
- `engine.jsonl`
- `camera.jsonl`
- `detections.jsonl`
- `runs/<run_id>/events.log`

Current code note:

- `VistaApp` creates `RunLogger(..., enable_text_events=False)`, so `runs/<run_id>/events.log` is not created in normal VISTA runs.

## Directory Split

- `logs/vision.log` is the current process text log. It is for `tail -f`, service readiness checks, and quick manual inspection.
- `runs/<run_id>/` stores structured artifacts for one specific execution.
- Do not mix structured JSON content into `logs/vision.log`.
- Do not duplicate the same semantic event into multiple structured files.

## Console Rules

- Console output stays human-readable text only.
- Console messages should be concise and milestone-oriented.
- Console messages may include file paths for `run_dir`, `meta.json`, `event.jsonl`, and `ipc.jsonl`.
- Console must not print raw request/response payload dumps every loop.
- Console must not print heartbeat lines by default.

## Structured File Rules

### Current writer entrypoints

- App events: `VistaApp._record_event(...)`
- Stage events: `StageController._emit_event(...) -> VistaApp._record_stage_event(...)`
- Backend events:
  - `ModeController -> VistaApp._record_backend_event(...)`
  - `RuntimeSupervisor` / managers -> `VisionEngine._emit_event(...) -> VistaApp._record_backend_event(...)`
- IPC records: `VistaApp._record_ipc(...)`
- Heartbeat records: `VistaApp._emit_heartbeat_if_needed(...)`

### `meta.json`

Purpose:

- Record one-time startup parameters for this run.

Content:

- service name
- run directory
- project root
- pid file
- console log file
- structured log file locations
- effective runtime config

Write timing:

- Exactly once during startup, before the service enters the main loop.

### `event.jsonl`

Purpose:

- Record the single ordered runtime event stream for VISTA logic and capability changes.

Use it for:

- service lifecycle
- accepted requests
- stage changes
- mode changes
- capability enable/disable
- interaction open/respond/result events
- warnings and failures that matter to flow control

Do not use it for:

- per-frame inference output
- repeated loop spam
- raw transport retries unless they change control flow

Field order:

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

Notes:

- Top-level fields stay stable and short.
- Extra details belong in `data`.
- Event names use uppercase snake case.

Recommended events:

- `SERVICE_STARTING`
- `SERVICE_READY`
- `SERVICE_STOPPING`
- `SERVICE_STOPPED`
- `VISION_REQ`
- `VISION_STOP`
- `STAGE_TRANSITION`
- `INTERACTION_RESPONSE_HANDLED`
- `INTERACTION_STATE_CHANGED`
- `MODE_APPLY_FAILED`
- `FATAL`
- backend events such as `BACKEND_LIFECYCLE_CHANGED`, `BACKEND_MODE_CHANGED`, `BACKEND_RUNTIME_RECONCILED`, `CAPABILITY_CHANGED`, `BACKEND_FAILURE`

Current backend note:

- `BACKEND_DIAGNOSTIC` is reserved by the app-side filter, but there is no active emitter in the current VISTA code path.

### `ipc.jsonl`

Purpose:

- Record all inter-process communication with other modules and sockets.

Use it for:

- listen/connect lifecycle
- send attempts and send failures
- receive success
- invalid json
- queue drop and reconnect warnings

Field order:

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

Notes:

- `direction` is `RX` or `TX`.
- `channel` is the logical channel name such as `req_in` or `obs_out`.
- Transport-specific details such as `transport`, `bind`, `queue_depth`, `fail_count`, and `status` belong in `data`.

### `heartbeat.jsonl`

Purpose:

- Optional low-frequency health snapshot for debugging stuck or silent services.

Default:

- Disabled unless `VISION_HEARTBEAT_ENABLED=1`.

Recommended interval:

- `5.0` seconds or slower.

Use it for:

- current `stage/mode`
- request and observation age
- ready state summary
- current engine capability summary

Current payload note:

- `heartbeat.jsonl` currently includes `req_in`, `obs_out`, and a compact `engine` block under `data`
- it is still a low-frequency health snapshot, not the main execution timeline

Do not use it as the primary runtime timeline.

## Insertion Rules

### Write to `event.jsonl` when

- a request is accepted and interpreted
- `stage` changes
- `mode` changes
- a capability is enabled, disabled, loaded, or released
- the service enters hot standby or idle
- an interaction is opened or resolved
- a result becomes available
- a failure changes behavior or requires investigation

### Write to `ipc.jsonl` when

- a socket begins listening
- a peer connects or reconnection is attempted
- JSONL is received successfully
- a send is queued, attempted, succeeds, or fails
- invalid json is detected
- queue overflow drops data

### Do not log by default when

- the main loop iterates normally
- no state changed
- per-frame detections only differ numerically
- a value can already be reconstructed from the emitted `vision_obs`

## Writer Contract

- Writers accept dictionaries from callers.
- The common writer normalizes field order before writing.
- Python `dict` preserves insertion order, but callers must not rely on that for readability.
- The writer is responsible for final top-level ordering.
- Unknown extra fields are folded into `data` instead of creating new unstable top-level columns.

Current implementation source:

- field order is defined in `common/runtime_logging.py` as `EVENT_FIELD_ORDER`, `IPC_FIELD_ORDER`, and `HEARTBEAT_FIELD_ORDER`

## Environment Variables

- `VISION_LOG_MODE`: `concise` or `full`
- `VISION_LOG_ENABLED`: `1` or `0`
- `VISION_HEARTBEAT_ENABLED`: `1` or `0`
- `VISION_HEARTBEAT_INTERVAL_S`: default `5.0`

## Operator Workflow

- Watch `logs/vision.log` for readiness and immediate failures.
- Use `runs/<run_id>/meta.json` to confirm the active run and output file paths.
- Use `event.jsonl` to reconstruct VISTA control flow.
- Use `ipc.jsonl` to debug communication with orchestrator or local test tools.
- Enable `heartbeat.jsonl` only when the service appears alive but silent.
