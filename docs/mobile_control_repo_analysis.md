# Mobile Control Repo Analysis

## Step 1. Workspace Scan Summary

This repository is already a three-service robot stack built around TCP JSONL IPC:

- `Voice/voice_service` produces `task_cmd` and consumes `task_ack` / `tts_event`
- `orchestrator/orchestrator_service` owns the runtime loop, task state machine, UART bridge, and task-level safety logic
- `VISTA/vision_module` consumes `vision_req` and produces `vision_obs`

Top-level references:

- `README.md`
- `orchestrator/README.md`
- `VISTA/INTERFACES.md`

## Core Entrypoints

### Orchestrator

- Process entry: `orchestrator/orchestrator_service/app/main.py`
- Service runner: `orchestrator/orchestrator_service/runtime/service.py`
- State machine core: `orchestrator/orchestrator_service/runtime/state_machine.py`

`main()` calls `run_orchestrator_service(CONFIG)`, which creates `OrchestratorService` and starts `run_forever()`.

The real inbound task path is:

1. `JsonlInboundServer` receives `task_cmd`
2. `OrchestratorService._drain_task_cmds()`
3. `TaskCmd.from_dict(...)`
4. `OrchestratorCore.handle_task_cmd(...)`

### Voice

- Process entry: `Voice/voice_service/app/main.py`
- Command interpretation: `Voice/voice_service/runtime/commands.py`
- IPC protocol helpers: `Voice/voice_service/ipc/protocol.py`

Voice already acts as an upstream adapter from local ASR text into canonical `task_cmd`.

### Vision

- Process entry: `VISTA/vision_module/app/app.py`
- Protocol doc: `VISTA/INTERFACES.md`
- Protocol types: `VISTA/vision_module/ipc/protocol.py`

## Existing IPC / JSONL / Event Flow

Confirmed default links:

- `task_cmd_in`: `127.0.0.1:9001`
- `vision_obs_in`: `127.0.0.1:9002`
- `vision_req_out`: `127.0.0.1:9003`
- `tts_event_out`: `127.0.0.1:9011`
- `task_ack_out`: `127.0.0.1:9012`

Relevant files:

- `orchestrator/orchestrator_service/config/schema.py`
- `orchestrator/orchestrator_service/config/board_config.py`
- `orchestrator/orchestrator_service/ipc/protocol.py`
- `orchestrator/orchestrator_service/ipc/transport.py`

The repository already has reusable building blocks:

- `JsonlInboundServer` and `JsonlClientSender` for TCP/UDS JSONL IPC
- `task_cmd` and `task_ack` datamodels with `cmd_id`, `session_id`, `epoch`, `accepted`, `state`, `reason`
- structured run artifacts such as `timeline.jsonl`, `ipc.jsonl`, `state_blocks.jsonl`, `heartbeat.jsonl`

## Orchestrator State And Status Sources

Confirmed status sources inside the current implementation:

- synchronous command accept/reject via `task_ack`
- coarse runtime state snapshots via `state_blocks.jsonl`
- low-frequency health via `heartbeat.jsonl`
- detailed event stream via `timeline.jsonl`

This matters because the new mobile gateway does not need to patch deep orchestrator logic to build a mobile-facing status channel.

## Recommended Mobile Gateway Insertion Point

The safest insertion point is the task-level IPC boundary, not the state machine internals:

- northbound: new mobile gateway accepts `mobile_cmd`
- southbound: gateway maps into existing `task_cmd`
- reverse path: gateway consumes `task_ack` and optionally watches orchestrator state snapshots

Why this path is recommended:

- it reuses the narrowest stable contract already in production
- it preserves board-side closed-loop behavior inside orchestrator
- it avoids coupling mobile concerns to vision envelopes or UART details
- it lets mobile become a parallel upstream entry, similar to Voice, instead of a rewrite of orchestration

## Modules That Should Not Be Heavily Modified

These files are central and should stay mostly unchanged:

- `orchestrator/orchestrator_service/runtime/state_machine.py`
- `orchestrator/orchestrator_service/runtime/service.py`
- `orchestrator/orchestrator_service/ipc/protocol.py`
- `orchestrator/orchestrator_service/bridge/uart_bridge.py`
- `VISTA/vision_module/app/app.py`
- `VISTA/vision_module/ipc/protocol.py`

## Minimal-Intrusion Integration Path

Recommended path for this round:

1. Add a board-side `mobile_gateway` adapter layer near orchestrator.
2. Keep mobile northbound protocol separate from southbound `task_cmd`.
3. Translate:
   - `fetch_object` -> `FIND`
   - `go_home` -> `RETURN`
   - `stop` -> `STOP`
   - `resume` / `retry_search` -> gateway-level replay of the last high-level task
4. Reuse TCP JSONL for local mock integration first.
5. Use mock backend for offline loop validation and optional orchestrator bridge mode for real integration.

## Confirmed Vs Inferred

Confirmed in code:

- orchestrator entrypoint and service loop
- `task_cmd` / `task_ack` protocol
- TCP JSONL transport implementation
- structured state snapshot outputs

Inferred and therefore isolated behind the new gateway layer:

- exact mobile northbound protocol
- resume semantics
- retry semantics
- MQTT/WebSocket topic layout
- whether future mobile runtime should coexist with Voice on the same `task_ack_out`

