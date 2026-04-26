# Mobile Gateway Real Bridge Notes

## Confirmed Real Orchestrator Paths

Entrypoints:

- `orchestrator/orchestrator_service/app/main.py`
- `orchestrator/orchestrator_service/runtime/service.py`

Task ingress:

- `OrchestratorService._drain_task_cmds()` in `orchestrator/orchestrator_service/runtime/service.py`

State machine:

- `orchestrator/orchestrator_service/runtime/state_machine.py`

Protocol / transport:

- `orchestrator/orchestrator_service/ipc/protocol.py`
- `orchestrator/orchestrator_service/ipc/transport.py`

## task_cmd Input Configuration

Default definition:

- `orchestrator/orchestrator_service/config/schema.py`

Default values:

- transport: `tcp`
- host: `127.0.0.1`
- port: `9001`

Environment overrides:

- `ORCH_TASK_CMD_IN_TRANSPORT`
- `ORCH_TASK_CMD_IN_HOST`
- `ORCH_TASK_CMD_IN_PORT`
- `ORCH_TASK_CMD_IN_UDS`

## task_ack Output Configuration

Default definition:

- `orchestrator/orchestrator_service/config/schema.py`

Default values:

- transport: `tcp`
- host: `127.0.0.1`
- port: `9012`

Environment overrides:

- `ORCH_TASK_ACK_OUT_TRANSPORT`
- `ORCH_TASK_ACK_OUT_HOST`
- `ORCH_TASK_ACK_OUT_PORT`
- `ORCH_TASK_ACK_OUT_UDS`
- `ORCH_TASK_ACK_SEND_MODE`

## state_blocks.jsonl Default Path

Default orchestrator run root:

- `orchestrator/orchestrator_service/config/board_config.py`
- runtime default resolves to `orchestrator/runs`

Per-run snapshot file:

- `orchestrator/runs/run_<timestamp>_<id>/state_blocks.jsonl`

The gateway now supports either:

- `MOBILE_GATEWAY_STATE_BLOCKS_PATH=/abs/path/to/state_blocks.jsonl`
- or scanning the configured orchestrator runs root for the latest run directory

## Gateway Environment Variables Used In Real Bridge Mode

Northbound gateway listener:

- `MOBILE_GATEWAY_CMD_IN_HOST`
  - mobile command TCP listener host
- `MOBILE_GATEWAY_CMD_IN_PORT`
  - mobile command TCP listener port
- `MOBILE_GATEWAY_STATUS_OUT_HOST`
  - optional status sink host
- `MOBILE_GATEWAY_STATUS_OUT_PORT`
  - optional status sink port

Southbound orchestrator bridge:

- `MOBILE_GATEWAY_BACKEND`
  - `mock` / `tcp_no_ack` / `orchestrator_tcp`
- `MOBILE_GATEWAY_ORCH_TASK_CMD_HOST`
  - orchestrator `task_cmd_in` host
- `MOBILE_GATEWAY_ORCH_TASK_CMD_PORT`
  - orchestrator `task_cmd_in` port
- `MOBILE_GATEWAY_ORCH_TASK_ACK_HOST`
  - host where gateway listens for orchestrator `task_ack_out`
- `MOBILE_GATEWAY_ORCH_TASK_ACK_PORT`
  - port where gateway listens for orchestrator `task_ack_out`
- `MOBILE_GATEWAY_STATE_BLOCKS_PATH`
  - optional direct path to a specific `state_blocks.jsonl`
- `MOBILE_GATEWAY_ORCH_RUNS_DIR`
  - orchestrator runs root used when scanning for the latest `state_blocks.jsonl`

MQTT northbound adapter:

- `MOBILE_GATEWAY_MQTT_ENABLED`
  - enable MQTT adapter
- `MOBILE_GATEWAY_MQTT_BROKER_HOST`
  - broker hostname
- `MOBILE_GATEWAY_MQTT_BROKER_PORT`
  - broker port
- `MOBILE_GATEWAY_MQTT_TRANSPORT`
  - `websocket` or `tcp`
- `MOBILE_GATEWAY_MQTT_USE_TLS`
  - whether TLS is enabled
- `MOBILE_GATEWAY_MQTT_WEBSOCKET_PATH`
  - websocket path such as `/mqtt`
- `MOBILE_GATEWAY_MQTT_USERNAME`
  - optional username
- `MOBILE_GATEWAY_MQTT_PASSWORD`
  - optional password
- `MOBILE_GATEWAY_MQTT_CLIENT_ID`
  - MQTT client id
- `MOBILE_GATEWAY_MQTT_CMD_TOPIC`
  - command topic template
- `MOBILE_GATEWAY_MQTT_ACK_TOPIC`
  - ack topic template
- `MOBILE_GATEWAY_MQTT_STATUS_TOPIC`
  - status topic template
- `MOBILE_GATEWAY_MQTT_HEARTBEAT_TOPIC`
  - heartbeat topic template

## Can Orchestrator Run Without Real Hardware

Repository evidence says yes, at least for board-side dry-run control transport:

- `ORCH_SERIAL_DRY_RUN=1` is supported in `orchestrator/orchestrator_service/config/board_config.py`
- `start_robot_stack.sh` exposes `STACK_PROFILE="dryrun"`
- `orchestrator/orchestrator_nohw_demo.py` explicitly starts orchestrator in UART dry-run mode and injects mock traffic

What dry-run means here:

- UART writes are suppressed or echoed instead of sent to real hardware
- `task_cmd` / `task_ack` / `vision_req` IPC remains real TCP JSONL

## Real Transport Validation Status In This Workspace

Code-level bridge is ready for:

- `task_cmd` forwarding to real orchestrator
- `task_ack` listening
- `state_blocks.jsonl` observation

What was confirmed directly:

- configuration locations
- default ports
- dry-run support in code and scripts
- repository-provided no-hardware demo path

What was not fully executed end-to-end inside this sandbox:

- a live TCP bind-and-connect session between `mobile_gateway` and a running orchestrator instance

Reason:

- the current execution sandbox blocks local socket binding during automated tests and local smoke runs

## Minimum Real-Transport Smoke Path

When running outside the sandbox, the recommended smoke path is:

1. start orchestrator with `ORCH_SERIAL_DRY_RUN=1`
2. route `ORCH_TASK_ACK_OUT_*` to the gateway listener port
3. start `mobile_gateway` in `orchestrator_tcp`
4. use `tools/smoke_mobile_to_orchestrator.py`
5. observe:
   - gateway `mobile_status`
   - orchestrator `task_ack`
   - orchestrator `state_blocks.jsonl`

If ACK fan-in is not ready yet, use `tcp_no_ack` and verify that:

- the gateway emits a correctly shaped southbound `task_cmd`
- orchestrator receives it on `task_cmd_in`
