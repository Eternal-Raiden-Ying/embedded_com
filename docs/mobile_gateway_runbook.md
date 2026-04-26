# Mobile Gateway Runbook

## Supported Runtime Modes

This round supports three gateway modes:

- `mock`
  - fully offline
  - does not require real orchestrator, vision, UART, or hardware
- `tcp_no_ack`
  - sends real southbound `task_cmd` over TCP JSONL
  - does not require `task_ack`
  - useful when only command delivery needs to be verified
- `orchestrator_tcp`
  - sends real southbound `task_cmd`
  - listens for real `task_ack`
  - can also watch orchestrator `state_blocks.jsonl`

## Common Preparation

Start a status listener from repo root:

```bash
python3 tools/mock_status_listener.py --host 127.0.0.1 --port 9102
```

Send commands from repo root:

```bash
python3 tools/mock_mobile_sender.py --host 127.0.0.1 --port 9101 fetch_object apple
python3 tools/mock_mobile_sender.py --host 127.0.0.1 --port 9101 stop
python3 tools/mock_mobile_sender.py --host 127.0.0.1 --port 9101 query_status
```

## Mode 1. Mock

Run:

```bash
bash tools/run_mobile_gateway_mock.sh
```

Expected flow:

1. `submitted`
2. `accepted`
3. `searching`
4. `approaching`
5. `completed`
6. `idle`

This is the default fallback mode for local regression testing.

## Mode 2. tcp_no_ack

Use this when you want to validate that `mobile_gateway` emits a correctly shaped `task_cmd`, but real orchestrator ACK fan-in is not ready yet.

Run:

```bash
bash tools/run_mobile_gateway_tcp_no_ack.sh
```

Important environment variables:

```bash
MOBILE_GATEWAY_BACKEND=tcp_no_ack
MOBILE_GATEWAY_ORCH_TASK_CMD_HOST=127.0.0.1
MOBILE_GATEWAY_ORCH_TASK_CMD_PORT=9001
MOBILE_GATEWAY_ORCH_TASK_ACK_TRANSPORT=disabled
MOBILE_GATEWAY_STATE_BLOCKS_PATH=/path/to/state_blocks.jsonl
```

Recommended usage:

- pair this mode with a separate collector or orchestrator dry-run instance
- use it when `task_ack_out` cannot yet be redirected to the gateway

## Mode 3. orchestrator_tcp

This is the intended real bridge mode.

Gateway side:

```bash
bash tools/run_mobile_gateway_real_tcp.sh
```

If orchestrator ACK should be routed back into the gateway, start orchestrator with:

```bash
ORCH_SERIAL_DRY_RUN=1 \
ORCH_TASK_ACK_OUT_TRANSPORT=tcp \
ORCH_TASK_ACK_OUT_HOST=127.0.0.1 \
ORCH_TASK_ACK_OUT_PORT=9103 \
python3 -m orchestrator_service.app.main
```

And start the gateway with:

```bash
MOBILE_GATEWAY_BACKEND=orchestrator_tcp \
MOBILE_GATEWAY_ORCH_TASK_CMD_HOST=127.0.0.1 \
MOBILE_GATEWAY_ORCH_TASK_CMD_PORT=9001 \
MOBILE_GATEWAY_ORCH_TASK_ACK_TRANSPORT=tcp \
MOBILE_GATEWAY_ORCH_TASK_ACK_HOST=127.0.0.1 \
MOBILE_GATEWAY_ORCH_TASK_ACK_PORT=9103 \
python3 -m orchestrator_service.mobile_gateway.app.main
```

Optional state snapshot bridge:

```bash
MOBILE_GATEWAY_STATE_BLOCKS_PATH=/abs/path/to/orchestrator/runs/run_xxx/state_blocks.jsonl
```

If `MOBILE_GATEWAY_STATE_BLOCKS_PATH` is empty, the gateway falls back to scanning the configured orchestrator runs directory for the latest `run_*/state_blocks.jsonl`.

## Real Orchestrator Notes

The repository already contains dry-run helpers:

- `start_robot_stack.sh` with `STACK_PROFILE="dryrun"`
- `orchestrator/orchestrator_nohw_demo.py`

In dry-run mode, orchestrator can avoid real UART writes by setting:

```bash
ORCH_SERIAL_DRY_RUN=1
```

This still keeps the real `task_cmd` / `task_ack` JSONL control surface alive.

## Smoke Sender

To emit a small fixed command sequence to the gateway:

```bash
python3 tools/smoke_mobile_to_orchestrator.py --host 127.0.0.1 --port 9101
```

The sequence is:

1. `query_status`
2. `fetch_object apple`
3. `stop`
4. `go_home`

## MQTT Configuration Entry

MQTT is optional and northbound only. If enabled, it reuses the same command handler and status payloads.

```bash
MOBILE_GATEWAY_MQTT_ENABLED=1
MOBILE_GATEWAY_MQTT_BROKER_HOST=broker.example.com
MOBILE_GATEWAY_MQTT_BROKER_PORT=443
MOBILE_GATEWAY_MQTT_TRANSPORT=websocket
MOBILE_GATEWAY_MQTT_USE_TLS=1
```

Config template:

- `configs/mobile_gateway.mqtt.example.yaml`

## Tests

Run the current regression set from repo root:

```bash
python3 -m unittest tests.test_command_protocol tests.test_gateway_mapping tests.test_mock_flow
python3 -m unittest tests.test_real_protocol_mapping
```

## Current Limits

- `resume` and `retry_search` are high-level task replays, not fine-grained orchestrator resume
- `orchestrator_tcp` assumes the gateway can become the `task_ack_out` sink or observe state snapshots
- MQTT support is a real adapter skeleton, but it still requires `paho-mqtt` to be installed before connecting
