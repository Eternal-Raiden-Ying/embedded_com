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

The formal runtime path is now fully inside this repository:

- `orchestrator/orchestrator_service/mobile_gateway/`

`~/mobile_gateway_cloud/cloud_mqtt_bridge.py` should no longer be treated as the main workflow.

## Common Preparation

Formal northbound identity and topics are now fixed:

- `robot_id = SC171`
- `robot/v1/SC171/mobile/cmd`
- `robot/v1/SC171/mobile/ack`
- `robot/v1/SC171/mobile/status`
- `robot/v1/SC171/heartbeat`

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

Formal mini-program command shape:

```json
{
  "type": "mobile_cmd",
  "robot_id": "SC171",
  "cmd_id": "cmd_1234567890",
  "session_id": "sess_1234567890",
  "epoch": 1,
  "cmd": "fetch_object",
  "target": "apple",
  "text": "拿苹果",
  "source": "wechat_miniprogram",
  "ts": 1713945600.0
}
```

Backward compatibility input still accepted for tests only:

```json
{
  "type": "FIND_AND_PICK",
  "target": "apple"
}
```

## Mode 1. Mock

Run:

```bash
bash tools/run_mobile_gateway_mock.sh
```

Expected flow:

1. MQTT-style or stdout `gateway_ack` with `kind=gateway_ack`
2. status with `kind=status` and `state=submitted`
3. Orchestrator-style `task_ack` or mock `task_ack` with `kind=task_ack`
4. status with `kind=status` and `state=accepted`
5. later progress states such as `searching`, `approaching`, `completed`, `idle`

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

Start Orchestrator first:

```bash
cd /home/aidlux/embedded_com/orchestrator
export ORCH_TASK_CMD_IN_HOST=127.0.0.1
export ORCH_TASK_CMD_IN_PORT=9001
export ORCH_TASK_ACK_OUT_HOST=127.0.0.1
export ORCH_TASK_ACK_OUT_PORT=9012
export ORCH_SERIAL_DRY_RUN=1
export ORCH_TTS_EVENT_OUT_TRANSPORT=disabled
python3 -m orchestrator_service.app.main
```

Then start the formal gateway with either fixed command below.

From repo root:

```bash
cd /home/aidlux/embedded_com
PYTHONPATH=/home/aidlux/embedded_com/orchestrator \
python3 -m orchestrator_service.mobile_gateway.runtime.service \
  --config configs/mobile_gateway.mqtt.yaml
```

From `orchestrator/`:

```bash
cd /home/aidlux/embedded_com/orchestrator
python3 -m orchestrator_service.mobile_gateway.runtime.service \
  --config ../configs/mobile_gateway.mqtt.yaml
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

The smoke sender already emits the formal command format:

- `type=mobile_cmd`
- `robot_id=SC171`
- `cmd_id=cmd_xxx`
- `source=wechat_miniprogram`

When bridge mode is healthy, the expected outbound sequence is:

1. `kind=gateway_ack`
2. `kind=status`
3. `kind=task_ack` when real Orchestrator responds
4. more `kind=status`
5. periodic `kind=heartbeat`

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

Real local runtime file:

- `configs/mobile_gateway.mqtt.yaml`

Create this file locally from the example. Do not commit it.

MQTT message kinds:

- `robot/v1/SC171/mobile/ack`
  - `kind=gateway_ack`
  - `kind=task_ack`
- `robot/v1/SC171/mobile/status`
  - `kind=status`
- `robot/v1/SC171/heartbeat`
  - `kind=heartbeat`

## EMQX phone_test Example

Publish to:

- `robot/v1/SC171/mobile/cmd`

Payload:

```json
{
  "type": "mobile_cmd",
  "robot_id": "SC171",
  "cmd_id": "cmd_demo_001",
  "session_id": "sess_demo_001",
  "epoch": 1,
  "cmd": "fetch_object",
  "target": "apple",
  "text": "拿苹果",
  "source": "wechat_miniprogram",
  "ts": 1713945600.0
}
```

Expected outbound sequence:

1. `robot/v1/SC171/mobile/ack` with `kind=gateway_ack`
2. `robot/v1/SC171/mobile/ack` with `kind=task_ack`
3. `robot/v1/SC171/mobile/status` with `kind=status`
4. `robot/v1/SC171/heartbeat` with `kind=heartbeat`

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
