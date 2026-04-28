# Mobile Gateway Runbook

## Runtime Modes

`mobile_gateway` now has two runtime styles:

- `production`
  - formal service layout
  - low-noise logs
  - no raw MQTT payload dumps
  - no raw backend fields in public MQTT payloads
- `debug`
  - keeps the same formal topics and command protocol
  - enables raw MQTT diagnostics
  - can expose `backend_state` and `raw_error`
  - suitable for bring-up, smoke, and fault tracing

The backend transport mode remains separate:

- `mock`
- `tcp_no_ack`
- `orchestrator_tcp`

## Fixed Topics

- `robot/v1/SC171/mobile/cmd`
- `robot/v1/SC171/mobile/ack`
- `robot/v1/SC171/mobile/status`
- `robot/v1/SC171/heartbeat`

These topics are fixed and unchanged.

## Recommended Startup

### 1. Start Orchestrator

From repo root:

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

### 2. Start Mobile Gateway

Formal recommended command:

```bash
cd /home/aidlux/embedded_com
PYTHONPATH=/home/aidlux/embedded_com/orchestrator \
/usr/bin/python3 -m orchestrator_service.mobile_gateway.runtime.service \
  --config configs/mobile_gateway.mqtt.yaml
```

`configs/mobile_gateway.mqtt.yaml` is a local board-side runtime file and should not contain secrets in git commits.

## Configuration Files

- example config: `configs/mobile_gateway.mqtt.example.yaml`
- local board config: `configs/mobile_gateway.mqtt.yaml`
- example env: `configs/mqtt_cloud.env.example`

Important runtime knobs:

- `runtime.mode`
- `runtime.log_level`
- `runtime.heartbeat_log_interval_s`
- `runtime.suppress_heartbeat_success_log`
- `runtime.enable_raw_mqtt_debug`
- `runtime.enable_legacy_command_compat`
- `runtime.cmd_dedup_cache_size`

## Formal Command Contract

Formal mini-program commands:

- `fetch_object`
- `stop`

Example:

```json
{
  "type": "mobile_cmd",
  "robot_id": "SC171",
  "cmd_id": "wx_1777293209382",
  "session_id": "wx_session_001",
  "epoch": 1,
  "cmd": "fetch_object",
  "target": "apple",
  "text": "拿苹果",
  "source": "wechat_miniprogram",
  "ts": 1777293208.5
}
```

## ACK, Status, Heartbeat Semantics

`gateway_ack`

- gateway has accepted or rejected the northbound command format

`task_ack`

- Orchestrator has accepted or rejected the southbound task

`status`

- unified mini-program progress stream
- suitable for direct UI message display

`heartbeat`

- liveness and current mode/state summary

## Production Log Strategy

Production logs keep:

- `gateway online`
- `mqtt connected`
- `mqtt disconnected`
- `cmd received`
- `gateway_ack sent`
- `task_cmd forwarded`
- `task_ack forwarded`
- `status changed`
- `heartbeat running`
- `error summary`

Production logs suppress:

- per-heartbeat MQTT publish success lines
- raw MQTT payload dumps
- excessive success chatter for every low-level publish

Heartbeat summary style:

```text
heartbeat running | count=30 | last_state=idle
```

## Debug Log Strategy

Debug mode additionally keeps:

- raw inbound MQTT payload
- raw outbound MQTT payload
- backend diagnostic fields
- compatibility behavior visibility

## Duplicate `cmd_id` Behavior

The gateway keeps a recent `cmd_id` cache, default `64`.

When a duplicate command arrives:

- `gateway_ack` can be sent again
- no second forward to Orchestrator
- no duplicate status spam

## Acceptance With Mini-Program

### Fetch Apple

Publish to:

- `robot/v1/SC171/mobile/cmd`

Payload:

```json
{
  "type": "mobile_cmd",
  "robot_id": "SC171",
  "cmd_id": "wx_1777293209382",
  "session_id": "wx_session_001",
  "epoch": 1,
  "cmd": "fetch_object",
  "target": "apple",
  "text": "拿苹果",
  "source": "wechat_miniprogram",
  "ts": 1777293208.5
}
```

Expected northbound sequence:

1. `mobile/ack` with `kind=gateway_ack`
2. `mobile/ack` with `kind=task_ack`
3. `mobile/status` with `state=submitted`
4. `mobile/status` with `state=accepted`
5. `mobile/status` with `state=searching`
6. later `running`, `idle`, or `error` depending on backend progress

### Stop

Publish:

```json
{
  "type": "mobile_cmd",
  "robot_id": "SC171",
  "cmd_id": "wx_1777293209383",
  "session_id": "wx_session_001",
  "epoch": 1,
  "cmd": "stop",
  "source": "wechat_miniprogram",
  "ts": 1777293215.0
}
```

Expected northbound sequence:

1. `mobile/ack` with `kind=gateway_ack`
2. `mobile/ack` with `kind=task_ack`
3. `mobile/status` showing `任务已停止`
4. `heartbeat` continues with `state=stopped` or later `idle`

## Diagnostic Tools

These scripts are still kept and are now documented as diagnostics, not production entrypoints.

In `tools/`:

- `mock_mobile_sender.py`
- `mock_status_listener.py`
- `smoke_mobile_to_orchestrator.py`
- `run_mobile_gateway_mock.sh`
- `run_mobile_gateway_tcp_no_ack.sh`
- `run_mobile_gateway_real_tcp.sh`

In `orchestrator/orchestrator_service/examples/`:

- `mock_task_cmd_sender.py`
- `mock_vision_obs_sender.py`
- `control_module_smoke_test.py`
- `uart_protocol_smoke_test.py`

## Test Command

Run from repo root:

```bash
python3 -m unittest \
  tests.test_command_protocol \
  tests.test_gateway_mapping \
  tests.test_mock_flow \
  tests.test_real_protocol_mapping
```
