# Mobile MQTT Adapter

## Role

`mobile_gateway` uses MQTT only as the northbound transport layer.

It carries:

- inbound `mobile_cmd`
- outbound `gateway_ack`
- outbound `task_ack`
- outbound `status`
- outbound `heartbeat`

It does not define a second protocol.

## Fixed MQTT Contract

- `robot_id = SC171`
- `cmd: robot/v1/SC171/mobile/cmd`
- `ack: robot/v1/SC171/mobile/ack`
- `status: robot/v1/SC171/mobile/status`
- `heartbeat: robot/v1/SC171/heartbeat`

These topics are fixed.

## Inbound

### `robot/v1/SC171/mobile/cmd`

Formal payload:

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

Formal mini-program commands:

- `fetch_object`
- `stop`

Compatibility input:

- `type=FIND_AND_PICK` can still be enabled for transition/debug use

## Outbound ACKs

### `robot/v1/SC171/mobile/ack`

Two `kind` values share the same topic.

`kind=gateway_ack`

- emitted immediately after gateway validation
- means the gateway accepted or rejected the MQTT/mobile command itself

`kind=task_ack`

- emitted after Orchestrator returns a southbound `task_ack`
- means the task has been accepted or rejected by the southbound service

Production ACK fields:

- `type`
- `kind`
- `robot_id`
- `cmd_id`
- `session_id`
- `epoch`
- `accepted`
- `message`
- `error_code`
- `source`
- `ts`

Debug ACK additions:

- `backend_state`

## Outbound Status

### `robot/v1/SC171/mobile/status`

Production status fields:

- `type=mobile_status`
- `kind=status`
- `robot_id`
- `session_id`
- `epoch`
- `state`
- `target`
- `message`
- `progress`
- `command`
- `source`
- `ts`

Formal states:

- `submitted`
- `accepted`
- `searching`
- `running`
- `idle`
- `stopped`
- `error`

Debug status additions:

- `backend_state`
- `raw_error`

## Outbound Heartbeat

### `robot/v1/SC171/heartbeat`

Production heartbeat fields:

- `type=mobile_gateway_heartbeat`
- `kind=heartbeat`
- `robot_id`
- `online`
- `backend_mode`
- `state`
- `session_id`
- `epoch`
- `ts`

Debug heartbeat additions:

- `status_age_s`
- `recent_states`

## Log Behavior

`production` mode:

- logs connection and disconnection events
- logs command receipt and status transitions
- suppresses raw MQTT payload logs
- suppresses per-heartbeat publish success logs

`debug` mode:

- logs MQTT TX/RX with raw payloads
- keeps heartbeat publish traces
- keeps compatibility/diagnostic data visible

## QoS And Retain Defaults

- `cmd`: QoS 1
- `ack`: QoS 1
- `status`: QoS 0
- `heartbeat`: QoS 0

- `status`: retain latest status
- `heartbeat`: no retain

## Security Notes

- use WSS/TLS
- keep credentials outside committed files
- restrict ACLs to the `SC171` topic namespace
- do not expose an unauthenticated write path to `robot/v1/SC171/mobile/cmd`

## Dependency

The adapter still uses optional `paho-mqtt`:

```bash
pip install paho-mqtt
```
