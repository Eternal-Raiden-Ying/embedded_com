# Mobile MQTT Adapter

## Scope

The MQTT adapter is the northbound transport for the formal mini-program protocol.

It does not define a second command schema. It only transports:

- inbound `mobile_cmd`
- outbound `gateway_ack`
- outbound `task_ack`
- outbound `status`
- outbound `heartbeat`

The formal runtime implementation is the repository-local gateway under:

- `orchestrator/orchestrator_service/mobile_gateway/`

The older validation helper at `~/mobile_gateway_cloud/cloud_mqtt_bridge.py` is no longer the primary workflow.

## Fixed Robot And Topics

The current MQTT northbound contract is fixed to:

- `robot_id = SC171`

Topics:

- `robot/v1/SC171/mobile/cmd`
- `robot/v1/SC171/mobile/ack`
- `robot/v1/SC171/mobile/status`
- `robot/v1/SC171/heartbeat`

## Inbound Topic

### `robot/v1/SC171/mobile/cmd`

Formal payload:

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

Compatibility-only payload:

```json
{
  "type": "FIND_AND_PICK",
  "target": "apple"
}
```

This legacy format is accepted only for transition and testing. It is not the public production contract.

## Outbound ACK Topic

### `robot/v1/SC171/mobile/ack`

Two different `kind` values are published on the same topic.

#### `kind = gateway_ack`

Published immediately after the gateway parses and validates a command.

Typical fields:

- `type = mobile_ack`
- `kind = gateway_ack`
- `robot_id = SC171`
- `cmd_id`
- `session_id`
- `epoch`
- `cmd`
- `target`
- `accepted`
- `message`
- `error_code`
- `ts`

#### `kind = task_ack`

Published after Orchestrator returns a southbound `task_ack`.

Typical fields:

- `type = mobile_ack`
- `kind = task_ack`
- `robot_id = SC171`
- `cmd_id`
- `session_id`
- `epoch`
- `accepted`
- `message`
- `backend_state`
- `error_code`
- `ts`

## Outbound Status Topic

### `robot/v1/SC171/mobile/status`

All progress/state messages use:

- `type = mobile_status`
- `kind = status`

Typical fields:

- `robot_id`
- `session_id`
- `epoch`
- `state`
- `target`
- `message`
- `progress`
- `command`
- `backend_state`
- `ts`

## Outbound Heartbeat Topic

### `robot/v1/SC171/heartbeat`

Heartbeat uses:

- `type = mobile_gateway_heartbeat`
- `kind = heartbeat`

Typical fields:

- `robot_id`
- `backend_mode`
- `state`
- `session_id`
- `epoch`
- `status_age_s`
- `recent_states`
- `ts`

## QoS Defaults

The current formal gateway defaults are:

- `cmd`: QoS 1
- `ack`: QoS 1
- `status`: QoS 0
- `heartbeat`: QoS 0

## Retain Defaults

The current formal gateway defaults are:

- `cmd`: no retain
- `ack`: no retain
- `status`: retain latest status
- `heartbeat`: no retain

## Reconnect Strategy

- use client auto-reconnect
- re-subscribe to `robot/v1/SC171/mobile/cmd` after reconnect
- keep gateway task memory intact on transient MQTT disconnect
- do not clear southbound task context just because MQTT reconnects

## Security Recommendations

For production:

- use WSS/TLS
- require authentication
- restrict topic ACLs to the `SC171` namespace
- never expose an unauthenticated write path to `robot/v1/SC171/mobile/cmd`

Formal production guidance:

- do not use raw `ws://`
- do not treat local IP direct-connect as the final mini-program architecture

## Mini-Program Integration Notes

For WeChat mini-program:

- configure the broker domain as a legal socket domain
- use WSS/TLS
- send only structured `mobile_cmd`
- keep ASR text normalization outside the board-side gateway

## Dependency

The adapter uses `paho-mqtt` as an optional runtime dependency:

```bash
pip install paho-mqtt
```

If MQTT is enabled without the dependency, the gateway raises a clear startup error.

## Startup Reference

Orchestrator:

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

Formal gateway:

```bash
cd /home/aidlux/embedded_com
PYTHONPATH=/home/aidlux/embedded_com/orchestrator \
python3 -m orchestrator_service.mobile_gateway.runtime.service \
  --config configs/mobile_gateway.mqtt.yaml
```
