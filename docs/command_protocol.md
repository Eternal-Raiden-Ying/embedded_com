# Mobile Command Protocol

## Scope

This document defines the formal northbound mobile protocol used by `mobile_gateway`.

The board-side southbound contract remains unchanged:

- `fetch_object` -> `task_cmd(intent=FIND)`
- `go_home` -> `task_cmd(intent=RETURN)`
- `stop` -> `task_cmd(intent=STOP)`

`resume`, `retry_search`, and `query_status` remain gateway-level semantics.

## Fixed Robot Identity

The northbound mobile protocol is fixed to:

- `robot_id = "SC171"`

This value is normalized by the gateway and should be treated as the only formal robot identifier for the mini-program / MQTT interface in this phase.

## Fixed MQTT Topics

- `robot/v1/SC171/mobile/cmd`
- `robot/v1/SC171/mobile/ack`
- `robot/v1/SC171/mobile/status`
- `robot/v1/SC171/heartbeat`

These are now the canonical MQTT topics for the northbound protocol.

## Formal Command Format

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

### Command Fields

- `type`
  - must be `mobile_cmd`
- `robot_id`
  - must be `SC171`
- `cmd_id`
  - required unique command id
- `session_id`
  - required for stable multi-command task control
- `epoch`
  - task generation inside a session
- `cmd`
  - one of:
    - `fetch_object`
    - `stop`
    - `resume`
    - `retry_search`
    - `go_home`
    - `query_status`
- `target`
  - required for `fetch_object`
  - allowed values:
    - `apple`
    - `banana`
    - `bottle`
    - `cup`
- `text`
  - optional natural-language mirror for UI/debug
- `source`
  - recommended value: `wechat_miniprogram`
- `ts`
  - unix timestamp in seconds

## Compatibility Input

The gateway still accepts this old test-only input:

```json
{
  "type": "FIND_AND_PICK",
  "target": "apple"
}
```

Compatibility behavior:

- it is normalized to `cmd=fetch_object`
- it is accepted only as a backward-compatible input form
- it is not part of the formal mini-program contract
- it must not be used as the documented production protocol

## Outbound MQTT Message Kinds

The gateway now uses explicit `kind` values.

### 1. Gateway ACK

Published immediately after the gateway accepts or rejects a northbound command.

Topic:

- `robot/v1/SC171/mobile/ack`

Example:

```json
{
  "type": "mobile_ack",
  "kind": "gateway_ack",
  "robot_id": "SC171",
  "cmd_id": "cmd_1234567890",
  "session_id": "sess_1234567890",
  "epoch": 1,
  "cmd": "fetch_object",
  "target": "apple",
  "accepted": true,
  "message": "gateway command accepted",
  "source": "mobile_gateway",
  "ts": 1713945600.1
}
```

### 2. Task ACK

Published after a real Orchestrator `task_ack` arrives.

Topic:

- `robot/v1/SC171/mobile/ack`

Example:

```json
{
  "type": "mobile_ack",
  "kind": "task_ack",
  "robot_id": "SC171",
  "cmd_id": "cmd_1234567890",
  "session_id": "sess_1234567890",
  "epoch": 1,
  "accepted": true,
  "message": "FIND accepted: apple",
  "backend_state": "SEARCH_TABLE",
  "source": "mobile_gateway",
  "ts": 1713945600.3
}
```

### 3. Status

Published as the unified progress/status stream.

Topic:

- `robot/v1/SC171/mobile/status`

Example:

```json
{
  "type": "mobile_status",
  "kind": "status",
  "robot_id": "SC171",
  "session_id": "sess_1234567890",
  "epoch": 1,
  "state": "searching",
  "target": "apple",
  "message": "正在搜索 apple",
  "progress": 75,
  "command": "fetch_object",
  "backend_state": "SEARCH_TARGET_INIT",
  "source": "mobile_gateway",
  "ts": 1713945605.0
}
```

### 4. Heartbeat

Topic:

- `robot/v1/SC171/heartbeat`

Example:

```json
{
  "type": "mobile_gateway_heartbeat",
  "kind": "heartbeat",
  "robot_id": "SC171",
  "backend_mode": "orchestrator_tcp",
  "state": "idle",
  "session_id": "",
  "epoch": 0,
  "status_age_s": 0.5,
  "recent_states": ["idle"],
  "ts": 1713945606.0
}
```

## Stable Status States

The mobile-facing `state` enum remains:

- `idle`
- `submitted`
- `accepted`
- `searching`
- `approaching`
- `returning`
- `stopping`
- `stopped`
- `completed`
- `error`
- `unknown`

The fine-grained orchestrator state is exposed only via `backend_state`.

## Privacy / Abstraction Rule

The formal mini-program protocol must not expose raw Orchestrator `raw` transport fields.

Allowed upstream-facing debug fields:

- `backend_state`
- `message`
- `progress`

Disallowed as formal public fields:

- serial raw lines
- raw UART payloads
- internal low-level transport dumps

