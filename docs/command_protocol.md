# Mobile Command Protocol

## Scope

This document defines the fixed northbound protocol between the WeChat mini-program and the board-side `mobile_gateway`.

The southbound Orchestrator TCP contract stays unchanged:

- `task_cmd` input port remains unchanged
- `task_ack` output port remains unchanged
- `fetch_object` still maps to `intent=FIND`
- `stop` still maps to `intent=STOP`

## Fixed Identity And Topics

- `robot_id = SC171`
- `robot/v1/SC171/mobile/cmd`
- `robot/v1/SC171/mobile/ack`
- `robot/v1/SC171/mobile/status`
- `robot/v1/SC171/heartbeat`

These topics are fixed and must not be changed.

## Formal Command Payload

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

Current compatibility behavior:

- `FIND_AND_PICK` can still be accepted when `runtime.enable_legacy_command_compat=true`
- gateway-local diagnostics such as `query_status`, `resume`, `retry_search`, `go_home` remain available for engineering/debug flows
- those compatibility inputs are not the formal mini-program contract

## Gateway ACK

The gateway publishes `gateway_ack` immediately after it parses and accepts or rejects the incoming command.

Topic:

- `robot/v1/SC171/mobile/ack`

Example:

```json
{
  "type": "mobile_ack",
  "kind": "gateway_ack",
  "robot_id": "SC171",
  "cmd_id": "wx_1777293209382",
  "session_id": "wx_session_001",
  "epoch": 1,
  "cmd": "fetch_object",
  "target": "apple",
  "message": "gateway command accepted",
  "accepted": true,
  "source": "mobile_gateway",
  "ts": 1777293208.6
}
```

Meaning:

- `gateway_ack` means the board-side gateway has accepted the northbound command format
- it does not mean Orchestrator has accepted the task yet

## Southbound Mapping

`fetch_object` maps to:

```json
{
  "type": "task_cmd",
  "intent": "FIND",
  "confidence": 1.0,
  "cmd_id": "wx_1777293209382",
  "session_id": "wx_session_001",
  "epoch": 1,
  "source": "wechat_miniprogram",
  "ts": 1777293208.6,
  "target": "apple"
}
```

`stop` maps to:

```json
{
  "type": "task_cmd",
  "intent": "STOP",
  "confidence": 1.0,
  "cmd_id": "wx_1777293209382",
  "session_id": "wx_session_001",
  "epoch": 1,
  "source": "wechat_miniprogram",
  "ts": 1777293208.7
}
```

## Task ACK

After Orchestrator returns `task_ack`, the gateway publishes a second ACK on the same MQTT ACK topic.

Topic:

- `robot/v1/SC171/mobile/ack`

Example:

```json
{
  "type": "mobile_ack",
  "kind": "task_ack",
  "robot_id": "SC171",
  "cmd_id": "wx_1777293209382",
  "session_id": "wx_session_001",
  "epoch": 1,
  "accepted": true,
  "message": "FIND accepted: apple",
  "source": "mobile_gateway",
  "ts": 1777293208.8
}
```

Meaning:

- `task_ack` means Orchestrator has accepted or rejected the southbound task
- in `debug` mode, diagnostic fields such as `backend_state` may be attached
- in `production` mode, raw backend fields are not exposed to the mini-program protocol

## Mobile Status

Topic:

- `robot/v1/SC171/mobile/status`

Example:

```json
{
  "type": "mobile_status",
  "kind": "status",
  "robot_id": "SC171",
  "session_id": "wx_session_001",
  "epoch": 1,
  "state": "searching",
  "target": "apple",
  "message": "开始桌边任务，目标 apple",
  "progress": 20,
  "command": "fetch_object",
  "source": "mobile_gateway",
  "ts": 1777293210.0
}
```

Formal mobile-facing states:

- `submitted`
- `accepted`
- `searching`
- `running`
- `idle`
- `stopped`
- `error`

Back-end state mapping:

- `IDLE -> idle`
- `SEARCH_TABLE -> searching`
- `COARSE_ALIGN -> running`
- `CONTROLLED_APPROACH -> running`
- `FINAL_LOCK -> running`
- `AT_TABLE_EDGE -> running`
- `SEARCH_TARGET_INIT -> searching`
- `EDGE_SLIDE_SEARCH -> searching`
- `TARGET_CONFIRM -> running`
- `TARGET_LOCKED -> running`
- `FREEZE_BASE -> running`
- `DONE -> idle`
- `ERROR_RECOVERY -> error`
- accepted `STOP` commands publish `stopped`

Display-oriented messages should be directly usable by the mini-program UI, for example:

- `已提交取物命令，目标 apple`
- `FIND accepted: apple`
- `开始桌边任务，目标 apple`
- `视觉模块未连接，任务暂时无法继续`
- `任务已停止`

Error mapping rule:

If the gateway detects any of the following back-end failure signals:

- `vision_req_out connect_failed`
- `Connection refused`
- `link_state=DEGRADED`

then it publishes:

```json
{
  "state": "error",
  "error_code": 1007,
  "message": "视觉模块未连接，任务暂时无法继续"
}
```

Debug-only diagnostic extension fields:

- `backend_state`
- `raw_error`

## Heartbeat

Topic:

- `robot/v1/SC171/heartbeat`

Production payload shape:

```json
{
  "type": "mobile_gateway_heartbeat",
  "kind": "heartbeat",
  "robot_id": "SC171",
  "online": true,
  "backend_mode": "orchestrator_tcp",
  "state": "idle",
  "session_id": "",
  "epoch": 0,
  "ts": 1777293212.0
}
```

Debug mode may append lightweight diagnostics such as:

- `status_age_s`
- `recent_states`

## Duplicate `cmd_id` Handling

The gateway keeps a recent `cmd_id` cache, default size `64`.

When a duplicate `cmd_id` is received:

- `gateway_ack` may be re-published
- the command is not forwarded again to Orchestrator
- status is not re-published just because of the duplicate
- duplicate STOP commands are also suppressed from repeat forwarding and log spam

## Production And Debug Modes

`production` mode:

- keeps the formal protocol stable
- suppresses raw MQTT payload logs
- suppresses per-heartbeat publish success logs
- does not expose raw backend fields to the mini-program payload

`debug` mode:

- keeps the same formal topics and command schema
- allows raw MQTT TX/RX diagnostics
- may attach `backend_state` and `raw_error`
- keeps compatibility hooks easier to observe during bring-up
