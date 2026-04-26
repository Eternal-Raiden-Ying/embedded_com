# Mobile Command Protocol

## Scope

This document defines the northbound protocol between a mobile client and the board-side mobile gateway.

The gateway keeps the existing southbound contract unchanged:

- `fetch_object` -> `task_cmd(intent=FIND)`
- `go_home` -> `task_cmd(intent=RETURN)`
- `stop` -> `task_cmd(intent=STOP)`

`resume`, `retry_search`, and `query_status` are handled by the gateway itself.

## Supported Commands

- `fetch_object`
- `stop`
- `resume`
- `retry_search`
- `go_home`
- `query_status`

Supported targets in this round:

- `apple`
- `banana`
- `bottle`
- `cup`

## Command JSON

```json
{
  "type": "mobile_cmd",
  "robot_id": "sc171_v2",
  "session_id": "sess_demo_001",
  "cmd": "fetch_object",
  "target": "apple",
  "text": "拿苹果",
  "epoch": 1,
  "ts": 1713945600.0,
  "source": "mobile"
}
```

### Required Fields

- `cmd`
- `robot_id`
- `ts`

### Conditional Fields

- `target` is required for `fetch_object`
- `session_id` is recommended for all commands after the first `fetch_object`

## Status JSON

```json
{
  "robot_id": "sc171_v2",
  "session_id": "sess_demo_001",
  "state": "searching",
  "target": "apple",
  "message": "正在搜索 apple",
  "progress": 75,
  "command": "fetch_object",
  "backend_state": "SEARCH_TARGET_INIT",
  "epoch": 1,
  "ts": 1713945605.0,
  "source": "mobile_gateway"
}
```

## Stable Mobile State Enum

The gateway exposes coarse, stable states to mobile:

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

The original orchestrator internal state is preserved in `backend_state` when available.

## Error Codes

| Code | Name | Meaning |
| --- | --- | --- |
| `1001` | `invalid_json` | malformed JSON or decode failure |
| `1002` | `invalid_command` | unsupported `cmd` |
| `1003` | `invalid_target` | target not in allowed whitelist |
| `1004` | `missing_target` | `fetch_object` missing target |
| `1005` | `busy` | gateway is already executing another high-level task |
| `1006` | `resume_unavailable` | no paused or retriable task exists |
| `1007` | `backend_unavailable` | southbound forward failed |
| `1008` | `task_rejected` | southbound task was rejected |

## Session / Robot / Timestamp / Epoch Guidance

- `robot_id`: stable deployment identity of the robot
- `session_id`: stable high-level task session; created on first `fetch_object`
- `epoch`: gateway-managed generation inside one session; increment when replaying or changing execution plan
- `ts`: Unix timestamp in seconds

Recommended behavior:

- first `fetch_object`: create session, `epoch=1`
- `stop`: same session, next epoch
- `resume`: same session, next epoch
- `retry_search`: same session, next epoch
- `go_home`: same session, next epoch
- `query_status`: does not change epoch

## Command Mapping To Board Internals

| Mobile command | Gateway action | Southbound action |
| --- | --- | --- |
| `fetch_object(target)` | validate target, open or reuse session | `task_cmd(intent=FIND,target=target)` |
| `stop` | freeze current resumable task, send highest-priority stop | `task_cmd(intent=STOP)` |
| `resume` | replay last paused high-level task | `FIND` or `RETURN` |
| `retry_search` | replay last `fetch_object` target from scratch | `task_cmd(intent=FIND,target=last_target)` |
| `go_home` | switch task to return-home flow | `task_cmd(intent=RETURN)` |
| `query_status` | return cached gateway status | none |

## Stop / Resume Rules

- `stop` has priority over normal commands
- the gateway serializes commands per robot
- `resume` is high-level replay, not restoration of internal orchestrator sub-state
- `retry_search` always means restart search from the beginning for the last target

## MQTT / WebSocket Topic Suggestions

- `robot/v1/{robot_id}/mobile/cmd`
- `robot/v1/{robot_id}/mobile/ack`
- `robot/v1/{robot_id}/mobile/status`
- `robot/v1/{robot_id}/mobile/event`
- `robot/v1/{robot_id}/session/{session_id}/event`

The northbound topic namespace is versioned and decoupled from the existing southbound `task_cmd` and `task_ack` channels.

