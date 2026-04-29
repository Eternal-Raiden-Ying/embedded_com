# Mobile Control Next Steps

## 1. MQTT Broker Integration

The current gateway already separates:

- northbound mobile protocol
- southbound orchestrator protocol

That makes MQTT integration straightforward:

1. keep the current command and status JSON unchanged
2. replace or augment local TCP input with MQTT subscription on:
   - `robot/v1/SC171/mobile/cmd`
3. publish ACK and status to:
   - `robot/v1/SC171/mobile/ack`
   - `robot/v1/SC171/mobile/status`
4. keep session-scoped event streaming optional:
   - `robot/v1/SC171/session/{session_id}/event`

Recommended broker-side behaviors:

- QoS 1 for commands and ACK
- retained last-known `status`
- deduplicate by `cmd_id`
- reject stale `epoch`
- prefer WSS/TLS in production, especially for mini-program access

## 1.1 Practical MQTT Rollout Steps

Recommended rollout order:

1. pick a broker:
   - EMQX
   - Mosquitto with WSS/TLS fronting
   - cloud MQTT service
2. expose WSS/TLS rather than raw `ws://`
3. configure authentication and per-robot topic permissions
4. point `mobile_gateway` northbound MQTT adapter at the broker
5. keep the southbound `task_cmd/task_ack` bridge unchanged

For WeChat mini-program deployment:

- configure a legal socket domain in the mini-program backend settings
- use WSS/TLS in production
- do not treat local IP direct-connect as the final architecture

## 2. Mobile Mini-Program / App Work

The mobile client only needs to do four things:

1. present a constrained command UI, not free-form control
2. build structured commands using the documented northbound schema
3. subscribe to ACK and status topics
4. render large, accessible feedback states

Recommended UI actions for the first version:

- fetch apple / banana / bottle / cup
- stop
- resume
- retry
- go home
- query status

Recommended transport split:

- mini-program sends structured `mobile_cmd`
- mini-program subscribes to `ack/status/heartbeat`
- microphone/ASR text, if used, is converted into `mobile_cmd` in the mini-program or a lightweight cloud-side helper

## 3. Microphone Text To Structured Command Mapping

Do not let raw speech text enter Orchestrator directly. The board-side `Voice/ASR` service has been archived out of the repository; current microphone handling should produce the same structured mobile command contract.

Recommended pipeline:

1. microphone/ASR text
2. intent parser at mobile or cloud edge
3. structured `mobile_cmd`
4. board-side gateway validation
5. existing `task_cmd`

Examples:

- "拿苹果" -> `fetch_object(target="apple")`
- "停下" -> `stop`
- "继续" -> `resume`
- "重新找杯子" -> `retry_search`
- "回去" -> `go_home`

Implementation guidance:

- avoid sending raw speech text to the board
- keep the board-side gateway deterministic and whitelist-driven
- let the mobile side or a lightweight backend own NLU ambiguity handling

## 4. Audio Feedback For Blind Users

For a visually impaired user flow, status feedback should be short, consistent, and state-based.

Recommended feedback layers:

- immediate ACK: "已收到，开始找苹果"
- progress milestone: "正在搜索桌边", "已经锁定苹果", "正在返回起点"
- stop confirmation: "已停止"
- error confirmation: "当前无法继续，请稍后重试"

Important design principles:

- avoid verbose progress spam
- keep the same sentence template for the same state
- always confirm destructive commands like `stop`
- never expose raw internal state names to the user

## 5. Production Bridge Hardening

Before connecting a real mobile client, add:

- `cmd_id` deduplication cache
- session timeout and stale epoch rejection
- optional fan-out only if another future upstream needs the same `task_ack`
- reconnect-safe status replay on gateway restart
- authentication for MQTT/WebSocket clients

## 6. Potential Future Refactors

If multi-entry upstream control becomes permanent, a later refactor could move task ingress into a dedicated control plane service.

That is not required yet. For now, the mobile gateway layer is intentionally a thin adapter so the current orchestrator stays in charge of robot closed-loop control.
