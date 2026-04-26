# Mobile MQTT Adapter

## Purpose

The MQTT adapter is a northbound transport layer for `mobile_gateway`.

It does not introduce a second command parser. Instead:

- MQTT `cmd` topic -> existing mobile command handler
- existing `mobile_status` payload -> MQTT `status`
- selected response states -> MQTT `ack`

## Topic Design

Recommended versioned topics:

- `robot/v1/{robot_id}/mobile/cmd`
- `robot/v1/{robot_id}/mobile/ack`
- `robot/v1/{robot_id}/mobile/status`
- `robot/v1/{robot_id}/heartbeat`

Why versioned topics:

- future schema evolution without breaking old clients
- safer robot-by-robot routing
- simpler broker-side ACL management

## QoS Recommendations

- `cmd`: QoS 1
- `ack`: QoS 1
- `status`: QoS 0 or 1 depending on broker load and UI expectations
- `heartbeat`: QoS 0

Reasonable first production choice:

- use QoS 1 for `cmd` and `ack`
- use QoS 0 for frequent `status` and heartbeat updates

## Retain Recommendations

- retain `status`: usually yes for the latest snapshot
- retain `ack`: no
- retain `heartbeat`: no
- retain `cmd`: no

## Heartbeat Mechanism

Heartbeat payload should include at least:

- `robot_id`
- gateway backend mode
- current coarse state
- latest session id
- recent state ring
- status age

The current adapter design already reserves:

- `robot/v1/{robot_id}/heartbeat`

## Reconnect Strategy

Recommended policy:

- let the MQTT client auto-reconnect
- re-subscribe to `cmd` on reconnect
- avoid clearing gateway task memory on transient MQTT disconnect
- continue southbound control if board-side orchestrator link remains healthy

## Security Guidance

For production:

- use TLS
- prefer WSS when the client is a mini-program
- enable authentication
- restrict per-robot topic ACLs
- never expose unauthenticated public write access to `cmd`

Do not use raw `ws://` or unsecured broker exposure as the final deployment pattern.

## Mini-Program Access Notes

For WeChat mini-program integration:

- use WSS/TLS
- configure the broker domain as a legal socket domain
- keep the mini-program payloads structured as `mobile_cmd`
- do not put free-form robot control logic in the board-side gateway

## Dependency Note

The adapter is implemented against `paho-mqtt` as an optional dependency.

If MQTT is enabled without the package installed, the gateway will raise a clear startup error instead of failing silently:

```bash
pip install paho-mqtt
```
