# Startup Path Audit

Current startup chain:

`start_robot_stack.sh -> VISTA -> Orchestrator -> mobile_gateway`

| Service | Working directory | Python entry | Launcher log |
| --- | --- | --- | --- |
| VISTA vision | `$VISION_ROOT` default `$STACK_ROOT/VISTA` | `/usr/bin/python3 -m vision_module.app.app` | `logs/runs/$STACK_RUN_ID/vision/vision.out` |
| Orchestrator | `$ORCH_ROOT` default `$STACK_ROOT/orchestrator` | `/usr/bin/python3 -m orchestrator_service.app.main` | `logs/runs/$STACK_RUN_ID/orchestrator/orchestrator.out` |
| Mobile gateway | `$STACK_ROOT` | `/usr/bin/python3 -m orchestrator_service.mobile_gateway.runtime.service --config "$GATEWAY_CONFIG"` | disabled by default; `logs/runs/$STACK_RUN_ID/mobile_gateway/mobile_gateway.out` when `ENABLE_GATEWAY_LOGS=true` |

Ready checks use configured endpoints: VISTA `vision_req`, Orchestrator `task_cmd` and `vision_obs`, and gateway HTTP/command connectivity. `STACK_PORTS` remains for stop-time cleanup.

Removed from the startup path: pre-run `logs/*.out` defaults, `COLLECT_GATEWAY_LOGS`, and the unused `wait_for_sockets`, `wait_for_ports`, `wait_for_log_pattern` helpers.
