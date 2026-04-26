# Mobile Gateway Design Notes

This repository now contains a minimal board-side mobile gateway implementation under:

- `orchestrator/orchestrator_service/mobile_gateway/`

Design intent:

- keep orchestrator as the only board-side execution authority
- add a new upstream control entry for mobile traffic
- keep northbound and southbound protocols separate
- preserve offline mock validation without hardware

Implemented runtime modes:

- `mock`
  - used for local closed-loop tests
  - does not require orchestrator, VISTA, or UART
- `orchestrator_bridge`
  - forwards `task_cmd` into the real orchestrator
  - can optionally consume `task_ack`
  - can optionally watch orchestrator `state_blocks.jsonl`

Command handling policy:

- single-flight by default for reliability
- `stop` preempts normal commands
- `resume` replays the last paused high-level task
- `retry_search` replays the last fetch target from scratch

