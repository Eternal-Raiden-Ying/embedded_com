# Robot Stack 2026

AidLux/QCS6490 板端机器人主链路。当前入口为手机小程序或云端 MQTT，经 `mobile_gateway` 转成 Orchestrator 任务，再由 Orchestrator 驱动 VISTA 与 STM32 底盘。

## 主链路

```text
小程序 / 云 MQTT
        |
        v
mobile_gateway --task_cmd:9001--> Orchestrator --vision_req:9003--> VISTA
        ^                         |   ^                            |
        |                         |   |                            |
        +--task_ack/status:9012---+   +------vision_obs:9002-------+
                                  |
                                  +--UART /dev/ttyHS1--> STM32
```

`Voice/ASR` 不在当前启动链路内。`tts_event` 是历史兼容输出，默认 `disabled`。

## 入口与端口

| 模块 | 路径 | 入口 | 日志 |
|------|------|------|------|
| mobile_gateway | repo root / `orchestrator_service.mobile_gateway` | `./start_robot_stack.sh` 或 `python3 -m orchestrator_service.mobile_gateway.runtime.service --config configs/mobile_gateway.mqtt.yaml` | `logs/mobile_gateway.out` |
| Orchestrator | `orchestrator/` | `python3 -m orchestrator_service.app.main` | `orchestrator/logs/orchestrator.out`, `orchestrator/runs/run_*/` |
| VISTA | `VISTA/` | `/usr/bin/python3 -m vision_module.app.app` | `VISTA/logs/vision.out`, `VISTA/runs/run_*/` |
| STM32 | UART | `/dev/ttyHS1 @ 115200` | Orchestrator `cmd_vel.jsonl` / `ipc.jsonl` |

| 链路 | 默认地址 |
|------|----------|
| mobile_gateway -> Orchestrator `task_cmd` | `127.0.0.1:9001` |
| VISTA -> Orchestrator `vision_obs` | `127.0.0.1:9002` |
| Orchestrator -> VISTA `vision_req` | `127.0.0.1:9003` |
| Orchestrator -> mobile_gateway `task_ack` | `127.0.0.1:9012` |
| Orchestrator `tts_event` 兼容口 | `127.0.0.1:9011`, 默认禁用 |

## 快速运行

```bash
# dry-run：不连真实车，只打印 UART 控制输出
STACK_PROFILE=dryrun ./start_robot_stack.sh

# 真实车：连接 STM32 串口
STACK_PROFILE=full UART_DEV=/dev/ttyHS1 ./start_robot_stack.sh

# 查看 / 停止
./start_robot_stack.sh status
./start_robot_stack.sh stop
```

启动脚本会拉起 `VISTA -> Orchestrator -> mobile_gateway`，并等待端口或日志 ready-check。

## 状态机主线

`FIND target` 的主路径：

```text
IDLE
 -> SEARCH_TABLE
 -> COARSE_ALIGN
 -> CONTROLLED_APPROACH
 -> FINAL_LOCK
 -> AT_TABLE_EDGE
 -> SEARCH_TARGET_INIT
 -> EDGE_SLIDE_SEARCH
 -> TARGET_CONFIRM
 -> TARGET_LOCKED
 -> FREEZE_BASE
 -> DONE
```

异常与换边状态包括 `DOCK_RETRY`, `LEAVE_EDGE`, `RELOCATE_TO_EDGE`, `REACQUIRE_EDGE`, `NEXT_TABLE`, `AVOID_OBSTACLE`, `ERROR_RECOVERY`。返航使用 `RETURN_HOME`。

VISTA 当前主要 mode：

| mode | 用途 |
|------|------|
| `TRACK_LOCAL` | RGB + 本地模型，输出 `target_obs` 或返航 tag 观测 |
| `DEPTH_PERCEPTION` | 深度为主的桌边感知，输出 `table_edge_obs` |
| `TABLE_EDGE_PERCEPTION` | RGB + depth + 本地模型，用于沿边目标搜索时同时输出桌边和目标信息 |

## 工程验收

详细步骤见 [docs/system_runbook.md](/home/aidlux/embedded_com/docs/system_runbook.md)。

最低验收：

- dry-run：`mobile_gateway` 收到 `fetch_object` 后，Orchestrator 进入 `SEARCH_TABLE`，`orchestrator/runs/run_*/ipc.jsonl` 有 `task_cmd/task_ack/vision_req`。
- 手持相机：VISTA 能切到 `DEPTH_PERCEPTION` 或 `TRACK_LOCAL`，`VISTA/runs/run_*/ipc.jsonl` 持续输出 `vision_obs`。
- 真实车低速：`STACK_PROFILE=full` 下 UART 有 `MODE`/`V`/`STOP`，STOP 后底盘停止，日志可回放状态切换。
