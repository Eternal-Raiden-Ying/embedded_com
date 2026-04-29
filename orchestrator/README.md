# Orchestrator

Orchestrator 是板端状态机与底盘控制服务。当前接收 `mobile_gateway` 转发的任务命令，向 VISTA 发送视觉请求，接收视觉观测后通过 UART 控制 STM32。

## 链路

```text
mobile_gateway --task_cmd:9001--> Orchestrator --vision_req:9003--> VISTA
mobile_gateway <--task_ack:9012--- Orchestrator <--vision_obs:9002--- VISTA
                                      |
                                      +--UART /dev/ttyHS1--> STM32
```

`tts_event_out` 保留为兼容字段，默认 `disabled`；板端 ASR/Voice 不在当前链路内。

## 运行

```bash
cd /home/aidlux/embedded_com/orchestrator

# dry-run，不打开真实串口
export ORCH_SERIAL_DRY_RUN=1
export ORCH_DRY_RUN_ECHO_STDOUT=1
export ORCH_TTS_EVENT_OUT_TRANSPORT=disabled
/usr/bin/python3 -m orchestrator_service.app.main

# 脚本方式
bash scripts/start_orchestrator.sh restart
bash scripts/start_orchestrator.sh status
bash scripts/start_orchestrator.sh tail
```

真实车由顶层脚本统一启动：

```bash
cd /home/aidlux/embedded_com
STACK_PROFILE=full UART_DEV=/dev/ttyHS1 ./start_robot_stack.sh
```

## 端口

| 方向 | 消息 | 默认地址 |
|------|------|----------|
| RX | `task_cmd` | `127.0.0.1:9001` |
| RX | `vision_obs` | `127.0.0.1:9002` |
| TX | `vision_req` | `127.0.0.1:9003` |
| TX | `task_ack` | `127.0.0.1:9012` |
| TX | `tts_event` | `127.0.0.1:9011`, 默认禁用 |

## 状态机

核心文件：[state_machine.py](/home/aidlux/embedded_com/orchestrator/orchestrator_service/runtime/state_machine.py) 和 [context.py](/home/aidlux/embedded_com/orchestrator/orchestrator_service/runtime/context.py)。

主要状态：

| 状态 | 说明 |
|------|------|
| `IDLE` | 空闲，等待 `task_cmd` |
| `SEARCH_TABLE` | 旋转/搜索桌边，请求 VISTA `DEPTH_PERCEPTION` |
| `COARSE_ALIGN` | 桌边可见后先粗对齐 yaw/横向偏差 |
| `CONTROLLED_APPROACH` | 按桌边观测受控靠近 |
| `FINAL_LOCK` | 最终锁边，检查 yaw/dist/lateral 和稳定帧 |
| `AT_TABLE_EDGE` | 已停靠桌边，短暂停稳 |
| `SEARCH_TARGET_INIT` | 初始化沿边找目标 |
| `EDGE_SLIDE_SEARCH` | 沿桌边滑动搜索目标，请求 VISTA `TRACK_LOCAL` |
| `TARGET_CONFIRM` | 目标候选稳定确认 |
| `TARGET_LOCKED` | 目标已锁定 |
| `FREEZE_BASE` | 冻结底盘，等待任务完成 |
| `DONE` | 完成后回到 `IDLE` |

恢复与异常状态：

| 状态 | 说明 |
|------|------|
| `DOCK_RETRY` | 停靠失败后后退重试 |
| `LEAVE_EDGE` | 当前边未找到目标，离开桌边 |
| `RELOCATE_TO_EDGE` | 转向下一条边 |
| `REACQUIRE_EDGE` | 重新捕获桌边 |
| `NEXT_TABLE` | 当前桌未完成，切换下一桌搜索 |
| `AVOID_OBSTACLE` | 避障暂停，清除后恢复 |
| `RETURN_HOME` | 返航 |
| `ERROR_RECOVERY` | 故障停车并恢复到空闲 |

## 关键配置

| 环境变量 | 默认 | 说明 |
|----------|------|------|
| `ORCH_SERIAL_DRY_RUN` | `1` | `1` 时不打开 UART |
| `ORCH_SERIAL_PORT` | `/dev/ttyHS1` | STM32 串口 |
| `ORCH_SERIAL_BAUDRATE` | `115200` | 串口波特率 |
| `ORCH_TASK_CMD_IN_PORT` | `9001` | 接收任务 |
| `ORCH_VISION_OBS_IN_PORT` | `9002` | 接收视觉观测 |
| `ORCH_VISION_REQ_OUT_PORT` | `9003` | 发送视觉请求 |
| `ORCH_TASK_ACK_OUT_PORT` | `9012` | 给 mobile_gateway 的任务 ACK |
| `ORCH_TTS_EVENT_OUT_TRANSPORT` | `disabled` | 兼容 TTS 输出 |

## 日志

每次启动生成 `orchestrator/runs/run_<timestamp>/`：

| 文件 | 内容 |
|------|------|
| `timeline.jsonl` | 状态切换、故障、关键事件 |
| `ipc.jsonl` | `task_cmd`, `task_ack`, `vision_req`, `vision_obs` |
| `state_blocks.jsonl` | 周期状态快照 |
| `cmd_vel.jsonl` | 下发给底盘的速度命令 |
| `events.log` | 文本事件 |

验收时优先看 `timeline.jsonl + ipc.jsonl + state_blocks.jsonl + cmd_vel.jsonl`。
