# Orchestrator

机器人状态机核心服务。负责接收语音指令、驱动视觉请求、控制底盘运动，并通过 ACK 机制向语音端反馈执行状态。

## 架构

```
voice_service ──task_cmd──▶ [state_machine] ──vision_req──▶ VISTA
              ◀──task_ack──                 ◀──vision_obs──
              ◀──tts_event──
                            ──serial──▶ STM32
```

状态机核心位于 `orchestrator_service/runtime/state_machine.py`，状态流转：
`IDLE → SEARCH → APPROACH → RETURN → STOPPED`

## 目录结构

```
orchestrator/
├── orchestrator_service/
│   ├── app/main.py             # 服务入口
│   ├── bridge/
│   │   ├── uart_bridge.py      # UART 串口桥接
│   │   └── simple_car_protocol.py
│   ├── config/
│   │   └── board_config.py     # 主配置文件（最常改这里）
│   ├── ipc/
│   │   ├── protocol.py         # 消息协议定义
│   │   └── transport.py        # TCP 传输层
│   └── runtime/
│       ├── state_machine.py    # 状态机主逻辑
│       ├── controller.py       # 运动控制器
│       ├── context.py          # 任务上下文
│       └── service.py          # 服务生命周期
├── scripts/
│   └── start_orchestrator.sh   # 启停脚本
├── examples/                   # 调试工具（mock sender/listener）
└── logs/ / runs/ / pids/
```

## 配置

主配置文件：`orchestrator_service/config/board_config.py`

| 配置项 | 说明 |
|--------|------|
| `task_ack_out` | task_ack 输出地址（默认 `127.0.0.1:9012`） |
| `vision_req_fail_threshold` | 视觉链路连续失败停车阈值（默认 2） |
| `state_block_period_s` | 状态块快照周期 |
| `UART_DEV` | 串口设备（`/dev/ttyHS1`） |

环境变量覆盖：

| 变量 | 说明 |
|------|------|
| `ORCH_SERIAL_DRY_RUN=1` | 不实际发串口，仅打印 |
| `ORCH_TTS_EVENT_OUT_TRANSPORT=disabled` | 禁用 TTS 事件输出 |
| `ORCH_DRY_RUN_ECHO_STDOUT=1` | 干跑模式下输出到 stdout |

## 运行

```bash
# 前台运行
cd /home/aidlux/2026/orchestrator
python3 -m orchestrator_service.app.main

# 脚本方式
bash scripts/start_orchestrator.sh restart
bash scripts/start_orchestrator.sh status
bash scripts/start_orchestrator.sh tail
```

## IPC 协议

| 方向 | 消息类型 | 地址 |
|------|----------|------|
| 接收 | `task_cmd` | `127.0.0.1:9001` |
| 接收 | `vision_obs` | `127.0.0.1:9002` |
| 发送 | `vision_req` | `127.0.0.1:9003` |
| 发送 | `tts_event` | `127.0.0.1:9011` |
| 发送 | `task_ack` | `127.0.0.1:9012` |

`task_cmd` 关键字段：`type`, `cmd_id`, `session_id`, `epoch`, `source`

`task_ack` 关键字段：`cmd_id`, `session_id`, `epoch`, `accepted`, `state`, `reason`

## 串口协议（STM32）

```
MODE <SEARCH|APPROACH|RETURN|STOP>   # 模式变化时发送一次
V <vx_mps> <wz_rps>                  # 主控制命令（m/s, rad/s）
STOP                                  # 立即停车
STATE <ok|timeout|estop|fault> <vL> <vR> <yaw>  # 底盘状态上报
```

## 日志结构

每次启动在 `runs/run_<timestamp>/` 下生成：

| 文件 | 内容 |
|------|------|
| `timeline.jsonl` | 主链路事件流（首选排查入口） |
| `ipc.jsonl` | task_cmd / task_ack / vision_req 收发记录 |
| `state_blocks.jsonl` | 状态块快照 |
| `cmd_vel.jsonl` | 速度指令记录 |
| `events.log` | 文本事件日志 |

排查 STOP / 视觉链路失败 / ACK 超时时，优先看 `timeline.jsonl + ipc.jsonl + state_blocks.jsonl`。
