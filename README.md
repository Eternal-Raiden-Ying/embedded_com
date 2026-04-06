# Robot Stack 2026

基于 AidLux (ARM/QCS6490) 的自主目标搜索机器人系统。三个服务通过 TCP IPC 协议协同工作，最终通过串口驱动底盘运动。

## 架构

```
Voice ──task_cmd──▶ Orchestrator ──vision_req──▶ VISTA
      ◀──task_ack──              ◀──vision_obs──
      ◀──tts_event──
                   ──serial──▶ STM32 (底盘)
```

## 模块

| 模块 | 路径 | 入口 |
|------|------|------|
| Orchestrator | `orchestrator/` | `python3 -m orchestrator_service.app.main` |
| VISTA | `VISTA/` | `python3 -m vision_module.app.app` |
| Voice | `Voice/` | `python3 -m voice_service.app.main`（conda: `asr`） |

## IPC 端口

| 链路 | 地址 |
|------|------|
| Voice → Orchestrator (task_cmd) | `127.0.0.1:9001` |
| VISTA → Orchestrator (vision_obs) | `127.0.0.1:9002` |
| Orchestrator → VISTA (vision_req) | `127.0.0.1:9003` |
| Orchestrator → Voice (tts_event) | `127.0.0.1:9011` |
| Orchestrator → Voice (task_ack) | `127.0.0.1:9012` |

## 快速启动

```bash
# 启动全栈（在 AidLux 上执行）
./start_robot_stack.sh

# 停止全栈
./start_robot_stack.sh stop
```

启动前在 `start_robot_stack.sh` 顶部确认配置：

```bash
STACK_PROFILE="full"    # full / dryrun
SPEAKER_ENABLED=0       # 0=不接扬声器  1=接
UART_DEV="/dev/ttyHS1"  # STM32 串口
```

## 运行环境

- 平台：AidLux（Android + Linux，ARM aarch64）
- SoC：Qualcomm QCS6490
- Python：`/usr/bin/python3`（vision / orchestrator），conda env `asr`（voice）
- 串口：`/dev/ttyHS1`（STM32 底盘）

## 目录结构

```
_2026/
├── start_robot_stack.sh     # 全栈启动脚本
├── orchestrator/            # 状态机核心
├── VISTA/                   # 视觉模块
└── Voice/                   # 语音模块
```
