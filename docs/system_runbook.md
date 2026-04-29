# 当前系统运行手册

## 范围

当前真实主链路：

```text
小程序/云 MQTT/mobile_gateway -> Orchestrator -> VISTA -> STM32
```

板端 ASR/Voice 不在启动链路内。麦克风由当前业务直接使用，不再通过仓库内 `Voice/` 服务承载。

## 端口与主题

| 项 | 默认值 |
|----|--------|
| MQTT cmd | `robot/v1/SC171/mobile/cmd` |
| MQTT ack | `robot/v1/SC171/mobile/ack` |
| MQTT status | `robot/v1/SC171/mobile/status` |
| MQTT heartbeat | `robot/v1/SC171/heartbeat` |
| `task_cmd` | `127.0.0.1:9001` |
| `vision_obs` | `127.0.0.1:9002` |
| `vision_req` | `127.0.0.1:9003` |
| `task_ack` | `127.0.0.1:9012` |
| UART | `/dev/ttyHS1 @ 115200` |

## 启动

```bash
cd /home/aidlux/embedded_com

# dry-run
STACK_PROFILE=dryrun ./start_robot_stack.sh

# 真实车
STACK_PROFILE=full UART_DEV=/dev/ttyHS1 ./start_robot_stack.sh

# 状态/停止
./start_robot_stack.sh status
./start_robot_stack.sh stop
```

`configs/mobile_gateway.mqtt.yaml` 是板端本地配置，不提交密钥。示例见 `configs/mobile_gateway.mqtt.example.yaml`。

## Orchestrator 状态

| 状态 | 入口/退出条件 |
|------|---------------|
| `SEARCH_TABLE` | 收到 `FIND` 后搜索桌边；桌边稳定可见后进 `COARSE_ALIGN` |
| `COARSE_ALIGN` | 粗对齐桌边 yaw；满足稳定帧后进 `CONTROLLED_APPROACH` |
| `CONTROLLED_APPROACH` | 受控接近桌边；边缘 ready 后进 `FINAL_LOCK` |
| `FINAL_LOCK` | 校验 yaw/dist/lateral 与稳定帧；成功进 `AT_TABLE_EDGE` |
| `AT_TABLE_EDGE` | 停稳；随后进 `SEARCH_TARGET_INIT` |
| `EDGE_SLIDE_SEARCH` | 沿边找目标；发现候选进 `TARGET_CONFIRM` |
| `TARGET_CONFIRM` | 多帧确认目标；成功进 `TARGET_LOCKED`，失败回 `EDGE_SLIDE_SEARCH` |

换边和恢复：`LEAVE_EDGE`, `RELOCATE_TO_EDGE`, `REACQUIRE_EDGE`, `NEXT_TABLE`。安全和故障：`AVOID_OBSTACLE`, `ERROR_RECOVERY`。返航：`RETURN_HOME`。

## VISTA Mode

| mode | 摄像头/模型 | 输出 | 使用场景 |
|------|-------------|------|----------|
| `TRACK_LOCAL` | RGB + 本地模型 | `target_obs`；返航时输出 home tag 观测 | 桌边目标搜索、返航 |
| `DEPTH_PERCEPTION` | depth；可选 RGB 桌面框模型 | `table_edge_obs` | 搜桌边、粗对齐、接近、锁边 |
| `TABLE_EDGE_PERCEPTION` | RGB + depth + 本地模型 | `target_obs` + `table_edge_obs` | 需要同时保持桌边与目标观测的沿边搜索 |

VISTA stage 仍是 `SEARCH` / `GRASP` / `RETURN` / `IDLE`，Orchestrator 主要通过 `stage=SEARCH` 和 `mode_hint` 请求上述 mode。

## 日志位置

| 模块 | 运行日志 | 结构化日志 |
|------|----------|------------|
| mobile_gateway | `logs/mobile_gateway.out` | gateway stdout |
| Orchestrator | `orchestrator/logs/orchestrator.out` | `orchestrator/runs/run_*/` |
| VISTA | `VISTA/logs/vision.out` | `VISTA/runs/run_*/` |

重点文件：

- `orchestrator/runs/run_*/timeline.jsonl`
- `orchestrator/runs/run_*/ipc.jsonl`
- `orchestrator/runs/run_*/state_blocks.jsonl`
- `orchestrator/runs/run_*/cmd_vel.jsonl`
- `VISTA/runs/run_*/ipc.jsonl`

## Dry-Run 测试

```bash
cd /home/aidlux/embedded_com
STACK_PROFILE=dryrun FOLLOW_STACK_LOGS_AFTER_START=0 ./start_robot_stack.sh
/usr/bin/python3 tools/smoke_mobile_to_orchestrator.py
./start_robot_stack.sh stop
```

验收标准：

- `logs/mobile_gateway.out` 出现 `cmd received` 和 `task_cmd forwarded`。
- `orchestrator/runs/run_*/ipc.jsonl` 有 `task_cmd`, `task_ack`, `vision_req`。
- `orchestrator/runs/run_*/state_blocks.jsonl` 出现 `SEARCH_TABLE`。
- `orchestrator/logs/orchestrator.out` 有 `DRY_RUN` 或 UART dry-run 输出。

## 手持相机测试

```bash
cd /home/aidlux/embedded_com
STACK_PROFILE=dryrun VISTA_PREVIEW_RGB=1 FOLLOW_STACK_LOGS_AFTER_START=1 ./start_robot_stack.sh
```

用小程序或 smoke 工具发送 `fetch_object/apple`，手持相机对准桌边和目标。

验收标准：

- `VISTA/logs/vision.out` 出现 `mode switched`，mode 为 `DEPTH_PERCEPTION` 或 `TRACK_LOCAL`。
- `VISTA/runs/run_*/ipc.jsonl` 持续发送 `vision_obs`。
- 搜桌边阶段有 `table_edge_obs`；沿边找目标阶段有 `target_obs`。
- Orchestrator 状态从 `SEARCH_TABLE` 推进到 `COARSE_ALIGN` 或后续状态。

## 真实车低速测试

```bash
cd /home/aidlux/embedded_com
STACK_PROFILE=full UART_DEV=/dev/ttyHS1 ORCH_SEARCH_TABLE_WZ=0.10 ORCH_EDGE_SLIDE_VY=0.06 ./start_robot_stack.sh
```

测试前确认：

- 车轮悬空或低速安全区域。
- 急停可用。
- `/dev/ttyHS1` 权限或 sudo 可用。

验收标准：

- `orchestrator/runs/run_*/cmd_vel.jsonl` 有 `MODE`, `V`, `STOP` 对应记录。
- STM32 收到低速控制，STOP 后立即停止。
- `FINAL_LOCK` 成功时状态进入 `AT_TABLE_EDGE`，随后进入 `EDGE_SLIDE_SEARCH`。
- 异常或视觉断链时进入 `ERROR_RECOVERY` 或停车，不持续下发前进命令。

## 回归测试

```bash
cd /home/aidlux/embedded_com
/usr/bin/python3 -m unittest \
  tests.test_command_protocol \
  tests.test_gateway_mapping \
  tests.test_mock_flow \
  tests.test_real_protocol_mapping \
  VISTA.vision_module.test.test_runtime_architecture
```

通过标准：单元测试通过；若板端缺少硬件或模型，只允许硬件相关测试跳过，不允许协议和状态机测试失败。
