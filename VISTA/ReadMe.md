# VISTA

VISTA 是板端视觉服务，接收 Orchestrator 的 `vision_req`，按 stage/mode 调度摄像头、模型和桌边感知，并输出 `vision_obs`。

## 链路

```text
Orchestrator --vision_req:9003--> VISTA
Orchestrator <--vision_obs:9002--- VISTA
```

## Stage 与 Mode

`stage` 表示业务阶段，当前注册：`SEARCH`, `GRASP`, `RETURN`, `IDLE`。

`mode` 表示实际资源模式，当前主链路使用：

| mode | 摄像头/模型 | 主要输出 | 说明 |
|------|-------------|----------|------|
| `TRACK_LOCAL` | RGB + 本地模型 | `target_obs` | 目标搜索；`RETURN` stage 复用该模式输出返航 tag 观测 |
| `DEPTH_PERCEPTION` | depth；可选 RGB 桌面框模型 | `table_edge_obs` | 搜桌边、粗对齐、接近、最终锁边 |
| `TABLE_EDGE_PERCEPTION` | RGB + depth + 本地模型 | `table_edge_obs` + `target_obs` | 同时需要桌边和目标观测时使用 |

Orchestrator 在 `SEARCH_TABLE`, `COARSE_ALIGN`, `CONTROLLED_APPROACH`, `FINAL_LOCK`, `REACQUIRE_EDGE` 请求 `DEPTH_PERCEPTION`；在 `AT_TABLE_EDGE`, `SEARCH_TARGET_INIT`, `EDGE_SLIDE_SEARCH`, `TARGET_CONFIRM`, `TARGET_LOCKED`, `FREEZE_BASE` 请求 `TRACK_LOCAL`。

## 运行

```bash
cd /home/aidlux/embedded_com/VISTA
/usr/bin/python3 -m vision_module.app.app
```

常用环境变量：

| 变量 | 默认 | 说明 |
|------|------|------|
| `VISION_REQ_PORT` | `9003` | 接收 `vision_req` |
| `VISION_OBS_PORT` | `9002` | 发送 `vision_obs` |
| `VISION_ACTIVE_MODEL` | `yolov7_detect` | 本地检测模型 profile |
| `VISTA_TABLE_BBOX_ENABLE` | `0` | `DEPTH_PERCEPTION` 是否启用 RGB 桌面框模型 |
| `VISTA_TABLE_MODEL` | `yolov7_detect` | 桌面框模型 |
| `VISION_PREVIEW` | 板端默认 `1` | OpenCV 预览 |

## 日志

| 路径 | 内容 |
|------|------|
| `VISTA/logs/vision.out` | 启动脚本 stdout/stderr |
| `VISTA/runs/run_*/timeline.jsonl` | stage/mode/backend 事件 |
| `VISTA/runs/run_*/ipc.jsonl` | `vision_req` / `vision_obs` |
| `VISTA/runs/run_*/events.jsonl` | 结构化事件 |

## 验收

```bash
cd /home/aidlux/embedded_com
STACK_PROFILE=dryrun VISTA_PREVIEW_RGB=1 ./start_robot_stack.sh
```

发送 `fetch_object` 后检查：

- `VISTA/logs/vision.out` 有 `mode switched`。
- 桌边阶段输出 `table_edge_obs`。
- 找目标阶段输出 `target_obs`。
- `orchestrator/runs/run_*/ipc.jsonl` 能看到 VISTA 回传的 `vision_obs`。
