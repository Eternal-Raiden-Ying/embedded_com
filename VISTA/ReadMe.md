# VISTA

VISTA 是板端视觉服务，接收 Orchestrator 的 `vision_req`，按 stage/mode 调度摄像头、模型、桌边感知、远程抓取能力，并输出 `vision_obs`。

参考文档：

- `ARCHITECTURE.md`：内部拓扑
- `INTERFACES.md`：外部 IPC contract
- `IMPLEMENTATION_STATUS.md`：当前实现状态
- `NEXT_TODO.md`：后续事项

## 主链路

```text
Orchestrator --vision_req:9003--> VISTA
Orchestrator <--vision_obs:9002--- VISTA
```

当前整车链路见 [docs/system_runbook.md](/home/aidlux/embedded_com/docs/system_runbook.md)。

## Stage 与 Mode

当前注册 stage：

| Stage | 默认 mode | 说明 |
|------|-----------|------|
| `SEARCH` | `TRACK_LOCAL` | 本地目标搜索；桌边任务会按 Orchestrator 请求切到深度 mode |
| `GRASP` | `MICRO_ADJUST` | 抓取前微调与远程抓取协作 |
| `RETURN` | `TRACK_LOCAL` | 返航 tag / 返回目标观测 |
| `IDLE` | `IDLE` | 空闲 |

当前主链路 mode：

| mode | 摄像头/模型 | 主要输出 | 说明 |
|------|-------------|----------|------|
| `TRACK_LOCAL` | RGB + depth + 本地模型 | `target_obs` + `table_edge_obs`，返航时输出 home tag 观测 | 桌边目标搜索、返航；轻量 depth 用于 table edge 提升抓取稳定性 |
| `DEPTH_PERCEPTION` | depth；可选 RGB 桌面框模型 | `table_edge_obs` | 搜桌边、粗对齐、接近、最终锁边 |
| `TABLE_EDGE_PERCEPTION` | RGB + depth + 本地模型 | `table_edge_obs` + `target_obs` | 同时保持桌边与目标观测 |
| `MICRO_ADJUST` | RGB + 本地模型 | micro-adjust proposal | 抓取前微调 |
| `GRASP_REMOTE` | RGB + depth + remote client | `remote_result` | 远程抓取预测 |
| `IDLE_HOT` | RGB | runtime_status / preview | 热待机 |

Orchestrator 在 `SEARCH_TABLE`, `COARSE_ALIGN`, `CONTROLLED_APPROACH`, `FINAL_LOCK`, `REACQUIRE_EDGE` 请求 `DEPTH_PERCEPTION`；在 `AT_TABLE_EDGE`, `SEARCH_TARGET_INIT`, `EDGE_SLIDE_SEARCH`, `TARGET_CONFIRM`, `TARGET_LOCKED`, `FREEZE_BASE` 请求 `TRACK_LOCAL`。

## Runtime Baseline

```text
VistaApp
  -> StageController
  -> ModeController
  -> VisionEngine / RuntimeSupervisor
  -> CameraManager / PredictorManager / RemoteManager / PreviewManager
  -> Scheduler
```

要点：

- `VISTA_BACKEND=mock|real|auto` 是 camera/predictor backend 选择入口。
- 默认 RGB contract 为 BGR，`1280x720 -> 640x640` 裁剪。
- `PredictorManager` 发布 `local_perception.v1`，`infer_boxes` 格式为 `[[x1, y1, x2, y2, score, class_id], ...]`。
- `local_perception.class_names` 是 stage 侧类别解码来源；缺省 detect profile 会显式 fallback 到 `coco80`。
- remote grasp 以显式 `class_id` 为准，`GRASP_REMOTE` 上传编码由 mode/profile/runtime 配置控制。

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
| `VISION_REMOTE_BASE_URL` | 空 | 远程抓取服务地址 |
| `VISION_PREVIEW` | 板端默认 `1` | OpenCV 预览 |

## 日志

| 路径 | 内容 |
|------|------|
| `VISTA/logs/vision.out` | 启动脚本 stdout/stderr |
| `VISTA/runs/<stack_run_id>/event.jsonl` | 结构化事件 |
| `VISTA/runs/<stack_run_id>/ipc.jsonl` | `vision_req` / `vision_obs` |
| `VISTA/runs/<stack_run_id>/heartbeat.jsonl` | 心跳事件（`VISION_HEARTBEAT_ENABLED=1` 时写入） |
| `VISTA/runs/<stack_run_id>/meta.json` | 运行元数据 |

注：`stack_run_id` 格式为 `run_20260505_120000_abc123`，在 VISTA 启动时创建。旧名 `timeline.jsonl` 和 `events.jsonl`（复数）已废弃不再使用。

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
