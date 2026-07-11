# Vision Docking Runtime Baseline v3.0.0

## Baseline Tag

`vision-docking-v3.0.0-baseline`

## Validation Run

`run_20260711_010959_b1d4bc`

## Scope

本版本冻结为 Vision/Docking Runtime Baseline，主要覆盖：

- Vision INIT、SILENT 和 FIND_EDGE 模式切换；
- `remote_init_auto_enabled=false` 配置；
- 启动阶段自动 Remote Init 关闭；
- Remote capability disabled 的正常 no-op 语义；
- FIND_EDGE 本地视觉链路；
- Table Edge 观测；
- Light Preview；
- Fresh/Reused Observation 统计；
- Canonical Critical Path；
- Dry-run 控制链路。

本版本不等同于完整机器人抓取系统最终发布版本。

## Verified Results

### Remote Lifecycle

- `remote_init_auto_enabled=false`
- 自动 `/init` 请求数：0
- 自动 Remote Init timeout 数：0
- `MODE_APPLY_FAILED`：0
- `enter_mode_apply_failed`：0
- `route_missing`：0
- `PUBLISH_REJECT`：0
- INIT 能够快速切换到 SILENT/IDLE
- FIND_EDGE 能够从 SILENT 正常进入

### Local Vision Runtime

- FIND_EDGE 模式应用成功
- RealSense RGB-D pipeline 正常启动
- QNN YOLO 模型正常加载
- Table Edge pipeline 正常运行
- Light Preview 正常运行
- Vision observation 能够发送至 Orchestrator

### Observation Statistics

- Fresh Observation 与 Reused Observation 已分开统计
- `vision_obs_age_ms` 与
  `vision_obs_age_at_first_consume_ms` 使用同一 Fresh 数据源
- Reused age 不再污染 Fresh age
- Reused observation 不再产生新的 canonical Critical Path

### Canonical Critical Path

Validation run result:

- complete count: 48
- p50: 217.27 ms
- p90: 303.87 ms
- p95: 344.36 ms
- max: 443.68 ms

以下输出使用同一 canonical 数据源：

- `critical_path.complete_latency_ms`
- `per_stage_duration_ms.critical_path_latency_ms`
- `top_slow_stages` 中的 Critical Path

同一 observation 后续 reused tick 不再覆盖 first endpoint。

## Frozen Components

在后续分支开发中，除非发现明确回归，不应直接修改：

- Table Edge ROI geometry
- FOV-aware bounded ROI
- Depth-margin extension
- Fallback lower-band ROI
- Adaptive sampling
- Plane Fitting 核心算法
- Front-edge vectorized fallback
- Light Preview depth-colormap + ROI 路径
- Fresh/Reused Observation 判定
- Canonical Critical Path first-endpoint 规则
- Remote auto/explicit init 分类
- Remote disabled successful no-op 语义

## Known Limitations

以下路径尚未被本次实机运行完整覆盖：

- `remote_init_auto_enabled=true`
- `GRASP_REMOTE_INIT` 显式远端初始化实机请求
- 完整 `YOLO_APPROACH`
- `FINAL_SLOW_STOP`
- `AT_TABLE_EDGE`
- `EDGE_SLIDE_SEARCH`
- 云端抓取与机械臂执行
- 完整 UART 非 dry-run 控制链路

因此，本标签定义为 Vision/Docking Runtime Baseline，而不是完整抓取系统最终发布版本。

## Tag Policy

该标签创建后不得移动、覆盖或强制更新。

后续修复应创建新的版本标签，例如：

- `vision-docking-v3.0.1-baseline`
- `vision-docking-v3.1.0-rc1`
