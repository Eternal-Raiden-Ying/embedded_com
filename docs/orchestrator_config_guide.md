# A1 Orchestrator 配置修改指南

## 结论

A1 上运行 `python3 -m orchestrator_service.app.main` 时，最终使用的是：

```text
orchestrator/orchestrator_service/config/board_config.py 生成的 CONFIG
```

但日常不要直接改 `board_config.py`。它是配置装配代码，不是现场调参文件。

Orchestrator 日常调参优先级：

1. 持久调参：修改 `orchestrator/configs/stage_params.yaml` 或 `orchestrator/configs/car_cmd_params.yaml`。
2. 临时测试：在启动命令前加 `ORCH_*` 环境变量。
3. 新增配置项或改兜底默认值：修改 `orchestrator/orchestrator_service/config/schema.py`，并在 `board_config.py` 里接环境变量。

不要用仓库根目录的 `configs/mobile_gateway.*` 调 orchestrator 运动参数。那些是手机/MQTT gateway 相关配置，不控制桌边停靠状态机。

## 配置生效顺序

启动入口：

```text
orchestrator/orchestrator_service/app/main.py
  -> from orchestrator_service.config.board_config import CONFIG
  -> run_orchestrator_service(CONFIG)
```

`CONFIG` 的装配顺序是：

1. `schema.py` 里的 dataclass 默认值。
2. 读取 `ORCH_PROJECT_ROOT`、日志路径等基础环境变量。
3. 读取 `ORCH_CAR_CMD_PARAMS_FILE`，默认是 `orchestrator/configs/car_cmd_params.yaml`。
4. 读取 `ORCH_STAGE_PARAMS_FILE`，默认是 `orchestrator/configs/stage_params.yaml`。
5. 再读取大量 `ORCH_*` 环境变量，环境变量最后覆盖 YAML。

因此，同一个参数如果同时在 YAML 和命令行环境变量里出现，最终以环境变量为准。

## 常见场景应该改哪里

### 明天现场切换测试模式

如果你不想记环境变量，优先改：

```text
orchestrator/configs/stage_params.yaml
```

但下面这些“只跑一次”的开关还没有进入 YAML，临时测试仍然用环境变量最直接：

```bash
ORCH_SERIAL_DRY_RUN=1 \
ORCH_TABLE_EDGE_ONLY_TEST=1 \
ORCH_TABLE_TARGET_DIST_CM=30 \
ORCH_TABLE_STOP_MARGIN_CM=5 \
python3 -m orchestrator_service.app.main
```

适合只临时打开：

- dry-run / real serial 切换。
- table edge-only 临时打开。
- 临时调 `target_dist`、`stop_margin`、`settle`。
- 临时打开 `ORCH_ASSUME_GRASP_SUCCESS_FOR_TEST=1` 测返航骨架。

### 桌边停靠、沿边、目标锁定等阶段参数

改：

```text
orchestrator/configs/stage_params.yaml
```

这里放的是现场最常调的状态机参数：

- `final_lock.target_dist_m`：桌边最终目标距离，单位 m。
- `final_lock.stop_margin_m`：提前 STOP 裕量，单位 m。
- `final_lock.dist_abs_th_m`：最终距离容差，单位 m。
- `final_lock.yaw_abs_th`：最终 yaw 容差，单位 rad。
- `final_lock.settle_ms`：STOP 后 settle 时间，单位 ms。
- `final_lock.stable_frames`：确认完成需要的稳定帧数。
- `final_lock.max_micro_adjust`：最大微调次数。
- `edge_slide_search.*`：沿桌边搜索速度、纠偏、丢边恢复。
- `edge_follow.*`：边缘观测新鲜度、置信度、identity mismatch。
- `target_confirm.*` / `target_locked.*`：目标确认和锁定策略。

### 底盘命令发送周期和速度上限

改：

```text
orchestrator/configs/car_cmd_params.yaml
```

常见字段：

- `send_period_ms`：底盘命令发送周期。
- `hold_ms`：底盘命令 hold 时间。
- `max_vx_norm`、`max_vy_norm`、`max_wz_norm`：归一化速度上限。
- `stop_on_state_enter`：进状态时是否强制 STOP。

### 串口、日志、IPC 端口

用环境变量更合适，避免提交机器相关路径。

常用：

- `ORCH_SERIAL_PORT=/dev/ttyHS1`
- `ORCH_SERIAL_BAUDRATE=115200`
- `ORCH_SERIAL_DRY_RUN=0`
- `ORCH_LOG_DIR=...`
- `ORCH_TASK_CMD_IN_PORT=...`
- `ORCH_VISION_OBS_IN_PORT=...`

## VISTA 配置应该改哪里

VISTA 启动入口 `VISTA/vision_module/app/app.py` 最终使用的是：

```text
VISTA/vision_module/config/board_config.py 生成的 CONFIG
```

但现场不要直接改 `VISTA/vision_module/config/board_config.py`。现在 VISTA 也和 orchestrator 一样，持久调参改 YAML：

```text
VISTA/configs/vision_params.yaml
```

`board_config.py` / `schema.py` 只负责默认值、结构和装配。环境变量仍然可以临时覆盖 YAML，但你明天不需要记环境变量名。

### VISTA 摄像头、模型、预览、IPC

改：

```text
VISTA/configs/vision_params.yaml
```

对应位置：

- `camera.rgb`：RGB 摄像头编号、输入分辨率、裁剪区域、格式、FPS。
- `camera.depth`：深度摄像头编号、分辨率、FPS。
- `model.active_model`：默认模型。
- `model.profiles.*`：模型路径、输入尺寸、置信度、IOU、类别数。
- `debug.preview`、`debug.draw_boxes`、`debug.draw_masks`：预览和绘制开关。
- `ipc.req_in` / `ipc.obs_out`：VISTA 和 orchestrator 之间的 IPC 地址与端口。

### VISTA 不同模式下用哪些相机和 crop

改：

```text
VISTA/configs/vision_params.yaml
```

看 `mode_profiles` 段。这里决定每个模式的运行计划：

- `TRACK_LOCAL`：本地 YOLO 跟踪时用的 RGB/depth、crop、预览布局。
- `TABLE_EDGE_PERCEPTION`：桌边深度检测时启用哪些相机和 table-edge capability。
- `MICRO_ADJUST`：微调阶段 RGB crop 和 FPS。
- `GRASP_REMOTE`：抓取阶段 RGB/depth、远程抓取配置。
- `DEPTH_PERCEPTION`：深度感知模式，是否附带 RGB/table bbox。

如果明天发现桌子总是在画面偏左、偏右、偏下，优先改 `mode_profiles.modes.TRACK_LOCAL.rgb` 或 `mode_profiles.modes.TABLE_EDGE_PERCEPTION.rgb` 里的 `crop_x/crop_y/crop_w/crop_h`；如果只是改相机设备号，改 `camera.rgb.source` 或 `camera.depth.source`。

### 桌边 ROI、EDGE_DBG、TABLE_DET

持久调试改：

```text
VISTA/configs/vision_params.yaml
```

字段：

- `table_edge.roi_preset`：桌边 ROI preset，可填 `""`、`center_mid`、`center_lower`、`full_width_lower`。
- `table_edge.static_roi_enabled`：是否强制使用 detector 静态 ROI。
- `table_edge.update_hz`：桌边检测主循环频率。
- `table_edge.track_local_update_hz`：TRACK_LOCAL 下的桌边更新频率。
- `table_edge.track_local_light_edge`：TRACK_LOCAL 下是否用轻量处理。
- `table_edge.track_local_edge_stride`：轻量处理 stride。
- `debug.edge_debug_enabled`：是否输出 `[EDGE_DBG]`。
- `debug.edge_debug_period_s`：`[EDGE_DBG]` 输出间隔。
- `debug.table_det_enabled`：是否输出 `[TABLE_DET]`。
- `debug.table_det_min_conf`：table detection debug 最低置信度。
- `debug.table_det_center_tol`：center/left/right 判定容差。

ROI preset 的具体矩形定义在：

```text
VISTA/vision_module/backend/table_edge_roi.py
```

如果你只是想切换 preset，改 `VISTA/configs/vision_params.yaml`；只有要新增一种 preset 或调整 preset 的几何范围，才改 `table_edge_roi.py`。

### 不建议现场直接改的 VISTA 文件

- `VISTA/vision_module/config/schema.py`：只定义配置结构和兜底默认值，不做现场调参。
- `VISTA/vision_module/config/board_config.py`：只做配置装配，不做现场调参。
- `VISTA/vision_module/config/mode_defaults.py`：只做模式默认生成和 YAML 覆盖逻辑，不做现场调参。
- `VISTA/vision_module/config/data.py`：类别名和 COCO 映射，不是运行参数。
- `VISTA/vision_module/model/...`：模型文件和模型工程，不是现场模式配置。

### 新增一个配置项

代码层面需要三处：

1. 在 `orchestrator/orchestrator_service/config/schema.py` 的 dataclass 里加默认值。
2. 在 `orchestrator/orchestrator_service/config/board_config.py` 里从 YAML 或环境变量读取。
3. 在使用配置的 runtime/control 代码里读取 `cfg.xxx`。

如果这个参数属于现场调参，还要把它加入 `orchestrator/configs/stage_params.yaml` 或 `car_cmd_params.yaml`。

## 当前建议

- 明天 A1 现场调桌边停靠：优先改 `orchestrator/configs/stage_params.yaml`。
- 明天 A1 现场调 VISTA 视觉：优先改 `VISTA/configs/vision_params.yaml`。
- 临时试不同模式：可以用环境变量；如果你不想记变量名，就把常用值写进上面的文件。
- 不要直接改 `orchestrator/orchestrator_service/config/board_config.py`，除非是在新增配置项或修配置加载逻辑。
- 不要直接改 `orchestrator/orchestrator_service/config/schema.py` 做现场调参；它只是兜底默认值和结构定义。
