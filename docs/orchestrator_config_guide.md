# A1 Orchestrator 配置修改指南

## 结论

A1 上运行 `python3 -m orchestrator_service.app.main` 时，最终使用的是：

```text
orchestrator/orchestrator_service/config/board_config.py 生成的 CONFIG
```

但日常不要直接改 `board_config.py`。它是配置装配代码，不是现场调参文件。

日常调参优先级：

1. 临时测试：在启动命令前加 `ORCH_*` 环境变量。
2. 持久调参：修改 `orchestrator/configs/stage_params.yaml` 或 `orchestrator/configs/car_cmd_params.yaml`。
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

### 明天现场临时切换模式

用环境变量，不改文件。

```bash
ORCH_SERIAL_DRY_RUN=1 \
ORCH_TABLE_EDGE_ONLY_TEST=1 \
ORCH_TABLE_TARGET_DIST_CM=30 \
ORCH_TABLE_STOP_MARGIN_CM=5 \
python3 -m orchestrator_service.app.main
```

适合：

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

### 新增一个配置项

代码层面需要三处：

1. 在 `orchestrator/orchestrator_service/config/schema.py` 的 dataclass 里加默认值。
2. 在 `orchestrator/orchestrator_service/config/board_config.py` 里从 YAML 或环境变量读取。
3. 在使用配置的 runtime/control 代码里读取 `cfg.xxx`。

如果这个参数属于现场调参，还要把它加入 `orchestrator/configs/stage_params.yaml` 或 `car_cmd_params.yaml`。

## 当前建议

- 明天 A1 现场调桌边停靠：优先改 `orchestrator/configs/stage_params.yaml`。
- 临时试不同模式：用 `ORCH_*` 环境变量。
- 不要直接改 `orchestrator/orchestrator_service/config/board_config.py`，除非是在新增配置项或修配置加载逻辑。
- 不要直接改 `orchestrator/orchestrator_service/config/schema.py` 做现场调参；它只是兜底默认值和结构定义。
